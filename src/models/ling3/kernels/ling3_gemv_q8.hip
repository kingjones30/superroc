// LS1 2026-10-01 — the decode-step GEMV as an integer dot against a Q8_1
// activation.
//
// Why this is faster than k_gemv_wide (which stays for the non-FP4 types):
// per 32 weights the float path issued 32 separate byte loads (the ROCmFP4 block
// is 17 bytes, so no wider load is legal at the block level) plus a ue4m3 decode
// and an FMA for each one. This path issues FOUR 32-bit loads of the block's 16
// nibble bytes, expands them to int8 through the fork's perm codebook, and does
// EIGHT sdot4. That is roughly a third of the instructions for the same bytes,
// and the FP4 scale is applied once per 32 instead of once per element.
//
// Numerics: the activation is quantised to int8 with ggml's quantize_row_q8_1
// recipe (max|x| over each 32, scale = max/127, round(x/scale)), and the FP4 code
// table is exactly ling3_rocmfp4_code (0,1,2,3,4,6,8,10 / -...), unscaled. This is
// the fork's / ggml's computation, not superroc's float one.
//
// There is deliberately NO cache of "which vector is currently quantised". A
// pointer-keyed one would be wrong: rocm_backend.cpp reuses st->normed for every
// layer, so the second layer's rms_norm overwrites the bytes behind the pointer
// the first layer quantised, and a pointer match would then mean the wrong data.
// Every entry point here either quantises the vector it was handed or is handed
// a buffer that the caller filled for that exact call. Slots exist only so the
// expert FFN can reuse one quantisation across several matvecs it owns itself.

#include "ling3_quant.hpp"
#include "src/core/hip_launch_check.hpp"

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int ls1_chk(const char * what) {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        fprintf(stderr, "ling3 q8 %s: %s\n", what, hipGetErrorString(e));
        return LING3_ERR_HIP;
    }
    return LING3_OK;
}

// ---------------------------------------------------------------------------
// FP4 nibble -> int8 codebook.
//
// Verbatim from the fork, ggml/rocmfp4/rocmfp4_hip_codebook.cuh
// (rocmfp4_get_int_from_codebook_16, its GGML_USE_HIP branch). Four nibbles go
// in; int2 of eight int8 comes out: .x is the LOW nibbles of bytes 0..3, which
// are elements 0..3 of the block, .y is the HIGH nibbles, elements 16..19. The
// table [0,1,2,3],[4,6,8,10],[0,-1,-2,-3],[-4,-6,-8,-10] is exactly
// ling3_rocmfp4_code(nib), and tools/ling3_gemv_q8_test.cpp checks all 65536
// four-nibble words against that predicate on the device.
//
// Two things about the expression that look wrong and are not, both confirmed on
// the GPU (a perm probe and the exhaustive selftest):
//   * the perm's LOW selector is its SECOND argument and its HIGH selector is its
//     FIRST (indices 0..3 from arg2, 4..7 from arg1), which is why the fork passes
//     (values1, values0) and not the other way round;
//   * `q_odd = q4 >> 4` shifts across byte boundaries, but both uses mask with
//     0x07070707 / 0x08080808, which land entirely inside the source byte, so the
//     nibbles it reads are still the high ones. v_odd_low / v_odd_high below are
//     computed and never used -- that is the fork's code, kept so the two stay
//     line-for-line comparable.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int2 ls1_fp4_codebook16(int q4) {
    constexpr uint32_t values0 = 0x03020100u;
    constexpr uint32_t values1 = 0x0a080604u;
    constexpr uint32_t values2 = 0xfdfeff00u;
    constexpr uint32_t values3 = 0xf6f8fafcu;

    const uint32_t q_even = (uint32_t) q4;
    const uint32_t q_odd  = (uint32_t) q4 >> 4;

    const uint32_t v_even_low  = __builtin_amdgcn_perm(values1, values0, q_even & 0x07070707u);
    const uint32_t v_odd_low   = __builtin_amdgcn_perm(values1, values0, q_odd  & 0x07070707u);
    const uint32_t v_even_high = __builtin_amdgcn_perm(values3, values2, q_even & 0x07070707u);
    const uint32_t v_odd_high  = __builtin_amdgcn_perm(values3, values2, q_odd  & 0x07070707u);

    const uint32_t mask_even = 0x03020100u | ((q_even & 0x08080808u) >> 1);
    const uint32_t mask_odd  = 0x03020100u | ((q_odd  & 0x08080808u) >> 1);

    return make_int2(
        __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even),
        __builtin_amdgcn_perm(v_odd_high,  v_odd_low,  mask_odd));
}

// Read the 4 nibble bytes at p as a little-endian 32-bit word.
//
// The fork does this with one direct `*(const int *)p` (its
// GGML_ROCMFP4_UNALIGNED_QS_DWORD_LOAD, on for HIP), which depends on the
// hardware accepting an unaligned 32-bit global load. The first revision of this
// file did exactly that and produced a GEMV wrong against the CPU reference by
// factors of 3 to 20 on every shape, with no fault and nothing in the log to
// point at it -- the compiler will not tell you whether it honoured the
// alignment, and the address is only 4-aligned for one block in four. Four byte
// loads instead, assembled explicitly; the compiler is free to fold them back
// into whatever it likes, and the byte order no longer depends on the pointer.
__device__ __forceinline__ int ls1_ld32(const uint8_t * p) {
    return (int) ((uint32_t) p[0] | ((uint32_t) p[1] << 8) |
                  ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24));
}

// One weight block, decoded once: the two packed int8 halves and the two FP4 scales.
//
// The codebook expansion (four v_perm) is the expensive half of a block, and in the
// small-m matvec below it is done ONCE per block and reused for every activation column.
// That sharing is the whole point of the m-column form: at m = 3 the weight bytes and
// the perms are paid for once and only the eight sudot4 per column are extra.
// FOUR dwords, not one: the block's 16 nibble bytes are four independent groups of four
// elements, so each dword carries its own codes. An earlier version of this struct held a
// single vx/vy expanded from bytes 0..3 and reused it for all four dwords, which silently
// scaled every row by ~4 on three quarters of the elements -- the phase-1 GEMV gate caught
// it immediately, and that is the only reason this comment exists.
struct ls1_blk32 {
    int vx[4], vy[4];    // per dword: codes of its 4 low-nibble and 4 high-nibble elements
    float slo, shi;
};

// Reads the block's 16 nibble bytes and its 1 or 2 scale bytes -- offsets 0..17 of a 17
// or 18 byte block, and nothing else, so a row touches exactly row_bytes bytes and can
// never read past the last row of a tightly sized tensor.
__device__ __forceinline__ ls1_blk32 ls1_load_blk32(int qtype, const uint8_t * blk) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    ls1_blk32 c;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int2 v = ls1_fp4_codebook16(ls1_ld32(blk + 4 * l));
        c.vx[l] = v.x;
        c.vy[l] = v.y;
    }
    c.slo = ling3_ue4m3_half(blk[16]);
    c.shi = (qtype == LING3_Q4_0_ROCMFP4) ? ling3_ue4m3_half(blk[17]) : c.slo;
    return c;
}

// The decoded block dotted with one activation block. The codebook is unscaled, so the
// only factor left is the activation's own.
__device__ __forceinline__ float ls1_dot_blk32(const ls1_blk32 & c, const int8_t * xq32,
                                               float xd) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int * q8 = (const int *) xq32;
    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        // sudot4, not sdot4: this ROCm's clang gates sdot4 behind a target feature the
        // project's flags do not set, and the fork uses the same call (ggml_cuda_dp4a,
        // ggml/src/ggml-cuda/common.cuh:720, RDNA3/RDA4 branch). Verified on the device as
        // a bilinear form -- lane i pairs with lane i, both operands signed:
        // sudot4(true, 0x01020304, true, 0x04030201, 0, false) == 20 == 4 + 6 + 6 + 4.
        // .x against the first 16 elements of the block, .y against the last 16 --
        // which is why the second dot takes v.y and not v.x. Getting that wrong is
        // silent: it scales every row by roughly 4 and faults nothing.
        sumi0 = (int) __builtin_amdgcn_sudot4(true, c.vx[l], true, q8[l + 0], sumi0, false);
        sumi1 = (int) __builtin_amdgcn_sudot4(true, c.vy[l], true, q8[l + 4], sumi1, false);
    }
    return xd * (c.slo * (float) sumi0 + c.shi * (float) sumi1);
}

// One weight block (32 values, tsz bytes) dotted with one activation block: decode and
// dot in one call. This is the m = 1 path, and the small-m kernel below is the same maths
// with the decode hoisted out of the column loop.
__device__ __forceinline__ float ls1_dot_fp4_blk32_q8(int qtype, const uint8_t * blk,
                                                      const int8_t * xq32, float xd) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int * q8 = (const int *) xq32;
    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int2 v = ls1_fp4_codebook16(ls1_ld32(blk + 4 * l));
        // sudot4, not sdot4: this ROCm's clang gates sdot4 behind a target feature the
        // project's flags do not set, and the fork uses the same call (ggml_cuda_dp4a,
        // ggml/src/ggml-cuda/common.cuh:720, RDNA3/RDA4 branch). Verified on the device as
        // a bilinear form -- lane i pairs with lane i, both operands signed:
        // sudot4(true, 0x01020304, true, 0x04030201, 0, false) == 20 == 4 + 6 + 6 + 4.
        // .x against the first 16 elements of the block, .y against the last 16 --
        // which is why the second dot takes v.y and not v.x. Getting that wrong is
        // silent: it scales every row by roughly 4 and faults nothing.
        sumi0 = (int) __builtin_amdgcn_sudot4(true, v.x, true, q8[l + 0], sumi0, false);
        sumi1 = (int) __builtin_amdgcn_sudot4(true, v.y, true, q8[l + 4], sumi1, false);
    }
    const float slo = ling3_ue4m3_half(blk[16]);
    const float shi = (qtype == LING3_Q4_0_ROCMFP4) ? ling3_ue4m3_half(blk[17]) : slo;
    // The codebook is unscaled, so the only factor left is the activation's own.
    return xd * (slo * (float) sumi0 + shi * (float) sumi1);
}

// ---------------------------------------------------------------------------
// Q8_1 activation buffers, one pair per slot.
//
//   qs[0 .. k)     int8, element i at qs[i]
//   ds[0 .. k/32)  float, the scale of qs[32*b .. 32*b+32)
//
// The ROCmFP4 weight block is also 32 values, so weight block b pairs with ds[b].
// ---------------------------------------------------------------------------
enum {
    LS1_SLOT_GENERIC = 0,   // ling3_gemv_fp4()'s own scratch
    LS1_SLOT_FFN_X  = 1,    // the expert FFN's input, shared by gate and up
    LS1_SLOT_FFN_H  = 2,    // the expert FFN's SwiGLU output, input to down
    LS1_SLOT_MOE_H  = 3,    // the grouped expert FFN's SwiGLU output, input to down
    LS1_SLOT_ATTN   = 4,    // st->normed, once per layer, shared by its projections
    LS1_NSLOT      = 5
};

namespace {

int8_t * g_qs[LS1_NSLOT] = {nullptr, nullptr, nullptr};
float * g_ds[LS1_NSLOT] = {nullptr, nullptr, nullptr};
int g_cap[LS1_NSLOT] = {0, 0, 0};

// ggml's quantize_row_q8_1, one THREAD per 32 values.
//
// Deliberately the plainest possible transcription of the reference recipe:
// serial loop, no warp shuffle, no cross-lane anything. A revision that reduced
// the maximum with __shfl_down produced the right ds but codes that disagreed
// with the CPU reference on roughly a quarter of the elements, so the bug lived
// somewhere in the shuffle that a straight transcription cannot have. This
// kernel runs at most 9 times per decoded token over at most 10 KB, so the
// coalescing it gives up is worth far less than being wrong.
//
// The signed maximum (not the magnitude) sets the scale, exactly as ggml does,
// so a block that is entirely negative gets a negative scale and its codes come
// out positive. Which of two equal-magnitude elements wins the tie does not
// matter: the two choices give scales that differ by a sign and codes that
// differ by that same sign, so the product ds * qs is the same.
__global__ void k_q8_1_quant(const float * x, int8_t * qs, float * ds, int nb) {
    // LA1: one WARP (32 lanes, one per value) per 32-value block, eight warps per
    // CTA, instead of one thread scanning the block serially. The old layout ran
    // one thread per block, so a k=2560 row quantised with only 80 active threads.
    // Same recipe and same arithmetic, bit-identical:
    //   * the scale is the signed value at the LOWEST index attaining the max |x|
    //     (the serial scan took the first max it saw; a warp butterfly resolves
    //     the tie to the same lowest index, and (amp, idx) is a total order, so
    //     the reduction order cannot change the outcome),
    //   * each lane j computes qb[j] = roundf(xb[j] * id) for its own element --
    //     the identical expression the serial loop used,
    //   * an all-zero block still gets d = 0, id = 0, codes all 0.
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int b = (int) blockIdx.x * 8 + warp;
    if (b >= nb) {
        return;
    }
    const float * xb = x + (size_t) b * 32;

    const float v = xb[lane];
    float ma = fabsf(v), mv = v;
    int   mi = lane;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float oa = __shfl_down(ma, off);
        const float ov = __shfl_down(mv, off);
        const int   oi = __shfl_down(mi, off);
        if (oa > ma || (oa == ma && oi < mi)) {
            ma = oa; mv = ov; mi = oi;
        }
    }
    const float d  = mv / 127.f;
    const float id = d != 0.f ? 1.0f / d : 0.f;
    if (lane == 0) {
        ds[b] = d;
    }
    const float id_b = __shfl(id, 0);
    int8_t * qb = qs + (size_t) b * 32;
    qb[lane] = (int8_t) roundf(xb[lane] * id_b);
}

}  // namespace

// Quantise x[0..k) into slot `slot`. Returns 0 on success.
extern "C" int ling3_quant_q8(const float * x, int k, int slot, hipStream_t stream) {
    if (x == nullptr || k <= 0 || (k % 32) != 0 || slot < 0 || slot >= LS1_NSLOT) {
        return LING3_ERR_ARG;
    }
    if (k > g_cap[slot] || g_qs[slot] == nullptr || g_ds[slot] == nullptr) {
        if (g_qs[slot]) { hipFree(g_qs[slot]); g_qs[slot] = nullptr; }
        if (g_ds[slot]) { hipFree(g_ds[slot]); g_ds[slot] = nullptr; }
        g_cap[slot] = 0;
        // +LING3_TAIL_PAD so a widened tail load still has readable bytes behind
        // the last element, the same contract every other activation has here.
        if (hipMalloc(&g_qs[slot], (size_t) k + LING3_TAIL_PAD) != hipSuccess) return LING3_ERR_HIP;
        if (hipMalloc(&g_ds[slot], (size_t) (k / 32) * sizeof(float) + LING3_TAIL_PAD) != hipSuccess) {
            hipFree(g_qs[slot]); g_qs[slot] = nullptr; return LING3_ERR_HIP;
        }
        g_cap[slot] = k;
    }
    const int nb = k / 32;
    // LA1: the kernel runs one warp per 32-value block, eight warps per CTA, so
    // the grid is ceil(nb/8), not ceil(nb/(blockDim)). blockDim stays 256.
    const int threads = 256;
    const int blocks = (nb + 7) / 8;
    k_q8_1_quant<<<blocks, threads, 0, stream>>>(x, g_qs[slot], g_ds[slot], nb);
    if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ls1_chk("q8 quant");
}

// LKF: hand the caller the slot buffers the wo GEMV will read. Same allocation
// as ling3_quant_q8, no kernel. The fused KDA epilogue fills them.
extern "C" int ling3_q8_slot_device(int slot, int k, int8_t ** qs, float ** ds) {
    if (qs == nullptr || ds == nullptr || k <= 0 || (k % 32) != 0 ||
        slot < 0 || slot >= LS1_NSLOT) {
        return LING3_ERR_ARG;
    }
    if (k > g_cap[slot] || g_qs[slot] == nullptr || g_ds[slot] == nullptr) {
        if (g_qs[slot]) { hipFree(g_qs[slot]); g_qs[slot] = nullptr; }
        if (g_ds[slot]) { hipFree(g_ds[slot]); g_ds[slot] = nullptr; }
        g_cap[slot] = 0;
        if (hipMalloc(&g_qs[slot], (size_t) k + LING3_TAIL_PAD) != hipSuccess) return LING3_ERR_HIP;
        if (hipMalloc(&g_ds[slot], (size_t) (k / 32) * sizeof(float) + LING3_TAIL_PAD) != hipSuccess) {
            hipFree(g_qs[slot]); g_qs[slot] = nullptr; return LING3_ERR_HIP;
        }
        g_cap[slot] = k;
    }
    *qs = g_qs[slot];
    *ds = g_ds[slot];
    return LING3_OK;
}

// One row of the integer matvec, before the cross-lane reduction: thread `threadIdx.x` and
// its stride partners walk the row's 32-value blocks. Kept separate from the kernel so the
// grouped expert FFN can use the same dot with a different thread mapping -- and, in phase 3,
// so that mapping's reduction goes through the same ls1_block_sum rather than a 32-lane one.
__device__ __forceinline__ float ls1_dot_row_q8(int qtype, const uint8_t * w,
                                                const int8_t * xq, const float * xd,
                                                int k) {
    const int tsz = ling3_type_size(qtype);
    const int nblk = k / 32;
    float s = 0.f;
    for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
        s += ls1_dot_fp4_blk32_q8(qtype, w + (size_t) qb * (size_t) tsz,
                                  xq + (size_t) qb * 32, xd[qb]);
    }
    return s;
}

// ---------------------------------------------------------------------------
// The matvec itself.
//
// One 64-thread CTA per output row: 64x the parallelism of a thread per row, the
// same shape k_gemv_wide uses, so the only thing that changes is what happens
// inside the loop. Threads stride over the row's 32-value blocks, so the 64
// lanes of one step cover 64 consecutive blocks = 1088 contiguous bytes; each
// lane's four 32-bit loads land in those same lines, so nothing is fetched
// twice and nothing is fetched that is not used.
//
// The reduction is warp shuffles plus one LDS slot per warp, not the five
// __syncthreads of the LDS tree: with nblk small relative to the CTA the tree
// was costing more than the loop it was reducing.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float ls1_warp_sum(float s) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down(s, off);
    }
    return s;
}

// Sum over the whole CTA, in EVERY thread.
//
// ls1_warp_sum on its own is only a 32-lane reduction: __shfl_down with offsets 16..1
// brings lanes 0..31 together and thread 0 never sees lanes 32..63. That is fine inside
// k_gemv_wide_q8, which follows it with the LDS slots and a __syncthreads, and it is
// exactly what the two grouped MoE kernels were missing in phase 2 -- they are launched
// with 64-thread CTAs, so every row silently dropped half its work. At k = 2560 that is
// 80 blocks spread over all 64 lanes and the result was wrong by 3-6x with no fault;
// at k = 512 it is 16 blocks, all on lanes 0-15, and the bug is invisible. Hence this
// helper, and hence the test running n_used = 1 AND n_used = 8 at the SAME real shape.
//
// `red` needs blockDim.x/32 floats and is reusable across calls: the trailing
// __syncthreads() is what frees it. Both barriers are reached by every thread, so the
// caller must not return early around a call.
__device__ __forceinline__ float ls1_block_sum(float s, float * red) {
    const int nw = (int) blockDim.x >> 5;
    s = ls1_warp_sum(s);
    if ((int) (threadIdx.x & 31) == 0) {
        red[threadIdx.x >> 5] = s;
    }
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < nw; ++i) {
        t += red[i];
    }
    __syncthreads();
    return t;
}

// The phase-1 kernel, kept because it is what the unit test compares the warp-per-row
// one against: k_gemv_wide_q8_rows must be BIT-IDENTICAL to this.
__global__ void k_gemv_wide_q8_cta64(int qtype, const uint8_t * W, size_t row_bytes,
                               const int8_t * xq, const float * xd,
                               float * y, int k, int n) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int row = (int) blockIdx.x;
    if (row >= n) {
        return;
    }
    const uint8_t * w = W + (size_t) row * row_bytes;
    const int tsz = ling3_type_size(qtype);
    const int nblk = k / 32;
    __shared__ float red[32];

    float s = 0.f;
    for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
        s += ls1_dot_fp4_blk32_q8(qtype, w + (size_t) qb * (size_t) tsz,
                                  xq + (size_t) qb * 32, xd[qb]);
    }
    s = ls1_block_sum(s, red);
    if (threadIdx.x == 0) {
        y[row] = s;
    }
}

// RV25 wave1 F3: the same accumulation with a STORE that reproduces the unfused
// pair hipMemset(y, 0) ; k_axpy(1.0f, p, y) bit for bit. That pair computes
//     y[i] = (+0.0f) + (1.0f * s) = 0.0f + s
// and the "0.0f +" is load-bearing: for s == -0.0f it yields +0.0f, which is what
// the memset+axpy pair produced. A raw `y[row] = s` would store -0.0f and change
// the bytes. Nothing else differs -- same accumulation order, same
// #pragma clang fp contract(off), same ls1_block_sum reduction.
__global__ void k_gemv_wide_q8_cta64_zp(int qtype, const uint8_t * W, size_t row_bytes,
                               const int8_t * xq, const float * xd,
                               float * y, int k, int n) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int row = (int) blockIdx.x;
    if (row >= n) {
        return;
    }
    const uint8_t * w = W + (size_t) row * row_bytes;
    const int tsz = ling3_type_size(qtype);
    const int nblk = k / 32;
    __shared__ float red[32];

    float s = 0.f;
    for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
        s += ls1_dot_fp4_blk32_q8(qtype, w + (size_t) qb * (size_t) tsz,
                                  xq + (size_t) qb * 32, xd[qb]);
    }
    s = ls1_block_sum(s, red);
    if (threadIdx.x == 0) {
        y[row] = 0.0f + s;
    }
}

// ---------------------------------------------------------------------------
// PHASE6: one WARP per row, several consecutive rows per CTA.
//
// The measured shape of the two weak decode GEMVs (REPORT5 §3, from the build21
// profile) is a lane-utilisation problem, not a bandwidth one:
//
//   k_moe_down_weighted   n_ff = 768  -> 24 blocks of 32 values, spread over a 64-lane
//                                     CTA: 24 of 64 lanes load anything, 37.5 %.
//   k_gemv_wide_q8        k    = 2560 -> 80 blocks over 64 lanes: 80 of 128 lane-slots,
//                                     62.5 %, the tail wasted on a second iteration.
//
// One warp per row fixes both: 24/32 = 75 % and 80/96 = 83 % lane occupancy. It
// also drops the CTA-wide reduction entirely -- a warp reduction is a shuffle tree
// with no LDS and no __syncthreads -- and because a CTA's warps take CONSECUTIVE
// rows, its reads span 8 x 1360 = 10,880 contiguous bytes instead of 8 separate
// 1360-byte strided ones, which is what keeps the 64-byte lines fully used even
// though a 1360-byte row is not a multiple of 64 and so straddles lines.
//
// The row loop is a grid-stride, so the grid can be sized to the machine rather
// than to n. No CTA-wide barrier exists here, which is also why `row >= n` may
// return per warp without any risk of a hang -- the old CTA-per-row kernels had
// to keep that check uniform for the sake of their __syncthreads.
__global__ void k_gemv_wide_q8_rows(int qtype, const uint8_t * W, size_t row_bytes,
                                    const int8_t * xq, const float * xd,
                                    float * y, int k, int n, int grid_rows) {
    const int lane = threadIdx.x & 31;
    const int sub = (int) (threadIdx.x >> 5);
    const int nwarps = (int) blockDim.x >> 5;
    const int tsz = ling3_type_size(qtype);
    const int nblk = k / 32;
    const int stride = grid_rows * nwarps;
    for (int row = blockIdx.x * nwarps + sub; row < n; row += stride) {
        const uint8_t * w = W + (size_t) row * row_bytes;
        float s = 0.f;
        for (int qb = lane; qb < nblk; qb += 32) {
            s += ls1_dot_fp4_blk32_q8(qtype, w + (size_t) qb * (size_t) tsz,
                                      xq + (size_t) qb * 32, xd[qb]);
        }
        s = ls1_warp_sum(s);
        if (lane == 0) {
            y[row] = s;
        }
    }
}

// RV25 wave1 F3: k_gemv_wide_q8_rows with the same 0.0f + store as
// k_gemv_wide_q8_cta64_zp above. Only reached when LING3_GEMV_GEMV / LING3_GEMV_VARIANT
// selects the warp-per-row matvec; it must carry the same bit-identity.
__global__ void k_gemv_wide_q8_rows_zp(int qtype, const uint8_t * W, size_t row_bytes,
                                    const int8_t * xq, const float * xd,
                                    float * y, int k, int n, int grid_rows) {
    const int lane = threadIdx.x & 31;
    const int sub = (int) (threadIdx.x >> 5);
    const int nwarps = (int) blockDim.x >> 5;
    const int tsz = ling3_type_size(qtype);
    const int nblk = k / 32;
    const int stride = grid_rows * nwarps;
    for (int row = blockIdx.x * nwarps + sub; row < n; row += stride) {
        const uint8_t * w = W + (size_t) row * row_bytes;
        float s = 0.f;
        for (int qb = lane; qb < nblk; qb += 32) {
            s += ls1_dot_fp4_blk32_q8(qtype, w + (size_t) qb * (size_t) tsz,
                                      xq + (size_t) qb * 32, xd[qb]);
        }
        s = ls1_warp_sum(s);
        if (lane == 0) {
            y[row] = 0.0f + s;
        }
    }
}

// ---------------------------------------------------------------------------
// LVL SPEC1: the Q6_K LM head of the Ling-3.0-flash-VL artifact.
//
// The generic float matvec (ling3_moe.hip's k_gemv_wide_gs) gave one thread a
// whole row and decoded every element through ling3_elem. Two things cost: the
// 32 lanes of a warp then walked 32 rows 2100 bytes apart, so each weight load
// was its own cache line, and every element re-derived its block indices from a
// runtime k. Measured on halo with rocprofv3 over 128 decode steps it was
// 1967 ms = 15.4 ms per token, 40.9 % of all GPU time on the box.
//
// Shape here: one warp per output row, and inside a row one lane per 32-element
// piece of every 256-element block -- lane `l` owns local index p*32 + l for
// p = 0..7. That is how ggml lays Q6_K out: element (half, piece, l) sits at
// ql[half*64 + (piece&1)*32 + l], qh[half*32 + l] and scales[half*8 + (l>>4) +
// piece*2], with half = p>>2 and piece = p&3. Every index in the decode is then a
// function of (p, l) alone, so the block loop carries no division and the lanes
// of a warp read consecutive bytes of ql / qh / scales.
//
// Arithmetic is ling3_term_block's Q6_K form element for element and in the same
// association -- (d * sc) * q, then times x, one fp32 `+=` per element with fp
// contraction off. The only difference from ling3_dot_row, which this replaces
// for this one tensor, is the order the per-lane partial sums combine in.
// ---------------------------------------------------------------------------
#define LVL_Q6K_TSZ 210         // bytes per 256-element Q6_K block
// One CTA per 8 rows, no grid-stride: 19648 CTAs at n = 157184. The MoE kernel
// beside this one runs 262144 CTAs and reaches ~129 GB/s where a 4096-CTA
// grid-strided form of this kernel reached ~60, so the loop is not free.
#define LVL_Q6K_MAX_BLOCKS 65536

__global__ __launch_bounds__(256) void lvl_gemv_q6k_rows(
        const uint8_t * __restrict__ W, const float * __restrict__ x,
        float * __restrict__ y, int k, int n, int grid_rows) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int lane = threadIdx.x & 31;
    const int sub = (int) (threadIdx.x >> 5);
    const int nwarps = (int) (blockDim.x >> 5);
    const int nb = k >> 8;                          // 256-element blocks per row
    const size_t rbytes = (size_t) nb * (size_t) LVL_Q6K_TSZ;
    const int stride = grid_rows * nwarps;
    for (int row = blockIdx.x * nwarps + sub; row < n; row += stride) {
        const uint8_t * w = W + (size_t) row * rbytes;
        float s = 0.0f;
        for (int b = 0; b < nb; ++b) {
            const uint8_t * blk = w + (size_t) b * (size_t) LVL_Q6K_TSZ;
            const float d = ling3_fp16_to_fp32(ling3_ld16(blk + 208));
            const float * xb = x + (size_t) b * 256u;
            // LVL SPEC2 item 3. Per 256-element block this lane needs FOUR
            // distinct ql bytes ((piece&1) and half) and TWO distinct qh bytes
            // (half). Read them once here, as plain loads so clang can batch
            // and pipeline them, instead of re-reading the same address through
            // the volatile ling3_ld8 eight times. `sc` is NOT hoisted: its
            // index depends on piece. Arithmetic and the p = 0..7 accumulation
            // order are unchanged, so the result is bit-identical.
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const uint8_t * qlp = blk + (size_t) half * 64u + (size_t) lane;
                const uint8_t ql_lo = qlp[0];      // piece & 1 == 0
                const uint8_t ql_hi = qlp[32];     // piece & 1 == 1
                const int qhb =
                    (int) (blk + 128u + (size_t) half * 32u)[(size_t) lane];
                const float * xh = xb + half * 128;
#pragma unroll
                for (int piece = 0; piece < 4; ++piece) {
                    const uint8_t qb = (piece & 1) ? ql_hi : ql_lo;
                    const int nib =
                        (piece >= 2) ? (int) (qb >> 4) : (int) (qb & 0x0f);
                    const int q = (nib | (((qhb >> (piece * 2)) & 3) << 4)) - 32;
                    const int sc = (int) (int8_t) ling3_ld8(
                        blk + 192u + (size_t) half * 8u + (size_t) (lane >> 4) +
                        (size_t) (piece * 2));
                    s += ((d * (float) sc) * (float) q) * xh[piece * 32 + lane];
                }
            }
        }
        s = ls1_warp_sum(s);
        if (lane == 0) {
            y[row] = 0.0f + s;
        }
    }
}

extern "C" int ling3_gemv_q6k(int qtype, const void * W, const float * x, float * y,
                              int k, int n, hipStream_t stream) {
    if (qtype != LING3_Q6_K) {
        return LING3_ERR_TYPE;
    }
    if (W == nullptr || x == nullptr || y == nullptr || n <= 0 || (k & 255) != 0) {
        return LING3_ERR_ARG;
    }
    const int block = 256;
    const int nwarps = block >> 5;
    int blocks = (n + nwarps - 1) / nwarps;   // ls1_gemv_blocks is defined below
    if (blocks > LVL_Q6K_MAX_BLOCKS) {
        blocks = LVL_Q6K_MAX_BLOCKS;
    }
    lvl_gemv_q6k_rows<<<blocks, block, 0, stream>>>(
        (const uint8_t *) W, x, y, k, n, blocks);
    if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ls1_chk("gemv q6k");
}

// The same shape for the down projection: one warp per output row, the n_used
// slots looped in slot order with a warp reduction and no CTA barrier.
__global__ void k_moe_down_weighted_rows(int qtype, const uint8_t * down, size_t rb_ff,
                                         const int * ids, const float * weights,
                                         const int8_t * hq, const float * hd,
                                         float * y, int n_used, int n_expert,
                                         int k, int n_ff, int grid_rows, int * rc) {
    const int lane = threadIdx.x & 31;
    const int sub = (int) (threadIdx.x >> 5);
    const int nwarps = (int) blockDim.x >> 5;
    const int nb = n_ff / 32;
    const int stride = grid_rows * nwarps;
    for (int j = blockIdx.x * nwarps + sub; j < k; j += stride) {
        float acc = 0.f;
        for (int s = 0; s < n_used; ++s) {
            const int id = ids[s];
            const bool ok = (id >= 0 && id < n_expert);
            if (!ok && lane == 0) {
                *rc = LING3_ERR_ARG;
            }
            const size_t base = ok ? (((size_t) id * (size_t) k + (size_t) j) * rb_ff) : 0;
            const uint8_t * drow = down + base;
            float part = 0.f;
            for (int qb = lane; qb < nb; qb += 32) {
                part += ls1_dot_fp4_blk32_q8(qtype, drow + (size_t) qb * (size_t) ling3_type_size(qtype),
                                            hq + (size_t) s * (size_t) n_ff + (size_t) qb * 32,
                                            hd[(size_t) s * (size_t) nb + (size_t) qb]);
            }
            part = ls1_warp_sum(part);
            if (lane == 0) {
                acc += weights[s] * part;
            }
        }
        if (lane == 0) {
            y[j] = acc;
        }
    }
}

// The LM head: n = 157184, so one CTA per row would be 157k CTAs of 64 threads
// for a 1360-byte job each and the scheduling cost would dominate. A fixed grid
// loops over rows, the way k_gemv_wide_gs does.
__global__ void k_gemv_wide_q8_gs(int qtype, const uint8_t * W, size_t row_bytes,
                                  const int8_t * xq, const float * xd,
                                  float * y, int k, int n) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int tsz = ling3_type_size(qtype);
    const int nblk = k / 32;
    __shared__ float red[32];

    for (int row = (int) blockIdx.x; row < n; row += (int) gridDim.x) {
        const uint8_t * w = W + (size_t) row * row_bytes;
        float s = 0.f;
        for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
            s += ls1_dot_fp4_blk32_q8(qtype, w + (size_t) qb * (size_t) tsz,
                                      xq + (size_t) qb * 32, xd[qb]);
        }
        // ls1_block_sum's own barriers handle red's reuse between rows.
        s = ls1_block_sum(s, red);
        if (threadIdx.x == 0) {
            y[row] = s;
        }
    }
}

// Below this many output rows a CTA per row still beats a grid-strided loop;
// above it there is already plenty of parallelism. Same cut as ling3_gemv.
#define LS1_GEMV_WIDE_MAX 8192
#define LS1_GEMV_GS_BLOCKS 2048

// Matvec against an activation the caller already quantised into `slot`.
// ---------------------------------------------------------------------------
// PHASE6 selection. Variant 0 is the phase-1 CTA-per-row kernel, unchanged; variant 1
// is the warp-per-row kernel. LING3_GEMV_VARIANT picks it and DEFAULTS TO 0, so a
// build with no environment set behaves exactly as it did before this phase. The
// unit test runs both and requires them to agree bit-for-bit.
// ---------------------------------------------------------------------------
// LS1 PHASE7 item 4: the flag is SPLIT, because its two halves are not the same claim.
//
//   LING3_GEMV_DOWN  (default ON)  -- the warp-per-row grouped expert FFN. MEASURED
//       bit-identical: max_rel_diff 0, 0 differing of 2560, in all three
//       configurations. There is no numerical risk in it.
//
//   LING3_GEMV_GEMV  (default OFF) -- the warp-per-row plain matvec. It regroups a
//       row's float additions (float addition is not associative), so ~60 % of output
//       elements differ in their low bits and the greedy text diverges from build24 at
//       character 194. Its error against the CPU integer reference is the same ~1e-6
//       band as the kernel it replaces, and the perplexity gates pass (step 6.1014 vs
//       batched 6.1037 on positions 4096-8191; 4096-token step 9.2905 vs batched
//       9.3094) -- but end to end it bought 0.1 % (37.90/37.95 tok/s against 37.89/37.87
//       with it off), which does not pay for changing greedy output. Jay decides.
static int ls1_env_on(const char * name, int dflt) {
    const char * e = getenv(name);
    if (e == nullptr || e[0] == '\0') {
        return dflt;
    }
    return (e[0] == '1') ? 1 : 0;
}
static int ls1_variant_down() {
    static int v = -1;
    if (v < 0) {
        v = ls1_env_on("LING3_GEMV_DOWN", 1);
        if (!v && ls1_env_on("LING3_GEMV_VARIANT", 0)) {
            v = 1;   // the old single switch still turns both on
        }
    }
    return v;
}
static int ls1_variant_gemv() {
    static int v = -1;
    if (v < 0) {
        v = ls1_env_on("LING3_GEMV_GEMV", 0);
        if (v == 0 && ls1_env_on("LING3_GEMV_VARIANT", 0)) {
            v = 1;   // the old single switch still turns both on
        }
    }
    return v;
}

static int ls1_gemv_blocks(int n, int nwarps) {
    if (nwarps <= 0) return 1;
    const int want = (n + nwarps - 1) / nwarps;
    return want < 1 ? 1 : want;
}

// Variant 1 of the plain matvec, called directly so the test can time it against
// the CTA-per-row form at the same shapes. slot must already hold the Q8_1 vector.
extern "C" int ling3_gemv_fp4_q8_rows(int qtype, const void * W, int slot, float * y,
                                      int k, int n, int block, hipStream_t stream) {
    const size_t rb = ling3_row_bytes(qtype, k);
    if (!W || !y || rb == 0 || n <= 0 || (k % 32) != 0 || block < 32) return LING3_ERR_ARG;
    if (slot < 0 || slot >= LS1_NSLOT || g_qs[slot] == nullptr || g_cap[slot] < k) return LING3_ERR_ARG;
    const int nwarps = block >> 5;
    k_gemv_wide_q8_rows<<<ls1_gemv_blocks(n, nwarps), block, 0, stream>>>(
        qtype, (const uint8_t *) W, rb, g_qs[slot], g_ds[slot], y, k, n, 1 << 20);
    if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ls1_chk("gemv rows");
}

// LS1 PHASE7 item 3: quantise into an explicitly named slot, for the call sites
// that run several matvecs against ONE activation. ling3_gemv_fp4 still quantises for
// itself; this is the opt-in that lets a caller pay once.
extern "C" int ling3_act_prepare(const float * x, int k, int slot, hipStream_t stream) {
    return ling3_quant_q8(x, k, slot, stream);
}

extern "C" int ling3_gemv_fp4_q8(int qtype, const void * W, int slot,
                                 float * y, int k, int n, hipStream_t stream) {
    size_t rb = ling3_row_bytes(qtype, k);
    if (W == nullptr || y == nullptr || rb == 0 || n <= 0 || (k % 32) != 0 ||
        slot < 0 || slot >= LS1_NSLOT || g_qs[slot] == nullptr || g_cap[slot] < k) {
        return LING3_ERR_ARG;
    }
    const int8_t * qs = g_qs[slot];
    const float * ds = g_ds[slot];
    if (ls1_variant_gemv() && n <= LS1_GEMV_WIDE_MAX) {
        k_gemv_wide_q8_rows<<<ls1_gemv_blocks(n, 8), 256, 0, stream>>>(
            qtype, (const uint8_t *) W, rb, qs, ds, y, k, n, 1 << 20);
        if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
        return ls1_chk("gemv rows sel");
    }
    if (n <= LS1_GEMV_WIDE_MAX) {
        k_gemv_wide_q8_cta64<<<n, 64, 0, stream>>>(qtype, (const uint8_t *) W, rb, qs, ds, y, k, n);
        if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    } else {
        const int blocks = n < LS1_GEMV_GS_BLOCKS ? n : LS1_GEMV_GS_BLOCKS;
        k_gemv_wide_q8_gs<<<blocks, 64, 0, stream>>>(qtype, (const uint8_t *) W, rb, qs, ds, y, k, n);
        if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    }
    return ls1_chk("gemv q8");
}

// RV25 wave1 F3: matvec whose STORE is y[row] = 0.0f + s, so a down projection can
// write the caller's output directly and take the place of
//     hipMemset(y, 0, ...) ; <gemv -> p> ; k_axpy(1.0f, p, y)
// with no second launch and no scratch. `slot` must already hold the Q8_1 vector,
// exactly like ling3_gemv_fp4_q8; the accumulation is the same kernel body.
// Decode only: n <= LS1_GEMV_WIDE_MAX (a down projection has n = hidden = 2560; the
// LM head is the only n > 8192 matvec and it is not a down projection).
extern "C" int ling3_gemv_fp4_q8_zp(int qtype, const void * W, int slot,
                                    float * y, int k, int n, hipStream_t stream) {
    size_t rb = ling3_row_bytes(qtype, k);
    if (W == nullptr || y == nullptr || rb == 0 || n <= 0 || (k % 32) != 0 ||
        slot < 0 || slot >= LS1_NSLOT || g_qs[slot] == nullptr || g_cap[slot] < k) {
        return LING3_ERR_ARG;
    }
    const int8_t * qs = g_qs[slot];
    const float * ds = g_ds[slot];
    if (ls1_variant_gemv() && n <= LS1_GEMV_WIDE_MAX) {
        k_gemv_wide_q8_rows_zp<<<ls1_gemv_blocks(n, 8), 256, 0, stream>>>(
            qtype, (const uint8_t *) W, rb, qs, ds, y, k, n, 1 << 20);
        if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
        return ls1_chk("gemv rows zp");
    }
    if (n > LS1_GEMV_WIDE_MAX) {
        return LING3_ERR_ARG;
    }
    k_gemv_wide_q8_cta64_zp<<<n, 64, 0, stream>>>(qtype, (const uint8_t *) W, rb, qs, ds, y, k, n);
    if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ls1_chk("gemv q8 zp");
}

// Matvec that quantises the activation itself. One extra tiny kernel per call;
// the caller gets the right answer with no cooperation.
extern "C" int ling3_gemv_fp4(int qtype, const void * W, const float * x,
                              float * y, int k, int n, hipStream_t stream) {
    const int rc = ling3_quant_q8(x, k, LS1_SLOT_GENERIC, stream);
    if (rc != LING3_OK) {
        return rc;
    }
    return ling3_gemv_fp4_q8(qtype, W, LS1_SLOT_GENERIC, y, k, n, stream);
}

// Copy a slot's quantised activation back to the host. Test-only: it exists so
// tools/ling3_gemv_q8_test.cpp can tell a wrong quantisation from a wrong dot
// without having to guess which one it is looking at.
extern "C" int ling3_q8_peek(int slot, int8_t * qs_out, float * ds_out, int k) {
    if (slot < 0 || slot >= LS1_NSLOT || k <= 0 || g_qs[slot] == nullptr ||
        g_cap[slot] < k) {
        return LING3_ERR_ARG;
    }
    hipError_t e = hipMemcpy(qs_out, g_qs[slot], (size_t) k, hipMemcpyDeviceToHost);
    if (e != hipSuccess) return LING3_ERR_HIP;
    e = hipMemcpy(ds_out, g_ds[slot], (size_t) (k / 32) * sizeof(float), hipMemcpyDeviceToHost);
    return e == hipSuccess ? LING3_OK : LING3_ERR_HIP;
}

// ---------------------------------------------------------------------------
// Exhaustively check that the perm codebook above really is
// ling3_rocmfp4_code, for all 65536 four-nibble words. Returns the number of
// byte lanes that disagree; 0 means the table is a faithful transcription. This
// is the one piece of the integer path that cannot be checked by comparing
// against a float reference, because a wrong code permutes the products instead
// of scaling them, and a float reference would just look like quantisation noise.
// (It is also the check that caught a 2x error in an earlier revision of this
// file, which a float comparison would have read as ordinary quantisation noise.)
// ---------------------------------------------------------------------------
__global__ void k_cb_selftest(int * bad) {
    const int w = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= 65536) {
        return;
    }
    const int2 v = ls1_fp4_codebook16(w);
    const int8_t * o = (const int8_t *) &v;
    const uint8_t * qb = (const uint8_t *) &w;
    int local = 0;
    for (int i = 0; i < 4; ++i) {
        // lane i of .x is the LOW nibble of byte i, which is element i of the block;
        // lane i of .y is the HIGH nibble of byte i, which is element 16 + i.
        if ((int) o[i] != ling3_rocmfp4_code(qb[i] & 0x0fu)) local++;
        if ((int) o[4 + i] != ling3_rocmfp4_code(qb[i] >> 4)) local++;
    }
    if (local != 0) {
        atomicAdd(bad, local);
    }
}

extern "C" int ling3_gemv_q8_codebook_check(void) {
    int * d_bad = nullptr;
    if (hipMalloc(&d_bad, sizeof(int)) != hipSuccess) {
        return -1;
    }
    hipMemset(d_bad, 0, sizeof(int));
    k_cb_selftest<<<65536 / 256, 256, 0, (hipStream_t) nullptr>>>(d_bad);
    if (::superroc::hip_check::LaunchFailed()) return -1;
    int bad = -1;
    if (hipDeviceSynchronize() != hipSuccess || hipMemcpy(&bad, d_bad, sizeof(int),
                                                         hipMemcpyDeviceToHost) != hipSuccess) {
        bad = -1;
    }
    hipFree(d_bad);
    return bad;
}
// ---------------------------------------------------------------------------
// Grouped, device-routed expert FFN (LS1 phases 2 and 3).
//
// The per-expert host loop this replaces did, per MoE layer, two blocking D2H copies of
// the router's 8 ids and 8 weights -- 40 GPU syncs per token -- and then roughly 56 kernel
// launches, ~2,240 of them per token across the 40 MoE layers, each a full round trip
// through the scheduler for about 1.3 MB of work.
//
// This is three launches per layer and zero syncs:
//
//   1. k_moe_gate_up_swiglu -- one CTA per (selected expert slot, output row). The CTA
//      reads its slot's id from DEVICE memory, dots the gate row and the up row against
//      the ONE quantisation of the FFN input, and fuses SwiGLU into its epilogue.
//   2. k_q8_1_quant over hs -- the down projection needs Q8_1, and a 32-value group's
//      scale spans 32 different CTAs of step 1, so it cannot be formed there.
//   3. k_moe_down_weighted -- one CTA per output row, looping the n_used slots IN SLOT
//      ORDER and accumulating in a register: the weighted sum is in a fixed order every
//      run, with no float atomics and no dependence on scheduling.
//
// One quantisation of the FFN input serves every expert and the weight bytes read are
// unchanged: 8 experts x (2 x 768 x 2560 + 2560 x 768) values per layer, the same 25 MB,
// now in three launches instead of ~56.
// ---------------------------------------------------------------------------
__global__ void k_moe_gate_up_swiglu(int qtype, const uint8_t * gate, const uint8_t * up,
                                     size_t rb_k, const int * ids,
                                     const int8_t * xq, const float * xd,
                                     float * hs, float limit,
                                     int n_ff, int n_expert, int k, int * rc) {
    // One flat grid over (slot, row) rather than a 2D one, so the slot stays in the same
    // expression as the row. The division is once per CTA on ~6k CTAs.
    const int r = (int) blockIdx.x;
    const int s = r / n_ff;                  // which selected expert
    const int j = r - s * n_ff;              // output row inside that expert
    __shared__ float red[32];

    // Every thread reads the same ids[s], so `ok` is uniform across the CTA and every
    // thread reaches both barriers inside ls1_block_sum. A bad id still dots a real (if
    // meaningless) row rather than returning, because a divergent return around a
    // __syncthreads is a hang, and the status is reported through *rc instead.
    const int id = ids[s];
    const bool ok = (id >= 0 && id < n_expert);
    if (!ok && threadIdx.x == 0) {
        *rc = LING3_ERR_ARG;
    }
    const size_t base = ok ? (((size_t) id * (size_t) n_ff + (size_t) j) * rb_k) : 0;

    const float g = ls1_block_sum(ls1_dot_row_q8(qtype, gate + base, xq, xd, k), red);
    const float u = ls1_block_sum(ls1_dot_row_q8(qtype, up + base, xq, xd, k), red);
    if (threadIdx.x == 0) {
        hs[(size_t) s * (size_t) n_ff + (size_t) j] = ling3_swiglu_elem(g, u, limit);
    }
}

__global__ void k_moe_down_weighted(int qtype, const uint8_t * down, size_t rb_ff,
                                    const int * ids, const float * weights,
                                    const int8_t * hq, const float * hd,
                                    float * y, int n_used, int n_expert,
                                    int k, int n_ff, int * rc) {
    const int j = (int) blockIdx.x;          // output row of the down projection
    __shared__ float red[32];
    float acc = 0.f;
    // Fixed slot order: this is what makes the sum deterministic. Do NOT reorder this loop
    // and do NOT replace acc with an atomicAdd -- both silently make the result depend on
    // which CTA wins, which is the property this path exists to have.
    for (int s = 0; s < n_used; ++s) {
        const int id = ids[s];
        const bool ok = (id >= 0 && id < n_expert);
        if (!ok && threadIdx.x == 0) {
            *rc = LING3_ERR_ARG;
        }
        const size_t base = ok ? (((size_t) id * (size_t) k + (size_t) j) * rb_ff) : 0;
        const float part = ls1_block_sum(ls1_dot_row_q8(qtype, down + base,
                                                         hq + (size_t) s * (size_t) n_ff,
                                                         hd + (size_t) s * (size_t) (n_ff / 32),
                                                         n_ff),
                                          red);
        if (threadIdx.x == 0) {
            acc += weights[s] * part;
        }
    }
    if (threadIdx.x == 0) {
        y[j] = acc;
    }
}

// Scratch for hs / its Q8_1 form, and the status word. n_used * n_ff floats is 24 KiB at
// Ling's sizes.
static float * ls1_moe_hs = nullptr;
static int ls1_moe_hs_cap = 0;
static int * ls1_moe_rc = nullptr;
// The router's own status word, registered by ling3_route (which cannot free it -- it owns
// the allocation) so that ONE read covers the whole decode token. See ling3_moe_dev_check.
static int * ls1_route_rc_d = nullptr;

// Device-routed grouped expert FFN for ONE token. `ids` and `weights` are DEVICE pointers
// (what ling3_route wrote); nothing is read back to the host and NOTHING SYNCS. The status
// word is left on the device for ling3_moe_dev_check() to read once per token -- reading it
// here would put back exactly the per-layer drain this path exists to remove.
// ROCmFP4 (100/101) and a single token; anything else returns LING3_ERR_ARG so the caller
// can fall back to ling3_expert_ffn.
extern "C" int ling3_expert_ffn_dev(
        int qtype, const void * gate, const void * up, const void * down,
        const float * x, const int * ids_d, const float * weights_d,
        int n_used, int n_expert, int k, int n_ff, float limit, float * y,
        hipStream_t stream) {
    const size_t rb_k = ling3_row_bytes(qtype, k);
    const size_t rb_ff = ling3_row_bytes(qtype, n_ff);
    if (!gate || !up || !down || !x || !ids_d || !weights_d || !y) return LING3_ERR_ARG;
    if (qtype != LING3_Q4_0_ROCMFP4 && qtype != LING3_Q4_0_ROCMFP4_FAST) return LING3_ERR_ARG;
    if (rb_k == 0 || rb_ff == 0 || n_used <= 0 || n_expert <= 0 || k <= 0 || n_ff <= 0) {
        return LING3_ERR_ARG;
    }
    if ((k % 32) != 0 || (n_ff % 32) != 0) return LING3_ERR_ARG;

    const size_t nn = (size_t) n_used * (size_t) n_ff;
    if ((int) nn > ls1_moe_hs_cap || ls1_moe_hs == nullptr || ls1_moe_rc == nullptr) {
        if (ls1_moe_hs) { hipFree(ls1_moe_hs); ls1_moe_hs = nullptr; }
        if (ls1_moe_rc) { hipFree(ls1_moe_rc); ls1_moe_rc = nullptr; }
        ls1_moe_hs_cap = 0;
        if (hipMalloc(&ls1_moe_hs, nn * sizeof(float) + LING3_TAIL_PAD) != hipSuccess) {
            return LING3_ERR_HIP;
        }
        if (hipMalloc(&ls1_moe_rc, sizeof(int)) != hipSuccess) {
            hipFree(ls1_moe_hs); ls1_moe_hs = nullptr; return LING3_ERR_HIP;
        }
        ls1_moe_hs_cap = (int) nn;
        hipMemset(ls1_moe_rc, 0, sizeof(int));
    }

    int rc = ling3_quant_q8(x, k, LS1_SLOT_FFN_X, stream);
    if (rc != LING3_OK) return rc;
    const int8_t * xq = g_qs[LS1_SLOT_FFN_X];
    const float * xd = g_ds[LS1_SLOT_FFN_X];

    k_moe_gate_up_swiglu<<<(unsigned) ((n_ff + 63) / 64) * 64 * (unsigned) n_used, 64, 0, stream>>>(
        qtype, (const uint8_t *) gate, (const uint8_t *) up, rb_k,
        ids_d, xq, xd, ls1_moe_hs, limit, n_ff, n_expert, k, ls1_moe_rc);
    if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    if ((rc = ls1_chk("moe gate/up")) != LING3_OK) return rc;

    rc = ling3_quant_q8(ls1_moe_hs, (int) nn, LS1_SLOT_MOE_H, stream);
    if (rc != LING3_OK) return rc;

    // LS1 PHASE7 item 4: the down half defaults ON. It is MEASURED bit-identical
    // (max_rel_diff 0, 0 differing of 2560, all three configurations), so it carries no
    // numerical risk, and it is 1.22x faster on the isolated A/B.
    // (An earlier revision of this file had lost the branch here entirely, which is why
    // the first phase-7 "variant ON" numbers turned out to have exercised ONLY the
    // plain-GEMV half.)
    if (ls1_variant_down()) {
        const int nwarps = 8;
        k_moe_down_weighted_rows<<<ls1_gemv_blocks(k, nwarps), nwarps * 32, 0, stream>>>(
            qtype, (const uint8_t *) down, rb_ff, ids_d, weights_d,
            g_qs[LS1_SLOT_MOE_H], g_ds[LS1_SLOT_MOE_H], y, n_used, n_expert, k, n_ff,
            1 << 20, ls1_moe_rc);
        if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
        return ls1_chk("moe down rows");
    }
    k_moe_down_weighted<<<(unsigned) k, 64, 0, stream>>>(qtype, (const uint8_t *) down, rb_ff,
                                              ids_d, weights_d,
                                              g_qs[LS1_SLOT_MOE_H], g_ds[LS1_SLOT_MOE_H],
                                              y, n_used, n_expert, k, n_ff, ls1_moe_rc);
    if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ls1_chk("moe down");
}

// Called by ling3_route once, when it allocates its status word, so that the per-token
// check below can read it without ling3_route reading it on every one of the 40 layers.
extern "C" void ling3_moe_dev_register_route_rc(int * rc_d) {
    ls1_route_rc_d = rc_d;
}

// Read the decode step's device status words -- the grouped FFN's and the router's -- and
// clear them. Call ONCE per token, after the last MoE layer, never per layer: this is the
// only blocking read in the decode path, and 40 of them is exactly what phase 3 removed.
//
// Clearing here rather than at the start of the token is deliberate: a status word that is
// only ever written on error would otherwise carry one token's failure into every token
// after it, and clearing costs nothing extra because the read is already blocking.
extern "C" int ling3_moe_dev_check(void) {
    int rc = LING3_OK;
    if (ls1_moe_rc == nullptr && ls1_route_rc_d == nullptr) {
        return LING3_OK;
    }
    int words[2] = {LING3_OK, LING3_OK};
    int * src[2] = {ls1_moe_rc, ls1_route_rc_d};
    for (int i = 0; i < 2; ++i) {
        if (src[i] == nullptr) {
            continue;
        }
        if (hipMemcpy(&words[i], src[i], sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
            return LING3_ERR_HIP;
        }
        if (words[i] != LING3_OK && rc == LING3_OK) {
            rc = words[i];
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (src[i] != nullptr) {
            hipMemset(src[i], 0, sizeof(int));
        }
    }
    return rc;
}

// Test-only: copy the grouped FFN's hs scratch back to the host.
extern "C" int ling3_moe_hs_peek(float * out, int n) {
    if (ls1_moe_hs == nullptr || n <= 0) return -1;
    return hipMemcpy(out, ls1_moe_hs, (size_t) n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
}

// ---------------------------------------------------------------------------
// LS1 phase 4, item 3: the small-m decode forward, for MTP verification.
//
// A speculative verify runs m = 2..4 rows of the SAME weights against m different
// activations. The m = 1 path reads every weight byte once per row, so m rows cost m times
// the bandwidth. This form reads each weight row ONCE and spends the extra work on m sets
// of dot products: the four v_perm of the codebook and the four 32-bit loads of the block
// are done once per 32 weights and reused for every column. That is llama.cpp's
// `mul_mat_vec_q` with ncols_dst <= 8, which is the model the fork uses too.
//
//   k_gemv_wide_q8_m  -- n CTAs, one per output row, m columns each, y is [m][n] token-major
//   k_moe_gu_m        -- the grouped expert FFN's gate+up+SwiGLU over (token x slot) pairs
//   k_moe_down_m      -- its down + weighted sum, m accumulators in one register set
//
// Both grouped kernels take the per-pair expert id from DEVICE memory and never sync, like
// their m = 1 counterparts. Nothing here is wired into Step; LS4 calls it.
enum { LS1_MAX_M = 4 };

__global__ void k_gemv_wide_q8_m(int qtype, const uint8_t * W, size_t row_bytes,
                                 const int8_t * xq, const float * xd,
                                 float * y, int k, int n, int m) {
    const int row = (int) blockIdx.x;
    if (row >= n) {
        return;
    }
    const uint8_t * w = W + (size_t) row * row_bytes;
    const int tsz = ling3_type_size(qtype);
    const int nblk = k / 32;
    const int nb32 = k / 32;
    // One 32-float reduction slot per column: ls1_block_sum's trailing barrier makes red
    // reusable, so the columns are reduced one after another without extra sync.
    __shared__ float red[LS1_MAX_M * 32];

    float s[LS1_MAX_M];
    for (int t = 0; t < LS1_MAX_M; ++t) {
        s[t] = 0.f;
    }
    for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
        // Decode the block ONCE; the m columns share it.
        const ls1_blk32 c = ls1_load_blk32(qtype, w + (size_t) qb * (size_t) tsz);
        for (int t = 0; t < m; ++t) {
            s[t] += ls1_dot_blk32(c, xq + (size_t) t * (size_t) k + (size_t) qb * 32,
                                  xd[(size_t) t * (size_t) nb32 + (size_t) qb]);
        }
    }
    for (int t = 0; t < m; ++t) {
        const float r = ls1_block_sum(s[t], red + t * 32);
        if (threadIdx.x == 0) {
            y[(size_t) t * (size_t) n + (size_t) row] = r;
        }
    }
}

// gate + up + SwiGLU for every (token, slot) pair. ids_d and weights_d are indexed by the
// flat slot s = token * n_used + slot, and are DEVICE pointers.
__global__ void k_moe_gu_m(int qtype, const uint8_t * gate, const uint8_t * up, size_t rb_k,
                           const int8_t * xq, const float * xd,
                           const int * ids_d, float * hs, float limit,
                           int n_used, int n_expert, int k, int n_ff, int n_slots,
                           int * rc) {
    const int r = (int) blockIdx.x;
    const int s = r / n_ff;                  // flat (token, slot)
    const int j = r - s * n_ff;              // output row inside that expert
    const int t = s / n_used;                // which token
    __shared__ float red[LS1_MAX_M * 32];

    const int id = ids_d[s];
    const bool ok = (id >= 0 && id < n_expert);
    if (!ok && threadIdx.x == 0) {
        *rc = LING3_ERR_ARG;
    }
    const size_t base = ok ? (((size_t) id * (size_t) n_ff + (size_t) j) * rb_k) : 0;
    const int8_t * xqt = xq + (size_t) t * (size_t) k;
    const float * xdt = xd + (size_t) t * (size_t) (k / 32);

    const float g = ls1_block_sum(ls1_dot_row_q8(qtype, gate + base, xqt, xdt, k), red);
    const float u = ls1_block_sum(ls1_dot_row_q8(qtype, up + base, xqt, xdt, k), red);
    if (threadIdx.x == 0) {
        hs[(size_t) s * (size_t) n_ff + (size_t) j] = ling3_swiglu_elem(g, u, limit);
    }
}

// down + the weighted sum over slots, m accumulators in one register set. The slot loop is
// in flat (token, slot) order and the tokens' accumulators are independent, so the order
// per token is fixed: the same determinism the m = 1 kernel has. The weight row is read
// ONCE per slot and the m tokens' hs rows each get their own dot, which is the same
// sharing the m-column matvec is built on.
__global__ void k_moe_down_m(int qtype, const uint8_t * down, size_t rb_ff,
                             const int * ids_d, const float * weights_d,
                             const int8_t * hq, const float * hd,
                             float * y, int n_used, int n_expert, int k, int n_ff,
                             int n_slots, int m, int * rc) {
    const int j = (int) blockIdx.x;          // output row of the down projection
    __shared__ float red[LS1_MAX_M * 32];
    float acc[LS1_MAX_M];
    for (int t = 0; t < LS1_MAX_M; ++t) {
        acc[t] = 0.f;
    }
    for (int s = 0; s < n_slots; ++s) {
        // hs[s] belongs to ONE token, s / n_used, so its dot goes into that token's
        // accumulator and no other. Adding it to all m of them was the first version of
        // this kernel and every element came out wrong.
        const int t = s / n_used;
        const int id = ids_d[s];
        const bool ok = (id >= 0 && id < n_expert);
        if (!ok && threadIdx.x == 0) {
            *rc = LING3_ERR_ARG;
        }
        const size_t base = ok ? (((size_t) id * (size_t) k + (size_t) j) * rb_ff) : 0;
        const float part = ls1_block_sum(
            ls1_dot_row_q8(qtype, down + base, hq + (size_t) s * (size_t) n_ff,
                           hd + (size_t) s * (size_t) (n_ff / 32), n_ff), red);
        if (threadIdx.x == 0) {
            acc[t] += weights_d[s] * part;
        }
    }
    for (int t = 0; t < m; ++t) {
        if (threadIdx.x == 0) {
            y[(size_t) t * (size_t) k + (size_t) j] = acc[t];
        }
    }
}

// LV1 R3: one expert, m rows. Each gate/up/down row is decoded once and dotted
// against every token. The per-thread block order, ls1_block_sum, SwiGLU and
// `acc += weight * part` match k_moe_gu_m / k_moe_down_m at n_used = 1.
__global__ void k_moe_gu_one_m(int qtype, const uint8_t * gate, const uint8_t * up,
                               size_t rb_k, const int8_t * xq, const float * xd,
                               const int * ids_d, float * hs, float limit,
                               int n_expert, int k, int n_ff, int m, int * rc) {
    const int j = (int) blockIdx.x;
    __shared__ float red[32];
    const int id = ids_d[0];
    bool ok = (id >= 0 && id < n_expert);
    for (int t = 1; t < m; ++t) {
        if (ids_d[t] != id) ok = false;
    }
    if (!ok && threadIdx.x == 0) {
        *rc = LING3_ERR_ARG;
    }
    const size_t base = ok ? (((size_t) id * (size_t) n_ff + (size_t) j) * rb_k) : 0;
    const int tsz = ling3_type_size(qtype);
    const int nblk = k / 32;
    const int nb32 = k / 32;
    const uint8_t * grow = gate + base;
    const uint8_t * urow = up + base;
    float sg[LS1_MAX_M], su[LS1_MAX_M];
    for (int t = 0; t < LS1_MAX_M; ++t) {
        sg[t] = 0.f;
        su[t] = 0.f;
    }
    for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
        const ls1_blk32 cg = ls1_load_blk32(qtype, grow + (size_t) qb * (size_t) tsz);
        const ls1_blk32 cu = ls1_load_blk32(qtype, urow + (size_t) qb * (size_t) tsz);
        for (int t = 0; t < m; ++t) {
            const int8_t * xqt = xq + (size_t) t * (size_t) k + (size_t) qb * 32;
            const float xdt = xd[(size_t) t * (size_t) nb32 + (size_t) qb];
            sg[t] += ls1_dot_blk32(cg, xqt, xdt);
            su[t] += ls1_dot_blk32(cu, xqt, xdt);
        }
    }
    for (int t = 0; t < m; ++t) {
        const float g = ls1_block_sum(sg[t], red);
        const float u = ls1_block_sum(su[t], red);
        if (threadIdx.x == 0) {
            hs[(size_t) t * (size_t) n_ff + (size_t) j] = ling3_swiglu_elem(g, u, limit);
        }
    }
}

__global__ void k_moe_down_one_m(int qtype, const uint8_t * down, size_t rb_ff,
                                 const int * ids_d, const float * weights_d,
                                 const int8_t * hq, const float * hd, float * y,
                                 int n_expert, int k, int n_ff, int m, int * rc) {
    const int j = (int) blockIdx.x;
    __shared__ float red[32];
    const int id = ids_d[0];
    bool ok = (id >= 0 && id < n_expert);
    for (int t = 1; t < m; ++t) {
        if (ids_d[t] != id) ok = false;
    }
    if (!ok && threadIdx.x == 0) {
        *rc = LING3_ERR_ARG;
    }
    const size_t base = ok ? (((size_t) id * (size_t) k + (size_t) j) * rb_ff) : 0;
    const int tsz = ling3_type_size(qtype);
    const int nblk = n_ff / 32;
    const uint8_t * drow = down + base;
    float part[LS1_MAX_M];
    float acc[LS1_MAX_M];
    for (int t = 0; t < LS1_MAX_M; ++t) {
        part[t] = 0.f;
        acc[t] = 0.f;
    }
    for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
        const ls1_blk32 c = ls1_load_blk32(qtype, drow + (size_t) qb * (size_t) tsz);
        for (int t = 0; t < m; ++t) {
            part[t] += ls1_dot_blk32(
                c, hq + (size_t) t * (size_t) n_ff + (size_t) qb * 32,
                hd[(size_t) t * (size_t) (n_ff / 32) + (size_t) qb]);
        }
    }
    for (int t = 0; t < m; ++t) {
        // n_used == 1, so slot s is the token and this is the only term.
        const float p = ls1_block_sum(part[t], red);
        if (threadIdx.x == 0) {
            acc[t] += weights_d[t] * p;
        }
    }
    if (threadIdx.x == 0) {
        for (int t = 0; t < m; ++t) {
            y[(size_t) t * (size_t) k + (size_t) j] = acc[t];
        }
    }
}

static float * ls1_moe_hs_m = nullptr;
static int ls1_moe_hs_m_cap = 0;
static int * ls1_moe_rc_m = nullptr;
static int8_t * g_qs_m = nullptr;
static float * g_ds_m = nullptr;
static int g_cap_m = 0;

// Quantise n_tokens * k floats (token-major) into the small-m path's own Q8_1 pair.
static int ls1_quant_m(const float * x, int total, hipStream_t stream) {
    if (x == nullptr || total <= 0 || (total % 32) != 0) return LING3_ERR_ARG;
    if (total > g_cap_m || g_qs_m == nullptr || g_ds_m == nullptr) {
        if (g_qs_m) { hipFree(g_qs_m); g_qs_m = nullptr; }
        if (g_ds_m) { hipFree(g_ds_m); g_ds_m = nullptr; }
        g_cap_m = 0;
        if (hipMalloc(&g_qs_m, (size_t) total + LING3_TAIL_PAD) != hipSuccess) return LING3_ERR_HIP;
        if (hipMalloc(&g_ds_m, (size_t) (total / 32) * sizeof(float) + LING3_TAIL_PAD) != hipSuccess) {
            hipFree(g_qs_m); g_qs_m = nullptr; return LING3_ERR_HIP;
        }
        g_cap_m = total;
    }
    const int nb = total / 32;
    const int threads = 256;
    k_q8_1_quant<<<(nb + 7) / 8, threads, 0, stream>>>(x, g_qs_m, g_ds_m, nb);
    if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ls1_chk("q8 quant m");
}

// Matvec of n rows against m activation columns, activations already in Q8_1.
// y is [m][n] token-major: y[t * n + row].
extern "C" int ling3_gemv_q8_m(int qtype, const void * W, const int8_t * xq, const float * xd,
                               float * y, int k, int n, int m, hipStream_t stream) {
    size_t rb = ling3_row_bytes(qtype, k);
    if (!W || !xq || !xd || !y || rb == 0 || n <= 0) return LING3_ERR_ARG;
    if (m < 1 || m > LS1_MAX_M || (k % 32) != 0) return LING3_ERR_ARG;
    if (qtype != LING3_Q4_0_ROCMFP4 && qtype != LING3_Q4_0_ROCMFP4_FAST) return LING3_ERR_ARG;
    k_gemv_wide_q8_m<<<(unsigned) n, 64, 0, stream>>>(qtype, (const uint8_t *) W, rb, xq, xd, y, k, n, m);
    if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ls1_chk("gemv q8 m");
}

// LM1 SPEC4 item 2: quantise-only entry. See the header for the sharing rule.
extern "C" int ling3_quant_m(const float * x, int n_tokens, int k, hipStream_t stream,
                             const int8_t ** xq, const float ** xd) {
    if (xq == nullptr || xd == nullptr) return LING3_ERR_ARG;
    if (n_tokens < 1 || n_tokens > LS1_MAX_M || k <= 0 || (k % 32) != 0) return LING3_ERR_ARG;
    const int rc = ls1_quant_m(x, n_tokens * k, stream);
    if (rc != LING3_OK) return rc;
    *xq = g_qs_m;
    *xd = g_ds_m;
    return LING3_OK;
}

// Same, but quantising m * k floats from float first.
extern "C" int ling3_gemv_m(int qtype, const void * W, const float * x, float * y,
                            int k, int n, int m, hipStream_t stream) {
    const int rc = ls1_quant_m(x, m * k, stream);
    if (rc != LING3_OK) return rc;
    return ling3_gemv_q8_m(qtype, W, g_qs_m, g_ds_m, y, k, n, m, stream);
}

// Grouped expert FFN over n_tokens * n_used (token, slot) pairs, for one token batch.
// ids_d / weights_d are DEVICE pointers, flat and indexed s = token * n_used + slot.
// y is [n_tokens][k] token-major. No sync, no host round trip.
extern "C" int ling3_expert_ffn_m(int qtype, const void * gate, const void * up,
                                 const void * down, const float * x,
                                 const int * ids_d, const float * weights_d,
                                 int n_tokens, int n_used, int n_expert,
                                 int k, int n_ff, float limit, float * y,
                                 hipStream_t stream) {
    const size_t rb_k = ling3_row_bytes(qtype, k);
    const size_t rb_ff = ling3_row_bytes(qtype, n_ff);
    if (!gate || !up || !down || !x || !ids_d || !weights_d || !y) return LING3_ERR_ARG;
    if (qtype != LING3_Q4_0_ROCMFP4 && qtype != LING3_Q4_0_ROCMFP4_FAST) return LING3_ERR_ARG;
    if (rb_k == 0 || rb_ff == 0) return LING3_ERR_ARG;
    if (n_tokens < 1 || n_tokens > LS1_MAX_M || n_used <= 0 || n_expert <= 0) return LING3_ERR_ARG;
    if (k <= 0 || n_ff <= 0 || (k % 32) != 0 || (n_ff % 32) != 0) return LING3_ERR_ARG;

    const int n_slots = n_tokens * n_used;
    const size_t nn = (size_t) n_slots * (size_t) n_ff;
    if ((int) nn > ls1_moe_hs_m_cap || ls1_moe_hs_m == nullptr || ls1_moe_rc_m == nullptr) {
        if (ls1_moe_hs_m) { hipFree(ls1_moe_hs_m); ls1_moe_hs_m = nullptr; }
        if (ls1_moe_rc_m) { hipFree(ls1_moe_rc_m); ls1_moe_rc_m = nullptr; }
        ls1_moe_hs_m_cap = 0;
        if (hipMalloc(&ls1_moe_hs_m, nn * sizeof(float) + LING3_TAIL_PAD) != hipSuccess) {
            return LING3_ERR_HIP;
        }
        if (hipMalloc(&ls1_moe_rc_m, sizeof(int)) != hipSuccess) {
            hipFree(ls1_moe_hs_m); ls1_moe_hs_m = nullptr; return LING3_ERR_HIP;
        }
        ls1_moe_hs_m_cap = (int) nn;
        hipMemset(ls1_moe_rc_m, 0, sizeof(int));
    }

    int rc = ls1_quant_m(x, n_tokens * k, stream);
    if (rc != LING3_OK) return rc;

    k_moe_gu_m<<<(unsigned) (n_ff * n_slots), 64, 0, stream>>>(qtype, (const uint8_t *) gate,
                                                     (const uint8_t *) up, rb_k,
                                                     g_qs_m, g_ds_m, ids_d,
                                                     ls1_moe_hs_m, limit,
                                                     n_used, n_expert, k, n_ff, n_slots,
                                                     ls1_moe_rc_m);
    if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    if ((rc = ls1_chk("moe m gate/up")) != LING3_OK) return rc;

    rc = ls1_quant_m(ls1_moe_hs_m, (int) nn, stream);
    if (rc != LING3_OK) return rc;

    k_moe_down_m<<<(unsigned) k, 64, 0, stream>>>(qtype, (const uint8_t *) down, rb_ff,
                                       ids_d, weights_d, g_qs_m, g_ds_m, y,
                                       n_used, n_expert, k, n_ff, n_slots, n_tokens,
                                       ls1_moe_rc_m);
    if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ls1_chk("moe m down");
}

// LV1 R3. n_tokens == 1 and n_used != 1 stay on ling3_expert_ffn_m, so Step's
// single-token expert and the routed verify (n_used = 8) do not change.
extern "C" int ling3_expert_ffn_shared_m(int qtype, const void * gate, const void * up,
                                        const void * down, const float * x,
                                        const int * ids_d, const float * weights_d,
                                        int n_tokens, int n_used, int n_expert,
                                        int k, int n_ff, float limit, float * y,
                                        hipStream_t stream) {
    if (n_used != 1 || n_tokens <= 1) {
        return ling3_expert_ffn_m(qtype, gate, up, down, x, ids_d, weights_d,
                                 n_tokens, n_used, n_expert, k, n_ff, limit, y, stream);
    }
    const size_t rb_k = ling3_row_bytes(qtype, k);
    const size_t rb_ff = ling3_row_bytes(qtype, n_ff);
    if (!gate || !up || !down || !x || !ids_d || !weights_d || !y) return LING3_ERR_ARG;
    if (qtype != LING3_Q4_0_ROCMFP4 && qtype != LING3_Q4_0_ROCMFP4_FAST) return LING3_ERR_ARG;
    if (rb_k == 0 || rb_ff == 0) return LING3_ERR_ARG;
    if (n_tokens > LS1_MAX_M || n_expert <= 0) return LING3_ERR_ARG;
    if (k <= 0 || n_ff <= 0 || (k % 32) != 0 || (n_ff % 32) != 0) return LING3_ERR_ARG;

    const size_t nn = (size_t) n_tokens * (size_t) n_ff;
    if ((int) nn > ls1_moe_hs_m_cap || ls1_moe_hs_m == nullptr || ls1_moe_rc_m == nullptr) {
        if (ls1_moe_hs_m) { hipFree(ls1_moe_hs_m); ls1_moe_hs_m = nullptr; }
        if (ls1_moe_rc_m) { hipFree(ls1_moe_rc_m); ls1_moe_rc_m = nullptr; }
        ls1_moe_hs_m_cap = 0;
        if (hipMalloc(&ls1_moe_hs_m, nn * sizeof(float) + LING3_TAIL_PAD) != hipSuccess) {
            return LING3_ERR_HIP;
        }
        if (hipMalloc(&ls1_moe_rc_m, sizeof(int)) != hipSuccess) {
            hipFree(ls1_moe_hs_m); ls1_moe_hs_m = nullptr; return LING3_ERR_HIP;
        }
        ls1_moe_hs_m_cap = (int) nn;
        hipMemset(ls1_moe_rc_m, 0, sizeof(int));
    }

    int rc = ls1_quant_m(x, n_tokens * k, stream);
    if (rc != LING3_OK) return rc;

    k_moe_gu_one_m<<<(unsigned) n_ff, 64, 0, stream>>>(
        qtype, (const uint8_t *) gate, (const uint8_t *) up, rb_k,
        g_qs_m, g_ds_m, ids_d, ls1_moe_hs_m, limit, n_expert, k, n_ff, n_tokens,
        ls1_moe_rc_m);
    if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    if ((rc = ls1_chk("shared m gate/up")) != LING3_OK) return rc;

    rc = ls1_quant_m(ls1_moe_hs_m, (int) nn, stream);
    if (rc != LING3_OK) return rc;

    k_moe_down_one_m<<<(unsigned) k, 64, 0, stream>>>(
        qtype, (const uint8_t *) down, rb_ff, ids_d, weights_d, g_qs_m, g_ds_m, y,
        n_expert, k, n_ff, n_tokens, ls1_moe_rc_m);
    if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ls1_chk("shared m down");
}

// Read the small-m path's status word and clear it. Blocking; one call per verify batch.
// Variant 1 of the grouped down projection. hs is already quantised into the MoE slot.
// The FULL grouped expert FFN with the warp-per-row down kernel, so the A/B in
// the test compares like with like: the previous version of that benchmark timed
// ling3_expert_ffn_dev (gate + up + quantise + down) against the down kernel
// alone and reported 0.26x, which is a comparison of three stages with one.
extern "C" int ling3_expert_ffn_dev_tuned(int qtype, const void * gate, const void * up,
                                         const void * down, const float * x,
                                         const int * ids_d, const float * weights_d,
                                         int n_used, int n_expert, int k, int n_ff,
                                         float limit, float * y, int block,
                                         hipStream_t stream) {
    const size_t rb_k = ling3_row_bytes(qtype, k);
    const size_t rb_ff = ling3_row_bytes(qtype, n_ff);
    if (!gate || !up || !down || !x || !ids_d || !weights_d || !y) return LING3_ERR_ARG;
    if (qtype != LING3_Q4_0_ROCMFP4 && qtype != LING3_Q4_0_ROCMFP4_FAST) return LING3_ERR_ARG;
    if (rb_k == 0 || rb_ff == 0) return LING3_ERR_ARG;
    if (n_used <= 0 || n_expert <= 0 || k <= 0 || n_ff <= 0) return LING3_ERR_ARG;
    if ((k % 32) != 0 || (n_ff % 32) != 0 || block < 32) return LING3_ERR_ARG;

    const size_t nn = (size_t) n_used * (size_t) n_ff;
    if ((int) nn > ls1_moe_hs_cap || ls1_moe_hs == nullptr || ls1_moe_rc == nullptr) {
        if (ls1_moe_hs) { hipFree(ls1_moe_hs); ls1_moe_hs = nullptr; }
        if (ls1_moe_rc) { hipFree(ls1_moe_rc); ls1_moe_rc = nullptr; }
        ls1_moe_hs_cap = 0;
        if (hipMalloc(&ls1_moe_hs, nn * sizeof(float) + LING3_TAIL_PAD) != hipSuccess) return LING3_ERR_HIP;
        if (hipMalloc(&ls1_moe_rc, sizeof(int)) != hipSuccess) {
            hipFree(ls1_moe_hs); ls1_moe_hs = nullptr; return LING3_ERR_HIP;
        }
        ls1_moe_hs_cap = (int) nn;
        hipMemset(ls1_moe_rc, 0, sizeof(int));
    }
    int rc = ling3_quant_q8(x, k, LS1_SLOT_FFN_X, stream);
    if (rc != LING3_OK) return rc;
    const int8_t * xq = g_qs[LS1_SLOT_FFN_X];
    const float * xd = g_ds[LS1_SLOT_FFN_X];

    k_moe_gate_up_swiglu<<<(unsigned) (((n_ff + 63) / 64) * 64 * n_used), 64, 0, stream>>>(
        qtype, (const uint8_t *) gate, (const uint8_t *) up, rb_k,
        ids_d, xq, xd, ls1_moe_hs, limit, n_ff, n_expert, k, ls1_moe_rc);
    if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    if ((rc = ls1_chk("moe gate/up")) != LING3_OK) return rc;
    rc = ling3_quant_q8(ls1_moe_hs, (int) nn, LS1_SLOT_MOE_H, stream);
    if (rc != LING3_OK) return rc;
    const int nwarps = block >> 5;
    k_moe_down_weighted_rows<<<ls1_gemv_blocks(k, nwarps), block, 0, stream>>>(
        qtype, (const uint8_t *) down, rb_ff, ids_d, weights_d,
        g_qs[LS1_SLOT_MOE_H], g_ds[LS1_SLOT_MOE_H], y, n_used, n_expert, k, n_ff,
        1 << 20, ls1_moe_rc);
    if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ls1_chk("moe down rows tuned");
}

extern "C" int ling3_expert_ffn_dev_rows(int qtype, const void * down,
                                         const int * ids_d, const float * weights_d,
                                         int n_used, int n_expert, int k, int n_ff,
                                         float * y, int block, hipStream_t stream) {
    const size_t rb_ff = ling3_row_bytes(qtype, n_ff);
    if (!down || !ids_d || !weights_d || !y || rb_ff == 0 || block < 32) return LING3_ERR_ARG;
    const int nwarps = block >> 5;
    k_moe_down_weighted_rows<<<ls1_gemv_blocks(k, nwarps), block, 0, stream>>>(
        qtype, (const uint8_t *) down, rb_ff, ids_d, weights_d,
        g_qs[LS1_SLOT_MOE_H], g_ds[LS1_SLOT_MOE_H], y, n_used, n_expert, k, n_ff,
        1 << 20, ls1_moe_rc);
    if (::superroc::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ls1_chk("moe down rows");
}

extern "C" int ling3_expert_ffn_m_check(void) {
    if (ls1_moe_rc_m == nullptr) return LING3_OK;
    int rc = LING3_OK;
    if (hipMemcpy(&rc, ls1_moe_rc_m, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
        return LING3_ERR_HIP;
    }
    hipMemset(ls1_moe_rc_m, 0, sizeof(int));
    return rc;
}

// Test-only: the small-m path's hs scratch.
extern "C" int ling3_moe_hs_m_peek(float * out, int n) {
    if (ls1_moe_hs_m == nullptr || n <= 0) return -1;
    return hipMemcpy(out, ls1_moe_hs_m, (size_t) n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
}
