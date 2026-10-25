// ling3_wmma.hip — LS3 phase 3: W8A8 int8-WMMA GEMM for the ROCmFP4 prefill.
// See ling3_wmma.hpp for the design and the contracts.
//
// Structure follows gufo's WKQuantA8BlockedWmmaGEMMKernel
// (src/models/qwen/hip/kernels/prefill_quant_gemm.hpp) — the kernel gufo's
// Qwen prefill runs on this exact GPU — reduced to what Ling needs:
//   * WaveSize 64 (the w64 builtin gufo's TryLaunchQuantPrefillWave64 uses),
//     raw __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64 — NOT rocWMMA/CK, which
//     silently no-op WMMA on gfx1151 under ROCm 7.2 (CK #3724).
//   * PerHalfScale = true, HasOffset = false: ROCmFP4's dual e[0]/e[1]
//     16-element scales are exactly the kernel's per-half-scale machinery;
//     for the FAST type the second half reuses e[0].
//   * weights decoded FP4 -> int8 in registers by the fork's Codebook10 perm
//     (rocmfp4_get_low_int_from_codebook_16, rocmfp4_hip_codebook.cuh; LS1
//     verified the perm's selector order exhaustively on this box); nothing
//     is repacked at load — the GGUF stays packed for the decode path.
//   * activations pre-quantised to the 576-byte tiled Q8_1 layout by
//     ling3_wmma_quant (gufo's QuantizeActivationToQ8_1Kernel, no sidecar:
//     ROCmFP4 has no offset term).
//   * out-of-range rows / k-blocks are clamped and their scale zeroed by a
//     multiply (gufo opt-c179): bit-identical to zero-filling the operands,
//     and every load stays inside a real block of a real row.
//   * the epilogue writes straight from the WMMA accumulator registers:
//     with our y layout (slot/token-major, y[j*n + r]) lane (sub_lane,
//     half_id) of element l owns C(row = (WS/16)*l + half_id, tok =
//     sub_lane) — the same ownership gufo's scratch dance implements — so
//     no LDS round trip and no wave barrier are needed.
//
// Instantiations:
//   * grouped MoE: <64, 16, 2, 4, 1, 64> — 64 weight rows x 16 slots x k.
//     Real routing (512 experts, 8 used) gives ~13 slots per used expert, so
//     a 16-wide token tile is the "2x the mean bucket" rule; the launch grid
//     is exactly the tiles the per-group counts imply (RW3 step 1's MoE-aware
//     grid, built in from the start), flattened through the grp_zp prefix so
//     no block is ever dispatched unused.
//   * dense: <128, 64, 2, 4, 1, 64> for the KDA/MLA-out/shared/dense
//     projections.

#include "ling3_wmma.hpp"
#include "src/core/hip_launch_check.hpp"

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <cstdlib>

namespace ls3w {

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;

constexpr int kActTileTokens = 16;
constexpr int kActTileBytes = 576;
constexpr int kActScaleOffset = 512;

// ---------------------------------------------------------------------------
// FP4 -> int8 decode of one 16-element sub-block (the fork's Codebook10).
// ---------------------------------------------------------------------------
// Byte layout of one 32-element block (both ROCmFP4 types):
//   bytes 0..15 : 32 packed nibbles, element i = byte i&15, low nibble for
//                 i < 16, high nibble for i >= 16
//   byte 16     : ue4m3 half-scale of elements 0..15
//   byte 17     : type 100: half-scale of elements 16..31; type 101: absent.
// `sub16` indexes 16-element sub-blocks of the ROW; sub16 & 1 selects the
// half. The 16-byte qs load is a memcpy from inside the block and cannot
// walk off the row; the scale bytes are read at block offsets 16/17, which
// every block of both types has.
__device__ __forceinline__ void fp4_sub16(int fast, const uint8_t * __restrict__ row,
                                         int sub16, int32x4_t & q, float & scale) {
    const int tsz = fast ? 17 : 18;
    const int b = sub16 >> 1;
    const int half = sub16 & 1;
    const uint8_t * blk = row + (size_t) b * (size_t) tsz;

    uint32_t packed[4];
    __builtin_memcpy(packed, blk, 16);

    constexpr uint32_t values0 = 0x03020100u;
    constexpr uint32_t values1 = 0x0a080604u;
    constexpr uint32_t values2 = 0xfdfeff00u;
    constexpr uint32_t values3 = 0xf6f8fafcu;

    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // One nibble per byte, in element order: half 0 keeps the low
        // nibbles (elements 0..15 of the block), half 1 shifts the high
        // nibbles down (elements 16..31).
        const uint32_t nib = half ? (packed[i] >> 4) : packed[i];
        const uint32_t v_low = __builtin_amdgcn_perm(values1, values0,
                                                     nib & 0x07070707u);
        const uint32_t v_high = __builtin_amdgcn_perm(values3, values2,
                                                     nib & 0x07070707u);
        const uint32_t mask = 0x03020100u | ((nib & 0x08080808u) >> 1);
        w[i] = __builtin_amdgcn_perm(v_high, v_low, mask);
    }
    __builtin_memcpy(&q, w, 16);
    scale = ling3_ue4m3_half(blk[16 + ((half != 0 && !fast) ? 1 : 0)]);
}

__device__ __forceinline__ int32x4_t wmma_w64(int32x4_t a, int32x4_t b, int32x4_t c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, a, true, b, c, true);
}

// ---------------------------------------------------------------------------
// activation quantiser: one warp per (row, 32-block), gufo's
// QuantizeActivationToQ8_1Kernel (max-abs/127, roundf) into the 576-byte
// tile layout, at the (possibly group-remapped) destination tile.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void k_q8_quant(
        const float * __restrict__ x, const int * __restrict__ x_rows,
        const int * __restrict__ dest, int8_t * __restrict__ q8,
        int n_rows, int num_blocks) {
    const long long w =
        (long long) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w >= (long long) n_rows * num_blocks) {
        return;
    }
    const int r = (int) (w / num_blocks);
    const int blk = (int) (w % num_blocks);
    const int lane = (int) threadIdx.x & 31;
    const int src = x_rows != nullptr ? x_rows[r] : r;
    const int dp = dest != nullptr ? dest[r] : r;
    const int k = num_blocks * 32;
    const float val = x[(size_t) src * (size_t) k + (size_t) blk * 32 + lane];
    float max_abs = fabsf(val);
    // Width 32 explicitly: this file compiles with -mwavefrontsize64 (the
    // WMMA builtin needs it), where the two-argument form would reduce
    // across the whole 64-lane wave and mix two work items.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off, 32));
    }
    const float d = max_abs / 127.0f;
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    const int8_t q = (int8_t) roundf(val * id);

    // tile[(k-half * 256) + (token * 16) + element-within-16], scale at
    // tile+512 + token*4 — gufo's StoreQ8ActLane without the sum sidecar.
    int8_t * tile =
        q8 + ((size_t) ((dp >> 4) * num_blocks + blk) * (size_t) kActTileBytes);
    tile[((lane >> 4) * 256) + ((dp & 15) * 16) + (lane & 15)] = q;
    if (lane == 0) {
        reinterpret_cast<float *>(tile + kActScaleOffset)[dp & 15] = d;
    }
}
// NOTE (measured, see the unit gate): this file compiles with -ffast-math,
// and on gfx11 that lowers `max_abs / 127.0f` to a reciprocal multiply --
// the same compiled form as gufo's own QuantizeActivationToQ8_1Kernel and
// LS1's shipped ling3_gemv_q8 quantizer (same flags), 1 ulp from the exact
// division. The quantise arithmetic is otherwise LS1's recipe verbatim
// (max|x| per 32, d = max/127, q = roundf(x/d)). Deterministic: same binary,
// same bytes.

// ---------------------------------------------------------------------------
// the GEMM kernel
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int WM, int WN, int WaveSize>
__global__ __launch_bounds__(WM * WN * WaveSize, 1) void k_wmma_grp(
        int fast, const uint8_t * __restrict__ W, size_t row_bytes,
        size_t expert_stride, const int * __restrict__ grp_expert,
        const int * __restrict__ grp_off, const int * __restrict__ grp_cnt,
        const int * __restrict__ grp_zp, int n_groups, int m,
        const int8_t * __restrict__ x_blocks, float * __restrict__ y,
        const float * __restrict__ y_scale, int k, int n) {
    static_assert(WaveSize == 64 && WM * WN == 4, "w64 instantiation only");
    static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
    constexpr int kThreads = WM * WN * WaveSize;
    constexpr int kAcc = 256 / WaveSize;
    constexpr int kRowTiles = BM / 16;
    constexpr int kTokTiles = BN / 16;
    constexpr int kWaveRowTiles = kRowTiles / WM;
    constexpr int kWaveTokTiles = kTokTiles / WN;
    constexpr int kStageWaves = kThreads / 32;
    static_assert(kTokTiles <= kStageWaves);
    constexpr int kStageTiles = 1;
    constexpr int kFetchUnits = BM * BK;
    constexpr int kPrefetch = (kFetchUnits + kThreads - 1) / kThreads;
    constexpr bool kPartialFetch = (kPrefetch * kThreads) != kFetchUnits;

    const int num_kb = k / 32;

    // ---- which (group, token-tile) this block is ----
    int expert, off, cnt, z, act_tile0;
    if (grp_zp != nullptr) {
        // LG1 Change 1: swap so the inner-loop grid dim carries the row tile.
        // binary search the 16-token-tile prefix for this block's group;
        // grp_zp[g] <= blockIdx.y < grp_zp[g+1]
        int lo = 0;
        int hi = n_groups;
        const int bflat = (int) blockIdx.y;
        while (lo + 1 < hi) {
            const int mid = (lo + hi) >> 1;
            if (grp_zp[mid] <= bflat) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        const int g = lo;
        z = bflat - grp_zp[g];
        expert = grp_expert[g];
        off = grp_off[g];
        cnt = grp_cnt[g];
        act_tile0 = grp_zp[g] + z * kTokTiles;
    } else {
        // LG1 Change 1: dense (LING3_WMMA_GEMM_DENSE) launch transposed; z uses blockIdx.y.
        z = (int) blockIdx.y;
        expert = 0;
        off = 0;
        cnt = m;
        act_tile0 = z * kTokTiles;
    }
    const int t_local = z * BN;   // group-local token base of this block
    if (t_local >= cnt) {
        return;   // the grouped grid is exact; this is dense-tail safety only
    }
    const uint8_t * Wg = W + (size_t) expert * expert_stride;

    const int tid = (int) threadIdx.x;
    const int wave_id = tid / WaveSize;
    const int lane_id = tid % WaveSize;
    const int stage_wave = tid >> 5;
    const int stage_lane = tid & 31;
    const int sub_lane = lane_id & 15;
    const int half_id = lane_id >> 4;
    const int wave_row = wave_id / WN;
    const int wave_tok = wave_id % WN;

    // LG1 Change 1: row tiles are now the inner-loop grid dim.
    const size_t r_block = (size_t) blockIdx.x * (size_t) BM;

    float acc[kWaveRowTiles][kWaveTokTiles][kAcc];
#pragma unroll
    for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
        for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
            for (int l = 0; l < kAcc; ++l) {
                acc[i][j][l] = 0.0f;
            }
        }
    }

    __shared__ int32x4_t s_a[BK][kRowTiles][32];
    __shared__ float s_dw[BK][2][kRowTiles][16];
    __shared__ int32x4_t s_b[BK][kTokTiles][32];
    __shared__ float s_dx[BK][kTokTiles][16];

    // ---- persistent weight-row pointers (clamped, opt-c179) ----
    const uint8_t * w_row[kPrefetch];
    float row_live[kPrefetch];
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        const int idx = (p * kThreads) + tid;
        const bool unit_live = !kPartialFetch || idx < kFetchUnits;
        const int idx_c = unit_live ? idx : 0;
        const int r = (int) r_block + (idx_c / BK);
        const int r_c = (r < n) ? r : (n - 1);
        w_row[p] = Wg + (size_t) r_c * row_bytes;
        row_live[p] = (r < n && unit_live) ? 1.0f : 0.0f;
    }

    // ---- activation tile pointers for this block's token span ----
    // One 16-token tile per (block, k-block); a BN-block covers kTokTiles
    // consecutive group-local tiles starting at act_tile0.
    int b_tile[kStageTiles];
    const int8_t * b_base[kStageTiles];
#pragma unroll
    for (int t = 0; t < kStageTiles; ++t) {
        b_tile[t] = stage_wave + (t * kStageWaves);
        b_base[t] = (b_tile[t] < kTokTiles)
                        ? x_blocks + (size_t) (act_tile0 + b_tile[t]) *
                                          (size_t) num_kb * (size_t) kActTileBytes
                        : x_blocks;
    }

    int32x4_t r_q0[kPrefetch];
    int32x4_t r_q1[kPrefetch];
    float r_dw0[kPrefetch];
    float r_dw1[kPrefetch];
    int32x4_t r_b[kStageTiles][BK];
    float r_dx[kStageTiles][BK];

    const auto fetch_stage = [&](int kb0) {
#pragma unroll
        for (int p = 0; p < kPrefetch; ++p) {
            const int idx = (p * kThreads) + tid;
            const int idx_c = (!kPartialFetch || idx < kFetchUnits) ? idx : 0;
            const int kb = kb0 + (idx_c % BK);
            const int kb_c = (kb < num_kb) ? kb : (num_kb - 1);
            const float live = (kb < num_kb) ? row_live[p] : 0.0f;
            fp4_sub16(fast, w_row[p], kb_c * 2, r_q0[p], r_dw0[p]);
            fp4_sub16(fast, w_row[p], kb_c * 2 + 1, r_q1[p], r_dw1[p]);
            r_dw0[p] *= live;
            r_dw1[p] *= live;
        }
#pragma unroll
        for (int t = 0; t < kStageTiles; ++t) {
            if (b_tile[t] >= kTokTiles) {
                continue;
            }
#pragma unroll
            for (int i = 0; i < BK; ++i) {
                const int kb = kb0 + i;
                const int kb_c = (kb < num_kb) ? kb : (num_kb - 1);
                const int8_t * tile =
                    b_base[t] + (size_t) kb_c * (size_t) kActTileBytes;
                // 16 bytes per lane: lanes 0..15 read the k-half-0 tokens,
                // 16..31 the k-half-1 tokens (the 576-byte tile layout).
                r_b[t][i] = reinterpret_cast<const int32x4_t *>(tile)[stage_lane];
                r_dx[t][i] = ((kb < num_kb) ? 1.0f : 0.0f) *
                             reinterpret_cast<const float *>(
                                 tile + kActScaleOffset)[lane_id & 15];
            }
        }
    };

    const auto commit_stage = [&]() {
#pragma unroll
        for (int p = 0; p < kPrefetch; ++p) {
            const int idx = (p * kThreads) + tid;
            if constexpr (kPartialFetch) {
                if (idx >= kFetchUnits) {
                    continue;
                }
            }
            const int rr = idx / BK;
            const int kk = idx % BK;
            const int rs = rr / 16;
            const int rl = rr % 16;
            // w64 D-fragment lane ownership (gufo, verbatim)
            const int slot = (rl % 4) * 4 + rl / 4;
            s_a[kk][rs][rl] = r_q0[p];
            s_a[kk][rs][16 + rl] = r_q1[p];
            s_dw[kk][0][rs][slot] = r_dw0[p];
            s_dw[kk][1][rs][slot] = r_dw1[p];
        }
#pragma unroll
        for (int t = 0; t < kStageTiles; ++t) {
            if (b_tile[t] >= kTokTiles) {
                continue;
            }
#pragma unroll
            for (int i = 0; i < BK; ++i) {
                s_b[i][b_tile[t]][stage_lane] = r_b[t][i];
                if (stage_lane < 16) {
                    s_dx[i][b_tile[t]][stage_lane] = r_dx[t][i];
                }
            }
        }
    };

    fetch_stage(0);
    for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
        commit_stage();
        __syncthreads();
        if (kb0 + BK < num_kb) {
            fetch_stage(kb0 + BK);
        }

#pragma unroll
        for (int kb = 0; kb < BK; ++kb) {
            int32x4_t a0[kWaveRowTiles];
            int32x4_t a1[kWaveRowTiles];
            float dw0[kWaveRowTiles][kAcc];
            float dw1[kWaveRowTiles][kAcc];
#pragma unroll
            for (int i = 0; i < kWaveRowTiles; ++i) {
                const int rs = (wave_row * kWaveRowTiles) + i;
                a0[i] = s_a[kb][rs][sub_lane];
                a1[i] = s_a[kb][rs][16 + sub_lane];
#pragma unroll
                for (int l = 0; l < kAcc; ++l) {
                    dw0[i][l] = s_dw[kb][0][rs][(half_id * kAcc) + l];
                    dw1[i][l] = s_dw[kb][1][rs][(half_id * kAcc) + l];
                }
            }
#pragma unroll
            for (int j = 0; j < kWaveTokTiles; ++j) {
                const int ts = (wave_tok * kWaveTokTiles) + j;
                const int32x4_t b0 = s_b[kb][ts][sub_lane];
                const int32x4_t b1 = s_b[kb][ts][16 + sub_lane];
                const float dx = s_dx[kb][ts][sub_lane];
#pragma unroll
                for (int i = 0; i < kWaveRowTiles; ++i) {
                    int32x4_t c0{};
                    int32x4_t c1{};
                    c0 = wmma_w64(a0[i], b0, c0);
                    c1 = wmma_w64(a1[i], b1, c1);
#pragma unroll
                    for (int l = 0; l < kAcc; ++l) {
                        // fork semantics per 32-block:
                        //   db * (e0 * sumi0 + e1 * sumi1)
                        acc[i][j][l] += (dw0[i][l] * dx) * (float) c0[l] +
                                        (dw1[i][l] * dx) * (float) c1[l];
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---- epilogue: straight from the accumulator registers ----
    // Lane (sub_lane, half_id) of accumulator element l owns
    // C(row = (WaveSize/16)*l + half_id, tok = sub_lane) of its wave's
    // 16x16 tile — the ownership gufo's scratch dance implements, written
    // here directly because y is slot/token-major already.
#pragma unroll
    for (int i = 0; i < kWaveRowTiles; ++i) {
        const int rs = (wave_row * kWaveRowTiles) + i;
#pragma unroll
        for (int j = 0; j < kWaveTokTiles; ++j) {
            const int ts = (wave_tok * kWaveTokTiles) + j;
#pragma unroll
            for (int l = 0; l < kAcc; ++l) {
                const int row = (int) r_block + rs * 16 +
                               ((WaveSize / 16) * l) + half_id;
                const int tok = t_local + ts * 16 + sub_lane;
                if (row < n && tok < cnt) {
                    const size_t jrow = (size_t) (off + tok);
                    const float sc = y_scale != nullptr ? y_scale[jrow] : 1.0f;
                    y[jrow * (size_t) n + (size_t) row] = sc * acc[i][j][l];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// on-device exhaustive codebook / scale check
// ---------------------------------------------------------------------------
// Every 16-bit four-nibble pattern, both halves, both block types, both
// blocks of a 2-block row: byte i of qs = pattern nibble i&3 replicated into
// both nibbles (so the half-0 and half-1 decodes see the same pattern), and
// the two scale bytes differ so the e[0]/e[1] split is checked too.
__global__ void k_codebook_check(int * bad) {
    const int stride = (int) (blockDim.x * gridDim.x);
    for (int w = (int) threadIdx.x + (int) blockIdx.x * (int) blockDim.x;
         w < 65536; w += stride) {
        for (int fast = 0; fast < 2; ++fast) {
            const int tsz = fast ? 17 : 18;
            for (int b = 0; b < 2; ++b) {
                uint8_t row[40];
                for (int i = 0; i < 16; ++i) {
                    const uint32_t nib = (w >> (4 * (i & 3))) & 0xfu;
                    row[b * tsz + i] = (uint8_t) (nib | (nib << 4));
                }
                row[b * tsz + 16] = 0x5au;
                row[b * tsz + 17] = 0x23u;
                for (int half = 0; half < 2; ++half) {
                    int32x4_t q{};
                    float sc;
                    fp4_sub16(fast, row, b * 2 + half, q, sc);
                    uint32_t wq[4];
                    __builtin_memcpy(wq, &q, 16);
                    for (int i = 0; i < 4; ++i) {
                        for (int byi = 0; byi < 4; ++byi) {
                            const uint8_t byte =
                                (uint8_t) ((wq[i] >> (8 * byi)) & 0xff);
                            const uint32_t nib = (w >> (4 * byi)) & 0xfu;
                            const int8_t got = (int8_t) byte;
                            const int8_t want =
                                (int8_t) ling3_rocmfp4_code(nib);
                            if (got != want) {
                                atomicAdd(bad, 1);
                            }
                        }
                    }
                    const float want_sc = ling3_ue4m3_half(
                        row[b * tsz + 16 + ((half != 0 && !fast) ? 1 : 0)]);
                    if (sc != want_sc) {
                        atomicAdd(bad, 1);
                    }
                }
            }
        }
    }
}

}  // namespace ls3w

// ---------------------------------------------------------------------------
// host entry points
// ---------------------------------------------------------------------------
static int ls3w_chk(const char * what) {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        fprintf(stderr, "ling3_wmma %s: %s\n", what, hipGetErrorString(e));
        return LING3_ERR_HIP;
    }
    return LING3_OK;
}

static int ls3w_args(int qtype, int k, int n, int m) {
    if (qtype != LING3_Q4_0_ROCMFP4 && qtype != LING3_Q4_0_ROCMFP4_FAST) {
        return LING3_ERR_TYPE;
    }
    if (k <= 0 || (k % 32) != 0 || n <= 0 || m <= 0) {
        return LING3_ERR_ARG;
    }
    return LING3_OK;
}

extern "C" int ling3_wmma_quant(const float * x, const int * x_rows, const int * dest,
                               void * q8, int n_rows, int k) {
    if (x == nullptr || q8 == nullptr || n_rows <= 0 || k <= 0 || (k % 32) != 0) {
        return LING3_ERR_ARG;
    }
    const int num_blocks = k / 32;
    const long long total_warps = (long long) n_rows * num_blocks;
    const int grid = (int) ((total_warps + 7) / 8);
    if (grid <= 0) {
        return LING3_ERR_ARG;
    }
    ls3w::k_q8_quant<<<(unsigned) grid, 256>>>(x, x_rows, dest,
                                               (int8_t *) q8, n_rows, num_blocks);
    if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ls3w_chk("quant");
}

// LS3 PHASE8 (RV2 Change 2): raise BK for the grouped MoE expert GEMM,
// behind LING3_MOE_V2 (0 = int27 BK=2, the default; 1 = BK=8; 4 = BK=4).
// Fewer K-stage turnarounds (fetch + commit + two __syncthreads per stage:
// 40 -> 20 stages at K=2560 with BK=4, -> 10 with BK=8) and kPartialFetch
// goes false (BM*BK is a multiple of 256), so no thread runs the dead-lane
// fp4_sub16 decode. The inner kb loop stays ascending over the same kb
// sequence. Stacks on LG1 Change 1 (the transposed grid). MEASURED on
// MAX-1 (REPORT8): BK=8 REGRESSES (+52 ms/chunk vs BK=2 -- VGPR 108 -> 208
// and LDS 6,656 -> 25,088 B cost more occupancy than the barriers save);
// BK=4 wins (-27..-32 ms/chunk vs BK=2, -26..-28 % on the moe section vs
// int27). NOT bit-identical to BK=2: -ffast-math reassociates the float
// accumulator chain differently across the wider unrolled loop; the unit
// gate (wmma_test --dump) shows all cases PASS at err/rms ~1e-6 vs the
// exact fork-semantics oracle, and the wave-2 numerics gates decide.
static int MoeV2(void) {
    // 0 = BK=2 (int27), 1 = BK=8, 4 = BK=4 (RV2's named fallback).
    static const int v = [] {
        const char * e = std::getenv("LING3_MOE_V2");
        if (e == nullptr || e[0] == 0 || e[0] == 48) return 0;
        const long x = std::strtol(e, nullptr, 10);
        return x == 4 ? 4 : 1;
    }();
    return v;
}

extern "C" int ling3_wmma_gemm_grp(int qtype, const void * W, size_t expert_stride,
                                   const int * grp_expert, const int * grp_off,
                                   const int * grp_cnt, const int * grp_zp,
                                   int n_groups, int total_tiles,
                                   const void * q8_x, float * y,
                                   const float * y_scale, int k, int n) {
    int rc = ls3w_args(qtype, k, n, /*m=*/1);
    if (rc != LING3_OK) {
        return rc;
    }
    if (W == nullptr || grp_expert == nullptr || grp_off == nullptr ||
        grp_cnt == nullptr || grp_zp == nullptr || q8_x == nullptr || y == nullptr ||
        n_groups <= 0 || total_tiles <= 0) {
        return LING3_ERR_ARG;
    }
    const size_t rb = ling3_row_bytes(qtype, k);
    if (rb == 0) {
        return LING3_ERR_TYPE;
    }
    // LG1 Change 1: transpose grid so consecutive blocks share one expert y-slice.
    // BM = 64: grid.x covers n rows; grid.y is exactly the per-group tiles.
    const int row_tiles = (n + 63) / 64;
    const dim3 grid((unsigned) row_tiles, (unsigned) total_tiles);
    if (MoeV2() == 4) {
        ls3w::k_wmma_grp<64, 16, 4, 4, 1, 64><<<grid, 256>>>(
            qtype == LING3_Q4_0_ROCMFP4_FAST ? 1 : 0, (const uint8_t *) W, rb,
            expert_stride, grp_expert, grp_off, grp_cnt, grp_zp, n_groups,
            /*m=*/0, (const int8_t *) q8_x, y, y_scale, k, n);
        if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    } else if (MoeV2() == 1) {
        ls3w::k_wmma_grp<64, 16, 8, 4, 1, 64><<<grid, 256>>>(
            qtype == LING3_Q4_0_ROCMFP4_FAST ? 1 : 0, (const uint8_t *) W, rb,
            expert_stride, grp_expert, grp_off, grp_cnt, grp_zp, n_groups,
            /*m=*/0, (const int8_t *) q8_x, y, y_scale, k, n);
        if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    } else {
        ls3w::k_wmma_grp<64, 16, 2, 4, 1, 64><<<grid, 256>>>(
            qtype == LING3_Q4_0_ROCMFP4_FAST ? 1 : 0, (const uint8_t *) W, rb,
            expert_stride, grp_expert, grp_off, grp_cnt, grp_zp, n_groups,
            /*m=*/0, (const int8_t *) q8_x, y, y_scale, k, n);
        if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    }
    return ls3w_chk("gemm_grp");
}

extern "C" int ling3_wmma_gemm_dense(int qtype, const void * W, const void * q8_x,
                                     float * y, int k, int n, int m) {
    int rc = ls3w_args(qtype, k, n, m);
    if (rc != LING3_OK) {
        return rc;
    }
    if (W == nullptr || q8_x == nullptr || y == nullptr) {
        return LING3_ERR_ARG;
    }
    const size_t rb = ling3_row_bytes(qtype, k);
    if (rb == 0) {
        return LING3_ERR_TYPE;
    }
    const int tok_tiles = (m + 63) / 64;   // BN = 64
    const int row_tiles = (n + 127) / 128; // BM = 128
    // LG1 Change 1: transpose dim3 so the inner-loop grid dim carries the row tile.
    const dim3 grid((unsigned) row_tiles, (unsigned) tok_tiles);
    ls3w::k_wmma_grp<128, 64, 2, 4, 1, 64><<<grid, 256>>>(
        qtype == LING3_Q4_0_ROCMFP4_FAST ? 1 : 0, (const uint8_t *) W, rb,
        /*expert_stride=*/0, /*grp_expert=*/nullptr, /*grp_off=*/nullptr,
        /*grp_cnt=*/nullptr, /*grp_zp=*/nullptr, /*n_groups=*/0, m,
        (const int8_t *) q8_x, y, /*y_scale=*/nullptr, k, n);
    if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ls3w_chk("gemm_dense");
}

extern "C" int ling3_wmma_codebook_check(void) {
    static const int kZero = 0;
    int * d_bad = nullptr;
    if (hipMalloc(&d_bad, sizeof(int)) != hipSuccess) {
        return LING3_ERR_HIP;
    }
    if (hipMemcpy(d_bad, &kZero, sizeof(int), hipMemcpyHostToDevice) !=
        hipSuccess) {
        hipFree(d_bad);
        return LING3_ERR_HIP;
    }
    ls3w::k_codebook_check<<<64, 256>>>(d_bad);
    if (::gufo::hip_check::LaunchFailed()) return -1;
    int bad = -1;
    const hipError_t e =
        hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost);
    hipFree(d_bad);
    if (e != hipSuccess) {
        return LING3_ERR_HIP;
    }
    return bad == 0 ? LING3_OK : LING3_ERR_TYPE;
}

const char * ling3_wmma_version(void) {
    return MoeV2() == 4 ? "ling3-wmma/1.2-ls3p8-bk4"
           : MoeV2() == 1 ? "ling3-wmma/1.1-ls3p8-bk8"
                  : "ling3-wmma/1.0-ls3p3";
}
