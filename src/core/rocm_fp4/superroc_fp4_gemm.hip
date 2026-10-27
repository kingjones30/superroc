// superroc_fp4_gemm.hip — batched-prefill GEMM + batched MoE router for Ling-3.0-flash.
// Lane LS3 (GLM 5.3), Ling speed wave, 2026-10-01. See superroc_fp4_gemm.hpp for the
// contracts. Nothing in LX's superroc_fp4_moe.hip / LK's kernel files is modified.
//
// The GEMM (superroc_fp4_k_gemm_grp) is a tiled kernel, not one-thread-per-row:
//   * one block = [64 weight rows x 32 tokens x 64 k per tile], 256 threads;
//     each thread owns 4 rows x 2 tokens = 8 accumulators, so every weight
//     element dequantized into LDS is reused by 32 tokens.
//   * the weight tile is dequantized once into LDS per (block, k-tile) with the
//     SAME per-element decode as superroc_fp4_elem for the FP4 types (scale hoisted
//     per 32-block, exactly h1_dot_fp4_blk32's decoded values) and via
//     superroc_fp4_elem itself for every other type.
//   * X is read straight from global: a thread needs only its own two token
//     rows, and the 16 threads sharing a token hit L1 together.
//   * per-output accumulation is k-ascending, one += per element, with fp
//     contraction off — the same statement order as superroc_fp4_dot_row, so the
//     values match the CPU reference up to the mul/add rounding it also has.
//   * out-of-range rows / tokens / k-tails are zero-padded in the tile and
//     masked in the epilogue; a block whose whole token tile is past the group
//     count returns before touching memory.
//
// Bounds discipline (this lane faults no GPUs):
//   * W reads: row < n and k0+kk < k are checked BEFORE the pointer is formed
//     (blk stays null otherwise); the byte reads inside are then in-row only.
//   * X reads: float4 at x + row*k + k0 + kk4*4 with k % 4 == 0 (checked on the
//     host) and kk4 clamped so k0+kk4*4+4 <= k — never past the row.
//   * Y writes: y + yrow*n + row with row < n and the token inside its group.

#include "superroc_fp4_gemm.hpp"

#include "superroc_fp4_moe_group.hpp"
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "src/core/env_compat.hpp"
namespace superroc_fp4_ks {

static int ls3_hip(hipError_t e, const char * what) {
    if (e == hipSuccess) {
        return SUPERROC_FP4_OK;
    }
    fprintf(stderr, "superroc_fp4_gemm %s: %s\n", what, hipGetErrorString(e));
    return SUPERROC_FP4_ERR_HIP;
}

// Same policy as superroc_fp4_moe.hip's superroc_fp4_chk: the NULL stream orders every
// launch here against each other and against hipMemcpy, so check the error
// state without forcing a host barrier.
static int ls3_chk(const char * what) {
    return ls3_hip(hipGetLastError(), what);
}

// ---------------------------------------------------------------------------
// tiled grouped GEMM
// ---------------------------------------------------------------------------
#define SUPERROC_FP4_LS3_TR 64        // weight rows per block
#define SUPERROC_FP4_LS3_TC_DENSE 64  // P2: dense path token tile (4 rows x 4 tokens per
                         // thread -- the u32-staged kernel measured
                         // LDS-read-bound; doubling the token span halved
                         // kda_proj).
// P2 rev 2: the grouped tile drops 32 -> 16. Real routing gives ~312 groups
// per layer per 512-token chunk with a ~13-slot average, so a 32-wide token
// tile idles ~60% of the lanes in the accumulate phase (the W-tile dequant
// is fixed work per block either way). 16 fills the small groups; the rare
// fat group (cap 64) re-reads its weight stack 4x instead of 2x -- a few
// such groups per chunk, negligible bytes against the win.
#define SUPERROC_FP4_LS3_TC_GRP 16
#define SUPERROC_FP4_LS3_BK 64        // k per tile
#define SUPERROC_FP4_LS3_WSTRIDE 68   // LDS row stride (floats): 16B-multiple for float4

template <int SUPERROC_FP4_LS3_TC>
__global__ __launch_bounds__(256) void superroc_fp4_k_gemm_grp(
        int qtype, const uint8_t * __restrict__ W, size_t row_bytes, size_t expert_stride,
        const int * __restrict__ grp_expert, const int * __restrict__ grp_off,
        const int * __restrict__ grp_cnt, const int * __restrict__ grp_zp,
        int n_groups_z,
        const float * __restrict__ x, const int * __restrict__ x_rows,
        float * __restrict__ y, const int * __restrict__ y_rows,
        const float * __restrict__ y_scale,
        int accumulate, int k, int n, int m) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    // P3 step 1 (MoE-aware grid): with grp_zp the token-tile axis is
    // FLATTENED into blockIdx.x through the per-group 16-token-tile prefix,
    // so a group of cnt tokens gets exactly ceil(cnt/SUPERROC_FP4_LS3_TC) blocks instead
    // of the worst-case grid every group paid before. Per-element work,
    // ownership and accumulation order are unchanged -> bit-identical.
    // grp_zp == nullptr keeps the legacy 3-D grid (unit-gate A/B).
    int g;
    int c0;
    if (grp_zp != nullptr) {
        int lo = 0;
        int hi = n_groups_z;
        const int bflat = (int) blockIdx.x;
        while (lo + 1 < hi) {
            const int mid = (lo + hi) >> 1;
            if (grp_zp[mid] <= bflat) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        g = lo;
        c0 = (bflat - grp_zp[g]) * SUPERROC_FP4_LS3_TC;
    } else {
        g = blockIdx.x;
        c0 = blockIdx.z * SUPERROC_FP4_LS3_TC;
    }
    const int expert = grp_expert != nullptr ? grp_expert[g] : 0;
    const int off = grp_off != nullptr ? grp_off[g] : 0;
    const int cnt = grp_cnt != nullptr ? grp_cnt[g] : m;
    if (c0 >= cnt) {
        return;   // block-uniform: the whole token tile is past this group
    }
    const uint8_t * Wg = W + (size_t) expert * expert_stride;
    constexpr int NTOK = SUPERROC_FP4_LS3_TC / 16;   // tokens per thread: 4 (dense) / 2 (grouped)
    const int r0 = blockIdx.y * SUPERROC_FP4_LS3_TR;
    const int ty = (int) threadIdx.x >> 4;   // 0..15 -> rows r0 + ty*4 .. +3
    const int tx = (int) threadIdx.x & 15;   // 0..15 -> tokens c0 + tx*NTOK ..
    const int rlo = r0 + ty * 4;

    // This thread's NTOK token rows in x (-1 = the slot is past the group).
    int xr[NTOK];
    const float * x4p[NTOK];
#pragma unroll
    for (int s = 0; s < NTOK; ++s) {
        const int c = c0 + tx * NTOK + s;
        if (c < cnt) {
            const int j = off + c;
            xr[s] = x_rows != nullptr ? x_rows[j] : c;
        } else {
            xr[s] = -1;
        }
        x4p[s] = xr[s] >= 0 ? x + (size_t) xr[s] * (size_t) k : nullptr;
    }

    float acc[4 * NTOK];  // acc[i*NTOK+s]: row rlo+i, token c0+tx*NTOK+s
#pragma unroll
    for (int a = 0; a < 4 * NTOK; ++a) {
        acc[a] = 0.f;
    }
    // Small-group path (LS3 perf): a MoE chunk of 512 tokens routes ~8 tokens
    // per expert, and the 4-row x 2-token mapping leaves 3/4 of the block
    // idle (tokens past cnt). For cnt <= 8 every thread instead takes ONE
    // token and two rows, so all 256 threads work. cnt is block-uniform, so
    // the two paths never diverge within a block.
    const bool small = cnt <= 8;
    const int s_rlo = r0 + ((int) threadIdx.x >> 3) * 2;   // 32 x 2 rows = 64
    const int s_tok = c0 + ((int) threadIdx.x & 7);
    const float * s_x = (small && s_tok < cnt)
                            ? x + (size_t) (x_rows != nullptr ? x_rows[off + s_tok]
                                                              : s_tok) * (size_t) k
                            : nullptr;

    __shared__ float Wt[SUPERROC_FP4_LS3_TR][SUPERROC_FP4_LS3_WSTRIDE];
    // P2-B': raw staging for the vectorized FP4 loader. Covers one k-tile's
    // two blocks per row (34/36 B) plus alignment slack.
    __shared__ uint32_t Wraw[SUPERROC_FP4_LS3_TR][12];

    const int tsz = superroc_fp4_type_size(qtype);
    const bool fp4 = (qtype == SUPERROC_FP4_Q4_0_ROCMFP4 || qtype == SUPERROC_FP4_Q4_0_ROCMFP4_FAST);
    // The u32 staging needs 4-aligned row starts, i.e. row_bytes % 4 == 0
    // (true for the model's shapes: 1360 B and 408 B rows). Small test shapes
    // with odd row_bytes keep the byte-load path.
    const bool vec_stage = fp4 && (row_bytes % 4) == 0;

    for (int k0 = 0; k0 < k; k0 += SUPERROC_FP4_LS3_BK) {
        // ---- load + dequant the [TR][BK] weight tile (zeros outside n / k) ----
        if (vec_stage) {
            const int kb0 = (k0 / 32) * tsz;               // first block's byte offset
            const int kb1 = kb0 + 2 * tsz;                 // past the last byte needed
            const int wbase = kb0 & ~3;                    // 4-aligned window start
            const int wu32 = (kb1 + 3 - wbase + 3) / 4;    // u32s covering it (<= 11)
            for (int e = (int) threadIdx.x; e < SUPERROC_FP4_LS3_TR * wu32; e += 256) {
                const int rl = e / wu32;
                const int u = e % wu32;
                const int rr2 = r0 + rl;
                uint32_t v = 0;
                if (rr2 < n && k0 < k) {
                    // rr2*row_bytes + wbase + 4u is 4-aligned: row starts and
                    // wbase are; the last u32 may touch up to 3 bytes of the
                    // next row, which is inside the allocation (TAIL_PAD on
                    // the very last row).
                    v = *(const uint32_t *) (Wg + (size_t) rr2 * row_bytes +
                                             (size_t) wbase + (size_t) (4 * u));
                }
                Wraw[rl][u] = v;
            }
            __syncthreads();
            const auto raw_byte = [&](int rl, int b) -> unsigned {
                const int o = b - wbase;
                return (Wraw[rl][o >> 2] >> (8 * (o & 3))) & 0xffu;
            };
            // 256 threads = 2 per (row, 32-block); same decoded values as
            // superroc_fp4_elem (code(nib) * scale, scales at bytes 16 / 17).
            const int b = (int) threadIdx.x >> 1;
            const int h = (int) threadIdx.x & 1;
            const int rr = r0 + (b >> 1);
            const int bi = b & 1;
            const int kb = kb0 + bi * tsz;
            const bool has = rr < n && (k0 + bi * 32) < k;
            const bool fast = (qtype == SUPERROC_FP4_Q4_0_ROCMFP4_FAST);
            const float slo =
                has ? superroc_fp4_ue4m3_half((uint8_t) raw_byte(b >> 1, kb + 16)) : 0.f;
            const float shi =
                has ? (fast ? slo
                            : superroc_fp4_ue4m3_half((uint8_t) raw_byte(b >> 1, kb + 17)))
                    : 0.f;
            for (int j = 0; j < 16; ++j) {
                const int loc = h * 16 + j;
                const int kk = bi * 32 + loc;
                float v = 0.f;
                if (has && k0 + kk < k) {
                    const uint8_t byte = (uint8_t) raw_byte(b >> 1, kb + (loc & 15));
                    const int nib = (loc < 16) ? (int) (byte & 0x0f) : (int) (byte >> 4);
                    v = (float) superroc_fp4_rocmfp4_code((uint32_t) nib) * (loc < 16 ? slo : shi);
                }
                Wt[b >> 1][kk] = v;
            }
        } else if (fp4) {
            // 256 threads = 2 per (row, 32-block): thread t -> block b = t>>1,
            // half h = t&1 (16 of the 32 elements). Same decoded values as
            // superroc_fp4_elem: code(nib) * scale, scales at bytes 16 / (17).
            // The block's byte offset is (k0/32 + bi) * tsz: the k-tile's own
            // blocks come first (the first version forgot k0/32 and every
            // shape with more than one k-tile read the wrong blocks).
            const int b = (int) threadIdx.x >> 1;
            const int h = (int) threadIdx.x & 1;
            const int rr = r0 + (b >> 1);
            const int bi = b & 1;
            const uint8_t * blk =
                (rr < n && k0 + bi * 32 < k)
                    ? Wg + (size_t) rr * row_bytes +
                          ((size_t) (k0 / 32) + (size_t) bi) * (size_t) tsz
                    : nullptr;
            const bool fast = (qtype == SUPERROC_FP4_Q4_0_ROCMFP4_FAST);
            const float slo = blk != nullptr ? superroc_fp4_ue4m3_half(superroc_fp4_ld8(blk + 16)) : 0.f;
            const float shi = blk != nullptr
                                  ? (fast ? slo : superroc_fp4_ue4m3_half(superroc_fp4_ld8(blk + 17)))
                                  : 0.f;
            for (int j = 0; j < 16; ++j) {
                const int loc = h * 16 + j;
                const int kk = bi * 32 + loc;
                float v = 0.f;
                if (blk != nullptr && k0 + kk < k) {
                    const uint8_t byte = superroc_fp4_ld8(blk + (loc & 15));
                    const int nib = (loc < 16) ? (int) (byte & 0x0f) : (int) (byte >> 4);
                    v = (float) superroc_fp4_rocmfp4_code((uint32_t) nib) * (loc < 16 ? slo : shi);
                }
                Wt[b >> 1][kk] = v;
            }
        } else {
            for (int e = (int) threadIdx.x; e < SUPERROC_FP4_LS3_TR * SUPERROC_FP4_LS3_BK; e += 256) {
                const int rl = e / SUPERROC_FP4_LS3_BK;
                const int kk = e % SUPERROC_FP4_LS3_BK;
                const int rr = r0 + rl;
                float v = 0.f;
                if (rr < n && k0 + kk < k) {
                    v = superroc_fp4_elem(qtype, Wg + (size_t) rr * row_bytes, k0 + kk);
                }
                Wt[rl][kk] = v;
            }
        }
        __syncthreads();

        // ---- accumulate; kk4 clamped so every float4 stays inside the row ----
        const int kk4n = (k - k0) / 4 < SUPERROC_FP4_LS3_BK / 4 ? (k - k0) / 4 : SUPERROC_FP4_LS3_BK / 4;
        if (small) {
            const float4 * sx4 =
                s_x != nullptr ? reinterpret_cast<const float4 *>(s_x + k0) : nullptr;
            for (int kk4 = 0; kk4 < kk4n; ++kk4) {
                const float4 xv = sx4 != nullptr ? sx4[kk4]
                                                 : make_float4(0.f, 0.f, 0.f, 0.f);
                const float4 wv0 =
                    *reinterpret_cast<const float4 *>(&Wt[((int) threadIdx.x >> 3) * 2 + 0]
                                                       [kk4 * 4]);
                const float4 wv1 =
                    *reinterpret_cast<const float4 *>(&Wt[((int) threadIdx.x >> 3) * 2 + 1]
                                                       [kk4 * 4]);
                acc[0] += wv0.x * xv.x;
                acc[0] += wv0.y * xv.y;
                acc[0] += wv0.z * xv.z;
                acc[0] += wv0.w * xv.w;
                acc[1] += wv1.x * xv.x;
                acc[1] += wv1.y * xv.y;
                acc[1] += wv1.z * xv.z;
                acc[1] += wv1.w * xv.w;
            }
        } else {
            const float4 * xs4[NTOK];
#pragma unroll
            for (int s = 0; s < NTOK; ++s) {
                xs4[s] = x4p[s] != nullptr ? reinterpret_cast<const float4 *>(x4p[s] + k0)
                                           : nullptr;
            }
            for (int kk4 = 0; kk4 < kk4n; ++kk4) {
                float4 xv[NTOK];
#pragma unroll
                for (int s = 0; s < NTOK; ++s) {
                    xv[s] = xs4[s] != nullptr ? xs4[s][kk4]
                                              : make_float4(0.f, 0.f, 0.f, 0.f);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 wv =
                        *reinterpret_cast<const float4 *>(&Wt[ty * 4 + i][kk4 * 4]);
#pragma unroll
                    for (int s = 0; s < NTOK; ++s) {
                        acc[i * NTOK + s] += wv.x * xv[s].x;
                        acc[i * NTOK + s] += wv.y * xv[s].y;
                        acc[i * NTOK + s] += wv.z * xv[s].z;
                        acc[i * NTOK + s] += wv.w * xv[s].w;
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---- epilogue ----
    // Each (row, slot) pair is owned by exactly one block, but with a slot->token
    // scatter (the expert down-projection) several slots of the SAME token land
    // on one y element from different blocks, so the accumulate path has to be
    // an atomicAdd. The plain first version raced there (the grouped unit gate
    // caught it); the store path stays a plain write.
    if (small) {
        if (s_tok < cnt) {
            const int j = off + s_tok;
            const size_t yrow =
                (size_t) (y_rows != nullptr ? y_rows[j] : s_tok);
            const float sc = y_scale != nullptr ? y_scale[j] : 1.0f;
            for (int i = 0; i < 2; ++i) {
                const int row = s_rlo + i;
                if (row < n) {
                    float * dst = y + yrow * (size_t) n + (size_t) row;
                    const float v = sc * acc[i];
                    if (accumulate) {
                        atomicAdd(dst, v);
                    } else {
                        dst[0] = v;
                    }
                }
            }
        }
        return;
    }
#pragma unroll
    for (int s = 0; s < NTOK; ++s) {
        const int c = c0 + tx * NTOK + s;
        if (c >= cnt) {
            continue;
        }
        const int j = off + c;
        const size_t yrow = (size_t) (y_rows != nullptr ? y_rows[j] : c);
        const float sc = y_scale != nullptr ? y_scale[j] : 1.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = rlo + i;
            if (row >= n) {
                continue;
            }
            float * dst = y + yrow * (size_t) n + (size_t) row;
            const float v = sc * acc[i * NTOK + s];
            if (accumulate) {
                atomicAdd(dst, v);
            } else {
                dst[0] = v;
            }
        }
    }
}

extern "C" int superroc_fp4_gemm_grp(int qtype, const void * W, size_t expert_stride,
                              const int * grp_expert, const int * grp_off,
                              const int * grp_cnt, int n_groups,
                              const float * x, const int * x_rows,
                              float * y, const int * y_rows, const float * y_scale,
                              int accumulate, int k, int n, int m,
                              const int * grp_zp, int total_tiles) {
    const size_t rb = superroc_fp4_row_bytes(qtype, k);
    if (W == nullptr || x == nullptr || y == nullptr || rb == 0 || n <= 0 || m <= 0 ||
        n_groups <= 0) {
        return SUPERROC_FP4_ERR_ARG;
    }
    if ((k % 4) != 0) {
        return SUPERROC_FP4_ERR_ARG;   // X is read as float4; every real shape is a multiple of 32
    }
    if (grp_zp != nullptr && total_tiles <= 0) {
        return SUPERROC_FP4_ERR_ARG;
    }
    if (grp_zp != nullptr) {
        // compacted: one block per (group token tile, row tile) that exists
        const dim3 grid((unsigned) total_tiles,
                        (unsigned) ((n + SUPERROC_FP4_LS3_TR - 1) / SUPERROC_FP4_LS3_TR));
        superroc_fp4_k_gemm_grp<SUPERROC_FP4_LS3_TC_GRP><<<grid, 256>>>(
            qtype, (const uint8_t *) W, rb, expert_stride, grp_expert, grp_off,
            grp_cnt, grp_zp, n_groups, x, x_rows, y, y_rows, y_scale, accumulate,
            k, n, m);
    } else {
        const dim3 grid((unsigned) n_groups,
                        (unsigned) ((n + SUPERROC_FP4_LS3_TR - 1) / SUPERROC_FP4_LS3_TR),
                        (unsigned) ((m + SUPERROC_FP4_LS3_TC_GRP - 1) / SUPERROC_FP4_LS3_TC_GRP));
        superroc_fp4_k_gemm_grp<SUPERROC_FP4_LS3_TC_GRP><<<grid, 256>>>(
            qtype, (const uint8_t *) W, rb, expert_stride, grp_expert, grp_off,
            grp_cnt, nullptr, 0, x, x_rows, y, y_rows, y_scale, accumulate,
            k, n, m);
    }
    return ls3_chk("gemm_grp");
}

extern "C" int superroc_fp4_gemm_fast(int qtype, const void * W, const float * x, float * y,
                               int k, int n, int m) {
    const size_t rb = superroc_fp4_row_bytes(qtype, k);
    if (W == nullptr || x == nullptr || y == nullptr || rb == 0 || n <= 0 || m <= 0) {
        return SUPERROC_FP4_ERR_ARG;
    }
    if ((k % 4) != 0) {
        return SUPERROC_FP4_ERR_ARG;
    }
    const dim3 grid(1, (unsigned) ((n + SUPERROC_FP4_LS3_TR - 1) / SUPERROC_FP4_LS3_TR),
                    (unsigned) ((m + SUPERROC_FP4_LS3_TC_DENSE - 1) / SUPERROC_FP4_LS3_TC_DENSE));
    superroc_fp4_k_gemm_grp<SUPERROC_FP4_LS3_TC_DENSE><<<grid, 256>>>(
        qtype, (const uint8_t *) W, rb, 0, nullptr, nullptr, nullptr, nullptr, 0,
        x, nullptr, y, nullptr, nullptr, 0, k, n, m);
    return ls3_chk("gemm_fast");
}

// ---------------------------------------------------------------------------
// P2-C: token-major gather-sum of slot-major routed-expert partials. One
// block per token, plain reads of its n_used slot rows -- no atomics, and
// the sum runs in slot order, the same order as superroc_fp4_expert_ffn_cpu.
// ---------------------------------------------------------------------------
__global__ void superroc_fp4_k_slot_gather_sum(const float * __restrict__ partials,
                                  const int * __restrict__ slot,
                                  float * __restrict__ y,
                                  int n_used, int k) {
    const size_t t = blockIdx.x;
    const int * ts = slot + t * (size_t) n_used;
    for (int r = (int) threadIdx.x; r < k; r += (int) blockDim.x) {
        float a = 0.f;
        for (int s = 0; s < n_used; ++s) {
            a += partials[(size_t) ts[s] * (size_t) k + r];
        }
        y[t * (size_t) k + r] = a;
    }
}

extern "C" int superroc_fp4_slot_gather_sum(const float * partials, const int * slot,
                                     float * y, int n_used, int k, int n_tokens) {
    if (partials == nullptr || slot == nullptr || y == nullptr || n_used <= 0 ||
        k <= 0 || n_tokens <= 0) {
        return SUPERROC_FP4_ERR_ARG;
    }
    superroc_fp4_k_slot_gather_sum<<<(unsigned) n_tokens, 256>>>(partials, slot, y, n_used, k);
    return ls3_chk("slot_gather_sum");
}

// ---------------------------------------------------------------------------
// batched router
//
// The per-token arithmetic is Jay/H1's k_route from superroc_fp4_moe.hip, copied
// verbatim (sigmoid+bias selection scores, group score = top-2 sum, group mask
// to -inf, bitonic DESC argsort with the same padding and strict-compare rules,
// weights = unbiased sigmoids, norm then w_scale) with exactly three changes:
// blockIdx.x is the token, the logits come in precomputed at
// logits[t*n_expert + e] (the same handoff k_route already uses), and the
// ids/weights/rc outputs are per-token token-major. h1_argsort_desc_par is a
// verbatim copy of the one in superroc_fp4_moe.hip.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int ls3_argsort_desc_par(const float * x, int * dst, int * wa,
                                                    int ncols, int nt) {
    const int pad = superroc_fp4_pow2_ge(ncols);
    if (pad < 0 || !x || !dst || !wa) {
        return SUPERROC_FP4_ERR_ARG;
    }
    for (int i = (int) threadIdx.x; i < pad; i += nt) {
        wa[i] = i;
    }
    __syncthreads();
    for (int k = 2; k <= pad; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            __syncthreads();
            for (int t = (int) threadIdx.x; t < pad / 2; t += nt) {
                const int col = 2 * j * (t / j) + (t % j);
                const int ixj = col ^ j;
                const int a = wa[col];
                const int b = wa[ixj];
                int do_swap = 0;
                if ((col & k) == 0) {
                    if (a >= ncols || (b < ncols && x[a] < x[b])) {
                        do_swap = 1;
                    }
                } else {
                    if (b >= ncols || (a < ncols && x[a] > x[b])) {
                        do_swap = 1;
                    }
                }
                if (do_swap) {
                    wa[col] = b;
                    wa[ixj] = a;
                }
            }
        }
    }
    __syncthreads();
    for (int i = (int) threadIdx.x; i < ncols; i += nt) {
        dst[i] = wa[i];
    }
    return SUPERROC_FP4_OK;
}

__global__ void superroc_fp4_k_route_batch_logits(const float * __restrict__ x,
                                     const float * __restrict__ gate,
                                     int k, int n_expert,
                                     float * __restrict__ logits) {
    const int e = blockIdx.x;
    const int t = blockIdx.y;
    const float * row = gate + (size_t) e * (size_t) k;
    const float * xt = x + (size_t) t * (size_t) k;
    float s = 0.f;
    for (int i = (int) threadIdx.x; i < k; i += (int) blockDim.x) {
        s += row[i] * xt[i];
    }
    __shared__ float red[64];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int off = 32; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) {
            red[threadIdx.x] += red[threadIdx.x + off];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        logits[(size_t) t * (size_t) n_expert + e] = red[0];
    }
}

// LV1 R2: one CTA per expert, m dots. Each token's products are added in the
// same thread stride as superroc_fp4_k_route_batch_logits, then the same 64-wide tree.
// The gate element is loaded once and reused. n_tokens is 2..4 (SUPERROC_FP4_LS1_MAX_M).
__global__ void superroc_fp4_k_route_batch_logits_m(const float * __restrict__ x,
                                       const float * __restrict__ gate,
                                       int k, int n_expert, int n_tokens,
                                       float * __restrict__ logits) {
    const int e = (int) blockIdx.x;
    const float * row = gate + (size_t) e * (size_t) k;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    for (int i = (int) threadIdx.x; i < k; i += (int) blockDim.x) {
        const float w = row[i];
        s0 += w * x[(size_t) 0 * (size_t) k + (size_t) i];
        if (n_tokens > 1) s1 += w * x[(size_t) 1 * (size_t) k + (size_t) i];
        if (n_tokens > 2) s2 += w * x[(size_t) 2 * (size_t) k + (size_t) i];
        if (n_tokens > 3) s3 += w * x[(size_t) 3 * (size_t) k + (size_t) i];
    }
    __shared__ float red[64];
    float ss[4];
    ss[0] = s0; ss[1] = s1; ss[2] = s2; ss[3] = s3;
    for (int t = 0; t < n_tokens; ++t) {
        red[threadIdx.x] = ss[t];
        __syncthreads();
        for (int off = 32; off > 0; off >>= 1) {
            if ((int) threadIdx.x < off) {
                red[threadIdx.x] += red[threadIdx.x + off];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            logits[(size_t) t * (size_t) n_expert + e] = red[0];
        }
        __syncthreads();
    }
}

__global__ void superroc_fp4_k_route_batch(
        const float * __restrict__ x, const float * __restrict__ gate,
        const float * __restrict__ bias,
        int k, int n_expert, int n_used, int n_groups, int n_group_used,
        float w_scale, int norm_w,
        int * __restrict__ ids, float * __restrict__ weights,
        const float * __restrict__ logits_in, int * __restrict__ rc_out) {
    extern __shared__ char ls3_smem[];
    const int t = blockIdx.x;
    const int nt = (int) blockDim.x;
    const float * logits_t = logits_in + (size_t) t * (size_t) n_expert;
    int * ids_t = ids + (size_t) t * (size_t) n_used;
    float * w_t = weights + (size_t) t * (size_t) n_used;
    const int nf = superroc_fp4_route_nfloat(n_expert, n_groups);
    const int ni = superroc_fp4_route_nint(n_expert, n_groups);
    float * sf = (float *) ls3_smem;
    int * si = (int *) (sf + nf);
    if (nf < 0 || ni < 0 || n_expert <= 0 || n_expert > 1024 || n_used <= 0 ||
        n_groups <= 0 || (n_expert % n_groups) != 0) {
        if (threadIdx.x == 0) rc_out[t] = SUPERROC_FP4_ERR_ARG;
        return;
    }
    const int npg = n_expert / n_groups;
    const int pad_e = superroc_fp4_pow2_ge(n_expert);
    const int pad_g = superroc_fp4_pow2_ge(n_groups < 1 ? 1 : n_groups);
    float * probs = sf + n_expert;
    float * sel = sf + 2 * n_expert;
    float * gscore = sf + 3 * n_expert;
    int * ba = si;
    int * order = si + 2 * pad_e;
    int * ga = order + n_expert;
    int * gorder = ga + pad_g;

    for (int e = (int) threadIdx.x; e < n_expert; e += nt) {
        const float p = 1.0f / (1.0f + expf(-logits_t[e]));
        const float b = bias != nullptr ? bias[e] : 0.f;
        probs[e] = p;
        sel[e] = p + b;
    }
    __syncthreads();

    if (n_groups > 1) {
        for (int g = 0; g < n_groups; ++g) {
            (void) ls3_argsort_desc_par(sel + g * npg, order, ba, npg, nt);
            __syncthreads();
            if (threadIdx.x == 0) {
                gscore[g] = sel[g * npg + order[0]] + sel[g * npg + order[1]];
            }
            __syncthreads();
        }
        (void) ls3_argsort_desc_par(gscore, gorder, ga, n_groups, nt);
        __syncthreads();
        for (int g = (int) threadIdx.x; g < n_groups; g += nt) {
            int keep = 0;
            for (int i = 0; i < n_group_used; ++i) {
                if (gorder[i] == g) {
                    keep = 1;
                }
            }
            if (!keep) {
                for (int j = 0; j < npg; ++j) {
                    sel[g * npg + j] = superroc_fp4_neg_inf();
                }
            }
        }
        __syncthreads();
    }

    (void) ls3_argsort_desc_par(sel, order, ba, n_expert, nt);
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int i = 0; i < n_used; ++i) {
            ids_t[i] = order[i];
            w_t[i] = probs[ids_t[i]];
            sum += w_t[i];
        }
        if (norm_w) {
            const float den = sum < SUPERROC_FP4_WEIGHT_SUM_MIN ? SUPERROC_FP4_WEIGHT_SUM_MIN : sum;
            for (int i = 0; i < n_used; ++i) {
                w_t[i] = w_t[i] / den;
            }
        }
        if (w_scale != 0.f && w_scale != 1.f) {
            for (int i = 0; i < n_used; ++i) {
                w_t[i] *= w_scale;
            }
        }
        rc_out[t] = SUPERROC_FP4_OK;
    }
}

extern "C" int superroc_fp4_route_batch(const float * x, const float * gate, const float * bias,
                                 int k, int n_tokens, int n_expert, int n_used,
                                 int n_groups, int n_group_used, float w_scale, int norm_w,
                                 int * ids, float * weights, int * rc_out) {
    const int nf = superroc_fp4_route_nfloat(n_expert, n_groups);
    const int ni = superroc_fp4_route_nint(n_expert, n_groups);
    if (x == nullptr || gate == nullptr || ids == nullptr || weights == nullptr ||
        rc_out == nullptr || nf < 0 || ni < 0 || n_tokens <= 0 || k <= 0) {
        return SUPERROC_FP4_ERR_ARG;
    }
    // logits scratch, grown on demand (sizes depend only on the model).
    static float * logits_scratch = nullptr;
    static int cap = 0;
    int rc = SUPERROC_FP4_OK;
    if (n_tokens > cap || logits_scratch == nullptr) {
        if (logits_scratch != nullptr) {
            hipFree(logits_scratch);
        }
        logits_scratch = nullptr;
        rc = ls3_hip(hipMalloc(&logits_scratch,
                               (size_t) n_tokens * (size_t) n_expert * sizeof(float) +
                                   SUPERROC_FP4_TAIL_PAD),
                     "route_batch logits malloc");
        if (rc != SUPERROC_FP4_OK) {
            cap = 0;
            return rc;
        }
        cap = n_tokens;
    }
    // ---- LS1 PHASE9: the router logits as a real 2D register-tiled GEMM ----
    // MEASURED on halo, same binary, same box, 8 chunks of 512 tokens, only this switch
    // differing: `route` 148.5 ms -> 53.0 ms, 2.80x, -95.5 ms a chunk. At 2*E*T*K =
    // 1.07 GFLOP in 53.0 ms that is 20.3 TFLOP/s, about 80 % of this APU's fp32 FMA peak,
    // so the kernel is now compute-bound and sitting at the roofline rather than hiding
    // latency behind 262,144 tiny CTAs the way the kernel below does.
    //
    // Every WAVE 2 numerics gate passes (DECISIONS 2026-10-03 00:55 permits the order to
    // change; the gates, not bit-identity, decide): PPL long6k 4K scalar 9.2956 EXACT and
    // batched 9.2962 vs 9.3094 = 0.142 %; FPPL 4096-8191 batched 6.1037 and step 6.1134,
    // both EXACT; trace 298 OK / 5 flagged / ' Paris', EXACT; tooltest 67/67; no fault.
    //
    // IT IS NOT UNCONDITIONAL, and that is the whole point PHASE8 learned the expensive
    // way. The grid is ceil(E/BM) x ceil(T/BN) = 8 x ceil(T/32), so at T=1 it launches
    // 8 CTAs on a 20-CU part while the kernel below launches E = 512 -- and the profile
    // shows route going 4.1 -> 27.4 ms on a 1-token chunk. So the GEMM runs only at or
    // above SUPERROC_FP4_LS1_ROUTE_GEMM_MIN_T tokens and the shipped kernel runs unchanged below that.
    // SUPERROC_FP4_ROUTE_GEMM=0 forces the shipped kernel everywhere; SUPERROC_FP4_ROUTE_GEMM_MIN_T
    // overrides the threshold so the crossover can be re-measured without a rebuild.
    //
    // LS1 PHASE8's 1D token tiling (SUPERROC_FP4_ROUTE_LOGITS_TILED) is kept below, default off.
    // It is proven bit-identical (0 differing of 262,144) and NOT faster (~207 ms at a
    // tile of 8, ~148 at a tile of 2, against ~154 for the kernel below): REPORT8's
    // finding was that the router logits kernel is latency-bound, not bandwidth-bound,
    // and this GEMM is the shape that acts on it. See REPORT9.md.
    static const int gemm_min_t = [] {
        const char * e = ::superroc::core::GetEnvCompat("SUPERROC_FP4_ROUTE_GEMM_MIN_T");
        return (e != nullptr && e[0] != '\0') ? atoi(e) : (int) SUPERROC_FP4_LS1_ROUTE_GEMM_MIN_T;
    }();
    const char * gemm_env = ::superroc::core::GetEnvCompat("SUPERROC_FP4_ROUTE_GEMM");
    const char * tiled_env = ::superroc::core::GetEnvCompat("SUPERROC_FP4_ROUTE_LOGITS_TILED");
    const bool gemm_on = (gemm_env == nullptr || gemm_env[0] != '0') &&
                         n_tokens >= gemm_min_t;
    if (gemm_on) {
        rc = superroc_fp4_route_logits_gemm_launch(x, gate, k, n_expert, n_tokens,
                                            logits_scratch);
    } else if (tiled_env != nullptr && tiled_env[0] == '1') {
        rc = superroc_fp4_route_logits_tiled(x, gate, k, n_expert, n_tokens, SUPERROC_FP4_LS1_ROUTE_TY,
                                      logits_scratch);
    } else if (n_tokens > 1 && n_tokens <= 4) {
        // LV1 R2. n_tokens == 1 stays on the kernel below. Prefill chunks are
        // 512 and take the GEMM above (SUPERROC_FP4_LS1_ROUTE_GEMM_MIN_T is 256), so a
        // 4096-token PPL run never enters here.
        superroc_fp4_k_route_batch_logits_m<<<n_expert, 64>>>(x, gate, k, n_expert, n_tokens,
                                                 logits_scratch);
        rc = ls3_chk("route_batch logits m");
    } else {
        superroc_fp4_k_route_batch_logits<<<dim3(n_expert, n_tokens), 64>>>(x, gate, k, n_expert,
                                                               logits_scratch);
        rc = ls3_chk("route_batch logits");
    }
    if (rc == SUPERROC_FP4_OK) {
        const size_t smem =
            ((size_t) nf * sizeof(float) + (size_t) ni * sizeof(int) + 255) & ~(size_t) 255;
        superroc_fp4_k_route_batch<<<n_tokens, 256, smem>>>(x, gate, bias, k, n_expert, n_used,
                                               n_groups, n_group_used, w_scale, norm_w,
                                               ids, weights, logits_scratch, rc_out);
        rc = ls3_chk("route_batch");
    }
    return rc;
}

// Test-only: run the per-token logits kernel and the single-read one on the
// same device buffers. Caller synchronises and compares the two outputs.
extern "C" int superroc_fp4_route_logits_pair(const float * x, const float * gate,
                                       int k, int n_expert, int n_tokens,
                                       float * logits_ref, float * logits_fused) {
    if (x == nullptr || gate == nullptr || logits_ref == nullptr ||
        logits_fused == nullptr || k <= 0 || n_expert <= 0 || n_tokens < 1 ||
        n_tokens > 4) {
        return SUPERROC_FP4_ERR_ARG;
    }
    superroc_fp4_k_route_batch_logits<<<dim3(n_expert, n_tokens), 64>>>(x, gate, k, n_expert,
                                                           logits_ref);
    int rc = ls3_chk("route logits ref");
    if (rc != SUPERROC_FP4_OK) return rc;
    superroc_fp4_k_route_batch_logits_m<<<n_expert, 64>>>(x, gate, k, n_expert, n_tokens,
                                             logits_fused);
    return ls3_chk("route logits fused");
}

const char * superroc_fp4_gemm_version(void) {
    return "ling3-gemm/1.0-ls3";
}
}  // namespace superroc_fp4_ks
