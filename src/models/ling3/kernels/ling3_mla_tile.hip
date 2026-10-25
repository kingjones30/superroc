// ling3_mla_tile.hip — LF1 (MiMo), wave 2: gufo's absorbed-MLA PREFILL
// attention as the fork's TILE kernel. 2026-10-03. Header: ling3_mla_tile.hpp.
//
// ATTRIBUTION — adapted from llama.cpp's ggml/src/ggml-cuda/fattn-tile.cuh
// (flash_attn_tile<DKQ=576, DV=512, ncols1=2, ncols2=16>) and common.cuh
// (ggml_cuda_mad), MIT License, Copyright (c) 2023-2024 The ggml authors.
// The algorithm (register-blocked KQ/VKQ dot2 products over f16 K/V tiles in
// LDS, online softmax with one warp reduction per tile, no shuffles in the
// inner loop) is theirs; the absorbed-MLA shapes, the per-column causal limit,
// the MQA cache sharing and the de-absorb GEMM are gufo's (see the .hpp).
//
// Arithmetic, identical to mla_attn_tok up to f16 inputs and reordered sums:
//   q576 = [wk_b[h] . q_nope | q_pe] * (1/sqrt(192))     (512 + 64)
//   score = q576 . cache[p]                              (576 f16 dot2 MAC)
//   w     = softmax(score) over the causal window        (f32)
//   lat   = sum_p w[p] . cache[p][0:512]                 (512 f16 dot2 MAC)
// The de-absorb + gate are a separate batched GEMM kernel (tile_deabsorb).

#include "ling3_mla_tile.hpp"

#include <hip/hip_fp16.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define LT_HIP_CHECK(expr)                                                        \
    do {                                                                          \
        hipError_t _err = (expr);                                                 \
        if (_err != hipSuccess) {                                                 \
            std::fprintf(stderr, "ling3_mla_tile: %s failed at %s:%d: %s\n",      \
                         #expr, __FILE__, __LINE__, hipGetErrorString(_err));     \
            std::abort();                                                         \
        }                                                                         \
    } while (0)

namespace ling3 {

namespace {

constexpr int LT_KV   = 512;              // latent / value width
constexpr int LT_ROT  = 64;               // rope key width
constexpr int LT_KVA  = LT_KV + LT_ROT;   // 576 floats per cached position
constexpr int LT_NOPE = 128;
constexpr int LT_HV   = 128;

constexpr int LT_WARP   = 32;             // wave32, as the fork uses
constexpr int LT_NWARPS = 8;              // 256 threads
constexpr int LT_NTHREADS = LT_WARP * LT_NWARPS;
constexpr int LT_NCOLS1 = 2;              // tokens per block
constexpr int LT_NCOLS2 = 16;             // heads per token in the block
constexpr int LT_NCOLS  = LT_NCOLS1 * LT_NCOLS2;   // 32 query columns
constexpr int LT_BATCH_FA = 64;           // KV rows per iteration
constexpr int LT_BATCH_K  = 64;           // K dims per load
constexpr int LT_KSTEPS   = LT_KVA / LT_BATCH_K;   // 9

// Q columns per warp and parallel warps per column (the fork's cpw / np).
constexpr int LT_CPW = LT_NCOLS / LT_NWARPS;        // 4
constexpr int LT_NP  = 1;
// Rows of the tile held per lane, and float2 output fragments per column.
constexpr int LT_ROWS_PER_LANE = LT_BATCH_FA / (LT_NP * LT_WARP);   // 2
constexpr int LT_DV_FRAG = (LT_KV / 2) / LT_WARP;                   // 8

static_assert(LT_NCOLS % LT_NWARPS == 0, "columns must split across warps");
static_assert(LT_BATCH_FA % (LT_NP * LT_WARP) == 0, "rows per warp");
static_assert(LT_KVA % LT_BATCH_K == 0, "contraction must split");
static_assert(LT_KV % (2 * LT_WARP) == 0, "DV fragments");

// The fork's dot2 mad (common.cuh): one v_dot2_f32_f16 per half2 pair.
__device__ __forceinline__ void lt_mad(float & acc, const __half2 v,
                                       const __half2 u) {
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(v), "v"(u));
}

__device__ __forceinline__ float lt_warp_max32(float v) {
#pragma unroll
    for (int off = 1; off < LT_WARP; off <<= 1) {
        v = fmaxf(v, __shfl_xor(v, off, LT_WARP));
    }
    return v;
}
__device__ __forceinline__ float lt_warp_sum32(float v) {
#pragma unroll
    for (int off = 1; off < LT_WARP; off <<= 1) {
        v += __shfl_xor(v, off, LT_WARP);
    }
    return v;
}

// ---------------------------------------------------------------------------
// The attention kernel. Grid: (ceil(tok_count/2), 1, n_head/16). One launch per
// layer per chunk; each block stages one K/V tile and reuses it for its 32
// query columns (2 tokens x 16 heads of the shared MQA cache).
// ---------------------------------------------------------------------------
static __global__ __launch_bounds__(LT_NTHREADS, 1) void attn_tile_kernel(
        const float * __restrict__ q,
        const __half * __restrict__ cache16,
        const float * __restrict__ wk_b,
        float * __restrict__ latent,
        int n_head, int head_k,
        int tok_base, int tok_count, int chunk_tokens, int n_pos,
        float scale,
        int64_t q_tok, int64_t l_tok) {
    const int lane = (int) threadIdx.x;
    const int wid  = (int) threadIdx.y;
    const int tid  = wid * LT_WARP + lane;

    const int tok0 = tok_base + (int) blockIdx.x * LT_NCOLS1;
    const int h0   = (int) blockIdx.z * LT_NCOLS2;
    const int seq_t = n_pos - chunk_tokens;   // where the whole chunk starts

    // Causal limit per query column: column jc is token (jc/16), head (jc%16).
    int col_lim[LT_NCOLS1];
#pragma unroll
    for (int j = 0; j < LT_NCOLS1; ++j) {
        const int t = tok0 + j;
        col_lim[j] = (t >= tok_base && t < tok_base + tok_count)
                         ? seq_t + t + 1
                         : 0;
    }
    int k_sup = col_lim[0] > col_lim[1] ? col_lim[0] : col_lim[1];

    // ---- Q staging: absorbed, pre-scaled, f16, in LDS for the whole loop ---
    // q576[c] = [sum_j wk_b[(h*kv+d)*nope+j] * q_nope[j]  |  q_pe[d-512]]
    __shared__ __half2 Q_tmp[LT_NCOLS * (LT_KVA / 2)];
    for (int idx = tid; idx < LT_NCOLS * (LT_KVA / 2); idx += LT_NTHREADS) {
        const int jc = idx / (LT_KVA / 2);
        const int d2 = idx % (LT_KVA / 2);
        const int t = tok0 + jc / LT_NCOLS2;
        const int h = h0 + jc % LT_NCOLS2;
        __half2 v = __float2half2_rn(0.0f);
        if (t >= tok_base && t < tok_base + tok_count) {
            float f[2];
            for (int e = 0; e < 2; ++e) {
                const int d = 2 * d2 + e;
                const float * qhp =
                    q + (int64_t) t * q_tok + (int64_t) h * head_k;
                float x;
                if (d < LT_KV) {
                    const float * wrow =
                        wk_b + ((int64_t) h * LT_KV + d) * LT_NOPE;
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                    for (int j = 0; j < LT_NOPE; j += 4) {
                        const float4 ww =
                            *reinterpret_cast<const float4 *>(wrow + j);
                        const float4 qq =
                            *reinterpret_cast<const float4 *>(qhp + j);
                        a0 += ww.x * qq.x;
                        a1 += ww.y * qq.y;
                        a2 += ww.z * qq.z;
                        a3 += ww.w * qq.w;
                    }
                    x = (a0 + a1) + (a2 + a3);
                } else {
                    x = qhp[LT_NOPE + d - LT_KV];
                }
                f[e] = x * scale;
            }
            v = __floats2half2_rn(f[0], f[1]);
        }
        Q_tmp[idx] = v;
    }

    // ---- per-lane softmax state and output accumulators (fork's shape) -----
    float KQ_max[LT_CPW], KQ_sum[LT_CPW];
#pragma unroll
    for (int c = 0; c < LT_CPW; ++c) {
        KQ_max[c] = -FLT_MAX / 2.0f;
        KQ_sum[c] = 0.0f;
    }
    float2 VKQ[LT_CPW * LT_DV_FRAG];
#pragma unroll
    for (int i = 0; i < LT_CPW * LT_DV_FRAG; ++i) {
        VKQ[i] = make_float2(0.0f, 0.0f);
    }

    // One K/V tile in LDS: rows [0, LT_BATCH_FA) x 576 f16 for the KQ pass,
    // reused row-by-row for the V pass (the fork's KV_tmp).
    __shared__ __half2 KV_tmp[LT_BATCH_FA * (LT_BATCH_K / 2 + 4) + 16];
    __shared__ __half  KQ_p[LT_NCOLS * LT_BATCH_FA];   // softmax output, f16

    for (int p0 = 0; p0 < k_sup; p0 += LT_BATCH_FA) {
        const int npt = k_sup - p0 < LT_BATCH_FA ? k_sup - p0 : LT_BATCH_FA;

        // ---- KQ: score tile = q576 . cache[p0:p0+64, 0:576] ---------------
        float KQ_acc[LT_ROWS_PER_LANE * LT_CPW];
#pragma unroll
        for (int i = 0; i < LT_ROWS_PER_LANE * LT_CPW; ++i) KQ_acc[i] = 0.0f;

        for (int k0 = 0; k0 < LT_KVA; k0 += LT_BATCH_K) {
            // stage this 64-dim slice of 64 rows
            for (int idx = tid; idx < LT_BATCH_FA * (LT_BATCH_K / 2);
                 idx += LT_NTHREADS) {
                const int r = idx / (LT_BATCH_K / 2);
                const int d2 = idx % (LT_BATCH_K / 2);
                __half2 v = __floats2half2_rn(0.0f, 0.0f);
                if (r < npt) {
                    v = reinterpret_cast<const __half2 *>(
                        cache16 + (int64_t) (p0 + r) * LT_KVA + k0)[d2];
                }
                KV_tmp[r * (LT_BATCH_K / 2 + 4) + d2] = v;
            }
            __syncthreads();

            // register-blocked dot2: each lane holds 2 rows x cpw columns
#pragma unroll
            for (int k1 = 0; k1 < LT_BATCH_K / 2; k1 += 4) {
                __half2 K_k[LT_ROWS_PER_LANE][4];
                __half2 Q_k[LT_CPW][4];
#pragma unroll
                for (int ir = 0; ir < LT_ROWS_PER_LANE; ++ir) {
                    const int r = ir * LT_WARP + lane;
                    const __half2 * src =
                        &KV_tmp[r * (LT_BATCH_K / 2 + 4) + k1];
#pragma unroll
                    for (int e = 0; e < 4; ++e) K_k[ir][e] = src[e];
                }
#pragma unroll
                for (int c = 0; c < LT_CPW; ++c) {
                    const int jc = wid * LT_CPW + c;
                    const __half2 * src = &Q_tmp[jc * (LT_KVA / 2) + k0 / 2 + k1];
#pragma unroll
                    for (int e = 0; e < 4; ++e) Q_k[c][e] = src[e];
                }
#pragma unroll
                for (int ir = 0; ir < LT_ROWS_PER_LANE; ++ir) {
#pragma unroll
                    for (int c = 0; c < LT_CPW; ++c) {
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            lt_mad(KQ_acc[ir * LT_CPW + c], K_k[ir][e],
                                   Q_k[c][e]);
                        }
                    }
                }
            }
            __syncthreads();
        }

        // ---- softmax over this tile, per column; P to LDS; rescale VKQ ----
        float KQ_max_new[LT_CPW];
#pragma unroll
        for (int c = 0; c < LT_CPW; ++c) {
            KQ_max_new[c] = KQ_max[c];
            {
                const int j = (wid * LT_CPW + c) / LT_NCOLS2;
#pragma unroll
                for (int ir = 0; ir < LT_ROWS_PER_LANE; ++ir) {
                    const int r = ir * LT_WARP + lane;
                    if (r < npt && (p0 + r) < col_lim[j]) {
                        KQ_max_new[c] =
                            fmaxf(KQ_max_new[c], KQ_acc[ir * LT_CPW + c]);
                    }
                }
            }
            KQ_max_new[c] = lt_warp_max32(KQ_max_new[c]);
        }
#pragma unroll
        for (int c = 0; c < LT_CPW; ++c) {
            const int jc = wid * LT_CPW + c;
            const int j  = jc / LT_NCOLS2;
            const float mscale = expf(KQ_max[c] - KQ_max_new[c]);
            KQ_max[c] = KQ_max_new[c];
            float add = 0.0f;
#pragma unroll
            for (int ir = 0; ir < LT_ROWS_PER_LANE; ++ir) {
                const int r = ir * LT_WARP + lane;
                const bool ok = r < npt && (p0 + r) < col_lim[j];
                const float val =
                    ok ? expf(KQ_acc[ir * LT_CPW + c] - KQ_max[c]) : 0.0f;
                add += val;
                KQ_p[jc * LT_BATCH_FA + r] = (__half) val;
            }
            KQ_sum[c] = KQ_sum[c] * mscale + add;
#pragma unroll
            for (int i = 0; i < LT_DV_FRAG; ++i) {
                VKQ[c * LT_DV_FRAG + i].x *= mscale;
                VKQ[c * LT_DV_FRAG + i].y *= mscale;
            }
        }
        __syncthreads();

        // ---- VKQ: latent tile += P . cache[p0:p0+npt, 0:512] ---------------
        // The fork's schedule (fattn-tile.cuh flash_attn_tile_iter, V pass):
        // stage nbatch_V rows of V through the SAME LDS scratch the KQ pass
        // used, then accumulate from LDS — never from global in the inner
        // loop. (The first draft read each row from global serially: 64
        // dependent 32-B loads per lane per tile, which is where the gap to
        // the fork's 38 ms lived.) V is the first LT_KV halves of the
        // LT_KVA-half cache row, so the ROW STRIDE is LT_KVA (576).
        constexpr int LT_NBV = 8;                       // nbatch_V (the fork)
        constexpr int LT_VS = LT_KV / 2;                // 256 half2 per V row
        for (int rb = 0; rb < npt; rb += LT_NBV) {
            const int nr = npt - rb < LT_NBV ? npt - rb : LT_NBV;
            for (int idx = tid; idx < LT_NBV * LT_VS; idx += LT_NTHREADS) {
                const int vr = idx / LT_VS;
                const int d2 = idx % LT_VS;
                __half2 v = __floats2half2_rn(0.0f, 0.0f);
                if (vr < nr) {
                    v = reinterpret_cast<const __half2 *>(
                        cache16 + (int64_t) (p0 + rb + vr) * LT_KVA)[d2];
                }
                KV_tmp[vr * LT_VS + d2] = v;
            }
            __syncthreads();
#pragma unroll
            for (int vr = 0; vr < LT_NBV; ++vr) {
                if (vr >= nr) break;
                const int r0 = rb + vr;
                __half2 P_k[LT_CPW];
#pragma unroll
                for (int c = 0; c < LT_CPW; ++c) {
                    const int jc = wid * LT_CPW + c;
                    const __half ph = KQ_p[jc * LT_BATCH_FA + r0];
                    P_k[c] = __half2half2(ph);
                }
                float2 V_k[LT_DV_FRAG];
                {
                    const __half2 * src =
                        &KV_tmp[vr * LT_VS + lane * LT_DV_FRAG];
#pragma unroll
                    for (int i = 0; i < LT_DV_FRAG; ++i) {
                        V_k[i] = __half22float2(src[i]);
                    }
                }
#pragma unroll
                for (int c = 0; c < LT_CPW; ++c) {
                    const float pk = __half2float(__low2half(P_k[c]));
#pragma unroll
                    for (int i = 0; i < LT_DV_FRAG; ++i) {
                        VKQ[c * LT_DV_FRAG + i].x += V_k[i].x * pk;
                        VKQ[c * LT_DV_FRAG + i].y += V_k[i].y * pk;
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }

    // ---- write the 512-dim latent per (token, head) ------------------------
#pragma unroll
    for (int c = 0; c < LT_CPW; ++c) {
        const int jc = wid * LT_CPW + c;
        const int j  = jc / LT_NCOLS2;
        const int t  = tok0 + j;
        const int h  = h0 + jc % LT_NCOLS2;
        if (t < tok_base || t >= tok_base + tok_count) continue;
        // The fork's warp_reduce_sum before the divide (fattn-tile.cuh:975):
        // each lane's KQ_sum[c] holds only its 2 rows of the tile's exp values
        // while VKQ accumulated all 64 rows, so the normalizer must be the
        // full-tile sum. Uniform across the warp (t depends on jc only).
        const float inv = 1.0f / lt_warp_sum32(KQ_sum[c]);
        float * row = latent + (int64_t) t * l_tok + (int64_t) h * LT_KV;
#pragma unroll
        for (int i = 0; i < LT_DV_FRAG; ++i) {
            const int d = lane * (2 * LT_DV_FRAG) + 2 * i;
            row[d + 0] = VKQ[c * LT_DV_FRAG + i].x * inv;
            row[d + 1] = VKQ[c * LT_DV_FRAG + i].y * inv;
        }
    }
}

// ---------------------------------------------------------------------------
// De-absorb + gate as a batched GEMM (LR1 §6.8): out[t,h,r] =
// sigmoid(gate[t,h]) * sum_d wv_b[h][r][d] * latent[t,h,d]. One LDS copy of a
// wv_b[h] r-tile serves every token of the chunk.
// ---------------------------------------------------------------------------
static __global__ __launch_bounds__(256) void tile_deabsorb_kernel(
        const float * __restrict__ latent,
        const float * __restrict__ wv_b,
        const float * __restrict__ gate,
        float * __restrict__ out,
        int tok_base, int tok_count, int n_head, int head_v,
        int64_t l_tok, int64_t o_tok, int64_t g_tok) {
    const int h = (int) blockIdx.x;
    const int r0 = (int) blockIdx.y * 16;
    __shared__ float W[LT_KV][16];   // wv_b[h][r0:r0+16][0:512]
    for (int idx = (int) threadIdx.x; idx < LT_KV * 16; idx += 256) {
        const int d = idx / 16;
        const int r = idx % 16;
        W[d][r] = wv_b[((int64_t) h * head_v + r0 + r) * LT_KV + d];
    }
    __syncthreads();
    for (int idx = (int) threadIdx.x; idx < tok_count * 16; idx += 256) {
        const int ti = idx / 16;
        const int r  = idx % 16;
        const int t  = tok_base + ti;
        const float * lat =
            latent + (int64_t) t * l_tok + (int64_t) h * LT_KV;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int d = 0; d < LT_KV; d += 4) {
            const float4 l4 = *reinterpret_cast<const float4 *>(lat + d);
            const float4 w0 = make_float4(W[d + 0][r], W[d + 1][r],
                                          W[d + 2][r], W[d + 3][r]);
            a0 += l4.x * w0.x;
            a1 += l4.y * w0.y;
            a2 += l4.z * w0.z;
            a3 += l4.w * w0.w;
        }
        const float g = 1.0f / (1.0f + expf(-gate[(int64_t) t * g_tok + h]));
        out[(int64_t) t * o_tok + (int64_t) h * head_v + r0 + r] =
            g * ((a0 + a1) + (a2 + a3));
    }
}

// f32 -> f16 conversion of the cache rows this call reads, once per call.
static __global__ __launch_bounds__(256) void cache_to_f16_kernel(
        const float * __restrict__ src, __half * __restrict__ dst, int n) {
    const int i = (int) (blockIdx.x * 256u + threadIdx.x);
    const int stride = (int) (gridDim.x * 256u);
    for (int e = i; e < n; e += stride) {
        dst[e] = __float2half(src[e]);
    }
}

} // namespace

bool mla_tile_supported(const MlaShape & sh, const MlaCache & cache) {
    return sh.n_head == 32 && sh.head_k == 192 && sh.head_v == LT_HV &&
           sh.nope == LT_NOPE && sh.rot == LT_ROT && sh.kv == LT_KV &&
           cache.s_pos == sh.kv_a();
}

namespace {

/// Host-side index audit (the lf_audit pattern): every index the two kernels
/// compute is enumerated against the caller's declared extents before launch.
void lt_audit(const MlaShape & sh, const MlaInputs & in, const View & out,
              const MlaCache & cache, int n_pos, int tok_base, int tok_count,
              int chunk_tokens, int64_t l_tok) {
    const bool ok_shape =
        sh.n_head == 32 && sh.head_k == 192 && sh.head_v == LT_HV &&
        sh.nope == LT_NOPE && sh.rot == LT_ROT && sh.kv == LT_KV &&
        cache.s_pos == sh.kv_a();
    const bool ok_tok = tok_base >= 0 && tok_count > 0 && chunk_tokens > 0 &&
                        tok_base + tok_count <= in.n_tokens;
    const bool ok_pos = n_pos >= chunk_tokens && n_pos <= cache.max_pos;
    const bool ok_ptrs = cache.kv != nullptr && in.q.p != nullptr &&
                         in.gate.p != nullptr && out.p != nullptr;
    const bool ok_stride = in.q.s_tok == sh.q_dim() &&
                           in.gate.s_tok == sh.n_head &&
                           out.s_tok == sh.o_dim();
    if (!(ok_shape && ok_tok && ok_pos && ok_ptrs && ok_stride)) {
        std::fprintf(stderr,
                     "ling3_mla_tile: REFUSING launch, index audit failed "
                     "shape=%d tok=%d pos=%d ptr=%d stride=%d\n"
                     "  tok_base=%d tok_count=%d chunk_tokens=%d n_tokens=%d "
                     "n_pos=%d max_pos=%d l_tok=%lld\n",
                     (int) ok_shape, (int) ok_tok, (int) ok_pos, (int) ok_ptrs,
                     (int) ok_stride, tok_base, tok_count, chunk_tokens,
                     in.n_tokens, n_pos, cache.max_pos, (long long) l_tok);
        std::abort();
    }
    const int64_t cache_elems = (int64_t) cache.max_pos * cache.s_pos;
    for (int t = tok_base; t < tok_base + tok_count; ++t) {
        const int p_limit = (n_pos - chunk_tokens) + t + 1;
        if (p_limit < 1 || p_limit > n_pos) abort();
        if ((int64_t) p_limit * cache.s_pos > cache_elems) abort();
        const int64_t q_hi = (int64_t) t * in.q.s_tok + sh.q_dim();
        if (q_hi <= 0 || (size_t) q_hi > (size_t) in.n_tokens * sh.q_dim())
            abort();
        const int64_t l_hi = (int64_t) t * l_tok + sh.n_head * sh.kv;
        if (l_hi <= 0) abort();
        const int64_t o_hi = (int64_t) t * out.s_tok + sh.n_head * sh.head_v;
        if (o_hi <= 0) abort();
    }
}

} // namespace

void mla_attn_tile(const MlaShape & sh, const MlaScalars & sc, const MlaWeights & w,
                   const MlaCache & cache, int n_pos, const MlaInputs & in,
                   const View & out, float * scores,
                   int tok_base, int tok_count, int chunk_tokens,
                   hipStream_t stream) {
    (void) scores;   // diagnostic scratch of the tok path; unused here
    if (!mla_tile_supported(sh, cache)) {
        mla_attn_tok(sh, sc, w, cache, n_pos, in, out, scores, tok_base, tok_count,
                     chunk_tokens, stream);
        return;
    }
    // The Q staging and the de-absorb read float4 (the fork does too); refuse
    // any buffer the allocator did not 16-B align and fall back for the chunk.
    if ((reinterpret_cast<std::uintptr_t>(in.q.p) % 16) != 0 ||
        (reinterpret_cast<std::uintptr_t>(w.wk_b) % 16) != 0 ||
        (reinterpret_cast<std::uintptr_t>(w.wv_b) % 16) != 0 ||
        (reinterpret_cast<std::uintptr_t>(cache.kv) % 16) != 0) {
        mla_attn_tok(sh, sc, w, cache, n_pos, in, out, scores, tok_base, tok_count,
                     chunk_tokens, stream);
        return;
    }

    // latent scratch: [n_tokens][n_head][kv] f32, one slice per call.
    static float * lat_base = nullptr;
    static size_t lat_cap = 0;
    const int64_t l_tok = (int64_t) sh.n_head * sh.kv;   // 16384
    const size_t need = (size_t) in.n_tokens * (size_t) l_tok;
    if (need > lat_cap) {
        if (lat_base != nullptr) hipFree(lat_base);
        lat_base = nullptr;
        LT_HIP_CHECK(hipMalloc(&lat_base, need * sizeof(float)));
        lat_cap = need;
    }
    float * lat = lat_base;
    lt_audit(sh, in, out, cache, n_pos, tok_base, tok_count, chunk_tokens, l_tok);

    // f16 cache scratch, converted once per call (SPEC2 2.6: converting per
    // workgroup in the staging loop measured ~90 ms of a 266 ms kernel).
    static __half * c16 = nullptr;
    static size_t c16_cap = 0;
    const size_t cn = (size_t) n_pos * (size_t) sh.kv_a();
    if (cn > c16_cap) {
        if (c16 != nullptr) hipFree(c16);
        c16 = nullptr;
        LT_HIP_CHECK(hipMalloc(&c16, cn * sizeof(__half)));
        c16_cap = cn;
    }
    {
        const int total = (int) cn;
        const unsigned blocks = (unsigned) ((total + 255) / 256);
        hipLaunchKernelGGL(cache_to_f16_kernel, dim3(blocks), 256, 0, stream,
                           cache.kv, c16, total);
        LT_HIP_CHECK(hipGetLastError());
    }

    const float kq_scale = sc.kq_scale != 0.0f ? sc.kq_scale
                                               : 1.0f / sqrtf((float) sh.head_k);
    const dim3 grid((unsigned) ((tok_count + LT_NCOLS1 - 1) / LT_NCOLS1), 1u,
                    (unsigned) (sh.n_head / LT_NCOLS2));
    hipLaunchKernelGGL(attn_tile_kernel, grid,
                       dim3(LT_WARP, LT_NWARPS), 0, stream,
                       in.q.p, c16, w.wk_b, lat,
                       sh.n_head, sh.head_k,
                       tok_base, tok_count, chunk_tokens, n_pos, kq_scale,
                       in.q.s_tok, l_tok);
    LT_HIP_CHECK(hipGetLastError());

    const dim3 grid2((unsigned) sh.n_head, (unsigned) (sh.head_v / 16));
    hipLaunchKernelGGL(tile_deabsorb_kernel, grid2, 256, 0, stream,
                       lat, w.wv_b, in.gate.p, out.p,
                       tok_base, tok_count, sh.n_head, sh.head_v,
                       l_tok, out.s_tok, in.gate.s_tok);
    LT_HIP_CHECK(hipGetLastError());
}

const char * mla_tile_version() { return "ling3-mla-tile/1.0"; }

} // namespace ling3
