// ling3_mla_w2.hip — LW1 (wave 2): two-pass WMMA absorbed-MLA prefill attention.
// See ling3_mla_w2.hpp for the design and the contracts.
//
// Arithmetic (identical per (token, head) to mla_attn / mla_attn_tok):
//   q_abs   = wk_b[h] . q_nope                      (fp32, i-ascending, attn_tok's order)
//   score   = (q_abs . latent + q_pe . k_pe) / sqrt(head_k)
//   w       = softmax(score) over the causal window
//   latent  = sum_p w[p] * cache[p][0:kv]
//   out     = sigmoid(gate) * wv_b[h] . latent
//
// The score dot and the latent accumulation become two f16 WMMA GEMMs
// (16x16x16, wave32 — the same builtin and the same fragment mapping as
// superroc's qwen attention_wmma.hip, validated there against a CPU double
// reference on this tree):
//   A fragment: lane L supplies matrix row L%16, 16 contiguous K values.
//   B fragment: lane L supplies matrix column L%16, 16 contiguous K values,
//               read from a [N][K]-major tile (so pass 1 reads the [pos][dim]
//               cache staging directly, pass 2 reads [dim][pos]).
//   C fragment: lane L element i is row (2i + L/16), column L%16.
//
// Determinism: no atomics anywhere; every reduction is a fixed-order block
// reduce; the WMMA instruction is fixed; the pass-2 k loop accumulates in
// ascending position order inside one block per output tile. Same inputs,
// same launch parameters -> bit-identical output run to run.
//
// Bounds discipline (LF1's lf_audit model — this lane faults no GPUs):
// every global index is enumerated here and re-checked host-side by
// w2_audit() before any launch; the kernels themselves add row guards.
//   q16/k16 rows  < ceil(R/64)*64 <= kMlaW2Rows / cap   (R <= 2048, cap >= n_pos, both mult of 64)
//   p16/vt  cols  < ceil(p_hi/64)*64 <= ceil(n_pos/64)*64 <= cap, and k16/vt carry explicit
//                  zeros on [n_pos, n_pos_pad) so rounded-up loads never touch poison
//   scores writes < R * n_pos   (row and causal-column guarded)
//   o writes      < R * kv      (row guarded)
//   out writes    follow mla_attn_tok's own orow arithmetic

#include "ling3_mla_w2.hpp"

#include <cmath>
#include <cstdio>

#define LING3_W2_HIP_CHECK(expr)                                                     \
    do {                                                                            \
        hipError_t _err = (expr);                                                   \
        if (_err != hipSuccess) {                                                   \
            fprintf(stderr, "ling3_mla_w2: %s failed at %s:%d: %s\n", #expr,         \
                    __FILE__, __LINE__, hipGetErrorString(_err));                    \
            abort();                                                                \
        }                                                                           \
    } while (0)

namespace ling3 {

namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f  = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// A/B fragments are 16 contiguous halves = 32 bytes; two 16-byte loads and a
// bitcast (the qwen kernel's LoadFrag, verbatim). Every fragment base below is
// 16-byte aligned: row strides are multiples of 8 halves and the k offset is a
// multiple of 16 halves.
__device__ __forceinline__ v16h LoadFrag(const __half * p) {
    union {
        v16h f;
        uint4 u[2];
    } cvt;
    cvt.u[0] = *reinterpret_cast<const uint4 *>(p);
    cvt.u[1] = *reinterpret_cast<const uint4 *>(p + 8);
    return cvt.f;
}

// Fixed-order block reductions (mla.hip's own, duplicated because they are
// file-static there). The snapshot barrier is load-bearing: the caller may
// reuse the shared buffer on the very next reduce.
template <int BLOCK>
__device__ __forceinline__ float w2_reduce_max(float v, float * smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off && smem[threadIdx.x + off] > smem[threadIdx.x]) {
            smem[threadIdx.x] = smem[threadIdx.x + off];
        }
        __syncthreads();
    }
    const float result = smem[0];
    __syncthreads();
    return result;
}

template <int BLOCK>
__device__ __forceinline__ float w2_reduce_sum(float v, float * smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) {
            smem[threadIdx.x] += smem[threadIdx.x + off];
        }
        __syncthreads();
    }
    const float result = smem[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float w2_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---------------------------------------------------------------------------
// 1. Cache staging: f32 rows [0, n_pos) -> k16[p][kv_a] and vt[j][p] (latent
// only), with explicit zeros on [n_pos, n_pos_pad) so the GEMMs' rounded-up
// fragment loads never multiply uninitialised memory.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w2_k16_kernel(
        const float * __restrict__ cache, int64_t s_pos,
        __half * __restrict__ k16, __half * __restrict__ vt,
        int kv_a, int kv, int n_pos, int n_pos_pad, int cap) {
    const int e = blockIdx.x * 256 + (int) threadIdx.x;
    if (e >= n_pos_pad * kv_a) return;
    const int p = e / kv_a;
    const int i = e % kv_a;
    const __half v = p < n_pos ? __float2half(cache[(int64_t) p * s_pos + i])
                               : __half(0.0f);
    k16[(int64_t) p * kv_a + i] = v;
    if (i < kv) {
        vt[(int64_t) i * cap + p] = v;
    }
}

// ---------------------------------------------------------------------------
// 2. Pack q: one warp per (token, head) pair. The absorption is attn_tok's
// own serial i-ascending dot per r (16 per lane), then the row is scaled by
// 1/sqrt(head_k) and stored f16 — the scale rides the query, so pass 1's WMMA
// output is the scaled score directly.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w2_packq_kernel(
        const float * __restrict__ q, const float * __restrict__ wk_b,
        __half * __restrict__ q16, int n_head, int head_k, int nope,
        int kv, int rot, int64_t q_tok, int tok_base, int tok_count,
        float scale) {
    const int R = tok_count * n_head;
    const int warp = (int) threadIdx.x >> 5;
    const int lane = (int) threadIdx.x & 31;
    const int r = blockIdx.x * 8 + warp;
    if (r >= R) return;
    const int lt = r / n_head;
    const int h = r % n_head;
    const int t = tok_base + lt;

    const float * qh = q + (int64_t) h * head_k + (int64_t) t * q_tok;   // s_seq == 0
    __half * row = q16 + (int64_t) r * (kv + rot);

#pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
        const int r_dim = lane + 32 * jj;                 // < kv (512)
        const float * wrow = wk_b + ((int64_t) h * kv + r_dim) * nope;
        float acc = 0.0f;
        for (int i = 0; i < nope; ++i) {
            acc += wrow[i] * qh[i];
        }
        row[r_dim] = __float2half(acc * scale);
    }
#pragma unroll
    for (int pe = 0; pe < 2; ++pe) {
        row[kv + lane + 32 * pe] = __float2half(qh[nope + lane + 32 * pe] * scale);
    }
}

// ---------------------------------------------------------------------------
// 3. Pass 1: S = q16 . k16^T on the matrix cores. One block covers a 64-row x
// 64-position tile; 8 wave32 warps split it as warp -> (row tile rt = w & 3,
// position tiles pt0 = w >> 2 and pt0 + 2), so each A fragment is loaded once
// and feeds two 16x16 C tiles. Blocks whose whole tile lies beyond their last
// row's causal limit exit before any load (block-uniform, before the first
// barrier).
//
// v2: fragments stage through LDS. The w32 fragment layout makes every
// global fragment load move 2x the tile's unique bytes (each element lives on
// two lanes) and every B row is re-read by the four row-tile warps; measured
// v1 ran ~2x short of the fork at depth with the mla section latency-bound on
// those loads. Staging the 64x64 A/B k-chunks through padded LDS (the qwen
// attention_wmma.hip design) keeps the WMMA inputs identical, so the outputs
// are bit-for-bit v1's.
// ---------------------------------------------------------------------------
constexpr int W2_TSTRIDE = 72;   // halves per LDS row: 64 + 8 pad (16B aligned,
                                 // 2-way bank conflict on the 16 frag lanes)

__global__ __launch_bounds__(256) void w2_scores_kernel(
        const __half * __restrict__ q16, const __half * __restrict__ k16,
        float * __restrict__ scores,
        int R, int n_pos, int base, int tok_base,
        int kv_a) {
    __shared__ __half As[64][W2_TSTRIDE];
    __shared__ __half Bs[64][W2_TSTRIDE];

    const int r0 = blockIdx.x * 64;
    const int p0 = blockIdx.y * 64;
    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int sub = lane & 15;
    const int half_id = lane >> 4;

    // this tile's LAST row decides how far the loads may reach
    int gr_hi = r0 + 63;
    if (gr_hi >= R) gr_hi = R - 1;
    const int pl_hi = base + tok_base + (gr_hi >> 5) + 1;
    if (p0 >= pl_hi) return;

    const int rt = warp & 3;
    const int pt0 = warp >> 2;
    const int pt1 = pt0 + 2;

    v8f c0 = {};
    v8f c1 = {};
    for (int kc = 0; kc < kv_a; kc += 64) {          // 9 chunks of 64 dims
        // stage the A and B k-chunks coalesced, 16B pieces
        for (int e = tid; e < 512; e += 256) {        // 64 rows x 8 pieces
            const int r = e >> 3;
            const int c8 = (e & 7) << 3;
            *(uint4 *) &As[r][c8] =
                *(const uint4 *) &q16[(int64_t) (r0 + r) * kv_a + kc + c8];
            *(uint4 *) &Bs[r][c8] =
                *(const uint4 *) &k16[(int64_t) (p0 + r) * kv_a + kc + c8];
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const v16h a  = LoadFrag(&As[rt * 16 + sub][ks * 16]);
            const v16h b0 = LoadFrag(&Bs[pt0 * 16 + sub][ks * 16]);
            const v16h b1 = LoadFrag(&Bs[pt1 * 16 + sub][ks * 16]);
            c0 = Wmma(a, b0, c0);
            c1 = Wmma(a, b1, c1);
        }
        __syncthreads();   // before restaging the buffers
    }

    // Epilogue: C element i of lane L is row (2i + half_id), column sub. Both
    // the row (query rows past R are staging padding) and the causal column
    // are masked per element.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int gr = r0 + rt * 16 + 2 * i + half_id;
        if (gr >= R) continue;
        const int pl = base + tok_base + (gr >> 5) + 1;
        const int cp0 = p0 + pt0 * 16 + sub;
        if (cp0 < pl) scores[(int64_t) gr * n_pos + cp0] = c0[i];
        const int cp1 = p0 + pt1 * 16 + sub;
        if (cp1 < pl) scores[(int64_t) gr * n_pos + cp1] = c1[i];
    }
}

// ---------------------------------------------------------------------------
// 4. Softmax per row, f32, unnormalised: P = exp(S - m), sum deferred to the
// epilogue kernel (exact algebra: latent = (P.V)/sum). Rows beyond each row's
// own causal limit are zero-filled up to the sub-batch's padded tile edge so
// pass 2's k loop can round up freely.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w2_softmax_kernel(
        const float * __restrict__ scores, __half * __restrict__ p16,
        float * __restrict__ rowsum,
        int n_pos, int cap, int base, int tok_base, int p_hi_pad) {
    __shared__ float red[256];
    const int r = blockIdx.x;                 // grid is exactly the R live rows
    const int lt = r >> 5;
    const int pl = base + tok_base + lt + 1;

    const float * srow = scores + (int64_t) r * n_pos;
    float m = -INFINITY;
    for (int p = (int) threadIdx.x; p < pl; p += 256) {
        m = fmaxf(m, srow[p]);
    }
    m = w2_reduce_max<256>(m, red);

    __half * prow = p16 + (int64_t) r * cap;
    float sum = 0.0f;
    for (int p = (int) threadIdx.x; p < pl; p += 256) {
        const float e = expf(srow[p] - m);
        prow[p] = __float2half(e);
        sum += e;
    }
    sum = w2_reduce_sum<256>(sum, red);
    if (threadIdx.x == 0) {
        rowsum[r] = sum;
    }
    for (int p = pl + (int) threadIdx.x; p < p_hi_pad; p += 256) {
        prow[p] = __half(0.0f);
    }
}

// ---------------------------------------------------------------------------
// 5. Pass 2: O = P . V on the matrix cores. One block covers 64 rows x 128
// latent dims; warp w owns the 16-wide j tile, all four 16-row A tiles, and
// walks positions in 64-wide chunks (4 k-steps). The k bound is this row
// tile's last live causal limit, rounded up to 64 — p16 is zero there.
//
// v2: the 64x64 P chunk and the 128x64 V chunk stage through padded LDS (v1
// re-read every P fragment from global for each of the four row tiles: 16x
// amplification on the tile's unique bytes). Identical WMMA inputs, so the
// outputs are bit-for-bit v1's.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w2_pv_kernel(
        const __half * __restrict__ p16, const __half * __restrict__ vt,
        float * __restrict__ o,
        int R, int kv, int cap, int base, int tok_base) {
    __shared__ __half Ps[64][W2_TSTRIDE];
    __shared__ __half Vs[128][W2_TSTRIDE];

    const int r0 = blockIdx.x * 64;
    const int j0b = blockIdx.y * 128;
    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int sub = lane & 15;
    const int half_id = lane >> 4;
    const int j0 = j0b + warp * 16;

    int gr_hi = r0 + 63;
    if (gr_hi >= R) gr_hi = R - 1;
    const int k_hi = base + tok_base + (gr_hi >> 5) + 1;
    const int k_end = (k_hi + 63) & ~63;

    v8f c[4] = {};
    for (int pk = 0; pk < k_end; pk += 64) {
        // stage the P chunk (64 rows x 64 positions) and the V chunk
        // (128 latent dims x 64 positions) coalesced, 16B pieces
        for (int e = tid; e < 512; e += 256) {
            const int r = e >> 3;
            const int c8 = (e & 7) << 3;
            *(uint4 *) &Ps[r][c8] =
                *(const uint4 *) &p16[(int64_t) (r0 + r) * cap + pk + c8];
        }
        for (int e = tid; e < 1024; e += 256) {
            const int j = e >> 3;
            const int c8 = (e & 7) << 3;
            *(uint4 *) &Vs[j][c8] =
                *(const uint4 *) &vt[(int64_t) (j0b + j) * cap + pk + c8];
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const v16h b = LoadFrag(&Vs[warp * 16 + sub][ks * 16]);
            c[0] = Wmma(LoadFrag(&Ps[0 * 16 + sub][ks * 16]), b, c[0]);
            c[1] = Wmma(LoadFrag(&Ps[1 * 16 + sub][ks * 16]), b, c[1]);
            c[2] = Wmma(LoadFrag(&Ps[2 * 16 + sub][ks * 16]), b, c[2]);
            c[3] = Wmma(LoadFrag(&Ps[3 * 16 + sub][ks * 16]), b, c[3]);
        }
        __syncthreads();   // before restaging the buffers
    }

#pragma unroll
    for (int rt = 0; rt < 4; ++rt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int gr = r0 + rt * 16 + 2 * i + half_id;
            if (gr >= R) continue;
            o[(int64_t) gr * kv + j0 + sub] = c[rt][i];
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Epilogue: normalise the latent, de-absorb through wv_b, apply the head
// gate. The de-absorb is attn_tok's own serial j-ascending dot per output
// element; the latent is staged through LDS because each element needs all
// kv values of its row (a warp reduction there would mix output elements).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void w2_out_kernel(
        const float * __restrict__ o, const float * __restrict__ rowsum,
        const float * __restrict__ gate, const float * __restrict__ wv_b,
        float * __restrict__ out,
        int R, int n_head, int head_v, int kv, int tok_base,
        int64_t g_tok, int64_t o_tok, int64_t o_seq) {
    __shared__ float lat[8][512];
    const int warp = (int) threadIdx.x >> 5;
    const int lane = (int) threadIdx.x & 31;
    const int r = blockIdx.x * 8 + warp;
    // R is a multiple of 32 (tok_count * n_head), so the grid is exact and no
    // warp is dead — but the barrier below must be reached unconditionally
    // regardless, so `live` guards the work and never returns early.
    const bool live = r < R;

    if (live) {
        const float inv = 1.0f / rowsum[r];
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            lat[warp][lane + 32 * jj] = o[(int64_t) r * kv + lane + 32 * jj] * inv;
        }
    }
    __syncthreads();

    const int lt = r / n_head;
    const int h = r % n_head;
    const int t = tok_base + lt;
    const float g = live ? w2_sigmoid(gate[h + (int64_t) t * g_tok]) : 0.0f;
    float * orow = out + (int64_t) h * head_v + (int64_t) t * o_tok +
                   /* s_seq == 0 */ (int64_t) 0 * o_seq;
    const float * latrow = lat[warp];

    if (live) {
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {          // head_v = 128 = 4 x 32
            const int r_out = lane + 32 * rr;
            const float * wrow = wv_b + ((int64_t) h * head_v + r_out) * kv;
            float acc = 0.0f;
            for (int j = 0; j < kv; ++j) {
                acc += wrow[j] * latrow[j];
            }
            orow[r_out] = g * acc;
        }
    }
}

// ---------------------------------------------------------------------------
// host side: audit + launch
// ---------------------------------------------------------------------------

// LF1's lf_audit model: enumerate every invariant the kernels rely on and
// refuse the launch (caller falls back to mla_attn_tok) if any fails. No GPU
// fault may come from a bad index.
bool w2_audit(const char * where, const MlaShape & sh, const MlaCache & cache,
              int n_pos, const MlaInputs & in, const View & out,
              const float * scores, const MlaW2Bufs & bufs,
              int tok_base, int tok_count, int chunk_tokens) {
    const int R = tok_count * sh.n_head;
    const char * bad = nullptr;
    if (in.n_seqs != 1) bad = "n_seqs != 1";
    else if (in.n_tokens != chunk_tokens) bad = "in.n_tokens != chunk_tokens";
    else if (tok_count <= 0 || tok_count > kMlaW2Batch) bad = "tok_count out of (0, 64]";
    else if (tok_base < 0) bad = "tok_base < 0";
    else if (tok_base + tok_count > chunk_tokens) bad = "tok_base + tok_count > chunk_tokens";
    else if (chunk_tokens <= 0 || chunk_tokens > n_pos) bad = "chunk_tokens out of (0, n_pos]";
    else if (n_pos <= 0 || n_pos > cache.max_pos) bad = "n_pos out of (0, cache.max_pos]";
    else if (R > kMlaW2Rows) bad = "R > kMlaW2Rows";
    else if (bufs.cap < n_pos || (bufs.cap & 63) != 0) bad = "bufs.cap < n_pos or not a multiple of 64";
    else if (bufs.k16 == nullptr || bufs.vt == nullptr || bufs.q16 == nullptr ||
             bufs.p16 == nullptr || bufs.o == nullptr || bufs.rowsum == nullptr)
        bad = "null staging buffer";
    else if (scores == nullptr) bad = "null scores";
    else if (in.q.p == nullptr || in.gate.p == nullptr) bad = "null input";
    else if (out.p == nullptr) bad = "null out";
    if (bad != nullptr) {
        fprintf(stderr, "ling3_mla_w2: REFUSING %s at %s: %s\n", "launch", where, bad);
        return false;
    }
    return true;
}

}  // namespace

bool mla_w2_prep(const MlaShape & sh, const MlaCache & cache, int n_pos,
                 MlaW2Bufs & bufs, hipStream_t stream) {
    // chunk-level audit: the staging buffers and the geometry. tok_* are
    // checked per sub-batch by mla_w2_attn.
    const char * bad = nullptr;
    if (n_pos <= 0 || n_pos > cache.max_pos) bad = "n_pos out of (0, cache.max_pos]";
    else if (bufs.cap < n_pos || (bufs.cap & 63) != 0) bad = "bufs.cap < n_pos or not a multiple of 64";
    else if (bufs.k16 == nullptr || bufs.vt == nullptr) bad = "null staging buffer";
    else if (cache.kv == nullptr) bad = "null cache";
    if (bad != nullptr) {
        fprintf(stderr, "ling3_mla_w2: REFUSING %s at %s: %s\n", "prep", "mla_w2_prep", bad);
        return false;
    }

    const int n_pos_pad = (n_pos + 63) & ~63;   // <= cap: cap >= n_pos, both mult of 64
    const int total = n_pos_pad * sh.kv_a();
    const dim3 grid((unsigned) ((total + 255) / 256));
    hipLaunchKernelGGL(w2_k16_kernel, grid, 256, 0, stream,
                       cache.kv, cache.s_pos, bufs.k16, bufs.vt,
                       sh.kv_a(), sh.kv, n_pos, n_pos_pad, bufs.cap);
    LING3_W2_HIP_CHECK(hipGetLastError());
    return true;
}

bool mla_w2_attn(const MlaShape & sh, const MlaScalars & sc, const MlaWeights & w,
                 int n_pos, const MlaInputs & in, const View & out,
                 float * scores, MlaW2Bufs & bufs,
                 int tok_base, int tok_count, int chunk_tokens,
                 hipStream_t stream) {
    // The f32 cache itself was validated by mla_w2_prep this chunk; this audit
    // covers the sub-batch geometry and the staging buffers. max_pos = n_pos
    // stands in for the cache bound prep already enforced.
    MlaCache c = {};
    c.max_pos = n_pos;
    if (!w2_audit("mla_w2_attn", sh, c, n_pos, in, out, scores, bufs,
                  tok_base, tok_count, chunk_tokens)) {
        return false;
    }

    const int R = tok_count * sh.n_head;
    const int base = n_pos - chunk_tokens;
    const int p_hi = base + tok_base + tok_count;          // <= n_pos
    const int p_hi_pad = (p_hi + 63) & ~63;                // <= cap
    const float scale = sc.kq_scale != 0.0f ? sc.kq_scale
                                            : 1.0f / sqrtf((float) sh.head_k);

    // 1. pack this sub-batch's queries (absorb + scale + f16)
    hipLaunchKernelGGL(w2_packq_kernel, dim3((unsigned) ((R + 7) / 8)), 256, 0,
                       stream, in.q.p, w.wk_b, bufs.q16,
                       sh.n_head, sh.head_k, sh.nope, sh.kv, sh.rot,
                       in.q.s_tok, tok_base, tok_count, scale);
    LING3_W2_HIP_CHECK(hipGetLastError());

    // 2. pass 1: scores
    hipLaunchKernelGGL(w2_scores_kernel,
                       dim3((unsigned) ((R + 63) / 64),
                            (unsigned) ((p_hi + 63) / 64)),
                       256, 0, stream,
                       bufs.q16, bufs.k16, scores,
                       R, n_pos, base, tok_base, sh.kv_a());
    LING3_W2_HIP_CHECK(hipGetLastError());

    // 3. softmax (grid = exactly R live rows)
    hipLaunchKernelGGL(w2_softmax_kernel, dim3((unsigned) R), 256, 0, stream,
                       scores, bufs.p16, bufs.rowsum,
                       n_pos, bufs.cap, base, tok_base, p_hi_pad);
    LING3_W2_HIP_CHECK(hipGetLastError());

    // 4. pass 2: latent out
    hipLaunchKernelGGL(w2_pv_kernel,
                       dim3((unsigned) ((R + 63) / 64),
                            (unsigned) (sh.kv / 128)),
                       256, 0, stream,
                       bufs.p16, bufs.vt, bufs.o,
                       R, sh.kv, bufs.cap, base, tok_base);
    LING3_W2_HIP_CHECK(hipGetLastError());

    // 5. de-absorb + gate
    hipLaunchKernelGGL(w2_out_kernel, dim3((unsigned) ((R + 7) / 8)), 256, 0,
                       stream, bufs.o, bufs.rowsum, in.gate.p, w.wv_b, out.p,
                       R, sh.n_head, sh.head_v, sh.kv, tok_base,
                       in.gate.s_tok, out.s_tok, out.s_seq);
    LING3_W2_HIP_CHECK(hipGetLastError());

    return true;
}

const char * mla_w2_version() { return "ling3-mla-w2/1.0-lw1"; }

} // namespace ling3
