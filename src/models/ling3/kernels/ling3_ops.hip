// ling3_ops.hip — the small ops around the attention blocks, lane LK phase 2.
// Lane: LK (MiMo) — Ling-on-superroc overnight build. Do not confuse with LG's drafts.
// See ling3_ops.hpp for scope. RMSNorm matches ggml-cuda rms_norm_f32 (float accumulation).

#include "ling3_ops.hpp"

#include "ling3_quant.hpp"   // LX's decoder: -I../ling-moe on the box, -Imactest/vendor on the Mac

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define LING3_HIP_CHECK(expr)                                                        \
    do {                                                                             \
        hipError_t _err = (expr);                                                    \
        if (_err != hipSuccess) {                                                    \
            fprintf(stderr, "ling3_ops: %s failed at %s:%d: %s\n", #expr, __FILE__,  \
                    __LINE__, hipGetErrorString(_err));                              \
            abort();                                                                 \
        }                                                                            \
    } while (0)

namespace ling3 {

template <int BLOCK>
__device__ __forceinline__ float block_sum(float v, float * smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) {
            smem[threadIdx.x] += smem[threadIdx.x + off];
        }
        __syncthreads();
    }

    // Snapshot the result before returning. The caller may reuse this shared buffer for
    // the very next reduction; without this barrier the first thread back would write
    // smem[threadIdx.x] while the others are still reading smem[0].
    const float result = smem[0];
    __syncthreads();
    return result;
}


// ---------------------------------------------------------------------------
// RMSNorm: y = x * rsqrtf(mean(x*x) + eps) * w
// ---------------------------------------------------------------------------
static __global__ void rms_norm_kernel(const float * __restrict__ x,
                                       const float * __restrict__ w,
                                       float * __restrict__ y,
                                       int n, float eps,
                                       int64_t x_tok, int64_t y_tok) {
    __shared__ float s_sum[256];

    const int row = blockIdx.x;
    const float * xr = x + (int64_t) row * x_tok;
    float *       yr = y + (int64_t) row * y_tok;

    float tmp = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = xr[i];
        tmp += xi * xi;
    }
    tmp = block_sum<256>(tmp, s_sum);

    const float mean  = tmp / (float) n;
    const float scale = rsqrtf(mean + eps);

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = xr[i];
        yr[i] = w ? (scale * xi) * w[i] : scale * xi;
    }
}

// LS2 PHASE7: residual add fused with the RMSNorm that consumes it -- the shape every block has.
// As two kernels it cost an extra dispatch per layer (42/token) and a second trip to global
// memory for a value the first kernel already had in a register. xi is the same a[i]+b[i], the
// sum of squares accumulates in the same thread-strided order, and the output is the same
// (scale * xi) * w[i].
//
// `rsum` receives a[i] + b[i] as well: the block's SECOND residual add (ffn_in + ffn_out -> x)
// reads it, and the LING3_GPU_TRACE hook prints it as `ffn_inp`, one of the tensors the trace
// gate compares. May be null.
static __global__ void add_rms_norm_kernel(const float * __restrict__ a,
                                           const float * __restrict__ b,
                                           const float * __restrict__ w,
                                           float * __restrict__ y,
                                           float * __restrict__ rsum,
                                           int n, float eps,
                                           int64_t a_tok, int64_t b_tok, int64_t y_tok) {
    __shared__ float s_sum[256];
    const int row = blockIdx.x;
    const float * ar = a + (int64_t) row * a_tok;
    const float * br = b + (int64_t) row * b_tok;
    float *       yr = y + (int64_t) row * y_tok;

    float tmp = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = ar[i] + br[i];
        if (rsum) rsum[i] = xi;
        tmp += xi * xi;
    }
    tmp = block_sum<256>(tmp, s_sum);
    const float mean  = tmp / (float) n;
    const float scale = rsqrtf(mean + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = ar[i] + br[i];
        yr[i] = w ? (scale * xi) * w[i] : scale * xi;
    }
}

// ---------------------------------------------------------------------------
// residual add
// ---------------------------------------------------------------------------
static __global__ void residual_add_kernel(const float * __restrict__ a,
                                           const float * __restrict__ b,
                                           float * __restrict__ y,
                                           int n,
                                           int64_t a_tok, int64_t b_tok, int64_t y_tok) {
    const int row = blockIdx.x;
    const float * ar = a + (int64_t) row * a_tok;
    const float * br = b + (int64_t) row * b_tok;
    float *       yr = y + (int64_t) row * y_tok;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        yr[i] = ar[i] + br[i];
    }
}

// LS2 PHASE7. This ran ling3_dequant_row over a whole 2,560-wide Q6_K row on ONE thread: a
// fully serial 2,100-byte unpack, measured at 1,583 us for a single row. A quant block is
// self-contained (ling3_term_block reads only inside [blk, blk+type_size)), so the row now
// splits 32 elements per thread -- one eighth of a 256-element Q6_K block -- through the SAME
// ling3_term_block / ling3_term_f32 calls on the SAME arguments. Every output float is identical.
#define LING3_EMB_PER_THREAD 32
static __global__ void embedding_row_kernel(const void * __restrict__ table,
                                            int qtype, int n_embd,
                                            const int * __restrict__ ids,
                                            float * __restrict__ out,
                                            int n_tokens) {
    const int bsz = ling3_blck_size(qtype);
    const int tsz = ling3_type_size(qtype);
    const int t = blockIdx.x;                       // token
    if (t >= n_tokens) return;
    const size_t stride = ling3_row_bytes(qtype, n_embd);
    const unsigned char * row = (const unsigned char *) table + (size_t) ids[t] * stride;
    float * orow = out + (size_t) t * (size_t) n_embd;
    const int base = (int) (blockIdx.y * blockDim.x + threadIdx.x) * LING3_EMB_PER_THREAD;
    if (base >= n_embd) return;
    const unsigned char * blk = row + (size_t) (base / bsz) * (size_t) tsz;
    const int local0 = base % bsz;
    #pragma unroll
    for (int j = 0; j < LING3_EMB_PER_THREAD; ++j) {
        // Q5_K decodes as (d*sc)*q - (dmin*m), which ling3_term cannot express.
        orow[base + j] = (qtype == LING3_Q5_K)
                             ? ling3_q5k_value(blk, local0 + j)
                             : ling3_term_f32(ling3_term_block(qtype, blk, local0 + j));
    }
}

// ---------------------------------------------------------------------------
// argmax over `n` floats per row; ties resolve to the lowest index.
// ---------------------------------------------------------------------------
// LM1 SPEC6 item 3 (RV14 3.2): argmax + top-1 softmax probability in one
// launch. The reduction is the same one argmax_kernel does for the max; the
// probability is a second strided pass of exp(x - max) summed in shared memory.
// p = 1 / sum exp(x - max) is exact to fp32 and needs no logarithm.
static __global__ void argmax_prob_kernel(const float * __restrict__ logits,
                                          int n, int32_t * __restrict__ out_index,
                                          float * __restrict__ out_p,
                                          int64_t logits_tok) {
    __shared__ float s_val[256];
    __shared__ int   s_idx[256];

    const int row  = blockIdx.x;
    const float * lr = logits + (int64_t) row * logits_tok;

    // pass 1: max (bit-identical reduction to argmax_kernel)
    float best_v = -INFINITY;
    int   best_i = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = lr[i];
        if (v > best_v || (v == best_v && i < best_i)) {
            best_v = v;
            best_i = i;
        }
    }
    s_val[threadIdx.x] = best_v;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) {
            const float v = s_val[threadIdx.x + off];
            const int   i = s_idx[threadIdx.x + off];
            if (v > s_val[threadIdx.x] || (v == s_val[threadIdx.x] && i < s_idx[threadIdx.x])) {
                s_val[threadIdx.x] = v;
                s_idx[threadIdx.x] = i;
            }
        }
        __syncthreads();
    }
    // Broadcast the winner to every thread before s_val is reused as the
    // exponent accumulator.
    const float max_v = s_val[0];
    const int   max_i = s_idx[0];
    __syncthreads();

    // pass 2: sum exp(x - max)
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        acc += expf(lr[i] - max_v);
    }
    s_val[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) {
            s_val[threadIdx.x] += s_val[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_index[row] = max_i;
        // out_p may alias into the same allocation as out_index (see hpp).
        out_p[row] = (s_val[0] > 0.0f) ? (1.0f / s_val[0]) : 0.0f;
    }
}

// LA1: two-phase argmax. The old kernel scanned all n floats with ONE 256-thread
// block, so a 157,184-logit row read the 629 KB at ~3.5 GB/s of a 244 GB/s box
// (~179 us). The answer -- max value, lowest index on a tie, NaN never wins, an
// all--INF row returns 0 -- is independent of reduction order, so the row is now
// split into ARGMAX_SLICES contiguous slices each reduced by its own block, then
// the S partials are combined by a tiny second kernel. Bit-identical by
// construction (both stages use the same (v, idx) compare; a tie and the
// all--INF case both fall through to the lowest index).
#define LING3_ARGMAX_SLICES 128

__global__ void argmax_slice_kernel(const float * __restrict__ logits,
                                    int n, int64_t logits_tok,
                                    float * __restrict__ pval,
                                    int * __restrict__ pidx,
                                    int slices) {
    __shared__ float s_val[256];
    __shared__ int   s_idx[256];

    const int row   = (int) blockIdx.y;
    const int slice = (int) blockIdx.x;
    const float * lr = logits + (int64_t) row * logits_tok;

    // Contiguous, non-overlapping, exhaustive slices of [0, n). n may not divide
    // evenly; the arithmetic keeps every element in exactly one slice.
    const int start = (int) (((int64_t) slice * n) / slices);
    const int end   = (int) (((int64_t) (slice + 1) * n) / slices);

    float best_v = -INFINITY;
    int   best_i = start;          // default = first index; the all--INF slice keeps it
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        const float v = lr[i];
        if (v > best_v || (v == best_v && i < best_i)) {
            best_v = v;
            best_i = i;
        }
    }
    s_val[threadIdx.x] = best_v;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) {
            const float v = s_val[threadIdx.x + off];
            const int   i = s_idx[threadIdx.x + off];
            if (v > s_val[threadIdx.x] || (v == s_val[threadIdx.x] && i < s_idx[threadIdx.x])) {
                s_val[threadIdx.x] = v;
                s_idx[threadIdx.x] = i;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int o = row * slices + slice;
        pval[o] = s_val[0];
        pidx[o] = s_idx[0];
    }
}

__global__ void argmax_combine_kernel(const float * __restrict__ pval,
                                      const int * __restrict__ pidx,
                                      int slices, int32_t * __restrict__ out_index) {
    __shared__ float s_val[256];
    __shared__ int   s_idx[256];

    const int row = (int) blockIdx.x;
    const float * pv = pval + (size_t) row * slices;
    const int   * pi = pidx + (size_t) row * slices;

    float best_v = -INFINITY;
    int   best_i = 0;
    for (int i = threadIdx.x; i < slices; i += blockDim.x) {
        const float v = pv[i];
        const int   idx = pi[i];
        if (v > best_v || (v == best_v && idx < best_i)) {
            best_v = v;
            best_i = idx;
        }
    }
    s_val[threadIdx.x] = best_v;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) {
            const float v = s_val[threadIdx.x + off];
            const int   i = s_idx[threadIdx.x + off];
            if (v > s_val[threadIdx.x] || (v == s_val[threadIdx.x] && i < s_idx[threadIdx.x])) {
                s_val[threadIdx.x] = v;
                s_idx[threadIdx.x] = i;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_index[row] = s_idx[0];
    }
}

// ---------------------------------------------------------------------------
// host launchers
// ---------------------------------------------------------------------------
void ops_rms_norm(const float * x, const float * w, float * y,
                  int n, int rows, float eps,
                  int64_t x_tok, int64_t y_tok, hipStream_t stream) {
    const int blocks = rows > 0 ? rows : 1;
    hipLaunchKernelGGL(rms_norm_kernel, dim3(blocks), 256, 0, stream,
                       x, w, y, n, eps, x_tok, y_tok);
    LING3_HIP_CHECK(hipGetLastError());
}

void ops_add_rms_norm(const float * a, const float * b, const float * w, float * y,
                      float * rsum, int n, int rows, float eps,
                      int64_t a_tok, int64_t b_tok, int64_t y_tok, hipStream_t stream) {
    const int blocks = rows > 0 ? rows : 1;
    hipLaunchKernelGGL(add_rms_norm_kernel, dim3(blocks), 256, 0, stream,
                       a, b, w, y, rsum, n, eps, a_tok, b_tok, y_tok);
    LING3_HIP_CHECK(hipGetLastError());
}

void ops_final_norm(const float * hidden, const float * output_norm_w, float * y,
                    int n_embd, int rows, hipStream_t stream) {
    ops_rms_norm(hidden, output_norm_w, y, n_embd, rows, LING3_RMS_EPS, stream);
}

void ops_residual_add(const float * a, const float * b, float * y,
                      int n, int rows,
                      int64_t a_tok, int64_t b_tok, int64_t y_tok, hipStream_t stream) {
    const int blocks = rows > 0 ? rows : 1;
    hipLaunchKernelGGL(residual_add_kernel, dim3(blocks), 256, 0, stream,
                       a, b, y, n, a_tok, b_tok, y_tok);
    LING3_HIP_CHECK(hipGetLastError());
}

void ops_embedding_row(int token_embd_type, const void * token_embd, int n_embd,
                       int n_vocab, const int * token_ids, int n_tokens,
                       float * out, hipStream_t stream) {
    (void) n_vocab;
    const int bsz = ling3_blck_size(token_embd_type);
    if (bsz <= 0 || n_embd % bsz != 0) {
        fprintf(stderr,
                "ling3_ops: token_embd type %d has block size %d, which does not divide n_embd %d\n",
                token_embd_type, bsz, n_embd);
        abort();
    }
    const int block = 128;
    const int chunks = (n_embd + LING3_EMB_PER_THREAD - 1) / LING3_EMB_PER_THREAD;
    const dim3 grid(n_tokens > 0 ? n_tokens : 1, (chunks + block - 1) / block);
    hipLaunchKernelGGL(embedding_row_kernel, grid, block, 0, stream,
                       token_embd, token_embd_type, n_embd, token_ids, out, n_tokens);
    LING3_HIP_CHECK(hipGetLastError());
}

// Persistent scratch for the two-phase argmax: LING3_ARGMAX_SLICES (val, idx)
// partials per row. Lazy growth, like the other per-call scratch buffers; argmax
// runs once per decode token, so a per-call av_malloc would cost us launches.
static float * g_ama_pval = nullptr;
static int   * g_ama_pidx = nullptr;
static int     g_ama_pairs = 0;   // capacity, in partials (one (val,idx) pair)

void ops_argmax(const float * logits, int n, int rows, int32_t * out_index,
                int64_t logits_tok, hipStream_t stream) {
    const int blocks = rows > 0 ? rows : 1;
    const int need = blocks * LING3_ARGMAX_SLICES;
    if (need > g_ama_pairs || g_ama_pval == nullptr || g_ama_pidx == nullptr) {
        if (g_ama_pval) { hipFree(g_ama_pval); g_ama_pval = nullptr; }
        if (g_ama_pidx) { hipFree(g_ama_pidx); g_ama_pidx = nullptr; }
        g_ama_pairs = 0;
        hipError_t e = hipMalloc(&g_ama_pval, (size_t) need * sizeof(float));
        if (e == hipSuccess) e = hipMalloc(&g_ama_pidx, (size_t) need * sizeof(int));
        if (e != hipSuccess) {
            if (g_ama_pval) { hipFree(g_ama_pval); g_ama_pval = nullptr; }
            if (g_ama_pidx) { hipFree(g_ama_pidx); g_ama_pidx = nullptr; }
            fprintf(stderr, "ling3_ops: argmax scratch alloc (%d partials) failed: %s\n",
                    need, hipGetErrorString(e));
            abort();
        }
        g_ama_pairs = need;
    }
    const dim3 grid(LING3_ARGMAX_SLICES, blocks);
    hipLaunchKernelGGL(argmax_slice_kernel, grid, 256, 0, stream,
                       logits, n, logits_tok, g_ama_pval, g_ama_pidx,
                       LING3_ARGMAX_SLICES);
    LING3_HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(argmax_combine_kernel, dim3(blocks), 256, 0, stream,
                       g_ama_pval, g_ama_pidx, LING3_ARGMAX_SLICES, out_index);
    LING3_HIP_CHECK(hipGetLastError());
}

// LM1 SPEC6 item 3: see the hpp note. `out_p` may alias out_index + 1 so both
// results land in one 8-byte device word for a single memcpy.
void ops_argmax_prob(const float * logits, int n, int rows, int32_t * out_index,
                     float * out_p, int64_t logits_tok, hipStream_t stream) {
    const int blocks = rows > 0 ? rows : 1;
    hipLaunchKernelGGL(argmax_prob_kernel, dim3(blocks), 256, 0, stream,
                       logits, n, out_index, out_p, logits_tok);
    LING3_HIP_CHECK(hipGetLastError());
}

const char * ops_version() { return "ling3-ops/1.0"; }

} // namespace ling3
