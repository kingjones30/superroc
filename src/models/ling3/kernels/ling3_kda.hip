// ling3_kda.hip — KDA kernels for Ling-3.0 (bailingmoe3), lane LK.
// Lane: LK (MiMo) — Ling-on-gufo overnight build. Do not confuse with LG's drafts.
//
// Arithmetic follows the fork exactly:
//   conv     ggml_compute_forward_ssm_conv_f32   (tap 0 = oldest stream sample)
//   silu     x / (1 + expf(-x))                  (ggml_cuda_op_silu_single)
//   sigmoid  1 / (1 + expf(-x))                  (ggml_cuda_op_sigmoid_single)
//   l2_norm  rsqrtf(fmaxf(sum(x*x), eps*eps))    (l2_norm_f32)
//   rms_norm rsqrtf(sum(x*x)/n + eps)            (rms_norm_f32)
//   scan     gated_delta_net_cuda<KDA=true>      (per-token recurrence, decay on the key axis)
//
// See ling3_kda.hpp for the layouts and the scope statement.

#include "ling3_kda.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define LING3_HIP_CHECK(expr)                                                        \
    do {                                                                             \
        hipError_t _err = (expr);                                                    \
        if (_err != hipSuccess) {                                                    \
            fprintf(stderr, "ling3_kda: %s failed at %s:%d: %s\n", #expr, __FILE__,  \
                    __LINE__, hipGetErrorString(_err));                              \
            abort();                                                                 \
        }                                                                            \
    } while (0)

namespace ling3 {

// ---------------------------------------------------------------------------
// small device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float silu_f(float x)    { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

// tree reduction in shared memory — wave-size independent (no __shfl)
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
// (1) causal conv + SiLU, with conv-state carry
//
//   stream[p]        = p < d_conv-1 ? conv_state[c*(d_conv-1)+p] : x[p-(d_conv-1)]
//   conv_out[t]      = silu( sum_{tap=0}^{d_conv-1} w[tap + c*d_conv] * stream[t + tap] )
//   conv_state[tap]  = stream[n_tokens + tap]        (the last d_conv-1 stream samples)
//
// conv_state points at the qkv slice of one sequence's state row:
//   base + s*embd_r + qkv*(d_conv-1)*d_inner + c*(d_conv-1) + tap
// ---------------------------------------------------------------------------
// LS2 PHASE7. One THREAD per channel, and q/k/v in ONE launch: blockIdx.z picks the view AND
// the conv-state slice. 4,096 blocks of 256 threads with 255 lanes idle became 16x3 blocks with
// every lane working, and 3 launches per layer became 1.
//
// THE FIX (Jay, PHASE7). The state update below must read st[p], NOT st[tap]:
//
//     original:  st[tap] = stream_at(n_tokens + tap)      with stream_at(p) = p < hist ? st[p] : ...
//     mine:      st[tap] = (p < hist) ? st[tap] : ...      <-- read st[tap], i.e. itself
//
// With n_tokens == 1 (every decode step) p = 1, 2 are both < hist = 3, so the original shifted
// st[1], st[2] DOWN into st[0], st[1] and mine wrote each slot back to itself. The history never
// advanced: from a zero state the two oldest taps stayed 0 forever, and after a batched prefill
// they froze at the prefill's values. Only st[2] ever updated. Prefill never hit it because
// n_tokens >= hist makes every p >= hist, so the p < hist branch is dead there -- which is why
// trace.sh and a batched-prefill gate both looked clean.
//
// The read stays ahead of the write (p = n_tokens + tap > tap), which is what makes the in-place
// shift safe for one lane owning st[c*hist .. c*hist+hist-1]. That is also why the original
// needed its single-lane serialisation: a lane owns the run, so no barrier is needed here.
//
// LC1 (wave 3, 2026-10-03). The per-output computation lives in ONE function,
// conv_silu_out_at, that BOTH this serial kernel and conv_silu_run_kernel call, so the
// compiler emits the same FMA contraction for both and the token-parallel path is
// bit-identical by construction: same tap order, same `acc += w * sv` statement, same index
// types, same silu_f. It takes the d_conv stream values as input (sv[0..d_conv-1] = stream[t
// .. t + d_conv - 1]), so the run kernel can keep them in a shared-memory window -- each input
// element leaving DRAM ONCE per run of tokens -- while this serial kernel gathers them fresh
// into a small local window. conv_silu_state_at is the serial body's trailing state loop,
// unchanged -- including st[p] (NOT st[tap]: 28g reads st[n_tokens + tap], and that is what the
// window shift is; it matters for 2 <= n_tokens < hist). n_tokens == 1 keeps this serial kernel.
// ---------------------------------------------------------------------------
static constexpr int kConvMaxD = 8;   // d_conv must fit in [2, kConvMaxD] (Ling is 4)

__device__ __forceinline__ float conv_silu_stream_at(const float * __restrict__ st,
                                                    const float * __restrict__ xr,
                                                    int hist, int64_t x_tok, int c, int64_t p) {
    return p < hist ? st[p] : xr[c + (p - hist) * x_tok];
}

// LC1 SPEC2: the conv writes each output exactly once and never reads it back, so the store is
// STREAMING (non-temporal). The default write-back policy makes the GPU fetch the cache line
// before overwriting it (read-for-ownership); MEASURED that dominated the call -- 222.7 of
// 276.7 us at n_tokens 512 (probe: identical loads/stores with the store deleted = 54.0 us).
// The VALUE stored is unchanged: bit-identity is about the value, not the cache policy.
__device__ __forceinline__ void conv_silu_store(float * __restrict__ p, float v) {
    __builtin_nontemporal_store(v, p);
}

__device__ __forceinline__ void conv_silu_out_at(
        const float * __restrict__ w, float * __restrict__ yr,
        int d_conv, int c, int64_t y_tok, int t,
        const float * __restrict__ sv) {
    float acc = 0.0f;
    for (int tap = 0; tap < d_conv; ++tap) {
        acc += w[tap + (int64_t) c * d_conv] * sv[tap];
    }
    conv_silu_store(&yr[c + (int64_t) t * y_tok], silu_f(acc));
}

// LC1: the serial body's trailing state loop, verbatim. One thread owns the whole hist-run of
// one channel and must walk the taps in increasing order: the read index p = n_tokens + tap is
// always ahead of the write index tap (n_tokens >= 1), which is what makes the in-place shift safe.
__device__ __forceinline__ void conv_silu_state_at(
        const float * __restrict__ x, float * __restrict__ conv_state,
        int d_conv, int d_inner, int n_tokens,
        int64_t x_tok, int64_t x_seq,
        int c, int s) {
    const int hist = d_conv - 1;
    float *       st = conv_state + (int64_t) s * 3 * hist * d_inner + (int64_t) c * hist;
    const float * xr = x + (int64_t) s * x_seq;

    for (int tap = 0; tap < hist; ++tap) {
        const int64_t p = (int64_t) n_tokens + tap;
        st[tap] = p < hist ? st[p] : xr[c + (p - hist) * x_tok];   // <-- st[p], the fix
    }
}

__device__ __forceinline__ void conv_silu_body(
        const float * __restrict__ x, float * __restrict__ y,
        float * __restrict__ conv_state, const float * __restrict__ w,
        int d_conv, int d_inner, int n_tokens,
        int64_t x_tok, int64_t x_seq, int64_t y_tok, int64_t y_seq,
        int c, int s) {
    const int hist = d_conv - 1;
    const float * st = conv_state + (int64_t) s * 3 * hist * d_inner + (int64_t) c * hist;
    const float * xr = x + (int64_t) s * x_seq;
    float *       yr = y + (int64_t) s * y_seq;
    float sv[kConvMaxD];

    for (int t = 0; t < n_tokens; ++t) {
        for (int tap = 0; tap < d_conv; ++tap) {
            sv[tap] = conv_silu_stream_at(st, xr, hist, x_tok, c, (int64_t) t + tap);
        }
        conv_silu_out_at(w, yr, d_conv, c, y_tok, t, sv);
    }
    conv_silu_state_at(x, conv_state, d_conv, d_inner, n_tokens, x_tok, x_seq, c, s);
}

static __global__ void conv_silu_kernel(const float * __restrict__ xq,
                                        const float * __restrict__ xk,
                                        const float * __restrict__ xv,
                                        float * __restrict__ yq,
                                        float * __restrict__ yk,
                                        float * __restrict__ yv,
                                        float * __restrict__ conv_state,
                                        const float * __restrict__ wq,
                                        const float * __restrict__ wk,
                                        const float * __restrict__ wv,
                                        int d_conv, int d_inner, int n_tokens,
                                        int64_t qi_tok, int64_t ki_tok, int64_t vi_tok,
                                        int64_t qi_seq,
                                        int64_t qo_tok, int64_t ko_tok, int64_t vo_tok,
                                        int64_t qo_seq) {
    const int c = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= d_inner) return;                     // no __syncthreads in this kernel
    const int which = (int) blockIdx.z;          // 0 = q, 1 = k, 2 = v
    const int s = (int) blockIdx.y;
    const int hist = d_conv - 1;
    float * st = conv_state + (size_t) which * (size_t) hist * (size_t) d_inner;
    switch (which) {
        case 0: conv_silu_body(xq, yq, st, wq, d_conv, d_inner, n_tokens, qi_tok, qi_seq, qo_tok, qo_seq, c, s); break;
        case 1: conv_silu_body(xk, yk, st, wk, d_conv, d_inner, n_tokens, ki_tok, qi_seq, ko_tok, qo_seq, c, s); break;
        default: conv_silu_body(xv, yv, st, wv, d_conv, d_inner, n_tokens, vi_tok, qi_seq, vo_tok, qo_seq, c, s); break;
    }
}

// ---------------------------------------------------------------------------
// LC1 SPEC2: run-parallel output pass (n_tokens > 1). One THREAD per channel; a block covers
// kRunChan channels x kRun consecutive tokens. Each thread keeps the last d_conv stream values
// in REGISTERS and slides the window one token at a time, so every input element leaves DRAM
// once instead of once per tap -- that is the whole speedup. This is the shape of the fork's
// ssm_conv_long_token_f32<false,128,4,32> (ggml/src/ggml-cuda/ssm-conv.cu): one thread owns one
// channel and a RUN of consecutive tokens. The earlier one-thread-per-(channel, token) map
// re-read each input d_conv times at the memory roof and measured 1.09x (616 vs 671 us/call).
//
// The per-output expression is the shared conv_silu_out_at: same tap order, same
// acc += w[tap + c*d_conv] * sv[tap] statement, same silu_f, so the compiler emits the same
// FMA contraction the serial kernel does and the two paths stay bit-identical. Only the loads
// are shared -- which is also why the run is bounded (kRun tokens) rather than the whole chunk:
// a run keeps enough blocks on the box to saturate DRAM while still reading each input once.
//
// d_conv is a template parameter (as in the fork) so the tap loop, the window slide and the
// weight loads all unroll into registers; a runtime d_conv would push the window through
// scratch memory and the slide would not unroll. conv_silu_check_d_conv() bounds it to [2,8].
//
// Writes no state: conv_silu_state_kernel runs after this on the same stream, so every output
// has been read out of the OLD state first, exactly the order the serial body has.
// ---------------------------------------------------------------------------
static constexpr int kRunChan = 128;   // channels per block (= blockDim.x)
static constexpr int kRun     = 32;    // tokens per block

template <int kDConv>
static __global__ void conv_silu_run_kernel(const float * __restrict__ x,
                                            float * __restrict__ y,
                                            float * __restrict__ conv_state,
                                            const float * __restrict__ w,
                                            int d_inner, int n_tokens, int n_seqs,
                                            int which,
                                            int64_t x_tok, int64_t x_seq,
                                            int64_t y_tok, int64_t y_seq) {
    const int hist    = kDConv - 1;
    const int c_local = (int) threadIdx.x;
    const int c0      = (int) blockIdx.x * kRunChan;
    const int t0      = (int) blockIdx.y * kRun;
    const int s       = (int) blockIdx.z;
    const int c       = c0 + c_local;
    if (c >= d_inner || s >= n_seqs) return;
    const int left = n_tokens - t0;
    const int local_n_t = (t0 < n_tokens) ? (left < kRun ? left : kRun) : 0;
    if (local_n_t <= 0) return;

    const float * st = conv_state + (size_t) which * (size_t) hist * (size_t) d_inner
                                   + (int64_t) s * 3 * hist * d_inner
                                   + (int64_t) c * hist;
    const float * xr = x + (int64_t) s * x_seq;
    float *       yr = y + (int64_t) s * y_seq;

    // The window this run of outputs needs: stream[t0 .. t0 + local_n_t + hist - 1]. It is
    // primed once (d_conv loads) and then slides: each step retires the oldest tap and pulls
    // exactly one new input, so the run moves local_n_t + hist values -- one read per element.
    float sv[kDConv];
    #pragma unroll
    for (int tap = 0; tap < kDConv; ++tap) {
        sv[tap] = conv_silu_stream_at(st, xr, hist, x_tok, c, (int64_t) t0 + tap);
    }

    #pragma unroll
    for (int i = 0; i < kRun; ++i) {
        if (i < local_n_t) {
            conv_silu_out_at(w, yr, kDConv, c, y_tok, t0 + i, sv);
            if (i + 1 < local_n_t) {
                const float nv =
                    conv_silu_stream_at(st, xr, hist, x_tok, c, (int64_t) t0 + i + kDConv);
                #pragma unroll
                for (int tap = 0; tap < kDConv - 1; ++tap) {
                    sv[tap] = sv[tap + 1];
                }
                sv[kDConv - 1] = nv;
            }
        }
    }
}

// One place that turns sh.d_conv into a kernel instantiation, so both launchers (the qkv
// entry and the one-stream entry) pick the same code path. conv_silu_check_d_conv() has
// already bounded d_conv to [2, kConvMaxD].
template <int kDConv>
static void conv_silu_run_launch_t(const dim3 & grid,
                                   const float * x, float * y,
                                   float * conv_state, const float * w,
                                   int d_inner, int n_tokens, int n_seqs, int which,
                                   int64_t x_tok, int64_t x_seq,
                                   int64_t y_tok, int64_t y_seq,
                                   hipStream_t stream) {
    hipLaunchKernelGGL(conv_silu_run_kernel<kDConv>, grid, kRunChan, 0, stream,
                       x, y, conv_state, w, d_inner, n_tokens, n_seqs, which,
                       x_tok, x_seq, y_tok, y_seq);
    LING3_HIP_CHECK(hipGetLastError());
}

static void conv_silu_run_launch(int d_conv, const dim3 & grid,
                                 const float * x, float * y,
                                 float * conv_state, const float * w,
                                 int d_inner, int n_tokens, int n_seqs, int which,
                                 int64_t x_tok, int64_t x_seq,
                                 int64_t y_tok, int64_t y_seq,
                                 hipStream_t stream) {
    switch (d_conv) {
        case 2: conv_silu_run_launch_t<2>(grid, x, y, conv_state, w, d_inner, n_tokens,
                                          n_seqs, which, x_tok, x_seq, y_tok, y_seq, stream); break;
        case 3: conv_silu_run_launch_t<3>(grid, x, y, conv_state, w, d_inner, n_tokens,
                                          n_seqs, which, x_tok, x_seq, y_tok, y_seq, stream); break;
        case 4: conv_silu_run_launch_t<4>(grid, x, y, conv_state, w, d_inner, n_tokens,
                                          n_seqs, which, x_tok, x_seq, y_tok, y_seq, stream); break;
        case 5: conv_silu_run_launch_t<5>(grid, x, y, conv_state, w, d_inner, n_tokens,
                                          n_seqs, which, x_tok, x_seq, y_tok, y_seq, stream); break;
        case 6: conv_silu_run_launch_t<6>(grid, x, y, conv_state, w, d_inner, n_tokens,
                                          n_seqs, which, x_tok, x_seq, y_tok, y_seq, stream); break;
        case 7: conv_silu_run_launch_t<7>(grid, x, y, conv_state, w, d_inner, n_tokens,
                                          n_seqs, which, x_tok, x_seq, y_tok, y_seq, stream); break;
        default: conv_silu_run_launch_t<8>(grid, x, y, conv_state, w, d_inner, n_tokens,
                                           n_seqs, which, x_tok, x_seq, y_tok, y_seq, stream); break;
    }
}

// LC1: state pass (n_tokens > 1), the second launch of the prefill split. One thread per channel
// applies conv_silu_state_at -- the serial body's trailing update -- with the same slice selection.
// It runs after conv_silu_run_kernel on the same stream, so every output has already been read
// out of the OLD state, exactly the order the serial body has.
static __global__ void conv_silu_state_kernel(const float * __restrict__ xq,
                                              const float * __restrict__ xk,
                                              const float * __restrict__ xv,
                                              float * __restrict__ conv_state,
                                              int d_conv, int d_inner, int n_tokens, int n_seqs,
                                              int64_t qi_tok, int64_t ki_tok, int64_t vi_tok,
                                              int64_t qi_seq) {
    const int c = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= d_inner) return;
    const int zs = (int) blockIdx.y;
    const int which = zs / n_seqs;
    const int s     = zs - which * n_seqs;
    const int hist = d_conv - 1;
    float * st = conv_state + (size_t) which * (size_t) hist * (size_t) d_inner;
    switch (which) {
        case 0: conv_silu_state_at(xq, st, d_conv, d_inner, n_tokens, qi_tok, qi_seq, c, s); break;
        case 1: conv_silu_state_at(xk, st, d_conv, d_inner, n_tokens, ki_tok, qi_seq, c, s); break;
        default: conv_silu_state_at(xv, st, d_conv, d_inner, n_tokens, vi_tok, qi_seq, c, s); break;
    }
}

// ---------------------------------------------------------------------------
// (2a) decay gate  g1 = gate_lower_bound * sigmoid((f + dt_bias) * a_exp[h])
//      a_exp holds exp(A_log); the fork multiplies BEFORE the sigmoid
// ---------------------------------------------------------------------------
static __global__ void gate_kernel(const float * __restrict__ f,
                                   float * __restrict__ g1,
                                   const float * __restrict__ a_exp,
                                   const float * __restrict__ dt_bias,
                                   float lower_bound,
                                   int head_dim, int n_tokens, int n_seqs, int d_inner,
                                   int64_t f_tok, int64_t f_seq,
                                   int64_t g_tok, int64_t g_seq) {
    // LS2 PHASE7: this was grid(d_inner) blocks of 256 threads with the body looping
    // `for (idx = threadIdx.x; idx < n_tokens * n_seqs; ...)`. On the decode path that total is
    // 1, so 255 of every 256 lanes did nothing: 4,096 blocks to compute 4,096 elements.
    // One THREAD per channel now. Index expression, sigmoid_f and multiply order unchanged.
    const int c = (int) (blockIdx.x * blockDim.x + threadIdx.x);   // 0..d_inner-1
    if (c >= d_inner) return;
    const float a = a_exp[c / head_dim];

    const int total = n_tokens * n_seqs;
    for (int idx = 0; idx < total; ++idx) {
        const int t = idx % n_tokens;
        const int s = idx / n_tokens;
        const float fv = f[c + (int64_t) t * f_tok + (int64_t) s * f_seq] + dt_bias[c];
        g1[c + (int64_t) t * g_tok + (int64_t) s * g_seq] = lower_bound * sigmoid_f(fv * a);
    }
}

// ---------------------------------------------------------------------------
// (2b) beta = sigmoid(beta_raw), [n_head, n_tokens, n_seqs]
// ---------------------------------------------------------------------------
static __global__ void beta_kernel(const float * __restrict__ b_in, float * __restrict__ b_out,
                                   int n_tokens, int n_seqs,
                                   int64_t i_tok, int64_t i_seq,
                                   int64_t o_tok, int64_t o_seq) {
    const int h = blockIdx.x;
    const int total = n_tokens * n_seqs;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int t = idx % n_tokens;
        const int s = idx / n_tokens;
        b_out[h + (int64_t) t * o_tok + (int64_t) s * o_seq] =
            sigmoid_f(b_in[h + (int64_t) t * i_tok + (int64_t) s * i_seq]);
    }
}

// ---------------------------------------------------------------------------
// L2 norm along head_dim (one row per (head, token, seq))
//   scale = rsqrtf(fmaxf(sum(x*x), eps*eps))
// ---------------------------------------------------------------------------
static __global__ void l2_norm_kernel(const float * __restrict__ x, float * __restrict__ y,
                                      int head_dim, float eps,
                                      int n_tokens, int n_seqs,
                                      int64_t x_tok, int64_t x_seq,
                                      int64_t y_tok, int64_t y_seq) {
    __shared__ float s_sum[256];

    const int h = blockIdx.x;
    const int idx = blockIdx.y;               // token*n_seqs + seq
    const int t = idx % n_tokens;
    const int s = idx / n_tokens;

    const float * xr = x + h * head_dim + (int64_t) t * x_tok + (int64_t) s * x_seq;
    float *       yr = y + h * head_dim + (int64_t) t * y_tok + (int64_t) s * y_seq;

    float tmp = 0.0f;
    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
        const float xi = xr[col];
        tmp += xi * xi;
    }
    tmp = block_sum<256>(tmp, s_sum);

    const float scale = rsqrtf(fmaxf(tmp, eps * eps));
    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
        yr[col] = scale * xr[col];
    }
}

// ---------------------------------------------------------------------------
// (3) gated delta-rule scan
//
// Per token t, per head h, thread j owns the value column j of S (S[key, value],
// the column being S[0..127][j] held in registers):
//   eg[i]   = expf(g1[i, t])            -- decay along the KEY axis i
//   kv      = sum_i eg[i] * S[i][j] * k[i]
//   delta   = (v[j] - kv) * beta
//   S[i][j] = eg[i] * S[i][j] + k[i] * delta
//   o[j]    = scale * sum_i S[i][j] * q[i]      (POST-update state, as in the fork)
//
// No inter-thread reduction inside the token loop: each column is private to its
// thread. eg[] / beta are shared across the block's 128 columns.
// ---------------------------------------------------------------------------
static __global__ void scan_kernel(const float * __restrict__ q,
                                   const float * __restrict__ k,
                                   const float * __restrict__ v,
                                   const float * __restrict__ g1,
                                   const float * __restrict__ beta,
                                   float * __restrict__ out,
                                   float * __restrict__ ssm_state,
                                   int n_head, int n_tokens, int n_seqs,
                                   float scale,
                                   int64_t q_tok, int64_t q_seq,
                                   int64_t v_tok, int64_t v_seq,
                                   int64_t b_tok, int64_t b_seq,
                                   int64_t o_tok, int64_t o_seq) {
    constexpr int HD = 128;

    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int j = threadIdx.x;                 // value channel

    const int64_t chan = (int64_t) h * HD;

    float * srow = ssm_state + (((int64_t) s * n_head + h) * HD + j) * HD;

    float S[HD];
    #pragma unroll
    for (int i = 0; i < HD; ++i) {
        S[i] = srow[i];
    }

    __shared__ float eg[HD];
    __shared__ float sbeta;

    for (int t = 0; t < n_tokens; ++t) {
        const float * krow = k + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;
        const float * qrow = q + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;
        const float * grow = g1 + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;

        // thread j publishes eg[j] = exp(g[j]); g is indexed by the key channel, and
        // every thread needs all 128 values, so this fills the shared table.
        eg[j] = expf(grow[j]);
        if (j == 0) {
            sbeta = beta[h + (int64_t) t * b_tok + (int64_t) s * b_seq];
        }
        __syncthreads();
        const float beta_val = sbeta;

        float kv = 0.0f;
        #pragma unroll
        for (int i = 0; i < HD; ++i) {
            kv += eg[i] * S[i] * krow[i];
        }

        const float delta =
            (v[chan + j + (int64_t) t * v_tok + (int64_t) s * v_seq] - kv) * beta_val;

        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < HD; ++i) {
            S[i] = eg[i] * S[i] + krow[i] * delta;
            acc += S[i] * qrow[i];
        }

        out[chan + j + (int64_t) t * o_tok + (int64_t) s * o_seq] = acc * scale;

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < HD; ++i) {
        srow[i] = S[i];
    }
}

// ---------------------------------------------------------------------------
// (4) RMSNorm(o, o_norm) * sigmoid(g2)
//   y = (rsqrtf(sum(x*x)/n + eps) * x) * w * sigmoid(g2)
// ---------------------------------------------------------------------------
static __global__ void out_norm_gate_kernel(const float * __restrict__ o_in,
                                            const float * __restrict__ g2,
                                            float * __restrict__ out,
                                            const float * __restrict__ w,
                                            int head_dim, float eps,
                                            int n_tokens, int n_seqs,
                                            int64_t i_tok, int64_t i_seq,
                                            int64_t g_tok, int64_t g_seq,
                                            int64_t o_tok, int64_t o_seq) {
    __shared__ float s_sum[256];

    const int h = blockIdx.x;
    const int idx = blockIdx.y;
    const int t = idx % n_tokens;
    const int s = idx / n_tokens;

    const float * xr = o_in + h * head_dim + (int64_t) t * i_tok + (int64_t) s * i_seq;
    const float * gr = g2   + h * head_dim + (int64_t) t * g_tok + (int64_t) s * g_seq;
    float *       yr = out  + h * head_dim + (int64_t) t * o_tok + (int64_t) s * o_seq;

    float tmp = 0.0f;
    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
        const float xi = xr[col];
        tmp += xi * xi;
    }
    tmp = block_sum<256>(tmp, s_sum);

    const float mean  = tmp / (float) head_dim;
    const float scale = rsqrtf(mean + eps);

    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
        yr[col] = (scale * xr[col]) * w[col] * sigmoid_f(gr[col]);
    }
}

static __global__ void zero_kernel(float * __restrict__ p, size_t n) {
    const size_t stride = (size_t) blockDim.x * gridDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        p[i] = 0.0f;
    }
}

// ---------------------------------------------------------------------------
// host launchers
// ---------------------------------------------------------------------------
static int grid_1d(int n) { return n > 0 ? n : 1; }

// LC1: which path the last conv call took (0 = the serial kernel, 1 = the token-parallel
// prefill pair). Diagnostic only: the parity test reads it so it cannot compare the serial
// path with itself the way LG1's did.
static std::atomic<int> g_conv_silu_path{0};

int kda_conv_silu_last_path() {
    return g_conv_silu_path.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// LC1: host-side index audit (the lf_audit pattern). Every index the two prefill kernels
// compute is enumerated against the ALLOCATION each pointer lives in (hipMemGetAddressRange,
// which returns the base and size of the block the pointer is inside, so a slice of a bigger
// arena -- conv_state is one -- is checked against the arena). An out-of-bounds launch is
// refused on the host, before anything reaches the GPU. Returns false when an extent cannot
// be established or is overrun: the caller then keeps the serial kernel, which is still
// bit-identical, just the old speed.
// ---------------------------------------------------------------------------
static bool conv_in_alloc(const void * p, int64_t n_elems, const char * what) {
    if (p == nullptr || n_elems <= 0) {
        fprintf(stderr, "ling3_kda: REFUSING launch, %s: null pointer or non-positive extent (%lld)\n",
                what, (long long) n_elems);
        return false;
    }
    void * base = nullptr;
    size_t bytes = 0;
    void * probe = const_cast<void *>(p);
    if (hipMemGetAddressRange(&base, &bytes, probe) != hipSuccess || base == nullptr) {
        fprintf(stderr, "ling3_kda: %s: cannot establish the allocation behind %p; keeping the serial kernel\n",
                what, p);
        return false;
    }
    const std::uintptr_t cp = (std::uintptr_t) p;
    const std::uintptr_t cb = (std::uintptr_t) base;
    const unsigned long long off  = (unsigned long long) (cp - cb);
    const unsigned long long need = (unsigned long long) n_elems * sizeof(float);
    if (cp < cb || need > (unsigned long long) bytes ||
        off > (unsigned long long) bytes - need) {
        fprintf(stderr,
                "ling3_kda: REFUSING launch, %s: %lld floats at +%llu of a %zu-byte allocation\n",
                what, (long long) n_elems, off, bytes);
        return false;
    }
    return true;
}

// Elements each buffer needs for one (channel, token, seq) lattice:
//   x / y : c + (p - hist) * s_tok + s * s_seq with c <= d_inner-1, p - hist <= n_tokens-1
//   w     : tap + c * d_conv, tap <= d_conv-1
//   state : which * hist * d_inner + s * 3 * hist * d_inner + c * hist + tap
// (p - hist <= n_tokens - 1 in BOTH passes: the output pass's largest is t + d_conv - 1 -
// hist = n_tokens - 1, the state pass's is n_tokens + hist - 1 - hist = n_tokens - 1.)
//
// It also refuses any stream whose output overlaps its input. conv_silu_body is safe in
// place -- it writes y[t] after reading x[<= t] -- but the token-parallel pass is not: at
// n_tokens > 1 the thread for t' > t reads x[t] while the thread for t writes it. The
// engine keeps the conv input (the projection output) and its output (scratch) apart, so
// this only ever sends a deliberate in-place call back to the serial kernel.
static bool conv_silu_stream_ok(const char * what,
                                const float * x, int64_t x_tok, int64_t x_seq,
                                const float * y, int64_t y_tok, int64_t y_seq,
                                const float * w,
                                int64_t d_conv, int64_t d_inner, int n_tokens, int n_seqs) {
    const int64_t xneed = (d_inner - 1) + (int64_t) (n_tokens - 1) * x_tok +
                          (int64_t) (n_seqs - 1) * x_seq + 1;
    const int64_t yneed = (d_inner - 1) + (int64_t) (n_tokens - 1) * y_tok +
                          (int64_t) (n_seqs - 1) * y_seq + 1;
    const int64_t wneed = d_conv * d_inner;
    char nx[32], ny[32], nw[32];
    snprintf(nx, sizeof nx, "%s.x", what);
    snprintf(ny, sizeof ny, "%s.y", what);
    snprintf(nw, sizeof nw, "%s.w", what);
    if (!conv_in_alloc(x, xneed, nx) || !conv_in_alloc(y, yneed, ny) ||
        !conv_in_alloc(w, wneed, nw)) {
        return false;
    }
    const std::uintptr_t xb = (std::uintptr_t) x;
    const std::uintptr_t yb = (std::uintptr_t) y;
    const std::uintptr_t xs = (std::uintptr_t) xneed * sizeof(float);
    const std::uintptr_t ys = (std::uintptr_t) yneed * sizeof(float);
    if (xb < yb + ys && yb < xb + xs) {
        fprintf(stderr,
                "ling3_kda: %s: in [%p,+%llu) and out [%p,+%llu) overlap; "
                "keeping the serial kernel (safe in place)\n",
                what, (const void *) x, (unsigned long long) xs,
                (const void *) y, (unsigned long long) ys);
        return false;
    }
    // The parallel pass gives every (channel, token, seq) its own thread, so the lattice has
    // to be injective in the index it computes: c + t*s_tok + s*s_seq. Contiguous strides
    // satisfy this; a tighter one would let two threads write the same element.
    if (x_tok < d_inner || y_tok < d_inner ||
        x_seq < (int64_t) n_tokens * x_tok || y_seq < (int64_t) n_tokens * y_tok) {
        fprintf(stderr,
                "ling3_kda: %s: strides (x_tok=%lld x_seq=%lld, y_tok=%lld y_seq=%lld) are not "
                "a [rows, n_tokens, n_seqs] lattice for %lld rows; keeping the serial kernel\n",
                what, (long long) x_tok, (long long) x_seq, (long long) y_tok,
                (long long) y_seq, (long long) d_inner);
        return false;
    }
    return true;
}

// The q/k/v entry: q/k/v share q's s_seq (the serial kernel passes qi_seq to all three
// bodies), and all three outputs share qo_seq, so the extents use those, not k's/v's.
static bool conv_silu_qkv_audit(const KdaShape & sh,
                                const ViewC & q, const ViewC & k, const ViewC & v,
                                const View & qo, const View & ko, const View & vo,
                                const float * wq, const float * wk, const float * wv,
                                const float * conv_state,
                                int n_tokens, int n_seqs) {
    const int64_t d_inner = sh.d_inner();
    const int64_t d_conv  = sh.d_conv;
    const int64_t hist    = d_conv - 1;
    if (!(d_conv >= 2 && d_inner > 0 && n_tokens > 0 && n_seqs > 0 &&
          q.p != nullptr && k.p != nullptr && v.p != nullptr &&
          qo.p != nullptr && ko.p != nullptr && vo.p != nullptr &&
          wq != nullptr && wk != nullptr && wv != nullptr && conv_state != nullptr)) {
        fprintf(stderr,
                "ling3_kda: REFUSING conv_silu_qkv prefill: bad geometry or null pointer "
                "(d_conv=%d d_inner=%lld n_tokens=%d n_seqs=%d)\n",
                sh.d_conv, (long long) d_inner, n_tokens, n_seqs);
        return false;
    }
    if (!conv_silu_stream_ok("q", q.p, q.s_tok, q.s_seq, qo.p, qo.s_tok, qo.s_seq, wq,
                             d_conv, d_inner, n_tokens, n_seqs) ||
        !conv_silu_stream_ok("k", k.p, k.s_tok, q.s_seq, ko.p, ko.s_tok, qo.s_seq, wk,
                             d_conv, d_inner, n_tokens, n_seqs) ||
        !conv_silu_stream_ok("v", v.p, v.s_tok, q.s_seq, vo.p, vo.s_tok, qo.s_seq, wv,
                             d_conv, d_inner, n_tokens, n_seqs)) {
        return false;
    }
    // state: which*hist*d_inner + s*3*hist*d_inner + c*hist + tap, which <= 2
    const int64_t sneed = (int64_t) (n_seqs - 1) * 3 * hist * d_inner +
                          (int64_t) 3 * hist * d_inner;
    return conv_in_alloc(conv_state, sneed, "conv_state");
}

// The one-stream entry: `st` is already offset by qkv*hist*d_inner on the host and the kernel
// takes slice 0 of it, so the state extent is one slice shorter than the q/k/v one.
static bool conv_silu_one_audit(const KdaShape & sh,
                                const ViewC & in, const View & out, const float * w,
                                const float * st, int n_tokens, int n_seqs) {
    const int64_t d_inner = sh.d_inner();
    const int64_t d_conv  = sh.d_conv;
    const int64_t hist    = d_conv - 1;
    if (!(d_conv >= 2 && d_inner > 0 && n_tokens > 0 && n_seqs > 0 &&
          in.p != nullptr && out.p != nullptr && w != nullptr && st != nullptr)) {
        fprintf(stderr,
                "ling3_kda: REFUSING conv_silu prefill: bad geometry or null pointer "
                "(d_conv=%d d_inner=%lld n_tokens=%d n_seqs=%d)\n",
                sh.d_conv, (long long) d_inner, n_tokens, n_seqs);
        return false;
    }
    if (!conv_silu_stream_ok("1", in.p, in.s_tok, in.s_seq, out.p, out.s_tok, out.s_seq, w,
                             d_conv, d_inner, n_tokens, n_seqs)) {
        return false;
    }
    const int64_t sneed = (int64_t) (n_seqs - 1) * 3 * hist * d_inner +
                          (int64_t) hist * d_inner;
    return conv_in_alloc(st, sneed, "conv_state");
}

// The serial body gathers the d_conv window into a kConvMaxD-slot array; refuse a shape that
// would overrun it. Ling's bailingmoe3.ssm.conv_kernel is 4.
static void conv_silu_check_d_conv(int d_conv) {
    if (d_conv < 2 || d_conv > kConvMaxD) {
        fprintf(stderr, "ling3_kda: d_conv %d outside [2, %d]; refusing the conv launch\n",
                d_conv, kConvMaxD);
        abort();
    }
}

void kda_conv_silu(const KdaShape & sh, const float * conv_w,
                   const ViewC & in, const View & out,
                   float * conv_state, int qkv,
                   int n_tokens, int n_seqs, hipStream_t stream) {
    conv_silu_check_d_conv(sh.d_conv);
    const int d_inner = sh.d_inner();
    const int hist    = sh.d_conv - 1;
    const int bs      = 256;
    // gridDim.z == 1, so blockIdx.z is 0 and the kernel takes conv-state slice 0 of whatever it
    // is given -- the qkv offset belongs here on the host, exactly as in the original.
    float * st = conv_state + (size_t) qkv * (size_t) hist * (size_t) d_inner;

    // LC1: n_tokens > 1 takes the run-parallel pair (outputs against the OLD state, then the
    // state update). n_tokens == 1 -- every decode step -- keeps the serial kernel untouched.
    // The z extent is n_seqs, not 1: which = zs / n_seqs is 0 for every slot and s = zs, so
    // every sequence gets its outputs and its state update (n_seqs == 1 hides this).
    if (n_tokens > 1 && conv_silu_one_audit(sh, in, out, conv_w, st, n_tokens, n_seqs)) {
        const dim3 grid((d_inner + kRunChan - 1) / kRunChan,
                        (n_tokens + kRun - 1) / kRun, n_seqs);
        // which == 0: the host already slid st to this stream's slice (the qkv offset is the
        // caller's here), so the kernel must not add a second one.
        conv_silu_run_launch(sh.d_conv, grid, in.p, out.p, st, conv_w,
                             d_inner, n_tokens, n_seqs, 0,
                             in.s_tok, in.s_seq, out.s_tok, out.s_seq, stream);
        hipLaunchKernelGGL(conv_silu_state_kernel,
                           dim3((d_inner + bs - 1) / bs, n_seqs), bs, 0, stream,
                           in.p, in.p, in.p,
                           st,
                           sh.d_conv, d_inner, n_tokens, n_seqs,
                           in.s_tok, in.s_tok, in.s_tok, in.s_seq);
        LING3_HIP_CHECK(hipGetLastError());
        g_conv_silu_path.store(1, std::memory_order_relaxed);
        return;
    }
    const dim3 grid((d_inner + bs - 1) / bs, n_seqs, 1);
    hipLaunchKernelGGL(conv_silu_kernel, grid, bs, 0, stream,
                       in.p, in.p, in.p,
                       out.p, out.p, out.p,
                       st,
                       conv_w, conv_w, conv_w,
                       sh.d_conv, d_inner, n_tokens,
                       in.s_tok, in.s_tok, in.s_tok, in.s_seq,
                       out.s_tok, out.s_tok, out.s_tok, out.s_seq);
    LING3_HIP_CHECK(hipGetLastError());
    g_conv_silu_path.store(0, std::memory_order_relaxed);
}

void kda_conv_silu_qkv(const KdaShape & sh,
                       const ViewC & q, const ViewC & k, const ViewC & v,
                       const View & qo, const View & ko, const View & vo,
                       const float * wq, const float * wk, const float * wv,
                       float * conv_state,
                       int n_tokens, int n_seqs, hipStream_t stream) {
    conv_silu_check_d_conv(sh.d_conv);
    const int d_inner = sh.d_inner();
    const int bs      = 256;

    // LC1: n_tokens > 1 takes the run-parallel pair. The audit is what makes this safe --
    // every index the two kernels compute is checked against the real allocation first.
    if (n_tokens > 1 && conv_silu_qkv_audit(sh, q, k, v, qo, ko, vo, wq, wk, wv,
                                            conv_state, n_tokens, n_seqs)) {
        // One launch per stream, like the fork's ssm_conv (three calls). The grid is
        // [channel tiles, token runs, sequences] and which picks the q/k/v slice of the
        // conv state.
        const dim3 grid((d_inner + kRunChan - 1) / kRunChan,
                        (n_tokens + kRun - 1) / kRun, n_seqs);
        for (int wsel = 0; wsel < 3; ++wsel) {
            const float * xs   = (wsel == 0) ? q.p : (wsel == 1 ? k.p : v.p);
            float *       ys   = (wsel == 0) ? qo.p : (wsel == 1 ? ko.p : vo.p);
            const float * wsp  = (wsel == 0) ? wq : (wsel == 1 ? wk : wv);
            const int64_t xt   = (wsel == 0) ? q.s_tok : (wsel == 1 ? k.s_tok : v.s_tok);
            const int64_t yt   = (wsel == 0) ? qo.s_tok : (wsel == 1 ? ko.s_tok : vo.s_tok);
            conv_silu_run_launch(sh.d_conv, grid, xs, ys, conv_state, wsp,
                                 d_inner, n_tokens, n_seqs, wsel,
                                 xt, q.s_seq, yt, qo.s_seq, stream);
        }
        hipLaunchKernelGGL(conv_silu_state_kernel,
                           dim3((d_inner + bs - 1) / bs, 3 * n_seqs), bs, 0, stream,
                           q.p, k.p, v.p,
                           conv_state,
                           sh.d_conv, d_inner, n_tokens, n_seqs,
                           q.s_tok, k.s_tok, v.s_tok, q.s_seq);
        LING3_HIP_CHECK(hipGetLastError());
        g_conv_silu_path.store(1, std::memory_order_relaxed);
        return;
    }
    const dim3 grid((d_inner + bs - 1) / bs, n_seqs, 3);
    hipLaunchKernelGGL(conv_silu_kernel, grid, bs, 0, stream,
                       q.p, k.p, v.p,
                       qo.p, ko.p, vo.p,
                       conv_state,
                       wq, wk, wv,
                       sh.d_conv, d_inner, n_tokens,
                       q.s_tok, k.s_tok, v.s_tok, q.s_seq,
                       qo.s_tok, ko.s_tok, vo.s_tok, qo.s_seq);
    LING3_HIP_CHECK(hipGetLastError());
    g_conv_silu_path.store(0, std::memory_order_relaxed);
}

void kda_gate_beta(const KdaShape & sh, const KdaScalars & sc, const KdaWeights & w,
                   const ViewC & f, const ViewC & beta,
                   const View & g1_out, const View & beta_out,
                   int n_tokens, int n_seqs, hipStream_t stream) {
    const int gbs = 256;
    hipLaunchKernelGGL(gate_kernel, dim3((sh.d_inner() + gbs - 1) / gbs), gbs, 0, stream,
                       f.p, g1_out.p, w.a_exp, w.dt_bias, sc.gate_lower_bound,
                       sh.head_dim, n_tokens, n_seqs, sh.d_inner(),
                       f.s_tok, f.s_seq, g1_out.s_tok, g1_out.s_seq);
    LING3_HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(beta_kernel, dim3(grid_1d(sh.n_head)), 128, 0, stream,
                       beta.p, beta_out.p, n_tokens, n_seqs,
                       beta.s_tok, beta.s_seq, beta_out.s_tok, beta_out.s_seq);
    LING3_HIP_CHECK(hipGetLastError());
}

void kda_l2_norm(const KdaShape & sh, float l2_eps,
                 const ViewC & in, const View & out,
                 int n_tokens, int n_seqs, hipStream_t stream) {
    const dim3 grid(sh.n_head, n_tokens * n_seqs);
    hipLaunchKernelGGL(l2_norm_kernel, grid, 256, 0, stream,
                       in.p, out.p, sh.head_dim, l2_eps,
                       n_tokens, n_seqs, in.s_tok, in.s_seq, out.s_tok, out.s_seq);
    LING3_HIP_CHECK(hipGetLastError());
}

// Decode scan, variant vec. Same grid and same arithmetic text as scan_kernel.
// Only the per-column state load/store is float4. LING3_KDA_DECODE_VARIANT=vec.
static __global__ void scan_kernel_decode_vec(const float * __restrict__ q,
                                   const float * __restrict__ k,
                                   const float * __restrict__ v,
                                   const float * __restrict__ g1,
                                   const float * __restrict__ beta,
                                   float * __restrict__ out,
                                   float * __restrict__ ssm_state,
                                   int n_head, int n_tokens, int n_seqs,
                                   float scale,
                                   int64_t q_tok, int64_t q_seq,
                                   int64_t v_tok, int64_t v_seq,
                                   int64_t b_tok, int64_t b_seq,
                                   int64_t o_tok, int64_t o_seq) {
    constexpr int HD = 128;

    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int j = threadIdx.x;                 // value channel

    const int64_t chan = (int64_t) h * HD;

    float * srow = ssm_state + (((int64_t) s * n_head + h) * HD + j) * HD;

    float S[HD];
    // Vector load of this thread's contiguous column. Values match the scalar loads.
    #pragma unroll
    for (int i = 0; i < HD; i += 4) {
        const float4 t4 = *reinterpret_cast<const float4 *>(srow + i);
        S[i]     = t4.x;
        S[i + 1] = t4.y;
        S[i + 2] = t4.z;
        S[i + 3] = t4.w;
    }

    __shared__ float eg[HD];
    __shared__ float sbeta;

    for (int t = 0; t < n_tokens; ++t) {
        const float * krow = k + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;
        const float * qrow = q + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;
        const float * grow = g1 + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;

        // thread j publishes eg[j] = exp(g[j]); g is indexed by the key channel, and
        // every thread needs all 128 values, so this fills the shared table.
        eg[j] = expf(grow[j]);
        if (j == 0) {
            sbeta = beta[h + (int64_t) t * b_tok + (int64_t) s * b_seq];
        }
        __syncthreads();
        const float beta_val = sbeta;

        float kv = 0.0f;
        #pragma unroll
        for (int i = 0; i < HD; ++i) {
            kv += eg[i] * S[i] * krow[i];
        }

        const float delta =
            (v[chan + j + (int64_t) t * v_tok + (int64_t) s * v_seq] - kv) * beta_val;

        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < HD; ++i) {
            S[i] = eg[i] * S[i] + krow[i] * delta;
            acc += S[i] * qrow[i];
        }

        out[chan + j + (int64_t) t * o_tok + (int64_t) s * o_seq] = acc * scale;

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < HD; i += 4) {
        float4 t4;
        t4.x = S[i];
        t4.y = S[i + 1];
        t4.z = S[i + 2];
        t4.w = S[i + 3];
        *reinterpret_cast<float4 *>(srow + i) = t4;
    }
}


// Decode scan, production. One thread owns one value-column and keeps float S[128],
// same per-column expressions as scan_kernel. State IO is a coalesced transposed tile.
// Block 128, grid (n_head, n_seqs). n_tokens is 1 on this path.

static __global__ void scan_kernel_decode(const float * __restrict__ q,
                                          const float * __restrict__ k,
                                          const float * __restrict__ v,
                                          const float * __restrict__ g1,
                                          const float * __restrict__ beta,
                                          float * __restrict__ out,
                                          float * __restrict__ ssm_state,
                                          int n_head, int n_tokens, int n_seqs,
                                          float scale,
                                          int64_t q_tok, int64_t q_seq,
                                          int64_t v_tok, int64_t v_seq,
                                          int64_t b_tok, int64_t b_seq,
                                          int64_t o_tok, int64_t o_seq) {
    constexpr int HD = 128;

    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int j = threadIdx.x;                 // value channel

    const int64_t chan = (int64_t) h * HD;
    float * head = ssm_state + (((int64_t) s * n_head + h) * HD) * HD;

    // 128*128 floats = 65536 bytes. pub overlays the front of the tile (129 floats)
    // after the column copy, so eg[] does not add a second shared allocation.
    __shared__ __align__(16) union {
        float tile[HD * HD];
        struct {
            float eg[HD];
            float sbeta;
        } pub;
    } mem;

    // Coalesced load. 4096 float4s, 128 threads, 32 each.
    // fidx = thread + m*128 covers 0..4095. elem = fidx*4 stays inside one column
    // (128 % 4 == 0). col = elem/128 is 0..127, i0 = elem%128 is 0,4,...,124.
    // tile[(i0+k)*128 + col] is then 0..16383.
    {
        const float4 * in4 = reinterpret_cast<const float4 *>(head);
        #pragma unroll 1
        for (int m = 0; m < HD / 4; ++m) {
            const int fidx = (int) threadIdx.x + m * HD;
            const float4 xv = in4[fidx];
            const int elem = fidx * 4;
            const int col = elem >> 7;
            const int i0 = elem & 127;
            mem.tile[(i0 + 0) * HD + col] = xv.x;
            mem.tile[(i0 + 1) * HD + col] = xv.y;
            mem.tile[(i0 + 2) * HD + col] = xv.z;
            mem.tile[(i0 + 3) * HD + col] = xv.w;
        }
    }
    __syncthreads();

    float S[HD];
    #pragma unroll
    for (int i = 0; i < HD; ++i) {
        S[i] = mem.tile[i * HD + j];
    }
    __syncthreads();

    for (int t = 0; t < n_tokens; ++t) {
        const float * krow = k + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;
        const float * qrow = q + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;
        const float * grow = g1 + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;

        mem.pub.eg[j] = expf(grow[j]);
        if (j == 0) {
            mem.pub.sbeta = beta[h + (int64_t) t * b_tok + (int64_t) s * b_seq];
        }
        __syncthreads();
        const float beta_val = mem.pub.sbeta;

        float kv = 0.0f;
        #pragma unroll
        for (int i = 0; i < HD; ++i) {
            kv += mem.pub.eg[i] * S[i] * krow[i];
        }

        const float delta =
            (v[chan + j + (int64_t) t * v_tok + (int64_t) s * v_seq] - kv) * beta_val;

        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < HD; ++i) {
            S[i] = mem.pub.eg[i] * S[i] + krow[i] * delta;
            acc += S[i] * qrow[i];
        }

        out[chan + j + (int64_t) t * o_tok + (int64_t) s * o_seq] = acc * scale;

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < HD; ++i) {
        mem.tile[i * HD + j] = S[i];
    }
    __syncthreads();
    {
        float4 * out4 = reinterpret_cast<float4 *>(head);
        #pragma unroll 1
        for (int m = 0; m < HD / 4; ++m) {
            const int fidx = (int) threadIdx.x + m * HD;
            const int elem = fidx * 4;
            const int col = elem >> 7;
            const int i0 = elem & 127;
            float4 xv;
            xv.x = mem.tile[(i0 + 0) * HD + col];
            xv.y = mem.tile[(i0 + 1) * HD + col];
            xv.z = mem.tile[(i0 + 2) * HD + col];
            xv.w = mem.tile[(i0 + 3) * HD + col];
            out4[fidx] = xv;
        }
    }
}

// Decode scan, variant fma. Same tile as scan_kernel_decode, explicit fmaf pairing.
// Not the default. LING3_KDA_DECODE_VARIANT=fma. Numerics may differ from scan_kernel.

static __global__ void scan_kernel_decode_fma(const float * __restrict__ q,
                                          const float * __restrict__ k,
                                          const float * __restrict__ v,
                                          const float * __restrict__ g1,
                                          const float * __restrict__ beta,
                                          float * __restrict__ out,
                                          float * __restrict__ ssm_state,
                                          int n_head, int n_tokens, int n_seqs,
                                          float scale,
                                          int64_t q_tok, int64_t q_seq,
                                          int64_t v_tok, int64_t v_seq,
                                          int64_t b_tok, int64_t b_seq,
                                          int64_t o_tok, int64_t o_seq) {
    constexpr int HD = 128;

    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int j = threadIdx.x;                 // value channel

    const int64_t chan = (int64_t) h * HD;
    float * head = ssm_state + (((int64_t) s * n_head + h) * HD) * HD;

    // 128*128 floats = 65536 bytes. pub overlays the front of the tile (129 floats)
    // after the column copy, so eg[] does not add a second shared allocation.
    __shared__ __align__(16) union {
        float tile[HD * HD];
        struct {
            float eg[HD];
            float sbeta;
        } pub;
    } mem;

    // Coalesced load. 4096 float4s, 128 threads, 32 each.
    // fidx = thread + m*128 covers 0..4095. elem = fidx*4 stays inside one column
    // (128 % 4 == 0). col = elem/128 is 0..127, i0 = elem%128 is 0,4,...,124.
    // tile[(i0+k)*128 + col] is then 0..16383.
    {
        const float4 * in4 = reinterpret_cast<const float4 *>(head);
        #pragma unroll 1
        for (int m = 0; m < HD / 4; ++m) {
            const int fidx = (int) threadIdx.x + m * HD;
            const float4 xv = in4[fidx];
            const int elem = fidx * 4;
            const int col = elem >> 7;
            const int i0 = elem & 127;
            mem.tile[(i0 + 0) * HD + col] = xv.x;
            mem.tile[(i0 + 1) * HD + col] = xv.y;
            mem.tile[(i0 + 2) * HD + col] = xv.z;
            mem.tile[(i0 + 3) * HD + col] = xv.w;
        }
    }
    __syncthreads();

    float S[HD];
    #pragma unroll
    for (int i = 0; i < HD; ++i) {
        S[i] = mem.tile[i * HD + j];
    }
    __syncthreads();

    for (int t = 0; t < n_tokens; ++t) {
        const float * krow = k + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;
        const float * qrow = q + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;
        const float * grow = g1 + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;

        mem.pub.eg[j] = expf(grow[j]);
        if (j == 0) {
            mem.pub.sbeta = beta[h + (int64_t) t * b_tok + (int64_t) s * b_seq];
        }
        __syncthreads();
        const float beta_val = mem.pub.sbeta;

        float kv = 0.0f;
        #pragma unroll
        for (int i = 0; i < HD; ++i) {
            kv = fmaf(krow[i], S[i] * mem.pub.eg[i], kv);
        }

        const float delta =
            (v[chan + j + (int64_t) t * v_tok + (int64_t) s * v_seq] - kv) * beta_val;

        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < HD; ++i) {
            S[i] = fmaf(mem.pub.eg[i], S[i], krow[i] * delta);
            acc = fmaf(S[i], qrow[i], acc);
        }

        out[chan + j + (int64_t) t * o_tok + (int64_t) s * o_seq] = acc * scale;

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < HD; ++i) {
        mem.tile[i * HD + j] = S[i];
    }
    __syncthreads();
    {
        float4 * out4 = reinterpret_cast<float4 *>(head);
        #pragma unroll 1
        for (int m = 0; m < HD / 4; ++m) {
            const int fidx = (int) threadIdx.x + m * HD;
            const int elem = fidx * 4;
            const int col = elem >> 7;
            const int i0 = elem & 127;
            float4 xv;
            xv.x = mem.tile[(i0 + 0) * HD + col];
            xv.y = mem.tile[(i0 + 1) * HD + col];
            xv.z = mem.tile[(i0 + 2) * HD + col];
            xv.w = mem.tile[(i0 + 3) * HD + col];
            out4[fidx] = xv;
        }
    }
}

__device__ __forceinline__ float warp_sum_32(float v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor(v, mask, 32);
    }
    return v;
}

// 32-thread tile. Each thread still owns one value-column and runs the same
// expressions as scan_kernel. The 32 columns are loaded coalesced through LDS,
// then copied into float S[128]. Direct float4 store of that column.
// LING3_KDA_DECODE_VARIANT=t32. Grid (n_head, n_seqs, 4), block 32.
static __global__ void scan_kernel_decode_t32(const float * __restrict__ q,
                                              const float * __restrict__ k,
                                              const float * __restrict__ v,
                                              const float * __restrict__ g1,
                                              const float * __restrict__ beta,
                                              float * __restrict__ out,
                                              float * __restrict__ ssm_state,
                                              int n_head, int n_tokens, int n_seqs,
                                              float scale,
                                              int64_t q_tok, int64_t q_seq,
                                              int64_t v_tok, int64_t v_seq,
                                              int64_t b_tok, int64_t b_seq,
                                              int64_t o_tok, int64_t o_seq) {
    constexpr int HD = 128;
    constexpr int COLS = 32;

    const int h = (int) blockIdx.x;
    const int s = (int) blockIdx.y;
    const int lane = (int) threadIdx.x;
    const int col0 = (int) blockIdx.z * COLS;
    const int j = col0 + lane;
    const int64_t chan = (int64_t) h * HD;

    float * head = ssm_state + (((int64_t) s * n_head + h) * HD) * HD;
    float * srow = head + (int64_t) j * HD;

    __shared__ __align__(16) float tile[COLS * HD];
    __shared__ float eg[HD];
    __shared__ float sbeta;

    {
        const float4 * in4 = reinterpret_cast<const float4 *>(head + (int64_t) col0 * HD);
        float4 * s4 = reinterpret_cast<float4 *>(tile);
        // 1024 float4s, 32 threads, 32 each. fidx = lane + m*32 covers 0..1023.
        #pragma unroll 1
        for (int m = 0; m < (COLS * HD) / 4 / COLS; ++m) {
            const int fidx = lane + m * COLS;
            s4[fidx] = in4[fidx];
        }
    }

    for (int t = 0; t < n_tokens; ++t) {
        const float * krow = k + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;
        const float * qrow = q + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;
        const float * grow = g1 + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;

        for (int i = lane; i < HD; i += COLS) {
            eg[i] = expf(grow[i]);
        }
        if (lane == 0) {
            sbeta = beta[h + (int64_t) t * b_tok + (int64_t) s * b_seq];
        }
        __syncthreads();

        float S[HD];
        #pragma unroll
        for (int i = 0; i < HD; ++i) {
            S[i] = tile[lane * HD + i];
        }

        const float beta_val = sbeta;

        float kv = 0.0f;
        #pragma unroll
        for (int i = 0; i < HD; ++i) {
            kv += eg[i] * S[i] * krow[i];
        }

        const float delta =
            (v[chan + j + (int64_t) t * v_tok + (int64_t) s * v_seq] - kv) * beta_val;

        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < HD; ++i) {
            S[i] = eg[i] * S[i] + krow[i] * delta;
            acc += S[i] * qrow[i];
        }

        out[chan + j + (int64_t) t * o_tok + (int64_t) s * o_seq] = acc * scale;

        #pragma unroll
        for (int i = 0; i < HD; i += 4) {
            float4 t4;
            t4.x = S[i];
            t4.y = S[i + 1];
            t4.z = S[i + 2];
            t4.w = S[i + 3];
            *reinterpret_cast<float4 *>(srow + i) = t4;
        }
        __syncthreads();
    }
}

// Numeric knob, default off. Fork shape: one warp owns one value-column, each lane
// holds 4 state rows, warp-sum of the partials. Sum order is not scan_kernel's.
// LING3_KDA_DECODE_VARIANT=red. Grid (n_head, n_seqs, 32), block (32, 4).
static __global__ void __launch_bounds__(128, 2)
scan_kernel_decode_red(const float * __restrict__ q,
                       const float * __restrict__ k,
                       const float * __restrict__ v,
                       const float * __restrict__ g1,
                       const float * __restrict__ beta,
                       float * __restrict__ out,
                       float * __restrict__ ssm_state,
                       int n_head, int n_tokens, int n_seqs,
                       float scale,
                       int64_t q_tok, int64_t q_seq,
                       int64_t v_tok, int64_t v_seq,
                       int64_t b_tok, int64_t b_seq,
                       int64_t o_tok, int64_t o_seq) {
    constexpr int HD = 128;
    constexpr int ROWS = 4;

    const int lane = (int) threadIdx.x;
    const int h = (int) blockIdx.x;
    const int s = (int) blockIdx.y;
    const int col = (int) blockIdx.z * (int) blockDim.y + (int) threadIdx.y;
    const int64_t chan = (int64_t) h * HD;

    float * head = ssm_state + (((int64_t) s * n_head + h) * HD) * HD;
    float * col_state = head + (int64_t) col * HD;

    float s_shard[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        s_shard[r] = col_state[r * 32 + lane];
    }

    for (int t = 0; t < n_tokens; ++t) {
        const float * krow = k + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;
        const float * qrow = q + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;
        const float * grow = g1 + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;
        const float beta_val = beta[h + (int64_t) t * b_tok + (int64_t) s * b_seq];

        float k_reg[ROWS];
        float q_reg[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int i = r * 32 + lane;
            k_reg[r] = krow[i];
            q_reg[r] = qrow[i];
        }

        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int i = r * 32 + lane;
            kv_shard += expf(grow[i]) * s_shard[r] * k_reg[r];
        }
        const float kv = warp_sum_32(kv_shard);
        const float delta =
            (v[chan + col + (int64_t) t * v_tok + (int64_t) s * v_seq] - kv) * beta_val;

        float acc_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int i = r * 32 + lane;
            s_shard[r] = expf(grow[i]) * s_shard[r] + k_reg[r] * delta;
            acc_shard += s_shard[r] * q_reg[r];
        }
        const float acc = warp_sum_32(acc_shard);
        if (lane == 0) {
            out[chan + col + (int64_t) t * o_tok + (int64_t) s * o_seq] = acc * scale;
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        col_state[r * 32 + lane] = s_shard[r];
    }
}

static int g_kda_fast_launches = 0;

int kda_decode_fast_launches() { return g_kda_fast_launches; }

// Test override. 0 = cached env (what serve uses). 1 = reference scan.
// 2 = the named variant. The unit bench and the scalar A/B walk set this.
// Serve never calls kda_decode_test_pick, so each knob is read once.
static int g_kda_pick = 0;
static char g_kda_pick_name[32] = "t32";

void kda_decode_test_pick(int pick, const char * variant) {
    if (pick != 1 && pick != 2) {
        g_kda_pick = 0;
        return;
    }
    g_kda_pick = pick;
    if (variant != nullptr && variant[0] != '\0') {
        std::snprintf(g_kda_pick_name, sizeof(g_kda_pick_name), "%s", variant);
    }
}

static bool kda_decode_ref_requested() {
    if (g_kda_pick == 1) return true;
    if (g_kda_pick == 2) return false;
    static const bool on = [] {
        const char * e = std::getenv("LING3_KDA_DECODE_REF");
        return e != nullptr && e[0] == '1' && e[1] == '\0';
    }();
    return on;
}

// Optional dump of decode-scan inputs (LING3_KDA_CAPTURE=path). Used to replay real model
// tensors. Records are call 0, stride, 2*stride, ... and call 1. Does not change values.
static void kda_capture_decode(const KdaShape & sh,
                               const float * q, const float * k, const float * v,
                               const float * g1, const float * beta, const float * ssm_state,
                               int n_tokens, int n_seqs, hipStream_t stream) {
    const char * path = std::getenv("LING3_KDA_CAPTURE");
    if (path == nullptr || path[0] == '\0') return;
    if (n_tokens != 1 || n_seqs != 1 || sh.head_dim != 128) return;

    static int calls = 0;
    const int call = calls++;
    const char * stride_e = std::getenv("LING3_KDA_CAPTURE_STRIDE");
    const char * n_e = std::getenv("LING3_KDA_CAPTURE_N");
    const int stride = (stride_e != nullptr && stride_e[0] != '\0') ? std::atoi(stride_e) : 35;
    const int want = (n_e != nullptr && n_e[0] != '\0') ? std::atoi(n_e) : 4;
    const int stride_use = stride > 0 ? stride : 35;
    const int want_use = want > 0 ? want : 4;
    bool take = (call == 1);
    if (call % stride_use == 0 && (call / stride_use) < want_use) take = true;
    if (!take) return;

    const int n_head = sh.n_head;
    const int d_inner = n_head * 128;
    const size_t n_state = (size_t) n_head * 128u * 128u;
    const size_t bytes = sizeof(int) + sizeof(float) * ((size_t) d_inner * 4u + (size_t) n_head + n_state);

    static FILE * fp = nullptr;
    static int nrec = 0;
    if (fp == nullptr) {
        fp = std::fopen(path, "w+b");
        if (fp == nullptr) {
            fprintf(stderr, "ling3_kda: capture open %s failed\n", path);
            return;
        }
        const uint32_t magic = 0x3141444Bu;
        const int32_t version = 1;
        const int32_t head_dim = 128;
        const int32_t n_seqs_w = 1;
        const int32_t z = 0;
        const int32_t stride_w = stride_use;
        if (std::fwrite(&magic, 4, 1, fp) != 1 ||
            std::fwrite(&version, 4, 1, fp) != 1 ||
            std::fwrite(&n_head, 4, 1, fp) != 1 ||
            std::fwrite(&head_dim, 4, 1, fp) != 1 ||
            std::fwrite(&n_seqs_w, 4, 1, fp) != 1 ||
            std::fwrite(&d_inner, 4, 1, fp) != 1 ||
            std::fwrite(&z, 4, 1, fp) != 1 ||
            std::fwrite(&stride_w, 4, 1, fp) != 1) {
            fprintf(stderr, "ling3_kda: capture header write failed\n");
            std::fclose(fp);
            fp = nullptr;
            return;
        }
        std::fflush(fp);
    }

    void * host = std::malloc(bytes);
    if (host == nullptr) return;
    char * hp = (char *) host;
    std::memcpy(hp, &call, sizeof(int));
    hp += sizeof(int);
    auto pull = [&](const float * src, size_t n) {
        LING3_HIP_CHECK(hipMemcpyAsync(hp, src, n * sizeof(float), hipMemcpyDeviceToHost, stream));
        LING3_HIP_CHECK(hipStreamSynchronize(stream));
        hp += n * sizeof(float);
    };
    pull(q, (size_t) d_inner);
    pull(k, (size_t) d_inner);
    pull(v, (size_t) d_inner);
    pull(g1, (size_t) d_inner);
    pull(beta, (size_t) n_head);
    pull(ssm_state, n_state);
    if (std::fwrite(host, 1, bytes, fp) != bytes) {
        fprintf(stderr, "ling3_kda: capture record write failed\n");
    } else {
        nrec += 1;
        if (std::fseek(fp, 24, SEEK_SET) == 0) {
            std::fwrite(&nrec, 4, 1, fp);
            std::fseek(fp, 0, SEEK_END);
        }
        std::fflush(fp);
        fprintf(stderr, "ling3_kda: captured call %d rec %d\n", call, nrec);
    }
    std::free(host);
}

static void launch_scan_ref(const KdaShape & sh, float scale,
                            const ViewC & q, const ViewC & k, const ViewC & v,
                            const ViewC & g1, const ViewC & beta,
                            const View & out, float * ssm_state,
                            int n_tokens, int n_seqs, hipStream_t stream) {
    const dim3 grid(sh.n_head, n_seqs);
    hipLaunchKernelGGL(scan_kernel, grid, sh.head_dim, 0, stream,
                       q.p, k.p, v.p, g1.p, beta.p, out.p, ssm_state,
                       sh.n_head, n_tokens, n_seqs, scale,
                       q.s_tok, q.s_seq, v.s_tok, v.s_seq,
                       beta.s_tok, beta.s_seq, out.s_tok, out.s_seq);
    LING3_HIP_CHECK(hipGetLastError());
}

static const char * kda_decode_variant_name() {
    if (g_kda_pick == 2) return g_kda_pick_name;
    // Read once. Same pattern as BatchAllLogits / MlaTile in rocm_backend.cpp.
    // t32 is the default: bit-identical, fastest identical kernel on lk3-unit3.
    static const char * cached = []() -> const char * {
        const char * e = std::getenv("LING3_KDA_DECODE_VARIANT");
        if (e == nullptr || e[0] == '\0') return "t32";
        static char buf[32];
        std::snprintf(buf, sizeof(buf), "%s", e);
        return buf;
    }();
    return cached;
}

static void launch_scan_decode(const KdaShape & sh, float scale,
                               const ViewC & q, const ViewC & k, const ViewC & v,
                               const ViewC & g1, const ViewC & beta,
                               const View & out, float * ssm_state,
                               int n_seqs, hipStream_t stream) {
    const char * variant = kda_decode_variant_name();
    const dim3 grid_h(sh.n_head, n_seqs);
    if (std::strcmp(variant, "coal") == 0) {
        hipLaunchKernelGGL(scan_kernel_decode, grid_h, sh.head_dim, 0, stream,
                           q.p, k.p, v.p, g1.p, beta.p, out.p, ssm_state,
                           sh.n_head, /*n_tokens*/ 1, n_seqs, scale,
                           q.s_tok, q.s_seq, v.s_tok, v.s_seq,
                           beta.s_tok, beta.s_seq, out.s_tok, out.s_seq);
    } else if (std::strcmp(variant, "fma") == 0) {
        hipLaunchKernelGGL(scan_kernel_decode_fma, grid_h, sh.head_dim, 0, stream,
                           q.p, k.p, v.p, g1.p, beta.p, out.p, ssm_state,
                           sh.n_head, /*n_tokens*/ 1, n_seqs, scale,
                           q.s_tok, q.s_seq, v.s_tok, v.s_seq,
                           beta.s_tok, beta.s_seq, out.s_tok, out.s_seq);
    } else if (std::strcmp(variant, "t32") == 0) {
        const dim3 grid(sh.n_head, n_seqs, 4);
        hipLaunchKernelGGL(scan_kernel_decode_t32, grid, 32, 0, stream,
                           q.p, k.p, v.p, g1.p, beta.p, out.p, ssm_state,
                           sh.n_head, /*n_tokens*/ 1, n_seqs, scale,
                           q.s_tok, q.s_seq, v.s_tok, v.s_seq,
                           beta.s_tok, beta.s_seq, out.s_tok, out.s_seq);
    } else if (std::strcmp(variant, "red") == 0) {
        const dim3 grid(sh.n_head, n_seqs, 32);
        const dim3 block(32, 4);
        hipLaunchKernelGGL(scan_kernel_decode_red, grid, block, 0, stream,
                           q.p, k.p, v.p, g1.p, beta.p, out.p, ssm_state,
                           sh.n_head, /*n_tokens*/ 1, n_seqs, scale,
                           q.s_tok, q.s_seq, v.s_tok, v.s_seq,
                           beta.s_tok, beta.s_seq, out.s_tok, out.s_seq);
    } else {
        hipLaunchKernelGGL(scan_kernel_decode_vec, grid_h, sh.head_dim, 0, stream,
                           q.p, k.p, v.p, g1.p, beta.p, out.p, ssm_state,
                           sh.n_head, /*n_tokens*/ 1, n_seqs, scale,
                           q.s_tok, q.s_seq, v.s_tok, v.s_seq,
                           beta.s_tok, beta.s_seq, out.s_tok, out.s_seq);
    }
    LING3_HIP_CHECK(hipGetLastError());
    g_kda_fast_launches += 1;
}

void kda_scan(const KdaShape & sh, const KdaScalars & sc,
              const ViewC & q, const ViewC & k, const ViewC & v,
              const ViewC & g1, const ViewC & beta,
              const View & out, float * ssm_state,
              int n_tokens, int n_seqs, hipStream_t stream) {
    if (sh.head_dim != 128) {
        fprintf(stderr, "ling3_kda: kda_scan requires head_dim == 128, got %d\n", sh.head_dim);
        abort();
    }
    const float scale = sc.qk_scale != 0.0f ? sc.qk_scale
                                            : 1.0f / sqrtf((float) sh.head_dim);

    const bool aligned = (reinterpret_cast<std::uintptr_t>(ssm_state) & 15u) == 0;
    const bool fast = n_tokens == 1 && n_seqs > 0 && sh.n_head > 0 && aligned &&
                      !kda_decode_ref_requested();
    if (fast) {
        kda_capture_decode(sh, q.p, k.p, v.p, g1.p, beta.p, ssm_state, n_tokens, n_seqs, stream);
        launch_scan_decode(sh, scale, q, k, v, g1, beta, out, ssm_state, n_seqs, stream);
    } else {
        if (n_tokens == 1) {
            kda_capture_decode(sh, q.p, k.p, v.p, g1.p, beta.p, ssm_state, n_tokens, n_seqs, stream);
        }
        launch_scan_ref(sh, scale, q, k, v, g1, beta, out, ssm_state, n_tokens, n_seqs, stream);
    }
}

void kda_out_norm_gate(const KdaShape & sh, const KdaScalars & sc, const KdaWeights & w,
                       const ViewC & o_in, const ViewC & g2, const View & out,
                       int n_tokens, int n_seqs, hipStream_t stream) {
    const dim3 grid(sh.n_head, n_tokens * n_seqs);
    hipLaunchKernelGGL(out_norm_gate_kernel, grid, 256, 0, stream,
                       o_in.p, g2.p, out.p, w.o_norm,
                       sh.head_dim, sc.rms_eps,
                       n_tokens, n_seqs,
                       o_in.s_tok, o_in.s_seq, g2.s_tok, g2.s_seq, out.s_tok, out.s_seq);
    LING3_HIP_CHECK(hipGetLastError());
}

void kda_state_zero(const KdaShape & sh, float * conv_state, float * ssm_state,
                    int n_seqs, hipStream_t stream) {
    const size_t nc = (size_t) sh.embd_r() * (size_t) n_seqs;
    const size_t ns = (size_t) sh.embd_s() * (size_t) n_seqs;
    if (conv_state) {
        const int blocks = (int) ((nc + 255) / 256);
        hipLaunchKernelGGL(zero_kernel, dim3(grid_1d(blocks)), 256, 0, stream, conv_state, nc);
        LING3_HIP_CHECK(hipGetLastError());
    }
    if (ssm_state) {
        const int blocks = (int) ((ns + 255) / 256);
        hipLaunchKernelGGL(zero_kernel, dim3(grid_1d(blocks)), 256, 0, stream, ssm_state, ns);
        LING3_HIP_CHECK(hipGetLastError());
    }
}

size_t kda_forward_scratch_bytes(const KdaShape & sh, int n_tokens, int n_seqs) {
    const size_t per_tok_seq = (size_t) (4 * sh.d_inner() + sh.n_head);
    return per_tok_seq * (size_t) n_tokens * (size_t) n_seqs * sizeof(float);
}

// LKF wave-3 K1: conv + gate + beta + l2(q) + l2(k), decode only.
// Grid (d_inner/256, 1, 4), block 256. z = 0,1,2 is the serial conv body on q,k,v
// (same helpers and the same acc loop as conv_silu_out_at). z = 3 is gate_kernel's
// per-channel formula; block 0 also writes beta (32 sigmoids). Each z<2 block then
// L2-normalises its 256 channels (two heads) with block_sum<256>, upper 128 lanes
// contributing 0, which is l2_norm_kernel's reduction. Existing kernels are not edited.
//
// Index proof for the Ling shape (d_inner = n_head*128, d_inner % 256 == 0, s = t = 0),
// checked again in kda_k1_ok before launch:
//   c = blockIdx.x*256 + threadIdx.x ∈ [0, d_inner).
//   conv-state index = which*(d_conv-1)*d_inner + c*(d_conv-1) + tap,
//   which ∈ [0,2], tap ∈ [0, d_conv-2], max = 3*(d_conv-1)*d_inner - 1 = embd_r()-1.
//   L2 head = 2*blockIdx.x + hloc, hloc ∈ {0,1}, last head = n_head-1.
//   beta index = threadIdx.x on block 0, and only when threadIdx.x < n_head ≤ 256.
__device__ __forceinline__ float lkf_conv_decode_channel(
        const float * __restrict__ x, float * __restrict__ y,
        float * __restrict__ conv_state, const float * __restrict__ w,
        int d_conv, int d_inner,
        int64_t x_tok, int64_t x_seq, int64_t y_tok, int64_t y_seq,
        int c, int s) {
    const int hist = d_conv - 1;
    const int n_tokens = 1;
    const float * st = conv_state + (int64_t) s * 3 * hist * d_inner + (int64_t) c * hist;
    const float * xr = x + (int64_t) s * x_seq;
    float *       yr = y + (int64_t) s * y_seq;
    float sv[kConvMaxD];
    float yv = 0.0f;
    for (int t = 0; t < n_tokens; ++t) {
        for (int tap = 0; tap < d_conv; ++tap) {
            sv[tap] = conv_silu_stream_at(st, xr, hist, x_tok, c, (int64_t) t + tap);
        }
        // conv_silu_out_at, verbatim, with the silu value kept for the L2 epilogue.
        float acc = 0.0f;
        for (int tap = 0; tap < d_conv; ++tap) {
            acc += w[tap + (int64_t) c * d_conv] * sv[tap];
        }
        yv = silu_f(acc);
        conv_silu_store(&yr[c + (int64_t) t * y_tok], yv);
    }
    conv_silu_state_at(x, conv_state, d_conv, d_inner, n_tokens, x_tok, x_seq, c, s);
    return yv;
}

static __global__ void kda_conv_gbl2(const float * __restrict__ xq,
                                     const float * __restrict__ xk,
                                     const float * __restrict__ xv,
                                     const float * __restrict__ f,
                                     const float * __restrict__ beta_in,
                                     float * __restrict__ yq,
                                     float * __restrict__ yk,
                                     float * __restrict__ yv,
                                     float * __restrict__ g1,
                                     float * __restrict__ beta_out,
                                     float * __restrict__ conv_state,
                                     const float * __restrict__ wq,
                                     const float * __restrict__ wk,
                                     const float * __restrict__ wv,
                                     const float * __restrict__ a_exp,
                                     const float * __restrict__ dt_bias,
                                     float lower_bound, float l2_eps,
                                     int d_conv, int d_inner, int head_dim, int n_head,
                                     int64_t q_tok, int64_t k_tok, int64_t v_tok, int64_t q_seq,
                                     int64_t yq_tok, int64_t yk_tok, int64_t yv_tok, int64_t y_seq,
                                     int64_t f_tok, int64_t f_seq, int64_t g_tok, int64_t g_seq,
                                     int64_t bi_tok, int64_t bi_seq, int64_t bo_tok, int64_t bo_seq) {
    const int which = (int) blockIdx.z;
    const int c = (int) blockIdx.x * 256 + (int) threadIdx.x;
    const int s = 0;
    const int t = 0;
    const int hist = d_conv - 1;
    __shared__ float ysm[256];
    __shared__ float s_sum[256];

    if (which < 3) {
        const float * x = which == 0 ? xq : (which == 1 ? xk : xv);
        float *       y = which == 0 ? yq : (which == 1 ? yk : yv);
        const float * w = which == 0 ? wq : (which == 1 ? wk : wv);
        const int64_t x_tok = which == 0 ? q_tok : (which == 1 ? k_tok : v_tok);
        const int64_t y_tok = which == 0 ? yq_tok : (which == 1 ? yk_tok : yv_tok);
        // q.s_seq / qo.s_seq for every stream, matching kda_conv_silu_qkv's launch.
        float * st = conv_state + (size_t) which * (size_t) hist * (size_t) d_inner;
        const float yv = lkf_conv_decode_channel(x, y, st, w, d_conv, d_inner,
                                                 x_tok, q_seq, y_tok, y_seq, c, s);
        if (which < 2) {
            ysm[threadIdx.x] = yv;
            __syncthreads();
            float * ybase = y + (int64_t) s * y_seq;
            for (int hloc = 0; hloc < 2; ++hloc) {
                float sq = 0.0f;
                if (threadIdx.x < 128) {
                    const float xv = ysm[hloc * 128 + (int) threadIdx.x];
                    sq = xv * xv;
                }
                const float sum = block_sum<256>(sq, s_sum);
                const float scale = rsqrtf(fmaxf(sum, l2_eps * l2_eps));
                if (threadIdx.x < 128) {
                    const int local = hloc * 128 + (int) threadIdx.x;
                    const int chan = (int) blockIdx.x * 256 + local;
                    ybase[chan + (int64_t) t * y_tok] = scale * ysm[local];
                }
            }
        }
    } else {
        const float a = a_exp[c / head_dim];
        const float fv = f[c + (int64_t) t * f_tok + (int64_t) s * f_seq] + dt_bias[c];
        g1[c + (int64_t) t * g_tok + (int64_t) s * g_seq] = lower_bound * sigmoid_f(fv * a);
        if (blockIdx.x == 0 && (int) threadIdx.x < n_head) {
            const int h = (int) threadIdx.x;
            beta_out[h + (int64_t) t * bo_tok + (int64_t) s * bo_seq] =
                sigmoid_f(beta_in[h + (int64_t) t * bi_tok + (int64_t) s * bi_seq]);
        }
    }
}

// k_q8_1_quant, verbatim (ling3_gemv_q8.hip). Fallback only: the t32 epilogue
// inlines the same butterfly so the wo slot does not need this launch.
static __global__ void kda_q8_1_quant(const float * x, int8_t * qs, float * ds, int nb) {
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

// LKF wave-3 K2 + Q2-KDA. One block per head, 128 threads, one thread per value
// column — the same statements as scan_kernel (the reference t32 is bit-identical
// to). A reused 32-column shared tile did not match t32 on a random state
// (unit rand-a, first differing state byte 2568). The out_norm tree is
// block_sum<256> with the upper 128 lanes at 0, which is what out_norm_gate_kernel
// does at block size 256. Then the four Q8 groups of this head, same butterfly as
// k_q8_1_quant. Grid (n_head, n_seqs), block 128. red/ref/vec/coal/fma do not
// launch this.
//
// Index proof (head_dim 128, n_tokens 1, block 128 so j = threadIdx.x < 128):
//   state index = ((s * n_head + h) * 128 + j) * 128 + i, i < 128,
//   max = n_seqs * n_head * 128 * 128 - 1.
//   quant group b = h * 4 + (j >> 5), element b * 32 + (j & 31) < n_head * 128.
static __global__ void scan_kernel_decode_t32_fused(const float * __restrict__ q,
                                                    const float * __restrict__ k,
                                                    const float * __restrict__ v,
                                                    const float * __restrict__ g1,
                                                    const float * __restrict__ beta,
                                                    const float * __restrict__ g2,
                                                    float * __restrict__ out,
                                                    float * __restrict__ ssm_state,
                                                    const float * __restrict__ o_norm,
                                                    int8_t * __restrict__ qs,
                                                    float * __restrict__ ds,
                                                    int n_head, int n_tokens,
                                                    float scale, float rms_eps,
                                                    int64_t q_tok, int64_t q_seq,
                                                    int64_t v_tok, int64_t v_seq,
                                                    int64_t b_tok, int64_t b_seq,
                                                    int64_t g_tok, int64_t g_seq,
                                                    int64_t o_tok, int64_t o_seq) {
    constexpr int HD = 128;

    const int h = (int) blockIdx.x;
    const int s = (int) blockIdx.y;
    const int j = (int) threadIdx.x;
    const int64_t chan = (int64_t) h * HD;

    float * srow = ssm_state + (((int64_t) s * n_head + h) * HD + j) * HD;

    float S[HD];
    #pragma unroll
    for (int i = 0; i < HD; ++i) {
        S[i] = srow[i];
    }

    __shared__ float eg[HD];
    __shared__ float sbeta;
    __shared__ float s_sum[256];

    for (int t = 0; t < n_tokens; ++t) {
        const float * krow = k + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;
        const float * qrow = q + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;
        const float * grow = g1 + chan + (int64_t) t * q_tok + (int64_t) s * q_seq;

        eg[j] = expf(grow[j]);
        if (j == 0) {
            sbeta = beta[h + (int64_t) t * b_tok + (int64_t) s * b_seq];
        }
        __syncthreads();
        const float beta_val = sbeta;

        float kv = 0.0f;
        #pragma unroll
        for (int i = 0; i < HD; ++i) {
            kv += eg[i] * S[i] * krow[i];
        }

        const float delta =
            (v[chan + j + (int64_t) t * v_tok + (int64_t) s * v_seq] - kv) * beta_val;

        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < HD; ++i) {
            S[i] = eg[i] * S[i] + krow[i] * delta;
            acc += S[i] * qrow[i];
        }

        const float yscan = acc * scale;
        out[chan + j + (int64_t) t * o_tok + (int64_t) s * o_seq] = yscan;
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < HD; ++i) {
        srow[i] = S[i];
    }

    // out_norm_gate on the last token, reading the scan output BACK FROM GLOBAL --
    // exactly the value out_norm_gate_kernel reads. The unfused sequence has a
    // kernel boundary here, which forces `acc * scale` to round to f32 before the
    // norm math; a fused kernel without that round trip lets the compiler regroup
    // nscale * (acc * scale) * w * sigmoid and drift 1-4 ULPs (measured: unit
    // rand-a out 1276/4096 floats off by 1 ULP with ssm_state bit-exact). The
    // volatile read makes the round trip real.
    //
    // Reduction is block_sum<256> with the upper 128 lanes contributing 0, which
    // is what out_norm_gate_kernel does at block size 256: same slots, same pairs,
    // same order.
    {
        const int t = n_tokens - 1;
        const float * gr = g2 + chan + (int64_t) t * g_tok + (int64_t) s * g_seq;
        const int64_t oi = chan + j + (int64_t) t * o_tok + (int64_t) s * o_seq;
        volatile float * vo = out;
        const float xi = vo[oi];
        float tmp = 0.0f;
        tmp += xi * xi;
        s_sum[j] = tmp;
        s_sum[j + 128] = 0.0f;
        __syncthreads();
        for (int off = 128; off > 0; off >>= 1) {
            if (j < off) {
                s_sum[j] += s_sum[j + off];
            }
            __syncthreads();
        }
        const float sum = s_sum[0];
        __syncthreads();
        const float mean = sum / (float) HD;
        const float nscale = rsqrtf(mean + rms_eps);
        // out_norm_gate_kernel writes exactly
        //     yr[col] = (scale * xr[col]) * w[col] * sigmoid_f(gr[col]);
        // -ffast-math may reshuffle that four-factor product differently in a
        // large kernel than in that small one, which alone moves the result
        // 1-4 ULPs (measured; the scan and the reduction are bit-exact). Pin the
        // source order with one volatile round trip per factor.
        volatile float p1 = nscale * xi;
        volatile float p2 = p1 * o_norm[j];
        const float y = p2 * sigmoid_f(gr[j]);
        out[oi] = y;
    }
    __syncthreads();

    if (qs != nullptr && ds != nullptr) {
        const int lane = j & 31;
        const int grp = j >> 5;
        const int t = n_tokens - 1;
        const int64_t dst = chan + j + (int64_t) t * o_tok + (int64_t) s * o_seq;
        // k_q8_1_quant reads its 32-value group from GLOBAL; do the same so the
        // butterfly sees the identical bits it sees on the unfused path.
        volatile float * vo = out;
        const float xv = vo[dst];
        float ma = fabsf(xv), mv = xv;
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
        const int b = h * 4 + grp;
        if (lane == 0) {
            ds[b] = d;
        }
        const float id_b = __shfl(id, 0);
        qs[(size_t) b * 32 + lane] = (int8_t) roundf(xv * id_b);
    }
}

static int g_kda_fused_pick = 0;
static int g_kda_k1_launches = 0;
static int g_kda_k2_launches = 0;

void kda_decode_fused_test_pick(int pick) {
    g_kda_fused_pick = (pick == 1 || pick == 2) ? pick : 0;
}

int kda_decode_fused_k1_launches() { return g_kda_k1_launches; }
int kda_decode_fused_k2_launches() { return g_kda_k2_launches; }

static bool kda_decode_fused_on() {
    if (g_kda_fused_pick == 1) return false;
    if (g_kda_fused_pick == 2) return true;
    static const bool on = [] {
        const char * e = std::getenv("LING3_KDA_DECODE_FUSED");
        if (e == nullptr || e[0] == '\0') return true;
        return !(e[0] == '0' && e[1] == '\0');
    }();
    return on;
}

bool kda_wo_q8_fused() {
    if (!kda_decode_fused_on()) return false;
    if (kda_decode_ref_requested()) return false;
    return std::strcmp(kda_decode_variant_name(), "t32") == 0;
}

static bool kda_k1_ok(const KdaShape & sh, const KdaInputs & in, const KdaState & st,
                      const KdaWeights & w) {
    if (in.n_tokens != 1 || in.n_seqs != 1) return false;
    if (sh.head_dim != 128 || sh.n_head < 1 || sh.n_head > 256) return false;
    if (sh.d_conv < 2 || sh.d_conv > kConvMaxD) return false;
    const int d = sh.d_inner();
    if (d != sh.n_head * sh.head_dim || (d % 256) != 0) return false;
    if (in.q.p == nullptr || in.k.p == nullptr || in.v.p == nullptr ||
        in.f.p == nullptr || in.beta.p == nullptr) return false;
    if (st.conv_state == nullptr || w.conv_wq == nullptr || w.conv_wk == nullptr ||
        w.conv_wv == nullptr || w.a_exp == nullptr || w.dt_bias == nullptr) return false;
    const int hist = sh.d_conv - 1;
    const int state_n = 3 * hist * d;
    const int max_state = 2 * hist * d + (d - 1) * hist + (hist - 1);
    if (max_state < 0 || max_state >= state_n) return false;
    const int last_head = 2 * (d / 256) - 1;
    if (last_head != sh.n_head - 1) return false;
    return true;
}

static bool kda_k2_ok(const KdaShape & sh, const float * o_norm, const ViewC & g2,
                      float * ssm_state, int n_tokens, int n_seqs) {
    if (n_tokens != 1 || n_seqs != 1) return false;
    if (sh.head_dim != 128 || sh.n_head < 1) return false;
    if (o_norm == nullptr || g2.p == nullptr || ssm_state == nullptr) return false;
    if ((reinterpret_cast<std::uintptr_t>(ssm_state) & 15u) != 0) return false;
    return true;
}

static void launch_k1(const KdaShape & sh, const KdaScalars & sc, const KdaWeights & w,
                      const KdaInputs & in, const KdaState & st,
                      const View & qv, const View & kv, const View & vv,
                      const View & gv, const View & bv, hipStream_t stream) {
    const int d_inner = sh.d_inner();
    const int blocks = d_inner / 256;
    hipLaunchKernelGGL(kda_conv_gbl2, dim3(blocks, 1, 4), dim3(256), 0, stream,
                       in.q.p, in.k.p, in.v.p, in.f.p, in.beta.p,
                       qv.p, kv.p, vv.p, gv.p, bv.p,
                       st.conv_state,
                       w.conv_wq, w.conv_wk, w.conv_wv, w.a_exp, w.dt_bias,
                       sc.gate_lower_bound, sc.l2_eps,
                       sh.d_conv, d_inner, sh.head_dim, sh.n_head,
                       in.q.s_tok, in.k.s_tok, in.v.s_tok, in.q.s_seq,
                       qv.s_tok, kv.s_tok, vv.s_tok, qv.s_seq,
                       in.f.s_tok, in.f.s_seq, gv.s_tok, gv.s_seq,
                       in.beta.s_tok, in.beta.s_seq, bv.s_tok, bv.s_seq);
    LING3_HIP_CHECK(hipGetLastError());
    g_kda_k1_launches += 1;
}

static void launch_k2(const KdaShape & sh, const KdaScalars & sc, const float * o_norm,
                      const ViewC & q, const ViewC & k, const ViewC & v,
                      const ViewC & g1, const ViewC & beta, const ViewC & g2,
                      const View & out, float * ssm_state,
                      int n_tokens, int n_seqs,
                      int8_t * wo_qs, float * wo_ds, hipStream_t stream) {
    const float scale = sc.qk_scale != 0.0f ? sc.qk_scale
                                            : 1.0f / sqrtf((float) sh.head_dim);
    hipLaunchKernelGGL(scan_kernel_decode_t32_fused, dim3(sh.n_head, n_seqs), dim3(128), 0, stream,
                       q.p, k.p, v.p, g1.p, beta.p, g2.p, out.p, ssm_state, o_norm,
                       wo_qs, wo_ds,
                       sh.n_head, n_tokens, scale, sc.rms_eps,
                       q.s_tok, q.s_seq, v.s_tok, v.s_seq,
                       beta.s_tok, beta.s_seq, g2.s_tok, g2.s_seq,
                       out.s_tok, out.s_seq);
    LING3_HIP_CHECK(hipGetLastError());
    g_kda_k2_launches += 1;
    g_kda_fast_launches += 1;
}

static void launch_q8_fallback(const float * x, int8_t * qs, float * ds, int k,
                               hipStream_t stream) {
    const int nb = k / 32;
    const int threads = 256;
    const int blocks = (nb + 7) / 8;
    hipLaunchKernelGGL(kda_q8_1_quant, dim3(blocks), dim3(threads), 0, stream, x, qs, ds, nb);
    LING3_HIP_CHECK(hipGetLastError());
}

// Scan + out_norm, or the fused t32 epilogue. Returns 1 when K2 launched.
static int kda_run_tail(const KdaShape & sh, const KdaScalars & sc, const float * o_norm,
                        const ViewC & q, const ViewC & k, const ViewC & v,
                        const ViewC & g1, const ViewC & beta, const ViewC & g2,
                        const View & out, float * ssm_state,
                        int n_tokens, int n_seqs,
                        int8_t * wo_qs, float * wo_ds, hipStream_t stream) {
    const bool want_k2 = kda_decode_fused_on() && !kda_decode_ref_requested() &&
                         std::strcmp(kda_decode_variant_name(), "t32") == 0 &&
                         kda_k2_ok(sh, o_norm, g2, ssm_state, n_tokens, n_seqs);
    if (want_k2) {
        kda_capture_decode(sh, q.p, k.p, v.p, g1.p, beta.p, ssm_state, n_tokens, n_seqs, stream);
        launch_k2(sh, sc, o_norm, q, k, v, g1, beta, g2, out, ssm_state,
                  n_tokens, n_seqs, wo_qs, wo_ds, stream);
        return 1;
    }
    kda_scan(sh, sc, q, k, v, g1, beta, out, ssm_state, n_tokens, n_seqs, stream);
    KdaWeights ow;
    ow.o_norm = o_norm;
    kda_out_norm_gate(sh, sc, ow, ViewC{out.p, out.s_tok, out.s_seq}, g2, out,
                      n_tokens, n_seqs, stream);
    if (wo_qs != nullptr && wo_ds != nullptr && n_tokens == 1 && n_seqs == 1 &&
        (sh.d_inner() % 32) == 0) {
        launch_q8_fallback(out.p, wo_qs, wo_ds, sh.d_inner(), stream);
    }
    return 0;
}

int kda_decode_scan_out(const KdaShape & sh, const KdaScalars & sc, const float * o_norm,
                        const ViewC & q, const ViewC & k, const ViewC & v,
                        const ViewC & g1, const ViewC & beta, const ViewC & g2,
                        const View & out, float * ssm_state,
                        int8_t * wo_qs, float * wo_ds, hipStream_t stream) {
    return kda_run_tail(sh, sc, o_norm, q, k, v, g1, beta, g2, out, ssm_state,
                        /*n_tokens*/ 1, /*n_seqs*/ 1, wo_qs, wo_ds, stream);
}

void kda_forward_scratch(const KdaShape & sh, const KdaScalars & sc, const KdaWeights & w,
                         const KdaInputs & in, const KdaState & st, const View & out,
                         void * scratch, size_t scratch_bytes, hipStream_t stream,
                         int8_t * wo_qs, float * wo_ds) {
    const int n_tokens = in.n_tokens;
    const int n_seqs   = in.n_seqs;
    const int d_inner  = sh.d_inner();

    const size_t need = kda_forward_scratch_bytes(sh, n_tokens, n_seqs);
    if (scratch == nullptr || scratch_bytes < need) {
        fprintf(stderr, "ling3_kda: scratch %zu bytes given, %zu needed\n", scratch_bytes, need);
        abort();
    }

    float * sp = (float *) scratch;
    float * q_c = sp; sp += (size_t) d_inner * n_tokens * n_seqs;
    float * k_c = sp; sp += (size_t) d_inner * n_tokens * n_seqs;
    float * v_c = sp; sp += (size_t) d_inner * n_tokens * n_seqs;
    float * g1  = sp; sp += (size_t) d_inner * n_tokens * n_seqs;
    float * b_o = sp;

    View qv = make_view(q_c, d_inner, n_tokens);
    View kv = make_view(k_c, d_inner, n_tokens);
    View vv = make_view(v_c, d_inner, n_tokens);
    View gv = make_view(g1,  d_inner, n_tokens);
    View bv = make_view(b_o, sh.n_head,  n_tokens);

    const bool want_k1 = kda_decode_fused_on() && kda_k1_ok(sh, in, st, w);
    if (want_k1) {
        launch_k1(sh, sc, w, in, st, qv, kv, vv, gv, bv, stream);
    } else {
        kda_conv_silu_qkv(sh, in.q, in.k, in.v, qv, kv, vv,
                          w.conv_wq, w.conv_wk, w.conv_wv,
                          st.conv_state, n_tokens, n_seqs, stream);
        kda_gate_beta(sh, sc, w, in.f, in.beta, gv, bv, n_tokens, n_seqs, stream);
        kda_l2_norm(sh, sc.l2_eps, ViewC{q_c, qv.s_tok, qv.s_seq}, qv, n_tokens, n_seqs, stream);
        kda_l2_norm(sh, sc.l2_eps, ViewC{k_c, kv.s_tok, kv.s_seq}, kv, n_tokens, n_seqs, stream);
    }

    kda_run_tail(sh, sc, w.o_norm, qv, kv, vv, gv, bv, in.g2, out, st.ssm_state,
                 n_tokens, n_seqs, wo_qs, wo_ds, stream);
}

void kda_forward(const KdaShape & sh, const KdaScalars & sc, const KdaWeights & w,
                 const KdaInputs & in, const KdaState & st, const View & out,
                 hipStream_t stream, int8_t * wo_qs, float * wo_ds) {
    const size_t need = kda_forward_scratch_bytes(sh, in.n_tokens, in.n_seqs);
    void * scratch = nullptr;
    LING3_HIP_CHECK(hipMalloc(&scratch, need));
    kda_forward_scratch(sh, sc, w, in, st, out, scratch, need, stream, wo_qs, wo_ds);
    LING3_HIP_CHECK(hipStreamSynchronize(stream));
    LING3_HIP_CHECK(hipFree(scratch));
}

void kda_forward_decode(const KdaShape & sh, const KdaScalars & sc, const KdaWeights & w,
                        const KdaInputs & in, const KdaState & st, const View & out,
                        hipStream_t stream) {
    if (in.n_tokens != 1) {
        fprintf(stderr, "ling3_kda: kda_forward_decode expects n_tokens == 1, got %d\n", in.n_tokens);
        abort();
    }
    kda_forward(sh, sc, w, in, st, out, stream);
}

const char * kda_version() { return "ling3-kda/1.0"; }

} // namespace ling3
