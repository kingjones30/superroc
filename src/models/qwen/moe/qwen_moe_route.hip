#include "src/models/qwen/moe/qwen_moe_route.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "src/core/rocm_fp4/gufo_fp4_moe_group.hpp"
#include "src/core/rocm_fp4/gufo_fp4_route.hpp"

namespace {

constexpr int kMaxExpert = 256;
constexpr int kMaxUsed = 8;
constexpr int kMaxK = 8192;
constexpr int kMaxTokens = 16384;
constexpr int kThreads = 256;

// ---------------------------------------------------------------------------
// KAT fast route (SPEC2 item 1). The S2 kernel was one block with thread e
// serially dotting gate row e: adjacent threads read 8 KB apart, which is why
// it measured 1093.02 us/call (~1.92 GB/s). This version splits the work into
// two kernels:
//
//   1. QwenMoeRouteLogitsKernel -- one CTA per expert row, 256 threads
//      striding the row, so a warp's loads are 32 consecutive floats
//      (coalesced). Each thread accumulates its strided partial serially in
//      ascending i, then a fixed binary tree reduces the 256 partials. Both
//      the per-thread order and the tree are fixed, so the kernel is
//      deterministic run to run. The summation order differs from the CPU
//      oracle's strictly serial dot, so logits may differ by a few ulp --
//      the same tolerance class the S2 test already gates (the device FMA
//      contraction); G1's real gates are the id ORDER and the weight cosine.
//
//   2. QwenMoeRouteHeadDevice -- top-k + weights, one 32-lane wave per
//      token (Super ROC QRH). The fork's norm_w divides each picked softmax
//      probability by the sum of the picked ones, so the full softmax
//      denominator cancels: the weights are a softmax over the n_used picked
//      logits, and ranking by probability is ranking by logit. The head
//      therefore picks the top n_used LOGITS (bigger first, equal logits ->
//      lower expert id, NaN read as -inf) and takes n_used float exps. The
//      former head (256 FP64 exps, a serial FP64 sum on one thread and a
//      256-wide bitonic network with a block barrier per sub-stage) measured
//      12.9 us per layer on KAT; this one uses no shared memory and no
//      barrier. Weights match the former head to a few float ulp; the id
//      order can differ only where two logits are closer than the former
//      head's float probability rounding (the gate is the zero-mean NLL
//      test, not bit identity).
//
// Index proof, KAT shape k=2048, n_expert=256, n_used=8:
//   * logits kernel: block e < n_expert reads gate[(size_t)e * k + i] for
//     i in [0, k) -- row e of the row-major [n_expert, k] gate the host
//     copied -- and x[i] for i in [0, k). Writes logits_out[e] once (thread
//     0). No other global access.
//   * head: lane l of a token's wave reads logits[l + 32 s] only where
//     l + 32 s < n_expert <= 256 (s < 8), and lane r < n_used writes
//     ids[r] and weights[r] once. No shared memory, no barrier; the batch
//     kernel's early return covers a whole wave (tok is wave-uniform).
// ---------------------------------------------------------------------------

__global__ void QwenMoeRouteLogitsKernel(const float* __restrict__ x,
                                         const float* __restrict__ gate, int k,
                                         int n_expert,
                                         float* __restrict__ logits_out) {
  __shared__ float red[kThreads];
  const int e = static_cast<int>(blockIdx.x);
  if (e >= n_expert) {
    return;  // block-uniform: the whole CTA exits before any barrier
  }
  const float* row = gate + (static_cast<size_t>(e) * static_cast<size_t>(k));
  const int t = static_cast<int>(threadIdx.x);
  float acc = 0.0F;
  for (int i = t; i < k; i += kThreads) {
    acc += row[i] * x[i];
  }
  red[t] = acc;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (t < s) {
      red[t] += red[t + s];
    }
    __syncthreads();
  }
  if (t == 0) {
    logits_out[e] = red[0];
  }
}

// Super ROC QVB: up to kMaxRouteRows token rows per block against one gate
// row, so the router is read once for all rows. Per row the strided
// accumulation and the fixed tree are exactly QwenMoeRouteLogitsKernel's, so
// each logit is bit-identical to a one-row call. With shexp_gate set, block
// n_expert computes each row's shared-expert gate exactly as
// QwenMoeShexpSigmoidKernel does (same dot order, tree and sigmoid), so that
// launch goes away too.
constexpr int kMaxRouteRows = 4;

template <int M>
__global__ void QwenMoeRouteLogitsRowsKernel(
    const float* __restrict__ x, const float* __restrict__ gate, int k,
    int n_expert, float* __restrict__ logits_out,
    const float* __restrict__ shexp_gate, float* __restrict__ sig_out) {
  __shared__ float red[M][kThreads];
  const int e = static_cast<int>(blockIdx.x);
  const bool shexp = e == n_expert;
  if (e > n_expert || (shexp && shexp_gate == nullptr)) {
    return;  // block-uniform: the whole CTA exits before any barrier
  }
  const float* row =
      shexp ? shexp_gate
            : gate + (static_cast<size_t>(e) * static_cast<size_t>(k));
  const int t = static_cast<int>(threadIdx.x);
  float acc[M];
#pragma unroll
  for (int r = 0; r < M; ++r) {
    acc[r] = 0.0F;
  }
  for (int i = t; i < k; i += kThreads) {
    const float w = row[i];
#pragma unroll
    for (int r = 0; r < M; ++r) {
      acc[r] += w * x[static_cast<size_t>(r) * static_cast<size_t>(k) + i];
    }
  }
#pragma unroll
  for (int r = 0; r < M; ++r) {
    red[r][t] = acc[r];
  }
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (t < s) {
#pragma unroll
      for (int r = 0; r < M; ++r) {
        red[r][t] += red[r][t + s];
      }
    }
    __syncthreads();
  }
  if (t == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      if (shexp) {
        sig_out[r] = 1.0F / (1.0F + expf(-red[r][0]));
      } else {
        logits_out[static_cast<size_t>(r) * static_cast<size_t>(n_expert) +
                   static_cast<size_t>(e)] = red[r][0];
      }
    }
  }
}

constexpr int kHeadLanes = 32;
constexpr int kHeadSlots = kMaxExpert / kHeadLanes;     // 8 experts per lane
constexpr int kHeadTokensPerBlock = kThreads / kHeadLanes;  // batch: 8 waves

// Strict total order on (logit, expert id): bigger logit first, equal logits
// -> lower id. Every lane applies it to the same operands, so the butterfly
// below leaves the same winner in every lane.
__device__ __forceinline__ bool QwenRouteBetter(float va, int ea, float vb,
                                                int eb) {
  return va > vb || (va == vb && ea < eb);
}

// One 32-lane wave. Lane l holds experts l, l + 32, ... in registers. Each of
// the n_used rounds takes the lane's best unpicked expert, then a 5-step xor
// butterfly over the wave. ids/weights are written in rank order. Weight r =
// exp(l_r - l_0) / sum over the picked (sum in rank order, e_0 = 1, so the
// fork's max(sum, 2^-14) clamp cannot bite). If the top logit is infinite,
// the experts equal to it share the weight and the rest get 0 (all -inf or
// NaN logits -> uniform weights over the lowest ids).
__device__ void QwenMoeRouteHeadDevice(int lane,
                                       const float* __restrict__ logits,
                                       int n_expert, int n_used,
                                       int* __restrict__ ids,
                                       float* __restrict__ weights) {
  const float neg_inf = -__builtin_huge_valf();
  float v[kHeadSlots];
#pragma unroll
  for (int s = 0; s < kHeadSlots; ++s) {
    const int e = lane + s * kHeadLanes;
    const float x = e < n_expert ? logits[e] : neg_inf;
    v[s] = x == x ? x : neg_inf;  // NaN -> -inf
  }
  unsigned taken = 0U;
  float top = neg_inf;
  float sum = 0.0F;
  int my_id = 0;
  float my_e = 0.0F;
  for (int r = 0; r < n_used; ++r) {
    float bv = neg_inf;
    int be = kMaxExpert;  // "no candidate": loses to every real expert
#pragma unroll
    for (int s = 0; s < kHeadSlots; ++s) {
      const int e = lane + s * kHeadLanes;
      if (e < n_expert && ((taken >> s) & 1U) == 0U &&
          QwenRouteBetter(v[s], e, bv, be)) {
        bv = v[s];
        be = e;
      }
    }
#pragma unroll
    for (int off = kHeadLanes / 2; off > 0; off >>= 1) {
      const float ov = __shfl_xor(bv, off, kHeadLanes);
      const int oe = __shfl_xor(be, off, kHeadLanes);
      if (QwenRouteBetter(ov, oe, bv, be)) {
        bv = ov;
        be = oe;
      }
    }
    if ((be % kHeadLanes) == lane) {
      taken |= 1U << (be / kHeadLanes);
    }
    if (r == 0) {
      top = bv;
    }
    const float ex = __builtin_isinf(top) ? (bv == top ? 1.0F : 0.0F)
                                          : expf(bv - top);
    sum += ex;
    if (lane == r) {
      my_id = be;
      my_e = ex;
    }
  }
  if (lane < n_used) {
    ids[lane] = my_id;
    weights[lane] = my_e / sum;
  }
}

__global__ void QwenMoeRouteHeadKernel(const float* __restrict__ logits,
                                       int n_expert, int n_used,
                                       int* __restrict__ ids,
                                       float* __restrict__ weights) {
  QwenMoeRouteHeadDevice(static_cast<int>(threadIdx.x), logits, n_expert,
                         n_used, ids, weights);
}

__global__ void QwenMoeRouteHeadBatchKernel(const float* __restrict__ logits,
                                            int n_tokens, int n_expert,
                                            int n_used, int* __restrict__ ids,
                                            float* __restrict__ weights) {
  // kHeadTokensPerBlock tokens per block, one wave each. The return is
  // wave-uniform and the head has no barrier.
  const int tok = static_cast<int>(blockIdx.x) * kHeadTokensPerBlock +
                  static_cast<int>(threadIdx.x) / kHeadLanes;
  if (tok >= n_tokens) {
    return;
  }
  QwenMoeRouteHeadDevice(static_cast<int>(threadIdx.x) % kHeadLanes,
                         logits + (static_cast<size_t>(tok) *
                                   static_cast<size_t>(n_expert)),
                         n_expert, n_used,
                         ids + (static_cast<size_t>(tok) *
                                static_cast<size_t>(n_used)),
                         weights + (static_cast<size_t>(tok) *
                                    static_cast<size_t>(n_used)));
}

int ValidateRouteArgs(const float* x, const float* gate, int k, int n_expert,
                      int n_used, const float* logits, const int* ids,
                      const float* weights) {
  if (x == nullptr || gate == nullptr || logits == nullptr || ids == nullptr ||
      weights == nullptr) {
    return 1;
  }
  if (k <= 0 || k > kMaxK || n_expert <= 0 || n_expert > kMaxExpert ||
      n_used <= 0 || n_used > n_expert || n_used > kMaxUsed) {
    return 1;
  }
  if (gufo_fp4_pow2_ge(n_expert) > kMaxExpert) {
    return 1;
  }
  const std::size_t cells =
      static_cast<std::size_t>(n_expert) * static_cast<std::size_t>(k);
  if (cells / static_cast<std::size_t>(n_expert) !=
      static_cast<std::size_t>(k)) {
    return 1;
  }
  return 0;
}

}  // namespace

int LaunchQwenMoeRoute(const float* x, const float* gate, int k, int n_expert,
                       int n_used, float* logits, int* ids, float* weights,
                       hipStream_t stream) {
  const int bad =
      ValidateRouteArgs(x, gate, k, n_expert, n_used, logits, ids, weights);
  if (bad != 0) {
    return bad;
  }
  // hipGetLastError is sticky for the whole host thread. The decode GEMV
  // path tries an FP4 launch and falls through on rejection; a rejected
  // launch leaves the sticky code set even when the fallback kernel is the
  // one that actually runs. Clearing it here keeps that from aborting a
  // route whose own launches succeeded. The stale name is printed so a real
  // earlier failure is still visible in the smoke log.
  const hipError_t stale = hipGetLastError();
  if (stale != hipSuccess) {
    std::fprintf(stderr, "LKAT route stale %s\n", hipGetErrorString(stale));
  }
  QwenMoeRouteLogitsKernel<<<n_expert, kThreads, 0, stream>>>(x, gate, k,
                                                              n_expert, logits);
  QwenMoeRouteHeadKernel<<<1, kHeadLanes, 0, stream>>>(logits, n_expert,
                                                       n_used, ids, weights);
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    std::fprintf(stderr, "LKAT route launch %s\n", hipGetErrorString(err));
    return 2;
  }
  return 0;
}

int LaunchQwenMoeRouteBatch(const float* x, const float* gate, int k,
                            int n_expert, int n_tokens, int n_used,
                            float* logits, int* ids, float* weights,
                            hipStream_t stream) {
  const int bad =
      ValidateRouteArgs(x, gate, k, n_expert, n_used, logits, ids, weights);
  if (bad != 0) {
    return bad;
  }
  if (n_tokens <= 0 || n_tokens > kMaxTokens) {
    return 1;
  }
  const std::size_t cells =
      static_cast<std::size_t>(n_tokens) * static_cast<std::size_t>(n_expert);
  if (cells / static_cast<std::size_t>(n_tokens) !=
      static_cast<std::size_t>(n_expert)) {
    return 1;
  }
  // The register-tiled logits GEMM from src/core/rocm_fp4 (LS1 PHASE9 LV1);
  // it steps K by 32, which the host check below mirrors. fp32 accumulate,
  // k ascending, one writer per output -- deterministic.
  if ((k % 32) != 0) {
    return 1;
  }
  // Super ROC RBS: the logits GEMM now runs on the caller's stream, so it is
  // ordered after the writes to x and before the head without the two host
  // synchronisations it needed on the null stream (80 per 40-layer prefill
  // chunk). Same kernel, same arithmetic.
  if (gufo_fp4_route_logits_gemm_launch(x, gate, k, n_expert, n_tokens,
                                        logits, stream) != GUFO_FP4_OK) {
    return 2;
  }
  const int head_blocks =
      (n_tokens + kHeadTokensPerBlock - 1) / kHeadTokensPerBlock;
  QwenMoeRouteHeadBatchKernel<<<head_blocks, kThreads, 0, stream>>>(
      logits, n_tokens, n_expert, n_used, ids, weights);
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return 2;
  }
  return 0;
}

int LaunchQwenMoeRouteRows(const float* x, const float* gate, int k,
                           int n_expert, int n_rows, int n_used,
                           float* logits, int* ids, float* weights,
                           const float* shexp_gate, float* sig_out,
                           hipStream_t stream) {
  const int bad =
      ValidateRouteArgs(x, gate, k, n_expert, n_used, logits, ids, weights);
  if (bad != 0) {
    return bad;
  }
  if (n_rows <= 0 || n_rows > kMaxRouteRows ||
      (shexp_gate == nullptr) != (sig_out == nullptr)) {
    return 1;
  }
  // Same sticky-error rule as LaunchQwenMoeRoute.
  const hipError_t stale = hipGetLastError();
  if (stale != hipSuccess) {
    std::fprintf(stderr, "LKAT route stale %s\n", hipGetErrorString(stale));
  }
  const int blocks = n_expert + (shexp_gate != nullptr ? 1 : 0);
  switch (n_rows) {
    case 1:
      QwenMoeRouteLogitsRowsKernel<1><<<blocks, kThreads, 0, stream>>>(
          x, gate, k, n_expert, logits, shexp_gate, sig_out);
      break;
    case 2:
      QwenMoeRouteLogitsRowsKernel<2><<<blocks, kThreads, 0, stream>>>(
          x, gate, k, n_expert, logits, shexp_gate, sig_out);
      break;
    case 3:
      QwenMoeRouteLogitsRowsKernel<3><<<blocks, kThreads, 0, stream>>>(
          x, gate, k, n_expert, logits, shexp_gate, sig_out);
      break;
    default:
      QwenMoeRouteLogitsRowsKernel<4><<<blocks, kThreads, 0, stream>>>(
          x, gate, k, n_expert, logits, shexp_gate, sig_out);
      break;
  }
  if (n_rows == 1) {
    QwenMoeRouteHeadKernel<<<1, kHeadLanes, 0, stream>>>(logits, n_expert,
                                                         n_used, ids, weights);
  } else {
    QwenMoeRouteHeadBatchKernel<<<1, kThreads, 0, stream>>>(
        logits, n_rows, n_expert, n_used, ids, weights);
  }
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    std::fprintf(stderr, "LKAT route rows launch %s\n", hipGetErrorString(err));
    return 2;
  }
  return 0;
}

int LaunchQwenMoeRouteHeadForTest(const float* logits, int n_tokens,
                                  int n_expert, int n_used, int* ids,
                                  float* weights, hipStream_t stream) {
  if (logits == nullptr || ids == nullptr || weights == nullptr ||
      n_expert <= 0 || n_expert > kMaxExpert || n_used <= 0 ||
      n_used > n_expert || n_used > kMaxUsed || n_tokens < 0 ||
      n_tokens > kMaxTokens) {
    return 1;
  }
  if (n_tokens == 0) {
    QwenMoeRouteHeadKernel<<<1, kHeadLanes, 0, stream>>>(logits, n_expert,
                                                         n_used, ids, weights);
  } else {
    const int head_blocks =
        (n_tokens + kHeadTokensPerBlock - 1) / kHeadTokensPerBlock;
    QwenMoeRouteHeadBatchKernel<<<head_blocks, kThreads, 0, stream>>>(
        logits, n_tokens, n_expert, n_used, ids, weights);
  }
  return hipGetLastError() == hipSuccess ? 0 : 2;
}
