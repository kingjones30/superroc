#ifndef GUFO_MODELS_QWEN_MOE_QWEN_MOE_ROUTE_HPP_
#define GUFO_MODELS_QWEN_MOE_QWEN_MOE_ROUTE_HPP_

#include <hip/hip_runtime.h>

// KAT: one-token Qwen MoE router. Coalesced F32 GEMV (one CTA per expert
// row, strided threads + fixed tree reduce) feeding a one-wave top-k head
// (Super ROC QRH): the top n_used logits in rank order (bigger first, equal
// logits -> lower expert id, NaN read as -inf) and weights = softmax over
// the picked logits, which equals the fork norm_w renorm of the full
// softmax (its denominator cancels). The id SET matches
// KatRouteSoftmaxTopK and the weights match it to a few float ulp; the
// order differs only between logits closer than its float probability
// rounding. Logits may differ from the CPU oracle by a few ulp (fixed but
// different summation order), the tolerance class G1 already gates.
// Returns 0 on a queued launch, 1 on a rejected argument, 2 on a HIP error.
// Host checks before launch:
//   x, gate, logits, ids, weights non-null
//   0 < k <= 8192, 0 < n_expert <= 256, 0 < n_used <= n_expert, n_used <= 8
//   n_expert * k fits in size_t
// The logits kernel touches gate row e only for e < n_expert and columns
// [0, k); the head reads logits[0, n_expert) and writes ids/weights
// [0, n_used). No other global access.
int LaunchQwenMoeRoute(const float* x, const float* gate, int k, int n_expert,
                       int n_used, float* logits, int* ids, float* weights,
                       hipStream_t stream);

// KAT: batched router for prefill. Logits come from the rocm_fp4
// register-tiled GEMM (deterministic; requires k % 32 == 0), then the same
// one-wave head per token (8 tokens per block) as the single-token path. x is
// [n_tokens, k] token-major, logits [n_tokens, n_expert], ids/weights
// [n_tokens, n_used]. Same return contract; n_tokens must be in [1, 16384].
int LaunchQwenMoeRouteBatch(const float* x, const float* gate, int k,
                            int n_expert, int n_tokens, int n_used,
                            float* logits, int* ids, float* weights,
                            hipStream_t stream);

// Super ROC QVB: LaunchQwenMoeRoute for n_rows (1..4) token rows x[r*k..],
// logits [n_rows][n_expert], ids/weights [n_rows][n_used], with the router
// read once for all rows; plus, when shexp_gate is set, each row's
// shared-expert gate sig_out[r] = sigmoid(shexp_gate . x_r) (what
// LaunchQwenMoeShexpSigmoid computes). Every output is bit-identical to the
// per-row LaunchQwenMoeRoute / LaunchQwenMoeShexpSigmoid calls. shexp_gate
// and sig_out are both null or both set. Same return contract.
int LaunchQwenMoeRouteRows(const float* x, const float* gate, int k,
                           int n_expert, int n_rows, int n_used, float* logits,
                           int* ids, float* weights, const float* shexp_gate,
                           float* sig_out, hipStream_t stream);

// Test seam (Super ROC QRH): runs only the head on caller logits.
// n_tokens == 0 -> the single-token kernel on logits[0, n_expert);
// n_tokens >= 1 -> the batch kernel on [n_tokens, n_expert] logits.
// Same return contract as above.
int LaunchQwenMoeRouteHeadForTest(const float* logits, int n_tokens,
                                  int n_expert, int n_used, int* ids,
                                  float* weights, hipStream_t stream);

#endif
