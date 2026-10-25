#ifndef GUFO_MODELS_QWEN_MOE_QWEN_MOE_FFN_HPP_
#define GUFO_MODELS_QWEN_MOE_QWEN_MOE_FFN_HPP_

#include <hip/hip_runtime.h>

// KAT S3/S4: the two small kernels of the MoE FFN that the rocm_fp4 expert
// path does not already cover.
//
// LaunchQwenMoeShexpSigmoid: sig_out[t] = sigmoid(gate_vec . x[t]) -- the
// shared expert's scalar gate (fork qwen35moe.cpp:535-540). One block of 256
// threads per token, strided coalesced partials + a fixed tree reduce, the
// same deterministic pattern as the route logits kernel. x is
// [n_tokens, k] token-major, gate_vec is k floats. Returns 0 queued, 1
// rejected argument, 2 HIP error. Host checks: pointers non-null,
// 0 < k <= 8192, 0 < n_tokens <= 16384.
int LaunchQwenMoeShexpSigmoid(const float* x, const float* gate_vec, int k,
                              int n_tokens, float* sig_out, hipStream_t stream);

// LaunchQwenMoeCombine: out[i] = moe_y[i] + shexp_y[i] over
// n_tokens * hidden floats. The expert weights (and the shexp sigmoid gate)
// are already applied inside the expert kernels, so the combine is a plain
// add, matching ForwardMoe's final accumulation order per element. Returns
// 0 queued, 1 rejected argument, 2 HIP error.
int LaunchQwenMoeCombine(const float* moe_y, const float* shexp_y, int hidden,
                         int n_tokens, float* out, hipStream_t stream);

#endif
