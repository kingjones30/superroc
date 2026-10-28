#ifndef SUPERROC_MODELS_QWEN_MOE_QWEN_MOE_PREFILL_HPP_
#define SUPERROC_MODELS_QWEN_MOE_QWEN_MOE_PREFILL_HPP_

#include <hip/hip_runtime.h>

// KAT S6: expert-grouped MoE prefill.
//
// LaunchKatMoeRoutedGrouped routes a whole chunk (the caller already wrote
// token-major ids/weights), sorts (token, slot) pairs by expert on the device,
// and runs one W8A8 WMMA GEMM tile per expert group: gate, up, SwiGLU, down.
// The down projection scales by the router weight. The eight partials of a
// token are then summed in ascending slot order (the same order the m <= 4
// kernel accumulates), with no atomics.
//
// LaunchKatMoeSharedDense is one dense W8A8 GEMM over the chunk for the
// sigmoid-gated shared expert. y[t] is multiplied by sig[t] afterwards.
//
// Both reuse superroc_fp4's Ling3 grouped WMMA (SUPERROC_FP4_MOE_V2 selects
// BK). FP4 types 100/101 only. Returns SUPERROC_FP4_OK (0) or a
// SUPERROC_FP4_ERR_* code. A rejected launch does not write a silent zero.
//
// Scratch is process-global and sized for a 1024-token sub-batch (8192
// slots at top-8, the parallel grouper's cap). Calls are sequential on
// `stream`: each one drains `stream` before reusing the scratch.

int LaunchKatMoeRoutedGrouped(int qtype, const void* gate, const void* up,
                              const void* down, const float* x, const int* ids,
                              const float* weights, int n_tokens, int n_used,
                              int n_expert, int hidden, int n_ff, float* y,
                              hipStream_t stream);

int LaunchKatMoeSharedDense(int qtype, const void* gate, const void* up,
                            const void* down, const float* x, const float* sig,
                            int n_tokens, int hidden, int n_ff, float* y,
                            hipStream_t stream);

#endif
