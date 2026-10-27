#ifndef SUPERROC_MODELS_QWEN_HIP_DFLASH_DSPARK_SAMPLING_HPP_
#define SUPERROC_MODELS_QWEN_HIP_DFLASH_DSPARK_SAMPLING_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace superroc::hip::kernels {

/// Adds the BF16 markov bonus `w1[prev] · w2[v]` into each proposal row of
/// `logits`, then picks top-k (greedy or temperature-sampled) and writes
/// `tokens[1..]`. Confidence is `sigmoid(proj · [hidden ‖ w1[prev]] + bias)`.
void LaunchDSparkMarkovSelect(
    const float* logits, const float* hidden, const void* markov_w1_bf16,
    const void* markov_w2_bf16, const float* conf_proj, const float* conf_bias,
    std::uint32_t* tokens, float* confidences, float* partial_scores,
    std::uint32_t* partial_ids, const float* uniforms,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::uint32_t rows, std::uint32_t vocab_size, std::uint32_t hidden_size,
    std::uint32_t markov_rank, std::uint32_t top_k, float temperature,
    hipStream_t stream);

/// Serial markov chain for `n_slots` in one kernel launch. `tokens[0]` is the
/// anchor; writes `tokens[1..n_slots]`. Slot k conditions on `tokens[k]`.
void LaunchDSparkMarkovChain(
    const float* logits, const float* hidden, const void* markov_w1_bf16,
    const void* markov_w2_bf16, const float* conf_proj, const float* conf_bias,
    std::uint32_t* tokens, float* confidences, float* partial_scores,
    std::uint32_t* partial_ids, const float* uniforms,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::uint32_t n_slots, std::uint32_t vocab_size, std::uint32_t hidden_size,
    std::uint32_t markov_rank, std::uint32_t top_k, float temperature,
    hipStream_t stream);

}  // namespace superroc::hip::kernels
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // SUPERROC_MODELS_QWEN_HIP_DFLASH_DSPARK_SAMPLING_HPP_
