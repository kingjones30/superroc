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
///
/// Trimmed draft vocabulary: with `head_rows > 0` each row
/// holds `head_rows` head logits followed by `vocab_size - tail_start` tail
/// logits in `tail_logits`; the scan covers the real ids [0, head_rows) and
/// [tail_start, vocab_size), and every emitted id (partial, candidate and
/// token) is a real token id. `head_rows == 0` keeps the full-vocab behaviour:
/// `logits` holds `vocab_size` per row and `tail_start` must be zero.
void LaunchDSparkMarkovSelect(
    const float* logits, std::uint32_t head_rows, std::uint32_t tail_start,
    const float* tail_logits, const float* hidden, const void* markov_w1_bf16,
    const void* markov_w2_bf16, const float* conf_proj, const float* conf_bias,
    std::uint32_t* tokens, float* confidences, float* partial_scores,
    std::uint32_t* partial_ids, const float* uniforms,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::uint32_t rows, std::uint32_t vocab_size, std::uint32_t hidden_size,
    std::uint32_t markov_rank, std::uint32_t top_k, float temperature,
    hipStream_t stream);

}  // namespace superroc::hip::kernels
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // SUPERROC_MODELS_QWEN_HIP_DFLASH_DSPARK_SAMPLING_HPP_
