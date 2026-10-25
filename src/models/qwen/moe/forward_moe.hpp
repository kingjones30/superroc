#ifndef GUFO_MODELS_QWEN_MOE_FORWARD_MOE_HPP_
#define GUFO_MODELS_QWEN_MOE_FORWARD_MOE_HPP_

#include <cstddef>
#include <span>

#include "src/core/model_config.hpp"
#include "src/models/qwen/state.hpp"

namespace gufo::models {

/// KAT: bytes in one expert row. F32 is k·4, F16 and BF16 are k·2, and a
/// quantized type uses QuantizedRowBytes. Returns 0 when the row cannot be
/// addressed (unknown type, or a quantized row that is not block-aligned).
std::size_t KatExpertRowBytes(core::GgmlType type, int k) noexcept;

/// KAT: one expert's matrix inside a stacked expert tensor.
/// expert < 0 returns the whole tensor (the shared expert).
/// A null data pointer or a 0 stride aborts. It does not return a null slice:
/// a null slice used to leave the previous expert's scratch in place.
QwenTensorRef KatExpertSlice(const QwenTensorRef& tensor, int expert,
                             int n_rows, int k);

/// KAT: softmax over all router logits, then the fork's bitonic DESC top-k.
/// Weights are those probabilities divided by their sum (fork norm_w).
/// The sum is float, in ascending top-k slot order (bitonic rank 0 first),
/// and the divisor is max(sum, 2^-14) = 6.103515625e-5. The KAT GGUF has no
/// expert_weights_scale, so the fork's w_scale multiply is not applied.
/// This is also the MTP block's route when that block is executed.
/// Returns false when the shape is outside the stack-bounded router (256).
bool KatRouteSoftmaxTopK(std::span<const float> logits, int n_used,
                         std::span<int> ids, std::span<float> weights) noexcept;

/// KAT: one routed expert, or the shared expert when expert < 0.
/// gate/up are [ff, hidden] per expert (K = hidden). down is [hidden, ff].
/// A bad shape or a null expert slice aborts with a message on stderr.
void KatExpertSwiglu(const QwenTensorRef& gate, const QwenTensorRef& up,
                     const QwenTensorRef& down, int expert,
                     std::span<const float> x, std::size_t hidden,
                     std::size_t ff, std::span<float> gate_scratch,
                     std::span<float> up_scratch, std::span<float> act_scratch,
                     std::span<float> hidden_out);

/// KAT: routed top-k plus sigmoid-gated shared expert. Residual stays with
/// the caller. `expert_hidden` is hidden-sized scratch.
/// A bad shape, a failed route, or a null expert slice aborts. It does not
/// return a zeroed vector.
void ForwardMoe(std::span<const float> x, const QwenLayerWeights& layer,
                const core::ModelConfig& config, std::span<float> logits,
                std::span<float> probs, std::span<float> gate_scratch,
                std::span<float> up_scratch, std::span<float> act_scratch,
                std::span<float> expert_hidden, std::span<float> out);

}  // namespace gufo::models

#endif  // GUFO_MODELS_QWEN_MOE_FORWARD_MOE_HPP_
