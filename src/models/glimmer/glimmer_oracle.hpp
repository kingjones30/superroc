#ifndef GUFO_MODELS_GLIMMER_GLIMMER_ORACLE_HPP_
#define GUFO_MODELS_GLIMMER_GLIMMER_ORACLE_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/qwen/state.hpp"

namespace gufo::models {

/// GLIMMER: RMSNorm. An empty weight is the all-ones scale (unweighted embed
/// norm). Writes `out`; does not read and write the same buffer.
void GlimmerRmsNorm(std::span<const float> x, std::span<const float> weight,
                    float eps, std::span<float> out) noexcept;

/// GLIMMER: NORM RoPE on one head. Pairs are (q[2i], q[2i+1]), not NEOX.
void GlimmerNormRope(std::span<float> head, std::uint32_t rotary_dim,
                     std::uint32_t pos, float theta) noexcept;

/// GLIMMER: first position kept by a standard window. Window 0 is full.
[[nodiscard]] constexpr std::uint32_t GlimmerSwaBegin(
    std::uint32_t pos, std::uint32_t window) noexcept {
  const std::uint32_t seq = pos + 1U;
  if (window == 0U || seq <= window) {
    return 0U;
  }
  return seq - window;
}

/// GLIMMER: y = scale * x, then cap * tanh(y / cap) when cap != 0.
void GlimmerScaleSoftcap(std::span<float> logits, float scale,
                         float cap) noexcept;

/// GLIMMER: integer Q8_1 activation dot against one ROCmFP4 row.
/// FAST (type 101) uses one UE4M3 scale per 32. Dual (type 100) uses two.
[[nodiscard]] float GlimmerQ8Dot(core::GgmlType type, const void* row,
                                 std::span<const float> x,
                                 std::size_t k) noexcept;

[[nodiscard]] double GlimmerCosine(std::span<const float> a,
                                   std::span<const float> b) noexcept;

void GlimmerForwardLayer(std::span<float> hidden, const QwenLayerWeights& layer,
                         const core::ModelConfig& config, QwenKvCache& kv_cache,
                         std::uint32_t layer_idx, std::uint32_t pos,
                         QwenScratchArena& arena) noexcept;

void GlimmerForwardModel(std::uint32_t token_id, std::uint32_t pos,
                         const QwenModelWeights& weights, QwenKvCache& kv_cache,
                         QwenScratchArena& arena,
                         std::span<float> logits_out) noexcept;

}  // namespace gufo::models

#endif  // GUFO_MODELS_GLIMMER_GLIMMER_ORACLE_HPP_
