#ifndef SUPERROC_CORE_TEXT_SAMPLING_DEFAULTS_HPP_
#define SUPERROC_CORE_TEXT_SAMPLING_DEFAULTS_HPP_

#include <optional>

#include "src/core/model_config.hpp"
#include "src/core/model_family.hpp"
#include "src/core/sampling.hpp"

namespace superroc::sampling {

enum class TextModelPreset {
  kUnspecified,
  kQwen38,
  kDeepSeekV4Flash,
  /// Ling-3.0-flash (bailingmoe3).
  kLing3,
};

// Presence is independent of value: an explicit zero must override a preset.
struct SamplingOverrides {
  bool temperature{false};
  bool top_k{false};
  bool top_p{false};
  bool min_p{false};
  bool min_keep{false};
  bool seed{false};
  bool repeat_penalty{false};
  bool repeat_last_n{false};
  bool frequency_penalty{false};
  bool presence_penalty{false};

  static constexpr SamplingOverrides All() noexcept {
    return {true, true, true, true, true, true, true, true, true, true};
  }
};

inline TextModelPreset TextPreset(const core::ModelConfig& config) {
  if (superroc::core::IsModelFamily(config.architecture,
                                    superroc::core::ModelFamilyId::kDeepSeek4))
    return TextModelPreset::kDeepSeekV4Flash;
  if (superroc::core::IsModelFamily(config.architecture,
                                    superroc::core::ModelFamilyId::kQwen4Exp) ||
      (config.num_layers == 64 && config.hidden_size == 5120 &&
       config.vocab_size == 248320))
    return TextModelPreset::kQwen38;
  return TextModelPreset::kUnspecified;
}

inline SamplingConfig ResolveTextSampling(TextModelPreset model,
                                          std::optional<bool> thinking,
                                          const SamplingConfig& configured,
                                          const SamplingOverrides& supplied) {
  if (model == TextModelPreset::kUnspecified)
    return configured;
  SamplingConfig result;
  // Presets change sampling controls, never the caller's output grammar.
  result.constraint = configured.constraint;
  result.temperature = 1.0F;
  result.top_p = 0.95F;
  if (model == TextModelPreset::kQwen38) {
    result.top_k = 20;
    if (!thinking.value_or(true)) {
      result.temperature = 0.7F;
      result.top_p = 0.8F;
      result.presence_penalty = 1.5F;
    }
  }
  // Jay 2026-10-01: Ling-3.0-flash fell through to the generic 1.0 / 0.95 / no
  // top-k, hot enough to drift into DeepSeek-style tool tags
  // (<\uFF5Ctool\uFF5C>, <\uFF5Ctool\u2581call\uFF5C>, ...) that no parser
  // accepts. King's fork seat runs Ling at temperature 0.6, top-p 0.95, top-k
  // 20; use the same. A request's own sampling fields still win.
  if (model == TextModelPreset::kLing3) {
    result.temperature = 0.6F;
    result.top_k = 20;
  }
  if (supplied.temperature)
    result.temperature = configured.temperature;
  if (supplied.top_k)
    result.top_k = configured.top_k;
  if (supplied.top_p)
    result.top_p = configured.top_p;
  if (supplied.min_p)
    result.min_p = configured.min_p;
  if (supplied.min_keep)
    result.min_keep = configured.min_keep;
  if (supplied.seed)
    result.seed = configured.seed;
  if (supplied.repeat_penalty)
    result.repeat_penalty = configured.repeat_penalty;
  if (supplied.repeat_last_n)
    result.repeat_last_n = configured.repeat_last_n;
  if (supplied.frequency_penalty)
    result.frequency_penalty = configured.frequency_penalty;
  if (supplied.presence_penalty)
    result.presence_penalty = configured.presence_penalty;
  return result;
}

}  // namespace superroc::sampling

#endif
