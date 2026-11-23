#ifndef SUPERROC_MODELS_QWEN_HIP_MTP_KAT_MTP_PMIN_HPP_
#define SUPERROC_MODELS_QWEN_HIP_MTP_KAT_MTP_PMIN_HPP_

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <span>

#include "src/core/env.hpp"

namespace superroc::hip {

// KAT-MTP: the effective SUPERROC_MTP_PMIN for this call, clamped to [0, 1].
// The single source of truth for the greedy p-min gate (mtp/backend.cpp)
// and the /health summary, so the two can never disagree. A draft token whose
// probability under the draft head is below it ends the draft there.
[[nodiscard]] inline float MtpPMinFromEnv() {
  float p_min = 0.0F;
  if (const char* env = ::superroc::core::GetEnv("SUPERROC_MTP_PMIN");
      env != nullptr && *env != '\0') {
    char* end = nullptr;
    p_min = std::strtof(env, &end);
    // Cosmetic: an unparsable value used to parse as 0 and silently
    // disable the gate. Log it once and keep the documented default (0 = no
    // p-min gate) instead of staying silent.
    if (end == env || std::isnan(p_min)) {
      static std::atomic<bool> warned{false};
      if (!warned.exchange(true)) {
        std::fprintf(stderr,
                     "KAT-MTP: SUPERROC_MTP_PMIN='%s' is not a number; using 0 "
                     "(no p-min gate)\n",
                     env);
      }
      p_min = 0.0F;
    }
  }
  if (p_min < 0.0F) {
    p_min = 0.0F;
  } else if (p_min > 1.0F) {
    p_min = 1.0F;
  }
  return p_min;
}

/// The p-min a dense MTP block uses when SUPERROC_MTP_PMIN is unset. With a
/// draft step this cheap next to a verification, a deep draft that stops at
/// the first unsure token beats any fixed depth (Qwen3.8-27B). A MoE block
/// keeps 0 (no gate) until it is measured.
inline constexpr float kMtpDensePMinDefault = 0.40F;

/// The draft length a dense MTP block uses when none is given. A drafted
/// token costs a draft step and a verification row, and the row grows with
/// the context; on Qwen3.8-27B four tokens with the gate above decoded as
/// fast as seven on a chat prompt and faster behind a long prompt.
inline constexpr std::uint32_t kMtpDenseDraftTokensDefault = 4;

/// The p-min in force for a block: SUPERROC_MTP_PMIN when set, else the
/// default for its kind.
[[nodiscard]] inline float MtpEffectivePMin(bool moe_block) {
  const char* env = ::superroc::core::GetEnv("SUPERROC_MTP_PMIN");
  if (env == nullptr || *env == '\0') {
    return moe_block ? 0.0F : kMtpDensePMinDefault;
  }
  return MtpPMinFromEnv();
}

// Probability of the largest logit under a softmax over all of `logits`.
// The host reference of what the draft computes on the device
// (MtpTop1ProbabilityKernel) for the p-min gate; logits that are not finite
// add nothing.
[[nodiscard]] inline float MtpTop1Probability(std::span<const float> logits) {
  bool found = false;
  float peak = 0.0F;
  for (const float value : logits) {
    if (std::isfinite(value) && (!found || value > peak)) {
      peak = value;
      found = true;
    }
  }
  if (!found) {
    return 0.0F;
  }
  double sum = 0.0;
  for (const float value : logits) {
    if (std::isfinite(value)) {
      sum += std::exp(static_cast<double>(value - peak));
    }
  }
  return static_cast<float>(1.0 / sum);
}

}  // namespace superroc::hip

#endif  // SUPERROC_MODELS_QWEN_HIP_MTP_KAT_MTP_PMIN_HPP_
