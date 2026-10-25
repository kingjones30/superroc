#ifndef GUFO_MODELS_QWEN_HIP_MTP_KAT_MTP_PMIN_HPP_
#define GUFO_MODELS_QWEN_HIP_MTP_KAT_MTP_PMIN_HPP_

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <span>

namespace gufo::hip {

// KAT-MTP: the effective GUFO_MTP_PMIN for this call, clamped to [0, 1].
// The single source of truth for the greedy p-min gate (mtp/backend.cpp)
// and the /health summary, so the two can never disagree.
[[nodiscard]] inline float MtpPMinFromEnv() {
  float p_min = 0.0F;
  if (const char* env = std::getenv("GUFO_MTP_PMIN");
      env != nullptr && *env != '\0') {
    char* end = nullptr;
    p_min = std::strtof(env, &end);
    // RV44 cosmetic: an unparsable value used to parse as 0 and silently
    // disable the gate. Log it once and keep the documented default (0 = no
    // p-min gate) instead of staying silent.
    if (end == env || std::isnan(p_min)) {
      static std::atomic<bool> warned{false};
      if (!warned.exchange(true)) {
        std::fprintf(stderr,
                     "KAT-MTP: GUFO_MTP_PMIN='%s' is not a number; using 0 "
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

// Fork draft-mtp probability: softmax over the top-10 logits, value of the
// maximum. common_sampler_sample_top_k_probs(..., MTP_DRAFT_TOP_K=10).
// The gate compares this with common_speculative_effective_p_min.
[[nodiscard]] inline float KatMtpTop1Probability(
    std::span<const float> logits) {
  if (logits.empty()) {
    return 0.0F;
  }
  const std::size_t k = std::min<std::size_t>(10, logits.size());
  float top[10];
  std::size_t filled = 0;
  for (const float value : logits) {
    if (filled < k) {
      top[filled] = value;
      std::size_t index = filled;
      while (index > 0 && top[index] > top[index - 1]) {
        std::swap(top[index], top[index - 1]);
        --index;
      }
      ++filled;
      continue;
    }
    if (!(value > top[k - 1])) {
      continue;
    }
    top[k - 1] = value;
    std::size_t index = k - 1;
    while (index > 0 && top[index] > top[index - 1]) {
      std::swap(top[index], top[index - 1]);
      --index;
    }
  }
  const float peak = top[0];
  double sum = 0.0;
  for (std::size_t index = 0; index < filled; ++index) {
    sum += std::exp(static_cast<double>(top[index] - peak));
  }
  if (!(sum > 0.0)) {
    return 0.0F;
  }
  return static_cast<float>(1.0 / sum);
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_MTP_KAT_MTP_PMIN_HPP_
