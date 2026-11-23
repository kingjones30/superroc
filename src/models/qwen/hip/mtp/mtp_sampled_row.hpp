#ifndef SUPERROC_MODELS_QWEN_HIP_MTP_MTP_SAMPLED_ROW_HPP_
#define SUPERROC_MODELS_QWEN_HIP_MTP_MTP_SAMPLED_ROW_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace superroc::hip {

// MTP sampled proposals. Scores the row's top-K finite logits, renormalizes
// them at this temperature and draws one token from that row. ids and probs
// are appended in descending logit order and always hold k rows: the row
// reports the distribution actually sampled. Verification rejects and
// corrects, so the output distribution stays the target's.
// Throws when fewer than k logits are finite.
template<class Logits>
[[nodiscard]] tokenization::TokenId DrawSampledRow(
    Logits logits, float temperature, std::size_t k, std::uint64_t* rng,
    std::vector<tokenization::TokenId>& ids, std::vector<float>& probs) {
  std::vector<std::pair<float, std::uint32_t>> top;
  for (std::uint32_t id = 0; id < logits.size(); ++id) {
    const float value = logits[id];
    if (!std::isfinite(value))
      continue;
    if (top.size() < k) {
      top.emplace_back(value, id);
      std::push_heap(top.begin(), top.end(), std::greater<>{});
    } else if (value > top.front().first) {
      std::pop_heap(top.begin(), top.end(), std::greater<>{});
      top.back() = {value, id};
      std::push_heap(top.begin(), top.end(), std::greater<>{});
    }
  }
  std::sort(top.begin(), top.end(), std::greater<>{});
  if (top.size() != k)
    throw std::runtime_error("MTP sampled proposal has too few scored rows");
  const float peak = top.front().first;
  double total = 0.0;
  std::array<double, 64> weight{};
  for (std::size_t i = 0; i < top.size(); ++i) {
    weight[i] =
        std::exp(static_cast<double>(top[i].first - peak) / temperature);
    total += weight[i];
  }
  double u = sampling::Uniform(rng) * total;
  std::size_t chosen = top.size() - 1;
  for (std::size_t i = 0; i < top.size(); ++i) {
    if (u < weight[i]) {
      chosen = i;
      break;
    }
    u -= weight[i];
  }
  for (std::size_t i = 0; i < top.size(); ++i) {
    ids.push_back(static_cast<tokenization::TokenId>(top[i].second));
    probs.push_back(static_cast<float>(weight[i] / total));
  }
  return static_cast<tokenization::TokenId>(top[chosen].second);
}

}  // namespace superroc::hip

#endif  // SUPERROC_MODELS_QWEN_HIP_MTP_MTP_SAMPLED_ROW_HPP_
