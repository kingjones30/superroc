#ifndef SUPERROC_MODELS_QWEN_HIP_DETAIL_INT4_PLAN_HPP_
#define SUPERROC_MODELS_QWEN_HIP_DETAIL_INT4_PLAN_HPP_
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "src/models/qwen/hip/detail/int4_route_file.hpp"
#include "src/models/qwen/hip/ops/prefill_int4.hpp"

namespace superroc::hip::detail {
// How one projection's input columns map onto packed K steps.
struct Int4Plan {
  std::vector<std::uint32_t>
      order;  // source column per packed column, see prefill_int4.hpp
  std::vector<float> inverse;  // one over the power of two per packed column
  Int4Shifts shifts{};
  float base = 1.F;  // weight of one accumulator unit, as a power of two
  std::size_t k_out = 0;
  std::size_t fine_columns = 0;
};

// Maps a projection's input columns onto packed K steps for its input scales.
inline Int4Plan MakeInt4Plan(const float* sigma, std::size_t k,
                             std::size_t fine_columns) {
  struct Step {
    std::array<std::uint32_t, 16> columns;
    int scale_exponent;   // the columns' power of two
    int weight_exponent;  // the power of two of this step's sum
    bool fine_input = false;
  };
  std::vector<std::uint32_t> sorted(k);
  std::iota(sorted.begin(), sorted.end(), 0U);
  std::stable_sort(
      sorted.begin(), sorted.end(),
      [&](std::uint32_t a, std::uint32_t b) { return sigma[a] > sigma[b]; });
  const auto value = [&](std::size_t i) {
    return std::max(sigma[sorted[i]], 1e-20F);
  };
  std::vector<Step> steps;
  std::size_t fine_done = 0;
  std::size_t fine_steps = 0;
  for (std::size_t i = 0; i < k;) {
    const float top = value(i);
    std::size_t n = 0;
    double energy = 0.0;
    while (n < 16 && i + n < k && value(i + n) * 2.F >= top) {
      energy += static_cast<double>(value(i + n)) * value(i + n);
      ++n;
    }
    const int exponent = static_cast<int>(
        std::lround(std::log2(std::sqrt(energy / static_cast<double>(n)))));
    Step step;
    step.columns.fill(kInt4NoColumn);
    for (std::size_t j = 0; j < n; ++j)
      step.columns[j] = sorted[i + j];
    step.scale_exponent = step.weight_exponent = exponent;
    steps.push_back(step);
    if (fine_done < fine_columns) {
      fine_done += n;
      ++fine_steps;
    }
    i += n;
  }
  const std::size_t plain_steps = steps.size();
  for (std::size_t s = 0; s < fine_steps; ++s) {
    for (const std::uint32_t flag : {kInt4FinePass, kInt4FineInput}) {
      Step fine = steps[s];
      for (auto& c : fine.columns) {
        if (c != kInt4NoColumn)
          c |= flag;
      }
      fine.weight_exponent = steps[s].scale_exponent - 4;
      fine.fine_input = flag == kInt4FineInput;
      steps.push_back(fine);
    }
  }
  // The largest sum a span can reach: every product at 49, every step full.
  const int top =
      std::max_element(steps.begin(), steps.end(),
                       [](const Step& a, const Step& b) {
                         return a.weight_exponent < b.weight_exponent;
                       })
          ->weight_exponent;
  int floor = top - 14;
  const auto bound = [&](int f) {
    double sum = 0.0;
    for (const Step& s : steps)
      sum += 16.0 * 49.0 * std::ldexp(1.0, std::max(s.weight_exponent, f) - f);
    return sum;
  };
  while (floor < top && bound(floor) >= 1073741824.0)
    ++floor;
  for (std::size_t s = 0; s < plain_steps; ++s) {
    steps[s].scale_exponent = std::max(steps[s].scale_exponent, floor);
    steps[s].weight_exponent = std::max(steps[s].weight_exponent, floor);
  }
  // A fine pass below the floor adds nothing that the sum can hold.
  steps.erase(
      std::remove_if(steps.begin() + static_cast<std::ptrdiff_t>(plain_steps),
                     steps.end(),
                     [&](const Step& s) { return s.weight_exponent < floor; }),
      steps.end());
  std::stable_sort(steps.begin(), steps.end(),
                   [](const Step& a, const Step& b) {
                     return a.weight_exponent > b.weight_exponent;
                   });
  // A finer input pass rests on the plain pass's rounding, so it cannot run at
  // another power of two. Past the last shift slot it goes.
  std::vector<int> exponents;
  for (const Step& s : steps) {
    if (exponents.empty() || exponents.back() != s.weight_exponent)
      exponents.push_back(s.weight_exponent);
  }
  if (exponents.size() > kInt4MaxShifts + 1) {
    const int last = exponents[kInt4MaxShifts];
    steps.erase(std::remove_if(steps.begin(), steps.end(),
                               [&](const Step& s) {
                                 return s.fine_input &&
                                        s.weight_exponent < last;
                               }),
                steps.end());
  }
  Int4Plan plan;
  int previous = steps.front().weight_exponent;
  for (std::size_t s = 0; s < steps.size(); ++s) {
    int exponent = steps[s].weight_exponent;
    if (exponent != previous &&
        plan.shifts.count == static_cast<std::int32_t>(kInt4MaxShifts)) {
      // Out of shift slots: the rest run at the last power of two.
      steps[s].scale_exponent += previous - exponent;
      exponent = previous;
    }
    if (exponent != previous) {
      plan.shifts.at[plan.shifts.count] = static_cast<std::int32_t>(s);
      plan.shifts.bits[plan.shifts.count] = previous - exponent;
      ++plan.shifts.count;
    }
    previous = exponent;
    const float inverse = std::ldexp(1.F, -steps[s].scale_exponent);
    for (const auto c : steps[s].columns) {
      plan.order.push_back(c);
      plan.inverse.push_back(inverse);
      if (c != kInt4NoColumn && (c & kInt4FinePass) != 0U)
        ++plan.fine_columns;
    }
  }
  std::size_t count = steps.size();
  while (count % kInt4BlockSteps != 0) {
    for (std::size_t j = 0; j < 16; ++j) {
      plan.order.push_back(kInt4NoColumn);
      plan.inverse.push_back(1.F);
    }
    ++count;
  }
  plan.base = std::ldexp(1.F, previous);
  plan.k_out = count * 16;
  return plan;
}

}  // namespace superroc::hip::detail
#endif  // SUPERROC_MODELS_QWEN_HIP_DETAIL_INT4_PLAN_HPP_
