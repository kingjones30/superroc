#ifndef SUPERROC_TOOLS_PPL_REF_HPP_
#define SUPERROC_TOOLS_PPL_REF_HPP_
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include "src/core/env.hpp"

namespace superroc::tools {
// SUPERROC_PPL_REF=FILE with SUPERROC_PPL_REF_MODE=write stores each scored position's top 64 log
// probabilities; without it the run reads them back and reports how far its own distributions moved
// (KL, top-1 agreement, paired NLL). Paired on the same positions, this shows differences the score hides.
struct PplRef {
  static constexpr std::size_t kTop = 64;
  struct Record {
    float nll;
    float rest;  // log probability of everything outside the top
    std::uint32_t index[kTop];
    float logp[kTop];
  };
  std::FILE* f = nullptr;
  bool writing = false;
  std::vector<double> kls, dnll;
  std::size_t agree = 0;
  std::vector<double> logp;
  void Open() {
    const char* path = ::superroc::core::GetEnv("SUPERROC_PPL_REF");
    if (path == nullptr || path[0] == '\0')
      return;
    const char* mode = ::superroc::core::GetEnv("SUPERROC_PPL_REF_MODE");
    writing = mode != nullptr && std::string(mode) == "write";
    f = std::fopen(path, writing ? "wb" : "rb");
    if (f == nullptr)
      throw std::runtime_error("SUPERROC_PPL_REF cannot be opened");
  }
  void Add(std::span<const float> view, double nll) {
    if (f == nullptr)
      return;
    double max_logit = -1e300;
    for (const float v : view)
      max_logit = std::max(max_logit, static_cast<double>(v));
    double sum_exp = 0.0;
    for (const float v : view)
      sum_exp += std::exp(static_cast<double>(v) - max_logit);
    const double norm = max_logit + std::log(sum_exp);
    if (writing) {
      std::vector<std::uint32_t> order(view.size());
      std::iota(order.begin(), order.end(), 0U);
      std::partial_sort(order.begin(), order.begin() + kTop, order.end(), [&](std::uint32_t a, std::uint32_t b) { return view[a] > view[b]; });
      Record r;
      r.nll = static_cast<float>(nll);
      double top = 0.0;
      for (std::size_t i = 0; i < kTop; ++i) {
        r.index[i] = order[i];
        const double lp = static_cast<double>(view[order[i]]) - norm;
        r.logp[i] = static_cast<float>(lp);
        top += std::exp(lp);
      }
      r.rest = static_cast<float>(std::log(std::max(1.0 - top, 1e-12)));
      std::fwrite(&r, sizeof(r), 1, f);
      return;
    }
    Record r;
    if (std::fread(&r, sizeof(r), 1, f) != 1)
      throw std::runtime_error("SUPERROC_PPL_REF is short");
    double kl = 0.0, top = 0.0;
    for (std::size_t i = 0; i < kTop; ++i) {
      const double lp = static_cast<double>(view[r.index[i]]) - norm;
      kl += std::exp(static_cast<double>(r.logp[i])) * (static_cast<double>(r.logp[i]) - lp);
      top += std::exp(lp);
    }
    const double rest = std::log(std::max(1.0 - top, 1e-12));
    kl += std::exp(static_cast<double>(r.rest)) * (static_cast<double>(r.rest) - rest);
    kls.push_back(kl);
    dnll.push_back(nll - static_cast<double>(r.nll));
    std::size_t best = 0;
    for (std::size_t i = 1; i < view.size(); ++i)
      if (view[i] > view[best])
        best = i;
    agree += best == r.index[0] ? 1 : 0;
  }
  void Report() {
    if (f == nullptr)
      return;
    std::fclose(f);
    if (writing || kls.empty())
      return;
    const double n = static_cast<double>(kls.size());
    double mk = 0.0, md = 0.0, vd = 0.0;
    for (const double v : kls) mk += v / n;
    for (const double v : dnll) md += v / n;
    for (const double v : dnll) vd += (v - md) * (v - md) / n;
    std::vector<double> s = kls;
    std::sort(s.begin(), s.end());
    std::cout << "KLREF mean=" << mk << " p50=" << s[s.size() / 2] << " p90=" << s[s.size() * 9 / 10] << " p99=" << s[s.size() * 99 / 100]
              << " top1_agree=" << static_cast<double>(agree) / n << " dnll_mean=" << md << " dnll_se=" << std::sqrt(vd / n)
              << " dnll_rms=" << std::sqrt(vd + md * md) << " n=" << kls.size() << "\n" << std::flush;
  }
};
}  // namespace superroc::tools
#endif  // SUPERROC_TOOLS_PPL_REF_HPP_
