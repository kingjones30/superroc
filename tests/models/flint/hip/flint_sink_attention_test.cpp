// Flint learned-sink attention: LaunchGlimmerWindowAttention /
// LaunchGlimmerPrefillAttention vs a CPU double reference. nullptr sinks must
// match the no-sink math. Exit 0 pass, 1 fail, 77 no HIP device.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/models/glimmer/hip/glimmer_decode.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

#if !defined(ENGINE_ENABLE_HIP)
int main() {
  std::cerr << "FAIL flint_sink_attention_test built without HIP\n";
  return 1;
}
#else
#include <hip/hip_runtime.h>

namespace {

constexpr std::uint32_t kHeads = 32;
constexpr std::uint32_t kKv = 4;
constexpr std::uint32_t kHeadDim = 128;
constexpr std::uint32_t kMaxCtx = 2048;
constexpr std::uint32_t kLayer = 0;
constexpr float kTol = 2.0e-5F;

int g_fail = 0;

bool HipOk(hipError_t err, const char* what) {
  if (err == hipSuccess) {
    return true;
  }
  std::cerr << "FAIL hip " << what << " code=" << static_cast<int>(err) << "\n";
  g_fail = 1;
  return false;
}

std::size_t KvIndex(std::uint32_t kv_h, std::uint32_t pos, std::uint32_t d) {
  return ((static_cast<std::size_t>(kLayer) * kKv + kv_h) * kMaxCtx + pos) *
             kHeadDim +
         d;
}

// CPU double reference. sinks == nullptr is the no-sink softmax. Window 513
// attends max(0, pos-512)..pos; window 0 attends 0..pos.
void CpuAttend(const float* q, const float* k, const float* v,
               const float* sinks, std::uint32_t pos, std::uint32_t window,
               float* out) {
  const std::uint32_t begin =
      (window == 0U || pos + 1U <= window) ? 0U : pos + 1U - window;
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
  const std::uint32_t group = kHeads / kKv;
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    const std::uint32_t kv_h = h / group;
    const float* qh = q + static_cast<std::size_t>(h) * kHeadDim;
    const bool have_sink = sinks != nullptr;
    const double sink = have_sink ? static_cast<double>(sinks[h]) : 0.0;
    std::vector<double> scores(pos - begin + 1U);
    double m = have_sink ? sink : -1.0e300;
    for (std::uint32_t p = begin; p <= pos; ++p) {
      const float* kp = k + KvIndex(kv_h, p, 0);
      double dot = 0.0;
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        dot += static_cast<double>(qh[d]) * static_cast<double>(kp[d]);
      }
      scores[p - begin] = dot * scale;
      m = std::max(m, scores[p - begin]);
    }
    double denom = have_sink ? std::exp(sink - m) : 0.0;
    for (double& s : scores) {
      s = std::exp(s - m);
      denom += s;
    }
    denom += 1.0e-9;
    for (std::uint32_t d = 0; d < kHeadDim; ++d) {
      double acc = 0.0;
      for (std::uint32_t p = begin; p <= pos; ++p) {
        acc += scores[p - begin] * static_cast<double>(v[KvIndex(kv_h, p, d)]);
      }
      out[static_cast<std::size_t>(h) * kHeadDim + d] =
          static_cast<float>(acc / denom);
    }
  }
}

float MaxAbs(const float* got, const float* ref, std::size_t n) {
  float gap = 0.0F;
  for (std::size_t i = 0; i < n; ++i) {
    gap = std::max(gap, std::fabs(got[i] - ref[i]));
  }
  return gap;
}

void Report(const std::string& cas, float gap) {
  const bool pass = gap <= kTol;
  std::cout << (pass ? "PASS " : "FAIL ") << cas << " max_abs=" << gap << "\n";
  if (!pass) {
    g_fail = 1;
  }
}

}  // namespace

int main() {
  const int gate = superroc::test::GateHipDevice(
      superroc::test::HipDeviceRequirement::kOptional,
      "flint_sink_attention_test");
  if (gate != 0) {
    return gate;
  }

  std::mt19937 rng(20261024);
  std::uniform_real_distribution<float> qkv_dist(-1.0F, 1.0F);
  std::uniform_real_distribution<float> sink_dist(-2.0F, 2.0F);

  const std::size_t q_elems = static_cast<std::size_t>(kHeads) * kHeadDim;
  const std::size_t cache_elems =
      static_cast<std::size_t>(kKv) * kMaxCtx * kHeadDim;
  std::vector<float> h_q(q_elems);
  std::vector<float> h_k(cache_elems);
  std::vector<float> h_v(cache_elems);
  std::vector<float> h_sinks(kHeads);
  for (float& x : h_q) {
    x = qkv_dist(rng);
  }
  for (float& x : h_k) {
    x = qkv_dist(rng);
  }
  for (float& x : h_v) {
    x = qkv_dist(rng);
  }
  for (float& x : h_sinks) {
    x = sink_dist(rng);
  }

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_out = nullptr,
        *d_sinks = nullptr;
  if (!HipOk(hipMalloc(&d_q, q_elems * sizeof(float)), "malloc q") ||
      !HipOk(hipMalloc(&d_k, cache_elems * sizeof(float)), "malloc k") ||
      !HipOk(hipMalloc(&d_v, cache_elems * sizeof(float)), "malloc v") ||
      !HipOk(hipMalloc(&d_out, q_elems * sizeof(float)), "malloc out") ||
      !HipOk(hipMalloc(&d_sinks, kHeads * sizeof(float)), "malloc sinks") ||
      !HipOk(hipMemcpy(d_q, h_q.data(), q_elems * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy q") ||
      !HipOk(hipMemcpy(d_k, h_k.data(), cache_elems * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy k") ||
      !HipOk(hipMemcpy(d_v, h_v.data(), cache_elems * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy v") ||
      !HipOk(hipMemcpy(d_sinks, h_sinks.data(), kHeads * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy sinks")) {
    return 1;
  }

  const std::uint32_t kPos[] = {0, 100, 600, 1500};
  const std::uint32_t kWin[] = {0, 513};
  std::vector<float> got(q_elems);
  std::vector<float> ref(q_elems);
  for (std::uint32_t pos : kPos) {
    for (std::uint32_t window : kWin) {
      CpuAttend(h_q.data(), h_k.data(), h_v.data(), h_sinks.data(), pos, window,
                ref.data());
      const auto st = superroc::hip::LaunchGlimmerWindowAttention(
          d_q, nullptr, d_k, d_v, d_out, kLayer, pos, window, kMaxCtx, kHeads,
          kKv, kHeadDim, q_elems, 0, cache_elems, q_elems, nullptr, d_sinks);
      if (!st.ok || !HipOk(hipDeviceSynchronize(), "decode sync") ||
          !HipOk(hipMemcpy(got.data(), d_out, q_elems * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "decode copy")) {
        std::cerr << "FAIL decode pos=" << pos << " window=" << window << " "
                  << (st.error != nullptr ? st.error : "copy") << "\n";
        g_fail = 1;
        continue;
      }
      Report("decode pos=" + std::to_string(pos) +
                 " window=" + std::to_string(window),
             MaxAbs(got.data(), ref.data(), q_elems));
    }
  }

  constexpr std::uint32_t kStart = 900;
  constexpr std::uint32_t kBatch = 700;
  const std::size_t prefill_elems = q_elems * kBatch;
  std::vector<float> h_pq(prefill_elems);
  for (float& x : h_pq) {
    x = qkv_dist(rng);
  }
  float *d_pq = nullptr, *d_pout = nullptr;
  if (!HipOk(hipMalloc(&d_pq, prefill_elems * sizeof(float)), "malloc pq") ||
      !HipOk(hipMalloc(&d_pout, prefill_elems * sizeof(float)),
             "malloc pout") ||
      !HipOk(hipMemcpy(d_pq, h_pq.data(), prefill_elems * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy pq")) {
    return 1;
  }
  std::vector<float> pgot(prefill_elems);
  const std::uint32_t kCheck[] = {0, 350, 699};
  for (std::uint32_t window : kWin) {
    const auto st = superroc::hip::LaunchGlimmerPrefillAttention(
        d_pq, nullptr, d_k, d_v, d_pout, kLayer, kStart, kBatch, window,
        kMaxCtx, kHeads, kKv, kHeadDim, prefill_elems, 0, cache_elems,
        prefill_elems, nullptr, d_sinks);
    if (!st.ok || !HipOk(hipDeviceSynchronize(), "prefill sync") ||
        !HipOk(hipMemcpy(pgot.data(), d_pout, prefill_elems * sizeof(float),
                         hipMemcpyDeviceToHost),
               "prefill copy")) {
      std::cerr << "FAIL prefill window=" << window << " "
                << (st.error != nullptr ? st.error : "copy") << "\n";
      g_fail = 1;
      continue;
    }
    for (std::uint32_t row : kCheck) {
      CpuAttend(h_pq.data() + static_cast<std::size_t>(row) * q_elems,
                h_k.data(), h_v.data(), h_sinks.data(), kStart + row, window,
                ref.data());
      Report("prefill row=" + std::to_string(row) +
                 " window=" + std::to_string(window),
             MaxAbs(pgot.data() + static_cast<std::size_t>(row) * q_elems,
                    ref.data(), q_elems));
    }
  }

  CpuAttend(h_q.data(), h_k.data(), h_v.data(), nullptr, 600, 513, ref.data());
  const auto nosink = superroc::hip::LaunchGlimmerWindowAttention(
      d_q, nullptr, d_k, d_v, d_out, kLayer, 600, 513, kMaxCtx, kHeads, kKv,
      kHeadDim, q_elems, 0, cache_elems, q_elems, nullptr, nullptr);
  if (!nosink.ok || !HipOk(hipDeviceSynchronize(), "nosink sync") ||
      !HipOk(hipMemcpy(got.data(), d_out, q_elems * sizeof(float),
                       hipMemcpyDeviceToHost),
             "nosink copy")) {
    std::cerr << "FAIL nosink "
              << (nosink.error != nullptr ? nosink.error : "copy") << "\n";
    g_fail = 1;
  } else {
    Report("nosink pos=600 window=513",
           MaxAbs(got.data(), ref.data(), q_elems));
  }

  hipFree(d_q);
  hipFree(d_k);
  hipFree(d_v);
  hipFree(d_out);
  hipFree(d_sinks);
  hipFree(d_pq);
  hipFree(d_pout);
  return g_fail;
}
#endif
