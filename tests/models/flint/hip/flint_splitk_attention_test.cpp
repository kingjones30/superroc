// Flint split-K decode attention: LaunchFlintDecodeAttention vs
// LaunchGlimmerWindowAttention. Exit 0 pass, 1 fail, 77 no HIP device.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/models/flint/hip/flint_attention.hpp"
#include "src/models/glimmer/hip/glimmer_decode.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

#if !defined(ENGINE_ENABLE_HIP)
int main() {
  std::cerr << "FAIL flint_splitk_attention_test built without HIP\n";
  return 1;
}
#else
#include <hip/hip_runtime.h>

namespace {

constexpr std::uint32_t kHeads = 32;
constexpr std::uint32_t kKv = 4;
constexpr std::uint32_t kHeadDim = 128;
constexpr std::uint32_t kMaxCtx = 4096;
constexpr std::uint32_t kLayer = 0;
constexpr std::uint32_t kSplits = 32;
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
  const int gate =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "flint_splitk_attention_test");
  if (gate != 0) {
    return gate;
  }

  std::mt19937 rng(20261024);
  std::uniform_real_distribution<float> qkv_dist(-1.0F, 1.0F);
  std::uniform_real_distribution<float> sink_dist(-2.0F, 2.0F);

  const std::size_t q_elems = static_cast<std::size_t>(kHeads) * kHeadDim;
  const std::size_t cache_elems =
      static_cast<std::size_t>(kKv) * kMaxCtx * kHeadDim;
  const std::size_t partial_elems =
      static_cast<std::size_t>(kHeads) * kSplits * (kHeadDim + 2U);
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
        *d_ref = nullptr, *d_sinks = nullptr, *d_partial = nullptr;
  std::uint32_t* d_pos = nullptr;
  if (!HipOk(hipMalloc(&d_q, q_elems * sizeof(float)), "malloc q") ||
      !HipOk(hipMalloc(&d_k, cache_elems * sizeof(float)), "malloc k") ||
      !HipOk(hipMalloc(&d_v, cache_elems * sizeof(float)), "malloc v") ||
      !HipOk(hipMalloc(&d_out, q_elems * sizeof(float)), "malloc out") ||
      !HipOk(hipMalloc(&d_ref, q_elems * sizeof(float)), "malloc ref") ||
      !HipOk(hipMalloc(&d_sinks, kHeads * sizeof(float)), "malloc sinks") ||
      !HipOk(hipMalloc(&d_partial, partial_elems * sizeof(float)),
             "malloc partial") ||
      !HipOk(hipMalloc(&d_pos, sizeof(std::uint32_t)), "malloc pos") ||
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

  const std::uint32_t kPos[] = {0, 100, 600, 1500, 4000};
  const std::uint32_t kWin[] = {0, 513};
  std::vector<float> got(q_elems);
  std::vector<float> ref(q_elems);
  for (std::uint32_t pos : kPos) {
    if (!HipOk(hipMemcpy(d_pos, &pos, sizeof(pos), hipMemcpyHostToDevice),
               "copy pos")) {
      return 1;
    }
    for (std::uint32_t window : kWin) {
      for (int with_sinks = 0; with_sinks < 2; ++with_sinks) {
        const float* sinks = with_sinks ? d_sinks : nullptr;
        const auto glim = gufo::hip::LaunchGlimmerWindowAttention(
            d_q, nullptr, d_k, d_v, d_ref, kLayer, pos, window, kMaxCtx, kHeads,
            kKv, kHeadDim, q_elems, 0, cache_elems, q_elems, nullptr, sinks);
        const auto split = gufo::hip::LaunchFlintDecodeAttention(
            d_q, d_k, d_v, d_out, kLayer, d_pos, window, kMaxCtx, kHeads, kKv,
            kHeadDim, kSplits, d_partial, sinks, nullptr);
        if (!glim.ok || !split.ok || !HipOk(hipDeviceSynchronize(), "sync") ||
            !HipOk(hipMemcpy(got.data(), d_out, q_elems * sizeof(float),
                             hipMemcpyDeviceToHost),
                   "copy got") ||
            !HipOk(hipMemcpy(ref.data(), d_ref, q_elems * sizeof(float),
                             hipMemcpyDeviceToHost),
                   "copy ref")) {
          std::cerr << "FAIL pos=" << pos << " window=" << window
                    << " sinks=" << with_sinks << " "
                    << (!glim.ok && glim.error != nullptr     ? glim.error
                        : !split.ok && split.error != nullptr ? split.error
                                                              : "copy")
                    << "\n";
          g_fail = 1;
          continue;
        }
        Report("pos=" + std::to_string(pos) + " window=" +
                   std::to_string(window) + (with_sinks ? " sinks" : " nosink"),
               MaxAbs(got.data(), ref.data(), q_elems));
      }
    }
  }

  hipFree(d_q);
  hipFree(d_k);
  hipFree(d_v);
  hipFree(d_out);
  hipFree(d_ref);
  hipFree(d_sinks);
  hipFree(d_partial);
  hipFree(d_pos);
  return g_fail;
}
#endif
