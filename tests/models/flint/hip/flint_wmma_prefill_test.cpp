// Flint WMMA prefill attention: T3d sequence (fp16 cast +
// LaunchQwenWmmaAttention) vs LaunchGlimmerPrefillAttention. Exit 0 pass, 1
// fail, 77 no HIP device.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "src/models/glimmer/hip/glimmer_attention.hpp"
#include "src/models/glimmer/hip/glimmer_decode.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

#if !defined(ENGINE_ENABLE_HIP)
int main() {
  std::cerr << "FAIL flint_wmma_prefill_test built without HIP\n";
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
constexpr std::uint32_t kBatch = 700;
constexpr float kTol = 3.0e-3F;

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
  const int gate = superroc::test::GateHipDevice(
      superroc::test::HipDeviceRequirement::kOptional,
      "flint_wmma_prefill_test");
  if (gate != 0) {
    return gate;
  }

  std::mt19937 rng(20261024);
  std::uniform_real_distribution<float> qkv_dist(-1.0F, 1.0F);
  std::uniform_real_distribution<float> sink_dist(-2.0F, 2.0F);

  const std::size_t q_row = static_cast<std::size_t>(kHeads) * kHeadDim;
  const std::size_t q_elems = q_row * kBatch;
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

  float *d_q = nullptr, *d_cache = nullptr, *d_out = nullptr, *d_ref = nullptr,
        *d_sinks = nullptr;
  void* d_f16 = nullptr;
  if (!HipOk(hipMalloc(&d_q, q_elems * sizeof(float)), "malloc q") ||
      !HipOk(hipMalloc(&d_cache, cache_elems * 2 * sizeof(float)),
             "malloc cache") ||
      !HipOk(hipMalloc(&d_out, q_elems * sizeof(float)), "malloc out") ||
      !HipOk(hipMalloc(&d_ref, q_elems * sizeof(float)), "malloc ref") ||
      !HipOk(hipMalloc(&d_sinks, kHeads * sizeof(float)), "malloc sinks") ||
      !HipOk(hipMalloc(&d_f16, cache_elems * 2 * sizeof(std::uint16_t)),
             "malloc f16") ||
      !HipOk(hipMemcpy(d_q, h_q.data(), q_elems * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy q") ||
      !HipOk(hipMemcpy(d_cache, h_k.data(), cache_elems * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy k") ||
      !HipOk(hipMemcpy(d_cache + cache_elems, h_v.data(),
                       cache_elems * sizeof(float), hipMemcpyHostToDevice),
             "copy v") ||
      !HipOk(hipMemcpy(d_sinks, h_sinks.data(), kHeads * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy sinks")) {
    return 1;
  }
  float* d_v = d_cache + cache_elems;
  void* v_f16 = static_cast<std::uint16_t*>(d_f16) + cache_elems;

  const std::uint32_t kStart[] = {0, 900};
  const std::uint32_t kWin[] = {0, 513};
  std::vector<float> got(q_elems);
  std::vector<float> ref(q_elems);
  for (std::uint32_t start_pos : kStart) {
    for (std::uint32_t window : kWin) {
      const auto glim = superroc::hip::LaunchGlimmerPrefillAttention(
          d_q, nullptr, d_cache, d_v, d_ref, kLayer, start_pos, kBatch, window,
          kMaxCtx, kHeads, kKv, kHeadDim, q_elems, 0, cache_elems, q_elems,
          nullptr, d_sinks);
      const auto cast = superroc::hip::LaunchGlimmerCastKvFp16(
          d_cache, d_f16, /*layers=*/1, kMaxCtx, kKv, kHeadDim,
          start_pos + kBatch, nullptr, kLayer, /*plane_layers=*/1);
      const bool wmma = superroc::hip::LaunchQwenWmmaAttention(
          d_q, nullptr, nullptr, nullptr, nullptr, nullptr, d_f16, v_f16, d_out,
          kLayer, start_pos, kBatch, kMaxCtx, kHeads, kKv, kHeadDim, nullptr,
          nullptr, 0, /*skip_kv_write=*/true, {}, {}, window, d_sinks);
      if (!glim.ok || !cast.ok || !wmma ||
          !HipOk(hipDeviceSynchronize(), "sync") ||
          !HipOk(hipMemcpy(got.data(), d_out, q_elems * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "copy got") ||
          !HipOk(hipMemcpy(ref.data(), d_ref, q_elems * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "copy ref")) {
        std::cerr << "FAIL start=" << start_pos << " window=" << window << " "
                  << (!glim.ok && glim.error != nullptr   ? glim.error
                      : !cast.ok && cast.error != nullptr ? cast.error
                      : !wmma                             ? "wmma refused"
                                                          : "copy")
                  << "\n";
        g_fail = 1;
        continue;
      }
      Report("start=" + std::to_string(start_pos) +
                 " window=" + std::to_string(window),
             MaxAbs(got.data(), ref.data(), q_elems));
    }
  }

  // PPL / long prefill uses batch >= 1024, which takes the packed-head WMMA
  // path (FFN scratch as workspace). T3e only covered batch 700 (canonical).
  constexpr std::uint32_t kPackedBatch = 1024;
  constexpr float kPackedIdentTol = 2.0e-5F;
  const std::size_t packed_q_elems = q_row * kPackedBatch;
  std::vector<float> h_pq(packed_q_elems);
  for (float& x : h_pq) {
    x = qkv_dist(rng);
  }
  // Real Flint sinks are O(1..12), not U(-2,2).
  const float kRealSinks[] = {7.91F, 2.97F, 3.77F,  4.91F,  4.28F, 3.06F,
                              1.55F, 5.66F, 10.94F, -2.28F, 9.75F, 0.77F,
                              3.55F, 8.69F, 12.38F, -9.00F};
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    h_sinks[h] = kRealSinks[h % 16];
  }
  float *d_pq = nullptr, *d_pout = nullptr, *d_punp = nullptr,
        *d_kwork = nullptr, *d_vwork = nullptr;
  const std::size_t length = kPackedBatch;
  const std::size_t k_floats = (length * static_cast<std::size_t>(kHeadDim) *
                                sizeof(std::uint16_t) * kKv) /
                               sizeof(float);
  const std::size_t v_floats =
      (((length + 15u) & ~std::size_t{15}) *
       static_cast<std::size_t>(kHeadDim) * sizeof(std::uint16_t) * kKv) /
      sizeof(float);
  if (!HipOk(hipMalloc(&d_pq, packed_q_elems * sizeof(float)), "malloc pq") ||
      !HipOk(hipMalloc(&d_pout, packed_q_elems * sizeof(float)),
             "malloc pout") ||
      !HipOk(hipMalloc(&d_punp, packed_q_elems * sizeof(float)),
             "malloc punp") ||
      !HipOk(hipMalloc(&d_kwork, k_floats * sizeof(float)), "malloc kwork") ||
      !HipOk(hipMalloc(&d_vwork, v_floats * sizeof(float)), "malloc vwork") ||
      !HipOk(hipMemcpy(d_pq, h_pq.data(), packed_q_elems * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy pq") ||
      !HipOk(hipMemcpy(d_sinks, h_sinks.data(), kHeads * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy real sinks")) {
    return 1;
  }
  std::vector<float> packed_got(packed_q_elems);
  std::vector<float> unpacked_got(packed_q_elems);
  std::vector<float> packed_ref(packed_q_elems);
  hipFree(d_out);
  d_out = nullptr;
  float* d_glim = nullptr;
  if (!HipOk(hipMalloc(&d_glim, packed_q_elems * sizeof(float)),
             "malloc glim")) {
    return 1;
  }
  const std::span<float> k_span{d_kwork, k_floats};
  const std::span<float> v_span{d_vwork, v_floats};
  for (std::uint32_t window : kWin) {
    const auto glim = superroc::hip::LaunchGlimmerPrefillAttention(
        d_pq, nullptr, d_cache, d_v, d_glim, kLayer, 0, kPackedBatch, window,
        kMaxCtx, kHeads, kKv, kHeadDim, packed_q_elems, 0, cache_elems,
        packed_q_elems, nullptr, d_sinks);
    const auto cast = superroc::hip::LaunchGlimmerCastKvFp16(
        d_cache, d_f16, /*layers=*/1, kMaxCtx, kKv, kHeadDim, kPackedBatch,
        nullptr, kLayer, /*plane_layers=*/1);
    const bool unpacked = superroc::hip::LaunchQwenWmmaAttention(
        d_pq, nullptr, nullptr, nullptr, nullptr, nullptr, d_f16, v_f16, d_punp,
        kLayer, 0, kPackedBatch, kMaxCtx, kHeads, kKv, kHeadDim, nullptr,
        nullptr, 0, /*skip_kv_write=*/true, {}, {}, window, d_sinks);
    const bool packed = superroc::hip::LaunchQwenWmmaAttention(
        d_pq, nullptr, nullptr, nullptr, nullptr, nullptr, d_f16, v_f16, d_pout,
        kLayer, 0, kPackedBatch, kMaxCtx, kHeads, kKv, kHeadDim, nullptr,
        nullptr, 0, /*skip_kv_write=*/true, k_span, v_span, window, d_sinks);
    if (!glim.ok || !cast.ok || !unpacked || !packed ||
        !HipOk(hipDeviceSynchronize(), "packed sync") ||
        !HipOk(hipMemcpy(packed_got.data(), d_pout,
                         packed_q_elems * sizeof(float), hipMemcpyDeviceToHost),
               "copy packed") ||
        !HipOk(hipMemcpy(unpacked_got.data(), d_punp,
                         packed_q_elems * sizeof(float), hipMemcpyDeviceToHost),
               "copy unpacked") ||
        !HipOk(hipMemcpy(packed_ref.data(), d_glim,
                         packed_q_elems * sizeof(float), hipMemcpyDeviceToHost),
               "copy glim")) {
      std::cerr << "FAIL packed window=" << window << " "
                << (!glim.ok && glim.error != nullptr   ? glim.error
                    : !cast.ok && cast.error != nullptr ? cast.error
                    : !unpacked                         ? "unpacked refused"
                    : !packed                           ? "packed refused"
                                                        : "copy")
                << "\n";
      g_fail = 1;
      continue;
    }
    const float vs_glim =
        MaxAbs(packed_got.data(), packed_ref.data(), packed_q_elems);
    const float unp_glim =
        MaxAbs(unpacked_got.data(), packed_ref.data(), packed_q_elems);
    const float vs_unp =
        MaxAbs(packed_got.data(), unpacked_got.data(), packed_q_elems);
    Report("packed vs glimmer window=" + std::to_string(window), vs_glim);
    Report("unpacked vs glimmer window=" + std::to_string(window), unp_glim);
    const bool ident = vs_unp <= kPackedIdentTol;
    std::cout << (ident ? "PASS " : "FAIL ")
              << "packed vs unpacked window=" << window << " max_abs=" << vs_unp
              << "\n";
    if (!ident) {
      g_fail = 1;
    }
  }

  hipFree(d_q);
  hipFree(d_cache);
  hipFree(d_ref);
  hipFree(d_sinks);
  hipFree(d_f16);
  hipFree(d_pq);
  hipFree(d_pout);
  hipFree(d_punp);
  hipFree(d_kwork);
  hipFree(d_vwork);
  hipFree(d_glim);
  return g_fail;
}
#endif
