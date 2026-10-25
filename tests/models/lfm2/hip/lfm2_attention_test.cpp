// LFM2 split-K decode attention GPU test: LaunchLfm2DecodeAttention vs a CPU
// double reference and vs LaunchGlimmerWindowAttention (window 0, no gate),
// plus LaunchLfm2WriteKvDev vs LaunchGlimmerWriteKv byte equality.
// Exit 0 pass, 1 fail, 77 no HIP device.
#include "src/models/lfm2/hip/lfm2_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/models/glimmer/hip/glimmer_decode.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

#if !defined(ENGINE_ENABLE_HIP)
int main() {
  std::cerr << "FAIL lfm2_attention_test built without HIP\n";
  return 1;
}
#else
#include <hip/hip_runtime.h>

namespace {

int g_fail = 0;

void Report(const std::string& cas, bool pass, const std::string& numbers) {
  std::cout << (pass ? "PASS " : "FAIL ") << cas << ": " << numbers << "\n";
  if (!pass) {
    g_fail = 1;
  }
}

bool HipOk(hipError_t err, const char* what) {
  if (err == hipSuccess) {
    return true;
  }
  std::cerr << "FAIL hip " << what << " code=" << static_cast<int>(err) << "\n";
  g_fail = 1;
  return false;
}

constexpr std::uint32_t kHeads = 32;
constexpr std::uint32_t kKvHeads = 8;
constexpr std::uint32_t kHeadDim = 64;
constexpr std::uint32_t kMaxCtx = 16384;
constexpr std::uint32_t kLayers = 2;
constexpr std::uint32_t kMaxSplits = 64;

// CPU double reference: exact softmax attention over positions 0..pos.
void CpuRef(const float* q, const float* kc, const float* vc, float* out,
            std::uint32_t layer, std::uint32_t pos) {
  const std::uint32_t group = kHeads / kKvHeads;
  const double scale = 1.0 / std::sqrt(64.0);
  const std::size_t plane =
      static_cast<std::size_t>(kLayers) * kKvHeads * kMaxCtx * kHeadDim;
  (void)plane;
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    const std::uint32_t kvh = h / group;
    const float* qh = q + static_cast<std::size_t>(h) * kHeadDim;
    std::vector<double> scores(pos + 1);
    double m = -1e300;
    for (std::uint32_t p = 0; p <= pos; ++p) {
      const float* kp =
          kc +
          ((static_cast<std::size_t>(layer) * kKvHeads + kvh) * kMaxCtx + p) *
              kHeadDim;
      double dot = 0.0;
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        dot += static_cast<double>(qh[d]) * static_cast<double>(kp[d]);
      }
      scores[p] = dot * scale;
      m = std::max(m, scores[p]);
    }
    double l = 0.0;
    for (std::uint32_t p = 0; p <= pos; ++p) {
      scores[p] = std::exp(scores[p] - m);
      l += scores[p];
    }
    for (std::uint32_t d = 0; d < kHeadDim; ++d) {
      double a = 0.0;
      for (std::uint32_t p = 0; p <= pos; ++p) {
        const float* vp =
            vc +
            ((static_cast<std::size_t>(layer) * kKvHeads + kvh) * kMaxCtx + p) *
                kHeadDim;
        a += scores[p] * static_cast<double>(vp[d]);
      }
      out[static_cast<std::size_t>(h) * kHeadDim + d] =
          static_cast<float>(a / l);
    }
  }
}

// Max |diff| / (1 + |ref|) over out vs ref.
double MaxRatio(const float* out, const float* ref, std::size_t n) {
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double diff =
        std::fabs(static_cast<double>(out[i]) - static_cast<double>(ref[i]));
    const double r = diff / (1.0 + std::fabs(static_cast<double>(ref[i])));
    if (r > worst) {
      worst = r;
    }
  }
  return worst;
}

bool CheckOk(const gufo::hip::Lfm2LaunchStatus& st, const char* what,
             const std::string& cas) {
  if (st.ok) {
    return true;
  }
  std::cout << "FAIL " << cas << ": " << what << " "
            << (st.error != nullptr ? st.error : "?") << "\n";
  g_fail = 1;
  return false;
}

bool CheckOk(const gufo::hip::GlimmerLaunchStatus& st, const char* what,
             const std::string& cas) {
  if (st.ok) {
    return true;
  }
  std::cout << "FAIL " << cas << ": " << what << " "
            << (st.error != nullptr ? st.error : "?") << "\n";
  g_fail = 1;
  return false;
}

}  // namespace

int main() {
  const int gate = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "lfm2_attention_test");
  if (gate != 0) {
    return gate;
  }

  std::mt19937 rng(20261014);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

  const std::size_t plane =
      static_cast<std::size_t>(kLayers) * kKvHeads * kMaxCtx * kHeadDim;
  const std::size_t q_elems = static_cast<std::size_t>(kHeads) * kHeadDim;
  const std::size_t partial_elems =
      static_cast<std::size_t>(kHeads) * kMaxSplits * (kHeadDim + 2);

  std::vector<float> h_kc(plane), h_vc(plane);
  for (auto& x : h_kc) {
    x = dist(rng);
  }
  for (auto& x : h_vc) {
    x = dist(rng);
  }

  float *d_kc = nullptr, *d_vc = nullptr, *d_q = nullptr, *d_out = nullptr,
        *d_out_g = nullptr, *d_partial = nullptr;
  std::uint32_t* d_pos = nullptr;
  if (!HipOk(hipMalloc(&d_kc, plane * sizeof(float)), "malloc kc") ||
      !HipOk(hipMalloc(&d_vc, plane * sizeof(float)), "malloc vc") ||
      !HipOk(hipMalloc(&d_q, q_elems * sizeof(float)), "malloc q") ||
      !HipOk(hipMalloc(&d_out, q_elems * sizeof(float)), "malloc out") ||
      !HipOk(hipMalloc(&d_out_g, q_elems * sizeof(float)), "malloc outg") ||
      !HipOk(hipMalloc(&d_partial, partial_elems * sizeof(float)),
             "malloc partial") ||
      !HipOk(hipMalloc(&d_pos, sizeof(std::uint32_t)), "malloc dpos")) {
    return 1;
  }
  if (!HipOk(hipMemcpy(d_kc, h_kc.data(), plane * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy kc") ||
      !HipOk(hipMemcpy(d_vc, h_vc.data(), plane * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy vc")) {
    return 1;
  }

  const std::uint32_t kPos[] = {0, 1, 31, 32, 33, 1000, 4095, 16383};
  const std::uint32_t kSplits[] = {1, 8, 32, 64};
  std::vector<float> h_q(q_elems), h_out(q_elems), h_out_g(q_elems),
      h_ref(q_elems);
  constexpr double kTol = 2e-5;

  for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
    for (std::uint32_t pos : kPos) {
      for (float& x : h_q) {
        x = dist(rng);
      }
      if (!HipOk(hipMemcpy(d_q, h_q.data(), q_elems * sizeof(float),
                           hipMemcpyHostToDevice),
                 "copy q") ||
          !HipOk(hipMemcpy(d_pos, &pos, sizeof(pos), hipMemcpyHostToDevice),
                 "copy pos")) {
        return 1;
      }
      CpuRef(h_q.data(), h_kc.data(), h_vc.data(), h_ref.data(), layer, pos);
      for (std::uint32_t ns : kSplits) {
        char cas[128];
        std::snprintf(cas, sizeof(cas), "attn layer=%u pos=%u splits=%u", layer,
                      pos, ns);
        if (!CheckOk(gufo::hip::LaunchLfm2DecodeAttention(
                         d_q, d_kc, d_vc, d_out, layer, d_pos, kMaxCtx, kHeads,
                         kKvHeads, kHeadDim, ns, d_partial, nullptr),
                     "launch", cas)) {
          continue;
        }
        if (!CheckOk(gufo::hip::LaunchGlimmerWindowAttention(
                         d_q, nullptr, d_kc, d_vc, d_out_g, layer, pos, 0,
                         kMaxCtx, kHeads, kKvHeads, kHeadDim, q_elems, 0, plane,
                         q_elems, nullptr),
                     "glimmer", cas)) {
          continue;
        }
        if (!HipOk(hipDeviceSynchronize(), "sync")) {
          return 1;
        }
        if (!HipOk(hipMemcpy(h_out.data(), d_out, q_elems * sizeof(float),
                             hipMemcpyDeviceToHost),
                   "copy out") ||
            !HipOk(hipMemcpy(h_out_g.data(), d_out_g, q_elems * sizeof(float),
                             hipMemcpyDeviceToHost),
                   "copy outg")) {
          return 1;
        }
        const double r_cpu = MaxRatio(h_out.data(), h_ref.data(), q_elems);
        const double r_glm = MaxRatio(h_out_g.data(), h_ref.data(), q_elems);
        char nums[160];
        std::snprintf(nums, sizeof(nums), "cpu_ratio=%.3g glimmer_ratio=%.3g",
                      r_cpu, r_glm);
        Report(cas, r_cpu <= kTol && r_glm <= kTol, nums);
      }
    }
  }

  // WriteKvDev writes the same bytes as GlimmerWriteKv.
  float *d_k2 = nullptr, *d_v2 = nullptr, *d_k = nullptr, *d_v = nullptr;
  const std::size_t kv_elems = static_cast<std::size_t>(kKvHeads) * kHeadDim;
  if (!HipOk(hipMalloc(&d_k2, plane * sizeof(float)), "malloc k2") ||
      !HipOk(hipMalloc(&d_v2, plane * sizeof(float)), "malloc v2") ||
      !HipOk(hipMalloc(&d_k, kv_elems * sizeof(float)), "malloc k") ||
      !HipOk(hipMalloc(&d_v, kv_elems * sizeof(float)), "malloc v")) {
    return 1;
  }
  std::vector<float> h_k(kv_elems), h_v(kv_elems), h_c1(plane), h_c2(plane);
  const std::uint32_t kWpos[] = {0, 1000, 16383};
  for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
    for (std::uint32_t pos : kWpos) {
      char cas[128];
      std::snprintf(cas, sizeof(cas), "writekv layer=%u pos=%u", layer, pos);
      for (float& x : h_k) {
        x = dist(rng);
      }
      for (float& x : h_v) {
        x = dist(rng);
      }
      if (!HipOk(hipMemcpy(d_k, h_k.data(), kv_elems * sizeof(float),
                           hipMemcpyHostToDevice),
                 "copy k") ||
          !HipOk(hipMemcpy(d_v, h_v.data(), kv_elems * sizeof(float),
                           hipMemcpyHostToDevice),
                 "copy v") ||
          !HipOk(hipMemset(d_kc, 0, plane * sizeof(float)), "zero kc") ||
          !HipOk(hipMemset(d_k2, 0, plane * sizeof(float)), "zero k2") ||
          !HipOk(hipMemset(d_vc, 0, plane * sizeof(float)), "zero vc") ||
          !HipOk(hipMemset(d_v2, 0, plane * sizeof(float)), "zero v2") ||
          !HipOk(hipMemcpy(d_pos, &pos, sizeof(pos), hipMemcpyHostToDevice),
                 "copy pos")) {
        return 1;
      }
      if (!CheckOk(gufo::hip::LaunchLfm2WriteKvDev(d_k, d_v, d_kc, d_vc, layer,
                                                   d_pos, kMaxCtx, kKvHeads,
                                                   kHeadDim, nullptr),
                   "launch", cas)) {
        continue;
      }
      if (!CheckOk(gufo::hip::LaunchGlimmerWriteKv(
                       d_k, d_v, d_k2, d_v2, layer, pos, kMaxCtx, kKvHeads,
                       kHeadDim, kv_elems, plane, nullptr),
                   "glimmer", cas)) {
        continue;
      }
      if (!HipOk(hipDeviceSynchronize(), "sync") ||
          !HipOk(hipMemcpy(h_c1.data(), d_kc, plane * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "copy c1") ||
          !HipOk(hipMemcpy(h_c2.data(), d_k2, plane * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "copy c2")) {
        return 1;
      }
      const bool k_eq =
          std::memcmp(h_c1.data(), h_c2.data(), plane * sizeof(float)) == 0;
      if (!HipOk(hipMemcpy(h_c1.data(), d_vc, plane * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "copy v1") ||
          !HipOk(hipMemcpy(h_c2.data(), d_v2, plane * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "copy v2")) {
        return 1;
      }
      const bool v_eq =
          std::memcmp(h_c1.data(), h_c2.data(), plane * sizeof(float)) == 0;
      Report(cas, k_eq && v_eq,
             std::string("k_eq=") + (k_eq ? "1" : "0") +
                 " v_eq=" + (v_eq ? "1" : "0"));
    }
  }

  std::cout << (g_fail == 0 ? "ALL LFM2_ATTENTION OK" : "LFM2_ATTENTION FAILED")
            << "\n";
  return g_fail == 0 ? 0 : 1;
}
#endif
