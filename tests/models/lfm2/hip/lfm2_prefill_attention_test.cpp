// LFM2 prefill attention kernel test:
// LaunchLfm2PrefillAttention vs LaunchGlimmerPrefillAttention (window 0, no //
// LFM2 gate), plus a CPU double reference for the three smallest cases. Prints
// // LFM2 the kernel time of both launchers for (0,4096) (hipEvent).
// Exit 0 pass, 1 fail, 77 no HIP device.

#include "src/models/lfm2/hip/lfm2_prefill_attention.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/models/glimmer/hip/glimmer_decode.hpp"
#include "src/models/lfm2/hip/lfm2_kernels.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

#if !defined(ENGINE_ENABLE_HIP)
int main() {
  std::cerr << "FAIL lfm2_prefill_attention_test built without HIP\n";
  return 1;
}
#else
#include <hip/hip_runtime.h>

namespace {

int g_fail = 0;

constexpr std::uint32_t kHeads = 32U;
constexpr std::uint32_t kKvHeads = 8U;
constexpr std::uint32_t kGroup = 4U;
constexpr std::uint32_t kHeadDim = 64U;
constexpr std::uint32_t kMaxCtx = 8192U;
constexpr std::uint32_t kLayer = 0U;
constexpr double kTol = 2e-5;  // LFM2 (max |diff| <= tol*(1+|ref|))

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

std::string F(double v) {
  char b[32];
  std::snprintf(b, sizeof(b), "%.6g", v);
  return b;
}

// LFM2: CPU double reference straight from the SPEC math (scale 1/sqrt(64), //
// LFM2 causal over [0, start_pos+row], fp32 head-major cache).
void CpuAttn(const std::vector<float>& q, const std::vector<float>& kc,
             const std::vector<float>& vc, std::uint32_t start_pos,
             std::uint32_t batch, std::vector<float>& out) {
  constexpr double kScale = 1.0 / 8.0;
  const std::size_t head_elems = static_cast<std::size_t>(kHeads) * kHeadDim;
  const std::size_t kv_span = static_cast<std::size_t>(kMaxCtx) * kHeadDim;
  out.assign(static_cast<std::size_t>(batch) * head_elems, 0.0F);
  std::vector<double> acc(kHeadDim);
  for (std::uint32_t row = 0; row < batch; ++row) {
    const std::uint32_t pos = start_pos + row;
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const std::uint32_t kvh = h / kGroup;
      const std::size_t qb =
          (static_cast<std::size_t>(row) * kHeads + h) * kHeadDim;
      const std::size_t kb = static_cast<std::size_t>(kvh) * kv_span;
      double m = -1.0e30;
      double l = 0.0;
      for (std::uint32_t d = 0; d < kHeadDim; ++d)
        acc[d] = 0.0;
      for (std::uint32_t p = 0; p <= pos; ++p) {
        double s = 0.0;
        const std::size_t kp = kb + static_cast<std::size_t>(p) * kHeadDim;
        for (std::uint32_t d = 0; d < kHeadDim; ++d) {
          s += static_cast<double>(q[qb + d]) * static_cast<double>(kc[kp + d]);
        }
        s *= kScale;
        const double m_new = s > m ? s : m;
        const double rescale = std::exp(m - m_new);
        const double w = std::exp(s - m_new);
        for (std::uint32_t d = 0; d < kHeadDim; ++d) {
          acc[d] = acc[d] * rescale + w * static_cast<double>(vc[kp + d]);
        }
        l = l * rescale + w;
        m = m_new;
      }
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        out[qb + d] = static_cast<float>(acc[d] / l);
      }
    }
  }
}

void CheckVs(const std::string& cas, const std::vector<float>& got,
             const std::vector<float>& ref) {
  double worst = 0.0;
  long bad = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double rel = std::fabs(static_cast<double>(got[i]) - ref[i]) /
                       (1.0 + std::fabs(static_cast<double>(ref[i])));
    if (rel > worst)
      worst = rel;
    if (rel > kTol)
      ++bad;
  }
  Report(cas, bad == 0,
         "worst_rel=" + F(worst) + " bad=" + std::to_string(bad) + "/" +
             std::to_string(got.size()) + " tol=" + F(kTol));
}

// LFM2: one (start_pos, batch) case; fresh random q, both launchers, mine vs //
// LFM2 Glimmer always and mine vs CPU when asked.
void CaseOne(std::mt19937& rng, const std::vector<float>& kc_host,
             const std::vector<float>& vc_host, const float* d_kc,
             const float* d_vc, std::uint32_t start_pos, std::uint32_t batch,
             bool cpu_ref) {
  const std::string cas = "case_start" + std::to_string(start_pos) + "_batch" +
                          std::to_string(batch);
  const std::size_t q_elems =
      static_cast<std::size_t>(batch) * kHeads * kHeadDim;
  const std::size_t cache_elems =
      static_cast<std::size_t>(kKvHeads) * kMaxCtx * kHeadDim;
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> q(q_elems);
  for (float& v : q)
    v = dist(rng);
  float *d_q = nullptr, *d_out = nullptr, *d_ref = nullptr;
  if (!HipOk(hipMalloc(&d_q, q_elems * sizeof(float)), "malloc q") ||
      !HipOk(hipMalloc(&d_out, q_elems * sizeof(float)), "malloc out") ||
      !HipOk(hipMalloc(&d_ref, q_elems * sizeof(float)), "malloc ref")) {
    return;
  }
  bool ok = HipOk(
      hipMemcpy(d_q, q.data(), q_elems * sizeof(float), hipMemcpyHostToDevice),
      "copy q");
  hipStream_t stream = nullptr;
  if (ok) {
    const superroc::hip::Lfm2LaunchStatus mine =
        superroc::hip::LaunchLfm2PrefillAttention(
            d_q, d_kc, d_vc, d_out, kLayer, start_pos, batch, kMaxCtx, kHeads,
            kKvHeads, stream);
    ok = mine.ok;
    if (!ok) {
      std::cerr << "FAIL " << cas << " mine refused: " << mine.error << "\n";
      g_fail = 1;
    }
  }
  if (ok) {
    const superroc::hip::GlimmerLaunchStatus ref =
        superroc::hip::LaunchGlimmerPrefillAttention(
            d_q, nullptr, d_kc, d_vc, d_ref, kLayer, start_pos, batch, 0,
            kMaxCtx, kHeads, kKvHeads, kHeadDim, q_elems, 0, cache_elems,
            q_elems, stream);
    ok = ref.ok;
    if (!ok) {
      std::cerr << "FAIL " << cas << " glimmer refused: " << ref.error << "\n";
      g_fail = 1;
    }
  }
  std::vector<float> mine_host(q_elems), ref_host(q_elems);
  if (ok) {
    ok = HipOk(hipDeviceSynchronize(), "sync attn") &&
         HipOk(hipMemcpy(mine_host.data(), d_out, q_elems * sizeof(float),
                         hipMemcpyDeviceToHost),
               "copy out") &&
         HipOk(hipMemcpy(ref_host.data(), d_ref, q_elems * sizeof(float),
                         hipMemcpyDeviceToHost),
               "copy ref");
  }
  (void)hipFree(d_q);
  (void)hipFree(d_out);
  (void)hipFree(d_ref);
  if (!ok)
    return;
  CheckVs(cas + "_vs_glimmer", mine_host, ref_host);
  if (cpu_ref) {
    std::vector<float> cpu;
    CpuAttn(q, kc_host, vc_host, start_pos, batch, cpu);
    CheckVs(cas + "_vs_cpu", mine_host, cpu);
  }
}

// LFM2: kernel time of both launchers for (0,4096) via hipEvent (mean of 5). //
void TimeBoth(std::mt19937& rng, const float* d_kc, const float* d_vc) {
  constexpr std::uint32_t kBatch = 4096U;
  const std::size_t q_elems =
      static_cast<std::size_t>(kBatch) * kHeads * kHeadDim;
  const std::size_t cache_elems =
      static_cast<std::size_t>(kKvHeads) * kMaxCtx * kHeadDim;
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> q(q_elems);
  for (float& v : q)
    v = dist(rng);
  float *d_q = nullptr, *d_out = nullptr;
  if (!HipOk(hipMalloc(&d_q, q_elems * sizeof(float)), "malloc tq") ||
      !HipOk(hipMalloc(&d_out, q_elems * sizeof(float)), "malloc tout")) {
    return;
  }
  bool ok = HipOk(
      hipMemcpy(d_q, q.data(), q_elems * sizeof(float), hipMemcpyHostToDevice),
      "copy tq");
  hipEvent_t start = nullptr, stop = nullptr;
  if (ok)
    ok = HipOk(hipEventCreate(&start), "event start") &&
         HipOk(hipEventCreate(&stop), "event stop");
  hipStream_t stream = nullptr;
  constexpr int kIters = 5;
  float mine_ms = 0.0F, ref_ms = 0.0F;
  if (ok) {
    ok = superroc::hip::LaunchLfm2PrefillAttention(
             d_q, d_kc, d_vc, d_out, kLayer, 0, kBatch, kMaxCtx, kHeads,
             kKvHeads, stream)  // LFM2 (warmup)
             .ok &&
         superroc::hip::LaunchGlimmerPrefillAttention(
             d_q, nullptr, d_kc, d_vc, d_out, kLayer, 0, kBatch, 0, kMaxCtx,
             kHeads, kKvHeads, kHeadDim, q_elems, 0, cache_elems, q_elems,
             stream)  // LFM2 (warmup)
             .ok;
    ok = HipOk(hipDeviceSynchronize(), "sync warmup");
  }
  for (int i = 0; i < kIters && ok; ++i) {
    float ms = 0.0F;
    ok = HipOk(hipEventRecord(start, stream), "record start") &&
         (superroc::hip::LaunchLfm2PrefillAttention(d_q, d_kc, d_vc, d_out,
                                                    kLayer, 0, kBatch, kMaxCtx,
                                                    kHeads, kKvHeads, stream)
              .ok) &&
         HipOk(hipEventRecord(stop, stream), "record stop") &&
         HipOk(hipEventSynchronize(stop), "sync stop") &&
         HipOk(hipEventElapsedTime(&ms, start, stop), "elapsed");
    mine_ms += ms;
  }
  for (int i = 0; i < kIters && ok; ++i) {
    float ms = 0.0F;
    ok = HipOk(hipEventRecord(start, stream), "record start") &&
         (superroc::hip::LaunchGlimmerPrefillAttention(
              d_q, nullptr, d_kc, d_vc, d_out, kLayer, 0, kBatch, 0, kMaxCtx,
              kHeads, kKvHeads, kHeadDim, q_elems, 0, cache_elems, q_elems,
              stream)
              .ok) &&
         HipOk(hipEventRecord(stop, stream), "record stop") &&
         HipOk(hipEventSynchronize(stop), "sync stop") &&
         HipOk(hipEventElapsedTime(&ms, start, stop), "elapsed");
    ref_ms += ms;
  }
  if (start != nullptr)
    (void)hipEventDestroy(start);
  if (stop != nullptr)
    (void)hipEventDestroy(stop);
  (void)hipFree(d_q);
  (void)hipFree(d_out);
  if (!ok)
    return;
  std::cout << "timing_start0_batch4096_iters" << kIters
            << ": lfm2_ms=" << F(mine_ms / kIters)
            << " glimmer_ms=" << F(ref_ms / kIters) << "\n";
}

}  // namespace

int main() {
  const int gate = superroc::test::GateHipDevice(
      superroc::test::HipDeviceRequirement::kOptional,
      "lfm2_prefill_attention_test");
  if (gate != 0) {
    return gate;
  }
  std::mt19937 rng(20261014);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  const std::size_t cache_elems =
      static_cast<std::size_t>(kKvHeads) * kMaxCtx * kHeadDim;
  std::vector<float> kc_host(cache_elems), vc_host(cache_elems);
  for (float& v : kc_host)
    v = dist(rng);
  for (float& v : vc_host)
    v = dist(rng);
  float *d_kc = nullptr, *d_vc = nullptr;
  if (!HipOk(hipMalloc(&d_kc, cache_elems * sizeof(float)), "malloc kc") ||
      !HipOk(hipMalloc(&d_vc, cache_elems * sizeof(float)), "malloc vc")) {
    return 1;
  }
  if (!HipOk(hipMemcpy(d_kc, kc_host.data(), cache_elems * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy kc") ||
      !HipOk(hipMemcpy(d_vc, vc_host.data(), cache_elems * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy vc")) {
    return 1;
  }
  CaseOne(rng, kc_host, vc_host, d_kc, d_vc, 0, 1, true);
  CaseOne(rng, kc_host, vc_host, d_kc, d_vc, 0, 17, true);
  CaseOne(rng, kc_host, vc_host, d_kc, d_vc, 0, 64, true);
  CaseOne(rng, kc_host, vc_host, d_kc, d_vc, 0, 65, false);
  CaseOne(rng, kc_host, vc_host, d_kc, d_vc, 0, 512, false);
  CaseOne(rng, kc_host, vc_host, d_kc, d_vc, 100, 300, false);
  CaseOne(rng, kc_host, vc_host, d_kc, d_vc, 0, 4096, false);
  CaseOne(rng, kc_host, vc_host, d_kc, d_vc, 4000, 96, false);
  TimeBoth(rng, d_kc, d_vc);
  (void)hipFree(d_kc);
  (void)hipFree(d_vc);
  if (g_fail == 0) {
    std::cout << "lfm2_prefill_attention_test ALL PASS\n";
  }
  return g_fail;
}
#endif
