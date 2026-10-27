// LFM2 prefill attention kernel test (SPEC2, lane LFS): // LFM2
// LaunchLfm2PrefillAttention vs LaunchGlimmerPrefillAttention (window 0, no //
// LFM2 gate), plus a CPU double reference for the three smallest cases. Prints
// // LFM2 the kernel time of both launchers for (0,4096) (hipEvent). // LFM2
// Exit 0 pass, 1 fail, 77 no HIP device. // LFM2

#include "src/models/lfm2/hip/lfm2_prefill_attention.hpp"  // LFM2

#include <cmath>     // LFM2
#include <cstdint>   // LFM2
#include <cstdio>    // LFM2
#include <iostream>  // LFM2
#include <random>    // LFM2
#include <string>    // LFM2
#include <vector>    // LFM2

#include "src/models/glimmer/hip/glimmer_decode.hpp"  // LFM2
#include "src/models/lfm2/hip/lfm2_kernels.hpp"       // LFM2
#include "tests/models/qwen/hip/support/device.hpp"   // LFM2

#if !defined(ENGINE_ENABLE_HIP)                                         // LFM2
int main() {                                                            // LFM2
  std::cerr << "FAIL lfm2_prefill_attention_test built without HIP\n";  // LFM2
  return 1;                                                             // LFM2
}  // LFM2
#else                         // LFM2
#include <hip/hip_runtime.h>  // LFM2

namespace {  // LFM2

int g_fail = 0;  // LFM2

constexpr std::uint32_t kHeads = 32U;     // LFM2
constexpr std::uint32_t kKvHeads = 8U;    // LFM2
constexpr std::uint32_t kGroup = 4U;      // LFM2
constexpr std::uint32_t kHeadDim = 64U;   // LFM2
constexpr std::uint32_t kMaxCtx = 8192U;  // LFM2
constexpr std::uint32_t kLayer = 0U;      // LFM2
constexpr double kTol = 2e-5;             // LFM2 (max |diff| <= tol*(1+|ref|))

void Report(const std::string& cas, bool pass,
            const std::string& numbers) {  // LFM2
  std::cout << (pass ? "PASS " : "FAIL ") << cas << ": " << numbers
            << "\n";  // LFM2
  if (!pass) {        // LFM2
    g_fail = 1;       // LFM2
  }  // LFM2
}  // LFM2

bool HipOk(hipError_t err, const char* what) {  // LFM2
  if (err == hipSuccess) {                      // LFM2
    return true;                                // LFM2
  }  // LFM2
  std::cerr << "FAIL hip " << what << " code=" << static_cast<int>(err)
            << "\n";  // LFM2
  g_fail = 1;         // LFM2
  return false;       // LFM2
}  // LFM2

std::string F(double v) {                  // LFM2
  char b[32];                              // LFM2
  std::snprintf(b, sizeof(b), "%.6g", v);  // LFM2
  return b;                                // LFM2
}  // LFM2

// LFM2: CPU double reference straight from the SPEC math (scale 1/sqrt(64), //
// LFM2 causal over [0, start_pos+row], fp32 head-major cache). // LFM2
void CpuAttn(const std::vector<float>& q, const std::vector<float>& kc,  // LFM2
             const std::vector<float>& vc, std::uint32_t start_pos,      // LFM2
             std::uint32_t batch, std::vector<float>& out) {             // LFM2
  constexpr double kScale = 1.0 / 8.0;                                   // LFM2
  const std::size_t head_elems =
      static_cast<std::size_t>(kHeads) * kHeadDim;  // LFM2
  const std::size_t kv_span =
      static_cast<std::size_t>(kMaxCtx) * kHeadDim;                    // LFM2
  out.assign(static_cast<std::size_t>(batch) * head_elems, 0.0F);      // LFM2
  std::vector<double> acc(kHeadDim);                                   // LFM2
  for (std::uint32_t row = 0; row < batch; ++row) {                    // LFM2
    const std::uint32_t pos = start_pos + row;                         // LFM2
    for (std::uint32_t h = 0; h < kHeads; ++h) {                       // LFM2
      const std::uint32_t kvh = h / kGroup;                            // LFM2
      const std::size_t qb =                                           // LFM2
          (static_cast<std::size_t>(row) * kHeads + h) * kHeadDim;     // LFM2
      const std::size_t kb = static_cast<std::size_t>(kvh) * kv_span;  // LFM2
      double m = -1.0e30;                                              // LFM2
      double l = 0.0;                                                  // LFM2
      for (std::uint32_t d = 0; d < kHeadDim; ++d)
        acc[d] = 0.0;                             // LFM2
      for (std::uint32_t p = 0; p <= pos; ++p) {  // LFM2
        double s = 0.0;                           // LFM2
        const std::size_t kp =
            kb + static_cast<std::size_t>(p) * kHeadDim;  // LFM2
        for (std::uint32_t d = 0; d < kHeadDim; ++d) {    // LFM2
          s += static_cast<double>(q[qb + d]) *
               static_cast<double>(kc[kp + d]);  // LFM2
        }  // LFM2
        s *= kScale;                                    // LFM2
        const double m_new = s > m ? s : m;             // LFM2
        const double rescale = std::exp(m - m_new);     // LFM2
        const double w = std::exp(s - m_new);           // LFM2
        for (std::uint32_t d = 0; d < kHeadDim; ++d) {  // LFM2
          acc[d] =
              acc[d] * rescale + w * static_cast<double>(vc[kp + d]);  // LFM2
        }  // LFM2
        l = l * rescale + w;  // LFM2
        m = m_new;            // LFM2
      }  // LFM2
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {   // LFM2
        out[qb + d] = static_cast<float>(acc[d] / l);  // LFM2
      }  // LFM2
    }  // LFM2
  }  // LFM2
}  // LFM2

void CheckVs(const std::string& cas, const std::vector<float>& got,  // LFM2
             const std::vector<float>& ref) {                        // LFM2
  double worst = 0.0;                                                // LFM2
  long bad = 0;                                                      // LFM2
  for (std::size_t i = 0; i < got.size(); ++i) {                     // LFM2
    const double rel =                                               // LFM2
        std::fabs(static_cast<double>(got[i]) - ref[i]) /            // LFM2
        (1.0 + std::fabs(static_cast<double>(ref[i])));              // LFM2
    if (rel > worst)
      worst = rel;  // LFM2
    if (rel > kTol)
      ++bad;  // LFM2
  }  // LFM2
  Report(cas, bad == 0,  // LFM2
         "worst_rel=" + F(worst) + " bad=" + std::to_string(bad) +
             "/" +                                             // LFM2
             std::to_string(got.size()) + " tol=" + F(kTol));  // LFM2
}  // LFM2

// LFM2: one (start_pos, batch) case; fresh random q, both launchers, mine vs //
// LFM2 Glimmer always and mine vs CPU when asked. // LFM2
void CaseOne(std::mt19937& rng, const std::vector<float>& kc_host,  // LFM2
             const std::vector<float>& vc_host, const float* d_kc,  // LFM2
             const float* d_vc, std::uint32_t start_pos,
             std::uint32_t batch,                                     // LFM2
             bool cpu_ref) {                                          // LFM2
  const std::string cas = "case_start" + std::to_string(start_pos) +  // LFM2
                          "_batch" + std::to_string(batch);           // LFM2
  const std::size_t q_elems =                                         // LFM2
      static_cast<std::size_t>(batch) * kHeads * kHeadDim;            // LFM2
  const std::size_t cache_elems =                                     // LFM2
      static_cast<std::size_t>(kKvHeads) * kMaxCtx * kHeadDim;        // LFM2
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);            // LFM2
  std::vector<float> q(q_elems);                                      // LFM2
  for (float& v : q)
    v = dist(rng);                                                     // LFM2
  float *d_q = nullptr, *d_out = nullptr, *d_ref = nullptr;            // LFM2
  if (!HipOk(hipMalloc(&d_q, q_elems * sizeof(float)), "malloc q") ||  // LFM2
      !HipOk(hipMalloc(&d_out, q_elems * sizeof(float)),
             "malloc out") ||  // LFM2
      !HipOk(hipMalloc(&d_ref, q_elems * sizeof(float)),
             "malloc ref")) {  // LFM2
    return;                    // LFM2
  }  // LFM2
  bool ok = HipOk(hipMemcpy(d_q, q.data(), q_elems * sizeof(float),  // LFM2
                            hipMemcpyHostToDevice),                  // LFM2
                  "copy q");                                         // LFM2
  hipStream_t stream = nullptr;                                      // LFM2
  if (ok) {                                                          // LFM2
    const superroc::hip::Lfm2LaunchStatus mine =                         // LFM2
        superroc::hip::LaunchLfm2PrefillAttention(                       // LFM2
            d_q, d_kc, d_vc, d_out, kLayer, start_pos, batch, kMaxCtx,
            kHeads,             // LFM2
            kKvHeads, stream);  // LFM2
    ok = mine.ok;               // LFM2
    if (!ok) {                  // LFM2
      std::cerr << "FAIL " << cas << " mine refused: " << mine.error
                << "\n";  // LFM2
      g_fail = 1;         // LFM2
    }  // LFM2
  }  // LFM2
  if (ok) {                                        // LFM2
    const superroc::hip::GlimmerLaunchStatus ref =     // LFM2
        superroc::hip::LaunchGlimmerPrefillAttention(  // LFM2
            d_q, nullptr, d_kc, d_vc, d_ref, kLayer, start_pos, batch,
            0,  // LFM2
            kMaxCtx, kHeads, kKvHeads, kHeadDim, q_elems, 0,
            cache_elems,       // LFM2
            q_elems, stream);  // LFM2
    ok = ref.ok;               // LFM2
    if (!ok) {                 // LFM2
      std::cerr << "FAIL " << cas << " glimmer refused: " << ref.error
                << "\n";  // LFM2
      g_fail = 1;         // LFM2
    }  // LFM2
  }  // LFM2
  std::vector<float> mine_host(q_elems), ref_host(q_elems);  // LFM2
  if (ok) {                                                  // LFM2
    ok = HipOk(hipDeviceSynchronize(), "sync attn") &&       // LFM2
         HipOk(hipMemcpy(mine_host.data(), d_out,
                         q_elems * sizeof(float),  // LFM2
                         hipMemcpyDeviceToHost),   // LFM2
               "copy out") &&                      // LFM2
         HipOk(
             hipMemcpy(ref_host.data(), d_ref, q_elems * sizeof(float),  // LFM2
                       hipMemcpyDeviceToHost),                           // LFM2
             "copy ref");                                                // LFM2
  }  // LFM2
  (void)hipFree(d_q);    // LFM2
  (void)hipFree(d_out);  // LFM2
  (void)hipFree(d_ref);  // LFM2
  if (!ok)
    return;                                               // LFM2
  CheckVs(cas + "_vs_glimmer", mine_host, ref_host);      // LFM2
  if (cpu_ref) {                                          // LFM2
    std::vector<float> cpu;                               // LFM2
    CpuAttn(q, kc_host, vc_host, start_pos, batch, cpu);  // LFM2
    CheckVs(cas + "_vs_cpu", mine_host, cpu);             // LFM2
  }  // LFM2
}  // LFM2

// LFM2: kernel time of both launchers for (0,4096) via hipEvent (mean of 5). //
// LFM2
void TimeBoth(std::mt19937& rng, const float* d_kc,
              const float* d_vc) {                              // LFM2
  constexpr std::uint32_t kBatch = 4096U;                       // LFM2
  const std::size_t q_elems =                                   // LFM2
      static_cast<std::size_t>(kBatch) * kHeads * kHeadDim;     // LFM2
  const std::size_t cache_elems =                               // LFM2
      static_cast<std::size_t>(kKvHeads) * kMaxCtx * kHeadDim;  // LFM2
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);      // LFM2
  std::vector<float> q(q_elems);                                // LFM2
  for (float& v : q)
    v = dist(rng);                                                      // LFM2
  float *d_q = nullptr, *d_out = nullptr;                               // LFM2
  if (!HipOk(hipMalloc(&d_q, q_elems * sizeof(float)), "malloc tq") ||  // LFM2
      !HipOk(hipMalloc(&d_out, q_elems * sizeof(float)),
             "malloc tout")) {  // LFM2
    return;                     // LFM2
  }  // LFM2
  bool ok = HipOk(hipMemcpy(d_q, q.data(), q_elems * sizeof(float),  // LFM2
                            hipMemcpyHostToDevice),                  // LFM2
                  "copy tq");                                        // LFM2
  hipEvent_t start = nullptr, stop = nullptr;                        // LFM2
  if (ok)
    ok = HipOk(hipEventCreate(&start), "event start") &&          // LFM2
         HipOk(hipEventCreate(&stop), "event stop");              // LFM2
  hipStream_t stream = nullptr;                                   // LFM2
  constexpr int kIters = 5;                                       // LFM2
  float mine_ms = 0.0F, ref_ms = 0.0F;                            // LFM2
  if (ok) {                                                       // LFM2
    ok = superroc::hip::LaunchLfm2PrefillAttention(                   // LFM2
             d_q, d_kc, d_vc, d_out, kLayer, 0, kBatch, kMaxCtx,  // LFM2
             kHeads, kKvHeads, stream)              // LFM2 (warmup)
             .ok &&                                 // LFM2
         superroc::hip::LaunchGlimmerPrefillAttention(  // LFM2
             d_q, nullptr, d_kc, d_vc, d_out, kLayer, 0, kBatch, 0,  // LFM2
             kMaxCtx, kHeads, kKvHeads, kHeadDim, q_elems, 0,        // LFM2
             cache_elems, q_elems, stream)              // LFM2 (warmup)
             .ok;                                       // LFM2
    ok = HipOk(hipDeviceSynchronize(), "sync warmup");  // LFM2
  }  // LFM2
  for (int i = 0; i < kIters && ok; ++i) {                        // LFM2
    float ms = 0.0F;                                              // LFM2
    ok = HipOk(hipEventRecord(start, stream), "record start") &&  // LFM2
         (superroc::hip::LaunchLfm2PrefillAttention(
              d_q, d_kc, d_vc, d_out, kLayer,                      // LFM2
              0, kBatch, kMaxCtx, kHeads,                          // LFM2
              kKvHeads, stream)                                    // LFM2
              .ok) &&                                              // LFM2
         HipOk(hipEventRecord(stop, stream), "record stop") &&     // LFM2
         HipOk(hipEventSynchronize(stop), "sync stop") &&          // LFM2
         HipOk(hipEventElapsedTime(&ms, start, stop), "elapsed");  // LFM2
    mine_ms += ms;                                                 // LFM2
  }  // LFM2
  for (int i = 0; i < kIters && ok; ++i) {                            // LFM2
    float ms = 0.0F;                                                  // LFM2
    ok = HipOk(hipEventRecord(start, stream), "record start") &&      // LFM2
         (superroc::hip::LaunchGlimmerPrefillAttention(                   // LFM2
              d_q, nullptr, d_kc, d_vc, d_out, kLayer, 0, kBatch, 0,  // LFM2
              kMaxCtx, kHeads, kKvHeads, kHeadDim, q_elems, 0,        // LFM2
              cache_elems, q_elems, stream)                           // LFM2
              .ok) &&                                                 // LFM2
         HipOk(hipEventRecord(stop, stream), "record stop") &&        // LFM2
         HipOk(hipEventSynchronize(stop), "sync stop") &&             // LFM2
         HipOk(hipEventElapsedTime(&ms, start, stop), "elapsed");     // LFM2
    ref_ms += ms;                                                     // LFM2
  }  // LFM2
  if (start != nullptr)
    (void)hipEventDestroy(start);  // LFM2
  if (stop != nullptr)
    (void)hipEventDestroy(stop);  // LFM2
  (void)hipFree(d_q);             // LFM2
  (void)hipFree(d_out);           // LFM2
  if (!ok)
    return;                                                   // LFM2
  std::cout << "timing_start0_batch4096_iters" << kIters      // LFM2
            << ": lfm2_ms=" << F(mine_ms / kIters)            // LFM2
            << " glimmer_ms=" << F(ref_ms / kIters) << "\n";  // LFM2
}  // LFM2

}  // namespace

int main() {                                        // LFM2
  const int gate = superroc::test::GateHipDevice(       // LFM2
      superroc::test::HipDeviceRequirement::kOptional,  // LFM2
      "lfm2_prefill_attention_test");               // LFM2
  if (gate != 0) {                                  // LFM2
    return gate;                                    // LFM2
  }  // LFM2
  std::mt19937 rng(20261014);                                     // LFM2
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);        // LFM2
  const std::size_t cache_elems =                                 // LFM2
      static_cast<std::size_t>(kKvHeads) * kMaxCtx * kHeadDim;    // LFM2
  std::vector<float> kc_host(cache_elems), vc_host(cache_elems);  // LFM2
  for (float& v : kc_host)
    v = dist(rng);  // LFM2
  for (float& v : vc_host)
    v = dist(rng);                         // LFM2
  float *d_kc = nullptr, *d_vc = nullptr;  // LFM2
  if (!HipOk(hipMalloc(&d_kc, cache_elems * sizeof(float)),
             "malloc kc") ||  // LFM2
      !HipOk(hipMalloc(&d_vc, cache_elems * sizeof(float)),
             "malloc vc")) {  // LFM2
    return 1;                 // LFM2
  }  // LFM2
  if (!HipOk(
          hipMemcpy(d_kc, kc_host.data(), cache_elems * sizeof(float),  // LFM2
                    hipMemcpyHostToDevice),                             // LFM2
          "copy kc") ||                                                 // LFM2
      !HipOk(
          hipMemcpy(d_vc, vc_host.data(), cache_elems * sizeof(float),  // LFM2
                    hipMemcpyHostToDevice),                             // LFM2
          "copy vc")) {                                                 // LFM2
    return 1;                                                           // LFM2
  }  // LFM2
  CaseOne(rng, kc_host, vc_host, d_kc, d_vc, 0, 1, true);       // LFM2
  CaseOne(rng, kc_host, vc_host, d_kc, d_vc, 0, 17, true);      // LFM2
  CaseOne(rng, kc_host, vc_host, d_kc, d_vc, 0, 64, true);      // LFM2
  CaseOne(rng, kc_host, vc_host, d_kc, d_vc, 0, 65, false);     // LFM2
  CaseOne(rng, kc_host, vc_host, d_kc, d_vc, 0, 512, false);    // LFM2
  CaseOne(rng, kc_host, vc_host, d_kc, d_vc, 100, 300, false);  // LFM2
  CaseOne(rng, kc_host, vc_host, d_kc, d_vc, 0, 4096, false);   // LFM2
  CaseOne(rng, kc_host, vc_host, d_kc, d_vc, 4000, 96, false);  // LFM2
  TimeBoth(rng, d_kc, d_vc);                                    // LFM2
  (void)hipFree(d_kc);                                          // LFM2
  (void)hipFree(d_vc);                                          // LFM2
  if (g_fail == 0) {                                            // LFM2
    std::cout << "lfm2_prefill_attention_test ALL PASS\n";      // LFM2
  }  // LFM2
  return g_fail;  // LFM2
}  // LFM2
#endif                        // LFM2
