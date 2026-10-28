// LFM2 kernels GPU test: short-conv decode/prefill + MoE router. No GGUF.
// Exit 0 pass, 1 fail, 77 no HIP device.

#include "src/models/lfm2/hip/lfm2_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "src/models/qwen/hip/ops/norm_residual.hpp"  // LKF reference
#include "tests/models/qwen/hip/support/device.hpp"

#if !defined(ENGINE_ENABLE_HIP)
int main() {
  std::cerr << "FAIL lfm2_kernels_test built without HIP\n";
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

// CPU reference straight from the SPEC math (double accumulation).
void CpuShortConvStep(const float* bcx_row, const float* conv_w, float* state,
                      float* y, std::uint32_t hidden, std::uint32_t l_cache) {
  const std::uint32_t nm1 = l_cache - 1U;
  for (std::uint32_t c = 0; c < hidden; ++c) {
    const double b = bcx_row[c];
    const double cc = bcx_row[hidden + c];
    const double xx = bcx_row[2U * hidden + c];
    const double bx = b * xx;
    double acc = static_cast<double>(conv_w[c * l_cache + l_cache - 1U]) * bx;
    for (std::uint32_t j = 0; j < nm1; ++j) {
      acc += static_cast<double>(conv_w[c * l_cache + j]) *
             static_cast<double>(state[j * hidden + c]);
    }
    y[c] = static_cast<float>(cc * acc);
    for (std::uint32_t j = 0; j + 1U < nm1; ++j) {
      state[j * hidden + c] = state[(j + 1U) * hidden + c];
    }
    state[(nm1 - 1U) * hidden + c] = static_cast<float>(bx);
  }
}

void CpuMoeRoute(const float* x, const float* gate_inp, const float* bias,
                 std::uint32_t k, std::uint32_t n_expert, std::uint32_t n_used,
                 float* logits, int* ids, float* weights) {
  constexpr double kFloor = 6.103515625e-5;
  std::vector<double> p(n_expert), sel(n_expert);
  for (std::uint32_t e = 0; e < n_expert; ++e) {
    double dot = 0.0;
    for (std::uint32_t i = 0; i < k; ++i) {
      dot +=
          static_cast<double>(gate_inp[e * k + i]) * static_cast<double>(x[i]);
    }
    logits[e] = static_cast<float>(dot);
    const double pe = 1.0 / (1.0 + std::exp(-dot));
    p[e] = pe;
    sel[e] = pe + static_cast<double>(bias[e]);
  }
  std::vector<int> taken(n_expert, 0);
  double sum = 0.0;
  for (std::uint32_t i = 0; i < n_used; ++i) {
    double best = -std::numeric_limits<double>::infinity();
    int be = -1;
    for (std::uint32_t e = 0; e < n_expert; ++e) {
      if (taken[e] == 0 && sel[e] > best) {
        best = sel[e];
        be = static_cast<int>(e);
      }
    }
    taken[be] = 1;
    ids[i] = be;
    sum += p[static_cast<std::size_t>(be)];
  }
  const double denom = sum > kFloor ? sum : kFloor;
  for (std::uint32_t i = 0; i < n_used; ++i) {
    weights[i] =
        static_cast<float>(p[static_cast<std::size_t>(ids[i])] / denom);
  }
}

std::string F(float v) {
  char b[32];
  std::snprintf(b, sizeof(b), "%.6g", static_cast<double>(v));
  return b;
}

}  // namespace
namespace {

// Case a: decode H=2048 L=3, 6 consecutive tokens through one zero-start state.
void CaseDecode(std::mt19937& rng) {
  constexpr std::uint32_t kH = 2048;
  constexpr std::uint32_t kL = 3;
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> conv(kH * kL);
  for (float& v : conv) {
    v = dist(rng);
  }
  float *d_bcx = nullptr, *d_conv = nullptr, *d_state = nullptr, *d_y = nullptr;
  if (!HipOk(hipMalloc(&d_bcx, 3U * kH * sizeof(float)), "malloc bcx") ||
      !HipOk(hipMalloc(&d_conv, kH * kL * sizeof(float)), "malloc conv") ||
      !HipOk(hipMalloc(&d_state, (kL - 1U) * kH * sizeof(float)),
             "malloc state") ||
      !HipOk(hipMalloc(&d_y, kH * sizeof(float)), "malloc y")) {
    return;
  }
  std::vector<float> zero((kL - 1U) * kH, 0.0F);
  std::vector<float> cpu_state = zero;
  bool alloc_ok =
      HipOk(hipMemcpy(d_conv, conv.data(), kH * kL * sizeof(float),
                      hipMemcpyHostToDevice),
            "copy conv") &&
      HipOk(hipMemcpy(d_state, zero.data(), (kL - 1U) * kH * sizeof(float),
                      hipMemcpyHostToDevice),
            "copy state");
  float worst_y = 0.0F;
  float worst_s = 0.0F;
  for (int step = 0; step < 6 && alloc_ok; ++step) {
    std::vector<float> bcx(3U * kH);
    for (float& v : bcx) {
      v = dist(rng);
    }
    std::vector<float> cpu_y(kH);
    CpuShortConvStep(bcx.data(), conv.data(), cpu_state.data(), cpu_y.data(),
                     kH, kL);
    alloc_ok = HipOk(hipMemcpy(d_bcx, bcx.data(), 3U * kH * sizeof(float),
                               hipMemcpyHostToDevice),
                     "copy bcx");
    const auto st = superroc::hip::LaunchLfm2ShortConvDecode(
        d_bcx, d_conv, d_state, d_y, kH, kL, nullptr);
    if (!st.ok) {
      Report("decode", false, st.error != nullptr ? st.error : "launch");
      break;
    }
    std::vector<float> gpu_y(kH), gpu_state((kL - 1U) * kH);
    alloc_ok =
        HipOk(hipDeviceSynchronize(), "sync decode") &&
        HipOk(hipMemcpy(gpu_y.data(), d_y, kH * sizeof(float),
                        hipMemcpyDeviceToHost),
              "back y") &&
        HipOk(hipMemcpy(gpu_state.data(), d_state,
                        (kL - 1U) * kH * sizeof(float), hipMemcpyDeviceToHost),
              "back state");
    for (std::uint32_t c = 0; c < kH && alloc_ok; ++c) {
      const float gap =
          std::fabs(gpu_y[c] - cpu_y[c]) / (1.0F + std::fabs(cpu_y[c]));
      worst_y = std::max(worst_y, gap);
      const float sgap = std::fabs(gpu_state[c] - cpu_state[c]);
      worst_s = std::max(worst_s, sgap);
      const float sgap2 = std::fabs(gpu_state[kH + c] - cpu_state[kH + c]);
      worst_s = std::max(worst_s, sgap2);
    }
  }
  hipFree(d_bcx);
  hipFree(d_conv);
  hipFree(d_state);
  hipFree(d_y);
  if (!alloc_ok) {
    Report("decode", false, "hip error");
    return;
  }
  Report("decode", worst_y <= 1e-5F && worst_s <= 1e-6F,
         "worst_y_scaled=" + F(worst_y) + " worst_state_abs=" + F(worst_s));
}

// Case b: prefill H=2048 L=3 from random non-zero state, plus prefill37 ==
// decode-one-by-one equivalence.
void CasePrefillOne(std::mt19937& rng,
                    std::uniform_real_distribution<float>& dist,
                    const std::vector<float>& conv, std::uint32_t n_tokens) {
  constexpr std::uint32_t kH = 2048;
  constexpr std::uint32_t kL = 3;
  std::vector<float> init_state((kL - 1U) * kH);
  for (float& v : init_state) {
    v = dist(rng);
    if (v == 0.0F) {
      v = 0.5F;
    }
  }
  std::vector<float> bcx(n_tokens * 3U * kH);
  for (float& v : bcx) {
    v = dist(rng);
  }
  std::vector<float> cpu_state = init_state;
  std::vector<float> cpu_y(n_tokens * kH);
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    CpuShortConvStep(bcx.data() + t * 3U * kH, conv.data(), cpu_state.data(),
                     cpu_y.data() + t * kH, kH, kL);
  }
  float *d_bcx = nullptr, *d_conv = nullptr, *d_state = nullptr, *d_y = nullptr;
  bool ok =
      HipOk(hipMalloc(&d_bcx, bcx.size() * sizeof(float)), "malloc bcx") &&
      HipOk(hipMalloc(&d_conv, conv.size() * sizeof(float)), "malloc conv") &&
      HipOk(hipMalloc(&d_state, init_state.size() * sizeof(float)),
            "malloc state") &&
      HipOk(hipMalloc(&d_y, cpu_y.size() * sizeof(float)), "malloc y") &&
      HipOk(hipMemcpy(d_bcx, bcx.data(), bcx.size() * sizeof(float),
                      hipMemcpyHostToDevice),
            "copy bcx") &&
      HipOk(hipMemcpy(d_conv, conv.data(), conv.size() * sizeof(float),
                      hipMemcpyHostToDevice),
            "copy conv") &&
      HipOk(hipMemcpy(d_state, init_state.data(),
                      init_state.size() * sizeof(float), hipMemcpyHostToDevice),
            "copy state");
  const auto st =
      ok ? superroc::hip::LaunchLfm2ShortConvPrefill(
               d_bcx, d_conv, d_state, d_y, kH, kL, n_tokens, nullptr)
         : superroc::hip::Lfm2LaunchStatus{false, "alloc"};
  if (!st.ok) {
    hipFree(d_bcx);
    hipFree(d_conv);
    hipFree(d_state);
    hipFree(d_y);
    Report("prefill_n" + std::to_string(n_tokens), false,
           st.error != nullptr ? st.error : "launch");
    return;
  }
  std::vector<float> gpu_y(cpu_y.size()), gpu_state(init_state.size());
  ok =
      HipOk(hipDeviceSynchronize(), "sync prefill") &&
      HipOk(hipMemcpy(gpu_y.data(), d_y, cpu_y.size() * sizeof(float),
                      hipMemcpyDeviceToHost),
            "back y") &&
      HipOk(hipMemcpy(gpu_state.data(), d_state,
                      init_state.size() * sizeof(float), hipMemcpyDeviceToHost),
            "back state");
  hipFree(d_bcx);
  hipFree(d_conv);
  hipFree(d_state);
  hipFree(d_y);
  if (!ok) {
    Report("prefill_n" + std::to_string(n_tokens), false, "hip error");
    return;
  }
  float worst_y = 0.0F;
  float worst_s = 0.0F;
  for (std::size_t i = 0; i < gpu_y.size(); ++i) {
    worst_y = std::max(
        worst_y, std::fabs(gpu_y[i] - cpu_y[i]) / (1.0F + std::fabs(cpu_y[i])));
  }
  for (std::size_t i = 0; i < gpu_state.size(); ++i) {
    worst_s = std::max(worst_s, std::fabs(gpu_state[i] - cpu_state[i]));
  }
  Report("prefill_n" + std::to_string(n_tokens),
         worst_y <= 1e-5F && worst_s <= 1e-6F,
         "worst_y_scaled=" + F(worst_y) + " worst_state_abs=" + F(worst_s));
}

void CasePrefillDecodeEquiv(std::mt19937& rng,
                            std::uniform_real_distribution<float>& dist,
                            const std::vector<float>& conv) {
  constexpr std::uint32_t kH = 2048;
  constexpr std::uint32_t kL = 3;
  constexpr std::uint32_t kN = 37;
  std::vector<float> bcx(kN * 3U * kH);
  for (float& v : bcx) {
    v = dist(rng);
  }
  float *d_bcx = nullptr, *d_conv = nullptr, *d_sp = nullptr, *d_yp = nullptr,
        *d_sd = nullptr, *d_yd = nullptr, *d_bd = nullptr;
  const std::vector<float> zero((kL - 1U) * kH, 0.0F);
  bool ok =
      HipOk(hipMalloc(&d_bcx, bcx.size() * sizeof(float)), "malloc bcx") &&
      HipOk(hipMalloc(&d_conv, conv.size() * sizeof(float)), "malloc conv") &&
      HipOk(hipMalloc(&d_sp, zero.size() * sizeof(float)), "malloc sp") &&
      HipOk(hipMalloc(&d_yp, kN * kH * sizeof(float)), "malloc yp") &&
      HipOk(hipMalloc(&d_sd, zero.size() * sizeof(float)), "malloc sd") &&
      HipOk(hipMalloc(&d_yd, kH * sizeof(float)), "malloc yd") &&
      HipOk(hipMalloc(&d_bd, 3U * kH * sizeof(float)), "malloc bd") &&
      HipOk(hipMemcpy(d_bcx, bcx.data(), bcx.size() * sizeof(float),
                      hipMemcpyHostToDevice),
            "copy bcx") &&
      HipOk(hipMemcpy(d_conv, conv.data(), conv.size() * sizeof(float),
                      hipMemcpyHostToDevice),
            "copy conv") &&
      HipOk(hipMemcpy(d_sp, zero.data(), zero.size() * sizeof(float),
                      hipMemcpyHostToDevice),
            "copy sp") &&
      HipOk(hipMemcpy(d_sd, zero.data(), zero.size() * sizeof(float),
                      hipMemcpyHostToDevice),
            "copy sd");
  bool launched = false;
  if (ok) {
    const auto st = superroc::hip::LaunchLfm2ShortConvPrefill(
        d_bcx, d_conv, d_sp, d_yp, kH, kL, kN, nullptr);
    launched = st.ok;
    if (!st.ok) {
      Report("prefill37_vs_decode", false,
             st.error != nullptr ? st.error : "prefill launch");
    }
  }
  std::vector<std::vector<float>> dec_y(kN, std::vector<float>(kH));
  for (std::uint32_t t = 0; t < kN && launched; ++t) {
    if (!HipOk(hipMemcpy(d_bd, bcx.data() + t * 3U * kH,
                         3U * kH * sizeof(float), hipMemcpyHostToDevice),
               "copy row")) {
      launched = false;
      break;
    }
    const auto st = superroc::hip::LaunchLfm2ShortConvDecode(
        d_bd, d_conv, d_sd, d_yd, kH, kL, nullptr);
    if (!st.ok) {
      Report("prefill37_vs_decode", false,
             st.error != nullptr ? st.error : "decode launch");
      launched = false;
      break;
    }
    if (!HipOk(hipDeviceSynchronize(), "sync decode") ||
        !HipOk(hipMemcpy(dec_y[t].data(), d_yd, kH * sizeof(float),
                         hipMemcpyDeviceToHost),
               "back row")) {
      launched = false;
      break;
    }
  }
  std::vector<float> gpu_yp(kN * kH), gpu_sp(zero.size()), gpu_sd(zero.size());
  if (launched) {
    launched = HipOk(hipDeviceSynchronize(), "sync all") &&
               HipOk(hipMemcpy(gpu_yp.data(), d_yp, kN * kH * sizeof(float),
                               hipMemcpyDeviceToHost),
                     "back yp") &&
               HipOk(hipMemcpy(gpu_sp.data(), d_sp, zero.size() * sizeof(float),
                               hipMemcpyDeviceToHost),
                     "back sp") &&
               HipOk(hipMemcpy(gpu_sd.data(), d_sd, zero.size() * sizeof(float),
                               hipMemcpyDeviceToHost),
                     "back sd");
  }
  hipFree(d_bcx);
  hipFree(d_conv);
  hipFree(d_sp);
  hipFree(d_yp);
  hipFree(d_sd);
  hipFree(d_yd);
  hipFree(d_bd);
  if (!launched) {
    if (ok) {
      Report("prefill37_vs_decode", false, "hip error");
    } else {
      Report("prefill37_vs_decode", false, "alloc");
    }
    return;
  }
  float worst_y = 0.0F;
  float worst_s = 0.0F;
  for (std::uint32_t t = 0; t < kN; ++t) {
    for (std::uint32_t c = 0; c < kH; ++c) {
      worst_y = std::max(worst_y, std::fabs(gpu_yp[t * kH + c] - dec_y[t][c]));
    }
  }
  for (std::size_t i = 0; i < gpu_sp.size(); ++i) {
    worst_s = std::max(worst_s, std::fabs(gpu_sp[i] - gpu_sd[i]));
  }
  Report("prefill37_vs_decode", worst_y <= 1e-6F && worst_s <= 1e-6F,
         "worst_y_abs=" + F(worst_y) + " worst_state_abs=" + F(worst_s));
}

}  // namespace
namespace {

// Case c: route k=2048 E=64 U=4 over 200 vectors, plus E=256 U=8 over 20.
void CaseRouteOne(std::mt19937& rng,
                  std::uniform_real_distribution<float>& dist,
                  std::uniform_real_distribution<float>& bias_dist,
                  std::uint32_t n_expert, std::uint32_t n_used, int n_vecs) {
  constexpr std::uint32_t kK = 2048;
  std::vector<float> gate(n_expert * kK);
  for (float& v : gate) {
    v = dist(rng) * 0.05F;
  }
  std::vector<float> bias(n_expert);
  for (float& v : bias) {
    v = bias_dist(rng);
  }
  float *d_x = nullptr, *d_gate = nullptr, *d_bias = nullptr,
        *d_logits = nullptr;
  int* d_ids = nullptr;
  float* d_w = nullptr;
  bool ok =
      HipOk(hipMalloc(&d_x, kK * sizeof(float)), "malloc x") &&
      HipOk(hipMalloc(&d_gate, gate.size() * sizeof(float)), "malloc gate") &&
      HipOk(hipMalloc(&d_bias, bias.size() * sizeof(float)), "malloc bias") &&
      HipOk(hipMalloc(&d_logits, n_expert * sizeof(float)), "malloc logits") &&
      HipOk(hipMalloc(&d_ids, n_used * sizeof(int)), "malloc ids") &&
      HipOk(hipMalloc(&d_w, n_used * sizeof(float)), "malloc w") &&
      HipOk(hipMemcpy(d_gate, gate.data(), gate.size() * sizeof(float),
                      hipMemcpyHostToDevice),
            "copy gate") &&
      HipOk(hipMemcpy(d_bias, bias.data(), bias.size() * sizeof(float),
                      hipMemcpyHostToDevice),
            "copy bias");
  int id_bad = 0;
  float worst_w = 0.0F;
  float worst_l = 0.0F;
  float worst_sum = 0.0F;
  for (int v = 0; v < n_vecs && ok; ++v) {
    std::vector<float> x(kK);
    for (float& f : x) {
      f = dist(rng);
    }
    std::vector<float> cpu_logits(n_expert);
    std::vector<int> cpu_ids(n_used);
    std::vector<float> cpu_w(n_used);
    CpuMoeRoute(x.data(), gate.data(), bias.data(), kK, n_expert, n_used,
                cpu_logits.data(), cpu_ids.data(), cpu_w.data());
    ok = HipOk(
        hipMemcpy(d_x, x.data(), kK * sizeof(float), hipMemcpyHostToDevice),
        "copy x");
    const auto st = ok ? superroc::hip::LaunchLfm2MoeRoute(
                             d_x, d_gate, d_bias, kK, n_expert, n_used,
                             d_logits, d_ids, d_w, nullptr)
                       : superroc::hip::Lfm2LaunchStatus{false, "copy"};
    if (!st.ok) {
      Report("route_E" + std::to_string(n_expert), false,
             st.error != nullptr ? st.error : "launch");
      ok = false;
      break;
    }
    std::vector<float> gpu_logits(n_expert);
    std::vector<int> gpu_ids(n_used);
    std::vector<float> gpu_w(n_used);
    ok = HipOk(hipDeviceSynchronize(), "sync route") &&
         HipOk(hipMemcpy(gpu_logits.data(), d_logits, n_expert * sizeof(float),
                         hipMemcpyDeviceToHost),
               "back logits") &&
         HipOk(hipMemcpy(gpu_ids.data(), d_ids, n_used * sizeof(int),
                         hipMemcpyDeviceToHost),
               "back ids") &&
         HipOk(hipMemcpy(gpu_w.data(), d_w, n_used * sizeof(float),
                         hipMemcpyDeviceToHost),
               "back w");
    if (!ok) {
      break;
    }
    for (std::uint32_t i = 0; i < n_used; ++i) {
      if (gpu_ids[i] != cpu_ids[i]) {
        ++id_bad;
      }
      worst_w = std::max(worst_w, std::fabs(gpu_w[i] - cpu_w[i]));
    }
    for (std::uint32_t e = 0; e < n_expert; ++e) {
      worst_l = std::max(worst_l, std::fabs(gpu_logits[e] - cpu_logits[e]) /
                                      (1.0F + std::fabs(cpu_logits[e])));
    }
    double sum = 0.0;
    for (float w : gpu_w) {
      sum += w;
    }
    worst_sum = std::max(worst_sum, static_cast<float>(std::fabs(sum - 1.0)));
  }
  hipFree(d_x);
  hipFree(d_gate);
  hipFree(d_bias);
  hipFree(d_logits);
  hipFree(d_ids);
  hipFree(d_w);
  const std::string cas = "route_E" + std::to_string(n_expert) + "_U" +
                          std::to_string(n_used) + "x" + std::to_string(n_vecs);
  if (!ok) {
    Report(cas, false, "hip error");
    return;
  }
  Report(
      cas,
      id_bad == 0 && worst_w <= 1e-6F && worst_l <= 1e-4F && worst_sum <= 1e-6F,
      "id_bad=" + std::to_string(id_bad) + " worst_w_abs=" + F(worst_w) +
          " worst_logits_scaled=" + F(worst_l) +
          " worst_wsum_gap=" + F(worst_sum));
}
// Case e (Super ROC): exact sel ties. Experts e and e + n_expert/2 share the
// gate row and the bias, so their sel values are equal; ids must equal the
// CPU scan (lower id wins), weights within 1e-6.
void CaseRouteTies(std::mt19937& rng,
                   std::uniform_real_distribution<float>& dist,
                   std::uint32_t n_expert, std::uint32_t n_used, int n_vecs) {
  constexpr std::uint32_t kK = 2048;
  const std::uint32_t half = n_expert / 2U;
  std::vector<float> gate(n_expert * kK);
  std::vector<float> bias(n_expert);
  for (std::uint32_t e = 0; e < half; ++e) {
    for (std::uint32_t i = 0; i < kK; ++i) {
      gate[e * kK + i] = dist(rng) * 0.05F;
      gate[(e + half) * kK + i] = gate[e * kK + i];
    }
    bias[e] = 0.0F;
    bias[e + half] = 0.0F;
  }
  float *d_x = nullptr, *d_gate = nullptr, *d_bias = nullptr,
        *d_logits = nullptr, *d_w = nullptr;
  int* d_ids = nullptr;
  bool ok =
      HipOk(hipMalloc(&d_x, kK * sizeof(float)), "malloc x") &&
      HipOk(hipMalloc(&d_gate, gate.size() * sizeof(float)), "malloc gate") &&
      HipOk(hipMalloc(&d_bias, bias.size() * sizeof(float)), "malloc bias") &&
      HipOk(hipMalloc(&d_logits, n_expert * sizeof(float)), "malloc logits") &&
      HipOk(hipMalloc(&d_ids, n_used * sizeof(int)), "malloc ids") &&
      HipOk(hipMalloc(&d_w, n_used * sizeof(float)), "malloc w") &&
      HipOk(hipMemcpy(d_gate, gate.data(), gate.size() * sizeof(float),
                      hipMemcpyHostToDevice),
            "copy gate") &&
      HipOk(hipMemcpy(d_bias, bias.data(), bias.size() * sizeof(float),
                      hipMemcpyHostToDevice),
            "copy bias");
  int id_bad = 0;
  int tie_picks = 0;
  float worst_w = 0.0F;
  for (int v = 0; v < n_vecs && ok; ++v) {
    std::vector<float> x(kK);
    for (float& f : x) {
      f = dist(rng);
    }
    std::vector<float> cpu_logits(n_expert);
    std::vector<int> cpu_ids(n_used);
    std::vector<float> cpu_w(n_used);
    CpuMoeRoute(x.data(), gate.data(), bias.data(), kK, n_expert, n_used,
                cpu_logits.data(), cpu_ids.data(), cpu_w.data());
    ok = HipOk(
        hipMemcpy(d_x, x.data(), kK * sizeof(float), hipMemcpyHostToDevice),
        "copy x");
    const auto st = ok ? superroc::hip::LaunchLfm2MoeRoute(
                             d_x, d_gate, d_bias, kK, n_expert, n_used,
                             d_logits, d_ids, d_w, nullptr)
                       : superroc::hip::Lfm2LaunchStatus{false, "copy"};
    if (!st.ok) {
      ok = false;
      break;
    }
    std::vector<int> gpu_ids(n_used);
    std::vector<float> gpu_w(n_used);
    ok = HipOk(hipDeviceSynchronize(), "sync ties") &&
         HipOk(hipMemcpy(gpu_ids.data(), d_ids, n_used * sizeof(int),
                         hipMemcpyDeviceToHost),
               "back ids") &&
         HipOk(hipMemcpy(gpu_w.data(), d_w, n_used * sizeof(float),
                         hipMemcpyDeviceToHost),
               "back w");
    for (std::uint32_t i = 0; ok && i < n_used; ++i) {
      if (gpu_ids[i] != cpu_ids[i]) {
        ++id_bad;
      }
      if (cpu_ids[i] >= static_cast<int>(half)) {
        ++tie_picks;
      }
      worst_w = std::max(worst_w, std::fabs(gpu_w[i] - cpu_w[i]));
    }
  }
  hipFree(d_x);
  hipFree(d_gate);
  hipFree(d_bias);
  hipFree(d_logits);
  hipFree(d_ids);
  hipFree(d_w);
  const std::string cas = "route_ties_E" + std::to_string(n_expert) + "_U" +
                          std::to_string(n_used) + "x" + std::to_string(n_vecs);
  if (!ok) {
    Report(cas, false, "hip error");
    return;
  }
  // Each tied pair is picked low id first, so the high twin follows it.
  Report(cas, id_bad == 0 && worst_w <= 1e-6F && tie_picks > 0,
         "id_bad=" + std::to_string(id_bad) + " high_twin_picks=" +
             std::to_string(tie_picks) + " worst_w_abs=" + F(worst_w));
}

// Case d: batch rows bit-identical (memcmp) to the single-token launcher.
void CaseRouteBatchOne(std::mt19937& rng,
                       std::uniform_real_distribution<float>& dist,
                       const std::vector<float>& gate,
                       const std::vector<float>& bias, std::uint32_t n_tokens) {
  constexpr std::uint32_t kK = 2048;
  constexpr std::uint32_t kE = 64;
  constexpr std::uint32_t kU = 4;
  std::vector<float> x(n_tokens * kK);
  for (float& v : x) {
    v = dist(rng);
  }
  float *d_x = nullptr, *d_gate = nullptr, *d_bias = nullptr;
  float *d_logits = nullptr, *d_w = nullptr;
  int* d_ids = nullptr;
  float *s_x = nullptr, *s_logits = nullptr, *s_w = nullptr;
  int* s_ids = nullptr;
  bool ok =
      HipOk(hipMalloc(&d_x, x.size() * sizeof(float)), "malloc x") &&
      HipOk(hipMalloc(&d_gate, gate.size() * sizeof(float)), "malloc gate") &&
      HipOk(hipMalloc(&d_bias, bias.size() * sizeof(float)), "malloc bias") &&
      HipOk(hipMalloc(&d_logits, n_tokens * kE * sizeof(float)),
            "malloc logits") &&
      HipOk(hipMalloc(&d_ids, n_tokens * kU * sizeof(int)), "malloc ids") &&
      HipOk(hipMalloc(&d_w, n_tokens * kU * sizeof(float)), "malloc w") &&
      HipOk(hipMalloc(&s_x, kK * sizeof(float)), "malloc sx") &&
      HipOk(hipMalloc(&s_logits, kE * sizeof(float)), "malloc slogits") &&
      HipOk(hipMalloc(&s_ids, kU * sizeof(int)), "malloc sids") &&
      HipOk(hipMalloc(&s_w, kU * sizeof(float)), "malloc sw") &&
      HipOk(hipMemcpy(d_x, x.data(), x.size() * sizeof(float),
                      hipMemcpyHostToDevice),
            "copy x") &&
      HipOk(hipMemcpy(d_gate, gate.data(), gate.size() * sizeof(float),
                      hipMemcpyHostToDevice),
            "copy gate") &&
      HipOk(hipMemcpy(d_bias, bias.data(), bias.size() * sizeof(float),
                      hipMemcpyHostToDevice),
            "copy bias");
  const auto bst = ok ? superroc::hip::LaunchLfm2MoeRouteBatch(
                            d_x, d_gate, d_bias, kK, kE, n_tokens, kU, d_logits,
                            d_ids, d_w, nullptr)
                      : superroc::hip::Lfm2LaunchStatus{false, "alloc"};
  if (!bst.ok) {
    Report("routebatch_n" + std::to_string(n_tokens), false,
           bst.error != nullptr ? bst.error : "batch launch");
    ok = false;
  }
  std::vector<float> b_logits(n_tokens * kE);
  std::vector<int> b_ids(n_tokens * kU);
  std::vector<float> b_w(n_tokens * kU);
  if (ok) {
    ok =
        HipOk(hipDeviceSynchronize(), "sync batch") &&
        HipOk(hipMemcpy(b_logits.data(), d_logits,
                        b_logits.size() * sizeof(float), hipMemcpyDeviceToHost),
              "back logits") &&
        HipOk(hipMemcpy(b_ids.data(), d_ids, b_ids.size() * sizeof(int),
                        hipMemcpyDeviceToHost),
              "back ids") &&
        HipOk(hipMemcpy(b_w.data(), d_w, b_w.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "back w");
  }
  int rows_bad = 0;
  for (std::uint32_t t = 0; t < n_tokens && ok; ++t) {
    ok = HipOk(hipMemcpy(s_x, x.data() + t * kK, kK * sizeof(float),
                         hipMemcpyHostToDevice),
               "copy row x");
    const auto sst =
        ok ? superroc::hip::LaunchLfm2MoeRoute(s_x, d_gate, d_bias, kK, kE, kU,
                                               s_logits, s_ids, s_w, nullptr)
           : superroc::hip::Lfm2LaunchStatus{false, "copy"};
    if (!sst.ok) {
      ok = false;
      break;
    }
    std::vector<float> s_logits_h(kE), s_w_h(kU);
    std::vector<int> s_ids_h(kU);
    ok = HipOk(hipDeviceSynchronize(), "sync single") &&
         HipOk(hipMemcpy(s_logits_h.data(), s_logits, kE * sizeof(float),
                         hipMemcpyDeviceToHost),
               "back slogits") &&
         HipOk(hipMemcpy(s_ids_h.data(), s_ids, kU * sizeof(int),
                         hipMemcpyDeviceToHost),
               "back sids") &&
         HipOk(hipMemcpy(s_w_h.data(), s_w, kU * sizeof(float),
                         hipMemcpyDeviceToHost),
               "back sw");
    if (!ok) {
      break;
    }
    if (std::memcmp(s_logits_h.data(), b_logits.data() + t * kE,
                    kE * sizeof(float)) != 0 ||
        std::memcmp(s_ids_h.data(), b_ids.data() + t * kU, kU * sizeof(int)) !=
            0 ||
        std::memcmp(s_w_h.data(), b_w.data() + t * kU, kU * sizeof(float)) !=
            0) {
      ++rows_bad;
    }
  }
  hipFree(d_x);
  hipFree(d_gate);
  hipFree(d_bias);
  hipFree(d_logits);
  hipFree(d_ids);
  hipFree(d_w);
  hipFree(s_x);
  hipFree(s_logits);
  hipFree(s_ids);
  hipFree(s_w);
  Report("routebatch_n" + std::to_string(n_tokens), ok && rows_bad == 0,
         ok ? "rows_bad=" + std::to_string(rows_bad) + "/" +
                  std::to_string(n_tokens)
            : "hip error");
}
// Case e: refusals return ok == false and launch nothing (buffers untouched,
// no HIP error).
void CheckRefused(const std::string& cas, superroc::hip::Lfm2LaunchStatus st,
                  bool untouched) {
  const bool pass = !st.ok && st.error != nullptr && untouched &&
                    hipGetLastError() == hipSuccess;
  std::string nums = std::string("ok=") + (st.ok ? "true" : "false") +
                     " err=" + (st.error != nullptr ? st.error : "null") +
                     " untouched=" + (untouched ? "yes" : "no");
  Report(cas, pass, nums);
}

bool DeviceUnchanged(float* d_ptr, const std::vector<float>& expect) {
  std::vector<float> back(expect.size());
  if (!HipOk(hipMemcpy(back.data(), d_ptr, expect.size() * sizeof(float),
                       hipMemcpyDeviceToHost),
             "verify back")) {
    return false;
  }
  return std::memcmp(back.data(), expect.data(),
                     expect.size() * sizeof(float)) == 0;
}

// Case f (LKF): LaunchLfm2AddRMSNorm must equal LaunchResidualAdd followed by
// LaunchRMSNorm bit for bit (hidden and normed), for dims 2048 and 3072, with
// and without a weight.
void CaseAddRMSNorm(std::mt19937& rng,
                    std::uniform_real_distribution<float>& dist) {
  for (std::uint32_t dim : {2048U, 3072U}) {
    for (bool with_weight : {true, false}) {
      std::vector<float> h(dim), a(dim), w(dim);
      for (float& v : h) {
        v = dist(rng) * 4.0F;
      }
      for (float& v : a) {
        v = dist(rng);
      }
      for (float& v : w) {
        v = 1.0F + 0.5F * dist(rng);
      }
      float *d_h1 = nullptr, *d_h2 = nullptr, *d_a = nullptr, *d_w = nullptr,
            *d_o1 = nullptr, *d_o2 = nullptr;
      bool ok = HipOk(hipMalloc(&d_h1, dim * sizeof(float)), "malloc h1") &&
                HipOk(hipMalloc(&d_h2, dim * sizeof(float)), "malloc h2") &&
                HipOk(hipMalloc(&d_a, dim * sizeof(float)), "malloc a") &&
                HipOk(hipMalloc(&d_w, dim * sizeof(float)), "malloc w") &&
                HipOk(hipMalloc(&d_o1, dim * sizeof(float)), "malloc o1") &&
                HipOk(hipMalloc(&d_o2, dim * sizeof(float)), "malloc o2") &&
                HipOk(hipMemcpy(d_h1, h.data(), dim * sizeof(float),
                                hipMemcpyHostToDevice),
                      "copy h1") &&
                HipOk(hipMemcpy(d_h2, h.data(), dim * sizeof(float),
                                hipMemcpyHostToDevice),
                      "copy h2") &&
                HipOk(hipMemcpy(d_a, a.data(), dim * sizeof(float),
                                hipMemcpyHostToDevice),
                      "copy a") &&
                HipOk(hipMemcpy(d_w, w.data(), dim * sizeof(float),
                                hipMemcpyHostToDevice),
                      "copy w");
      const float* weight = with_weight ? d_w : nullptr;
      const std::string cas =
          "addrmsnorm_d" + std::to_string(dim) + (with_weight ? "_w" : "_now");
      if (ok) {
        superroc::hip::LaunchResidualAdd(d_h1, d_a, d_h1, dim, nullptr);
        superroc::hip::LaunchRMSNorm(d_h1, weight, d_o1, dim, 1e-5F, nullptr);
        const auto st = superroc::hip::LaunchLfm2AddRMSNorm(
            d_h2, d_a, weight, d_o2, dim, 1e-5F, nullptr);
        ok = st.ok && HipOk(hipDeviceSynchronize(), "sync addrmsnorm");
      }
      std::vector<float> h1(dim), h2(dim), o1(dim), o2(dim);
      ok = ok &&
           HipOk(hipMemcpy(h1.data(), d_h1, dim * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "back h1") &&
           HipOk(hipMemcpy(h2.data(), d_h2, dim * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "back h2") &&
           HipOk(hipMemcpy(o1.data(), d_o1, dim * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "back o1") &&
           HipOk(hipMemcpy(o2.data(), d_o2, dim * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "back o2");
      hipFree(d_h1);
      hipFree(d_h2);
      hipFree(d_a);
      hipFree(d_w);
      hipFree(d_o1);
      hipFree(d_o2);
      const bool h_eq =
          ok && std::memcmp(h1.data(), h2.data(), dim * sizeof(float)) == 0;
      const bool o_eq =
          ok && std::memcmp(o1.data(), o2.data(), dim * sizeof(float)) == 0;
      Report(cas, h_eq && o_eq,
             std::string("hidden_eq=") + (h_eq ? "1" : "0") +
                 " normed_eq=" + (o_eq ? "1" : "0"));
    }
  }
}

void CaseRefusals(std::mt19937& rng) {
  constexpr std::uint32_t kH = 2048;
  constexpr std::uint32_t kK = 2048;
  constexpr std::uint32_t kE = 64;
  constexpr std::uint32_t kU = 4;
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> bcx(3U * kH), conv(kH * 3U), state(2U * kH), y(kH);
  for (float& v : bcx) {
    v = dist(rng);
  }
  for (float& v : conv) {
    v = dist(rng);
  }
  for (float& v : state) {
    v = dist(rng);
  }
  for (float& v : y) {
    v = 0.25F;
  }
  std::vector<float> x(kK), gate(kE * kK), bias(kE), logits(kE);
  std::vector<float> weights(9, 0.25F);
  std::vector<int> ids(9, 17);
  for (float& v : x) {
    v = dist(rng);
  }
  for (float& v : gate) {
    v = dist(rng) * 0.05F;
  }
  for (float& v : bias) {
    v = dist(rng) * 0.1F;
  }
  float *d_bcx = nullptr, *d_conv = nullptr, *d_state = nullptr, *d_y = nullptr;
  float *d_x = nullptr, *d_gate = nullptr, *d_bias = nullptr;
  float *d_logits = nullptr, *d_w = nullptr;
  int* d_ids = nullptr;
  if (!HipOk(hipMalloc(&d_bcx, bcx.size() * sizeof(float)), "malloc bcx") ||
      !HipOk(hipMalloc(&d_conv, conv.size() * sizeof(float)), "malloc conv") ||
      !HipOk(hipMalloc(&d_state, state.size() * sizeof(float)),
             "malloc state") ||
      !HipOk(hipMalloc(&d_y, y.size() * sizeof(float)), "malloc y") ||
      !HipOk(hipMalloc(&d_x, x.size() * sizeof(float)), "malloc x") ||
      !HipOk(hipMalloc(&d_gate, gate.size() * sizeof(float)), "malloc gate") ||
      !HipOk(hipMalloc(&d_bias, bias.size() * sizeof(float)), "malloc bias") ||
      !HipOk(hipMalloc(&d_logits, logits.size() * sizeof(float)),
             "malloc logits") ||
      !HipOk(hipMalloc(&d_ids, ids.size() * sizeof(int)), "malloc ids") ||
      !HipOk(hipMalloc(&d_w, weights.size() * sizeof(float)), "malloc w") ||
      !HipOk(hipMemcpy(d_bcx, bcx.data(), bcx.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy bcx") ||
      !HipOk(hipMemcpy(d_conv, conv.data(), conv.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy conv") ||
      !HipOk(hipMemcpy(d_state, state.data(), state.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy state") ||
      !HipOk(hipMemcpy(d_y, y.data(), y.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy y") ||
      !HipOk(hipMemcpy(d_x, x.data(), x.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy x") ||
      !HipOk(hipMemcpy(d_gate, gate.data(), gate.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy gate") ||
      !HipOk(hipMemcpy(d_bias, bias.data(), bias.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy bias") ||
      !HipOk(hipMemcpy(d_logits, logits.data(), logits.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy logits") ||
      !HipOk(hipMemcpy(d_ids, ids.data(), ids.size() * sizeof(int),
                       hipMemcpyHostToDevice),
             "copy ids") ||
      !HipOk(hipMemcpy(d_w, weights.data(), weights.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy w")) {
    return;
  }
  hipGetLastError();
  CheckRefused(
      "refuse_n_used9",
      superroc::hip::LaunchLfm2MoeRoute(d_x, d_gate, d_bias, kK, kE, 9U,
                                        d_logits, d_ids, d_w, nullptr),
      DeviceUnchanged(d_w, weights));
  hipGetLastError();
  CheckRefused(
      "refuse_n_expert0",
      superroc::hip::LaunchLfm2MoeRoute(d_x, d_gate, d_bias, kK, 0U, kU,
                                        d_logits, d_ids, d_w, nullptr),
      DeviceUnchanged(d_w, weights));
  hipGetLastError();
  CheckRefused("refuse_l_cache1",
               superroc::hip::LaunchLfm2ShortConvDecode(d_bcx, d_conv, d_state,
                                                        d_y, kH, 1U, nullptr),
               DeviceUnchanged(d_y, y) && DeviceUnchanged(d_state, state));
  hipGetLastError();
  CheckRefused("refuse_l_cache9",
               superroc::hip::LaunchLfm2ShortConvDecode(d_bcx, d_conv, d_state,
                                                        d_y, kH, 9U, nullptr),
               DeviceUnchanged(d_y, y) && DeviceUnchanged(d_state, state));
  hipGetLastError();
  CheckRefused("refuse_hidden2047",
               superroc::hip::LaunchLfm2ShortConvDecode(
                   d_bcx, d_conv, d_state, d_y, 2047U, 3U, nullptr),
               DeviceUnchanged(d_y, y) && DeviceUnchanged(d_state, state));
  hipGetLastError();
  CheckRefused("refuse_null",
               superroc::hip::LaunchLfm2ShortConvDecode(
                   d_bcx, d_conv, d_state, nullptr, kH, 3U, nullptr),
               DeviceUnchanged(d_y, y) && DeviceUnchanged(d_state, state));
  hipFree(d_bcx);
  hipFree(d_conv);
  hipFree(d_state);
  hipFree(d_y);
  hipFree(d_x);
  hipFree(d_gate);
  hipFree(d_bias);
  hipFree(d_logits);
  hipFree(d_ids);
  hipFree(d_w);
}

}  // namespace

int main() {
  const int gate = superroc::test::GateHipDevice(
      superroc::test::HipDeviceRequirement::kOptional, "lfm2_kernels_test");
  if (gate != 0) {
    return gate;
  }
  std::mt19937 rng(20261014);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::uniform_real_distribution<float> bias_dist(-0.1F, 0.1F);
  CaseDecode(rng);
  std::vector<float> conv(2048U * 3U);
  for (float& v : conv) {
    v = dist(rng);
  }
  for (std::uint32_t n : {1U, 2U, 3U, 37U, 512U, 4096U}) {
    CasePrefillOne(rng, dist, conv, n);
  }
  CasePrefillDecodeEquiv(rng, dist, conv);
  CaseRouteOne(rng, dist, bias_dist, 64U, 4U, 200);
  CaseRouteOne(rng, dist, bias_dist, 256U, 8U, 20);
  CaseRouteTies(rng, dist, 64U, 4U, 100);
  CaseRouteTies(rng, dist, 256U, 8U, 20);
  std::vector<float> batch_gate(64U * 2048U);
  for (float& v : batch_gate) {
    v = dist(rng) * 0.05F;
  }
  std::vector<float> batch_bias(64U);
  for (float& v : batch_bias) {
    v = bias_dist(rng);
  }
  for (std::uint32_t n : {1U, 7U, 512U}) {
    CaseRouteBatchOne(rng, dist, batch_gate, batch_bias, n);
  }
  CaseAddRMSNorm(rng, dist);
  CaseRefusals(rng);
  if (g_fail == 0) {
    std::cout << "lfm2_kernels_test ALL PASS\n";
  }
  return g_fail;
}
#endif
