// KAT S3/S4 G1: the GPU MoE FFN against the CPU oracle on the real KAT GGUF.
//   A: routed expert slices e in {0, 8, 255} through superroc_fp4_expert_ffn_dev
//   B: the sigmoid-gated shared expert (QwenMoeShexpSigmoid + expert_ffn_dev)
//   C: the full single-token decode MoE (fast route + experts + shexp +
//      combine) vs ForwardMoe, incl. the router id set
//   D: the batched route (prefill head) over 17 tokens vs per-token CPU
//   E: the batched prefill MoE sequence over 5 tokens vs per-token ForwardMoe
// The GPU legs quantise the activation to Q8_1 (integer dot), the oracle dots
// in float, so the gate is the Q8-activation leg: cosine >= 0.9999.
#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/rocm_fp4/superroc_fp4_gemv_q8.hpp"
#include "src/core/rocm_fp4/superroc_fp4_quant.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/moe/forward_moe.hpp"
#include "src/models/qwen/moe/qwen_moe_ffn.hpp"
#include "src/models/qwen/moe/qwen_moe_prefill.hpp"
#include "src/models/qwen/moe/qwen_moe_route.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Expect(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL " << msg << "\n";
    std::exit(1);
  }
}

// exists() throws or says yes for a fixture the user cannot read; open it instead.
bool FixtureReadable(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return f.good();
}

#define HIP_REQUIRE(expr)                                                      \
  do {                                                                         \
    const hipError_t err_ = (expr);                                            \
    if (err_ != hipSuccess) {                                                  \
      std::cerr << "FAIL hip " << hipGetErrorString(err_) << "\n";             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct DeviceBuf {
  void* p = nullptr;
  explicit DeviceBuf(std::size_t bytes) { HIP_REQUIRE(hipMalloc(&p, bytes)); }
  ~DeviceBuf() {
    const hipError_t e = hipFree(p);
    (void)e;
  }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
  float* f32() { return static_cast<float*>(p); }
  int* i32() { return static_cast<int*>(p); }
};

std::uint32_t Mix(std::uint32_t x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

void Fill(std::vector<float>& v, std::uint32_t seed) {
  std::uint32_t s = seed;
  for (float& x : v) {
    s = Mix(s);
    x = static_cast<float>(static_cast<int>(s % 2001) - 1000) / 1000.0F;
  }
}

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  return dot / std::sqrt(na * nb);
}

void Upload(DeviceBuf& dst, const void* src, std::size_t bytes) {
  HIP_REQUIRE(hipMemcpy(dst.p, src, bytes, hipMemcpyHostToDevice));
}

void Fetch(std::vector<float>& dst, const DeviceBuf& src) {
  HIP_REQUIRE(hipMemcpy(dst.data(), src.p, dst.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
}

void FetchIds(std::vector<int>& dst, const DeviceBuf& src) {
  HIP_REQUIRE(hipMemcpy(dst.data(), src.p, dst.size() * sizeof(int),
                        hipMemcpyDeviceToHost));
}

}  // namespace

int main() {
  // Argument rejection before any launch.
  Expect(LaunchQwenMoeShexpSigmoid(nullptr, nullptr, 32, 1, nullptr,
                                   nullptr) == 1,
         "sigmoid null rejected");
  Expect(LaunchQwenMoeCombine(nullptr, nullptr, 32, 1, nullptr, nullptr) == 1,
         "combine null rejected");
  Expect(LaunchQwenMoeCombineSlots(nullptr, nullptr, nullptr, nullptr, 8, 32,
                                   1, nullptr, nullptr) == 1,
         "combine-slots null rejected");
  // CombineSlots vs the CPU ForwardMoe order (slots s=0..used-1
  // then sig*shexp) on synthetic data; bit-exact.
  {
    const int used = 3, hidden = 64, n = 5;
    std::vector<float> slot(n * used * hidden), w(n * used), sy(n * hidden),
        sig(n), ref(n * hidden);
    Fill(slot, 11);
    Fill(w, 23);
    Fill(sy, 37);
    Fill(sig, 53);
    for (int t = 0; t < n; ++t) {
      for (int h = 0; h < hidden; ++h) {
        // ForwardMoe order: routed slots s=0..used-1, then the shexp term.
        float acc = 0.0F;
        for (int s = 0; s < used; ++s)
          acc += w[t * used + s] * slot[(t * used + s) * hidden + h];
        acc += sig[t] * sy[t * hidden + h];
        ref[t * hidden + h] = acc;
      }
    }
    DeviceBuf d_slot(slot.size() * sizeof(float)),
        d_w(w.size() * sizeof(float)), d_sy(sy.size() * sizeof(float)),
        d_sig(sig.size() * sizeof(float)),
        d_out(ref.size() * sizeof(float));
    Upload(d_slot, slot.data(), slot.size() * sizeof(float));
    Upload(d_w, w.data(), w.size() * sizeof(float));
    Upload(d_sy, sy.data(), sy.size() * sizeof(float));
    Upload(d_sig, sig.data(), sig.size() * sizeof(float));
    Expect(LaunchQwenMoeCombineSlots(d_slot.f32(), d_w.f32(), d_sy.f32(),
                                     d_sig.f32(), used, hidden, n, d_out.f32(),
                                     nullptr) == 0,
           "combine-slots launch");
    HIP_REQUIRE(hipDeviceSynchronize());
    std::vector<float> out(ref.size());
    Fetch(out, d_out);
    // FMA contraction on the device (vs separate mul+add on the host) can
    // differ by 1 ulp per slot; compare with tolerance like the oracle legs.
    {
      double worst = 0.0;
      for (std::size_t i = 0; i < ref.size(); ++i)
        worst = std::max(
            worst, std::abs(static_cast<double>(out[i]) - ref[i]));
      Expect(worst < 1e-5, "combine-slots matches CPU order");
    }
  }
  Expect(LaunchQwenMoeRouteBatch(nullptr, nullptr, 32, 32, 1, 8, nullptr,
                                 nullptr, nullptr, nullptr) == 1,
         "batch route null rejected");

  const char* path_env = std::getenv("KAT_GGUF");
  const std::string path =
      path_env != nullptr
          ? path_env
          : "/path/to/models/swap/kat-coder/"
            "KAT-Coder-V2.5-Dev-Q4_0_ROCMFP4_STRIX-MTP.gguf";
  if (!FixtureReadable(path)) {
    std::cout << "SKIP: KAT GGUF not found: " << path << " (set KAT_GGUF)\n";
    return 77;
  }
  std::string error;
  auto reader = superroc::core::GgufReader::OpenFile(path, &error);
  Expect(reader != nullptr, "open: " + error);
  auto weights = superroc::models::QwenModelWeights::LoadFromGguf(*reader, &error);
  Expect(weights.has_value(), "bind: " + error);
  const auto& cfg = weights->config;
  const auto& layer0 = weights->layers[0];
  const int hidden = static_cast<int>(cfg.hidden_size);
  const int n_expert = static_cast<int>(cfg.expert_count);
  const int n_used = static_cast<int>(cfg.expert_used_count);
  const int ff = static_cast<int>(cfg.expert_feed_forward_length);
  const int ff_s = static_cast<int>(cfg.expert_shared_feed_forward_length);
  Expect(hidden == 2048 && n_expert == 256 && n_used == 8 && ff == 512 &&
             ff_s == 512,
         "kat shape");
  Expect(layer0.is_moe, "layer0 is moe");

  const int qtype = static_cast<int>(layer0.ffn_gate_exps.type);
  Expect(qtype == SUPERROC_FP4_Q4_0_ROCMFP4_FAST ||
             qtype == SUPERROC_FP4_Q4_0_ROCMFP4,
         "expert type fp4");
  Expect(layer0.ffn_up_exps.type == layer0.ffn_gate_exps.type &&
             layer0.ffn_down_exps.type == layer0.ffn_gate_exps.type,
         "expert types agree");
  const int qtype_sh = static_cast<int>(layer0.ffn_gate_shexp.type);
  Expect(layer0.ffn_gate_inp.type == superroc::core::GgmlType::kF32 &&
             layer0.ffn_gate_inp_shexp.type == superroc::core::GgmlType::kF32,
         "router + shexp gate f32");

  std::vector<float> x(static_cast<std::size_t>(hidden));
  Fill(x, 0x6A71u);

  // Device copies of the bound tensors (the host bind is mmapped read-only;
  // the expert kernels want device pointers, as in the remapped executor).
  // Byte counts follow the binder's convention (forward_moe.cpp ExpertSlice):
  // expert stride = n_rows * QuantizedRowBytes(type, k).
  const std::size_t rb_k =
      superroc::quant::QuantizedRowBytes(layer0.ffn_gate_exps.type,
                                     static_cast<std::size_t>(hidden));
  const std::size_t rb_ff =
      superroc::quant::QuantizedRowBytes(layer0.ffn_down_exps.type,
                                     static_cast<std::size_t>(ff));
  const std::size_t sh_rb_k =
      superroc::quant::QuantizedRowBytes(layer0.ffn_gate_shexp.type,
                                     static_cast<std::size_t>(hidden));
  const std::size_t sh_rb_ff =
      superroc::quant::QuantizedRowBytes(layer0.ffn_down_shexp.type,
                                     static_cast<std::size_t>(ff_s));
  Expect(rb_k > 0 && rb_ff > 0 && sh_rb_k > 0 && sh_rb_ff > 0, "row bytes");
  const std::size_t gate_bytes =
      static_cast<std::size_t>(n_expert) * ff * rb_k;
  const std::size_t up_bytes = gate_bytes;
  const std::size_t down_bytes =
      static_cast<std::size_t>(n_expert) * hidden * rb_ff;
  const std::size_t shg_bytes = static_cast<std::size_t>(ff_s) * sh_rb_k;
  const std::size_t shu_bytes = shg_bytes;
  const std::size_t shd_bytes = static_cast<std::size_t>(hidden) * sh_rb_ff;
  DeviceBuf d_gate(gate_bytes);
  DeviceBuf d_up(up_bytes);
  DeviceBuf d_down(down_bytes);
  Upload(d_gate, layer0.ffn_gate_exps.data, gate_bytes);
  Upload(d_up, layer0.ffn_up_exps.data, up_bytes);
  Upload(d_down, layer0.ffn_down_exps.data, down_bytes);
  DeviceBuf d_shgate(shg_bytes);
  DeviceBuf d_shup(shu_bytes);
  DeviceBuf d_shdown(shd_bytes);
  Upload(d_shgate, layer0.ffn_gate_shexp.data, shg_bytes);
  Upload(d_shup, layer0.ffn_up_shexp.data, shu_bytes);
  Upload(d_shdown, layer0.ffn_down_shexp.data, shd_bytes);
  DeviceBuf d_router(static_cast<std::size_t>(n_expert) * hidden *
                     sizeof(float));
  Upload(d_router, layer0.ffn_gate_inp.data,
         static_cast<std::size_t>(n_expert) * hidden * sizeof(float));
  DeviceBuf d_gatevec(static_cast<std::size_t>(hidden) * sizeof(float));
  Upload(d_gatevec, layer0.ffn_gate_inp_shexp.data,
         static_cast<std::size_t>(hidden) * sizeof(float));
  DeviceBuf d_x(static_cast<std::size_t>(hidden) * sizeof(float));
  Upload(d_x, x.data(), x.size() * sizeof(float));
  DeviceBuf d_logits(static_cast<std::size_t>(n_expert) * sizeof(float));
  DeviceBuf d_ids(n_used * sizeof(int));
  DeviceBuf d_w(n_used * sizeof(float));
  DeviceBuf d_y(static_cast<std::size_t>(hidden) * sizeof(float));
  DeviceBuf d_sig(sizeof(float));
  DeviceBuf d_zero(sizeof(int));
  HIP_REQUIRE(hipMemset(d_zero.p, 0, sizeof(int)));

  std::vector<float> gscratch(512), uscratch(512), ascratch(512);
  std::vector<float> cpu(static_cast<std::size_t>(hidden));
  std::vector<float> got(static_cast<std::size_t>(hidden));

  // --- A: routed slices -----------------------------------------------------
  for (int e : {0, 8, 255}) {
    HIP_REQUIRE(hipMemcpy(d_ids.p, &e, sizeof(int), hipMemcpyHostToDevice));
    const float one = 1.0F;
    HIP_REQUIRE(hipMemcpy(d_w.p, &one, sizeof(float), hipMemcpyHostToDevice));
    Expect(superroc_fp4_expert_ffn_dev(qtype, d_gate.p, d_up.p, d_down.p, d_x.f32(),
                                   d_ids.i32(), d_w.f32(), 1, n_expert, hidden,
                                   ff, 0.0F, d_y.f32(), nullptr) == SUPERROC_FP4_OK,
           "slice launch");
    HIP_REQUIRE(hipDeviceSynchronize());
    Fetch(got, d_y);
    superroc::models::KatExpertSwiglu(layer0.ffn_gate_exps, layer0.ffn_up_exps,
                                  layer0.ffn_down_exps, e, x, hidden, ff,
                                  gscratch, uscratch, ascratch, cpu);
    const double cosine = Cosine(got, cpu);
    std::cout << "KAT_FFN slice e=" << e << " cosine=" << cosine << "\n";
    Expect(cosine >= 0.9999, "slice cosine");
  }
  Expect(superroc_fp4_moe_dev_check() == SUPERROC_FP4_OK, "slice status");

  // --- B: shared expert + sigmoid gate --------------------------------------
  Expect(LaunchQwenMoeShexpSigmoid(d_x.f32(), d_gatevec.f32(), hidden, 1,
                                   d_sig.f32(), nullptr) == 0,
         "sigmoid launch");
  Expect(superroc_fp4_expert_ffn_dev(qtype_sh, d_shgate.p, d_shup.p, d_shdown.p,
                                 d_x.f32(), d_zero.i32(), d_sig.f32(), 1, 1,
                                 hidden, ff_s, 0.0F, d_y.f32(),
                                 nullptr) == SUPERROC_FP4_OK,
         "shexp launch");
  HIP_REQUIRE(hipDeviceSynchronize());
  Fetch(got, d_y);
  std::vector<float> sig_host(1);
  Fetch(sig_host, d_sig);
  superroc::models::KatExpertSwiglu(layer0.ffn_gate_shexp, layer0.ffn_up_shexp,
                                layer0.ffn_down_shexp, -1, x, hidden, ff_s,
                                gscratch, uscratch, ascratch, cpu);
  float dot = 0.0F;
  const auto gv = layer0.ffn_gate_inp_shexp.AsFloatSpan();
  Expect(gv.size() >= static_cast<std::size_t>(hidden), "gate vec size");
  for (int h = 0; h < hidden; ++h) {
    dot += gv[static_cast<std::size_t>(h)] * x[static_cast<std::size_t>(h)];
  }
  const float sig_ref = 1.0F / (1.0F + std::exp(-dot));
  for (int h = 0; h < hidden; ++h) {
    cpu[static_cast<std::size_t>(h)] *= sig_ref;
  }
  const double shexp_cos = Cosine(got, cpu);
  std::cout << "KAT_FFN shexp cosine=" << shexp_cos
            << " sig_device=" << sig_host[0] << " sig_ref=" << sig_ref
            << " sig_abs=" << std::fabs(sig_host[0] - sig_ref) << "\n";
  Expect(shexp_cos >= 0.9999, "shexp cosine");
  Expect(std::fabs(sig_host[0] - sig_ref) < 1.0e-4F, "sigmoid delta");
  Expect(superroc_fp4_moe_dev_check() == SUPERROC_FP4_OK, "shexp status");

  // --- C: full single-token decode MoE --------------------------------------
  Expect(LaunchQwenMoeRoute(d_x.f32(), d_router.f32(), hidden, n_expert,
                            n_used, d_logits.f32(), d_ids.i32(), d_w.f32(),
                            nullptr) == 0,
         "route launch");
  Expect(superroc_fp4_expert_ffn_dev(qtype, d_gate.p, d_up.p, d_down.p, d_x.f32(),
                                 d_ids.i32(), d_w.f32(), n_used, n_expert,
                                 hidden, ff, 0.0F, d_y.f32(),
                                 nullptr) == SUPERROC_FP4_OK,
         "experts launch");
  DeviceBuf d_sy(static_cast<std::size_t>(hidden) * sizeof(float));
  Expect(LaunchQwenMoeShexpSigmoid(d_x.f32(), d_gatevec.f32(), hidden, 1,
                                   d_sig.f32(), nullptr) == 0,
         "sigmoid relaunch");
  Expect(superroc_fp4_expert_ffn_dev(qtype_sh, d_shgate.p, d_shup.p, d_shdown.p,
                                 d_x.f32(), d_zero.i32(), d_sig.f32(), 1, 1,
                                 hidden, ff_s, 0.0F, d_sy.f32(),
                                 nullptr) == SUPERROC_FP4_OK,
         "shexp relaunch");
  DeviceBuf d_out(static_cast<std::size_t>(hidden) * sizeof(float));
  Expect(LaunchQwenMoeCombine(d_y.f32(), d_sy.f32(), hidden, 1, d_out.f32(),
                              nullptr) == 0,
         "combine launch");
  HIP_REQUIRE(hipDeviceSynchronize());
  Fetch(got, d_out);

  std::vector<float> moe_logits(static_cast<std::size_t>(n_expert));
  std::vector<float> moe_probs(static_cast<std::size_t>(n_expert));
  std::vector<float> eh(static_cast<std::size_t>(hidden));
  superroc::models::ForwardMoe(x, layer0, cfg, moe_logits, moe_probs, gscratch,
                           uscratch, ascratch, eh, cpu);
  const double moe_cos = Cosine(got, cpu);
  std::vector<int> cpu_ids(n_used);
  std::vector<float> cpu_w(n_used);
  Expect(superroc::models::KatRouteSoftmaxTopK(moe_logits, n_used, cpu_ids, cpu_w),
         "cpu route");
  std::vector<int> gpu_ids(static_cast<std::size_t>(n_used));
  FetchIds(gpu_ids, d_ids);
  std::sort(cpu_ids.begin(), cpu_ids.end());
  std::sort(gpu_ids.begin(), gpu_ids.end());
  Expect(cpu_ids == gpu_ids, "decode id set");
  std::cout << "KAT_FFN decode_moe cosine=" << moe_cos << " ids_match=1\n";
  Expect(moe_cos >= 0.9999, "decode moe cosine");
  Expect(superroc_fp4_moe_dev_check() == SUPERROC_FP4_OK, "decode status");

  // --- D: batched route head -------------------------------------------------
  constexpr int kTB = 17;
  std::vector<float> xb(static_cast<std::size_t>(kTB) * hidden);
  for (int t = 0; t < kTB; ++t) {
    std::vector<float> xt(static_cast<std::size_t>(hidden));
    Fill(xt, 0x1000u + static_cast<std::uint32_t>(t));
    std::memcpy(xb.data() + static_cast<std::size_t>(t) * hidden, xt.data(),
                hidden * sizeof(float));
  }
  DeviceBuf d_xb(xb.size() * sizeof(float));
  Upload(d_xb, xb.data(), xb.size() * sizeof(float));
  DeviceBuf d_logitsb(static_cast<std::size_t>(kTB) * n_expert * sizeof(float));
  DeviceBuf d_idsb(static_cast<std::size_t>(kTB) * n_used * sizeof(int));
  DeviceBuf d_wb(static_cast<std::size_t>(kTB) * n_used * sizeof(float));
  Expect(LaunchQwenMoeRouteBatch(d_xb.f32(), d_router.f32(), hidden, n_expert,
                                 kTB, n_used, d_logitsb.f32(), d_idsb.i32(),
                                 d_wb.f32(), nullptr) == 0,
         "batch route launch");
  HIP_REQUIRE(hipDeviceSynchronize());
  std::vector<int> ids_b(static_cast<std::size_t>(kTB) * n_used);
  std::vector<float> w_b(static_cast<std::size_t>(kTB) * n_used);
  FetchIds(ids_b, d_idsb);
  Fetch(w_b, d_wb);
  double worst_w = 1.0;
  for (int t = 0; t < kTB; ++t) {
    std::vector<float> xt(xb.begin() + static_cast<std::size_t>(t) * hidden,
                          xb.begin() + static_cast<std::size_t>(t + 1) * hidden);
    std::vector<float> lg(static_cast<std::size_t>(n_expert));
    superroc::models::TensorGEMV(layer0.ffn_gate_inp, xt,
                             static_cast<std::size_t>(n_expert), hidden, lg);
    std::vector<int> ci(n_used);
    std::vector<float> cw(n_used);
    Expect(superroc::models::KatRouteSoftmaxTopK(lg, n_used, ci, cw), "cpu route t");
    std::vector<int> gi(ids_b.begin() + static_cast<std::size_t>(t) * n_used,
                        ids_b.begin() +
                            static_cast<std::size_t>(t + 1) * n_used);
    std::vector<float> gw(w_b.begin() + static_cast<std::size_t>(t) * n_used,
                          w_b.begin() + static_cast<std::size_t>(t + 1) * n_used);
    std::sort(ci.begin(), ci.end());
    std::sort(gi.begin(), gi.end());
    Expect(ci == gi, "batch id set t=" + std::to_string(t));
    worst_w = std::min(worst_w, Cosine(cw, gw));
  }
  std::cout << "KAT_FFN batch_route tokens=" << kTB
            << " ids_match=1 worst_weight_cosine=" << worst_w << "\n";
  Expect(worst_w >= 0.999999, "batch weight cosine");

  // --- E: batched prefill MoE sequence --------------------------------------
  constexpr int kTP = 5;
  DeviceBuf d_sigb(static_cast<std::size_t>(kTP) * sizeof(float));
  DeviceBuf d_zerob(static_cast<std::size_t>(kTP) * sizeof(int));
  HIP_REQUIRE(hipMemset(d_zerob.p, 0, static_cast<std::size_t>(kTP) * sizeof(int)));
  DeviceBuf d_moey(static_cast<std::size_t>(kTP) * hidden * sizeof(float));
  DeviceBuf d_shey(static_cast<std::size_t>(kTP) * hidden * sizeof(float));
  DeviceBuf d_outb(static_cast<std::size_t>(kTP) * hidden * sizeof(float));
  Expect(LaunchQwenMoeRouteBatch(d_xb.f32(), d_router.f32(), hidden, n_expert,
                                 kTP, n_used, d_logitsb.f32(), d_idsb.i32(),
                                 d_wb.f32(), nullptr) == 0,
         "prefill route");
  Expect(LaunchQwenMoeShexpSigmoid(d_xb.f32(), d_gatevec.f32(), hidden, kTP,
                                   d_sigb.f32(), nullptr) == 0,
         "prefill sigmoid");
  for (int t0 = 0; t0 < kTP; t0 += 4) {
    const int m = std::min(4, kTP - t0);
    Expect(superroc_fp4_expert_ffn_m(
               qtype, d_gate.p, d_up.p, d_down.p,
               d_xb.f32() + static_cast<std::size_t>(t0) * hidden,
               d_idsb.i32() + static_cast<std::size_t>(t0) * n_used,
               d_wb.f32() + static_cast<std::size_t>(t0) * n_used, m, n_used,
               n_expert, hidden, ff, 0.0F,
               d_moey.f32() + static_cast<std::size_t>(t0) * hidden,
               nullptr) == SUPERROC_FP4_OK,
           "prefill experts");
    Expect(superroc_fp4_expert_ffn_shared_m(
               qtype_sh, d_shgate.p, d_shup.p, d_shdown.p,
               d_xb.f32() + static_cast<std::size_t>(t0) * hidden,
               d_zerob.i32() + t0, d_sigb.f32() + t0, m, 1, 1, hidden, ff_s,
               0.0F, d_shey.f32() + static_cast<std::size_t>(t0) * hidden,
               nullptr) == SUPERROC_FP4_OK,
           "prefill shexp");
  }
  Expect(superroc_fp4_expert_ffn_m_check() == SUPERROC_FP4_OK, "prefill status");
  Expect(LaunchQwenMoeCombine(d_moey.f32(), d_shey.f32(), hidden, kTP,
                              d_outb.f32(), nullptr) == 0,
         "prefill combine");
  HIP_REQUIRE(hipDeviceSynchronize());
  std::vector<float> out_b(static_cast<std::size_t>(kTP) * hidden);
  Fetch(out_b, d_outb);
  double worst_moe = 1.0;
  for (int t = 0; t < kTP; ++t) {
    std::vector<float> xt(xb.begin() + static_cast<std::size_t>(t) * hidden,
                          xb.begin() + static_cast<std::size_t>(t + 1) * hidden);
    superroc::models::ForwardMoe(xt, layer0, cfg, moe_logits, moe_probs, gscratch,
                             uscratch, ascratch, eh, cpu);
    std::vector<float> gt(out_b.begin() + static_cast<std::size_t>(t) * hidden,
                          out_b.begin() +
                              static_cast<std::size_t>(t + 1) * hidden);
    worst_moe = std::min(worst_moe, Cosine(gt, cpu));
  }
  std::cout << "KAT_FFN prefill_moe tokens=" << kTP
            << " worst_cosine=" << worst_moe << "\n";
  Expect(worst_moe >= 0.9999, "prefill moe cosine");


  // --- F: grouped path vs the m <= 4 path, one real layer. -----
  // KAT S6. 512 tokens is one sub-batch. 1500 tokens crosses the 1024
  // sub-batch split (1024 + 476), so scratch reuse and the second Ensure
  // actually run. Cosine is per token on the combined MoE output.
  Expect(LaunchKatMoeRoutedGrouped(0, nullptr, nullptr, nullptr, nullptr,
                                   nullptr, nullptr, 1, 8, 256, 32, 32, nullptr,
                                   nullptr) != 0,
         "grouped routed null");
  Expect(LaunchKatMoeSharedDense(0, nullptr, nullptr, nullptr, nullptr, nullptr,
                                 1, 32, 32, nullptr, nullptr) != 0,
         "grouped shared null");
  const int kTokenLegs[] = {512, 1500};
  for (const int kTG : kTokenLegs) {
  std::vector<float> xg(static_cast<std::size_t>(kTG) * hidden);
  Fill(xg, 0x5600u);
  DeviceBuf d_xg(xg.size() * sizeof(float));
  Upload(d_xg, xg.data(), xg.size() * sizeof(float));
  DeviceBuf d_logg(static_cast<std::size_t>(kTG) * n_expert * sizeof(float));
  DeviceBuf d_idg(static_cast<std::size_t>(kTG) * n_used * sizeof(int));
  DeviceBuf d_wg(static_cast<std::size_t>(kTG) * n_used * sizeof(float));
  DeviceBuf d_sigg(static_cast<std::size_t>(kTG) * sizeof(float));
  DeviceBuf d_zerg(static_cast<std::size_t>(kTG) * sizeof(int));
  HIP_REQUIRE(hipMemset(d_zerg.p, 0, static_cast<std::size_t>(kTG) * sizeof(int)));
  Expect(LaunchQwenMoeRouteBatch(d_xg.f32(), d_router.f32(), hidden, n_expert,
                                 kTG, n_used, d_logg.f32(), d_idg.i32(),
                                 d_wg.f32(), nullptr) == 0,
         "group route");
  Expect(LaunchQwenMoeShexpSigmoid(d_xg.f32(), d_gatevec.f32(), hidden, kTG,
                                   d_sigg.f32(), nullptr) == 0,
         "group sigmoid");
  DeviceBuf d_ref_y(static_cast<std::size_t>(kTG) * hidden * sizeof(float));
  DeviceBuf d_ref_s(static_cast<std::size_t>(kTG) * hidden * sizeof(float));
  DeviceBuf d_grp_y(static_cast<std::size_t>(kTG) * hidden * sizeof(float));
  DeviceBuf d_grp_s(static_cast<std::size_t>(kTG) * hidden * sizeof(float));
  for (int t0 = 0; t0 < kTG; t0 += 4) {
    const int m = std::min(4, kTG - t0);
    const std::size_t off = static_cast<std::size_t>(t0) * hidden;
    Expect(superroc_fp4_expert_ffn_m(
               qtype, d_gate.p, d_up.p, d_down.p, d_xg.f32() + off,
               d_idg.i32() + static_cast<std::size_t>(t0) * n_used,
               d_wg.f32() + static_cast<std::size_t>(t0) * n_used, m, n_used,
               n_expert, hidden, ff, 0.0F, d_ref_y.f32() + off,
               nullptr) == SUPERROC_FP4_OK,
           "m4 experts");
    Expect(superroc_fp4_expert_ffn_shared_m(
               qtype_sh, d_shgate.p, d_shup.p, d_shdown.p, d_xg.f32() + off,
               d_zerg.i32() + t0, d_sigg.f32() + t0, m, 1, 1, hidden, ff_s, 0.0F,
               d_ref_s.f32() + off, nullptr) == SUPERROC_FP4_OK,
           "m4 shared");
  }
  Expect(superroc_fp4_expert_ffn_m_check() == SUPERROC_FP4_OK, "m4 status");
  Expect(LaunchKatMoeRoutedGrouped(qtype, d_gate.p, d_up.p, d_down.p, d_xg.f32(),
                                   d_idg.i32(), d_wg.f32(), kTG, n_used,
                                   n_expert, hidden, ff, d_grp_y.f32(),
                                   nullptr) == 0,
         "grouped routed");
  Expect(LaunchKatMoeSharedDense(qtype_sh, d_shgate.p, d_shup.p, d_shdown.p,
                                 d_xg.f32(), d_sigg.f32(), kTG, hidden, ff_s,
                                 d_grp_s.f32(), nullptr) == 0,
         "grouped shared");
  HIP_REQUIRE(hipDeviceSynchronize());
  std::vector<float> ref_y(static_cast<std::size_t>(kTG) * hidden);
  std::vector<float> ref_s(ref_y.size());
  std::vector<float> grp_y(ref_y.size());
  std::vector<float> grp_s(ref_y.size());
  Fetch(ref_y, d_ref_y);
  Fetch(ref_s, d_ref_s);
  Fetch(grp_y, d_grp_y);
  Fetch(grp_s, d_grp_s);
  double worst = 1.0;
  double worst_r = 1.0;
  double worst_s = 1.0;
  float max_abs = 0.0F;
  for (int t = 0; t < kTG; ++t) {
    const std::size_t b = static_cast<std::size_t>(t) * hidden;
    const std::size_t e = b + hidden;
    std::vector<float> ry(ref_y.begin() + b, ref_y.begin() + e);
    std::vector<float> rs(ref_s.begin() + b, ref_s.begin() + e);
    std::vector<float> gy(grp_y.begin() + b, grp_y.begin() + e);
    std::vector<float> gs(grp_s.begin() + b, grp_s.begin() + e);
    std::vector<float> rc(ry.size());
    std::vector<float> gc(gy.size());
    for (int i = 0; i < hidden; ++i) {
      rc[i] = ry[i] + rs[i];
      gc[i] = gy[i] + gs[i];
      max_abs = std::max(max_abs, std::fabs(rc[i] - gc[i]));
    }
    worst = std::min(worst, Cosine(rc, gc));
    worst_r = std::min(worst_r, Cosine(ry, gy));
    worst_s = std::min(worst_s, Cosine(rs, gs));
  }
  std::cout << "KAT_FFN grouped_vs_m4 tokens=" << kTG
            << " worst_cosine=" << worst << " routed_cosine=" << worst_r
            << " shared_cosine=" << worst_s << " max_abs=" << max_abs << "\n";
  Expect(worst >= 0.9999, "grouped vs m4 cosine");
  Expect(worst_r >= 0.9999, "grouped routed cosine");
  Expect(worst_s >= 0.9999, "grouped shared cosine");
  }

  // --- moebatch leg F: the batch FFN vs one call per row, bit-identical. -----
  // Row 0 is leg C's input; rows 1 and 2 are the same vector scaled. Ids are
  // {0,1,..} / {8,9,..} / {16,17,..}, weights 1/n_used each.
  constexpr int kMB = 3;
  // legs G and I run up to 8 verify rows, so the batch input carries 8 scaled rows
  constexpr int kXR = 8;
  const float xsc[kXR - 1] = {0.5F, -0.25F, 0.125F, -0.5F, 0.75F, -0.125F, 0.375F};
  std::vector<float> xm(static_cast<std::size_t>(kXR) * hidden);
  std::memcpy(xm.data(), x.data(), static_cast<std::size_t>(hidden) * sizeof(float));
  for (int t = 1; t < kXR; ++t) {
    for (int h = 0; h < hidden; ++h) {
      xm[static_cast<std::size_t>(t) * hidden + h] = xsc[t - 1] * x[static_cast<std::size_t>(h)];
    }
  }
  std::vector<int> idm(static_cast<std::size_t>(kMB) * n_used);
  std::vector<float> wm(static_cast<std::size_t>(kMB) * n_used);
  for (int t = 0; t < kMB; ++t) {
    for (int s = 0; s < n_used; ++s) {
      idm[static_cast<std::size_t>(t) * n_used + s] = n_used * t + s;
      wm[static_cast<std::size_t>(t) * n_used + s] = 1.0F / static_cast<float>(n_used);
    }
  }
  Expect(n_used * (kMB - 1) + (n_used - 1) < n_expert, "leg F ids in range");
  DeviceBuf d_xm(xm.size() * sizeof(float));
  Upload(d_xm, xm.data(), xm.size() * sizeof(float));
  DeviceBuf d_idm(idm.size() * sizeof(int));
  Upload(d_idm, idm.data(), idm.size() * sizeof(int));
  DeviceBuf d_wm(wm.size() * sizeof(float));
  Upload(d_wm, wm.data(), wm.size() * sizeof(float));
  DeviceBuf d_zm(static_cast<std::size_t>(kMB) * sizeof(int));
  HIP_REQUIRE(hipMemset(d_zm.p, 0, static_cast<std::size_t>(kMB) * sizeof(int)));
  const float sigv[kMB] = {0.9F, 0.5F, 0.1F};
  DeviceBuf d_sigm(static_cast<std::size_t>(kMB) * sizeof(float));
  Upload(d_sigm, sigv, sizeof(sigv));
  DeviceBuf d_yb(static_cast<std::size_t>(kMB) * hidden * sizeof(float));
  DeviceBuf d_syb(static_cast<std::size_t>(kMB) * hidden * sizeof(float));
  DeviceBuf d_ref1(static_cast<std::size_t>(kMB) * hidden * sizeof(float));
  DeviceBuf d_refs1(static_cast<std::size_t>(kMB) * hidden * sizeof(float));
  DeviceBuf d_ref2(static_cast<std::size_t>(kMB) * hidden * sizeof(float));
  DeviceBuf d_refs2(static_cast<std::size_t>(kMB) * hidden * sizeof(float));
  DeviceBuf d_xr(static_cast<std::size_t>(hidden) * sizeof(float));
  DeviceBuf d_idr(n_used * sizeof(int));
  DeviceBuf d_wr(n_used * sizeof(float));
  DeviceBuf d_sigr(sizeof(float));
  DeviceBuf d_zr(sizeof(int));
  HIP_REQUIRE(hipMemset(d_zr.p, 0, sizeof(int)));
  auto run_leg_f_ref = [&](DeviceBuf& dy, DeviceBuf& dsy) {
    for (int t = 0; t < kMB; ++t) {
      const std::size_t xo = static_cast<std::size_t>(t) * hidden;
      HIP_REQUIRE(hipMemcpy(d_xr.p, xm.data() + xo,
                            static_cast<std::size_t>(hidden) * sizeof(float),
                            hipMemcpyHostToDevice));
      HIP_REQUIRE(hipMemcpy(d_idr.p, idm.data() + static_cast<std::size_t>(t) * n_used,
                            static_cast<std::size_t>(n_used) * sizeof(int),
                            hipMemcpyHostToDevice));
      HIP_REQUIRE(hipMemcpy(d_wr.p, wm.data() + static_cast<std::size_t>(t) * n_used,
                            static_cast<std::size_t>(n_used) * sizeof(float),
                            hipMemcpyHostToDevice));
      HIP_REQUIRE(hipMemcpy(d_sigr.p, sigv + t, sizeof(float), hipMemcpyHostToDevice));
      Expect(superroc_fp4_expert_ffn_dev(qtype, d_gate.p, d_up.p, d_down.p, d_xr.f32(),
                                     d_idr.i32(), d_wr.f32(), n_used, n_expert,
                                     hidden, ff, 0.0F, dy.f32() + xo,
                                     nullptr) == SUPERROC_FP4_OK,
             "leg F ref routed");
      // The shared expert right after its routed call, as LaunchVerifyFfn does.
      Expect(superroc_fp4_expert_ffn_dev_xq(qtype_sh, d_shgate.p, d_shup.p, d_shdown.p,
                                        d_xr.f32(), d_zr.i32(), d_sigr.f32(), 1, 1,
                                        hidden, ff_s, 0.0F, dsy.f32() + xo,
                                        nullptr) == SUPERROC_FP4_OK,
             "leg F ref shexp");
    }
    HIP_REQUIRE(hipDeviceSynchronize());
  };
  run_leg_f_ref(d_ref1, d_refs1);
  Expect(superroc_fp4_expert_ffn_dev_batch(qtype, d_gate.p, d_up.p, d_down.p, d_xm.f32(),
                                       d_idm.i32(), d_wm.f32(), kMB, n_used,
                                       n_expert, hidden, ff, 0.0F, d_yb.f32(),
                                       nullptr, 1) == SUPERROC_FP4_OK,
         "leg F batch routed");
  Expect(superroc_fp4_expert_ffn_dev_batch(qtype_sh, d_shgate.p, d_shup.p, d_shdown.p,
                                       d_xm.f32(), d_zm.i32(), d_sigm.f32(), kMB, 1,
                                       1, hidden, ff_s, 0.0F, d_syb.f32(), nullptr,
                                       0) == SUPERROC_FP4_OK,
         "leg F batch shexp");
  HIP_REQUIRE(hipDeviceSynchronize());
  // The batch must not disturb the slots: the per-row reference again, into
  // fresh buffers, has to match the first pass byte for byte.
  run_leg_f_ref(d_ref2, d_refs2);
  std::vector<float> r1(static_cast<std::size_t>(kMB) * hidden);
  std::vector<float> r2(r1.size());
  std::vector<float> bt(r1.size());
  std::vector<float> s1(r1.size());
  std::vector<float> s2(r1.size());
  std::vector<float> sb(r1.size());
  Fetch(r1, d_ref1);
  Fetch(r2, d_ref2);
  Fetch(bt, d_yb);
  Fetch(s1, d_refs1);
  Fetch(s2, d_refs2);
  Fetch(sb, d_syb);
  Expect(std::memcmp(r1.data(), r2.data(), r1.size() * sizeof(float)) == 0,
         "leg F slots undisturbed");
  for (int t = 0; t < kMB; ++t) {
    const std::size_t o = static_cast<std::size_t>(t) * hidden;
    int bad_r = -1;
    int bad_s = -1;
    for (int h = 0; h < hidden; ++h) {
      if (bad_r < 0 &&
          std::memcmp(r1.data() + o + h, bt.data() + o + h, sizeof(float)) != 0) {
        bad_r = h;
      }
      if (bad_s < 0 &&
          std::memcmp(s1.data() + o + h, sb.data() + o + h, sizeof(float)) != 0) {
        bad_s = h;
      }
    }
    if (bad_r >= 0) {
      std::cerr << "moebatch leg F routed row=" << t << " first_diff=" << bad_r
                << " ref=" << r1[o + static_cast<std::size_t>(bad_r)]
                << " batch=" << bt[o + static_cast<std::size_t>(bad_r)] << "\n";
    }
    if (bad_s >= 0) {
      std::cerr << "moebatch leg F shared row=" << t << " first_diff=" << bad_s
                << " ref=" << s1[o + static_cast<std::size_t>(bad_s)]
                << " batch=" << sb[o + static_cast<std::size_t>(bad_s)] << "\n";
    }
    Expect(bad_r < 0, "leg F routed bit-identical");
    Expect(bad_s < 0, "leg F shared bit-identical");
  }
  Expect(superroc_fp4_moe_dev_check() == SUPERROC_FP4_OK, "leg F status");
  std::cout << "moebatch leg F OK\n";

  // --- moebatch leg G: routed + shared experts in one launch, bit-identical. -----
  // Same weights as leg F, but the ids repeat experts ACROSS rows -- row 1 walks row 0's
  // first half backwards, row 2 keeps expert 0 and expert n_used-1, rows 3..7 follow row
  // 2's recipe -- which is the dedup the group record exists for. The loop runs 2..8 rows
  // of one round. Weights 1/(slot+1) make the slot order visible in the sum. The reference
  // is the exact LaunchVerifyFfn sequence, per row, then one combine.
  constexpr int kGR = 3;
  std::vector<int> idg(static_cast<std::size_t>(kXR) * n_used);
  for (int s = 0; s < n_used; ++s) {
    idg[static_cast<std::size_t>(s)] = s % n_expert;                       // row 0: 0,1,...
  }
  for (int s = 0; s < n_used; ++s) {
    const int v = s < n_used / 2 ? idg[static_cast<std::size_t>(n_used / 2 - 1 - s)]
                                 : n_used + (s - n_used / 2);
    idg[static_cast<std::size_t>(n_used) + s] = ((v % n_expert) + n_expert) % n_expert;
    idg[static_cast<std::size_t>(2 * n_used) + s] = s == 0   ? 0
                                                    : s == 1 ? (n_used - 1) % n_expert
                                                             : (2 * n_used + s - 2) % n_expert;
  }
  // rows 3..7: row 2's recipe, so experts 0 and n_used-1 repeat across rows
  for (int t = kGR; t < kXR; ++t) {
    for (int s = 0; s < n_used; ++s) {
      idg[static_cast<std::size_t>(t) * n_used + s] = s == 0 ? 0
                                                     : s == 1 ? (n_used - 1) % n_expert
                                                              : (t * n_used + s - 2) % n_expert;
    }
  }
  for (int t = 0; t < kXR; ++t) {
    for (int a = 0; a < n_used; ++a) {
      for (int b = a + 1; b < n_used; ++b) {
        Expect(idg[static_cast<std::size_t>(t) * n_used + a] !=
                   idg[static_cast<std::size_t>(t) * n_used + b],
               "leg G row ids distinct");
      }
    }
  }
  const float sigg[kXR] = {0.9F, 0.5F, 0.1F, 0.3F, 0.7F, 0.2F, 0.6F, 0.4F};
  DeviceBuf d_xgr(static_cast<std::size_t>(hidden) * sizeof(float));
  DeviceBuf d_idgr(static_cast<std::size_t>(n_used) * sizeof(int));
  DeviceBuf d_wrgr(static_cast<std::size_t>(n_used) * sizeof(float));
  DeviceBuf d_siggr(sizeof(float));
  DeviceBuf d_zgr(sizeof(int));
  HIP_REQUIRE(hipMemset(d_zgr.p, 0, sizeof(int)));
  for (const int rows : {kGR, 2, 5, 6, 7, kXR}) {
    const std::size_t slots = static_cast<std::size_t>(rows) * n_used;
    std::vector<int> idr(idg.begin(), idg.begin() + static_cast<std::ptrdiff_t>(slots));
    std::vector<float> wgr(slots);
    for (std::size_t i = 0; i < slots; ++i) {
      wgr[i] = 1.0F / static_cast<float>(static_cast<int>(i % static_cast<std::size_t>(n_used)) + 1);
    }
    DeviceBuf d_id_gr(slots * sizeof(int));
    Upload(d_id_gr, idr.data(), idr.size() * sizeof(int));
    DeviceBuf d_w_gr(slots * sizeof(float));
    Upload(d_w_gr, wgr.data(), wgr.size() * sizeof(float));
    DeviceBuf d_sig_gr(static_cast<std::size_t>(rows) * sizeof(float));
    Upload(d_sig_gr, sigg, static_cast<std::size_t>(rows) * sizeof(float));
    DeviceBuf d_ref_y(slots * hidden * sizeof(float));
    DeviceBuf d_ref_s(slots * hidden * sizeof(float));
    DeviceBuf d_ref_out(slots * hidden * sizeof(float));
    DeviceBuf d_all_out(slots * hidden * sizeof(float));
    for (int t = 0; t < rows; ++t) {
      const std::size_t xo = static_cast<std::size_t>(t) * hidden;
      HIP_REQUIRE(hipMemcpy(d_xgr.p, xm.data() + xo,
                            static_cast<std::size_t>(hidden) * sizeof(float),
                            hipMemcpyHostToDevice));
      HIP_REQUIRE(hipMemcpy(d_idgr.p, idr.data() + static_cast<std::size_t>(t) * n_used,
                            static_cast<std::size_t>(n_used) * sizeof(int),
                            hipMemcpyHostToDevice));
      HIP_REQUIRE(hipMemcpy(d_wrgr.p, wgr.data() + static_cast<std::size_t>(t) * n_used,
                            static_cast<std::size_t>(n_used) * sizeof(float),
                            hipMemcpyHostToDevice));
      HIP_REQUIRE(hipMemcpy(d_siggr.p, sigg + t, sizeof(float), hipMemcpyHostToDevice));
      Expect(superroc_fp4_expert_ffn_dev(qtype, d_gate.p, d_up.p, d_down.p, d_xgr.f32(),
                                     d_idgr.i32(), d_wrgr.f32(), n_used, n_expert,
                                     hidden, ff, 0.0F, d_ref_y.f32() + xo,
                                     nullptr) == SUPERROC_FP4_OK,
             "leg G ref routed");
      Expect(superroc_fp4_expert_ffn_dev_xq(qtype_sh, d_shgate.p, d_shup.p, d_shdown.p,
                                        d_xgr.f32(), d_zgr.i32(), d_siggr.f32(), 1, 1,
                                        hidden, ff_s, 0.0F, d_ref_s.f32() + xo,
                                        nullptr) == SUPERROC_FP4_OK,
             "leg G ref shexp");
    }
    Expect(LaunchQwenMoeCombine(d_ref_y.f32(), d_ref_s.f32(), hidden, rows,
                                d_ref_out.f32(), nullptr) == 0,
           "leg G ref combine");
    Expect(superroc_fp4_expert_ffn_dev_batch_all(
               qtype, d_gate.p, d_up.p, d_down.p, d_shgate.p, d_shup.p, d_shdown.p,
               d_xm.f32(), d_id_gr.i32(), d_w_gr.f32(), d_sig_gr.f32(), rows, n_used,
               n_expert, hidden, ff, ff_s, 0.0F, d_all_out.f32(),
               nullptr) == SUPERROC_FP4_OK,
           "leg G batch all");
    HIP_REQUIRE(hipDeviceSynchronize());
    Expect(superroc_fp4_moe_dev_check() == SUPERROC_FP4_OK, "leg G status");
    std::vector<float> refg(slots * hidden);
    std::vector<float> allg(refg.size());
    Fetch(refg, d_ref_out);
    Fetch(allg, d_all_out);
    for (int t = 0; t < rows; ++t) {
      const std::size_t o = static_cast<std::size_t>(t) * hidden;
      int bad = -1;
      for (int h = 0; h < hidden; ++h) {
        if (std::memcmp(refg.data() + o + h, allg.data() + o + h, sizeof(float)) != 0) {
          bad = h;
          break;
        }
      }
      if (bad >= 0) {
        std::cerr << "moebatch leg G row=" << t << " first_diff=" << bad
                  << " ref=" << refg[o + static_cast<std::size_t>(bad)]
                  << " all=" << allg[o + static_cast<std::size_t>(bad)] << "\n";
      }
      Expect(std::memcmp(refg.data() + o, allg.data() + o,
                         static_cast<std::size_t>(hidden) * sizeof(float)) == 0,
             "leg G bit-identical");
    }
    std::cout << "moebatch leg G OK rows=" << rows << "\n";
  }

  // --- moebatch leg H: one expert fills every slot of both rows. ------------
  // Same reference sequence and per-row memcmp as leg G's rows == 2 case, but
  // every id is expert 0, so one group record takes 2*n_used fan-out slots.
  // The group builder has to keep every occurrence (only a full record starts
  // a new one); dropping the fifth would lose slots and set the status word.
  constexpr int kHR = 2;
  const std::size_t h_slots = static_cast<std::size_t>(kHR) * n_used;
  Expect(static_cast<std::size_t>(kHR) * static_cast<std::size_t>(n_used) > 4,
         "leg H slots exceed one record");
  std::vector<int> idh(h_slots, 0);
  std::vector<float> wh(h_slots);
  for (std::size_t i = 0; i < h_slots; ++i) {
    wh[i] = 1.0F / static_cast<float>(static_cast<int>(i % static_cast<std::size_t>(n_used)) + 1);
  }
  const float sigh[kHR] = {0.9F, 0.5F};
  DeviceBuf d_xgh(static_cast<std::size_t>(hidden) * sizeof(float));
  DeviceBuf d_idgh(static_cast<std::size_t>(n_used) * sizeof(int));
  DeviceBuf d_wrgh(static_cast<std::size_t>(n_used) * sizeof(float));
  DeviceBuf d_siggh(sizeof(float));
  DeviceBuf d_zgh(sizeof(int));
  HIP_REQUIRE(hipMemset(d_zgh.p, 0, sizeof(int)));
  DeviceBuf d_id_h(h_slots * sizeof(int));
  Upload(d_id_h, idh.data(), idh.size() * sizeof(int));
  DeviceBuf d_w_h(h_slots * sizeof(float));
  Upload(d_w_h, wh.data(), wh.size() * sizeof(float));
  DeviceBuf d_sig_h(static_cast<std::size_t>(kHR) * sizeof(float));
  Upload(d_sig_h, sigh, static_cast<std::size_t>(kHR) * sizeof(float));
  DeviceBuf d_ref_yh(h_slots * hidden * sizeof(float));
  DeviceBuf d_ref_sh(h_slots * hidden * sizeof(float));
  DeviceBuf d_ref_outh(h_slots * hidden * sizeof(float));
  DeviceBuf d_all_outh(h_slots * hidden * sizeof(float));
  for (int t = 0; t < kHR; ++t) {
    const std::size_t xo = static_cast<std::size_t>(t) * hidden;
    HIP_REQUIRE(hipMemcpy(d_xgh.p, xm.data() + xo,
                          static_cast<std::size_t>(hidden) * sizeof(float),
                          hipMemcpyHostToDevice));
    HIP_REQUIRE(hipMemcpy(d_idgh.p, idh.data() + static_cast<std::size_t>(t) * n_used,
                          static_cast<std::size_t>(n_used) * sizeof(int),
                          hipMemcpyHostToDevice));
    HIP_REQUIRE(hipMemcpy(d_wrgh.p, wh.data() + static_cast<std::size_t>(t) * n_used,
                          static_cast<std::size_t>(n_used) * sizeof(float),
                          hipMemcpyHostToDevice));
    HIP_REQUIRE(hipMemcpy(d_siggh.p, sigh + t, sizeof(float),
                          hipMemcpyHostToDevice));
    Expect(superroc_fp4_expert_ffn_dev(qtype, d_gate.p, d_up.p, d_down.p, d_xgh.f32(),
                                   d_idgh.i32(), d_wrgh.f32(), n_used, n_expert,
                                   hidden, ff, 0.0F, d_ref_yh.f32() + xo,
                                   nullptr) == SUPERROC_FP4_OK,
           "leg H ref routed");
    Expect(superroc_fp4_expert_ffn_dev_xq(qtype_sh, d_shgate.p, d_shup.p, d_shdown.p,
                                      d_xgh.f32(), d_zgh.i32(), d_siggh.f32(), 1, 1,
                                      hidden, ff_s, 0.0F, d_ref_sh.f32() + xo,
                                      nullptr) == SUPERROC_FP4_OK,
           "leg H ref shexp");
  }
  Expect(LaunchQwenMoeCombine(d_ref_yh.f32(), d_ref_sh.f32(), hidden, kHR,
                              d_ref_outh.f32(), nullptr) == 0,
         "leg H ref combine");
  Expect(superroc_fp4_expert_ffn_dev_batch_all(
             qtype, d_gate.p, d_up.p, d_down.p, d_shgate.p, d_shup.p, d_shdown.p,
             d_xm.f32(), d_id_h.i32(), d_w_h.f32(), d_sig_h.f32(), kHR, n_used,
             n_expert, hidden, ff, ff_s, 0.0F, d_all_outh.f32(),
             nullptr) == SUPERROC_FP4_OK,
         "leg H batch all");
  HIP_REQUIRE(hipDeviceSynchronize());
  Expect(superroc_fp4_moe_dev_check() == SUPERROC_FP4_OK, "leg H status");
  std::vector<float> refh(h_slots * hidden);
  std::vector<float> allh(refh.size());
  Fetch(refh, d_ref_outh);
  Fetch(allh, d_all_outh);
  for (int t = 0; t < kHR; ++t) {
    const std::size_t o = static_cast<std::size_t>(t) * hidden;
    int bad = -1;
    for (int h = 0; h < hidden; ++h) {
      if (std::memcmp(refh.data() + o + h, allh.data() + o + h, sizeof(float)) != 0) {
        bad = h;
        break;
      }
    }
    if (bad >= 0) {
      std::cerr << "moebatch leg H row=" << t << " first_diff=" << bad
                << " ref=" << refh[o + static_cast<std::size_t>(bad)]
                << " all=" << allh[o + static_cast<std::size_t>(bad)] << "\n";
    }
    Expect(std::memcmp(refh.data() + o, allh.data() + o,
                       static_cast<std::size_t>(hidden) * sizeof(float)) == 0,
           "leg H bit-identical");
  }
  std::cout << "moebatch leg H OK\n";

  // --- moebatch leg I: eight rows on the same ids, two records per expert. ---
  // Every row walks the same n_used experts in the same order, so each expert
  // takes 8 fan-outs: one full record and one follow-on record. Same reference
  // sequence and per-row memcmp as leg H.
  constexpr int kIR = 8;
  const std::size_t i_slots = static_cast<std::size_t>(kIR) * n_used;
  Expect(n_used > 4, "leg I fan-outs exceed one record");
  std::vector<int> idi(i_slots);
  for (int t = 0; t < kIR; ++t) {
    for (int s = 0; s < n_used; ++s) {
      idi[static_cast<std::size_t>(t) * n_used + s] = s % n_expert;
    }
  }
  std::vector<float> wi(i_slots);
  for (std::size_t i = 0; i < i_slots; ++i) {
    wi[i] = 1.0F / static_cast<float>(static_cast<int>(i % static_cast<std::size_t>(n_used)) + 1);
  }
  const float sigi[kIR] = {0.9F, 0.5F, 0.1F, 0.3F, 0.7F, 0.2F, 0.6F, 0.4F};
  DeviceBuf d_xgi(static_cast<std::size_t>(hidden) * sizeof(float));
  DeviceBuf d_idgi(static_cast<std::size_t>(n_used) * sizeof(int));
  DeviceBuf d_wrgi(static_cast<std::size_t>(n_used) * sizeof(float));
  DeviceBuf d_siggi(sizeof(float));
  DeviceBuf d_zgi(sizeof(int));
  HIP_REQUIRE(hipMemset(d_zgi.p, 0, sizeof(int)));
  DeviceBuf d_id_i(i_slots * sizeof(int));
  Upload(d_id_i, idi.data(), idi.size() * sizeof(int));
  DeviceBuf d_w_i(i_slots * sizeof(float));
  Upload(d_w_i, wi.data(), wi.size() * sizeof(float));
  DeviceBuf d_sig_i(static_cast<std::size_t>(kIR) * sizeof(float));
  Upload(d_sig_i, sigi, sizeof(sigi));
  DeviceBuf d_ref_yi(i_slots * hidden * sizeof(float));
  DeviceBuf d_ref_si(i_slots * hidden * sizeof(float));
  DeviceBuf d_ref_outi(i_slots * hidden * sizeof(float));
  DeviceBuf d_all_outi(i_slots * hidden * sizeof(float));
  for (int t = 0; t < kIR; ++t) {
    const std::size_t xo = static_cast<std::size_t>(t) * hidden;
    HIP_REQUIRE(hipMemcpy(d_xgi.p, xm.data() + xo,
                          static_cast<std::size_t>(hidden) * sizeof(float),
                          hipMemcpyHostToDevice));
    HIP_REQUIRE(hipMemcpy(d_idgi.p, idi.data() + static_cast<std::size_t>(t) * n_used,
                          static_cast<std::size_t>(n_used) * sizeof(int),
                          hipMemcpyHostToDevice));
    HIP_REQUIRE(hipMemcpy(d_wrgi.p, wi.data() + static_cast<std::size_t>(t) * n_used,
                          static_cast<std::size_t>(n_used) * sizeof(float),
                          hipMemcpyHostToDevice));
    HIP_REQUIRE(hipMemcpy(d_siggi.p, sigi + t, sizeof(float),
                          hipMemcpyHostToDevice));
    Expect(superroc_fp4_expert_ffn_dev(qtype, d_gate.p, d_up.p, d_down.p, d_xgi.f32(),
                                   d_idgi.i32(), d_wrgi.f32(), n_used, n_expert,
                                   hidden, ff, 0.0F, d_ref_yi.f32() + xo,
                                   nullptr) == SUPERROC_FP4_OK,
           "leg I ref routed");
    Expect(superroc_fp4_expert_ffn_dev_xq(qtype_sh, d_shgate.p, d_shup.p, d_shdown.p,
                                      d_xgi.f32(), d_zgi.i32(), d_siggi.f32(), 1, 1,
                                      hidden, ff_s, 0.0F, d_ref_si.f32() + xo,
                                      nullptr) == SUPERROC_FP4_OK,
           "leg I ref shexp");
  }
  Expect(LaunchQwenMoeCombine(d_ref_yi.f32(), d_ref_si.f32(), hidden, kIR,
                              d_ref_outi.f32(), nullptr) == 0,
         "leg I ref combine");
  Expect(superroc_fp4_expert_ffn_dev_batch_all(
             qtype, d_gate.p, d_up.p, d_down.p, d_shgate.p, d_shup.p, d_shdown.p,
             d_xm.f32(), d_id_i.i32(), d_w_i.f32(), d_sig_i.f32(), kIR, n_used,
             n_expert, hidden, ff, ff_s, 0.0F, d_all_outi.f32(),
             nullptr) == SUPERROC_FP4_OK,
         "leg I batch all");
  HIP_REQUIRE(hipDeviceSynchronize());
  Expect(superroc_fp4_moe_dev_check() == SUPERROC_FP4_OK, "leg I status");
  std::vector<float> refi(i_slots * hidden);
  std::vector<float> alli(refi.size());
  Fetch(refi, d_ref_outi);
  Fetch(alli, d_all_outi);
  for (int t = 0; t < kIR; ++t) {
    const std::size_t o = static_cast<std::size_t>(t) * hidden;
    int bad = -1;
    for (int h = 0; h < hidden; ++h) {
      if (std::memcmp(refi.data() + o + h, alli.data() + o + h, sizeof(float)) != 0) {
        bad = h;
        break;
      }
    }
    if (bad >= 0) {
      std::cerr << "moebatch leg I row=" << t << " first_diff=" << bad
                << " ref=" << refi[o + static_cast<std::size_t>(bad)]
                << " all=" << alli[o + static_cast<std::size_t>(bad)] << "\n";
    }
    Expect(std::memcmp(refi.data() + o, alli.data() + o,
                       static_cast<std::size_t>(hidden) * sizeof(float)) == 0,
           "leg I bit-identical");
  }
  std::cout << "moebatch leg I OK\n";

  std::cout << "KAT_FFN PASS\n";
  return 0;
}
