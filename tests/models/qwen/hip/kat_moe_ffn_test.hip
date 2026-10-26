// KAT S3/S4 G1: the GPU MoE FFN against the CPU oracle on the real KAT GGUF.
//   A: routed expert slices e in {0, 8, 255} through gufo_fp4_expert_ffn_dev
//   B: the sigmoid-gated shared expert (QwenMoeShexpSigmoid + expert_ffn_dev)
//   C: the full single-token decode MoE (fast route + experts + shexp +
//      combine) vs ForwardMoe, incl. the router id set
//   D: the batched route (prefill head) over 17 tokens vs per-token CPU
//   E: the batched prefill MoE sequence over 5 tokens vs per-token ForwardMoe
// The GPU legs quantise the activation to Q8_1 (integer dot), the oracle dots
// in float, so the gate is RV24's Q8-activation leg: cosine >= 0.9999.
#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/rocm_fp4/gufo_fp4_gemv_q8.hpp"
#include "src/core/rocm_fp4/gufo_fp4_quant.hpp"
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
  Expect(LaunchQwenMoeRouteBatch(nullptr, nullptr, 32, 32, 1, 8, nullptr,
                                 nullptr, nullptr, nullptr) == 1,
         "batch route null rejected");

  const char* path_env = std::getenv("KAT_GGUF");
  const std::string path =
      path_env != nullptr
          ? path_env
          : "/path/to/models/swap/kat-coder/"
            "KAT-Coder-V2.5-Dev-Q4_0_ROCMFP4_STRIX-MTP.gguf";
  if (!std::filesystem::exists(path)) {
    std::cout << "SKIP: KAT GGUF not found: " << path << " (set KAT_GGUF)\n";
    return 77;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  Expect(reader != nullptr, "open: " + error);
  auto weights = gufo::models::QwenModelWeights::LoadFromGguf(*reader, &error);
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
  Expect(qtype == GUFO_FP4_Q4_0_ROCMFP4_FAST ||
             qtype == GUFO_FP4_Q4_0_ROCMFP4,
         "expert type fp4");
  Expect(layer0.ffn_up_exps.type == layer0.ffn_gate_exps.type &&
             layer0.ffn_down_exps.type == layer0.ffn_gate_exps.type,
         "expert types agree");
  const int qtype_sh = static_cast<int>(layer0.ffn_gate_shexp.type);
  Expect(layer0.ffn_gate_inp.type == gufo::core::GgmlType::kF32 &&
             layer0.ffn_gate_inp_shexp.type == gufo::core::GgmlType::kF32,
         "router + shexp gate f32");

  std::vector<float> x(static_cast<std::size_t>(hidden));
  Fill(x, 0x6A71u);

  // Device copies of the bound tensors (the host bind is mmapped read-only;
  // the expert kernels want device pointers, as in the remapped executor).
  // Byte counts follow the binder's convention (forward_moe.cpp ExpertSlice):
  // expert stride = n_rows * QuantizedRowBytes(type, k).
  const std::size_t rb_k =
      gufo::quant::QuantizedRowBytes(layer0.ffn_gate_exps.type,
                                     static_cast<std::size_t>(hidden));
  const std::size_t rb_ff =
      gufo::quant::QuantizedRowBytes(layer0.ffn_down_exps.type,
                                     static_cast<std::size_t>(ff));
  const std::size_t sh_rb_k =
      gufo::quant::QuantizedRowBytes(layer0.ffn_gate_shexp.type,
                                     static_cast<std::size_t>(hidden));
  const std::size_t sh_rb_ff =
      gufo::quant::QuantizedRowBytes(layer0.ffn_down_shexp.type,
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
    Expect(gufo_fp4_expert_ffn_dev(qtype, d_gate.p, d_up.p, d_down.p, d_x.f32(),
                                   d_ids.i32(), d_w.f32(), 1, n_expert, hidden,
                                   ff, 0.0F, d_y.f32(), nullptr) == GUFO_FP4_OK,
           "slice launch");
    HIP_REQUIRE(hipDeviceSynchronize());
    Fetch(got, d_y);
    gufo::models::KatExpertSwiglu(layer0.ffn_gate_exps, layer0.ffn_up_exps,
                                  layer0.ffn_down_exps, e, x, hidden, ff,
                                  gscratch, uscratch, ascratch, cpu);
    const double cosine = Cosine(got, cpu);
    std::cout << "KAT_FFN slice e=" << e << " cosine=" << cosine << "\n";
    Expect(cosine >= 0.9999, "slice cosine");
  }
  Expect(gufo_fp4_moe_dev_check() == GUFO_FP4_OK, "slice status");

  // --- B: shared expert + sigmoid gate --------------------------------------
  Expect(LaunchQwenMoeShexpSigmoid(d_x.f32(), d_gatevec.f32(), hidden, 1,
                                   d_sig.f32(), nullptr) == 0,
         "sigmoid launch");
  Expect(gufo_fp4_expert_ffn_dev(qtype_sh, d_shgate.p, d_shup.p, d_shdown.p,
                                 d_x.f32(), d_zero.i32(), d_sig.f32(), 1, 1,
                                 hidden, ff_s, 0.0F, d_y.f32(),
                                 nullptr) == GUFO_FP4_OK,
         "shexp launch");
  HIP_REQUIRE(hipDeviceSynchronize());
  Fetch(got, d_y);
  std::vector<float> sig_host(1);
  Fetch(sig_host, d_sig);
  gufo::models::KatExpertSwiglu(layer0.ffn_gate_shexp, layer0.ffn_up_shexp,
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
  Expect(gufo_fp4_moe_dev_check() == GUFO_FP4_OK, "shexp status");

  // --- C: full single-token decode MoE --------------------------------------
  Expect(LaunchQwenMoeRoute(d_x.f32(), d_router.f32(), hidden, n_expert,
                            n_used, d_logits.f32(), d_ids.i32(), d_w.f32(),
                            nullptr) == 0,
         "route launch");
  Expect(gufo_fp4_expert_ffn_dev(qtype, d_gate.p, d_up.p, d_down.p, d_x.f32(),
                                 d_ids.i32(), d_w.f32(), n_used, n_expert,
                                 hidden, ff, 0.0F, d_y.f32(),
                                 nullptr) == GUFO_FP4_OK,
         "experts launch");
  DeviceBuf d_sy(static_cast<std::size_t>(hidden) * sizeof(float));
  Expect(LaunchQwenMoeShexpSigmoid(d_x.f32(), d_gatevec.f32(), hidden, 1,
                                   d_sig.f32(), nullptr) == 0,
         "sigmoid relaunch");
  Expect(gufo_fp4_expert_ffn_dev(qtype_sh, d_shgate.p, d_shup.p, d_shdown.p,
                                 d_x.f32(), d_zero.i32(), d_sig.f32(), 1, 1,
                                 hidden, ff_s, 0.0F, d_sy.f32(),
                                 nullptr) == GUFO_FP4_OK,
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
  gufo::models::ForwardMoe(x, layer0, cfg, moe_logits, moe_probs, gscratch,
                           uscratch, ascratch, eh, cpu);
  const double moe_cos = Cosine(got, cpu);
  std::vector<int> cpu_ids(n_used);
  std::vector<float> cpu_w(n_used);
  Expect(gufo::models::KatRouteSoftmaxTopK(moe_logits, n_used, cpu_ids, cpu_w),
         "cpu route");
  std::vector<int> gpu_ids(static_cast<std::size_t>(n_used));
  FetchIds(gpu_ids, d_ids);
  std::sort(cpu_ids.begin(), cpu_ids.end());
  std::sort(gpu_ids.begin(), gpu_ids.end());
  Expect(cpu_ids == gpu_ids, "decode id set");
  std::cout << "KAT_FFN decode_moe cosine=" << moe_cos << " ids_match=1\n";
  Expect(moe_cos >= 0.9999, "decode moe cosine");
  Expect(gufo_fp4_moe_dev_check() == GUFO_FP4_OK, "decode status");

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
    gufo::models::TensorGEMV(layer0.ffn_gate_inp, xt,
                             static_cast<std::size_t>(n_expert), hidden, lg);
    std::vector<int> ci(n_used);
    std::vector<float> cw(n_used);
    Expect(gufo::models::KatRouteSoftmaxTopK(lg, n_used, ci, cw), "cpu route t");
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
    Expect(gufo_fp4_expert_ffn_m(
               qtype, d_gate.p, d_up.p, d_down.p,
               d_xb.f32() + static_cast<std::size_t>(t0) * hidden,
               d_idsb.i32() + static_cast<std::size_t>(t0) * n_used,
               d_wb.f32() + static_cast<std::size_t>(t0) * n_used, m, n_used,
               n_expert, hidden, ff, 0.0F,
               d_moey.f32() + static_cast<std::size_t>(t0) * hidden,
               nullptr) == GUFO_FP4_OK,
           "prefill experts");
    Expect(gufo_fp4_expert_ffn_shared_m(
               qtype_sh, d_shgate.p, d_shup.p, d_shdown.p,
               d_xb.f32() + static_cast<std::size_t>(t0) * hidden,
               d_zerob.i32() + t0, d_sigb.f32() + t0, m, 1, 1, hidden, ff_s,
               0.0F, d_shey.f32() + static_cast<std::size_t>(t0) * hidden,
               nullptr) == GUFO_FP4_OK,
           "prefill shexp");
  }
  Expect(gufo_fp4_expert_ffn_m_check() == GUFO_FP4_OK, "prefill status");
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
    gufo::models::ForwardMoe(xt, layer0, cfg, moe_logits, moe_probs, gscratch,
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
    Expect(gufo_fp4_expert_ffn_m(
               qtype, d_gate.p, d_up.p, d_down.p, d_xg.f32() + off,
               d_idg.i32() + static_cast<std::size_t>(t0) * n_used,
               d_wg.f32() + static_cast<std::size_t>(t0) * n_used, m, n_used,
               n_expert, hidden, ff, 0.0F, d_ref_y.f32() + off,
               nullptr) == GUFO_FP4_OK,
           "m4 experts");
    Expect(gufo_fp4_expert_ffn_shared_m(
               qtype_sh, d_shgate.p, d_shup.p, d_shdown.p, d_xg.f32() + off,
               d_zerg.i32() + t0, d_sigg.f32() + t0, m, 1, 1, hidden, ff_s, 0.0F,
               d_ref_s.f32() + off, nullptr) == GUFO_FP4_OK,
           "m4 shared");
  }
  Expect(gufo_fp4_expert_ffn_m_check() == GUFO_FP4_OK, "m4 status");
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

  std::cout << "KAT_FFN PASS\n";
  return 0;
}
