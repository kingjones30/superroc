// KDG SPEC1 step 4: KAT device-dispatch bit-identity gate.
// On the real embedded MTP block (blk.40: the Q8_0 experts the production
// step executes), runs the OLD host-loop dispatch (static reference below,
// the algorithm LaunchKatMtpMoeFfn used before SPEC1 step 2) and the NEW
// device-dispatch path (LaunchKatMtpMoeFfn) on 8 random hidden vectors and
// asserts bit-identical outputs (memcmp).
//
// NOTE on the fixture: SPEC1 names blk.0, but blk.0's experts are FP4
// (type 101), which the Q8 GEMV this step runs refuses ("KAT-MTP: expert
// matrix must be Q8_0 or Q8_K"). blk.40 is the one real KAT layer slice
// this step executes, so the test binds it.
#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/hip/mtp/kat_moe_ffn.hpp"
#include "src/models/qwen/moe/forward_moe.hpp"
#include "src/models/qwen/moe/qwen_moe_ffn.hpp"
#include "src/models/qwen/moe/qwen_moe_route.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

__global__ void RefSwigluKernel(const float* __restrict__ gate,
                                const float* __restrict__ up,
                                float* __restrict__ act, int n) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= n) {
    return;
  }
  const float g = gate[index];
  act[index] = (g / (1.0F + expf(-g))) * up[index];
}

__global__ void RefAxpyKernel(const float* __restrict__ x, float scale,
                              float* __restrict__ out, int n) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= n) {
    return;
  }
  out[index] += scale * x[index];
}

__global__ void RefScaleAddKernel(const float* __restrict__ x,
                                 const float* __restrict__ scale,
                                 float* __restrict__ out, int n) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= n) {
    return;
  }
  out[index] += scale[0] * x[index];
}

void RefSwiglu(const float* gate, const float* up, float* act, int n) {
  RefSwigluKernel<<<(n + 255) / 256, 256, 0, nullptr>>>(gate, up, act, n);
  HIP_REQUIRE(hipGetLastError());
}

void RefAxpy(const float* x, float scale, float* out, int n) {
  RefAxpyKernel<<<(n + 255) / 256, 256, 0, nullptr>>>(x, scale, out, n);
  HIP_REQUIRE(hipGetLastError());
}

void RefScaleAdd(const float* x, const float* scale, float* out, int n) {
  RefScaleAddKernel<<<(n + 255) / 256, 256, 0, nullptr>>>(x, scale, out, n);
  HIP_REQUIRE(hipGetLastError());
}

// OLD host-loop path (pre-SPEC1-step-2 algorithm): route on device, read
// ids/weights back, loop slots on the host. Uses the same public GEMM
// (LaunchBatchedQuantGEMMFp32, batch 1) the old GemvQ8 called.
void ReferenceMoe(const gufo::models::QwenLayerWeights& layer,
                  const gufo::core::ModelConfig& config, const float* d_x,
                  float* d_out, const float* d_router, const float* d_gatevec,
                  float* d_logits, int* d_ids, float* d_w, float* d_sig,
                  float* d_gate, float* d_up, float* d_act, float* d_h,
                  hipStream_t stream) {
  const int n_expert = static_cast<int>(config.expert_count);
  const int n_used = static_cast<int>(config.expert_used_count);
  const int hidden = static_cast<int>(config.hidden_size);
  const int ff = static_cast<int>(config.expert_feed_forward_length);
  const int ff_s = static_cast<int>(config.expert_shared_feed_forward_length);
  Expect(LaunchQwenMoeRoute(d_x, d_router, hidden, n_expert, n_used,
                                  d_logits, d_ids, d_w, stream) == 0,
         "ref route launch");
  std::vector<int> host_ids(static_cast<std::size_t>(n_used));
  std::vector<float> host_w(static_cast<std::size_t>(n_used));
  HIP_REQUIRE(hipMemcpyAsync(host_ids.data(), d_ids,
                             host_ids.size() * sizeof(int),
                             hipMemcpyDeviceToHost, stream));
  HIP_REQUIRE(hipMemcpyAsync(host_w.data(), d_w, host_w.size() * sizeof(float),
                             hipMemcpyDeviceToHost, stream));
  HIP_REQUIRE(hipStreamSynchronize(stream));
  HIP_REQUIRE(hipMemsetAsync(d_out, 0, static_cast<std::size_t>(hidden) * sizeof(float), stream));
  for (int slot = 0; slot < n_used; ++slot) {
    Expect(host_ids[static_cast<std::size_t>(slot)] >= 0 &&
               host_ids[static_cast<std::size_t>(slot)] < n_expert,
           "ref id range");
    const auto gate_w = gufo::models::KatExpertSlice(
        layer.ffn_gate_exps, host_ids[static_cast<std::size_t>(slot)], ff,
        hidden);
    const auto up_w = gufo::models::KatExpertSlice(
        layer.ffn_up_exps, host_ids[static_cast<std::size_t>(slot)], ff,
        hidden);
    const auto down_w = gufo::models::KatExpertSlice(
        layer.ffn_down_exps, host_ids[static_cast<std::size_t>(slot)], hidden,
        ff);
    gufo::hip::LaunchBatchedQuantGEMMFp32(gate_w.type, gate_w.data, d_x,
                                          d_gate, 1, ff, hidden, stream);
    gufo::hip::LaunchBatchedQuantGEMMFp32(up_w.type, up_w.data, d_x, d_up, 1,
                                          ff, hidden, stream);
    RefSwiglu(d_gate, d_up, d_act, ff);
    gufo::hip::LaunchBatchedQuantGEMMFp32(down_w.type, down_w.data, d_act,
                                          d_h, 1, hidden, ff, stream);
    RefAxpy(d_h, host_w[static_cast<std::size_t>(slot)], d_out, hidden);
  }
  if (!layer.ffn_up_shexp.empty()) {
    Expect(LaunchQwenMoeShexpSigmoid(d_x, d_gatevec, hidden, 1, d_sig,
                                           stream) == 0,
           "ref sigmoid launch");
    const auto gate_w =
        gufo::models::KatExpertSlice(layer.ffn_gate_shexp, -1, ff_s, hidden);
    const auto up_w =
        gufo::models::KatExpertSlice(layer.ffn_up_shexp, -1, ff_s, hidden);
    const auto down_w =
        gufo::models::KatExpertSlice(layer.ffn_down_shexp, -1, hidden, ff_s);
    gufo::hip::LaunchBatchedQuantGEMMFp32(gate_w.type, gate_w.data, d_x,
                                          d_gate, 1, ff_s, hidden, stream);
    gufo::hip::LaunchBatchedQuantGEMMFp32(up_w.type, up_w.data, d_x, d_up, 1,
                                          ff_s, hidden, stream);
    RefSwiglu(d_gate, d_up, d_act, ff_s);
    gufo::hip::LaunchBatchedQuantGEMMFp32(down_w.type, down_w.data, d_act,
                                          d_h, 1, hidden, ff_s, stream);
    RefScaleAdd(d_h, d_sig, d_out, hidden);
  }
  HIP_REQUIRE(hipStreamSynchronize(stream));
}

}  // namespace

int main() {
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
  Expect(weights->mtp.has_value(), "mtp block bound");
  // The MTP block (blk.40) carries the Q8_0 experts the production step
  // executes; blk.0's experts are FP4 and cannot run on this path.
  gufo::models::QwenLayerWeights layer = *weights->mtp;
  const int hidden = static_cast<int>(cfg.hidden_size);
  const int n_expert = static_cast<int>(cfg.expert_count);
  const int n_used = static_cast<int>(cfg.expert_used_count);
  const int ff = static_cast<int>(cfg.expert_feed_forward_length);
  const int ff_s = static_cast<int>(cfg.expert_shared_feed_forward_length);
  Expect(hidden == 2048 && n_expert == 256 && n_used == 8 && ff == 512 &&
             ff_s == 512,
         "kat shape");
  Expect(layer.ffn_gate_exps.type == gufo::core::GgmlType::kQ8_0 &&
             layer.ffn_up_exps.type == gufo::core::GgmlType::kQ8_0 &&
             layer.ffn_down_exps.type == gufo::core::GgmlType::kQ8_0,
         "mtp experts q8");
  Expect(!layer.ffn_up_shexp.empty(), "mtp shexp bound");
  std::cout << "KAT_DISPATCH shape hidden=" << hidden << " experts="
            << n_expert << " topk=" << n_used << " ff=" << ff << "\n";

  const std::size_t rb_k = gufo::quant::QuantizedRowBytes(
      layer.ffn_gate_exps.type, static_cast<std::size_t>(hidden));
  const std::size_t rb_ff = gufo::quant::QuantizedRowBytes(
      layer.ffn_down_exps.type, static_cast<std::size_t>(ff));
  Expect(rb_k > 0 && rb_ff > 0, "row bytes");
  DeviceBuf d_gate(static_cast<std::size_t>(n_expert) * ff * rb_k);
  DeviceBuf d_up(static_cast<std::size_t>(n_expert) * ff * rb_k);
  DeviceBuf d_down(static_cast<std::size_t>(n_expert) * hidden * rb_ff);
  HIP_REQUIRE(hipMemcpy(d_gate.p, layer.ffn_gate_exps.data,
                        static_cast<std::size_t>(n_expert) * ff * rb_k,
                        hipMemcpyHostToDevice));
  HIP_REQUIRE(hipMemcpy(d_up.p, layer.ffn_up_exps.data,
                        static_cast<std::size_t>(n_expert) * ff * rb_k,
                        hipMemcpyHostToDevice));
  HIP_REQUIRE(hipMemcpy(d_down.p, layer.ffn_down_exps.data,
                        static_cast<std::size_t>(n_expert) * hidden * rb_ff,
                        hipMemcpyHostToDevice));
  const std::size_t sh_rb_k = gufo::quant::QuantizedRowBytes(
      layer.ffn_gate_shexp.type, static_cast<std::size_t>(hidden));
  const std::size_t sh_rb_ff = gufo::quant::QuantizedRowBytes(
      layer.ffn_down_shexp.type, static_cast<std::size_t>(ff_s));
  Expect(sh_rb_k > 0 && sh_rb_ff > 0, "shexp row bytes");
  DeviceBuf d_shgate(static_cast<std::size_t>(ff_s) * sh_rb_k);
  DeviceBuf d_shup(static_cast<std::size_t>(ff_s) * sh_rb_k);
  DeviceBuf d_shdown(static_cast<std::size_t>(hidden) * sh_rb_ff);
  HIP_REQUIRE(hipMemcpy(d_shgate.p, layer.ffn_gate_shexp.data,
                        static_cast<std::size_t>(ff_s) * sh_rb_k,
                        hipMemcpyHostToDevice));
  HIP_REQUIRE(hipMemcpy(d_shup.p, layer.ffn_up_shexp.data,
                        static_cast<std::size_t>(ff_s) * sh_rb_k,
                        hipMemcpyHostToDevice));
  HIP_REQUIRE(hipMemcpy(d_shdown.p, layer.ffn_down_shexp.data,
                        static_cast<std::size_t>(hidden) * sh_rb_ff,
                        hipMemcpyHostToDevice));
  DeviceBuf d_router(static_cast<std::size_t>(n_expert) * hidden *
                     sizeof(float));
  HIP_REQUIRE(hipMemcpy(d_router.p, layer.ffn_gate_inp.data,
                        static_cast<std::size_t>(n_expert) * hidden *
                            sizeof(float),
                        hipMemcpyHostToDevice));
  DeviceBuf d_gatevec(static_cast<std::size_t>(hidden) * sizeof(float));
  HIP_REQUIRE(hipMemcpy(d_gatevec.p, layer.ffn_gate_inp_shexp.data,
                        static_cast<std::size_t>(hidden) * sizeof(float),
                        hipMemcpyHostToDevice));

  // Rebind the layer at device addresses (production binds remapped device
  // tensors; only the data pointers move, the layout math is unchanged).
  layer.ffn_gate_exps.data = d_gate.p;
  layer.ffn_up_exps.data = d_up.p;
  layer.ffn_down_exps.data = d_down.p;
  layer.ffn_gate_shexp.data = d_shgate.p;
  layer.ffn_up_shexp.data = d_shup.p;
  layer.ffn_down_shexp.data = d_shdown.p;
  layer.ffn_gate_inp.data = d_router.p;
  layer.ffn_gate_inp_shexp.data = d_gatevec.p;

  DeviceBuf d_x(static_cast<std::size_t>(hidden) * sizeof(float));
  DeviceBuf d_logits(static_cast<std::size_t>(n_expert) * sizeof(float));
  DeviceBuf d_ids(static_cast<std::size_t>(n_used) * sizeof(int));
  DeviceBuf d_w(static_cast<std::size_t>(n_used) * sizeof(float));
  DeviceBuf d_sig(sizeof(float));
  DeviceBuf d_sgate(static_cast<std::size_t>(ff) * sizeof(float));
  DeviceBuf d_sup(static_cast<std::size_t>(ff) * sizeof(float));
  DeviceBuf d_sact(static_cast<std::size_t>(ff) * sizeof(float));
  DeviceBuf d_sh(static_cast<std::size_t>(hidden) * sizeof(float));
  DeviceBuf d_ref(static_cast<std::size_t>(hidden) * sizeof(float));
  DeviceBuf d_new(static_cast<std::size_t>(hidden) * sizeof(float));

  gufo::hip::KatMtpMoeScratch scratch;
  scratch.logits = d_logits.f32();
  scratch.ids = d_ids.i32();
  scratch.weights = d_w.f32();
  scratch.sig = d_sig.f32();
  scratch.gate = d_sgate.f32();
  scratch.up = d_sup.f32();
  scratch.act = d_sact.f32();
  scratch.hidden = d_sh.f32();

  std::vector<float> x(static_cast<std::size_t>(hidden));
  std::vector<float> ref(static_cast<std::size_t>(hidden));
  std::vector<float> got(static_cast<std::size_t>(hidden));
  int identical = 0;
  for (int v = 0; v < 8; ++v) {
    Fill(x, 0xD15u + static_cast<std::uint32_t>(v));
    HIP_REQUIRE(hipMemcpy(d_x.p, x.data(), x.size() * sizeof(float),
                          hipMemcpyHostToDevice));
    ReferenceMoe(layer, cfg, d_x.f32(), d_ref.f32(), d_router.f32(),
                 d_gatevec.f32(), d_logits.f32(), d_ids.i32(), d_w.f32(),
                 d_sig.f32(), d_sgate.f32(), d_sup.f32(), d_sact.f32(),
                 d_sh.f32(), nullptr);
    HIP_REQUIRE(hipMemcpy(ref.data(), d_ref.p, ref.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
    gufo::hip::LaunchKatMtpMoeFfn(layer, cfg, d_x.f32(), d_new.f32(), scratch,
                                  nullptr);
    HIP_REQUIRE(hipDeviceSynchronize());
    HIP_REQUIRE(hipMemcpy(got.data(), d_new.p, got.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
    const bool same =
        std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) == 0;
    float max_abs = 0.0F;
    if (!same) {
      for (std::size_t i = 0; i < ref.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(ref[i] - got[i]));
      }
    }
    std::cout << "KAT_DISPATCH vec=" << v
              << " bit_identical=" << (same ? 1 : 0);
    if (!same) {
      std::cout << " max_abs=" << max_abs;
    }
    std::cout << "\n";
    identical += same ? 1 : 0;
  }
  Expect(identical == 8, "dispatch bit-identity 8/8");
  std::cout << "KAT_DISPATCH identical=8/8\n";
  return 0;
}
