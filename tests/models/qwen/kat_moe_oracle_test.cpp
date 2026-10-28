#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/ggml_gemm.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/moe/forward_moe.hpp"
#include "src/models/qwen/ssm.hpp"

namespace {

void Expect(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL " << msg << "\n";
    std::exit(1);
  }
}

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
    const int mag = static_cast<int>(s % 2001) - 1000;
    x = static_cast<float>(mag) / 1000.0F;
  }
}

float RowDot(const superroc::models::QwenTensorRef& tensor, std::size_t row,
             std::size_t k, const std::vector<float>& x) {
  // KAT: QuantizedRowBytes returns 0 for kF32 (it sizes quantized blocks only;
  // the F32 stride lives in EncodedSizeBytes). A 0 stride aliased every row to
  // row 0, which is why router row0 passed bit-exact and row255 failed.
  const std::size_t row_bytes =
      tensor.type == superroc::core::GgmlType::kF32
          ? k * sizeof(float)
          : superroc::quant::QuantizedRowBytes(tensor.type, k);
  const auto* ptr =
      static_cast<const std::uint8_t*>(tensor.data) + row * row_bytes;
  if (tensor.type == superroc::core::GgmlType::kF32) {
    const auto* f = reinterpret_cast<const float*>(ptr);
    float sum = 0.0F;
    for (std::size_t i = 0; i < k; ++i) {
      sum += f[i] * x[i];
    }
    return sum;
  }
  return superroc::quant::Dot(tensor.type, ptr, x, k);
}

int TestBitonicDesc(const float* x, int* dst, int* work_a, int* work_b,
                    int ncols) {
  if (ncols <= 0 || ncols > 1024) {
    return 1;
  }
  int pad = 1;
  while (pad < ncols) {
    pad <<= 1;
  }
  for (int i = 0; i < pad; ++i) {
    work_a[i] = i;
  }
  for (int k = 2; k <= pad; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int col = 0; col < pad; ++col) {
        work_b[col] = work_a[col];
      }
      for (int col = 0; col < pad; ++col) {
        const int ixj = col ^ j;
        if (ixj > col) {
          int do_swap = 0;
          if ((col & k) == 0) {
            if (work_a[col] >= ncols ||
                (work_a[ixj] < ncols && x[work_a[col]] < x[work_a[ixj]])) {
              do_swap = 1;
            }
          } else if (work_a[ixj] >= ncols ||
                     (work_a[col] < ncols && x[work_a[col]] > x[work_a[ixj]])) {
            do_swap = 1;
          }
          if (do_swap) {
            work_b[col] = work_a[ixj];
            work_b[ixj] = work_a[col];
          }
        }
      }
      for (int col = 0; col < pad; ++col) {
        work_a[col] = work_b[col];
      }
    }
  }
  for (int i = 0; i < ncols; ++i) {
    dst[i] = work_a[i];
  }
  return 0;
}

void Softmax(const std::vector<float>& logits, std::vector<float>& probs) {
  float max_val = logits[0];
  for (float v : logits) {
    max_val = std::max(max_val, v);
  }
  double sum = 0.0;
  probs.resize(logits.size());
  for (float v : logits) {
    sum += std::exp(static_cast<double>(v - max_val));
  }
  const double inv = 1.0 / sum;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    probs[i] = static_cast<float>(
        std::exp(static_cast<double>(logits[i] - max_val)) * inv);
  }
}

void CheckRoute(const std::vector<float>& logits, const char* name) {
  const int n = static_cast<int>(logits.size());
  const int n_used = 8;
  std::vector<int> ids(n_used);
  std::vector<float> weights(n_used);
  Expect(superroc::models::KatRouteSoftmaxTopK(logits, n_used, ids, weights),
         std::string("route ") + name);
  std::vector<float> probs;
  Softmax(logits, probs);
  int order[256];
  int wa[512];
  int wb[512];
  Expect(TestBitonicDesc(probs.data(), order, wa, wb, n) == 0,
         std::string("bitonic ") + name);
  // Fork norm_w. raw_sum is the old unnormalised top-k mass. A router that
  // still returns the raw probability fails "renorm weight" and, when that
  // mass is below 0.5, "not unnormalised".
  float raw_sum = 0.0F;
  for (int i = 0; i < n_used; ++i) {
    raw_sum += probs[static_cast<std::size_t>(order[i])];
  }
  const float den = raw_sum < 0x1p-14F ? 0x1p-14F : raw_sum;
  float sum_w = 0.0F;
  for (int i = 0; i < n_used; ++i) {
    Expect(ids[static_cast<std::size_t>(i)] == order[i],
           std::string("id ") + name + " " + std::to_string(i));
    const float raw = probs[static_cast<std::size_t>(order[i])];
    Expect(
        std::fabs(weights[static_cast<std::size_t>(i)] - (raw / den)) < 1.0e-6F,
        std::string("renorm weight ") + name);
    if (raw_sum < 0.5F) {
      Expect(std::fabs(weights[static_cast<std::size_t>(i)] - raw) > 1.0e-4F,
             std::string("not unnormalised ") + name);
    }
    sum_w += weights[static_cast<std::size_t>(i)];
  }
  const float cut = probs[static_cast<std::size_t>(order[n_used - 1])];
  for (int i = n_used; i < n; ++i) {
    Expect(probs[static_cast<std::size_t>(order[i])] <= cut + 1.0e-7F,
           std::string("top set ") + name);
  }
  Expect(std::fabs(sum_w - (raw_sum / den)) < 1.0e-5F,
         std::string("weight sum ") + name);
  if (logits.front() == logits.back()) {
    // Uniform softmax: each renormed weight is 1/n_used and the eight sum
    // to 1. The old code expected n_used/n (8/256 = 0.03125) and fails here.
    Expect(std::fabs(sum_w - 1.0F) < 1.0e-5F,
           std::string("uniform renorm ") + name);
    Expect(std::fabs(raw_sum - static_cast<float>(n_used) /
                                   static_cast<float>(n)) < 1.0e-5F,
           std::string("uniform raw mass ") + name);
  }
}

void RefExpert(const superroc::models::QwenTensorRef& gate,
               const superroc::models::QwenTensorRef& up,
               const superroc::models::QwenTensorRef& down, int expert,
               const std::vector<float>& x, std::size_t hidden, std::size_t ff,
               std::vector<float>& hidden_out) {
  const std::size_t up_bytes =
      superroc::quant::QuantizedRowBytes(gate.type, hidden);
  const std::size_t down_bytes =
      superroc::quant::QuantizedRowBytes(down.type, ff);
  const auto* gate_base = static_cast<const std::uint8_t*>(gate.data);
  const auto* up_base = static_cast<const std::uint8_t*>(up.data);
  const auto* down_base = static_cast<const std::uint8_t*>(down.data);
  const std::size_t expert_up = static_cast<std::size_t>(std::max(expert, 0));
  if (expert < 0) {
    // Shared expert is a single 2D matrix at the base pointer.
  }
  const std::size_t up_off = (expert < 0 ? 0 : expert_up * ff) * up_bytes;
  const std::size_t down_off =
      (expert < 0 ? 0 : expert_up * hidden) * down_bytes;
  std::vector<float> act(ff);
  for (std::size_t r = 0; r < ff; ++r) {
    const float g = superroc::quant::Dot(
        gate.type, gate_base + up_off + r * up_bytes, x, hidden);
    const float u = superroc::quant::Dot(
        up.type, up_base + up_off + r * up_bytes, x, hidden);
    act[r] = (g / (1.0F + std::exp(-g))) * u;
  }
  hidden_out.assign(hidden, 0.0F);
  for (std::size_t r = 0; r < hidden; ++r) {
    hidden_out[r] = superroc::quant::Dot(
        down.type, down_base + down_off + r * down_bytes, act, ff);
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

}  // namespace

int main() {
  const char* path_env = std::getenv("KAT_GGUF");
  const std::string path =
      path_env != nullptr ? path_env
                          : "/path/to/models/swap/kat-coder/"
                            "KAT-Coder-V2.5-Dev-Q4_0_ROCMFP4_STRIX-MTP.gguf";
  if (!std::filesystem::exists(path)) {
    std::cout << "SKIP: KAT GGUF not found: " << path << " (set KAT_GGUF)\n";
    return 77;
  }
  std::string error;
  auto reader = superroc::core::GgufReader::OpenFile(path, &error);
  Expect(reader != nullptr, "open: " + error);
  auto weights =
      superroc::models::QwenModelWeights::LoadFromGguf(*reader, &error);
  Expect(weights.has_value(), "bind: " + error);
  const auto& cfg = weights->config;
  const std::size_t hidden = cfg.hidden_size;

  std::vector<float> rising(256), falling(256), flat(256), ties(256), near(256),
      neg(256), wide(256);
  for (int i = 0; i < 256; ++i) {
    rising[static_cast<std::size_t>(i)] = static_cast<float>(i);
    falling[static_cast<std::size_t>(i)] = static_cast<float>(255 - i);
    flat[static_cast<std::size_t>(i)] = 0.25F;
    ties[static_cast<std::size_t>(i)] = static_cast<float>(i % 16);
    neg[static_cast<std::size_t>(i)] = static_cast<float>(i - 128);
    wide[static_cast<std::size_t>(i)] = (i * 17 % 50) - 25.0F;
  }
  near = rising;
  near[10] = near[11];
  near[200] = std::nextafter(near[199], 1.0e9F);
  CheckRoute(rising, "rising");
  CheckRoute(falling, "falling");
  CheckRoute(flat, "flat");
  CheckRoute(ties, "ties");
  CheckRoute(near, "near");
  CheckRoute(neg, "neg");
  CheckRoute(wide, "wide");
  std::cout << "KAT_ORACLE router_sweep=7 ids=bitonic weights=norm_w "
               "clamp=2^-14 sum_order=rank\n";

  std::vector<float> x(hidden);
  Fill(x, 0x6A71u);
  const auto& layer0 = weights->layers[0];
  std::vector<float> logits(cfg.expert_count);
  superroc::models::TensorGEMV(layer0.ffn_gate_inp, x, cfg.expert_count, hidden,
                               logits);
  Expect(
      std::fabs(logits[0] - RowDot(layer0.ffn_gate_inp, 0, hidden, x)) == 0.0F,
      "router row0");
  Expect(std::fabs(logits[255] - RowDot(layer0.ffn_gate_inp, 255, hidden, x)) ==
             0.0F,
         "router row255");
  CheckRoute(logits, "layer0");
  std::cout << "KAT_ORACLE router_layer0=ok\n";

  auto check_rows = [&](const superroc::models::QwenTensorRef& tensor,
                        std::size_t k, std::size_t rows, const char* name) {
    std::vector<float> y(rows);
    superroc::models::TensorGEMV(tensor, x, rows, k, y);
    for (std::size_t row : {std::size_t{0}, std::size_t{7}, rows - 1}) {
      const float ref = RowDot(tensor, row, k, x);
      Expect(y[row] == ref, std::string(name) + " row " + std::to_string(row));
    }
    std::cout << "KAT_ORACLE proj " << name
              << " rows=3 match=bitwise type=" << static_cast<int>(tensor.type)
              << "\n";
  };
  check_rows(layer0.attn_qkv, hidden, cfg.SsmQkvSize(), "ssm_qkv_l0");
  check_rows(weights->layers[3].attn_q, hidden,
             static_cast<std::size_t>(2) * cfg.AttentionSize(), "attn_q_l3");
  check_rows(weights->mtp->attn_q, hidden,
             static_cast<std::size_t>(2) * cfg.AttentionSize(), "attn_q_mtp");

  std::vector<float> prod(hidden), ref(hidden);
  std::vector<float> gscratch(512), uscratch(512), ascratch(512);
  for (int expert : {0, 8, 255}) {
    superroc::models::KatExpertSwiglu(layer0.ffn_gate_exps, layer0.ffn_up_exps,
                                      layer0.ffn_down_exps, expert, x, hidden,
                                      512, gscratch, uscratch, ascratch, prod);
    RefExpert(layer0.ffn_gate_exps, layer0.ffn_up_exps, layer0.ffn_down_exps,
              expert, x, hidden, 512, ref);
    const double cosine = Cosine(prod, ref);
    std::cout << "KAT_ORACLE expert e=" << expert << " cosine=" << cosine
              << "\n";
    Expect(cosine >= 0.999999, "expert cosine");
  }
  superroc::models::KatExpertSwiglu(layer0.ffn_gate_shexp, layer0.ffn_up_shexp,
                                    layer0.ffn_down_shexp, -1, x, hidden, 512,
                                    gscratch, uscratch, ascratch, prod);
  RefExpert(layer0.ffn_gate_shexp, layer0.ffn_up_shexp, layer0.ffn_down_shexp,
            -1, x, hidden, 512, ref);
  const double shexp_cos = Cosine(prod, ref);
  std::cout << "KAT_ORACLE shexp cosine=" << shexp_cos << "\n";
  Expect(shexp_cos >= 0.999999, "shexp cosine");

  superroc::models::QwenScratchArena arena(cfg);
  Expect(arena.moe_logits.size() == cfg.expert_count, "moe logit scratch");
  std::vector<float> probs(cfg.expert_count);
  superroc::models::ForwardMoe(x, layer0, cfg, logits, probs, arena.mlp_gate,
                               arena.mlp_up, arena.mlp_act, arena.attn_out,
                               arena.mlp_out);
  float moe_sum = 0.0F;
  for (std::size_t i = 0; i < 8; ++i) {
    moe_sum += 1.0F;
  }
  (void)moe_sum;
  double moe_energy = 0.0;
  for (float v : arena.mlp_out) {
    moe_energy += static_cast<double>(v) * v;
  }
  Expect(moe_energy > 0.0, "moe energy");
  std::cout << "KAT_ORACLE forward_moe energy=" << moe_energy << "\n";

  superroc::models::QwenKvCache kv(cfg.num_layers, cfg.num_key_value_heads, 8,
                                   cfg.head_dim);
  superroc::models::QwenSsmCache ssm(
      cfg.num_layers, cfg.SsmQkvSize(), cfg.ssm_conv_kernel,
      cfg.ssm_time_step_rank, cfg.ssm_state_size, cfg.SsmValueSize());
  auto run_layer = [&](std::uint32_t layer_index, std::vector<float>& out) {
    kv.Reset();
    ssm.Reset();
    std::copy(x.begin(), x.end(), arena.hidden.begin());
    superroc::models::ForwardLayer(arena.hidden, weights->layers[layer_index],
                                   cfg, kv, ssm, layer_index, 0, arena);
    out.assign(arena.hidden.begin(), arena.hidden.end());
  };
  std::vector<float> a, b;
  run_layer(0, a);
  run_layer(0, b);
  Expect(a == b, "ssm layer deterministic");
  bool finite = true;
  double energy = 0.0;
  for (float v : a) {
    finite = finite && std::isfinite(v);
    energy += static_cast<double>(v) * v;
  }
  Expect(finite && energy > 0.0, "ssm layer finite");
  Expect(arena.moe_logits.size() == cfg.expert_count &&
             (arena.moe_logits[0] != 0.0F || arena.moe_logits[1] != 0.0F),
         "trunk forward wrote the router");
  bool conv_written = false;
  for (float v : ssm.GetConvState(0)) {
    if (v != 0.0F) {
      conv_written = true;
      break;
    }
  }
  Expect(conv_written, "ssm conv state written");
  std::cout << "KAT_ORACLE ssm_layer0 energy=" << energy
            << " deterministic=1\n";
  run_layer(3, a);
  run_layer(3, b);
  Expect(a == b, "attn layer deterministic");
  finite = true;
  energy = 0.0;
  for (float v : a) {
    finite = finite && std::isfinite(v);
    energy += static_cast<double>(v) * v;
  }
  Expect(finite && energy > 0.0, "attn layer finite");
  std::cout << "KAT_ORACLE attn_layer3 energy=" << energy
            << " deterministic=1\n";
  std::cout << "KAT_ORACLE PASS\n";
  return 0;
}
