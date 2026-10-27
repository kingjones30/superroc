// FLASHNEXT G1-GPU (RV22 S6): dense-path type-mix qualification against the
// real ROC artifact. For every sampled matrix this compares
//   1. the float SmallGemm path (B3 device decode through RowElement) with
//      the superroc::quant::DotProductROCMFP4_* CPU oracle: cosine >= 0.999999;
//   2. Ling's activation-quantised superroc_fp4_gemv_m (the B4 narrow decode
//      intercept's kernel) with the same oracle: cosine >= 0.9999 (Q8_1
//      activations carry real quantisation error);
//   3. the executor's <=4-token chunk loop at m = 6 (4 + 2), which the
//      Dense intercept runs for decode batches above gemv_m's limit.
// Weights are mmapped from the file and copied to the device raw -- the
// artifact itself is never copied. Run under gpu-run.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/rocm_fp4/superroc_fp4_gemv_q8.hpp"
#include "src/core/rocm_fp4/superroc_fp4_quant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace {

namespace qfn = superroc::models::qwen38_flash_next;
namespace rocm = qfn::rocm;
using superroc::core::GgmlType;

int failures = 0;
int checked = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAIL %s\n", what.c_str());
    ++failures;
  }
}

#define HIP_CHECK(expr, what)                                               \
  do {                                                                      \
    const hipError_t err = (expr);                                          \
    if (err != hipSuccess) {                                                \
      std::printf("FAIL %s: %s\n", (what).c_str(), hipGetErrorString(err)); \
      ++failures;                                                           \
      return;                                                               \
    }                                                                       \
  } while (0)

// Deterministic activation; the same LCG as the G1-CPU oracle test so both
// gates chew on identical numbers.
std::uint64_t lcg_state = 0x243F6A8885A308D3ULL;
double NextUnit() {
  lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<double>((lcg_state >> 33) & 0xFFFFU) / 65535.0 * 2.0 - 1.0;
}
std::vector<float> MakeX(std::size_t n) {
  std::vector<float> x(n);
  for (float& v : x) {
    v = static_cast<float>(NextUnit());
  }
  return x;
}

double Cosine(const float* a, const float* b, std::size_t n) {
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  if (na == 0.0 || nb == 0.0) {
    return na == nb ? 1.0 : 0.0;
  }
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

constexpr std::uint32_t kMaxTokens = 12;

// Spread rows across the whole matrix: ends, thirds and LCG picks. The last
// rows are the ones that expose a wrong RowBytes stride.
std::vector<std::uint64_t> SampleRows(std::uint64_t rows) {
  std::vector<std::uint64_t> out;
  const auto add = [&](std::uint64_t r) {
    if (r < rows && std::find(out.begin(), out.end(), r) == out.end()) {
      out.push_back(r);
    }
  };
  add(0);
  add(1);
  add(rows / 3);
  add(rows / 2);
  if (rows >= 2) {
    add(rows - 2);
  }
  add(rows - 1);
  std::uint64_t s = rows * 2654435761ULL + 12345ULL;
  while (out.size() < 64 && out.size() < rows) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    add((s >> 33) % rows);
  }
  return out;
}

void CheckMatrix(const char* what, const qfn::TensorRef& t, std::uint64_t e) {
  const std::string tag(what);
  if (t.empty()) {
    Check(false, tag + " not bound");
    return;
  }
  if (t.type != GgmlType::kQ4_0_ROCMFP4 &&
      t.type != GgmlType::kQ4_0_ROCMFP4_FAST) {
    Check(false, tag + " not ROCmFP4");
    return;
  }
  Check(e < t.experts, tag + " expert in range");
  Check(t.cols % 32 == 0, tag + " cols not 32-aligned");
  Check(t.rows > 0, tag + " empty matrix");
  if (t.cols % 32 != 0 || t.rows == 0) {
    return;
  }
  const bool fast = t.type == GgmlType::kQ4_0_ROCMFP4_FAST;
  const std::size_t cols = static_cast<std::size_t>(t.cols);
  const std::size_t rows = static_cast<std::size_t>(t.rows);
  const std::size_t rb = t.RowBytes();
  const std::uint8_t* host = t.Expert(e);
  const std::size_t bytes = rb * rows;

  void* w_dev = nullptr;
  HIP_CHECK(hipMalloc(&w_dev, bytes + 4096), tag + " weight malloc");
  HIP_CHECK(hipMemcpy(w_dev, host, bytes, hipMemcpyHostToDevice),
            tag + " weight upload");
  HIP_CHECK(hipMemset(static_cast<std::uint8_t*>(w_dev) + bytes, 0, 4096),
            tag + " tail memset");

  const std::vector<float> x = MakeX(cols * kMaxTokens);
  float* x_dev = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&x_dev), x.size() * 4),
            tag + " x malloc");
  HIP_CHECK(hipMemcpy(x_dev, x.data(), x.size() * 4, hipMemcpyHostToDevice),
            tag + " x upload");

  const std::vector<std::uint64_t> sample = SampleRows(rows);

  // CPU oracle references for the sampled rows, per token.
  std::vector<std::vector<float>> ref(kMaxTokens,
                                      std::vector<float>(sample.size(), 0.0F));
  for (std::uint32_t tok = 0; tok < kMaxTokens; ++tok) {
    const std::span<const float> xv(x.data() + tok * cols, cols);
    for (std::size_t i = 0; i < sample.size(); ++i) {
      const std::uint8_t* row = host + sample[i] * rb;
      ref[tok][i] = fast ? superroc::quant::DotProductROCMFP4_Fast(row, xv, cols)
                         : superroc::quant::DotProductROCMFP4_Dual(row, xv, cols);
    }
  }

  float* y_dev = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&y_dev),
                      sizeof(float) * kMaxTokens * rows),
            tag + " y malloc");
  std::vector<float> y(static_cast<std::size_t>(kMaxTokens) * rows, 0.0F);
  std::vector<float> got(sample.size(), 0.0F);
  double worst_small = 1e9;
  double worst_gemv = 1e9;

  const auto readback_and_score = [&](std::uint32_t m, double* worst) {
    for (std::uint32_t tok = 0; tok < m; ++tok) {
      bool finite = true;
      for (std::size_t i = 0; i < sample.size(); ++i) {
        got[i] = y[static_cast<std::size_t>(tok) * rows + sample[i]];
        finite = finite && std::isfinite(got[i]);
      }
      Check(finite, tag + " non-finite output, m=" + std::to_string(m));
      *worst =
          std::min(*worst, Cosine(got.data(), ref[tok].data(), sample.size()));
    }
  };

  // 1) float SmallGemm path: m = 1, 6 and 12 (12 exercises the grouped
  //    >8-token launches inside SmallGemmForType).
  for (const std::uint32_t m : {1U, 6U, 12U}) {
    rocm::SmallGemm(w_dev,
                    fast ? rocm::WeightType::kQ4_0_ROCMFP4_FAST
                         : rocm::WeightType::kQ4_0_ROCMFP4,
                    x_dev, y_dev, m, static_cast<std::uint32_t>(rows),
                    static_cast<std::uint32_t>(cols), nullptr);
    HIP_CHECK(hipGetLastError(), tag + " SmallGemm launch");
    HIP_CHECK(hipDeviceSynchronize(), tag + " SmallGemm sync");
    HIP_CHECK(hipMemcpy(y.data(), y_dev, sizeof(float) * m * rows,
                        hipMemcpyDeviceToHost),
              tag + " SmallGemm readback");
    readback_and_score(m, &worst_small);
  }

  // 2) activation-quantised GEMV: single-shot m = 1 and m = 4, then the
  //    executor's chunk loop (4 + 2) at m = 6.
  for (const std::uint32_t m : {1U, 4U, 6U}) {
    for (std::uint32_t t0 = 0; t0 < m; t0 += 4) {
      const int chunk = static_cast<int>(std::min(4U, m - t0));
      const int rc = superroc_fp4_gemv_m(
          static_cast<int>(t.type), w_dev,
          x_dev + static_cast<std::size_t>(t0) * cols,
          y_dev + static_cast<std::size_t>(t0) * rows, static_cast<int>(cols),
          static_cast<int>(rows), chunk, nullptr);
      Check(rc == SUPERROC_FP4_OK, tag + " gemv_m rc=" + std::to_string(rc) +
                                   " m=" + std::to_string(m));
    }
    HIP_CHECK(hipDeviceSynchronize(), tag + " gemv_m sync");
    HIP_CHECK(hipMemcpy(y.data(), y_dev, sizeof(float) * m * rows,
                        hipMemcpyDeviceToHost),
              tag + " gemv_m readback");
    readback_and_score(m, &worst_gemv);
  }

  Check(
      worst_small >= 0.999999,
      tag + " SmallGemm cosine " + std::to_string(worst_small) + " < 0.999999");
  Check(worst_gemv >= 0.9999,
        tag + " gemv_m cosine " + std::to_string(worst_gemv) + " < 0.9999");
  ++checked;
  std::printf(
      "%-30s type=%3d cols=%6zu rows=%7zu e=%3llu sample=%3zu "
      "cos_small=%.7f cos_gemv=%.6f\n",
      what, static_cast<int>(t.type), cols, rows,
      static_cast<unsigned long long>(e), sample.size(), worst_small,
      worst_gemv);

  (void)hipFree(y_dev);
  (void)hipFree(x_dev);
  (void)hipFree(w_dev);
}

// S7: grouped FP4 expert FFN (superroc_fp4_expert_ffn_m) on the real l47 expert
// tensors vs an exact CPU reference built from the oracle row dots:
//   y[t] = sum_s w[t][s] * down_id( silu(gate_id . x_t) * (up_id . x_t) )
// accumulated in double. The kernel quantises activations to Q8_1 internally
// (and its SwiGLU intermediate), so the gate is the gemv-class cosine, not
// bit-exactness. Weights are renormalised per token like RouterTopK.
int moe_checked = 0;

void CheckMoe(const char* what, const qfn::LayerWeights& l,
              const qfn::Config& c) {
  const std::string tag(what);
  const qfn::TensorRef& gate = l.ffn_gate_exps;
  const qfn::TensorRef& up = l.ffn_up_exps;
  const qfn::TensorRef& down = l.ffn_down_exps;
  Check(!gate.empty() && !up.empty() && !down.empty(), tag + " experts bound");
  if (gate.empty() || up.empty() || down.empty()) {
    return;
  }
  const std::size_t k = c.hidden_size;
  const std::size_t nff = c.expert_ff;
  const std::uint32_t used = c.num_experts_used;
  Check(gate.type == GgmlType::kQ4_0_ROCMFP4_FAST && up.type == gate.type &&
            down.type == gate.type,
        tag + " experts all fast FP4");
  Check(gate.experts == c.num_experts && gate.cols == k && gate.rows == nff &&
            up.cols == k && up.rows == nff && down.cols == nff &&
            down.rows == k,
        tag + " expert geometry");
  if (gate.type != GgmlType::kQ4_0_ROCMFP4_FAST ||
      gate.experts != c.num_experts) {
    return;
  }

  void* gate_dev = nullptr;
  void* up_dev = nullptr;
  void* down_dev = nullptr;
  const auto upload = [&](const qfn::TensorRef& t, void** dev, const char* nm) {
    const std::size_t bytes = t.SizeBytes();
    if (hipMalloc(dev, bytes + 4096) != hipSuccess ||
        hipMemcpy(*dev, t.data, bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemset(static_cast<std::uint8_t*>(*dev) + bytes, 0, 4096) !=
            hipSuccess) {
      Check(false, tag + " " + nm + " expert upload");
      return false;
    }
    return true;
  };
  if (!upload(gate, &gate_dev, "gate") || !upload(up, &up_dev, "up") ||
      !upload(down, &down_dev, "down")) {
    (void)hipFree(gate_dev);
    (void)hipFree(up_dev);
    (void)hipFree(down_dev);
    return;
  }

  const std::uint8_t* gate_h = gate.Expert(0);
  const std::uint8_t* up_h = up.Expert(0);
  const std::uint8_t* down_h = down.Expert(0);
  const std::size_t gate_rb = gate.RowBytes();
  const std::size_t up_rb = up.RowBytes();
  const std::size_t down_rb = down.RowBytes();
  const std::size_t gate_estride = gate_rb * gate.rows;
  const std::size_t up_estride = up_rb * up.rows;
  const std::size_t down_estride = down_rb * down.rows;

  for (const std::uint32_t ntok : {1U, 4U}) {
    // Deterministic routing: distinct expert ids per token, weights in
    // [0.2, 1.0) renormalised to sum 1 (what RouterTopK produces).
    std::vector<std::int32_t> ids(static_cast<std::size_t>(ntok) * used);
    std::vector<float> wts(static_cast<std::size_t>(ntok) * used);
    std::uint64_t s = 0x9E3779B97F4A7C15ULL ^ (ntok * 0xD1B54A32D192ED03ULL);
    for (std::uint32_t t = 0; t < ntok; ++t) {
      std::vector<bool> taken(c.num_experts, false);
      float sum = 0.0F;
      for (std::uint32_t j = 0; j < used; ++j) {
        std::uint32_t id = 0;
        do {
          s = s * 6364136223846793005ULL + 1442695040888963407ULL;
          id = static_cast<std::uint32_t>((s >> 33) % c.num_experts);
        } while (taken[id]);
        taken[id] = true;
        ids[static_cast<std::size_t>(t) * used + j] =
            static_cast<std::int32_t>(id);
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        wts[static_cast<std::size_t>(t) * used + j] = static_cast<float>(
            (static_cast<double>((s >> 33) & 0xFFFFU) / 65535.0) * 0.8 + 0.2);
        sum += wts[static_cast<std::size_t>(t) * used + j];
      }
      for (std::uint32_t j = 0; j < used; ++j) {
        wts[static_cast<std::size_t>(t) * used + j] /= sum;
      }
    }

    const std::vector<float> x = MakeX(k * ntok);
    float* x_dev = nullptr;
    std::int32_t* ids_dev = nullptr;
    float* w_dev = nullptr;
    float* y_dev = nullptr;
    const std::size_t ids_b = ids.size() * sizeof(std::int32_t);
    const std::size_t w_b = wts.size() * sizeof(float);
    const bool bufs_ok =
        hipMalloc(reinterpret_cast<void**>(&x_dev), x.size() * 4) ==
            hipSuccess &&
        hipMemcpy(x_dev, x.data(), x.size() * 4, hipMemcpyHostToDevice) ==
            hipSuccess &&
        hipMalloc(reinterpret_cast<void**>(&ids_dev), ids_b) == hipSuccess &&
        hipMemcpy(ids_dev, ids.data(), ids_b, hipMemcpyHostToDevice) ==
            hipSuccess &&
        hipMalloc(reinterpret_cast<void**>(&w_dev), w_b) == hipSuccess &&
        hipMemcpy(w_dev, wts.data(), w_b, hipMemcpyHostToDevice) ==
            hipSuccess &&
        hipMalloc(reinterpret_cast<void**>(&y_dev),
                  static_cast<std::size_t>(ntok) * k * 4) == hipSuccess;
    Check(bufs_ok, tag + " buffers m=" + std::to_string(ntok));
    if (!bufs_ok) {
      break;
    }

    const int rc = superroc_fp4_expert_ffn_m(
        101, gate_dev, up_dev, down_dev, x_dev, ids_dev, w_dev,
        static_cast<int>(ntok), static_cast<int>(used),
        static_cast<int>(c.num_experts), static_cast<int>(k),
        static_cast<int>(nff), 0.0F, y_dev, nullptr);
    Check(rc == SUPERROC_FP4_OK, tag + " expert_ffn_m rc=" + std::to_string(rc) +
                                 " m=" + std::to_string(ntok));
    const int st = superroc_fp4_expert_ffn_m_check();
    Check(st == SUPERROC_FP4_OK, tag + " expert_ffn_m_check=" + std::to_string(st));
    Check(hipDeviceSynchronize() == hipSuccess, tag + " moe sync");
    std::vector<float> y(static_cast<std::size_t>(ntok) * k, 0.0F);
    Check(hipMemcpy(y.data(), y_dev, y.size() * 4, hipMemcpyDeviceToHost) ==
              hipSuccess,
          tag + " moe readback");

    // Exact CPU reference (oracle row dots, double accumulation).
    std::vector<float> ref(k, 0.0F);
    std::vector<float> hf(nff, 0.0F);
    std::vector<double> acc(k, 0.0);
    double worst_cos = 1e9;
    for (std::uint32_t t = 0; t < ntok; ++t) {
      std::fill(acc.begin(), acc.end(), 0.0);
      const std::span<const float> xt(
          x.data() + static_cast<std::size_t>(t) * k, k);
      for (std::uint32_t j = 0; j < used; ++j) {
        const std::uint32_t id = static_cast<std::uint32_t>(
            ids[static_cast<std::size_t>(t) * used + j]);
        const float wj = wts[static_cast<std::size_t>(t) * used + j];
        const std::uint8_t* g_e =
            gate_h + static_cast<std::size_t>(id) * gate_estride;
        const std::uint8_t* u_e =
            up_h + static_cast<std::size_t>(id) * up_estride;
        const std::uint8_t* d_e =
            down_h + static_cast<std::size_t>(id) * down_estride;
        for (std::size_t j2 = 0; j2 < nff; ++j2) {
          const double gv =
              superroc::quant::DotProductROCMFP4_Fast(g_e + j2 * gate_rb, xt, k);
          const double uv =
              superroc::quant::DotProductROCMFP4_Fast(u_e + j2 * up_rb, xt, k);
          hf[j2] = static_cast<float>(gv / (1.0 + std::exp(-gv)) * uv);
        }
        const std::span<const float> hs(hf.data(), nff);
        for (std::size_t i = 0; i < k; ++i) {
          acc[i] +=
              static_cast<double>(wj) *
              superroc::quant::DotProductROCMFP4_Fast(d_e + i * down_rb, hs, nff);
        }
      }
      for (std::size_t i = 0; i < k; ++i) {
        ref[i] = static_cast<float>(acc[i]);
      }
      const float* got = y.data() + static_cast<std::size_t>(t) * k;
      bool finite = true;
      for (std::size_t i = 0; i < k; ++i) {
        finite = finite && std::isfinite(got[i]);
      }
      Check(finite, tag + " non-finite moe output, m=" + std::to_string(ntok));
      worst_cos = std::min(worst_cos, Cosine(got, ref.data(), k));
    }
    Check(worst_cos >= 0.9999, tag + " expert_ffn cosine " +
                                   std::to_string(worst_cos) +
                                   " < 0.9999 m=" + std::to_string(ntok));
    ++moe_checked;
    std::printf("%-30s tokens=%u used=%u experts=%u cos=%.6f\n", what, ntok,
                used, c.num_experts, worst_cos);

    (void)hipFree(y_dev);
    (void)hipFree(w_dev);
    (void)hipFree(ids_dev);
    (void)hipFree(x_dev);
  }

  (void)hipFree(down_dev);
  (void)hipFree(up_dev);
  (void)hipFree(gate_dev);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s ROC.gguf\n", argv[0]);
    return 2;
  }
  std::string error;
  auto reader = superroc::core::GgufReader::OpenFile(argv[1], &error);
  if (reader == nullptr) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 2;
  }
  auto weights = qfn::ModelWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    std::fprintf(stderr, "BIND FAILED: %s\n", error.c_str());
    return 1;
  }
  const qfn::ModelWeights& w = *weights;
  Check(w.config.compress_ratio == 4, "compress_ratio reconstructed to 4");
  Check(w.config.num_layers == 48, "48 trunk layers");

  const auto& l0 = w.layers[0];    // linear layer: dual ssm_qkv, fast gates
  const auto& l3 = w.layers[3];    // sparse layer: dual attn_k/v, fast q/out
  const auto& l47 = w.layers[47];  // last trunk layer: experts + shexp

  // The type mix: both FP4 types, wide and narrow matrices, the hc mixer
  // family, expert slices e=0/7/511 and the shared expert.
  CheckMatrix("l0.ssm_qkv (dual)", l0.ssm_qkv, 0);
  CheckMatrix("l0.ssm_gate (fast)", l0.ssm_gate, 0);
  CheckMatrix("l0.ssm_out (fast)", l0.ssm_out, 0);
  CheckMatrix("l0.ssm_alpha (fast)", l0.ssm_alpha, 0);
  CheckMatrix("l3.attn_k (dual)", l3.attn_k, 0);
  CheckMatrix("l3.attn_v (dual)", l3.attn_v, 0);
  CheckMatrix("l3.attn_q (fast)", l3.attn_q, 0);
  CheckMatrix("l3.attn_out (fast)", l3.attn_out, 0);
  CheckMatrix("l3.indexer_q (fast)", l3.indexer_q, 0);
  CheckMatrix("hc.down (fast)", w.hc_head.down, 0);
  CheckMatrix("hc.up (fast)", w.hc_head.up, 0);
  // w.hc_head is bound with_inject=false; the inject matrices live on the
  // per-layer hc_attn mixers (blk*.hc_attn_inject.weight, fast FP4).
  CheckMatrix("l0.hc_attn.inject (fast)", l0.hc_attn.inject, 0);
  if (w.config.ple_layer >= 0) {
    const auto& lp = w.layers[static_cast<std::size_t>(w.config.ple_layer)];
    CheckMatrix("ple.key (fast)", lp.ple_key, 0);
    CheckMatrix("ple.value (fast)", lp.ple_value, 0);
  } else {
    Check(false, "artifact has no PLE layer");
  }
  CheckMatrix("l47.gate_exps e=0", l47.ffn_gate_exps, 0);
  CheckMatrix("l47.gate_exps e=7", l47.ffn_gate_exps, 7);
  CheckMatrix("l47.up_exps e=511", l47.ffn_up_exps, 511);
  CheckMatrix("l47.down_exps e=511", l47.ffn_down_exps, 511);
  CheckMatrix("l47.shexp_gate (fast)", l47.shexp_gate, 0);
  CheckMatrix("l47.shexp_down (fast)", l47.shexp_down, 0);

  // S7: the grouped expert FFN over the real 512-expert tensors, m=1 and
  // m=4 (the executor chunk limit), vs the exact oracle reference.
  CheckMoe("l47.moe (fast)", l47, w.config);

  std::printf("checked %d matrices + %d moe runs, %d failures\n", checked,
              moe_checked, failures);
  if (failures == 0) {
    std::printf("G1-GPU PASS\n");
    return 0;
  }
  std::printf("G1-GPU FAIL\n");
  return 1;
}
