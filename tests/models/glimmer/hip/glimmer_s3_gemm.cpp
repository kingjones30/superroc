// GLM: one layer of real Glimmer FP4 weights, batch 2048.
// Each projection: W8A8 prequantized GEMM versus dequant-to-BF16 hipBLAS.
// Tolerance, fixed before the run: cosine >= 0.995 and
// rms(delta) / rms(reference) <= 0.05. Q8_1 activations are not bit-exact.
// Exit 0 pass, 1 fail, 2 usage, 77 skipped (no file named).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/state.hpp"

namespace {

constexpr std::size_t kBatch = 2048;
constexpr double kMinCosine = 0.995;
constexpr double kMaxRelRms = 0.05;

std::size_t ArenaQ8ScratchBytes(std::size_t batch, std::size_t row_elems) {
  const std::size_t q8_rows = ((batch + 15) / 16) * 16;
  return ((((q8_rows * row_elems) + 31) / 32) * sizeof(float) * 10) + 4096;
}

struct Score {
  double cosine = 0;
  double max_abs = 0;
  double rel_rms = 0;
  bool ok = false;
};

Score ScorePair(const float* reference, const float* candidate, std::size_t n,
                hipStream_t stream) {
  constexpr std::size_t kTile = 1 << 20;
  std::vector<float> href(kTile);
  std::vector<float> hnew(kTile);
  double dot = 0;
  double ref_sq = 0;
  double new_sq = 0;
  double err_sq = 0;
  double max_abs = 0;
  for (std::size_t offset = 0; offset < n; offset += kTile) {
    const std::size_t count = std::min(kTile, n - offset);
    HIP_CHECK(hipMemcpyAsync(href.data(), reference + offset,
                             count * sizeof(float), hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipMemcpyAsync(hnew.data(), candidate + offset,
                             count * sizeof(float), hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    for (std::size_t i = 0; i < count; ++i) {
      if (!std::isfinite(href[i]) || !std::isfinite(hnew[i])) {
        throw std::runtime_error("non-finite gemm output");
      }
      const double a = href[i];
      const double b = hnew[i];
      const double delta = b - a;
      dot += a * b;
      ref_sq += a * a;
      new_sq += b * b;
      err_sq += delta * delta;
      max_abs = std::max(max_abs, std::fabs(delta));
    }
  }
  Score score;
  score.max_abs = max_abs;
  const double denom = std::sqrt(ref_sq * new_sq);
  score.cosine = (denom > 0) ? dot / denom : 0;
  const double rms = std::sqrt(err_sq / static_cast<double>(n));
  const double ref_rms = std::sqrt(ref_sq / static_cast<double>(n));
  score.rel_rms = (ref_rms > 0) ? rms / ref_rms : rms;
  score.ok = score.cosine >= kMinCosine && score.rel_rms <= kMaxRelRms;
  return score;
}

void FillActivation(std::vector<float>& host, float* device, void* bf16,
                    std::size_t n, hipStream_t stream) {
  host.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    host[i] = std::sin(static_cast<float>(i % 997) * 0.017F) *
              (1.0F + static_cast<float>(i % 13) * 0.01F);
  }
  HIP_CHECK(hipMemcpyAsync(device, host.data(), n * sizeof(float),
                           hipMemcpyHostToDevice, stream));
  superroc::hip::LaunchFloatToBfloat16(device, bf16, n, stream);
  HIP_CHECK(hipGetLastError());
}

void ReferenceGemm(hipblasHandle_t blas, superroc::core::GgmlType type,
                   const void* weight, hip_bfloat16* weight_bf16,
                   const void* act_bf16, float* y, std::size_t m, std::size_t k,
                   hipStream_t stream) {
  superroc::hip::LaunchDequantizeToBf16(type, weight, weight_bf16, m * k, stream);
  HIP_CHECK(hipGetLastError());
  superroc::hip::LaunchHipblasGEMMBF16(blas, weight_bf16, act_bf16, y, kBatch, m, k,
                                   stream);
  HIP_CHECK(hipGetLastError());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 2) {
    std::cerr << "usage: glimmer_s3_gemm <gguf>\n";
    return 2;
  }
  // The muse-glimmer file is named on the command line or, under ctest, by
  // GLIMMER_S3_GGUF. A file that is neither named nor present skips (ctest
  // reports Skipped, exit 77) instead of failing, so a machine without the
  // weights still builds green.
  const char* configured = argc == 2 ? argv[1] : std::getenv("GLIMMER_S3_GGUF");
  if (configured == nullptr) {
    std::cerr << "SKIP glimmer_s3_gemm: GLIMMER_S3_GGUF is not set\n";
    return 77;
  }
  const std::string path = configured;
  try {
    std::string err;
    auto opened = superroc::core::GgufReader::OpenFile(path, &err);
    if (opened == nullptr) {
      std::cerr << "SKIP glimmer_s3_gemm: " << path << " unreadable: " << err
                << '\n';
      return 77;
    }
    superroc::hip::QwenExecutionPolicy policy;
    policy.kv_cache_storage = superroc::hip::QwenKvCacheStorage::kFp32;
    policy.fuse_qk_norm_rope_kv = false;
    auto executor = superroc::hip::QwenGpuExecutor::CreateFromGguf(
        std::shared_ptr<const superroc::core::GgufReader>(std::move(opened)), &err,
        256, policy);
    if (executor == nullptr) {
      std::cerr << "FAIL load " << err << '\n';
      return 1;
    }
    const auto& config = executor->GetConfig();
    const auto& layer = executor->GetModel().GetWeights().layers.at(0);
    const std::size_t hidden = config.hidden_size;
    const std::size_t attention = config.AttentionSize();
    const std::size_t kv =
        static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
    const std::size_t intermediate = config.intermediate_size;
    const std::size_t q_projection = 2 * attention;
    const std::size_t ssm_qkv = config.SsmQkvSize();
    const std::size_t row = std::max<std::size_t>(
        {intermediate, hidden, std::max(q_projection, ssm_qkv),
         ssm_qkv + config.ssm_inner_size +
             2 * static_cast<std::size_t>(config.ssm_time_step_rank)});
    const std::size_t have = ArenaQ8ScratchBytes(4096, row);
    const std::size_t need =
        superroc::hip::QuantizedActivationBytes(4096, intermediate);
    std::cout << "SCRATCH batch=4096 k=" << intermediate << " need=" << need
              << " have=" << have << " fit=" << (need <= have ? 1 : 0)
              << " cosine_min=" << kMinCosine << " rel_rms_max=" << kMaxRelRms
              << '\n';
    if (need > have) {
      std::cerr << "FAIL scratch\n";
      return 1;
    }

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    hipblasHandle_t blas = nullptr;
    HIPBLAS_CHECK(hipblasCreate(&blas));
    HIPBLAS_CHECK(hipblasSetStream(blas, stream));

    const std::size_t max_k = std::max(hidden, intermediate);
    const std::size_t max_m = std::max(intermediate, hidden);
    const std::size_t act_n = kBatch * max_k;
    const std::size_t y_n = kBatch * max_m;
    const std::size_t w_n = max_m * max_k;
    float* d_fp32 = nullptr;
    void* d_bf16 = nullptr;
    void* d_q8 = nullptr;
    hip_bfloat16* d_w = nullptr;
    float* d_ref = nullptr;
    float* d_new = nullptr;
    float* d_ref_b = nullptr;
    float* d_new_b = nullptr;
    HIP_CHECK(hipMalloc(&d_fp32, act_n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_bf16, act_n * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(
        &d_q8, superroc::hip::QuantizedActivationBytes(kBatch, max_k) + 4096));
    HIP_CHECK(hipMalloc(&d_w, w_n * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_ref, y_n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_new, y_n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ref_b, y_n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_new_b, y_n * sizeof(float)));

    std::vector<float> host;
    auto one = [&](const char* name, const superroc::models::QwenTensorRef& weight,
                   std::size_t m, std::size_t k, bool from_fp32) {
      if (weight.num_elements < m * k) {
        throw std::runtime_error(std::string(name) + " weight is short");
      }
      FillActivation(host, d_fp32, d_bf16, kBatch * k, stream);
      ReferenceGemm(blas, weight.type, weight.data, d_w, d_bf16, d_ref, m, k,
                    stream);
      if (from_fp32) {
        superroc::hip::LaunchQuantizeActivationQ8_1FromFp32(d_fp32, d_q8, kBatch, k,
                                                        stream);
      } else {
        superroc::hip::LaunchQuantizeActivationQ8_1(d_bf16, d_q8, kBatch, k,
                                                stream);
      }
      HIP_CHECK(hipGetLastError());
      superroc::hip::LaunchBatchedQuantGEMMPreQuantized(
          weight.type, weight.data, d_q8, d_new, kBatch, m, k, stream);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipStreamSynchronize(stream));
      const Score score = ScorePair(d_ref, d_new, kBatch * m, stream);
      std::cout << "GOLD " << name << " m=" << m << " k=" << k
                << " from_fp32=" << (from_fp32 ? 1 : 0)
                << " cosine=" << score.cosine << " max_abs=" << score.max_abs
                << " rel_rms=" << score.rel_rms
                << (score.ok ? " PASS" : " FAIL") << '\n';
      return score.ok;
    };

    bool ok = true;
    ok = one("attn_q", layer.attn_q, attention, hidden, false) && ok;
    ok = one("attn_k", layer.attn_k, kv, hidden, false) && ok;
    ok = one("attn_v", layer.attn_v, kv, hidden, false) && ok;
    ok = one("attn_gate", layer.attn_gate, attention, hidden, false) && ok;
    ok = one("attn_out", layer.attn_output, hidden, attention, true) && ok;
    ok = one("ffn_gate", layer.ffn_gate, intermediate, hidden, false) && ok;
    ok = one("ffn_up", layer.ffn_up, intermediate, hidden, false) && ok;
    ok = one("ffn_down", layer.ffn_down, hidden, intermediate, false) && ok;

    FillActivation(host, d_fp32, d_bf16, kBatch * hidden, stream);
    ReferenceGemm(blas, layer.ffn_gate.type, layer.ffn_gate.data, d_w, d_bf16,
                  d_ref, intermediate, hidden, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    ReferenceGemm(blas, layer.ffn_up.type, layer.ffn_up.data, d_w, d_bf16,
                  d_ref_b, intermediate, hidden, stream);
    superroc::hip::LaunchQuantizeActivationQ8_1(d_bf16, d_q8, kBatch, hidden,
                                            stream);
    HIP_CHECK(hipGetLastError());
    superroc::hip::LaunchBatchedDualQuantGEMMPreQuantized(
        layer.ffn_gate.type, layer.ffn_gate.data, layer.ffn_up.data, d_q8,
        d_new, d_new_b, kBatch, intermediate, hidden, stream);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    const Score dual_gate =
        ScorePair(d_ref, d_new, kBatch * intermediate, stream);
    const Score dual_up =
        ScorePair(d_ref_b, d_new_b, kBatch * intermediate, stream);
    std::cout << "GOLD ffn_dual_gate cosine=" << dual_gate.cosine
              << " max_abs=" << dual_gate.max_abs
              << " rel_rms=" << dual_gate.rel_rms
              << (dual_gate.ok ? " PASS" : " FAIL") << '\n';
    std::cout << "GOLD ffn_dual_up cosine=" << dual_up.cosine
              << " max_abs=" << dual_up.max_abs
              << " rel_rms=" << dual_up.rel_rms
              << (dual_up.ok ? " PASS" : " FAIL") << '\n';
    ok = ok && dual_gate.ok && dual_up.ok;
    std::cout << (ok ? "GOLD PASS\n" : "GOLD FAIL\n");

    hipFree(d_fp32);
    hipFree(d_bf16);
    hipFree(d_q8);
    hipFree(d_w);
    hipFree(d_ref);
    hipFree(d_new);
    hipFree(d_ref_b);
    hipFree(d_new_b);
    hipblasDestroy(blas);
    hipStreamDestroy(stream);
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::cerr << "FAIL " << ex.what() << '\n';
    return 1;
  }
}
