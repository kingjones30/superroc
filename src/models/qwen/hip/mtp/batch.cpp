#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/kernels/q4_0_gemv.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen/hip/mtp/kat_moe_ffn.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/core/env_compat.hpp"

namespace superroc::hip {

/// LaunchMtpFusionNorms lives in kernels/mtp_batch.hip.
void LaunchMtpFusionNorms(const float* embed_rows, const float* h_rows,
                          const float* enorm, const float* hnorm, float* fusion,
                          std::size_t batch_size, std::size_t hidden, float eps,
                          hipStream_t stream);

void QwenMtpGpuExecutor::ForwardTargetBatch(
    std::span<const tokenization::TokenId> tokens,
    std::span<const float> hidden_rows, std::uint32_t start_pos) {
  const std::size_t batch = tokens.size();
  if (batch == 0 ||
      hidden_rows.size() != batch * model_->GetConfig().hidden_size) {
    throw std::invalid_argument("MTP batched prime inputs are malformed");
  }
  if (start_pos != next_position_ ||
      static_cast<std::size_t>(start_pos) + batch > max_context_) {
    throw std::out_of_range("MTP batched prime positions are not sequential");
  }
  const auto& weights = model_->GetWeights();
  const auto& config = weights.config;
  const auto& layer = weights.layer;
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context_ *
      config.head_dim;
  if (batch > kPrimeBatchRows) {
    throw std::out_of_range("MTP batched prime chunk exceeds the scratch");
  }
  // KAT-MTP F1 (RV44): the full-block prime below runs the dense FFN GEMMs.
  // On a MoE block those weights are empty, so it would launch a null-weight
  // GEMM and fault the GPU with King's seats resident. Refuse before any
  // device work. The caller (PrimeTargetContext) catches this and falls back
  // to the serial prime, which runs the real MoE FFN.
  if (layer.is_moe && ::superroc::core::GetEnvCompat("SUPERROC_MTP_PRIME_FULL") != nullptr) {
    static bool announced_full_moe = false;
    if (!announced_full_moe) {
      announced_full_moe = true;
      std::fprintf(stderr,
                   "mtp-prime path=full refused on a MoE block; using the "
                   "serial prime\n");
    }
    throw std::runtime_error(
        "MTP full-block prime is unsupported on a MoE block");
  }
  switch (weights.token_embedding.type) {
    case core::GgmlType::kBF16:
    case core::GgmlType::kF32:
    case core::GgmlType::kQ4_0:
    case core::GgmlType::kQ8_0:
    case core::GgmlType::kQ3_K:
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ6_K:
    case core::GgmlType::kQ8_K:
    case core::GgmlType::kIQ4_NL:
    case core::GgmlType::kIQ4_XS:
    case core::GgmlType::kQ4_0_ROCMFP4:  // KAT-MTP: tied trunk embedding
    case core::GgmlType::kQ4_0_ROCMFP4_FAST:
      break;
    default:
      throw std::runtime_error(
          "MTP batched prime embedding type is unsupported");
  }

  std::vector<std::uint32_t> host_tokens(tokens.begin(), tokens.end());
  const auto token_error = hipMemcpyAsync(d_b_tokens_, host_tokens.data(),
                                          batch * sizeof(std::uint32_t),
                                          hipMemcpyHostToDevice, stream_);
  const auto hidden_error = hipMemcpyAsync(d_b_h_, hidden_rows.data(),
                                           hidden_rows.size() * sizeof(float),
                                           hipMemcpyHostToDevice, stream_);
  // The host sources die (or get reused by the next chunk) when this
  // returns. The copies are queued on stream_; wait for them before any
  // kernel, so a pageable upload cannot race the next fill.
  if (token_error != hipSuccess || hidden_error != hipSuccess ||
      hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP batched prime upload failed");
  }

  LaunchBatchedEmbeddingLookup(weights.token_embedding.data,
                               weights.token_embedding.type, d_b_tokens_,
                               d_b_embed_, batch, hidden, stream_);
  LaunchMtpFusionNorms(d_b_embed_, d_b_h_,
                       static_cast<const float*>(weights.embedding_norm.data),
                       static_cast<const float*>(weights.hidden_norm.data),
                       d_b_fusion_, batch, hidden, 1.0e-6F, stream_);
  RunPrimeGemm(weights.fusion_projection.data, weights.fusion_projection.type,
               d_b_fusion_, d_b_hidden_, batch, hidden, 2 * hidden, stream_);

  LaunchBatchedRMSNorm(d_b_hidden_,
                       static_cast<const float*>(layer.attn_norm.data),
                       d_b_normed_, nullptr, batch, hidden, 1.0e-6F, stream_);
  // SPEC4 item 2. Priming only fills the draft KV. K and V are projections of
  // this fused hidden, taken before attention, so Q, attention, the output
  // projection, and the FFN do not change the cache. Decode runs the full
  // block itself. SUPERROC_MTP_PRIME_FULL=1 keeps the old full block.
  const bool full_block = ::superroc::core::GetEnvCompat("SUPERROC_MTP_PRIME_FULL") != nullptr;
  static bool announced = false;
  if (!announced) {
    announced = true;
    std::fprintf(stderr, "mtp-prime path=%s\n",
                 full_block ? "full" : "kv-only");
  }
  RunPrimeGemm(layer.attn_k.data, layer.attn_k.type, d_b_normed_, d_b_k_, batch,
               kv, hidden, stream_);
  RunPrimeGemm(layer.attn_v.data, layer.attn_v.type, d_b_normed_, d_b_v_, batch,
               kv, hidden, stream_);
  if (!full_block) {
    const detail::AttentionSupportParams tiled{
        .batch_size = batch,
        .start_pos = start_pos,
        .max_context = max_context_,
        .num_heads = config.num_attention_heads,
        .num_kv_heads = config.num_key_value_heads,
        .head_dim = config.head_dim,
        .has_k_cache_f16 = true,
        .has_v_cache_f16 = true,
    };
    LaunchBatchedPerHeadRMSNorm(
        d_b_k_, static_cast<const float*>(layer.attn_k_norm.data), d_b_k_,
        batch, config.num_key_value_heads, config.head_dim, 1.0e-6F, stream_);
    LaunchBatchedRoPE(nullptr, d_b_k_, batch, /*num_heads=*/0,
                      config.num_key_value_heads, config.head_dim,
                      config.rotary_dim, start_pos, config.rope_theta, stream_,
                      nullptr);
    if (detail::IsTiledAttentionSupported(tiled)) {
      const bool packed = LaunchPackTiledAttentionKv(
          d_b_k_, d_b_v_, d_kv_cache_, d_kv_cache_ + total_kv, d_kv_cache_f16_,
          static_cast<std::uint16_t*>(d_kv_cache_f16_) + total_kv,
          /*layer_idx=*/0, start_pos, batch, max_context_,
          config.num_attention_heads, config.num_key_value_heads,
          config.head_dim, stream_,
          /*sync_prefix=*/start_pos > f16_prefix_packed_);
      if (!packed) {
        throw std::runtime_error("MTP batched prime KV pack shape unsupported");
      }
    } else {
      // KAT-MTP: 2 KV heads. Same bytes WriteKVCacheKernel stores for layer 0.
      static bool announced_general = false;
      if (!announced_general) {
        announced_general = true;
        std::fprintf(stderr, "mtp-prime path=kv-general\n");
      }
      LaunchKatMtpWriteKv(
          d_b_k_, d_b_v_, d_kv_cache_, d_kv_cache_ + total_kv, d_kv_cache_f16_,
          static_cast<std::uint16_t*>(d_kv_cache_f16_) + total_kv, start_pos,
          batch, max_context_, config.num_key_value_heads, config.head_dim,
          stream_);
    }
    f16_prefix_packed_ = start_pos + static_cast<std::uint32_t>(batch);
    next_position_ += static_cast<std::uint32_t>(batch);
    // SQ10 SPEC3 (S3-1): terminal drain (the :100 sync is ingress only).
    // Every drafter entry that queues a launch drains before returning.
    if (hipStreamSynchronize(stream_) != hipSuccess) {
      throw std::runtime_error("MTP batched prime completion failed");
    }
    return;
  }
  LaunchBatchedPerHeadRMSNorm(
      d_b_k_, static_cast<const float*>(layer.attn_k_norm.data), d_b_k_, batch,
      config.num_key_value_heads, config.head_dim, 1.0e-6F, stream_);
  RunPrimeGemm(layer.attn_q.data, layer.attn_q.type, d_b_normed_, d_b_qg_,
               batch, 2 * attention, hidden, stream_);
  LaunchBatchedUnpackQG(d_b_qg_, d_b_q_, d_b_gate_, batch,
                        config.num_attention_heads, config.head_dim, stream_);
  LaunchBatchedPerHeadRMSNorm(
      d_b_q_, static_cast<const float*>(layer.attn_q_norm.data), d_b_q_, batch,
      config.num_attention_heads, config.head_dim, 1.0e-6F, stream_);
  LaunchBatchedRoPE(d_b_q_, d_b_k_, batch, config.num_attention_heads,
                    config.num_key_value_heads, config.head_dim,
                    config.rotary_dim, start_pos, config.rope_theta, stream_,
                    nullptr);
  const bool tiled = LaunchBatchedAttentionTile(
      d_b_q_, d_b_k_, d_b_v_, d_b_gate_, d_kv_cache_, d_kv_cache_ + total_kv,
      d_kv_cache_f16_, static_cast<std::uint16_t*>(d_kv_cache_f16_) + total_kv,
      d_b_context_, 0, start_pos, batch, max_context_,
      config.num_attention_heads, config.num_key_value_heads, config.head_dim,
      stream_);
  if (!tiled) {
    throw std::runtime_error("MTP batched prime attention shape unsupported");
  }
  RunPrimeGemm(layer.attn_output.data, layer.attn_output.type, d_b_context_,
               d_b_attn_out_, batch, hidden, attention, stream_);
  LaunchBatchedResidualAdd(d_b_hidden_, d_b_attn_out_, d_b_hidden_, batch,
                           hidden, stream_);

  LaunchBatchedRMSNorm(d_b_hidden_,
                       static_cast<const float*>(layer.ffn_norm.data),
                       d_b_normed_, nullptr, batch, hidden, 1.0e-6F, stream_);
  RunPrimeGemm(layer.ffn_gate.data, layer.ffn_gate.type, d_b_normed_,
               d_b_ffn_act_, batch, config.intermediate_size, hidden, stream_);
  RunPrimeGemm(layer.ffn_up.data, layer.ffn_up.type, d_b_normed_, d_b_ffn_up_,
               batch, config.intermediate_size, hidden, stream_);
  LaunchBatchedSwiGLUActivation(d_b_ffn_act_, d_b_ffn_up_, d_b_ffn_act_,
                                nullptr, batch * config.intermediate_size,
                                stream_);
  RunPrimeGemm(layer.ffn_down.data, layer.ffn_down.type, d_b_ffn_act_,
               d_b_ffn_out_, batch, hidden, config.intermediate_size, stream_);
  LaunchBatchedResidualAdd(d_b_hidden_, d_b_ffn_out_, d_b_hidden_, batch,
                           hidden, stream_);

  f16_prefix_packed_ = start_pos + static_cast<std::uint32_t>(batch);
  next_position_ += static_cast<std::uint32_t>(batch);
  // SQ10 SPEC3 (S3-1): terminal drain, as on the kv-only return above.
  if (hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP batched prime completion failed");
  }
}

void QwenMtpGpuExecutor::RunPrimeGemm(const void* weight, core::GgmlType type,
                                      const float* input, float* output,
                                      std::size_t batch, std::size_t rows_out,
                                      std::size_t cols_in, hipStream_t stream) {
  (void)stream;
  if (type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ8_K) {
    // KAT-MTP: the small-batch Q8 kernel is general for 1..8 rows. Wider
    // prime chunks share a weight read eight rows at a time.
    std::size_t done = 0;
    while (done < batch) {
      const std::size_t rows = std::min<std::size_t>(8, batch - done);
      LaunchBatchedQuantGEMMFp32(type, weight, input + done * cols_in,
                                 output + done * rows_out, rows, rows_out,
                                 cols_in, stream_);
      done += rows;
    }
    return;
  }
  const void* bf16_weight = weight;
  if (type == core::GgmlType::kQ4_0) {
    const std::size_t elements = rows_out * cols_in;
    if (d_q4_bf16_ == nullptr || elements > q4_bf16_elems_) {
      throw std::runtime_error("MTP prime Q4_0 weight scratch is too small");
    }
    LaunchDequantQ4_0ToBf16(weight, d_q4_bf16_, elements, stream_);
    bf16_weight = d_q4_bf16_;
  }
  if (batch > 16 && hipblas_ != nullptr && d_b_bf16_ != nullptr &&
      batch * cols_in <= b_bf16_elems_) {
    LaunchFloatToBfloat16(input, d_b_bf16_, batch * cols_in, stream_);
    LaunchHipblasGEMMBF16(hipblas_, bf16_weight, d_b_bf16_, output, batch,
                          rows_out, cols_in, stream_);
    return;
  }
  LaunchExactBf16GEMMFp32SmallBatch(bf16_weight, input, output, batch, rows_out,
                                    cols_in, stream_);
}

std::vector<float> QwenMtpGpuExecutor::CopyKvCacheF32() {
  const auto& config = model_->GetConfig();
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context_ *
      config.head_dim;
  std::vector<float> host(2 * total_kv);
  const auto error =
      hipMemcpyAsync(host.data(), d_kv_cache_, host.size() * sizeof(float),
                     hipMemcpyDeviceToHost, stream_);
  if (error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP KV export failed");
  }
  return host;
}

}  // namespace superroc::hip
#endif  // defined(ENGINE_ENABLE_HIP)
