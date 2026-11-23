// Flint batched prefill. Batch >= 2; caller handles 1.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/executor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/models/glimmer/hip/glimmer_attention.hpp"
#include "src/models/glimmer/hip/glimmer_decode.hpp"  // LFM2 (fp32 KV + attention)
#include "src/models/lfm2/hip/lfm2_kernels.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/moe/qwen_moe_prefill.hpp"  // LFM2 (grouped experts)

namespace superroc::hip {
namespace {

// LFM2: Glimmer's CheckGpu renamed.
void FlintCheckGpu(const char* what) {
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error(what);
  }
}

// LFM2: Glimmer's WeightScratchElems renamed.
std::size_t FlintWeightScratchElems(const core::ModelConfig& config) {
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection = 2 * attention;
  return hidden * std::max<std::size_t>(
                      {q_projection, kv, attention, config.intermediate_size,
                       config.SsmQkvSize(), config.ssm_inner_size,
                       config.ssm_time_step_rank});
}

// LFM2: Glimmer's W8A8 decision renamed.
bool FlintUseW8A8() {
  static const bool on = [] {
    const char* env = std::getenv("GLIMMER_W8A8");
    return !(env != nullptr && env[0] == '0' && env[1] == '\0');
  }();
  return on;
}

// LFM2: Glimmer's W8A8MaxBatch renamed.
std::size_t FlintW8A8MaxBatch() {
  static const std::size_t rows = [] {
    const char* env = std::getenv("GLIMMER_W8A8_MAX_BATCH");
    if (env != nullptr && env[0] != '\0') {
      return static_cast<std::size_t>(std::strtoul(env, nullptr, 10));
    }
    return std::size_t{2048};
  }();
  return rows;
}

// LFM2: Glimmer's UseW8A8ForChunk renamed.
bool FlintUseW8A8ForChunk(std::size_t batch) {
  return FlintUseW8A8() && batch <= FlintW8A8MaxBatch();
}

// LFM2 (Super Roc): W8A8 needs every projection the w8 branches touch to be
// a native WMMA quant. Layer 0 is short-conv (no attn_q), so look at all.
bool FlintAllProjectionsNative(const models::QwenModelWeights& weights) {
  for (const auto& layer : weights.layers) {
    const models::QwenTensorRef* refs[] = {
        &layer.attn_q, &layer.attn_k, &layer.attn_v, &layer.attn_output,
        &layer.ffn_gate, &layer.ffn_up, &layer.ffn_down,
        &layer.ffn_gate_shexp, &layer.ffn_up_shexp, &layer.ffn_down_shexp};
    for (const models::QwenTensorRef* ref : refs) {
      if (!ref->empty() && !detail::IsNativeWmmaQuant(ref->type)) {
        return false;
      }
    }
  }
  return true;
}

// LFM2: Glimmer's ArenaQ8ScratchBytes renamed.
std::size_t FlintArenaQ8ScratchBytes(std::size_t batch,
                                    std::size_t row_elems) {
  const std::size_t q8_rows = ((batch + 15) / 16) * 16;
  return ((((q8_rows * row_elems) + 31) / 32) * sizeof(float) * 10) + 4096;
}

// LFM2: Glimmer's Require pattern for LFM2 launch status.
void FlintRequire(const Lfm2LaunchStatus& status) {
  if (!status.ok) {
    throw std::runtime_error(status.error != nullptr ? status.error
                                                     : "flint launch failed");
  }
}

// LFM2: Glimmer's Require pattern for Glimmer launch status.
void FlintRequire(const GlimmerLaunchStatus& status) {
  if (!status.ok) {
    throw std::runtime_error(status.error != nullptr ? status.error
                                                     : "glimmer launch failed");
  }
}

bool FlintWmmaEnabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("FLINT_WMMA");
    return env == nullptr || std::strcmp(env, "0") != 0;
  }();
  return enabled;
}

// LFM2: Glimmer's Project renamed (BF16 dequant + hipBLAS, batch 1 via GEMV).
void FlintProject(QwenGpuArena& arena, const models::QwenTensorRef& weight,
                 const float* fp32, const void* bf16, float* y,
                 std::size_t batch, std::size_t m, std::size_t k,
                 std::size_t weight_elems) {
  if (m == 0 || k == 0 || batch == 0) {
    throw std::runtime_error("flint prefill projection is empty");
  }
  if (batch == 1) {
    LaunchGEMV(weight.data, weight.type, fp32, y, m, k, arena.stream);
    FlintCheckGpu("flint prefill gemv failed");
    return;
  }
  if (m * k > weight_elems) {
    throw std::runtime_error("flint prefill weight scratch is short");
  }
  LaunchDequantizeToBf16(weight.type, weight.data, arena.d_weights_bf16,
                         m * k, arena.stream);
  FlintCheckGpu("flint prefill dequant failed");
  bool ran = false;
  if (arena.hipblaslt_gemm != nullptr && m >= 1024 && k >= 1024) {
    ran = arena.hipblaslt_gemm->RunBf16(arena.d_weights_bf16, bf16, y, batch,
                                        m, k, arena.stream);
  }
  if (!ran) {
    LaunchHipblasGEMMBF16(arena.hipblas_handle, arena.d_weights_bf16, bf16,
                          y, batch, m, k, arena.stream);
  }
  FlintCheckGpu("flint prefill bf16 gemm failed");
}

// LFM2: Glimmer's ProjectQ8 renamed (prequantized W8A8 WMMA).
void FlintProjectQ8(QwenGpuArena& arena, const models::QwenTensorRef& weight,
                    const void* q8, float* y, std::size_t batch,
                    std::size_t m, std::size_t k) {
  if (m == 0 || k == 0 || batch == 0 || q8 == nullptr) {
    throw std::runtime_error("flint prefill w8a8 projection is empty");
  }
  if (!detail::IsNativeWmmaQuant(weight.type)) {
    throw std::runtime_error("flint prefill w8a8 weight is not native quant");
  }
  LaunchBatchedQuantGEMMPreQuantized(weight.type, weight.data, q8, y, batch,
                                     m, k, arena.stream);
  FlintCheckGpu("flint prefill w8a8 gemm failed");
}

}  // namespace

// LFM2: batched prefill for flint, batch >= 2 (caller handles 1).
tokenization::TokenId QwenGpuExecutor::ForwardFlintPromptBatch(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  CheckReset();
  if (vision_input_.Bytes() != 0) {
    throw std::runtime_error("flint vision is not supported");
  }
  const auto& config = weights_.config;
  if (!config.flint) {
    throw std::runtime_error("flint prefill called for a non-flint config");
  }
  if (arena_.d_kv_cache == nullptr) {
    throw std::runtime_error("flint requires the fp32 KV cache");
  }
  const std::size_t batch = prompt_tokens.size();
  if (batch == 0) {
    return 0;
  }
  if (batch < 2) {  // LFM2 (caller handles 1)
    throw std::runtime_error("flint batched prefill needs batch >= 2");
  }
  if (batch > arena_.GetMaxBatch()) {
    throw std::length_error("prompt chunk exceeds the GPU batch length");
  }
  const std::size_t end_pos = static_cast<std::size_t>(start_pos) + batch;
  if (end_pos > arena_.GetMaxContext()) {
    throw std::length_error("prompt exceeds the GPU context length");
  }

  const std::size_t H = config.hidden_size;
  const std::size_t attn =
      static_cast<std::size_t>(config.num_attention_heads) * config.head_dim;
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t I = config.intermediate_size;
  const float eps = config.norm_eps_pre;
  const std::uint32_t max_ctx = arena_.GetMaxContext();
  const std::uint32_t heads = config.num_attention_heads;
  const std::uint32_t kv_heads = config.num_key_value_heads;
  const std::uint32_t head_dim = config.head_dim;
  const float rope_theta = config.rope_theta;
  const std::uint32_t n_expert = config.expert_count;
  const std::uint32_t n_used = config.expert_used_count;
  const std::uint32_t expert_ff = config.expert_feed_forward_length;
  const std::size_t shexp_ff = config.expert_shared_feed_forward_length;
  const std::size_t weight_elems = FlintWeightScratchElems(config);
  const std::size_t plane = arena_.GetAttentionKvPlaneElements();
  float* v_cache = arena_.d_kv_cache + plane;
  auto* bf16 = static_cast<hip_bfloat16*>(arena_.d_scratch_bf16);
  const bool w8 =
      FlintUseW8A8ForChunk(batch) && !weights_.layers.empty() &&
      FlintAllProjectionsNative(weights_);
  if (w8) {
    const std::size_t q_projection = 2 * attn;
    const std::size_t ssm_qkv = config.SsmQkvSize();
    const std::size_t projection_width = std::max(q_projection, ssm_qkv);
    const std::size_t row = std::max<std::size_t>(
        {I, H, projection_width,
         ssm_qkv + config.ssm_inner_size +
             (2 * static_cast<std::size_t>(config.ssm_time_step_rank))});
    const std::size_t have =
        FlintArenaQ8ScratchBytes(arena_.GetMaxBatch(), row);
    const std::size_t need =
        QuantizedActivationBytes(arena_.GetMaxBatch(), I);
    if (need > have) {
      throw std::runtime_error("flint prefill q8 scratch is short");
    }
  }

  const std::size_t max_batch = arena_.GetMaxBatch();
  if (arena_.d_hidden == nullptr || arena_.d_normed == nullptr ||
      arena_.d_q == nullptr || arena_.d_k == nullptr ||
      arena_.d_v == nullptr || arena_.d_attn_out == nullptr ||
      arena_.d_ffn_gate == nullptr || arena_.d_ffn_up == nullptr ||
      arena_.d_ffn_act == nullptr || arena_.d_ffn_out == nullptr ||
      arena_.d_ssm_out == nullptr || arena_.d_prompt_tokens == nullptr ||
      arena_.d_logits == nullptr || arena_.d_alpha_buf == nullptr ||
      arena_.d_scratch_bf16 == nullptr ||
      arena_.d_weights_bf16 == nullptr ||
      arena_.d_kv_cache == nullptr) {
    throw std::runtime_error("flint prefill scratch is short");
  }
  if (max_batch * H < batch * H || max_batch * attn < batch * attn ||
      max_batch * kv < batch * kv || max_batch * I < batch * I) {
    throw std::runtime_error("flint prefill scratch is short");
  }
  if (config.is_moe || n_expert > 0) {
    if (arena_.d_moe_logits == nullptr || arena_.d_moe_ids == nullptr ||
        arena_.d_moe_weights == nullptr) {
      throw std::runtime_error("flint prefill moe scratch is short");
    }
    if (max_batch * static_cast<std::size_t>(n_expert) <
            batch * static_cast<std::size_t>(n_expert) ||
        max_batch * static_cast<std::size_t>(n_used) <
            batch * static_cast<std::size_t>(n_used)) {
      throw std::runtime_error("flint prefill moe scratch is short");
    }
  }

  // LFM2: copy tokens to arena_.d_prompt_tokens.
  std::vector<std::uint32_t> host_tokens(prompt_tokens.begin(),
                                         prompt_tokens.end());
  if (hipMemcpyAsync(arena_.d_prompt_tokens, host_tokens.data(),
                     batch * sizeof(std::uint32_t), hipMemcpyHostToDevice,
                     arena_.stream) != hipSuccess) {
    throw std::runtime_error("flint prefill token copy failed");
  }
  LaunchBatchedEmbeddingLookup(weights_.token_embd.data,
                               weights_.token_embd.type,
                               arena_.d_prompt_tokens, arena_.d_hidden, batch,
                               H, arena_.stream);
  FlintCheckGpu("flint prefill embedding failed");
  // LFM2: NO norm after the embedding.

  // DSpark taps: [token][tap][hidden] rows for the drafter, laid out
  // exactly as the Qwen prefill capture the verifier primes from.
  const std::size_t target_layer_count =
      arena_.GetTargetLayerCapture().size();
  const std::size_t prompt_capture_offset = h_prompt_hidden_.size();
  if (capture_prompt_hidden_ && target_layer_count > 0) {
    h_prompt_hidden_.resize(prompt_capture_offset +
                            batch * target_layer_count * H);
  }

  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    PrefillLayerCheckpoint(l);
    const auto& layer = weights_.layers[l];
    LaunchBatchedRMSNorm(arena_.d_hidden,
                         static_cast<const float*>(layer.attn_norm.data),
                         arena_.d_normed, bf16, batch, H, eps,
                         arena_.stream);
    FlintCheckGpu("flint prefill attn norm failed");
    if (w8) {  // LFM2 (like Glimmer, no attn_gate)
      LaunchQuantizeActivationQ8_1(bf16, arena_.d_scratch_q8_act, batch, H,
                                   arena_.stream);
      FlintCheckGpu("flint prefill attn q8 failed");
      FlintProjectQ8(arena_, layer.attn_q, arena_.d_scratch_q8_act,
                    arena_.d_q, batch, attn, H);
      FlintProjectQ8(arena_, layer.attn_k, arena_.d_scratch_q8_act,
                    arena_.d_k, batch, kv, H);
      FlintProjectQ8(arena_, layer.attn_v, arena_.d_scratch_q8_act,
                    arena_.d_v, batch, kv, H);
    } else {
      FlintProject(arena_, layer.attn_q, arena_.d_normed, bf16,
                  arena_.d_q, batch, attn, H, weight_elems);
      FlintProject(arena_, layer.attn_k, arena_.d_normed, bf16,
                  arena_.d_k, batch, kv, H, weight_elems);
      FlintProject(arena_, layer.attn_v, arena_.d_normed, bf16,
                  arena_.d_v, batch, kv, H, weight_elems);
    }
    LaunchBatchedPerHeadRMSNorm(
        arena_.d_q, static_cast<const float*>(layer.attn_q_norm.data),
        arena_.d_q, batch, heads, head_dim, eps, arena_.stream);
    LaunchBatchedPerHeadRMSNorm(
        arena_.d_k, static_cast<const float*>(layer.attn_k_norm.data),
        arena_.d_k, batch, kv_heads, head_dim, eps, arena_.stream);
    FlintCheckGpu("flint prefill qk norm failed");
    if (config.LayerUsesRope(l)) {
      LaunchBatchedRoPE(arena_.d_q, arena_.d_k, batch, heads, kv_heads,
                        head_dim, head_dim, start_pos, rope_theta,
                        arena_.stream, nullptr);  // LFM2 (NEOX, positions start_pos..)
      FlintCheckGpu("flint prefill rope failed");
    }
    FlintRequire(LaunchGlimmerBatchedWriteKv(
        arena_.d_k, arena_.d_v, arena_.d_kv_cache, v_cache, l, start_pos,
        static_cast<std::uint32_t>(batch), max_ctx, kv_heads, head_dim,
        batch * kv, plane, arena_.stream));
    bool wmma_ok = false;
    void* k_f16 = arena_.d_glimmer_kv_f16;
    void* v_f16 =
        k_f16 == nullptr ? nullptr
                         : static_cast<std::uint16_t*>(k_f16) + plane;
    if (FlintWmmaEnabled() && k_f16 != nullptr) {
      const std::size_t per_layer =
          static_cast<std::size_t>(arena_.GetMaxContext()) *
          config.num_key_value_heads * config.head_dim;
      const std::uint32_t plane_layers =
          per_layer == 0 ? 0U : static_cast<std::uint32_t>(plane / per_layer);
      FlintRequire(LaunchGlimmerCastKvFp16(
          arena_.d_kv_cache, k_f16, /*layers=*/1, arena_.GetMaxContext(),
          config.num_key_value_heads, config.head_dim,
          static_cast<std::uint32_t>(end_pos), arena_.stream, l,
          plane_layers));
      const auto ws = arena_.GetScratchView(arena_.GetMaxBatch()).ffn;
      wmma_ok = LaunchQwenWmmaAttention(
          arena_.d_q, nullptr, nullptr, nullptr, nullptr, nullptr, k_f16,
          v_f16, arena_.d_ssm_out, l, start_pos, batch, max_ctx, heads,
          kv_heads, head_dim, arena_.stream, nullptr, 0,
          /*skip_kv_write=*/true, ws.gate, ws.up, config.LayerSwaWindow(l),
          static_cast<const float*>(layer.attn_sinks.data));
      if (!wmma_ok) {
        static bool logged = false;
        if (!logged) {
          std::fprintf(stderr,
                       "flint prefill: WMMA attention unavailable, using "
                       "fp32\n");
          logged = true;
        }
      }
    }
    if (!wmma_ok) {
      FlintRequire(LaunchGlimmerPrefillAttention(
          arena_.d_q, nullptr, arena_.d_kv_cache, v_cache, arena_.d_ssm_out, l,
          start_pos, static_cast<std::uint32_t>(batch),
          config.LayerSwaWindow(l), max_ctx, heads, kv_heads, head_dim,
          batch * attn, 0, plane, batch * attn, arena_.stream,
          static_cast<const float*>(layer.attn_sinks.data)));
    }
    if (w8) {  // LFM2 (as Glimmer does for attn_output)
      LaunchQuantizeActivationQ8_1FromFp32(
          arena_.d_ssm_out, arena_.d_scratch_q8_act, batch, attn,
          arena_.stream);
      FlintCheckGpu("flint prefill attn-out q8 failed");
      FlintProjectQ8(arena_, layer.attn_output, arena_.d_scratch_q8_act,
                    arena_.d_attn_out, batch, H, attn);
    } else {
      LaunchFloatToBfloat16(arena_.d_ssm_out, bf16, batch * attn,
                            arena_.stream);
      FlintCheckGpu("flint prefill attn bf16 cast failed");
      FlintProject(arena_, layer.attn_output, arena_.d_ssm_out, bf16,
                  arena_.d_attn_out, batch, H, attn, weight_elems);
    }
    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_attn_out,
                             arena_.d_hidden, batch, H, arena_.stream);
    FlintCheckGpu("flint prefill attn residual failed");
    LaunchBatchedRMSNorm(arena_.d_hidden,
                         static_cast<const float*>(layer.ffn_norm.data),
                         arena_.d_normed, bf16, batch, H, eps,
                         arena_.stream);
    FlintCheckGpu("flint prefill ffn norm failed");
    if (!layer.is_moe) {  // LFM2 dense lead layer (Glimmer FFN verbatim)
      if (w8) {
        LaunchQuantizeActivationQ8_1(bf16, arena_.d_scratch_q8_act, batch, H,
                                     arena_.stream);
        FlintCheckGpu("flint prefill ffn q8 failed");
        if (layer.ffn_gate.type == layer.ffn_up.type) {
          LaunchBatchedDualQuantGEMMPreQuantized(
              layer.ffn_gate.type, layer.ffn_gate.data, layer.ffn_up.data,
              arena_.d_scratch_q8_act, arena_.d_ffn_gate, arena_.d_ffn_up,
              batch, I, H, arena_.stream);
          FlintCheckGpu("flint prefill dual ffn gemm failed");
        } else {
          FlintProjectQ8(arena_, layer.ffn_gate, arena_.d_scratch_q8_act,
                        arena_.d_ffn_gate, batch, I, H);
          FlintProjectQ8(arena_, layer.ffn_up, arena_.d_scratch_q8_act,
                        arena_.d_ffn_up, batch, I, H);
        }
        LaunchBatchedFusedSwiGLUQuantizeQ8_1(
            arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_scratch_q8_act,
            batch, I, arena_.stream);
        FlintCheckGpu("flint prefill swiglu q8 failed");
        FlintProjectQ8(arena_, layer.ffn_down, arena_.d_scratch_q8_act,
                      arena_.d_ffn_out, batch, H, I);
      } else {
        FlintProject(arena_, layer.ffn_gate, arena_.d_normed, bf16,
                    arena_.d_ffn_gate, batch, I, H, weight_elems);
        FlintProject(arena_, layer.ffn_up, arena_.d_normed, bf16,
                    arena_.d_ffn_up, batch, I, H, weight_elems);
        LaunchBatchedSwiGLUActivation(
            arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_ffn_act,
            batch == 1 ? nullptr : static_cast<void*>(bf16),
            batch * I, arena_.stream);
        FlintCheckGpu("flint prefill swiglu failed");
        FlintProject(arena_, layer.ffn_down, arena_.d_ffn_act, bf16,
                    arena_.d_ffn_out, batch, H, I, weight_elems);
      }
    } else {  // LFM2 MoE layer
      if (w8) {
        LaunchQuantizeActivationQ8_1(bf16, arena_.d_scratch_q8_act, batch, H,
                                     arena_.stream);
        FlintCheckGpu("flint prefill shexp q8 failed");
        if (layer.ffn_gate_shexp.type == layer.ffn_up_shexp.type) {
          LaunchBatchedDualQuantGEMMPreQuantized(
              layer.ffn_gate_shexp.type, layer.ffn_gate_shexp.data,
              layer.ffn_up_shexp.data, arena_.d_scratch_q8_act,
              arena_.d_ffn_gate, arena_.d_ffn_up, batch, shexp_ff, H,
              arena_.stream);
          FlintCheckGpu("flint prefill dual shexp gemm failed");
        } else {
          FlintProjectQ8(arena_, layer.ffn_gate_shexp, arena_.d_scratch_q8_act,
                        arena_.d_ffn_gate, batch, shexp_ff, H);
          FlintProjectQ8(arena_, layer.ffn_up_shexp, arena_.d_scratch_q8_act,
                        arena_.d_ffn_up, batch, shexp_ff, H);
        }
        LaunchBatchedFusedSwiGLUQuantizeQ8_1(
            arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_scratch_q8_act, batch,
            shexp_ff, arena_.stream);
        FlintCheckGpu("flint prefill shexp swiglu q8 failed");
        FlintProjectQ8(arena_, layer.ffn_down_shexp, arena_.d_scratch_q8_act,
                      arena_.d_attn_out, batch, H, shexp_ff);
      } else {
        FlintProject(arena_, layer.ffn_gate_shexp, arena_.d_normed, bf16,
                    arena_.d_ffn_gate, batch, shexp_ff, H, weight_elems);
        FlintProject(arena_, layer.ffn_up_shexp, arena_.d_normed, bf16,
                    arena_.d_ffn_up, batch, shexp_ff, H, weight_elems);
        LaunchBatchedSwiGLUActivation(
            arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_ffn_act,
            batch == 1 ? nullptr : static_cast<void*>(bf16), batch * shexp_ff,
            arena_.stream);
        FlintCheckGpu("flint prefill shexp swiglu failed");
        FlintProject(arena_, layer.ffn_down_shexp, arena_.d_ffn_act, bf16,
                    arena_.d_attn_out, batch, H, shexp_ff, weight_elems);
      }
      FlintRequire(LaunchLfm2MoeRouteBatch(
          arena_.d_normed,
          static_cast<const float*>(layer.ffn_gate_inp.data),
          static_cast<const float*>(layer.exp_probs_b.data),
          static_cast<std::uint32_t>(H), n_expert,
          static_cast<std::uint32_t>(batch), n_used, arena_.d_moe_logits,
          arena_.d_moe_ids, arena_.d_moe_weights, arena_.stream));
      if (LaunchKatMoeRoutedGrouped(
              static_cast<int>(layer.ffn_gate_exps.type),
              layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
              layer.ffn_down_exps.data, arena_.d_normed, arena_.d_moe_ids,
              arena_.d_moe_weights, static_cast<int>(batch),
              static_cast<int>(n_used), static_cast<int>(n_expert),
              static_cast<int>(H), static_cast<int>(expert_ff),
              arena_.d_ffn_out, arena_.stream) != 0) {
        throw std::runtime_error("flint grouped prefill experts failed");
      }
      LaunchBatchedResidualAdd(arena_.d_ffn_out, arena_.d_attn_out,
                               arena_.d_ffn_out, batch, H, arena_.stream);
      FlintCheckGpu("flint prefill shexp residual failed");
    }
    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_ffn_out,
                             arena_.d_hidden, batch, H, arena_.stream);
    FlintCheckGpu("flint prefill ffn residual failed");
    if (capture_prompt_hidden_ && target_layer_count > 0) {
      if (const auto tap = arena_.GetTargetLayerCaptureIndex(l);
          tap.has_value()) {
        float* destination = h_prompt_hidden_.data() +
                             prompt_capture_offset + (*tap * H);
        if (hipMemcpy2DAsync(
                destination, target_layer_count * H * sizeof(float),
                arena_.d_hidden, H * sizeof(float), H * sizeof(float),
                batch, hipMemcpyDeviceToHost, arena_.stream) != hipSuccess) {
          throw std::runtime_error("flint target-layer tap copy failed");
        }
      }
    }
  }

  // LFM2: tail exactly as Glimmer's but WITHOUT the scale/softcap call.
  last_hidden_offset_ = (batch - 1) * H;
  if (!compute_logits) {
    if (hipStreamSynchronize(arena_.stream) != hipSuccess) {
      throw std::runtime_error("flint prefill synchronize failed");
    }
    return 0;
  }
  const float* final_hidden = arena_.d_hidden + last_hidden_offset_;
  LaunchRMSNorm(final_hidden,
                static_cast<const float*>(weights_.output_norm.data),
                arena_.d_normed, H, eps, arena_.stream);
  LaunchGEMV(weights_.output.data, weights_.output.type, arena_.d_normed,
             arena_.d_logits, config.vocab_size, H, arena_.stream);
  // LFM2: NO scale/softcap call.
  LaunchBatchedGPUArgmax(
      arena_.d_logits, reinterpret_cast<std::uint32_t*>(arena_.d_alpha_buf),
      1, config.vocab_size,
      std::span<float>(arena_.d_ffn_out, arena_.GetMaxBatch() * H),
      arena_.stream);
  std::uint32_t next_token_id = 0;
  if (hipMemcpyAsync(&next_token_id, arena_.d_alpha_buf,
                     sizeof(std::uint32_t), hipMemcpyDeviceToHost,
                     arena_.stream) != hipSuccess ||
      hipStreamSynchronize(arena_.stream) != hipSuccess) {
    throw std::runtime_error("flint prefill token read failed");
  }
  return CheckedSampleToken(next_token_id, config.vocab_size);
}

void QwenGpuExecutor::MaterializeFlintRowLogits(std::size_t row) {
  CheckReset();
  const auto& config = weights_.config;
  if (!config.flint) {
    throw std::runtime_error("flint row logits called for a non-flint config");
  }
  if (row >= arena_.GetMaxBatch()) {
    throw std::runtime_error("flint row logits row is past the batch");
  }
  const std::size_t hidden = config.hidden_size;
  const float* src = arena_.d_hidden + row * hidden;
  LaunchRMSNorm(src, static_cast<const float*>(weights_.output_norm.data),
                arena_.d_normed, hidden, config.norm_eps_pre, arena_.stream);
  LaunchGEMV(weights_.output.data, weights_.output.type, arena_.d_normed,
             arena_.d_logits, config.vocab_size, hidden, arena_.stream);
  // LFM2: NO scale/softcap call.
  if (hipStreamSynchronize(arena_.stream) != hipSuccess) {
    throw std::runtime_error("flint row logits synchronize failed");
  }
}

}  // namespace superroc::hip
#endif  // defined(ENGINE_ENABLE_HIP)
