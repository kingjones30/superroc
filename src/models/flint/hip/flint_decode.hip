// Flint decode step: embedding, per-layer RMSNorm, GQA attention with fp32 KV
// and learned sinks, residual, dense SwiGLU or ungated shexp + sigmoid-routed
// FP4 MoE, residual; optional untied-head logits + argmax.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/flint/hip/flint_decode.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/rocm_fp4/gufo_fp4_gemv_q8.hpp"
#include "src/core/rocm_fp4/gufo_fp4_quant.hpp"
#include "src/models/flint/hip/flint_attention.hpp"
#include "src/models/glimmer/hip/glimmer_decode.hpp"
#include "src/models/lfm2/hip/lfm2_kernels.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/fp4_launch.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/vision/device_input.hpp"

namespace gufo::hip {
namespace {

void Require(const Lfm2LaunchStatus& status) {
  if (!status.ok) {
    throw std::runtime_error(status.error != nullptr ? status.error
                                                     : "flint launch failed");
  }
}

void Require(const GlimmerLaunchStatus& status) {
  if (!status.ok) {
    throw std::runtime_error(status.error != nullptr ? status.error
                                                     : "glimmer launch failed");
  }
}

void Require(const FlintLaunchStatus& status) {
  if (!status.ok) {
    throw std::runtime_error(status.error != nullptr ? status.error
                                                     : "flint launch failed");
  }
}

bool FlintSplitKEnabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("FLINT_SPLITK");
    return env == nullptr || std::strcmp(env, "0") != 0;
  }();
  return enabled;
}

}  // namespace

void ExecuteFlintDecodeStep(QwenGpuArena& arena,
                            const models::QwenModelWeights& weights,
                            tokenization::TokenId token_id, std::uint32_t pos,
                            bool compute_logits,
                            models::qwen::vision::DeviceInput* image_input) {
  if (image_input != nullptr && image_input->Bytes() != 0) {
    throw std::runtime_error("flint vision is not supported");
  }
  const auto& config = weights.config;
  if (!config.flint) {
    throw std::runtime_error("flint decode called for a non-flint config");
  }
  if (arena.d_kv_cache == nullptr) {
    throw std::runtime_error("flint requires the fp32 KV cache");
  }
  if (pos >= arena.GetMaxContext()) {
    throw std::runtime_error("flint token position exceeds the GPU context");
  }

  auto scratch = arena.GetScratchView();
  const std::size_t hidden = config.hidden_size;
  const std::size_t attn =
      static_cast<std::size_t>(config.num_attention_heads) * config.head_dim;
  const std::size_t kv = static_cast<std::size_t>(config.num_key_value_heads) *
                         config.head_dim;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t shexp_ff = config.expert_shared_feed_forward_length;
  if (scratch.decode.hidden.size() < hidden ||
      scratch.decode.normed.size() < hidden ||
      scratch.attention.q.size() < attn || scratch.attention.k.size() < kv ||
      scratch.attention.v.size() < kv ||
      scratch.ssm.out.size() < std::max(hidden, attn) ||
      scratch.attention.output.size() < hidden ||
      scratch.ffn.activation.size() < std::max(intermediate, shexp_ff) ||
      scratch.ffn.out.size() < hidden ||
      scratch.decode.prompt_tokens.size() < 2) {
    throw std::runtime_error("flint decode scratch is short");
  }

  const std::uint32_t params[2] = {token_id, pos};
  if (hipMemcpyAsync(scratch.decode.prompt_tokens.data(), params,
                     sizeof(params), hipMemcpyHostToDevice,
                     arena.stream) != hipSuccess) {
    throw std::runtime_error("flint token copy failed");
  }

  LaunchEmbeddingLookup(weights.token_embd.data, weights.token_embd.type,
                        scratch.decode.prompt_tokens.data(),
                        scratch.decode.hidden.data(), hidden, arena.stream);

  const std::size_t plane = arena.GetAttentionKvPlaneElements();
  float* v_cache = arena.d_kv_cache + plane;
  const float eps = config.norm_eps_pre;
  const std::uint32_t max_ctx = arena.GetMaxContext();
  const std::uint32_t heads = config.num_attention_heads;
  const std::uint32_t kv_heads = config.num_key_value_heads;
  const std::uint32_t head_dim = config.head_dim;
  const std::uint32_t rotary_dim = config.rotary_dim;
  const float rope_theta = config.rope_theta;
  const std::uint32_t n_expert = config.expert_count;
  const std::uint32_t n_used = config.expert_used_count;
  const std::uint32_t expert_ff = config.expert_feed_forward_length;
  if (config.num_layers != 0) {
    LaunchRMSNorm(scratch.decode.hidden.data(),
                  static_cast<const float*>(weights.layers[0].attn_norm.data),
                  scratch.decode.normed.data(), hidden, eps, arena.stream);
  }
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const auto& layer = weights.layers[l];
    const bool shared_qkv =
        IsRocmFp4(layer.attn_q.type) && IsRocmFp4(layer.attn_k.type) &&
        IsRocmFp4(layer.attn_v.type) && RocmFp4ShapeOk(attn, hidden) &&
        RocmFp4ShapeOk(kv, hidden) &&
        PrepareRocmFp4Act(scratch.decode.normed.data(), hidden,
                          GUFO_FP4_Q8_SLOT_ATTN_NORM, arena.stream);
    if (!shared_qkv ||
        !TryLaunchRocmFp4GemvSlot(layer.attn_q.type, layer.attn_q.data,
                                  GUFO_FP4_Q8_SLOT_ATTN_NORM,
                                  scratch.attention.q.data(), attn, hidden,
                                  arena.stream)) {
      LaunchGEMV(layer.attn_q.data, layer.attn_q.type,
                 scratch.decode.normed.data(), scratch.attention.q.data(),
                 attn, hidden, arena.stream);
    }
    if (!shared_qkv ||
        !TryLaunchRocmFp4GemvSlot(layer.attn_k.type, layer.attn_k.data,
                                  GUFO_FP4_Q8_SLOT_ATTN_NORM,
                                  scratch.attention.k.data(), kv, hidden,
                                  arena.stream)) {
      LaunchGEMV(layer.attn_k.data, layer.attn_k.type,
                 scratch.decode.normed.data(), scratch.attention.k.data(),
                 kv, hidden, arena.stream);
    }
    if (!shared_qkv ||
        !TryLaunchRocmFp4GemvSlot(layer.attn_v.type, layer.attn_v.data,
                                  GUFO_FP4_Q8_SLOT_ATTN_NORM,
                                  scratch.attention.v.data(), kv, hidden,
                                  arena.stream)) {
      LaunchGEMV(layer.attn_v.data, layer.attn_v.type,
                 scratch.decode.normed.data(), scratch.attention.v.data(),
                 kv, hidden, arena.stream);
    }
    LaunchPerHeadRMSNorm(scratch.attention.q.data(),
                         static_cast<const float*>(layer.attn_q_norm.data),
                         scratch.attention.q.data(), heads, head_dim, eps,
                         arena.stream);
    LaunchPerHeadRMSNorm(scratch.attention.k.data(),
                         static_cast<const float*>(layer.attn_k_norm.data),
                         scratch.attention.k.data(), kv_heads, head_dim, eps,
                         arena.stream);
    const std::uint32_t* d_pos = scratch.decode.prompt_tokens.data() + 1;
    if (config.LayerUsesRope(l)) {
      LaunchRoPE(scratch.attention.q.data(), scratch.attention.k.data(), heads,
                 kv_heads, head_dim, rotary_dim, d_pos, rope_theta,
                 arena.stream, nullptr);
    }
    Require(LaunchGlimmerWriteKv(
        scratch.attention.k.data(), scratch.attention.v.data(),
        arena.d_kv_cache, v_cache, l, pos, max_ctx, kv_heads, head_dim, kv,
        plane, arena.stream));
    if (FlintSplitKEnabled()) {
      const std::uint32_t n_splits =
          config.LayerSwaWindow(l) == 0U ? 32U : 16U;
      Require(LaunchFlintDecodeAttention(
          scratch.attention.q.data(), arena.d_kv_cache, v_cache,
          scratch.ssm.out.data(), l, d_pos, config.LayerSwaWindow(l), max_ctx,
          heads, kv_heads, head_dim, n_splits, arena.d_lfm2_attn_partial,
          static_cast<const float*>(layer.attn_sinks.data), arena.stream));
    } else {
      Require(LaunchGlimmerWindowAttention(
          scratch.attention.q.data(), nullptr, arena.d_kv_cache, v_cache,
          scratch.ssm.out.data(), l, pos, config.LayerSwaWindow(l), max_ctx,
          heads, kv_heads, head_dim, attn, 0, plane, scratch.ssm.out.size(),
          arena.stream, static_cast<const float*>(layer.attn_sinks.data)));
    }
    LaunchGEMV(layer.attn_output.data, layer.attn_output.type,
               scratch.ssm.out.data(), scratch.attention.output.data(), hidden,
               attn, arena.stream);
    Require(LaunchLfm2AddRMSNorm(
        scratch.decode.hidden.data(), scratch.attention.output.data(),
        static_cast<const float*>(layer.ffn_norm.data),
        scratch.decode.normed.data(), static_cast<std::uint32_t>(hidden), eps,
        arena.stream));
    if (!layer.is_moe) {
      LaunchFusedSwiGLUGEMV(layer.ffn_gate.data, layer.ffn_gate.type,
                            layer.ffn_up.data, layer.ffn_up.type,
                            scratch.decode.normed.data(),
                            scratch.ffn.activation.data(), intermediate, hidden,
                            arena.stream);
      LaunchGEMV(layer.ffn_down.data, layer.ffn_down.type,
                 scratch.ffn.activation.data(), scratch.ffn.out.data(), hidden,
                 intermediate, arena.stream);
    } else {
      LaunchFusedSwiGLUGEMV(layer.ffn_gate_shexp.data, layer.ffn_gate_shexp.type,
                            layer.ffn_up_shexp.data, layer.ffn_up_shexp.type,
                            scratch.decode.normed.data(),
                            scratch.ffn.activation.data(), shexp_ff, hidden,
                            arena.stream);
      LaunchGEMV(layer.ffn_down_shexp.data, layer.ffn_down_shexp.type,
                 scratch.ffn.activation.data(), scratch.attention.output.data(),
                 hidden, shexp_ff, arena.stream);
      Require(LaunchLfm2MoeRoute(
          scratch.decode.normed.data(),
          static_cast<const float*>(layer.ffn_gate_inp.data),
          static_cast<const float*>(layer.exp_probs_b.data),
          static_cast<std::uint32_t>(hidden), n_expert, n_used,
          arena.d_moe_logits, arena.d_moe_ids, arena.d_moe_weights,
          arena.stream));
      if (gufo_fp4_expert_ffn_dev(
              static_cast<int>(layer.ffn_gate_exps.type),
              layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
              layer.ffn_down_exps.data, scratch.decode.normed.data(),
              arena.d_moe_ids, arena.d_moe_weights, static_cast<int>(n_used),
              static_cast<int>(n_expert), static_cast<int>(hidden),
              static_cast<int>(expert_ff), 0.0F, scratch.ffn.out.data(),
              arena.stream) != GUFO_FP4_OK) {
        throw std::runtime_error("flint decode experts failed");
      }
      LaunchResidualAdd(scratch.ffn.out.data(), scratch.attention.output.data(),
                        scratch.ffn.out.data(), hidden, arena.stream);
    }
    const float* next_norm =
        l + 1U < config.num_layers
            ? static_cast<const float*>(weights.layers[l + 1U].attn_norm.data)
            : (compute_logits
                   ? static_cast<const float*>(weights.output_norm.data)
                   : nullptr);
    if (next_norm != nullptr) {
      Require(LaunchLfm2AddRMSNorm(
          scratch.decode.hidden.data(), scratch.ffn.out.data(), next_norm,
          scratch.decode.normed.data(), static_cast<std::uint32_t>(hidden),
          eps, arena.stream));
    } else {
      LaunchResidualAdd(scratch.decode.hidden.data(), scratch.ffn.out.data(),
                        scratch.decode.hidden.data(), hidden, arena.stream);
    }
    if (const auto tap = arena.GetTargetLayerCaptureIndex(l); tap.has_value()) {
      if (hipMemcpyAsync(arena.d_target_layer_features + (*tap * hidden),
                         scratch.decode.hidden.data(), hidden * sizeof(float),
                         hipMemcpyDeviceToDevice, arena.stream) != hipSuccess) {
        throw std::runtime_error("flint target-layer tap copy failed");
      }
    }
  }

  if (!compute_logits) {
    return;
  }
  LaunchGEMV(weights.output.data, weights.output.type,
             scratch.decode.normed.data(), scratch.decode.logits.data(),
             config.vocab_size, hidden, arena.stream);
  LaunchBatchedGPUArgmax(scratch.decode.logits.data(),
                         scratch.decode.sampled_token.data(), 1,
                         config.vocab_size, scratch.ffn.out, arena.stream);
}

tokenization::TokenId QwenGpuExecutor::ForwardFlintPromptChunk(
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
  const std::size_t batch = prompt_tokens.size();
  if (batch == 0) {
    return 0;
  }
  if (batch > arena_.GetMaxBatch()) {
    throw std::length_error("prompt chunk exceeds the GPU batch length");
  }
  const std::size_t end_pos = static_cast<std::size_t>(start_pos) + batch;
  if (end_pos > arena_.GetMaxContext()) {
    throw std::length_error("prompt exceeds the GPU context length");
  }
  if (batch >= 2) {
    return ForwardFlintPromptBatch(prompt_tokens, start_pos, compute_logits);
  }
  last_hidden_offset_ = 0;
  for (std::size_t i = 0; i < batch; ++i) {
    const std::uint32_t pos = start_pos + static_cast<std::uint32_t>(i);
    ExecuteFlintDecodeStep(arena_, weights_, prompt_tokens[i], pos,
                           compute_logits && i + 1 == batch, &vision_input_);
    if (capture_prompt_hidden_) {
      const auto hidden = CopyLastHidden();
      h_prompt_hidden_.insert(h_prompt_hidden_.end(), hidden.begin(),
                              hidden.end());
    }
  }
  if (!compute_logits) {
    return 0;
  }
  auto scratch = arena_.GetScratchView();
  auto* d_out_token = scratch.decode.sampled_token.data();
  std::uint32_t next_token_id = 0;
  HIP_CHECK(hipMemcpyAsync(&next_token_id, d_out_token, sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return CheckedSampleToken(next_token_id, config.vocab_size);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
