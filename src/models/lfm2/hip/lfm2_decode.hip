// LFM2 decode step (SPEC2, lane LFD): embedding, per-layer RMSNorm,
// short-conv operator or GQA attention with fp32 KV, residual, dense SwiGLU
// or sigmoid-routed FP4 MoE, residual; optional tied-head logits + argmax.
// Prefill v1 (ForwardLfm2PromptChunk) is the single-token loop over this step.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/lfm2/hip/lfm2_decode.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/rocm_fp4/superroc_fp4_gemv_q8.hpp"
#include "src/core/rocm_fp4/superroc_fp4_quant.hpp"  // LFM2 (SUPERROC_FP4_OK)
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/fp4_launch.hpp"  // LFM2 (LKF: shared q/k/v quant)
#include "src/models/glimmer/hip/glimmer_decode.hpp"  // LFM2 (fp32 KV + attention)
#include "src/models/lfm2/hip/lfm2_kernels.hpp"  // LFM2
#include "src/models/lfm2/hip/lfm2_attention.hpp"  // LFM2
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/hip/ops/token.hpp"  // LFM2 (LaunchRoPE)
#include "src/models/qwen/vision/device_input.hpp"

namespace superroc::hip {
namespace {

// LFM2: Glimmer's Require pattern, for both launcher status types.
void Require(const Lfm2LaunchStatus& status) {
  if (!status.ok) {
    throw std::runtime_error(status.error != nullptr ? status.error
                                                    : "lfm2moe launch failed");
  }
}

void Require(const GlimmerLaunchStatus& status) {
  if (!status.ok) {
    throw std::runtime_error(status.error != nullptr ? status.error
                                                    : "glimmer launch failed");
  }
}

}  // namespace

void ExecuteLfm2DecodeStep(QwenGpuArena& arena,
                           const models::QwenModelWeights& weights,
                           tokenization::TokenId token_id, std::uint32_t pos,
                           bool compute_logits,
                           models::qwen::vision::DeviceInput* image_input) {
  if (image_input != nullptr && image_input->Bytes() != 0) {
    throw std::runtime_error("lfm2moe vision is not supported");
  }
  const auto& config = weights.config;
  if (!config.lfm2) {
    throw std::runtime_error("lfm2moe decode called for a non-lfm2 config");
  }
  if (arena.d_kv_cache == nullptr && arena.d_attention_kv_f16 == nullptr) {
    throw std::runtime_error("lfm2moe requires a KV cache");  // LFM2 (SPEC3)
  }
  if (arena.d_ssm_conv_state == nullptr) {
    throw std::runtime_error("lfm2moe requires ssm conv state");
  }
  if (pos >= arena.GetMaxContext()) {
    throw std::runtime_error("lfm2moe token position exceeds the GPU context");
  }

  auto scratch = arena.GetScratchView();
  const std::size_t hidden = config.hidden_size;
  const std::size_t attn =
      static_cast<std::size_t>(config.num_attention_heads) * config.head_dim;
  const std::size_t kv = static_cast<std::size_t>(config.num_key_value_heads) *
                         config.head_dim;
  const std::size_t intermediate = config.intermediate_size;
  if (scratch.decode.hidden.size() < hidden ||
      scratch.decode.normed.size() < hidden ||
      scratch.attention.q.size() < attn || scratch.attention.k.size() < kv ||
      scratch.attention.v.size() < kv ||
      scratch.ssm.out.size() < std::max(hidden, attn) ||
      scratch.attention.output.size() < hidden ||
      scratch.ffn.activation.size() <
          std::max(intermediate, 3 * hidden) ||
      scratch.ffn.out.size() < hidden ||
      scratch.decode.prompt_tokens.size() < 2) {
    throw std::runtime_error("lfm2moe decode scratch is short");
  }

  const std::uint32_t params[2] = {token_id, pos};
  if (hipMemcpyAsync(scratch.decode.prompt_tokens.data(), params,
                     sizeof(params), hipMemcpyHostToDevice,
                     arena.stream) != hipSuccess) {
    throw std::runtime_error("lfm2moe token copy failed");
  }

  LaunchEmbeddingLookup(weights.token_embd.data, weights.token_embd.type,
                        scratch.decode.prompt_tokens.data(),
                        scratch.decode.hidden.data(), hidden, arena.stream);
  // LFM2: NO norm after the embedding.

  const std::size_t plane = arena.GetAttentionKvPlaneElements();
  // LFM2 (SPEC3): fp16 token-major plane when the arena has one, else fp32.
  const bool kv_f16 = arena.d_attention_kv_f16 != nullptr;
  float* v_cache = kv_f16 ? nullptr : arena.d_kv_cache + plane;  // LFM2
  auto* k_f16 = static_cast<std::uint16_t*>(arena.d_attention_kv_f16);
  std::uint16_t* v_f16 = kv_f16 ? k_f16 + plane : nullptr;
  const float eps = config.norm_eps_pre;  // LFM2
  const std::uint32_t max_ctx = arena.GetMaxContext();
  const std::uint32_t heads = config.num_attention_heads;
  const std::uint32_t kv_heads = config.num_key_value_heads;
  const std::uint32_t head_dim = config.head_dim;
  const std::uint32_t rotary_dim = config.rotary_dim;
  const float rope_theta = config.rope_theta;
  const std::uint32_t n_expert = config.expert_count;
  const std::uint32_t n_used = config.expert_used_count;
  const std::uint32_t expert_ff = config.expert_feed_forward_length;
  const std::uint32_t l_cache = config.shortconv_l_cache;
  // LFM2 (LKF): normed holds attn_norm(hidden) at the top of every layer.
  // Layer 0 norms here; every residual add below is fused with the RMSNorm
  // that follows it (ffn_norm, the next layer's attn_norm, output_norm).
  if (config.num_layers != 0) {
    LaunchRMSNorm(scratch.decode.hidden.data(),
                  static_cast<const float*>(weights.layers[0].attn_norm.data),
                  scratch.decode.normed.data(), hidden, eps, arena.stream);
  }
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const auto& layer = weights.layers[l];
    if (layer.is_shortconv) {  // LFM2
      // Rows B | C | X of the in-projection land in ffn.activation [3H].
      LaunchGEMV(layer.shortconv_in_proj.data, layer.shortconv_in_proj.type,
                 scratch.decode.normed.data(), scratch.ffn.activation.data(),
                 3 * hidden, hidden, arena.stream);
      Require(LaunchLfm2ShortConvDecode(
          scratch.ffn.activation.data(),
          static_cast<const float*>(layer.shortconv_conv.data),
          arena.d_ssm_conv_state +
              config.Lfm2KindIndex(l) * config.Lfm2ConvStateFloats(),
          scratch.ssm.out.data(), static_cast<std::uint32_t>(hidden), l_cache,
          arena.stream));
      LaunchGEMV(layer.shortconv_out_proj.data, layer.shortconv_out_proj.type,
                 scratch.ssm.out.data(), scratch.attention.output.data(),
                 hidden, hidden, arena.stream);
    } else {  // LFM2 attention layer
      // LFM2 (LKF): with ROCmFP4 q/k/v, quantise normed ONCE into the
      // attention slot and run the three matvecs against it -- the same
      // quant and matvec kernels LaunchGEMV runs per matrix, so bit-identical.
      // Any refusal falls back to LaunchGEMV for that matrix.
      const bool shared_qkv =
          IsRocmFp4(layer.attn_q.type) && IsRocmFp4(layer.attn_k.type) &&
          IsRocmFp4(layer.attn_v.type) && RocmFp4ShapeOk(attn, hidden) &&
          RocmFp4ShapeOk(kv, hidden) &&
          PrepareRocmFp4Act(scratch.decode.normed.data(), hidden,
                            SUPERROC_FP4_Q8_SLOT_ATTN_NORM, arena.stream);
      if (!shared_qkv ||
          !TryLaunchRocmFp4GemvSlot(layer.attn_q.type, layer.attn_q.data,
                                    SUPERROC_FP4_Q8_SLOT_ATTN_NORM,
                                    scratch.attention.q.data(), attn, hidden,
                                    arena.stream)) {
        LaunchGEMV(layer.attn_q.data, layer.attn_q.type,
                   scratch.decode.normed.data(), scratch.attention.q.data(),
                   attn, hidden, arena.stream);
      }
      if (!shared_qkv ||
          !TryLaunchRocmFp4GemvSlot(layer.attn_k.type, layer.attn_k.data,
                                    SUPERROC_FP4_Q8_SLOT_ATTN_NORM,
                                    scratch.attention.k.data(), kv, hidden,
                                    arena.stream)) {
        LaunchGEMV(layer.attn_k.data, layer.attn_k.type,
                   scratch.decode.normed.data(), scratch.attention.k.data(),
                   kv, hidden, arena.stream);
      }
      if (!shared_qkv ||
          !TryLaunchRocmFp4GemvSlot(layer.attn_v.type, layer.attn_v.data,
                                    SUPERROC_FP4_Q8_SLOT_ATTN_NORM,
                                    scratch.attention.v.data(), kv, hidden,
                                    arena.stream)) {
        LaunchGEMV(layer.attn_v.data, layer.attn_v.type,
                   scratch.decode.normed.data(), scratch.attention.v.data(),
                   kv, hidden, arena.stream);
      }
      LaunchPerHeadRMSNorm(
          scratch.attention.q.data(),
          static_cast<const float*>(layer.attn_q_norm.data),
          scratch.attention.q.data(), heads, head_dim, eps, arena.stream);
      LaunchPerHeadRMSNorm(
          scratch.attention.k.data(),
          static_cast<const float*>(layer.attn_k_norm.data),
          scratch.attention.k.data(), kv_heads, head_dim, eps, arena.stream);
      const std::uint32_t* d_pos =
          scratch.decode.prompt_tokens.data() + 1;  // LFM2 (device-side pos)
      LaunchRoPE(scratch.attention.q.data(), scratch.attention.k.data(), heads,
                 kv_heads, head_dim, rotary_dim, d_pos, rope_theta,
                 arena.stream, nullptr);  // LFM2 (NEOX)
      if (kv_f16) {  // LFM2 (SPEC3)
        Require(LaunchLfm2WriteKvDevF16(
            scratch.attention.k.data(), scratch.attention.v.data(), k_f16,
            v_f16, config.Lfm2KindIndex(l), d_pos, max_ctx, kv_heads,
            head_dim, arena.stream));
        Require(LaunchLfm2DecodeAttentionF16(
            scratch.attention.q.data(), k_f16, v_f16, scratch.ssm.out.data(),
            config.Lfm2KindIndex(l), d_pos, max_ctx, heads, kv_heads,
            head_dim, 32, arena.d_lfm2_attn_partial, arena.stream));
      } else {
        Require(LaunchLfm2WriteKvDev(
            scratch.attention.k.data(), scratch.attention.v.data(),
            arena.d_kv_cache, v_cache, config.Lfm2KindIndex(l), d_pos,
            max_ctx, kv_heads, head_dim, arena.stream));
        Require(LaunchLfm2DecodeAttention(
            scratch.attention.q.data(), arena.d_kv_cache, v_cache,
            scratch.ssm.out.data(), config.Lfm2KindIndex(l), d_pos, max_ctx,
            heads, kv_heads, head_dim, 32, arena.d_lfm2_attn_partial,
            arena.stream));  // LFM2 (split-K flash-decoding)
      }
      LaunchGEMV(layer.attn_output.data, layer.attn_output.type,
                 scratch.ssm.out.data(), scratch.attention.output.data(),
                 hidden, attn, arena.stream);
    }
    Require(LaunchLfm2AddRMSNorm(  // LFM2 (LKF: residual + ffn_norm)
        scratch.decode.hidden.data(), scratch.attention.output.data(),
        static_cast<const float*>(layer.ffn_norm.data),
        scratch.decode.normed.data(), static_cast<std::uint32_t>(hidden), eps,
        arena.stream));
    if (!layer.is_moe) {  // LFM2 dense lead layer
      LaunchFusedSwiGLUGEMV(layer.ffn_gate.data, layer.ffn_gate.type,
                            layer.ffn_up.data, layer.ffn_up.type,
                            scratch.decode.normed.data(),
                            scratch.ffn.activation.data(), intermediate, hidden,
                            arena.stream);
      LaunchGEMV(layer.ffn_down.data, layer.ffn_down.type,
                 scratch.ffn.activation.data(), scratch.ffn.out.data(), hidden,
                 intermediate, arena.stream);
    } else {  // LFM2 MoE layer
      Require(LaunchLfm2MoeRoute(
          scratch.decode.normed.data(),
          static_cast<const float*>(layer.ffn_gate_inp.data),
          static_cast<const float*>(layer.exp_probs_b.data),
          static_cast<std::uint32_t>(hidden), n_expert, n_used,
          arena.d_moe_logits, arena.d_moe_ids, arena.d_moe_weights,
          arena.stream));
      if (superroc_fp4_expert_ffn_dev(
              static_cast<int>(layer.ffn_gate_exps.type),
              layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
              layer.ffn_down_exps.data, scratch.decode.normed.data(),
              arena.d_moe_ids, arena.d_moe_weights, static_cast<int>(n_used),
              static_cast<int>(n_expert), static_cast<int>(hidden),
              static_cast<int>(expert_ff), 0.0F, scratch.ffn.out.data(),
              arena.stream) != SUPERROC_FP4_OK) {
        throw std::runtime_error("lfm2moe decode experts failed");
      }
    }
    // LFM2 (LKF): residual + the next RMSNorm (next layer's attn_norm, or
    // output_norm after the last layer when logits are wanted).
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
    // DSpark tap: the layer output is the hidden state the drafter's
    // feature-fusion encoder consumes (same point as the Qwen paths).
    if (const auto tap = arena.GetTargetLayerCaptureIndex(l);
        tap.has_value()) {
      if (hipMemcpyAsync(arena.d_target_layer_features + (*tap * hidden),
                         scratch.decode.hidden.data(), hidden * sizeof(float),
                         hipMemcpyDeviceToDevice, arena.stream) != hipSuccess) {
        throw std::runtime_error("lfm2 target-layer tap copy failed");
      }
    }
  }

  if (!compute_logits) {
    return;
  }
  // LFM2 (LKF): normed already holds output_norm(hidden).
  LaunchGEMV(weights.output.data, weights.output.type,
             scratch.decode.normed.data(), scratch.decode.logits.data(),
             config.vocab_size, hidden, arena.stream);
  LaunchBatchedGPUArgmax(scratch.decode.logits.data(),
                         scratch.decode.sampled_token.data(), 1,
                         config.vocab_size, scratch.ffn.out, arena.stream);
}

// LFM2 prefill v1: token loop over the decode step (batched prefill later).
tokenization::TokenId QwenGpuExecutor::ForwardLfm2PromptChunk(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  CheckReset();
  if (vision_input_.Bytes() != 0) {
    throw std::runtime_error("lfm2moe vision is not supported");
  }
  const auto& config = weights_.config;
  if (!config.lfm2) {
    throw std::runtime_error("lfm2moe prefill called for a non-lfm2 config");
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
  if (batch >= 2) {  // LFM2
    return ForwardLfm2PromptBatch(prompt_tokens, start_pos, compute_logits);  // LFM2
  }  // LFM2
  last_hidden_offset_ = 0;  // LFM2 (single-token scratch)
  for (std::size_t i = 0; i < batch; ++i) {
    const std::uint32_t pos = start_pos + static_cast<std::uint32_t>(i);
    ExecuteLfm2DecodeStep(arena_, weights_, prompt_tokens[i], pos,
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
  // LFM2: read the sampled token exactly as decode.cpp's Glimmer branch does.
  auto scratch = arena_.GetScratchView();
  auto* d_out_token = scratch.decode.sampled_token.data();
  std::uint32_t next_token_id = 0;
  HIP_CHECK(hipMemcpyAsync(&next_token_id, d_out_token, sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return CheckedSampleToken(next_token_id, config.vocab_size);
}

}  // namespace superroc::hip
#endif  // defined(ENGINE_ENABLE_HIP)
