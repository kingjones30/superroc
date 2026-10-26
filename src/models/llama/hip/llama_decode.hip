#if defined(ENGINE_ENABLE_HIP)
#include "src/models/llama/hip/llama_decode.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/flint/hip/flint_attention.hpp"
#include "src/models/glimmer/hip/glimmer_decode.hpp"
#include "src/models/lfm2/hip/lfm2_kernels.hpp"
#include "src/models/llama/hip/llama_rope.hpp"
#include "src/models/qwen/hip/executor.hpp"
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
                                                     : "llama launch failed");
  }
}

void Require(const GlimmerLaunchStatus& status) {
  if (!status.ok) {
    throw std::runtime_error(status.error != nullptr ? status.error
                                                     : "llama launch failed");
  }
}

void Require(const FlintLaunchStatus& status) {
  if (!status.ok) {
    throw std::runtime_error(status.error != nullptr ? status.error
                                                     : "llama launch failed");
  }
}

}  // namespace

void ExecuteLlamaDecodeStep(QwenGpuArena& arena,
                            const models::QwenModelWeights& weights,
                            tokenization::TokenId token_id, std::uint32_t pos,
                            bool compute_logits,
                            models::qwen::vision::DeviceInput* image_input) {
  if (image_input != nullptr && image_input->Bytes() != 0) {
    throw std::runtime_error("llama vision is not supported");
  }
  const auto& config = weights.config;
  if (!config.llama) {
    throw std::runtime_error("llama decode called for a non-llama config");
  }
  if (arena.d_kv_cache == nullptr) {
    throw std::runtime_error("llama requires the fp32 KV cache");
  }
  if (pos >= arena.GetMaxContext()) {
    throw std::runtime_error("llama token position exceeds the GPU context");
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
      scratch.ffn.activation.size() < intermediate ||
      scratch.ffn.out.size() < hidden ||
      scratch.decode.prompt_tokens.size() < 2) {
    throw std::runtime_error("llama decode scratch is short");
  }
  if (arena.d_lfm2_attn_partial == nullptr) {
    throw std::runtime_error("llama split-K attention scratch is missing");
  }

  const std::uint32_t params[2] = {token_id, pos};
  if (hipMemcpyAsync(scratch.decode.prompt_tokens.data(), params,
                     sizeof(params), hipMemcpyHostToDevice,
                     arena.stream) != hipSuccess) {
    throw std::runtime_error("llama token copy failed");
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
  if (config.num_layers != 0) {
    LaunchRMSNorm(scratch.decode.hidden.data(),
                  static_cast<const float*>(weights.layers[0].attn_norm.data),
                  scratch.decode.normed.data(), hidden, eps, arena.stream);
  }
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const auto& layer = weights.layers[l];
    LaunchGEMV(layer.attn_q.data, layer.attn_q.type,
               scratch.decode.normed.data(), scratch.attention.q.data(), attn,
               hidden, arena.stream);
    LaunchGEMV(layer.attn_k.data, layer.attn_k.type,
               scratch.decode.normed.data(), scratch.attention.k.data(), kv,
               hidden, arena.stream);
    LaunchGEMV(layer.attn_v.data, layer.attn_v.type,
               scratch.decode.normed.data(), scratch.attention.v.data(), kv,
               hidden, arena.stream);
    const std::uint32_t* d_pos = scratch.decode.prompt_tokens.data() + 1;
    LaunchRoPENorm(scratch.attention.q.data(), scratch.attention.k.data(),
                   heads, kv_heads, head_dim, rotary_dim, d_pos, rope_theta,
                   arena.stream);
    Require(LaunchGlimmerWriteKv(
        scratch.attention.k.data(), scratch.attention.v.data(),
        arena.d_kv_cache, v_cache, l, pos, max_ctx, kv_heads, head_dim, kv,
        plane, arena.stream));
    Require(LaunchFlintDecodeAttention(
        scratch.attention.q.data(), arena.d_kv_cache, v_cache,
        scratch.ssm.out.data(), l, d_pos, 0U, max_ctx, heads, kv_heads,
        head_dim, 32U, arena.d_lfm2_attn_partial, nullptr, arena.stream));
    LaunchGEMV(layer.attn_output.data, layer.attn_output.type,
               scratch.ssm.out.data(), scratch.attention.output.data(), hidden,
               attn, arena.stream);
    Require(LaunchLfm2AddRMSNorm(
        scratch.decode.hidden.data(), scratch.attention.output.data(),
        static_cast<const float*>(layer.ffn_norm.data),
        scratch.decode.normed.data(), static_cast<std::uint32_t>(hidden), eps,
        arena.stream));
    LaunchFusedSwiGLUGEMV(layer.ffn_gate.data, layer.ffn_gate.type,
                          layer.ffn_up.data, layer.ffn_up.type,
                          scratch.decode.normed.data(),
                          scratch.ffn.activation.data(), intermediate, hidden,
                          arena.stream);
    LaunchGEMV(layer.ffn_down.data, layer.ffn_down.type,
               scratch.ffn.activation.data(), scratch.ffn.out.data(), hidden,
               intermediate, arena.stream);
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
    // NBFAST T1: DFlash2 target-layer tap, same point as the Qwen decode
    // step (layer output after the FFN residual add).
    if (const auto tap_index = arena.GetTargetLayerCaptureIndex(l);
        tap_index.has_value()) {
      if (hipMemcpyAsync(arena.d_target_layer_features +
                             (*tap_index * hidden),
                         scratch.decode.hidden.data(),
                         hidden * sizeof(float), hipMemcpyDeviceToDevice,
                         arena.stream) != hipSuccess) {
        throw std::runtime_error("llama target-layer tap copy failed");
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

tokenization::TokenId QwenGpuExecutor::ForwardLlamaPromptChunk(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  CheckReset();
  if (vision_input_.Bytes() != 0) {
    throw std::runtime_error("llama vision is not supported");
  }
  const auto& config = weights_.config;
  if (!config.llama) {
    throw std::runtime_error("llama prefill called for a non-llama config");
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
    return ForwardLlamaPromptBatch(prompt_tokens, start_pos, compute_logits);
  }
  last_hidden_offset_ = 0;
  for (std::size_t i = 0; i < batch; ++i) {
    const std::uint32_t pos = start_pos + static_cast<std::uint32_t>(i);
    ExecuteLlamaDecodeStep(arena_, weights_, prompt_tokens[i], pos,
                           compute_logits && i + 1 == batch, &vision_input_);
    // NBFAST T1: prompt hidden capture for the single-token chunk path,
    // appended in the same [token][tap][hidden] order as the batched path.
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
