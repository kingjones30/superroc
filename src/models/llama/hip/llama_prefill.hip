#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/executor.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/models/glimmer/hip/glimmer_attention.hpp"
#include "src/models/glimmer/hip/glimmer_decode.hpp"
#include "src/models/llama/hip/llama_rope.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "src/models/qwen/hip/ops/token.hpp"

namespace gufo::hip {
namespace {

void LlamaCheckGpu(const char* what) {
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error(what);
  }
}

void LlamaRequire(const GlimmerLaunchStatus& status) {
  if (!status.ok) {
    throw std::runtime_error(status.error != nullptr ? status.error
                                                     : "llama launch failed");
  }
}

bool LlamaWmmaEnabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("LLAMA_WMMA");
    return env == nullptr || std::strcmp(env, "0") != 0;
  }();
  return enabled;
}

}  // namespace

tokenization::TokenId QwenGpuExecutor::ForwardLlamaPromptBatch(
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
  if (arena_.d_kv_cache == nullptr) {
    throw std::runtime_error("llama requires the fp32 KV cache");
  }
  const std::size_t batch = prompt_tokens.size();
  if (batch < 2) {
    throw std::runtime_error("llama batched prefill needs batch >= 2");
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
  const std::uint32_t rotary_dim = config.rotary_dim;
  const float rope_theta = config.rope_theta;
  const std::size_t plane = arena_.GetAttentionKvPlaneElements();
  float* v_cache = arena_.d_kv_cache + plane;
  void* act_f16 = arena_.d_scratch_bf16;

  if (arena_.d_hidden == nullptr || arena_.d_normed == nullptr ||
      arena_.d_q == nullptr || arena_.d_k == nullptr ||
      arena_.d_v == nullptr || arena_.d_attn_out == nullptr ||
      arena_.d_ffn_gate == nullptr || arena_.d_ffn_up == nullptr ||
      arena_.d_ffn_act == nullptr || arena_.d_ffn_out == nullptr ||
      arena_.d_ssm_out == nullptr || arena_.d_prompt_tokens == nullptr ||
      arena_.d_logits == nullptr || arena_.d_alpha_buf == nullptr ||
      arena_.d_scratch_bf16 == nullptr) {
    throw std::runtime_error("llama prefill scratch is short");
  }

  std::vector<std::uint32_t> host_tokens(prompt_tokens.begin(),
                                         prompt_tokens.end());
  if (hipMemcpyAsync(arena_.d_prompt_tokens, host_tokens.data(),
                     batch * sizeof(std::uint32_t), hipMemcpyHostToDevice,
                     arena_.stream) != hipSuccess) {
    throw std::runtime_error("llama prefill token copy failed");
  }
  LaunchBatchedEmbeddingLookup(weights_.token_embd.data, weights_.token_embd.type,
                               arena_.d_prompt_tokens, arena_.d_hidden, batch,
                               H, arena_.stream);
  LlamaCheckGpu("llama prefill embedding failed");

  const auto ws = arena_.GetScratchView(arena_.GetMaxBatch()).ffn;
  void* k_f16 = arena_.d_glimmer_kv_f16;
  void* v_f16 =
      k_f16 == nullptr ? nullptr
                       : static_cast<std::uint16_t*>(k_f16) + plane;
  const bool time_parts = [] {
    const char* env = std::getenv("LLAMA_PREFILL_TIME");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  // LLAMA (F16): the layer loop below throws on cancellation or launch
  // failure; hold the timer events in a small RAII guard created before the
  // loop so a throw cannot leak them (including a second create failing
  // after the first succeeded).
  struct ScopedPrefillTimers {
    hipEvent_t first{nullptr};
    hipEvent_t second{nullptr};
    ~ScopedPrefillTimers() {
      if (first != nullptr) {
        (void)hipEventDestroy(first);
      }
      if (second != nullptr) {
        (void)hipEventDestroy(second);
      }
    }
  } timers;
  hipEvent_t time_a = nullptr;
  hipEvent_t time_b = nullptr;
  float layer0_gemm_ms = 0.0F;
  float layer0_attn_ms = 0.0F;
  if (time_parts) {
    if (hipEventCreate(&timers.first) != hipSuccess ||
        hipEventCreate(&timers.second) != hipSuccess) {
      throw std::runtime_error("llama prefill timer create failed");
    }
    time_a = timers.first;
    time_b = timers.second;
  }
  const auto stamp = [&](hipEvent_t ev) {
    if (time_parts) {
      hipEventRecord(ev, arena_.stream);
    }
  };
  const auto elapsed_ms = [&]() -> float {
    if (!time_parts) {
      return 0.0F;
    }
    hipEventSynchronize(time_b);
    float ms = 0.0F;
    hipEventElapsedTime(&ms, time_a, time_b);
    return ms;
  };

  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    PrefillLayerCheckpoint(l);
    const auto& layer = weights_.layers[l];
    stamp(time_a);
    LaunchBatchedRMSNormFp16(arena_.d_hidden, nullptr,
                             static_cast<const float*>(layer.attn_norm.data),
                             nullptr, act_f16, batch, H, eps, arena_.stream);
    LlamaCheckGpu("llama prefill attn norm failed");
    LaunchBatchedQuantGEMMFp16(layer.attn_q.type, layer.attn_q.data, act_f16,
                               arena_.d_q, batch, attn, H, arena_.stream);
    LaunchBatchedQuantGEMMFp16(layer.attn_k.type, layer.attn_k.data, act_f16,
                               arena_.d_k, batch, kv, H, arena_.stream);
    LaunchBatchedQuantGEMMFp16(layer.attn_v.type, layer.attn_v.data, act_f16,
                               arena_.d_v, batch, kv, H, arena_.stream);
    LlamaCheckGpu("llama prefill qkv gemm failed");
    LaunchBatchedRoPENorm(arena_.d_q, arena_.d_k, batch, heads, kv_heads,
                          head_dim, rotary_dim, start_pos, rope_theta,
                          arena_.stream);
    LlamaCheckGpu("llama prefill rope failed");
    LlamaRequire(LaunchGlimmerBatchedWriteKv(
        arena_.d_k, arena_.d_v, arena_.d_kv_cache, v_cache, l, start_pos,
        static_cast<std::uint32_t>(batch), max_ctx, kv_heads, head_dim,
        batch * kv, plane, arena_.stream));
    stamp(time_b);
    if (l == 0) {
      layer0_gemm_ms += elapsed_ms();
    }
    stamp(time_a);
    bool wmma_ok = false;
    if (LlamaWmmaEnabled() && k_f16 != nullptr) {
      const std::size_t per_layer =
          static_cast<std::size_t>(arena_.GetMaxContext()) *
          config.num_key_value_heads * config.head_dim;
      const std::uint32_t plane_layers =
          per_layer == 0 ? 0U : static_cast<std::uint32_t>(plane / per_layer);
      LlamaRequire(LaunchGlimmerCastKvFp16(
          arena_.d_kv_cache, k_f16, /*layers=*/1, arena_.GetMaxContext(),
          config.num_key_value_heads, config.head_dim,
          static_cast<std::uint32_t>(end_pos), arena_.stream, l, plane_layers));
      wmma_ok = LaunchQwenWmmaAttention(
          arena_.d_q, nullptr, nullptr, nullptr, nullptr, nullptr, k_f16,
          v_f16, arena_.d_ssm_out, l, start_pos, batch, max_ctx, heads,
          kv_heads, head_dim, arena_.stream, nullptr, 0,
          /*skip_kv_write=*/true, ws.gate, ws.up, 0, nullptr);
      if (!wmma_ok) {
        static bool logged = false;
        if (!logged) {
          std::fprintf(stderr,
                       "llama prefill: WMMA attention unavailable, using "
                       "fp32\n");
          logged = true;
        }
      }
    }
    if (!wmma_ok) {
      LlamaRequire(LaunchGlimmerPrefillAttention(
          arena_.d_q, nullptr, arena_.d_kv_cache, v_cache, arena_.d_ssm_out, l,
          start_pos, static_cast<std::uint32_t>(batch), 0U, max_ctx, heads,
          kv_heads, head_dim, batch * attn, 0, plane, batch * attn,
          arena_.stream, nullptr));
    }
    stamp(time_b);
    if (l == 0) {
      layer0_attn_ms += elapsed_ms();
    }
    stamp(time_a);
    LaunchFloatToFp16(arena_.d_ssm_out, act_f16, batch * attn, arena_.stream);
    LlamaCheckGpu("llama prefill attn fp16 cast failed");
    LaunchBatchedQuantGEMMResidualFp16(layer.attn_output.type,
                                       layer.attn_output.data, act_f16,
                                       arena_.d_hidden, batch, H, attn,
                                       arena_.stream);
    LlamaCheckGpu("llama prefill attn residual failed");
    LaunchBatchedRMSNormFp16(arena_.d_hidden, nullptr,
                             static_cast<const float*>(layer.ffn_norm.data),
                             nullptr, act_f16, batch, H, eps, arena_.stream);
    LlamaCheckGpu("llama prefill ffn norm failed");
    const bool paired = TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
        layer.ffn_gate.type, layer.ffn_up.type, layer.ffn_gate.data,
        layer.ffn_up.data, act_f16, arena_.d_ffn_up, batch, I, H,
        arena_.stream);
    if (!paired) {
      LaunchBatchedQuantGEMMFp16(layer.ffn_gate.type, layer.ffn_gate.data,
                                 act_f16, arena_.d_ffn_gate, batch, I, H,
                                 arena_.stream);
      LaunchBatchedQuantGEMMSwiGLUFp16(layer.ffn_up.type, layer.ffn_up.data,
                                       act_f16, arena_.d_ffn_gate,
                                       arena_.d_ffn_up, batch, I, H,
                                       arena_.stream);
    }
    LlamaCheckGpu("llama prefill swiglu gemm failed");
    LaunchBatchedQuantGEMMResidualFp16(layer.ffn_down.type, layer.ffn_down.data,
                                       arena_.d_ffn_up, arena_.d_hidden, batch,
                                       H, I, arena_.stream);
    LlamaCheckGpu("llama prefill ffn residual failed");
    stamp(time_b);
    if (l == 0) {
      layer0_gemm_ms += elapsed_ms();
    }
  }
  if (time_parts) {
    std::fprintf(stderr,
                 "llama prefill timing layer0 gemm_ms=%.3f attn_ms=%.3f "
                 "batch=%zu start_pos=%u wmma=%d\n",
                 layer0_gemm_ms, layer0_attn_ms, batch, start_pos,
                 static_cast<int>(LlamaWmmaEnabled() && k_f16 != nullptr));
    // Events are destroyed by the ScopedPrefillTimers guard above.
  }

  last_hidden_offset_ = (batch - 1) * H;
  if (!compute_logits) {
    if (hipStreamSynchronize(arena_.stream) != hipSuccess) {
      throw std::runtime_error("llama prefill synchronize failed");
    }
    return 0;
  }
  const float* final_hidden = arena_.d_hidden + last_hidden_offset_;
  LaunchRMSNorm(final_hidden, static_cast<const float*>(weights_.output_norm.data),
                arena_.d_normed, H, eps, arena_.stream);
  LaunchGEMV(weights_.output.data, weights_.output.type, arena_.d_normed,
             arena_.d_logits, config.vocab_size, H, arena_.stream);
  LaunchBatchedGPUArgmax(
      arena_.d_logits, reinterpret_cast<std::uint32_t*>(arena_.d_alpha_buf), 1,
      config.vocab_size,
      std::span<float>(arena_.d_ffn_out, arena_.GetMaxBatch() * H),
      arena_.stream);
  std::uint32_t next_token_id = 0;
  if (hipMemcpyAsync(&next_token_id, arena_.d_alpha_buf, sizeof(std::uint32_t),
                     hipMemcpyDeviceToHost, arena_.stream) != hipSuccess ||
      hipStreamSynchronize(arena_.stream) != hipSuccess) {
    throw std::runtime_error("llama prefill token read failed");
  }
  return CheckedSampleToken(next_token_id, config.vocab_size);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
