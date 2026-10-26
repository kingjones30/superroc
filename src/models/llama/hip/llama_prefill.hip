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
#include "src/models/qwen/hip/ops/swiglu.hpp"
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

  // NBFAST T1: DFlash2 prompt hidden capture, appended per chunk in the
  // [token][tap][hidden] layout the Qwen prefill capture produces.
  const std::size_t target_layer_count = arena_.GetTargetLayerCapture().size();
  const std::size_t prompt_capture_offset = h_prompt_hidden_.size();
  if (capture_prompt_hidden_ && target_layer_count > 0) {
    h_prompt_hidden_.resize(prompt_capture_offset +
                            batch * target_layer_count * H);
  }

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
    if (capture_prompt_hidden_) {
      if (const auto tap_index = arena_.GetTargetLayerCaptureIndex(l);
          tap_index.has_value()) {
        float* destination = h_prompt_hidden_.data() + prompt_capture_offset +
                             (*tap_index * H);
        if (hipMemcpy2DAsync(destination,
                             target_layer_count * H * sizeof(float),
                             arena_.d_hidden, H * sizeof(float),
                             H * sizeof(float), batch, hipMemcpyDeviceToHost,
                             arena_.stream) != hipSuccess) {
          throw std::runtime_error("llama prompt tap copy failed");
        }
      }
    }
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

  if (capture_prompt_hidden_) {
    if (target_layer_count == 0) {
      const std::size_t old_size = h_prompt_hidden_.size();
      h_prompt_hidden_.resize(old_size + batch * H);
      if (hipMemcpyAsync(h_prompt_hidden_.data() + old_size, arena_.d_hidden,
                         batch * H * sizeof(float), hipMemcpyDeviceToHost,
                         arena_.stream) != hipSuccess) {
        throw std::runtime_error("llama prompt hidden copy failed");
      }
    }
    if (hipStreamSynchronize(arena_.stream) != hipSuccess) {
      throw std::runtime_error("llama prompt capture synchronize failed");
    }
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

// NBFAST T1: block verification through the llama prefill path. Same layer
// stack as ForwardLlamaPromptBatch, but every row gets final-norm + LM-head
// logits (into d_verification_logits_) and, when capture is armed, its
// target-layer taps in h_verification_hidden_ ([row][tap][hidden]). KV for
// all candidate rows is written; the verifier's SaveState/RestoreState +
// CommitVerificationChunk machinery rolls rejected rows back by re-forwarding
// the committed prefix with decode arithmetic.
std::vector<tokenization::TokenId>
QwenGpuExecutor::ForwardLlamaVerificationChunk(
    std::span<const tokenization::TokenId> candidate_tokens,
    std::uint32_t start_pos, bool capture_logits) {
  CheckReset();
  if (vision_input_.Bytes() != 0) {
    throw std::runtime_error("llama vision is not supported");
  }
  const auto& config = weights_.config;
  if (!config.llama) {
    throw std::runtime_error("llama verify called for a non-llama config");
  }
  h_verification_hidden_.clear();
  h_verification_logits_.clear();
  last_verification_rows_ = candidate_tokens.size();
  if (candidate_tokens.empty()) {
    return {};
  }
  const std::size_t batch = candidate_tokens.size();
  if (batch > arena_.GetMaxBatch()) {
    throw std::length_error("llama verification exceeds the GPU batch length");
  }
  const std::size_t end_pos = static_cast<std::size_t>(start_pos) + batch;
  if (end_pos > arena_.GetMaxContext()) {
    throw std::length_error("llama verification exceeds the GPU context");
  }
  EnsureVerificationLogits(batch);
  const std::size_t vocab = config.vocab_size;

  if (batch == 1) {
    // Single row: the exact decode step, mirroring the LFM2 branch.
    std::vector<tokenization::TokenId> predictions;
    predictions.push_back(ForwardToken(candidate_tokens[0], start_pos, true));
    if (capture_prompt_hidden_) {
      const auto hidden = CopyLastHidden();
      h_verification_hidden_.assign(hidden.begin(), hidden.end());
    }
    auto scratch = arena_.GetScratchView();
    if (hipMemcpyAsync(d_verification_logits_, scratch.decode.logits.data(),
                       vocab * sizeof(float), hipMemcpyDeviceToDevice,
                       arena_.stream) != hipSuccess) {
      throw std::runtime_error("llama verification logits copy failed");
    }
    if (capture_logits) {
      const auto logits = CopyLastLogits();
      h_verification_logits_.assign(logits.begin(), logits.end());
    }
    last_hidden_offset_ = 0;
    return predictions;
  }

  if (arena_.d_kv_cache == nullptr) {
    throw std::runtime_error("llama requires the fp32 KV cache");
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
    const std::size_t target_layer_count = arena_.GetTargetLayerCapture().size();
  if (capture_prompt_hidden_ && target_layer_count > 0) {
    h_verification_hidden_.resize(batch * target_layer_count * H);
  }
  if (arena_.d_hidden == nullptr || arena_.d_normed == nullptr ||
      arena_.d_q == nullptr || arena_.d_k == nullptr ||
      arena_.d_v == nullptr || arena_.d_attn_out == nullptr ||
      arena_.d_ffn_gate == nullptr || arena_.d_ffn_up == nullptr ||
      arena_.d_ffn_act == nullptr || arena_.d_ffn_out == nullptr ||
      arena_.d_ssm_out == nullptr || arena_.d_prompt_tokens == nullptr ||
      arena_.d_logits == nullptr || arena_.d_alpha_buf == nullptr ||
      arena_.d_scratch_bf16 == nullptr) {
    throw std::runtime_error("llama verification scratch is short");
  }

  std::vector<std::uint32_t> host_tokens(candidate_tokens.begin(),
                                         candidate_tokens.end());
  if (hipMemcpyAsync(arena_.d_prompt_tokens, host_tokens.data(),
                     batch * sizeof(std::uint32_t), hipMemcpyHostToDevice,
                     arena_.stream) != hipSuccess) {
    throw std::runtime_error("llama verification token copy failed");
  }
  LaunchBatchedEmbeddingLookup(weights_.token_embd.data,
                               weights_.token_embd.type, arena_.d_prompt_tokens,
                               arena_.d_hidden, batch, H, arena_.stream);
  LlamaCheckGpu("llama verification embedding failed");

  // The FP16 prefill GEMMs tile the token dimension by 256 rows, so an
  // 8-row verify block would pay 256 rows of WMMA compute (measured
  // ~480 ms per verify step). The FP32 small-batch GEMMs below share one
  // weight stream across the block at exact width, like the Qwen verify
  // projections; attention stays on the FP32 prefill kernel.
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    PrefillLayerCheckpoint(l);
    const auto& layer = weights_.layers[l];
    LaunchBatchedRMSNorm(arena_.d_hidden,
                         static_cast<const float*>(layer.attn_norm.data),
                         arena_.d_normed, nullptr, batch, H, eps,
                         arena_.stream);
    LaunchBatchedQuantGEMMFp32(layer.attn_q.type, layer.attn_q.data,
                               arena_.d_normed, arena_.d_q, batch, attn, H,
                               arena_.stream);
    LaunchBatchedQuantGEMMFp32(layer.attn_k.type, layer.attn_k.data,
                               arena_.d_normed, arena_.d_k, batch, kv, H,
                               arena_.stream);
    LaunchBatchedQuantGEMMFp32(layer.attn_v.type, layer.attn_v.data,
                               arena_.d_normed, arena_.d_v, batch, kv, H,
                               arena_.stream);
    LlamaCheckGpu("llama verification qkv gemm failed");
    LaunchBatchedRoPENorm(arena_.d_q, arena_.d_k, batch, heads, kv_heads,
                          head_dim, rotary_dim, start_pos, rope_theta,
                          arena_.stream);
    LlamaCheckGpu("llama verification rope failed");
    LlamaRequire(LaunchGlimmerBatchedWriteKv(
        arena_.d_k, arena_.d_v, arena_.d_kv_cache, v_cache, l, start_pos,
        static_cast<std::uint32_t>(batch), max_ctx, kv_heads, head_dim,
        batch * kv, plane, arena_.stream));
    // WMMA attention over the FP16 KV copy, as in prefill: the FP32
    // Glimmer kernel re-reads the KV cache per query row and dominated
    // verify time at 4K context (~430 ms/step vs ~145 ms at 512).
    bool wmma_ok = false;
    if (LlamaWmmaEnabled() && arena_.d_glimmer_kv_f16 != nullptr) {
      void* k_f16 = arena_.d_glimmer_kv_f16;
      void* v_f16 = static_cast<std::uint16_t*>(k_f16) + plane;
      const std::size_t per_layer =
          static_cast<std::size_t>(arena_.GetMaxContext()) *
          config.num_key_value_heads * config.head_dim;
      const std::uint32_t plane_layers =
          per_layer == 0 ? 0U : static_cast<std::uint32_t>(plane / per_layer);
      LlamaRequire(LaunchGlimmerCastKvFp16(
          arena_.d_kv_cache, k_f16, /*layers=*/1, arena_.GetMaxContext(),
          config.num_key_value_heads, config.head_dim,
          static_cast<std::uint32_t>(end_pos), arena_.stream, l, plane_layers));
      const auto ws = arena_.GetScratchView(arena_.GetMaxBatch()).ffn;
      wmma_ok = LaunchQwenWmmaAttention(
          arena_.d_q, nullptr, nullptr, nullptr, nullptr, nullptr, k_f16,
          v_f16, arena_.d_ssm_out, l, start_pos, batch, max_ctx, heads,
          kv_heads, head_dim, arena_.stream, nullptr, 0,
          /*skip_kv_write=*/true, ws.gate, ws.up, 0, nullptr);
    }
    if (!wmma_ok) {
      LlamaRequire(LaunchGlimmerPrefillAttention(
          arena_.d_q, nullptr, arena_.d_kv_cache, v_cache, arena_.d_ssm_out, l,
          start_pos, static_cast<std::uint32_t>(batch), 0U, max_ctx, heads,
          kv_heads, head_dim, batch * attn, 0, plane, batch * attn,
          arena_.stream, nullptr));
    }
    LaunchBatchedQuantGEMMFp32(layer.attn_output.type, layer.attn_output.data,
                               arena_.d_ssm_out, arena_.d_attn_out, batch, H,
                               attn, arena_.stream);
    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_attn_out,
                             arena_.d_hidden, batch, H, arena_.stream);
    LaunchBatchedRMSNorm(arena_.d_hidden,
                         static_cast<const float*>(layer.ffn_norm.data),
                         arena_.d_normed, nullptr, batch, H, eps,
                         arena_.stream);
    LaunchBatchedQuantGEMMFp32(layer.ffn_gate.type, layer.ffn_gate.data,
                               arena_.d_normed, arena_.d_ffn_gate, batch, I, H,
                               arena_.stream);
    LaunchBatchedQuantGEMMFp32(layer.ffn_up.type, layer.ffn_up.data,
                               arena_.d_normed, arena_.d_ffn_up, batch, I, H,
                               arena_.stream);
    LaunchBatchedSwiGLUActivation(arena_.d_ffn_gate, arena_.d_ffn_up,
                                  arena_.d_ffn_act, nullptr, batch * I,
                                  arena_.stream);
    LaunchBatchedQuantGEMMFp32(layer.ffn_down.type, layer.ffn_down.data,
                               arena_.d_ffn_act, arena_.d_ffn_out, batch, H, I,
                               arena_.stream);
    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_ffn_out,
                             arena_.d_hidden, batch, H, arena_.stream);
    LlamaCheckGpu("llama verification layer failed");
    if (capture_prompt_hidden_) {
      if (const auto tap_index = arena_.GetTargetLayerCaptureIndex(l);
          tap_index.has_value()) {
        float* destination =
            h_verification_hidden_.data() + (*tap_index * H);
        if (hipMemcpy2DAsync(destination,
                             target_layer_count * H * sizeof(float),
                             arena_.d_hidden, H * sizeof(float),
                             H * sizeof(float), batch, hipMemcpyDeviceToHost,
                             arena_.stream) != hipSuccess) {
          throw std::runtime_error("llama verification tap copy failed");
        }
      }
    }
  }

  if (capture_prompt_hidden_ && target_layer_count == 0) {
    h_verification_hidden_.resize(batch * H);
    if (hipMemcpyAsync(h_verification_hidden_.data(), arena_.d_hidden,
                       batch * H * sizeof(float), hipMemcpyDeviceToHost,
                       arena_.stream) != hipSuccess) {
      throw std::runtime_error("llama verification hidden copy failed");
    }
  }

  LaunchBatchedRMSNorm(arena_.d_hidden,
                       static_cast<const float*>(weights_.output_norm.data),
                       arena_.d_normed, nullptr, batch, H, eps, arena_.stream);
  LaunchBatchedQuantGEMMFp32(weights_.output.type, weights_.output.data,
                             arena_.d_normed, d_verification_logits_, batch,
                             vocab, H, arena_.stream);
  LlamaCheckGpu("llama verification lm head failed");
  LaunchBatchedGPUArgmax(
      d_verification_logits_, arena_.d_prompt_tokens, batch, vocab,
      std::span<float>(arena_.d_ffn_out, arena_.GetMaxBatch() * H),
      arena_.stream);
  std::vector<std::uint32_t> host_predictions(batch, 0);
  if (hipMemcpyAsync(host_predictions.data(), arena_.d_prompt_tokens,
                     batch * sizeof(std::uint32_t), hipMemcpyDeviceToHost,
                     arena_.stream) != hipSuccess) {
    throw std::runtime_error("llama verification argmax read failed");
  }
  if (capture_logits) {
    h_verification_logits_.resize(batch * vocab);
    if (hipMemcpyAsync(h_verification_logits_.data(), d_verification_logits_,
                       batch * vocab * sizeof(float), hipMemcpyDeviceToHost,
                       arena_.stream) != hipSuccess) {
      throw std::runtime_error("llama verification logits read failed");
    }
  }
  // CopyLastLogits reads the decode logits slot; keep the last row there so
  // host-side samplers that ask for it see this forward's frontier.
  if (hipMemcpyAsync(arena_.d_logits,
                     d_verification_logits_ + (batch - 1) * vocab,
                     vocab * sizeof(float), hipMemcpyDeviceToDevice,
                     arena_.stream) != hipSuccess) {
    throw std::runtime_error("llama verification frontier copy failed");
  }
  if (hipStreamSynchronize(arena_.stream) != hipSuccess) {
    throw std::runtime_error("llama verification synchronize failed");
  }
  last_hidden_offset_ = (batch - 1) * H;
  std::vector<tokenization::TokenId> predictions;
  predictions.reserve(batch);
  for (const std::uint32_t token : host_predictions) {
    predictions.push_back(CheckedSampleToken(token, config.vocab_size));
  }
  return predictions;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
