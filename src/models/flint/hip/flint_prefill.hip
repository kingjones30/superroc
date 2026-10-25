// Flint batched prefill (SPEC1, lane LFS). Batch >= 2; caller handles 1. // LFM2
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/executor.hpp"  // LFM2

#include <algorithm>  // LFM2
#include <cstddef>  // LFM2
#include <cstdint>  // LFM2
#include <cstdio>  // LFM2
#include <cstdlib>  // LFM2
#include <cstring>
#include <span>  // LFM2
#include <stdexcept>  // LFM2
#include <vector>  // LFM2

#include "src/models/glimmer/hip/glimmer_attention.hpp"
#include "src/models/glimmer/hip/glimmer_decode.hpp"  // LFM2 (fp32 KV + attention)
#include "src/models/lfm2/hip/lfm2_kernels.hpp"  // LFM2
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"  // LFM2
#include "src/models/qwen/hip/ops/norm_residual.hpp"  // LFM2
#include "src/models/qwen/hip/ops/swiglu.hpp"  // LFM2
#include "src/models/qwen/hip/ops/token.hpp"  // LFM2
#include "src/models/qwen/moe/qwen_moe_prefill.hpp"  // LFM2 (grouped experts)

namespace gufo::hip {
namespace {

// LFM2: Glimmer's CheckGpu renamed. // LFM2
void FlintCheckGpu(const char* what) {  // LFM2
  const hipError_t err = hipGetLastError();  // LFM2
  if (err != hipSuccess) {  // LFM2
    throw std::runtime_error(what);  // LFM2
  }
}  // LFM2

// LFM2: Glimmer's WeightScratchElems renamed. // LFM2
std::size_t FlintWeightScratchElems(const core::ModelConfig& config) {  // LFM2
  const std::size_t hidden = config.hidden_size;  // LFM2
  const std::size_t attention = config.AttentionSize();  // LFM2
  const std::size_t kv =  // LFM2
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;  // LFM2
  const std::size_t q_projection = 2 * attention;  // LFM2
  return hidden * std::max<std::size_t>(  // LFM2
                      {q_projection, kv, attention, config.intermediate_size,  // LFM2
                       config.SsmQkvSize(), config.ssm_inner_size,  // LFM2
                       config.ssm_time_step_rank});  // LFM2
}  // LFM2

// LFM2: Glimmer's W8A8 decision renamed. // LFM2
bool FlintUseW8A8() {  // LFM2
  static const bool on = [] {  // LFM2
    const char* env = std::getenv("GLIMMER_W8A8");  // LFM2
    return !(env != nullptr && env[0] == '0' && env[1] == '\0');  // LFM2
  }();  // LFM2
  return on;  // LFM2
}  // LFM2

// LFM2: Glimmer's W8A8MaxBatch renamed. // LFM2
std::size_t FlintW8A8MaxBatch() {  // LFM2
  static const std::size_t rows = [] {  // LFM2
    const char* env = std::getenv("GLIMMER_W8A8_MAX_BATCH");  // LFM2
    if (env != nullptr && env[0] != '\0') {  // LFM2
      return static_cast<std::size_t>(std::strtoul(env, nullptr, 10));  // LFM2
    }  // LFM2
    return std::size_t{2048};  // LFM2
  }();  // LFM2
  return rows;  // LFM2
}  // LFM2

// LFM2: Glimmer's UseW8A8ForChunk renamed. // LFM2
bool FlintUseW8A8ForChunk(std::size_t batch) {  // LFM2
  return FlintUseW8A8() && batch <= FlintW8A8MaxBatch();  // LFM2
}  // LFM2

// LFM2 (Super ROC): W8A8 needs every projection the w8 branches touch to be
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

// LFM2: Glimmer's ArenaQ8ScratchBytes renamed. // LFM2
std::size_t FlintArenaQ8ScratchBytes(std::size_t batch,  // LFM2
                                    std::size_t row_elems) {  // LFM2
  const std::size_t q8_rows = ((batch + 15) / 16) * 16;  // LFM2
  return ((((q8_rows * row_elems) + 31) / 32) * sizeof(float) * 10) + 4096;  // LFM2
}  // LFM2

// LFM2: Glimmer's Require pattern for LFM2 launch status. // LFM2
void FlintRequire(const Lfm2LaunchStatus& status) {  // LFM2
  if (!status.ok) {  // LFM2
    throw std::runtime_error(status.error != nullptr ? status.error  // LFM2
                                                     : "flint launch failed");  // LFM2
  }  // LFM2
}  // LFM2

// LFM2: Glimmer's Require pattern for Glimmer launch status. // LFM2
void FlintRequire(const GlimmerLaunchStatus& status) {  // LFM2
  if (!status.ok) {  // LFM2
    throw std::runtime_error(status.error != nullptr ? status.error  // LFM2
                                                     : "glimmer launch failed");  // LFM2
  }  // LFM2
}  // LFM2

bool FlintWmmaEnabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("FLINT_WMMA");
    return env == nullptr || std::strcmp(env, "0") != 0;
  }();
  return enabled;
}

// LFM2: Glimmer's Project renamed (BF16 dequant + hipBLAS, batch 1 via GEMV). // LFM2
void FlintProject(QwenGpuArena& arena, const models::QwenTensorRef& weight,  // LFM2
                 const float* fp32, const void* bf16, float* y,  // LFM2
                 std::size_t batch, std::size_t m, std::size_t k,  // LFM2
                 std::size_t weight_elems) {  // LFM2
  if (m == 0 || k == 0 || batch == 0) {  // LFM2
    throw std::runtime_error("flint prefill projection is empty");  // LFM2
  }  // LFM2
  if (batch == 1) {  // LFM2
    LaunchGEMV(weight.data, weight.type, fp32, y, m, k, arena.stream);  // LFM2
    FlintCheckGpu("flint prefill gemv failed");  // LFM2
    return;  // LFM2
  }  // LFM2
  if (m * k > weight_elems) {  // LFM2
    throw std::runtime_error("flint prefill weight scratch is short");  // LFM2
  }  // LFM2
  LaunchDequantizeToBf16(weight.type, weight.data, arena.d_weights_bf16,  // LFM2
                         m * k, arena.stream);  // LFM2
  FlintCheckGpu("flint prefill dequant failed");  // LFM2
  bool ran = false;  // LFM2
  if (arena.hipblaslt_gemm != nullptr && m >= 1024 && k >= 1024) {  // LFM2
    ran = arena.hipblaslt_gemm->RunBf16(arena.d_weights_bf16, bf16, y, batch,  // LFM2
                                        m, k, arena.stream);  // LFM2
  }  // LFM2
  if (!ran) {  // LFM2
    LaunchHipblasGEMMBF16(arena.hipblas_handle, arena.d_weights_bf16, bf16,  // LFM2
                          y, batch, m, k, arena.stream);  // LFM2
  }  // LFM2
  FlintCheckGpu("flint prefill bf16 gemm failed");  // LFM2
}  // LFM2

// LFM2: Glimmer's ProjectQ8 renamed (prequantized W8A8 WMMA). // LFM2
void FlintProjectQ8(QwenGpuArena& arena, const models::QwenTensorRef& weight,  // LFM2
                    const void* q8, float* y, std::size_t batch,  // LFM2
                    std::size_t m, std::size_t k) {  // LFM2
  if (m == 0 || k == 0 || batch == 0 || q8 == nullptr) {  // LFM2
    throw std::runtime_error("flint prefill w8a8 projection is empty");  // LFM2
  }  // LFM2
  if (!detail::IsNativeWmmaQuant(weight.type)) {  // LFM2
    throw std::runtime_error("flint prefill w8a8 weight is not native quant");  // LFM2
  }  // LFM2
  LaunchBatchedQuantGEMMPreQuantized(weight.type, weight.data, q8, y, batch,  // LFM2
                                     m, k, arena.stream);  // LFM2
  FlintCheckGpu("flint prefill w8a8 gemm failed");  // LFM2
}  // LFM2

}  // namespace

// LFM2: batched prefill for flint, batch >= 2 (caller handles 1). // LFM2
tokenization::TokenId QwenGpuExecutor::ForwardFlintPromptBatch(  // LFM2
    std::span<const tokenization::TokenId> prompt_tokens,  // LFM2
    std::uint32_t start_pos, bool compute_logits) {  // LFM2
  CheckReset();  // LFM2
  if (vision_input_.Bytes() != 0) {  // LFM2
    throw std::runtime_error("flint vision is not supported");  // LFM2
  }  // LFM2
  const auto& config = weights_.config;  // LFM2
  if (!config.flint) {
    throw std::runtime_error("flint prefill called for a non-flint config");  // LFM2
  }  // LFM2
  if (arena_.d_kv_cache == nullptr) {
    throw std::runtime_error("flint requires the fp32 KV cache");
  }  // LFM2
  const std::size_t batch = prompt_tokens.size();  // LFM2
  if (batch == 0) {  // LFM2
    return 0;  // LFM2
  }  // LFM2
  if (batch < 2) {  // LFM2 (caller handles 1)
    throw std::runtime_error("flint batched prefill needs batch >= 2");  // LFM2
  }  // LFM2
  if (batch > arena_.GetMaxBatch()) {  // LFM2
    throw std::length_error("prompt chunk exceeds the GPU batch length");  // LFM2
  }  // LFM2
  const std::size_t end_pos = static_cast<std::size_t>(start_pos) + batch;  // LFM2
  if (end_pos > arena_.GetMaxContext()) {  // LFM2
    throw std::length_error("prompt exceeds the GPU context length");  // LFM2
  }  // LFM2

  const std::size_t H = config.hidden_size;  // LFM2
  const std::size_t attn =  // LFM2
      static_cast<std::size_t>(config.num_attention_heads) * config.head_dim;  // LFM2
  const std::size_t kv =  // LFM2
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;  // LFM2
  const std::size_t I = config.intermediate_size;  // LFM2
  const float eps = config.norm_eps_pre;  // LFM2
  const std::uint32_t max_ctx = arena_.GetMaxContext();  // LFM2
  const std::uint32_t heads = config.num_attention_heads;  // LFM2
  const std::uint32_t kv_heads = config.num_key_value_heads;  // LFM2
  const std::uint32_t head_dim = config.head_dim;  // LFM2
  const float rope_theta = config.rope_theta;  // LFM2
  const std::uint32_t n_expert = config.expert_count;  // LFM2
  const std::uint32_t n_used = config.expert_used_count;  // LFM2
  const std::uint32_t expert_ff = config.expert_feed_forward_length;  // LFM2
  const std::size_t shexp_ff = config.expert_shared_feed_forward_length;
  const std::size_t weight_elems = FlintWeightScratchElems(config);  // LFM2
  const std::size_t plane = arena_.GetAttentionKvPlaneElements();  // LFM2
  float* v_cache = arena_.d_kv_cache + plane;
  auto* bf16 = static_cast<hip_bfloat16*>(arena_.d_scratch_bf16);  // LFM2
  const bool w8 =  // LFM2
      FlintUseW8A8ForChunk(batch) && !weights_.layers.empty() &&  // LFM2
      FlintAllProjectionsNative(weights_);  // LFM2
  if (w8) {  // LFM2
    const std::size_t q_projection = 2 * attn;  // LFM2
    const std::size_t ssm_qkv = config.SsmQkvSize();  // LFM2
    const std::size_t projection_width = std::max(q_projection, ssm_qkv);  // LFM2
    const std::size_t row = std::max<std::size_t>(  // LFM2
        {I, H, projection_width,  // LFM2
         ssm_qkv + config.ssm_inner_size +  // LFM2
             (2 * static_cast<std::size_t>(config.ssm_time_step_rank))});  // LFM2
    const std::size_t have =  // LFM2
        FlintArenaQ8ScratchBytes(arena_.GetMaxBatch(), row);  // LFM2
    const std::size_t need =  // LFM2
        QuantizedActivationBytes(arena_.GetMaxBatch(), I);  // LFM2
    if (need > have) {  // LFM2
      throw std::runtime_error("flint prefill q8 scratch is short");  // LFM2
    }  // LFM2
  }  // LFM2

  const std::size_t max_batch = arena_.GetMaxBatch();  // LFM2
  if (arena_.d_hidden == nullptr || arena_.d_normed == nullptr ||  // LFM2
      arena_.d_q == nullptr || arena_.d_k == nullptr ||  // LFM2
      arena_.d_v == nullptr || arena_.d_attn_out == nullptr ||  // LFM2
      arena_.d_ffn_gate == nullptr || arena_.d_ffn_up == nullptr ||  // LFM2
      arena_.d_ffn_act == nullptr || arena_.d_ffn_out == nullptr ||  // LFM2
      arena_.d_ssm_out == nullptr || arena_.d_prompt_tokens == nullptr ||  // LFM2
      arena_.d_logits == nullptr || arena_.d_alpha_buf == nullptr ||  // LFM2
      arena_.d_scratch_bf16 == nullptr ||  // LFM2
      arena_.d_weights_bf16 == nullptr ||  // LFM2
      arena_.d_kv_cache == nullptr) {
    throw std::runtime_error("flint prefill scratch is short");  // LFM2
  }  // LFM2
  if (max_batch * H < batch * H || max_batch * attn < batch * attn ||  // LFM2
      max_batch * kv < batch * kv || max_batch * I < batch * I) {  // LFM2
    throw std::runtime_error("flint prefill scratch is short");  // LFM2
  }  // LFM2
  if (config.is_moe || n_expert > 0) {  // LFM2
    if (arena_.d_moe_logits == nullptr || arena_.d_moe_ids == nullptr ||  // LFM2
        arena_.d_moe_weights == nullptr) {  // LFM2
      throw std::runtime_error("flint prefill moe scratch is short");  // LFM2
    }  // LFM2
    if (max_batch * static_cast<std::size_t>(n_expert) <  // LFM2
            batch * static_cast<std::size_t>(n_expert) ||  // LFM2
        max_batch * static_cast<std::size_t>(n_used) <  // LFM2
            batch * static_cast<std::size_t>(n_used)) {  // LFM2
      throw std::runtime_error("flint prefill moe scratch is short");  // LFM2
    }  // LFM2
  }  // LFM2

  // LFM2: copy tokens to arena_.d_prompt_tokens. // LFM2
  std::vector<std::uint32_t> host_tokens(prompt_tokens.begin(),  // LFM2
                                         prompt_tokens.end());  // LFM2
  if (hipMemcpyAsync(arena_.d_prompt_tokens, host_tokens.data(),  // LFM2
                     batch * sizeof(std::uint32_t), hipMemcpyHostToDevice,  // LFM2
                     arena_.stream) != hipSuccess) {  // LFM2
    throw std::runtime_error("flint prefill token copy failed");  // LFM2
  }  // LFM2
  LaunchBatchedEmbeddingLookup(weights_.token_embd.data,  // LFM2
                               weights_.token_embd.type,  // LFM2
                               arena_.d_prompt_tokens, arena_.d_hidden, batch,  // LFM2
                               H, arena_.stream);  // LFM2
  FlintCheckGpu("flint prefill embedding failed");  // LFM2
  // LFM2: NO norm after the embedding. // LFM2

  // DSpark taps: [token][tap][hidden] rows for the drafter, laid out
  // exactly as the Qwen prefill capture the verifier primes from.
  const std::size_t target_layer_count =
      arena_.GetTargetLayerCapture().size();  // LFM2
  const std::size_t prompt_capture_offset = h_prompt_hidden_.size();  // LFM2
  if (capture_prompt_hidden_ && target_layer_count > 0) {  // LFM2
    h_prompt_hidden_.resize(prompt_capture_offset +
                            batch * target_layer_count * H);  // LFM2
  }  // LFM2

  for (std::uint32_t l = 0; l < config.num_layers; ++l) {  // LFM2
    PrefillLayerCheckpoint(l);  // LFM2
    const auto& layer = weights_.layers[l];  // LFM2
    LaunchBatchedRMSNorm(arena_.d_hidden,  // LFM2
                         static_cast<const float*>(layer.attn_norm.data),  // LFM2
                         arena_.d_normed, bf16, batch, H, eps,  // LFM2
                         arena_.stream);  // LFM2
    FlintCheckGpu("flint prefill attn norm failed");  // LFM2
    if (w8) {  // LFM2 (like Glimmer, no attn_gate)
      LaunchQuantizeActivationQ8_1(bf16, arena_.d_scratch_q8_act, batch, H,  // LFM2
                                   arena_.stream);  // LFM2
      FlintCheckGpu("flint prefill attn q8 failed");  // LFM2
      FlintProjectQ8(arena_, layer.attn_q, arena_.d_scratch_q8_act,  // LFM2
                    arena_.d_q, batch, attn, H);  // LFM2
      FlintProjectQ8(arena_, layer.attn_k, arena_.d_scratch_q8_act,  // LFM2
                    arena_.d_k, batch, kv, H);  // LFM2
      FlintProjectQ8(arena_, layer.attn_v, arena_.d_scratch_q8_act,  // LFM2
                    arena_.d_v, batch, kv, H);  // LFM2
    } else {  // LFM2
      FlintProject(arena_, layer.attn_q, arena_.d_normed, bf16,  // LFM2
                  arena_.d_q, batch, attn, H, weight_elems);  // LFM2
      FlintProject(arena_, layer.attn_k, arena_.d_normed, bf16,  // LFM2
                  arena_.d_k, batch, kv, H, weight_elems);  // LFM2
      FlintProject(arena_, layer.attn_v, arena_.d_normed, bf16,  // LFM2
                  arena_.d_v, batch, kv, H, weight_elems);  // LFM2
    }  // LFM2
    LaunchBatchedPerHeadRMSNorm(  // LFM2
        arena_.d_q, static_cast<const float*>(layer.attn_q_norm.data),  // LFM2
        arena_.d_q, batch, heads, head_dim, eps, arena_.stream);  // LFM2
    LaunchBatchedPerHeadRMSNorm(  // LFM2
        arena_.d_k, static_cast<const float*>(layer.attn_k_norm.data),  // LFM2
        arena_.d_k, batch, kv_heads, head_dim, eps, arena_.stream);  // LFM2
    FlintCheckGpu("flint prefill qk norm failed");  // LFM2
    if (config.LayerUsesRope(l)) {
      LaunchBatchedRoPE(arena_.d_q, arena_.d_k, batch, heads, kv_heads,  // LFM2
                        head_dim, head_dim, start_pos, rope_theta,  // LFM2
                        arena_.stream, nullptr);  // LFM2 (NEOX, positions start_pos..)
      FlintCheckGpu("flint prefill rope failed");  // LFM2
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
      LaunchQuantizeActivationQ8_1FromFp32(  // LFM2
          arena_.d_ssm_out, arena_.d_scratch_q8_act, batch, attn,  // LFM2
          arena_.stream);  // LFM2
      FlintCheckGpu("flint prefill attn-out q8 failed");  // LFM2
      FlintProjectQ8(arena_, layer.attn_output, arena_.d_scratch_q8_act,  // LFM2
                    arena_.d_attn_out, batch, H, attn);  // LFM2
    } else {  // LFM2
      LaunchFloatToBfloat16(arena_.d_ssm_out, bf16, batch * attn,  // LFM2
                            arena_.stream);  // LFM2
      FlintCheckGpu("flint prefill attn bf16 cast failed");  // LFM2
      FlintProject(arena_, layer.attn_output, arena_.d_ssm_out, bf16,  // LFM2
                  arena_.d_attn_out, batch, H, attn, weight_elems);  // LFM2
    }  // LFM2
    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_attn_out,  // LFM2
                             arena_.d_hidden, batch, H, arena_.stream);  // LFM2
    FlintCheckGpu("flint prefill attn residual failed");  // LFM2
    LaunchBatchedRMSNorm(arena_.d_hidden,  // LFM2
                         static_cast<const float*>(layer.ffn_norm.data),  // LFM2
                         arena_.d_normed, bf16, batch, H, eps,  // LFM2
                         arena_.stream);  // LFM2
    FlintCheckGpu("flint prefill ffn norm failed");  // LFM2
    if (!layer.is_moe) {  // LFM2 dense lead layer (Glimmer FFN verbatim)
      if (w8) {  // LFM2
        LaunchQuantizeActivationQ8_1(bf16, arena_.d_scratch_q8_act, batch, H,  // LFM2
                                     arena_.stream);  // LFM2
        FlintCheckGpu("flint prefill ffn q8 failed");  // LFM2
        if (layer.ffn_gate.type == layer.ffn_up.type) {  // LFM2
          LaunchBatchedDualQuantGEMMPreQuantized(  // LFM2
              layer.ffn_gate.type, layer.ffn_gate.data, layer.ffn_up.data,  // LFM2
              arena_.d_scratch_q8_act, arena_.d_ffn_gate, arena_.d_ffn_up,  // LFM2
              batch, I, H, arena_.stream);  // LFM2
          FlintCheckGpu("flint prefill dual ffn gemm failed");  // LFM2
        } else {  // LFM2
          FlintProjectQ8(arena_, layer.ffn_gate, arena_.d_scratch_q8_act,  // LFM2
                        arena_.d_ffn_gate, batch, I, H);  // LFM2
          FlintProjectQ8(arena_, layer.ffn_up, arena_.d_scratch_q8_act,  // LFM2
                        arena_.d_ffn_up, batch, I, H);  // LFM2
        }  // LFM2
        LaunchBatchedFusedSwiGLUQuantizeQ8_1(  // LFM2
            arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_scratch_q8_act,  // LFM2
            batch, I, arena_.stream);  // LFM2
        FlintCheckGpu("flint prefill swiglu q8 failed");  // LFM2
        FlintProjectQ8(arena_, layer.ffn_down, arena_.d_scratch_q8_act,  // LFM2
                      arena_.d_ffn_out, batch, H, I);  // LFM2
      } else {  // LFM2
        FlintProject(arena_, layer.ffn_gate, arena_.d_normed, bf16,  // LFM2
                    arena_.d_ffn_gate, batch, I, H, weight_elems);  // LFM2
        FlintProject(arena_, layer.ffn_up, arena_.d_normed, bf16,  // LFM2
                    arena_.d_ffn_up, batch, I, H, weight_elems);  // LFM2
        LaunchBatchedSwiGLUActivation(  // LFM2
            arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_ffn_act,  // LFM2
            batch == 1 ? nullptr : static_cast<void*>(bf16),  // LFM2
            batch * I, arena_.stream);  // LFM2
        FlintCheckGpu("flint prefill swiglu failed");  // LFM2
        FlintProject(arena_, layer.ffn_down, arena_.d_ffn_act, bf16,  // LFM2
                    arena_.d_ffn_out, batch, H, I, weight_elems);  // LFM2
      }  // LFM2
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
      FlintRequire(LaunchLfm2MoeRouteBatch(  // LFM2
          arena_.d_normed,  // LFM2
          static_cast<const float*>(layer.ffn_gate_inp.data),  // LFM2
          static_cast<const float*>(layer.exp_probs_b.data),  // LFM2
          static_cast<std::uint32_t>(H), n_expert,  // LFM2
          static_cast<std::uint32_t>(batch), n_used, arena_.d_moe_logits,  // LFM2
          arena_.d_moe_ids, arena_.d_moe_weights, arena_.stream));  // LFM2
      if (LaunchKatMoeRoutedGrouped(  // LFM2
              static_cast<int>(layer.ffn_gate_exps.type),  // LFM2
              layer.ffn_gate_exps.data, layer.ffn_up_exps.data,  // LFM2
              layer.ffn_down_exps.data, arena_.d_normed, arena_.d_moe_ids,  // LFM2
              arena_.d_moe_weights, static_cast<int>(batch),  // LFM2
              static_cast<int>(n_used), static_cast<int>(n_expert),  // LFM2
              static_cast<int>(H), static_cast<int>(expert_ff),  // LFM2
              arena_.d_ffn_out, arena_.stream) != 0) {  // LFM2
        throw std::runtime_error("flint grouped prefill experts failed");  // LFM2
      }  // LFM2
      LaunchBatchedResidualAdd(arena_.d_ffn_out, arena_.d_attn_out,
                               arena_.d_ffn_out, batch, H, arena_.stream);
      FlintCheckGpu("flint prefill shexp residual failed");
    }  // LFM2
    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_ffn_out,  // LFM2
                             arena_.d_hidden, batch, H, arena_.stream);  // LFM2
    FlintCheckGpu("flint prefill ffn residual failed");  // LFM2
    if (capture_prompt_hidden_ && target_layer_count > 0) {  // LFM2
      if (const auto tap = arena_.GetTargetLayerCaptureIndex(l);  // LFM2
          tap.has_value()) {  // LFM2
        float* destination = h_prompt_hidden_.data() +  // LFM2
                             prompt_capture_offset + (*tap * H);  // LFM2
        if (hipMemcpy2DAsync(  // LFM2
                destination, target_layer_count * H * sizeof(float),
                arena_.d_hidden, H * sizeof(float), H * sizeof(float),
                batch, hipMemcpyDeviceToHost, arena_.stream) != hipSuccess) {
          throw std::runtime_error("flint target-layer tap copy failed");
        }
      }  // LFM2
    }  // LFM2
  }  // LFM2

  // LFM2: tail exactly as Glimmer's but WITHOUT the scale/softcap call. // LFM2
  last_hidden_offset_ = (batch - 1) * H;  // LFM2
  if (!compute_logits) {  // LFM2
    if (hipStreamSynchronize(arena_.stream) != hipSuccess) {  // LFM2
      throw std::runtime_error("flint prefill synchronize failed");  // LFM2
    }  // LFM2
    return 0;  // LFM2
  }  // LFM2
  const float* final_hidden = arena_.d_hidden + last_hidden_offset_;  // LFM2
  LaunchRMSNorm(final_hidden,  // LFM2
                static_cast<const float*>(weights_.output_norm.data),  // LFM2
                arena_.d_normed, H, eps, arena_.stream);  // LFM2
  LaunchGEMV(weights_.output.data, weights_.output.type, arena_.d_normed,  // LFM2
             arena_.d_logits, config.vocab_size, H, arena_.stream);  // LFM2
  // LFM2: NO scale/softcap call. // LFM2
  LaunchBatchedGPUArgmax(  // LFM2
      arena_.d_logits, reinterpret_cast<std::uint32_t*>(arena_.d_alpha_buf),  // LFM2
      1, config.vocab_size,  // LFM2
      std::span<float>(arena_.d_ffn_out, arena_.GetMaxBatch() * H),  // LFM2
      arena_.stream);  // LFM2
  std::uint32_t next_token_id = 0;  // LFM2
  if (hipMemcpyAsync(&next_token_id, arena_.d_alpha_buf,  // LFM2
                     sizeof(std::uint32_t), hipMemcpyDeviceToHost,  // LFM2
                     arena_.stream) != hipSuccess ||  // LFM2
      hipStreamSynchronize(arena_.stream) != hipSuccess) {  // LFM2
    throw std::runtime_error("flint prefill token read failed");  // LFM2
  }  // LFM2
  return CheckedSampleToken(next_token_id, config.vocab_size);  // LFM2
}  // LFM2

void QwenGpuExecutor::MaterializeFlintRowLogits(std::size_t row) {  // LFM2
  CheckReset();  // LFM2
  const auto& config = weights_.config;  // LFM2
  if (!config.flint) {
    throw std::runtime_error("flint row logits called for a non-flint config");
  }  // LFM2
  if (row >= arena_.GetMaxBatch()) {  // LFM2
    throw std::runtime_error("flint row logits row is past the batch");  // LFM2
  }  // LFM2
  const std::size_t hidden = config.hidden_size;  // LFM2
  const float* src = arena_.d_hidden + row * hidden;  // LFM2
  LaunchRMSNorm(src, static_cast<const float*>(weights_.output_norm.data),  // LFM2
                arena_.d_normed, hidden, config.norm_eps_pre, arena_.stream);  // LFM2
  LaunchGEMV(weights_.output.data, weights_.output.type, arena_.d_normed,  // LFM2
             arena_.d_logits, config.vocab_size, hidden, arena_.stream);  // LFM2
  // LFM2: NO scale/softcap call. // LFM2
  if (hipStreamSynchronize(arena_.stream) != hipSuccess) {  // LFM2
    throw std::runtime_error("flint row logits synchronize failed");  // LFM2
  }  // LFM2
}  // LFM2

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
