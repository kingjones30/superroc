#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/detail/decode_step.hpp"

#include <stdexcept>  // KAT:

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/rocm_fp4/superroc_fp4_gemv_q8.hpp"  // KAT:
#include "src/core/rocm_fp4/superroc_fp4_quant.hpp"    // KAT:
#include "src/models/flint/hip/flint_decode.hpp"
#include "src/models/glimmer/hip/glimmer_decode.hpp"  // GLIMMER
#include "src/models/lfm2/hip/lfm2_decode.hpp"        // LFM2
#include "src/models/llama/hip/llama_decode.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/modules/modules.hpp"
#include "src/models/qwen/moe/qwen_moe_ffn.hpp"    // KAT:
#include "src/models/qwen/moe/qwen_moe_route.hpp"  // KAT:

namespace superroc::hip {

void EmitDecodeRouteTelemetry(const models::QwenModelWeights& weights,
                              const QwenExecutionPolicy& policy) {
  if (!detail::DispatchTelemetryEnabled()) {
    return;
  }
  for (std::uint32_t layer_index = 0; layer_index < weights.config.num_layers;
       ++layer_index) {
    const auto& layer = weights.layers[layer_index];
    const auto resolution = ResolveQwenLayerRouteWithReasons(
        policy, QwenExecutionMode::kDecode, layer.is_full_attention);
    detail::EmitQwenRouteResolution(
        "decode", layer_index, layer.is_full_attention ? "attention" : "ssm",
        resolution.plan.Fingerprint(),
        static_cast<std::uint32_t>(resolution.rejected));
  }
}

void ExecuteDecodeStep(QwenGpuArena& arena,
                       const models::QwenModelWeights& weights,
                       const QwenExecutionPolicy& policy,
                       tokenization::TokenId token_id, std::uint32_t pos,
                       bool compute_logits,
                       QwenRecurrentStateIo recurrent_state,
                       models::qwen::vision::DeviceInput* image_input) {
  // GLIMMER: sandwich norms, separate gate, NORM RoPE, SWA. Not graph-safe.
  if (weights.config.attention_only) {
    ExecuteGlimmerDecodeStep(arena, weights, token_id, pos, compute_logits,
                             image_input);
    return;
  }
  if (weights.config.lfm2) {  // LFM2
    ExecuteLfm2DecodeStep(arena, weights, token_id, pos, compute_logits,
                          image_input);
    return;
  }
  if (weights.config.flint) {
    ExecuteFlintDecodeStep(arena, weights, token_id, pos, compute_logits,
                           image_input);
    return;
  }
  if (weights.config.llama) {
    ExecuteLlamaDecodeStep(arena, weights, token_id, pos, compute_logits,
                           image_input);
    return;
  }
  const auto* rope = image_input != nullptr ? image_input->rope() : nullptr;
  (void)token_id;
  const auto& config = weights.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;
  auto scratch = arena.GetScratchView();
  auto& decode_scratch = scratch.decode;
  auto& attention_scratch = scratch.attention;
  auto& ssm_scratch = scratch.ssm;
  auto& ffn_scratch = scratch.ffn;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const bool use_split_k_decode = detail::IsSplitKDecodeAttentionSupported(
      sequence_length, config.num_attention_heads, config.num_key_value_heads,
      config.head_dim);

  // GPU parameter buffers
  const std::uint32_t* d_in_token = decode_scratch.prompt_tokens.data();
  const std::uint32_t* d_in_pos = decode_scratch.prompt_tokens.data() + 1;
  auto* d_out_token = decode_scratch.sampled_token.data();

  // 1. Embedding lookup
  LaunchEmbeddingLookup(weights.token_embd.data, weights.token_embd.type,
                        d_in_token, decode_scratch.hidden.data(), hidden_size,
                        arena.stream);
  arena.SnapVerifyStage("emb", detail::kVerifyDebugNoLayer,
                        decode_scratch.hidden.data(),
                        decode_scratch.hidden.size(), 1);

  if (image_input != nullptr)
    image_input->Inject(decode_scratch.hidden.data(), pos, 1, hidden_size, 1,
                        arena.stream);

  // 2. Layer stack
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const auto& layer = weights.layers[l];
    const auto route_plan = ResolveQwenLayerRoute(
        policy, QwenExecutionMode::kDecode, layer.is_full_attention);
    const superroc::models::qwen::HipModuleContext module_ctx(
        static_cast<void*>(arena.stream), l, pos);

    {
      // Pre-RMSNorm, routed through the norm module (HIP backend). Same
      // kernel, same args, same arena slices (d_hidden in, d_normed out);
      // behavior identical to the former inline `LaunchRMSNorm` call.
      const auto norm_view =
          superroc::models::qwen::MakeAttnNormView(layer, config);
      superroc::models::qwen::NormForward(
          module_ctx, norm_view, decode_scratch.hidden, decode_scratch.normed);
    }
    arena.SnapVerifyStage("attn_norm", l, decode_scratch.normed.data(),
                          decode_scratch.normed.size(), 1);

    if (layer.is_full_attention) {
      // Full attention path
      const std::size_t total_k = arena.GetAttentionKvPlaneElements();
      const std::uint32_t attn_layer_idx = l / config.full_attention_interval;

      LaunchFusedQKVProjections(
          layer.attn_q.data, layer.attn_q.type, layer.attn_k.data,
          layer.attn_k.type, layer.attn_v.data, layer.attn_v.type,
          decode_scratch.normed.data(), ssm_scratch.qkv.data(),
          attention_scratch.k.data(), attention_scratch.v.data(),
          q_projection_size, kv_size, hidden_size, arena.stream);
      // De-interleave Q and Gate from attn_q projection
      LaunchUnpackQG(ssm_scratch.qkv.data(), attention_scratch.q.data(),
                     ssm_scratch.gate.data(), config.num_attention_heads,
                     config.head_dim, arena.stream);
      arena.SnapVerifyStage("q", l, attention_scratch.q.data(),
                            attention_scratch.q.size(), 1);
      arena.SnapVerifyStage("k", l, attention_scratch.k.data(),
                            attention_scratch.k.size(), 1);
      arena.SnapVerifyStage("v", l, attention_scratch.v.data(),
                            attention_scratch.v.size(), 1);
      arena.SnapVerifyStage("gate", l, ssm_scratch.gate.data(),
                            ssm_scratch.gate.size(), 1);

      // QK-Norm + RoPE + KV-cache write fused into one kernel
      // (opt-c010-qk-rope-kv). The unfused chain stays wired behind the
      // policy toggle as the independent reference.
      const bool fused_qknorm_rope_kv =
          route_plan.fuse_qk_norm_rope_kv &&
          detail::IsFusedQkNormSupported(config.head_dim);
      if (fused_qknorm_rope_kv) {
        LaunchFusedQKNormRoPEKvWrite(
            attention_scratch.q.data(), attention_scratch.k.data(),
            attention_scratch.v.data(),
            static_cast<const float*>(layer.attn_q_norm.data),
            static_cast<const float*>(layer.attn_k_norm.data),
            attention_scratch.q.data(), attention_scratch.k.data(),
            arena.d_kv_cache, OffsetIfPresent(arena.d_kv_cache, total_k),
            arena.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(arena.d_attention_kv_f16), total_k),
            attn_layer_idx, d_in_pos, arena.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, config.rotary_dim, config.rope_theta, 1e-6F,
            arena.stream, rope);
        arena.SnapVerifyStage("qkrope_q", l, attention_scratch.q.data(),
                              attention_scratch.q.size(), 1);
        arena.SnapVerifyStage("qkrope_k", l, attention_scratch.k.data(),
                              attention_scratch.k.size(), 1);
      } else {
        if (!layer.attn_q_norm.empty()) {
          LaunchPerHeadRMSNorm(
              attention_scratch.q.data(),
              static_cast<const float*>(layer.attn_q_norm.data),
              attention_scratch.q.data(), config.num_attention_heads,
              config.head_dim, 1e-6F, arena.stream);
        }
        if (!layer.attn_k_norm.empty()) {
          LaunchPerHeadRMSNorm(
              attention_scratch.k.data(),
              static_cast<const float*>(layer.attn_k_norm.data),
              attention_scratch.k.data(), config.num_key_value_heads,
              config.head_dim, 1e-6F, arena.stream);
        }

        arena.SnapVerifyStage("qkn_q", l, attention_scratch.q.data(),
                              attention_scratch.q.size(), 1);
        arena.SnapVerifyStage("qkn_k", l, attention_scratch.k.data(),
                              attention_scratch.k.size(), 1);
        // RoPE (using device pos pointer for graph capture invariance)
        LaunchRoPE(attention_scratch.q.data(), attention_scratch.k.data(),
                   config.num_attention_heads, config.num_key_value_heads,
                   config.head_dim, config.rotary_dim, d_in_pos,
                   config.rope_theta, arena.stream, rope);
        arena.SnapVerifyStage("rope_q", l, attention_scratch.q.data(),
                              attention_scratch.q.size(), 1);
        arena.SnapVerifyStage("rope_k", l, attention_scratch.k.data(),
                              attention_scratch.k.size(), 1);
      }

      // Softmax Attention + Gating
      if (use_split_k_decode) {
        LaunchAttention(
            attention_scratch.q.data(), attention_scratch.k.data(),
            attention_scratch.v.data(), ssm_scratch.gate.data(),
            arena.d_kv_cache, OffsetIfPresent(arena.d_kv_cache, total_k),
            arena.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(arena.d_attention_kv_f16), total_k),
            ssm_scratch.out.data(), attn_layer_idx, pos, arena.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, arena.stream, attention_scratch.split_k.data(),
            fused_qknorm_rope_kv);
      } else {
        LaunchAttention(
            attention_scratch.q.data(), attention_scratch.k.data(),
            attention_scratch.v.data(), ssm_scratch.gate.data(),
            arena.d_kv_cache, OffsetIfPresent(arena.d_kv_cache, total_k),
            arena.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(arena.d_attention_kv_f16), total_k),
            ssm_scratch.out.data(), attn_layer_idx, d_in_pos,
            arena.GetMaxContext(), config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, arena.stream,
            fused_qknorm_rope_kv);
      }

      arena.SnapVerifyStage("attn_out", l, ssm_scratch.out.data(),
                            ssm_scratch.out.size(), 1);
      // Output projection, routed through the quant_gemm module (HIP
      // backend). Same kernel, same args, same arena slices (d_ssm_out in,
      // d_attn_out out); behavior-identical to the former inline `LaunchGEMV`.
      superroc::models::qwen::QuantGemm(
          module_ctx, layer.attn_output, ssm_scratch.out.first(attention_size),
          hidden_size, attention_size, attention_scratch.output);
      arena.SnapVerifyStage("mix_proj", l, attention_scratch.output.data(),
                            attention_scratch.output.size(), 1);
    } else {
      // SSM path

      LaunchFusedSSMInputProjections(
          layer.attn_qkv.data, layer.attn_qkv.type, layer.attn_gate.data,
          layer.attn_gate.type, layer.ssm_alpha.data, layer.ssm_alpha.type,
          layer.ssm_beta.data, layer.ssm_beta.type,
          decode_scratch.normed.data(), ssm_scratch.qkv.data(),
          ssm_scratch.gate.data(), ssm_scratch.alpha.data(),
          ssm_scratch.beta.data(), hidden_size, ssm_qkv_size, ssm_inner_size,
          time_step_rank, arena.stream);
      arena.SnapVerifyStage("qkv", l, ssm_scratch.qkv.data(),
                            ssm_scratch.qkv.size(), 1);
      arena.SnapVerifyStage("gate", l, ssm_scratch.gate.data(),
                            ssm_scratch.gate.size(), 1);
      arena.SnapVerifyStage("alpha", l, ssm_scratch.alpha.data(),
                            ssm_scratch.alpha.size(), 1);
      arena.SnapVerifyStage("beta", l, ssm_scratch.beta.data(),
                            ssm_scratch.beta.size(), 1);
      LaunchSSMConvRecurrence(
          ssm_scratch.qkv.data(),
          static_cast<const float*>(layer.ssm_conv1d.data), recurrent_state,
          ssm_scratch.conv_out.data(), ssm_scratch.alpha.data(),
          ssm_scratch.beta.data(), static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data),
          static_cast<const float*>(layer.ssm_norm.data),
          ssm_scratch.gate.data(), ssm_scratch.out.data(),
          config.SsmLayerIndex(l), ssm_qkv_size, config.ssm_group_count,
          config.ssm_time_step_rank, config.ssm_state_size,
          config.SsmValueSize(), arena.stream, arena.GetSsmReplayCapture(),
          arena.GetRecurrentStateStorage());

      arena.SnapVerifyStage("conv_out", l, ssm_scratch.conv_out.data(),
                            ssm_scratch.conv_out.size(), 1);
      arena.SnapVerifyStage("ssm_out", l, ssm_scratch.out.data(),
                            ssm_scratch.out.size(), 1);
      LaunchGEMV(layer.ssm_out.data, layer.ssm_out.type, ssm_scratch.out.data(),
                 attention_scratch.output.data(), hidden_size, ssm_inner_size,
                 arena.stream);
      arena.SnapVerifyStage("mix_proj", l, attention_scratch.output.data(),
                            attention_scratch.output.size(), 1);
    }

    {
      superroc::models::qwen::ResidualAdd(module_ctx, decode_scratch.hidden,
                                          attention_scratch.output);

      {
        // FFN Pre-RMSNorm
        LaunchRMSNorm(decode_scratch.hidden.data(),
                      static_cast<const float*>(layer.ffn_norm.data),
                      decode_scratch.normed.data(), hidden_size, 1e-6F,
                      arena.stream);
      }
      arena.SnapVerifyStage("ffn_norm", l, decode_scratch.normed.data(),
                            decode_scratch.normed.size(), 1);
    }

    // Fused SwiGLU FFN. KAT S3: a MoE layer has empty dense projections; it
    // runs the fast router, the 8 routed experts and the sigmoid-gated shared
    // expert through the rocm_fp4 grouped FFN, then the combine. Every host
    // status is checked here and the expert kernels' device status word is
    // read once per token below, so a failure throws instead of contributing
    // a silent zero.
    if (layer.is_moe) {
      const int moe_expert = static_cast<int>(config.expert_count);
      const int moe_used = static_cast<int>(config.expert_used_count);
      const int moe_ff = static_cast<int>(config.expert_feed_forward_length);
      const int moe_ffs =
          static_cast<int>(config.expert_shared_feed_forward_length);
      const int moe_hidden = static_cast<int>(hidden_size);
      // QVB: the route and the shared-expert gate in one launch pair
      // (bit-identical to LaunchQwenMoeRoute + LaunchQwenMoeShexpSigmoid).
      if (LaunchQwenMoeRouteRows(
              decode_scratch.normed.data(),
              static_cast<const float*>(layer.ffn_gate_inp.data), moe_hidden,
              moe_expert, 1, moe_used, arena.d_moe_logits, arena.d_moe_ids,
              arena.d_moe_weights,
              static_cast<const float*>(layer.ffn_gate_inp_shexp.data),
              arena.d_moe_sig, arena.stream) != 0) {
        throw std::runtime_error("qwen35moe decode route launch failed");
      }
      if (superroc_fp4_expert_ffn_dev(
              static_cast<int>(layer.ffn_gate_exps.type),
              layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
              layer.ffn_down_exps.data, decode_scratch.normed.data(),
              arena.d_moe_ids, arena.d_moe_weights, moe_used, moe_expert,
              moe_hidden, moe_ff, 0.0F, arena.d_moe_y,
              arena.stream) != SUPERROC_FP4_OK) {
        throw std::runtime_error("qwen35moe decode experts failed");
      }
      // QSM: same normed x on the same stream as the routed call above, so
      // the shared expert reuses its Q8 copy (bit-identical, one launch less).
      if (superroc_fp4_expert_ffn_dev_xq(
              static_cast<int>(layer.ffn_gate_shexp.type),
              layer.ffn_gate_shexp.data, layer.ffn_up_shexp.data,
              layer.ffn_down_shexp.data, decode_scratch.normed.data(),
              arena.d_moe_zero_ids, arena.d_moe_sig, 1, 1, moe_hidden, moe_ffs,
              0.0F, arena.d_moe_shexp_y, arena.stream) != SUPERROC_FP4_OK) {
        throw std::runtime_error("qwen35moe decode shared expert failed");
      }
      if (LaunchQwenMoeCombine(arena.d_moe_y, arena.d_moe_shexp_y, moe_hidden,
                               1, ffn_scratch.out.data(), arena.stream) != 0) {
        throw std::runtime_error("qwen35moe decode combine failed");
      }
    } else {
      const auto ffn_view = superroc::models::qwen::MakeFfnView(layer, config);
      superroc::models::qwen::FfnForward(
          module_ctx, ffn_view, decode_scratch.normed, ffn_scratch.activation,
          ffn_scratch.activation, ffn_scratch.activation, ffn_scratch.out);
    }
    arena.SnapVerifyStage("ffn_out", l, ffn_scratch.out.data(),
                          ffn_scratch.out.size(), 1);

    // Residual Add
    superroc::models::qwen::ResidualAdd(module_ctx, decode_scratch.hidden,
                                        ffn_scratch.out);
    arena.SnapVerifyStage("hidden", l, decode_scratch.hidden.data(),
                          decode_scratch.hidden.size(), 1);

    if (const auto tap_index = arena.GetTargetLayerCaptureIndex(l);
        tap_index.has_value()) {
      HIP_CHECK(hipMemcpyAsync(
          arena.d_target_layer_features + (*tap_index * hidden_size),
          decode_scratch.hidden.data(), hidden_size * sizeof(float),
          hipMemcpyDeviceToDevice, arena.stream));
    }
  }

  if (config.is_moe) {
    // KAT S3: the grouped expert FFN reports device-side errors (a bad router
    // id) through a status word; read and clear it once per token, never per
    // layer. A non-OK status means this token's MoE output was invalid.
    // SQ10: name this stream's set, so concurrent sessions check only theirs.
    // SPEC2 (R2): a refusal fails the step loudly; it cannot happen in
    // serve (the session stream pre-attached at load).
    const int slot_set = superroc_fp4_slot_stream_attach_at(
        arena.stream, __func__, static_cast<long>(hidden_size));
    if (slot_set < 0) {
      throw std::runtime_error("qwen35moe decode slot set refused");
    }
    // SPEC2 (R6): stream-ordered check (the bare read proved stale on the
    // box: STALE_STATUS missed 1/200).
    if (superroc_fp4_moe_dev_check_set_on_stream(slot_set, arena.stream) !=
        SUPERROC_FP4_OK) {
      throw std::runtime_error("qwen35moe decode expert device status");
    }
  }

  if (compute_logits) {
    // 3. Final Output Norm
    LaunchRMSNorm(decode_scratch.hidden.data(),
                  static_cast<const float*>(weights.output_norm.data),
                  decode_scratch.normed.data(), hidden_size, 1e-6F,
                  arena.stream);
    arena.SnapVerifyStage("final_norm", detail::kVerifyDebugNoLayer,
                          decode_scratch.normed.data(),
                          decode_scratch.normed.size(), 1);

    // 4. LM Head Logits GEMV on final token
    LaunchGEMV(weights.output.data, weights.output.type,
               decode_scratch.normed.data(), decode_scratch.logits.data(),
               vocab_size, hidden_size, arena.stream);
    arena.SnapVerifyStage("logits", detail::kVerifyDebugNoLayer,
                          decode_scratch.logits.data(),
                          decode_scratch.logits.size(), 1);

    // 5. Parallel GPU Argmax
    LaunchBatchedGPUArgmax(decode_scratch.logits.data(), d_out_token, 1,
                           vocab_size, ffn_scratch.out, arena.stream);
  }
}

}  // namespace superroc::hip
#endif  // defined(ENGINE_ENABLE_HIP)
