#ifndef SUPERROC_CORE_MODEL_CONFIG_HPP_
#define SUPERROC_CORE_MODEL_CONFIG_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace superroc::core {

/// GLIMMER: attention output gate is a separate GEMV, not the fused 2× q|gate.
enum class AttnGateMode : std::uint8_t { kFused = 0, kSeparate = 1 };

/// GLIMMER: NORM rotates consecutive pairs. Qwen stays NEOX.
enum class RopeVariant : std::uint8_t { kNeox = 0, kNorm = 1 };

/// Unified architectural configuration parsed dynamically from GGUF metadata or
/// JSON.
struct ModelConfig {
  std::string architecture{"qwen35"};
  std::string model_name{"qwen3.5-4b-text"};
  std::uint32_t num_layers{36};  ///< Main transformer layers, excluding MTP.
  std::uint32_t hidden_size{2560};
  std::uint32_t intermediate_size{9728};  ///< 17408 for Qwen3.8-27B.
  std::uint32_t num_attention_heads{16};
  std::uint32_t num_key_value_heads{4};
  std::uint32_t head_dim{256};
  std::uint32_t vocab_size{248320};  ///< 248320 for Qwen3.5/3.8
  std::uint32_t context_length{32768};
  std::uint32_t full_attention_interval{
      4};  ///< 3 linear attention + 1 full attention
  std::uint32_t mtp_num_layers{1};
  std::uint32_t ssm_conv_kernel{4};
  std::uint32_t ssm_state_size{128};
  std::uint32_t ssm_group_count{16};
  std::uint32_t ssm_time_step_rank{32};
  std::uint32_t ssm_inner_size{4096};
  std::uint32_t rotary_dim{64};
  float rope_theta{10000000.0F};
  float rope_scale{1.0F};
  bool is_text_only{true};
  // KAT: qwen35moe routed experts. is_moe is true when expert_count > 0.
  // intermediate_size stays at its default when feed_forward_length is absent.
  std::uint32_t expert_count{0};
  std::uint32_t expert_used_count{0};
  std::uint32_t expert_feed_forward_length{0};
  std::uint32_t expert_shared_feed_forward_length{0};
  bool is_moe{false};

  // GLIMMER: default-off. Qwen leaves these at the defaults below.
  bool attention_only{false};
  AttnGateMode attn_gate_mode{AttnGateMode::kFused};
  RopeVariant rope_variant{RopeVariant::kNeox};
  float norm_eps_pre{1.0e-6F};
  float norm_eps_post{1.0e-8F};
  /// 1 = SWA layer (RoPE on, window = swa_window). 0 = full attention, NoPE.
  std::array<std::uint8_t, 128> use_rope{};
  std::uint32_t swa_window{0};
  float logit_scale{1.0F};
  float final_softcap{0.0F};
  // LFM2: default-off. LiquidAI lfm2moe on the qwen chassis (short-conv + GQA
  // attention, sigmoid-routed MoE).
  bool lfm2{false};  // LFM2
  /// LFM2: 1 = attention layer, 0 = short-conv layer (head_count_kv[i] > 0).
  std::array<std::uint8_t, 128> lfm2_attn{};  // LFM2
  /// LFM2: ordinal of the layer within its kind. Attention layers index the KV
  /// cache with it, conv layers index the conv state with it.
  std::array<std::uint8_t, 128> lfm2_kind_index{};  // LFM2
  /// LFM2: layers [0, dense_lead_layers) use the dense SwiGLU FFN; the rest are
  /// MoE.
  std::uint32_t dense_lead_layers{0};  // LFM2
  /// LFM2: short-conv kernel length (GGUF shortconv.l_cache). The state keeps
  /// l_cache - 1 columns.
  std::uint32_t shortconv_l_cache{0};  // LFM2
  /// LFM2: sigmoid router; selection adds exp_probs_b; weights are the unbiased
  /// scores normalized over the top-k.
  bool moe_sigmoid_bias{false};  // LFM2
  /// Flint: RecursiveSI/flint on the qwen chassis (sliding/NoPE GQA, sigmoid
  /// MoE, attention sinks).
  bool flint{false};
  /// Flint: per-head softmax sinks (`blk.N.attn_sinks`) participate in
  /// decode/prefill softmax.
  bool attn_sinks{false};
  /// Llama: plain dense transformer (GQA + SwiGLU in every layer, full causal
  /// attention, adjacent-pair RoPE, untied output).
  bool llama{false};

  [[nodiscard]] constexpr std::uint32_t AttentionSize() const noexcept {
    const std::uint64_t n =
        static_cast<std::uint64_t>(num_attention_heads) * head_dim;
    return n > 0xFFFFFFFFULL ? 0U : static_cast<std::uint32_t>(n);  // audit F05
  }

  [[nodiscard]] constexpr std::uint32_t FullAttentionLayerCount()
      const noexcept {
    if (flint || llama) {
      return num_layers;
    }
    if (lfm2) {
      std::uint32_t n = 0;
      for (std::uint32_t i = 0; i < num_layers && i < lfm2_attn.size(); ++i)
        n += lfm2_attn[i] != 0 ? 1U : 0U;
      return n;
    }  // LFM2
    return full_attention_interval == 0 ? 0
                                        : num_layers / full_attention_interval;
  }

  [[nodiscard]] constexpr std::uint32_t SsmLayerCount() const noexcept {
    return num_layers - FullAttentionLayerCount();
  }

  /// Dense state index for a recurrent transformer layer.
  [[nodiscard]] constexpr std::uint32_t SsmLayerIndex(
      std::uint32_t layer) const noexcept {
    if (lfm2)
      return Lfm2KindIndex(layer);  // LFM2
    return full_attention_interval == 0
               ? layer
               : layer - layer / full_attention_interval;
  }

  [[nodiscard]] constexpr std::uint32_t SsmValueSize() const noexcept {
    return ssm_time_step_rank == 0 ? 0 : ssm_inner_size / ssm_time_step_rank;
  }

  [[nodiscard]] constexpr std::uint32_t SsmQkvSize() const noexcept {
    return (2 * ssm_group_count * ssm_state_size) + ssm_inner_size;
  }

  /// GLIMMER: pattern true means RoPE + sliding window. False means NoPE.
  [[nodiscard]] constexpr bool LayerUsesRope(
      std::uint32_t layer) const noexcept {
    return layer < use_rope.size() && use_rope[layer] != 0;
  }

  /// GLIMMER: 0 means full (unwindowed) attention.
  [[nodiscard]] constexpr std::uint32_t LayerSwaWindow(
      std::uint32_t layer) const noexcept {
    return LayerUsesRope(layer) ? swa_window : 0U;
  }

  /// LFM2: true for an attention layer of an lfm2 config.
  [[nodiscard]] constexpr bool LayerIsLfm2Attention(
      std::uint32_t layer) const noexcept {  // LFM2
    return lfm2 && layer < lfm2_attn.size() && lfm2_attn[layer] != 0;
  }
  /// LFM2: ordinal within the layer's kind (KV slot for attention, conv-state
  /// slot for conv).
  [[nodiscard]] constexpr std::uint32_t Lfm2KindIndex(
      std::uint32_t layer) const noexcept {  // LFM2
    return layer < lfm2_kind_index.size() ? lfm2_kind_index[layer] : 0U;
  }
  /// LFM2: floats of conv state per conv layer: (l_cache - 1) columns of
  /// hidden_size.
  [[nodiscard]] constexpr std::size_t Lfm2ConvStateFloats()
      const noexcept {  // LFM2
    return shortconv_l_cache < 2
               ? 0
               : static_cast<std::size_t>(shortconv_l_cache - 1) * hidden_size;
  }

  /// Returns true if this configuration conforms to the Qwen3.5/3.8 repeating
  /// block structure, or to the muse-glimmer attention-only block.
  [[nodiscard]] constexpr bool IsValidQwen() const noexcept {
    // Llama: plain dense geometry. Every layer is full causal attention, the
    // FFN is dense SwiGLU, RoPE rotates adjacent pairs and the output head is
    // untied.
    if (llama) {
      return num_layers > 0 && num_layers <= use_rope.size() &&
             hidden_size > 0 && intermediate_size > 0 &&
             num_attention_heads > 0 && num_key_value_heads > 0 &&
             num_key_value_heads <= num_attention_heads &&
             (num_attention_heads % num_key_value_heads) == 0 &&
             AttentionSize() > 0 && vocab_size > 0 && rotary_dim == head_dim &&
             (rotary_dim % 2) == 0 && norm_eps_pre > 0.0F &&
             rope_theta > 0.0F && context_length > 0 &&
             full_attention_interval == 1 &&
             rope_variant == RopeVariant::kNorm && !is_moe && !flint && !lfm2 &&
             !attention_only && is_text_only;
    }
    if (flint) {
      const bool ffn_ok =
          is_moe && expert_count > 0 && expert_used_count > 0 &&
          expert_used_count <= expert_count && expert_feed_forward_length > 0 &&
          expert_shared_feed_forward_length > 0 && dense_lead_layers > 0 &&
          dense_lead_layers < num_layers && moe_sigmoid_bias;
      return num_layers > 0 && num_layers <= use_rope.size() &&
             hidden_size > 0 && head_dim > 0 && intermediate_size > 0 &&
             num_attention_heads > 0 && num_key_value_heads > 0 &&
             num_key_value_heads <= num_attention_heads &&
             (num_attention_heads % num_key_value_heads) == 0 &&
             AttentionSize() > 0 && vocab_size > 0 && ffn_ok &&
             rotary_dim == head_dim && (rotary_dim % 2) == 0 &&
             rope_variant == RopeVariant::kNeox && swa_window > 0 &&
             attn_sinks && norm_eps_pre > 0.0F && rope_theta > 0.0F &&
             context_length > 0 && is_text_only;
    }
    // LFM2: own structural rule; the interval-4 / DeltaNet rule below does not
    // apply.
    if (lfm2) {  // LFM2
      // Super ROC: lfm2 (dense: every layer SwiGLU) or lfm2moe (MoE after
      // dense_lead_layers).
      const bool ffn_ok =
          is_moe ? (expert_count > 0 && expert_count <= 256 &&
                    expert_used_count > 0 && expert_used_count <= 8 &&
                    expert_used_count <= expert_count &&
                    expert_feed_forward_length > 0 &&
                    expert_shared_feed_forward_length == 0 &&
                    dense_lead_layers < num_layers && moe_sigmoid_bias)
                 : (expert_count == 0 && expert_used_count == 0 &&
                    dense_lead_layers == num_layers && !moe_sigmoid_bias);
      return num_layers > 0 && num_layers <= lfm2_attn.size() &&
             hidden_size > 0 && head_dim > 0 && intermediate_size > 0 &&
             num_attention_heads > 0 && num_key_value_heads > 0 &&
             num_key_value_heads <= num_attention_heads &&
             (num_attention_heads % num_key_value_heads) == 0 &&
             AttentionSize() == hidden_size && vocab_size > 0 && ffn_ok &&
             shortconv_l_cache >= 2 && shortconv_l_cache <= 8 &&
             FullAttentionLayerCount() > 0 &&
             FullAttentionLayerCount() < num_layers && rotary_dim == head_dim &&
             (rotary_dim % 2) == 0 && norm_eps_pre > 0.0F &&
             rope_theta > 0.0F && rope_theta < 3.0e38F && context_length > 0 &&
             is_text_only && ssm_conv_kernel == shortconv_l_cache - 1 &&
             SsmQkvSize() == hidden_size && ssm_state_size == 0;
    }
    // GLIMMER: 52 all-attention layers must not take the interval-4 / SSM rule.
    if (attention_only) {
      return head_dim > 0 && num_layers > 0 && num_layers <= use_rope.size() &&
             hidden_size > 0 && intermediate_size > 0 &&
             num_attention_heads > 0 && num_key_value_heads > 0 &&
             vocab_size > 0 &&
             (num_attention_heads % num_key_value_heads) == 0 &&
             full_attention_interval == 1 && rotary_dim > 0 &&
             (rotary_dim % 2) == 0 && rotary_dim <= head_dim && is_text_only &&
             swa_window > 0 && norm_eps_pre > 0.0F && norm_eps_post > 0.0F &&
             attn_gate_mode == AttnGateMode::kSeparate &&
             rope_variant == RopeVariant::kNorm && logit_scale != 0.0F &&
             ssm_time_step_rank > 0;
    }
    // KAT: a MoE trunk has no dense feed_forward_length. Require the expert
    // sizes instead of intermediate_size, which stays at the struct default.
    const bool ffn_ok = is_moe ? (expert_count > 0 && expert_used_count > 0 &&
                                  expert_used_count <= expert_count &&
                                  expert_feed_forward_length > 0 &&
                                  expert_shared_feed_forward_length > 0)
                               : (intermediate_size > 0);
    return head_dim > 0 && num_layers > 0 && hidden_size > 0 && ffn_ok &&
           num_attention_heads > 0 && num_key_value_heads > 0 &&
           num_key_value_heads <= num_attention_heads &&
           (num_attention_heads % num_key_value_heads) == 0 &&
           num_layers <= 512 &&  // audit F02 F09
           vocab_size > 0 && full_attention_interval == 4 &&
           (num_layers % full_attention_interval) == 0 && rotary_dim > 0 &&
           rotary_dim <= head_dim && (rotary_dim % 2) == 0 &&
           ssm_conv_kernel == 4 && ssm_state_size == 128 &&
           ssm_group_count > 0 && ssm_time_step_rank > 0 &&
           ssm_inner_size > 0 && (ssm_inner_size % ssm_time_step_rank) == 0 &&
           SsmValueSize() == 128 && is_text_only;
  }
};

}  // namespace superroc::core

#endif  // SUPERROC_CORE_MODEL_CONFIG_HPP_
