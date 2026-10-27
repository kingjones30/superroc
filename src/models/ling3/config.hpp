#ifndef SUPERROC_MODELS_LING3_CONFIG_HPP_
#define SUPERROC_MODELS_LING3_CONFIG_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace superroc::models::ling3 {

/// Architecture parameters of a `bailingmoe3` GGUF artifact (Ling-3.0).
///
/// Every value below is read from the file; nothing is defaulted from another
/// model. `FromGguf` is fail-closed: any missing, malformed, or out-of-contract
/// value leaves the whole config unset and fills `error_msg`.
///
/// The layer map is the `attention.head_count_kv` array, which carries one
/// entry per block: 0 marks a KDA (recurrent) layer, non-zero a gated-MLA
/// layer. It is read verbatim rather than inferred from a stride, because the
/// MTP block and the trunk can disagree about spacing.
struct Config {
  std::uint32_t block_count{0};     ///< n_layer, includes the MTP block (43).
  std::uint32_t num_layers{0};      ///< Trunk layers, MTP excluded (42).
  std::uint32_t nextn_layers{0};    ///< 1; the MTP block stays unused.
  std::uint32_t hidden_size{0};     ///< n_embd (2560).
  std::uint32_t vocab_size{0};      ///< 157184.
  std::uint32_t context_length{0};  ///< 262144.
  float rms_eps{1e-6F};

  // Shared attention geometry.
  std::uint32_t num_heads{0};  ///< 32.

  // KDA (linear attention) layers.
  std::uint32_t kda_head_dim{0};     ///< 128; state key and value dim.
  std::uint32_t kda_conv_kernel{0};  ///< 4.
  float kda_gate_lower_bound{0.0F};  ///< -5.0; must be negative.

  // Gated MLA layers.
  std::uint32_t mla_kv_lora_rank{0};  ///< 512.
  std::uint32_t mla_head_dim_k{0};    ///< 192.
  std::uint32_t mla_head_dim_v{0};    ///< 128.
  std::uint32_t rotary_dim{0};        ///< 64.
  std::uint32_t rope_mode{0};         ///< 0 = NORM (consecutive pairs).
  float rope_theta{0.0F};             ///< 6e6.

  // Feed forward.
  std::uint32_t ffn_length{0};          ///< 6144, the leading dense blocks.
  std::uint32_t dense_lead_layers{0};   ///< 2.
  std::uint32_t num_experts{0};         ///< 512.
  std::uint32_t num_experts_used{0};    ///< 8.
  std::uint32_t num_expert_groups{0};   ///< 8.
  std::uint32_t num_group_used{0};      ///< 4.
  std::uint32_t expert_ff{0};           ///< 768.
  std::uint32_t shared_expert_ff{0};    ///< 768.
  std::uint32_t num_shared_experts{0};  ///< 1.
  float expert_weights_scale{1.0F};     ///< 2.5.
  bool expert_weights_norm{false};      ///< true.
  std::uint32_t expert_gating_func{0};  ///< 2 = sigmoid.

  /// Per-block SwiGLU clamps, indexed by block. A limit at or below 1e-6 means
  /// the fork emits no clamp at all and the plain silu(gate) * up applies.
  std::vector<float> swiglu_clamp_exp;    ///< routed experts, size block_count.
  std::vector<float> swiglu_clamp_shexp;  ///< shared expert, size block_count.

  /// Raw head_count_kv, one entry per block. Drives IsMla().
  std::vector<std::uint32_t> head_count_kv;

  [[nodiscard]] std::uint32_t KdaInnerSize() const noexcept {
    return num_heads * kda_head_dim;
  }
  /// Key part of the MLA query head that does not rotate.
  [[nodiscard]] std::uint32_t MlaQkNopeDim() const noexcept {
    return mla_head_dim_k - rotary_dim;
  }
  /// Output width of wkv_a_mqa: the compressed latent plus the shared rope
  /// part.
  [[nodiscard]] std::uint32_t MlaKvAOutDim() const noexcept {
    return mla_kv_lora_rank + rotary_dim;
  }
  /// Query projection width of the absorbed MLA path.
  [[nodiscard]] std::uint32_t MlaQOutDim() const noexcept {
    return num_heads * mla_head_dim_k;
  }
  [[nodiscard]] std::uint32_t MlaOutDim() const noexcept {
    return num_heads * mla_head_dim_v;
  }
  /// Experts per routing group. Expert ids are laid out contiguously by group.
  [[nodiscard]] std::uint32_t ExpertsPerGroup() const noexcept {
    return num_expert_groups ? num_experts / num_expert_groups : 0;
  }

  /// A block is gated MLA when its head_count_kv entry is non-zero. The MTP
  /// block is always full attention and is never marked recurrent.
  [[nodiscard]] bool IsMla(std::uint32_t block) const noexcept {
    return block < num_layers && block < head_count_kv.size() &&
           head_count_kv[block] != 0;
  }
  /// The clamp is skipped entirely by the fork when the limit is this small.
  [[nodiscard]] static bool ClampActive(float limit) noexcept {
    return limit > 1e-6F;
  }

  /// Reads and validates the `bailingmoe3.*` metadata. Fails closed on any
  /// missing required key, on a block_count that disagrees with
  /// head_count_kv, or on a routing configuration the expert math cannot
  /// satisfy (for example num_experts_used above num_group_used *
  /// ExpertsPerGroup()).
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace superroc::models::ling3

#endif  // SUPERROC_MODELS_LING3_CONFIG_HPP_
