#include "src/models/ling3/config.hpp"

#include <cmath>
#include <limits>
#include <utility>
#include <variant>

#include "src/core/model_family.hpp"

namespace superroc::models::ling3 {
namespace {

constexpr std::string_view kArchitecture =
    superroc::core::ModelFamilyArch(superroc::core::ModelFamilyId::kBailingMoe3);

/// GGML_ROPE_TYPE_NORM: pairs of CONSECUTIVE head values (2i, 2i+1) -- see
/// llama_model_rope_type, which puts BAILINGMOE3 under "a normal RoPE,
/// operating on pairs of consecutive head values", and ggml-cuda/rope.cu
/// rope_norm. NEOX is the half-split pairing (i, i + n_dims/2).
constexpr std::uint32_t kRopeModeNorm = 0;
/// LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID.
constexpr std::uint32_t kGatingSigmoid = 2;

/// superroc's GgufReader does not prepend the architecture prefix, so every
/// bailingmoe3.* key is spelled out here once.
constexpr std::string_view kPrefix = "bailingmoe3.";

struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  [[nodiscard]] static std::string Key(std::string_view key) {
    return std::string(kPrefix) + std::string(key);
  }

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  std::uint32_t U32(std::string_view key) {
    const auto value = gguf.GetMetadataUint32(Key(key));
    if (!value.has_value()) {
      Fail("missing or invalid GGUF integer " + Key(key));
      return 0;
    }
    return *value;
  }

  std::uint32_t U32Opt(std::string_view key, std::uint32_t fallback) {
    const auto value = gguf.GetMetadataUint32(Key(key));
    if (value.has_value()) {
      return *value;
    }
    if (gguf.FindMetadata(Key(key)) != nullptr) {
      Fail("invalid GGUF integer " + Key(key));
    }
    return fallback;
  }

  float F32(std::string_view key) {
    const auto value = gguf.GetMetadataFloat32(Key(key));
    if (!value || !std::isfinite(*value)) {
      Fail("missing or invalid GGUF float " + Key(key));
      return 0.0F;
    }
    return *value;
  }

  /// Optional float: absent is not an error here, the caller decides.
  std::optional<float> MaybeF32(std::string_view key) const {
    return gguf.GetMetadataFloat32(Key(key));
  }

  bool Bool(std::string_view key, bool fallback) {
    const auto value = gguf.GetMetadataBool(Key(key));
    if (value.has_value()) {
      return *value;
    }
    if (gguf.FindMetadata(Key(key)) != nullptr) {
      Fail("invalid GGUF bool " + Key(key));
    }
    return fallback;
  }

  /// Integer arrays of any width; GGUF writers pick the narrowest type.
  std::vector<std::uint64_t> U64Array(std::string_view key) {
    const auto* meta = gguf.FindMetadata(Key(key));
    std::vector<std::uint64_t> out;
    if (meta == nullptr) {
      Fail("missing GGUF array " + Key(key));
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (auto v : *s) {
        if (v < 0) {
          Fail("negative GGUF array value in " + Key(key));
          return {};
        }
        out.push_back(static_cast<std::uint64_t>(v));
      }
    } else {
      Fail("GGUF array must contain integers: " + Key(key));
    }
    return out;
  }

  /// Float arrays of any width.
  std::vector<float> F32Array(std::string_view key, std::size_t expected) {
    const auto* meta = gguf.FindMetadata(Key(key));
    std::vector<float> out;
    if (meta == nullptr) {
      return out;  // Absent clamps are all-zero, which the fork also treats
                   // as "no clamp".
    }
    if (const auto* f = std::get_if<std::vector<double>>(&meta->value)) {
      out.reserve(f->size());
      for (auto v : *f) {
        out.push_back(static_cast<float>(v));
      }
    } else if (const auto* u =
                   std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      for (auto v : *u) {
        out.push_back(static_cast<float>(v));
      }
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (auto v : *s) {
        out.push_back(static_cast<float>(v));
      }
    } else {
      Fail("GGUF array must contain numbers: " + Key(key));
      return out;
    }
    if (out.size() != expected) {
      Fail(Key(key) + " must have " + std::to_string(expected) +
           " entries, found " + std::to_string(out.size()));
      return {};
    }
    for (auto v : out) {
      if (!std::isfinite(v) || v < 0.0F) {
        Fail(Key(key) + " entries must be finite and non-negative");
        return {};
      }
    }
    return out;
  }
};

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != kArchitecture) {
    if (error_msg != nullptr) {
      *error_msg = "general.architecture is not bailingmoe3";
    }
    return std::nullopt;
  }

  Config c;
  c.block_count = r.U32("block_count");
  c.nextn_layers = r.U32Opt("nextn_predict_layers", 0);
  c.hidden_size = r.U32("embedding_length");
  c.vocab_size = r.U32("vocab_size");
  c.context_length = r.U32("context_length");
  c.rms_eps = r.F32("attention.layer_norm_rms_epsilon");

  c.num_heads = r.U32("attention.head_count");
  c.kda_head_dim = r.U32("kda.head_dim");
  c.kda_conv_kernel = r.U32("ssm.conv_kernel");
  c.kda_gate_lower_bound = r.F32("kda.gate_lower_bound");

  c.mla_kv_lora_rank = r.U32("attention.kv_lora_rank");
  c.mla_head_dim_k = r.U32("attention.key_length_mla");
  c.mla_head_dim_v = r.U32("attention.value_length_mla");
  c.rotary_dim = r.U32("rope.dimension_count");
  c.rope_mode = r.U32Opt("rope.scaling.type", kRopeModeNorm);
  c.rope_theta = r.F32("rope.freq_base");

  c.ffn_length = r.U32("feed_forward_length");
  c.dense_lead_layers = r.U32Opt("leading_dense_block_count", 0);
  c.num_experts = r.U32("expert_count");
  c.num_experts_used = r.U32("expert_used_count");
  c.num_expert_groups = r.U32Opt("expert_group_count", 1);
  c.num_group_used = r.U32Opt("expert_group_used_count", c.num_experts_used);
  c.expert_ff = r.U32("expert_feed_forward_length");
  c.shared_expert_ff = r.U32Opt("expert_shared_feed_forward_length",
                                c.expert_ff * r.U32("expert_shared_count"));
  c.num_shared_experts = r.U32("expert_shared_count");
  c.expert_weights_scale = 1.0F;
  {
    const auto scale = r.MaybeF32("expert_weights_scale");
    if (scale.has_value() && std::isfinite(*scale)) {
      c.expert_weights_scale = *scale;
    }
  }
  c.expert_weights_norm = r.Bool("expert_weights_norm", false);
  c.expert_gating_func = r.U32("expert_gating_func");

  const auto kv = r.U64Array("attention.head_count_kv");
  c.head_count_kv.reserve(kv.size());
  for (auto v : kv) {
    c.head_count_kv.push_back(static_cast<std::uint32_t>(
        v > std::numeric_limits<std::uint32_t>::max()
            ? std::numeric_limits<std::uint32_t>::max()
            : v));
  }

  const auto clamp_exp = r.F32Array("swiglu_clamp_exp", c.block_count);
  const auto clamp_shexp = r.F32Array("swiglu_clamp_shexp", c.block_count);
  // A present-but-wrong-length array leaves `ok` false above. An absent array
  // is not an error: the fork defaults every layer's limit to zero, which
  // means "emit no clamp at all".
  if (r.ok) {
    c.swiglu_clamp_exp =
        clamp_exp.empty() ? std::vector<float>(c.block_count, 0.0F) : clamp_exp;
    c.swiglu_clamp_shexp = clamp_shexp.empty()
                               ? std::vector<float>(c.block_count, 0.0F)
                               : clamp_shexp;
  }

  if (!r.ok) {
    return std::nullopt;
  }

  if (c.nextn_layers >= c.block_count) {
    if (error_msg != nullptr) {
      *error_msg = "nextn_predict_layers must be less than block_count";
    }
    return std::nullopt;
  }
  c.num_layers = c.block_count - c.nextn_layers;

  if (c.head_count_kv.size() != c.block_count) {
    if (error_msg != nullptr) {
      *error_msg = "attention.head_count_kv length " +
                   std::to_string(c.head_count_kv.size()) +
                   " disagrees with block_count " +
                   std::to_string(c.block_count);
    }
    return std::nullopt;
  }
  if (c.kda_gate_lower_bound >= 0.0F) {
    if (error_msg != nullptr) {
      *error_msg = "kda.gate_lower_bound must be negative (the KDA safe gate)";
    }
    return std::nullopt;
  }
  if (c.rope_mode != kRopeModeNorm) {
    if (error_msg != nullptr) {
      *error_msg = "only unscaled NORM rope is supported, file asks for mode " +
                   std::to_string(c.rope_mode);
    }
    return std::nullopt;
  }
  if (c.rotary_dim == 0 || c.rotary_dim > c.mla_head_dim_k) {
    if (error_msg != nullptr) {
      *error_msg = "rope.dimension_count must be in (0, key_length_mla]";
    }
    return std::nullopt;
  }
  if (c.rotary_dim % 2 != 0) {
    if (error_msg != nullptr) {
      *error_msg = "rope.dimension_count must be even for NORM rope";
    }
    return std::nullopt;
  }
  if (c.expert_gating_func != kGatingSigmoid) {
    if (error_msg != nullptr) {
      *error_msg =
          "only the sigmoid expert gating function is implemented, "
          "file asks for " +
          std::to_string(c.expert_gating_func);
    }
    return std::nullopt;
  }
  if (c.num_expert_groups == 0 || c.num_experts % c.num_expert_groups != 0) {
    if (error_msg != nullptr) {
      *error_msg = "expert_count must divide evenly by expert_group_count";
    }
    return std::nullopt;
  }
  if (c.num_group_used > c.num_expert_groups ||
      c.num_experts_used > c.num_group_used * c.ExpertsPerGroup()) {
    if (error_msg != nullptr) {
      *error_msg =
          "group-limited routing cannot supply expert_used_count "
          "experts";
    }
    return std::nullopt;
  }
  if (c.num_experts_used == 0 || c.num_group_used == 0) {
    if (error_msg != nullptr) {
      *error_msg = "expert_used_count and expert_group_used_count must be > 0";
    }
    return std::nullopt;
  }
  if (c.kda_conv_kernel == 0) {
    if (error_msg != nullptr) {
      *error_msg = "ssm.conv_kernel must be > 0";
    }
    return std::nullopt;
  }
  if (c.hidden_size == 0 || c.vocab_size == 0 || c.rms_eps <= 0.0F ||
      c.rope_theta <= 0.0F) {
    if (error_msg != nullptr) {
      *error_msg =
          "embedding_length, vocab_size, rms epsilon and rope base "
          "must all be positive";
    }
    return std::nullopt;
  }

  return c;
}

}  // namespace superroc::models::ling3
