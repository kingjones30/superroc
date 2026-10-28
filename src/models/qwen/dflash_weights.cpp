#include "src/models/qwen/dflash_weights.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/env_compat.hpp"

namespace superroc::speculative {
namespace {

[[nodiscard]] bool DsparkEnvOn(const char* name) {
  const char* value = ::superroc::core::GetEnvCompat(name);
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

models::QwenTensorRef TensorRef(const core::GgufReader& reader,
                                std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr) {
    return {};
  }

  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor->data);
  std::size_t available_bytes = 0;
  for (const auto& region : reader.GetMappedRegions()) {
    const auto region_address = reinterpret_cast<std::uintptr_t>(region.data);
    if (tensor_address < region_address) {
      continue;
    }
    const auto offset = tensor_address - region_address;
    if (offset < region.size) {
      available_bytes = region.size - offset;
      break;
    }
  }
  return {.data = tensor->data,
          .type = tensor->type,
          .num_elements = tensor->ElementCount(),
          .available_bytes = available_bytes};
}

void SetError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
}

[[nodiscard]] bool IsSupportedMatrixType(core::GgmlType type) noexcept {
  return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
         type == core::GgmlType::kF16 || type == core::GgmlType::kQ8_0 ||
         type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ6_K;
}

[[nodiscard]] bool IsSupportedVectorType(core::GgmlType type) noexcept {
  // F16: the LFM2.5 DSpark sidecars store conf_proj.weight in F16; every
  // consumer reads tensors through the elementwise Get() conversion.
  return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
         type == core::GgmlType::kF16;
}

[[nodiscard]] bool BindRequiredTensor(const core::GgufReader& reader,
                                      std::string_view name,
                                      std::size_t expected_elements,
                                      bool matrix,
                                      models::QwenTensorRef& destination,
                                      std::string* error_msg) {
  destination = TensorRef(reader, name);
  if (destination.empty()) {
    SetError(error_msg, "DFlash-2 GGUF is missing required tensor '" +
                            std::string(name) + "'");
    return false;
  }
  if (destination.num_elements != expected_elements) {
    SetError(error_msg, "DFlash-2 tensor '" + std::string(name) + "' has " +
                            std::to_string(destination.num_elements) +
                            " elements; expected " +
                            std::to_string(expected_elements));
    return false;
  }
  const bool supported = matrix ? IsSupportedMatrixType(destination.type)
                                : IsSupportedVectorType(destination.type);
  if (!supported) {
    SetError(error_msg, "DFlash-2 tensor '" + std::string(name) +
                            "' has unsupported type " +
                            std::string(core::ToString(destination.type)));
    return false;
  }
  if (!destination.FitsAvailableStorage()) {
    SetError(error_msg, "DFlash-2 tensor '" + std::string(name) +
                            "' exceeds its mapped GGUF storage");
    return false;
  }
  return true;
}

[[nodiscard]] bool BindOptionalTensor(const core::GgufReader& reader,
                                      std::string_view name,
                                      std::size_t expected_elements,
                                      bool matrix,
                                      models::QwenTensorRef& destination,
                                      std::string* error_msg) {
  if (reader.FindTensor(name) == nullptr) {
    destination = {};
    return true;
  }
  return BindRequiredTensor(reader, name, expected_elements, matrix,
                            destination, error_msg);
}

[[nodiscard]] std::optional<std::uint32_t> RequiredUint32(
    const core::GgufReader& reader, std::string_view key,
    std::string* error_msg) {
  auto value = reader.GetMetadataUint32(key);
  if (!value.has_value()) {
    SetError(error_msg, "DFlash-2 GGUF is missing required metadata '" +
                            std::string(key) + "'");
  }
  return value;
}

[[nodiscard]] std::optional<std::vector<std::uint64_t>>
GetNonnegativeIntegerArray(const core::GgufReader& reader,
                           std::string_view key) {
  const auto* metadata = reader.FindMetadata(key);
  if (metadata == nullptr) {
    return std::nullopt;
  }
  if (const auto* values =
          std::get_if<std::vector<std::uint64_t>>(&metadata->value)) {
    return *values;
  }
  if (const auto* values =
          std::get_if<std::vector<std::int64_t>>(&metadata->value)) {
    std::vector<std::uint64_t> converted;
    converted.reserve(values->size());
    for (const std::int64_t value : *values) {
      if (value < 0) {
        return std::nullopt;
      }
      converted.push_back(static_cast<std::uint64_t>(value));
    }
    return converted;
  }
  return std::nullopt;
}

}  // namespace

std::optional<QwenDFlashWeights> QwenDFlashWeights::LoadFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto architecture = reader.GetMetadataString("general.architecture");
  if (!architecture.has_value() || *architecture != "dflash") {
    SetError(error_msg, "DFlash-2 GGUF requires general.architecture='dflash'");
    return std::nullopt;
  }

  core::ModelConfig config;
  config.architecture = "dflash";
  const std::string prefix = "dflash.";

  const auto hidden_size =
      RequiredUint32(reader, prefix + "embedding_length", error_msg);
  const auto intermediate_size =
      RequiredUint32(reader, prefix + "feed_forward_length", error_msg);
  const auto num_attention_heads =
      RequiredUint32(reader, prefix + "attention.head_count", error_msg);
  const auto num_key_value_heads =
      RequiredUint32(reader, prefix + "attention.head_count_kv", error_msg);
  const auto head_dim =
      RequiredUint32(reader, prefix + "attention.key_length", error_msg);
  auto rotary_dim = reader.GetMetadataUint32(prefix + "rope.dimension_count");
  if (!rotary_dim.has_value()) {
    const auto sections =
        GetNonnegativeIntegerArray(reader, prefix + "rope.dimension_sections");
    if (sections.has_value()) {
      const auto& values = *sections;
      if (values.size() == 4 && values[0] > 0 && values[1] == 0 &&
          values[2] == 0 && values[3] == 0 &&
          values[0] <= std::numeric_limits<std::uint32_t>::max() / 2U) {
        rotary_dim = static_cast<std::uint32_t>(values[0] * 2U);
      }
    }
  }
  // LFM2.5 DSpark sidecars carry neither key: their attention rotates the
  // full head, exactly as the LFM2 target config defaults (gguf_reader.cpp).
  if (!rotary_dim.has_value() && head_dim.has_value()) {
    rotary_dim = head_dim;
  }
  const auto context_length =
      RequiredUint32(reader, prefix + "context_length", error_msg);
  const auto num_layers =
      RequiredUint32(reader, prefix + "block_count", error_msg);
  const auto rope_theta = reader.GetMetadataFloat32(prefix + "rope.freq_base");
  if (!hidden_size || !intermediate_size || !num_attention_heads ||
      !num_key_value_heads || !head_dim || !rotary_dim || !context_length ||
      !num_layers || !rope_theta) {
    if (!rotary_dim) {
      SetError(error_msg,
               "DFlash-2 GGUF requires a valid rotary dimension or "
               "degenerate four-section M-RoPE layout");
    }
    if (!rope_theta) {
      SetError(
          error_msg,
          "DFlash-2 GGUF is missing required metadata 'dflash.rope.freq_base'");
    }
    return std::nullopt;
  }

  // NBFAST T1: carry the drafter's RMSNorm epsilon (Nanbeige: 1e-5; the
  // Qwen file's 1e-6 matches the executor's former hardcoded value, so
  // existing drafters are unchanged). Absent key keeps the 1e-6 default.
  if (const auto eps = reader.GetMetadataFloat32(
          prefix + "attention.layer_norm_rms_epsilon");
      eps.has_value() && *eps > 0.0F) {
    config.norm_eps_pre = *eps;
  }
  config.hidden_size = *hidden_size;
  config.intermediate_size = *intermediate_size;
  config.num_attention_heads = *num_attention_heads;
  config.num_key_value_heads = *num_key_value_heads;
  config.head_dim = *head_dim;
  config.rotary_dim = *rotary_dim;
  config.rope_theta = *rope_theta;
  config.context_length = *context_length;
  config.num_layers = *num_layers;
  config.full_attention_interval = 1;
  config.mtp_num_layers = 0;

  if (config.hidden_size == 0 || config.intermediate_size == 0 ||
      config.num_attention_heads == 0 || config.num_key_value_heads == 0 ||
      config.head_dim == 0 || config.num_layers == 0 ||
      config.num_attention_heads % config.num_key_value_heads != 0 ||
      config.rotary_dim == 0 || config.rotary_dim > config.head_dim ||
      (config.rotary_dim % 2U) != 0) {
    SetError(error_msg, "DFlash-2 GGUF has an invalid transformer topology");
    return std::nullopt;
  }

  QwenDFlashWeights weights;
  weights.config = config;
  auto& df_cfg = weights.dflash_config;

  const auto block_size =
      RequiredUint32(reader, prefix + "block_size", error_msg);
  const auto conv_kernel_size =
      reader.GetMetadataUint32(prefix + "conv_kernel_size");
  const auto conv_group_size =
      reader.GetMetadataUint32(prefix + "conv_group_size");
  const auto selector_rank = reader.GetMetadataUint32(prefix + "selector_rank");
  const auto selector_top_k =
      reader.GetMetadataUint32(prefix + "selector_top_k");
  const auto sliding_window =
      reader.GetMetadataUint32(prefix + "attention.sliding_window");
  const auto mask_token_id =
      RequiredUint32(reader, "tokenizer.ggml.mask_token_id", error_msg);
  const auto is_causal = reader.GetMetadataBool(prefix + "attention.causal");
  if (!block_size || !mask_token_id) {
    return std::nullopt;
  }

  const bool has_selector_tensor =
      reader.FindTensor("selector_predecessor.weight") != nullptr;
  const bool has_markov_tensor =
      reader.FindTensor("markov_w1.weight") != nullptr;
  df_cfg.has_selector = has_selector_tensor;
  df_cfg.has_dynamic_conv =
      reader.FindTensor("blk.0.attn_conv_base") != nullptr &&
      conv_kernel_size.has_value() && conv_group_size.has_value() &&
      *conv_kernel_size > 0 && *conv_group_size > 0;
  df_cfg.sample_from_anchor =
      reader.GetMetadataBool(prefix + "sample_from_anchor").value_or(false);
  df_cfg.has_confidence_head =
      reader.GetMetadataBool(prefix + "has_confidence_head")
          .value_or(reader.FindTensor("conf_proj.weight") != nullptr);

  df_cfg.block_size = *block_size;
  df_cfg.mask_token_id = *mask_token_id;
  df_cfg.num_layers = config.num_layers;
  df_cfg.conv_kernel_size = conv_kernel_size.value_or(0);
  df_cfg.conv_group_size = conv_group_size.value_or(0);
  df_cfg.selector_rank = selector_rank.value_or(0);
  df_cfg.selector_top_k = selector_top_k.value_or(0);
  df_cfg.sliding_window = sliding_window.value_or(0);

  if (df_cfg.has_selector) {
    if (!conv_kernel_size || !conv_group_size || !selector_rank ||
        !selector_top_k || !sliding_window || !is_causal.has_value()) {
      if (!is_causal.has_value()) {
        SetError(error_msg,
                 "DFlash-2 GGUF is missing required metadata "
                 "'dflash.attention.causal'");
      } else if (!conv_kernel_size) {
        SetError(error_msg,
                 "DFlash-2 GGUF is missing required metadata "
                 "'dflash.conv_kernel_size'");
      } else if (!selector_rank) {
        SetError(error_msg,
                 "DFlash-2 GGUF is missing required metadata "
                 "'dflash.selector_rank'");
      } else if (!sliding_window) {
        SetError(error_msg,
                 "DFlash-2 GGUF is missing required metadata "
                 "'dflash.attention.sliding_window'");
      }
      return std::nullopt;
    }
    if (df_cfg.block_size < 2 || df_cfg.conv_kernel_size == 0 ||
        df_cfg.conv_group_size == 0 ||
        config.hidden_size % df_cfg.conv_group_size != 0 ||
        df_cfg.selector_rank == 0 || df_cfg.selector_top_k == 0 ||
        // NBFAST T1: the Nanbeige DFlash2 drafter is trained with top-k 32;
        // the selector kernels size their lists by kMaxSelectorTopK (32).
        df_cfg.selector_top_k > 32U ||
        df_cfg.sliding_window < df_cfg.block_size || *is_causal) {
      SetError(error_msg, "DFlash-2 GGUF has an invalid diffusion topology");
      return std::nullopt;
    }
  } else {
    if (!has_markov_tensor) {
      SetError(error_msg,
               "DFlash GGUF requires selector_predecessor.weight or "
               "markov_w1.weight");
      return std::nullopt;
    }
    if (df_cfg.block_size < 2 || is_causal.value_or(false)) {
      SetError(error_msg, "DFlash-2 GGUF has an invalid diffusion topology");
      return std::nullopt;
    }
    if (df_cfg.sliding_window < df_cfg.block_size) {
      df_cfg.sliding_window = config.context_length;
    }
    if (const char* window =
            ::superroc::core::GetEnvCompat("SUPERROC_DSPARK_WINDOW")) {
      char* end = nullptr;
      const unsigned long parsed = std::strtoul(window, &end, 10);
      if (end != window && parsed >= df_cfg.block_size &&
          parsed <= std::numeric_limits<std::uint32_t>::max()) {
        df_cfg.sliding_window = static_cast<std::uint32_t>(parsed);
      }
    }
    if (df_cfg.selector_top_k == 0 || df_cfg.selector_top_k > 16U) {
      df_cfg.selector_top_k = 8;
    }
  }

  // GGUF target_layers are layer-input indices (llama.cpp converter writes
  // HF id + 1). SuperROC captures layer OUTPUT of the stored id. The selector
  // (27B) path always rewrites N → N−1. DSpark pdk-1 stored GGUF as-is;
  // SUPERROC_DSPARK_TAP_MINUS1=1 applies the same N−1 rewrite (HF
  // 3,11,19,27,35).
  const auto encoded_target_layers =
      GetNonnegativeIntegerArray(reader, prefix + "target_layers");
  if (!encoded_target_layers.has_value()) {
    SetError(error_msg,
             "DFlash-2 GGUF is missing required integer array "
             "'dflash.target_layers'");
    return std::nullopt;
  }
  if (encoded_target_layers->empty()) {
    SetError(error_msg, "DFlash-2 target layer list must not be empty");
    return std::nullopt;
  }
  df_cfg.target_layer_ids.reserve(encoded_target_layers->size());
  for (const std::uint64_t encoded_layer : *encoded_target_layers) {
    std::uint32_t normalized = 0;
    if (df_cfg.has_selector) {
      if (encoded_layer == 0 ||
          !std::in_range<std::uint32_t>(encoded_layer - 1U)) {
        SetError(error_msg,
                 "DFlash-2 target layer indices must use positive GGUF "
                 "layer-input numbering");
        return std::nullopt;
      }
      normalized = static_cast<std::uint32_t>(encoded_layer - 1U);
    } else if (DsparkEnvOn("SUPERROC_DSPARK_TAP_MINUS1")) {
      if (encoded_layer == 0 ||
          !std::in_range<std::uint32_t>(encoded_layer - 1U)) {
        SetError(error_msg,
                 "DSpark target layer indices must use positive GGUF "
                 "layer-input numbering");
        return std::nullopt;
      }
      normalized = static_cast<std::uint32_t>(encoded_layer - 1U);
    } else {
      if (!std::in_range<std::uint32_t>(encoded_layer)) {
        SetError(error_msg, "DFlash-2 target layer indices are out of range");
        return std::nullopt;
      }
      normalized = static_cast<std::uint32_t>(encoded_layer);
    }
    if (!df_cfg.target_layer_ids.empty() &&
        normalized <= df_cfg.target_layer_ids.back()) {
      SetError(error_msg,
               "DFlash-2 target layer indices must be strictly increasing");
      return std::nullopt;
    }
    df_cfg.target_layer_ids.push_back(normalized);
  }
  if (!df_cfg.has_selector) {
    std::fprintf(stderr, "dspark loader tap_minus1=%d window=%u taps=",
                 static_cast<int>(DsparkEnvOn("SUPERROC_DSPARK_TAP_MINUS1")),
                 df_cfg.sliding_window);
    for (std::size_t i = 0; i < df_cfg.target_layer_ids.size(); ++i) {
      std::fprintf(stderr, "%s%u", i == 0 ? "" : ",",
                   df_cfg.target_layer_ids[i]);
    }
    std::fprintf(stderr, " sample_from_anchor=%d\n",
                 static_cast<int>(df_cfg.sample_from_anchor));
  }

  const std::size_t hidden = config.hidden_size;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t groups =
      df_cfg.has_dynamic_conv ? hidden / df_cfg.conv_group_size : 0;
  const std::size_t dynamic_coefficients =
      std::size_t{2} * df_cfg.conv_kernel_size * groups;

  // DFlash-2 uses the target model's embedding and output head. Private copies
  // are accepted but not required.
  if (reader.FindTensor("d2t") != nullptr) {
    SetError(error_msg,
             "DFlash-2 reduced-vocabulary d2t exports are not supported yet");
    return std::nullopt;
  }

  if (df_cfg.has_selector) {
    const auto* predecessor_tensor =
        reader.FindTensor("selector_predecessor.weight");
    if (predecessor_tensor == nullptr || predecessor_tensor->data == nullptr ||
        predecessor_tensor->ElementCount() % df_cfg.selector_rank != 0) {
      SetError(error_msg,
               "DFlash-2 selector_predecessor.weight has invalid dimensions");
      return std::nullopt;
    }
    weights.selector_predecessor =
        TensorRef(reader, "selector_predecessor.weight");
    if (!IsSupportedMatrixType(weights.selector_predecessor.type) ||
        !weights.selector_predecessor.FitsAvailableStorage()) {
      SetError(error_msg,
               "DFlash-2 selector_predecessor.weight has unsupported storage");
      return std::nullopt;
    }
    const std::size_t inferred_vocab =
        weights.selector_predecessor.num_elements / df_cfg.selector_rank;
    if (inferred_vocab == 0 || !std::in_range<std::uint32_t>(inferred_vocab)) {
      SetError(error_msg, "DFlash-2 selector vocabulary size is invalid");
      return std::nullopt;
    }
    config.vocab_size = static_cast<std::uint32_t>(inferred_vocab);
    weights.config.vocab_size = config.vocab_size;

    if (!BindRequiredTensor(reader, "selector_successor.weight",
                            inferred_vocab * df_cfg.selector_rank, true,
                            weights.selector_successor, error_msg) ||
        !BindRequiredTensor(reader, "selector_hidden.weight",
                            hidden * df_cfg.selector_rank, true,
                            weights.selector_hidden, error_msg)) {
      return std::nullopt;
    }
  } else {
    const auto* markov_tensor = reader.FindTensor("markov_w1.weight");
    if (markov_tensor == nullptr || markov_tensor->data == nullptr ||
        markov_tensor->dimensions.size() != 2 ||
        markov_tensor->dimensions[0] == 0) {
      SetError(error_msg, "DSpark markov_w1.weight has invalid dimensions");
      return std::nullopt;
    }
    df_cfg.markov_rank =
        static_cast<std::uint32_t>(markov_tensor->dimensions[0]);
    if (markov_tensor->ElementCount() % df_cfg.markov_rank != 0) {
      SetError(error_msg, "DSpark markov_w1.weight has invalid dimensions");
      return std::nullopt;
    }
    const std::size_t inferred_vocab =
        markov_tensor->ElementCount() / df_cfg.markov_rank;
    if (inferred_vocab == 0 || !std::in_range<std::uint32_t>(inferred_vocab) ||
        df_cfg.markov_rank == 0) {
      SetError(error_msg, "DSpark markov vocabulary size is invalid");
      return std::nullopt;
    }
    config.vocab_size = static_cast<std::uint32_t>(inferred_vocab);
    weights.config.vocab_size = config.vocab_size;
    if (!BindRequiredTensor(reader, "markov_w1.weight",
                            inferred_vocab * df_cfg.markov_rank, true,
                            weights.markov_w1, error_msg) ||
        !BindRequiredTensor(reader, "markov_w2.weight",
                            inferred_vocab * df_cfg.markov_rank, true,
                            weights.markov_w2, error_msg)) {
      return std::nullopt;
    }
  }

  const std::size_t encoder_input = df_cfg.target_layer_ids.size() * hidden;
  if (!BindRequiredTensor(reader, "fc.weight", hidden * encoder_input, true,
                          weights.fc_projection, error_msg) ||
      !BindRequiredTensor(reader, "enc.output_norm.weight", hidden, false,
                          weights.fc_norm, error_msg) ||
      !BindRequiredTensor(reader, "output_norm.weight", hidden, false,
                          weights.output_norm, error_msg)) {
    return std::nullopt;
  }
  const auto fc_scale = TensorRef(reader, "fc.scale");
  if (!fc_scale.empty() &&
      (fc_scale.num_elements != 1 || !IsSupportedVectorType(fc_scale.type) ||
       !fc_scale.FitsAvailableStorage() || fc_scale.Get(0) != 1.0F)) {
    SetError(error_msg,
             "DFlash-2 fc.scale must be absent or a unit F32/BF16 scalar");
    return std::nullopt;
  }

  if (df_cfg.has_confidence_head) {
    const std::size_t conf_width = hidden + df_cfg.markov_rank;
    if (conf_width == hidden) {
      SetError(error_msg,
               "DSpark confidence head requires a markov rank from "
               "markov_w1.weight");
      return std::nullopt;
    }
    if (!BindRequiredTensor(reader, "conf_proj.weight", conf_width, true,
                            weights.conf_proj, error_msg) ||
        !BindRequiredTensor(reader, "conf_proj.bias", 1, false,
                            weights.conf_proj_bias, error_msg)) {
      return std::nullopt;
    }
  } else if (!BindOptionalTensor(reader, "conf_proj.weight",
                                 hidden + df_cfg.markov_rank, true,
                                 weights.conf_proj, error_msg) ||
             !BindOptionalTensor(reader, "conf_proj.bias", 1, false,
                                 weights.conf_proj_bias, error_msg)) {
    return std::nullopt;
  }

  weights.layers.resize(df_cfg.num_layers);
  for (std::uint32_t i = 0; i < df_cfg.num_layers; ++i) {
    const std::string layer_prefix = "blk." + std::to_string(i) + ".";
    auto& layer = weights.layers[i];
    auto& transformer = layer.transformer;
    transformer.is_full_attention = true;

    if (!BindRequiredTensor(reader, layer_prefix + "attn_norm.weight", hidden,
                            false, transformer.attn_norm, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_q.weight",
                            attention * hidden, true, transformer.attn_q,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_k.weight", kv * hidden,
                            true, transformer.attn_k, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_v.weight", kv * hidden,
                            true, transformer.attn_v, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_output.weight",
                            hidden * attention, true, transformer.attn_output,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_q_norm.weight",
                            config.head_dim, false, transformer.attn_q_norm,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_k_norm.weight",
                            config.head_dim, false, transformer.attn_k_norm,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_norm.weight", hidden,
                            false, transformer.ffn_norm, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_gate.weight",
                            intermediate * hidden, true, transformer.ffn_gate,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_up.weight",
                            intermediate * hidden, true, transformer.ffn_up,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_down.weight",
                            hidden * intermediate, true, transformer.ffn_down,
                            error_msg)) {
      return std::nullopt;
    }
    if (df_cfg.has_dynamic_conv) {
      if (!BindRequiredTensor(reader, layer_prefix + "attn_conv_base",
                              hidden * df_cfg.conv_kernel_size * 2U, false,
                              layer.attention_conv_base, error_msg) ||
          !BindRequiredTensor(reader, layer_prefix + "attn_conv_proj.weight",
                              dynamic_coefficients * hidden, true,
                              layer.attention_conv_projection, error_msg) ||
          !BindRequiredTensor(reader, layer_prefix + "ffn_conv_base",
                              hidden * df_cfg.conv_kernel_size * 2U, false,
                              layer.ffn_conv_base, error_msg) ||
          !BindRequiredTensor(reader, layer_prefix + "ffn_conv_proj.weight",
                              dynamic_coefficients * hidden, true,
                              layer.ffn_conv_projection, error_msg)) {
        return std::nullopt;
      }
    }
  }

  return weights;
}

}  // namespace superroc::speculative
