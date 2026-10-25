#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "src/models/qwen/state.hpp"

namespace gufo::models {
namespace {

QwenTensorRef ExtractTensorRef(const core::GgufReader& reader,
                               std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr) {
    return {};
  }

  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor->data);
  std::size_t available_bytes = 0;
  for (const auto& region : reader.GetMappedRegions()) {
    const auto region_address = reinterpret_cast<std::uintptr_t>(region.data);
    if (tensor_address >= region_address) {
      const auto offset = tensor_address - region_address;
      if (offset < region.size) {
        available_bytes = region.size - offset;
        break;
      }
    }
  }
  return {.data = tensor->data,
          .type = tensor->type,
          .num_elements = tensor->ElementCount(),
          .available_bytes = available_bytes};
}

enum class TensorRole : std::uint8_t {
  kEmbedding,
  kNorm,
  kProjection,
  kSsmParameter,
};

[[nodiscard]] constexpr std::string_view TensorRoleName(
    TensorRole role) noexcept {
  switch (role) {
    case TensorRole::kEmbedding:
      return "embedding";
    case TensorRole::kNorm:
      return "normalization";
    case TensorRole::kProjection:
      return "projection";
    case TensorRole::kSsmParameter:
      return "SSM parameter";
  }
  return "unknown";
}

[[nodiscard]] constexpr bool SupportsTensorType(core::GgmlType type,
                                                TensorRole role) noexcept {
  switch (role) {
    case TensorRole::kEmbedding:
      // The production HIP embedding kernel has exact F32, BF16, and Q8_0
      // implementations. Treating every other type as Q8_0 would decode the
      // wrong block layout.
      // opt-q4kxl: token_embd.weight is Q4_K in the UD-Q4_K_XL shard; the
      // lookup kernels decode any DecodeQuantSub16-backed format directly.
      return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
             type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ4_K ||
             type == core::GgmlType::kQ3_K || type == core::GgmlType::kQ5_K ||
             type == core::GgmlType::kQ6_K || type == core::GgmlType::kIQ4_NL ||
             type == core::GgmlType::kIQ4_XS ||
             type == core::GgmlType::kIQ3_S ||
             type == core::GgmlType::kQ4_0_ROCMFP4 ||  // lqw:weights-emb
             type == core::GgmlType::kQ4_0_ROCMFP4_FAST;
    case TensorRole::kNorm:
    case TensorRole::kSsmParameter:
      // HIP norm, convolution, and recurrence kernels consume these tensors as
      // float pointers. Accepting BF16 or packed data would reinterpret bytes.
      return type == core::GgmlType::kF32;
    case TensorRole::kProjection:
      // opt-q4kxl: the Unsloth UD-Q4_K_XL shard mixes Q5_K, IQ4_XS, Q4_K, Q6_K,
      // IQ4_NL, Q3_K and IQ3_S across projections, and every one of them has an
      // in-kernel decoder, so all are accepted here.
      return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
             type == core::GgmlType::kQ8_K || type == core::GgmlType::kQ8_0 ||
             type == core::GgmlType::kQ5_K || type == core::GgmlType::kQ6_K ||
             type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ3_K ||
             type == core::GgmlType::kIQ4_NL ||
             type == core::GgmlType::kIQ4_XS ||
             type == core::GgmlType::kIQ3_S ||
             type == core::GgmlType::kQ4_0_ROCMFP4 ||  // lqw:weights-proj
             type == core::GgmlType::kQ4_0_ROCMFP4_FAST;
  }
  return false;
}

bool ValidateTensor(const QwenTensorRef& tensor, std::size_t expected_elements,
                    TensorRole role, std::string_view name,
                    std::string* error_msg, std::size_t row_elements = 0) {
  if (tensor.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "Missing required Qwen tensor: " + std::string(name);
    }
    return false;
  }
  if (!SupportsTensorType(tensor.type, role)) {
    if (error_msg != nullptr) {
      *error_msg = "Unsupported " + std::string(TensorRoleName(role)) +
                   " tensor type for " + std::string(name) + ": " +
                   std::string(core::ToString(tensor.type));
    }
    return false;
  }
  if (tensor.num_elements != expected_elements) {
    if (error_msg != nullptr) {
      *error_msg = "Tensor shape mismatch for " + std::string(name) +
                   ": expected " + std::to_string(expected_elements) +
                   " elements, found " + std::to_string(tensor.num_elements);
    }
    return false;
  }
  if (!tensor.FitsAvailableStorage()) {
    if (error_msg != nullptr) {
      *error_msg = "Tensor payload is truncated or has invalid encoded size: " +
                   std::string(name);
    }
    return false;
  }
  const bool quantized = tensor.type != core::GgmlType::kF32 &&
                         tensor.type != core::GgmlType::kF16 &&
                         tensor.type != core::GgmlType::kBF16;
  if (quantized && row_elements != 0 &&
      gufo::quant::QuantizedRowBytes(tensor.type, row_elements) == 0) {
    if (error_msg != nullptr) {
      *error_msg =
          "Quantized tensor row is not block aligned: " + std::string(name) +
          " (row elements " + std::to_string(row_elements) + ")";
    }
    return false;
  }
  return true;
}

// GLIMMER: attention-only binder. Every layer has 1× attn_q, a separate
// attn_gate, and real post_attention_norm / post_ffw_norm tensors. The qwen
// path aliases post_attention_norm onto ffn_norm when ffn_norm is missing;
// muse has both, so that alias must not run.
std::optional<QwenModelWeights> BindGlimmerWeights(
    const core::GgufReader& reader, const core::ModelConfig& config,
    std::string* error_msg) {
  QwenModelWeights weights;
  weights.config = config;

  weights.token_embd = ExtractTensorRef(reader, "token_embd.weight");
  weights.output_norm = ExtractTensorRef(reader, "output_norm.weight");
  weights.output = ExtractTensorRef(reader, "output.weight");
  if (weights.output.empty()) {
    if (error_msg != nullptr) {
      *error_msg =
          "muse-glimmer requires output.weight (no tied embedding fallback)";
    }
    return std::nullopt;
  }

  const std::size_t hidden_size = config.hidden_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t intermediate_size = config.intermediate_size;
  if (!ValidateTensor(weights.token_embd, vocab_size * hidden_size,
                      TensorRole::kEmbedding, "token_embd.weight", error_msg,
                      hidden_size) ||
      !ValidateTensor(weights.output_norm, hidden_size, TensorRole::kNorm,
                      "output_norm.weight", error_msg) ||
      !ValidateTensor(weights.output, vocab_size * hidden_size,
                      TensorRole::kProjection, "output.weight", error_msg,
                      hidden_size)) {
    return std::nullopt;
  }

  weights.layers.resize(config.num_layers);
  for (std::uint32_t i = 0; i < config.num_layers; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    auto& layer = weights.layers[i];
    layer.is_full_attention = true;
    layer.attn_norm = ExtractTensorRef(reader, prefix + "attn_norm.weight");
    layer.ffn_norm = ExtractTensorRef(reader, prefix + "ffn_norm.weight");
    layer.attn_post_norm =
        ExtractTensorRef(reader, prefix + "post_attention_norm.weight");
    layer.ffn_post_norm =
        ExtractTensorRef(reader, prefix + "post_ffw_norm.weight");
    layer.attn_q = ExtractTensorRef(reader, prefix + "attn_q.weight");
    layer.attn_k = ExtractTensorRef(reader, prefix + "attn_k.weight");
    layer.attn_v = ExtractTensorRef(reader, prefix + "attn_v.weight");
    layer.attn_output = ExtractTensorRef(reader, prefix + "attn_output.weight");
    layer.attn_q_norm = ExtractTensorRef(reader, prefix + "attn_q_norm.weight");
    layer.attn_k_norm = ExtractTensorRef(reader, prefix + "attn_k_norm.weight");
    layer.attn_gate = ExtractTensorRef(reader, prefix + "attn_gate.weight");
    layer.ffn_gate = ExtractTensorRef(reader, prefix + "ffn_gate.weight");
    layer.ffn_up = ExtractTensorRef(reader, prefix + "ffn_up.weight");
    layer.ffn_down = ExtractTensorRef(reader, prefix + "ffn_down.weight");

    if (!layer.attn_post_norm.empty() && !layer.ffn_norm.empty() &&
        layer.attn_post_norm.data == layer.ffn_norm.data) {
      if (error_msg != nullptr) {
        *error_msg = "post_attention_norm aliased onto ffn_norm at " + prefix;
      }
      return std::nullopt;
    }
    if (!layer.ffn_post_norm.empty() && !layer.ffn_norm.empty() &&
        layer.ffn_post_norm.data == layer.ffn_norm.data) {
      if (error_msg != nullptr) {
        *error_msg = "post_ffw_norm aliased onto ffn_norm at " + prefix;
      }
      return std::nullopt;
    }

    if (!ValidateTensor(layer.attn_norm, hidden_size, TensorRole::kNorm,
                        prefix + "attn_norm.weight", error_msg) ||
        !ValidateTensor(layer.ffn_norm, hidden_size, TensorRole::kNorm,
                        prefix + "ffn_norm.weight", error_msg) ||
        !ValidateTensor(layer.attn_post_norm, hidden_size, TensorRole::kNorm,
                        prefix + "post_attention_norm.weight", error_msg) ||
        !ValidateTensor(layer.ffn_post_norm, hidden_size, TensorRole::kNorm,
                        prefix + "post_ffw_norm.weight", error_msg) ||
        !ValidateTensor(layer.attn_q, attention_size * hidden_size,
                        TensorRole::kProjection, prefix + "attn_q.weight",
                        error_msg, hidden_size) ||
        !ValidateTensor(layer.attn_k, kv_size * hidden_size,
                        TensorRole::kProjection, prefix + "attn_k.weight",
                        error_msg, hidden_size) ||
        !ValidateTensor(layer.attn_v, kv_size * hidden_size,
                        TensorRole::kProjection, prefix + "attn_v.weight",
                        error_msg, hidden_size) ||
        !ValidateTensor(layer.attn_output, hidden_size * attention_size,
                        TensorRole::kProjection, prefix + "attn_output.weight",
                        error_msg, attention_size) ||
        !ValidateTensor(layer.attn_q_norm, config.head_dim, TensorRole::kNorm,
                        prefix + "attn_q_norm.weight", error_msg) ||
        !ValidateTensor(layer.attn_k_norm, config.head_dim, TensorRole::kNorm,
                        prefix + "attn_k_norm.weight", error_msg) ||
        !ValidateTensor(layer.attn_gate, attention_size * hidden_size,
                        TensorRole::kProjection, prefix + "attn_gate.weight",
                        error_msg, hidden_size) ||
        !ValidateTensor(layer.ffn_gate, intermediate_size * hidden_size,
                        TensorRole::kProjection, prefix + "ffn_gate.weight",
                        error_msg, hidden_size) ||
        !ValidateTensor(layer.ffn_up, intermediate_size * hidden_size,
                        TensorRole::kProjection, prefix + "ffn_up.weight",
                        error_msg, hidden_size) ||
        !ValidateTensor(layer.ffn_down, hidden_size * intermediate_size,
                        TensorRole::kProjection, prefix + "ffn_down.weight",
                        error_msg, intermediate_size)) {
      return std::nullopt;
    }
  }
  return weights;
}

// LFM2: lfm2moe binder. Short-conv layers carry the conv operator, attention
// layers carry GQA projections, lead layers are dense SwiGLU and the rest are
// sigmoid-routed ROCmFP4 experts. Mirrors the ExtractTensorRef/ValidateTensor
// discipline of BindGlimmerWeights.
std::optional<QwenModelWeights> BindLfm2Weights(  // LFM2
    const core::GgufReader& reader, const core::ModelConfig& config,
    std::string* error_msg) {
  auto fail = [&](const std::string& msg)  // LFM2
      -> std::optional<QwenModelWeights> {
    if (error_msg != nullptr) {
      *error_msg = msg;
    }
    return std::nullopt;
  };
  auto mul = [&](std::size_t a, std::size_t b)  // LFM2
      -> std::optional<std::size_t> {
    if (a != 0 && b > SIZE_MAX / a) {
      if (error_msg != nullptr) {
        *error_msg = "lfm2moe tensor size overflows";
      }
      return std::nullopt;
    }
    return a * b;
  };
  auto expect_dims = [&](const std::string& name,
                         const std::uint64_t* dims,  // LFM2
                         std::size_t n_dims) -> bool {
    const auto* info = reader.FindTensor(name);
    if (info == nullptr || info->dimensions.size() != n_dims) {
      if (error_msg != nullptr) {
        *error_msg = "lfm2moe tensor rank mismatch for " + name;
      }
      return false;
    }
    for (std::size_t axis = 0; axis < n_dims; ++axis) {
      if (info->dimensions[axis] != dims[axis]) {
        if (error_msg != nullptr) {
          *error_msg = "lfm2moe tensor dimension mismatch for " + name +
                       ": axis " + std::to_string(axis) + " expected " +
                       std::to_string(dims[axis]) + " found " +
                       std::to_string(info->dimensions[axis]);
        }
        return false;
      }
    }
    return true;
  };
  auto require_fp4 = [&](const QwenTensorRef& ref,  // LFM2
                         const std::string& name) -> bool {
    if (!ref.empty() && ref.type != core::GgmlType::kQ4_0_ROCMFP4 &&
        ref.type != core::GgmlType::kQ4_0_ROCMFP4_FAST) {
      if (error_msg != nullptr) {
        *error_msg = "lfm2moe experts must be ROCmFP4 (Q4_0_ROCMFP4/_FAST); " +
                     name + " is " + std::string(core::ToString(ref.type)) +
                     ": build the file with gufo-quant LEAN";
      }
      return false;
    }
    return true;
  };

  QwenModelWeights weights;  // LFM2
  weights.config = config;
  const std::size_t hidden = config.hidden_size;    // LFM2
  const std::size_t vocab = config.vocab_size;      // LFM2
  const std::size_t attn = config.AttentionSize();  // LFM2
  const std::size_t kv =                            // LFM2
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t inter = config.intermediate_size;          // LFM2
  const std::size_t n_expert = config.expert_count;            // LFM2
  const std::size_t n_ff = config.expert_feed_forward_length;  // LFM2
  const std::size_t l_cache = config.shortconv_l_cache;        // LFM2
  const auto hidden_vocab = mul(hidden, vocab);                // LFM2
  if (!hidden_vocab.has_value()) {
    return std::nullopt;
  }

  weights.token_embd =  // LFM2
      ExtractTensorRef(reader, "token_embd.weight");
  if (reader.FindTensor("output_norm.weight") != nullptr) {  // LFM2
    return fail("lfm2moe final norm must be token_embd_norm.weight");
  }
  weights.output_norm =  // LFM2
      ExtractTensorRef(reader, "token_embd_norm.weight");
  weights.output = ExtractTensorRef(reader, "output.weight");  // LFM2
  if (weights.output.empty()) {                                // LFM2
    weights.output = weights.token_embd;
    weights.output_tied_to_token_embd = true;
  }
  if (!ValidateTensor(weights.token_embd, *hidden_vocab,  // LFM2
                      TensorRole::kEmbedding, "token_embd.weight", error_msg,
                      hidden) ||
      !ValidateTensor(weights.output_norm, hidden, TensorRole::kNorm,  // LFM2
                      "token_embd_norm.weight", error_msg) ||
      !ValidateTensor(weights.output, *hidden_vocab,
                      TensorRole::kProjection,  // LFM2
                      "output.weight", error_msg, hidden)) {
    return std::nullopt;
  }

  weights.layers.resize(
      config.num_layers);  // LFM2 (audit F09: resize only after the top-level
                           // tensors validated)
  for (std::uint32_t i = 0; i < config.num_layers; ++i) {  // LFM2
    const std::string prefix = "blk." + std::to_string(i) + ".";
    auto& layer = weights.layers[i];
    layer.attn_norm = ExtractTensorRef(reader, prefix + "attn_norm.weight");
    layer.ffn_norm = ExtractTensorRef(reader, prefix + "ffn_norm.weight");
    if (!ValidateTensor(layer.attn_norm, hidden, TensorRole::kNorm,
                        prefix + "attn_norm.weight", error_msg) ||
        !ValidateTensor(layer.ffn_norm, hidden, TensorRole::kNorm,
                        prefix + "ffn_norm.weight", error_msg)) {
      return std::nullopt;
    }
    if (config.LayerIsLfm2Attention(i)) {  // LFM2
      layer.is_full_attention = true;
      layer.attn_q = ExtractTensorRef(reader, prefix + "attn_q.weight");
      layer.attn_k = ExtractTensorRef(reader, prefix + "attn_k.weight");
      layer.attn_v = ExtractTensorRef(reader, prefix + "attn_v.weight");
      layer.attn_output =
          ExtractTensorRef(reader, prefix + "attn_output.weight");
      layer.attn_q_norm =
          ExtractTensorRef(reader, prefix + "attn_q_norm.weight");
      layer.attn_k_norm =
          ExtractTensorRef(reader, prefix + "attn_k_norm.weight");
      const auto attn_hidden = mul(attn, hidden);
      const auto kv_hidden = mul(kv, hidden);
      const auto hidden_attn = mul(hidden, attn);
      if (!attn_hidden.has_value() || !kv_hidden.has_value() ||
          !hidden_attn.has_value()) {
        return std::nullopt;
      }
      if (!ValidateTensor(layer.attn_q, *attn_hidden, TensorRole::kProjection,
                          prefix + "attn_q.weight", error_msg, hidden) ||
          !ValidateTensor(layer.attn_k, *kv_hidden, TensorRole::kProjection,
                          prefix + "attn_k.weight", error_msg, hidden) ||
          !ValidateTensor(layer.attn_v, *kv_hidden, TensorRole::kProjection,
                          prefix + "attn_v.weight", error_msg, hidden) ||
          !ValidateTensor(layer.attn_output, *hidden_attn,
                          TensorRole::kProjection,
                          prefix + "attn_output.weight", error_msg, attn) ||
          !ValidateTensor(layer.attn_q_norm, config.head_dim, TensorRole::kNorm,
                          prefix + "attn_q_norm.weight", error_msg) ||
          !ValidateTensor(layer.attn_k_norm, config.head_dim, TensorRole::kNorm,
                          prefix + "attn_k_norm.weight", error_msg)) {
        return std::nullopt;
      }
      if (reader.FindTensor(prefix + "shortconv.in_proj.weight") !=
          nullptr) {  // LFM2
        return fail("lfm2moe attention layer " + std::to_string(i) +
                    " has shortconv tensors");
      }
    } else {
      layer.is_shortconv = true;  // LFM2
      layer.shortconv_conv =
          ExtractTensorRef(reader, prefix + "shortconv.conv.weight");
      layer.shortconv_in_proj =
          ExtractTensorRef(reader, prefix + "shortconv.in_proj.weight");
      layer.shortconv_out_proj =
          ExtractTensorRef(reader, prefix + "shortconv.out_proj.weight");
      const auto three_hidden = mul(3, hidden);
      const auto conv_elems = mul(l_cache, hidden);
      if (!three_hidden.has_value() || !conv_elems.has_value()) {
        return std::nullopt;
      }
      const auto in_elems = mul(*three_hidden, hidden);
      const auto out_elems = mul(hidden, hidden);
      if (!in_elems.has_value() || !out_elems.has_value()) {
        return std::nullopt;
      }
      const std::uint64_t conv_dims[] = {l_cache, hidden};
      if (!ValidateTensor(layer.shortconv_conv, *conv_elems,
                          TensorRole::kSsmParameter,
                          prefix + "shortconv.conv.weight", error_msg) ||
          !expect_dims(prefix + "shortconv.conv.weight", conv_dims, 2) ||
          !ValidateTensor(
              layer.shortconv_in_proj, *in_elems, TensorRole::kProjection,
              prefix + "shortconv.in_proj.weight", error_msg, hidden) ||
          !ValidateTensor(
              layer.shortconv_out_proj, *out_elems, TensorRole::kProjection,
              prefix + "shortconv.out_proj.weight", error_msg, hidden)) {
        return std::nullopt;
      }
      if (reader.FindTensor(prefix + "attn_q.weight") != nullptr) {  // LFM2
        return fail("lfm2moe conv layer " + std::to_string(i) +
                    " has attention tensors");
      }
    }
    if (i < config.dense_lead_layers) {  // LFM2
      layer.ffn_gate = ExtractTensorRef(reader, prefix + "ffn_gate.weight");
      layer.ffn_up = ExtractTensorRef(reader, prefix + "ffn_up.weight");
      layer.ffn_down = ExtractTensorRef(reader, prefix + "ffn_down.weight");
      const auto gate_elems = mul(inter, hidden);
      const auto down_elems = mul(hidden, inter);
      if (!gate_elems.has_value() || !down_elems.has_value()) {
        return std::nullopt;
      }
      if (!ValidateTensor(layer.ffn_gate, *gate_elems, TensorRole::kProjection,
                          prefix + "ffn_gate.weight", error_msg, hidden) ||
          !ValidateTensor(layer.ffn_up, *gate_elems, TensorRole::kProjection,
                          prefix + "ffn_up.weight", error_msg, hidden) ||
          !ValidateTensor(layer.ffn_down, *down_elems, TensorRole::kProjection,
                          prefix + "ffn_down.weight", error_msg, inter)) {
        return std::nullopt;
      }
      if (reader.FindTensor(prefix + "ffn_gate_inp.weight") !=
          nullptr) {  // LFM2
        return fail("lfm2moe dense layer " + std::to_string(i) +
                    " has a router");
      }
    } else {
      layer.is_moe = true;  // LFM2
      layer.ffn_gate_inp =
          ExtractTensorRef(reader, prefix + "ffn_gate_inp.weight");
      layer.exp_probs_b = ExtractTensorRef(reader, prefix + "exp_probs_b.bias");
      layer.ffn_gate_exps =
          ExtractTensorRef(reader, prefix + "ffn_gate_exps.weight");
      layer.ffn_up_exps =
          ExtractTensorRef(reader, prefix + "ffn_up_exps.weight");
      layer.ffn_down_exps =
          ExtractTensorRef(reader, prefix + "ffn_down_exps.weight");
      const auto router_elems = mul(hidden, n_expert);
      const auto hidden_ff = mul(hidden, n_ff);
      const auto ff_hidden = mul(n_ff, hidden);
      if (!router_elems.has_value() || !hidden_ff.has_value() ||
          !ff_hidden.has_value()) {
        return std::nullopt;
      }
      const auto exp_gate = mul(*hidden_ff, n_expert);
      const auto exp_down = mul(*ff_hidden, n_expert);
      if (!exp_gate.has_value() || !exp_down.has_value()) {
        return std::nullopt;
      }
      if (!ValidateTensor(layer.ffn_gate_inp, *router_elems,
                          TensorRole::kSsmParameter,
                          prefix + "ffn_gate_inp.weight", error_msg) ||
          !ValidateTensor(layer.exp_probs_b, n_expert,
                          TensorRole::kSsmParameter,
                          prefix + "exp_probs_b.bias", error_msg) ||
          !ValidateTensor(layer.ffn_gate_exps, *exp_gate,
                          TensorRole::kProjection,
                          prefix + "ffn_gate_exps.weight", error_msg, hidden) ||
          !ValidateTensor(layer.ffn_up_exps, *exp_gate, TensorRole::kProjection,
                          prefix + "ffn_up_exps.weight", error_msg, hidden) ||
          !ValidateTensor(layer.ffn_down_exps, *exp_down,
                          TensorRole::kProjection,
                          prefix + "ffn_down_exps.weight", error_msg, n_ff)) {
        return std::nullopt;
      }
      const std::uint64_t router_dims[] = {hidden, n_expert};
      const std::uint64_t gate_dims[] = {hidden, n_ff, n_expert};
      const std::uint64_t down_dims[] = {n_ff, hidden, n_expert};
      if (!expect_dims(prefix + "ffn_gate_inp.weight", router_dims, 2) ||
          !expect_dims(prefix + "ffn_gate_exps.weight", gate_dims, 3) ||
          !expect_dims(prefix + "ffn_up_exps.weight", gate_dims, 3) ||
          !expect_dims(prefix + "ffn_down_exps.weight", down_dims, 3)) {
        return std::nullopt;
      }
      if (!require_fp4(layer.ffn_gate_exps,  // LFM2
                       prefix + "ffn_gate_exps.weight") ||
          !require_fp4(layer.ffn_up_exps, prefix + "ffn_up_exps.weight") ||
          !require_fp4(layer.ffn_down_exps, prefix + "ffn_down_exps.weight")) {
        return std::nullopt;
      }
      const std::string shexp = prefix + "ffn_gate_shexp.weight";  // LFM2
      if (reader.FindTensor(shexp) != nullptr) {
        return fail("lfm2moe has no shared expert; " + shexp + " found");
      }
    }
  }

  struct NamedPtr {  // LFM2
    const void* ptr = nullptr;
    std::string name;
  };
  std::vector<NamedPtr> allocs;  // LFM2 (audit F11: alias check, LFM2 only)
  auto track = [&](const QwenTensorRef& ref, const std::string& name) {
    if (!ref.empty()) {
      allocs.push_back({ref.data, name});
    }
  };
  track(weights.token_embd, "token_embd.weight");        // LFM2
  track(weights.output_norm, "token_embd_norm.weight");  // LFM2
  if (!weights.output_tied_to_token_embd) {              // LFM2
    track(weights.output, "output.weight");
  }
  for (std::uint32_t i = 0; i < config.num_layers; ++i) {  // LFM2
    const std::string prefix = "blk." + std::to_string(i) + ".";
    const auto& layer = weights.layers[i];
    track(layer.attn_norm, prefix + "attn_norm.weight");
    track(layer.ffn_norm, prefix + "ffn_norm.weight");
    track(layer.attn_q, prefix + "attn_q.weight");
    track(layer.attn_k, prefix + "attn_k.weight");
    track(layer.attn_v, prefix + "attn_v.weight");
    track(layer.attn_output, prefix + "attn_output.weight");
    track(layer.attn_q_norm, prefix + "attn_q_norm.weight");
    track(layer.attn_k_norm, prefix + "attn_k_norm.weight");
    track(layer.shortconv_conv, prefix + "shortconv.conv.weight");
    track(layer.shortconv_in_proj, prefix + "shortconv.in_proj.weight");
    track(layer.shortconv_out_proj, prefix + "shortconv.out_proj.weight");
    track(layer.ffn_gate, prefix + "ffn_gate.weight");
    track(layer.ffn_up, prefix + "ffn_up.weight");
    track(layer.ffn_down, prefix + "ffn_down.weight");
    track(layer.ffn_gate_inp, prefix + "ffn_gate_inp.weight");
    track(layer.exp_probs_b, prefix + "exp_probs_b.bias");
    track(layer.ffn_gate_exps, prefix + "ffn_gate_exps.weight");
    track(layer.ffn_up_exps, prefix + "ffn_up_exps.weight");
    track(layer.ffn_down_exps, prefix + "ffn_down_exps.weight");
  }
  std::sort(allocs.begin(), allocs.end(),  // LFM2
            [](const NamedPtr& a, const NamedPtr& b) { return a.ptr < b.ptr; });
  for (std::size_t k = 1; k < allocs.size(); ++k) {  // LFM2
    if (allocs[k].ptr == allocs[k - 1].ptr) {
      return fail("lfm2moe tensors " + allocs[k - 1].name + " and " +
                  allocs[k].name + " share storage");
    }
  }
  return weights;
}

// FLINT: qwen-chassis binder. Every layer is GQA attention (no shortconv),
// lead layers dense SwiGLU, the rest sigmoid-routed ROCmFP4 experts plus an
// ungated shared expert. Learned per-head sinks are F32 [num_attention_heads].
std::optional<QwenModelWeights> BindFlintWeights(
    const core::GgufReader& reader, const core::ModelConfig& config,
    std::string* error_msg) {
  auto fail = [&](const std::string& msg) -> std::optional<QwenModelWeights> {
    if (error_msg != nullptr) {
      *error_msg = msg;
    }
    return std::nullopt;
  };
  auto mul = [&](std::size_t a, std::size_t b) -> std::optional<std::size_t> {
    if (a != 0 && b > SIZE_MAX / a) {
      if (error_msg != nullptr) {
        *error_msg = "flint tensor size overflows";
      }
      return std::nullopt;
    }
    return a * b;
  };
  auto expect_dims = [&](const std::string& name, const std::uint64_t* dims,
                         std::size_t n_dims) -> bool {
    const auto* info = reader.FindTensor(name);
    if (info == nullptr || info->dimensions.size() != n_dims) {
      if (error_msg != nullptr) {
        *error_msg = "flint tensor rank mismatch for " + name;
      }
      return false;
    }
    for (std::size_t axis = 0; axis < n_dims; ++axis) {
      if (info->dimensions[axis] != dims[axis]) {
        if (error_msg != nullptr) {
          *error_msg = "flint tensor dimension mismatch for " + name +
                       ": axis " + std::to_string(axis) + " expected " +
                       std::to_string(dims[axis]) + " found " +
                       std::to_string(info->dimensions[axis]);
        }
        return false;
      }
    }
    return true;
  };
  auto require_fp4 = [&](const QwenTensorRef& ref,
                         const std::string& name) -> bool {
    if (!ref.empty() && ref.type != core::GgmlType::kQ4_0_ROCMFP4 &&
        ref.type != core::GgmlType::kQ4_0_ROCMFP4_FAST) {
      if (error_msg != nullptr) {
        *error_msg = "flint experts must be ROCmFP4 (Q4_0_ROCMFP4/_FAST); " +
                     name + " is " + std::string(core::ToString(ref.type)) +
                     ": build the file with gufo-quant LEAN";
      }
      return false;
    }
    return true;
  };

  QwenModelWeights weights;
  weights.config = config;
  const std::size_t hidden = config.hidden_size;
  const std::size_t vocab = config.vocab_size;
  const std::size_t attn = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t inter = config.intermediate_size;
  const std::size_t n_expert = config.expert_count;
  const std::size_t n_ff = config.expert_feed_forward_length;
  const std::size_t shexp_ff = config.expert_shared_feed_forward_length;
  const auto hidden_vocab = mul(hidden, vocab);
  if (!hidden_vocab.has_value()) {
    return std::nullopt;
  }

  weights.token_embd = ExtractTensorRef(reader, "token_embd.weight");
  if (reader.FindTensor("token_embd_norm.weight") != nullptr) {
    return fail("flint final norm must be output_norm.weight");
  }
  weights.output_norm = ExtractTensorRef(reader, "output_norm.weight");
  weights.output = ExtractTensorRef(reader, "output.weight");
  if (weights.output.empty()) {
    return fail("flint output.weight is required (untied)");
  }
  if (!ValidateTensor(weights.token_embd, *hidden_vocab, TensorRole::kEmbedding,
                      "token_embd.weight", error_msg, hidden) ||
      !ValidateTensor(weights.output_norm, hidden, TensorRole::kNorm,
                      "output_norm.weight", error_msg) ||
      !ValidateTensor(weights.output, *hidden_vocab, TensorRole::kProjection,
                      "output.weight", error_msg, hidden)) {
    return std::nullopt;
  }

  weights.layers.resize(config.num_layers);
  for (std::uint32_t i = 0; i < config.num_layers; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    auto& layer = weights.layers[i];
    layer.attn_norm = ExtractTensorRef(reader, prefix + "attn_norm.weight");
    layer.ffn_norm = ExtractTensorRef(reader, prefix + "ffn_norm.weight");
    if (!ValidateTensor(layer.attn_norm, hidden, TensorRole::kNorm,
                        prefix + "attn_norm.weight", error_msg) ||
        !ValidateTensor(layer.ffn_norm, hidden, TensorRole::kNorm,
                        prefix + "ffn_norm.weight", error_msg)) {
      return std::nullopt;
    }
    layer.is_full_attention = true;
    layer.attn_q = ExtractTensorRef(reader, prefix + "attn_q.weight");
    layer.attn_k = ExtractTensorRef(reader, prefix + "attn_k.weight");
    layer.attn_v = ExtractTensorRef(reader, prefix + "attn_v.weight");
    layer.attn_output = ExtractTensorRef(reader, prefix + "attn_output.weight");
    layer.attn_q_norm = ExtractTensorRef(reader, prefix + "attn_q_norm.weight");
    layer.attn_k_norm = ExtractTensorRef(reader, prefix + "attn_k_norm.weight");
    layer.attn_sinks = ExtractTensorRef(reader, prefix + "attn_sinks");
    const auto attn_hidden = mul(attn, hidden);
    const auto kv_hidden = mul(kv, hidden);
    const auto hidden_attn = mul(hidden, attn);
    if (!attn_hidden.has_value() || !kv_hidden.has_value() ||
        !hidden_attn.has_value()) {
      return std::nullopt;
    }
    if (!ValidateTensor(layer.attn_q, *attn_hidden, TensorRole::kProjection,
                        prefix + "attn_q.weight", error_msg, hidden) ||
        !ValidateTensor(layer.attn_k, *kv_hidden, TensorRole::kProjection,
                        prefix + "attn_k.weight", error_msg, hidden) ||
        !ValidateTensor(layer.attn_v, *kv_hidden, TensorRole::kProjection,
                        prefix + "attn_v.weight", error_msg, hidden) ||
        !ValidateTensor(layer.attn_output, *hidden_attn,
                        TensorRole::kProjection, prefix + "attn_output.weight",
                        error_msg, attn) ||
        !ValidateTensor(layer.attn_q_norm, config.head_dim, TensorRole::kNorm,
                        prefix + "attn_q_norm.weight", error_msg) ||
        !ValidateTensor(layer.attn_k_norm, config.head_dim, TensorRole::kNorm,
                        prefix + "attn_k_norm.weight", error_msg) ||
        !ValidateTensor(layer.attn_sinks, config.num_attention_heads,
                        TensorRole::kNorm, prefix + "attn_sinks", error_msg)) {
      return std::nullopt;
    }
    if (layer.attn_sinks.type != core::GgmlType::kF32) {
      return fail("flint " + prefix + "attn_sinks must be F32");
    }
    if (reader.FindTensor(prefix + "shortconv.in_proj.weight") != nullptr) {
      return fail("flint attention layer " + std::to_string(i) +
                  " has shortconv tensors");
    }
    if (i < config.dense_lead_layers) {
      layer.ffn_gate = ExtractTensorRef(reader, prefix + "ffn_gate.weight");
      layer.ffn_up = ExtractTensorRef(reader, prefix + "ffn_up.weight");
      layer.ffn_down = ExtractTensorRef(reader, prefix + "ffn_down.weight");
      const auto gate_elems = mul(inter, hidden);
      const auto down_elems = mul(hidden, inter);
      if (!gate_elems.has_value() || !down_elems.has_value()) {
        return std::nullopt;
      }
      if (!ValidateTensor(layer.ffn_gate, *gate_elems, TensorRole::kProjection,
                          prefix + "ffn_gate.weight", error_msg, hidden) ||
          !ValidateTensor(layer.ffn_up, *gate_elems, TensorRole::kProjection,
                          prefix + "ffn_up.weight", error_msg, hidden) ||
          !ValidateTensor(layer.ffn_down, *down_elems, TensorRole::kProjection,
                          prefix + "ffn_down.weight", error_msg, inter)) {
        return std::nullopt;
      }
      if (reader.FindTensor(prefix + "ffn_gate_inp.weight") != nullptr) {
        return fail("flint dense layer " + std::to_string(i) + " has a router");
      }
    } else {
      layer.is_moe = true;
      layer.ffn_gate_inp =
          ExtractTensorRef(reader, prefix + "ffn_gate_inp.weight");
      layer.exp_probs_b = ExtractTensorRef(reader, prefix + "exp_probs_b.bias");
      layer.ffn_gate_exps =
          ExtractTensorRef(reader, prefix + "ffn_gate_exps.weight");
      layer.ffn_up_exps =
          ExtractTensorRef(reader, prefix + "ffn_up_exps.weight");
      layer.ffn_down_exps =
          ExtractTensorRef(reader, prefix + "ffn_down_exps.weight");
      layer.ffn_gate_shexp =
          ExtractTensorRef(reader, prefix + "ffn_gate_shexp.weight");
      layer.ffn_up_shexp =
          ExtractTensorRef(reader, prefix + "ffn_up_shexp.weight");
      layer.ffn_down_shexp =
          ExtractTensorRef(reader, prefix + "ffn_down_shexp.weight");
      const auto router_elems = mul(hidden, n_expert);
      const auto hidden_ff = mul(hidden, n_ff);
      const auto ff_hidden = mul(n_ff, hidden);
      const auto shexp_gate = mul(hidden, shexp_ff);
      const auto shexp_down = mul(shexp_ff, hidden);
      if (!router_elems.has_value() || !hidden_ff.has_value() ||
          !ff_hidden.has_value() || !shexp_gate.has_value() ||
          !shexp_down.has_value()) {
        return std::nullopt;
      }
      const auto exp_gate = mul(*hidden_ff, n_expert);
      const auto exp_down = mul(*ff_hidden, n_expert);
      if (!exp_gate.has_value() || !exp_down.has_value()) {
        return std::nullopt;
      }
      if (!ValidateTensor(layer.ffn_gate_inp, *router_elems,
                          TensorRole::kSsmParameter,
                          prefix + "ffn_gate_inp.weight", error_msg) ||
          !ValidateTensor(layer.exp_probs_b, n_expert,
                          TensorRole::kSsmParameter,
                          prefix + "exp_probs_b.bias", error_msg) ||
          !ValidateTensor(layer.ffn_gate_exps, *exp_gate,
                          TensorRole::kProjection,
                          prefix + "ffn_gate_exps.weight", error_msg, hidden) ||
          !ValidateTensor(layer.ffn_up_exps, *exp_gate, TensorRole::kProjection,
                          prefix + "ffn_up_exps.weight", error_msg, hidden) ||
          !ValidateTensor(layer.ffn_down_exps, *exp_down,
                          TensorRole::kProjection,
                          prefix + "ffn_down_exps.weight", error_msg, n_ff) ||
          !ValidateTensor(
              layer.ffn_gate_shexp, *shexp_gate, TensorRole::kProjection,
              prefix + "ffn_gate_shexp.weight", error_msg, hidden) ||
          !ValidateTensor(layer.ffn_up_shexp, *shexp_gate,
                          TensorRole::kProjection,
                          prefix + "ffn_up_shexp.weight", error_msg, hidden) ||
          !ValidateTensor(
              layer.ffn_down_shexp, *shexp_down, TensorRole::kProjection,
              prefix + "ffn_down_shexp.weight", error_msg, shexp_ff)) {
        return std::nullopt;
      }
      if (layer.ffn_gate_inp.type != core::GgmlType::kF32) {
        return fail("flint router must be F32; " + prefix +
                    "ffn_gate_inp.weight is " +
                    std::string(core::ToString(layer.ffn_gate_inp.type)));
      }
      const std::uint64_t router_dims[] = {hidden, n_expert};
      const std::uint64_t gate_dims[] = {hidden, n_ff, n_expert};
      const std::uint64_t down_dims[] = {n_ff, hidden, n_expert};
      const std::uint64_t shexp_gate_dims[] = {hidden, shexp_ff};
      const std::uint64_t shexp_down_dims[] = {shexp_ff, hidden};
      if (!expect_dims(prefix + "ffn_gate_inp.weight", router_dims, 2) ||
          !expect_dims(prefix + "ffn_gate_exps.weight", gate_dims, 3) ||
          !expect_dims(prefix + "ffn_up_exps.weight", gate_dims, 3) ||
          !expect_dims(prefix + "ffn_down_exps.weight", down_dims, 3) ||
          !expect_dims(prefix + "ffn_gate_shexp.weight", shexp_gate_dims, 2) ||
          !expect_dims(prefix + "ffn_up_shexp.weight", shexp_gate_dims, 2) ||
          !expect_dims(prefix + "ffn_down_shexp.weight", shexp_down_dims, 2)) {
        return std::nullopt;
      }
      if (!require_fp4(layer.ffn_gate_exps, prefix + "ffn_gate_exps.weight") ||
          !require_fp4(layer.ffn_up_exps, prefix + "ffn_up_exps.weight") ||
          !require_fp4(layer.ffn_down_exps, prefix + "ffn_down_exps.weight")) {
        return std::nullopt;
      }
      if (reader.FindTensor(prefix + "ffn_gate_inp_shexp.weight") != nullptr) {
        return fail("flint's shared expert is ungated");
      }
    }
  }

  struct NamedPtr {
    const void* ptr = nullptr;
    std::string name;
  };
  std::vector<NamedPtr> allocs;
  auto track = [&](const QwenTensorRef& ref, const std::string& name) {
    if (!ref.empty()) {
      allocs.push_back({ref.data, name});
    }
  };
  track(weights.token_embd, "token_embd.weight");
  track(weights.output_norm, "output_norm.weight");
  track(weights.output, "output.weight");
  for (std::uint32_t i = 0; i < config.num_layers; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    const auto& layer = weights.layers[i];
    track(layer.attn_norm, prefix + "attn_norm.weight");
    track(layer.ffn_norm, prefix + "ffn_norm.weight");
    track(layer.attn_q, prefix + "attn_q.weight");
    track(layer.attn_k, prefix + "attn_k.weight");
    track(layer.attn_v, prefix + "attn_v.weight");
    track(layer.attn_output, prefix + "attn_output.weight");
    track(layer.attn_q_norm, prefix + "attn_q_norm.weight");
    track(layer.attn_k_norm, prefix + "attn_k_norm.weight");
    track(layer.attn_sinks, prefix + "attn_sinks");
    track(layer.ffn_gate, prefix + "ffn_gate.weight");
    track(layer.ffn_up, prefix + "ffn_up.weight");
    track(layer.ffn_down, prefix + "ffn_down.weight");
    track(layer.ffn_gate_inp, prefix + "ffn_gate_inp.weight");
    track(layer.exp_probs_b, prefix + "exp_probs_b.bias");
    track(layer.ffn_gate_exps, prefix + "ffn_gate_exps.weight");
    track(layer.ffn_up_exps, prefix + "ffn_up_exps.weight");
    track(layer.ffn_down_exps, prefix + "ffn_down_exps.weight");
    track(layer.ffn_gate_shexp, prefix + "ffn_gate_shexp.weight");
    track(layer.ffn_up_shexp, prefix + "ffn_up_shexp.weight");
    track(layer.ffn_down_shexp, prefix + "ffn_down_shexp.weight");
  }
  std::sort(allocs.begin(), allocs.end(),
            [](const NamedPtr& a, const NamedPtr& b) { return a.ptr < b.ptr; });
  for (std::size_t k = 1; k < allocs.size(); ++k) {
    if (allocs[k].ptr == allocs[k - 1].ptr) {
      return fail("flint tensors " + allocs[k - 1].name + " and " +
                  allocs[k].name + " share storage");
    }
  }
  return weights;
}

// KAT: ffn_gate_inp present (or config.is_moe) selects the routed expert
// branch. Expert row stride is n_rows * QuantizedRowBytes, matching
// gufo_fp4 / ling3_moe. The scalar shared-expert gate is a norm-role F32
// vector. Router logits are an F32 projection.
bool BindKatMoe(const core::GgufReader& reader, QwenLayerWeights& layer,
                const core::ModelConfig& config, const std::string& prefix,
                std::string* error_msg, bool require_fp4 = false) {  // audit J7
  layer.is_moe = true;
  layer.ffn_gate_inp = ExtractTensorRef(reader, prefix + "ffn_gate_inp.weight");
  layer.ffn_gate_exps =
      ExtractTensorRef(reader, prefix + "ffn_gate_exps.weight");
  layer.ffn_up_exps = ExtractTensorRef(reader, prefix + "ffn_up_exps.weight");
  layer.ffn_down_exps =
      ExtractTensorRef(reader, prefix + "ffn_down_exps.weight");
  layer.ffn_gate_inp_shexp =
      ExtractTensorRef(reader, prefix + "ffn_gate_inp_shexp.weight");
  layer.ffn_gate_shexp =
      ExtractTensorRef(reader, prefix + "ffn_gate_shexp.weight");
  layer.ffn_up_shexp = ExtractTensorRef(reader, prefix + "ffn_up_shexp.weight");
  layer.ffn_down_shexp =
      ExtractTensorRef(reader, prefix + "ffn_down_shexp.weight");

  const std::size_t hidden = config.hidden_size;
  const std::size_t n_expert = config.expert_count;
  const std::size_t n_ff = config.expert_feed_forward_length;
  const std::size_t n_shexp = config.expert_shared_feed_forward_length;
  const std::size_t exp_gate = hidden * n_ff * n_expert;
  const std::size_t exp_down = n_ff * hidden * n_expert;
  if (!ValidateTensor(layer.ffn_gate_inp, hidden * n_expert,
                      TensorRole::kProjection, prefix + "ffn_gate_inp.weight",
                      error_msg) ||
      !ValidateTensor(layer.ffn_gate_exps, exp_gate, TensorRole::kProjection,
                      prefix + "ffn_gate_exps.weight", error_msg, hidden) ||
      !ValidateTensor(layer.ffn_up_exps, exp_gate, TensorRole::kProjection,
                      prefix + "ffn_up_exps.weight", error_msg, hidden) ||
      !ValidateTensor(layer.ffn_down_exps, exp_down, TensorRole::kProjection,
                      prefix + "ffn_down_exps.weight", error_msg, n_ff) ||
      !ValidateTensor(layer.ffn_gate_inp_shexp, hidden, TensorRole::kNorm,
                      prefix + "ffn_gate_inp_shexp.weight", error_msg) ||
      !ValidateTensor(layer.ffn_gate_shexp, hidden * n_shexp,
                      TensorRole::kProjection, prefix + "ffn_gate_shexp.weight",
                      error_msg, hidden) ||
      !ValidateTensor(layer.ffn_up_shexp, hidden * n_shexp,
                      TensorRole::kProjection, prefix + "ffn_up_shexp.weight",
                      error_msg, hidden) ||
      !ValidateTensor(layer.ffn_down_shexp, n_shexp * hidden,
                      TensorRole::kProjection, prefix + "ffn_down_shexp.weight",
                      error_msg, n_shexp)) {
    return false;
  }

  {
    const std::uint64_t h = hidden;
    const std::uint64_t ff = n_ff;
    const std::uint64_t sh = n_shexp;
    const std::uint64_t ne = n_expert;
    const std::uint64_t gate_dims[] = {h, ff, ne};
    const std::uint64_t down_dims[] = {ff, h, ne};
    const std::uint64_t router_dims[] = {h, ne};
    const std::uint64_t shexp_up_dims[] = {h, sh};
    const std::uint64_t shexp_down_dims[] = {sh, h};
    const std::uint64_t shexp_gate_dims[] = {h};
    const auto expect_dims = [&](const std::string& name,
                                 const std::uint64_t* dims,
                                 std::size_t n_dims) -> bool {
      const auto* info = reader.FindTensor(name);
      if (info == nullptr || info->dimensions.size() != n_dims) {
        if (error_msg != nullptr) {
          *error_msg = "KAT tensor rank mismatch for " + name;
        }
        return false;
      }
      for (std::size_t axis = 0; axis < n_dims; ++axis) {
        if (info->dimensions[axis] != dims[axis]) {
          if (error_msg != nullptr) {
            *error_msg = "KAT tensor dimension mismatch for " + name +
                         ": axis " + std::to_string(axis) + " expected " +
                         std::to_string(dims[axis]) + " found " +
                         std::to_string(info->dimensions[axis]);
          }
          return false;
        }
      }
      return true;
    };
    if (!expect_dims(prefix + "ffn_gate_inp.weight", router_dims, 2) ||
        !expect_dims(prefix + "ffn_gate_exps.weight", gate_dims, 3) ||
        !expect_dims(prefix + "ffn_up_exps.weight", gate_dims, 3) ||
        !expect_dims(prefix + "ffn_down_exps.weight", down_dims, 3) ||
        !expect_dims(prefix + "ffn_gate_inp_shexp.weight", shexp_gate_dims,
                     1) ||
        !expect_dims(prefix + "ffn_gate_shexp.weight", shexp_up_dims, 2) ||
        !expect_dims(prefix + "ffn_up_shexp.weight", shexp_up_dims, 2) ||
        !expect_dims(prefix + "ffn_down_shexp.weight", shexp_down_dims, 2)) {
      return false;
    }
  }
  if (require_fp4) {  // audit J7
    const auto require_fp4_type = [&](const QwenTensorRef& ref,
                                      const std::string& name) -> bool {
      if (ref.type != core::GgmlType::kQ4_0_ROCMFP4 &&
          ref.type != core::GgmlType::kQ4_0_ROCMFP4_FAST) {
        if (error_msg != nullptr) {
          *error_msg =
              "qwen35moe trunk experts must be ROCmFP4 (Q4_0_ROCMFP4/_FAST); " +
              name + " is " + std::string(core::ToString(ref.type));
        }
        return false;
      }
      return true;
    };
    if (!require_fp4_type(layer.ffn_gate_exps,
                          prefix + "ffn_gate_exps.weight") ||
        !require_fp4_type(layer.ffn_up_exps, prefix + "ffn_up_exps.weight") ||
        !require_fp4_type(layer.ffn_down_exps,
                          prefix + "ffn_down_exps.weight") ||
        !require_fp4_type(layer.ffn_gate_shexp,
                          prefix + "ffn_gate_shexp.weight") ||
        !require_fp4_type(layer.ffn_up_shexp, prefix + "ffn_up_shexp.weight") ||
        !require_fp4_type(layer.ffn_down_shexp,
                          prefix + "ffn_down_shexp.weight")) {
      return false;
    }
  }
  return true;
}

bool BindKatFullAttention(const core::GgufReader& reader,
                          QwenLayerWeights& layer,
                          const core::ModelConfig& config,
                          const std::string& prefix, std::string* error_msg) {
  const std::size_t hidden = config.hidden_size;
  layer.is_full_attention = true;
  layer.attn_q = ExtractTensorRef(reader, prefix + "attn_q.weight");
  layer.attn_k = ExtractTensorRef(reader, prefix + "attn_k.weight");
  layer.attn_v = ExtractTensorRef(reader, prefix + "attn_v.weight");
  layer.attn_output = ExtractTensorRef(reader, prefix + "attn_output.weight");
  layer.attn_q_norm = ExtractTensorRef(reader, prefix + "attn_q_norm.weight");
  layer.attn_k_norm = ExtractTensorRef(reader, prefix + "attn_k_norm.weight");
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  return ValidateTensor(layer.attn_q, 2 * attention_size * hidden,
                        TensorRole::kProjection, prefix + "attn_q.weight",
                        error_msg, hidden) &&
         ValidateTensor(layer.attn_k, kv_size * hidden, TensorRole::kProjection,
                        prefix + "attn_k.weight", error_msg, hidden) &&
         ValidateTensor(layer.attn_v, kv_size * hidden, TensorRole::kProjection,
                        prefix + "attn_v.weight", error_msg, hidden) &&
         ValidateTensor(layer.attn_output, hidden * attention_size,
                        TensorRole::kProjection, prefix + "attn_output.weight",
                        error_msg, attention_size) &&
         ValidateTensor(layer.attn_q_norm, config.head_dim, TensorRole::kNorm,
                        prefix + "attn_q_norm.weight", error_msg) &&
         ValidateTensor(layer.attn_k_norm, config.head_dim, TensorRole::kNorm,
                        prefix + "attn_k_norm.weight", error_msg);
}

}  // namespace

std::optional<QwenModelWeights> QwenModelWeights::LoadFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto config_opt = reader.ExtractModelConfig(error_msg);
  if (!config_opt.has_value()) {
    return std::nullopt;
  }
  // GLIMMER: do not run the interval-4 kind check or the 2× attn_q check.
  if (config_opt->attention_only) {
    return BindGlimmerWeights(reader, *config_opt, error_msg);
  }
  if (config_opt->lfm2) {
    return BindLfm2Weights(reader, *config_opt, error_msg);
  }  // LFM2
  if (config_opt->flint) {
    return BindFlintWeights(reader, *config_opt, error_msg);
  }

  QwenModelWeights weights;
  weights.config = *config_opt;
  // KAT: the router stack and the GPU kernels cap at 256 experts and top-8.
  // Refuse at bind. Binding and then contributing zero is a wrong answer.
  if (weights.config.is_moe && (weights.config.expert_count > 256 ||
                                weights.config.expert_used_count > 8)) {
    if (error_msg != nullptr) {
      *error_msg =
          "KAT qwen35moe refuses expert_count > 256 or expert_used_count > 8"
          " (found expert_count=" +
          std::to_string(weights.config.expert_count) + " expert_used_count=" +
          std::to_string(weights.config.expert_used_count) + ")";
    }
    return std::nullopt;
  }

  weights.token_embd = ExtractTensorRef(reader, "token_embd.weight");
  weights.output_norm = ExtractTensorRef(reader, "output_norm.weight");
  weights.output = ExtractTensorRef(reader, "output.weight");
  if (weights.output.empty()) {
    // Tied LM head shares token embeddings
    weights.output = weights.token_embd;
    weights.output_tied_to_token_embd = true;  // KAT:
  }
  const std::size_t hidden_size = weights.config.hidden_size;
  const std::size_t vocab_size = weights.config.vocab_size;
  if (!ValidateTensor(weights.token_embd, vocab_size * hidden_size,
                      TensorRole::kEmbedding, "token_embd.weight", error_msg,
                      hidden_size) ||
      !ValidateTensor(weights.output_norm, hidden_size, TensorRole::kNorm,
                      "output_norm.weight", error_msg) ||
      !ValidateTensor(weights.output, vocab_size * hidden_size,
                      TensorRole::kProjection, "output.weight", error_msg,
                      hidden_size)) {
    return std::nullopt;
  }

  weights.layers.resize(weights.config.num_layers);
  for (std::uint32_t i = 0; i < weights.config.num_layers; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    auto& l = weights.layers[i];

    l.attn_norm = ExtractTensorRef(reader, prefix + "attn_norm.weight");
    if (l.attn_norm.empty()) {
      l.attn_norm = ExtractTensorRef(reader, prefix + "input_norm.weight");
    }

    l.ffn_norm = ExtractTensorRef(reader, prefix + "ffn_norm.weight");
    if (l.ffn_norm.empty()) {
      l.ffn_norm =
          ExtractTensorRef(reader, prefix + "post_attention_norm.weight");
    }
    if (l.ffn_norm.empty()) {
      l.ffn_norm = ExtractTensorRef(reader, prefix + "attn_post_norm.weight");
    }

    // Check if full attention layer
    l.attn_q = ExtractTensorRef(reader, prefix + "attn_q.weight");
    if (l.attn_q.empty()) {
      l.attn_q = ExtractTensorRef(reader, prefix + "wq.weight");
    }

    if (!l.attn_q.empty()) {
      l.is_full_attention = true;
      l.attn_k = ExtractTensorRef(reader, prefix + "attn_k.weight");
      if (l.attn_k.empty()) {
        l.attn_k = ExtractTensorRef(reader, prefix + "wk.weight");
      }
      l.attn_v = ExtractTensorRef(reader, prefix + "attn_v.weight");
      if (l.attn_v.empty()) {
        l.attn_v = ExtractTensorRef(reader, prefix + "wv.weight");
      }
      l.attn_output = ExtractTensorRef(reader, prefix + "attn_output.weight");
      if (l.attn_output.empty()) {
        l.attn_output = ExtractTensorRef(reader, prefix + "attn_out.weight");
      }
      if (l.attn_output.empty()) {
        l.attn_output = ExtractTensorRef(reader, prefix + "wo.weight");
      }
      l.attn_q_norm = ExtractTensorRef(reader, prefix + "attn_q_norm.weight");
      l.attn_k_norm = ExtractTensorRef(reader, prefix + "attn_k_norm.weight");
    } else {
      l.is_full_attention = false;
      l.attn_qkv = ExtractTensorRef(reader, prefix + "attn_qkv.weight");
      if (l.attn_qkv.empty()) {
        l.attn_qkv = ExtractTensorRef(reader, prefix + "wqkv.weight");
      }
      l.attn_gate = ExtractTensorRef(reader, prefix + "attn_gate.weight");
      if (l.attn_gate.empty()) {
        l.attn_gate = ExtractTensorRef(reader, prefix + "wqkv_gate.weight");
      }
      l.ssm_a = ExtractTensorRef(reader, prefix + "ssm_a");
      if (l.ssm_a.empty()) {
        l.ssm_a = ExtractTensorRef(reader, prefix + "ssm_a.weight");
      }
      l.ssm_conv1d = ExtractTensorRef(reader, prefix + "ssm_conv1d.weight");
      if (l.ssm_conv1d.empty()) {
        l.ssm_conv1d = ExtractTensorRef(reader, prefix + "ssm_conv1d");
      }
      l.ssm_dt = ExtractTensorRef(reader, prefix + "ssm_dt.bias");
      if (l.ssm_dt.empty()) {
        l.ssm_dt = ExtractTensorRef(reader, prefix + "ssm_dt.weight");
      }
      if (l.ssm_dt.empty()) {
        l.ssm_dt = ExtractTensorRef(reader, prefix + "ssm_dt");
      }
      l.ssm_alpha = ExtractTensorRef(reader, prefix + "ssm_alpha.weight");
      l.ssm_beta = ExtractTensorRef(reader, prefix + "ssm_beta.weight");
      l.ssm_norm = ExtractTensorRef(reader, prefix + "ssm_norm.weight");
      l.ssm_out = ExtractTensorRef(reader, prefix + "ssm_out.weight");
    }

    l.ffn_gate = ExtractTensorRef(reader, prefix + "ffn_gate.weight");
    l.ffn_up = ExtractTensorRef(reader, prefix + "ffn_up.weight");
    l.ffn_down = ExtractTensorRef(reader, prefix + "ffn_down.weight");

    const bool expected_full_attention =
        ((i + 1) % weights.config.full_attention_interval) == 0;
    if (l.is_full_attention != expected_full_attention) {
      if (error_msg != nullptr) {
        *error_msg =
            "Unexpected Qwen layer kind at blk." + std::to_string(i) +
            ": expected " +
            (expected_full_attention ? "full attention" : "Gated DeltaNet");
      }
      return std::nullopt;
    }

    l.ffn_gate_inp = ExtractTensorRef(reader, prefix + "ffn_gate_inp.weight");
    // KAT: MoE layer. Dense ffn_gate/up/down are absent on qwen35moe.
    const bool moe_layer = weights.config.is_moe || !l.ffn_gate_inp.empty();
    if (!ValidateTensor(l.attn_norm, hidden_size, TensorRole::kNorm,
                        prefix + "attn_norm.weight", error_msg) ||
        !ValidateTensor(l.ffn_norm, hidden_size, TensorRole::kNorm,
                        prefix + "post_attention_norm.weight", error_msg)) {
      return std::nullopt;
    }
    if (moe_layer) {
      if (!BindKatMoe(reader, l, weights.config, prefix, error_msg,
                      true)) {  // audit J7: trunk experts must be ROCmFP4
        return std::nullopt;
      }
    } else {
      const std::size_t intermediate_size = weights.config.intermediate_size;
      if (!ValidateTensor(l.ffn_gate, intermediate_size * hidden_size,
                          TensorRole::kProjection, prefix + "ffn_gate.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.ffn_up, intermediate_size * hidden_size,
                          TensorRole::kProjection, prefix + "ffn_up.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.ffn_down, hidden_size * intermediate_size,
                          TensorRole::kProjection, prefix + "ffn_down.weight",
                          error_msg, intermediate_size)) {
        return std::nullopt;
      }
    }

    if (l.is_full_attention) {
      const std::size_t attention_size = weights.config.AttentionSize();
      const std::size_t kv_size =
          static_cast<std::size_t>(weights.config.num_key_value_heads) *
          weights.config.head_dim;
      if (!ValidateTensor(l.attn_q, 2 * attention_size * hidden_size,
                          TensorRole::kProjection, prefix + "attn_q.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.attn_k, kv_size * hidden_size,
                          TensorRole::kProjection, prefix + "attn_k.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.attn_v, kv_size * hidden_size,
                          TensorRole::kProjection, prefix + "attn_v.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.attn_output, hidden_size * attention_size,
                          TensorRole::kProjection,
                          prefix + "attn_output.weight", error_msg,
                          attention_size) ||
          !ValidateTensor(l.attn_q_norm, weights.config.head_dim,
                          TensorRole::kNorm, prefix + "attn_q_norm.weight",
                          error_msg) ||
          !ValidateTensor(l.attn_k_norm, weights.config.head_dim,
                          TensorRole::kNorm, prefix + "attn_k_norm.weight",
                          error_msg)) {
        return std::nullopt;
      }
    } else {
      const std::size_t qkv_size = weights.config.SsmQkvSize();
      const std::size_t inner_size = weights.config.ssm_inner_size;
      const std::size_t rank = weights.config.ssm_time_step_rank;
      if (!ValidateTensor(l.attn_qkv, qkv_size * hidden_size,
                          TensorRole::kProjection, prefix + "attn_qkv.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.attn_gate, inner_size * hidden_size,
                          TensorRole::kProjection, prefix + "attn_gate.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.ssm_a, rank, TensorRole::kSsmParameter,
                          prefix + "ssm_a", error_msg) ||
          !ValidateTensor(l.ssm_conv1d,
                          qkv_size * weights.config.ssm_conv_kernel,
                          TensorRole::kSsmParameter,
                          prefix + "ssm_conv1d.weight", error_msg) ||
          !ValidateTensor(l.ssm_dt, rank, TensorRole::kSsmParameter,
                          prefix + "ssm_dt.bias", error_msg) ||
          !ValidateTensor(l.ssm_alpha, rank * hidden_size,
                          TensorRole::kProjection, prefix + "ssm_alpha.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.ssm_beta, rank * hidden_size,
                          TensorRole::kProjection, prefix + "ssm_beta.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.ssm_norm, weights.config.SsmValueSize(),
                          TensorRole::kSsmParameter, prefix + "ssm_norm.weight",
                          error_msg) ||
          !ValidateTensor(l.ssm_out, hidden_size * inner_size,
                          TensorRole::kProjection, prefix + "ssm_out.weight",
                          error_msg, inner_size)) {
        return std::nullopt;
      }
    }
  }

  // KAT: MTP blk.{num_layers} is bound beside the trunk, not inside it.
  // ForwardModel iterates layers.size()==num_layers, so the block stays
  // dormant.
  if (weights.config.is_moe && weights.config.mtp_num_layers > 0) {
    if (weights.config.mtp_num_layers != 1) {
      if (error_msg != nullptr) {
        *error_msg = "KAT qwen35moe binds a single MTP block";
      }
      return std::nullopt;
    }
    QwenLayerWeights mtp;
    const std::uint32_t mtp_index = weights.config.num_layers;
    const std::string prefix = "blk." + std::to_string(mtp_index) + ".";
    mtp.attn_norm = ExtractTensorRef(reader, prefix + "attn_norm.weight");
    mtp.ffn_norm =
        ExtractTensorRef(reader, prefix + "post_attention_norm.weight");
    if (mtp.ffn_norm.empty()) {
      mtp.ffn_norm = ExtractTensorRef(reader, prefix + "attn_post_norm.weight");
    }
    if (!ValidateTensor(mtp.attn_norm, hidden_size, TensorRole::kNorm,
                        prefix + "attn_norm.weight", error_msg) ||
        !ValidateTensor(mtp.ffn_norm, hidden_size, TensorRole::kNorm,
                        prefix + "post_attention_norm.weight", error_msg) ||
        !BindKatFullAttention(reader, mtp, weights.config, prefix, error_msg) ||
        !BindKatMoe(reader, mtp, weights.config, prefix, error_msg)) {
      return std::nullopt;
    }
    mtp.nextn_eh_proj =
        ExtractTensorRef(reader, prefix + "nextn.eh_proj.weight");
    mtp.nextn_enorm = ExtractTensorRef(reader, prefix + "nextn.enorm.weight");
    mtp.nextn_hnorm = ExtractTensorRef(reader, prefix + "nextn.hnorm.weight");
    mtp.nextn_shared_head_norm =
        ExtractTensorRef(reader, prefix + "nextn.shared_head_norm.weight");
    const std::size_t concat = hidden_size * 2;
    if (!ValidateTensor(mtp.nextn_eh_proj, hidden_size * concat,
                        TensorRole::kProjection,
                        prefix + "nextn.eh_proj.weight", error_msg, concat) ||
        !ValidateTensor(mtp.nextn_enorm, hidden_size, TensorRole::kNorm,
                        prefix + "nextn.enorm.weight", error_msg) ||
        !ValidateTensor(mtp.nextn_hnorm, hidden_size, TensorRole::kNorm,
                        prefix + "nextn.hnorm.weight", error_msg) ||
        !ValidateTensor(mtp.nextn_shared_head_norm, hidden_size,
                        TensorRole::kNorm,
                        prefix + "nextn.shared_head_norm.weight", error_msg)) {
      return std::nullopt;
    }
    const bool head_present =
        reader.FindTensor(prefix + "nextn.shared_head_head.weight") != nullptr;
    const bool emb_present =
        reader.FindTensor(prefix + "nextn.embed_tokens.weight") != nullptr;
    weights.mtp_head_tied_to_output = !head_present;
    weights.mtp_embed_tied_to_token = !emb_present;
    if (head_present || emb_present) {
      if (error_msg != nullptr) {
        *error_msg =
            "KAT MTP refuses an untied nextn head or embedding at blk." +
            std::to_string(mtp_index) +
            " (nextn.shared_head_head.weight or nextn.embed_tokens.weight is "
            "present)";
      }
      return std::nullopt;
    }
    weights.mtp_block_index = mtp_index;
    weights.mtp = std::move(mtp);
  }

  return weights;
}

}  // namespace gufo::models
