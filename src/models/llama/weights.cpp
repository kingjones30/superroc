#include "src/models/llama/weights.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace superroc::models {
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

enum class TensorRole : std::uint8_t { kEmbedding, kNorm, kProjection };

[[nodiscard]] constexpr bool SupportsTensorType(core::GgmlType type,
                                                TensorRole role) noexcept {
  switch (role) {
    case TensorRole::kEmbedding:
      return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
             type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ4_K ||
             type == core::GgmlType::kQ3_K || type == core::GgmlType::kQ5_K ||
             type == core::GgmlType::kQ6_K;
    case TensorRole::kNorm:
      return type == core::GgmlType::kF32;
    // LLAMA (F6): the batched prefill FP16 GEMM only implements
    // Q8_0/Q4_K/Q3_K/Q5_K/Q6_K. F32/BF16/Q8_K projections load but then
    // throw "unexpected half prefill weight format" on the first 2+ token
    // chunk, so refuse them here at bind time.
    case TensorRole::kProjection:
      return type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ5_K ||
             type == core::GgmlType::kQ6_K || type == core::GgmlType::kQ4_K ||
             type == core::GgmlType::kQ3_K;
  }
  return false;
}

bool ValidateTensor(const QwenTensorRef& tensor, std::size_t expected_elements,
                    TensorRole role, std::string_view name,
                    std::string* error_msg, std::size_t row_elements = 0) {
  if (tensor.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "Missing required llama tensor: " + std::string(name);
    }
    return false;
  }
  if (!SupportsTensorType(tensor.type, role)) {
    if (error_msg != nullptr) {
      *error_msg = "Unsupported llama tensor type for " + std::string(name) +
                   ": " + std::string(core::ToString(tensor.type));
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
      *error_msg = "Tensor payload is truncated: " + std::string(name);
    }
    return false;
  }
  const bool quantized = tensor.type != core::GgmlType::kF32 &&
                         tensor.type != core::GgmlType::kF16 &&
                         tensor.type != core::GgmlType::kBF16;
  if (quantized && row_elements != 0 &&
      superroc::quant::QuantizedRowBytes(tensor.type, row_elements) == 0) {
    if (error_msg != nullptr) {
      *error_msg =
          "Quantized tensor row is not block aligned: " + std::string(name);
    }
    return false;
  }
  return true;
}

}  // namespace

std::optional<QwenModelWeights> BindLlamaWeights(
    const core::GgufReader& reader, const core::ModelConfig& config,
    std::string* error_msg) {
  auto fail = [&](const std::string& msg) -> std::optional<QwenModelWeights> {
    if (error_msg != nullptr) {
      *error_msg = msg;
    }
    return std::nullopt;
  };
  if (!config.llama) {
    return fail("BindLlamaWeights requires config.llama");
  }

  QwenModelWeights weights;
  weights.config = config;
  const std::size_t hidden = config.hidden_size;
  const std::size_t vocab = config.vocab_size;
  const std::size_t attn = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t inter = config.intermediate_size;

  const std::string_view banned[] = {
      "attn_q_norm.weight", "attn_k_norm.weight", "attn_sinks",
      "attn_gate.weight",   "attn_qkv.weight",    "ffn_gate_inp.weight"};

  weights.token_embd = ExtractTensorRef(reader, "token_embd.weight");
  weights.output_norm = ExtractTensorRef(reader, "output_norm.weight");
  weights.output = ExtractTensorRef(reader, "output.weight");
  if (weights.output.empty()) {
    return fail("llama output.weight is required (untied)");
  }
  if (!ValidateTensor(weights.token_embd, vocab * hidden,
                      TensorRole::kEmbedding, "token_embd.weight", error_msg,
                      hidden) ||
      !ValidateTensor(weights.output_norm, hidden, TensorRole::kNorm,
                      "output_norm.weight", error_msg) ||
      !ValidateTensor(weights.output, vocab * hidden, TensorRole::kProjection,
                      "output.weight", error_msg, hidden)) {
    return std::nullopt;
  }

  std::vector<std::string> bound;
  bound.reserve(static_cast<std::size_t>(config.num_layers) * 9U + 3U);
  bound.emplace_back("token_embd.weight");
  bound.emplace_back("output_norm.weight");
  bound.emplace_back("output.weight");

  weights.layers.resize(config.num_layers);
  for (std::uint32_t i = 0; i < config.num_layers; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    for (const auto name : banned) {
      if (reader.FindTensor(prefix + std::string(name)) != nullptr) {
        return fail("llama refuses " + prefix + std::string(name));
      }
    }
    auto& layer = weights.layers[i];
    layer.is_full_attention = true;
    layer.is_moe = false;
    layer.attn_norm = ExtractTensorRef(reader, prefix + "attn_norm.weight");
    layer.ffn_norm = ExtractTensorRef(reader, prefix + "ffn_norm.weight");
    layer.attn_q = ExtractTensorRef(reader, prefix + "attn_q.weight");
    layer.attn_k = ExtractTensorRef(reader, prefix + "attn_k.weight");
    layer.attn_v = ExtractTensorRef(reader, prefix + "attn_v.weight");
    layer.attn_output = ExtractTensorRef(reader, prefix + "attn_output.weight");
    layer.ffn_gate = ExtractTensorRef(reader, prefix + "ffn_gate.weight");
    layer.ffn_up = ExtractTensorRef(reader, prefix + "ffn_up.weight");
    layer.ffn_down = ExtractTensorRef(reader, prefix + "ffn_down.weight");
    if (!ValidateTensor(layer.attn_norm, hidden, TensorRole::kNorm,
                        prefix + "attn_norm.weight", error_msg) ||
        !ValidateTensor(layer.ffn_norm, hidden, TensorRole::kNorm,
                        prefix + "ffn_norm.weight", error_msg) ||
        !ValidateTensor(layer.attn_q, attn * hidden, TensorRole::kProjection,
                        prefix + "attn_q.weight", error_msg, hidden) ||
        !ValidateTensor(layer.attn_k, kv * hidden, TensorRole::kProjection,
                        prefix + "attn_k.weight", error_msg, hidden) ||
        !ValidateTensor(layer.attn_v, kv * hidden, TensorRole::kProjection,
                        prefix + "attn_v.weight", error_msg, hidden) ||
        !ValidateTensor(layer.attn_output, hidden * attn,
                        TensorRole::kProjection, prefix + "attn_output.weight",
                        error_msg, attn) ||
        !ValidateTensor(layer.ffn_gate, inter * hidden, TensorRole::kProjection,
                        prefix + "ffn_gate.weight", error_msg, hidden) ||
        !ValidateTensor(layer.ffn_up, inter * hidden, TensorRole::kProjection,
                        prefix + "ffn_up.weight", error_msg, hidden) ||
        !ValidateTensor(layer.ffn_down, hidden * inter, TensorRole::kProjection,
                        prefix + "ffn_down.weight", error_msg, inter)) {
      return std::nullopt;
    }
    bound.push_back(prefix + "attn_norm.weight");
    bound.push_back(prefix + "ffn_norm.weight");
    bound.push_back(prefix + "attn_q.weight");
    bound.push_back(prefix + "attn_k.weight");
    bound.push_back(prefix + "attn_v.weight");
    bound.push_back(prefix + "attn_output.weight");
    bound.push_back(prefix + "ffn_gate.weight");
    bound.push_back(prefix + "ffn_up.weight");
    bound.push_back(prefix + "ffn_down.weight");
  }

  if (bound.size() != reader.GetTensorCount()) {
    return fail("llama bound " + std::to_string(bound.size()) +
                " tensors but the file has " +
                std::to_string(reader.GetTensorCount()));
  }
  // LLAMA (F15): no stdout census on load; the prompt CLI owns stdout as
  // the generation stream and the server logs through its own logger.
  for (const auto& tensor : reader.GetTensors()) {
    bool found = false;
    for (const auto& name : bound) {
      if (tensor.name == name) {
        found = true;
        break;
      }
    }
    if (!found) {
      return fail("llama file has unbound tensor " + std::string(tensor.name));
    }
  }

  return weights;
}

}  // namespace superroc::models
