#include "src/models/qwen/mtp_reference.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/forward.hpp"

namespace gufo::speculative {
namespace {

models::QwenTensorRef TensorRef(const core::GgufReader& reader,
                                std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr) {
    return {};
  }
  return {.data = tensor->data,
          .type = tensor->type,
          .num_elements = tensor->ElementCount()};
}

bool IsNormType(core::GgmlType type) noexcept {
  return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16;
}

bool IsMatrixType(core::GgmlType type) noexcept {
  return IsNormType(type) || type == core::GgmlType::kQ3_K ||
         type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ6_K ||
         type == core::GgmlType::kQ4_0 || type == core::GgmlType::kQ8_0 ||
         type == core::GgmlType::kIQ4_NL || type == core::GgmlType::kIQ4_XS ||
         // MTPFP4: an MTP block quantized with King's ROCmFP4 formats
         // (OxCoder's own block is type 101). The GPU draft packs these to BF16
         // on load.
         type == core::GgmlType::kQ4_0_ROCMFP4 ||
         type == core::GgmlType::kQ4_0_ROCMFP4_FAST;
}

bool Validate(const models::QwenTensorRef& tensor, std::size_t elements,
              std::string_view name, bool matrix, std::string* error_msg) {
  if (tensor.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "Missing Qwen MTP tensor: " + std::string(name);
    }
    return false;
  }
  if ((matrix && !IsMatrixType(tensor.type)) ||
      (!matrix && !IsNormType(tensor.type))) {
    if (error_msg != nullptr) {
      *error_msg = "Unsupported Qwen MTP tensor type for " + std::string(name) +
                   ": " + std::string(core::ToString(tensor.type));
    }
    return false;
  }
  if (tensor.num_elements != elements) {
    if (error_msg != nullptr) {
      *error_msg = "Qwen MTP tensor shape mismatch for " + std::string(name);
    }
    return false;
  }
  return true;
}

const void* OutputRow(const models::QwenTensorRef& output, std::size_t token_id,
                      std::size_t hidden_size) noexcept {
  if (output.type == core::GgmlType::kF32) {
    return static_cast<const float*>(output.data) + (token_id * hidden_size);
  }
  if (output.type == core::GgmlType::kBF16) {
    return static_cast<const std::uint16_t*>(output.data) +
           (token_id * hidden_size);
  }
  const std::size_t row_bytes =
      quant::QuantizedRowBytes(output.type, hidden_size);
  return static_cast<const std::uint8_t*>(output.data) + (token_id * row_bytes);
}

}  // namespace

std::optional<QwenMtpWeights> QwenMtpWeights::LoadFromGguf(
    const core::GgufReader& reader, std::string* error_msg,
    bool require_tied_embeddings) {
  const auto config = reader.ExtractModelConfig(error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  // The MTP (nextn) block is the file's LAST block: blk.64 in the Qwen3.8-27B
  // sidecar (block_count 65), blk.72 in Agnes-3.0-Flash's, blk.32 in
  // OxCoder-9B. It was hard-coded to 64, which refused every other dense qwen35
  // drafter with "Missing Qwen MTP tensor: blk.64.nextn.enorm.weight" (lane
  // PMX, 2026-10-14). 64 stays the fallback when the file has no block_count.
  const std::string arch_name(
      reader.GetMetadataString("general.architecture").value_or("qwen35"));
  const auto block_count = reader.GetMetadataUint32(arch_name + ".block_count");
  const std::uint32_t mtp_block =
      (block_count.has_value() && *block_count > 0) ? *block_count - 1 : 64;
  const std::string blk = "blk." + std::to_string(mtp_block) + ".";

  QwenMtpWeights weights;
  weights.config = *config;
  const std::size_t hidden = weights.config.hidden_size;
  const std::size_t intermediate = weights.config.intermediate_size;
  const std::size_t attention = weights.config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(weights.config.num_key_value_heads) *
      weights.config.head_dim;
  const std::size_t vocab = weights.config.vocab_size;

  weights.token_embedding = TensorRef(reader, "token_embd.weight");
  weights.output = TensorRef(reader, "output.weight");
  if (weights.output.empty()) {
    weights.output = weights.token_embedding;
  }
  weights.embedding_norm = TensorRef(reader, blk + "nextn.enorm.weight");
  weights.hidden_norm = TensorRef(reader, blk + "nextn.hnorm.weight");
  weights.fusion_projection = TensorRef(reader, blk + "nextn.eh_proj.weight");
  weights.shared_head_norm =
      TensorRef(reader, blk + "nextn.shared_head_norm.weight");

  auto& layer = weights.layer;
  layer.is_full_attention = true;
  layer.attn_norm = TensorRef(reader, blk + "attn_norm.weight");
  layer.attn_q = TensorRef(reader, blk + "attn_q.weight");
  layer.attn_k = TensorRef(reader, blk + "attn_k.weight");
  layer.attn_v = TensorRef(reader, blk + "attn_v.weight");
  layer.attn_output = TensorRef(reader, blk + "attn_output.weight");
  layer.attn_q_norm = TensorRef(reader, blk + "attn_q_norm.weight");
  layer.attn_k_norm = TensorRef(reader, blk + "attn_k_norm.weight");
  layer.ffn_norm = TensorRef(reader, blk + "post_attention_norm.weight");
  layer.ffn_gate = TensorRef(reader, blk + "ffn_gate.weight");
  layer.ffn_up = TensorRef(reader, blk + "ffn_up.weight");
  layer.ffn_down = TensorRef(reader, blk + "ffn_down.weight");

  const bool valid =
      (!require_tied_embeddings ||
       (Validate(weights.token_embedding, vocab * hidden, "token_embd.weight",
                 true, error_msg) &&
        Validate(weights.output, vocab * hidden, "output.weight", true,
                 error_msg))) &&
      Validate(weights.embedding_norm, hidden, blk + "nextn.enorm.weight",
               false, error_msg) &&
      Validate(weights.hidden_norm, hidden, blk + "nextn.hnorm.weight", false,
               error_msg) &&
      Validate(weights.fusion_projection, hidden * hidden * 2,
               blk + "nextn.eh_proj.weight", true, error_msg) &&
      Validate(weights.shared_head_norm, hidden,
               blk + "nextn.shared_head_norm.weight", false, error_msg) &&
      Validate(layer.attn_norm, hidden, blk + "attn_norm.weight", false,
               error_msg) &&
      Validate(layer.attn_q, 2 * attention * hidden, blk + "attn_q.weight",
               true, error_msg) &&
      Validate(layer.attn_k, kv * hidden, blk + "attn_k.weight", true,
               error_msg) &&
      Validate(layer.attn_v, kv * hidden, blk + "attn_v.weight", true,
               error_msg) &&
      Validate(layer.attn_output, hidden * attention,
               blk + "attn_output.weight", true, error_msg) &&
      Validate(layer.attn_q_norm, weights.config.head_dim,
               blk + "attn_q_norm.weight", false, error_msg) &&
      Validate(layer.attn_k_norm, weights.config.head_dim,
               blk + "attn_k_norm.weight", false, error_msg) &&
      Validate(layer.ffn_norm, hidden, blk + "post_attention_norm.weight",
               false, error_msg) &&
      Validate(layer.ffn_gate, intermediate * hidden, blk + "ffn_gate.weight",
               true, error_msg) &&
      Validate(layer.ffn_up, intermediate * hidden, blk + "ffn_up.weight", true,
               error_msg) &&
      Validate(layer.ffn_down, hidden * intermediate, blk + "ffn_down.weight",
               true, error_msg);
  if (!valid) {
    return std::nullopt;
  }
  return weights;
}

std::unique_ptr<QwenMtpReference> QwenMtpReference::Create(
    std::shared_ptr<const core::GgufReader> reader, std::uint32_t max_context,
    std::string* error_msg) {
  if (reader == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "Qwen MTP GGUF reader must not be null";
    }
    return nullptr;
  }
  auto weights = QwenMtpWeights::LoadFromGguf(*reader, error_msg);
  if (!weights.has_value()) {
    return nullptr;
  }
  if (max_context == 0 || max_context > weights->config.context_length) {
    if (error_msg != nullptr) {
      *error_msg = "Qwen MTP context length is invalid";
    }
    return nullptr;
  }
  return std::unique_ptr<QwenMtpReference>(new QwenMtpReference(
      std::move(reader), nullptr, std::move(*weights), max_context));
}

std::unique_ptr<QwenMtpReference> QwenMtpReference::CreateWithTiedWeights(
    std::shared_ptr<const core::GgufReader> reader,
    std::shared_ptr<const core::GgufReader> tied_reader,
    std::uint32_t max_context, std::string* error_msg) {
  if (reader == nullptr || tied_reader == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "Qwen MTP and tied-weight readers must not be null";
    }
    return nullptr;
  }
  auto weights = QwenMtpWeights::LoadFromGguf(*reader, error_msg);
  auto tied_weights =
      models::QwenModelWeights::LoadFromGguf(*tied_reader, error_msg);
  if (!weights.has_value() || !tied_weights.has_value()) {
    return nullptr;
  }
  if (weights->config.hidden_size != tied_weights->config.hidden_size ||
      weights->config.vocab_size != tied_weights->config.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "Qwen MTP and tied model dimensions are incompatible";
    }
    return nullptr;
  }
  if (max_context == 0 || max_context > weights->config.context_length) {
    if (error_msg != nullptr) {
      *error_msg = "Qwen MTP context length is invalid";
    }
    return nullptr;
  }
  weights->token_embedding = tied_weights->token_embd;
  weights->output = tied_weights->output;
  return std::unique_ptr<QwenMtpReference>(
      new QwenMtpReference(std::move(reader), std::move(tied_reader),
                           std::move(*weights), max_context));
}

QwenMtpReference::QwenMtpReference(
    std::shared_ptr<const core::GgufReader> reader,
    std::shared_ptr<const core::GgufReader> tied_reader, QwenMtpWeights weights,
    std::uint32_t max_context)
    : reader_(std::move(reader)),
      tied_reader_(std::move(tied_reader)),
      weights_(std::move(weights)),
      kv_cache_(1, weights_.config.num_key_value_heads, max_context,
                weights_.config.head_dim),
      ssm_cache_(
          1, weights_.config.SsmQkvSize(), weights_.config.ssm_conv_kernel,
          weights_.config.ssm_time_step_rank, weights_.config.ssm_state_size,
          weights_.config.SsmValueSize()),
      arena_(weights_.config),
      max_context_(max_context),
      embedding_(weights_.config.hidden_size),
      normalized_embedding_(weights_.config.hidden_size),
      normalized_hidden_(weights_.config.hidden_size),
      fusion_input_(std::size_t{2} * weights_.config.hidden_size),
      feedback_hidden_(weights_.config.hidden_size) {}

void QwenMtpReference::Reset() noexcept {
  kv_cache_.Reset();
  ssm_cache_.Reset();
  next_position_ = 0;
  std::ranges::fill(feedback_hidden_, 0.0F);
}

std::span<const float> QwenMtpReference::ForwardHidden(
    tokenization::TokenId input_token,
    std::span<const float> target_or_feedback_hidden, std::uint32_t position) {
  const std::size_t hidden = weights_.config.hidden_size;
  if (target_or_feedback_hidden.size() != hidden || position >= max_context_ ||
      position != next_position_) {
    return {};
  }

  models::ForwardEmbedding(input_token, weights_.token_embedding, hidden,
                           embedding_);
  models::ForwardRMSNorm(embedding_, weights_.embedding_norm, 1.0e-6F,
                         normalized_embedding_);
  models::ForwardRMSNorm(target_or_feedback_hidden, weights_.hidden_norm,
                         1.0e-6F, normalized_hidden_);
  std::ranges::copy(normalized_embedding_, fusion_input_.begin());
  std::ranges::copy(
      normalized_hidden_,
      fusion_input_.begin() + static_cast<std::ptrdiff_t>(hidden));
  models::TensorGEMV(weights_.fusion_projection, fusion_input_, hidden,
                     2 * hidden, arena_.hidden);
  models::ForwardLayer(arena_.hidden, weights_.layer, weights_.config,
                       kv_cache_, ssm_cache_, 0, position, arena_);
  models::ForwardRMSNorm(arena_.hidden, weights_.shared_head_norm, 1.0e-6F,
                         feedback_hidden_);
  ++next_position_;
  return feedback_hidden_;
}

float QwenMtpReference::ComputeLogit(std::uint32_t token_id) const noexcept {
  const std::size_t hidden = weights_.config.hidden_size;
  if (token_id >= weights_.config.vocab_size) {
    return 0.0F;
  }
  const void* row = OutputRow(weights_.output, token_id, hidden);
  const models::QwenTensorRef tensor{
      .data = row, .type = weights_.output.type, .num_elements = hidden};
  float result = 0.0F;
  models::TensorGEMV(tensor, feedback_hidden_, 1, hidden,
                     std::span<float>(&result, 1));
  return result;
}

}  // namespace gufo::speculative
