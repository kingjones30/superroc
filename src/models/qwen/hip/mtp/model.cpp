#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen/hip/mtp/detail/allocation.hpp"
#include "src/models/qwen/hip/mtp/kat_mtp_types.hpp"

namespace gufo::hip {
namespace {

constexpr std::size_t kPackChunkRows = 32;

[[nodiscard]] std::uint16_t FloatToBfloat16Bits(float value) noexcept {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t rounding_bias = 0x7FFFU + ((bits >> 16U) & 1U);
  bits += rounding_bias;
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] const void* QuantizedRow(const models::QwenTensorRef& tensor,
                                       std::size_t row,
                                       std::size_t columns) noexcept {
  const std::size_t row_bytes = quant::QuantizedRowBytes(tensor.type, columns);
  if (row_bytes == 0) {
    return nullptr;
  }
  return static_cast<const std::uint8_t*>(tensor.data) + (row * row_bytes);
}

void ReadMatrixRow(const models::QwenTensorRef& tensor, std::size_t row,
                   std::size_t columns, float* output) {
  if (tensor.type == core::GgmlType::kF32) {
    const auto* source =
        static_cast<const float*>(tensor.data) + (row * columns);
    std::copy_n(source, columns, output);
    return;
  }
  if (tensor.type == core::GgmlType::kBF16) {
    const std::size_t offset = row * columns;
    for (std::size_t column = 0; column < columns; ++column) {
      const auto bits =
          static_cast<const std::uint16_t*>(tensor.data)[offset + column];
      output[column] =
          std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
    }
    return;
  }

  const void* source = QuantizedRow(tensor, row, columns);
  if (source == nullptr) {
    throw std::runtime_error("unsupported MTP matrix quantization");
  }
  switch (tensor.type) {
    case core::GgmlType::kQ3_K:
      quant::DequantizeQ3_K(source, output, columns);
      return;
    case core::GgmlType::kQ4_K:
      quant::DequantizeQ4_K(source, output, columns);
      return;
    case core::GgmlType::kQ6_K:
      quant::DequantizeQ6_K(source, output, columns);
      return;
    case core::GgmlType::kQ4_0:
      quant::DequantizeQ4_0(source, output, columns);
      return;
    case core::GgmlType::kQ8_0:
      quant::DequantizeQ8_0(source, output, columns);
      return;
    case core::GgmlType::kIQ4_NL:
      quant::DequantizeIQ4_NL(source, output, columns);
      return;
    case core::GgmlType::kIQ4_XS:
      quant::DequantizeIQ4_XS(source, output, columns);
      return;
    case core::GgmlType::kQ4_0_ROCMFP4:  // MTPFP4: packed to BF16 like the rest
      quant::DequantizeROCMFP4_Dual(source, output, columns);
      return;
    case core::GgmlType::kQ4_0_ROCMFP4_FAST:
      quant::DequantizeROCMFP4_Fast(source, output, columns);
      return;
    default:
      throw std::runtime_error("unsupported MTP matrix type");
  }
}

void CopyToDevice(void* destination, const void* source, std::size_t bytes) {
  const auto error =
      hipMemcpy(destination, source, bytes, hipMemcpyHostToDevice);
  if (error != hipSuccess) {
    throw std::runtime_error(std::string("MTP hipMemcpy failed: ") +
                             hipGetErrorString(error));
  }
}

models::QwenTensorRef CopyVectorF32(const models::QwenTensorRef& source,
                                    std::size_t elements,
                                    std::vector<void*>& allocations,
                                    std::size_t& packed_bytes) {
  std::vector<float> host(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    host[index] = source.Get(index);
  }
  const std::size_t bytes = elements * sizeof(float);
  void* device = detail::AllocateDevice(bytes);
  allocations.push_back(device);
  CopyToDevice(device, host.data(), bytes);
  packed_bytes += bytes;
  return {
      .data = device,
      .type = core::GgmlType::kF32,
      .num_elements = elements,
  };
}

models::QwenTensorRef PackMatrixBf16(const models::QwenTensorRef& source,
                                     std::size_t rows, std::size_t columns,
                                     std::vector<void*>& allocations,
                                     std::size_t& packed_bytes) {
  const std::size_t total_elements = rows * columns;
  const std::size_t total_bytes = total_elements * sizeof(std::uint16_t);
  void* device = detail::AllocateDevice(total_bytes);
  allocations.push_back(device);

  const std::size_t chunk_rows = std::min(rows, kPackChunkRows);
  std::vector<float> float_rows(chunk_rows * columns);
  std::vector<std::uint16_t> bf16_rows(chunk_rows * columns);
  for (std::size_t start = 0; start < rows; start += chunk_rows) {
    const std::size_t count = std::min(chunk_rows, rows - start);
    for (std::size_t local_row = 0; local_row < count; ++local_row) {
      ReadMatrixRow(source, start + local_row, columns,
                    float_rows.data() + (local_row * columns));
    }
    const std::size_t chunk_elements = count * columns;
    for (std::size_t index = 0; index < chunk_elements; ++index) {
      bf16_rows[index] = FloatToBfloat16Bits(float_rows[index]);
    }
    CopyToDevice(static_cast<std::uint8_t*>(device) +
                     (start * columns * sizeof(std::uint16_t)),
                 bf16_rows.data(), chunk_elements * sizeof(std::uint16_t));
  }

  packed_bytes += total_bytes;
  return {
      .data = device,
      .type = core::GgmlType::kBF16,
      .num_elements = total_elements,
  };
}

// lqw:spec5-q4. Sidecar Q4_0 stays packed. Decode reads it with the Q4_0
// GEMV; the prime path dequantizes a matrix into BF16 scratch on demand.
models::QwenTensorRef PackMatrix(const models::QwenTensorRef& source,
                                 std::size_t rows, std::size_t columns,
                                 std::vector<void*>& allocations,
                                 std::size_t& packed_bytes) {
  if (source.type != core::GgmlType::kQ4_0) {
    return PackMatrixBf16(source, rows, columns, allocations, packed_bytes);
  }
  const std::size_t row_bytes = quant::QuantizedRowBytes(source.type, columns);
  if (row_bytes == 0 || source.data == nullptr) {
    throw std::runtime_error("MTP Q4_0 matrix is not packable");
  }
  const std::size_t bytes = row_bytes * rows;
  void* device = detail::AllocateDevice(bytes);
  allocations.push_back(device);
  CopyToDevice(device, source.data, bytes);
  packed_bytes += bytes;
  return {
      .data = device,
      .type = core::GgmlType::kQ4_0,
      .num_elements = rows * columns,
  };
}

void ReleaseAllocations(std::vector<void*>& allocations) noexcept {
  for (void* allocation : allocations) {
    if (allocation != nullptr) {
      (void)hipFree(allocation);
    }
  }
  allocations.clear();
}

void PackPrivateWeights(speculative::QwenMtpWeights& weights,
                        std::vector<void*>& allocations,
                        std::size_t& packed_bytes) {
  const std::size_t hidden = weights.config.hidden_size;
  const std::size_t intermediate = weights.config.intermediate_size;
  const std::size_t attention = weights.config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(weights.config.num_key_value_heads) *
      weights.config.head_dim;

  weights.embedding_norm =
      CopyVectorF32(weights.embedding_norm, hidden, allocations, packed_bytes);
  weights.hidden_norm =
      CopyVectorF32(weights.hidden_norm, hidden, allocations, packed_bytes);
  weights.shared_head_norm = CopyVectorF32(weights.shared_head_norm, hidden,
                                           allocations, packed_bytes);
  weights.fusion_projection = PackMatrix(weights.fusion_projection, hidden,
                                         2 * hidden, allocations, packed_bytes);

  auto& layer = weights.layer;
  layer.attn_norm =
      CopyVectorF32(layer.attn_norm, hidden, allocations, packed_bytes);
  layer.attn_q = PackMatrix(layer.attn_q, 2 * attention, hidden, allocations,
                            packed_bytes);
  layer.attn_k =
      PackMatrix(layer.attn_k, kv, hidden, allocations, packed_bytes);
  layer.attn_v =
      PackMatrix(layer.attn_v, kv, hidden, allocations, packed_bytes);
  layer.attn_output = PackMatrix(layer.attn_output, hidden, attention,
                                 allocations, packed_bytes);
  layer.attn_q_norm = CopyVectorF32(layer.attn_q_norm, weights.config.head_dim,
                                    allocations, packed_bytes);
  layer.attn_k_norm = CopyVectorF32(layer.attn_k_norm, weights.config.head_dim,
                                    allocations, packed_bytes);
  layer.ffn_norm =
      CopyVectorF32(layer.ffn_norm, hidden, allocations, packed_bytes);
  layer.ffn_gate = PackMatrix(layer.ffn_gate, intermediate, hidden, allocations,
                              packed_bytes);
  layer.ffn_up =
      PackMatrix(layer.ffn_up, intermediate, hidden, allocations, packed_bytes);
  layer.ffn_down = PackMatrix(layer.ffn_down, hidden, intermediate, allocations,
                              packed_bytes);
}

}  // namespace

QwenMtpGpuModel::QwenMtpGpuModel(
    std::shared_ptr<const core::GgufReader> mtp_reader,
    std::shared_ptr<const QwenGpuModel> target_model,
    speculative::QwenMtpWeights weights, std::vector<void*> allocations,
    std::size_t packed_weight_bytes, double pack_time_seconds,
    std::vector<float> target_output_norm)
    : mtp_reader_(std::move(mtp_reader)),
      target_model_(std::move(target_model)),
      weights_(std::move(weights)),
      allocations_(std::move(allocations)),
      packed_weight_bytes_(packed_weight_bytes),
      pack_time_seconds_(pack_time_seconds),
      target_output_norm_(std::move(target_output_norm)) {}

QwenMtpGpuModel::~QwenMtpGpuModel() {
  ReleaseAllocations(allocations_);
}

std::shared_ptr<const QwenMtpGpuModel> QwenMtpGpuModel::Create(
    std::shared_ptr<const core::GgufReader> mtp_reader,
    std::shared_ptr<const QwenGpuModel> target_model, std::string* error_msg) {
  if (mtp_reader == nullptr || target_model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "MTP and target GPU models must not be null";
    }
    return nullptr;
  }
  // The GPU draft borrows token_embd and output from the target model a few
  // lines below, so a standalone one-layer sidecar (what llama-gguf-split
  // emits) is valid input here even though it carries neither tensor.
  auto weights = speculative::QwenMtpWeights::LoadFromGguf(
      *mtp_reader, error_msg, /*require_tied_embeddings=*/false);
  if (!weights.has_value()) {
    return nullptr;
  }
  const auto& target_weights = target_model->GetWeights();
  if (target_weights.config.hidden_size != weights->config.hidden_size ||
      target_weights.config.vocab_size != weights->config.vocab_size ||
      target_weights.token_embd.empty() || target_weights.output.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "MTP and target GPU model dimensions are incompatible";
    }
    return nullptr;
  }

  std::vector<void*> allocations;
  std::size_t packed_bytes = 0;
  std::vector<float> target_output_norm;
  const auto start = std::chrono::steady_clock::now();
  try {
    PackPrivateWeights(*weights, allocations, packed_bytes);
    // F1 support: the MTP head needs the target's final output_norm on
    // the host to normalize captured pre-norm features at ingest.
    target_output_norm.resize(target_weights.output_norm.num_elements);
    if (target_output_norm.empty() ||
        target_weights.output_norm.data == nullptr) {
      throw std::runtime_error("target output_norm is unavailable");
    }
    const auto norm_error = hipMemcpy(
        target_output_norm.data(), target_weights.output_norm.data,
        target_output_norm.size() * sizeof(float), hipMemcpyDeviceToHost);
    if (norm_error != hipSuccess) {
      throw std::runtime_error(
          std::string("MTP target output_norm copy failed: ") +
          hipGetErrorString(norm_error));
    }
  } catch (const std::exception& exception) {
    ReleaseAllocations(allocations);
    if (error_msg != nullptr) {
      *error_msg = exception.what();
    }
    return nullptr;
  }
  // F2, fork parity: the reference draft-mtp graph reads
  // nextn.embed_tokens ?: the SIDECAR's own token_embd and
  // nextn.shared_head_head ?: the sidecar's own output. King's Q4_0
  // sidecar carries both. SPEC5 keeps those matrices Q4_0 on device
  // (the BF16 expansion was a bandwidth tax and did not move acceptance).
  // A sidecar without either tensor still borrows the target's.
  const bool borrow_head = [] {
    const char* borrow = std::getenv("GUFO_MTP_BORROW_HEAD");
    return borrow != nullptr && *borrow != '\0' && std::atoi(borrow) != 0;
  }();
  if (borrow_head) {
    // Drop the raw sidecar refs (they point at the host mmap, not device
    // memory) so the borrow fallback below takes over.
    weights->token_embedding = models::QwenTensorRef{};
    weights->output = models::QwenTensorRef{};
  }
  const auto sidecar_embedding =
      borrow_head ? models::QwenTensorRef{} : weights->token_embedding;
  const void* const sidecar_embedding_data = sidecar_embedding.data;
  const auto sidecar_output =
      borrow_head ? models::QwenTensorRef{} : weights->output;
  if (!sidecar_embedding.empty()) {
    weights->token_embedding =
        PackMatrix(sidecar_embedding, weights->config.vocab_size,
                   weights->config.hidden_size, allocations, packed_bytes);
  }
  if (!sidecar_output.empty()) {
    weights->output =
        (sidecar_output.data == sidecar_embedding_data &&
         !sidecar_embedding.empty())
            ? weights->token_embedding
            : PackMatrix(sidecar_output, weights->config.vocab_size,
                         weights->config.hidden_size, allocations,
                         packed_bytes);
  }
  if (weights->token_embedding.empty()) {
    weights->token_embedding = target_weights.token_embd;
  }
  if (weights->output.empty()) {
    weights->output = target_weights.output;
  }
  const double pack_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();

  return std::shared_ptr<const QwenMtpGpuModel>(new QwenMtpGpuModel(
      std::move(mtp_reader), std::move(target_model), std::move(*weights),
      std::move(allocations), packed_bytes, pack_seconds,
      std::move(target_output_norm)));
}

std::shared_ptr<const QwenMtpGpuModel> QwenMtpGpuModel::CreateFromEmbedded(
    std::shared_ptr<const QwenGpuModel> target_model, std::string* error_msg) {
  // KAT-MTP: alias the target's already-resident nextn block.
  if (target_model == nullptr || !target_model->GetConfig().is_moe ||
      !target_model->GetWeights().mtp.has_value()) {
    if (error_msg != nullptr) {
      *error_msg = "Qwen MTP requires --mtp-model";
    }
    return nullptr;
  }
  const auto& target_weights = target_model->GetWeights();
  const auto& mtp = *target_weights.mtp;
  if (!mtp.is_moe || mtp.nextn_eh_proj.empty() ||
      mtp.nextn_enorm.type != core::GgmlType::kF32 ||
      mtp.nextn_hnorm.type != core::GgmlType::kF32 ||
      mtp.nextn_shared_head_norm.type != core::GgmlType::kF32 ||
      target_weights.token_embd.empty() || target_weights.output.empty() ||
      target_weights.output_norm.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "KAT MTP embedded block is incomplete";
    }
    return nullptr;
  }

  speculative::QwenMtpWeights weights;
  weights.config = target_weights.config;
  weights.embedding_norm = mtp.nextn_enorm;
  weights.hidden_norm = mtp.nextn_hnorm;
  weights.fusion_projection = mtp.nextn_eh_proj;
  weights.shared_head_norm = mtp.nextn_shared_head_norm;
  weights.layer = mtp;
  weights.token_embedding = target_weights.token_embd;
  weights.output = target_weights.output;

  // KAT-MTP F2 (RV44): refuse, at load time, any tensor type the draft kernels
  // cannot execute. Expert GEMV accepts only Q8_0/Q8_K; the KV-only prime runs
  // RunPrimeGemm, which handles Q4_0/Q8_0/Q8_K/BF16 and would otherwise read
  // anything else as BF16, producing a garbage draft KV.
  if (!KatMtpValidateEmbeddedBlockTypes(weights, error_msg)) {
    return nullptr;
  }

  std::vector<float> target_output_norm(
      target_weights.output_norm.num_elements);
  const auto norm_error = hipMemcpy(
      target_output_norm.data(), target_weights.output_norm.data,
      target_output_norm.size() * sizeof(float), hipMemcpyDeviceToHost);
  if (norm_error != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = std::string("KAT MTP output_norm copy failed: ") +
                   hipGetErrorString(norm_error);
    }
    return nullptr;
  }
  std::fprintf(stderr,
               "KAT-MTP embedded experts=%u topk=%u ff=%u eh_type=%d "
               "expert_type=%d embed_type=%d\n",
               weights.config.expert_count, weights.config.expert_used_count,
               weights.config.expert_feed_forward_length,
               static_cast<int>(mtp.nextn_eh_proj.type),
               static_cast<int>(mtp.ffn_gate_exps.type),
               static_cast<int>(weights.token_embedding.type));
  return std::shared_ptr<const QwenMtpGpuModel>(
      new QwenMtpGpuModel(nullptr, std::move(target_model), std::move(weights),
                          {}, 0, 0.0, std::move(target_output_norm)));
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
