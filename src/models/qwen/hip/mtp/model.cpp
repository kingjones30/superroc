#if defined(ENGINE_ENABLE_HIP)
#include <sys/stat.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/env.hpp"
#include "src/core/model_family.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen/hip/mtp/detail/allocation.hpp"
#include "src/models/qwen/hip/mtp/kat_mtp_types.hpp"
#include "src/models/qwen/state.hpp"

namespace superroc::hip {
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
    case core::GgmlType::kQ5_K:
      quant::DequantizeQ5_K(source, output, columns);
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
    case core::GgmlType::kIQ3_S:
      quant::DequantizeIQ3_S(source, output, columns);
      return;
    case core::GgmlType::kQ4_0_ROCMFP4:  // Packed to BF16 like the rest
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

// Re-encodes a matrix as Q4_0 on the host and uploads it. Rows are independent,
// so a few threads share each chunk.
models::QwenTensorRef PackMatrixQ4_0(const models::QwenTensorRef& source,
                                     std::size_t rows, std::size_t columns,
                                     std::vector<void*>& allocations,
                                     std::size_t& packed_bytes) {
  const std::size_t row_bytes =
      quant::QuantizedRowBytes(core::GgmlType::kQ4_0, columns);
  if (row_bytes == 0 || rows == 0) {
    throw std::runtime_error("MTP matrix has no Q4_0 row size");
  }
  const std::size_t total_bytes = rows * row_bytes;
  void* device = detail::AllocateDevice(total_bytes);
  allocations.push_back(device);

  // The first row runs here, so an unsupported source type throws on this
  // thread before any worker starts.
  std::vector<float> first(columns);
  ReadMatrixRow(source, 0, columns, first.data());

  constexpr std::size_t kChunkRows = 2048;
  const std::size_t chunk_rows = std::min(rows, kChunkRows);
  const std::size_t workers = std::min<std::size_t>(
      {8, std::max(1U, std::thread::hardware_concurrency()), chunk_rows});
  std::vector<std::uint8_t> packed(chunk_rows * row_bytes);
  for (std::size_t start = 0; start < rows; start += chunk_rows) {
    const std::size_t count = std::min(chunk_rows, rows - start);
    std::vector<std::exception_ptr> failures(workers);
    {
      std::vector<std::jthread> pool;
      pool.reserve(workers);
      for (std::size_t worker = 0; worker < workers; ++worker) {
        pool.emplace_back([&, worker] {
          try {
            std::vector<float> row(columns);
            for (std::size_t local = worker; local < count; local += workers) {
              ReadMatrixRow(source, start + local, columns, row.data());
              quant::QuantizeQ4_0(row.data(),
                                  packed.data() + (local * row_bytes), columns);
            }
          } catch (...) {
            failures[worker] = std::current_exception();
          }
        });
      }
    }
    for (const auto& failure : failures) {
      if (failure != nullptr) {
        std::rethrow_exception(failure);
      }
    }
    CopyToDevice(static_cast<std::uint8_t*>(device) + (start * row_bytes),
                 packed.data(), count * row_bytes);
  }

  packed_bytes += total_bytes;
  return {
      .data = device,
      .type = core::GgmlType::kQ4_0,
      .num_elements = rows * columns,
  };
}

// Sidecar Q4_0 stays packed. Decode reads it with the Q4_0
// GEMV; the prime path dequantizes a matrix into BF16 scratch on demand.
models::QwenTensorRef PackMatrix(const models::QwenTensorRef& source,
                                 std::size_t rows, std::size_t columns,
                                 std::vector<void*>& allocations,
                                 std::size_t& packed_bytes) {
  // A head matrix stays in the file's packed form when decode's launchers have
  // an integer route for it and RunPrimeGemm can read it. SUPERROC_MTP_PACK=0
  // expands everything but Q4_0 to BF16, as before.
  static const bool keep_packed = [] {
    const char* e = ::superroc::core::GetEnv("SUPERROC_MTP_PACK");
    return e == nullptr || e[0] != '0';
  }();
  const bool packed_type =
      source.type == core::GgmlType::kQ4_0 ||
      (keep_packed && (source.type == core::GgmlType::kQ8_0 ||
                       source.type == core::GgmlType::kQ8_K ||
                       source.type == core::GgmlType::kQ4_0_ROCMFP4 ||
                       source.type == core::GgmlType::kQ4_0_ROCMFP4_FAST));
  if (!packed_type) {
    // A type the draft launchers have no packed route for (the K and IQ
    // quants, BF16, F32) used to be expanded to BF16, which every draft step
    // then read in full. Q4_0 is a quarter of those bytes. Draft tokens are
    // proposals the target verifies, so the encoding moves the draft's speed
    // and acceptance rate, never the output. SUPERROC_MTP_DRAFT_Q4=0 keeps
    // the BF16 expansion.
    static const bool draft_q4 = [] {
      const char* e = ::superroc::core::GetEnv("SUPERROC_MTP_DRAFT_Q4");
      return e == nullptr || e[0] != '0';
    }();
    if (draft_q4 && columns != 0 && (columns % 32) == 0) {
      static bool announced = false;
      if (!announced) {
        announced = true;
        std::fprintf(stderr,
                     "MTP draft weights: re-encoded as Q4_0 for the draft "
                     "(SUPERROC_MTP_DRAFT_Q4=0 keeps BF16)\n");
      }
      return PackMatrixQ4_0(source, rows, columns, allocations, packed_bytes);
    }
    return PackMatrixBf16(source, rows, columns, allocations, packed_bytes);
  }
  const std::size_t row_bytes = quant::QuantizedRowBytes(source.type, columns);
  if (row_bytes == 0 || source.data == nullptr) {
    throw std::runtime_error("MTP packed matrix is not packable");
  }
  const std::size_t bytes = row_bytes * rows;
  void* device = detail::AllocateDevice(bytes);
  allocations.push_back(device);
  CopyToDevice(device, source.data, bytes);
  packed_bytes += bytes;
  return {
      .data = device,
      .type = source.type,
      .num_elements = rows * columns,
  };
}

// True when both readers map the same file or files (device and inode), which
// is how an MTP file that is the model file itself is recognized.
bool MapSameFiles(const core::GgufReader& left, const core::GgufReader& right) {
  const auto a = left.GetMappedRegions();
  const auto b = right.GetMappedRegions();
  if (a.empty() || a.size() != b.size()) {
    return false;
  }
  for (std::size_t index = 0; index < a.size(); ++index) {
    struct stat sa{};
    struct stat sb{};
    if (a[index].file_descriptor < 0 || b[index].file_descriptor < 0 ||
        ::fstat(a[index].file_descriptor, &sa) != 0 ||
        ::fstat(b[index].file_descriptor, &sb) != 0 || sa.st_dev != sb.st_dev ||
        sa.st_ino != sb.st_ino) {
      return false;
    }
  }
  return true;
}

void ReleaseAllocations(std::vector<void*>& allocations) noexcept {
  for (void* allocation : allocations) {
    if (allocation != nullptr) {
      (void)hipFree(allocation);
    }
  }
  allocations.clear();
}

// Sidecar block tensors stay in their file type: the encoded bytes are copied
// as-is, so the draft runs the same matrices the file carries.
models::QwenTensorRef CopyTensorBytes(const models::QwenTensorRef& source,
                                      std::vector<void*>& allocations,
                                      std::size_t& packed_bytes) {
  const std::size_t bytes = source.EncodedSizeBytes();
  void* device = detail::AllocateDevice(bytes);
  allocations.push_back(device);
  CopyToDevice(device, source.data, bytes);
  packed_bytes += bytes;
  return {
      .data = device,
      .type = source.type,
      .num_elements = source.num_elements,
  };
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

bool QwenMtpGpuModel::BuildBlockWeights(speculative::QwenMtpWeights& weights,
                                        const QwenGpuModel& target_model,
                                        const models::QwenLayerWeights& block,
                                        std::vector<float>* target_output_norm,
                                        std::string* error_msg) {
  const auto& target_weights = target_model.GetWeights();
  weights.config = target_weights.config;
  weights.embedding_norm = block.nextn_enorm;
  weights.hidden_norm = block.nextn_hnorm;
  weights.fusion_projection = block.nextn_eh_proj;
  weights.shared_head_norm = block.nextn_shared_head_norm;
  weights.layer = block;
  // The head's tied embedding and LM head are always the target's.
  weights.token_embedding = target_weights.token_embd;
  weights.output = target_weights.output;

  // KAT-MTP F2: refuse, at load time, any tensor type the draft kernels
  // cannot execute. Expert GEMV accepts only Q8_0/Q8_K; the KV-only prime runs
  // RunPrimeGemm, which handles Q4_0/Q8_0/Q8_K/BF16 and would otherwise read
  // anything else as BF16, producing a garbage draft KV.
  if (!KatMtpValidateEmbeddedBlockTypes(weights, error_msg)) {
    return false;
  }

  // The MTP head needs the target's final output_norm on the host
  // to normalize captured pre-norm features at ingest.
  target_output_norm->resize(target_weights.output_norm.num_elements);
  const auto norm_error = hipMemcpy(
      target_output_norm->data(), target_weights.output_norm.data,
      target_output_norm->size() * sizeof(float), hipMemcpyDeviceToHost);
  if (norm_error != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = std::string("KAT MTP output_norm copy failed: ") +
                   hipGetErrorString(norm_error);
    }
    return false;
  }
  return true;
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
    // The MTP head needs the target's final output_norm on
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
  // Fork parity: the reference draft-mtp graph reads
  // nextn.embed_tokens ?: the SIDECAR's own token_embd and
  // nextn.shared_head_head ?: the sidecar's own output. The Q4_0
  // sidecar carries both. Those matrices stay Q4_0 on device
  // (the BF16 expansion was a bandwidth tax and did not move acceptance).
  // A sidecar without either tensor still borrows the target's. So does an
  // MTP file that is the model file: its token_embd and output are the
  // tensors the target already holds on the device, and a second copy of
  // both would only cost memory. Each tensor is borrowed independently:
  // SUPERROC_MTP_BORROW_EMBD=1 borrows token_embd, =0 keeps the sidecar's;
  // SUPERROC_MTP_BORROW_HEAD=1 borrows output, =0 keeps the sidecar's. When
  // neither is set, an MTP file that is the model file borrows both.
  // (Before M4, BORROW_HEAD=1 borrowed both; it now borrows output only, so
  // the old BORROW_HEAD=1 spells EMBD=1 plus HEAD=1.)
  // SUPERROC_MTP_LEAN=1 borrows both (memory-tight seats); the three
  // single switches keep working for tests.
  const char* const borrow_embd_env =
      ::superroc::core::GetEnv("SUPERROC_MTP_BORROW_EMBD");
  const char* const borrow_head_env =
      ::superroc::core::GetEnv("SUPERROC_MTP_BORROW_HEAD");
  const char* const lean_env = ::superroc::core::GetEnv("SUPERROC_MTP_LEAN");
  const bool embd_set = borrow_embd_env != nullptr && *borrow_embd_env != '\0';
  const bool head_set = borrow_head_env != nullptr && *borrow_head_env != '\0';
  const bool lean =
      lean_env != nullptr && *lean_env != '\0' && std::atoi(lean_env) != 0;
  const bool same_file = MapSameFiles(*mtp_reader, target_model->GetReader());
  const bool borrow_embd =
      lean ||
      ((embd_set || head_set) ? (embd_set && std::atoi(borrow_embd_env) != 0)
                              : same_file);
  const bool borrow_head =
      lean ||
      ((embd_set || head_set) ? (head_set && std::atoi(borrow_head_env) != 0)
                              : same_file);
  const char* const borrow_why =
      lean ? "SUPERROC_MTP_LEAN=1"
           : ((embd_set || head_set) ? "SUPERROC_MTP_BORROW_EMBD/HEAD"
                                     : "the MTP file is the model file");
  if ((borrow_embd && !weights->token_embedding.empty()) ||
      (borrow_head && !weights->output.empty())) {
    std::fprintf(stderr, "MTP head: borrowing the model's %s%s%s (%s)\n",
                 borrow_embd ? "token_embd" : "",
                 (borrow_embd && borrow_head) ? " and " : "",
                 borrow_head ? "output" : "", borrow_why);
  }
  if (borrow_embd) {
    // Drop the raw sidecar ref (it points at the host mmap, not device
    // memory) so the borrow fallback below takes over.
    weights->token_embedding = models::QwenTensorRef{};
  }
  if (borrow_head) {
    weights->output = models::QwenTensorRef{};
  }
  const auto sidecar_embedding =
      borrow_embd ? models::QwenTensorRef{} : weights->token_embedding;
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
  std::vector<float> target_output_norm;
  if (!BuildBlockWeights(weights, *target_model, mtp, &target_output_norm,
                         error_msg)) {
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

std::shared_ptr<const QwenMtpGpuModel> QwenMtpGpuModel::CreateFromMoeSidecar(
    std::shared_ptr<const core::GgufReader> reader,
    std::shared_ptr<const QwenGpuModel> target_model, std::string* error_msg) {
  if (reader == nullptr || target_model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "MTP sidecar and target GPU models must not be null";
    }
    return nullptr;
  }
  const auto& target_weights = target_model->GetWeights();
  if (target_weights.token_embd.empty() || target_weights.output.empty() ||
      target_weights.output_norm.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "KAT MTP sidecar target lacks the tied head weights";
    }
    return nullptr;
  }
  // The sidecar's config, parsed by the same function the target used: the
  // nextn layer count is excluded from num_layers, so blk.<num_layers> is
  // the head block.
  std::string config_error;
  auto sidecar_config = reader->ExtractModelConfig(&config_error);
  if (!sidecar_config.has_value()) {
    if (error_msg != nullptr) {
      *error_msg = "KAT MTP sidecar config is unreadable: " + config_error;
    }
    return nullptr;
  }
  if (!core::IsModelFamily(sidecar_config->architecture,
                           core::ModelFamilyId::kQwen35Moe)) {
    if (error_msg != nullptr) {
      *error_msg = "KAT MTP sidecar must be a qwen35moe GGUF; got " +
                   sidecar_config->architecture;
    }
    return nullptr;
  }
  if (sidecar_config->mtp_num_layers != 1) {
    if (error_msg != nullptr) {
      *error_msg = "KAT MTP sidecar must carry exactly one nextn layer; got " +
                   std::to_string(sidecar_config->mtp_num_layers);
    }
    return nullptr;
  }
  const auto& cfg = *sidecar_config;
  // The draft executes the sidecar's block against the target's shapes, so
  // every dimension the block binds and routes against must match. Vocabulary
  // need not: the tied embedding and LM head are the target's.
  const auto differs = [&error_msg](const char* name, std::uint32_t sidecar,
                                    std::uint32_t target) {
    if (sidecar == target) {
      return false;
    }
    if (error_msg != nullptr) {
      *error_msg = std::string("KAT MTP sidecar ") + name +
                   " differs from the target: sidecar " +
                   std::to_string(sidecar) + ", target " +
                   std::to_string(target);
    }
    return true;
  };
  const auto& target_config = target_model->GetConfig();
  if (differs("hidden_size", cfg.hidden_size, target_config.hidden_size) ||
      differs("attention.head_count", cfg.num_attention_heads,
              target_config.num_attention_heads) ||
      differs("attention.head_count_kv", cfg.num_key_value_heads,
              target_config.num_key_value_heads) ||
      differs("attention.key_length", cfg.head_dim, target_config.head_dim) ||
      differs("expert_count", cfg.expert_count, target_config.expert_count) ||
      differs("expert_used_count", cfg.expert_used_count,
              target_config.expert_used_count) ||
      differs("expert_feed_forward_length", cfg.expert_feed_forward_length,
              target_config.expert_feed_forward_length) ||
      differs("intermediate_size", cfg.intermediate_size,
              target_config.intermediate_size) ||
      differs("expert_shared_feed_forward_length",
              cfg.expert_shared_feed_forward_length,
              target_config.expert_shared_feed_forward_length)) {
    return nullptr;
  }
  std::fprintf(stderr, "KAT-MTP sidecar vocab=%u\n", cfg.vocab_size);

  auto block = models::LoadMoeMtpBlock(*reader, cfg, cfg.num_layers, error_msg);
  if (!block.has_value()) {
    return nullptr;
  }
  if (!block->is_moe || block->nextn_eh_proj.empty() ||
      block->nextn_enorm.type != core::GgmlType::kF32 ||
      block->nextn_hnorm.type != core::GgmlType::kF32 ||
      block->nextn_shared_head_norm.type != core::GgmlType::kF32) {
    if (error_msg != nullptr) {
      *error_msg = "KAT MTP sidecar nextn block is incomplete";
    }
    return nullptr;
  }

  std::vector<void*> allocations;
  std::size_t packed_bytes = 0;
  const auto start = std::chrono::steady_clock::now();
  try {
    const auto copy = [&allocations,
                       &packed_bytes](models::QwenTensorRef& ref) {
      if (!ref.empty()) {
        ref = CopyTensorBytes(ref, allocations, packed_bytes);
      }
    };
    copy(block->attn_norm);
    copy(block->attn_q);
    copy(block->attn_k);
    copy(block->attn_v);
    copy(block->attn_output);
    copy(block->attn_q_norm);
    copy(block->attn_k_norm);
    copy(block->ffn_norm);
    copy(block->ffn_gate_inp);
    copy(block->ffn_gate_exps);
    copy(block->ffn_up_exps);
    copy(block->ffn_down_exps);
    copy(block->ffn_gate_inp_shexp);
    copy(block->ffn_gate_shexp);
    copy(block->ffn_up_shexp);
    copy(block->ffn_down_shexp);
    copy(block->nextn_eh_proj);
    copy(block->nextn_enorm);
    copy(block->nextn_hnorm);
    copy(block->nextn_shared_head_norm);
  } catch (const std::exception& exception) {
    ReleaseAllocations(allocations);
    if (error_msg != nullptr) {
      *error_msg = exception.what();
    }
    return nullptr;
  }

  speculative::QwenMtpWeights weights;
  std::vector<float> target_output_norm;
  if (!BuildBlockWeights(weights, *target_model, *block, &target_output_norm,
                         error_msg)) {
    ReleaseAllocations(allocations);
    return nullptr;
  }
  const double pack_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  std::fprintf(stderr,
               "KAT-MTP sidecar experts=%u topk=%u ff=%u eh_type=%d "
               "expert_type=%d bytes=%zu\n",
               weights.config.expert_count, weights.config.expert_used_count,
               weights.config.expert_feed_forward_length,
               static_cast<int>(weights.fusion_projection.type),
               static_cast<int>(weights.layer.ffn_gate_exps.type),
               packed_bytes);
  return std::shared_ptr<const QwenMtpGpuModel>(new QwenMtpGpuModel(
      std::move(reader), std::move(target_model), std::move(weights),
      std::move(allocations), packed_bytes, pack_seconds,
      std::move(target_output_norm)));
}

}  // namespace superroc::hip
#endif  // defined(ENGINE_ENABLE_HIP)
