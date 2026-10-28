#if defined(ENGINE_ENABLE_HIP)
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <limits>
#include <memory>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#include "src/models/llama/spm_tokenizer.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/fp4_launch.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace superroc::hip {
namespace {

void ReleaseWeightRegions(std::vector<QwenGpuWeightRegion>& regions) noexcept {
  for (auto& region : regions) {
    if (region.host_copy != nullptr) {
      (void)hipHostUnregister(region.host_copy);
      (void)munmap(region.host_copy, region.size);
    }
    region = {};
  }
}

void CopyMappedWeights(const core::GgufMappedRegion& source, void* copy) {
  constexpr std::size_t kChunkBytes = 16ULL << 20;
  const auto chunks =
      source.size / kChunkBytes + (source.size % kChunkBytes != 0);
  std::atomic<std::size_t> next{0};
  std::atomic<int> failure{0};
  auto copy_chunk = [&] {
    while (failure.load(std::memory_order_relaxed) == 0) {
      const auto index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= chunks)
        break;
      const auto offset = index * kChunkBytes;
      const auto bytes = std::min(kChunkBytes, source.size - offset);
      auto* input = const_cast<std::uint8_t*>(
                        static_cast<const std::uint8_t*>(source.data)) +
                    offset;
      if (madvise(input, bytes, MADV_POPULATE_READ) != 0) {
        failure.store(errno, std::memory_order_relaxed);
        break;
      }
      std::memcpy(static_cast<std::uint8_t*>(copy) + offset, input, bytes);
      // The immutable copy now owns these resident bytes. Keep the original
      // mapping valid for the reader without retaining its populated PTEs.
      (void)madvise(input, bytes, MADV_DONTNEED);
    }
  };
  {
    std::vector<std::jthread> workers;
    for (std::size_t i = 1; i < std::min<std::size_t>(16, chunks); ++i)
      workers.emplace_back(copy_chunk);
    copy_chunk();
  }
  if (const auto error = failure.load(); error != 0)
    throw std::system_error(error, std::generic_category(),
                            "cannot read mapped Qwen weights");
}

[[nodiscard]] hipError_t MapRegisteredRegion(
    const core::GgufMappedRegion& source, QwenGpuWeightRegion& destination) {
  void* host_copy = mmap(nullptr, source.size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (host_copy == MAP_FAILED)
    return hipErrorOutOfMemory;
  const auto unmap = [&](void* pointer) { (void)munmap(pointer, source.size); };
  std::unique_ptr<void, decltype(unmap)> owned_copy(host_copy, unmap);
  (void)madvise(host_copy, source.size, MADV_HUGEPAGE);
  CopyMappedWeights(source, host_copy);
  const auto register_error = hipHostRegister(
      host_copy, source.size, hipHostRegisterMapped | hipHostRegisterReadOnly);
  if (register_error != hipSuccess)
    return register_error;

  void* device_data = nullptr;
  const auto pointer_error =
      hipHostGetDevicePointer(&device_data, host_copy, 0);
  if (pointer_error != hipSuccess) {
    (void)hipHostUnregister(host_copy);
    return pointer_error;
  }

  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .host_copy = owned_copy.release(),
                 .size = source.size};
  return hipSuccess;
}

[[nodiscard]] bool CreateWeightRegions(
    const core::GgufReader& reader,
    std::vector<QwenGpuWeightRegion>& weight_regions, std::string* error_msg) {
  const auto source_regions = reader.GetMappedRegions();
  if (source_regions.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader has no mapped weight regions";
    }
    return false;
  }

  weight_regions.resize(source_regions.size());
  for (std::size_t i = 0; i < source_regions.size(); ++i) {
    const auto& source = source_regions[i];
    auto& destination = weight_regions[i];
    hipError_t map_error;
    try {
      map_error = MapRegisteredRegion(source, destination);
    } catch (const std::exception& e) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg)
        *error_msg = e.what();
      return false;
    }
    if (destination.device_data == nullptr) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg != nullptr) {
        *error_msg = "Failed to make GGUF shard " + std::to_string(i) +
                     " GPU-visible through mapped registration: " +
                     hipGetErrorString(map_error);
      }
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool RemapTensor(
    models::QwenTensorRef& tensor,
    std::span<const QwenGpuWeightRegion> regions) noexcept {
  if (tensor.empty()) {
    return true;
  }
  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor.data);
  for (const auto& region : regions) {
    const auto region_address =
        reinterpret_cast<std::uintptr_t>(region.host_data);
    if (tensor_address >= region_address) {
      const auto offset = tensor_address - region_address;
      const std::size_t encoded_bytes = tensor.EncodedSizeBytes();
      if (offset < region.size && encoded_bytes != 0 &&
          encoded_bytes <= region.size - offset) {
        tensor.data =
            static_cast<const std::uint8_t*>(region.device_data) + offset;
        tensor.available_bytes = region.size - offset;
        return true;
      }
    }
  }
  return false;
}

}  // namespace

QwenGpuModel::QwenGpuModel(
    std::shared_ptr<const core::GgufReader> reader,
    models::QwenModelWeights weights,
    std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
    std::vector<QwenGpuWeightRegion> weight_regions,
    std::shared_ptr<models::qwen::vision::Encoder> vision,
    std::shared_ptr<const models::llama::SpmTokenizer> llama_tokenizer)
    : reader_(std::move(reader)),
      weights_(std::move(weights)),
      tokenizer_(std::move(tokenizer)),
      llama_tokenizer_(std::move(llama_tokenizer)),
      weight_regions_(std::move(weight_regions)),
      vision_(std::move(vision)) {}

QwenGpuModel::~QwenGpuModel() {
  ReleaseWeightRegions(weight_regions_);
}

std::size_t QwenGpuModel::GetResidentBytes() const noexcept {
  std::size_t total = vision_ ? vision_->ResidentBytes() : 0;
  for (const auto& region : weight_regions_) {
    if (region.size > std::numeric_limits<std::size_t>::max() - total) {
      return std::numeric_limits<std::size_t>::max();
    }
    total += region.size;
  }
  return total;
}

std::shared_ptr<const QwenGpuModel> QwenGpuModel::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg,
    std::shared_ptr<models::qwen::vision::Encoder> vision) {
  if (reader == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader must not be null";
    }
    return nullptr;
  }

  auto weights_opt = models::QwenModelWeights::LoadFromGguf(*reader, error_msg);
  if (!weights_opt.has_value()) {
    return nullptr;
  }

  std::shared_ptr<const models::llama::SpmTokenizer> llama_tokenizer;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer;
  if (weights_opt->config.llama) {
    auto spm = models::llama::SpmTokenizer::FromGguf(*reader, error_msg);
    if (!spm.has_value()) {
      return nullptr;
    }
    llama_tokenizer =
        std::make_shared<const models::llama::SpmTokenizer>(std::move(*spm));
  } else {
    if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                              error_msg)) {
      return nullptr;
    }
    tokenizer = tokenization::QwenTokenizer::CreateFromGguf(*reader, error_msg);
    if (!tokenizer)
      return nullptr;
  }

  std::vector<QwenGpuWeightRegion> weight_regions;
  if (!CreateWeightRegions(*reader, weight_regions, error_msg)) {
    return nullptr;
  }

  const auto remap = [&](models::QwenTensorRef& tensor) {
    return RemapTensor(tensor, weight_regions);
  };
  bool remapped = remap(weights_opt->token_embd) &&
                  remap(weights_opt->output_norm) && remap(weights_opt->output);
  for (auto& layer : weights_opt->layers) {
    remapped =
        remapped && remap(layer.attn_norm) && remap(layer.attn_q) &&
        remap(layer.attn_k) && remap(layer.attn_v) &&
        remap(layer.attn_output) && remap(layer.attn_q_norm) &&
        remap(layer.attn_k_norm) && remap(layer.attn_qkv) &&
        remap(layer.attn_gate) && remap(layer.ssm_out) &&
        remap(layer.ssm_conv1d) && remap(layer.ssm_alpha) &&
        remap(layer.ssm_beta) && remap(layer.ssm_a) && remap(layer.ssm_dt) &&
        remap(layer.ssm_norm) && remap(layer.ffn_norm) &&
        remap(layer.ffn_gate) && remap(layer.ffn_up) && remap(layer.ffn_down) &&
        remap(layer.attn_post_norm) &&  // GLIMMER
        remap(layer.ffn_post_norm) && remap(layer.ffn_gate_inp) &&
        remap(layer.ffn_gate_exps) && remap(layer.ffn_up_exps) &&
        remap(layer.ffn_down_exps) && remap(layer.ffn_gate_inp_shexp) &&
        remap(layer.ffn_gate_shexp) && remap(layer.ffn_up_shexp) &&
        remap(layer.ffn_down_shexp) && remap(layer.nextn_eh_proj) &&
        remap(layer.nextn_enorm) && remap(layer.nextn_hnorm) &&
        remap(layer.nextn_shared_head_norm) &&  // KAT:
        remap(layer.shortconv_conv) &&
        remap(layer.shortconv_in_proj) &&                               // LFM2
        remap(layer.shortconv_out_proj) && remap(layer.exp_probs_b) &&  // LFM2
        remap(layer.attn_sinks);
  }
  // KAT: the MTP block is not in layers. Empty tensors remap as true.
  if (weights_opt->mtp.has_value()) {
    auto& layer = *weights_opt->mtp;
    remapped = remapped && remap(layer.attn_norm) && remap(layer.attn_q) &&
               remap(layer.attn_k) && remap(layer.attn_v) &&
               remap(layer.attn_output) && remap(layer.attn_q_norm) &&
               remap(layer.attn_k_norm) && remap(layer.ffn_norm) &&
               remap(layer.ffn_gate_inp) && remap(layer.ffn_gate_exps) &&
               remap(layer.ffn_up_exps) && remap(layer.ffn_down_exps) &&
               remap(layer.ffn_gate_inp_shexp) && remap(layer.ffn_gate_shexp) &&
               remap(layer.ffn_up_shexp) && remap(layer.ffn_down_shexp) &&
               remap(layer.nextn_eh_proj) && remap(layer.nextn_enorm) &&
               remap(layer.nextn_hnorm) && remap(layer.nextn_shared_head_norm);
  }
  if (!remapped) {
    ReleaseWeightRegions(weight_regions);
    if (error_msg != nullptr) {
      *error_msg = "A Qwen tensor does not belong to any mapped GGUF shard";
    }
    return nullptr;
  }

  std::shared_ptr<const tokenization::QwenTokenizer> shared_tokenizer(
      std::move(tokenizer));
  return std::make_shared<const QwenGpuModel>(
      std::move(reader), std::move(*weights_opt), std::move(shared_tokenizer),
      std::move(weight_regions), std::move(vision), std::move(llama_tokenizer));
}

namespace {

// lqw:rv27-fix1. True when the model actually carries a ROCmFP4 (type
// 100/101) tensor, so the capture warmup below only runs for FP4 seats. The
// walk lists every QwenLayerWeights member: a new tensor field means a new
// term here.
bool HasRocmFp4Tensor(const models::QwenTensorRef& tensor) noexcept {
  return !tensor.empty() && IsRocmFp4(tensor.type);
}

bool ModelHasRocmFp4Tensors(const models::QwenModelWeights& weights) {
  if (HasRocmFp4Tensor(weights.token_embd) ||
      HasRocmFp4Tensor(weights.output)) {
    return true;
  }
  for (const auto& layer : weights.layers) {
    if (HasRocmFp4Tensor(layer.attn_norm) || HasRocmFp4Tensor(layer.ffn_norm) ||
        HasRocmFp4Tensor(layer.attn_q) || HasRocmFp4Tensor(layer.attn_k) ||
        HasRocmFp4Tensor(layer.attn_v) || HasRocmFp4Tensor(layer.attn_output) ||
        HasRocmFp4Tensor(layer.attn_q_norm) ||
        HasRocmFp4Tensor(layer.attn_k_norm) ||
        HasRocmFp4Tensor(layer.attn_qkv) || HasRocmFp4Tensor(layer.attn_gate) ||
        HasRocmFp4Tensor(layer.ssm_a) || HasRocmFp4Tensor(layer.ssm_conv1d) ||
        HasRocmFp4Tensor(layer.ssm_dt) || HasRocmFp4Tensor(layer.ssm_alpha) ||
        HasRocmFp4Tensor(layer.ssm_beta) || HasRocmFp4Tensor(layer.ssm_norm) ||
        HasRocmFp4Tensor(layer.ssm_out) || HasRocmFp4Tensor(layer.ffn_gate) ||
        HasRocmFp4Tensor(layer.ffn_up) || HasRocmFp4Tensor(layer.ffn_down) ||
        HasRocmFp4Tensor(layer.ffn_gate_exps) ||
        HasRocmFp4Tensor(layer.ffn_up_exps) ||
        HasRocmFp4Tensor(layer.ffn_down_exps) ||
        HasRocmFp4Tensor(layer.ffn_gate_shexp) ||
        HasRocmFp4Tensor(layer.ffn_up_shexp) ||
        HasRocmFp4Tensor(layer.ffn_down_shexp)) {  // KAT:
      return true;
    }
  }
  // KAT: the dormant MTP block can carry type-101 experts too.
  if (weights.mtp.has_value()) {
    const auto& layer = *weights.mtp;
    if (HasRocmFp4Tensor(layer.ffn_gate_exps) ||
        HasRocmFp4Tensor(layer.ffn_up_exps) ||
        HasRocmFp4Tensor(layer.ffn_down_exps) ||
        HasRocmFp4Tensor(layer.ffn_gate_shexp) ||
        HasRocmFp4Tensor(layer.ffn_up_shexp) ||
        HasRocmFp4Tensor(layer.ffn_down_shexp) ||
        HasRocmFp4Tensor(layer.attn_q) || HasRocmFp4Tensor(layer.attn_qkv)) {
      return true;
    }
  }
  return false;
}

}  // namespace

// SQ19-2: load-time attach for one executor stream, called from the
// QwenGpuExecutor constructor (raw-constructed executors never pass through
// Create). Sizes and failure handling mirror Create's pre-warm below: every
// buffer of the attached set is allocated and warmed here, outside any
// capture, so the first step can neither hipMalloc nor decline to a
// different arithmetic than the executor that owns set 0. The warm calls run
// unconditionally (each is a cheap no-op once its sizes are covered) because
// a reused pre-sized set may never have been warmed: configure() grows the
// table without warming, and only Create pre-warms, so a raw-constructed
// executor reusing a pre-sized set would otherwise hipMalloc mid-capture.
void AttachAndWarmRocmFp4ExecutorSet(hipStream_t stream,
                                     const models::QwenModelWeights& weights,
                                     const core::ModelConfig& config) {
  if (!ModelHasRocmFp4Tensors(weights)) {
    return;
  }
  const std::size_t wide =
      std::max<std::size_t>({config.hidden_size, config.intermediate_size,
                             config.ssm_inner_size, config.AttentionSize()});
  const std::size_t wk = (wide + 31) / 32 * 32;
  (void)superroc_fp4_slot_stream_attach_grow(stream, __func__, 0);
  (void)WarmupRocmFp4Slots(wk, stream);
  const std::size_t hs_elems =
      std::max(config.expert_used_count * config.expert_feed_forward_length,
               config.expert_shared_feed_forward_length);
  const std::size_t hs_m_elems =
      4 * std::max(config.expert_used_count * config.expert_feed_forward_length,
                   config.expert_shared_feed_forward_length);
  const std::size_t m_total =
      std::max<std::size_t>({4 * wk, (hs_m_elems + 31) / 32 * 32});
  if (superroc_fp4_slots_warm_moe(
          static_cast<int>(hs_elems), static_cast<int>(hs_m_elems),
          static_cast<int>(m_total)) != SUPERROC_FP4_OK) {
    DiscardRocmFp4Warmup();
  }
  if (!RocmFp4SwigluScratchWarm(config.intermediate_size)) {
    DiscardRocmFp4Warmup();
  }
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::Create(
    std::shared_ptr<const QwenGpuModel> model, std::string* error_msg,
    std::uint32_t max_context, QwenExecutionPolicy policy) {
  if (model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "Qwen GPU model must not be null";
    }
    return nullptr;
  }

  const auto& config = model->GetConfig();
  // LLAMA (F4): apply the llama policy centrally here so every creator
  // (serve, prompt, bench) gets it, not only the serve runner.
  if (config.llama) {
    ApplyLlamaExecutionPolicy(policy);
  }
  const std::uint32_t model_context =
      config.context_length > 0 ? config.context_length : max_context;
  if (max_context == 0 || max_context > model_context) {
    if (error_msg != nullptr) {
      *error_msg = "Requested GPU context exceeds the model context length";
    }
    return nullptr;
  }

  // lqw:capture-warmup. Pre-grow the copied Ling FP4 GEMV slots and the fused
  // SwiGLU scratch before the first decode graph capture: both allocate
  // lazily, and an allocation inside an active capture poisons it.
  // rv27 fix 1: only a model that actually carries a type-100/101 tensor pays
  // this, so stock seats (IQ4_XS, Q4_K_XL, Q8_0, BF16) spend zero extra load
  // time and pin no extra device memory. rv27 fix 2: if the load-time scratch
  // grow fails, drop the slot warm mark as well - every FP4 intercept then
  // declines and the A4 sub16 fallback serves the step, so a captured step
  // can never reach a lazy allocation.
  if (ModelHasRocmFp4Tensors(model->GetWeights())) {
    const std::size_t wide =
        std::max<std::size_t>({config.hidden_size, config.intermediate_size,
                               config.ssm_inner_size, config.AttentionSize()});
    const std::size_t wk = (wide + 31) / 32 * 32;
    (void)WarmupRocmFp4Slots(wk, nullptr);
    // SQ10: size every set's MoE scratch from the config maxima (the
    // single-token hs covers the routed and shared experts; the m-pair covers
    // 4-token input quants and hs quants), and the SwiGLU pair for every set.
    const std::size_t hs_elems =
        std::max(config.expert_used_count * config.expert_feed_forward_length,
                 config.expert_shared_feed_forward_length);
    // SPEC2 (R5): the small-m scratch serves the shared expert too
    // (expert_ffn_shared_m grows moe_hs_m to n_tokens * shared_ff, n <= 4),
    // so cover it exactly like the single-token group above.
    const std::size_t hs_m_elems =
        4 *
        std::max(config.expert_used_count * config.expert_feed_forward_length,
                 config.expert_shared_feed_forward_length);
    const std::size_t m_total =
        std::max<std::size_t>({4 * wk, (hs_m_elems + 31) / 32 * 32});
    if (superroc_fp4_slots_warm_moe(
            static_cast<int>(hs_elems), static_cast<int>(hs_m_elems),
            static_cast<int>(m_total)) != SUPERROC_FP4_OK) {
      DiscardRocmFp4Warmup();
    }
    if (!RocmFp4SwigluScratchWarm(config.intermediate_size)) {
      DiscardRocmFp4Warmup();
    }
  }
  auto executor =
      std::make_unique<QwenGpuExecutor>(std::move(model), max_context, policy);
  // SQ10: this executor's stream owns one slot set for the session's life
  // (detached in ~QwenGpuExecutor, R3). SQ19-2: the load-time entry grows
  // the table when no set is free (a second backend, or a reload with
  // another session count), so this never refuses below the cap; for an
  // FP4 model the constructor already attached this stream and this only
  // re-resolves it, while a stock model (no FP4 tensors, ctor skipped)
  // still owns its set without paying any warmup.
  (void)superroc_fp4_slot_stream_attach_grow(executor->arena_.stream, __func__,
                                             0);
  return executor;
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg,
    std::uint32_t max_context, QwenExecutionPolicy policy) {
  auto model = QwenGpuModel::CreateFromGguf(std::move(reader), error_msg);
  if (model == nullptr) {
    return nullptr;
  }
  return Create(std::move(model), error_msg, max_context, policy);
}

}  // namespace superroc::hip
#endif  // defined(ENGINE_ENABLE_HIP)
