#include "src/cli/serve/inference_backend.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

#include "src/cli/serve/component_text_cache.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/gguf_identity.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/core/model_family.hpp"
#include "src/core/sampling.hpp"
#include "src/models/flint/flint_chat.hpp"
#include "src/models/glimmer/glimmer_chat.hpp"
#include "src/models/lfm2/lfm2_chat.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/control_tokens.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/vision/image_cache.hpp"

// Ling-3.0-flash (bailingmoe3). Outside the HIP guard on purpose: the package's
// CPU reference backend is a complete implementation, and the bailingmoe3
// branch below is reached in CPU-only builds too, so these types must be
// visible there.
#include "src/cli/serve/ling3_text_runner.hpp"
#include "src/models/ling3/engine.hpp"
#include "src/models/ling3/tokenizer.hpp"
#include "src/models/ling3/vision_tower.hpp"
#include "src/models/ling3/weights.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/core/canonical.hpp"
#include "src/core/env.hpp"
#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/heap_placement.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/host_allocations.hpp"
#include "src/core/rocm_fp4/superroc_fp4_gemv_q8.hpp"  // Slot sets.
#include "src/core/rocm_fp4/superroc_fp4_quant.hpp"    // codes.
#include "src/core/speculative/lookup_draft_backend.hpp"
#include "src/core/speculative/speculative_verifier.hpp"
#include "src/models/deepseek_v4_flash/dspark_sampler.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/llama/spm_tokenizer.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen/hip/mtp/kat_mtp_pmin.hpp"  // KAT-MTP:
#include "src/models/qwen/hip/mtp/snapshot_guard.hpp"  // F7: draft/target position check.
#include "src/models/qwen/moe/dispatch.hpp"            // KAT:
#include "src/models/qwen/state.hpp"                   // KAT:
#include "src/models/qwen38_flash_next/continuation_adapter.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#endif

namespace superroc::server {

// A selected token ends the turn only when the request admits EOS and the token
// is one of the model's stop tokens. The engine stops there itself and the stop
// is reported by the engine, so this stays per-request state.
template<typename State, typename Tokens>
bool EndsTurn(const State& state, const Tokens& tokens, int token) {
  return state.stop_at_eos() && tokens.IsStopToken(token);
}

namespace {

using Clock = std::chrono::steady_clock;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

#if defined(ENGINE_ENABLE_HIP)
// The FP4 slot refusal counter in the server log, at load
// end and at shutdown. A refusal falls back to the eager kernels (a valid
// result from the same weights), so the counter is the operator's only sign
// that a set was under-provisioned or a registration dropped.
void LogFp4SlotRefusals(const char* phase) {
  std::fprintf(stderr, "SUPERROC_FP4_SLOTS phase=%s refusals=%lu\n", phase,
               superroc_fp4_slot_refuse_count());
}
#endif

#if defined(ENGINE_ENABLE_HIP)

// KAT-MTP: /health summary of the speculative draft configuration. The MTP
// draft head reports lossless sampled proposals, so sampled
// (temperature > 0), penalized and constrained requests keep drafting;
// /health states that instead of leaving it silent.
[[nodiscard]] std::string SpeculativeStatus(const TextSpeculativeConfig& config,
                                            bool moe_block) {
  if (config.backend == TextSpeculativeBackend::kMtp) {
    char buffer[96];
    std::snprintf(buffer, sizeof(buffer),
                  "mtp depth=%u p_min=%.2f sampled_requests=draft",
                  config.max_draft_tokens,
                  static_cast<double>(hip::MtpEffectivePMin(moe_block)));
    return buffer;
  }
  if (config.backend == TextSpeculativeBackend::kDFlash)
    return "dflash2";
  if (config.backend == TextSpeculativeBackend::kDSpark)
    return "dspark";
  return {};
}

struct QwenImageContext final : TextPromptContext {
  std::shared_ptr<const models::qwen::vision::Prompt> prompt;
};

tokenization::ChatTemplateOptions QwenChatOptions(const ChatRequest& request,
                                                  std::uint32_t max_context) {
  auto options = tokenization::ResolveQwenChatOptions(request.reasoning,
                                                      request.add_vision_id);
  options.require_tool_call =
      request.tool_choice == ChatRequest::ToolChoice::kRequired;
  // The rendered prompt may be as long as the context can hold.
  options.max_output_bytes =
      tokenization::RenderedPromptBoundBytes(max_context);
  return options;
}

std::optional<std::size_t> CountQwenPromptTokens(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context, models::qwen::vision::ImageCache& image_cache) {
  const auto options = QwenChatOptions(request, max_context);
  const auto tools =
      request.tool_choice == ChatRequest::ToolChoice::kNone
          ? std::span<const tokenization::ChatTool>{}
          : std::span<const tokenization::ChatTool>{request.tools};
  if (std::ranges::none_of(request.messages,
                           [](const auto& m) { return !m.images.empty(); })) {
    const auto tokens = tokenization::QwenChatTemplate::RenderAndTokenize(
        tokenizer, request.messages, tools, options);
    if (!tokens)
      return std::nullopt;
    return tokens->size();
  }
  // Reuse generation's CPU image preparation, including resize, token
  // expansion and projector validation. Counting does not encode images on
  // the GPU or admit the prompt to an execution context. Keep only the shared
  // layout's signed position limit so oversized prompts can still be sized.
  return models::qwen::vision::Prepare(
             tokenizer, request.messages, tools, options,
             encoder ? encoder->identity() : std::string_view{},
             std::numeric_limits<std::int32_t>::max(),
             request.cache_prompt ? &image_cache : nullptr)
      .tokens.size();
}

TextPreparedPrompt PrepareQwenPrompt(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context, models::qwen::vision::ImageCache& image_cache) {
  const bool has_images = std::ranges::any_of(
      request.messages, [](const auto& m) { return !m.images.empty(); });
  const auto options = QwenChatOptions(request, max_context);
  auto prompt = std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          tokenizer, request.messages,
          request.tool_choice == ChatRequest::ToolChoice::kNone
              ? std::span<const tokenization::ChatTool>{}
              : std::span<const tokenization::ChatTool>{request.tools},
          options,
          encoder && has_images ? encoder->identity() : std::string_view{},
          max_context, request.cache_prompt ? &image_cache : nullptr));
  if (has_images && Logger::Enabled(LogLevel::kDebug)) {
    const auto stats = image_cache.GetStats();
    Logger::LogFormatted(LogLevel::kDebug, "vision",
                         "event=image_cache enabled=%d entries=%zu "
                         "pixel_bytes=%zu encoded_bytes=%zu "
                         "hits=%zu misses=%zu evictions=%zu",
                         static_cast<int>(request.cache_prompt), stats.entries,
                         stats.pixel_bytes, stats.encoded_bytes, stats.hits,
                         stats.misses, stats.evictions);
  }
  const auto cache_prefix = prompt->stable_prefix_tokens;
  if (prompt->images.empty())
    return {std::move(prompt->tokens), {}, cache_prefix};
  if (!encoder)
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  auto context = std::make_shared<QwenImageContext>();
  context->cache_identity = prompt->cache_identity;
  for (const auto& image : prompt->images) {
    const auto identity = prompt->IdentityForPrefix(image.grid.offset);
    context->cache_prefixes.push_back(
        {image.grid.offset, {identity.begin(), identity.end()}});
  }
  context->prompt = prompt;
  return {prompt->tokens, std::move(context), cache_prefix};
}

std::shared_ptr<const models::qwen::vision::Prompt> QwenPrompt(
    const std::shared_ptr<const TextPromptContext>& context) {
  if (!context)
    return {};
  const auto* image = dynamic_cast<const QwenImageContext*>(context.get());
  if (!image)
    throw std::invalid_argument("invalid Qwen prompt context");
  return image->prompt;
}

// Hash the sidecar and upload its weights while the model loads, before the
// runner claims state capacity and the RAM cache is sized. Otherwise the first
// image request hashes the whole sidecar in its queue time and uploads the
// weights in its prefill, after admission and the cache were sized without
// them.
void PrepareVision(
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder) {
  if (!encoder)
    return;
  const auto started = Clock::now();
  (void)encoder->identity();
  const auto hashed = Clock::now();
  try {
    encoder->Upload();
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("could not upload the Qwen vision sidecar: ") + e.what());
  }
  const auto uploaded = Clock::now();
  std::size_t heap_free = 0, heap_total = 0;
  (void)hipMemGetInfo(&heap_free, &heap_total);
  using Milliseconds = std::chrono::duration<double, std::milli>;
  Logger::Info("loader",
               "event=vision_prepared resident_bytes=" +
                   std::to_string(encoder->ResidentBytes()) + " identity_ms=" +
                   std::to_string(Milliseconds(hashed - started).count()) +
                   " upload_ms=" +
                   std::to_string(Milliseconds(uploaded - hashed).count()) +
                   " device_heap_free_bytes=" + std::to_string(heap_free) +
                   " " + Logger::MemoryStatus());
}

constexpr std::string_view kDeepSeekStateAbi =
    "deepseek-v4-flash-gfx1151-state-v4";
constexpr std::array<std::uint8_t, 8> kQwenPersistentSnapshotMagic = {
    'G', 'Q', 'W', 'R', 'U', 'N', '0', '1'};
constexpr std::uint32_t kQwenPersistentPayloadVersion = 3;
constexpr std::size_t kQwenMaxResumeTokens = 9;
constexpr std::size_t kQwenPersistentSnapshotHeaderBytes = 112;
constexpr std::uint32_t kQwenPersistentSpeculativeFlag = 1U << 0U;
constexpr std::uint32_t kQwenPersistentPendingTokenFlag = 1U << 1U;

constexpr std::string_view QwenStateAbi(bool speculative,
                                        bool fp16_attention_kv,
                                        bool bf16_recurrent_state) noexcept {
  if (bf16_recurrent_state) {
    if (speculative) {
      return fp16_attention_kv
                 ? "qwen-gfx1151-dflash-state-v4-fp16-kv-bf16-recurrent"
                 : "qwen-gfx1151-dflash-state-v4-fp32-kv-bf16-recurrent";
    }
    return fp16_attention_kv ? "qwen-gfx1151-state-v3-fp16-kv-bf16-recurrent"
                             : "qwen-gfx1151-state-v3-fp32-kv-bf16-recurrent";
  }
  if (speculative) {
    return fp16_attention_kv ? "qwen-gfx1151-dflash-state-v3-fp16-kv"
                             : "qwen-gfx1151-dflash-state-v3-fp32-kv";
  }
  return fp16_attention_kv ? "qwen-gfx1151-state-v2-fp16-kv"
                           : "qwen-gfx1151-state-v2-fp32-kv";
}

// LFM2: a DSpark session carries draft state a plain session cannot
// resume, so its state ABI names the drafter exactly as QwenStateAbi does.
constexpr std::string_view Lfm2StateAbi(bool speculative,
                                        bool fp16_attention_kv) noexcept {
  if (speculative) {
    return fp16_attention_kv ? "lfm2-gfx1151-dflash-state-v2-fp16-kv-conv"
                             : "lfm2-gfx1151-dflash-state-v1-fp32-kv-conv";
  }
  return fp16_attention_kv ? "lfm2-gfx1151-state-v2-fp16-kv-conv"
                           : "lfm2-gfx1151-state-v1-fp32-kv-conv";
}

constexpr std::string_view FlintStateAbi(bool glimmer_fp16_kv) noexcept {
  return glimmer_fp16_kv ? "flint-gfx1151-state-v1-fp32-kv-f16plane"
                         : "flint-gfx1151-state-v1-fp32-kv";
}

bool DiskCacheEnabled(const TextDiskCacheConfig& config) noexcept {
  return !config.directory.empty();
}

bool IsSha256Hex(std::string_view value) noexcept {
  return value.size() == 64 && std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::vector<std::uint8_t> DeepSeekCompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view support_fingerprint,
    std::uint32_t max_context, std::uint32_t max_draft_tokens) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "DeepSeek disk cache requires an artifact fingerprint");
  }
  std::ostringstream identity;
  identity << "schema=superroc-text-continuation-v2\n"
           << "model_kind=deepseek4\n"
           << "artifact_id=" << core::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=joyai-byte-bpe-v1\n"
           << "chat_template=" << models::deepseek_v4_flash::ChatTemplateId()
           << '\n'
           << "chat_template_reference_sha256="
           << models::deepseek_v4_flash::EncoderReferenceSha256() << '\n'
           << "state_abi=" << kDeepSeekStateAbi << '\n'
           << "payload_layout=ds4-rocm-v4-window128-continuation\n"
           // Cached frontiers also depend on the compiled numerical routes.
           << "numerics=ds4-scalar-verifier-v1\n"
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "rope_window_policy=deepseek4-compiled-v1\n"
           << "adapters=none\n";
  identity << "support_id=" << core::kGgufIdentityScheme << ':'
           << support_fingerprint << '\n'
           << "max_draft_tokens=" << max_draft_tokens << '\n'
           << "draft_policy=dspark-cost-v5-window128\n";
  // A cache written with the canonical cache holds other checkpoints, and
  // prompt rows from other kernel routes, than one written without it.
  if (::superroc::core::CanonicalCacheOn())
    identity << "prompt_cache=canonical-v1\n";
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

std::vector<std::uint8_t> QwenFlashNextCompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view mtp_fingerprint,
    bool has_mtp, std::uint32_t max_context, std::uint32_t max_draft_tokens,
    std::uint32_t decode_concurrency) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen3.8-Flash-Next disk cache requires an artifact fingerprint");
  }
  if (has_mtp && !IsSha256Hex(mtp_fingerprint)) {
    throw std::invalid_argument(
        "Qwen3.8-Flash-Next MTP disk cache requires a draft artifact "
        "fingerprint");
  }
  std::ostringstream identity;
  identity << "schema=superroc-text-continuation-v2\n"
           << "model_kind=qwen38-flash-next\n"
           << "artifact_id=" << core::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=embedded-in-artifact\n"
           << "chat_template=qwen38-reasoning-compiled-v3\n"
           << "chat_template_reference_sha256="
           << tokenization::QwenChatTemplate::OfficialTemplateSha256() << '\n'
           << "state_abi=qwen38-flash-next-rocm-session-v1\n"
           << "payload_layout=qfn-rocm-session-snapshot-v"
           << models::qwen38_flash_next::Session::kSnapshotPayloadVersion
           << '\n'
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "adapters=none\n";
  if (has_mtp) {
    identity << "draft_backend=qfn-mtp-v1\n"
             << "draft_artifact_id=" << core::kGgufIdentityScheme << ':'
             << mtp_fingerprint << '\n'
             << "draft_max_tokens=" << max_draft_tokens << '\n'
             << "draft_cost_concurrency=" << decode_concurrency << '\n';
  }
  // A cache written with the canonical cache holds other checkpoints, and
  // prompt rows from other kernel routes, than one written without it.
  if (::superroc::core::CanonicalCacheOn())
    identity << "prompt_cache=canonical-v1\n";
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

std::vector<std::uint8_t> QwenCompatibilityIdentity(
    std::string_view artifact_fingerprint,
    std::string_view draft_artifact_fingerprint, std::uint32_t max_context,
    const hip::QwenExecutionPolicy& execution_policy, bool speculative,
    const speculative::SpeculativeOptions& speculative_options, bool lfm2,
    bool flint, bool mtp) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen disk cache requires an artifact fingerprint");
  }
  // Prompt lookup with no drafter has no draft artifact to name and no draft
  // state to persist.
  const bool lookup_only = speculative && !mtp &&
                           speculative_options.prompt_lookup &&
                           draft_artifact_fingerprint.empty();
  if (speculative && !lookup_only && !IsSha256Hex(draft_artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen DFlash disk cache requires a draft artifact fingerprint");
  }
  // LFM2: state ABI follows the KV storage, selected by the caller arch test.
  const std::string_view lfm2_state_abi =
      Lfm2StateAbi(speculative, execution_policy.UsesFp16AttentionKv());
  const std::string_view state_abi =
      flint  ? FlintStateAbi(execution_policy.glimmer_fp16_kv)
      : lfm2 ? lfm2_state_abi
             : QwenStateAbi(speculative, execution_policy.UsesFp16AttentionKv(),
                            execution_policy.UsesBf16RecurrentState());
  std::ostringstream identity;
  identity
      << "schema=superroc-text-continuation-v2\n"
      << "model_kind="
      << (flint  ? std::string_view("flint")
          : lfm2 ? superroc::core::ModelFamilyArch(
                       superroc::core::ModelFamilyId::kLfm2Moe)
                 : std::string_view("qwen3.8"))
      << "\n"
      << "artifact_id=" << core::kGgufIdentityScheme << ':'
      << artifact_fingerprint << '\n'
      << "tokenizer=embedded-in-artifact\n"
      << "chat_template="
      << (flint  ? "flint-roles-v1"
          : lfm2 ? "lfm2-chatml-think-v1"
                 : "qwen38-reasoning-compiled-v3")
      << "\n"
      << "chat_template_reference_sha256="
      << (flint  ? tokenization::kFlintChatTemplateSha256
          : lfm2 ? tokenization::QwenChatTemplate::Lfm2TemplateSha256()
                 : tokenization::QwenChatTemplate::OfficialTemplateSha256())
      << '\n'
      << "state_abi=" << state_abi << '\n'
      << "payload_layout="
      << (lfm2 ? "lfm2-gfx1151-live-prefix-v1" : "qwen-gfx1151-live-prefix-v3")
      << "\n"
      // Older learned boundaries could keep logits from an earlier
      // prefix. Their payload has no readiness tag to distinguish them.
      << "checkpoint_frontier=complete-prefix-v1\n"
      << "numerics=" << (lfm2 ? "lfm2-fp32-v1" : "qwen-bf16-fp32-prefill-v1")
      << "\n"
      << "rmsnorm=fused-square-sum-v1\n"
      << "prefill_attention=visible-causal-tail-v1\n"
      << "attention_split_min_context="
      << hip::detail::kSplitKDecodeAttentionMinContext << '\n'
      << "attention_split_count="
      << hip::detail::kSplitKDecodeAttentionMaxSplits << '\n'
      << "kv_storage="
      << (execution_policy.UsesFp16AttentionKv() ? "fp16" : "fp32") << '\n'
      << "recurrent_storage="
      << (execution_policy.UsesBf16RecurrentState() ? "bf16" : "fp32") << '\n'
      << "context_tokens=" << max_context << '\n'
      << "position_policy=absolute-v1\n"
      << "rope_window_policy=qwen-gguf-config-v1\n"
      << "adapters=none\n";
  if (speculative && mtp) {
    identity << "draft_backend=mtp-gfx1151-v1\n"
             << "draft_artifact_id=" << core::kGgufIdentityScheme << ':'
             << draft_artifact_fingerprint << '\n'
             << "draft_state_layout=mtp-kv-prefix-v1\n"
             << "draft_max_tokens=" << speculative_options.max_draft_tokens
             << '\n'
             << "draft_min_tokens=" << speculative_options.min_draft_tokens
             << '\n';
  }
  if (lookup_only) {
    identity << "draft_backend=prompt-lookup-v1\n"
             << "draft_state_layout=none\n";
  }
  if (speculative && !mtp && !lookup_only) {
    identity
        << "draft_backend=dflash2-gfx1151-v1\n"
        << "draft_artifact_id=" << core::kGgufIdentityScheme << ':'
        << draft_artifact_fingerprint << '\n'
        << "draft_state_layout=dflash-window-kv-and-frontier-v4\n"
        << "draft_policy="
        << speculative::DFlashDraftPolicyName(speculative_options.dflash_policy)
        << "-v3\n"
        << "draft_max_tokens=" << speculative_options.max_draft_tokens << '\n'
        << "draft_min_tokens=" << speculative_options.min_draft_tokens << '\n'
        << "draft_initial_tokens=" << speculative_options.initial_draft_tokens
        << '\n'
        << "draft_rolling_window=" << speculative_options.rolling_window << '\n'
        << std::setprecision(std::numeric_limits<float>::max_digits10)
        << "draft_target_acceptance="
        << speculative_options.target_acceptance_rate << '\n'
        << "draft_adaptive="
        << (speculative_options.enable_adaptive_draft_length ? "true" : "false")
        << '\n'
        << "draft_batched_verification="
        << (speculative_options.use_batched_verification ? "true" : "false")
        << " prompt_lookup="
        << (speculative_options.prompt_lookup ? "true" : "false") << '\n'
        << "draft_target_verification=decode-equivalent-v1\n";
  }
  // A cache written with the canonical cache holds other checkpoints, and
  // prompt rows from other kernel routes, than one written without it.
  if (::superroc::core::CanonicalCacheOn())
    identity << "prompt_cache=canonical-v1\n";
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

template<typename T>
  requires(std::is_unsigned_v<T>)
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     T value) {
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    throw std::length_error("Qwen persistent snapshot header is truncated");
  }
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename T>
  requires(std::is_unsigned_v<T>)
T GetLittleEndian(std::span<const std::uint8_t> source, std::size_t offset) {
  if (offset > source.size() || sizeof(T) > source.size() - offset) {
    throw std::invalid_argument("Qwen persistent snapshot header is truncated");
  }
  T value = 0;
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    value |= static_cast<T>(source[offset + byte]) << (byte * 8U);
  }
  return value;
}

std::size_t CheckedPersistentAdd(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("Qwen persistent snapshot size overflows");
  }
  return left + right;
}

std::size_t PersistentSizeFromU64(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("Qwen persistent snapshot size overflows");
  }
  return static_cast<std::size_t>(value);
}

struct QwenRunnerStateHooks {
  virtual ~QwenRunnerStateHooks() = default;
  virtual void ArmTargetLogitsHook(const std::function<void()>* hook) = 0;
};

class QwenTextRunnerState final : public TextRunnerState,
                                  public QwenRunnerStateHooks {
public:
  QwenTextRunnerState(
      std::shared_ptr<const hip::QwenGpuModel> model, std::uint32_t max_context,
      std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model,
      std::shared_ptr<const hip::QwenMtpGpuModel> mtp_model,
      speculative::SpeculativeOptions speculative_options,
      hip::QwenExecutionPolicy execution_policy)
      : model_(std::move(model)) {
    resume_tokens_.reserve(kQwenMaxResumeTokens);
    rollback_tokens_.reserve(kQwenMaxResumeTokens);
    std::string error;
    executor_ = hip::QwenGpuExecutor::Create(model_, &error, max_context,
                                             execution_policy);
    if (executor_ == nullptr) {
      throw std::runtime_error("Failed to create GPU session: " + error);
    }
    const auto wrap_lookup =
        [&](std::unique_ptr<speculative::IDraftBackend> inner)
        -> std::unique_ptr<speculative::IDraftBackend> {
      if (!speculative_options.prompt_lookup)
        return inner;
      return std::make_unique<speculative::LookupDraftBackend>(
          std::move(inner));
    };
    if (dflash_model != nullptr && mtp_model != nullptr) {
      throw std::invalid_argument(
          "Qwen sessions accept one speculative draft backend");
    }
    if (dflash_model != nullptr) {
      auto draft_backend = hip::QwenDFlashGpuDraftBackend::Create(
          std::move(dflash_model),
          hip::QwenDFlashGpuDraftConfig{
              .max_context = max_context,
              .max_draft_tokens = speculative_options.max_draft_tokens,
              .policy = speculative_options.dflash_policy,
          },
          &error);
      if (draft_backend == nullptr) {
        throw std::runtime_error("Failed to create DFlash session: " + error);
      }
      draft_backend_ = draft_backend.get();
      verifier_ = std::make_unique<speculative::SpeculativeVerifier>(
          *executor_, wrap_lookup(std::move(draft_backend)),
          speculative_options);
    } else if (mtp_model != nullptr) {
      auto draft_backend = hip::QwenMtpGpuDraftBackend::Create(
          std::move(mtp_model),
          hip::QwenMtpGpuDraftConfig{
              .max_context = max_context,
              .max_draft_tokens = speculative_options.max_draft_tokens,
              // The MTP draft consumes the target's vision embeddings, exactly
              // as superroc prompt does (prompt.cpp:758). Without this an image
              // prompt would draft against a null vision input.
              .vision_input = &executor_->VisionInput(),
          },
          &error);
      if (draft_backend == nullptr) {
        throw std::runtime_error("Failed to create MTP session: " + error);
      }
      // The drafter's stream uses this session's slot
      // set (the host alternates the two streams, so they never collide).
      // The MTP dtor removes the alias.
      if (superroc_fp4_slot_stream_alias(draft_backend->DraftStream(),
                                         executor_->ArenaStream()) < 0) {
        throw std::runtime_error("Failed to alias MTP draft stream");
      }
      mtp_backend_ = draft_backend.get();
      verifier_ = std::make_unique<speculative::SpeculativeVerifier>(
          *executor_, wrap_lookup(std::move(draft_backend)),
          speculative_options);
    } else if (speculative_options.prompt_lookup) {
      verifier_ = std::make_unique<speculative::SpeculativeVerifier>(
          *executor_, wrap_lookup(nullptr), speculative_options);
    }
  }

  void SetCancellationCheck(const CancellationCheck& check) override {
    executor_->SetCancellationCheck(check);
  }

  void Invalidate() noexcept override {
    if (verifier_ != nullptr) {
      verifier_->Reset();
    }
    executor_->Reset();
    sequence_.clear();
    position_ = 0;
    frontier_.reset();
    frontier_logits_.clear();
    frontier_published_ = false;
    resume_tokens_.clear();
    rollback_position_.reset();
    rollback_tokens_.clear();
    rollback_logits_.clear();
  }

  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    try {
      auto usage = executor_->GetMemoryUsage();
      if (draft_backend_ != nullptr) {
        const auto draft = draft_backend_->GetMemoryUsage();
        usage.request_state_bytes += draft.request_state_bytes;
        usage.temporary_scratch_bytes += draft.temporary_scratch_bytes;
      }
      if (mtp_backend_ != nullptr) {
        const auto draft = mtp_backend_->GetMemoryUsage();
        usage.request_state_bytes += draft.request_state_bytes;
        usage.temporary_scratch_bytes += draft.temporary_scratch_bytes;
      }
      return {
          .per_request_state_bytes = usage.request_state_bytes,
          .temporary_scratch_bytes = usage.temporary_scratch_bytes,
      };
    } catch (...) {
      return {};
    }
  }

  [[nodiscard]] bool speculative() const noexcept {
    return verifier_ != nullptr;
  }

  /// The prompt's first token can be published when the target's logits are
  /// readable: the drafter's prompt pass that follows does not change it.
  struct TargetLogitsHook {
    QwenRunnerStateHooks& state;
    TargetLogitsHook(QwenRunnerStateHooks& s, const std::function<void()>* hook)
        : state(s) {
      state.ArmTargetLogitsHook(hook);
    }
    ~TargetLogitsHook() { state.ArmTargetLogitsHook(nullptr); }
  };

  void ArmTargetLogitsHook(const std::function<void()>* hook) override {
    if (hook == nullptr || !*hook) {
      verifier_->SetTargetLogitsHook({});
      return;
    }
    verifier_->SetTargetLogitsHook([this, hook](tokenization::TokenId first) {
      frontier_ = first;
      const auto logits = verifier_->CopyLastTargetLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
      (*hook)();
    });
  }

  void PrimeSpeculative(std::span<const TextRunnerToken> prompt,
                        bool decode_ready, std::uint32_t total_prompt_tokens,
                        const std::function<void()>* logits_hook = nullptr) {
    if (verifier_ == nullptr) {
      throw std::logic_error("Qwen state has no speculative verifier");
    }
    // Cleared first: the hook leaves the first token pending during the pass.
    resume_tokens_.clear();
    const TargetLogitsHook armed{*this, decode_ready ? logits_hook : nullptr};
    frontier_ = verifier_->Prime(prompt, decode_ready, total_prompt_tokens);
    frontier_logits_.clear();
    if (decode_ready) {
      const auto logits = verifier_->CopyLastTargetLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
    }
    sequence_.assign(prompt.begin(), prompt.end());
    position_ = prompt.size();
    frontier_published_ = false;
    rollback_position_.reset();
  }

  /// Bytes of the snapshot PrefillThroughCheckpoint will build at `boundary`,
  /// when this state can hold its cache boundary inside one prompt pass.
  [[nodiscard]] std::optional<std::size_t> PromptCheckpointBytes(
      std::uint32_t boundary) const {
    if (!executor_->CanCheckpointPromptInPass()) {
      return std::nullopt;
    }
    // A state restored without its frontier token (a disk checkpoint), or
    // whose verifier history was not rebuilt, cannot run the rest of the
    // prompt as one pass. The caller then stops at the boundary as usual.
    if (position_ != 0 &&
        (!frontier_.has_value() ||
         (verifier_ != nullptr && sequence_.size() != position_))) {
      return std::nullopt;
    }
    std::size_t bytes = executor_->SnapshotPayloadBytes(boundary) +
                        model_->GetConfig().vocab_size * sizeof(float);
    if (verifier_ != nullptr) {
      const auto verifier =
          verifier_->SnapshotPayloadBytesAt(boundary, position_ == 0);
      if (!verifier.has_value()) {
        return std::nullopt;
      }
      bytes += *verifier;
    }
    return bytes;
  }

  /// Runs the rest of the prompt as one pass. The first token goes out from
  /// the hook; the boundary's state is captured after it.
  void PrefillThroughCheckpoint(
      std::span<const TextRunnerToken> tokens, std::uint32_t boundary,
      std::uint32_t total_prompt_tokens,
      const std::function<void()>* logits_hook,
      std::unique_ptr<hip::QwenGpuSnapshot>* target,
      std::optional<TextRunnerToken>* boundary_frontier,
      std::vector<float>* boundary_logits,
      std::unique_ptr<speculative::SpeculativeVerifierSnapshot>* verifier) {
    const auto position = static_cast<std::uint32_t>(position_);
    if (position != 0 &&
        (!frontier_.has_value() ||
         (verifier_ != nullptr && sequence_.size() != position_))) {
      throw std::logic_error("Qwen prompt checkpoint has no retained prefix");
    }
    // Any token still pending goes through this pass with the rest. The hook
    // below may leave the first token of this request pending.
    resume_tokens_.clear();
    executor_->ArmPromptCheckpoint(boundary);
    struct Disarm {
      hip::QwenGpuExecutor& executor;
      ~Disarm() { executor.DisarmPromptCheckpoint(); }
    } disarm{*executor_};
    const auto save_target = [&]() -> tokenization::TokenId {
      *target = executor_->SavePromptCheckpoint();
      *boundary_frontier = executor_->PromptCheckpointFrontier();
      *boundary_logits = executor_->TakePromptCheckpointLogits();
      return executor_->PromptCheckpointFrontier();
    };
    if (verifier_ != nullptr) {
      const TargetLogitsHook armed{*this, logits_hook};
      frontier_ =
          verifier_->PromptThrough(tokens, position, boundary,
                                   total_prompt_tokens, save_target, verifier);
      const auto logits = verifier_->CopyLastTargetLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
      if (position == 0) {
        sequence_.assign(tokens.begin(), tokens.end());
      } else {
        sequence_.insert(sequence_.end(), tokens.begin(), tokens.end());
      }
    } else {
      frontier_ = executor_->ForwardPromptBatch(tokens, position, true);
      const auto logits = executor_->CopyLastLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
      if (logits_hook != nullptr && *logits_hook) {
        (*logits_hook)();
      }
      (void)save_target();
    }
    position_ += tokens.size();
    frontier_published_ = false;
    rollback_position_.reset();
  }

  void PreparePrefixReuse(std::span<const TextRunnerToken> prefix) {
    if (position_ != prefix.size() || !frontier_.has_value()) {
      throw std::logic_error(
          "Qwen retained prefix does not match its target state");
    }
    if (verifier_ != nullptr) {
      verifier_->BeginRequest();
      sequence_.assign(prefix.begin(), prefix.end());
      // Verification's frontier may be a sampled draw. A new request must
      // select from its distribution again, including when it switches to
      // greedy sampling or omits the previously published pending token.
      if (!resume_tokens_.empty() && !frontier_logits_.empty()) {
        sampling::SamplerState greedy;
        frontier_ = greedy.Sample(frontier_logits_);
      }
    }
    frontier_published_ = false;
    rollback_position_.reset();
  }

  bool ResumeGeneratedToken(TextRunnerToken token, bool decode_ready) {
    if (resume_tokens_.empty())
      return false;
    if (resume_tokens_.front() != token) {
      resume_tokens_.clear();
      return false;
    }
    resume_tokens_.erase(resume_tokens_.begin());
    // This token was already published by the previous request. Consume it
    // with decode arithmetic, exactly as AR did, before chunking the new
    // prompt suffix. Including it in a prefill chunk changes recurrent state.
    const auto position = static_cast<std::uint32_t>(position_);
    frontier_ = verifier_ != nullptr
                    ? verifier_->AdvanceCommittedToken(token, position)
                    : executor_->ForwardToken(token, position);
    if (verifier_ != nullptr)
      sequence_.push_back(token);
    ++position_;
    frontier_logits_.clear();
    if (decode_ready) {
      const auto logits = executor_->CopyLastLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
    }
    return true;
  }

  void ExtendSpeculative(std::span<const TextRunnerToken> suffix,
                         bool decode_ready, std::uint32_t total_prompt_tokens,
                         const std::function<void()>* logits_hook = nullptr) {
    if (verifier_ == nullptr || !frontier_.has_value()) {
      throw std::logic_error("Qwen speculative prefix is not initialized");
    }
    const TargetLogitsHook armed{*this, decode_ready ? logits_hook : nullptr};
    if (sequence_.size() != position_) {
      throw std::logic_error(
          "Qwen speculative sequence does not match retained position");
    }
    frontier_ =
        verifier_->ExtendPrompt(suffix, static_cast<std::uint32_t>(position_),
                                decode_ready, total_prompt_tokens);
    sequence_.insert(sequence_.end(), suffix.begin(), suffix.end());
    position_ += suffix.size();
    frontier_logits_.clear();
    if (decode_ready) {
      const auto logits = verifier_->CopyLastTargetLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
    }
    frontier_published_ = false;
  }

  [[nodiscard]] TextRunnerToken SelectFrontier(
      sampling::SamplerState& sampler) const {
    if (!frontier_.has_value()) {
      throw std::logic_error("Qwen state has no next-token frontier");
    }
    // A fired loop guard forces its end sequence even though the config still
    // reports the raw-argmax path; CanSelectArgmax already admits only the
    // forced token, and the cached-logits path below defers to the sampler.
    // A banned frontier diverts the same way: one lookup on the raw argmax,
    // then a masked host selection from this row (no device resample, which
    // would draw the banned token again).
    if ((sampler.config().can_use_unmodified_argmax() &&
         !sampler.ForcedToken().has_value() && !sampler.IsBanned(*frontier_)) ||
        (verifier_ == nullptr && sampler.CanSelectArgmax(*frontier_))) {
      return *frontier_;
    }
    if (sampler.IsBanned(*frontier_)) {
      if (frontier_logits_.empty()) {
        return sampler.Sample(executor_->CopyLastLogits());
      }
      return sampler.Sample(std::span<const float>(frontier_logits_));
    }
    if (frontier_logits_.empty()) {
      return executor_->SampleLastLogits(sampler);
    }
    return executor_->SampleCachedLogits(frontier_logits_, sampler);
  }

  [[nodiscard]] bool PrepareSpeculativeDecode(std::size_t max_tokens,
                                              sampling::SamplerState& sampler,
                                              TextDecodeStep& result) {
    if (verifier_ == nullptr || !frontier_.has_value()) {
      throw std::logic_error("Qwen speculative state has no frontier");
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "Qwen speculative decode budget must be at least one token");
    }
    resume_tokens_.clear();
    if (!frontier_published_) {
      if (!sampler.config().can_use_unmodified_argmax() ||
          sampler.ForcedToken().has_value() || sampler.IsBanned(*frontier_))
        frontier_ = SelectFrontier(sampler);
      // The frontier drew from these logits (prefill-final or the verified
      // next row); report them when the request armed logprobs.
      std::optional<sampling::CapturedTokenLogprobs> frontier_logprobs;
      if (sampler.logprobs_enabled()) {
        const auto top_n = sampler.logprobs_top_n();
        const auto row = CopyFrontierLogits();
        if (!row.empty()) {
          frontier_logprobs = sampler.CaptureLogprobs(
              row, static_cast<sampling::TokenId>(*frontier_), top_n);
        }
      }
      if (!AppendSpeculativeSelection(*frontier_, sampler, result,
                                      std::move(frontier_logprobs)))
        return false;
      frontier_published_ = true;
    }
    return result.selections.size() < max_tokens;
  }

  [[nodiscard]] speculative::SpeculativeVerifier::StepRequest
  VerificationRequest(std::size_t remaining, sampling::SamplerState& sampler) {
    rollback_position_ = position_;
    rollback_tokens_.assign(1, *frontier_);
    rollback_logits_ = std::move(frontier_logits_);
    // Verification keeps its own EOS handling: the tokenizer stop set already
    // covers EOS, so a sentinel id here would change nothing. An ignore-EOS
    // request instead publishes the stop token in AppendSpeculativeSelection
    // and continues, at the cost of a window clipped on every EOS crossing.
    const auto eos_id = model_->GetConfig().llama
                            ? static_cast<tokenization::TokenId>(
                                  model_->GetLlamaTokenizer() != nullptr
                                      ? model_->GetLlamaTokenizer()->Eos()
                                      : 0)
                            : model_->GetTokenizer().GetEosTokenId();
    return {*verifier_,
            sequence_,
            static_cast<std::uint32_t>(position_),
            *frontier_,
            eos_id,
            static_cast<std::uint32_t>(std::min<std::size_t>(
                remaining, std::numeric_limits<std::uint32_t>::max())),
            sampler,
            sampler.logprobs_enabled(),
            sampler.logprobs_top_n()};
  }

  void FinishSpeculativeDecode(
      speculative::SpeculativeVerifier::StepResult verification,
      sampling::SamplerState& sampler, TextDecodeStep& result) {
    result.draft_rounds = verification.draft_count > 0 ? 1 : 0;
    result.draft_tokens = verification.draft_count;
    result.draft_accepted_tokens = verification.accepted_count;
    result.execution_plan = {
        .kind = verification.physical_width > 1
                    ? TextExecutionPlanKind::kBatched
                    : TextExecutionPlanKind::kSerial,
        .physical_width = verification.physical_width,
    };
    std::size_t emitted_index = 0;
    for (const TextRunnerToken token : verification.emitted_tokens) {
      // Each prediction consumes one input, including a terminal prediction.
      // EOS itself remains the unconsumed frontier and is not published.
      ++position_;
      std::optional<sampling::CapturedTokenLogprobs> emitted_logprobs;
      if (emitted_index < verification.emitted_target_logprobs.size()) {
        emitted_logprobs = verification.emitted_target_logprobs[emitted_index];
        emitted_logprobs->token = token;
      }
      if (!AppendSpeculativeSelection(token, sampler, result,
                                      std::move(emitted_logprobs)))
        break;
      rollback_tokens_.push_back(token);
      ++emitted_index;
    }
    frontier_ = verification.next_token;
    frontier_logits_ = std::move(verification.next_token_logits);
    frontier_published_ = !result.stop;
    set_pending_token(frontier_published_ ? frontier_ : std::nullopt);
    if (verification.draft_count == 0)
      rollback_position_.reset();
  }

  void PrepareCancellation() {
    if (!rollback_position_)
      return;
    if (rollback_tokens_.empty() ||
        rollback_tokens_.size() > kQwenMaxResumeTokens ||
        rollback_logits_.empty())
      throw std::logic_error(
          "Qwen cancelled verification has no exact frontier");
    // Verification already saved the target state. Its target features are
    // still pending in the draft, so rewinding needs no new per-step copy.
    if (mtp_backend_ != nullptr) {
      mtp_backend_->DiscardPendingTargetContext(
          static_cast<std::uint32_t>(*rollback_position_));
    } else {
      draft_backend_->DiscardPendingTargetContext(
          static_cast<std::uint32_t>(*rollback_position_));
    }
    executor_->RestoreState();
    position_ = *rollback_position_;
    sequence_.resize(position_);
    frontier_logits_ = std::move(rollback_logits_);
    sampling::SamplerState greedy;
    frontier_ = greedy.Sample(frontier_logits_);
    frontier_published_ = false;
    resume_tokens_ = std::move(rollback_tokens_);
    rollback_position_.reset();
  }

  [[nodiscard]] TextDecodeStep DecodeSpeculative(
      std::size_t max_tokens, sampling::SamplerState& sampler) {
    TextDecodeStep result;
    sampling::SamplerState working_sampler = sampler;
    if (PrepareSpeculativeDecode(max_tokens, working_sampler, result)) {
      const auto request = VerificationRequest(
          max_tokens - result.selections.size(), working_sampler);
      auto verification = verifier_->VerifyStep(
          sequence_, request.position, request.current_token, request.eos_id,
          request.max_emitted_tokens, working_sampler);
      FinishSpeculativeDecode(std::move(verification), working_sampler, result);
    }
    sampler.SetRngState(working_sampler.rng_state());
    return result;
  }

  [[nodiscard]] hip::QwenGpuExecutor& executor() const { return *executor_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }
  [[nodiscard]] const std::optional<TextRunnerToken>& frontier()
      const noexcept {
    return frontier_;
  }
  void set_frontier(TextRunnerToken frontier) noexcept {
    frontier_ = frontier;
    frontier_logits_.clear();
    frontier_published_ = false;
    resume_tokens_.clear();
  }
  [[nodiscard]] std::vector<float> CopyFrontierLogits() const {
    if (!frontier_logits_.empty()) {
      return frontier_logits_;
    }
    const auto logits = executor_->CopyLastLogits();
    return {logits.begin(), logits.end()};
  }
  void RestoreFrontierLogits(std::vector<float> logits) {
    frontier_logits_ = std::move(logits);
  }
  [[nodiscard]] const std::vector<TextRunnerToken>& pending_tokens()
      const noexcept {
    return resume_tokens_;
  }
  void set_pending_token(std::optional<TextRunnerToken> token) {
    resume_tokens_.clear();
    if (token)
      resume_tokens_.push_back(*token);
  }
  void RestorePendingTokens(std::span<const TextRunnerToken> tokens) {
    resume_tokens_.assign(tokens.begin(), tokens.end());
    rollback_position_.reset();
  }
  [[nodiscard]] std::unique_ptr<speculative::SpeculativeVerifierSnapshot>
  SaveVerifierSnapshot() const {
    return verifier_ != nullptr ? verifier_->Snapshot() : nullptr;
  }
  // F7: refuse an MTP capture unless the draft prefix covers exactly the
  // target's committed context. A lagging draft would restore at the wrong
  // position and every later Propose would throw (an HTTP 500 in serve),
  // so the capture is refused here and the cache path records a skip.
  void CheckMtpDraftPosition() const {
    if (mtp_backend_ != nullptr) {
      hip::CheckMtpSnapshotPosition(mtp_backend_->DraftPosition(), position_);
    }
  }
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const {
    std::size_t bytes = executor_->SnapshotPayloadBytes(position_);
    const auto checked_add = [&bytes](std::size_t value) {
      if (value > std::numeric_limits<std::size_t>::max() - bytes) {
        throw std::overflow_error("Qwen snapshot size overflows");
      }
      bytes += value;
    };
    const std::size_t logits_count = frontier_logits_.empty()
                                         ? model_->GetConfig().vocab_size
                                         : frontier_logits_.size();
    if (logits_count >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      throw std::overflow_error("Qwen snapshot size overflows");
    }
    checked_add(logits_count * sizeof(float));
    checked_add(resume_tokens_.size() * sizeof(TextRunnerToken));
    checked_add(std::ranges::count_if(executor_->VisionLayout().images,
                                      [this](const auto& image) {
                                        return image.offset < position_;
                                      }) *
                sizeof(models::qwen::vision::ImageGrid));
    if (verifier_ != nullptr) {
      checked_add(verifier_->SnapshotPayloadBytes());
    }
    return bytes;
  }
  void RestoreVerifierSnapshot(
      const speculative::SpeculativeVerifierSnapshot& snapshot) {
    if (verifier_ == nullptr) {
      throw std::invalid_argument(
          "cannot restore speculative state into a plain Qwen session");
    }
    verifier_->RestoreSnapshot(snapshot);
  }
  void RestoreVerifierPersistentSnapshot(
      std::span<const std::uint8_t> payload) {
    if (verifier_ == nullptr) {
      throw std::invalid_argument(
          "cannot restore speculative persistent state into a plain Qwen "
          "session");
    }
    verifier_->RestorePersistentSnapshot(payload);
  }

private:
  bool AppendSpeculativeSelection(
      TextRunnerToken token, sampling::SamplerState& sampler,
      TextDecodeStep& result,
      std::optional<sampling::CapturedTokenLogprobs> logprobs = {}) {
    const bool stop =
        model_->GetConfig().llama
            ? (stop_at_eos() && model_->GetLlamaTokenizer() != nullptr &&
               model_->GetLlamaTokenizer()->IsStop(
                   static_cast<std::int32_t>(token)))
            : (stop_at_eos() && model_->GetTokenizer().IsStopToken(token));
    if (stop) {
      result.stop = true;
      return false;
    }
    std::string piece;
    if (model_->GetConfig().llama) {
      const auto* tok = model_->GetLlamaTokenizer();
      piece = tok != nullptr ? tok->TokenText(static_cast<std::int32_t>(token))
                             : std::string{};
    } else {
      piece = std::string(model_->GetTokenizer().DecodeToken(token));
    }
    if (logprobs.has_value() && logprobs->text.empty()) {
      logprobs->text = piece;
    }
    result.selections.push_back({
        .stop = false,
        .token = token,
        .piece = std::move(piece),
        .logprobs = std::move(logprobs),
    });
    sequence_.push_back(token);
    set_pending_token(token);
    sampler.Accept(token);
    return true;
  }

  std::shared_ptr<const hip::QwenGpuModel> model_;
  std::unique_ptr<hip::QwenGpuExecutor> executor_;
  std::unique_ptr<speculative::SpeculativeVerifier> verifier_;
  // Owned by verifier_; used only to account the session's draft allocations.
  hip::QwenDFlashGpuDraftBackend* draft_backend_{nullptr};
  hip::QwenMtpGpuDraftBackend* mtp_backend_{nullptr};
  std::vector<TextRunnerToken> sequence_;
  std::size_t position_{0};
  std::optional<TextRunnerToken> frontier_;
  std::vector<float> frontier_logits_;
  bool frontier_published_{false};
  std::vector<TextRunnerToken> resume_tokens_;
  std::optional<std::size_t> rollback_position_;
  std::vector<TextRunnerToken> rollback_tokens_;
  std::vector<float> rollback_logits_;
};

class QwenTextRunnerSnapshot final : public TextRunnerSnapshot {
public:
  QwenTextRunnerSnapshot(
      std::shared_ptr<const hip::QwenGpuModel> model,
      std::unique_ptr<hip::QwenGpuSnapshot> snapshot, std::size_t position,
      std::optional<TextRunnerToken> frontier,
      std::vector<float> frontier_logits,
      std::vector<TextRunnerToken> pending_tokens,
      std::unique_ptr<speculative::SpeculativeVerifierSnapshot>
          verifier_snapshot)
      : model(std::move(model)),
        snapshot(std::move(snapshot)),
        position(position),
        frontier(frontier),
        frontier_logits(std::move(frontier_logits)),
        pending_tokens(std::move(pending_tokens)),
        verifier_snapshot(std::move(verifier_snapshot)) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return (snapshot != nullptr ? snapshot->PayloadBytes() : 0) +
           frontier_logits.size() * sizeof(float) +
           pending_tokens.size() * sizeof(TextRunnerToken) +
           (verifier_snapshot != nullptr ? verifier_snapshot->PayloadBytes()
                                         : 0);
  }

  std::shared_ptr<const hip::QwenGpuModel> model;
  std::unique_ptr<hip::QwenGpuSnapshot> snapshot;
  std::size_t position;
  std::optional<TextRunnerToken> frontier;
  std::vector<float> frontier_logits;
  std::vector<TextRunnerToken> pending_tokens;
  std::unique_ptr<speculative::SpeculativeVerifierSnapshot> verifier_snapshot;
};

QwenTextRunnerState& RequireQwenState(TextRunnerState& state) {
  auto* qwen = dynamic_cast<QwenTextRunnerState*>(&state);
  if (qwen == nullptr) {
    throw std::logic_error("text runner state is not Qwen");
  }
  return *qwen;
}

const QwenTextRunnerState& RequireQwenState(const TextRunnerState& state) {
  const auto* qwen = dynamic_cast<const QwenTextRunnerState*>(&state);
  if (qwen == nullptr) {
    throw std::logic_error("text runner state is not Qwen");
  }
  return *qwen;
}

// Owns a four-byte device buffer and a private stream from load, so probing
// a lost context needs no allocation. HIP context loss is sticky: after a GPU
// reset even this memset fails with a hard error. Only such an error marks the
// device lost; a memset still pending at the deadline may be queued behind
// long kernels on the shared hardware queues, so it counts as usable.
class HipTextModelRunner : public TextModelRunner {
public:
  HipTextModelRunner() {
    HIP_CHECK(hipStreamCreateWithFlags(&probe_stream_, hipStreamNonBlocking));
    if (const auto error = hipMalloc(&probe_buffer_, sizeof(std::uint32_t));
        error != hipSuccess) {
      hip::LogCleanupError(hipStreamDestroy(probe_stream_));
      throw std::runtime_error(std::string("device probe allocation: ") +
                               hipGetErrorString(error));
    }
  }
  ~HipTextModelRunner() override {
    hip::LogCleanupError(hipFree(probe_buffer_));
    hip::LogCleanupError(hipStreamDestroy(probe_stream_));
  }
  HipTextModelRunner(const HipTextModelRunner&) = delete;
  HipTextModelRunner& operator=(const HipTextModelRunner&) = delete;
  HipTextModelRunner(HipTextModelRunner&&) = delete;
  HipTextModelRunner& operator=(HipTextModelRunner&&) = delete;

  [[nodiscard]] DeviceProbeStatus PollDevice() const override {
    if (!probe_pending_) {
      // Clear the failed work unit's thread-local error before probing.
      (void)hipGetLastError();
      if (hipMemsetAsync(probe_buffer_, 0, sizeof(std::uint32_t),
                         probe_stream_) != hipSuccess ||
          hipGetLastError() != hipSuccess)
        return DeviceProbeStatus::kLost;
      probe_pending_ = true;
    }
    const auto status = hipStreamQuery(probe_stream_);
    if (status == hipErrorNotReady)
      return DeviceProbeStatus::kPending;
    probe_pending_ = false;
    return status == hipSuccess ? DeviceProbeStatus::kUsable
                                : DeviceProbeStatus::kLost;
  }

  [[nodiscard]] bool DeviceUsable() const override {
    constexpr auto kTimeout = std::chrono::seconds(5);
    const auto deadline = Clock::now() + kTimeout;
    for (;;) {
      const auto status = PollDevice();
      if (status == DeviceProbeStatus::kUsable)
        return true;
      if (status == DeviceProbeStatus::kLost)
        return false;
      if (Clock::now() >= deadline) {
        Logger::Warn("scheduler", "event=device_probe_timeout timeout_s=" +
                                      std::to_string(kTimeout.count()));
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

private:
  hipStream_t probe_stream_{};
  void* probe_buffer_{};
  mutable bool probe_pending_{false};
};

class QwenTextRunner final : public HipTextModelRunner {
public:
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    // Flint's template teaches <tool_call>NAME<arg_key>K</arg_key>...
    return model_->GetConfig().flint
               ? sampling::JsonConstraint::ToolFormat::kFlint
               : sampling::JsonConstraint::ToolFormat::kQwen;
  }
  QwenTextRunner(
      std::shared_ptr<const hip::QwenGpuModel> model, std::uint32_t max_context,
      std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model = nullptr,
      std::shared_ptr<const hip::QwenMtpGpuModel> mtp_model = nullptr,
      speculative::SpeculativeOptions speculative_options = {},
      std::string artifact_fingerprint = {},
      std::string draft_artifact_fingerprint = {})
      : model_(std::move(model)),
        dflash_model_(std::move(dflash_model)),
        mtp_model_(std::move(mtp_model)),
        max_context_(max_context),
        speculative_options_(speculative_options),
        execution_policy_(hip::QwenExecutionPolicy::Production()) {
    if (dflash_model_ != nullptr && mtp_model_ != nullptr) {
      throw std::invalid_argument(
          "Qwen runner accepts one speculative draft backend");
    }
    // GLIMMER: fp32 KV. The fused QK-RoPE kernel is NEOX and always-on rope.
    if (model_->GetConfig().attention_only) {
      execution_policy_.kv_cache_storage = hip::QwenKvCacheStorage::kFp32;
      execution_policy_.fuse_qk_norm_rope_kv = false;
      execution_policy_.glimmer_fp16_kv = true;  // Glimmer fp16-KV plane
    }
    // LFM2: fp16 KV (SUPERROC_LFM2_KV=fp32 for the reference), no fused
    // QK-RoPE-KV (NOT glimmer_fp16_kv).
    if (superroc::core::IsLfm2Family(model_->GetConfig().architecture)) {
      execution_policy_.kv_cache_storage = hip::Lfm2KvCacheStorage();
      execution_policy_.fuse_qk_norm_rope_kv = false;
    }
    if (model_->GetConfig().flint) {
      execution_policy_.kv_cache_storage = hip::QwenKvCacheStorage::kFp32;
      execution_policy_.fuse_qk_norm_rope_kv = false;
      execution_policy_.glimmer_fp16_kv = true;
    }
    if (model_->GetConfig().llama) {
      execution_policy_.kv_cache_storage = hip::QwenKvCacheStorage::kFp32;
      execution_policy_.fuse_qk_norm_rope_kv = false;
      execution_policy_.glimmer_fp16_kv = true;
    }
    if (!artifact_fingerprint.empty()) {
      // Persistent snapshots cover DFlash and MTP sessions: the identity
      // encodes the draft artifact, and each draft backend persists its draft
      // KV prefix next to the target state, so a file written without a
      // drafter never restores into a drafter session and vice versa.
      const bool speculative = dflash_model_ != nullptr ||
                               mtp_model_ != nullptr ||
                               speculative_options_.prompt_lookup;
      persistence_ = TextRunnerPersistenceDescriptor{
          .compatibility_identity = QwenCompatibilityIdentity(
              artifact_fingerprint, draft_artifact_fingerprint, max_context_,
              execution_policy_, speculative, speculative_options_,
              superroc::core::IsLfm2Family(model_->GetConfig().architecture),
              model_->GetConfig().flint, mtp_model_ != nullptr),
          .payload_version = kQwenPersistentPayloadVersion,
      };
    }
  }

  [[nodiscard]] bool SpeculationEnabled() const noexcept {
    return dflash_model_ != nullptr || mtp_model_ != nullptr ||
           speculative_options_.prompt_lookup;
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    const bool speculative_enabled = SpeculationEnabled();
    return {
        .model_id = model_->GetConfig().model_name,
        .state_abi =
            model_->GetConfig().flint
                ? std::string(FlintStateAbi(execution_policy_.glimmer_fp16_kv))
            : superroc::core::IsLfm2Family(model_->GetConfig().architecture)
                ? std::string(
                      Lfm2StateAbi(speculative_enabled,
                                   execution_policy_.UsesFp16AttentionKv()))
                : std::string(
                      QwenStateAbi(speculative_enabled,
                                   execution_policy_.UsesFp16AttentionKv(),
                                   execution_policy_.UsesBf16RecurrentState())),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = !speculative_enabled,
                // LLAMA (F14): TokenText pieces are not Decode()-exact.
                .incremental_text_is_exact =
                    RunnerIncrementalTextIsExact(model_->GetConfig().llama),
                .multi_token_decode = speculative_enabled,
                // LFM2 verifies per session: the batched verifier kernels
                // assume the Qwen DeltaNet stack (see
                // ForwardVerificationChunk).
                .batched_multi_token_decode =
                    speculative_enabled &&
                    !superroc::core::IsLfm2Family(
                        model_->GetConfig().architecture) &&
                    MaximumDecodeBatchWidth() > 1,
                .batched_multi_token_decode_max_width =
                    speculative_enabled && !superroc::core::IsLfm2Family(
                                               model_->GetConfig().architecture)
                        ? MaximumDecodeBatchWidth()
                        : 0,
                .prefix_reuse = true,
                .in_pass_checkpoint = true,
                // Each prompt copy is a real device copy (27B, KAT, Ornith
                // MoE and their MTP / DFlash2 drafters): the host floor
                // governs this runner. DeepSeek and Flash-Next share
                // session rows, so they leave the capability false.
                .snapshot_device_copies = true,
            },
        .persistence = persistence_,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    auto usage = hip::QwenGpuExecutor::EstimateMemoryUsage(
        model_->GetConfig(), max_context_, execution_policy_);
    std::size_t resident_weights = model_->GetResidentBytes();
    if (dflash_model_ != nullptr) {
      const auto draft = hip::QwenDFlashGpuExecutor::EstimateMemoryUsage(
          *dflash_model_, max_context_, MaximumDecodeBatchWidth());
      usage.request_state_bytes += draft.request_state_bytes;
      usage.temporary_scratch_bytes += draft.temporary_scratch_bytes;
      resident_weights += dflash_model_->GetPackedWeightBytes();
    }
    if (mtp_model_ != nullptr) {
      // QwenMtpGpuExecutor has no EstimateMemoryUsage, so mirror the two
      // allocations that actually scale with the served context: the f32 and
      // f16 copies of its own full-size KV cache (hip/mtp/executor.cpp:74-102)
      // plus the per-session vocabulary logits. This is the "MTP allocates its
      // own KV" cost, and it is why --context must be sized for two caches.
      const auto& mtp_config = mtp_model_->GetConfig();
      const std::size_t total_kv =
          static_cast<std::size_t>(mtp_config.num_key_value_heads) *
          max_context_ * mtp_config.head_dim;
      resident_weights += mtp_model_->GetPackedWeightBytes();
      usage.request_state_bytes +=
          2 * total_kv * (sizeof(float) + sizeof(std::uint16_t));
      usage.request_state_bytes += mtp_config.vocab_size * sizeof(float);
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      capacity = free_bytes;
    }
    // Snapshot buffers are device allocations, but on a unified-memory APU
    // they are carved from the same RAM as every host allocation, and HIP's
    // free figure does not see host pressure. The static cap is fixed; the
    // host floor binds live on top of it.
    std::size_t snapshot_capacity =
        TextRunnerRamCacheOptions::kAutomaticMaxBytes;
    std::size_t snapshot_ceiling = SystemSnapshotCeilingBytes();
    if (capacity.has_value()) {
      snapshot_capacity = std::min(snapshot_capacity, *capacity);
      snapshot_ceiling = std::min(snapshot_ceiling, *capacity);
    }
    return {
        .resident_weights_bytes = resident_weights,
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = usage.request_state_bytes,
        .temporary_scratch_bytes = usage.temporary_scratch_bytes,
        .retained_snapshot_capacity_bytes = snapshot_capacity,
        .retained_snapshot_ceiling_bytes = snapshot_ceiling,
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {
            .kind = TextExecutionPlanKind::kSerial,
            .physical_width = 1,
        },
    };
    // Report actual session widths, including C6. Speculation reserves up to
    // eight token rows per session in the coordinator's existing scratch.
    for (std::size_t width = 2; width <= MaximumDecodeBatchWidth(); ++width) {
      plans.push_back({
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = width,
      });
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    if (model_->GetConfig().llama) {
      const auto* tok = model_->GetLlamaTokenizer();
      const std::uint32_t n = tok != nullptr ? tok->Size() : 0;
      return std::make_shared<const sampling::ConstraintVocabulary>(
          n, [tok](std::uint32_t id) {
            return sampling::ConstraintVocabulary::Piece{
                tok != nullptr ? tok->TokenText(static_cast<std::int32_t>(id))
                               : std::string{},
                tok != nullptr && tok->IsStop(static_cast<std::int32_t>(id))};
          });
    }
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->GetTokenizer().GetVocabSize(), [this](std::uint32_t id) {
          const auto& tokenizer = model_->GetTokenizer();
          return sampling::ConstraintVocabulary::Piece{
              tokenizer.DecodeTokenCopy(id), tokenizer.IsStopToken(id)};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    if (model_->GetConfig().llama) {
      const auto* tok = model_->GetLlamaTokenizer();
      return tok != nullptr ? tok->Size() : 0;
    }
    return static_cast<std::uint32_t>(model_->GetTokenizer().GetVocabSize());
  }

  // llama.cpp's Qwen3-Coder format forces "\n</think>" and also ends
  // reasoning at a tool call.
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {"<think>", "\n</think>", {"</think>", "<tool_call>"}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    if (model_->GetConfig().llama) {
      const auto* tok = model_->GetLlamaTokenizer();
      if (tok == nullptr) {
        return {};
      }
      models::llama::SpmTokenizer::EncodeOptions options;
      options.add_special = true;
      options.parse_special = true;
      const auto ids = tok->Encode(text, options);
      std::vector<TextRunnerToken> out;
      out.reserve(ids.size());
      for (const std::int32_t id : ids) {
        out.push_back(static_cast<TextRunnerToken>(id));
      }
      return out;
    }
    return model_->GetTokenizer().Encode(text);
  }

  // LLAMA: SpmTokenizer::Encode(add_special) already prepends the BOS its
  // GGUF asks for, so Tokenize is complete here and RawPromptBos stays
  // nullopt. Every other family encodes flag-blind and reports the GGUF BOS.
  [[nodiscard]] std::optional<TextRunnerToken> RawPromptBos() const override {
    if (model_->GetConfig().llama) {
      return std::nullopt;
    }
    return model_->GetTokenizer().RawPromptBos();
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    // GLIMMER: pinned Onyx ATEM template, including tool definitions.
    // Tool-role history still fails closed inside GlimmerRender.
    if (model_->GetConfig().attention_only) {
      // C++20 span has no range constructor. Build it from pointer and size.
      const auto messages = std::span<const tokenization::ChatMessage>(
          request.messages.data(), request.messages.size());
      const auto tools = request.tool_choice == ChatRequest::ToolChoice::kNone
                             ? std::span<const tokenization::ChatTool>{}
                             : std::span<const tokenization::ChatTool>(
                                   request.tools.data(), request.tools.size());
      return tokenization::GlimmerRenderAndTokenize(
          model_->GetTokenizer(), messages, tools,
          QwenChatOptions(request, max_context_));
    }
    // LFM2: pinned lfm2moe template, same span/tool_choice handling.
    if (superroc::core::IsLfm2Family(model_->GetConfig().architecture)) {
      const auto messages = std::span<const tokenization::ChatMessage>(
          request.messages.data(), request.messages.size());
      const auto tools = request.tool_choice == ChatRequest::ToolChoice::kNone
                             ? std::span<const tokenization::ChatTool>{}
                             : std::span<const tokenization::ChatTool>(
                                   request.tools.data(), request.tools.size());
      const tokenization::Lfm2TemplateVariant* lfm2_variant =
          tokenization::Lfm2TemplateVariantForGguf(model_->GetReader());
      if (lfm2_variant == nullptr)
        return std::nullopt;
      return tokenization::Lfm2RenderAndTokenize(
          model_->GetTokenizer(), messages, tools,
          QwenChatOptions(request, max_context_), nullptr, *lfm2_variant);
    }
    if (model_->GetConfig().flint) {
      const auto messages = std::span<const tokenization::ChatMessage>(
          request.messages.data(), request.messages.size());
      const auto tools = request.tool_choice == ChatRequest::ToolChoice::kNone
                             ? std::span<const tokenization::ChatTool>{}
                             : std::span<const tokenization::ChatTool>(
                                   request.tools.data(), request.tools.size());
      return tokenization::FlintRenderAndTokenize(
          model_->GetTokenizer(), messages, tools,
          QwenChatOptions(request, max_context_));
    }
    if (model_->GetConfig().llama) {
      // LLAMA (F13): content plus framing_suffix (structured-output
      // instructions); tool requests are refused, not silently dropped.
      // SR-03: client content overlapping a control spelling is encoded with
      // special parsing off; otherwise the render encodes exactly as before.
      std::vector<core::TextSpan> spans;
      const auto rendered = LlamaRenderPrompt(request, &spans);
      if (!rendered.has_value()) {
        return std::nullopt;
      }
      const auto* tok = model_->GetLlamaTokenizer();
      if (tok == nullptr) {
        return std::vector<TextRunnerToken>{};
      }
      const auto spellings = tok->SpecialSpellings();
      return core::EncodeWithContentSpans<TextRunnerToken>(
          *rendered, spans, spellings,
          [&](std::size_t offset, std::string_view piece, bool parse_special) {
            models::llama::SpmTokenizer::EncodeOptions options;
            // BOS once, at the very start only.
            options.add_special = (offset == 0);
            options.parse_special = parse_special;
            const auto ids = tok->Encode(piece, options);
            std::vector<TextRunnerToken> out;
            out.reserve(ids.size());
            for (const std::int32_t id : ids) {
              out.push_back(static_cast<TextRunnerToken>(id));
            }
            return out;
          });
    }
    return tokenization::QwenChatTemplate::RenderAndTokenize(
        model_->GetTokenizer(), request.messages,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        QwenChatOptions(request, max_context_));
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    // The Qwen vision prompt renders ChatML. Glimmer chat has to use
    // GlimmerRender or the model never opens to=self / ATEM turns.
    if (model_->GetConfig().attention_only) {
      auto tokens = RenderAndTokenize(request);
      if (!tokens.has_value()) {
        return std::nullopt;
      }
      return TextPreparedPrompt{std::move(*tokens), {}, 0};
    }
    // LFM2: the same tokens-only TextPreparedPrompt as Glimmer.
    if (superroc::core::IsLfm2Family(model_->GetConfig().architecture)) {
      auto tokens = RenderAndTokenize(request);
      if (!tokens.has_value()) {
        return std::nullopt;
      }
      return TextPreparedPrompt{std::move(*tokens), {}, 0};
    }
    if (model_->GetConfig().llama) {
      auto tokens = RenderAndTokenize(request);
      if (!tokens.has_value()) {
        return std::nullopt;
      }
      return TextPreparedPrompt{std::move(*tokens), {}, 0};
    }
    if (model_->GetConfig().flint) {
      auto tokens = RenderAndTokenize(request);
      if (!tokens.has_value()) {
        return std::nullopt;
      }
      TextPreparedPrompt prepared{std::move(*tokens), {}, 0};
      // Flint: snapshot before the generation framing (<|assistant|>), the
      // DeepSeek runner's stable-frontier rule. An agent that resends the
      // assistant turn without its reasoning diverges exactly there.
      auto continuation = request;
      continuation.messages.emplace_back(tokenization::ChatRole::kUser, "");
      if (const auto continued = RenderAndTokenize(continuation)) {
        prepared.cache_prefix_tokens = static_cast<std::size_t>(
            std::mismatch(prepared.tokens.begin(), prepared.tokens.end(),
                          continued->begin(), continued->end())
                .first -
            prepared.tokens.begin());
      }
      return prepared;
    }
    return PrepareQwenPrompt(request, model_->GetTokenizer(),
                             model_->VisionEncoder(), max_context_,
                             image_cache_);
  }

  [[nodiscard]] std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const override {
    if (model_->GetConfig().flint || model_->GetConfig().llama) {
      auto tokens = RenderAndTokenize(request);
      if (!tokens.has_value()) {
        return std::nullopt;
      }
      return tokens->size();
    }
    return CountQwenPromptTokens(request, model_->GetTokenizer(),
                                 model_->VisionEncoder(), max_context_,
                                 image_cache_);
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    RequireQwenState(state).executor().ConfigureVision(QwenPrompt(context),
                                                       model_->VisionEncoder());
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    if (model_->GetConfig().llama) {
      return TextGenerationBackend::InitialOutputState::kContent;
    }
    if (model_->GetConfig().flint)
      return TextGenerationBackend::InitialOutputState::kAuto;
    // Thinking on means the first Glimmer channel is reasoning. The
    // OpenAI layer splits to=self / to=user; it does not look for <think>.
    // LFM2: a required tool call prefills past </think>, so parsing starts
    // in content even when thinking is on.
    if (model_->GetConfig().lfm2 &&
        request.tool_choice == ChatRequest::ToolChoice::kRequired) {
      return TextGenerationBackend::InitialOutputState::kContent;
    }
    // LFM2 (Super Roc): the 8B-A1B and 1.2B templates do not open <think> in
    // the generation prompt; those models open it themselves when they think.
    // Auto starts in content and switches on a leading <think>.
    if (model_->GetConfig().lfm2 &&
        QwenChatOptions(request, max_context_).enable_thinking) {
      const tokenization::Lfm2TemplateVariant* variant =
          tokenization::Lfm2TemplateVariantForGguf(model_->GetReader());
      if (variant != nullptr && !variant->generation_opens_think) {
        return TextGenerationBackend::InitialOutputState::kAuto;
      }
    }
    return QwenChatOptions(request, max_context_).enable_thinking
               ? TextGenerationBackend::InitialOutputState::kReasoning
               : TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] bool MuseChannelParse() const override {
    return model_->GetConfig().attention_only;
  }

  // LFM2: LFM2.5 pythonic tool calls
  // (<|tool_call_start|>[f(a='x')]<|tool_call_end|>).
  [[nodiscard]] bool Lfm2PythonicTools() const override {
    return model_->GetConfig().lfm2;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    if (model_->GetConfig().llama) {
      const auto* tok = model_->GetLlamaTokenizer();
      if (tok == nullptr) {
        return {};
      }
      std::vector<std::int32_t> ids;
      ids.reserve(tokens.size());
      for (const auto id : tokens) {
        ids.push_back(static_cast<std::int32_t>(id));
      }
      return tok->Decode(ids);
    }
    return model_->GetTokenizer().Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<QwenTextRunnerState>(
        model_, max_context_, dflash_model_, mtp_model_, speculative_options_,
        execution_policy_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    auto& qwen = RequireQwenState(state);
    qwen.PreparePrefixReuse(prefix);
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& qwen = RequireQwenState(state);
    if (offset != qwen.position()) {
      throw std::logic_error(
          "Qwen prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("Qwen prefill has no remaining input");
    }
    max_input_tokens = std::min<std::size_t>(
        max_input_tokens, qwen.executor().GetMaxPromptBatch());
    // A pending token that ends the prompt takes its own step. With more prompt
    // behind it, it goes in their chunk: a step of its own would read every
    // weight once more for one token, on each turn of an agent.
    // With the canonical cache every prompt token goes through the prompt
    // path: a token consumed with decode arithmetic leaves another state than
    // the same token read in a request that holds the whole history.
    if (max_input_tokens != 0 && offset != 0) {
      if (offset + 1 == prompt.size() &&
          !::superroc::core::CanonicalCacheOn()) {
        if (qwen.ResumeGeneratedToken(prompt[offset], true)) {
          return {.consumed_tokens = 1, .decode_ready = true};
        }
      } else {
        qwen.set_pending_token(std::nullopt);
      }
    }
    if (qwen.speculative()) {
      if (offset == 0) {
        const auto consumed = std::min(max_input_tokens, prompt.size());
        if (consumed == 0) {
          throw std::logic_error(
              "Qwen prefill requires a nonzero token budget");
        }
        // The whole prompt is in scope here; its final length lets the
        // drafter skip rows that can never enter its attention window.
        qwen.PrimeSpeculative(prompt.first(consumed), consumed == prompt.size(),
                              static_cast<std::uint32_t>(prompt.size()),
                              state.prompt_logits_hook());
        return {.consumed_tokens = consumed,
                .decode_ready = consumed == prompt.size()};
      }
      const std::size_t consumed =
          std::min(max_input_tokens, prompt.size() - offset);
      if (consumed == 0) {
        throw std::logic_error(
            "Qwen DFlash retained prefix has no suffix to prefill");
      }
      qwen.ExtendSpeculative(prompt.subspan(offset, consumed),
                             offset + consumed == prompt.size(),
                             static_cast<std::uint32_t>(prompt.size()),
                             state.prompt_logits_hook());
      return {
          .consumed_tokens = consumed,
          .decode_ready = offset + consumed == prompt.size(),
      };
    }

    const std::size_t consumed =
        std::min(max_input_tokens, prompt.size() - offset);
    const bool decode_ready = offset + consumed == prompt.size();
    const auto frontier = qwen.executor().ForwardPromptBatch(
        prompt.subspan(offset, consumed), static_cast<std::uint32_t>(offset),
        decode_ready);
    qwen.set_position(offset + consumed);
    if (decode_ready) {
      qwen.set_frontier(frontier);
    }
    return {
        .consumed_tokens = consumed,
        .decode_ready = decode_ready,
    };
  }

  [[nodiscard]] std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget,
      std::size_t boundary) const override {
    static const bool enabled = [] {
      const char* e = ::superroc::core::GetEnv("SUPERROC_INPASS_CHECKPOINT");
      return e == nullptr || e[0] != '0';
    }();
    const auto& qwen = RequireQwenState(state);
    const std::size_t consumed = std::min<std::size_t>(
        {budget, prompt.size() - offset, qwen.executor().GetMaxPromptBatch()});
    if (!enabled || offset != qwen.position() ||
        offset + consumed != prompt.size() || boundary <= offset ||
        boundary >= prompt.size()) {
      return std::nullopt;
    }
    return qwen.PromptCheckpointBytes(static_cast<std::uint32_t>(boundary));
  }

  [[nodiscard]] TextPrefillStep PrefillThrough(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget, std::size_t boundary,
      std::unique_ptr<TextRunnerSnapshot>* checkpoint,
      const std::function<std::unique_ptr<TextRunnerSnapshot>()>&)
      const override {
    auto& qwen = RequireQwenState(state);
    if (checkpoint == nullptr || offset != qwen.position()) {
      throw std::logic_error("invalid Qwen in-pass checkpoint request");
    }
    const std::size_t consumed = std::min<std::size_t>(
        {budget, prompt.size() - offset, qwen.executor().GetMaxPromptBatch()});
    std::unique_ptr<hip::QwenGpuSnapshot> target;
    std::optional<TextRunnerToken> frontier;
    std::vector<float> logits;
    std::unique_ptr<speculative::SpeculativeVerifierSnapshot> verifier;
    qwen.PrefillThroughCheckpoint(
        prompt.subspan(offset, consumed), static_cast<std::uint32_t>(boundary),
        static_cast<std::uint32_t>(prompt.size()), state.prompt_logits_hook(),
        &target, &frontier, &logits, &verifier);
    *checkpoint = std::make_unique<QwenTextRunnerSnapshot>(
        model_, std::move(target), boundary, frontier, std::move(logits),
        std::vector<TextRunnerToken>{}, std::move(verifier));
    return {
        .consumed_tokens = consumed,
        .decode_ready = offset + consumed == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& qwen = RequireQwenState(state);
    const TextRunnerToken token = qwen.SelectFrontier(sampler);
    const bool stop =
        model_->GetConfig().llama
            ? (qwen.stop_at_eos() && model_->GetLlamaTokenizer() != nullptr &&
               model_->GetLlamaTokenizer()->IsStop(
                   static_cast<std::int32_t>(token)))
            : EndsTurn(qwen, model_->GetTokenizer(), token);
    if (stop) {
      return {
          .stop = true,
          .token = 0,
          .piece = {},
      };
    }
    // Keep the selected token separate from the argmax frontier: sampled
    // cancellation and prompt snapshot restore must not replace that argmax.
    qwen.set_pending_token(token);
    std::string piece;
    if (model_->GetConfig().llama) {
      const auto* tok = model_->GetLlamaTokenizer();
      piece = tok != nullptr ? tok->TokenText(static_cast<std::int32_t>(token))
                             : std::string{};
    } else {
      piece = std::string(model_->GetTokenizer().DecodeToken(token));
    }
    // The frontier drew from this row (prefill-final, a verified next row,
    // or the device's last row); report it when the request armed logprobs.
    std::optional<sampling::CapturedTokenLogprobs> logprobs;
    if (sampler.logprobs_enabled()) {
      const auto top_n = sampler.logprobs_top_n();
      const auto row = qwen.CopyFrontierLogits();
      if (!row.empty()) {
        logprobs = sampler.CaptureLogprobs(
            row, static_cast<sampling::TokenId>(token), top_n);
        if (!piece.empty())
          logprobs->text = piece;
      }
    }
    return {
        .stop = false,
        .token = token,
        .piece = std::move(piece),
        .logprobs = std::move(logprobs),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& qwen = RequireQwenState(state);
    if (qwen.speculative()) {
      throw std::logic_error(
          "Qwen DFlash decoding requires a multi-token decode step");
    }
    if (!qwen.frontier().has_value()) {
      throw std::logic_error("Qwen decode state has no retained frontier");
    }
    const auto frontier = qwen.executor().ForwardToken(
        token, static_cast<std::uint32_t>(qwen.position()));
    qwen.set_position(qwen.position() + 1);
    qwen.set_frontier(frontier);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    auto& qwen = RequireQwenState(state);
    if (!qwen.speculative()) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    return qwen.DecodeSpeculative(max_tokens, sampler);
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (!SpeculationEnabled() || decodes.size() < 2 ||
        decodes.size() > MaximumDecodeBatchWidth()) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    std::vector<TextDecodeStep> steps(decodes.size());
    std::vector<QwenTextRunnerState*> states;
    std::vector<sampling::SamplerState> samplers;
    std::vector<speculative::SpeculativeVerifier::StepRequest> requests;
    std::vector<std::size_t> indices;
    states.reserve(decodes.size());
    samplers.reserve(decodes.size());
    requests.reserve(decodes.size());
    indices.reserve(decodes.size());
    for (std::size_t index = 0; index < decodes.size(); ++index) {
      const auto& decode = decodes[index];
      auto& state = RequireQwenState(decode.state.get());
      states.push_back(&state);
      auto& sampler = samplers.emplace_back(decode.sampler.get());
      if (state.PrepareSpeculativeDecode(decode.max_tokens, sampler,
                                         steps[index])) {
        requests.push_back(state.VerificationRequest(
            decode.max_tokens - steps[index].selections.size(), sampler));
        indices.push_back(index);
      }
    }
    auto verified = speculative::SpeculativeVerifier::VerifyBatch(requests);
    for (std::size_t item = 0; item < indices.size(); ++item) {
      const std::size_t index = indices[item];
      states[index]->FinishSpeculativeDecode(std::move(verified[item]),
                                             samplers[index], steps[index]);
    }
    for (std::size_t index = 0; index < decodes.size(); ++index)
      decodes[index].sampler.get().SetRngState(samplers[index].rng_state());
    return steps;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (SpeculationEnabled()) {
      throw std::logic_error(
          "Qwen speculative sessions do not support batch advance");
    }
    if (advances.size() < 2 || advances.size() > 8) {
      throw std::invalid_argument(
          "Qwen batched decode requires two to eight sessions");
    }

    std::vector<hip::QwenGpuBatchItem> items;
    std::vector<QwenTextRunnerState*> states;
    items.reserve(advances.size());
    states.reserve(advances.size());
    for (const auto& advance : advances) {
      auto& qwen = RequireQwenState(advance.state.get());
      if (!qwen.frontier().has_value()) {
        throw std::logic_error("Qwen batched state has no retained frontier");
      }
      states.push_back(&qwen);
      items.push_back({
          .executor = &qwen.executor(),
          .token_id = advance.token,
          .position = static_cast<std::uint32_t>(qwen.position()),
      });
    }

    const auto frontiers = hip::QwenGpuExecutor::ForwardTokenBatch(items);
    if (frontiers.size() != states.size()) {
      throw std::runtime_error(
          "Qwen batched decode returned an invalid frontier count");
    }
    for (std::size_t index = 0; index < states.size(); ++index) {
      states[index]->set_position(states[index]->position() + 1);
      states[index]->set_frontier(frontiers[index]);
    }
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireQwenState(state).position();
  }
  void PrepareCancellation(TextRunnerState& state) const override {
    RequireQwenState(state).PrepareCancellation();
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const override {
    return RequireQwenState(state).SnapshotPayloadBytes();
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    const auto& qwen = RequireQwenState(state);
    if (!qwen.frontier().has_value()) {
      throw std::logic_error("Qwen state has no exact frontier to snapshot");
    }
    // MTP sessions snapshot like DFlash ones: the target executor plus the
    // draft backend's own KV prefix, restored exactly (the MTP draft persists
    // its draft KV prefix, so disk admission accepts it like DFlash).
    // F7: refuse the capture when the draft prefix does not cover exactly
    // the target's committed context (the cache path records a skip).
    qwen.CheckMtpDraftPosition();
    return std::make_unique<QwenTextRunnerSnapshot>(
        model_,
        qwen.executor().SaveSnapshot(
            static_cast<std::uint32_t>(qwen.position())),
        qwen.position(), qwen.frontier(), qwen.CopyFrontierLogits(),
        qwen.pending_tokens(), qwen.SaveVerifierSnapshot());
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto* qwen_snapshot =
        dynamic_cast<const QwenTextRunnerSnapshot*>(&snapshot);
    if (qwen_snapshot == nullptr ||
        qwen_snapshot->model.get() != model_.get() ||
        qwen_snapshot->snapshot == nullptr ||
        !qwen_snapshot->frontier.has_value()) {
      throw std::invalid_argument(
          "Qwen snapshot does not belong to this model");
    }
    auto& restored = RequireQwenState(state);
    if (restored.speculative() !=
        (qwen_snapshot->verifier_snapshot != nullptr)) {
      throw std::invalid_argument(
          "Qwen snapshot speculative mode does not match the destination");
    }
    restored.executor().RestoreSnapshot(*qwen_snapshot->snapshot);
    restored.set_position(qwen_snapshot->position);
    restored.set_frontier(*qwen_snapshot->frontier);
    restored.RestoreFrontierLogits(qwen_snapshot->frontier_logits);
    restored.RestorePendingTokens(qwen_snapshot->pending_tokens);
    if (qwen_snapshot->verifier_snapshot != nullptr) {
      restored.RestoreVerifierSnapshot(*qwen_snapshot->verifier_snapshot);
    }
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    const auto* qwen_snapshot =
        dynamic_cast<const QwenTextRunnerSnapshot*>(&snapshot);
    const bool speculative = SpeculationEnabled();
    if (qwen_snapshot == nullptr ||
        qwen_snapshot->model.get() != model_.get() ||
        qwen_snapshot->snapshot == nullptr ||
        !qwen_snapshot->frontier.has_value() ||
        (qwen_snapshot->verifier_snapshot != nullptr) != speculative ||
        qwen_snapshot->position != qwen_snapshot->snapshot->ValidContext() ||
        qwen_snapshot->pending_tokens.size() > kQwenMaxResumeTokens ||
        qwen_snapshot->frontier_logits.size() !=
            model_->GetConfig().vocab_size) {
      throw std::invalid_argument(
          "Qwen persistent snapshot does not belong to this runner");
    }
    const std::size_t logits_bytes = CheckedPersistentAdd(
        0, qwen_snapshot->frontier_logits.size() * sizeof(float));
    const std::size_t verifier_payload_bytes =
        qwen_snapshot->verifier_snapshot != nullptr
            ? qwen_snapshot->verifier_snapshot->PersistentPayloadBytes()
            : 0;
    return CheckedPersistentAdd(
        CheckedPersistentAdd(
            CheckedPersistentAdd(
                kQwenPersistentSnapshotHeaderBytes,
                qwen_snapshot->snapshot->CompactPayloadBytes()),
            logits_bytes),
        verifier_payload_bytes);
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto* qwen_snapshot =
        dynamic_cast<const QwenTextRunnerSnapshot*>(&snapshot);
    const std::size_t expected_bytes = PersistentSnapshotPayloadBytes(snapshot);
    if (qwen_snapshot == nullptr || destination.size() != expected_bytes) {
      throw std::invalid_argument(
          "Qwen persistent snapshot destination size is invalid");
    }
    const std::size_t gpu_payload_bytes =
        qwen_snapshot->snapshot->CompactPayloadBytes();
    const std::size_t logits_bytes =
        qwen_snapshot->frontier_logits.size() * sizeof(float);
    const std::size_t verifier_payload_bytes =
        qwen_snapshot->verifier_snapshot != nullptr
            ? qwen_snapshot->verifier_snapshot->PersistentPayloadBytes()
            : 0;
    // Preserve zeroed reserved fields and unused pending-token slots without
    // clearing the large payload that the serializers below overwrite.
    std::fill_n(destination.begin(), kQwenPersistentSnapshotHeaderBytes,
                std::uint8_t{0});
    std::copy(kQwenPersistentSnapshotMagic.begin(),
              kQwenPersistentSnapshotMagic.end(), destination.begin());
    PutLittleEndian<std::uint32_t>(destination, 8,
                                   kQwenPersistentPayloadVersion);
    PutLittleEndian<std::uint32_t>(
        destination, 12,
        static_cast<std::uint32_t>(kQwenPersistentSnapshotHeaderBytes));
    PutLittleEndian<std::uint64_t>(
        destination, 16, static_cast<std::uint64_t>(qwen_snapshot->position));
    PutLittleEndian<std::uint32_t>(destination, 24, *qwen_snapshot->frontier);
    PutLittleEndian<std::uint32_t>(destination, 28,
                                   (qwen_snapshot->verifier_snapshot != nullptr
                                        ? kQwenPersistentSpeculativeFlag
                                        : 0U) |
                                       (!qwen_snapshot->pending_tokens.empty()
                                            ? kQwenPersistentPendingTokenFlag
                                            : 0U));
    PutLittleEndian<std::uint64_t>(
        destination, 32,
        static_cast<std::uint64_t>(qwen_snapshot->frontier_logits.size()));
    PutLittleEndian<std::uint64_t>(
        destination, 40, static_cast<std::uint64_t>(gpu_payload_bytes));
    PutLittleEndian<std::uint64_t>(destination, 48,
                                   static_cast<std::uint64_t>(expected_bytes));
    PutLittleEndian<std::uint64_t>(
        destination, 56, static_cast<std::uint64_t>(verifier_payload_bytes));
    PutLittleEndian<std::uint32_t>(
        destination, 64,
        static_cast<std::uint32_t>(qwen_snapshot->pending_tokens.size()));
    for (std::size_t i = 0; i < qwen_snapshot->pending_tokens.size(); ++i)
      PutLittleEndian<std::uint32_t>(destination,
                                     72 + i * sizeof(std::uint32_t),
                                     qwen_snapshot->pending_tokens[i]);
    const std::size_t gpu_offset = kQwenPersistentSnapshotHeaderBytes;
    const std::size_t logits_offset =
        CheckedPersistentAdd(gpu_offset, gpu_payload_bytes);
    const std::size_t verifier_offset =
        CheckedPersistentAdd(logits_offset, logits_bytes);
    const std::size_t written = qwen_snapshot->snapshot->SerializeCompact(
        destination.subspan(gpu_offset, gpu_payload_bytes));
    if (written != gpu_payload_bytes) {
      throw std::runtime_error(
          "Qwen compact snapshot serializer returned the wrong byte count");
    }
    std::memcpy(destination.data() + logits_offset,
                qwen_snapshot->frontier_logits.data(), logits_bytes);
    if (qwen_snapshot->verifier_snapshot != nullptr) {
      const std::size_t verifier_written =
          qwen_snapshot->verifier_snapshot->SerializePersistent(
              destination.subspan(verifier_offset, verifier_payload_bytes));
      if (verifier_written != verifier_payload_bytes) {
        throw std::runtime_error(
            "Qwen verifier serializer returned the wrong byte count");
      }
    }
    return destination.size();
  }

  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    auto& restored = RequireQwenState(state);
    if (payload.size() < kQwenPersistentSnapshotHeaderBytes ||
        !std::equal(kQwenPersistentSnapshotMagic.begin(),
                    kQwenPersistentSnapshotMagic.end(), payload.begin()) ||
        GetLittleEndian<std::uint32_t>(payload, 8) !=
            kQwenPersistentPayloadVersion ||
        GetLittleEndian<std::uint32_t>(payload, 12) !=
            kQwenPersistentSnapshotHeaderBytes) {
      throw std::invalid_argument("Qwen persistent snapshot header is invalid");
    }
    const std::size_t position =
        PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 16));
    const TextRunnerToken frontier =
        GetLittleEndian<std::uint32_t>(payload, 24);
    const std::uint32_t flags = GetLittleEndian<std::uint32_t>(payload, 28);
    const std::size_t logits_count =
        PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 32));
    const std::size_t gpu_payload_bytes =
        PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 40));
    const std::size_t total_bytes =
        PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 48));
    const std::size_t verifier_payload_bytes =
        PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 56));
    const bool speculative = (flags & kQwenPersistentSpeculativeFlag) != 0;
    const bool has_pending_token =
        (flags & kQwenPersistentPendingTokenFlag) != 0;
    const std::uint32_t pending_count =
        GetLittleEndian<std::uint32_t>(payload, 64);
    if (position == 0 || position > max_context_ ||
        position > std::numeric_limits<std::uint32_t>::max() ||
        (flags & ~(kQwenPersistentSpeculativeFlag |
                   kQwenPersistentPendingTokenFlag)) != 0 ||
        has_pending_token != (pending_count != 0) ||
        pending_count > kQwenMaxResumeTokens ||
        GetLittleEndian<std::uint32_t>(payload, 68) != 0 ||
        GetLittleEndian<std::uint32_t>(payload, 108) != 0 ||
        speculative != restored.speculative() ||
        frontier >= model_->GetConfig().vocab_size ||
        logits_count != model_->GetConfig().vocab_size ||
        logits_count >
            std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        gpu_payload_bytes == 0 ||
        speculative != (verifier_payload_bytes != 0) ||
        total_bytes != payload.size()) {
      throw std::invalid_argument(
          "Qwen persistent snapshot metadata is invalid");
    }
    std::vector<TextRunnerToken> pending_tokens;
    for (std::size_t i = 0; i < kQwenMaxResumeTokens; ++i) {
      const auto token = GetLittleEndian<std::uint32_t>(
          payload, 72 + i * sizeof(std::uint32_t));
      if (i < pending_count) {
        if (token >= model_->GetConfig().vocab_size)
          throw std::invalid_argument("Qwen snapshot pending token is invalid");
        pending_tokens.push_back(token);
      } else if (token != 0) {
        throw std::invalid_argument(
            "Qwen snapshot has nonzero reserved tokens");
      }
    }
    const std::size_t logits_bytes = logits_count * sizeof(float);
    const std::size_t gpu_offset = kQwenPersistentSnapshotHeaderBytes;
    const std::size_t logits_offset =
        CheckedPersistentAdd(gpu_offset, gpu_payload_bytes);
    const std::size_t verifier_offset =
        CheckedPersistentAdd(logits_offset, logits_bytes);
    if (CheckedPersistentAdd(verifier_offset, verifier_payload_bytes) !=
        payload.size()) {
      throw std::invalid_argument(
          "Qwen persistent snapshot payload size is invalid");
    }

    std::vector<float> frontier_logits(logits_count);
    std::memcpy(frontier_logits.data(), payload.data() + logits_offset,
                logits_bytes);
    restored.executor().RestoreCompactSnapshot(
        payload.subspan(gpu_offset, gpu_payload_bytes),
        static_cast<std::uint32_t>(position));
    if (speculative) {
      restored.RestoreVerifierPersistentSnapshot(
          payload.subspan(verifier_offset, verifier_payload_bytes));
    }
    restored.set_position(position);
    restored.set_frontier(frontier);
    restored.RestoreFrontierLogits(std::move(frontier_logits));
    restored.RestorePendingTokens(pending_tokens);
  }

private:
  [[nodiscard]] std::size_t MaximumDecodeBatchWidth() const noexcept {
    if (model_->GetConfig().flint || model_->GetConfig().llama) {
      return 1;
    }
    const std::size_t rows_per_session = SpeculationEnabled() ? 8 : 1;
    return std::clamp<std::size_t>(max_context_ / rows_per_session, 1, 8);
  }

  mutable models::qwen::vision::ImageCache image_cache_;
  std::shared_ptr<const hip::QwenGpuModel> model_;
  std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model_;
  std::shared_ptr<const hip::QwenMtpGpuModel> mtp_model_;
  std::uint32_t max_context_;
  speculative::SpeculativeOptions speculative_options_;
  hip::QwenExecutionPolicy execution_policy_;
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
};

std::vector<TextRunnerToken> DeepSeekRunnerTokens(std::span<const int> tokens) {
  std::vector<TextRunnerToken> converted;
  converted.reserve(tokens.size());
  for (const int token : tokens) {
    if (token < 0) {
      throw std::invalid_argument("DeepSeek token ID must not be negative");
    }
    converted.push_back(static_cast<TextRunnerToken>(token));
  }
  return converted;
}

std::vector<int> DeepSeekEngineTokens(std::span<const TextRunnerToken> tokens) {
  std::vector<int> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("DeepSeek token ID exceeds engine range");
    }
    converted.push_back(static_cast<int>(token));
  }
  return converted;
}

std::string_view ChatRoleName(tokenization::ChatRole role) {
  switch (role) {
    case tokenization::ChatRole::kSystem:
      return "system";
    case tokenization::ChatRole::kDeveloper:
      return "developer";
    case tokenization::ChatRole::kAssistant:
      return "assistant";
    case tokenization::ChatRole::kTool:
      return "tool";
    case tokenization::ChatRole::kUser:
      return "user";
  }
  return "user";
}

class DeepSeekTextRunnerState final : public TextRunnerState {
public:
  DeepSeekTextRunnerState(
      const std::shared_ptr<models::deepseek_v4_flash::Model>& model,
      std::uint32_t max_context, bool use_dspark) {
    std::string error;
    session_ = model->CreateSession(
        use_dspark ? superroc::core::SessionMode::kSpeculative
                   : superroc::core::SessionMode::kAutoregressive,
        max_context, &error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create DeepSeek session: " + error);
    }
  }

  void SetCancellationCheck(const CancellationCheck& is_cancelled) override {
    session_->SetCancellationCheck(is_cancelled);
  }

  void Invalidate() noexcept override {
    session_->SetCancellationCheck({});
    session_->Invalidate();
    position_ = 0;
  }

  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    const std::uint64_t bytes = session_->PayloadBytes();
    if (bytes == 0 || bytes > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::size_t>::max())) {
      return {};
    }
    return {
        .per_request_state_bytes = static_cast<std::size_t>(bytes),
        .temporary_scratch_bytes = std::nullopt,
    };
  }

  [[nodiscard]] models::deepseek_v4_flash::Session& session() const {
    return *session_;
  }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  std::unique_ptr<models::deepseek_v4_flash::Session> session_;
  std::size_t position_{0};
};

class DeepSeekTextRunnerSnapshot final : public TextRunnerSnapshot {
public:
  DeepSeekTextRunnerSnapshot(
      std::shared_ptr<models::deepseek_v4_flash::Model> model,
      std::unique_ptr<models::deepseek_v4_flash::SessionSnapshot> snapshot,
      std::size_t position)
      : model(std::move(model)),
        snapshot(std::move(snapshot)),
        position(position) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    if (snapshot == nullptr ||
        snapshot->SizeBytes() > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::size_t>::max())) {
      return 0;
    }
    return static_cast<std::size_t>(snapshot->SizeBytes());
  }

  std::shared_ptr<models::deepseek_v4_flash::Model> model;
  std::unique_ptr<models::deepseek_v4_flash::SessionSnapshot> snapshot;
  std::size_t position;
};

DeepSeekTextRunnerState& RequireDeepSeekState(TextRunnerState& state) {
  auto* deepseek = dynamic_cast<DeepSeekTextRunnerState*>(&state);
  if (deepseek == nullptr) {
    throw std::logic_error("text runner state is not DeepSeek");
  }
  return *deepseek;
}

const DeepSeekTextRunnerState& RequireDeepSeekState(
    const TextRunnerState& state) {
  const auto* deepseek = dynamic_cast<const DeepSeekTextRunnerState*>(&state);
  if (deepseek == nullptr) {
    throw std::logic_error("text runner state is not DeepSeek");
  }
  return *deepseek;
}

class DeepSeekTextRunner final : public HipTextModelRunner {
public:
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kDeepSeek;
  }
  DeepSeekTextRunner(std::shared_ptr<models::deepseek_v4_flash::Model> model,
                     std::uint32_t max_context, bool use_dspark,
                     std::uint32_t max_draft_tokens,
                     std::string artifact_fingerprint = {},
                     std::string support_fingerprint = {})
      : model_(std::move(model)),
        max_context_(max_context),
        use_dspark_(use_dspark),
        max_draft_tokens_(std::max(max_draft_tokens, 1u)) {
    if (!artifact_fingerprint.empty()) {
      if (use_dspark_ && !IsSha256Hex(support_fingerprint)) {
        throw std::invalid_argument(
            "DSpark disk cache requires a support SHA-256 fingerprint");
      }
      persistence_ = TextRunnerPersistenceDescriptor{
          .compatibility_identity = DeepSeekCompatibilityIdentity(
              artifact_fingerprint,
              use_dspark_ ? support_fingerprint : std::string{}, max_context_,
              max_draft_tokens_),
          .payload_version = DS4_SESSION_PAYLOAD_VERSION,
      };
    }
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    const bool dspark = use_dspark_;
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kDeepSeekStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = dspark,
                .batched_multi_token_decode = dspark,
                .batched_multi_token_decode_max_width = 8u,
                .prefix_reuse = true,
            },
        .persistence = persistence_,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      capacity = free_bytes;
    }
    return {
        .resident_weights_bytes = std::nullopt,
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = std::nullopt,
        .temporary_scratch_bytes = std::nullopt,
        .retained_snapshot_capacity_bytes =
            TextRunnerRamCacheOptions::kAutomaticMaxBytes,
        .retained_snapshot_ceiling_bytes = SystemSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {
            .kind = TextExecutionPlanKind::kSerial,
            .physical_width = 1,
        },
    };
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back({
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = width,
      });
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->VocabSize(), [this](std::uint32_t id) {
          return sampling::ConstraintVocabulary::Piece{
              model_->DecodeToken(static_cast<int>(id)),
              model_->IsStopToken(static_cast<int>(id))};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(model_->VocabSize());
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return DeepSeekRunnerTokens(model_->Tokenize(text));
  }
  [[nodiscard]] std::vector<TextRunnerToken> TokenizeControl(
      std::string_view text) const override {
    return DeepSeekRunnerTokens(model_->Tokenize(text, true));
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    std::vector<models::deepseek_v4_flash::ChatMessage> messages;
    messages.reserve(request.messages.size());
    for (const auto& message : request.messages) {
      models::deepseek_v4_flash::ChatMessage converted{
          .role = std::string(ChatRoleName(message.role)),
          .content = message.content + message.framing_suffix,
          .reasoning_content = message.thought,
          .tool_calls = {},
          .tool_call_id = message.tool_call_id,
      };
      converted.tool_calls.reserve(message.tool_calls.size());
      for (const auto& call : message.tool_calls) {
        models::deepseek_v4_flash::ChatMessage::ToolCall converted_call{
            .name = call.name,
            .arguments = {},
            .id = call.id,
        };
        converted_call.arguments.reserve(call.arguments.size());
        for (const auto& argument : call.arguments) {
          converted_call.arguments.push_back({
              .name = argument.name,
              .value = argument.value,
              .is_string = argument.is_string,
          });
        }
        converted.tool_calls.push_back(std::move(converted_call));
      }
      messages.push_back(std::move(converted));
    }

    std::vector<models::deepseek_v4_flash::ChatTool> tools;
    if (request.tool_choice != ChatRequest::ToolChoice::kNone) {
      tools.reserve(request.tools.size());
      for (const auto& tool : request.tools) {
        tools.push_back({
            .name = tool.name,
            .description = tool.description,
            .parameters_json = tool.parameters_json,
            .definition_json =
                tool.definition_json.empty()
                    ? std::string{}
                    : json::parse(tool.definition_json)["function"].dump(),
        });
      }
    }
    auto options = models::deepseek_v4_flash::ResolveDeepSeekChatOptions(
        request.reasoning);
    options.tools_present =
        !request.tools.empty() &&
        request.tool_choice != ChatRequest::ToolChoice::kNone;
    options.require_tool_call =
        request.tool_choice == ChatRequest::ToolChoice::kRequired;
    auto tokens =
        DeepSeekRunnerTokens(model_->EncodeChat(messages, tools, options));
    if (tokens.empty()) {
      return std::nullopt;
    }
    return tokens;
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return models::deepseek_v4_flash::ResolveDeepSeekChatOptions(
               request.reasoning)
                   .enable_thinking
               ? TextGenerationBackend::InitialOutputState::kReasoning
               : TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    auto prepared = TextModelRunner::PreparePrompt(request);
    if (prepared) {
      // DeepSeek joins adjacent user/tool messages into one user block.
      // A new user turn after a discarded assistant can therefore also change
      // the final BPE token before the generation suffix (e.g. ">\n\n").
      // Find the exact stable frontier using the renderer/tokenizer instead
      // of assuming the suffix alone is the only changed part of the prompt.
      auto continuation = request;
      continuation.messages.emplace_back(tokenization::ChatRole::kUser, "");
      const auto continued = RenderAndTokenize(continuation);
      if (continued) {
        prepared->cache_prefix_tokens = static_cast<std::size_t>(
            std::mismatch(prepared->tokens.begin(), prepared->tokens.end(),
                          continued->begin(), continued->end())
                .first -
            prepared->tokens.begin());
      }
    }
    return prepared;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    std::string text;
    for (const TextRunnerToken token : tokens) {
      if (token >
          static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("DeepSeek token ID exceeds engine range");
      }
      text += model_->DecodeToken(static_cast<int>(token));
    }
    return text;
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<DeepSeekTextRunnerState>(model_, max_context_,
                                                     use_dspark_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() != prefix.size()) {
      throw std::logic_error(
          "DeepSeek reused prefix does not match checkpoint");
    }
    deepseek.session().BeginRequest();
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (offset != deepseek.position()) {
      throw std::logic_error(
          "DeepSeek prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("DeepSeek prefill has no remaining input");
    }

    const std::size_t consumed =
        std::min<std::size_t>({max_input_tokens, prompt.size() - offset,
                               deepseek.session().PrefillCapacity()});
    const std::size_t next_position = offset + consumed;
    const auto prefix = DeepSeekEngineTokens(prompt.first(next_position));
    std::string error;
    if (!deepseek.session().Sync(prefix, &error)) {
      throw std::runtime_error("DeepSeek prefill failed: " + error);
    }
    deepseek.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() >= max_context_)
      return {.stop = true, .piece = {}};
    // A sampled DSpark cycle may have drawn the next token already.
    int token = deepseek.session().TakePendingDsparkToken();
    const bool pending_dspark = token >= 0;
    std::optional<sampling::CapturedTokenLogprobs> logprobs;
    if (token < 0) {
      std::string error;
      const auto logits = deepseek.session().CopyLogits(&error);
      if (logits.empty()) {
        throw std::runtime_error("DeepSeek token selection failed: " + error);
      }
      token = static_cast<int>(sampler.Sample(logits));
      if (sampler.logprobs_enabled()) {
        const auto top_n = sampler.logprobs_top_n();
        logprobs = sampler.CaptureLogprobs(
            logits, static_cast<sampling::TokenId>(token), top_n);
      }
    }
    if (deepseek.stop_at_eos() && model_->IsStopToken(token)) {
      return {
          .stop = true,
          .token = 0,
          .piece = {},
      };
    }
    if (logprobs.has_value()) {
      logprobs->text = model_->DecodeToken(token);
    } else if (pending_dspark && sampler.logprobs_enabled()) {
      // The sampled DSpark cycle drew this token without the sampler, so
      // there is no distribution to quote: report a placeholder, only when
      // armed, so streaming and non-streaming stay parallel.
      const std::string piece = model_->DecodeToken(token);
      logprobs = sampling::CapturedTokenLogprobs{
          .token = static_cast<sampling::TokenId>(token),
          .logprob = std::numeric_limits<double>::quiet_NaN(),
          .text = piece};
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = model_->DecodeToken(token),
        .logprobs = std::move(logprobs),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("DeepSeek token ID exceeds engine range");
    }
    auto& deepseek = RequireDeepSeekState(state);
    std::string error;
    if (!deepseek.session().Evaluate(static_cast<int>(token), &error)) {
      throw std::runtime_error("DeepSeek decode failed: " + error);
    }
    deepseek.set_position(deepseek.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() >= max_context_)
      return TextDecodeSelection{.stop = true, .piece = {}};
    std::string error;
    const auto logits = deepseek.session().CopyLogits(&error);
    if (logits.empty())
      throw std::runtime_error("DeepSeek token preview failed: " + error);
    const auto token = sampler.Sample(logits);
    std::optional<sampling::CapturedTokenLogprobs> logprobs;
    if (sampler.logprobs_enabled()) {
      const auto top_n = sampler.logprobs_top_n();
      logprobs = sampler.CaptureLogprobs(logits, token, top_n);
      logprobs->text = model_->DecodeToken(static_cast<int>(token));
    }
    return TextDecodeSelection{
        .stop = deepseek.stop_at_eos() &&
                model_->IsStopToken(static_cast<int>(token)),
        .token = token,
        .piece = model_->DecodeToken(static_cast<int>(token)),
        .logprobs = std::move(logprobs),
    };
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_dspark_) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "DeepSeek DSpark decode budget must be at least one token");
    }

    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() >= max_context_)
      return {.selections = {}, .stop = true};
    const auto stats_before = deepseek.session().DsparkStatistics();
    std::optional<models::deepseek_v4_flash::DsparkSamplerBridge> bridge;
    // A fired loop guard forces its end through the bridge's sampler, exactly
    // as a spent reasoning budget would; the bridge already defers to the
    // host sampler while ForcedToken() is set. An armed word ban installs the
    // bridge too, so a banned device proposal is rejected through
    // CanSelectArgmax and redrawn masked; unbanned steps keep the device
    // argmax with one lookup per token and no row download.
    if (!sampler.config().can_use_unmodified_argmax() ||
        sampler.ForcedToken().has_value() || sampler.ReasoningBanActive()) {
      bridge.emplace(sampler);
    }
    std::vector<int> emitted;
    std::string error;
    if (!deepseek.session().DsparkStep(
            max_tokens, max_draft_tokens_, &emitted, &error,
            bridge ? bridge->hook() : nullptr, deepseek.stop_at_eos())) {
      throw std::runtime_error("DeepSeek DSpark decode failed: " + error);
    }
    if (emitted.empty()) {
      throw std::runtime_error("DeepSeek DSpark decode produced no tokens");
    }
    if (bridge) {
      sampler.SetRngState(bridge->rng_state());
    }

    TextDecodeStep step;
    deepseek.set_position(deepseek.session().Position());
    step.selections.reserve(emitted.size());
    for (const int token : emitted) {
      if (deepseek.stop_at_eos() && model_->IsStopToken(token)) {
        step.stop = true;
        break;
      }
      const std::string piece = model_->DecodeToken(token);
      std::optional<sampling::CapturedTokenLogprobs> block_logprobs;
      if (sampler.logprobs_enabled()) {
        // A sampled DSpark cycle drew these tokens without the sampler, so
        // there is no distribution to quote: report a placeholder per token,
        // only when armed, so streaming and non-streaming stay parallel.
        block_logprobs = sampling::CapturedTokenLogprobs{
            .token = static_cast<sampling::TokenId>(token),
            .logprob = std::numeric_limits<double>::quiet_NaN(),
            .text = piece};
      }
      step.selections.push_back({
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = piece,
          .logprobs = std::move(block_logprobs),
      });
    }
    const auto stats_after = deepseek.session().DsparkStatistics();
    step.draft_rounds = stats_after.steps - stats_before.steps;
    step.draft_tokens =
        stats_after.support_drafted - stats_before.support_drafted;
    step.draft_accepted_tokens =
        stats_after.support_accepted - stats_before.support_accepted;
    return step;
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (!use_dspark_ || decodes.size() < 2 || decodes.size() > 8) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    if (std::any_of(decodes.begin(), decodes.end(), [&](const auto& item) {
          return RequireDeepSeekState(item.state.get()).position() >=
                 max_context_;
        })) {
      std::vector<TextRunnerDecode> active;
      std::vector<std::size_t> active_indices;
      std::vector<TextDecodeStep> steps(decodes.size());
      for (std::size_t i = 0; i < decodes.size(); ++i) {
        if (RequireDeepSeekState(decodes[i].state.get()).position() >=
            max_context_) {
          steps[i].stop = true;
        } else {
          active.push_back(decodes[i]);
          active_indices.push_back(i);
        }
      }
      if (!active.empty()) {
        auto active_steps = DecodeBatch(active);
        for (std::size_t i = 0; i < active.size(); ++i)
          steps[active_indices[i]] = std::move(active_steps[i]);
      }
      return steps;
    }
    // Greedy and sampled requests share one DSpark cohort; the runtime keeps
    // the greedy verifier for items without a sampler.
    std::array<models::deepseek_v4_flash::SessionDsparkBatchItem, 8> items{};
    std::array<DeepSeekTextRunnerState*, 8> states{};
    std::array<models::deepseek_v4_flash::Session::DsparkStats, 8>
        stats_before{};
    std::array<std::vector<int>, 8> emitted{};
    std::array<std::optional<models::deepseek_v4_flash::DsparkSamplerBridge>, 8>
        bridges{};
    for (std::size_t index = 0; index < decodes.size(); ++index) {
      auto& deepseek = RequireDeepSeekState(decodes[index].state.get());
      states[index] = &deepseek;
      stats_before[index] = deepseek.session().DsparkStatistics();
      auto& sampler = decodes[index].sampler.get();
      if (!sampler.config().can_use_unmodified_argmax() ||
          sampler.ForcedToken().has_value() || sampler.ReasoningBanActive()) {
        bridges[index].emplace(sampler);
      }
      items[index] = {
          .session = &deepseek.session(),
          .max_tokens = decodes[index].max_tokens,
          .max_draft_tokens = max_draft_tokens_,
          .emitted = &emitted[index],
          .sampler = bridges[index] ? bridges[index]->hook() : nullptr,
          .stop_at_eos = states[index]->stop_at_eos(),
      };
    }

    std::string error;
    bool interrupted = false;
    if (!model_->DsparkStepBatch(
            std::span<const models::deepseek_v4_flash::SessionDsparkBatchItem>(
                items.data(), decodes.size()),
            &error, &interrupted)) {
      // Cancellation is rejected before sampling or device-state mutation.
      // Isolate the cancelled row rather than failing every client in its
      // batch. Ordinary cycles retain the shared physical batch.
      if (interrupted)
        return TextModelRunner::DecodeBatch(decodes);
      throw std::runtime_error("DeepSeek DSpark batch decode failed: " + error);
    }

    std::vector<TextDecodeStep> steps(decodes.size());
    for (std::size_t index = 0; index < decodes.size(); ++index) {
      if (emitted[index].empty()) {
        throw std::runtime_error("DeepSeek DSpark batch produced no tokens");
      }
      if (bridges[index]) {
        decodes[index].sampler.get().SetRngState(bridges[index]->rng_state());
      }
      auto& step = steps[index];
      step.execution_plan = {
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = decodes.size(),
      };
      states[index]->set_position(states[index]->session().Position());
      step.selections.reserve(emitted[index].size());
      for (const int token : emitted[index]) {
        if (states[index]->stop_at_eos() && model_->IsStopToken(token)) {
          step.stop = true;
          break;
        }
        step.selections.push_back({
            .stop = false,
            .token = static_cast<TextRunnerToken>(token),
            .piece = model_->DecodeToken(token),
        });
      }
      const auto stats_after = states[index]->session().DsparkStatistics();
      step.draft_rounds = stats_after.steps - stats_before[index].steps;
      step.draft_tokens =
          stats_after.support_drafted - stats_before[index].support_drafted;
      step.draft_accepted_tokens =
          stats_after.support_accepted - stats_before[index].support_accepted;
    }
    return steps;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2 || advances.size() > 8) {
      throw std::invalid_argument(
          "DeepSeek batched decode requires two to eight sessions");
    }

    std::array<models::deepseek_v4_flash::SessionBatchItem, 8> items{};
    std::array<DeepSeekTextRunnerState*, 8> states{};
    std::size_t item_count = 0;
    for (const auto& advance : advances) {
      if (advance.token >
          static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("DeepSeek token ID exceeds engine range");
      }
      auto& deepseek = RequireDeepSeekState(advance.state.get());
      states[item_count] = &deepseek;
      items[item_count] = {
          .session = &deepseek.session(),
          .token = static_cast<int>(advance.token),
      };
      ++item_count;
    }

    std::string error;
    bool interrupted = false;
    if (!model_->EvaluateBatch(
            std::span<const models::deepseek_v4_flash::SessionBatchItem>(
                items.data(), item_count),
            &error, &interrupted)) {
      if (interrupted) {
        TextModelRunner::AdvanceBatch(advances);
        return;
      }
      throw std::runtime_error("DeepSeek batch decode failed: " + error);
    }
    for (std::size_t index = 0; index < item_count; ++index) {
      states[index]->set_position(states[index]->position() + 1);
    }
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireDeepSeekState(state).position();
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const override {
    const auto& session = RequireDeepSeekState(state).session();
    if (use_dspark_ && session.DsparkStatistics().context_tokens !=
                           static_cast<uint32_t>(session.Position())) {
      throw std::runtime_error("DSpark prefix lacks complete support state");
    }
    const std::uint64_t bytes = session.PayloadBytes();
    if (bytes == 0 || bytes > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::size_t>::max())) {
      throw std::overflow_error("DeepSeek snapshot size is unavailable");
    }
    return static_cast<std::size_t>(bytes);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    const auto& deepseek = RequireDeepSeekState(state);
    std::string error;
    auto snapshot = deepseek.session().SaveSnapshot(&error);
    if (snapshot == nullptr) {
      throw std::runtime_error("DeepSeek snapshot failed: " + error);
    }
    return std::make_unique<DeepSeekTextRunnerSnapshot>(
        model_, std::move(snapshot), deepseek.position());
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto* deepseek_snapshot =
        dynamic_cast<const DeepSeekTextRunnerSnapshot*>(&snapshot);
    if (deepseek_snapshot == nullptr ||
        deepseek_snapshot->model.get() != model_.get() ||
        deepseek_snapshot->snapshot == nullptr) {
      throw std::invalid_argument(
          "DeepSeek snapshot does not belong to this model");
    }
    auto& restored = RequireDeepSeekState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(*deepseek_snapshot->snapshot,
                                            &error)) {
      throw std::runtime_error("DeepSeek snapshot restore failed: " + error);
    }
    restored.set_position(deepseek_snapshot->position);
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    const auto* deepseek_snapshot =
        dynamic_cast<const DeepSeekTextRunnerSnapshot*>(&snapshot);
    if (deepseek_snapshot == nullptr ||
        deepseek_snapshot->model.get() != model_.get() ||
        deepseek_snapshot->snapshot == nullptr) {
      throw std::invalid_argument(
          "DeepSeek persistent snapshot does not belong to this model");
    }
    return deepseek_snapshot->PayloadBytes();
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto* deepseek_snapshot =
        dynamic_cast<const DeepSeekTextRunnerSnapshot*>(&snapshot);
    if (deepseek_snapshot == nullptr ||
        deepseek_snapshot->model.get() != model_.get() ||
        deepseek_snapshot->snapshot == nullptr ||
        destination.size() != deepseek_snapshot->PayloadBytes() ||
        !deepseek_snapshot->snapshot->CopyTo(destination)) {
      throw std::invalid_argument(
          "DeepSeek persistent snapshot serialization failed");
    }
    return destination.size();
  }

  void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                const SnapshotSink& sink) const override {
    (void)PersistentSnapshotPayloadBytes(snapshot);
    sink(dynamic_cast<const DeepSeekTextRunnerSnapshot&>(snapshot)
             .snapshot->bytes());
  }

  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    auto& restored = RequireDeepSeekState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(payload, &error)) {
      throw std::runtime_error("DeepSeek persistent snapshot restore failed: " +
                               error);
    }
    const int position = restored.session().Position();
    if (position <= 0 || static_cast<std::uint64_t>(position) > max_context_) {
      restored.session().Invalidate();
      throw std::runtime_error(
          "DeepSeek persistent snapshot restored an invalid position");
    }
    restored.set_position(static_cast<std::size_t>(position));
  }

private:
  std::shared_ptr<models::deepseek_v4_flash::Model> model_;
  std::uint32_t max_context_;
  bool use_dspark_;
  std::uint32_t max_draft_tokens_;
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
};

/// Derives the content identity that keys persistent continuations for one
/// GGUF artifact. Full digests are cached by the open file metadata.
bool FingerprintArtifact(std::string_view label, const core::GgufReader& reader,
                         std::string* fingerprint, std::string* error) {
  const auto start = std::chrono::steady_clock::now();
  Logger::Info("loader", "event=load_phase phase=artifact_identity model=" +
                             std::string(label));
  try {
    *fingerprint = core::GgufIdentityHex(reader);
  } catch (const std::exception& exception) {
    SetError(error, std::string("Failed to fingerprint ") + std::string(label) +
                        " GGUF: " + exception.what());
    return false;
  }
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count();
  std::ostringstream message;
  message << "Fingerprinted " << label << " artifact ("
          << core::kGgufIdentityScheme << ", " << reader.GetTensorCount()
          << " tensors) in " << std::fixed << std::setprecision(0) << elapsed_ms
          << " ms";
  Logger::Info("engine", message.str());
  return true;
}

bool FingerprintArtifactFile(std::string_view label,
                             const std::filesystem::path& path,
                             std::string* fingerprint, std::string* error) {
  // Only draft files are fingerprinted here, and a draft is read as it is.
  std::string open_error;
  const auto reader = core::GgufReader::OpenFile(
      path, &open_error, core::GgufReader::FileRole::kSidecar);
  if (reader == nullptr) {
    SetError(error, std::string("Failed to open ") + std::string(label) +
                        " GGUF for fingerprinting: " + open_error);
    return false;
  }
  return FingerprintArtifact(label, *reader, fingerprint, error);
}

#endif

#if defined(ENGINE_ENABLE_HIP)
// Each Flash-Next request owns its recurrent/KV state and snapshots. Target
// projections share batches; sampling and rollback remain request-local.
constexpr std::string_view kQwenFlashNextStateAbi =
    "qwen38-flash-next-rocm-session-v1";

using QwenFlashNextModel = models::qwen38_flash_next::Model;
using QwenFlashNextSession = models::qwen38_flash_next::Session;

std::vector<std::int32_t> QwenFlashNextEngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

/// Arms the session's logits hook for one Sync.
struct QwenFlashNextLogitsHook {
  QwenFlashNextSession& session;
  QwenFlashNextLogitsHook(QwenFlashNextSession& s,
                          const std::function<void()>* hook) noexcept
      : session(s) {
    session.SetLogitsHook(hook);
  }
  ~QwenFlashNextLogitsHook() { session.SetLogitsHook(nullptr); }
};

class QwenFlashNextTextRunnerState final : public TextRunnerState {
public:
  QwenFlashNextTextRunnerState(QwenFlashNextSession& session,
                               cache::Adapter& adapter, cache::Slot& slot)
      : session_(&session), adapter_(&adapter), slot_(&slot) {}

  void Invalidate() noexcept override {
    session_->SetCancellationCheck({});
    (void)adapter_->Invalidate(*slot_);
    position_ = 0;
  }
  void SetCancellationCheck(const CancellationCheck& check) override {
    session_->SetCancellationCheck(check);
  }
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {.per_request_state_bytes = session_->AllocatedBytes(),
            .temporary_scratch_bytes = 0};
  }

  [[nodiscard]] QwenFlashNextSession& session() const { return *session_; }
  [[nodiscard]] models::qwen38_flash_next::ContinuationAdapter& adapter()
      const {
    return dynamic_cast<models::qwen38_flash_next::ContinuationAdapter&>(
        *adapter_);
  }
  [[nodiscard]] cache::Slot& slot() const { return *slot_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  QwenFlashNextSession* session_{};
  cache::Adapter* adapter_{};
  cache::Slot* slot_{};
  std::size_t position_{0};
};

QwenFlashNextTextRunnerState& RequireQwenFlashNextState(
    TextRunnerState& state) {
  auto* qfn = dynamic_cast<QwenFlashNextTextRunnerState*>(&state);
  if (qfn == nullptr) {
    throw std::logic_error("text runner state is not Qwen3.8-Flash-Next");
  }
  return *qfn;
}

const QwenFlashNextTextRunnerState& RequireQwenFlashNextState(
    const TextRunnerState& state) {
  const auto* qfn = dynamic_cast<const QwenFlashNextTextRunnerState*>(&state);
  if (qfn == nullptr) {
    throw std::logic_error("text runner state is not Qwen3.8-Flash-Next");
  }
  return *qfn;
}

class QwenFlashNextTextRunner final : public HipTextModelRunner {
public:
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kQwen;
  }
  QwenFlashNextTextRunner(std::shared_ptr<QwenFlashNextModel> model,
                          std::uint32_t max_context, bool use_mtp,
                          std::uint32_t max_draft_tokens,
                          std::string artifact_fingerprint = {},
                          std::string mtp_fingerprint = {})
      : model_(std::move(model)),
        max_context_(max_context),
        use_mtp_(use_mtp),
        max_draft_tokens_(max_draft_tokens) {
    // The typed in-process API owns an immutable loaded model but need not
    // supply file fingerprints when disk caching is disabled. Its model
    // lifetime is an exact process-local identity; durable caches still
    // require the validated full artifact fingerprints below.
    if (!artifact_fingerprint.empty()) {
      persistence_ = TextRunnerPersistenceDescriptor{
          .compatibility_identity = QwenFlashNextCompatibilityIdentity(
              artifact_fingerprint, use_mtp_ ? mtp_fingerprint : std::string{},
              use_mtp_, max_context_, max_draft_tokens_,
              model_->DecodeConcurrency()),
          .payload_version =
              models::qwen38_flash_next::Session::kSnapshotPayloadVersion,
      };
      component_identity_ = persistence_->compatibility_identity;
    } else {
      std::ostringstream identity;
      identity << "schema=superroc-process-text-continuation-v1\n"
               << "model_instance=" << model_.get() << '\n'
               << "context_tokens=" << max_context_ << '\n'
               << "mtp=" << use_mtp_ << '\n'
               << "draft_max_tokens=" << max_draft_tokens_ << '\n'
               << "draft_cost_concurrency=" << model_->DecodeConcurrency()
               << '\n';
      const auto canonical = identity.str();
      component_identity_.assign(canonical.begin(), canonical.end());
    }
  }

  [[nodiscard]] bool UsesComponentCache() const override { return true; }
  [[nodiscard]] std::unique_ptr<ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t ram) const override {
    struct Backing {
      hip::TransferPool streams{64};
      std::unique_ptr<hip::CommittedBackingPool> pool;
      std::map<std::uint32_t, std::size_t> component_classes;
    };
    auto result = std::make_unique<ComponentCacheResources>();
    auto adapter =
        std::make_unique<models::qwen38_flash_next::ContinuationAdapter>(
            model_,
            use_mtp_ ? core::SessionMode::kSpeculative
                     : core::SessionMode::kAutoregressive,
            max_context_, component_identity_);
    (void)ram;  // Backing sizing follows actual slot allocation below.
    result->adapter = std::move(adapter);
    auto backing = std::make_shared<Backing>();
    result->transfer_batch_size = 16;
    const auto rounded = [](std::size_t bytes) {
      return (bytes + 4095) / 4096 * 4096;
    };
    for (const auto& d : result->adapter->Components()) {
      const auto bytes = rounded(d.kind == cache::ComponentKind::kPrivateState
                                     ? d.state_bytes
                                     : d.row_bytes * d.rows_per_chunk);
      backing->component_classes.emplace(d.id.value, bytes);
    }
    // The remainder belongs to index, histories, disk catalog and queue
    // metadata. The pools charge their own per-block allocator headroom.
    result->commit_backing = [backing, &ledger,
                              adapter =
                                  result->adapter.get()](std::size_t budget) {
      const auto plan =
          static_cast<models::qwen38_flash_next::ContinuationAdapter*>(adapter)
              ->BackingPlan(budget);
      if (plan.size() != 1)
        throw std::invalid_argument("insufficient component backing budget");
      const auto& allocation = plan.front();
      // Pages commit as captures need them. A payload cannot span segments,
      // so segments of about 128 MiB bound registration pauses and stranded
      // edge space, fitting at least two of the largest payloads. Prefill
      // chunks prepare backing before GPU work, after any fallback capture.
      // Growth stops when the host keeps less than its reserve.
      std::size_t largest = 0;
      for (const auto& [id, bytes] : backing->component_classes)
        largest = std::max(largest, bytes);
      const auto block = allocation.block_bytes;
      const auto target =
          std::max<std::size_t>(std::size_t{128} << 20, 2 * largest) / block;
      const auto parts =
          std::max<std::size_t>((allocation.blocks + target - 1) / target, 1);
      const auto segment = (allocation.blocks + parts - 1) / parts * block;
      struct Limited {
        std::mutex mutex;
        std::chrono::steady_clock::time_point refused;
        bool active{};
      };
      backing->pool = std::make_unique<hip::CommittedBackingPool>(
          ledger,
          hip::BackingPoolConfig{
              allocation.blocks * allocation.block_bytes +
                  allocation.metadata_bytes,
              allocation.block_bytes, allocation.metadata_bytes, segment,
              [limited = std::make_shared<Limited>()](std::size_t bytes) {
                const std::lock_guard lock(limited->mutex);
                const auto now = std::chrono::steady_clock::now();
                // Pressure retries ask once per capture; recheck at most
                // once a second while the host is short.
                if (limited->active &&
                    now - limited->refused < std::chrono::seconds(1))
                  return false;
                const auto ceiling = HostSnapshotCeilingBytes();
                // Pinned host pages come out of the same device heap that
                // session and scratch allocations draw on; keep its dispatch
                // reserve so they do not fall back to managed memory.
                std::size_t heap_free = 0, heap_total = 0;
                const bool heap_fits =
                    hipMemGetInfo(&heap_free, &heap_total) == hipSuccess &&
                    hip::HeapFitsAboveReserve(heap_free, bytes);
                if (ceiling >= bytes && heap_fits) {
                  limited->active = false;
                  return true;
                }
                if (!limited->active)
                  Logger::Warn(
                      "cache",
                      "schema=component-cache-v1 "
                      "event=backing_limited reason=" +
                          std::string(heap_fits ? "host_reserve "
                                                : "device_heap_reserve ") +
                          "segment_bytes=" + std::to_string(bytes) +
                          " host_headroom_bytes=" + std::to_string(ceiling) +
                          " device_heap_free_bytes=" +
                          std::to_string(heap_free));
                limited->active = true;
                limited->refused = now;
                return false;
              }});
      // Prepare a bounded reserve before readiness. The first shared-prefix
      // capture needs more than one segment; preparing 2 GiB avoids moving
      // registration onto that request's capture or decode path.
      backing->pool->GrowAhead(2ULL << 30);
      backing->pool->WaitForGrowth();
      Logger::Info(
          "cache",
          "schema=component-cache-v1 event=backing_prepared committed_bytes=" +
              std::to_string(backing->pool->CommittedBytes()) +
              " long_term_pinned_bytes=" +
              std::to_string(backing->pool->LongTermPinnedBytes()));
    };
    result->prepare_prefill = [backing](std::size_t tokens) {
      // Prepare after any fallback capture, alongside actual prefill work.
      // Short turns need a smaller reserve for private states and row tails.
      backing->pool->GrowAhead((tokens >= 512 ? 2ULL : 1ULL) << 30);
    };
    result->can_capture = [backing](auto payloads) {
      return backing->pool->CanFit(payloads);
    };
    result->owner = backing;
    result->stream = [backing]() -> std::unique_ptr<cache::Stream> {
      return backing->streams.TryAcquire();
    };
    result->allocate = [backing](const cache::PayloadRequest& request) {
      // Bound the logical capture, then assign whole shared backing pages.
      const auto capacity =
          backing->component_classes.at(request.component.value);
      if (request.bytes > capacity)
        throw std::logic_error("component capture exceeds its backing class");
      const auto category =
          request.category == cache::ResourceCategory::kBackingAssigned
              ? cache::ResourceCategory::kBackingAssigned
              : request.category;
      const auto segments = backing->pool->PageCommitAllocations();
      const auto started = std::chrono::steady_clock::now();
      auto block = backing->pool->TryAcquire(category, request.bytes);
      if (backing->pool->PageCommitAllocations() != segments)
        Logger::Info(
            "cache",
            "schema=component-cache-v1 event=backing_grown committed_bytes=" +
                std::to_string(backing->pool->CommittedBytes()) + " segments=" +
                std::to_string(backing->pool->PageCommitAllocations()) +
                " long_term_pinned_bytes=" +
                std::to_string(backing->pool->LongTermPinnedBytes()) +
                " acquire_ms=" +
                std::to_string(std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - started)
                                   .count()));
      if (!block)
        throw cache::ResourceExhausted();
      // Do not start registration between a capture's allocations and copies.
      // Each prefill chunk prepares its own reserve before GPU work.
      auto owner = std::make_shared<hip::BackingBlock>(std::move(*block));
      return ComponentCaptureBuffer{
          owner->Bytes(),
          [owner] {
            owner->Convert();
            return std::move(*owner).TakePayload();
          },
          backing->pool.get(),
          [owner, request](cache::SlotLease& lease, auto source) {
            return cache::Payload::Borrowed(std::move(*owner).Borrow(
                lease, request.component, request.first, request.end,
                request.bytes, source));
          }};
    };
    result->staging = [](std::size_t bytes) {
      auto allocation = std::make_shared<hip::HostAllocations>();
      void* data = nullptr;
      const auto status =
          allocation->Allocate(&data, bytes, hipHostMallocCoherent);
      if (status != hipSuccess)
        throw std::runtime_error(hipGetErrorString(status));
      auto owner = std::shared_ptr<void>(std::move(allocation), data);
      std::memset(data, 0, bytes);
      return cache::StagingAllocation{
          std::move(owner), {static_cast<std::uint8_t*>(data), bytes}};
    };
    result->copy_observed_rows = [](auto source, auto destination,
                                    cache::Stream& stream) {
      return dynamic_cast<hip::TransferStream&>(stream).Copy(
          destination.data(), source.data(), destination.size(),
          hip::CopyDirection::kDeviceToHost, destination.size());
    };
    return result;
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    auto& flash =
        dynamic_cast<models::qwen38_flash_next::ContinuationAdapter&>(adapter);
    return std::make_unique<QwenFlashNextTextRunnerState>(
        flash.GetSession(slot), adapter, slot);
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t position) const override {
    RequireQwenFlashNextState(state).set_position(position);
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kQwenFlashNextStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = use_mtp_,
                .batched_multi_token_decode = use_mtp_,
                .batched_multi_token_decode_max_width = use_mtp_ ? 8u : 0u,
                .prefix_reuse = true,
                .in_pass_checkpoint = true,
                .partial_prefill_retention = true,
                // Measured with qwen38_flash_next_split_probe: the session
                // state is the same bits for every split whose cuts fall on
                // multiples of 4 rows (the sparse attention's key blocks)
                // and whose passes hold 16 rows or more (below that the
                // projections take other kernels).
                .canonical_min_pass_rows = 16,
            },
        .persistence = persistence_,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      const auto deferred = model_->DeferredScratchBytes();
      capacity = free_bytes > deferred ? free_bytes - deferred : 0;
    }
    // Admission charges complete snapshot payloads. Flash-Next can retain
    // mutable state and protected K/V rows in private device storage.
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = model_->SessionBytes(
            use_mtp_ ? superroc::core::SessionMode::kSpeculative
                     : superroc::core::SessionMode::kAutoregressive,
            max_context_),
        // Runtime scratch is shared and already allocated at model load;
        // reserve its remaining lazy buffers once from aggregate capacity.
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes =
            TextRunnerRamCacheOptions::kAutomaticMaxBytes,
        .retained_snapshot_ceiling_bytes = SystemSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back(
          {.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->tokenizer().GetVocabSize(), [this](std::uint32_t id) {
          const auto& tokenizer = model_->tokenizer();
          return sampling::ConstraintVocabulary::Piece{
              tokenizer.DecodeTokenCopy(id), tokenizer.IsStopToken(id)};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(model_->tokenizer().GetVocabSize());
  }

  // llama.cpp's Qwen3-Coder format forces "\n</think>" and also ends
  // reasoning at a tool call.
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {"<think>", "\n</think>", {"</think>", "<tool_call>"}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->tokenizer().Encode(text);
  }

  [[nodiscard]] std::optional<TextRunnerToken> RawPromptBos() const override {
    return model_->tokenizer().RawPromptBos();
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return tokenization::QwenChatTemplate::RenderAndTokenize(
        model_->tokenizer(), request.messages,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        QwenChatOptions(request, max_context_));
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    return PrepareQwenPrompt(request, model_->tokenizer(),
                             model_->VisionEncoder(), max_context_,
                             image_cache_);
  }

  [[nodiscard]] std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const override {
    return CountQwenPromptTokens(request, model_->tokenizer(),
                                 model_->VisionEncoder(), max_context_,
                                 image_cache_);
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    RequireQwenFlashNextState(state).session().ConfigureVision(
        QwenPrompt(context));
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return QwenChatOptions(request, max_context_).enable_thinking
               ? TextGenerationBackend::InitialOutputState::kReasoning
               : TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->tokenizer().Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    throw std::logic_error(
        "Flash-Next execution slots are owned by the component cache");
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    const auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() != prefix.size()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next reused prefix does not match checkpoint");
    }
    qfn.session().ResetDraftPolicy();
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (offset != qfn.position()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next prefill has no remaining input");
    }
    const std::size_t remaining = prompt.size() - offset;
    const std::size_t consumed =
        std::min<std::size_t>({max_input_tokens, remaining,
                               remaining <= model_->PrefillThroughCapacity()
                                   ? model_->PrefillThroughCapacity()
                                   : model_->PrefillCapacity()});
    const std::size_t next_position = offset + consumed;
    const auto prefix = QwenFlashNextEngineTokens(prompt.first(next_position));
    // The next step most likely takes the same budget; its n-gram rows are
    // read during this one and used for the tokens it shares. A step ending
    // at a cache frontier continues into the pool's lookahead.
    const auto rest = next_position == prompt.size()
                          ? state.prefill_lookahead()
                          : prompt.subspan(next_position);
    const std::size_t after = rest.size();
    const auto next = QwenFlashNextEngineTokens(rest.subspan(
        0, std::min<std::size_t>({max_input_tokens, after,
                                  after <= model_->PrefillThroughCapacity()
                                      ? model_->PrefillThroughCapacity()
                                      : model_->PrefillCapacity()})));
    std::string error;
    const QwenFlashNextLogitsHook logits_hook{
        qfn.session(),
        next_position == prompt.size() ? state.prompt_logits_hook() : nullptr};
    if (!qfn.session().Sync(prefix, &error, next)) {
      qfn.set_position(0);
      throw std::runtime_error("Qwen3.8-Flash-Next prefill failed: " + error);
    }
    qfn.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget,
      std::size_t boundary) const override {
    const auto& qfn = RequireQwenFlashNextState(state);
    const auto consumed = std::min<std::size_t>(
        {budget, prompt.size() - offset, model_->PrefillThroughCapacity()});
    if (boundary <= offset + 1 || boundary >= offset + consumed ||
        offset + consumed - boundary > 8)
      return std::nullopt;
    const auto bytes = qfn.session().PrefillCheckpointBytes(
        static_cast<std::uint32_t>(boundary));
    return bytes ? std::optional<std::size_t>(bytes) : std::nullopt;
  }

  [[nodiscard]] TextPrefillStep PrefillThrough(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget, std::size_t boundary,
      std::unique_ptr<TextRunnerSnapshot>* checkpoint,
      const std::function<std::unique_ptr<TextRunnerSnapshot>()>& capture)
      const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (!checkpoint || !capture || offset != qfn.position())
      throw std::logic_error("invalid component in-pass checkpoint request");
    const auto consumed = std::min<std::size_t>(
        {budget, prompt.size() - offset, model_->PrefillThroughCapacity()});
    const auto next_position = offset + consumed;
    const auto prefix = QwenFlashNextEngineTokens(prompt.first(next_position));
    double capture_ms = 0;
    const QwenFlashNextLogitsHook logits_hook{
        qfn.session(),
        next_position == prompt.size() ? state.prompt_logits_hook() : nullptr};
    try {
      qfn.adapter().PrefillThrough(
          qfn.slot(), prefix, static_cast<std::uint32_t>(boundary), [&] {
            const auto started = std::chrono::steady_clock::now();
            try {
              *checkpoint = capture();
            } catch (...) {
              // Optional retention may refuse; the complete live prefill is
              // valid.
              checkpoint->reset();
            }
            capture_ms += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
          });
    } catch (...) {
      qfn.set_position(0);
      throw;
    }
    qfn.set_position(next_position);
    return {.consumed_tokens = consumed,
            .decode_ready = next_position == prompt.size(),
            .checkpoint_ms = capture_ms};
  }

  // Dual in-pass checkpoints: the
  // stable S keeps the text-tail-8 rule; the earlier P must sit at least 64
  // rows into the pass, at least 64 rows below S, and within the widened
  // 256-row tail of the pass end.
  [[nodiscard]] std::optional<PrefillCheckpointPair> PrefillCheckpointPairBytes(
      const TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget, std::size_t boundary_first,
      std::size_t boundary) const override {
    const auto& qfn = RequireQwenFlashNextState(state);
    const auto consumed = std::min<std::size_t>(
        {budget, prompt.size() - offset, model_->PrefillThroughCapacity()});
    if (boundary_first <= offset + 1 || boundary_first + 64 > boundary ||
        boundary >= offset + consumed || offset + consumed - boundary > 8 ||
        boundary_first < offset + 64 ||
        offset + consumed - boundary_first > 256)
      return std::nullopt;
    const auto bytes_first = qfn.session().PrefillCheckpointBytes(
        static_cast<std::uint32_t>(boundary_first));
    const auto bytes_second = qfn.session().PrefillCheckpointBytes(
        static_cast<std::uint32_t>(boundary));
    if (!bytes_first || !bytes_second)
      return std::nullopt;
    return PrefillCheckpointPair{static_cast<std::size_t>(bytes_first),
                                 static_cast<std::size_t>(bytes_second)};
  }

  [[nodiscard]] TextPrefillStep PrefillThroughPair(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget, std::size_t boundary_first,
      std::size_t boundary,
      std::unique_ptr<TextRunnerSnapshot>* checkpoint_first,
      std::unique_ptr<TextRunnerSnapshot>* checkpoint,
      const std::function<std::unique_ptr<TextRunnerSnapshot>(std::size_t)>&
          capture) const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (!checkpoint_first || !checkpoint || !capture ||
        offset != qfn.position())
      throw std::logic_error("invalid dual in-pass checkpoint request");
    const auto consumed = std::min<std::size_t>(
        {budget, prompt.size() - offset, model_->PrefillThroughCapacity()});
    const auto next_position = offset + consumed;
    const auto prefix = QwenFlashNextEngineTokens(prompt.first(next_position));
    double capture_ms = 0;
    const QwenFlashNextLogitsHook logits_hook{
        qfn.session(),
        next_position == prompt.size() ? state.prompt_logits_hook() : nullptr};
    try {
      qfn.adapter().PrefillThroughPair(
          qfn.slot(), prefix, static_cast<std::uint32_t>(boundary_first),
          static_cast<std::uint32_t>(boundary), [&](std::uint32_t at) {
            const auto started = std::chrono::steady_clock::now();
            try {
              if (at == boundary_first)
                *checkpoint_first = capture(at);
              else
                *checkpoint = capture(at);
            } catch (...) {
              // Optional retention may refuse; the complete live prefill
              // stays valid and only the affected snapshot is dropped.
              if (at == boundary_first)
                checkpoint_first->reset();
              else
                checkpoint->reset();
            }
            capture_ms += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
          });
    } catch (...) {
      qfn.set_position(0);
      throw;
    }
    qfn.set_position(next_position);
    return {.consumed_tokens = consumed,
            .decode_ready = next_position == prompt.size(),
            .checkpoint_ms = capture_ms};
  }

  // One in-pass checkpoint at a canonical boundary: P must take at least one
  // row of its own pass, sit inside the widened 256-row tail of the pass
  // end, and never be a chunk end of it (the engine refuses that: a pass
  // end is today's split capture, not an in-pass view).
  [[nodiscard]] std::optional<std::size_t> PrefillCheckpointBoundaryBytes(
      const TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget,
      std::size_t boundary) const override {
    const auto& qfn = RequireQwenFlashNextState(state);
    const auto consumed = std::min<std::size_t>(
        {budget, prompt.size() - offset, model_->PrefillThroughCapacity()});
    if (boundary <= offset || boundary >= offset + consumed ||
        offset + consumed - boundary > 256 ||
        (boundary - offset) % model_->PrefillCapacity() == 0)
      return std::nullopt;
    const auto bytes = qfn.session().PrefillCheckpointBytes(
        static_cast<std::uint32_t>(boundary));
    return bytes ? std::optional<std::size_t>(bytes) : std::nullopt;
  }

  [[nodiscard]] TextPrefillStep PrefillThroughBoundary(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget, std::size_t boundary,
      std::unique_ptr<TextRunnerSnapshot>* checkpoint,
      const std::function<std::unique_ptr<TextRunnerSnapshot>()>& capture)
      const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (!checkpoint || !capture || offset != qfn.position())
      throw std::logic_error("invalid component in-pass boundary request");
    const auto consumed = std::min<std::size_t>(
        {budget, prompt.size() - offset, model_->PrefillThroughCapacity()});
    const auto next_position = offset + consumed;
    const auto prefix = QwenFlashNextEngineTokens(prompt.first(next_position));
    double capture_ms = 0;
    const QwenFlashNextLogitsHook logits_hook{
        qfn.session(),
        next_position == prompt.size() ? state.prompt_logits_hook() : nullptr};
    try {
      qfn.adapter().PrefillThroughBoundary(
          qfn.slot(), prefix, static_cast<std::uint32_t>(boundary),
          [&](std::uint32_t) {
            const auto started = std::chrono::steady_clock::now();
            try {
              *checkpoint = capture();
            } catch (...) {
              // Optional retention may refuse; the complete live prefill
              // stays valid and the snapshot is dropped.
              checkpoint->reset();
            }
            capture_ms += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
          });
    } catch (...) {
      qfn.set_position(0);
      throw;
    }
    qfn.set_position(next_position);
    return {.consumed_tokens = consumed,
            .decode_ready = next_position == prompt.size(),
            .checkpoint_ms = capture_ms};
  }

  // The engine stops before committing EOS when the request enables it.
  // Every returned token is committed work and is emitted exactly once.
  // Logprobs ride parallel to tokens; entries carry their top-N ids here
  // and the scheduler fills any missing surface text from the runner.
  void ResolveStepTokens(QwenFlashNextTextRunnerState&,
                         const QwenFlashNextSession::DecodeResult& decoded,
                         TextDecodeStep* step) const {
    step->stop = decoded.stop;
    step->selections.reserve(decoded.tokens.size());
    for (std::size_t i = 0; i < decoded.tokens.size(); ++i) {
      const std::int32_t token = decoded.tokens[i];
      TextDecodeSelection selection{
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = model_->TokenText(token),
      };
      if (i < decoded.logprobs.size()) {
        auto logged = decoded.logprobs[i];
        logged.token = selection.token;
        logged.text = selection.piece;
        selection.logprobs = std::move(logged);
      }
      step->selections.push_back(std::move(selection));
    }
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() >= max_context_) {
      return {.stop = true, .piece = {}};
    }
    const auto logits = qfn.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error(
          "Qwen3.8-Flash-Next token selection has no logits");
    }
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    std::optional<sampling::CapturedTokenLogprobs> logprobs;
    if (sampler.logprobs_enabled()) {
      const auto top_n = sampler.logprobs_top_n();
      logprobs = sampler.CaptureLogprobs(
          logits, static_cast<sampling::TokenId>(token), top_n);
      logprobs->text = model_->TokenText(token);
    }
    if (EndsTurn(qfn, *model_, token)) {
      return {.stop = true, .token = 0, .piece = {}};
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = model_->TokenText(token),
        .logprobs = std::move(logprobs),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next token ID exceeds engine range");
    }
    auto& qfn = RequireQwenFlashNextState(state);
    std::string error;
    if (!qfn.session().Evaluate(static_cast<std::int32_t>(token), &error)) {
      throw std::runtime_error("Qwen3.8-Flash-Next decode failed: " + error);
    }
    qfn.set_position(qfn.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_mtp_ || max_tokens == 1) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next MTP decode budget must be at least one token");
    }
    auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() >= max_context_) {
      return {.selections = {}, .stop = true};
    }
    const auto stats_before = qfn.session().Statistics();
    sampling::SamplerState working_sampler = sampler;
    QwenFlashNextSession::DecodeResult decoded;
    std::string error;
    // Use the same request policy as SelectNext and the batched MTP path.
    // The engine stops before committing EOS or work beyond it, and caps a
    // step at the rows its verify serves.
    if (!qfn.session().DecodeStep(max_tokens, working_sampler, &decoded, &error,
                                  qfn.stop_at_eos())) {
      throw std::runtime_error("Qwen3.8-Flash-Next MTP decode failed: " +
                               error);
    }
    // The pool accepts the returned tokens once. Publish the RNG and residual
    // draw so the next batch retains the rejection-conditioned distribution.
    sampler.CopyDrawStateFrom(working_sampler);
    TextDecodeStep step;
    ResolveStepTokens(qfn, decoded, &step);
    qfn.set_position(qfn.session().Position());
    const auto stats_after = qfn.session().Statistics();
    step.draft_rounds = stats_after.cycles - stats_before.cycles;
    step.draft_tokens = stats_after.drafted - stats_before.drafted;
    step.draft_accepted_tokens = stats_after.accepted - stats_before.accepted;
    step.lookup_tokens = stats_after.lookup - stats_before.lookup;
    step.lookup_accepted_tokens =
        stats_after.lookup_accepted - stats_before.lookup_accepted;
    return step;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2) {
      return TextModelRunner::AdvanceBatch(advances);
    }
    std::vector<QwenFlashNextSession::BatchOutcome> outcomes(advances.size());
    std::vector<QwenFlashNextSession::AdvanceRequest> requests;
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(
                              std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Flash-Next token exceeds engine range");
      }
      requests.push_back(
          {&RequireQwenFlashNextState(advance.state.get()).session(),
           static_cast<std::int32_t>(advance.token),
           &outcomes[requests.size()]});
    }
    std::string error;
    (void)QwenFlashNextSession::EvaluateBatch(requests, &error);
    for (std::size_t i = 0; i < advances.size(); ++i) {
      const auto& advance = advances[i];
      if (!outcomes[i].completed) {
        auto failure = std::make_exception_ptr(std::runtime_error(
            "Flash-Next advance failed: " + outcomes[i].error));
        if (!advance.failure)
          std::rethrow_exception(failure);
        *advance.failure = failure;
        continue;
      }
      auto& state = RequireQwenFlashNextState(advance.state.get());
      state.set_position(state.session().Position());
    }
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (decodes.size() < 2 || !use_mtp_) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    const auto count = decodes.size();
    std::vector<sampling::SamplerState> samplers;
    std::vector<QwenFlashNextSession::DecodeResult> results(count);
    std::vector<QwenFlashNextSession::BatchOutcome> outcomes(count);
    std::vector<QwenFlashNextSession::SpeculativeStats> before;
    std::vector<QwenFlashNextSession::DecodeRequest> requests;
    samplers.reserve(count);
    before.reserve(count);
    requests.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      auto& state = RequireQwenFlashNextState(decodes[i].state.get());
      auto& session = state.session();
      auto& sampler = samplers.emplace_back(decodes[i].sampler.get());
      before.push_back(session.Statistics());
      requests.push_back({&session, decodes[i].max_tokens, &sampler,
                          &results[i], state.stop_at_eos(), &outcomes[i]});
    }
    std::string error;
    (void)QwenFlashNextSession::DecodeBatch(requests, &error);
    const auto active_count = static_cast<std::size_t>(std::count_if(
        results.begin(), results.end(),
        [](const auto& result) { return !result.tokens.empty(); }));
    std::vector<TextDecodeStep> steps(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (!outcomes[i].completed) {
        steps[i].failure = std::make_exception_ptr(
            std::runtime_error("Flash-Next MTP failed: " + outcomes[i].error));
        continue;
      }
      auto& state = RequireQwenFlashNextState(decodes[i].state.get());
      decodes[i].sampler.get().CopyDrawStateFrom(samplers[i]);
      state.set_position(state.session().Position());
      auto& step = steps[i];
      ResolveStepTokens(state, results[i], &step);
      if (active_count > 1 && !results[i].tokens.empty()) {
        step.execution_plan = {.kind = TextExecutionPlanKind::kBatched,
                               .physical_width = active_count};
      }
      const auto stats = state.session().Statistics();
      step.draft_rounds = stats.cycles - before[i].cycles;
      step.draft_tokens = stats.drafted - before[i].drafted;
      step.draft_accepted_tokens = stats.accepted - before[i].accepted;
      // A request whose peers dropped out runs a single-session step,
      // which can use prompt lookup.
      step.lookup_tokens = stats.lookup - before[i].lookup;
      step.lookup_accepted_tokens =
          stats.lookup_accepted - before[i].lookup_accepted;
    }
    return steps;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireQwenFlashNextState(state).position();
  }

  [[nodiscard]] TextRunnerNgramRows NgramRows(
      const TextRunnerState& state) const override {
    const auto totals =
        RequireQwenFlashNextState(state).session().NgramRowTotals();
    return {totals.rows, totals.dedup_rows, totals.cache_hits,
            totals.disk_reads};
  }

private:
  mutable models::qwen::vision::ImageCache image_cache_;
  std::shared_ptr<QwenFlashNextModel> model_;
  std::uint32_t max_context_;
  bool use_mtp_;
  std::uint32_t max_draft_tokens_;
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
  cache::Identity component_identity_;
};
#endif

// The markers that end the reasoning span a chat reply opens with, for a
// request that carries stop sequences; empty when the reply starts with the
// answer. A reply whose start is decided by the model (kAuto) keeps matching
// stop sequences in all of its text.
[[maybe_unused]] std::vector<std::string> StopReasoningEnds(
    const TextModelRunner& runner,
    const std::vector<std::string>& stop_sequences,
    TextGenerationBackend::InitialOutputState initial) {
  std::vector<std::string> ends;
  if (stop_sequences.empty() ||
      initial != TextGenerationBackend::InitialOutputState::kReasoning) {
    return ends;
  }
  for (const std::string_view end : runner.ReasoningBudgetTags().ends) {
    ends.emplace_back(end);
  }
  return ends;
}

// Arms the opt-in reasoning loop guard for a request that starts in its
// reasoning phase, beside the token budget. Unlike the budget the guard never
// changes `can_use_unmodified_argmax()`: the request keeps the fast argmax
// path until the sampler reports a forced token. A request may carry both; a
// null build (tags the vocabulary cannot tokenize) leaves the guard off.
void MaybeArmReasoningLoopGuard(const TextModelRunner& runner,
                                sampling::SamplingConfig* sampling) {
  if (sampling->reasoning_loop_guard != nullptr)
    return;
  const auto periods = SharedReasoningLoopGuardPeriods();
  if (!periods.has_value())
    return;
  sampling->reasoning_loop_guard = runner.BuildReasoningLoopGuard(*periods);
}

// Arms the opt-in reasoning word ban for a request that starts in its
// reasoning phase, beside the budget and the loop guard. Like the guard the
// ban never changes `can_use_unmodified_argmax()`: the request keeps the fast
// argmax path until a step's raw argmax is itself banned. A null build (no
// word matches, or tags the vocabulary cannot tokenize) leaves the ban off.
void MaybeArmReasoningWordBan(const TextModelRunner& runner,
                              sampling::SamplingConfig* sampling) {
  if (sampling->reasoning_word_ban != nullptr)
    return;
  sampling->reasoning_word_ban = runner.BuildReasoningWordBan();
}

}  // namespace

struct InferenceBackend::Impl {
#if defined(ENGINE_ENABLE_HIP)
  struct State {
    std::shared_ptr<TextGenerationScheduler> scheduler;
    std::string model_id;
    SamplingDefaults sampling_defaults;
    std::uint32_t max_context{0};
    bool supports_images{false};
    ReasoningOptions reasoning_defaults;
    std::string speculative_status;
    std::uint32_t speculative_draft_limit{0};
    std::string model_path;
    std::string build_info;
  };

  class ScheduledGenerationRequest final : public GenerationRequest {
  public:
    ScheduledGenerationRequest(
        std::shared_ptr<const State> model_state,
        TextGenerationScheduler::Request scheduled_request,
        InitialOutputState initial = InitialOutputState::kContent,
        std::optional<sampling::JsonConstraint::ToolFormat> tool_format = {})
        : state_(std::move(model_state)),
          request_(std::move(scheduled_request)),
          tool_format_(tool_format) {
      if (initial == InitialOutputState::kReasoning)
        reasoning_end_ =
            state_->scheduler->runner().TokenizeControl("</think>");
    }

    Result Wait(const TokenCallback& on_token,
                const ProgressCallback& on_progress,
                const StartCallback& on_start) override {
      auto result = request_.Wait(on_token, on_progress, on_start);
      if (!reasoning_end_.empty()) {
        const auto end =
            std::search(result.tokens.begin(), result.tokens.end(),
                        reasoning_end_.begin(), reasoning_end_.end());
        result.reasoning_tokens =
            static_cast<std::size_t>(end - result.tokens.begin());
      }
      return result;
    }

    void SetTokenMetaCallback(const TokenMetaCallback& callback) override {
      request_.SetTokenMetaCallback(callback);
    }

    void Cancel() noexcept override { request_.Cancel(); }

    std::optional<sampling::JsonConstraint::ToolFormat> ToolFormat()
        const override {
      return tool_format_;
    }

  private:
    std::shared_ptr<const State> state_;
    TextGenerationScheduler::Request request_;
    std::vector<tokenization::TokenId> reasoning_end_;
    const std::optional<sampling::JsonConstraint::ToolFormat> tool_format_;
  };

  [[nodiscard]] std::shared_ptr<const State> Snapshot() const {
    const std::lock_guard<std::mutex> lock(state_mutex);
    return state;
  }

  Result GenerateScheduled(
      std::shared_ptr<const State> current,
      std::vector<TextRunnerToken> prompt_tokens,
      Clock::time_point request_start, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled, const TokenCallback& on_token,
      std::string client_id,
      std::shared_ptr<const TextPromptContext> context = {},
      bool cache_prompt = true, std::size_t cache_prefix_tokens = 0,
      const std::vector<std::string>& stop_sequences = {},
      InitialOutputState initial = InitialOutputState::kContent) const {
    Result result;
    result.prompt_tokens = prompt_tokens.size();
    result.client_id = client_id.empty() ? "anonymous" : client_id;
    if (current == nullptr || prompt_tokens.empty()) {
      return result;
    }
    if (is_cancelled && is_cancelled()) {
      result.cancelled = true;
      return result;
    }

    auto request = current->scheduler->Submit(
        std::move(prompt_tokens), max_tokens, sampling, is_cancelled,
        static_cast<bool>(on_token),
        TextRequestMetadata{
            .client_id = std::move(client_id),
            .deadline = std::nullopt,
            .request_start = request_start,
            .prompt_context = std::move(context),
            .cache_prompt = cache_prompt,
            .cache_prefix_tokens = cache_prefix_tokens,
            .stop_sequences = stop_sequences,
            .stop_reasoning_ends = StopReasoningEnds(
                current->scheduler->runner(), stop_sequences, initial),
        });
    ScheduledGenerationRequest generation(std::move(current),
                                          std::move(request), initial);
    return generation.Wait(on_token, {}, {});
  }

  mutable std::mutex state_mutex;
  std::shared_ptr<const State> state;
#endif
};

InferenceBackend::InferenceBackend() : impl_(std::make_unique<Impl>()) {}

InferenceBackend::~InferenceBackend() {
#if defined(ENGINE_ENABLE_HIP)
  LogFp4SlotRefusals("shutdown");
#endif
}

bool InferenceBackend::load(const std::string& model_path, std::string* error,
                            std::uint32_t max_context,
                            std::size_t session_count,
                            TextPrefillPolicy prefill_policy,
                            TextSchedulerPolicy scheduler_policy,
                            const TextSpeculativeConfig& speculative_config,
                            const TextDiskCacheConfig& disk_cache_config,
                            const std::string& vision_model_path,
                            TextRunnerRamCacheOptions ram_cache_config) {
  // Ling-3.0-flash (bailingmoe3). Recognised before any architecture-specific
  // loader so a Ling artifact is never handed to the Qwen one, which would
  // reinterpret a ROCmFP4/KDA file as a Qwen file and fail deep in tensor
  // binding with a misleading message. Bound and validated here, in every
  // build configuration, so a bad artifact reports why rather than what broke
  // downstream.
  {
    std::string probe_error;
    auto probe = core::GgufReader::OpenFile(model_path, &probe_error);
    if (probe != nullptr &&
        superroc::core::IsModelFamily(
            probe->GetMetadataString("general.architecture").value_or(""),
            superroc::core::ModelFamilyId::kBailingMoe3)) {
      std::shared_ptr<models::ling3::vision::VisionTower> vision_tower;
      if (!vision_model_path.empty()) {
        const auto general_name =
            probe->GetMetadataString("general.name").value_or("");
        bool name_is_vl = false;
        for (std::size_t i = 0; i + 1 < general_name.size(); ++i) {
          char a = general_name[i];
          char b = general_name[i + 1];
          if (a >= 'A' && a <= 'Z')
            a = static_cast<char>(a - 'A' + 'a');
          if (b >= 'A' && b <= 'Z')
            b = static_cast<char>(b - 'A' + 'a');
          if (a == 'v' && b == 'l')
            name_is_vl = true;
        }
        if (!name_is_vl) {
          SetError(error, "Ling-3.0-flash text file does not support --mmproj");
          return false;
        }
      }
      if (speculative_config.backend != TextSpeculativeBackend::kDisabled) {
        SetError(error,
                 "Ling-3.0-flash speculative decoding is not wired; the MTP "
                 "block is bound but unused");
        return false;
      }
      auto tokenizer =
          models::ling3::LingTokenizer::FromGguf(*probe, &probe_error);
      if (!tokenizer.has_value()) {
        SetError(error, "Unsupported Ling-3.0-flash tokenizer: " + probe_error);
        return false;
      }
      if (!vision_model_path.empty()) {
        for (const char* marker :
             {"<|image_pad|>", "<|vision_start|>", "<|vision_end|>"}) {
          if (tokenizer->Encode(marker, false, true).size() != 1) {
            SetError(error, "Ling-VL file has no vision placeholder tokens");
            return false;
          }
        }
        std::string vision_error;
        vision_tower = models::ling3::vision::VisionTower::Open(
            vision_model_path, &vision_error);
        if (!vision_tower) {
          SetError(error, "Ling-VL mmproj rejected: " + vision_error);
          return false;
        }
      }
      auto weights = models::ling3::ModelWeights::Bind(*probe, &probe_error);
      if (!weights.has_value()) {
        SetError(error, "Unsupported Ling-3.0-flash artifact: " + probe_error);
        return false;
      }
      auto ling_model = models::ling3::Model::Load(
          model_path,
          models::ling3::ModelOptions{
              .max_context = max_context,
#if defined(ENGINE_ENABLE_HIP)
              // The ROCm executor is not written yet; rocm_backend.cpp refuses
              // with a message naming what is missing, so this is honest
              // rather than silently falling back.
              .backend = models::ling3::BackendKind::kRocm,
#else
              .backend = models::ling3::BackendKind::kCpuReference,
#endif
          },
          &probe_error);
      if (ling_model == nullptr) {
        SetError(error,
                 "Failed to create the Ling-3.0-flash model: " + probe_error);
        return false;
      }
      // This branch runs before the HIP-guarded section, so `reader` and
      // `resolved_disk_cache_config` are not in scope here -- the probe is
      // `probe` and the disk cache is passed through unchanged.
      if (vision_tower &&
          vision_tower->OutputWidth() !=
              static_cast<int>(ling_model->config().hidden_size)) {
        SetError(error, "mmproj projector width does not match this Ling file");
        return false;
      }
      return load(std::move(ling_model), error, max_context, session_count,
                  prefill_policy, scheduler_policy, speculative_config,
                  disk_cache_config, std::move(vision_tower));
    }
  }

#if defined(ENGINE_ENABLE_HIP)
  TextDiskCacheConfig resolved_disk_cache_config = disk_cache_config;
  std::string load_error;
  auto reader_owner = core::GgufReader::OpenFile(model_path, &load_error);
  if (reader_owner == nullptr) {
    SetError(error, "Failed to open GGUF: " + load_error);
    return false;
  }
  const std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  if (max_context == 0) {
    const auto architecture =
        reader->GetMetadataString("general.architecture").value_or("");
    const auto native =
        reader->GetMetadataUint64(std::string(architecture) + ".context_length")
            .value_or(0);
    if (native < 2 || native > std::numeric_limits<std::uint32_t>::max()) {
      SetError(error,
               "GGUF has no valid native context length; specify --context");
      return false;
    }
    max_context = static_cast<std::uint32_t>(native);
  }

  if (superroc::core::IsModelFamily(
          reader->GetMetadataString("general.architecture").value_or(""),
          superroc::core::ModelFamilyId::kDeepSeek4)) {
    if (!vision_model_path.empty()) {
      SetError(error, "DeepSeek does not support --mmproj");
      return false;
    }
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
        speculative_config.backend != TextSpeculativeBackend::kDSpark) {
      SetError(error,
               "DeepSeek HTTP models support only DSpark speculative decoding");
      return false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kDSpark &&
        speculative_config.draft_model_path.empty()) {
      SetError(error, "DeepSeek DSpark HTTP decoding requires --dspark-model");
      return false;
    }
    if (!models::deepseek_v4_flash::ValidateGgufTemplate(*reader,
                                                         &load_error)) {
      SetError(error, "Unsupported DeepSeek chat template: " + load_error);
      return false;
    }
    auto model = models::deepseek_v4_flash::Model::Load(
        model_path,
        models::deepseek_v4_flash::ModelOptions{
            .max_context = max_context,
            .dspark_model_path =
                speculative_config.backend == TextSpeculativeBackend::kDSpark
                    ? speculative_config.draft_model_path
                    : std::string{},
        },
        &load_error);
    if (model == nullptr) {
      SetError(error, "Failed to create DeepSeek model: " + load_error);
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        resolved_disk_cache_config.model_artifact_fingerprint.empty() &&
        !FingerprintArtifact(
            "DeepSeek", *reader,
            &resolved_disk_cache_config.model_artifact_fingerprint, error)) {
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        speculative_config.backend == TextSpeculativeBackend::kDSpark &&
        resolved_disk_cache_config.draft_model_artifact_fingerprint.empty() &&
        !FingerprintArtifactFile(
            "DSpark", speculative_config.draft_model_path,
            &resolved_disk_cache_config.draft_model_artifact_fingerprint,
            error)) {
      return false;
    }
    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config), ram_cache_config);
  }
  if (superroc::core::IsModelFamily(
          reader->GetMetadataString("general.architecture").value_or(""),
          superroc::core::ModelFamilyId::kQwen4Exp)) {
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
        speculative_config.backend != TextSpeculativeBackend::kMtp) {
      SetError(error,
               "Qwen3.8-Flash-Next HTTP models support only MTP speculative "
               "decoding (--speculative mtp --mtp-model)");
      return false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        speculative_config.draft_model_path.empty()) {
      SetError(error,
               "Qwen3.8-Flash-Next MTP HTTP decoding requires --mtp-model");
      return false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        (speculative_config.max_draft_tokens == 0 ||
         speculative_config.min_draft_tokens != 1)) {
      SetError(error,
               "Flash-Next MTP requires a positive draft limit and "
               "--min-draft-tokens 1");
      return false;
    }
    // The compiled Qwen3.8 chat template renders through the artifact's
    // own tokenizer (the same vocabulary and pre-tokenizer as Qwen3.8).
    if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                              &load_error)) {
      SetError(error,
               "Unsupported Qwen3.8-Flash-Next chat template: " + load_error);
      return false;
    }
    // The model owns prefill geometry for both bulk and scheduled requests.
    auto model = models::qwen38_flash_next::Model::Load(
        model_path,
        models::qwen38_flash_next::ModelOptions{
            .max_context = max_context,
            .mtp_model_path =
                speculative_config.backend == TextSpeculativeBackend::kMtp
                    ? speculative_config.draft_model_path
                    : std::string{},
            .max_draft_tokens = speculative_config.max_draft_tokens,
            .vision_model_path = vision_model_path,
            .decode_concurrency = static_cast<std::uint32_t>(
                std::clamp<std::size_t>(session_count, 1, 8)),
        },
        &load_error);
    if (model == nullptr) {
      SetError(error,
               "Failed to create Qwen3.8-Flash-Next model: " + load_error);
      return false;
    }
    // RAM checkpoints cannot outlive this loaded model. Their process-local
    // identity is sufficient; hashing every shard here adds a complete model
    // read on each fresh container start even when disk caching is disabled.
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        resolved_disk_cache_config.model_artifact_fingerprint.empty() &&
        !FingerprintArtifact(
            "Qwen3.8-Flash-Next", *reader,
            &resolved_disk_cache_config.model_artifact_fingerprint, error)) {
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        speculative_config.backend == TextSpeculativeBackend::kMtp &&
        resolved_disk_cache_config.draft_model_artifact_fingerprint.empty() &&
        !FingerprintArtifactFile(
            "Qwen3.8-Flash-Next MTP", speculative_config.draft_model_path,
            &resolved_disk_cache_config.draft_model_artifact_fingerprint,
            error)) {
      return false;
    }
    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config), ram_cache_config);
  }
  // The registry is the architecture gate. Unknown arches
  // fail closed instead of falling through into the Qwen loader; the
  // per-family refusals come from the table.
  {
    const auto arch =
        reader->GetMetadataString("general.architecture").value_or("");
    const superroc::core::ModelFamily* family =
        superroc::core::FindModelFamily(arch);
    if (family == nullptr) {
      SetError(error, "unsupported architecture: " + std::string(arch));
      return false;
    }
    if (!family->vision_refusal.empty() && !vision_model_path.empty()) {
      SetError(error, std::string(family->vision_refusal));
      return false;
    }
    if (!family->speculative_refusal.empty() &&
        speculative_config.backend != TextSpeculativeBackend::kDisabled) {
      SetError(error, std::string(family->speculative_refusal));
      return false;
    }
    // LFM2: DSpark drafting via dflash2 only. There is no MTP head (and
    // the DeepSeek DSpark enumerator is a different support model), so
    // every other backend refuses here with the arch in the message.
    if (superroc::core::IsLfm2Family(arch) &&
        speculative_config.backend != TextSpeculativeBackend::kDisabled &&
        speculative_config.backend != TextSpeculativeBackend::kDFlash) {
      SetError(error, std::string(arch) +
                          " supports only --speculative dflash2 with a DSpark "
                          "drafter (--dflash-model); it has no MTP head");
      return false;
    }
    // plain Llama accepts only DFlash2 drafting, for the same
    // reason as LFM2 (no MTP head in the target file).
    if (superroc::core::IsModelFamily(arch,
                                      superroc::core::ModelFamilyId::kLlama) &&
        speculative_config.backend != TextSpeculativeBackend::kDisabled &&
        speculative_config.backend != TextSpeculativeBackend::kDFlash) {
      SetError(error, std::string(arch) +
                          " supports only --speculative dflash2 "
                          "(--dflash-model); it has no MTP head");
      return false;
    }
  }
  // KAT: qwen35moe binds on the Qwen chassis. Refuse vision, the DeepSeek
  // DSpark enumerator, and a sidecar MTP draft. `--speculative dflash2` is
  // accepted only when the sidecar is the markov/DSpark kind.
  if (superroc::core::IsModelFamily(
          reader->GetMetadataString("general.architecture").value_or(""),
          superroc::core::ModelFamilyId::kQwen35Moe)) {
    // KAT: map the real enumerator by name. Do not cast the integer.
    models::qwen::KatSpecKind kat_spec = models::qwen::KatSpecKind::kOff;
    switch (speculative_config.backend) {
      case TextSpeculativeBackend::kDisabled:
        kat_spec = models::qwen::KatSpecKind::kOff;
        break;
      case TextSpeculativeBackend::kDFlash:
        kat_spec = models::qwen::KatSpecKind::kDFlash;
        break;
      case TextSpeculativeBackend::kDSpark:
        kat_spec = models::qwen::KatSpecKind::kDSpark;
        break;
      case TextSpeculativeBackend::kMtp:
        kat_spec = models::qwen::KatSpecKind::kMtp;
        break;
      default:
        SetError(error, "KAT qwen35moe refuses this speculative backend");
        return false;
    }
    bool dflash_markov_sidecar = false;
    if (kat_spec == models::qwen::KatSpecKind::kDFlash &&
        !speculative_config.draft_model_path.empty()) {
      std::string peek_error;
      auto peek = core::GgufReader::OpenFile(
          speculative_config.draft_model_path, &peek_error,
          core::GgufReader::FileRole::kSidecar);
      if (peek != nullptr) {
        dflash_markov_sidecar = models::qwen::DflashSidecarIsMarkovKind(
            peek->FindTensor("markov_w1.weight") != nullptr,
            peek->FindTensor("selector_predecessor.weight") != nullptr);
      }
    }
    const auto decision = models::qwen::ClassifyKatDispatch(
        superroc::core::ModelFamilyArch(
            superroc::core::ModelFamilyId::kQwen35Moe),
        kat_spec, speculative_config.draft_model_path.empty(),
        reader->HasVisionTensors(), dflash_markov_sidecar);
    if (decision == models::qwen::KatDispatch::kRefuseVision) {
      SetError(error, "KAT qwen35moe refuses a vision encoder");
      return false;
    }
    if (decision == models::qwen::KatDispatch::kRefuseDSpark) {
      SetError(error, "KAT qwen35moe refuses DSpark");
      return false;
    }
    if (decision == models::qwen::KatDispatch::kRefuseOther ||
        decision == models::qwen::KatDispatch::kNotKat) {
      SetError(error, "KAT qwen35moe refuses this speculative backend");
      return false;
    }
    // KAT S3: GPU expert decode is installed. Fall through into
    // QwenGpuModel::CreateFromGguf (same path as `superroc prompt`). MoE layers
    // have empty dense ffn projections; decode_step and prefill_chunk take
    // the expert path when is_moe, so the dense SwiGLU is not launched.
  }
  std::shared_ptr<models::qwen::vision::Encoder> vision;
  try {
    if (reader->GetMetadataUint64("qwen35.embedding_length") == 5120) {
      vision = models::qwen::vision::Encoder::Open(model_path,
                                                   vision_model_path, 5120);
    } else if (const auto moe =
                   reader->GetMetadataUint64("qwen35moe.embedding_length");
               !vision_model_path.empty() && (moe == 2048 || moe == 3072)) {
      // Qwen3.5/3.6 MoE: same ViT; the merger is 2048 wide (35B-A3B) or 3072
      // wide (122B-A10B).
      vision = models::qwen::vision::Encoder::Open(
          model_path, vision_model_path, static_cast<std::uint32_t>(*moe));
    } else if (const auto dense =
                   reader->GetMetadataUint64("qwen35.embedding_length");
               !vision_model_path.empty() && (dense == 4096 || dense == 2048)) {
      // Qwen3.5 dense 9B (4096) and the 2048-wide dense sizes: same ViT,
      // narrower merger.
      vision = models::qwen::vision::Encoder::Open(
          model_path, vision_model_path, static_cast<std::uint32_t>(*dense));
    } else if (!vision_model_path.empty()) {
      throw std::invalid_argument(
          "image input supports Qwen3.8-27B, Flash-Next, the Qwen3.5 dense "
          "9B/2048 sizes and the Qwen3.5/3.6 MoE family");
    }
  } catch (const std::exception& e) {
    SetError(error, e.what());
    return false;
  }
  Logger::Info("loader", "event=load_phase phase=target_weights " +
                             Logger::MemoryStatus());
  auto model =
      hip::QwenGpuModel::CreateFromGguf(reader, &load_error, std::move(vision));
  if (model == nullptr) {
    SetError(error, "Failed to create GPU model: " + load_error);
    return false;
  }
  if (DiskCacheEnabled(resolved_disk_cache_config) &&
      resolved_disk_cache_config.model_artifact_fingerprint.empty() &&
      !FingerprintArtifact(
          "Qwen", *reader,
          &resolved_disk_cache_config.model_artifact_fingerprint, error)) {
    return false;
  }
  return load(std::move(model), error, max_context, session_count,
              prefill_policy, scheduler_policy, speculative_config,
              std::move(resolved_disk_cache_config), ram_cache_config);
#else
  (void)model_path;
  (void)max_context;
  (void)session_count;
  (void)prefill_policy;
  (void)scheduler_policy;
  (void)speculative_config;
  (void)disk_cache_config;
  (void)vision_model_path;
  (void)ram_cache_config;
  SetError(error, "HTTP inference requires the HIP backend");
  return false;
#endif
}

#if defined(ENGINE_ENABLE_HIP)
// Every serve load sizes the FP4 slot manager before its
// pool is built (Qwen: one set per session stream; Flash-Next: one shared
// executor stream; DeepSeek/Ling: no FP4 streams, the call only sizes the
// vectors). Configure() only pre-sizes and never fails, so a second
// backend (or a reload with another session count) grows the table instead
// of failing the load; every load-time attach still gets a warmed set.
bool ConfigureFp4SlotSets(std::size_t session_count, std::string* error) {
  (void)error;
  if (session_count < 1)
    session_count = 1;
  (void)superroc_fp4_slot_sets_configure(static_cast<int>(session_count));
  return true;
}

bool InferenceBackend::load(std::shared_ptr<const hip::QwenGpuModel> model,
                            std::string* error, std::uint32_t max_context,
                            std::size_t session_count,
                            TextPrefillPolicy prefill_policy,
                            TextSchedulerPolicy scheduler_policy,
                            TextSpeculativeConfig speculative_config,
                            TextDiskCacheConfig disk_cache_config,
                            TextRunnerRamCacheOptions ram_cache_config) {
  if (model == nullptr) {
    SetError(error, "Qwen GPU model must not be null");
    return false;
  }
  if (max_context == 0)
    max_context = model->GetConfig().context_length;
  if (max_context < 2 || max_context > model->GetConfig().context_length) {
    SetError(error, "HTTP context exceeds the loaded Qwen model context");
    return false;
  }

  if (speculative_config.backend == TextSpeculativeBackend::kDSpark) {
    SetError(error, "DSpark HTTP decoding requires a DeepSeek model");
    return false;
  }
  // KAT-MTP: a KAT qwen35moe carries its nextn block in the model file, so an
  // empty --mtp-model selects the embedded path instead of being refused.
  const bool embedded_mtp =
      speculative_config.backend == TextSpeculativeBackend::kMtp &&
      speculative_config.draft_model_path.empty() &&
      model->GetConfig().is_moe && model->GetWeights().mtp.has_value();
  // A dense file that carries its own MTP block is its own MTP file: it needs
  // no --mtp-model, and when no speculative option was given at all the
  // block is used. --speculative off keeps the model without a drafter.
  bool own_file_mtp = false;
  if (!model->GetConfig().is_moe &&
      speculative_config.draft_model_path.empty() &&
      (speculative_config.backend == TextSpeculativeBackend::kMtp ||
       (speculative_config.backend == TextSpeculativeBackend::kDisabled &&
        speculative_config.automatic))) {
    std::string block_error;
    own_file_mtp = speculative::QwenMtpWeights::LoadFromGguf(
                       model->GetReader(), &block_error,
                       /*require_tied_embeddings=*/false)
                       .has_value();
    if (own_file_mtp &&
        speculative_config.backend == TextSpeculativeBackend::kDisabled) {
      speculative_config.backend = TextSpeculativeBackend::kMtp;
      Logger::Info("loader",
                   "event=speculative_auto backend=mtp "
                   "reason=model_file_mtp_block draft_limit=" +
                       std::to_string(speculative_config.max_draft_tokens));
    }
  }
  if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
      speculative_config.draft_model_path.empty() && !embedded_mtp &&
      !own_file_mtp) {
    SetError(error, "Qwen MTP HTTP decoding requires --mtp-model");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
      speculative_config.default_draft_tokens && !model->GetConfig().is_moe) {
    speculative_config.max_draft_tokens = std::max(
        hip::kMtpDenseDraftTokensDefault, speculative_config.min_draft_tokens);
  }
  // A longer draft would not fit one verification, and the request that first
  // drafted to the limit would fail instead of the load.
  if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
      speculative_config.max_draft_tokens > hip::kQwenMaxDraftTokens) {
    SetError(error, "Qwen MTP drafts at most " +
                        std::to_string(hip::kQwenMaxDraftTokens) +
                        " tokens per step; lower --draft-tokens");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config) &&
      (!IsSha256Hex(disk_cache_config.model_artifact_fingerprint) ||
       (!disk_cache_config.draft_model_artifact_fingerprint.empty() &&
        !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) ||
       disk_cache_config.capacity_bytes == 0)) {
    SetError(error, "Qwen persistent disk cache configuration is invalid");
    return false;
  }
  if (speculative_config.max_draft_tokens == 0 ||
      speculative_config.min_draft_tokens == 0 ||
      speculative_config.min_draft_tokens >
          speculative_config.max_draft_tokens) {
    SetError(error, "HTTP speculative draft limits are invalid");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kDFlash &&
      speculative_config.min_draft_tokens != 1) {
    SetError(error,
             "DFlash2 requires --min-draft-tokens 1; bound blocks with "
             "--draft-tokens");
    return false;
  }

  try {
    std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model;
    std::shared_ptr<const hip::QwenMtpGpuModel> mtp_model;
    speculative::SpeculativeOptions speculative_options;
    speculative_options.prompt_lookup = speculative_config.prompt_lookup;
    if (speculative_config.backend == TextSpeculativeBackend::kDFlash) {
      if (speculative_config.draft_model_path.empty()) {
        SetError(error, "DFlash HTTP decoding requires --dflash-model");
        return false;
      }
      std::string dflash_error;
      auto dflash_reader_owner = core::GgufReader::OpenFile(
          speculative_config.draft_model_path, &dflash_error,
          core::GgufReader::FileRole::kSidecar);
      if (dflash_reader_owner == nullptr) {
        SetError(error, "Failed to open DFlash GGUF: " + dflash_error);
        return false;
      }
      std::shared_ptr<const core::GgufReader> dflash_reader(
          std::move(dflash_reader_owner));
      if (DiskCacheEnabled(disk_cache_config) &&
          disk_cache_config.draft_model_artifact_fingerprint.empty() &&
          !FingerprintArtifact(
              "DFlash", *dflash_reader,
              &disk_cache_config.draft_model_artifact_fingerprint, error)) {
        return false;
      }
      if (DiskCacheEnabled(disk_cache_config) &&
          !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) {
        SetError(error,
                 "Qwen DFlash persistent disk cache configuration is "
                 "invalid");
        return false;
      }
      Logger::Info("loader", "event=load_phase phase=draft_weights " +
                                 Logger::MemoryStatus());
      dflash_model = hip::QwenDFlashGpuModel::Create(std::move(dflash_reader),
                                                     model, &dflash_error);
      if (dflash_model == nullptr) {
        SetError(error, "Failed to create DFlash model: " + dflash_error);
        return false;
      }

      speculative_options.dflash_policy = speculative_config.dflash_policy;
      speculative_options.max_draft_tokens =
          speculative_config.max_draft_tokens;
      speculative_options.min_draft_tokens =
          speculative_config.min_draft_tokens;
      speculative_options.initial_draft_tokens =
          speculative_config.max_draft_tokens;
      speculative_options.use_batched_verification = true;
      speculative_options.retain_frontier_logits = true;
      speculative_options.enable_adaptive_draft_length = false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kMtp) {
      std::string mtp_error;
      Logger::Info("loader", "event=load_phase phase=draft_weights " +
                                 Logger::MemoryStatus());
      if (embedded_mtp) {
        // KAT-MTP: alias the target's own nextn block. No sidecar reader.
        mtp_model = hip::QwenMtpGpuModel::CreateFromEmbedded(model, &mtp_error);
        if (DiskCacheEnabled(disk_cache_config) &&
            disk_cache_config.draft_model_artifact_fingerprint.empty()) {
          disk_cache_config.draft_model_artifact_fingerprint =
              disk_cache_config.model_artifact_fingerprint;
        }
      } else if (own_file_mtp) {
        // The block inside the model file, bound through the model's reader.
        mtp_model = hip::QwenMtpGpuModel::Create(model->ShareReader(), model,
                                                 &mtp_error);
        if (DiskCacheEnabled(disk_cache_config) &&
            disk_cache_config.draft_model_artifact_fingerprint.empty()) {
          disk_cache_config.draft_model_artifact_fingerprint =
              disk_cache_config.model_artifact_fingerprint;
        }
      } else {
        auto mtp_reader_owner = core::GgufReader::OpenFile(
            speculative_config.draft_model_path, &mtp_error,
            core::GgufReader::FileRole::kSidecar);
        if (mtp_reader_owner == nullptr) {
          SetError(error, "Failed to open MTP GGUF: " + mtp_error);
          return false;
        }
        std::shared_ptr<const core::GgufReader> mtp_reader(
            std::move(mtp_reader_owner));
        if (DiskCacheEnabled(disk_cache_config) &&
            disk_cache_config.draft_model_artifact_fingerprint.empty() &&
            !FingerprintArtifact(
                "MTP", *mtp_reader,
                &disk_cache_config.draft_model_artifact_fingerprint, error)) {
          return false;
        }
        // A MoE target binds the sidecar's own nextn block; a dense target
        // keeps the standalone-sidecar pack.
        mtp_model = model->GetConfig().is_moe
                        ? hip::QwenMtpGpuModel::CreateFromMoeSidecar(
                              std::move(mtp_reader), model, &mtp_error)
                        : hip::QwenMtpGpuModel::Create(std::move(mtp_reader),
                                                       model, &mtp_error);
      }
      if (mtp_model == nullptr) {
        SetError(error, "Failed to create Qwen MTP model: " + mtp_error);
        return false;
      }

      speculative_options.max_draft_tokens =
          speculative_config.max_draft_tokens;
      speculative_options.min_draft_tokens =
          speculative_config.min_draft_tokens;
      speculative_options.initial_draft_tokens =
          speculative_config.max_draft_tokens;
      speculative_options.use_batched_verification = true;
      speculative_options.retain_frontier_logits = true;
      // MTP drafts a chain, so the adaptive length controller is meaningful
      // here (this is what superroc prompt already selects for --speculative
      // mtp). DFlash2 diffuses a fixed block, so it pins the length instead.
      speculative_options.enable_adaptive_draft_length = true;
    }
    if (speculative_config.prompt_lookup &&
        speculative_config.backend == TextSpeculativeBackend::kDisabled) {
      speculative_options.max_draft_tokens = 1;
      speculative_options.min_draft_tokens = 1;
      speculative_options.initial_draft_tokens = 1;
      speculative_options.use_batched_verification = true;
      speculative_options.retain_frontier_logits = true;
      speculative_options.enable_adaptive_draft_length = false;
    }

    PrepareVision(model->VisionEncoder());
    auto new_state = std::make_shared<Impl::State>();
    new_state->sampling_defaults.model =
        sampling::TextPreset(model->GetConfig());
    new_state->sampling_defaults.supplied = {};
    new_state->speculative_status =
        SpeculativeStatus(speculative_config, model->GetConfig().is_moe);
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled)
      new_state->speculative_draft_limit = speculative_config.max_draft_tokens;
    new_state->supports_images = model->VisionEncoder() != nullptr;
    auto runner = std::make_shared<QwenTextRunner>(
        std::move(model), max_context, std::move(dflash_model),
        std::move(mtp_model), speculative_options,
        disk_cache_config.model_artifact_fingerprint,
        disk_cache_config.draft_model_artifact_fingerprint);
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = max_context;
    std::optional<TextRunnerDiskCacheOptions> runner_disk_cache;
    if (DiskCacheEnabled(disk_cache_config)) {
      runner_disk_cache = TextRunnerDiskCacheOptions{
          .directory = std::move(disk_cache_config.directory),
          .capacity_bytes = disk_cache_config.capacity_bytes,
          .staging_capacity_bytes = disk_cache_config.staging_capacity_bytes,
      };
    }
    Logger::Info("loader",
                 "event=load_phase phase=sessions " + Logger::MemoryStatus());
    if (!ConfigureFp4SlotSets(session_count, error)) {
      return false;
    }
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::move(runner_disk_cache),
        ram_cache_config);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    LogFp4SlotRefusals("load_end");
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::deepseek_v4_flash::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config,
    TextRunnerRamCacheOptions ram_cache_config) {
  if (model == nullptr) {
    SetError(error, "DeepSeek model must not be null");
    return false;
  }
  if (max_context == 0)
    max_context = model->MaxContext();

  const bool use_dspark =
      speculative_config.backend == TextSpeculativeBackend::kDSpark;
  if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
      (!use_dspark || !model->HasDspark())) {
    SetError(error, "DeepSeek DSpark requires a loaded support model");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context > model->MaxContext()) {
    SetError(error, "HTTP context exceeds the loaded DeepSeek model context");
    return false;
  }
  if (use_dspark && (speculative_config.max_draft_tokens == 0 ||
                     speculative_config.min_draft_tokens == 0 ||
                     speculative_config.min_draft_tokens >
                         speculative_config.max_draft_tokens)) {
    SetError(error, "DeepSeek DSpark draft limits are invalid");
    return false;
  }
  if (use_dspark && speculative_config.min_draft_tokens != 1) {
    SetError(error,
             "DSpark uses model-owned adaptive drafting; custom draft "
             "floors are unsupported");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config) &&
      (!IsSha256Hex(disk_cache_config.model_artifact_fingerprint) ||
       (use_dspark &&
        !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) ||
       disk_cache_config.capacity_bytes == 0)) {
    SetError(error, "DeepSeek persistent disk cache configuration is invalid");
    return false;
  }

  try {
    auto new_state = std::make_shared<Impl::State>();
    new_state->sampling_defaults.model =
        sampling::TextModelPreset::kDeepSeekV4Flash;
    new_state->sampling_defaults.supplied = {};
    auto runner = std::make_shared<DeepSeekTextRunner>(
        std::move(model), max_context, use_dspark,
        speculative_config.max_draft_tokens,
        disk_cache_config.model_artifact_fingerprint,
        disk_cache_config.draft_model_artifact_fingerprint);
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = max_context;
    std::optional<TextRunnerDiskCacheOptions> runner_disk_cache;
    if (DiskCacheEnabled(disk_cache_config)) {
      runner_disk_cache = TextRunnerDiskCacheOptions{
          .directory = std::move(disk_cache_config.directory),
          .capacity_bytes = disk_cache_config.capacity_bytes,
          .staging_capacity_bytes = disk_cache_config.staging_capacity_bytes,
      };
    }
    if (!ConfigureFp4SlotSets(session_count, error)) {
      return false;
    }
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::move(runner_disk_cache),
        ram_cache_config);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    LogFp4SlotRefusals("load_end");
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::qwen38_flash_next::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config,
    TextRunnerRamCacheOptions ram_cache_config) {
  if (model == nullptr) {
    SetError(error, "Qwen3.8-Flash-Next model must not be null");
    return false;
  }
  if (max_context == 0)
    max_context = model->MaxContext();

  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context == 0 || max_context > model->MaxContext()) {
    SetError(
        error,
        "HTTP context exceeds the loaded Qwen3.8-Flash-Next model context");
    return false;
  }
  if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
      (speculative_config.backend != TextSpeculativeBackend::kMtp ||
       !model->HasMtp())) {
    SetError(error,
             "Flash-Next MTP requires a model loaded with its draft sidecar");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
      (speculative_config.max_draft_tokens == 0 ||
       speculative_config.min_draft_tokens != 1)) {
    SetError(error,
             "Qwen3.8-Flash-Next MTP drafts a fixed chain; custom draft "
             "floors are unsupported");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config) &&
      (!IsSha256Hex(disk_cache_config.model_artifact_fingerprint) ||
       (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) ||
       disk_cache_config.capacity_bytes == 0)) {
    SetError(error,
             "Qwen3.8-Flash-Next persistent disk cache configuration is "
             "invalid");
    return false;
  }
  try {
    PrepareVision(model->VisionEncoder());
    auto new_state = std::make_shared<Impl::State>();
    new_state->sampling_defaults.model = sampling::TextModelPreset::kQwen38;
    new_state->sampling_defaults.supplied = {};
    new_state->supports_images = model->VisionEncoder() != nullptr;
    auto runner = std::make_shared<QwenFlashNextTextRunner>(
        std::move(model), max_context,
        speculative_config.backend == TextSpeculativeBackend::kMtp,
        speculative_config.max_draft_tokens,
        disk_cache_config.model_artifact_fingerprint,
        disk_cache_config.draft_model_artifact_fingerprint);
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = max_context;
    std::optional<TextRunnerDiskCacheOptions> runner_disk_cache;
    if (DiskCacheEnabled(disk_cache_config)) {
      runner_disk_cache = TextRunnerDiskCacheOptions{
          .directory = std::move(disk_cache_config.directory),
          .capacity_bytes = disk_cache_config.capacity_bytes,
          .staging_capacity_bytes = disk_cache_config.staging_capacity_bytes,
      };
    }
    if (!ConfigureFp4SlotSets(session_count, error)) {
      return false;
    }
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::move(runner_disk_cache),
        ram_cache_config);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    LogFp4SlotRefusals("load_end");
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::ling3::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config,
    std::shared_ptr<models::ling3::vision::VisionTower> vision_tower) {
  if (model == nullptr) {
    SetError(error, "Ling-3.0-flash model must not be null");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context == 0) {
    max_context = model->MaxContext();
  }
  if (max_context < 2 || max_context > model->MaxContext()) {
    SetError(error,
             "HTTP context exceeds the loaded Ling-3.0-flash model context");
    return false;
  }
  // Ling's MTP block is bound and validated but never executed, so any
  // speculative backend is refused rather than silently ignored.
  if (speculative_config.backend != TextSpeculativeBackend::kDisabled) {
    SetError(error,
             "Ling-3.0-flash speculative decoding is not wired; the MTP block "
             "is bound but unused");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config) &&
      (!IsSha256Hex(disk_cache_config.model_artifact_fingerprint) ||
       disk_cache_config.capacity_bytes == 0)) {
    SetError(error,
             "Ling-3.0-flash persistent disk cache configuration is invalid");
    return false;
  }
  try {
    auto new_state = std::make_shared<Impl::State>();
    new_state->sampling_defaults.model = sampling::TextModelPreset::kLing3;
    new_state->sampling_defaults.supplied = {};
    auto runner = std::make_shared<Ling3TextRunner>(
        std::move(model), max_context, std::string{}, std::move(vision_tower));
    new_state->model_id = runner->Descriptor().model_id;
    new_state->max_context = max_context;
    std::optional<TextRunnerDiskCacheOptions> runner_disk_cache;
    if (DiskCacheEnabled(disk_cache_config)) {
      runner_disk_cache = TextRunnerDiskCacheOptions{
          .directory = std::move(disk_cache_config.directory),
          .capacity_bytes = disk_cache_config.capacity_bytes,
          .staging_capacity_bytes = disk_cache_config.staging_capacity_bytes,
      };
    }
    if (!ConfigureFp4SlotSets(session_count, error)) {
      return false;
    }
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::move(runner_disk_cache));
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    LogFp4SlotRefusals("load_end");
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

#else  // !defined(ENGINE_ENABLE_HIP)

/// CPU builds have no `Impl::State`, `state` or `state_mutex` -- the whole
/// scheduler state is HIP-only -- so the Ling installer cannot exist there.
/// The DECLARATION and the `bailingmoe3` call site are both unconditional, so
/// without this stub such a build failed to LINK with an undefined reference to
/// this function. Refusing with the real reason is the same contract every
/// other CPU-only path in this file already has.
bool InferenceBackend::load(
    std::shared_ptr<models::ling3::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config,
    std::shared_ptr<models::ling3::vision::VisionTower> vision_tower) {
  (void)model;
  (void)vision_tower;
  (void)max_context;
  (void)session_count;
  (void)prefill_policy;
  (void)scheduler_policy;
  (void)speculative_config;
  (void)disk_cache_config;
  SetError(
      error,
      "Ling-3.0-flash (bailingmoe3) needs the HIP backend; this superroc was "
      "built with ENGINE_ENABLE_HIP=OFF. Use superroc_ling3_ref for the CPU "
      "oracle, or rebuild with -DENGINE_ENABLE_HIP=ON.");
  return false;
}

#endif  // ENGINE_ENABLE_HIP

std::string InferenceBackend::model_id() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->model_id : "unknown";
#else
  return "unknown";
#endif
}

bool InferenceBackend::ready() const {
#if defined(ENGINE_ENABLE_HIP)
  return impl_->Snapshot() != nullptr;
#else
  return false;
#endif
}

bool InferenceBackend::device_lost() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr && state->scheduler->device_lost();
#else
  return false;
#endif
}

bool InferenceBackend::supports_images() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr && state->supports_images;
#else
  return false;
#endif
}

std::uint32_t InferenceBackend::max_context() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->max_context : 0;
#else
  return 0;
#endif
}

std::string InferenceBackend::speculative_status() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->speculative_status : std::string{};
#else
  return {};
#endif
}

std::uint32_t InferenceBackend::speculative_draft_limit() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->speculative_draft_limit : 0;
#else
  return 0;
#endif
}

std::vector<InferenceBackend::SessionState> InferenceBackend::session_states()
    const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->scheduler->SessionStates()
                          : std::vector<SessionState>{};
#else
  return {};
#endif
}

InferenceBackend::SamplingDefaults InferenceBackend::sampling_defaults() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->sampling_defaults : SamplingDefaults{};
#else
  return {};
#endif
}

ReasoningOptions InferenceBackend::reasoning_defaults() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->reasoning_defaults : ReasoningOptions{};
#else
  return {};
#endif
}

InferenceBackend::InitialOutputState InferenceBackend::initial_output_state(
    const ChatRequest& request) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr
             ? state->scheduler->runner().InitialOutputState(request)
             : InitialOutputState::kAuto;
#else
  (void)request;
  return InitialOutputState::kAuto;
#endif
}

bool InferenceBackend::muse_channel_parse() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr && state->scheduler->runner().MuseChannelParse();
#else
  return false;
#endif
}

bool InferenceBackend::lfm2_pythonic_tools() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr && state->scheduler->runner().Lfm2PythonicTools();
#else
  return false;
#endif
}

// Ling3
bool InferenceBackend::unconstrained_native_tools() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr &&
         state->scheduler->runner().UnconstrainedNativeTools();
#else
  return false;
#endif
}

std::string InferenceBackend::model_path() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->model_path : std::string{};
#else
  return {};
#endif
}

std::string InferenceBackend::build_info() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->build_info : std::string{};
#else
  return {};
#endif
}

void InferenceBackend::set_served_artifact(const std::string& model_path,
                                           const std::string& build_info) {
#if defined(ENGINE_ENABLE_HIP)
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->model_path = model_path;
  updated->build_info = build_info;
  impl_->state = std::move(updated);
#else
  (void)model_path;
  (void)build_info;
#endif
}

void InferenceBackend::set_model_id(const std::string& model_id) {
#if defined(ENGINE_ENABLE_HIP)
  if (model_id.empty()) {
    return;
  }
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->model_id = model_id;
  impl_->state = std::move(updated);
#else
  (void)model_id;
#endif
}

void InferenceBackend::set_sampling_defaults(
    std::size_t max_tokens, const sampling::SamplingConfig& sampling_config,
    sampling::SamplingOverrides supplied) {
#if defined(ENGINE_ENABLE_HIP)
  sampling_config.Validate();
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->sampling_defaults.max_tokens = max_tokens;
  updated->sampling_defaults.sampling = sampling_config;
  updated->sampling_defaults.supplied = supplied;
  impl_->state = std::move(updated);
#else
  (void)max_tokens;
  (void)sampling_config;
  (void)supplied;
#endif
}

void InferenceBackend::set_reasoning_defaults(
    const ReasoningOptions& reasoning) {
#if defined(ENGINE_ENABLE_HIP)
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->reasoning_defaults = reasoning;
  impl_->state = std::move(updated);
#else
  (void)reasoning;
#endif
}

InferenceBackend::Result InferenceBackend::complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token,
    std::string_view client_id, const std::vector<std::string>& stop_sequences,
    bool cache_prompt) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  // Raw prompt, no chat template: the GGUF BOS the model asks for belongs
  // here, and usage.prompt_tokens counts it.
  auto prompt_tokens = state->scheduler->runner().TokenizeRawPrompt(prompt);
  return impl_->GenerateScheduled(
      state, std::move(prompt_tokens), request_start, max_tokens,
      sampling_config, is_cancelled, on_token, std::string(client_id), {},
      cache_prompt, 0, stop_sequences);
#else
  (void)cache_prompt;
  (void)client_id;
  (void)stop_sequences;
  (void)prompt;
  (void)max_tokens;
  (void)sampling_config;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const ChatRequest& request, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  auto effective_sampling = sampling_config;
  auto constrained = ConstrainChatRequest(request, state->scheduler->runner(),
                                          &effective_sampling);
  const auto& effective_request = constrained ? *constrained : request;
  auto prompt = state->scheduler->runner().PreparePrompt(effective_request);
  if (!prompt.has_value() || prompt->tokens.empty()) {
    return {};
  }
  const auto initial =
      state->scheduler->runner().InitialOutputState(effective_request);
  if (request.reasoning.budget_tokens &&
      initial == TextGenerationBackend::InitialOutputState::kReasoning)
    effective_sampling.reasoning_budget =
        state->scheduler->runner().BuildReasoningBudget(
            *request.reasoning.budget_tokens, prompt->tokens);
  if (initial == TextGenerationBackend::InitialOutputState::kReasoning) {
    MaybeArmReasoningLoopGuard(state->scheduler->runner(), &effective_sampling);
    MaybeArmReasoningWordBan(state->scheduler->runner(), &effective_sampling);
  }
  return impl_->GenerateScheduled(
      state, std::move(prompt->tokens), request_start, max_tokens,
      effective_sampling, is_cancelled, on_token, request.client_id,
      std::move(prompt->context), request.cache_prompt,
      prompt->cache_prefix_tokens, request.stop_sequences, initial);
#else
  (void)request;
  (void)max_tokens;
  (void)sampling_config;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

std::shared_ptr<InferenceBackend::GenerationRequest>
InferenceBackend::start_complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, bool stream_output, bool ignore_eos,
    std::string_view client_id, const std::vector<std::string>& stop_sequences,
    bool return_progress, bool cache_prompt) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return TextGenerationBackend::start_complete(
        prompt, max_tokens, sampling_config, is_cancelled, stream_output,
        ignore_eos, client_id, stop_sequences, return_progress, cache_prompt);
  }
  // Raw prompt, no chat template: the GGUF BOS the model asks for belongs
  // here, and usage.prompt_tokens counts it.
  auto prompt_tokens = state->scheduler->runner().TokenizeRawPrompt(prompt);
  auto scheduled_request = state->scheduler->Submit(
      std::move(prompt_tokens), max_tokens, sampling_config, is_cancelled,
      stream_output,
      TextRequestMetadata{
          .client_id = client_id.empty() ? "anonymous" : std::string(client_id),
          .deadline = std::nullopt,
          .request_start = request_start,
          .prompt_context = {},
          .cache_prompt = cache_prompt,
          .cache_prefix_tokens = 0,
          .stop_at_eos = !ignore_eos,
          .stop_sequences = stop_sequences,
          .return_progress = return_progress,
      });
  return std::make_shared<Impl::ScheduledGenerationRequest>(
      state, std::move(scheduled_request));
#else
  return TextGenerationBackend::start_complete(
      prompt, max_tokens, sampling_config, is_cancelled, stream_output,
      ignore_eos, client_id, stop_sequences, return_progress, cache_prompt);
#endif
}

void InferenceBackend::WarmReasoningCaches() {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr || state->scheduler == nullptr)
    return;
  // Builds (and logs) the process-wide word-ban token set and the closing
  // message tokens once, before serving, so process start carries the
  // per-word counts. Each builder is a no-op when its option is unset.
  (void)state->scheduler->runner().BuildReasoningWordBan();
  (void)state->scheduler->runner().ReasoningCloseMessageTokens();
#endif
}

std::size_t InferenceBackend::warm_up(std::size_t max_tokens) {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr || state->scheduler == nullptr || max_tokens == 0) {
    return 0;
  }
  // Plain prose as a raw prompt: no chat template, so a chat request shares
  // none of its tokens and starts like any new conversation, and ordinary
  // text gives the drafter a mix of accepted and rejected rounds to run.
  auto prompt_tokens = state->scheduler->runner().TokenizeRawPrompt(
      "The following is a short note about the history of the bicycle.\n\n"
      "The bicycle");
  if (prompt_tokens.empty()) {
    return 0;
  }
  sampling::SamplingConfig greedy;
  greedy.temperature = 0.0F;
  auto scheduled_request = state->scheduler->Submit(
      std::move(prompt_tokens), max_tokens, greedy, {}, false,
      TextRequestMetadata{
          .client_id = "warmup",
          .deadline = std::nullopt,
          .request_start = Clock::now(),
          .prompt_context = {},
          .cache_prompt = false,
          .cache_prefix_tokens = 0,
          .stop_at_eos = false,
          .stop_sequences = {},
          .return_progress = false,
          .warmup = true,
      });
  Impl::ScheduledGenerationRequest generation(state,
                                              std::move(scheduled_request));
  return generation.Wait({}, {}, {}).tokens.size();
#else
  (void)max_tokens;
  return 0;
#endif
}

std::shared_ptr<InferenceBackend::GenerationRequest>
InferenceBackend::start_chat(const ChatRequest& request, std::size_t max_tokens,
                             const sampling::SamplingConfig& sampling_config,
                             const CancellationCheck& is_cancelled,
                             bool stream_output) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return TextGenerationBackend::start_chat(
        request, max_tokens, sampling_config, is_cancelled, stream_output);
  }

  auto effective_sampling = sampling_config;
  std::optional<sampling::JsonConstraint::ToolFormat> tool_format;
  auto constrained = ConstrainChatRequest(request, state->scheduler->runner(),
                                          &effective_sampling, &tool_format);
  const auto& effective_request = constrained ? *constrained : request;
  auto prompt = state->scheduler->runner().PreparePrompt(effective_request);
  if (!prompt.has_value() || prompt->tokens.empty()) {
    return TextGenerationBackend::start_chat(
        request, max_tokens, sampling_config, is_cancelled, stream_output);
  }

  const auto initial =
      state->scheduler->runner().InitialOutputState(effective_request);
  if (request.reasoning.budget_tokens &&
      initial == TextGenerationBackend::InitialOutputState::kReasoning)
    effective_sampling.reasoning_budget =
        state->scheduler->runner().BuildReasoningBudget(
            *request.reasoning.budget_tokens, prompt->tokens);
  if (initial == TextGenerationBackend::InitialOutputState::kReasoning) {
    MaybeArmReasoningLoopGuard(state->scheduler->runner(), &effective_sampling);
    MaybeArmReasoningWordBan(state->scheduler->runner(), &effective_sampling);
  }
  const std::string client_id =
      request.client_id.empty() ? "anonymous" : request.client_id;
  auto scheduled_request = state->scheduler->Submit(
      std::move(prompt->tokens), max_tokens, effective_sampling, is_cancelled,
      stream_output,
      TextRequestMetadata{
          .client_id = client_id,
          .deadline = std::nullopt,
          .request_start = request_start,
          .prompt_context = std::move(prompt->context),
          .cache_prompt = request.cache_prompt,
          .cache_prefix_tokens = prompt->cache_prefix_tokens,
          .stop_sequences = request.stop_sequences,
          .stop_reasoning_ends = StopReasoningEnds(
              state->scheduler->runner(), request.stop_sequences, initial),
          .return_progress = request.return_progress,
      });
  return std::make_shared<Impl::ScheduledGenerationRequest>(
      state, std::move(scheduled_request), initial, tool_format);
#else
  return TextGenerationBackend::start_chat(request, max_tokens, sampling_config,
                                           is_cancelled, stream_output);
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const std::vector<tokenization::ChatMessage>& messages,
    std::size_t max_tokens, const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled) {
  return chat(ChatRequest{messages}, max_tokens, sampling_config, is_cancelled);
}

std::size_t InferenceBackend::count_tokens(std::string_view text) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return 0;
  }
  // Same tokens a raw generation would see, BOS included.
  return state->scheduler->runner().TokenizeRawPrompt(text).size();
#else
  (void)text;
  return 0;
#endif
}

std::optional<std::size_t> InferenceBackend::count_chat_tokens(
    const ChatRequest& request) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return std::nullopt;
  }
  // The same preparation as chat(): constrained tool or format requests add
  // prompt text before the template is rendered.
  auto sampling = sampling::SamplingConfig{};
  auto constrained =
      ConstrainChatRequest(request, state->scheduler->runner(), &sampling);
  const auto& effective_request = constrained ? *constrained : request;
  return state->scheduler->runner().CountPromptTokens(effective_request);
#else
  (void)request;
  return std::nullopt;
#endif
}

}  // namespace superroc::server
