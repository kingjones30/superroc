#ifndef SUPERROC_SERVER_INFERENCE_BACKEND_HPP_
#define SUPERROC_SERVER_INFERENCE_BACKEND_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/models/qwen/dflash_policy.hpp"

namespace superroc::hip {
class QwenGpuModel;
}

namespace superroc::models::deepseek_v4_flash {
class Model;
}

namespace superroc::models::qwen38_flash_next {
class Model;
}

// Ling-3.0-flash (bailingmoe3). Deliberately NOT HIP-guarded: the CPU
// reference backend is a complete implementation, so `superroc serve` can serve
// Ling in a CPU-only build as well.
namespace superroc::models::ling3 {
class Model;
}  // namespace superroc::models::ling3

namespace superroc::models::ling3::vision {
class VisionTower;
}

namespace superroc::tokenization {
class QwenTokenizer;
}

namespace superroc::server {

enum class TextSpeculativeBackend : std::uint8_t {
  kDisabled,
  kDFlash,
  kDSpark,
  kMtp,  ///< Qwen MTP draft block
};

struct TextSpeculativeConfig {
  TextSpeculativeBackend backend{TextSpeculativeBackend::kDisabled};
  std::string draft_model_path;
  std::uint32_t max_draft_tokens{7};
  /// The draft length was not given: the backend may use the default of the
  /// drafter it loads instead of max_draft_tokens.
  bool default_draft_tokens{false};
  std::uint32_t min_draft_tokens{1};
  speculative::DFlashDraftPolicy dflash_policy{
      speculative::DFlashDraftPolicy::kAdaptive};
  bool prompt_lookup{false};
  /// No speculative option was given. A dense Qwen file that carries its own
  /// MTP block then drafts with it; any other model stays without a drafter.
  bool automatic{false};
};

struct TextDiskCacheConfig {
  std::filesystem::path directory;
  std::size_t capacity_bytes{TextRunnerDiskCacheOptions::kDefaultCapacityBytes};
  /// Zero selects the automatic host RAM budget after session allocation.
  std::size_t staging_capacity_bytes{0};
  std::string model_artifact_fingerprint;
  std::string draft_model_artifact_fingerprint;
};

/// Thread-safe HTTP inference facade over shared immutable GPU model resources
/// and a bounded pool of request-owned executor sessions.
class InferenceBackend final : public TextGenerationBackend {
public:
  using TextGenerationBackend::chat;
  using TextGenerationBackend::complete;
  using TextGenerationBackend::start_chat;

  InferenceBackend();
  ~InferenceBackend() override;

  InferenceBackend(const InferenceBackend&) = delete;
  InferenceBackend& operator=(const InferenceBackend&) = delete;
  InferenceBackend(InferenceBackend&&) = delete;
  InferenceBackend& operator=(InferenceBackend&&) = delete;

  /// Loads weights from a GGUF file. Returns false and sets *error on failure.
  bool load(const std::string& model_path, std::string* error,
            std::uint32_t max_context = 0, std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            const TextSpeculativeConfig& speculative_config = {},
            const TextDiskCacheConfig& disk_cache_config = {},
            const std::string& vision_model_path = {},
            TextRunnerRamCacheOptions ram_cache_config = {});

#if defined(ENGINE_ENABLE_HIP)
  /// Installs a previously loaded model without duplicating mapped weights.
  bool load(std::shared_ptr<const hip::QwenGpuModel> model, std::string* error,
            std::uint32_t max_context = 0, std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            TextSpeculativeConfig speculative_config = {},
            TextDiskCacheConfig disk_cache_config = {},
            TextRunnerRamCacheOptions ram_cache_config = {});

  /// Installs a previously loaded DeepSeek model with request-owned sessions.
  bool load(std::shared_ptr<models::deepseek_v4_flash::Model> model,
            std::string* error, std::uint32_t max_context = 0,
            std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            TextSpeculativeConfig speculative_config = {},
            TextDiskCacheConfig disk_cache_config = {},
            TextRunnerRamCacheOptions ram_cache_config = {});

  /// Installs a previously loaded Qwen3.8-Flash-Next model with
  /// request-owned sessions, the model's tokenizer, and host-memory
  /// continuation snapshots.
  bool load(std::shared_ptr<models::qwen38_flash_next::Model> model,
            std::string* error, std::uint32_t max_context = 0,
            std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            TextSpeculativeConfig speculative_config = {},
            TextDiskCacheConfig disk_cache_config = {},
            TextRunnerRamCacheOptions ram_cache_config = {});
#endif

  /// Installs a previously loaded Ling-3.0-flash (bailingmoe3) model with
  /// request-owned sessions and the model's own tokenizer and chat template.
  /// Serial plan only: Ling's KDA recurrence is token-sequential in the fork.
  bool load(
      std::shared_ptr<models::ling3::Model> model, std::string* error,
      std::uint32_t max_context = 0, std::size_t session_count = 1,
      TextPrefillPolicy prefill_policy = {},
      TextSchedulerPolicy scheduler_policy = {},
      TextSpeculativeConfig speculative_config = {},
      TextDiskCacheConfig disk_cache_config = {},
      std::shared_ptr<models::ling3::vision::VisionTower> vision_tower = {});

  /// Stable model identifier used in API responses.
  [[nodiscard]] std::string model_id() const override;
  [[nodiscard]] bool ready() const override;
  [[nodiscard]] bool device_lost() const override;
  [[nodiscard]] bool supports_images() const override;
  [[nodiscard]] std::string model_path() const override;
  [[nodiscard]] std::string build_info() const override;
  [[nodiscard]] SamplingDefaults sampling_defaults() const override;
  [[nodiscard]] std::uint32_t max_context() const override;
  [[nodiscard]] std::vector<SessionState> session_states() const override;
  [[nodiscard]] ReasoningOptions reasoning_defaults() const override;
  [[nodiscard]] InitialOutputState initial_output_state(
      const ChatRequest& request) const override;
  [[nodiscard]] bool muse_channel_parse() const override;
  [[nodiscard]] bool lfm2_pythonic_tools() const override;
  [[nodiscard]] bool unconstrained_native_tools() const override;  // Ling3
  [[nodiscard]] std::string speculative_status() const override;
  /// The draft length the loaded drafter uses when it chose it itself (a
  /// model file's own MTP block, or no --draft-tokens); zero otherwise.
  [[nodiscard]] std::uint32_t speculative_draft_limit() const;
  void set_model_id(const std::string& model_id);
  void set_served_artifact(const std::string& model_path,
                           const std::string& build_info);
  void set_sampling_defaults(std::size_t max_tokens,
                             const sampling::SamplingConfig& sampling,
                             sampling::SamplingOverrides supplied =
                                 sampling::SamplingOverrides::All());
  void set_reasoning_defaults(const ReasoningOptions& reasoning);

  /// Plain text completion (no chat framing).
  Result complete(std::string_view prompt, std::size_t max_tokens,
                  const sampling::SamplingConfig& sampling,
                  const CancellationCheck& is_cancelled = {},
                  const TokenCallback& on_token = {},
                  std::string_view client_id = "anonymous",
                  const std::vector<std::string>& stop_sequences = {},
                  bool cache_prompt = true) override;

  /// Framed chat conversation; returns the assistant reply.
  Result chat(const ChatRequest& request, std::size_t max_tokens,
              const sampling::SamplingConfig& sampling,
              const CancellationCheck& is_cancelled = {},
              const TokenCallback& on_token = {}) override;

  std::shared_ptr<GenerationRequest> start_chat(
      const ChatRequest& request, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {},
      bool stream_output = false) override;

  std::shared_ptr<GenerationRequest> start_complete(
      std::string_view prompt, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {}, bool stream_output = false,
      bool ignore_eos = false, std::string_view client_id = "anonymous",
      const std::vector<std::string>& stop_sequences = {},
      bool return_progress = false, bool cache_prompt = true) override;

  /// Runs one short greedy generation through the normal request path, so
  /// the first real request does not pay the first-use costs (kernel code,
  /// decode graphs, draft buffers, fresh scratch pages). It is a raw-text
  /// prompt that reuses nothing, leaves no state behind and is not a request
  /// in the metrics. Returns the tokens generated; zero when no text model
  /// that runs through the scheduler is loaded.
  std::size_t warm_up(std::size_t max_tokens);

  /// Builds the process-wide reasoning caches (word-ban token set and
  /// closing-message tokens) once, so process start logs the per-word token
  /// counts before serving. No-op when both options are unset.
  void WarmReasoningCaches();

  Result chat(const std::vector<tokenization::ChatMessage>& messages,
              std::size_t max_tokens, const sampling::SamplingConfig& sampling,
              const CancellationCheck& is_cancelled = {});

  /// Token count of raw text (no generation).
  [[nodiscard]] std::size_t count_tokens(std::string_view text) const override;

  /// Prompt tokens of a chat request as generation would prepare them.
  [[nodiscard]] std::optional<std::size_t> count_chat_tokens(
      const ChatRequest& request) const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace superroc::server

#endif  // SUPERROC_SERVER_INFERENCE_BACKEND_HPP_
