#ifndef SUPERROC_MODELS_LING3_ENGINE_HPP_
#define SUPERROC_MODELS_LING3_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/ling3/config.hpp"
#include "src/models/ling3/mtp_gpu.hpp"
#include "src/models/ling3/tokenizer.hpp"
#include "src/models/ling3/weights.hpp"

namespace superroc::models::ling3 {

namespace vision {
struct PreparedPrompt;
}

/// LS3 batched-prefill knobs (read once per process):
///   LING3_PREFILL_BATCH=0  disables batching (token-at-a-time, the old path)
///   LING3_PREFILL_CHUNK=N  tokens per ForwardBatch call, clamped to [1, 512]
bool PrefillBatchEnabled();
std::uint32_t PrefillBatchChunk();

class Session;
class Model;

/// Which execution path a Model uses.
enum class BackendKind {
  /// The scalar float32 reference in reference.cpp. Always available, and the
  /// oracle every other path is checked against.
  kCpuReference,
  /// The ROCm path: device weights and the kernels under kernels/.
  kRocm,
};

struct ModelOptions {
  /// Zero selects the artifact's own `context_length`.
  std::uint32_t max_context{0};
  BackendKind backend{BackendKind::kCpuReference};
  /// Prefill is token-at-a-time in both paths tonight; speed is not a gate and
  /// the KDA recurrence is token-sequential in the fork anyway.
  std::uint32_t prefill_batch{1};
  /// ROCm only. False keeps all 512 routed experts resident per MoE layer
  /// (~59.8 GiB here, which fits 128 GiB of GTT) and needs no per-token
  /// staging; true streams the 8 selected experts instead, for a smaller box.
  bool stream_experts{false};
};

/// One conversation's state. Prefill and Evaluate share a single position
/// counter; the recurrent state (KDA conv + SSM) and the MLA latent cache live
/// inside whichever Backend is running.
class Session final : public std::enable_shared_from_this<Session> {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Makes the session equal to `prompt`: keeps the longest common prefix with
  /// the tokens already held, reprocesses the rest.
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  /// Runs one token at the current position and leaves its logits behind.
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  /// LS4's MTP probe: while `Sync` is replaying a prompt the positions are
  /// prompt positions, not decode steps, so the probe must not score a draft
  /// made there. `Evaluate` reads this; only `Sync` sets it.
  [[nodiscard]] bool in_prefill() const noexcept { return in_prefill_; }
  /// Greedy argmax of the current logits, without advancing. The serve layer
  /// samples from the logits span itself (that is what the Flash-Next runner
  /// does), so the engine deliberately carries no sampler of its own.
  [[nodiscard]] std::int32_t Argmax() const noexcept;

  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return logits_;
  }
  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  [[nodiscard]] const std::vector<std::int32_t>& Tokens() const noexcept {
    return tokens_;
  }
  void Reset();
  void SetCancellationCheck(std::function<bool()> check) {
    cancel_ = std::move(check);
  }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;

  /// How many tokens one Prefill call may consume. LS3 2026-10-01: batched
  /// prefill runs chunks of up to this many tokens in one pass per layer
  /// (ForwardBatch); the effective chunk is min(this, LING3_PREFILL_CHUNK).
  /// LING3_PREFILL_BATCH=0 caps prefill back to token-at-a-time.
  static constexpr std::uint32_t kPrefillChunk = 512;

  /// LS4 (LING3_MTP=1): one speculative decode step.
  ///
  /// `base` is the token about to be fed -- the model's own argmax from the
  /// previous step. This drafts up to n_max tokens with the MTP head, verifies
  /// them together with `base` in ONE batched ForwardBatch, accepts the
  /// greedy-matching prefix, rewinds the trunk's KDA state when a draft was
  /// rejected, and appends every consumed token to tokens_.
  ///
  /// Afterwards `logits_` holds the model's distribution for the next position
  /// and `Position()` is exactly as if the appended tokens had been fed one at
  /// a time -- so at temperature 0 the emitted text is byte-identical to what
  /// Evaluate() alone produces.
  ///
  /// Falls back to a plain Evaluate(base) whenever speculation is off or
  /// cannot run, so the caller never has to check anything.
  /// `consumed` receives every token this step fed to the trunk -- `base` plus
  /// whichever drafts were accepted -- so the caller can emit their text. It is
  /// the same vector that was appended to Tokens().
  [[nodiscard]] bool EvaluateSpeculative(std::int32_t base,
                                         std::vector<std::int32_t>* consumed,
                                         std::string* error_msg = nullptr);

  /// LS4: the process-wide counters and the per-session draft length, for the
  /// runner's end-of-request report.
  [[nodiscard]] const mtp::SpecStats& SpecStats() const noexcept {
    return spec_stats_;
  }
  [[nodiscard]] int SpecDraftLength() const noexcept;
  [[nodiscard]] mtp::SpecPolicy SpecPolicy() const { return spec_policy_; }

  /// LS3: prefill `tokens` in one batched pass (ForwardBatch on the ROCm
  /// backend; the CPU reference deliberately loops single-token steps because
  /// it is the oracle). Leaves the logits of the LAST token in logits_.
  [[nodiscard]] bool EvaluateBatch(std::span<const std::int32_t> tokens,
                                   std::string* error_msg = nullptr);
  /// Publish image rows and M-RoPE for the request that is about to prefill.
  void BindVision(const vision::PreparedPrompt& prepared);
  /// Drop them. A text request must not keep the previous image's positions.
  void UnbindVision();
  /// Image prompt: arm M-RoPE and projected rows, then prefill `prepared`.
  [[nodiscard]] bool PrefillVision(const vision::PreparedPrompt& prepared,
                                   std::string* error_msg = nullptr);

private:
  friend class Model;
  Session(std::shared_ptr<Model> model, std::uint32_t max_context);

  [[nodiscard]] bool CheckCancel(std::string* error_msg) const;

  /// SPEC6 item 4: report the request that just finished and zero the tally.
  /// Called at every new prompt (EvaluateBatch), at Reset, and from the
  /// destructor for the last one -- each guarded by spec_stats_.steps, so a
  /// path that fires twice reports once.
  void FlushSpecReport();

  std::shared_ptr<Model> model_;
  std::uint32_t max_context_{0};
  std::uint32_t position_{0};
  std::vector<std::int32_t> tokens_;
  std::vector<float> logits_;
  std::function<bool()> cancel_;
  bool in_prefill_{false};
  bool vision_guard_{false};
  /// LS4: this session's speculative state -- the head's latent cache, the
  /// KDA snapshot and the adaptive-policy window. Owned by the session and
  /// destroyed with it. Raw rather than a smart pointer because SpecState is
  /// only forward-declared in this header.
  mtp::SpecState* spec_{nullptr};
  mtp::SpecPolicy spec_policy_{};
  mtp::SpecStats spec_stats_{};
  /// SPEC6 item 4: label for FlushSpecReport's row -- which request this is,
  /// and how long the prompt that produced the stats was.
  int spec_report_req_{0};
  int spec_report_prompt_tokens_{0};
  /// LS4: the post-final-norm hidden of the most recent forward, and whether it
  /// is real. Fed to the MTP head as `h_prev`; the batched path writes its own
  /// scratch, so `rocm::ReadHidden` cannot supply it.
  std::vector<float> pending_h_;
  /// Staging for ForwardBatch's per-row hidden during batched prefill. The last
  /// row is kept in pending_h_; the rest only feed the head priming.
  std::vector<float> spec_rows_;
  bool pending_h_valid_{false};
  /// Opaque per-session state owned by the backend (recurrent state, caches).
  /// Destroyed through Model, which knows which backend created it. Raw rather
  /// than a smart pointer because the deleter is the model's, not a type-erased
  /// one; shared_ptr<void> cannot take a plain void* deleter here.
  void* state_{nullptr};
};

/// One resident model, any number of sessions.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::shared_ptr<Session> CreateSession(
      core::SessionMode mode, std::uint32_t max_context,
      std::string* error_msg = nullptr);

  [[nodiscard]] const Config& config() const noexcept {
    return weights_->config;
  }
  [[nodiscard]] const LingTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] std::string TokenText(std::int32_t token) const;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return max_context_;
  }
  [[nodiscard]] std::uint32_t PrefillCapacity() const noexcept {
    return Session::kPrefillChunk;
  }
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] BackendKind backend() const noexcept {
    return options_.backend;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  [[nodiscard]] std::size_t SessionBytes(core::SessionMode mode,
                                         std::uint32_t context) const noexcept;
  [[nodiscard]] std::size_t DeferredScratchBytes() const { return 0; }

  /// Runs one token for `state` (a backend-owned session blob) and writes the
  /// logits. This is the seam the ROCm path and the CPU reference share.
  [[nodiscard]] bool Step(std::int32_t token, void* state,
                          std::span<float> logits, std::string* error_msg);
  /// LS3: batched prefill seam. On the ROCm backend this is one
  /// ForwardBatch pass (out_logits sized vocab for the last position, or
  /// n * vocab for every position); the CPU reference loops Step because it
  /// is the oracle. Advances the backend state by n positions.
  /// LS4: `hidden_out`, when non-empty, receives the post-final-norm hidden of
  /// every row of the batch -- [n][hidden]. The batched path normalises into
  /// its own scratch, so this is the only way a caller can see the hidden state
  /// a batched forward produced.
  /// LS4 phase 3: `hidden_rows_out`, when its size is a whole number of
  /// [hidden] rows and no more than n, receives the LAST that many rows'
  /// post-`output_norm` hidden. The prefill path already normalised every row;
  /// this is how the MTP head gets primed over the prompt from them.
  [[nodiscard]] bool StepBatch(const std::int32_t* tokens, int n, void* state,
                               std::span<float> logits,
                               std::span<float> hidden_out,
                               std::span<float> hidden_rows_out,
                               std::string* error_msg);
  /// LS3: true when this load should feed Prefill in batches (ROCm with the
  /// routed experts resident; the streamed-expert load has no batched path).
  [[nodiscard]] bool PrefersBatchPrefill() const noexcept;
  /// Allocates the backend state for one sequence.
  [[nodiscard]] void* NewState(std::string* error_msg) const;
  void DestroyState(void* state) const noexcept;
  void ResetState(void* state) const;

  /// LS4 (MTP draft head): the opaque backend blob behind `session`, so the
  /// probe in `mtp_gpu.hpp` can pair `mtp::DraftNext` with the trunk's own
  /// step. The blob is owned by the Session and is not to be freed here.
  /// Returns nullptr for a null session or a mismatched model.
  [[nodiscard]] void* ProbeState(const Session* session) const noexcept;

private:
  Model() = default;
  [[nodiscard]] bool Init(const std::string& model_path,
                          const ModelOptions& options, std::string* error_msg);

  ModelOptions options_;
  std::uint32_t max_context_{0};
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<LingTokenizer> tokenizer_;
  std::size_t resident_bytes_{0};
  /// The CPU reference, when that backend is selected.
  struct CpuState;
  CpuState* cpu_state_{nullptr};

  friend class Session;
};

}  // namespace superroc::models::ling3

#endif  // SUPERROC_MODELS_LING3_ENGINE_HPP_
