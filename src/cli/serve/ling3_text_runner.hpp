#ifndef GUFO_SERVER_LING3_TEXT_RUNNER_HPP_
#define GUFO_SERVER_LING3_TEXT_RUNNER_HPP_

#include <cstdint>
#include <memory>
#include <string>

#include "src/cli/serve/text_model_runner.hpp"
#include "src/models/ling3/engine.hpp"
#include "src/models/ling3/vision_prompt.hpp"

namespace gufo::models::ling3::vision {
class VisionTower;
}

namespace gufo::server {

/// Per-request state for Ling-3.0-flash. Holds only the model session plus the
/// position the scheduler believes it is at, which the runner cross-checks so a
/// scheduling bug surfaces instead of silently re-running tokens.
class Ling3RunnerState final : public TextRunnerState {
public:
  explicit Ling3RunnerState(std::shared_ptr<models::ling3::Session> session)
      : session_(std::move(session)) {}

  [[nodiscard]] models::ling3::Session& session() noexcept { return *session_; }
  [[nodiscard]] const models::ling3::Session& session() const noexcept {
    return *session_;
  }
  [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
  void set_position(std::uint32_t p) noexcept { position_ = p; }

  void SetVision(
      std::shared_ptr<const models::ling3::vision::PreparedPrompt> prompt) {
    vision_ = std::move(prompt);
  }
  [[nodiscard]] const models::ling3::vision::PreparedPrompt* Vision()
      const noexcept {
    return vision_.get();
  }

  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override;

  /// ContinuationState: the scheduler drops a state it no longer needs. The
  /// Ling session is owned outright, so invalidating it is a reset -- there is
  /// nothing to hand back to a pool.
  void Invalidate() noexcept override;

private:
  std::shared_ptr<models::ling3::Session> session_;
  std::uint32_t position_{0};
  std::shared_ptr<const models::ling3::vision::PreparedPrompt> vision_;
};

/// Adapts Ling-3.0-flash to the serve text-runner contract.
///
/// Serial plan only, no snapshot, no batching: Ling's KDA recurrence is
/// token-sequential in the fork, and the snapshot family defaults in
/// TextModelRunner throw for runners that do not advertise the capability, so
/// this deliberately leaves `capabilities.snapshot` false.
class Ling3TextRunner final : public TextModelRunner {
public:
  explicit Ling3TextRunner(
      std::shared_ptr<models::ling3::Model> model, std::uint32_t max_context,
      std::string disk_cache_directory = {},
      std::shared_ptr<models::ling3::vision::VisionTower> vision = {});
  ~Ling3TextRunner() override;

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override;
  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override;
  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override;
  /// Ling3: GLM-style <arg_key>/<arg_value> calls from the Bailing-V3
  /// template, parsed after generation (no constraint vocabulary).
  [[nodiscard]] bool UnconstrainedNativeTools() const override { return true; }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override;
  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override;
  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override;
  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override;
  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override;

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override;

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override;
  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override;
  void Advance(TextRunnerState& state, TextRunnerToken token) const override;

  /// LS4 (LING3_MTP=1): the speculative step. The default implementation in
  /// TextModelRunner is the one-token SelectNext/Advance loop; this override
  /// replaces it with Session::EvaluateSpeculative, which drafts with the MTP
  /// head, verifies the whole block in ONE batched ForwardBatch, and can
  /// therefore return several `selections` at once -- plus the
  /// draft_tokens / draft_accepted_tokens the scheduler reports.
  ///
  /// Every returned selection is a token the model itself chose or confirmed,
  /// so the emitted text is byte-identical to the non-speculative path at
  /// temperature 0.
  ///
  /// Falls back to the default one-token behaviour whenever speculation is off,
  /// cannot run, or the remaining budget cannot hold a whole block -- so the
  /// tail of a request decodes normally.
  /// RV4 (Jay build28f): samples the first token without advancing, so the
  /// scheduler can publish it before the next forward. The scheduler passes a
  /// sampler copy; DecodeStep must reproduce this token first (it checks).
  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override;

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override;

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override;

  [[nodiscard]] const models::ling3::Model& model() const noexcept {
    return *model_;
  }

private:
  std::shared_ptr<models::ling3::Model> model_;
  std::uint32_t max_context_{0};
  std::string disk_cache_directory_;
  std::shared_ptr<models::ling3::vision::VisionTower> vision_;
};

/// Casts with a message that names the runner, matching the other runners.
[[nodiscard]] Ling3RunnerState& RequireLing3State(TextRunnerState& state);
[[nodiscard]] const Ling3RunnerState& RequireLing3State(
    const TextRunnerState& state);

}  // namespace gufo::server

#endif  // GUFO_SERVER_LING3_TEXT_RUNNER_HPP_
