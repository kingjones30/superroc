#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"
#include "src/core/speculative/speculative_verifier.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace {

using superroc::speculative::DraftProposal;
using superroc::speculative::DraftTargetContext;
using superroc::speculative::IDraftBackend;
using superroc::speculative::ISpeculativeTargetExecutor;
using superroc::speculative::SpeculativeVerifier;
using superroc::tokenization::TokenId;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

constexpr TokenId kEosId = 7;

/// Minimal target executor: records prompt rows and replays deterministic
/// per-token hidden states for Prime and ExtendPrompt.
class PromptCaptureTarget final : public ISpeculativeTargetExecutor {
public:
  void Reset() noexcept override {
    state_.clear();
    prompt_hidden_.clear();
  }

  TokenId ForwardPromptBatch(std::span<const TokenId> prompt_tokens) override {
    state_.assign(prompt_tokens.begin(), prompt_tokens.end());
    RecordHidden(prompt_tokens);
    return kEosId;
  }

  TokenId ForwardPromptSuffix(std::span<const TokenId> tokens,
                              std::uint32_t position,
                              bool compute_logits) override {
    (void)compute_logits;
    if (position != state_.size()) {
      throw std::logic_error("prompt suffix does not follow target state");
    }
    state_.insert(state_.end(), tokens.begin(), tokens.end());
    RecordHidden(tokens);
    return kEosId;
  }

  TokenId ForwardToken(TokenId token_id, std::uint32_t pos,
                       bool compute_logits) override {
    (void)compute_logits;
    if (pos != state_.size()) {
      throw std::logic_error("decode position does not follow target state");
    }
    state_.push_back(token_id);
    return kEosId;
  }

  void SaveState(std::uint32_t valid_context) override { (void)valid_context; }

  void RestoreState() override {}

  void SetPromptHiddenCapture(
      bool enabled, std::span<const std::uint32_t> target_layer_ids) override {
    (void)target_layer_ids;
    capture_enabled_ = enabled;
    capture_states_.push_back(enabled);
  }

  std::span<const float> GetPromptHiddenStates() const noexcept override {
    return prompt_hidden_;
  }

  TokenId GetEosTokenId() const noexcept override { return kEosId; }

  std::string_view DecodeToken(TokenId token_id) const noexcept override {
    (void)token_id;
    return "token";
  }

  [[nodiscard]] bool CaptureEnabled() const noexcept {
    return capture_enabled_;
  }

  [[nodiscard]] const std::vector<bool>& CaptureStates() const noexcept {
    return capture_states_;
  }

private:
  void RecordHidden(std::span<const TokenId> tokens) {
    prompt_hidden_.clear();
    for (const auto token : tokens) {
      prompt_hidden_.push_back(static_cast<float>(token));
      prompt_hidden_.push_back(static_cast<float>(token) * 0.5F);
      prompt_hidden_.push_back(-static_cast<float>(token));
    }
  }

  std::vector<TokenId> state_;
  std::vector<float> prompt_hidden_;
  std::vector<bool> capture_states_;
  bool capture_enabled_{false};
};

struct ReceivedContext {
  std::vector<TokenId> tokens;
  std::size_t hidden_width{0};
  std::uint32_t skip_tokens{0};
  std::uint32_t position{0};
};

/// Draft backend with a bounded history window that records every primed and
/// appended prompt context.
class WindowedRecordingBackend final : public IDraftBackend {
public:
  explicit WindowedRecordingBackend(std::uint32_t history_capacity)
      : history_capacity_(history_capacity) {}

  std::string_view Name() const noexcept override {
    return "WindowedRecordingBackend";
  }

  DraftProposal Propose(std::span<const TokenId> prompt_tokens,
                        std::uint32_t current_pos,
                        std::uint32_t max_tokens) override {
    (void)prompt_tokens;
    (void)current_pos;
    (void)max_tokens;
    return {};
  }

  bool RequiresTargetHiddenStates() const noexcept override { return true; }

  std::uint32_t TargetHistoryCapacity() const noexcept override {
    return history_capacity_;
  }

  bool PrimeTargetContext(const DraftTargetContext& context) override {
    primed_.push_back(Record(context, 0));
    return true;
  }

  bool AppendTargetContext(const DraftTargetContext& context,
                           std::uint32_t position) override {
    appended_.push_back(Record(context, position));
    return true;
  }

  const std::vector<ReceivedContext>& Primed() const noexcept {
    return primed_;
  }

  const std::vector<ReceivedContext>& Appended() const noexcept {
    return appended_;
  }

private:
  static ReceivedContext Record(const DraftTargetContext& context,
                                std::uint32_t position) {
    ReceivedContext received;
    received.tokens.assign(context.prompt_tokens.begin(),
                           context.prompt_tokens.end());
    received.hidden_width = context.hidden_size;
    received.skip_tokens = context.skip_tokens;
    received.position = position;
    return received;
  }

  std::uint32_t history_capacity_{0};
  std::vector<ReceivedContext> primed_;
  std::vector<ReceivedContext> appended_;
};

std::vector<TokenId> MakeTokens(std::uint32_t count, TokenId first) {
  std::vector<TokenId> tokens;
  tokens.reserve(count);
  for (std::uint32_t index = 0; index < count; ++index) {
    tokens.push_back(static_cast<TokenId>(first + index));
  }
  return tokens;
}

void TestPrimeSkipsRowsOutsideTheDrafterWindow() {
  PromptCaptureTarget target;
  auto backend = std::make_unique<WindowedRecordingBackend>(64U /*window*/);
  auto* backend_view = backend.get();
  SpeculativeVerifier verifier(target, std::move(backend));

  const auto prompt = MakeTokens(100U, 1U);
  (void)verifier.Prime(prompt, true, 300U /*total prompt*/);
  Expect(backend_view->Primed().size() == 1,
         "prime reaches the draft backend once");
  const auto& primed = backend_view->Primed().front();
  Expect(primed.tokens.size() == 100,
         "prime passes the whole prompt chunk, not a sliced one");
  Expect(primed.hidden_width == 0,
         "a fully skipped prime chunk carries no features");
  Expect(primed.skip_tokens == 100,
         "rows below total - window are skipped for the whole chunk");
}

void TestPrimeSplitsAStraddlingChunk() {
  PromptCaptureTarget target;
  auto backend = std::make_unique<WindowedRecordingBackend>(64U);
  auto* backend_view = backend.get();
  SpeculativeVerifier verifier(target, std::move(backend));

  const auto prompt = MakeTokens(100U, 1U);
  (void)verifier.Prime(prompt, true, 100U);
  Expect(backend_view->Primed().front().skip_tokens == 36,
         "a chunk straddling total - window skips only its leading rows");
}

void TestUnknownTotalPromptLengthSkipsNothing() {
  PromptCaptureTarget target;
  auto backend = std::make_unique<WindowedRecordingBackend>(64U);
  auto* backend_view = backend.get();
  SpeculativeVerifier verifier(target, std::move(backend));

  const auto prompt = MakeTokens(100U, 1U);
  (void)verifier.Prime(prompt, true);
  Expect(backend_view->Primed().front().skip_tokens == 0,
         "unknown final prompt length disables the skip");
}

void TestUnboundedWindowSkipsNothing() {
  PromptCaptureTarget target;
  auto backend = std::make_unique<WindowedRecordingBackend>(0U);
  auto* backend_view = backend.get();
  SpeculativeVerifier verifier(target, std::move(backend));

  const auto prompt = MakeTokens(100U, 1U);
  (void)verifier.Prime(prompt, true, 300U);
  Expect(backend_view->Primed().front().skip_tokens == 0,
         "a drafter without a bounded history window skips nothing");
}

void TestShortPromptSkipsNothing() {
  PromptCaptureTarget target;
  auto backend = std::make_unique<WindowedRecordingBackend>(64U);
  auto* backend_view = backend.get();
  SpeculativeVerifier verifier(target, std::move(backend));

  const auto prompt = MakeTokens(30U, 1U);
  (void)verifier.Prime(prompt, true, 30U);
  Expect(backend_view->Primed().front().skip_tokens == 0,
         "a prompt inside the window skips nothing");
}

void TestFullySkippedChunkNeedsNoFeatures() {
  PromptCaptureTarget target;
  auto backend = std::make_unique<WindowedRecordingBackend>(64U);
  auto* backend_view = backend.get();
  SpeculativeVerifier verifier(target, std::move(backend));

  const auto first = MakeTokens(40U, 1U);
  (void)verifier.Prime(first, false, 200U);
  Expect(backend_view->Primed().front().skip_tokens == 40,
         "the fully skipped prime chunk skips entirely");
  Expect(backend_view->Primed().front().hidden_width == 0,
         "a fully skipped chunk carries no features");
  Expect(!target.CaptureStates().empty() && !target.CaptureStates().back(),
         "a fully skipped chunk disables the target hidden capture");

  const auto second = MakeTokens(40U, 41U);
  (void)verifier.ExtendPrompt(second, 40U, false, 200U);
  const auto third = MakeTokens(40U, 81U);
  (void)verifier.ExtendPrompt(third, 80U, false, 200U);
  const auto fourth = MakeTokens(40U, 121U);
  (void)verifier.ExtendPrompt(fourth, 120U, true, 200U);
  Expect(backend_view->Appended()[0].hidden_width == 0 &&
             backend_view->Appended()[1].hidden_width == 0,
         "fully skipped extension chunks carry no features");
  Expect(backend_view->Appended()[2].hidden_width == 3,
         "a straddling chunk still captures features");
}

void TestExtendPromptSkipsPerChunkBoundary() {
  PromptCaptureTarget target;
  auto backend = std::make_unique<WindowedRecordingBackend>(64U);
  auto* backend_view = backend.get();
  SpeculativeVerifier verifier(target, std::move(backend));

  const auto first = MakeTokens(40U, 1U);
  (void)verifier.Prime(first, false, 200U);
  Expect(backend_view->Primed().front().skip_tokens == 40,
         "the first chunk below the boundary skips entirely");

  const auto second = MakeTokens(40U, 41U);
  (void)verifier.ExtendPrompt(second, 40U, false, 200U);
  const auto third = MakeTokens(40U, 81U);
  (void)verifier.ExtendPrompt(third, 80U, false, 200U);
  const auto fourth = MakeTokens(40U, 121U);
  (void)verifier.ExtendPrompt(fourth, 120U, true, 200U);
  const auto fifth = MakeTokens(40U, 161U);
  (void)verifier.ExtendPrompt(fifth, 160U, true, 200U);

  Expect(backend_view->Appended().size() == 4,
         "every extension chunk reaches the draft backend");
  Expect(backend_view->Appended()[0].position == 40 &&
             backend_view->Appended()[0].skip_tokens == 40,
         "a chunk fully below the boundary skips entirely");
  Expect(backend_view->Appended()[1].position == 80 &&
             backend_view->Appended()[1].skip_tokens == 40,
         "a chunk still below the boundary skips entirely");
  Expect(backend_view->Appended()[2].position == 120 &&
             backend_view->Appended()[2].skip_tokens == 16,
         "a chunk straddling the boundary skips its leading rows");
  Expect(backend_view->Appended()[3].position == 160 &&
             backend_view->Appended()[3].skip_tokens == 0,
         "the final chunk past the boundary stays fully injected");
  Expect(backend_view->Appended()[1].tokens.size() == 40,
         "extensions pass the whole chunk and let the backend slice it");
  Expect(backend_view->Appended()[2].hidden_width == 3,
         "a chunk with live rows still captures features");
}

}  // namespace

int main() {
  TestPrimeSkipsRowsOutsideTheDrafterWindow();
  TestFullySkippedChunkNeedsNoFeatures();
  TestPrimeSplitsAStraddlingChunk();
  TestUnknownTotalPromptLengthSkipsNothing();
  TestUnboundedWindowSkipsNothing();
  TestShortPromptSkipsNothing();
  TestExtendPromptSkipsPerChunkBoundary();
  std::cout << "All speculative prompt-skip tests passed.\n";
  return 0;
}

#else
int main() {
  std::cout << "HIP disabled, skipping speculative prompt-skip tests.\n";
  return 0;
}
#endif
