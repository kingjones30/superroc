#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"
#include "src/core/speculative/lookup_draft_backend.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace {

using superroc::speculative::DraftProposal;
using superroc::speculative::DraftTargetContext;
using superroc::speculative::IDraftBackend;
using superroc::speculative::LookupDraftBackend;
using superroc::tokenization::TokenId;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

/// Records every call it receives; Propose hands back one fixed token.
class RecordingDraftBackend final : public IDraftBackend {
public:
  struct AppendCall {
    std::vector<TokenId> tokens;
    std::vector<float> hidden;
    std::size_t hidden_size{0};
    TokenId first_token{0};
    std::uint32_t position{0};
  };

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "Recording";
  }

  [[nodiscard]] DraftProposal Propose(std::span<const TokenId> prompt_tokens,
                                      std::uint32_t current_pos,
                                      std::uint32_t max_tokens) override {
    ++propose_calls;
    last_propose_position = current_pos;
    last_propose_max_tokens = max_tokens;
    (void)prompt_tokens;
    DraftProposal proposal;
    proposal.start_pos = current_pos;
    proposal.tokens = {kFixedToken};
    (void)max_tokens;
    return proposal;
  }

  [[nodiscard]] DraftProposal ProposeSampled(
      std::span<const TokenId> prompt_tokens, std::uint32_t current_pos,
      std::uint32_t max_tokens, float temperature,
      std::uint64_t* rng_state) override {
    ++sampled_calls;
    last_sampled_max_tokens = max_tokens;
    (void)prompt_tokens;
    (void)current_pos;
    (void)temperature;
    (void)rng_state;
    DraftProposal proposal;
    proposal.tokens = {kFixedToken};
    return proposal;
  }

  [[nodiscard]] bool SupportsSampledProposals() const noexcept override {
    return true;
  }

  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }

  [[nodiscard]] bool AppendTargetContext(const DraftTargetContext& context,
                                         std::uint32_t position) override {
    AppendCall call;
    call.tokens.assign(context.prompt_tokens.begin(),
                       context.prompt_tokens.end());
    call.hidden.assign(context.prompt_hidden_states.begin(),
                       context.prompt_hidden_states.end());
    call.hidden_size = context.hidden_size;
    call.first_token = context.first_token;
    call.position = position;
    appends.push_back(std::move(call));
    return true;
  }

  void AcceptFeedback(std::span<const TokenId> accepted,
                      TokenId correction_token) override {
    ++feedback_calls;
    last_feedback_correction = correction_token;
    last_feedback_count = accepted.size();
  }

  static constexpr TokenId kFixedToken = 999;

  std::size_t propose_calls{0};
  std::size_t sampled_calls{0};
  std::size_t feedback_calls{0};
  std::uint32_t last_propose_position{0};
  std::uint32_t last_propose_max_tokens{0};
  std::uint32_t last_sampled_max_tokens{0};
  TokenId last_feedback_correction{0};
  std::size_t last_feedback_count{0};
  std::vector<AppendCall> appends;
};

std::vector<TokenId> Sequence() {
  // A (20 tokens) + 20 fresh tokens + the first 14 tokens of A again.
  std::vector<TokenId> tokens;
  for (TokenId i = 10; i < 30; ++i)
    tokens.push_back(i);  // A
  for (TokenId i = 100; i < 120; ++i)
    tokens.push_back(i);  // fresh
  for (TokenId i = 10; i < 24; ++i)
    tokens.push_back(i);  // A[0..13]
  return tokens;
}

/// A request's first speculative step copies nothing, so the sequence that
/// establishes the request boundary must be shorter than the repeated one.
void ProposeOnce(IDraftBackend& backend, const std::vector<TokenId>& tokens,
                 std::uint32_t position) {
  (void)backend.Propose(tokens, position, 7);
}

}  // namespace

int main() {
  // (a) The first Propose of a request goes to the inner drafter.
  {
    auto inner = std::make_unique<RecordingDraftBackend>();
    auto* inner_ptr = inner.get();
    LookupDraftBackend backend(std::move(inner));
    const std::vector<TokenId> tokens{1, 2, 3, 4, 5, 6, 7, 8};
    const auto proposal = backend.Propose(tokens, 8, 7);
    Expect(inner_ptr->propose_calls == 1, "(a) inner drafter was not called");
    Expect(proposal.tokens.size() == 1 &&
               proposal.tokens[0] == RecordingDraftBackend::kFixedToken,
           "(a) proposal did not come from the inner drafter");
    Expect(!backend.ProposalBypassesAdaptiveLength(),
           "(a) first proposal must not claim a bypassing copy");
  }

  // (b) A repeated tail is copied from the session's own committed tokens.
  // (c) The inner drafter still receives the committed rows it never saw.
  {
    auto inner = std::make_unique<RecordingDraftBackend>();
    auto* inner_ptr = inner.get();
    LookupDraftBackend backend(std::move(inner));
    backend.SetAdaptiveDraftLimit(5);
    const std::vector<TokenId> first{10, 11, 12, 13, 14, 15,
                                     16, 17, 18, 19, 20};
    ProposeOnce(backend, first, 21);
    const std::vector<TokenId> sequence = Sequence();
    const auto proposal = backend.Propose(sequence, sequence.size(), 7);
    Expect(inner_ptr->propose_calls == 1,
           "(b) a copied proposal must not reach the inner drafter");
    Expect(backend.ProposalBypassesAdaptiveLength(),
           "(b) copied proposal must bypass the adaptive length");
    const std::uint32_t start = 14;  // the repeat begins at sequence[14]
    Expect(proposal.tokens.size() == 3,
           "(b) a fresh match proposes three rows");
    Expect(proposal.start_pos == sequence.size(),
           "(b) proposal starts at the current position");
    for (std::size_t i = 0; i < proposal.tokens.size(); ++i) {
      Expect(proposal.tokens[i] == sequence[start + i],
             "(b) proposal tokens are not the tokens that followed in A");
    }
    Expect(proposal.tokens[0] == 24,
           "(b) proposals must continue the repeated prefix of A");

    // (c) The copy bypassed the neural drafter, so it is fed the committed
    // rows the verifier would have given it for a step without a proposal.
    const std::vector<float> row_a{1.0F, 2.0F};
    const std::vector<float> row_b{3.0F, 4.0F};
    backend.UpdateTargetHidden(row_a);
    backend.UpdateTargetHidden(row_b);
    const std::vector<TokenId> accepted{proposal.tokens.begin(),
                                        proposal.tokens.begin() + 2};
    backend.AcceptFeedback(accepted, 777);
    Expect(inner_ptr->feedback_calls == 0,
           "(c) a copied round must not feed the inner drafter's feedback");
    Expect(inner_ptr->appends.size() == 1,
           "(c) inner AppendTargetContext missing");
    const auto& call = inner_ptr->appends.front();
    Expect(call.tokens.size() == accepted.size() + 1 &&
               call.tokens.front() == sequence.back() &&
               call.tokens[1] == accepted[0] && call.tokens[2] == accepted[1],
           "(c) appended rows are not {current, accepted...}");
    Expect(call.hidden.size() == 4 && call.hidden[0] == 1.0F &&
               call.hidden[1] == 2.0F && call.hidden[2] == 3.0F &&
               call.hidden[3] == 4.0F,
           "(c) appended hidden rows do not match UpdateTargetHidden");
    Expect(call.hidden_size == 2, "(c) hidden width was not recorded");
    Expect(call.first_token == 777, "(c) correction token was not forwarded");
    Expect(call.position == sequence.size(),
           "(c) appended position is not the copied current position");
    Expect(!backend.ProposalBypassesAdaptiveLength(),
           "(c) the copy flag must clear after feedback");
  }

  // (b2) A copy earns its width: an accepted one widens the next, a rejected
  // one does not.
  {
    LookupDraftBackend backend(nullptr);
    const std::vector<TokenId> first{10, 11, 12, 13, 14, 15,
                                     16, 17, 18, 19, 20};
    ProposeOnce(backend, first, 21);
    std::vector<TokenId> committed = Sequence();
    const auto fresh = backend.Propose(committed, committed.size(), 7);
    Expect(fresh.tokens.size() == 3, "(b2) a fresh match proposes three rows");
    backend.AcceptFeedback(fresh.tokens, 27);
    committed.insert(committed.end(), fresh.tokens.begin(), fresh.tokens.end());
    committed.push_back(27);
    const auto wide = backend.Propose(committed, committed.size(), 7);
    Expect(wide.tokens.size() == 7, "(b2) a copy that held takes every row");
    Expect(wide.tokens[0] == 28, "(b2) the wide copy continues the source");
    backend.AcceptFeedback({}, 500);
    committed.push_back(500);
    const auto after = backend.Propose(committed, committed.size(), 7);
    Expect(after.tokens.empty(), "(b2) a rejected copy ends the run");
  }

  // (d) Without a repeat the inner drafter is used, with its length clamped.
  {
    auto inner = std::make_unique<RecordingDraftBackend>();
    auto* inner_ptr = inner.get();
    LookupDraftBackend backend(std::move(inner));
    backend.SetAdaptiveDraftLimit(3);
    const std::vector<TokenId> tokens{50, 51, 52, 53, 54, 55, 56, 57};
    const auto proposal = backend.Propose(tokens, 8, 7);
    Expect(inner_ptr->propose_calls == 1, "(d) inner drafter was not called");
    Expect(inner_ptr->last_propose_max_tokens == 3,
           "(d) inner proposal length is not clamped by the adaptive limit");
    Expect(proposal.tokens.size() == 1 &&
               proposal.tokens[0] == RecordingDraftBackend::kFixedToken,
           "(d) proposal did not come from the inner drafter");
    Expect(!backend.ProposalBypassesAdaptiveLength(),
           "(d) a neural proposal must not claim a bypassing copy");
    const std::vector<TokenId> accepted{60, 61};
    backend.AcceptFeedback(accepted, 888);
    Expect(inner_ptr->feedback_calls == 1 &&
               inner_ptr->last_feedback_count == 2 &&
               inner_ptr->last_feedback_correction == 888,
           "(d) inner drafter did not receive the feedback");
    Expect(inner_ptr->appends.empty(),
           "(d) a neural round must not append target context");
  }

  // (e) Without an inner drafter a miss is an empty proposal.
  {
    LookupDraftBackend backend(nullptr);
    const std::vector<TokenId> tokens{1, 2, 3, 4, 5, 6, 7, 8};
    const auto proposal = backend.Propose(tokens, 8, 7);
    Expect(proposal.tokens.empty(), "(e) a miss must propose nothing");
    Expect(proposal.start_pos == 8, "(e) empty proposal position");
    Expect(backend.Snapshot() != nullptr, "(e) Snapshot() failed");
    Expect(backend.SnapshotPayloadBytes() == 0,
           "(e) snapshot payload must be empty");
  }

  // (f) Sampled proposals always come from the inner drafter.
  {
    auto inner = std::make_unique<RecordingDraftBackend>();
    auto* inner_ptr = inner.get();
    LookupDraftBackend backend(std::move(inner));
    backend.SetAdaptiveDraftLimit(4);
    Expect(backend.SupportsSampledProposals(),
           "(f) sampled support must follow the inner drafter");
    const std::vector<TokenId> tokens{1, 2, 3, 4, 5, 6, 7, 8};
    std::uint64_t rng = 7;
    const auto proposal = backend.ProposeSampled(tokens, 8, 9, 0.5F, &rng);
    Expect(inner_ptr->sampled_calls == 1,
           "(f) sampled proposal did not reach the inner drafter");
    Expect(inner_ptr->last_sampled_max_tokens == 4,
           "(f) sampled proposal length is not clamped");
    Expect(proposal.tokens.size() == 1 &&
               proposal.tokens[0] == RecordingDraftBackend::kFixedToken,
           "(f) sampled proposal did not come from the inner drafter");
    Expect(!backend.ProposalBypassesAdaptiveLength(),
           "(f) a sampled proposal must not claim a bypassing copy");
  }

  std::cout << "All prompt lookup draft backend tests passed.\n";
  return 0;
}

#else
int main() {
  std::cout << "HIP disabled, skipping prompt lookup draft backend tests.\n";
  return 0;
}
#endif
