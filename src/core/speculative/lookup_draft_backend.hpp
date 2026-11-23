#ifndef SUPERROC_CORE_SPECULATIVE_LOOKUP_DRAFT_BACKEND_HPP_
#define SUPERROC_CORE_SPECULATIVE_LOOKUP_DRAFT_BACKEND_HPP_

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/env.hpp"
#include "src/core/speculative/draft_backend.hpp"
#include "src/models/qwen38_flash_next/prompt_lookup.hpp"

namespace superroc::speculative {

/// Copies draft tokens from the session's own committed tokens when the
/// recent output repeats them, and asks the inner drafter (may be null)
/// otherwise. Verification decides every token, so output is unchanged.
class LookupDraftBackend final : public IDraftBackend {
public:
  explicit LookupDraftBackend(std::unique_ptr<IDraftBackend> inner)
      : inner_(std::move(inner)),
        name_(inner_ != nullptr ? "PromptLookup+" + std::string(inner_->Name())
                                : std::string("PromptLookup")) {
    if (const char* e = ::superroc::core::GetEnv("SUPERROC_LOOKUP_DEPTH");
        e && *e)
      depth_ = static_cast<std::uint32_t>(
          std::clamp(std::atoi(e), 1, static_cast<int>(kMaxDepth)));
    if (const char* e = ::superroc::core::GetEnv("SUPERROC_LOOKUP_MIN_MATCH");
        e && *e)
      min_match_ = static_cast<std::size_t>(
          std::clamp(std::strtol(e, nullptr, 10), 3L,
                     static_cast<long>(
                         models::qwen38_flash_next::PromptLookup::kMaxMatch)));
    trace_ = ::superroc::core::GetEnv("SUPERROC_LOOKUP_TRACE") != nullptr;
  }
  ~LookupDraftBackend() override {
    if (trace_)
      std::fprintf(stderr,
                   "[PromptLookup] rounds=%llu proposed=%llu accepted=%llu\n",
                   static_cast<unsigned long long>(rounds_),
                   static_cast<unsigned long long>(proposed_total_),
                   static_cast<unsigned long long>(accepted_total_));
  }

  [[nodiscard]] std::string_view Name() const noexcept override {
    return name_;
  }

  [[nodiscard]] std::uint32_t ExtraDraftCapacity() const noexcept override {
    return depth_;
  }
  void SetAdaptiveDraftLimit(std::uint32_t limit) noexcept override {
    neural_limit_ = limit;
  }
  [[nodiscard]] bool ProposalBypassesAdaptiveLength() const noexcept override {
    return copying_;
  }

  [[nodiscard]] DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override {
    copying_ = false;
    const std::span<const std::int32_t> committed(
        reinterpret_cast<const std::int32_t*>(prompt_tokens.data()),
        prompt_tokens.size());
    if (lookup_.Step(current_pos)) {
      lookup_.Extend(committed);
      const auto match = lookup_.Find(committed, {}, min_match_);
      if (match.length != 0 && match.start < prompt_tokens.size()) {
        // A copy earns its width: a wrong fresh match wastes only a few rows.
        const std::size_t width =
            lookup_.Continues(match.start)
                ? depth_
                : std::min<std::size_t>(depth_,
                                        models::qwen38_flash_next::
                                            PromptLookup::kFreshCopyTokens);
        const auto count = static_cast<std::uint32_t>(std::min<std::size_t>(
            {width, max_tokens, prompt_tokens.size() - match.start}));
        if (count != 0) {
          DraftProposal proposal;
          proposal.start_pos = current_pos;
          proposal.tokens.assign(prompt_tokens.begin() + match.start,
                                 prompt_tokens.begin() + match.start + count);
          copying_ = true;
          copy_start_ = match.start;
          copy_position_ = current_pos;
          copy_count_ = count;
          copy_input_ = prompt_tokens.back();
          copy_hidden_.clear();
          hidden_width_ = 0;
          ++rounds_;
          proposed_total_ += count;
          return proposal;
        }
      }
    }
    if (inner_ != nullptr)
      return inner_->Propose(prompt_tokens, current_pos,
                             std::min(max_tokens, neural_limit_));
    DraftProposal empty;
    empty.start_pos = current_pos;
    return empty;
  }

  [[nodiscard]] DraftProposal ProposeSampled(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
      std::uint64_t* rng_state) override {
    copying_ = false;
    if (inner_ == nullptr)
      throw std::logic_error("prompt lookup has no sampled proposals");
    return inner_->ProposeSampled(prompt_tokens, current_pos,
                                  std::min(max_tokens, neural_limit_),
                                  temperature, rng_state);
  }

  [[nodiscard]] bool SupportsSampledProposals() const noexcept override {
    return inner_ != nullptr && inner_->SupportsSampledProposals();
  }
  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return inner_ != nullptr && inner_->RequiresTargetHiddenStates();
  }
  [[nodiscard]] std::uint32_t TargetHistoryCapacity() const noexcept override {
    return inner_ != nullptr ? inner_->TargetHistoryCapacity() : 0U;
  }
  [[nodiscard]] std::span<const std::uint32_t> TargetHiddenLayerIds()
      const noexcept override {
    return inner_ != nullptr ? inner_->TargetHiddenLayerIds()
                             : std::span<const std::uint32_t>{};
  }
  [[nodiscard]] bool PrimeTargetContext(
      const DraftTargetContext& context) override {
    return inner_ == nullptr || inner_->PrimeTargetContext(context);
  }
  [[nodiscard]] bool AppendTargetContext(const DraftTargetContext& context,
                                         std::uint32_t position) override {
    return inner_ == nullptr || inner_->AppendTargetContext(context, position);
  }
  void UpdateTargetHidden(std::span<const float> hidden) override {
    if (copying_) {
      hidden_width_ = hidden.size();
      copy_hidden_.insert(copy_hidden_.end(), hidden.begin(), hidden.end());
      return;
    }
    if (inner_ != nullptr)
      inner_->UpdateTargetHidden(hidden);
  }
  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override {
    if (!copying_) {
      if (inner_ != nullptr)
        inner_->AcceptFeedback(accepted, correction_token);
      return;
    }
    copying_ = false;
    accepted_total_ += accepted.size();
    lookup_.Observe(copy_start_, copy_count_, accepted.size());
    if (inner_ == nullptr || !inner_->RequiresTargetHiddenStates())
      return;
    // The inner drafter never saw this round: give it the committed rows,
    // exactly as the verifier does for a step with no proposal.
    std::vector<tokenization::TokenId> rows;
    rows.reserve(accepted.size() + 1);
    rows.push_back(copy_input_);
    rows.insert(rows.end(), accepted.begin(), accepted.end());
    if (!inner_->AppendTargetContext({.prompt_tokens = rows,
                                      .prompt_hidden_states = copy_hidden_,
                                      .hidden_size = hidden_width_,
                                      .first_token = correction_token},
                                     copy_position_)) {
      throw std::runtime_error(
          "draft failed to append committed target tokens after a copied "
          "draft");
    }
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes() const override {
    return inner_ != nullptr ? inner_->SnapshotPayloadBytes() : 0U;
  }
  [[nodiscard]] std::optional<std::size_t> SnapshotPayloadBytesAt(
      std::uint32_t position) const override {
    return inner_ != nullptr ? inner_->SnapshotPayloadBytesAt(position)
                             : std::optional<std::size_t>(0U);
  }
  [[nodiscard]] std::unique_ptr<IDraftBackendSnapshot> Snapshot()
      const override {
    if (inner_ != nullptr)
      return inner_->Snapshot();
    return std::make_unique<EmptySnapshot>();
  }
  void RestoreSnapshot(const IDraftBackendSnapshot& snapshot) override {
    lookup_.Clear();
    copying_ = false;
    if (inner_ != nullptr)
      inner_->RestoreSnapshot(snapshot);
  }
  void RestorePersistentSnapshot(
      std::span<const std::uint8_t> payload) override {
    lookup_.Clear();
    copying_ = false;
    if (inner_ != nullptr)
      inner_->RestorePersistentSnapshot(payload);
  }
  void Reset() noexcept override {
    lookup_.Clear();
    copying_ = false;
    if (inner_ != nullptr)
      inner_->Reset();
  }
  void BeginRequest() noexcept override {
    lookup_.Clear();
    copying_ = false;
    if (inner_ != nullptr)
      inner_->BeginRequest();
  }

private:
  static constexpr std::uint32_t kMaxDepth =
      7;  // kMaxDecodeBatch rows per verify
  class EmptySnapshot final : public IDraftBackendSnapshot {
  public:
    [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
      return 0;
    }
    [[nodiscard]] std::size_t PersistentPayloadBytes() const override {
      return 0;
    }
    [[nodiscard]] std::size_t SerializePersistent(
        std::span<std::uint8_t>) const override {
      return 0;
    }
  };

  std::unique_ptr<IDraftBackend> inner_;
  std::string name_;
  models::qwen38_flash_next::PromptLookup lookup_;
  std::uint32_t depth_{kMaxDepth};
  std::size_t min_match_{models::qwen38_flash_next::PromptLookup::kMinMatch};
  std::uint32_t neural_limit_{kMaxDepth};
  bool copying_{false};
  bool trace_{false};
  std::size_t copy_start_{0};
  std::uint32_t copy_position_{0};
  std::uint32_t copy_count_{0};
  tokenization::TokenId copy_input_{0};
  std::vector<float> copy_hidden_;
  std::size_t hidden_width_{0};
  std::uint64_t rounds_{0}, proposed_total_{0}, accepted_total_{0};
};

}  // namespace superroc::speculative

#endif  // SUPERROC_CORE_SPECULATIVE_LOOKUP_DRAFT_BACKEND_HPP_
