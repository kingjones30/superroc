#ifndef SUPERROC_CORE_SAMPLING_HPP_
#define SUPERROC_CORE_SAMPLING_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/json_constraint.hpp"

namespace superroc::sampling {

using TokenId = std::uint32_t;

/// A reasoning token budget, enforced like llama.cpp's reasoning-budget
/// sampler (#266). Tokens are counted after the reasoning start, including
/// reasoning text already in the prompt. Once the budget is spent, and the
/// last token completed its UTF-8 character, `forced` is emitted token by
/// token. Any token in `ends` closes reasoning early. Later tokens are answer
/// content, including literal reasoning tags, as in SuperROC's response
/// framing.
struct ReasoningBudget {
  std::size_t tokens{0};
  std::vector<TokenId> forced;
  std::vector<TokenId> ends;
  std::vector<TokenId> prompt;
  /// Indexed by token: its text ends inside a UTF-8 character.
  std::shared_ptr<const std::vector<bool>> partial_utf8;
};

/// Opt-in exact-repetition detector for the reasoning phase, armed per
/// request by `SUPERROC_REASONING_LOOP_GUARD=<min>:<max>:<repeats>` together
/// with the reasoning end sequence of `BuildReasoningBudget`. Unlike
/// `ReasoningBudget` it never diverts the fast argmax path:
/// `SamplingConfig::can_use_unmodified_argmax()` ignores it, and the sampler
/// reports `ForcedToken()` only after the guard fires. Only tokens generated
/// in this request are scanned, never prompt history.
struct ReasoningLoopGuard {
  std::uint32_t min_period{0};
  std::uint32_t max_period{0};
  std::uint32_t repeats{0};
  std::vector<TokenId> forced;
  std::vector<TokenId> ends;
  /// Indexed by token: its text ends inside a UTF-8 character.
  std::shared_ptr<const std::vector<bool>> partial_utf8;
};

/// Periods of `SUPERROC_REASONING_LOOP_GUARD=<min_period>:<max_period>:
/// <repeats>`, for example `16:512:3`. `SamplingConfig::Validate` requires
/// `1 <= min_period <= max_period`, `repeats >= 2` and
/// `max_period * repeats <= 65536` (the detection window bound).
struct ReasoningLoopGuardPeriods {
  std::uint32_t min_period{0};
  std::uint32_t max_period{0};
  std::uint32_t repeats{0};
};

/// Parses the variable's value. Throws `std::invalid_argument` on any
/// malformed value. Callers treat unset or empty as off before calling.
[[nodiscard]] ReasoningLoopGuardPeriods ParseReasoningLoopGuardPeriods(
    std::string_view text);

/// Opt-in per-word ban for the reasoning phase, armed per request by
/// `SUPERROC_REASONING_BAN=<word>,<word>,...` together with the reasoning end
/// sequence of `BuildReasoningBudget`. `banned` is indexed by vocabulary
/// token: true when the token's decoded text, with leading whitespace
/// (`space`, `tab`, `newline`) removed and trailing `,.!?:;` removed, equals
/// one of the words exactly (case-sensitive). An empty `banned` set is inert.
/// Like the loop guard this never diverts the fast argmax path by itself:
/// `SamplingConfig::can_use_unmodified_argmax()` ignores it, and every
/// selection path keeps its raw argmax unless that token is banned.
struct ReasoningWordBan {
  std::vector<bool> banned;
  std::vector<TokenId> ends;
};

/// Model-independent controls for selecting a token from a logit row.
///
/// Filters run in this order: penalties, temperature, top-k, top-p, min-p.
/// The defaults preserve SuperROC's existing greedy decoding behavior.
struct SamplingConfig {
  float temperature{0.0F};
  /// Observability side-channel for the OpenAI logprobs fields. `logprobs`
  /// arms per-emitted-token reporting (off by default: no extra softmax or
  /// copy on the hot path); `logprobs_top_n` is the number of top tokens
  /// reported per position (0 keeps only the emitted token's logprob). Both
  /// never change which token is drawn: sampling paths ignore them, and the
  /// HTTP layer validates the endpoint-specific bound (N <= 20).
  bool logprobs{false};
  std::uint32_t logprobs_top_n{0};
  std::int32_t top_k{0};
  float top_p{1.0F};
  float min_p{0.0F};
  std::size_t min_keep{0};
  std::int64_t seed{-1};
  float repeat_penalty{1.0F};
  std::size_t repeat_last_n{64};
  float frequency_penalty{0.0F};
  float presence_penalty{0.0F};
  std::shared_ptr<const TokenConstraint> constraint;
  /// Set only for requests that start in their reasoning phase.
  std::shared_ptr<const ReasoningBudget> reasoning_budget;
  /// Set only for requests that start in their reasoning phase when the
  /// operator armed `SUPERROC_REASONING_LOOP_GUARD`. Unlike `reasoning_budget`
  /// this never diverts the fast argmax path: `can_use_unmodified_argmax()`
  /// ignores it, and every sampling path keeps its raw selection until the
  /// sampler reports `ForcedToken()` after the guard fires.
  std::shared_ptr<const ReasoningLoopGuard> reasoning_loop_guard;
  /// Set only for requests that start in their reasoning phase when the
  /// operator armed `SUPERROC_REASONING_BAN`. Like the loop guard this never
  /// diverts the fast argmax path by itself: `can_use_unmodified_argmax()`
  /// ignores it, and only a step whose raw argmax is banned falls back to a
  /// masked host selection. Answer tokens are never banned.
  std::shared_ptr<const ReasoningWordBan> reasoning_word_ban;

  void Validate() const;

  [[nodiscard]] bool penalties_enabled() const noexcept;
  [[nodiscard]] bool uses_random_sampling() const noexcept;
  [[nodiscard]] bool can_use_unmodified_argmax() const noexcept;
};

/// Unique token counts: repetition window and full generated response.
struct TokenPenalty {
  TokenId token{0};
  std::uint32_t generated_count{0};
  std::uint32_t repeated{0};
};

struct Probability {
  TokenId token{0};
  double value{0.0};
};

/// Per-emitted-token logprobs, captured from the target model's distribution.
/// `logprob` is the natural log of the token's probability (negative
/// infinity when the distribution assigns it zero mass). `top` holds the
/// highest-probability entries first, at most the requested N. `text` is
/// filled by the serving layer; the sampler leaves it empty.
struct CapturedTokenLogprobs {
  struct Entry {
    TokenId token{0};
    double logprob{0.0};
    std::string text;
  };
  TokenId token{0};
  double logprob{0.0};
  std::string text;
  std::vector<Entry> top;
};

/// A normalized, post-filter sampling distribution.
class SamplingDistribution {
public:
  SamplingDistribution() = default;
  explicit SamplingDistribution(std::vector<Probability> entries);

  [[nodiscard]] std::span<const Probability> entries() const noexcept;
  [[nodiscard]] TokenId best_token() const;
  [[nodiscard]] double probability(TokenId token) const noexcept;
  [[nodiscard]] TokenId Sample(std::uint64_t* rng_state) const;
  [[nodiscard]] TokenId SampleResidual(
      std::span<const TokenId> candidate_ids,
      std::span<const float> candidate_probabilities,
      std::uint64_t* rng_state) const;

private:
  friend class SamplerState;
  // Entries generated internally have unique IDs; retain their stable order.
  SamplingDistribution(std::vector<Probability> entries, double total);
  std::vector<Probability> entries_;
};

/// The first `top_n` normalized entries of a distribution, highest
/// probability first (ties break toward the lower token ID).
[[nodiscard]] inline std::vector<CapturedTokenLogprobs::Entry>
TopLogprobEntries(const SamplingDistribution& distribution, std::size_t top_n) {
  std::vector<CapturedTokenLogprobs::Entry> ranked;
  const auto entries = distribution.entries();
  ranked.reserve(entries.size());
  for (const auto& entry : entries) {
    if (!(entry.value > 0.0) || !std::isfinite(entry.value))
      continue;
    ranked.push_back({entry.token, std::log(entry.value), {}});
  }
  if (ranked.size() > top_n) {
    std::ranges::partial_sort(
        ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(top_n),
        ranked.end(), [](const auto& left, const auto& right) {
          if (left.logprob == right.logprob)
            return left.token < right.token;
          return left.logprob > right.logprob;
        });
    ranked.resize(top_n);
  } else {
    std::ranges::sort(ranked, [](const auto& left, const auto& right) {
      if (left.logprob == right.logprob)
        return left.token < right.token;
      return left.logprob > right.logprob;
    });
  }
  return ranked;
}

/// Builds the exact distribution from prompt and request-generated tokens.
[[nodiscard]] SamplingDistribution BuildDistribution(
    std::span<const float> logits, const SamplingConfig& config,
    std::span<const TokenId> prompt_tokens = {},
    std::span<const TokenId> generated_tokens = {});

/// A speculative residual draw retained for the next Sample call, with the
/// target model's logprobs report for the token when the deferring step
/// captured one.
struct PendingSample {
  TokenId token{0};
  std::optional<CapturedTokenLogprobs> logprobs;
};

/// Request-owned controls, repetition window, full generated counts and RNG.
///
/// Call Accept only for committed tokens. The type is copyable so speculative
/// decode can work on a tentative state and discard it on rollback.
class SamplerState {
public:
  explicit SamplerState(SamplingConfig config = {},
                        std::span<const TokenId> initial_history = {});
  ~SamplerState() = default;
  SamplerState(const SamplerState& other);
  SamplerState& operator=(const SamplerState& other);
  SamplerState(SamplerState&&) noexcept = default;
  SamplerState& operator=(SamplerState&&) noexcept = default;

  [[nodiscard]] const SamplingConfig& config() const noexcept;
  /// Draft q may remain unconstrained; target p always uses this request's
  /// grammar.
  [[nodiscard]] SamplerState WithoutConstraint() const;
  /// A finite argmax can bypass downloading logits if it satisfies the grammar.
  /// Set penalties_applied only for a winner computed with this state's exact
  /// penalty arithmetic and history; raw model argmaxes leave it false.
  [[nodiscard]] bool CanSelectArgmax(TokenId token,
                                     bool penalties_applied = false) const;
  /// Whether a reasoning word ban is armed and this token is banned: while
  /// reasoning, a banned raw argmax must fall back to a masked host selection
  /// instead of being emitted. False when no ban is armed (unset, empty match
  /// set, or reasoning already ended) and for every answer token. One
  /// bitset lookup on the token when armed, one branch when not.
  [[nodiscard]] bool IsBanned(TokenId token) const noexcept;
  /// Whether the reasoning word ban currently diverts banned raw argmaxes.
  /// True only while reasoning with a non-empty ban armed; installing a
  /// sampler bridge (DeepSeek) on this, rather than on the config, keeps
  /// answer tokens on the raw path.
  [[nodiscard]] bool ReasoningBanActive() const noexcept { return ban_armed_; }
  /// A lazy tool grammar can admit the entire vocabulary before a call.
  /// Retain ordinary CPU/GPU sampling while its mask makes no restriction.
  /// Also true while ForcedToken() is set, so GPU samplers defer to the host.
  [[nodiscard]] bool NeedsConstraintMask() const;
  [[nodiscard]] std::span<const TokenId> history() const noexcept;
  /// The token every sampling path returns next, if any: an exhausted
  /// reasoning budget forces its end sequence.
  [[nodiscard]] std::optional<TokenId> ForcedToken() const noexcept;
  /// Latched when the loop guard started emitting its forced end sequence.
  /// Stays false when the variable is unset, when no loop is found, and when
  /// reasoning ends or the token budget fires first.
  [[nodiscard]] bool LoopGuardFired() const noexcept {
    return loop_guard_fired_;
  }
  /// The repetition period that fired, valid only when `LoopGuardFired()`.
  [[nodiscard]] std::uint32_t LoopGuardPeriod() const noexcept {
    return loop_guard_period_;
  }
  /// The configured repeat count, valid only when `LoopGuardFired()`.
  [[nodiscard]] std::uint32_t LoopGuardRepeats() const noexcept {
    return loop_guard_repeats_;
  }
  /// Generated reasoning tokens accepted when the loop was found, valid only
  /// when `LoopGuardFired()`.
  [[nodiscard]] std::size_t LoopGuardReasoningTokens() const noexcept {
    return loop_guard_reasoning_tokens_;
  }
  [[nodiscard]] std::span<const TokenPenalty> penalties() const noexcept {
    return penalty_counts_;
  }
  [[nodiscard]] std::uint64_t rng_state() const noexcept;
  [[nodiscard]] std::uint64_t* mutable_rng_state() noexcept;
  void SetRngState(std::uint64_t state) noexcept;
  /// Publish RNG and any deferred draw without accepting tentative history.
  void CopyDrawStateFrom(const SamplerState& other) noexcept;
  struct DrawState {
    std::uint64_t rng;
    std::optional<PendingSample> pending;
  };
  [[nodiscard]] DrawState SaveDrawState() const noexcept {
    return {rng_state_, pending_sample_};
  }
  void RestoreDrawState(DrawState state) noexcept {
    rng_state_ = state.rng;
    pending_sample_ = state.pending;
  }

  /// Starts a request: tokens initialize repetition only; generated counts
  /// reset.
  void ResetHistory(std::span<const TokenId> tokens);
  void Accept(TokenId token);
  void Accept(std::span<const TokenId> tokens);

  [[nodiscard]] SamplingDistribution Distribution(
      std::span<const float> logits) const;
  /// Compact logits use token_ids for penalties; returned IDs index logits.
  [[nodiscard]] SamplingDistribution Distribution(
      std::span<const float> logits, std::span<const TokenId> token_ids) const;
  /// The target model's logprobs for an emitted token: its log-probability
  /// plus the top `top_n` entries of the distribution the sampler draws
  /// from (sampling.cpp `SamplerState::Distribution`). Greedy decoding has
  /// no distribution, so it reports a temperature-1 full-vocabulary ranking
  /// over the same penalty-adjusted logits, whose top entry is the greedy
  /// winner. Call only when `logprobs_enabled()`: zero `top_n` keeps an empty
  /// `top` but still fills the token's logprob. Call before Accept: penalties
  /// change the distribution.
  [[nodiscard]] CapturedTokenLogprobs CaptureLogprobs(
      std::span<const float> logits, TokenId token, std::size_t top_n) const;
  /// Per-emitted-token logprobs reporting is armed (off by default).
  [[nodiscard]] bool logprobs_enabled() const noexcept {
    return config_.logprobs;
  }
  /// This request's logprobs depth (top entries per token).
  [[nodiscard]] std::uint32_t logprobs_top_n() const noexcept {
    return config_.logprobs_top_n;
  }
  /// Retain a speculative residual draw for the next Sample call. Copies
  /// preserve this draw along with the RNG; resetting history discards it.
  /// The logprobs overload carries the target model's report for the
  /// deferred token, which the next step emits without recomputing.
  void DeferSample(TokenId token);
  void DeferSample(TokenId token, CapturedTokenLogprobs logprobs);
  /// The retained draw, if any. Read before Sample: consuming Sample resets
  /// it. The payload rides along so an emission can report the distribution
  /// that actually drew a deferred token.
  [[nodiscard]] std::optional<PendingSample> pending_sample() const noexcept {
    return pending_sample_;
  }
  [[nodiscard]] TokenId Sample(std::span<const float> logits);
  [[nodiscard]] TokenId SampleResidual(
      std::span<const float> target_logits,
      std::span<const TokenId> candidate_ids,
      std::span<const float> candidate_probabilities);
  [[nodiscard]] double Uniform();

private:
  [[nodiscard]] std::vector<float> ConstrainedLogits(
      std::span<const float> logits, std::span<const TokenId> ids = {}) const;
  void TrimHistory();
  void RebuildPenaltyCounts();
  [[nodiscard]] double AdjustedLogit(TokenId token, float logit) const noexcept;
  [[nodiscard]] TokenId SampleGreedy(std::span<const float> logits) const;
  [[nodiscard]] TokenId SampleConstrainedGreedy(
      std::span<const float> logits) const;
  [[nodiscard]] SamplingDistribution LinearDistribution(
      std::span<const float> logits) const;
  void PrepareSelected(std::span<const float> logits);
  void StartReasoning();
  void AdvanceReasoning(TokenId token);
  void AdvanceLoopGuard(TokenId token);
  [[nodiscard]] bool LoopGuardCharComplete(TokenId token) const noexcept;
  void AdvanceBan(TokenId token);

  SamplingConfig config_;
  JsonConstraint::State constraint_state_;
  mutable std::shared_ptr<const std::vector<std::uint8_t>> constraint_mask_;
  mutable bool constraint_restricts_{false};
  std::vector<TokenId> history_;
  std::vector<TokenPenalty> penalty_counts_;
  std::vector<Probability> candidate_scratch_;
  std::uint64_t rng_state_{0};
  std::optional<PendingSample> pending_sample_;
  enum class ReasoningPhase : std::uint8_t {
    kOff,
    kCounting,
    kWaitingUtf8,
    kForcing,
  };
  ReasoningPhase reasoning_phase_{ReasoningPhase::kOff};
  std::size_t reasoning_remaining_{0};
  std::size_t reasoning_forced_{0};
  /// Exact-repetition loop guard: while armed, every accepted reasoning token
  /// extends a bounded window of the most recent generated reasoning tokens.
  /// Detection scans periods smallest-first with early exit, capped at
  /// `max_period * repeats` integer compares per token; the window keeps that
  /// many tokens, reserved once when armed, so no allocation happens per
  /// token after the first. A fired guard emits the same forced end sequence
  /// the token budget would, then disarms: answer tokens are never scanned.
  std::vector<TokenId> loop_guard_window_;
  bool loop_guard_armed_{false};
  /// A loop was found but the last token ends inside a UTF-8 character, so
  /// forcing waits for the completing token, exactly as the budget does.
  bool loop_guard_pending_{false};
  bool loop_guard_forcing_{false};
  std::size_t loop_guard_forced_{0};
  std::size_t loop_guard_seen_{0};
  bool loop_guard_fired_{false};
  std::uint32_t loop_guard_period_{0};
  std::uint32_t loop_guard_repeats_{0};
  std::size_t loop_guard_reasoning_tokens_{0};
  /// Reasoning word ban: armed at request start when the config carries a
  /// non-empty ban, disarmed when reasoning ends (an `ends` token, a spent
  /// budget or a fired guard completing). While armed, banned raw argmaxes
  /// divert to masked host selections; every other step keeps its raw path.
  bool ban_armed_{false};
};

[[nodiscard]] std::uint64_t NextRandom(std::uint64_t* state);
[[nodiscard]] double Uniform(std::uint64_t* state);

// Compatibility helpers for callers that have not yet adopted SamplerState.
[[nodiscard]] TokenId SampleLogits(std::span<const float> logits,
                                   float temperature, std::uint64_t* rng_state);
[[nodiscard]] double TokenProbability(std::span<const float> logits,
                                      float temperature, TokenId token);
[[nodiscard]] TokenId SampleResidual(
    std::span<const float> target_logits, float temperature,
    std::span<const TokenId> candidate_ids,
    std::span<const float> candidate_probabilities, std::uint64_t* rng_state);

}  // namespace superroc::sampling

#endif  // SUPERROC_CORE_SAMPLING_HPP_
