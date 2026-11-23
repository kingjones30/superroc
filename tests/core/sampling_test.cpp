#include "src/core/sampling.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json_constraint.hpp"

namespace {

using superroc::sampling::ConstraintVocabulary;
using superroc::sampling::JsonConstraint;
using superroc::sampling::ReasoningBudget;
using superroc::sampling::ReasoningLoopGuard;
using superroc::sampling::ReasoningWordBan;
using superroc::sampling::SamplerState;
using superroc::sampling::SamplingConfig;
using superroc::sampling::TokenConstraint;
using superroc::sampling::TokenId;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

// Greedy logits whose argmax is `winner`; token 3 stands for `</think>`.
std::vector<float> Logits(TokenId winner) {
  std::vector<float> logits(8, 0.0F);
  logits[winner] = 5.0F;
  return logits;
}

std::shared_ptr<const ReasoningBudget> Budget(
    std::size_t tokens, std::vector<TokenId> forced = {3},
    std::vector<TokenId> prompt = {}) {
  auto budget = std::make_shared<ReasoningBudget>();
  budget->tokens = tokens;
  budget->forced = std::move(forced);
  budget->ends = {3};
  budget->prompt = std::move(prompt);
  return budget;
}

void TestReasoningBudgetForcesTheEnd() {
  SamplingConfig config;
  config.reasoning_budget = Budget(2);
  SamplerState sampler(config);
  Expect(!config.can_use_unmodified_argmax(),
         "a budget keeps requests off the unmodified argmax path");
  Expect(!sampler.ForcedToken() && sampler.CanSelectArgmax(1) &&
             !sampler.NeedsConstraintMask(),
         "the budget does not restrict reasoning before it is spent");
  sampler.Accept(sampler.Sample(Logits(1)));
  sampler.Accept(sampler.Sample(Logits(2)));
  Expect(sampler.ForcedToken() == TokenId{3},
         "two reasoning tokens exhaust a budget of two");
  Expect(sampler.NeedsConstraintMask(),
         "GPU sampling paths defer the forced end to the host sampler");
  Expect(!sampler.CanSelectArgmax(1) && sampler.CanSelectArgmax(3),
         "only the reasoning end is a selectable argmax");
  Expect(sampler.Sample(Logits(1)) == 3 &&
             sampler.Distribution(Logits(1)).best_token() == 3,
         "sampling and distributions return the forced end");
  const std::vector<TokenId> ids{5, 3, 1};
  const std::vector<float> compact{1.0F, 0.0F, 2.0F};
  Expect(sampler.Distribution(compact, ids).best_token() == 1,
         "compact distributions index the forced end among their IDs");
  // An MTP draft shortlist may not contain the forced end. Its proposal then
  // follows the shortlist; target verification rejects it and emits the end.
  const std::vector<TokenId> shortlist{5, 4, 1};
  Expect(sampler.Distribution(compact, shortlist).best_token() == 2,
         "a shortlist without the forced end keeps its own distribution");
  auto copy = sampler;
  copy.Accept(3);
  Expect(!copy.ForcedToken() && !copy.NeedsConstraintMask() &&
             sampler.ForcedToken(),
         "copies advance independently, as speculative verification needs");
  Expect(copy.Sample(Logits(6)) == 6, "the answer is sampled normally");
  copy.ResetHistory({});
  Expect(!copy.ForcedToken(), "a restart resets the reasoning budget");
}

void TestEarlyEndAndNoBudget() {
  SamplingConfig config;
  config.reasoning_budget = Budget(4);
  SamplerState sampler(config);
  sampler.Accept(std::vector<TokenId>{1, 3, 1, 1, 1, 1});
  Expect(!sampler.ForcedToken(),
         "a reasoning end within the budget closes the phase for good");
  SamplerState unlimited(SamplingConfig{});
  unlimited.Accept(std::vector<TokenId>(100, 1));
  Expect(!unlimited.ForcedToken(), "requests without a budget are unchanged");
  config.reasoning_budget = Budget(0);
  Expect(SamplerState(config).ForcedToken() == TokenId{3},
         "a zero budget closes reasoning immediately");
}

// Qwen as llama.cpp runs it: the prompt ends with "<think>\n", the forced end
// is "\n</think>" (tokens 4, 3) and token 7 (a tool call) also ends reasoning.
void TestReasoningBudgetMatchesLlamaCpp() {
  SamplingConfig config;
  for (const std::size_t tokens : {1U, 2U, 4U}) {
    config.reasoning_budget = Budget(tokens, {4, 3}, {4});
    SamplerState sampler(config);
    std::vector<TokenId> emitted;
    for (int step = 0; step < 8; ++step) {
      const auto token = sampler.Sample(Logits(1));
      sampler.Accept(token);
      emitted.push_back(token);
    }
    std::vector<TokenId> expected(tokens - 1, 1);
    expected.insert(expected.end(), {4, 3});
    expected.resize(8, 1);
    Expect(emitted == expected,
           "the prompt newline counts and the newline end is forced");
  }
  config.reasoning_budget = Budget(0, {4, 3}, {4});
  SamplerState zero(config);
  Expect(zero.ForcedToken() == TokenId{3},
         "a zero budget's prompt newline stands for the forced newline");
  zero.Accept(3);
  Expect(!zero.ForcedToken(), "the forced end completes reasoning");

  auto ending = std::make_shared<ReasoningBudget>(*Budget(2, {4, 3}));
  ending->ends.push_back(7);
  config.reasoning_budget = ending;
  SamplerState tool(config);
  tool.Accept(7);
  tool.Accept(std::vector<TokenId>(4, 1));
  Expect(!tool.ForcedToken(), "a tool call ends reasoning before the budget");

  // Token 5 ends inside a UTF-8 character and token 6 completes it.
  auto utf8 = std::make_shared<ReasoningBudget>(*Budget(2));
  utf8->partial_utf8 = std::make_shared<std::vector<bool>>(
      std::vector<bool>{false, false, false, false, false, true, false, false});
  config.reasoning_budget = utf8;
  SamplerState split(config);
  split.Accept(std::vector<TokenId>{1, 5, 5});
  Expect(!split.ForcedToken(), "a spent budget waits for a whole character");
  split.Accept(6);
  Expect(split.ForcedToken() == TokenId{3},
         "the end is forced after the character completes");
}

void TestReasoningBudgetKeepsLiteralAnswer() {
  SamplingConfig config;
  config.reasoning_budget = Budget(2, {4, 3}, {4});
  for (const bool natural_end : {false, true}) {
    SamplerState sampler(config);
    if (natural_end) {
      sampler.Accept(3);
    } else {
      sampler.Accept(std::vector<TokenId>{1, 4, 3});
    }
    // Token 0 stands for a literal <think> inside answer/tool data. SuperROC's
    // framing recognizes reasoning markup only in the initial phase.
    sampler.Accept(std::vector<TokenId>{6, 0, 1, 2, 1, 2, 1, 2});
    Expect(!sampler.ForcedToken() && sampler.Sample(Logits(6)) == 6,
           "literal answer tokens must not restart a completed budget");
  }

  const std::vector<TokenId> thinking{1};
  Expect(superroc::sampling::BuildDistribution(Logits(1), config, {}, thinking)
                 .best_token() == 4,
         "distribution construction enforces an exhausted reasoning budget");
  const std::vector<TokenId> answer{1, 4, 3, 6, 0, 1, 2};
  Expect(superroc::sampling::BuildDistribution(Logits(1), config, {}, answer)
                 .best_token() == 1,
         "distribution construction preserves literal answer content");
}

// With a tool or format grammar, the forced end must be a legal grammar step
// and the answer after it stays constrained.
std::shared_ptr<const ReasoningLoopGuard> Guard(
    std::uint32_t min_period, std::uint32_t max_period, std::uint32_t repeats,
    std::vector<TokenId> forced = {3}, std::vector<TokenId> ends = {3}) {
  auto guard = std::make_shared<ReasoningLoopGuard>();
  guard->min_period = min_period;
  guard->max_period = max_period;
  guard->repeats = repeats;
  guard->forced = std::move(forced);
  guard->ends = std::move(ends);
  // No partial_utf8 table: every token completes its character, so a found
  // loop forces immediately.
  return guard;
}

// Twenty distinct tokens: no period below 20 divides the stream, so only a
// period-20 check can match it.
std::vector<TokenId> TwentyBlock() {
  std::vector<TokenId> block;
  for (TokenId token = 10; token < 30; ++token)
    block.push_back(token);
  return block;
}

void TestLoopGuardFiresOnThirdCopy() {
  SamplingConfig config;
  config.reasoning_loop_guard = Guard(16, 512, 3);
  SamplerState sampler(config);
  Expect(!sampler.ForcedToken() && !sampler.LoopGuardFired(),
         "the guard is quiet before any reasoning token");
  const auto block = TwentyBlock();
  for (int copy = 0; copy < 2; ++copy)
    sampler.Accept(block);
  Expect(!sampler.ForcedToken() && !sampler.LoopGuardFired(),
         "two copies are not a loop");
  for (std::size_t i = 0; i + 1 < block.size(); ++i)
    sampler.Accept(block[i]);
  Expect(!sampler.ForcedToken() && !sampler.LoopGuardFired(),
         "an incomplete third copy does not fire");
  sampler.Accept(block.back());
  Expect(sampler.ForcedToken() == TokenId{3}, "the third copy fires the guard");
  Expect(sampler.LoopGuardFired() && sampler.LoopGuardPeriod() == 20 &&
             sampler.LoopGuardRepeats() == 3 &&
             sampler.LoopGuardReasoningTokens() == 60,
         "the firing latches the period, repeats and token count");
}

// A block of `size` distinct tokens repeated three times fires a 16:512:3
// guard exactly on the last token of the third copy, for periods across the
// whole range (an early version checked only the low periods each token).
void TestLoopGuardFiresAtLongPeriods() {
  for (const std::uint32_t size : {41u, 42u, 64u, 300u, 512u}) {
    SamplingConfig config;
    config.reasoning_loop_guard = Guard(16, 512, 3);
    SamplerState sampler(config);
    std::vector<TokenId> block;
    for (TokenId token = 1000; token < 1000 + size; ++token)
      block.push_back(token);
    for (int copy = 0; copy < 2; ++copy)
      sampler.Accept(block);
    for (std::size_t i = 0; i + 1 < block.size(); ++i)
      sampler.Accept(block[i]);
    Expect(!sampler.ForcedToken(),
           "an incomplete third long copy does not fire");
    sampler.Accept(block.back());
    Expect(sampler.ForcedToken() == TokenId{3} &&
               sampler.LoopGuardPeriod() == size,
           "a long period fires on its third copy");
  }
}

void TestLoopGuardCopiesCarryState() {
  SamplingConfig config;
  config.reasoning_loop_guard = Guard(16, 512, 3);
  SamplerState sampler(config);
  const auto block = TwentyBlock();
  for (int copy = 0; copy < 2; ++copy)
    sampler.Accept(block);
  SamplerState twin(sampler);
  SamplerState assigned;
  assigned = sampler;
  for (std::size_t i = 0; i < block.size(); ++i) {
    sampler.Accept(block[i]);
    twin.Accept(block[i]);
    assigned.Accept(block[i]);
    const bool last = i + 1 == block.size();
    Expect(!sampler.ForcedToken() == !last, "the original fires on time");
    Expect((twin.ForcedToken() == TokenId{3}) == last,
           "a copied sampler fires in step");
    Expect((assigned.ForcedToken() == TokenId{3}) == last,
           "an assigned sampler fires in step");
  }
  Expect(sampler.LoopGuardPeriod() == 20 && twin.LoopGuardPeriod() == 20 &&
             assigned.LoopGuardPeriod() == 20,
         "the period latch survives copies");
}

void TestLoopGuardIgnoresCorruptedCopy() {
  SamplingConfig config;
  config.reasoning_loop_guard = Guard(16, 512, 3);
  SamplerState sampler(config);
  const auto block = TwentyBlock();
  for (int copy = 0; copy < 2; ++copy)
    sampler.Accept(block);
  auto corrupted = block;
  corrupted[7] = 100;
  sampler.Accept(corrupted);
  Expect(!sampler.ForcedToken() && !sampler.LoopGuardFired(),
         "one changed token in the third copy is not a loop");
  // A fourth clean copy still holds the corrupted token in every period-20
  // window, so nothing fires.
  sampler.Accept(block);
  Expect(!sampler.ForcedToken() && !sampler.LoopGuardFired(),
         "the corrupted token keeps later windows from matching");
}

void TestLoopGuardHonoursMinPeriod() {
  SamplingConfig config;
  config.reasoning_loop_guard = Guard(16, 512, 3);
  SamplerState sampler(config);
  std::vector<TokenId> block;
  for (TokenId token = 10; token < 20; ++token)
    block.push_back(token);
  for (int copy = 0; copy < 3; ++copy)
    sampler.Accept(block);
  Expect(!sampler.ForcedToken() && !sampler.LoopGuardFired(),
         "a period-10 loop is below the minimum period");
}

void TestLoopGuardForcesEndThenAnswers() {
  SamplingConfig config;
  config.reasoning_loop_guard = Guard(16, 512, 3, {4, 3});
  SamplerState sampler(config);
  const auto block = TwentyBlock();
  for (int copy = 0; copy < 3; ++copy)
    sampler.Accept(block);
  Expect(sampler.ForcedToken() == TokenId{4}, "the forced end comes first");
  Expect(sampler.NeedsConstraintMask(),
         "GPU sampling paths defer the forced end to the host sampler");
  Expect(sampler.CanSelectArgmax(4) && !sampler.CanSelectArgmax(1),
         "only the forced token is a selectable argmax");
  Expect(sampler.Sample(Logits(1)) == 4 &&
             sampler.Distribution(Logits(1)).best_token() == 4,
         "sampling and distributions return the forced end");
  sampler.Accept(4);
  Expect(sampler.ForcedToken() == TokenId{3}, "the end emits in order");
  sampler.Accept(3);
  Expect(!sampler.ForcedToken(), "the sequence ends after the forced tokens");
  Expect(sampler.LoopGuardFired(), "the firing latches for the request log");
  for (int copy = 0; copy < 3; ++copy)
    sampler.Accept(block);
  Expect(!sampler.ForcedToken(),
         "the answer tokens are not scanned, even when they repeat");
}

void TestLoopGuardEndsNaturally() {
  SamplingConfig config;
  config.reasoning_loop_guard = Guard(16, 512, 3);
  SamplerState sampler(config);
  const auto block = TwentyBlock();
  for (int copy = 0; copy < 2; ++copy)
    sampler.Accept(block);
  sampler.Accept(3);
  for (int copy = 0; copy < 3; ++copy)
    sampler.Accept(block);
  Expect(!sampler.ForcedToken() && !sampler.LoopGuardFired(),
         "a reasoning end before any loop stops guarding for good");
}

void TestLoopGuardUnsetIsUnchanged() {
  SamplerState sampler(SamplingConfig{});
  const auto block = TwentyBlock();
  for (int copy = 0; copy < 3; ++copy)
    sampler.Accept(block);
  Expect(!sampler.ForcedToken() && !sampler.LoopGuardFired(),
         "an unset guard never reports a forced token");
  auto copy = sampler;
  copy.Accept(block);
  Expect(!copy.ForcedToken() && !copy.LoopGuardFired(),
         "copies of an unguarded sampler stay unguarded");
}

void TestLoopGuardWithBudget() {
  // The budget fires first: two reasoning tokens spend a budget of two.
  {
    SamplingConfig config;
    config.reasoning_budget = Budget(2);
    config.reasoning_loop_guard = Guard(16, 512, 3);
    SamplerState sampler(config);
    sampler.Accept(1);
    sampler.Accept(2);
    Expect(sampler.ForcedToken() == TokenId{3}, "the budget spends first");
    sampler.Accept(3);
    Expect(!sampler.ForcedToken() && !sampler.LoopGuardFired(),
           "the budget's end disarms the guard");
    const auto block = TwentyBlock();
    for (int copy = 0; copy < 3; ++copy)
      sampler.Accept(block);
    Expect(!sampler.ForcedToken() && !sampler.LoopGuardFired(),
           "a later repeat does not fire after the budget won");
  }
  // The guard fires first: a loop at token 60 of a budget of 1000.
  {
    SamplingConfig config;
    config.reasoning_budget = Budget(1000);
    config.reasoning_loop_guard = Guard(16, 512, 3);
    SamplerState sampler(config);
    const auto block = TwentyBlock();
    for (int copy = 0; copy < 3; ++copy)
      sampler.Accept(block);
    Expect(sampler.ForcedToken() == TokenId{3}, "the guard fires first");
    Expect(sampler.LoopGuardFired(), "the guard's firing latches");
    sampler.Accept(3);
    Expect(!sampler.ForcedToken(),
           "the forced end completes and the budget stays down");
    for (int round = 0; round < 100; ++round)
      sampler.Accept(std::vector<TokenId>(10, 6));
    Expect(!sampler.ForcedToken(),
           "a lingering budget cannot spend mid-answer after the guard won");
  }
}

void TestLoopGuardPeriodParsing() {
  const auto parsed =
      superroc::sampling::ParseReasoningLoopGuardPeriods("16:512:3");
  Expect(parsed.min_period == 16 && parsed.max_period == 512 &&
             parsed.repeats == 3,
         "a well-formed value parses");
  for (const std::string_view bad :
       {"16:512", "16:512:3:4", "16:512:1", "0:512:3", "512:16:3", "16::3",
        ":16:3", "16:512:3:", "16:512:x", "16:512:100000", "-1:512:3"}) {
    bool threw = false;
    try {
      (void)superroc::sampling::ParseReasoningLoopGuardPeriods(bad);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    Expect(threw,
           std::string("a malformed value is rejected: ") + std::string(bad));
  }
}

// A ban over the 8-token test vocabulary: `banned[i]` marks token `i`.
// Ends default to token 3 (`</think>` in the other tests).
std::shared_ptr<const ReasoningWordBan> Ban(std::vector<bool> banned,
                                            std::vector<TokenId> ends = {3}) {
  auto ban = std::make_shared<ReasoningWordBan>();
  ban->banned = std::move(banned);
  ban->ends = std::move(ends);
  return ban;
}

// Eight-token row whose argmax is 5, then 7, then 6.
std::vector<float> BannedRow() {
  return {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 9.0F, 6.0F, 7.0F};
}

void TestReasoningCloseMessageWithBudget() {
  // Forced end `{4, 3}` with a one-token closing message `7`: the message
  // emits exactly once, then the end, then the model answers.
  SamplingConfig config;
  config.reasoning_budget = Budget(2, {7, 4, 3});
  SamplerState sampler(config);
  Expect(sampler.Sample(Logits(1)) == 1, "reasoning still selects freely");
  sampler.Accept(1);
  Expect(sampler.Sample(Logits(2)) == 2, "the budget is not spent yet");
  sampler.Accept(2);
  Expect(sampler.ForcedToken() == TokenId{7}, "the message comes first");
  Expect(!sampler.CanSelectArgmax(1) && sampler.CanSelectArgmax(7),
         "only the message token is selectable while forcing");
  std::vector<TokenId> emitted;
  for (int step = 0; step < 3; ++step) {
    emitted.push_back(sampler.Sample(Logits(1)));
    sampler.Accept(emitted.back());
  }
  Expect(emitted == std::vector<TokenId>{7, 4, 3},
         "message then forced end, each exactly once");
  Expect(!sampler.ForcedToken() && sampler.Sample(Logits(6)) == 6,
         "the closed budget answers normally");
}

void TestReasoningCloseMessageWithGuard() {
  SamplingConfig config;
  config.reasoning_loop_guard = Guard(16, 512, 3, {7, 4, 3});
  SamplerState sampler(config);
  const auto block = TwentyBlock();
  for (int copy = 0; copy < 3; ++copy)
    sampler.Accept(block);
  Expect(sampler.LoopGuardFired(), "the loop fires the guard");
  Expect(sampler.Sample(Logits(1)) == 7, "the message precedes the end");
  sampler.Accept(7);
  Expect(sampler.Sample(Logits(1)) == 4, "the forced end follows in order");
  sampler.Accept(4);
  Expect(sampler.Sample(Logits(1)) == 3, "the end completes the sequence");
  sampler.Accept(3);
  Expect(!sampler.ForcedToken() && sampler.Sample(Logits(1)) == 1,
         "answers follow the message and the end");
}

void TestReasoningWordBanGreedy() {
  SamplingConfig config;
  config.reasoning_word_ban =
      Ban({false, false, false, false, false, true, false, false});
  SamplerState sampler(config);
  Expect(sampler.ReasoningBanActive(), "a matching ban arms the request");
  Expect(!sampler.CanSelectArgmax(5) && sampler.CanSelectArgmax(7),
         "a banned raw argmax is rejected while reasoning");
  Expect(config.can_use_unmodified_argmax(),
         "the ban keeps the raw-argmax fast path armed");
  Expect(sampler.Sample(BannedRow()) == 7,
         "greedy picks the best non-banned token");
  Expect(sampler.Distribution(BannedRow()).best_token() == 7 &&
             sampler.Distribution(BannedRow()).probability(5) == 0.0,
         "distributions exclude the banned token");
  const std::vector<TokenId> ids{5, 7, 6};
  const std::vector<float> compact{9.0F, 7.0F, 6.0F};
  Expect(sampler.Distribution(compact, ids).best_token() == 1,
         "a shortlist with the banned argmax resolves to the next best");
}

void TestReasoningWordBanSampled() {
  SamplingConfig config;
  config.temperature = 1.0F;
  config.reasoning_word_ban =
      Ban({false, false, false, false, false, true, false, false});
  SamplerState sampler(config);
  const auto distribution = sampler.Distribution(BannedRow());
  Expect(distribution.probability(5) == 0.0,
         "a sampled request renormalises without the banned token");
  double total = 0.0;
  for (const auto& entry : distribution.entries())
    total += entry.value;
  Expect(total > 0.999 && total < 1.001,
         "the renormalised distribution still sums to one");
}

void TestReasoningWordBanEndsAfterReasoning() {
  SamplingConfig config;
  config.reasoning_word_ban =
      Ban({false, false, false, false, false, true, false, false});
  SamplerState sampler(config);
  sampler.Accept(std::vector<TokenId>{1, 2});
  Expect(sampler.Sample(BannedRow()) == 7, "the ban applies while reasoning");
  sampler.Accept(3);
  Expect(!sampler.ReasoningBanActive(), "the ends token disarms the ban");
  Expect(sampler.Sample(BannedRow()) == 5 && sampler.CanSelectArgmax(5) &&
             sampler.Distribution(BannedRow()).best_token() == 5,
         "answer tokens are not affected");
}

void TestReasoningWordBanEmptyMatchSet() {
  SamplingConfig config;
  config.reasoning_word_ban = Ban({});
  SamplerState sampler(config);
  Expect(!sampler.ReasoningBanActive(),
         "an empty match set arms nothing at all");
  Expect(sampler.CanSelectArgmax(5) && sampler.Sample(Logits(5)) == 5,
         "an empty match set selects exactly like no ban");
  SamplingConfig all_clear;
  all_clear.reasoning_word_ban =
      Ban({false, false, false, false, false, false, false, false});
  SamplerState quiet(all_clear);
  Expect(!quiet.ReasoningBanActive() && quiet.Sample(Logits(5)) == 5,
         "an all-false set is inert too");
}

void TestReasoningWordBanDraftRejected() {
  // Models the verifier's row decision (select_row, the sequential replay and
  // the non-sampled correction): a drafted token equal to a banned target
  // argmax is rejected through CanSelectArgmax and corrected with the masked
  // argmax of that row.
  SamplingConfig config;
  config.reasoning_word_ban =
      Ban({false, false, false, false, false, true, false, false});
  SamplerState sampler(config);
  const TokenId draft = 5;
  const auto row = BannedRow();
  TokenId raw = 5;
  for (std::size_t i = 0; i < row.size(); ++i)
    if (row[i] > row[raw])
      raw = static_cast<TokenId>(i);
  Expect(raw == draft, "the draft equals the target raw argmax");
  Expect(!sampler.CanSelectArgmax(raw),
         "the banned target argmax is rejected as a draft");
  const TokenId correction = sampler.Sample(row);
  Expect(correction == 7 && correction != draft,
         "the correction is the masked argmax of the same row");
}

void TestReasoningBudgetGuardMessageBanCombined() {
  // Budget 1000, guard 16:512:3, one-token message, token 6 banned. The loop
  // uses unbanned tokens so the guard fires first; forcing emits the message
  // then the end exactly once; afterwards the ban is off for answers.
  SamplingConfig config;
  config.reasoning_budget = Budget(1000, {7, 4, 3});
  config.reasoning_loop_guard = Guard(16, 512, 3, {7, 4, 3});
  config.reasoning_word_ban =
      Ban({false, false, false, false, false, false, true, false});
  SamplerState sampler(config);
  Expect(sampler.ReasoningBanActive(), "the ban arms beside budget and guard");
  Expect(sampler.Sample(BannedRow()) == 5,
         "token 6 is banned, so greedy takes token 5 while counting");
  const auto block = TwentyBlock();
  for (int copy = 0; copy < 3; ++copy)
    sampler.Accept(block);
  Expect(sampler.LoopGuardFired(), "the guard wins the race at 60 tokens");
  Expect(sampler.ForcedToken() == TokenId{7}, "the message emits first");
  // Row whose argmax (token 6) is banned: forcing still emits message/end.
  const std::vector<float> banned_winner{1.0F, 2.0F, 3.0F, 4.0F,
                                         5.0F, 6.0F, 9.0F, 7.0F};
  Expect(sampler.Sample(banned_winner) == 7,
         "forcing bypasses the ban even off a banned row");
  std::vector<TokenId> emitted;
  for (int step = 0; step < 3; ++step) {
    emitted.push_back(sampler.Sample(banned_winner));
    sampler.Accept(emitted.back());
  }
  Expect(emitted == std::vector<TokenId>{7, 4, 3},
         "forcing emits message then end, each exactly once");
  Expect(!sampler.ForcedToken() && !sampler.ReasoningBanActive(),
         "the completed forcing closes reasoning and disarms the ban");
  Expect(sampler.Sample(BannedRow()) == 5,
         "the answer ignores the ban, and no message repeats");
}

void TestReasoningBudgetWithGrammar() {
  const std::vector<std::string> pieces{
      "thinking", "</think>", "{\"x\":", "true}", "prose", ""};
  auto binding = std::make_shared<TokenConstraint>();
  binding->grammar = JsonConstraint::WithReasoning(JsonConstraint::Object());
  binding->vocabulary = std::make_shared<ConstraintVocabulary>(
      pieces.size(), [&](std::uint32_t i) {
        return ConstraintVocabulary::Piece{pieces[i], i == 5};
      });
  SamplingConfig config;
  config.constraint = binding;
  auto budget = std::make_shared<ReasoningBudget>();
  budget->tokens = 1;
  budget->forced = {1};
  budget->ends = {1};
  config.reasoning_budget = budget;
  SamplerState sampler(config);
  sampler.Accept(sampler.Sample(std::vector<float>{5, 0, 0, 0, 0, 0}));
  const auto forced = sampler.Sample(std::vector<float>{5, 0, 0, 0, 4, 0});
  Expect(forced == 1, "the grammar request also closes at its budget");
  sampler.Accept(forced);
  Expect(!sampler.ForcedToken() &&
             sampler.Sample(std::vector<float>{5, 0, 1, 0, 4, 0}) == 2,
         "after the forced end the answer still follows the grammar");
}

}  // namespace

int main() {
  TestReasoningBudgetForcesTheEnd();
  TestEarlyEndAndNoBudget();
  TestReasoningBudgetMatchesLlamaCpp();
  TestReasoningBudgetKeepsLiteralAnswer();
  TestReasoningBudgetWithGrammar();
  TestLoopGuardFiresOnThirdCopy();
  TestLoopGuardFiresAtLongPeriods();
  TestLoopGuardCopiesCarryState();
  TestLoopGuardIgnoresCorruptedCopy();
  TestLoopGuardHonoursMinPeriod();
  TestLoopGuardForcesEndThenAnswers();
  TestLoopGuardEndsNaturally();
  TestLoopGuardUnsetIsUnchanged();
  TestLoopGuardWithBudget();
  TestLoopGuardPeriodParsing();
  TestReasoningCloseMessageWithBudget();
  TestReasoningCloseMessageWithGuard();
  TestReasoningWordBanGreedy();
  TestReasoningWordBanSampled();
  TestReasoningWordBanEndsAfterReasoning();
  TestReasoningWordBanEmptyMatchSet();
  TestReasoningWordBanDraftRejected();
  TestReasoningBudgetGuardMessageBanCombined();
  std::cout << "sampling_test passed\n";
  return 0;
}
