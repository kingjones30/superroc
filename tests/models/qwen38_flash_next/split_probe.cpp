// Prompt split probe: does the state after N prompt tokens depend on where
// earlier Sync calls cut them? For each cut plan it prints whether the logits
// and the session payload are the same bits as after one Sync of the whole
// prompt. A diagnostic: it reports and exits 0. With `require` as the last
// argument it fails when a plan whose cuts fall on multiples of 4 rows and
// whose passes hold 16 rows or more leaves another state: the rule the
// canonical cache relies on for ROCmFP4 files. It also freezes
// in-pass views: the pair P/S plan (P the last multiple of 64 rows with 16+
// rows after it, S inside the last 8 rows) and the boundary plan (P alone,
// no S view) compare the checkpoint frozen at P with a pass ending at P and
// their final state with a two-pass run cut at P. `require` fails when the
// boundary plan differs — a view at P is the canonical scheme's own
// checkpoint. The pair and tail captures report only: they freeze extra
// views beside P, and the default build already knows an in-pass capture
// may change the remaining rows' bits.
//   split_probe --model FIRST.gguf --mtp-model MTP.gguf [TOKENS] [require]
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = superroc::models::qwen38_flash_next;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct State {
  std::vector<float> logits;
  std::vector<std::uint8_t> payload;
};

State Finish(qfn::Session& session) {
  std::string error;
  State state;
  state.logits.assign(session.Logits().begin(), session.Logits().end());
  auto snapshot = session.SaveSnapshot(&error);
  Require(snapshot != nullptr, error);
  state.payload.resize(snapshot->SizeBytes());
  Require(snapshot->CopyTo(state.payload), "materialize snapshot");
  return state;
}

bool SameLogits(const State& left, const State& right) {
  return left.logits.size() == right.logits.size() &&
         std::memcmp(left.logits.data(), right.logits.data(),
                     left.logits.size() * sizeof(float)) == 0;
}

// A pass that ends at `tokens`: a checkpoint equal to an in-pass view there.
State ReadPrefix(qfn::Model& model, superroc::core::SessionMode mode,
                 std::span<const std::int32_t> tokens) {
  std::string error;
  auto session = model.CreateSession(mode, 8192, &error);
  Require(session != nullptr, error);
  Require(session->Sync(tokens, &error), error);
  return Finish(*session);
}

State Read(qfn::Model& model, superroc::core::SessionMode mode,
           std::span<const std::int32_t> prompt,
           std::span<const std::size_t> cuts) {
  std::string error;
  auto session = model.CreateSession(mode, 8192, &error);
  Require(session != nullptr, error);
  for (const std::size_t cut : cuts)
    Require(session->Sync(prompt.first(cut), &error), error);
  Require(session->Sync(prompt, &error), error);
  return Finish(*session);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5 || std::string_view(argv[1]) != "--model" ||
      std::string_view(argv[3]) != "--mtp-model") {
    std::cerr << "Usage: split_probe --model FIRST.gguf --mtp-model MTP.gguf "
                 "[TOKENS] [require]\n";
    return 77;
  }
  try {
    std::string error;
    const bool require =
        argc > 5 && std::string_view(argv[argc - 1]) == "require";
    const std::size_t n =
        argc > 5 && std::string_view(argv[5]) != "require"
            ? static_cast<std::size_t>(std::strtoul(argv[5], nullptr, 10))
            : 900;
    Require(n >= 512 && n <= 8000, "TOKENS must be 512 to 8000");
    auto model = qfn::Model::Load(
        argv[2],
        {.max_context = 8192, .mtp_model_path = argv[4], .max_draft_tokens = 7},
        &error);
    Require(model != nullptr, error);
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. "
        "Strix Halo executes this deterministic benchmark sequence. ");
    Require(!pattern.empty(), "empty prompt pattern");
    std::vector<std::int32_t> prompt(n);
    for (std::size_t i = 0; i < n; ++i)
      prompt[i] = pattern[i % pattern.size()];
    const auto mode = model->HasMtp()
                          ? superroc::core::SessionMode::kSpeculative
                          : superroc::core::SessionMode::kAutoregressive;
    const State whole = Read(*model, mode, prompt, {});
    std::vector<std::vector<std::size_t>> plans;
    for (const std::size_t last :
         {1,  2,  4,  7,  8,  9,   12,  15,  16,  17,  20,  24,  28,  31, 32,
          33, 48, 63, 64, 65, 100, 127, 128, 129, 200, 255, 256, 257, 400})
      plans.push_back({n - last});
    for (const std::size_t first : {1, 8, 15, 16, 17, 20, 127, 128, 129})
      plans.push_back({first});
    plans.push_back({n / 3, 2 * n / 3});
    plans.push_back({n / 2, n - 16});
    plans.push_back({n - 16, n - 8});
    plans.push_back({n - 64, n - 32, n - 16});
    // Cuts rounded down to a multiple of 4 and of 64 that leave 16 rows or
    // more, wherever the prompt itself ends.
    for (const std::size_t back : {16, 17, 19, 31, 63, 127, 255})
      plans.push_back({(n - back) / 4 * 4});
    plans.push_back({(n - 64) / 64 * 64});
    plans.push_back({(n / 2) / 4 * 4 + 1});
    plans.push_back({(n / 2) / 4 * 4 + 2});
    plans.push_back({(n / 2) / 4 * 4, (n - 100) / 4 * 4});
    std::size_t same_logits = 0;
    std::size_t same_payload = 0;
    std::size_t rule_plans = 0;
    std::size_t rule_same = 0;
    std::size_t other_same = 0;
    for (const auto& plan : plans) {
      const State split = Read(*model, mode, prompt, plan);
      const bool logits = split.logits.size() == whole.logits.size() &&
                          std::memcmp(split.logits.data(), whole.logits.data(),
                                      whole.logits.size() * sizeof(float)) == 0;
      const bool payload = split.payload == whole.payload;
      same_logits += logits ? 1 : 0;
      same_payload += payload ? 1 : 0;
      std::cout << "tokens=" << n << " cuts";
      for (const std::size_t cut : plan)
        std::cout << ' ' << cut;
      bool aligned = true;
      std::size_t min_pass = n;
      std::size_t previous = 0;
      for (const std::size_t cut : plan) {
        aligned = aligned && cut % 4 == 0;
        min_pass = std::min(min_pass, cut - previous);
        previous = cut;
      }
      min_pass = std::min(min_pass, n - previous);
      const bool rule = aligned && min_pass >= 16;
      if (rule) {
        ++rule_plans;
        rule_same += logits && payload ? 1 : 0;
      } else {
        other_same += logits && payload ? 1 : 0;
      }
      std::cout << " min_pass=" << min_pass << " cuts_x4=" << (aligned ? 1 : 0)
                << " logits=" << (logits ? "same" : "DIFFER")
                << " payload=" << (payload ? "same" : "DIFFER") << "\n";
    }
    std::cout << "split_probe tokens=" << n << " plans=" << plans.size()
              << " same_logits=" << same_logits
              << " same_payload=" << same_payload
              << " | cuts on multiples of 4 with passes of 16+ rows: "
              << rule_same << " of " << rule_plans
              << " same; other plans: " << other_same << " of "
              << plans.size() - rule_plans << " same\n";
    // In-pass frozen views: the boundary checkpoint is captured inside one
    // pass instead of ending a pass there. P is the canonical planned cut,
    // the last multiple of 64 rows with 16+ rows after it; (a) compares the
    // in-pass view frozen at P with a pass ending at P, (b) compares the
    // in-pass run's final logits and payload with a plain two-pass run cut
    // at P. P is frozen through the widened pair tail, so it shares its
    // pass with S; a capture at S applies only inside the last 8 rows.
    const std::size_t restore = (n - 16) / 64 * 64;
    const std::size_t s = n - 8;
    bool pair_view_same = false;
    bool alone_view_same = false;
    bool alone_final_same = false;
    bool alone_ok = false;
    if (restore >= 64 && restore < s) {
      const std::span<const std::int32_t> tokens(prompt);
      const State prefix = ReadPrefix(*model, mode, tokens.first(restore));
      const State cut =
          Read(*model, mode, prompt, std::vector<std::size_t>{restore});
      std::string error;
      auto session = model->CreateSession(mode, 8192, &error);
      Require(session != nullptr, error);
      std::unique_ptr<qfn::SessionSnapshot> view_first;
      std::unique_ptr<qfn::SessionSnapshot> view;
      if (session->SyncThroughPair(prompt, static_cast<std::uint32_t>(restore),
                                   static_cast<std::uint32_t>(s), &view_first,
                                   &view, &error)) {
        Require(view_first != nullptr && view != nullptr, error);
        std::vector<std::uint8_t> frozen(view_first->SizeBytes());
        Require(view_first->CopyTo(frozen), "materialize in-pass view");
        const State done = Finish(*session);
        pair_view_same = frozen == prefix.payload;
        std::cout << "in-pass tokens=" << n << " P=" << restore << " S=" << s
                  << " plan=pair"
                  << " P_view_payload=" << (pair_view_same ? "same" : "DIFFER")
                  << " final_logits="
                  << (SameLogits(done, cut) ? "same" : "DIFFER")
                  << " final_payload="
                  << (done.payload == cut.payload ? "same" : "DIFFER") << "\n";
      } else {
        std::cout << "in-pass tokens=" << n << " P=" << restore << " S=" << s
                  << " plan=pair refused: " << error << "\n";
      }
      // The pair path with only P, no S view at all: what a canonical
      // in-pass boundary capture would freeze.
      auto alone = model->CreateSession(mode, 8192, &error);
      Require(alone != nullptr, error);
      std::unique_ptr<qfn::SessionSnapshot> alone_view;
      if (alone->SyncThroughBoundary(prompt,
                                     static_cast<std::uint32_t>(restore),
                                     &alone_view, &error)) {
        Require(alone_view != nullptr, error);
        std::vector<std::uint8_t> frozen_alone(alone_view->SizeBytes());
        Require(alone_view->CopyTo(frozen_alone), "materialize in-pass view");
        const State alone_done = Finish(*alone);
        alone_ok = true;
        alone_view_same = frozen_alone == prefix.payload;
        alone_final_same =
            SameLogits(alone_done, cut) && alone_done.payload == cut.payload;
        std::cout << "in-pass tokens=" << n << " P=" << restore
                  << " plan=boundary"
                  << " P_view_payload=" << (alone_view_same ? "same" : "DIFFER")
                  << " final_logits="
                  << (SameLogits(alone_done, cut) ? "same" : "DIFFER")
                  << " final_payload="
                  << (alone_done.payload == cut.payload ? "same" : "DIFFER")
                  << "\n";
      } else {
        std::cout << "in-pass tokens=" << n << " P=" << restore
                  << " plan=boundary refused: " << error << "\n";
      }
      // A single in-pass capture at the text-tail boundary S.
      auto tail = model->CreateSession(mode, 8192, &error);
      Require(tail != nullptr, error);
      std::unique_ptr<qfn::SessionSnapshot> tail_view;
      if (tail->SyncThrough(prompt, static_cast<std::uint32_t>(s), &tail_view,
                            &error)) {
        Require(tail_view != nullptr, error);
        const State tail_prefix = ReadPrefix(*model, mode, tokens.first(s));
        const State tail_cut =
            Read(*model, mode, prompt, std::vector<std::size_t>{s});
        std::vector<std::uint8_t> frozen(tail_view->SizeBytes());
        Require(tail_view->CopyTo(frozen), "materialize in-pass view");
        const State tail_done = Finish(*tail);
        std::cout << "in-pass tokens=" << n << " S=" << s
                  << " plan=tail_capture"
                  << " S_view_payload="
                  << (frozen == tail_prefix.payload ? "same" : "DIFFER")
                  << " final_logits="
                  << (SameLogits(tail_done, tail_cut) ? "same" : "DIFFER")
                  << " final_payload="
                  << (tail_done.payload == tail_cut.payload ? "same" : "DIFFER")
                  << "\n";
      } else {
        std::cout << "in-pass tokens=" << n << " S=" << s
                  << " plan=tail_capture refused: " << error << "\n";
      }
    }
    if (require && (rule_same != rule_plans || !alone_ok || !alone_view_same ||
                    !alone_final_same))
      throw std::runtime_error("the state depends on allowed cut points");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
