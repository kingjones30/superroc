// With SUPERROC_CANONICAL_CACHE=1 a prompt's state must not depend on where
// earlier requests cut it. N prompt tokens are read in one ForwardPromptBatch
// call, then again in two or more calls, once directly and once with a
// snapshot, reset and restore at the last cut; the logits after the last
// prompt token and after each of 16 greedy decode steps must be the same
// bits. The cut plans leave passes of 1 to 257 rows, every size
// up to 17 among them, on both sides of the 128-row point where the default
// build changes its kernel route; a 2,304-token prompt adds passes on both
// sides of 1,024 rows. The test arms the switch itself; with
// SUPERROC_CANONICAL_CACHE=0 in the environment it only reports how many plans
// agree.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/canonical.hpp"
#include "src/core/env.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/testing/compare/logit_comparator.hpp"

namespace {
using Token = superroc::tokenization::TokenId;
using Executor = superroc::hip::QwenGpuExecutor;

constexpr int kSkipped = 77;
constexpr std::size_t kPrompt = 384;
constexpr std::size_t kLongPrompt = 2304;
constexpr std::size_t kDecode = 16;

constexpr std::string_view kText =
    "Virtual memory lets an operating system give each process its own address "
    "space. Pages map virtual addresses to physical memory. A page fault "
    "occurs when the requested page is unavailable and the operating system "
    "must act. Caches hide latency by keeping copies close to where the "
    "processor needs them most. Branch predictors guess which way a branch "
    "goes before it resolves. Pipelines overlap fetch, decode and execute "
    "across many instructions at once. ";

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

struct Reply {
  std::vector<float> prompt_logits;
  std::vector<Token> tokens;
  std::vector<std::uint64_t> logit_hashes;
};

std::uint64_t Hash(std::span<const float> values) {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
  for (std::size_t i = 0; i < values.size_bytes(); ++i)
    hash = (hash ^ bytes[i]) * 1099511628211ULL;
  return hash;
}

// Reads the prompt in passes that end at `cuts`, then decodes kDecode greedy
// tokens. With `restore`, the state at the last cut goes through a snapshot
// and a reset first, as a prompt checkpoint does between two requests.
Reply ReadAndDecode(Executor& executor, std::span<const Token> tokens,
                    std::span<const std::size_t> cuts, bool restore) {
  executor.Reset();
  std::size_t position = 0;
  Token next = 0;
  // A pass holds at most the executor's largest prompt batch, as in the
  // server.
  const std::size_t largest = executor.GetMaxPromptBatch();
  const auto pass = [&](std::size_t end) {
    while (position < end) {
      const std::size_t rows = std::min(end - position, largest);
      next = executor.ForwardPromptBatch(tokens.subspan(position, rows),
                                         static_cast<std::uint32_t>(position));
      position += rows;
    }
  };
  for (std::size_t i = 0; i < cuts.size(); ++i) {
    pass(cuts[i]);
    if (restore && i + 1 == cuts.size()) {
      auto snapshot =
          executor.SaveSnapshot(static_cast<std::uint32_t>(position));
      Expect(snapshot != nullptr, "snapshot at the cut");
      executor.Reset();
      executor.RestoreSnapshot(*snapshot);
    }
  }
  pass(tokens.size());
  Reply reply;
  const auto logits = executor.CopyLastLogits();
  reply.prompt_logits.assign(logits.begin(), logits.end());
  for (std::size_t i = 0; i < kDecode; ++i) {
    reply.tokens.push_back(next);
    next = executor.ForwardToken(next,
                                 static_cast<std::uint32_t>(tokens.size() + i));
    reply.logit_hashes.push_back(Hash(executor.CopyLastLogits()));
  }
  return reply;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const char* model =
        argc > 1 ? argv[1] : ::superroc::core::GetEnv("SUPERROC_VERIFY_MODEL");
    if (model == nullptr) {
      std::cout << "Set SUPERROC_VERIFY_MODEL for prompt_split_test.\n";
      return kSkipped;
    }
    // Read once per process, so arm it before the model loads.
    if (::setenv("SUPERROC_CANONICAL_CACHE", "1", 0) != 0)
      return 2;
    const bool canonical = ::superroc::core::CanonicalCacheOn();
    std::string error;
    auto owner = superroc::core::GgufReader::OpenFile(model, &error);
    Expect(owner != nullptr, error);
    auto executor = Executor::CreateFromGguf(
        std::shared_ptr<const superroc::core::GgufReader>(std::move(owner)),
        &error, 4096);
    Expect(executor != nullptr, error);

    std::string text;
    while (text.size() < 48 * kText.size())
      text += kText;
    const auto all = executor->GetTokenizer().Encode(text);
    Expect(all.size() >= kLongPrompt, "fixture text tokenizes too short");

    // Every last-pass size from 1 to 17 rows, the sizes around 32, 64, 128
    // and 256, short first passes, and chains of short passes.
    std::vector<std::vector<std::size_t>> plans;
    for (std::size_t last = 1; last <= 17; ++last)
      plans.push_back({kPrompt - last});
    for (const std::size_t last :
         {24, 31, 32, 33, 40, 63, 64, 65, 76, 127, 128, 129, 255, 256, 257})
      plans.push_back({kPrompt - last});
    for (const std::size_t first : {1, 2, 3, 8, 15, 16, 17, 127, 128, 129})
      plans.push_back({first});
    plans.push_back({100, 200, 300});
    plans.push_back({129, 130, 383});
    plans.push_back({64, 128, 192, 256, 320});
    plans.push_back({kPrompt - 40, kPrompt - 39, kPrompt - 37, kPrompt - 34,
                     kPrompt - 26, kPrompt - 25, kPrompt - 16, kPrompt - 15,
                     kPrompt - 12, kPrompt - 9, kPrompt - 7});
    // A longer prompt with passes on both sides of 1,024 rows, where the
    // GEMM tile and the attention kernels change.
    const std::vector<std::vector<std::size_t>> long_plans = {
        {kLongPrompt - 1},
        {kLongPrompt - 16},
        {kLongPrompt - 128},
        {kLongPrompt - 1023},
        {kLongPrompt - 1024},
        {kLongPrompt - 1025},
        {1023},
        {1024},
        {1025},
        {1280},
        {512, 1536},
        {1024, 2048},
    };
    std::size_t same = 0;
    std::size_t runs = 0;
    const auto check = [&](std::size_t prompt_tokens,
                           const std::vector<std::vector<std::size_t>>& set) {
      const std::span<const Token> tokens(all.data(), prompt_tokens);
      const Reply whole = ReadAndDecode(*executor, tokens, {}, false);
      for (const auto& plan : set) {
        for (const bool restore : {false, true}) {
          const Reply split = ReadAndDecode(*executor, tokens, plan, restore);
          const bool prompt_equal =
              split.prompt_logits.size() == whole.prompt_logits.size() &&
              std::memcmp(split.prompt_logits.data(),
                          whole.prompt_logits.data(),
                          whole.prompt_logits.size() * sizeof(float)) == 0;
          std::size_t step = 0;
          while (step < kDecode && split.tokens[step] == whole.tokens[step] &&
                 split.logit_hashes[step] == whole.logit_hashes[step])
            ++step;
          const bool equal = prompt_equal && step == kDecode;
          ++runs;
          same += equal ? 1 : 0;
          std::cout << "tokens " << prompt_tokens << " cuts";
          for (const std::size_t cut : plan)
            std::cout << ' ' << cut;
          std::cout << (restore ? " restored" : "");
          if (equal) {
            std::cout << " same bits\n";
          } else if (!prompt_equal) {
            const auto cmp = superroc::testing::CompareLogits(
                whole.prompt_logits, split.prompt_logits);
            std::cout << " prompt differs max_abs=" << cmp.max_abs_diff << "\n";
          } else {
            std::cout << " decode differs at step " << step << "\n";
          }
        }
      }
    };
    check(kPrompt, plans);
    check(kLongPrompt, long_plans);
    std::cout << "prompt_split canonical=" << (canonical ? 1 : 0)
              << " same=" << same << " of " << runs << "\n";
    if (canonical && same != runs)
      throw std::runtime_error("the state depends on the cut points");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
