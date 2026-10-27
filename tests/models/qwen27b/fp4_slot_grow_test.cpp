// SQ19-2: one FP4 slot set per executor, however the process was configured.
//
// Scenario A (unconfigured): four executors share one KAT model without any
// configure call; each attach grows the table (sets == 4, zero refusals).
// Scenario B (configured): configure(1), one executor (backend 1), then
// configure(4) and four more executors (backend 2, as in the serving test's
// C4 forks); the table holds 5 sets with zero refusals. In both scenarios
// every executor's prefill and decode logits are byte-equal to executor 1 on
// the same prompt. With no flag this binary supervises: it runs each
// scenario in a fresh child process (the slot table is process-global) and
// reports both; --unconfigured / --configured-1-plus-4 run one scenario.
// Needs SUPERROC_FP4_TEST_KAT_MODEL (a ROCmFP4 file); SKIP 77 without it.
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

#if defined(__unix__)
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "src/core/gguf_reader.hpp"
#include "src/core/rocm_fp4/superroc_fp4_gemv_q8.hpp"
#include "src/core/rocm_fp4/superroc_fp4_quant.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/core/env_compat.hpp"

namespace {
using Token = superroc::tokenization::TokenId;
using Executor = superroc::hip::QwenGpuExecutor;
using Model = superroc::hip::QwenGpuModel;

constexpr int kSkipped = 77;
constexpr int kExecutors = 4;
constexpr std::uint32_t kPromptSize = 24;
constexpr std::uint32_t kContinuation = 2;

constexpr std::string_view kText =
    "Virtual memory lets an operating system give each process its own address "
    "space. Pages map virtual addresses to physical memory. A page fault "
    "occurs when the requested page is unavailable and the operating system "
    "must act. Caches hide latency by keeping copies close to where the "
    "processor needs them most. Branch predictors guess which way a branch "
    "goes before it resolves.";

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool ByteEqual(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

std::vector<float> Logits(Executor& executor) {
  const auto view = executor.CopyLastLogits();
  Expect(!view.empty(), "logits must be nonempty");
  return {view.begin(), view.end()};
}

std::vector<Token> PromptTokens(Executor& executor) {
  auto tokens = executor.GetTokenizer().Encode(std::string(kText));
  Expect(tokens.size() >= kPromptSize + kContinuation,
         "fixture text tokenizes too short");
  tokens.resize(kPromptSize + kContinuation);
  return tokens;
}

// Prefill + two decode steps; returns the three logit rows.
std::vector<std::vector<float>> RunPrompt(Executor& executor,
                                          std::span<const Token> tokens) {
  std::vector<std::vector<float>> rows;
  executor.Reset();
  (void)executor.ForwardPromptBatch(tokens.first(kPromptSize));
  rows.push_back(Logits(executor));
  for (std::uint32_t step = 0; step < kContinuation; ++step) {
    (void)executor.ForwardToken(tokens[kPromptSize + step], kPromptSize + step);
    rows.push_back(Logits(executor));
  }
  return rows;
}

void ExpectRowsEqual(const std::vector<std::vector<float>>& got,
                     const std::vector<std::vector<float>>& expected,
                     const std::string& who) {
  Expect(got.size() == expected.size(), who + " row count differs");
  for (std::size_t row = 0; row < expected.size(); ++row) {
    Expect(ByteEqual(got[row], expected[row]),
           who + (row == 0 ? " prefill differs" : " decode differs"));
  }
}

std::shared_ptr<const Model> LoadModel(const char* model) {
  std::string error;
  auto owner = superroc::core::GgufReader::OpenFile(model, &error);
  Expect(owner != nullptr, error);
  auto shared = Model::CreateFromGguf(
      std::shared_ptr<const superroc::core::GgufReader>(std::move(owner)), &error);
  Expect(shared != nullptr, error);
  return shared;
}

int RunUnconfigured(const char* model) {
  // The premise: this process never configures a count, so the table
  // starts at one set with zero refusals.
  Expect(superroc_fp4_slot_sets_count() == 1, "unconfigured table starts at 1");
  Expect(superroc_fp4_slot_refuse_count() == 0, "no refusals before load");
  auto shared = LoadModel(model);

  std::string error;
  auto first = Executor::Create(shared, &error, 128);
  Expect(first != nullptr, error);
  Expect(superroc_fp4_slot_sets_count() == 1, "first executor keeps set 0");
  Expect(superroc_fp4_slot_refuse_count() == 0, "first executor never refused");

  const auto tokens = PromptTokens(*first);
  const auto expected = RunPrompt(*first, tokens);
  std::cout << "executor 1 sets=" << superroc_fp4_slot_sets_count()
            << " refusals=" << superroc_fp4_slot_refuse_count() << '\n';

  // All sessions stay alive together (as in the target test's concurrency
  // check): each construction grows the table by exactly one set.
  std::vector<std::unique_ptr<Executor>> sessions;
  for (int index = 2; index <= kExecutors; ++index) {
    sessions.push_back(
        std::make_unique<Executor>(first->GetSharedModel(), 128));
    Expect(superroc_fp4_slot_sets_count() == index,
           "executor " + std::to_string(index) + " grows the table");
    Expect(superroc_fp4_slot_refuse_count() == 0,
           "executor " + std::to_string(index) + " never refused");
  }
  for (int index = 2; index <= kExecutors; ++index) {
    auto& session = *sessions[static_cast<std::size_t>(index - 2)];
    ExpectRowsEqual(RunPrompt(session, tokens), expected,
                    "executor " + std::to_string(index));
    std::cout << "executor " << index << " sets=" << superroc_fp4_slot_sets_count()
              << " refusals=" << superroc_fp4_slot_refuse_count()
              << " prefill/decode exact=1\n";
  }
  sessions.clear();
  Expect(superroc_fp4_slot_debug_map_size() == 1, "destroy releases sets");
  Expect(superroc_fp4_slot_refuse_count() == 0, "no refusals at shutdown");
  std::cout << "unconfigured: " << kExecutors
            << " executors, one set each, zero refusals, logits exact\n";
  return 0;
}

int RunConfigured1Plus4(const char* model) {
  Expect(superroc_fp4_slot_sets_count() == 1, "configured table starts at 1");
  Expect(superroc_fp4_slot_refuse_count() == 0, "no refusals before load");
  auto shared = LoadModel(model);

  // Backend 1 with one session, as in serve -j 1.
  Expect(superroc_fp4_slot_sets_configure(1) == SUPERROC_FP4_OK,
         "configure(1) never fails");
  Expect(superroc_fp4_slot_sets_count() == 1, "configure(1) pre-sizes to 1");
  std::string error;
  auto backend1 = Executor::Create(shared, &error, 128);
  Expect(backend1 != nullptr, error);
  Expect(superroc_fp4_slot_sets_count() == 1, "backend 1 keeps set 0");
  Expect(superroc_fp4_slot_refuse_count() == 0, "backend 1 never refused");

  const auto tokens = PromptTokens(*backend1);
  const auto expected = RunPrompt(*backend1, tokens);
  std::cout << "backend 1 sets=" << superroc_fp4_slot_sets_count()
            << " refusals=" << superroc_fp4_slot_refuse_count() << '\n';

  // Backend 2 with four sessions while backend 1 is alive, as in the
  // serving test's C4 forks (oracle.load(1) then forks.load(4)). The second
  // configure must not fail and must not renumber backend 1's set.
  Expect(superroc_fp4_slot_sets_configure(4) == SUPERROC_FP4_OK,
         "configure(4) after configure(1) never fails");
  Expect(superroc_fp4_slot_sets_count() == 4, "configure(4) pre-sizes to 4");
  std::vector<std::unique_ptr<Executor>> backend2;
  for (int index = 1; index <= kExecutors; ++index) {
    backend2.push_back(Executor::Create(shared, &error, 128));
    Expect(backend2.back() != nullptr, error);
    Expect(superroc_fp4_slot_refuse_count() == 0,
           "backend 2 session " + std::to_string(index) + " never refused");
  }
  // Sets 1..3 reused, the fourth session grows the table: 5 sets total.
  Expect(superroc_fp4_slot_sets_count() == 1 + kExecutors,
         "configured 1+4 holds 5 sets");
  for (int index = 1; index <= kExecutors; ++index) {
    ExpectRowsEqual(
        RunPrompt(*backend2[static_cast<std::size_t>(index - 1)], tokens),
        expected, "backend 2 session " + std::to_string(index));
    std::cout << "backend 2 session " << index
              << " sets=" << superroc_fp4_slot_sets_count()
              << " refusals=" << superroc_fp4_slot_refuse_count()
              << " prefill/decode exact=1\n";
  }
  backend2.clear();
  backend1.reset();
  Expect(superroc_fp4_slot_debug_map_size() == 0, "destroy releases every set");
  Expect(superroc_fp4_slot_refuse_count() == 0, "no refusals at shutdown");
  std::cout << "configured-1-plus-4: 5 executors, 5 sets, zero refusals, "
               "logits exact\n";
  return 0;
}

}  // namespace

int main(int argc, const char* const* argv) {
#if defined(__unix__)
  try {
    const char* model = nullptr;
    const char* mode = nullptr;
    for (int index = 1; index < argc; ++index) {
      const std::string_view argument = argv[index];
      if (argument == "--unconfigured" || argument == "--configured-1-plus-4") {
        mode = argv[index];
      } else if (model == nullptr) {
        model = argv[index];
      }
    }
    if (model == nullptr) {
      model = ::superroc::core::GetEnvCompat("SUPERROC_FP4_TEST_KAT_MODEL");
    }
    if (model == nullptr) {
      std::cout << "Set SUPERROC_FP4_TEST_KAT_MODEL for fp4_slot_grow_test.\n";
      return kSkipped;
    }
    if (mode != nullptr) {
      if (std::string_view(mode) == "--unconfigured") {
        return RunUnconfigured(model);
      }
      return RunConfigured1Plus4(model);
    }
    // Supervisor: each scenario in a fresh child (the slot table is
    // process-global), waited on by PID.
    int status = 0;
    for (const char* child_mode : {"--unconfigured", "--configured-1-plus-4"}) {
      const pid_t child = fork();
      Expect(child >= 0, "fork for grow scenario");
      if (child == 0) {
        execlp(argv[0], argv[0], model, child_mode,
               static_cast<char*>(nullptr));
        _exit(127);
      }
      int child_status = 0;
      Expect(waitpid(child, &child_status, 0) == child,
             "waitpid for grow scenario");
      if (WIFEXITED(child_status)) {
        status = WEXITSTATUS(child_status);
      } else {
        status = 1;
      }
      std::cout << "scenario " << child_mode << " exit=" << status << '\n';
      if (status != 0) {
        return status;
      }
    }
    std::cout << "fp4_slot_grow_test: both scenarios passed\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "fp4_slot_grow_test failed: " << exception.what() << '\n';
    return 1;
  }
#else
  (void)argc;
  (void)argv;
  std::cout << "fp4_slot_grow_test requires fork/exec (unix).\n";
  return kSkipped;
#endif
}
