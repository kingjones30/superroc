// LFM2 decode e2e (SPEC2, lane LFD). A-LEAN weights, 4 fork prompts.
// For each prompt: ForwardPromptBatch chunk -> first token, then 47 greedy
// ForwardToken steps -> 48 ids. Compares the leading run against the fork's
// greedy 48 ids on the same A file.
// usage: lfm2_decode_test [gguf]  (or LFM2_A_GGUF under ctest)
// Exit 0 pass, 1 fail, 77 skipped (no file named).
// NOTE (deviation): SPEC2 step 6 names ForwardPromptChunk, which is a private
// member; the test uses the public ForwardPromptBatch, which forwards to
// ForwardPromptChunk in one chunk for these prompt lengths.

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {

// LFM2: one line is a JSON list like [1, 2, 3].
std::vector<gufo::tokenization::TokenId> ParseIdList(const std::string& line,
                                                     const char* what) {
  const auto open = line.find('[');
  const auto close = line.find(']');
  if (open == std::string::npos || close == std::string::npos ||
      close <= open) {
    throw std::runtime_error(std::string("bad id list in ") + what);
  }
  std::vector<gufo::tokenization::TokenId> ids;
  std::string cell;
  std::istringstream cells(line.substr(open + 1, close - open - 1));
  while (std::getline(cells, cell, ',')) {
    std::istringstream num(cell);
    unsigned long value = 0;
    num >> value;
    if (num.fail()) {
      throw std::runtime_error(std::string("bad id cell in ") + what);
    }
    ids.push_back(static_cast<gufo::tokenization::TokenId>(value));
  }
  return ids;
}

std::vector<std::vector<gufo::tokenization::TokenId>> ReadIdLines(
    const std::string& path, const char* what, std::size_t expect_lines) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error(std::string("cannot open ") + path);
  }
  std::vector<std::vector<gufo::tokenization::TokenId>> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    rows.push_back(ParseIdList(line, what));
  }
  if (rows.size() != expect_lines) {
    throw std::runtime_error(std::string("expected ") +
                             std::to_string(expect_lines) + " lines in " +
                             what);
  }
  return rows;
}

// LFM2: greedy 48 from a prompt: chunk first token + 47 ForwardToken steps.
std::vector<gufo::tokenization::TokenId> Generate48(
    gufo::hip::QwenGpuExecutor* executor,
    const std::vector<gufo::tokenization::TokenId>& prompt) {
  executor->Reset();
  std::vector<gufo::tokenization::TokenId> ids;
  ids.reserve(48);
  ids.push_back(executor->ForwardPromptBatch(prompt, 0, true));
  for (std::size_t s = 1; s < 48; ++s) {
    const std::uint32_t pos = static_cast<std::uint32_t>(prompt.size() + s - 1);
    ids.push_back(executor->ForwardToken(ids.back(), pos, true));
  }
  return ids;
}

}  // namespace

int main(int argc, char** argv) {
  const char* configured = argc >= 2 ? argv[1] : std::getenv("LFM2_A_GGUF");
  if (configured == nullptr) {
    std::cerr << "SKIP lfm2_decode_test: LFM2_A_GGUF is not set\n";
    return 77;
  }
  const std::string path = configured;
  try {
    std::string err;
    auto opened = gufo::core::GgufReader::OpenFile(path, &err);
    if (opened == nullptr) {
      std::cerr << "SKIP lfm2_decode_test: " << path << " unreadable: " << err
                << '\n';
      return 77;
    }
    std::shared_ptr<const gufo::core::GgufReader> reader(std::move(opened));
    gufo::hip::QwenExecutionPolicy policy;  // LFM2 (lfm2_ppl policy)
    policy.kv_cache_storage = gufo::hip::QwenKvCacheStorage::kFp32;  // LFM2
    policy.fuse_qk_norm_rope_kv = false;                             // LFM2
    constexpr std::uint32_t kContext = 4096;                         // LFM2
    auto executor = gufo::hip::QwenGpuExecutor::CreateFromGguf(
        reader, &err, kContext, policy);
    if (executor == nullptr) {
      std::cerr << "FAIL load " << err << '\n';
      return 1;
    }
    const auto& config = executor->GetConfig();
    if (!config.lfm2) {  // LFM2
      std::cerr << "FAIL not an lfm2 config\n";
      return 1;
    }
    const char* home = std::getenv("HOME");
    const std::string e2e =
        (home != nullptr ? home : ".") + std::string("/ls/lfm/e2e/");
    const auto prompts = ReadIdLines(e2e + "fork_prompt_ids.txt",  // LFM2
                                     "fork_prompt_ids.txt", 4);    // LFM2
    const auto fork = ReadIdLines(e2e + "fork_ids.txt",            // LFM2
                                  "fork_ids.txt", 4);              // LFM2
    for (const auto& row : fork) {                                 // LFM2
      if (row.size() != 48) {                                      // LFM2
        std::cerr << "FAIL fork_ids row has " << row.size()        // LFM2
                  << " ids, want 48\n";                            // LFM2
        return 1;                                                  // LFM2
      }
    }

    int prompts_ge8 = 0;                   // LFM2
    for (std::size_t p = 0; p < 4; ++p) {  // LFM2
      const auto& prompt = prompts[p];
      const auto ids = Generate48(executor.get(), prompt);
      // LFM2 checks: every id in vocab.
      for (const auto id : ids) {                                   // LFM2
        if (id >= config.vocab_size) {                              // LFM2
          std::cerr << "FAIL prompt" << p << " id out of range\n";  // LFM2
          return 1;                                                 // LFM2
        }
      }
      // LFM2 checks: no id repeated 12 times in a row.
      std::size_t run = 1;                                     // LFM2
      for (std::size_t i = 1; i < ids.size(); ++i) {           // LFM2
        run = ids[i] == ids[i - 1] ? run + 1 : 1;              // LFM2
        if (run >= 12) {                                       // LFM2
          std::cerr << "FAIL prompt" << p << " id " << ids[i]  // LFM2
                    << " repeats 12x\n";                       // LFM2
          return 1;                                            // LFM2
        }
      }
      // LFM2 checks: same prompt twice gives the same 48 ids.
      const auto again = Generate48(executor.get(), prompt);     // LFM2
      if (again != ids) {                                        // LFM2
        std::cerr << "FAIL prompt" << p << " rerun diverges\n";  // LFM2
        return 1;                                                // LFM2
      }
      // LFM2 checks: token-at-a-time gives the same first token as the chunk.
      executor->Reset();                                 // LFM2
      gufo::tokenization::TokenId streamed = 0;          // LFM2
      for (std::size_t i = 0; i < prompt.size(); ++i) {  // LFM2
        streamed = executor->ForwardToken(               // LFM2
            prompt[i], static_cast<std::uint32_t>(i),    // LFM2
            i + 1 == prompt.size());                     // LFM2
      }
      if (streamed != ids[0]) {  // LFM2
        std::cerr << "FAIL prompt" << p
                  << " streamed first=" << streamed      // LFM2
                  << " chunk first=" << ids[0] << '\n';  // LFM2
        return 1;                                        // LFM2
      }
      std::size_t leading = 0;  // LFM2
      while (leading < ids.size() &&
             ids[leading] == fork[p][leading]) {  // LFM2
        ++leading;                                // LFM2
      }
      if (leading >= 8) {  // LFM2
        ++prompts_ge8;     // LFM2
      }
      std::cout << "prompt" << p << " n=" << prompt.size()  // LFM2
                << " first=" << ids[0] << '\n';             // LFM2
      std::cout << "ids:";                                  // LFM2
      for (const auto id : ids) {                           // LFM2
        std::cout << ' ' << id;                             // LFM2
      }
      std::cout << '\n';                                              // LFM2
      std::cout << "text: "                                           // LFM2
                << executor->GetTokenizer().Decode(                   // LFM2
                       std::span<const gufo::tokenization::TokenId>(  // LFM2
                           ids.data(), ids.size()))                   // LFM2
                << '\n';                                              // LFM2
      std::cout << "prompt" << p << " leading_equal=" << leading      // LFM2
                << "/48\n";                                           // LFM2
    }
    std::cout << "prompts_ge8=" << prompts_ge8 << " want>=3\n";  // LFM2
    if (prompts_ge8 < 3) {                                       // LFM2
      std::cerr
          << "FAIL fork agreement below 8 leading on 3 prompts\n";  // LFM2
      return 1;                                                     // LFM2
    }
    std::cout << "lfm2_decode_test ALL PASS\n";  // LFM2
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "FAIL " << ex.what() << '\n';
    return 1;
  }
}
