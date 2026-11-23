// LFM2 decode e2e. A-LEAN weights, 4 fork prompts.
// For each prompt: ForwardPromptBatch chunk -> first token, then 47 greedy
// ForwardToken steps -> 48 ids. Compares the leading run against the fork's
// greedy 48 ids on the same A file.
// usage: lfm2_decode_test [gguf]  (or LFM2_A_GGUF under ctest)
// Exit 0 pass, 1 fail, 77 skipped (no file named).
// NOTE (deviation): ForwardPromptChunk is a private
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
std::vector<superroc::tokenization::TokenId> ParseIdList(
    const std::string& line, const char* what) {
  const auto open = line.find('[');
  const auto close = line.find(']');
  if (open == std::string::npos || close == std::string::npos ||
      close <= open) {
    throw std::runtime_error(std::string("bad id list in ") + what);
  }
  std::vector<superroc::tokenization::TokenId> ids;
  std::string cell;
  std::istringstream cells(line.substr(open + 1, close - open - 1));
  while (std::getline(cells, cell, ',')) {
    std::istringstream num(cell);
    unsigned long value = 0;
    num >> value;
    if (num.fail()) {
      throw std::runtime_error(std::string("bad id cell in ") + what);
    }
    ids.push_back(static_cast<superroc::tokenization::TokenId>(value));
  }
  return ids;
}

std::vector<std::vector<superroc::tokenization::TokenId>> ReadIdLines(
    const std::string& path, const char* what, std::size_t expect_lines) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error(std::string("cannot open ") + path);
  }
  std::vector<std::vector<superroc::tokenization::TokenId>> rows;
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
std::vector<superroc::tokenization::TokenId> Generate48(
    superroc::hip::QwenGpuExecutor* executor,
    const std::vector<superroc::tokenization::TokenId>& prompt) {
  executor->Reset();
  std::vector<superroc::tokenization::TokenId> ids;
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
    auto opened = superroc::core::GgufReader::OpenFile(path, &err);
    if (opened == nullptr) {
      std::cerr << "SKIP lfm2_decode_test: " << path << " unreadable: " << err
                << '\n';
      return 77;
    }
    std::shared_ptr<const superroc::core::GgufReader> reader(std::move(opened));
    superroc::hip::QwenExecutionPolicy policy;  // LFM2 (lfm2_ppl policy)
    policy.kv_cache_storage = superroc::hip::QwenKvCacheStorage::kFp32;
    policy.fuse_qk_norm_rope_kv = false;
    constexpr std::uint32_t kContext = 4096;
    auto executor = superroc::hip::QwenGpuExecutor::CreateFromGguf(
        reader, &err, kContext, policy);
    if (executor == nullptr) {
      std::cerr << "FAIL load " << err << '\n';
      return 1;
    }
    const auto& config = executor->GetConfig();
    if (!config.lfm2) {
      std::cerr << "FAIL not an lfm2 config\n";
      return 1;
    }
    const char* home = std::getenv("HOME");
    const std::string e2e =
        (home != nullptr ? home : ".") + std::string("/ls/lfm/e2e/");
    const auto prompts =
        ReadIdLines(e2e + "fork_prompt_ids.txt", "fork_prompt_ids.txt", 4);
    const auto fork = ReadIdLines(e2e + "fork_ids.txt", "fork_ids.txt", 4);
    for (const auto& row : fork) {
      if (row.size() != 48) {
        std::cerr << "FAIL fork_ids row has " << row.size()
                  << " ids, want 48\n";
        return 1;
      }
    }

    int prompts_ge8 = 0;
    for (std::size_t p = 0; p < 4; ++p) {
      const auto& prompt = prompts[p];
      const auto ids = Generate48(executor.get(), prompt);
      // LFM2 checks: every id in vocab.
      for (const auto id : ids) {
        if (id >= config.vocab_size) {
          std::cerr << "FAIL prompt" << p << " id out of range\n";
          return 1;
        }
      }
      // LFM2 checks: no id repeated 12 times in a row.
      std::size_t run = 1;
      for (std::size_t i = 1; i < ids.size(); ++i) {
        run = ids[i] == ids[i - 1] ? run + 1 : 1;
        if (run >= 12) {
          std::cerr << "FAIL prompt" << p << " id " << ids[i]
                    << " repeats 12x\n";
          return 1;
        }
      }
      // LFM2 checks: same prompt twice gives the same 48 ids.
      const auto again = Generate48(executor.get(), prompt);
      if (again != ids) {
        std::cerr << "FAIL prompt" << p << " rerun diverges\n";
        return 1;
      }
      // LFM2 checks: token-at-a-time gives the same first token as the chunk.
      executor->Reset();
      superroc::tokenization::TokenId streamed = 0;
      for (std::size_t i = 0; i < prompt.size(); ++i) {
        streamed = executor->ForwardToken(
            prompt[i], static_cast<std::uint32_t>(i), i + 1 == prompt.size());
      }
      if (streamed != ids[0]) {
        std::cerr << "FAIL prompt" << p << " streamed first=" << streamed
                  << " chunk first=" << ids[0] << '\n';
        return 1;
      }
      std::size_t leading = 0;
      while (leading < ids.size() && ids[leading] == fork[p][leading]) {
        ++leading;
      }
      if (leading >= 8) {
        ++prompts_ge8;
      }
      std::cout << "prompt" << p << " n=" << prompt.size()
                << " first=" << ids[0] << '\n';
      std::cout << "ids:";
      for (const auto id : ids) {
        std::cout << ' ' << id;
      }
      std::cout << '\n';
      std::cout << "text: "
                << executor->GetTokenizer().Decode(
                       std::span<const superroc::tokenization::TokenId>(
                           ids.data(), ids.size()))
                << '\n';
      std::cout << "prompt" << p << " leading_equal=" << leading << "/48\n";
    }
    std::cout << "prompts_ge8=" << prompts_ge8 << " want>=3\n";
    if (prompts_ge8 < 3) {
      std::cerr << "FAIL fork agreement below 8 leading on 3 prompts\n";
      return 1;
    }
    std::cout << "lfm2_decode_test ALL PASS\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "FAIL " << ex.what() << '\n';
    return 1;
  }
}
