// Flint: tokenizer equality against tokref.jsonl (official tokenizers library).
// Opens FLINT_GGUF (tokenizer only). Exits 77 (SKIP) when the GGUF is absent.

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

namespace fs = std::filesystem;
namespace json = superroc::json;

std::string ReadFile(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

}  // namespace

int main() {
  const char* gguf_env = std::getenv("FLINT_GGUF");
  fs::path gguf;
  if (gguf_env != nullptr && gguf_env[0] != '\0') {
    gguf = fs::path(gguf_env);
  } else if (const char* home = std::getenv("HOME");
             home != nullptr && home[0] != '\0') {
    gguf = fs::path(home) / "models" / "flint" /
           "Flint-40B-A2.5B-Q4_0-ROCmFP4-COHERENT-PAS.gguf";
  }
  if (gguf.empty() || !fs::exists(gguf)) {
    std::cout << "SKIP flint_tokenizer_test: GGUF is absent\n";
    return 77;
  }

#ifndef FLINT_TOKREF
  std::cout << "FAIL flint_tokenizer_test: FLINT_TOKREF is unset\n";
  return 1;
#else
  const fs::path tokref(FLINT_TOKREF);
  if (!fs::exists(tokref)) {
    std::cout << "FAIL flint_tokenizer_test: missing " << tokref << "\n";
    return 1;
  }

  std::string error;
  const auto reader = superroc::core::GgufReader::OpenFile(gguf, &error);
  if (reader == nullptr) {
    std::cout << "FAIL flint_tokenizer_test: cannot open GGUF: " << error
              << "\n";
    return 1;
  }
  auto tokenizer =
      superroc::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (tokenizer == nullptr) {
    std::cout << "FAIL flint_tokenizer_test: cannot load tokenizer: " << error
              << "\n";
    return 1;
  }

  const auto stop_ok =
      tokenizer->IsStopToken(150001) && tokenizer->IsStopToken(150003) &&
      tokenizer->IsStopToken(150005) && tokenizer->IsStopToken(150018) &&
      !tokenizer->IsStopToken(150004) && !tokenizer->IsStopToken(150006);
  if (!stop_ok) {
    std::cout << "FAIL flint_tokenizer_test: IsStopToken "
              << "150001=" << tokenizer->IsStopToken(150001)
              << " 150003=" << tokenizer->IsStopToken(150003)
              << " 150005=" << tokenizer->IsStopToken(150005)
              << " 150018=" << tokenizer->IsStopToken(150018)
              << " 150004=" << tokenizer->IsStopToken(150004)
              << " 150006=" << tokenizer->IsStopToken(150006) << "\n";
    return 1;
  }
  std::cout << "PASS flint_tokenizer_test: IsStopToken "
            << "150001/150003/150005/150018 true, 150004/150006 false\n";

  superroc::tokenization::TokenizerOptions options;
  options.add_bos = false;
  options.add_eos = false;
  options.parse_special_tokens = true;

  std::ifstream input(tokref);
  std::string line;
  int n = 0;
  int failures = 0;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    ++n;
    json::Value row;
    try {
      row = json::parse(line);
    } catch (const std::exception& e) {
      std::cout << "FAIL line " << n << ": parse: " << e.what() << "\n";
      ++failures;
      continue;
    }
    const std::string text = row.member_str("text");
    const json::Value* ids_node = row.find("ids");
    if (ids_node == nullptr || !ids_node->is_array()) {
      std::cout << "FAIL line " << n << ": no ids array\n";
      ++failures;
      continue;
    }
    std::vector<superroc::tokenization::TokenId> expected;
    for (const json::Value& id : ids_node->items()) {
      expected.push_back(
          static_cast<superroc::tokenization::TokenId>(id.as_size(0)));
    }
    const auto actual = tokenizer->Encode(text, options);
    if (actual == expected) {
      std::cout << "PASS " << n << ": " << actual.size() << " ids match\n";
      continue;
    }
    ++failures;
    std::size_t first = 0;
    while (first < actual.size() && first < expected.size() &&
           actual[first] == expected[first]) {
      ++first;
    }
    std::cout << "FAIL " << n << ": first mismatch at index " << first
              << " (actual " << actual.size() << " ids, expected "
              << expected.size() << " ids)\n";
  }

  if (n != 24) {
    std::cout << "FAIL flint_tokenizer_test: expected 24 lines, got " << n
              << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cout << "FAIL flint_tokenizer_test: " << failures
              << " line(s) differ\n";
    return 1;
  }
  std::cout << "PASS flint_tokenizer_test: all " << n << " lines match\n";
  return 0;
#endif
}
