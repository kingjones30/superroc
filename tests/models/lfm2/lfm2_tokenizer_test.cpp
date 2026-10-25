// LFM2: tokenizer equality against the fork's llama-tokenize output.
// Opens LFM2_GGUF (tokenizer only), encodes every *.txt in LFM2_TOK_DIR with
// add_bos=true, parse_special_tokens=false and compares to <name>.ids.
// Exits 77 (SKIP) when LFM2_GGUF is unset.

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

namespace fs = std::filesystem;

std::string ReadFile(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

std::string Trim(std::string_view value) {
  std::size_t first = 0;
  while (first < value.size() &&
         (value[first] == ' ' || value[first] == '\t' || value[first] == '\n' ||
          value[first] == '\r')) {
    ++first;
  }
  std::size_t last = value.size();
  while (last > first && (value[last - 1] == ' ' || value[last - 1] == '\t' ||
                          value[last - 1] == '\n' || value[last - 1] == '\r')) {
    --last;
  }
  return std::string(value.substr(first, last - first));
}

// One line holding a Python list, e.g. "[124894, 35808, 20]".
bool ParseIdList(const std::string& text, std::vector<std::uint32_t>* out,
                 std::string* error) {
  const auto open = text.find('[');
  const auto close = text.rfind(']');
  if (open == std::string::npos || close == std::string::npos || close < open) {
    *error = "ids file is not a [...] list";
    return false;
  }
  std::string body = text.substr(open + 1, close - open - 1);
  std::stringstream stream(body);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const std::string cell = Trim(item);
    if (cell.empty()) {
      continue;
    }
    try {
      out->push_back(static_cast<std::uint32_t>(std::stoul(cell)));
    } catch (const std::exception&) {
      *error = std::string("bad id cell: ") + cell;
      return false;
    }
  }
  return true;
}

void PrintWindow(const std::vector<std::uint32_t>& ids, std::size_t center,
                 const char* label) {
  const std::size_t first = center >= 4 ? center - 4 : 0;
  const std::size_t last = std::min(ids.size(), center + 4);
  std::cout << "  " << label << " ids[" << first << ".." << last << "]:";
  for (std::size_t i = first; i < last; ++i) {
    std::cout << (i == center ? " >>" : " ") << ids[i];
  }
  std::cout << "\n";
}

}  // namespace

int main() {
  const char* gguf = std::getenv("LFM2_GGUF");
  if (gguf == nullptr || gguf[0] == '\0') {
    std::cout << "SKIP lfm2_tokenizer_test: LFM2_GGUF is unset\n";
    return 77;
  }
  const char* tok_dir = std::getenv("LFM2_TOK_DIR");
  if (tok_dir == nullptr || tok_dir[0] == '\0') {
    std::cout << "SKIP lfm2_tokenizer_test: LFM2_TOK_DIR is unset\n";
    return 77;
  }

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(fs::path(gguf), &error);
  if (reader == nullptr) {
    std::cout << "FAIL lfm2_tokenizer_test: cannot open GGUF: " << error
              << "\n";
    return 1;
  }
  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (tokenizer == nullptr) {
    std::cout << "FAIL lfm2_tokenizer_test: cannot load tokenizer: " << error
              << "\n";
    return 1;
  }

  std::vector<fs::path> inputs;
  for (const auto& entry : fs::directory_iterator(fs::path(tok_dir))) {
    if (entry.is_regular_file() && entry.path().extension() == ".txt") {
      inputs.push_back(entry.path());
    }
  }
  std::sort(inputs.begin(), inputs.end());
  if (inputs.empty()) {
    std::cout << "FAIL lfm2_tokenizer_test: no *.txt in " << tok_dir << "\n";
    return 1;
  }

  gufo::tokenization::TokenizerOptions options;
  options.add_bos = true;
  options.add_eos = false;
  options.parse_special_tokens = false;

  int failures = 0;
  for (const auto& path : inputs) {
    const std::string name = path.stem().string();
    const fs::path ids_path = path.parent_path() / (name + ".ids");
    std::vector<std::uint32_t> expected;
    std::string parse_error;
    if (!ParseIdList(ReadFile(ids_path), &expected, &parse_error)) {
      std::cout << "FAIL " << name << ": " << parse_error << "\n";
      ++failures;
      continue;
    }
    const std::string text = ReadFile(path);
    const std::vector<std::uint32_t> actual = tokenizer->Encode(text, options);
    if (actual == expected) {
      std::cout << "PASS " << name << ": " << actual.size() << " ids match\n";
      continue;
    }
    ++failures;
    std::size_t first = 0;
    while (first < actual.size() && first < expected.size() &&
           actual[first] == expected[first]) {
      ++first;
    }
    std::cout << "FAIL " << name << ": first mismatch at index " << first
              << " (actual " << actual.size() << " ids, expected "
              << expected.size() << " ids)\n";
    PrintWindow(expected, std::min(first, expected.size()), "expected");
    PrintWindow(actual, std::min(first, actual.size()), "actual  ");
  }

  if (failures != 0) {
    std::cout << "FAIL lfm2_tokenizer_test: " << failures
              << " file(s) differ\n";
    return 1;
  }
  std::cout << "PASS lfm2_tokenizer_test: all " << inputs.size()
            << " files match\n";
  return 0;
}
