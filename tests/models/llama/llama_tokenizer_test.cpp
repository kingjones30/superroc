// LLAMA: SentencePiece ids must match llama.cpp /tokenize (add_special true)
// for the six texts in the tokref fixture, and decode(encode(text)) == text.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/llama/spm_tokenizer.hpp"

namespace {

namespace fs = std::filesystem;
namespace json = gufo::json;

void Fail(const std::string& msg) {
  std::cerr << "FAIL llama_tokenizer_test: " << msg << "\n";
  std::exit(1);
}

std::string ReadFile(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    Fail("cannot read " + path.string());
  }
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

}  // namespace

int main() {
  const char* gguf_env = std::getenv("NANBEIGE_GGUF");
  if (gguf_env == nullptr || gguf_env[0] == '\0') {
    std::cout << "SKIP llama_tokenizer_test: NANBEIGE_GGUF is not set\n";
    return 77;
  }
  const fs::path gguf{gguf_env};
  if (!fs::exists(gguf)) {
    std::cout << "SKIP llama_tokenizer_test: " << gguf << " is absent\n";
    return 77;
  }

#ifndef NBG_TOKREF
  Fail("NBG_TOKREF is unset");
#else
  const fs::path tokref(NBG_TOKREF);
  if (!fs::exists(tokref)) {
    Fail("missing " + tokref.string());
  }

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(gguf, &error);
  if (reader == nullptr) {
    Fail("open GGUF: " + error);
  }
  auto tokenizer = gufo::models::llama::SpmTokenizer::FromGguf(*reader, &error);
  if (!tokenizer.has_value()) {
    Fail("FromGguf: " + error);
  }

  json::Value root;
  try {
    root = json::parse(ReadFile(tokref));
  } catch (const std::exception& e) {
    Fail(std::string("tokref parse: ") + e.what());
  }
  const json::Value* cases = root.find("cases");
  if (cases == nullptr || !cases->is_array() || cases->size() != 6) {
    Fail("tokref must contain 6 cases");
  }

  gufo::models::llama::SpmTokenizer::EncodeOptions options;
  options.add_special = true;
  options.parse_special = true;

  int matched = 0;
  int roundtrip_ok = 0;
  for (const json::Value& row : cases->items()) {
    const std::string name = row.member_str("name");
    const std::string text = row.member_str("text");
    const json::Value* ids_node = row.find("ids");
    if (ids_node == nullptr || !ids_node->is_array()) {
      Fail(name + ": no ids array");
    }
    std::vector<std::int32_t> expected;
    expected.reserve(ids_node->size());
    for (const json::Value& id : ids_node->items()) {
      expected.push_back(static_cast<std::int32_t>(id.as_size(0)));
    }
    const auto actual = tokenizer->Encode(text, options);
    const std::string decoded = tokenizer->Decode(actual);
    const bool ids_ok = actual == expected;
    const bool rt_ok = decoded == text;
    std::cout << "case " << name << " n_ref=" << expected.size()
              << " n_got=" << actual.size()
              << " ids=" << (ids_ok ? "OK" : "DIFF")
              << " roundtrip=" << (rt_ok ? "OK" : "DIFF") << "\n";
    if (!ids_ok) {
      std::size_t first = 0;
      while (first < actual.size() && first < expected.size() &&
             actual[first] == expected[first]) {
        ++first;
      }
      std::cerr << "  first_diff_index=" << first;
      if (first < expected.size()) {
        std::cerr << " ref=" << expected[first];
      }
      if (first < actual.size()) {
        std::cerr << " got=" << actual[first];
      }
      std::cerr << "\n";
    }
    if (!rt_ok) {
      std::cerr << "  decode got " << decoded.size() << " bytes, text "
                << text.size() << " bytes\n";
    }
    matched += ids_ok ? 1 : 0;
    roundtrip_ok += rt_ok ? 1 : 0;
  }

  std::cout << "llama_tokenizer_test matched=" << matched
            << "/6 roundtrip=" << roundtrip_ok
            << "/6 vocab=" << tokenizer->Size() << "\n";
  if (matched != 6 || roundtrip_ok != 6) {
    Fail("need 6/6 identical ids and 6/6 decode(encode(text)) == text");
  }
  std::cout << "PASS llama_tokenizer_test\n";
  return 0;
#endif
}
