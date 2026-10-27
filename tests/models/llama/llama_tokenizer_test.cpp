// LLAMA: SentencePiece ids must match llama.cpp /tokenize (add_special true)
// for the six texts in the tokref fixture, and decode(encode(text)) == text.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/llama/spm_tokenizer.hpp"

namespace {

namespace fs = std::filesystem;
namespace json = superroc::json;

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

void Expect(bool ok, const std::string& msg) {
  if (!ok) {
    Fail(msg);
  }
}

// Minimal metadata-only GGUF writer for the synthetic tokenizer tests.
void AppendPod(std::vector<std::uint8_t>& buf, std::uint64_t value,
               std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    buf.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  }
}

void AppendString(std::vector<std::uint8_t>& buf, std::string_view text) {
  AppendPod(buf, text.size(), 8);
  buf.insert(buf.end(), text.begin(), text.end());
}

void AppendStringValue(std::vector<std::uint8_t>& buf, std::string_view key,
                       std::string_view value) {
  AppendString(buf, key);
  AppendPod(buf, 8, 4);  // kString
  AppendString(buf, value);
}

void AppendUint32Value(std::vector<std::uint8_t>& buf, std::string_view key,
                       std::uint32_t value) {
  AppendString(buf, key);
  AppendPod(buf, 4, 4);  // kUint32
  AppendPod(buf, value, 4);
}

// Builds a 4-token llama SPM tokenizer GGUF: <unk>=0, <s>=1, </s>=2,
// hello=3. `pre` empty means the key is absent. `unk_type` is the
// token_type of token 0; `unknown_id` >= 0 adds the
// tokenizer.ggml.unknown_token_id key with that value.
std::vector<std::uint8_t> BuildSyntheticTokenizer(
    std::string_view pre, std::uint32_t vocab_size, std::uint32_t bos,
    std::uint32_t unk_type = 2, std::int64_t unknown_id = -1) {
  std::vector<std::uint8_t> buf;
  buf.insert(buf.end(), {'G', 'G', 'U', 'F'});
  AppendPod(buf, 3, 4);  // version
  AppendPod(buf, 0, 8);  // tensor count
  // arch, model, vocab_size, bos, eos, pad, tokens, scores, token_type,
  // plus pre when present, plus unknown_token_id when present.
  std::uint64_t metadata_count =
      9 + (pre.empty() ? 0 : 1) + (unknown_id < 0 ? 0 : 1);
  AppendPod(buf, metadata_count, 8);
  AppendStringValue(buf, "general.architecture", "llama");
  AppendStringValue(buf, "tokenizer.ggml.model", "llama");
  if (!pre.empty()) {
    AppendStringValue(buf, "tokenizer.ggml.pre", pre);
  }
  AppendUint32Value(buf, "llama.vocab_size", vocab_size);
  AppendUint32Value(buf, "tokenizer.ggml.bos_token_id", bos);
  AppendUint32Value(buf, "tokenizer.ggml.eos_token_id", 2);
  AppendUint32Value(buf, "tokenizer.ggml.padding_token_id", 0);
  if (unknown_id >= 0) {
    AppendUint32Value(buf, "tokenizer.ggml.unknown_token_id",
                      static_cast<std::uint32_t>(unknown_id));
  }
  // tokens array
  AppendString(buf, "tokenizer.ggml.tokens");
  AppendPod(buf, 9, 4);  // kArray
  AppendPod(buf, 8, 4);  // kString
  AppendPod(buf, 4, 8);
  for (std::string_view token : {"<unk>", "<s>", "</s>", "hello"}) {
    AppendString(buf, token);
  }
  // scores array
  AppendString(buf, "tokenizer.ggml.scores");
  AppendPod(buf, 9, 4);
  AppendPod(buf, 6, 4);  // kFloat32
  AppendPod(buf, 4, 8);
  for (int i = 0; i < 4; ++i) {
    AppendPod(buf, 0, 4);
  }
  // token_type array: unk_type, control, control, normal
  AppendString(buf, "tokenizer.ggml.token_type");
  AppendPod(buf, 9, 4);
  AppendPod(buf, 4, 4);  // kUint32
  AppendPod(buf, 4, 8);
  for (std::uint32_t type : {unk_type, 3U, 3U, 1U}) {
    AppendPod(buf, type, 4);
  }
  // ParseHeaders aligns the tensor payload base to the 32-byte GGUF
  // alignment; a real file always carries tensor bytes after the headers, so
  // pad this header-only buffer the same way.
  while (buf.size() % 32 != 0) {
    buf.push_back(0);
  }
  return buf;
}

std::unique_ptr<superroc::core::GgufReader> OpenSynthetic(
    const std::vector<std::uint8_t>& file, std::string& error) {
  auto reader =
      superroc::core::GgufReader::OpenMemory(file.data(), file.size(), &error);
  Expect(reader != nullptr, "open synthetic tokenizer GGUF: " + error);
  return reader;
}

// F7: the tokenizer array must match llama.vocab_size, and bos/eos/pad must
// be in range. Runs without NANBEIGE_GGUF.
void TestTokenizerVocabValidation() {
  std::string error;
  const auto short_vocab =
      BuildSyntheticTokenizer("default", /*vocab_size=*/3, /*bos=*/1);
  Expect(!superroc::models::llama::SpmTokenizer::FromGguf(
              *OpenSynthetic(short_vocab, error), &error)
              .has_value(),
         "tokenizer longer than llama.vocab_size must be refused");
  const auto bad_bos =
      BuildSyntheticTokenizer("default", /*vocab_size=*/4, /*bos=*/99);
  Expect(!superroc::models::llama::SpmTokenizer::FromGguf(
              *OpenSynthetic(bad_bos, error), &error)
              .has_value(),
         "out-of-range bos_token_id must be refused");
  const auto good =
      BuildSyntheticTokenizer("default", /*vocab_size=*/4, /*bos=*/1);
  Expect(superroc::models::llama::SpmTokenizer::FromGguf(
             *OpenSynthetic(good, error), &error)
             .has_value(),
         "matching vocab with in-range ids must load: " + error);
}

// F8: a character with no vocab piece must fall back to <unk>, never emit
// -1 (the serve backend casts ids to unsigned for the embedding lookup).
void TestByteFallbackNeverNegative() {
  std::string error;
  const auto file =
      BuildSyntheticTokenizer("default", /*vocab_size=*/4, /*bos=*/1);
  auto tokenizer = superroc::models::llama::SpmTokenizer::FromGguf(
      *OpenSynthetic(file, error), &error);
  Expect(tokenizer.has_value(), "synthetic tokenizer must load: " + error);
  superroc::models::llama::SpmTokenizer::EncodeOptions options;
  options.add_special = false;
  options.parse_special = false;
  const auto ids = tokenizer->Encode("~", options);
  Expect(!ids.empty(), "byte fallback must emit a token");
  for (const auto id : ids) {
    Expect(id >= 0, "Encode must never emit a negative id");
  }
}

// NBFAST: a file that types <unk> as CONTROL (bartowski instruct GGUF)
// loads through tokenizer.ggml.unknown_token_id; without the key, or with
// an out-of-range id, it is still refused.
void TestUnknownTokenIdFallback() {
  std::string error;
  const auto control_unk = BuildSyntheticTokenizer("default", /*vocab_size=*/4,
                                                   /*bos=*/1, /*unk_type=*/3,
                                                   /*unknown_id=*/0);
  auto tokenizer = superroc::models::llama::SpmTokenizer::FromGguf(
      *OpenSynthetic(control_unk, error), &error);
  Expect(tokenizer.has_value(),
         "CONTROL <unk> with unknown_token_id must load: " + error);
  superroc::models::llama::SpmTokenizer::EncodeOptions options;
  options.add_special = false;
  options.parse_special = false;
  const auto ids = tokenizer->Encode("~", options);
  Expect(!ids.empty(), "byte fallback must emit a token");
  for (const auto id : ids) {
    Expect(id == 0, "byte fallback must emit unknown_token_id 0");
  }
  const auto no_key = BuildSyntheticTokenizer("default", /*vocab_size=*/4,
                                              /*bos=*/1, /*unk_type=*/3);
  Expect(!superroc::models::llama::SpmTokenizer::FromGguf(
              *OpenSynthetic(no_key, error), &error)
              .has_value(),
         "CONTROL <unk> without unknown_token_id must be refused");
  const auto bad_id = BuildSyntheticTokenizer("default", /*vocab_size=*/4,
                                              /*bos=*/1, /*unk_type=*/3,
                                              /*unknown_id=*/99);
  Expect(!superroc::models::llama::SpmTokenizer::FromGguf(
              *OpenSynthetic(bad_id, error), &error)
              .has_value(),
         "out-of-range unknown_token_id must be refused");
}

// F12: only the default pre-tokenizer is implemented.
void TestNonDefaultPreRefused() {
  std::string error;
  const auto file =
      BuildSyntheticTokenizer("llama-bpe", /*vocab_size=*/4, /*bos=*/1);
  Expect(!superroc::models::llama::SpmTokenizer::FromGguf(
              *OpenSynthetic(file, error), &error)
              .has_value(),
         "non-default tokenizer.ggml.pre must be refused");
}

}  // namespace

int main() {
  TestTokenizerVocabValidation();
  TestByteFallbackNeverNegative();
  TestUnknownTokenIdFallback();
  TestNonDefaultPreRefused();
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
  const auto reader = superroc::core::GgufReader::OpenFile(gguf, &error);
  if (reader == nullptr) {
    Fail("open GGUF: " + error);
  }
  auto tokenizer = superroc::models::llama::SpmTokenizer::FromGguf(*reader, &error);
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

  superroc::models::llama::SpmTokenizer::EncodeOptions options;
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
