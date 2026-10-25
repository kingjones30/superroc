// ling3_tokenize_dump.cpp — LQ1's tokenizer parity probe.
//
// Prints gufo's token ids for a text file so they can be diffed against
// llama.cpp's `llama-tokenize`. Loads ONLY the vocabulary from the GGUF: no
// weights are read, no engine is constructed, no GPU is touched. The ids come
// from LingTokenizer::Encode with exactly the arguments tools/ling3_batch_test.cpp
// uses (defaults: no BOS, parse_special on), so what is compared here is what
// the perplexity gate actually feeds the model.
//
//   gufo_ling3_tokenize <gguf> <text-file> [mode] [flags]
//
//   modes (default is --lines)
//     --lines        one id per line
//     --ids          python-style list, the shape llama-tokenize --ids prints
//     --count        just the number of ids
//     --pieces       per token: id \t hex(text the id contributes)
//     --dump-vocab   id \t type \t hex(text) for every id; <text-file> ignored
//                    (type is GGUF tokenizer.ggml.token_type, see llama.h)
//
//   flags
//     --no-parse-special   pass parse_special=false to Encode
//     --add-bos            prepend BOS (this model's add_bos_token is false)
//
// Everything parseable goes to stdout; the load banner and errors go to stderr.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling3/tokenizer.hpp"

namespace {

namespace l3 = ::gufo::models::ling3;

enum class Mode { kLines, kIds, kCount, kPieces, kDumpVocab };

std::string Hex(std::string_view s) {
  static const char* kDigits = "0123456789abcdef";
  std::string out;
  out.reserve(s.size() * 2);
  for (const unsigned char c : s) {
    out.push_back(kDigits[c >> 4]);
    out.push_back(kDigits[c & 0xF]);
  }
  return out;
}

bool ReadWholeFile(const std::filesystem::path& path, std::string* out,
                   std::string* error_msg) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error_msg = "cannot open text file: " + path.string();
    return false;
  }
  char buf[65536];
  while (in) {
    in.read(buf, static_cast<std::streamsize>(sizeof buf));
    const auto n = in.gcount();
    if (n > 0) {
      out->append(buf, static_cast<std::size_t>(n));
    }
  }
  return true;
}

void PrintUsage() {
  std::fprintf(stderr,
               "usage: gufo_ling3_tokenize <gguf> <text-file> "
               "[--lines|--ids|--count|--pieces|--dump-vocab] "
               "[--no-parse-special] [--add-bos]\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::setbuf(stdout, nullptr);

  Mode mode = Mode::kLines;
  bool parse_special = true;
  bool add_bos = false;
  std::string gguf_path;
  std::string text_path;

  for (int i = 1; i < argc; ++i) {
    const std::string_view a(argv[i]);
    if (a == "--lines") {
      mode = Mode::kLines;
    } else if (a == "--ids") {
      mode = Mode::kIds;
    } else if (a == "--count") {
      mode = Mode::kCount;
    } else if (a == "--pieces") {
      mode = Mode::kPieces;
    } else if (a == "--dump-vocab") {
      mode = Mode::kDumpVocab;
    } else if (a == "--no-parse-special") {
      parse_special = false;
    } else if (a == "--add-bos") {
      add_bos = true;
    } else if (a == "--help" || a == "-h") {
      PrintUsage();
      return 0;
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "unknown flag: %s\n", argv[i]);
      PrintUsage();
      return 2;
    } else if (gguf_path.empty()) {
      gguf_path = argv[i];
    } else if (text_path.empty()) {
      text_path = argv[i];
    } else {
      std::fprintf(stderr, "extra argument: %s\n", argv[i]);
      PrintUsage();
      return 2;
    }
  }

  if (gguf_path.empty()) {
    PrintUsage();
    return 2;
  }
  if (text_path.empty() && mode != Mode::kDumpVocab) {
    std::fprintf(stderr, "missing text file\n");
    PrintUsage();
    return 2;
  }

  std::string err;
  std::fprintf(stderr, "reading vocabulary from %s\n", gguf_path.c_str());
  auto reader = gufo::core::GgufReader::OpenFile(gguf_path, &err);
  if (!reader) {
    std::fprintf(stderr, "FAIL open gguf: %s\n", err.c_str());
    return 1;
  }
  auto tok = l3::LingTokenizer::FromGguf(*reader, &err);
  if (!tok) {
    std::fprintf(stderr, "FAIL tokenizer: %s\n", err.c_str());
    return 1;
  }
  std::fprintf(stderr, "vocabulary size %u\n", tok->Size());

  if (mode == Mode::kDumpVocab) {
    // The raw GGUF token_type alongside the decoded text: the report needs to
    // know which ids llama.cpp would partition out as special tokens.
    // GgufMetadataValue widens every array of integers to int64/uint64.
    std::vector<std::int64_t> types;
    const auto* type_meta = reader->FindMetadata("tokenizer.ggml.token_type");
    if (type_meta != nullptr) {
      if (const auto* v =
              std::get_if<std::vector<std::int64_t>>(&type_meta->value)) {
        types.assign(v->begin(), v->end());
      } else if (const auto* v =
                     std::get_if<std::vector<std::uint64_t>>(&type_meta->value)) {
        types.reserve(v->size());
        for (const auto u : *v) {
          types.push_back(static_cast<std::int64_t>(u));
        }
      }
    }
    const std::uint32_t n = tok->Size();
    for (std::uint32_t i = 0; i < n; ++i) {
      const auto ty = i < types.size()
                          ? static_cast<std::int32_t>(types[i])
                          : static_cast<std::int32_t>(-1);
      std::printf("%u\t%d\t%s\n", i, ty,
                  Hex(tok->TokenText(static_cast<std::int32_t>(i))).c_str());
    }
    return 0;
  }

  std::string text;
  if (!ReadWholeFile(text_path, &text, &err)) {
    std::fprintf(stderr, "FAIL read: %s\n", err.c_str());
    return 1;
  }
  std::fprintf(stderr, "text %s: %zu bytes\n", text_path.c_str(), text.size());

  const std::vector<std::int32_t> ids = tok->Encode(text, add_bos, parse_special);

  switch (mode) {
    case Mode::kCount:
      std::printf("%zu\n", ids.size());
      return 0;
    case Mode::kLines:
      for (const std::int32_t id : ids) {
        std::printf("%d\n", id);
      }
      return 0;
    case Mode::kIds: {
      std::printf("[");
      for (std::size_t i = 0; i < ids.size(); ++i) {
        std::printf(i == 0 ? "%d" : ", %d", ids[i]);
      }
      std::printf("]\n");
      return 0;
    }
    case Mode::kPieces:
      // id \t hex(text this id contributes). The GPT-2 byte table is a
      // bijection, so this names the exact input span the id covers.
      for (const std::int32_t id : ids) {
        std::printf("%d\t%s\n", id, Hex(tok->TokenText(id)).c_str());
      }
      return 0;
    case Mode::kDumpVocab:
      break;
  }
  return 0;
}
