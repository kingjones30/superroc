#include "src/models/llama/spm_tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <queue>
#include <utility>
#include <variant>

namespace superroc::models::llama {
namespace {

constexpr char kUnderline[] = "\xE2\x96\x81";  // U+2581

void ReplaceAll(std::string* text, std::string_view from, std::string_view to) {
  std::size_t pos = 0;
  while ((pos = text->find(from, pos)) != std::string::npos) {
    text->replace(pos, from.size(), to);
    pos += to.size();
  }
}

void EscapeWhitespace(std::string* text) {
  ReplaceAll(text, " ", kUnderline);
}

void UnescapeWhitespace(std::string* text) {
  ReplaceAll(text, kUnderline, " ");
}

bool ParseByteToken(std::string_view text, unsigned char* byte) {
  if (text.size() != 6 || text[0] != '<' || text[1] != '0' || text[2] != 'x' ||
      text[5] != '>') {
    return false;
  }
  auto hex = [](char c) -> int {
    if (c >= '0' && c <= '9') {
      return c - '0';
    }
    if (c >= 'A' && c <= 'F') {
      return 10 + (c - 'A');
    }
    if (c >= 'a' && c <= 'f') {
      return 10 + (c - 'a');
    }
    return -1;
  };
  const int hi = hex(text[3]);
  const int lo = hex(text[4]);
  if (hi < 0 || lo < 0) {
    return false;
  }
  *byte = static_cast<unsigned char>((hi << 4) | lo);
  return true;
}

}  // namespace

std::size_t SpmTokenizer::Utf8Len(unsigned char lead) {
  static constexpr std::uint8_t kLen[16] = {1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 2, 2, 3, 4};
  return kLen[lead >> 4];
}

std::optional<SpmTokenizer> SpmTokenizer::FromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto fail = [error_msg](std::string_view msg) {
    if (error_msg != nullptr) {
      *error_msg = std::string(msg);
    }
    return std::nullopt;
  };
  const auto model = reader.GetMetadataString("tokenizer.ggml.model");
  if (!model.has_value() || *model != "llama") {
    return fail("tokenizer.ggml.model must be llama");
  }
  // LLAMA (F12): only the default pre-tokenizer is implemented; any other
  // pre would silently mis-segment every prompt.
  const auto pre = reader.GetMetadataString("tokenizer.ggml.pre");
  if (pre.has_value() && *pre != "default") {
    return fail("tokenizer.ggml.pre must be default");
  }
  const auto tokens = reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  const auto* scores_meta = reader.FindMetadata("tokenizer.ggml.scores");
  const auto* score_values =
      scores_meta != nullptr
          ? std::get_if<std::vector<double>>(&scores_meta->value)
          : nullptr;
  const auto types = reader.GetMetadataUintArray("tokenizer.ggml.token_type");
  if (tokens.empty() || score_values == nullptr ||
      tokens.size() != score_values->size() || tokens.size() != types.size()) {
    return fail("tokenizer tokens/scores/token_type length mismatch");
  }
  // LLAMA (F7): an id >= vocab_size overruns the embedding table, so the
  // tokenizer array must match the model vocab exactly.
  const auto vocab_size = reader.GetMetadataUint32("llama.vocab_size");
  if (!vocab_size.has_value() || *vocab_size == 0U ||
      tokens.size() != *vocab_size) {
    return fail("tokenizer tokens must match llama.vocab_size");
  }

  SpmTokenizer tok;
  tok.texts_.reserve(tokens.size());
  tok.scores_.reserve(tokens.size());
  tok.types_.reserve(tokens.size());
  tok.byte_id_.fill(-1);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tok.texts_.emplace_back(tokens[i]);
    tok.scores_.push_back(static_cast<float>((*score_values)[i]));
    tok.types_.push_back(static_cast<TokenType>(types[i]));
    tok.token_to_id_[tok.texts_.back()] = static_cast<std::int32_t>(i);
    if (tok.types_.back() == TokenType::kByte) {
      unsigned char byte = 0;
      if (ParseByteToken(tok.texts_.back(), &byte)) {
        tok.byte_id_[byte] = static_cast<std::int32_t>(i);
      }
    }
  }
  for (std::size_t i = 0; i < tok.types_.size(); ++i) {
    const TokenType type = tok.types_[i];
    if (type == TokenType::kControl || type == TokenType::kUserDefined ||
        type == TokenType::kUnknown) {
      tok.specials_.push_back(static_cast<std::int32_t>(i));
    }
  }
  std::sort(tok.specials_.begin(), tok.specials_.end(),
            [&](std::int32_t a, std::int32_t b) {
              return tok.texts_[static_cast<std::size_t>(a)].size() >
                     tok.texts_[static_cast<std::size_t>(b)].size();
            });
  // LLAMA (F8): a byte with no fallback token made Encode emit -1, which the
  // serve backend casts to 4294967295 and reads out of bounds. Fall back to
  // the unknown token so Encode never emits a negative id.
  std::int32_t unk = -1;
  for (std::size_t i = 0; i < tok.types_.size(); ++i) {
    if (tok.types_[i] == TokenType::kUnknown) {
      unk = static_cast<std::int32_t>(i);
      break;
    }
  }
  if (unk < 0) {
    // NBFAST: some published GGUFs (bartowski Nanbeige4.3-33B instruct)
    // type <unk> as CONTROL instead of UNKNOWN; honor the explicit
    // unknown_token_id metadata before refusing the file. Files with an
    // UNKNOWN-typed token never reach this branch.
    if (const auto unk_id =
            reader.GetMetadataUint32("tokenizer.ggml.unknown_token_id")) {
      if (*unk_id >= tokens.size()) {
        return fail("tokenizer.ggml.unknown_token_id out of range");
      }
      unk = static_cast<std::int32_t>(*unk_id);
    }
  }
  if (unk < 0) {
    return fail("tokenizer has no unknown token");
  }
  for (unsigned i = 0; i < 256; ++i) {
    if (tok.byte_id_[i] < 0) {
      tok.byte_id_[i] = unk;
    }
  }

  // LLAMA (F7): bos/eos/pad are emitted unchecked by Encode; an id outside
  // the vocab overruns the embedding table.
  if (const auto bos =
          reader.GetMetadataUint32("tokenizer.ggml.bos_token_id")) {
    if (*bos >= tokens.size()) {
      return fail("tokenizer.ggml.bos_token_id out of range");
    }
    tok.bos_ = static_cast<std::int32_t>(*bos);
  }
  if (const auto eos =
          reader.GetMetadataUint32("tokenizer.ggml.eos_token_id")) {
    if (*eos >= tokens.size()) {
      return fail("tokenizer.ggml.eos_token_id out of range");
    }
    tok.eos_ = static_cast<std::int32_t>(*eos);
  }
  if (const auto pad =
          reader.GetMetadataUint32("tokenizer.ggml.padding_token_id")) {
    if (*pad >= tokens.size()) {
      return fail("tokenizer.ggml.padding_token_id out of range");
    }
    tok.pad_ = static_cast<std::int32_t>(*pad);
  }
  if (const auto add_bos =
          reader.GetMetadataBool("tokenizer.ggml.add_bos_token")) {
    tok.add_bos_ = *add_bos;
  }
  if (const auto add_eos =
          reader.GetMetadataBool("tokenizer.ggml.add_eos_token")) {
    tok.add_eos_ = *add_eos;
  }
  return tok;
}

std::int32_t SpmTokenizer::TokenId(std::string_view text) const {
  const auto it = token_to_id_.find(std::string(text));
  return it == token_to_id_.end() ? -1 : it->second;
}

std::int32_t SpmTokenizer::ByteToken(unsigned char byte) const {
  const std::int32_t id = byte_id_[byte];
  if (id >= 0) {
    return id;
  }
  char buf[8];
  std::snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned>(byte));
  return TokenId(buf);
}

void SpmTokenizer::Resegment(
    const std::vector<Symbol>& symbols,
    const std::unordered_map<std::string, std::pair<int, int>>& rev_merge,
    int index, std::vector<std::int32_t>* out) const {
  const Symbol& symbol = symbols[static_cast<std::size_t>(index)];
  const std::string text(symbol.text, symbol.n);
  const std::int32_t id = TokenId(text);
  if (id >= 0) {
    out->push_back(id);
    return;
  }
  const auto it = rev_merge.find(text);
  if (it == rev_merge.end()) {
    for (std::size_t j = 0; j < symbol.n; ++j) {
      out->push_back(ByteToken(static_cast<unsigned char>(symbol.text[j])));
    }
    return;
  }
  Resegment(symbols, rev_merge, it->second.first, out);
  Resegment(symbols, rev_merge, it->second.second, out);
}

void SpmTokenizer::SpmPiece(std::string_view text,
                            std::vector<std::int32_t>* out) const {
  if (text.empty()) {
    return;
  }
  std::vector<Symbol> symbols;
  std::size_t offs = 0;
  while (offs < text.size()) {
    std::size_t len = Utf8Len(static_cast<unsigned char>(text[offs]));
    if (len > text.size() - offs) {
      len = text.size() - offs;
    }
    Symbol sym;
    sym.text = text.data() + offs;
    sym.n = len;
    sym.prev = static_cast<int>(symbols.size()) - 1;
    sym.next = -1;
    if (!symbols.empty()) {
      symbols.back().next = static_cast<int>(symbols.size());
    }
    offs += len;
    symbols.push_back(sym);
  }

  std::priority_queue<Bigram, std::vector<Bigram>, BigramGreater> work;
  std::unordered_map<std::string, std::pair<int, int>> rev_merge;
  const auto try_add = [&](int left, int right) {
    if (left < 0 || right < 0) {
      return;
    }
    const Symbol& l = symbols[static_cast<std::size_t>(left)];
    const Symbol& r = symbols[static_cast<std::size_t>(right)];
    const std::string merged(l.text, l.n + r.n);
    const std::int32_t id = TokenId(merged);
    if (id < 0) {
      return;
    }
    Bigram bigram;
    bigram.left = left;
    bigram.right = right;
    bigram.score = scores_[static_cast<std::size_t>(id)];
    bigram.size = merged.size();
    work.push(bigram);
    rev_merge[merged] = {left, right};
  };
  for (int i = 1; i < static_cast<int>(symbols.size()); ++i) {
    try_add(i - 1, i);
  }
  while (!work.empty()) {
    const Bigram bigram = work.top();
    work.pop();
    Symbol& left = symbols[static_cast<std::size_t>(bigram.left)];
    Symbol& right = symbols[static_cast<std::size_t>(bigram.right)];
    if (left.n == 0 || right.n == 0 || left.n + right.n != bigram.size) {
      continue;
    }
    left.n += right.n;
    right.n = 0;
    left.next = right.next;
    if (right.next >= 0) {
      symbols[static_cast<std::size_t>(right.next)].prev = bigram.left;
    }
    try_add(left.prev, bigram.left);
    try_add(bigram.left, left.next);
  }
  for (int i = 0; i != -1; i = symbols[static_cast<std::size_t>(i)].next) {
    Resegment(symbols, rev_merge, i, out);
  }
}

std::vector<std::int32_t> SpmTokenizer::Encode(std::string_view text,
                                               EncodeOptions options) const {
  struct Frag {
    bool is_token{false};
    std::int32_t id{-1};
    std::string text;
  };
  std::vector<Frag> frags;
  if (!text.empty()) {
    frags.push_back(Frag{false, -1, std::string(text)});
    for (const std::int32_t sid : specials_) {
      const TokenType type = types_[static_cast<std::size_t>(sid)];
      if (!options.parse_special &&
          (type == TokenType::kControl || type == TokenType::kUnknown)) {
        continue;
      }
      const std::string& needle = texts_[static_cast<std::size_t>(sid)];
      if (needle.empty()) {
        continue;
      }
      std::vector<Frag> next;
      next.reserve(frags.size());
      for (const Frag& frag : frags) {
        if (frag.is_token) {
          next.push_back(frag);
          continue;
        }
        std::size_t start = 0;
        while (start < frag.text.size()) {
          const std::size_t match = frag.text.find(needle, start);
          if (match == std::string::npos) {
            next.push_back(Frag{false, -1, frag.text.substr(start)});
            break;
          }
          if (match > start) {
            next.push_back(
                Frag{false, -1, frag.text.substr(start, match - start)});
          }
          next.push_back(Frag{true, sid, {}});
          start = match + needle.size();
          if (start == frag.text.size()) {
            break;
          }
        }
      }
      frags.swap(next);
    }
  }

  std::vector<std::int32_t> out;
  if (options.add_special && add_bos_) {
    out.push_back(bos_);
  }
  bool prev_special = true;
  for (const Frag& frag : frags) {
    if (frag.is_token) {
      out.push_back(frag.id);
      prev_special = true;
      continue;
    }
    std::string piece;
    if (prev_special) {
      piece.push_back(' ');
    }
    piece.append(frag.text);
    EscapeWhitespace(&piece);
    SpmPiece(piece, &out);
    prev_special = false;
  }
  if (options.add_special && add_eos_) {
    out.push_back(eos_);
  }
  return out;
}

std::string SpmTokenizer::TokenText(std::int32_t token) const {
  if (token < 0 || static_cast<std::size_t>(token) >= texts_.size()) {
    return {};
  }
  const TokenType type = types_[static_cast<std::size_t>(token)];
  const std::string& raw = texts_[static_cast<std::size_t>(token)];
  if (type == TokenType::kByte) {
    unsigned char byte = 0;
    if (ParseByteToken(raw, &byte)) {
      return std::string(1, static_cast<char>(byte));
    }
  }
  if (type == TokenType::kControl || type == TokenType::kUnknown ||
      type == TokenType::kUnused || type == TokenType::kUndefined) {
    return {};
  }
  if (type == TokenType::kUserDefined) {
    return raw;
  }
  std::string text = raw;
  UnescapeWhitespace(&text);
  return text;
}

std::string SpmTokenizer::Decode(std::span<const std::int32_t> tokens) const {
  std::string out;
  for (const std::int32_t token : tokens) {
    out += TokenText(token);
  }
  if (!out.empty() && out.front() == ' ') {
    out.erase(out.begin());
  }
  return out;
}

}  // namespace superroc::models::llama
