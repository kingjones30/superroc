#include "src/models/ling3/tokenizer.hpp"

#include <unicode/uchar.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <regex>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace superroc::models::ling3 {
namespace {

using core::GgufValueType;

/// The BAILINGMOE pre-tokenizer pattern, verbatim from the fork's
/// llama-vocab.cpp. Note what is deliberately *not* the stock GPT-2 pattern in
/// the same file: `\p{N}` is a single digit (no `{1,3}` grouping),
/// `\s*[\r\n]` has no `+`, the punctuation trailing class has no `/`, and
/// `'ll` folds only as `lL`+`lL`.
constexpr std::string_view kPreTokenPattern =
    R"('(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+)";

/// The fork does not run this pattern over the raw UTF-8 bytes. llama.cpp's
/// `unicode_regex_split` collapses the text first (`unicode.cpp`): codepoints
/// below 128 keep their own byte, every other codepoint becomes the single
/// byte its Unicode category stands for, and `\p{X}` in the pattern becomes
/// that byte followed by the ASCII run of the same class. Matching the
/// collapsed string is what keeps "café" and "世界" in one pre-token each;
/// matching the raw bytes splits both, because every byte of a multi-byte
/// character falls into `[^\s\p{L}\p{N}]` on its own.
///
/// The two classes this pattern names, with the byte the fork assigns them:
///   \p{L} -> '\xD2' + A-Za-z      \p{N} -> '\xD1' + 0-9
/// and every non-ASCII whitespace becomes '\x0B' so `\s` still sees it.
constexpr char kCptNumber = '\xD1';
constexpr char kCptLetter = '\xD2';
constexpr char kCptWhitespace = '\x0B';

/// True for the Unicode classes the fork's table calls a letter / a number /
/// whitespace -- the same predicates the other superroc tokenizers take from ICU
/// (see models/qwen/tokenizer.cpp).
[[nodiscard]] bool IsUnicodeLetter(std::uint32_t cpt) noexcept {
  return (U_GET_GC_MASK(static_cast<UChar32>(cpt)) & U_GC_L_MASK) != 0;
}

[[nodiscard]] bool IsUnicodeNumber(std::uint32_t cpt) noexcept {
  const auto category =
      static_cast<UCharCategory>(u_charType(static_cast<UChar32>(cpt)));
  return category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER ||
         category == U_OTHER_NUMBER;
}

[[nodiscard]] bool IsUnicodeWhitespace(std::uint32_t cpt) noexcept {
  return u_isUWhiteSpace(static_cast<UChar32>(cpt)) != 0;
}

/// UTF-8 -> codepoints with the fork's `unicode_cpts_from_utf8` semantics: a
/// malformed byte becomes U+FFFD and is skipped one byte at a time, so nothing
/// is dropped and nothing throws out of the tokenizer.
[[nodiscard]] std::vector<std::uint32_t> Utf8ToCodepoints(std::string_view s) {
  std::vector<std::uint32_t> out;
  out.reserve(s.size());
  std::size_t offset = 0;
  while (offset < s.size()) {
    const auto lead = static_cast<unsigned char>(s[offset]);
    std::size_t len = 1;
    if ((lead & 0x80) == 0) {
      len = 1;
    } else if ((lead & 0xE0) == 0xC0) {
      len = 2;
    } else if ((lead & 0xF0) == 0xE0) {
      len = 3;
    } else if ((lead & 0xF8) == 0xF0) {
      len = 4;
    } else {
      len = 0;  // continuation byte or 5/6-byte lead: not a codepoint start
    }
    bool ok = len > 0 && offset + len <= s.size();
    for (std::size_t k = 1; ok && k < len; ++k) {
      if ((static_cast<unsigned char>(s[offset + k]) & 0xC0) != 0x80) {
        ok = false;
      }
    }
    if (len == 4 && ok) {
      // reject the encodings the fork rejects: overlong, surrogate, > U+10FFFF
      const std::uint32_t v =
          static_cast<std::uint32_t>((lead & 0x07U) << 18) |
          ((static_cast<unsigned char>(s[offset + 1]) & 0x3FU) << 12) |
          ((static_cast<unsigned char>(s[offset + 2]) & 0x3FU) << 6) |
          (static_cast<unsigned char>(s[offset + 3]) & 0x3FU);
      if (v > 0x10FFFFU || (v >= 0xD800U && v <= 0xDFFFU)) {
        ok = false;
      }
    }
    if (!ok) {
      out.push_back(0xFFFD);
      offset += 1;
      continue;
    }
    std::uint32_t v = 0;
    if (len == 1) {
      v = lead;
    } else if (len == 2) {
      v = static_cast<std::uint32_t>((lead & 0x1FU) << 6) |
          (static_cast<unsigned char>(s[offset + 1]) & 0x3FU);
    } else if (len == 3) {
      v = static_cast<std::uint32_t>((lead & 0x0FU) << 12) |
          ((static_cast<unsigned char>(s[offset + 1]) & 0x3FU) << 6) |
          (static_cast<unsigned char>(s[offset + 2]) & 0x3FU);
    } else {
      v = static_cast<std::uint32_t>((lead & 0x07U) << 18) |
          ((static_cast<unsigned char>(s[offset + 1]) & 0x3FU) << 12) |
          ((static_cast<unsigned char>(s[offset + 2]) & 0x3FU) << 6) |
          (static_cast<unsigned char>(s[offset + 3]) & 0x3FU);
    }
    out.push_back(v);
    offset += len;
  }
  return out;
}

[[nodiscard]] std::string CptToUtf8(std::uint32_t cpt) {
  std::string s;
  if (cpt < 0x80) {
    s.push_back(static_cast<char>(cpt));
  } else if (cpt < 0x800) {
    s.push_back(static_cast<char>(0xC0 | (cpt >> 6)));
    s.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  } else if (cpt < 0x10000) {
    s.push_back(static_cast<char>(0xE0 | (cpt >> 12)));
    s.push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  } else {
    s.push_back(static_cast<char>(0xF0 | (cpt >> 18)));
    s.push_back(static_cast<char>(0x80 | ((cpt >> 12) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  }
  return s;
}

/// One collapsed byte per codepoint, exactly as the fork builds it.
[[nodiscard]] std::string CollapseCodepoints(
    const std::vector<std::uint32_t>& cpts) {
  std::string out;
  out.reserve(cpts.size());
  for (const std::uint32_t cpt : cpts) {
    if (cpt < 128) {
      out.push_back(static_cast<char>(cpt));
    } else if (IsUnicodeWhitespace(cpt)) {
      out.push_back(kCptWhitespace);
    } else if (IsUnicodeLetter(cpt)) {
      out.push_back(kCptLetter);
    } else if (IsUnicodeNumber(cpt)) {
      out.push_back(kCptNumber);
    } else {
      // punctuation, marks, symbols, separators, controls and unassigned all
      // land here; the pattern distinguishes none of them.
      out.push_back(static_cast<char>(0xD0));
    }
  }
  return out;
}

/// Rewrites `\p{L}` / `\p{N}` the way the fork does before compiling: the
/// category byte plus that class's ASCII run, wrapped in `[]` only when the
/// substitution is not already inside a character class.
[[nodiscard]] std::string CollapsePattern(std::string_view pattern) {
  std::string out;
  out.reserve(pattern.size() + 16);
  bool inside = false;
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    if (pattern[i] == '[' && (i == 0 || pattern[i - 1] != '\\')) {
      out += '[';
      inside = true;
      continue;
    }
    if (inside && pattern[i] == ']' && pattern[i - 1] != '\\') {
      out += ']';
      inside = false;
      continue;
    }
    if (pattern[i] == '\\' && i + 3 < pattern.size() && pattern[i + 1] == 'p' &&
        pattern[i + 2] == '{') {
      const std::size_t close = pattern.find('}', i + 3);
      if (close != std::string_view::npos && close <= i + 10) {
        const std::string_view cat = pattern.substr(i, close - i + 1);
        const bool is_letter = cat == "\\p{L}";
        const bool is_number = cat == "\\p{N}";
        if (is_letter || is_number) {
          if (!inside) {
            out += '[';
          }
          out += is_letter ? kCptLetter : kCptNumber;
          out += is_letter ? "A-Za-z" : "0-9";
          if (!inside) {
            out += ']';
          }
          i = close;
          continue;
        }
      }
    }
    out += pattern[i];
  }
  return out;
}

const std::regex& PreTokenRegex() {
  static const std::regex re(CollapsePattern(kPreTokenPattern),
                             std::regex_constants::ECMAScript |
                                 std::regex_constants::optimize |
                                 std::regex_constants::nosubs);
  return re;
}

/// Strings the fork force-classifies as end-of-turn (llama-vocab.cpp EOT/EOG
/// lists). `<|role_end|>` is *not* in them, which is why this model's EOS comes
/// from `tokenizer.ggml.eos_token_id` instead.
constexpr std::string_view kForkEotStrings[] = {
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|flush|>",
    "<|calls|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "<|eom_id|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]",
    "[EOS]",
    "<|end_of_text|>",
    "<end_of_utterance>",
    "<eos>",
    "<turn>",
    "<|tool_response|>",
    "<|｜end▁of▁sentence｜>",
};

bool IsForkEot(std::string_view s) {
  for (const auto& e : kForkEotStrings) {
    if (s == e) {
      return true;
    }
  }
  return false;
}

std::vector<std::string> ReadStringArray(const core::GgufReader& gguf,
                                         std::string_view key,
                                         std::string* error_msg, bool* found) {
  std::vector<std::string> out;
  const auto* meta = gguf.FindMetadata(key);
  *found = meta != nullptr;
  if (meta == nullptr) {
    return out;
  }
  if (const auto* v =
          std::get_if<std::vector<std::string_view>>(&meta->value)) {
    out.reserve(v->size());
    for (auto s : *v) {
      out.emplace_back(s);
    }
    return out;
  }
  if (error_msg != nullptr) {
    *error_msg = std::string(key) + " must be an array of strings";
  }
  return out;
}

}  // namespace

const std::unordered_map<std::uint8_t, std::string>&
LingTokenizer::ByteToUnicode() {
  // The canonical GPT-2 table: printable ASCII and the Latin-1 range map to
  // themselves; the remaining 68 bytes map to U+0100 upwards in byte order.
  static const std::unordered_map<std::uint8_t, std::string> map = [] {
    std::unordered_map<std::uint8_t, std::string> m;
    auto utf8 = [](std::uint32_t c) {
      std::string s;
      if (c < 0x80) {
        s.push_back(static_cast<char>(c));
      } else if (c < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (c >> 6)));
        s.push_back(static_cast<char>(0x80 | (c & 0x3F)));
      } else {
        s.push_back(static_cast<char>(0xE0 | (c >> 12)));
        s.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (c & 0x3F)));
      }
      return s;
    };
    for (int ch = 0x21; ch <= 0x7E; ++ch) {
      m[static_cast<std::uint8_t>(ch)] = utf8(static_cast<std::uint32_t>(ch));
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {
      m[static_cast<std::uint8_t>(ch)] = utf8(static_cast<std::uint32_t>(ch));
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {
      m[static_cast<std::uint8_t>(ch)] = utf8(static_cast<std::uint32_t>(ch));
    }
    int n = 0;
    for (int ch = 0; ch < 256; ++ch) {
      if (m.find(static_cast<std::uint8_t>(ch)) == m.end()) {
        m[static_cast<std::uint8_t>(ch)] =
            utf8(static_cast<std::uint32_t>(256 + n));
        ++n;
      }
    }
    return m;
  }();
  return map;
}

std::vector<std::string> LingTokenizer::PreTokenize(std::string_view text) {
  // Match on the collapsed string, then rebuild each piece from the
  // codepoints the fork would have used, so the pieces are the same bytes
  // llama.cpp hands to its BPE step.
  const std::vector<std::uint32_t> cpts = Utf8ToCodepoints(text);
  const std::string collapsed = CollapseCodepoints(cpts);

  std::vector<std::string> pieces;
  auto join = [&cpts](std::size_t start, std::size_t count) {
    std::string out;
    for (std::size_t i = start; i < start + count && i < cpts.size(); ++i) {
      out += CptToUtf8(cpts[i]);
    }
    return out;
  };

  auto begin =
      std::sregex_iterator(collapsed.begin(), collapsed.end(), PreTokenRegex());
  auto end = std::sregex_iterator();
  std::size_t cursor = 0;
  for (auto it = begin; it != end; ++it) {
    // Offsets are codepoint counts: one collapsed byte per codepoint.
    const std::size_t pos = static_cast<std::size_t>(it->position(0));
    const std::size_t len = static_cast<std::size_t>(it->length(0));
    // The pattern is exhaustive over letters, digits, whitespace and
    // punctuation, so a gap should be unreachable; emit it rather than drop it
    // if it ever happens, so no input is silently lost.
    if (pos > cursor) {
      pieces.emplace_back(join(cursor, pos - cursor));
    }
    pieces.emplace_back(join(pos, len));
    cursor = pos + len;
  }
  if (cursor < cpts.size()) {
    pieces.emplace_back(join(cursor, cpts.size() - cursor));
  }
  return pieces;
}

std::string LingTokenizer::ByteEncode(std::string_view piece) {
  const auto& map = ByteToUnicode();
  std::string out;
  out.reserve(piece.size() * 2);
  for (const char c : piece) {
    const auto it = map.find(static_cast<std::uint8_t>(c));
    if (it != map.end()) {
      out += it->second;
    }
  }
  return out;
}

std::optional<LingTokenizer> LingTokenizer::FromGguf(
    const core::GgufReader& gguf, std::string* error_msg) {
  if (gguf.GetMetadataString("tokenizer.ggml.model") != "gpt2") {
    if (error_msg != nullptr) {
      *error_msg = "tokenizer.ggml.model must be gpt2";
    }
    return std::nullopt;
  }
  const std::string pre(gguf.GetMetadataString("tokenizer.ggml.pre")
                            .value_or(std::string_view()));
  if (pre != "bailingmoe2" && pre != "bailingmoe" && pre != "llada-moe") {
    if (error_msg != nullptr) {
      *error_msg = "unsupported tokenizer.ggml.pre: '" + pre +
                   "' (this package implements bailingmoe2)";
    }
    return std::nullopt;
  }

  bool has_tokens = false;
  bool has_merges = false;
  std::vector<std::string> texts =
      ReadStringArray(gguf, "tokenizer.ggml.tokens", error_msg, &has_tokens);
  if (!has_tokens) {
    if (error_msg != nullptr) {
      *error_msg = "cannot find tokenizer vocab (tokenizer.ggml.tokens)";
    }
    return std::nullopt;
  }
  std::vector<std::string> merges =
      ReadStringArray(gguf, "tokenizer.ggml.merges", error_msg, &has_merges);
  if (!has_merges) {
    if (error_msg != nullptr) {
      *error_msg = "cannot find tokenizer merges (tokenizer.ggml.merges)";
    }
    return std::nullopt;
  }

  LingTokenizer t;
  t.texts_ = std::move(texts);
  t.token_to_id_.reserve(t.texts_.size() * 2);
  for (std::size_t i = 0; i < t.texts_.size(); ++i) {
    t.token_to_id_.emplace(t.texts_[i], static_cast<std::int32_t>(i));
  }

  // Merge rank is the index in the list; the fork never consults
  // tokenizer.ggml.scores for BPE, and this file carries no scores key.
  t.merge_ranks_.reserve(merges.size() * 2);
  for (std::size_t i = 0; i < merges.size(); ++i) {
    // Split on the first space at or after index 1, so a one-character left
    // side works.
    const std::string& m = merges[i];
    const std::size_t pos = m.find(' ', 1);
    if (pos == std::string::npos) {
      continue;
    }
    t.merge_ranks_.emplace(Bigram{m.substr(0, pos), m.substr(pos + 1)},
                           static_cast<std::int32_t>(i));
  }

  const std::size_t n = t.texts_.size();
  t.control_.assign(n, 0);
  t.eog_.assign(n, 0);
  bool has_types = false;
  const auto* type_meta = gguf.FindMetadata("tokenizer.ggml.token_type");
  if (type_meta != nullptr) {
    std::vector<std::int64_t> types;
    if (const auto* v =
            std::get_if<std::vector<std::int64_t>>(&type_meta->value)) {
      types.assign(v->begin(), v->end());
    } else if (const auto* v =
                   std::get_if<std::vector<std::uint64_t>>(&type_meta->value)) {
      for (auto u : *v) {
        types.push_back(static_cast<std::int64_t>(u));
      }
    }
    if (types.size() < n) {
      if (error_msg != nullptr) {
        *error_msg = "tokenizer.ggml.token_type is shorter than the vocabulary";
      }
      return std::nullopt;
    }
    has_types = true;
    for (std::size_t i = 0; i < n; ++i) {
      const auto tt = static_cast<TokenType>(types[i]);
      if (tt == TokenType::kControl || tt == TokenType::kUserDefined ||
          tt == TokenType::kUnknown) {
        t.special_cache_.push_back(static_cast<std::int32_t>(i));
      }
      if (tt == TokenType::kControl) {
        t.control_[i] = 1;
      }
    }
  }

  // The fork's forced EOT/EOG overrides, applied by string, unconditionally.
  for (std::size_t i = 0; i < n; ++i) {
    if (IsForkEot(t.texts_[i])) {
      t.eog_[i] = 1;
      t.control_[i] = 1;
      if (!t.control_[i]) {
        t.control_[i] = 1;
      }
    }
  }

  // Longest text first so a longer overlapping marker wins the partition.
  std::sort(t.special_cache_.begin(), t.special_cache_.end(),
            [&t](std::int32_t a, std::int32_t b) {
              return t.texts_[a].size() > t.texts_[b].size();
            });
  (void)has_types;

  const auto id_or = [&](std::string_view key, std::int32_t fallback) {
    const auto v = gguf.GetMetadataUint32(key);
    if (!v.has_value() || *v >= n) {
      return fallback;
    }
    return static_cast<std::int32_t>(*v);
  };
  t.bos_ = id_or("tokenizer.ggml.bos_token_id", -1);
  t.eos_ = id_or("tokenizer.ggml.eos_token_id", -1);
  t.pad_ = id_or("tokenizer.ggml.padding_token_id", -1);
  if (t.eos_ >= 0) {
    t.eog_[static_cast<std::size_t>(t.eos_)] = 1;
  }
  return t;
}

std::vector<std::int32_t> LingTokenizer::BpePiece(
    const std::string& encoded) const {
  // Initial symbols are UTF-8 codepoints, which on byte-encoded text means one
  // mapped byte each.
  std::vector<std::string> symbols;
  for (std::size_t offset = 0; offset < encoded.size();) {
    const auto lead = static_cast<unsigned char>(encoded[offset]);
    std::size_t len = 1;
    if ((lead & 0xE0) == 0xC0) {
      len = 2;
    } else if ((lead & 0xF0) == 0xE0) {
      len = 3;
    } else if ((lead & 0xF8) == 0xF0) {
      len = 4;
    }
    len = std::min(len, encoded.size() - offset);
    symbols.emplace_back(encoded.substr(offset, len));
    offset += len;
  }

  // Merge the lowest-rank adjacent pair repeatedly. Ranks are unique per merge
  // string, so a tie cannot change the outcome.
  while (symbols.size() > 1) {
    std::size_t best = symbols.size();
    std::int32_t best_rank = 0;
    for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto it = merge_ranks_.find(Bigram{symbols[i], symbols[i + 1]});
      if (it == merge_ranks_.end()) {
        continue;
      }
      if (best == symbols.size() || it->second < best_rank) {
        best = i;
        best_rank = it->second;
      }
    }
    if (best == symbols.size()) {
      break;
    }
    symbols[best] += symbols[best + 1];
    symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best) + 1);
  }

  std::vector<std::int32_t> out;
  out.reserve(symbols.size());
  for (const auto& sym : symbols) {
    if (const auto it = token_to_id_.find(sym); it != token_to_id_.end()) {
      out.push_back(it->second);
      continue;
    }
    // Byte fallback, exactly as the fork does when byte_encode is on: emit one
    // single-byte token per byte so nothing is silently dropped.
    for (std::size_t i = 0; i < sym.size(); ++i) {
      const std::string byte(1, sym[i]);
      if (const auto bit = token_to_id_.find(byte); bit != token_to_id_.end()) {
        out.push_back(bit->second);
      }
    }
  }
  return out;
}

std::vector<std::int32_t> LingTokenizer::Encode(std::string_view text,
                                                bool add_bos,
                                                bool parse_special) const {
  std::vector<std::int32_t> out;
  if (add_bos && bos_ >= 0) {
    out.push_back(bos_);
  }

  // Partition: walk the special cache longest-first, emitting verbatim matches
  // and running everything else through the pre-tokenizer. CONTROL and UNKNOWN
  // markers are skipped when parse_special is false; USER_DEFINED ones are
  // still matched, matching the fork's asymmetry.
  std::string buffer;
  auto flush_buffer = [&] {
    if (buffer.empty()) {
      return;
    }
    for (const auto& piece : PreTokenize(buffer)) {
      const std::string encoded = ByteEncode(piece);
      const auto ids = BpePiece(encoded);
      out.insert(out.end(), ids.begin(), ids.end());
    }
    buffer.clear();
  };

  std::size_t cursor = 0;
  while (cursor < text.size()) {
    std::size_t best_pos = std::string::npos;
    std::int32_t best_id = -1;
    for (const std::int32_t id : special_cache_) {
      const std::string& tok = texts_[static_cast<std::size_t>(id)];
      if (tok.empty()) {
        continue;
      }
      const auto type_control = control_[static_cast<std::size_t>(id)] != 0;
      if (!parse_special && type_control) {
        continue;
      }
      const std::size_t pos = text.find(tok, cursor);
      if (pos == std::string_view::npos) {
        continue;
      }
      if (best_pos == std::string::npos || pos < best_pos ||
          (pos == best_pos &&
           tok.size() > texts_[static_cast<std::size_t>(best_id)].size())) {
        best_pos = pos;
        best_id = id;
      }
    }
    if (best_pos == std::string::npos) {
      buffer.append(text.substr(cursor));
      cursor = text.size();
      break;
    }
    buffer.append(text.substr(cursor, best_pos - cursor));
    flush_buffer();
    out.push_back(best_id);
    cursor = best_pos + texts_[static_cast<std::size_t>(best_id)].size();
  }
  flush_buffer();
  return out;
}

std::string LingTokenizer::TokenText(std::int32_t token) const {
  if (token < 0 || static_cast<std::size_t>(token) >= texts_.size()) {
    return {};
  }
  const std::string& t = texts_[static_cast<std::size_t>(token)];
  // Undo the byte table: each mapped codepoint becomes its original byte.
  static const std::unordered_map<std::string, std::uint8_t> inverse = [] {
    std::unordered_map<std::string, std::uint8_t> m;
    for (const auto& [k, v] : ByteToUnicode()) {
      m[v] = k;
    }
    return m;
  }();
  std::string out;
  for (std::size_t i = 0; i < t.size();) {
    const auto lead = static_cast<unsigned char>(t[i]);
    std::size_t len = 1;
    if ((lead & 0xE0) == 0xC0) {
      len = 2;
    } else if ((lead & 0xF0) == 0xE0) {
      len = 3;
    } else if ((lead & 0xF8) == 0xF0) {
      len = 4;
    }
    len = std::min(len, t.size() - i);
    const auto it = inverse.find(t.substr(i, len));
    if (it != inverse.end()) {
      out.push_back(static_cast<char>(it->second));
    } else {
      out.append(t, i, len);
    }
    i += len;
  }
  return out;
}

std::string LingTokenizer::Decode(std::span<const std::int32_t> tokens) const {
  std::string out;
  for (const std::int32_t t : tokens) {
    out += TokenText(t);
  }
  return out;
}

}  // namespace superroc::models::ling3
