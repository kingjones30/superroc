#ifndef GUFO_MODELS_LING3_TOKENIZER_HPP_
#define GUFO_MODELS_LING3_TOKENIZER_HPP_

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::ling3 {

/// GPT-2 style byte-level BPE as the fork configures it for Ling-3.0
/// (`tokenizer.ggml.model = gpt2`, `tokenizer.ggml.pre = bailingmoe2`).
///
/// `bailingmoe2` is not its own pre-type: the fork folds it into
/// `LLAMA_VOCAB_PRE_TYPE_BAILINGMOE`, which sets only `clean_spaces = false`.
/// Byte encoding stays on, `ignore_merges` stays off, and merge priority is the
/// **index in `tokenizer.ggml.merges`**, not `tokenizer.ggml.scores` (this file
/// carries no scores key at all).
///
/// The pre-tokenizer pattern is the one in the fork's `llama-vocab.cpp` for
/// BAILINGMOE. It has no fast-path handler there, so `\p{L}` and `\p{N}` are
/// served by the fork's ASCII fallback (`A-Za-z` and `0-9`). This
/// implementation uses the same substitution so token ids agree; on non-ASCII
/// input a full-Unicode engine would diverge from the fork, and so would this.
class LingTokenizer {
public:
  /// GGUF `tokenizer.ggml.token_type` values (include/llama.h).
  enum class TokenType : std::int32_t {
    kUndefined = 0,
    kNormal = 1,
    kUnknown = 2,
    kControl = 3,
    kUserDefined = 4,
    kUnused = 5,
    kByte = 6,
  };

  /// Reads and validates the vocabulary. Fails closed when the artifact is not
  /// a bailingmoe2 GPT-2 BPE, when `tokens` is missing, or when `merges` is
  /// missing (the fork throws for both).
  [[nodiscard]] static std::optional<LingTokenizer> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Encodes UTF-8 `text`. No BOS is prepended unless `add_bos` is set: this
  /// file carries `add_bos_token = false`, matching the fork, so the default is
  /// false. With `parse_special` false, CONTROL and UNKNOWN markers are
  /// byte-encoded as ordinary text, while USER_DEFINED markers are still
  /// matched verbatim (the fork's asymmetry).
  [[nodiscard]] std::vector<std::int32_t> Encode(
      std::string_view text, bool add_bos = false,
      bool parse_special = true) const;

  /// Renders one token's text, undoing the byte-level mapping.
  [[nodiscard]] std::string TokenText(std::int32_t token) const;
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> tokens) const;

  [[nodiscard]] std::int32_t Bos() const noexcept { return bos_; }
  [[nodiscard]] std::int32_t Eos() const noexcept { return eos_; }
  [[nodiscard]] std::int32_t Pad() const noexcept { return pad_; }
  [[nodiscard]] std::uint32_t Size() const noexcept {
    return static_cast<std::uint32_t>(texts_.size());
  }
  /// The metadata EOS, plus anything the fork would classify as end-of-turn.
  [[nodiscard]] bool IsStop(std::int32_t token) const noexcept {
    return token == eos_ ||
           (token >= 0 && eog_[static_cast<std::size_t>(token)]);
  }
  [[nodiscard]] bool IsControl(std::int32_t token) const noexcept {
    return token >= 0 && control_[static_cast<std::size_t>(token)];
  }

private:
  struct Bigram {
    std::string left;
    std::string right;
    bool operator==(const Bigram& o) const {
      return left == o.left && right == o.right;
    }
  };
  struct BigramHash {
    std::size_t operator()(const Bigram& b) const noexcept {
      return std::hash<std::string>{}(b.left) ^
             (std::hash<std::string>{}(b.right) * 1099511628211ULL);
    }
  };

  /// Splits `text` into pre-tokens with the BAILINGMOE pattern.
  [[nodiscard]] static std::vector<std::string> PreTokenize(
      std::string_view text);
  /// Maps each raw UTF-8 byte of `piece` through the GPT-2 byte table.
  [[nodiscard]] static std::string ByteEncode(std::string_view piece);
  /// Byte table lookup, built once per process.
  [[nodiscard]] static const std::unordered_map<std::uint8_t, std::string>&
  ByteToUnicode();
  /// Applies the merge loop, then the per-byte fallback.
  [[nodiscard]] std::vector<std::int32_t> BpePiece(
      const std::string& encoded) const;

  std::vector<std::string> texts_;
  std::unordered_map<std::string, std::int32_t> token_to_id_;
  std::unordered_map<Bigram, std::int32_t, BigramHash> merge_ranks_;
  /// CONTROL / USER_DEFINED / UNKNOWN ids, longest text first, so a literal
  /// marker in the input wins over a shorter overlapping one.
  std::vector<std::int32_t> special_cache_;
  std::vector<std::uint8_t> control_;
  std::vector<std::uint8_t> eog_;
  std::int32_t bos_{0};
  std::int32_t eos_{-1};
  std::int32_t pad_{-1};
};

}  // namespace gufo::models::ling3

#endif  // GUFO_MODELS_LING3_TOKENIZER_HPP_
