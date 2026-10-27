#ifndef SUPERROC_MODELS_LLAMA_SPM_TOKENIZER_HPP_
#define SUPERROC_MODELS_LLAMA_SPM_TOKENIZER_HPP_

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace superroc::models::llama {

/// SentencePiece tokenizer matching llama.cpp's LLAMA_VOCAB_TYPE_SPM path
/// (`tokenizer.ggml.model = llama`, `tokenizer.ggml.pre = default`).
///
/// Encode: optional special-token partition, dummy prefix space after a
/// special (including the start of the prompt), replace ' ' with U+2581,
/// merge adjacent UTF-8 symbols by highest vocab score (leftmost on ties),
/// byte-fallback unmatched symbols to `<0xNN>`, prepend BOS when add_bos.
class SpmTokenizer {
public:
  enum class TokenType : std::uint32_t {
    kUndefined = 0,
    kNormal = 1,
    kUnknown = 2,
    kControl = 3,
    kUserDefined = 4,
    kUnused = 5,
    kByte = 6,
  };

  struct EncodeOptions {
    bool add_special{true};
    bool parse_special{true};
  };

  [[nodiscard]] static std::optional<SpmTokenizer> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Encode(std::string_view text,
                                                 EncodeOptions options) const;
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::string TokenText(std::int32_t token) const;

  [[nodiscard]] std::int32_t Bos() const noexcept { return bos_; }
  [[nodiscard]] std::int32_t Eos() const noexcept { return eos_; }
  [[nodiscard]] std::int32_t Pad() const noexcept { return pad_; }
  [[nodiscard]] bool AddBos() const noexcept { return add_bos_; }
  [[nodiscard]] bool AddEos() const noexcept { return add_eos_; }
  [[nodiscard]] std::uint32_t Size() const noexcept {
    return static_cast<std::uint32_t>(texts_.size());
  }
  [[nodiscard]] bool IsStop(std::int32_t token) const noexcept {
    return token == eos_;
  }
  [[nodiscard]] TokenType TypeOf(std::int32_t token) const noexcept {
    if (token < 0 || static_cast<std::size_t>(token) >= types_.size()) {
      return TokenType::kUndefined;
    }
    return types_[static_cast<std::size_t>(token)];
  }

private:
  struct Symbol {
    const char* text{nullptr};
    std::size_t n{0};
    int prev{-1};
    int next{-1};
  };
  struct Bigram {
    int left{0};
    int right{0};
    float score{0.0F};
    std::size_t size{0};
  };
  struct BigramGreater {
    bool operator()(const Bigram& a, const Bigram& b) const noexcept {
      if (a.score != b.score) {
        return a.score < b.score;
      }
      return a.left > b.left;
    }
  };

  void PartitionSpecials(std::string_view text, bool parse_special,
                         std::vector<std::int32_t>* out) const;
  void SpmPiece(std::string_view text, std::vector<std::int32_t>* out) const;
  void Resegment(
      const std::vector<Symbol>& symbols,
      const std::unordered_map<std::string, std::pair<int, int>>& rev_merge,
      int index, std::vector<std::int32_t>* out) const;
  [[nodiscard]] std::int32_t TokenId(std::string_view text) const;
  [[nodiscard]] std::int32_t ByteToken(unsigned char byte) const;
  [[nodiscard]] static std::size_t Utf8Len(unsigned char lead);

  std::vector<std::string> texts_;
  std::vector<float> scores_;
  std::vector<TokenType> types_;
  std::unordered_map<std::string, std::int32_t> token_to_id_;
  std::vector<std::int32_t> specials_;
  std::array<std::int32_t, 256> byte_id_{};
  std::int32_t bos_{1};
  std::int32_t eos_{2};
  std::int32_t pad_{0};
  bool add_bos_{true};
  bool add_eos_{false};
};

}  // namespace superroc::models::llama

#endif  // SUPERROC_MODELS_LLAMA_SPM_TOKENIZER_HPP_
