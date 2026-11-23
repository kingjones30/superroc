#ifndef SUPERROC_SERVER_LOGPROBS_JSON_HPP_
#define SUPERROC_SERVER_LOGPROBS_JSON_HPP_

#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/sampling.hpp"
#include "src/core/utf8.hpp"

namespace superroc::server {

// OpenAI logprobs response shapes, built from the per-emitted-token reports
// the runners capture. A non-finite logprob (a position the target never
// exposed on the host) renders as null so the arrays stay parallel.

/// A token's text as a JSON string. A token can be part of a multi-byte
/// character, so its bytes are not always valid UTF-8 on their own: malformed
/// and incomplete sequences become U+FFFD here, and `bytes` (chat) carries the
/// exact bytes.
inline std::string LogprobTokenText(std::string_view text) {
  return core::Utf8Decoder{}.Push(text, true);
}

/// The Completions dialect has no `bytes` field, so a token that is not valid
/// UTF-8 is written the way the OpenAI API writes it: `bytes:\xNN...`.
inline std::string CompletionTokenText(std::string_view text) {
  if (core::Utf8Decoder{}.Push(text, true) == text) {
    return std::string(text);
  }
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out = "bytes:";
  for (const unsigned char byte : text) {
    out += "\\x";
    out += kHex[byte >> 4];
    out += kHex[byte & 0x0F];
  }
  return out;
}

inline json::Value LogprobBytesJson(std::string_view text) {
  json::Value bytes = json::Value::array();
  for (const unsigned char byte : text) {
    bytes.push_back(static_cast<long long>(byte));
  }
  return bytes;
}

inline json::Value LogprobValueJson(double logprob) {
  if (!std::isfinite(logprob)) {
    return json::Value();
  }
  return json::Value(logprob);
}

inline json::Value TopLogprobsJson(
    const std::vector<sampling::CapturedTokenLogprobs::Entry>& top) {
  json::Value out = json::Value::array();
  for (const auto& entry : top) {
    json::Value item = json::Value::object();
    item["token"] = LogprobTokenText(entry.text);
    item["bytes"] = LogprobBytesJson(entry.text);
    item["logprob"] = LogprobValueJson(entry.logprob);
    out.push_back(std::move(item));
  }
  return out;
}

/// Chat Completions `choice.logprobs`: one entry per emitted token in order.
/// Reasoning tokens and tokens consumed by tool framing stay listed, so the
/// list covers every generated token and can be longer than the message's
/// content; without thinking or tools it matches the content exactly.
inline json::Value ChatContentLogprobsJson(
    const std::vector<sampling::CapturedTokenLogprobs>& entries) {
  json::Value content = json::Value::array();
  for (const auto& entry : entries) {
    json::Value item = json::Value::object();
    item["token"] = LogprobTokenText(entry.text);
    item["bytes"] = LogprobBytesJson(entry.text);
    item["logprob"] = LogprobValueJson(entry.logprob);
    item["top_logprobs"] = TopLogprobsJson(entry.top);
    content.push_back(std::move(item));
  }
  json::Value out = json::Value::object();
  out["content"] = std::move(content);
  return out;
}

/// Completions `choices[].logprobs` for one streaming chunk or the whole
/// completion: single-element arrays per token, with the byte offset of each
/// token's text from `base_offset`.
inline json::Value CompletionLogprobsJson(
    const std::vector<sampling::CapturedTokenLogprobs>& entries,
    std::size_t base_offset = 0) {
  json::Value tokens = json::Value::array();
  json::Value token_logprobs = json::Value::array();
  json::Value top_logprobs = json::Value::array();
  json::Value text_offset = json::Value::array();
  std::size_t offset = base_offset;
  for (const auto& entry : entries) {
    tokens.push_back(CompletionTokenText(entry.text));
    token_logprobs.push_back(LogprobValueJson(entry.logprob));
    json::Value top = json::Value::object();
    for (const auto& candidate : entry.top) {
      top[CompletionTokenText(candidate.text)] =
          LogprobValueJson(candidate.logprob);
    }
    top_logprobs.push_back(std::move(top));
    text_offset.push_back(static_cast<long long>(offset));
    offset += entry.text.size();
  }
  json::Value out = json::Value::object();
  out["tokens"] = std::move(tokens);
  out["token_logprobs"] = std::move(token_logprobs);
  out["top_logprobs"] = std::move(top_logprobs);
  out["text_offset"] = std::move(text_offset);
  return out;
}

}  // namespace superroc::server

#endif  // SUPERROC_SERVER_LOGPROBS_JSON_HPP_
