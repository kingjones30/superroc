// Unit tests for the OpenAI logprobs JSON shapes: a token's text can be part
// of a multi-byte character, and the response must still be valid UTF-8.
// CPU only. Plain main, PASS/FAIL lines, nonzero exit on any failure.

#include "src/cli/serve/logprobs_json.hpp"

#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/sampling.hpp"
#include "src/core/utf8.hpp"

namespace {

using superroc::sampling::CapturedTokenLogprobs;

int failures = 0;

void Check(bool condition, const std::string& message) {
  std::cout << (condition ? "PASS " : "FAIL ") << message << "\n";
  if (!condition) {
    ++failures;
  }
}

bool ValidUtf8(const std::string& text) {
  return superroc::core::Utf8Decoder{}.Push(text, true) == text;
}

CapturedTokenLogprobs Entry(std::string text, double logprob) {
  CapturedTokenLogprobs entry;
  entry.token = 1;
  entry.logprob = logprob;
  entry.text = text;
  entry.top.push_back({1, logprob, std::move(text)});
  return entry;
}

}  // namespace

int main() {
  // "\xE4\xB8" is the first two bytes of a three-byte character; "\xAD" is
  // the lone continuation byte that completes it in the next token.
  const std::vector<CapturedTokenLogprobs> entries = {
      Entry("Hello", -0.25), Entry("\xE4\xB8", -1.5), Entry("\xAD", -0.5)};

  const auto chat = superroc::server::ChatContentLogprobsJson(entries);
  const std::string chat_text = chat.dump();
  Check(ValidUtf8(chat_text), "chat logprobs are valid UTF-8");
  const auto reparsed = superroc::json::parse(chat_text);
  const auto& content = reparsed.find("content")->items();
  Check(content.size() == 3, "chat keeps one entry per token");
  Check(content[0].find("token")->str() == "Hello",
        "a complete token keeps its text");
  Check(content[1].find("token")->str() == "\xEF\xBF\xBD",
        "a partial character becomes U+FFFD in `token`");
  const auto& bytes = content[1].find("bytes")->items();
  Check(bytes.size() == 2 && bytes[0].as_double() == 0xE4 &&
            bytes[1].as_double() == 0xB8,
        "`bytes` carries the exact bytes");
  Check(content[1].find("top_logprobs")->items()[0].find("token")->str() ==
            "\xEF\xBF\xBD",
        "top_logprobs tokens are sanitized too");

  const auto completion = superroc::server::CompletionLogprobsJson(entries);
  const std::string completion_text = completion.dump();
  Check(ValidUtf8(completion_text), "completion logprobs are valid UTF-8");
  const auto legacy = superroc::json::parse(completion_text);
  const auto& tokens = legacy.find("tokens")->items();
  Check(tokens.size() == 3 && tokens[0].str() == "Hello" &&
            tokens[1].str() == "bytes:\\xe4\\xb8" &&
            tokens[2].str() == "bytes:\\xad",
        "completion tokens use the bytes: form for partial characters");
  const auto& offsets = legacy.find("text_offset")->items();
  Check(offsets.size() == 3 && offsets[0].as_double() == 0 &&
            offsets[1].as_double() == 5 && offsets[2].as_double() == 7,
        "text offsets count the real bytes");
  const auto& top = legacy.find("top_logprobs")->items();
  Check(top[1].find("bytes:\\xe4\\xb8") != nullptr &&
            top[2].find("bytes:\\xad") != nullptr,
        "distinct partial tokens keep distinct top_logprobs keys");

  // A value the target never exposed is not a number: it must reach the
  // client as JSON null, never as `nan` or `inf` (invalid JSON) or as 0.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<CapturedTokenLogprobs> unknown = {
      Entry("a", nan), Entry("b", -inf), Entry("c", inf)};
  const std::string chat_unknown =
      superroc::server::ChatContentLogprobsJson(unknown).dump();
  const auto chat_items =
      superroc::json::parse(chat_unknown).find("content")->items();
  const std::string completion_unknown =
      superroc::server::CompletionLogprobsJson(unknown).dump();
  const auto completion_items =
      superroc::json::parse(completion_unknown).find("token_logprobs")->items();
  bool all_null = chat_items.size() == 3 && completion_items.size() == 3;
  for (std::size_t i = 0; all_null && i < 3; ++i) {
    all_null = chat_items[i].find("logprob")->is_null() &&
               chat_items[i]
                   .find("top_logprobs")
                   ->items()[0]
                   .find("logprob")
                   ->is_null() &&
               completion_items[i].is_null();
  }
  Check(all_null, "non-finite log probabilities are JSON null");

  if (failures != 0) {
    std::cout << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "logprobs JSON tests passed\n";
  return 0;
}
