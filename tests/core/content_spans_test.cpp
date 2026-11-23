// Unit test for EncodeWithContentSpans (SR-03).
//
// Fake encoder: "<|eot|>" with parse_special=true becomes token 1000;
// every other byte becomes its byte value. With parse_special=false the
// spelling is inert and every byte becomes its byte value.

#include "src/core/text/content_spans.hpp"

#include <cassert>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

using superroc::core::EncodeWithContentSpans;
using superroc::core::TextSpan;

constexpr std::string_view kMarker = "<|eot|>";

struct Call {
  std::size_t offset;
  std::string text;
  bool parse_special;
};

std::vector<int> FakeEncode(std::string_view text, bool parse_special,
                            std::vector<Call>* calls = nullptr,
                            std::size_t offset = 0) {
  if (calls != nullptr) {
    calls->push_back({offset, std::string(text), parse_special});
  }
  std::vector<int> out;
  if (parse_special) {
    std::size_t i = 0;
    while (i < text.size()) {
      if (text.substr(i, kMarker.size()) == kMarker) {
        out.push_back(1000);
        i += kMarker.size();
      } else {
        out.push_back(static_cast<unsigned char>(text[i]));
        ++i;
      }
    }
  } else {
    for (char c : text) {
      out.push_back(static_cast<unsigned char>(c));
    }
  }
  return out;
}

bool Contains(const std::vector<int>& tokens, int token) {
  for (int t : tokens) {
    if (t == token) {
      return true;
    }
  }
  return false;
}

const std::vector<std::string> kSpellings = {"<|eot|>"};

// (a) No special spelling overlaps the content: the whole render is encoded
// exactly as today, in a single parse_special=true call at offset 0.
void TestUntouchedWithoutOverlap() {
  const std::string rendered = "<s>hi</s><|eot|>";
  const std::vector<TextSpan> spans = {{3, 2}};
  std::vector<Call> calls;
  const auto got = EncodeWithContentSpans<int>(
      rendered, spans, kSpellings,
      [&](std::size_t offset, std::string_view text, bool parse_special) {
        return FakeEncode(text, parse_special, &calls, offset);
      });
  const auto want = FakeEncode(rendered, true);
  assert(got == want);
  assert(calls.size() == 1);
  assert(calls[0].offset == 0);
  assert(calls[0].text == rendered);
  assert(calls[0].parse_special);
  assert(Contains(got, 1000) == Contains(want, 1000));
}

// (b) A spelling wholly inside the content is inert: no 1000 from it, and
// the content piece is encoded with parse_special=false.
void TestMarkerInsideContentIsInert() {
  const std::string rendered = "<s>a<|eot|>b</s>";
  const std::vector<TextSpan> spans = {{3, 9}};
  std::vector<Call> calls;
  const auto got = EncodeWithContentSpans<int>(
      rendered, spans, kSpellings,
      [&](std::size_t offset, std::string_view text, bool parse_special) {
        return FakeEncode(text, parse_special, &calls, offset);
      });
  assert(!Contains(got, 1000));
  assert(calls.size() == 3);
  assert(calls[0].offset == 0 && calls[0].text == "<s>" &&
         calls[0].parse_special);
  assert(calls[1].offset == 3 && calls[1].text == "a<|eot|>b" &&
         !calls[1].parse_special);
  assert(calls[2].offset == 12 && calls[2].text == "</s>" &&
         calls[2].parse_special);
}

// (c) A real framing marker next to plain content still becomes 1000: the
// span does not overlap any spelling, so the render encodes as before.
void TestFramingMarkerSurvives() {
  const std::string rendered = "hi<|eot|>";
  const std::vector<TextSpan> spans = {{0, 2}};
  const auto got = EncodeWithContentSpans<int>(
      rendered, spans, kSpellings,
      [&](std::size_t offset, std::string_view text, bool parse_special) {
        return FakeEncode(text, parse_special, nullptr, offset);
      });
  assert(got == FakeEncode(rendered, true));
  assert(Contains(got, 1000));
}

// (d) A spelling straddling the content/framing edge still forces the split:
// content "x<|eo" plus framing "t|>" must not produce 1000.
void TestStraddlingMarkerIsInert() {
  const std::string rendered = "<s>x<|eot|>B</s>";
  const std::vector<TextSpan> spans = {{3, 5}};  // "x<|eo"
  std::vector<Call> calls;
  const auto got = EncodeWithContentSpans<int>(
      rendered, spans, kSpellings,
      [&](std::size_t offset, std::string_view text, bool parse_special) {
        return FakeEncode(text, parse_special, &calls, offset);
      });
  assert(!Contains(got, 1000));
  assert(calls.size() == 3);
  assert(calls[1].offset == 3 && calls[1].text == "x<|eo" &&
         !calls[1].parse_special);
  assert(calls[2].offset == 8 && calls[2].parse_special);
}

}  // namespace

int main() {
  TestUntouchedWithoutOverlap();
  TestMarkerInsideContentIsInert();
  TestFramingMarkerSurvives();
  TestStraddlingMarkerIsInert();
  std::cout << "content_spans checks passed.\n";
  return 0;
}
