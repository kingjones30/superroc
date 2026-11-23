// Content-span aware encoding for chat templates (SR-03).
//
// A rendered prompt mixes renderer framing (control markers such as
// "<|im_start|>") with client-supplied message text. A special-token spelling
// that appears inside client text must reach the model as the characters it
// contains, never as a control token. At the same time, ordinary prompts
// must tokenize exactly as before so the prompt cache stays byte-identical.
//
// Rule (copied from QwenChatTemplate::EncodeRendered): a content span is
// split out and encoded with special parsing OFF only when a special-token
// spelling overlaps the span; otherwise the whole render is encoded exactly
// as today. Any spelling that OVERLAPS the span counts -- wholly inside it,
// or straddling either edge -- so client text "x<|eot|" followed by the
// renderer's own "t|>" cannot become a real control token. A spelling wholly
// inside the framing (a real marker next to the span) does not count. The
// overlap search runs over a window of len - 1 bytes either side of the span
// only (linear in the prompt).
//
// Spans are sorted by offset and non-overlapping. Offsets are byte offsets
// into `rendered`.

#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace superroc::core {

// A half-open [offset, offset + size) span of client-supplied text inside a
// rendered prompt.
struct TextSpan {
  std::size_t offset;
  std::size_t size;
};

// Encodes `rendered` with the rule above.
// `encode(offset, text, parse_special)` encodes one piece starting at byte
// `offset` of the render; `parse_special` selects special-token matching.
// `special_spellings` holds the text of every control/special token.
template<class Token, class EncodeFn>
std::vector<Token> EncodeWithContentSpans(
    std::string_view rendered, std::span<const TextSpan> spans,
    std::span<const std::string> special_spellings, EncodeFn encode) {
  const std::size_t begin = 0;
  const std::size_t end = rendered.size();
  std::vector<Token> tokens;
  std::size_t cursor = begin;
  for (const auto& span : spans) {
    const auto span_end = span.offset + span.size;
    if (span_end <= begin || span.offset >= end) {
      continue;
    }
    const auto from = std::max(span.offset, begin);
    const auto to = std::min(span_end, end);
    if (to <= from) {
      continue;
    }
    bool split = false;
    for (const auto& spelling : special_spellings) {
      if (spelling.empty()) {
        continue;
      }
      const std::size_t len = spelling.size();
      const std::size_t lo = from >= len - 1 ? from - (len - 1) : 0;
      const std::size_t hi = std::min(rendered.size(), to + (len - 1));
      const auto window = rendered.substr(lo, hi - lo);
      const std::string_view needle(spelling);
      for (std::size_t at = window.find(needle); at != std::string_view::npos;
           at = window.find(needle, at + 1)) {
        if (lo + at < to && lo + at + len > from) {
          split = true;
          break;
        }
      }
      if (split) {
        break;
      }
    }
    if (!split) {
      continue;
    }
    if (from > cursor) {
      auto framing = encode(cursor, rendered.substr(cursor, from - cursor),
                            /*parse_special=*/true);
      tokens.insert(tokens.end(), framing.begin(), framing.end());
    }
    auto content = encode(from, rendered.substr(from, to - from),
                          /*parse_special=*/false);
    tokens.insert(tokens.end(), content.begin(), content.end());
    cursor = to;
  }
  if (cursor == begin) {
    return encode(begin, rendered.substr(begin, end - begin),
                  /*parse_special=*/true);
  }
  if (cursor < end) {
    auto tail = encode(cursor, rendered.substr(cursor, end - cursor),
                       /*parse_special=*/true);
    tokens.insert(tokens.end(), tail.begin(), tail.end());
  }
  return tokens;
}

}  // namespace superroc::core
