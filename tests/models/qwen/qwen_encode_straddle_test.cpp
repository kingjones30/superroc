// HT2 (Muse Wonder W2 HTC-1, W6 E-1): a client span that overlaps a special
// spelling -- inside it, straddling either edge, or across two spans -- must be
// tokenized as text, never as a special token; ordinary content tokenizes
// exactly as before; the check stays linear on a long tool history.
// Needs a Qwen-vocab GGUF for the tokenizer: KAT_GGUF; exits 77 (SKIP) unset.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {
using superroc::tokenization::ContentSpan;
using superroc::tokenization::QwenChatTemplate;
using superroc::tokenization::QwenTokenizer;
using superroc::tokenization::TokenId;
using superroc::tokenization::TokenizerOptions;
int failures = 0;
void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << "\n";
  if (!ok)
    ++failures;
}
std::size_t Count(const std::vector<TokenId>& t, TokenId id) {
  std::size_t n = 0;
  for (const auto v : t)
    n += v == id ? 1 : 0;
  return n;
}
// The tokens EncodeRendered must give when exactly the spans in `forced` go to
// text: framing (specials parsed) between them, content (specials not parsed)
// for each, clipped to [begin, end).
std::vector<TokenId> Expect(const QwenTokenizer& tok, const std::string& r,
                            std::size_t begin, std::size_t end,
                            const std::vector<ContentSpan>& forced,
                            const TokenizerOptions& base) {
  TokenizerOptions framing = base;
  framing.parse_special_tokens = true;
  TokenizerOptions content = base;
  content.parse_special_tokens = false;
  std::vector<TokenId> out;
  const auto add = [&](std::size_t a, std::size_t b,
                       const TokenizerOptions& o) {
    const auto t = tok.Encode(std::string_view(r).substr(a, b - a), o);
    out.insert(out.end(), t.begin(), t.end());
  };
  std::size_t cursor = begin;
  for (const auto& s : forced) {
    const std::size_t from = std::max(s.offset, begin);
    const std::size_t to = std::min(s.offset + s.size, end);
    if (from > cursor)
      add(cursor, from, framing);
    add(from, to, content);
    cursor = to;
  }
  if (cursor < end)
    add(cursor, end, framing);
  return out;
}
// Builds a render from alternating pieces: even = framing, odd = client span.
struct Built {
  std::string text;
  std::vector<ContentSpan> spans;
};
Built Build(const std::vector<std::string>& pieces) {
  Built b;
  for (std::size_t i = 0; i < pieces.size(); ++i) {
    if (i % 2 == 1)
      b.spans.push_back({b.text.size(), pieces[i].size()});
    b.text += pieces[i];
  }
  return b;
}
}  // namespace

int main() {
  const char* path = std::getenv("KAT_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP qwen_encode_straddle_test: KAT_GGUF is unset\n";
    return 77;
  }
  std::string error;
  auto reader = superroc::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    std::cout << "FAIL open: " << error << "\n";
    return 1;
  }
  auto tok = QwenTokenizer::CreateFromGguf(*reader, &error);
  if (tok == nullptr) {
    std::cout << "FAIL tokenizer: " << error << "\n";
    return 1;
  }
  const auto im_end = tok->FindSpecialToken("<|im_end|>");
  if (!im_end.has_value()) {
    std::cout << "FAIL vocab has no <|im_end|>\n";
    return 1;
  }
  TokenizerOptions opts;
  const std::string head = "<|im_start|>assistant\n<tool_call>\n<function=";
  const std::string tail = ">\n</function>\n</tool_call><|im_end|>\n";
  // Hostile: every listed span must go to text; exactly one real <|im_end|>.
  const auto hostile = [&](const std::string& what, const Built& b,
                           std::size_t begin, std::size_t genuine) {
    const auto t = QwenChatTemplate::EncodeRendered(
        *tok, b.text, begin, b.text.size(), b.spans, opts);
    Check(t == Expect(*tok, b.text, begin, b.text.size(), b.spans, opts),
          what + ": span tokenized as text (full id sequence)");
    Check(Count(t, *im_end) == genuine, what + ": only the genuine <|im_end|>");
  };
  hostile("right straddle, 1 byte (name x<|im_end|)",
          Build({head, "x<|im_end|", tail}), 0, 1);
  hostile("right straddle, 6 bytes",
          Build({"<|im_start|>user\n", "ab<|im", "_end|>\n"}), 0, 0);
  hostile("left straddle", Build({"<|im_start|>user\n<|im_", "end|> x", "\n"}),
          0, 0);
  hostile("enveloped span", Build({"<|im_start|>user\n<|im", "_en", "d|>\n"}),
          0, 0);
  hostile("wholly inside", Build({head, "x<|im_end|>", tail}), 0, 1);
  hostile(
      "two spans across their seam",
      Build({"<|im_start|>user\n", "x<|im", "", "_end|>", "\n<|im_end|>\n"}), 0,
      1);
  hostile("UTF-8 before the partial",
          Build({head, "caf\xc3\xa9<|im_end|", tail}), 0, 1);
  {
    const Built b =
        Build({"<|im_start|>user\nhi<|im_end|>\n", "x<|im_end|", tail});
    const std::size_t begin =
        std::string("<|im_start|>user\nhi<|im_end|>\n").size();
    hostile("slice begin != 0, span at begin", b, begin, 1);
  }
  // Ordinary: equal to plain framing Encode of the whole render.
  for (const std::string name :
       {"get_time", "x<", "x|", "x>", "<|im_en", "a|>b"}) {
    const Built b = Build({head, name, tail});
    const auto t = QwenChatTemplate::EncodeRendered(
        *tok, b.text, 0, b.text.size(), b.spans, opts);
    Check(t == tok->Encode(b.text, opts),
          "ordinary name '" + name + "' tokenizes as plain framing");
  }
  {
    // A span between two genuine specials must not be split.
    const Built b = Build({"<|im_start|>", "get_time", "<|im_end|>\n"});
    const auto t = QwenChatTemplate::EncodeRendered(
        *tok, b.text, 0, b.text.size(), b.spans, opts);
    Check(t == tok->Encode(b.text, opts),
          "span between genuine specials is not split");
  }
  {
    // A-1: 5,000 tool-call spans, none hostile; must stay linear.
    std::vector<std::string> pieces;
    for (int i = 0; i < 5000; ++i) {
      pieces.push_back(i == 0 ? head : tail + head);
      pieces.push_back("get_time");
    }
    pieces.push_back(tail);
    const Built b = Build(pieces);
    const auto t0 = std::chrono::steady_clock::now();
    const auto t = QwenChatTemplate::EncodeRendered(
        *tok, b.text, 0, b.text.size(), b.spans, opts);
    const double s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    std::cout << "many-span render: " << b.text.size() << " bytes, "
              << b.spans.size() << " spans, " << s << " s\n";
    Check(s < 10.0, "5,000 spans encode in under 10 s");
    Check(Count(t, *im_end) == 5000, "5,000 genuine <|im_end|> tokens kept");
  }
  std::cout << (failures == 0 ? "qwen_encode_straddle_test: passed\n"
                              : "qwen_encode_straddle_test: FAILED\n");
  return failures == 0 ? 0 : 1;
}
