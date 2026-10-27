// ling3-ref — scalar CPU oracle for Ling-3.0-flash.
//
// Loads the GGUF, runs the reference forward one token at a time, and prints
// the greedy token sequence. This is the process the Ling-on-superroc gate (a)
// compares against the fork's `llama-completion` output.
//
// Usage:
//   ling3-ref <model.gguf> [--prompt "text"] [--tokens N] [--temp 0]
//             [--dump-prefix] [--dump-tokens "1,2,3"]
//
// It is a correctness oracle, not a server: expect minutes per token.

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling3/chat_template.hpp"
#include "src/models/ling3/reference.hpp"
#include "src/models/ling3/tokenizer.hpp"
#include "src/models/ling3/weights.hpp"

namespace chat = superroc::models::ling3::chat;

namespace {

int Fail(const std::string& what) {
  std::fprintf(stderr, "ling3-ref: %s\n", what.c_str());
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <model.gguf> [--prompt TEXT] [--tokens N]\n", argv[0]);
    return 2;
  }
  const std::string model_path = argv[1];

  std::string prompt;
  std::string prompt_file;
  std::string dump_tokens;
  std::size_t max_tokens = 32;
  bool tokenize_only = false;
  bool trace = false;
  bool render_chat = false;
  std::string teacher_file;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "ling3-ref: %s needs a value\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--prompt") {
      prompt = next("--prompt");
    } else if (arg == "--prompt-file") {
      prompt_file = next("--prompt-file");
    } else if (arg == "--tokens") {
      max_tokens = std::strtoul(next("--tokens").c_str(), nullptr, 10);
    } else if (arg == "--dump-tokens") {
      dump_tokens = next("--dump-tokens");
    } else if (arg == "--teacher-force") {
      teacher_file = next("--teacher-force");
    } else if (arg == "--render-chat") {
      render_chat = true;
    } else if (arg == "--tokenize-only") {
      tokenize_only = true;
    } else if (arg == "--trace") {
      trace = true;
    } else {
      std::fprintf(stderr, "ling3-ref: unknown option %s\n", arg.c_str());
      return 2;
    }
  }

  if (render_chat) {
    // Renders the gate-(b) prompt with this package's own chat template and
    // writes it to stdout, so the gate feeds byte-identical text to both the
    // fork and superroc and proves the template rather than a hand-written string.
    std::vector<chat::Tool> tools;
    chat::Tool weather;
    weather.name = "get_weather";
    weather.description = "Get the current weather for a city";
    weather.parameters_json =
        "{\"type\": \"object\", \"properties\": {\"city\": {\"type\": "
        "\"string\", \"description\": \"The city name\"}}, \"required\": "
        "[\"city\"]}";
    tools.push_back(weather);
    std::vector<chat::Message> msgs;
    msgs.push_back({"user", prompt.empty() ? "weather in Paris?" : prompt, "", {}});
    std::string cerr;
    auto rendered = chat::RenderChat(msgs, tools, /*enable_thinking=*/false,
                                     /*add_generation_prompt=*/true, &cerr);
    if (!rendered.has_value()) {
      return Fail("chat template: " + cerr);
    }
    std::fwrite(rendered->data(), 1, rendered->size(), stdout);
    std::fputc('\n', stdout);
    return 0;
  }

  if (!prompt_file.empty()) {
    // Reading the prompt from a file keeps the bytes exactly as written, which
    // matters when the text contains backslashes or quotes: llama-tokenize
    // interprets escapes in its -p argument, so a shell-passed prompt is not
    // a fair comparison.
    std::FILE* fh = std::fopen(prompt_file.c_str(), "rb");
    if (fh == nullptr) {
      return Fail("cannot read " + prompt_file);
    }
    char buf[4096];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), fh)) > 0) {
      prompt.append(buf, n);
    }
    std::fclose(fh);
  }

  std::string error;
  auto reader = superroc::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    return Fail("cannot open " + model_path + ": " + error);
  }

  auto weights = superroc::models::ling3::ModelWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    return Fail("cannot bind " + model_path + ": " + error);
  }
  const superroc::models::ling3::Config& c = weights->config;
  std::fprintf(stderr,
               "ling3-ref: bound %u trunk blocks (+%u MTP), hidden %u, vocab %u, "
               "%u MLA / %u KDA\n",
               c.num_layers, c.nextn_layers, c.hidden_size, c.vocab_size,
               [&] {
                 std::uint32_t n = 0;
                 for (std::uint32_t i = 0; i < c.num_layers; ++i) {
                   n += c.IsMla(i) ? 1u : 0u;
                 }
                 return n;
               }(),
               c.num_layers - [&] {
                 std::uint32_t n = 0;
                 for (std::uint32_t i = 0; i < c.num_layers; ++i) {
                   n += c.IsMla(i) ? 1u : 0u;
                 }
                 return n;
               }());

  auto tokenizer = superroc::models::ling3::LingTokenizer::FromGguf(*reader, &error);
  if (!tokenizer.has_value()) {
    return Fail("cannot build the Ling tokenizer: " + error);
  }
  // The tokenizer is not used for the oracle when an explicit token list is
  // given, but a mismatch here would silently change every token id.
  // This file carries tokenizer.ggml.add_bos_token = false, so the fork
  // prepends nothing and neither do we.
  const std::vector<std::int32_t> prompt_tokens =
      prompt.empty() ? std::vector<std::int32_t>{} : tokenizer->Encode(prompt);
  std::fprintf(stderr, "ling3-ref: prompt is %zu tokens\n", prompt_tokens.size());
  for (const auto t : prompt_tokens) {
    const std::vector<std::int32_t> one{t};
    std::fprintf(stderr, "  %d '%s'\n", t, tokenizer->Decode(one).c_str());
  }

  std::vector<std::size_t> dump_at;
  if (!dump_tokens.empty()) {
    std::size_t start = 0;
    while (start <= dump_tokens.size()) {
      const std::size_t comma = dump_tokens.find(',', start);
      const std::string part = dump_tokens.substr(
          start, comma == std::string::npos ? std::string::npos : comma - start);
      if (!part.empty()) {
        dump_at.push_back(std::strtoul(part.c_str(), nullptr, 10));
      }
      if (comma == std::string::npos) {
        break;
      }
      start = comma + 1;
    }
  }

  if (tokenize_only) {
    std::printf("[");
    for (std::size_t i = 0; i < prompt_tokens.size(); ++i) {
      std::printf("%s%d", i ? ", " : "", prompt_tokens[i]);
    }
    std::printf("]\n");
    std::printf("count=%zu\n", prompt_tokens.size());
    for (const std::int32_t t : prompt_tokens) {
      std::printf("  %d\t%s\n", t, tokenizer->TokenText(t).c_str());
    }
    return 0;
  }

  superroc::models::ling3::ReferenceModel model(*weights, c.context_length);

  std::vector<float> logits;
  std::vector<std::int32_t> generated;
  auto feed = [&](std::int32_t token) {
    if (!model.Step(token, logits, &error)) {
      std::fprintf(stderr, "ling3-ref: %s\n", error.c_str());
      std::exit(1);
    }
  };

  // ---- teacher-forced top-1 agreement -------------------------------------
  // Feed a known token sequence and, at every position, compare the model's
  // argmax with the token the fork actually produced. This measures whether
  // the forward pass agrees with the fork on ranking, which is the question
  // that matters once the two use different matmul arithmetic (exact float here,
  // Q8_0-quantised activations in ggml's CPU path, Q8_1 in its GPU MMQ path).
  // Exact 32-token equality is not achievable across those, and is not asked for.
  if (!teacher_file.empty()) {
    std::FILE* fh = std::fopen(teacher_file.c_str(), "rb");
    if (fh == nullptr) {
      return Fail("cannot open " + teacher_file);
    }
    std::vector<std::int32_t> seq;
    {
      char buf[65536];
      const std::size_t n = std::fread(buf, 1, sizeof(buf) - 1, fh);
      buf[n] = 0;
      std::fclose(fh);
      const std::string text(buf);
      std::size_t i = 0;
      while (i < text.size()) {
        while (i < text.size() && !std::isdigit(static_cast<unsigned char>(text[i])) &&
               text[i] != '-') {
          ++i;
        }
        if (i >= text.size()) break;
        const std::size_t start = i;
        if (text[i] == '-') ++i;
        while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) ++i;
        seq.push_back(std::atoi(text.substr(start, i - start).c_str()));
      }
    }
    if (seq.size() < 2) {
      return Fail("teacher-force file needs at least two token ids");
    }
    std::size_t hits = 0;
    for (std::size_t i = 0; i + 1 < seq.size(); ++i) {
      if (!model.Step(seq[i], logits, &error)) {
        return Fail(error);
      }
      std::int32_t best = 0;
      for (std::size_t t = 1; t < logits.size(); ++t) {
        if (logits[t] > logits[static_cast<std::size_t>(best)]) {
          best = static_cast<std::int32_t>(t);
        }
      }
      const bool ok = best == seq[i + 1];
      if (ok) ++hits;
      std::printf("%s pos=%zu pred=%d ref=%d\n", ok ? "HIT " : "MISS", i, best,
                  seq[i + 1]);
    }
    const std::size_t total = seq.size() - 1;
    std::printf("TEACHER-FORCED top1: %zu/%zu = %.1f%%\n", hits, total,
                total ? 100.0 * static_cast<double>(hits) / static_cast<double>(total) : 0.0);
    return 0;
  }

  if (trace) {
    model.SetTrace([](std::uint32_t block, const char* name, const float* v,
                      std::size_t n) {
      std::printf("LG-TRACE %u %s", block, name);
      // ggml's debug print emits the FIRST 3 and the LAST 3 elements of the
      // fastest axis, in that order. Emitting the same 6 (rather than just the
      // first 3) is what lets the comparator line up row for row.
      const std::size_t lead = n < 3 ? n : 3;
      for (std::size_t r = 0; r < lead; ++r) {
        std::printf(" %.6f", static_cast<double>(v[r]));
      }
      if (n > 6) {
        for (std::size_t r = n - 3; r < n; ++r) {
          std::printf(" %.6f", static_cast<double>(v[r]));
        }
      }
      std::printf("\n");
    });
  }

  // Prefill: every prompt token but the last; the last one is fed by the loop.
  for (std::size_t i = 0; i + 1 < prompt_tokens.size(); ++i) {
    feed(prompt_tokens[i]);
  }
  std::int32_t next =
      prompt_tokens.empty() ? tokenizer->Bos() : prompt_tokens.back();

  std::printf("LG-REF-BEGIN\n");
  for (std::size_t step = 0; step < max_tokens; ++step) {
    feed(next);
    std::size_t best = 0;
    for (std::size_t t = 1; t < logits.size(); ++t) {
      if (logits[t] > logits[best]) {
        best = t;
      }
    }
    const bool dump =
        std::find(dump_at.begin(), dump_at.end(), step) != dump_at.end();
    if (dump) {
      std::printf("LG-REF-HIDDEN pos=%zu token=%d\n", model.Position() - 1, next);
      // Top-5 so a divergence is diagnosable without re-running.
      std::vector<std::size_t> order(logits.size());
      for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
      }
      std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                        [&](std::size_t a, std::size_t b) {
                          return logits[a] > logits[b];
                        });
      for (int k = 0; k < 5; ++k) {
        std::printf("  top%d %zu %.6f\n", k + 1, order[k], logits[order[k]]);
      }
    }
    generated.push_back(static_cast<std::int32_t>(best));
    std::printf("%d\n", static_cast<int>(best));
    std::fflush(stdout);
    next = static_cast<std::int32_t>(best);
    if (tokenizer->IsStop(next)) {
      break;
    }
  }
  std::printf("LG-REF-END\n");

  std::fprintf(stderr, "ling3-ref: generated %zu tokens: %s\n", generated.size(),
               tokenizer->Decode(generated).c_str());
  return 0;
}
