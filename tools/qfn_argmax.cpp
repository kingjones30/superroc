// LFT SPEC2 item 1: G2 checkpoint top-2 gaps from the gufo executor.
//
// Jay's G2 arbiter rule (DECISIONS 14:39) ties a flip against the fork when the
// FLOAT reference path (LFN_FP4_PREFILL_W8A8=0) has a top-2 gap < 0.05 nats at
// that checkpoint. gufo serve prints no logprobs, so the clause was
// unmeasurable (RV-LFP1-final section 7). This tool teacher-forces each G2
// checkpoint prefix through the linked executor and prints the argmax, the
// runner-up, their logprobs and the gap in nats for ONE binary and ONE env.
//
// Tokenization is serve-identical: Encode(prefix) with default options
// (add_bos=false, parse_special=true), the same call the /v1/completions
// backend makes (inference_backend.cpp Tokenize). Each checkpoint is an
// independent prefill (Reset + Sync), like a separate n_predict=1 request.
//
// Usage: qfn_argmax <model.gguf> <prompt.txt> <checkpoints.txt> [n_ctx]
//   checkpoints.txt: one char-cut integer per line (blank lines and # ignored).
//   n_ctx defaults to 8192; every encoded prefix must fit.
// stdout: one CP line per checkpoint (machine-parseable); stderr: audit header.
//
// Sibling of qfn_ppl (whose default output stays byte-identical: untouched).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace {

// Byte offset of char cut in UTF-8 text (ASCII: identity). Throws past end.
std::size_t CharCutToBytes(const std::string& text, std::size_t cut) {
  std::size_t bytes = 0, chars = 0;
  while (bytes < text.size() && chars < cut) {
    const unsigned char c = static_cast<unsigned char>(text[bytes]);
    std::size_t len = 1;
    if ((c & 0x80) == 0x00) {
      len = 1;
    } else if ((c & 0xE0) == 0xC0) {
      len = 2;
    } else if ((c & 0xF0) == 0xE0) {
      len = 3;
    } else if ((c & 0xF8) == 0xF0) {
      len = 4;
    } else {
      throw std::runtime_error("invalid UTF-8 in prompt");
    }
    if (bytes + len > text.size())
      throw std::runtime_error("truncated UTF-8 in prompt");
    bytes += len;
    ++chars;
  }
  if (chars < cut)
    throw std::runtime_error("checkpoint cut past end of prompt");
  return bytes;
}

// JSON-quoted string for the CP line (token pieces hold quotes/newlines).
std::string JsonQuote(const std::string& s) {
  std::string out = "\"";
  for (const unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  out += "\"";
  return out;
}

const char* EnvOr(const char* name, const char* dflt) {
  const char* v = std::getenv(name);
  return (v == nullptr || v[0] == 0) ? dflt : v;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4) {
      std::cerr << "usage: qfn_argmax <model.gguf> <prompt.txt> "
                   "<checkpoints.txt> [n_ctx]\n";
      return 2;
    }
    const std::uint32_t n_ctx =
        argc > 4 ? static_cast<std::uint32_t>(std::stoul(argv[4])) : 8192;
    if (n_ctx < 2)
      throw std::runtime_error("n_ctx must be >= 2");

    std::ifstream pin(argv[2]);
    if (!pin)
      throw std::runtime_error(std::string("cannot open ") + argv[2]);
    std::ostringstream pss;
    pss << pin.rdbuf();
    const std::string prompt = pss.str();

    std::vector<std::size_t> cuts;
    std::ifstream cin(argv[3]);
    if (!cin)
      throw std::runtime_error(std::string("cannot open ") + argv[3]);
    std::string line;
    while (std::getline(cin, line)) {
      if (line.empty() || line[0] == '#')
        continue;
      cuts.push_back(static_cast<std::size_t>(std::stoul(line)));
    }
    if (cuts.empty())
      throw std::runtime_error("no checkpoints in file");

    std::string error;
    namespace qfn = gufo::models::qwen38_flash_next;
    auto model = qfn::Model::Load(
        argv[1], qfn::ModelOptions{.max_context = n_ctx}, &error);
    if (model == nullptr)
      throw std::runtime_error(error);
    auto session = model->CreateSession(
        gufo::core::SessionMode::kAutoregressive, n_ctx, &error);
    if (session == nullptr)
      throw std::runtime_error(error);

    std::cerr << "model=" << argv[1] << " prompt=" << argv[2]
              << " prompt_chars=" << prompt.size() << " n_ctx=" << n_ctx
              << " n_cp=" << cuts.size()
              << " LFN_FP4_PREFILL=" << EnvOr("LFN_FP4_PREFILL", "(unset)")
              << " LFN_FP4_PREFILL_W8A8="
              << EnvOr("LFN_FP4_PREFILL_W8A8", "(unset)") << "\n";

    for (const std::size_t cut : cuts) {
      const std::size_t nbytes = CharCutToBytes(prompt, cut);
      const std::string prefix = prompt.substr(0, nbytes);
      // Serve-identical tokenization (Encode defaults: no BOS, parse special).
      const auto encoded = model->tokenizer().Encode(prefix);
      std::vector<std::int32_t> ids;
      ids.reserve(encoded.size());
      for (const auto id : encoded)
        ids.push_back(static_cast<std::int32_t>(id));
      if (ids.empty())
        throw std::runtime_error("checkpoint encodes to zero tokens");
      if (ids.size() >= n_ctx)
        throw std::runtime_error("checkpoint prefix exceeds n_ctx");

      session->Reset();
      if (!session->Sync(std::span<const std::int32_t>(ids), &error))
        throw std::runtime_error("prefill failed: " + error);
      const auto logits = session->Logits();
      if (logits.empty() ||
          !std::all_of(logits.begin(), logits.end(),
                       [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("non-finite logits");

      double max_logit = -1e300;
      for (const float v : logits)
        max_logit = std::max(max_logit, static_cast<double>(v));
      double sum_exp = 0.0;
      for (const float v : logits)
        sum_exp += std::exp(static_cast<double>(v) - max_logit);
      const double log_z = max_logit + std::log(sum_exp);
      std::size_t top1 = 0, top2 = 1;
      if (logits[1] > logits[0])
        std::swap(top1, top2);
      for (std::size_t i = 2; i < logits.size(); ++i) {
        if (logits[i] > logits[top1]) {
          top2 = top1;
          top1 = i;
        } else if (logits[i] > logits[top2]) {
          top2 = i;
        }
      }
      const double lp1 = static_cast<double>(logits[top1]) - log_z;
      const double lp2 = static_cast<double>(logits[top2]) - log_z;
      const std::string s1 = model->tokenizer().DecodeTokenCopy(
          static_cast<gufo::tokenization::TokenId>(top1));
      const std::string s2 = model->tokenizer().DecodeTokenCopy(
          static_cast<gufo::tokenization::TokenId>(top2));
      std::printf("CP chars=%zu ntok=%zu top1=%zu %s lp1=%.6f top2=%zu %s "
                  "lp2=%.6f gap=%.6f\n",
                  cut, ids.size(), top1, JsonQuote(s1).c_str(), lp1, top2,
                  JsonQuote(s2).c_str(), lp2, lp1 - lp2);
      std::fflush(stdout);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "qfn_argmax: " << e.what() << "\n";
    return 1;
  }
}
