// Flint greedy / teacher-forced parity vs the torch reference ids, plus a
// wall-clock speed mode. Policy is fp32 KV, no fused QK-RoPE-KV, no Glimmer
// fp16 plane.
// Usage:
//   flint_parity <gguf> <ref-greedy.txt>
//   flint_parity --speed <gguf> <corpus> <n_prompt> <n_decode>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

struct Case {
  std::string name;
  std::vector<gufo::tokenization::TokenId> prompt;
  std::vector<gufo::tokenization::TokenId> generated;
};

std::vector<gufo::tokenization::TokenId> ParseIds(const std::string& s) {
  std::vector<gufo::tokenization::TokenId> ids;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) {
      continue;
    }
    ids.push_back(static_cast<gufo::tokenization::TokenId>(std::stoul(tok)));
  }
  return ids;
}

std::vector<Case> LoadCases(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  std::vector<Case> cases;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    const auto p1 = line.find('|');
    const auto p2 = line.find('|', p1 == std::string::npos ? 0 : p1 + 1);
    if (p1 == std::string::npos || p2 == std::string::npos) {
      throw std::runtime_error("bad ref-greedy line: " + line);
    }
    Case c;
    c.name = line.substr(0, p1);
    c.prompt = ParseIds(line.substr(p1 + 1, p2 - p1 - 1));
    c.generated = ParseIds(line.substr(p2 + 1));
    if (c.prompt.empty() || c.generated.empty()) {
      throw std::runtime_error("empty ids in " + c.name);
    }
    cases.push_back(std::move(c));
  }
  return cases;
}

gufo::hip::QwenExecutionPolicy FlintPolicy() {
  gufo::hip::QwenExecutionPolicy policy;
  policy.kv_cache_storage = gufo::hip::QwenKvCacheStorage::kFp32;
  policy.fuse_qk_norm_rope_kv = false;
  policy.glimmer_fp16_kv = true;
  return policy;
}

std::unique_ptr<gufo::hip::QwenGpuExecutor> MakeExecutor(
    const char* gguf, std::uint32_t n_ctx) {
  std::string error;
  auto owner = gufo::core::GgufReader::OpenFile(gguf, &error);
  if (owner == nullptr) {
    throw std::runtime_error(error);
  }
  auto executor = gufo::hip::QwenGpuExecutor::CreateFromGguf(
      std::shared_ptr<const gufo::core::GgufReader>(std::move(owner)), &error,
      n_ctx, FlintPolicy());
  if (executor == nullptr) {
    throw std::runtime_error(error);
  }
  return executor;
}

gufo::tokenization::TokenId Argmax(std::span<const float> logits) {
  if (logits.empty()) {
    throw std::runtime_error("empty logits");
  }
  std::size_t best = 0;
  float best_v = logits[0];
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > best_v) {
      best_v = logits[i];
      best = i;
    }
  }
  return static_cast<gufo::tokenization::TokenId>(best);
}

double TokPerSec(std::size_t n, double seconds) {
  if (!(seconds > 0.0) || n == 0) {
    return 0.0;
  }
  return static_cast<double>(n) / seconds;
}

int RunParity(const char* gguf, const char* fixture) {
  auto executor = MakeExecutor(gguf, 4096);
  const auto cases = LoadCases(fixture);
  for (const auto& c : cases) {
    const std::size_t n = c.generated.size();
    executor->Reset();
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<gufo::tokenization::TokenId> got;
    got.reserve(n);
    got.push_back(executor->ForwardPromptBatch(c.prompt, 0, true));
    const auto t1 = std::chrono::steady_clock::now();
    for (std::size_t i = 1; i < n; ++i) {
      const std::uint32_t pos =
          static_cast<std::uint32_t>(c.prompt.size() + i - 1);
      got.push_back(executor->ForwardToken(got.back(), pos, true));
    }
    const auto t2 = std::chrono::steady_clock::now();
    std::size_t k = 0;
    int first_diff = -1;
    for (std::size_t i = 0; i < n; ++i) {
      if (got[i] == c.generated[i]) {
        ++k;
      } else {
        first_diff = static_cast<int>(i);
        break;
      }
    }
    if (first_diff < 0) {
      k = n;
    }
    const double prefill_s =
        std::chrono::duration<double>(t1 - t0).count();
    const double decode_s =
        std::chrono::duration<double>(t2 - t1).count();
    std::printf("GREEDY %s prefix_match=%zu/%zu first_diff=%d\n", c.name.c_str(),
                k, n, first_diff);
    std::printf("SPEED %s prefill_tok_s=%.4f decode_tok_s=%.4f\n",
                c.name.c_str(), TokPerSec(c.prompt.size(), prefill_s),
                TokPerSec(n > 1 ? n - 1 : 0, decode_s));

    executor->Reset();
    std::vector<gufo::tokenization::TokenId> ids = c.prompt;
    if (n >= 2) {
      ids.insert(ids.end(), c.generated.begin(),
                 c.generated.begin() + static_cast<std::ptrdiff_t>(n - 1));
    }
    (void)executor->ForwardPromptBatch(ids, 0, false);
    std::size_t agree = 0;
    const std::size_t r0 = c.prompt.size() - 1;
    const std::size_t r1 = ids.size() - 1;
    for (std::size_t r = r0; r <= r1; ++r) {
      executor->MaterializeFlintRowLogits(r);
      const auto pred = Argmax(executor->CopyLastLogits());
      const auto want = c.generated[r - c.prompt.size() + 1];
      if (pred == want) {
        ++agree;
      }
    }
    std::printf("TF %s agree=%zu/%zu\n", c.name.c_str(), agree, n);
    std::fflush(stdout);
  }
  return 0;
}

int RunSpeed(const char* gguf, const char* corpus, int n_prompt,
             int n_decode) {
  if (n_prompt < 1 || n_decode < 1) {
    throw std::runtime_error("n_prompt and n_decode must be positive");
  }
  const std::uint32_t n_ctx =
      static_cast<std::uint32_t>(n_prompt + n_decode + 16);
  auto executor = MakeExecutor(gguf, n_ctx);
  std::ifstream in(corpus);
  if (!in) {
    throw std::runtime_error(std::string("cannot open ") + corpus);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  const auto& tokenizer = executor->GetTokenizer();
  gufo::tokenization::TokenizerOptions options;
  options.add_bos = tokenizer.AddBosToken();
  const auto tokens = tokenizer.Encode(ss.str(), options);
  if (static_cast<int>(tokens.size()) < n_prompt) {
    throw std::runtime_error("corpus is shorter than n_prompt");
  }

  auto run = [&](int np, int nd) {
    std::vector<gufo::tokenization::TokenId> prompt(
        tokens.begin(), tokens.begin() + np);
    executor->Reset();
    const auto t0 = std::chrono::steady_clock::now();
    auto tok = executor->ForwardPromptBatch(prompt, 0, true);
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < nd; ++i) {
      const std::uint32_t pos = static_cast<std::uint32_t>(np + i);
      tok = executor->ForwardToken(tok, pos, true);
    }
    const auto t2 = std::chrono::steady_clock::now();
    const double prefill_s =
        std::chrono::duration<double>(t1 - t0).count();
    const double decode_s =
        std::chrono::duration<double>(t2 - t1).count();
    return std::pair<double, double>{TokPerSec(static_cast<std::size_t>(np),
                                               prefill_s),
                                     TokPerSec(static_cast<std::size_t>(nd),
                                               decode_s)};
  };

  (void)run(64, 16);
  const auto [prefill, decode] = run(n_prompt, n_decode);
  std::printf("SPEED n_prompt=%d prefill_tok_s=%.4f decode_tok_s=%.4f\n",
              n_prompt, prefill, decode);
  return 0;
}

// Jay: per-position argmax + last-row top-5 of the batched prefill and of the
// token loop on the same ids (compare with tools/flint_ref.py --ids).
int RunDiag(const char* gguf, const std::string& csv) {
  std::vector<gufo::tokenization::TokenId> ids;
  std::stringstream ss(csv);
  for (std::string item; std::getline(ss, item, ',');) {
    ids.push_back(static_cast<gufo::tokenization::TokenId>(std::stoul(item)));
  }
  auto executor = MakeExecutor(gguf, 4096);
  auto top5 = [](std::span<const float> logits) {
    std::vector<std::size_t> idx(logits.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](std::size_t a, std::size_t b) {
                        return logits[a] > logits[b];
                      });
    std::string out;
    for (int i = 0; i < 5; ++i) {
      out += " " + std::to_string(idx[i]) + ":" + std::to_string(logits[idx[i]]);
    }
    return out;
  };
  for (int mode = 0; mode < 2; ++mode) {
    executor->Reset();
    std::string argmax;
    std::string last;
    std::string nll;
    if (mode == 0) {
      (void)executor->ForwardPromptBatch(ids, 0, false);
      for (std::size_t r = 0; r < ids.size(); ++r) {
        executor->MaterializeFlintRowLogits(r);
        const auto logits = executor->CopyLastLogits();
        argmax += (r ? "," : "") + std::to_string(Argmax(logits));
        if (r + 1 < ids.size()) {
          double mx = -1e300, se = 0.0;
          for (const float v : logits) mx = std::max(mx, static_cast<double>(v));
          for (const float v : logits) se += std::exp(static_cast<double>(v) - mx);
          char buf[32];
          std::snprintf(buf, sizeof(buf), "%s%.5f", r ? "," : "",
                        mx + std::log(se) - logits[ids[r + 1]]);
          nll += buf;
        }
        if (r + 1 == ids.size()) last = top5(logits);
      }
    } else {
      for (std::size_t r = 0; r < ids.size(); ++r) {
        (void)executor->ForwardToken(ids[r], static_cast<std::uint32_t>(r), true);
        const auto logits = executor->CopyLastLogits();
        argmax += (r ? "," : "") + std::to_string(Argmax(logits));
        if (r + 1 < ids.size()) {
          double mx = -1e300, se = 0.0;
          for (const float v : logits) mx = std::max(mx, static_cast<double>(v));
          for (const float v : logits) se += std::exp(static_cast<double>(v) - mx);
          char buf[32];
          std::snprintf(buf, sizeof(buf), "%s%.5f", r ? "," : "",
                        mx + std::log(se) - logits[ids[r + 1]]);
          nll += buf;
        }
        if (r + 1 == ids.size()) last = top5(logits);
      }
    }
    std::printf("%s nll_per_pos %s\n", mode == 0 ? "BATCH" : "TOKEN", nll.c_str());
    std::printf("%s argmax_per_pos %s\n%s last_top5%s\n",
                mode == 0 ? "BATCH" : "TOKEN", argmax.c_str(),
                mode == 0 ? "BATCH" : "TOKEN", last.c_str());
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "--diag") {
      return RunDiag(argv[2], argv[3]);
    }
    // Jay: --hidden <gguf> <ids csv> <layers csv> <out.bin>: the batched
    // prefill's layer outputs, float32 [token][layer][hidden].
    if (argc == 6 && std::string(argv[1]) == "--hidden") {
      std::vector<gufo::tokenization::TokenId> ids;
      std::vector<std::uint32_t> layers;
      std::stringstream a(argv[3]);
      for (std::string t; std::getline(a, t, ',');) ids.push_back(std::stoul(t));
      std::stringstream b(argv[4]);
      for (std::string t; std::getline(b, t, ',');) layers.push_back(std::stoul(t));
      auto executor = MakeExecutor(argv[2], 4096);
      executor->Reset();
      executor->SetPromptHiddenCapture(true, layers);
      (void)executor->ForwardPromptBatch(ids, 0, false);
      const auto h = executor->GetPromptHiddenStates();
      std::ofstream out(argv[5], std::ios::binary);
      out.write(reinterpret_cast<const char*>(h.data()),
                static_cast<std::streamsize>(h.size() * sizeof(float)));
      std::printf("HIDDEN floats=%zu tokens=%zu layers=%zu\n", h.size(),
                  ids.size(), layers.size());
      return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--speed") {
      if (argc != 6) {
        std::cerr << "usage: flint_parity --speed <gguf> <corpus> <n_prompt> "
                     "<n_decode>\n";
        return 2;
      }
      return RunSpeed(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]));
    }
    if (argc != 3) {
      std::cerr << "usage: flint_parity <gguf> <ref-greedy.txt>\n";
      return 2;
    }
    return RunParity(argv[1], argv[2]);
  } catch (const std::exception& ex) {
    std::cerr << "flint_parity: " << ex.what() << "\n";
    return 1;
  }
}
