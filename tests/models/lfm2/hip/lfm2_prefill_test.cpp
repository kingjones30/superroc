// LFM2 batched prefill test. Compares ForwardPromptBatch //
// LFM2 against the token loop + 32 greedy steps. SKIP 77 without LFM2_A_GGUF.
// // LFM2 NOTE (deviation): ForwardPromptChunk is
// private; // LFM2 the test uses the public ForwardPromptBatch, which forwards
// to // LFM2 ForwardLfm2PromptChunk in one chunk for these lengths (6000 splits
// // LFM2 4096+1904 at the 4096 max batch, still exercising the batched path).
//

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error(std::string("cannot open ") + path);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::vector<std::size_t> Top5(const std::vector<float>& logits) {
  std::vector<std::size_t> idx(logits.size());
  std::iota(idx.begin(), idx.end(), 0);
  const std::size_t k = std::min<std::size_t>(5, idx.size());
  std::partial_sort(
      idx.begin(), idx.begin() + k, idx.end(),
      [&](std::size_t a, std::size_t b) { return logits[a] > logits[b]; });
  idx.resize(k);
  return idx;
}

}  // namespace

int main(int argc, char** argv) {
  const char* configured = argc >= 2 ? argv[1] : std::getenv("LFM2_A_GGUF");
  if (configured == nullptr) {
    std::cerr << "SKIP lfm2_prefill_test: LFM2_A_GGUF is not set\n";
    return 77;
  }
  const std::string path = configured;
  try {
    std::string err;
    auto opened = superroc::core::GgufReader::OpenFile(path, &err);
    if (opened == nullptr) {
      std::cerr << "SKIP lfm2_prefill_test: " << path << " unreadable: " << err
                << '\n';
      return 77;
    }
    std::shared_ptr<const superroc::core::GgufReader> reader(std::move(opened));
    superroc::hip::QwenExecutionPolicy policy;  // LFM2 (lfm2_ppl policy)
    policy.kv_cache_storage = superroc::hip::QwenKvCacheStorage::kFp32;
    policy.fuse_qk_norm_rope_kv = false;
    constexpr std::uint32_t kContext = 8192;
    auto executor = superroc::hip::QwenGpuExecutor::CreateFromGguf(
        reader, &err, kContext, policy);
    if (executor == nullptr) {
      std::cerr << "FAIL load " << err << '\n';
      return 1;
    }
    const auto& config = executor->GetConfig();
    if (!config.lfm2) {
      std::cerr << "FAIL not an lfm2 config\n";
      return 1;
    }
    const char* configured_text = std::getenv("LFM2_TEXT_FILE");
    if (configured_text == nullptr) {
      std::cerr << "SKIP lfm2_prefill_test: LFM2_TEXT_FILE is not set\n";
      return 77;
    }
    std::string text;
    try {
      text = ReadFile(configured_text);
    } catch (const std::exception&) {
      std::cerr << "SKIP lfm2_prefill_test: LFM2_TEXT_FILE is not set\n";
      return 77;
    }
    // LFM2: repeat the text so the tokenization reaches 6000 ids.
    std::string big;
    for (int r = 0; r < 8; ++r)
      big += text;
    const auto& tokenizer = executor->GetTokenizer();
    superroc::tokenization::TokenizerOptions options;
    options.add_bos = tokenizer.AddBosToken();
    auto all = tokenizer.Encode(big, options);
    std::cout << "readme_bytes=" << text.size() << " tokens=" << all.size()
              << " add_bos=" << (options.add_bos ? 1 : 0) << '\n';
    if (all.size() < 6000) {
      std::cerr << "FAIL only " << all.size() << " tokens, want >= 6000\n";
      return 1;
    }

    const std::vector<std::size_t> lens = {2, 17, 512, 3000, 6000};
    bool all_ok = true;
    for (const std::size_t L : lens) {
      std::vector<superroc::tokenization::TokenId> prompt(all.begin(),
                                                          all.begin() + L);
      // LFM2 (i) batched path.
      double tok_per_s = 0;
      executor->Reset();
      superroc::tokenization::TokenId tok_a = 0;
      if (L == 3000 || L == 6000) {
        (void)hipDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        tok_a = executor->ForwardPromptBatch(prompt, 0, true);
        (void)hipDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        tok_per_s = static_cast<double>(L) / (sec > 0 ? sec : 1e-9);
      } else {
        tok_a = executor->ForwardPromptBatch(prompt, 0, true);
      }
      const std::vector<float> logits_a(executor->CopyLastLogits().begin(),
                                        executor->CopyLastLogits().end());
      std::vector<superroc::tokenization::TokenId> cont_a;
      cont_a.reserve(32);
      superroc::tokenization::TokenId cur = tok_a;
      for (int s = 0; s < 32; ++s) {
        const std::uint32_t pos = static_cast<std::uint32_t>(L + s);
        cur = executor->ForwardToken(cur, pos, true);
        cont_a.push_back(cur);
      }

      // LFM2 (ii) reference token loop.
      executor->Reset();
      superroc::tokenization::TokenId tok_b = 0;
      for (std::size_t i = 0; i < L; ++i) {
        const bool last = (i + 1 == L);
        tok_b = executor->ForwardToken(prompt[i], static_cast<std::uint32_t>(i),
                                       last);
      }
      const std::vector<float> logits_b(executor->CopyLastLogits().begin(),
                                        executor->CopyLastLogits().end());
      std::vector<superroc::tokenization::TokenId> cont_b;
      cont_b.reserve(32);
      cur = tok_b;
      for (int s = 0; s < 32; ++s) {
        const std::uint32_t pos = static_cast<std::uint32_t>(L + s);
        cur = executor->ForwardToken(cur, pos, true);
        cont_b.push_back(cur);
      }

      const bool eq = (tok_a == tok_b);
      double max_abs = 0;
      for (std::size_t i = 0; i < logits_a.size(); ++i) {
        max_abs = std::max(
            max_abs, static_cast<double>(std::fabs(logits_a[i] - logits_b[i])));
      }
      const auto top_a = Top5(logits_a);
      const auto top_b = Top5(logits_b);
      int agree = 0;
      int first_diff = -1;
      for (int s = 0; s < 32; ++s) {
        if (cont_a[s] == cont_b[s]) {
          ++agree;
        } else if (first_diff < 0) {
          first_diff = s;
        }
      }
      std::cout << "len=" << L << " A=" << tok_a << " B=" << tok_b
                << " eq=" << (eq ? 1 : 0) << " max_abs=" << max_abs << '\n';
      std::cout << "  topA:";
      for (const auto id : top_a)
        std::cout << ' ' << id;
      std::cout << "\n  topB:";
      for (const auto id : top_b)
        std::cout << ' ' << id;
      std::cout << "\n  cont_agree=" << agree << "/32";
      if (first_diff >= 0) {
        std::cout << " first_diff=" << first_diff << " A=" << cont_a[first_diff]
                  << " B=" << cont_b[first_diff];
      }
      std::cout << '\n';
      if (L == 3000 || L == 6000) {
        std::cout << "  prefill_tok_per_s=" << tok_per_s << '\n';
      }
      if (!eq || agree < 30)
        all_ok = false;
    }
    if (!all_ok) {
      std::cerr << "FAIL lfm2_prefill_test mismatch\n";
      return 1;
    }
    std::cout << "lfm2_prefill_test ALL PASS\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "FAIL " << ex.what() << '\n';
    return 1;
  }
}
