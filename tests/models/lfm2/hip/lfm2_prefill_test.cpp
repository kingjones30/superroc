// LFM2 batched prefill test (SPEC1, lane LFS). Compares ForwardPromptBatch //
// LFM2 against the token loop + 32 greedy steps. SKIP 77 without LFM2_A_GGUF.
// // LFM2 NOTE (deviation): SPEC1 step 4 names ForwardPromptChunk, which is
// private; // LFM2 the test uses the public ForwardPromptBatch, which forwards
// to // LFM2 ForwardLfm2PromptChunk in one chunk for these lengths (6000 splits
// // LFM2 4096+1904 at the 4096 max batch, still exercising the batched path).
// // LFM2

#include <hip/hip_runtime.h>  // LFM2

#include <algorithm>  // LFM2
#include <chrono>     // LFM2
#include <cmath>      // LFM2
#include <cstdlib>    // LFM2
#include <fstream>    // LFM2
#include <iostream>   // LFM2
#include <numeric>    // LFM2
#include <sstream>    // LFM2
#include <string>     // LFM2
#include <vector>     // LFM2

#include "src/core/gguf_reader.hpp"          // LFM2
#include "src/models/qwen/hip/executor.hpp"  // LFM2

namespace {  // LFM2

std::string ReadFile(const std::string& path) {                    // LFM2
  std::ifstream in(path, std::ios::binary);                        // LFM2
  if (!in) {                                                       // LFM2
    throw std::runtime_error(std::string("cannot open ") + path);  // LFM2
  }  // LFM2
  std::ostringstream ss;  // LFM2
  ss << in.rdbuf();       // LFM2
  return ss.str();        // LFM2
}  // LFM2

std::vector<std::size_t> Top5(const std::vector<float>& logits) {  // LFM2
  std::vector<std::size_t> idx(logits.size());                     // LFM2
  std::iota(idx.begin(), idx.end(), 0);                            // LFM2
  const std::size_t k = std::min<std::size_t>(5, idx.size());      // LFM2
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),       // LFM2
                    [&](std::size_t a, std::size_t b) {            // LFM2
                      return logits[a] > logits[b];                // LFM2
                    });                                            // LFM2
  idx.resize(k);                                                   // LFM2
  return idx;                                                      // LFM2
}  // LFM2

}  // namespace

int main(int argc, char** argv) {  // LFM2
  const char* configured =
      argc >= 2 ? argv[1] : std::getenv("LFM2_A_GGUF");               // LFM2
  if (configured == nullptr) {                                        // LFM2
    std::cerr << "SKIP lfm2_prefill_test: LFM2_A_GGUF is not set\n";  // LFM2
    return 77;                                                        // LFM2
  }  // LFM2
  const std::string path = configured;                               // LFM2
  try {                                                              // LFM2
    std::string err;                                                 // LFM2
    auto opened = superroc::core::GgufReader::OpenFile(path, &err);  // LFM2
    if (opened == nullptr) {                                         // LFM2
      std::cerr << "SKIP lfm2_prefill_test: " << path
                << " unreadable: " << err  // LFM2
                << '\n';                   // LFM2
      return 77;                           // LFM2
    }  // LFM2
    std::shared_ptr<const superroc::core::GgufReader> reader(
        std::move(opened));                     // LFM2
    superroc::hip::QwenExecutionPolicy policy;  // LFM2 (lfm2_ppl policy)
    policy.kv_cache_storage = superroc::hip::QwenKvCacheStorage::kFp32;  // LFM2
    policy.fuse_qk_norm_rope_kv = false;                                 // LFM2
    constexpr std::uint32_t kContext = 8192;                             // LFM2
    auto executor = superroc::hip::QwenGpuExecutor::CreateFromGguf(      // LFM2
        reader, &err, kContext, policy);                                 // LFM2
    if (executor == nullptr) {                                           // LFM2
      std::cerr << "FAIL load " << err << '\n';                          // LFM2
      return 1;                                                          // LFM2
    }  // LFM2
    const auto& config = executor->GetConfig();  // LFM2
    if (!config.lfm2) {                          // LFM2
      std::cerr << "FAIL not an lfm2 config\n";  // LFM2
      return 1;                                  // LFM2
    }  // LFM2
    const char* home = std::getenv("HOME");          // LFM2
    const std::string readme =                       // LFM2
        (home != nullptr ? home : ".") +             // LFM2
        std::string("/src/gufo-lfm-lfs/README.md");  // LFM2
    std::string text = ReadFile(readme);             // LFM2
    // LFM2: repeat the text so the tokenization reaches 6000 ids. // LFM2
    std::string big;  // LFM2
    for (int r = 0; r < 8; ++r)
      big += text;                                     // LFM2
    const auto& tokenizer = executor->GetTokenizer();  // LFM2
    superroc::tokenization::TokenizerOptions options;  // LFM2
    options.add_bos = tokenizer.AddBosToken();         // LFM2
    auto all = tokenizer.Encode(big, options);         // LFM2
    std::cout << "readme_bytes=" << text.size()
              << " tokens=" << all.size()                           // LFM2
              << " add_bos=" << (options.add_bos ? 1 : 0) << '\n';  // LFM2
    if (all.size() < 6000) {                                        // LFM2
      std::cerr << "FAIL only " << all.size()
                << " tokens, want >= 6000\n";  // LFM2
      return 1;                                // LFM2
    }  // LFM2

    const std::vector<std::size_t> lens = {2, 17, 512, 3000, 6000};  // LFM2
    bool all_ok = true;                                              // LFM2
    for (const std::size_t L : lens) {                               // LFM2
      std::vector<superroc::tokenization::TokenId> prompt(
          all.begin(),       // LFM2
          all.begin() + L);  // LFM2
      // LFM2 (i) batched path. // LFM2
      double tok_per_s = 0;                                           // LFM2
      executor->Reset();                                              // LFM2
      superroc::tokenization::TokenId tok_a = 0;                      // LFM2
      if (L == 3000 || L == 6000) {                                   // LFM2
        (void)hipDeviceSynchronize();                                 // LFM2
        const auto t0 = std::chrono::steady_clock::now();             // LFM2
        tok_a = executor->ForwardPromptBatch(prompt, 0, true);        // LFM2
        (void)hipDeviceSynchronize();                                 // LFM2
        const auto t1 = std::chrono::steady_clock::now();             // LFM2
        const double sec =                                            // LFM2
            std::chrono::duration<double>(t1 - t0).count();           // LFM2
        tok_per_s = static_cast<double>(L) / (sec > 0 ? sec : 1e-9);  // LFM2
      } else {                                                        // LFM2
        tok_a = executor->ForwardPromptBatch(prompt, 0, true);        // LFM2
      }  // LFM2
      const std::vector<float> logits_a(
          executor->CopyLastLogits().begin(),                         // LFM2
          executor->CopyLastLogits().end());                          // LFM2
      std::vector<superroc::tokenization::TokenId> cont_a;            // LFM2
      cont_a.reserve(32);                                             // LFM2
      superroc::tokenization::TokenId cur = tok_a;                    // LFM2
      for (int s = 0; s < 32; ++s) {                                  // LFM2
        const std::uint32_t pos = static_cast<std::uint32_t>(L + s);  // LFM2
        cur = executor->ForwardToken(cur, pos, true);                 // LFM2
        cont_a.push_back(cur);                                        // LFM2
      }  // LFM2

      // LFM2 (ii) reference token loop. // LFM2
      executor->Reset();                           // LFM2
      superroc::tokenization::TokenId tok_b = 0;   // LFM2
      for (std::size_t i = 0; i < L; ++i) {        // LFM2
        const bool last = (i + 1 == L);            // LFM2
        tok_b = executor->ForwardToken(prompt[i],  // LFM2
                                       static_cast<std::uint32_t>(i),
                                       last);  // LFM2
      }  // LFM2
      const std::vector<float> logits_b(
          executor->CopyLastLogits().begin(),                         // LFM2
          executor->CopyLastLogits().end());                          // LFM2
      std::vector<superroc::tokenization::TokenId> cont_b;            // LFM2
      cont_b.reserve(32);                                             // LFM2
      cur = tok_b;                                                    // LFM2
      for (int s = 0; s < 32; ++s) {                                  // LFM2
        const std::uint32_t pos = static_cast<std::uint32_t>(L + s);  // LFM2
        cur = executor->ForwardToken(cur, pos, true);                 // LFM2
        cont_b.push_back(cur);                                        // LFM2
      }  // LFM2

      const bool eq = (tok_a == tok_b);                    // LFM2
      double max_abs = 0;                                  // LFM2
      for (std::size_t i = 0; i < logits_a.size(); ++i) {  // LFM2
        max_abs =
            std::max(max_abs,                                       // LFM2
                     static_cast<double>(std::fabs(logits_a[i] -    // LFM2
                                                   logits_b[i])));  // LFM2
      }  // LFM2
      const auto top_a = Top5(logits_a);  // LFM2
      const auto top_b = Top5(logits_b);  // LFM2
      int agree = 0;                      // LFM2
      int first_diff = -1;                // LFM2
      for (int s = 0; s < 32; ++s) {      // LFM2
        if (cont_a[s] == cont_b[s]) {     // LFM2
          ++agree;                        // LFM2
        } else if (first_diff < 0) {      // LFM2
          first_diff = s;                 // LFM2
        }  // LFM2
      }  // LFM2
      std::cout << "len=" << L << " A=" << tok_a << " B=" << tok_b  // LFM2
                << " eq=" << (eq ? 1 : 0) << " max_abs=" << max_abs
                << '\n';       // LFM2
      std::cout << "  topA:";  // LFM2
      for (const auto id : top_a)
        std::cout << ' ' << id;  // LFM2
      std::cout << "\n  topB:";  // LFM2
      for (const auto id : top_b)
        std::cout << ' ' << id;                          // LFM2
      std::cout << "\n  cont_agree=" << agree << "/32";  // LFM2
      if (first_diff >= 0) {                             // LFM2
        std::cout << " first_diff=" << first_diff
                  << " A=" << cont_a[first_diff]   // LFM2
                  << " B=" << cont_b[first_diff];  // LFM2
      }  // LFM2
      std::cout << '\n';                                           // LFM2
      if (L == 3000 || L == 6000) {                                // LFM2
        std::cout << "  prefill_tok_per_s=" << tok_per_s << '\n';  // LFM2
      }  // LFM2
      if (!eq || agree < 30)
        all_ok = false;  // LFM2
    }  // LFM2
    if (!all_ok) {                                       // LFM2
      std::cerr << "FAIL lfm2_prefill_test mismatch\n";  // LFM2
      return 1;                                          // LFM2
    }  // LFM2
    std::cout << "lfm2_prefill_test ALL PASS\n";  // LFM2
    return 0;                                     // LFM2
  } catch (const std::exception& ex) {            // LFM2
    std::cerr << "FAIL " << ex.what() << '\n';    // LFM2
    return 1;                                     // LFM2
  }  // LFM2
}  // LFM2
