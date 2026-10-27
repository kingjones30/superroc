// GLM: perplexity in the fork llama-perplexity protocol (ppl_stride == 0).
// Chunks of n_ctx, non-overlapping. When add_bos is set, position 0 of each
// chunk is the BOS token. Score logit positions first..n_ctx-2 with
// first = n_ctx/2. Print the running perplexity after each chunk.
// Usage:
//   glimmer_ppl <model.gguf> <corpus.txt> [n_ctx] [n_chunks]
#include <algorithm>
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
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

double TokenNll(std::span<const float> logits, std::uint32_t target) {
  if (target >= logits.size()) {
    throw std::runtime_error("perplexity target is outside the vocabulary");
  }
  double max_logit = -1.0e300;
  for (const float value : logits) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("non-finite logit");
    }
    max_logit = std::max(max_logit, static_cast<double>(value));
  }
  double sum_exp = 0.0;
  for (const float value : logits) {
    sum_exp += std::exp(static_cast<double>(value) - max_logit);
  }
  return (max_logit + std::log(sum_exp)) - static_cast<double>(logits[target]);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      std::cerr << "usage: glimmer_ppl <model.gguf> <corpus.txt> "
                   "[n_ctx] [n_chunks]\n";
      return 2;
    }
    const int n_ctx = argc > 3 ? std::stoi(argv[3]) : 4096;
    const int n_chunks_req = argc > 4 ? std::stoi(argv[4]) : 2;
    if (n_ctx < 2 || n_chunks_req < 1) {
      throw std::runtime_error("n_ctx and n_chunks must be positive");
    }
    std::ifstream in(argv[2]);
    if (!in) {
      throw std::runtime_error(std::string("cannot open ") + argv[2]);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();

    std::string error;
    auto owner = superroc::core::GgufReader::OpenFile(argv[1], &error);
    if (owner == nullptr) {
      throw std::runtime_error(error);
    }
    superroc::hip::QwenExecutionPolicy policy;
    policy.kv_cache_storage = superroc::hip::QwenKvCacheStorage::kFp32;
    policy.fuse_qk_norm_rope_kv = false;
    policy.glimmer_fp16_kv = true;  // LGA: Glimmer fp16-KV plane
    auto executor = superroc::hip::QwenGpuExecutor::CreateFromGguf(
        std::shared_ptr<const superroc::core::GgufReader>(std::move(owner)),
        &error, static_cast<std::uint32_t>(n_ctx), policy);
    if (executor == nullptr) {
      throw std::runtime_error(error);
    }
    if (static_cast<int>(executor->GetMaxPromptBatch()) < n_ctx) {
      throw std::runtime_error(
          "n_ctx is wider than one prefill chunk; the score would not see "
          "every row");
    }
    const auto& tokenizer = executor->GetTokenizer();
    const bool add_bos = tokenizer.AddBosToken();
    superroc::tokenization::TokenizerOptions options;
    options.add_bos = add_bos;
    const auto tokens = tokenizer.Encode(text, options);
    const int n_chunk_max = static_cast<int>(tokens.size()) / n_ctx;
    const int n_chunk = std::min(n_chunks_req, n_chunk_max);
    const int first = n_ctx / 2;
    std::cerr << "head";
    for (int i = 0; i < 8 && i < static_cast<int>(tokens.size()); ++i) {
      std::cerr << " " << tokens[static_cast<std::size_t>(i)];
    }
    std::cerr << "\ntokens=" << tokens.size() << " add_bos=" << (add_bos ? 1 : 0)
              << " bos=" << tokenizer.GetBosTokenId() << " n_ctx=" << n_ctx
              << " first=" << first << " n_chunk_max=" << n_chunk_max
              << " n_chunk=" << n_chunk << " model=" << argv[1] << "\n";
    if (n_chunk < 1) {
      throw std::runtime_error("corpus is shorter than one n_ctx chunk");
    }

    double sum_nll = 0.0;
    double sum_nll2 = 0.0;
    std::size_t count = 0;
    for (int chunk = 0; chunk < n_chunk; ++chunk) {
      const std::size_t start = static_cast<std::size_t>(chunk) * n_ctx;
      std::vector<superroc::tokenization::TokenId> ids(
          tokens.begin() + static_cast<std::ptrdiff_t>(start),
          tokens.begin() + static_cast<std::ptrdiff_t>(start + n_ctx));
      if (add_bos) {
        ids[0] = tokenizer.GetBosTokenId();
      }
      executor->Reset();
      (void)executor->ForwardPromptBatch(ids, 0, false);
      for (int pos = first; pos <= n_ctx - 2; ++pos) {
        executor->MaterializeGlimmerRowLogits(static_cast<std::size_t>(pos));
        const double nll = TokenNll(executor->CopyLastLogits(),
                                     ids[static_cast<std::size_t>(pos + 1)]);
        sum_nll += nll;
        sum_nll2 += nll * nll;
        ++count;
      }
      const double mean = sum_nll / static_cast<double>(count);
      std::printf("[%d]%.4f,", chunk + 1, std::exp(mean));
      std::fflush(stdout);
    }
    std::printf("\n");
    double mean = sum_nll / static_cast<double>(count);
    double second = sum_nll2 / static_cast<double>(count) - mean * mean;
    const double ppl = std::exp(mean);
    double band = 0.0;
    if (second > 0.0 && count > 1) {
      second = std::sqrt(second / static_cast<double>(count - 1));
      band = second * ppl;
    }
    std::printf("Final estimate: PPL = %.4f +/- %.5f\n", ppl, band);
    std::cout << "PPL " << ppl << " scored_tokens=" << count
              << " n_chunk=" << n_chunk << " n_ctx=" << n_ctx
              << " corpus=" << argv[2] << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "glimmer_ppl: " << ex.what() << "\n";
    return 1;
  }
}
