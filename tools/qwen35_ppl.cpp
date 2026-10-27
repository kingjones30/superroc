// lqw:qwen35_ppl. Teacher-forced perplexity on the GPU executor, scored the
// way llama.cpp tools/perplexity/perplexity.cpp scores a chunk:
//   first = n_ctx/2
//   logits at position p (p >= first) score token p+1
//   scored count per chunk is n_ctx - first - 1
//   when tokenizer.ggml.add_bos_token is set, chunk token 0 is replaced
//   with the BOS id for the forward (the corpus token is not what is scored)
// Prints the running PPL after each chunk, which is what llama-perplexity
// prints as [1]…,[2]…. Usage: qwen35_ppl <model.gguf> <corpus.txt> [n_ctx]
// SQ18 dual mode: qwen35_ppl <model.gguf> <corpus.txt> [n_ctx] dual
// [max_chunks] [n_prefill]: per chunk, score every position through the
// decode path exactly as the gate loop does, then re-score n_prefill evenly
// spaced positions (default 32) through the pure prefill path (Reset +
// one-shot prefix prefill per position, batch >= first+1 so the F16 route
// engages), and report paired agreement. Per-position DPOS lines on stderr
// carry the paired NLLs for the d/p standard error. Tool code only.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

namespace {

double ScoreToken(std::span<const float> view,
                  superroc::tokenization::TokenId target, std::size_t* argmax_hit) {
  if (view.empty() || target >= view.size() ||
      !std::all_of(view.begin(), view.end(),
                   [](float v) { return std::isfinite(v); }))
    throw std::runtime_error("non-finite or short logits");
  double max_logit = -1e300;
  for (const float v : view)
    max_logit = std::max(max_logit, static_cast<double>(v));
  double sum_exp = 0.0;
  for (const float v : view)
    sum_exp += std::exp(static_cast<double>(v) - max_logit);
  const double nll = max_logit + std::log(sum_exp) -
                     static_cast<double>(view[target]);
  std::size_t argmax = 0;
  for (std::size_t i = 1; i < view.size(); ++i)
    if (view[i] > view[argmax])
      argmax = i;
  if (argmax_hit != nullptr)
    *argmax_hit += argmax == target ? 1 : 0;
  return nll;
}

// SQ18: scored-position record shared by both paths (paired comparison).
struct ScoredPos {
  double nll = 0.0;
  double logp_true = 0.0;
  std::size_t top = 0;
};

ScoredPos ScoreView(std::span<const float> view,
                    superroc::tokenization::TokenId target) {
  if (view.empty() || target >= view.size() ||
      !std::all_of(view.begin(), view.end(),
                   [](float v) { return std::isfinite(v); }))
    throw std::runtime_error("non-finite or short logits");
  double max_logit = -1e300;
  for (const float v : view)
    max_logit = std::max(max_logit, static_cast<double>(v));
  double sum_exp = 0.0;
  for (const float v : view)
    sum_exp += std::exp(static_cast<double>(v) - max_logit);
  const double lse = max_logit + std::log(sum_exp);
  ScoredPos out;
  out.nll = lse - static_cast<double>(view[target]);
  out.logp_true = static_cast<double>(view[target]) - lse;
  out.top = 0;
  for (std::size_t i = 1; i < view.size(); ++i)
    if (view[i] > view[out.top])
      out.top = i;
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      std::cerr << "usage: qwen35_ppl <model.gguf> <corpus.txt> [n_ctx] "
                   "[dual [max_chunks [n_prefill]]]\n";
      return 2;
    }
    const std::size_t n_ctx =
        argc > 3 ? static_cast<std::size_t>(std::stoul(argv[3])) : 4096;
    if (n_ctx < 2 || (n_ctx % 2) != 0)
      throw std::runtime_error("n_ctx must be even and at least 2");
    const std::size_t first = n_ctx / 2;
    // SQ18: mode switch. Absent (or anything but "dual") keeps the gate loop
    // below byte-identical; "dual" adds the prefill-path pass + comparison.
    const bool dual = argc > 4 && std::string(argv[4]) == "dual";
    std::size_t max_chunks = 0;
    if (argc > 5)
      max_chunks = static_cast<std::size_t>(std::stoul(argv[5]));
    std::size_t n_prefill = 32;
    if (argc > 6)
      n_prefill = static_cast<std::size_t>(std::stoul(argv[6]));
    if (dual && n_prefill == 0)
      throw std::runtime_error("n_prefill must be >= 1 in dual mode");

    std::ifstream in(argv[2]);
    if (!in)
      throw std::runtime_error(std::string("cannot open ") + argv[2]);
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();

    std::string error;
    auto owner = superroc::core::GgufReader::OpenFile(argv[1], &error);
    if (owner == nullptr)
      throw std::runtime_error(error);
    bool add_bos = false;
    const char* add_bos_src = "absent-default-false";
    if (auto flag = owner->GetMetadataBool("tokenizer.ggml.add_bos_token")) {
      add_bos = *flag;
      add_bos_src = "bool";
    } else if (auto number =
                   owner->GetMetadataUint32("tokenizer.ggml.add_bos_token")) {
      add_bos = *number != 0;
      add_bos_src = "u32";
    }

    auto executor = superroc::hip::QwenGpuExecutor::CreateFromGguf(
        std::shared_ptr<const superroc::core::GgufReader>(std::move(owner)),
        &error, 12288);
    if (executor == nullptr)
      throw std::runtime_error(error);
    const auto bos = executor->GetTokenizer().GetBosTokenId();
    if (add_bos && bos == superroc::tokenization::kInvalidTokenId)
      throw std::runtime_error("add_bos is set but the vocab has no BOS id");
    const auto tokens = executor->GetTokenizer().Encode(text);
    std::size_t n_chunk = tokens.size() / n_ctx;
    if (n_chunk == 0)
      throw std::runtime_error("corpus is shorter than one n_ctx chunk");
    if (max_chunks > 0)
      n_chunk = std::min(n_chunk, max_chunks);
    std::cerr << "tokens=" << tokens.size() << " n_ctx=" << n_ctx
              << " chunks=" << n_chunk << " first=" << first
              << " add_bos=" << (add_bos ? 1 : 0) << " (" << add_bos_src
              << ") bos=" << bos << (dual ? " mode=dual" : " mode=gate")
              << "\n";

    const std::size_t scored_per = n_ctx - first - 1;
    if (dual && n_prefill > scored_per)
      throw std::runtime_error("n_prefill exceeds scored positions per chunk");

    double nll = 0.0;
    std::size_t scored = 0, argmax_hits = 0, vocab = 0;
    // SQ18 dual accumulators (paired positions only for the _p side).
    double nll_d = 0.0, nll_p = 0.0, abs_dlogp = 0.0;
    std::size_t scored_paired = 0, agree = 0, hits_d = 0, hits_p = 0;
    for (std::size_t chunk = 0; chunk < n_chunk; ++chunk) {
      executor->Reset();
      std::vector<superroc::tokenization::TokenId> window(
          tokens.begin() + static_cast<std::ptrdiff_t>(chunk * n_ctx),
          tokens.begin() + static_cast<std::ptrdiff_t>((chunk + 1) * n_ctx));
      if (add_bos)
        window[0] = bos;
      if (!dual) {
        // SQ18 step 3: phase timing on stderr only (stdout unchanged).
        const auto t0 = std::chrono::steady_clock::now();
        (void)executor->ForwardPromptBatch(std::span<const superroc::tokenization::TokenId>(
            window.data(), first + 1));
        const auto t1 = std::chrono::steady_clock::now();
        double chunk_nll = 0.0;
        std::size_t chunk_scored = 0;
        for (std::size_t pos = first; pos + 1 < n_ctx; ++pos) {
          if (pos > first) {
            (void)executor->ForwardToken(window[pos],
                                         static_cast<std::uint32_t>(pos));
          }
          const auto view = executor->CopyLastLogits();
          vocab = view.size();
          const double one = ScoreToken(view, window[pos + 1], &argmax_hits);
          chunk_nll += one;
          nll += one;
          ++chunk_scored;
          ++scored;
        }
        const auto t2 = std::chrono::steady_clock::now();
        std::cout << "[" << (chunk + 1) << "]"
                  << std::exp(nll / static_cast<double>(scored))
                  << " chunk_ppl="
                  << std::exp(chunk_nll / static_cast<double>(chunk_scored))
                  << " chunk_scored=" << chunk_scored << "\n"
                  << std::flush;
        const double pre_s =
            std::chrono::duration<double>(t1 - t0).count();
        const double dec_s =
            std::chrono::duration<double>(t2 - t1).count();
        std::cerr << "PHASE mode=gate chunk=" << (chunk + 1)
                  << " prefill_s=" << pre_s
                  << " decode_steps=" << (chunk_scored - 1)
                  << " decode_step_s=" << dec_s << "\n";
        continue;
      }
      // SQ18 dual mode. Decode pass first: identical call sequence to the
      // gate loop (same prefill + same ForwardToken steps), so its PPL must
      // reproduce the gate number; per-position records are kept for pairing.
      std::vector<ScoredPos> dec(scored_per);
      (void)executor->ForwardPromptBatch(std::span<const superroc::tokenization::TokenId>(
          window.data(), first + 1));
      double chunk_nll_d = 0.0;
      for (std::size_t i = 0; i < scored_per; ++i) {
        const std::size_t pos = first + i;
        if (pos > first) {
          (void)executor->ForwardToken(window[pos],
                                       static_cast<std::uint32_t>(pos));
        }
        const auto view = executor->CopyLastLogits();
        vocab = view.size();
        dec[i] = ScoreView(view, window[pos + 1]);
        chunk_nll_d += dec[i].nll;
        if ((i + 1) % 256 == 0)
          std::cerr << "chunk " << (chunk + 1) << " decode " << (i + 1) << "/"
                    << scored_per << "\n";
      }
      // Prefill pass: n_prefill evenly spaced scored positions (endpoints
      // included), each Reset + one-shot prefix prefill so every GEMM runs
      // the prefill route (batch = pos+1 >= first+1 >= 1024 keeps F16).
      double chunk_nll_p = 0.0, chunk_nll_dp = 0.0, chunk_abs = 0.0;
      std::size_t chunk_agree = 0, chunk_hits_d = 0, chunk_hits_p = 0;
      for (std::size_t j = 0; j < n_prefill; ++j) {
        const std::size_t i = (n_prefill == 1)
                                  ? scored_per / 2
                                  : (j * (scored_per - 1)) / (n_prefill - 1);
        const std::size_t pos = first + i;
        executor->Reset();
        (void)executor->ForwardPromptBatch(std::span<const superroc::tokenization::TokenId>(
            window.data(), pos + 1));
        const auto view = executor->CopyLastLogits();
        const ScoredPos pre = ScoreView(view, window[pos + 1]);
        chunk_nll_p += pre.nll;
        nll_p += pre.nll;
        chunk_nll_dp += dec[i].nll;
        nll_d += dec[i].nll;
        chunk_abs += std::fabs(dec[i].logp_true - pre.logp_true);
        abs_dlogp += std::fabs(dec[i].logp_true - pre.logp_true);
        chunk_agree += dec[i].top == pre.top ? 1 : 0;
        agree += dec[i].top == pre.top ? 1 : 0;
        chunk_hits_d += dec[i].top == window[pos + 1] ? 1 : 0;
        hits_d += dec[i].top == window[pos + 1] ? 1 : 0;
        chunk_hits_p += pre.top == window[pos + 1] ? 1 : 0;
        hits_p += pre.top == window[pos + 1] ? 1 : 0;
        ++scored_paired;
        // SQ18 SPEC2 (a): per-paired-position NLLs for the d/p standard
        // error (delta method over per-position deltas). stderr keeps the
        // DUAL/stdout contract untouched.
        std::cerr << "DPOS chunk=" << (chunk + 1) << " j=" << j << " pos=" << pos
                  << " nll_d=" << dec[i].nll << " nll_p=" << pre.nll
                  << " top_d=" << dec[i].top << " top_p=" << pre.top
                  << " target=" << window[pos + 1] << "\n";
        if ((j + 1) % 8 == 0 || j + 1 == n_prefill)
          std::cerr << "chunk " << (chunk + 1) << " prefill " << (j + 1) << "/"
                    << n_prefill << "\n";
      }
      nll += chunk_nll_d;
      scored += scored_per;
      std::cout << "[dual " << (chunk + 1) << "]"
                << " ppl_decode_full="
                << std::exp(chunk_nll_d / static_cast<double>(scored_per))
                << " ppl_decode_paired="
                << std::exp(chunk_nll_dp / static_cast<double>(n_prefill))
                << " ppl_prefill="
                << std::exp(chunk_nll_p / static_cast<double>(n_prefill))
                << " paired=" << n_prefill << " agree=" << chunk_agree
                << " mean_abs_dlogp="
                << (chunk_abs / static_cast<double>(n_prefill))
                << " hits_d=" << chunk_hits_d << " hits_p=" << chunk_hits_p
                << "\n"
                << std::flush;
    }
    if (!dual) {
      std::cout << "PPL " << std::exp(nll / static_cast<double>(scored))
                << " scored_tokens=" << scored << " vocab=" << vocab
                << " argmax_hits=" << argmax_hits << " n_ctx=" << n_ctx
                << " chunks=" << n_chunk << " add_bos=" << (add_bos ? 1 : 0)
                << " model=" << argv[1] << " corpus=" << argv[2] << "\n";
      return 0;
    }
    std::cout << "DUAL ppl_decode_full="
              << std::exp(nll / static_cast<double>(scored))
              << " ppl_decode_paired="
              << std::exp(nll_d / static_cast<double>(scored_paired))
              << " ppl_prefill="
              << std::exp(nll_p / static_cast<double>(scored_paired))
              << " scored_tokens=" << scored << " paired=" << scored_paired
              << " agree=" << agree << " agree_frac="
              << (static_cast<double>(agree) /
                  static_cast<double>(scored_paired))
              << " mean_abs_dlogp="
              << (abs_dlogp / static_cast<double>(scored_paired))
              << " argmax_hits_d=" << hits_d << " argmax_hits_p=" << hits_p
              << " vocab=" << vocab << " n_ctx=" << n_ctx
              << " chunks=" << n_chunk << " add_bos=" << (add_bos ? 1 : 0)
              << " model=" << argv[1] << " corpus=" << argv[2] << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "qwen35_ppl: " << e.what() << "\n";
    return 1;
  }
}
