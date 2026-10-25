// FLASHNEXT S8 / G3: llama-perplexity's protocol on the gufo executor.
//
// Matches rocmfpx tools/perplexity/perplexity.cpp perplexity() (ppl_stride == 0):
//   * tokenize the whole file the way common_tokenize(ctx, text, true) does
//     (add_special, parse_special = false): one leading BOS only when
//     tokenizer.ggml.add_bos_token is set;
//   * cut into chunks of n_ctx tokens and drop the short tail;
//   * clear the KV cache before each chunk;
//   * when add_bos, the forward sees BOS in place of that chunk's first token
//     (the original token is restored for scoring, same as the fork);
//   * score logits at positions first..n_ctx-2 (first = n_ctx/2) against
//     tokens first+1..n_ctx-1, i.e. n_ctx-first-1 tokens per chunk;
//   * print cumulative [i] PPL.
//
// Usage: qfn_ppl <model.gguf> <corpus.txt> [n_ctx] [max_chunks]
// n_ctx defaults to 4096. max_chunks 0 (default) scores every full chunk.
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
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace {

double ScoreToken(std::span<const float> view, std::int32_t target,
                   std::size_t* argmax_hits) {
  if (view.empty() ||
      !std::all_of(view.begin(), view.end(),
                   [](float v) { return std::isfinite(v); }))
    throw std::runtime_error("non-finite logits");
  if (target < 0 || static_cast<std::size_t>(target) >= view.size())
    throw std::runtime_error("target token outside vocab");
  double max_logit = -1e300;
  for (const float v : view)
    max_logit = std::max(max_logit, static_cast<double>(v));
  double sum_exp = 0.0;
  for (const float v : view)
    sum_exp += std::exp(static_cast<double>(v) - max_logit);
  const double nll =
      max_logit + std::log(sum_exp) - static_cast<double>(view[target]);
  std::size_t argmax = 0;
  for (std::size_t i = 1; i < view.size(); ++i)
    if (view[i] > view[argmax])
      argmax = i;
  *argmax_hits += argmax == static_cast<std::size_t>(target);
  return nll;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      std::cerr << "usage: qfn_ppl <model.gguf> <corpus.txt> [n_ctx] "
                   "[max_chunks]\n";
      return 2;
    }
    const std::uint32_t n_ctx =
        argc > 3 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 4096;
    const std::uint32_t max_chunks =
        argc > 4 ? static_cast<std::uint32_t>(std::stoul(argv[4])) : 0;
    if (n_ctx < 2)
      throw std::runtime_error("n_ctx must be >= 2");
    const std::size_t first = static_cast<std::size_t>(n_ctx) / 2;
    const std::size_t scored_per =
        static_cast<std::size_t>(n_ctx) - first - 1;

    std::ifstream in(argv[2]);
    if (!in)
      throw std::runtime_error(std::string("cannot open ") + argv[2]);
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();

    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
    if (reader == nullptr)
      throw std::runtime_error(error);
    const bool add_bos =
        reader->GetMetadataBool("tokenizer.ggml.add_bos_token").value_or(false);
    const auto bos_meta =
        reader->GetMetadataUint32("tokenizer.ggml.bos_token_id");
    reader.reset();

    namespace qfn = gufo::models::qwen38_flash_next;
    auto model = qfn::Model::Load(
        argv[1], qfn::ModelOptions{.max_context = n_ctx}, &error);
    if (model == nullptr)
      throw std::runtime_error(error);
    const auto bos = static_cast<std::int32_t>(
        bos_meta.value_or(model->tokenizer().GetBosTokenId()));
    gufo::tokenization::TokenizerOptions opt;
    opt.add_bos = add_bos;
    opt.add_eos = false;
    opt.parse_special_tokens = false;
    const auto encoded = model->tokenizer().Encode(text, opt);
    std::vector<std::int32_t> tokens;
    tokens.reserve(encoded.size());
    for (const auto id : encoded)
      tokens.push_back(static_cast<std::int32_t>(id));

    if (tokens.size() < static_cast<std::size_t>(n_ctx) * 2)
      throw std::runtime_error(
          "corpus tokenizes to fewer than 2*n_ctx tokens (got " +
          std::to_string(tokens.size()) + ")");
    std::size_t n_chunk = tokens.size() / n_ctx;
    if (max_chunks > 0)
      n_chunk = std::min<std::size_t>(n_chunk, max_chunks);
    std::cerr << "tokens=" << tokens.size() << " n_ctx=" << n_ctx
              << " n_chunk=" << n_chunk << " first=" << first
              << " scored_per_chunk=" << scored_per
              << " add_bos=" << (add_bos ? 1 : 0) << " bos=" << bos
              << " token0=" << tokens[0] << "\n";

    auto session = model->CreateSession(
        gufo::core::SessionMode::kAutoregressive, n_ctx, &error);
    if (session == nullptr)
      throw std::runtime_error(error);

    double nll = 0.0;
    std::size_t count = 0;
    std::size_t argmax_hits = 0;
    std::size_t vocab = 0;
    for (std::size_t chunk = 0; chunk < n_chunk; ++chunk) {
      const std::size_t start = chunk * n_ctx;
      // Forward sees BOS at position 0 when add_bos; scoring uses the
      // original ids (the fork restores the token before process_logits).
      std::vector<std::int32_t> seen(
          tokens.begin() + static_cast<std::ptrdiff_t>(start),
          tokens.begin() + static_cast<std::ptrdiff_t>(start + n_ctx));
      if (add_bos)
        seen[0] = bos;
      // Feed tokens 0..n_ctx-2. Logits after consuming index p predict
      // token p+1. The first scored position is `first`.
      session->Reset();
      if (!session->Sync(std::span<const std::int32_t>(seen).subspan(0, first + 1),
                         &error))
        throw std::runtime_error("chunk prefill failed: " + error);

      double chunk_nll = 0.0;
      std::size_t chunk_scored = 0;
      std::size_t chunk_hits = 0;
      auto take = [&](std::int32_t target) {
        const auto view = session->Logits();
        vocab = view.size();
        const double one = ScoreToken(view, target, &chunk_hits);
        chunk_nll += one;
        ++chunk_scored;
        if ((count + chunk_scored) % 256 == 0)
          std::cerr << "chunk " << (chunk + 1) << " scored "
                    << (count + chunk_scored) << "\n";
      };
      // i = 0: logits at position `first` vs original token first+1.
      take(tokens[start + first + 1]);
      for (std::size_t pos = first + 1; pos + 1 < n_ctx; ++pos) {
        if (!session->Evaluate(seen[pos], &error))
          throw std::runtime_error("teacher-forced step failed: " + error);
        take(tokens[start + pos + 1]);
      }
      if (chunk_scored != scored_per)
        throw std::runtime_error(
            "chunk scored " + std::to_string(chunk_scored) +
            " tokens, expected " + std::to_string(scored_per));
      if (session->Tokens().size() != n_ctx - 1)
        throw std::runtime_error(
            "chunk consumed " + std::to_string(session->Tokens().size()) +
            " tokens, expected " + std::to_string(n_ctx - 1));
      nll += chunk_nll;
      count += chunk_scored;
      argmax_hits += chunk_hits;
      const double ppl_chunk =
          std::exp(chunk_nll / static_cast<double>(chunk_scored));
      const double ppl_cum = std::exp(nll / static_cast<double>(count));
      std::cerr << "chunk=" << (chunk + 1) << " start=" << start
                << " scored=" << chunk_scored << " nll=" << chunk_nll
                << " ppl=" << ppl_chunk << " ppl_cum=" << ppl_cum
                << " argmax_hits=" << chunk_hits << "\n";
      std::printf("[%zu]%.4f,", chunk + 1, ppl_cum);
      std::fflush(stdout);
    }
    const double ppl = std::exp(nll / static_cast<double>(count));
    std::printf("\nFinal estimate: PPL = %.4f\n", ppl);
    std::cout << "PPL " << ppl << " n_ctx=" << n_ctx << " n_chunk=" << n_chunk
              << " scored_tokens=" << count << " vocab=" << vocab
              << " argmax_hits=" << argmax_hits << " add_bos="
              << (add_bos ? 1 : 0) << " model=" << argv[1]
              << " corpus=" << argv[2] << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "qfn_ppl: " << e.what() << "\n";
    return 1;
  }
}
