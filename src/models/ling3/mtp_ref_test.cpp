// LS4 phase 1: drive the Ling-3.0 MTP draft head on the CPU oracle and
// MEASURE draft acceptance.
//
// The point is not that the oracle agrees with itself -- it is the only
// implementation running. The check that matters is that the MTP head, wired
// exactly as the fork wires it (speculative.cpp:1759-1930), reproduces the
// fork's black-box behaviour: a draft token that the main model then produces
// most of the time (Jay MEASURED 82-84 % on code, 73-75 % on prose today). A
// wrong norm, a transposed concat, or a residual added after the attention
// norm instead of before it all show up here as a collapsed acceptance rate.
//
// Run it nice and single-threaded, and never while a bench is on the GPU:
//   nice -n 19 env OMP_NUM_THREADS=8 ./build-ls4/mtp_ref_test <model.gguf> \
//       --prompt-file ~/ls/long-prompt.txt --tokens 40
//
// Everything printed is measured on the run you are looking at; there are no
// canned expectations anywhere in this file.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling3/chat_template.hpp"
#include "src/models/ling3/reference.hpp"
#include "src/models/ling3/tokenizer.hpp"
#include "src/models/ling3/weights.hpp"

namespace {

using superroc::models::ling3::ModelWeights;
using superroc::models::ling3::ReferenceModel;

[[noreturn]] void Die(const std::string& why) {
  std::fprintf(stderr, "mtp_ref_test: %s\n", why.c_str());
  std::exit(1);
}

/// Reads one whitespace-separated integer per line, ignoring blanks.
std::vector<std::int32_t> ReadTokens(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr)
    Die("cannot open token file " + path);
  std::vector<std::int32_t> out;
  char line[256];
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    char* end = nullptr;
    const long v = std::strtol(line, &end, 10);
    if (end != line)
      out.push_back(static_cast<std::int32_t>(v));
  }
  std::fclose(f);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const char* model_path = nullptr;
  std::string token_file;
  std::string prompt_text;
  std::string dump_path;
  std::uint32_t dump_ctx = 0;
  int n_tokens = 40;
  bool layer_out_norm = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--tokens" && i + 1 < argc) {
      n_tokens = std::atoi(argv[++i]);
    } else if (a == "--prompt-file" && i + 1 < argc) {
      token_file = argv[++i];
    } else if (a == "--prompt-text" && i + 1 < argc) {
      prompt_text = argv[++i];
    } else if (a == "--dump" && i + 1 < argc) {
      dump_path = argv[++i];
    } else if (a == "--dump-context" && i + 1 < argc) {
      dump_ctx = static_cast<std::uint32_t>(std::atoi(argv[++i]));
    } else if (a == "--layer-out-norm") {
      // A/B: normalise the MTP head's final hidden with
      // blk.42.layer_output_norm instead of the main output_norm. See
      // MTP_NOTES.md section 2.
      layer_out_norm = true;
    } else if (a.rfind("--", 0) == 0) {
      Die("unknown flag " + a);
    } else {
      model_path = argv[i];
    }
  }
  if (model_path == nullptr) {
    Die("usage: mtp_ref_test <model.gguf> (--prompt-file F | --prompt-text T) "
        "[--tokens N] [--layer-out-norm]");
  }
  if (token_file.empty() && prompt_text.empty()) {
    Die("one of --prompt-file / --prompt-text is required");
  }

  std::string err;
  // OpenFile resolves the "-0000N-of-0000M" split set itself; pass shard 1.
  auto reader = superroc::core::GgufReader::OpenFile(
      std::filesystem::path(model_path), &err);
  if (reader == nullptr)
    Die("cannot open the GGUF: " + err);
  auto weights = ModelWeights::Bind(*reader, &err);
  if (!weights.has_value())
    Die("ModelWeights::Bind: " + err);
  std::printf(
      "# block_count=%u num_layers=%u nextn_layers=%u head_norm_bound=%d\n",
      weights->config.block_count, weights->config.num_layers,
      weights->config.nextn_layers,
      weights->mtp_head.layer_out_norm.empty() ? 0 : 1);

  std::vector<std::int32_t> prompt;
  if (!token_file.empty()) {
    prompt = ReadTokens(token_file);
    if (prompt.empty())
      Die("prompt token file is empty");
  } else {
    // Render exactly what the server renders for a one-user-message chat
    // request, so acceptance is comparable with bench.sh's prompts.
    std::vector<superroc::models::ling3::chat::Message> msgs;
    msgs.push_back({"user", prompt_text, "", {}});
    auto rendered =
        superroc::models::ling3::chat::RenderChat(msgs, {}, true, true, &err);
    if (!rendered.has_value())
      Die("RenderChat: " + err);
    auto tok = superroc::models::ling3::LingTokenizer::FromGguf(*reader, &err);
    if (!tok.has_value())
      Die("LingTokenizer::FromGguf: " + err);
    prompt = tok->Encode(*rendered, /*add_bos=*/false, /*parse_special=*/true);
    if (prompt.empty())
      Die("encoded prompt is empty");
  }
  std::printf("# model=%s prompt_tokens=%zu decode=%d head_norm=%s\n",
              model_path, prompt.size(), n_tokens,
              layer_out_norm ? "blk.42.layer_output_norm" : "output_norm");

  const std::uint32_t ctx = dump_ctx != 0
                                ? dump_ctx
                                : (weights->config.context_length < 2048
                                       ? weights->config.context_length
                                       : 2048);
  ReferenceModel ref(*weights, ctx);
  ref.SetMtpUseLayerOutNorm(layer_out_norm);

  // --dump records one (pos, x_next, draft, h_prev[]) row per position over
  // the WHOLE sequence, prefill included, so mtp_gpu_check.cpp can replay the
  // identical inputs on the GPU and compare token for token. Only the CPU
  // oracle's INPUTS are dumped; the GPU run recomputes the drafts itself.
  //
  // Position p's h_prev is the hidden of the token at p-1, which is exactly
  // what ref.Hidden() holds immediately before the Step of token p. Position 0
  // has no predecessor and is not dumped (rows start at 1).
  std::FILE* dump = nullptr;
  std::size_t dump_rows = 0;
  if (!dump_path.empty()) {
    dump = std::fopen(dump_path.c_str(), "w");
    if (dump == nullptr)
      Die("cannot write " + dump_path);
    std::fprintf(dump, "# ls4 mtp cpu-oracle dump v1\n");
    std::fprintf(dump, "hidden %u\n", weights->config.hidden_size);
    std::fprintf(dump, "context %u\n", ctx);
    std::fprintf(dump, "head_norm %s\n",
                 layer_out_norm ? "layer_out_norm" : "output_norm");
    std::fprintf(dump, "rows %zu\n", prompt.size() + 1);
  }
  auto dump_row = [&](std::uint32_t pos, std::int32_t x_next,
                      const std::span<const float> h_prev, std::int32_t draft) {
    if (dump == nullptr)
      return;
    std::fprintf(dump, "%u %d %d", pos, x_next, draft);
    for (std::size_t i = 0; i < h_prev.size(); ++i) {
      std::fprintf(dump, " %.9g", static_cast<double>(h_prev[i]));
    }
    std::fprintf(dump, "\n");
    ++dump_rows;
  };

  std::vector<float> logits;
  auto argmax = [&logits]() {
    std::size_t best = 0;
    for (std::size_t i = 1; i < logits.size(); ++i) {
      if (logits[i] > logits[best])
        best = i;
    }
    return static_cast<std::int32_t>(best);
  };

  const std::size_t H = weights->config.hidden_size;
  // Positions 0..prompt.size()-1 are the prompt; from prompt.size() on they
  // are generated. One loop, because the MTP block's latent cache has to be
  // filled at EVERY position in order -- the head attends over positions
  // [0, pos], and a gap would make every later draft meaningless.
  const std::size_t total = prompt.size() + static_cast<std::size_t>(n_tokens);
  int n_compared = 0;
  int n_accepted = 0;
  std::vector<std::int32_t> drafts, targets;
  std::string drafted_text;
  std::string actual_text;

  for (std::size_t p = 0; p < total; ++p) {
    // `next` is the token that sits at position p: a prompt token while
    // prefilling, otherwise what the model generated last step.
    const std::int32_t next = p < prompt.size() ? prompt[p] : argmax();
    if (next < 0 ||
        static_cast<std::uint32_t>(next) >= weights->config.vocab_size) {
      Die("token id out of vocabulary at position " + std::to_string(p));
    }

    const std::uint32_t pos = ref.Position();
    if (pos != p)
      Die("position bookkeeping diverged");

    const bool is_decode = p >= prompt.size();
    std::int32_t draft = -1;
    std::vector<float> h_new(H);
    if (p > 0) {
      const std::span<const float> h_prev = ref.Hidden();
      if (h_prev.size() != H)
        Die("Hidden() has the wrong width");
      if (!ref.MtpDraft(next, h_prev, pos, &draft, h_new, &err)) {
        Die("MtpDraft: " + err);
      }
      if (draft < 0 ||
          static_cast<std::uint32_t>(draft) >= weights->config.vocab_size) {
        Die("MtpDraft produced an out-of-vocabulary token");
      }
      dump_row(pos, next, h_prev, draft);
    }

    if (!ref.Step(next, logits, &err))
      Die("decode: " + err);

    if (is_decode && p > 0) {
      // The draft made at position p predicted the token the model produces at
      // p+1, which is exactly this Step's argmax.
      const std::int32_t produced = argmax();
      ++n_compared;
      if (draft == produced)
        ++n_accepted;
      drafts.push_back(draft);
      targets.push_back(produced);
    }
  }

  for (const std::int32_t t : drafts)
    drafted_text += std::to_string(t) + ' ';
  for (const std::int32_t t : targets)
    actual_text += std::to_string(t) + ' ';

  if (dump != nullptr) {
    std::fclose(dump);
    std::printf("# dumped %zu rows to %s\n", dump_rows, dump_path.c_str());
  }

  const double acc = n_compared > 0 ? static_cast<double>(n_accepted) /
                                          static_cast<double>(n_compared)
                                    : 0.0;
  std::printf(
      "\nMTP_MEASURED compared=%d accepted=%d acceptance=%.1f%% head_norm=%s\n",
      n_compared, n_accepted, 100.0 * acc,
      layer_out_norm ? "blk.42.layer_output_norm" : "output_norm");
  std::printf("# drafts : %s\n", drafted_text.c_str());
  std::printf("# target : %s\n", actual_text.c_str());
  return 0;
}
