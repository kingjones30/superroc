// GLM: GLIMMER S4 probe. Reproduces one greedy trajectory through the same
// render and the same kernels the serve path uses, and dumps per-step token
// ids with their top-k logprobs so the sequence can be aligned with the
// fork's llama-server logprobs (s3-eom-fork.json) token for token.
// Read-only: one GGUF, one prompt, no writes outside stdout.
// usage: glimmer_s4_probe <gguf> [max_new_tokens] [prompt_text]
// Exit 0 pass, 1 fail, 2 usage, 77 skipped (no file named).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glimmer/glimmer_chat.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

constexpr std::size_t kTopK = 8;

std::string Quote(std::string_view text) {
  std::string out = "'";
  for (const char raw : text) {
    const auto byte = static_cast<unsigned char>(raw);
    if (raw == '\\') {
      out.append("\\\\");
    } else if (raw == '\'') {
      out.append("\\'");
    } else if (raw == '\n') {
      out.append("\\n");
    } else if (raw == '\r') {
      out.append("\\r");
    } else if (raw == '\t') {
      out.append("\\t");
    } else if (byte < 0x20 || byte == 0x7f) {
      char buffer[8];
      std::snprintf(buffer, sizeof(buffer), "\\x%02x", byte);
      out.append(buffer);
    } else {
      out.push_back(raw);
    }
  }
  out.push_back('\'');
  return out;
}

struct RankedToken {
  float logprob{0.0F};
  std::uint32_t id{0};
};

// Log-softmax over the scaled, softcapped logits, then the k best entries.
std::vector<RankedToken> TopK(const std::vector<float>& logits, std::size_t k,
                              float* logsumexp_out) {
  std::vector<RankedToken> ranked;
  if (logits.empty()) {
    return ranked;
  }
  const float peak = *std::max_element(logits.begin(), logits.end());
  double total = 0.0;
  for (const float value : logits) {
    total += std::exp(static_cast<double>(value - peak));
  }
  const float logsumexp = peak + static_cast<float>(std::log(total));
  if (logsumexp_out != nullptr) {
    *logsumexp_out = logsumexp;
  }
  const std::size_t take = std::min(k, logits.size());
  std::vector<std::uint32_t> order(logits.size());
  for (std::size_t index = 0; index < logits.size(); ++index) {
    order[index] = static_cast<std::uint32_t>(index);
  }
  std::partial_sort(
      order.begin(), order.begin() + static_cast<std::ptrdiff_t>(take),
      order.end(), [&logits](std::uint32_t lhs, std::uint32_t rhs) {
        return logits[lhs] > logits[rhs];
      });
  ranked.reserve(take);
  for (std::size_t index = 0; index < take; ++index) {
    const std::uint32_t id = order[index];
    ranked.push_back(RankedToken{logits[id] - logsumexp, id});
  }
  return ranked;
}

int RunProbe(const char* path, std::size_t max_new_tokens,
             const std::string& prompt_text) {
  try {
    std::string err;
    auto opened = gufo::core::GgufReader::OpenFile(path, &err);
    if (opened == nullptr) {
      std::cerr << "SKIP glimmer_s4_probe: " << path << " unreadable: " << err
                << '\n';
      return 77;
    }
    std::shared_ptr<const gufo::core::GgufReader> reader(std::move(opened));

    gufo::hip::QwenExecutionPolicy policy;
    policy.kv_cache_storage = gufo::hip::QwenKvCacheStorage::kFp32;
    policy.fuse_qk_norm_rope_kv = false;
    auto executor =
        gufo::hip::QwenGpuExecutor::CreateFromGguf(reader, &err, 32768, policy);
    if (executor == nullptr) {
      std::cerr << "FAIL load " << err << '\n';
      return 1;
    }
    const auto& config = executor->GetConfig();
    if (!config.attention_only || config.architecture != "muse-glimmer") {
      std::cerr << "FAIL not muse-glimmer attention-only\n";
      return 1;
    }
    const auto& tokenizer = executor->GetTokenizer();

    const std::vector<gufo::tokenization::ChatMessage> messages = {
        gufo::tokenization::ChatMessage(gufo::tokenization::ChatRole::kUser,
                                        prompt_text)};
    gufo::tokenization::ChatTemplateOptions options;
    options.reasoning_strength = "low";
    options.add_generation_prompt = true;
    std::string render_error;
    const auto rendered = gufo::tokenization::GlimmerRender(
        tokenizer, messages, {}, options, &render_error);
    if (!rendered.has_value()) {
      std::cerr << "FAIL render " << render_error << '\n';
      return 1;
    }
    gufo::tokenization::TokenizerOptions tokenizer_options;
    tokenizer_options.add_bos = false;
    tokenizer_options.add_eos = false;
    tokenizer_options.parse_special_tokens = true;
    const std::vector<gufo::tokenization::TokenId> prompt =
        tokenizer.Encode(*rendered, tokenizer_options);
    if (prompt.empty()) {
      std::cerr << "FAIL empty prompt\n";
      return 1;
    }

    std::cout << "PROBE_RENDERED " << Quote(*rendered) << '\n';
    std::cout << "PROBE_PROMPT n=" << prompt.size() << " ids=";
    for (const auto id : prompt) {
      std::cout << id << ",";
    }
    std::cout << '\n';

    const std::size_t vocab_size = config.vocab_size;
    auto emit_step = [&](std::size_t step, gufo::tokenization::TokenId token) {
      const auto logits_span = executor->CopyLastLogits();
      const std::vector<float> logits(logits_span.begin(), logits_span.end());
      float logsumexp = 0.0F;
      const auto ranked = TopK(logits, kTopK, &logsumexp);
      std::cout << "STEP " << step << " id=" << token
                << " piece=" << Quote(tokenizer.DecodeToken(token)) << " lp=";
      if (logits.size() == vocab_size && !ranked.empty() &&
          ranked.front().id == token) {
        std::cout << ranked.front().logprob;
      } else {
        // The forward pass must leave the argmax at the vocabulary head.
        std::cout << "MISMATCH(argmax="
                  << (ranked.empty() ? 0 : ranked.front().id) << ")";
      }
      std::cout << " top=";
      for (const RankedToken& entry : ranked) {
        std::cout << entry.id << ":" << Quote(tokenizer.DecodeToken(entry.id))
                  << ":" << entry.logprob << " ";
      }
      std::cout << '\n';
    };

    gufo::tokenization::TokenId next =
        executor->ForwardPromptBatch(prompt, 0, /*compute_logits=*/true);
    std::size_t position = prompt.size();
    std::size_t emitted = 0;
    emit_step(emitted, next);
    ++emitted;
    while (emitted < max_new_tokens) {
      if (tokenizer.IsStopToken(next)) {
        std::cout << "STOP_TOKEN " << next << " at_step " << emitted << '\n';
        break;
      }
      next = executor->ForwardToken(next, static_cast<std::uint32_t>(position),
                                    /*compute_logits=*/true);
      ++position;
      emit_step(emitted, next);
      ++emitted;
    }
    std::cout << "PROBE_DONE steps=" << emitted << '\n';
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "FAIL " << ex.what() << '\n';
    return 1;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 4) {
    std::cerr << "usage: glimmer_s4_probe <gguf> [max_new_tokens] "
                 "[prompt_text]\n";
    return 2;
  }
  // The muse-glimmer file is named on the command line or, under ctest, by
  // GLIMMER_S4_GGUF. A file that is neither named nor present skips (ctest
  // reports Skipped, exit 77) instead of failing, so a machine without the
  // weights still builds green.
  const char* configured = argc >= 2 ? argv[1] : std::getenv("GLIMMER_S4_GGUF");
  if (configured == nullptr) {
    std::cerr << "SKIP glimmer_s4_probe: GLIMMER_S4_GGUF is not set\n";
    return 77;
  }
  const std::size_t max_new_tokens =
      argc >= 3 ? static_cast<std::size_t>(std::stoul(argv[2])) : 80U;
  const std::string prompt_text =
      argc >= 4 ? argv[3] : "Summarize in two sentences why the sky is blue.";
  return RunProbe(configured, max_new_tokens, prompt_text);
}
