// flintv: the batched Flint verification rows must match the single-token
// decode path bit for bit. For widths 2-8, w candidate tokens run through
// ForwardToken (reference) and ForwardVerificationChunk from the same
// SaveState point; every row must agree on argmax and memcmp on logits, and
// after CommitVerificationChunk accepts the first k rows the next decode
// token must still reproduce the reference row (KV of a partial accept --
// Flint keeps no recurrent state).
// usage: flint_verify_test [gguf]  (or FLINT_GGUF under ctest)
// Exit 0 pass, 1 fail, 77 skipped (no file named).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {

using Token = superroc::tokenization::TokenId;
using Executor = superroc::hip::QwenGpuExecutor;

constexpr int kSkipped = 77;
constexpr std::uint32_t kPromptSize = 24;
constexpr std::size_t kMaxWidth = 8;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool ByteEqual(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

std::size_t Argmax(std::span<const float> values) {
  return static_cast<std::size_t>(
      std::max_element(values.begin(), values.end()) - values.begin());
}

// First differing logit index, for the failure line.
std::size_t FirstDiff(std::span<const float> a, std::span<const float> b) {
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
      return i;
    }
  }
  return n;
}

void FailRow(std::size_t width, std::size_t row, std::span<const float> ref,
             std::span<const float> got) {
  const std::size_t idx = FirstDiff(ref, got);
  std::cout << "width=" << width << " row=" << row << " idx=" << idx << " ref=";
  if (idx < ref.size()) {
    std::cout << ref[idx];
  } else {
    std::cout << "none";
  }
  std::cout << " verify=";
  if (idx < got.size()) {
    std::cout << got[idx];
  } else {
    std::cout << "none";
  }
  std::cout << " size=" << ref.size() << "/" << got.size() << "\n";
}

}  // namespace

int main(int argc, const char* const* argv) {
  const char* model = argc > 1 ? argv[1] : std::getenv("FLINT_GGUF");
  if (model == nullptr) {
    std::cout << "SKIP flint_verify_test: FLINT_GGUF is unset\n";
    return kSkipped;
  }
  try {
    std::string error;
    auto owner = superroc::core::GgufReader::OpenFile(model, &error);
    Expect(owner != nullptr, error);
    auto shared = superroc::hip::QwenGpuModel::CreateFromGguf(
        std::shared_ptr<const superroc::core::GgufReader>(std::move(owner)),
        &error);
    Expect(shared != nullptr, error);
    // The policy the server runs Flint with.
    auto policy = superroc::hip::QwenExecutionPolicy::Production();
    policy.kv_cache_storage = superroc::hip::QwenKvCacheStorage::kFp32;
    policy.fuse_qk_norm_rope_kv = false;
    policy.glimmer_fp16_kv = true;
    auto executor = Executor::Create(shared, &error, 512, policy);
    Expect(executor != nullptr, error);
    const auto& config = executor->GetConfig();
    Expect(config.flint, "model is not flint");
    const std::size_t vocab = config.vocab_size;
    std::cout << "model layers=" << config.num_layers
              << " hidden=" << config.hidden_size << " vocab=" << vocab << "\n";

    std::vector<Token> prime(kPromptSize);
    std::vector<Token> cand(kMaxWidth);
    for (std::size_t i = 0; i < kPromptSize; ++i) {
      prime[i] = static_cast<Token>(1000 + i);
    }
    for (std::size_t i = 0; i < kMaxWidth; ++i) {
      cand[i] = static_cast<Token>(2000 + i);
    }

    bool failed = false;
    for (std::size_t w = 2; w <= kMaxWidth; ++w) {
      const auto tokens = std::span(cand).first(w);
      // Reference: one decode step per row from the same state.
      executor->Reset();
      (void)executor->ForwardPromptBatch(prime);
      executor->SaveState(kPromptSize);
      std::vector<std::vector<float>> ref_logits;
      std::vector<Token> ref_pred;
      for (std::size_t r = 0; r < w; ++r) {
        ref_pred.push_back(executor->ForwardToken(
            tokens[r], kPromptSize + static_cast<std::uint32_t>(r), true));
        const auto logits = executor->CopyLastLogits();
        Expect(logits.size() == vocab, "reference logits size");
        ref_logits.emplace_back(logits.begin(), logits.end());
      }
      executor->RestoreState();

      // Verify: one chunk over the same positions and state.
      executor->SaveState(kPromptSize);
      const auto pred =
          executor->ForwardVerificationChunk(tokens, kPromptSize, true);
      Expect(pred.size() == w, "verification width mismatch");
      const auto verify_logits = executor->GetVerificationLogits();
      Expect(verify_logits.size() == w * vocab, "verification logits size");
      bool width_ok = true;
      for (std::size_t r = 0; r < w; ++r) {
        const auto row = verify_logits.subspan(r * vocab, vocab);
        if (!ByteEqual(row, ref_logits[r])) {
          FailRow(w, r, ref_logits[r], row);
          width_ok = false;
          continue;
        }
        if (pred[r] != ref_pred[r] || pred[r] != Argmax(ref_logits[r])) {
          std::cout << "width=" << w << " row=" << r
                    << " argmax=" << Argmax(ref_logits[r])
                    << " decode=" << ref_pred[r] << " verify=" << pred[r]
                    << "\n";
          width_ok = false;
        }
      }
      executor->RestoreState();

      // Partial accept: commit the first k verified rows, then decode one
      // more token; it must reproduce reference row k exactly.
      for (std::size_t k = 1; width_ok && k < w; ++k) {
        executor->Reset();
        (void)executor->ForwardPromptBatch(prime);
        executor->SaveState(kPromptSize);
        (void)executor->ForwardVerificationChunk(tokens, kPromptSize, true);
        executor->RestoreState();
        executor->CommitVerificationChunk(tokens.first(k), kPromptSize);
        const Token next = executor->ForwardToken(
            tokens[k], kPromptSize + static_cast<std::uint32_t>(k), true);
        const auto logits = executor->CopyLastLogits();
        if (!ByteEqual(logits, ref_logits[k])) {
          FailRow(w, k, ref_logits[k], logits);
          width_ok = false;
          break;
        }
        if (next != ref_pred[k]) {
          std::cout << "width=" << w << " row=" << k
                    << " argmax=" << Argmax(ref_logits[k])
                    << " decode=" << ref_pred[k] << " committed=" << next
                    << "\n";
          width_ok = false;
          break;
        }
      }
      if (width_ok) {
        std::cout << "flint verify w=" << w << " OK\n";
      }
      failed = failed || !width_ok;
    }

    if (failed) {
      throw std::runtime_error("flint verify differs from decode");
    }
    std::cout << "FLINT_VERIFY PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "flint_verify_test: " << e.what() << "\n";
    return 1;
  }
}
