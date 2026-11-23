// The trimmed DSpark markov scan (head rows plus special
// token tail) must pick the same winner as the full-vocab scan whenever
// that winner is a kept id, and a tail winner must come back as its real
// token id. Runs the shared partial + finalize kernels directly on a small
// vocabulary; no model files are needed.
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "src/models/qwen/hip/dflash/dspark_sampling.hpp"

namespace {

constexpr std::uint32_t kVocab = 2048;
constexpr std::uint32_t kHeadRows = 1024;
constexpr std::uint32_t kTailStart = 1792;
constexpr std::uint32_t kTailColumns = kVocab - kTailStart;
constexpr std::uint32_t kRank = 16;
constexpr std::uint32_t kRows = 2;
constexpr std::uint32_t kHidden = 8;
constexpr std::uint32_t kTopK = 8;
// The partial grid is one tile wide at this vocabulary; keep slack anyway.
constexpr std::uint32_t kMaxPartialBlocks = 64;
constexpr std::uint32_t kTrials = 8;

std::uint64_t NextRandom(std::uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = state;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31U);
}

float NextFloat(std::uint64_t& state) {
  return (static_cast<float>(NextRandom(state) & 0xFFFFU) / 65535.0F) *
             2.0F -
         1.0F;
}

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::cerr << "dspark_trim_test: " << what << ": "
              << hipGetErrorString(status) << '\n';
    std::exit(1);
  }
}

void Expect(bool condition, const char* what) {
  if (!condition) {
    std::cerr << "dspark_trim_test failed: " << what << '\n';
    std::exit(1);
  }
}

struct DeviceBuffers {
  float* full_logits{nullptr};
  float* head_logits{nullptr};
  float* tail_logits{nullptr};
  hip_bfloat16* w1{nullptr};
  hip_bfloat16* w2{nullptr};
  float* hidden{nullptr};
  std::uint32_t* tokens{nullptr};
  float* confidences{nullptr};
  float* partial_scores{nullptr};
  std::uint32_t* partial_ids{nullptr};
  std::uint32_t* candidate_ids{nullptr};
  float* candidate_probabilities{nullptr};
};

template <typename T>
T* Allocate(std::size_t elements) {
  T* pointer = nullptr;
  Check(hipMalloc(&pointer, elements * sizeof(T)), "hipMalloc");
  return pointer;
}

// Runs the shared partial + finalize kernels once and returns tokens[1..rows].
std::vector<std::uint32_t> RunSelect(const DeviceBuffers& buffers,
                                     bool trimmed) {
  const std::array<std::uint32_t, kRows + 1> previous{7U, 11U, 0U};
  Check(hipMemcpy(buffers.tokens, previous.data(),
                  previous.size() * sizeof(std::uint32_t),
                  hipMemcpyHostToDevice),
        "upload tokens");
  superroc::hip::kernels::LaunchDSparkMarkovSelect(
      trimmed ? buffers.head_logits : buffers.full_logits,
      trimmed ? kHeadRows : 0U, trimmed ? kTailStart : 0U,
      trimmed ? buffers.tail_logits : nullptr, buffers.hidden, buffers.w1,
      buffers.w2, nullptr, nullptr, buffers.tokens, buffers.confidences,
      buffers.partial_scores, buffers.partial_ids, nullptr,
      buffers.candidate_ids, buffers.candidate_probabilities, kRows, kVocab,
      kHidden, kRank, kTopK, 0.0F, hipStreamLegacy);
  Check(hipStreamSynchronize(hipStreamLegacy), "synchronize");
  std::vector<std::uint32_t> tokens(kRows, 0);
  Check(hipMemcpy(tokens.data(), buffers.tokens + 1,
                  kRows * sizeof(std::uint32_t), hipMemcpyDeviceToHost),
        "download tokens");
  return tokens;
}

}  // namespace

int main() {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cout << "dspark_trim_test: no HIP device; skipping\n";
    return 77;
  }

  DeviceBuffers buffers;
  buffers.full_logits = Allocate<float>(kRows * kVocab);
  buffers.head_logits = Allocate<float>(kRows * kHeadRows);
  buffers.tail_logits = Allocate<float>(kRows * kTailColumns);
  buffers.w1 = Allocate<hip_bfloat16>(kVocab * kRank);
  buffers.w2 = Allocate<hip_bfloat16>(kVocab * kRank);
  buffers.hidden = Allocate<float>(kRows * kHidden);
  buffers.tokens = Allocate<std::uint32_t>(kRows + 1);
  buffers.confidences = Allocate<float>(kRows);
  buffers.partial_scores =
      Allocate<float>(kRows * kMaxPartialBlocks * kTopK);
  buffers.partial_ids =
      Allocate<std::uint32_t>(kRows * kMaxPartialBlocks * kTopK);
  buffers.candidate_ids = Allocate<std::uint32_t>(kRows * kTopK);
  buffers.candidate_probabilities = Allocate<float>(kRows * kTopK);

  std::uint64_t rng = 0x243F6A8885A308D3ULL;
  std::vector<hip_bfloat16> w1(kVocab * kRank);
  std::vector<hip_bfloat16> w2(kVocab * kRank);
  for (hip_bfloat16& value : w1) {
    value = hip_bfloat16(NextFloat(rng) * 0.5F);
  }
  for (hip_bfloat16& value : w2) {
    value = hip_bfloat16(NextFloat(rng) * 0.5F);
  }
  Check(hipMemcpy(buffers.w1, w1.data(), w1.size() * sizeof(hip_bfloat16),
                  hipMemcpyHostToDevice),
        "upload w1");
  Check(hipMemcpy(buffers.w2, w2.data(), w2.size() * sizeof(hip_bfloat16),
                  hipMemcpyHostToDevice),
        "upload w2");
  std::vector<float> hidden(kRows * kHidden, 0.0F);
  Check(hipMemcpy(buffers.hidden, hidden.data(),
                  hidden.size() * sizeof(float), hipMemcpyHostToDevice),
        "upload hidden");

  for (std::uint32_t trial = 0; trial < kTrials; ++trial) {
    // Dropped ids [H, T) score far below every kept id, so the full-vocab
    // winner is always a kept id; the markov bonus stays small next to the
    // forced winners below.
    std::vector<float> full(kRows * kVocab);
    for (std::uint32_t row = 0; row < kRows; ++row) {
      for (std::uint32_t id = 0; id < kVocab; ++id) {
        full[(row * kVocab) + id] = id < kHeadRows || id >= kTailStart
                                       ? NextFloat(rng)
                                       : -1.0e30F;
      }
    }
    const std::uint32_t forced_head = 500U;    // trial 0: row 1 wins in head
    const std::uint32_t forced_tail = 1900U;  // trial 0: row 0 wins in tail
    if (trial == 0) {
      full[forced_tail] = 4096.0F;
      full[kVocab + forced_head] = 4096.0F;
    }
    std::vector<float> head(kRows * kHeadRows);
    std::vector<float> tail(kRows * kTailColumns);
    for (std::uint32_t row = 0; row < kRows; ++row) {
      for (std::uint32_t id = 0; id < kHeadRows; ++id) {
        head[(row * kHeadRows) + id] = full[(row * kVocab) + id];
      }
      for (std::uint32_t id = kTailStart; id < kVocab; ++id) {
        tail[(row * kTailColumns) + (id - kTailStart)] =
            full[(row * kVocab) + id];
      }
    }
    Check(hipMemcpy(buffers.full_logits, full.data(),
                    full.size() * sizeof(float), hipMemcpyHostToDevice),
          "upload full logits");
    Check(hipMemcpy(buffers.head_logits, head.data(),
                    head.size() * sizeof(float), hipMemcpyHostToDevice),
          "upload head logits");
    Check(hipMemcpy(buffers.tail_logits, tail.data(),
                    tail.size() * sizeof(float), hipMemcpyHostToDevice),
          "upload tail logits");

    const std::vector<std::uint32_t> full_tokens = RunSelect(buffers, false);
    const std::vector<std::uint32_t> trimmed_tokens = RunSelect(buffers, true);
    for (std::uint32_t row = 0; row < kRows; ++row) {
      Expect(full_tokens[row] < kHeadRows || full_tokens[row] >= kTailStart,
             "full-vocab winner is a kept id");
      Expect(trimmed_tokens[row] == full_tokens[row],
             "trimmed winner equals full winner");
      Expect(trimmed_tokens[row] < kVocab, "trimmed winner is a real id");
    }
    if (trial == 0) {
      Expect(trimmed_tokens[0] == forced_tail,
             "tail winner keeps its real token id");
      Expect(trimmed_tokens[1] == forced_head,
             "head winner keeps its real token id");
      std::vector<std::uint32_t> candidates(kRows * kTopK, 0);
      Check(hipMemcpy(candidates.data(), buffers.candidate_ids,
                      candidates.size() * sizeof(std::uint32_t),
                      hipMemcpyDeviceToHost),
            "download candidates");
      for (const std::uint32_t id : candidates) {
        Expect(id < kHeadRows || id >= kTailStart,
               "trimmed candidate id is a kept real id");
      }
    }
  }

  std::cout << "dspark_trim_test: trimmed markov matches full-vocab scan\n";
  return 0;
}
