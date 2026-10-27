#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen/hip/kernels/dflash_kernels.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "tests/models/qwen/hip/support/comparisons.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"
#include "src/core/env_compat.hpp"

namespace {

// Official DFlash2 config: 32 query heads, 8 KV heads, head dimension 128.
constexpr std::uint32_t kNumQueryHeads = 32;
constexpr std::uint32_t kNumKvHeads = 8;
constexpr std::uint32_t kHeadDim = 128;
constexpr std::uint32_t kBlockCount = 8;
constexpr std::uint32_t kSlidingWindow = 2048;
// Forced tile for the tiled route: not a multiple of the 32-key prefetch, so
// tails occur, and small enough that one row's keys span many tiles.
constexpr std::uint32_t kTestTileKeys = 72;
// Forced key range for the split route: not a multiple of its 8-value
// prefetch, and short enough that one row spans many ranges and tiles.
constexpr std::uint32_t kTestSplitKeys = 41;

float Sample(std::size_t index, std::size_t salt) {
  return std::sin(static_cast<float>((index * 7U) + (salt * 13U) + 1U)) * 0.5F;
}

enum class Route { kScalar, kWave, kTiled, kSplit };

// Independent equations from z-lab/dflash model_mlx.py, pinned in eval/README.
// Keep these small operator checks with the ring/attention checks.
void TestDynamicConvolution() {
  using superroc::test::DeviceBuffer;
  constexpr std::uint32_t tokens = 8, hidden = 5120, taps = 2, group_size = 16;
  constexpr std::uint32_t groups = hidden / group_size;
  std::vector<float> input(tokens * hidden),
      dynamic(tokens * 2 * taps * groups), base(2 * taps * hidden);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = Sample(i, 11);
  for (std::size_t i = 0; i < dynamic.size(); ++i)
    dynamic[i] = Sample(i, 12);
  for (std::size_t i = 0; i < base.size(); ++i)
    base[i] = Sample(i, 13);
  DeviceBuffer<float> d_input(input), d_dynamic(dynamic), d_base(base),
      d_output(input.size());
  for (std::uint32_t side = 0; side < 2; ++side) {
    superroc::hip::kernels::LaunchDFlashGroupedDynamicConv(
        d_input.data(), d_dynamic.data(), d_base.data(), d_output.data(),
        tokens, hidden, taps, group_size, side, nullptr);
    const auto output = d_output.CopyToHost();
    for (std::uint32_t token = 0; token < tokens; ++token) {
      for (std::uint32_t channel = 0; channel < hidden; ++channel) {
        double reference = 0;
        for (std::uint32_t tap = 0; tap < taps && tap <= token; ++tap) {
          const double value = input[(token - tap) * hidden + channel];
          reference += base[(side * taps + tap) * hidden + channel] * value;
          reference += dynamic[((token * 2 + side) * taps + tap) * groups +
                               channel / group_size] *
                       value;
        }
        superroc::test::Expect(
            std::isfinite(output[token * hidden + channel]) &&
                std::abs(output[token * hidden + channel] - reference) < 2e-7,
            "DFlash grouped convolution differs from the causal equation");
      }
    }
  }
}

void TestProposalVerification(std::span<const std::uint32_t> ids,
                              std::span<const float> q, std::uint32_t vocab) {
  using superroc::test::DeviceBuffer;
  using superroc::test::Expect;
  Expect(std::ranges::find(ids, 0U) == ids.end() &&
             std::ranges::find(ids, 1U) == ids.end(),
         "residual fixture must include tokens outside draft top-k");
  std::vector<float> logits(vocab, -INFINITY);
  logits[0] = std::log(0.55F);
  logits[1] = std::log(0.10F);
  logits[ids[0]] = std::log(0.13F);
  logits[ids[1]] = std::log(0.07F);
  logits[ids[2]] = std::log(0.15F);
  DeviceBuffer<float> d_logits(logits);
  DeviceBuffer<std::uint32_t> d_token(1), d_accepted(1);
  superroc::hip::GpuSamplingWorkspace workspace;
  superroc::hip::AllocateGpuSamplingWorkspace(&workspace, vocab, ids.size());
  for (const bool filtered : {false, true}) {
    superroc::sampling::SamplingConfig config;
    config.temperature = filtered ? 0.8F : 1.0F;
    config.top_k = filtered ? 4 : 0;
    config.top_p = filtered ? 0.95F : 1.0F;
    config.min_p = filtered ? 0.05F : 0.0F;
    config.repeat_penalty = filtered ? 1.2F : 1.0F;
    const std::vector<std::uint32_t> history = {ids[0], ids[0]};
    const auto p = superroc::sampling::BuildDistribution(logits, config, history);
    std::vector<superroc::sampling::Probability> residual;
    double rejection_mass = 0;
    for (const auto& entry : p.entries()) {
      const auto found = std::ranges::find(ids, entry.token);
      const double draft_p = found == ids.end() ? 0 : q[found - ids.begin()];
      const double mass = std::max(entry.value - draft_p, 0.0);
      if (mass > 0) {
        residual.push_back({entry.token, mass});
        rejection_mass += mass;
      }
    }
    // Linear GPU sampling walks token IDs; the filtered route walks sorted
    // target logits. Select a midpoint in every nonempty residual interval.
    if (!filtered)
      std::ranges::sort(residual, {}, &superroc::sampling::Probability::token);
    superroc::hip::GpuSamplingParameters parameters{
        .temperature = config.temperature,
        .top_k = config.top_k,
        .top_p = config.top_p,
        .min_p = config.min_p,
        .repeat_penalty = config.repeat_penalty,
    };
    const std::array<superroc::sampling::TokenPenalty, 1> penalties{
        {{ids[0], 0, 1}}};
    const auto verify = [&](std::size_t candidate, double accept_u,
                            double residual_u, bool accepted,
                            std::uint32_t expected) {
      superroc::hip::LaunchGPUSpeculativeSampling(
          d_logits.data(), d_token.data(), d_accepted.data(), vocab, parameters,
          ids[candidate], q[candidate], ids.data(), q.data(), ids.size(),
          accept_u, residual_u, penalties.data(), filtered ? 1 : 0, &workspace);
      Expect(
          d_accepted.CopyToHost()[0] == static_cast<std::uint32_t>(accepted) &&
              d_token.CopyToHost()[0] == expected,
          "GPU verifier differs from independent p/q and residual equations");
    };
    std::vector<double> emitted(vocab, 0);
    for (std::size_t candidate = 0; candidate < ids.size(); ++candidate) {
      if (q[candidate] == 0)
        continue;
      const double acceptance =
          std::min(1.0, p.probability(ids[candidate]) / q[candidate]);
      if (acceptance > 0)
        verify(candidate, acceptance * 0.5, 0.5, true, ids[candidate]);
      emitted[ids[candidate]] += q[candidate] * acceptance;
      if (acceptance < 1) {
        double cumulative = 0;
        for (const auto& entry : residual) {
          const double probability = entry.value / rejection_mass;
          verify(candidate, (1 + acceptance) * 0.5,
                 cumulative + probability * 0.5, false, entry.token);
          emitted[entry.token] += q[candidate] * (1 - acceptance) * probability;
          cumulative += probability;
        }
      }
    }
    for (std::uint32_t token = 0; token < vocab; ++token)
      Expect(std::abs(emitted[token] - p.probability(token)) < 2e-6,
             "selector plus verifier must recover the complete target "
             "distribution");
  }
  superroc::hip::FreeGpuSamplingWorkspace(&workspace);
}

void TestSelector(std::uint32_t vocab) {
  using superroc::test::DeviceBuffer;
  // Cross a partial-top-k partition boundary and leave a short final partition.
  constexpr std::uint32_t rank = 256, anchor = 3;
  std::vector<float> logits(vocab), projected(rank);
  std::vector<std::uint16_t> predecessors(vocab * rank),
      successors(vocab * rank);
  const auto bf16 = [](float value) {
    return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >>
                                      16);
  };
  const auto fp32 = [](std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
  };
  for (std::size_t i = 0; i < logits.size(); ++i) {
    // Distinct unary scores, deliberately scattered across both partitions.
    logits[i] = static_cast<float>((i * 193) % vocab) / vocab * 4.0F;
  }
  for (std::size_t i = 0; i < rank; ++i)
    projected[i] = Sample(i, 17);
  for (std::size_t i = 0; i < predecessors.size(); ++i) {
    predecessors[i] = bf16(Sample(i, 18));
    successors[i] = bf16(Sample(i, 19));
  }
  DeviceBuffer<float> d_logits(logits), d_projected(projected), d_confidence(1),
      d_uniform(std::vector<float>{0.37F});
  DeviceBuffer<std::uint16_t> d_predecessors(predecessors),
      d_successors(successors);
  DeviceBuffer<std::uint32_t> d_anchor(std::vector<std::uint32_t>{anchor}),
      d_selected(1);
  const auto scratch_size =
      superroc::hip::kernels::DFlashSelectorScratchElements(vocab);
  DeviceBuffer<float> d_scores(scratch_size);
  DeviceBuffer<std::uint32_t> d_ids(scratch_size);
  std::vector<std::uint32_t> sorted(vocab);
  std::iota(sorted.begin(), sorted.end(), 0U);
  std::ranges::sort(sorted,
                    [&](auto a, auto b) { return logits[a] > logits[b]; });
  for (const std::uint32_t top_k : {1U, 7U, 16U}) {
    DeviceBuffer<std::uint32_t> d_candidates(top_k);
    DeviceBuffer<float> d_probabilities(top_k);
    std::vector<double> scores(top_k);
    for (std::size_t c = 0; c < top_k; ++c) {
      scores[c] = logits[sorted[c]];
      for (std::size_t r = 0; r < rank; ++r) {
        scores[c] +=
            static_cast<double>(fp32(predecessors[anchor * rank + r])) *
            projected[r] * fp32(successors[sorted[c] * rank + r]);
      }
    }
    for (const float temperature : {0.0F, 0.8F, 1e-38F}) {
      // Unwritten partial slots must never enter the candidate set.
      d_scores.CopyFrom(std::vector<float>(scratch_size, 1e20F));
      d_ids.CopyFrom(std::vector<std::uint32_t>(scratch_size, vocab - 1));
      superroc::hip::kernels::LaunchDFlashSelectorStep(
          d_logits.data(), d_projected.data(), d_predecessors.data(),
          d_successors.data(), d_anchor.data(), d_selected.data(),
          d_confidence.data(), d_scores.data(), d_ids.data(), temperature,
          d_uniform.data(), d_candidates.data(), d_probabilities.data(), vocab,
          rank, top_k, nullptr);
      const auto candidates = d_candidates.CopyToHost();
      superroc::test::Expect(
          std::equal(candidates.begin(), candidates.end(), sorted.begin()),
          "DFlash selector top-k differs from full vocabulary sort");
      const auto probabilities = d_probabilities.CopyToHost();
      const double maximum = *std::ranges::max_element(scores);
      const double divisor = temperature > 0 ? temperature : 1.0;
      double denominator = 0;
      for (const double value : scores)
        denominator += std::exp((value - maximum) / divisor);
      std::size_t selected = std::ranges::max_element(scores) - scores.begin();
      if (temperature > 0) {
        double cumulative = 0;
        selected = top_k - 1;
        for (std::size_t c = 0; c < top_k; ++c) {
          const double p =
              std::exp((scores[c] - maximum) / divisor) / denominator;
          if (!std::isfinite(probabilities[c]) ||
              std::abs(probabilities[c] - p) >= 2e-6) {
            std::cerr << "selector vocab=" << vocab << " k=" << top_k
                      << " temperature=" << temperature << " candidate=" << c
                      << " actual=" << probabilities[c] << " expected=" << p
                      << '\n';
          }
          superroc::test::Expect(std::isfinite(probabilities[c]) &&
                                 std::abs(probabilities[c] - p) < 2e-6,
                             "DFlash sparse proposal probability is incorrect");
          const double previous = cumulative;
          cumulative += p;
          if (previous < 0.37 && cumulative >= 0.37)
            selected = c;
        }
      }
      superroc::test::Expect(d_selected.CopyToHost()[0] == sorted[selected],
                         "DFlash selector chose the wrong candidate");
      const double confidence =
          temperature > 0
              ? std::exp((scores[selected] - maximum) / divisor) / denominator
              : 1.0;
      if (temperature == 0) {
        for (std::size_t c = 0; c < top_k; ++c)
          superroc::test::Expect(probabilities[c] == (c == selected ? 1.0F : 0.0F),
                             "greedy proposal probabilities must be one-hot");
      }
      superroc::test::Expect(
          std::abs(d_confidence.CopyToHost()[0] - confidence) < 2e-6,
          "DFlash selected-token probability is incorrect");
      if (vocab == 2065 && top_k == 16 && temperature == 0.8F)
        TestProposalVerification(candidates, probabilities, vocab);
    }
  }
}

void TestSelectorZeroUniform() {
  using superroc::test::DeviceBuffer;
  const auto bf16 = [](float value) {
    return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >>
                                      16);
  };
  // Reranking makes the first unary candidate have zero sampling mass.
  DeviceBuffer<float> logits(std::vector<float>{1, 0, -2000}),
      projected(std::vector<float>{1}), uniform(std::vector<float>{0}),
      confidence(1), probabilities(2);
  DeviceBuffer<std::uint16_t> predecessors(
      std::vector<std::uint16_t>{bf16(0), bf16(0), bf16(1)}),
      successors(std::vector<std::uint16_t>{bf16(-1000), bf16(0), bf16(0)});
  DeviceBuffer<std::uint32_t> anchor(std::vector<std::uint32_t>{2}),
      selected(1), candidates(2);
  const auto size = superroc::hip::kernels::DFlashSelectorScratchElements(3);
  DeviceBuffer<float> scores(size);
  DeviceBuffer<std::uint32_t> ids(size);
  superroc::hip::kernels::LaunchDFlashSelectorStep(
      logits.data(), projected.data(), predecessors.data(), successors.data(),
      anchor.data(), selected.data(), confidence.data(), scores.data(),
      ids.data(), 1.0F, uniform.data(), candidates.data(), probabilities.data(),
      3, 1, 2, nullptr);
  superroc::test::Expect(
      selected.CopyToHost()[0] == 1 && confidence.CopyToHost()[0] == 1.0F,
      "a zero random draw must not select zero probability");
}

void TestBatchedSelector() {
  using superroc::hip::kernels::DFlashSelectorSequence;
  using superroc::test::DeviceBuffer;
  using superroc::test::Expect;
  constexpr std::uint32_t vocab = 2065, rank = 32;
  std::vector<std::uint16_t> predecessor(vocab * rank), successor(vocab * rank);
  for (std::size_t index = 0; index < predecessor.size(); ++index) {
    predecessor[index] = static_cast<std::uint16_t>(
        std::bit_cast<std::uint32_t>(Sample(index, 31)) >> 16);
    successor[index] = static_cast<std::uint16_t>(
        std::bit_cast<std::uint32_t>(Sample(index, 47)) >> 16);
  }
  DeviceBuffer<std::uint16_t> d_predecessor(predecessor),
      d_successor(successor);
  const auto exact = [](const auto& left, const auto& right) {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin(),
                      [](float a, float b) {
                        return std::bit_cast<std::uint32_t>(a) ==
                               std::bit_cast<std::uint32_t>(b);
                      });
  };
  for (const auto users : {1U, 2U, 4U, 6U, 8U}) {
    std::vector<DFlashSelectorSequence> sequences(users);
    std::uint32_t rows = 0;
    for (std::uint32_t index = 0; index < users; ++index) {
      sequences[index] = {users == 1 ? 7U : 1U + (index * 3U) % 7U,
                          std::array{0.0F, 0.8F, 1e-38F, 1.2F}[index % 4]};
      rows += sequences[index].count;
    }
    std::vector<std::uint32_t> tokens(rows + users, 0);
    for (std::uint32_t index = 0, offset = 0; index < users; ++index) {
      tokens[offset] = 7U + index;
      offset += sequences[index].count + 1U;
    }
    std::vector<float> logits(rows * vocab), hidden(rows * rank),
        uniforms(rows);
    for (std::size_t index = 0; index < logits.size(); ++index)
      logits[index] = Sample(index, 73);
    for (std::size_t index = 0; index < hidden.size(); ++index)
      hidden[index] = Sample(index, 89);
    for (std::uint32_t row = 0; row < rows; ++row)
      uniforms[row] = static_cast<float>((row * 37U) % 101U) / 101.0F;
    DeviceBuffer<float> d_logits(logits), d_hidden(hidden),
        d_uniforms(uniforms), d_confidences(rows);
    DeviceBuffer<std::uint32_t> d_tokens(tokens);
    const auto partials =
        superroc::hip::kernels::DFlashSelectorScratchElements(vocab);
    DeviceBuffer<float> d_scores(rows * partials);
    DeviceBuffer<std::uint32_t> d_ids(rows * partials);
    for (const auto top_k : {1U, 7U, 16U}) {
      DeviceBuffer<float> d_probabilities(
          std::vector<float>(rows * top_k, -1.0F));
      DeviceBuffer<std::uint32_t> d_candidates(
          std::vector<std::uint32_t>(rows * top_k, vocab));
      d_tokens.CopyFrom(tokens);
      for (std::uint32_t index = 0, first_row = 0; index < users; ++index) {
        const auto sequence = sequences[index];
        for (std::uint32_t step = 0; step < sequence.count; ++step) {
          const auto row = first_row + step;
          superroc::hip::kernels::LaunchDFlashSelectorStep(
              d_logits.data() + row * vocab, d_hidden.data() + row * rank,
              d_predecessor.data(), d_successor.data(),
              d_tokens.data() + row + index, d_tokens.data() + row + index + 1U,
              d_confidences.data() + row, d_scores.data(), d_ids.data(),
              sequence.temperature, d_uniforms.data() + row,
              d_candidates.data() + row * top_k,
              d_probabilities.data() + row * top_k, vocab, rank, top_k,
              nullptr);
        }
        first_row += sequence.count;
      }
      const auto expected_tokens = d_tokens.CopyToHost();
      const auto expected_candidates = d_candidates.CopyToHost();
      const auto expected_probabilities = d_probabilities.CopyToHost();
      const auto expected_confidences = d_confidences.CopyToHost();
      d_tokens.CopyFrom(tokens);
      d_candidates.CopyFrom(std::vector<std::uint32_t>(rows * top_k, vocab));
      d_probabilities.CopyFrom(std::vector<float>(rows * top_k, -1.0F));
      d_confidences.CopyFrom(std::vector<float>(rows, -1.0F));
      d_scores.CopyFrom(std::vector<float>(rows * partials, 1e20F));
      d_ids.CopyFrom(std::vector<std::uint32_t>(rows * partials, vocab - 1));
      superroc::hip::kernels::LaunchDFlashSelectorBatch(
          d_logits.data(), d_hidden.data(), d_predecessor.data(),
          d_successor.data(), d_tokens.data(), d_confidences.data(),
          d_scores.data(), d_ids.data(), d_uniforms.data(), d_candidates.data(),
          d_probabilities.data(), sequences, vocab, rank, top_k, nullptr);
      const auto actual_tokens = d_tokens.CopyToHost();
      const auto actual_candidates = d_candidates.CopyToHost();
      const auto actual_probabilities = d_probabilities.CopyToHost();
      const auto actual_confidences = d_confidences.CopyToHost();
      const bool matches =
          actual_tokens == expected_tokens &&
          actual_candidates == expected_candidates &&
          exact(actual_probabilities, expected_probabilities) &&
          exact(actual_confidences, expected_confidences);
      if (!matches) {
        std::cerr << "selector C=" << users << " top_k=" << top_k
                  << " tokens=" << (actual_tokens == expected_tokens)
                  << " candidates="
                  << (actual_candidates == expected_candidates)
                  << " probabilities="
                  << exact(actual_probabilities, expected_probabilities)
                  << " confidences="
                  << exact(actual_confidences, expected_confidences) << '\n';
      }
      Expect(matches,
             "batched selectors must preserve ragged private token chains, "
             "sampled probabilities and uniforms");
    }
  }
}

/// Runs the reference and one candidate route over the same operands and
/// reports the largest absolute difference. The kernels accumulate the QK dot
/// product in a different order -- the reference walks head_dim in one thread,
/// the candidates reduce across a wave -- so the comparison is a float
/// tolerance, not bit equality.
float MaxRouteDifference(Route route, std::uint32_t current_pos,
                         std::uint32_t history_length,
                         std::uint32_t block_count,
                         std::uint32_t tile_keys = kTestTileKeys,
                         std::uint32_t split_keys = kTestSplitKeys,
                         std::uint32_t window = kSlidingWindow) {
  const std::size_t q_dim = static_cast<std::size_t>(kNumQueryHeads) * kHeadDim;
  const std::size_t kv_dim = static_cast<std::size_t>(kNumKvHeads) * kHeadDim;
  const std::size_t q_elements = block_count * q_dim;
  const std::size_t block_elements = block_count * kv_dim;
  const std::size_t history_elements =
      static_cast<std::size_t>(current_pos + block_count) * kv_dim;

  std::vector<float> h_q(q_elements);
  std::vector<float> h_block_k(block_elements);
  std::vector<float> h_block_v(block_elements);
  std::vector<float> h_injected_k(history_elements);
  std::vector<float> h_injected_v(history_elements);
  for (std::size_t index = 0; index < q_elements; ++index) {
    h_q[index] = Sample(index, 1);
  }
  for (std::size_t index = 0; index < block_elements; ++index) {
    h_block_k[index] = Sample(index, 2);
    h_block_v[index] = Sample(index, 3);
  }
  for (std::size_t index = 0; index < history_elements; ++index) {
    h_injected_k[index] = Sample(index, 4);
    h_injected_v[index] = Sample(index, 5);
  }

  const auto upload = [](const std::vector<float>& host) {
    float* device = nullptr;
    HIP_CHECK(hipMalloc(&device, host.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(device, host.data(), host.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    return device;
  };

  // Store only live history in a wrapped ring. Poison all other slots so an
  // off-by-one mask or addressing error cannot accidentally compare equal.
  std::vector<float> ring_k(window * kv_dim, NAN);
  std::vector<float> ring_v(window * kv_dim, NAN);
  const auto first = history_length > window ? history_length - window : 0U;
  for (auto pos = first; pos < history_length; ++pos) {
    std::copy_n(h_injected_k.data() + pos * kv_dim, kv_dim,
                ring_k.data() + (pos % window) * kv_dim);
    std::copy_n(h_injected_v.data() + pos * kv_dim, kv_dim,
                ring_v.data() + (pos % window) * kv_dim);
  }
  float* d_ring_k = upload(ring_k);
  float* d_ring_v = upload(ring_v);
  float* d_q = upload(h_q);
  float* d_block_k = upload(h_block_k);
  float* d_block_v = upload(h_block_v);
  float* d_injected_k = upload(h_injected_k);
  float* d_injected_v = upload(h_injected_v);
  float* d_scalar_out = nullptr;
  float* d_wave_out = nullptr;
  HIP_CHECK(hipMalloc(&d_scalar_out, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_wave_out, q_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_scalar_out, 0, q_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_wave_out, 0, q_elements * sizeof(float)));

  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  superroc::hip::kernels::LaunchDFlashNonCausalAttentionScalar(
      d_q, d_injected_k, d_injected_v, d_block_k, d_block_v, d_scalar_out,
      current_pos, history_length, block_count, window, kNumQueryHeads,
      kNumKvHeads, kHeadDim, scale, nullptr, current_pos + block_count);
  if (route == Route::kWave) {
    superroc::hip::kernels::LaunchDFlashNonCausalAttentionWave(
        d_q, d_ring_k, d_ring_v, d_block_k, d_block_v, d_wave_out, current_pos,
        history_length, block_count, window, kNumQueryHeads, kNumKvHeads,
        kHeadDim, scale, nullptr, window);
  }
  if (route == Route::kScalar) {
    superroc::hip::kernels::LaunchDFlashNonCausalAttentionScalar(
        d_q, d_ring_k, d_ring_v, d_block_k, d_block_v, d_wave_out, current_pos,
        history_length, block_count, window, kNumQueryHeads, kNumKvHeads,
        kHeadDim, scale, nullptr, window);
  }
  if (route == Route::kTiled) {
    superroc::hip::kernels::LaunchDFlashNonCausalAttentionTiled(
        d_q, d_ring_k, d_ring_v, d_block_k, d_block_v, d_wave_out, current_pos,
        history_length, block_count, window, kNumQueryHeads, kNumKvHeads,
        kHeadDim, scale, nullptr, window, tile_keys);
  }
  if (route == Route::kSplit) {
    const std::size_t scratch_elements =
        superroc::hip::kernels::DFlashAttentionSplitScratchElements(
            block_count, kNumQueryHeads, kHeadDim);
    superroc::test::DeviceBuffer<float> d_scratch(scratch_elements);
    superroc::hip::kernels::LaunchDFlashNonCausalAttentionSplit(
        d_q, d_ring_k, d_ring_v, d_block_k, d_block_v, d_wave_out, current_pos,
        history_length, block_count, window, kNumQueryHeads, kNumKvHeads,
        kHeadDim, scale, nullptr, window, d_scratch.data(), scratch_elements,
        split_keys);
    HIP_CHECK(hipDeviceSynchronize());
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> scalar_out(q_elements);
  std::vector<float> wave_out(q_elements);
  HIP_CHECK(hipMemcpy(scalar_out.data(), d_scalar_out,
                      q_elements * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(wave_out.data(), d_wave_out, q_elements * sizeof(float),
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(d_ring_k));
  HIP_CHECK(hipFree(d_ring_v));
  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_block_k));
  HIP_CHECK(hipFree(d_block_v));
  HIP_CHECK(hipFree(d_injected_k));
  HIP_CHECK(hipFree(d_injected_v));
  HIP_CHECK(hipFree(d_scalar_out));
  HIP_CHECK(hipFree(d_wave_out));

  float max_difference = 0.0F;
  for (std::size_t index = 0; index < q_elements; ++index) {
    superroc::test::Expect(std::isfinite(wave_out[index]),
                       "DFlash candidate attention produced a non-finite "
                       "output");
    max_difference =
        std::max(max_difference, std::abs(scalar_out[index] - wave_out[index]));
  }
  // Independent double-precision softmax for the first/last query and GQA
  // group. All proposal keys are visible; only historical keys are windowed.
  for (const auto query : {0U, block_count - 1}) {
    for (const auto head : {0U, kNumQueryHeads - 1}) {
      const auto kv_head = head / (kNumQueryHeads / kNumKvHeads);
      const auto q_offset = query * q_dim + head * kHeadDim;
      std::vector<double> scores(history_length + block_count, -INFINITY);
      double maximum = -INFINITY;
      for (std::size_t key = 0; key < scores.size(); ++key) {
        if (key < history_length && current_pos + query - key >= window)
          continue;
        const auto* keys =
            key < history_length ? h_injected_k.data() : h_block_k.data();
        const auto row = key < history_length ? key : key - history_length;
        double dot = 0;
        for (std::size_t d = 0; d < kHeadDim; ++d) {
          dot += static_cast<double>(h_q[q_offset + d]) *
                 keys[row * kv_dim + kv_head * kHeadDim + d];
        }
        scores[key] = dot * scale;
        maximum = std::max(maximum, scores[key]);
      }
      double denominator = 0;
      for (auto& score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      for (std::size_t d = 0; d < kHeadDim; ++d) {
        double reference = 0;
        for (std::size_t key = 0; key < scores.size(); ++key) {
          const auto* values =
              key < history_length ? h_injected_v.data() : h_block_v.data();
          const auto row = key < history_length ? key : key - history_length;
          reference +=
              scores[key] * values[row * kv_dim + kv_head * kHeadDim + d];
        }
        superroc::test::Expect(
            std::abs(wave_out[q_offset + d] - reference / denominator) < 1e-5,
            "DFlash attention differs from the windowed softmax equation");
      }
    }
  }
  return max_difference;
}

void TestRouteEquivalence(std::uint32_t current_pos,
                          std::uint32_t history_length, const char* label,
                          std::uint32_t block_count = kBlockCount) {
  superroc::test::Expect(MaxRouteDifference(Route::kScalar, current_pos,
                                        history_length, block_count) == 0.0F,
                     "ring addressing must preserve scalar attention exactly");
  const float wave = MaxRouteDifference(Route::kWave, current_pos,
                                        history_length, block_count);
  std::cout << "DFlash non-causal attention " << label
            << ": max |scalar - wave| = " << wave << "\n";
  superroc::test::Expect(wave < 1e-5F,
                     "DFlash wave attention diverges from the reference");
  const float tiled = MaxRouteDifference(Route::kTiled, current_pos,
                                         history_length, block_count);
  std::cout << "DFlash non-causal attention " << label
            << ": max |scalar - tiled| = " << tiled << "\n";
  superroc::test::Expect(tiled < 1e-5F,
                     "DFlash tiled attention diverges from the reference");
  // Forced short ranges, then the production range length.
  for (const auto split_keys : {kTestSplitKeys, 0U}) {
    const float split =
        MaxRouteDifference(Route::kSplit, current_pos, history_length,
                           block_count, kTestTileKeys, split_keys);
    std::cout << "DFlash non-causal attention " << label
              << ": max |scalar - split(" << split_keys << ")| = " << split
              << "\n";
    superroc::test::Expect(split < 1e-5F,
                       "DFlash split attention diverges from the reference");
  }
}

// DSpark drafter of MiniCPM-V 4.7 35B-A3B: 8 query heads, 2 KV heads, head
// dimension 256, and no sliding window in the GGUF, so the loader falls back to
// the 262144-token context length and the drafter attends the whole prompt.
constexpr std::uint32_t kLongQueryHeads = 8;
constexpr std::uint32_t kLongKvHeads = 2;
constexpr std::uint32_t kLongHeadDim = 256;
constexpr std::uint32_t kLongWindow = 262144;
constexpr std::uint32_t kLongCapacity = 32768;

/// Production router at a history whose whole score row may not fit in the
/// workgroup's 64 KiB of LDS. Until the tiled route, 26,821 keys failed to
/// launch (invalid argument) and the build before the launch checks silently
/// skipped the kernel. Checks every output finite and the first/last query
/// and head against a double-precision softmax over every key.
enum class LongRoute { kRouter, kRouterSplit, kSplitForced };

void TestLongHistory(std::uint32_t history_length, std::uint32_t block_count,
                     const char* label, LongRoute route = LongRoute::kRouter,
                     std::uint32_t split_keys = 0) {
  const std::size_t q_dim = std::size_t{kLongQueryHeads} * kLongHeadDim;
  const std::size_t kv_dim = std::size_t{kLongKvHeads} * kLongHeadDim;
  std::vector<float> q(block_count * q_dim), block_k(block_count * kv_dim),
      block_v(block_count * kv_dim), history_k(kLongCapacity * kv_dim, NAN),
      history_v(kLongCapacity * kv_dim, NAN);
  for (std::size_t i = 0; i < q.size(); ++i)
    q[i] = Sample(i, 21);
  for (std::size_t i = 0; i < block_k.size(); ++i) {
    block_k[i] = Sample(i, 22);
    block_v[i] = Sample(i, 23);
  }
  // Live history only; NaN slots catch a read past history_length.
  for (std::size_t i = 0; i < history_length * kv_dim; ++i) {
    history_k[i] = Sample(i, 24);
    history_v[i] = Sample(i, 25);
  }
  superroc::test::DeviceBuffer<float> d_q(q), d_block_k(block_k),
      d_block_v(block_v), d_history_k(history_k), d_history_v(history_v),
      d_out(q.size());
  const float scale = 1.0F / std::sqrt(static_cast<float>(kLongHeadDim));
  const std::size_t scratch_elements =
      superroc::hip::kernels::DFlashAttentionSplitScratchElements(
          block_count, kLongQueryHeads, kLongHeadDim);
  superroc::test::DeviceBuffer<float> d_scratch(scratch_elements);
  try {
    if (route == LongRoute::kSplitForced) {
      superroc::hip::kernels::LaunchDFlashNonCausalAttentionSplit(
          d_q.data(), d_history_k.data(), d_history_v.data(), d_block_k.data(),
          d_block_v.data(), d_out.data(), history_length, history_length,
          block_count, kLongWindow, kLongQueryHeads, kLongKvHeads, kLongHeadDim,
          scale, nullptr, kLongCapacity, d_scratch.data(), scratch_elements,
          split_keys);
    } else {
      superroc::hip::kernels::LaunchDFlashNonCausalAttention(
          d_q.data(), d_history_k.data(), d_history_v.data(), d_block_k.data(),
          d_block_v.data(), d_out.data(), history_length, history_length,
          block_count, kLongWindow, kLongQueryHeads, kLongKvHeads, kLongHeadDim,
          scale, nullptr, kLongCapacity,
          route == LongRoute::kRouterSplit ? d_scratch.data() : nullptr,
          route == LongRoute::kRouterSplit ? scratch_elements : 0U);
    }
  } catch (const std::exception& error) {
    std::cerr << "DFlash long-history attention " << label << ": "
              << error.what() << "\n";
    superroc::test::Expect(false, "DFlash attention must launch at any history");
  }
  HIP_CHECK(hipDeviceSynchronize());
  const auto out = d_out.CopyToHost();
  for (const float value : out) {
    superroc::test::Expect(std::isfinite(value),
                       "DFlash long-history attention produced a non-finite "
                       "output");
  }
  double max_error = 0;
  for (const auto query : {0U, block_count - 1}) {
    for (const auto head : {0U, kLongQueryHeads - 1}) {
      const auto kv_head = head / (kLongQueryHeads / kLongKvHeads);
      const auto q_offset = query * q_dim + head * kLongHeadDim;
      std::vector<double> scores(history_length + block_count);
      double maximum = -INFINITY;
      for (std::size_t key = 0; key < scores.size(); ++key) {
        const auto* keys =
            key < history_length ? history_k.data() : block_k.data();
        const auto row = key < history_length ? key : key - history_length;
        double dot = 0;
        for (std::size_t d = 0; d < kLongHeadDim; ++d) {
          dot += static_cast<double>(q[q_offset + d]) *
                 keys[row * kv_dim + kv_head * kLongHeadDim + d];
        }
        scores[key] = dot * scale;
        maximum = std::max(maximum, scores[key]);
      }
      double denominator = 0;
      for (auto& score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      for (std::size_t d = 0; d < kLongHeadDim; ++d) {
        double reference = 0;
        for (std::size_t key = 0; key < scores.size(); ++key) {
          const auto* values =
              key < history_length ? history_v.data() : block_v.data();
          const auto row = key < history_length ? key : key - history_length;
          reference +=
              scores[key] * values[row * kv_dim + kv_head * kLongHeadDim + d];
        }
        max_error = std::max(
            max_error, std::abs(out[q_offset + d] - reference / denominator));
      }
    }
  }
  std::cout << "DFlash long-history attention " << label
            << ": max |gpu - double| = " << max_error << "\n";
  superroc::test::Expect(max_error < 1e-5,
                     "DFlash long-history attention differs from the softmax "
                     "equation");
}

/// Below kDFlashSplitMinKeys the production entry point must not change: with
/// split scratch passed it has to produce the same bits as the call without
/// scratch, which takes the wave or scalar routes exactly as before.
void TestRouterShortHistoryUnchanged(std::uint32_t history_length,
                                     std::uint32_t block_count) {
  const std::size_t q_dim = std::size_t{kLongQueryHeads} * kLongHeadDim;
  const std::size_t kv_dim = std::size_t{kLongKvHeads} * kLongHeadDim;
  std::vector<float> q(block_count * q_dim), block_k(block_count * kv_dim),
      block_v(block_count * kv_dim),
      history_k(std::max<std::size_t>(history_length, 1) * kv_dim),
      history_v(history_k.size());
  for (std::size_t i = 0; i < q.size(); ++i)
    q[i] = Sample(i, 31);
  for (std::size_t i = 0; i < block_k.size(); ++i) {
    block_k[i] = Sample(i, 32);
    block_v[i] = Sample(i, 33);
  }
  for (std::size_t i = 0; i < history_k.size(); ++i) {
    history_k[i] = Sample(i, 34);
    history_v[i] = Sample(i, 35);
  }
  superroc::test::DeviceBuffer<float> d_q(q), d_block_k(block_k),
      d_block_v(block_v), d_history_k(history_k), d_history_v(history_v),
      d_plain(q.size()), d_with_scratch(q.size());
  const std::size_t scratch_elements =
      superroc::hip::kernels::DFlashAttentionSplitScratchElements(
          block_count, kLongQueryHeads, kLongHeadDim);
  superroc::test::DeviceBuffer<float> d_scratch(scratch_elements);
  const float scale = 1.0F / std::sqrt(static_cast<float>(kLongHeadDim));
  const auto capacity = std::max<std::uint32_t>(history_length, 1);
  for (const bool scratch : {false, true}) {
    superroc::hip::kernels::LaunchDFlashNonCausalAttention(
        d_q.data(), d_history_k.data(), d_history_v.data(), d_block_k.data(),
        d_block_v.data(), scratch ? d_with_scratch.data() : d_plain.data(),
        history_length, history_length, block_count, kLongWindow,
        kLongQueryHeads, kLongKvHeads, kLongHeadDim, scale, nullptr, capacity,
        scratch ? d_scratch.data() : nullptr, scratch ? scratch_elements : 0U);
  }
  HIP_CHECK(hipDeviceSynchronize());
  const auto plain = d_plain.CopyToHost();
  const auto with_scratch = d_with_scratch.CopyToHost();
  superroc::test::Expect(
      std::memcmp(plain.data(), with_scratch.data(),
                  plain.size() * sizeof(float)) == 0,
      "DFlash attention below the split threshold must be bit-identical");
}

/// Opt-in (SUPERROC_DFLASH_ATTN_BENCH=1) timing of the DSpark drafter attention:
/// the routes without scratch against the split route, per history length.
void BenchLongHistory() {
  constexpr std::uint32_t block_count = 4;
  const std::size_t q_dim = std::size_t{kLongQueryHeads} * kLongHeadDim;
  const std::size_t kv_dim = std::size_t{kLongKvHeads} * kLongHeadDim;
  std::vector<float> q(block_count * q_dim), block_k(block_count * kv_dim),
      history(kLongCapacity * kv_dim);
  for (std::size_t i = 0; i < q.size(); ++i)
    q[i] = Sample(i, 41);
  for (std::size_t i = 0; i < block_k.size(); ++i)
    block_k[i] = Sample(i, 42);
  for (std::size_t i = 0; i < history.size(); ++i)
    history[i] = Sample(i, 43);
  superroc::test::DeviceBuffer<float> d_q(q), d_block_k(block_k),
      d_block_v(block_k), d_history_k(history), d_history_v(history),
      d_out(q.size());
  const std::size_t scratch_elements =
      superroc::hip::kernels::DFlashAttentionSplitScratchElements(
          block_count, kLongQueryHeads, kLongHeadDim);
  superroc::test::DeviceBuffer<float> d_scratch(scratch_elements);
  const float scale = 1.0F / std::sqrt(static_cast<float>(kLongHeadDim));
  hipEvent_t begin, end;
  HIP_CHECK(hipEventCreate(&begin));
  HIP_CHECK(hipEventCreate(&end));
  const auto time = [&](auto&& launch) {
    for (int i = 0; i < 5; ++i)
      launch();
    HIP_CHECK(hipEventRecord(begin, nullptr));
    constexpr int kIterations = 50;
    for (int i = 0; i < kIterations; ++i)
      launch();
    HIP_CHECK(hipEventRecord(end, nullptr));
    HIP_CHECK(hipEventSynchronize(end));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, begin, end));
    return 1000.0F * ms / kIterations;
  };
  for (const std::uint32_t history_length :
       {1566U, 2048U, 4096U, 8192U, 13384U, 26821U}) {
    const float plain = time([&] {
      superroc::hip::kernels::LaunchDFlashNonCausalAttention(
          d_q.data(), d_history_k.data(), d_history_v.data(), d_block_k.data(),
          d_block_v.data(), d_out.data(), history_length, history_length,
          block_count, kLongWindow, kLongQueryHeads, kLongKvHeads, kLongHeadDim,
          scale, nullptr, kLongCapacity);
    });
    std::cout << "BENCH history=" << history_length
              << " route=plain us=" << plain << "\n";
    for (const std::uint32_t split_keys : {0U, 128U, 256U, 512U, 1024U}) {
      const float split = time([&] {
        superroc::hip::kernels::LaunchDFlashNonCausalAttentionSplit(
            d_q.data(), d_history_k.data(), d_history_v.data(),
            d_block_k.data(), d_block_v.data(), d_out.data(), history_length,
            history_length, block_count, kLongWindow, kLongQueryHeads,
            kLongKvHeads, kLongHeadDim, scale, nullptr, kLongCapacity,
            d_scratch.data(), scratch_elements, split_keys);
      });
      std::cout << "BENCH history=" << history_length << " route=split"
                << split_keys << " us=" << split << "\n";
    }
  }
  HIP_CHECK(hipEventDestroy(begin));
  HIP_CHECK(hipEventDestroy(end));
}

}  // namespace

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      superroc::test::GateHipDevice(superroc::test::HipDeviceRequirement::kOptional,
                                "Qwen DFlash non-causal attention ops test");
  if (device_status != superroc::test::kHipTestSuccess) {
    return device_status;
  }

  TestDynamicConvolution();
  TestSelectorZeroUniform();
  TestSelector(2065);
  TestSelector(248320);
  TestBatchedSelector();
  // No injected history at all: only the eight in-block keys are attended.
  TestRouteEquivalence(0, 0, "empty history");
  // Shallow: the whole history is inside the sliding window.
  TestRouteEquivalence(128, 128, "128-token history");
  // Prefetch boundaries and unequal draft widths cover both head-group routes.
  for (const auto rows : {1U, 3U, 4U, 7U}) {
    TestRouteEquivalence(31, 31, "short prefetch tail", rows);
    TestRouteEquivalence(32, 32, "complete prefetch tile", rows);
    TestRouteEquivalence(33, 33, "prefetch tile plus tail", rows);
    TestRouteEquivalence(2051, 2051, "ragged wrapped window", rows);
  }
  // Deep enough that the window clips the history, which is the case the
  // coalesced route exists for.
  TestRouteEquivalence(2047, 2047, "window minus one");
  TestRouteEquivalence(2048, 2048, "window boundary");
  TestRouteEquivalence(2051, 2051, "wrapped window");
  TestRouteEquivalence(4096, 4096, "4096-token history, window-clipped");
  // One-key tiles: every draft row past the first starts with an all-masked
  // tile, which the online softmax must cancel once a visible key raises the
  // running maximum.
  for (const auto rows : {4U, 7U}) {
    superroc::test::Expect(
        MaxRouteDifference(Route::kTiled, 2051, 2051, rows, 1) < 1e-5F,
        "DFlash tiled attention with one-key tiles diverges");
  }
  // DSpark (no sliding window): the score row outgrows LDS at 16,128 keys.
  // The p16k prompt that failed in the server was 26,821 tokens.
  TestLongHistory(16124, 4, "16,128 keys, last whole row");
  TestLongHistory(16125, 4, "16,129 keys, first tiled row");
  TestLongHistory(26821, 4, "26,821-token prompt, paired heads");
  TestLongHistory(26821, 1, "26,821-token prompt, single head");
  // Split route: the production entry point with scratch, a ragged final
  // range, one and three draft rows, and forced ranges past the 64-range cap.
  TestLongHistory(26821, 4, "26,821-token prompt, split",
                  LongRoute::kRouterSplit);
  TestLongHistory(26821, 1, "26,821-token prompt, split, one row",
                  LongRoute::kRouterSplit);
  TestLongHistory(4093, 3, "4,096 keys, split threshold, three rows",
                  LongRoute::kRouterSplit);
  TestLongHistory(26821, 4, "26,821-token prompt, 100-key ranges capped",
                  LongRoute::kSplitForced, 100);
  // One-key ranges under a window of eight: for draft row t the first t
  // ranges hold only masked keys, which must enter the combine with weight 0.
  for (const auto rows : {1U, 4U, 7U}) {
    superroc::test::Expect(MaxRouteDifference(Route::kSplit, 20, 20, rows,
                                          kTestTileKeys, 1, 8) < 1e-5F,
                       "DFlash split attention with masked ranges diverges");
  }
  // Below the threshold the router is unchanged, bit for bit.
  for (const auto history :
       {0U, 1566U, superroc::hip::kernels::kDFlashSplitMinKeys - 5U}) {
    TestRouterShortHistoryUnchanged(history, 4);
    TestRouterShortHistoryUnchanged(history, 1);
  }
  if (const char* bench = ::superroc::core::GetEnvCompat("SUPERROC_DFLASH_ATTN_BENCH");
      bench != nullptr && bench[0] == '1') {
    BenchLongHistory();
  }
  std::cout << "Qwen DFlash non-causal attention ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen DFlash non-causal attention ops "
               "test.\n";
  return 77;
#endif
}
