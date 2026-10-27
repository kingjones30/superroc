// Oracle test for the KAT (16,2,256) instantiation of the masked WMMA prefill
// attention (KAT-ATTN).
//
// Reference: the baseline LaunchBatchedAttention over the same visible key
// range. Both read the same FP16 K/V cache bytes (the candidate packs them
// with the new <16,2> pack kernel), the same causal rule and the same gate
// epilogue, so they agree to FP16 output precision rather than exactly.
//
// Tolerances (fixed before the run; not tuned to its output):
//   * WMMA vs baseline context: rel (max_abs / magnitude) <= 2e-3 -- the same
//     bar as the 27B WMMA-vs-tiled oracle -- AND per-row cosine >= 0.9999.
//   * WMMA vs the FP64 original-input oracle (first and last query of head
//     zero, plus their LSE values on the LSE path): rel <= 2e-3, same as 27B.
//   * Replay, chunk boundaries (budgets 31/512) and packed-vs-canonical
//     workspace runs: byte-identical, like the 27B oracle.
//   * F32 K/V cache after the pack: bitwise identical to the baseline's write.
//
// Covered: spans 1K/4K/8K at start 0 and at depth, a 16K production-shaped span
// (12288+4096; the baseline needs span*4 dynamic LDS, so that case compares
// packed-vs-canonical plus the FP64 oracle instead), partial tiles
// (batch % 32, length % 16), unaligned depth, nonzero key_begin (split-half
// contract, aligned and unaligned), the LSE path, and the group loop
// (heads_per_group < Hkv) via constrained workspace spans with guard bytes.
//
// Also asserts the dispatch contract: the shared IsTiledAttentionSupported
// still rejects (16,2) (the tiled kernel stays (24,4)-only), the WMMA launcher
// accepts (16,2), (24,4) and (16,4) (SQ17 MiMo), and rejects other head
// configs.
//
// SQ17: the MiMo (16,4,256) instantiation runs the same oracle (RunCase is
// templated on head config), and CheckBaselineGuard proves the baseline
// launcher refuses an over-LDS span with length_error instead of faulting.
// SQ17-2 (F2): two (16,4) cases past the guarded boundary, the guard's
// positive side (a fitting span must not throw), and the timing gate reads
// the device block-LDS limit instead of a hardcoded 64 KiB.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

constexpr std::uint32_t kNumHeads = 16;
constexpr std::uint32_t kNumKvHeads = 2;
constexpr std::uint32_t kHeadDim = 256;
constexpr double kRelTolerance = 2e-3;
constexpr double kMinCosine = 0.9999;

class Rng {
public:
  explicit Rng(std::uint32_t seed) : state_(seed) {}
  float Uniform(float lo, float hi) {
    state_ = (state_ * 1664525U) + 1013904223U;
    const float u = static_cast<float>((state_ >> 8) & 0xFFFFU) / 65535.0F;
    return lo + (u * (hi - lo));
  }

private:
  std::uint32_t state_;
};

void Compare(const char* label, const std::vector<float>& got,
             const std::vector<float>& want, std::size_t row_stride) {
  double max_abs = 0.0;
  double magnitude = 0.0;
  double min_cosine = 1.0;
  std::size_t non_finite = 0;
  std::size_t rows_checked = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (!std::isfinite(got[i])) {
      ++non_finite;
      continue;
    }
    max_abs = std::fmax(max_abs, std::fabs(static_cast<double>(got[i]) -
                                           static_cast<double>(want[i])));
    magnitude = std::fmax(magnitude, std::fabs(static_cast<double>(want[i])));
  }
  if (row_stride != 0 && want.size() % row_stride == 0) {
    for (std::size_t row = 0; row < want.size() / row_stride; ++row) {
      double dot = 0.0, aa = 0.0, bb = 0.0;
      for (std::size_t j = 0; j < row_stride; ++j) {
        const double a = got[row * row_stride + j];
        const double b = want[row * row_stride + j];
        if (!std::isfinite(a) || !std::isfinite(b))
          break;
        dot += a * b;
        aa += a * a;
        bb += b * b;
      }
      if (aa > 1e-12 && bb > 1e-12) {
        min_cosine = std::fmin(min_cosine, dot / std::sqrt(aa * bb));
        ++rows_checked;
      }
    }
  }
  const double rel = (magnitude > 0.0) ? (max_abs / magnitude) : max_abs;
  std::cout << "  " << label << ": max_abs=" << max_abs
            << " magnitude=" << magnitude << " rel=" << rel
            << " min_cosine=" << min_cosine << " rows=" << rows_checked
            << " non_finite=" << non_finite << "\n";
  if (non_finite != 0 || rel > kRelTolerance || min_cosine < kMinCosine) {
    std::cerr << label
              << ": KAT WMMA prefill attention disagrees with the reference\n";
    std::abort();
  }
}

// SQ17: templated on head config so the MiMo (16,4) instantiation runs the
// same oracle. The locals shadow the KAT file constants; <16,2> folds to the
// same values as before.
template<std::uint32_t Hq, std::uint32_t Hkv>
void RunCase(std::uint32_t start_pos, std::size_t batch_size, bool want_lse,
             std::uint32_t key_begin, bool use_baseline) {
  constexpr std::uint32_t kNumHeads = Hq;
  constexpr std::uint32_t kNumKvHeads = Hkv;
  std::cout << "wmma prefill attention (" << Hq << "," << Hkv
            << "): start_pos=" << start_pos << " batch=" << batch_size
            << " lse=" << (want_lse ? "yes" : "no")
            << " key_begin=" << key_begin
            << " baseline=" << (use_baseline ? "yes" : "no") << "\n";

  const auto kMaxContext = std::max<std::uint32_t>(
      1024, start_pos + static_cast<std::uint32_t>(batch_size) + 64);
  const std::size_t attention_size =
      static_cast<std::size_t>(kNumHeads) * kHeadDim;
  const std::size_t kv_size = static_cast<std::size_t>(kNumKvHeads) * kHeadDim;
  const std::size_t total_kv =
      static_cast<std::size_t>(kNumKvHeads) * kMaxContext * kHeadDim;
  const std::size_t q_elements = batch_size * attention_size;
  const std::size_t lse_elements = batch_size * kNumHeads;

  Rng rng(0xCA7U ^ (start_pos * 131U) ^ static_cast<std::uint32_t>(batch_size));
  std::vector<float> h_q(q_elements);
  std::vector<float> h_gate(q_elements);
  std::vector<float> h_k(batch_size * kv_size);
  std::vector<float> h_v(batch_size * kv_size);
  std::vector<float> h_cache(total_kv * 2);
  for (auto& x : h_q) {
    x = rng.Uniform(-1.0F, 1.0F);
  }
  for (auto& x : h_gate) {
    x = rng.Uniform(-2.0F, 2.0F);
  }
  for (auto& x : h_k) {
    x = rng.Uniform(-1.0F, 1.0F);
  }
  for (auto& x : h_v) {
    x = rng.Uniform(-1.0F, 1.0F);
  }
  for (auto& x : h_cache) {
    x = rng.Uniform(-1.0F, 1.0F);
  }

  float* d_q = nullptr;
  float* d_gate = nullptr;
  float* d_k = nullptr;
  float* d_v = nullptr;
  float* d_cache = nullptr;
  void* d_cache_f16 = nullptr;
  float* d_out_ref = nullptr;
  float* d_out_new = nullptr;
  float* d_lse_new = nullptr;
  HIP_CHECK(hipMalloc(&d_q, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, batch_size * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, batch_size * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache, total_kv * 2 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_f16, total_kv * 2 * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out_ref, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_new, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_lse_new, lse_elements * sizeof(float)));

  const auto upload = [](float* dst, const std::vector<float>& src) {
    HIP_CHECK(hipMemcpy(dst, src.data(), src.size() * sizeof(float),
                        hipMemcpyHostToDevice));
  };
  upload(d_q, h_q);
  upload(d_gate, h_gate);
  upload(d_k, h_k);
  upload(d_v, h_v);
  upload(d_cache, h_cache);
  HIP_CHECK(hipMemset(d_cache_f16, 0, total_kv * 2 * sizeof(std::uint16_t)));

  auto* v_cache = d_cache + total_kv;
  auto* cache_f16_v = static_cast<std::uint16_t*>(d_cache_f16) + total_kv;
  float* lse_new = want_lse ? d_lse_new : nullptr;

  std::vector<float> base_cache;
  if (use_baseline) {
    // Reference: the baseline batched attention, which also writes the cache.
    HIP_CHECK(hipMemset(d_out_ref, 0, q_elements * sizeof(float)));
    superroc::hip::LaunchBatchedAttention(
        d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16, cache_f16_v,
        d_out_ref, /*layer_idx=*/0, start_pos, batch_size, kMaxContext,
        kNumHeads, kNumKvHeads, kHeadDim, nullptr,
        /*skip_kv_write=*/false);
    HIP_CHECK(hipDeviceSynchronize());
    base_cache.resize(total_kv * 2);
    HIP_CHECK(hipMemcpy(base_cache.data(), d_cache,
                        base_cache.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    // Reset the caches so the candidate's own pack is what its kernel reads.
    upload(d_cache, h_cache);
    HIP_CHECK(hipMemset(d_cache_f16, 0, total_kv * 2 * sizeof(std::uint16_t)));
  }

  // Candidate: the WMMA kernel over the same range, packing the FP16 cache
  // with the new <16,2> pack kernel.
  HIP_CHECK(hipMemset(d_out_new, 0, q_elements * sizeof(float)));
  if (!superroc::hip::LaunchQwenWmmaAttention(
          d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16, cache_f16_v,
          d_out_new, /*layer_idx=*/0, start_pos, batch_size, kMaxContext,
          kNumHeads, kNumKvHeads, kHeadDim, nullptr, lse_new, key_begin,
          /*skip_kv_write=*/false)) {
    std::cerr << "WMMA attention rejected the (" << Hq << "," << Hkv
              << ") shape\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> got(q_elements);
  HIP_CHECK(hipMemcpy(got.data(), d_out_new, q_elements * sizeof(float),
                      hipMemcpyDeviceToHost));
  if (use_baseline) {
    std::vector<float> ref(q_elements);
    HIP_CHECK(hipMemcpy(ref.data(), d_out_ref, q_elements * sizeof(float),
                        hipMemcpyDeviceToHost));
    Compare("context", got, ref, attention_size);
    // The pack must write the same persistent FP32 bytes as the baseline.
    std::vector<float> packed_cache(total_kv * 2);
    HIP_CHECK(hipMemcpy(packed_cache.data(), d_cache,
                        packed_cache.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    if (std::memcmp(packed_cache.data(), base_cache.data(),
                    packed_cache.size() * sizeof(float)) != 0) {
      std::cerr << "(" << Hq << "," << Hkv
                << ") pack wrote different FP32 cache bytes than baseline\n";
      std::abort();
    }
    std::cout << "  cache: bitwise identical to baseline\n";
  }

  // Independent FP64 attention over the original FP32 inputs, before the
  // kernel rounds Q/K/V and probabilities. Include the first and last query
  // of head zero so partial causal tiles cannot hide behind kernel agreement.
  // On the LSE path also check the two LSE values (the gate is suppressed).
  std::vector<float> oracle, selected, oracle_lse, selected_lse;
  for (const auto row : {std::size_t{0}, batch_size - 1}) {
    const auto end = start_pos + row + 1;
    std::vector<double> scores(end, -INFINITY);
    double maximum = -INFINITY;
    for (std::size_t key = key_begin; key < end; ++key) {
      double dot = 0;
      for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
        const float value = key < start_pos
                                ? h_cache[key * kHeadDim + dim]
                                : h_k[(key - start_pos) * kv_size + dim];
        dot += double(h_q[row * attention_size + dim]) * value;
      }
      scores[key] = dot / std::sqrt(double(kHeadDim));
      maximum = std::max(maximum, scores[key]);
    }
    double sum = 0;
    for (std::size_t key = key_begin; key < end; ++key) {
      scores[key] = std::exp(scores[key] - maximum);
      sum += scores[key];
    }
    if (want_lse) {
      oracle_lse.push_back(
          static_cast<float>(sum > 0 ? maximum + std::log(sum) : -INFINITY));
    }
    for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
      double value = 0;
      for (std::size_t key = key_begin; key < end; ++key) {
        const float v = key < start_pos
                            ? h_cache[total_kv + key * kHeadDim + dim]
                            : h_v[(key - start_pos) * kv_size + dim];
        value += scores[key] * v;
      }
      value = sum > 0 ? value / sum : 0;
      if (!want_lse)
        value /= 1 + std::exp(-double(h_gate[row * attention_size + dim]));
      oracle.push_back(static_cast<float>(value));
      selected.push_back(got[row * attention_size + dim]);
    }
  }
  Compare("FP64 original-input oracle", selected, oracle, kHeadDim);
  if (want_lse) {
    std::vector<float> lse_host(lse_elements);
    HIP_CHECK(hipMemcpy(lse_host.data(), d_lse_new,
                        lse_elements * sizeof(float), hipMemcpyDeviceToHost));
    for (const auto row : {std::size_t{0}, batch_size - 1}) {
      selected_lse.push_back(lse_host[row]);  // head zero, [head, token]
    }
    Compare("FP64 LSE oracle", selected_lse, oracle_lse, 0);
  }

  // Same launch again: the prefetch must not let a register from one tile reach
  // another, which would show up here as run-to-run drift.
  HIP_CHECK(hipMemset(d_out_new, 0, q_elements * sizeof(float)));
  if (!superroc::hip::LaunchQwenWmmaAttention(
          d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16, cache_f16_v,
          d_out_new, /*layer_idx=*/0, start_pos, batch_size, kMaxContext,
          kNumHeads, kNumKvHeads, kHeadDim, nullptr, lse_new, key_begin,
          /*skip_kv_write=*/true)) {
    std::cerr << "WMMA attention rejected the (" << Hq << "," << Hkv
              << ") shape on replay\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> again(q_elements);
  HIP_CHECK(hipMemcpy(again.data(), d_out_new, q_elements * sizeof(float),
                      hipMemcpyDeviceToHost));
  if (std::memcmp(again.data(), got.data(), q_elements * sizeof(float)) != 0) {
    std::size_t differing = 0;
    for (std::size_t i = 0; i < q_elements; ++i) {
      differing += (again[i] != got[i]) ? 1 : 0;
    }
    std::cerr << "(" << Hq << "," << Hkv
              << ") WMMA prefill attention is not deterministic: " << differing
              << " of " << q_elements << " elements differ between two runs\n";
    std::abort();
  }
  std::cout << "  replay: byte-identical over " << q_elements << " elements\n";

  if (!want_lse && batch_size > 1) {
    // Populate future KV once, then vary only the chunk boundaries. A masked
    // future value must have exactly the same effect as absent padding.
    for (const std::size_t budget : {std::size_t{31}, std::size_t{512}}) {
      for (std::size_t offset = 0; offset < batch_size; offset += budget) {
        const auto count = std::min(budget, batch_size - offset);
        if (!superroc::hip::LaunchQwenWmmaAttention(
                d_q + offset * attention_size, d_k, d_v,
                d_gate + offset * attention_size, d_cache, v_cache, d_cache_f16,
                cache_f16_v, d_out_new + offset * attention_size, 0,
                start_pos + offset, count, kMaxContext, kNumHeads, kNumKvHeads,
                kHeadDim, nullptr, nullptr, key_begin, true))
          std::abort();
      }
      HIP_CHECK(hipMemcpy(again.data(), d_out_new, q_elements * sizeof(float),
                          hipMemcpyDeviceToHost));
      if (std::memcmp(again.data(), got.data(), q_elements * sizeof(float)) !=
          0) {
        std::cerr << "(" << Hq << "," << Hkv
                  << ") WMMA attention changed across chunk boundaries\n";
        std::abort();
      }
    }
    std::cout << "  chunk boundaries: byte-identical\n";
  }

  if (start_pos >= 8192 || batch_size >= 1024) {
    const std::size_t length = start_pos + batch_size;
    const std::array<std::size_t, 2> words{
        length * kv_size / 2,
        ((length + 15u) & ~std::size_t{15}) * kv_size / 2};
    std::array<float*, 2> storage{};
    std::array<std::span<float>, 2> work{};
    for (std::size_t i = 0; i < storage.size(); ++i) {
      HIP_CHECK(hipMalloc(&storage[i], (words[i] + 32) * sizeof(float)));
      HIP_CHECK(hipMemset(storage[i], 0xA5, (words[i] + 32) * sizeof(float)));
      work[i] = {storage[i] + 16, words[i]};
    }
    std::vector<float> lse_expected(lse_elements);
    if (want_lse)
      HIP_CHECK(hipMemcpy(lse_expected.data(), d_lse_new,
                          lse_elements * sizeof(float), hipMemcpyDeviceToHost));
    // (key_heads, value_heads) of the KV heads the workspace fits; the
    // single-head rows force the group loop, the zero rows force the
    // canonical-cache fallback. For KAT this is the historical six rows.
    const std::uint32_t kHeads = kNumKvHeads;
    const std::pair<std::uint32_t, std::uint32_t> kPairs[] = {
        {kHeads, kHeads}, {kHeads / 2, kHeads}, {kHeads, kHeads / 2},
        {1u, 1u},         {0u, kHeads},         {kHeads, 0u}};
    for (const auto& [key_heads, value_heads] : kPairs) {
      for (std::size_t i = 0; i < storage.size(); ++i)
        HIP_CHECK(hipMemset(storage[i], 0xA5, (words[i] + 32) * sizeof(float)));
      const std::array<std::size_t, 2> sizes{
          key_heads ? words[0] / kNumKvHeads * key_heads
                    : words[0] / kNumKvHeads - 1,
          value_heads ? words[1] / kNumKvHeads * value_heads
                      : words[1] / kNumKvHeads - 1};
      const auto k_work = work[0].first(sizes[0]);
      const auto v_work = work[1].first(sizes[1]);
      if (!superroc::hip::LaunchQwenWmmaAttention(
              d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16, cache_f16_v,
              d_out_new, 0, start_pos, batch_size, kMaxContext, kNumHeads,
              kNumKvHeads, kHeadDim, nullptr, lse_new, key_begin, true, k_work,
              v_work))
        std::abort();
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(again.data(), d_out_new, q_elements * sizeof(float),
                          hipMemcpyDeviceToHost));
      if (std::memcmp(again.data(), got.data(), q_elements * sizeof(float)) !=
          0)
        std::abort();
      if (want_lse) {
        std::vector<float> actual(lse_elements);
        HIP_CHECK(hipMemcpy(actual.data(), d_lse_new,
                            lse_elements * sizeof(float),
                            hipMemcpyDeviceToHost));
        if (std::memcmp(actual.data(), lse_expected.data(),
                        lse_elements * sizeof(float)) != 0)
          std::abort();
      }
      for (std::size_t i = 0; i < storage.size(); ++i) {
        float* data = storage[i];
        std::array<std::uint8_t, 64> first{};
        HIP_CHECK(hipMemcpy(first.data(), data + 16, first.size(),
                            hipMemcpyDeviceToHost));
        const bool untouched = std::all_of(
            first.begin(), first.end(), [](auto byte) { return byte == 0xA5; });
        if (untouched ==
            (key_heads != 0 && value_heads != 0 && batch_size >= 1024 &&
             key_begin < length && key_begin % 16 == 0))
          std::abort();
        for (std::size_t off : {std::size_t{0}, sizes[i] + 16}) {
          std::array<std::uint8_t, 64> guard{};
          HIP_CHECK(hipMemcpy(guard.data(), data + off, guard.size(),
                              hipMemcpyDeviceToHost));
          if (!std::all_of(guard.begin(), guard.end(),
                           [](auto byte) { return byte == 0xA5; }))
            std::abort();
        }
      }
    }
    for (float* data : storage)
      HIP_CHECK(hipFree(data));
    std::cout << "  contiguous heads: exact output/LSE, bounded scratch and "
                 "fallback\n";
  }

  for (float* p :
       {d_q, d_gate, d_k, d_v, d_cache, d_out_ref, d_out_new, d_lse_new}) {
    HIP_CHECK(hipFree(p));
  }
  HIP_CHECK(hipFree(d_cache_f16));
}

// HIP-event timing of one full-attention layer on production chunk shapes:
// baseline vs WMMA canonical-cache vs WMMA packed (single-group and grouped).
// Informational only (like the route test's us_per_call); the only aborts
// are rejected launches. Multiply by 10 (KAT attention layers) for a chunk's
// attention-stage total.
void BenchTiming() {
  // SQ17-2 (F2): the baseline-compare gate reads the device block-LDS limit,
  // like the launcher guard does, not a hardcoded 64 KiB.
  int bench_device = 0;
  HIP_CHECK(hipGetDevice(&bench_device));
  int bench_lds_limit = 0;
  HIP_CHECK(hipDeviceGetAttribute(&bench_lds_limit,
                                  hipDeviceAttributeMaxSharedMemoryPerBlock,
                                  bench_device));
  const std::pair<std::uint32_t, std::size_t> shapes[] = {
      {0, 1598},    {0, 4096},     {4096, 2267},  {4096, 4096},
      {8192, 4096}, {12288, 1128}, {12288, 4096},
  };
  const std::size_t attention_size =
      static_cast<std::size_t>(kNumHeads) * kHeadDim;
  const std::size_t kv_size = static_cast<std::size_t>(kNumKvHeads) * kHeadDim;
  for (const auto [start_pos, batch_size] : shapes) {
    const auto kMaxContext = std::max<std::uint32_t>(
        1024, start_pos + static_cast<std::uint32_t>(batch_size) + 64);
    const std::size_t total_kv =
        static_cast<std::size_t>(kNumKvHeads) * kMaxContext * kHeadDim;
    const std::size_t q_elements = batch_size * attention_size;
    Rng rng(0xBE71U ^ start_pos);
    std::vector<float> h_q(q_elements), h_gate(q_elements),
        h_kv(batch_size * kv_size), h_cache(total_kv * 2);
    for (auto& x : h_q)
      x = rng.Uniform(-1.0F, 1.0F);
    for (auto& x : h_gate)
      x = rng.Uniform(-2.0F, 2.0F);
    for (auto& x : h_kv)
      x = rng.Uniform(-1.0F, 1.0F);
    for (auto& x : h_cache)
      x = rng.Uniform(-1.0F, 1.0F);
    float *d_q = nullptr, *d_gate = nullptr, *d_k = nullptr, *d_v = nullptr,
          *d_cache = nullptr, *d_out = nullptr;
    void* d_cache_f16 = nullptr;
    HIP_CHECK(hipMalloc(&d_q, q_elements * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gate, q_elements * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_k, batch_size * kv_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_v, batch_size * kv_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_cache, total_kv * 2 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_cache_f16, total_kv * 2 * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&d_out, q_elements * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_q, h_q.data(), q_elements * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_elements * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_k, h_kv.data(), batch_size * kv_size * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_v, h_kv.data(), batch_size * kv_size * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cache, h_cache.data(), total_kv * 2 * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_cache_f16, 0, total_kv * 2 * sizeof(std::uint16_t)));
    auto* v_cache = d_cache + total_kv;
    auto* cache_f16_v = static_cast<std::uint16_t*>(d_cache_f16) + total_kv;
    const std::size_t length = start_pos + batch_size;
    const std::size_t words_k = length * kv_size / 2;
    const std::size_t words_v =
        ((length + 15u) & ~std::size_t{15}) * kv_size / 2;
    float *wk = nullptr, *wv = nullptr;
    HIP_CHECK(hipMalloc(&wk, words_k * sizeof(float)));
    HIP_CHECK(hipMalloc(&wv, words_v * sizeof(float)));
    const std::span<float> full_k{wk, words_k}, full_v{wv, words_v};
    const std::span<float> half_k{wk, words_k / kNumKvHeads};
    const std::span<float> half_v{wv, words_v / kNumKvHeads};
    hipEvent_t t0 = nullptr, t1 = nullptr;
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));
    const auto time_it = [&](const char* what, int iters, auto&& fn) {
      for (int i = 0; i < 2; ++i)
        fn();  // warmup
      HIP_CHECK(hipEventRecord(t0));
      for (int i = 0; i < iters; ++i)
        fn();
      HIP_CHECK(hipEventRecord(t1));
      HIP_CHECK(hipEventSynchronize(t1));
      float ms = 0;
      HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
      std::cout << "  BENCH start=" << start_pos << " batch=" << batch_size
                << " " << what << "_ms=" << (ms / iters) << "\n";
      return ms / iters;
    };
    // Pack once so timed runs read a packed cache (skip=true), like the
    // second and later layers of a chunk.
    if (!superroc::hip::LaunchQwenWmmaAttention(
            d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16, cache_f16_v,
            d_out, 0, start_pos, batch_size, kMaxContext, kNumHeads,
            kNumKvHeads, kHeadDim, nullptr, nullptr, 0, false))
      std::abort();
    HIP_CHECK(hipDeviceSynchronize());
    const bool baseline_fits =
        (256 + static_cast<std::size_t>(start_pos) + batch_size) *
            sizeof(float) <=
        static_cast<std::size_t>(bench_lds_limit);
    float base_ms = 0;
    if (baseline_fits) {
      base_ms = time_it("baseline", 3, [&] {
        superroc::hip::LaunchBatchedAttention(
            d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16, cache_f16_v,
            d_out, 0, start_pos, batch_size, kMaxContext, kNumHeads,
            kNumKvHeads, kHeadDim, nullptr, true);
      });
    }
    const float canon_ms = time_it("wmma_canonical", 10, [&] {
      if (!superroc::hip::LaunchQwenWmmaAttention(
              d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16, cache_f16_v,
              d_out, 0, start_pos, batch_size, kMaxContext, kNumHeads,
              kNumKvHeads, kHeadDim, nullptr, nullptr, 0, true))
        std::abort();
    });
    float packed_ms = 0, grouped_ms = 0;
    if (batch_size >= 1024) {
      packed_ms = time_it("wmma_packed", 10, [&] {
        if (!superroc::hip::LaunchQwenWmmaAttention(
                d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16,
                cache_f16_v, d_out, 0, start_pos, batch_size, kMaxContext,
                kNumHeads, kNumKvHeads, kHeadDim, nullptr, nullptr, 0, true,
                full_k, full_v))
          std::abort();
      });
      grouped_ms = time_it("wmma_grouped", 10, [&] {
        if (!superroc::hip::LaunchQwenWmmaAttention(
                d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16,
                cache_f16_v, d_out, 0, start_pos, batch_size, kMaxContext,
                kNumHeads, kNumKvHeads, kHeadDim, nullptr, nullptr, 0, true,
                half_k, half_v))
          std::abort();
      });
    }
    if (baseline_fits && base_ms > 0 && packed_ms > 0) {
      std::cout << "  BENCH start=" << start_pos << " batch=" << batch_size
                << " speedup_packed=" << (base_ms / packed_ms) << "x\n";
    }
    (void)canon_ms;
    (void)grouped_ms;
    HIP_CHECK(hipEventDestroy(t0));
    HIP_CHECK(hipEventDestroy(t1));
    for (float* p : {d_q, d_gate, d_k, d_v, d_cache, d_out, wk, wv}) {
      HIP_CHECK(hipFree(p));
    }
    HIP_CHECK(hipFree(d_cache_f16));
  }
}

// The dispatch contract: the shared tiled gate still rejects KAT (the tiled
// kernel stays (24,4)-only), the WMMA launcher accepts (16,2), (24,4) and
// (16,4) (SQ17 MiMo) and rejects anything else, so unknown shapes keep
// falling back to baseline.
void CheckDispatch() {
  constexpr std::size_t kBatch = 64;
  constexpr std::uint32_t kCtx = 1024;
  constexpr std::size_t kMaxAttn = 24 * 256;
  constexpr std::size_t kMaxKv = 4 * 256;
  constexpr std::size_t kMaxTotalKv = 4 * kCtx * 256;
  float *d_q = nullptr, *d_gate = nullptr, *d_k = nullptr, *d_v = nullptr;
  float *d_cache = nullptr, *d_out = nullptr;
  void* d_cache_f16 = nullptr;
  HIP_CHECK(hipMalloc(&d_q, kBatch * kMaxAttn * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, kBatch * kMaxAttn * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kBatch * kMaxKv * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kBatch * kMaxKv * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache, 2 * kMaxTotalKv * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_f16, 2 * kMaxTotalKv * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out, kBatch * kMaxAttn * sizeof(float)));
  HIP_CHECK(hipMemset(d_q, 0, kBatch * kMaxAttn * sizeof(float)));
  HIP_CHECK(hipMemset(d_gate, 0, kBatch * kMaxAttn * sizeof(float)));
  HIP_CHECK(hipMemset(d_k, 0, kBatch * kMaxKv * sizeof(float)));
  HIP_CHECK(hipMemset(d_v, 0, kBatch * kMaxKv * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache, 0, 2 * kMaxTotalKv * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_f16, 0, 2 * kMaxTotalKv * sizeof(std::uint16_t)));
  const auto wmma = [&](std::uint32_t hq, std::uint32_t hkv) {
    return superroc::hip::LaunchQwenWmmaAttention(
        d_q, d_k, d_v, d_gate, d_cache, d_cache + kMaxTotalKv, d_cache_f16,
        static_cast<std::uint16_t*>(d_cache_f16) + kMaxTotalKv, d_out, 0, 0,
        kBatch, kCtx, hq, hkv, kHeadDim);
  };
  if (!wmma(16, 2) || !wmma(24, 4) || !wmma(16, 4)) {
    std::cerr << "WMMA launcher rejected a supported head config\n";
    std::abort();
  }
  if (wmma(8, 2) || wmma(16, 1) || wmma(24, 2) || wmma(32, 8)) {
    std::cerr << "WMMA launcher accepted an unsupported head config\n";
    std::abort();
  }
  if (superroc::hip::LaunchBatchedAttentionTile(
          d_q, d_k, d_v, d_gate, d_cache, d_cache + kMaxTotalKv, d_cache_f16,
          static_cast<std::uint16_t*>(d_cache_f16) + kMaxTotalKv, d_out, 0, 0,
          kBatch, kCtx, 16, 2, kHeadDim)) {
    std::cerr << "tiled launcher accepted KAT; the shared gate widened\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());
  std::cout << "  dispatch: WMMA accepts (16,2)+(24,4)+(16,4), rejects the "
               "rest; tiled still rejects (16,2)\n";
  for (float* p : {d_q, d_gate, d_k, d_v, d_cache, d_out}) {
    HIP_CHECK(hipFree(p));
  }
  HIP_CHECK(hipFree(d_cache_f16));
}

// SQ17: the baseline launcher refuses an over-LDS span with length_error
// before anything is enqueued (MiMo's 27.7K failure class), instead of
// faulting the queue. The span derives from the device's own block limit so
// this holds on any GPU. Null device pointers are safe: the guard runs first.
void CheckBaselineGuard() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  int lds_limit = 0;
  HIP_CHECK(hipDeviceGetAttribute(
      &lds_limit, hipDeviceAttributeMaxSharedMemoryPerBlock, device));
  const auto span = static_cast<std::size_t>(lds_limit) / sizeof(float);
  bool threw = false;
  try {
    superroc::hip::LaunchBatchedAttention(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, /*layer_idx=*/0, /*start_pos=*/0, span,
        /*max_context=*/262144, /*num_heads=*/16, /*num_kv_heads=*/4,
        /*head_dim=*/kHeadDim, nullptr, /*skip_kv_write=*/false);
  } catch (const std::length_error& e) {
    threw = true;
    std::cout << "  guard: refused span " << span << ": " << e.what() << "\n";
  }
  if (!threw) {
    std::cerr << "baseline launcher accepted an over-LDS span\n";
    std::abort();
  }
  // SQ17-2 (F2): the positive side — a span that fits must not throw. Valid
  // device buffers this time: past the guard the launcher really enqueues.
  constexpr std::size_t kFitBatch = 1024;
  constexpr std::uint32_t kFitContext = 2048;
  constexpr std::uint32_t kFitHeads = 16;
  constexpr std::uint32_t kFitKvHeads = 4;
  const std::size_t fit_need = (256 + kFitBatch) * sizeof(float);
  if (fit_need > static_cast<std::size_t>(lds_limit)) {
    std::cerr << "device LDS limit " << lds_limit
              << " cannot stage the 1024-token fit case\n";
    std::abort();
  }
  const std::size_t fit_attn = static_cast<std::size_t>(kFitHeads) * kHeadDim;
  const std::size_t fit_kv = static_cast<std::size_t>(kFitKvHeads) * kHeadDim;
  const std::size_t fit_total =
      static_cast<std::size_t>(kFitKvHeads) * kFitContext * kHeadDim;
  float *f_q = nullptr, *f_gate = nullptr, *f_k = nullptr, *f_v = nullptr;
  float *f_cache = nullptr, *f_out = nullptr;
  void* f_cache_f16 = nullptr;
  HIP_CHECK(hipMalloc(&f_q, kFitBatch * fit_attn * sizeof(float)));
  HIP_CHECK(hipMalloc(&f_gate, kFitBatch * fit_attn * sizeof(float)));
  HIP_CHECK(hipMalloc(&f_k, kFitBatch * fit_kv * sizeof(float)));
  HIP_CHECK(hipMalloc(&f_v, kFitBatch * fit_kv * sizeof(float)));
  HIP_CHECK(hipMalloc(&f_cache, fit_total * 2 * sizeof(float)));
  HIP_CHECK(hipMalloc(&f_cache_f16, fit_total * 2 * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&f_out, kFitBatch * fit_attn * sizeof(float)));
  HIP_CHECK(hipMemset(f_q, 0, kFitBatch * fit_attn * sizeof(float)));
  HIP_CHECK(hipMemset(f_gate, 0, kFitBatch * fit_attn * sizeof(float)));
  HIP_CHECK(hipMemset(f_k, 0, kFitBatch * fit_kv * sizeof(float)));
  HIP_CHECK(hipMemset(f_v, 0, kFitBatch * fit_kv * sizeof(float)));
  HIP_CHECK(hipMemset(f_cache, 0, fit_total * 2 * sizeof(float)));
  HIP_CHECK(hipMemset(f_cache_f16, 0, fit_total * 2 * sizeof(std::uint16_t)));
  try {
    superroc::hip::LaunchBatchedAttention(
        f_q, f_k, f_v, f_gate, f_cache, f_cache + fit_total, f_cache_f16,
        static_cast<std::uint16_t*>(f_cache_f16) + fit_total, f_out,
        /*layer_idx=*/0, /*start_pos=*/0, kFitBatch, kFitContext, kFitHeads,
        kFitKvHeads, kHeadDim, nullptr, /*skip_kv_write=*/false);
    HIP_CHECK(hipDeviceSynchronize());
  } catch (const std::exception& e) {
    std::cerr << "baseline launcher refused a fitting span: " << e.what()
              << "\n";
    std::abort();
  }
  std::cout << "  guard: accepted fitting span " << kFitBatch << "\n";
  for (float* p : {f_q, f_gate, f_k, f_v, f_cache, f_out}) {
    HIP_CHECK(hipFree(p));
  }
  HIP_CHECK(hipFree(f_cache_f16));
}

}  // namespace

constexpr superroc::hip::detail::AttentionSupportParams kKatDispatchParams{
    .batch_size = 1024,
    .start_pos = 0,
    .max_context = 32768,
    .num_heads = 16,
    .num_kv_heads = 2,
    .head_dim = 256,
    .has_k_cache_f16 = true,
    .has_v_cache_f16 = true};
static_assert(!superroc::hip::detail::IsTiledAttentionSupported(kKatDispatchParams),
              "the shared tiled gate must keep rejecting KAT");
static_assert(
    superroc::hip::detail::IsWmmaAttentionSupportedFor<16, 2>(kKatDispatchParams),
    "the WMMA gate must accept KAT");

// SQ17: MiMo (16,4) dispatch contract at compile time.
constexpr superroc::hip::detail::AttentionSupportParams kMimoDispatchParams{
    .batch_size = 1024,
    .start_pos = 0,
    .max_context = 32768,
    .num_heads = 16,
    .num_kv_heads = 4,
    .head_dim = 256,
    .has_k_cache_f16 = true,
    .has_v_cache_f16 = true};
static_assert(
    !superroc::hip::detail::IsTiledAttentionSupported(kMimoDispatchParams),
    "the shared tiled gate must keep rejecting MiMo");
static_assert(
    superroc::hip::detail::IsWmmaAttentionSupportedFor<16, 4>(kMimoDispatchParams),
    "the WMMA gate must accept MiMo");

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      superroc::test::GateHipDevice(superroc::test::HipDeviceRequirement::kOptional,
                                "KAT WMMA prefill attention ops test");
  if (device_status != superroc::test::kHipTestSuccess) {
    return device_status;
  }

  CheckDispatch();
  CheckBaselineGuard();
  RunCase<16, 2>(0, 1024, false, 0, true);
  RunCase<16, 2>(0, 1025, false, 0, true);
  RunCase<16, 2>(0, 2048, false, 0, true);
  RunCase<16, 2>(0, 4096, false, 0, true);
  RunCase<16, 2>(0, 8192, false, 0, true);
  RunCase<16, 2>(2048, 2048, false, 0, true);
  RunCase<16, 2>(4096, 4096, false, 0, true);
  // 16K production-shaped span. The baseline needs span*4 dynamic LDS, so
  // this case proves packed-vs-canonical identity plus the FP64 oracle.
  RunCase<16, 2>(12288, 4096, false, 0, false);
  // Partial tiles, prefetch edges, unaligned depth.
  RunCase<16, 2>(0, 100, false, 0, true);
  RunCase<16, 2>(0, 16, false, 0, true);
  RunCase<16, 2>(0, 32, false, 0, true);
  RunCase<16, 2>(1500, 128, false, 0, true);
  RunCase<16, 2>(8192, 100, false, 0, true);
  RunCase<16, 2>(8192, 1023, false, 0, true);
  RunCase<16, 2>(8192, 1024, false, 0, true);
  RunCase<16, 2>(8192, 1025, false, 0, true);
  // LSE path (no baseline: it has no split-half output), split-half keys.
  RunCase<16, 2>(0, 2048, true, 0, false);
  RunCase<16, 2>(1024, 256, true, 0, false);
  RunCase<16, 2>(8192, 1024, true, 16, false);
  RunCase<16, 2>(8192, 1024, true, 17, false);
  RunCase<16, 2>(8192, 1024, false, 9216, false);
  // SQ17: MiMo (16,4) instantiation under the same oracle bars: production
  // chunks at start 0 and at depth, the 16K span (packed-vs-canonical plus
  // the FP64 oracle; the baseline cannot stage that span), partial tiles,
  // and the LSE path. SQ17-2 (F2): two cases past the guarded boundary
  // (visible 17408 and 24096; no baseline compare there either).
  RunCase<16, 4>(0, 1024, false, 0, true);
  RunCase<16, 4>(0, 2048, false, 0, true);
  RunCase<16, 4>(8192, 1024, false, 0, true);
  RunCase<16, 4>(12288, 4096, false, 0, false);
  RunCase<16, 4>(0, 100, false, 0, true);
  RunCase<16, 4>(8192, 1024, true, 16, false);
  RunCase<16, 4>(16384, 1024, false, 0, false);
  RunCase<16, 4>(20000, 4096, false, 0, false);

  BenchTiming();

  std::cout << "KAT+MiMo WMMA prefill attention ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping KAT WMMA prefill attention ops test.\n";
  return 77;
#endif
}
