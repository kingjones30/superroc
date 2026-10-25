// Oracle test for the Glimmer (32,2,128) instantiations of the masked WMMA
// prefill attention (LGA): windowless (the 13 full layers) and windowed (the
// 39 SWA layers).
//
// Reference 1: the fp32 LaunchGlimmerPrefillAttention over the same visible
// key range. Reference 2: an FP64 CPU oracle over the original FP32 inputs,
// written from GlimmerSwaBegin (keys [pos+1-W, pos], count min(pos+1, W)).
// The candidate packs nothing itself in production (skip_kv_write); the test
// feeds it the token-major fp16 plane the dual-write produces, plus one
// skip=false case pinning the <32,2,128> pack kernel against the fp32 write.
//
// Tolerances (fixed before the run; not tuned to its output):
//   * WMMA/shared vs fp32 context: rel (max_abs / magnitude) <= 2e-3 -- the
//     same bar as the 27B/KAT WMMA-vs-reference oracles -- AND per-row
//     cosine >= 0.9999 AND max_abs <= 2e-2 (the fp16-KV rounding floor LX1
//     documents for this pattern).
//   * WMMA/shared vs the FP64 original-input oracle (first and last query
//     of head zero, plus any row at pos W-1/W/W+1): rel <= 2e-3.
//   * Replay, chunk boundaries (budgets 31/512) and packed-vs-canonical
//     workspace runs: byte-identical, like the 27B/KAT oracles.
//   * F32 K/V cache after the pack: bitwise identical to the fp32 write,
//     and the packed fp16 plane bitwise identical to the dual-write plane.
//
// Covered: window 0 at spans 1K/4K/8K/16K at start 0 and at depth, windows
// 64/128/2048/2050 (2050 is not a multiple of 16), window edges
// (pos W-1/W/W+1, a block straddling pos-W, a chunk starting at start_pos =
// W, pos 0), key_begin aligned/unaligned/32 with a window, the LSE path on
// the windowless instantiation, LSE rejection on the windowed one, partial
// tiles (batch % 32, length % 16), unaligned depth, layer 1, and the group
// loop (heads_per_group < Hkv) via constrained workspace spans with guard
// bytes.
//
// Also asserts the dispatch contract: the shared IsTiledAttentionSupported
// still rejects (32,2,128) and (16,2), the WMMA launcher accepts
// (32,2,128) with window 0 and window > 0 plus (24,4,256) and (16,2,256),
// and rejects (32,2,256), (16,2,128) and (32,4,128).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/glimmer/hip/glimmer_attention.hpp"
#include "src/models/glimmer/hip/glimmer_decode.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

constexpr std::uint32_t kNumHeads = 32;
constexpr std::uint32_t kNumKvHeads = 2;
constexpr std::uint32_t kHeadDim = 128;
constexpr std::uint32_t kLayers = 2;
constexpr double kRelTolerance = 2e-3;
constexpr double kMinCosine = 0.9999;
constexpr double kMaxAbs = 2e-2;

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
  if (non_finite != 0 || rel > kRelTolerance || min_cosine < kMinCosine ||
      max_abs > kMaxAbs) {
    std::cerr << label
              << ": Glimmer WMMA prefill attention disagrees with the "
                 "reference\n";
    std::abort();
  }
}

// GlimmerSwaBegin from src/models/glimmer/glimmer_oracle.hpp: keys are
// [pos+1-W, pos], count min(pos+1, W). Window 0 is full attention.
std::uint32_t GlimmerSwaBegin(std::uint32_t pos, std::uint32_t window) {
  const std::uint32_t seq = pos + 1U;
  if (window == 0U || seq <= window) {
    return 0U;
  }
  return seq - window;
}

void RunCase(std::uint32_t start_pos, std::size_t batch_size,
             std::uint32_t window, bool want_lse, std::uint32_t key_begin,
             bool use_baseline, std::uint32_t layer = 0) {
  std::cout << "glimmer wmma prefill attention: start_pos=" << start_pos
            << " batch=" << batch_size << " window=" << window
            << " lse=" << (want_lse ? "yes" : "no")
            << " key_begin=" << key_begin
            << " baseline=" << (use_baseline ? "yes" : "no")
            << " layer=" << layer << "\n";

  const auto kMaxContext = std::max<std::uint32_t>(
      1024, start_pos + static_cast<std::uint32_t>(batch_size) + 64);
  const std::size_t attention_size =
      static_cast<std::size_t>(kNumHeads) * kHeadDim;
  const std::size_t kv_size = static_cast<std::size_t>(kNumKvHeads) * kHeadDim;
  const std::size_t total_kv =
      static_cast<std::size_t>(kLayers) * kNumKvHeads * kMaxContext * kHeadDim;
  const std::size_t q_elements = batch_size * attention_size;
  const std::size_t lse_elements = batch_size * kNumHeads;

  Rng rng(0x61A7U ^ (start_pos * 131U) ^
          static_cast<std::uint32_t>(batch_size) ^ (window * 17U) ^
          (layer * 101U));
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
  // The fp32 cache is head-major; the fp16 plane is token-major. History
  // rows [0, start_pos) come from h_cache, chunk rows from h_k/h_v.
  for (std::uint32_t l = 0; l < kLayers; ++l) {
    for (std::size_t row = 0; row < batch_size; ++row) {
      const std::size_t pos = start_pos + row;
      for (std::uint32_t h = 0; h < kNumKvHeads; ++h) {
        for (std::uint32_t d = 0; d < kHeadDim; ++d) {
          const std::size_t dst =
              (((static_cast<std::size_t>(l) * kNumKvHeads + h) * kMaxContext +
                pos) *
               kHeadDim) +
              d;
          const std::size_t src = (row * kNumKvHeads + h) * kHeadDim + d;
          h_cache[dst] = h_k[src];
          h_cache[total_kv + dst] = h_v[src];
        }
      }
    }
  }
  std::vector<__half> h_f16(total_kv * 2);
  for (std::uint32_t l = 0; l < kLayers; ++l) {
    for (std::uint32_t p = 0; p < kMaxContext; ++p) {
      for (std::uint32_t h = 0; h < kNumKvHeads; ++h) {
        for (std::uint32_t d = 0; d < kHeadDim; ++d) {
          const std::size_t f32 =
              (((static_cast<std::size_t>(l) * kNumKvHeads + h) * kMaxContext +
                p) *
               kHeadDim) +
              d;
          const std::size_t tm =
              (((static_cast<std::size_t>(l) * kMaxContext + p) * kNumKvHeads +
                h) *
               kHeadDim) +
              d;
          h_f16[tm] = __float2half(h_cache[f32]);
          h_f16[total_kv + tm] = __float2half(h_cache[total_kv + f32]);
        }
      }
    }
  }

  float* d_q = nullptr;
  float* d_gate = nullptr;
  float* d_k = nullptr;
  float* d_v = nullptr;
  float* d_cache = nullptr;
  void* d_cache_f16 = nullptr;
  float* d_out_ref = nullptr;
  float* d_out_new = nullptr;
  float* d_out_shared = nullptr;
  float* d_lse_new = nullptr;
  HIP_CHECK(hipMalloc(&d_q, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, batch_size * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, batch_size * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache, total_kv * 2 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_f16, total_kv * 2 * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out_ref, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_new, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_shared, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_lse_new, lse_elements * sizeof(float)));

  const auto upload = [](auto* dst, const auto& src, std::size_t bytes) {
    HIP_CHECK(hipMemcpy(dst, src.data(), bytes, hipMemcpyHostToDevice));
  };
  upload(d_q, h_q, q_elements * sizeof(float));
  upload(d_gate, h_gate, q_elements * sizeof(float));
  upload(d_k, h_k, batch_size * kv_size * sizeof(float));
  upload(d_v, h_v, batch_size * kv_size * sizeof(float));
  upload(d_cache, h_cache, total_kv * 2 * sizeof(float));
  upload(d_cache_f16, h_f16, total_kv * 2 * sizeof(std::uint16_t));

  auto* v_cache = d_cache + total_kv;
  auto* cache_f16_v = static_cast<std::uint16_t*>(d_cache_f16) + total_kv;
  float* lse_new = want_lse ? d_lse_new : nullptr;

  if (use_baseline) {
    // Reference: the fp32 prefill attention over the head-major cache.
    HIP_CHECK(hipMemset(d_out_ref, 0, q_elements * sizeof(float)));
    const auto ref = gufo::hip::LaunchGlimmerPrefillAttention(
        d_q, d_gate, d_cache, v_cache, d_out_ref, layer, start_pos,
        static_cast<std::uint32_t>(batch_size), window, kMaxContext, kNumHeads,
        kNumKvHeads, kHeadDim, q_elements, q_elements, total_kv, q_elements,
        nullptr);
    if (!ref.ok) {
      std::cerr << "fp32 reference refused: "
                << (ref.error != nullptr ? ref.error : "?") << "\n";
      std::abort();
    }
    HIP_CHECK(hipDeviceSynchronize());
  }

  // Fallback: the GQA-shared fp16 kernel over the token-major plane.
  HIP_CHECK(hipMemset(d_out_shared, 0, q_elements * sizeof(float)));
  const auto shared = gufo::hip::LaunchGlimmerSharedAttention(
      d_q, d_gate, d_cache_f16, cache_f16_v, d_out_shared, layer, start_pos,
      static_cast<std::uint32_t>(batch_size), window, kMaxContext, kNumHeads,
      kNumKvHeads, kHeadDim, q_elements, q_elements, total_kv, q_elements,
      nullptr);
  if (!shared.ok) {
    std::cerr << "shared kernel refused: "
              << (shared.error != nullptr ? shared.error : "?") << "\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());

  // Candidate: the WMMA kernel over the same plane. The LSE and key_begin
  // cases go through the public launcher directly; the rest use the
  // production wrapper (skip_kv_write, key_begin 0).
  HIP_CHECK(hipMemset(d_out_new, 0, q_elements * sizeof(float)));
  if (want_lse || key_begin != 0) {
    if (!gufo::hip::LaunchQwenWmmaAttention(
            d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16, cache_f16_v,
            d_out_new, layer, start_pos, batch_size, kMaxContext, kNumHeads,
            kNumKvHeads, kHeadDim, nullptr, lse_new, key_begin,
            /*skip_kv_write=*/true, {}, {}, window)) {
      std::cerr << "WMMA attention rejected the Glimmer shape\n";
      std::abort();
    }
  } else {
    const auto wmma = gufo::hip::LaunchGlimmerWmmaPrefillAttention(
        d_q, d_gate, d_cache_f16, cache_f16_v, d_out_new, layer, start_pos,
        static_cast<std::uint32_t>(batch_size), window, kMaxContext, kNumHeads,
        kNumKvHeads, kHeadDim, q_elements, q_elements, total_kv, q_elements, {},
        {}, nullptr);
    if (!wmma.ok) {
      std::cerr << "WMMA wrapper refused: "
                << (wmma.error != nullptr ? wmma.error : "?") << "\n";
      std::abort();
    }
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> got(q_elements);
  HIP_CHECK(hipMemcpy(got.data(), d_out_new, q_elements * sizeof(float),
                      hipMemcpyDeviceToHost));
  std::vector<float> got_shared(q_elements);
  HIP_CHECK(hipMemcpy(got_shared.data(), d_out_shared,
                      q_elements * sizeof(float), hipMemcpyDeviceToHost));
  if (use_baseline) {
    std::vector<float> ref(q_elements);
    HIP_CHECK(hipMemcpy(ref.data(), d_out_ref, q_elements * sizeof(float),
                        hipMemcpyDeviceToHost));
    Compare("wmma context", got, ref, attention_size);
    Compare("shared context", got_shared, ref, attention_size);
  }

  // Independent FP64 attention over the original FP32 inputs, before any
  // kernel rounds Q/K/V and probabilities. First and last query of head
  // zero, plus any row sitting at pos W-1/W/W+1, so the window edge cannot
  // hide behind kernel agreement. On the LSE path also check the LSE values
  // (the gate is suppressed there).
  std::vector<std::size_t> oracle_rows{0, batch_size - 1};
  if (window != 0) {
    for (std::uint32_t edge : {window - 1, window, window + 1}) {
      if (edge >= start_pos && edge < start_pos + batch_size) {
        oracle_rows.push_back(edge - start_pos);
      }
    }
  }
  std::sort(oracle_rows.begin(), oracle_rows.end());
  oracle_rows.erase(std::unique(oracle_rows.begin(), oracle_rows.end()),
                    oracle_rows.end());
  std::vector<float> oracle, selected, oracle_shared, selected_shared;
  std::vector<float> oracle_lse, selected_lse;
  for (const auto row : oracle_rows) {
    const std::uint32_t pos = start_pos + static_cast<std::uint32_t>(row);
    const std::uint32_t begin =
        std::max(GlimmerSwaBegin(pos, window), key_begin);
    const std::uint32_t end = pos + 1;
    std::vector<double> scores(end, -INFINITY);
    double maximum = -INFINITY;
    for (std::uint32_t key = begin; key < end; ++key) {
      double dot = 0;
      for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
        const std::size_t at =
            (((static_cast<std::size_t>(layer) * kNumKvHeads) * kMaxContext +
              key) *
             kHeadDim) +
            dim;
        dot += double(h_q[row * attention_size + dim]) * h_cache[at];
      }
      scores[key] = dot / std::sqrt(double(kHeadDim));
      maximum = std::max(maximum, scores[key]);
    }
    double sum = 0;
    for (std::uint32_t key = begin; key < end; ++key) {
      scores[key] = std::exp(scores[key] - maximum);
      sum += scores[key];
    }
    if (want_lse) {
      oracle_lse.push_back(
          static_cast<float>(sum > 0 ? maximum + std::log(sum) : -INFINITY));
    }
    for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
      double value = 0;
      for (std::uint32_t key = begin; key < end; ++key) {
        const std::size_t at =
            total_kv +
            (((static_cast<std::size_t>(layer) * kNumKvHeads) * kMaxContext +
              key) *
             kHeadDim) +
            dim;
        value += scores[key] * h_cache[at];
      }
      value = sum > 0 ? value / sum : 0;
      if (!want_lse)
        value /= 1 + std::exp(-double(h_gate[row * attention_size + dim]));
      oracle.push_back(static_cast<float>(value));
      selected.push_back(got[row * attention_size + dim]);
      oracle_shared.push_back(static_cast<float>(value));
      selected_shared.push_back(got_shared[row * attention_size + dim]);
    }
  }
  Compare("FP64 original-input oracle (wmma)", selected, oracle, kHeadDim);
  // The LSE path suppresses the gate, and key_begin > 0 is the split-half
  // contract; the shared kernel has neither mode (it always gates over the
  // full causal span), so its output cannot match the oracle there.
  if (!want_lse && key_begin == 0) {
    Compare("FP64 original-input oracle (shared)", selected_shared,
            oracle_shared, kHeadDim);
  }
  if (want_lse) {
    std::vector<float> lse_host(lse_elements);
    HIP_CHECK(hipMemcpy(lse_host.data(), d_lse_new,
                        lse_elements * sizeof(float), hipMemcpyDeviceToHost));
    for (const auto row : oracle_rows) {
      selected_lse.push_back(lse_host[row]);  // head zero, [head, token]
    }
    Compare("FP64 LSE oracle", selected_lse, oracle_lse, 0);
  }

  // Same launch again: the prefetch must not let a register from one tile
  // reach another, which would show up here as run-to-run drift.
  HIP_CHECK(hipMemset(d_out_new, 0, q_elements * sizeof(float)));
  if (want_lse || key_begin != 0) {
    if (!gufo::hip::LaunchQwenWmmaAttention(
            d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16, cache_f16_v,
            d_out_new, layer, start_pos, batch_size, kMaxContext, kNumHeads,
            kNumKvHeads, kHeadDim, nullptr, lse_new, key_begin,
            /*skip_kv_write=*/true, {}, {}, window)) {
      std::cerr << "WMMA attention rejected the Glimmer shape on replay\n";
      std::abort();
    }
  } else {
    const auto wmma = gufo::hip::LaunchGlimmerWmmaPrefillAttention(
        d_q, d_gate, d_cache_f16, cache_f16_v, d_out_new, layer, start_pos,
        static_cast<std::uint32_t>(batch_size), window, kMaxContext, kNumHeads,
        kNumKvHeads, kHeadDim, q_elements, q_elements, total_kv, q_elements, {},
        {}, nullptr);
    if (!wmma.ok) {
      std::cerr << "WMMA wrapper refused on replay\n";
      std::abort();
    }
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
    std::cerr << "Glimmer WMMA prefill attention is not deterministic: "
              << differing << " of " << q_elements
              << " elements differ between two runs\n";
    std::abort();
  }
  std::cout << "  replay: byte-identical over " << q_elements << " elements\n";

  if (!want_lse && key_begin == 0 && batch_size > 1) {
    // Vary only the chunk boundaries. A masked future value must have
    // exactly the same effect as absent padding, including at the window's
    // left edge (the per-row begin moves with the chunk).
    for (const std::size_t budget : {std::size_t{31}, std::size_t{512}}) {
      for (std::size_t offset = 0; offset < batch_size; offset += budget) {
        const auto count = std::min(budget, batch_size - offset);
        const auto wmma = gufo::hip::LaunchGlimmerWmmaPrefillAttention(
            d_q + offset * attention_size, d_gate + offset * attention_size,
            d_cache_f16, cache_f16_v, d_out_new + offset * attention_size,
            layer, start_pos + static_cast<std::uint32_t>(offset),
            static_cast<std::uint32_t>(count), window, kMaxContext, kNumHeads,
            kNumKvHeads, kHeadDim, q_elements - offset * attention_size,
            q_elements - offset * attention_size, total_kv,
            q_elements - offset * attention_size, {}, {}, nullptr);
        if (!wmma.ok) {
          std::cerr << "WMMA wrapper refused a chunk-boundary launch\n";
          std::abort();
        }
      }
      HIP_CHECK(hipMemcpy(again.data(), d_out_new, q_elements * sizeof(float),
                          hipMemcpyDeviceToHost));
      if (std::memcmp(again.data(), got.data(), q_elements * sizeof(float)) !=
          0) {
        std::cerr << "Glimmer WMMA attention changed across chunk boundaries\n";
        std::abort();
      }
    }
    std::cout << "  chunk boundaries: byte-identical\n";
  }

  if (!want_lse && key_begin == 0 &&
      (start_pos >= 8192 || batch_size >= 1024)) {
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
    // (key_heads, value_heads) of the 2 KV heads the workspace fits; the
    // single-head rows force the group loop, the zero rows force the
    // canonical-cache fallback.
    for (const auto [key_heads, value_heads] :
         {std::pair{2u, 2u}, std::pair{1u, 2u}, std::pair{2u, 1u},
          std::pair{1u, 1u}, std::pair{0u, 2u}, std::pair{2u, 0u}}) {
      for (std::size_t i = 0; i < storage.size(); ++i)
        HIP_CHECK(hipMemset(storage[i], 0xA5, (words[i] + 32) * sizeof(float)));
      const std::array<std::size_t, 2> sizes{
          key_heads ? words[0] / kNumKvHeads * key_heads
                    : words[0] / kNumKvHeads - 1,
          value_heads ? words[1] / kNumKvHeads * value_heads
                      : words[1] / kNumKvHeads - 1};
      const auto k_work = work[0].first(sizes[0]);
      const auto v_work = work[1].first(sizes[1]);
      const auto wmma = gufo::hip::LaunchGlimmerWmmaPrefillAttention(
          d_q, d_gate, d_cache_f16, cache_f16_v, d_out_new, layer, start_pos,
          static_cast<std::uint32_t>(batch_size), window, kMaxContext,
          kNumHeads, kNumKvHeads, kHeadDim, q_elements, q_elements, total_kv,
          q_elements, k_work, v_work, nullptr);
      if (!wmma.ok) {
        std::cerr << "WMMA wrapper refused a workspace launch\n";
        std::abort();
      }
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(again.data(), d_out_new, q_elements * sizeof(float),
                          hipMemcpyDeviceToHost));
      if (std::memcmp(again.data(), got.data(), q_elements * sizeof(float)) !=
          0)
        std::abort();
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
    std::cout << "  contiguous heads: exact output, bounded scratch and "
                 "fallback\n";
  }

  for (float* p : {d_q, d_gate, d_k, d_v, d_cache, d_out_ref, d_out_new,
                   d_out_shared, d_lse_new}) {
    HIP_CHECK(hipFree(p));
  }
  HIP_CHECK(hipFree(d_cache_f16));
}

// The <32,2,128> pack kernel must write the same persistent bytes as the
// Glimmer fp32 write (head-major f32) and the dual-write (token-major f16).
void CheckPack() {
  constexpr std::uint32_t kStart = 0;
  constexpr std::size_t kBatch = 1024;
  constexpr std::uint32_t kCtx = 2048;
  // PGA: the plane must hold MORE layers than the one being cast, or the
  // K/V half stride (the whole plane) and the cast range (one layer) coincide
  // and a wrong stride still lands in the right slot. Glimmer 30B has dozens
  // of attention layers, so the check runs on a 4-layer plane casting layer 1.
  constexpr std::uint32_t kPlaneLayers = 4;
  constexpr std::uint32_t kCastLayer = 1;
  const std::size_t kv_size = kNumKvHeads * kHeadDim;
  const std::size_t total_kv =
      static_cast<std::size_t>(kPlaneLayers) * kNumKvHeads * kCtx * kHeadDim;
  Rng rng(0x9AC4U);
  std::vector<float> h_k(kBatch * kv_size), h_v(kBatch * kv_size);
  for (auto& x : h_k)
    x = rng.Uniform(-1.0F, 1.0F);
  for (auto& x : h_v)
    x = rng.Uniform(-1.0F, 1.0F);
  float *d_k = nullptr, *d_v = nullptr, *d_cache = nullptr;
  void* d_cache_f16 = nullptr;
  HIP_CHECK(hipMalloc(&d_k, kBatch * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kBatch * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache, total_kv * 2 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_f16, total_kv * 2 * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), kBatch * kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), kBatch * kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  auto* v_cache = d_cache + total_kv;
  auto* cache_f16_v = static_cast<std::uint16_t*>(d_cache_f16) + total_kv;
  // Reference: the Glimmer batched write plus, on c20cf22, the token-major
  // fp16 cast. PGA: main's LaunchGlimmerBatchedWriteKv carries no fp16
  // dual-write -- lane LX1's never landed, and glimmer_decode.{hpp,hip} are
  // outside this lane's zone -- so the prefill path issues
  // LaunchGlimmerCastKvFp16 instead (glimmer_prefill.hip). Same plane, same
  // remap, so the bytes compared below are the ones the kernel reads.
  HIP_CHECK(hipMemset(d_cache, 0xA5, total_kv * 2 * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_f16, 0xA5, total_kv * 2 * sizeof(std::uint16_t)));
  const auto wrote = gufo::hip::LaunchGlimmerBatchedWriteKv(
      d_k, d_v, d_cache, v_cache, kCastLayer, kStart,
      static_cast<std::uint32_t>(kBatch), kCtx, kNumKvHeads, kHeadDim,
      kBatch * kv_size, total_kv, nullptr);
  if (!wrote.ok) {
    std::cerr << "fp32 kv write refused in pack check\n";
    std::abort();
  }
  const auto cast = gufo::hip::LaunchGlimmerCastKvFp16(
      d_cache, d_cache_f16, /*layers=*/1, kCtx, kNumKvHeads, kHeadDim,
      kStart + static_cast<std::uint32_t>(kBatch), nullptr,
      /*layer_base=*/kCastLayer, /*plane_layers=*/kPlaneLayers);
  if (!cast.ok) {
    std::cerr << "fp16 kv cast refused in pack check\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> want_f32(total_kv * 2);
  std::vector<std::uint16_t> want_f16(total_kv * 2);
  HIP_CHECK(hipMemcpy(want_f32.data(), d_cache, want_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(want_f16.data(), d_cache_f16,
                      want_f16.size() * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  // Candidate: the WMMA launcher with skip_kv_write=false runs the
  // <32,2,128> pack kernel over the same rows (layer 1).
  HIP_CHECK(hipMemset(d_cache, 0xA5, total_kv * 2 * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_f16, 0xA5, total_kv * 2 * sizeof(std::uint16_t)));
  float* d_q = nullptr;
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_q, kBatch * kNumHeads * kHeadDim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, kBatch * kNumHeads * kHeadDim * sizeof(float)));
  HIP_CHECK(hipMemset(d_q, 0, kBatch * kNumHeads * kHeadDim * sizeof(float)));
  if (!gufo::hip::LaunchQwenWmmaAttention(
          d_q, d_k, d_v, nullptr, d_cache, v_cache, d_cache_f16, cache_f16_v,
          d_out, /*layer_idx=*/kCastLayer, kStart, kBatch, kCtx, kNumHeads,
          kNumKvHeads, kHeadDim, nullptr, nullptr, 0, /*skip_kv_write=*/false,
          {}, {},
          /*window=*/0)) {
    std::cerr << "WMMA launcher refused the pack check\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got_f32(total_kv * 2);
  std::vector<std::uint16_t> got_f16(total_kv * 2);
  HIP_CHECK(hipMemcpy(got_f32.data(), d_cache, got_f32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(got_f16.data(), d_cache_f16,
                      got_f16.size() * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  if (std::memcmp(got_f32.data(), want_f32.data(),
                  got_f32.size() * sizeof(float)) != 0) {
    std::cerr << "pack wrote different FP32 cache bytes than the fp32 write\n";
    std::abort();
  }
  if (std::memcmp(got_f16.data(), want_f16.data(),
                  got_f16.size() * sizeof(std::uint16_t)) != 0) {
    std::cerr << "pack wrote different fp16 bytes than the fp32 write + cast\n";
    std::abort();
  }
  std::cout << "  pack: f32 and f16 planes bitwise identical to the writes\n";
  for (float* p : {d_k, d_v, d_cache, d_q, d_out}) {
    HIP_CHECK(hipFree(p));
  }
  HIP_CHECK(hipFree(d_cache_f16));
}

std::uint64_t CausalPairs(std::uint32_t start, std::size_t batch,
                          std::uint32_t window) {
  std::uint64_t pairs = 0;
  for (std::size_t i = 0; i < batch; ++i) {
    const std::uint32_t keys = start + static_cast<std::uint32_t>(i) + 1;
    pairs += window == 0 ? keys : std::min(keys, window);
  }
  return pairs;
}

// HIP-event timing of one Glimmer attention layer on production chunk
// shapes: fp32 vs shared vs WMMA canonical vs WMMA packed. Informational
// (like the KAT bench); the only abort is a WMMA slower than fp32, which
// would mean the matrix-core path is not engaged. The wire gate (>= 3x at
// (0,4096) and >= 10x at (8192,4056) vs fp32, parity vs shared) is evaluated
// from the printed BENCH lines.
void BenchTiming() {
  const std::pair<std::uint32_t, std::size_t> shapes[] = {
      {0, 1447}, {0, 4096}, {4096, 1777}, {4096, 4096}, {8192, 4056},
  };
  const std::size_t attention_size =
      static_cast<std::size_t>(kNumHeads) * kHeadDim;
  const std::size_t kv_size = static_cast<std::size_t>(kNumKvHeads) * kHeadDim;
  for (const std::uint32_t window : {0U, 2048U}) {
    for (const auto [start_pos, batch_size] : shapes) {
      const auto kMaxContext = std::max<std::uint32_t>(
          1024, start_pos + static_cast<std::uint32_t>(batch_size) + 64);
      const std::size_t total_kv =
          static_cast<std::size_t>(kNumKvHeads) * kMaxContext * kHeadDim;
      const std::size_t q_elements = batch_size * attention_size;
      Rng rng(0xBE71U ^ start_pos ^ (window * 31U));
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
      std::vector<__half> h_f16(total_kv * 2);
      for (std::uint32_t p = 0; p < kMaxContext; ++p) {
        for (std::uint32_t h = 0; h < kNumKvHeads; ++h) {
          for (std::uint32_t d = 0; d < kHeadDim; ++d) {
            const std::size_t f32 =
                ((static_cast<std::size_t>(h) * kMaxContext + p) * kHeadDim) +
                d;
            const std::size_t tm =
                ((static_cast<std::size_t>(p) * kNumKvHeads + h) * kHeadDim) +
                d;
            h_f16[tm] = __float2half(h_cache[f32]);
            h_f16[total_kv + tm] = __float2half(h_cache[total_kv + f32]);
          }
        }
      }
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
      HIP_CHECK(hipMemcpy(d_k, h_kv.data(),
                          batch_size * kv_size * sizeof(float),
                          hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(d_v, h_kv.data(),
                          batch_size * kv_size * sizeof(float),
                          hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(d_cache, h_cache.data(), total_kv * 2 * sizeof(float),
                          hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(d_cache_f16, h_f16.data(),
                          total_kv * 2 * sizeof(std::uint16_t),
                          hipMemcpyHostToDevice));
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
      hipEvent_t t0 = nullptr, t1 = nullptr;
      HIP_CHECK(hipEventCreate(&t0));
      HIP_CHECK(hipEventCreate(&t1));
      const auto time_it = [&](const char* what, int iters, auto&& fn) {
        for (int i = 0; i < 1; ++i)
          fn();  // warmup
        HIP_CHECK(hipEventRecord(t0));
        for (int i = 0; i < iters; ++i)
          fn();
        HIP_CHECK(hipEventRecord(t1));
        HIP_CHECK(hipEventSynchronize(t1));
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        std::cout << "  BENCH window=" << window << " start=" << start_pos
                  << " batch=" << batch_size << " " << what
                  << "_ms=" << (ms / iters) << "\n";
        return ms / iters;
      };
      const std::uint64_t pairs = CausalPairs(start_pos, batch_size, window);
      const int fp32_iters = pairs > 20000000ULL ? 1 : 3;
      const float fp32_ms = time_it("fp32", fp32_iters, [&] {
        const auto ref = gufo::hip::LaunchGlimmerPrefillAttention(
            d_q, d_gate, d_cache, v_cache, d_out, 0, start_pos,
            static_cast<std::uint32_t>(batch_size), window, kMaxContext,
            kNumHeads, kNumKvHeads, kHeadDim, q_elements, q_elements, total_kv,
            q_elements, nullptr);
        if (!ref.ok)
          std::abort();
      });
      const float shared_ms = time_it("shared", 5, [&] {
        const auto sh = gufo::hip::LaunchGlimmerSharedAttention(
            d_q, d_gate, d_cache_f16, cache_f16_v, d_out, 0, start_pos,
            static_cast<std::uint32_t>(batch_size), window, kMaxContext,
            kNumHeads, kNumKvHeads, kHeadDim, q_elements, q_elements, total_kv,
            q_elements, nullptr);
        if (!sh.ok)
          std::abort();
      });
      const float canon_ms = time_it("wmma_canonical", 10, [&] {
        const auto wmma = gufo::hip::LaunchGlimmerWmmaPrefillAttention(
            d_q, d_gate, d_cache_f16, cache_f16_v, d_out, 0, start_pos,
            static_cast<std::uint32_t>(batch_size), window, kMaxContext,
            kNumHeads, kNumKvHeads, kHeadDim, q_elements, q_elements, total_kv,
            q_elements, {}, {}, nullptr);
        if (!wmma.ok)
          std::abort();
      });
      float packed_ms = 0;
      if (batch_size >= 1024) {
        packed_ms = time_it("wmma_packed", 10, [&] {
          const auto wmma = gufo::hip::LaunchGlimmerWmmaPrefillAttention(
              d_q, d_gate, d_cache_f16, cache_f16_v, d_out, 0, start_pos,
              static_cast<std::uint32_t>(batch_size), window, kMaxContext,
              kNumHeads, kNumKvHeads, kHeadDim, q_elements, q_elements,
              total_kv, q_elements, full_k, full_v, nullptr);
          if (!wmma.ok)
            std::abort();
        });
      }
      const float best_ms =
          packed_ms > 0 ? std::min(canon_ms, packed_ms) : canon_ms;
      const double bytes = static_cast<double>(pairs) * 256.0;
      std::cout << "  BENCH window=" << window << " start=" << start_pos
                << " batch=" << batch_size
                << " speedup_fp32=" << (fp32_ms / best_ms)
                << " speedup_shared=" << (shared_ms / best_ms)
                << " gbps=" << (bytes / (best_ms * 1.0e-3) / 1.0e9) << "\n";
      if (best_ms >= fp32_ms) {
        std::cerr << "WMMA is not faster than fp32 at window=" << window
                  << " start=" << start_pos << " batch=" << batch_size << "\n";
        std::abort();
      }
      (void)canon_ms;
      HIP_CHECK(hipEventDestroy(t0));
      HIP_CHECK(hipEventDestroy(t1));
      for (float* p : {d_q, d_gate, d_k, d_v, d_cache, d_out, wk, wv}) {
        HIP_CHECK(hipFree(p));
      }
      HIP_CHECK(hipFree(d_cache_f16));
    }
  }
}

// The scale pair must multiply to 1/sqrt(head_dim): 2^-4*sqrt(2) at 128,
// exactly 1/16 at 256.
void CheckScalePair() {
  const double prod128 = 0.0625 * 1.4142135623730951;
  const double want128 = 1.0 / std::sqrt(128.0);
  const double rel128 = std::fabs(prod128 - want128) / want128;
  std::cout << "  scale128: product=" << prod128 << " want=" << want128
            << " rel=" << rel128 << "\n";
  if (rel128 > 1e-6) {
    std::cerr << "scale pair does not multiply to 1/sqrt(128)\n";
    std::abort();
  }
  if ((1.0 / 16.0) * 1.0 != 1.0 / std::sqrt(256.0)) {
    std::cerr << "scale pair changed the 1/16 path\n";
    std::abort();
  }
  std::cout << "  scale256: 1/16 * 1 exact\n";
}

// The dispatch contract: the WMMA launcher accepts (32,2,128) with window
// 0 and window > 0 plus (24,4,256) and (16,2,256), and rejects anything
// else, so unknown shapes keep falling back. The windowed instantiation
// rejects lse_out (no split path with a per-row window).
void CheckDispatch() {
  constexpr std::size_t kBatch = 64;
  constexpr std::uint32_t kCtx = 1024;
  constexpr std::size_t kMaxAttn27 = 24 * 256;
  constexpr std::size_t kMaxKv = 4 * 256;
  constexpr std::size_t kMaxTotalKv = 4 * kCtx * 256;
  float *d_q = nullptr, *d_gate = nullptr, *d_k = nullptr, *d_v = nullptr;
  float *d_cache = nullptr, *d_out = nullptr, *d_lse = nullptr;
  void* d_cache_f16 = nullptr;
  HIP_CHECK(hipMalloc(&d_q, kBatch * kMaxAttn27 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, kBatch * kMaxAttn27 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kBatch * kMaxKv * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kBatch * kMaxKv * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache, 2 * kMaxTotalKv * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_f16, 2 * kMaxTotalKv * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out, kBatch * kMaxAttn27 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_lse, kBatch * 32 * sizeof(float)));
  HIP_CHECK(hipMemset(d_q, 0, kBatch * kMaxAttn27 * sizeof(float)));
  HIP_CHECK(hipMemset(d_gate, 0, kBatch * kMaxAttn27 * sizeof(float)));
  HIP_CHECK(hipMemset(d_k, 0, kBatch * kMaxKv * sizeof(float)));
  HIP_CHECK(hipMemset(d_v, 0, kBatch * kMaxKv * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache, 0, 2 * kMaxTotalKv * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_f16, 0, 2 * kMaxTotalKv * sizeof(std::uint16_t)));
  const auto wmma = [&](std::uint32_t hq, std::uint32_t hkv, std::uint32_t hd,
                        std::uint32_t window, float* lse = nullptr) {
    return gufo::hip::LaunchQwenWmmaAttention(
        d_q, d_k, d_v, d_gate, d_cache, d_cache + kMaxTotalKv, d_cache_f16,
        static_cast<std::uint16_t*>(d_cache_f16) + kMaxTotalKv, d_out, 0, 0,
        kBatch, kCtx, hq, hkv, hd, nullptr, lse, 0, false, {}, {}, window);
  };
  if (!wmma(16, 2, 256, 0) || !wmma(24, 4, 256, 0) || !wmma(32, 2, 128, 0) ||
      !wmma(32, 2, 128, 2048)) {
    std::cerr << "WMMA launcher rejected a supported head config\n";
    std::abort();
  }
  if (wmma(32, 2, 256, 0) || wmma(16, 2, 128, 0) || wmma(32, 4, 128, 0)) {
    std::cerr << "WMMA launcher accepted an unsupported head config\n";
    std::abort();
  }
  if (wmma(32, 2, 128, 2048, d_lse)) {
    std::cerr << "windowed WMMA accepted lse_out\n";
    std::abort();
  }
  if (!wmma(32, 2, 128, 0, d_lse)) {
    std::cerr << "windowless Glimmer WMMA rejected lse_out\n";
    std::abort();
  }
  if (gufo::hip::LaunchBatchedAttentionTile(
          d_q, d_k, d_v, d_gate, d_cache, d_cache + kMaxTotalKv, d_cache_f16,
          static_cast<std::uint16_t*>(d_cache_f16) + kMaxTotalKv, d_out, 0, 0,
          kBatch, kCtx, 32, 2, 128)) {
    std::cerr << "tiled launcher accepted Glimmer; the shared gate widened\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());
  std::cout << "  dispatch: WMMA accepts (32,2,128)x2+(24,4)+(16,2), rejects "
               "the rest; tiled still rejects Glimmer\n";
  for (float* p : {d_q, d_gate, d_k, d_v, d_cache, d_out, d_lse}) {
    HIP_CHECK(hipFree(p));
  }
  HIP_CHECK(hipFree(d_cache_f16));
}

}  // namespace

constexpr gufo::hip::detail::AttentionSupportParams kGlimmerDispatchParams{
    .batch_size = 1024,
    .start_pos = 0,
    .max_context = 32768,
    .num_heads = 32,
    .num_kv_heads = 2,
    .head_dim = 128,
    .has_k_cache_f16 = true,
    .has_v_cache_f16 = true};
static_assert(
    !gufo::hip::detail::IsTiledAttentionSupported(kGlimmerDispatchParams),
    "the shared tiled gate must keep rejecting Glimmer");
static_assert(gufo::hip::detail::IsWmmaAttentionSupportedFor<32, 2, 128>(
                  kGlimmerDispatchParams),
              "the WMMA gate must accept Glimmer");
static_assert(!gufo::hip::detail::IsWmmaAttentionSupportedFor<32, 2>(
                  kGlimmerDispatchParams),
              "the default-256 WMMA gate must reject head_dim 128");

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Glimmer WMMA prefill attention ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }

  CheckDispatch();
  CheckScalePair();
  CheckPack();
  // Window 0 (the 13 full layers) at spans 1K/4K/8K at start 0 and depth.
  RunCase(0, 1024, 0, false, 0, true);
  RunCase(0, 1025, 0, false, 0, true);
  RunCase(0, 4096, 0, false, 0, true);
  RunCase(4096, 4096, 0, false, 0, true);
  // 16K production-shaped span, with the fp32 reference (it tiles at 48K).
  RunCase(12288, 4096, 0, false, 0, true);
  // Partial tiles, prefetch edges, unaligned depth.
  RunCase(0, 100, 0, false, 0, true);
  RunCase(0, 16, 0, false, 0, true);
  RunCase(0, 32, 0, false, 0, true);
  RunCase(1500, 128, 0, false, 0, true);
  RunCase(8192, 100, 0, false, 0, true);
  RunCase(8192, 1023, 0, false, 0, true);
  RunCase(8192, 1025, 0, false, 0, true);
  // LSE path on the windowless instantiation (no baseline: it has no
  // split-half output), split-half keys.
  RunCase(0, 2048, 0, true, 0, false);
  RunCase(8192, 1024, 0, true, 16, false);
  RunCase(8192, 1024, 0, false, 9216, false);
  // Window 2048 (the 39 SWA layers).
  RunCase(0, 1024, 2048, false, 0, true);
  RunCase(0, 4096, 2048, false, 0, true);
  RunCase(4096, 4096, 2048, false, 0, true);
  RunCase(8192, 4056, 2048, false, 0, true);
  RunCase(0, 100, 2048, false, 0, true);
  RunCase(1500, 128, 2048, false, 0, true);
  RunCase(8192, 1023, 2048, false, 0, true);
  RunCase(8192, 1025, 2048, false, 0, true);
  // key_begin with a window: aligned, unaligned, 32-aligned.
  RunCase(8192, 1024, 2048, false, 16, false);
  RunCase(8192, 1024, 2048, false, 17, false);
  RunCase(8192, 1024, 2048, false, 32, false);
  // Window edges: rows at W-1/W/W+1 for W in 64/128/2048/2050, a chunk
  // starting exactly at start_pos = W, and pos 0 (covered by (0,16)).
  RunCase(0, 128, 64, false, 0, true);
  RunCase(0, 256, 128, false, 0, true);
  RunCase(2000, 128, 2048, false, 0, true);
  RunCase(2000, 128, 2050, false, 0, true);
  RunCase(2048, 512, 2048, false, 0, true);
  // Layer 1 (pins the layer stride in both index orders).
  RunCase(1024, 512, 2048, false, 0, true, 1);

  BenchTiming();

  std::cout << "Glimmer WMMA prefill attention ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Glimmer WMMA prefill attention ops "
               "test.\n";
  return 77;
#endif
}
