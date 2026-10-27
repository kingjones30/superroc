// GLIMMER S1 small-shape GPU test. No GGUF and no full-model launch.
// Exit 0 pass, 1 fail.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

#include "src/models/glimmer/glimmer_oracle.hpp"
#include "src/models/glimmer/hip/glimmer_decode.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"

#if !defined(ENGINE_ENABLE_HIP)
int main() {
  std::cerr << "FAIL glimmer_s1_test built without HIP\n";
  return 1;
}
#else
#include <hip/hip_runtime.h>

namespace {

int Fail(const char* msg) {
  std::cerr << "FAIL " << msg << '\n';
  return 1;
}

bool HipOk(hipError_t err, const char* what) {
  if (err == hipSuccess) {
    return true;
  }
  std::cerr << "FAIL " << what << " hip " << static_cast<int>(err) << '\n';
  return false;
}

float MaxAbs(const std::vector<float>& got, const std::vector<float>& ref) {
  float gap = 0.0F;
  const std::size_t n = got.size() < ref.size() ? got.size() : ref.size();
  for (std::size_t i = 0; i < n; ++i) {
    gap = std::max(gap, std::fabs(got[i] - ref[i]));
  }
  return gap;
}

std::vector<float> CpuAttention(
    const std::vector<float>& q, const std::vector<float>& gate,
    const std::vector<float>& k_cache, const std::vector<float>& v_cache,
    std::uint32_t pos, std::uint32_t window, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    bool apply_window) {
  std::vector<float> out(static_cast<std::size_t>(num_heads) * head_dim, 0.0F);
  const std::uint32_t begin =
      apply_window ? superroc::models::GlimmerSwaBegin(pos, window) : 0U;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  const std::uint32_t group = num_heads / num_kv_heads;
  for (std::uint32_t h = 0; h < num_heads; ++h) {
    const std::uint32_t kv_h = h / group;
    std::vector<double> scores(pos - begin + 1U);
    double max_score = -1.0e30;
    for (std::uint32_t p = begin; p <= pos; ++p) {
      double dot = 0.0;
      const std::size_t q_base = static_cast<std::size_t>(h) * head_dim;
      const std::size_t k_base =
          ((static_cast<std::size_t>(kv_h) * max_context + p) * head_dim);
      for (std::uint32_t d = 0; d < head_dim; ++d) {
        dot += static_cast<double>(q[q_base + d]) *
               static_cast<double>(k_cache[k_base + d]);
      }
      const double score = dot * scale;
      scores[p - begin] = score;
      max_score = std::max(max_score, score);
    }
    double sum = 0.0;
    for (double& score : scores) {
      score = std::exp(score - max_score);
      sum += score;
    }
    const double inv = sum == 0.0 ? 0.0 : 1.0 / sum;
    for (std::uint32_t d = 0; d < head_dim; ++d) {
      double ctx = 0.0;
      for (std::uint32_t p = begin; p <= pos; ++p) {
        const std::size_t v_base =
            ((static_cast<std::size_t>(kv_h) * max_context + p) * head_dim);
        ctx +=
            scores[p - begin] * inv * static_cast<double>(v_cache[v_base + d]);
      }
      const double g = gate[static_cast<std::size_t>(h) * head_dim + d];
      ctx *= 1.0 / (1.0 + std::exp(-g));
      out[static_cast<std::size_t>(h) * head_dim + d] = static_cast<float>(ctx);
    }
  }
  return out;
}

int CheckSpan(const char* label, std::uint32_t ctx, std::uint32_t pos,
              std::uint32_t window) {
  constexpr std::uint32_t kHeads = 4;
  constexpr std::uint32_t kKv = 2;
  constexpr std::uint32_t kDim = 128;
  const std::size_t q_elems = static_cast<std::size_t>(kHeads) * kDim;
  const std::size_t cache_elems = static_cast<std::size_t>(kKv) * ctx * kDim;
  std::vector<float> q(q_elems), gate(q_elems), k(cache_elems), v(cache_elems);
  for (std::size_t i = 0; i < q.size(); ++i) {
    // Strictly positive so a position-0 key spike of 40 dominates full
    // attention and a window that excludes it must disagree.
    q[i] = 0.05F;
    gate[i] = (i % 2 == 0) ? 0.4F : -0.2F;
  }
  for (std::uint32_t kv_h = 0; kv_h < kKv; ++kv_h) {
    for (std::uint32_t p = 0; p < ctx; ++p) {
      for (std::uint32_t d = 0; d < kDim; ++d) {
        const std::size_t idx =
            (static_cast<std::size_t>(kv_h) * ctx + p) * kDim + d;
        const float spike = (p == 0) ? 40.0F : 0.0F;
        k[idx] = spike + 0.01F * static_cast<float>((p + d) % 11);
        // Position 0 carries a large value so dropping it (the SWA window)
        // moves the output by far more than the numeric tolerance.
        v[idx] = (p == 0) ? 5.0F : 0.02F * static_cast<float>((d + 3 * p) % 9);
      }
    }
  }
  const auto window_ref =
      CpuAttention(q, gate, k, v, pos, window, ctx, kHeads, kKv, kDim, true);
  const auto full_ref =
      CpuAttention(q, gate, k, v, pos, window, ctx, kHeads, kKv, kDim, false);
  float *d_q = nullptr, *d_gate = nullptr, *d_k = nullptr, *d_v = nullptr,
        *d_out = nullptr;
  if (!HipOk(hipMalloc(&d_q, q.size() * sizeof(float)), "span q") ||
      !HipOk(hipMalloc(&d_gate, gate.size() * sizeof(float)), "span gate") ||
      !HipOk(hipMalloc(&d_k, k.size() * sizeof(float)), "span k") ||
      !HipOk(hipMalloc(&d_v, v.size() * sizeof(float)), "span v") ||
      !HipOk(hipMalloc(&d_out, q.size() * sizeof(float)), "span out") ||
      !HipOk(hipMemcpy(d_q, q.data(), q.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "span copy q") ||
      !HipOk(hipMemcpy(d_gate, gate.data(), gate.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "span copy gate") ||
      !HipOk(hipMemcpy(d_k, k.data(), k.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "span copy k") ||
      !HipOk(hipMemcpy(d_v, v.data(), v.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "span copy v")) {
    return 1;
  }
  const auto attn = superroc::hip::LaunchGlimmerWindowAttention(
      d_q, d_gate, d_k, d_v, d_out, 0, pos, window, ctx, kHeads, kKv, kDim,
      q_elems, q_elems, cache_elems, q_elems, nullptr);
  std::vector<float> got(q.size());
  if (!attn.ok || !HipOk(hipDeviceSynchronize(), "span sync") ||
      !HipOk(hipMemcpy(got.data(), d_out, got.size() * sizeof(float),
                       hipMemcpyDeviceToHost),
             "span back")) {
    std::cerr << label << " " << (attn.error != nullptr ? attn.error : "copy")
              << '\n';
    hipFree(d_q);
    hipFree(d_gate);
    hipFree(d_k);
    hipFree(d_v);
    hipFree(d_out);
    return 1;
  }
  hipFree(d_q);
  hipFree(d_gate);
  hipFree(d_k);
  hipFree(d_v);
  hipFree(d_out);
  const float gap = MaxAbs(got, window_ref);
  const float full_gap = MaxAbs(got, full_ref);
  std::cout << label << "_max_abs=" << gap << " full_gap=" << full_gap << '\n';
  if (!(gap < 1.0e-2F)) {
    return Fail(label);
  }
  if (window != 0 && pos + 1 > window && !(full_gap > 0.1F)) {
    return Fail("long window did not differ from full attention");
  }
  return 0;
}

int CheckPrefillBatch() {
  constexpr std::uint32_t kHeads = 4;
  constexpr std::uint32_t kKv = 2;
  constexpr std::uint32_t kDim = 128;
  constexpr std::uint32_t kCtx = 16;
  constexpr std::uint32_t kStart = 4;
  constexpr std::uint32_t kBatch = 3;
  constexpr std::uint32_t kWindow = 4;
  const std::size_t row = static_cast<std::size_t>(kHeads) * kDim;
  const std::size_t q_elems = row * kBatch;
  const std::size_t cache_elems = static_cast<std::size_t>(kKv) * kCtx * kDim;
  std::vector<float> q(q_elems), gate(q_elems), k(cache_elems), v(cache_elems);
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = 0.03F * static_cast<float>(static_cast<int>(i % 5) - 2);
    gate[i] = (i % 3 == 0) ? 0.2F : -0.1F;
  }
  for (std::size_t i = 0; i < k.size(); ++i) {
    k[i] = 0.01F * static_cast<float>(i % 13);
    v[i] = 0.02F * static_cast<float>((i + 3) % 9);
  }
  float *d_q = nullptr, *d_gate = nullptr, *d_k = nullptr, *d_v = nullptr,
        *d_out = nullptr;
  if (!HipOk(hipMalloc(&d_q, q.size() * sizeof(float)), "batch q") ||
      !HipOk(hipMalloc(&d_gate, gate.size() * sizeof(float)), "batch gate") ||
      !HipOk(hipMalloc(&d_k, k.size() * sizeof(float)), "batch k") ||
      !HipOk(hipMalloc(&d_v, v.size() * sizeof(float)), "batch v") ||
      !HipOk(hipMalloc(&d_out, q.size() * sizeof(float)), "batch out") ||
      !HipOk(hipMemcpy(d_q, q.data(), q.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "batch copy q") ||
      !HipOk(hipMemcpy(d_gate, gate.data(), gate.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "batch copy gate") ||
      !HipOk(hipMemcpy(d_k, k.data(), k.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "batch copy k") ||
      !HipOk(hipMemcpy(d_v, v.data(), v.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "batch copy v")) {
    return 1;
  }
  const auto attn = superroc::hip::LaunchGlimmerPrefillAttention(
      d_q, d_gate, d_k, d_v, d_out, 0, kStart, kBatch, kWindow, kCtx, kHeads,
      kKv, kDim, q_elems, q_elems, cache_elems, q_elems, nullptr);
  std::vector<float> got(q.size());
  if (!attn.ok || !HipOk(hipDeviceSynchronize(), "batch sync") ||
      !HipOk(hipMemcpy(got.data(), d_out, got.size() * sizeof(float),
                       hipMemcpyDeviceToHost),
             "batch back")) {
    std::cerr << (attn.error != nullptr ? attn.error : "prefill attn") << '\n';
    return 1;
  }
  hipFree(d_q);
  hipFree(d_gate);
  hipFree(d_k);
  hipFree(d_v);
  hipFree(d_out);
  float gap = 0.0F;
  for (std::uint32_t row_i = 0; row_i < kBatch; ++row_i) {
    std::vector<float> q_row(q.begin() + row_i * row,
                             q.begin() + (row_i + 1) * row);
    std::vector<float> g_row(gate.begin() + row_i * row,
                             gate.begin() + (row_i + 1) * row);
    const auto ref = CpuAttention(q_row, g_row, k, v, kStart + row_i, kWindow,
                                  kCtx, kHeads, kKv, kDim, true);
    for (std::size_t i = 0; i < ref.size(); ++i) {
      gap = std::max(gap, std::fabs(got[row_i * row + i] - ref[i]));
    }
  }
  std::cout << "S1 prefill_batch_max_abs=" << gap << '\n';
  if (!(gap < 1.0e-2F)) {
    return Fail("prefill batch attention");
  }
  return 0;
}

}  // namespace

int main() {
  using superroc::hip::detail::AttentionSupportParams;
  using superroc::hip::detail::IsSplitKDecodeAttentionSupported;
  using superroc::hip::detail::IsTiledAttentionSupported;
  // Sequence length is past the split-k minimum so the head_dim gate is
  // the reason head_dim 128 is refused. A short sequence would hide that.
  AttentionSupportParams params;
  params.num_heads = 32;
  params.num_kv_heads = 2;
  params.head_dim = 128;
  params.batch_size = 1;
  params.start_pos = 0;
  params.max_context = 4096;
  params.has_k_cache_f16 = true;
  params.has_v_cache_f16 = true;
  if (IsTiledAttentionSupported(params)) {
    return Fail("tiled 256-head policy selected muse geometry");
  }
  if (IsSplitKDecodeAttentionSupported(2048, 32, 2, 128)) {
    return Fail("split-k policy selected head_dim 128");
  }
  if (!IsSplitKDecodeAttentionSupported(2048, 32, 2, 256)) {
    return Fail("split-k policy rejected head_dim 256");
  }

  constexpr std::uint32_t kHeads = 2;
  constexpr std::uint32_t kDim = 8;
  std::vector<float> rope(kHeads * kDim);
  for (std::size_t i = 0; i < rope.size(); ++i) {
    rope[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.25F;
  }
  std::vector<float> rope_ref = rope;
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    superroc::models::GlimmerNormRope(
        std::span<float>(rope_ref.data() + static_cast<std::size_t>(h) * kDim,
                         kDim),
        kDim, 7, 500000.0F);
  }
  float* d_rope = nullptr;
  if (!HipOk(hipMalloc(&d_rope, rope.size() * sizeof(float)), "malloc rope") ||
      !HipOk(hipMemcpy(d_rope, rope.data(), rope.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy rope")) {
    return 1;
  }
  const auto rope_status = superroc::hip::LaunchGlimmerNormRope(
      d_rope, kHeads, kDim, kDim, 7, 500000.0F, nullptr);
  if (!rope_status.ok) {
    std::cerr << rope_status.error << '\n';
    hipFree(d_rope);
    return 1;
  }
  if (!HipOk(hipDeviceSynchronize(), "sync rope") ||
      !HipOk(hipMemcpy(rope.data(), d_rope, rope.size() * sizeof(float),
                       hipMemcpyDeviceToHost),
             "back rope")) {
    hipFree(d_rope);
    return 1;
  }
  hipFree(d_rope);
  const float rope_gap = MaxAbs(rope, rope_ref);
  std::cout << "S1 rope_max_abs=" << rope_gap << '\n';
  if (!(rope_gap < 1.0e-3F)) {
    return Fail("norm rope");
  }

  // Layer 1, two KV heads, so the fp32 index is not the layer-0 packing.
  constexpr std::uint32_t kWriteLayer = 1;
  constexpr std::uint32_t kWriteKv = 2;
  constexpr std::uint32_t kWriteDim = 8;
  constexpr std::uint32_t kWritePos = 2;
  constexpr std::uint32_t kWriteCtx = 4;
  const std::size_t write_src = static_cast<std::size_t>(kWriteKv) * kWriteDim;
  const std::size_t write_cache = static_cast<std::size_t>(kWriteLayer + 1U) *
                                  kWriteKv * kWriteCtx * kWriteDim;
  std::vector<float> host_k(write_src), host_v(write_src);
  for (std::size_t i = 0; i < write_src; ++i) {
    host_k[i] = 0.5F + static_cast<float>(i);
    host_v[i] = -1.5F - static_cast<float>(i);
  }
  float *d_wk = nullptr, *d_wv = nullptr, *d_wkc = nullptr, *d_wvc = nullptr;
  if (!HipOk(hipMalloc(&d_wk, write_src * sizeof(float)), "malloc wk") ||
      !HipOk(hipMalloc(&d_wv, write_src * sizeof(float)), "malloc wv") ||
      !HipOk(hipMalloc(&d_wkc, write_cache * sizeof(float)), "malloc wkc") ||
      !HipOk(hipMalloc(&d_wvc, write_cache * sizeof(float)), "malloc wvc") ||
      !HipOk(hipMemcpy(d_wk, host_k.data(), write_src * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy wk") ||
      !HipOk(hipMemcpy(d_wv, host_v.data(), write_src * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy wv") ||
      !HipOk(hipMemset(d_wkc, 0, write_cache * sizeof(float)), "clear wkc") ||
      !HipOk(hipMemset(d_wvc, 0, write_cache * sizeof(float)), "clear wvc")) {
    return 1;
  }
  const auto wrote = superroc::hip::LaunchGlimmerWriteKv(
      d_wk, d_wv, d_wkc, d_wvc, kWriteLayer, kWritePos, kWriteCtx, kWriteKv,
      kWriteDim, write_src, write_cache, nullptr);
  std::vector<float> back_k(write_cache), back_v(write_cache);
  if (!wrote.ok || !HipOk(hipDeviceSynchronize(), "sync kv") ||
      !HipOk(hipMemcpy(back_k.data(), d_wkc, write_cache * sizeof(float),
                       hipMemcpyDeviceToHost),
             "back k") ||
      !HipOk(hipMemcpy(back_v.data(), d_wvc, write_cache * sizeof(float),
                       hipMemcpyDeviceToHost),
             "back v")) {
    std::cerr << (wrote.error != nullptr ? wrote.error : "kv write") << '\n';
    return 1;
  }
  float kv_gap = 0.0F;
  for (std::uint32_t kv_h = 0; kv_h < kWriteKv; ++kv_h) {
    const std::size_t dst =
        (((static_cast<std::size_t>(kWriteLayer) * kWriteKv + kv_h) *
              kWriteCtx +
          kWritePos) *
         kWriteDim);
    for (std::uint32_t d = 0; d < kWriteDim; ++d) {
      const std::size_t src = static_cast<std::size_t>(kv_h) * kWriteDim + d;
      kv_gap = std::max(kv_gap, std::fabs(back_k[dst + d] - host_k[src]));
      kv_gap = std::max(kv_gap, std::fabs(back_v[dst + d] - host_v[src]));
    }
  }
  std::cout << "S1 kv_write_max_abs=" << kv_gap << '\n';
  if (!(kv_gap < 1.0e-6F)) {
    return Fail("fp32 kv write");
  }
  const auto kv_oob = superroc::hip::LaunchGlimmerWriteKv(
      d_wk, d_wv, d_wkc, d_wvc, kWriteLayer, kWriteCtx, kWriteCtx, kWriteKv,
      kWriteDim, write_src, write_cache, nullptr);
  hipFree(d_wk);
  hipFree(d_wv);
  hipFree(d_wkc);
  hipFree(d_wvc);
  if (kv_oob.ok) {
    return Fail("kv write launched an out of range position");
  }
  std::cout << "S1 kv_oob_refused=" << kv_oob.error << '\n';

  constexpr std::uint32_t kAttnHeads = 4;
  constexpr std::uint32_t kKv = 2;
  constexpr std::uint32_t kHead = 128;
  constexpr std::uint32_t kPos = 7;
  constexpr std::uint32_t kWindow = 4;
  constexpr std::uint32_t kCtx = 8;
  const std::size_t q_elems = static_cast<std::size_t>(kAttnHeads) * kHead;
  const std::size_t cache_elems = static_cast<std::size_t>(kKv) * kCtx * kHead;
  std::vector<float> q(q_elems), gate(q_elems), k(cache_elems), v(cache_elems);
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = 0.01F * static_cast<float>(static_cast<int>(i % 9) - 4);
    gate[i] = (i % 2 == 0) ? 0.5F : -0.25F;
  }
  for (std::uint32_t kv_h = 0; kv_h < kKv; ++kv_h) {
    for (std::uint32_t p = 0; p < kCtx; ++p) {
      for (std::uint32_t d = 0; d < kHead; ++d) {
        const std::size_t idx =
            (static_cast<std::size_t>(kv_h) * kCtx + p) * kHead + d;
        // Position 0 is huge so a missing window dominates the output.
        const float spike = p == 0 ? 50.0F : 0.0F;
        k[idx] = spike + 0.02F * static_cast<float>(p + d % 3);
        v[idx] = spike * 0.1F + 0.03F * static_cast<float>((d + p) % 5);
      }
    }
  }
  const auto window_ref = CpuAttention(q, gate, k, v, kPos, kWindow, kCtx,
                                       kAttnHeads, kKv, kHead, true);
  const auto full_ref = CpuAttention(q, gate, k, v, kPos, kWindow, kCtx,
                                     kAttnHeads, kKv, kHead, false);
  float *d_q = nullptr, *d_gate = nullptr, *d_k = nullptr, *d_v = nullptr,
        *d_out = nullptr;
  if (!HipOk(hipMalloc(&d_q, q.size() * sizeof(float)), "malloc q") ||
      !HipOk(hipMalloc(&d_gate, gate.size() * sizeof(float)), "malloc gate") ||
      !HipOk(hipMalloc(&d_k, k.size() * sizeof(float)), "malloc k") ||
      !HipOk(hipMalloc(&d_v, v.size() * sizeof(float)), "malloc v") ||
      !HipOk(hipMalloc(&d_out, q.size() * sizeof(float)), "malloc out") ||
      !HipOk(hipMemcpy(d_q, q.data(), q.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy q") ||
      !HipOk(hipMemcpy(d_gate, gate.data(), gate.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy gate") ||
      !HipOk(hipMemcpy(d_k, k.data(), k.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy k") ||
      !HipOk(hipMemcpy(d_v, v.data(), v.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "copy v")) {
    return 1;
  }
  const auto attn = superroc::hip::LaunchGlimmerWindowAttention(
      d_q, d_gate, d_k, d_v, d_out, 0, kPos, kWindow, kCtx, kAttnHeads, kKv,
      kHead, q_elems, q_elems, cache_elems, q_elems, nullptr);
  if (!attn.ok) {
    std::cerr << attn.error << '\n';
    return 1;
  }
  std::vector<float> got(q.size());
  if (!HipOk(hipDeviceSynchronize(), "sync attn") ||
      !HipOk(hipMemcpy(got.data(), d_out, got.size() * sizeof(float),
                       hipMemcpyDeviceToHost),
             "back attn")) {
    return 1;
  }
  const float attn_gap = MaxAbs(got, window_ref);
  const float full_gap = MaxAbs(got, full_ref);
  std::cout << "S1 attn_window_max_abs=" << attn_gap
            << " attn_full_gap=" << full_gap << '\n';
  if (!(attn_gap < 1.0e-2F) || !(full_gap > 0.1F)) {
    return Fail("windowed attention");
  }

  const auto oob = superroc::hip::LaunchGlimmerWindowAttention(
      d_q, d_gate, d_k, d_v, d_out, 0, kCtx, kWindow, kCtx, kAttnHeads, kKv,
      kHead, q_elems, q_elems, cache_elems, q_elems, nullptr);
  if (oob.ok) {
    return Fail("out of range position was launched");
  }
  std::cout << "S1 oob_refused=" << oob.error << '\n';

  const std::uint32_t tile = superroc::hip::GlimmerScoreTileElems();
  std::cout << "S1 score_tile_elems=" << tile
            << " score_tile_bytes=" << (tile * sizeof(float)) << '\n';
  if (tile * sizeof(float) != 48U * 1024U) {
    return Fail("score tile is not 48 KiB");
  }
  hipFree(d_q);
  hipFree(d_gate);
  hipFree(d_k);
  hipFree(d_v);
  hipFree(d_out);
  d_q = d_gate = d_k = d_v = d_out = nullptr;
  if (CheckPrefillBatch() != 0) {
    return 1;
  }
  // 8192 is one tile. 13000 and 32768 cross the 48 KiB tile. SWA stays 2048.
  if (CheckSpan("S1 span_8192", 8192, 8191, 0) != 0 ||
      CheckSpan("S1 span_13000", 13000, 12999, 0) != 0 ||
      CheckSpan("S1 span_32768", 32768, 32767, 0) != 0 ||
      CheckSpan("S1 span_swa_32768", 32768, 32767, 2048) != 0) {
    return 1;
  }

  float logits[4] = {40.0F, -40.0F, 1.0F, 0.0F};
  float expect[4] = {40.0F, -40.0F, 1.0F, 0.0F};
  superroc::models::GlimmerScaleSoftcap(expect, 0.1961161345243454F, 20.0F);
  float* d_logits = nullptr;
  if (!HipOk(hipMalloc(&d_logits, sizeof(logits)), "malloc logits") ||
      !HipOk(hipMemcpy(d_logits, logits, sizeof(logits), hipMemcpyHostToDevice),
             "copy logits")) {
    return 1;
  }
  const auto cap = superroc::hip::LaunchGlimmerScaleSoftcap(
      d_logits, 4, 0.1961161345243454F, 20.0F, nullptr);
  if (!cap.ok || !HipOk(hipDeviceSynchronize(), "sync cap") ||
      !HipOk(hipMemcpy(logits, d_logits, sizeof(logits), hipMemcpyDeviceToHost),
             "back logits")) {
    std::cerr << (cap.error != nullptr ? cap.error : "softcap") << '\n';
    return 1;
  }
  hipFree(d_logits);
  hipFree(d_q);
  hipFree(d_gate);
  hipFree(d_k);
  hipFree(d_v);
  hipFree(d_out);
  float cap_gap = 0.0F;
  for (int i = 0; i < 4; ++i) {
    cap_gap = std::max(cap_gap, std::fabs(logits[i] - expect[i]));
  }
  std::cout << "S1 softcap_max_abs=" << cap_gap << '\n';
  if (!(cap_gap < 1.0e-4F)) {
    return Fail("softcap");
  }
  std::cout << "S1 small PASS\n";
  return 0;
}
#endif
