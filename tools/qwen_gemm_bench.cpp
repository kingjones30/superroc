// tools/qwen_gemm_bench.cpp — SQ1 SPEC1 item 2 (the kill criterion) +
// SQ1 SPEC2 step 1 (the missing rows; corrected SPEC2 21:08 CDT).
//
// Times, per 27B dense (m, k) and T in {512, 1024, 2048, 4096}:
//   * stock:  F32->F16 narrow + LaunchBatchedQuantGEMMFp16 over Q4_K weights
//             (the F16 route over Q4_K; the stock file IS on this route at
//             batch >= 1024 — num_layers = 65 - 1 draft = 64),
//   * today:  Q8_1 quant + LaunchBatchedQuantGEMMPreQuantized over type-101
//             weights (today's ROC route on our engine),
//   * real:   Q8_1 quant (shared with today) + PreQuantized over the weight
//             types the stock UD file REALLY uses per role (SPEC2 step 1):
//             Q4_K (= the requested int8-Q4K row), Q5_K, Q6_K, IQ4_XS and
//             Q8_0 — the censused types,
//   * f16:    F32->F16 narrow (shared with stock) + LaunchBatchedQuantGEMMFp16
//             over the stock file's real per-role types (SPEC2 step 1, the
//             TRUE stock route per tensor; Q4_K is the stock row above),
//             plus f16-100/f16-101 (SPEC2 bench3: the FP4-in-F16 fix under
//             test) and dual-Q4K on the gate/up cell (production stock takes
//             the dual kernel there; bar-soundness check),
//   * ported: F32->F16 narrow + the ported Fp4DenseF16Gemm over type-101 and
//             type-100 weights.
// Random weights/activations (timing only), HIP-event timing, median of N
// reps, finite-output smoke check per route. SPEC2 bar: an FP4 total <=
// 1/0.9 of the stock-F16 total (speed) on every cell, both FP4 types, every
// T the serve path uses (chunks of <= 4096: prefill.cpp ForwardPromptBatch /
// arena.cpp kMaxPromptBatch).
//
// Usage: qwen_gemm_bench [T=512,1024,2048,4096] [reps=20] [--csv]
// Run inside gpu-run. No model load; peak device use is ~2 GB.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/rocm_fp4/fp4_f16_prefill.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"

namespace hipq = superroc::hip;
namespace f16 = superroc::models::qwen38_flash_next::rocm;

#define HIP_OK(call)                                                          \
  do {                                                                        \
    const hipError_t e_hip_ok_ = (call);                                      \
    if (e_hip_ok_ != hipSuccess) {                                            \
      fprintf(stderr, "HIP %s:%d: %s failed: %s\n", __FILE__, __LINE__,        \
              #call, hipGetErrorString(e_hip_ok_));                           \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

static float Median(std::vector<float> v) {
  auto mid = v.begin() + v.size() / 2;
  std::nth_element(v.begin(), mid, v.end());
  float m = *mid;
  if ((v.size() % 2) == 0) {
    auto mid2 = std::max_element(v.begin(), mid);
    m = 0.5f * (m + *mid2);
  }
  return m;
}

// Median ms of `fn` over `reps` reps (3 warmup, device-synced each rep).
static float TimeIt(hipStream_t stream, const std::function<void()>& fn,
                    int reps) {
  hipEvent_t a, b;
  HIP_OK(hipEventCreate(&a));
  HIP_OK(hipEventCreate(&b));
  for (int i = 0; i < 3; ++i) fn();
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> ms;
  ms.reserve(reps);
  for (int i = 0; i < reps; ++i) {
    HIP_OK(hipEventRecord(a, stream));
    fn();
    HIP_OK(hipEventRecord(b, stream));
    HIP_OK(hipDeviceSynchronize());
    float e = 0;
    HIP_OK(hipEventElapsedTime(&e, a, b));
    ms.push_back(e);
  }
  HIP_OK(hipEventDestroy(a));
  HIP_OK(hipEventDestroy(b));
  return Median(ms);
}

static bool FiniteProbe(const float* d, std::size_t n) {
  const std::size_t m = std::min<std::size_t>(n, 64);
  std::vector<float> h(m);
  HIP_OK(hipMemcpy(h.data(), d, m * sizeof(float), hipMemcpyDeviceToHost));
  for (float v : h)
    if (!std::isfinite(v)) return false;
  return true;
}

static void* DevAlloc(std::size_t bytes) {
  void* p = nullptr;
  HIP_OK(hipMalloc(&p, bytes ? bytes : 1));
  return p;
}

static void ToDevice(void* d, const void* h, std::size_t bytes) {
  HIP_OK(hipMemcpy(d, h, bytes, hipMemcpyHostToDevice));
}

static void FillAct(std::vector<float>& v, float amp, std::mt19937& rng) {
  std::uniform_real_distribution<float> d(-amp, amp);
  for (float& x : v) x = d(rng);
}

static void FillQ4K(std::vector<std::uint8_t>& buf, std::size_t rows,
                    std::size_t k, std::mt19937& rng) {
  const std::size_t nb = k / 256;
  buf.resize(rows * nb * 144);
  std::uniform_int_distribution<int> b(0, 255);
  std::uniform_int_distribution<int> s(0, 4);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t n = 0; n < nb; ++n) {
      auto* blk = reinterpret_cast<superroc::quant::block_q4_K*>(
          &buf[(r * nb + n) * 144]);
      blk->d = 0x3400;  // ~0.002 in F16 bits (timing only)
      blk->dmin = 0x0000;
      for (int j = 0; j < 12; ++j) blk->scales[j] = std::uint8_t(s(rng));
      for (int j = 0; j < 128; ++j) blk->qs[j] = std::uint8_t(b(rng));
    }
}

// ROCmFP4: 16 nibble bytes + scale byte(s); 0x1C ~= 0.047. fast=false writes
// the second scale byte (18-byte blocks).
static void FillFP4(std::vector<std::uint8_t>& buf, std::size_t rows,
                    std::size_t k, bool fast, std::mt19937& rng) {
  const std::size_t nb = k / 32;
  const std::size_t bb = fast ? 17 : 18;
  buf.resize(rows * nb * bb);
  std::uniform_int_distribution<int> b(0, 255);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t n = 0; n < nb; ++n) {
      auto* blk = &buf[(r * nb + n) * bb];
      for (std::size_t j = 0; j < 16; ++j) blk[j] = std::uint8_t(b(rng));
      blk[16] = 0x1C;
      if (!fast) blk[17] = 0x1C;
    }
}

// SPEC2 step 1 fillers: timing-only random blocks for the stock file's real
// per-role types. Block layouts from src/core/quant/ggml_dequant.hpp;
// sizes via sizeof (never hardcoded).
static void FillQ5K(std::vector<std::uint8_t>& buf, std::size_t rows,
                    std::size_t k, std::mt19937& rng) {
  using superroc::quant::block_q5_K;
  const std::size_t nb = k / 256;
  buf.resize(rows * nb * sizeof(block_q5_K));
  std::uniform_int_distribution<int> b(0, 255);
  std::uniform_int_distribution<int> s(0, 4);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t n = 0; n < nb; ++n) {
      auto* blk = reinterpret_cast<block_q5_K*>(
          &buf[(r * nb + n) * sizeof(block_q5_K)]);
      blk->d = 0x3400;
      blk->dmin = 0x0000;
      for (int j = 0; j < 12; ++j) blk->scales[j] = std::uint8_t(s(rng));
      for (int j = 0; j < 32; ++j) blk->qh[j] = std::uint8_t(b(rng));
      for (int j = 0; j < 128; ++j) blk->qs[j] = std::uint8_t(b(rng));
    }
}

static void FillQ6K(std::vector<std::uint8_t>& buf, std::size_t rows,
                    std::size_t k, std::mt19937& rng) {
  using superroc::quant::block_q6_K;
  const std::size_t nb = k / 256;
  buf.resize(rows * nb * sizeof(block_q6_K));
  std::uniform_int_distribution<int> b(0, 255);
  std::uniform_int_distribution<int> s(-4, 4);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t n = 0; n < nb; ++n) {
      auto* blk = reinterpret_cast<block_q6_K*>(
          &buf[(r * nb + n) * sizeof(block_q6_K)]);
      for (int j = 0; j < 128; ++j) blk->ql[j] = std::uint8_t(b(rng));
      for (int j = 0; j < 64; ++j) blk->qh[j] = std::uint8_t(b(rng));
      for (int j = 0; j < 16; ++j)
        blk->scales[j] = std::int8_t(s(rng));
      blk->d = 0x3400;
    }
}

static void FillIQ4XS(std::vector<std::uint8_t>& buf, std::size_t rows,
                      std::size_t k, std::mt19937& rng) {
  using superroc::quant::block_iq4_xs;
  const std::size_t nb = k / 256;
  buf.resize(rows * nb * sizeof(block_iq4_xs));
  std::uniform_int_distribution<int> b(0, 255);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t n = 0; n < nb; ++n) {
      auto* blk = reinterpret_cast<block_iq4_xs*>(
          &buf[(r * nb + n) * sizeof(block_iq4_xs)]);
      blk->d = 0x3400;
      blk->scales_h = std::uint16_t(b(rng) | (b(rng) << 8));
      for (int j = 0; j < 4; ++j) blk->scales_l[j] = std::uint8_t(b(rng));
      for (int j = 0; j < 128; ++j) blk->qs[j] = std::uint8_t(b(rng));
    }
}

static void FillQ80(std::vector<std::uint8_t>& buf, std::size_t rows,
                    std::size_t k, std::mt19937& rng) {
  using superroc::quant::block_q8_0;
  const std::size_t nb = k / 32;
  buf.resize(rows * nb * sizeof(block_q8_0));
  std::uniform_int_distribution<int> b(-128, 127);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t n = 0; n < nb; ++n) {
      auto* blk = reinterpret_cast<block_q8_0*>(
          &buf[(r * nb + n) * sizeof(block_q8_0)]);
      blk->d = 0x3400;
      for (int j = 0; j < 32; ++j) blk->qs[j] = std::int8_t(b(rng));
    }
}

// Stock UD-Q4_K_XL per-role census (MAX1, /root/ls/kit/gguf_census.py,
// 2026-10-10; block_count=65). Printed in the header so the real-stock rows
// below are traceable to the file.
static void PrintCensus() {
  printf("# census Qwen3.8-27B-UD-Q4_K_XL.gguf (block_count=65):\n");
  printf("#   ffn_gate [5120,17408]: IQ4_XS x25,Q5_K x19,Q4_K x13,Q6_K x5\n");
  printf("#   ffn_up   [5120,17408]: Q5_K x32,IQ4_XS x21,Q4_K x6,Q6_K x3\n");
  printf("#   ffn_down [17408,5120]: Q5_K x35,IQ4_XS x16,Q6_K x6,Q4_K x5\n");
  printf("#   attn_gate[5120,6144]:  Q5_K x31,Q4_K x13,IQ4_XS x3,Q6_K x1\n");
  printf("#   ssm_out  [6144,5120]:  Q5_K x33,Q6_K x13,Q8_0 x1,Q4_K x1\n");
  printf("#   attn_out [6144,5120]:  Q6_K x9,Q5_K x7,IQ4_XS x1 (17 full-attn)\n");
  printf("#   attn_q   [5120,12288]: Q5_K x8,Q4_K x6,IQ4_XS x2,Q6_K x1 (17 FA)\n");
  printf("#   attn_k   [5120,1024]:  Q6_K x12,Q5_K x2,Q8_0 x2,Q4_K x1 (17 FA)\n");
  printf("#   attn_v   [5120,1024]:  Q8_0 x11,Q5_K x3,Q6_K x3 (17 FA)\n");
  printf("#   attn_qkv [5120,10240]: Q4_K x23,Q5_K x21,IQ4_XS x2 (48 fused)\n");
}

static bool g_csv = false;

static void Row(const char* cell, const char* route, const char* stage,
                float ms, double tflops, double gbs, bool fin,
                const char* note = "") {
  if (g_csv) {
    printf("%s,%s,%s,%.4f,%.3f,%.1f,%d,%s\n", cell, route, stage, ms, tflops,
           gbs, fin ? 1 : 0, note);
  } else {
    printf("%-16s %-12s %-8s %10.4f %8.3f %10.1f %s %s\n", cell, route, stage,
           ms, tflops, gbs, fin ? "fin" : "NONFIN", note);
  }
}

struct Shape {
  const char* cell;
  const char* roles;
  std::size_t m, k;
};

// SPEC1 item 2: the 27B shapes (m = batch is the T sweep), plus the fused
// qkv projection (SPEC2 bench3: a real production GEMM in all 48 fused
// layers; every route supports m=10240). ssm_alpha/beta (m=32) stay out:
// negligible time, noise-dominated cells.
static const Shape kShapes[] = {
    {"d17408x5120", "ffn_gate/ffn_up", 17408, 5120},
    {"d5120x17408", "ffn_down", 5120, 17408},
    {"d6144x5120", "attn_gate/ssm_out", 6144, 5120},
    {"d5120x6144", "attn_output", 5120, 6144},
    {"d12288x5120", "attn_q", 12288, 5120},
    {"d1024x5120", "attn_k/attn_v", 1024, 5120},
    {"d10240x5120", "attn_qkv", 10240, 5120},
};

static int g_fail = 0;

static void Bench(std::size_t T, int reps, hipStream_t stream,
                  std::mt19937& rng) {
  if (!g_csv)
    printf("%-16s %-12s %-8s %10s %8s %10s %s\n", "cell", "route", "stage",
           "ms", "TFLOPS", "GB/s", "fin");
  // Pad activation rows to a 128-token macro tile (the launched work is
  // still exactly T rows); only the timed grid depends on T.
  const std::size_t Tp = (T + 127) / 128 * 128;
  for (const Shape& s : kShapes) {
    const double flops = 2.0 * double(T) * double(s.m) * double(s.k);
    const double act_bytes = double(T) * double(s.k) * 4.0;
    std::vector<float> x(Tp * s.k, 0.0f);
    {
      std::vector<float> live(T * s.k);
      FillAct(live, 0.1f, rng);
      for (std::size_t t = 0; t < T; ++t)
        memcpy(&x[t * s.k], &live[t * s.k], s.k * 4);
    }
    std::vector<std::uint8_t> wq4, wf4fast, wf4;
    FillQ4K(wq4, s.m, s.k, rng);
    FillFP4(wf4fast, s.m, s.k, true, rng);
    FillFP4(wf4, s.m, s.k, false, rng);
    // SPEC2 step 1: the stock file's real per-role types.
    std::vector<std::uint8_t> wq5, wq6, wiq4xs, wq8;
    FillQ5K(wq5, s.m, s.k, rng);
    FillQ6K(wq6, s.m, s.k, rng);
    FillIQ4XS(wiq4xs, s.m, s.k, rng);
    FillQ80(wq8, s.m, s.k, rng);
    const double wbytes_q4 = double(wq4.size());
    const double wbytes_f4 = double(wf4fast.size());
    float* dx = (float*)DevAlloc(x.size() * 4);
    void* dwq4 = DevAlloc(wq4.size());
    void* dwf4fast = DevAlloc(wf4fast.size());
    void* dwf4 = DevAlloc(wf4.size());
    void* dwq5 = DevAlloc(wq5.size());
    void* dwq6 = DevAlloc(wq6.size());
    void* dwiq4xs = DevAlloc(wiq4xs.size());
    void* dwq8 = DevAlloc(wq8.size());
    ToDevice(dx, x.data(), x.size() * 4);
    ToDevice(dwq4, wq4.data(), wq4.size());
    ToDevice(dwf4fast, wf4fast.data(), wf4fast.size());
    ToDevice(dwf4, wf4.data(), wf4.size());
    ToDevice(dwq5, wq5.data(), wq5.size());
    ToDevice(dwq6, wq6.data(), wq6.size());
    ToDevice(dwiq4xs, wiq4xs.data(), wiq4xs.size());
    ToDevice(dwq8, wq8.data(), wq8.size());
    __half* dxh = (__half*)DevAlloc(Tp * s.k * 2);
    void* dq8 = DevAlloc(hipq::QuantizedActivationBytes(Tp, s.k));
    float* out = (float*)DevAlloc(Tp * s.m * 4);
    char cell[32];
    snprintf(cell, sizeof(cell), "T%zu-%s", T, s.cell);

    // stock: narrow + K-quant WMMA GEMM (the route to beat).
    float narrow_ms = TimeIt(stream, [&] {
      hipq::LaunchFloatToFp16(dx, dxh, T * s.k, stream);
    }, reps);
    Row(cell, "stock-Q4K", "narrow", narrow_ms, -1, act_bytes / narrow_ms / 1e6,
        true);
    hipq::LaunchFloatToFp16(dx, dxh, T * s.k, stream);
    HIP_OK(hipDeviceSynchronize());
    float stock_ms = TimeIt(stream, [&] {
      hipq::LaunchBatchedQuantGEMMFp16(superroc::core::GgmlType::kQ4_K, dwq4, dxh,
                                      out, T, s.m, s.k, stream);
    }, reps);
    Row(cell, "stock-Q4K", "gemm", stock_ms, flops / stock_ms / 1e9,
        wbytes_q4 / stock_ms / 1e6, FiniteProbe(out, T * s.m));
    const float stock_total = narrow_ms + stock_ms;
    Row(cell, "stock-Q4K", "total", stock_total, flops / stock_total / 1e9,
        wbytes_q4 / stock_total / 1e6, true, "narrow+gemm");

    // today: Q8_1 quant + pre-quantized FP4 GEMM (today's ROC route).
    float q_ms = TimeIt(stream, [&] {
      hipq::LaunchQuantizeActivationQ8_1FromFp32(dx, dq8, T, s.k, stream);
    }, reps);
    Row(cell, "today-FP4", "quant", q_ms, -1, act_bytes / q_ms / 1e6, true);
    hipq::LaunchQuantizeActivationQ8_1FromFp32(dx, dq8, T, s.k, stream);
    HIP_OK(hipDeviceSynchronize());
    float today_ms = TimeIt(stream, [&] {
      hipq::LaunchBatchedQuantGEMMPreQuantized(
          superroc::core::GgmlType::kQ4_0_ROCMFP4_FAST, dwf4fast, dq8, out, T, s.m,
          s.k, stream);
    }, reps);
    Row(cell, "today-FP4", "gemm", today_ms, flops / today_ms / 1e9,
        wbytes_f4 / today_ms / 1e6, FiniteProbe(out, T * s.m));
    Row(cell, "today-FP4", "total", q_ms + today_ms,
        flops / (q_ms + today_ms) / 1e9, wbytes_f4 / (q_ms + today_ms) / 1e6,
        true, "quant+gemm");

    // SPEC2 step 1: real-stock = the same Q8_1 quant (shared dq8) +
    // PreQuantized over the stock file's real per-role types. Answers (a)
    // FP4-vs-Q4_K inside one int8 kernel and (b) F16-vs-int8 for stock types.
    struct RealType {
      const char* route;
      superroc::core::GgmlType type;
      const void* w;
      double wbytes;
    };
    const RealType kReal[] = {
        {"real-Q4K", superroc::core::GgmlType::kQ4_K, dwq4, wbytes_q4},
        {"real-Q5K", superroc::core::GgmlType::kQ5_K, dwq5, double(wq5.size())},
        {"real-Q6K", superroc::core::GgmlType::kQ6_K, dwq6, double(wq6.size())},
        {"real-IQ4XS", superroc::core::GgmlType::kIQ4_XS, dwiq4xs,
         double(wiq4xs.size())},
        {"real-Q80", superroc::core::GgmlType::kQ8_0, dwq8, double(wq8.size())},
    };
    for (const RealType& rt : kReal) {
      float r_ms = TimeIt(stream, [&] {
        hipq::LaunchBatchedQuantGEMMPreQuantized(rt.type, rt.w, dq8, out, T,
                                                s.m, s.k, stream);
      }, reps);
      Row(cell, rt.route, "gemm", r_ms, flops / r_ms / 1e9,
          rt.wbytes / r_ms / 1e6, FiniteProbe(out, T * s.m));
      Row(cell, rt.route, "total", q_ms + r_ms, flops / (q_ms + r_ms) / 1e9,
          rt.wbytes / (q_ms + r_ms) / 1e6, true, "quant+gemm");
    }

    // SPEC2 step 1 (corrected): the stock F16 route over the stock file's
    // real per-role types (shared F16 narrow). Q4_K is the stock row above.
    // Every type here is in DirectGemm's switch (prefill_fp16.hip) and takes
    // this route in production on stock files.
    const RealType kF16Real[] = {
        {"f16-Q5K", superroc::core::GgmlType::kQ5_K, dwq5, double(wq5.size())},
        {"f16-Q6K", superroc::core::GgmlType::kQ6_K, dwq6, double(wq6.size())},
        {"f16-IQ4XS", superroc::core::GgmlType::kIQ4_XS, dwiq4xs,
         double(wiq4xs.size())},
        {"f16-Q80", superroc::core::GgmlType::kQ8_0, dwq8, double(wq8.size())},
    };
    for (const RealType& rt : kF16Real) {
      float f_ms = TimeIt(stream, [&] {
        hipq::LaunchBatchedQuantGEMMFp16(rt.type, rt.w, dxh, out, T, s.m, s.k,
                                        stream);
      }, reps);
      Row(cell, rt.route, "gemm", f_ms, flops / f_ms / 1e9,
          rt.wbytes / f_ms / 1e6, FiniteProbe(out, T * s.m));
      Row(cell, rt.route, "total", narrow_ms + f_ms,
          flops / (narrow_ms + f_ms) / 1e9,
          rt.wbytes / (narrow_ms + f_ms) / 1e6, true, "narrow+gemm");
    }

    // SPEC2 step 2 (bench3): FP4 through the stock F16 kernel (the fix under
    // test: DirectGemm FP4 cases in prefill_fp16.hip). Shared F16 narrow.
    struct Fp4Row {
      const char* route;
      superroc::core::GgmlType type;
      const void* w;
      double wbytes;
    };
    const Fp4Row kF16Fp4[] = {
        {"f16-101", superroc::core::GgmlType::kQ4_0_ROCMFP4_FAST, dwf4fast,
         wbytes_f4},
        {"f16-100", superroc::core::GgmlType::kQ4_0_ROCMFP4, dwf4,
         double(wf4.size())},
    };
    for (const Fp4Row& fr : kF16Fp4) {
      float p_ms = TimeIt(stream, [&] {
        hipq::LaunchBatchedQuantGEMMFp16(fr.type, fr.w, dxh, out, T, s.m, s.k,
                                        stream);
      }, reps);
      Row(cell, fr.route, "gemm", p_ms, flops / p_ms / 1e9,
          fr.wbytes / p_ms / 1e6, FiniteProbe(out, T * s.m));
      Row(cell, fr.route, "total", narrow_ms + p_ms,
          flops / (narrow_ms + p_ms) / 1e9,
          fr.wbytes / (narrow_ms + p_ms) / 1e6, true, "narrow+gemm");
      const double fratio = stock_total / (narrow_ms + p_ms);
      if (!g_csv)
        printf("# %s %s: f16fp4/stock-Q4K = %.3f (bar: >= 0.9 vs the stock-F16 mix, in REPORT2)\n",
               cell, fr.route, fratio);
    }

    // ported: narrow (shared with stock) + Fp4DenseF16Gemm, both types.
    for (int fi = 0; fi < 2; ++fi) {
      const bool fast = fi == 0;
      const char* route = fast ? "ported-101" : "ported-100";
      bool ok = false;
      float g_ms = TimeIt(stream, [&] {
        ok = f16::Fp4DenseF16Gemm(fast ? dwf4fast : dwf4, dxh, out, T, s.m,
                                  s.k, fast, stream);
      }, reps);
      if (!ok) {
        Row(cell, route, "gemm", 0, -1, -1, false, "unsupported");
        printf("FAIL %s %s unsupported\n", cell, route);
        ++g_fail;
        continue;
      }
      Row(cell, route, "gemm", g_ms, flops / g_ms / 1e9,
          (fast ? wbytes_f4 : double(wf4.size())) / g_ms / 1e6,
          FiniteProbe(out, T * s.m));
      const float total = narrow_ms + g_ms;
      Row(cell, route, "total", total, flops / total / 1e9,
          wbytes_f4 / total / 1e6, true, "narrow+gemm");
      const double ratio = stock_total / total;
      if (!g_csv)
        printf("# %s %s: ported/stock-Q4K = %.3f (info only; SPEC2 bar is vs the stock-F16 mix, in REPORT2)\n",
               cell, route, ratio);
    }
    // SPEC2 step 2 (bench3, bar soundness): production stock gate/up takes the
    // DUAL kernel when types qualify. Measure it on the gate/up cell (Q4_K/Q4_K;
    // one [B,M] F16 SwiGLU activation out) against 2x the single.
    if (std::strcmp(s.cell, "d17408x5120") == 0) {
      __half* out16 = (__half*)DevAlloc(Tp * s.m * 2);
      float d_ms = TimeIt(stream, [&] {
        if (!hipq::TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
                superroc::core::GgmlType::kQ4_K, superroc::core::GgmlType::kQ4_K, dwq4,
                dwq4, dxh, out16, T, s.m, s.k, stream)) {
          printf("FAIL %s dual-Q4K declined\n", cell);
          ++g_fail;
        }
      }, reps);
      const double dflops = 2.0 * flops;  // two GEMMs + SwiGLU, one launch
      Row(cell, "dual-Q4K", "gemm", d_ms, dflops / d_ms / 1e9,
          2.0 * wbytes_q4 / d_ms / 1e6, true);
      Row(cell, "dual-Q4K", "total", narrow_ms + d_ms,
          dflops / (narrow_ms + d_ms) / 1e9,
          2.0 * wbytes_q4 / (narrow_ms + d_ms) / 1e6, true, "narrow+dual");
      if (!g_csv)
        printf("# %s dual-Q4K: dual/(2x single) = %.3f (production stock gate/up shape)\n",
               cell, (narrow_ms + d_ms) / (narrow_ms + 2.0f * stock_ms));
      HIP_OK(hipFree(out16));
    }
    if (!g_csv) printf("# %s roles: %s\n", cell, s.roles);

    HIP_OK(hipFree(dx));
    HIP_OK(hipFree(dwq4));
    HIP_OK(hipFree(dwf4fast));
    HIP_OK(hipFree(dwf4));
    HIP_OK(hipFree(dwq5));
    HIP_OK(hipFree(dwq6));
    HIP_OK(hipFree(dwiq4xs));
    HIP_OK(hipFree(dwq8));
    HIP_OK(hipFree(dxh));
    HIP_OK(hipFree(dq8));
    HIP_OK(hipFree(out));
  }
}

static void Usage(const char* a0) {
  printf("usage: %s [T=512,1024,2048,4096] [reps=20] [--csv]\n", a0);
  exit(2);
}

int main(int argc, char** argv) {
  std::vector<std::size_t> Ts = {512, 1024, 2048, 4096};
  int reps = 20;
  bool t_set = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--csv") {
      g_csv = true;
    } else if (a == "--help" || a == "-h") {
      Usage(argv[0]);
    } else if (a.find_first_not_of("0123456789,") == std::string::npos &&
               a.find(',') != std::string::npos) {
      Ts.clear();
      std::size_t p = 0;
      while (p < a.size()) {
        std::size_t q = a.find(',', p);
        Ts.push_back(std::stoul(a.substr(p, q - p)));
        p = q == std::string::npos ? q : q + 1;
        if (q == std::string::npos) break;
      }
      t_set = true;
    } else if (a.find_first_not_of("0123456789") == std::string::npos) {
      if (t_set) {
        reps = std::stoi(a);
      } else {
        Ts = {std::size_t(std::stoul(a))};
        t_set = true;
      }
    } else {
      Usage(argv[0]);
    }
  }
  if (!g_csv) PrintCensus();
  hipStream_t stream = nullptr;
  HIP_OK(hipStreamCreate(&stream));
  std::mt19937 rng(0x5EED5EEDu);
  for (std::size_t T : Ts) Bench(T, reps, stream, rng);
  HIP_OK(hipStreamDestroy(stream));
  if (!g_csv) printf(g_fail == 0 ? "QWEN_GEMM_BENCH PASS\n"
                                 : "QWEN_GEMM_BENCH FAIL (%d)\n", g_fail);
  return g_fail == 0 ? 0 : 1;
}
