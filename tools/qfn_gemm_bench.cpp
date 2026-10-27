// tools/qfn_gemm_bench.cpp — LFT SPEC1 item 2 + SPEC2 item 2: per-shape GEMM
// microbench.
//
// Times, per dense (m, k) of the Flash-Next shapes in lft/DECISIONS.md:
//   * DenseF16Gemm over Q8_0 weights (F16 WMMA, pre-narrowed activations),
//   * the stock int8 W8A8Gemm over Q8_0 (QuantizeQ8Tiled + GEMM),
//   * today's ROC route over type-101 weights (superroc_fp4_wmma_quant + GEMM),
//   * --with-lfp: LFP's Fp4DenseF16Gemm over type-101 weights (v1.4+),
// and per routed geometry (512 experts, 10 used):
//   * stock RoutedCompact + Narrow + gate/up (gated when the executor uses
//     it) + down + MoeEpilogueVec4F16,
//   * ROC superroc_fp4_moe_group_batch + quant + 3x gemm_grp + Swiglu + gather,
//   * --with-lfq: LFQ's RoutedCompact + Narrow + gated/down over type-101
//     weights (needs the LFQ stack tree; on v1.4 the launcher declines and
//     the cells report failed-or-no-LFQ).
// Random weights/activations (timing only), HIP-event timing, median of N
// reps, finite-output smoke check per route. Routed total rows carry the
// tile width and the weight bytes the cell moves (the LFR mechanism:
// 16-token tiles re-read each expert stack, 48-token tiles read it once).
//
// Usage: qfn_gemm_bench [tokens=2048] [reps=20] [--csv] [--dense-only]
//        [--routed-only] [--with-lfp] [--with-lfq]
// Run inside gpu-run (see ~/ls/kit/gemm_shapes.sh <tree>).
// No model load; peak device use is ~4 GB (routed cells at 2048 tokens).

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/rocm_fp4/superroc_fp4_gemm.hpp"
#include "src/core/rocm_fp4/superroc_fp4_moe_group.hpp"
#include "src/core/rocm_fp4/superroc_fp4_quant.hpp"
#include "src/core/rocm_fp4/superroc_fp4_wmma.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace qfn = superroc::models::qwen38_flash_next::rocm;

#define HIP_OK(call)                                                          \
  do {                                                                        \
    const hipError_t e_hip_ok_ = (call);                                      \
    if (e_hip_ok_ != hipSuccess) {                                            \
      fprintf(stderr, "HIP %s:%d: %s failed: %s\n", __FILE__, __LINE__,        \
              #call, hipGetErrorString(e_hip_ok_));                           \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

// Round-to-nearest-even float -> f16 bits; infinities clamp to max normal.
static std::uint16_t F32ToF16(float f) {
  std::uint32_t u;
  memcpy(&u, &f, sizeof(u));
  const std::uint32_t sign = (u >> 16) & 0x8000u;
  const int exp = int((u >> 23) & 0xFFu) - 112;
  const std::uint32_t mant = u & 0x7FFFFFu;
  if (exp <= 0) return std::uint16_t(sign);  // flush subnormals to zero
  if (exp >= 31) return std::uint16_t(sign | 0x7BFFu);
  return std::uint16_t(sign | (std::uint32_t(exp) << 10) | (mant >> 13));
}

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

// ---- random weights (timing only; magnitudes kept small and finite) ----

static void FillQ80(std::vector<std::uint8_t>& buf, std::size_t rows,
                    std::size_t k, std::mt19937& rng) {
  const std::size_t nb = k / 32;
  buf.resize(rows * nb * 34);
  std::uniform_int_distribution<int> q(-16, 16);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t b = 0; b < nb; ++b) {
      auto* blk = reinterpret_cast<superroc::quant::block_q8_0*>(
          &buf[(r * nb + b) * 34]);
      blk->d = F32ToF16(0.02f);
      for (int j = 0; j < 32; ++j) blk->qs[j] = std::int8_t(q(rng));
    }
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
      blk->d = F32ToF16(0.002f);
      blk->dmin = F32ToF16(0.0f);
      for (int j = 0; j < 12; ++j) blk->scales[j] = std::uint8_t(s(rng));
      for (int j = 0; j < 128; ++j) blk->qs[j] = std::uint8_t(b(rng));
    }
}

// Type-101 ROCMFP4_FAST: 16 nibble bytes + one UE4M3 scale byte; 0x1C ~= 0.047.
static void FillFP4(std::vector<std::uint8_t>& buf, std::size_t rows,
                    std::size_t k, std::mt19937& rng) {
  const std::size_t nb = k / 32;
  buf.resize(rows * nb * 17);
  std::uniform_int_distribution<int> b(0, 255);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t n = 0; n < nb; ++n) {
      std::uint8_t* blk = &buf[(r * nb + n) * 17];
      for (int j = 0; j < 16; ++j) blk[j] = std::uint8_t(b(rng));
      blk[16] = 0x1C;
    }
}

static void FillAct(std::vector<float>& v, float amp, std::mt19937& rng) {
  std::uniform_real_distribution<float> d(-amp, amp);
  for (float& x : v) x = d(rng);
}

// ---- reporting ----

static bool g_csv = false;

static void Row(const char* cell, const char* route, const char* stage,
                double ms, double tflops, double gbps, int finite,
                const char* note = "") {
  if (g_csv) {
    char tf[32], gb[32];
    snprintf(tf, sizeof(tf), tflops < 0 ? "" : "%.4f", tflops);
    snprintf(gb, sizeof(gb), gbps < 0 ? "" : "%.2f", gbps);
    printf("%s,%s,%s,%.4f,%s,%s,%d,%s\n", cell, route, stage, ms, tf, gb,
           finite, note);
    return;
  }
  char tf[32], gb[32];
  snprintf(tf, sizeof(tf), tflops < 0 ? "n/a" : "%.2f", tflops);
  snprintf(gb, sizeof(gb), gbps < 0 ? "n/a" : "%.1f", gbps);
  printf("%-16s %-10s %-8s %10.3f %8s %10s %d %s\n", cell, route, stage, ms,
         tf, gb, finite, note);
  fflush(stdout);  // exact fault attribution: no buffered doubt
}

// ---- dense cells ----

struct DenseShape {
  const char* cell;
  const char* roles;
  std::size_t m, k;
};

// Every unique dense (m, k) in lft/DECISIONS.md (GGUF dims are [k, m]).
static const DenseShape kDense[] = {
    {"d10240x2560", "attn_qkv/ple_key", 10240, 2560},
    {"d12288x2560", "attn_q", 12288, 2560},
    {"d6144x2560", "attn_gate", 6144, 2560},
    {"d640x2560", "shexp_gate/up", 640, 2560},
    {"d512x2560", "attn_k/attn_v/indexer_q", 512, 2560},
    {"d128x2560", "indexer_k", 128, 2560},
    {"d48x2560", "ssm_alpha/ssm_beta", 48, 2560},
    {"d2560x6144", "ssm_out/attn_output", 2560, 6144},
    {"d2560x2560", "ple_value", 2560, 2560},
    {"d2560x640", "shexp_down", 2560, 640},
    {"d10240x320", "hc_attn/ffn_up", 10240, 320},
    {"d320x10240", "hc_attn/ffn_down", 320, 10240},
    {"d4x10240", "hc_attn/ffn_inject", 4, 10240},
};

static void BenchDense(std::size_t T, int reps, hipStream_t stream,
                       std::mt19937& rng) {
  if (!g_csv)
    printf("%-16s %-10s %-8s %10s %8s %10s %s\n", "cell", "route", "stage",
           "ms", "TFLOPS", "GB/s", "fin");
  // Row buffers are padded to a 128-token macro tile (the launched work is
  // still exactly T rows): the dense kernels address whole 64-token spans,
  // so an exactly-T-sized buffer over-reads when T < 64 (this faulted the
  // GPU at T=16 on 2026-10-09). The executor does the same (max-batch
  // buffers); only the timed grid depends on T.
  const std::size_t Tp = (T + 127) / 128 * 128;
  for (const DenseShape& s : kDense) {
    const double flops = 2.0 * double(T) * double(s.m) * double(s.k);
    const double act_bytes = double(T) * double(s.k) * 4.0;
    // Host buffers.
    std::vector<float> x(Tp * s.k, 0.0f);
    {  // live T rows random, pad rows zero
      std::vector<float> live(T * s.k);
      FillAct(live, 0.1f, rng);
      for (std::size_t t = 0; t < T; ++t)
        memcpy(&x[t * s.k], &live[t * s.k], s.k * 4);
    }
    std::vector<std::uint8_t> wq8, wf4;
    FillQ80(wq8, s.m, s.k, rng);
    FillFP4(wf4, s.m, s.k, rng);
    const double wbytes_q8 = double(wq8.size());
    const double wbytes_f4 = double(wf4.size());
    // Device buffers.
    float* dx = (float*)DevAlloc(x.size() * 4);
    void* dwq8 = DevAlloc(wq8.size());
    void* dwf4 = DevAlloc(wf4.size());
    ToDevice(dx, x.data(), x.size() * 4);
    ToDevice(dwq8, wq8.data(), wq8.size());
    ToDevice(dwf4, wf4.data(), wf4.size());
    __half* dxh = (__half*)DevAlloc(Tp * s.k * 2);
    void* dq8 = DevAlloc(qfn::Q8TiledBytes(Tp, s.k));
    float* out = (float*)DevAlloc(Tp * s.m * 4);
    // A dense block spans 4 activation tiles per 64-token tile; +2 margin.
    const std::size_t f4tiles = (Tp + 63) / 64 * 4 + 2;
    void* f4q8 = DevAlloc(f4tiles * std::max<std::size_t>(1, s.k / 32) * 576);

    // F16 route: narrow once (timed separately), GEMM on cached rows.
    float narrow_ms =
        TimeIt(stream, [&] { qfn::NarrowActivations(dx, dxh, false,
                                                   T * s.k, stream); }, reps);
    Row(s.cell, "F16-Q8_0", "narrow", narrow_ms, -1,
        act_bytes / narrow_ms / 1e6, 1);
    qfn::NarrowActivations(dx, dxh, false, T * s.k, stream);
    HIP_OK(hipDeviceSynchronize());
    bool ok16 = false;
    float f16_ms = TimeIt(stream, [&] {
      ok16 = qfn::DenseF16Gemm(dwq8, dxh, out, T, s.m, s.k, stream);
    }, reps);
    if (!ok16) {
      Row(s.cell, "F16-Q8_0", "gemm", 0, -1, -1, 0, "unsupported");
    } else {
      Row(s.cell, "F16-Q8_0", "gemm", f16_ms, flops / f16_ms / 1e9,
          wbytes_q8 / f16_ms / 1e6, FiniteProbe(out, T * s.m));
      Row(s.cell, "F16-Q8_0", "total", narrow_ms + f16_ms,
          flops / (narrow_ms + f16_ms) / 1e9,
          wbytes_q8 / (narrow_ms + f16_ms) / 1e6, 1, "narrow+gemm");
    }

    // W8A8 route.
    float q8_ms = TimeIt(stream, [&] {
      qfn::QuantizeQ8Tiled(dx, dq8, T, s.k, stream);
    }, reps);
    Row(s.cell, "W8A8-Q8_0", "quant", q8_ms, -1, act_bytes / q8_ms / 1e6, 1);
    qfn::QuantizeQ8Tiled(dx, dq8, T, s.k, stream);
    HIP_OK(hipDeviceSynchronize());
    bool ok8 = false;
    float w8_ms = TimeIt(stream, [&] {
      ok8 = qfn::W8A8Gemm(dwq8, dq8, out, T, s.m, s.k, stream);
    }, reps);
    if (!ok8) {
      Row(s.cell, "W8A8-Q8_0", "gemm", 0, -1, -1, 0, "unsupported");
    } else {
      Row(s.cell, "W8A8-Q8_0", "gemm", w8_ms, flops / w8_ms / 1e9,
          wbytes_q8 / w8_ms / 1e6, FiniteProbe(out, T * s.m));
      Row(s.cell, "W8A8-Q8_0", "total", q8_ms + w8_ms,
          flops / (q8_ms + w8_ms) / 1e9, wbytes_q8 / (q8_ms + w8_ms) / 1e6, 1,
          "quant+gemm");
    }

    // FP4 route (superroc_fp4 calls run on the NULL stream).
    int qrc = 0;
    float f4q_ms = TimeIt(nullptr, [&] {
      qrc = superroc_fp4_wmma_quant(dx, nullptr, nullptr, f4q8, int(T), int(s.k));
    }, reps);
    if (qrc != SUPERROC_FP4_OK) {
      Row(s.cell, "FP4-101", "quant", 0, -1, -1, 0, "quant-failed");
    } else {
      Row(s.cell, "FP4-101", "quant", f4q_ms, -1, act_bytes / f4q_ms / 1e6, 1);
      int grc = 0;
      float f4g_ms = TimeIt(nullptr, [&] {
        grc = superroc_fp4_wmma_gemm_dense(SUPERROC_FP4_Q4_0_ROCMFP4_FAST, dwf4, f4q8,
                                      out, int(s.k), int(s.m), int(T));
      }, reps);
      if (grc != SUPERROC_FP4_OK) {
        Row(s.cell, "FP4-101", "gemm", 0, -1, -1, 0, "gemm-failed");
      } else {
        Row(s.cell, "FP4-101", "gemm", f4g_ms, flops / f4g_ms / 1e9,
            wbytes_f4 / f4g_ms / 1e6, FiniteProbe(out, T * s.m));
        Row(s.cell, "FP4-101", "total", f4q_ms + f4g_ms,
            flops / (f4q_ms + f4g_ms) / 1e9,
            wbytes_f4 / (f4q_ms + f4g_ms) / 1e6, 1, "quant+gemm");
      }
    }
    if (!g_csv) printf("# %s roles: %s\n", s.cell, s.roles);

    HIP_OK(hipFree(dx));
    HIP_OK(hipFree(dwq8));
    HIP_OK(hipFree(dwf4));
    HIP_OK(hipFree(dxh));
    HIP_OK(hipFree(dq8));
    HIP_OK(hipFree(out));
    HIP_OK(hipFree(f4q8));
  }
}

// ---- dense LFP cells (SPEC2 item 2): Fp4DenseF16Gemm over type-101 ----

static void BenchDenseLFP(std::size_t T, int reps, hipStream_t stream,
                          std::mt19937& rng) {
  // Same padding contract as BenchDense (128-token macro tile; the timed
  // grid is exactly T rows).
  const std::size_t Tp = (T + 127) / 128 * 128;
  for (const DenseShape& s : kDense) {
    const double flops = 2.0 * double(T) * double(s.m) * double(s.k);
    const double act_bytes = double(T) * double(s.k) * 4.0;
    std::vector<float> x(Tp * s.k, 0.0f);
    {
      std::vector<float> live(T * s.k);
      FillAct(live, 0.1f, rng);
      for (std::size_t t = 0; t < T; ++t)
        memcpy(&x[t * s.k], &live[t * s.k], s.k * 4);
    }
    std::vector<std::uint8_t> wf4;
    FillFP4(wf4, s.m, s.k, rng);
    const double wbytes_f4 = double(wf4.size());
    float* dx = (float*)DevAlloc(x.size() * 4);
    void* dwf4 = DevAlloc(wf4.size());
    ToDevice(dx, x.data(), x.size() * 4);
    ToDevice(dwf4, wf4.data(), wf4.size());
    __half* dxh = (__half*)DevAlloc(Tp * s.k * 2);
    float* out = (float*)DevAlloc(Tp * s.m * 4);

    // Narrow is timed per shape, exactly like the F16-Q8_0 cells, so the
    // totals compare apples-to-apples (the executor narrows once per
    // distinct width per chunk and shares it across shapes).
    float narrow_ms =
        TimeIt(stream, [&] { qfn::NarrowActivations(dx, dxh, false,
                                                   T * s.k, stream); }, reps);
    Row(s.cell, "LFP-F16", "narrow", narrow_ms, -1,
        act_bytes / narrow_ms / 1e6, 1);
    qfn::NarrowActivations(dx, dxh, false, T * s.k, stream);
    HIP_OK(hipDeviceSynchronize());
    bool ok = false;
    float g_ms = TimeIt(stream, [&] {
      ok = qfn::Fp4DenseF16Gemm(dwf4, dxh, out, T, s.m, s.k,
                                /*fast=*/true, stream);
    }, reps);
    if (!ok) {
      Row(s.cell, "LFP-F16", "gemm", 0, -1, -1, 0, "unsupported");
    } else {
      Row(s.cell, "LFP-F16", "gemm", g_ms, flops / g_ms / 1e9,
          wbytes_f4 / g_ms / 1e6, FiniteProbe(out, T * s.m));
      Row(s.cell, "LFP-F16", "total", narrow_ms + g_ms,
          flops / (narrow_ms + g_ms) / 1e9,
          wbytes_f4 / (narrow_ms + g_ms) / 1e6, 1, "narrow+gemm");
    }
    HIP_OK(hipFree(dx));
    HIP_OK(hipFree(dwf4));
    HIP_OK(hipFree(dxh));
    HIP_OK(hipFree(out));
  }
}

// ---- routed cells: 512 experts, 10 used, hidden 2560, ff 640 ----

static void BenchRouted(std::size_t T, int reps, hipStream_t stream,
                        std::mt19937& rng) {
  const std::uint32_t E = 512, U = 10, H = 2560, F = 640;
  const std::size_t slots = T * U;
  // Row buffers padded to a 128-token macro tile (see BenchDense); index
  // arrays stay exact (all routed indices are slot/token-bounded).
  const std::size_t Tp = (T + 127) / 128 * 128;
  const std::size_t slots_p = Tp * U;
  const double flops =
      2.0 * double(slots) * (2.0 * double(F) * double(H) +
                             double(H) * double(F));  // gate+up+down
  const char* cell = "routed512x10";

  // Deterministic routing: slot j -> expert (j*7919+17) % 512, uniform wts.
  std::vector<std::int32_t> ids(slots);
  std::vector<float> wts(slots, 0.1f);
  std::vector<std::uint32_t> counts(E, 0);
  for (std::size_t j = 0; j < slots; ++j) {
    ids[j] = std::int32_t((j * 7919 + 17) % E);
    counts[ids[j]]++;
  }
  std::vector<float> x(Tp * H, 0.0f);
  {
    std::vector<float> live(T * H);
    FillAct(live, 0.1f, rng);
    for (std::size_t t = 0; t < T; ++t)
      memcpy(&x[t * H], &live[t * H], H * 4);
  }

  // ---- stock: compact + narrow + gate/up + down + epilogue ----
  {
    std::vector<std::uint8_t> gate, up, down;
    FillQ4K(gate, F, H, rng);  // per expert; stacked below
    FillQ4K(up, F, H, rng);
    FillQ80(down, H, F, rng);
    const std::size_t gate_eb = gate.size(), down_eb = down.size();
    std::vector<std::uint8_t> gate_all(gate_eb * E), up_all(gate_eb * E),
        down_all(down_eb * E);
    for (std::uint32_t e = 0; e < E; ++e) {
      memcpy(&gate_all[std::size_t(e) * gate_eb], gate.data(), gate_eb);
      memcpy(&up_all[std::size_t(e) * gate_eb], up.data(), gate_eb);
      memcpy(&down_all[std::size_t(e) * down_eb], down.data(), down_eb);
    }
    const double wbytes = double(gate_all.size() + up_all.size() +
                                 down_all.size());  // full sweep
    std::int32_t* dids = (std::int32_t*)DevAlloc(slots * 4);
    std::uint32_t* dcounts = (std::uint32_t*)DevAlloc(E * 4);
    std::int32_t* bounds = (std::int32_t*)DevAlloc((E + 1) * 4);
    std::int32_t* cursors = (std::int32_t*)DevAlloc(E * 4);
    const std::size_t compact = qfn::RoutedCompactRows(slots, E);
    std::int32_t* rows_tok = (std::int32_t*)DevAlloc(compact * 4);
    std::int32_t* rows_slot = (std::int32_t*)DevAlloc(compact * 4);
    float* dx = (float*)DevAlloc(x.size() * 4);
    __half* dxh = (__half*)DevAlloc(Tp * H * 2);
    void* dgate = DevAlloc(gate_all.size());
    void* dup = DevAlloc(up_all.size());
    void* ddown = DevAlloc(down_all.size());
    float* gate_e = (float*)DevAlloc(slots_p * F * 4);
    __half* up_e = (__half*)DevAlloc(slots_p * F * 2);
    __half* down_e = (__half*)DevAlloc(slots_p * H * 2);
    float* dwts = (float*)DevAlloc(slots * 4);
    float* shared = (float*)DevAlloc(Tp * H * 4);
    float* dgate1 = (float*)DevAlloc(Tp * 4);
    float* out = (float*)DevAlloc(Tp * H * 4);
    ToDevice(dids, ids.data(), slots * 4);
    ToDevice(dcounts, counts.data(), E * 4);
    ToDevice(dx, x.data(), x.size() * 4);
    ToDevice(dgate, gate_all.data(), gate_all.size());
    ToDevice(dup, up_all.data(), up_all.size());
    ToDevice(ddown, down_all.data(), down_all.size());
    ToDevice(dwts, wts.data(), slots * 4);
    HIP_OK(hipMemset(shared, 0, Tp * H * 4));
    HIP_OK(hipMemset(dgate1, 0, Tp * 4));

    float compact_ms = TimeIt(stream, [&] {
      qfn::RoutedCompact(dids, dcounts, bounds, cursors, rows_tok, rows_slot,
                         std::uint32_t(T), U, E, stream);
    }, reps);
    Row(cell, "STK-Q4K", "compact", compact_ms, -1, -1, 1);
    qfn::RoutedCompact(dids, dcounts, bounds, cursors, rows_tok, rows_slot,
                       std::uint32_t(T), U, E, stream);
    float narrow_ms = TimeIt(stream, [&] {
      qfn::NarrowActivations(dx, dxh, false, T * H, stream);
    }, reps);
    Row(cell, "STK-Q4K", "narrow", narrow_ms, -1,
        double(T) * double(H) * 4.0 / narrow_ms / 1e6, 1);
    qfn::NarrowActivations(dx, dxh, false, T * H, stream);
    HIP_OK(hipDeviceSynchronize());

    // Tile maps, replicating Executor::RouteHints (48/64/128-row maps).
    const std::uint32_t tile_rows =
        slots >= std::size_t(16) * E ? 48 : 16;
    std::vector<std::int32_t> tiles;
    tiles.reserve(slots / 8 + 3 * E + 8);
    for (std::uint32_t e = 0; e < E; ++e) {
      const std::uint32_t padded = (counts[e] + 15) / 16 * 16;
      for (std::uint32_t j = 0; j < (padded + tile_rows - 1) / tile_rows; ++j)
        tiles.push_back(std::int32_t(e | (j << 16)));
    }
    const std::uint32_t n48 = std::uint32_t(tiles.size());
    std::uint32_t pair_off = 0, pair_tiles = 0, pair_rows = 64;
    std::uint32_t n64 = 0;
    if (T >= 1024 && tile_rows == 48) {
      for (std::uint32_t e = 0; e < E; ++e) {
        const std::uint32_t padded = (counts[e] + 15) / 16 * 16;
        for (std::uint32_t j = 0; j < (padded + 63) / 64; ++j)
          tiles.push_back(std::int32_t(e | (j << 16)));
      }
      n64 = std::uint32_t(tiles.size()) - n48;
      pair_off = n48;
      pair_tiles = n64;
      std::uint32_t t128 = 0;
      for (std::uint32_t e = 0; e < E; ++e) t128 += (counts[e] + 127) / 128;
      if (t128 * 4 <= n64 * 3) {
        pair_rows = 128;
        pair_off = std::uint32_t(tiles.size());
        pair_tiles = t128;
        for (std::uint32_t e = 0; e < E; ++e)
          for (std::uint32_t j = 0; j < (counts[e] + 127) / 128; ++j)
            tiles.push_back(std::int32_t(e | (j << 16)));
      }
    }
    std::int32_t* dtiles = (std::int32_t*)DevAlloc(tiles.size() * 4);
    ToDevice(dtiles, tiles.data(), tiles.size() * 4);

    float gu_ms = 0;
    bool gu_ok = true;
    if (T >= 1024 && tile_rows == 48) {
      gu_ms = TimeIt(stream, [&] {
        gu_ok = qfn::RoutedGatedF16Gemm(dgate, dup, qfn::WeightType::kQ4_K,
                                       dxh, dtiles + pair_off, pair_tiles,
                                       pair_rows, bounds, rows_tok, rows_slot,
                                       up_e, F, H, stream);
      }, reps);
      char gnote[48];
      if (gu_ok)
        snprintf(gnote, sizeof(gnote), "tile%u tiles=%u", pair_rows,
                 pair_tiles);
      else
        snprintf(gnote, sizeof(gnote), "gated-failed");
      Row(cell, "STK-Q4K", "gated", gu_ms,
          gu_ok ? (2.0 * double(slots) * 2.0 * double(F) * double(H) / gu_ms /
                   1e9)
                : -1,
          -1, gu_ok ? 1 : 0, gnote);
    } else {
      float g_ms = TimeIt(stream, [&] {
        gu_ok = qfn::RoutedF16Gemm(dgate, qfn::WeightType::kQ4_K, dxh, dtiles,
                                  n48, tile_rows, bounds, rows_tok, rows_slot,
                                  nullptr, gate_e, nullptr, F, H, stream) &&
                 gu_ok;
      }, reps);
      float u_ms = TimeIt(stream, [&] {
        gu_ok = qfn::RoutedF16Gemm(dup, qfn::WeightType::kQ4_K, dxh, dtiles,
                                  n48, tile_rows, bounds, rows_tok, rows_slot,
                                  gate_e, nullptr, up_e, F, H, stream) &&
                 gu_ok;
      }, reps);
      gu_ms = g_ms + u_ms;
      char tbuf[48];
      snprintf(tbuf, sizeof(tbuf), "tile%u tiles=%u", tile_rows, n48);
      Row(cell, "STK-Q4K", "gate", g_ms, -1, -1, gu_ok ? 1 : 0,
          gu_ok ? tbuf : "gate-failed");
      Row(cell, "STK-Q4K", "up", u_ms, -1, -1, gu_ok ? 1 : 0,
          gu_ok ? tbuf : "up-failed");
    }
    const bool wide_down = tile_rows == 48 && n64 != 0 && n64 * 4 <= n48 * 3;
    float d_ms = 0;
    bool d_ok = false;
    if (gu_ok) {
      d_ms = TimeIt(stream, [&] {
        d_ok = qfn::RoutedF16Gemm(ddown, qfn::WeightType::kQ8_0, up_e,
                                 dtiles + (wide_down ? n48 : 0),
                                 wide_down ? n64 : n48,
                                 wide_down ? 64 : tile_rows, bounds, rows_slot,
                                 rows_slot, nullptr, nullptr, down_e, H, F,
                                 stream);
      }, reps);
    }
    char dnote[64];
    if (d_ok)
      snprintf(dnote, sizeof(dnote), "tile%u tiles=%u",
               wide_down ? 64 : tile_rows, wide_down ? n64 : n48);
    else
      snprintf(dnote, sizeof(dnote), "down-failed");
    Row(cell, "STK-Q4K", "down", d_ms, -1, -1, d_ok ? 1 : 0, dnote);
    float epi_ms = 0;
    if (d_ok) {
      epi_ms = TimeIt(stream, [&] {
        qfn::MoeEpilogueVec4F16(down_e, dwts, shared, dgate1, 1, out,
                               std::uint32_t(T), U, H, stream);
      }, reps);
      Row(cell, "STK-Q4K", "epilogue", epi_ms, -1, -1,
          FiniteProbe(out, T * H));
      const float total = compact_ms + narrow_ms + gu_ms + d_ms + epi_ms;
      // Weight bytes moved: every tile sweeps its expert's full matrix.
      const std::size_t gu_tiles =
          (T >= 1024 && tile_rows == 48) ? pair_tiles : n48;
      const std::size_t dn_tiles = wide_down ? n64 : n48;
      const double wmoved = double(gu_tiles) * double(gate_eb) * 2.0 +
                            double(dn_tiles) * double(down_eb);
      char tnote[128];
      snprintf(tnote, sizeof(tnote), "Wmoved=%.3fGB sweep=%.3fGB",
               wmoved / 1e9, wbytes / 1e9);
      Row(cell, "STK-Q4K", "total", total, flops / total / 1e9,
          wbytes / total / 1e6, 1, tnote);
    }
    HIP_OK(hipFree(dids));
    HIP_OK(hipFree(dcounts));
    HIP_OK(hipFree(bounds));
    HIP_OK(hipFree(cursors));
    HIP_OK(hipFree(rows_tok));
    HIP_OK(hipFree(rows_slot));
    HIP_OK(hipFree(dx));
    HIP_OK(hipFree(dxh));
    HIP_OK(hipFree(dgate));
    HIP_OK(hipFree(dup));
    HIP_OK(hipFree(ddown));
    HIP_OK(hipFree(gate_e));
    HIP_OK(hipFree(up_e));
    HIP_OK(hipFree(down_e));
    HIP_OK(hipFree(dwts));
    HIP_OK(hipFree(shared));
    HIP_OK(hipFree(dgate1));
    HIP_OK(hipFree(out));
    HIP_OK(hipFree(dtiles));
  }

  // ---- ROC: group + quant + gate/up + swiglu + quant + down + gather ----
  {
    const int Q = SUPERROC_FP4_Q4_0_ROCMFP4_FAST;
    const std::size_t rb_k = superroc_fp4_row_bytes(Q, H);
    const std::size_t rb_f = superroc_fp4_row_bytes(Q, F);
    const std::size_t stride_k = std::size_t(F) * rb_k;
    const std::size_t stride_f = std::size_t(H) * rb_f;
    std::vector<std::uint8_t> g1, u1, d1;
    FillFP4(g1, F, H, rng);
    FillFP4(u1, F, H, rng);
    FillFP4(d1, H, F, rng);
    std::vector<std::uint8_t> gate_all(stride_k * E), up_all(stride_k * E),
        down_all(stride_f * E);
    for (std::uint32_t e = 0; e < E; ++e) {
      memcpy(&gate_all[std::size_t(e) * stride_k], g1.data(), stride_k);
      memcpy(&up_all[std::size_t(e) * stride_k], u1.data(), stride_k);
      memcpy(&down_all[std::size_t(e) * stride_f], d1.data(), stride_f);
    }
    const double wbytes = double(gate_all.size() + up_all.size() +
                                 down_all.size());  // full sweep
    const int n = int(T), used = int(U), nex = int(E), hid = int(H),
              nff = int(F), nslots = int(slots);
    const int gcap = 256;  // kFp4GroupCap
    const int garr = int(slots_p) / gcap + nex + 2;
    // Fp4TileCapacity at the padded slot count (margin over the exact
    // formula); the runtime total_tiles check below still applies.
    const std::size_t tiles_cap = slots_p / 16 + slots_p / gcap + E + 2;
    std::int32_t* dids = (std::int32_t*)DevAlloc(slots * 4);
    float* dwts = (float*)DevAlloc(slots * 4);
    std::int32_t* slot_tok = (std::int32_t*)DevAlloc(slots * 4);
    float* slot_w = (float*)DevAlloc(slots * 4);
    std::int32_t* token_slots = (std::int32_t*)DevAlloc(slots * 4);
    std::int32_t* slot_dest = (std::int32_t*)DevAlloc(slots * 4);
    std::int32_t* grp_e = (std::int32_t*)DevAlloc(garr * 4);
    std::int32_t* grp_off = (std::int32_t*)DevAlloc(garr * 4);
    std::int32_t* grp_cnt = (std::int32_t*)DevAlloc(garr * 4);
    std::int32_t* grp_zp = (std::int32_t*)DevAlloc((garr + 1) * 4);
    void* q8_e = DevAlloc(tiles_cap * (H / 32) * 576);
    void* q8_h = DevAlloc(tiles_cap * (F / 32) * 576);
    float* dx = (float*)DevAlloc(x.size() * 4);
    void* dgate = DevAlloc(gate_all.size());
    void* dup = DevAlloc(up_all.size());
    void* ddown = DevAlloc(down_all.size());
    float* gate_e = (float*)DevAlloc(slots_p * F * 4);
    float* up_e = (float*)DevAlloc(slots_p * F * 4);
    float* down_e = (float*)DevAlloc(slots_p * H * 4);
    float* out = (float*)DevAlloc(Tp * H * 4);
    ToDevice(dids, ids.data(), slots * 4);
    ToDevice(dwts, wts.data(), slots * 4);
    ToDevice(dx, x.data(), x.size() * 4);
    ToDevice(dgate, gate_all.data(), gate_all.size());
    ToDevice(dup, up_all.data(), up_all.size());
    ToDevice(ddown, down_all.data(), down_all.size());

    int hdr[4] = {0, 0, 0, 0};
    int grc = 0;
    float grp_ms = TimeIt(nullptr, [&] {
      grc = superroc_fp4_moe_group_batch(dids, dwts, nullptr, n, used, used, nex,
                                     gcap, slot_tok, slot_w, token_slots,
                                     grp_e, grp_off, grp_cnt, grp_zp,
                                     slot_dest, garr, nslots, hdr);
    }, reps);
    const int ngroups = hdr[0], total_tiles = hdr[2], status = hdr[3];
    const bool gok = grc == SUPERROC_FP4_OK && status == SUPERROC_FP4_OK &&
                     ngroups > 0 && total_tiles > 0 &&
                     std::uint32_t(total_tiles) <= tiles_cap;
    char gnote[96];
    if (gok)
      snprintf(gnote, sizeof(gnote), "ngroups=%d tiles=%d", ngroups,
               total_tiles);
    else
      snprintf(gnote, sizeof(gnote), "group-failed rc=%d status=%d", grc,
               status);
    Row(cell, "FP4-101", "group", grp_ms, -1, -1, gok ? 1 : 0, gnote);
    if (gok) {
      // Re-group once outside the timer so the timed GEMMs see valid maps.
      superroc_fp4_moe_group_batch(dids, dwts, nullptr, n, used, used, nex, gcap,
                               slot_tok, slot_w, token_slots, grp_e, grp_off,
                               grp_cnt, grp_zp, slot_dest, garr, nslots, hdr);
      int rc = 0;
      float q1 = TimeIt(nullptr, [&] {
        rc = superroc_fp4_wmma_quant(dx, slot_tok, slot_dest, q8_e, nslots, hid);
      }, reps);
      Row(cell, "FP4-101", "quant1", q1, -1,
          double(slots) * double(H) * 4.0 / q1 / 1e6, rc == SUPERROC_FP4_OK);
      float g_ms = TimeIt(nullptr, [&] {
        rc = superroc_fp4_wmma_gemm_grp(Q, dgate, stride_k, grp_e, grp_off,
                                    grp_cnt, grp_zp, ngroups, total_tiles,
                                    q8_e, gate_e, nullptr, hid, nff);
      }, reps);
      Row(cell, "FP4-101", "gate", g_ms, -1, -1, rc == SUPERROC_FP4_OK);
      float u_ms = TimeIt(nullptr, [&] {
        rc = superroc_fp4_wmma_gemm_grp(Q, dup, stride_k, grp_e, grp_off, grp_cnt,
                                    grp_zp, ngroups, total_tiles, q8_e, up_e,
                                    nullptr, hid, nff);
      }, reps);
      Row(cell, "FP4-101", "up", u_ms, -1, -1, rc == SUPERROC_FP4_OK);
      float sw_ms = TimeIt(stream, [&] {
        qfn::Swiglu(gate_e, up_e, slots * F, stream);
      }, reps);
      Row(cell, "FP4-101", "swiglu", sw_ms, -1, -1, 1);
      HIP_OK(hipDeviceSynchronize());
      float q2 = TimeIt(nullptr, [&] {
        rc = superroc_fp4_wmma_quant(gate_e, nullptr, slot_dest, q8_h, nslots,
                                 nff);
      }, reps);
      Row(cell, "FP4-101", "quant2", q2, -1,
          double(slots) * double(F) * 4.0 / q2 / 1e6, rc == SUPERROC_FP4_OK);
      float d_ms = TimeIt(nullptr, [&] {
        rc = superroc_fp4_wmma_gemm_grp(Q, ddown, stride_f, grp_e, grp_off,
                                    grp_cnt, grp_zp, ngroups, total_tiles,
                                    q8_h, down_e, slot_w, nff, hid);
      }, reps);
      Row(cell, "FP4-101", "down", d_ms, -1, -1, rc == SUPERROC_FP4_OK);
      float ga_ms = TimeIt(nullptr, [&] {
        rc = superroc_fp4_slot_gather_sum(down_e, token_slots, out, used, hid, n);
      }, reps);
      Row(cell, "FP4-101", "gather", ga_ms, -1, -1, rc == SUPERROC_FP4_OK);
      const float total = grp_ms + q1 + g_ms + u_ms + sw_ms + q2 + d_ms + ga_ms;
      // 16-token tiles: each tile re-sweeps its expert's full stack.
      const double wmoved =
          double(total_tiles) * (2.0 * double(stride_k) + double(stride_f));
      char tnote[160];
      snprintf(tnote, sizeof(tnote), "tile16 tiles=%d Wmoved=%.3fGB sweep=%.3fGB",
               total_tiles, wmoved / 1e9, wbytes / 1e9);
      Row(cell, "FP4-101", "total", total, flops / total / 1e9,
          wbytes / total / 1e6, FiniteProbe(out, T * H), tnote);
    }
    HIP_OK(hipFree(dids));
    HIP_OK(hipFree(dwts));
    HIP_OK(hipFree(slot_tok));
    HIP_OK(hipFree(slot_w));
    HIP_OK(hipFree(token_slots));
    HIP_OK(hipFree(slot_dest));
    HIP_OK(hipFree(grp_e));
    HIP_OK(hipFree(grp_off));
    HIP_OK(hipFree(grp_cnt));
    HIP_OK(hipFree(grp_zp));
    HIP_OK(hipFree(q8_e));
    HIP_OK(hipFree(q8_h));
    HIP_OK(hipFree(dx));
    HIP_OK(hipFree(dgate));
    HIP_OK(hipFree(dup));
    HIP_OK(hipFree(ddown));
    HIP_OK(hipFree(gate_e));
    HIP_OK(hipFree(up_e));
    HIP_OK(hipFree(down_e));
    HIP_OK(hipFree(out));
  }
}

// ---- routed LFQ cells (SPEC2 item 2): RoutedCompact + gated/down over
// type-101 weights. Mirrors the stock block in BenchRouted exactly (same
// routing, same tile plan, same epilogue); only the weight bytes and the
// launcher type change. Needs the LFQ stack tree: on v1.4 the launcher
// declines FP4 types and every GEMM row reports failed-or-no-LFQ.

static void BenchRoutedLFQ(std::size_t T, int reps, hipStream_t stream,
                           std::mt19937& rng) {
  const std::uint32_t E = 512, U = 10, H = 2560, F = 640;
  const std::size_t slots = T * U;
  const std::size_t Tp = (T + 127) / 128 * 128;
  const std::size_t slots_p = Tp * U;
  const double flops =
      2.0 * double(slots) * (2.0 * double(F) * double(H) +
                             double(H) * double(F));  // gate+up+down
  const char* cell = "routed512x10";
  const qfn::WeightType kFP4 = qfn::WeightType::kQ4_0_ROCMFP4_FAST;

  std::vector<std::int32_t> ids(slots);
  std::vector<float> wts(slots, 0.1f);
  std::vector<std::uint32_t> counts(E, 0);
  for (std::size_t j = 0; j < slots; ++j) {
    ids[j] = std::int32_t((j * 7919 + 17) % E);
    counts[ids[j]]++;
  }
  std::vector<float> x(Tp * H, 0.0f);
  {
    std::vector<float> live(T * H);
    FillAct(live, 0.1f, rng);
    for (std::size_t t = 0; t < T; ++t)
      memcpy(&x[t * H], &live[t * H], H * 4);
  }

  std::vector<std::uint8_t> gate, up, down;
  FillFP4(gate, F, H, rng);
  FillFP4(up, F, H, rng);
  FillFP4(down, H, F, rng);
  const std::size_t gate_eb = gate.size(), down_eb = down.size();
  std::vector<std::uint8_t> gate_all(gate_eb * E), up_all(gate_eb * E),
      down_all(down_eb * E);
  for (std::uint32_t e = 0; e < E; ++e) {
    memcpy(&gate_all[std::size_t(e) * gate_eb], gate.data(), gate_eb);
    memcpy(&up_all[std::size_t(e) * gate_eb], up.data(), gate_eb);
    memcpy(&down_all[std::size_t(e) * down_eb], down.data(), down_eb);
  }
  const double wbytes = double(gate_all.size() + up_all.size() +
                               down_all.size());  // full sweep
  std::int32_t* dids = (std::int32_t*)DevAlloc(slots * 4);
  std::uint32_t* dcounts = (std::uint32_t*)DevAlloc(E * 4);
  std::int32_t* bounds = (std::int32_t*)DevAlloc((E + 1) * 4);
  std::int32_t* cursors = (std::int32_t*)DevAlloc(E * 4);
  const std::size_t compact = qfn::RoutedCompactRows(slots, E);
  std::int32_t* rows_tok = (std::int32_t*)DevAlloc(compact * 4);
  std::int32_t* rows_slot = (std::int32_t*)DevAlloc(compact * 4);
  float* dx = (float*)DevAlloc(x.size() * 4);
  __half* dxh = (__half*)DevAlloc(Tp * H * 2);
  void* dgate = DevAlloc(gate_all.size());
  void* dup = DevAlloc(up_all.size());
  void* ddown = DevAlloc(down_all.size());
  float* gate_e = (float*)DevAlloc(slots_p * F * 4);
  __half* up_e = (__half*)DevAlloc(slots_p * F * 2);
  __half* down_e = (__half*)DevAlloc(slots_p * H * 2);
  float* dwts = (float*)DevAlloc(slots * 4);
  float* shared = (float*)DevAlloc(Tp * H * 4);
  float* dgate1 = (float*)DevAlloc(Tp * 4);
  float* out = (float*)DevAlloc(Tp * H * 4);
  ToDevice(dids, ids.data(), slots * 4);
  ToDevice(dcounts, counts.data(), E * 4);
  ToDevice(dx, x.data(), x.size() * 4);
  ToDevice(dgate, gate_all.data(), gate_all.size());
  ToDevice(dup, up_all.data(), up_all.size());
  ToDevice(ddown, down_all.data(), down_all.size());
  ToDevice(dwts, wts.data(), slots * 4);
  HIP_OK(hipMemset(shared, 0, Tp * H * 4));
  HIP_OK(hipMemset(dgate1, 0, Tp * 4));

  float compact_ms = TimeIt(stream, [&] {
    qfn::RoutedCompact(dids, dcounts, bounds, cursors, rows_tok, rows_slot,
                       std::uint32_t(T), U, E, stream);
  }, reps);
  Row(cell, "LFQ-F16", "compact", compact_ms, -1, -1, 1);
  qfn::RoutedCompact(dids, dcounts, bounds, cursors, rows_tok, rows_slot,
                     std::uint32_t(T), U, E, stream);
  float narrow_ms = TimeIt(stream, [&] {
    qfn::NarrowActivations(dx, dxh, false, T * H, stream);
  }, reps);
  Row(cell, "LFQ-F16", "narrow", narrow_ms, -1,
      double(T) * double(H) * 4.0 / narrow_ms / 1e6, 1);
  qfn::NarrowActivations(dx, dxh, false, T * H, stream);
  HIP_OK(hipDeviceSynchronize());

  const std::uint32_t tile_rows = slots >= std::size_t(16) * E ? 48 : 16;
  std::vector<std::int32_t> tiles;
  tiles.reserve(slots / 8 + 3 * E + 8);
  for (std::uint32_t e = 0; e < E; ++e) {
    const std::uint32_t padded = (counts[e] + 15) / 16 * 16;
    for (std::uint32_t j = 0; j < (padded + tile_rows - 1) / tile_rows; ++j)
      tiles.push_back(std::int32_t(e | (j << 16)));
  }
  const std::uint32_t n48 = std::uint32_t(tiles.size());
  std::uint32_t pair_off = 0, pair_tiles = 0, pair_rows = 64;
  std::uint32_t n64 = 0;
  if (T >= 1024 && tile_rows == 48) {
    for (std::uint32_t e = 0; e < E; ++e) {
      const std::uint32_t padded = (counts[e] + 15) / 16 * 16;
      for (std::uint32_t j = 0; j < (padded + 63) / 64; ++j)
        tiles.push_back(std::int32_t(e | (j << 16)));
    }
    n64 = std::uint32_t(tiles.size()) - n48;
    pair_off = n48;
    pair_tiles = n64;
    std::uint32_t t128 = 0;
    for (std::uint32_t e = 0; e < E; ++e) t128 += (counts[e] + 127) / 128;
    if (t128 * 4 <= n64 * 3) {
      pair_rows = 128;
      pair_off = std::uint32_t(tiles.size());
      pair_tiles = t128;
      for (std::uint32_t e = 0; e < E; ++e)
        for (std::uint32_t j = 0; j < (counts[e] + 127) / 128; ++j)
          tiles.push_back(std::int32_t(e | (j << 16)));
    }
  }
  std::int32_t* dtiles = (std::int32_t*)DevAlloc(tiles.size() * 4);
  ToDevice(dtiles, tiles.data(), tiles.size() * 4);

  float gu_ms = 0;
  bool gu_ok = true;
  if (T >= 1024 && tile_rows == 48) {
    gu_ms = TimeIt(stream, [&] {
      gu_ok = qfn::RoutedGatedF16Gemm(dgate, dup, kFP4, dxh,
                                     dtiles + pair_off, pair_tiles, pair_rows,
                                     bounds, rows_tok, rows_slot, up_e, F, H,
                                     stream);
    }, reps);
    char gnote[64];
    if (gu_ok)
      snprintf(gnote, sizeof(gnote), "tile%u tiles=%u", pair_rows, pair_tiles);
    else
      snprintf(gnote, sizeof(gnote), "gated-failed-or-no-LFQ");
    Row(cell, "LFQ-F16", "gated", gu_ms,
        gu_ok ? (2.0 * double(slots) * 2.0 * double(F) * double(H) / gu_ms /
                 1e9)
              : -1,
        -1, gu_ok ? 1 : 0, gnote);
  } else {
    float g_ms = TimeIt(stream, [&] {
      gu_ok = qfn::RoutedF16Gemm(dgate, kFP4, dxh, dtiles, n48, tile_rows,
                                bounds, rows_tok, rows_slot, nullptr, gate_e,
                                nullptr, F, H, stream) &&
               gu_ok;
    }, reps);
    float u_ms = TimeIt(stream, [&] {
      gu_ok = qfn::RoutedF16Gemm(dup, kFP4, dxh, dtiles, n48, tile_rows,
                                bounds, rows_tok, rows_slot, gate_e, nullptr,
                                up_e, F, H, stream) &&
               gu_ok;
    }, reps);
    gu_ms = g_ms + u_ms;
    char tbuf[48];
    snprintf(tbuf, sizeof(tbuf), "tile%u tiles=%u", tile_rows, n48);
    Row(cell, "LFQ-F16", "gate", g_ms, -1, -1, gu_ok ? 1 : 0,
        gu_ok ? tbuf : "gate-failed-or-no-LFQ");
    Row(cell, "LFQ-F16", "up", u_ms, -1, -1, gu_ok ? 1 : 0,
        gu_ok ? tbuf : "up-failed-or-no-LFQ");
  }
  const bool wide_down = tile_rows == 48 && n64 != 0 && n64 * 4 <= n48 * 3;
  float d_ms = 0;
  bool d_ok = false;
  if (gu_ok) {
    d_ms = TimeIt(stream, [&] {
      d_ok = qfn::RoutedF16Gemm(ddown, kFP4, up_e,
                               dtiles + (wide_down ? n48 : 0),
                               wide_down ? n64 : n48,
                               wide_down ? 64 : tile_rows, bounds, rows_slot,
                               rows_slot, nullptr, nullptr, down_e, H, F,
                               stream);
    }, reps);
  }
  char dnote[64];
  if (d_ok)
    snprintf(dnote, sizeof(dnote), "tile%u tiles=%u",
             wide_down ? 64 : tile_rows, wide_down ? n64 : n48);
  else
    snprintf(dnote, sizeof(dnote), "down-failed-or-no-LFQ");
  Row(cell, "LFQ-F16", "down", d_ms, -1, -1, d_ok ? 1 : 0, dnote);
  float epi_ms = 0;
  if (d_ok) {
    epi_ms = TimeIt(stream, [&] {
      qfn::MoeEpilogueVec4F16(down_e, dwts, shared, dgate1, 1, out,
                             std::uint32_t(T), U, H, stream);
    }, reps);
    Row(cell, "LFQ-F16", "epilogue", epi_ms, -1, -1,
        FiniteProbe(out, T * H));
    const float total = compact_ms + narrow_ms + gu_ms + d_ms + epi_ms;
    const std::size_t gu_tiles =
        (T >= 1024 && tile_rows == 48) ? pair_tiles : n48;
    const std::size_t dn_tiles = wide_down ? n64 : n48;
    const double wmoved = double(gu_tiles) * double(gate_eb) * 2.0 +
                          double(dn_tiles) * double(down_eb);
    char tnote[128];
    snprintf(tnote, sizeof(tnote), "Wmoved=%.3fGB sweep=%.3fGB",
             wmoved / 1e9, wbytes / 1e9);
    Row(cell, "LFQ-F16", "total", total, flops / total / 1e9,
        wbytes / total / 1e6, 1, tnote);
  } else if (!gu_ok) {
    printf("# LFQ-F16 unsupported in this build (needs the LFQ stack tree)\n");
  }
  HIP_OK(hipFree(dids));
  HIP_OK(hipFree(dcounts));
  HIP_OK(hipFree(bounds));
  HIP_OK(hipFree(cursors));
  HIP_OK(hipFree(rows_tok));
  HIP_OK(hipFree(rows_slot));
  HIP_OK(hipFree(dx));
  HIP_OK(hipFree(dxh));
  HIP_OK(hipFree(dgate));
  HIP_OK(hipFree(dup));
  HIP_OK(hipFree(ddown));
  HIP_OK(hipFree(gate_e));
  HIP_OK(hipFree(up_e));
  HIP_OK(hipFree(down_e));
  HIP_OK(hipFree(dwts));
  HIP_OK(hipFree(shared));
  HIP_OK(hipFree(dgate1));
  HIP_OK(hipFree(out));
  HIP_OK(hipFree(dtiles));
}

static void Usage(const char* a0) {
  fprintf(stderr,
          "usage: %s [tokens=2048] [reps=20] [--csv] [--dense-only] "
          "[--routed-only] [--with-lfp] [--with-lfq]\n",
          a0);
}

int main(int argc, char** argv) {
  std::size_t T = 2048;
  int reps = 20;
  bool dense = true, routed = true, lfp = false, lfq = false;
  int pos = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--csv")
      g_csv = true;
    else if (a == "--dense-only") {
      dense = true;
      routed = false;
    } else if (a == "--routed-only") {
      dense = false;
      routed = true;
    } else if (a == "--with-lfp")
      lfp = true;
    else if (a == "--with-lfq")
      lfq = true;
    else if (a == "--help" || a == "-h") {
      Usage(argv[0]);
      return 0;
    } else if (a[0] == '-') {
      Usage(argv[0]);
      return 64;
    } else if (pos == 0) {
      T = std::size_t(std::stoul(a));
      pos++;
    } else if (pos == 1) {
      reps = std::stoi(a);
      pos++;
    } else {
      Usage(argv[0]);
      return 64;
    }
  }
  if (T == 0 || reps <= 0) {
    Usage(argv[0]);
    return 64;
  }
  hipDeviceProp_t prop;
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  std::mt19937 rng(0x1F7u);
  hipStream_t stream;
  HIP_OK(hipStreamCreate(&stream));
  if (g_csv) {
    printf("cell,route,stage,ms,tflops,gbps,finite,note\n");
  } else {
    printf("# qfn_gemm_bench tokens=%zu reps=%d device=%s\n", T, reps,
           prop.name);
  }
  if (dense) BenchDense(T, reps, stream, rng);
  if (routed) BenchRouted(T, reps, stream, rng);
  if (lfp) BenchDenseLFP(T, reps, stream, rng);
  if (lfq) BenchRoutedLFQ(T, reps, stream, rng);
  HIP_OK(hipStreamDestroy(stream));
  return 0;
}

