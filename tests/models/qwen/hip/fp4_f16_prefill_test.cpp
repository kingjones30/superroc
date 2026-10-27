// tests/models/qwen/hip/fp4_f16_prefill_test.cpp — SQ1 SPEC1 item 6.
//
// The ported FP4->F16 prefill kernels vs a CPU reference:
// (a) occupancy: every ported instantiation holds >= 2 waves/SIMD;
// (b) Fp4DenseF16Gemm vs F64 on small shapes incl. (m, k) boundary cases,
//     both types, plus bitwise determinism and an independent-kernel
//     cross-check (DenseF16Gemm over the same values as Q8_0);
// (c) Fp4SplitF16Gemm / Fp4SsmF16Gemm at exact geometry vs F64;
// (d) Fp4Int8Gemm vs F64 on small shapes (tiled Q8 from QuantizeQ8Tiled,
//     the producer the route consumes in production).
// FP4 decode on the host uses the header's scalar twins, pinned to the
// superroc_fp4 CPU oracle over all 256 scales x 16 codes x both types first.
// Run under gpu-run. Links superroc_core + superroc_qwen38_flash_next.

#include "src/core/rocm_fp4/fp4_f16_prefill.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/rocm_fp4/superroc_fp4_quant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace f16 = superroc::models::qwen38_flash_next::rocm;
namespace fp4 = f16::fp4_f16;

static int g_fail = 0;

static void Check(bool ok, const std::string& what) {
  if (!ok) {
    printf("FAIL %s\n", what.c_str());
    ++g_fail;
  }
}

#define HIP_OK(call)                                                        \
  do {                                                                      \
    const hipError_t e_hip_ok_ = (call);                                    \
    if (e_hip_ok_ != hipSuccess) {                                          \
      printf("HIP FATAL %s:%d: %s failed: %s\n", __FILE__, __LINE__, #call, \
             hipGetErrorString(e_hip_ok_));                                 \
      fflush(stdout);                                                       \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

static void* DevAlloc(std::size_t bytes) {
  void* p = nullptr;
  HIP_OK(hipMalloc(&p, bytes ? bytes : 1));
  return p;
}

static std::uint64_t g_rng = 0x5EED1234ABCDEFULL;
static std::uint64_t Next() {
  g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return g_rng >> 11;
}

// Round-to-nearest-even float -> f16 bits. Matches __float2half_rn on the
// finite normal range the test uses (|x| in [1e-3, 8]); subnormals and
// infinities never occur here.
static std::uint16_t F32ToF16(float f) {
  std::uint32_t u;
  memcpy(&u, &f, sizeof(u));
  const std::uint32_t sign = (u >> 16) & 0x8000u;
  const int exp = int((u >> 23) & 0xFFu) - 112;
  const std::uint32_t mant = u & 0x7FFFFFu;
  if (exp <= 0)
    return std::uint16_t(sign);
  if (exp >= 31)
    return std::uint16_t(sign | 0x7BFFu);
  return std::uint16_t(sign | (std::uint32_t(exp) << 10) | (mant >> 13));
}

static float F16ToF32(std::uint16_t h) {
  const std::uint32_t sign = (h & 0x8000u) << 16;
  const unsigned exp = (h >> 10) & 0x1Fu;
  const unsigned mant = h & 0x3FFu;
  std::uint32_t u;
  if (exp == 0) {
    if (mant == 0) {
      u = sign;
    } else {
      // Subnormal: normalize (exact; only the oracle pin can hit these).
      unsigned m = mant;
      int e = -14;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        --e;
      }
      u = sign | (std::uint32_t(e + 127) << 23) | ((m & 0x3FFu) << 13);
    }
  } else if (exp == 31) {
    u = sign | 0x7F800000u | (mant << 13);
  } else {
    u = sign | (std::uint32_t(exp + 112) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

// Exact host FP4 decode through the header's scalar twins.
static float Fp4Host(std::uint8_t nib, std::uint8_t scale) {
  const int code = int(fp4::Fp4CodeBiased(nib)) - 16;
  const float s = F16ToF32(fp4::Fp4Ue4m3ToHalfU16(scale));
  float v = float(code) * s;
  return v == 0.0F ? 0.0F : v;  // normalize -0 (code 8), like the oracle
}

static void FillFP4(std::vector<std::uint8_t>& buf, std::size_t rows,
                    std::size_t k, bool fast) {
  const std::size_t nb = k / 32;
  const std::size_t bb = fast ? 17 : 18;
  buf.resize(rows * nb * bb);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t n = 0; n < nb; ++n) {
      auto* blk = &buf[(r * nb + n) * bb];
      for (int j = 0; j < 16; ++j)
        blk[j] = std::uint8_t(Next());
      // Cover subnormal/zero/overflow scale paths, not just 0x1C.
      const std::uint64_t pick = Next() % 8;
      blk[16] = pick == 0   ? 0x00
                : pick == 1 ? 0x07
                : pick == 2 ? 0x7F
                : pick == 3 ? 0xFF
                            : std::uint8_t(0x10 + Next() % 0x50);
      if (!fast)
        blk[17] = std::uint8_t(Next());
    }
}

static float Fp4WeightAt(const std::vector<std::uint8_t>& buf, std::size_t k,
                         bool fast, std::size_t r, std::size_t c) {
  const std::size_t nb = k / 32;
  const std::size_t bb = fast ? 17 : 18;
  const std::uint8_t* blk = &buf[(r * nb + c / 32) * bb];
  const std::size_t j = c % 32;
  const std::uint8_t byte = blk[(j % 16)];
  const std::uint8_t nib = j < 16 ? (byte & 0xF) : (byte >> 4);
  const std::uint8_t s = fast ? blk[16] : (j < 16 ? blk[16] : blk[17]);
  return Fp4Host(nib, s);
}

static void FillAct(std::vector<float>& v, float amp) {
  for (float& x : v) {
    // Uniform in [amp/8, amp], random sign: no subnormals, no denormal risk.
    float u = amp * (0.125f + 0.875f * float(Next() % 1000000) / 1000000.0f);
    x = (Next() & 1) ? u : -u;
  }
}

// (pin) header scalar twins vs the superroc_fp4 CPU oracle, all scales x codes.
static void TestOraclePin() {
  int diffs = 0;
  for (unsigned s = 0; s < 256; ++s) {
    std::uint8_t blk[18];
    memset(blk, 0, sizeof(blk));
    blk[16] = std::uint8_t(s);
    blk[17] = std::uint8_t(s);
    for (unsigned n = 0; n < 16; ++n) {
      blk[0] = std::uint8_t(n);
      const float o0 = superroc_fp4_elem(100, blk, 0);
      const float o1 = superroc_fp4_elem(101, blk, 0);
      const float h = Fp4Host(std::uint8_t(n), std::uint8_t(s));
      // Bitwise except the sign of zero (code 8), per the header contract.
      if (o0 != h || o1 != h) {
        if (!(o0 == 0.0F && h == 0.0F))
          ++diffs;
      }
    }
  }
  printf("oracle-pin: diffs=%d (256 scales x 16 codes x 2 types)\n", diffs);
  Check(diffs == 0, "oracle pin vs superroc_fp4_elem");
}

// Narrow F32 activations to F16 on the host (bitwise == device RN here).
static void NarrowHost(const std::vector<float>& x,
                       std::vector<std::uint16_t>& h) {
  h.resize(x.size());
  for (std::size_t i = 0; i < x.size(); ++i)
    h[i] = F32ToF16(x[i]);
}

struct DenseCase {
  std::size_t T, m, k;
  bool fast;
};

static void TestDenseOne(const DenseCase& c, hipStream_t stream) {
  char name[96];
  snprintf(name, sizeof(name), "dense T=%zu m=%zu k=%zu %s", c.T, c.m, c.k,
           c.fast ? "101" : "100");
  std::vector<std::uint8_t> w;
  FillFP4(w, c.m, c.k, c.fast);
  const std::size_t Tp = (c.T + 127) / 128 * 128;
  std::vector<float> x(Tp * c.k, 0.0f);
  {
    std::vector<float> live(c.T * c.k);
    FillAct(live, 0.5f);
    for (std::size_t t = 0; t < c.T; ++t)
      memcpy(&x[t * c.k], &live[t * c.k], c.k * 4);
  }
  std::vector<std::uint16_t> xh;
  NarrowHost(x, xh);

  void* dw = DevAlloc(w.size());
  void* dx = DevAlloc(xh.size() * 2);
  float* out = (float*)DevAlloc(Tp * c.m * 4);
  HIP_OK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dx, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
  bool ok = f16::Fp4DenseF16Gemm(dw, (const __half*)dx, out, c.T, c.m, c.k,
                                 c.fast, stream);
  HIP_OK(hipGetLastError());
  HIP_OK(hipStreamSynchronize(stream));
  if (!ok) {
    printf("FAIL %s: launcher declined\n", name);
    ++g_fail;
    return;
  }
  std::vector<float> got(c.T * c.m);
  HIP_OK(hipMemcpy(got.data(), out, got.size() * 4, hipMemcpyDeviceToHost));
  // Outputs to verify: all when cheap, else a 2048-output random sample
  // (plus a full finite scan either way).
  for (float v : got) {
    if (!std::isfinite(v)) {
      printf("FAIL %s: non-finite output\n", name);
      ++g_fail;
      return;
    }
  }
  std::vector<std::size_t> idx;
  if (c.T * c.m <= 65536) {
    for (std::size_t i = 0; i < c.T * c.m; ++i)
      idx.push_back(i);
  } else {
    for (int i = 0; i < 2048; ++i)
      idx.push_back(Next() % (c.T * c.m));
  }
  // Bound: F16 input rounding (2^-11 each) + F32 K-accumulation (k * 2^-24),
  // x4 headroom, + absolute floor. All terms are first-principles epsilons.
  const double tol_coef = 4.0 * (0.5 / 1024 + c.k / 16777216.0);
  double max_err = 0, max_ref = 1e-30;
  for (std::size_t i : idx) {
    const std::size_t t = i / c.m, r = i % c.m;
    double acc = 0.0;
    for (std::size_t cc = 0; cc < c.k; ++cc)
      acc += double(Fp4WeightAt(w, c.k, c.fast, r, cc)) *
             double(F16ToF32(xh[t * c.k + cc]));
    max_ref = std::max(max_ref, fabs(acc));
    max_err = std::max(max_err, fabs(double(got[i]) - acc));
  }
  const double tol = max_ref * tol_coef + 1e-5;
  printf("%s: checked=%zu max_ref=%.6g max_err=%.3g tol=%.3g %s\n", name,
         idx.size(), max_ref, max_err, tol, max_err <= tol ? "ok" : "OVER");
  Check(max_err <= tol, std::string(name) + " vs F64");
  // Bitwise determinism: run twice, memcmp.
  float* out2 = (float*)DevAlloc(Tp * c.m * 4);
  HIP_OK(hipMemset(out2, 0xA5, Tp * c.m * 4));
  ok = f16::Fp4DenseF16Gemm(dw, (const __half*)dx, out2, c.T, c.m, c.k, c.fast,
                            stream);
  HIP_OK(hipGetLastError());
  HIP_OK(hipStreamSynchronize(stream));
  std::vector<float> got2(c.T * c.m);
  HIP_OK(hipMemcpy(got2.data(), out2, got2.size() * 4, hipMemcpyDeviceToHost));
  Check(ok && memcmp(got.data(), got2.data(), got.size() * 4) == 0,
        std::string(name) + " determinism");
  HIP_OK(hipFree(dw));
  HIP_OK(hipFree(dx));
  HIP_OK(hipFree(out));
  HIP_OK(hipFree(out2));
}

static void TestDense(hipStream_t stream) {
  // (m, k) boundary cases around the launcher's tile cuts (m<=1024 narrow,
  // m>=2560 + k in {320,640,2560,6144} wide-81, batch>=96) plus small ms.
  const std::size_t ms[] = {1,  31,   32,   33,   48,   63,   64,
                            65, 1023, 1024, 1025, 2559, 2560, 2561};
  const std::size_t ks[] = {32, 64, 96, 320, 640};
  const std::size_t Ts[] = {1, 17, 95, 96, 97, 128};
  for (bool fast : {true, false}) {
    for (std::size_t m : ms)
      for (std::size_t k : ks)
        for (std::size_t T : Ts) {
          TestDenseOne({T, m, k, fast}, stream);
        }
    // The production-width branches (one T each to bound time).
    TestDenseOne({128, 2560, 2560, fast}, stream);
    TestDenseOne({128, 2560, 6144, fast}, stream);
    TestDenseOne({128, 12288, 5120, fast}, stream);
    TestDenseOne({100, 48, 5120, fast}, stream);
    TestDenseOne({17, 17408, 5120, fast}, stream);
  }
}

static void TestSplitOne(std::size_t T, bool fast, hipStream_t stream) {
  char name[64];
  snprintf(name, sizeof(name), "split T=%zu %s", T, fast ? "101" : "100");
  const std::size_t m = 12288, k = 2560, width = m / 2;
  std::vector<std::uint8_t> w;
  FillFP4(w, m, k, fast);
  const std::size_t Tp = (T + 127) / 128 * 128;
  std::vector<float> x(Tp * k, 0.0f);
  {
    std::vector<float> live(T * k);
    FillAct(live, 0.25f);
    for (std::size_t t = 0; t < T; ++t)
      memcpy(&x[t * k], &live[t * k], k * 4);
  }
  std::vector<std::uint16_t> xh;
  NarrowHost(x, xh);
  void* dw = DevAlloc(w.size());
  void* dx = DevAlloc(xh.size() * 2);
  float* dq = (float*)DevAlloc(Tp * width * 4);
  float* dg = (float*)DevAlloc(Tp * width * 4);
  HIP_OK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dx, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
  bool ok = f16::Fp4SplitF16Gemm(dw, (const __half*)dx, dq, dg, T, m, k, fast,
                                 stream);
  HIP_OK(hipGetLastError());
  HIP_OK(hipStreamSynchronize(stream));
  Check(ok, std::string(name) + " launcher accepted");
  if (!ok)
    return;
  std::vector<float> gotq(T * width), gotg(T * width);
  HIP_OK(hipMemcpy(gotq.data(), dq, gotq.size() * 4, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(gotg.data(), dg, gotg.size() * 4, hipMemcpyDeviceToHost));
  // 2048-output sample: full F64 over 12288x2560 is too slow.
  const double tol_coef = 4.0 * (0.5 / 1024 + k / 16777216.0);
  double max_err = 0, max_ref = 1e-30;
  for (int i = 0; i < 2048; ++i) {
    const std::size_t t = Next() % T, r = Next() % m;
    double acc = 0.0;
    for (std::size_t c = 0; c < k; ++c)
      acc += double(Fp4WeightAt(w, k, fast, r, c)) *
             double(F16ToF32(xh[t * k + c]));
    const std::size_t h = r / 512, j = r % 512;
    const std::size_t off = t * width + h * 256 + (j < 256 ? j : j - 256);
    const double got = double((j < 256 ? gotq : gotg)[off]);
    max_ref = std::max(max_ref, fabs(acc));
    max_err = std::max(max_err, fabs(got - acc));
  }
  const double tol = max_ref * tol_coef + 1e-5;
  printf("%s: max_ref=%.6g max_err=%.3g tol=%.3g %s\n", name, max_ref, max_err,
         tol, max_err <= tol ? "ok" : "OVER");
  Check(max_err <= tol, std::string(name) + " vs F64");
  HIP_OK(hipFree(dw));
  HIP_OK(hipFree(dx));
  HIP_OK(hipFree(dq));
  HIP_OK(hipFree(dg));
}

static double SiluRef(double x) {
  return x / (1.0 + exp(-x));
}

static void TestSsm(hipStream_t stream) {
  const char* name = "ssm T=1024";
  const std::size_t T = 1024, m = 16384, k = 2560, ch = 10240;
  std::vector<std::uint8_t> w0, w1;
  FillFP4(w0, 10240, k, false);     // qkv rows are type 100 ...
  FillFP4(w1, m - 10240, k, true);  // ... gate rows type 101
  const std::size_t Tp = (T + 127) / 128 * 128;
  std::vector<float> x(Tp * k, 0.0f);
  {
    std::vector<float> live(T * k);
    FillAct(live, 0.25f);
    for (std::size_t t = 0; t < T; ++t)
      memcpy(&x[t * k], &live[t * k], k * 4);
  }
  std::vector<std::uint16_t> xh;
  NarrowHost(x, xh);
  std::vector<float> conv_w(ch * 4), hist(3 * ch, 0.0f);
  FillAct(conv_w, 0.25f);
  void* dw0 = DevAlloc(w0.size());
  void* dw1 = DevAlloc(w1.size());
  void* dx = DevAlloc(xh.size() * 2);
  float* dconv = (float*)DevAlloc(conv_w.size() * 4);
  float* dhist = (float*)DevAlloc(hist.size() * 4);
  float* dqkvz = (float*)DevAlloc(Tp * m * 4);
  float* dconv_out = (float*)DevAlloc(Tp * ch * 4);
  HIP_OK(hipMemcpy(dw0, w0.data(), w0.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dw1, w1.data(), w1.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dx, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dconv, conv_w.data(), conv_w.size() * 4,
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dhist, hist.data(), hist.size() * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemset(dqkvz, 0xA5, Tp * m * 4));
  HIP_OK(hipMemset(dconv_out, 0xA5, Tp * ch * 4));
  bool ok = f16::Fp4SsmF16Gemm(dw0, dw1, (const __half*)dx, dconv, dhist, dqkvz,
                               dconv_out, (std::uint32_t)T, (std::uint32_t)m,
                               (std::uint32_t)k, (std::uint32_t)ch, 4, stream);
  HIP_OK(hipGetLastError());
  HIP_OK(hipStreamSynchronize(stream));
  Check(ok, "ssm launcher accepted");
  if (!ok)
    return;
  std::vector<float> gotq(T * m), gotc(T * ch);
  HIP_OK(hipMemcpy(gotq.data(), dqkvz, gotq.size() * 4, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(gotc.data(), dconv_out, gotc.size() * 4,
                   hipMemcpyDeviceToHost));
  // Full convolved check needs full raw rows: compute raw for the sampled
  // tokens' conv windows (1024 tokens x 10240 ch x 2560 k sampled sparsely).
  const double tol_coef = 4.0 * (0.5 / 1024 + k / 16777216.0);
  double gain = 1.0;
  {
    double g = 0;
    for (float v : conv_w)
      g = std::max(g, fabs(double(v)));
    gain = 1.0 + 4.0 * g;
  }
  double max_err = 0, max_ref = 1e-30;
  for (int i = 0; i < 512; ++i) {
    const std::size_t t = 3 + Next() % (T - 3);
    const std::size_t r = Next() % ch;
    // Raw rows t-3..t for channel r (type-100 qkv tensor).
    double raw[4];
    for (int j = 0; j < 4; ++j) {
      double acc = 0.0;
      for (std::size_t c = 0; c < k; ++c)
        acc += double(Fp4WeightAt(w0, k, false, r, c)) *
               double(F16ToF32(xh[(t - 3 + j) * k + c]));
      raw[j] = acc;
    }
    double conv = 0.0;
    for (int j = 0; j < 4; ++j)
      conv += double(conv_w[r * 4 + j]) * raw[j];
    const double ref = SiluRef(conv);
    max_ref = std::max(max_ref, fabs(ref));
    max_err = std::max(max_err, fabs(double(gotc[t * ch + r]) - ref));
  }
  const double tol = max_ref * tol_coef * gain + 1e-4;
  printf("%s: convolved max_ref=%.6g max_err=%.3g tol=%.3g %s\n", name, max_ref,
         max_err, tol, max_err <= tol ? "ok" : "OVER");
  Check(max_err <= tol, "ssm convolved vs F64");
  // Gate rows of qkvz are always written: spot-check raw GEMM there.
  max_err = 0;
  max_ref = 1e-30;
  for (int i = 0; i < 512; ++i) {
    const std::size_t t = Next() % T, r = 10240 + Next() % (m - 10240);
    double acc = 0.0;
    for (std::size_t c = 0; c < k; ++c)
      acc += double(Fp4WeightAt(w1, k, true, r - 10240, c)) *
             double(F16ToF32(xh[t * k + c]));
    max_ref = std::max(max_ref, fabs(acc));
    max_err = std::max(max_err, fabs(double(gotq[t * m + r]) - acc));
  }
  const double tol2 = max_ref * tol_coef + 1e-5;
  printf("%s: qkvz-gate max_ref=%.6g max_err=%.3g tol=%.3g %s\n", name, max_ref,
         max_err, tol2, max_err <= tol2 ? "ok" : "OVER");
  Check(max_err <= tol2, "ssm qkvz gate rows vs F64");
  HIP_OK(hipFree(dw0));
  HIP_OK(hipFree(dw1));
  HIP_OK(hipFree(dx));
  HIP_OK(hipFree(dconv));
  HIP_OK(hipFree(dhist));
  HIP_OK(hipFree(dqkvz));
  HIP_OK(hipFree(dconv_out));
}

// Tiled-Q8 host parse, copied from Flash-Next's fp4_f16_test.cpp
// (TiledQ8At; mirrors QuantizeQ8TiledKernel's layout: 576-byte tiles, two
// 256-byte code halves of 16 tokens x 16 codes, 16 float scales at byte 512).
static float TiledQ8At(const std::uint8_t* tiled, std::size_t k, std::size_t t,
                       std::size_t i) {
  const std::size_t nb = k / 32;
  const std::size_t kb = i / 32, lane = i % 32;
  const std::size_t half = lane >> 4, pos = lane & 15;
  const std::size_t tt = t / 16, tl = t % 16;
  const std::uint8_t* tile = tiled + ((tt * nb + kb) * 576);
  const auto q = static_cast<std::int8_t>(tile[half * 256 + tl * 16 + pos]);
  float d = 0;
  __builtin_memcpy(&d, tile + 512 + tl * 4, 4);
  return q * d;
}

// Int8 vs F64, FN's Int8GemmCase approach: the reference decodes the
// DEVICE's quantized activations (read back, TiledQ8At), isolating kernel
// error from quantizer error. (Replicating the quantizer in F32 diverges
// at big k — tkA measured 2.9e-4 at k=10240 — while FN qualifies this
// kernel at kernel-rel < 2e-6 on (320,10240) with device readback.)
static void TestInt8One(std::size_t T, std::size_t m, std::size_t k,
                        hipStream_t stream) {
  char name[64];
  snprintf(name, sizeof(name), "int8 T=%zu m=%zu k=%zu", T, m, k);
  std::vector<std::uint8_t> w;
  FillFP4(w, m, k, true);
  const std::size_t Tp = (T + 127) / 128 * 128;
  std::vector<float> x(Tp * k, 0.0f);
  {
    std::vector<float> live(T * k);
    FillAct(live, 0.5f);
    for (std::size_t t = 0; t < T; ++t)
      memcpy(&x[t * k], &live[t * k], k * 4);
  }
  float* dx = (float*)DevAlloc(x.size() * 4);
  void* dw = DevAlloc(w.size());
  HIP_OK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
  void* dq8 = DevAlloc(f16::Q8TiledBytes(Tp, k));
  f16::QuantizeQ8Tiled(dx, dq8, T, k, stream);
  HIP_OK(hipGetLastError());
  float* out = (float*)DevAlloc(Tp * m * 4);
  bool ok = f16::Fp4Int8Gemm(dw, dq8, out, T, m, k, true, stream);
  HIP_OK(hipGetLastError());
  HIP_OK(hipStreamSynchronize(stream));
  Check(ok, std::string(name) + " launcher accepted");
  if (!ok)
    return;
  std::vector<float> got(T * m);
  HIP_OK(hipMemcpy(got.data(), out, got.size() * 4, hipMemcpyDeviceToHost));
  // Device-quantized activations, read back: the reference below is exact
  // for these q values, so the kernel bar stays at FN's 2e-6 relative.
  const std::size_t qb = f16::Q8TiledBytes(Tp, k);
  std::vector<std::uint8_t> q(qb);
  HIP_OK(hipMemcpy(q.data(), dq8, qb, hipMemcpyDeviceToHost));
  double max_err = 0, max_ref = 1e-30;
  const std::size_t nb = k / 32;
  for (std::size_t t = 0; t < T; ++t) {
    for (std::size_t r = 0; r < m; ++r) {
      double acc = 0.0;
      for (std::size_t i = 0; i < k; ++i) {
        const std::size_t b = i / 32, j = i % 32;
        const std::uint8_t* blk = &w[(r * nb + b) * 17];
        const std::uint8_t byte = blk[j % 16];
        const int nib = j < 16 ? (byte & 0xF) : (byte >> 4);
        const int code = int(fp4::Fp4CodeBiased(nib)) - 16;
        const float s = fp4::Fp4ScaleByteToFloat(blk[16]);
        acc += double(code) * double(s) * double(TiledQ8At(q.data(), k, t, i));
      }
      max_ref = std::max(max_ref, fabs(acc));
      max_err = std::max(max_err, fabs(double(got[t * m + r]) - acc));
    }
  }
  const double tol = max_ref * 2e-6 + 1e-6;
  printf("%s: max_ref=%.6g max_err=%.3g tol=%.3g %s\n", name, max_ref, max_err,
         tol, max_err <= tol ? "ok" : "OVER");
  Check(max_err <= tol, std::string(name) + " vs F64");
  HIP_OK(hipFree(dx));
  HIP_OK(hipFree(dw));
  HIP_OK(hipFree(dq8));
  HIP_OK(hipFree(out));
}

// Independent-kernel cross-check: the ported dense kernel vs DenseF16Gemm
// (Q8_0, still Flash-Next-resident) over the same logical values.
static void TestCrossQ8(hipStream_t stream) {
  const char* name = "cross-q8 T=128 m=512 k=1024";
  const std::size_t T = 128, m = 512, k = 1024;
  std::vector<std::uint8_t> wf4;
  FillFP4(wf4, m, k, true);
  // Same values as Q8_0: per-row absmax scale, round codes.
  std::vector<std::uint8_t> wq8(m * (k / 32) * 34);
  for (std::size_t r = 0; r < m; ++r)
    for (std::size_t b = 0; b < k / 32; ++b) {
      float mx = 0;
      for (int j = 0; j < 32; ++j)
        mx = std::max(mx, fabsf(Fp4WeightAt(wf4, k, true, r, b * 32 + j)));
      auto* blk = &wq8[(r * (k / 32) + b) * 34];
      const std::uint16_t dbits = F32ToF16(mx / 127.0f);
      memcpy(blk, &dbits, 2);
      for (int j = 0; j < 32; ++j) {
        const float v = Fp4WeightAt(wf4, k, true, r, b * 32 + j);
        blk[2 + j] = std::uint8_t(
            std::int8_t(roundf(v / (mx / 127.0f == 0 ? 1.0f : mx / 127.0f))));
      }
    }
  const std::size_t Tp = (T + 127) / 128 * 128;
  std::vector<float> x(Tp * k, 0.0f);
  {
    std::vector<float> live(T * k);
    FillAct(live, 0.5f);
    for (std::size_t t = 0; t < T; ++t)
      memcpy(&x[t * k], &live[t * k], k * 4);
  }
  std::vector<std::uint16_t> xh;
  NarrowHost(x, xh);
  void* dw4 = DevAlloc(wf4.size());
  void* dw8 = DevAlloc(wq8.size());
  void* dx = DevAlloc(xh.size() * 2);
  float* o4 = (float*)DevAlloc(Tp * m * 4);
  float* o8 = (float*)DevAlloc(Tp * m * 4);
  HIP_OK(hipMemcpy(dw4, wf4.data(), wf4.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dw8, wq8.data(), wq8.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dx, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
  bool ok4 =
      f16::Fp4DenseF16Gemm(dw4, (const __half*)dx, o4, T, m, k, true, stream);
  bool ok8 = f16::DenseF16Gemm(dw8, (const __half*)dx, o8, T, m, k, stream);
  HIP_OK(hipGetLastError());
  HIP_OK(hipStreamSynchronize(stream));
  Check(ok4 && ok8, "cross-q8 launchers accepted");
  if (!(ok4 && ok8))
    return;
  std::vector<float> g4(T * m), g8(T * m);
  HIP_OK(hipMemcpy(g4.data(), o4, g4.size() * 4, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(g8.data(), o8, g8.size() * 4, hipMemcpyDeviceToHost));
  // Bound: Q8_0 weight rounding (1/127 of row absmax each) dominates.
  double max_err = 0, max_ref = 1e-30;
  for (std::size_t i = 0; i < g4.size(); ++i) {
    max_ref = std::max(max_ref, fabs(double(g4[i])));
    max_err = std::max(max_err, fabs(double(g4[i]) - double(g8[i])));
  }
  const double tol = max_ref * 0.02 + 1e-4;
  printf("%s: max_ref=%.6g max_err=%.3g tol=%.3g %s\n", name, max_ref, max_err,
         tol, max_err <= tol ? "ok" : "OVER");
  Check(max_err <= tol, "cross-q8 ported-vs-Q8_0");
  HIP_OK(hipFree(dw4));
  HIP_OK(hipFree(dw8));
  HIP_OK(hipFree(dx));
  HIP_OK(hipFree(o4));
  HIP_OK(hipFree(o8));
}

static void TestOccupancy() {
  char report[4096];
  const int min_waves =
      f16::Fp4F16PrefillMinWavesPerSimd(report, sizeof(report));
  printf("occupancy report:\n%s", report);
  printf("min waves/SIMD x100 = %d (bar >= 200)\n", min_waves);
  Check(min_waves >= 200, "occupancy >= 2 waves/SIMD every instantiation");
}

int main() {
  HIP_OK(hipSetDevice(0));
  hipStream_t stream = nullptr;
  HIP_OK(hipStreamCreate(&stream));
  TestOccupancy();
  TestOraclePin();
  TestDense(stream);
  TestSplitOne(96, true, stream);
  TestSplitOne(96, false, stream);
  TestSplitOne(128, true, stream);
  TestSsm(stream);
  TestInt8One(32, 256, 512, stream);
  TestInt8One(100, 640, 1024, stream);
  TestInt8One(128, 320, 10240, stream);
  // SQ1-diag (tkA 00:22Z: the (128,320,10240) cell OVERed 144x while small-k
  // cells pass with 8-16x margin — a cliff, not noise growth). Separate an
  // m=320 tail-tile effect from a big-k effect + locate the cliff.
  TestInt8One(128, 320, 1024, stream);
  TestInt8One(32, 256, 10240, stream);
  TestInt8One(128, 320, 4096, stream);
  TestCrossQ8(stream);
  HIP_OK(hipStreamDestroy(stream));
  printf(g_fail == 0 ? "FP4_F16_PREFILL_TEST PASS\n"
                     : "FP4_F16_PREFILL_TEST FAIL (%d)\n",
         g_fail);
  return g_fail == 0 ? 0 : 1;
}
