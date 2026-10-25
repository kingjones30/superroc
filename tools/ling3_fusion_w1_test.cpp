// ling3-fusion-w1-test -- RV25 wave 1 (F1..F5) unit gate. LF1, 2026-10-03.
//
// Wave 1 is backend-only: every candidate is a rewrite that the design claims is
// BIT-IDENTICAL, not merely close. This test proves the two that have arithmetic
// in them, byte for byte on the device, at Ling's real shapes:
//
//   F4  residual#2 + the next norm  ->  one add_rms_norm
//       add_rms_norm(a, b, w, y, rsum)  vs  residual_add(a, b, x) ; rms_norm(x, w, y)
//       at H = 2560, rows = 1. memcmp on BOTH outputs (y and rsum == x).
//
//   F3  the n_used == 1, w == 1.0 expert writes y directly from the down
//       projection with y[i] = 0.0f + acc, instead of
//       hipMemset(y, 0, ...) ; <gemv -> p> ; k_axpy(1.0f, p, y).
//       (a) the STORE EXPRESSION, on a crafted vector that contains -0.0 -- this
//           is the exactness trap: a raw `y[i] = p[i]` is NOT the same bytes;
//       (b) the real path end to end: ling3_dense_ffn vs ling3_dense_ffn_w1 on
//           synthetic ROCmFP4 weights at both Ling n_ff values, with xq_ready 0
//           (dense layer) and 1 (shared expert, quant already done).
//
// F2's claim (the shared expert's FFN_X quant rewrote identical bytes over
// themselves) is exercised by the xq_ready = 1 case of (b): skipping the quant
// must leave the output byte-identical. F1 and F5 have no arithmetic -- they are
// checked by the 64-token launch trace (REPORT8), not here.
//
// Run inside gpu-run. Small and fast: no model, no server, synthetic weights.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/models/ling3/kernels/ling3_gemv_q8.hpp"
#include "src/models/ling3/kernels/ling3_moe.hpp"
#include "src/models/ling3/kernels/ling3_ops.hpp"
#include "src/models/ling3/kernels/ling3_quant.hpp"

namespace {

std::uint64_t rng = 0x9E3779B97F4A7C15ULL;
std::uint32_t Next() {
  rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<std::uint32_t>(rng >> 32);
}

int failures = 0;
#define CHECK(cond, msg)                                                        \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (msg));       \
      ++failures;                                                                \
    }                                                                           \
  } while (0)

#define HIPCK(expr)                                                             \
  do {                                                                          \
    const hipError_t e_ = (expr);                                                \
    if (e_ != hipSuccess) {                                                      \
      std::fprintf(stderr, "FAIL %s:%d  HIP %s: %s\n", __FILE__, __LINE__,       \
                   #expr, hipGetErrorString(e_));                                \
      ++failures;                                                                \
      return;                                                                    \
    }                                                                           \
  } while (0)

// The unfused pair, verbatim: k_axpy's body is `y[i] += w * p[i]` and the
// memset that precedes it is a zero. Reproduced here as its own kernels so the
// test can put -0.0 through the EXACT expression the old path used.
__global__ void t_axpy(float w, const float* p, float* y, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  y[i] += w * p[i];
}
__global__ void t_zp(const float* p, float* y, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  y[i] = 0.0f + p[i];   // RV25 F3's store
}
__global__ void t_raw(const float* p, float* y, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  y[i] = p[i];          // the WRONG store -- must differ on -0.0
}

void FillRandom(std::vector<float>& v, float amp) {
  for (auto& x : v) {
    x = -amp + 2.f * amp * (float)Next() / 4294967296.0F;
  }
}

std::vector<std::uint8_t> MakeWeights(int qtype, int k, int n) {
  const std::size_t rb = ling3_row_bytes(qtype, k);
  std::vector<std::uint8_t> w(n * rb);
  const int nb = k / 32;
  const std::size_t tsz = ling3_type_size(qtype);
  for (int r = 0; r < n; ++r) {
    std::uint8_t* row = w.data() + (std::size_t)r * rb;
    for (int b = 0; b < nb; ++b) {
      std::uint8_t* blk = row + (std::size_t)b * tsz;
      for (int i = 0; i < 16; ++i) blk[i] = (std::uint8_t)(Next() & 0xFF);
      blk[16] = (std::uint8_t)(Next() % 256);
      if (qtype == LING3_Q4_0_ROCMFP4) blk[17] = (std::uint8_t)(Next() % 256);
    }
  }
  return w;
}

float* DevFloats(const std::vector<float>& h) {
  float* d = nullptr;
  if (hipMalloc(&d, h.size() * sizeof(float)) != hipSuccess) return nullptr;
  if (hipMemcpy(d, h.data(), h.size() * sizeof(float), hipMemcpyHostToDevice) !=
      hipSuccess) {
    hipFree(d);
    return nullptr;
  }
  return d;
}

std::uint32_t Bits(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

bool SameBytes(const float* a, const float* b, std::size_t n) {
  return std::memcmp(a, b, n * sizeof(float)) == 0;
}

// Report the first differing element so a failure is diagnosable at a glance.
void ReportDiff(const char* what, const float* a, const float* b, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
      std::fprintf(stderr,
                   "  %s first diff at [%zu]: %a (0x%08x) vs %a (0x%08x)\n", what,
                   i, (double)a[i], Bits(a[i]), (double)b[i], Bits(b[i]));
      return;
    }
  }
}

// ---------------------------------------------------------------------------
// F4: residual#2 + next norm fused into one add_rms_norm.
// ---------------------------------------------------------------------------
void TestF4(int H, int cases) {
  std::printf("== F4: add_rms_norm vs residual_add + rms_norm  H=%d rows=1\n", H);
  const int n = H;
  for (int c = 0; c < cases; ++c) {
    std::vector<float> ha(n), hb(n), hw(n);
    FillRandom(ha, 3.f);
    FillRandom(hb, 3.f);
    FillRandom(hw, 1.5f);
    if (c == 0) {
      // Crafted: exact zeros, both signs, a couple of exact cancellations.
      for (int i = 0; i < n; i += 97) {
        ha[i] = 0.0f;
        hb[i] = -0.0f;
      }
      for (int i = 3; i < n; i += 211) {
        ha[i] = 1.25f;
        hb[i] = -1.25f;   // sums to +0.0
      }
      ha[0] = -0.0f;
      hb[0] = -0.0f;      // (-0) + (-0) = -0
      hw[0] = 0.0f;
    }

    float *da = DevFloats(ha), *db = DevFloats(hb), *dw = DevFloats(hw);
    float *d_xref = nullptr, *d_yref = nullptr, *d_yfus = nullptr, *d_rfus = nullptr;
    HIPCK(hipMalloc(&d_xref, n * sizeof(float)));
    HIPCK(hipMalloc(&d_yref, n * sizeof(float)));
    HIPCK(hipMalloc(&d_yfus, n * sizeof(float)));
    HIPCK(hipMalloc(&d_rfus, n * sizeof(float)));
    if (!da || !db || !dw) {
      std::fprintf(stderr, "FAIL F4: device alloc\n");
      ++failures;
      return;
    }

    // reference: residual_add then rms_norm, exactly the two kernels F4 replaces
    ::ling3::ops_residual_add(da, db, d_xref, n, 1, nullptr);
    ::ling3::ops_rms_norm(d_xref, dw, d_yref, n, 1, ::ling3::LING3_RMS_EPS, nullptr);
    // fused
    ::ling3::ops_add_rms_norm(da, db, dw, d_yfus, d_rfus, n, 1,
                              ::ling3::LING3_RMS_EPS, nullptr);
    HIPCK(hipDeviceSynchronize());

    std::vector<float> xref(n), yref(n), yfus(n), rfus(n);
    HIPCK(hipMemcpy(xref.data(), d_xref, n * sizeof(float), hipMemcpyDeviceToHost));
    HIPCK(hipMemcpy(yref.data(), d_yref, n * sizeof(float), hipMemcpyDeviceToHost));
    HIPCK(hipMemcpy(yfus.data(), d_yfus, n * sizeof(float), hipMemcpyDeviceToHost));
    HIPCK(hipMemcpy(rfus.data(), d_rfus, n * sizeof(float), hipMemcpyDeviceToHost));

    const bool ok_y = SameBytes(yfus.data(), yref.data(), n);
    const bool ok_r = SameBytes(rfus.data(), xref.data(), n);
    CHECK(ok_y, "F4: normed output differs");
    CHECK(ok_r, "F4: rsum (= residual x) differs");
    if (!ok_y) ReportDiff("F4 y", yfus.data(), yref.data(), n);
    if (!ok_r) ReportDiff("F4 rsum", rfus.data(), xref.data(), n);
    std::printf("   case %d: y %s  rsum %s\n", c, ok_y ? "IDENTICAL" : "DIFFERS",
                ok_r ? "IDENTICAL" : "DIFFERS");

    hipFree(da); hipFree(db); hipFree(dw);
    hipFree(d_xref); hipFree(d_yref); hipFree(d_yfus); hipFree(d_rfus);
  }
}

// ---------------------------------------------------------------------------
// F3 (a): the store expression, on a vector that actually contains -0.0.
// ---------------------------------------------------------------------------
void TestF3Store() {
  std::printf("== F3 (a): y[i] = 0.0f + p[i]  vs  memset + axpy(1.0f)\n");
  const int n = 2560;
  std::vector<float> hp(n);
  FillRandom(hp, 4.f);
  // The exactness trap, spelled out in every element of the pattern below.
  const float specials[12] = {
      +0.0f, -0.0f, 1.0f, -1.0f, 0.0f, -0.0f,
      1e-45f /*min subnormal*/, -1e-45f, 3.5f, -3.5f, -0.0f, +0.0f};
  for (int i = 0; i < n; i += 211) hp[i] = specials[(i / 211) % 12];
  hp[1] = -0.0f;
  hp[2] = +0.0f;

  float *dp = DevFloats(hp);
  float *d_unf = nullptr, *d_zp = nullptr, *d_raw = nullptr;
  HIPCK(hipMalloc(&d_unf, n * sizeof(float)));
  HIPCK(hipMalloc(&d_zp, n * sizeof(float)));
  HIPCK(hipMalloc(&d_raw, n * sizeof(float)));
  if (!dp) {
    std::fprintf(stderr, "FAIL F3a: device alloc\n");
    ++failures;
    return;
  }
  const int blocks = (n + 63) / 64;
  // unfused: memset y = 0, then y[i] += 1.0f * p[i]  (k_axpy's exact body)
  HIPCK(hipMemset(d_unf, 0, n * sizeof(float)));
  t_axpy<<<blocks, 64>>>(1.0f, dp, d_unf, n);
  t_zp<<<blocks, 64>>>(dp, d_zp, n);
  t_raw<<<blocks, 64>>>(dp, d_raw, n);
  HIPCK(hipDeviceSynchronize());

  std::vector<float> unf(n), zp(n), raw(n);
  HIPCK(hipMemcpy(unf.data(), d_unf, n * sizeof(float), hipMemcpyDeviceToHost));
  HIPCK(hipMemcpy(zp.data(), d_zp, n * sizeof(float), hipMemcpyDeviceToHost));
  HIPCK(hipMemcpy(raw.data(), d_raw, n * sizeof(float), hipMemcpyDeviceToHost));

  const bool ok = SameBytes(zp.data(), unf.data(), n);
  const bool raw_bad = !SameBytes(raw.data(), unf.data(), n);
  CHECK(ok, "F3a: 0.0f + p is NOT byte-equal to memset+axpy");
  CHECK(raw_bad, "F3a: a raw copy unexpectedly matched -- the -0.0 case is not being tested");
  std::printf("   0.0f + p  vs memset+axpy : %s\n", ok ? "IDENTICAL" : "DIFFERS");
  std::printf("   raw copy  vs memset+axpy : %s (must DIFFER -- this is the trap)\n",
              raw_bad ? "DIFFERS (as it must)" : "IDENTICAL (test not exercising -0.0)");
  if (!ok) ReportDiff("F3a zp", zp.data(), unf.data(), n);
  if (raw_bad) {
    int shown = 0;
    for (int i = 0; i < n && shown < 4; ++i) {
      if (std::memcmp(&raw[i], &unf[i], sizeof(float)) != 0) {
        std::printf("     trap at [%d]: p=%a -> raw 0x%08x, unfused 0x%08x\n", i,
                    (double)hp[i], Bits(raw[i]), Bits(unf[i]));
        ++shown;
      }
    }
  }
  hipFree(dp); hipFree(d_unf); hipFree(d_zp); hipFree(d_raw);
}

// ---------------------------------------------------------------------------
// F3 (b) + F2: the real fused FFN against the unfused one, end to end.
// ---------------------------------------------------------------------------
void TestF3F2Ffn(int qtype, int k, int n_ff, int xq_ready, float limit) {
  std::printf("== F3 (b)+F2: dense_ffn vs dense_ffn_w1  qtype=%d k=%d n_ff=%d xq_ready=%d limit=%g\n",
              qtype, k, n_ff, xq_ready, (double)limit);

  std::vector<float> hx(k), hx2(k);
  FillRandom(hx, 2.f);
  FillRandom(hx2, 2.f);
  hx2[0] += 1.0f;   // guaranteed to quantise differently from hx
  const std::vector<std::uint8_t> gate = MakeWeights(qtype, k, n_ff);
  const std::vector<std::uint8_t> up = MakeWeights(qtype, k, n_ff);
  const std::vector<std::uint8_t> down = MakeWeights(qtype, n_ff, k);

  float *dx = DevFloats(hx), *dx2 = DevFloats(hx2);
  std::uint8_t *dg = nullptr, *du = nullptr, *dd = nullptr;
  float *y_ref = nullptr, *y_fus = nullptr, *y_ctl = nullptr;
  HIPCK(hipMalloc(&dg, gate.size()));
  HIPCK(hipMalloc(&du, up.size()));
  HIPCK(hipMalloc(&dd, down.size()));
  HIPCK(hipMalloc(&y_ref, k * sizeof(float)));
  HIPCK(hipMalloc(&y_fus, k * sizeof(float)));
  HIPCK(hipMalloc(&y_ctl, k * sizeof(float)));
  HIPCK(hipMemcpy(dg, gate.data(), gate.size(), hipMemcpyHostToDevice));
  HIPCK(hipMemcpy(du, up.data(), up.size(), hipMemcpyHostToDevice));
  HIPCK(hipMemcpy(dd, down.data(), down.size(), hipMemcpyHostToDevice));
  if (!dx || !dx2) {
    std::fprintf(stderr, "FAIL F3b: device alloc\n");
    ++failures;
    return;
  }
  // Poison all three outputs so a path that writes nothing cannot look like a match.
  HIPCK(hipMemset(y_ref, 0xAB, k * sizeof(float)));
  HIPCK(hipMemset(y_fus, 0xAB, k * sizeof(float)));
  HIPCK(hipMemset(y_ctl, 0xAB, k * sizeof(float)));

  if (xq_ready) {
    // Simulate what ling3_expert_ffn_dev did just before the shared-expert call:
    // quantise the SAME x into the SAME slot. F2 says the fused path may rely on it.
    CHECK(ling3_quant_q8(dx, k, LING3_Q8_SLOT_FFN_X, nullptr) == LING3_OK,
          "F2 setup: quant of x into SLOT_FFN_X failed");
  }

  // Unfused reference -- ling3_expert_ffn with n_used = 1, w = 1.0 (what
  // ling3_dense_ffn runs): memset y, quant x, gate/up, swiglu, quant h, down, axpy.
  const int rc_ref = ling3_dense_ffn(qtype, dg, du, dd, dx, 1, k, n_ff, limit,
                                     y_ref, nullptr);
  // F2/F3 fused form. For xq_ready = 1 this SKIPS the x quant entirely, so it only
  // matches if the bytes in SLOT_FFN_X really are the quantisation of this same x.
  const int rc_fus = ling3_dense_ffn_w1(qtype, dg, du, dd, dx, 1, k, n_ff, limit,
                                        y_fus, xq_ready, nullptr);
  CHECK(rc_ref == LING3_OK, "unfused ling3_dense_ffn failed");
  CHECK(rc_fus == LING3_OK, "fused ling3_dense_ffn_w1 failed");
  HIPCK(hipDeviceSynchronize());

  std::vector<float> a(k), b(k), c(k);
  HIPCK(hipMemcpy(a.data(), y_ref, k * sizeof(float), hipMemcpyDeviceToHost));
  HIPCK(hipMemcpy(b.data(), y_fus, k * sizeof(float), hipMemcpyDeviceToHost));
  const bool ok = SameBytes(b.data(), a.data(), k);
  CHECK(ok, "F3b: fused FFN output differs from the unfused one");
  std::printf("   rc_ref=%d rc_fus=%d  output %s\n", rc_ref, rc_fus,
              ok ? "IDENTICAL" : "DIFFERS");
  if (!ok) ReportDiff("F3b", b.data(), a.data(), k);

  if (xq_ready) {
    // Negative control: put a DIFFERENT vector in the slot and skip the quant again.
    // The output must now DIFFER, or this test cannot tell a stale slot from a live
    // one and the F2 match above would be vacuous.
    CHECK(ling3_quant_q8(dx2, k, LING3_Q8_SLOT_FFN_X, nullptr) == LING3_OK,
          "F2 control: quant of x2 into SLOT_FFN_X failed");
    const int rc_ctl = ling3_dense_ffn_w1(qtype, dg, du, dd, dx, 1, k, n_ff, limit,
                                          y_ctl, 1, nullptr);
    CHECK(rc_ctl == LING3_OK, "F2 control: fused call failed");
    HIPCK(hipDeviceSynchronize());
    HIPCK(hipMemcpy(c.data(), y_ctl, k * sizeof(float), hipMemcpyDeviceToHost));
    const bool ctl_differs = !SameBytes(c.data(), a.data(), k);
    CHECK(ctl_differs,
          "F2 control: a stale SLOT_FFN_X produced the SAME output -- the xq_ready "
          "path is not reading the slot the test thinks it is");
    std::printf("   F2 stale-slot control: %s\n",
                ctl_differs ? "DIFFERS (as it must)" : "IDENTICAL (BAD)");
  }

  hipFree(dx); hipFree(dx2); hipFree(dg); hipFree(du); hipFree(dd);
  hipFree(y_ref); hipFree(y_fus); hipFree(y_ctl);
}

}  // namespace

int main() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1) {
    std::fprintf(stderr, "FAIL: no HIP device\n");
    return 2;
  }
  std::printf("ling3_fusion_w1_test  H=2560  dev=%d\n", ndev);

  TestF4(2560, 4);
  TestF3Store();
  // Ling's two FFN widths: the shared expert (n_ff = g.expert_ff) and the dense
  // layers (n_ff = g.ffn_len). Both block types, both xq_ready values, and both
  // the unclamped swiglu (limit 0 = "no clamp") and a real clamp.
  for (int qt : {LING3_Q4_0_ROCMFP4, LING3_Q4_0_ROCMFP4_FAST}) {
    for (float lim : {0.0f, 7.5f}) {
      TestF3F2Ffn(qt, 2560, 768, 0, lim);
      TestF3F2Ffn(qt, 2560, 768, 1, lim);
      TestF3F2Ffn(qt, 2560, 4096, 0, lim);
      TestF3F2Ffn(qt, 2560, 4096, 1, lim);
    }
  }

  if (failures == 0) {
    std::printf("FUSION-W1 ALL PASS\n");
    return 0;
  }
  std::printf("FUSION-W1 FAIL fail=%d\n", failures);
  return 1;
}
