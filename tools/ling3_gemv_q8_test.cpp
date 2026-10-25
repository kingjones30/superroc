// ling3-gemv-q8-test — is LS1's integer ROCmFP4 GEMV correct, on small shapes,
// against a CPU reference, BEFORE it runs on the 44 GB model?
//
// A page fault in a decode kernel resets the GPU and kills every live seat on
// amdaimax2, so this runs as its own short gpu-run job and has to be green
// first. It checks three things:
//
//   1. the FP4 nibble -> int8 perm codebook, exhaustively, for all 65536
//      four-nibble words (a wrong perm permutes products instead of scaling
//      them, which a float reference would read as ordinary quantisation noise;
//      it is what caught a 2x scale error in an earlier revision of the kernel,
//      which the float comparison alone would have passed as noise);
//   2. the integer GEMV against a CPU transcription of the SAME algorithm --
//      this catches every index, stride and scale bug, and it is what caught
//      the second dot using .x where it has to use .y;
//   3. the integer GEMV against the float dequant reference build15 uses -- this
//      is the number that says what the activation quantisation costs, and it is
//      what the LM head and the expert matrices will live with.
//
// Errors are normalised by the RMS of the reference over the rows, not by each
// row's own value: with random weights some rows' dots cancel to within a
// rounding error of zero, and a relative error against that denominator is
// meaningless (it read 1e2 on rows whose absolute error was 3e-3). Relative
// error is still reported, over the rows that carry signal.
//
// Weights are synthetic but not kind: every one of the sixteen FP4 codes appears
// many times in every block, and the scale bytes are drawn from the whole ue4m3
// range including the 0x7f / 0x7e guards ling3_ue4m3_half has.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/models/ling3/kernels/ling3_gemv_q8.hpp"
#include "src/models/ling3/kernels/ling3_gemm.hpp"
#include "src/models/ling3/kernels/ling3_moe.hpp"
#include "src/models/ling3/kernels/ling3_quant.hpp"

namespace {

std::uint64_t rng = 0x9E3779B97F4A7C15ULL;
std::uint32_t Next() {
  rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<std::uint32_t>(rng >> 32);
}

#define CHECK(cond, msg)                                                        \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (msg));       \
      ++failures;                                                               \
    }                                                                           \
  } while (0)

int failures = 0;

std::size_t RowBytes(int qtype, int k) {
  return ling3_row_bytes(qtype, k);
}

/// n rows of k values. Nibble bytes are random so all sixteen codes appear many
/// times in every block, and the scale bytes are drawn over the whole ue4m3
/// range: 0x00..0x7e are finite, 0x7f and 0x80..0xff hit the `e > 0x7e -> 0`
/// guard, and both must behave the same here and on the CPU.
std::vector<std::uint8_t> MakeWeights(int qtype, int k, int n) {
  const std::size_t rb = RowBytes(qtype, k);
  std::vector<std::uint8_t> w(n * rb);
  const int nb = k / 32;
  const std::size_t tsz = ling3_type_size(qtype);
  for (int r = 0; r < n; ++r) {
    std::uint8_t * row = w.data() + (std::size_t) r * rb;
    for (int b = 0; b < nb; ++b) {
      std::uint8_t * blk = row + (std::size_t) b * tsz;
      for (int i = 0; i < 16; ++i) {
        blk[i] = (std::uint8_t) (Next() & 0xFF);  // both nibbles of every byte vary
      }
      if (qtype == LING3_Q4_0_ROCMFP4) {
        blk[16] = (std::uint8_t) (Next() % 256);
        blk[17] = (std::uint8_t) (Next() % 256);
      } else {
        blk[16] = (std::uint8_t) (Next() % 256);
      }
    }
  }
  return w;
}

std::vector<float> MakeActivation(int k) {
  std::vector<float> x(k);
  for (int i = 0; i < k; ++i) {
    x[i] = -3.f + 6.f * (float) Next() / 4294967296.0F;
  }
  // A few exact zeros and one very small value: the all-zero Q8_1 block and the
  // divide-by-d end of the scale are both places a rounding fault would show up.
  for (int i = 0; i < k; i += 97) {
    x[i] = 0.f;
  }
  x[k / 2] = 1e-7f;
  return x;
}

/// Exactly what the kernel computes: quantise the activation the ggml way, then per
/// 32-weight block accumulate code * q8 on each half and apply the FP4 scales. Any
/// index or stride error in the kernel shows up here as a big miss.
float RefInteger(int qtype, const std::uint8_t * wrow, const float * x, int k,
                 std::vector<std::int8_t> * q_out, std::vector<float> * d_out) {
  const int nb = k / 32;
  const std::size_t tsz = ling3_type_size(qtype);
  std::vector<std::int8_t> q(k);
  std::vector<float> d(nb);
  for (int b = 0; b < nb; ++b) {
    float amax = 0.f, mx = 0.f;
    for (int j = 0; j < 32; ++j) {
      const float v = x[b * 32 + j];
      if (std::fabs(v) > amax) { amax = std::fabs(v); mx = v; }
    }
    const float dd = mx / 127.f;
    const float id = dd != 0.f ? 1.f / dd : 0.f;
    d[b] = dd;
    for (int j = 0; j < 32; ++j) {
      q[b * 32 + j] = (std::int8_t) std::round(x[b * 32 + j] * id);
    }
  }
  float s = 0.f;
  for (int b = 0; b < nb; ++b) {
    const std::uint8_t * blk = wrow + (std::size_t) b * tsz;
    int sumi0 = 0, sumi1 = 0;
    for (int l = 0; l < 4; ++l) {
      for (int i = 0; i < 4; ++i) {
        const std::uint8_t byte = blk[4 * l + i];
        sumi0 += ling3_rocmfp4_code(byte & 0x0fu) * (int) q[b * 32 + 4 * l + i];
        sumi1 += ling3_rocmfp4_code(byte >> 4) * (int) q[b * 32 + 16 + 4 * l + i];
      }
    }
    const float slo = ling3_ue4m3_half(blk[16]);
    const float shi = (qtype == LING3_Q4_0_ROCMFP4) ? ling3_ue4m3_half(blk[17]) : slo;
    s += d[b] * (slo * (float) sumi0 + shi * (float) sumi1);
  }
  if (q_out) *q_out = q;
  if (d_out) *d_out = d;
  return s;
}

/// What build15's k_gemv_wide computes, written out: each weight decoded exactly as
/// ling3_elem would (same ue4m3 guard, same code table) and multiplied by the FLOAT
/// activation. The gap between this and the kernel is the price of quantising the
/// activation, which is what the report has to state.
float RefFloatExact(int qtype, const std::uint8_t * wrow, const float * x, int k) {
  const int nb = k / 32;
  const std::size_t tsz = ling3_type_size(qtype);
  float s = 0.f;
  for (int b = 0; b < nb; ++b) {
    const std::uint8_t * blk = wrow + (std::size_t) b * tsz;
    const float slo = ling3_ue4m3_half(blk[16]);
    const float shi = (qtype == LING3_Q4_0_ROCMFP4) ? ling3_ue4m3_half(blk[17]) : slo;
    for (int j = 0; j < 32; ++j) {
      const std::uint8_t byte = blk[j & 15];
      const int nib = (j < 16) ? (int) (byte & 0x0f) : (int) (byte >> 4);
      const float f = (j < 16) ? slo : shi;
      s += ((float) ling3_rocmfp4_code((std::uint32_t) nib) * f) * x[b * 32 + j];
    }
  }
  return s;
}


// ---------------------------------------------------------------------------
// The grouped device-routed expert FFN (LS1 phase 3) against the per-expert host loop it
// replaces.
//
// The gate for this kernel, and the reason it was wrong for a whole phase: BOTH grouped
// kernels are launched with 64-thread CTAs, and phase 2 reduced them with a 32-lane warp
// sum, so lanes 32-63 were dropped. Whether that shows up depends on how many 32-value
// blocks there are to spread over the lanes -- at k=512, n_ff=256 the work lands entirely
// on lanes 0-15 and the kernel looks perfect; at Ling's real k=2560 it lands on all 64 and
// the result is 3-6x wrong. So the cases below change ONE THING at a time: n_used = 1 and
// n_used = 8 at the same real shape, plus a run-to-run determinism check.
// ---------------------------------------------------------------------------
void RunGroupedExperts(int qtype, int n_expert, int n_used, int k, int n_ff, bool print_head) {
  const std::size_t rb_k = RowBytes(qtype, k);
  const std::size_t rb_ff = RowBytes(qtype, n_ff);
  if (rb_k == 0 || rb_ff == 0) return;

  std::vector<int> ids(n_used);
  std::vector<float> wts(n_used);
  for (int s = 0; s < n_used; ++s) {
    int e;
    bool clash;
    do {
      e = (int) (Next() % (std::uint32_t) n_expert);
      clash = false;
      for (int t = 0; t < s; ++t) if (ids[t] == e) clash = true;
    } while (clash);
    ids[s] = e;
    wts[s] = 0.05f + 0.2f * (float) Next() / 4294967296.0F;
  }

  const std::size_t gate_sz = (std::size_t) n_expert * (std::size_t) n_ff * rb_k;
  const std::size_t down_sz = (std::size_t) n_expert * (std::size_t) k * rb_ff;
  const std::size_t tsz = ling3_type_size(qtype);
  std::vector<std::uint8_t> gate(gate_sz, 0), up(gate_sz, 0), down(down_sz, 0);
  // Scale bytes in [0x2c, 0x44), where phase 2 drew them up to 0x5c. This does NOT remove
  // the clamp saturation -- with x in [-3,3] and 2560 terms the gate/up preactivations are
  // ~1e2 whatever the weights, so hs still pins at +-limit and the peek below still shows
  // a [-49, 49] field. I had guessed in REPORT2 that the saturated test data was what hid
  // the phase-2 bug; it was not, the bug was a 32-lane reduction in a 64-thread CTA. The
  // narrower range is kept because it is a more honest stand-in for the model's weights,
  // not because it fixed anything.
  const std::uint8_t smin = 0x2c, srange = 0x18;
  for (int s = 0; s < n_used; ++s) {
    std::uint8_t * g = gate.data() + (std::size_t) ids[s] * (std::size_t) n_ff * rb_k;
    std::uint8_t * u = up.data() + (std::size_t) ids[s] * (std::size_t) n_ff * rb_k;
    std::uint8_t * d = down.data() + (std::size_t) ids[s] * (std::size_t) k * rb_ff;
    for (int r = 0; r < n_ff; ++r) {
      for (int b = 0; b < k / 32; ++b) {
        const std::size_t o = (std::size_t) r * rb_k + (std::size_t) b * tsz;
        for (int q = 0; q < 16; ++q) g[o + q] = (std::uint8_t) (Next() & 0xFF);
        g[o + 16] = (std::uint8_t) (smin + Next() % srange);
        for (int q = 0; q < 16; ++q) u[o + q] = (std::uint8_t) (Next() & 0xFF);
        u[o + 16] = (std::uint8_t) (smin + Next() % srange);
      }
    }
    for (int r = 0; r < k; ++r) {
      for (int b = 0; b < n_ff / 32; ++b) {
        const std::size_t o = (std::size_t) r * rb_ff + (std::size_t) b * tsz;
        for (int q = 0; q < 16; ++q) d[o + q] = (std::uint8_t) (Next() & 0xFF);
        d[o + 16] = (std::uint8_t) (smin + Next() % srange);
      }
    }
  }
  const std::vector<float> x = MakeActivation(k);

  std::uint8_t * d_gate = nullptr; std::uint8_t * d_up = nullptr; std::uint8_t * d_down = nullptr;
  float * d_x = nullptr; int * d_ids = nullptr; float * d_w = nullptr;
  float * d_y1 = nullptr; float * d_y2 = nullptr;
  const int before = failures;
  CHECK(hipMalloc(&d_gate, gate_sz + LING3_TAIL_PAD) == hipSuccess, "ge malloc gate");
  CHECK(hipMalloc(&d_up, gate_sz + LING3_TAIL_PAD) == hipSuccess, "ge malloc up");
  CHECK(hipMalloc(&d_down, down_sz + LING3_TAIL_PAD) == hipSuccess, "ge malloc down");
  CHECK(hipMalloc(&d_x, ((std::size_t) k + LING3_TAIL_PAD) * sizeof(float)) == hipSuccess, "ge malloc x");
  CHECK(hipMalloc(&d_ids, n_used * sizeof(int)) == hipSuccess, "ge malloc ids");
  CHECK(hipMalloc(&d_w, n_used * sizeof(float)) == hipSuccess, "ge malloc w");
  CHECK(hipMalloc(&d_y1, ((std::size_t) k + LING3_TAIL_PAD) * sizeof(float)) == hipSuccess, "ge malloc y1");
  CHECK(hipMalloc(&d_y2, ((std::size_t) k + LING3_TAIL_PAD) * sizeof(float)) == hipSuccess, "ge malloc y2");
  if (failures != before) return;
  CHECK(hipMemcpy(d_gate, gate.data(), gate_sz, hipMemcpyHostToDevice) == hipSuccess, "ge H2D gate");
  CHECK(hipMemcpy(d_up, up.data(), gate_sz, hipMemcpyHostToDevice) == hipSuccess, "ge H2D up");
  CHECK(hipMemcpy(d_down, down.data(), down_sz, hipMemcpyHostToDevice) == hipSuccess, "ge H2D down");
  CHECK(hipMemcpy(d_x, x.data(), (std::size_t) k * sizeof(float), hipMemcpyHostToDevice) == hipSuccess, "ge H2D x");
  CHECK(hipMemcpy(d_ids, ids.data(), n_used * sizeof(int), hipMemcpyHostToDevice) == hipSuccess, "ge H2D ids");
  CHECK(hipMemcpy(d_w, wts.data(), n_used * sizeof(float), hipMemcpyHostToDevice) == hipSuccess, "ge H2D w");

  const float limit = 7.0f;
  std::vector<float> y_ref(k), y_new(k), y_again(k);

  // Reference: the per-expert host loop, which is what the seat runs today.
  CHECK(ling3_expert_ffn(qtype, d_gate, d_up, d_down, d_x, ids.data(), wts.data(),
                         1, n_used, n_expert, k, n_ff, limit, d_y1) == LING3_OK, "ge host loop rc");
  CHECK(hipDeviceSynchronize() == hipSuccess, "ge host loop sync");
  CHECK(hipMemcpy(y_ref.data(), d_y1, (std::size_t) k * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess, "ge D2H ref");

  // The grouped device-routed path, twice.
  CHECK(ling3_expert_ffn_dev(qtype, d_gate, d_up, d_down, d_x, d_ids, d_w,
                             n_used, n_expert, k, n_ff, limit, d_y2) == LING3_OK, "ge dev rc");
  CHECK(hipDeviceSynchronize() == hipSuccess, "ge dev sync");
  CHECK(hipMemcpy(y_new.data(), d_y2, (std::size_t) k * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess, "ge D2H new");
  CHECK(ling3_expert_ffn_dev(qtype, d_gate, d_up, d_down, d_x, d_ids, d_w,
                             n_used, n_expert, k, n_ff, limit, d_y2) == LING3_OK, "ge dev rc 2");
  CHECK(hipDeviceSynchronize() == hipSuccess, "ge dev sync 2");
  CHECK(hipMemcpy(y_again.data(), d_y2, (std::size_t) k * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess, "ge D2H again");
  CHECK(ling3_moe_dev_check() == LING3_OK, "ge dev status word");

  double sumsq = 0.0;
  for (int i = 0; i < k; ++i) sumsq += (double) y_ref[i] * (double) y_ref[i];
  const double rms = std::sqrt(sumsq / k);
  const double scale = rms > 1e-6 ? rms : 1.0;
  double worst = 0.0;
  int bitdiff = 0;
  for (int i = 0; i < k; ++i) {
    const double e = std::fabs((double) y_new[i] - (double) y_ref[i]) / scale;
    if (e > worst) worst = e;
    if (y_new[i] != y_again[i]) ++bitdiff;
  }
  // The same bar the phase-1 GEMV holds itself to against its own CPU transcription.
  CHECK(worst < 1e-5, "grouped expert FFN != per-expert host loop (rms-normalised)");
  CHECK(bitdiff == 0, "grouped expert FFN is not run-to-run deterministic");
  std::printf("  GROUPED qtype=%3d n_expert=%4d n_used=%d k=%d n_ff=%d  err/rms=%.3g  rms=%.5g  "
              "nondet=%d/%d\n", qtype, n_expert, n_used, k, n_ff, worst, rms, bitdiff, k);
  if (print_head) {
    std::printf("    head host_loop= %.6g %.6g %.6g %.6g\n", y_ref[0], y_ref[1], y_ref[2], y_ref[3]);
    std::printf("    head grouped  = %.6g %.6g %.6g %.6g\n", y_new[0], y_new[1], y_new[2], y_new[3]);
    std::vector<float> hs(n_used * n_ff);
    if (ling3_moe_hs_peek(hs.data(), n_used * n_ff) == 0) {
      double lo = 1e30, hi = -1e30;
      for (int i = 0; i < n_used * n_ff; ++i) { lo = std::min(lo, (double) hs[i]); hi = std::max(hi, (double) hs[i]); }
      std::printf("    hs range [%.5g, %.5g]  hs[0]=%.6g  hs[n_ff]=%.6g\n", lo, hi, hs[0], hs[n_ff]);
    }
  }

  hipFree(d_gate); hipFree(d_up); hipFree(d_down); hipFree(d_x); hipFree(d_ids);
  hipFree(d_w); hipFree(d_y1); hipFree(d_y2);
}


// ---------------------------------------------------------------------------
// LS1 phase 4 item 3: the small-m decode forward.
//
// The gate is PER COLUMN, against the m = 1 path, not against a float reference: the whole
// point of the kernel is that column t of an m-way matvec is the same arithmetic as an
// m = 1 matvec on activation t, so any difference is a bug in the sharing -- the hoisted
// codebook, the per-column scale, the reduction, the token-major output indexing. Plus
// run-to-run determinism, plus timings for m = 1..4 against the m = 3 <= 1.4x m = 1 target.
// ---------------------------------------------------------------------------
void RunSmallM(int qtype, int k, int n, bool time_it) {
  const std::size_t rb = RowBytes(qtype, k);
  if (rb == 0) return;
  const std::vector<std::uint8_t> w = MakeWeights(qtype, k, n);
  std::uint8_t * d_w = nullptr;
  float * d_y1 = nullptr; float * d_ym = nullptr;
  std::vector<float> x(4 * k);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = -3.f + 6.f * (float) Next() / 4294967296.0F;
  float * d_x = nullptr;
  const int before = failures;
  CHECK(hipMalloc(&d_w, w.size() + LING3_TAIL_PAD) == hipSuccess, "sm malloc W");
  CHECK(hipMalloc(&d_x, (4 * k + LING3_TAIL_PAD) * sizeof(float)) == hipSuccess, "sm malloc x");
  CHECK(hipMalloc(&d_y1, (k > 0 ? n : 1) * sizeof(float)) == hipSuccess, "sm malloc y1");
  CHECK(hipMalloc(&d_ym, 4 * n * sizeof(float)) == hipSuccess, "sm malloc ym");
  if (failures != before) return;
  CHECK(hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice) == hipSuccess, "sm H2D W");
  CHECK(hipMemcpy(d_x, x.data(), 4 * k * sizeof(float), hipMemcpyHostToDevice) == hipSuccess, "sm H2D x");

  std::vector<float> y1(n), ym(4 * n);
  double worst = 0.0;
  int m = 0;
  for (m = 1; m <= 4; ++m) {
    // m = 1 column at a time, from the m = 1 entry point.
    std::fill(y1.begin(), y1.end(), 0.f);
    for (int t = 0; t < m; ++t) {
      CHECK(ling3_gemv(qtype, d_w, d_x + (std::size_t) t * k, d_y1, k, n) == LING3_OK, "sm gemv1");
      CHECK(hipDeviceSynchronize() == hipSuccess, "sm sync1");
      std::vector<float> col(n);
      CHECK(hipMemcpy(col.data(), d_y1, (std::size_t) n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess, "sm D2H1");
      for (int r = 0; r < n; ++r) y1[r] = col[r];
      std::vector<float> run(n);
      CHECK(hipMemcpy(run.data(), d_y1, (std::size_t) n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess, "sm D2H1b");
      (void) run;
      break;   // one m=1 reference per column is enough; see the loop below
    }
    // Reference: m separate m=1 calls, one per column.
    std::vector<float> ref(4 * n, 0.f);
    for (int t = 0; t < m; ++t) {
      CHECK(ling3_gemv(qtype, d_w, d_x + (std::size_t) t * k, d_y1, k, n) == LING3_OK, "sm ref gemv1");
      CHECK(hipDeviceSynchronize() == hipSuccess, "sm ref sync");
      std::vector<float> col(n);
      CHECK(hipMemcpy(col.data(), d_y1, (std::size_t) n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess, "sm ref D2H");
      for (int r = 0; r < n; ++r) ref[(std::size_t) t * n + r] = col[r];
    }
    // The m-way kernel.
    CHECK(hipMemset(d_ym, 0x7f, 4 * n * sizeof(float)) == hipSuccess, "sm memset");
    CHECK(ling3_gemv_m(qtype, d_w, d_x, d_ym, k, n, m) == LING3_OK, "sm gemv_m");
    CHECK(hipDeviceSynchronize() == hipSuccess, "sm sync_m");
    CHECK(hipMemcpy(ym.data(), d_ym, 4 * n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess, "sm D2H_m");

    // Second run, for determinism.
    std::vector<float> again(4 * n, 0.f);
    CHECK(ling3_gemv_m(qtype, d_w, d_x, d_ym, k, n, m) == LING3_OK, "sm gemv_m 2");
    CHECK(hipDeviceSynchronize() == hipSuccess, "sm sync_m2");
    CHECK(hipMemcpy(again.data(), d_ym, 4 * n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess, "sm D2H_m2");
    int nondet = 0;
    for (std::size_t i = 0; i < 4 * (std::size_t) n; ++i) if (ym[i] != again[i]) ++nondet;

    double sumsq = 0.0;
    for (int r = 0; r < n; ++r) sumsq += (double) ref[r] * (double) ref[r];
    const double rms = std::sqrt(sumsq / (n > 0 ? n : 1));
    const double scale = rms > 1e-6 ? rms : 1.0;
    double w = 0.0;
    for (int t = 0; t < m; ++t) {
      for (int r = 0; r < n; ++r) {
        const double e = std::fabs((double) ym[(std::size_t) t * n + r] - (double) ref[(std::size_t) t * n + r]) / scale;
        if (e > w) w = e;
      }
    }
    CHECK(w == 0.0, "small-m column differs from the m=1 path (must be BIT-EXACT)");
    CHECK(nondet == 0, "small-m GEMV is not run-to-run deterministic");
    worst = std::max(worst, w);
    if (time_it) {
      std::printf("  SMALLM qtype=%3d k=%5d n=%6d m=%d  bit-exact=%s  nondet=%d/%zu\n",
                  qtype, k, n, m, w == 0.0 ? "yes" : "NO", nondet, 4 * (std::size_t) n);
    }
  }
  (void) worst;
  hipFree(d_w); hipFree(d_x); hipFree(d_y1); hipFree(d_ym);
}

// Timings for the shapes Ling actually decodes, m = 1..4. Target: m = 3 <= 1.4x m = 1.
// MALL trap, straight out of the wave's NUDGE: this box has a 32 MB Infinity Cache, and a
// microbenchmark whose weights FIT in it reports 3-7x the real bandwidth -- the first
// version of this timing measured 461 GB/s on a part whose real read bandwidth is ~256.
// So the working set is padded to at least 160 MB with copies of the matrix and the
// kernel is launched against a DIFFERENT copy each rep, which is what keeps the MALL cold.
void TimeSmallM(int qtype, int k, int n) {
  const std::size_t rb = RowBytes(qtype, k);
  if (rb == 0) return;
  const std::vector<std::uint8_t> w = MakeWeights(qtype, k, n);
  const std::size_t one = w.size();
  int copies = 1;
  while (one * (std::size_t) copies < (std::size_t) 160 * 1024 * 1024) copies *= 2;
  const std::size_t total = one * (std::size_t) copies;
  std::uint8_t * d_w = nullptr; float * d_x = nullptr; float * d_ym = nullptr;
  if (hipMalloc(&d_w, total + LING3_TAIL_PAD) != hipSuccess) return;
  if (hipMalloc(&d_x, (4 * k + LING3_TAIL_PAD) * sizeof(float)) != hipSuccess) { hipFree(d_w); return; }
  if (hipMalloc(&d_ym, 4 * n * sizeof(float)) != hipSuccess) { hipFree(d_w); hipFree(d_x); return; }
  for (int c = 0; c < copies; ++c) {
    hipMemcpy(d_w + (std::size_t) c * one, w.data(), one, hipMemcpyHostToDevice);
  }
  std::vector<float> x(4 * k, 0.5f);
  hipMemcpy(d_x, x.data(), 4 * k * sizeof(float), hipMemcpyHostToDevice);
  const int reps = 20;
  float t1 = 0.f, t3 = 0.f;
  std::printf("  (working set %zu MiB over %d copies, so the 32 MB MALL cannot hold it)\n",
              total / (1024 * 1024), copies);
  for (int m = 1; m <= 4; ++m) {
    ling3_gemv_m(qtype, d_w, d_x, d_ym, k, n, m);
    hipDeviceSynchronize();
    hipEvent_t a, b;
    hipEventCreate(&a); hipEventCreate(&b);
    hipEventRecord(a, (hipStream_t) 0);
    for (int i = 0; i < reps; ++i) {
      ling3_gemv_m(qtype, d_w + (std::size_t) (i % copies) * one, d_x, d_ym, k, n, m);
    }
    hipEventRecord(b, (hipStream_t) 0);
    hipEventSynchronize(b);
    float ms = 0.f;
    hipEventElapsedTime(&ms, a, b);
    ms /= (float) reps;
    if (m == 1) t1 = ms;
    if (m == 3) t3 = ms;
    std::printf("  TIMING qtype=%3d k=%5d n=%6d m=%d  %8.3f ms  %6.1f GB/s  x%.2f vs m=1\n",
                qtype, k, n, m, ms,
                (double) one / (ms > 1e-6f ? ms * 1e-3f : 1e-9) / 1e9, ms / (t1 > 0 ? t1 : 1.f));
    hipEventDestroy(a); hipEventDestroy(b);
  }
  CHECK(t3 <= 1.4f * t1, "small-m: m=3 costs more than 1.4x m=1");
  std::printf("  TIMING summary: m=3 is %.2fx m=1 (target <= 1.40x)\n", t3 / (t1 > 0 ? t1 : 1.f));
  hipFree(d_w); hipFree(d_x); hipFree(d_ym);
}

// The small-m grouped expert FFN over (token x slot) pairs, against running the m = 1
// kernel once per token with that token's own ids and weights.
void RunSmallMExperts(int qtype, int n_expert, int n_used, int n_tokens, int k, int n_ff) {
  const std::size_t rb_k = RowBytes(qtype, k);
  const std::size_t rb_ff = RowBytes(qtype, n_ff);
  if (rb_k == 0 || rb_ff == 0) return;
  std::vector<int> ids(n_tokens * n_used);
  std::vector<float> wts(n_tokens * n_used);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    int e; bool clash;
    do { e = (int) (Next() % (std::uint32_t) n_expert); clash = false;
         for (std::size_t j = 0; j < i; ++j) if (ids[j] == e) clash = true; } while (clash);
    ids[i] = e;
    wts[i] = 0.05f + 0.2f * (float) Next() / 4294967296.0F;
  }
  const std::size_t gate_sz = (std::size_t) n_expert * (std::size_t) n_ff * rb_k;
  const std::size_t down_sz = (std::size_t) n_expert * (std::size_t) k * rb_ff;
  const std::size_t tsz = ling3_type_size(qtype);
  std::vector<std::uint8_t> gate(gate_sz, 0), up(gate_sz, 0), down(down_sz, 0);
  const std::uint8_t smin = 0x2c, srange = 0x18;
  for (std::size_t s = 0; s < ids.size(); ++s) {
    std::uint8_t * g = gate.data() + (std::size_t) ids[s] * (std::size_t) n_ff * rb_k;
    std::uint8_t * u = up.data() + (std::size_t) ids[s] * (std::size_t) n_ff * rb_k;
    std::uint8_t * d = down.data() + (std::size_t) ids[s] * (std::size_t) k * rb_ff;
    for (int r = 0; r < n_ff; ++r) for (int b = 0; b < k / 32; ++b) {
      const std::size_t o = (std::size_t) r * rb_k + (std::size_t) b * tsz;
      for (int q = 0; q < 16; ++q) g[o+q] = (std::uint8_t)(Next() & 0xFF);
      g[o+16] = (std::uint8_t)(smin + Next() % srange);
      for (int q = 0; q < 16; ++q) u[o+q] = (std::uint8_t)(Next() & 0xFF);
      u[o+16] = (std::uint8_t)(smin + Next() % srange);
    }
    for (int r = 0; r < k; ++r) for (int b = 0; b < n_ff / 32; ++b) {
      const std::size_t o = (std::size_t) r * rb_ff + (std::size_t) b * tsz;
      for (int q = 0; q < 16; ++q) d[o+q] = (std::uint8_t)(Next() & 0xFF);
      d[o+16] = (std::uint8_t)(smin + Next() % srange);
    }
  }
  std::vector<float> x((std::size_t) n_tokens * k);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = -3.f + 6.f * (float) Next() / 4294967296.0F;

  std::uint8_t * d_gate=nullptr,* d_up=nullptr,* d_down=nullptr; float * d_x=nullptr;
  int * d_ids=nullptr; float * d_w=nullptr; float * d_ym=nullptr; float * d_y1=nullptr;
  const int before = failures;
  CHECK(hipMalloc(&d_gate, gate_sz + LING3_TAIL_PAD) == hipSuccess, "sme gate");
  CHECK(hipMalloc(&d_up, gate_sz + LING3_TAIL_PAD) == hipSuccess, "sme up");
  CHECK(hipMalloc(&d_down, down_sz + LING3_TAIL_PAD) == hipSuccess, "sme down");
  CHECK(hipMalloc(&d_x, ((std::size_t) n_tokens * k + LING3_TAIL_PAD) * sizeof(float)) == hipSuccess, "sme x");
  CHECK(hipMalloc(&d_ids, ids.size() * sizeof(int)) == hipSuccess, "sme ids");
  CHECK(hipMalloc(&d_w, wts.size() * sizeof(float)) == hipSuccess, "sme w");
  CHECK(hipMalloc(&d_ym, (std::size_t) n_tokens * k * sizeof(float)) == hipSuccess, "sme ym");
  CHECK(hipMalloc(&d_y1, ((std::size_t) k + LING3_TAIL_PAD) * sizeof(float)) == hipSuccess, "sme y1");
  if (failures != before) return;
  hipMemcpy(d_gate, gate.data(), gate_sz, hipMemcpyHostToDevice);
  hipMemcpy(d_up, up.data(), gate_sz, hipMemcpyHostToDevice);
  hipMemcpy(d_down, down.data(), down_sz, hipMemcpyHostToDevice);
  hipMemcpy(d_x, x.data(), (std::size_t) n_tokens * k * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_ids, ids.data(), ids.size() * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(d_w, wts.data(), wts.size() * sizeof(float), hipMemcpyHostToDevice);

  const float limit = 7.0f;
  std::vector<float> ym((std::size_t) n_tokens * k, 0.f), again((std::size_t) n_tokens * k, 0.f);
  CHECK(ling3_expert_ffn_m(qtype, d_gate, d_up, d_down, d_x, d_ids, d_w,
                           n_tokens, n_used, n_expert, k, n_ff, limit, d_ym) == LING3_OK, "sme rc");
  CHECK(hipDeviceSynchronize() == hipSuccess, "sme sync");
  CHECK(ling3_expert_ffn_m_check() == LING3_OK, "sme status");
  hipMemcpy(ym.data(), d_ym, (std::size_t) n_tokens * k * sizeof(float), hipMemcpyDeviceToHost);
  CHECK(ling3_expert_ffn_m(qtype, d_gate, d_up, d_down, d_x, d_ids, d_w,
                           n_tokens, n_used, n_expert, k, n_ff, limit, d_ym) == LING3_OK, "sme rc2");
  CHECK(hipDeviceSynchronize() == hipSuccess, "sme sync2");
  hipMemcpy(again.data(), d_ym, (std::size_t) n_tokens * k * sizeof(float), hipMemcpyDeviceToHost);

  // Reference: the m = 1 grouped kernel, once per token, with that token's own ids/weights.
  int bad = 0;
  double worst = 0.0;
  std::vector<float> col(k);
  std::vector<int> ids1(1); std::vector<float> w1(1);
  for (int t = 0; t < n_tokens; ++t) {
    // ling3_expert_ffn_dev wants n_used ids; pass this token's n_used.
    std::vector<int> tt_ids((std::size_t) n_used);
    std::vector<float> tt_w((std::size_t) n_used);
    for (int j = 0; j < n_used; ++j) {
      tt_ids[j] = ids[(std::size_t) t * n_used + j];
      tt_w[j] = wts[(std::size_t) t * n_used + j];
    }
    int * dt=nullptr; float * dw=nullptr;
    hipMalloc(&dt, (std::size_t) n_used * sizeof(int));
    hipMalloc(&dw, (std::size_t) n_used * sizeof(float));
    hipMemcpy(dt, tt_ids.data(), (std::size_t) n_used * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(dw, tt_w.data(), (std::size_t) n_used * sizeof(float), hipMemcpyHostToDevice);
    CHECK(ling3_expert_ffn_dev(qtype, d_gate, d_up, d_down, d_x + (std::size_t) t * k,
                                dt, dw, n_used, n_expert, k, n_ff, limit, d_y1) == LING3_OK, "sme ref rc");
    CHECK(hipDeviceSynchronize() == hipSuccess, "sme ref sync");
    CHECK(ling3_moe_dev_check() == LING3_OK, "sme ref status");
    hipMemcpy(col.data(), d_y1, (std::size_t) k * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(dt); hipFree(dw);
    for (int i = 0; i < k; ++i) {
      if (col[i] != ym[(std::size_t) t * k + i]) ++bad;
      const double e = std::fabs((double) ym[(std::size_t) t * k + i] - (double) col[i]);
      if (e > worst) worst = e;
    }
  }
  int nondet = 0;
  for (std::size_t i = 0; i < ym.size(); ++i) if (ym[i] != again[i]) ++nondet;
  CHECK(bad == 0, "small-m expert FFN differs from the m=1 grouped kernel (must be BIT-EXACT)");
  CHECK(nondet == 0, "small-m expert FFN is not run-to-run deterministic");
  std::printf("  SME qtype=%3d n_expert=%4d n_used=%d n_tokens=%d k=%d n_ff=%d  bit-exact=%s  "
              "mismatching=%d/%zu  max_abs=%.3g  nondet=%d\n",
              qtype, n_expert, n_used, n_tokens, k, n_ff, bad == 0 ? "yes" : "NO",
              bad, (std::size_t) n_tokens * k, worst, nondet);
  hipFree(d_gate); hipFree(d_up); hipFree(d_down); hipFree(d_x);
  hipFree(d_ids); hipFree(d_w); hipFree(d_ym); hipFree(d_y1);
}


// ---------------------------------------------------------------------------
// MALL-cold timing of the two kernels PHASE6 asks me to tune, at the shapes Ling
// actually decodes them at. Same discipline as TimeSmallM: the working set is
// padded past 160 MB with copies and each rep reads a DIFFERENT copy, because
// this box's 32 MB Infinity Cache will otherwise report 3-7x the real bandwidth
// (it measured 461 GB/s for me in phase 4).
//
// These are the local before/after signal. The absolute denominator -- the roof
// -- comes from tools/ling3_roof_bench.cpp, which is a separate job.
void TimeGemvM1(int qtype, int k, int n) {
  const std::size_t rb = RowBytes(qtype, k);
  if (rb == 0) return;
  const std::vector<std::uint8_t> w = MakeWeights(qtype, k, n);
  const std::size_t one = w.size();
  int copies = 1;
  while (one * (std::size_t) copies < (std::size_t) 160 * 1024 * 1024) copies *= 2;
  std::uint8_t * d_w = nullptr; float * d_x = nullptr; float * d_y = nullptr;
  if (hipMalloc(&d_w, one * (std::size_t) copies + LING3_TAIL_PAD) != hipSuccess) return;
  if (hipMalloc(&d_x, ((std::size_t) k + LING3_TAIL_PAD) * sizeof(float)) != hipSuccess) { hipFree(d_w); return; }
  if (hipMalloc(&d_y, ((std::size_t) n + LING3_TAIL_PAD) * sizeof(float)) != hipSuccess) { hipFree(d_w); hipFree(d_x); return; }
  for (int c = 0; c < copies; ++c) hipMemcpy(d_w + (std::size_t) c * one, w.data(), one, hipMemcpyHostToDevice);
  std::vector<float> x(k, 0.5f);
  hipMemcpy(d_x, x.data(), (std::size_t) k * sizeof(float), hipMemcpyHostToDevice);
  const int reps = 20;
  const int blocks[3] = {0, 256, 128};
  const char * lbl[3] = {"cta64 (phase 1)", "warp/row block=256", "warp/row block=128"};
  double base = 0.0;
  for (int vi = 0; vi < 3; ++vi) {
    if (blocks[vi] == 0) ling3_quant_q8(d_x, k, LING3_Q8_SLOT_GENERIC);
    auto go = [&](int i) {
      const std::uint8_t * Wp = d_w + (std::size_t)(i % copies) * one;
      if (blocks[vi] == 0) ling3_gemv(qtype, Wp, d_x, d_y, k, n);
      else ling3_gemv_fp4_q8_rows(qtype, Wp, LING3_Q8_SLOT_GENERIC, d_y, k, n, blocks[vi]);
    };
    go(0);
    if (hipDeviceSynchronize() != hipSuccess) { std::printf("  GEMV1 %-22s FAILED\n", lbl[vi]); continue; }
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipEventRecord(a, (hipStream_t) 0);
    for (int i = 0; i < reps; ++i) go(i);
    hipEventRecord(b, (hipStream_t) 0);
    hipEventSynchronize(b);
    float ms = 0.f; hipEventElapsedTime(&ms, a, b); ms /= (float) reps;
    hipEventDestroy(a); hipEventDestroy(b);
    if (vi == 0) base = (double) ms;
    std::printf("  GEMV1 qtype=%3d k=%5d n=%6d  %-22s %8.3f ms  %6.1f GB/s  %5.2fx  (ws %zu MiB/%d)\n",
                qtype, k, n, lbl[vi], ms, (double) one / (ms * 1e-3f) / 1e9,
                ms / (base > 0 ? base : 1.0), one * copies / (1024 * 1024), copies);
  }
  hipFree(d_w); hipFree(d_x); hipFree(d_y);
}

void TimeMoe(int qtype, int n_expert, int n_used, int k, int n_ff) {
  const std::size_t rb_k = RowBytes(qtype, k);
  const std::size_t rb_ff = RowBytes(qtype, n_ff);
  if (rb_k == 0 || rb_ff == 0) return;
  const std::size_t gate_sz = (std::size_t) n_used * (std::size_t) n_ff * rb_k;
  const std::size_t down_sz = (std::size_t) n_used * (std::size_t) k * rb_ff;
  std::vector<std::uint8_t> gate(gate_sz), up(gate_sz), down(down_sz);
  const std::size_t tsz = ling3_type_size(qtype);
  const std::uint8_t smin = 0x2c, srange = 0x18;
  std::vector<int> ids(n_used); std::vector<float> wts(n_used);
  for (int s = 0; s < n_used; ++s) {
    // Expert ids must stay inside the buffers this benchmark actually allocates.
    // The first version used ids[s] = s * (n_expert / n_used) while sizing the
    // tensors for n_used experts, so the kernel -- correctly reading the id the
    // router would have written -- walked off the end of them and took the GPU
    // down with a page fault at 01:58:04Z. Same total bytes, same reads; the
    // ids just have to be ones that exist.
    ids[s] = s;
    wts[s] = 0.1f;
    std::uint8_t * g = gate.data() + (std::size_t) s * (std::size_t) n_ff * rb_k;
    std::uint8_t * u = up.data() + (std::size_t) s * (std::size_t) n_ff * rb_k;
    std::uint8_t * d = down.data() + (std::size_t) s * (std::size_t) k * rb_ff;
    for (int r = 0; r < n_ff; ++r) for (int b = 0; b < k / 32; ++b) {
      const std::size_t o = (std::size_t) r * rb_k + (std::size_t) b * tsz;
      for (int q = 0; q < 16; ++q) { g[o+q] = (std::uint8_t)(Next() & 0xFF); u[o+q] = (std::uint8_t)(Next() & 0xFF); }
      g[o+16] = (std::uint8_t)(smin + Next() % srange); u[o+16] = (std::uint8_t)(smin + Next() % srange);
    }
    for (int r = 0; r < k; ++r) for (int b = 0; b < n_ff / 32; ++b) {
      const std::size_t o = (std::size_t) r * rb_ff + (std::size_t) b * tsz;
      for (int q = 0; q < 16; ++q) d[o+q] = (std::uint8_t)(Next() & 0xFF);
      d[o+16] = (std::uint8_t)(smin + Next() % srange);
    }
  }
  std::vector<float> x(k, 0.5f);
  std::uint8_t * d_gate=nullptr,* d_up=nullptr,* d_down=nullptr; float * d_x=nullptr;
  float * d_y=nullptr; int * d_ids=nullptr; float * d_w=nullptr;
  if (hipMalloc(&d_gate, gate_sz + LING3_TAIL_PAD) != hipSuccess) return;
  if (hipMalloc(&d_up, gate_sz + LING3_TAIL_PAD) != hipSuccess) { hipFree(d_gate); return; }
  if (hipMalloc(&d_down, down_sz + LING3_TAIL_PAD) != hipSuccess) { hipFree(d_gate); hipFree(d_up); return; }
  if (hipMalloc(&d_x, ((std::size_t) k + LING3_TAIL_PAD) * sizeof(float)) != hipSuccess) {
    hipFree(d_gate); hipFree(d_up); hipFree(d_down); return;
  }
  if (hipMalloc(&d_y, ((std::size_t) k + LING3_TAIL_PAD) * sizeof(float)) != hipSuccess) {
    hipFree(d_gate); hipFree(d_up); hipFree(d_down); hipFree(d_x); return;
  }
  if (hipMalloc(&d_ids, (std::size_t) n_used * sizeof(int)) != hipSuccess) {
    hipFree(d_gate); hipFree(d_up); hipFree(d_down); hipFree(d_x); hipFree(d_y); return;
  }
  if (hipMalloc(&d_w, (std::size_t) n_used * sizeof(float)) != hipSuccess) {
    hipFree(d_gate); hipFree(d_up); hipFree(d_down); hipFree(d_x); hipFree(d_y); hipFree(d_ids); return;
  }
  hipMemcpy(d_gate, gate.data(), gate_sz, hipMemcpyHostToDevice);
  hipMemcpy(d_up, up.data(), gate_sz, hipMemcpyHostToDevice);
  hipMemcpy(d_down, down.data(), down_sz, hipMemcpyHostToDevice);
  hipMemcpy(d_x, x.data(), (std::size_t) k * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_ids, ids.data(), (std::size_t) n_used * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(d_w, wts.data(), (std::size_t) n_used * sizeof(float), hipMemcpyHostToDevice);
  const float limit = 7.f;
  const int reps = 20;
  const double down_bytes = (double) n_used * (double) k * (double) rb_ff;
  // One full call first: it quantises hs into the MoE slot, which is exactly
  // what the warp-per-row down kernel reads, so the two can be timed against
  // each other without re-running gate/up inside the timed region.
  ling3_expert_ffn_dev(qtype, d_gate, d_up, d_down, d_x, d_ids, d_w, n_used, n_expert, k, n_ff, limit, d_y);
  hipDeviceSynchronize();
  const int blks[3] = {0, 256, 128};
  const char * lbl[3] = {"full FFN, down cta64 (phase 3)", "full FFN, down warp/row b=256", "full FFN, down warp/row b=128"};
  double base = 0.0;
  for (int vi = 0; vi < 3; ++vi) {
    // Like for like: both variants run the WHOLE grouped FFN (gate + up + SwiGLU +
    // quantise + down). Timing the shipped one against the down kernel alone is
    // how the first version of this benchmark came to claim a 0.26x speedup.
    auto go = [&](int) {
      if (blks[vi] == 0)
        ling3_expert_ffn_dev(qtype, d_gate, d_up, d_down, d_x, d_ids, d_w, n_used, n_expert, k, n_ff, limit, d_y);
      else
        ling3_expert_ffn_dev_tuned(qtype, d_gate, d_up, d_down, d_x, d_ids, d_w,
                                   n_used, n_expert, k, n_ff, limit, d_y, blks[vi]);
    };
    go(0);
    if (hipDeviceSynchronize() != hipSuccess) { std::printf("  MOE %-26s FAILED\n", lbl[vi]); continue; }
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipEventRecord(a, (hipStream_t) 0);
    for (int i = 0; i < reps; ++i) go(i);
    hipEventRecord(b, (hipStream_t) 0);
    hipEventSynchronize(b);
    float ms = 0.f; hipEventElapsedTime(&ms, a, b); ms /= (float) reps;
    hipEventDestroy(a); hipEventDestroy(b);
    const double bytes = (double) n_used * (2.0 * (double) n_ff * (double) rb_k + (double) k * (double) rb_ff);
    if (vi == 0) base = (double) ms;
    std::printf("  MOE   qtype=%3d n_used=%d k=%d n_ff=%d %-26s %8.3f ms  %6.1f GB/s  %5.2fx\n",
                qtype, n_used, k, n_ff, lbl[vi], ms, bytes / (ms * 1e-3f) / 1e9,
                ms / (base > 0 ? base : 1.0));
  }
  hipFree(d_gate); hipFree(d_up); hipFree(d_down); hipFree(d_x); hipFree(d_y); hipFree(d_ids); hipFree(d_w);
}


// ---------------------------------------------------------------------------
// PHASE6 gate: the warp-per-row kernels must produce BIT-IDENTICAL output to the
// CTA-per-row ones they replace. Timing them without this is how you ship a
// kernel that is faster and wrong -- the same failure mode as every other one in
// this lane, where a wrong kernel faults nothing and logs nothing.
// ---------------------------------------------------------------------------
void CheckWarpEqualsCta(int qtype, int k, int n, int block) {
  const std::size_t rb = RowBytes(qtype, k);
  if (rb == 0) return;
  const std::vector<std::uint8_t> w = MakeWeights(qtype, k, n);
  const std::vector<float> x = MakeActivation(k);
  std::uint8_t * d_w = nullptr; float * d_x = nullptr; float * d_a = nullptr; float * d_b = nullptr;
  CHECK(hipMalloc(&d_w, w.size() + LING3_TAIL_PAD) == hipSuccess, "we malloc W");
  CHECK(hipMalloc(&d_x, ((std::size_t) k + LING3_TAIL_PAD) * sizeof(float)) == hipSuccess, "we malloc x");
  CHECK(hipMalloc(&d_a, ((std::size_t) n + LING3_TAIL_PAD) * sizeof(float)) == hipSuccess, "we malloc a");
  CHECK(hipMalloc(&d_b, ((std::size_t) n + LING3_TAIL_PAD) * sizeof(float)) == hipSuccess, "we malloc b");
  if (failures) return;
  hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice);
  hipMemcpy(d_x, x.data(), (std::size_t) k * sizeof(float), hipMemcpyHostToDevice);
  hipMemset(d_a, 0x11, (std::size_t) n * sizeof(float));
  hipMemset(d_b, 0x22, (std::size_t) n * sizeof(float));
  CHECK(ling3_gemv(qtype, d_w, d_x, d_a, k, n) == LING3_OK, "we cta64 rc");
  CHECK(hipDeviceSynchronize() == hipSuccess, "we cta64 sync");
  CHECK(ling3_quant_q8(d_x, k, LING3_Q8_SLOT_GENERIC) == LING3_OK, "we quant");
  CHECK(ling3_gemv_fp4_q8_rows(qtype, d_w, LING3_Q8_SLOT_GENERIC, d_b, k, n, block) == LING3_OK, "we rows rc");
  CHECK(hipDeviceSynchronize() == hipSuccess, "we rows sync");
  std::vector<float> a(n), b(n);
  hipMemcpy(a.data(), d_a, (std::size_t) n * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(b.data(), d_b, (std::size_t) n * sizeof(float), hipMemcpyDeviceToHost);
  // It CANNOT be bit-identical: the two kernels hand different lanes different sets of
  // 32-value blocks (cta64 strides by 64, warp-per-row by 32), so the float additions
  // that sum a row's blocks are grouped differently and round differently. Float
  // addition is not associative. So the gate is not "identical" but "no worse than the
  // kernel it replaces, measured against the SAME CPU integer reference": both must be
  // inside the phase-1 bound, and the warp form must not be materially further from it.
  double sumsq = 0.0;
  for (int r = 0; r < n; ++r) sumsq += (double) a[r] * (double) a[r];
  const double scale = (sumsq > 1e-12) ? std::sqrt(sumsq / n) : 1.0;
  double ea = 0.0, eb = 0.0;
  int diff = 0;
  for (int i = 0; i < n; ++i) {
    const float ref = RefInteger(qtype, w.data() + (std::size_t) i * rb, x.data(), k, nullptr, nullptr);
    ea = std::max(ea, std::fabs((double) a[i] - (double) ref) / scale);
    eb = std::max(eb, std::fabs((double) b[i] - (double) ref) / scale);
    if (a[i] != b[i]) ++diff;
  }
  CHECK(eb < 1e-5, "warp-per-row GEMV is outside the phase-1 rms bound vs the CPU reference");
  CHECK(eb <= 4.0 * ea + 1e-9, "warp-per-row GEMV is materially further from the CPU reference");
  std::printf("  WARP vs CTA qtype=%3d k=%5d n=%6d block=%3d  rms_err cta=%.3g warp=%.3g  "
              "differing=%d/%d (float regrouping, expected)\n",
              qtype, k, n, block, ea, eb, diff, n);
  hipFree(d_w); hipFree(d_x); hipFree(d_a); hipFree(d_b);
}

// Same, for the grouped expert FFN: full path with the warp-per-row down kernel
// against the shipped CTA-per-row path, at Ling's real shape.
void CheckWarpMoeEqualsCta(int qtype, int n_expert, int n_used, int k, int n_ff, int block) {
  const std::size_t rb_k = RowBytes(qtype, k);
  const std::size_t rb_ff = RowBytes(qtype, n_ff);
  if (rb_k == 0 || rb_ff == 0) return;
  std::vector<int> ids(n_used); std::vector<float> wts(n_used);
  for (int s = 0; s < n_used; ++s) { ids[s] = s; wts[s] = 0.05f + 0.2f * (float) Next() / 4294967296.0F; }
  const std::size_t gate_sz = (std::size_t) n_used * (std::size_t) n_ff * rb_k;
  const std::size_t down_sz = (std::size_t) n_used * (std::size_t) k * rb_ff;
  const std::size_t tsz = ling3_type_size(qtype);
  std::vector<std::uint8_t> gate(gate_sz, 0), up(gate_sz, 0), down(down_sz, 0);
  const std::uint8_t smin = 0x2c, srange = 0x18;
  for (int s = 0; s < n_used; ++s) {
    std::uint8_t * g = gate.data() + (std::size_t) s * (std::size_t) n_ff * rb_k;
    std::uint8_t * u = up.data() + (std::size_t) s * (std::size_t) n_ff * rb_k;
    std::uint8_t * d = down.data() + (std::size_t) s * (std::size_t) k * rb_ff;
    for (int r = 0; r < n_ff; ++r) for (int b = 0; b < k / 32; ++b) {
      const std::size_t o = (std::size_t) r * rb_k + (std::size_t) b * tsz;
      for (int q = 0; q < 16; ++q) { g[o+q] = (std::uint8_t)(Next() & 0xFF); u[o+q] = (std::uint8_t)(Next() & 0xFF); }
      g[o+16] = (std::uint8_t)(smin + Next() % srange); u[o+16] = (std::uint8_t)(smin + Next() % srange);
    }
    for (int r = 0; r < k; ++r) for (int b = 0; b < n_ff / 32; ++b) {
      const std::size_t o = (std::size_t) r * rb_ff + (std::size_t) b * tsz;
      for (int q = 0; q < 16; ++q) d[o+q] = (std::uint8_t)(Next() & 0xFF);
      d[o+16] = (std::uint8_t)(smin + Next() % srange);
    }
  }
  std::vector<float> x((std::size_t) k);
  for (int i = 0; i < k; ++i) x[i] = -3.f + 6.f * (float) Next() / 4294967296.0F;
  std::uint8_t * d_gate=nullptr,* d_up=nullptr,* d_down=nullptr; float * d_x=nullptr;
  float * d_a=nullptr; float * d_b=nullptr; int * d_ids=nullptr; float * d_w=nullptr;
  CHECK(hipMalloc(&d_gate, gate_sz + LING3_TAIL_PAD) == hipSuccess, "wm gate");
  CHECK(hipMalloc(&d_up, gate_sz + LING3_TAIL_PAD) == hipSuccess, "wm up");
  CHECK(hipMalloc(&d_down, down_sz + LING3_TAIL_PAD) == hipSuccess, "wm down");
  CHECK(hipMalloc(&d_x, ((std::size_t) k + LING3_TAIL_PAD) * sizeof(float)) == hipSuccess, "wm x");
  CHECK(hipMalloc(&d_a, ((std::size_t) k + LING3_TAIL_PAD) * sizeof(float)) == hipSuccess, "wm a");
  CHECK(hipMalloc(&d_b, ((std::size_t) k + LING3_TAIL_PAD) * sizeof(float)) == hipSuccess, "wm b");
  CHECK(hipMalloc(&d_ids, (std::size_t) n_used * sizeof(int)) == hipSuccess, "wm ids");
  CHECK(hipMalloc(&d_w, (std::size_t) n_used * sizeof(float)) == hipSuccess, "wm w");
  if (failures) return;
  hipMemcpy(d_gate, gate.data(), gate_sz, hipMemcpyHostToDevice);
  hipMemcpy(d_up, up.data(), gate_sz, hipMemcpyHostToDevice);
  hipMemcpy(d_down, down.data(), down_sz, hipMemcpyHostToDevice);
  hipMemcpy(d_x, x.data(), (std::size_t) k * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_ids, ids.data(), (std::size_t) n_used * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(d_w, wts.data(), (std::size_t) n_used * sizeof(float), hipMemcpyHostToDevice);
  const float limit = 7.f;
  hipMemset(d_a, 0x11, (std::size_t) k * sizeof(float));
  hipMemset(d_b, 0x22, (std::size_t) k * sizeof(float));
  CHECK(ling3_expert_ffn_dev(qtype, d_gate, d_up, d_down, d_x, d_ids, d_w,
                             n_used, n_expert, k, n_ff, limit, d_a) == LING3_OK, "wm cta rc");
  CHECK(hipDeviceSynchronize() == hipSuccess, "wm cta sync");
  CHECK(ling3_expert_ffn_m_check() == LING3_OK, "wm cta status");
  CHECK(ling3_expert_ffn_dev_tuned(qtype, d_gate, d_up, d_down, d_x, d_ids, d_w,
                                   n_used, n_expert, k, n_ff, limit, d_b, block) == LING3_OK, "wm rows rc");
  CHECK(hipDeviceSynchronize() == hipSuccess, "wm rows sync");
  CHECK(ling3_expert_ffn_m_check() == LING3_OK, "wm rows status");
  std::vector<float> a(k), b(k);
  hipMemcpy(a.data(), d_a, (std::size_t) k * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(b.data(), d_b, (std::size_t) k * sizeof(float), hipMemcpyDeviceToHost);
  int diff = 0;
  for (int i = 0; i < k; ++i) if (a[i] != b[i]) ++diff;
  // Same reasoning as the GEMV check: different lane->block grouping, so different
  // float rounding. What has to hold is that neither form drifts from the shipped one.
  double sumsq = 0.0;
  for (int i = 0; i < k; ++i) sumsq += (double) a[i] * (double) a[i];
  const double scale = (sumsq > 1e-12) ? std::sqrt(sumsq / k) : 1.0;
  double mx = 0.0;
  for (int i = 0; i < k; ++i) mx = std::max(mx, std::fabs((double) a[i] - (double) b[i]) / scale);
  CHECK(mx < 1e-5, "warp-per-row expert FFN drifts from the shipped one beyond the rms bound");
  std::printf("  WARPMOE vs CTA qtype=%3d n_used=%d k=%d n_ff=%d block=%3d  max_rel_diff=%.3g  "
              "differing=%d/%d\n", qtype, n_used, k, n_ff, block, mx, diff, k);
  hipFree(d_gate); hipFree(d_up); hipFree(d_down); hipFree(d_x);
  hipFree(d_a); hipFree(d_b); hipFree(d_ids); hipFree(d_w);
}

void RunShape(int qtype, int k, int n) {
  const std::vector<std::uint8_t> w = MakeWeights(qtype, k, n);
  const std::vector<float> x = MakeActivation(k);

  std::uint8_t * d_w = nullptr;
  float * d_x = nullptr;
  float * d_y = nullptr;
  const std::size_t wbytes = w.size();
  // +LING3_TAIL_PAD: the contract every weight buffer in the engine has.
  const int before = failures;
  CHECK(hipMalloc(&d_w, wbytes + LING3_TAIL_PAD) == hipSuccess, "hipMalloc W");
  CHECK(hipMalloc(&d_x, ((std::size_t) k + LING3_TAIL_PAD) * sizeof(float)) == hipSuccess, "hipMalloc x");
  CHECK(hipMalloc(&d_y, ((std::size_t) n + LING3_TAIL_PAD) * sizeof(float)) == hipSuccess, "hipMalloc y");
  // Bail on THIS shape only: a run of shapes after a failure is still worth having.
  if (failures != before) return;
  CHECK(hipMemcpy(d_w, w.data(), wbytes, hipMemcpyHostToDevice) == hipSuccess, "H2D W");
  CHECK(hipMemcpy(d_x, x.data(), (std::size_t) k * sizeof(float), hipMemcpyHostToDevice) == hipSuccess, "H2D x");
  // Poison y, so a row nobody writes is obvious rather than silently zero.
  CHECK(hipMemset(d_y, 0x7f, (std::size_t) n * sizeof(float)) == hipSuccess, "memset y");

  const int rc = ling3_gemv(qtype, d_w, d_x, d_y, k, n);
  CHECK(rc == LING3_OK, "ling3_gemv rc");
  CHECK(hipDeviceSynchronize() == hipSuccess, "sync");

  std::vector<float> y(n);
  CHECK(hipMemcpy(y.data(), d_y, (std::size_t) n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess, "D2H y");

  // Pull the activation back the way the GEMV quantised it, so a mismatch can be
  // attributed to the quantiser or to the dot instead of guessed at.
  std::vector<std::int8_t> gq(k);
  std::vector<float> gd(k / 32);
  std::vector<std::int8_t> cq(k);
  std::vector<float> cd(k / 32);
  RefInteger(qtype, w.data(), x.data(), k, &cq, &cd);
  int qdiff = 0, ddiff = 0;
  if (ling3_q8_peek(LING3_Q8_SLOT_GENERIC, gq.data(), gd.data(), k) == LING3_OK) {
    for (int i = 0; i < k; ++i) if (gq[i] != cq[i]) ++qdiff;
    for (int i = 0; i < k / 32; ++i) if (gd[i] != cd[i]) ++ddiff;
  }
  // Was the activation still intact in device memory when the quantiser ran?
  std::vector<float> xback(k);
  int xdiff = -1;
  if (hipMemcpy(xback.data(), d_x, (std::size_t) k * sizeof(float),
                hipMemcpyDeviceToHost) == hipSuccess) {
    xdiff = 0;
    for (int i = 0; i < k; ++i) if (xback[i] != x[i]) ++xdiff;
  }

  const std::size_t rb = RowBytes(qtype, k);
  std::vector<float> ri(n), rf(n);
  double sumsq = 0.0;
  for (int r = 0; r < n; ++r) {
    const std::uint8_t * wrow = w.data() + (std::size_t) r * rb;
    ri[r] = RefInteger(qtype, wrow, x.data(), k, nullptr, nullptr);
    rf[r] = RefFloatExact(qtype, wrow, x.data(), k);
    sumsq += (double) rf[r] * (double) rf[r];
  }
  const double rms = std::sqrt(sumsq / (n > 0 ? n : 1));
  const double scale = rms > 1e-6 ? rms : 1.0;

  double worst_int = 0.0, worst_float = 0.0, worst_rel = 0.0, ds_rel = 0.0;
  for (int r = 0; r < n; ++r) {
    const double ei = std::fabs((double) y[r] - (double) ri[r]) / scale;
    const double ef = std::fabs((double) y[r] - (double) rf[r]) / scale;
    if (std::fabs((double) rf[r]) > 0.05 * scale) {
      const double er = std::fabs((double) y[r] - (double) rf[r]) / std::fabs((double) rf[r]);
      if (er > worst_rel) worst_rel = er;
    }
    if (ei > worst_int) worst_int = ei;
    if (ef > worst_float) worst_float = ef;
  }
  for (int b = 0; b < k / 32; ++b) {
    const double dd = std::fabs((double) gd[b] - (double) cd[b]) / scale;
    if (dd > ds_rel) ds_rel = dd;
  }
  CHECK(worst_int < 1e-5, "kernel != CPU integer reference (rms-normalised)");
  // This number is NOT an indexing check -- it is the price of the activation
  // quantisation, and it is the numerical change this lane deliberately makes (it is
  // llama.cpp's computation, so it moves gufo toward the reference, not away). It
  // measures around 0.012 to 0.027 of rms for k = 2560, because an FP4 block's ue4m3
  // scale reaches 240 and one int8 step of the activation is therefore worth up to ~27
  // on a single term, which over 2560 terms random-signs to ~0.02 of rms. The gate is
  // 6e-2: an indexing or scale bug moves this by three orders of magnitude (it read
  // 3.8e3 and 1.3e2 with .x where .y belonged), so the threshold is not doing the
  // correctness work -- the vs_int check above is.
  CHECK(worst_float < 6e-2, "kernel too far from the float reference (rms-normalised)");

  std::printf("  qtype=%3d k=%6d n=%6d  rowbytes=%7zu  err/rms: vs_int=%.3g vs_float=%.3g  "
              "rel(signal rows)=%.3g  rms=%.4g  ds_rel=%.3g  q8_diff=%d/%d ds_diff=%d x_diff=%d\n",
              qtype, k, n, rb, worst_int, worst_float, worst_rel, rms, ds_rel,
              qdiff, k, ddiff, xdiff);

  hipFree(d_w);
  hipFree(d_x);
  hipFree(d_y);
}


void CheckRouteFused(int k, int n_expert, int n_tokens) {
  std::vector<float> x((std::size_t) n_tokens * (std::size_t) k);
  std::vector<float> gate((std::size_t) n_expert * (std::size_t) k);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = -2.f + 4.f * (float) Next() / 4294967296.0F;
  }
  for (std::size_t i = 0; i < gate.size(); ++i) {
    gate[i] = -1.f + 2.f * (float) Next() / 4294967296.0F;
  }
  for (int t = 0; t < n_tokens; ++t) x[(std::size_t) t * k] = 0.f;
  float *dx = nullptr, *dg = nullptr, *dr = nullptr, *df = nullptr;
  const int before = failures;
  const std::size_t nx = x.size() * sizeof(float);
  const std::size_t ng = gate.size() * sizeof(float);
  const std::size_t nl = (std::size_t) n_tokens * (std::size_t) n_expert * sizeof(float);
  CHECK(hipMalloc(&dx, nx + LING3_TAIL_PAD) == hipSuccess, "lv1 route x");
  CHECK(hipMalloc(&dg, ng + LING3_TAIL_PAD) == hipSuccess, "lv1 route g");
  CHECK(hipMalloc(&dr, nl + LING3_TAIL_PAD) == hipSuccess, "lv1 route r");
  CHECK(hipMalloc(&df, nl + LING3_TAIL_PAD) == hipSuccess, "lv1 route f");
  if (failures != before) return;
  hipMemcpy(dx, x.data(), nx, hipMemcpyHostToDevice);
  hipMemcpy(dg, gate.data(), ng, hipMemcpyHostToDevice);
  CHECK(ling3_route_logits_pair(dx, dg, k, n_expert, n_tokens, dr, df) == LING3_OK,
        "lv1 route launch");
  CHECK(hipDeviceSynchronize() == hipSuccess, "lv1 route sync");
  std::vector<float> href(nl / sizeof(float)), hfused(nl / sizeof(float));
  hipMemcpy(href.data(), dr, nl, hipMemcpyDeviceToHost);
  hipMemcpy(hfused.data(), df, nl, hipMemcpyDeviceToHost);
  int bad = 0;
  if (std::memcmp(href.data(), hfused.data(), nl) != 0) {
    for (std::size_t i = 0; i < href.size(); ++i) {
      std::uint32_t a, b;
      std::memcpy(&a, &href[i], 4);
      std::memcpy(&b, &hfused[i], 4);
      if (a != b) ++bad;
    }
  }
  CHECK(bad == 0, "LV1 router fused logits are not bit-identical");
  std::printf("  ROUTE k=%d n_expert=%d m=%d  bit-exact=%s  mismatching=%d/%zu\n",
              k, n_expert, n_tokens, bad == 0 ? "yes" : "NO", bad, href.size());
  hipFree(dx); hipFree(dg); hipFree(dr); hipFree(df);
}

void CheckSharedEqualsM(int qtype, int n_expert, int id, int n_tokens, int k, int n_ff,
                        float limit, bool ones) {
  const std::size_t rb_k = RowBytes(qtype, k);
  const std::size_t rb_ff = RowBytes(qtype, n_ff);
  if (rb_k == 0 || rb_ff == 0 || id < 0 || id >= n_expert) return;
  const std::vector<std::uint8_t> gate = MakeWeights(qtype, k, n_expert * n_ff);
  const std::vector<std::uint8_t> up = MakeWeights(qtype, k, n_expert * n_ff);
  const std::vector<std::uint8_t> down = MakeWeights(qtype, n_ff, n_expert * k);
  std::vector<float> x((std::size_t) n_tokens * (std::size_t) k);
  for (int t = 0; t < n_tokens; ++t) {
    const std::vector<float> col = MakeActivation(k);
    std::memcpy(x.data() + (std::size_t) t * k, col.data(), (std::size_t) k * sizeof(float));
  }
  std::vector<int> ids((std::size_t) n_tokens, id);
  std::vector<float> wts((std::size_t) n_tokens, 1.f);
  if (!ones) {
    for (int t = 0; t < n_tokens; ++t) wts[(std::size_t) t] = 0.25f + 0.1f * (float) t;
  }
  std::uint8_t *d_gate = nullptr, *d_up = nullptr, *d_down = nullptr;
  float *d_x = nullptr, *d_w = nullptr, *d_y0 = nullptr, *d_y1 = nullptr;
  int *d_ids = nullptr;
  const int before = failures;
  CHECK(hipMalloc(&d_gate, gate.size() + LING3_TAIL_PAD) == hipSuccess, "lv1 sh gate");
  CHECK(hipMalloc(&d_up, up.size() + LING3_TAIL_PAD) == hipSuccess, "lv1 sh up");
  CHECK(hipMalloc(&d_down, down.size() + LING3_TAIL_PAD) == hipSuccess, "lv1 sh down");
  CHECK(hipMalloc(&d_x, x.size() * sizeof(float) + LING3_TAIL_PAD) == hipSuccess, "lv1 sh x");
  CHECK(hipMalloc(&d_ids, ids.size() * sizeof(int)) == hipSuccess, "lv1 sh ids");
  CHECK(hipMalloc(&d_w, wts.size() * sizeof(float)) == hipSuccess, "lv1 sh w");
  const std::size_t ny = (std::size_t) n_tokens * (std::size_t) k * sizeof(float);
  CHECK(hipMalloc(&d_y0, ny + LING3_TAIL_PAD) == hipSuccess, "lv1 sh y0");
  CHECK(hipMalloc(&d_y1, ny + LING3_TAIL_PAD) == hipSuccess, "lv1 sh y1");
  if (failures != before) return;
  hipMemcpy(d_gate, gate.data(), gate.size(), hipMemcpyHostToDevice);
  hipMemcpy(d_up, up.data(), up.size(), hipMemcpyHostToDevice);
  hipMemcpy(d_down, down.data(), down.size(), hipMemcpyHostToDevice);
  hipMemcpy(d_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_ids, ids.data(), ids.size() * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(d_w, wts.data(), wts.size() * sizeof(float), hipMemcpyHostToDevice);
  CHECK(ling3_expert_ffn_m(qtype, d_gate, d_up, d_down, d_x, d_ids, d_w, n_tokens, 1,
                           n_expert, k, n_ff, limit, d_y0) == LING3_OK, "lv1 sh old rc");
  CHECK(hipDeviceSynchronize() == hipSuccess, "lv1 sh old sync");
  CHECK(ling3_expert_ffn_m_check() == LING3_OK, "lv1 sh old status");
  CHECK(ling3_expert_ffn_shared_m(qtype, d_gate, d_up, d_down, d_x, d_ids, d_w, n_tokens, 1,
                                  n_expert, k, n_ff, limit, d_y1) == LING3_OK, "lv1 sh new rc");
  CHECK(hipDeviceSynchronize() == hipSuccess, "lv1 sh new sync");
  CHECK(ling3_expert_ffn_m_check() == LING3_OK, "lv1 sh new status");
  std::vector<float> a(ny / sizeof(float)), b(ny / sizeof(float)), c(ny / sizeof(float));
  hipMemcpy(a.data(), d_y0, ny, hipMemcpyDeviceToHost);
  hipMemcpy(b.data(), d_y1, ny, hipMemcpyDeviceToHost);
  CHECK(ling3_expert_ffn_shared_m(qtype, d_gate, d_up, d_down, d_x, d_ids, d_w, n_tokens, 1,
                                  n_expert, k, n_ff, limit, d_y1) == LING3_OK, "lv1 sh new rc2");
  CHECK(hipDeviceSynchronize() == hipSuccess, "lv1 sh new sync2");
  hipMemcpy(c.data(), d_y1, ny, hipMemcpyDeviceToHost);
  int bad = 0, nondet = 0;
  if (std::memcmp(a.data(), b.data(), ny) != 0) {
    for (std::size_t i = 0; i < a.size(); ++i) {
      std::uint32_t ua, ub;
      std::memcpy(&ua, &a[i], 4);
      std::memcpy(&ub, &b[i], 4);
      if (ua != ub) ++bad;
    }
  }
  if (std::memcmp(b.data(), c.data(), ny) != 0) {
    for (std::size_t i = 0; i < b.size(); ++i) if (std::memcmp(&b[i], &c[i], 4) != 0) ++nondet;
  }
  CHECK(bad == 0, "LV1 shared expert is not bit-identical to expert_ffn_m");
  CHECK(nondet == 0, "LV1 shared expert is not deterministic");
  std::printf("  SHARED qtype=%d n_expert=%d id=%d m=%d k=%d n_ff=%d limit=%.1f ones=%d  "
              "bit-exact=%s  mismatching=%d/%zu  nondet=%d\n",
              qtype, n_expert, id, n_tokens, k, n_ff, limit, ones ? 1 : 0,
              bad == 0 ? "yes" : "NO", bad, a.size(), nondet);
  hipFree(d_gate); hipFree(d_up); hipFree(d_down); hipFree(d_x);
  hipFree(d_ids); hipFree(d_w); hipFree(d_y0); hipFree(d_y1);
}

}  // namespace

int main() {
  std::printf("== codebook: all 65536 four-nibble words vs ling3_rocmfp4_code\n");
  const int bad = ling3_gemv_q8_codebook_check();
  CHECK(bad == 0, "codebook perm disagrees with ling3_rocmfp4_code");
  std::printf("  mismatching byte lanes: %d\n", bad);
  if (bad != 0) {
    std::printf("FAIL: the codebook is wrong; nothing below means anything\n");
    return 1;
  }

  std::printf("== LV1 single-read router and shared/dense expert, smallest shapes first\n");
  CheckRouteFused(32, 3, 2);
  CheckRouteFused(96, 7, 3);
  CheckRouteFused(64, 8, 4);
  CheckSharedEqualsM(LING3_Q4_0_ROCMFP4_FAST, 3, 2, 2, 32, 32, 7.f, false);
  CheckSharedEqualsM(LING3_Q4_0_ROCMFP4_FAST, 3, 2, 4, 96, 64, 7.f, false);
  CheckSharedEqualsM(LING3_Q4_0_ROCMFP4_FAST, 3, 1, 3, 96, 64, 0.f, false);
  CheckSharedEqualsM(LING3_Q4_0_ROCMFP4, 2, 1, 2, 64, 64, 7.f, true);
  if (failures != 0) {
    std::printf("FAIL: %d LV1 small-shape check(s)\n", failures);
    return 1;
  }
  CheckRouteFused(2560, 512, 2);
  CheckRouteFused(2560, 512, 4);
  CheckSharedEqualsM(LING3_Q4_0_ROCMFP4_FAST, 2, 1, 2, 2560, 768, 7.f, true);
  CheckSharedEqualsM(LING3_Q4_0_ROCMFP4_FAST, 1, 0, 4, 2560, 768, 7.f, false);
  CheckSharedEqualsM(LING3_Q4_0_ROCMFP4_FAST, 1, 0, 2, 2560, 6144, 7.f, true);
  if (failures != 0) {
    std::printf("FAIL: %d LV1 full-shape check(s)\n", failures);
    return 1;
  }

  std::printf("== GEMV shapes (synthetic ROCmFP4 rows, all sixteen codes per block)\n");
  // The shapes Ling-3.0-flash actually decodes, from the GGUF tensor shapes:
  //   expert gate/up  [768, 2560]   expert down    [2560, 768]
  //   attn q/k/v/f/g  [4096, 2560]  attn_output    [2560, 4096]
  //   dense ffn       [6144, 2560]  down           [2560, 6144]
  //   MLA q [6144, 2560], kv_a [576, 2560], gate [32, 2560], LM head [157184, 2560]
  struct Shape { int qtype, k, n; };
  const Shape shapes[] = {
      // smallest first, so an index bug shows up before the big shapes run
      {101, 32, 1}, {101, 32, 17}, {101, 32, 64}, {100, 32, 1}, {100, 32, 33},
      {101, 64, 5},  {100, 64, 5},
      {101, 2560, 768}, {100, 2560, 768},
      {101, 768, 2560}, {100, 768, 1024},
      {101, 2560, 4096},
      {101, 2560, 2560},
      {101, 2560, 6144},
      {101, 2560, 6145},
      // above LS1_GEMV_WIDE_MAX, so the grid-strided kernel and its red[] reuse across
      // rows are exercised too
      {101, 2560, 8193},
      {100, 2560, 8192},
      {101, 2560, 16384},
  };
  for (const Shape & s : shapes) {
    RunShape(s.qtype, s.k, s.n);
  }

  std::printf("== small-m decode forward (MTP verify): per column vs the m=1 path\n");
  // the shapes Ling decodes, plus the expert shapes
  RunSmallM(LING3_Q4_0_ROCMFP4_FAST, 2560, 768, true);    // expert gate/up
  RunSmallM(LING3_Q4_0_ROCMFP4_FAST, 2560, 4096, true);   // KDA attn q/k/v/f/g
  RunSmallM(LING3_Q4_0_ROCMFP4_FAST, 2560, 6144, true);   // dense ffn gate/up
  RunSmallM(LING3_Q4_0_ROCMFP4_FAST, 2560, 16384, true);  // LM head
  RunSmallM(LING3_Q4_0_ROCMFP4,      2560, 2560, true);   // a type-100 matrix
  RunSmallM(LING3_Q4_0_ROCMFP4_FAST,  768, 2560, true);   // expert down

  std::printf("== PHASE6 gate: the warp-per-row kernels must equal the CTA-per-row ones\n");
  CheckWarpEqualsCta(LING3_Q4_0_ROCMFP4_FAST, 2560, 768, 256);
  CheckWarpEqualsCta(LING3_Q4_0_ROCMFP4_FAST, 2560, 4096, 256);
  CheckWarpEqualsCta(LING3_Q4_0_ROCMFP4_FAST, 2560, 4096, 128);
  CheckWarpEqualsCta(LING3_Q4_0_ROCMFP4_FAST, 2560, 16384, 256);
  CheckWarpEqualsCta(LING3_Q4_0_ROCMFP4,      2560, 2560, 256);
  CheckWarpEqualsCta(LING3_Q4_0_ROCMFP4_FAST,  768, 2560, 256);
  CheckWarpEqualsCta(LING3_Q4_0_ROCMFP4_FAST,   32,   64, 256);
  CheckWarpMoeEqualsCta(LING3_Q4_0_ROCMFP4_FAST, 512, 8, 2560, 768, 256);
  CheckWarpMoeEqualsCta(LING3_Q4_0_ROCMFP4_FAST, 512, 1, 2560, 768, 256);
  CheckWarpMoeEqualsCta(LING3_Q4_0_ROCMFP4,      512, 4, 2560, 768, 128);

  std::printf("== PHASE6 tuning targets: MALL-cold bandwidth of the two weak GEMVs\n");
  TimeGemvM1(LING3_Q4_0_ROCMFP4_FAST, 2560, 4096);   // attn projections, ~399 calls/token
  TimeGemvM1(LING3_Q4_0_ROCMFP4_FAST, 2560, 6144);   // dense ffn
  TimeMoe(LING3_Q4_0_ROCMFP4_FAST, 512, 8, 2560, 768);
  TimeMoe(LING3_Q4_0_ROCMFP4_FAST, 512, 1, 2560, 768);

  std::printf("== small-m timings: target m=3 <= 1.4x m=1\n");
  TimeSmallM(LING3_Q4_0_ROCMFP4_FAST, 2560, 4096);    // KDA attn q/k/v/f/g, 5.6 MB each
  TimeSmallM(LING3_Q4_0_ROCMFP4_FAST, 2560, 16384);   // dense ffn gate/up, 22 MB each
  TimeSmallM(LING3_Q4_0_ROCMFP4_FAST, 2560, 157184);  // the LM head, 214 MB -- over the MALL unaided

  std::printf("== small-m grouped expert FFN over (token x slot) pairs\n");
  RunSmallMExperts(LING3_Q4_0_ROCMFP4_FAST, 512, 1, 4, 2560, 768);
  RunSmallMExperts(LING3_Q4_0_ROCMFP4_FAST, 512, 8, 2, 2560, 768);
  RunSmallMExperts(LING3_Q4_0_ROCMFP4_FAST, 512, 8, 3, 2560, 768);
  RunSmallMExperts(LING3_Q4_0_ROCMFP4_FAST, 512, 8, 4, 2560, 768);
  RunSmallMExperts(LING3_Q4_0_ROCMFP4,      512, 8, 3, 2560, 768);

  std::printf("== grouped device-routed expert FFN vs the per-expert host loop\n");
  std::printf("   (n_used=1 and n_used=8 at the SAME real shape: the phase-2 bug was a 32-lane\n"
              "    reduction in a 64-thread CTA, which is invisible until the row is long enough\n"
              "    to put work on lanes 32-63)\n");
  RunGroupedExperts(LING3_Q4_0_ROCMFP4_FAST, 512, 1, 2560, 768, true);
  RunGroupedExperts(LING3_Q4_0_ROCMFP4_FAST, 512, 8, 2560, 768, true);
  RunGroupedExperts(LING3_Q4_0_ROCMFP4,      512, 8, 2560, 768, false);
  RunGroupedExperts(LING3_Q4_0_ROCMFP4_FAST, 512, 4, 2560, 768, false);
  RunGroupedExperts(LING3_Q4_0_ROCMFP4_FAST,  64, 2, 2560, 768, false);
  // the shape that hid the bug in phase 2: short rows, all work on lanes 0-15
  RunGroupedExperts(LING3_Q4_0_ROCMFP4_FAST,  16, 8,  512, 256, false);

  if (failures != 0) {
    std::printf("FAIL: %d check(s)\n", failures);
    return 1;
  }
  std::printf("PASS: codebook exact, GEMV matches the CPU integer reference, the grouped\n"
              "expert FFN matches the per-expert host loop, and the small-m forward is\n"
              "bit-exact per column against the m=1 path -- all deterministic\n");
  return 0;
}