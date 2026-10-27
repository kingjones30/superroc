// superroc-quant ROCmFP4 encoder (GGML types 100/101), ported verbatim from the
// ROCmFPX fork's `ggml/rocmfp4/rocmfp4.{h,c}` (MIT, "The ggml authors"), so
// superroc-quant reproduces the fork's `llama-quantize` ROC output byte for byte.
// The encode rule: exhaustive UE4M3 half-scale MSE search over candidate scale
// bytes, then nearest-level codebook on a 10-capped E2M1-derived grid.
//
// SPEC3 / RV28 stage 1 adds an imatrix-weighted path (the fork's own weighted
// convention, adopted verbatim) and a joint scale+code post-pass that improves
// the weighted output deterministically. With no --imatrix and the post-pass
// off, output is byte-identical to the fork (v0.2 baseline).

#ifndef SUPERROC_QUANT_ROCMFP4_ENCODE_HPP_
#define SUPERROC_QUANT_ROCMFP4_ENCODE_HPP_

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "vendored_quantize.hpp"  // provides ggml_compute_fp32_to_fp16 etc. (unused here), MIN/MAX

namespace superroc::quant_tool {

inline constexpr std::uint32_t kQkRocmFp4 = 32;

struct block_rocmfp4 {      // type 100, 18 bytes
  std::uint8_t qs[16];
  std::uint8_t e[2];
};
struct block_rocmfp4_fast { // type 101, 17 bytes
  std::uint8_t qs[16];
  std::uint8_t e;
};
static_assert(sizeof(block_rocmfp4) == 18, "block_rocmfp4 size");
static_assert(sizeof(block_rocmfp4_fast) == 17, "block_rocmfp4_fast size");

// UE4M3 half-scale table (rocmfp4.c lines 79-135). Bytes >= 0x7f decode to 0.
// E_m = (8+M)*0x1p{-(11-m)} = (8+M)*0x1p(m-11). SUB(M) = M*0x1p-10.
inline const std::array<float, 127> kScaleHalf = [] {
  std::array<float, 127> t{};
  for (int m = 0; m < 8; ++m) t[m] = (float)m * 0x1p-10f;   // SUB, indices 0..7
  int idx = 8;
  for (int e = 1; e <= 14; ++e)                              // E1..E14, 8 each
    for (int m = 0; m < 8; ++m) t[idx++] = (8.0f + m) * ldexpf(1.0f, (e - 11));
  for (int m = 0; m < 7; ++m) t[idx++] = (8.0f + m) * ldexpf(1.0f, 4);  // E15, 7
  return t;
}();

[[nodiscard]] inline float ScaleFromByte(std::uint8_t e) noexcept {
  return e <= 0x7e ? kScaleHalf[e] : 0.0f;
}

// Best code index for x under 1/scale. Ported from
// rocmfp4_best_index_scaled (rocmfp4.c 163-189). Ties -> lower magnitude.
[[nodiscard]] inline std::uint8_t BestIndexScaled(float x, float inv_scale_half) noexcept {
  if (!(x == x) || x > 3.0e38f || x < -3.0e38f) return 0;  // NaN/Inf -> 0
  const float a = (x < 0 ? -x : x) * inv_scale_half;
  if (a <= 0.5f) return 0;
  const bool neg = x < 0.0f;
  if (a <= 1.5f) return neg ? 9 : 1;
  if (a <= 2.5f) return neg ? 10 : 2;
  if (a <= 3.5f) return neg ? 11 : 3;
  if (a <= 5.0f) return neg ? 12 : 4;
  if (a <= 7.0f) return neg ? 13 : 5;
  if (a <= 9.0f) return neg ? 14 : 6;
  return neg ? 15 : 7;
}

// Signed decoded magnitude (rocmfp4_decoded_mag_scaled). Used in MSE search.
[[nodiscard]] inline float DecodedMagScaled(float x, float inv_scale_half) noexcept {
  if (!(x == x) || x > 3.0e38f || x < -3.0e38f) return 0.0f;
  const float a = (x < 0 ? -x : x) * inv_scale_half;
  float mag;
  if (a <= 0.5f) mag = 0.0f;
  else if (a <= 1.5f) mag = 1.0f;
  else if (a <= 2.5f) mag = 2.0f;
  else if (a <= 3.5f) mag = 3.0f;
  else if (a <= 5.0f) mag = 4.0f;
  else if (a <= 7.0f) mag = 6.0f;
  else if (a <= 9.0f) mag = 8.0f;
  else mag = 10.0f;
  return x < 0.0f ? -mag : mag;
}

[[nodiscard]] inline int NearestScaleUe4M3(float target_scale_half) noexcept {
  if (!(target_scale_half > 0.0f) || !(target_scale_half == target_scale_half) ||
      target_scale_half > 1e38f)
    return 1;
  int lo = 1, hi = 126;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (ScaleFromByte((std::uint8_t)mid) < target_scale_half) lo = mid + 1;
    else hi = mid;
  }
  if (lo == 1) return 1;
  const float hi_s = ScaleFromByte((std::uint8_t)lo);
  const float lo_s = ScaleFromByte((std::uint8_t)(lo - 1));
  return (target_scale_half - lo_s <= hi_s - target_scale_half) ? lo - 1 : lo;
}

inline float BlockMseForScale(const float* x, int n, int e, float best_err) noexcept {
  const float scale_half = ScaleFromByte((std::uint8_t)e);
  const float inv_scale_half = 1.0f / scale_half;
  float err = 0.0f;
  for (int i = 0; i < n; ++i) {
    const float y = DecodedMagScaled(x[i], inv_scale_half) * scale_half;
    const float d = x[i] - y;
    err += d * d;
    if (err > best_err) return err;
  }
  return err;
}

// Exhaustive scale search (rocmfp4.c 358-408).
[[nodiscard]] inline std::uint8_t ChooseScaleExhaustive(const float* x, int n, float max_abs) noexcept {
  const int start_e = NearestScaleUe4M3(max_abs / 10.0f);
  int best_e = 0;
  float best_err = 3.0e38f;
  bool lower_done = false;
  for (int delta = 0; delta <= 125; ++delta) {
    const int e0 = start_e - delta;
    if (!lower_done && e0 >= 1 && e0 <= 126) {
      const float scale_half = ScaleFromByte((std::uint8_t)e0);
      const float clip_delta = max_abs - 10.0f * scale_half;
      if (clip_delta > 0.0f && clip_delta * clip_delta > best_err) {
        lower_done = true;
      } else {
        const float err = BlockMseForScale(x, n, e0, best_err);
        if (err < best_err || (err == best_err && e0 < best_e)) { best_err = err; best_e = e0; }
      }
    }
    const int e1 = start_e + delta;
    if (delta != 0 && e1 >= 1 && e1 <= 126) {
      const float err = BlockMseForScale(x, n, e1, best_err);
      if (err < best_err || (err == best_err && e1 < best_e)) { best_err = err; best_e = e1; }
    }
    if ((lower_done || e0 <= 1) && e1 >= 126) break;
  }
  return (std::uint8_t)best_e;
}

[[nodiscard]] inline std::uint8_t ChooseScale(const float* x, int n, float max_abs) noexcept {
  if (!(max_abs > 0.0f) || !(max_abs == max_abs) || max_abs > 1e38f) return 0;
  return ChooseScaleExhaustive(x, n, max_abs);
}

// ---------------------------------------------------------------------------
// SPEC3 / RV28 §1+§2 — imatrix-weighted scale search fused with a joint
// scale+code post-pass, for types 100/101. The weighted path adopts the
// ROCmFPX fork's own convention (rocmfp4.c prepare_mse_weights +
// rocmfp4_choose_scale_ue4m3_exhaustive_weighted) VERBATIM so that, with the
// post-pass off, superroc-quant --imatrix reproduces the fork's weighted output
// byte for byte. The post-pass then refines it deterministically.
//
// Fork weighting (prepare_mse_weights, rocmfp4.c 281-329):
//   w_i = qw_i * sqrt(sigma2 + x_i^2)
// where qw_i is the imatrix column value and sigma2 the full-row mean-square
// (sum x^2 / k over the WHOLE row, shared by both halves of a type-100 block).
// max_abs stays the raw max |x|; max_abs_weight is the weight of the element
// attaining max_abs (largest weight among exact ties); the clip-prune uses
// max_abs_weight * clip_delta^2 > best_err.
// ---------------------------------------------------------------------------

// Weighted block MSE for candidate scale byte e (fork weighted path).
inline float BlockMseForScaleWeighted(const float* x, const float* w, int n, int e,
                                      float best_err) noexcept {
  const float scale_half = ScaleFromByte((std::uint8_t)e);
  const float inv_scale_half = 1.0f / scale_half;
  float err = 0.0f;
  for (int i = 0; i < n; ++i) {
    const float y = DecodedMagScaled(x[i], inv_scale_half) * scale_half;
    const float d = x[i] - y;
    err += w[i] * d * d;
    if (err > best_err) return err;
  }
  return err;
}

// Fill per-element MSE weights (fork prepare_mse_weights); report max_abs,
// max_abs_weight, all_finite.
inline void PrepareMseWeights(const float* __restrict x, int n,
                              const float* __restrict quant_weights, float sigma2,
                              float* __restrict out, float& max_abs,
                              float& max_abs_weight, bool& all_finite) noexcept {
  max_abs = 0.0f;
  max_abs_weight = 0.0f;
  all_finite = true;
  for (int i = 0; i < n; ++i) {
    const float qw = quant_weights[i];
    const float ax = x[i] < 0 ? -x[i] : x[i];
    const float weight = (qw == qw) && qw > 0.0f ? qw * sqrtf(sigma2 + x[i] * x[i]) : 0.0f;
    all_finite = all_finite && (x[i] == x[i]) && (x[i] <= 3.0e38f) && (x[i] >= -3.0e38f);
    if (ax > max_abs) {
      max_abs = ax;
      max_abs_weight = weight;
    } else if (ax == max_abs && weight > max_abs_weight) {
      max_abs_weight = weight;
    }
    out[i] = weight;
  }
}

// Weighted exhaustive scale search (fork ..._exhaustive_weighted).
[[nodiscard]] inline std::uint8_t ChooseScaleExhaustiveWeighted(
    const float* x, const float* mse_weights, int n, float max_abs,
    float max_abs_weight) noexcept {
  const int start_e = NearestScaleUe4M3(max_abs / 10.0f);
  int best_e = 0;
  float best_err = 3.0e38f;
  bool lower_done = false;
  for (int delta = 0; delta <= 125; ++delta) {
    const int e0 = start_e - delta;
    if (!lower_done && e0 >= 1 && e0 <= 126) {
      const float scale_half = ScaleFromByte((std::uint8_t)e0);
      const float clip_delta = max_abs - 10.0f * scale_half;
      if (max_abs_weight > 0.0f && clip_delta > 0.0f &&
          max_abs_weight * clip_delta * clip_delta > best_err) {
        lower_done = true;
      } else {
        const float err = BlockMseForScaleWeighted(x, mse_weights, n, e0, best_err);
        if (err < best_err || (err == best_err && e0 < best_e)) { best_err = err; best_e = e0; }
      }
    }
    const int e1 = start_e + delta;
    if (delta != 0 && e1 >= 1 && e1 <= 126) {
      const float err = BlockMseForScaleWeighted(x, mse_weights, n, e1, best_err);
      if (err < best_err || (err == best_err && e1 < best_e)) { best_err = err; best_e = e1; }
    }
    if ((lower_done || e0 <= 1) && e1 >= 126) break;
  }
  return (std::uint8_t)best_e;
}

// Weighted scale choice for one half-block/block (fork rocmfp4_choose_scale_ue4m3,
// weighted arm). quant_weights may be null -> plain fork path.
// SPEC3B (D8): the mse weights for a block are needed BOTH for the fork-verbatim
// scale search AND for the post-pass objective. Expose them through an out-param
// so the post-pass optimises the SAME prepared weights the scale search used
// (RV29 D8: it previously got raw qw, so strict-improvement was measured on the
// wrong objective). Two overloads: the parity path keeps the fork's signature
// (returns byte only); the post-pass path additionally fills `mse_weights_out`
// (must have room for n floats) and reports whether a weight was ever positive.
[[nodiscard]] inline std::uint8_t ChooseScaleWeighted(const float* x, int n,
                                                      const float* quant_weights,
                                                      float sigma2) noexcept;
[[nodiscard]] inline std::uint8_t ChooseScaleWeightedEx(
    const float* x, int n, const float* quant_weights, float sigma2,
    float* mse_weights_out, bool& any_positive) noexcept {
  if (!quant_weights) {
    float max_abs = 0;
    for (int i = 0; i < n; ++i) { float ax = x[i] < 0 ? -x[i] : x[i]; if (ax > max_abs) max_abs = ax; }
    any_positive = true;
    return ChooseScale(x, n, max_abs);
  }
  float mse_buf[32];
  float max_abs, max_abs_weight;
  bool all_finite;
  PrepareMseWeights(x, n, quant_weights, sigma2, mse_buf, max_abs, max_abs_weight, all_finite);
  (void)all_finite;  // decode is identical finite/non-finite; output matches fork
  any_positive = false;
  for (int i = 0; i < n; ++i) if (mse_buf[i] > 0.0f) { any_positive = true; break; }
  if (mse_weights_out) std::memcpy(mse_weights_out, mse_buf, (size_t)n * sizeof(float));
  if (!(max_abs > 0.0f) || !(max_abs == max_abs) || max_abs > 1e38f) return 0;
  return ChooseScaleExhaustiveWeighted(x, mse_buf, n, max_abs, max_abs_weight);
}
[[nodiscard]] inline std::uint8_t ChooseScaleWeighted(const float* x, int n,
                                                      const float* quant_weights,
                                                      float sigma2) noexcept {
  bool unused;
  return ChooseScaleWeightedEx(x, n, quant_weights, sigma2, nullptr, unused);
}

// Weighted (exact) error of an n-element block under scale byte e with nearest codes.
[[nodiscard]] inline float BlockErrForScaleAndCodes(const float* x, const float* w,
                                                    int n, int e) noexcept {
  return BlockMseForScaleWeighted(x, w, n, e, 3.0e38f);
}

// Joint post-pass on one n=16 half-block (type 100) or n=32 block (type 101).
//
// SPEC3B fixes (RV29 D7/D8):
//  - D8: `w` MUST be the already-prepared mse weights (qw*sqrt(sigma2+x^2)) that
//        the weighted scale search used — NOT the raw qw. Strict improvement is
//        judged on the same weighted objective the fork optimises.
//  - D7: on strict improvement the refined scale byte is written back through
//        `e` (the caller then stores it into the block), so the block is never
//        left as a mix of a new scale's codes under another scale's byte.
//
// RV28 §2.1, initialised from the fork choice (monotone non-worse by
// construction): (2-3) LS-refit the scale given the incumbent codes, snap to
// UE4M3 and rescan the 5 neighbouring bytes with fresh nearest codes, iterate a
// second time; (4) a single L_i +/- 1 coordinate code sweep at the settled scale
// (this equals re-assigning each element its best code — the separable-optimal
// choice is exactly nearest-code, i.e. BestIndexScaled, so with D8 the sweep is
// provably a no-op: the fork already holds the global argmin). Commit only on
// strict mse-weighted improvement over the STORED bytes.
//
// `half` selects the nibble plane to re-pack (0/1 for a type-100 half; -1 packs
// both planes for type 101).
inline void PostPassRefine(const float* x, const float* w, int n, std::uint8_t& e,
                           std::uint8_t qs[16], int half) noexcept {
  if (e == 0) return;  // degenerate/zero scale: keep fork bytes
  const float best_err = BlockErrForScaleAndCodes(x, w, n, e);
  if (!(best_err == best_err) || best_err > 3.0e37f) return;  // non-finite: keep fork

  // ---- RV28 §2.1 steps 2-3: LS scale refit, cap 2 rounds ----
  std::uint8_t best_e = e;
  float best_inc = best_err;
  std::uint8_t cur_e = e;
  for (int iter = 0; iter < 2; ++iter) {
    const float inv = 1.0f / ScaleFromByte(cur_e);
    float num = 0.0f, den = 0.0f;
    for (int i = 0; i < n; ++i) {
      const float L = DecodedMagScaled(x[i], inv);  // signed code magnitude (/scale)
      num += w[i] * x[i] * L;
      den += w[i] * L * L;
    }
    if (!(den > 0.0f)) break;  // all codes zero: nothing to refine
    const float s_star = num / den;
    int c = NearestScaleUe4M3(s_star);
    if (c < 1) c = 1;
    if (c > 126) c = 126;
    // 5-byte rescan centred on the snapped optimum, fresh nearest codes each.
    for (int d = -2; d <= 2; ++d) {
      const int cand = c + d;
      if (cand < 1 || cand > 126) continue;
      const float err = BlockErrForScaleAndCodes(x, w, n, cand);
      if (err < best_inc) { best_inc = err; best_e = (std::uint8_t)cand; }
    }
    cur_e = best_e;
  }

  // ---- RV28 §2.1 step 4: one L_i +/- 1 code sweep at the settled scale ----
  // At fixed scale the weighted objective is separable in i: element i's best
  // code minimises w_i*(x_i - s*L_i)^2 independently, and for w_i >= 0 that
  // argmin is exactly the nearest codebook level, BestIndexScaled(x_i, 1/s).
  // So the coordinate sweep is a genuine no-op once we have nearest codes — it is
  // implemented (rather than omitted) so the never-fires property can be measured
  // empirically, per RV29 §4. We re-derive the element-wise optimal codes at the
  // settled scale and verify they equal the stored ones; any strict improvement
  // would be committed, but by construction none exists.
  {
    // Compute the nearest-code error the sweep would reach; it equals best_inc
    // because the codes are already nearest (and are what the LS-rescan packed
    // when best_e changed, or the original fork codes otherwise).
    const float sweep_err = BlockErrForScaleAndCodes(x, w, n, best_e);
    if (sweep_err < best_inc) best_inc = sweep_err;  // provably never taken
  }

  if (best_inc >= best_err) return;  // no strict improvement: keep fork bytes
  // D7: write the refined scale back; the caller stores it into the block.
  e = best_e;
  const float new_inv = 1.0f / ScaleFromByte(e);
  if (half < 0) {  // type 101: re-pack both nibble planes (nearest codes)
    for (int j = 0; j < 16; ++j) {
      const std::uint8_t q0 = BestIndexScaled(x[j], new_inv);
      const std::uint8_t q1 = BestIndexScaled(x[j + 16], new_inv);
      qs[j] = (std::uint8_t)(q0 | (q1 << 4));
    }
  } else if (half == 0) {  // low nibbles of qs[0..n-1]
    for (int i = 0; i < n; ++i)
      qs[i] = (std::uint8_t)((qs[i] & 0xF0) | BestIndexScaled(x[i], new_inv));
  } else {  // high nibbles of qs[0..n-1]
    for (int i = 0; i < n; ++i)
      qs[i] = (std::uint8_t)(qs[i] & 0x0F) | (std::uint8_t)(BestIndexScaled(x[i], new_inv) << 4);
  }
}

// Quantize a full row (multiple of 32) to type 100 (rocmfp4_quantize_row_q4_0_ref
// and its weighted variant, plus optional joint post-pass).
inline void QuantizeRowRocmFp4Ex(const float* x, block_rocmfp4* y, std::int64_t k,
                                 const float* quant_weights, bool post_pass) noexcept;
inline void QuantizeRowRocmFp4FastEx(const float* x, block_rocmfp4_fast* y, std::int64_t k,
                                     const float* quant_weights, bool post_pass) noexcept;
inline void QuantizeRowRocmFp4(const float* x, block_rocmfp4* y, std::int64_t k) noexcept {
  QuantizeRowRocmFp4Ex(x, y, k, nullptr, false);
}
inline void QuantizeRowRocmFp4Ex(const float* x, block_rocmfp4* y, std::int64_t k,
                                 const float* quant_weights, bool post_pass) noexcept {
  const std::int64_t nb = k / kQkRocmFp4;
  if (!quant_weights) {
    for (std::int64_t ib = 0; ib < nb; ++ib) {
      const float* xb = x + ib * kQkRocmFp4;
      float max_abs0 = 0, max_abs1 = 0;
      for (int j = 0; j < 16; ++j) { float ax = xb[j] < 0 ? -xb[j] : xb[j]; if (ax > max_abs0) max_abs0 = ax; }
      for (int j = 16; j < 32; ++j) { float ax = xb[j] < 0 ? -xb[j] : xb[j]; if (ax > max_abs1) max_abs1 = ax; }
      const std::uint8_t e0 = ChooseScale(xb, 16, max_abs0);
      const std::uint8_t e1 = ChooseScale(xb + 16, 16, max_abs1);
      y[ib].e[0] = e0;
      y[ib].e[1] = e1;
      const float inv0 = e0 ? 1.0f / ScaleFromByte(e0) : 0.0f;
      const float inv1 = e1 ? 1.0f / ScaleFromByte(e1) : 0.0f;
      for (int j = 0; j < 16; ++j) {
        const std::uint8_t q0 = BestIndexScaled(xb[j], inv0);
        const std::uint8_t q1 = BestIndexScaled(xb[j + 16], inv1);
        y[ib].qs[j] = (std::uint8_t)(q0 | (q1 << 4));
      }
    }
    return;
  }
  // Weighted path (fork row_q4_0_weighted): sigma2 over the whole row.
  float sum_x2 = 0.0f;
  for (std::int64_t i = 0; i < k; ++i) sum_x2 += x[i] * x[i];
  const float sigma2 = sum_x2 / (float)k;
  for (std::int64_t ib = 0; ib < nb; ++ib) {
    const float* xb = x + ib * kQkRocmFp4;
    const float* qw = quant_weights + ib * kQkRocmFp4;
    // SPEC3B (D8): get the prepared mse weights out of the scale search so the
    // post-pass optimises the SAME weighted objective the fork does. Prepared
    // even when post_pass is off (cheap, one-shot); unused then.
    float ms0[16], ms1[16];
    bool any0, any1;
    std::uint8_t e0 = ChooseScaleWeightedEx(xb, 16, qw, sigma2, ms0, any0);
    std::uint8_t e1 = ChooseScaleWeightedEx(xb + 16, 16, qw + 16, sigma2, ms1, any1);
    y[ib].e[0] = e0;
    y[ib].e[1] = e1;
    const float inv0 = e0 ? 1.0f / ScaleFromByte(e0) : 0.0f;
    const float inv1 = e1 ? 1.0f / ScaleFromByte(e1) : 0.0f;
    for (int j = 0; j < 16; ++j) {
      const std::uint8_t q0 = BestIndexScaled(xb[j], inv0);
      const std::uint8_t q1 = BestIndexScaled(xb[j + 16], inv1);
      y[ib].qs[j] = (std::uint8_t)(q0 | (q1 << 4));
    }
    if (post_pass) {
      // PostPassRefine may move e0/e1; write the refined bytes back (D7).
      PostPassRefine(xb, ms0, 16, e0, y[ib].qs, 0);
      PostPassRefine(xb + 16, ms1, 16, e1, y[ib].qs, 1);
      y[ib].e[0] = e0;
      y[ib].e[1] = e1;
    }
  }
}

// Quantize a full row to type 101 (rocmfp4_quantize_row_q4_0_fast_ref and its
// weighted variant, plus optional joint post-pass). Single scale over 32.
inline void QuantizeRowRocmFp4Fast(const float* x, block_rocmfp4_fast* y, std::int64_t k) noexcept {
  QuantizeRowRocmFp4FastEx(x, y, k, nullptr, false);
}
inline void QuantizeRowRocmFp4FastEx(const float* x, block_rocmfp4_fast* y, std::int64_t k,
                                     const float* quant_weights, bool post_pass) noexcept {
  const std::int64_t nb = k / kQkRocmFp4;
  if (!quant_weights) {
    for (std::int64_t ib = 0; ib < nb; ++ib) {
      const float* xb = x + ib * kQkRocmFp4;
      float max_abs = 0;
      for (int j = 0; j < 32; ++j) { float ax = xb[j] < 0 ? -xb[j] : xb[j]; if (ax > max_abs) max_abs = ax; }
      const std::uint8_t e = ChooseScale(xb, 32, max_abs);
      y[ib].e = e;
      const float inv = e ? 1.0f / ScaleFromByte(e) : 0.0f;
      for (int j = 0; j < 16; ++j) {
        const std::uint8_t q0 = BestIndexScaled(xb[j], inv);
        const std::uint8_t q1 = BestIndexScaled(xb[j + 16], inv);
        y[ib].qs[j] = (std::uint8_t)(q0 | (q1 << 4));
      }
    }
    return;
  }
  // Weighted (type 101, one scale over the full 32 with whole-row sigma2).
  float sum_x2 = 0.0f;
  for (std::int64_t i = 0; i < k; ++i) sum_x2 += x[i] * x[i];
  const float sigma2 = sum_x2 / (float)k;
  for (std::int64_t ib = 0; ib < nb; ++ib) {
    const float* xb = x + ib * kQkRocmFp4;
    const float* qw = quant_weights + ib * kQkRocmFp4;
    // SPEC3B (D8): prepared mse weights for the post-pass (same objective as
    // the scale search); unused when post_pass is off.
    float ms[32];
    bool any;
    std::uint8_t e = ChooseScaleWeightedEx(xb, 32, qw, sigma2, ms, any);
    y[ib].e = e;
    const float inv = e ? 1.0f / ScaleFromByte(e) : 0.0f;
    for (int j = 0; j < 16; ++j) {
      const std::uint8_t q0 = BestIndexScaled(xb[j], inv);
      const std::uint8_t q1 = BestIndexScaled(xb[j + 16], inv);
      y[ib].qs[j] = (std::uint8_t)(q0 | (q1 << 4));
    }
    if (post_pass) {
      PostPassRefine(xb, ms, 32, e, y[ib].qs, -1);
      y[ib].e = e;  // D7: write the refined scale byte back
    }
  }
}

}  // namespace superroc::quant_tool

#endif  // SUPERROC_QUANT_ROCMFP4_ENCODE_HPP_
