// kda_math_test — validate LG's KDA math against the fork's CPU kernels,
// standalone. No model file, no GPU; runs on the Mac.
//
// Three pieces, compared bit-for-bit:
//   1. ggml_ssm_conv        vs LG's causal depthwise conv
//   2. ggml_l2_norm         vs LG's L2Norm
//   3. ggml_gated_delta_net vs LG's KdaScan recurrence (expf vs std::exp)
//
// Fork loops transcribed from ggml/src/ggml-cpu/ops.cpp:
//   ggml_compute_forward_ssm_conv_f32          (~9582)
//   ggml_compute_forward_l2_norm_f32           (~4132)
//   ggml_compute_forward_gated_delta_net_one_chunk (~10959)
// with ggml_vec_* replaced by the scalar loops they expand to.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

std::uint64_t rng_state = 0x1234567890ABCDEFULL;
double Next() {
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<double>(rng_state >> 11) / 9007199254740992.0;
}
float NextF(float lo, float hi) {
  return static_cast<float>(lo + Next() * (hi - lo));
}

// ---------------------------------------------------------------------------
// 1. ggml_ssm_conv
//
// src0 = {ncs = kc-1+n_t, nr = inner}, src1 = {nc = kc, nr = inner}
// ops.cpp: s = src0->data + ir0*nb[1] + i2*nb[0];  then s[i0 + i1*ncs]
//          c = src1->data + ir0*nb[1];              then c[i0 + i1*nc]
// so out[t][ch] = sum_i0 src0[t + i0 + ch*ncs] * src1[i0 + ch*nc].
// ---------------------------------------------------------------------------
void ForkSsmConv(const float* src0, std::size_t ncs, const float* src1,
                 std::size_t nc, float* dst, std::size_t nr, std::size_t n_t) {
  for (std::size_t t = 0; t < n_t; ++t) {
    const float* s = src0 + t;  // ir0 == 0 in a single-threaded reference
    const float* c = src1;
    float* x = dst + t * nr;
    for (std::size_t ch = 0; ch < nr; ++ch) {
      float sumf = 0.0f;
      for (std::size_t i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0 + ch * ncs] * c[i0 + ch * nc];
      }
      x[ch] = sumf;
    }
  }
}

/// LG: conv_buf is [kc][inner] row-major (row c = all channels for tap c),
/// matching reference.cpp. taps_by_channel is [inner][kc] because
/// `DequantizeRow(convw, 0, d, taps_)` reads row d of the [kc, inner] view.
void LgCausalConv(const float* conv_buf, std::size_t kc, std::size_t inner,
                  const float* taps_by_channel, float* out) {
  for (std::size_t d = 0; d < inner; ++d) {
    float acc = 0.0f;
    for (std::size_t c = 0; c < kc; ++c) {
      acc += conv_buf[c * inner + d] * taps_by_channel[d * kc + c];
    }
    out[d] = acc;
  }
}

// ---------------------------------------------------------------------------
// 2. ggml_l2_norm: scale = 1 / fmaxf(sqrtf(sum), eps)
// ---------------------------------------------------------------------------
void ForkL2Norm(float* x, std::size_t n, float eps) {
  // ops.cpp: `ggml_float sum` (double) but `sum += (ggml_float)(xi * xi)` —
  // the product is rounded to float first, then widened. Reproduce exactly.
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const float xi = x[i];
    sum += static_cast<double>(xi * xi);
  }
  const float scale = 1.0f / std::fmax(std::sqrt(static_cast<float>(sum)), eps);
  for (std::size_t i = 0; i < n; ++i) x[i] *= scale;
}

/// LG, after the fix: same float product, same double accumulator.
void LgL2Norm(float* x, std::size_t n, float eps) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const float xi = x[i];
    sum += static_cast<double>(xi * xi);
  }
  // sqrtf, not sqrt: the fork narrows the double accumulator to float
  // *before* taking the root, so rounding the root differs.
  const float norm = std::sqrt(static_cast<float>(sum));
  const float scale = 1.0f / std::fmax(norm, eps);
  for (std::size_t i = 0; i < n; ++i) x[i] *= scale;
}

// ---------------------------------------------------------------------------
// 3a. fork: expf, exactly as ops.cpp writes it.
// ---------------------------------------------------------------------------
void ForkGdnKda(std::vector<float>& state, const float* q, const float* k,
                const float* v, const float* g, float beta, std::size_t sv,
                float* out) {
  std::vector<float> delta(sv);
  const float scale = 1.0f / std::sqrt(static_cast<float>(sv));
  for (std::size_t i = 0; i < sv; ++i) delta[i] = std::exp(g[i]);
  for (std::size_t j = 0; j < sv; ++j) {
    float* row = state.data() + j * sv;
    for (std::size_t i = 0; i < sv; ++i) row[i] *= delta[i];
  }
  for (std::size_t j = 0; j < sv; ++j) {
    const float* row = state.data() + j * sv;
    float sum = 0.0f;
    for (std::size_t i = 0; i < sv; ++i) sum += row[i] * k[i];
    delta[j] = (v[j] - sum) * beta;
  }
  for (std::size_t j = 0; j < sv; ++j) {
    float* row = state.data() + j * sv;
    const float d = delta[j];
    for (std::size_t i = 0; i < sv; ++i) row[i] += k[i] * d;
  }
  for (std::size_t j = 0; j < sv; ++j) {
    const float* row = state.data() + j * sv;
    float sum = 0.0f;
    for (std::size_t i = 0; i < sv; ++i) sum += row[i] * q[i];
    out[j] = sum * scale;
  }
}

/// 3b. LG: a literal copy of the phase structure in reference.cpp, including
/// the reusable `decay_` / `delta_` scratch and std::exp.
void LgKdaScan(std::vector<float>& state, const float* qh, const float* kh,
               const float* vh, const float* gh, float beta, std::size_t hd,
               float* attn_out) {
  std::vector<float> decay(hd), delta(hd);
  float* S = state.data();
  const float scan_scale = 1.0F / std::sqrt(static_cast<float>(hd));
  for (std::size_t i = 0; i < hd; ++i) decay[i] = std::exp(gh[i]);
  for (std::size_t j = 0; j < hd; ++j) {
    float* row = S + j * hd;
    for (std::size_t i = 0; i < hd; ++i) row[i] *= decay[i];
  }
  for (std::size_t j = 0; j < hd; ++j) {
    const float* row = S + j * hd;
    float sum = 0.0F;
    for (std::size_t i = 0; i < hd; ++i) sum += row[i] * kh[i];
    delta[j] = (vh[j] - sum) * beta;
  }
  for (std::size_t j = 0; j < hd; ++j) {
    float* row = S + j * hd;
    const float d = delta[j];
    for (std::size_t i = 0; i < hd; ++i) row[i] += kh[i] * d;
  }
  for (std::size_t j = 0; j < hd; ++j) {
    const float* row = S + j * hd;
    float sum = 0.0F;
    for (std::size_t i = 0; i < hd; ++i) sum += row[i] * qh[i];
    attn_out[j] = sum * scan_scale;
  }
}

}  // namespace

int main() {
  int failures = 0;

  // ---- conv ---------------------------------------------------------------
  {
    int bad_trials = 0;
    for (int trial = 0; trial < 300; ++trial) {
      const std::size_t kc = 4;
      const std::size_t inner = 1 + static_cast<std::size_t>(Next() * 512);
      const std::size_t n_t = 1 + static_cast<std::size_t>(Next() * 5);
      const std::size_t ncs = kc - 1 + n_t;

      // Build the same logical conv_x[channel][row] twice, once in ggml's
      // layout (index row + channel*ncs, because conv_x is a {ncs, inner}
      // tensor) and once in LG's [row][channel] layout (index row*inner +
      // channel). Mixing the two up is exactly the mistake this test exists to
      // catch, so both are built explicitly from one set of values.
      std::vector<float> ggml_x(ncs * inner);
      std::vector<float> lg_x(ncs * inner);
      for (std::size_t ch = 0; ch < inner; ++ch) {
        for (std::size_t r = 0; r < ncs; ++r) {
          const float v = NextF(-2.0F, 2.0F);
          ggml_x[r + ch * ncs] = v;
          lg_x[r * inner + ch] = v;
        }
      }
      // conv_w[ch][c]: the fork reads it as reshape_2d([kc,1,inner]) -> [kc,
      // inner], i.e. index c + ch*kc. LG reads row ch of the same matrix.
      std::vector<float> ggml_w(kc * inner);
      std::vector<float> taps(inner * kc);
      for (std::size_t ch = 0; ch < inner; ++ch) {
        for (std::size_t c = 0; c < kc; ++c) {
          const float v = NextF(-1.0F, 1.0F);
          ggml_w[c + ch * kc] = v;
          taps[ch * kc + c] = v;
        }
      }

      std::vector<float> fork(n_t * inner, -7.0F);
      ForkSsmConv(ggml_x.data(), ncs, ggml_w.data(), kc, fork.data(), inner, n_t);

      std::vector<float> lg(n_t * inner, -7.0F);
      for (std::size_t t = 0; t < n_t; ++t) {
        LgCausalConv(lg_x.data() + t * inner, kc, inner, taps.data(),
                     lg.data() + t * inner);
      }

      std::size_t bad = 0;
      for (std::size_t i = 0; i < fork.size(); ++i) {
        if (std::memcmp(&fork[i], &lg[i], sizeof(float)) != 0) ++bad;
      }
      if (bad) {
        std::printf("  conv  trial %3d inner=%zu n_t=%zu: %zu/%zu differ\n",
                    trial, inner, n_t, bad, fork.size());
        ++bad_trials;
      }
    }
    std::printf("conv   : %s (300 trials)\n", bad_trials ? "FAIL" : "OK");
    failures += bad_trials;
  }

  // ---- l2 norm ------------------------------------------------------------
  {
    int bad_trials = 0;
    for (int trial = 0; trial < 500; ++trial) {
      const std::size_t n = 1 + static_cast<std::size_t>(Next() * 256);
      std::vector<float> a(n), b(n);
      for (std::size_t i = 0; i < n; ++i) a[i] = b[i] = NextF(-3.0F, 3.0F);
      if (trial % 7 == 0) {
        for (auto& x : a) x *= 1e-9F;  // make the eps clamp bite
        b = a;
      }
      ForkL2Norm(a.data(), n, 1e-6F);
      LgL2Norm(b.data(), n, 1e-6F);
      for (std::size_t i = 0; i < n; ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
          std::printf("  l2    trial %d n=%zu elem %zu: fork=%.9g lg=%.9g\n", trial,
                      n, i, static_cast<double>(a[i]), static_cast<double>(b[i]));
          ++bad_trials;
          break;
        }
      }
    }
    std::printf("l2norm : %s (500 trials, incl. eps clamp)\n",
                bad_trials ? "FAIL" : "OK");
    failures += bad_trials;
  }

  // ---- gated delta net ----------------------------------------------------
  {
    int bad_trials = 0;
    for (int trial = 0; trial < 400; ++trial) {
      const std::size_t sv = 1 + static_cast<std::size_t>(Next() * 128);
      std::vector<float> q(sv), k(sv), v(sv), g(sv);
      for (auto& x : q) x = NextF(-1.0F, 1.0F);
      for (auto& x : k) x = NextF(-1.0F, 1.0F);
      for (auto& x : v) x = NextF(-1.0F, 1.0F);
      for (auto& x : g) x = NextF(-5.0F, 0.0F);  // the safe gate is <= 0
      const float beta = NextF(0.0F, 1.0F);

      std::vector<float> s1(sv * sv);
      for (auto& x : s1) x = NextF(-0.5F, 0.5F);
      std::vector<float> s2 = s1;
      std::vector<float> o1(sv, -1.0F), o2(sv, -1.0F);

      ForkGdnKda(s1, q.data(), k.data(), v.data(), g.data(), beta, sv, o1.data());
      LgKdaScan(s2, q.data(), k.data(), v.data(), g.data(), beta, sv, o2.data());

      std::size_t bad = 0, first = 0;
      for (std::size_t i = 0; i < sv; ++i) {
        if (std::memcmp(&o1[i], &o2[i], sizeof(float)) != 0) { ++bad; first = i; }
      }
      std::size_t badstate = 0;
      for (std::size_t i = 0; i < sv * sv; ++i) {
        if (std::memcmp(&s1[i], &s2[i], sizeof(float)) != 0) ++badstate;
      }
      if (bad || badstate) {
        std::printf("  gdn   trial %3d sv=%3zu: out %zu differ (first %zu: "
                    "fork=%.9g lg=%.9g), state %zu differ\n",
                    trial, sv, bad, first, bad ? static_cast<double>(o1[first]) : 0.0,
                    bad ? static_cast<double>(o2[first]) : 0.0, badstate);
        if (++bad_trials > 5) break;
      }
    }
    std::printf("gdn    : %s (400 trials; LG's phases vs the fork's)\n",
                bad_trials ? "FAIL" : "OK");
    failures += bad_trials;
  }

  std::printf("\n%s: %d failures\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
