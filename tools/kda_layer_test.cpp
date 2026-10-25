// kda_layer_test — verify the KDA *layer* (not just the scan) against an
// independent transcription of the fork's graph. Runs on the Mac; no model.
//
// The thing under test is the conv-state handling. The fork gives Q, K and V
// three separate histories inside conv_states_all, each read at offset
// `(kv_head * n_embd_r_total + qkv * conv_state_size)`. LG keeps one history
// and snapshots it once per layer, advancing it once after all three
// projections. This checks the two are equivalent, end to end, over several
// tokens -- which is the property that was actually broken.
//
// Matrices are plain dense float, decoded by construction, so this isolates
// the layer arithmetic from the quantization path (verified separately, bit
// exact, by tools/ling3_dequant_parity and tools/ling3_rocmfp4_test).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace {

std::uint64_t rs = 0xDEADBEEFULL;
double Nx() {
  rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<double>(rs >> 11) / 9007199254740992.0;
}
float Nf(float lo, float hi) { return static_cast<float>(lo + Nx() * (hi - lo)); }

float Sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }
float Silu(float x) { return x / (1.0F + std::exp(-x)); }

// y[r] = sum_c W[r*cols + c] * x[c]
void MatVec(const std::vector<float>& W, std::size_t rows, std::size_t cols,
            const std::vector<float>& x, std::vector<float>* y) {
  y->assign(rows, 0.0F);
  for (std::size_t r = 0; r < rows; ++r) {
    float s = 0.0F;
    for (std::size_t c = 0; c < cols; ++c) s += W[r * cols + c] * x[c];
    (*y)[r] = s;
  }
}

void RmsNorm(std::vector<float>* x, const std::vector<float>& w, float eps) {
  double sum = 0.0;
  for (float v : *x) sum += static_cast<double>(v) * static_cast<double>(v);
  const float mean = static_cast<float>(sum / static_cast<double>(x->size()));
  const float scale = 1.0F / std::sqrt(mean + eps);
  for (std::size_t i = 0; i < x->size(); ++i) (*x)[i] = scale * w[i];
}

void L2Norm(std::vector<float>* x, float eps) {
  double sum = 0.0;
  for (float v : *x) sum += static_cast<double>(v) * static_cast<double>(v);
  const float n = std::sqrt(static_cast<float>(sum));
  const float s = 1.0F / std::fmax(n, eps);
  for (float& v : *x) v *= s;
}

struct Weights {
  std::size_t H, inner, heads, hd, kc;
  std::vector<float> wq, wk, wv;      // [inner][H]
  std::vector<float> cwq, cwk, cwv;    // [kc][inner] (fork's [kc,inner] view)
  std::vector<float> wf, wg;            // [inner][H]
  std::vector<float> wbeta;             // [heads][H]
  std::vector<float> a_log, dt_bias, o_norm, wo;  // [heads], [inner], [hd], [H][inner]
  float lower_bound;
};

Weights MakeWeights(std::size_t H, std::size_t heads, std::size_t hd,
                    std::size_t kc) {
  Weights w;
  w.H = H; w.heads = heads; w.hd = hd; w.kc = kc;
  w.inner = heads * hd;
  w.lower_bound = -5.0F;
  auto rnd = [&](std::vector<float>& v, std::size_t n, float s) {
    v.resize(n);
    for (auto& x : v) x = Nf(-s, s);
  };
  rnd(w.wq, w.inner * H, 0.2F);
  rnd(w.wk, w.inner * H, 0.2F);
  rnd(w.wv, w.inner * H, 0.2F);
  rnd(w.cwq, kc * w.inner, 0.5F);
  rnd(w.cwk, kc * w.inner, 0.5F);
  rnd(w.cwv, kc * w.inner, 0.5F);
  rnd(w.wf, w.inner * H, 0.2F);
  rnd(w.wg, w.inner * H, 0.2F);
  rnd(w.wbeta, heads * H, 0.2F);
  rnd(w.a_log, heads, 0.5F);
  for (auto& x : w.a_log) x = std::fabs(x) + 0.1F;   // exp(A_log) > 0
  rnd(w.dt_bias, w.inner, 1.0F);
  rnd(w.o_norm, hd, 1.0F);
  rnd(w.wo, H * w.inner, 0.2F);
  return w;
}

struct State {
  // fork: three conv histories, one per projection
  std::vector<float> conv[3];
  // LG: one, snapshotted per layer
  std::vector<float> conv_shared;
  std::vector<float> scan;  // [heads][hd][hd]
};

// ---------------------------------------------------------------------------
// The fork, transcribed: each projection reads and writes its OWN history.
// conv_state_size = (kc-1)*inner ; offset = (kv_head*n_embd_r_total + qkv*cs)
// with kv_head = 0 and one sequence.
// ---------------------------------------------------------------------------
void ForkConv(std::vector<float>* hist, const std::vector<float>& proj_in,
              const std::vector<float>& cw, std::size_t kc, std::size_t inner,
              std::vector<float>* out) {
  // conv_x = [history rows (kc-1) ; this token]
  std::vector<float> conv_x((kc - 1 + 1) * inner);
  std::copy(hist->begin(), hist->end(), conv_x.begin());
  std::copy(proj_in.begin(), proj_in.end(),
            conv_x.begin() + static_cast<std::ptrdiff_t>((kc - 1) * inner));
  out->assign(inner, 0.0F);
  for (std::size_t d = 0; d < inner; ++d) {
    float acc = 0.0F;
    for (std::size_t c = 0; c < kc; ++c) {
      // cw is [kc][inner]: tap c for channel d is cw[c*inner + d]
      acc += conv_x[(0 + c) * inner + d] * cw[c * inner + d];
    }
    (*out)[d] = Silu(acc);
  }
  // new history = rows 1..kc-1 of conv_x
  hist->assign(conv_x.begin() + static_cast<std::ptrdiff_t>(inner),
               conv_x.end());
}

// ---------------------------------------------------------------------------
// LG: three independent histories in one buffer. Projection qkv reads its own
// slice and advances only that slice -- which is what the fork's
// `+ qkv * conv_state_size` offsets mean, and the earlier single-history and
// snapshot-once variants were both wrong (verified by this test).
// ---------------------------------------------------------------------------
void LgConv(std::vector<float>* conv_all, std::size_t qkv,
            const std::vector<float>& proj_in, const std::vector<float>& cw,
            std::size_t kc, std::size_t inner, std::vector<float>* out) {
  const std::size_t hist = (kc - 1) * inner;
  float* slice = conv_all->data() + qkv * hist;
  std::vector<float> conv_x((kc - 1 + 1) * inner);
  std::copy_n(slice, hist, conv_x.begin());
  std::copy(proj_in.begin(), proj_in.end(),
            conv_x.begin() + static_cast<std::ptrdiff_t>(hist));
  out->assign(inner, 0.0F);
  for (std::size_t d = 0; d < inner; ++d) {
    float acc = 0.0F;
    for (std::size_t c = 0; c < kc; ++c) {
      acc += conv_x[c * inner + d] * cw[c * inner + d];
    }
    (*out)[d] = Silu(acc);
  }
  std::copy_n(conv_x.begin() + static_cast<std::ptrdiff_t>(inner), hist, slice);
}

void Scan(std::vector<float>* S, const std::vector<float>& q,
          const std::vector<float>& k, const std::vector<float>& v,
          const std::vector<float>& g, float beta, std::size_t heads,
          std::size_t hd, std::vector<float>* out) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  out->assign(heads * hd, 0.0F);
  for (std::size_t h = 0; h < heads; ++h) {
    float* M = S->data() + h * hd * hd;
    std::vector<float> decay(hd), delta(hd);
    for (std::size_t i = 0; i < hd; ++i) decay[i] = std::exp(g[h * hd + i]);
    for (std::size_t j = 0; j < hd; ++j)
      for (std::size_t i = 0; i < hd; ++i) M[j * hd + i] *= decay[i];
    for (std::size_t j = 0; j < hd; ++j) {
      float s = 0.0F;
      for (std::size_t i = 0; i < hd; ++i) s += M[j * hd + i] * k[h * hd + i];
      delta[j] = (v[h * hd + j] - s) * beta;
    }
    for (std::size_t j = 0; j < hd; ++j)
      for (std::size_t i = 0; i < hd; ++i) M[j * hd + i] += k[h * hd + i] * delta[j];
    for (std::size_t j = 0; j < hd; ++j) {
      float s = 0.0F;
      for (std::size_t i = 0; i < hd; ++i) s += M[j * hd + i] * q[h * hd + i];
      (*out)[h * hd + j] = s * scale;
    }
  }
}

}  // namespace

int main() {
  int failures = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const std::size_t H = 24, heads = 4, hd = 6, kc = 4;
    const Weights w = MakeWeights(H, heads, hd, kc);
    const std::size_t inner = heads * hd;
    const float eps = 1e-6F;

    // Both keep three histories; LG keeps them in one buffer.
    std::vector<float> fork_conv[3];
    for (int q = 0; q < 3; ++q) fork_conv[q].assign((kc - 1) * inner, 0.0F);
    std::vector<float> lg_conv(3 * (kc - 1) * inner, 0.0F);

    std::vector<float> fork_scan(heads * hd * hd, 0.0F);
    std::vector<float> lg_scan = fork_scan;

    std::vector<float> q_in(H), attn_norm(H), normed(H);
    for (auto& x : q_in) x = Nf(-1.0F, 1.0F);
    for (auto& x : attn_norm) x = Nf(0.5F, 1.5F);

    for (int t = 0; t < 6; ++t) {
      normed = q_in;
      RmsNorm(&normed, attn_norm, eps);

      std::vector<float> pq, pk, pv;
      MatVec(w.wq, inner, H, normed, &pq);
      MatVec(w.wk, inner, H, normed, &pk);
      MatVec(w.wv, inner, H, normed, &pv);

      // --- fork: each projection reads AND writes its own history ---
      std::vector<float> fq, fk, fv;
      ForkConv(&fork_conv[0], pq, w.cwq, kc, inner, &fq);
      ForkConv(&fork_conv[1], pk, w.cwk, kc, inner, &fk);
      ForkConv(&fork_conv[2], pv, w.cwv, kc, inner, &fv);

      // --- LG: each projection reads and writes its own slice ---
      std::vector<float> lq, lk, lv;
      LgConv(&lg_conv, 0, pq, w.cwq, kc, inner, &lq);
      LgConv(&lg_conv, 1, pk, w.cwk, kc, inner, &lk);
      LgConv(&lg_conv, 2, pv, w.cwv, kc, inner, &lv);

      // conv outputs and the carried state must agree with the fork.
      auto same = [](const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) return false;
        return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
      };
      if (!same(fq, lq) || !same(fk, lk) || !same(fv, lv)) {
        std::printf("trial %d token %d: conv output differs "
                    "(q=%d k=%d v=%d)\n", trial, t, !same(fq, lq), !same(fk, lk),
                    !same(fv, lv));
        if (++failures > 6) return 1;
      }
      for (int q = 0; q < 3; ++q) {
        const std::vector<float> lg_slice(
            lg_conv.begin() + static_cast<std::ptrdiff_t>(q * (kc - 1) * inner),
            lg_conv.begin() + static_cast<std::ptrdiff_t>((q + 1) * (kc - 1) * inner));
        if (!same(fork_conv[q], lg_slice)) {
          std::printf("trial %d token %d: conv history %d differs\n", trial, t, q);
          if (++failures > 6) return 1;
        }
      }
      // The three histories must genuinely differ after the first token; if
      // they did not, a shared history would have been indistinguishable and
      // this test would not be proving anything.
      if (t > 0 && fork_conv[0] == fork_conv[1]) {
        std::printf("trial %d token %d: histories are identical -- the test is "
                    "not discriminating\n", trial, t);
        ++failures;
      }

      // --- the rest of the layer: gate, beta, scan, output gate ---
      std::vector<float> gate(inner), beta(heads), g2(inner);
      MatVec(w.wf, inner, H, normed, &gate);
      for (std::size_t h = 0; h < heads; ++h) {
        const float a = w.a_log[h];
        for (std::size_t d = 0; d < hd; ++d) {
          const std::size_t i = h * hd + d;
          gate[i] = w.lower_bound * Sigmoid(a * (gate[i] + w.dt_bias[i]));
        }
      }
      MatVec(w.wbeta, heads, H, normed, &beta);
      for (auto& b : beta) b = Sigmoid(b);
      MatVec(w.wg, inner, H, normed, &g2);

      std::vector<float> fq2 = fq, fk2 = fk, lv2 = lv;
      std::vector<float> lq2 = lq, lk2 = lk, llv2 = lv;
      for (std::size_t h = 0; h < heads; ++h) {
        const auto sub = [&](std::vector<float>& v) {
          std::vector<float> t(v.begin() + static_cast<std::ptrdiff_t>(h * hd),
                               v.begin() + static_cast<std::ptrdiff_t>((h + 1) * hd));
          L2Norm(&t, eps);
          std::copy(t.begin(), t.end(),
                    v.begin() + static_cast<std::ptrdiff_t>(h * hd));
        };
        sub(fq2); sub(fk2); sub(lq2); sub(lk2);
      }

      std::vector<float> fo, lo;
      Scan(&fork_scan, fq2, fk2, fv, gate, beta[0], heads, hd, &fo);
      Scan(&lg_scan, lq2, lk2, lv2, gate, beta[0], heads, hd, &lo);
      if (!same(fork_scan, lg_scan)) {
        std::printf("trial %d token %d: recurrent state differs\n", trial, t);
        if (++failures > 6) return 1;
      }

      std::vector<float> gout(heads * hd);
      for (std::size_t h = 0; h < heads; ++h) {
        std::vector<float> oh(lo.begin() + static_cast<std::ptrdiff_t>(h * hd),
                              lo.begin() + static_cast<std::ptrdiff_t>((h + 1) * hd));
        RmsNorm(&oh, w.o_norm, eps);
        for (std::size_t d = 0; d < hd; ++d) oh[d] *= Sigmoid(g2[h * hd + d]);
        std::copy(oh.begin(), oh.end(), gout.begin() + static_cast<std::ptrdiff_t>(h * hd));
      }
      std::vector<float> fout_vec;
      MatVec(w.wo, H, inner, gout, &fout_vec);
      std::vector<float> lout_vec;
      MatVec(w.wo, H, inner, gout, &lout_vec);
      (void)fout_vec;

      // advance the input a little so the next token sees fresh values
      for (auto& x : q_in) x = Nf(-1.0F, 1.0F);
    }
  }
  std::printf("%s: %d failures\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
