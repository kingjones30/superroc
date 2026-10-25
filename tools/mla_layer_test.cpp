// mla_layer_test — verify LG's absorbed gated-MLA layer against a literal
// transcription of the fork's graph. Runs on the Mac; no model.
//
// Ling ships attn_k_b / attn_v_b, not attn_kv_b, so the fork takes the
// ABSORBED branch of bailingmoe3.cpp: the non-rotated part of each query head is
// mapped into the 512-dim latent space by wk_b, both sides of the dot product
// then live in a 576-dim (512 latent + 64 rope) single-head cache, and the value
// comes back out through wv_b. That is what this tests, end to end, over
// several tokens so the latent cache accumulates.
//
// Weights are dense floats by construction; the quantized decode path is
// verified separately and bit-exactly by tools/ling3_rocmfp4_test and
// tools/ling3_dequant_parity.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

std::uint64_t rs = 0xABCDEF0123456789ULL;
double Nx() {
  rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<double>(rs >> 11) / 9007199254740992.0;
}
float Nf(float lo, float hi) { return static_cast<float>(lo + Nx() * (hi - lo)); }
float Sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }

constexpr std::size_t kHeads = 4, kDk = 8, kDv = 5, kKv = 6, kRot = 4, kH = 12;
constexpr std::size_t kNope = kDk - kRot;

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

/// NORM rope: pair (i, i + rot/2) rotates by pos * base^(-2i/rot).
void Rope(float* head, std::size_t n, std::size_t rot, std::size_t pos,
          float base) {
  const std::size_t half = rot / 2;
  for (std::size_t i = 0; i < half; ++i) {
    const float freq = 1.0F / std::pow(base, 2.0F * static_cast<float>(i) /
                                                  static_cast<float>(rot));
    const float ang = static_cast<float>(pos) * freq;
    const float c = std::cos(ang), s = std::sin(ang);
    const float a = head[i], b = head[i + half];
    head[i] = a * c - b * s;
    head[i + half] = a * s + b * c;
  }
  (void)n;
}

struct W {
  std::vector<float> wq;            // [heads*dk][H]
  std::vector<float> wkv_a_mqa;     // [kv+rot][H]
  std::vector<float> kv_a_norm;     // [kv]
  std::vector<float> wk_b;          // per head: [kv][nope]
  std::vector<float> wv_b;          // per head: [dv][kv]
  std::vector<float> wqkv_gate;     // [heads][H]
  std::vector<float> wo;            // [H][heads*dv]
  float theta = 6000000.0F, eps = 1e-6F;
};

W MakeW() {
  W w;
  auto rnd = [&](std::vector<float>& v, std::size_t n, float s) {
    v.resize(n);
    for (auto& x : v) x = Nf(-s, s);
  };
  rnd(w.wq, kHeads * kDk * kH, 0.3F);
  rnd(w.wkv_a_mqa, (kKv + kRot) * kH, 0.3F);
  rnd(w.kv_a_norm, kKv, 1.0F);
  rnd(w.wk_b, kHeads * kKv * kNope, 0.3F);
  rnd(w.wv_b, kHeads * kDv * kKv, 0.3F);
  rnd(w.wqkv_gate, kHeads * kH, 0.3F);
  rnd(w.wo, kH * kHeads * kDv, 0.3F);
  return w;
}

// ---------------------------------------------------------------------------
// LG's MlaLayer, mirroring src/models/ling3/reference.cpp.
// ---------------------------------------------------------------------------
void LgLayer(const W& w, std::vector<float>* latent, const std::vector<float>& x,
             std::size_t pos, std::vector<float>* out) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(kDk));

  std::vector<float> q_all, kv_a;
  MatVec(w.wq, kHeads * kDk, kH, x, &q_all);
  std::vector<float> q_pe(kHeads * kRot);
  for (std::size_t h = 0; h < kHeads; ++h) {
    float* head = q_all.data() + h * kDk;
    Rope(head + kNope, kDk, kRot, pos, w.theta);
    std::copy_n(head + kNope, kRot, q_pe.begin() + static_cast<std::ptrdiff_t>(h * kRot));
  }

  MatVec(w.wkv_a_mqa, kKv + kRot, kH, x, &kv_a);
  std::vector<float> k_pe(kRot);
  std::copy_n(kv_a.begin() + static_cast<std::ptrdiff_t>(kKv), kRot, k_pe.begin());
  Rope(k_pe.data(), kRot, kRot, pos, w.theta);
  std::vector<float> cmpr(kv_a.begin(), kv_a.begin() + static_cast<std::ptrdiff_t>(kKv));
  RmsNorm(&cmpr, w.kv_a_norm, w.eps);

  latent->insert(latent->end(), cmpr.begin(), cmpr.end());
  latent->insert(latent->end(), k_pe.begin(), k_pe.end());

  std::vector<float> q_abs(kHeads * kKv);
  for (std::size_t h = 0; h < kHeads; ++h) {
    const std::vector<float> qn(q_all.begin() + static_cast<std::ptrdiff_t>(h * kDk),
                                q_all.begin() + static_cast<std::ptrdiff_t>(h * kDk + kNope));
    for (std::size_t r = 0; r < kKv; ++r) {
      float s = 0.0F;
      const float* Wb = w.wk_b.data() + (h * kKv + r) * kNope;
      for (std::size_t i = 0; i < kNope; ++i) s += Wb[i] * qn[i];
      q_abs[h * kKv + r] = s;
    }
  }

  const std::size_t n_pos = latent->size() / (kKv + kRot);
  // Per-POSITION scores. Indexing this by head was the bug this test exists to
  // catch: with a one-position cache it is indistinguishable, from the second
  // token on it weights every position by the last exponential.
  std::vector<float> scores(n_pos), out_latent(kKv), head_out(kHeads * kDv);
  for (std::size_t h = 0; h < kHeads; ++h) {
    const float* qh = q_abs.data() + h * kKv;
    const float* qp = q_pe.data() + h * kRot;
    float mx = -std::numeric_limits<float>::infinity();
    for (std::size_t p = 0; p < n_pos; ++p) {
      const float* c = latent->data() + p * (kKv + kRot);
      float s = 0.0F;
      for (std::size_t i = 0; i < kKv; ++i) s += qh[i] * c[i];
      for (std::size_t i = 0; i < kRot; ++i) s += qp[i] * c[kKv + i];
      scores[p] = s * scale;
      mx = std::max(mx, scores[p]);
    }
    // ggml scales into the score tensor first and the softmax subtracts the
    // max in a separate op, so `s * scale - mx` must NOT be written as one
    // expression here: that lets the compiler contract it into an FMA and
    // shifts the result by an ULP. reference.cpp has the same two-step shape.
    float denom = 0.0F;
    for (std::size_t p = 0; p < n_pos; ++p) {
      scores[p] = std::exp(scores[p] - mx);
      denom += scores[p];
    }
    std::fill(out_latent.begin(), out_latent.end(), 0.0F);
    for (std::size_t p = 0; p < n_pos; ++p) {
      const float a = scores[p] / denom;
      const float* c = latent->data() + p * (kKv + kRot);
      for (std::size_t i = 0; i < kKv; ++i) out_latent[i] += a * c[i];
    }
    for (std::size_t r = 0; r < kDv; ++r) {
      float s = 0.0F;
      const float* Wv = w.wv_b.data() + (h * kDv + r) * kKv;
      for (std::size_t i = 0; i < kKv; ++i) s += Wv[i] * out_latent[i];
      head_out[h * kDv + r] = s;
    }
  }

  std::vector<float> gate(kHeads);
  MatVec(w.wqkv_gate, kHeads, kH, x, &gate);
  for (std::size_t h = 0; h < kHeads; ++h) {
    const float g = Sigmoid(gate[h]);
    for (std::size_t r = 0; r < kDv; ++r) head_out[h * kDv + r] *= g;
  }
  MatVec(w.wo, kH, kHeads * kDv, head_out, out);
}

// ---------------------------------------------------------------------------
// An independent reading of the fork's graph, written op by op.
// ---------------------------------------------------------------------------
void ForkLayer(const W& w, std::vector<float>* cache, const std::vector<float>& x,
               std::size_t pos, std::vector<float>* out) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(kDk));

  // q = wq x, laid out head-major; the last `rot` dims of each head rotate.
  std::vector<float> q(kHeads * kDk);
  for (std::size_t h = 0; h < kHeads; ++h) {
    for (std::size_t r = 0; r < kDk; ++r) {
      float s = 0.0F;
      const float* Wq = w.wq.data() + (h * kDk + r) * kH;
      for (std::size_t c = 0; c < kH; ++c) s += Wq[c] * x[c];
      q[h * kDk + r] = s;
    }
  }
  for (std::size_t h = 0; h < kHeads; ++h) {
    Rope(q.data() + h * kDk + kNope, kDk, kRot, pos, w.theta);
  }

  // kv_cmpr_pe = wkv_a_mqa x
  std::vector<float> kvp(kKv + kRot);
  for (std::size_t r = 0; r < kKv + kRot; ++r) {
    float s = 0.0F;
    const float* Wk = w.wkv_a_mqa.data() + r * kH;
    for (std::size_t c = 0; c < kH; ++c) s += Wk[c] * x[c];
    kvp[r] = s;
  }
  for (std::size_t i = 0; i < kRot; ++i) kvp[kKv + i] = kvp[kKv + i];
  {
    std::vector<float> kp(kRot);
    for (std::size_t i = 0; i < kRot; ++i) kp[i] = kvp[kKv + i];
    Rope(kp.data(), kRot, kRot, pos, w.theta);
    for (std::size_t i = 0; i < kRot; ++i) kvp[kKv + i] = kp[i];
  }
  std::vector<float> cmpr(kKv);
  for (std::size_t i = 0; i < kKv; ++i) cmpr[i] = kvp[i];
  RmsNorm(&cmpr, w.kv_a_norm, w.eps);

  // The cache row is [normalised latent | rotated key part].
  for (std::size_t i = 0; i < kKv; ++i) cache->push_back(cmpr[i]);
  for (std::size_t i = 0; i < kRot; ++i) cache->push_back(kvp[kKv + i]);

  const std::size_t n_pos = cache->size() / (kKv + kRot);
  // One softmax over positions, per head, then wv_b per head.
  std::vector<float> head_out(kHeads * kDv);
  std::vector<float> sm(n_pos), acc(kKv);
  for (std::size_t h = 0; h < kHeads; ++h) {
    // q_absorbed = wk_b[h] @ q_nope[h]
    std::vector<float> qa(kKv);
    for (std::size_t r = 0; r < kKv; ++r) {
      float s = 0.0F;
      const float* Wb = w.wk_b.data() + (h * kKv + r) * kNope;
      for (std::size_t i = 0; i < kNope; ++i) s += Wb[i] * q[h * kDk + i];
      qa[r] = s;
    }
    float mx = -std::numeric_limits<float>::infinity();
    for (std::size_t p = 0; p < n_pos; ++p) {
      const float* c = cache->data() + p * (kKv + kRot);
      float s = 0.0F;
      for (std::size_t i = 0; i < kKv; ++i) s += qa[i] * c[i];
      for (std::size_t i = 0; i < kRot; ++i) s += q[h * kDk + kNope + i] * c[kKv + i];
      sm[p] = s * scale;
      mx = std::max(mx, sm[p]);
    }
    float denom = 0.0F;
    for (std::size_t p = 0; p < n_pos; ++p) {
      sm[p] = std::exp(sm[p] - mx);
      denom += sm[p];
    }
    for (std::size_t i = 0; i < kKv; ++i) acc[i] = 0.0F;
    for (std::size_t p = 0; p < n_pos; ++p) {
      const float a = sm[p] / denom;
      for (std::size_t i = 0; i < kKv; ++i) acc[i] += a * (*cache)[p * (kKv + kRot) + i];
    }
    for (std::size_t r = 0; r < kDv; ++r) {
      float s = 0.0F;
      const float* Wv = w.wv_b.data() + (h * kDv + r) * kKv;
      for (std::size_t i = 0; i < kKv; ++i) s += Wv[i] * acc[i];
      head_out[h * kDv + r] = s;
    }
  }

  std::vector<float> gate(kHeads);
  for (std::size_t h = 0; h < kHeads; ++h) {
    float s = 0.0F;
    const float* Wg = w.wqkv_gate.data() + h * kH;
    for (std::size_t c = 0; c < kH; ++c) s += Wg[c] * x[c];
    gate[h] = Sigmoid(s);
  }
  for (std::size_t h = 0; h < kHeads; ++h) {
    for (std::size_t r = 0; r < kDv; ++r) head_out[h * kDv + r] *= gate[h];
  }
  out->assign(kH, 0.0F);
  for (std::size_t r = 0; r < kH; ++r) {
    float s = 0.0F;
    const float* Wo = w.wo.data() + r * kHeads * kDv;
    for (std::size_t c = 0; c < kHeads * kDv; ++c) s += Wo[c] * head_out[c];
    (*out)[r] = s;
  }
}

}  // namespace

int main() {
  int failures = 0;
  for (int trial = 0; trial < 200; ++trial) {
    const W w = MakeW();
    std::vector<float> lg_cache, fork_cache;
    for (std::size_t pos = 0; pos < 5; ++pos) {
      std::vector<float> x(kH);
      for (auto& v : x) v = Nf(-1.0F, 1.0F);

      std::vector<float> lo, fo;
      LgLayer(w, &lg_cache, x, pos, &lo);
      ForkLayer(w, &fork_cache, x, pos, &fo);

      if (lg_cache != fork_cache) {
        std::printf("trial %d pos %zu: latent cache differs\\n", trial, pos);
        if (++failures > 4) return 1;
      }
      if (lo.size() != fo.size() ||
          std::memcmp(lo.data(), fo.data(), lo.size() * sizeof(float)) != 0) {
        std::printf("trial %d pos %zu: layer output differs "
                    "(lg[0]=%.9g fork[0]=%.9g)\\n", trial, pos,
                    static_cast<double>(lo[0]), static_cast<double>(fo[0]));
        if (++failures > 4) return 1;
      }
    }
  }
  std::printf("%s: %d failures\\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
