// moe_router_test — verify LG's MoE router and clamped SwiGLU against a
// separate, literal transcription of the fork's build_moe_ffn. Mac, no model.
//
// The two bugs already found by this method were both "right for the first
// token". The router is the last unverified path and it feeds 40 of the 42
// blocks, so it gets the same treatment. The points being pinned:
//
//   * `exp_probs_b` biases **selection only**; the final weight is the unbiased
//     sigmoid;
//   * a group score is the sum of the **top two** experts in that group, not the
//     max and not the group mean;
//   * experts are contiguous by group: expert = group * E_g + slot;
//   * non-surviving groups are masked out, then top-K is taken;
//   * weights are sum-normalised (clamped at the smallest F16 normal) and only
//     then scaled by 2.5, and the weight multiplies **after** the down
//     projection;
//   * the SwiGLU clamp is silu(gate) clamped **above** at L, times up clamped
//     **symmetrically**, and is skipped entirely when L <= 1e-6.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

std::uint64_t rs = 0x5EED1234ABCD9876ULL;
double Nx() {
  rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<double>(rs >> 11) / 9007199254740992.0;
}
float Nf(float lo, float hi) { return static_cast<float>(lo + Nx() * (hi - lo)); }
float Sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }
float Silu(float x) { return x / (1.0F + std::exp(-x)); }

constexpr std::size_t kH = 16, kE = 32, kG = 8, kGU = 4, kK = 8, kFf = 6;
constexpr float kScale = 2.5F;
constexpr float kF16MinNormal = 6.103515625e-5F;

struct Router {
  std::vector<float> gate_inp;  // [E][H]
  std::vector<float> probs_b;   // [E]
  float clamp_limit = 0.0F;     // swiglu_clamp_exp for this layer
};

Router MakeRouter(float limit) {
  Router r;
  r.gate_inp.resize(kE * kH);
  r.probs_b.resize(kE);
  for (auto& v : r.gate_inp) v = Nf(-2.0F, 2.0F);
  for (auto& v : r.probs_b) v = Nf(-0.5F, 0.5F);
  r.clamp_limit = limit;
  return r;
}

// --- LG's MoeFfn selection, mirroring reference.cpp ----------------------
void LgSelect(const Router& r, const std::vector<float>& x,
              std::vector<std::size_t>* chosen, std::vector<float>* weights) {
  const std::size_t Eg = kE / kG;
  std::vector<float> s(kE), sel(kE);
  for (std::size_t e = 0; e < kE; ++e) {
    float acc = 0.0F;
    for (std::size_t c = 0; c < kH; ++c) acc += r.gate_inp[e * kH + c] * x[c];
    s[e] = Sigmoid(acc);
  }
  for (std::size_t e = 0; e < kE; ++e) sel[e] = s[e] + r.probs_b[e];

  std::vector<std::pair<float, std::size_t>> tmp(Eg), gs(kG);
  for (std::size_t g = 0; g < kG; ++g) {
    for (std::size_t i = 0; i < Eg; ++i) tmp[i] = {sel[g * Eg + i], g * Eg + i};
    std::partial_sort(tmp.begin(), tmp.begin() + 2, tmp.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    gs[g] = {tmp[0].first + tmp[1].first, g};
  }
  std::partial_sort(gs.begin(), gs.begin() + kGU, gs.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

  std::vector<bool> keep(kE, false);
  for (std::size_t i = 0; i < kGU; ++i) {
    const std::size_t g = gs[i].second;
    for (std::size_t i2 = 0; i2 < Eg; ++i2) keep[g * Eg + i2] = true;
  }

  std::vector<std::pair<float, std::size_t>> cand;
  for (std::size_t e = 0; e < kE; ++e) {
    if (keep[e]) cand.emplace_back(sel[e], e);
  }
  const std::size_t take = std::min(kK, cand.size());
  std::partial_sort(cand.begin(), cand.begin() + take, cand.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

  float wsum = 0.0F;
  for (std::size_t j = 0; j < take; ++j) wsum += s[cand[j].second];
  wsum = std::fmax(wsum, kF16MinNormal);

  chosen->clear();
  weights->assign(take, 0.0F);
  for (std::size_t j = 0; j < take; ++j) {
    chosen->push_back(cand[j].second);
    (*weights)[j] = s[cand[j].second] / wsum * kScale;
  }
}

// --- an independent reading of llama-graph.cpp's router --------------------
void ForkSelect(const Router& r, const std::vector<float>& x,
                std::vector<std::size_t>* chosen, std::vector<float>* weights) {
  // logits = gate_inp x ; probs = sigmoid(logits)
  // ggml's F32 mul_mat accumulates in float (ggml_vec_dot_f32 keeps a float
  // accumulator on the SIMD path), so this must be float too -- a double
  // accumulator shifts the score by an ULP and the weight with it.
  std::vector<float> probs(kE);
  for (std::size_t e = 0; e < kE; ++e) {
    float acc = 0.0F;
    for (std::size_t c = 0; c < kH; ++c) acc += r.gate_inp[e * kH + c] * x[c];
    probs[e] = Sigmoid(acc);
  }
  // selection_probs = probs + exp_probs_b  (used ONLY for choosing)
  std::vector<float> sp(kE);
  for (std::size_t e = 0; e < kE; ++e) sp[e] = probs[e] + r.probs_b[e];

  const std::size_t Eg = kE / kG;
  // group score = sum of the top 2 in the group
  std::vector<float> gscore(kG);
  for (std::size_t g = 0; g < kG; ++g) {
    std::vector<float> tmp;
    for (std::size_t i = 0; i < Eg; ++i) tmp.push_back(sp[g * Eg + i]);
    std::sort(tmp.begin(), tmp.end(), std::greater<float>());
    gscore[g] = tmp[0] + tmp[1];
  }
  std::vector<std::size_t> order(kG);
  for (std::size_t g = 0; g < kG; ++g) order[g] = g;
  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) { return gscore[a] > gscore[b]; });

  // mask everything, then unmask the top GU groups
  std::vector<float> masked(kE, -std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < kGU; ++i) {
    const std::size_t g = order[i];
    for (std::size_t i2 = 0; i2 < Eg; ++i2) masked[g * Eg + i2] = sp[g * Eg + i2];
  }
  std::vector<std::size_t> sel;
  std::vector<float> selscore;
  for (std::size_t e = 0; e < kE; ++e) {
    if (masked[e] > -std::numeric_limits<float>::infinity()) {
      sel.push_back(e);
      selscore.push_back(masked[e]);
    }
  }
  std::vector<std::size_t> ord(sel.size());
  for (std::size_t i = 0; i < ord.size(); ++i) ord[i] = i;
  std::sort(ord.begin(), ord.end(), [&](std::size_t a, std::size_t b) {
    return selscore[a] > selscore[b];
  });

  // weights come from the UNBIASED probs
  chosen->clear();
  weights->clear();
  float wsum = 0.0F;
  for (std::size_t i = 0; i < ord.size() && i < kK; ++i) {
    chosen->push_back(sel[ord[i]]);
    wsum += probs[sel[ord[i]]];
  }
  wsum = std::fmax(wsum, kF16MinNormal);
  for (std::size_t i = 0; i < chosen->size(); ++i) {
    weights->push_back(probs[(*chosen)[i]] / wsum * kScale);
  }
}

// --- the clamped SwiGLU ---------------------------------------------------
float LgSwiglu(float gate, float up, float limit) {
  if (limit <= 1e-6F) return Silu(gate) * up;
  return std::fmin(Silu(gate), limit) * std::fmin(std::fmax(up, -limit), limit);
}

float ForkSwiglu(float gate, float up, float limit) {
  if (limit > 1e-6F) {
    const float act = std::fmin(Silu(gate), limit);
    return act * std::fmin(std::fmax(up, -limit), limit);
  }
  return Silu(gate) * up;
}

}  // namespace

int main() {
  int failures = 0;

  // --- router ---
  for (int trial = 0; trial < 3000; ++trial) {
    const Router r = MakeRouter(0.0F);
    std::vector<float> x(kH);
    for (auto& v : x) v = Nf(-2.0F, 2.0F);

    std::vector<std::size_t> lc, fc;
    std::vector<float> lw, fw;
    LgSelect(r, x, &lc, &lw);
    ForkSelect(r, x, &fc, &fw);

    if (lc != fc) {
      if (++failures < 4) {
        std::printf("trial %d: expert sets differ (lg=%zu fork=%zu)\n", trial,
                    lc.size(), fc.size());
      }
      continue;
    }
    for (std::size_t j = 0; j < lc.size(); ++j) {
      if (std::memcmp(&lw[j], &fw[j], sizeof(float)) != 0) {
        if (++failures < 4) {
          std::printf("trial %d slot %zu: weight lg=%.9g fork=%.9g\n", trial, j,
                      static_cast<double>(lw[j]), static_cast<double>(fw[j]));
        }
        break;
      }
    }
  }
  std::printf("router  : %s (3000 trials)\n", failures ? "FAIL" : "OK");

  // --- clamped SwiGLU, including the skipped case ------------------------
  {
    int bad = 0;
    for (int trial = 0; trial < 20000; ++trial) {
      const float g = Nf(-10.0F, 10.0F);
      const float u = Nf(-10.0F, 10.0F);
      const float limits[] = {0.0F, 1e-7F, 1e-6F, 4.0F, 5.0F, 7.0F};
      for (const float L : limits) {
        const float a = LgSwiglu(g, u, L);
        const float b = ForkSwiglu(g, u, L);
        if (std::memcmp(&a, &b, sizeof(float)) != 0) {
          if (++bad < 4)
            std::printf("swiglu g=%.6g u=%.6g L=%.6g: lg=%.9g fork=%.9g\n",
                        static_cast<double>(g), static_cast<double>(u),
                        static_cast<double>(L), static_cast<double>(a),
                        static_cast<double>(b));
        }
      }
    }
    std::printf("swiglu  : %s (20000 trials x 6 limits, incl. L<=1e-6 skip)\n",
                bad ? "FAIL" : "OK");
    failures += bad;
  }

  // --- the bias really is selection-only: prove the two differ -------------
  {
    Router r = MakeRouter(0.0F);
    for (auto& v : r.probs_b) v = Nf(-1.0F, 1.0F);
    std::vector<float> x(kH);
    for (auto& v : x) v = Nf(-2.0F, 2.0F);
    std::vector<std::size_t> lc, fc;
    std::vector<float> lw, fw;
    LgSelect(r, x, &lc, &lw);
    ForkSelect(r, x, &fc, &fw);
    // And show that adding the bias to the *weights* would give something
    // else, i.e. the distinction is not vacuous.
    std::vector<float> with_bias = lw;
    for (std::size_t j = 0; j < with_bias.size(); ++j) with_bias[j] += r.probs_b[lc[j]];
    const bool differs = std::memcmp(with_bias.data(), lw.data(),
                                     with_bias.size() * sizeof(float)) != 0;
    std::printf("bias is selection-only: %s (adding it to weights would change "
                "them: %s)\n", lc == fc ? "ok" : "MISMATCH",
                differs ? "yes" : "no");
    if (lc != fc || !differs) ++failures;
  }

  std::printf("\n%s: %d failures\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
