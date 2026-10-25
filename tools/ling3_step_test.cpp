// ling3_step_test — structural check that ReferenceModel::Step WIRES every
// layer type together and stays finite across several tokens.
//
// What it proves: the graph runs end to end over a KDA block, an MLA block, a
// dense leading block and a MoE block with a clamped shared expert, without an
// out-of-range access, a size mismatch, or a non-finite distribution. That is
// the class of mistake the per-operator tests structurally cannot see -- a wrong
// buffer bound, a dense block taking the MoE path, a state tensor not sized for
// the layer type.
//
// What it does NOT prove: numerical equality with the fork. That is covered
// operator by operator by kda_layer_test, mla_layer_test and moe_router_test,
// and end to end by the real gates, which diff against llama-eval-callback on
// the actual artifact. A full synthetic re-implementation of the whole graph was
// started and abandoned: at ~400 lines of second-hand transcription it became
// more likely to encode my own misunderstanding than to catch one in the
// reference, and the real bisect gives a sharper answer in minutes.
//
// Runs on the Mac. No model file, no GPU, all tensors F32, and no gufo_core at
// runtime (the few quant entry points are stubbed because F32 never reaches
// them).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

#include "src/models/ling3/reference.hpp"

// ---------------------------------------------------------------------------
// Stubs for gufo::quant entry points that cpu_ops.cpp references. The synthetic
// model is entirely F32, so none of these is ever called; they exist only to
// satisfy the linker.
// ---------------------------------------------------------------------------
namespace gufo::quant {
float Fp16ToFloat(std::uint16_t h) noexcept {
  const int s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  float v;
  if (e == 0) v = static_cast<float>(m / 1024.0) * 1.52587890625e-5F;
  else v = static_cast<float>(1 + m / 1024.0) * std::pow(2.0F, e - 15);
  return s ? -v : v;
}
void DequantizeQ6_K(const void*, float*, std::size_t) {}
void DequantizeQ8_0(const void*, float*, std::size_t) {}
float DotProductQ6_K(const void*, std::span<const float>, std::size_t) { return 0; }
float DotProductQ8_0(const void*, std::span<const float>, std::size_t) { return 0; }
}  // namespace gufo::quant

namespace lg = gufo::models::ling3;

// TensorRef::RowBytes lives in weights.cpp, which would drag in the whole
// GgufReader. This test only ever uses F32 tensors, so the one function the
// scalar ops need is defined here instead of linking it.
namespace gufo::models::ling3 {
std::size_t TensorRef::RowBytes() const noexcept {
  return static_cast<std::size_t>(cols) * sizeof(float);
}
}  // namespace gufo::models::ling3

// ---------------------------------------------------------------------------
// A tiny synthetic model.
// ---------------------------------------------------------------------------
namespace {

std::uint64_t rs = 0xFEEDFACE12345678ULL;
double Nx() {
  rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<double>(rs >> 11) / 9007199254740992.0;
}
float Nf(float lo, float hi) { return static_cast<float>(lo + Nx() * (hi - lo)); }

constexpr std::size_t kH = 16, kVocab = 8, kHeads = 2, kHd = 4, kKc = 4;
constexpr std::size_t kInner = kHeads * kHd;                 // 8
constexpr std::size_t kKv = 4, kDk = 6, kDv = 3, kRot = 2;   // nope = 4
constexpr std::size_t kFfn = 10, kE = 8, kEused = 2, kG = 4, kGu = 2, kEff = 5;

lg::TensorRef Vec(std::vector<float>& data) {
  lg::TensorRef t;
  t.data = data.data();
  t.type = gufo::core::GgmlType::kF32;
  t.cols = data.size();
  t.rows = 1;
  t.experts = 1;
  t.name = "v";
  return t;
}
lg::TensorRef Mat(std::vector<float>& data, std::size_t rows, std::size_t cols) {
  lg::TensorRef t;
  t.data = data.data();
  t.type = gufo::core::GgmlType::kF32;
  t.cols = cols;
  t.rows = rows;
  t.experts = 1;
  t.name = "m";
  return t;
}
/// Expert-stacked [cols, rows, experts], matching GGUF ne order.
lg::TensorRef Stacked(std::vector<float>& data, std::size_t rows, std::size_t cols,
                      std::size_t experts) {
  lg::TensorRef t;
  t.data = data.data();
  t.type = gufo::core::GgmlType::kF32;
  t.cols = cols;
  t.rows = rows;
  t.experts = experts;
  t.name = "s";
  return t;
}
std::vector<float> Rnd(std::size_t n, float s) {
  std::vector<float> v(n);
  for (auto& x : v) x = Nf(-s, s);
  return v;
}

struct Blob {
  // Per block: norms, attention, ffn.
  std::vector<float> attn_norm[3], ffn_norm[3], o_norm[3];
  std::vector<float> wq[3], wk[3], wv[3], wf[3], wg[3], wbeta[3], a_log[3], dt[3];
  std::vector<float> cwq[3], cwk[3], cwv[3], kwo[3];
  std::vector<float> m_wq[3], m_kvamqa[3], m_kvanorm[3], m_wkb[3], m_wvb[3],
      m_gate[3], m_wo[3];
  std::vector<float> d_gate[3], d_up[3], d_down[3];
  std::vector<float> r_inp[3], r_bias[3], r_ge[3], r_ue[3], r_de[3];
  std::vector<float> r_gs[3], r_us[3], r_ds[3];
  std::vector<float> tok_embd, out_norm, out;
};

Blob blob;
lg::ModelWeights W;

void BuildModel() {
  W.config = lg::Config{};
  auto& c = W.config;
  c.block_count = 3;
  c.nextn_layers = 0;
  c.num_layers = 3;
  c.hidden_size = kH;
  c.vocab_size = kVocab;
  c.context_length = 64;
  c.rms_eps = 1e-6F;
  c.num_heads = kHeads;
  c.kda_head_dim = kHd;
  c.kda_conv_kernel = kKc;
  c.kda_gate_lower_bound = -5.0F;
  c.mla_kv_lora_rank = kKv;
  c.mla_head_dim_k = kDk;
  c.mla_head_dim_v = kDv;
  c.rotary_dim = kRot;
  c.rope_mode = 0;
  c.rope_theta = 6000000.0F;
  c.ffn_length = kFfn;
  c.dense_lead_layers = 1;
  c.num_experts = kE;
  c.num_experts_used = kEused;
  c.num_expert_groups = kG;
  c.num_group_used = kGu;
  c.expert_ff = kEff;
  c.shared_expert_ff = kEff;
  c.num_shared_experts = 1;
  c.expert_weights_scale = 2.5F;
  c.expert_weights_norm = true;
  c.expert_gating_func = 2;
  c.swiglu_clamp_exp.assign(3, 0.0F);
  c.swiglu_clamp_shexp = {0.0F, 5.0F, 0.0F};  // exercise the clamp path
  c.head_count_kv = {0, 1, 0};                 // KDA, MLA, KDA

  blob.tok_embd = Rnd(kH * kVocab, 0.3F);
  blob.out_norm = Rnd(kH, 1.0F);
  blob.out = Rnd(kH * kVocab, 0.3F);
  W.token_embd = Mat(blob.tok_embd, kVocab, kH);
  W.output_norm.tensor = Vec(blob.out_norm);
  W.output = Mat(blob.out, kVocab, kH);

  W.layers.resize(3);
  for (std::size_t i = 0; i < 3; ++i) {
    auto& l = W.layers[i];
    l.mla = c.IsMla(static_cast<std::uint32_t>(i));
    blob.attn_norm[i] = Rnd(kH, 1.0F);
    blob.ffn_norm[i] = Rnd(kH, 1.0F);
    l.attn_norm.tensor = Vec(blob.attn_norm[i]);
    l.ffn_norm.tensor = Vec(blob.ffn_norm[i]);

    if (l.mla) {
      blob.m_wq[i] = Rnd(kHeads * kDk * kH, 0.3F);
      blob.m_kvamqa[i] = Rnd((kKv + kRot) * kH, 0.3F);
      blob.m_kvanorm[i] = Rnd(kKv, 1.0F);
      blob.m_wkb[i] = Rnd(kHeads * kKv * (kDk - kRot), 0.3F);
      blob.m_wvb[i] = Rnd(kHeads * kDv * kKv, 0.3F);
      blob.m_gate[i] = Rnd(kHeads * kH, 0.3F);
      blob.m_wo[i] = Rnd(kH * kHeads * kDv, 0.3F);
      l.mla_weights.wq = Mat(blob.m_wq[i], kHeads * kDk, kH);
      l.mla_weights.wkv_a_mqa = Mat(blob.m_kvamqa[i], kKv + kRot, kH);
      l.mla_weights.kv_a_norm.tensor = Vec(blob.m_kvanorm[i]);
      l.mla_weights.wk_b = Stacked(blob.m_wkb[i], kKv, kDk - kRot, kHeads);
      l.mla_weights.wv_b = Stacked(blob.m_wvb[i], kDv, kKv, kHeads);
      l.mla_weights.wqkv_gate = Mat(blob.m_gate[i], kHeads, kH);
      l.mla_weights.wo = Mat(blob.m_wo[i], kH, kHeads * kDv);
    } else {
      blob.wq[i] = Rnd(kInner * kH, 0.3F);
      blob.wk[i] = Rnd(kInner * kH, 0.3F);
      blob.wv[i] = Rnd(kInner * kH, 0.3F);
      blob.cwq[i] = Rnd(kKc * kInner, 0.5F);
      blob.cwk[i] = Rnd(kKc * kInner, 0.5F);
      blob.cwv[i] = Rnd(kKc * kInner, 0.5F);
      blob.wf[i] = Rnd(kInner * kH, 0.3F);
      blob.wg[i] = Rnd(kInner * kH, 0.3F);
      blob.wbeta[i] = Rnd(kHeads * kH, 0.3F);
      blob.a_log[i] = Rnd(kHeads, 0.5F);
      blob.dt[i] = Rnd(kInner, 1.0F);
      blob.o_norm[i] = Rnd(kHd, 1.0F);
      l.kda.wq = Mat(blob.wq[i], kInner, kH);
      l.kda.wk = Mat(blob.wk[i], kInner, kH);
      l.kda.wv = Mat(blob.wv[i], kInner, kH);
      // The binder re-views the [kc, 1, inner] conv weight as [kc][inner].
      l.kda.conv_q = Mat(blob.cwq[i], kInner, kKc);
      l.kda.conv_k = Mat(blob.cwk[i], kInner, kKc);
      l.kda.conv_v = Mat(blob.cwv[i], kInner, kKc);
      l.kda.w_f = Mat(blob.wf[i], kInner, kH);
      l.kda.w_g = Mat(blob.wg[i], kInner, kH);
      l.kda.w_beta = Mat(blob.wbeta[i], kHeads, kH);
      l.kda.a_log = Mat(blob.a_log[i], kHeads, 1);
      l.kda.dt_bias.tensor = Vec(blob.dt[i]);
      l.kda.o_norm.tensor = Vec(blob.o_norm[i]);
      blob.kwo[i] = Rnd(kH * kInner, 0.3F);
      l.kda.wo = Mat(blob.kwo[i], kH, kInner);
    }

    if (i < c.dense_lead_layers) {
      blob.d_gate[i] = Rnd(kFfn * kH, 0.3F);
      blob.d_up[i] = Rnd(kFfn * kH, 0.3F);
      blob.d_down[i] = Rnd(kH * kFfn, 0.3F);
      l.ffn_gate = Mat(blob.d_gate[i], kFfn, kH);
      l.ffn_up = Mat(blob.d_up[i], kFfn, kH);
      l.ffn_down = Mat(blob.d_down[i], kH, kFfn);
    } else {
      blob.r_inp[i] = Rnd(kE * kH, 0.3F);
      blob.r_bias[i] = Rnd(kE, 0.5F);
      blob.r_ge[i] = Rnd(kE * kEff * kH, 0.3F);
      blob.r_ue[i] = Rnd(kE * kEff * kH, 0.3F);
      blob.r_de[i] = Rnd(kE * kH * kEff, 0.3F);
      blob.r_gs[i] = Rnd(kEff * kH, 0.3F);
      blob.r_us[i] = Rnd(kEff * kH, 0.3F);
      blob.r_ds[i] = Rnd(kH * kEff, 0.3F);
      l.moe.gate_inp = Mat(blob.r_inp[i], kE, kH);
      l.moe.probs_bias.tensor = Vec(blob.r_bias[i]);
      l.moe.gate_exps = Stacked(blob.r_ge[i], kEff, kH, kE);
      l.moe.up_exps = Stacked(blob.r_ue[i], kEff, kH, kE);
      l.moe.down_exps = Stacked(blob.r_de[i], kH, kEff, kE);
      l.moe.gate_shexp = Mat(blob.r_gs[i], kEff, kH);
      l.moe.up_shexp = Mat(blob.r_us[i], kEff, kH);
      l.moe.down_shexp = Mat(blob.r_ds[i], kH, kEff);
    }
  }
}

}  // namespace

int main() {
  BuildModel();
  lg::ReferenceModel m(W, 64);

  std::vector<float> logits;
  std::string err;
  const std::int32_t toks[] = {3, 5, 1};
  for (std::size_t t = 0; t < 3; ++t) {
    if (!m.Step(toks[t], logits, &err)) {
      std::printf("FAIL  step %zu: %s\n", t, err.c_str());
      return 1;
    }
  }
  std::printf("ran 3 steps, position=%u, logits=%zu\n", m.Position(), logits.size());

  // The point is that Step completed and produced a well-formed distribution
  // across a KDA block, an MLA block and both a dense and a MoE FFN, including
  // the clamped shared expert. Exact numbers are already pinned per-operator by
  // the other tests; this one exists to prove the graph is wired the way
  // bailingmoe3.cpp builds it rather than to re-derive the arithmetic.
  if (logits.size() != kVocab) {
    std::printf("FAIL  logits size %zu\n", logits.size());
    return 1;
  }
  bool any_nan = false;
  for (float v : logits) {
    if (!std::isfinite(v)) any_nan = true;
  }
  if (any_nan) {
    std::printf("FAIL  non-finite logits\n");
    return 1;
  }
  std::printf("PASS: Step runs KDA + MLA + dense FFN + clamped MoE over %d "
              "tokens, finite throughout (structural, not numeric)\n", 3);
  return 0;
}
