#include "src/models/ling3/reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "src/models/ling3/cpu_ops.hpp"

namespace gufo::models::ling3 {

ReferenceModel::ReferenceModel(const ModelWeights& weights,
                               std::uint32_t max_context)
    : w_(weights), c_(weights.config), max_context_(max_context) {
  const std::size_t inner = c_.KdaInnerSize();
  kda_.resize(c_.num_layers);
  mla_.resize(c_.num_layers);

  const std::size_t state_elems = static_cast<std::size_t>(c_.num_heads) *
                                  c_.kda_head_dim * c_.kda_head_dim;
  for (std::uint32_t i = 0; i < c_.num_layers; ++i) {
    if (c_.IsMla(i)) {
      continue;
    }
    // Three histories, one per projection: 3 * (kernel-1) * inner.
    kda_[i].conv.assign(
        3 * static_cast<std::size_t>(c_.kda_conv_kernel - 1) * inner, 0.0F);
    kda_[i].state.assign(state_elems, 0.0F);
  }

  const std::size_t heads = c_.num_heads;
  const std::size_t hd = c_.kda_head_dim;
  normed_.resize(c_.hidden_size);
  q_.resize(inner);
  k_.resize(inner);
  v_.resize(inner);
  gate_.resize(inner);
  beta_.resize(heads);
  attn_.resize(inner);
  out_.resize(inner);
  proj_.resize(inner);
  dt_bias_.resize(inner);
  a_log_.resize(heads);
  o_norm_.resize(hd);
  conv_buf_.resize(static_cast<std::size_t>(c_.kda_conv_kernel) * inner);
  taps_.resize(c_.kda_conv_kernel);
  delta_.resize(hd);
  decay_.resize(hd);

  q_all_.resize(heads * c_.mla_head_dim_k);
  q_pe_.resize(heads * c_.rotary_dim);
  kv_a_.resize(c_.MlaKvAOutDim());
  k_pe_.resize(c_.rotary_dim);
  cmpr_.resize(c_.mla_kv_lora_rank);
  kv_a_norm_.resize(c_.mla_kv_lora_rank);
  q_abs_.resize(heads * c_.mla_kv_lora_rank);
  ml_scores_.resize(64);
  head_out_.resize(heads * c_.mla_head_dim_v);
  out_latent_.resize(c_.mla_kv_lora_rank);
  ml_gate_.resize(heads);

  const std::size_t ff = std::max(c_.expert_ff, c_.shared_expert_ff);
  gate_ff_.resize(ff);
  up_ff_.resize(ff);
  act_ff_.resize(ff);
  down_.resize(c_.hidden_size);
  scores_.resize(c_.num_experts);
  biased_.resize(c_.num_experts);
  weights_.resize(c_.num_experts);
  probs_bias_.resize(c_.num_experts);
}

void ReferenceModel::Reset() {
  for (auto& s : kda_) {
    std::fill(s.conv.begin(), s.conv.end(), 0.0F);
    std::fill(s.state.begin(), s.state.end(), 0.0F);
  }
  for (auto& m : mla_) {
    m.latent.clear();
  }
  position_ = 0;
}

std::uint64_t ReferenceModel::StateBytes() const noexcept {
  std::uint64_t bytes = 0;
  for (const auto& s : kda_) {
    bytes += static_cast<std::uint64_t>(s.conv.size() + s.state.size()) *
             sizeof(float);
  }
  for (const auto& m : mla_) {
    bytes += static_cast<std::uint64_t>(m.latent.size()) * sizeof(float);
  }
  return bytes;
}

// ---------------------------------------------------------------------------
// KDA: causal depthwise conv, safe forget gate, delta-rule scan, output gate.
// ---------------------------------------------------------------------------
void ReferenceModel::KdaLayer(const KdaWeights& k, KdaState& s,
                              std::uint32_t block, std::span<const float> x,
                              std::span<float> out) {
  const std::size_t inner = c_.KdaInnerSize();
  const std::size_t hd = c_.kda_head_dim;
  const std::size_t heads = c_.num_heads;
  const std::size_t kc = c_.kda_conv_kernel;

  // The fork keeps **three** conv histories, one per projection: the state
  // write lands at `(kv_head * n_embd_r_total + qkv * conv_state_size)` inside
  // conv_states_all, so Q, K and V each read and write their own slice. They
  // do NOT hold the same values -- after one token they carry the Q, K and V
  // projections respectively -- so a single shared history is simply wrong.
  // Each projection below therefore reads slice `qkv` and advances only that
  // slice.
  const std::size_t hist = (kc - 1) * inner;

  // Projection, then a depthwise causal conv over the (kc-1)-token history,
  // then silu. The conv weight is bound as [kc][inner], so row d holds the kc
  // taps for channel d and `taps[c]` is conv_w[c][d].
  auto project = [&](const TensorRef& wproj, const TensorRef& convw,
                     std::vector<float>& dst, const char* name,
                     std::size_t qkv) {
    float* hist_slice = s.conv.data() + qkv * hist;
    cpu::MatVec(wproj, 0, x, std::span<float>(dst.data(), inner));
    // Rows 0..kc-2 of conv_buf are the carried history, oldest first; row kc-1
    // is the current token. ggml_ssm_conv slides a kc-wide window down the
    // rows.
    std::copy_n(hist_slice, hist, conv_buf_.data());
    std::copy_n(dst.data(), inner, conv_buf_.data() + (kc - 1) * inner);
    for (std::size_t t = 0; t < 1; ++t) {
      float* dst_row = dst.data() + t * inner;
      std::fill_n(dst_row, inner, 0.0F);
      for (std::size_t d = 0; d < inner; ++d) {
        // Row d of the conv weight is the kc taps for channel d.
        cpu::DequantizeRow(convw, 0, d, taps_.data());
        float acc = dst_row[d];
        for (std::size_t c = 0; c < kc; ++c) {
          acc += conv_buf_[(t + c) * inner + d] * taps_[c];
        }
        dst_row[d] = acc;
      }
    }
    for (std::size_t i = 0; i < inner; ++i) {
      dst[i] = cpu::Silu(dst[i]);
    }
    if (trace_) {
      trace_(block, name, dst.data(), inner);
    }
    // This projection's history advances on its own: the new history is the
    // last kc-1 rows of conv_buf, i.e. [old1, old2, this projection's token].
    std::copy_n(conv_buf_.data() + inner, hist, hist_slice);
  };

  project(k.wq, k.conv_q, q_, "kda_q", 0);
  project(k.wk, k.conv_k, k_, "kda_k", 1);
  project(k.wv, k.conv_v, v_, "kda_v", 2);

  // Safe forget gate: g = lower_bound * sigmoid(exp(A_log) * (f(x) + dt_bias)).
  // ssm_a already holds exp(A_log); the lower bound is negative, so g <= 0.
  cpu::MatVec(k.w_f, 0, x, std::span<float>(gate_.data(), inner));
  cpu::ReadVector(k.dt_bias, dt_bias_.data());
  cpu::ReadVector({k.a_log}, a_log_.data());
  for (std::size_t h = 0; h < heads; ++h) {
    const float a = a_log_[h];
    for (std::size_t d = 0; d < hd; ++d) {
      const std::size_t i = h * hd + d;
      gate_[i] =
          c_.kda_gate_lower_bound * cpu::Sigmoid(a * (gate_[i] + dt_bias_[i]));
    }
  }

  if (trace_) {
    // kda_g1 is {head_dim, n_head, n_tokens} in the fork, so ggml iterates the
    // HEAD axis on the printed rows and the last printed row is the LAST head.
    // Hand the printer that head's slice so its 6 numbers (dims 0,1,2,125,126,
    // 127) line up with what the fork printed.
    trace_(block, "kda_g1", gate_.data() + (heads - 1) * hd, hd);
  }

  cpu::MatVec(k.w_beta, 0, x, std::span<float>(beta_.data(), heads));
  for (std::size_t h = 0; h < heads; ++h) {
    beta_[h] = cpu::Sigmoid(beta_[h]);
  }
  if (trace_) {
    // kda_beta is {1, n_head, n_tokens}: same head-axis alignment as kda_g1.
    trace_(block, "kda_beta", beta_.data() + (heads - 1), 1);
  }

  // q and k are L2-normalised per head before the scan; v is not.
  for (std::size_t h = 0; h < heads; ++h) {
    cpu::L2Norm(std::span<float>(q_.data() + h * hd, hd), c_.rms_eps);
    cpu::L2Norm(std::span<float>(k_.data() + h * hd, hd), c_.rms_eps);
  }

  // Delta-rule scan, in the fork's exact four-phase order. State is logical
  // S[i][j] with i the key axis (fastest) and j the value axis, which is
  // ggml's transposed storage. The output reads the state *after* the write.
  const float scan_scale = 1.0F / std::sqrt(static_cast<float>(hd));
  for (std::size_t h = 0; h < heads; ++h) {
    float* S = s.state.data() + h * hd * hd;
    const float* qh = q_.data() + h * hd;
    const float* kh = k_.data() + h * hd;
    const float* vh = v_.data() + h * hd;
    const float* gh = gate_.data() + h * hd;
    const float beta = beta_[h];

    // Phase 1: decay, per key-channel.
    for (std::size_t i = 0; i < hd; ++i) {
      decay_[i] = std::exp(gh[i]);
    }
    for (std::size_t j = 0; j < hd; ++j) {
      float* row = S + j * hd;
      for (std::size_t i = 0; i < hd; ++i) {
        row[i] *= decay_[i];
      }
    }
    // Phase 2: read the decayed state.
    for (std::size_t j = 0; j < hd; ++j) {
      const float* row = S + j * hd;
      float sum = 0.0F;
      for (std::size_t i = 0; i < hd; ++i) {
        sum += row[i] * kh[i];
      }
      delta_[j] = (vh[j] - sum) * beta;
    }
    // Phase 3: accumulate the outer product.
    for (std::size_t j = 0; j < hd; ++j) {
      float* row = S + j * hd;
      const float d = delta_[j];
      for (std::size_t i = 0; i < hd; ++i) {
        row[i] += kh[i] * d;
      }
    }
    // Phase 4: read the post-update state.
    for (std::size_t j = 0; j < hd; ++j) {
      const float* row = S + j * hd;
      float sum = 0.0F;
      for (std::size_t i = 0; i < hd; ++i) {
        sum += row[i] * qh[i];
      }
      attn_[h * hd + j] = sum * scan_scale;
    }
  }

  // Output gate: RMSNorm(o) * sigmoid(g_proj(x)), then the output projection.
  cpu::ReadVector(k.o_norm, o_norm_.data());
  cpu::MatVec(k.w_g, 0, x, std::span<float>(proj_.data(), inner));
  for (std::size_t h = 0; h < heads; ++h) {
    std::span<float> oh(attn_.data() + h * hd, hd);
    cpu::RmsNorm(oh, o_norm_.data(), c_.rms_eps);
    for (std::size_t d = 0; d < hd; ++d) {
      oh[d] *= cpu::Sigmoid(proj_[h * hd + d]);
    }
  }
  if (trace_) {
    trace_(block, "kda_gated", attn_.data(), inner);
  }
  cpu::MatVec(k.wo, 0, std::span<const float>(attn_.data(), inner), out);
}

// ---------------------------------------------------------------------------
// Gated MLA, absorbed form. The KV cache holds the compressed latent (512)
// concatenated with the shared rotated key part (64) as a single head; the
// query's non-rotated part is absorbed into the same latent space by wk_b, and
// the value side leaves through wv_b.
// ---------------------------------------------------------------------------
void ReferenceModel::MlaLayer(const MlaWeights& m, MlaState& s,
                              std::uint32_t block, std::uint32_t pos,
                              std::span<const float> x, std::span<float> out) {
  const std::size_t heads = c_.num_heads;
  const std::size_t dk = c_.mla_head_dim_k;
  const std::size_t dv = c_.mla_head_dim_v;
  const std::size_t rot = c_.rotary_dim;
  const std::size_t nope = c_.MlaQkNopeDim();
  const std::size_t kv = c_.mla_kv_lora_rank;
  const std::size_t kv_a = c_.MlaKvAOutDim();
  const float scale = 1.0F / std::sqrt(static_cast<float>(dk));

  // q = Wq x, laid out head-major as [heads][dk]; dims [0, nope) are the
  // non-rotated part and [nope, dk) rotate. Copy q_pe *after* rotating so the
  // stored copy carries the rotation.
  cpu::MatVec(m.wq, 0, x, std::span<float>(q_all_.data(), heads * dk));
  if (trace_) {
    trace_(block, "mla_q", q_all_.data(), heads * dk);
  }
  for (std::size_t h = 0; h < heads; ++h) {
    float* head = q_all_.data() + h * dk;
    cpu::Rope(head + nope, 1, rot, rot, pos, c_.rope_theta);
    std::copy_n(head + nope, rot, q_pe_.data() + h * rot);
  }

  // kv_cmpr_pe = Wkv_a x; split, rotate the key part, norm the latent.
  cpu::MatVec(m.wkv_a_mqa, 0, x, std::span<float>(kv_a_.data(), kv_a));
  std::copy_n(kv_a_.data() + kv, rot, k_pe_.data());
  cpu::Rope(k_pe_.data(), 1, rot, rot, pos, c_.rope_theta);
  std::copy_n(kv_a_.data(), kv, cmpr_.data());
  cpu::ReadVector(m.kv_a_norm, kv_a_norm_.data());
  cpu::RmsNorm(std::span<float>(cmpr_.data(), kv), kv_a_norm_.data(),
               c_.rms_eps);
  if (trace_) {
    trace_(block, "mla_kv_cmpr", cmpr_.data(), kv);
    trace_(block, "mla_k_pe", k_pe_.data(), rot);
  }

  std::vector<float>& cache = s.latent;
  const std::size_t base = cache.size();
  cache.resize(base + kv_a);
  std::copy_n(cmpr_.data(), kv, cache.begin() + base);
  std::copy_n(k_pe_.data(), rot, cache.begin() + base + kv);
  const std::size_t n_pos = cache.size() / kv_a;

  // Absorb q_nope into latent space: q_abs[h] = wk_b[h] @ q_nope[h].
  for (std::size_t h = 0; h < heads; ++h) {
    const std::span<const float> qn(q_all_.data() + h * dk, nope);
    for (std::size_t r = 0; r < kv; ++r) {
      q_abs_[h * kv + r] = cpu::DotRow(m.wk_b, h, r, qn);
    }
  }

  // One softmax over positions per head; the value at every position is the
  // 512-dim compressed latent, shared across heads.
  //
  // The scores are kept **per position**, not per head. Holding only the most
  // recent head score and reusing it in the accumulation below silently weights
  // every position by the last one's exponential -- invisible with a
  // single-position cache and wrong from the second token on.
  ml_scores_.resize(n_pos);
  for (std::size_t h = 0; h < heads; ++h) {
    const float* qh = q_abs_.data() + h * kv;
    const float* qp = q_pe_.data() + h * rot;
    float mx = -std::numeric_limits<float>::infinity();
    for (std::size_t p = 0; p < n_pos; ++p) {
      const float* cached = cache.data() + p * kv_a;
      float sum = 0.0F;
      for (std::size_t i = 0; i < kv; ++i) {
        sum += qh[i] * cached[i];
      }
      for (std::size_t i = 0; i < rot; ++i) {
        sum += qp[i] * cached[kv + i];
      }
      ml_scores_[p] = sum * scale;
      mx = std::max(mx, ml_scores_[p]);
    }
    float denom = 0.0F;
    for (std::size_t p = 0; p < n_pos; ++p) {
      ml_scores_[p] = std::exp(ml_scores_[p] - mx);
      denom += ml_scores_[p];
    }
    std::fill(out_latent_.begin(), out_latent_.end(), 0.0F);
    for (std::size_t p = 0; p < n_pos; ++p) {
      const float w = ml_scores_[p] / denom;
      const float* cached = cache.data() + p * kv_a;
      for (std::size_t i = 0; i < kv; ++i) {
        out_latent_[i] += w * cached[i];
      }
    }
    for (std::size_t r = 0; r < dv; ++r) {
      head_out_[h * dv + r] = cpu::DotRow(
          m.wv_b, h, r, std::span<const float>(out_latent_.data(), kv));
    }
  }

  // Head-wise sigmoid gate, computed from the same normed hidden state as q/kv.
  cpu::MatVec(m.wqkv_gate, 0, x, std::span<float>(ml_gate_.data(), heads));
  for (std::size_t h = 0; h < heads; ++h) {
    const float g = cpu::Sigmoid(ml_gate_[h]);
    for (std::size_t r = 0; r < dv; ++r) {
      head_out_[h * dv + r] *= g;
    }
  }
  if (trace_) {
    trace_(block, "mla_attn_out", head_out_.data(), heads * dv);
  }
  cpu::MatVec(m.wo, 0, std::span<const float>(head_out_.data(), heads * dv),
              out);
}

// ---------------------------------------------------------------------------
// Feed forward.
// ---------------------------------------------------------------------------
void ReferenceModel::DenseFfn(const LayerWeights& l, std::uint32_t block,
                              std::span<const float> x, std::span<float> out) {
  const std::size_t n_ff = c_.ffn_length;
  std::vector<float> gate(n_ff), up(n_ff), h(n_ff);
  cpu::MatVec(l.ffn_gate, 0, x, std::span<float>(gate.data(), n_ff));
  cpu::MatVec(l.ffn_up, 0, x, std::span<float>(up.data(), n_ff));
  if (trace_) {
    trace_(block, "ffn_gate", gate.data(), n_ff);
    trace_(block, "ffn_up", up.data(), n_ff);
  }

  // build_ffn applies the swiglu_clamp_shexp limit here as well.
  const float limit = block < c_.swiglu_clamp_shexp.size()
                          ? c_.swiglu_clamp_shexp[block]
                          : 0.0F;
  if (Config::ClampActive(limit)) {
    cpu::SwigluClamped(gate.data(), up.data(), h.data(), n_ff, limit);
  } else {
    for (std::size_t i = 0; i < n_ff; ++i) {
      h[i] = cpu::Silu(gate[i]) * up[i];
    }
  }
  if (trace_) {
    trace_(block, "ffn_swiglu", h.data(), n_ff);
  }
  cpu::MatVec(l.ffn_down, 0, std::span<const float>(h.data(), n_ff), out);
  if (trace_) {
    // `out` is hidden_size wide, NOT n_ff (6144 at block 0). Passing n_ff here
    // made the trace printer read past the end of the buffer.
    trace_(block, "ffn_out", out.data(), c_.hidden_size);
  }
}

void ReferenceModel::MoeFfn(const LayerWeights& l, std::uint32_t block,
                            std::span<const float> x, std::span<float> out) {
  const MoeWeights& m = l.moe;
  const std::size_t E = c_.num_experts;
  const std::size_t K = c_.num_experts_used;
  const std::size_t G = c_.num_expert_groups;
  const std::size_t GU = c_.num_group_used;
  const std::size_t Eg = c_.ExpertsPerGroup();
  const std::size_t ff = c_.expert_ff;
  const std::size_t sff = c_.shared_expert_ff;
  const std::size_t H = c_.hidden_size;
  const float limit =
      block < c_.swiglu_clamp_exp.size() ? c_.swiglu_clamp_exp[block] : 0.0F;
  const bool clamped = Config::ClampActive(limit);

  // Router: sigmoid of the raw logit, never a softmax.
  cpu::MatVec(m.gate_inp, 0, x, std::span<float>(scores_.data(), E));
  if (trace_) {
    trace_(block, "ffn_moe_logits", scores_.data(), E);
  }
  for (std::size_t e = 0; e < E; ++e) {
    scores_[e] = cpu::Sigmoid(scores_[e]);
  }
  if (trace_) {
    trace_(block, "ffn_moe_probs", scores_.data(), E);
  }
  // The expert bias steers *selection* only; the final weight is the unbiased
  // sigmoid. Getting this wrong is the classic port bug.
  cpu::ReadVector(m.probs_bias, probs_bias_.data());
  for (std::size_t e = 0; e < E; ++e) {
    biased_[e] = scores_[e] + probs_bias_[e];
  }
  if (trace_) {
    trace_(block, "ffn_moe_probs_biased", biased_.data(), E);
  }

  // Group score is the sum of the top two experts inside the group. Expert ids
  // are contiguous by group: expert = group * Eg + slot.
  std::vector<std::pair<float, std::size_t>> tmp(Eg), group_scores(G);
  for (std::size_t g = 0; g < G; ++g) {
    for (std::size_t s = 0; s < Eg; ++s) {
      tmp[s] = {biased_[g * Eg + s], g * Eg + s};
    }
    std::partial_sort(
        tmp.begin(), tmp.begin() + 2, tmp.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    group_scores[g] = {tmp[0].first + tmp[1].first, g};
  }
  std::partial_sort(
      group_scores.begin(), group_scores.begin() + GU, group_scores.end(),
      [](const auto& a, const auto& b) { return a.first > b.first; });

  std::vector<bool> keep(E, false);
  for (std::size_t i = 0; i < GU; ++i) {
    const std::size_t g = group_scores[i].second;
    for (std::size_t s = 0; s < Eg; ++s) {
      keep[g * Eg + s] = true;
    }
  }

  std::vector<std::pair<float, std::size_t>> cand;
  cand.reserve(E);
  for (std::size_t e = 0; e < E; ++e) {
    if (keep[e]) {
      cand.emplace_back(biased_[e], e);
    }
  }
  const std::size_t take = std::min(K, cand.size());
  std::partial_sort(
      cand.begin(), cand.begin() + take, cand.end(),
      [](const auto& a, const auto& b) { return a.first > b.first; });

  // The fork's debug dump prints `ffn_moe_topk` (an i32 tensor) as floats, so
  // the selected expert ids are directly comparable. Trace them in selection
  // order; the printer emits the same first-3/last-3 layout ggml does.
  if (trace_) {
    std::vector<float> ids(take);
    for (std::size_t j = 0; j < take; ++j) {
      ids[j] = static_cast<float>(cand[j].second);
    }
    trace_(block, "ffn_moe_topk", ids.data(), take);
    // The biased score of each SELECTED expert, in rank order. The fork does
    // not print these (its probs dump only shows experts 0,1,2,509,510,511), so
    // this is how the tie margin behind any rank swap gets measured instead of
    // inferred: if two ranks carry near-identical biased scores, their order is
    // decided by arithmetic noise rather than by the model.
    std::vector<float> sel(take);
    for (std::size_t j = 0; j < take; ++j) {
      sel[j] = biased_[cand[j].second];
    }
    trace_(block, "ffn_moe_topk_biased", sel.data(), take);
  }

  // Weights: sum-normalised over the selected experts, then scaled by 2.5.
  // The fork clamps the sum to the smallest F16 normal to avoid dividing by 0.
  float wsum = 0.0F;
  for (std::size_t j = 0; j < take; ++j) {
    wsum += scores_[cand[j].second];
  }
  if (trace_) {
    trace_(block, "ffn_moe_weights_sum", &wsum, 1);
  }
  wsum = std::fmax(wsum, 6.103515625e-5F);
  if (trace_) {
    trace_(block, "ffn_moe_weights_sum_clamped", &wsum, 1);
  }
  for (std::size_t j = 0; j < take; ++j) {
    weights_[j] = scores_[cand[j].second] / wsum;
  }
  if (trace_) {
    trace_(block, "ffn_moe_weights_norm", weights_.data(), take);
  }
  for (std::size_t j = 0; j < take; ++j) {
    weights_[j] *= c_.expert_weights_scale;
  }
  if (trace_) {
    trace_(block, "ffn_moe_weights_scaled", weights_.data(), take);
  }

  std::fill(out.begin(), out.end(), 0.0F);
  for (std::size_t j = 0; j < take; ++j) {
    const std::uint64_t e = cand[j].second;
    cpu::MatVec(m.gate_exps, e, x, std::span<float>(gate_ff_.data(), ff));
    cpu::MatVec(m.up_exps, e, x, std::span<float>(up_ff_.data(), ff));
    // {ff, n_selected, n_tokens}: ne1 is the expert axis, so only the LAST
    // selected expert's slice is what the fork printed as its final row.
    if (trace_ && j + 1 == take) {
      trace_(block, "ffn_moe_gate", gate_ff_.data(), ff);
      trace_(block, "ffn_moe_up", up_ff_.data(), ff);
    }
    if (clamped) {
      cpu::SwigluClamped(gate_ff_.data(), up_ff_.data(), act_ff_.data(), ff,
                         limit);
    } else {
      for (std::size_t i = 0; i < ff; ++i) {
        act_ff_[i] = cpu::Silu(gate_ff_[i]) * up_ff_[i];
      }
    }
    // The router weight multiplies *after* the down projection.
    cpu::MatVec(m.down_exps, e, std::span<const float>(act_ff_.data(), ff),
                std::span<float>(down_.data(), H));
    if (trace_) {
      // The fork computes all 8 experts in one MUL_MAT_ID with ne1 = 8, so the
      // LAST printed row is the LAST selected expert. Trace only that one so
      // the comparison lines up.
      if (j + 1 == take) {
        trace_(block, "ffn_moe_down", down_.data(), H);
        // Pre-scaling in place is arithmetically identical to `out += w*d` --
        // one rounding on the product either way -- and lets the weighted
        // vector be traced. Only done when tracing, so the shipping path is
        // untouched.
        for (std::size_t i = 0; i < H; ++i) {
          down_[i] *= weights_[j];
        }
        trace_(block, "ffn_moe_weighted", down_.data(), H);
      }
    }
    for (std::size_t i = 0; i < H; ++i) {
      out[i] += weights_[j] * down_[i];
    }
  }
  if (trace_) {
    trace_(block, "ffn_moe_out", out.data(), H);
  }

  // Shared expert: parallel SwiGLU with its own limit, never routed, never
  // scaled by a router weight.
  cpu::MatVec(m.gate_shexp, 0, x, std::span<float>(gate_ff_.data(), sff));
  cpu::MatVec(m.up_shexp, 0, x, std::span<float>(up_ff_.data(), sff));
  const float slimit = block < c_.swiglu_clamp_shexp.size()
                           ? c_.swiglu_clamp_shexp[block]
                           : 0.0F;
  if (Config::ClampActive(slimit)) {
    cpu::SwigluClamped(gate_ff_.data(), up_ff_.data(), act_ff_.data(), sff,
                       slimit);
  } else {
    for (std::size_t i = 0; i < sff; ++i) {
      act_ff_[i] = cpu::Silu(gate_ff_[i]) * up_ff_[i];
    }
  }
  cpu::MatVec(m.down_shexp, 0, std::span<const float>(act_ff_.data(), sff),
              std::span<float>(down_.data(), H));
  for (std::size_t i = 0; i < H; ++i) {
    out[i] += down_[i];
  }
  if (trace_) {
    trace_(block, "ffn_shexp", down_.data(), H);
    // ffn_out is ADD(ffn_moe_out, ffn_shexp) in the fork -- the routed and
    // shared halves summed -- so it can only be traced after both are in `out`.
    trace_(block, "ffn_out", out.data(), H);
  }
}

// ---------------------------------------------------------------------------
// One token through the whole trunk.
// ---------------------------------------------------------------------------
bool ReferenceModel::Step(std::int32_t token, std::vector<float>& logits,
                          std::string* error_msg) {
  const std::size_t H = c_.hidden_size;
  if (position_ >= max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "context length exceeded";
    }
    return false;
  }
  if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "token id out of vocabulary";
    }
    return false;
  }

  cpu::DequantizeRow(w_.token_embd, 0, static_cast<std::uint64_t>(token),
                     normed_.data());
  if (trace_) {
    trace_(static_cast<std::uint32_t>(-1), "inp_embd", normed_.data(), H);
  }

  residual_.resize(H);
  block_out_.resize(H);
  ffn_in_.resize(H);
  tmp_.resize(H);
  attn_norm_.resize(H);
  ffn_norm_.resize(H);
  out_norm_.assign(H, 0.0F);
  std::copy_n(normed_.data(), H, residual_.data());

  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    const LayerWeights& l = w_.layers[il];

    cpu::ReadVector(l.attn_norm, attn_norm_.data());
    std::copy_n(residual_.data(), H, tmp_.data());
    cpu::RmsNorm(std::span<float>(tmp_.data(), H), attn_norm_.data(),
                 c_.rms_eps);

    if (l.mla) {
      MlaLayer(l.mla_weights, mla_[il], il, position_,
               std::span<const float>(tmp_.data(), H),
               std::span<float>(block_out_.data(), H));
    } else {
      KdaLayer(l.kda, kda_[il], il, std::span<const float>(tmp_.data(), H),
               std::span<float>(block_out_.data(), H));
    }

    if (trace_) {
      // tmp_ still holds the attention-normed hidden state here.
      trace_(il, "attn_norm", tmp_.data(), H);
      trace_(il, l.mla ? "attn_out" : "kda_out", block_out_.data(), H);
    }

    for (std::size_t i = 0; i < H; ++i) {
      ffn_in_[i] = residual_[i] + block_out_[i];
    }
    cpu::ReadVector(l.ffn_norm, ffn_norm_.data());
    std::copy_n(ffn_in_.data(), H, tmp_.data());
    cpu::RmsNorm(std::span<float>(tmp_.data(), H), ffn_norm_.data(),
                 c_.rms_eps);

    if (il < c_.dense_lead_layers) {
      DenseFfn(l, il, std::span<const float>(tmp_.data(), H),
               std::span<float>(block_out_.data(), H));
    } else {
      MoeFfn(l, il, std::span<const float>(tmp_.data(), H),
             std::span<float>(block_out_.data(), H));
    }
    for (std::size_t i = 0; i < H; ++i) {
      residual_[i] = ffn_in_[i] + block_out_[i];
    }
    if (trace_) {
      trace_(il, "ffn_inp", ffn_in_.data(), H);
      trace_(il, "ffn_norm", tmp_.data(), H);
      trace_(il, "l_out", residual_.data(), H);
    }
  }

  if (trace_) {
    trace_(static_cast<std::uint32_t>(-1), "result_norm_input",
           residual_.data(), H);
  }
  cpu::ReadVector(w_.output_norm, out_norm_.data());
  std::copy_n(residual_.data(), H, tmp_.data());
  cpu::RmsNorm(std::span<float>(tmp_.data(), H), out_norm_.data(), c_.rms_eps);

  logits.resize(c_.vocab_size);
  {
    // Over vocabulary rows only; each row accumulates identically.
    const std::int64_t vocab = static_cast<std::int64_t>(c_.vocab_size);
    float* out = logits.data();
    const float* x = tmp_.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t t = 0; t < vocab; ++t) {
      out[t] = cpu::DotRow(w_.output, 0, static_cast<std::uint64_t>(t),
                           std::span<const float>(x, H));
    }
  }

  ++position_;
  return true;
}

}  // namespace gufo::models::ling3
