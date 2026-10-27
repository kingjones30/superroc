#include "src/models/glimmer/glimmer_oracle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "src/core/quant/ggml_gemm.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/oracles.hpp"

namespace superroc::models {
namespace {

constexpr std::int8_t kFp4Code[16] = {0, 1,  2,  3,  4,  6,  8,  10,
                                      0, -1, -2, -3, -4, -6, -8, -10};

float Ue4m3Half(std::uint8_t e) noexcept {
  if (e > 0x7e) {
    return 0.0F;
  }
  const int exp = (e >> 3) & 0xF;
  const int man = e & 0x7;
  if (exp == 0) {
    return static_cast<float>(man) * (1.0F / 1024.0F);
  }
  const std::uint32_t bits = (static_cast<std::uint32_t>(exp) + 119U) << 23 |
                             (static_cast<std::uint32_t>(man) << 20);
  float out = 0.0F;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

const void* RowPtr(const QwenTensorRef& tensor, std::size_t row,
                   std::size_t columns) noexcept {
  const std::size_t row_bytes = quant::QuantizedRowBytes(tensor.type, columns);
  if (row_bytes == 0 || tensor.data == nullptr) {
    return nullptr;
  }
  return static_cast<const std::uint8_t*>(tensor.data) + row * row_bytes;
}

void AddResidual(std::span<float> hidden,
                 std::span<const float> delta) noexcept {
  const std::size_t n = std::min(hidden.size(), delta.size());
  for (std::size_t i = 0; i < n; ++i) {
    hidden[i] += delta[i];
  }
}

void PerHeadNorm(std::span<float> heads, std::uint32_t count,
                 std::uint32_t head_dim, std::span<const float> weight,
                 float eps) noexcept {
  if (head_dim == 0 || head_dim > 256 || heads.size() < count * head_dim) {
    return;
  }
  std::array<float, 256> tmp{};
  for (std::uint32_t h = 0; h < count; ++h) {
    auto head = heads.subspan(static_cast<std::size_t>(h) * head_dim, head_dim);
    GlimmerRmsNorm(head, weight, eps, std::span<float>(tmp.data(), head_dim));
    std::copy_n(tmp.data(), head_dim, head.begin());
  }
}

void ApplyNormRope(std::span<float> heads, std::uint32_t count,
                   std::uint32_t head_dim, std::uint32_t rotary_dim,
                   std::uint32_t pos, float theta, bool use_rope) noexcept {
  if (!use_rope || head_dim == 0) {
    return;
  }
  for (std::uint32_t h = 0; h < count; ++h) {
    GlimmerNormRope(
        heads.subspan(static_cast<std::size_t>(h) * head_dim, head_dim),
        rotary_dim, pos, theta);
  }
}

}  // namespace

void GlimmerRmsNorm(std::span<const float> x, std::span<const float> weight,
                    float eps, std::span<float> out) noexcept {
  if (x.empty() || out.size() < x.size()) {
    return;
  }
  double sum = 0.0;
  for (const float value : x) {
    sum += static_cast<double>(value) * static_cast<double>(value);
  }
  const double inv = 1.0 / std::sqrt(sum / static_cast<double>(x.size()) +
                                     static_cast<double>(eps));
  const bool weighted = weight.size() == x.size();
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double scale = weighted ? static_cast<double>(weight[i]) : 1.0;
    out[i] = static_cast<float>(static_cast<double>(x[i]) * inv * scale);
  }
}

void GlimmerNormRope(std::span<float> head, std::uint32_t rotary_dim,
                     std::uint32_t pos, float theta) noexcept {
  if (theta == 0.0F || rotary_dim < 2 || head.size() < rotary_dim ||
      (rotary_dim % 2U) != 0U) {
    return;
  }
  const std::size_t pairs = rotary_dim / 2U;
  for (std::size_t i = 0; i < pairs; ++i) {
    const double freq = std::pow(
        static_cast<double>(theta),
        -2.0 * static_cast<double>(i) / static_cast<double>(rotary_dim));
    const double ang = static_cast<double>(pos) * freq;
    const double c = std::cos(ang);
    const double s = std::sin(ang);
    const std::size_t i0 = 2 * i;
    const double x0 = head[i0];
    const double x1 = head[i0 + 1];
    head[i0] = static_cast<float>(x0 * c - x1 * s);
    head[i0 + 1] = static_cast<float>(x0 * s + x1 * c);
  }
}

void GlimmerScaleSoftcap(std::span<float> logits, float scale,
                         float cap) noexcept {
  for (float& value : logits) {
    value *= scale;
    if (cap != 0.0F) {
      value = cap * std::tanh(value / cap);
    }
  }
}

float GlimmerQ8Dot(core::GgmlType type, const void* row,
                   std::span<const float> x, std::size_t k) noexcept {
  if (row == nullptr || x.size() < k || k == 0 || (k % 32U) != 0U) {
    return 0.0F;
  }
  const bool fast = type == core::GgmlType::kQ4_0_ROCMFP4_FAST;
  const bool dual = type == core::GgmlType::kQ4_0_ROCMFP4;
  if (!fast && !dual) {
    return 0.0F;
  }
  const std::size_t stride = fast ? 17U : 18U;
  const auto* bytes = static_cast<const std::uint8_t*>(row);
  double sum = 0.0;
  const std::size_t blocks = k / 32U;
  for (std::size_t b = 0; b < blocks; ++b) {
    const std::uint8_t* blk = bytes + b * stride;
    const float* xv = x.data() + b * 32U;
    float amax = 0.0F;
    for (int j = 0; j < 32; ++j) {
      amax = std::max(amax, std::fabs(xv[j]));
    }
    const float d = amax / 127.0F;
    const float inv = d == 0.0F ? 0.0F : 1.0F / d;
    int acc_lo = 0;
    int acc_hi = 0;
    for (int j = 0; j < 32; ++j) {
      const std::uint8_t byte = blk[j & 15];
      const unsigned nib = j < 16 ? (byte & 0x0FU) : (byte >> 4);
      int qs = static_cast<int>(std::round(static_cast<double>(xv[j]) * inv));
      if (qs > 127) {
        qs = 127;
      } else if (qs < -127) {
        qs = -127;
      }
      const int prod = static_cast<int>(kFp4Code[nib]) * qs;
      if (j < 16) {
        acc_lo += prod;
      } else {
        acc_hi += prod;
      }
    }
    const float scale_lo = Ue4m3Half(blk[16]);
    const float scale_hi = fast ? scale_lo : Ue4m3Half(blk[17]);
    sum += static_cast<double>(acc_lo) * scale_lo * d;
    sum += static_cast<double>(acc_hi) * scale_hi * d;
  }
  return static_cast<float>(sum);
}

double GlimmerCosine(std::span<const float> a,
                     std::span<const float> b) noexcept {
  const std::size_t n = std::min(a.size(), b.size());
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double av = a[i];
    const double bv = b[i];
    dot += av * bv;
    na += av * av;
    nb += bv * bv;
  }
  if (na < 1e-20 && nb < 1e-20) {
    return 1.0;
  }
  if (na < 1e-20 || nb < 1e-20) {
    return 0.0;
  }
  return dot / std::sqrt(na * nb);
}

void GlimmerForwardLayer(std::span<float> hidden, const QwenLayerWeights& layer,
                         const core::ModelConfig& config, QwenKvCache& kv_cache,
                         std::uint32_t layer_idx, std::uint32_t pos,
                         QwenScratchArena& arena) noexcept {
  const std::size_t hidden_size = config.hidden_size;
  const std::uint32_t n_heads = config.num_attention_heads;
  const std::uint32_t n_kv = config.num_key_value_heads;
  const std::uint32_t head_dim = config.head_dim;
  const std::size_t q_size = static_cast<std::size_t>(n_heads) * head_dim;
  const std::size_t kv_size = static_cast<std::size_t>(n_kv) * head_dim;
  if (hidden.size() < hidden_size || arena.normed.size() < hidden_size ||
      arena.q.size() < q_size || arena.k.size() < kv_size ||
      arena.v.size() < kv_size || arena.ssm_gate.size() < q_size ||
      arena.attn_out.size() < hidden_size || n_kv == 0) {
    return;
  }

  const float eps_pre = config.norm_eps_pre;
  const float eps_post = config.norm_eps_post;
  GlimmerRmsNorm(hidden.first(hidden_size), layer.attn_norm.AsFloatSpan(),
                 eps_pre, arena.normed.first(hidden_size));

  TensorGEMV(layer.attn_q, arena.normed, q_size, hidden_size, arena.q);
  TensorGEMV(layer.attn_k, arena.normed, kv_size, hidden_size, arena.k);
  TensorGEMV(layer.attn_v, arena.normed, kv_size, hidden_size, arena.v);
  TensorGEMV(layer.attn_gate, arena.normed, q_size, hidden_size,
             arena.ssm_gate);

  PerHeadNorm(arena.q.first(q_size), n_heads, head_dim,
              layer.attn_q_norm.AsFloatSpan(), eps_pre);
  PerHeadNorm(arena.k.first(kv_size), n_kv, head_dim,
              layer.attn_k_norm.AsFloatSpan(), eps_pre);
  const bool use_rope = config.LayerUsesRope(layer_idx);
  ApplyNormRope(arena.q.first(q_size), n_heads, head_dim, config.rotary_dim,
                pos, config.rope_theta, use_rope);
  ApplyNormRope(arena.k.first(kv_size), n_kv, head_dim, config.rotary_dim, pos,
                config.rope_theta, use_rope);

  for (std::uint32_t kv_h = 0; kv_h < n_kv; ++kv_h) {
    const auto k_src =
        arena.k.subspan(static_cast<std::size_t>(kv_h) * head_dim, head_dim);
    const auto v_src =
        arena.v.subspan(static_cast<std::size_t>(kv_h) * head_dim, head_dim);
    std::ranges::copy(k_src,
                      kv_cache.GetKeySlice(layer_idx, kv_h, pos).begin());
    std::ranges::copy(v_src,
                      kv_cache.GetValueSlice(layer_idx, kv_h, pos).begin());
  }

  const std::uint32_t group = n_heads / n_kv;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  const std::uint32_t begin =
      GlimmerSwaBegin(pos, config.LayerSwaWindow(layer_idx));
  const std::uint32_t seq = pos + 1U;
  const std::uint32_t nscore = seq - begin;
  if (arena.ssm_out_buf.size() < q_size || arena.attn_scores.size() < nscore) {
    return;
  }
  auto context = arena.ssm_out_buf.first(q_size);
  auto scores = arena.attn_scores.first(nscore);
  std::fill(context.begin(), context.end(), 0.0F);
  for (std::uint32_t h = 0; h < n_heads; ++h) {
    const std::uint32_t kv_h = h / group;
    const auto q_head =
        arena.q.subspan(static_cast<std::size_t>(h) * head_dim, head_dim);
    float max_score = -1.0e30F;
    for (std::uint32_t p = begin; p < seq; ++p) {
      const auto k_p = kv_cache.GetKeySlice(layer_idx, kv_h, p);
      double dot = 0.0;
      for (std::uint32_t d = 0; d < head_dim; ++d) {
        dot += static_cast<double>(q_head[d]) * static_cast<double>(k_p[d]);
      }
      const float score = static_cast<float>(dot * scale);
      scores[p - begin] = score;
      max_score = std::max(score, max_score);
    }
    double sum_exp = 0.0;
    for (std::uint32_t p = begin; p < seq; ++p) {
      sum_exp += std::exp(static_cast<double>(scores[p - begin] - max_score));
    }
    const double inv = sum_exp == 0.0 ? 0.0 : 1.0 / sum_exp;
    auto ctx =
        context.subspan(static_cast<std::size_t>(h) * head_dim, head_dim);
    std::fill(ctx.begin(), ctx.end(), 0.0F);
    for (std::uint32_t p = begin; p < seq; ++p) {
      const double weight =
          std::exp(static_cast<double>(scores[p - begin] - max_score)) * inv;
      const auto v_p = kv_cache.GetValueSlice(layer_idx, kv_h, p);
      for (std::uint32_t d = 0; d < head_dim; ++d) {
        ctx[d] = static_cast<float>(static_cast<double>(ctx[d]) +
                                    weight * static_cast<double>(v_p[d]));
      }
    }
  }
  for (std::size_t i = 0; i < q_size; ++i) {
    const float g = arena.ssm_gate[i];
    context[i] *= 1.0F / (1.0F + std::exp(-g));
  }
  TensorGEMV(layer.attn_output, context, hidden_size, q_size, arena.attn_out);

  GlimmerRmsNorm(arena.attn_out.first(hidden_size),
                 layer.attn_post_norm.AsFloatSpan(), eps_post,
                 arena.normed.first(hidden_size));
  AddResidual(hidden.first(hidden_size), arena.normed.first(hidden_size));

  GlimmerRmsNorm(hidden.first(hidden_size), layer.ffn_norm.AsFloatSpan(),
                 eps_pre, arena.normed.first(hidden_size));
  ForwardFFN(arena.normed.first(hidden_size), layer.ffn_gate, layer.ffn_up,
             layer.ffn_down, hidden_size, config.intermediate_size,
             arena.mlp_gate, arena.mlp_up, arena.mlp_act, arena.mlp_out);
  GlimmerRmsNorm(arena.mlp_out.first(hidden_size),
                 layer.ffn_post_norm.AsFloatSpan(), eps_post,
                 arena.normed.first(hidden_size));
  AddResidual(hidden.first(hidden_size), arena.normed.first(hidden_size));
}

void GlimmerForwardModel(std::uint32_t token_id, std::uint32_t pos,
                         const QwenModelWeights& weights, QwenKvCache& kv_cache,
                         QwenScratchArena& arena,
                         std::span<float> logits_out) noexcept {
  const std::size_t hidden_size = weights.config.hidden_size;
  ForwardEmbedding(token_id, weights.token_embd, hidden_size, arena.hidden);
  GlimmerRmsNorm(arena.hidden.first(hidden_size), {},
                 weights.config.norm_eps_pre, arena.normed.first(hidden_size));
  std::copy_n(arena.normed.begin(), hidden_size, arena.hidden.begin());

  for (std::uint32_t layer = 0; layer < weights.config.num_layers; ++layer) {
    GlimmerForwardLayer(arena.hidden, weights.layers[layer], weights.config,
                        kv_cache, layer, pos, arena);
  }

  GlimmerRmsNorm(arena.hidden.first(hidden_size),
                 weights.output_norm.AsFloatSpan(), weights.config.norm_eps_pre,
                 arena.normed.first(hidden_size));
  if (logits_out.size() < weights.config.vocab_size) {
    return;
  }
  TensorGEMV(weights.output, arena.normed, weights.config.vocab_size,
             hidden_size, logits_out);
  GlimmerScaleSoftcap(logits_out.first(weights.config.vocab_size),
                      weights.config.logit_scale, weights.config.final_softcap);
}

}  // namespace superroc::models
