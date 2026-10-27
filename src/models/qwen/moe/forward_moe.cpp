#include "src/models/qwen/moe/forward_moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/rocm_fp4/superroc_fp4_route.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/oracles.hpp"

namespace superroc::models {
namespace {

[[noreturn]] void KatMoeFail(const char* msg) {
  std::fprintf(stderr, "KAT %s\n", msg);
  std::fflush(stderr);
  std::abort();
}

}  // namespace

std::size_t KatExpertRowBytes(core::GgmlType type, int k) noexcept {
  if (k <= 0) {
    return 0;
  }
  const auto kk = static_cast<std::size_t>(k);
  switch (type) {
    case core::GgmlType::kF32:
      if (kk > std::numeric_limits<std::size_t>::max() / 4) {
        return 0;
      }
      return kk * 4;
    case core::GgmlType::kF16:
    case core::GgmlType::kBF16:
      if (kk > std::numeric_limits<std::size_t>::max() / 2) {
        return 0;
      }
      return kk * 2;
    default:
      return quant::QuantizedRowBytes(type, kk);
  }
}

QwenTensorRef KatExpertSlice(const QwenTensorRef& tensor, int expert,
                             int n_rows, int k) {
  if (expert < 0) {
    if (tensor.data == nullptr || tensor.num_elements == 0) {
      KatMoeFail("KatExpertSlice: shared expert tensor is empty");
    }
    return tensor;
  }
  const std::size_t row_bytes = KatExpertRowBytes(tensor.type, k);
  if (row_bytes == 0 || tensor.data == nullptr || n_rows <= 0) {
    KatMoeFail(
        "KatExpertSlice: null expert slice (stride 0 or empty tensor); "
        "refusing instead of reusing the previous expert");
  }
  const auto rows = static_cast<std::size_t>(n_rows);
  const auto kk = static_cast<std::size_t>(k);
  if (rows > std::numeric_limits<std::size_t>::max() / row_bytes) {
    KatMoeFail("KatExpertSlice: expert byte count overflows");
  }
  const std::size_t expert_bytes = row_bytes * rows;
  const auto expert_index = static_cast<std::size_t>(expert);
  if (expert_index > std::numeric_limits<std::size_t>::max() / expert_bytes) {
    KatMoeFail("KatExpertSlice: expert offset overflows");
  }
  const std::size_t offset = expert_index * expert_bytes;
  if (kk != 0 && rows > std::numeric_limits<std::size_t>::max() / kk) {
    KatMoeFail("KatExpertSlice: expert element count overflows");
  }
  const std::size_t elems = rows * kk;
  if (expert_index > std::numeric_limits<std::size_t>::max() / elems ||
      tensor.num_elements < (expert_index + 1) * elems) {
    KatMoeFail("KatExpertSlice: expert index is outside the tensor");
  }
  if (tensor.available_bytes != std::numeric_limits<std::size_t>::max() &&
      (offset > tensor.available_bytes ||
       expert_bytes > tensor.available_bytes - offset)) {
    KatMoeFail("KatExpertSlice: expert slice extends past mapped storage");
  }
  QwenTensorRef slice = tensor;
  slice.data = static_cast<const std::uint8_t*>(tensor.data) + offset;
  slice.num_elements = elems;
  if (tensor.available_bytes != std::numeric_limits<std::size_t>::max()) {
    slice.available_bytes = tensor.available_bytes - offset;
  }
  return slice;
}

bool KatRouteSoftmaxTopK(std::span<const float> logits, int n_used,
                         std::span<int> ids,
                         std::span<float> weights) noexcept {
  const int n_expert = static_cast<int>(logits.size());
  if (n_expert <= 0 || n_expert > 256 || n_used <= 0 || n_used > n_expert ||
      static_cast<int>(ids.size()) < n_used ||
      static_cast<int>(weights.size()) < n_used) {
    return false;
  }
  // Same reduction as ReferenceSoftmax, kept local so the router does not
  // require a second buffer from the caller beyond `weights`' sibling probs
  // stored in a fixed stack array.
  float max_val = logits[0];
  for (int i = 1; i < n_expert; ++i) {
    max_val = std::max(max_val, logits[i]);
  }
  double sum_exp = 0.0;
  float probs[256];
  for (int i = 0; i < n_expert; ++i) {
    sum_exp += std::exp(static_cast<double>(logits[i] - max_val));
  }
  const double inv_sum = 1.0 / sum_exp;
  for (int i = 0; i < n_expert; ++i) {
    probs[i] = static_cast<float>(
        std::exp(static_cast<double>(logits[i] - max_val)) * inv_sum);
  }

  int order[256];
  int work_a[512];
  int work_b[512];
  const int pad = superroc_fp4_pow2_ge(n_expert);
  if (pad < 0 || pad > 256) {
    return false;
  }
  // work arrays must cover `pad`, which equals n_expert when n_expert is a
  // power of two and is larger otherwise. 512 covers pad<=256.
  if (superroc_fp4_bitonic_argsort_desc(probs, order, work_a, work_b, n_expert) !=
      SUPERROC_FP4_OK) {
    return false;
  }
  // Fork norm_w (llama-graph): sum the picked probabilities in ascending
  // top-k slot order, then divide by max(sum, 2^-14). w_scale is 0 on the
  // KAT GGUF (no expert_weights_scale key), so it is not applied. Decode,
  // prefill, and the MTP block share this function, so they share the order.
  float raw_sum = 0.0F;
  for (int i = 0; i < n_used; ++i) {
    raw_sum += probs[order[i]];
  }
  const float den = raw_sum < 0x1p-14F ? 0x1p-14F : raw_sum;
  for (int i = 0; i < n_used; ++i) {
    ids[static_cast<std::size_t>(i)] = order[i];
    weights[static_cast<std::size_t>(i)] = probs[order[i]] / den;
  }
  return true;
}

void KatExpertSwiglu(const QwenTensorRef& gate, const QwenTensorRef& up,
                     const QwenTensorRef& down, int expert,
                     std::span<const float> x, std::size_t hidden,
                     std::size_t ff, std::span<float> gate_scratch,
                     std::span<float> up_scratch, std::span<float> act_scratch,
                     std::span<float> hidden_out) {
  if (x.size() < hidden || gate_scratch.size() < ff || up_scratch.size() < ff ||
      act_scratch.size() < ff || hidden_out.size() < hidden || ff == 0 ||
      hidden == 0) {
    KatMoeFail(
        "KatExpertSwiglu: activation or scratch is shorter than the expert");
  }
  const int rows_up = static_cast<int>(ff);
  const int rows_down = static_cast<int>(hidden);
  const QwenTensorRef gate_e =
      KatExpertSlice(gate, expert, rows_up, static_cast<int>(hidden));
  const QwenTensorRef up_e =
      KatExpertSlice(up, expert, rows_up, static_cast<int>(hidden));
  const QwenTensorRef down_e =
      KatExpertSlice(down, expert, rows_down, static_cast<int>(ff));
  TensorGEMV(gate_e, x, ff, hidden, gate_scratch);
  TensorGEMV(up_e, x, ff, hidden, up_scratch);
  for (std::size_t i = 0; i < ff; ++i) {
    const float g = gate_scratch[i];
    act_scratch[i] = (g / (1.0F + std::exp(-g))) * up_scratch[i];
  }
  TensorGEMV(down_e, act_scratch, hidden, ff, hidden_out);
}

void ForwardMoe(std::span<const float> x, const QwenLayerWeights& layer,
                const core::ModelConfig& config, std::span<float> logits,
                std::span<float> probs, std::span<float> gate_scratch,
                std::span<float> up_scratch, std::span<float> act_scratch,
                std::span<float> expert_hidden, std::span<float> out) {
  const std::size_t hidden = config.hidden_size;
  const int n_expert = static_cast<int>(config.expert_count);
  const int n_used = static_cast<int>(config.expert_used_count);
  const std::size_t ff = config.expert_feed_forward_length;
  const std::size_t ff_s = config.expert_shared_feed_forward_length;
  if (hidden == 0 || x.size() < hidden || out.size() < hidden ||
      n_expert <= 0 || logits.size() < static_cast<std::size_t>(n_expert) ||
      expert_hidden.size() < hidden || ff == 0) {
    KatMoeFail("ForwardMoe: hidden, router, or expert scratch is short");
  }
  std::fill_n(out.begin(), hidden, 0.0F);
  TensorGEMV(layer.ffn_gate_inp, x, static_cast<std::size_t>(n_expert), hidden,
             logits);
  int ids[8];
  float w[8];
  if (n_used <= 0 || n_used > 8 ||
      !KatRouteSoftmaxTopK(logits.first(static_cast<std::size_t>(n_expert)),
                           n_used, ids, w)) {
    KatMoeFail(
        "ForwardMoe: router top-k failed (expert_used_count must be 1..8)");
  }
  if (probs.size() >= static_cast<std::size_t>(n_expert)) {
    float max_val = logits[0];
    for (int i = 1; i < n_expert; ++i) {
      max_val = std::max(max_val, logits[i]);
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_expert; ++i) {
      sum_exp += std::exp(static_cast<double>(logits[i] - max_val));
    }
    const double inv_sum = 1.0 / sum_exp;
    for (int i = 0; i < n_expert; ++i) {
      probs[static_cast<std::size_t>(i)] = static_cast<float>(
          std::exp(static_cast<double>(logits[i] - max_val)) * inv_sum);
    }
  }
  if (gate_scratch.size() < ff || up_scratch.size() < ff ||
      act_scratch.size() < ff) {
    KatMoeFail(
        "ForwardMoe: routed-expert scratch is shorter than "
        "expert_feed_forward_length");
  }
  for (int i = 0; i < n_used; ++i) {
    KatExpertSwiglu(layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps,
                    ids[i], x, hidden, ff, gate_scratch, up_scratch,
                    act_scratch, expert_hidden);
    const float weight = w[i];
    for (std::size_t h = 0; h < hidden; ++h) {
      out[h] += weight * expert_hidden[h];
    }
  }
  if (!layer.ffn_up_shexp.empty()) {
    if (ff_s == 0 || gate_scratch.size() < ff_s || up_scratch.size() < ff_s ||
        act_scratch.size() < ff_s) {
      KatMoeFail("ForwardMoe: shared-expert scratch is short");
    }
    KatExpertSwiglu(layer.ffn_gate_shexp, layer.ffn_up_shexp,
                    layer.ffn_down_shexp, -1, x, hidden, ff_s, gate_scratch,
                    up_scratch, act_scratch, expert_hidden);
    const auto gate_v = layer.ffn_gate_inp_shexp.AsFloatSpan();
    if (gate_v.size() < hidden) {
      KatMoeFail("ForwardMoe: shared-expert gate is not an F32 hidden vector");
    }
    float dot = 0.0F;
    for (std::size_t h = 0; h < hidden; ++h) {
      dot += gate_v[h] * x[h];
    }
    const float sig = 1.0F / (1.0F + std::exp(-dot));
    for (std::size_t h = 0; h < hidden; ++h) {
      out[h] += sig * expert_hidden[h];
    }
  }
}

}  // namespace superroc::models
