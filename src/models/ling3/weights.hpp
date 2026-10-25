#ifndef GUFO_MODELS_LING3_WEIGHTS_HPP_
#define GUFO_MODELS_LING3_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling3/config.hpp"

namespace gufo::models::ling3 {

/// Non-owning view of one GGUF tensor. `cols` x `rows` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `experts` (ne[2]) stacks whole matrices.
/// Ling never uses ne[3].
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  std::uint32_t shard{0};
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] std::uint64_t ElementCount() const noexcept {
    return cols * rows * experts;
  }
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * static_cast<std::size_t>(rows) *
           static_cast<std::size_t>(experts);
  }
  /// Base address of one stacked expert matrix.
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) +
           RowBytes() * static_cast<std::size_t>(rows) *
               static_cast<std::size_t>(e);
  }
};

/// A dense 1-D parameter (norm gains, biases, the KDA `ssm_a` vector).
struct VectorRef {
  TensorRef tensor;
  [[nodiscard]] bool empty() const noexcept { return tensor.empty(); }
};

/// KDA (linear attention) parameters of one block.
struct KdaWeights {
  TensorRef wq;      ///< [hidden -> inner]
  TensorRef wk;      ///< [hidden -> inner]
  TensorRef wv;      ///< [hidden -> inner]
  TensorRef conv_q;  ///< [conv_kernel, inner]
  TensorRef conv_k;
  TensorRef conv_v;
  TensorRef w_f;      ///< [hidden -> inner], the forget-gate projection
  TensorRef w_g;      ///< [hidden -> inner], the output-gate projection
  TensorRef w_beta;   ///< [hidden -> heads]
  TensorRef a_log;    ///< [heads]; the converter already stored exp(A_log)
  VectorRef dt_bias;  ///< [inner]
  VectorRef o_norm;   ///< [head_dim]
  TensorRef wo;       ///< [inner -> hidden]
};

/// Gated MLA parameters of one block. Ling ships the absorbed form
/// (`attn_k_b` / `attn_v_b`), so there is no separate `wkv_b`.
struct MlaWeights {
  TensorRef wq;         ///< [hidden -> heads * key_dim]
  TensorRef wkv_a_mqa;  ///< [hidden -> kv_lora_rank + rotary_dim]
  VectorRef kv_a_norm;  ///< [kv_lora_rank]
  TensorRef wk_b;       ///< [nope_dim, kv_lora_rank, heads]
  TensorRef wv_b;       ///< [kv_lora_rank, value_dim, heads]
  TensorRef wqkv_gate;  ///< [hidden -> heads]
  TensorRef wo;         ///< [heads * value_dim -> hidden]
};

/// Mixture-of-experts parameters of one block. Gate and up are separate
/// tensors; there is no merged gate_up.
struct MoeWeights {
  TensorRef gate_inp;    ///< [hidden -> num_experts], F32
  VectorRef probs_bias;  ///< [num_experts], selection only
  TensorRef gate_exps;   ///< [hidden -> expert_ff] x experts
  TensorRef up_exps;     ///< [hidden -> expert_ff] x experts
  TensorRef down_exps;   ///< [expert_ff -> hidden] x experts
  TensorRef gate_shexp;  ///< [hidden -> shared_ff]
  TensorRef up_shexp;
  TensorRef down_shexp;  ///< [shared_ff -> hidden]
};

struct LayerWeights {
  bool mla{false};
  VectorRef attn_norm;
  KdaWeights kda;
  MlaWeights mla_weights;
  VectorRef ffn_norm;
  // Leading dense blocks only.
  TensorRef ffn_gate;
  TensorRef ffn_up;
  TensorRef ffn_down;
  MoeWeights moe;
};

/// The `nextn.*` glue of the MTP head, which sits on top of the block bound in
/// `ModelWeights::mtp`.
///
/// Ling's artifact ships exactly three of these: `nextn.eh_proj` [2*hidden,
/// hidden], `nextn.enorm` [hidden], `nextn.hnorm` [hidden]. It ships **no**
/// `nextn.embed_tokens`, **no** `nextn.shared_head_head` and **no**
/// `nextn.shared_head_norm`, so the head borrows the main `token_embd` and
/// `output`.
///
/// `layer_out_norm` is `blk.42.layer_output_norm`, which the artifact *does*
/// ship. On `bailing-hybrid` the fork never binds or reads it
/// (`bailing-hybrid.cpp:1011-1014` falls straight through to
/// `model.output_norm`), unlike `bailingmoe3.cpp:718-720` which prefers it.
/// Bound so both can be measured; MTP_NOTES.md section 2. MEASURED: identical
/// acceptance either way.
struct MtpHeadWeights {
  TensorRef eh_proj;         ///< [2 * hidden -> hidden], Q8_0
  VectorRef enorm;           ///< [hidden]
  VectorRef hnorm;           ///< [hidden]
  VectorRef layer_out_norm;  ///< [hidden], the alternative final norm
};

/// The trunk artifact: 42 decoder blocks. Block 42 (the MTP block) is bound
/// into `mtp` and executed only by the MTP draft path.
struct ModelWeights {
  Config config;
  TensorRef token_embd;  ///< [hidden -> vocab]
  VectorRef output_norm;
  TensorRef output;  ///< [hidden -> vocab]
  std::vector<LayerWeights> layers;
  LayerWeights mtp;         ///< The nextn block (block 42), gated MLA + MoE.
  MtpHeadWeights mtp_head;  ///< Its nextn.* projections and norms.

  /// Binds and validates the trunk. Every tensor is checked for presence,
  /// exact shape and a supported storage type; the payloads stay mapped.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::ling3

#endif  // GUFO_MODELS_LING3_WEIGHTS_HPP_
