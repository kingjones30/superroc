#ifndef SUPERROC_MODELS_LING3_REFERENCE_HPP_
#define SUPERROC_MODELS_LING3_REFERENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "src/models/ling3/weights.hpp"

namespace superroc::models::ling3 {

/// Single-token, float32, scalar reference of the Ling-3.0 (bailingmoe3)
/// graph. It exists to pin the semantics of every operator the GPU runtime
/// must reproduce: the KDA linear-attention recurrence with its causal conv and
/// safe forget gate, absorbed gated MLA over a latent KV cache, the two dense
/// leading blocks, and the sigmoid router with group-limited selection and
/// per-layer clamped SwiGLU. Speed is irrelevant here.
///
/// Every intermediate is float32 and the accumulation order mirrors the fork's
/// CPU kernels, so the logits are reproducible rather than merely close. The
/// MTP block (block 42) is bound but never executed.
class ReferenceModel {
public:
  /// Called once per traced tensor with (block, name, data, count). Used to
  /// diff layer by layer against the fork's llama-eval-callback dumps; empty
  /// by default so the production path pays nothing.
  using Trace = std::function<void(std::uint32_t, const char*, const float*,
                                   std::size_t)>;

  explicit ReferenceModel(const ModelWeights& weights,
                          std::uint32_t max_context);

  void SetTrace(Trace trace) { trace_ = std::move(trace); }

  /// Runs one token at the next position. `logits` is resized to the
  /// vocabulary and receives the output distribution.
  [[nodiscard]] bool Step(std::int32_t token, std::vector<float>& logits,
                          std::string* error_msg = nullptr);

  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  /// Bytes of persistent state (latent KV cache plus recurrent state and conv
  /// history) at the current context length.
  [[nodiscard]] std::uint64_t StateBytes() const noexcept;
  void Reset();

private:
  struct KdaState {
    /// Three conv histories, one per projection (Q, K, V), each
    /// [conv_kernel - 1][inner] oldest first. They differ after the first
    /// token: each holds that projection's own input, which is what the fork's
    /// `(kv_head * n_embd_r_total + qkv * conv_state_size)` offsets encode.
    std::vector<float> conv;
    /// [heads][head_dim][head_dim]: logical S[i][j] with i the key axis
    /// (fastest) and j the value axis, which is ggml's transposed storage.
    std::vector<float> state;
  };
  struct MlaState {
    /// [pos][kv_lora_rank + rotary_dim]: the compressed latent concatenated
    /// with the shared rotated key part. A single head.
    std::vector<float> latent;
  };

  void KdaLayer(const KdaWeights& k, KdaState& s, std::uint32_t block,
                std::span<const float> x, std::span<float> out);
  void MlaLayer(const MlaWeights& m, MlaState& s, std::uint32_t block,
                std::uint32_t pos, std::span<const float> x,
                std::span<float> out);
  void DenseFfn(const LayerWeights& l, std::uint32_t block,
                std::span<const float> x, std::span<float> out);
  void MoeFfn(const LayerWeights& l, std::uint32_t block,
              std::span<const float> x, std::span<float> out);

  const ModelWeights& w_;
  const Config& c_;
  std::uint32_t max_context_;
  std::uint32_t position_{0};
  Trace trace_;
  std::vector<KdaState> kda_;
  std::vector<MlaState> mla_;

  // Block-level scratch, sized once so Step allocates nothing.
  std::vector<float> normed_, residual_, block_out_, ffn_in_, tmp_;
  std::vector<float> attn_norm_, ffn_norm_, out_norm_;

  // KDA scratch.
  std::vector<float> q_, k_, v_, gate_, proj_, attn_, out_;
  std::vector<float> beta_, a_log_, dt_bias_, o_norm_;
  std::vector<float> conv_buf_, taps_, delta_, decay_;

  // MLA scratch.
  std::vector<float> q_all_, q_pe_, kv_a_, k_pe_, cmpr_, kv_a_norm_;
  // Softmax scores, one per cached position (not per head).
  std::vector<float> q_abs_, ml_scores_, head_out_, out_latent_, ml_gate_;

  // MoE scratch.
  std::vector<float> gate_ff_, up_ff_, act_ff_, down_;
  std::vector<float> scores_, biased_, weights_, probs_bias_;
};

}  // namespace superroc::models::ling3

#endif  // SUPERROC_MODELS_LING3_REFERENCE_HPP_
