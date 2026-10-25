#ifndef GUFO_MODELS_LING3_CPU_OPS_HPP_
#define GUFO_MODELS_LING3_CPU_OPS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/ling3/weights.hpp"

/// Scalar CPU operators for the Ling-3.0 reference model. One row at a time,
/// every format decoded on the fly. Correctness oracle only; nothing here is
/// tuned, and no transcendental is reassociated.
namespace gufo::models::ling3::cpu {

/// Decodes one row of `t` (expert `e`) into `out` (t.cols floats).
void DequantizeRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
                   float* out);

/// y[r] = sum_c W[e][r][c] * x[c] for every row of the matrix.
void MatVec(const TensorRef& t, std::uint64_t e, std::span<const float> x,
            std::span<float> y);

/// GGML_FP32_TO_FP16, round-to-nearest-even. ggml keeps the Q8_0 block scale in
/// a half and converts it back before every use, so this round trip is part of
/// the ROCmFP4 matmul arithmetic and has to be reproduced exactly.
///
/// Exposed (rather than kept file-local) so tools/ling3_dot_parity quantizes
/// the activation with THIS function. An earlier version of that tool
/// re-transcribed the conversion; the transcription carried the same missing
/// exponent rebase, so the test compared the fork against itself and passed on
/// values both sides got wrong.
[[nodiscard]] std::uint16_t FloatToFp16(float f);

/// Dot product of one weight row with x, decoding the row and multiplying.
///
/// Set LING3_DOT to choose the ROCmFP4 arithmetic: `q8` (default) is ggml's
/// -- quantize the activation to Q8_0 and accumulate an integer dot
/// (`rocmfp4_vec_dot_q4_0_{,fast_}q8_0`, since `vec_dot_type = GGML_TYPE_Q8_0`
/// for these types); `exact` is the exact decoded product; `both` runs the two
/// and reports the first call sites where they disagree by more than 1 %.
float DotRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
             std::span<const float> x);

/// Reads a whole 1-D parameter (v.tensor.cols floats) into `out`.
void ReadVector(const VectorRef& v, float* out);

/// RMSNorm over `x` in place with gamma `w` (nullptr keeps the raw scale).
/// Matches ggml_rms_norm: x * rsqrt(mean(x^2) + eps) * w.
void RmsNorm(std::span<float> x, const float* w, float eps);

/// x / max(|x|, eps), the DeltaNet query/key normalization. ggml_l2_norm puts
/// the fmaxf *outside* the square root, so this is a clamp, not a soft
/// epsilon; reproducing it as x / sqrt(|x|^2 + eps) would not match.
void L2Norm(std::span<float> x, float eps);

[[nodiscard]] float Sigmoid(float x) noexcept;
/// Exact silu, x / (1 + exp(-x)); not the tanh approximation.
[[nodiscard]] float Silu(float x) noexcept;

/// GGML_ROPE_TYPE_NORM partial rotary embedding: the first `rotary_dim`
/// elements of each head are read as pairs (2i, 2i+1), and pair i rotates by
/// pos * theta_i with theta_i = base^(-2i/rotary_dim). This is what ggml's
/// rope_norm does (ggml-cuda/rope.cu); the half-split pairing (i,
/// i+rotary_dim/2) is GGML_ROPE_TYPE_NEOX and is wrong for this arch.
void Rope(float* x, std::uint32_t heads, std::uint32_t head_dim,
          std::uint32_t rotary_dim, std::uint32_t pos, float theta);

/// silu(gate) clamped above at `limit`, times up clamped to [-limit, limit].
/// The fork skips the clamp entirely when limit <= 1e-6; this function is only
/// called in that case.
void SwigluClamped(const float* gate, const float* up, float* out,
                   std::size_t n, float limit);

}  // namespace gufo::models::ling3::cpu

#endif  // GUFO_MODELS_LING3_CPU_OPS_HPP_
