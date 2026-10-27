// ling3_ops.hpp — the small ops a Ling-3.0 decode step needs around the
// attention blocks. Lane: LK (MiMo) — Ling-on-superroc overnight build. Do not
// confuse with LG's drafts. Lane LK, phase 2 of the Ling-on-superroc build.
//
// Five things (phase-2 spec):
//   1. RMSNorm          eps from the GGUF
//   (bailingmoe3.attention.layer_norm_rms_epsilon = 1e-6)
//   2. residual add
//   3. embedding row    token_embd.weight is Q6_K [2560, 157184] -> F32
//   4. final norm       output_norm.weight F32 [2560] (RMSNorm, same op as 1)
//   5. argmax over the 157184-entry vocab
//
// RMSNorm matches ggml_rms_norm / LG's cpu::RmsNorm:
//     mean = sum(x*x)/n ; scale = 1/sqrt(mean + eps) ; y = x * scale * w
// The sum is accumulated in double on the CPU oracle and in float on the GPU,
// which is what ggml-cuda's rms_norm_f32 does.
//
// The Q6_K decoder is LX's, vendored by include: `../ling-moe/ling3_quant.hpp`
// (self-contained, host/device). It follows ggml-quants.c dequantize_row_q6_K.

#ifndef LING3_OPS_HPP
#define LING3_OPS_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ling3 {

// GGUF bailingmoe3.attention.layer_norm_rms_epsilon
constexpr float LING3_RMS_EPS = 1e-6f;

// ---------------------------------------------------------------------------
// 1. RMSNorm over `rows` vectors of `n` elements.
//    x : [n, rows] contiguous (or with explicit strides), y may alias x
//    w : [n] gamma (nullptr => plain normalisation)
// ---------------------------------------------------------------------------
void ops_rms_norm(const float* x, const float* w, float* y, int n, int rows,
                  float eps, int64_t x_tok, int64_t y_tok, hipStream_t stream);

// contiguous convenience: x, y are [n, rows]
inline void ops_rms_norm(const float* x, const float* w, float* y, int n,
                         int rows, float eps, hipStream_t stream) {
  ops_rms_norm(x, w, y, n, rows, eps, n, n, stream);
}

// 4. final norm = RMSNorm with output_norm.weight over `rows` hidden states of
// n_embd.
void ops_final_norm(const float* hidden, const float* output_norm_w, float* y,
                    int n_embd, int rows, hipStream_t stream);

// ---------------------------------------------------------------------------
// 2. residual add: y = a + b  (y may alias a or b)
//    vectors of `n` elements, `rows` of them
// ---------------------------------------------------------------------------
void ops_residual_add(const float* a, const float* b, float* y, int n, int rows,
                      int64_t a_tok, int64_t b_tok, int64_t y_tok,
                      hipStream_t stream);

inline void ops_residual_add(const float* a, const float* b, float* y, int n,
                             int rows, hipStream_t stream) {
  ops_residual_add(a, b, y, n, rows, n, n, n, stream);
}

// ---------------------------------------------------------------------------
// 3. embedding row: look up `token_id` in a K-quant token_embd table and write
//    F32. `token_embd_type` is the tensor's ggml type: Q6_K in the text file,
//    Q5_K in the VL one (both 256-element blocks, 210 / 176 bytes).
//    table : the raw weight blob of token_embd.weight [n_embd, n_vocab]
//    row stride is ling3_row_bytes(token_embd_type, n_embd).
//    n_embd must be a multiple of that type's block size (256).
//    ⚠ the decoder does one widened load at the end of a row, so the table
//    buffer
//      must have LING3_TAIL_PAD (64) readable bytes past the last row.
// ---------------------------------------------------------------------------
void ops_embedding_row(int token_embd_type, const void* token_embd, int n_embd,
                       int n_vocab, const int* token_ids, int n_tokens,
                       float* out, hipStream_t stream);

// ---------------------------------------------------------------------------
// 5. argmax over `n` floats. `out_index` is device int32[rows].
//    Ties resolve to the lowest index (the CPU oracle does the same).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 2b. LS2 PHASE7: residual add fused with the RMSNorm that consumes it.
//     y = ((a + b) * rsqrtf(mean((a+b)^2) + eps)) * w ; rsum (optional) also
//     gets a + b. Same arithmetic as ops_residual_add followed by ops_rms_norm.
// ---------------------------------------------------------------------------
void ops_add_rms_norm(const float* a, const float* b, const float* w, float* y,
                      float* rsum, int n, int rows, float eps, int64_t a_tok,
                      int64_t b_tok, int64_t y_tok, hipStream_t stream);

inline void ops_add_rms_norm(const float* a, const float* b, const float* w,
                             float* y, float* rsum, int n, int rows, float eps,
                             hipStream_t stream) {
  ops_add_rms_norm(a, b, w, y, rsum, n, rows, eps, n, n, n, stream);
}

void ops_argmax(const float* logits, int n, int rows, int32_t* out_index,
                int64_t logits_tok, hipStream_t stream);

inline void ops_argmax(const float* logits, int n, int rows, int32_t* out_index,
                       hipStream_t stream) {
  ops_argmax(logits, n, rows, out_index, n, stream);
}

// ---------------------------------------------------------------------------
// LM1 SPEC6 item 3 (RV14 3.2): argmax AND its top-1 softmax probability in one
// pass over the logits. p_top1 = 1 / sum_i exp(x_i - x_max), exact to fp32.
//
// Two loops, not one: the max must be known before exp(x - max) is safe. The
// logits buffer is 157,184 floats (~628 KB), so the second pass is L2-resident.
// out_index and out_p may point into ONE device allocation (e.g. an int[2]
// with out_p = (float*)(out_index + 1)) so the caller reads both back in a
// single 8-byte memcpy -- the D2H count is what the draft loop pays per token.
// ---------------------------------------------------------------------------
void ops_argmax_prob(const float* logits, int n, int rows, int32_t* out_index,
                     float* out_p, int64_t logits_tok, hipStream_t stream);

const char* ops_version();  // "ling3-ops/1.0"

}  // namespace ling3

#endif  // LING3_OPS_HPP
