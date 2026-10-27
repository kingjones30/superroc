// LFM2 short-conv + MoE-router GPU kernels (decode/prefill/router).
#ifndef SUPERROC_MODELS_LFM2_HIP_LFM2_KERNELS_HPP_
#define SUPERROC_MODELS_LFM2_HIP_LFM2_KERNELS_HPP_

#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace superroc::hip {

// LFM2: host launch result. `error` is a static string when ok is false.
struct Lfm2LaunchStatus {
  bool ok{false};
  const char* error{nullptr};
};

// LFM2: short-conv decode, one token. bcx = in_proj output [3*hidden]:
// B = bcx[0..H), C = bcx[H..2H), X = bcx[2H..3H).
// conv_w = F32 [hidden][l_cache] (tap k of channel c at c*l_cache + k).
// state = [(l_cache-1)][hidden], column 0 oldest.
// For each channel c: bx = B[c]*X[c];
// acc = conv_w[c*L + L-1]*bx + sum_{j<L-1} conv_w[c*L + j]*state[j*H + c];
// y[c] = C[c]*acc; then state columns shift left by one and
// state[(L-2)*H + c] = bx.
Lfm2LaunchStatus LaunchLfm2ShortConvDecode(const float* bcx,
                                           const float* conv_w, float* state,
                                           float* y, std::uint32_t hidden,
                                           std::uint32_t l_cache,
                                           hipStream_t stream);

// LFM2: short-conv prefill, n_tokens rows, token-major: bcx
// [n_tokens][3*hidden], y [n_tokens][hidden]. Same math applied token after
// token (the window for token t is: the incoming state columns, then the bx of
// tokens 0..t). On return the state holds the last l_cache-1 bx columns of the
// whole window (for n_tokens >= l_cache-1 that is the last l_cache-1 tokens'
// bx).
Lfm2LaunchStatus LaunchLfm2ShortConvPrefill(const float* bcx,
                                            const float* conv_w, float* state,
                                            float* y, std::uint32_t hidden,
                                            std::uint32_t l_cache,
                                            std::uint32_t n_tokens,
                                            hipStream_t stream);

// LFM2: MoE router, one token. gate_inp = F32 [n_expert][k]
// (row e at gate_inp + e*k), bias = F32 [n_expert].
// logits[e] = dot(gate_inp row e, x); p[e] = 1/(1+exp(-logits[e]));
// sel[e] = p[e] + bias[e];
// ids = the n_used experts with the largest sel, in descending sel order
// (equal sel: lower id first);
// weights[i] = p[ids[i]] / max(sum_i p[ids[i]], 6.103515625e-5).
// (bias selects only; it never enters the weights.)
Lfm2LaunchStatus LaunchLfm2MoeRoute(const float* x, const float* gate_inp,
                                    const float* bias, std::uint32_t k,
                                    std::uint32_t n_expert,
                                    std::uint32_t n_used, float* logits,
                                    int* ids, float* weights,
                                    hipStream_t stream);

// LFM2: tiled fp32 flash-attention prefill for head_dim 64 / GQA (SPEC2).
// q/out are [batch][heads][64]; the KV cache is fp32 head-major
// (k_cache[((layer*kv_heads+kvh)*max_ctx+pos)*64+d], v_cache = k_cache+plane).
// Query row r is at absolute position start_pos+r and attends [0,start_pos+r]
// with scale 1/sqrt(64). Head-dim 64 fixed; requires heads/kv_heads == 4.
Lfm2LaunchStatus LaunchLfm2PrefillAttention(
    const float* q, const float* k_cache, const float* v_cache, float* out,
    std::uint32_t layer, std::uint32_t start_pos, std::uint32_t batch,
    std::uint32_t max_ctx, std::uint32_t heads, std::uint32_t kv_heads,
    hipStream_t stream);

// LFM2 (SPEC3): the same prefill attention over the fp16 token-major cache
// (element ((layer*max_ctx+pos)*kv_heads+kvh)*64+d, V plane = K + plane).
Lfm2LaunchStatus LaunchLfm2PrefillAttentionF16(
    const float* q, const std::uint16_t* k_cache, const std::uint16_t* v_cache,
    float* out, std::uint32_t layer, std::uint32_t start_pos,
    std::uint32_t batch, std::uint32_t max_ctx, std::uint32_t heads,
    std::uint32_t kv_heads, hipStream_t stream);

// LFM2 (SPEC3): batched fp16 KV write. k/v are [batch][kv] fp32 (kv =
// kv_heads*head_dim); row r lands at token-major row layer*max_ctx +
// start_pos + r. Round to nearest even.
Lfm2LaunchStatus LaunchLfm2BatchedWriteKvF16(
    const float* k, const float* v, std::uint16_t* k_cache,
    std::uint16_t* v_cache, std::uint32_t layer, std::uint32_t start_pos,
    std::uint32_t batch, std::uint32_t max_ctx, std::uint32_t kv,
    hipStream_t stream);

// LFM2: MoE router, n_tokens rows: x [n_tokens][k],
// logits [n_tokens][n_expert], ids/weights [n_tokens][n_used]. Same math per
// row.
Lfm2LaunchStatus LaunchLfm2MoeRouteBatch(
    const float* x, const float* gate_inp, const float* bias, std::uint32_t k,
    std::uint32_t n_expert, std::uint32_t n_tokens, std::uint32_t n_used,
    float* logits, int* ids, float* weights, hipStream_t stream);

// LFM2 (LKF): hidden += add, then out = RMSNorm(hidden, eps) * weight, in one
// 256-thread block. Same arithmetic as LaunchResidualAdd followed by
// LaunchRMSNorm (generic RMSNormKernel<>), so the result is bit-identical;
// one launch instead of two. weight may be null (scale 1).
Lfm2LaunchStatus LaunchLfm2AddRMSNorm(float* hidden, const float* add,
                                      const float* weight, float* out,
                                      std::uint32_t dim, float eps,
                                      hipStream_t stream);

}  // namespace superroc::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // SUPERROC_MODELS_LFM2_HIP_LFM2_KERNELS_HPP_
