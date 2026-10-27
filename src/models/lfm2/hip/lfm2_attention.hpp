// LFM2 split-K "flash-decoding" attention (decode) + device-side KV write.
#ifndef SUPERROC_MODELS_LFM2_HIP_LFM2_ATTENTION_HPP_
#define SUPERROC_MODELS_LFM2_HIP_LFM2_ATTENTION_HPP_

#include <cstdint>

#include "src/models/lfm2/hip/lfm2_kernels.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace superroc::hip {

// LFM2: fp32 KV write reading the position from device memory. Same math as
// GlimmerWriteKvKernel, but pos = *d_pos (d_pos points at the pos word of the
// decode prompt_tokens pair, copied to the device before the step).
Lfm2LaunchStatus LaunchLfm2WriteKvDev(
    const float* k, const float* v, float* k_cache, float* v_cache,
    std::uint32_t layer, const std::uint32_t* d_pos, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, hipStream_t stream);

// LFM2: split-K flash-decoding attention. Causal over positions 0..*d_pos
// (no window, no gate), scale 1/sqrt(head_dim). Kernel A grid
// (num_kv_heads, n_splits) splits the n = *d_pos + 1 positions into
// ceil(n / n_splits) chunks and writes per (q head, split) the running max m,
// sum l and accumulator acc[head_dim] into partial, layout
// [num_heads][n_splits][head_dim + 2]. Kernel B grid (num_heads) combines the
// splits into out. Grid shapes depend only on heads/kv_heads/n_splits (not on
// the position), so the pair can live in a HIP graph. Host checks:
// head_dim == 64, heads % num_kv_heads == 0, group <= 8, 1 <= n_splits <= 64,
// pointers non-null; hipGetLastError() after each launch (audit F04 style).
Lfm2LaunchStatus LaunchLfm2DecodeAttention(
    const float* q, const float* k_cache, const float* v_cache, float* out,
    std::uint32_t layer, const std::uint32_t* d_pos, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t n_splits, float* partial, hipStream_t stream);

// LFM2 (SPEC3): fp16 token-major KV, the production plane layout: element
// ((layer*max_ctx+pos)*num_kv_heads+kvh)*head_dim+d, V plane = K + plane
// elements. Write rounds to nearest even; attention converts to fp32 on load
// and is otherwise the same split-K math and checks as above.
Lfm2LaunchStatus LaunchLfm2WriteKvDevF16(
    const float* k, const float* v, std::uint16_t* k_cache,
    std::uint16_t* v_cache, std::uint32_t layer, const std::uint32_t* d_pos,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, hipStream_t stream);

Lfm2LaunchStatus LaunchLfm2DecodeAttentionF16(
    const float* q, const std::uint16_t* k_cache, const std::uint16_t* v_cache,
    float* out, std::uint32_t layer, const std::uint32_t* d_pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, std::uint32_t n_splits,
    float* partial, hipStream_t stream);

}  // namespace superroc::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // SUPERROC_MODELS_LFM2_HIP_LFM2_ATTENTION_HPP_
