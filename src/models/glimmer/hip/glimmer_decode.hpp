#ifndef GUFO_MODELS_GLIMMER_HIP_GLIMMER_DECODE_HPP_
#define GUFO_MODELS_GLIMMER_HIP_GLIMMER_DECODE_HPP_

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::models {
struct QwenModelWeights;
namespace qwen::vision {
class DeviceInput;
}
}  // namespace gufo::models

namespace gufo::hip {

class QwenGpuArena;

/// GLIMMER: host launch result. `error` is a static string when ok is false.
struct GlimmerLaunchStatus {
  bool ok{false};
  const char* error{nullptr};
};

/// GLIMMER: NORM RoPE. Pairs are (q[2i], q[2i+1]). Refuses before launch.
[[nodiscard]] GlimmerLaunchStatus LaunchGlimmerNormRope(
    float* heads, std::uint32_t count, std::uint32_t head_dim,
    std::uint32_t rotary_dim, std::uint32_t pos, float theta,
    hipStream_t stream);

/// GLIMMER: fp32 KV write. Layout matches the fp32 half of WriteKVCacheKernel.
[[nodiscard]] GlimmerLaunchStatus LaunchGlimmerWriteKv(
    const float* k, const float* v, float* k_cache, float* v_cache,
    std::uint32_t layer, std::uint32_t pos, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, std::size_t kv_elems,
    std::size_t cache_elems, hipStream_t stream);

/// GLIMMER: floats that fit in the 48 KiB dynamic score tile.
[[nodiscard]] std::uint32_t GlimmerScoreTileElems() noexcept;

/// GLIMMER: windowed fp32 attention. Window 0 keeps every position. A span
/// longer than one 48 KiB score tile is online-softmaxed across tiles.
[[nodiscard]] GlimmerLaunchStatus LaunchGlimmerWindowAttention(
    const float* q, const float* gate, const float* k_cache,
    const float* v_cache, float* out, std::uint32_t layer, std::uint32_t pos,
    std::uint32_t window, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, std::size_t q_elems,
    std::size_t gate_elems, std::size_t cache_elems, std::size_t out_elems,
    hipStream_t stream, const float* sinks = nullptr);

/// GLIMMER: NORM RoPE for a prefill chunk. Position of row b is start_pos+b.
[[nodiscard]] GlimmerLaunchStatus LaunchGlimmerBatchedNormRope(
    float* heads, std::uint32_t count, std::uint32_t head_dim,
    std::uint32_t rotary_dim, std::uint32_t start_pos, std::uint32_t batch,
    float theta, hipStream_t stream);

/// GLIMMER: fp32 KV write for a prefill chunk. Row b lands at start_pos+b.
[[nodiscard]] GlimmerLaunchStatus LaunchGlimmerBatchedWriteKv(
    const float* k, const float* v, float* k_cache, float* v_cache,
    std::uint32_t layer, std::uint32_t start_pos, std::uint32_t batch,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, std::size_t kv_row_elems, std::size_t cache_elems,
    hipStream_t stream);

/// GLIMMER: causal windowed attention for a prefill chunk. Query row b attends
/// through position start_pos+b. Each block's score tile stays within 48 KiB.
[[nodiscard]] GlimmerLaunchStatus LaunchGlimmerPrefillAttention(
    const float* q, const float* gate, const float* k_cache,
    const float* v_cache, float* out, std::uint32_t layer,
    std::uint32_t start_pos, std::uint32_t batch, std::uint32_t window,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, std::size_t q_elems,
    std::size_t gate_elems, std::size_t cache_elems, std::size_t out_elems,
    hipStream_t stream, const float* sinks = nullptr);

/// GLIMMER: y = scale * x, then cap * tanh(y / cap) when cap != 0.
[[nodiscard]] GlimmerLaunchStatus LaunchGlimmerScaleSoftcap(float* logits,
                                                            std::size_t count,
                                                            float scale,
                                                            float cap,
                                                            hipStream_t stream);

/// GLIMMER: one decode step. Throws before any kernel if an index is illegal.
/// Does not enter HIP graph capture.
void ExecuteGlimmerDecodeStep(QwenGpuArena& arena,
                              const models::QwenModelWeights& weights,
                              tokenization::TokenId token_id, std::uint32_t pos,
                              bool compute_logits,
                              models::qwen::vision::DeviceInput* image_input);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_GLIMMER_HIP_GLIMMER_DECODE_HPP_
