#ifndef SUPERROC_MODELS_LLAMA_HIP_LLAMA_ROPE_HPP_
#define SUPERROC_MODELS_LLAMA_HIP_LLAMA_ROPE_HPP_

#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace superroc::hip {

/// Adjacent-pair (llama.cpp ROPE_TYPE_NORM) RoPE: rotate dims 2i and 2i+1
/// with theta_i = freq_base^(-2i/rotary_dim), angle = pos * theta_i.
void LaunchRoPENorm(float* q, float* k, std::uint32_t num_heads,
                    std::uint32_t num_kv_heads, std::uint32_t head_dim,
                    std::uint32_t rotary_dim, const std::uint32_t* d_pos,
                    float rope_theta, hipStream_t stream);

/// Prefill: row b uses position start_pos+b.
void LaunchBatchedRoPENorm(float* q, float* k, std::size_t batch_size,
                           std::uint32_t num_heads, std::uint32_t num_kv_heads,
                           std::uint32_t head_dim, std::uint32_t rotary_dim,
                           std::uint32_t start_pos, float rope_theta,
                           hipStream_t stream);

}  // namespace superroc::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // SUPERROC_MODELS_LLAMA_HIP_LLAMA_ROPE_HPP_
