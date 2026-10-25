#if defined(ENGINE_ENABLE_HIP)
#include "src/models/llama/hip/llama_rope.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/hip_launch_check.hpp"

namespace gufo::hip {

__global__ void RoPENormPtrKernel(float* __restrict__ q, float* __restrict__ k,
                                  std::uint32_t num_heads,
                                  std::uint32_t num_kv_heads,
                                  std::uint32_t head_dim,
                                  std::uint32_t rotary_dim,
                                  const std::uint32_t* __restrict__ d_pos,
                                  float rope_theta) {
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t n_pairs = rotary_dim / 2;
  const std::size_t total_q = static_cast<std::size_t>(num_heads) * n_pairs;
  const std::size_t total_k = static_cast<std::size_t>(num_kv_heads) * n_pairs;
  const std::uint32_t pos = *d_pos;

  if (tid < total_q && q != nullptr) {
    const std::uint32_t h = static_cast<std::uint32_t>(tid / n_pairs);
    const std::uint32_t pair = static_cast<std::uint32_t>(tid % n_pairs);
    const float exponent =
        static_cast<float>(2 * pair) / static_cast<float>(rotary_dim);
    const float inv_freq = 1.0F / powf(rope_theta, exponent);
    const float angle = static_cast<float>(pos) * inv_freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    const std::size_t idx0 =
        static_cast<std::size_t>(h) * head_dim + static_cast<std::size_t>(pair) * 2U;
    const float x0 = q[idx0];
    const float x1 = q[idx0 + 1U];
    q[idx0] = x0 * cos_val - x1 * sin_val;
    q[idx0 + 1U] = x0 * sin_val + x1 * cos_val;
  }
  if (tid < total_k && k != nullptr) {
    const std::uint32_t h = static_cast<std::uint32_t>(tid / n_pairs);
    const std::uint32_t pair = static_cast<std::uint32_t>(tid % n_pairs);
    const float exponent =
        static_cast<float>(2 * pair) / static_cast<float>(rotary_dim);
    const float inv_freq = 1.0F / powf(rope_theta, exponent);
    const float angle = static_cast<float>(pos) * inv_freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    const std::size_t idx0 =
        static_cast<std::size_t>(h) * head_dim + static_cast<std::size_t>(pair) * 2U;
    const float x0 = k[idx0];
    const float x1 = k[idx0 + 1U];
    k[idx0] = x0 * cos_val - x1 * sin_val;
    k[idx0 + 1U] = x0 * sin_val + x1 * cos_val;
  }
}

void LaunchRoPENorm(float* q, float* k, std::uint32_t num_heads,
                    std::uint32_t num_kv_heads, std::uint32_t head_dim,
                    std::uint32_t rotary_dim, const std::uint32_t* d_pos,
                    float rope_theta, hipStream_t stream) {
  const std::size_t n_pairs = rotary_dim / 2;
  const std::size_t total_pairs =
      std::max(static_cast<std::size_t>(num_heads) * n_pairs,
               static_cast<std::size_t>(num_kv_heads) * n_pairs);
  const std::size_t block_size = 256;
  const std::size_t grid_size = (total_pairs + block_size - 1) / block_size;
  RoPENormPtrKernel<<<grid_size, block_size, 0, stream>>>(
      q, k, num_heads, num_kv_heads, head_dim, rotary_dim, d_pos, rope_theta);
  GUFO_CHECK_LAUNCH("RoPENormPtrKernel");
}

__global__ void BatchedRoPENormKernel(float* __restrict__ q,
                                      float* __restrict__ k,
                                      std::size_t batch_size,
                                      std::uint32_t num_heads,
                                      std::uint32_t num_kv_heads,
                                      std::uint32_t head_dim,
                                      std::uint32_t rotary_dim,
                                      std::uint32_t start_pos,
                                      float rope_theta) {
  const std::size_t b = blockIdx.y;
  if (b >= batch_size) {
    return;
  }
  const std::size_t n_pairs = rotary_dim / 2;
  const std::size_t pair_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::uint32_t pos = start_pos + static_cast<std::uint32_t>(b);

  if (pair_idx < static_cast<std::size_t>(num_heads) * n_pairs &&
      q != nullptr) {
    const std::uint32_t h = static_cast<std::uint32_t>(pair_idx / n_pairs);
    const std::uint32_t pair = static_cast<std::uint32_t>(pair_idx % n_pairs);
    const float exponent =
        static_cast<float>(2 * pair) / static_cast<float>(rotary_dim);
    const float inv_freq = 1.0F / powf(rope_theta, exponent);
    const float angle = static_cast<float>(pos) * inv_freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    const std::size_t q_stride =
        static_cast<std::size_t>(num_heads) * head_dim;
    const std::size_t idx0 = b * q_stride +
                             static_cast<std::size_t>(h) * head_dim +
                             static_cast<std::size_t>(pair) * 2U;
    const float x0 = q[idx0];
    const float x1 = q[idx0 + 1U];
    q[idx0] = x0 * cos_val - x1 * sin_val;
    q[idx0 + 1U] = x0 * sin_val + x1 * cos_val;
  }
  if (pair_idx < static_cast<std::size_t>(num_kv_heads) * n_pairs &&
      k != nullptr) {
    const std::uint32_t h = static_cast<std::uint32_t>(pair_idx / n_pairs);
    const std::uint32_t pair = static_cast<std::uint32_t>(pair_idx % n_pairs);
    const float exponent =
        static_cast<float>(2 * pair) / static_cast<float>(rotary_dim);
    const float inv_freq = 1.0F / powf(rope_theta, exponent);
    const float angle = static_cast<float>(pos) * inv_freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    const std::size_t k_stride =
        static_cast<std::size_t>(num_kv_heads) * head_dim;
    const std::size_t idx0 = b * k_stride +
                             static_cast<std::size_t>(h) * head_dim +
                             static_cast<std::size_t>(pair) * 2U;
    const float x0 = k[idx0];
    const float x1 = k[idx0 + 1U];
    k[idx0] = x0 * cos_val - x1 * sin_val;
    k[idx0 + 1U] = x0 * sin_val + x1 * cos_val;
  }
}

void LaunchBatchedRoPENorm(float* q, float* k, std::size_t batch_size,
                           std::uint32_t num_heads, std::uint32_t num_kv_heads,
                           std::uint32_t head_dim, std::uint32_t rotary_dim,
                           std::uint32_t start_pos, float rope_theta,
                           hipStream_t stream) {
  const std::size_t n_pairs = rotary_dim / 2;
  const std::size_t total_pairs =
      std::max(static_cast<std::size_t>(num_heads) * n_pairs,
               static_cast<std::size_t>(num_kv_heads) * n_pairs);
  const std::size_t block_size = 256;
  const std::size_t grid_x = (total_pairs + block_size - 1) / block_size;
  const dim3 grid(static_cast<unsigned int>(grid_x),
                  static_cast<unsigned int>(batch_size));
  BatchedRoPENormKernel<<<grid, static_cast<unsigned int>(block_size), 0,
                          stream>>>(q, k, batch_size, num_heads, num_kv_heads,
                                    head_dim, rotary_dim, start_pos,
                                    rope_theta);
  GUFO_CHECK_LAUNCH("BatchedRoPENormKernel");
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
