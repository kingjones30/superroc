#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "src/models/qwen/hip/kernels/dflash_kernels.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"
#include "src/core/hip_launch_check.hpp"

namespace gufo::hip::kernels {

// The DFlash-2 draft head is the target LM head, which the Q8 target keeps in
// BF16 at 2.54 GB -- half of everything the draft graph reads. Drafting is
// purely weight-bandwidth bound, and the draft only steers which tokens the
// target verifies, so a draft-private Q8_0 copy cannot change an emitted token:
// the verifier still runs the target's BF16 head.
__global__ void dflash_quantize_bf16_to_q8_0_kernel(
    const hip_bfloat16* __restrict__ source,
    Q8_0Block* __restrict__ destination, uint64_t total_blocks) {
  const uint64_t block =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block >= total_blocks) {
    return;
  }
  const hip_bfloat16* values = source + (block * kQ8_0BlockSize);
  float absolute_max = 0.0F;
  for (uint32_t index = 0; index < kQ8_0BlockSize; ++index) {
    absolute_max =
        fmaxf(absolute_max, fabsf(static_cast<float>(values[index])));
  }
  const float scale = absolute_max / 127.0F;
  const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
  Q8_0Block& out = destination[block];
  out.d = __float2half(scale);
  for (uint32_t index = 0; index < kQ8_0BlockSize; ++index) {
    const float quantized = roundf(static_cast<float>(values[index]) * inverse);
    out.qs[index] =
        static_cast<int8_t>(fminf(fmaxf(quantized, -127.0F), 127.0F));
  }
}

void LaunchDFlashQuantizeBf16ToQ8_0(const void* bf16_source,
                                    void* q8_destination,
                                    std::size_t total_elements,
                                    hipStream_t stream) {
  if (bf16_source == nullptr || q8_destination == nullptr ||
      total_elements % kQ8_0BlockSize != 0) {
    return;
  }
  const std::uint64_t total_blocks = total_elements / kQ8_0BlockSize;
  constexpr std::uint32_t kThreads = 256;
  const std::uint64_t grid = (total_blocks + kThreads - 1) / kThreads;
  dflash_quantize_bf16_to_q8_0_kernel<<<static_cast<unsigned int>(grid),
                                        kThreads, 0, stream>>>(
      static_cast<const hip_bfloat16*>(bf16_source),
      static_cast<Q8_0Block*>(q8_destination), total_blocks);
  GUFO_CHECK_LAUNCH("dflash_quantize_bf16_to_q8_0_kernel");
}

__global__ void dflash_grouped_dynamic_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ dynamic_coefficients,
    const float* __restrict__ base_kernel, float* __restrict__ output,
    uint32_t num_tokens, uint32_t hidden_size, uint32_t kernel_size,
    uint32_t group_size, uint32_t side) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = static_cast<uint64_t>(num_tokens) * hidden_size;
  if (index >= total) {
    return;
  }

  const uint32_t token = index / hidden_size;
  const uint32_t channel = index % hidden_size;
  const uint32_t groups = hidden_size / group_size;
  const uint32_t group = channel / group_size;
  const uint32_t coefficients_per_token = 2U * kernel_size * groups;

  float sum = 0.0F;
  for (uint32_t tap = 0; tap < kernel_size; ++tap) {
    if (token < tap) {
      continue;
    }
    const uint64_t dynamic_index =
        static_cast<uint64_t>(token) * coefficients_per_token +
        (side * kernel_size + tap) * groups + group;
    const uint64_t base_index =
        static_cast<uint64_t>(side * kernel_size + tap) * hidden_size + channel;
    const uint64_t input_index =
        static_cast<uint64_t>(token - tap) * hidden_size + channel;
    sum += (base_kernel[base_index] + dynamic_coefficients[dynamic_index]) *
           input[input_index];
  }
  output[index] = sum;
}

__global__ void dflash_silu_mul_kernel(float* __restrict__ gate,
                                       const float* __restrict__ up,
                                       uint32_t total_elements) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  const float g = gate[idx];
  const float silu_g = g / (1.0F + expf(-g));
  gate[idx] = silu_g * up[idx];
}

// Coalesced QK loads and shared K/V reads for adjacent query heads. Each
// head retains its original dot-product and 256-thread softmax reduction.
template<uint32_t QueryHeads>
__global__ void dflash_noncausal_attention_wave_kernel(
    const float* __restrict__ q, const float* __restrict__ injected_k,
    const float* __restrict__ injected_v, const float* __restrict__ block_k,
    const float* __restrict__ block_v, float* __restrict__ out,
    uint32_t current_pos, uint32_t history_length, uint32_t block_count,
    uint32_t sliding_window, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t head_dim, float scale, uint32_t history_capacity) {
  constexpr uint32_t kPrefetch = 32;
  const uint32_t tok = blockIdx.x, h = blockIdx.y * QueryHeads;
  if (tok >= block_count || h + QueryHeads > num_q_heads)
    return;
  const uint32_t kv_h = h / (num_q_heads / num_kv_heads),
                 q_dim = num_q_heads * head_dim,
                 kv_dim = num_kv_heads * head_dim;
  const uint32_t query_position = current_pos + tok;
  const uint32_t history_start = current_pos + 1U > sliding_window
                                     ? current_pos + 1U - sliding_window
                                     : 0U;
  const uint32_t history_count = history_length > history_start
                                     ? history_length - history_start
                                     : 0U,
                 total_k = history_count + block_count;
  const float4* q_vec = reinterpret_cast<const float4*>(
      q + uint64_t(tok) * q_dim + uint64_t(h) * head_dim);
  const uint32_t num_vec = head_dim / 4U, lane = threadIdx.x & 31U,
                 wave = threadIdx.x >> 5U, waves = blockDim.x >> 5U;
  extern __shared__ float scores[];
  __shared__ float reduce[QueryHeads][256];
  for (uint32_t local = wave; local < history_count; local += waves) {
    const uint32_t position = history_start + local;
    const float4* key = reinterpret_cast<const float4*>(
        injected_k + uint64_t(position % history_capacity) * kv_dim +
        uint64_t(kv_h) * head_dim);
    float dot[QueryHeads] = {};
    for (uint32_t d = lane; d < num_vec; d += 32U) {
      const float4 kv = key[d];
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g) {
        const float4 qv = q_vec[g * num_vec + d];
        dot[g] += (qv.x * kv.x) + (qv.y * kv.y) + (qv.z * kv.z) + (qv.w * kv.w);
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        dot[g] += __shfl_xor(dot[g], offset);
    }
    if (lane == 0) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        scores[g * total_k + local] = query_position - position < sliding_window
                                          ? dot[g] * scale
                                          : -1e20F;
    }
  }
  for (uint32_t p = wave; p < block_count; p += waves) {
    const float4* key = reinterpret_cast<const float4*>(
        block_k + uint64_t(p) * kv_dim + uint64_t(kv_h) * head_dim);
    float dot[QueryHeads] = {};
    for (uint32_t d = lane; d < num_vec; d += 32U) {
      const float4 kv = key[d];
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g) {
        const float4 qv = q_vec[g * num_vec + d];
        dot[g] += (qv.x * kv.x) + (qv.y * kv.y) + (qv.z * kv.z) + (qv.w * kv.w);
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        dot[g] += __shfl_xor(dot[g], offset);
    }
    if (lane == 0) {
      const uint32_t key_position = current_pos + p;
      const uint32_t distance = query_position > key_position
                                    ? query_position - key_position
                                    : key_position - query_position;
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        scores[g * total_k + history_count + p] =
            distance < sliding_window ? dot[g] * scale : -1e20F;
    }
  }
  __syncthreads();
  float maxima[QueryHeads];
#pragma unroll
  for (unsigned g = 0; g < QueryHeads; ++g)
    maxima[g] = -1e20F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g)
      maxima[g] = fmaxf(maxima[g], scores[g * total_k + p]);
  }
#pragma unroll
  for (unsigned g = 0; g < QueryHeads; ++g)
    reduce[g][threadIdx.x] = maxima[g];
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        reduce[g][threadIdx.x] =
            fmaxf(reduce[g][threadIdx.x], reduce[g][threadIdx.x + stride]);
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned g = 0; g < QueryHeads; ++g)
    maxima[g] = reduce[g][0];
  __syncthreads();
  float sums[QueryHeads] = {};
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g) {
      const float e = expf(scores[g * total_k + p] - maxima[g]);
      scores[g * total_k + p] = e;
      sums[g] += e;
    }
  }
#pragma unroll
  for (unsigned g = 0; g < QueryHeads; ++g)
    reduce[g][threadIdx.x] = sums[g];
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        reduce[g][threadIdx.x] += reduce[g][threadIdx.x + stride];
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned g = 0; g < QueryHeads; ++g)
    sums[g] = 1.f / (reduce[g][0] + 1e-9F);
  __syncthreads();
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g)
      scores[g * total_k + p] *= sums[g];
  }
  __syncthreads();
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc[QueryHeads] = {};
    uint32_t slot = history_count > 0 ? history_start % history_capacity : 0;
    // Issue independent value loads together, then retain the original
    // ascending-key FMA sequence. Advancing the ring slot avoids a division
    // for every value without changing the persistent KV layout.
    uint32_t local = 0;
    for (; history_count - local >= kPrefetch; local += kPrefetch) {
      float values[kPrefetch];
#pragma unroll
      for (unsigned i = 0; i < kPrefetch; ++i) {
        values[i] =
            injected_v[uint64_t(slot) * kv_dim + uint64_t(kv_h) * head_dim + d];
        slot = slot + 1 == history_capacity ? 0 : slot + 1;
      }
#pragma unroll
      for (unsigned i = 0; i < kPrefetch; ++i) {
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g)
          acc[g] += scores[g * total_k + local + i] * values[i];
      }
    }
    for (; local < history_count; ++local) {
      const float value =
          injected_v[uint64_t(slot) * kv_dim + uint64_t(kv_h) * head_dim + d];
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        acc[g] += scores[g * total_k + local] * value;
      slot = slot + 1 == history_capacity ? 0 : slot + 1;
    }
    for (uint32_t p = 0; p < block_count; ++p) {
      const float value =
          block_v[uint64_t(p) * kv_dim + uint64_t(kv_h) * head_dim + d];
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        acc[g] += scores[g * total_k + history_count + p] * value;
    }
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g)
      out[uint64_t(tok) * q_dim + uint64_t(h + g) * head_dim + d] = acc[g];
  }
}

// Largest workgroup LDS allocation on gfx1151: dynamic plus static bytes.
constexpr std::size_t kDFlashMaxSharedBytes = 64U * 1024U;
// The tiled route holds head_dim / 256 value accumulators per thread.
constexpr uint32_t kDFlashTiledMaxDimsPerThread = 4;

// Long-history form of the wave kernel. The wave kernel keeps one LDS score
// per attended key, so a drafter without a sliding window (DSpark: the window
// falls back to the context length) cannot launch once history_count +
// block_count exceeds the workgroup's 64 KiB. Here only one tile of scores is
// resident: each tile keeps the wave kernel's per-key dot products and
// 256-thread reductions, and an online softmax rescales the running sum and
// value accumulators whenever a later tile raises the maximum. Block keys are
// always visible, so the final maximum is finite and any all-masked tile
// before it is multiplied out by its correction of exp(-1e20 - max) = 0.
template<uint32_t QueryHeads>
__global__ void dflash_noncausal_attention_tiled_kernel(
    const float* __restrict__ q, const float* __restrict__ injected_k,
    const float* __restrict__ injected_v, const float* __restrict__ block_k,
    const float* __restrict__ block_v, float* __restrict__ out,
    uint32_t current_pos, uint32_t history_length, uint32_t block_count,
    uint32_t sliding_window, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t head_dim, float scale, uint32_t history_capacity,
    uint32_t tile_keys) {
  constexpr uint32_t kPrefetch = 32;
  const uint32_t tok = blockIdx.x, h = blockIdx.y * QueryHeads;
  if (tok >= block_count || h + QueryHeads > num_q_heads)
    return;
  const uint32_t kv_h = h / (num_q_heads / num_kv_heads),
                 q_dim = num_q_heads * head_dim,
                 kv_dim = num_kv_heads * head_dim;
  const uint32_t query_position = current_pos + tok;
  const uint32_t history_start = current_pos + 1U > sliding_window
                                     ? current_pos + 1U - sliding_window
                                     : 0U;
  const uint32_t history_count = history_length > history_start
                                     ? history_length - history_start
                                     : 0U,
                 total_k = history_count + block_count;
  const float4* q_vec = reinterpret_cast<const float4*>(
      q + uint64_t(tok) * q_dim + uint64_t(h) * head_dim);
  const uint32_t num_vec = head_dim / 4U, lane = threadIdx.x & 31U,
                 wave = threadIdx.x >> 5U, waves = blockDim.x >> 5U;
  extern __shared__ float scores[];
  __shared__ float reduce[QueryHeads][256];
  float running_max[QueryHeads], running_sum[QueryHeads];
  float acc[kDFlashTiledMaxDimsPerThread][QueryHeads];
#pragma unroll
  for (unsigned g = 0; g < QueryHeads; ++g) {
    running_max[g] = -1e20F;
    running_sum[g] = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < kDFlashTiledMaxDimsPerThread; ++i)
      acc[i][g] = 0.0F;
  }
  for (uint32_t tile_start = 0; tile_start < total_k; tile_start += tile_keys) {
    const uint32_t tile_count = min(tile_keys, total_k - tile_start);
    for (uint32_t local = wave; local < tile_count; local += waves) {
      const uint32_t key_index = tile_start + local;
      const float* key_row;
      bool visible;
      if (key_index < history_count) {
        const uint32_t position = history_start + key_index;
        key_row = injected_k + uint64_t(position % history_capacity) * kv_dim;
        visible = query_position - position < sliding_window;
      } else {
        const uint32_t p = key_index - history_count;
        key_row = block_k + uint64_t(p) * kv_dim;
        const uint32_t key_position = current_pos + p;
        const uint32_t distance = query_position > key_position
                                      ? query_position - key_position
                                      : key_position - query_position;
        visible = distance < sliding_window;
      }
      const float4* key =
          reinterpret_cast<const float4*>(key_row + uint64_t(kv_h) * head_dim);
      float dot[QueryHeads] = {};
      for (uint32_t d = lane; d < num_vec; d += 32U) {
        const float4 kv = key[d];
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g) {
          const float4 qv = q_vec[g * num_vec + d];
          dot[g] +=
              (qv.x * kv.x) + (qv.y * kv.y) + (qv.z * kv.z) + (qv.w * kv.w);
        }
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g)
          dot[g] += __shfl_xor(dot[g], offset);
      }
      if (lane == 0) {
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g)
          scores[g * tile_keys + local] = visible ? dot[g] * scale : -1e20F;
      }
    }
    __syncthreads();
    float maxima[QueryHeads];
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g)
      maxima[g] = -1e20F;
    for (uint32_t p = threadIdx.x; p < tile_count; p += blockDim.x) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        maxima[g] = fmaxf(maxima[g], scores[g * tile_keys + p]);
    }
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g)
      reduce[g][threadIdx.x] = maxima[g];
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) {
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g)
          reduce[g][threadIdx.x] =
              fmaxf(reduce[g][threadIdx.x], reduce[g][threadIdx.x + stride]);
      }
      __syncthreads();
    }
    float correction[QueryHeads];
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g) {
      const float next_max = fmaxf(running_max[g], reduce[g][0]);
      correction[g] = expf(running_max[g] - next_max);
      running_max[g] = next_max;
    }
    __syncthreads();
    float sums[QueryHeads] = {};
    for (uint32_t p = threadIdx.x; p < tile_count; p += blockDim.x) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g) {
        const float e = expf(scores[g * tile_keys + p] - running_max[g]);
        scores[g * tile_keys + p] = e;
        sums[g] += e;
      }
    }
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g)
      reduce[g][threadIdx.x] = sums[g];
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) {
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g)
          reduce[g][threadIdx.x] += reduce[g][threadIdx.x + stride];
      }
      __syncthreads();
    }
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g)
      running_sum[g] = running_sum[g] * correction[g] + reduce[g][0];
    // The history keys of this tile, then its block keys, in ascending order.
    const uint32_t tile_history =
        tile_start < history_count
            ? min(history_count, tile_start + tile_count) - tile_start
            : 0U;
#pragma unroll
    for (unsigned i = 0; i < kDFlashTiledMaxDimsPerThread; ++i) {
      const uint32_t d = threadIdx.x + i * blockDim.x;
      if (d >= head_dim)
        continue;
      float tile_acc[QueryHeads] = {};
      uint32_t slot = tile_history > 0
                          ? (history_start + tile_start) % history_capacity
                          : 0U;
      uint32_t local = 0;
      for (; tile_history - local >= kPrefetch; local += kPrefetch) {
        float values[kPrefetch];
#pragma unroll
        for (unsigned j = 0; j < kPrefetch; ++j) {
          values[j] = injected_v[uint64_t(slot) * kv_dim +
                                 uint64_t(kv_h) * head_dim + d];
          slot = slot + 1 == history_capacity ? 0 : slot + 1;
        }
#pragma unroll
        for (unsigned j = 0; j < kPrefetch; ++j) {
#pragma unroll
          for (unsigned g = 0; g < QueryHeads; ++g)
            tile_acc[g] += scores[g * tile_keys + local + j] * values[j];
        }
      }
      for (; local < tile_history; ++local) {
        const float value =
            injected_v[uint64_t(slot) * kv_dim + uint64_t(kv_h) * head_dim + d];
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g)
          tile_acc[g] += scores[g * tile_keys + local] * value;
        slot = slot + 1 == history_capacity ? 0 : slot + 1;
      }
      for (; local < tile_count; ++local) {
        const uint32_t p = tile_start + local - history_count;
        const float value =
            block_v[uint64_t(p) * kv_dim + uint64_t(kv_h) * head_dim + d];
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g)
          tile_acc[g] += scores[g * tile_keys + local] * value;
      }
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        acc[i][g] = acc[i][g] * correction[g] + tile_acc[g];
    }
    // The next tile overwrites the scores this one just read.
    __syncthreads();
  }
#pragma unroll
  for (unsigned i = 0; i < kDFlashTiledMaxDimsPerThread; ++i) {
    const uint32_t d = threadIdx.x + i * blockDim.x;
    if (d >= head_dim)
      continue;
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g)
      out[uint64_t(tok) * q_dim + uint64_t(h + g) * head_dim + d] =
          acc[i][g] / (running_sum[g] + 1e-9F);
  }
}

// Split-K (flash-decoding) form for long histories. The wave and tiled
// kernels launch block_count x num_q_heads / 2 workgroups (16 for DSpark), and
// each streams its KV head's whole history, so a 26,821-key prompt leaves most
// of the GPU idle while a few workgroups read tens of megabytes each. Here one
// workgroup covers one draft row and the four query heads of one KV head over
// one contiguous key range, and blockIdx.z spreads the ranges across the GPU.
// Each workgroup keeps the tiled kernel's online softmax over its range and
// writes the unnormalised row (running max, running sum, value accumulator) to
// the partial buffers; the combine kernel rescales the ranges to the global
// maximum. Block keys are always visible and sit in the last range, so the
// global maximum is finite and a range whose keys are all masked for a row
// enters with weight exp(-1e20 - max) = 0. Four query rows per workgroup and
// two keys per wave pass measured fastest on gfx1151 (one DSpark layer at
// 26,821 keys: 540 us, against 881 us with sixteen rows per workgroup and
// 4,020 us for the tiled route): wider workgroups hold more Q registers and
// leave fewer loads in flight.
constexpr uint32_t kDFlashSplitThreads = 256;
// Query heads per workgroup: the route needs num_q_heads / num_kv_heads to be
// a multiple of this.
constexpr uint32_t kDFlashSplitQueryHeads = 4;
// float4 Q vectors each lane keeps per query head: head_dim <= 256.
constexpr uint32_t kDFlashSplitMaxVec = 2;
// Keys each wave scores per pass, and V values each thread loads ahead.
constexpr uint32_t kDFlashSplitKeysPerWave = 2;
constexpr uint32_t kDFlashSplitPrefetch = 8;

__global__ void __launch_bounds__(kDFlashSplitThreads)
    dflash_noncausal_attention_split_kernel(
        const float* __restrict__ q, const float* __restrict__ injected_k,
        const float* __restrict__ injected_v, const float* __restrict__ block_k,
        const float* __restrict__ block_v, float* __restrict__ partial_acc,
        float* __restrict__ partial_ml, uint32_t current_pos,
        uint32_t history_length, uint32_t block_count, uint32_t sliding_window,
        uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim,
        float scale, uint32_t history_capacity, uint32_t keys_per_split,
        uint32_t tile_keys) {
  constexpr uint32_t kHeads = kDFlashSplitQueryHeads;
  constexpr uint32_t kWaves = kDFlashSplitThreads / 32U;
  constexpr uint32_t kKeysPerWave = kDFlashSplitKeysPerWave;
  constexpr uint32_t kPrefetch = kDFlashSplitPrefetch;
  const uint32_t tok = blockIdx.x, h = blockIdx.y * kHeads,
                 split = blockIdx.z;
  const uint32_t kv_h = h / (num_q_heads / num_kv_heads),
                 q_dim = num_q_heads * head_dim,
                 kv_dim = num_kv_heads * head_dim;
  const uint32_t query_position = current_pos + tok;
  const uint32_t history_start = current_pos + 1U > sliding_window
                                     ? current_pos + 1U - sliding_window
                                     : 0U;
  const uint32_t history_count = history_length > history_start
                                     ? history_length - history_start
                                     : 0U,
                 total_k = history_count + block_count;
  const uint32_t key_begin = split * keys_per_split,
                 key_end = min(total_k, key_begin + keys_per_split);
  const uint32_t num_vec = head_dim / 4U, lane = threadIdx.x & 31U,
                 wave = threadIdx.x >> 5U;
  extern __shared__ float scores[];
  __shared__ float wave_stat[kWaves][kHeads];

  float4 q_reg[kHeads][kDFlashSplitMaxVec];
#pragma unroll
  for (unsigned g = 0; g < kHeads; ++g) {
    const float4* q_vec = reinterpret_cast<const float4*>(
        q + uint64_t(tok) * q_dim + uint64_t(h + g) * head_dim);
#pragma unroll
    for (unsigned v = 0; v < kDFlashSplitMaxVec; ++v) {
      const uint32_t d = lane + v * 32U;
      q_reg[g][v] = d < num_vec ? q_vec[d] : make_float4(0.F, 0.F, 0.F, 0.F);
    }
  }
  float running_max[kHeads], running_sum[kHeads], acc[kHeads];
#pragma unroll
  for (unsigned g = 0; g < kHeads; ++g) {
    running_max[g] = -1e20F;
    running_sum[g] = 0.0F;
    acc[g] = 0.0F;
  }
  const uint32_t d_own = threadIdx.x;

  for (uint32_t tile_start = key_begin; tile_start < key_end;
       tile_start += tile_keys) {
    const uint32_t tile_count = min(tile_keys, key_end - tile_start);
    // Each wave scores kKeysPerWave keys per pass, all loads issued before
    // the first dot product.
    for (uint32_t base = wave * kKeysPerWave; base < tile_count;
         base += kWaves * kKeysPerWave) {
      float4 kv[kKeysPerWave][kDFlashSplitMaxVec];
      bool visible[kKeysPerWave];
#pragma unroll
      for (unsigned u = 0; u < kKeysPerWave; ++u) {
        const uint32_t key_index = tile_start + base + u;
        const bool live = base + u < tile_count;
        const float* key_row;
        if (key_index < history_count) {
          const uint32_t position = history_start + key_index;
          key_row =
              injected_k + uint64_t(position % history_capacity) * kv_dim;
          visible[u] = query_position - position < sliding_window;
        } else {
          const uint32_t p = live ? key_index - history_count : 0U;
          key_row = block_k + uint64_t(p) * kv_dim;
          visible[u] = (tok > p ? tok - p : p - tok) < sliding_window;
        }
        const float4* key = reinterpret_cast<const float4*>(
            key_row + uint64_t(kv_h) * head_dim);
#pragma unroll
        for (unsigned v = 0; v < kDFlashSplitMaxVec; ++v) {
          const uint32_t d = lane + v * 32U;
          kv[u][v] = live && d < num_vec ? key[d]
                                         : make_float4(0.F, 0.F, 0.F, 0.F);
        }
      }
      float dot[kKeysPerWave][kHeads];
#pragma unroll
      for (unsigned u = 0; u < kKeysPerWave; ++u) {
#pragma unroll
        for (unsigned g = 0; g < kHeads; ++g) {
          float sum = 0.0F;
#pragma unroll
          for (unsigned v = 0; v < kDFlashSplitMaxVec; ++v) {
            const float4 qv = q_reg[g][v];
            sum += (qv.x * kv[u][v].x) + (qv.y * kv[u][v].y) +
                   (qv.z * kv[u][v].z) + (qv.w * kv[u][v].w);
          }
          dot[u][g] = sum;
        }
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (unsigned u = 0; u < kKeysPerWave; ++u) {
#pragma unroll
          for (unsigned g = 0; g < kHeads; ++g)
            dot[u][g] += __shfl_xor(dot[u][g], offset);
        }
      }
      if (lane == 0) {
#pragma unroll
        for (unsigned u = 0; u < kKeysPerWave; ++u) {
          if (base + u >= tile_count)
            continue;
#pragma unroll
          for (unsigned g = 0; g < kHeads; ++g)
            scores[g * tile_keys + base + u] =
                visible[u] ? dot[u][g] * scale : -1e20F;
        }
      }
    }
    __syncthreads();

    float tile_max[kHeads];
#pragma unroll
    for (unsigned g = 0; g < kHeads; ++g)
      tile_max[g] = -1e20F;
    for (uint32_t p = threadIdx.x; p < tile_count; p += kDFlashSplitThreads) {
#pragma unroll
      for (unsigned g = 0; g < kHeads; ++g)
        tile_max[g] = fmaxf(tile_max[g], scores[g * tile_keys + p]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (unsigned g = 0; g < kHeads; ++g)
        tile_max[g] = fmaxf(tile_max[g], __shfl_xor(tile_max[g], offset));
    }
    if (lane == 0) {
#pragma unroll
      for (unsigned g = 0; g < kHeads; ++g)
        wave_stat[wave][g] = tile_max[g];
    }
    __syncthreads();
    float correction[kHeads];
#pragma unroll
    for (unsigned g = 0; g < kHeads; ++g) {
      float maximum = wave_stat[0][g];
#pragma unroll
      for (unsigned w = 1; w < kWaves; ++w)
        maximum = fmaxf(maximum, wave_stat[w][g]);
      const float next_max = fmaxf(running_max[g], maximum);
      correction[g] = expf(running_max[g] - next_max);
      running_max[g] = next_max;
    }
    __syncthreads();

    float sums[kHeads] = {};
    for (uint32_t p = threadIdx.x; p < tile_count; p += kDFlashSplitThreads) {
#pragma unroll
      for (unsigned g = 0; g < kHeads; ++g) {
        const float e = expf(scores[g * tile_keys + p] - running_max[g]);
        scores[g * tile_keys + p] = e;
        sums[g] += e;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (unsigned g = 0; g < kHeads; ++g)
        sums[g] += __shfl_xor(sums[g], offset);
    }
    if (lane == 0) {
#pragma unroll
      for (unsigned g = 0; g < kHeads; ++g)
        wave_stat[wave][g] = sums[g];
    }
    __syncthreads();
#pragma unroll
    for (unsigned g = 0; g < kHeads; ++g) {
      float total = 0.0F;
#pragma unroll
      for (unsigned w = 0; w < kWaves; ++w)
        total += wave_stat[w][g];
      running_sum[g] = running_sum[g] * correction[g] + total;
    }

    // The history keys of this tile, then its block keys, in ascending order.
    if (d_own < head_dim) {
      const uint32_t tile_history =
          tile_start < history_count
              ? min(history_count, tile_start + tile_count) - tile_start
              : 0U;
      float tile_acc[kHeads] = {};
      uint32_t slot = tile_history > 0
                          ? (history_start + tile_start) % history_capacity
                          : 0U;
      const float* value_column =
          injected_v + uint64_t(kv_h) * head_dim + d_own;
      uint32_t local = 0;
      for (; tile_history - local >= kPrefetch; local += kPrefetch) {
        float values[kPrefetch];
#pragma unroll
        for (unsigned j = 0; j < kPrefetch; ++j) {
          values[j] = value_column[uint64_t(slot) * kv_dim];
          slot = slot + 1 == history_capacity ? 0 : slot + 1;
        }
#pragma unroll
        for (unsigned j = 0; j < kPrefetch; ++j) {
#pragma unroll
          for (unsigned g = 0; g < kHeads; ++g)
            tile_acc[g] += scores[g * tile_keys + local + j] * values[j];
        }
      }
      for (; local < tile_history; ++local) {
        const float value = value_column[uint64_t(slot) * kv_dim];
#pragma unroll
        for (unsigned g = 0; g < kHeads; ++g)
          tile_acc[g] += scores[g * tile_keys + local] * value;
        slot = slot + 1 == history_capacity ? 0 : slot + 1;
      }
      for (; local < tile_count; ++local) {
        const uint32_t p = tile_start + local - history_count;
        const float value =
            block_v[uint64_t(p) * kv_dim + uint64_t(kv_h) * head_dim + d_own];
#pragma unroll
        for (unsigned g = 0; g < kHeads; ++g)
          tile_acc[g] += scores[g * tile_keys + local] * value;
      }
#pragma unroll
      for (unsigned g = 0; g < kHeads; ++g)
        acc[g] = acc[g] * correction[g] + tile_acc[g];
    }
    // The next tile overwrites the scores and statistics this one just read.
    __syncthreads();
  }

#pragma unroll
  for (unsigned g = 0; g < kHeads; ++g) {
    const uint64_t slot_index =
        (uint64_t(split) * block_count + tok) * num_q_heads + h + g;
    if (d_own < head_dim)
      partial_acc[slot_index * head_dim + d_own] = acc[g];
    if (threadIdx.x == 0) {
      partial_ml[slot_index * 2U] = running_max[g];
      partial_ml[slot_index * 2U + 1U] = running_sum[g];
    }
  }
}

// Most key ranges the split route launches per row; the combine kernel holds
// one weight per range in LDS.
constexpr uint32_t kDFlashSplitMaxSplits = 64;

// Rescales every key range of one (draft token, query head) row to the global
// maximum and divides by the combined sum, as the tiled kernel does at the end
// of its single range.
__global__ void dflash_noncausal_attention_split_combine_kernel(
    const float* __restrict__ partial_acc, const float* __restrict__ partial_ml,
    float* __restrict__ out, uint32_t block_count, uint32_t num_q_heads,
    uint32_t head_dim, uint32_t splits) {
  const uint32_t tok = blockIdx.x, h = blockIdx.y;
  __shared__ float weight[kDFlashSplitMaxSplits];
  __shared__ float denominator;
  const auto slot_index = [&](uint32_t split) {
    return (uint64_t(split) * block_count + tok) * num_q_heads + h;
  };
  if (threadIdx.x == 0) {
    float maximum = -1e20F;
    for (uint32_t s = 0; s < splits; ++s)
      maximum = fmaxf(maximum, partial_ml[slot_index(s) * 2U]);
    float sum = 0.0F;
    for (uint32_t s = 0; s < splits; ++s) {
      const float w = expf(partial_ml[slot_index(s) * 2U] - maximum);
      weight[s] = w;
      sum += w * partial_ml[slot_index(s) * 2U + 1U];
    }
    denominator = sum + 1e-9F;
  }
  __syncthreads();
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float value = 0.0F;
    for (uint32_t s = 0; s < splits; ++s)
      value += weight[s] * partial_acc[slot_index(s) * head_dim + d];
    out[uint64_t(tok) * num_q_heads * head_dim + uint64_t(h) * head_dim + d] =
        value / denominator;
  }
}

__global__ void dflash_noncausal_attention_kernel(
    const float* __restrict__ q, const float* __restrict__ injected_k,
    const float* __restrict__ injected_v, const float* __restrict__ block_k,
    const float* __restrict__ block_v, float* __restrict__ out,
    uint32_t current_pos, uint32_t history_length, uint32_t block_count,
    uint32_t sliding_window, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t head_dim, float scale, uint32_t history_capacity) {
  const uint32_t tok = blockIdx.x;
  const uint32_t h = blockIdx.y;
  if (tok >= block_count || h >= num_q_heads)
    return;

  const uint32_t gqa_group = num_q_heads / num_kv_heads;
  const uint32_t kv_h = h / gqa_group;
  const uint32_t q_dim = num_q_heads * head_dim;
  const uint32_t kv_dim = num_kv_heads * head_dim;
  const uint32_t query_position = current_pos + tok;
  const uint32_t history_start = current_pos + 1U > sliding_window
                                     ? current_pos + 1U - sliding_window
                                     : 0U;
  const uint32_t history_count =
      history_length > history_start ? history_length - history_start : 0U;
  const uint32_t total_k = history_count + block_count;

  const float* q_ptr = q + ((uint64_t)tok * q_dim + (uint64_t)h * head_dim);
  float* out_ptr = out + ((uint64_t)tok * q_dim + (uint64_t)h * head_dim);

  extern __shared__ float s_scores[];

  for (uint32_t local = threadIdx.x; local < history_count;
       local += blockDim.x) {
    const uint32_t position = history_start + local;
    const float* k_ptr =
        injected_k + ((uint64_t)(position % history_capacity) * kv_dim +
                      (uint64_t)kv_h * head_dim);
    float dot = 0.0F;
    for (uint32_t d = 0; d < head_dim; ++d) {
      dot += q_ptr[d] * k_ptr[d];
    }
    s_scores[local] =
        query_position - position < sliding_window ? dot * scale : -1e20F;
  }

  for (uint32_t p = threadIdx.x; p < block_count; p += blockDim.x) {
    const float* k_ptr =
        block_k + ((uint64_t)p * kv_dim + (uint64_t)kv_h * head_dim);
    float dot = 0.0F;
    for (uint32_t d = 0; d < head_dim; ++d) {
      dot += q_ptr[d] * k_ptr[d];
    }
    const uint32_t key_position = current_pos + p;
    const uint32_t distance = query_position > key_position
                                  ? query_position - key_position
                                  : key_position - query_position;
    s_scores[history_count + p] =
        distance < sliding_window ? dot * scale : -1e20F;
  }
  __syncthreads();

  // Find max score
  __shared__ float s_reduce[256];
  float local_max = -1e20F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
    local_max = fmaxf(local_max, s_scores[p]);
  }
  s_reduce[threadIdx.x] = local_max;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_reduce[threadIdx.x] =
          fmaxf(s_reduce[threadIdx.x], s_reduce[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_val = s_reduce[0];
  __syncthreads();

  // Exponentiate & sum denominator
  float local_sum = 0.0F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
    const float exp_score = expf(s_scores[p] - max_val);
    s_scores[p] = exp_score;
    local_sum += exp_score;
  }
  s_reduce[threadIdx.x] = local_sum;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0F / (s_reduce[0] + 1e-9F);
  __syncthreads();

  // Normalize scores
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
    s_scores[p] *= inv_sum;
  }
  __syncthreads();

  // Multiply by V
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0F;
    for (uint32_t local = 0; local < history_count; ++local) {
      const uint32_t position = history_start + local;
      const float* v_ptr =
          injected_v + ((uint64_t)(position % history_capacity) * kv_dim +
                        (uint64_t)kv_h * head_dim);
      acc += s_scores[local] * v_ptr[d];
    }
    for (uint32_t p = 0; p < block_count; ++p) {
      const float* v_ptr =
          block_v + ((uint64_t)p * kv_dim + (uint64_t)kv_h * head_dim);
      acc += s_scores[history_count + p] * v_ptr[d];
    }
    out_ptr[d] = acc;
  }
}

constexpr uint32_t kSelectorThreads = 128;
// NBFAST T1: the Nanbeige DFlash2 drafter is trained with selector_top_k 32
// where the Qwen drafter uses 16. The selector device code below is templated
// on the list width and instantiated for both, so a top-k <= 16 drafter runs
// the original 16-wide kernels unchanged; only wider drafters pay for the
// 32-wide lists. kMaxSelectorTopK remains the host-side cap and scratch size.
constexpr uint32_t kMaxSelectorTopK = 32;
constexpr uint32_t kSelectorTokensPerBlock = 2048;

__device__ __forceinline__ void InsertSelectorCandidate(
    float score, uint32_t id, float* scores, uint32_t* ids,
    uint32_t selector_top_k) {
  if (score <= scores[selector_top_k - 1U]) {
    return;
  }
  uint32_t insertion = selector_top_k - 1U;
  while (insertion > 0U && score > scores[insertion - 1U]) {
    scores[insertion] = scores[insertion - 1U];
    ids[insertion] = ids[insertion - 1U];
    --insertion;
  }
  scores[insertion] = score;
  ids[insertion] = id;
}

/// Tree-merges the per-thread descending top-k lists held in shared memory into
/// slot zero. The previous implementation had thread zero fold all
/// kSelectorThreads lists serially, which dominated both selector kernels; a
/// pairwise merge does the same work in log2(threads) rounds. Ties keep the
/// lower slot, which is the same preference the serial fold had.
template <uint32_t MaxTopK>
__device__ void MergeSelectorLists(float* scores, uint32_t* ids,
                                   uint32_t selector_top_k) {
  for (uint32_t stride = kSelectorThreads / 2U; stride > 0U; stride >>= 1U) {
    __syncthreads();
    if (threadIdx.x < stride) {
      float* mine_scores = scores + (threadIdx.x * MaxTopK);
      uint32_t* mine_ids = ids + (threadIdx.x * MaxTopK);
      const float* other_scores =
          scores + ((threadIdx.x + stride) * MaxTopK);
      const uint32_t* other_ids =
          ids + ((threadIdx.x + stride) * MaxTopK);
      float merged_scores[MaxTopK];
      uint32_t merged_ids[MaxTopK];
      uint32_t mine = 0;
      uint32_t other = 0;
      for (uint32_t index = 0; index < selector_top_k; ++index) {
        const bool take_mine =
            other >= selector_top_k ||
            (mine < selector_top_k && mine_scores[mine] >= other_scores[other]);
        if (take_mine) {
          merged_scores[index] = mine_scores[mine];
          merged_ids[index] = mine_ids[mine];
          ++mine;
        } else {
          merged_scores[index] = other_scores[other];
          merged_ids[index] = other_ids[other];
          ++other;
        }
      }
      for (uint32_t index = 0; index < selector_top_k; ++index) {
        mine_scores[index] = merged_scores[index];
        mine_ids[index] = merged_ids[index];
      }
    }
  }
  __syncthreads();
}

template <uint32_t MaxTopK>
__global__ void dflash_selector_partial_topk_kernel(
    const float* __restrict__ logits, float* __restrict__ partial_scores,
    uint32_t* __restrict__ partial_ids, uint32_t vocab_size,
    uint32_t selector_top_k) {
  logits += uint64_t(blockIdx.y) * vocab_size;
  partial_scores += uint64_t(blockIdx.y) * gridDim.x * selector_top_k;
  partial_ids += uint64_t(blockIdx.y) * gridDim.x * selector_top_k;
  __shared__ float shared_scores[kSelectorThreads * MaxTopK];
  __shared__ uint32_t shared_ids[kSelectorThreads * MaxTopK];

  float local_scores[MaxTopK];
  uint32_t local_ids[MaxTopK];
#pragma unroll
  for (uint32_t index = 0; index < MaxTopK; ++index) {
    local_scores[index] = -INFINITY;
    local_ids[index] = 0;
  }

  const uint32_t begin = blockIdx.x * kSelectorTokensPerBlock;
  const uint32_t requested_end = begin + kSelectorTokensPerBlock;
  const uint32_t end = requested_end < vocab_size ? requested_end : vocab_size;
  for (uint32_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
    InsertSelectorCandidate(logits[token], token, local_scores, local_ids,
                            selector_top_k);
  }

  const uint32_t shared_offset = threadIdx.x * MaxTopK;
#pragma unroll
  for (uint32_t index = 0; index < MaxTopK; ++index) {
    shared_scores[shared_offset + index] = local_scores[index];
    shared_ids[shared_offset + index] = local_ids[index];
  }

  MergeSelectorLists<MaxTopK>(shared_scores, shared_ids, selector_top_k);

  const uint32_t output_offset = blockIdx.x * selector_top_k;
  for (uint32_t index = threadIdx.x; index < selector_top_k;
       index += blockDim.x) {
    partial_scores[output_offset + index] = shared_scores[index];
    partial_ids[output_offset + index] = shared_ids[index];
  }
}

template <uint32_t MaxTopK>
__device__ __forceinline__ void FinalizeSelector(
    const float* __restrict__ partial_scores,
    const uint32_t* __restrict__ partial_ids,
    const float* __restrict__ projected_hidden,
    const hip_bfloat16* __restrict__ predecessor_codebook,
    const hip_bfloat16* __restrict__ successor_codebook,
    const uint32_t* __restrict__ predecessor_token,
    uint32_t* __restrict__ output_token, float* __restrict__ output_confidence,
    float temperature, const float* __restrict__ sample_uniform,
    uint32_t* __restrict__ candidate_ids,
    float* __restrict__ candidate_probabilities, uint32_t partial_count,
    uint32_t selector_rank, uint32_t selector_top_k) {
  // Threads that cooperate on one candidate's low-rank transition score.
  constexpr uint32_t kTransitionLanes = kSelectorThreads / MaxTopK;
  __shared__ float shared_scores[kSelectorThreads * MaxTopK];
  __shared__ uint32_t shared_ids[kSelectorThreads * MaxTopK];
  __shared__ float
      shared_transition[MaxTopK * kTransitionLanes];

  float local_scores[MaxTopK];
  uint32_t local_ids[MaxTopK];
#pragma unroll
  for (uint32_t index = 0; index < MaxTopK; ++index) {
    local_scores[index] = -INFINITY;
    local_ids[index] = 0;
  }
  const uint32_t candidate_count = partial_count * selector_top_k;
  for (uint32_t index = threadIdx.x; index < candidate_count;
       index += blockDim.x) {
    InsertSelectorCandidate(partial_scores[index], partial_ids[index],
                            local_scores, local_ids, selector_top_k);
  }

  const uint32_t shared_offset = threadIdx.x * MaxTopK;
#pragma unroll
  for (uint32_t index = 0; index < MaxTopK; ++index) {
    shared_scores[shared_offset + index] = local_scores[index];
    shared_ids[shared_offset + index] = local_ids[index];
  }

  MergeSelectorLists<MaxTopK>(shared_scores, shared_ids, selector_top_k);

  const uint32_t predecessor = *predecessor_token;
  const hip_bfloat16* predecessor_row =
      predecessor_codebook + (uint64_t)predecessor * selector_rank;

  // Score the low-rank transitions in parallel: kTransitionLanes
  // threads share each candidate and reduce through shared memory. Thread zero
  // used to walk all selector_top_k x selector_rank products by itself.
  {
    const uint32_t candidate_index = threadIdx.x / kTransitionLanes;
    const uint32_t part = threadIdx.x % kTransitionLanes;
    if (candidate_index < selector_top_k) {
      const uint32_t candidate = shared_ids[candidate_index];
      const hip_bfloat16* successor_row =
          successor_codebook + (uint64_t)candidate * selector_rank;
      float partial = 0.0F;
      for (uint32_t rank = part; rank < selector_rank;
           rank += kTransitionLanes) {
        partial += static_cast<float>(predecessor_row[rank]) *
                   projected_hidden[rank] *
                   static_cast<float>(successor_row[rank]);
      }
      shared_transition[(candidate_index * kTransitionLanes) + part] =
          partial;
    }
  }
  __syncthreads();

  if (threadIdx.x != 0U) {
    return;
  }

  float top_scores[MaxTopK];
  uint32_t top_ids[MaxTopK];
  for (uint32_t index = 0; index < selector_top_k; ++index) {
    top_scores[index] = shared_scores[index];
    top_ids[index] = shared_ids[index];
  }

  float best_score = -INFINITY;
  uint32_t best_index = 0;
  float path_scores[MaxTopK];
  for (uint32_t candidate_index = 0; candidate_index < selector_top_k;
       ++candidate_index) {
    float transition = 0.0F;
    for (uint32_t part = 0; part < kTransitionLanes; ++part) {
      transition +=
          shared_transition[(candidate_index * kTransitionLanes) +
                            part];
    }
    const float score = top_scores[candidate_index] + transition;
    path_scores[candidate_index] = score;
    if (score > best_score) {
      best_score = score;
      best_index = candidate_index;
    }
  }

  if (candidate_ids != nullptr) {
    for (uint32_t index = 0; index < selector_top_k; ++index) {
      candidate_ids[index] = top_ids[index];
    }
  }

  if (temperature > 0.0F) {
    const float divisor = fmaxf(temperature, FLT_MIN);
    float denominator = 0.0F;
    for (uint32_t index = 0; index < selector_top_k; ++index) {
      // Subtract before division so a finite, very small temperature cannot
      // overflow both logits and their maximum to infinity.
      const float probability_weight =
          expf((path_scores[index] - best_score) / divisor);
      path_scores[index] = probability_weight;
      denominator += probability_weight;
    }
    const float inverse_denominator = 1.0F / fmaxf(denominator, 1e-30F);
    float threshold =
        fminf(fmaxf(*sample_uniform, 0.0F), 0.99999994F) * denominator;
    uint32_t sampled_index = selector_top_k - 1U;
    for (uint32_t index = 0; index < selector_top_k; ++index) {
      const float probability = path_scores[index] * inverse_denominator;
      if (candidate_probabilities != nullptr) {
        candidate_probabilities[index] = probability;
      }
      threshold -= path_scores[index];
      if (threshold < 0.0F && sampled_index == selector_top_k - 1U) {
        sampled_index = index;
      }
    }
    *output_token = top_ids[sampled_index];
    if (output_confidence != nullptr) {
      *output_confidence = path_scores[sampled_index] * inverse_denominator;
    }
  } else {
    *output_token = top_ids[best_index];
    if (candidate_probabilities != nullptr) {
      for (uint32_t index = 0; index < selector_top_k; ++index)
        candidate_probabilities[index] = index == best_index ? 1.0F : 0.0F;
    }
    if (output_confidence != nullptr) {
      *output_confidence = 1.0F;
    }
  }
}

template <uint32_t MaxTopK>
__global__ void dflash_selector_finalize_kernel(
    const float* partial_scores, const uint32_t* partial_ids,
    const float* projected_hidden, const hip_bfloat16* predecessor_codebook,
    const hip_bfloat16* successor_codebook, const uint32_t* predecessor_token,
    uint32_t* output_token, float* output_confidence, float temperature,
    const float* sample_uniform, uint32_t* candidate_ids,
    float* candidate_probabilities, uint32_t partial_count,
    uint32_t selector_rank, uint32_t selector_top_k) {
  FinalizeSelector<MaxTopK>(partial_scores, partial_ids, projected_hidden,
                   predecessor_codebook, successor_codebook, predecessor_token,
                   output_token, output_confidence, temperature, sample_uniform,
                   candidate_ids, candidate_probabilities, partial_count,
                   selector_rank, selector_top_k);
}

struct SelectorBatch {
  DFlashSelectorSequence sequences[8];
  uint32_t row_offsets[8];
  uint32_t token_offsets[8];
};

template <uint32_t MaxTopK>
__global__ void dflash_selector_finalize_batch_kernel(
    const float* partial_scores, const uint32_t* partial_ids,
    const float* projected_hidden, const hip_bfloat16* predecessor_codebook,
    const hip_bfloat16* successor_codebook, uint32_t* tokens,
    float* confidences, const float* uniforms, uint32_t* candidate_ids,
    float* candidate_probabilities, SelectorBatch batch, uint32_t partial_count,
    uint32_t selector_rank, uint32_t selector_top_k) {
  const auto sequence = batch.sequences[blockIdx.x];
  __shared__ uint32_t predecessor;
  if (threadIdx.x == 0U)
    predecessor = tokens[batch.token_offsets[blockIdx.x]];
  __syncthreads();
  for (uint32_t step = 0; step < sequence.count; ++step) {
    const uint32_t row = batch.row_offsets[blockIdx.x] + step;
    const uint32_t token = batch.token_offsets[blockIdx.x] + step;
    const uint64_t partial = uint64_t(row) * partial_count * selector_top_k;
    const uint64_t candidate = uint64_t(row) * selector_top_k;
    FinalizeSelector<MaxTopK>(
        partial_scores + partial, partial_ids + partial,
        projected_hidden + uint64_t(row) * selector_rank, predecessor_codebook,
        successor_codebook, &predecessor, tokens + token + 1U,
        confidences == nullptr ? nullptr : confidences + row,
        sequence.temperature,
        sequence.temperature > 0.0F ? uniforms + row : nullptr,
        candidate_ids != nullptr ? candidate_ids + candidate : nullptr,
        candidate_probabilities != nullptr ? candidate_probabilities + candidate
                                           : nullptr,
        partial_count, selector_rank, selector_top_k);
    // Thread zero publishes the predecessor for this sequence's next step.
    // Other sequences own separate token, probability and partial buffers.
    if (threadIdx.x == 0U)
      predecessor = tokens[token + 1U];
    __syncthreads();
  }
}

void LaunchDFlashGroupedDynamicConv(
    const float* input, const float* dynamic_coefficients,
    const float* base_kernel, float* output, std::uint32_t num_tokens,
    std::uint32_t hidden_size, std::uint32_t kernel_size,
    std::uint32_t group_size, std::uint32_t side, hipStream_t stream) {
  const uint64_t total = static_cast<uint64_t>(num_tokens) * hidden_size;
  const uint32_t blocks = static_cast<uint32_t>((total + 255U) / 256U);
  dflash_grouped_dynamic_conv_kernel<<<blocks, 256, 0, stream>>>(
      input, dynamic_coefficients, base_kernel, output, num_tokens, hidden_size,
      kernel_size, group_size, side);
  GUFO_CHECK_LAUNCH("dflash_grouped_dynamic_conv_kernel");
}

void LaunchDFlashSiLUMul(float* gate, const float* up,
                         std::uint32_t total_elements, hipStream_t stream) {
  const uint32_t blocks = (total_elements + 255u) / 256u;
  dflash_silu_mul_kernel<<<blocks, 256, 0, stream>>>(gate, up, total_elements);
  GUFO_CHECK_LAUNCH("dflash_silu_mul_kernel");
}

namespace {

/// Shared launch geometry for both non-causal attention routes: one workgroup
/// per (draft token, query head) and one shared score slot per attended key.
struct DFlashAttentionLaunch {
  dim3 grid;
  std::size_t shared_bytes;
};

[[nodiscard]] DFlashAttentionLaunch DFlashAttentionGeometry(
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads) noexcept {
  const std::uint32_t history_start = current_pos + 1U > sliding_window
                                          ? current_pos + 1U - sliding_window
                                          : 0U;
  const std::uint32_t history_count =
      history_length > history_start ? history_length - history_start : 0U;
  return {.grid = dim3(block_count, num_q_heads),
          .shared_bytes = (history_count + block_count) * sizeof(float)};
}

}  // namespace

void LaunchDFlashNonCausalAttentionScalar(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity) {
  const auto geometry = DFlashAttentionGeometry(
      current_pos, history_length, block_count, sliding_window, num_q_heads);
  // One score per attended key plus the 256-float reduction row. Shipped
  // drafters take the wave route; this one has no long-history form.
  if (geometry.shared_bytes + 256U * sizeof(float) > kDFlashMaxSharedBytes) {
    throw std::invalid_argument(
        "DFlash scalar attention: attended keys exceed workgroup LDS");
  }
  dflash_noncausal_attention_kernel<<<geometry.grid, 256, geometry.shared_bytes,
                                      stream>>>(
      q, injected_k, injected_v, block_k, block_v, out, current_pos,
      history_length, block_count, sliding_window, num_q_heads, num_kv_heads,
      head_dim, scale, history_capacity);
  GUFO_CHECK_LAUNCH("dflash_noncausal_attention_kernel");
}

void LaunchDFlashNonCausalAttentionWave(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity) {
  const auto geometry = DFlashAttentionGeometry(
      current_pos, history_length, block_count, sliding_window, num_q_heads);
  // Wider blocks benefit from two query heads per CTA. Keep narrow blocks
  // independent and leave enough LDS for both 256-thread reduction rows.
  const auto heads_per_kv = num_q_heads / num_kv_heads;
  if (block_count >= 4 && (heads_per_kv % 2U) == 0U &&
      geometry.shared_bytes <=
          (kDFlashMaxSharedBytes - 2U * 256U * sizeof(float)) / 2U) {
    auto grid = geometry.grid;
    grid.y /= 2U;
    dflash_noncausal_attention_wave_kernel<2>
        <<<grid, 256, 2U * geometry.shared_bytes, stream>>>(
            q, injected_k, injected_v, block_k, block_v, out, current_pos,
            history_length, block_count, sliding_window, num_q_heads,
            num_kv_heads, head_dim, scale, history_capacity);
    GUFO_CHECK_LAUNCH("dflash_noncausal_attention_wave_kernel<2>");
    return;
  }
  // A score row longer than LDS (about 16K keys) cannot launch whole: a drafter
  // without a sliding window attends the full prompt, so tile it.
  if (geometry.shared_bytes + 256U * sizeof(float) > kDFlashMaxSharedBytes) {
    LaunchDFlashNonCausalAttentionTiled(
        q, injected_k, injected_v, block_k, block_v, out, current_pos,
        history_length, block_count, sliding_window, num_q_heads, num_kv_heads,
        head_dim, scale, stream, history_capacity, kDFlashAttentionTileKeys);
    return;
  }
  dflash_noncausal_attention_wave_kernel<1>
      <<<geometry.grid, 256, geometry.shared_bytes, stream>>>(
          q, injected_k, injected_v, block_k, block_v, out, current_pos,
          history_length, block_count, sliding_window, num_q_heads,
          num_kv_heads, head_dim, scale, history_capacity);
  GUFO_CHECK_LAUNCH("dflash_noncausal_attention_wave_kernel<1>");
}

void LaunchDFlashNonCausalAttentionTiled(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity, std::uint32_t tile_keys) {
  constexpr std::size_t kThreads = 256;
  const bool pair_heads = block_count >= 4 && num_kv_heads != 0 &&
                          ((num_q_heads / num_kv_heads) % 2U) == 0U;
  const std::size_t heads = pair_heads ? 2U : 1U;
  if (tile_keys == 0 || (head_dim % 4U) != 0U ||
      head_dim > kDFlashTiledMaxDimsPerThread * kThreads ||
      heads * (tile_keys + kThreads) * sizeof(float) > kDFlashMaxSharedBytes) {
    throw std::invalid_argument("invalid DFlash tiled attention shape");
  }
  const std::size_t shared_bytes = heads * tile_keys * sizeof(float);
  if (pair_heads) {
    dflash_noncausal_attention_tiled_kernel<2>
        <<<dim3(block_count, num_q_heads / 2U), kThreads, shared_bytes,
           stream>>>(q, injected_k, injected_v, block_k, block_v, out,
                     current_pos, history_length, block_count, sliding_window,
                     num_q_heads, num_kv_heads, head_dim, scale,
                     history_capacity, tile_keys);
    GUFO_CHECK_LAUNCH("dflash_noncausal_attention_tiled_kernel<2>");
    return;
  }
  dflash_noncausal_attention_tiled_kernel<1>
      <<<dim3(block_count, num_q_heads), kThreads, shared_bytes, stream>>>(
          q, injected_k, injected_v, block_k, block_v, out, current_pos,
          history_length, block_count, sliding_window, num_q_heads,
          num_kv_heads, head_dim, scale, history_capacity, tile_keys);
  GUFO_CHECK_LAUNCH("dflash_noncausal_attention_tiled_kernel<1>");
}

std::size_t DFlashAttentionSplitScratchElements(
    std::uint32_t block_count, std::uint32_t num_q_heads,
    std::uint32_t head_dim) noexcept {
  return std::size_t{kDFlashSplitMaxSplits} * block_count * num_q_heads *
         (std::size_t{head_dim} + 2U);
}

namespace {

/// Shape of one split-route launch: how many key ranges and how long each is.
struct DFlashSplitPlan {
  std::uint32_t splits;
  std::uint32_t keys_per_split;
};

[[nodiscard]] DFlashSplitPlan DFlashSplitGeometry(
    std::uint32_t total_k, std::uint32_t keys_per_split) noexcept {
  // Production ranges are at least kDFlashSplitKeysPerSplit keys and at most
  // kDFlashSplitTargetSplits per row; a forced length is capped at
  // kDFlashSplitMaxSplits ranges. Past either, the ranges get longer.
  std::uint32_t keys = keys_per_split == 0 ? kDFlashSplitKeysPerSplit
                                           : keys_per_split;
  const std::uint32_t most =
      keys_per_split == 0 ? kDFlashSplitTargetSplits : kDFlashSplitMaxSplits;
  const std::uint32_t fewest = (total_k + most - 1U) / most;
  keys = std::max(keys, std::max(fewest, 1U));
  return {.splits = (total_k + keys - 1U) / keys, .keys_per_split = keys};
}

}  // namespace

void LaunchDFlashNonCausalAttentionSplit(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity, float* split_scratch,
    std::size_t split_scratch_elements, std::uint32_t keys_per_split) {
  if (split_scratch == nullptr || block_count == 0 || num_kv_heads == 0 ||
      (num_q_heads % num_kv_heads) != 0U ||
      ((num_q_heads / num_kv_heads) % kDFlashSplitQueryHeads) != 0U ||
      (head_dim % 4U) != 0U || head_dim > kDFlashSplitThreads ||
      head_dim > kDFlashSplitMaxVec * 4U * 32U) {
    throw std::invalid_argument("invalid DFlash split attention shape");
  }
  const std::uint32_t history_start = current_pos + 1U > sliding_window
                                          ? current_pos + 1U - sliding_window
                                          : 0U;
  const std::uint32_t history_count =
      history_length > history_start ? history_length - history_start : 0U;
  const auto plan =
      DFlashSplitGeometry(history_count + block_count, keys_per_split);
  const std::size_t rows = std::size_t{block_count} * num_q_heads;
  if (plan.splits * rows * (std::size_t{head_dim} + 2U) >
      split_scratch_elements) {
    throw std::invalid_argument("DFlash split attention scratch too small");
  }
  // One score row per query head of the workgroup, at most 32 KiB in all so
  // the launch stays well under the 64 KiB a workgroup can address.
  const std::uint32_t tile_keys =
      std::min(plan.keys_per_split,
               kDFlashSplitScoreFloats / kDFlashSplitQueryHeads);
  const std::size_t shared_bytes =
      std::size_t{kDFlashSplitQueryHeads} * tile_keys * sizeof(float);
  float* partial_acc = split_scratch;
  float* partial_ml = split_scratch + plan.splits * rows * head_dim;
  dflash_noncausal_attention_split_kernel<<<
      dim3(block_count, num_q_heads / kDFlashSplitQueryHeads, plan.splits),
      kDFlashSplitThreads, shared_bytes, stream>>>(
      q, injected_k, injected_v, block_k, block_v, partial_acc, partial_ml,
      current_pos, history_length, block_count, sliding_window, num_q_heads,
      num_kv_heads, head_dim, scale, history_capacity, plan.keys_per_split,
      tile_keys);
  GUFO_CHECK_LAUNCH("dflash_noncausal_attention_split_kernel");
  dflash_noncausal_attention_split_combine_kernel<<<
      dim3(block_count, num_q_heads), kDFlashSplitThreads, 0, stream>>>(
      partial_acc, partial_ml, out, block_count, num_q_heads, head_dim,
      plan.splits);
  GUFO_CHECK_LAUNCH("dflash_noncausal_attention_split_combine_kernel");
}

void LaunchDFlashNonCausalAttention(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity, float* split_scratch,
    std::size_t split_scratch_elements) {
  // Long histories spread their keys across the GPU; shorter ones keep the
  // routes below unchanged.
  if (split_scratch != nullptr && (head_dim % 4U) == 0U &&
      head_dim <= kDFlashSplitThreads && num_kv_heads != 0 &&
      (num_q_heads % num_kv_heads) == 0U &&
      ((num_q_heads / num_kv_heads) % kDFlashSplitQueryHeads) == 0U) {
    const std::uint32_t history_start =
        current_pos + 1U > sliding_window ? current_pos + 1U - sliding_window
                                          : 0U;
    const std::uint32_t history_count =
        history_length > history_start ? history_length - history_start : 0U;
    if (history_count + block_count >= kDFlashSplitMinKeys &&
        DFlashSplitGeometry(history_count + block_count, 0).splits *
                std::size_t{block_count} * num_q_heads *
                (std::size_t{head_dim} + 2U) <=
            split_scratch_elements) {
      LaunchDFlashNonCausalAttentionSplit(
          q, injected_k, injected_v, block_k, block_v, out, current_pos,
          history_length, block_count, sliding_window, num_q_heads,
          num_kv_heads, head_dim, scale, stream, history_capacity,
          split_scratch, split_scratch_elements, 0);
      return;
    }
  }
  // Coalesced float4 reads cover all shipped DFlash2 head dimensions.
  if ((head_dim % 4U) == 0U) {
    LaunchDFlashNonCausalAttentionWave(
        q, injected_k, injected_v, block_k, block_v, out, current_pos,
        history_length, block_count, sliding_window, num_q_heads, num_kv_heads,
        head_dim, scale, stream, history_capacity);
    return;
  }
  LaunchDFlashNonCausalAttentionScalar(
      q, injected_k, injected_v, block_k, block_v, out, current_pos,
      history_length, block_count, sliding_window, num_q_heads, num_kv_heads,
      head_dim, scale, stream, history_capacity);
}

std::size_t DFlashSelectorScratchElements(std::uint32_t vocab_size) noexcept {
  const std::size_t partial_count =
      (static_cast<std::size_t>(vocab_size) + kSelectorTokensPerBlock - 1U) /
      kSelectorTokensPerBlock;
  return partial_count * kMaxSelectorTopK;
}

void LaunchDFlashSelectorStep(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    const std::uint32_t* predecessor_token, std::uint32_t* output_token,
    float* output_confidence, float* partial_scores, std::uint32_t* partial_ids,
    float temperature, const float* sample_uniform,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::uint32_t vocab_size, std::uint32_t selector_rank,
    std::uint32_t selector_top_k, hipStream_t stream) {
  if (partial_scores == nullptr || partial_ids == nullptr || vocab_size == 0 ||
      selector_top_k == 0 || selector_top_k > kMaxSelectorTopK ||
      selector_top_k > vocab_size) {
    throw std::invalid_argument("invalid DFlash selector shape or scratch");
  }
  const std::uint32_t partial_count =
      (vocab_size + kSelectorTokensPerBlock - 1U) / kSelectorTokensPerBlock;
  if (selector_top_k <= 16U) {
    dflash_selector_partial_topk_kernel<16U><<<partial_count, kSelectorThreads,
                                               0, stream>>>(
        logits, partial_scores, partial_ids, vocab_size, selector_top_k);
  } else {
    dflash_selector_partial_topk_kernel<32U><<<partial_count, kSelectorThreads,
                                               0, stream>>>(
        logits, partial_scores, partial_ids, vocab_size, selector_top_k);
  }
  GUFO_CHECK_LAUNCH("dflash_selector_partial_topk_kernel");
  if (selector_top_k <= 16U) {
    dflash_selector_finalize_kernel<16U><<<1, kSelectorThreads, 0, stream>>>(
      partial_scores, partial_ids, projected_hidden,
      static_cast<const hip_bfloat16*>(predecessor_codebook_bf16),
      static_cast<const hip_bfloat16*>(successor_codebook_bf16),
      predecessor_token, output_token, output_confidence, temperature,
      sample_uniform, candidate_ids, candidate_probabilities, partial_count,
      selector_rank, selector_top_k);
  } else {
    dflash_selector_finalize_kernel<32U><<<1, kSelectorThreads, 0, stream>>>(
        partial_scores, partial_ids, projected_hidden,
        static_cast<const hip_bfloat16*>(predecessor_codebook_bf16),
        static_cast<const hip_bfloat16*>(successor_codebook_bf16),
        predecessor_token, output_token, output_confidence, temperature,
        sample_uniform, candidate_ids, candidate_probabilities, partial_count,
        selector_rank, selector_top_k);
  }
  GUFO_CHECK_LAUNCH("dflash_selector_finalize_kernel");
}

void LaunchDFlashSelectorBatch(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    std::uint32_t* tokens, float* confidences, float* partial_scores,
    std::uint32_t* partial_ids, const float* uniforms,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::span<const DFlashSelectorSequence> sequences, std::uint32_t vocab_size,
    std::uint32_t selector_rank, std::uint32_t selector_top_k,
    hipStream_t stream) {
  if (sequences.empty() || sequences.size() > 8 || logits == nullptr ||
      projected_hidden == nullptr || predecessor_codebook_bf16 == nullptr ||
      successor_codebook_bf16 == nullptr || tokens == nullptr ||
      partial_scores == nullptr || partial_ids == nullptr || vocab_size == 0 ||
      selector_rank == 0 || selector_top_k == 0 ||
      selector_top_k > kMaxSelectorTopK || selector_top_k > vocab_size)
    throw std::invalid_argument("invalid DFlash selector batch or scratch");
  SelectorBatch batch{};
  std::uint32_t rows = 0;
  for (std::size_t index = 0; index < sequences.size(); ++index) {
    const auto sequence = sequences[index];
    if (sequence.count == 0 ||
        sequence.count >
            std::numeric_limits<std::uint32_t>::max() - 8U - rows ||
        !std::isfinite(sequence.temperature) || sequence.temperature < 0.0F ||
        (sequence.temperature > 0.0F && uniforms == nullptr))
      throw std::invalid_argument("invalid DFlash selector sequence");
    batch.sequences[index] = sequence;
    batch.row_offsets[index] = rows;
    batch.token_offsets[index] = rows + static_cast<std::uint32_t>(index);
    rows += sequence.count;
  }
  const auto partial_count = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(vocab_size) + kSelectorTokensPerBlock - 1U) /
      kSelectorTokensPerBlock);
  if (selector_top_k <= 16U) {
    dflash_selector_partial_topk_kernel<16U><<<dim3(partial_count, rows),
                                               kSelectorThreads, 0, stream>>>(
        logits, partial_scores, partial_ids, vocab_size, selector_top_k);
  } else {
    dflash_selector_partial_topk_kernel<32U><<<dim3(partial_count, rows),
                                               kSelectorThreads, 0, stream>>>(
        logits, partial_scores, partial_ids, vocab_size, selector_top_k);
  }
  GUFO_CHECK_LAUNCH("dflash_selector_partial_topk_kernel");
  if (selector_top_k <= 16U) {
    dflash_selector_finalize_batch_kernel<16U><<<sequences.size(),
                                                 kSelectorThreads, 0, stream>>>(
        partial_scores, partial_ids, projected_hidden,
        static_cast<const hip_bfloat16*>(predecessor_codebook_bf16),
        static_cast<const hip_bfloat16*>(successor_codebook_bf16), tokens,
        confidences, uniforms, candidate_ids, candidate_probabilities, batch,
        partial_count, selector_rank, selector_top_k);
  } else {
    dflash_selector_finalize_batch_kernel<32U><<<sequences.size(),
                                                 kSelectorThreads, 0, stream>>>(
        partial_scores, partial_ids, projected_hidden,
        static_cast<const hip_bfloat16*>(predecessor_codebook_bf16),
        static_cast<const hip_bfloat16*>(successor_codebook_bf16), tokens,
        confidences, uniforms, candidate_ids, candidate_probabilities, batch,
        partial_count, selector_rank, selector_top_k);
  }
  GUFO_CHECK_LAUNCH("dflash_selector_finalize_batch_kernel");
}

}  // namespace gufo::hip::kernels
#endif  // defined(ENGINE_ENABLE_HIP)
