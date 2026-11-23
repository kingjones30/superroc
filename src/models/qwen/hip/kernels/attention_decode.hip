#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/models/qwen/hip/detail/kv_f16_store.hpp"  // QKS

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <cstdio>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/core/hip_launch_check.hpp"
#include "src/core/env.hpp"

namespace superroc::hip {

__global__ void WriteKVCacheKernel(
    const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    std::uint32_t layer_idx, std::uint32_t pos, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::uint32_t kv_h = blockIdx.x;
  if (kv_h >= num_kv_heads)
    return;
  const std::size_t tid = threadIdx.x;
  if (tid >= head_dim)
    return;

  const std::size_t kv_offset =
      (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
            max_context +
        pos) *
       head_dim);
  const float key = k[static_cast<std::size_t>(kv_h) * head_dim + tid];
  const float val = v[static_cast<std::size_t>(kv_h) * head_dim + tid];
  if (k_cache != nullptr) {
    k_cache[kv_offset + tid] = key;
  }
  if (v_cache != nullptr) {
    v_cache[kv_offset + tid] = val;
  }

  if (k_cache_f16 != nullptr || v_cache_f16 != nullptr) {
    const std::size_t kv_width =
        static_cast<std::size_t>(num_kv_heads) * head_dim;
    const std::size_t f16_offset =
        ((static_cast<std::size_t>(layer_idx) * max_context + pos) * kv_width) +
        (static_cast<std::size_t>(kv_h) * head_dim) + tid;
    if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_offset] = QwenKvF16(key);
    }
    if (v_cache_f16 != nullptr) {
      v_cache_f16[f16_offset] = QwenKvF16(val);
    }
  }
}

__device__ __forceinline__ float4 LoadHalf4(const __half* source) {
  const auto* pairs = reinterpret_cast<const __half2*>(source);
  const float2 lower = __half22float2(pairs[0]);
  const float2 upper = __half22float2(pairs[1]);
  return make_float4(lower.x, lower.y, upper.x, upper.y);
}

__global__ void AttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx, std::uint32_t pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads)
    return;

  const std::uint32_t group_size =
      (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const std::uint32_t kv_h = h / group_size;
  const std::size_t tid = threadIdx.x;

  const float* q_h = q + (static_cast<std::size_t>(h) * head_dim);
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;

  extern __shared__ float s_scores[];
  __shared__ float s_scratch[256];

  // Step 1: Compute scores for each position p = 0..seq_len-1
  for (std::size_t p = 0; p < seq_len; ++p) {
    const std::size_t p_offset =
        (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
              max_context +
          p) *
         head_dim);
    const float* k_p = k_cache + p_offset;

    float dot = 0.0F;
    for (std::size_t d = tid; d < head_dim; d += blockDim.x) {
      dot += q_h[d] * k_p[d];
    }
    s_scratch[tid] = dot;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
        s_scratch[tid] += s_scratch[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      s_scores[p] = s_scratch[0] * scale;
    }
    __syncthreads();
  }

  // Step 2: Softmax across s_scores[0..seq_len-1]
  float local_max = -1e30F;
  for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
    local_max = fmaxf(local_max, s_scores[p]);
  }
  s_scratch[tid] = local_max;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_scratch[tid] = fmaxf(s_scratch[tid], s_scratch[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = s_scratch[0];
  __syncthreads();

  float local_sum = 0.0F;
  for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
    const float e = expf(s_scores[p] - max_score);
    s_scores[p] = e;
    local_sum += e;
  }
  s_scratch[tid] = local_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_scratch[tid] += s_scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum_exp = 1.0F / (s_scratch[0] + 1e-9F);
  __syncthreads();

  for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
    s_scores[p] *= inv_sum_exp;
  }
  __syncthreads();

  // Step 3: Weighted sum of V
  if (tid < head_dim) {
    float ctx = 0.0F;
    for (std::size_t p = 0; p < seq_len; ++p) {
      const std::size_t p_offset =
          (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
                max_context +
            p) *
           head_dim);
      const float* v_p = v_cache + p_offset;
      ctx += s_scores[p] * v_p[tid];
    }

    if (gate != nullptr) {
      const float g = gate[static_cast<std::size_t>(h) * head_dim + tid];
      const float sig = 1.0F / (1.0F + expf(-g));
      ctx *= sig;
    }
    out_context[static_cast<std::size_t>(h) * head_dim + tid] = ctx;
  }
}

__global__ void AttentionHalfKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx, std::uint32_t pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::uint32_t head = blockIdx.x;
  if (head >= num_heads) {
    return;
  }

  const std::uint32_t group_size =
      (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const std::uint32_t kv_head = head / group_size;
  const std::size_t tid = threadIdx.x;
  const float* query = q + (static_cast<std::size_t>(head) * head_dim);
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;

  extern __shared__ float scores[];
  __shared__ float scratch[256];

  for (std::size_t position = 0; position < sequence_length; ++position) {
    const std::size_t cache_offset =
        ((static_cast<std::size_t>(layer_idx) * max_context + position) *
         kv_width) +
        (static_cast<std::size_t>(kv_head) * head_dim);
    const __half* key = k_cache + cache_offset;
    float dot = 0.0F;
    for (std::size_t dim = tid; dim < head_dim; dim += blockDim.x) {
      dot += query[dim] * __half2float(key[dim]);
    }
    scratch[tid] = dot;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
        scratch[tid] += scratch[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      scores[position] = scratch[0] * scale;
    }
    __syncthreads();
  }

  float local_max = -INFINITY;
  for (std::size_t position = tid; position < sequence_length;
       position += blockDim.x) {
    local_max = fmaxf(local_max, scores[position]);
  }
  scratch[tid] = local_max;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = scratch[0];

  float local_sum = 0.0F;
  for (std::size_t position = tid; position < sequence_length;
       position += blockDim.x) {
    const float exponential = expf(scores[position] - max_score);
    scores[position] = exponential;
    local_sum += exponential;
  }
  scratch[tid] = local_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inverse_sum = 1.0F / (scratch[0] + 1e-9F);
  for (std::size_t position = tid; position < sequence_length;
       position += blockDim.x) {
    scores[position] *= inverse_sum;
  }
  __syncthreads();

  if (tid < head_dim) {
    float context = 0.0F;
    for (std::size_t position = 0; position < sequence_length; ++position) {
      const std::size_t cache_offset =
          ((static_cast<std::size_t>(layer_idx) * max_context + position) *
           kv_width) +
          (static_cast<std::size_t>(kv_head) * head_dim);
      context += scores[position] * __half2float(v_cache[cache_offset + tid]);
    }
    if (gate != nullptr) {
      const float gate_value =
          gate[(static_cast<std::size_t>(head) * head_dim) + tid];
      context *= 1.0F / (1.0F + expf(-gate_value));
    }
    out_context[(static_cast<std::size_t>(head) * head_dim) + tid] = context;
  }
}

__launch_bounds__(32) __global__ void QwenDecodeOnlineAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx, std::uint32_t pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads) {
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / 32;

  const std::size_t row_offset =
      static_cast<std::size_t>(blockIdx.y) * num_heads * head_dim;
  q += row_offset;
  if (gate != nullptr) {
    gate += row_offset;
  }
  out_context += row_offset;
  pos += blockIdx.y;

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || lane >= 32) {
    return;
  }

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * head_dim));
  const std::size_t cache_head_offset =
      ((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
       max_context * head_dim);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float4 query_values[vectors_per_lane];
  float4 output_values[vectors_per_lane] = {};
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    query_values[vector] = query[lane + (vector * 32)];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  for (std::size_t position = 0; position < sequence_length; ++position) {
    const auto* key = reinterpret_cast<const float4*>(
        k_cache + cache_head_offset + (position * head_dim));
    float score = 0.0F;
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 query_value = query_values[vector];
      const float4 key_value = key[lane + (vector * 32)];
      score += (query_value.x * key_value.x) + (query_value.y * key_value.y) +
               (query_value.z * key_value.z) + (query_value.w * key_value.w);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      score += __shfl_down(score, offset);
    }
    score = __shfl(score, 0) * scale;

    const float new_max = fmaxf(running_max, score);
    const float previous_scale = expf(running_max - new_max);
    const float value_scale = expf(score - new_max);
    running_sum = (running_sum * previous_scale) + value_scale;
    running_max = new_max;

    const auto* value = reinterpret_cast<const float4*>(
        v_cache + cache_head_offset + (position * head_dim));
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 value_data = value[lane + (vector * 32)];
      float4& output = output_values[vector];
      output.x = (output.x * previous_scale) + (value_data.x * value_scale);
      output.y = (output.y * previous_scale) + (value_data.y * value_scale);
      output.z = (output.z * previous_scale) + (value_data.z * value_scale);
      output.w = (output.w * previous_scale) + (value_data.w * value_scale);
    }
  }

  auto* output = reinterpret_cast<float4*>(
      out_context + (static_cast<std::size_t>(head) * head_dim));
  const auto* gate_values =
      gate == nullptr ? nullptr
                      : reinterpret_cast<const float4*>(
                            gate + (static_cast<std::size_t>(head) * head_dim));
  const float inverse_sum = 1.0F / running_sum;
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    float4 result = output_values[vector];
    result.x *= inverse_sum;
    result.y *= inverse_sum;
    result.z *= inverse_sum;
    result.w *= inverse_sum;
    if (gate_values != nullptr) {
      const float4 gate_value = gate_values[lane + (vector * 32)];
      result.x *= 1.0F / (1.0F + expf(-gate_value.x));
      result.y *= 1.0F / (1.0F + expf(-gate_value.y));
      result.z *= 1.0F / (1.0F + expf(-gate_value.z));
      result.w *= 1.0F / (1.0F + expf(-gate_value.w));
    }
    output[lane + (vector * 32)] = result;
  }
}

__launch_bounds__(32) __global__ void QwenDecodeOnlineAttentionHalfKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx, std::uint32_t pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads) {
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / 32;

  const std::size_t row_offset =
      static_cast<std::size_t>(blockIdx.y) * num_heads * head_dim;
  q += row_offset;
  if (gate != nullptr) {
    gate += row_offset;
  }
  out_context += row_offset;
  pos += blockIdx.y;

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || lane >= 32) {
    return;
  }

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * head_dim));
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t cache_head_offset =
      (static_cast<std::size_t>(layer_idx) * max_context * kv_width) +
      (static_cast<std::size_t>(kv_head) * head_dim);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float4 query_values[vectors_per_lane];
  float4 output_values[vectors_per_lane] = {};
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    query_values[vector] = query[lane + (vector * 32)];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  // Load neighboring keys and values together, then retain the scalar
  // position order for every softmax and output update.
  constexpr std::size_t positions_per_load = 2;
  for (std::size_t position = 0; position < sequence_length;
       position += positions_per_load) {
    float scores[positions_per_load]{};
    float4 values[positions_per_load][vectors_per_lane];
#pragma unroll
    for (std::size_t step = 0; step < positions_per_load; ++step) {
      if (position + step >= sequence_length)
        break;
      const auto offset = cache_head_offset + ((position + step) * kv_width);
      const __half* key = k_cache + offset;
      const __half* value = v_cache + offset;
      float score = 0.0F;
#pragma unroll
      for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
        const float4 query_value = query_values[vector];
        const auto element = (lane + (vector * 32)) * 4;
        const float4 key_value = LoadHalf4(key + element);
        values[step][vector] = LoadHalf4(value + element);
        score += (query_value.x * key_value.x) + (query_value.y * key_value.y) +
                 (query_value.z * key_value.z) + (query_value.w * key_value.w);
      }
      // Lane zero keeps the descending reduction tree. Inside each
      // 16-lane row, DPP avoids the LDS path used by shuffle instructions.
      score += __shfl_down(score, 16);
      score += __builtin_bit_cast(
          float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, score),
                                             0x168, 0xF, 0xF, false));
      score += __builtin_bit_cast(
          float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, score),
                                             0x164, 0xF, 0xF, false));
      score += __builtin_bit_cast(
          float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, score),
                                             0x162, 0xF, 0xF, false));
      score += __builtin_bit_cast(
          float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, score),
                                             0x161, 0xF, 0xF, false));
      scores[step] = __shfl(score, 0) * scale;
    }

#pragma unroll
    for (std::size_t step = 0; step < positions_per_load; ++step) {
      if (position + step >= sequence_length)
        break;
      const float new_max = fmaxf(running_max, scores[step]);
      const float previous_scale = expf(running_max - new_max);
      const float value_scale = expf(scores[step] - new_max);
      running_sum = (running_sum * previous_scale) + value_scale;
      running_max = new_max;

#pragma unroll
      for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
        const float4 value_data = values[step][vector];
        float4& output = output_values[vector];
        // Retain the original half-pair kernel's rounded product and FMA.
        output.x = __fmaf_rn(output.x, previous_scale,
                             __fmul_rn(value_data.x, value_scale));
        output.y = __fmaf_rn(output.y, previous_scale,
                             __fmul_rn(value_data.y, value_scale));
        output.z = __fmaf_rn(value_data.z, value_scale,
                             __fmul_rn(output.z, previous_scale));
        output.w = __fmaf_rn(value_data.w, value_scale,
                             __fmul_rn(output.w, previous_scale));
      }
    }
  }

  auto* output = reinterpret_cast<float4*>(
      out_context + (static_cast<std::size_t>(head) * head_dim));
  const auto* gate_values =
      gate == nullptr ? nullptr
                      : reinterpret_cast<const float4*>(
                            gate + (static_cast<std::size_t>(head) * head_dim));
  const float inverse_sum = 1.0F / running_sum;
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    float4 result = output_values[vector];
    result.x *= inverse_sum;
    result.y *= inverse_sum;
    result.z *= inverse_sum;
    result.w *= inverse_sum;
    if (gate_values != nullptr) {
      const float4 gate_value = gate_values[lane + (vector * 32)];
      result.x *= 1.0F / (1.0F + expf(-gate_value.x));
      result.y *= 1.0F / (1.0F + expf(-gate_value.y));
      result.z *= 1.0F / (1.0F + expf(-gate_value.z));
      result.w *= 1.0F / (1.0F + expf(-gate_value.w));
    }
    output[lane + (vector * 32)] = result;
  }
}

__launch_bounds__(32) __global__ void QwenDecodeSplitKAttentionPartialsKernel(
    const float* __restrict__ q, const float* __restrict__ k_cache,
    const float* __restrict__ v_cache, float* __restrict__ scratch,
    std::uint32_t layer_idx, std::uint32_t pos, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads,
    std::uint32_t split_count) {
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / 32;

  const std::size_t row = blockIdx.y;
  q += row * num_heads * head_dim;
  scratch += row * num_heads * detail::kSplitKDecodeAttentionMaxSplits *
             (head_dim + 2);
  pos += blockIdx.y;

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t split = blockIdx.z;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || split >= split_count || lane >= 32) {
    return;
  }

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * head_dim));
  const std::size_t cache_head_offset =
      ((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
       max_context * head_dim);
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const std::size_t partition_length =
      (sequence_length + split_count - 1) / split_count;
  const std::size_t partition_begin =
      static_cast<std::size_t>(split) * partition_length;
  const std::size_t partition_end =
      min(partition_begin + partition_length, sequence_length);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float4 query_values[vectors_per_lane];
  float4 output_values[vectors_per_lane] = {};
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    query_values[vector] = query[lane + (vector * 32)];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0F;
  for (std::size_t position = partition_begin; position < partition_end;
       ++position) {
    const auto* key = reinterpret_cast<const float4*>(
        k_cache + cache_head_offset + (position * head_dim));
    float score = 0.0F;
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 query_value = query_values[vector];
      const float4 key_value = key[lane + (vector * 32)];
      score += (query_value.x * key_value.x) + (query_value.y * key_value.y) +
               (query_value.z * key_value.z) + (query_value.w * key_value.w);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      score += __shfl_down(score, offset);
    }
    score = __shfl(score, 0) * scale;

    const float new_max = fmaxf(running_max, score);
    const float previous_scale = expf(running_max - new_max);
    const float value_scale = expf(score - new_max);
    running_sum = (running_sum * previous_scale) + value_scale;
    running_max = new_max;

    const auto* value = reinterpret_cast<const float4*>(
        v_cache + cache_head_offset + (position * head_dim));
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 value_data = value[lane + (vector * 32)];
      float4& output = output_values[vector];
      output.x = (output.x * previous_scale) + (value_data.x * value_scale);
      output.y = (output.y * previous_scale) + (value_data.y * value_scale);
      output.z = (output.z * previous_scale) + (value_data.z * value_scale);
      output.w = (output.w * previous_scale) + (value_data.w * value_scale);
    }
  }

  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  const std::size_t partition_index =
      (static_cast<std::size_t>(head) * max_splits) + split;
  if (lane == 0) {
    scratch[(partition_index * 2) + 0] = running_max;
    scratch[(partition_index * 2) + 1] = running_sum;
  }
  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
  auto* partial_outputs = reinterpret_cast<float4*>(scratch + stats_elements);
  auto* partial_output = partial_outputs + (partition_index * vector_width);
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    partial_output[lane + (vector * 32)] = output_values[vector];
  }
}

template<bool ShareKv = false, unsigned Lanes = 32, unsigned kGroup = 6,
         unsigned kMaxSplits = 32>
__launch_bounds__(ShareKv ? kGroup * Lanes : 32) __global__
    void QwenDecodeSplitKAttentionHalfPartialsKernel(
        const float* __restrict__ q, const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache, float* __restrict__ scratch,
        std::uint32_t layer_idx, std::uint32_t pos, std::uint32_t max_context,
        std::uint32_t num_heads, std::uint32_t num_kv_heads,
        std::uint32_t split_count) {
  static_assert(Lanes == 32 || (ShareKv && Lanes == 16));
  static_assert(!ShareKv || kGroup == 6 || kGroup == 8);
  static_assert(kMaxSplits == 32 || kMaxSplits == detail::kKatGqa8MaxSplits);
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / Lanes;

  const std::size_t row = blockIdx.y;
  q += row * num_heads * head_dim;
  scratch += row * num_heads * kMaxSplits * (head_dim + 2);
  pos += blockIdx.y;

  constexpr std::uint32_t heads_per_block = ShareKv ? kGroup : 1;
  const std::uint32_t head = blockIdx.x * heads_per_block + threadIdx.x / Lanes;
  const std::uint32_t split = blockIdx.z;
  const std::uint32_t lane = threadIdx.x % Lanes;
  if (head >= num_heads || split >= split_count) {
    return;
  }

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * head_dim));
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t cache_head_offset =
      (static_cast<std::size_t>(layer_idx) * max_context * kv_width) +
      (static_cast<std::size_t>(kv_head) * head_dim);
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const std::size_t partition_length =
      (sequence_length + split_count - 1) / split_count;
  const std::size_t partition_begin =
      static_cast<std::size_t>(split) * partition_length;
  const std::size_t partition_end =
      min(partition_begin + partition_length, sequence_length);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float4 query_values[vectors_per_lane];
  float4 output_values[vectors_per_lane] = {};
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    query_values[vector] =
        query[lane + (vector / 2) * Lanes + (vector % 2) * 32];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const auto accumulate = [&](const __half* key, const __half* value) {
    float score = 0.0F;
    if constexpr (Lanes == 32) {
#pragma unroll
      for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
        const float4 query_value = query_values[vector];
        const float4 key_value = LoadHalf4(key + ((lane + (vector * 32)) * 4));
        score += (query_value.x * key_value.x) + (query_value.y * key_value.y) +
                 (query_value.z * key_value.z) + (query_value.w * key_value.w);
      }
    } else {
      // Two query heads share a hardware wave. Keep both original lane
      // partials independent until the original offset-16 reduction stage.
      float partial[2] = {};
#pragma unroll
      for (unsigned half = 0; half < 2; ++half) {
#pragma unroll
        for (unsigned vector = 0; vector < 2; ++vector) {
          const float4 query_value = query_values[half * 2 + vector];
          const float4 key_value =
              LoadHalf4(key + ((lane + half * 16 + vector * 32) * 4));
          partial[half] +=
              (query_value.x * key_value.x) + (query_value.y * key_value.y) +
              (query_value.z * key_value.z) + (query_value.w * key_value.w);
        }
      }
      score = partial[0] + partial[1];
    }
#pragma unroll
    for (int offset = Lanes / 2; offset > 0; offset /= 2) {
      score += __shfl_down(score, offset, Lanes);
    }
    score = __shfl(score, 0, Lanes) * scale;

    const float new_max = fmaxf(running_max, score);
    const float previous_scale = expf(running_max - new_max);
    const float value_scale = expf(score - new_max);
    running_sum = (running_sum * previous_scale) + value_scale;
    running_max = new_max;

#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 value_data = LoadHalf4(
          value + ((lane + (vector / 2) * Lanes + (vector % 2) * 32) * 4));
      float4& output = output_values[vector];
      // Preserve the scalar kernel's rounded product and FMA operands.
      output.x =
          fmaf(output.x, previous_scale, __fmul_rn(value_data.x, value_scale));
      output.y =
          fmaf(output.y, previous_scale, __fmul_rn(value_data.y, value_scale));
      output.z =
          fmaf(value_data.z, value_scale, __fmul_rn(output.z, previous_scale));
      output.w =
          fmaf(value_data.w, value_scale, __fmul_rn(output.w, previous_scale));
    }
  };
  if constexpr (ShareKv) {
    // Query heads of one KV head share the tile (kGroup 6 is 27B 24/4,
    // kGroup 8 is KAT 16/2). Stage the encoded KV once without changing
    // any head's score, softmax or value accumulation order.
    // kGroup 8 is a 256-thread decode block. A tile of 8 tokens is one
    // uint4 per thread, so the cooperative load has no idle lanes. kGroup 6
    // stays at 4 and is the 27B kernel.
    constexpr unsigned tile = kGroup == 8 ? 8u : 4u;
    __shared__ __align__(16) __half keys[tile][head_dim];
    __shared__ __align__(16) __half values[tile][head_dim];
    for (std::size_t begin = partition_begin; begin < partition_end;
         begin += tile) {
      const unsigned count =
          static_cast<unsigned>(min(std::size_t{tile}, partition_end - begin));
      for (unsigned vector = threadIdx.x; vector < count * 32;
           vector += heads_per_block * Lanes) {
        const unsigned token = vector / 32;
        const unsigned dim = (vector % 32) * 8;
        const std::size_t offset =
            cache_head_offset + (begin + token) * kv_width + dim;
        *reinterpret_cast<uint4*>(keys[token] + dim) =
            *reinterpret_cast<const uint4*>(k_cache + offset);
        *reinterpret_cast<uint4*>(values[token] + dim) =
            *reinterpret_cast<const uint4*>(v_cache + offset);
      }
      __syncthreads();
#pragma unroll 1
      for (unsigned token = 0; token < count; ++token) {
        accumulate(keys[token], values[token]);
      }
      __syncthreads();
    }
  } else {
    for (std::size_t position = partition_begin; position < partition_end;
         ++position) {
      const std::size_t offset = cache_head_offset + position * kv_width;
      accumulate(k_cache + offset, v_cache + offset);
    }
  }

  constexpr std::size_t max_splits = kMaxSplits;
  const std::size_t partition_index =
      (static_cast<std::size_t>(head) * max_splits) + split;
  if (lane == 0) {
    scratch[(partition_index * 2) + 0] = running_max;
    scratch[(partition_index * 2) + 1] = running_sum;
  }
  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
  auto* partial_outputs = reinterpret_cast<float4*>(scratch + stats_elements);
  auto* partial_output = partial_outputs + (partition_index * vector_width);
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    partial_output[lane + (vector / 2) * Lanes + (vector % 2) * 32] =
        output_values[vector];
  }
}

template<unsigned kMaxSplits = 32>
__launch_bounds__(32) __global__ void QwenDecodeSplitKAttentionReduceKernel(
    const float* __restrict__ gate, const float* __restrict__ scratch,
    float* __restrict__ out_context, std::uint32_t num_heads,
    std::uint32_t split_count) {
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / 32;
  constexpr std::size_t max_splits = kMaxSplits;
  static_assert(kMaxSplits == 32 || kMaxSplits == detail::kKatGqa8MaxSplits);

  const std::size_t row = blockIdx.y;
  if (gate != nullptr)
    gate += row * num_heads * head_dim;
  out_context += row * num_heads * head_dim;
  scratch += row * num_heads * max_splits * (head_dim + 2);

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || lane >= 32) {
    return;
  }

  const std::size_t head_partition_offset =
      static_cast<std::size_t>(head) * max_splits;
  float global_max = -INFINITY;
  for (std::uint32_t split = 0; split < split_count; ++split) {
    global_max =
        fmaxf(global_max, scratch[((head_partition_offset + split) * 2) + 0]);
  }

  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
  const auto* partial_outputs =
      reinterpret_cast<const float4*>(scratch + stats_elements);
  auto* output = reinterpret_cast<float4*>(
      out_context + (static_cast<std::size_t>(head) * head_dim));
  const auto* gate_values =
      gate == nullptr ? nullptr
                      : reinterpret_cast<const float4*>(
                            gate + (static_cast<std::size_t>(head) * head_dim));
  // Each output vector and the denominator visit partitions in the same
  // order. Share their scale instead of recomputing its exponential.
  float global_sum = 0.0F;
  float4 results[vectors_per_lane]{};
  for (std::uint32_t split = 0; split < split_count; ++split) {
    const std::size_t partition_index = head_partition_offset + split;
    const float partition_scale =
        expf(scratch[partition_index * 2] - global_max);
    global_sum += scratch[partition_index * 2 + 1] * partition_scale;
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 partial = partial_outputs[(partition_index * vector_width) +
                                             lane + (vector * 32)];
      auto& result = results[vector];
      result.x += partial.x * partition_scale;
      result.y += partial.y * partition_scale;
      result.z += partial.z * partition_scale;
      result.w += partial.w * partition_scale;
    }
  }
  const float inverse_sum = 1.0F / global_sum;

#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    auto result = results[vector];
    result.x *= inverse_sum;
    result.y *= inverse_sum;
    result.z *= inverse_sum;
    result.w *= inverse_sum;
    if (gate_values != nullptr) {
      const float4 gate_value = gate_values[lane + (vector * 32)];
      result.x *= 1.0F / (1.0F + expf(-gate_value.x));
      result.y *= 1.0F / (1.0F + expf(-gate_value.y));
      result.z *= 1.0F / (1.0F + expf(-gate_value.z));
      result.w *= 1.0F / (1.0F + expf(-gate_value.w));
    }
    output[lane + (vector * 32)] = result;
  }
}

// Verification rows of one KV head and split share one block, so the KV partition is read
// once for every row instead of once per row. Each row keeps its own causal endpoint, query,
// softmax state and output, and each value matches the per-row kernel bit for bit. The softmax
// inputs of a token are uniform across a wave, so lane r carries row r through one exp.
__device__ __forceinline__ __attribute__((convergent)) void SharedKvSync() {
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_s_barrier();
  asm volatile("" ::: "memory");
}

// The partner of a lane across one lane bit, for the score trees below.
template<unsigned kBit>
__device__ __forceinline__ float SharedKvXor(float x) {
  union Bits {
    float f;
    unsigned u;
  };
  Bits a;
  Bits b;
  a.f = x;
  if constexpr (kBit == 16) {
    b.u = __builtin_amdgcn_permlanex16(0u, a.u, 0x76543210u, 0xfedcba98u, false, false);
  } else if constexpr (kBit == 8) {
    b.u = __builtin_amdgcn_permlane16(0u, a.u, 0xfedcba98u, 0x76543210u, false, false);
  } else if constexpr (kBit == 4) {
    b.u = __builtin_amdgcn_permlane16(0u, a.u, 0x32107654u, 0xba98fedcu, false, false);
  } else if constexpr (kBit == 2) {
    b.u = __builtin_amdgcn_permlane16(0u, a.u, 0x54761032u, 0xdcfe98bau, false, false);
  } else {
    static_assert(kBit == 1);
    b.u = __builtin_amdgcn_permlane16(0u, a.u, 0x67452301u, 0xefcdab89u, false, false);
  }
  return b.f;
}

// One pass over lane bit kBit for N rows: a lane keeps the half of the rows its bit
// selects and adds its partner's partial of the same rows.
template<unsigned kBit, unsigned N>
__device__ __forceinline__ void SharedKvHalve(float (&w)[8], unsigned lane) {
  const bool upper = (lane & kBit) != 0;
#pragma unroll
  for (unsigned k = 0; k < N / 2; ++k) {
    const float give = upper ? w[k] : w[k + N / 2];
    const float keep = upper ? w[k + N / 2] : w[k];
    w[k] = keep + SharedKvXor<kBit>(give);
  }
}

// The scores of P rows of one token, reduced together. Each row's partials pair over lane
// bits 16, 8, 4, 2 and 1 in that order, the tree a single score takes, so every score
// keeps its bits; the halving passes only move whole rows between lanes, about five times
// fewer cross-lane steps than one tree per row. Row r ends on lanes r * (32 / P) onward.
template<unsigned P>
__device__ __forceinline__ float SharedKvScoresBatched(float (&w)[8], unsigned lane) {
  static_assert(P == 2 || P == 4 || P == 8);
  SharedKvHalve<16, P>(w, lane);
  if constexpr (P >= 4) SharedKvHalve<8, P / 2>(w, lane);
  if constexpr (P >= 8) SharedKvHalve<4, P / 4>(w, lane);
  float s = w[0];
  if constexpr (P < 4) s = s + SharedKvXor<8>(s);
  if constexpr (P < 8) s = s + SharedKvXor<4>(s);
  s = s + SharedKvXor<2>(s);
  return s + SharedKvXor<1>(s);
}

template<unsigned kRows, unsigned kGroup, unsigned kMaxSplits>
__launch_bounds__(kGroup * 32) __global__
    void QwenVerifySharedKvPartialsKernel(
        const float* __restrict__ q, const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache, float* __restrict__ scratch,
        std::uint32_t layer_idx, std::uint32_t start_pos,
        std::uint32_t max_context, std::uint32_t num_heads,
        std::uint32_t num_kv_heads, std::uint32_t split_count) {
#pragma clang fp contract(off)
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t tile = kGroup;  // one load per thread for a full tile
  // Rows rounded up for the score tree, and the lanes a row's state lives on.
  constexpr unsigned kRowPow = kRows <= 2 ? 2 : (kRows <= 4 ? 4 : 8);
  constexpr unsigned kRowLanes = 32 / kRowPow;
  union Bits {
    float f;
    unsigned u;
  };
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t head = kv_head * kGroup + threadIdx.x / 32;
  const std::uint32_t split = blockIdx.y;
  const std::uint32_t lane = threadIdx.x % 32;
  if (head >= num_heads || split >= split_count) {
    return;
  }
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t cache_head_offset =
      (static_cast<std::size_t>(layer_idx) * max_context * kv_width) +
      (static_cast<std::size_t>(kv_head) * head_dim);
  std::uint32_t row_begin[kRows];
  std::uint32_t row_end[kRows];
  std::uint32_t walk = 0xffffffffu;
  std::uint32_t walk_end = 0;
#pragma unroll
  for (unsigned r = 0; r < kRows; ++r) {
    const std::uint32_t sequence_length = start_pos + r + 1;
    const std::uint32_t partition_length =
        (sequence_length + split_count - 1) / split_count;
    row_begin[r] = split * partition_length;
    row_end[r] = min(row_begin[r] + partition_length, sequence_length);
    if (row_begin[r] < row_end[r]) {
      walk = min(walk, row_begin[r]);
      walk_end = max(walk_end, row_end[r]);
    }
  }
  const float scale = rsqrtf(static_cast<float>(head_dim));
  float negative_zero = -0.0F;
  asm volatile("" : "+v"(negative_zero));

  float4 query_values[kRows][2];
  float4 output_values[kRows][2] = {};
#pragma unroll
  for (unsigned r = 0; r < kRows; ++r) {
    const auto* query = reinterpret_cast<const float4*>(
        q + ((static_cast<std::size_t>(r) * num_heads + head) * head_dim));
    query_values[r][0] = query[lane];
    query_values[r][1] = query[lane + 32];
  }
  // Lane r holds row r's running maximum and sum.
  float running_max = -INFINITY;
  float running_sum = 0.0F;

  const auto dot = [&](const float4& query, __half2 lower, __half2 upper) {
    float d = fmaf(query.y, __high2float(lower), negative_zero);
    d = fmaf(query.x, __low2float(lower), d);
    d = fmaf(query.z, __low2float(upper), d);
    return fmaf(query.w, __high2float(upper), d);
  };
  const auto update = [&](float4& output, __half2 lower, __half2 upper,
                          float previous_scale, float value_scale) {
    output.x = fmaf(output.x, previous_scale,
                    fmaf(value_scale, __low2float(lower), negative_zero));
    output.y = fmaf(output.y, previous_scale,
                    fmaf(value_scale, __high2float(lower), negative_zero));
    output.z = fmaf(__low2float(upper), value_scale,
                    __fmul_rn(output.z, previous_scale));
    output.w = fmaf(__high2float(upper), value_scale,
                    __fmul_rn(output.w, previous_scale));
  };

  __shared__ __align__(16) __half keys[tile][head_dim];
  __shared__ __align__(16) __half values[tile][head_dim];
  while (walk < walk_end) {
    // The rows live at `walk`, and where that set next changes.
    std::uint32_t live = 0;
    std::uint32_t stop = walk_end;
#pragma unroll
    for (unsigned r = 0; r < kRows; ++r) {
      if (walk >= row_begin[r] && walk < row_end[r]) {
        live |= 1u << r;
      }
      if (row_begin[r] > walk) {
        stop = min(stop, row_begin[r]);
      }
      if (row_end[r] > walk) {
        stop = min(stop, row_end[r]);
      }
    }
    if (live == 0) {
      walk = stop;
      continue;
    }
    const unsigned lane_row = lane / kRowLanes;
    const bool lane_live = lane % kRowLanes == 0 && lane_row < kRows &&
                           ((live >> lane_row) & 1u) != 0;
    while (walk < stop) {
      const unsigned count = min(tile, stop - walk);
      for (unsigned vector = threadIdx.x; vector < count * 32;
           vector += kGroup * 32) {
        const unsigned token = vector / 32;
        const unsigned dim = (vector % 32) * 8;
        const std::size_t offset = cache_head_offset +
                                   (static_cast<std::size_t>(walk) + token) * kv_width + dim;
        *reinterpret_cast<uint4*>(keys[token] + dim) =
            *reinterpret_cast<const uint4*>(k_cache + offset);
        *reinterpret_cast<uint4*>(values[token] + dim) =
            *reinterpret_cast<const uint4*>(v_cache + offset);
      }
      SharedKvSync();
#pragma unroll 1
      for (unsigned token = 0; token < count; ++token) {
        const auto* kp = reinterpret_cast<const __half2*>(keys[token]);
        const __half2 k0 = kp[lane * 2];
        const __half2 k1 = kp[lane * 2 + 1];
        const __half2 k2 = kp[(lane + 32) * 2];
        const __half2 k3 = kp[(lane + 32) * 2 + 1];
        float partials[8];
#pragma unroll
        for (unsigned r = 0; r < kRowPow; ++r) {
          float partial = 0.0F;
          if (r < kRows && ((live >> r) & 1u)) {
            const unsigned q = r < kRows ? r : 0;
            partial = 0.0F + dot(query_values[q][0], k0, k1);
            partial = partial + dot(query_values[q][1], k2, k3);
          }
          partials[r] = partial;
        }
        const float score = SharedKvScoresBatched<kRowPow>(partials, lane) * scale;
        const float new_max = fmaxf(running_max, score);
        const float previous_scale = expf(running_max - new_max);
        const float value_scale = expf(score - new_max);
        const float new_sum = fmaf(running_sum, previous_scale, value_scale);
        running_max = lane_live ? new_max : running_max;
        running_sum = lane_live ? new_sum : running_sum;
        const auto* vp = reinterpret_cast<const __half2*>(values[token]);
        const __half2 v0 = vp[lane * 2];
        const __half2 v1 = vp[lane * 2 + 1];
        const __half2 v2 = vp[(lane + 32) * 2];
        const __half2 v3 = vp[(lane + 32) * 2 + 1];
        Bits pb;
        Bits vb;
        pb.f = previous_scale;
        vb.f = value_scale;
#pragma unroll
        for (unsigned r = 0; r < kRows; ++r) {
          if ((live >> r) & 1u) {
            Bits ps;
            Bits vs;
            ps.u = __builtin_amdgcn_readlane(pb.u, r * kRowLanes);
            vs.u = __builtin_amdgcn_readlane(vb.u, r * kRowLanes);
            update(output_values[r][0], v0, v1, ps.f, vs.f);
            update(output_values[r][1], v2, v3, ps.f, vs.f);
          }
        }
      }
      SharedKvSync();
      walk += count;
    }
  }

  const std::size_t partition_index =
      (static_cast<std::size_t>(head) * kMaxSplits) + split;
  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * kMaxSplits * 2;
#pragma unroll
  for (unsigned r = 0; r < kRows; ++r) {
    float* row_scratch =
        scratch + (static_cast<std::size_t>(r) * num_heads * kMaxSplits * (head_dim + 2));
    if (lane == r * kRowLanes) {
      row_scratch[(partition_index * 2) + 0] = running_max;
      row_scratch[(partition_index * 2) + 1] = running_sum;
    }
    auto* partial_output = reinterpret_cast<float4*>(row_scratch + stats_elements) +
                           (partition_index * (head_dim / 4));
    partial_output[lane] = output_values[r][0];
    partial_output[lane + 32] = output_values[r][1];
  }
}

template<unsigned kGroup, unsigned kMaxSplits>
static void LaunchSharedKvPartials(const float* q, const __half* k_cache,
                                   const __half* v_cache, float* scratch,
                                   std::uint32_t layer_idx, std::uint32_t start_pos,
                                   unsigned rows, std::uint32_t max_context,
                                   std::uint32_t num_heads, std::uint32_t num_kv_heads,
                                   std::uint32_t splits, hipStream_t stream) {
  const dim3 grid(num_kv_heads, splits);
#define SUPERROC_SHARED_KV_LAUNCH(R)                                                    \
  QwenVerifySharedKvPartialsKernel<R, kGroup, kMaxSplits>                           \
      <<<grid, kGroup * 32, 0, stream>>>(q, k_cache, v_cache, scratch, layer_idx,   \
                                         start_pos, max_context, num_heads,         \
                                         num_kv_heads, splits);                     \
  SUPERROC_CHECK_LAUNCH("QwenVerifySharedKvPartialsKernel");
  switch (rows) {
    case 2: SUPERROC_SHARED_KV_LAUNCH(2) break;
    case 3: SUPERROC_SHARED_KV_LAUNCH(3) break;
    case 4: SUPERROC_SHARED_KV_LAUNCH(4) break;
    case 5: SUPERROC_SHARED_KV_LAUNCH(5) break;
    case 6: SUPERROC_SHARED_KV_LAUNCH(6) break;
    case 7: SUPERROC_SHARED_KV_LAUNCH(7) break;
    default: SUPERROC_SHARED_KV_LAUNCH(8) break;
  }
#undef SUPERROC_SHARED_KV_LAUNCH
}

// A verify longer than one block holds splits its rows into groups of at most eight, each
// sharing every KV tile, so sixteen rows read the cache twice instead of sixteen times. A row
// keeps its own causal endpoint and scratch slot, so its bits are those of the per-row form.
template<unsigned kGroup, unsigned kMaxSplits>
static void LaunchSharedKvRowGroups(const float* q, const __half* k_cache,
                                    const __half* v_cache, float* scratch,
                                    std::uint32_t layer_idx, std::uint32_t start_pos,
                                    unsigned rows, std::uint32_t max_context,
                                    std::uint32_t num_heads, std::uint32_t num_kv_heads,
                                    std::uint32_t splits, hipStream_t stream) {
  const unsigned groups = (rows + 7) / 8;
  unsigned first = 0;
  for (unsigned group = 0; group < groups; ++group) {
    const unsigned count = rows / groups + (group < rows % groups ? 1U : 0U);
    LaunchSharedKvPartials<kGroup, kMaxSplits>(
        q + static_cast<std::size_t>(first) * num_heads * 256, k_cache, v_cache,
        scratch + static_cast<std::size_t>(first) * num_heads * kMaxSplits * (256 + 2),
        layer_idx, start_pos + first, count, max_context, num_heads, num_kv_heads, splits,
        stream);
    first += count;
  }
}

// SUPERROC_ATTN_SHARED_CHECK=1 runs both forms on the same inputs and compares every bit.
static bool SharedKvCheckOn() {
  static const bool on = ::superroc::core::GetEnv("SUPERROC_ATTN_SHARED_CHECK") != nullptr;
  return on;
}

template<typename OldLaunch, typename NewLaunch>
static void SharedKvCompare(OldLaunch old_launch, NewLaunch new_launch, float* scratch,
                            std::size_t elements, unsigned rows, std::uint32_t start_pos,
                            hipStream_t stream) {
  static unsigned long long calls = 0;
  static unsigned long long bad = 0;
  std::vector<std::uint32_t> a(elements);
  std::vector<std::uint32_t> b(elements);
  old_launch();
  hipStreamSynchronize(stream);
  hipMemcpy(a.data(), scratch, elements * sizeof(std::uint32_t), hipMemcpyDeviceToHost);
  new_launch();
  hipStreamSynchronize(stream);
  hipMemcpy(b.data(), scratch, elements * sizeof(std::uint32_t), hipMemcpyDeviceToHost);
  ++calls;
  for (std::size_t i = 0; i < elements; ++i) {
    if (a[i] != b[i]) {
      ++bad;
      fprintf(stderr, "ATTNCHECK differ rows=%u pos=%u element=%zu old=%08x new=%08x\n", rows,
              start_pos, i, a[i], b[i]);
      break;
    }
  }
  if (calls % 200 == 0) {
    fprintf(stderr, "ATTNCHECK calls=%llu bad=%llu pos=%u rows=%u\n", calls, bad, start_pos, rows);
  }
}

static void LaunchSplitKAttentionHalfPartials(
    const float* q, const __half* k_cache, const __half* v_cache,
    float* scratch, std::uint32_t layer_idx, std::uint32_t start_pos,
    unsigned rows, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t splits, hipStream_t stream) {
  // Share each KV tile across its six query heads in both scalar decoding
  // and verification. Rows retain independent causal endpoints and scratch.
  if (num_heads == 6 * num_kv_heads &&
      static_cast<std::size_t>(start_pos) + 1 >= 512) {
    if (rows > 1) {
      const auto per_row = [&] {
        QwenDecodeSplitKAttentionHalfPartialsKernel<true, 16>
            <<<dim3(num_kv_heads, rows, splits), 96, 0, stream>>>(
                q, k_cache, v_cache, scratch, layer_idx, start_pos, max_context,
                num_heads, num_kv_heads, splits);
        SUPERROC_CHECK_LAUNCH("QwenDecodeSplitKAttentionHalfPartialsKernel<true, 16>");
      };
      const auto shared = [&] {
        LaunchSharedKvRowGroups<6, 32>(q, k_cache, v_cache, scratch, layer_idx, start_pos,
                                       rows, max_context, num_heads, num_kv_heads, splits,
                                       stream);
      };
      if (SharedKvCheckOn()) {
        SharedKvCompare(per_row, shared, scratch,
                        static_cast<std::size_t>(rows) * num_heads * 32 * (256 + 2), rows,
                        start_pos, stream);
      } else {
        shared();
      }
    } else {
      QwenDecodeSplitKAttentionHalfPartialsKernel<true>
          <<<dim3(num_kv_heads, rows, splits), 192, 0, stream>>>(
              q, k_cache, v_cache, scratch, layer_idx, start_pos, max_context,
              num_heads, num_kv_heads, splits);
      SUPERROC_CHECK_LAUNCH("QwenDecodeSplitKAttentionHalfPartialsKernel<true>");
    }
  } else if (num_heads == 16 && num_kv_heads == 2 &&
             static_cast<std::size_t>(start_pos) + 1 >= 512) {
    // KAT 16/2/256 only. 27B 24/4 stays on the branch above.
    if (rows > 1) {
      const auto per_row = [&] {
        QwenDecodeSplitKAttentionHalfPartialsKernel<true, 16, 8,
                                                    detail::kKatGqa8MaxSplits>
            <<<dim3(num_kv_heads, rows, splits), 128, 0, stream>>>(
                q, k_cache, v_cache, scratch, layer_idx, start_pos, max_context,
                num_heads, num_kv_heads, splits);
        SUPERROC_CHECK_LAUNCH("QwenDecodeSplitKAttentionHalfPartialsKernel<true, 16, 8, detail::kKatGqa8MaxSplits>");
      };
      const auto shared = [&] {
        LaunchSharedKvPartials<8, detail::kKatGqa8MaxSplits>(
            q, k_cache, v_cache, scratch, layer_idx, start_pos, rows, max_context, num_heads,
            num_kv_heads, splits, stream);
      };
      if (rows > 8) {
        per_row();
      } else if (SharedKvCheckOn()) {
        SharedKvCompare(per_row, shared, scratch,
                        static_cast<std::size_t>(rows) * num_heads *
                            detail::kKatGqa8MaxSplits * (256 + 2),
                        rows, start_pos, stream);
      } else {
        shared();
      }
    } else {
      QwenDecodeSplitKAttentionHalfPartialsKernel<true, 32, 8,
                                                    detail::kKatGqa8MaxSplits>
          <<<dim3(num_kv_heads, rows, splits), 256, 0, stream>>>(
              q, k_cache, v_cache, scratch, layer_idx, start_pos, max_context,
              num_heads, num_kv_heads, splits);
      SUPERROC_CHECK_LAUNCH("QwenDecodeSplitKAttentionHalfPartialsKernel<true, 32, 8, detail::kKatGqa8MaxSplits>");
    }
  } else {
    QwenDecodeSplitKAttentionHalfPartialsKernel<>
        <<<dim3(num_heads, rows, splits), 32, 0, stream>>>(
            q, k_cache, v_cache, scratch, layer_idx, start_pos, max_context,
            num_heads, num_kv_heads, splits);
    SUPERROC_CHECK_LAUNCH("QwenDecodeSplitKAttentionHalfPartialsKernel<>");
  }
}

void LaunchAttention(const float* q, const float* k, const float* v,
                     const float* gate, float* k_cache, float* v_cache,
                     void* k_cache_f16, void* v_cache_f16, float* out_context,
                     std::uint32_t layer_idx, std::uint32_t pos,
                     std::uint32_t max_context, std::uint32_t num_heads,
                     std::uint32_t num_kv_heads, std::uint32_t head_dim,
                     hipStream_t stream, float* split_k_scratch,
                     bool skip_kv_write) {
  // 1. Write K and V into every non-null selected KV representation across all
  // num_kv_heads. May be skipped when a fused kernel already wrote the cache
  // (opt-c010-qk-rope-kv).
  if (!skip_kv_write) {
    WriteKVCacheKernel<<<num_kv_heads, 256, 0, stream>>>(
        k, v, k_cache, v_cache, static_cast<__half*>(k_cache_f16),
        static_cast<__half*>(v_cache_f16), layer_idx, pos, max_context,
        num_kv_heads, head_dim);
    SUPERROC_CHECK_LAUNCH("WriteKVCacheKernel");
  }

  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const bool split_k_supported = detail::IsSplitKDecodeAttentionSupported(
      sequence_length, num_heads, num_kv_heads, head_dim);
  // Same predicate as the kGroup-8 launch. Its scratch stride is
  // kKatGqa8MaxSplits even when the split count is still 32.
  const bool kat_gqa8_long = k_cache == nullptr && num_heads == 16 &&
                             num_kv_heads == 2 && sequence_length >= 512;
  if (split_k_supported && split_k_scratch != nullptr) {
    const std::uint32_t split_count =
        kat_gqa8_long
            ? detail::SelectKatGqa8SplitCount(sequence_length)
            : detail::SelectDecodeAttentionSplitCount(sequence_length);
    const dim3 partial_grid(num_heads, 1, split_count);
    if (k_cache != nullptr && v_cache != nullptr) {
      detail::EmitAttentionDispatch("decode_split_k_fp32", "");
      QwenDecodeSplitKAttentionPartialsKernel<<<partial_grid, 32, 0, stream>>>(
          q, k_cache, v_cache, split_k_scratch, layer_idx, pos, max_context,
          num_heads, num_kv_heads, split_count);
      SUPERROC_CHECK_LAUNCH("QwenDecodeSplitKAttentionPartialsKernel");
    } else {
      detail::EmitAttentionDispatch("decode_split_k_fp16", "");
      LaunchSplitKAttentionHalfPartials(
          q, static_cast<const __half*>(k_cache_f16),
          static_cast<const __half*>(v_cache_f16), split_k_scratch, layer_idx,
          pos, 1, max_context, num_heads, num_kv_heads, split_count, stream);
    }
    if (kat_gqa8_long) {
      QwenDecodeSplitKAttentionReduceKernel<detail::kKatGqa8MaxSplits>
          <<<num_heads, 32, 0, stream>>>(
              gate, split_k_scratch, out_context, num_heads, split_count);
      SUPERROC_CHECK_LAUNCH("QwenDecodeSplitKAttentionReduceKernel<detail::kKatGqa8MaxSplits>");
    } else {
      QwenDecodeSplitKAttentionReduceKernel<<<num_heads, 32, 0, stream>>>(
          gate, split_k_scratch, out_context, num_heads, split_count);
      SUPERROC_CHECK_LAUNCH("QwenDecodeSplitKAttentionReduceKernel");
    }
  } else if (head_dim == 256 && num_kv_heads > 0 &&
             (num_heads % num_kv_heads) == 0) {
    if (k_cache != nullptr && v_cache != nullptr) {
      detail::EmitAttentionDispatch(
          "decode_online_fp32", split_k_supported
                                    ? "decode_split_k_fp32: missing_scratch"
                                    : "decode_split_k_fp32: below_threshold");
      QwenDecodeOnlineAttentionKernel<<<num_heads, 32, 0, stream>>>(
          q, gate, k_cache, v_cache, out_context, layer_idx, pos, max_context,
          num_heads, num_kv_heads);
      SUPERROC_CHECK_LAUNCH("QwenDecodeOnlineAttentionKernel");
    } else {
      detail::EmitAttentionDispatch(
          "decode_online_fp16", split_k_supported
                                    ? "decode_split_k_fp16: missing_scratch"
                                    : "decode_split_k_fp16: below_threshold");
      QwenDecodeOnlineAttentionHalfKernel<<<num_heads, 32, 0, stream>>>(
          q, gate, static_cast<const __half*>(k_cache_f16),
          static_cast<const __half*>(v_cache_f16), out_context, layer_idx, pos,
          max_context, num_heads, num_kv_heads);
      SUPERROC_CHECK_LAUNCH("QwenDecodeOnlineAttentionHalfKernel");
    }
  } else {
    const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;
    const std::size_t shared_scores_bytes = seq_len * sizeof(float);
    if (k_cache != nullptr && v_cache != nullptr) {
      detail::EmitAttentionDispatch(
          "decode_baseline_fp32",
          "decode_split_k_fp32: unsupported_shape; decode_online_fp32: "
          "unsupported_shape");
      AttentionKernel<<<num_heads, 256, shared_scores_bytes, stream>>>(
          q, gate, k_cache, v_cache, out_context, layer_idx, pos, max_context,
          num_heads, num_kv_heads, head_dim);
      SUPERROC_CHECK_LAUNCH("AttentionKernel");
    } else {
      detail::EmitAttentionDispatch(
          "decode_baseline_fp16",
          "decode_split_k_fp16: unsupported_shape; decode_online_fp16: "
          "unsupported_shape");
      AttentionHalfKernel<<<num_heads, 256, shared_scores_bytes, stream>>>(
          q, gate, static_cast<const __half*>(k_cache_f16),
          static_cast<const __half*>(v_cache_f16), out_context, layer_idx, pos,
          max_context, num_heads, num_kv_heads, head_dim);
      SUPERROC_CHECK_LAUNCH("AttentionHalfKernel");
    }
  }
}

void LaunchCausalDecodeAttention(
    const float* q, const float* gate, float* k_cache, float* v_cache,
    void* k_cache_f16, void* v_cache_f16, float* out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, hipStream_t stream,
    std::span<float> scratch) {
  const std::size_t row_scratch =
      detail::DecodeAttentionScratchElements(num_heads, head_dim);
  float* const split_k_scratch =
      scratch.size() >= row_scratch ? scratch.data() : nullptr;
  if (batch_size > 1 && scratch.size() / batch_size >= row_scratch &&
      split_k_scratch != nullptr &&
      detail::IsSplitKDecodeAttentionSupported(
          static_cast<std::size_t>(start_pos) + 1, num_heads, num_kv_heads,
          head_dim)) {
    const std::size_t sequence_length = static_cast<std::size_t>(start_pos) + 1;
    const bool kat_gqa8_long = k_cache == nullptr && num_heads == 16 &&
                               num_kv_heads == 2 && sequence_length >= 512;
    const auto splits =
        kat_gqa8_long ? detail::SelectKatGqa8SplitCount(sequence_length)
                      : detail::SelectDecodeAttentionSplitCount(sequence_length);
    // Keep a partition's KV resident while visiting every query row. With the
    // row as the slow grid dimension, each row rereads the entire long cache.
    const dim3 grid(num_heads, static_cast<unsigned>(batch_size), splits);
    if (k_cache != nullptr && v_cache != nullptr) {
      QwenDecodeSplitKAttentionPartialsKernel<<<grid, 32, 0, stream>>>(
          q, k_cache, v_cache, split_k_scratch, layer_idx, start_pos,
          max_context, num_heads, num_kv_heads, splits);
      SUPERROC_CHECK_LAUNCH("QwenDecodeSplitKAttentionPartialsKernel");
    } else {
      LaunchSplitKAttentionHalfPartials(
          q, static_cast<const __half*>(k_cache_f16),
          static_cast<const __half*>(v_cache_f16), split_k_scratch, layer_idx,
          start_pos, static_cast<unsigned>(batch_size), max_context, num_heads,
          num_kv_heads, splits, stream);
    }
    if (kat_gqa8_long) {
      QwenDecodeSplitKAttentionReduceKernel<detail::kKatGqa8MaxSplits><<<
          dim3(num_heads, static_cast<unsigned>(batch_size)), 32, 0, stream>>>(
          gate, split_k_scratch, out_context, num_heads, splits);
      SUPERROC_CHECK_LAUNCH("QwenDecodeSplitKAttentionReduceKernel<detail::kKatGqa8MaxSplits>");
    } else {
      QwenDecodeSplitKAttentionReduceKernel<<<
          dim3(num_heads, static_cast<unsigned>(batch_size)), 32, 0, stream>>>(
          gate, split_k_scratch, out_context, num_heads, splits);
      SUPERROC_CHECK_LAUNCH("QwenDecodeSplitKAttentionReduceKernel");
    }
    return;
  }
  // Each query keeps the scalar decode reduction and its own causal endpoint.
  // Crossing the split-K threshold retains the per-row route and scratch.
  if (batch_size > 1 && head_dim == 256 && num_kv_heads > 0 &&
      num_heads % num_kv_heads == 0 &&
      static_cast<std::size_t>(start_pos) + batch_size <
          detail::kSplitKDecodeAttentionMinContext) {
    const dim3 grid(num_heads, static_cast<unsigned>(batch_size));
    if (k_cache != nullptr && v_cache != nullptr) {
      QwenDecodeOnlineAttentionKernel<<<grid, 32, 0, stream>>>(
          q, gate, k_cache, v_cache, out_context, layer_idx, start_pos,
          max_context, num_heads, num_kv_heads);
      SUPERROC_CHECK_LAUNCH("QwenDecodeOnlineAttentionKernel");
    } else {
      QwenDecodeOnlineAttentionHalfKernel<<<grid, 32, 0, stream>>>(
          q, gate, static_cast<const __half*>(k_cache_f16),
          static_cast<const __half*>(v_cache_f16), out_context, layer_idx,
          start_pos, max_context, num_heads, num_kv_heads);
      SUPERROC_CHECK_LAUNCH("QwenDecodeOnlineAttentionHalfKernel");
    }
    return;
  }
  const std::size_t width = static_cast<std::size_t>(num_heads) * head_dim;
  for (std::size_t row = 0; row < batch_size; ++row) {
    LaunchAttention(
        q + row * width, nullptr, nullptr,
        gate == nullptr ? nullptr : gate + row * width, k_cache, v_cache,
        k_cache_f16, v_cache_f16, out_context + row * width, layer_idx,
        start_pos + static_cast<std::uint32_t>(row), max_context, num_heads,
        num_kv_heads, head_dim, stream, split_k_scratch, true);
  }
}

}  // namespace superroc::hip
#endif  // defined(ENGINE_ENABLE_HIP)
