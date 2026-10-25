#if defined(ENGINE_ENABLE_HIP)
#include "src/models/glimmer/hip/glimmer_decode.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/vision/device_input.hpp"

namespace gufo::hip {
namespace {

constexpr std::size_t kScoreBytesLimit = 48U * 1024U;
constexpr std::uint32_t kMaxScoreElems =
    static_cast<std::uint32_t>(kScoreBytesLimit / sizeof(float));
constexpr std::uint32_t kBlock = 256U;

GlimmerLaunchStatus Ok() noexcept { return {true, nullptr}; }

GlimmerLaunchStatus Bad(const char* error) noexcept { return {false, error}; }

bool Mul(std::size_t a, std::size_t b, std::size_t* out) noexcept {
  if (a != 0 && b > (static_cast<std::size_t>(-1) / a)) {
    return false;
  }
  *out = a * b;
  return true;
}

// Last fp32 KV element touched by (layer, last kv head, pos, last dim).
bool KvLast(std::uint32_t layer, std::uint32_t pos, std::uint32_t max_context,
            std::uint32_t num_kv_heads, std::uint32_t head_dim,
            std::size_t* last) noexcept {
  if (num_kv_heads == 0 || head_dim == 0 || max_context == 0) {
    return false;
  }
  std::size_t heads = 0;
  std::size_t span = 0;
  std::size_t slot = 0;
  std::size_t base = 0;
  if (!Mul(layer, num_kv_heads, &heads) ||
      !Mul(heads + (num_kv_heads - 1U), max_context, &span)) {
    return false;
  }
  if (span > static_cast<std::size_t>(-1) - pos) {
    return false;
  }
  slot = span + pos;
  if (!Mul(slot, head_dim, &base)) {
    return false;
  }
  if (base > static_cast<std::size_t>(-1) - (head_dim - 1U)) {
    return false;
  }
  *last = base + (head_dim - 1U);
  return true;
}

__host__ __device__ std::uint32_t ScoreCount(std::uint32_t pos,
                                            std::uint32_t window) noexcept {
  const std::uint32_t seq = pos + 1U;
  if (window == 0U || seq <= window) {
    return seq;
  }
  return window;
}

GlimmerLaunchStatus CheckLaunch(const char* what) noexcept {
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return Bad(what);
  }
  return Ok();
}

__global__ void GlimmerNormRopeKernel(float* __restrict__ heads,
                                      std::uint32_t head_dim,
                                      std::uint32_t rotary_dim,
                                      std::uint32_t pos, float theta) {
  float* head = heads + static_cast<std::size_t>(blockIdx.x) * head_dim;
  const std::uint32_t pairs = rotary_dim / 2U;
  for (std::uint32_t i = threadIdx.x; i < pairs; i += blockDim.x) {
    const float freq = powf(theta, (-2.0F * static_cast<float>(i)) /
                                       static_cast<float>(rotary_dim));
    const float ang = static_cast<float>(pos) * freq;
    const float c = cosf(ang);
    const float s = sinf(ang);
    const std::uint32_t i0 = i * 2U;
    const float x0 = head[i0];
    const float x1 = head[i0 + 1U];
    head[i0] = x0 * c - x1 * s;
    head[i0 + 1U] = x0 * s + x1 * c;
  }
}

__global__ void GlimmerWriteKvKernel(const float* __restrict__ k,
                                     const float* __restrict__ v,
                                     float* __restrict__ k_cache,
                                     float* __restrict__ v_cache,
                                     std::uint32_t layer, std::uint32_t pos,
                                     std::uint32_t max_context,
                                     std::uint32_t num_kv_heads,
                                     std::uint32_t head_dim) {
  const std::uint32_t kv_h = blockIdx.x;
  if (kv_h >= num_kv_heads || threadIdx.x >= head_dim) {
    return;
  }
  const std::size_t dst =
      (((static_cast<std::size_t>(layer) * num_kv_heads + kv_h) * max_context +
        pos) *
       head_dim) +
      threadIdx.x;
  const std::size_t src =
      static_cast<std::size_t>(kv_h) * head_dim + threadIdx.x;
  k_cache[dst] = k[src];
  v_cache[dst] = v[src];
}

__device__ __forceinline__ float GlimmerReduce(float value, float* scratch) {
  const std::uint32_t tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float total = scratch[0];
  __syncthreads();
  return total;
}

__device__ __forceinline__ float GlimmerReduceMax(float value, float* scratch) {
  const std::uint32_t tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
  }
  const float total = scratch[0];
  __syncthreads();
  return total;
}

// One query attends [p_begin, p_begin+nscore). Scores stay in a dynamic tile
// of at most kMaxScoreElems. Online softmax stitches the tiles.
__device__ __forceinline__ void GlimmerAttendQuery(const float* __restrict__ q_h,
                                   const float* __restrict__ gate_h,
                                   const float* __restrict__ sink_h,
                                   const float* __restrict__ k_cache,
                                   const float* __restrict__ v_cache,
                                   float* __restrict__ out_h,
                                   std::uint32_t layer, std::uint32_t p_begin,
                                   std::uint32_t nscore,
                                   std::uint32_t tile_limit,
                                   std::uint32_t max_context,
                                   std::uint32_t kv_h,
                                   std::uint32_t num_kv_heads,
                                   std::uint32_t head_dim, float* s_scores,
                                   float* s_scratch) {
  const std::uint32_t tid = threadIdx.x;
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  float acc = 0.0F;
  float running_max = sink_h != nullptr ? *sink_h : -1.0e30F;
  float running_sum = sink_h != nullptr ? 1.0F : 0.0F;

  for (std::uint32_t tile_begin = 0; tile_begin < nscore;
       tile_begin += tile_limit) {
    const std::uint32_t tile_n = tile_limit < (nscore - tile_begin)
                                    ? tile_limit
                                    : (nscore - tile_begin);
    for (std::uint32_t local = tid; local < tile_n; local += blockDim.x) {
      const std::uint32_t p = p_begin + tile_begin + local;
      const float* k_p =
          k_cache +
          (((static_cast<std::size_t>(layer) * num_kv_heads + kv_h) *
                max_context +
            p) *
           head_dim);
      float dot = 0.0F;
      for (std::uint32_t d = 0; d < head_dim; ++d) {
        dot += q_h[d] * k_p[d];
      }
      s_scores[local] = dot * scale;
    }
    __syncthreads();

    float local_max = -1.0e30F;
    for (std::uint32_t local = tid; local < tile_n; local += blockDim.x) {
      local_max = fmaxf(local_max, s_scores[local]);
    }
    const float tile_max = GlimmerReduceMax(local_max, s_scratch);
    const float new_max = fmaxf(running_max, tile_max);
    const float rescale = expf(running_max - new_max);
    if (tid < head_dim) {
      acc *= rescale;
    }

    for (std::uint32_t local = tid; local < tile_n; local += blockDim.x) {
      s_scores[local] = expf(s_scores[local] - new_max);
    }
    __syncthreads();
    float local_sum = 0.0F;
    for (std::uint32_t local = tid; local < tile_n; local += blockDim.x) {
      local_sum += s_scores[local];
    }
    const float tile_sum = GlimmerReduce(local_sum, s_scratch);
    running_sum = running_sum * rescale + tile_sum;

    if (tid < head_dim) {
      for (std::uint32_t local = 0; local < tile_n; ++local) {
        const std::uint32_t p = p_begin + tile_begin + local;
        const float* v_p =
            v_cache +
            (((static_cast<std::size_t>(layer) * num_kv_heads + kv_h) *
                  max_context +
              p) *
             head_dim);
        acc += s_scores[local] * v_p[tid];
      }
    }
    running_max = new_max;
    __syncthreads();
  }

  if (tid < head_dim) {
    float ctx = acc / (running_sum + 1.0e-9F);
    if (gate_h != nullptr) {
      const float g = gate_h[tid];
      ctx *= 1.0F / (1.0F + expf(-g));
    }
    out_h[tid] = ctx;
  }
}

__global__ void GlimmerWindowAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    float* __restrict__ out, std::uint32_t layer, std::uint32_t p_begin,
    std::uint32_t nscore, std::uint32_t tile_limit, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, const float* __restrict__ sinks) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads || num_kv_heads == 0 || nscore == 0 || tile_limit == 0) {
    return;
  }
  extern __shared__ float s_scores[];
  __shared__ float s_scratch[256];
  const std::uint32_t kv_h = h / (num_heads / num_kv_heads);
  const float* gate_h =
      gate == nullptr ? nullptr
                      : gate + static_cast<std::size_t>(h) * head_dim;
  const float* sink_h = sinks == nullptr ? nullptr : sinks + h;
  GlimmerAttendQuery(q + static_cast<std::size_t>(h) * head_dim, gate_h, sink_h,
                     k_cache, v_cache,
                     out + static_cast<std::size_t>(h) * head_dim, layer,
                     p_begin, nscore, tile_limit, max_context, kv_h,
                     num_kv_heads, head_dim, s_scores, s_scratch);
}

__global__ void GlimmerBatchedNormRopeKernel(float* __restrict__ heads,
                                             std::uint32_t head_dim,
                                             std::uint32_t rotary_dim,
                                             std::uint32_t start_pos,
                                             std::uint32_t batch,
                                             std::uint32_t count, float theta) {
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (head >= count || row >= batch) {
    return;
  }
  float* data = heads + (static_cast<std::size_t>(row) * count + head) * head_dim;
  const std::uint32_t pos = start_pos + row;
  const std::uint32_t pairs = rotary_dim / 2U;
  for (std::uint32_t i = threadIdx.x; i < pairs; i += blockDim.x) {
    const float freq = powf(theta, (-2.0F * static_cast<float>(i)) /
                                       static_cast<float>(rotary_dim));
    const float ang = static_cast<float>(pos) * freq;
    const float c = cosf(ang);
    const float s = sinf(ang);
    const std::uint32_t i0 = i * 2U;
    const float x0 = data[i0];
    const float x1 = data[i0 + 1U];
    data[i0] = x0 * c - x1 * s;
    data[i0 + 1U] = x0 * s + x1 * c;
  }
}

__global__ void GlimmerBatchedWriteKvKernel(const float* __restrict__ k,
                                            const float* __restrict__ v,
                                            float* __restrict__ k_cache,
                                            float* __restrict__ v_cache,
                                            std::uint32_t layer,
                                            std::uint32_t start_pos,
                                            std::uint32_t batch,
                                            std::uint32_t max_context,
                                            std::uint32_t num_kv_heads,
                                            std::uint32_t head_dim) {
  const std::uint32_t kv_h = blockIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (kv_h >= num_kv_heads || row >= batch || threadIdx.x >= head_dim) {
    return;
  }
  const std::uint32_t pos = start_pos + row;
  const std::size_t dst =
      (((static_cast<std::size_t>(layer) * num_kv_heads + kv_h) * max_context +
        pos) *
       head_dim) +
      threadIdx.x;
  const std::size_t src =
      (static_cast<std::size_t>(row) * num_kv_heads + kv_h) * head_dim +
      threadIdx.x;
  k_cache[dst] = k[src];
  v_cache[dst] = v[src];
}

__global__ void GlimmerPrefillAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    float* __restrict__ out, std::uint32_t layer, std::uint32_t start_pos,
    std::uint32_t batch, std::uint32_t window, std::uint32_t tile_limit,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    const float* __restrict__ sinks) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (h >= num_heads || row >= batch || num_kv_heads == 0 || tile_limit == 0) {
    return;
  }
  const std::uint32_t pos = start_pos + row;
  const std::uint32_t nscore = ScoreCount(pos, window);
  const std::uint32_t p_begin = pos + 1U - nscore;
  extern __shared__ float s_scores[];
  __shared__ float s_scratch[256];
  const std::uint32_t kv_h = h / (num_heads / num_kv_heads);
  const std::size_t head_elems = static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t base =
      static_cast<std::size_t>(row) * head_elems +
      static_cast<std::size_t>(h) * head_dim;
  const float* gate_h = gate == nullptr ? nullptr : gate + base;
  const float* sink_h = sinks == nullptr ? nullptr : sinks + h;
  GlimmerAttendQuery(q + base, gate_h, sink_h, k_cache, v_cache, out + base,
                     layer, p_begin, nscore, tile_limit, max_context, kv_h,
                     num_kv_heads, head_dim, s_scores, s_scratch);
}

__global__ void GlimmerScaleSoftcapKernel(float* __restrict__ logits,
                                          std::size_t count, float scale,
                                          float cap) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  float value = logits[idx] * scale;
  if (cap != 0.0F) {
    value = cap * tanhf(value / cap);
  }
  logits[idx] = value;
}

void Require(const GlimmerLaunchStatus& status) {
  if (!status.ok) {
    throw std::runtime_error(status.error != nullptr ? status.error
                                                     : "glimmer launch failed");
  }
}

}  // namespace

GlimmerLaunchStatus LaunchGlimmerNormRope(float* heads, std::uint32_t count,
                                          std::uint32_t head_dim,
                                          std::uint32_t rotary_dim,
                                          std::uint32_t pos, float theta,
                                          hipStream_t stream) {
  if (heads == nullptr || count == 0 || head_dim == 0 || head_dim > kBlock ||
      rotary_dim == 0 || rotary_dim > head_dim || (rotary_dim % 2U) != 0 ||
      !(theta > 0.0F)) {
    return Bad("glimmer norm rope refused by host audit");
  }
  std::size_t need = 0;
  if (!Mul(count, head_dim, &need)) {
    return Bad("glimmer norm rope index overflow");
  }
  GlimmerNormRopeKernel<<<count, kBlock, 0, stream>>>(heads, head_dim,
                                                      rotary_dim, pos, theta);
  return CheckLaunch("glimmer norm rope launch failed");
}

GlimmerLaunchStatus LaunchGlimmerWriteKv(
    const float* k, const float* v, float* k_cache, float* v_cache,
    std::uint32_t layer, std::uint32_t pos, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, std::size_t kv_elems,
    std::size_t cache_elems, hipStream_t stream) {
  if (k == nullptr || v == nullptr || k_cache == nullptr || v_cache == nullptr ||
      num_kv_heads == 0 || head_dim == 0 || head_dim > kBlock ||
      pos >= max_context) {
    return Bad("glimmer kv write refused by host audit");
  }
  std::size_t need = 0;
  std::size_t last = 0;
  if (!Mul(num_kv_heads, head_dim, &need) || need > kv_elems ||
      !KvLast(layer, pos, max_context, num_kv_heads, head_dim, &last) ||
      last >= cache_elems) {
    return Bad("glimmer kv write index out of range");
  }
  GlimmerWriteKvKernel<<<num_kv_heads, kBlock, 0, stream>>>(
      k, v, k_cache, v_cache, layer, pos, max_context, num_kv_heads, head_dim);
  return CheckLaunch("glimmer kv write launch failed");
}

GlimmerLaunchStatus LaunchGlimmerWindowAttention(
    const float* q, const float* gate, const float* k_cache,
    const float* v_cache, float* out, std::uint32_t layer, std::uint32_t pos,
    std::uint32_t window, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, std::size_t q_elems,
    std::size_t gate_elems, std::size_t cache_elems, std::size_t out_elems,
    hipStream_t stream, const float* sinks) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      out == nullptr || num_heads == 0 || num_kv_heads == 0 ||
      (num_heads % num_kv_heads) != 0 || head_dim == 0 || head_dim > kBlock ||
      pos >= max_context) {
    return Bad("glimmer window attention refused by host audit");
  }
  if (gate != nullptr && gate_elems < q_elems) {
    return Bad("glimmer window attention gate is short");
  }
  const std::uint32_t nscore = ScoreCount(pos, window);
  const std::uint32_t p_begin = pos + 1U - nscore;
  if (nscore == 0) {
    return Bad("glimmer window attention has an empty score span");
  }
  const std::uint32_t tile =
      nscore < kMaxScoreElems ? nscore : kMaxScoreElems;
  if (static_cast<std::size_t>(tile) * sizeof(float) > kScoreBytesLimit) {
    return Bad("glimmer window attention score buffer exceeds 48 KiB");
  }
  std::size_t q_need = 0;
  std::size_t last = 0;
  if (!Mul(num_heads, head_dim, &q_need) || q_need > q_elems ||
      q_need > out_elems ||
      !KvLast(layer, pos, max_context, num_kv_heads, head_dim, &last) ||
      last >= cache_elems) {
    return Bad("glimmer window attention index out of range");
  }
  GlimmerWindowAttentionKernel<<<num_heads, kBlock, tile * sizeof(float),
                                 stream>>>(q, gate, k_cache, v_cache, out,
                                           layer, p_begin, nscore, tile,
                                           max_context, num_heads, num_kv_heads,
                                           head_dim, sinks);
  return CheckLaunch("glimmer window attention launch failed");
}

std::uint32_t GlimmerScoreTileElems() noexcept { return kMaxScoreElems; }

GlimmerLaunchStatus LaunchGlimmerBatchedNormRope(
    float* heads, std::uint32_t count, std::uint32_t head_dim,
    std::uint32_t rotary_dim, std::uint32_t start_pos, std::uint32_t batch,
    float theta, hipStream_t stream) {
  if (heads == nullptr || count == 0 || batch == 0 || head_dim == 0 ||
      head_dim > kBlock || rotary_dim == 0 || rotary_dim > head_dim ||
      (rotary_dim % 2U) != 0 || !(theta > 0.0F)) {
    return Bad("glimmer batched norm rope refused by host audit");
  }
  if (start_pos > static_cast<std::uint32_t>(-1) - (batch - 1U)) {
    return Bad("glimmer batched norm rope position overflow");
  }
  const dim3 grid(count, batch);
  GlimmerBatchedNormRopeKernel<<<grid, kBlock, 0, stream>>>(
      heads, head_dim, rotary_dim, start_pos, batch, count, theta);
  return CheckLaunch("glimmer batched norm rope launch failed");
}

GlimmerLaunchStatus LaunchGlimmerBatchedWriteKv(
    const float* k, const float* v, float* k_cache, float* v_cache,
    std::uint32_t layer, std::uint32_t start_pos, std::uint32_t batch,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, std::size_t kv_row_elems, std::size_t cache_elems,
    hipStream_t stream) {
  if (k == nullptr || v == nullptr || k_cache == nullptr || v_cache == nullptr ||
      num_kv_heads == 0 || head_dim == 0 || head_dim > kBlock || batch == 0 ||
      start_pos >= max_context ||
      batch > max_context - start_pos) {
    return Bad("glimmer batched kv write refused by host audit");
  }
  std::size_t row = 0;
  std::size_t need = 0;
  std::size_t last = 0;
  const std::uint32_t last_pos = start_pos + batch - 1U;
  if (!Mul(num_kv_heads, head_dim, &row) ||
      !Mul(row, batch, &need) || need > kv_row_elems ||
      !KvLast(layer, last_pos, max_context, num_kv_heads, head_dim, &last) ||
      last >= cache_elems) {
    return Bad("glimmer batched kv write index out of range");
  }
  const dim3 grid(num_kv_heads, batch);
  GlimmerBatchedWriteKvKernel<<<grid, kBlock, 0, stream>>>(
      k, v, k_cache, v_cache, layer, start_pos, batch, max_context,
      num_kv_heads, head_dim);
  return CheckLaunch("glimmer batched kv write launch failed");
}

GlimmerLaunchStatus LaunchGlimmerPrefillAttention(
    const float* q, const float* gate, const float* k_cache,
    const float* v_cache, float* out, std::uint32_t layer,
    std::uint32_t start_pos, std::uint32_t batch, std::uint32_t window,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, std::size_t q_elems,
    std::size_t gate_elems, std::size_t cache_elems, std::size_t out_elems,
    hipStream_t stream, const float* sinks) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      out == nullptr || num_heads == 0 || num_kv_heads == 0 ||
      (num_heads % num_kv_heads) != 0 || head_dim == 0 || head_dim > kBlock ||
      batch == 0 || start_pos >= max_context ||
      batch > max_context - start_pos) {
    return Bad("glimmer prefill attention refused by host audit");
  }
  if (gate != nullptr && gate_elems < q_elems) {
    return Bad("glimmer prefill attention gate is short");
  }
  const std::uint32_t last_pos = start_pos + batch - 1U;
  const std::uint32_t nscore = ScoreCount(last_pos, window);
  if (nscore == 0) {
    return Bad("glimmer prefill attention has an empty score span");
  }
  const std::uint32_t tile =
      nscore < kMaxScoreElems ? nscore : kMaxScoreElems;
  if (static_cast<std::size_t>(tile) * sizeof(float) > kScoreBytesLimit) {
    return Bad("glimmer prefill attention score buffer exceeds 48 KiB");
  }
  std::size_t row = 0;
  std::size_t need = 0;
  std::size_t last = 0;
  if (!Mul(num_heads, head_dim, &row) || !Mul(row, batch, &need) ||
      need > q_elems || need > out_elems ||
      !KvLast(layer, last_pos, max_context, num_kv_heads, head_dim, &last) ||
      last >= cache_elems) {
    return Bad("glimmer prefill attention index out of range");
  }
  const dim3 grid(num_heads, batch);
  GlimmerPrefillAttentionKernel<<<grid, kBlock, tile * sizeof(float), stream>>>(
      q, gate, k_cache, v_cache, out, layer, start_pos, batch, window, tile,
      max_context, num_heads, num_kv_heads, head_dim, sinks);
  return CheckLaunch("glimmer prefill attention launch failed");
}

GlimmerLaunchStatus LaunchGlimmerScaleSoftcap(float* logits, std::size_t count,
                                              float scale, float cap,
                                              hipStream_t stream) {
  if (logits == nullptr || count == 0 || scale == 0.0F) {
    return Bad("glimmer scale softcap refused by host audit");
  }
  const std::size_t grid = (count + kBlock - 1U) / kBlock;
  GlimmerScaleSoftcapKernel<<<grid, kBlock, 0, stream>>>(logits, count, scale,
                                                         cap);
  return CheckLaunch("glimmer scale softcap launch failed");
}

void ExecuteGlimmerDecodeStep(QwenGpuArena& arena,
                              const models::QwenModelWeights& weights,
                              tokenization::TokenId token_id, std::uint32_t pos,
                              bool compute_logits,
                              models::qwen::vision::DeviceInput* image_input) {
  if (image_input != nullptr && image_input->Bytes() != 0) {
    throw std::runtime_error("muse-glimmer vision is not supported");
  }
  const auto& config = weights.config;
  if (!config.attention_only) {
    throw std::runtime_error("glimmer decode called for a qwen config");
  }
  if (arena.d_kv_cache == nullptr) {
    throw std::runtime_error("muse-glimmer requires fp32 KV");
  }
  if (pos >= arena.GetMaxContext()) {
    throw std::runtime_error("glimmer token position exceeds the GPU context");
  }

  auto scratch = arena.GetScratchView();
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv = static_cast<std::size_t>(config.num_key_value_heads) *
                         config.head_dim;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t plane = arena.GetAttentionKvPlaneElements();
  if (scratch.decode.hidden.size() < hidden ||
      scratch.decode.normed.size() < hidden || scratch.attention.q.size() < attention ||
      scratch.attention.k.size() < kv || scratch.attention.v.size() < kv ||
      scratch.ssm.gate.size() < attention || scratch.ssm.out.size() < attention ||
      scratch.attention.output.size() < hidden ||
      scratch.ffn.activation.size() < intermediate ||
      scratch.ffn.out.size() < hidden ||
      scratch.decode.prompt_tokens.size() < 2) {
    throw std::runtime_error("glimmer decode scratch is short");
  }

  const std::uint32_t params[2] = {token_id, pos};
  if (hipMemcpyAsync(scratch.decode.prompt_tokens.data(), params, sizeof(params),
                     hipMemcpyHostToDevice, arena.stream) != hipSuccess) {
    throw std::runtime_error("glimmer token copy failed");
  }

  LaunchEmbeddingLookup(weights.token_embd.data, weights.token_embd.type,
                        scratch.decode.prompt_tokens.data(),
                        scratch.decode.hidden.data(), hidden, arena.stream);
  LaunchRMSNorm(scratch.decode.hidden.data(), nullptr,
                scratch.decode.hidden.data(), hidden, config.norm_eps_pre,
                arena.stream);

  float* v_cache = arena.d_kv_cache + plane;
  for (std::uint32_t layer_index = 0; layer_index < config.num_layers;
       ++layer_index) {
    const auto& layer = weights.layers[layer_index];
    if (layer.attn_q_norm.data == nullptr || layer.attn_k_norm.data == nullptr ||
        layer.attn_post_norm.data == nullptr ||
        layer.ffn_post_norm.data == nullptr) {
      throw std::runtime_error("glimmer layer is missing a norm tensor");
    }
    LaunchRMSNorm(scratch.decode.hidden.data(),
                  static_cast<const float*>(layer.attn_norm.data),
                  scratch.decode.normed.data(), hidden, config.norm_eps_pre,
                  arena.stream);
    LaunchGEMV(layer.attn_q.data, layer.attn_q.type, scratch.decode.normed.data(),
               scratch.attention.q.data(), attention, hidden, arena.stream);
    LaunchGEMV(layer.attn_k.data, layer.attn_k.type, scratch.decode.normed.data(),
               scratch.attention.k.data(), kv, hidden, arena.stream);
    LaunchGEMV(layer.attn_v.data, layer.attn_v.type, scratch.decode.normed.data(),
               scratch.attention.v.data(), kv, hidden, arena.stream);
    LaunchGEMV(layer.attn_gate.data, layer.attn_gate.type,
               scratch.decode.normed.data(), scratch.ssm.gate.data(), attention,
               hidden, arena.stream);
    LaunchPerHeadRMSNorm(scratch.attention.q.data(),
                         static_cast<const float*>(layer.attn_q_norm.data),
                         scratch.attention.q.data(), config.num_attention_heads,
                         config.head_dim, config.norm_eps_pre, arena.stream);
    LaunchPerHeadRMSNorm(scratch.attention.k.data(),
                         static_cast<const float*>(layer.attn_k_norm.data),
                         scratch.attention.k.data(), config.num_key_value_heads,
                         config.head_dim, config.norm_eps_pre, arena.stream);
    if (config.LayerUsesRope(layer_index)) {
      Require(LaunchGlimmerNormRope(
          scratch.attention.q.data(), config.num_attention_heads, config.head_dim,
          config.rotary_dim, pos, config.rope_theta, arena.stream));
      Require(LaunchGlimmerNormRope(
          scratch.attention.k.data(), config.num_key_value_heads, config.head_dim,
          config.rotary_dim, pos, config.rope_theta, arena.stream));
    }
    Require(LaunchGlimmerWriteKv(
        scratch.attention.k.data(), scratch.attention.v.data(), arena.d_kv_cache,
        v_cache, layer_index, pos, arena.GetMaxContext(),
        config.num_key_value_heads, config.head_dim, kv, plane, arena.stream));
    Require(LaunchGlimmerWindowAttention(
        scratch.attention.q.data(), scratch.ssm.gate.data(), arena.d_kv_cache,
        v_cache, scratch.ssm.out.data(), layer_index, pos,
        config.LayerSwaWindow(layer_index), arena.GetMaxContext(),
        config.num_attention_heads, config.num_key_value_heads, config.head_dim,
        attention, attention, plane, attention, arena.stream));
    LaunchGEMV(layer.attn_output.data, layer.attn_output.type,
               scratch.ssm.out.data(), scratch.attention.output.data(), hidden,
               attention, arena.stream);
    LaunchRMSNorm(scratch.attention.output.data(),
                  static_cast<const float*>(layer.attn_post_norm.data),
                  scratch.decode.normed.data(), hidden, config.norm_eps_post,
                  arena.stream);
    LaunchResidualAdd(scratch.decode.hidden.data(), scratch.decode.normed.data(),
                      scratch.decode.hidden.data(), hidden, arena.stream);
    LaunchRMSNorm(scratch.decode.hidden.data(),
                  static_cast<const float*>(layer.ffn_norm.data),
                  scratch.decode.normed.data(), hidden, config.norm_eps_pre,
                  arena.stream);
    LaunchFusedSwiGLUGEMV(layer.ffn_gate.data, layer.ffn_gate.type,
                          layer.ffn_up.data, layer.ffn_up.type,
                          scratch.decode.normed.data(),
                          scratch.ffn.activation.data(), intermediate, hidden,
                          arena.stream);
    LaunchGEMV(layer.ffn_down.data, layer.ffn_down.type,
               scratch.ffn.activation.data(), scratch.ffn.out.data(), hidden,
               intermediate, arena.stream);
    LaunchRMSNorm(scratch.ffn.out.data(),
                  static_cast<const float*>(layer.ffn_post_norm.data),
                  scratch.decode.normed.data(), hidden, config.norm_eps_post,
                  arena.stream);
    LaunchResidualAdd(scratch.decode.hidden.data(), scratch.decode.normed.data(),
                      scratch.decode.hidden.data(), hidden, arena.stream);
  }

  if (!compute_logits) {
    return;
  }
  // Argmax falls back to one workgroup when FFN scratch cannot hold partials.
  if (scratch.decode.logits.size() < config.vocab_size ||
      scratch.decode.sampled_token.empty() || scratch.ffn.out.empty()) {
    throw std::runtime_error("glimmer logit scratch is short");
  }
  LaunchRMSNorm(scratch.decode.hidden.data(),
                static_cast<const float*>(weights.output_norm.data),
                scratch.decode.normed.data(), hidden, config.norm_eps_pre,
                arena.stream);
  LaunchGEMV(weights.output.data, weights.output.type,
             scratch.decode.normed.data(), scratch.decode.logits.data(),
             config.vocab_size, hidden, arena.stream);
  Require(LaunchGlimmerScaleSoftcap(scratch.decode.logits.data(),
                                    config.vocab_size, config.logit_scale,
                                    config.final_softcap, arena.stream));
  LaunchBatchedGPUArgmax(scratch.decode.logits.data(),
                         scratch.decode.sampled_token.data(), 1,
                         config.vocab_size, scratch.ffn.out, arena.stream);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
