#if defined(ENGINE_ENABLE_HIP)
#include "src/models/glimmer/hip/glimmer_attention.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdlib>

#include "src/models/qwen/hip/ops/attention.hpp"

namespace gufo::hip {
namespace {

constexpr std::uint32_t kRowsPerBlock = 4U;
constexpr std::uint32_t kGroup = 16U;
constexpr std::uint32_t kHeadDim = 128U;
constexpr std::uint32_t kPart = 64U;
constexpr std::uint32_t kKeyTile = 16U;
constexpr std::uint32_t kThreads = kRowsPerBlock * 32U;

__host__ __device__ std::uint32_t SharedMin(std::uint32_t a,
                                            std::uint32_t b) noexcept {
  return a < b ? a : b;
}

__host__ __device__ std::uint32_t SharedMax(std::uint32_t a,
                                            std::uint32_t b) noexcept {
  return a > b ? a : b;
}

__host__ __device__ std::uint32_t SharedScoreCount(std::uint32_t pos,
                                                  std::uint32_t window) noexcept {
  const std::uint32_t seq = pos + 1U;
  if (window == 0U || seq <= window) {
    return seq;
  }
  return window;
}

// LGA: scalar inner loop. Each K/V row is staged once per KV head.
// Warp w owns query row row0+w. Lane L owns query head (L%16) and dims
// [64*(L/16), 64*(L/16)+64). The two lanes of a head add their partial dots
// with a warp shuffle, then each keeps a 64-wide fp32 accumulator.
// The fp16 plane is token-major ((layer*maxctx+pos)*nkv+kv)*hd+d.
__global__ void __launch_bounds__(kThreads)
    GlimmerSharedAttnKernel(const float* __restrict__ q,
                            const float* __restrict__ gate,
                            const __half* __restrict__ k_cache,
                            const __half* __restrict__ v_cache,
                            float* __restrict__ out, std::uint32_t layer,
                            std::uint32_t start_pos, std::uint32_t batch,
                            std::uint32_t window, std::uint32_t max_context,
                            std::uint32_t num_heads, std::uint32_t num_kv_heads) {
  const std::uint32_t kv_h = blockIdx.x;
  const std::uint32_t row0 = blockIdx.y * kRowsPerBlock;
  const std::uint32_t warp = threadIdx.x >> 5U;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t row = row0 + warp;
  const bool live = row < batch;
  const std::uint32_t head = lane & 15U;
  const std::uint32_t part = lane >> 4U;
  const std::uint32_t qh = kv_h * kGroup + head;

  __shared__ float q_tile[kRowsPerBlock][kGroup][kHeadDim];
  __shared__ __half k_tile[kKeyTile][kHeadDim];
  __shared__ __half v_tile[kKeyTile][kHeadDim];
  __shared__ std::uint32_t begin_s[kRowsPerBlock];
  __shared__ std::uint32_t end_s[kRowsPerBlock];

  const std::uint32_t valid_rows =
      row0 >= batch ? 0U : SharedMin(kRowsPerBlock, batch - row0);
  const std::uint32_t q_elems = valid_rows * kGroup * kHeadDim;
  for (std::uint32_t elem = threadIdx.x; elem < q_elems; elem += blockDim.x) {
    const std::uint32_t dim = elem % kHeadDim;
    const std::uint32_t tmp = elem / kHeadDim;
    const std::uint32_t local_head = tmp % kGroup;
    const std::uint32_t local_row = tmp / kGroup;
    const std::size_t src =
        (static_cast<std::size_t>(row0 + local_row) * num_heads +
         (static_cast<std::size_t>(kv_h) * kGroup + local_head)) *
            kHeadDim +
        dim;
    q_tile[local_row][local_head][dim] = q[src];
  }
  if (lane == 0U) {
    if (live) {
      const std::uint32_t pos = start_pos + row;
      const std::uint32_t nscore = SharedScoreCount(pos, window);
      begin_s[warp] = pos + 1U - nscore;
      end_s[warp] = pos + 1U;
    } else {
      begin_s[warp] = 0xFFFFFFFFU;
      end_s[warp] = 0U;
    }
  }
  __syncthreads();

  std::uint32_t min_begin = begin_s[0];
  std::uint32_t max_end = end_s[0];
#pragma unroll
  for (std::uint32_t i = 1; i < kRowsPerBlock; ++i) {
    min_begin = SharedMin(min_begin, begin_s[i]);
    max_end = SharedMax(max_end, end_s[i]);
  }

  float acc[kPart];
#pragma unroll
  for (std::uint32_t dim = 0; dim < kPart; ++dim) {
    acc[dim] = 0.0F;
  }
  float running_max = -1.0e30F;
  float running_sum = 0.0F;
  const float scale = 0.08838834764831845F;  // 1/sqrt(128)
  const std::uint32_t row_begin = live ? begin_s[warp] : 0U;
  const std::uint32_t row_end = live ? end_s[warp] : 0U;

  for (std::uint32_t tile = min_begin; tile < max_end; tile += kKeyTile) {
    const std::uint32_t nload = SharedMin(kKeyTile, max_end - tile);
    const std::uint32_t load_elems = nload * kHeadDim;
    for (std::uint32_t elem = threadIdx.x; elem < load_elems;
         elem += blockDim.x) {
      const std::uint32_t dim = elem % kHeadDim;
      const std::uint32_t local = elem / kHeadDim;
      const std::uint32_t pos = tile + local;
      const std::size_t src =
          (((static_cast<std::size_t>(layer) * max_context + pos) *
                num_kv_heads +
            kv_h) *
           kHeadDim) +
          dim;
      k_tile[local][dim] = k_cache[src];
      v_tile[local][dim] = v_cache[src];
    }
    __syncthreads();

    if (live) {
#pragma unroll 1
      for (std::uint32_t local = 0; local < kKeyTile; ++local) {
        const std::uint32_t pos = tile + local;
        if (local >= nload || pos < row_begin || pos >= row_end) {
          continue;
        }
        float partial = 0.0F;
        {
#pragma clang fp contract(off)
#pragma unroll
          for (std::uint32_t dim = 0; dim < kPart; ++dim) {
            partial += q_tile[warp][head][part * kPart + dim] *
                       __half2float(k_tile[local][part * kPart + dim]);
          }
        }
        const float dot = partial + __shfl_xor(partial, 16U);
        const float score = dot * scale;
        const float new_max = fmaxf(running_max, score);
        const float rescale = expf(running_max - new_max);
        const float weight = expf(score - new_max);
#pragma unroll
        for (std::uint32_t dim = 0; dim < kPart; ++dim) {
          acc[dim] = acc[dim] * rescale +
                     weight * __half2float(v_tile[local][part * kPart + dim]);
        }
        running_sum = running_sum * rescale + weight;
        running_max = new_max;
      }
    }
    __syncthreads();
  }

  if (live) {
    const float inv = 1.0F / (running_sum + 1.0e-9F);
#pragma unroll
    for (std::uint32_t dim = 0; dim < kPart; ++dim) {
      float value = acc[dim] * inv;
      if (gate != nullptr) {
        const std::size_t gate_at =
            (static_cast<std::size_t>(row) * num_heads + qh) * kHeadDim +
            part * kPart + dim;
        const float g = gate[gate_at];
        value *= 1.0F / (1.0F + expf(-g));
      }
      const std::size_t out_at =
          (static_cast<std::size_t>(row) * num_heads + qh) * kHeadDim +
          part * kPart + dim;
      out[out_at] = value;
    }
  }
}

// LGA: remap the head-major fp32 reference into the token-major fp16 plane.
// blockIdx.y selects (K/V half, layer, KV head) with per-block index math;
// blockIdx.x walks the head's (valid_rows x head_dim) elements flat. One
// 32-bit divmod per thread; restore-only, so throughput is not critical.
// PGA: layer_base lets a caller cast ONE layer of a many-layer plane, and
// plane_layers is how many layers the WHOLE plane holds. The K/V half stride
// is that whole per-half plane, not the block being cast, because V always
// starts one full K plane past K (both halves are [all layers]; the WMMA
// launcher assumes the same when it walks layer_k / layer_v off k_f16 / v_f16).
// plane_layers 0 means "stride = layers", which is LGA's original behaviour:
// arena.cpp's two snapshot-restore calls pass neither argument and are
// byte-for-byte unchanged.
// PGA MEASURED: deriving the stride from (layer_base + layers) instead put V
// layer_index+1 planes past K, i.e. in the wrong layer's slot for every layer
// but the last -- PPL 337811 and incoherent greedy text.
__global__ void GlimmerCastKvKernel(const float* __restrict__ src,
                                    __half* __restrict__ dst,
                                    std::uint32_t layers,
                                    std::uint32_t max_context,
                                    std::uint32_t num_kv_heads,
                                    std::uint32_t head_dim,
                                    std::uint32_t valid_rows,
                                    std::uint32_t layer_base,
                                    std::uint32_t plane_layers) {
  const std::uint32_t per_half = layers * num_kv_heads;
  const std::uint32_t half = blockIdx.y / per_half;
  const std::uint32_t r = blockIdx.y % per_half;
  const std::uint32_t l = r / num_kv_heads;
  const std::uint32_t h = r % num_kv_heads;
  const std::size_t layer = static_cast<std::size_t>(l) + layer_base;
  const std::size_t plane =
      (plane_layers != 0U ? plane_layers : layers) * max_context *
      num_kv_heads * head_dim;
  const std::size_t per_head =
      static_cast<std::size_t>(valid_rows) * head_dim;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t head_f32 =
      ((layer * num_kv_heads + h) * max_context) * head_dim;
  const std::size_t head_tm =
      layer * max_context * num_kv_heads * head_dim;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < per_head; i += stride) {
    const std::uint32_t d = static_cast<std::uint32_t>(i % head_dim);
    const std::uint32_t p = static_cast<std::uint32_t>(i / head_dim);
    const std::size_t f32 =
        half * plane + head_f32 + static_cast<std::size_t>(p) * head_dim + d;
    const std::size_t tm = half * plane + head_tm +
                           (static_cast<std::size_t>(p) * num_kv_heads + h) *
                               head_dim +
                           d;
    dst[tm] = __float2half_rn(src[f32]);
  }
}

bool MulOk(std::size_t a, std::size_t b, std::size_t* out) noexcept {
  if (a != 0 && b > (static_cast<std::size_t>(-1) / a)) {
    return false;
  }
  *out = a * b;
  return true;
}

}  // namespace

GlimmerLaunchStatus LaunchGlimmerCastKvFp16(
    const float* src, void* dst, std::uint32_t layers,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, std::uint32_t valid_rows, hipStream_t stream,
    std::uint32_t layer_base, std::uint32_t plane_layers) {
  if (valid_rows == 0) {
    return {true, nullptr};
  }
  if (src == nullptr || dst == nullptr || layers == 0 || max_context == 0 ||
      num_kv_heads == 0 || head_dim == 0 || valid_rows > max_context) {
    return {false, "glimmer fp16 kv cast refused by host audit"};
  }
  if (plane_layers != 0 &&
      static_cast<std::uint64_t>(layer_base) + layers > plane_layers) {
    return {false, "glimmer fp16 kv cast layer range outside the plane"};
  }
  if (static_cast<std::uint64_t>(layer_base) + layers > 65536U) {
    return {false, "glimmer fp16 kv cast layer range out of range"};
  }
  std::size_t per_half = 0;
  std::size_t plane = 0;
  std::size_t per_head = 0;
  if (!MulOk(layers, num_kv_heads, &per_half) || per_half == 0 ||
      per_half > 16384U ||
      !MulOk(plane_layers != 0U ? plane_layers : layers,
             static_cast<std::size_t>(max_context) * head_dim *
                 num_kv_heads,
             &plane) ||
      !MulOk(valid_rows, head_dim, &per_head) || per_head == 0) {
    return {false, "glimmer fp16 kv cast index out of range"};
  }
  constexpr std::uint32_t kBlock = 256U;
  const std::size_t need = (per_head + kBlock - 1U) / kBlock;
  const std::uint32_t grid_x = static_cast<std::uint32_t>(
      need > 16384U ? static_cast<std::size_t>(16384U) : need);
  const dim3 grid(grid_x, static_cast<unsigned int>(2 * per_half));
  GlimmerCastKvKernel<<<grid, kBlock, 0, stream>>>(
      src, static_cast<__half*>(dst), layers, max_context, num_kv_heads,
      head_dim, valid_rows, layer_base, plane_layers);
  if (hipGetLastError() != hipSuccess) {
    return {false, "glimmer fp16 kv cast launch failed"};
  }
  return {true, nullptr};
}

bool GlimmerSharedAttnOn() noexcept {
  static int on = -1;
  if (on < 0) {
    const char* value = std::getenv("GLIMMER_ATTN");
    on = (value != nullptr && value[0] == '0' && value[1] == '\0') ? 0 : 1;
  }
  return on == 1;
}

GlimmerLaunchStatus LaunchGlimmerSharedAttention(
    const float* q, const float* gate, const void* k_f16, const void* v_f16,
    float* out, std::uint32_t layer, std::uint32_t start_pos,
    std::uint32_t batch, std::uint32_t window, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, std::size_t q_elems, std::size_t gate_elems,
    std::size_t cache_elems, std::size_t out_elems, hipStream_t stream) {
  if (q == nullptr || k_f16 == nullptr || v_f16 == nullptr || out == nullptr ||
      num_kv_heads == 0 || num_heads != num_kv_heads * kGroup ||
      head_dim != kHeadDim || batch == 0 || start_pos >= max_context ||
      batch > max_context - start_pos) {
    return {false, "glimmer shared attention refused by host audit"};
  }
  if (gate != nullptr && gate_elems < q_elems) {
    return {false, "glimmer shared attention gate is short"};
  }
  const std::uint32_t last_pos = start_pos + batch - 1U;
  std::size_t row = 0;
  std::size_t need = 0;
  std::size_t layer_span = 0;
  std::size_t slot = 0;
  std::size_t base = 0;
  if (!MulOk(num_heads, head_dim, &row) || !MulOk(row, batch, &need) ||
      need > q_elems || need > out_elems) {
    return {false, "glimmer shared attention index out of range"};
  }
  // LGA: token-major last element touched:
  // ((layer*maxctx+last_pos)*nkv + (nkv-1))*hd + (hd-1).
  if (!MulOk(layer, max_context, &layer_span) ||
      layer_span > static_cast<std::size_t>(-1) - last_pos ||
      !MulOk(layer_span + last_pos, num_kv_heads, &slot) ||
      slot > static_cast<std::size_t>(-1) - (num_kv_heads - 1U) ||
      !MulOk(slot + (num_kv_heads - 1U), head_dim, &base) ||
      base > static_cast<std::size_t>(-1) - (head_dim - 1U) ||
      base + (head_dim - 1U) >= cache_elems) {
    return {false, "glimmer shared attention cache index out of range"};
  }
  const dim3 grid(num_kv_heads, (batch + kRowsPerBlock - 1U) / kRowsPerBlock);
  GlimmerSharedAttnKernel<<<grid, kThreads, 0, stream>>>(
      q, gate, static_cast<const __half*>(k_f16),
      static_cast<const __half*>(v_f16), out, layer, start_pos, batch, window,
      max_context, num_heads, num_kv_heads);
  if (hipGetLastError() != hipSuccess) {
    return {false, "glimmer shared attention launch failed"};
  }
  return {true, nullptr};
}

GlimmerLaunchStatus LaunchGlimmerWmmaPrefillAttention(
    const float* q, const float* gate, const void* k_f16, const void* v_f16,
    float* out, std::uint32_t layer, std::uint32_t start_pos,
    std::uint32_t batch, std::uint32_t window, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, std::size_t q_elems, std::size_t gate_elems,
    std::size_t cache_elems, std::size_t out_elems,
    std::span<float> packed_k_workspace, std::span<float> packed_v_workspace,
    hipStream_t stream) {
  // LGA: same host audit as the shared path (token-major cache bound), then
  // the WMMA dispatcher. skip_kv_write is always true: the prefill
  // dual-write already wrote both planes.
  if (q == nullptr || k_f16 == nullptr || v_f16 == nullptr || out == nullptr ||
      num_kv_heads == 0 || num_heads != num_kv_heads * kGroup ||
      head_dim != kHeadDim || batch == 0 || start_pos >= max_context ||
      batch > max_context - start_pos) {
    return {false, "glimmer wmma prefill refused by host audit"};
  }
  if (gate != nullptr && gate_elems < q_elems) {
    return {false, "glimmer wmma prefill gate is short"};
  }
  const std::uint32_t last_pos = start_pos + batch - 1U;
  std::size_t row = 0;
  std::size_t need = 0;
  std::size_t layer_span = 0;
  std::size_t slot = 0;
  std::size_t base = 0;
  if (!MulOk(num_heads, head_dim, &row) || !MulOk(row, batch, &need) ||
      need > q_elems || need > out_elems) {
    return {false, "glimmer wmma prefill index out of range"};
  }
  if (!MulOk(layer, max_context, &layer_span) ||
      layer_span > static_cast<std::size_t>(-1) - last_pos ||
      !MulOk(layer_span + last_pos, num_kv_heads, &slot) ||
      slot > static_cast<std::size_t>(-1) - (num_kv_heads - 1U) ||
      !MulOk(slot + (num_kv_heads - 1U), head_dim, &base) ||
      base > static_cast<std::size_t>(-1) - (head_dim - 1U) ||
      base + (head_dim - 1U) >= cache_elems) {
    return {false, "glimmer wmma prefill cache index out of range"};
  }
  // The WMMA launcher takes void* caches but never writes them when
  // skip_kv_write is true, so the const_cast is sound.
  const bool ok = LaunchQwenWmmaAttention(
      q, /*k=*/nullptr, /*v=*/nullptr, gate, /*k_cache=*/nullptr,
      /*v_cache=*/nullptr, const_cast<void*>(k_f16), const_cast<void*>(v_f16),
      out, layer, start_pos, batch, max_context, num_heads, num_kv_heads,
      head_dim, stream, /*lse_out=*/nullptr, /*key_begin=*/0,
      /*skip_kv_write=*/true, packed_k_workspace, packed_v_workspace, window);
  if (!ok) {
    return {false, "glimmer wmma prefill refused by the dispatcher"};
  }
  if (hipGetLastError() != hipSuccess) {
    return {false, "glimmer wmma prefill launch failed"};
  }
  return {true, nullptr};
}

}  // namespace gufo::hip
#endif
