// LFM2 tiled fp32 flash-attention prefill for head_dim 64 / GQA.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/lfm2/hip/lfm2_prefill_attention.hpp"

#include <hip/hip_fp16.h>  // LFM2
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace superroc::hip {
namespace {

// LFM2: fixed geometry: head_dim 64, group 4, 64 rows x 4
// heads per block (256 threads), key tiles of 32 positions.
constexpr std::uint32_t kLfm2AttnHeadDim = 64U;
constexpr std::uint32_t kLfm2AttnRowsPerBlock = 64U;
constexpr std::uint32_t kLfm2AttnGroup = 4U;
constexpr std::uint32_t kLfm2AttnThreads = 256U;
constexpr std::uint32_t kLfm2AttnKeyTile = 32U;
constexpr float kLfm2AttnQScale = 0.125F;  // LFM2 (1/sqrt(64))
constexpr float kLfm2AttnNegInf = -1.0e30F;

Lfm2LaunchStatus Lfm2AttnOk() noexcept { return {true, nullptr}; }

Lfm2LaunchStatus Lfm2AttnBad(const char* error) noexcept { return {false, error}; }

// LFM2: report an async launch failure under a static string.
Lfm2LaunchStatus Lfm2AttnCheckLaunch(const char* what) noexcept {
  if (hipGetLastError() != hipSuccess) {
    return Lfm2AttnBad(what);
  }
  return Lfm2AttnOk();
}

// LFM2: one KV element as fp32 (fp32 cache or fp16 bits).
__device__ __forceinline__ float Lfm2PrefillKvLoad(const float* p,
                                                   std::size_t i) {
  return p[i];
}
__device__ __forceinline__ float Lfm2PrefillKvLoad(const std::uint16_t* p,
                                                   std::size_t i) {
  return __half2float(__ushort_as_half(p[i]));
}

// Saturating fp16 store, same rule as the decode write
// (+-65504 instead of +-inf; NaN stays NaN; in-range values unchanged).
__device__ __forceinline__ std::uint16_t Lfm2F16SatPrefill(float x) {
  const float c = x != x ? x : fminf(fmaxf(x, -65504.0F), 65504.0F);
  return __half_as_ushort(__float2half(c));
}

// LFM2: row r of k/v [batch][kv] -> token-major fp16 row
// (layer*max_ctx + start_pos + r), round to nearest even.
__global__ void Lfm2BatchedWriteKvF16Kernel(
    const float* __restrict__ k, const float* __restrict__ v,
    std::uint16_t* __restrict__ k_cache, std::uint16_t* __restrict__ v_cache,
    std::uint32_t layer, std::uint32_t start_pos, std::uint32_t max_ctx,
    std::uint32_t kv) {
  const std::uint32_t r = blockIdx.x;
  const std::size_t src_row = static_cast<std::size_t>(r) * kv;
  const std::size_t dst_row =
      (static_cast<std::size_t>(layer) * max_ctx + start_pos + r) * kv;
  for (std::uint32_t i = threadIdx.x; i < kv; i += blockDim.x) {
    k_cache[dst_row + i] = Lfm2F16SatPrefill(k[src_row + i]);
    v_cache[dst_row + i] = Lfm2F16SatPrefill(v[src_row + i]);
  }
}

// LFM2: grid (ceil(batch/64), kv_heads), block 256 = 64 rows x 4 heads:
// thread t serves row r0 + t/4, head kvh*4 + t%4. Rows past batch join the
// barriers but compute and write nothing.
// LFM2: KvT float = fp32 head-major; uint16 + kTokenMajor = fp16
// token-major.
template <typename KvT, bool kTokenMajor>
__global__ void Lfm2PrefillAttentionKernel(
    const float* __restrict__ q, const KvT* __restrict__ k_cache,
    const KvT* __restrict__ v_cache, float* __restrict__ out,
    std::uint32_t layer, std::uint32_t start_pos, std::uint32_t batch,
    std::uint32_t max_ctx, std::uint32_t heads, std::uint32_t kv_heads) {
  const std::uint32_t r0 = blockIdx.x * kLfm2AttnRowsPerBlock;
  const std::uint32_t kvh = blockIdx.y;
  const std::uint32_t t = threadIdx.x;
  const std::uint32_t row = r0 + t / kLfm2AttnGroup;
  const std::uint32_t h = kvh * kLfm2AttnGroup + t % kLfm2AttnGroup;
  const bool active = row < batch && h < heads;
  const std::uint32_t last_rel =
      r0 + kLfm2AttnRowsPerBlock - 1U < batch - 1U
          ? r0 + kLfm2AttnRowsPerBlock - 1U
          : batch - 1U;
  const std::uint32_t last_abs = start_pos + last_rel;
  const std::uint32_t my_pos = start_pos + row;
  float qreg[kLfm2AttnHeadDim];
  if (active) {
    const float* qr =
        q + (static_cast<std::size_t>(row) * heads + h) * kLfm2AttnHeadDim;
    for (std::uint32_t d = 0; d < kLfm2AttnHeadDim; ++d) {
      qreg[d] = qr[d] * kLfm2AttnQScale;
    }
  }
  __shared__ float sK[kLfm2AttnKeyTile][kLfm2AttnHeadDim];  // LFM2 (8 KiB)
  __shared__ float sV[kLfm2AttnKeyTile][kLfm2AttnHeadDim];  // LFM2 (8 KiB)
  float acc[kLfm2AttnHeadDim];
  for (std::uint32_t d = 0; d < kLfm2AttnHeadDim; ++d) {
    acc[d] = 0.0F;
  }
  float m = kLfm2AttnNegInf;  // LFM2 (running max)
  float l = 0.0F;  // LFM2 (running sum)
  const std::size_t kv_base =
      (static_cast<std::size_t>(layer) * kv_heads + kvh) * max_ctx;
  for (std::uint32_t t0 = 0; t0 <= last_abs; t0 += kLfm2AttnKeyTile) {
    for (std::uint32_t i = t; i < kLfm2AttnKeyTile * kLfm2AttnHeadDim;
         i += kLfm2AttnThreads) {
      const std::uint32_t j = i / kLfm2AttnHeadDim;
      const std::uint32_t d = i % kLfm2AttnHeadDim;
      const std::uint32_t p = t0 + j;
      float kk = 0.0F;
      float vv = 0.0F;
      if (p < max_ctx) {
        const std::size_t idx =
            kTokenMajor  // LFM2
                ? ((static_cast<std::size_t>(layer) * max_ctx + p) * kv_heads +
                   kvh) * kLfm2AttnHeadDim + d
                : (kv_base + p) * kLfm2AttnHeadDim + d;
        kk = Lfm2PrefillKvLoad(k_cache, idx);
        vv = Lfm2PrefillKvLoad(v_cache, idx);
      }
      sK[j][d] = kk;
      sV[j][d] = vv;
    }
    __syncthreads();
    if (active) {
      for (std::uint32_t j = 0; j < kLfm2AttnKeyTile; ++j) {
        const std::uint32_t p = t0 + j;
        if (p > my_pos) {  // LFM2 (causal; later positions are -inf)
          continue;
        }
        float s = 0.0F;
        for (std::uint32_t d = 0; d < kLfm2AttnHeadDim; ++d) {
          s += qreg[d] * sK[j][d];
        }
        const float m_new = s > m ? s : m;
        const float rescale = expf(m - m_new);
        const float w = expf(s - m_new);
        for (std::uint32_t d = 0; d < kLfm2AttnHeadDim; ++d) {
          acc[d] = acc[d] * rescale + w * sV[j][d];
        }
        l = l * rescale + w;
        m = m_new;
      }
    }
    __syncthreads();
  }
  if (active) {
    const float inv = 1.0F / l;
    float* out_row =
        out + (static_cast<std::size_t>(row) * heads + h) * kLfm2AttnHeadDim;
    for (std::uint32_t d = 0; d < kLfm2AttnHeadDim; ++d) {
      out_row[d] = acc[d] * inv;
    }
  }
}

template <typename KvT, bool kTokenMajor>  // LFM2
Lfm2LaunchStatus LaunchLfm2PrefillAttentionT(
    const float* q, const KvT* k_cache, const KvT* v_cache, float* out,
    std::uint32_t layer, std::uint32_t start_pos, std::uint32_t batch,
    std::uint32_t max_ctx, std::uint32_t heads, std::uint32_t kv_heads,
    hipStream_t stream) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      out == nullptr) {
    return Lfm2AttnBad("lfm2 prefill attention refused: null pointer");
  }
  if (heads == 0 || kv_heads == 0 || (heads % kv_heads) != 0 ||
      heads / kv_heads != kLfm2AttnGroup) {
    return Lfm2AttnBad("lfm2 prefill attention refused: group");
  }
  if (batch == 0 || start_pos >= max_ctx || batch > max_ctx - start_pos) {
    return Lfm2AttnBad("lfm2 prefill attention refused: range");
  }
  const std::uint32_t grid_x =
      (batch + kLfm2AttnRowsPerBlock - 1U) / kLfm2AttnRowsPerBlock;
  const dim3 grid(grid_x, kv_heads);
  Lfm2PrefillAttentionKernel<KvT, kTokenMajor>
      <<<grid, kLfm2AttnThreads, 0, stream>>>(
          q, k_cache, v_cache, out, layer, start_pos, batch, max_ctx, heads,
          kv_heads);
  return Lfm2AttnCheckLaunch("lfm2 prefill attention launch failed");
}

}  // namespace

Lfm2LaunchStatus LaunchLfm2PrefillAttention(
    const float* q, const float* k_cache, const float* v_cache, float* out,
    std::uint32_t layer, std::uint32_t start_pos, std::uint32_t batch,
    std::uint32_t max_ctx, std::uint32_t heads, std::uint32_t kv_heads,
    hipStream_t stream) {
  return LaunchLfm2PrefillAttentionT<float, false>(
      q, k_cache, v_cache, out, layer, start_pos, batch, max_ctx, heads,
      kv_heads, stream);
}

Lfm2LaunchStatus LaunchLfm2PrefillAttentionF16(  // LFM2
    const float* q, const std::uint16_t* k_cache,
    const std::uint16_t* v_cache, float* out, std::uint32_t layer,
    std::uint32_t start_pos, std::uint32_t batch, std::uint32_t max_ctx,
    std::uint32_t heads, std::uint32_t kv_heads, hipStream_t stream) {
  return LaunchLfm2PrefillAttentionT<std::uint16_t, true>(
      q, k_cache, v_cache, out, layer, start_pos, batch, max_ctx, heads,
      kv_heads, stream);
}

Lfm2LaunchStatus LaunchLfm2BatchedWriteKvF16(  // LFM2
    const float* k, const float* v, std::uint16_t* k_cache,
    std::uint16_t* v_cache, std::uint32_t layer, std::uint32_t start_pos,
    std::uint32_t batch, std::uint32_t max_ctx, std::uint32_t kv,
    hipStream_t stream) {
  if (k == nullptr || v == nullptr || k_cache == nullptr ||
      v_cache == nullptr) {
    return Lfm2AttnBad("lfm2 batched writekv f16 refused: null pointer");
  }
  if (kv == 0) {
    return Lfm2AttnBad("lfm2 batched writekv f16 refused: kv");
  }
  if (batch == 0 || start_pos >= max_ctx || batch > max_ctx - start_pos) {
    return Lfm2AttnBad("lfm2 batched writekv f16 refused: range");
  }
  Lfm2BatchedWriteKvF16Kernel<<<batch, kLfm2AttnThreads, 0, stream>>>(
      k, v, k_cache, v_cache, layer, start_pos, max_ctx, kv);
  return Lfm2AttnCheckLaunch("lfm2 batched writekv f16 launch failed");
}

}  // namespace superroc::hip
#endif  // defined(ENGINE_ENABLE_HIP)
