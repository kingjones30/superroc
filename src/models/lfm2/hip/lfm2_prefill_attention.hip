// LFM2 tiled fp32 flash-attention prefill for head_dim 64 / GQA (SPEC2, lane LFS). // LFM2
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/lfm2/hip/lfm2_prefill_attention.hpp"  // LFM2

#include <hip/hip_fp16.h>  // LFM2 (SPEC3)
#include <hip/hip_runtime.h>  // LFM2

#include <cmath>  // LFM2
#include <cstddef>  // LFM2
#include <cstdint>  // LFM2

namespace gufo::hip {  // LFM2
namespace {  // LFM2

// LFM2: fixed geometry (SPEC2 step 1): head_dim 64, group 4, 64 rows x 4 // LFM2
// heads per block (256 threads), key tiles of 32 positions. // LFM2
constexpr std::uint32_t kLfm2AttnHeadDim = 64U;  // LFM2
constexpr std::uint32_t kLfm2AttnRowsPerBlock = 64U;  // LFM2
constexpr std::uint32_t kLfm2AttnGroup = 4U;  // LFM2
constexpr std::uint32_t kLfm2AttnThreads = 256U;  // LFM2
constexpr std::uint32_t kLfm2AttnKeyTile = 32U;  // LFM2
constexpr float kLfm2AttnQScale = 0.125F;  // LFM2 (1/sqrt(64))
constexpr float kLfm2AttnNegInf = -1.0e30F;  // LFM2

Lfm2LaunchStatus Lfm2AttnOk() noexcept { return {true, nullptr}; }  // LFM2

Lfm2LaunchStatus Lfm2AttnBad(const char* error) noexcept { return {false, error}; }  // LFM2

// LFM2: report an async launch failure under a static string. // LFM2
Lfm2LaunchStatus Lfm2AttnCheckLaunch(const char* what) noexcept {  // LFM2
  if (hipGetLastError() != hipSuccess) {  // LFM2
    return Lfm2AttnBad(what);  // LFM2
  }  // LFM2
  return Lfm2AttnOk();  // LFM2
}  // LFM2

// LFM2 (SPEC3): one KV element as fp32 (fp32 cache or fp16 bits). // LFM2
__device__ __forceinline__ float Lfm2PrefillKvLoad(const float* p,  // LFM2
                                                   std::size_t i) {  // LFM2
  return p[i];  // LFM2
}  // LFM2
__device__ __forceinline__ float Lfm2PrefillKvLoad(const std::uint16_t* p,  // LFM2
                                                   std::size_t i) {  // LFM2
  return __half2float(__ushort_as_half(p[i]));  // LFM2
}  // LFM2

// HKV (RVW2 V1): saturating fp16 store, same rule as the decode write // LFM2
// (+-65504 instead of +-inf; NaN stays NaN; in-range values unchanged). // LFM2
__device__ __forceinline__ std::uint16_t Lfm2F16SatPrefill(float x) {  // LFM2
  const float c = x != x ? x : fminf(fmaxf(x, -65504.0F), 65504.0F);  // LFM2
  return __half_as_ushort(__float2half(c));  // LFM2
}  // LFM2

// LFM2 (SPEC3): row r of k/v [batch][kv] -> token-major fp16 row // LFM2
// (layer*max_ctx + start_pos + r), round to nearest even. // LFM2
__global__ void Lfm2BatchedWriteKvF16Kernel(  // LFM2
    const float* __restrict__ k, const float* __restrict__ v,  // LFM2
    std::uint16_t* __restrict__ k_cache, std::uint16_t* __restrict__ v_cache,  // LFM2
    std::uint32_t layer, std::uint32_t start_pos, std::uint32_t max_ctx,  // LFM2
    std::uint32_t kv) {  // LFM2
  const std::uint32_t r = blockIdx.x;  // LFM2
  const std::size_t src_row = static_cast<std::size_t>(r) * kv;  // LFM2
  const std::size_t dst_row =  // LFM2
      (static_cast<std::size_t>(layer) * max_ctx + start_pos + r) * kv;  // LFM2
  for (std::uint32_t i = threadIdx.x; i < kv; i += blockDim.x) {  // LFM2
    k_cache[dst_row + i] = Lfm2F16SatPrefill(k[src_row + i]);  // LFM2
    v_cache[dst_row + i] = Lfm2F16SatPrefill(v[src_row + i]);  // LFM2
  }  // LFM2
}  // LFM2

// LFM2: grid (ceil(batch/64), kv_heads), block 256 = 64 rows x 4 heads: // LFM2
// thread t serves row r0 + t/4, head kvh*4 + t%4. Rows past batch join the // LFM2
// barriers but compute and write nothing. // LFM2
// LFM2 (SPEC3): KvT float = fp32 head-major; uint16 + kTokenMajor = fp16 // LFM2
// token-major. // LFM2
template <typename KvT, bool kTokenMajor>  // LFM2
__global__ void Lfm2PrefillAttentionKernel(  // LFM2
    const float* __restrict__ q, const KvT* __restrict__ k_cache,  // LFM2
    const KvT* __restrict__ v_cache, float* __restrict__ out,  // LFM2
    std::uint32_t layer, std::uint32_t start_pos, std::uint32_t batch,  // LFM2
    std::uint32_t max_ctx, std::uint32_t heads, std::uint32_t kv_heads) {  // LFM2
  const std::uint32_t r0 = blockIdx.x * kLfm2AttnRowsPerBlock;  // LFM2
  const std::uint32_t kvh = blockIdx.y;  // LFM2
  const std::uint32_t t = threadIdx.x;  // LFM2
  const std::uint32_t row = r0 + t / kLfm2AttnGroup;  // LFM2
  const std::uint32_t h = kvh * kLfm2AttnGroup + t % kLfm2AttnGroup;  // LFM2
  const bool active = row < batch && h < heads;  // LFM2
  const std::uint32_t last_rel =  // LFM2
      r0 + kLfm2AttnRowsPerBlock - 1U < batch - 1U  // LFM2
          ? r0 + kLfm2AttnRowsPerBlock - 1U  // LFM2
          : batch - 1U;  // LFM2
  const std::uint32_t last_abs = start_pos + last_rel;  // LFM2
  const std::uint32_t my_pos = start_pos + row;  // LFM2
  float qreg[kLfm2AttnHeadDim];  // LFM2
  if (active) {  // LFM2
    const float* qr =  // LFM2
        q + (static_cast<std::size_t>(row) * heads + h) * kLfm2AttnHeadDim;  // LFM2
    for (std::uint32_t d = 0; d < kLfm2AttnHeadDim; ++d) {  // LFM2
      qreg[d] = qr[d] * kLfm2AttnQScale;  // LFM2
    }  // LFM2
  }  // LFM2
  __shared__ float sK[kLfm2AttnKeyTile][kLfm2AttnHeadDim];  // LFM2 (8 KiB)
  __shared__ float sV[kLfm2AttnKeyTile][kLfm2AttnHeadDim];  // LFM2 (8 KiB)
  float acc[kLfm2AttnHeadDim];  // LFM2
  for (std::uint32_t d = 0; d < kLfm2AttnHeadDim; ++d) {  // LFM2
    acc[d] = 0.0F;  // LFM2
  }  // LFM2
  float m = kLfm2AttnNegInf;  // LFM2 (running max)
  float l = 0.0F;  // LFM2 (running sum)
  const std::size_t kv_base =  // LFM2
      (static_cast<std::size_t>(layer) * kv_heads + kvh) * max_ctx;  // LFM2
  for (std::uint32_t t0 = 0; t0 <= last_abs; t0 += kLfm2AttnKeyTile) {  // LFM2
    for (std::uint32_t i = t; i < kLfm2AttnKeyTile * kLfm2AttnHeadDim;  // LFM2
         i += kLfm2AttnThreads) {  // LFM2
      const std::uint32_t j = i / kLfm2AttnHeadDim;  // LFM2
      const std::uint32_t d = i % kLfm2AttnHeadDim;  // LFM2
      const std::uint32_t p = t0 + j;  // LFM2
      float kk = 0.0F;  // LFM2
      float vv = 0.0F;  // LFM2
      if (p < max_ctx) {  // LFM2
        const std::size_t idx =  // LFM2
            kTokenMajor  // LFM2 (SPEC3)
                ? ((static_cast<std::size_t>(layer) * max_ctx + p) * kv_heads +  // LFM2
                   kvh) * kLfm2AttnHeadDim + d  // LFM2
                : (kv_base + p) * kLfm2AttnHeadDim + d;  // LFM2
        kk = Lfm2PrefillKvLoad(k_cache, idx);  // LFM2
        vv = Lfm2PrefillKvLoad(v_cache, idx);  // LFM2
      }  // LFM2
      sK[j][d] = kk;  // LFM2
      sV[j][d] = vv;  // LFM2
    }  // LFM2
    __syncthreads();  // LFM2
    if (active) {  // LFM2
      for (std::uint32_t j = 0; j < kLfm2AttnKeyTile; ++j) {  // LFM2
        const std::uint32_t p = t0 + j;  // LFM2
        if (p > my_pos) {  // LFM2 (causal; later positions are -inf)
          continue;  // LFM2
        }  // LFM2
        float s = 0.0F;  // LFM2
        for (std::uint32_t d = 0; d < kLfm2AttnHeadDim; ++d) {  // LFM2
          s += qreg[d] * sK[j][d];  // LFM2
        }  // LFM2
        const float m_new = s > m ? s : m;  // LFM2
        const float rescale = expf(m - m_new);  // LFM2
        const float w = expf(s - m_new);  // LFM2
        for (std::uint32_t d = 0; d < kLfm2AttnHeadDim; ++d) {  // LFM2
          acc[d] = acc[d] * rescale + w * sV[j][d];  // LFM2
        }  // LFM2
        l = l * rescale + w;  // LFM2
        m = m_new;  // LFM2
      }  // LFM2
    }  // LFM2
    __syncthreads();  // LFM2
  }  // LFM2
  if (active) {  // LFM2
    const float inv = 1.0F / l;  // LFM2
    float* out_row =  // LFM2
        out + (static_cast<std::size_t>(row) * heads + h) * kLfm2AttnHeadDim;  // LFM2
    for (std::uint32_t d = 0; d < kLfm2AttnHeadDim; ++d) {  // LFM2
      out_row[d] = acc[d] * inv;  // LFM2
    }  // LFM2
  }  // LFM2
}  // LFM2

template <typename KvT, bool kTokenMajor>  // LFM2 (SPEC3)
Lfm2LaunchStatus LaunchLfm2PrefillAttentionT(  // LFM2
    const float* q, const KvT* k_cache, const KvT* v_cache, float* out,  // LFM2
    std::uint32_t layer, std::uint32_t start_pos, std::uint32_t batch,  // LFM2
    std::uint32_t max_ctx, std::uint32_t heads, std::uint32_t kv_heads,  // LFM2
    hipStream_t stream) {  // LFM2
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||  // LFM2
      out == nullptr) {  // LFM2
    return Lfm2AttnBad("lfm2 prefill attention refused: null pointer");  // LFM2
  }  // LFM2
  if (heads == 0 || kv_heads == 0 || (heads % kv_heads) != 0 ||  // LFM2
      heads / kv_heads != kLfm2AttnGroup) {  // LFM2
    return Lfm2AttnBad("lfm2 prefill attention refused: group");  // LFM2
  }  // LFM2
  if (batch == 0 || start_pos >= max_ctx || batch > max_ctx - start_pos) {  // LFM2
    return Lfm2AttnBad("lfm2 prefill attention refused: range");  // LFM2
  }  // LFM2
  const std::uint32_t grid_x =  // LFM2
      (batch + kLfm2AttnRowsPerBlock - 1U) / kLfm2AttnRowsPerBlock;  // LFM2
  const dim3 grid(grid_x, kv_heads);  // LFM2
  Lfm2PrefillAttentionKernel<KvT, kTokenMajor>  // LFM2
      <<<grid, kLfm2AttnThreads, 0, stream>>>(  // LFM2
          q, k_cache, v_cache, out, layer, start_pos, batch, max_ctx, heads,  // LFM2
          kv_heads);  // LFM2
  return Lfm2AttnCheckLaunch("lfm2 prefill attention launch failed");  // LFM2
}  // LFM2

}  // namespace  // LFM2

Lfm2LaunchStatus LaunchLfm2PrefillAttention(  // LFM2
    const float* q, const float* k_cache, const float* v_cache, float* out,  // LFM2
    std::uint32_t layer, std::uint32_t start_pos, std::uint32_t batch,  // LFM2
    std::uint32_t max_ctx, std::uint32_t heads, std::uint32_t kv_heads,  // LFM2
    hipStream_t stream) {  // LFM2
  return LaunchLfm2PrefillAttentionT<float, false>(  // LFM2
      q, k_cache, v_cache, out, layer, start_pos, batch, max_ctx, heads,  // LFM2
      kv_heads, stream);  // LFM2
}  // LFM2

Lfm2LaunchStatus LaunchLfm2PrefillAttentionF16(  // LFM2 (SPEC3)
    const float* q, const std::uint16_t* k_cache,  // LFM2
    const std::uint16_t* v_cache, float* out, std::uint32_t layer,  // LFM2
    std::uint32_t start_pos, std::uint32_t batch, std::uint32_t max_ctx,  // LFM2
    std::uint32_t heads, std::uint32_t kv_heads, hipStream_t stream) {  // LFM2
  return LaunchLfm2PrefillAttentionT<std::uint16_t, true>(  // LFM2
      q, k_cache, v_cache, out, layer, start_pos, batch, max_ctx, heads,  // LFM2
      kv_heads, stream);  // LFM2
}  // LFM2

Lfm2LaunchStatus LaunchLfm2BatchedWriteKvF16(  // LFM2 (SPEC3)
    const float* k, const float* v, std::uint16_t* k_cache,  // LFM2
    std::uint16_t* v_cache, std::uint32_t layer, std::uint32_t start_pos,  // LFM2
    std::uint32_t batch, std::uint32_t max_ctx, std::uint32_t kv,  // LFM2
    hipStream_t stream) {  // LFM2
  if (k == nullptr || v == nullptr || k_cache == nullptr ||  // LFM2
      v_cache == nullptr) {  // LFM2
    return Lfm2AttnBad("lfm2 batched writekv f16 refused: null pointer");  // LFM2
  }  // LFM2
  if (kv == 0) {  // LFM2
    return Lfm2AttnBad("lfm2 batched writekv f16 refused: kv");  // LFM2
  }  // LFM2
  if (batch == 0 || start_pos >= max_ctx || batch > max_ctx - start_pos) {  // LFM2
    return Lfm2AttnBad("lfm2 batched writekv f16 refused: range");  // LFM2
  }  // LFM2
  Lfm2BatchedWriteKvF16Kernel<<<batch, kLfm2AttnThreads, 0, stream>>>(  // LFM2
      k, v, k_cache, v_cache, layer, start_pos, max_ctx, kv);  // LFM2
  return Lfm2AttnCheckLaunch("lfm2 batched writekv f16 launch failed");  // LFM2
}  // LFM2

}  // namespace gufo::hip  // LFM2
#endif  // defined(ENGINE_ENABLE_HIP)  // LFM2
