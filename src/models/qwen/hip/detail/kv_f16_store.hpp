#ifndef SUPERROC_MODELS_QWEN_HIP_DETAIL_KV_F16_STORE_HPP_
#define SUPERROC_MODELS_QWEN_HIP_DETAIL_KV_F16_STORE_HPP_

#include <hip/hip_fp16.h>

namespace superroc::hip {

// Super ROC QKS (Muse Wonder W7-A #1): the fp16 KV store saturates at +-65504
// instead of overflowing to +-inf. An inf key turns every later attention
// score into inf or inf - inf = NaN, and the answer is garbage from that token
// on. Below 65520 round-to-nearest already gives the same half (65504 is the
// largest finite half; 65520 is the midpoint to infinity), so every value that
// was finite before is stored bit-identically; NaN stays NaN. Same rule as the
// LFM2 store (Lfm2F16Sat, HKV).
__device__ __forceinline__ __half QwenKvF16(float x) {
  const float c = x != x ? x : fminf(fmaxf(x, -65504.0F), 65504.0F);
  return __float2half_rn(c);
}

}  // namespace superroc::hip

#endif  // SUPERROC_MODELS_QWEN_HIP_DETAIL_KV_F16_STORE_HPP_
