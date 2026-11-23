#ifndef SUPERROC_MODELS_QWEN_HIP_KERNELS_FP4_DECODE_HPP_
#define SUPERROC_MODELS_QWEN_HIP_KERNELS_FP4_DECODE_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace superroc::hip {

// Branchless ROCm-FP4 nibble decode. Same perm-LUT shape as
// Iq4NlLookup4 (quant_ops.hpp): the low 3 bits index entries 0..7/8..15,
// bit 3 selects the half. Oracle: RocmFp4Code (quant_ops.hpp) —
// nibble -> code is [0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10],
// verified entry by entry (mag3<=4 ? mag3 : 2*mag3-4, sign on bit 3).
__device__ inline std::uint32_t RocmFp4Lookup4(std::uint32_t nibbles) noexcept {
  constexpr std::uint32_t kTable0 = 0x03020100U;  // codes 0,1,2,3
  constexpr std::uint32_t kTable1 = 0x0A080604U;  // codes 4,6,8,10
  constexpr std::uint32_t kTable2 = 0xFDFEFF00U;  // codes 0,-1,-2,-3
  constexpr std::uint32_t kTable3 = 0xF6F8FAFCU;  // codes -4,-6,-8,-10
  const std::uint32_t low_index = nibbles & 0x07070707U;
  const std::uint32_t lower =
      __builtin_amdgcn_perm(kTable1, kTable0, low_index);
  const std::uint32_t upper =
      __builtin_amdgcn_perm(kTable3, kTable2, low_index);
  const std::uint32_t select = ((nibbles & 0x08080808U) >> 3U) * 0xFFU;
  return (lower & ~select) | (upper & select);
}

// Branchless Ue4m3 scale. Oracle: RocmFp4Ue4m3Half
// (quant_ops.hpp) — e > 0x7e -> 0, exp == 0 -> man/1024, else the
// (exp+119)<<23|man<<20 bits; the two branches diverge across a wave
// (every thread fetches a different sub16), so select instead. Bit-exact:
// same expressions, same order, exp+119 cannot overflow (exp <= 15).
__device__ inline float RocmFp4Ue4m3Select(std::uint8_t e) noexcept {
  const int exp = (e >> 3) & 0xF;
  const int man = e & 0x7;
  std::uint32_t normal_u = (static_cast<std::uint32_t>(exp + 119) << 23) |
                           (static_cast<std::uint32_t>(man) << 20);
  float normal;
  __builtin_memcpy(&normal, &normal_u, 4);
  const float sub = static_cast<float>(man) * (1.0F / 1024.0F);
  const float v = (exp == 0) ? sub : normal;
  return (e > 0x7e) ? 0.0F : v;
}

}  // namespace superroc::hip

#endif  // SUPERROC_MODELS_QWEN_HIP_KERNELS_FP4_DECODE_HPP_
