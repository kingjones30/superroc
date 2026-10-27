// lqw:spec5-q4. Declarations plus the device element helper. Kernel bodies
// live in q4_0_gemv.hip and are compiled into one translation unit.
#pragma once

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace superroc::hip {

__device__ inline float Q4_0Scale(const std::uint8_t* block) {
  __half h;
  __builtin_memcpy(&h, block, sizeof(h));
  return __half2float(h);
}

// ggml block_q4_0: low nibble is element j, high nibble is element j+16,
// both biased by 8. `element` is the linear index in a tightly packed matrix.
__device__ inline float Q4_0At(const std::uint8_t* table, std::size_t element) {
  const std::size_t block_index = element / 32;
  const unsigned j = static_cast<unsigned>(element % 32);
  const std::uint8_t* block = table + block_index * 18;
  const unsigned byte = block[2 + (j & 15U)];
  const int nibble =
      (j < 16U) ? static_cast<int>(byte & 0x0FU) : static_cast<int>(byte >> 4);
  return Q4_0Scale(block) * static_cast<float>(nibble - 8);
}

void LaunchQ4_0Gemv(const void* weights, const float* x, float* y,
                    std::size_t rows, std::size_t cols, hipStream_t stream);

void LaunchQ4_0Swiglu(const void* gate, const void* up, const float* x,
                      float* out, std::size_t rows, std::size_t cols,
                      hipStream_t stream);

void LaunchDequantQ4_0ToBf16(const void* src, void* dst, std::size_t elements,
                             hipStream_t stream);

}  // namespace superroc::hip
#endif
