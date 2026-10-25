#if defined(ENGINE_ENABLE_HIP)
#if defined(LQW_Q4_SMOKE)
#include "q4_0_gemv.hpp"
#else
#include "src/models/qwen/hip/kernels/q4_0_gemv.hpp"
#include "src/core/hip_launch_check.hpp"
#endif

namespace gufo::hip {
namespace {

__device__ inline float Q4_0DotBlock(const std::uint8_t* block, const float* x) {
  const float scale = Q4_0Scale(block);
  float dot = 0.0F;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const unsigned byte = block[2 + j];
    const int lo = static_cast<int>(byte & 0x0FU) - 8;
    const int hi = static_cast<int>(byte >> 4) - 8;
    dot += static_cast<float>(lo) * x[j] + static_cast<float>(hi) * x[j + 16];
  }
  return scale * dot;
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Q4_0GemvKernel(const std::uint8_t* __restrict__ weights,
                        const float* __restrict__ x, float* __restrict__ y,
                        std::size_t rows, std::size_t cols) {
  const std::size_t lane = threadIdx.x & 31U;
  const std::size_t warp = threadIdx.x >> 5U;
  const std::size_t row =
      (static_cast<std::size_t>(blockIdx.x) * WavesPerBlock) + warp;
  if (row >= rows || (cols % 32) != 0) {
    return;
  }
  const std::size_t blocks = cols / 32;
  const std::uint8_t* base = weights + row * blocks * 18;
  float sum = 0.0F;
  for (std::size_t b = lane; b < blocks; b += 32) {
    sum += Q4_0DotBlock(base + b * 18, x + b * 32);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }
  if (lane == 0) {
    y[row] = sum;
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Q4_0SwigluKernel(const std::uint8_t* __restrict__ gate,
                          const std::uint8_t* __restrict__ up,
                          const float* __restrict__ x, float* __restrict__ out,
                          std::size_t rows, std::size_t cols) {
  const std::size_t lane = threadIdx.x & 31U;
  const std::size_t warp = threadIdx.x >> 5U;
  const std::size_t row =
      (static_cast<std::size_t>(blockIdx.x) * WavesPerBlock) + warp;
  if (row >= rows || (cols % 32) != 0) {
    return;
  }
  const std::size_t blocks = cols / 32;
  const std::size_t stride = blocks * 18;
  const std::uint8_t* gate_row = gate + row * stride;
  const std::uint8_t* up_row = up + row * stride;
  float gate_sum = 0.0F;
  float up_sum = 0.0F;
  for (std::size_t b = lane; b < blocks; b += 32) {
    const float* xb = x + b * 32;
    gate_sum += Q4_0DotBlock(gate_row + b * 18, xb);
    up_sum += Q4_0DotBlock(up_row + b * 18, xb);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    gate_sum += __shfl_xor(gate_sum, offset);
    up_sum += __shfl_xor(up_sum, offset);
  }
  if (lane == 0) {
    const float sig = 1.0F / (1.0F + expf(-gate_sum));
    out[row] = (gate_sum * sig) * up_sum;
  }
}

__device__ inline std::uint16_t FloatToBf16Rne(float value) {
  union {
    float f;
    std::uint32_t u;
  } cast{};
  cast.f = value;
  std::uint32_t bits = cast.u;
  const std::uint32_t rounding_bias = 0x7FFFU + ((bits >> 16U) & 1U);
  bits += rounding_bias;
  return static_cast<std::uint16_t>(bits >> 16U);
}

__global__ void DequantQ4_0ToBf16Kernel(const std::uint8_t* __restrict__ src,
                                        std::uint16_t* __restrict__ dst,
                                        std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  dst[index] = FloatToBf16Rne(Q4_0At(src, index));
}

}  // namespace

void LaunchQ4_0Gemv(const void* weights, const float* x, float* y,
                    std::size_t rows, std::size_t cols, hipStream_t stream) {
  constexpr std::uint32_t kWaves = 4;
  const std::size_t grid = (rows + kWaves - 1) / kWaves;
  Q4_0GemvKernel<kWaves><<<grid, kWaves * 32, 0, stream>>>(
      static_cast<const std::uint8_t*>(weights), x, y, rows, cols);
  GUFO_CHECK_LAUNCH("Q4_0GemvKernel<kWaves>");
}

void LaunchQ4_0Swiglu(const void* gate, const void* up, const float* x,
                      float* out, std::size_t rows, std::size_t cols,
                      hipStream_t stream) {
  constexpr std::uint32_t kWaves = 4;
  const std::size_t grid = (rows + kWaves - 1) / kWaves;
  Q4_0SwigluKernel<kWaves><<<grid, kWaves * 32, 0, stream>>>(
      static_cast<const std::uint8_t*>(gate),
      static_cast<const std::uint8_t*>(up), x, out, rows, cols);
  GUFO_CHECK_LAUNCH("Q4_0SwigluKernel<kWaves>");
}

void LaunchDequantQ4_0ToBf16(const void* src, void* dst, std::size_t elements,
                             hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  const std::size_t grid = (elements + kThreads - 1) / kThreads;
  DequantQ4_0ToBf16Kernel<<<grid, kThreads, 0, stream>>>(
      static_cast<const std::uint8_t*>(src), static_cast<std::uint16_t*>(dst),
      elements);
  GUFO_CHECK_LAUNCH("DequantQ4_0ToBf16Kernel");
}

}  // namespace gufo::hip
#endif
