#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/core/hip_launch_check.hpp"

namespace superroc::hip {

// One block per row. Thread t owns columns t, t+blockDim, ... of the hidden
// width for both halves, loads each element exactly once into registers,
// block-reduces the two row sums of squares, then writes
// fusion[row, 0:H] = e/rms_e * enorm and fusion[row, H:2H] = h/rms_h * hnorm.
__global__ void MtpFusionNormsKernel(const float* __restrict__ embed_rows,
                                     const float* __restrict__ h_rows,
                                     const float* __restrict__ enorm,
                                     const float* __restrict__ hnorm,
                                     float* __restrict__ fusion,
                                     std::size_t hidden, float eps) {
  const std::size_t row = blockIdx.x;
  const float* e_in = embed_rows + row * hidden;
  const float* h_in = h_rows + row * hidden;
  float* out = fusion + row * 2 * hidden;

  float e_sq = 0.0F;
  float h_sq = 0.0F;
  const std::size_t span =
      (hidden + blockDim.x - 1) / blockDim.x;  // elements per thread per half
  float e_cache[32];
  float h_cache[32];
  for (std::size_t i = 0; i < span; ++i) {
    const std::size_t col = threadIdx.x + i * blockDim.x;
    if (col < hidden) {
      const float e = e_in[col];
      const float h = h_in[col];
      e_cache[i] = e;
      h_cache[i] = h;
      e_sq += e * e;
      h_sq += h * h;
    }
  }

  __shared__ float s_e[1024];
  __shared__ float s_h[1024];
  s_e[threadIdx.x] = e_sq;
  s_h[threadIdx.x] = h_sq;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_e[threadIdx.x] += s_e[threadIdx.x + stride];
      s_h[threadIdx.x] += s_h[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float e_scale =
      1.0F / sqrtf(s_e[0] / static_cast<float>(hidden) + eps);
  const float h_scale =
      1.0F / sqrtf(s_h[0] / static_cast<float>(hidden) + eps);
  for (std::size_t i = 0; i < span; ++i) {
    const std::size_t col = threadIdx.x + i * blockDim.x;
    if (col < hidden) {
      out[col] = e_cache[i] * e_scale * enorm[col];
      out[hidden + col] = h_cache[i] * h_scale * hnorm[col];
    }
  }
}

void LaunchMtpFusionNorms(const float* embed_rows, const float* h_rows,
                          const float* enorm, const float* hnorm,
                          float* fusion, std::size_t batch_size,
                          std::size_t hidden, float eps, hipStream_t stream) {
  // e_cache/h_cache hold `span` elements per thread; 32 covers
  // hidden <= 32 * 1024 = 32768.
  const std::uint32_t threads = 1024;
  const std::size_t span = (hidden + threads - 1) / threads;
  if (span > 32) {
    throw std::runtime_error("MTP fusion norm hidden width too large");
  }
  MtpFusionNormsKernel<<<static_cast<std::uint32_t>(batch_size), threads, 0,
                         stream>>>(embed_rows, h_rows, enorm, hnorm, fusion,
                                   hidden, eps);
  SUPERROC_CHECK_LAUNCH("MtpFusionNormsKernel");
}

// Probability of the drafted token under the draft head: one block sums
// exp(logit - peak) over the scored rows, where peak is the logit at the token
// the argmax kernel just wrote. The peak's own term is 1, so the sum is at
// least 1 and the result is in (0, 1]. Logits that are not finite add nothing.
__global__ void MtpTop1ProbabilityKernel(const float* __restrict__ logits,
                                         std::uint32_t rows,
                                         const std::uint32_t* __restrict__ token,
                                         float* __restrict__ probability) {
  const std::uint32_t peak_row = token[0] < rows ? token[0] : 0U;
  const float peak = logits[peak_row];
  float sum = 0.0F;
  for (std::uint32_t row = threadIdx.x; row < rows; row += blockDim.x) {
    const float value = logits[row];
    if (isfinite(value)) {
      sum += expf(value - peak);
    }
  }
  __shared__ float s_sum[1024];
  s_sum[threadIdx.x] = sum;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float total = s_sum[0];
    probability[0] = (isfinite(peak) && total >= 1.0F) ? 1.0F / total : 0.0F;
  }
}

void LaunchMtpTop1Probability(const float* logits, std::size_t rows,
                              const std::uint32_t* token, float* probability,
                              hipStream_t stream) {
  if (rows == 0 || rows > 0xFFFFFFFFULL) {
    throw std::runtime_error("MTP draft probability row count is invalid");
  }
  MtpTop1ProbabilityKernel<<<1, 1024, 0, stream>>>(
      logits, static_cast<std::uint32_t>(rows), token, probability);
  SUPERROC_CHECK_LAUNCH("MtpTop1ProbabilityKernel");
}

}  // namespace superroc::hip
#endif  // defined(ENGINE_ENABLE_HIP)
