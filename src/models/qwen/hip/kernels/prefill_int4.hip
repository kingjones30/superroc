#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/ops/prefill_int4.hpp"

#include <hip/hip_fp16.h>

#include "src/core/hip_launch_check.hpp"

// Waves exchange data through LDS only, so a barrier needs no global cache invalidate.
__device__ __forceinline__ __attribute__((convergent)) void Int4LdsBarrier() {
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_s_barrier();
  asm volatile("" ::: "memory");
}

namespace superroc::hip {
namespace {
__device__ __forceinline__ __half Int4RoundActivation(float value) {
  asm volatile("" : "+v"(value));
  return __float2half_rn(value);
}
__device__ __forceinline__ std::uint64_t Int4Nibble(float v, float step) {
  const float q = fminf(fmaxf(floorf(v / step + 0.5F), -7.F), 7.F);
  return static_cast<std::uint64_t>(static_cast<std::uint32_t>(static_cast<int>(q)) & 15U);
}
// Deterministic sum over the 256 threads of a group.
__device__ __forceinline__ float Int4Reduce(float* sums, std::uint32_t tid, float sum) {
  sums[tid] = sum;
  Int4LdsBarrier();
  for (std::uint32_t stride = 128; stride != 0U; stride /= 2U) {
    if (tid < stride)
      sums[tid] += sums[tid + stride];
    Int4LdsBarrier();
  }
  return sums[0];
}

__global__ void Int4PackHalf(const __half* __restrict__ x, std::uint8_t* __restrict__ out,
                             float* __restrict__ token_scale, const std::uint32_t* __restrict__ order,
                             const float* __restrict__ inverse, float base, std::uint32_t k_in, std::uint32_t k_out,
                             float clip) {
  __shared__ float sums[256];
  const std::uint32_t t = blockIdx.x, tid = threadIdx.x, steps = k_out / 16U;
  const __half* src = x + static_cast<std::size_t>(t) * k_in;
  auto* dst = reinterpret_cast<std::uint64_t*>(out + static_cast<std::size_t>(t) * (k_out / 2U));
  float sum = 0.F;
  for (std::uint32_t s = tid; s < steps; s += 256U) {
    for (std::uint32_t j = 0; j < 16U; ++j) {
      const std::uint32_t p = s * 16U + j, entry = order[p];
      if (entry <= kInt4ColumnMask) {
        const float v = __half2float(src[entry]) * inverse[p];
        sum += v * v;
      }
    }
  }
  const float step = fmaxf(sqrtf(Int4Reduce(sums, tid, sum) / static_cast<float>(k_in)) * clip / 7.F, 1e-12F);
  if (tid == 0U)
    token_scale[t] = step * base;
  for (std::uint32_t s = tid; s < steps; s += 256U) {
    std::uint64_t packed = 0U;
    for (std::uint32_t j = 0; j < 16U; ++j) {
      const std::uint32_t p = s * 16U + j, entry = order[p];
      if (entry == kInt4NoColumn)
        continue;
      const float v = __half2float(src[entry & kInt4ColumnMask]) * inverse[p];
      if ((entry & kInt4FineInput) != 0U) {
        const float coarse = fminf(fmaxf(floorf(v / step + 0.5F), -7.F), 7.F);
        packed |= Int4Nibble(v - coarse * step, step / 16.F) << (4U * j);
      } else {
        packed |= Int4Nibble(v, step) << (4U * j);
      }
    }
    dst[s] = packed;
  }
}

using Int4V2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using Int4V8 = __attribute__((__vector_size__(8 * sizeof(int)))) int;

enum class Int4Mode { kPair, kStore };

// kPair: even 16-row fragments are gate and odd ones up, for the same logical rows, with
// SwiGLU applied and half floats written. kStore: one matrix, floats written.
// Row tiles run in groups so one token tile serves several rows before it leaves the cache.
template <Int4Mode Mode, int WM, int WN, int BK>
__launch_bounds__(WM* WN * 32) __global__ void Int4GemmKernel(
    const std::uint8_t* __restrict__ wg, const std::uint8_t* __restrict__ wu, const std::uint8_t* __restrict__ x,
    void* __restrict__ out, const float* __restrict__ rg, const float* __restrict__ ru, const float* __restrict__ ts,
    Int4Shifts shifts, int batch, int m, int k) {
  constexpr bool kPair = Mode == Int4Mode::kPair;
  constexpr int BM = 256, BN = 256;
  constexpr int kThreads = WM * WN * 32;
  constexpr int RS = BM / 16, TS = BN / 16, WRS = RS / WM, WTS = TS / WN;
  constexpr int PFA = (BM * BK + kThreads - 1) / kThreads, PFB = (BN * BK + kThreads - 1) / kThreads;
  static_assert(!kPair || WRS % 2 == 0);
  __shared__ __attribute__((aligned(16))) std::uint64_t s_a[BK][RS][16];
  __shared__ __attribute__((aligned(16))) std::uint64_t s_b[BK][TS][16];
  const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, sl = lane & 15, wr = wave / WN, wt = wave % WN;
  constexpr unsigned kGroup = 8U;
  const unsigned first = (blockIdx.y / kGroup) * kGroup, within = (blockIdx.y - first) * gridDim.x + blockIdx.x;
  const unsigned rows_in_group = min(kGroup, gridDim.y - first);
  const int r_block = (first + within % rows_in_group) * (kPair ? BM / 2 : BM), t_block = (within / rows_in_group) * BN;
  const std::size_t bytes = k / 2;
  const int ksteps = k / 16;
  Int4V8 acc[WRS][WTS] = {};
  const std::uint8_t* wr_p[PFA];
  std::uint64_t ra[PFA], rb[PFB];
#pragma unroll
  for (int p = 0; p < PFA; ++p) {
    const int physical = (p * kThreads + tid) / BK;
    if constexpr (kPair) {
      const int r = r_block + (physical / 32) * 16 + physical % 16;
      wr_p[p] = (physical % 32 >= 16 ? wu : wg) + static_cast<std::size_t>(r < m ? r : m - 1) * bytes;
    } else {
      const int r = r_block + physical;
      wr_p[p] = wg + static_cast<std::size_t>(r < m ? r : m - 1) * bytes;
    }
  }
  const std::uint8_t* xr_p[PFB];
#pragma unroll
  for (int p = 0; p < PFB; ++p) {
    const int t = t_block + (p * kThreads + tid) / BK;
    xr_p[p] = x + static_cast<std::size_t>(t < batch ? t : batch - 1) * bytes;
  }
  const auto fetch = [&](int ks0) {
#pragma unroll
    for (int p = 0; p < PFA; ++p) {
      const int ks = (p * kThreads + tid) % BK;
      ra[p] = *reinterpret_cast<const std::uint64_t*>(wr_p[p] + static_cast<std::size_t>(ks0 + ks) * 8U);
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int ks = (p * kThreads + tid) % BK;
      rb[p] = *reinterpret_cast<const std::uint64_t*>(xr_p[p] + static_cast<std::size_t>(ks0 + ks) * 8U);
    }
  };
  fetch(0);
  const bool live = t_block + wt * WTS * 16 < batch;
  int next_shift = 0;
  for (int ks0 = 0; ks0 < ksteps; ks0 += BK) {
#pragma unroll
    for (int p = 0; p < PFA; ++p) {
      const int idx = p * kThreads + tid, row = idx / BK, ks = idx % BK;
      if (idx < BM * BK)
        s_a[ks][row / 16][row % 16] = ra[p];
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = p * kThreads + tid, row = idx / BK, ks = idx % BK;
      if (idx < BN * BK)
        s_b[ks][row / 16][row % 16] = rb[p];
    }
    Int4LdsBarrier();
    if (ks0 + BK < ksteps)
      fetch(ks0 + BK);
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int ks = 0; ks < BK && live; ++ks) {
      if (next_shift < shifts.count && ks0 + ks == shifts.at[next_shift]) {
        const int bits = shifts.bits[next_shift++];
#pragma unroll
        for (int i = 0; i < WRS; ++i) {
#pragma unroll
          for (int j = 0; j < WTS; ++j)
            acc[i][j] <<= bits;
        }
      }
      Int4V2 a[WRS];
#pragma unroll
      for (int i = 0; i < WRS; ++i)
        a[i] = *reinterpret_cast<const Int4V2*>(&s_a[ks][wr * WRS + i][sl]);
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const Int4V2 b = *reinterpret_cast<const Int4V2*>(&s_b[ks][wt * WTS + j][sl]);
#pragma unroll
        for (int i = 0; i < WRS; ++i)
          acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, b, true, a[i], acc[i][j], false);
        __builtin_amdgcn_sched_barrier(0);
      }
    }
    Int4LdsBarrier();
  }
  const int hi = lane / 16;
#pragma unroll
  for (int i = 0; i < WRS; i += (kPair ? 2 : 1)) {
#pragma unroll
    for (int j = 0; j < WTS; ++j) {
      const int row = r_block + (kPair ? wr * (WRS / 2) + i / 2 : wr * WRS + i) * 16 + sl;
      const int t0 = t_block + (wt * WTS + j) * 16;
      if (row >= m)
        continue;
      const float sg = rg[row];
      const float su = kPair ? ru[row] : 0.F;
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        const int token = t0 + 2 * l + hi;
        if (token >= batch)
          continue;
        const std::size_t index = static_cast<std::size_t>(token) * m + row;
        const float t_scale = ts[token];
        const float g = static_cast<float>(acc[i][j][l]) * sg * t_scale;
        if constexpr (kPair) {
          const float u = static_cast<float>(acc[i + 1][j][l]) * su * t_scale;
          const float sig = 1.0F / (1.0F + expf(-g));
          static_cast<__half*>(out)[index] = Int4RoundActivation(__fmul_rn(g * sig, u));
        } else {
          static_cast<float*>(out)[index] = g;
        }
      }
    }
  }
}
}  // namespace

void LaunchInt4PackHalf(const void* x, std::uint8_t* out, float* token_scale, const std::uint32_t* order,
                        const float* inverse, float base, std::size_t batch, std::size_t k_in, std::size_t k_out,
                        float clip, hipStream_t stream) {
  Int4PackHalf<<<batch, 256, 0, stream>>>(static_cast<const __half*>(x), out, token_scale, order, inverse, base,
                                          static_cast<std::uint32_t>(k_in), static_cast<std::uint32_t>(k_out), clip);
  SUPERROC_CHECK_LAUNCH("Int4PackHalf");
}
void LaunchInt4GemmPair(const std::uint8_t* gate, const std::uint8_t* up, const std::uint8_t* x,
                        const float* gate_scale, const float* up_scale, const float* token_scale,
                        const Int4Shifts& shifts, void* y, std::size_t batch, std::size_t m, std::size_t k_out,
                        hipStream_t stream) {
  Int4GemmKernel<Int4Mode::kPair, 8, 4, kInt4BlockSteps><<<dim3((batch + 255) / 256, (m + 127) / 128), 1024, 0, stream>>>(
      gate, up, x, y, gate_scale, up_scale, token_scale, shifts, static_cast<int>(batch), static_cast<int>(m),
      static_cast<int>(k_out));
  SUPERROC_CHECK_LAUNCH("Int4GemmPair");
}
void LaunchInt4GemmStore(const std::uint8_t* w, const std::uint8_t* x, const float* row_scale, const float* token_scale,
                         const Int4Shifts& shifts, float* y, std::size_t batch, std::size_t m, std::size_t k_out,
                         hipStream_t stream) {
  Int4GemmKernel<Int4Mode::kStore, 8, 4, kInt4BlockSteps><<<dim3((batch + 255) / 256, (m + 255) / 256), 1024, 0, stream>>>(
      w, w, x, y, row_scale, row_scale, token_scale, shifts, static_cast<int>(batch), static_cast<int>(m),
      static_cast<int>(k_out));
  SUPERROC_CHECK_LAUNCH("Int4GemmStore");
}
}  // namespace superroc::hip
#endif
