// Flint split-K "flash-decoding" attention (decode). Copied from
// Lfm2SplitKernelA (fp32 head-major) and Lfm2CombineKernelB; head_dim 128,
// windowed range, sinks in the combine.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/flint/hip/flint_attention.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::hip {
namespace {

constexpr std::uint32_t kCombineBlock = 128U;
constexpr std::uint32_t kTeam = 32U;
constexpr std::uint32_t kTeams = 4U;
constexpr std::uint32_t kSplitThreads = kTeam * kTeams;
constexpr std::uint32_t kMaxGroup = 8U;
constexpr std::uint32_t kFixedHeadDim = 128U;
constexpr std::uint32_t kMaxSplits = 64U;

FlintLaunchStatus Ok() noexcept { return {true, nullptr}; }

FlintLaunchStatus Bad(const char* error) noexcept { return {false, error}; }

FlintLaunchStatus CheckLaunch(const char* what) noexcept {
  if (hipGetLastError() != hipSuccess) {
    return Bad(what);
  }
  return Ok();
}

__device__ __forceinline__ void FlintKvLoad8(const float* p, float* out) {
  const float4 lo = *reinterpret_cast<const float4*>(p);
  const float4 hi = *reinterpret_cast<const float4*>(p + 4);
  out[0] = lo.x;
  out[1] = lo.y;
  out[2] = lo.z;
  out[3] = lo.w;
  out[4] = hi.x;
  out[5] = hi.y;
  out[6] = hi.z;
  out[7] = hi.w;
}

__device__ __forceinline__ std::size_t FlintKvRow(std::uint32_t layer,
                                                  std::uint32_t kvh,
                                                  std::uint32_t pos,
                                                  std::uint32_t max_context,
                                                  std::uint32_t num_kv_heads) {
  return (static_cast<std::size_t>(layer) * num_kv_heads + kvh) * max_context +
         pos;
}

__global__ void __launch_bounds__(kSplitThreads)
    FlintSplitKernelA(const float* __restrict__ q,
                      const float* __restrict__ k_cache,
                      const float* __restrict__ v_cache,
                      float* __restrict__ partial, std::uint32_t layer,
                      const std::uint32_t* __restrict__ d_pos,
                      std::uint32_t window, std::uint32_t max_context,
                      std::uint32_t num_heads, std::uint32_t num_kv_heads,
                      std::uint32_t head_dim, std::uint32_t n_splits) {
  const std::uint32_t kvh = blockIdx.x;
  const std::uint32_t s = blockIdx.y;
  if (kvh >= num_kv_heads || s >= n_splits) {
    return;
  }
  const std::uint32_t group = num_heads / num_kv_heads;
  const std::uint32_t n = *d_pos + 1U;
  const std::uint32_t begin = (window != 0U && n > window) ? n - window : 0U;
  const std::uint32_t chunk = (n - begin + n_splits - 1U) / n_splits;
  const std::uint32_t start = begin + s * chunk;
  const std::uint32_t end = start + chunk < n ? start + chunk : n;

  __shared__ float qsm[kMaxGroup][128];
  __shared__ float team_m[kTeams][kMaxGroup];
  __shared__ float team_l[kTeams][kMaxGroup];
  __shared__ float team_acc[kTeams][kMaxGroup][128];

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t team = tid / kTeam;
  const std::uint32_t lane = tid % kTeam;
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  const float* qh = q + static_cast<std::size_t>(kvh) * group * head_dim;
  for (std::uint32_t i = tid; i < group * 128U; i += blockDim.x) {
    qsm[i / 128U][i % 128U] = qh[i] * scale;
  }
  __syncthreads();

  float m[kMaxGroup];
  float l[kMaxGroup];
  float a[4][kMaxGroup];
#pragma unroll
  for (std::uint32_t g = 0; g < kMaxGroup; ++g) {
    m[g] = -INFINITY;
    l[g] = 0.0F;
#pragma unroll
    for (std::uint32_t k = 0; k < 4U; ++k) {
      a[k][g] = 0.0F;
    }
  }

  for (std::uint32_t tb = start + team * kTeam; tb < end;
       tb += kTeams * kTeam) {
    const std::uint32_t tn = end - tb < kTeam ? end - tb : kTeam;
    const bool live = lane < tn;
    float w[kMaxGroup];
#pragma unroll
    for (std::uint32_t g = 0; g < kMaxGroup; ++g) {
      w[g] = 0.0F;
    }
    if (live) {
      const float* krow =
          k_cache + FlintKvRow(layer, kvh, tb + lane, max_context,
                               num_kv_heads) *
                        128U;
      for (std::uint32_t c = 0; c < 128U; c += 8U) {
        float k8[8];
        FlintKvLoad8(krow + c, k8);
#pragma unroll
        for (std::uint32_t g = 0; g < kMaxGroup; ++g) {
          if (g < group) {
            float dot = w[g];
#pragma unroll
            for (std::uint32_t d = 0; d < 8U; ++d) {
              dot += qsm[g][c + d] * k8[d];
            }
            w[g] = dot;
          }
        }
      }
    }
#pragma unroll
    for (std::uint32_t g = 0; g < kMaxGroup; ++g) {
      if (g < group) {
        const float score = live ? w[g] : -INFINITY;
        float tmax = score;
        for (std::uint32_t off = kTeam / 2U; off > 0U; off >>= 1U) {
          tmax = fmaxf(tmax, __shfl_xor(tmax, static_cast<int>(off),
                                        static_cast<int>(kTeam)));
        }
        const float m_new = fmaxf(m[g], tmax);
        const float m_ref = m_new == -INFINITY ? 0.0F : m_new;
        const float resc = expf(m[g] - m_ref);
        const float p = live ? expf(score - m_ref) : 0.0F;
        float tsum = p;
        for (std::uint32_t off = kTeam / 2U; off > 0U; off >>= 1U) {
          tsum += __shfl_xor(tsum, static_cast<int>(off),
                             static_cast<int>(kTeam));
        }
        l[g] = l[g] * resc + tsum;
#pragma unroll
        for (std::uint32_t k = 0; k < 4U; ++k) {
          a[k][g] *= resc;
        }
        m[g] = m_new;
        w[g] = p;
      }
    }
    for (std::uint32_t j = 0; j < tn; ++j) {
      const std::size_t vbase =
          FlintKvRow(layer, kvh, tb + j, max_context, num_kv_heads) * 128U;
      float vk[4];
#pragma unroll
      for (std::uint32_t k = 0; k < 4U; ++k) {
        vk[k] = v_cache[vbase + lane + 32U * k];
      }
#pragma unroll
      for (std::uint32_t g = 0; g < kMaxGroup; ++g) {
        if (g < group) {
          const float wj =
              __shfl(w[g], static_cast<int>(j), static_cast<int>(kTeam));
#pragma unroll
          for (std::uint32_t k = 0; k < 4U; ++k) {
            a[k][g] += wj * vk[k];
          }
        }
      }
    }
  }

#pragma unroll
  for (std::uint32_t g = 0; g < kMaxGroup; ++g) {
    if (g < group) {
      if (lane == 0U) {
        team_m[team][g] = m[g];
        team_l[team][g] = l[g];
      }
#pragma unroll
      for (std::uint32_t k = 0; k < 4U; ++k) {
        team_acc[team][g][lane + 32U * k] = a[k][g];
      }
    }
  }
  __syncthreads();
  const std::uint32_t row = head_dim + 2U;
  for (std::uint32_t i = tid; i < group * row; i += blockDim.x) {
    const std::uint32_t g = i / row;
    const std::uint32_t j = i % row;
    float big = -INFINITY;
    for (std::uint32_t t = 0; t < kTeams; ++t) {
      big = fmaxf(big, team_m[t][g]);
    }
    float val = 0.0F;
    if (big == -INFINITY) {
      val = j == 0U ? -INFINITY : 0.0F;
    } else if (j == 0U) {
      val = big;
    } else {
      for (std::uint32_t t = 0; t < kTeams; ++t) {
        const float e = expf(team_m[t][g] - big);
        val += e * (j == 1U ? team_l[t][g] : team_acc[t][g][j - 2U]);
      }
    }
    const std::uint32_t h = kvh * group + g;
    partial[(static_cast<std::size_t>(h) * n_splits + s) * row + j] = val;
  }
}

__global__ void FlintCombineKernelB(const float* __restrict__ partial,
                                    float* __restrict__ out,
                                    const float* __restrict__ sinks,
                                    std::uint32_t num_heads,
                                    std::uint32_t head_dim,
                                    std::uint32_t n_splits) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t d = threadIdx.x;
  if (h >= num_heads || d >= head_dim) {
    return;
  }
  const float* p =
      partial + static_cast<std::size_t>(h) * n_splits * (head_dim + 2U);
  float gmax = sinks != nullptr ? sinks[h] : -INFINITY;
  for (std::uint32_t sp = 0; sp < n_splits; ++sp) {
    const float m = p[static_cast<std::size_t>(sp) * (head_dim + 2U)];
    if (m > gmax) {
      gmax = m;
    }
  }
  float num = 0.0F;
  float den = sinks != nullptr ? expf(sinks[h] - gmax) : 0.0F;
  for (std::uint32_t sp = 0; sp < n_splits; ++sp) {
    const std::size_t base = static_cast<std::size_t>(sp) * (head_dim + 2U);
    const float w = gmax == -INFINITY ? 0.0F : expf(p[base] - gmax);
    num += w * p[base + 2U + d];
    den += w * p[base + 1U];
  }
  out[static_cast<std::size_t>(h) * head_dim + d] =
      den == 0.0F ? 0.0F : num / den;
}

}  // namespace

FlintLaunchStatus LaunchFlintDecodeAttention(
    const float* q, const float* k_cache, const float* v_cache, float* out,
    std::uint32_t layer, const std::uint32_t* d_pos, std::uint32_t window,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, std::uint32_t n_splits,
    float* partial, const float* sinks, hipStream_t stream) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      out == nullptr || d_pos == nullptr || partial == nullptr) {
    return Bad("flint decode attention refused: null pointer");
  }
  if (head_dim != kFixedHeadDim) {
    return Bad("flint decode attention refused: head_dim");
  }
  if (num_kv_heads == 0 || num_heads == 0 ||
      (num_heads % num_kv_heads) != 0) {
    return Bad("flint decode attention refused: heads");
  }
  const std::uint32_t group = num_heads / num_kv_heads;
  if (group > kMaxGroup) {
    return Bad("flint decode attention refused: group");
  }
  if (n_splits == 0 || n_splits > kMaxSplits) {
    return Bad("flint decode attention refused: n_splits");
  }
  if (max_context == 0) {
    return Bad("flint decode attention refused: max_ctx");
  }
  const dim3 grid(num_kv_heads, n_splits);
  FlintSplitKernelA<<<grid, kSplitThreads, 0, stream>>>(
      q, k_cache, v_cache, partial, layer, d_pos, window, max_context,
      num_heads, num_kv_heads, head_dim, n_splits);
  const FlintLaunchStatus split = CheckLaunch("flint split launch failed");
  if (!split.ok) {
    return split;
  }
  FlintCombineKernelB<<<num_heads, kCombineBlock, 0, stream>>>(
      partial, out, sinks, num_heads, head_dim, n_splits);
  return CheckLaunch("flint combine launch failed");
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
