// LFM2 split-K "flash-decoding" attention (decode) + device-side KV write.
// fp32 math; KV either fp32 head-major or fp16 token-major (SPEC3).
// head_dim 64, group (heads/kv_heads) <= 8, n_splits <= 64.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/lfm2/hip/lfm2_attention.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace gufo::hip {
namespace {

constexpr std::uint32_t kSplitBlock = 256U;    // kernel A width
constexpr std::uint32_t kCombineBlock = 64U;   // kernel B width
constexpr std::uint32_t kTeam = 32U;           // lanes per team (shuffle width)
constexpr std::uint32_t kTeams = 4U;           // teams per kernel A block
constexpr std::uint32_t kSplitThreads = kTeam * kTeams;  // kernel A width
constexpr std::uint32_t kMaxGroup = 8U;        // max q heads per kv head
constexpr std::uint32_t kFixedHeadDim = 64U;   // only supported head_dim
constexpr std::uint32_t kMaxSplits = 64U;      // max n_splits

Lfm2LaunchStatus Ok() noexcept { return {true, nullptr}; }

Lfm2LaunchStatus Bad(const char* error) noexcept { return {false, error}; }

// Audit F04: report an async launch failure under a static string.
Lfm2LaunchStatus CheckLaunch(const char* what) noexcept {
  if (hipGetLastError() != hipSuccess) {
    return Bad(what);
  }
  return Ok();
}

// SPEC3: one KV element as fp32 (fp32 cache or fp16 bits).
__device__ __forceinline__ float Lfm2KvLoad(const float* p, std::size_t i) {
  return p[i];
}
__device__ __forceinline__ float Lfm2KvLoad(const std::uint16_t* p,
                                            std::size_t i) {
  return __half2float(__ushort_as_half(p[i]));
}

// Same math as GlimmerWriteKvKernel, but pos is read on the device.
__global__ void Lfm2WriteKvDevKernel(const float* __restrict__ k,
                                    const float* __restrict__ v,
                                    float* __restrict__ k_cache,
                                    float* __restrict__ v_cache,
                                    std::uint32_t layer,
                                    const std::uint32_t* __restrict__ d_pos,
                                    std::uint32_t max_context,
                                    std::uint32_t num_kv_heads,
                                    std::uint32_t head_dim) {
  const std::uint32_t pos = *d_pos;
  const std::uint32_t kv_h = blockIdx.x;
  if (kv_h >= num_kv_heads || threadIdx.x >= head_dim) {
    return;
  }
  const std::size_t dst =
      (((static_cast<std::size_t>(layer) * num_kv_heads + kv_h) * max_context +
        pos) *
       head_dim) +
      threadIdx.x;
  const std::size_t src =
      static_cast<std::size_t>(kv_h) * head_dim + threadIdx.x;
  k_cache[dst] = k[src];
  v_cache[dst] = v[src];
}

// HKV (RVW2 V1): the fp16 store saturates at +-65504 instead of overflowing
// to +-inf (an inf key turns every later score into inf/NaN); NaN stays NaN.
// In-range values convert exactly as before (round to nearest even).
__device__ __forceinline__ std::uint16_t Lfm2F16Sat(float x) {
  const float c = x != x ? x : fminf(fmaxf(x, -65504.0F), 65504.0F);
  return __half_as_ushort(__float2half(c));
}

// SPEC3: fp16 token-major write, element ((layer*max_ctx+pos)*kv_heads+kvh)*hd+d
// (the production plane layout), round to nearest even.
__global__ void Lfm2WriteKvDevF16Kernel(const float* __restrict__ k,
                                       const float* __restrict__ v,
                                       std::uint16_t* __restrict__ k_cache,
                                       std::uint16_t* __restrict__ v_cache,
                                       std::uint32_t layer,
                                       const std::uint32_t* __restrict__ d_pos,
                                       std::uint32_t max_context,
                                       std::uint32_t num_kv_heads,
                                       std::uint32_t head_dim) {
  const std::uint32_t pos = *d_pos;
  const std::uint32_t kv_h = blockIdx.x;
  if (kv_h >= num_kv_heads || threadIdx.x >= head_dim) {
    return;
  }
  const std::size_t src =
      static_cast<std::size_t>(kv_h) * head_dim + threadIdx.x;
  const std::size_t dst =
      (static_cast<std::size_t>(layer) * max_context + pos) * num_kv_heads *
          head_dim +
      src;
  k_cache[dst] = Lfm2F16Sat(k[src]);
  v_cache[dst] = Lfm2F16Sat(v[src]);
}

// SPEC3/LDW: eight consecutive KV elements as fp32 (vector loads; the row
// start is 64-element aligned and c is a multiple of 8).
__device__ __forceinline__ void Lfm2KvLoad8(const float* p, float* out) {
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
__device__ __forceinline__ float Lfm2HalfBits(unsigned int bits) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}
__device__ __forceinline__ void Lfm2KvLoad8(const std::uint16_t* p,
                                            float* out) {
  const uint4 w = *reinterpret_cast<const uint4*>(p);
  out[0] = Lfm2HalfBits(w.x & 0xFFFFU);
  out[1] = Lfm2HalfBits(w.x >> 16U);
  out[2] = Lfm2HalfBits(w.y & 0xFFFFU);
  out[3] = Lfm2HalfBits(w.y >> 16U);
  out[4] = Lfm2HalfBits(w.z & 0xFFFFU);
  out[5] = Lfm2HalfBits(w.z >> 16U);
  out[6] = Lfm2HalfBits(w.w & 0xFFFFU);
  out[7] = Lfm2HalfBits(w.w >> 16U);
}

// LDW: element row (64 values) of position pos for kv head kvh.
template <bool kTokenMajor>
__device__ __forceinline__ std::size_t Lfm2KvRow(std::uint32_t layer,
                                                 std::uint32_t kvh,
                                                 std::uint32_t pos,
                                                 std::uint32_t max_context,
                                                 std::uint32_t num_kv_heads) {
  return kTokenMajor
             ? (static_cast<std::size_t>(layer) * max_context + pos) *
                       num_kv_heads +
                   kvh
             : (static_cast<std::size_t>(layer) * num_kv_heads + kvh) *
                       max_context +
                   pos;
}

// Flash-decoding split pass (LDW). Block (kvh, s) owns kv head kvh and the
// positions [s*chunk, min(n, (s+1)*chunk)), chunk = ceil(n / n_splits). Four
// 32-lane teams take 32-position tiles of that range in turn. In a tile, lane
// j scores position j against the group's q heads (q pre-scaled in shared
// memory, K row read with vector loads); the team forms the online-softmax
// update with xor-shuffles; then every lane accumulates its two dims (lane,
// lane + 32) over the tile's V rows (coalesced) with the weights broadcast by
// shuffle. The teams merge in shared memory and the block writes m, l and
// acc[64] per q head of the group into `partial` (an empty range writes
// m = -inf, l = 0, acc = 0). KvT float = fp32 head-major cache; KvT uint16
// with kTokenMajor = fp16 token-major cache.
template <typename KvT, bool kTokenMajor>
__global__ void __launch_bounds__(kSplitThreads)
    Lfm2SplitKernelA(const float* __restrict__ q,
                     const KvT* __restrict__ k_cache,
                     const KvT* __restrict__ v_cache,
                     float* __restrict__ partial, std::uint32_t layer,
                     const std::uint32_t* __restrict__ d_pos,
                     std::uint32_t max_context, std::uint32_t num_heads,
                     std::uint32_t num_kv_heads, std::uint32_t head_dim,
                     std::uint32_t n_splits) {
  const std::uint32_t kvh = blockIdx.x;
  const std::uint32_t s = blockIdx.y;
  if (kvh >= num_kv_heads || s >= n_splits) {
    return;
  }
  const std::uint32_t group = num_heads / num_kv_heads;
  const std::uint32_t n = *d_pos + 1U;
  const std::uint32_t chunk = (n + n_splits - 1U) / n_splits;
  const std::uint32_t start = s * chunk;
  const std::uint32_t end = start + chunk < n ? start + chunk : n;

  __shared__ float qsm[kMaxGroup][kFixedHeadDim];
  __shared__ float team_m[kTeams][kMaxGroup];
  __shared__ float team_l[kTeams][kMaxGroup];
  __shared__ float team_acc[kTeams][kMaxGroup][kFixedHeadDim];

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t team = tid / kTeam;
  const std::uint32_t lane = tid % kTeam;
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  const float* qh = q + static_cast<std::size_t>(kvh) * group * head_dim;
  for (std::uint32_t i = tid; i < group * kFixedHeadDim; i += blockDim.x) {
    qsm[i / kFixedHeadDim][i % kFixedHeadDim] = qh[i] * scale;
  }
  __syncthreads();

  float m[kMaxGroup];
  float l[kMaxGroup];
  float a0[kMaxGroup];
  float a1[kMaxGroup];
#pragma unroll
  for (std::uint32_t g = 0; g < kMaxGroup; ++g) {
    m[g] = -INFINITY;
    l[g] = 0.0F;
    a0[g] = 0.0F;
    a1[g] = 0.0F;
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
      const KvT* krow =
          k_cache + Lfm2KvRow<kTokenMajor>(layer, kvh, tb + lane, max_context,
                                           num_kv_heads) *
                        kFixedHeadDim;
      for (std::uint32_t c = 0; c < kFixedHeadDim; c += 8U) {
        float k8[8];
        Lfm2KvLoad8(krow + c, k8);
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
        // HKV (RVW2 D3): with every score so far -inf, exp(-inf - -inf) is
        // NaN; shift by 0 instead so the tile adds nothing. Finite m_new: as
        // before.
        const float m_ref = m_new == -INFINITY ? 0.0F : m_new;
        const float resc = expf(m[g] - m_ref);
        const float p = live ? expf(score - m_ref) : 0.0F;
        float tsum = p;
        for (std::uint32_t off = kTeam / 2U; off > 0U; off >>= 1U) {
          tsum += __shfl_xor(tsum, static_cast<int>(off),
                             static_cast<int>(kTeam));
        }
        l[g] = l[g] * resc + tsum;
        a0[g] *= resc;
        a1[g] *= resc;
        m[g] = m_new;
        w[g] = p;
      }
    }
    for (std::uint32_t j = 0; j < tn; ++j) {
      const std::size_t vbase =
          Lfm2KvRow<kTokenMajor>(layer, kvh, tb + j, max_context,
                                 num_kv_heads) *
          kFixedHeadDim;
      const float v0 = Lfm2KvLoad(v_cache, vbase + lane);
      const float v1 = Lfm2KvLoad(v_cache, vbase + lane + kTeam);
#pragma unroll
      for (std::uint32_t g = 0; g < kMaxGroup; ++g) {
        if (g < group) {
          const float wj = __shfl(w[g], static_cast<int>(j),
                                  static_cast<int>(kTeam));
          a0[g] += wj * v0;
          a1[g] += wj * v1;
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
      team_acc[team][g][lane] = a0[g];
      team_acc[team][g][lane + kTeam] = a1[g];
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

// Flash-decoding combine pass. Block h combines the n_splits partials of q
// head h (global max, rescaled sums) into out[h*head_dim + d].
__global__ void Lfm2CombineKernelB(const float* __restrict__ partial,
                                   float* __restrict__ out,
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
  float gmax = -INFINITY;
  for (std::uint32_t sp = 0; sp < n_splits; ++sp) {
    const float m = p[static_cast<std::size_t>(sp) * (head_dim + 2U)];
    if (m > gmax) {
      gmax = m;
    }
  }
  float num = 0.0F;
  float den = 0.0F;
  for (std::uint32_t sp = 0; sp < n_splits; ++sp) {
    const std::size_t base = static_cast<std::size_t>(sp) * (head_dim + 2U);
    // HKV (RVW2 D3): all splits -inf -> no weight (output 0), not NaN.
    const float w = gmax == -INFINITY ? 0.0F : expf(p[base] - gmax);
    num += w * p[base + 2U + d];
    den += w * p[base + 1U];
  }
  out[static_cast<std::size_t>(h) * head_dim + d] =
      den == 0.0F ? 0.0F : num / den;
}

}  // namespace

Lfm2LaunchStatus LaunchLfm2WriteKvDev(
    const float* k, const float* v, float* k_cache, float* v_cache,
    std::uint32_t layer, const std::uint32_t* d_pos,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, hipStream_t stream) {
  if (k == nullptr || v == nullptr || k_cache == nullptr ||
      v_cache == nullptr || d_pos == nullptr) {
    return Bad("lfm2 writekv dev refused: null pointer");
  }
  if (num_kv_heads == 0 || head_dim == 0 || head_dim > kSplitBlock) {
    return Bad("lfm2 writekv dev refused: dims");
  }
  if (max_context == 0) {
    return Bad("lfm2 writekv dev refused: max_ctx");
  }
  Lfm2WriteKvDevKernel<<<num_kv_heads, kSplitBlock, 0, stream>>>(
      k, v, k_cache, v_cache, layer, d_pos, max_context, num_kv_heads,
      head_dim);
  return CheckLaunch("lfm2 writekv dev launch failed");
}

Lfm2LaunchStatus LaunchLfm2WriteKvDevF16(
    const float* k, const float* v, std::uint16_t* k_cache,
    std::uint16_t* v_cache, std::uint32_t layer, const std::uint32_t* d_pos,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, hipStream_t stream) {
  if (k == nullptr || v == nullptr || k_cache == nullptr ||
      v_cache == nullptr || d_pos == nullptr) {
    return Bad("lfm2 writekv f16 refused: null pointer");
  }
  if (num_kv_heads == 0 || head_dim == 0 || head_dim > kSplitBlock) {
    return Bad("lfm2 writekv f16 refused: dims");
  }
  if (max_context == 0) {
    return Bad("lfm2 writekv f16 refused: max_ctx");
  }
  Lfm2WriteKvDevF16Kernel<<<num_kv_heads, kSplitBlock, 0, stream>>>(
      k, v, k_cache, v_cache, layer, d_pos, max_context, num_kv_heads,
      head_dim);
  return CheckLaunch("lfm2 writekv f16 launch failed");
}

namespace {

template <typename KvT, bool kTokenMajor>
Lfm2LaunchStatus LaunchLfm2DecodeAttentionT(
    const float* q, const KvT* k_cache, const KvT* v_cache, float* out,
    std::uint32_t layer, const std::uint32_t* d_pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t n_splits, float* partial, hipStream_t stream) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      out == nullptr || d_pos == nullptr || partial == nullptr) {
    return Bad("lfm2 decode attention refused: null pointer");
  }
  if (head_dim != kFixedHeadDim) {
    return Bad("lfm2 decode attention refused: head_dim");
  }
  if (num_kv_heads == 0 || num_heads == 0 ||
      (num_heads % num_kv_heads) != 0) {
    return Bad("lfm2 decode attention refused: heads");
  }
  const std::uint32_t group = num_heads / num_kv_heads;
  if (group > kMaxGroup) {
    return Bad("lfm2 decode attention refused: group");
  }
  if (n_splits == 0 || n_splits > kMaxSplits) {
    return Bad("lfm2 decode attention refused: n_splits");
  }
  if (max_context == 0) {
    return Bad("lfm2 decode attention refused: max_ctx");
  }
  const dim3 grid(num_kv_heads, n_splits);
  Lfm2SplitKernelA<KvT, kTokenMajor><<<grid, kSplitThreads, 0, stream>>>(
      q, k_cache, v_cache, partial, layer, d_pos, max_context, num_heads,
      num_kv_heads, head_dim, n_splits);
  const Lfm2LaunchStatus split = CheckLaunch("lfm2 split launch failed");
  if (!split.ok) {
    return split;
  }
  Lfm2CombineKernelB<<<num_heads, kCombineBlock, 0, stream>>>(
      partial, out, num_heads, head_dim, n_splits);
  return CheckLaunch("lfm2 combine launch failed");
}

}  // namespace

Lfm2LaunchStatus LaunchLfm2DecodeAttention(
    const float* q, const float* k_cache, const float* v_cache, float* out,
    std::uint32_t layer, const std::uint32_t* d_pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t n_splits, float* partial, hipStream_t stream) {
  return LaunchLfm2DecodeAttentionT<float, false>(
      q, k_cache, v_cache, out, layer, d_pos, max_context, num_heads,
      num_kv_heads, head_dim, n_splits, partial, stream);
}

Lfm2LaunchStatus LaunchLfm2DecodeAttentionF16(
    const float* q, const std::uint16_t* k_cache,
    const std::uint16_t* v_cache, float* out, std::uint32_t layer,
    const std::uint32_t* d_pos, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, std::uint32_t n_splits, float* partial,
    hipStream_t stream) {
  return LaunchLfm2DecodeAttentionT<std::uint16_t, true>(
      q, k_cache, v_cache, out, layer, d_pos, max_context, num_heads,
      num_kv_heads, head_dim, n_splits, partial, stream);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
