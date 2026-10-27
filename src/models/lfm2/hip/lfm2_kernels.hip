// LFM2 short-conv (decode/prefill) + MoE-router GPU kernels. fp32 throughout.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/lfm2/hip/lfm2_kernels.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace superroc::hip {
namespace {

constexpr std::uint32_t kBlock = 256U;
constexpr double kMoeWeightFloor = 6.103515625e-5;

Lfm2LaunchStatus Ok() noexcept { return {true, nullptr}; }

Lfm2LaunchStatus Bad(const char* error) noexcept { return {false, error}; }

// Audit F04: report an async launch failure under a static string.
Lfm2LaunchStatus CheckLaunch(const char* what) noexcept {
  if (hipGetLastError() != hipSuccess) {
    return Bad(what);
  }
  return Ok();
}

Lfm2LaunchStatus CheckShortConv(const float* bcx, const float* conv_w,
                                const float* state, const float* y,
                                std::uint32_t hidden,
                                std::uint32_t l_cache) noexcept {
  if (bcx == nullptr || conv_w == nullptr || state == nullptr || y == nullptr) {
    return Bad("lfm2 shortconv refused: null pointer");
  }
  if (hidden == 0 || hidden % 32U != 0) {
    return Bad("lfm2 shortconv refused: hidden");
  }
  if (l_cache < 2U || l_cache > 8U) {
    return Bad("lfm2 shortconv refused: l_cache");
  }
  return Ok();
}

Lfm2LaunchStatus CheckRoute(const float* x, const float* gate_inp,
                            const float* bias, const float* logits,
                            const int* ids, const float* weights,
                            std::uint32_t k, std::uint32_t n_expert,
                            std::uint32_t n_used) noexcept {
  if (x == nullptr || gate_inp == nullptr || bias == nullptr ||
      logits == nullptr || ids == nullptr || weights == nullptr) {
    return Bad("lfm2 moe route refused: null pointer");
  }
  if (k == 0 || k > 8192U) {
    return Bad("lfm2 moe route refused: k");
  }
  if (n_expert == 0 || n_expert > 256U) {
    return Bad("lfm2 moe route refused: n_expert");
  }
  if (n_used == 0 || n_used > 8U || n_used > n_expert) {
    return Bad("lfm2 moe route refused: n_used");
  }
  return Ok();
}

Lfm2LaunchStatus CheckTokens(std::uint32_t n_tokens) noexcept {
  if (n_tokens == 0 || n_tokens > 32768U) {
    return Bad("lfm2 refused: n_tokens");
  }
  return Ok();
}

__global__ void Lfm2ShortConvDecodeKernel(const float* __restrict__ bcx,
                                          const float* __restrict__ conv_w,
                                          float* __restrict__ state,
                                          float* __restrict__ y,
                                          std::uint32_t hidden,
                                          std::uint32_t l_cache) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= hidden) {
    return;
  }
  const float b = bcx[c];
  const float cc = bcx[static_cast<std::size_t>(hidden) + c];
  const float xx = bcx[static_cast<std::size_t>(2U) * hidden + c];
  const float bx = b * xx;
  const float* w =
      conv_w + static_cast<std::size_t>(c) * l_cache;
  float acc = w[l_cache - 1U] * bx;
  for (std::uint32_t j = 0; j + 1U < l_cache; ++j) {
    acc += w[j] * state[static_cast<std::size_t>(j) * hidden + c];
  }
  y[c] = cc * acc;
  for (std::uint32_t j = 0; j + 2U < l_cache; ++j) {
    state[static_cast<std::size_t>(j) * hidden + c] =
        state[static_cast<std::size_t>(j + 1U) * hidden + c];
  }
  state[static_cast<std::size_t>(l_cache - 2U) * hidden + c] = bx;
}

// LPF: parallel short-conv prefill. Thread (t, c) forms y[t][c] exactly as the
// former one-thread-per-channel loop did: acc = w[l-1] * bx_t, then
// acc += w[j] * v_j for j = 0..l-2, where v_j is bx of token t-(l-1)+j, or
// the incoming state row t+j when that token is before the chunk. The state
// rows are rewritten afterwards by Lfm2ShortConvStateKernel (same stream).
__global__ void Lfm2ShortConvPrefillKernel(const float* __restrict__ bcx,
                                          const float* __restrict__ conv_w,
                                          const float* __restrict__ state,
                                          float* __restrict__ y,
                                          std::uint32_t hidden,
                                          std::uint32_t l_cache,
                                          std::uint32_t n_tokens) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t t = blockIdx.y;
  if (c >= hidden || t >= n_tokens) {
    return;
  }
  const std::uint32_t nm1 = l_cache - 1U;
  const float* w = conv_w + static_cast<std::size_t>(c) * l_cache;
  const float* brow = bcx + static_cast<std::size_t>(t) * 3U * hidden;
  const float bx = brow[c] * brow[static_cast<std::size_t>(2U) * hidden + c];
  float acc = w[nm1] * bx;
  for (std::uint32_t j = 0; j < nm1; ++j) {
    const std::int64_t src = static_cast<std::int64_t>(t) -
                             static_cast<std::int64_t>(nm1) +
                             static_cast<std::int64_t>(j);
    float v = 0.0F;
    if (src >= 0) {
      const float* srow = bcx + static_cast<std::size_t>(src) * 3U * hidden;
      v = srow[c] * srow[static_cast<std::size_t>(2U) * hidden + c];
    } else {
      v = state[static_cast<std::size_t>(t + j) * hidden + c];
    }
    acc += w[j] * v;
  }
  y[static_cast<std::size_t>(t) * hidden + c] =
      brow[static_cast<std::size_t>(hidden) + c] * acc;
}

// LPF: after the chunk, state row j holds bx of token n-(l-1)+j, or the old
// state row n+j when the chunk is shorter than the state.
__global__ void Lfm2ShortConvStateKernel(const float* __restrict__ bcx,
                                        float* __restrict__ state,
                                        std::uint32_t hidden,
                                        std::uint32_t l_cache,
                                        std::uint32_t n_tokens) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= hidden) {
    return;
  }
  const std::uint32_t nm1 = l_cache - 1U;
  float next[7];
  for (std::uint32_t j = 0; j < nm1; ++j) {
    const std::int64_t src = static_cast<std::int64_t>(n_tokens) -
                             static_cast<std::int64_t>(nm1) +
                             static_cast<std::int64_t>(j);
    if (src >= 0) {
      const float* srow = bcx + static_cast<std::size_t>(src) * 3U * hidden;
      next[j] = srow[c] * srow[static_cast<std::size_t>(2U) * hidden + c];
    } else {
      next[j] = state[static_cast<std::size_t>(n_tokens + j) * hidden + c];
    }
  }
  for (std::uint32_t j = 0; j < nm1; ++j) {
    state[static_cast<std::size_t>(j) * hidden + c] = next[j];
  }
}

// One block per (expert, token): block (e, t) dots gate row e with x row t.
// Thread i sums columns i, i+256, ...; then a fixed halving tree reduction.
__global__ void Lfm2MoeDotKernel(const float* __restrict__ x,
                                 const float* __restrict__ gate_inp,
                                 std::uint32_t k, std::uint32_t n_expert,
                                 std::uint32_t n_tokens,
                                 float* __restrict__ logits) {
  const std::uint32_t e = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  if (e >= n_expert || t >= n_tokens) {
    return;
  }
  const float* xr = x + static_cast<std::size_t>(t) * k;
  const float* gr = gate_inp + static_cast<std::size_t>(e) * k;
  float partial = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
    partial += xr[i] * gr[i];
  }
  __shared__ float red[256];
  red[threadIdx.x] = partial;
  __syncthreads();
  for (std::uint32_t s = blockDim.x / 2U; s > 0U; s >>= 1U) {
    if (threadIdx.x < s) {
      red[threadIdx.x] += red[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    logits[static_cast<std::size_t>(t) * n_expert + e] = red[0];
  }
}

// LPF: router logits for a batch, one block per token computing every
// expert. Each logit is formed exactly as Lfm2MoeDotKernel forms it (thread i
// sums columns i, i+256, ... in order, then the same halving tree), so batch
// rows stay bit-identical to the single-token launcher; the token's row is
// read once into shared memory and the experts go in passes of kDotPass.
constexpr std::uint32_t kDotPass = 8U;
__global__ void Lfm2MoeDotRowsKernel(const float* __restrict__ x,
                                     const float* __restrict__ gate_inp,
                                     std::uint32_t k, std::uint32_t n_expert,
                                     std::uint32_t n_tokens,
                                     float* __restrict__ logits) {
  const std::uint32_t t = blockIdx.x;
  if (t >= n_tokens) {
    return;
  }
  extern __shared__ float xs[];
  __shared__ float red[kDotPass][kBlock];
  const std::uint32_t tid = threadIdx.x;
  const float* xr = x + static_cast<std::size_t>(t) * k;
  for (std::uint32_t i = tid; i < k; i += blockDim.x) {
    xs[i] = xr[i];
  }
  __syncthreads();
  for (std::uint32_t e0 = 0; e0 < n_expert; e0 += kDotPass) {
    for (std::uint32_t q = 0; q < kDotPass; ++q) {
      const std::uint32_t e = e0 + q;
      float partial = 0.0F;
      if (e < n_expert) {
        const float* gr = gate_inp + static_cast<std::size_t>(e) * k;
        for (std::uint32_t i = tid; i < k; i += blockDim.x) {
          partial += xs[i] * gr[i];
        }
      }
      red[q][tid] = partial;
    }
    __syncthreads();
    for (std::uint32_t st = blockDim.x / 2U; st > 0U; st >>= 1U) {
      if (tid < st) {
        for (std::uint32_t q = 0; q < kDotPass; ++q) {
          red[q][tid] += red[q][tid + st];
        }
      }
      __syncthreads();
    }
    if (tid < kDotPass && e0 + tid < n_expert) {
      logits[static_cast<std::size_t>(t) * n_expert + e0 + tid] = red[tid][0];
    }
    __syncthreads();
  }
}

// One block per token (blockDim = kBlock, a power of two >= n_expert): thread
// e < n_expert forms p/sel. Then n_used rounds of a block argmax over the
// experts not yet taken: larger sel wins, equal sel -> lower id, which is the
// order the former serial scan used (strict >, ascending). NaN sel never
// wins. Thread 0 normalizes in pick order. (Super ROC: the serial scan of
// one thread over a 256-entry local array cost 21.5 us per layer.)
__global__ void Lfm2MoePickKernel(const float* __restrict__ logits,
                                  const float* __restrict__ bias,
                                  std::uint32_t n_expert,
                                  std::uint32_t n_tokens, std::uint32_t n_used,
                                  int* __restrict__ ids,
                                  float* __restrict__ weights) {
  const std::uint32_t t = blockIdx.x;
  if (t >= n_tokens) {
    return;
  }
  const std::uint32_t e = threadIdx.x;
  __shared__ float p_sh[kBlock];
  __shared__ float red_v[kBlock];
  __shared__ int red_i[kBlock];
  __shared__ int esel[8];
  constexpr int kNoExpert = std::numeric_limits<int>::max();
  float sel = -INFINITY;
  if (e < n_expert) {
    const float logit =
        logits[static_cast<std::size_t>(t) * n_expert + e];
    const float p = 1.0F / (1.0F + expf(-logit));
    p_sh[e] = p;
    sel = p + bias[e];
    if (!(sel == sel)) {
      sel = -INFINITY;
    }
  }
  bool taken = e >= n_expert;
  for (std::uint32_t i = 0; i < n_used; ++i) {
    red_v[e] = taken ? -INFINITY : sel;
    red_i[e] = taken ? kNoExpert : static_cast<int>(e);
    __syncthreads();
    for (std::uint32_t s = blockDim.x / 2U; s > 0U; s >>= 1U) {
      if (e < s) {
        const float ov = red_v[e + s];
        const int oi = red_i[e + s];
        if (ov > red_v[e] || (ov == red_v[e] && oi < red_i[e])) {
          red_v[e] = ov;
          red_i[e] = oi;
        }
      }
      __syncthreads();
    }
    const int best = red_i[0];
    __syncthreads();
    if (static_cast<int>(e) == best) {
      taken = true;
    }
    if (e == 0) {
      esel[i] = best;
    }
  }
  if (e != 0) {
    return;
  }
  double sum = 0.0;
  for (std::uint32_t i = 0; i < n_used; ++i) {
    sum += static_cast<double>(p_sh[esel[i]]);
  }
  const float denom =
      static_cast<float>(sum > kMoeWeightFloor ? sum : kMoeWeightFloor);
  int* irow = ids + static_cast<std::size_t>(t) * n_used;
  float* wrow = weights + static_cast<std::size_t>(t) * n_used;
  for (std::uint32_t i = 0; i < n_used; ++i) {
    irow[i] = esel[i];
    wrow[i] = p_sh[esel[i]] / denom;
  }
}

}  // namespace

Lfm2LaunchStatus LaunchLfm2ShortConvDecode(const float* bcx, const float* conv_w,
                                           float* state, float* y,
                                           std::uint32_t hidden,
                                           std::uint32_t l_cache,
                                           hipStream_t stream) {
  const Lfm2LaunchStatus checked =
      CheckShortConv(bcx, conv_w, state, y, hidden, l_cache);
  if (!checked.ok) {
    return checked;
  }
  const std::uint32_t grid = (hidden + kBlock - 1U) / kBlock;
  Lfm2ShortConvDecodeKernel<<<grid, kBlock, 0, stream>>>(bcx, conv_w, state, y,
                                                         hidden, l_cache);
  return CheckLaunch("lfm2 shortconv decode launch failed");
}

Lfm2LaunchStatus LaunchLfm2ShortConvPrefill(const float* bcx, const float* conv_w,
                                           float* state, float* y,
                                           std::uint32_t hidden,
                                           std::uint32_t l_cache,
                                           std::uint32_t n_tokens,
                                           hipStream_t stream) {
  const Lfm2LaunchStatus checked =
      CheckShortConv(bcx, conv_w, state, y, hidden, l_cache);
  if (!checked.ok) {
    return checked;
  }
  const Lfm2LaunchStatus tok = CheckTokens(n_tokens);
  if (!tok.ok) {
    return tok;
  }
  const std::uint32_t grid = (hidden + kBlock - 1U) / kBlock;
  Lfm2ShortConvPrefillKernel<<<dim3(grid, n_tokens), kBlock, 0, stream>>>(
      bcx, conv_w, state, y, hidden, l_cache, n_tokens);
  const Lfm2LaunchStatus conv = CheckLaunch("lfm2 shortconv prefill launch failed");
  if (!conv.ok) {
    return conv;
  }
  Lfm2ShortConvStateKernel<<<grid, kBlock, 0, stream>>>(bcx, state, hidden,
                                                        l_cache, n_tokens);
  return CheckLaunch("lfm2 shortconv state launch failed");
}

Lfm2LaunchStatus LaunchLfm2MoeRoute(const float* x, const float* gate_inp,
                                    const float* bias, std::uint32_t k,
                                    std::uint32_t n_expert, std::uint32_t n_used,
                                    float* logits, int* ids, float* weights,
                                    hipStream_t stream) {
  const Lfm2LaunchStatus checked =
      CheckRoute(x, gate_inp, bias, logits, ids, weights, k, n_expert, n_used);
  if (!checked.ok) {
    return checked;
  }
  const dim3 dot_grid(n_expert, 1U);
  Lfm2MoeDotKernel<<<dot_grid, kBlock, 0, stream>>>(x, gate_inp, k, n_expert, 1U,
                                                   logits);
  const Lfm2LaunchStatus dot = CheckLaunch("lfm2 moe route logits launch failed");
  if (!dot.ok) {
    return dot;
  }
  Lfm2MoePickKernel<<<1, kBlock, 0, stream>>>(logits, bias, n_expert, 1U, n_used,
                                             ids, weights);
  return CheckLaunch("lfm2 moe route pick launch failed");
}

Lfm2LaunchStatus LaunchLfm2MoeRouteBatch(
    const float* x, const float* gate_inp, const float* bias, std::uint32_t k,
    std::uint32_t n_expert, std::uint32_t n_tokens, std::uint32_t n_used,
    float* logits, int* ids, float* weights, hipStream_t stream) {
  const Lfm2LaunchStatus checked =
      CheckRoute(x, gate_inp, bias, logits, ids, weights, k, n_expert, n_used);
  if (!checked.ok) {
    return checked;
  }
  const Lfm2LaunchStatus tok = CheckTokens(n_tokens);
  if (!tok.ok) {
    return tok;
  }
  Lfm2MoeDotRowsKernel<<<n_tokens, kBlock, k * sizeof(float), stream>>>(
      x, gate_inp, k, n_expert, n_tokens, logits);  // LPF
  const Lfm2LaunchStatus dot = CheckLaunch("lfm2 moe route logits launch failed");
  if (!dot.ok) {
    return dot;
  }
  Lfm2MoePickKernel<<<n_tokens, kBlock, 0, stream>>>(
      logits, bias, n_expert, n_tokens, n_used, ids, weights);
  return CheckLaunch("lfm2 moe route pick launch failed");
}

namespace {

// LKF: ResidualAddKernel (out = a + b) then RMSNormKernel<> (one 256-thread
// block, strided sum of squares, tree reduction, (x * inv_rms) * w), fused.
// Every value is formed by the same expression in the same order.
__global__ void Lfm2AddRMSNormKernel(float* __restrict__ hidden,
                                     const float* __restrict__ add,
                                     const float* __restrict__ weight,
                                     float* __restrict__ out,
                                     std::uint32_t dim, float eps) {
  __shared__ float s_sum[kBlock];
  const std::uint32_t tid = threadIdx.x;
  float sum_sq = 0.0F;
  for (std::uint32_t i = tid; i < dim; i += blockDim.x) {
    const float val = hidden[i] + add[i];
    hidden[i] = val;
    sum_sq += val * val;
  }
  s_sum[tid] = sum_sq;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }
  const float mean_sq = s_sum[0] / static_cast<float>(dim);
  const float inv_rms = rsqrtf(mean_sq + eps);
  for (std::uint32_t i = tid; i < dim; i += blockDim.x) {
    const float w = weight != nullptr ? weight[i] : 1.0F;
    out[i] = (hidden[i] * inv_rms) * w;
  }
}

}  // namespace

Lfm2LaunchStatus LaunchLfm2AddRMSNorm(float* hidden, const float* add,
                                      const float* weight, float* out,
                                      std::uint32_t dim, float eps,
                                      hipStream_t stream) {
  if (hidden == nullptr || add == nullptr || out == nullptr) {
    return Bad("lfm2 add rmsnorm refused: null pointer");
  }
  if (dim == 0) {
    return Bad("lfm2 add rmsnorm refused: dim");
  }
  Lfm2AddRMSNormKernel<<<1, kBlock, 0, stream>>>(hidden, add, weight, out, dim,
                                                 eps);
  return CheckLaunch("lfm2 add rmsnorm launch failed");
}

}  // namespace superroc::hip
#endif  // defined(ENGINE_ENABLE_HIP)
