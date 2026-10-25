#include "src/models/qwen/moe/qwen_moe_ffn.hpp"

#include <cstddef>
#include <cstdio>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxK = 8192;
constexpr int kMaxTokens = 16384;

// Index proof: block t < n_tokens reads x[(size_t)t * k + i] and gate_vec[i]
// for i in [0, k) only, writes sig_out[t] once (thread 0). All threads reach
// every barrier; the blockIdx guard is block-uniform.
__global__ void QwenMoeShexpSigmoidKernel(const float* __restrict__ x,
                                          const float* __restrict__ gate_vec,
                                          int k, int n_tokens,
                                          float* __restrict__ sig_out) {
  __shared__ float red[kThreads];
  const int tok = static_cast<int>(blockIdx.x);
  if (tok >= n_tokens) {
    return;  // block-uniform
  }
  const float* xt = x + (static_cast<size_t>(tok) * static_cast<size_t>(k));
  const int t = static_cast<int>(threadIdx.x);
  float acc = 0.0F;
  for (int i = t; i < k; i += kThreads) {
    acc += gate_vec[i] * xt[i];
  }
  red[t] = acc;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (t < s) {
      red[t] += red[t + s];
    }
    __syncthreads();
  }
  if (t == 0) {
    sig_out[tok] = 1.0F / (1.0F + expf(-red[0]));
  }
}

__global__ void QwenMoeCombineKernel(const float* __restrict__ moe_y,
                                     const float* __restrict__ shexp_y, int n,
                                     float* __restrict__ out) {
  const int i =
      static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
  if (i < n) {
    out[i] = moe_y[i] + shexp_y[i];
  }
}

}  // namespace

int LaunchQwenMoeShexpSigmoid(const float* x, const float* gate_vec, int k,
                              int n_tokens, float* sig_out,
                              hipStream_t stream) {
  if (x == nullptr || gate_vec == nullptr || sig_out == nullptr) {
    return 1;
  }
  if (k <= 0 || k > kMaxK || n_tokens <= 0 || n_tokens > kMaxTokens) {
    return 1;
  }
  // Same sticky-error rule as LaunchQwenMoeRoute: a prior GEMV fallback can
  // leave a host error set after its replacement kernel has been queued.
  const hipError_t stale = hipGetLastError();
  if (stale != hipSuccess) {
    std::fprintf(stderr, "LKAT sigmoid stale %s\n", hipGetErrorString(stale));
  }
  QwenMoeShexpSigmoidKernel<<<n_tokens, kThreads, 0, stream>>>(
      x, gate_vec, k, n_tokens, sig_out);
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    std::fprintf(stderr, "LKAT sigmoid launch %s\n", hipGetErrorString(err));
    return 2;
  }
  return 0;
}

int LaunchQwenMoeCombine(const float* moe_y, const float* shexp_y, int hidden,
                         int n_tokens, float* out, hipStream_t stream) {
  if (moe_y == nullptr || shexp_y == nullptr || out == nullptr) {
    return 1;
  }
  if (hidden <= 0 || n_tokens <= 0 || n_tokens > kMaxTokens) {
    return 1;
  }
  const std::size_t n =
      static_cast<std::size_t>(hidden) * static_cast<std::size_t>(n_tokens);
  if (n / static_cast<std::size_t>(hidden) !=
          static_cast<std::size_t>(n_tokens) ||
      n > static_cast<std::size_t>(kMaxTokens) *
              static_cast<std::size_t>(kMaxK)) {
    return 1;
  }
  const int blocks =
      static_cast<int>((n + kThreads - 1) / static_cast<std::size_t>(kThreads));
  const hipError_t stale = hipGetLastError();
  if (stale != hipSuccess) {
    std::fprintf(stderr, "LKAT combine stale %s\n", hipGetErrorString(stale));
  }
  QwenMoeCombineKernel<<<blocks, kThreads, 0, stream>>>(
      moe_y, shexp_y, static_cast<int>(n), out);
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    std::fprintf(stderr, "LKAT combine launch %s\n", hipGetErrorString(err));
    return 2;
  }
  return 0;
}
