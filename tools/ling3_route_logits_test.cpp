// ling3-route-logits-test — is the TILED router logits kernel bit-identical to
// k_route_batch_logits, the one CTA per (expert, token) pair it replaces?
//
// PHASE8 item 1. This is the whole gate. Ling's router has a near-tie at block 2 that flips
// on a 1-ulp change (LS2 measured it), so "the same to six figures" is worthless here --
// the two kernels have to agree on EVERY BIT, or the routing moves and every text moves
// with it. The tiled form keeps the old per-row order exactly: lane t accumulates
// row[t + 64j] * x[t + 64j] for increasing j, then the same binary tree off = 32,16,8,4,2,1.
//
// Runs at the smallest shape first, per the lane's rule after the 2026-10-02 fault.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/models/ling3/kernels/ling3_moe_group.hpp"
#include "src/models/ling3/kernels/ling3_quant.hpp"

namespace {

int failures = 0;
#define CHECK(cond, msg)                                                        \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (msg));       \
      ++failures;                                                               \
    }                                                                           \
  } while (0)

std::uint64_t rng = 0x243F6A8885A308D3ULL;
std::uint32_t Next() {
  rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<std::uint32_t>(rng >> 32);
}
float Nf(float lo, float hi) {
  return lo + (hi - lo) * (float) Next() / 4294967296.0F;
}

// The kernel being replaced, transcribed from ling3_gemm.hip exactly.
__global__ void k_route_batch_logits_ref(const float * __restrict__ x,
                                         const float * __restrict__ gate,
                                         int k, int n_expert,
                                         float * __restrict__ logits) {
    const int e = blockIdx.x;
    const int t = blockIdx.y;
    const float * row = gate + (size_t) e * (size_t) k;
    const float * xt = x + (size_t) t * (size_t) k;
    float s = 0.f;
    for (int i = (int) threadIdx.x; i < k; i += (int) blockDim.x) {
        s += row[i] * xt[i];
    }
    __shared__ float red[64];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int off = 32; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) {
            red[threadIdx.x] += red[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        logits[(size_t) t * (size_t) n_expert + e] = red[0];
    }
}

void RunCase(const char * label, int n_tokens, int n_expert, int k, int ty) {
  std::vector<float> x((std::size_t) n_tokens * k), gate((std::size_t) n_expert * k);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = Nf(-3.f, 3.f);
  for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = Nf(-1.f, 1.f);

  float * d_x = nullptr; float * d_g = nullptr; float * d_a = nullptr; float * d_b = nullptr;
  const std::size_t lg = (std::size_t) n_tokens * n_expert * sizeof(float);
  const size_t pad = LING3_TAIL_PAD;
  bool a = hipMalloc(&d_x, (x.size() + pad) * sizeof(float)) == hipSuccess;
  a = a && hipMalloc(&d_g, (gate.size() + pad) * sizeof(float)) == hipSuccess;
  a = a && hipMalloc(&d_a, lg + pad) == hipSuccess;
  a = a && hipMalloc(&d_b, lg + pad) == hipSuccess;
  CHECK(a, "device allocations");
  if (!a) return;
  hipMemcpy(d_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_g, gate.data(), gate.size() * sizeof(float), hipMemcpyHostToDevice);
  hipMemset(d_a, 0x7f, lg);
  hipMemset(d_b, 0x7f, lg);

  k_route_batch_logits_ref<<<dim3(n_expert, n_tokens), 64>>>(d_x, d_g, k, n_expert, d_a);
  CHECK(hipDeviceSynchronize() == hipSuccess, "ref sync");
  CHECK(ling3_route_logits_tiled(d_x, d_g, k, n_expert, n_tokens, ty, d_b) == LING3_OK, "tiled rc");
  CHECK(hipDeviceSynchronize() == hipSuccess, "tiled sync");

  std::vector<float> ra(n_tokens * n_expert), rb(n_tokens * n_expert);
  hipMemcpy(ra.data(), d_a, lg, hipMemcpyDeviceToHost);
  hipMemcpy(rb.data(), d_b, lg, hipMemcpyDeviceToHost);
  int diff = 0;
  float worst = 0.f;
  for (std::size_t i = 0; i < ra.size(); ++i) {
    if (ra[i] != rb[i]) {
      ++diff;
      const float d2 = std::fabs(ra[i] - rb[i]);
      if (d2 > worst) worst = d2;
    }
  }
  CHECK(diff == 0, "tiled logits differ from the reference (router near-tie would move)");
  std::printf("  LOGITS %-22s n_tokens=%5d n_expert=%4d k=%5d ty=%2d  differing=%d/%zu  "
              "max_abs=%.3g\n", label, n_tokens, n_expert, k, ty, diff, ra.size(), worst);
  hipFree(d_x); hipFree(d_g); hipFree(d_a); hipFree(d_b);
}

}  // namespace

int main() {
  int nt = 8, ne = 16, k = 256, ty = LS1_ROUTE_TY;
  if (const char * e = getenv("LOGITS_NT")) nt = atoi(e);
  if (const char * e = getenv("LOGITS_NE")) ne = atoi(e);
  if (const char * e = getenv("LOGITS_K")) k = atoi(e);
  if (const char * e = getenv("LOGITS_TY")) ty = atoi(e);
  std::printf("== router logits test: n_tokens=%d n_expert=%d k=%d ty=%d\n", nt, ne, k, ty);
  RunCase("smallest", nt, ne, k, ty);
  if (nt >= 512 && ne >= 512 && k >= 2048) {
    RunCase("real shape", 512, 512, 2048, ty);
  }
  for (int t = 1; t <= LS1_ROUTE_TY; ++t) {
    RunCase("tile sweep", 37, 19, 131, t);
  }
  if (failures) {
    std::printf("FAIL: %d check(s)\n", failures);
    return 1;
  }
  std::printf("PASS: tiled router logits are BIT-IDENTICAL to k_route_batch_logits\n");
  return 0;
}
