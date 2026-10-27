// ling3-route-gemm-test — the 2D register-tiled router-logits GEMM against the kernel it
// replaces, at Ling's real shape.
//
// PHASE9 item 1. This kernel is NOT bit-identical and is not meant to be: DECISIONS
// 2026-10-03 00:55 permits the router's accumulation order to change, because superroc's order
// was arbitrary to begin with and the block-2 near-tie moving is a numerics change like any
// other. So this reports what actually matters: max_abs and max_rel of the logits against the
// kernel shipped today, and -- the number Jay will care about most -- HOW MANY TOP-8 PICKS
// DIFFER per 512 tokens, because that is what can move a text.
//
// Runs at the smallest shape first, per the lane's rule after the 2026-10-02 fault.

#include <hip/hip_runtime.h>

#include <algorithm>
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

std::uint64_t rng = 0x9E3779B97F4A7C15ULL;
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

// The top-n_used experts a router would pick, in the order the tie rule produces. The
// production rule is an argsort over the scores; here all that matters for the gate is the
// SET, so a stable sort by descending score is enough, and ties resolve by index exactly as
// the kernel's strict comparisons do.
void TopK(const std::vector<float> & logits, int T, int E, int n_used, int t,
          std::vector<int> & out) {
  std::vector<int> idx(E);
  for (int e = 0; e < E; ++e) idx[e] = e;
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
    return logits[(std::size_t) t * E + a] > logits[(std::size_t) t * E + b];
  });
  out.assign(idx.begin(), idx.begin() + n_used);
}

void RunCase(const char * label, int T, int E, int K, int n_used,
             float x_scale = 1.f, float g_scale = 1.f) {
  std::vector<float> x((std::size_t) T * K), gate((std::size_t) E * K);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = Nf(-3.f, 3.f) * x_scale;
  for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = Nf(-1.f, 1.f) * g_scale;

  float * d_x = nullptr; float * d_g = nullptr; float * d_a = nullptr; float * d_b = nullptr;
  float * d_c = nullptr;
  const std::size_t lg = (std::size_t) T * E * sizeof(float);
  const size_t pad = LING3_TAIL_PAD;
  bool a = hipMalloc(&d_x, (x.size() + pad) * sizeof(float)) == hipSuccess;
  a = a && hipMalloc(&d_g, (gate.size() + pad) * sizeof(float)) == hipSuccess;
  a = a && hipMalloc(&d_a, lg + pad) == hipSuccess;
  a = a && hipMalloc(&d_b, lg + pad) == hipSuccess;
  a = a && hipMalloc(&d_c, lg + pad) == hipSuccess;
  CHECK(a, "device allocations");
  if (!a) return;
  hipMemcpy(d_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_g, gate.data(), gate.size() * sizeof(float), hipMemcpyHostToDevice);
  hipMemset(d_a, 0x7f, lg); hipMemset(d_b, 0x7f, lg); hipMemset(d_c, 0x7f, lg);

  k_route_batch_logits_ref<<<dim3(E, T), 64>>>(d_x, d_g, K, E, d_a);
  CHECK(hipDeviceSynchronize() == hipSuccess, "ref sync");
  CHECK(ling3_route_logits_gemm_launch(d_x, d_g, K, E, T, d_b) == LING3_OK, "gemm rc");
  CHECK(hipDeviceSynchronize() == hipSuccess, "gemm sync");
  // Run it twice: determinism is a property, not an accident.
  CHECK(ling3_route_logits_gemm_launch(d_x, d_g, K, E, T, d_c) == LING3_OK, "gemm rc 2");
  CHECK(hipDeviceSynchronize() == hipSuccess, "gemm sync 2");

  std::vector<float> ra(T * E), rb(T * E), rc(T * E);
  hipMemcpy(ra.data(), d_a, lg, hipMemcpyDeviceToHost);
  hipMemcpy(rb.data(), d_b, lg, hipMemcpyDeviceToHost);
  hipMemcpy(rc.data(), d_c, lg, hipMemcpyDeviceToHost);

  int nondet = 0;
  for (std::size_t i = 0; i < rb.size(); ++i) if (rb[i] != rc[i]) ++nondet;
  CHECK(nondet == 0, "GEMM is not deterministic run to run");

  double max_abs = 0.0, max_rel = 0.0, rms_num = 0.0, rms_den = 0.0;
  for (std::size_t i = 0; i < ra.size(); ++i) {
    const double d2 = std::fabs((double) ra[i] - (double) rb[i]);
    if (d2 > max_abs) max_abs = d2;
    const double den = std::fabs((double) ra[i]) > 1e-6 ? std::fabs((double) ra[i]) : 1.0;
    if (d2 / den > max_rel) max_rel = d2 / den;
    rms_num += d2 * d2;
    rms_den += (double) ra[i] * (double) ra[i];
  }
  const double rms_rel = rms_den > 0 ? std::sqrt(rms_num / rms_den) : 0.0;

  int tokens_moved = 0, picks_moved = 0;
  std::vector<int> ta, tb;
  for (int t = 0; t < T; ++t) {
    TopK(ra, T, E, n_used, t, ta);
    TopK(rb, T, E, n_used, t, tb);
    int moved = 0;
    for (int j = 0; j < n_used; ++j) if (ta[j] != tb[j]) ++moved;
    if (moved) ++tokens_moved;
    picks_moved += moved;
  }
  std::printf("  vs_ref64 %-16s T=%4d E=%4d K=%5d  max_abs=%.4g max_rel=%.4g rms_rel=%.4g  "
              "nondet=%d  top%d moves: %d/%d tokens, %d/%d picks\n",
              label, T, E, K, max_abs, max_rel, rms_rel, nondet, n_used,
              tokens_moved, T, picks_moved, T * n_used);

  CHECK(ling3_route_logits_gemm_legacy_launch(d_x, d_g, K, E, T, d_a) == LING3_OK, "legacy rc");
  CHECK(hipDeviceSynchronize() == hipSuccess, "legacy sync");
  std::vector<float> rleg((std::size_t) T * E);
  hipMemcpy(rleg.data(), d_a, lg, hipMemcpyDeviceToHost);
  double leg_abs = 0.0, leg_rel = 0.0;
  for (std::size_t i = 0; i < rb.size(); ++i) {
    const double d2 = std::fabs((double) rleg[i] - (double) rb[i]);
    if (d2 > leg_abs) leg_abs = d2;
    const double den = std::fabs((double) rleg[i]) > 1e-6 ? std::fabs((double) rleg[i]) : 1.0;
    if (d2 / den > leg_rel) leg_rel = d2 / den;
  }
  int leg_tokens = 0, leg_picks = 0;
  for (int t = 0; t < T; ++t) {
    TopK(rleg, T, E, n_used, t, ta);
    TopK(rb, T, E, n_used, t, tb);
    int moved = 0;
    for (int j = 0; j < n_used; ++j) if (ta[j] != tb[j]) ++moved;
    if (moved) ++leg_tokens;
    leg_picks += moved;
  }
  std::printf("  vs_today %-16s max_abs=%.4g max_rel=%.4g  top%d moves: %d/%d tokens, %d/%d picks\n",
              label, leg_abs, leg_rel, n_used, leg_tokens, T, leg_picks, T * n_used);

  if ((long long) T * E * K >= 100000000LL) {
    for (int i = 0; i < 2; ++i) {
      ling3_route_logits_gemm_launch(d_x, d_g, K, E, T, d_b);
    }
    CHECK(hipDeviceSynchronize() == hipSuccess, "time warmup");
    std::vector<float> us;
    for (int i = 0; i < 7; ++i) {
      hipEvent_t ev0, ev1;
      hipEventCreate(&ev0);
      hipEventCreate(&ev1);
      hipEventRecord(ev0);
      ling3_route_logits_gemm_launch(d_x, d_g, K, E, T, d_b);
      hipEventRecord(ev1);
      hipEventSynchronize(ev1);
      float ms = 0.f;
      hipEventElapsedTime(&ms, ev0, ev1);
      us.push_back(ms * 1000.f);
      hipEventDestroy(ev0);
      hipEventDestroy(ev1);
    }
    std::sort(us.begin(), us.end());
    std::printf("  time %-16s best_us=%.1f med_us=%.1f\n", label, us.front(), us[us.size() / 2]);
  }

  hipFree(d_x); hipFree(d_g); hipFree(d_a); hipFree(d_b); hipFree(d_c);
}

}  // namespace

int main() {
  int T = 32, E = 32, K = 64, nu = 8;
  if (const char * e = getenv("GEMM_T")) T = atoi(e);
  if (const char * e = getenv("GEMM_E")) E = atoi(e);
  if (const char * e = getenv("GEMM_K")) K = atoi(e);
  if (const char * e = getenv("GEMM_NUSED")) nu = atoi(e);
  std::printf("== router GEMM test: T=%d E=%d K=%d n_used=%d\n", T, E, K, nu);
  RunCase("smallest", T, E, K, nu);
  RunCase("real shape", 512, 512, 2560, nu);
  RunCase("real narrow", 512, 512, 2560, nu, 2.f / 3.f, 0.05f);
  if (failures) {
    std::printf("FAIL: %d check(s)\n", failures);
    return 1;
  }
  std::printf("PASS: GEMM is deterministic; see the error and top-8 numbers above\n");
  return 0;
}
