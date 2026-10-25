// Super ROC QKS: the Qwen fp16 KV store (QwenKvF16) saturates at +-65504
// instead of overflowing to +-inf, keeps NaN, and is bit-identical to
// __float2half_rn for every input that was finite before (|x| < 65520).
// Runs on a live ROCm device (label gpu).
#include "src/models/qwen/hip/detail/kv_f16_store.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

__global__ void StoreBoth(const float* in, std::uint16_t* sat, std::uint16_t* plain, int n) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  sat[i] = __half_as_ushort(gufo::hip::QwenKvF16(in[i]));
  plain[i] = __half_as_ushort(__float2half_rn(in[i]));
}

bool Ok(hipError_t e, const char* what) {
  if (e != hipSuccess) std::printf("FAIL %s: %s\n", what, hipGetErrorString(e));
  return e == hipSuccess;
}

}  // namespace

int main() {
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> in = {0.0F, -0.0F, 1.5F, -2.25F, 6.0e-8F, -6.0e-8F, 1000.0F, 65504.0F, -65504.0F,
                           65519.0F, -65519.0F, 65520.0F, -65520.0F, 70000.0F, 1.0e6F, -1.0e6F,
                           3.0e38F, -3.0e38F, inf, -inf, std::nanf("")};
  for (int k = 0; k < 8192; ++k) {  // deterministic sweep over the finite half range
    in.push_back(-65519.0F + 131038.0F * static_cast<float>(k) / 8191.0F);
  }
  const int n = static_cast<int>(in.size());
  float* d_in = nullptr;
  std::uint16_t* d_sat = nullptr;
  std::uint16_t* d_plain = nullptr;
  if (!Ok(hipMalloc(&d_in, n * sizeof(float)), "malloc in") ||
      !Ok(hipMalloc(&d_sat, n * sizeof(std::uint16_t)), "malloc sat") ||
      !Ok(hipMalloc(&d_plain, n * sizeof(std::uint16_t)), "malloc plain") ||
      !Ok(hipMemcpy(d_in, in.data(), n * sizeof(float), hipMemcpyHostToDevice), "copy in")) {
    return 1;
  }
  StoreBoth<<<(n + 255) / 256, 256>>>(d_in, d_sat, d_plain, n);
  std::vector<std::uint16_t> sat(n), plain(n);
  if (!Ok(hipGetLastError(), "launch") || !Ok(hipDeviceSynchronize(), "sync") ||
      !Ok(hipMemcpy(sat.data(), d_sat, n * sizeof(std::uint16_t), hipMemcpyDeviceToHost), "copy sat") ||
      !Ok(hipMemcpy(plain.data(), d_plain, n * sizeof(std::uint16_t), hipMemcpyDeviceToHost), "copy plain")) {
    return 1;
  }
  int failures = 0, identical = 0, saturated = 0;
  for (int i = 0; i < n; ++i) {
    const float x = in[i];
    const bool is_nan_half = (sat[i] & 0x7C00U) == 0x7C00U && (sat[i] & 0x03FFU) != 0;
    if (std::isnan(x)) {
      if (!is_nan_half) { std::printf("FAIL NaN not kept: 0x%04x\n", sat[i]); ++failures; }
    } else if (std::fabs(x) < 65520.0F) {
      if (sat[i] != plain[i]) { std::printf("FAIL x=%g sat=0x%04x plain=0x%04x\n", x, sat[i], plain[i]); ++failures; }
      ++identical;
    } else {
      const std::uint16_t want = x > 0 ? 0x7BFFU : 0xFBFFU;
      const std::uint16_t was = x > 0 ? 0x7C00U : 0xFC00U;
      if (sat[i] != want || plain[i] != was) {
        std::printf("FAIL x=%g sat=0x%04x (want 0x%04x) plain=0x%04x (want 0x%04x)\n", x, sat[i], want, plain[i], was);
        ++failures;
      }
      ++saturated;
    }
  }
  (void)hipFree(d_in);
  (void)hipFree(d_sat);
  (void)hipFree(d_plain);
  std::printf("qwen_kv_f16_test: %d bit-identical, %d saturated (were inf), NaN kept; %s\n", identical, saturated,
              failures == 0 ? "passed" : "FAILED");
  return failures == 0 ? 0 : 1;
}
