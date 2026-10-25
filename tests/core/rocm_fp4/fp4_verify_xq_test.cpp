// Super ROC QVO: gufo_fp4_gemv_verify_m_xq reuses the Q8 copy of x that the
// stream's slot set holds. It must be bit-identical to gufo_fp4_gemv_verify_m
// for the same x, m and k, and refuse (GUFO_FP4_ERR_ARG) when the copy is of
// a different x, m or k. Runs on a live ROCm device (label gpu).
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/core/rocm_fp4/gufo_fp4_gemv_q8.hpp"
#include "src/core/rocm_fp4/gufo_fp4_quant.hpp"

namespace {
int failures = 0;
void Expect(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok)
    ++failures;
}
#define CHECK_HIP(x)                                           \
  do {                                                         \
    const hipError_t e_ = (x);                                 \
    if (e_ != hipSuccess) {                                    \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_)); \
      return 1;                                                \
    }                                                          \
  } while (0)
}  // namespace

int main() {
  const int k = 2048, n1 = 512, n2 = 96, m = 4;
  const int qtype = GUFO_FP4_Q4_0_ROCMFP4_FAST;
  const std::size_t rb = gufo_fp4_row_bytes(qtype, k);
  if (rb == 0) {
    std::printf("FAIL row_bytes\n");
    return 1;
  }
  std::vector<std::uint8_t> w1(static_cast<std::size_t>(n1) * rb),
      w2(static_cast<std::size_t>(n2) * rb);
  for (std::size_t i = 0; i < w1.size(); ++i)
    w1[i] = static_cast<std::uint8_t>((i * 2654435761u) >> 13);
  for (std::size_t i = 0; i < w2.size(); ++i)
    w2[i] = static_cast<std::uint8_t>((i * 40503u + 7u) >> 5);
  std::vector<float> hx(static_cast<std::size_t>(m) * k),
      hx2(static_cast<std::size_t>(m) * k);
  std::srand(12345);
  for (auto& v : hx)
    v = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 4.0f -
        2.0f;
  for (auto& v : hx2)
    v = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 4.0f -
        2.0f;

  hipStream_t s = nullptr;
  CHECK_HIP(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  if (gufo_fp4_slot_stream_attach(s) < 0) {
    std::printf("FAIL slot_stream_attach\n");
    return 1;
  }
  if (gufo_fp4_verify_m_reserve(8 * k) != GUFO_FP4_OK) {
    std::printf("FAIL verify_m_reserve\n");
    return 1;
  }
  void *dw1 = nullptr, *dw2 = nullptr;
  float *dx = nullptr, *dx2 = nullptr, *dy_ref = nullptr, *dy_xq = nullptr,
        *dy1 = nullptr;
  CHECK_HIP(hipMalloc(&dw1, w1.size()));
  CHECK_HIP(hipMalloc(&dw2, w2.size()));
  CHECK_HIP(hipMalloc(&dx, hx.size() * sizeof(float)));
  CHECK_HIP(hipMalloc(&dx2, hx2.size() * sizeof(float)));
  CHECK_HIP(
      hipMalloc(&dy_ref, static_cast<std::size_t>(m) * n2 * sizeof(float)));
  CHECK_HIP(
      hipMalloc(&dy_xq, static_cast<std::size_t>(m) * n2 * sizeof(float)));
  CHECK_HIP(hipMalloc(&dy1, static_cast<std::size_t>(m) * n1 * sizeof(float)));
  CHECK_HIP(hipMemcpy(dw1, w1.data(), w1.size(), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(dw2, w2.data(), w2.size(), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(dx, hx.data(), hx.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(dx2, hx2.data(), hx2.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  // Reference: the second member quantises x itself.
  Expect(gufo_fp4_gemv_verify_m(qtype, dw2, dx, dy_ref, k, n2, m, s) ==
             GUFO_FP4_OK,
         "verify_m W2 (reference)");
  // Group: the first member quantises x, the second reuses it.
  Expect(
      gufo_fp4_gemv_verify_m(qtype, dw1, dx, dy1, k, n1, m, s) == GUFO_FP4_OK,
      "verify_m W1 (group first member)");
  Expect(gufo_fp4_gemv_verify_m_xq(qtype, dw2, dx, dy_xq, k, n2, m, s) ==
             GUFO_FP4_OK,
         "verify_m_xq W2 (reuse)");
  CHECK_HIP(hipStreamSynchronize(s));
  std::vector<float> ref(static_cast<std::size_t>(m) * n2),
      xq(static_cast<std::size_t>(m) * n2);
  CHECK_HIP(hipMemcpy(ref.data(), dy_ref, ref.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(xq.data(), dy_xq, xq.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  Expect(std::memcmp(ref.data(), xq.data(), ref.size() * sizeof(float)) == 0,
         "reuse is bit-identical to quantising again");
  // Refusals: a different x, m or k than the held copy.
  Expect(gufo_fp4_gemv_verify_m_xq(qtype, dw2, dx2, dy_xq, k, n2, m, s) ==
             GUFO_FP4_ERR_ARG,
         "refuses another x");
  Expect(gufo_fp4_gemv_verify_m_xq(qtype, dw2, dx, dy_xq, k, n2, 3, s) ==
             GUFO_FP4_ERR_ARG,
         "refuses another m");
  Expect(gufo_fp4_gemv_verify_m_xq(qtype, dw2, dx, dy_xq, 1024, n2, m, s) ==
             GUFO_FP4_ERR_ARG,
         "refuses another k");
  // After quantising x2, x is no longer held.
  Expect(
      gufo_fp4_gemv_verify_m(qtype, dw1, dx2, dy1, k, n1, m, s) == GUFO_FP4_OK,
      "verify_m W1 on x2");
  Expect(gufo_fp4_gemv_verify_m_xq(qtype, dw2, dx, dy_xq, k, n2, m, s) ==
             GUFO_FP4_ERR_ARG,
         "refuses the old x after a new quantisation");
  CHECK_HIP(hipStreamSynchronize(s));
  (void)hipFree(dw1);
  (void)hipFree(dw2);
  (void)hipFree(dx);
  (void)hipFree(dx2);
  (void)hipFree(dy_ref);
  (void)hipFree(dy_xq);
  (void)hipFree(dy1);
  (void)hipStreamDestroy(s);
  std::printf("fp4_verify_xq_test: %s\n", failures == 0 ? "passed" : "FAILED");
  return failures == 0 ? 0 : 1;
}
