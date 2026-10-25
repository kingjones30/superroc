// ling3-argmax-test -- LA1 bit-identical check of the two-phase argmax.
//
// The old argmax scanned the whole row with one 256-thread block; LA1 splits it
// into LING3_ARGMAX_SLICES contiguous slices each reduced by its own block and
// combines the partials in a tiny second kernel. The spec says the new one has
// to be BYTE-FOR-BYTE identical to the CPU reference (and therefore to the old
// kernel). That holds by construction (the (max, lowest-index) reduction is
// order-independent), and this test proves it on the device:
//
//   1. CPU reference computes the exact same (max, lowest-index) answer.
//   2. GPU ops_argmax on RANDOM rows (n >= some thousands), then on a second
//      buffer of CRAFTED rows (exact ties, all -INF, NaNs, n = 1 / 31 /
//      157184, rows 1..4 with a stride) -- byte-for-byte against the CPU.
//   3. A CPU-reduced reference over the SAME inputs tells us the expected id,
//      and we require equality on every row.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/models/ling3/kernels/ling3_ops.hpp"
#include "src/models/ling3/kernels/ling3_gemv_q8.hpp"

namespace {

std::uint64_t rng = 0x9E3779B97F4A7C15ULL;
std::uint32_t Next() {
  rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<std::uint32_t>(rng >> 32);
}

int failures = 0;
#define CHECK(cond, msg)                                                        \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (msg));       \
      ++failures;                                                                \
    }                                                                           \
  } while (0)

/// CPU reference: max value, lowest index on a tie. NaN never wins (v > best_v).
int CpuArgmax(const float * x, int n) {
  float best_v = -INFINITY;
  int   best_i = 0;
  for (int i = 0; i < n; ++i) {
    const float v = x[i];
    if (v > best_v || (v == best_v && i < best_i)) {
      best_v = v;
      best_i = i;
    }
  }
  return best_i;
}

void RunRandomCase(int n, int rows, int stride) {
  if (rows <= 0) return;
  const int nt = n;                       // a row is `n` contiguous floats
  std::vector<float> h((size_t) rows * (size_t) nt);
  for (auto & v : h) {
    v = -50.f + 100.f * (float) Next() / 4294967296.0F;
  }
  // A few exact ties and a NaN sprinkled in, so the tie rule and the NaN rule
  // both get exercised on random data.
  for (int t = 0; t < 7 && (std::size_t) t < h.size(); ++t) {
    const std::size_t i = (std::size_t) (Next() % (std::uint32_t) (h.size() - 1));
    h[i + 1] = h[i];                      // an exact tie
  }
  if (n > 4) h[((std::size_t) (Next() % (std::uint32_t) rows)) * n + 2] =
      std::nanf("");

  float * d = nullptr;
  CHECK(hipMalloc(&d, h.size() * sizeof(float)) == hipSuccess, "malloc h");
  int32_t * dout = nullptr;
  CHECK(hipMalloc(&dout, (size_t) rows * sizeof(int32_t)) == hipSuccess, "malloc out");
  CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess,
        "copy h");
  // logits_tok == n (contiguous rows); stride is handled by the caller by
  // building rows with a larger pitch when asked.
  ::ling3::ops_argmax(d, n, rows, dout, /*logits_tok=*/n, /*stream=*/nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess, "sync");
  std::vector<int32_t> got(rows);
  CHECK(hipMemcpy(got.data(), dout, (size_t) rows * sizeof(int32_t), hipMemcpyDeviceToHost) ==
            hipSuccess,
        "copy out");
  bool ok = true;
  for (int r = 0; r < rows; ++r) {
    const int exp = CpuArgmax(h.data() + (std::size_t) r * n, n);
    if (got[r] != exp) {
      ok = false;
      if (ok || 1) {
        std::fprintf(stderr, "  row %d: got %d want %d (n=%d rows=%d)\n", r, (int) got[r], exp, n, rows);
      }
    }
  }
  CHECK(ok, "random argmax matches CPU");
  if (hipFree(d) != hipSuccess); if (hipFree(dout) != hipSuccess);
}

/// Crafted rows: exact ties across slice boundaries, all -INF, NaN on top, and
/// the max at index 0 / n-1 / middle. Strides a row over `pitch`.
void RunCraftedCase(int n, std::vector<float> row, int pitch) {
  if (row.empty()) return;
  const int rows = 1;
  std::vector<float> h((size_t) rows * (size_t) pitch, -INFINITY);
  for (std::size_t i = 0; i < row.size(); ++i) h[i] = row[i];
  float * d = nullptr; int32_t * dout = nullptr;
  CHECK(hipMalloc(&d, (size_t) rows * (size_t) pitch * sizeof(float)) == hipSuccess, "m c");
  CHECK(hipMalloc(&dout, (size_t) rows * sizeof(int32_t)) == hipSuccess, "m co");
  CHECK(hipMemcpy(d, h.data(), (size_t) rows * (size_t) pitch * sizeof(float),
                  hipMemcpyHostToDevice) == hipSuccess, "cp c");
  ::ling3::ops_argmax(d, n, rows, dout, /*logits_tok=*/pitch, /*stream=*/nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess, "sync c");
  int32_t got = -1; CHECK(hipMemcpy(&got, dout, sizeof(int32_t), hipMemcpyDeviceToHost) == hipSuccess, "co c");
  const int exp = CpuArgmax(h.data(), n);
  CHECK((int) got == exp, "crafted argmax matches CPU");
  if ((int) got != exp) {
    std::fprintf(stderr, "  crafted n=%d got %d want %d\n", n, (int) got, exp);
  }
  if (hipFree(d) != hipSuccess); if (hipFree(dout) != hipSuccess);
}

}  // namespace

void Timings() {
  const int N = 157184;
  const int kq = 2560;
  std::vector<float> hp(N, 0.f), hq(kq, 0.f);
  for (auto & v : hp) v = -5.f + 10.f * (float) Next() / 4294967296.0F;
  for (auto & v : hq) v = -3.f + 6.f * (float) Next() / 4294967296.0F;
  float * d = nullptr; int32_t * dout = nullptr;
  if (hipMalloc(&d, N * sizeof(float)) == hipSuccess &&
      hipMalloc(&dout, sizeof(int32_t)) == hipSuccess) {
    hipMemcpy(d, hp.data(), N * sizeof(float), hipMemcpyHostToDevice);
    ::ling3::ops_argmax(d, N, 1, dout, N, nullptr);
    hipDeviceSynchronize();
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    const int reps = 50;
    hipEventRecord(a, (hipStream_t) 0);
    for (int i = 0; i < reps; ++i) ::ling3::ops_argmax(d, N, 1, dout, N, nullptr);
    hipEventRecord(b, (hipStream_t) 0); hipEventSynchronize(b);
    float ms = 0.f; hipEventElapsedTime(&ms, a, b);
    char line[128];
    std::snprintf(line, sizeof(line), "  argmax n=%d: %.2f us/call", N, ms * 1000.f / (float) reps);
    std::printf("%s\n", line);
    hipEventDestroy(a); hipEventDestroy(b);
    hipFree(d); hipFree(dout);
  }
  float * dq = nullptr;
  if (hipMalloc(&dq, kq * sizeof(float)) == hipSuccess) {
    hipMemcpy(dq, hq.data(), kq * sizeof(float), hipMemcpyHostToDevice);
    ling3_quant_q8(dq, kq, LING3_Q8_SLOT_GENERIC);
    hipDeviceSynchronize();
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    const int reps = 200;
    hipEventRecord(a, (hipStream_t) 0);
    for (int i = 0; i < reps; ++i) ling3_quant_q8(dq, kq, LING3_Q8_SLOT_GENERIC);
    hipEventRecord(b, (hipStream_t) 0); hipEventSynchronize(b);
    float ms = 0.f; hipEventElapsedTime(&ms, a, b);
    char line[128];
    std::snprintf(line, sizeof(line), "  quant k=%d: %.3f us/call", kq, ms * 1000.f / (float) reps);
    std::printf("%s\n", line);
    hipEventDestroy(a); hipEventDestroy(b);
    hipFree(dq);
  }
}

int main() {
  std::printf("== LA1 argmax: two-phase vs CPU reference, byte-for-byte\n");
  Timings();

  // Random, many rows, several n including the real 157,184.
  const int ns[] = {1, 31, 256, 1234, 15697, 157184};
  for (int ni = 0; ni < (int) (sizeof(ns) / sizeof(ns[0])); ++ni) {
    const int n = ns[ni];
    const int rows = (n == 157184) ? 4 : 40;
    RunRandomCase(n, rows, n);
  }
  // A sweep over n just around slice boundaries: 128 equal slices of n elements
  // put the largest @ boundaries when n % 128 is small. exercise n = k*128+[0..130].
  for (int k = 0; k < 4; ++k) {
    RunRandomCase(k * 128 + 0, 24, k * 128);
    RunRandomCase(k * 128 + 1, 24, k * 128 + 1);
    RunRandomCase(k * 128 + 129, 24, k * 128 + 129);
  }

  // Crafted: max at index 0 / middle / n-1; exact ties (the tie lowest index);
  // all -INF -> 0; NaN on top (should not win); max exactly on a slice boundary.
  const int N = 157184;
  RunCraftedCase(N, std::vector<float>(N, -INFINITY), N);                    // all -INF -> 0
  {
    std::vector<float> r(N, -INFINITY); r[0] = 7.f; RunCraftedCase(N, r, N); // max @0
  }
  {
    std::vector<float> r(N, -INFINITY); r[N - 1] = 9.f; RunCraftedCase(N, r, N); // max @N-1
  }
  {
    std::vector<float> r(N, -INFINITY); r[N / 2] = 5.f; RunCraftedCase(N, r, N); // max @middle
  }
  {
    std::vector<float> r(N, -INFINITY); r[N / 3] = 5.f; r[2 * N / 3] = 5.f;  // exact tie
    RunCraftedCase(N, r, N);
  }
  {
    std::vector<float> r(N, -INFINITY);
    r[0] = std::nanf(""); r[1] = 1.f;                                        // NaN @0, normal max @1
    RunCraftedCase(N, r, N);
  }
  {
    std::vector<float> r(N, -INFINITY);
    // tie: last element of slice 0 (index 1227 = N/128 - 1) vs an interior index;
    // both are the max; the tie resolves to the lowest index 1227.
    const int b0 = N / 128 - 1;                  // 1227, last element of slice 0
    r[b0] = 8.f; r[78600] = 8.f;
    RunCraftedCase(N, r, N);
  }
  // n = 1 and n = 31 single-row crafted.
  RunCraftedCase(1, {5.f}, 1);
  RunCraftedCase(31, std::vector<float>(31, -INFINITY), 31);
  // rows > 1 with a stride (logits_tok > n): 4 rows, pitch 157200.
  {
    const int pitch = 157200;
    const int rows = 4;
    std::vector<float> h((std::size_t) rows * pitch, -INFINITY);
    h[0 * pitch + 100] = 3.f;
    h[1 * pitch + 50] = 4.f;
    h[2 * pitch + 157183] = 6.f;
    h[3 * pitch + 0] = -2.f;             // a row whose answer is index 0
    // seeds so each row has a definite max
    h[0 * pitch + 0] = 3.5f;  h[3 * pitch + 5] = 1.f;
    float * d = nullptr; int32_t * dout = nullptr;
    CHECK(hipMalloc(&d, h.size() * sizeof(float)) == hipSuccess, "m str");
    CHECK(hipMalloc(&dout, (std::size_t) rows * sizeof(int32_t)) == hipSuccess, "m stro");
    CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess, "cp str");
    ::ling3::ops_argmax(d, 157184, rows, dout, /*logits_tok=*/pitch, nullptr);
    CHECK(hipDeviceSynchronize() == hipSuccess, "sync str");
    std::vector<int32_t> got(rows);
    CHECK(hipMemcpy(got.data(), dout, (std::size_t) rows * sizeof(int32_t), hipMemcpyDeviceToHost) == hipSuccess, "co str");
    for (int r = 0; r < rows; ++r) {
      const int exp = CpuArgmax(h.data() + (std::size_t) r * pitch, 157184);
      CHECK((int) got[r] == exp, "stride row argmax matches CPU");
      if ((int) got[r] != exp) std::fprintf(stderr, "  stride row %d got %d want %d\n", r, (int) got[r], exp);
    }
    if (hipFree(d) != hipSuccess); if (hipFree(dout) != hipSuccess);
  }

  if (failures == 0) {
    std::printf("ALL PASS: argmax byte-identical to CPU reference\n");
    return 0;
  }
  std::printf("FAIL: %d check(s)\n", failures);
  return 1;
}
