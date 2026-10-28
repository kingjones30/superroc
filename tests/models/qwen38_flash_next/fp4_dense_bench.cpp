// LFP item 1: per-shape dense GEMM timing. Q8_0 via DenseF16Gemm (the ceiling),
// Q8_0 via QuantizeQ8Tiled+W8A8Gemm, type-101 via
// superroc_fp4_wmma_quant+gemm_dense. Random weights (timing only). HIP events,
// 20 reps, median. Run under gpu-run.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/core/rocm_fp4/superroc_fp4_wmma.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {
namespace rocm = superroc::models::qwen38_flash_next::rocm;

// 20:02 fault ruling: a dead context must FAIL the bench. Every HIP/hipblas
// call is checked; launches are followed by hipGetLastError + a checked sync.
const char* g_hip_case = "startup";
[[noreturn]] void LfpHipDie(hipError_t e, const char* what, int line) {
  std::printf("HIP FATAL [%s] %s:%d: %s (%d)\n", g_hip_case, what, line,
              hipGetErrorString(e), static_cast<int>(e));
  std::fflush(stdout);
  std::exit(1);
}
inline void LfpHipCheck(hipError_t e, const char* what, int line) {
  if (e != hipSuccess) {
    LfpHipDie(e, what, line);
  }
}
#define LFP_HIP_CHECK(x) LfpHipCheck((x), #x, __LINE__)
#define LFP_LAUNCHED() \
  LfpHipCheck(hipGetLastError(), "hipGetLastError", __LINE__)
inline void LfpBlasCheck(hipblasStatus_t e, const char* what, int line) {
  if (e != HIPBLAS_STATUS_SUCCESS) {
    std::printf("BLAS FATAL [%s] %s:%d: status %d\n", g_hip_case, what, line,
                static_cast<int>(e));
    std::fflush(stdout);
    std::exit(1);
  }
}
#define LFP_BLAS_CHECK(x) LfpBlasCheck((x), #x, __LINE__)
// superroc_fp4 lib calls return 0 (SUPERROC_FP4_OK) or an error code.
#define LFP_SUPERROC4_CHECK(x)                                        \
  do {                                                                \
    if ((x) != 0) {                                                   \
      std::printf("SUPERROC4 FATAL [%s]:%d\n", g_hip_case, __LINE__); \
      std::fflush(stdout);                                            \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

std::uint64_t rng = 0x123456789ABCDEFULL;
std::uint64_t Next() {
  rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return rng >> 11;
}

// Q8_0 weights: F16 scale + 32 int8 codes per block.
std::vector<std::uint8_t> MakeQ80(std::size_t m, std::size_t k) {
  const std::size_t nb = k / 32;
  std::vector<std::uint8_t> w(m * nb * 34);
  for (std::size_t r = 0; r < m; ++r) {
    for (std::size_t b = 0; b < nb; ++b) {
      std::uint8_t* blk = &w[(r * nb + b) * 34];
      const __half d = __float2half(0.02f);
      __builtin_memcpy(blk, &d, 2);
      for (int i = 0; i < 32; ++i)
        blk[2 + i] = static_cast<std::uint8_t>(Next() & 0xFF);
    }
  }
  return w;
}

// Type-101 weights: 16 code bytes + 1 scale byte per block.
std::vector<std::uint8_t> MakeFp4(std::size_t m, std::size_t k) {
  const std::size_t nb = k / 32;
  std::vector<std::uint8_t> w(m * nb * 17);
  for (std::size_t i = 0; i < m * nb; ++i) {
    std::uint8_t* blk = &w[i * 17];
    for (int j = 0; j < 16; ++j)
      blk[j] = static_cast<std::uint8_t>(Next() & 0xFF);
    blk[16] = static_cast<std::uint8_t>(Next() & 0xFF);
  }
  return w;
}

double Median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

struct Dev {
  void* p = nullptr;
  explicit Dev(std::size_t n) {
    if (hipMalloc(&p, n ? n : 1) != hipSuccess) {
      std::puts("OOM");
      std::exit(1);
    }
  }
  ~Dev() {
    if (p)
      LFP_HIP_CHECK(hipFree(p));
  }
};

int Run(std::size_t batch, std::size_t m, std::size_t k) {
  g_hip_case = __func__;
  const double flops = 2.0 * batch * m * k;
  const double q8bytes = static_cast<double>(m) * (k / 32) * 34;
  const double fp4bytes = static_cast<double>(m) * (k / 32) * 17;
  // Host data.
  const auto w_q8 = MakeQ80(m, k);
  const auto w_fp4 = MakeFp4(m, k);
  std::vector<float> xf(batch * k);
  for (float& v : xf)
    v = (static_cast<float>(Next() & 0xFFFF) / 65535.f - 0.5f);
  std::vector<__half> xh(batch * k);
  for (std::size_t i = 0; i < xf.size(); ++i)
    xh[i] = __float2half(xf[i]);
  // Device.
  Dev d_q8(w_q8.size()), d_fp4(w_fp4.size());
  Dev d_xf(xf.size() * 4), d_xh(xh.size() * 2), d_out(batch * m * 4);
  Dev d_q8t(rocm::Q8TiledBytes(batch, k));
  // Seat-sized scratch: the executor always allocates for max_batch.
  const std::size_t fp4_tiles = 130;
  Dev d_fq8(fp4_tiles * (k / 32) * 576);
  LFP_HIP_CHECK(
      hipMemcpy(d_q8.p, w_q8.data(), w_q8.size(), hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_fp4.p, w_fp4.data(), w_fp4.size(), hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_xf.p, xf.data(), xf.size() * 4, hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_xh.p, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
  hipStream_t st = nullptr;
  LFP_HIP_CHECK(hipStreamCreate(&st));
  hipEvent_t e0, e1;
  LFP_HIP_CHECK(hipEventCreate(&e0));
  LFP_HIP_CHECK(hipEventCreate(&e1));
  auto time_it = [&](auto fn, hipStream_t s, const char* what) {
    for (int i = 0; i < 3; ++i)
      fn();  // warmup
    LFP_HIP_CHECK(hipGetLastError());
    LFP_HIP_CHECK(hipStreamSynchronize(s));
    std::vector<float> ms;
    for (int i = 0; i < 20; ++i) {
      LFP_HIP_CHECK(hipEventRecord(e0, s));
      fn();
      LFP_HIP_CHECK(hipEventRecord(e1, s));
      LFP_HIP_CHECK(hipEventSynchronize(e1));
      float v = 0;
      LFP_HIP_CHECK(hipEventElapsedTime(&v, e0, e1));
      ms.push_back(v);
    }
    LFP_HIP_CHECK(hipGetLastError());
    LFP_HIP_CHECK(hipStreamSynchronize(s));
    return Median(ms);
  };
  const auto* wq = static_cast<const void*>(d_q8.p);
  const auto* wf = static_cast<const void*>(d_fp4.p);
  const auto* xhp = static_cast<const __half*>(d_xh.p);
  const auto* xfp = static_cast<const float*>(d_xf.p);
  auto* out = static_cast<float*>(d_out.p);
  const double t_f16 = time_it(
      [&] {
        rocm::DenseF16Gemm(wq, xhp, out, batch, m, k, st);
        LFP_LAUNCHED();
      },
      st, "f16");
  const double t_w8a8 = time_it(
      [&] {
        rocm::QuantizeQ8Tiled(xfp, d_q8t.p, batch, k, st);
        LFP_LAUNCHED();
        rocm::W8A8Gemm(wq, d_q8t.p, out, batch, m, k, st);
        LFP_LAUNCHED();
      },
      st, "w8a8");
  const double t_fp4 = time_it(
      [&] {
        LFP_SUPERROC4_CHECK(superroc_fp4_wmma_quant(
            xfp, nullptr, nullptr, d_fq8.p, (int)batch, (int)k));
        LFP_SUPERROC4_CHECK(superroc_fp4_wmma_gemm_dense(
            101, wf, d_fq8.p, out, (int)k, (int)m, (int)batch));
      },
      nullptr, "fp4");
  double t_sgemm = -1.0;
  if (m <= 640) {
    std::vector<float> wf32(m * k);
    for (float& v : wf32)
      v = (static_cast<float>(Next() & 0xFFFF) / 65535.f - 0.5f) * 0.04f;
    Dev d_wf32(wf32.size() * 4);
    LFP_HIP_CHECK(hipMemcpy(d_wf32.p, wf32.data(), wf32.size() * 4,
                            hipMemcpyHostToDevice));
    hipblasHandle_t blas = nullptr;
    LFP_BLAS_CHECK(hipblasCreate(&blas));
    LFP_BLAS_CHECK(hipblasSetStream(blas, st));
    const float alpha = 1.0F, beta = 0.0F;
    t_sgemm = time_it(
        [&] {
          LFP_BLAS_CHECK(hipblasSgemm(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, (int)m,
                                      (int)batch, (int)k, &alpha,
                                      static_cast<const float*>(d_wf32.p),
                                      (int)k, xfp, (int)k, &beta, out, (int)m));
        },
        st, "sgemm");
    LFP_BLAS_CHECK(hipblasDestroy(blas));
  }
  const double t_fp4f16 = time_it(
      [&] {
        rocm::Fp4DenseF16Gemm(wf, xhp, out, batch, m, k, true, st);
        LFP_LAUNCHED();
      },
      st, "fp4f16");
  const double t_narrow = time_it(
      [&] {
        rocm::NarrowActivations(xfp, d_xh.p, false, batch * k, st);
        LFP_LAUNCHED();
      },
      st, "narrow");
  std::printf(
      "%6zu x %-5zu  F16-Q8_0 %8.3f ms %6.1f TF  | W8A8-Q8_0 %8.3f ms %6.1f TF "
      " | FP4-W8A8 %8.3f ms %6.1f TF | narrow %.3f ms\n",
      m, k, t_f16, flops / t_f16 / 1e9, t_w8a8, flops / t_w8a8 / 1e9, t_fp4,
      flops / t_fp4 / 1e9, t_narrow);
  std::printf("    FP4-F16 %8.3f ms %6.1f TF (%.2f x F16-Q8_0)\n", t_fp4f16,
              flops / t_fp4f16 / 1e9, t_fp4f16 / t_f16);
  if (t_sgemm >= 0)
    std::printf("    F32-hipblas %.3f ms (%.1f TF)\n", t_sgemm,
                flops / t_sgemm / 1e9);
  std::printf("    weight GB/s: F16 %.0f  W8A8 %.0f  FP4 %.0f\n",
              q8bytes / t_f16 / 1e6, q8bytes / t_w8a8 / 1e6,
              fp4bytes / t_fp4 / 1e6);
  LFP_HIP_CHECK(hipEventDestroy(e0));
  LFP_HIP_CHECK(hipEventDestroy(e1));
  LFP_HIP_CHECK(hipStreamDestroy(st));
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  g_hip_case = __func__;
  std::size_t batch = 2048;
  if (argc > 1)
    batch = std::strtoul(argv[1], nullptr, 10);
  std::printf("batch=%zu\n", batch);
  // 96x2560 = the stacked ssm_alpha_beta (device_model Stack).
  const std::size_t shapes[][2] = {
      {10240, 2560}, {6144, 2560}, {2560, 6144}, {96, 2560},   {48, 2560},
      {12288, 2560}, {512, 2560},  {128, 2560},  {10240, 320}, {320, 10240},
      {4, 10240},    {640, 2560},  {2560, 640},  {2560, 2560}};
  for (const auto& s : shapes)
    Run(batch, s[0], s[1]);
  return 0;
}
