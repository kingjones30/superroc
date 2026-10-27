// NBFAST T2 step 1: microbench of the four Nanbeige (llama) prefill GEMM
// shapes on today's FP16 WMMA path vs the int8 WMMA W8A8 path.
//
// Timing only: weights are deterministic pseudo-random bytes in valid
// Q4_K/Q6_K layouts and outputs are never read back. Without NBF_BENCH in
// the environment the program runs two iterations per configuration as a
// launch smoke (ctest); with NBF_BENCH=1 it runs the full timing loop
// (NBF_BENCH_ITERS overrides the iteration count). All launches must
// succeed either way.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

using superroc::core::GgmlType;

struct Shape {
  const char* name;
  std::size_t m;  // output features
  std::size_t k;  // input features
  GgmlType type;
  bool dual;  // also time the fused gate+up variants
};

void FillRandom(std::vector<std::uint8_t>* buf, std::uint32_t seed) {
  std::uint32_t state = seed;
  for (auto& byte : *buf) {
    state = (state * 1664525U) + 1013904223U;
    byte = static_cast<std::uint8_t>(state >> 24U);
  }
}

template<typename Fn>
float TimeIt(const char* label, int warmup, int iters, Fn&& fn) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  HIP_CHECK(hipDeviceSynchronize());
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));
  float ms = 0.0F;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  std::printf("    %-26s %9.4f ms/iter\n", label, ms / iters);
  return ms / iters;
}

void PrintTflops(const char* label, double flops_per_iter, float ms) {
  std::printf("    %-26s %9.2f TFLOPS\n", label,
              flops_per_iter / (static_cast<double>(ms) * 1e9));
}

void RunShape(const Shape& shape, std::size_t batch, int iters) {
  const std::size_t weight_bytes =
      superroc::quant::EncodedSizeBytes(shape.type, shape.m * shape.k);
  if (weight_bytes == 0) {
    std::fprintf(stderr, "no encoded size for shape %s\n", shape.name);
    std::abort();
  }
  std::printf("shape %s batch=%zu m=%zu k=%zu type=%d iters=%d\n", shape.name,
              batch, shape.m, shape.k, static_cast<int>(shape.type), iters);
  const double flops = 2.0 * static_cast<double>(batch) *
                       static_cast<double>(shape.m) *
                       static_cast<double>(shape.k);

  std::vector<std::uint8_t> h_w(weight_bytes);
  FillRandom(&h_w, 0x1234567U);
  std::vector<std::uint8_t> h_w2(weight_bytes);
  FillRandom(&h_w2, 0x7654321U);
  std::vector<float> h_x(batch * shape.k, 0.25F);
  std::vector<std::uint16_t> h_x16(batch * shape.k, 0x3400U);  // fp16 0.25

  void* d_w = nullptr;
  void* d_w2 = nullptr;
  float* d_x = nullptr;
  void* d_x16 = nullptr;
  void* d_q8 = nullptr;
  void* d_q8_out = nullptr;
  float* d_y = nullptr;
  float* d_y2 = nullptr;
  const std::size_t q8_bytes =
      superroc::hip::QuantizedActivationBytes(batch, shape.k) + 4096;
  const std::size_t q8_out_bytes =
      superroc::hip::QuantizedActivationBytes(batch, shape.m) + 4096;
  HIP_CHECK(hipMalloc(&d_w, weight_bytes));
  HIP_CHECK(hipMalloc(&d_w2, weight_bytes));
  HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_x16, h_x16.size() * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_q8, q8_bytes));
  HIP_CHECK(hipMalloc(&d_q8_out, q8_out_bytes));
  HIP_CHECK(hipMalloc(&d_y, batch * shape.m * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y2, batch * shape.m * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), weight_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w2, h_w2.data(), weight_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x16, h_x16.data(), h_x16.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));

  superroc::hip::LaunchQuantizeActivationQ8_1FromFp32(d_x, d_q8, batch, shape.k);
  HIP_CHECK(hipDeviceSynchronize());

  const float fp16_ms = TimeIt("fp16_gemm", 2, iters, [&] {
    superroc::hip::LaunchBatchedQuantGEMMFp16(shape.type, d_w, d_x16, d_y, batch,
                                          shape.m, shape.k, nullptr);
  });
  PrintTflops("fp16_gemm", flops, fp16_ms);

  const float q8_ms = TimeIt("q8_gemm_prequantized", 2, iters, [&] {
    superroc::hip::LaunchBatchedQuantGEMMPreQuantized(
        shape.type, d_w, d_q8, d_y, batch, shape.m, shape.k, nullptr);
  });
  PrintTflops("q8_gemm_prequantized", flops, q8_ms);

  TimeIt("q8_quantize_only", 2, iters, [&] {
    superroc::hip::LaunchQuantizeActivationQ8_1FromFp32(d_x, d_q8, batch, shape.k,
                                                    nullptr);
  });
  TimeIt("q8_quantize+gemm", 2, iters, [&] {
    superroc::hip::LaunchQuantizeActivationQ8_1FromFp32(d_x, d_q8, batch, shape.k,
                                                    nullptr);
    superroc::hip::LaunchBatchedQuantGEMMPreQuantized(
        shape.type, d_w, d_q8, d_y, batch, shape.m, shape.k, nullptr);
  });

  if (shape.dual) {
    const float dual_fp16_ms = TimeIt("fp16_dual_swiglu", 2, iters, [&] {
      if (!superroc::hip::TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
              shape.type, shape.type, d_w, d_w2, d_x16, d_y2, batch, shape.m,
              shape.k, nullptr)) {
        std::fprintf(stderr, "fp16 dual swiglu refused\n");
        std::abort();
      }
    });
    PrintTflops("fp16_dual_swiglu", 2.0 * flops, dual_fp16_ms);

    const float dual_q8_ms = TimeIt("q8_dual", 2, iters, [&] {
      superroc::hip::LaunchBatchedDualQuantGEMMPreQuantized(
          shape.type, d_w, d_w2, d_q8, d_y, d_y2, batch, shape.m, shape.k,
          nullptr);
    });
    PrintTflops("q8_dual", 2.0 * flops, dual_q8_ms);

    if (superroc::hip::IsFusedSwiGluGemmEpilogueSupported(shape.type, batch,
                                                      shape.m, q8_out_bytes)) {
      const float epi_ms = TimeIt("q8_dual_swiglu_epilogue", 2, iters, [&] {
        superroc::hip::LaunchBatchedDualQuantGEMMSwiGLUQuantizeQ8_1(
            shape.type, d_w, d_w2, d_q8, d_y, d_q8_out, batch, shape.m, shape.k,
            nullptr);
      });
      PrintTflops("q8_dual_swiglu_epilogue", 2.0 * flops, epi_ms);
    } else {
      std::printf("    q8_dual_swiglu_epilogue   unsupported\n");
    }
  }

  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_w2));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_x16));
  HIP_CHECK(hipFree(d_q8));
  HIP_CHECK(hipFree(d_q8_out));
  HIP_CHECK(hipFree(d_y));
  HIP_CHECK(hipFree(d_y2));
}

}  // namespace

int main() {
  const int device_status = superroc::test::GateHipDevice(
      superroc::test::HipDeviceRequirement::kOptional, "llama prefill GEMM bench");
  if (device_status != superroc::test::kHipTestSuccess) {
    return device_status;
  }
  const bool full = std::getenv("NBF_BENCH") != nullptr;
  int iters_override = 0;
  if (const char* env = std::getenv("NBF_BENCH_ITERS")) {
    iters_override = std::atoi(env);
  }
  const Shape shapes[] = {
      {"attn_q/o", 7168, 7168, GgmlType::kQ4_K, false},
      {"attn_k/v", 1024, 7168, GgmlType::kQ4_K, false},
      {"attn_v_q6", 1024, 7168, GgmlType::kQ6_K, false},
      {"ffn_gate+up", 22016, 7168, GgmlType::kQ4_K, true},
      {"ffn_down", 7168, 22016, GgmlType::kQ4_K, false},
      {"ffn_down_q6", 7168, 22016, GgmlType::kQ6_K, false},
  };
  for (const std::size_t batch : {512U, 4096U}) {
    const int iters = iters_override > 0 ? iters_override
                      : full             ? (batch <= 512 ? 50 : 10)
                                         : 2;
    for (const Shape& shape : shapes) {
      RunShape(shape, batch, iters);
    }
  }
  std::printf("PASS llama_prefill_gemm_bench\n");
  return 0;
}
#else
int main() {
  return 77;
}
#endif
