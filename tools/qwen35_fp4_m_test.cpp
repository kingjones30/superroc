// lqw:spec6. Batched ROCmFP4 GEMV vs m separate m=1 calls, bitwise, on real
// 27B layers. gpu-run only.
#include <hip/hip_runtime.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/rocm_fp4/gufo_fp4_gemv_q8.hpp"

namespace {

void Die(const char* what) {
  std::fprintf(stderr, "FP4_M_FAIL %s\n", what);
  std::exit(1);
}

void Hip(hipError_t error, const char* what) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "FP4_M_FAIL %s: %s\n", what, hipGetErrorString(error));
    std::exit(1);
  }
}

int RowBytes(int qtype, int k) {
  const int blocks = k / 32;
  if (qtype == 100) {
    return blocks * 18;
  }
  if (qtype == 101) {
    return blocks * 17;
  }
  return 0;
}

bool BitExact(const std::vector<float>& got, const std::vector<float>& ref,
              int m, int n, const char* name) {
  if (got.size() != ref.size()) {
    std::fprintf(stderr, "FP4_M_FAIL %s size %zu vs %zu\n", name, got.size(),
                 ref.size());
    return false;
  }
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const std::uint32_t a = std::bit_cast<std::uint32_t>(got[i]);
    const std::uint32_t b = std::bit_cast<std::uint32_t>(ref[i]);
    if (a != b) {
      const int row = static_cast<int>(i % static_cast<std::size_t>(n));
      const int token = static_cast<int>(i / static_cast<std::size_t>(n));
      std::fprintf(stderr,
                   "FP4_M_FAIL %s m=%d token=%d row=%d got=%a ref=%a\n", name,
                   m, token, row, got[i], ref[i]);
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const char* path =
      argc > 1 ? argv[1]
               : "/path/to/models/qwen38-27b-bench/"
                 "Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf";
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    Die(error.c_str());
  }

  struct Case {
    const char* name;
    int n_sub;  // rows dropped from the tensor's n (odd-n tail coverage)
  };
  const Case cases[] = {
      {"blk.0.ffn_gate.weight", 0},
      {"blk.0.ffn_down.weight", 0},
      {"blk.0.attn_qkv.weight", 0},
      {"blk.17.ffn_gate.weight", 0},
      {"output.weight", 0},
      // lqw:spec7. Odd-n tail pair of the two-row verify kernel.
      {"blk.0.ffn_down.weight", 1},
  };

  bool all = true;
  for (const Case& c : cases) {
    const char* name = c.name;
    const auto* info = reader->FindTensor(name);
    if (info == nullptr || info->data == nullptr || info->dimensions.size() < 2) {
      std::fprintf(stderr, "FP4_M_FAIL missing %s\n", name);
      return 1;
    }
    const int k = static_cast<int>(info->dimensions[0]);
    const int n = static_cast<int>(info->dimensions[1]) - c.n_sub;
    const int qtype = static_cast<int>(info->type);
    const int row = RowBytes(qtype, k);
    if (row <= 0 || (k % 32) != 0) {
      std::fprintf(stderr, "FP4_M_FAIL %s type=%d k=%d n=%d\n", name, qtype, k,
                   n);
      return 1;
    }
    const std::size_t bytes =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(n);
    if (info->size_bytes < bytes) {
      std::fprintf(stderr, "FP4_M_FAIL %s bytes %zu < %zu\n", name,
                   info->size_bytes, bytes);
      return 1;
    }

    std::uint8_t* weights = nullptr;
    Hip(hipMalloc(&weights, bytes), "malloc weights");
    Hip(hipMemcpy(weights, info->data, bytes, hipMemcpyHostToDevice),
        "copy weights");

    const int max_m = 8;
    const int total = max_m * k;
    if (gufo_fp4_verify_m_reserve(total) != 0) {
      Die("reserve");
    }

    float* x = nullptr;
    float* y = nullptr;
    Hip(hipMalloc(&x, static_cast<std::size_t>(total) * sizeof(float)),
        "malloc x");
    Hip(hipMalloc(&y, static_cast<std::size_t>(max_m) * static_cast<std::size_t>(n) *
                          sizeof(float)),
        "malloc y");

    for (int m = 2; m <= 8; ++m) {
      std::vector<float> host_x(static_cast<std::size_t>(m) * static_cast<std::size_t>(k));
      for (int t = 0; t < m; ++t) {
        for (int i = 0; i < k; ++i) {
          host_x[static_cast<std::size_t>(t) * static_cast<std::size_t>(k) +
                 static_cast<std::size_t>(i)] =
              std::sin(0.017F * static_cast<float>(i + 1) + static_cast<float>(t)) *
              (0.5F + 0.1F * static_cast<float>(t));
        }
      }
      Hip(hipMemcpy(x, host_x.data(), host_x.size() * sizeof(float),
                    hipMemcpyHostToDevice),
          "copy x");

      hipEvent_t ev0{};
      hipEvent_t ev1{};
      Hip(hipEventCreate(&ev0), "event");
      Hip(hipEventCreate(&ev1), "event");
      Hip(hipEventRecord(ev0, nullptr), "record");
      for (int t = 0; t < m; ++t) {
        if (gufo_fp4_gemv_fp4(qtype, weights, x + static_cast<std::size_t>(t) * k,
                              y + static_cast<std::size_t>(t) * n, k, n,
                              nullptr) != 0) {
          Die("m=1 gemv");
        }
      }
      Hip(hipEventRecord(ev1, nullptr), "record");
      Hip(hipEventSynchronize(ev1), "sync serial");
      float serial_ms = 0.0F;
      Hip(hipEventElapsedTime(&serial_ms, ev0, ev1), "elapsed");
      std::vector<float> ref(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
      Hip(hipMemcpy(ref.data(), y, ref.size() * sizeof(float),
                    hipMemcpyDeviceToHost),
          "copy ref");

      Hip(hipEventRecord(ev0, nullptr), "record");
      if (gufo_fp4_gemv_verify_m(qtype, weights, x, y, k, n, m, nullptr) != 0) {
        Die("verify gemv");
      }
      Hip(hipEventRecord(ev1, nullptr), "record");
      Hip(hipEventSynchronize(ev1), "sync batch");
      float batch_ms = 0.0F;
      Hip(hipEventElapsedTime(&batch_ms, ev0, ev1), "elapsed");
      std::vector<float> got(ref.size());
      Hip(hipMemcpy(got.data(), y, got.size() * sizeof(float),
                    hipMemcpyDeviceToHost),
          "copy got");
      Hip(hipEventDestroy(ev0), "destroy");
      Hip(hipEventDestroy(ev1), "destroy");

      if (!BitExact(got, ref, m, n, name)) {
        all = false;
        break;
      }
      // lqw:spec7. Row count, not tensor bytes: the odd-n case reads fewer.
      const double gb =
          static_cast<double>(row) * static_cast<double>(n) / 1.0e9;
      std::printf(
          "FP4_M_PASS %s type=%d m=%d n=%d k=%d serial_ms=%.3f batch_ms=%.3f "
          "once_GBps=%.1f\n",
          name, qtype, m, n, k, serial_ms, batch_ms,
          batch_ms > 0.0F ? gb / (static_cast<double>(batch_ms) / 1.0e3) : 0.0);
      std::fflush(stdout);
    }
    Hip(hipFree(weights), "free w");
    Hip(hipFree(x), "free x");
    Hip(hipFree(y), "free y");
    if (!all) {
      return 1;
    }
  }
  std::printf("FP4_M_ALL_PASS\n");
  return 0;
}
