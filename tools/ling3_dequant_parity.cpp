// ling3-dequant-parity — checks superroc's Q6_K and Q8_0 row decoders against the
// fork's, bit for bit, via dlopen of libggml-base.so.
//
// Needed because Ling's token_embd.weight is Q6_K: the ROCmFP4 path was
// verified but this one had not been, and a wrong embedding row diverges the
// very first layer norm.
//
// Usage: ling3-dequant-parity <libggml-base.so> [trials]

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"

namespace {

std::uint64_t Next(std::uint64_t* s) {
  *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
  return *s >> 33;
}

using DequantFn = void (*)(const void*, float*, std::int64_t);

int Compare(const char* label, superroc::core::GgmlType type, const char* symbol,
            std::size_t block_elems, std::size_t block_bytes, void* handle,
            const std::uint8_t* src, std::size_t n_elems) {
  auto theirs = reinterpret_cast<DequantFn>(dlsym(handle, symbol));
  if (theirs == nullptr) {
    std::printf("%-6s SYMBOL NOT EXPORTED (%s)\n", label, symbol);
    return 1;
  }
  std::vector<float> a(n_elems, -1.0F), b(n_elems, 2.0F);
  switch (type) {
    case superroc::core::GgmlType::kQ6_K:
      superroc::quant::DequantizeQ6_K(src, a.data(), n_elems);
      break;
    case superroc::core::GgmlType::kQ8_0:
      superroc::quant::DequantizeQ8_0(src, a.data(), n_elems);
      break;
    default:
      std::printf("%-6s unsupported type in this test\n", label);
      return 1;
  }
  theirs(src, b.data(), static_cast<std::int64_t>(n_elems));

  std::size_t bad = 0;
  for (std::size_t i = 0; i < n_elems; ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
      if (bad < 4) {
        std::printf("  %s[%zu]: superroc=%.9g fork=%.9g\n", label, i,
                    static_cast<double>(a[i]), static_cast<double>(b[i]));
      }
      ++bad;
    }
  }
  std::printf("%-6s %zu elems (%zu blocks of %zu/%zuB): %zu mismatches\n", label,
              n_elems, n_elems / block_elems, block_elems, block_bytes, bad);
  return bad == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <libggml-base.so> [trials]\n", argv[0]);
    return 2;
  }
  void* handle = dlopen(argv[1], RTLD_NOW);
  if (!handle) {
    std::fprintf(stderr, "dlopen: %s\n", dlerror());
    return 1;
  }
  const std::size_t trials = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 50;
  std::uint64_t rng = 0xDEADBEEFCAFEF00DULL;
  int failures = 0;

  std::printf("superroc block sizes: Q6_K=%zu elems/%zuB  Q8_0=%zu elems/%zuB\n",
              superroc::quant::QuantizedBlockElements(superroc::core::GgmlType::kQ6_K),
              superroc::quant::QuantizedRowBytes(superroc::core::GgmlType::kQ6_K, 256),
              superroc::quant::QuantizedBlockElements(superroc::core::GgmlType::kQ8_0),
              superroc::quant::QuantizedRowBytes(superroc::core::GgmlType::kQ8_0, 32));

  for (std::size_t t = 0; t < trials; ++t) {
    // Exactly the shapes this artifact uses: 2560-element rows.
    const std::size_t n6 = 2560;
    const std::size_t n8 = 2560;
    std::vector<std::uint8_t> q6(
        superroc::quant::QuantizedRowBytes(superroc::core::GgmlType::kQ6_K, n6));
    std::vector<std::uint8_t> q8(
        superroc::quant::QuantizedRowBytes(superroc::core::GgmlType::kQ8_0, n8));
    for (auto& b : q6) b = static_cast<std::uint8_t>(Next(&rng));
    for (auto& b : q8) b = static_cast<std::uint8_t>(Next(&rng));

    failures += Compare("Q6_K", superroc::core::GgmlType::kQ6_K,
                        "dequantize_row_q6_K", 256,
                        superroc::quant::QuantizedRowBytes(
                            superroc::core::GgmlType::kQ6_K, 256),
                        handle, q6.data(), n6);
    failures += Compare("Q8_0", superroc::core::GgmlType::kQ8_0,
                        "dequantize_row_q8_0", 32,
                        superroc::quant::QuantizedRowBytes(
                            superroc::core::GgmlType::kQ8_0, 32),
                        handle, q8.data(), n8);
  }
  std::printf("%s: %zu trials, %d failures\n", failures == 0 ? "PASS" : "FAIL",
              trials, failures);
  return failures == 0 ? 0 : 1;
}
