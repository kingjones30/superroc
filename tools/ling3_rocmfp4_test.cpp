// ling3-rocmfp4-test — verifies superroc's ROCmFP4 decoder against the fork's.
//
// The fork exports rocmfp4_dequantize_row_q4_0 and _fast from libggml-base.so.
// This test feeds both decoders the same deterministic pseudo-random blocks and
// compares the results *bit for bit*, so a rounding or table difference cannot
// hide.
//
// Usage: ling3-rocmfp4-test <path-to-libggml-base.so> [trials]

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "src/models/ling3/rocmfp4.hpp"

namespace {

struct BlockDual {  // ggml/rocmfp4/rocmfp4.h block_rocmfp4
  std::uint8_t qs[16];
  std::uint8_t e[2];
};
struct BlockFast {  // ggml/rocmfp4/rocmfp4.h block_rocmfp4_fast
  std::uint8_t qs[16];
  std::uint8_t e;
};

using DequantDual = void (*)(const BlockDual*, float*, std::int64_t);
using DequantFast = void (*)(const BlockFast*, float*, std::int64_t);

/// Deterministic 64-bit LCG. A fixed seed keeps this a regression test rather
/// than a flaky one; the point is coverage of the scale and codebook tables,
/// not randomness per se.
std::uint64_t Next(std::uint64_t* state) {
  *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
  return *state >> 33;
}

int Compare(const char* label, const void* blocks, std::size_t n_blocks,
            void (*mine)(superroc::core::GgmlType, const void*, float*, std::size_t),
            superroc::core::GgmlType type, void (*theirs)(const void*, float*, std::int64_t),
            const std::uint8_t* src) {
  std::vector<float> a(n_blocks * 32, -12345.0F);
  std::vector<float> b(n_blocks * 32, 54321.0F);
  mine(type, src, a.data(), n_blocks * 32);
  theirs(blocks, b.data(), static_cast<std::int64_t>(n_blocks * 32));

  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
      if (mismatches < 5) {
        std::printf("  %s[%zu]: mine=%.9g (%08x) fork=%.9g (%08x)\n", label, i,
                    static_cast<double>(a[i]), *reinterpret_cast<const unsigned*>(&a[i]),
                    static_cast<double>(b[i]), *reinterpret_cast<const unsigned*>(&b[i]));
      }
      ++mismatches;
    }
  }
  std::printf("%-6s %6zu blocks: %zu mismatches\n", label, n_blocks, mismatches);
  return mismatches == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <libggml-base.so> [trials]\n", argv[0]);
    return 2;
  }
  void* handle = dlopen(argv[1], RTLD_NOW);
  if (handle == nullptr) {
    std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }
  auto dual =
      reinterpret_cast<DequantDual>(dlsym(handle, "rocmfp4_dequantize_row_q4_0"));
  auto fast = reinterpret_cast<DequantFast>(
      dlsym(handle, "rocmfp4_dequantize_row_q4_0_fast"));
  if (dual == nullptr || fast == nullptr) {
    std::fprintf(stderr,
                 "fork ROCmFP4 decoders not exported: dual=%p fast=%p\n",
                 reinterpret_cast<void*>(dual), reinterpret_cast<void*>(fast));
    return 1;
  }

  const std::size_t trials = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 200;
  std::uint64_t rng = 0x9E3779B97F4A7C15ULL;
  int failures = 0;

  for (std::size_t trial = 0; trial < trials; ++trial) {
    const std::size_t n_blocks = 1 + (Next(&rng) % 64);

    // Dense coverage of the codebook (all 16 nibbles) and the scale table
    // (every byte, including the reserved 0x7f that must decode to zero).
    std::vector<BlockDual> d(n_blocks);
    std::vector<BlockFast> f(n_blocks);
    for (std::size_t b = 0; b < n_blocks; ++b) {
      for (int j = 0; j < 16; ++j) {
        const std::uint8_t byte = static_cast<std::uint8_t>(Next(&rng));
        d[b].qs[j] = byte;
        f[b].qs[j] = byte;
      }
      if (trial == 0) {
        // Exercise every legal scale byte once, plus the reserved 0x7f.
        d[b].e[0] = static_cast<std::uint8_t>((b * 2) & 0x7f);
        d[b].e[1] = static_cast<std::uint8_t>((b * 2 + 1) & 0x7f);
        f[b].e = static_cast<std::uint8_t>(b & 0x7f);
      } else {
        d[b].e[0] = static_cast<std::uint8_t>(Next(&rng) & 0x7f);
        d[b].e[1] = static_cast<std::uint8_t>(Next(&rng) & 0x7f);
        f[b].e = static_cast<std::uint8_t>(Next(&rng) & 0x7f);
      }
    }

    failures += Compare("dual", d.data(), n_blocks,
                        superroc::models::ling3::rocmfp4::DequantizeRow,
                        superroc::core::GgmlType::kQ4_0_ROCMFP4,
                        reinterpret_cast<void (*)(const void*, float*, std::int64_t)>(dual),
                        reinterpret_cast<const std::uint8_t*>(d.data()));
    failures += Compare("fast", f.data(), n_blocks,
                        superroc::models::ling3::rocmfp4::DequantizeRow,
                        superroc::core::GgmlType::kQ4_0_ROCMFP4_FAST,
                        reinterpret_cast<void (*)(const void*, float*, std::int64_t)>(fast),
                        reinterpret_cast<const std::uint8_t*>(f.data()));
  }

  // The reserved scale byte must decode to exactly zero, not a denormal.
  if (superroc::models::ling3::rocmfp4::ScaleFromByte(0x7f) != 0.0F) {
    std::printf("ScaleFromByte(0x7f) is %g, expected 0\n",
                static_cast<double>(superroc::models::ling3::rocmfp4::ScaleFromByte(0x7f)));
    ++failures;
  }

  std::printf("%s: %zu trials, %d failures\n",
              failures == 0 ? "PASS" : "FAIL", trials, failures);
  return failures == 0 ? 0 : 1;
}
