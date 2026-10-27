// ling3-dot-parity — is LG's ROCmFP4 matvec the SAME ARITHMETIC as ggml's?
//
// ggml's CPU mul_mat does not decode a ROCmFP4 row and multiply in float. It
// quantizes the ACTIVATION to Q8_0 and runs an integer dot
// (`vec_dot_type = GGML_TYPE_Q8_0`, `rocmfp4_vec_dot_q4_0_q8_0`). That is a
// different computation, and over 42 blocks of a recurrent model the difference
// grows until argmax flips -- which is exactly what gate (a) measured.
//
// This compares LG's `cpu::DotRow` against the fork's own kernel, dlopen'd
// from libggml-base.so, with `memcmp` per float. Any mismatch here is a real
// mismatch in the arithmetic, not a tolerance question.
//
// Usage: ling3-dot-parity <libggml-base.so> [trials]

#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/ling3/cpu_ops.hpp"
#include "src/models/ling3/rocmfp4.hpp"
#include "src/models/ling3/weights.hpp"

namespace {

std::uint64_t rng = 0x5A17C0DEULL;
std::uint32_t Next() {
  rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<std::uint32_t>(rng >> 32);
}
float Nf(float lo, float hi) {
  return lo + (hi - lo) * static_cast<float>(Next()) / 4294967296.0F;
}

/// ggml's block_q8_0: half d then 32 int8.
struct BlockQ8 {
  std::uint16_t d;
  std::int8_t qs[32];
};
struct BlockFp4 { std::uint8_t qs[16], e[2]; };
struct BlockFp4Fast { std::uint8_t qs[16], e; };

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <libggml-base.so> [trials]\n", argv[0]);
    return 2;
  }
  void* h = dlopen(argv[1], RTLD_NOW);
  if (!h) {
    std::fprintf(stderr, "dlopen: %s\n", dlerror());
    return 1;
  }
  auto dual = reinterpret_cast<void (*)(int, float*, std::size_t, const void*,
                                        std::size_t, const void*, std::size_t,
                                        int)>(
      dlsym(h, "rocmfp4_vec_dot_q4_0_q8_0"));
  auto fast = reinterpret_cast<void (*)(int, float*, std::size_t, const void*,
                                        std::size_t, const void*, std::size_t,
                                        int)>(
      dlsym(h, "rocmfp4_vec_dot_q4_0_fast_q8_0"));
  if (!dual || !fast) {
    std::fprintf(stderr, "fork vec_dot symbols not exported: dual=%p fast=%p\n",
                 reinterpret_cast<void*>(dual), reinterpret_cast<void*>(fast));
    return 1;
  }

  // Independently verify cpu::FloatToFp16 against the fork's own converter, if
  // it is exported. Without this the Q8_0 half scale is only self-consistent,
  // which is exactly the failure mode this test already had once.
  int fp16_bad = 0, fp16_n = 0;
  if (auto f16row = reinterpret_cast<void (*)(const float*, std::uint16_t*,
                                             std::int64_t)>(
          dlsym(h, "ggml_fp32_to_fp16_row"))) {
    std::vector<float> probe(4096);
    std::vector<std::uint16_t> theirs2(4096);
    for (int trial = 0; trial < 64; ++trial) {
      for (auto& v : probe) {
        // Spans the half subnormal, normal and overflow ranges.
        v = Nf(-1.0F, 1.0F) * std::pow(10.0F, static_cast<float>(Next() % 12 - 8));
      }
      f16row(probe.data(), theirs2.data(), 4096);
      for (int i = 0; i < 4096; ++i) {
        ++fp16_n;
        if (superroc::models::ling3::cpu::FloatToFp16(probe[i]) != theirs2[i]) {
          if (fp16_bad < 3) {
            std::printf("  fp16: lg=%04x fork=%04x for %.9g\n",
                        superroc::models::ling3::cpu::FloatToFp16(probe[i]),
                        theirs2[i], static_cast<double>(probe[i]));
          }
          ++fp16_bad;
        }
      }
    }
    std::printf("%s: FloatToFp16 vs fork ggml_fp32_to_fp16_row: %d values, "
                "%d mismatches\n",
                fp16_bad == 0 ? "PASS" : "FAIL", fp16_n, fp16_bad);
  } else {
    std::printf("SKIP: fork ggml_fp32_to_fp16_row not exported; "
                "FloatToFp16 is unverified against the fork\n");
  }

  const int trials = argc > 2 ? std::atoi(argv[2]) : 200;
  const std::size_t k = 2560;  // n_embd, the shape that matters here
  int bad = 0;
  std::size_t worst_ulp = 0;

  for (int trial = 0; trial < trials; ++trial) {
    const bool use_fast = (trial % 2) != 0;
    const std::size_t nb = k / 32;
    const std::size_t rb = use_fast ? 17 : 18;

    std::vector<std::uint8_t> W(nb * rb);
    for (auto& b : W) b = static_cast<std::uint8_t>(Next() & 0xff);
    if (use_fast) {
      for (std::size_t b = 0; b < nb; ++b) W[b * rb + 16] = Next() & 0x7f;
    } else {
      for (std::size_t b = 0; b < nb; ++b) {
        W[b * rb + 16] = Next() & 0x7f;
        W[b * rb + 17] = Next() & 0x7f;
      }
    }

    std::vector<float> x(k);
    for (auto& v : x) v = Nf(-1.0F, 1.0F);

    // Quantize the activation the way ggml's mul_mat does, so both sides see
    // the same Q8_0 block.
    std::vector<BlockQ8> y(nb);
    {
      // quantize_row_q8_0_ref, verbatim.
      for (std::size_t i = 0; i < nb; ++i) {
        float amax = 0.0F;
        for (std::size_t j = 0; j < 32; ++j)
          amax = std::fmax(amax, std::fabs(x[i * 32 + j]));
        const float d = amax / ((1 << 7) - 1);
        // ggml's CPU mul_mat calls the SIMD `quantize_row_q8_0`, not the _ref:
        // `id` is ONE division 127/amax (not 1/(amax/127)) and the rounding is
        // ties-to-EVEN (_mm256_round_ps _MM_ROUND_NEAREST), not roundf's
        // ties-away-from-zero. Both sides must be fed the same blocks or this
        // test would report a mismatch that is really in the quantizer.
        const float id = amax != 0.0F ? 127.0F / amax : 0.0F;
        y[i].d = superroc::models::ling3::cpu::FloatToFp16(d);
        for (std::size_t j = 0; j < 32; ++j) {
          y[i].qs[j] = static_cast<std::int8_t>(std::nearbyintf(x[i * 32 + j] * id));
        }
      }
    }

    float theirs = 0.0F;
    (use_fast ? fast : dual)(static_cast<int>(k), &theirs, 0, W.data(), 0,
                             y.data(), 0, 1);

    // LG's DotRow on the same packed row.
    superroc::models::ling3::TensorRef tr;
    tr.data = W.data();
    tr.type = use_fast ? superroc::core::GgmlType::kQ4_0_ROCMFP4_FAST
                     : superroc::core::GgmlType::kQ4_0_ROCMFP4;
    tr.cols = k;
    tr.rows = 1;
    tr.experts = 1;
    const float mine =
        superroc::models::ling3::cpu::DotRow(tr, 0, 0, std::span<const float>(x));

    if (std::memcmp(&mine, &theirs, sizeof(float)) != 0) {
      ++bad;
      std::uint32_t a, c;
      std::memcpy(&a, &mine, 4);
      std::memcpy(&c, &theirs, 4);
      const std::uint32_t ulp =
          (a > c ? a - c : c - a);
      worst_ulp = ulp > worst_ulp ? ulp : worst_ulp;
      if (bad <= 3) {
        std::printf("  trial %d (%s): lg=%.9g (%08x) fork=%.9g (%08x)\n", trial,
                    use_fast ? "FAST" : "dual", static_cast<double>(mine), a,
                    static_cast<double>(theirs), c);
      }
    }
  }

  std::printf("%s: %d trials, %d mismatches, worst ULP %zu\n",
              bad == 0 ? "PASS" : "FAIL", trials, bad, worst_ulp);
  return bad == 0 ? 0 : 1;
}