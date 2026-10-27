#include "src/models/ling3/cpu_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/ling3/rocmfp4.hpp"

namespace superroc::models::ling3::cpu {

using core::GgmlType;

/// float -> IEEE half, round-to-nearest-even. ggml stores the Q8_0 block scale
/// as a half and converts it back before every use, so this round trip is part
/// of the arithmetic and has to be reproduced exactly.
///
/// The exponent must be REBASED from the float32 bias (127) to the half bias
/// (15) -- i.e. 112 << 10 -- before the mantissa is shifted down. Doing the
/// shift without the rebase yields a half whose value is off by a factor of
/// 112 * 2^10, which is enough to make the whole model diverge.
std::uint16_t FloatToFp16(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  const std::uint32_t mag = x & 0x7fffffffu;

  if (mag >= 0x7f800000u) {  // inf / NaN
    return static_cast<std::uint16_t>(sign | 0x7c00u |
                                      (mag > 0x7f800000u ? 0x200u : 0u));
  }
  if (mag >= 0x47800000u) {  // >= 2^15, overflows half -> inf
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  if (mag >= 0x38800000u) {  // >= 2^-14, normal half
    // Rebased exponent and top 10 mantissa bits; rounding then adds to this
    // combined value so a mantissa carry propagates into the exponent.
    std::uint32_t val = (((mag >> 23) - 112u) << 10) | ((mag >> 13) & 0x3ffu);
    const std::uint32_t rem = mag & 0x1fffu;  // discarded bits
    if (rem > 0x1000u || (rem == 0x1000u && (val & 1u) != 0u)) {
      ++val;
    }
    return static_cast<std::uint16_t>(sign | val);
  }
  if (mag < 0x33800000u) {  // < 2^-24, rounds to zero (ties to even)
    return static_cast<std::uint16_t>(sign);
  }
  // subnormal half (2^-24 <= |f| < 2^-14): shift the mantissa up, then round.
  // The implicit leading 1 has to be made explicit -- dropping it is worth a
  // factor of ~3.7 on the scale.
  const std::uint32_t m = (mag & 0x7fffffu) | 0x800000u;
  const std::uint32_t shift = 126u - (mag >> 23);  // in [14, 23] here
  std::uint32_t h = m >> shift;
  const std::uint32_t rem = m & ((1u << shift) - 1u);
  const std::uint32_t half = 1u << (shift - 1);
  if (rem > half || (rem == half && (h & 1u) != 0u)) {
    ++h;
  }
  return static_cast<std::uint16_t>(sign | h);
}

namespace {

/// ggml's `quantize_row_q8_0` -- the SIMD one that CPU `mul_mat` actually
/// calls, NOT `quantize_row_q8_0_ref`. Two differences from the ref version
/// matter:
///
/// 1. the inverse scale is ONE division `id = 127.f / amax`, where the ref
///    computes `id = 1.0f / (amax / 127.0f)`. Those disagree by up to 1 ULP,
///    and the stored scale `d` is `amax / 127.f` in BOTH.
/// 2. rounding is `_mm256_round_ps(..., _MM_ROUND_NEAREST)` -- ties to EVEN --
///    followed by `_mm256_cvtps_epi32`. `roundf()` is ties AWAY FROM ZERO.
///    At an exact .5 the two pick different integers.
///
/// Both feed a sub-print-precision difference into every matmul, which is the
/// kind of seed that compounds over 42 blocks of a recurrent model.
struct Q8Block {
  float d;
  std::int8_t qs[32];
};

void QuantizeRowQ8_0(std::span<const float> x, std::vector<Q8Block>* out) {
  const std::size_t nb = x.size() / 32;
  out->assign(nb, Q8Block{});
  for (std::size_t i = 0; i < nb; ++i) {
    float amax = 0.0F;
    for (std::size_t j = 0; j < 32; ++j) {
      amax = std::max(amax, std::fabs(x[i * 32 + j]));
    }
    const float d = amax / 127.0F;
    const float id = amax != 0.0F ? 127.0F / amax : 0.0F;
    (*out)[i].d = quant::Fp16ToFloat(FloatToFp16(d));
    for (std::size_t j = 0; j < 32; ++j) {
      // nearbyintf, not roundf: ties to even under the default FE_TONEAREST,
      // matching _mm256_round_ps(_MM_ROUND_NEAREST).
      const float q = std::nearbyintf(x[i * 32 + j] * id);
      (*out)[i].qs[j] = static_cast<std::int8_t>(q);
    }
  }
}

/// ggml's `rocmfp4_vec_dot_q4_0_q8_0` / `..._fast_q8_0`, scalar form. The AVX2
/// variant sums the same integers in the same order, so this is bit-identical.
float RocmFp4DotQ8_0(int type, const std::uint8_t* row,
                     std::span<const Q8Block> y) {
  const bool fast =
      type == static_cast<int>(core::GgmlType::kQ4_0_ROCMFP4_FAST);
  float sumf = 0.0F;
  for (std::size_t ib = 0; ib < y.size(); ++ib) {
    const std::uint8_t* blk = row + ib * (fast ? 17u : 18u);
    int sumi0 = 0;
    int sumi1 = 0;
    for (int j = 0; j < 16; ++j) {
      const std::uint8_t q = blk[j];
      sumi0 += rocmfp4::Codebook()[q & 0x0f] * y[ib].qs[j];
      sumi1 += rocmfp4::Codebook()[q >> 4] * y[ib].qs[j + 16];
    }
    // The two ROCmFP4 types do NOT combine the same way, and getting this
    // wrong is worth up to ~24 ULP per block, which is enough to flip argmax
    // 42 blocks later:
    //   dual-scale (100): sumf += d0 * sumi0 + d1 * sumi1   -- two multiplies
    //   fast (101):       sumf += d * (sumi0 + sumi1)      -- one multiply,
    //                                                       integers first
    if (fast) {
      const float d = rocmfp4::ScaleFromByte(blk[16]) * y[ib].d;
      sumf += d * static_cast<float>(sumi0 + sumi1);
    } else {
      const float d0 = rocmfp4::ScaleFromByte(blk[16]) * y[ib].d;
      const float d1 = rocmfp4::ScaleFromByte(blk[17]) * y[ib].d;
      sumf += d0 * static_cast<float>(sumi0) + d1 * static_cast<float>(sumi1);
    }
  }
  return sumf;
}

/// Selects the ROCmFP4 matmul arithmetic. See `DotRow`.
enum DotMode { kDotQ8, kDotExact, kDotBoth };

DotMode DotMode() {
  const char* s = std::getenv("LING3_DOT");
  if (s == nullptr || std::strcmp(s, "q8") == 0)
    return kDotQ8;
  if (std::strcmp(s, "exact") == 0)
    return kDotExact;
  return kDotBoth;
}

/// The exact value of the packed product: decode the row, multiply in float.
/// This is what ggml does for every type EXCEPT ROCmFP4.
float ExactDot(const TensorRef& t, const std::uint8_t* src,
               std::span<const float> x) {
  const std::size_t k = std::min(static_cast<std::size_t>(t.cols), x.size());
  std::vector<float> w(k);
  rocmfp4::DequantizeRow(t.type, src, w.data(), k);
  float sum = 0.0F;
  for (std::size_t i = 0; i < k; ++i) {
    sum += w[i] * x[i];
  }
  return sum;
}

/// First few large disagreements per thread. MatVec is OpenMP-parallel over
/// rows, so each thread reports for itself; grepping the TOTALS lines sums
/// them.
void ReportDivergence(const TensorRef& t, std::uint64_t row, std::size_t nx,
                      float exacte, float dotq) {
  static thread_local int shown = 0;
  ++shown;
  if (shown > 8)
    return;
  std::fprintf(stderr,
               "DIVERGE dot: type=%d row=%llu cols=%llu x.size()=%zu "
               "exact=%.6g q8=%.6g\n",
               static_cast<int>(t.type), static_cast<unsigned long long>(row),
               static_cast<unsigned long long>(t.cols), nx, exacte, dotq);
}

/// LING3_DOT_DUMP=<n>: on the first divergence, dump block 0 of the weight row,
/// the activation's first 32 values, and the Q8 block ggml would have built.
/// This is the decisive measurement: the parity tool only ever fed RANDOM
/// packed rows, so it proved the arithmetic and not the operands.
void DumpFirstDivergence(const TensorRef& t, const std::uint8_t* src,
                         std::span<const float> x,
                         const std::vector<Q8Block>& q, float exacte,
                         float dotq) {
  static thread_local bool done = false;
  if (done)
    return;
  done = true;
  const bool fast = t.type == core::GgmlType::kQ4_0_ROCMFP4_FAST;
  const std::size_t nblk = t.cols / 32;
  std::fprintf(stderr, "DUMP type=%d fast=%d nblk=%zu rowbytes=%llu\n",
               static_cast<int>(t.type), fast ? 1 : 0, nblk,
               static_cast<unsigned long long>(t.RowBytes()));
  std::fprintf(stderr, "DUMP exact=%.9g q8=%.9g\n", static_cast<double>(exacte),
               static_cast<double>(dotq));
  const std::uint8_t* blk = src;
  std::fprintf(stderr, "DUMP blk0 bytes:");
  for (int i = 0; i < 18; ++i)
    std::fprintf(stderr, " %02x", blk[i]);
  std::fprintf(stderr, "\n");
  std::fprintf(stderr, "DUMP blk1 bytes:");
  for (int i = 0; i < 18; ++i)
    std::fprintf(stderr, " %02x", blk[(fast ? 17u : 18u) + i]);
  std::fprintf(stderr, "\n");
  std::fprintf(stderr, "DUMP x[0..32):");
  for (std::size_t i = 0; i < 32 && i < x.size(); ++i) {
    std::fprintf(stderr, " %.6g", static_cast<double>(x[i]));
  }
  std::fprintf(stderr, "\n");
  float amax = 0.0F;
  for (std::size_t i = 0; i < 32 && i < x.size(); ++i)
    amax = std::fmax(amax, std::fabs(x[i]));
  std::fprintf(stderr, "DUMP block0 amax=%.9g q.d=%.9g q.qs:",
               static_cast<double>(amax), static_cast<double>(q[0].d));
  for (std::size_t i = 0; i < 32; ++i)
    std::fprintf(stderr, " %d", static_cast<int>(q[0].qs[i]));
  std::fprintf(stderr, "\n");
  std::fprintf(stderr, "DUMP codebook:");
  for (int i = 0; i < 16; ++i)
    std::fprintf(stderr, " %d", rocmfp4::Codebook()[i]);
  std::fprintf(stderr, "\n");
  std::fprintf(stderr, "DUMP e0=%.9g e1=%.9g  d0*y.d=%.9g d1*y.d=%.9g\n",
               static_cast<double>(rocmfp4::ScaleFromByte(blk[16])),
               static_cast<double>(rocmfp4::ScaleFromByte(blk[17])),
               static_cast<double>(rocmfp4::ScaleFromByte(blk[16]) * q[0].d),
               static_cast<double>(rocmfp4::ScaleFromByte(blk[17]) * q[0].d));
}

}  // namespace

void DequantizeRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
                   float* out) {
  const std::size_t k = static_cast<std::size_t>(t.cols);
  const std::uint8_t* src = t.Expert(e) + row * t.RowBytes();
  switch (t.type) {
    case GgmlType::kF32:
      std::memcpy(out, src, k * sizeof(float));
      break;
    case GgmlType::kF16:
    case GgmlType::kBF16: {
      const bool bf16 = t.type == GgmlType::kBF16;
      for (std::size_t i = 0; i < k; ++i) {
        std::uint16_t h = 0;
        std::memcpy(&h, src + 2 * i, 2);
        if (bf16) {
          const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
          std::memcpy(out + i, &bits, 4);
        } else {
          out[i] = quant::Fp16ToFloat(h);
        }
      }
      break;
    }
    case GgmlType::kQ4_0_ROCMFP4:
    case GgmlType::kQ4_0_ROCMFP4_FAST:
      rocmfp4::DequantizeRow(t.type, src, out, k);
      break;
    case GgmlType::kQ8_0:
      quant::DequantizeQ8_0(src, out, k);
      break;
    case GgmlType::kQ5_K:
      quant::DequantizeQ5_K(src, out, k);
      break;
    case GgmlType::kQ6_K:
      quant::DequantizeQ6_K(src, out, k);
      break;
    default:
      break;
  }
}

float DotRow(const TensorRef& t, std::uint64_t e, std::uint64_t row,
             std::span<const float> x) {
  const std::size_t k = static_cast<std::size_t>(t.cols);
  const std::uint8_t* src = t.Expert(e) + row * t.RowBytes();
  switch (t.type) {
    case GgmlType::kF32: {
      const auto* w = reinterpret_cast<const float*>(src);
      float sum = 0.0F;
      for (std::size_t i = 0; i < k; ++i) {
        sum += w[i] * x[i];
      }
      return sum;
    }
    case GgmlType::kF16:
    case GgmlType::kBF16: {
      const bool bf16 = t.type == GgmlType::kBF16;
      float sum = 0.0F;
      for (std::size_t i = 0; i < k; ++i) {
        std::uint16_t h = 0;
        std::memcpy(&h, src + 2 * i, 2);
        float w;
        if (bf16) {
          const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
          std::memcpy(&w, &bits, 4);
        } else {
          w = quant::Fp16ToFloat(h);
        }
        sum += w * x[i];
      }
      return sum;
    }
    case GgmlType::kQ4_0_ROCMFP4:
    case GgmlType::kQ4_0_ROCMFP4_FAST: {
      // ggml's CPU mul_mat does NOT decode-and-multiply here: its
      // type_traits_cpu entry sets vec_dot_type = GGML_TYPE_Q8_0, so the
      // ACTIVATION is quantized to Q8_0 and an integer dot is accumulated.
      // `tools/ling3_dot_parity` checks this against the fork's own
      // rocmfp4_vec_dot_q4_0_{,fast_}q8_0 with memcmp per float.
      //
      // Matching it matters: summing the exact decoded floats instead is a
      // different computation, and 42 blocks of a recurrent model amplify the
      // difference until argmax flips. Measured with exact-float decoding,
      // teacher-forced top-1 agreement against the fork was 79.8 %.
      //
      // LING3_DOT=q8 (default) is ggml's arithmetic; LING3_DOT=exact is the
      // exact decoded product; LING3_DOT=both runs both and reports the first
      // call sites where they disagree by more than 1 % relative.
      static thread_local std::vector<Q8Block> q;
      QuantizeRowQ8_0(x, &q);
      const float dotq =
          RocmFp4DotQ8_0(static_cast<int>(t.type), src,
                         std::span<const Q8Block>(q.data(), q.size()));
      switch (DotMode()) {
        case kDotExact:
          return ExactDot(t, src, x);
        case kDotQ8:
          return dotq;
        case kDotBoth: {
          const float exacte = ExactDot(t, src, x);
          const float mag = std::fmax(std::fabs(exacte), 1e-30F);
          if (std::fabs(dotq - exacte) > 1e-3F * mag) {
            ReportDivergence(t, row, x.size(), exacte, dotq);
            if (std::getenv("LING3_DOT_DUMP") != nullptr) {
              DumpFirstDivergence(t, src, x, q, exacte, dotq);
            }
          }
          return dotq;
        }
      }
      return dotq;
    }
    case GgmlType::kQ8_0:
      return quant::DotProductQ8_0(src, x, k);
    case GgmlType::kQ5_K:
      return quant::DotProductQ5_K(src, x, k);
    case GgmlType::kQ6_K:
      return quant::DotProductQ6_K(src, x, k);
    default:
      return 0.0F;
  }
}

void MatVec(const TensorRef& t, std::uint64_t e, std::span<const float> x,
            std::span<float> y) {
  // Parallelism is over output rows only. Each row keeps its own sequential
  // accumulation, so the result is identical to the serial loop; this is a
  // scheduling change, not an arithmetic one.
  const std::int64_t rows = static_cast<std::int64_t>(t.rows);
  float* out = y.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::int64_t r = 0; r < rows; ++r) {
    out[r] = DotRow(t, e, static_cast<std::uint64_t>(r), x);
  }
}

void ReadVector(const VectorRef& v, float* out) {
  DequantizeRow(v.tensor, 0, 0, out);
}

void RmsNorm(std::span<float> x, const float* w, float eps) {
  const std::size_t n = x.size();
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += static_cast<double>(x[i]) * static_cast<double>(x[i]);
  }
  const float mean = static_cast<float>(sum / static_cast<double>(n));
  const float scale = 1.0F / std::sqrt(mean + eps);
  if (w == nullptr) {
    for (std::size_t i = 0; i < n; ++i) {
      x[i] *= scale;
    }
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = x[i] * scale * w[i];
  }
}

void L2Norm(std::span<float> x, float eps) {
  // ggml_l2_norm declares `ggml_float sum` (a double) but accumulates
  // `sum += (ggml_float)(xi * xi)` -- the product is rounded to float first and
  // only then widened -- and it narrows to float *before* the sqrt rather than
  // rounding a double root. Both details are worth about 1 ULP; verified
  // bit-exact against the fork in tools/kda_math_test.cpp on the Mac.
  double sum = 0.0;
  for (const float v : x) {
    sum += static_cast<double>(v * v);
  }
  const float norm = std::sqrt(static_cast<float>(sum));
  // 1 / fmaxf(sqrtf(sum), eps): the clamp is outside the root.
  const float scale = 1.0F / std::fmax(norm, eps);
  for (float& v : x) {
    v *= scale;
  }
}

float Sigmoid(float x) noexcept {
  return 1.0F / (1.0F + std::exp(-x));
}

float Silu(float x) noexcept {
  return x / (1.0F + std::exp(-x));
}

void Rope(float* x, std::uint32_t heads, std::uint32_t head_dim,
          std::uint32_t rotary_dim, std::uint32_t pos, float theta) {
  const std::uint32_t pairs = rotary_dim / 2;
  for (std::uint32_t h = 0; h < heads; ++h) {
    float* head = x + static_cast<std::size_t>(h) * head_dim;
    for (std::uint32_t i = 0; i < pairs; ++i) {
      const float freq =
          1.0F / std::pow(theta, 2.0F * static_cast<float>(i) /
                                     static_cast<float>(rotary_dim));
      const float angle = static_cast<float>(pos) * freq;
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      // GGML_ROPE_TYPE_NORM pairs CONSECUTIVE elements (ggml-cuda/rope.cu
      // rope_norm: x0 = x[ix+0], x1 = x[ix+1], i0 stepped by 2). Half-split
      // (i, i + rot/2) is GGML_ROPE_TYPE_NEOX and is what this used to do.
      const float a = head[2 * i];
      const float b = head[2 * i + 1];
      head[2 * i] = a * c - b * s;
      head[2 * i + 1] = a * s + b * c;
    }
  }
}

void SwigluClamped(const float* gate, const float* up, float* out,
                   std::size_t n, float limit) {
  for (std::size_t i = 0; i < n; ++i) {
    const float act = std::fmin(Silu(gate[i]), limit);
    const float u = std::fmin(std::fmax(up[i], -limit), limit);
    out[i] = act * u;
  }
}

}  // namespace superroc::models::ling3::cpu
