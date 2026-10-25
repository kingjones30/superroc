#include "src/models/ling3/rocmfp4.hpp"

#include <cstring>

namespace gufo::models::ling3::rocmfp4 {
namespace {

using core::GgmlType;

// The fork keeps a 127-entry table of scale values. Every entry there is built
// by one of these macros, so the closed form below reproduces the table
// exactly (each value is an integer times a power of two, hence exact in
// float). Index e decodes as:
//
//   e <  8 : e            * 2^-10          (ROCMFP4_SCALE_SUB)
//   e >= 8 : (8 + (e & 7)) * 2^(-11 + e/8) (ROCMFP4_SCALE_E<1 + e/8>)
//
// Bytes above 0x7e have no table entry and decode to 0.
constexpr float ScaleTable(std::size_t e) noexcept {
  return e < 8
             ? static_cast<float>(e) * 0x1p-10F
             : static_cast<float>(8 + (e & 7)) *
                   (0x1p-10F * static_cast<float>(1u << (e >= 8 ? e / 8 : 0)));
}

// Unused-parameter-free form of the exponent, written without the shift above
// so the intent stays literal with the fork's macros.
constexpr float ScaleValue(std::size_t e) noexcept {
  if (e < 8) {
    return static_cast<float>(e) * 0x1p-10F;
  }
  const int exponent = static_cast<int>(e / 8);  // 1..15
  const float mantissa = static_cast<float>(8 + (e & 7));
  // ROCMFP4_SCALE_E<k>(M) is (8 + M) * 2^(k - 11), so the power starts at
  // 2^-11 and is doubled once per exponent step.
  float scale = 0x1p-11F;
  for (int i = 0; i < exponent; ++i) {
    scale *= 2.0F;
  }
  return mantissa * scale;
}

static_assert(ScaleValue(0) == 0.0F);
static_assert(ScaleValue(1) == 0x1p-10F);
static_assert(ScaleValue(7) == 7 * 0x1p-10F);
// ROCMFP4_SCALE_E1(0) == (8 + 0) * 2^-10
static_assert(ScaleValue(8) == 8 * 0x1p-10F);
// ROCMFP4_SCALE_E1(7) == (8 + 7) * 2^-10
static_assert(ScaleValue(15) == 15 * 0x1p-10F);
// ROCMFP4_SCALE_E15(6) == (8 + 6) * 2^4 == 224
static_assert(ScaleValue(126) == 224.0F);

// E2M1-derived codebook with the top magnitude retuned 12 -> 10. Index 8 is the
// zero code, not a sign bit; nibbles 9..15 are the negatives of 1..7.
constexpr std::int8_t kCodebook[16] = {0, 1,  2,  3,  4,  6,  8,  10,
                                       0, -1, -2, -3, -4, -6, -8, -10};

}  // namespace

const std::int8_t* Codebook() noexcept {
  return kCodebook;
}

float ScaleFromByte(std::uint8_t e) noexcept {
  return e <= 0x7e ? ScaleValue(e) : 0.0F;
}

void DequantizeRow(GgmlType type, const void* src, float* dst, std::size_t k) {
  const std::uint8_t* bytes = static_cast<const std::uint8_t*>(src);
  const std::size_t blocks = k / kBlockElements;

  if (type == GgmlType::kQ4_0_ROCMFP4_FAST) {
    for (std::size_t b = 0; b < blocks; ++b) {
      const std::uint8_t* blk = bytes + b * kBytesFastScale;
      const float d = ScaleFromByte(blk[16]);
      float* y = dst + b * kBlockElements;
      for (std::size_t j = 0; j < 16; ++j) {
        y[j] = static_cast<float>(kCodebook[blk[j] & 0x0F]) * d;
        y[j + 16] = static_cast<float>(kCodebook[blk[j] >> 4]) * d;
      }
    }
    return;
  }

  for (std::size_t b = 0; b < blocks; ++b) {
    const std::uint8_t* blk = bytes + b * kBytesDualScale;
    const float d0 = ScaleFromByte(blk[16]);
    const float d1 = ScaleFromByte(blk[17]);
    float* y = dst + b * kBlockElements;
    for (std::size_t j = 0; j < 16; ++j) {
      y[j] = static_cast<float>(kCodebook[blk[j] & 0x0F]) * d0;
      y[j + 16] = static_cast<float>(kCodebook[blk[j] >> 4]) * d1;
    }
  }
}

}  // namespace gufo::models::ling3::rocmfp4
