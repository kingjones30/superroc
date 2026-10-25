#ifndef GUFO_MODELS_LING3_ROCMFP4_HPP_
#define GUFO_MODELS_LING3_ROCMFP4_HPP_

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

/// ROCmFP4 (GGML type 100 / 101) host decoding.
///
/// Both formats pack 32 four-bit codes per block. The codebook is E2M1-derived
/// with the largest magnitude retuned from 12 to 10; nibble 8 is the zero code,
/// not a sign bit. Scales are unsigned E4M3 bytes indexed through a 127-entry
/// table, with anything above 0x7e decoding to zero.
///
///   type 100  Q4_0_ROCMFP4       18 bytes/block: qs[16] then e[0], e[1].
///                                 e[0] scales elements 0..15, e[1]
///                                 scales 16..31.
///   type 101  Q4_0_ROCMFP4_FAST  17 bytes/block: qs[16] then e[0] for the
///                                 whole block.
///
/// Transcribed from the fork's ggml/rocmfp4/rocmfp4.{h,c}; S2 verified these
/// decoders bit-exact against types 100 and 101 (0 mismatches).
namespace gufo::models::ling3::rocmfp4 {

/// Logical elements in one ROCmFP4 block.
inline constexpr std::size_t kBlockElements = 32;
/// Encoded bytes of one block, per GGML type.
inline constexpr std::size_t kBytesDualScale = 18;  // type 100
inline constexpr std::size_t kBytesFastScale = 17;  // type 101

/// True when `type` is one of the two ROCmFP4 storage formats.
[[nodiscard]] constexpr bool IsRocmFp4(core::GgmlType type) noexcept {
  return type == core::GgmlType::kQ4_0_ROCMFP4 ||
         type == core::GgmlType::kQ4_0_ROCMFP4_FAST;
}

/// Decodes `k` elements (a multiple of 32) from `src` into `dst`.
/// `kBytes` must match the block size of `type`.
void DequantizeRow(core::GgmlType type, const void* src, float* dst,
                   std::size_t k);

/// The four-bit codebook: index -> dequantized multiplier before scaling.
[[nodiscard]] const std::int8_t* Codebook() noexcept;

/// Unsigned E4M3 scale byte -> half-scale. Bytes above 0x7e decode to 0.
[[nodiscard]] float ScaleFromByte(std::uint8_t e) noexcept;

}  // namespace gufo::models::ling3::rocmfp4

#endif  // GUFO_MODELS_LING3_ROCMFP4_HPP_
