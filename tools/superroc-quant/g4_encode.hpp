// superroc-quant G4 types 120/121 — "split scale plane" layout (RV20-DESIGN.md §3.1).
//
// The encode math is IDENTICAL to the ROCmFP4 types (same UE4M3 exhaustive
// scale search, same codebook, same nibble packing); only the *addressing*
// changes. For a row of `nb` 32-element blocks:
//
//   type 100:  block k = [ qs[16] , e[2] ]   (dual scale, 18 B/block)
//   type 120:  all qs planes first (nb*16 B, 16-byte aligned), then the scale
//              plane (nb*2 B). Total bytes == 100.
//   type 101:  block k = [ qs[16] , e[1] ]   (single scale, 17 B/block)
//   type 121:  all qs planes first (nb*16 B), then scale plane (nb*1 B).
//
// So a G4 twin decodes to bit-identical f32 as its 100/101 counterpart; only
// addresses move. Codes/scales are carried byte-for-byte from the same
// per-block computation.
//
// Type 120/121 add a split per row: the n-block row stores the concatenation
// of every block's qs plane, then every block's scale byte(s).

#ifndef SUPERROC_QUANT_G4_ENCODE_HPP_
#define SUPERROC_QUANT_G4_ENCODE_HPP_

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "gguf.hpp"
#include "rocmfp4_encode.hpp"

namespace superroc::quant_tool {

// Decode helpers (shared with the classifier): given the 32 raw decoded
// nibbles for a block (in ROCmFP4 nibble order) and the block scale(s), return
// the 32 f32 values. `qs[16]`: nibble j = low of qs[j] (elem 0..15), high of
// qs[j] (elem 16..31). Same decode table as rocmfp4_decode.
inline float RocmFp4DecodeNibble(std::uint8_t q) {
  const std::uint8_t mag = (q & 7) <= 4 ? (q & 7) : 2 * (q & 7) - 4;
  const bool neg = (q & 8) != 0;
  float m = (float)mag;
  return neg ? -m : m;
}

// Decode a single 32-element block of type 100/120 (dual scale) given:
//   qs  : the 16 packed nibble bytes
//   e0  : scale for elems 0..15
//   e1  : scale for elems 16..31
inline void DecodeBlockDual(const std::uint8_t* qs, std::uint8_t e0, std::uint8_t e1,
                            float* out) {
  const float s0 = ScaleFromByte(e0);
  const float s1 = ScaleFromByte(e1);
  for (int j = 0; j < 16; ++j) {
    const std::uint8_t qlo = qs[j] & 0x0F;
    const std::uint8_t qhi = qs[j] >> 4;
    out[j] = RocmFp4DecodeNibble(qlo) * s0;
    out[j + 16] = RocmFp4DecodeNibble(qhi) * s1;
  }
}

// Decode a single 32-element block of type 101/121 (single scale).
inline void DecodeBlockSingle(const std::uint8_t* qs, std::uint8_t e, float* out) {
  const float s = ScaleFromByte(e);
  for (int j = 0; j < 16; ++j) {
    const std::uint8_t qlo = qs[j] & 0x0F;
    const std::uint8_t qhi = qs[j] >> 4;
    out[j] = RocmFp4DecodeNibble(qlo) * s;
    out[j + 16] = RocmFp4DecodeNibble(qhi) * s;
  }
}

// Encode a full row (multiple of 32) to G4 type 120 (split scale plane, dual).
// `k` must be divisible by 32. Bytes are written in the split layout:
//   [ qs-plane (nb*16 bytes) ][ scale-plane (nb*2 bytes) ]
inline void EncodeRowG4Dual(const float* x, std::uint8_t* out, std::int64_t k) {
  const std::int64_t nb = k / kQkRocmFp4;
  // qs planes first (16 B per block, packed contiguously)
  std::uint8_t* qs = out;
  // scale plane
  std::uint8_t* scale = out + nb * 16;
  for (std::int64_t ib = 0; ib < nb; ++ib) {
    const float* xb = x + ib * 32;
    float max_abs0 = 0, max_abs1 = 0;
    for (int j = 0; j < 16; ++j) { float ax = xb[j] < 0 ? -xb[j] : xb[j]; if (ax > max_abs0) max_abs0 = ax; }
    for (int j = 16; j < 32; ++j) { float ax = xb[j] < 0 ? -xb[j] : xb[j]; if (ax > max_abs1) max_abs1 = ax; }
    const std::uint8_t e0 = ChooseScale(xb, 16, max_abs0);
    const std::uint8_t e1 = ChooseScale(xb + 16, 16, max_abs1);
    const float inv0 = e0 ? 1.0f / ScaleFromByte(e0) : 0.0f;
    const float inv1 = e1 ? 1.0f / ScaleFromByte(e1) : 0.0f;
    std::uint8_t qb[16];
    for (int j = 0; j < 16; ++j) {
      const std::uint8_t q0 = BestIndexScaled(xb[j], inv0);
      const std::uint8_t q1 = BestIndexScaled(xb[j + 16], inv1);
      qb[j] = (std::uint8_t)(q0 | (q1 << 4));
    }
    // copy qs plane into place
    std::memcpy(qs + ib * 16, qb, 16);
    scale[ib * 2 + 0] = e0;
    scale[ib * 2 + 1] = e1;
  }
}

// Encode a full row to G4 type 121 (single scale): [ qs-plane ][ scale nb*1 ].
inline void EncodeRowG4Fast(const float* x, std::uint8_t* out, std::int64_t k) {
  const std::int64_t nb = k / kQkRocmFp4;
  std::uint8_t* qs = out;
  std::uint8_t* scale = out + nb * 16;
  for (std::int64_t ib = 0; ib < nb; ++ib) {
    const float* xb = x + ib * 32;
    float max_abs = 0;
    for (int j = 0; j < 32; ++j) { float ax = xb[j] < 0 ? -xb[j] : xb[j]; if (ax > max_abs) max_abs = ax; }
    const std::uint8_t e = ChooseScale(xb, 32, max_abs);
    const float inv = e ? 1.0f / ScaleFromByte(e) : 0.0f;
    std::uint8_t qb[16];
    for (int j = 0; j < 16; ++j) {
      const std::uint8_t q0 = BestIndexScaled(xb[j], inv);
      const std::uint8_t q1 = BestIndexScaled(xb[j + 16], inv);
      qb[j] = (std::uint8_t)(q0 | (q1 << 4));
    }
    std::memcpy(qs + ib * 16, qb, 16);
    scale[ib] = e;
  }
}

// Decode a full row of G4 type 120/121 back to f32.
// `out` must hold k floats. `type` must be 120 or 121.
inline void DecodeRowG4(const std::uint8_t* row, Type type, std::int64_t k, float* out) {
  const std::int64_t nb = k / kQkRocmFp4;
  const std::uint8_t* qs = row;
  if (type == (Type)120) {
    const std::uint8_t* scale = row + nb * 16;
    for (std::int64_t ib = 0; ib < nb; ++ib) {
      DecodeBlockDual(qs + ib * 16, scale[ib * 2 + 0], scale[ib * 2 + 1], out + ib * 32);
    }
  } else if (type == (Type)121) {
    const std::uint8_t* scale = row + nb * 16;
    for (std::int64_t ib = 0; ib < nb; ++ib) {
      DecodeBlockSingle(qs + ib * 16, scale[ib], out + ib * 32);
    }
  } else {
    throw std::runtime_error("DecodeRowG4: not a G4 type");
  }
}

// Decode a raw type-100/101 tensor row to f32 (for the twin-differential test).
inline void DecodeRowRocmFp4(const std::uint8_t* row, Type type, std::int64_t k, float* out) {
  const std::int64_t nb = k / kQkRocmFp4;
  if (type == (Type)100) {
    const block_rocmfp4* blk = (const block_rocmfp4*)row;
    for (std::int64_t ib = 0; ib < nb; ++ib)
      DecodeBlockDual(blk[ib].qs, blk[ib].e[0], blk[ib].e[1], out + ib * 32);
  } else if (type == (Type)101) {
    const block_rocmfp4_fast* blk = (const block_rocmfp4_fast*)row;
    for (std::int64_t ib = 0; ib < nb; ++ib)
      DecodeBlockSingle(blk[ib].qs, blk[ib].e, out + ib * 32);
  } else {
    throw std::runtime_error("DecodeRowRocmFp4: not a 100/101 type");
  }
}

}  // namespace superroc::quant_tool

#endif  // SUPERROC_QUANT_G4_ENCODE_HPP_
