#ifndef SUPERROC_CORE_ROCM_FP4_FP4_F16_PREFILL_HPP_
#define SUPERROC_CORE_ROCM_FP4_FP4_F16_PREFILL_HPP_

// SQ1 (Super ROC wave, A1a; rd1 DESIGN.md section 2.2.1): the Flash-Next
// FP4->F16 prefill kernels, promoted from
// src/models/qwen38_flash_next/kernels/rocm/{kernels.hpp,fp4_f16.hpp}
// at 1194fea. The kernel and launcher symbol names are UNCHANGED (same
// namespaces) so the HIP fat-binary symdiff shows a pure move; the
// qwen family (src/models/qwen/hip/prefill_chunk.cpp) calls the same
// launchers. (The Flash-Next namespace inside src/core is deliberate
// for symbol stability; the B2 op-registry rename is Jay's follow-up.)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// LFP (wave 3): FP4 weight decode to F16 for the Flash-Next F16 WMMA GEMMs.
// Shared by the dense FP4 kernel (LFP) and the routed FP4 kernel (LFQ).
//
// Types 100 (ROCMFP4, 18-byte blocks, one ue4m3 scale per 16-value half) and
// 101 (ROCMFP4_FAST, 17-byte blocks, one scale per 32-value block). Layout:
// blk[0..15] code bytes, byte i holding element i (low nibble) and element
// i+16 (high nibble); scale at blk[16] (elements 0..15) and, for type 100,
// blk[17] (elements 16..31). Decode (code x ue4m3_half, code 8 = +0,
// e > 0x7e -> 0) matches src/core/rocm_fp4/superroc_fp4_quant.hpp, the CPU
// oracle the exactness test compares against bit-exactly.
//
// Exactness: |code| <= 10 (4 significant bits) and the ue4m3 scale (4
// significant bits) multiply to at most 8 significant bits, exactly
// representable in F16; the exponent range ([2^-10, 2400]) sits inside F16's.
// (The wave brief says +/-12; the oracle and the fork codebook say
// 2*7-4 = 10. Max code 10 = 0x4900, low byte still 0x00.)
//
// The F16 codes of every nibble have a zero low byte, so a code is one table
// byte shifted to the F16 high position; the magic-number trick used for Q8_0
// (CodesToHalves) does NOT apply here (1024 * scale overflows F16). Code 8's
// -0 is normalized to +0 by scaling with hfma(-0, s, +0) = +0 (IEEE RN),
// at the same cost as a multiply.

#include <cstdint>

// SPEC2 unification (LFQ): the routed kernel's scalar twins live here
// so this header is the single FP4->F16 decode source. The device-only
// parts are gated for non-HIP translation units (the routed unit test
// compiles as CXX and needs only the scalars).
#if defined(__HIPCC__) || defined(__HIP__) || defined(__CUDACC__) || \
    defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define FP4_F16_HAVE_HIP 1
#define FP4_F16_HD __host__ __device__
#include <hip/hip_fp16.h>
#else
#define FP4_F16_HAVE_HIP 0
#define FP4_F16_HD
#endif

namespace superroc::models::qwen38_flash_next::rocm::fp4_f16 {

// ue4m3 scale byte -> F16 bits. Value-identical to superroc_fp4_ue4m3_half:
// e > 0x7e -> +0; exp == 0 -> man * 2^-10 (normal in F16: 2^-10 >> 2^-14);
// else 2^(exp-8) * (1 + man/8).
constexpr std::uint16_t Fp4ScaleByteToHalf(std::uint8_t e) {
  if (e > 0x7e) {
    return 0;
  }
  const unsigned exp = (e >> 3) & 0xF;
  const unsigned man = e & 0x7;
  if (exp == 0) {
    if (man == 0) {
      return 0;
    }
    if (man >= 4) {
      return static_cast<std::uint16_t>((7u << 10) | ((man & 3u) << 8));
    }
    if (man >= 2) {
      return static_cast<std::uint16_t>((6u << 10) | ((man & 1u) << 9));
    }
    return 5u << 10;
  }
  return static_cast<std::uint16_t>(((exp + 7u) << 10) | (man << 7));
}

// 256-entry scale table, compile-time, constant memory (same precedent as
// GGML_TABLE_BEGIN in mmq/ggml-common.h). No host upload, no init hook.
struct Fp4ScaleTable {
  std::uint16_t v[256];
};
constexpr Fp4ScaleTable Fp4BuildScaleTable() {
  Fp4ScaleTable t{};
  for (unsigned i = 0; i < 256; ++i) {
    t.v[i] = Fp4ScaleByteToHalf(static_cast<std::uint8_t>(i));
  }
  return t;
}
// Scalar twins of the device decode (moved here verbatim from the retired
// fp4_f16_routed.hpp, SPEC2 unification; the routed unit test checks them
// against the superroc_fp4 CPU oracle). Callable from host and device.

// Unsigned E4M3 scale byte -> F16 bits. Exact everywhere: normal scales need
// 4 significand bits, subnormals at most 3 after normalizing (F16's exponent
// range covers man*2^-10 as a normal), and 0x7F+ matches the oracle's 0.
FP4_F16_HD inline std::uint16_t Fp4Ue4m3ToHalfU16(std::uint8_t e) {
  if (e > 0x7EU) {
    return 0;
  }
  const std::uint32_t exp = (e >> 3) & 0xFU;
  const std::uint32_t man = e & 7U;
  if (exp == 0) {
    if (man == 0) {
      return 0;
    }
    // man*2^-10 = (1 + f/8) * 2^(l-10), l = floor(log2(man)).
    const std::uint32_t l = man >= 4U ? 2U : (man >= 2U ? 1U : 0U);
    const std::uint32_t man3 = (man - (1U << l)) << (3U - l);
    return static_cast<std::uint16_t>(((l + 5U) << 10) | (man3 << 7));
  }
  return static_cast<std::uint16_t>(((exp + 7U) << 10) | (man << 7));
}

// One FP4 nibble (low 4 bits) -> signed code biased by +16, scalar twin of
// Fp4NibblesToBiased below. Nibble 8 is +0, like the oracle.
FP4_F16_HD inline std::uint8_t Fp4CodeBiased(std::uint32_t nib) {
  // Indexed by nibble: 16,17,18,19,20,22,24,26,16,15,14,13,12,10,8,6.
  constexpr std::uint8_t kTable[16] = {16, 17, 18, 19, 20, 22, 24, 26,
                                       16, 15, 14, 13, 12, 10, 8,  6};
  return kTable[nib & 0xFU];
}

#if FP4_F16_HAVE_HIP
static const __device__ Fp4ScaleTable kFp4ScaleHalf = Fp4BuildScaleTable();

// Two scale bytes -> half2 from the table (s0 = elements 0..15, s1 = 16..31;
// type 101 passes the same byte twice). Device only.
__device__ __forceinline__ __half2 Fp4Scale2(std::uint8_t s0, std::uint8_t s1) {
  const std::uint32_t lo = kFp4ScaleHalf.v[s0];
  const std::uint32_t hi = kFp4ScaleHalf.v[s1];
  return __builtin_bit_cast(__half2, lo | (hi << 16));
}

// Four nibbles (bits [3:0] of each byte of `nib`; upper bits ignored, so the
// caller passes the packed dword for the low half and packed >> 4 for the
// high half) -> the F16 codes of four elements as two half2 bit patterns.
// Pure ALU (3 perms); mirrors the fork's Codebook10 selector order
// (superroc_fp4_wmma.hip fp4_sub16): the mag perm's selector bit 2 picks the
// dword, bits [1:0] the byte, so MAG0 covers mag3 0..3 and MAG1 mag3 4..7.
// The sign (bit 3 of each nibble) shifts to the F16 sign bit and is OR-ed in
// (the dword shift leaves no residue: every byte is 0x00 or 0x08).
// Device only.
__device__ __forceinline__ void Fp4NibblesToCodes(std::uint32_t nib,
                                                  std::uint32_t& c0,
                                                  std::uint32_t& c1) {
  // mag3 0..3 -> F16 high bytes 0x00,0x3C,0x40,0x42 (codes 0,1,2,3);
  // mag3 4..7 -> 0x44,0x46,0x48,0x49 (codes 4,6,8,10).
  constexpr std::uint32_t kMag0 = 0x42403C00u;
  constexpr std::uint32_t kMag1 = 0x49484644u;
  const std::uint32_t mh =
      __builtin_amdgcn_perm(kMag1, kMag0, nib & 0x07070707u);
  const std::uint32_t hi = mh | ((nib & 0x08080808u) << 4);
  // F16 codes have a zero low byte; interleave with zeros.
  c0 = __builtin_amdgcn_perm(hi, 0u, 0x05010400u);
  c1 = __builtin_amdgcn_perm(hi, 0u, 0x07030602u);
}

// One full 32-element block: code4 = the 16 code bytes as 4 dwords, s0/s1 =
// the scale bytes (type 101 passes the same byte twice). h[16] comes out in K
// order. Scaling uses hfma(code, scale, +0): every product is exact, and
// code 8's -0 normalizes to the oracle's +0. Device only.
__device__ __forceinline__ void Fp4DecodeBlock(const std::uint32_t code4[4],
                                               std::uint8_t s0, std::uint8_t s1,
                                               __half2 h[16]) {
  const __half2 sc_lo = Fp4Scale2(s0, s0);
  const __half2 sc_hi = Fp4Scale2(s1, s1);
  const __half2 zero2 = __floats2half2_rn(0.0F, 0.0F);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    std::uint32_t c0 = 0;
    std::uint32_t c1 = 0;
    Fp4NibblesToCodes(code4[i], c0, c1);
    h[2 * i] = __hfma2(__builtin_bit_cast(__half2, c0), sc_lo, zero2);
    h[2 * i + 1] = __hfma2(__builtin_bit_cast(__half2, c1), sc_lo, zero2);
    Fp4NibblesToCodes(code4[i] >> 4, c0, c1);
    h[8 + 2 * i] = __hfma2(__builtin_bit_cast(__half2, c0), sc_hi, zero2);
    h[8 + 2 * i + 1] = __hfma2(__builtin_bit_cast(__half2, c1), sc_hi, zero2);
  }
}

// Four nibbles (low 4 bits of each byte of nib) -> four biased code bytes
// (code + 16, matching Fp4CodeBiased). Moved here verbatim from the retired
// fp4_f16_routed.hpp (SPEC2 unification). The fork's Codebook10 shape: two
// 8-entry perm LUTs (sign 0 / sign 1) plus a perm blend on bit 3. The caller
// feeds the bytes to the 1040 magic (0x6400 | q = 1024 + q, minus 1040),
// so the FMA computes code * scale with a zero offset. Device only.
__device__ __forceinline__ std::uint32_t Fp4NibblesToBiased(std::uint32_t nib) {
  // Bytes (little-endian): entries 0-3 then 4-7 of each sign table.
  constexpr std::uint32_t kBias0 = 0x13121110U;  // 16,17,18,19
  constexpr std::uint32_t kBias1 = 0x1a181614U;  // 20,22,24,26
  constexpr std::uint32_t kBias2 = 0x0d0e0f10U;  // 16,15,14,13
  constexpr std::uint32_t kBias3 = 0x06080a0cU;  // 12,10,8,6
  const std::uint32_t idx = nib & 0x07070707U;
  const std::uint32_t v_low = __builtin_amdgcn_perm(kBias1, kBias0, idx);
  const std::uint32_t v_high = __builtin_amdgcn_perm(kBias3, kBias2, idx);
  const std::uint32_t mask = 0x03020100U | ((nib & 0x08080808U) >> 1);
  return __builtin_amdgcn_perm(v_high, v_low, mask);
}

// Contracts this header keeps (LFR DESIGN.md section 6, adopted SPEC2):
// 1. Every decode entry point returns the EXACT value: max_abs against the
//    F32 oracle (superroc_fp4_quant.hpp) is 0 over all 256 scales x 16 codes x
//    both types. The product has <= 8 significant bits (F16 has 11).
// 2. The only permitted difference from the oracle is the sign of zero (+0
//    vs -0), and only for code = +/-0. Tests assert max_abs == 0 AND no
//    non-sign-zero diffs, never bitwise equality.
// 3. fast == true never reads byte 17 (a FAST block's last byte may be the
//    tensor's last byte).
// 4. No table larger than 512 B; no __constant__; perm-based constants only.
// 5. Determinism: no atomics, no cross-lane reductions; K-order staging.

// ue4m3 scale byte -> F32 scale. Value-identical to superroc_fp4_ue4m3_half for
// the int8 route's per-block float scale (SPEC2 item 3): e > 0x7e -> +0;
// exp == 0 -> man / 1024; else 2^(exp-8) * (1 + man/8). Host + device.
constexpr float Fp4ScaleByteToFloat(std::uint8_t e) {
  if (e > 0x7e) {
    return 0.0F;
  }
  const unsigned exp = (e >> 3) & 0xF;
  const unsigned man = e & 0x7;
  if (exp == 0) {
    return static_cast<float>(man) / 1024.0F;
  }
  // 2^(exp-8): exp in 1..14 -> -7..6, exact; (8+man)/8 exact.
  unsigned bits = ((exp + 119u) << 23) | (man << 20);
  return __builtin_bit_cast(float, bits);
}

// Four nibbles (bits [3:0] of each byte of `nib`; upper bits ignored) ->
// four signed int8 FP4 codes packed little-endian in `out`. Pure ALU
// (3 perms): a magnitude gather, a negated-magnitude gather, and a select
// by the sign bits. Matches superroc_fp4_rocmfp4_code on all 16 nibbles.
// Device only.
__device__ __forceinline__ void Fp4NibblesToInt8(std::uint32_t nib,
                                                 std::uint32_t& out) {
  // mag3 0..3 -> {0,1,2,3}, mag3 4..7 -> {4,6,8,10}; negated the same.
  constexpr std::uint32_t kPos0 = 0x03020100u;
  constexpr std::uint32_t kPos1 = 0x0A080604u;
  constexpr std::uint32_t kNeg0 = 0xFDFEFF00u;  // {0,-1,-2,-3}
  constexpr std::uint32_t kNeg1 = 0xF6F8FAFCu;  // {-4,-6,-8,-10}
  const std::uint32_t pos =
      __builtin_amdgcn_perm(kPos1, kPos0, nib & 0x07070707u);
  const std::uint32_t neg =
      __builtin_amdgcn_perm(kNeg1, kNeg0, nib & 0x07070707u);
  // Selector byte i: i for a clear sign bit (second operand = pos),
  // 4+i for a set one (first operand = neg).
  const std::uint32_t sel = 0x03020100u | ((nib & 0x08080808u) >> 1);
  out = __builtin_amdgcn_perm(neg, pos, sel);
}

#endif

}  // namespace superroc::models::qwen38_flash_next::rocm::fp4_f16

namespace superroc::models::qwen38_flash_next::rocm {

/// W8A8 WMMA GEMM over FP4 FAST weights (LFP SPEC2 item 3): the same
/// tiled-Q8 activation side as W8A8Gemm; weights decode to int8 codes plus
/// the ue4m3 block scale. FAST only (fast == false returns false).
bool Fp4Int8Gemm(const void* w, const void* x_tiled, float* out,
                 std::size_t batch, std::size_t m, std::size_t k, bool fast,
                 hipStream_t stream);

/// F16 activation rows [batch][k], ROCmFP4 weights (type 100 when fast is
/// false, 101 when true) dequantized to F16 in LDS, F32 accumulation.
/// out is [batch][m]. Same tile plan as DenseF16Gemm, plain row mapping.
/// Unsupported shapes launch nothing. (LFP.)
bool Fp4DenseF16Gemm(const void* w, const __half* x, float* out,
                     std::size_t batch, std::size_t m, std::size_t k, bool fast,
                     hipStream_t stream);

/// Fused FP4 SSM projection (LFP SPEC2 item 1b): rows [0,10240) from the
/// type-100 qkv tensor, rows [10240,16384) from the type-101 gate tensor,
/// one activation pass, with the SsmConv epilogue (raw rows to `qkvz`,
/// convolved qkv to `convolved`, tile edges via SsmConvBoundaryKernel).
/// Wide batches only (n_tokens >= 1024); anything else returns false.
/// Split-output FP4 GEMM (LFP SPEC2 item 1b): rows [h*512, h*512+256)
/// to `q`, rows [h*512+256, h*512+512) to `gate` (the 24-head x d256
/// interleave); q/gate are [batch][m/2]. Exact geometry (m == 12288,
/// k == 2560) and wide batches only; anything else returns false.
bool Fp4SplitF16Gemm(const void* w, const __half* x, float* q, float* gate,
                     std::size_t batch, std::size_t m, std::size_t k, bool fast,
                     hipStream_t stream);

bool Fp4SsmF16Gemm(const void* w0, const void* w1, const __half* x,
                   const float* conv_w, const float* history, float* qkvz,
                   float* convolved, std::uint32_t n_tokens, std::uint32_t m,
                   std::uint32_t k, std::uint32_t channels,
                   std::uint32_t kernel, hipStream_t stream);

// SQ1 occupancy proof hook (rd1 section 2.4): every ported
// instantiation must hold >= 2 waves/SIMD. Returns the minimum
// waves/SIMD x100 over all 15 instantiations, or -1 when a HIP query
// fails; writes one line per instantiation into report (truncated to
// report_bytes, always NUL-terminated when report_bytes > 0).
int Fp4F16PrefillMinWavesPerSimd(char* report, std::size_t report_bytes);

}  // namespace superroc::models::qwen38_flash_next::rocm

#endif  // SUPERROC_CORE_ROCM_FP4_FP4_F16_PREFILL_HPP_
