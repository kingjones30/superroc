#pragma once
// Quant decode for Ling-3.0-flash GEMV/GEMM.
//
// Layout matches ggml (ne[0] = K, contiguous). One row is K elements.
// Block bytes are not padded. Q4_0_ROCMFP4_FAST blocks are 17 bytes, so a
// row's later blocks are not 4-byte aligned. Loads are volatile bytes so a
// widened vector load cannot walk off the end of a tightly sized buffer.
// Callers still pad allocations by LING3_TAIL_PAD bytes (see ling3_moe.hpp).
//
// Type 100/101 decode follows the fork's rocmfp4_decode /
// rocmfp4_dequantize_row (ggml/rocmfp4/rocmfp4.c) and S2's
// mmf16dq_rocmfp4_value magnitudes (mmf16dq.cuh). The integer code is cast to
// float, matching to_float (code 8 is +0, not the float negation's -0). Scales
// use the fork's unsigned-E4M3 half-scale bit pattern, with S2's e > 0x7e -> 0
// guard (mmf16dq_ue4m3), which matches rocmfp4_ue4m3_to_fp32_half(). Q8_0 and
// Q6_K follow ggml-quants.c dequantize_row_q8_0 / dequantize_row_q6_K.

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#define LING3_HD __host__ __device__
#elif defined(__HIPCC__) || defined(__CUDACC__)
#define LING3_HD __host__ __device__
#else
#define LING3_HD
#endif

enum { LING3_OK = 0, LING3_ERR_ARG = 1, LING3_ERR_TYPE = 2, LING3_ERR_HIP = 3 };

enum {
  LING3_F32 = 0,
  LING3_Q8_0 = 8,
  LING3_Q5_K = 13,
  LING3_Q6_K = 14,
  LING3_Q4_0_ROCMFP4 = 100,
  LING3_Q4_0_ROCMFP4_FAST = 101
};

// Readable bytes past the last row. Covers a widened tail load.
enum { LING3_TAIL_PAD = 64 };

// Fork: weights_sum clamp, smallest positive fp16 normal.
// Fork: clamp is applied only when limit > 1e-6.
static constexpr float LING3_WEIGHT_SUM_MIN = 0x1p-14f;
static constexpr float LING3_CLAMP_EPS = 1e-6f;

LING3_HD inline float ling3_u32_as_f32(uint32_t u) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = u;
  return v.f;
}

LING3_HD inline uint32_t ling3_f32_as_u32(float f) {
  union {
    float f;
    uint32_t u;
  } v;
  v.f = f;
  return v.u;
}

LING3_HD inline float ling3_neg_inf(void) {
  return ling3_u32_as_f32(0xFF800000u);
}

LING3_HD inline uint8_t ling3_ld8(const uint8_t* p) {
  const volatile uint8_t* v = p;
  return *v;
}

LING3_HD inline uint16_t ling3_ld16(const uint8_t* p) {
  uint16_t lo = ling3_ld8(p);
  uint16_t hi = ling3_ld8(p + 1);
  return (uint16_t)(lo | (hi << 8));
}

LING3_HD inline float ling3_ld_f32(const uint8_t* p) {
  uint32_t u = (uint32_t)ling3_ld8(p) | ((uint32_t)ling3_ld8(p + 1) << 8) |
               ((uint32_t)ling3_ld8(p + 2) << 16) |
               ((uint32_t)ling3_ld8(p + 3) << 24);
  return ling3_u32_as_f32(u);
}

// ggml_compute_fp16_to_fp32 (ggml-impl.h, Maratyszcza FP16). Finite values
// match a plain IEEE widen; NaN payloads match ggml, which is what to_float
// uses for Q8_0 / Q6_K scales.
LING3_HD inline float ling3_fp16_to_fp32(uint16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & 0x80000000u;
  const uint32_t two_w = w + w;
  const uint32_t exp_offset = 0xE0u << 23;
  const float exp_scale = 0x1.0p-112f;
  const float normalized_value =
      ling3_u32_as_f32((two_w >> 4) + exp_offset) * exp_scale;
  const uint32_t magic_mask = 126u << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value =
      ling3_u32_as_f32((two_w >> 17) | magic_mask) - magic_bias;
  const uint32_t denormalized_cutoff = 1u << 27;
  const uint32_t result =
      sign | (two_w < denormalized_cutoff ? ling3_f32_as_u32(denormalized_value)
                                          : ling3_f32_as_u32(normalized_value));
  return ling3_u32_as_f32(result);
}

// Unsigned E4M3 interpreted at half scale. e > 0x7e (NaN / sign bit) -> 0.
LING3_HD inline float ling3_ue4m3_half(uint8_t e) {
  if (e > 0x7e) {
    return 0.f;
  }
  int exp = (e >> 3) & 0xF;
  int man = e & 0x7;
  if (exp == 0) {
    return (float)man * (1.0f / 1024.0f);
  }
  uint32_t bits = ((uint32_t)exp + 119u) << 23 | ((uint32_t)man << 20);
  return ling3_u32_as_f32(bits);
}

// Fork rocmfp4_decode: mag = m<=4 ? m : 2m-4, sign bit 3.
// Equivalent to S2 mmf16dq_rocmfp4_value for m in 0..7, except code 8 is +0.
LING3_HD inline int ling3_rocmfp4_code(uint32_t q) {
  q &= 0x0fu;
  uint32_t mag3 = q & 7u;
  int mag = mag3 <= 4u ? (int)mag3 : (int)(2u * mag3 - 4u);
  return (q & 8u) ? -mag : mag;
}

LING3_HD inline int ling3_blck_size(int qtype) {
  switch (qtype) {
    case LING3_F32:
      return 1;
    case LING3_Q8_0:
      return 32;
    case LING3_Q5_K:
      return 256;
    case LING3_Q6_K:
      return 256;
    case LING3_Q4_0_ROCMFP4:
      return 32;
    case LING3_Q4_0_ROCMFP4_FAST:
      return 32;
    default:
      return 0;
  }
}

LING3_HD inline int ling3_type_size(int qtype) {
  switch (qtype) {
    case LING3_F32:
      return 4;
    case LING3_Q8_0:
      return 34;
    case LING3_Q5_K:
      return 176;
    case LING3_Q6_K:
      return 210;
    case LING3_Q4_0_ROCMFP4:
      return 18;
    case LING3_Q4_0_ROCMFP4_FAST:
      return 17;
    default:
      return 0;
  }
}

LING3_HD inline int ling3_type_ok(int qtype) {
  return ling3_blck_size(qtype) != 0;
}

// 0 if k is not a multiple of the block, or the type is unknown.
LING3_HD inline size_t ling3_row_bytes(int qtype, int k) {
  int b = ling3_blck_size(qtype);
  int t = ling3_type_size(qtype);
  if (b <= 0 || t <= 0 || k <= 0 || (k % b) != 0) {
    return 0;
  }
  return (size_t)(k / b) * (size_t)t;
}

// One decoded term. f32 value is (c1 == 1) ? (float)c0 * f : (f * (float)c1) *
// (float)c0. That association matches ggml: Q8/FP4 are a single multiply; Q6_K
// is (d * scale) * q.
struct ling3_term {
  int c0;
  int c1;
  float f;
};

// blk points at one quant block. local is the index within the block.
LING3_HD inline struct ling3_term ling3_term_block(int qtype,
                                                   const uint8_t* blk,
                                                   int local) {
  struct ling3_term t;
  t.c0 = 0;
  t.c1 = 1;
  t.f = 0.f;
  if (qtype == LING3_F32) {
    t.c0 = 1;
    t.f = ling3_ld_f32(blk + (size_t)local * 4);
    return t;
  }
  if (qtype == LING3_Q8_0) {
    // block_q8_0: ggml_half d; int8 qs[32]
    t.f = ling3_fp16_to_fp32(ling3_ld16(blk));
    t.c0 = (int)(int8_t)ling3_ld8(blk + 2 + local);
    return t;
  }
  if (qtype == LING3_Q4_0_ROCMFP4 || qtype == LING3_Q4_0_ROCMFP4_FAST) {
    int qs_i = local & 15;
    uint8_t b = ling3_ld8(blk + qs_i);
    int nib = (local < 16) ? (int)(b & 0x0f) : (int)(b >> 4);
    int eoff = 16;
    if (qtype == LING3_Q4_0_ROCMFP4 && local >= 16) {
      eoff = 17;
    }
    t.c0 = ling3_rocmfp4_code((uint32_t)nib);
    t.f = ling3_ue4m3_half(ling3_ld8(blk + eoff));
    return t;
  }
  if (qtype == LING3_Q6_K) {
    // ggml-quants.c dequantize_row_q6_K, one 256-element block.
    // ql[128], qh[64], scales[16], d fp16. Two 128-halves.
    int half = local >> 7;
    int within = local & 127;
    int piece = within >> 5;
    int l = within & 31;
    const uint8_t* ql = blk + half * 64;
    const uint8_t* qh = blk + 128 + half * 32;
    const uint8_t* sc = blk + 192 + half * 8;
    int ql_index = l + ((piece & 1) ? 32 : 0);
    uint8_t byte = ling3_ld8(ql + ql_index);
    int nibble = (piece >= 2) ? (int)(byte >> 4) : (int)(byte & 0x0f);
    int qhbits = (int)((ling3_ld8(qh + l) >> (piece * 2)) & 3);
    int is = l >> 4;
    int sc_i = piece * 2 + is;
    t.c0 = (nibble | (qhbits << 4)) - 32;
    t.c1 = (int)(int8_t)ling3_ld8(sc + sc_i);
    t.f = ling3_fp16_to_fp32(ling3_ld16(blk + 208));
    return t;
  }
  return t;
}

LING3_HD inline float ling3_term_f32(struct ling3_term t) {
  if (t.c1 == 1) {
    return (float)t.c0 * t.f;
  }
  return (t.f * (float)t.c1) * (float)t.c0;
}

// Q5_K is the one type here whose decoded value is a difference of two
// products, (d*sc)*q - (dmin*m), so it cannot ride ling3_term. Layout is
// block_q5_K: d fp16 (2) | dmin fp16 (2) | scales[12] | qh[32] | qs[128], 176
// bytes. Scales and mins are the packed 6-bit K-quant pair Q4_K uses.
// Association follows ggml dequantize_row_q5_K and this tree's core Q5Value,
// which ling3_dequant_parity pins to the fork.
LING3_HD inline float ling3_q5k_value(const uint8_t* blk, int local) {
  const int gg = local >> 6;  // 0..3, the 64-element group
  const int wv = local & 63;  // 0..63 within the group
  const int lane = wv & 31;   // 0..31
  const int lohalf = wv < 32;
  const uint8_t qb = ling3_ld8(blk + 48 + (size_t)gg * 32 + (size_t)lane);
  const int quant4 = lohalf ? (int)(qb & 0x0f) : (int)(qb >> 4);
  const int qhb = (int)ling3_ld8(blk + 16 + (size_t)lane);
  const int bit = 2 * gg + (lohalf ? 0 : 1);  // 0..7
  const int quant = quant4 + (((qhb >> bit) & 1) ? 16 : 0);
  const int sis = 2 * gg + (lohalf ? 0 : 1);  // 0..7
  uint8_t sc;
  uint8_t m;
  if (sis < 4) {
    sc = (uint8_t)(ling3_ld8(blk + 4 + (size_t)sis) & 0x3fu);
    m = (uint8_t)(ling3_ld8(blk + 8 + (size_t)sis) & 0x3fu);
  } else {
    sc = (uint8_t)((ling3_ld8(blk + 8 + (size_t)sis) & 0x0fu) |
                   (((ling3_ld8(blk + (size_t)sis) >> 6) & 0x03u) << 4));
    m = (uint8_t)((ling3_ld8(blk + 8 + (size_t)sis) >> 4) |
                  (((ling3_ld8(blk + 4 + (size_t)sis) >> 6) & 0x03u) << 4));
  }
  const float d = ling3_fp16_to_fp32(ling3_ld16(blk));
  const float dmin = ling3_fp16_to_fp32(ling3_ld16(blk + 2));
  return (d * (float)sc) * (float)quant - (dmin * (float)m);
}

LING3_HD inline float ling3_elem(int qtype, const void* row, int i) {
  int bsz = ling3_blck_size(qtype);
  int tsz = ling3_type_size(qtype);
  if (bsz <= 0) {
    return 0.f;
  }
  const uint8_t* p = (const uint8_t*)row + ((size_t)(i / bsz) * (size_t)tsz);
  return ling3_term_f32(ling3_term_block(qtype, p, i % bsz));
}

LING3_HD inline void ling3_dequant_row(int qtype, const void* row, float* y,
                                       int k) {
  for (int i = 0; i < k; ++i) {
    y[i] = ling3_elem(qtype, row, i);
  }
}

LING3_HD inline float ling3_dot_row(int qtype, const void* row, const float* x,
                                    int k) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  float acc = 0.f;
  for (int i = 0; i < k; ++i) {
    acc += ling3_elem(qtype, row, i) * x[i];
  }
  return acc;
}

// BailingMoeV3 / Step3.5 clamp (llama-graph.cpp):
//   silu(gate) = gate / (1 + exp(-gate))
//   if limit > 1e-6:  min(silu(gate), limit) * clamp(up, -limit, limit)
//   else:             silu(gate) * up
// Gate clamp is upper-only (ggml_clamp -inf..limit). Up is two-sided.
LING3_HD inline float ling3_swiglu_elem(float gate, float up, float limit) {
  float s = gate / (1.0f + expf(-gate));
  if (limit > LING3_CLAMP_EPS) {
    float gs = s <= limit ? s : limit;
    float uu = up <= limit ? up : limit;
    uu = uu >= -limit ? uu : -limit;
    return gs * uu;
  }
  return s * up;
}
