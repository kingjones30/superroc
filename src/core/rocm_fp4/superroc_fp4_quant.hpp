#pragma once
// Quant decode for Ling-3.0-flash GEMV/GEMM.
//
// Layout matches ggml (ne[0] = K, contiguous). One row is K elements.
// Block bytes are not padded. Q4_0_ROCMFP4_FAST blocks are 17 bytes, so a
// row's later blocks are not 4-byte aligned. Loads are volatile bytes so a
// widened vector load cannot walk off the end of a tightly sized buffer.
// Callers still pad allocations by SUPERROC_FP4_TAIL_PAD bytes (see
// superroc_fp4_moe.hpp).
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
#define SUPERROC_FP4_HD __host__ __device__
#elif defined(__HIPCC__) || defined(__CUDACC__)
#define SUPERROC_FP4_HD __host__ __device__
#else
#define SUPERROC_FP4_HD
#endif

enum {
  SUPERROC_FP4_OK = 0,
  SUPERROC_FP4_ERR_ARG = 1,
  SUPERROC_FP4_ERR_TYPE = 2,
  SUPERROC_FP4_ERR_HIP = 3,
  // SQ10 SPEC2 (R2): no free slot set for an unseen stream. Reported as
  // -SUPERROC_FP4_ERR_BUSY (negative, never a valid set index) by the resolver,
  // attach() and every entry that resolves a set.
  SUPERROC_FP4_ERR_BUSY = 4
};

enum {
  SUPERROC_FP4_F32 = 0,
  SUPERROC_FP4_Q8_0 = 8,
  SUPERROC_FP4_Q6_K = 14,
  SUPERROC_FP4_Q4_0_ROCMFP4 = 100,
  SUPERROC_FP4_Q4_0_ROCMFP4_FAST = 101
};

// Readable bytes past the last row. Covers a widened tail load.
enum { SUPERROC_FP4_TAIL_PAD = 64 };

// Fork: weights_sum clamp, smallest positive fp16 normal.
// Fork: clamp is applied only when limit > 1e-6.
static constexpr float SUPERROC_FP4_WEIGHT_SUM_MIN = 0x1p-14f;
static constexpr float SUPERROC_FP4_CLAMP_EPS = 1e-6f;

SUPERROC_FP4_HD inline float superroc_fp4_u32_as_f32(uint32_t u) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = u;
  return v.f;
}

SUPERROC_FP4_HD inline uint32_t superroc_fp4_f32_as_u32(float f) {
  union {
    float f;
    uint32_t u;
  } v;
  v.f = f;
  return v.u;
}

SUPERROC_FP4_HD inline float superroc_fp4_neg_inf(void) {
  return superroc_fp4_u32_as_f32(0xFF800000u);
}

SUPERROC_FP4_HD inline uint8_t superroc_fp4_ld8(const uint8_t* p) {
  const volatile uint8_t* v = p;
  return *v;
}

SUPERROC_FP4_HD inline uint16_t superroc_fp4_ld16(const uint8_t* p) {
  uint16_t lo = superroc_fp4_ld8(p);
  uint16_t hi = superroc_fp4_ld8(p + 1);
  return (uint16_t)(lo | (hi << 8));
}

SUPERROC_FP4_HD inline float superroc_fp4_ld_f32(const uint8_t* p) {
  uint32_t u = (uint32_t)superroc_fp4_ld8(p) |
               ((uint32_t)superroc_fp4_ld8(p + 1) << 8) |
               ((uint32_t)superroc_fp4_ld8(p + 2) << 16) |
               ((uint32_t)superroc_fp4_ld8(p + 3) << 24);
  return superroc_fp4_u32_as_f32(u);
}

// ggml_compute_fp16_to_fp32 (ggml-impl.h, Maratyszcza FP16). Finite values
// match a plain IEEE widen; NaN payloads match ggml, which is what to_float
// uses for Q8_0 / Q6_K scales.
SUPERROC_FP4_HD inline float superroc_fp4_fp16_to_fp32(uint16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & 0x80000000u;
  const uint32_t two_w = w + w;
  const uint32_t exp_offset = 0xE0u << 23;
  const float exp_scale = 0x1.0p-112f;
  const float normalized_value =
      superroc_fp4_u32_as_f32((two_w >> 4) + exp_offset) * exp_scale;
  const uint32_t magic_mask = 126u << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value =
      superroc_fp4_u32_as_f32((two_w >> 17) | magic_mask) - magic_bias;
  const uint32_t denormalized_cutoff = 1u << 27;
  const uint32_t result =
      sign |
      (two_w < denormalized_cutoff ? superroc_fp4_f32_as_u32(denormalized_value)
                                   : superroc_fp4_f32_as_u32(normalized_value));
  return superroc_fp4_u32_as_f32(result);
}

// Unsigned E4M3 interpreted at half scale. e > 0x7e (NaN / sign bit) -> 0.
SUPERROC_FP4_HD inline float superroc_fp4_ue4m3_half(uint8_t e) {
  if (e > 0x7e) {
    return 0.f;
  }
  int exp = (e >> 3) & 0xF;
  int man = e & 0x7;
  if (exp == 0) {
    return (float)man * (1.0f / 1024.0f);
  }
  uint32_t bits = ((uint32_t)exp + 119u) << 23 | ((uint32_t)man << 20);
  return superroc_fp4_u32_as_f32(bits);
}

// Fork rocmfp4_decode: mag = m<=4 ? m : 2m-4, sign bit 3.
// Equivalent to S2 mmf16dq_rocmfp4_value for m in 0..7, except code 8 is +0.
SUPERROC_FP4_HD inline int superroc_fp4_rocmfp4_code(uint32_t q) {
  q &= 0x0fu;
  uint32_t mag3 = q & 7u;
  int mag = mag3 <= 4u ? (int)mag3 : (int)(2u * mag3 - 4u);
  return (q & 8u) ? -mag : mag;
}

SUPERROC_FP4_HD inline int superroc_fp4_blck_size(int qtype) {
  switch (qtype) {
    case SUPERROC_FP4_F32:
      return 1;
    case SUPERROC_FP4_Q8_0:
      return 32;
    case SUPERROC_FP4_Q6_K:
      return 256;
    case SUPERROC_FP4_Q4_0_ROCMFP4:
      return 32;
    case SUPERROC_FP4_Q4_0_ROCMFP4_FAST:
      return 32;
    default:
      return 0;
  }
}

SUPERROC_FP4_HD inline int superroc_fp4_type_size(int qtype) {
  switch (qtype) {
    case SUPERROC_FP4_F32:
      return 4;
    case SUPERROC_FP4_Q8_0:
      return 34;
    case SUPERROC_FP4_Q6_K:
      return 210;
    case SUPERROC_FP4_Q4_0_ROCMFP4:
      return 18;
    case SUPERROC_FP4_Q4_0_ROCMFP4_FAST:
      return 17;
    default:
      return 0;
  }
}

SUPERROC_FP4_HD inline int superroc_fp4_type_ok(int qtype) {
  return superroc_fp4_blck_size(qtype) != 0;
}

// 0 if k is not a multiple of the block, or the type is unknown.
SUPERROC_FP4_HD inline size_t superroc_fp4_row_bytes(int qtype, int k) {
  int b = superroc_fp4_blck_size(qtype);
  int t = superroc_fp4_type_size(qtype);
  if (b <= 0 || t <= 0 || k <= 0 || (k % b) != 0) {
    return 0;
  }
  return (size_t)(k / b) * (size_t)t;
}

// One decoded term. f32 value is (c1 == 1) ? (float)c0 * f : (f * (float)c1) *
// (float)c0. That association matches ggml: Q8/FP4 are a single multiply; Q6_K
// is (d * scale) * q.
struct superroc_fp4_term {
  int c0;
  int c1;
  float f;
};

// blk points at one quant block. local is the index within the block.
SUPERROC_FP4_HD inline struct superroc_fp4_term superroc_fp4_term_block(
    int qtype, const uint8_t* blk, int local) {
  struct superroc_fp4_term t;
  t.c0 = 0;
  t.c1 = 1;
  t.f = 0.f;
  if (qtype == SUPERROC_FP4_F32) {
    t.c0 = 1;
    t.f = superroc_fp4_ld_f32(blk + (size_t)local * 4);
    return t;
  }
  if (qtype == SUPERROC_FP4_Q8_0) {
    // block_q8_0: ggml_half d; int8 qs[32]
    t.f = superroc_fp4_fp16_to_fp32(superroc_fp4_ld16(blk));
    t.c0 = (int)(int8_t)superroc_fp4_ld8(blk + 2 + local);
    return t;
  }
  if (qtype == SUPERROC_FP4_Q4_0_ROCMFP4 ||
      qtype == SUPERROC_FP4_Q4_0_ROCMFP4_FAST) {
    int qs_i = local & 15;
    uint8_t b = superroc_fp4_ld8(blk + qs_i);
    int nib = (local < 16) ? (int)(b & 0x0f) : (int)(b >> 4);
    int eoff = 16;
    if (qtype == SUPERROC_FP4_Q4_0_ROCMFP4 && local >= 16) {
      eoff = 17;
    }
    t.c0 = superroc_fp4_rocmfp4_code((uint32_t)nib);
    t.f = superroc_fp4_ue4m3_half(superroc_fp4_ld8(blk + eoff));
    return t;
  }
  if (qtype == SUPERROC_FP4_Q6_K) {
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
    uint8_t byte = superroc_fp4_ld8(ql + ql_index);
    int nibble = (piece >= 2) ? (int)(byte >> 4) : (int)(byte & 0x0f);
    int qhbits = (int)((superroc_fp4_ld8(qh + l) >> (piece * 2)) & 3);
    int is = l >> 4;
    int sc_i = piece * 2 + is;
    t.c0 = (nibble | (qhbits << 4)) - 32;
    t.c1 = (int)(int8_t)superroc_fp4_ld8(sc + sc_i);
    t.f = superroc_fp4_fp16_to_fp32(superroc_fp4_ld16(blk + 208));
    return t;
  }
  return t;
}

SUPERROC_FP4_HD inline float superroc_fp4_term_f32(struct superroc_fp4_term t) {
  if (t.c1 == 1) {
    return (float)t.c0 * t.f;
  }
  return (t.f * (float)t.c1) * (float)t.c0;
}

SUPERROC_FP4_HD inline float superroc_fp4_elem(int qtype, const void* row,
                                               int i) {
  int bsz = superroc_fp4_blck_size(qtype);
  int tsz = superroc_fp4_type_size(qtype);
  if (bsz <= 0) {
    return 0.f;
  }
  const uint8_t* p = (const uint8_t*)row + ((size_t)(i / bsz) * (size_t)tsz);
  return superroc_fp4_term_f32(superroc_fp4_term_block(qtype, p, i % bsz));
}

SUPERROC_FP4_HD inline void superroc_fp4_dequant_row(int qtype, const void* row,
                                                     float* y, int k) {
  for (int i = 0; i < k; ++i) {
    y[i] = superroc_fp4_elem(qtype, row, i);
  }
}

SUPERROC_FP4_HD inline float superroc_fp4_dot_row(int qtype, const void* row,
                                                  const float* x, int k) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  float acc = 0.f;
  for (int i = 0; i < k; ++i) {
    acc += superroc_fp4_elem(qtype, row, i) * x[i];
  }
  return acc;
}

// BailingMoeV3 / Step3.5 clamp (llama-graph.cpp):
//   silu(gate) = gate / (1 + exp(-gate))
//   if limit > 1e-6:  min(silu(gate), limit) * clamp(up, -limit, limit)
//   else:             silu(gate) * up
// Gate clamp is upper-only (ggml_clamp -inf..limit). Up is two-sided.
SUPERROC_FP4_HD inline float superroc_fp4_swiglu_elem(float gate, float up,
                                                      float limit) {
  float s = gate / (1.0f + expf(-gate));
  if (limit > SUPERROC_FP4_CLAMP_EPS) {
    float gs = s <= limit ? s : limit;
    float uu = up <= limit ? up : limit;
    uu = uu >= -limit ? uu : -limit;
    return gs * uu;
  }
  return s * up;
}
