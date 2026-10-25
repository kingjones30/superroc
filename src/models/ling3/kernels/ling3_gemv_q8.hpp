#pragma once
// LI1: LS2 phase-14 dropped-stream fixes -- see ling3_moe.hpp.
#include <hip/hip_runtime.h>
// LS1 2026-10-01: activation-quantised ROCmFP4 GEMV.
//
// The decode step's matvecs are ~78 % of GPU time and move only ~54 GB/s of the
// box's ~256 GB/s, so the bottleneck is the per-element work: ling3_moe.hip's
// k_gemv_wide decodes every FP4 weight to float and multiplies. This is the
// fork's MMVQ arithmetic instead -- the ACTIVATION is quantised to Q8_1 (int8 +
// one scale per 32 values), and the row dot is an integer dot of the int8-
// expanded FP4 codes against it, with one scale multiply per 32 weights. That
// is what ggml's rocmfp4_vec_dot_q4_0_q8_1 does, so this also moves gufo's
// numerics toward the reference: llama.cpp quantises the activation for these
// tensors too.
//
// Defined in kernels/ling3_gemv_q8.hip; declared here so ling3_moe.hip can call
// it. Nothing in rocm_backend.cpp has to change: ling3_gemv() routes ROCmFP4
// (types 100 and 101, which is every weight matrix in Ling-3.0-flash) here on
// its own.
//
// Q8_1 buffer contract (internal, not ggml's on-disk layout):
//   qs[0 .. k)          int8, element i at qs[i]
//   ds[0 .. k/32)       float, the scale of qs[32*b .. 32*b+32)
// The ROCmFP4 weight block is also 32 values, so weight block b pairs with
// ds[b].

#ifdef __cplusplus
extern "C" {
#endif

/// Slots for ling3_quant_q8(). Each is an independent pair of device buffers.
/// 0 is ling3_gemv_fp4()'s own scratch; 1 and 2 are the expert FFN's input
/// (shared by gate and up) and its SwiGLU output (input to down).
enum {
  LING3_Q8_SLOT_GENERIC = 0,
  LING3_Q8_SLOT_FFN_X = 1,
  LING3_Q8_SLOT_FFN_H = 2,
  LING3_Q8_SLOT_MOE_H = 3,
  // LS1 PHASE7 item 3: st->normed, quantised ONCE per layer and shared by that
  // layer's attention projections instead of being re-quantised by each of
  // them.
  LING3_Q8_SLOT_ATTN_NORM = 4
};

/// Quantise x[0..k) into a caller-chosen slot (k % 32 == 0). LS1 PHASE7 item 3.
///
/// This is the ONLY supported way to quantise once and reuse: the caller NAMES
/// the slot, so nothing can go stale. A pointer-keyed cache must never be used
/// here -- st->normed holds the NEXT layer's activation the moment rms_norm
/// returns, and a cache keyed on the pointer would silently hand back the
/// previous layer's vector.
int ling3_act_prepare(const float* x, int k, int slot,
                      hipStream_t stream = nullptr);

/// Quantise x[0..k) into a slot (k % 32 == 0). Returns LING3_OK or
/// LING3_ERR_ARG.
int ling3_quant_q8(const float* x, int k, int slot,
                   hipStream_t stream = nullptr);

/// Allocate (if needed) and return the device qs/ds of `slot` without
/// quantising. k % 32 == 0. Pointers stay valid until the slot is grown.
int ling3_q8_slot_device(int slot, int k, int8_t** qs, float** ds);

/// Matvec against the activation already quantised into `slot`.
int ling3_gemv_fp4_q8(int qtype, const void* W, int slot, float* y, int k,
                      int n, hipStream_t stream = nullptr);

/// RV25 wave1 F3: as ling3_gemv_fp4_q8, but the store is y[row] = 0.0f + s --
/// the exact value of the unfused hipMemset(y,0) + k_axpy(1.0f, p, y) pair, so
/// a down projection can write the caller's output directly with no scratch and
/// no second launch. Decode only (n <= 8192). Never use it where the old pair
/// was NOT preceded by a zero of the output: the 0.0f + normalises -0.0 to
/// +0.0.
int ling3_gemv_fp4_q8_zp(int qtype, const void* W, int slot, float* y, int k,
                         int n, hipStream_t stream = nullptr);

/// Matvec that quantises the activation itself.
int ling3_gemv_fp4(int qtype, const void* W, const float* x, float* y, int k,
                   int n, hipStream_t stream = nullptr);

/// Grouped, device-routed expert FFN for ONE token: reads the router's ids and
/// weights where ling3_route left them (both DEVICE pointers -- no D2H copy, no
/// host loop, and no sync), runs gate+up+SwiGLU in one launch, quantises that
/// for the down projection in a second, and does down plus the weighted expert
/// sum in a third, accumulating the n_used experts in a fixed slot order.
/// WRITES y (it does not accumulate into it). ROCmFP4 (100/101) and a single
/// token only; anything else returns LING3_ERR_ARG so the caller can fall back
/// to ling3_expert_ffn.
int ling3_expert_ffn_dev(int qtype, const void* gate, const void* up,
                         const void* down, const float* x, const int* ids_d,
                         const float* weights_d, int n_used, int n_expert,
                         int k, int n_ff, float limit, float* y,
                         hipStream_t stream = nullptr);

/// Register the router's device status word so ling3_moe_dev_check() can read
/// it. Called once by ling3_route, when it allocates the word it owns.
void ling3_moe_dev_register_route_rc(int* rc_d);

/// Read the decode step's device status words (grouped FFN + router) and clear
/// them. Blocking; call ONCE per token, after the last MoE layer -- never per
/// layer.
int ling3_moe_dev_check(void);

// ---------------------------------------------------------------------------
// LS1 phase 4 item 3: the small-m decode forward, for MTP verification.
// ---------------------------------------------------------------------------

/// LVL SPEC1: matvec for a Q6_K weight matrix, the format of the VL artifact's
/// LM head. One warp per output row, warp shuffle reduction, k a multiple of
/// 256. Declared here so ling3_moe.hip's ling3_gemv() can route LING3_Q6_K to
/// it instead of the per-element float path. Reads x as float; nothing is
/// quantised.
int ling3_gemv_q6k(int qtype, const void* W, const float* x, float* y, int k,
                   int n, hipStream_t stream);

/// LM1 SPEC4 item 2: quantise n_tokens * k floats (token-major) into the
/// small-m path's Q8_1 pair and hand the pair back, so a caller running several
/// matrices against ONE activation pays for one quantise instead of one per
/// matrix. This is the m-row counterpart of ling3_act_prepare +
/// ling3_gemv_fp4_q8 that `Step` uses (one quantisation of st->normed for every
/// projection in the layer).
///
/// The pair is SHARED with ling3_gemv_m and ling3_expert_ffn_m -- both quantise
/// into it -- so nothing may be held across a call to either. It is valid from
/// this call until the next one.
int ling3_quant_m(const float* x, int n_tokens, int k, hipStream_t stream,
                  const int8_t** xq, const float** xd);

/// Matvec of n rows against m activation columns (m = 1..4), reading each
/// weight row ONCE for all columns -- llama.cpp's `mul_mat_vec_q` with
/// ncols_dst <= 8. The activation is quantised to Q8_1 first; y is [m][n]
/// token-major, y[t * n + row]. ROCmFP4 only.
int ling3_gemv_m(int qtype, const void* W, const float* x, float* y, int k,
                 int n, int m, hipStream_t stream = nullptr);

/// As ling3_gemv_m, but against an activation the caller already holds as Q8_1
/// (xq[k*m], xd[(k/32)*m]), so a caller running many matrices against one
/// activation quantises once.
int ling3_gemv_q8_m(int qtype, const void* W, const int8_t* xq, const float* xd,
                    float* y, int k, int n, int m,
                    hipStream_t stream = nullptr);

/// Grouped expert FFN over n_tokens * n_used (token, slot) pairs, n_tokens
/// = 1..4. ids_d and weights_d are DEVICE pointers, flat and indexed s = token
/// * n_used + slot -- the per-pair expert id is read on the device, never on
/// the host. y is [n_tokens][k] token-major. No sync; call
/// ling3_expert_ffn_m_check() once afterwards for the status.
int ling3_expert_ffn_m(int qtype, const void* gate, const void* up,
                       const void* down, const float* x, const int* ids_d,
                       const float* weights_d, int n_tokens, int n_used,
                       int n_expert, int k, int n_ff, float limit, float* y,
                       hipStream_t stream = nullptr);

/// LV1 R3: n_used == 1 and n_tokens > 1 reads each weight row once. Otherwise
/// this is ling3_expert_ffn_m. Same status word; ling3_expert_ffn_m_check
/// covers it.
int ling3_expert_ffn_shared_m(int qtype, const void* gate, const void* up,
                              const void* down, const float* x,
                              const int* ids_d, const float* weights_d,
                              int n_tokens, int n_used, int n_expert, int k,
                              int n_ff, float limit, float* y,
                              hipStream_t stream = nullptr);

/// Read and clear the small-m path's status word. Blocking; once per verify
/// batch.
int ling3_expert_ffn_m_check(void);

// PHASE6: the warp-per-row variant of the plain matvec and of the grouped down
// projection. Both must agree BIT-EXACTLY with the CTA-per-row kernels above,
// which tools/ling3_gemv_q8_test.cpp checks, and LING3_GEMV_VARIANT=1 makes
// the shipped entry points use them (default 0, so nothing changes unset).
int ling3_gemv_fp4_q8_rows(int qtype, const void* W, int slot, float* y, int k,
                           int n, int block, hipStream_t stream = nullptr);
int ling3_expert_ffn_dev_tuned(int qtype, const void* gate, const void* up,
                               const void* down, const float* x,
                               const int* ids_d, const float* weights_d,
                               int n_used, int n_expert, int k, int n_ff,
                               float limit, float* y, int block,
                               hipStream_t stream = nullptr);
int ling3_expert_ffn_dev_rows(int qtype, const void* down, const int* ids_d,
                              const float* weights_d, int n_used, int n_expert,
                              int k, int n_ff, float* y, int block,
                              hipStream_t stream = nullptr);

/// Test-only: the small-m path's hs scratch.
int ling3_moe_hs_m_peek(float* out, int n);

/// Test-only: copy the grouped expert FFN's hs scratch back to the host.
int ling3_moe_hs_peek(float* out, int n);

/// Copy a slot's quantised activation back to the host (test-only diagnostic).
int ling3_q8_peek(int slot, int8_t* qs_out, float* ds_out, int k);

/// Exhaustively check the FP4 nibble -> int8 perm codebook against
/// 2 * ling3_rocmfp4_code. Returns the number of disagreeing byte lanes (0 =
/// good), or -1 on a HIP failure.
int ling3_gemv_q8_codebook_check(void);

#ifdef __cplusplus
}
#endif