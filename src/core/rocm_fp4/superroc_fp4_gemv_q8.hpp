#pragma once
// LI1: LS2 phase-14 dropped-stream fixes -- see superroc_fp4_moe.hpp.
#include <hip/hip_runtime.h>

// SQ10: slot-set ownership. Defined when this header declares the ownership
// API below (configure/attach/warm/check_set/peek_set); tests use it to
// compile their ownership checks only against a patched tree.
#define SUPERROC_FP4_SLOT_OWNER 1
// LS1 2026-10-01: activation-quantised ROCmFP4 GEMV.
//
// The decode step's matvecs are ~78 % of GPU time and move only ~54 GB/s of the
// box's ~256 GB/s, so the bottleneck is the per-element work:
// superroc_fp4_moe.hip's k_gemv_wide decodes every FP4 weight to float and
// multiplies. This is the fork's MMVQ arithmetic instead -- the ACTIVATION is
// quantised to Q8_1 (int8 + one scale per 32 values), and the row dot is an
// integer dot of the int8- expanded FP4 codes against it, with one scale
// multiply per 32 weights. That is what ggml's rocmfp4_vec_dot_q4_0_q8_1 does,
// so this also moves superroc's numerics toward the reference: llama.cpp
// quantises the activation for these tensors too.
//
// Defined in kernels/superroc_fp4_gemv_q8.hip; declared here so
// superroc_fp4_moe.hip can call it. Nothing in rocm_backend.cpp has to change:
// superroc_fp4_gemv() routes ROCmFP4 (types 100 and 101, which is every weight
// matrix in Ling-3.0-flash) here on its own.
//
// Q8_1 buffer contract (internal, not ggml's on-disk layout):
//   qs[0 .. k)          int8, element i at qs[i]
//   ds[0 .. k/32)       float, the scale of qs[32*b .. 32*b+32)
// The ROCmFP4 weight block is also 32 values, so weight block b pairs with
// ds[b].

#ifdef __cplusplus
extern "C" {
#endif

/// Slots for superroc_fp4_quant_q8(). Each is an independent pair of device
/// buffers. 0 is superroc_fp4_gemv_fp4()'s own scratch; 1 and 2 are the expert
/// FFN's input (shared by gate and up) and its SwiGLU output (input to down).
enum {
  SUPERROC_FP4_Q8_SLOT_GENERIC = 0,
  SUPERROC_FP4_Q8_SLOT_FFN_X = 1,
  SUPERROC_FP4_Q8_SLOT_FFN_H = 2,
  SUPERROC_FP4_Q8_SLOT_MOE_H = 3,
  // LS1 PHASE7 item 3: st->normed, quantised ONCE per layer and shared by that
  // layer's attention projections instead of being re-quantised by each of
  // them.
  SUPERROC_FP4_Q8_SLOT_ATTN_NORM = 4,
  // lqw:capture-warmup. Slot count for WarmupRocmFp4Slots; keep in step
  // with the kernel enum in superroc_fp4_gemv_q8.hip.
  SUPERROC_FP4_Q8_SLOT_COUNT = SUPERROC_FP4_Q8_SLOT_ATTN_NORM + 1,
};

/// Quantise x[0..k) into a caller-chosen slot (k % 32 == 0). LS1 PHASE7 item 3.
///
/// This is the ONLY supported way to quantise once and reuse: the caller NAMES
/// the slot, so nothing can go stale. A pointer-keyed cache must never be used
/// here -- st->normed holds the NEXT layer's activation the moment rms_norm
/// returns, and a cache keyed on the pointer would silently hand back the
/// previous layer's vector.
int superroc_fp4_act_prepare(const float* x, int k, int slot,
                             hipStream_t stream = nullptr);

/// Quantise x[0..k) into a slot (k % 32 == 0). Returns SUPERROC_FP4_OK or
/// SUPERROC_FP4_ERR_ARG.
int superroc_fp4_quant_q8(const float* x, int k, int slot,
                          hipStream_t stream = nullptr);

/// Matvec against the activation already quantised into `slot`.
int superroc_fp4_gemv_fp4_q8(int qtype, const void* W, int slot, float* y,
                             int k, int n, hipStream_t stream = nullptr);

/// Matvec that quantises the activation itself.
int superroc_fp4_gemv_fp4(int qtype, const void* W, const float* x, float* y,
                          int k, int n, hipStream_t stream = nullptr);

/// Grouped, device-routed expert FFN for ONE token: reads the router's ids and
/// weights where superroc_fp4_route left them (both DEVICE pointers -- no D2H
/// copy, no host loop, and no sync), runs gate+up+SwiGLU in one launch,
/// quantises that for the down projection in a second, and does down plus the
/// weighted expert sum in a third, accumulating the n_used experts in a fixed
/// slot order. WRITES y (it does not accumulate into it). ROCmFP4 (100/101) and
/// a single token only; anything else returns SUPERROC_FP4_ERR_ARG so the
/// caller can fall back to superroc_fp4_expert_ffn.
int superroc_fp4_expert_ffn_dev(int qtype, const void* gate, const void* up,
                                const void* down, const float* x,
                                const int* ids_d, const float* weights_d,
                                int n_used, int n_expert, int k, int n_ff,
                                float limit, float* y,
                                hipStream_t stream = nullptr);

/// Super ROC QSM: superroc_fp4_expert_ffn_dev that does NOT quantise x: it
/// reuses the Q8 copy of x already in this stream's FFN_X slot. Valid only
/// right after a superroc_fp4_expert_ffn_dev on the SAME stream with the SAME x
/// and k (the shared expert after the routed experts of one token), with
/// nothing in between quantising into FFN_X. Results are bit-identical to
/// superroc_fp4_expert_ffn_dev; one quantise launch fewer.
int superroc_fp4_expert_ffn_dev_xq(int qtype, const void* gate, const void* up,
                                   const void* down, const float* x,
                                   const int* ids_d, const float* weights_d,
                                   int n_used, int n_expert, int k, int n_ff,
                                   float limit, float* y,
                                   hipStream_t stream = nullptr);

/// Register the router's device status word so superroc_fp4_moe_dev_check() can
/// read it. Called once by superroc_fp4_route, when it allocates the word it
/// owns.
void superroc_fp4_moe_dev_register_route_rc(int* rc_d);

/// Read the decode step's device status words (grouped FFN + router) and clear
/// them. Blocking; call ONCE per token, after the last MoE layer -- never per
/// layer.
int superroc_fp4_moe_dev_check(void);

// ---------------------------------------------------------------------------
// LS1 phase 4 item 3: the small-m decode forward, for MTP verification.
// ---------------------------------------------------------------------------

/// LM1 SPEC4 item 2: quantise n_tokens * k floats (token-major) into the
/// small-m path's Q8_1 pair and hand the pair back, so a caller running several
/// matrices against ONE activation pays for one quantise instead of one per
/// matrix. This is the m-row counterpart of superroc_fp4_act_prepare +
/// superroc_fp4_gemv_fp4_q8 that `Step` uses (one quantisation of st->normed
/// for every projection in the layer).
///
/// The pair is SHARED with superroc_fp4_gemv_m and superroc_fp4_expert_ffn_m --
/// both quantise into it -- so nothing may be held across a call to either. It
/// is valid from this call until the next one.
int superroc_fp4_quant_m(const float* x, int n_tokens, int k,
                         hipStream_t stream, const int8_t** xq,
                         const float** xd);

/// Matvec of n rows against m activation columns (m = 1..4), reading each
/// weight row ONCE for all columns -- llama.cpp's `mul_mat_vec_q` with
/// ncols_dst <= 8. The activation is quantised to Q8_1 first; y is [m][n]
/// token-major, y[t * n + row]. ROCmFP4 only.
int superroc_fp4_gemv_m(int qtype, const void* W, const float* x, float* y,
                        int k, int n, int m, hipStream_t stream = nullptr);

/// As superroc_fp4_gemv_m, but against an activation the caller already holds
/// as Q8_1 (xq[k*m], xd[(k/32)*m]), so a caller running many matrices against
/// one activation quantises once.
int superroc_fp4_gemv_q8_m(int qtype, const void* W, const int8_t* xq,
                           const float* xd, float* y, int k, int n, int m,
                           hipStream_t stream = nullptr);

/// Grouped expert FFN over n_tokens * n_used (token, slot) pairs, n_tokens
/// = 1..4. ids_d and weights_d are DEVICE pointers, flat and indexed s = token
/// * n_used + slot -- the per-pair expert id is read on the device, never on
/// the host. y is [n_tokens][k] token-major. No sync; call
/// superroc_fp4_expert_ffn_m_check() once afterwards for the status.
int superroc_fp4_expert_ffn_m(int qtype, const void* gate, const void* up,
                              const void* down, const float* x,
                              const int* ids_d, const float* weights_d,
                              int n_tokens, int n_used, int n_expert, int k,
                              int n_ff, float limit, float* y,
                              hipStream_t stream = nullptr);

/// LV1 R3: n_used == 1 and n_tokens > 1 reads each weight row once. Otherwise
/// this is superroc_fp4_expert_ffn_m. Same status word;
/// superroc_fp4_expert_ffn_m_check covers it.
int superroc_fp4_expert_ffn_shared_m(int qtype, const void* gate,
                                     const void* up, const void* down,
                                     const float* x, const int* ids_d,
                                     const float* weights_d, int n_tokens,
                                     int n_used, int n_expert, int k, int n_ff,
                                     float limit, float* y,
                                     hipStream_t stream = nullptr);

/// Read and clear the small-m path's status word. Blocking; once per verify
/// batch.
int superroc_fp4_expert_ffn_m_check(void);

// PHASE6: the warp-per-row variant of the plain matvec and of the grouped down
// projection. Both must agree BIT-EXACTLY with the CTA-per-row kernels above,
// which tools/superroc_fp4_gemv_q8_test.cpp checks, and
// SUPERROC_FP4_GEMV_VARIANT=1 makes the shipped entry points use them (default
// 0, so nothing changes unset).
int superroc_fp4_gemv_fp4_q8_rows(int qtype, const void* W, int slot, float* y,
                                  int k, int n, int block,
                                  hipStream_t stream = nullptr);
int superroc_fp4_expert_ffn_dev_tuned(int qtype, const void* gate,
                                      const void* up, const void* down,
                                      const float* x, const int* ids_d,
                                      const float* weights_d, int n_used,
                                      int n_expert, int k, int n_ff,
                                      float limit, float* y, int block,
                                      hipStream_t stream = nullptr);
int superroc_fp4_expert_ffn_dev_rows(int qtype, const void* down,
                                     const int* ids_d, const float* weights_d,
                                     int n_used, int n_expert, int k, int n_ff,
                                     float* y, int block,
                                     hipStream_t stream = nullptr);

/// Test-only: the small-m path's hs scratch.
int superroc_fp4_moe_hs_m_peek(float* out, int n);

/// Test-only: copy the grouped expert FFN's hs scratch back to the host.
int superroc_fp4_moe_hs_peek(float* out, int n);

/// Copy a slot's quantised activation back to the host (test-only diagnostic).
int superroc_fp4_q8_peek(int slot, int8_t* qs_out, float* ds_out, int k);

/// lqw:spec6. Pre-grow the small-m verify Q8 buffer (m * k floats). No-op once
/// the capacity is enough. Not safe to call inside a captured graph.
int superroc_fp4_verify_m_reserve(int total);

/// lqw:spec6. Matvec of n rows against m = 2..8 token-major activation rows.
/// y[t * n + row] matches m calls to superroc_fp4_gemv_fp4 bitwise. Returns
/// SUPERROC_FP4_ERR_ARG (no allocation) when the reserve is short.
int superroc_fp4_gemv_verify_m(int qtype, const void* W, const float* x,
                               float* y, int k, int n, int m,
                               hipStream_t stream = nullptr);

/// Super ROC QVO: superroc_fp4_gemv_verify_m without its input quantisation. It
/// reuses the Q8 copy the calling stream's set already holds and refuses
/// (SUPERROC_FP4_ERR_ARG) unless the last verify_m quantisation in that set was
/// of this same x, m and k. For the second and later members of a projection
/// group that share one input; bit-identical to superroc_fp4_gemv_verify_m.
int superroc_fp4_gemv_verify_m_xq(int qtype, const void* W, const float* x,
                                  float* y, int k, int n, int m,
                                  hipStream_t stream = nullptr);

/// Exhaustively check the FP4 nibble -> int8 perm codebook against
/// 2 * superroc_fp4_rocmfp4_code. Returns the number of disagreeing byte lanes
/// (0 = good), or -1 on a HIP failure.
int superroc_fp4_gemv_q8_codebook_check(void);

// ---------------------------------------------------------------------------
// SQ10: slot-set ownership. Every Q8/MoE/verify scratch buffer above exists
// once per SET, and streams attach to sets 1:1, so two decode streams can
// never share scratch -- except a session's helper (drafter) stream, which
// is ALIASED to its owner's set (SPEC2 R1). Existing entries keep their
// signatures and resolve the calling stream's set internally, attaching
// lazily; when no set is free they refuse at once with -SUPERROC_FP4_ERR_BUSY
// (SPEC2 R2) and the caller declines to its eager fallback -- nothing in
// production waits. At `-j 1` everything resolves to set 0 and is
// bit-identical.
// ---------------------------------------------------------------------------

/// SPEC2 ownership revision: alias + refuse-at-once + wait-only test entry.
/// The unit test ifdefs the new cases on this.
#define SUPERROC_FP4_SLOT_HAS_ALIAS 1

/// Ensure at least N slot sets exist (1..64; out of range clamps). Pre-sizing
/// for a server's sessions; grows only, so existing owners keep their
/// indices, and repeating with any N is a no-op. Never fails, never
/// renumbers: always returns SUPERROC_FP4_OK.
int superroc_fp4_slot_sets_configure(int n);

/// The configured set count (1 before any configure call).
int superroc_fp4_slot_sets_count(void);

/// Attach `stream` to a free set and return its index (>= 0). Idempotent for
/// an attached stream; an aliased helper resolves to its owner's set. When
/// no set is free, refuses at once with -SUPERROC_FP4_ERR_BUSY (one WARN per
/// stream on stderr, counted by superroc_fp4_slot_refuse_count); never waits.
/// Host-side only; legal inside capture.
int superroc_fp4_slot_stream_attach(hipStream_t stream);

/// SPEC3 (S3-5): attach that names the declining launch in the WARN
/// (`func`, e.g. __func__, and the tensor width, e.g. k) so a fallback is
/// diagnosable. Production attach sites use this; the plain attach labels
/// the WARN with its own name.
int superroc_fp4_slot_stream_attach_at(hipStream_t stream, const char* func,
                                       long width);

/// SQ19-2: load-time attach. Same resolution as attach_at, but with no free
/// set it grows the table by one set (up to 64) instead of refusing,
/// configured or not. The caller must warm the set's buffers before any
/// step (a first hipMalloc inside a captured decode step poisons the
/// capture); in-launch callers must keep using attach_at, which never
/// grows. Past the cap this refuses exactly like attach_at (same WARN,
/// same counter).
int superroc_fp4_slot_stream_attach_grow(hipStream_t stream, const char* func,
                                         long width);

/// The waiting attach. Unit test only: same resolution, but an unseen stream
/// with no free set blocks until a detach frees one. Production must never
/// call this (R2).
int superroc_fp4_slot_stream_attach_wait(hipStream_t stream);

/// Alias `helper` (a session's drafter/helper stream) to `owner`'s set.
/// The owner must already be attached; the helper must not own a set and the
/// owner must not itself be aliased. Idempotent for the same pair; returns
/// the owner's set, or SUPERROC_FP4_ERR_ARG. Register where the helper is
/// created, remove (via detach) where it is destroyed.
int superroc_fp4_slot_stream_alias(hipStream_t helper, hipStream_t owner);

/// Release `stream`'s set, waking one waiter, and remove its aliases in both
/// directions (as helper and as owner). Returns SUPERROC_FP4_ERR_ARG when the
/// stream is neither attached nor aliased. Call before destroying the
/// stream: a destroyed address that HIP later reuses would otherwise inherit
/// the old set (R3).
int superroc_fp4_slot_stream_detach(hipStream_t stream);

/// Lifetime count of slot hipMallocs (every set, plus SwiGLU pairs via
/// superroc_fp4_slot_count_alloc). Warmup moves it; decode must not.
unsigned long superroc_fp4_slot_alloc_count(void);

/// Count one allocation performed outside this TU (the SwiGLU pairs).
void superroc_fp4_slot_count_alloc(void);

/// Lifetime count of attaches that waited for a free set, and their total
/// wait in microseconds. Only the test's attach_wait entry waits; the gates
/// assert both are 0 in production (serve with SUPERROC_FP4_SLOT_TRACE=1 to log
/// each wait to stderr).
unsigned long superroc_fp4_slot_wait_count(void);
unsigned long superroc_fp4_slot_wait_us(void);

/// Lifetime count of at-once refusals (-SUPERROC_FP4_ERR_BUSY). The gates
/// assert this is 0; each refusal also logs one SUPERROC_FP4_SLOT_REFUSE line
/// per stream to stderr.
unsigned long superroc_fp4_slot_refuse_count(void);

/// Test-only: entries currently in the stream->set map and in the alias map.
int superroc_fp4_slot_debug_map_size(void);
int superroc_fp4_slot_debug_alias_size(void);

/// Pre-grow every set's five Q8 slots to k plus the verify pair to 8*k,
/// launching the same warm quant as before. Load-time, outside any capture.
int superroc_fp4_slots_warm_q8(int k, hipStream_t stream);

/// Pre-grow every set's single-token MoE hs (hs_elems floats), small-m MoE hs
/// (hs_m_elems) and m-pair (m_total). Zeros are a no-op (dense models).
int superroc_fp4_slots_warm_moe(int hs_elems, int hs_m_elems, int m_total);

/// Read and clear one set's single-token MoE status word (+ the registered
/// router word). The bare check() aggregates over every set for callers that
/// do not name one (Flash-Next, prefill, tools).
int superroc_fp4_moe_dev_check_set(int set);

/// SPEC2 (R6): stream-ordered check. The arena stream is non-blocking, so a
/// default-stream DtoH can read the word before this token's kernels ran
/// (proven on the box: STALE_STATUS missed 1/200 same-token checks); draining
/// the stream first makes the same token's check report its own error. The
/// decode path (never captured for MoE) uses this; callers that already
/// synced (Flash-Next) keep the bare check.
#define SUPERROC_FP4_SLOT_HAS_STREAM_CHECK 1
int superroc_fp4_moe_dev_check_set_on_stream(int set, hipStream_t stream);

/// Read and clear one set's small-m status word; the bare check aggregates.
int superroc_fp4_expert_ffn_m_check_set(int set);

/// SPEC3 (S3-2): stream-ordered small-m check, mirroring the R6 decode
/// wrapper: the prefill MoE kernels run on the non-blocking arena stream,
/// so the bare aggregate read can precede them and miss a bad-router-id
/// error (same stale class as R6). Prefill names its own set, which also
/// stops one session's prefill clearing another session's word.
#define SUPERROC_FP4_SLOT_HAS_STREAM_MCHECK 1
int superroc_fp4_expert_ffn_m_check_set_on_stream(int set, hipStream_t stream);

/// Test-only peeks for one set; the bare peeks read set 0.
int superroc_fp4_moe_hs_peek_set(int set, float* out, int n);
int superroc_fp4_moe_hs_m_peek_set(int set, float* out, int n);
int superroc_fp4_q8_peek_set(int set, int slot, int8_t* qs_out, float* ds_out,
                             int k);

#ifdef __cplusplus
}
#endif