// ling3_mla.hpp — gated-MLA attention block of a Ling-3.0 `bailingmoe3` layer.
// Lane: LK (MiMo) — Ling-on-superroc overnight build. Do not confuse with LG's
// drafts. Lane LK, phase 2 of the Ling-on-superroc build. Phase-1 companion:
// ling3_kda.hpp.
//
// The fork's code is the spec:
//   src/models/bailingmoe3.cpp   the gated-MLA branch (absorbed form, wk_b &&
//   wv_b) src/llama-graph.cpp          build_attn / build_attn_mha (how v_mla
//   is applied) ggml rope (GGML_ROPE_TYPE_NORM) for the rotary variant
// Cross-checked against LG's CPU implementation, which passes
// tools/mla_layer_test.cpp:
//   ~/src/port/gufo-ling/src/models/ling3/reference.cpp
//   ReferenceModel::MlaLayer
//
// SCOPE — the four projections are the CALLER's GEMMs; their outputs are this
// module's inputs:
//   wq          [2560 -> 6144]  -> q      (phase 1's rule: projections are the
//   caller's) wkv_a_mqa   [2560 -> 576]   -> kv_a wqkv_gate   [2560 -> 32] ->
//   gate   (sigmoid applied HERE) attn_output [4096 -> 2560]  -> applied by the
//   CALLER to `out`
// This module owns the rest of the block:
//   RoPE (NORM variant, 64 dims) on q_pe and k_pe · RMSNorm on the 512-dim
//   latent · the KV cache write · the wk_b absorption of q_nope · softmax
//   attention over the cache · the wv_b de-absorption · the head-wise sigmoid
//   output gate.
//
// `out` is the gated head output [n_head*n_embd_head_v_mla, n_tokens] = [4096,
// T], i.e. exactly what feeds attn_output.weight — the same boundary as phase
// 1's kda_forward.

#ifndef LING3_MLA_HPP
#define LING3_MLA_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

// Shared with phase 1: View / ViewC / make_view and the activation-stride
// convention.
#include "ling3_kda.hpp"

namespace ling3 {

// ---------------------------------------------------------------------------
// Geometry of one Ling-3.0-flash gated-MLA layer (GGUF bailingmoe3.*)
//   n_embd 2560 · n_head 32
//   attention.key_length_mla  192   (= qk_nope 128 + qk_rope 64)
//   attention.value_length_mla 128
//   attention.kv_lora_rank 512
//   attention.key_length 576   (= 512 latent + 64 rope key)
//   attention.value_length 512
//   rope.dimension_count 64 · rope.freq_base 6e6   (GGML_ROPE_TYPE_NORM)
//   attention.layer_norm_rms_epsilon 1e-6
// MLA blocks are every 6th block (5, 11, 17, 23, 29, 35, 41); the rest are KDA.
// ---------------------------------------------------------------------------
struct MlaShape {
  int n_embd = 2560;
  int n_head = 32;
  int head_k = 192;  // dk  = nope + rot
  int head_v = 128;  // dv
  int nope = 128;    // qk_nope_head_dim
  int rot = 64;      // qk_rope_head_dim
  int kv = 512;      // kv_lora_rank (the latent, and the value width)
  float rope_theta = 6000000.0f;

  int kv_a() const { return kv + rot; }          // 576, one cached position
  int q_dim() const { return head_k * n_head; }  // 6144
  int o_dim() const { return head_v * n_head; }  // 4096
};

struct MlaScalars {
  float rms_eps = 1e-6f;
  float kq_scale = 0.0f;  // 0 => 1/sqrt(head_k) = 1/sqrt(192)
};

// ---------------------------------------------------------------------------
// Weights (device pointers, caller-owned, must outlive the launch).
// ggml layout: ne[0] is the contiguous dimension.
//   wk_b : [nope, kv, n_head]   blk.N.attn_k_b.weight  [128, 512, 32]
//          wk_b[(h*kv + r)*nope + i]  i = qk_nope dim, r = latent dim, h = head
//   wv_b : [kv, head_v, n_head] blk.N.attn_v_b.weight  [512, 128, 32]
//          wv_b[(h*head_v + r)*kv + j]  j = latent dim, r = output dim, h =
//          head
//   kv_a_norm : [kv]            blk.N.attn_kv_a_norm.weight (RMSNorm gamma on
//   the latent)
// ---------------------------------------------------------------------------
struct MlaWeights {
  const float* wk_b = nullptr;
  const float* wv_b = nullptr;
  const float* kv_a_norm = nullptr;
  // SPEC6: optional row-transposed copies for the DECODE path only.
  //   wk_bT[h][i][r] == wk_b[h][r][i]      (n_head, nope, kv)
  //   wv_bT[h][j][r] == wv_b[h][r][j]      (n_head, kv, head_v)
  // They exist so attn_sk_prep_kernel / attn_sk_reduce_kernel can read one
  // column of the matrix per i step -- consecutive threads, consecutive
  // addresses -- instead of one 512-B row each. nullptr keeps the original
  // layout, which the prefill kernels use untouched.
  const float* wk_bT = nullptr;
  const float* wv_bT = nullptr;
};

/// Transpose each head of a [n_head][rows][cols] row-major matrix into
/// [n_head][cols][rows] (the layout the decode kernels read). Pure copy of
/// values: no arithmetic, so the bits are exactly the source's.
void mla_weights_transpose(const float* src, float* dst, int n_head, int rows,
                           int cols, hipStream_t stream);

// ---------------------------------------------------------------------------
// KV cache. One row of `kv_a()` F32 per (sequence, position):
//     kv[s_seq + p*s_pos + i] ,  i in [0, kv)      = RMSNorm'd latent  (the
//     VALUE)
//                                    i in [kv, kv_a) = RoPE'd k_pe     (shared
//                                    key part)
// `n_pos` is the number of VALID positions including the tokens of this call,
// i.e. this call appends its n_tokens at positions [n_pos - n_tokens, n_pos).
// All sequences in one call must have the same n_pos. The cache is the
// caller's; this module only writes the new positions.
// ---------------------------------------------------------------------------
struct MlaCache {
  float* kv = nullptr;
  int64_t s_pos = 0;  // elements between positions
  int64_t s_seq = 0;  // elements between sequences
  int max_pos = 0;
};

// ---------------------------------------------------------------------------
// Inputs — the caller's GEMM outputs, before this module's elementwise work.
//   q     : [head_k, n_head, n_tokens]  = [6144, T] contiguous
//           head-major; within a head dims [0, nope) are non-rotated and
//           dims [nope, head_k) are the RoPE part
//   kv_a  : [kv_a, n_tokens] = [576, T]  raw wkv_a_mqa output
//           [0, kv) the latent, [kv, kv_a) the key rope part
//   gate  : [n_head, n_tokens]  raw wqkv_gate output (sigmoid applied here)
// ---------------------------------------------------------------------------
struct MlaInputs {
  ViewC q, kv_a, gate;
  int n_tokens = 0;
  int n_seqs = 1;
};

// ---------------------------------------------------------------------------
// The pieces (each independently callable so superroc can schedule/fuse).
// ---------------------------------------------------------------------------

// RoPE, GGML_ROPE_TYPE_NORM, applied over `rot` dims of one head's rope
// section:
//     half = rot/2; for i in [0, half):
//       freq  = 1 / theta^(2i/rot)
//       angle = pos * freq
//       (x[i], x[i+half]) = (x[i]*cos - x[i+half]*sin, x[i]*sin +
//       x[i+half]*cos)
// `x` points at the rope section's first element. `s_head` / `n_head` cover the
// head axis (for the shared key part use n_head 1 and s_head 0), `s_tok` /
// `n_tokens` the token axis and `s_seq` / `n_seqs` the sequence axis.
void mla_rope(const MlaShape& sh, float* x, int64_t s_head, int n_head,
              int n_tokens, int64_t s_tok, int64_t s_seq, int n_seqs,
              const int* pos, hipStream_t stream);

// RMSNorm the 512-dim latent of every token, in place: x * rsqrt(mean(x*x) +
// eps) * w
void mla_latent_norm(const MlaShape& sh, const MlaScalars& sc,
                     const MlaWeights& w, float* kv_a_inout, int n_tokens,
                     int n_seqs, int64_t s_tok, int64_t s_seq,
                     hipStream_t stream);

// Append the n_tokens of `kv_a_inout` ([kv_a, T], latent already RMSNorm'd and
// k_pe already rotated) to the cache at positions [n_pos - n_tokens, n_pos).
void mla_cache_append(const MlaShape& sh, const MlaCache& cache,
                      const float* kv_a_in, int n_tokens, int64_t s_tok,
                      int n_seqs, int n_pos, hipStream_t stream,
                      const int* pos_d = nullptr);

// ---------------------------------------------------------------------------
// LC1 SPEC4 (RV25 M1 + Q2-MLA): the decode fusions.
// ---------------------------------------------------------------------------

/// M1: rope(q) + rope(kv) + latent_norm + cache_append in ONE kernel (-3
/// launches per MLA layer). Same per-element expressions as the four, so the
/// result is bit-identical; both MLA arms and MTP verify's per-row calls go
/// through it. `pos` is a DEVICE pointer (mla_rope's contract). `pos_d` is the
/// optional single device position cache_append derives `npos` from -- pass the
/// same value the unfused call site passed (split-K passes none, the serial arm
/// passes its pos_d), so `npos` cannot move.
void mla_pre_fused(const MlaShape& sh, const MlaScalars& sc,
                   const MlaWeights& w, const MlaCache& cache,
                   const MlaInputs& in, const int* pos, int n_pos,
                   const int* pos_d, hipStream_t stream);

/// LING3_MLA_DECODE_FUSED (default 1), read once at first use. Off restores the
/// four-launch pre sequence and the unfused wo quant+GEMV exactly.
bool mla_decode_fused();

/// The wo Q8_1 pair the Q2-MLA epilogue writes and the wo GEMV reads
/// (qs[o_dim] int8 + ds[o_dim/32] float, superroc's internal layout). Allocated
/// on first use; `mla_wo_q8_qs()` stays null until then or if allocation
/// failed, which is the caller's signal to fall back to quantising `out`
/// itself.
bool mla_wo_q8_prepare(int o_dim);
const std::int8_t* mla_wo_q8_qs();
const float* mla_wo_q8_ds();

/// Launch counters, so a unit test can assert the fused path actually ran.
int mla_pre_fused_launches();
int mla_wo_q8_epilogues();
void mla_fusion_stats_reset();

// Attention + absorption + de-absorption + head gate.
//   q        [head_k, n_head, T]  raw (rope part already rotated in place)
//   gate     [n_head, T]          raw
//   out      [head_v*n_head, T]   = [4096, T] gated, pre-`attn_output`
// Reads the first `n_pos` cache positions of every sequence.
// `scores` is caller-owned scratch of mla_block_scratch_bytes() floats.
// `wo_qs` / `wo_ds` (LC1 SPEC4 Q2-MLA) are the Q8_1 activation the wo GEMV
// reads: when non-null and the call is one token of one sequence, the kernel
// quantises `out` into them as its epilogue (k_q8_1_quant's recipe verbatim)
// and the caller skips its own quantise. nullptr keeps the old kernel exactly.
void mla_attn(const MlaShape& sh, const MlaScalars& sc, const MlaWeights& w,
              const MlaCache& cache, int n_pos, const MlaInputs& in,
              const View& out, float* scores, hipStream_t stream,
              const int* pos_d = nullptr, std::int8_t* wo_qs = nullptr,
              float* wo_ds = nullptr);

// Scratch bytes `mla_block` needs: one softmax score row per (head, sequence)
// of n_pos floats.
size_t mla_block_scratch_bytes(const MlaShape& sh, int n_seqs, int n_pos);

// ---------------------------------------------------------------------------
// Umbrella: the whole gated-MLA block.
//   1. RoPE on q's rope section and on kv_a's key rope section
//   2. RMSNorm the latent
//   3. append the new positions to the cache
//   4. absorb q_nope, attend over the cache, de-absorb through wv_b, apply the
//   head gate
// `out` is [head_v*n_head, n_tokens]; the caller applies attn_output.weight.
// ---------------------------------------------------------------------------
void mla_block(const MlaShape& sh, const MlaScalars& sc, const MlaWeights& w,
               const MlaCache& cache, const MlaInputs& in, const int* pos,
               int n_pos, const View& out, hipStream_t stream,
               const int* pos_d = nullptr);

// core: identical, but with caller-owned softmax-score scratch of
// mla_block_scratch_bytes(sh, n_seqs, n_pos) floats
void mla_block(const MlaShape& sh, const MlaScalars& sc, const MlaWeights& w,
               const MlaCache& cache, const MlaInputs& in, const int* pos,
               int n_pos, const View& out, float* scores, hipStream_t stream,
               const int* pos_d = nullptr);

// LS3 2026-10-01: token-parallel attention for batched prefill. mla_attn's
// grid is (n_head, n_seqs) with a serial token loop inside, so a 512-token
// chunk runs on 32 blocks and takes ~1.5 s as the context grows. This
// variant puts the token axis in the grid -- one block per (head, token of
// [tok_base, tok_base + tok_count)) -- so the whole GPU fills; the caller
// sub-batches tok_count (e.g. 64) to bound the score scratch, which must
// hold tok_count * n_head * n_pos floats laid out [(t - tok_base) * n_head + h]
// * n_pos. chunk_tokens is the WHOLE chunk's token count (the causal base is
// n_pos - chunk_tokens); n_pos covers the chunk. Same arithmetic as mla_attn,
// per token; everything else in the block (RoPE / latent norm / cache
// append) is already token-parallel.
void mla_attn_tok(const MlaShape& sh, const MlaScalars& sc, const MlaWeights& w,
                  const MlaCache& cache, int n_pos, const MlaInputs& in,
                  const View& out, float* scores, int tok_base, int tok_count,
                  int chunk_tokens, hipStream_t stream);

void mla_block_scratch(const MlaShape& sh, const MlaScalars& sc,
                       const MlaWeights& w, const MlaCache& cache,
                       const MlaInputs& in, const int* pos, int n_pos,
                       const View& out, void* scratch, size_t scratch_bytes,
                       hipStream_t stream, const int* pos_d = nullptr);

// ---------------------------------------------------------------------------
// LS3 PHASE4 2026-10-01: split-K ("flash decoding") attention for the DECODE
// step (n_tokens == 1). mla_attn runs one block per head (32 blocks) and
// walks the whole cache serially, so decode slows as the context grows
// (36 tok/s at 50 positions -> 22 at 6.5K). Here the cache positions are
// partitioned into `slice`-position slices across workgroups: each computes
// its slice's (max, sum, weighted latent) partial, and one reduce kernel
// combines the partials in FIXED ascending slice order -- no atomics, so the
// result is deterministic run-to-run. Same arithmetic per position as
// mla_attn (q_abs absorption, causal window = n_pos, j-ascending latent sum).
//
// Scratch, caller-owned and PREALLOCATED FOR max_ctx (no malloc or sync in
// the step, per LS2's HIP-graph wrapping):
//   part_m / part_l : [n_head][k_slices] floats, k_slices = max_ctx / slice
//   part_acc        : [n_head][k_slices][kv] floats
// mla_splitk_bytes() sizes the whole set; slice is 1024 unless overridden.
// ---------------------------------------------------------------------------
constexpr int kMlaSplitkSlice = 1024;

/// P7: partial-slot cap. The partial buffers are sized for THIS many slices
/// regardless of the slice length, so an adaptive (short) slice cannot
/// over-run them; the callers clamp the launch to k_slices <= this.
constexpr int kMlaSplitkMaxSlices = 192;

inline size_t mla_splitk_slices(std::uint32_t max_ctx,
                                int slice = kMlaSplitkSlice) {
  return (size_t)((max_ctx + (std::uint32_t)slice - 1) / (std::uint32_t)slice);
}

/// P7: partial-buffer size -- fixed slots, independent of the slice length.
inline size_t mla_splitk_bytes(const MlaShape& sh) {
  return ((size_t)2 + (size_t)sh.kv) * (size_t)sh.n_head *
         (size_t)kMlaSplitkMaxSlices * sizeof(float);
}

/// P7: adaptive slice length. One workgroup per slice means a 6.5K context
/// at slice 1024 launches 7 workgroups on a 40-CU GPU; picking the slice so
/// the grid is ~target_wg workgroups fills the box at every depth. Clamped
/// to [64, 1024] and to kMlaSplitkMaxSlices slices; a pure function of
/// n_pos and target_wg, so a given context always splits identically
/// (determinism).
inline int mla_splitk_slice_auto(int n_pos, int target_wg) {
  if (n_pos <= 0 || target_wg <= 0) {
    return kMlaSplitkSlice;
  }
  int slice = ((n_pos + target_wg - 1) / target_wg + 63) & ~63;
  if (slice < 64)
    slice = 64;
  if (slice > kMlaSplitkSlice)
    slice = kMlaSplitkSlice;
  while ((n_pos + slice - 1) / slice > kMlaSplitkMaxSlices &&
         slice < kMlaSplitkSlice) {
    slice *= 2;  // forced-short slice at huge n_pos: clamp up to fit the slots
  }
  return slice;
}

// `wo_qs` / `wo_ds` (LC1 SPEC4 Q2-MLA): see mla_attn -- the reduce kernel
// quantises `out` into this pair as its epilogue when they are non-null.
void mla_attn_splitk(
    const MlaShape& sh, const MlaScalars& sc, const MlaWeights& w,
    const MlaCache& cache, int n_pos, const MlaInputs& in, const View& out,
    float* q_abs,  // [n_head][kv], the hoisted absorbed q
    float* part_m, float* part_l, float* part_acc, int slice_cap,
    int head_shared,  // LS3 P5: 1 = one workgroup per (8-head group,
                      // slice) so each cache slice is read once per
                      // group instead of once per head
    hipStream_t stream, std::int8_t* wo_qs = nullptr, float* wo_ds = nullptr);

// The whole decode block (RoPE, latent norm, cache append, split-K attention).
// `pos` is a DEVICE pointer, exactly like mla_block's.
void mla_block_splitk(const MlaShape& sh, const MlaScalars& sc,
                      const MlaWeights& w, const MlaCache& cache,
                      const MlaInputs& in, const int* pos, int n_pos,
                      const View& out, float* q_abs, float* part_m,
                      float* part_l, float* part_acc, int slice_cap,
                      int head_shared, hipStream_t stream);

// LS3 PHASE7: the m-row split-K decode attention for LM1's multi-row verify.
// m query rows (2 <= m <= 4) at CONSECUTIVE positions n_pos-m .. n_pos-1,
// already roped / latent-normed / appended to the cache by the caller (the
// shared block stages above handle n_tokens = m); row j attends causally to
// its own n_pos - m + j + 1 positions. Each (slice, row) workgroup is the
// register-fused single-row kernel with a per-row causal limit; the partials
// are indexed [row][head][slice] and the reduce combines them per row in the
// same fixed ascending order. in.n_tokens must be m; out is [o_dim, m].
// head_shared must be 2 (the register-fused variant; other shapes abort).
// Partials must hold m * kMlaSplitkMaxSlices slices per head (q_abs must
// hold m * n_head * kv floats); the launch is clamped to
// kMlaSplitkMaxSlices slices, so an m-row-sized buffer is never overrun.
void mla_attn_splitk_m(const MlaShape& sh, const MlaScalars& sc,
                       const MlaWeights& w, const MlaCache& cache, int n_pos,
                       const MlaInputs& in, const View& out, int m,
                       float* q_abs, float* part_m, float* part_l,
                       float* part_acc, int slice_cap, int head_shared,
                       hipStream_t stream);

// Decode convenience: n_tokens == 1, n_pos is the length AFTER this step.
//
// `pos` is a DEVICE pointer to one int32, exactly like mla_block's -- NOT a
// host value. It is read by the RoPE kernel, so it must live in device memory.
// (Taking the address of a host int here and passing that down faults the GPU
// with GCVM_L2_PROTECTION_FAULT; this signature makes that a compile error
// instead.)
void mla_block_decode(const MlaShape& sh, const MlaScalars& sc,
                      const MlaWeights& w, const MlaCache& cache,
                      const MlaInputs& in, const int* pos, int n_pos,
                      const View& out, hipStream_t stream);

// LVV: null keeps every rope angle equal to pos[t] * freq. A device table
// of (t, h, w) per physical token selects sections [8, 12, 12].
void mla_set_mrope(const int* device_triples_or_null);

const char* mla_version();  // "ling3-mla/1.0"

}  // namespace ling3

#endif  // LING3_MLA_HPP
