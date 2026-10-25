#pragma once
// ling3_gemm.hpp — batched-prefill GEMM + batched MoE router for
// Ling-3.0-flash. Lane LS3 (GLM 5.3), Ling speed wave, 2026-10-01. ADDITIVE to
// LX's ling3_moe.hip: nothing in that file (or any other lane's file) is
// modified here.
//
// Why this exists (H1 REPORT7 §3/§5): LX's ling3_gemm is
// one-thread-per-output-row with a serial k dot and the naive per-element FP4
// decode, so batching the projections through it was 5x SLOWER than the scalar
// path, and it was never validated at m > 1. Every projection of a batched
// prefill chunk goes through THIS kernel instead; the single-token Step() path
// keeps using ling3_gemv.
//
// Layout contract (identical to ling3_moe.hpp, so the two never transpose):
//   * weights: ggml rows, ne[0] = K contiguous, row r of a matrix at
//     r * ling3_row_bytes(qtype, K); an expert stack stores expert e at
//     e * expert_stride bytes (caller passes n_rows * row_bytes).
//   * activations x and outputs y are F32 token-major: x[t*K + i], y[t*N + r].
//   * every device allocation of weights/activations needs LING3_TAIL_PAD
//     readable bytes past the last element (same rule as ling3_moe.hpp).
//
// Grouping contract for ling3_gemm_grp: n_groups independent (expert,
// token-set) groups are computed in ONE launch. Group g covers slots j in
// [grp_off[g], grp_off[g] + grp_cnt[g]) of the parallel slot arrays:
//   x row of slot j : x_rows ? x_rows[j] : (j - grp_off[g])   <- null = LOCAL
//   0..cnt-1 y row of slot j : y_rows ? y_rows[j] : (j - grp_off[g]) y scale of
//   slot j: y_scale ? y_scale[j] : 1.0f weight rows    : W + grp_expert[g] *
//   expert_stride
// "null = local index" is what lets the same kernel serve both sides of an
// expert FFN: gate/up read the gathered token rows (global slot array) and
// write local scratch rows, while down reads local h rows and scatters back
// into y with the router weights (global slot array).
//
// All device entry points launch on the NULL stream (like ling3_moe.hip) and
// return LING3_OK / an LING3_ERR_* code, checking hipGetLastError.

#include <stddef.h>

#include "ling3_quant.hpp"
#include "ling3_route.hpp"  // ling3_pow2_ge / ling3_route_n{float,int} for the router

#ifdef __cplusplus
extern "C" {
#endif

/// y[yr[c]*n + r] (+)= sc[c] * dot(W_row_r, x[xr[c]]) for every group g, c in
/// [0, grp_cnt[g]), r in [0, n). accumulate != 0 adds into y instead of
/// storing (y must be zeroed by the caller in that case; each (r, slot) pair
/// is written by exactly one block, so no atomics are needed).
///
/// P3 step 1 (MoE-aware grid): with grp_zp (device [n_groups+1] prefix of
/// ceil(cnt/LS3_TC_GRP) 16-token tiles, grp_zp[0]=0) and total_tiles ==
/// grp_zp[n_groups], the token-tile axis is flattened through that prefix so
/// only the blocks the per-group counts imply are launched. Bit-identical to
/// the legacy grid; grp_zp == nullptr keeps the old 3-D worst-case grid.
int ling3_gemm_grp(int qtype, const void* W, size_t expert_stride,
                   const int* grp_expert, const int* grp_off,
                   const int* grp_cnt, int n_groups, const float* x,
                   const int* x_rows, float* y, const int* y_rows,
                   const float* y_scale, int accumulate, int k, int n, int m,
                   const int* grp_zp, int total_tiles);

/// Dense convenience: y[t*n + r] = dot(W_row_r, x_t), t in [0, m). Same math
/// and layout as LX's ling3_gemm, on the fast tiled kernel.
int ling3_gemm_fast(int qtype, const void* W, const float* x, float* y, int k,
                    int n, int m);

/// P2-C: token-major gather-sum of the slot-major routed-expert partials:
/// y[t*K + r] = sum_{s < n_used} partials[slot[t*n_used + s]*K + r].
/// Replaces the atomicAdd scatter (deterministic, slot order, like the CPU
/// reference). partials and y are [slots]/[tokens] rows of K floats.
int ling3_slot_gather_sum(const float* partials, const int* slot, float* y,
                          int n_used, int k, int n_tokens);

/// The bailingmoe3 router for T tokens in two launches + one logits pass.
/// ids/weights are DEVICE [n_tokens * n_used], token-major (ids[t*n_used + s]),
/// exactly what ling3_route writes per token. rc_out (device [n_tokens]) gets
/// LING3_OK or the per-token error; the caller reads it back with the ids.
int ling3_route_batch(const float* x, const float* gate, const float* bias,
                      int k, int n_tokens, int n_expert, int n_used,
                      int n_groups, int n_group_used, float w_scale, int norm_w,
                      int* ids, float* weights, int* rc_out);

// Test-only. Both buffers are device [n_tokens * n_expert]. n_tokens is 1..4.
int ling3_route_logits_pair(const float* x, const float* gate, int k,
                            int n_expert, int n_tokens, float* logits_ref,
                            float* logits_fused);

const char* ling3_gemm_version(void);

#ifdef __cplusplus
}
#endif
