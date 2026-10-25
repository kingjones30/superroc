// ling3_mla_w2.hpp — LW1 (wave 2): two-pass WMMA absorbed-MLA PREFILL
// attention.
//
// The batched prefill's attention (mla_attn_tok in ling3_mla.hip) computes the
// 576-wide score dot and the 512-wide latent accumulation on fp32 vector ALUs:
// 2,799 of ~3,930 ms per 512-token chunk at 11.5K context (LS4's profile), and
// that share grows linearly with position. This module computes the SAME
// arithmetic as two plain f16 GEMMs on the WMMA matrix cores with a
// materialized score tile — deliberately the simplest correct shape:
//
//   pass 1 (w2_scores):  S[r][p] = q16[r][0:576] . k16[p][0:576]   (f16 WMMA,
//   f32 accum) softmax  (w2_softmax): row max / exp / sum in f32; P[r][p] =
//   exp(S-m) in f16 pass 2 (w2_pv):      O[r][0:512] = P[r][0:p_hi] .
//   V[p][0:512]  (f16 WMMA, f32 accum) epilogue (w2_out):   latent = O[r][:] /
//   sum[r], then wv_b de-absorption and
//                        the head sigmoid gate — the same serial j-ascending
//                        statement order as attn_tok_kernel's de-absorb.
//
// Query rows r = local_t * n_head + h of one sub-batch, EXACTLY the score-row
// layout mla_attn_tok's caller already uses: `scores` is the caller's
// [kMlaW2Rows][n_pos] f32 scratch (rocm_backend's B->mla_scores).
//
// The sub-batch cap equals rocm_backend's kMlaTokBatchCap (64); the caller
// loops sub-batches exactly as it loops mla_attn_tok today. Every entry point
// returns false on ANY audit failure and the caller falls back to mla_attn_tok
// (fail-closed). The decode path (Step / split-K) is untouched.
//
// Buffers are caller-owned (rocm_backend's BatchBufs); MlaW2Bufs carries the
// position capacity `cap`, which must be a multiple of 64 and >= n_pos:
//   k16   [cap][kv_a()]        f16 cache staging, written by mla_w2_prep
//                              (pass 1's B operand, [pos][dim] as WMMA wants)
//   vt    [kv][cap]            f16 transposed latent, written by mla_w2_prep
//                              (pass 2's B operand, [dim][pos])
//   q16   [kMlaW2Rows][kv_a()] f16 packed, 1/sqrt(192)-scaled queries
//   p16   [kMlaW2Rows][cap]    f16 unnormalised softmax probabilities
//   o     [kMlaW2Rows][kv]     f32 unnormalised latent out
//   rowsum[kMlaW2Rows]         f32 softmax denominators
//
// k16/vt are refreshed from the f32 cache ONCE PER (layer, chunk) by
// mla_w2_prep, which must run after mla_cache_append of that chunk. Positions
// [n_pos, n_pos_pad) are written as explicit zeros so pass 1/2 fragment loads
// that round up to a 64-tile boundary never multiply uninitialised memory
// (DevPoison'ed allocations contain NaN patterns; 0 * NaN is NaN).

#ifndef LING3_MLA_W2_HPP
#define LING3_MLA_W2_HPP

#include <hip/hip_fp16.h>

#include "ling3_mla.hpp"

namespace ling3 {

/// Query rows of one sub-batch: kMlaW2Batch tokens * n_head. Must equal
/// rocm_backend's kMlaTokBatchCap * heads (64 * 32); the host audit enforces
/// tok_count <= kMlaW2Batch at run time.
constexpr int kMlaW2Batch = 64;
constexpr int kMlaW2Rows = kMlaW2Batch * 32;  // 2048

struct MlaW2Bufs {
  __half* k16 = nullptr;    // [cap][kv_a()]
  __half* vt = nullptr;     // [kv][cap]
  __half* q16 = nullptr;    // [kMlaW2Rows][kv_a()]
  __half* p16 = nullptr;    // [kMlaW2Rows][cap]
  float* o = nullptr;       // [kMlaW2Rows][kv]
  float* rowsum = nullptr;  // [kMlaW2Rows]
  int cap = 0;              // position capacity of k16/vt/p16 (mult of 64)
};

/// Convert the f32 cache rows [0, n_pos) into the f16 k16/vt staging buffers
/// (one launch, per layer per chunk, AFTER this chunk's mla_cache_append).
/// Returns false (and launches nothing further) if the audit fails; the caller
/// then runs its mla_attn_tok fallback for the whole chunk.
bool mla_w2_prep(const MlaShape& sh, const MlaCache& cache, int n_pos,
                 MlaW2Bufs& bufs, hipStream_t stream);

/// One sub-batch's attention, [tok_base, tok_base + tok_count) of a chunk of
/// chunk_tokens tokens ending at n_pos. All cache bytes it touches were
/// staged by mla_w2_prep, so the f32 cache is not an argument. `scores` is
/// the caller's f32 scratch, >= kMlaW2Rows * n_pos floats (rocm_backend's
/// B->mla_scores). Returns false on any audit failure; the caller falls back
/// for this sub-batch.
bool mla_w2_attn(const MlaShape& sh, const MlaScalars& sc, const MlaWeights& w,
                 int n_pos, const MlaInputs& in, const View& out, float* scores,
                 MlaW2Bufs& bufs, int tok_base, int tok_count, int chunk_tokens,
                 hipStream_t stream);

const char* mla_w2_version();  // "ling3-mla-w2/1.0-lw1"

}  // namespace ling3

#endif  // LING3_MLA_W2_HPP
