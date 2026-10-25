// ling3_mla_tile.hpp — LF1 (MiMo), wave 2: gufo's absorbed-MLA PREFILL
// attention as the fork's TILE kernel. 2026-10-03. Companion:
// ling3_mla_tile.hip.
//
// WHAT — a port of llama.cpp ROCmFPX's `flash_attn_tile<DKQ=576, DV=512,
// ncols1=2, ncols2=16>` (ggml/src/ggml-cuda/fattn-tile.cuh, MIT; attribution in
// the .hip) to gufo's absorbed MLA prefill, per Jay's DECISIONS 2026-10-03
// 01:05. The fork's kernel is the proven design on gfx1151: one launch per
// layer per chunk, 16 heads x 2 tokens per block sharing each K/V tile,
// f16 K/V in LDS (half2), Q f32->f16 pre-scaled, register-blocked KQ and VKQ
// dot2 products with NO shuffles in the inner loop, online softmax with one
// warp reduction per tile.
//
// gufo-specific shape of the same algorithm:
//   - the 32 query columns of a block are 2 tokens x 16 heads of ONE shared
//     cache (MQA): every column reads the same K/V tile, so the tile is staged
//     once and reused 32x;
//   - K = the f16 copy of the cache slice, made ONCE per layer per chunk by
//     cache_to_f16_kernel (SPEC2 2.6 measured this: converting per workgroup
//     in the staging loop costs ~90 ms of a 266 ms kernel at 8 K);
//   - causal limit per query column (token t sees [0, (n_pos-chunk)+t+1)),
//     replacing the fork's mask tensor;
//   - scale 1/sqrt(192) is applied to Q at staging (the fork does the same);
//   - attention outputs the 512-dim latent per (token, head), and the wv_b
//     de-absorb + sigmoid gate run as a SEPARATE batched GEMM kernel
//     (LR1 §6.8): one LDS copy of wv_b[h] per (head, r-tile) serves all
//     tokens, instead of ~4.2 GB of wv_b re-reads inside attention.
//
// ARITHMETIC — identical to mla_attn_tok up to f16 inputs and the reordered
// sums (f32 accumulation, f32 online softmax):
//   q576 = [wk_b[h] . q_nope | q_pe]  (512 + 64), pre-scaled by 1/sqrt(192)
//   score[t,h,p] = q576 . cache[p]              (576 dot2 MACs)
//   w = softmax over the causal window
//   latent[t,h] = sum_p w[p] . cache[p][0:512]  (512 dot2 MACs)
//   out[t,h,r]  = sigmoid(gate[t,h]) . sum_d wv_b[h][r][d] . latent[t,h,d]
//
// Selection: LING3_MLA_TILE=1 (default OFF) in ForwardBatch. Unsupported
// shapes fall back to mla_attn_tok from inside the launcher.

#ifndef LING3_MLA_TILE_HPP
#define LING3_MLA_TILE_HPP

#include "ling3_mla.hpp"

namespace ling3 {

bool mla_tile_supported(const MlaShape& sh, const MlaCache& cache);

// Same contract as mla_attn_tok (kernels/ling3_mla.hpp): attention over
// tok_base..tok_base+tok_count with the causal window
// [0, (n_pos - chunk_tokens) + t + 1). Runs the tile attention (one launch) and
// the de-absorb GEMM (one launch) for the whole window.
void mla_attn_tile(const MlaShape& sh, const MlaScalars& sc,
                   const MlaWeights& w, const MlaCache& cache, int n_pos,
                   const MlaInputs& in, const View& out, float* scores,
                   int tok_base, int tok_count, int chunk_tokens,
                   hipStream_t stream);

const char* mla_tile_version();  // "ling3-mla-tile/1.0"

}  // namespace ling3

#endif  // LING3_MLA_TILE_HPP
