// ling3_mla_fw.hpp — LX1 (wave 3): fused WMMA flash-attention for the
// absorbed-MLA PREFILL, the whole 512-token chunk in ONE kernel launch.
//
// Design (see the lane's STATUS.md): per (layer, chunk) the f32 cache is
// staged to f16 by mla_w2_prep (kernels/ling3_mla_w2.hip — SHARED buffers,
// one prep per layer per chunk, unchanged; it is the ONLY w2 piece used).
// Everything else is in this module (RV3 C2: w2's serial-dot pack/out were
// the latency bound at the whole-chunk row count, MEASURED 77/84 ms —
// replaced by f16 WMMA GEMMs at ~0.3 ms each):
//   fw_qtrans_kernel  stages q_nope [head][token][dim] + q16's rope tail
//   fw_pack_kernel    absorbs: q16[r][0:512] = f16(scale * wk_b . q_nope)
// Then, per layer per chunk:
//
//   S = q16 . k16^T        (WMMA f16->f32, one 16x16x16 tile per warp)
//   online softmax        (f32, in registers + one LDS pass, no atomics;
//                          STANDARD running max m_new = fmax(m_old, mx) —
//                          every rescale factor <= 1, RV17 H1)
//   O = P . V             (WMMA f16->f32, P and V read as f16 fragments)
//   fw_deabsorb_kernel out = sigmoid(gate) * wv_b . (O / rowsum)
//
// with NOTHING materialized in global memory (w2's materialized S/P16 tiles
// and its 24 launches per chunk become four launches total).
//
// One workgroup = ONE token's 32 query rows (its 32 heads) — a 32-row block
// with a UNIFORM causal limit pl = (n_pos - chunk_tokens) + t + 1. 256
// threads = 8 wave32 warps. Position loop in 64-position chunks: GEMM1's
// 8 C tiles (2 row tiles x 4 position tiles) cover the 32x64 score tile
// exactly once (no wave computes S redundantly); the score tile goes
// through LDS, the softmax runs one 8-lane group per row (f32, fixed order),
// P is an f16 [32][64] LDS tile; GEMM2 splits the 32x512 output accumulator
// across the 8 warps BY DIMENSION (warp w owns dims [64w, 64w+64) x all 32
// rows = 8 WMMA C fragments = 64 f32 registers per lane), so no wave ever
// holds more than 64 accumulator registers — LF1's parked WMMA kernel
// (REPORT2) died at 256.
//
// Numerics: f16 K/V and f16 P (the fork's precision class, same as tile),
// f32 online softmax, f32 O accumulation — but the ABSORB and DE-ABSORB run
// as f16 WMMA GEMMs (f16 operands, f32 accumulation; RV3 C2), and the sums
// are reordered (online softmax over 64-wide chunks, WMMA k-order). NOT
// bit-identical to any existing path; gate is max_rel <= 2e-3 vs
// mla_attn_tok plus determinism (same input -> identical output bytes; no
// atomics, fixed reduction order everywhere).
//
// Determinism: no atomics; every reduction (softmax butterflies, WMMA
// instruction, running max/sum update) has a fixed order.
//
// Bounds discipline (LF1's lf_audit model): every global index is enumerated
// in the header comment of ling3_mla_fw.hip and re-checked host-side by
// fw_audit() before any launch; the kernels only touch indices the audit
// proved in-bounds. Any refusal returns false and the caller falls back to
// the proven path for the chunk (fail-closed).

#ifndef LING3_MLA_FW_HPP
#define LING3_MLA_FW_HPP

#include <hip/hip_fp16.h>

#include "ling3_mla.hpp"
#include "ling3_mla_w2.hpp"  // MlaW2Bufs (k16/vt staging + mla_w2_prep)

namespace ling3 {

/// Query rows of one launch: one full prefill chunk (512 tokens x 32 heads).
constexpr int kMlaFwChunkTokens = 512;
constexpr int kMlaFwRows = kMlaFwChunkTokens * 32;  // 16,384

/// Caller-owned scratch for the fused path (rocm_backend's BatchBufs):
///   q16    [kMlaFwRows][kv_a()] f16 — absorbed, scaled queries
///          (row layout: r = local_token * n_head + h)
///   o      [kMlaFwRows][kv] f32 — unnormalised latent out of the flash
///          kernel; consumed by the de-absorb GEMM (divide by rowsum,
///          wv_b, head gate)
///   rowsum [kMlaFwRows] f32 — softmax denominators
///   qt     [n_head][512][nope] f32 — q_nope staged [head][token][dim] for
///          the pack GEMM's B fragments (8.4 MB)
struct MlaFwBufs {
  __half* q16 = nullptr;
  float* o = nullptr;
  float* rowsum = nullptr;
  float* qt = nullptr;
  int rows_cap = 0;  // >= tok_count * n_head of every launch
};

/// Fused WMMA flash attention for ONE chunk's [tok_base, tok_base+tok_count)
/// tokens. k16/vt must already be staged for n_pos by mla_w2_prep (the same
/// MlaW2Bufs the w2 path uses, so one prep serves both). Does absorb
/// (fw_pack_kernel) -> flash (fw_attn_kernel) -> de-absorb + gate
/// (fw_deabsorb_kernel) and returns false on ANY audit failure (shape
/// 32/192/128/64/512/576/head_v 128, chunk_tokens <= in.n_tokens, ...),
/// leaving the caller to run its proven fallback for the chunk.
/// tok_count <= 512.
bool mla_fw_attn(const MlaShape& sh, const MlaScalars& sc, const MlaWeights& w,
                 int n_pos, const MlaInputs& in, const View& out,
                 const MlaW2Bufs& w2bufs, MlaFwBufs& bufs, int tok_base,
                 int tok_count, int chunk_tokens, hipStream_t stream);

const char* mla_fw_version();  // "ling3-mla-fw/1.2-lx1" (1.2 = RV17 H1/H2/H3)

}  // namespace ling3

#endif  // LING3_MLA_FW_HPP
