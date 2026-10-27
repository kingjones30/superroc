#pragma once
// superroc_fp4_wmma.hpp — LS3 phase 3: W8A8 int8-WMMA GEMM for the ROCmFP4 prefill.
//
// The phase-1/2 tiled kernel (superroc_fp4_gemm.hip) dequantises FP4 weights to
// float in LDS and does scalar float FMA: ~5-6 % of the box's fp32 peak, while
// the same silicon demonstrates 55 TOPS on the int8 WMMA matrix core (superroc's
// own Qwen W8A8 prefill, and the fork's ROCmFP4 prefill, which is W8A8 too: FP4
// nibbles -> int8 in registers, Q8_1 activations, one int32 dot per
// 16x16x16 tile, block scales applied in the epilogue).
//
// This file is the Ling instantiation of that design, built from superroc's
// WKQuantA8BlockedWmmaGEMMKernel skeleton (src/models/qwen/hip/kernels/
// prefill_quant_gemm.hpp) and the fork's FP4->int8 decode
// (ggml/rocmfp4/rocmfp4_hip_codebook.cuh). It is a NEW file in the LS3 zone:
// no qwen file, no other lane's file is touched.
//
// Layouts
//   * weights: ggml rows exactly as superroc_fp4_moe.hpp/superroc_fp4_gemm.hpp see them
//   --
//     row r at r*superroc_fp4_row_bytes(qtype,k), an expert stack at expert_stride.
//     Only SUPERROC_FP4_Q4_0_ROCMFP4 / _FAST (types 100/101) is implemented; every
//     weight matrix Ling projects through in prefill is one of these.
//   * activations: PRE-QUANTISED to the tiled Q8_1 layout of superroc's kernel
//     (kQ8ActTileBytes = 576-byte tiles: [b0: 16 tokens x 16 bytes]
//     [b1: same, k-half 1] [16 fp32 token scales]), by superroc_fp4_wmma_quant.
//     The tile ORDER for the grouped kernel is GROUP-LOCAL: slot j of group g
//     (local l = j - grp_off[g]) lives at tile dest = grp_zp[g] + l/16,
//     lane l % 16, so a 16-token WMMA tile never spans two experts and an
//     expert's weight stack is re-read once per ceil(cnt/16) tiles, not once
//     per arbitrary-alignment tile. For dense calls dest[r] = r.
//   * outputs: F32 slot/token-major y[row_tokens * n + r], same as the
//     existing grouped contract with y_rows = identity.
//
// Determinism: the only reductions are the WMMA instruction's own fixed
// int32 sum and one k-ascending float += per 32-block per output element.
// No atomics anywhere; same binary twice -> bit-identical output.
//
// Bounds discipline (this lane faults no GPUs):
//   * weight reads: the fetch clamps row to n-1 and k-block to num_kb-1 and
//     zeroes the block scale by a multiply (superroc opt-c179: bit-identical to
//     zero-filling, and it keeps every load inside the row's blocks).
//     The 16-byte qs load is a memcpy from a 17/18-byte block: it can never
//     leave the block, and scales are read at block bytes 16/17.
//   * activation reads: every tile the grid addresses is a tile the
//     quantiser wrote (the grid is exactly ceil(cnt/16) tiles per group);
//     padding lanes inside a tile are never written by the epilogue.
//   * y writes: (slot, row) with row < n and slot inside its group only.

#include <stddef.h>

#include "superroc_fp4_quant.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/// Quantise `n_rows` gathered rows of the F32 activation x into the tiled
/// Q8_1 layout. Row r reads x[x_rows ? x_rows[r] : r] * k floats and lands at
/// tile dest[r] >> 4, lane dest[r] & 15 (dest == nullptr -> identity, which
/// is also the correct dense mapping). One warp per (row, 32-block); the
/// scale is max-abs/127 with roundf, superroc's QuantizeActivationToQ8_1Kernel
/// verbatim.
int superroc_fp4_wmma_quant(const float* x, const int* x_rows, const int* dest,
                        void* q8, int n_rows, int k);

/// Grouped W8A8 WMMA GEMM over the phase-2 grouping arrays (same contract as
/// superroc_fp4_gemm_grp, y_rows = identity): for every group g, local token
/// c in [0, grp_cnt[g]) -- global slot j = grp_off[g] + c --
///   y[j * n + r] = y_scale ? y_scale[j] : 1  *  sum_k W[expert_g][r][k] *
///   q8_x[j][k]
/// with the W8A8 arithmetic above. grp_zp is the device prefix array of
/// 16-token tiles: grp_zp[g] .. grp_zp[g+1] are g's ACTIVATION tiles and the
/// launch grid is exactly those tiles (only the blocks the counts imply).
/// total_tiles must equal grp_zp[n_groups] (the host builds the prefix, so
/// it already knows it). n must be <= the allocated rows; k a multiple of 32.
int superroc_fp4_wmma_gemm_grp(int qtype, const void* W, size_t expert_stride,
                           const int* grp_expert, const int* grp_off,
                           const int* grp_cnt, const int* grp_zp, int n_groups,
                           int total_tiles, const void* q8_x, float* y,
                           const float* y_scale, int k, int n);

/// Same contract as superroc_fp4_wmma_gemm_grp, with the K-stage BK chosen by the
/// caller (2, 4, or 8). Does not read SUPERROC_FP4_MOE_V2, so a KAT call at BK=4
/// leaves Ling's process-wide default at BK=2.
int superroc_fp4_wmma_gemm_grp_bk(int qtype, const void* W, size_t expert_stride,
                              const int* grp_expert, const int* grp_off,
                              const int* grp_cnt, const int* grp_zp,
                              int n_groups, int total_tiles, const void* q8_x,
                              float* y, const float* y_scale, int k, int n,
                              int bk);

/// Dense W8A8 WMMA GEMM: y[t * n + r] = dot(W_row_r, x_t) for t in [0, m),
/// x pre-quantised by superroc_fp4_wmma_quant(x, nullptr, nullptr, q8_x, m, k).
/// MG64: 64-token-block grouped GEMM: same contract as
/// superroc_fp4_wmma_gemm_grp_bk(..., 4), but each block covers BN = 64 tokens
/// indexed through the grp_zb block prefix (grp_zb[n_groups] = total blocks,
/// built by superroc_fp4_moe_group_blocks). total_tiles must be >= the total
/// block count; extra blocks exit in the kernel. Bit-identical to the
/// 16-token path.
int superroc_fp4_wmma_gemm_grp64(int qtype, const void* W, size_t expert_stride,
                             const int* grp_expert, const int* grp_off,
                             const int* grp_cnt, const int* grp_zp,
                             int n_groups, int total_tiles, const void* q8_x,
                             float* y, const float* y_scale, int k, int n,
                             const int* grp_zb);

/// KPF2: device-sized grouped W8A8 WMMA GEMM. Same contract as
/// superroc_fp4_wmma_gemm_grp_bk, but n_groups/total_tiles are read from the
/// 32k grouper's hdr_dev (4 device words) inside the kernel; the grid is
/// dim3(row_tiles, max_tiles) with max_tiles = GroupQ8Tiles(slots,
/// n_expert), and a block with blockIdx.y >= hdr_dev[TILES] returns at
/// once. A non-OK hdr_dev[STATUS] makes every block return at once.
int superroc_fp4_wmma_gemm_grp_bk_dev(int qtype, const void* W,
                                  size_t expert_stride, const int* grp_expert,
                                  const int* grp_off, const int* grp_cnt,
                                  const int* grp_zp, const int* hdr_dev,
                                  int max_tiles, const void* q8_x, float* y,
                                  const float* y_scale, int k, int n, int bk);

/// KPF2: device-sized MG64 grouped GEMM. Same contract as
/// superroc_fp4_wmma_gemm_grp64 with hdr_dev/max_tiles in place of
/// n_groups/total_tiles (grp_zb still passed; built by
/// superroc_fp4_moe_group_blocks_dev).
int superroc_fp4_wmma_gemm_grp64_dev(int qtype, const void* W, size_t expert_stride,
                                 const int* grp_expert, const int* grp_off,
                                 const int* grp_cnt, const int* grp_zp,
                                 const int* hdr_dev, int max_tiles,
                                 const void* q8_x, float* y,
                                 const float* y_scale, int k, int n,
                                 const int* grp_zb);

int superroc_fp4_wmma_gemm_dense(int qtype, const void* W, const void* q8_x,
                             float* y, int k, int n, int m);

/// On-device exhaustive check of the FP4->int8 perm codebook and the ue4m3
/// half scales against superroc_fp4_rocmfp4_code / superroc_fp4_ue4m3_half: returns 0
/// when every nibble of every 32-bit word decodes exactly like the scalar path.
int superroc_fp4_wmma_codebook_check(void);

const char* superroc_fp4_wmma_version(void);

#ifdef __cplusplus
}
#endif
