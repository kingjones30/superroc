#pragma once
#include <hip/hip_runtime.h>
// LS1 PHASE8 item 2. See superroc_fp4_moe_group.hip for what this replaces and
// why it is ordered the way it is.
/// Tokens per CTA tile in the tiled router logits kernel.
enum { SUPERROC_FP4_LS1_ROUTE_TY = 8 };

// LS1 PHASE9: the smallest chunk, in tokens, at which the 2D register-tiled
// router GEMM is used. Its grid is ceil(E/BM) x ceil(T/BN), so it needs enough
// tokens to fill the 20 CUs; below this the kernel it replaces runs unchanged.
// Overridable at runtime with SUPERROC_FP4_ROUTE_GEMM_MIN_T so the crossover
// can be re-measured without a rebuild.
enum { SUPERROC_FP4_LS1_ROUTE_GEMM_MIN_T = 256 };

#ifdef __cplusplus
extern "C" {
#endif

/// Tiled router logits: one CTA per (expert, SUPERROC_FP4_LS1_ROUTE_TY tokens)
/// instead of one per (expert, token) PAIR. BIT-IDENTICAL to
/// k_route_batch_logits -- same 64-lane strided partials in the same order,
/// same binary tree -- so the router's block-2 near-tie cannot move. `ty` is
/// clamped to SUPERROC_FP4_LS1_ROUTE_TY on the host.
int superroc_fp4_route_logits_tiled(const float* x, const float* gate, int k,
                                    int n_expert, int n_tokens, int ty,
                                    float* logits);

/// LV1 router logits: logits[t][e] = sum_i x[t][i] * gate[e][i]. One CTA owns
/// 64 experts x 64 tokens, 128 threads, each an 8x4 register tile, K stepped by
/// 32 in float4. fp32 accumulate, k ascending, no atomics, one writer per
/// output. Replaces the PHASE9 64x32 scalar tile. Returns SUPERROC_FP4_ERR_ARG
/// unless K is a multiple of 32.
int superroc_fp4_route_logits_gemm_launch(const float* x, const float* gate,
                                          int K, int E, int T, float* logits,
                                          hipStream_t stream = nullptr);

/// The PHASE9 tile superroc_fp4_route_logits_gemm_launch used to run. Same K %
/// 32 rule. The unit test reports max_abs / top-8 of the LV1 kernel against
/// this one.
int superroc_fp4_route_logits_gemm_legacy_launch(const float* x,
                                                 const float* gate, int K,
                                                 int E, int T, float* logits,
                                                 hipStream_t stream = nullptr);

// The four host-visible words the grouped GEMM's launch config needs.
enum {
  SUPERROC_FP4_LS1_HDR_NGROUPS = 0,
  SUPERROC_FP4_LS1_HDR_MAXCNT = 1,
  SUPERROC_FP4_LS1_HDR_TILES = 2,
  SUPERROC_FP4_LS1_HDR_STATUS = 3,
  SUPERROC_FP4_LS1_HDR_WORDS = 4
};

/// Reproduce, ON THE DEVICE and in the SAME ORDER, the expert grouping that
/// ForwardBatch used to do on the host between superroc_fp4_route_batch and the
/// grouped GEMM: the stable counting sort by expert, the kMoeGroupCap sub-group
/// split, the 16-aligned tile prefix, and the slot -> destination map.
///
/// `out_hdr_host` receives SUPERROC_FP4_LS1_HDR_WORDS ints (n_groups, max_cnt,
/// tiles, status). That read is the ONE transfer left in the path; the call
/// replaces three D2H and eight H2D copies per MoE layer per chunk.
///
/// Every extent is audited on the host before the launch, and a router id
/// outside [0, n_expert) sets the status word rather than indexing -- so this
/// cannot repeat the lane's 2026-10-02T01:58:04Z out-of-range fault.
int superroc_fp4_moe_group_batch(const int* ids, const float* wts,
                                 const int* rc_in, int n, int kMaxExperts,
                                 int n_used, int n_expert, int group_cap,
                                 int* slot_tok, float* slot_w, int* token_slots,
                                 int* grp_e, int* grp_off, int* grp_cnt,
                                 int* grp_zp, int* slot_dest, int grp_capacity,
                                 int slots, int* out_hdr_host);

/// KPF2: multi-block stable grouper for 0 < slots <= 32768, bit-identical
/// to the serial oracle (same counting sort, same kGroupCap split, same
/// 16-aligned tile prefix, same slot -> destination map). Same argument
/// list and output layout as superroc_fp4_moe_group_batch, but the 4 header
/// words go to the DEVICE buffer hdr_dev (PRE-ZEROED by the caller:
/// SUPERROC_FP4_OK is 0) instead of a host readback -- no D2H, no host sync.
/// STATUS is sticky-ERR (kernels set it, never clear it); NGROUPS/MAXCNT/
/// TILES are written by the scan kernel. n_expert <= 256, slots <= 32768.
/// A slot id outside [0, n_expert) or a bad rc sets STATUS exactly as the
/// oracle does, and the host path later refuses the chunk.
int superroc_fp4_moe_group_batch32k(const int* ids, const float* wts,
                                    const int* rc_in, int n, int kMaxExperts,
                                    int n_used, int n_expert, int group_cap,
                                    int* slot_tok, float* slot_w,
                                    int* token_slots, int* grp_e, int* grp_off,
                                    int* grp_cnt, int* grp_zp, int* slot_dest,
                                    int grp_capacity, int slots, int* hdr_dev);

/// KPF2: device-sized grp_zb block prefix. Same contract as
/// superroc_fp4_moe_group_blocks, but n_groups is read from the 32k grouper's
/// hdr_dev inside the kernel instead of passed on the host. A non-OK
/// STATUS leaves grp_zb untouched.
int superroc_fp4_moe_group_blocks_dev(const int* grp_cnt, const int* hdr_dev,
                                      int bn, int* grp_zb, hipStream_t stream);

/// MG64: BN-block prefix over the group counts for the 64-token grouped GEMM.
/// One block, one thread on `stream`: grp_zb[g] is group g's first BN-block
/// and grp_zb[n_groups] is the total block count. bn must be 16, 32, or 64.
/// Returns SUPERROC_FP4_OK, SUPERROC_FP4_ERR_ARG (null pointers, n_groups <= 0,
/// or a bad bn), or the launch check.
int superroc_fp4_moe_group_blocks(const int* grp_cnt, int n_groups, int bn,
                                  int* grp_zb, hipStream_t stream);

#ifdef __cplusplus
}
#endif
