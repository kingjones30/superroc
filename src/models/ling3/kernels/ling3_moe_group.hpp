#pragma once
// LS1 PHASE8 item 2. See ling3_moe_group.hip for what this replaces and why it
// is ordered the way it is.
/// Tokens per CTA tile in the tiled router logits kernel.
enum { LS1_ROUTE_TY = 8 };

// LS1 PHASE9: the smallest chunk, in tokens, at which the 2D register-tiled
// router GEMM is used. Its grid is ceil(E/BM) x ceil(T/BN), so it needs enough
// tokens to fill the 20 CUs; below this the kernel it replaces runs unchanged.
// Overridable at runtime with LING3_ROUTE_GEMM_MIN_T so the crossover can be
// re-measured without a rebuild.
enum { LS1_ROUTE_GEMM_MIN_T = 256 };

#ifdef __cplusplus
extern "C" {
#endif

/// Tiled router logits: one CTA per (expert, LS1_ROUTE_TY tokens) instead of
/// one per (expert, token) PAIR. BIT-IDENTICAL to k_route_batch_logits -- same
/// 64-lane strided partials in the same order, same binary tree -- so the
/// router's block-2 near-tie cannot move. `ty` is clamped to LS1_ROUTE_TY on
/// the host.
int ling3_route_logits_tiled(const float* x, const float* gate, int k,
                             int n_expert, int n_tokens, int ty, float* logits);

/// LV1 router logits: logits[t][e] = sum_i x[t][i] * gate[e][i]. One CTA owns
/// 64 experts x 64 tokens, 128 threads, each an 8x4 register tile, K stepped by
/// 32 in float4. fp32 accumulate, k ascending, no atomics, one writer per
/// output. Replaces the PHASE9 64x32 scalar tile. Returns LING3_ERR_ARG unless
/// K is a multiple of 32.
int ling3_route_logits_gemm_launch(const float* x, const float* gate, int K,
                                   int E, int T, float* logits);

/// The PHASE9 tile ling3_route_logits_gemm_launch used to run. Same K % 32
/// rule. The unit test reports max_abs / top-8 of the LV1 kernel against this
/// one.
int ling3_route_logits_gemm_legacy_launch(const float* x, const float* gate,
                                          int K, int E, int T, float* logits);

// The four host-visible words the grouped GEMM's launch config needs.
enum {
  LS1_HDR_NGROUPS = 0,
  LS1_HDR_MAXCNT = 1,
  LS1_HDR_TILES = 2,
  LS1_HDR_STATUS = 3,
  LS1_HDR_WORDS = 4
};

/// Reproduce, ON THE DEVICE and in the SAME ORDER, the expert grouping that
/// ForwardBatch used to do on the host between ling3_route_batch and the
/// grouped GEMM: the stable counting sort by expert, the kMoeGroupCap sub-group
/// split, the 16-aligned tile prefix, and the slot -> destination map.
///
/// `out_hdr_host` receives LS1_HDR_WORDS ints (n_groups, max_cnt, tiles,
/// status). That read is the ONE transfer left in the path; the call replaces
/// three D2H and eight H2D copies per MoE layer per chunk.
///
/// Every extent is audited on the host before the launch, and a router id
/// outside [0, n_expert) sets the status word rather than indexing -- so this
/// cannot repeat the lane's 2026-10-02T01:58:04Z out-of-range fault.
int ling3_moe_group_batch(const int* ids, const float* wts, const int* rc_in,
                          int n, int kMaxExperts, int n_used, int n_expert,
                          int group_cap, int* slot_tok, float* slot_w,
                          int* token_slots, int* grp_e, int* grp_off,
                          int* grp_cnt, int* grp_zp, int* slot_dest,
                          int grp_capacity, int slots, int* out_hdr_host);

#ifdef __cplusplus
}
#endif
