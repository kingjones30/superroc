// ling3_moe_group.hip — LS1 PHASE8 item 2: the MoE expert grouping ON THE DEVICE.
//
// WHY THIS EXISTS. In ForwardBatch (rocm_backend.cpp) every MoE layer, every 512-token
// chunk, did:
//
//     ling3_route_batch(...)                  // device
//     hipMemcpy D2H ids, wts, rc              // three blocking copies -> drains the GPU
//     a host counting sort by expert, the kMoeGroupCap sub-group split, the 16-aligned
//       tile prefix, and the slot->destination map               // all on the CPU
//     hipMemcpy H2D slot_tok, slot_w, token_slots, grp_e, grp_off, grp_cnt, grp_zp,
//       slot_dest                                                 // eight more copies
//     ling3_wmma_gemm_grp / ling3_gemm_grp
//
// Eleven transfers and one full drain, per MoE layer, per chunk. Jay's build27 profile on
// amdaimax2 has `route` at 145-163 ms per chunk; that is what this file is aimed at.
//
// WHAT THIS DOES. The whole sort is reproduced on the device, in the SAME order, by the SAME
// algorithm -- so the grouped GEMM computes bit-identical results. Not "an equivalent
// grouping": the counting sort's stability (token order inside an expert) is part of the
// contract, because ling3_slot_gather_sum folds a token's slots in slot order.
//
// The algorithm, copied from the host it replaces, one for one:
//
//     counts[e+1]++                    for every slot j, in slot order
//     counts[e+1] += counts[e]         prefix
//     fill[e] = counts[e]
//     for i, for s: j = i*kMaxExperts + s
//         p = fill[hid[j]]++
//         slot_tok[p] = i ; slot_w[p] = hw[j] ; token_slots[j] = p
//     for e: for o in 0..c_e step cap:  grp_e/off/cnt += (e, counts[e]+o, min(c_e-o, cap))
//     grp_zp[gi] = tile_acc ; tile_acc += (grp_cnt[gi] + 15) / 16 ; grp_zp[n_groups] = tile_acc
//     for gi, for c: slot_dest[goff+c] = ((zp[gi] + (c >> 4)) << 4) | (c & 15)
//
// LK1 (wave 3). The serial body below is the oracle: LING3_MOE_GROUP_SERIAL=1
// launches it, and the parallel kernel calls it for rc / id / group-cap errors so
// those partial writes stay byte-identical. The default kernel is parallel. Rank
// is counts[e] + #{j' < j : ids[j'] == e}, which is the stable counting sort.
// Tile-count atomics only add 1s; their order does not change the sum. The lane
// scan that places each slot walks increasing j, so it does not use atomics.
//
// BOUNDS DISCIPLINE (this lane's rule, after the 2026-10-02T01:58:04Z fault). Every size is
// derived from the same constants the caller sized the buffers with, and every index the
// kernel forms is checked against them on the host before the launch, in
// ling3_moe_group_batch. A slot id outside [0, n_expert) sets the status word instead of
// indexing, and the caller turns that into a loud failure -- exactly what the host loop it
// replaces did with `return fail("router id out of range...")`.

#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <cstdlib>

#include "src/models/ling3/kernels/ling3_quant.hpp"

// ---------------------------------------------------------------------------
// LS1 PHASE8 item 1 (NUDGE 2): the ROUTER LOGITS kernel, tiled over tokens.
//
// LR1 MEASURED, both engines on the same 13.8K prompt: k_route_batch_logits costs
// 3.39 ms per call x 1,080 calls = 3.66 s -- 5.1 % of gufo's prefill, and 13.3x the fork's
// router (276 ms). That is ~136 ms of the ~155 ms `route` section per 512-token chunk.
//
// WHY IT IS SLOW, exactly. The old kernel is <<<n_expert, n_tokens>>>, 64 threads: one CTA
// per (expert, token) PAIR, and every CTA walks the whole k = 2048 of BOTH the activation row
// and the gate row. So a chunk reads n_expert x n_tokens x k floats where its inputs are only
// n_tokens x k + n_expert x k. On halo, 479 x 512 x 2048 x 4 B is about 2.0 GB of loads for
// 8 MB of input -- roughly 250x the minimum -- which is why one matvec costs 3.39 ms.
//
// WHAT THIS DOES. One CTA per (expert, tile of LS1_ROUTE_TY tokens). The gate row is read
// ONCE per CTA instead of once per token, and each load it serves feeds LS1_ROUTE_TY
// accumulators. x is still touched per (expert, token), but all of x for a chunk is 3.9 MB
// and stays resident in L2, so DRAM traffic falls by about two orders of magnitude and what
// remains is L2 traffic.
//
// BIT-IDENTICAL, as a design constraint rather than a hope. The old per-row order was:
// lane t accumulates row[t + 64j] * x[t + 64j] for increasing j, then a binary tree
// off = 32, 16, 8, 4, 2, 1 over 64 lanes. This keeps BOTH exactly -- blockDim.x is still
// the lane index, the strided accumulation is still in increasing j for the same lane, the
// tree is the same. Only the set of (expert, token) pairs one CTA serves changed, which
// touches no arithmetic. Ling has a router near-tie at block 2 that flips on a 1-ulp change,
// so "close enough" is not a standard worth shipping against.

// Tokens per CTA tile. 8 puts the reduction LDS at 2 KiB and gives the gate row eight uses
// per load; REPORT8.md has the measured 4 / 8 / 16 spread.
enum { LS1_ROUTE_TY = 2 };

__global__ void k_route_logits_tiled(const float * __restrict__ x,
                                     const float * __restrict__ gate,
                                     int k, int n_expert, int n_tokens,
                                     int ty, float * __restrict__ logits) {
    const int e = blockIdx.x;
    const int t0 = blockIdx.y * ty;
    const int lane = (int) threadIdx.x;
    const float * row = gate + (size_t) e * (size_t) k;

    int nhere = n_tokens - t0;
    if (nhere > ty) {
        nhere = ty;
    }
    if (nhere <= 0) {
        return;
    }
    float acc[LS1_ROUTE_TY];
    for (int u = 0; u < ty; ++u) {
        acc[u] = 0.f;
    }
    // The same lane -> element assignment as before (i = lane, lane + 64, ...), so each
    // lane's partials are bit-identical to the old kernel's; only ty of them at a time.
    for (int i = lane; i < k; i += 64) {
        const float g = row[i];
        for (int u = 0; u < nhere; ++u) {
            acc[u] += g * x[(size_t) (t0 + u) * (size_t) k + (size_t) i];
        }
    }

    // The same 64-lane tree, once per token in the tile.
    __shared__ float red[LS1_ROUTE_TY][64];
    for (int u = 0; u < nhere; ++u) {
        red[u][lane] = acc[u];
    }
    __syncthreads();
    for (int off = 32; off > 0; off >>= 1) {
        if (lane < off) {
            for (int u = 0; u < nhere; ++u) {
                red[u][lane] += red[u][lane + off];
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
        for (int u = 0; u < nhere; ++u) {
            logits[(size_t) (t0 + u) * (size_t) n_expert + (size_t) e] = red[u][0];
        }
    }
}

// ---------------------------------------------------------------------------
// LS1 PHASE9 item 1: the router logits as a real 2D register-tiled GEMM.
//
// REPORT8 proved the bit-identical rewrites cannot win: k_route_batch_logits is
// LATENCY-bound, and keeping its per-row dot order is exactly what forbids the fix.
// Jay's decision (DECISIONS 2026-10-03 00:55) is that the router's accumulation order MAY
// change -- gufo's order is arbitrary (the fork's differs) and the block-2 near-tie moving is
// a numerics change like any other, so the WAVE 2 numerics gates decide, not byte-identity.
//
// THE PROBLEM. logits[t][e] = sum_i x[t][i] * gate[e][i] for t in [0,T), e in [0,E),
// i in [0,K). At Ling's shape that is 512 x 512 x 2048 = 1.07 GFLOP per MoE layer per chunk,
// ~0.2 ms even at 5 TFLOP/s. The old kernel spends 3.39 ms.
//
// WHY THE OLD ONE IS SLOW, restated because it drives the design: it was one CTA per
// (expert, token) PAIR with 64 threads and a 64-deep LDS tree, so it read n_expert x
// n_tokens x k floats where the inputs are n_tokens x k + n_expert x k -- about 2.0 GB of
// loads for 8 MB of input -- and, per REPORT8, cutting the CTA count to amortise the gate row
// made it worse, because the problem is latency and parallelism, not traffic.
//
// THE DESIGN. Each CTA owns a BM(experts) x BN(tokens) output tile and marches over K in
// BK-deep steps, staging an x tile and a gate tile in LDS and keeping a REG_T x REG_T register
// tile per thread. Every gate element loaded is reused across BN tokens and every activation
// element across BM experts, so the traffic is the GEMM's, not n_expert x n_tokens times it.
// That keeps the CTA count at ceil(E/BM) x ceil(T/BN) = 128 here -- high enough to hide
// latency, which is what the 1D tiling destroyed.
//
// DETERMINISTIC. One fixed order -- k ascending, tiles in ascending k -- no atomics, no
// cross-CTA reduction, and each output element is written by exactly one thread. Two runs on the
// same input give the same bits, which the unit test checks as well as comparing against the
// kernel it replaces.
//
// ACCUMULATE in fp32 on both operands: the gate matrix is stored F32 ([n_expert, k] as
// ggml {k, n_expert}), so there is no f16 decision to make. Doing the multiply in f16 to reach
// the WMMA path would change the numerics a second time for no measured gain yet.

enum {
    LS1_RG_BM = 64,     // experts per CTA
    LS1_RG_BN = 32,     // tokens per CTA
    LS1_RG_BK = 32,     // K per step
    LS1_RG_RT = 4,      // register tile per thread, RT x RT
    LS1_RG_THREADS = (LS1_RG_BM * LS1_RG_BN) / (LS1_RG_RT * LS1_RG_RT)   // 128
};

__global__ void k_route_logits_gemm(const float * __restrict__ x,
                                   const float * __restrict__ gate,
                                   int K, int E, int T, float * __restrict__ logits) {
    // LDS: one x tile (BN x BK) and one gate tile (BM x BK).
    __shared__ float xs[LS1_RG_BN][LS1_RG_BK];
    __shared__ float gs[LS1_RG_BM][LS1_RG_BK];

    const int e0 = blockIdx.x * LS1_RG_BM;   // expert tile
    const int t0 = blockIdx.y * LS1_RG_BN;   // token tile
    const int tid = (int) threadIdx.x;

    // Which RT x RT sub-tile this thread owns. A simple 2D split, fixed and identical on
    // every launch -- no swizzle, no atomics, nothing that could make a run differ.
    float acc[LS1_RG_RT][LS1_RG_RT];
#pragma unroll
    for (int a = 0; a < LS1_RG_RT; ++a) {
#pragma unroll
        for (int b = 0; b < LS1_RG_RT; ++b) {
            acc[a][b] = 0.f;
        }
    }

    // Thread -> (expert row, token col) within the tile. 128 threads, 64 experts, 32 tokens:
    // thread `tid` covers expert rows tid/8 .. tid/8+3 and token cols (tid%8)*4 .. +3.
    const int e_loc = (tid / (LS1_RG_BN / LS1_RG_RT)) * LS1_RG_RT;
    const int t_loc = (tid % (LS1_RG_BN / LS1_RG_RT)) * LS1_RG_RT;

    for (int k0 = 0; k0 < K; k0 += LS1_RG_BK) {
        __syncthreads();
        for (int idx = tid; idx < LS1_RG_BN * LS1_RG_BK; idx += LS1_RG_THREADS) {
            const int r = idx / LS1_RG_BK;
            const int c = idx - r * LS1_RG_BK;
            const int t = t0 + r;
            xs[r][c] = (t < T) ? x[(size_t) t * K + (size_t) k0 + c] : 0.f;
        }
        for (int idx = tid; idx < LS1_RG_BM * LS1_RG_BK; idx += LS1_RG_THREADS) {
            const int r = idx / LS1_RG_BK;
            const int c = idx - r * LS1_RG_BK;
            const int e = e0 + r;
            gs[r][c] = (e < E) ? gate[(size_t) e * K + (size_t) k0 + c] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < LS1_RG_BK; ++c) {
            float gv[LS1_RG_RT];
            float xv[LS1_RG_RT];
#pragma unroll
            for (int a = 0; a < LS1_RG_RT; ++a) {
                gv[a] = gs[e_loc + a][c];
            }
#pragma unroll
            for (int b = 0; b < LS1_RG_RT; ++b) {
                xv[b] = xs[t_loc + b][c];
            }
#pragma unroll
            for (int a = 0; a < LS1_RG_RT; ++a) {
#pragma unroll
                for (int b = 0; b < LS1_RG_RT; ++b) {
                    acc[a][b] += gv[a] * xv[b];
                }
            }
        }
    }
#pragma unroll
    for (int a = 0; a < LS1_RG_RT; ++a) {
        const int e = e0 + e_loc + a;
        if (e >= E) {
            continue;
        }
#pragma unroll
        for (int b = 0; b < LS1_RG_RT; ++b) {
            const int t = t0 + t_loc + b;
            if (t >= T) {
                continue;
            }
            logits[(size_t) t * E + e] = acc[a][b];
        }
    }
}


// LV1 SPEC2. The router logits GEMM that ling3_route_logits_gemm_launch runs.
// Tile: 64 experts x 64 tokens, K step 32, 128 threads, each thread an 8x4
// register tile, float4 along K. One LDS buffer, padded float4 stride (8+1)
// so a warp's token rows do not all hit one bank. No atomics, k ascending,
// each output written by exactly one thread.
//
// MEASURED (route_bench2, amdaimax2, 20 CU, T=512 E=512 K=2560, narrow weights):
// median 440.8 us vs k_route_logits_gemm 1041.4 us (2.36x), nondet=0,
// top-8 picks moved 0/4096, max_abs 5.48e-6. rocBLAS on the same buffers was
// 505 us. ~/ls/lv1/out-rbench2.txt.
//
// Index audit. Host rejects K that is not a multiple of 32, and null/non-positive
// extents. Inside the kernel:
//   tid in [0, 128). te = tid/16 in [0, 8), tt = tid%16 in [0, 16).
//   e_loc = te*8 in [0, 64), t_loc = tt*4 in [0, 64).
//   a in [0, 8) and b in [0, 4), so e_loc+a < 64 and t_loc+b < 64. Those are
//   the only shared-memory rows this thread reads.
//   Load: p in [0, 4), i = tid + p*128 covers [0, 512) once.
//   Gate float4s in the tile: 64*8 = 512, so e = i/8 < 64 and k4 = i%8 < 8.
//   Token float4s: the same cover. k0 = kt*32 with kt < K/32, so
//   k0 + k4*4 + 3 <= k0 + 31 < k0 + 32 <= K. A global load happens only when
//   the expert (ee = e0+e) or token (tok = t0+t) is in range; otherwise the
//   float4 stored to LDS is zero. The store writes logits[t*E + e] only when
//   both t < T and e < E.
__global__ void k_route_logits_lv2(const float* __restrict__ x, const float* __restrict__ gate,
                                   int K, int E, int T, float* __restrict__ logits) {
    constexpr int BM = 64, BN = 64, BK = 32, TM = 8, TN = 4;
    constexpr int THREADS = (BM / TM) * (BN / TN);  // 128
    constexpr int K4 = BK / 4;                      // 8
    constexpr int K4P = K4 + 1;                     // 9, odd: avoids a 32-bank clash
    constexpr int NG = (BM * K4) / THREADS;         // 4
    constexpr int NX = (BN * K4) / THREADS;         // 4
    static_assert(THREADS == 128 && NG == 4 && NX == 4, "lv2 tile");

    union Sm {
        struct {
            float4 sG[BM][K4P];
            float4 sX[BN][K4P];
        } in;
        float sC[BN][BM];
    };
    __shared__ __align__(16) Sm sm;

    const int tid = (int)threadIdx.x;
    const int e0 = (int)blockIdx.x * BM;
    const int t0 = (int)blockIdx.y * BN;
    const int te = tid / (BN / TN);
    const int tt = tid % (BN / TN);
    const int e_loc = te * TM;
    const int t_loc = tt * TN;

    float acc[TM][TN];
#pragma unroll
    for (int a = 0; a < TM; ++a)
#pragma unroll
        for (int b = 0; b < TN; ++b) acc[a][b] = 0.f;

    auto load_regs = [&](float4 greg[NG], float4 xreg[NX], int k0) {
#pragma unroll
        for (int p = 0; p < NG; ++p) {
            const int i = tid + p * THREADS;
            const int e = i / K4;
            const int k4 = i - e * K4;
            const int ee = e0 + e;
            greg[p] = (ee < E) ? *reinterpret_cast<const float4*>(
                                     gate + ((size_t)ee * (size_t)K + (size_t)k0 + (size_t)k4 * 4))
                               : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int p = 0; p < NX; ++p) {
            const int i = tid + p * THREADS;
            const int t = i / K4;
            const int k4 = i - t * K4;
            const int tok = t0 + t;
            xreg[p] = (tok < T) ? *reinterpret_cast<const float4*>(
                                      x + ((size_t)tok * (size_t)K + (size_t)k0 + (size_t)k4 * 4))
                                : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    };
    auto store_smem = [&](float4 greg[NG], float4 xreg[NX]) {
#pragma unroll
        for (int p = 0; p < NG; ++p) {
            const int i = tid + p * THREADS;
            const int e = i / K4;
            const int k4 = i - e * K4;
            sm.in.sG[e][k4] = greg[p];
        }
#pragma unroll
        for (int p = 0; p < NX; ++p) {
            const int i = tid + p * THREADS;
            const int t = i / K4;
            const int k4 = i - t * K4;
            sm.in.sX[t][k4] = xreg[p];
        }
    };
    auto compute = [&]() {
#pragma unroll
        for (int k4 = 0; k4 < K4; ++k4) {
            float4 gv[TM];
            float4 xv[TN];
#pragma unroll
            for (int a = 0; a < TM; ++a) gv[a] = sm.in.sG[e_loc + a][k4];
#pragma unroll
            for (int b = 0; b < TN; ++b) xv[b] = sm.in.sX[t_loc + b][k4];
#pragma unroll
            for (int a = 0; a < TM; ++a)
#pragma unroll
                for (int b = 0; b < TN; ++b) {
                    acc[a][b] = fmaf(gv[a].x, xv[b].x, acc[a][b]);
                    acc[a][b] = fmaf(gv[a].y, xv[b].y, acc[a][b]);
                    acc[a][b] = fmaf(gv[a].z, xv[b].z, acc[a][b]);
                    acc[a][b] = fmaf(gv[a].w, xv[b].w, acc[a][b]);
                }
        }
    };

    const int ntiles = K / BK;
    for (int kt = 0; kt < ntiles; ++kt) {
        float4 greg[NG], xreg[NX];
        load_regs(greg, xreg, kt * BK);
        store_smem(greg, xreg);
        __syncthreads();
        compute();
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < TM; ++a) {
        const int e = e0 + e_loc + a;
        if (e >= E) continue;
#pragma unroll
        for (int b = 0; b < TN; ++b) {
            const int t = t0 + t_loc + b;
            if (t >= T) continue;
            logits[(size_t)t * (size_t)E + e] = acc[a][b];
        }
    }
}

extern "C" {

// Header word layout, shared with the caller.
enum {
    LS1_HDR_NGROUPS = 0,
    LS1_HDR_MAXCNT = 1,
    LS1_HDR_TILES = 2,
    LS1_HDR_STATUS = 3,
    LS1_HDR_WORDS = 4
};

// scratch_counts: n_expert + 1 ints (the counting-sort histogram and its prefix)
// scratch_fill:   n_expert ints
// Both grow on demand and are never freed until process exit, like the other kernels'
// scratch in this tree.
static int * g_scratch = nullptr;
static int g_scratch_words = 0;
static int g_counts_n = 0;

__device__ __noinline__ void moe_group_serial(const int * ids, const float * wts, const int * rc_in,
                            int n, int kMaxExperts, int n_used, int n_expert,
                            int group_cap, int grp_capacity, int slots,
                            int * slot_tok, float * slot_w, int * token_slots,
                            int * grp_e, int * grp_off, int * grp_cnt,
                            int * grp_zp, int * slot_dest,
                            int * counts, int * fill, int * hdr) {
    // STATUS starts OK and is only ever set, never cleared, so an error is sticky.
    hdr[LS1_HDR_STATUS] = LING3_OK;

    // The router's own per-token status, exactly as the host loop re-read it per token.
    for (int i = 0; i < n; ++i) {
        if (rc_in != nullptr && rc_in[i] != LING3_OK) {
            hdr[LS1_HDR_STATUS] = LING3_ERR_ARG;
            return;
        }
    }

    // Histogram, slot order.
    for (int e = 0; e <= n_expert; ++e) {
        counts[e] = 0;
    }
    for (int e = 0; e < n_expert; ++e) {
        fill[e] = 0;
    }
    for (int j = 0; j < slots; ++j) {
        const int e = ids[j];
        if (e < 0 || e >= n_expert) {
            // The host returned fail("router id out of range") here. Same outcome, no
            // out-of-bounds write, and the caller still fails loudly.
            hdr[LS1_HDR_STATUS] = LING3_ERR_ARG;
            return;
        }
        counts[e + 1]++;
    }
    // Exclusive prefix, then the fill cursors.
    for (int e = 0; e < n_expert; ++e) {
        counts[e + 1] += counts[e];
        fill[e] = counts[e];
    }
    // The stable scatter. j ascends, so inside one expert the slots keep token order --
    // that is the property ling3_slot_gather_sum depends on.
    for (int i = 0; i < n; ++i) {
        for (int s = 0; s < kMaxExperts; ++s) {
            const int j = i * kMaxExperts + s;
            if (j >= slots) {
                break;
            }
            const int p = fill[ids[j]]++;
            slot_tok[p] = i;
            slot_w[p] = wts[j];
            token_slots[j] = p;
        }
    }

    // Sub-group capping, then the 16-aligned tile prefix.
    int n_groups = 0;
    int max_cnt = 0;
    int tile_acc = 0;
    for (int e = 0; e < n_expert; ++e) {
        const int c = counts[e + 1] - counts[e];
        for (int o = 0; o < c; o += group_cap) {
            const int take = (c - o) < group_cap ? (c - o) : group_cap;
            if (n_groups >= grp_capacity) {
                hdr[LS1_HDR_STATUS] = LING3_ERR_ARG;
                return;
            }
            grp_e[n_groups] = e;
            grp_off[n_groups] = counts[e] + o;
            grp_cnt[n_groups] = take;
            grp_zp[n_groups] = tile_acc;
            tile_acc += (take + 15) / 16;
            if (take > max_cnt) {
                max_cnt = take;
            }
            ++n_groups;
        }
    }
    grp_zp[n_groups] = tile_acc;

    // Slot -> destination, group by group.
    for (int gi = 0; gi < n_groups; ++gi) {
        const int base = grp_zp[gi];
        const int goff = grp_off[gi];
        const int gcnt = grp_cnt[gi];
        for (int c = 0; c < gcnt; ++c) {
            const int d = goff + c;
            if (d >= 0 && d < slots) {
                slot_dest[d] = ((base + (c >> 4)) << 4) | (c & 15);
            }
        }
    }

    hdr[LS1_HDR_NGROUPS] = n_groups;
    hdr[LS1_HDR_MAXCNT] = max_cnt;
    hdr[LS1_HDR_TILES] = tile_acc;
}

__global__ void k_moe_group_serial(const int * ids, const float * wts, const int * rc_in,
                                   int n, int kMaxExperts, int n_used, int n_expert,
                                   int group_cap, int grp_capacity, int slots,
                                   int * slot_tok, float * slot_w, int * token_slots,
                                   int * grp_e, int * grp_off, int * grp_cnt,
                                   int * grp_zp, int * slot_dest,
                                   int * counts, int * fill, int * hdr) {
    if ((int)threadIdx.x == 0) {
        moe_group_serial(ids, wts, rc_in, n, kMaxExperts, n_used, n_expert,
                         group_cap, grp_capacity, slots,
                         slot_tok, slot_w, token_slots,
                         grp_e, grp_off, grp_cnt, grp_zp, slot_dest,
                         counts, fill, hdr);
    }
}


__global__ void k_moe_group_parallel(const int * ids, const float * wts, const int * rc_in,
                     int n, int kMaxExperts, int n_used, int n_expert,
                     int group_cap, int grp_capacity, int slots,
                     int * slot_tok, float * slot_w, int * token_slots,
                     int * grp_e, int * grp_off, int * grp_cnt,
                     int * grp_zp, int * slot_dest,
                     int * counts, int * fill, int * hdr) {
    const int tx = (int)threadIdx.x;
    const int NG = 8;
    const int LANES = 32;
    const int SPR = NG * LANES; // 256
    (void)n_used;

    extern __shared__ int sm[];
    int * ids_all = sm;
    int * tot = ids_all + slots;
    int * cursor = tot + n_expert;
    int * round_count = cursor + n_expert;          // [NG][n_expert]
    int * round_total = round_count + NG * n_expert;
    int * ids_tile = round_total + n_expert;        // SPR

    __shared__ int err;
    __shared__ int do_serial;
    __shared__ int n_groups_s;

    // blockDim is part of the rank mapping (8 groups of 32). Anything else
    // would be a silent wrong answer; the serial body is the oracle.
    if (blockDim.x != 256) {
        if (tx == 0) {
            moe_group_serial(ids, wts, rc_in, n, kMaxExperts, n_used, n_expert,
                             group_cap, grp_capacity, slots,
                             slot_tok, slot_w, token_slots,
                             grp_e, grp_off, grp_cnt, grp_zp, slot_dest,
                             counts, fill, hdr);
        }
        return;
    }

    if (tx == 0) {
        err = 0;
        do_serial = 0;
        hdr[LS1_HDR_STATUS] = LING3_OK;
    }
    __syncthreads();

    if (rc_in != nullptr) {
        for (int i = tx; i < n; i += blockDim.x) {
            if (rc_in[i] != LING3_OK) {
                atomicExch(&err, 1);
            }
        }
    }
    __syncthreads();
    if (err) {
        if (tx == 0) {
            moe_group_serial(ids, wts, rc_in, n, kMaxExperts, n_used, n_expert,
                             group_cap, grp_capacity, slots,
                             slot_tok, slot_w, token_slots,
                             grp_e, grp_off, grp_cnt, grp_zp, slot_dest,
                             counts, fill, hdr);
        }
        return;
    }

    for (int j = tx; j < slots; j += blockDim.x) {
        const int e = ids[j];
        if (e < 0 || e >= n_expert) {
            atomicExch(&err, 1);
        } else {
            ids_all[j] = e;
        }
    }
    __syncthreads();
    if (err) {
        if (tx == 0) {
            moe_group_serial(ids, wts, rc_in, n, kMaxExperts, n_used, n_expert,
                             group_cap, grp_capacity, slots,
                             slot_tok, slot_w, token_slots,
                             grp_e, grp_off, grp_cnt, grp_zp, slot_dest,
                             counts, fill, hdr);
        }
        return;
    }

    // Zero first so a missed expert is a wrong count, not a garbage index
    // into the group loop (that would fault the GPU).
    for (int e = tx; e < n_expert; e += blockDim.x) {
        tot[e] = 0;
    }
    __syncthreads();

    // Hot shape (Ling): 512 experts, 256 threads, exactly two experts each.
    // An unrolled shared scan. The general path below covers E not 512 and
    // a slot count that is not a multiple of 8. Skew does not change the
    // trip count.
    if (n_expert == 512 && (slots & 7) == 0) {
        const int e0 = tx;
        const int e1 = tx + 256;
        int c0 = 0, c1 = 0;
        for (int t = 0; t < slots; t += 8) {
            const int i0 = ids_all[t];
            const int i1 = ids_all[t + 1];
            const int i2 = ids_all[t + 2];
            const int i3 = ids_all[t + 3];
            const int i4 = ids_all[t + 4];
            const int i5 = ids_all[t + 5];
            const int i6 = ids_all[t + 6];
            const int i7 = ids_all[t + 7];
            c0 += (i0 == e0) + (i1 == e0) + (i2 == e0) + (i3 == e0)
                + (i4 == e0) + (i5 == e0) + (i6 == e0) + (i7 == e0);
            c1 += (i0 == e1) + (i1 == e1) + (i2 == e1) + (i3 == e1)
                + (i4 == e1) + (i5 == e1) + (i6 == e1) + (i7 == e1);
        }
        tot[e0] = c0;
        tot[e1] = c1;
    } else {
        int mine0 = -1, mine1 = -1, mine2 = -1, mine3 = -1;
        int nm = 0;
        for (int e = tx; e < n_expert; e += blockDim.x) {
            if (nm == 0) mine0 = e;
            else if (nm == 1) mine1 = e;
            else if (nm == 2) mine2 = e;
            else if (nm == 3) mine3 = e;
            else atomicExch(&err, 1);
            ++nm;
        }
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (int j = 0; j < slots; ++j) {
            const int id = ids_all[j];
            if (nm > 0 && id == mine0) ++c0;
            if (nm > 1 && id == mine1) ++c1;
            if (nm > 2 && id == mine2) ++c2;
            if (nm > 3 && id == mine3) ++c3;
        }
        if (nm > 0) tot[mine0] = c0;
        if (nm > 1) tot[mine1] = c1;
        if (nm > 2) tot[mine2] = c2;
        if (nm > 3) tot[mine3] = c3;
    }
    __syncthreads();
    if (err) {
        if (tx == 0) {
            moe_group_serial(ids, wts, rc_in, n, kMaxExperts, n_used, n_expert,
                             group_cap, grp_capacity, slots,
                             slot_tok, slot_w, token_slots,
                             grp_e, grp_off, grp_cnt, grp_zp, slot_dest,
                             counts, fill, hdr);
        }
        return;
    }

    if (tx == 0) {
        int acc = 0;
        int needed = 0;
        for (int e = 0; e < n_expert; ++e) {
            const int c = tot[e];
            counts[e] = acc;
            fill[e] = acc + c;
            cursor[e] = acc;
            acc += c;
            if (c > 0) {
                needed += (c + group_cap - 1) / group_cap;
            }
        }
        counts[n_expert] = acc;
        // Strictly more groups than grp_capacity: the serial loop rejects
        // when it is about to write index grp_capacity. Equal fits.
        do_serial = needed > grp_capacity;
    }
    __syncthreads();
    if (do_serial) {
        if (tx == 0) {
            moe_group_serial(ids, wts, rc_in, n, kMaxExperts, n_used, n_expert,
                             group_cap, grp_capacity, slots,
                             slot_tok, slot_w, token_slots,
                             grp_e, grp_off, grp_cnt, grp_zp, slot_dest,
                             counts, fill, hdr);
        }
        return;
    }

    const int group = tx >> 5;
    const int lane = tx & 31;
    (void)ids_tile;
    (void)LANES;
    (void)SPR;
    // 1024 slots per round (8 groups x 128) so a 4096-slot chunk is 4 rounds,
    // not 16. Each lane owns 4 slots in its group: j = group_base + k*32 + lane.
    const int PER = 1024;
    const int GSPAN = 128;
    const int n_rounds = (slots + PER - 1) / PER;
    for (int round = 0; round < n_rounds; ++round) {
        for (int i = tx; i < NG * n_expert; i += blockDim.x) {
            round_count[i] = 0;
        }
        __syncthreads();

        const int group_base = round * PER + group * GSPAN;
        if (group_base < slots) {
            for (int k = 0; k < 4; ++k) {
                const int j = group_base + k * 32 + lane;
                if (j < slots) {
                    atomicAdd(round_count + (group * n_expert + ids_all[j]), 1);
                }
            }
        }
        __syncthreads();

        for (int e = tx; e < n_expert; e += blockDim.x) {
            int run = 0;
            for (int g = 0; g < NG; ++g) {
                const int c = round_count[g * n_expert + e];
                round_count[g * n_expert + e] = run;
                run += c;
            }
            round_total[e] = run;
        }
        __syncthreads();

        if (group_base < slots) {
            for (int k = 0; k < 4; ++k) {
                const int j = group_base + k * 32 + lane;
                if (j >= slots) {
                    continue;
                }
                const int e = ids_all[j];
                int local = 0;
                for (int t = group_base; t < j; ++t) {
                    local += (ids_all[t] == e);
                }
                const int p = cursor[e] + round_count[group * n_expert + e] + local;
                if ((unsigned)p < (unsigned)slots) {
                    slot_tok[p] = j / kMaxExperts;
                    slot_w[p] = wts[j];
                    token_slots[j] = p;
                } else {
                    atomicExch(&err, 1);
                }
            }
        }
        __syncthreads();
        if (err) {
            if (tx == 0) {
                hdr[LS1_HDR_STATUS] = LING3_ERR_ARG;
            }
            return;
        }

        for (int e = tx; e < n_expert; e += blockDim.x) {
            cursor[e] += round_total[e];
        }
        __syncthreads();
    }

    if (tx == 0) {
        int n_groups = 0;
        int max_cnt = 0;
        int tile_acc = 0;
        int rejected = 0;
        for (int e = 0; e < n_expert && !rejected; ++e) {
            const int c = counts[e + 1] - counts[e];
            for (int o = 0; o < c; o += group_cap) {
                const int take = (c - o) < group_cap ? (c - o) : group_cap;
                if (n_groups >= grp_capacity) {
                    hdr[LS1_HDR_STATUS] = LING3_ERR_ARG;
                    rejected = 1;
                    break;
                }
                grp_e[n_groups] = e;
                grp_off[n_groups] = counts[e] + o;
                grp_cnt[n_groups] = take;
                grp_zp[n_groups] = tile_acc;
                tile_acc += (take + 15) / 16;
                if (take > max_cnt) {
                    max_cnt = take;
                }
                ++n_groups;
            }
        }
        if (!rejected) {
            grp_zp[n_groups] = tile_acc;
            hdr[LS1_HDR_NGROUPS] = n_groups;
            hdr[LS1_HDR_MAXCNT] = max_cnt;
            hdr[LS1_HDR_TILES] = tile_acc;
            hdr[LS1_HDR_STATUS] = LING3_OK;
        }
        n_groups_s = rejected ? -1 : n_groups;
    }
    __syncthreads();
    if (n_groups_s < 0) {
        return;
    }
    for (int gi = tx; gi < n_groups_s; gi += blockDim.x) {
        const int base = grp_zp[gi];
        const int goff = grp_off[gi];
        const int gcnt = grp_cnt[gi];
        for (int c = 0; c < gcnt; ++c) {
            const int d = goff + c;
            if (d >= 0 && d < slots) {
                slot_dest[d] = ((base + (c >> 4)) << 4) | (c & 15);
            }
        }
    }
}



/// Launch the tiled router logits. `ty` is clamped to LS1_ROUTE_TY on the HOST, so the
/// kernel can trust the LDS it indexes -- the lane's bounds discipline: every extent the
/// device code touches is checked against the constant it was compiled with, before launch.
int ling3_route_logits_tiled(const float * x, const float * gate, int k, int n_expert,
                             int n_tokens, int ty, float * logits) {
    if (!x || !gate || !logits || k <= 0 || n_expert <= 0 || n_tokens <= 0) {
        return LING3_ERR_ARG;
    }
    if (ty > LS1_ROUTE_TY) {
        ty = LS1_ROUTE_TY;
    }
    if (ty <= 0) {
        return LING3_ERR_ARG;
    }
    const int ty_blocks = (n_tokens + ty - 1) / ty;
    k_route_logits_tiled<<<dim3(n_expert, ty_blocks), 64>>>(
        x, gate, k, n_expert, n_tokens, ty, logits);
    return hipGetLastError() == hipSuccess ? LING3_OK : 3;
}

/// Launch the 2D register-tiled router logits. Behind LING3_ROUTE_GEMM=1; unset keeps the
/// kernel that shipped. The extents are audited on the HOST before the launch -- the lane's
/// rule since the 2026-10-02T01:58:04Z fault -- and every index the device forms is either
/// bounds-checked above or provably inside the tiles.
int ling3_route_logits_gemm_legacy_launch(const float * x, const float * gate, int K, int E, int T,
                                         float * logits);

int ling3_route_logits_gemm_launch(const float * x, const float * gate, int K, int E, int T,
                                  float * logits) {
    if (!x || !gate || !logits || K <= 0 || E <= 0 || T <= 0) {
        return LING3_ERR_ARG;
    }
    // Jay 2026-10-03 (RV12): LING3_ROUTE_LV2_OFF=1 sends the router GEMM back to the 28g tile kernel, for an A/B
    // of top-8 picks or a hot disable without a rebuild.
    static const bool lv2_off = [] {
        const char * e = std::getenv("LING3_ROUTE_LV2_OFF");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    if (lv2_off) {
        return ling3_route_logits_gemm_legacy_launch(x, gate, K, E, T, logits);
    }
    // k_route_logits_lv2 steps K by 32 and loads float4s, so a short row would
    // read off the end. The legacy tile has the same constraint.
    if ((K % 32) != 0) {
        return LING3_ERR_ARG;
    }
    k_route_logits_lv2<<<dim3((E + 63) / 64, (T + 63) / 64), 128>>>(x, gate, K, E, T, logits);
    return hipGetLastError() == hipSuccess ? LING3_OK : 3;
}

int ling3_route_logits_gemm_legacy_launch(const float * x, const float * gate, int K, int E, int T,
                                         float * logits) {
    if (!x || !gate || !logits || K <= 0 || E <= 0 || T <= 0) {
        return LING3_ERR_ARG;
    }
    if ((K % LS1_RG_BK) != 0) {
        return LING3_ERR_ARG;
    }
    const dim3 grid((E + LS1_RG_BM - 1) / LS1_RG_BM, (T + LS1_RG_BN - 1) / LS1_RG_BN);
    k_route_logits_gemm<<<grid, LS1_RG_THREADS>>>(x, gate, K, E, T, logits);
    return hipGetLastError() == hipSuccess ? LING3_OK : 3;
}

int ling3_moe_group_batch(const int * ids, const float * wts, const int * rc_in,
                         int n, int kMaxExperts, int n_used, int n_expert,
                         int group_cap,
                         int * slot_tok, float * slot_w, int * token_slots,
                         int * grp_e, int * grp_off, int * grp_cnt, int * grp_zp,
                         int * slot_dest, int grp_capacity, int slots,
                         int * out_hdr_host) {
    if (!ids || !wts || !slot_tok || !slot_w || !token_slots || !grp_e || !grp_off ||
        !grp_cnt || !grp_zp || !slot_dest || !out_hdr_host) {
        return LING3_ERR_ARG;
    }
    if (n <= 0 || n_expert <= 0 || n_expert > 1024 || kMaxExperts <= 0 || n_used <= 0 ||
        group_cap <= 0 || grp_capacity <= 0 || slots <= 0) {
        return LING3_ERR_ARG;
    }
    // ---- host-side bounds audit, before anything is launched ----
    if ((long long) n * (long long) kMaxExperts > (long long) slots) {
        fprintf(stderr, "ling3_moe_group: n*kMaxExperts=%lld exceeds slots=%d\n",
                (long long) n * (long long) kMaxExperts, slots);
        return LING3_ERR_ARG;
    }
    // Worst case groups: expert e contributes ceil(c_e / cap) <= c_e/cap + 1, and the
    // c_e sum to at most the slot count, so n_expert + slots/cap is a hard upper bound.
    // The first cut used slots + n_expert, which is valid but far looser than what the
    // host actually allocates (n_expert + slots/8 + 2), so the audit rejected
    // legitimate calls. grp_zp needs one entry more than the groups, hence the +1.
    const long long max_groups = (long long) n_expert + (long long) slots / group_cap + 1;
    // Test-only. Production never sets this. The in-kernel cap check is what the
    // overflow case compares; the host audit rejects that shape before launch.
    const char * skip_e = std::getenv("LING3_MOE_GROUP_SKIP_AUDIT");
    const bool skip_audit = skip_e != nullptr && skip_e[0] == '1' && skip_e[1] == '\0';
    if (!skip_audit && max_groups > (long long) grp_capacity + 1) {
        fprintf(stderr, "ling3_moe_group: need grp_capacity >= %lld (+1), have %d\n",
                max_groups, grp_capacity);
        return LING3_ERR_ARG;
    }

    // counts[n_expert+1] + fill[n_expert] + hdr[LS1_HDR_WORDS]
    const int need = (n_expert + 1) + n_expert + LS1_HDR_WORDS;
    if (need > g_scratch_words || g_scratch == nullptr) {
        if (g_scratch) { hipFree(g_scratch); }
        g_scratch = nullptr;
        g_scratch_words = 0;
        if (hipMalloc(&g_scratch, (std::size_t) need * sizeof(int)) != hipSuccess) {
            return 3 /* LING3_ERR_HIP */;
        }
        g_scratch_words = need;
    }
    int * counts = g_scratch;
    int * fill = g_scratch + (n_expert + 1);
    int * hdr = g_scratch + (n_expert + 1) + n_expert;
    g_counts_n = n_expert + 1;

    // Layout must match k_moe_group_parallel: ids_all[slots], tot[E], cursor[E],
    // round_count[8*E], round_total[E], ids_tile[256].
    const int lk1_ng = 8;
    const int lk1_spr = 256;
    const long smem_ints = (long)slots + (long)n_expert * (3 + lk1_ng) + lk1_spr;
    const long smem_bytes = smem_ints * (long)sizeof(int);
    const char * serial_e = std::getenv("LING3_MOE_GROUP_SERIAL");
    const bool force_serial = serial_e != nullptr && serial_e[0] == '1' && serial_e[1] == '\0';
    const bool shape_ok = slots == n * kMaxExperts && smem_bytes > 0 && smem_bytes <= 48 * 1024
                          && slots <= 8192;
    if (force_serial || !shape_ok) {
        k_moe_group_serial<<<1, 1>>>(ids, wts, rc_in, n, kMaxExperts, n_used, n_expert,
                                     group_cap, grp_capacity, slots,
                                     slot_tok, slot_w, token_slots,
                                     grp_e, grp_off, grp_cnt, grp_zp, slot_dest,
                                     counts, fill, hdr);
    } else {
        k_moe_group_parallel<<<1, 256, (std::size_t)smem_bytes>>>(
            ids, wts, rc_in, n, kMaxExperts, n_used, n_expert, group_cap,
            grp_capacity, slots,
            slot_tok, slot_w, token_slots,
            grp_e, grp_off, grp_cnt, grp_zp, slot_dest,
            counts, fill, hdr);
    }
    if (hipGetLastError() != hipSuccess) {
        return 3;
    }
    // The ONE readback left in the path: four ints. It replaces three D2H and eight H2D.
    if (hipMemcpy(out_hdr_host, hdr, LS1_HDR_WORDS * sizeof(int),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        return 3;
    }
    return LING3_OK;
}


int ling3_moe_group_debug_counts(int * host, int n_out) {
    if (!host || !g_scratch || n_out <= 0 || g_counts_n <= 0) {
        return LING3_ERR_ARG;
    }
    if (n_out > g_counts_n) {
        n_out = g_counts_n;
    }
    if (hipMemcpy(host, g_scratch, (std::size_t)n_out * sizeof(int),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        return 3;
    }
    return LING3_OK;
}

}  // extern "C"
