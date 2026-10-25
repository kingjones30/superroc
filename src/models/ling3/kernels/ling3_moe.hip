#include "ling3_moe.hpp"

#include "ling3_gemv_q8.hpp"
#include "src/core/hip_launch_check.hpp"

#include <hip/hip_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ling3_hip(hipError_t e, const char * what) {
    if (e == hipSuccess) {
        return LING3_OK;
    }
    fprintf(stderr, "ling3 %s: %s\n", what, hipGetErrorString(e));
    return LING3_ERR_HIP;
}

static int ling3_sync(const char * what) {
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        return ling3_hip(e, what);
    }
    return ling3_hip(hipDeviceSynchronize(), what);
}

// H1 2026-10-01: every kernel here runs on the NULL stream, so launches are already
// ordered against each other and against hipMemcpy. The per-call hipDeviceSynchronize()
// in ling3_sync was serialising ~2200 launches per token against the host for no
// ordering benefit: the only places the host reads device data go through hipMemcpy,
// which synchronises on its own. Keep the error check, drop the barrier.
static int ling3_chk(const char * what) {
    return ling3_hip(hipGetLastError(), what);
}

__global__ void k_dequant(int qtype, const uint8_t * rows, size_t row_bytes, float * y, int k, int n) {
    int row = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (row < 0 || row >= n) {
        return;
    }
    ling3_dequant_row(qtype, rows + (size_t) row * row_bytes, y + (size_t) row * (size_t) k, k);
}

__global__ void k_gemv(int qtype, const uint8_t * W, size_t row_bytes, const float * x, float * y, int k, int n) {
    int row = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (row < 0 || row >= n) {
        return;
    }
    y[row] = ling3_dot_row(qtype, W + (size_t) row * row_bytes, x, k);
}

// H1 2026-10-01: ling3_elem() re-reads the block's ue4m3 scale byte and re-runs ling3_ue4m3_half()
// for EVERY one of the 32 elements of a ROCmFP4 block. This decodes the scale once per block and
// then walks the nibbles. Identical decode, identical values: element j takes the scale at offset
// 16 (j<16) or 17 (j>=16) for ROCmFP4, and offset 16 for ROCmFP4_FAST, exactly as ling3_term_block
// does, and the value is code(nib) * scale, exactly as ling3_term_f32 does for c1 == 1.
__device__ __forceinline__ float h1_dot_fp4_blk32(int qtype, const uint8_t * blk,
                                                  const float * x, int j0) {
    const bool fast = (qtype == LING3_Q4_0_ROCMFP4_FAST);
    const float slo = ling3_ue4m3_half(ling3_ld8(blk + 16));
    const float shi = fast ? slo : ling3_ue4m3_half(ling3_ld8(blk + 17));
    float a = 0.f;
    for (int j = 0; j < 32; ++j) {
        // Two details of ling3_term_block that must be copied exactly:
        //   * the nibble byte is indexed `local & 15`, so elements 16..31 REUSE bytes 0..15;
        //   * low-vs-high nibble (and which scale byte) switch at local == 16, NOT every element.
        // My first two attempts got one of these wrong and the trace gate caught both.
        const uint8_t byte = ling3_ld8(blk + (j & 15));
        const int nib = (j < 16) ? (int) (byte & 0x0f) : (int) (byte >> 4);
        const float f = (j < 16) ? slo : shi;
        a += ((float) ling3_rocmfp4_code((uint32_t) nib) * f) * x[j0 + j];
    }
    return a;
}

// H1 2026-10-01 (profile: k_gemv was 89.5% of GPU time, 1.34 ms per call, 2210 calls/token).
// The old shape was ONE THREAD PER OUTPUT ROW: n=768 gave 12 blocks of 64 = 12 wavefronts on a
// 20-CU GPU, so there was no latency hiding whatsoever and every byte fetch was fully exposed.
// Here one 64-thread block owns a row and strides k, then tree-reduces. That is 64x the
// parallelism for the same work. The dequant is the SAME ling3_elem call, so every value is
// bit-identical; only the order of the F32 additions changes (quantisation-noise level).
__global__ void k_gemv_wide(int qtype, const uint8_t * W, size_t row_bytes,
                            const float * x, float * y, int k, int n) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int row = (int) blockIdx.x;
    if (row >= n) {
        return;
    }
    const uint8_t * w = W + (size_t) row * row_bytes;
    float s = 0.f;
    if (qtype == LING3_Q4_0_ROCMFP4 || qtype == LING3_Q4_0_ROCMFP4_FAST) {
        const int tsz = ling3_type_size(qtype);
        const int nblk = k / 32;
        for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
            s += h1_dot_fp4_blk32(qtype, w + (size_t) qb * (size_t) tsz, x, qb * 32);
        }
    } else {
        for (int i = (int) threadIdx.x; i < k; i += (int) blockDim.x) {
            s += ling3_elem(qtype, w, i) * x[i];
        }
    }
    __shared__ float red[64];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int off = 32; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) {
            red[threadIdx.x] += red[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[row] = red[0];
    }
}

// Same wide shape, but grid-strided over rows from a FIXED grid. The LM head is n=157184: one
// block per row would be 157k blocks of 64, all of them two-word jobs, and the block-scheduling
// overhead dominates. A few thousand resident blocks, each looping over ~38 rows, keeps every CU
// busy without that. Same ling3_elem dequant, same per-row reduction, so the values are the same
// as k_gemv_wide's.
__global__ void k_gemv_wide_gs(int qtype, const uint8_t * W, size_t row_bytes,
                               const float * x, float * y, int k, int n) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    __shared__ float red[64];
    for (int row = (int) blockIdx.x; row < n; row += (int) gridDim.x) {
        const uint8_t * w = W + (size_t) row * row_bytes;
        float s = 0.f;
        if (qtype == LING3_Q4_0_ROCMFP4 || qtype == LING3_Q4_0_ROCMFP4_FAST) {
            const int tsz = ling3_type_size(qtype);
            const int nblk = k / 32;
            for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
                s += h1_dot_fp4_blk32(qtype, w + (size_t) qb * (size_t) tsz, x, qb * 32);
            }
        } else {
            for (int i = (int) threadIdx.x; i < k; i += (int) blockDim.x) {
                s += ling3_elem(qtype, w, i) * x[i];
            }
        }
        __syncthreads();
        red[threadIdx.x] = s;
        __syncthreads();
        for (int off = 32; off > 0; off >>= 1) {
            if ((int) threadIdx.x < off) {
                red[threadIdx.x] += red[threadIdx.x + off];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            y[row] = red[0];
        }
    }
}

__global__ void k_gemm(int qtype, const uint8_t * W, size_t row_bytes,
        const float * x, float * y, int k, int n, int m) {
    int row = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    int token = (int) blockIdx.y;
    if (row < 0 || row >= n || token < 0 || token >= m) {
        return;
    }
    y[(size_t) token * (size_t) n + row] =
        ling3_dot_row(qtype, W + (size_t) row * row_bytes, x + (size_t) token * (size_t) k, k);
}

__global__ void k_swiglu(const float * gate, const float * up, float * h, int n, float limit) {
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < 0 || i >= n) {
        return;
    }
    h[i] = ling3_swiglu_elem(gate[i], up[i], limit);
}

__global__ void k_axpy(float w, const float * p, float * y, int n) {
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < 0 || i >= n) {
        return;
    }
    y[i] += w * p[i];
}

__global__ void k_add(const float * a, const float * b, float * y, int n) {
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < 0 || i >= n) {
        return;
    }
    y[i] = a[i] + b[i];
}

// ---- H1 2026-10-01: parallel replacement for ling3_bitonic_argsort_desc ------------
// The profile put k_route at 25% (later 52%) of GPU time, 3.5-5 ms per call. The cause is that
// ling3_route_core runs on ONE thread and its bitonic argsort does, per (k,j) stage, three full
// passes over `pad` elements with a dependent global/LDS round trip each. That is ~69k serial
// memory operations for a 512-element sort.
//
// This is the SAME algorithm with the compare-exchange spread over the block. Within one (k,j)
// stage the pairs (col, col^j) for col with bit j clear are DISJOINT, so the order in which the
// block touches them cannot change the result, and the predicate, the padding rule
// (index >= ncols sorts to the end) and the strict < / > (ties do not swap) are copied verbatim.
// The two copy passes of the original are dropped because writing in place is equivalent: the
// original computes work_b from work_a, conditionally swaps within the pair, then copies back.
__device__ __forceinline__ int h1_argsort_desc_par(const float * x, int * dst, int * wa,
                                                    int ncols, int nt) {
    const int pad = ling3_pow2_ge(ncols);
    if (pad < 0 || !x || !dst || !wa) {
        return LING3_ERR_ARG;
    }
    for (int i = (int) threadIdx.x; i < pad; i += nt) {
        wa[i] = i;
    }
    __syncthreads();
    for (int k = 2; k <= pad; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            __syncthreads();
            for (int t = (int) threadIdx.x; t < pad / 2; t += nt) {
                // t-th col (ascending) whose j-th bit is 0; its partner is col ^ j.
                const int col = 2 * j * (t / j) + (t % j);
                const int ixj = col ^ j;
                const int a = wa[col];
                const int b = wa[ixj];
                int do_swap = 0;
                if ((col & k) == 0) {
                    if (a >= ncols || (b < ncols && x[a] < x[b])) {
                        do_swap = 1;
                    }
                } else {
                    if (b >= ncols || (a < ncols && x[a] > x[b])) {
                        do_swap = 1;
                    }
                }
                if (do_swap) {
                    wa[col] = b;
                    wa[ixj] = a;
                }
            }
        }
    }
    __syncthreads();
    for (int i = (int) threadIdx.x; i < ncols; i += nt) {
        dst[i] = wa[i];
    }
    return LING3_OK;
}

// H1 2026-10-01: the whole route, on one block. Structure, padding rule, comparison operators,
// the group keep/n_group_used logic, the weight normalisation order and the w_scale multiply are
// all copied from ling3_route_core; only the thread mapping and the scratch residency differ.
__global__ void k_route(
        const float * x, const float * gate, const float * bias,
        int k, int n_expert, int n_used, int n_groups, int n_group_used,
        float w_scale, int norm_w,
        int * ids, float * weights,
        const float * logits_in, int * rc_out) {
    extern __shared__ char h1_smem[];
    const int nt = (int) blockDim.x;
    const int nf = ling3_route_nfloat(n_expert, n_groups);
    const int ni = ling3_route_nint(n_expert, n_groups);
    float * sf = (float *) h1_smem;
    int *   si = (int *) (sf + nf);
    if (nf < 0 || ni < 0 || n_expert <= 0 || n_expert > 1024 || n_used <= 0 ||
        n_groups <= 0 || (n_expert % n_groups) != 0) {
        if (threadIdx.x == 0) *rc_out = LING3_ERR_ARG;
        return;
    }
    const int npg   = n_expert / n_groups;
    const int pad_e = ling3_pow2_ge(n_expert);
    const int pad_g = ling3_pow2_ge(n_groups < 1 ? 1 : n_groups);
    float * probs  = sf + n_expert;
    float * sel    = sf + 2 * n_expert;
    float * gscore = sf + 3 * n_expert;
    int * ba    = si;
    int * order = si + 2 * pad_e;
    int * ga    = order + n_expert;
    int * gorder = ga + pad_g;

    for (int e = (int) threadIdx.x; e < n_expert; e += nt) {
        const float p = 1.0f / (1.0f + expf(-logits_in[e]));
        const float b = bias ? bias[e] : 0.f;
        probs[e] = p;
        sel[e] = p + b;
    }
    __syncthreads();

    if (n_groups > 1) {
        for (int g = 0; g < n_groups; ++g) {
            (void) h1_argsort_desc_par(sel + g * npg, order, ba, npg, nt);
            __syncthreads();
            if (threadIdx.x == 0) {
                gscore[g] = sel[g * npg + order[0]] + sel[g * npg + order[1]];
            }
            __syncthreads();
        }
        (void) h1_argsort_desc_par(gscore, gorder, ga, n_groups, nt);
        __syncthreads();
        for (int g = (int) threadIdx.x; g < n_groups; g += nt) {
            int keep = 0;
            for (int i = 0; i < n_group_used; ++i) {
                if (gorder[i] == g) { keep = 1; }
            }
            if (!keep) {
                for (int j = 0; j < npg; ++j) { sel[g * npg + j] = ling3_neg_inf(); }
            }
        }
        __syncthreads();
    }

    (void) h1_argsort_desc_par(sel, order, ba, n_expert, nt);
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int i = 0; i < n_used; ++i) {
            ids[i] = order[i];
            weights[i] = probs[ids[i]];
            sum += weights[i];
        }
        if (norm_w) {
            const float den = sum < LING3_WEIGHT_SUM_MIN ? LING3_WEIGHT_SUM_MIN : sum;
            for (int i = 0; i < n_used; ++i) { weights[i] = weights[i] / den; }
        }
        if (w_scale != 0.f && w_scale != 1.f) {
            for (int i = 0; i < n_used; ++i) { weights[i] *= w_scale; }
        }
        *rc_out = LING3_OK;
    }
}

// LR2 SPEC2 2026-10-03 (v2): selection-based replacement for k_route, bit-identical at
// EVERY shape k_route accepts -- in particular n_group_used = 4 of 8 groups, which is what
// the Ling-3.0-flash GGUF carries (expert_group_used_count = 4, config.cpp:189-190). v1 of
// this kernel refused n_group_used != 2, wrote no ids/weights in the model, and moved the
// scalar fingerprint to 14650fb0739d0383; v2 supports any n_group_used in [1, n_groups].
//
// What is replaced, and why it stays bit-identical:
//   * Per-group top-2 (k_route: n_groups sequential 64-wide bitonic sorts, ~168 of its
//     ~219 barrier sub-stages and the bulk of its 47.8 us): one warp top-2-pair reduction
//     per group, all groups in parallel. gscore[g] = sel[order[0]] + sel[order[1]] is a
//     function of the group's VALUE MULTISET alone: on ties the bitonic may keep different
//     ELEMENTS, but tied elements hold equal VALUES, and IEEE-754 addition of the same two
//     values in the same order (top-1 then top-2) is bitwise identical. So any exact
//     top-2-values selection reproduces gscore bit for bit -- unlike the final sort below,
//     where tie ORDER is observable in ids and only k_route's own network can reproduce it.
//   * Everything else is k_route's code VERBATIM: the group bitonic (h1_argsort_desc_par
//     over gscore, so on tied gscores the kept set gorder[0..n_group_used-1] is exactly
//     k_route's network tie order, for any n_group_used), the keep/mask loop, the final
//     512-wide bitonic, and the weight normalisation. v1's hand-rolled group arg-max is
//     gone: it hardcoded a 2-group keep set AND its smaller-index tie rule is not the
//     bitonic network's tie order (see the ITEM-2b record below in ling3_route's comment
//     and this lane's REPORT.md "What I tried that did not work").
//
// Contract notes (inputs where k_route itself is not well-defined): npg < 2 refuses with
// LING3_ERR_ARG because k_route reads stale scratch there (order[1] of a 1-wide sort);
// n_group_used outside [1, n_groups] refuses for the same reason (k_route reads stale
// gorder) -- ling3_route_core rejects both host-side, so no real shape hits either.
// Assumes NaN-free sel (sigmoid of finite logits plus finite bias): with NaNs, k_route's
// strict comparators make its order input-history dependent and nothing can match it.
// Shape: 1 block, 256 threads, 8 warps. Keeps ~51 of k_route's ~219 barrier sub-stages.

// Strict total order on (value desc, index asc). Expert indices are unique, so the order
// is TOTAL and the top-2 of any subset is well defined regardless of merge order -- that
// is what makes the warp butterfly below deterministic and lane-order independent.
__device__ __forceinline__ bool lr2_better(float av, int ai, float bv, int bi) {
    return av > bv || (av == bv && ai < bi);
}

struct lr2_top2 { float v0; int i0; float v1; int i1; };  // (v0,i0) better than (v1,i1)

// Merge two ordered pairs into the top-2 of their union: with a0 >= a1 and b0 >= b1 the
// winner is better(a0, b0), and the runner-up is better(a1, b0) if a0 won else better(a0, b1).
__device__ __forceinline__ lr2_top2 lr2_merge2(lr2_top2 a, lr2_top2 b) {
    lr2_top2 r;
    if (lr2_better(a.v0, a.i0, b.v0, b.i0)) {
        r.v0 = a.v0; r.i0 = a.i0;
        if (lr2_better(a.v1, a.i1, b.v0, b.i0)) { r.v1 = a.v1; r.i1 = a.i1; }
        else                                    { r.v1 = b.v0; r.i1 = b.i0; }
    } else {
        r.v0 = b.v0; r.i0 = b.i0;
        if (lr2_better(b.v1, b.i1, a.v0, a.i0)) { r.v1 = b.v1; r.i1 = b.i1; }
        else                                    { r.v1 = a.v0; r.i1 = a.i0; }
    }
    return r;
}

__device__ __forceinline__ lr2_top2 lr2_warp_reduce_top2(lr2_top2 t) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        lr2_top2 o;
        o.v0 = __shfl_xor(t.v0, off);
        o.i0 = __shfl_xor(t.i0, off);
        o.v1 = __shfl_xor(t.v1, off);
        o.i1 = __shfl_xor(t.i1, off);
        t = lr2_merge2(t, o);
    }
    return t;
}

__global__ void k_route_fast(
        const float * x, const float * gate, const float * bias,
        int k, int n_expert, int n_used, int n_groups, int n_group_used,
        float w_scale, int norm_w,
        int * ids, float * weights,
        const float * logits_in, int * rc_out) {
    extern __shared__ char lr2_smem[];
    const int lane   = (int) (threadIdx.x & 31);
    const int warp   = (int) (threadIdx.x >> 5);
    const int nt     = (int) blockDim.x;
    const int nwarps = nt / 32;
    const int nf = ling3_route_nfloat(n_expert, n_groups);
    const int ni = ling3_route_nint(n_expert, n_groups);
    if (nf < 0 || ni < 0 || n_expert <= 0 || n_expert > 1024 || n_used <= 0 ||
        n_used > n_expert || n_groups <= 0 || (n_expert % n_groups) != 0 ||
        n_group_used < 1 || n_group_used > n_groups) {
        if (threadIdx.x == 0) *rc_out = LING3_ERR_ARG;
        return;
    }
    const int npg = n_expert / n_groups;
    if (npg < 2) {
        if (threadIdx.x == 0) *rc_out = LING3_ERR_ARG;
        return;
    }

    // LDS layout: probs | sel | gscore floats, then k_route's int layout verbatim
    // (ba 2*pad_e | order n_expert | ga pad_g | gorder n_groups) so both verbatim
    // h1_argsort_desc_par calls index exactly what they expect. This is strictly
    // smaller than the host-allocated smem (nf floats + ni ints): floats use
    // 2*n_expert + n_groups <= nf, ints use 2*pad_e + n_expert + pad_g + n_groups <= ni.
    const int pad_e = ling3_pow2_ge(n_expert);
    const int pad_g = ling3_pow2_ge(n_groups < 1 ? 1 : n_groups);
    float * probs  = (float *) lr2_smem;
    float * sel    = probs  + n_expert;
    float * gscore = sel    + n_expert;
    int   * si_all = (int *) (gscore + n_groups);
    int   * ba     = si_all;
    int   * order  = si_all + 2 * pad_e;
    int   * ga     = order + n_expert;
    int   * gorder = ga + pad_g;

    // Phase 1 (k_route verbatim): probs = sigmoid(logits_in), sel = probs + bias.
    for (int e = (int) threadIdx.x; e < n_expert; e += nt) {
        const float p = 1.0f / (1.0f + expf(-logits_in[e]));
        const float b = bias ? bias[e] : 0.f;
        probs[e] = p;
        sel[e]   = p + b;
    }
    __syncthreads();

    if (n_groups > 1) {
        // Phase 2 (the only physical change): per-group top-2 VALUES by warp butterfly.
        // Groups stride over warps (any n_groups, not just 8); each lane owns a 2-wide
        // slice and keeps its best ordered pair; 5 shfl rounds merge the warp. Sentinel
        // indices are >= n_expert so they always lose ties to real elements, and with
        // npg >= 2 the reduced top-2 are always real, finite sel values.
        for (int g = warp; g < n_groups; g += nwarps) {
            const int base = g * npg;
            lr2_top2 t;
            t.v0 = ling3_neg_inf(); t.i0 = n_expert + base;
            t.v1 = ling3_neg_inf(); t.i1 = n_expert + base + 1;
            for (int idx = lane * 2; idx < npg; idx += 64) {
                lr2_top2 c;
                c.v0 = sel[base + idx];
                c.i0 = base + idx;
                if (idx + 1 < npg) {
                    c.v1 = sel[base + idx + 1];
                    c.i1 = base + idx + 1;
                } else {
                    c.v1 = ling3_neg_inf();
                    c.i1 = n_expert + base + idx;
                }
                if (!lr2_better(c.v0, c.i0, c.v1, c.i1)) {
                    const float tv = c.v0; const int ti = c.i0;
                    c.v0 = c.v1; c.i0 = c.i1; c.v1 = tv; c.i1 = ti;
                }
                t = lr2_merge2(t, c);
            }
            t = lr2_warp_reduce_top2(t);
            if (lane == 0) {
                // Same two values in the same add order as k_route's
                // sel[g*npg + order[0]] + sel[g*npg + order[1]] -> gscore bit-identical.
                gscore[g] = t.v0 + t.v1;
            }
        }
        __syncthreads();

        // Phase 3 (k_route verbatim): rank the group scores with the SAME bitonic
        // network, so on tied gscores the kept set is k_route's exact tie order.
        (void) h1_argsort_desc_par(gscore, gorder, ga, n_groups, nt);
        __syncthreads();

        // Phase 4 (k_route verbatim): mask every non-kept group's sel to -inf.
        for (int g = (int) threadIdx.x; g < n_groups; g += nt) {
            int keep = 0;
            for (int i = 0; i < n_group_used; ++i) {
                if (gorder[i] == g) { keep = 1; }
            }
            if (!keep) {
                for (int j = 0; j < npg; ++j) { sel[g * npg + j] = ling3_neg_inf(); }
            }
        }
        __syncthreads();
    }

    // Phase 5 (k_route verbatim): the final full bitonic. Its network tie order is part
    // of the ids contract and cannot be replaced by any arg-max (REPORT.md, ITEM 2b).
    (void) h1_argsort_desc_par(sel, order, ba, n_expert, nt);
    __syncthreads();

    // Phase 6 (k_route verbatim): ids, weights, normalisation, w_scale.
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int i = 0; i < n_used; ++i) {
            ids[i] = order[i];
            weights[i] = probs[ids[i]];
            sum += weights[i];
        }
        if (norm_w) {
            const float den = sum < LING3_WEIGHT_SUM_MIN ? LING3_WEIGHT_SUM_MIN : sum;
            for (int i = 0; i < n_used; ++i) { weights[i] = weights[i] / den; }
        }
        if (w_scale != 0.f && w_scale != 1.f) {
            for (int i = 0; i < n_used; ++i) { weights[i] *= w_scale; }
        }
        *rc_out = LING3_OK;
    }
}

// Jay 2026-10-01: router logits in parallel -- one block per expert, 64 threads stride over k, shared-memory reduce.
// LS1 PHASE7 item 2a: build24's kernel, kept VERBATIM as `k_route_logits_ref` so
// tools/ling3_route_test.cpp can run it against the new one on the device and require
// bit-identical logits. Do not "improve" it; the new one is k_route_logits, below.
__global__ void k_route_logits_ref(const float * x, const float * gate, int k, float * logits) {
    const int e = blockIdx.x;
    const float * row = gate + (size_t) e * (size_t) k;
    float s = 0.f;
    for (int i = (int) threadIdx.x; i < k; i += (int) blockDim.x) s += row[i] * x[i];
    __shared__ float red[64];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int off = 32; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) logits[e] = red[0];
}

static int launch_1d(int n) {
    return (n + 63) / 64;
}

extern "C" int ling3_dequant(int qtype, const void * rows, float * y, int k, int n,
                            hipStream_t stream) {
    size_t rb = ling3_row_bytes(qtype, k);
    if (!rows || !y || rb == 0 || n <= 0) {
        return LING3_ERR_ARG;
    }
    k_dequant<<<launch_1d(n), 64, 0, stream>>>(qtype, (const uint8_t *) rows, rb, y, k, n);
    if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ling3_chk("dequant");
}

// Below this many output rows, one-thread-per-row cannot fill a 20-CU GPU and the
// block-per-row form wins. Above it there is already plenty of parallelism and the
// original kernel is kept so the LM head's summation order is untouched.
#define LING3_GEMV_WIDE_MAX 8192
// Cap on resident blocks for the grid-strided form. 2048 blocks x 64 threads = 131k threads,
// far more than a 20-CU GPU needs, and each block still has ~38 rows of work at n=157184.
#define LING3_GEMV_GS_BLOCKS 2048

extern "C" int ling3_gemv(int qtype, const void * W, const float * x, float * y, int k, int n,
                         hipStream_t stream) {
    size_t rb = ling3_row_bytes(qtype, k);
    if (!W || !x || !y || rb == 0 || n <= 0) {
        return LING3_ERR_ARG;
    }
    // LS1 2026-10-01: ROCmFP4 (100/101) goes through the integer path. Every weight
    // matrix Ling-3.0-flash decodes is one of those two; the float kernels below stay
    // for F32 / Q8_0 / Q6_K, which only the norms and the router's F32 gate use.
    if (qtype == LING3_Q4_0_ROCMFP4 || qtype == LING3_Q4_0_ROCMFP4_FAST) {
        return ling3_gemv_fp4(qtype, W, x, y, k, n, stream);
    }
    // LVL SPEC1: the VL artifact's LM head is Q6_K, and the float path below cost
    // 15.4 ms per token on it (41 % of the decode step on halo). Its own warp-per-row
    // matvec takes that tensor; every other non-FP4 type keeps the path it had.
    if (qtype == LING3_Q6_K && (k % 256) == 0) {
        return ling3_gemv_q6k(qtype, W, x, y, k, n, stream);
    }
    if (n <= LING3_GEMV_WIDE_MAX) {
        k_gemv_wide<<<n, 64, 0, stream>>>(qtype, (const uint8_t *) W, rb, x, y, k, n);
        if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    } else {
        const int blocks = n < LING3_GEMV_GS_BLOCKS ? n : LING3_GEMV_GS_BLOCKS;
        k_gemv_wide_gs<<<blocks, 64, 0, stream>>>(qtype, (const uint8_t *) W, rb, x, y, k, n);
        if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    }
    return ling3_chk("gemv");
}

extern "C" int ling3_gemm(int qtype, const void * W, const float * x, float * y, int k, int n, int m,
                         hipStream_t stream) {
    size_t rb = ling3_row_bytes(qtype, k);
    if (!W || !x || !y || rb == 0 || n <= 0 || m <= 0) {
        return LING3_ERR_ARG;
    }
    dim3 grid(launch_1d(n), m);
    k_gemm<<<grid, 64, 0, stream>>>(qtype, (const uint8_t *) W, rb, x, y, k, n, m);
    if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ling3_chk("gemm");
}

extern "C" int ling3_swiglu(const float * gate, const float * up, float * h, int n, float limit,
                           hipStream_t stream) {
    if (!gate || !up || !h || n <= 0) {
        return LING3_ERR_ARG;
    }
    k_swiglu<<<launch_1d(n), 64, 0, stream>>>(gate, up, h, n, limit);
    if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ling3_chk("swiglu");
}

extern "C" int ling3_vec_add(const float * a, const float * b, float * y, int n,
                            hipStream_t stream) {
    if (!a || !b || !y || n <= 0) {
        return LING3_ERR_ARG;
    }
    k_add<<<launch_1d(n), 64, 0, stream>>>(a, b, y, n);
    if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    return ling3_chk("add");
}

// build24's entry point, unchanged, kept as the reference for the route test.
extern "C" int ling3_route_ref(
        const float * x, const float * gate, const float * bias,
        int k, int n_expert, int n_used, int n_groups, int n_group_used,
        float w_scale, int norm_w,
        int * ids, float * weights) {
    int nf = ling3_route_nfloat(n_expert, n_groups);
    int ni = ling3_route_nint(n_expert, n_groups);
    if (!x || !gate || !ids || !weights || nf < 0 || ni < 0) {
        return LING3_ERR_ARG;
    }
    // Jay 2026-10-01: scratch allocated once (sizes depend only on n_expert / n_groups, fixed per model) instead of three
    // hipMalloc + three hipFree per call (40 calls per token).
    static float * sf = nullptr;
    static int * si = nullptr;
    static int * rc_d = nullptr;
    static int nf_cap = 0, ni_cap = 0;
    int rc = LING3_OK;
    if (nf > nf_cap || ni > ni_cap || !sf || !si || !rc_d) {
        if (sf) hipFree(sf);
        if (si) hipFree(si);
        if (rc_d) hipFree(rc_d);
        sf = nullptr; si = nullptr; rc_d = nullptr;
        rc = ling3_hip(hipMalloc(&sf, (size_t) nf * sizeof(float) + LING3_TAIL_PAD), "route malloc f");
        if (rc == LING3_OK) rc = ling3_hip(hipMalloc(&si, (size_t) ni * sizeof(int) + LING3_TAIL_PAD), "route malloc i");
        if (rc == LING3_OK) rc = ling3_hip(hipMalloc(&rc_d, sizeof(int)), "route malloc rc");
        if (rc != LING3_OK) return rc;
        nf_cap = nf; ni_cap = ni;
        // LS1 phase 4a: hand the status word to the once-per-token device check. This read
        // used to be one blocking D2H per MoE layer -- 40 GPU drains per token, the last of
        // them -- and the value it returned was discarded by the only caller anyway.
        ling3_moe_dev_register_route_rc(rc_d);
    }
    k_route_logits_ref<<<n_expert, 64, 0, (hipStream_t) nullptr>>>(x, gate, k, sf);
    if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    {
        const size_t smem = ((size_t) nf * sizeof(float) + (size_t) ni * sizeof(int) + 255) & ~(size_t) 255;
        k_route<<<1, 256, smem, (hipStream_t) nullptr>>>(x, gate, bias, k, n_expert, n_used, n_groups, n_group_used,
                w_scale, norm_w, ids, weights, sf, rc_d);
        if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
    }
    // Launch errors are still reported synchronously -- that is not a D2H and costs nothing.
    // k_route's own status word now stays on the device until ling3_moe_dev_check() reads it,
    // once, after the last MoE layer.
    return ling3_chk("route");
}

extern "C" int ling3_expert_ffn(
        int qtype,
        const void * gate, const void * up, const void * down,
        const float * x,
        const int * ids, const float * weights,
        int n_tokens, int n_used, int n_expert,
        int k, int n_ff, float limit,
        float * y, hipStream_t stream) {
    size_t rb_k = ling3_row_bytes(qtype, k);
    size_t rb_ff = ling3_row_bytes(qtype, n_ff);
    if (!gate || !up || !down || !x || !ids || !weights || !y) {
        return LING3_ERR_ARG;
    }
    if (rb_k == 0 || rb_ff == 0 || n_tokens <= 0 || n_used <= 0 || n_expert <= 0 || n_ff <= 0) {
        return LING3_ERR_ARG;
    }
    // Jay 2026-10-01: one set of scratch buffers, grown on demand, instead of 4 hipMalloc + 4 hipFree per call (each hipFree
    // is an implicit device sync; ~84 calls per token).
    static float *g = nullptr, *u = nullptr, *h = nullptr, *p = nullptr;
    static int ff_cap = 0, k_cap = 0;
    int rc = LING3_OK;
    if (n_ff > ff_cap || k > k_cap) {
        hipFree(g); hipFree(u); hipFree(h); hipFree(p);
        g = u = h = p = nullptr;
        const int ff = n_ff > ff_cap ? n_ff : ff_cap, kk = k > k_cap ? k : k_cap;
        rc = ling3_hip(hipMalloc(&g, (size_t) ff * sizeof(float) + LING3_TAIL_PAD), "ffn g");
        if (rc == LING3_OK) rc = ling3_hip(hipMalloc(&u, (size_t) ff * sizeof(float) + LING3_TAIL_PAD), "ffn u");
        if (rc == LING3_OK) rc = ling3_hip(hipMalloc(&h, (size_t) ff * sizeof(float) + LING3_TAIL_PAD), "ffn h");
        if (rc == LING3_OK) rc = ling3_hip(hipMalloc(&p, (size_t) kk * sizeof(float) + LING3_TAIL_PAD), "ffn p");
        if (rc != LING3_OK) {
            hipFree(g); hipFree(u); hipFree(h); hipFree(p);
            g = u = h = p = nullptr; ff_cap = k_cap = 0;
            return rc;
        }
        ff_cap = ff; k_cap = kk;
    }
    rc = ling3_hip(hipMemset(y, 0, (size_t) n_tokens * (size_t) k * sizeof(float)), "ffn zero");
    const uint8_t * gate_b = (const uint8_t *) gate;
    const uint8_t * up_b = (const uint8_t *) up;
    const uint8_t * down_b = (const uint8_t *) down;
    // LS1 2026-10-01: with the integer path, every matvec wants its input as Q8_1.
    // Here both producer and consumer are inside this function, so the activation is
    // quantised once and reused: xt is shared by gate and up of every selected expert
    // (one quantisation per token, not per expert), and h is per expert (one each).
    // That is 1 + n_used quantisations per token where the naive path would do 3 *
    // n_used. Both slots are private to this call, so nothing can go stale between the
    // quantise and the matvecs that read it.
    const int fp4 = (qtype == LING3_Q4_0_ROCMFP4 || qtype == LING3_Q4_0_ROCMFP4_FAST);
    for (int t = 0; t < n_tokens && rc == LING3_OK; ++t) {
        const float * xt = x + (size_t) t * (size_t) k;
        float * yt = y + (size_t) t * (size_t) k;
        if (fp4) {
            rc = ling3_quant_q8(xt, k, LING3_Q8_SLOT_FFN_X, stream);
        }
        for (int s = 0; s < n_used && rc == LING3_OK; ++s) {
            int id = ids[t * n_used + s];
            if (id < 0 || id >= n_expert) {
                rc = LING3_ERR_ARG;
                break;
            }
            float w = weights[t * n_used + s];
            const uint8_t * gb = gate_b + (size_t) id * (size_t) n_ff * rb_k;
            const uint8_t * ub = up_b + (size_t) id * (size_t) n_ff * rb_k;
            const uint8_t * db = down_b + (size_t) id * (size_t) k * rb_ff;
            if (fp4) {
                rc = ling3_gemv_fp4_q8(qtype, gb, LING3_Q8_SLOT_FFN_X, g, k, n_ff, stream);
            } else {
                rc = ling3_gemv(qtype, gb, xt, g, k, n_ff, stream);
            }
            if (rc == LING3_OK) {
                if (fp4) {
                    rc = ling3_gemv_fp4_q8(qtype, ub, LING3_Q8_SLOT_FFN_X, u, k, n_ff, stream);
                } else {
                    rc = ling3_gemv(qtype, ub, xt, u, k, n_ff, stream);
                }
            }
            if (rc == LING3_OK) rc = ling3_swiglu(g, u, h, n_ff, limit, stream);
            if (rc == LING3_OK) {
                if (fp4) {
                    rc = ling3_quant_q8(h, n_ff, LING3_Q8_SLOT_FFN_H, stream);
                }
                if (rc == LING3_OK) {
                    if (fp4) {
                        rc = ling3_gemv_fp4_q8(qtype, db, LING3_Q8_SLOT_FFN_H, p, n_ff, k, stream);
                    } else {
                        rc = ling3_gemv(qtype, db, h, p, n_ff, k, stream);
                    }
                }
            }
            if (rc == LING3_OK) {
                k_axpy<<<launch_1d(k), 64, 0, stream>>>(w, p, yt, k);
                if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
                rc = ling3_chk("ffn axpy");
            }
        }
    }
    return rc;
}

extern "C" int ling3_dense_ffn(
        int qtype,
        const void * gate, const void * up, const void * down,
        const float * x,
        int n_tokens, int k, int n_ff, float limit,
        float * y, hipStream_t stream) {
    if (n_tokens <= 0) {
        return LING3_ERR_ARG;
    }
    int * ids = (int *) malloc((size_t) n_tokens * sizeof(int));
    float * w = (float *) malloc((size_t) n_tokens * sizeof(float));
    if (!ids || !w) {
        free(ids);
        free(w);
        return LING3_ERR_ARG;
    }
    for (int t = 0; t < n_tokens; ++t) {
        ids[t] = 0;
        w[t] = 1.f;
    }
    int rc = ling3_expert_ffn(qtype, gate, up, down, x, ids, w, n_tokens, 1, 1, k, n_ff, limit, y,
                              stream);
    free(ids);
    free(w);
    return rc;
}

// ---------------------------------------------------------------------------
// RV25 WAVE 1, F2 + F3 -- the n_used == 1, w == 1.0 expert: the shared expert of a
// MoE layer, and the whole FFN of a dense layer. Decode only (n_tokens == 1).
// Everything the streaming branch and the MTP verify use is untouched: they keep
// ling3_expert_ffn / ling3_dense_ffn exactly as they were (RV25 §5).
//
// F2 -- `xq_ready` says the caller has ALREADY quantised `x` into LING3_Q8_SLOT_FFN_X.
//   On a MoE layer the resident grouped call (ling3_expert_ffn_dev) did exactly
//   that to the same st->t, with the same kernel and the same slot, and NOTHING
//   between the two calls writes that slot (it only reads it), so re-quantising
//   rewrote identical bytes over themselves. Pass 0 where no grouped call ran
//   first -- the dense layers, which must quantise for themselves.
//
// F3 -- the down projection writes `y` directly through ling3_gemv_fp4_q8_zp,
//   whose store is y[i] = 0.0f + acc. That is bitwise what the unfused
//   hipMemset(y, 0, ...) ; k_axpy(1.0f, p, y) produced (0.0f + turns a -0.0
//   accumulator into +0.0 exactly as the zeroed y + 1.0f * p did). A raw copy
//   would not. Two launches and one k-sized scratch buffer per call go away.
//
// Returns LING3_ERR_ARG before launching anything if the fused form does not
// apply (n_tokens != 1, or a non-ROCmFP4 qtype) -- the caller then falls back to
// ling3_dense_ffn. On this path neither can happen: Step's resident MoE branch
// already refused anything but ROCmFP4.
// ---------------------------------------------------------------------------
static float * w1_g = nullptr, * w1_u = nullptr, * w1_h = nullptr;
static int w1_ff_cap = 0, w1_k_cap = 0;

extern "C" int ling3_dense_ffn_w1(
        int qtype,
        const void * gate, const void * up, const void * down,
        const float * x,
        int n_tokens, int k, int n_ff, float limit,
        float * y, int xq_ready, hipStream_t stream) {
    if (n_tokens != 1) return LING3_ERR_ARG;
    if (qtype != LING3_Q4_0_ROCMFP4 && qtype != LING3_Q4_0_ROCMFP4_FAST) return LING3_ERR_ARG;
    const size_t rb_k = ling3_row_bytes(qtype, k);
    const size_t rb_ff = ling3_row_bytes(qtype, n_ff);
    if (!gate || !up || !down || !x || !y) return LING3_ERR_ARG;
    if (rb_k == 0 || rb_ff == 0 || k <= 0 || n_ff <= 0 || (k % 32) != 0) return LING3_ERR_ARG;

    if (n_ff > w1_ff_cap || k > w1_k_cap) {
        hipFree(w1_g); hipFree(w1_u); hipFree(w1_h);
        w1_g = w1_u = w1_h = nullptr;
        const int ff = n_ff > w1_ff_cap ? n_ff : w1_ff_cap;
        const int kk = k > w1_k_cap ? k : w1_k_cap;
        int rc = ling3_hip(hipMalloc(&w1_g, (size_t) ff * sizeof(float) + LING3_TAIL_PAD), "w1 ffn g");
        if (rc == LING3_OK) rc = ling3_hip(hipMalloc(&w1_u, (size_t) ff * sizeof(float) + LING3_TAIL_PAD), "w1 ffn u");
        if (rc == LING3_OK) rc = ling3_hip(hipMalloc(&w1_h, (size_t) ff * sizeof(float) + LING3_TAIL_PAD), "w1 ffn h");
        if (rc != LING3_OK) {
            hipFree(w1_g); hipFree(w1_u); hipFree(w1_h);
            w1_g = w1_u = w1_h = nullptr; w1_ff_cap = w1_k_cap = 0;
            return rc;
        }
        w1_ff_cap = ff; w1_k_cap = kk;
    }

    int rc = LING3_OK;
    if (!xq_ready) {
        rc = ling3_quant_q8(x, k, LING3_Q8_SLOT_FFN_X, stream);
        if (rc != LING3_OK) return rc;
    }
    rc = ling3_gemv_fp4_q8(qtype, gate, LING3_Q8_SLOT_FFN_X, w1_g, k, n_ff, stream);
    if (rc == LING3_OK) rc = ling3_gemv_fp4_q8(qtype, up, LING3_Q8_SLOT_FFN_X, w1_u, k, n_ff, stream);
    if (rc == LING3_OK) rc = ling3_swiglu(w1_g, w1_u, w1_h, n_ff, limit, stream);
    if (rc == LING3_OK) rc = ling3_quant_q8(w1_h, n_ff, LING3_Q8_SLOT_FFN_H, stream);
    // F3: down writes y directly -- y[i] = 0.0f + acc, not memset + axpy.
    if (rc == LING3_OK) rc = ling3_gemv_fp4_q8_zp(qtype, down, LING3_Q8_SLOT_FFN_H, y, n_ff, k, stream);
    return rc;
}

// ---------------------------------------------------------------------------
// LS1 PHASE7 ITEM 2a -- the new router logit matvec. This is what landed, and it is
// PROVEN bit-identical: tools/ling3_route_test.cpp reports logits_diff = 0 on every
// case at both the small shape and Ling's real one.
//
// build24's k_route_logits (kept verbatim above as k_route_logits_ref) was
// <<<n_expert, 64, 0, stream>>> with a 64-deep LDS reduction tree and six __syncthreads, measured
// at 117.7 GB/s -- 48.5 % of this box's 231.06 GB/s 4-byte read roof
// (tools/ling3_roof_bench.cpp, 1 GiB working set, MALL-cold). It is a plain dense matvec
// of an F32 [512, 2560] matrix and was the slowest kernel in the decode step.
//
// One warp per row, eight warps per CTA, no LDS tree, no __syncthreads, and the
// activation staged in LDS once per CTA and shared by every row that CTA handles.
//
// BIT-IDENTICAL BY CONSTRUCTION, not by tolerance. build24's reduction was
//     s_t    = SUM over increasing j of row[t + 64j] * x[t + 64j]
//     red[t] = s_t ; then off = 32, 16, 8, 4, 2, 1 ; logits = red[0]
// With one warp per row, lane l carries BOTH old thread l's series AND old thread l+32's
// series -- each accumulated in the old per-thread order, in its OWN register -- and adds
// them, which is exactly the tree's off = 32 step; off = 16..1 become __shfl_down. Same
// values, same association. Interleaving the two series into one loop would regroup the
// additions and lose bit-identity, which is the trap the phase-6 GEMV variant walked into.
__global__ void k_route_logits(const float * x, const float * gate, int n_expert,
                               int k, float * logits) {
    extern __shared__ float xs[];
    const int lane = threadIdx.x & 31;
    const int sub = (int) (threadIdx.x >> 5);
    const int nwarps = (int) blockDim.x >> 5;
    for (int i = (int) threadIdx.x; i < k; i += (int) blockDim.x) {
        xs[i] = x[i];
    }
    __syncthreads();
    const int stride = (int) gridDim.x * nwarps;
    for (int e = blockIdx.x * nwarps + sub; e < n_expert; e += stride) {
        const float * row = gate + (size_t) e * (size_t) k;
        float s0 = 0.f;
        float s1 = 0.f;
        for (int i = lane; i < k; i += 64) {
            s0 += row[i] * xs[i];
        }
        for (int i = lane + 32; i < k; i += 64) {
            s1 += row[i] * xs[i];
        }
        float s = s0 + s1;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            s += __shfl_down(s, off);
        }
        if (lane == 0) {
            logits[e] = s;
        }
    }
}

// The shipped router. Same signature and the same results as ling3_route_ref: the
// logits kernel is proven bit-identical, and k_route -- the core -- is build24's network,
// UNCHANGED.
//
// ITEM 2b WAS ATTEMPTED AND REVERTED. A selection-based core (per-group top-2 sums, then a
// top-(n_used+1) selection, with build24's bitonic network kept as a tie-fallback) was
// bit-identical at n_expert=64 / n_used=4 / k=256 and WRONG at Ling's real shape
// (512 experts, n_used=8, k=2560) -- and forcing the fallback there produced the SAME wrong
// ids as the fast path, so the fast path was not the only thing at fault. Five bugs were
// found and fixed on the way (an unbroadcast warp-argmax threshold; a top-2 warp merge that
// dropped the other lane's runner-up; a round-0 filter starting at 0.f instead of -INFINITY;
// an in-place group mask that corrupted the fallback before it ran; and a keep-mask pointer
// one past the end of the LDS float array, which returned ids like -1072994156). The sixth
// was not found. Shipping a router that is right at one shape and wrong at the model's own
// was not an option, so the core is build24's network. REPORT7.md has the record.
extern "C" int ling3_route(
        const float * x, const float * gate, const float * bias,
        int k, int n_expert, int n_used, int n_groups, int n_group_used,
        float w_scale, int norm_w,
        int * ids, float * weights, hipStream_t stream) {
    const int nf = ling3_route_nfloat(n_expert, n_groups);
    const int ni = ling3_route_nint(n_expert, n_groups);
    if (!x || !gate || !ids || !weights || nf < 0 || ni < 0) {
        return LING3_ERR_ARG;
    }
    static float * sf = nullptr;
    static int * si = nullptr;
    static int * rc_d = nullptr;
    static int nf_cap = 0, ni_cap = 0;
    int rc = LING3_OK;
    if (nf > nf_cap || ni > ni_cap || !sf || !si || !rc_d) {
        if (sf) hipFree(sf);
        if (si) hipFree(si);
        if (rc_d) hipFree(rc_d);
        sf = nullptr; si = nullptr; rc_d = nullptr;
        rc = ling3_hip(hipMalloc(&sf, (size_t) nf * sizeof(float) + LING3_TAIL_PAD), "route malloc f");
        if (rc == LING3_OK) rc = ling3_hip(hipMalloc(&si, (size_t) ni * sizeof(int) + LING3_TAIL_PAD), "route malloc i");
        if (rc == LING3_OK) rc = ling3_hip(hipMalloc(&rc_d, sizeof(int)), "route malloc rc");
        if (rc != LING3_OK) return rc;
        nf_cap = nf; ni_cap = ni;
        // Phase 4a: the status word goes to the once-per-token device check, so the
        // decode step has no blocking D2H inside the layer loop.
        ling3_moe_dev_register_route_rc(rc_d);
    }
    {
        // One warp per row, eight warps per CTA, x staged in LDS. ceil(n_expert / 8)
        // CTAs of 256 threads covers every row once, and the grid-stride loop keeps it
        // correct for any grid.
        const int nwarps = 8;
        const int threads = nwarps * 32;
        int blocks = (n_expert + nwarps - 1) / nwarps;
        if (blocks < 1) blocks = 1;
        const size_t smem_x = ((size_t) k * sizeof(float) + 255) & ~(size_t) 255;
        k_route_logits<<<blocks, threads, smem_x, stream>>>(x, gate, n_expert, k, sf);
        if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
        if (ling3_chk("route logits") != LING3_OK) return LING3_ERR_HIP;
    }
    {
        const size_t smem = ((size_t) nf * sizeof(float) + (size_t) ni * sizeof(int) + 255) & ~(size_t) 255;
        // LR2 SPEC2 2026-10-03 (v2): k_route_fast is the default at EVERY in-contract
        // shape -- including the model's own n_group_used = 4 of 8 (v1 refused != 2 and
        // moved the scalar fingerprint to 14650fb0739d0383). LING3_ROUTE_REF=1 falls back
        // to build24's bitonic network for the route test and regression hunts; cached
        // once on first call so the per-token cost is a branch.
        static int use_ref = -1;
        if (use_ref < 0) {
            const char * e = getenv("LING3_ROUTE_REF");
            use_ref = (e && e[0] == '1') ? 1 : 0;
        }
        if (use_ref) {
            k_route<<<1, 256, smem, stream>>>(x, gate, bias, k, n_expert, n_used, n_groups, n_group_used,
                                      w_scale, norm_w, ids, weights, sf, rc_d);
            if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
        } else {
            k_route_fast<<<1, 256, smem, stream>>>(x, gate, bias, k, n_expert, n_used, n_groups, n_group_used,
                                           w_scale, norm_w, ids, weights, sf, rc_d);
            if (::gufo::hip_check::LaunchFailed()) return LING3_ERR_HIP;
        }
    }
    return ling3_chk("route");
}
