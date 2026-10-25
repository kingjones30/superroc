// ling3_mla.hip — gated-MLA kernels for Ling-3.0 (bailingmoe3), lane LK phase 2.
// Lane: LK (MiMo) — Ling-on-gufo overnight build. Do not confuse with LG's drafts.
//
// The fork's absorbed form (bailingmoe3.cpp gated-MLA branch, build_attn_mha with v_mla).
// Arithmetic:
//   rope       GGML_ROPE_TYPE_NORM, CONSECUTIVE pairs (2i, 2i+1), theta 6e6, rot 64
//   rms_norm   rsqrtf(sum(x*x)/n + eps)   (ggml-cuda rms_norm_f32; float accumulation)
//   attn       softmax over the cache positions, scale 1/sqrt(192), value = the 512 latent
//   de-absorb  wv_b, then the head-wise sigmoid gate
//
// See ling3_mla.hpp for layouts and scope.

#include "ling3_mla.hpp"

// LC1 SPEC4: LING3_TAIL_PAD (the activation-buffer tail contract) for the wo
// Q8_1 pair this module owns.
#include "ling3_quant.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define LING3_HIP_CHECK(expr)                                                        \
    do {                                                                             \
        hipError_t _err = (expr);                                                    \
        if (_err != hipSuccess) {                                                    \
            fprintf(stderr, "ling3_mla: %s failed at %s:%d: %s\n", #expr, __FILE__,  \
                    __LINE__, hipGetErrorString(_err));                              \
            abort();                                                                 \
        }                                                                            \
    } while (0)

namespace ling3 {

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

template <int BLOCK>
__device__ __forceinline__ float block_reduce_max(float v, float * smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off && smem[threadIdx.x + off] > smem[threadIdx.x]) {
            smem[threadIdx.x] = smem[threadIdx.x + off];
        }
        __syncthreads();
    }

    // Snapshot the result before returning. The caller may reuse this shared buffer for
    // the very next reduction; without this barrier the first thread back would write
    // smem[threadIdx.x] while the others are still reading smem[0].
    const float result = smem[0];
    __syncthreads();
    return result;
}


template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum(float v, float * smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) {
            smem[threadIdx.x] += smem[threadIdx.x + off];
        }
        __syncthreads();
    }

    // Snapshot the result before returning. The caller may reuse this shared buffer for
    // the very next reduction; without this barrier the first thread back would write
    // smem[threadIdx.x] while the others are still reading smem[0].
    const float result = smem[0];
    __syncthreads();
    return result;
}


// ---------------------------------------------------------------------------
// LC1 SPEC4: the decode-fusion knob and the wo Q8_1 handoff buffers.
//
// LING3_MLA_DECODE_FUSED (default 1) is read ONCE, at first use. When it is on,
// the MLA decode path runs M1's single pre-attention kernel and the Q2-MLA
// wo-Q8_1 epilogue, and the caller projects `out` with the fused activation
// instead of quantising it again. When it is off, the four old launches and the
// old quant+GEMV sequence run byte for byte -- that is the A/B the bench gate
// measures and the escape hatch if a gate ever moves.
//
// g_wo_qs / g_wo_ds are the Q8_1 activation the wo GEMV reads: gufo's internal
// layout, qs[o_dim] int8 + ds[o_dim/32] floats. They are the epilogue's output
// and the handoff's input, sized for one decode row (o_dim 4096 here).
// ---------------------------------------------------------------------------
namespace {
std::size_t g_wo_cap = 0;
int8_t * g_wo_qs = nullptr;
float *  g_wo_ds = nullptr;
int g_mla_pre_fused_launches = 0;
int g_mla_wo_q8_epilogues = 0;
}  // namespace

bool mla_decode_fused() {
    static const bool on = [] {
        const char * e = std::getenv("LING3_MLA_DECODE_FUSED");
        return !(e != nullptr && e[0] == '0' && e[1] == '\0');
    }();
    return on;
}

// Allocate (once) and report whether the wo Q8_1 pair is ready.
bool mla_wo_q8_prepare(int o_dim) {
    if (o_dim <= 0 || (o_dim % 32) != 0) return false;
    if (g_wo_qs != nullptr && g_wo_ds != nullptr && g_wo_cap >= (std::size_t) o_dim) {
        return true;
    }
    if (g_wo_qs != nullptr) { hipFree(g_wo_qs); g_wo_qs = nullptr; }
    if (g_wo_ds != nullptr) { hipFree(g_wo_ds); g_wo_ds = nullptr; }
    g_wo_cap = 0;
    // +LING3_TAIL_PAD, the same contract every other activation buffer here has.
    if (hipMalloc(&g_wo_qs, (std::size_t) o_dim + LING3_TAIL_PAD) != hipSuccess) return false;
    if (hipMalloc(&g_wo_ds, (std::size_t) (o_dim / 32) * sizeof(float) + LING3_TAIL_PAD)
        != hipSuccess) {
        hipFree(g_wo_qs); g_wo_qs = nullptr; return false;
    }
    g_wo_cap = (std::size_t) o_dim;
    return true;
}

const int8_t * mla_wo_q8_qs() { return g_wo_qs; }
const float *  mla_wo_q8_ds() { return g_wo_ds; }
int mla_pre_fused_launches() { return g_mla_pre_fused_launches; }
int mla_wo_q8_epilogues() { return g_mla_wo_q8_epilogues; }
void mla_fusion_stats_reset() { g_mla_pre_fused_launches = 0; g_mla_wo_q8_epilogues = 0; }


// ---------------------------------------------------------------------------
// RoPE, GGML_ROPE_TYPE_NORM. One thread per (pair, head, token*sequence) item,
//   256 items per block over a flat index (LW1 PHASE2: the original launched one
//   thread per BLOCK -- 524,288 blocks for a 512-token chunk's q-rope -- and the
//   block launch rate, not the math, was the cost).
//   x points at the rope section's first element; s_head / s_tok / s_seq are element strides.
//   (x[2i], x[2i+1]) rotated by pos*theta^(-2i/rot) -- NORM pairs consecutive
//   elements; (x[i], x[i+half]) would be NEOX.
// ---------------------------------------------------------------------------
// LVV: M-RoPE sections [8, 12, 12, 0] over the 32 NORM pairs. A null table
// returns the original (float) pos[t] * freq expression, so a text token is
// unchanged. Channel 0 is temporal, 1 is height, 2 is width. The pointer is a
// kernel argument so a text launch passes null and never reads the table.
static const int * h_lvv_mrope = nullptr;

__device__ __forceinline__ float lvv_rope_angle(const int * pos, int t, int pair,
                                                float freq, const int * mrope) {
    if (mrope == nullptr) {
        return (float) pos[t] * freq;
    }
    int axis = 0;
    if (pair >= 8 && pair < 20) axis = 1;
    else if (pair >= 20) axis = 2;
    const int p = mrope[(long long) pos[t] * 3 + axis];
    return (float) p * freq;
}

void mla_set_mrope(const int * device_triples_or_null) {
    h_lvv_mrope = device_triples_or_null;
}

static __global__ __launch_bounds__(256) void rope_kernel(
    float * __restrict__ x, const int * __restrict__ pos,
    float theta, int rot, int n_tokens, int n_head, int n_seqs,
    int64_t s_head, int64_t s_tok, int64_t s_seq,
    const int * __restrict__ mrope) {
    // LW1 PHASE2: flat index w -> (pair, head, token, seq) is a bijection, so
    // each thread owns exactly one item and its two elements (base[2i],
    // base[2i+1]); the in-place rotation below stays race-free and every
    // element's arithmetic is IDENTICAL to the original, statement for
    // statement.
    const int n_pairs = rot / 2;
    const long total = (long) n_pairs * (long) n_head * n_tokens * n_seqs;
    const long w = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= total) return;

    const int i = (int) (w % n_pairs);        // pair index, 0..rot/2-1
    const long wh = w / n_pairs;
    const int h = (int) (wh % n_head);        // head (1 for the shared key part)
    const long wt = wh / n_head;
    const int t = (int) (wt % n_tokens);      // token
    const int s = (int) (wt / n_tokens);      // sequence

    float * base = x + (int64_t) h * s_head + (int64_t) t * s_tok + (int64_t) s * s_seq;

    const float freq  = 1.0f / powf(theta, 2.0f * (float) i / (float) rot);
    const float angle = lvv_rope_angle(pos, t, i, freq, mrope);
    const float c = cosf(angle);
    const float sn = sinf(angle);

    // GGML_ROPE_TYPE_NORM pairs CONSECUTIVE elements: rope_norm reads
    // x[ix+0], x[ix+1] with i0 stepped by 2. (base[i], base[i+half]) is NEOX.
    const float a = base[2 * i];
    const float b = base[2 * i + 1];
    base[2 * i]     = a * c - b * sn;
    base[2 * i + 1] = a * sn + b * c;
}

// ---------------------------------------------------------------------------
// RMSNorm the 512-dim latent of every token, in place.
//   y = x * rsqrtf(mean(x*x) + eps) * w
// ---------------------------------------------------------------------------
static __global__ void latent_norm_kernel(float * __restrict__ x,
                                          const float * __restrict__ w,
                                          int kv, int n_tokens, int n_seqs,
                                          int64_t s_tok, int64_t s_seq,
                                          float eps) {
    __shared__ float s_sum[256];

    const int idx = blockIdx.x;
    const int t = idx % n_tokens;
    const int s = idx / n_tokens;

    float * row = x + (int64_t) t * s_tok + (int64_t) s * s_seq;

    float tmp = 0.0f;
    for (int i = threadIdx.x; i < kv; i += blockDim.x) {
        const float xi = row[i];
        tmp += xi * xi;
    }
    tmp = block_reduce_sum<256>(tmp, s_sum);

    const float mean  = tmp / (float) kv;
    const float scale = rsqrtf(mean + eps);

    for (int i = threadIdx.x; i < kv; i += blockDim.x) {
        row[i] = (scale * row[i]) * w[i];
    }
}

// ---------------------------------------------------------------------------
// Append this call's n_tokens kv_a rows to the cache at [n_pos - n_tokens, n_pos).
// ---------------------------------------------------------------------------
static __global__ __launch_bounds__(256) void cache_append_kernel(
    const float * __restrict__ kv_a, float * __restrict__ cache,
    int kv_a_dim, int n_tokens,
    int64_t a_tok, int64_t a_seq,
    int64_t s_pos, int64_t s_seq,
    int n_seqs, int n_pos, const int * pos_d) {
    // LW1 PHASE2: one block per (token, sequence), threads over the cache
    // columns. The original mapped blocks to columns and threads to tokens, so
    // every warp touched 32 different cache rows one 4-byte element each (stride
    // s_pos). This layout is coalesced on both sides and moves each element
    // exactly once -- a pure copy, bit-identical by construction.
    const int t = blockIdx.x;                 // token, 0..n_tokens-1
    const int s = blockIdx.y;                 // sequence
    if (t >= n_tokens) return;
    // LS2 PHASE12: n_pos is per-token and a captured graph freezes kernel arguments, so it
    // cannot arrive by value. n_pos == pos + 1 and pos is already device memory here, so
    // read the same pointer instead of adding a second one. On the plain path
    // pos_d[0] == st->pos == n_pos - 1 exactly (Step uploads it at the top of the call), so
    // npos == n_pos and nothing moves.
    const int npos = pos_d ? pos_d[0] + 1 : n_pos;
    const int base = npos - n_tokens;

    const float * src = kv_a + (int64_t) s * a_seq;
    float *       dst = cache + (int64_t) s * s_seq;

    for (int i = threadIdx.x; i < kv_a_dim; i += blockDim.x) {
        dst[(int64_t) (base + t) * s_pos + i] = src[i + (int64_t) t * a_tok];
    }
}

// ---------------------------------------------------------------------------
// LC1 SPEC4 item 1 (M1): rope(q) + rope(kv) + latent_norm + cache_append in ONE
// decode kernel, replacing four launches with one (-3 per MLA layer, -21 per
// token). Both MLA arms call the same four stages -- serial `mla_block` and
// split-K `mla_block_splitk` -- and MTP verify walks `mla_block_splitk` per row,
// so one kernel serves all of them and it MUST be bit-identical to the four.
//
// Bit-identity is by construction, statement for statement:
//   * the two RoPE passes are rope_kernel's own per-item code (same freq, same
//     angle, same c/sn, same a*c - b*sn / a*sn + b*c on the same elements);
//   * the norm is latent_norm_kernel's own loop and reduction -- one block per
//     (token, sequence), 256 threads, `tmp += xi * xi` in the same i order and
//     the same block_reduce_sum<256> tree, then (scale * row[i]) * w[i];
//   * the append is cache_append_kernel's own copy, including its `npos`
//     expression (pos_d[0] + 1 when a device position pointer is given, the
//     caller's n_pos otherwise -- the two forms differ per call site, so both
//     are passed through untouched).
// A __syncthreads() between stages replaces the kernel boundaries the four
// launches had; the values each stage writes are the values the next reads.
// One block per (token, sequence) of 256 threads, exactly like
// latent_norm_kernel, so the reduction's thread -> element map is unchanged.
// ---------------------------------------------------------------------------
static __global__ __launch_bounds__(256) void mla_pre_fused_kernel(
    float * __restrict__ q,
    float * __restrict__ kv_a,
    const float * __restrict__ kv_a_norm,
    float * __restrict__ cache,
    const int * __restrict__ pos,
    const int * __restrict__ pos_d,
    int n_tokens, int n_head, int n_seqs,
    int nope, int rot, int kv, int kv_a_dim,
    int64_t q_s_head, int64_t q_s_tok, int64_t q_s_seq,
    int64_t k_s_tok, int64_t k_s_seq,
    int64_t c_s_pos, int64_t c_s_seq,
    int n_pos, float theta, float eps,
    const int * __restrict__ mrope) {
    __shared__ float s_sum[256];

    const int idx = blockIdx.x;
    const int t = idx % n_tokens;
    const int s = idx / n_tokens;

    const int n_pairs = rot / 2;

    // --- 1. RoPE on q's rope section (rope_kernel, n_head heads) -----------
    for (int w = (int) threadIdx.x; w < n_pairs * n_head; w += (int) blockDim.x) {
        const int i = w % n_pairs;        // pair index, 0..rot/2-1
        const int h = w / n_pairs;
        float * base = q + nope + (int64_t) h * q_s_head + (int64_t) t * q_s_tok
                       + (int64_t) s * q_s_seq;

        const float freq  = 1.0f / powf(theta, 2.0f * (float) i / (float) rot);
        const float angle = lvv_rope_angle(pos, t, i, freq, mrope);
        const float c = cosf(angle);
        const float sn = sinf(angle);

        const float a = base[2 * i];
        const float b = base[2 * i + 1];
        base[2 * i]     = a * c - b * sn;
        base[2 * i + 1] = a * sn + b * c;
    }
    __syncthreads();

    // --- 2. RoPE on kv_a's key section (rope_kernel, n_head 1, s_head 0) ---
    {
        float * kbase = kv_a + kv + (int64_t) t * k_s_tok + (int64_t) s * k_s_seq;
        for (int i = (int) threadIdx.x; i < n_pairs; i += (int) blockDim.x) {
            const float freq  = 1.0f / powf(theta, 2.0f * (float) i / (float) rot);
            const float angle = lvv_rope_angle(pos, t, i, freq, mrope);
            const float c = cosf(angle);
            const float sn = sinf(angle);

            const float a = kbase[2 * i];
            const float b = kbase[2 * i + 1];
            kbase[2 * i]     = a * c - b * sn;
            kbase[2 * i + 1] = a * sn + b * c;
        }
    }
    __syncthreads();

    // --- 3. RMSNorm the latent (latent_norm_kernel, verbatim) --------------
    float * row = kv_a + (int64_t) t * k_s_tok + (int64_t) s * k_s_seq;
    float tmp = 0.0f;
    for (int i = threadIdx.x; i < kv; i += blockDim.x) {
        const float xi = row[i];
        tmp += xi * xi;
    }
    tmp = block_reduce_sum<256>(tmp, s_sum);

    const float mean  = tmp / (float) kv;
    const float scale = rsqrtf(mean + eps);

    for (int i = threadIdx.x; i < kv; i += blockDim.x) {
        row[i] = (scale * row[i]) * kv_a_norm[i];
    }
    __syncthreads();

    // --- 4. append to the cache (cache_append_kernel, verbatim) ------------
    // LS2 PHASE12: n_pos is per-token and a captured graph freezes kernel arguments, so it
    // cannot arrive by value. n_pos == pos + 1 and pos is already device memory here, so
    // read the same pointer instead of adding a second one. On the plain path
    // pos_d[0] == st->pos == n_pos - 1 exactly (Step uploads it at the top of the call), so
    // npos == n_pos and nothing moves.
    const int npos = pos_d ? pos_d[0] + 1 : n_pos;
    const int base = npos - n_tokens;

    const float * src = kv_a + (int64_t) s * k_s_tok * (int64_t) n_tokens;
    float *       dst = cache + (int64_t) s * c_s_seq;

    for (int i = threadIdx.x; i < kv_a_dim; i += blockDim.x) {
        dst[(int64_t) (base + t) * c_s_pos + i] = src[i + (int64_t) t * k_s_tok];
    }
}

// ---------------------------------------------------------------------------
// LC1 SPEC4 item 2 (Q2-MLA): the wo Q8_1 quantisation as an attention epilogue.
//
// k_q8_1_quant's recipe, statement for statement (LA1's one-warp-per-group
// form): a 32-value group is owned by one WARP, lane l holding element l; the
// scale is the signed value at the LOWEST index attaining the max |x| (the
// `(oa, oi)` butterfly resolves the tie to the same lowest index a serial scan
// takes, and (amp, idx) is a total order so the reduction order cannot change
// the outcome), d = mv / 127.f, id = d != 0.f ? 1.0f / d : 0.f, and each lane
// stores roundf(v * id) with the id it broadcasts from lane 0. An all-zero
// group still gets d = 0, id = 0, codes all 0. The layout is gufo's internal
// Q8_1 contract (qs[k] int8 + ds[k/32] floats), which has NO ggml `s` field to
// preserve -- the ds/qs pair is everything the wo GEMV reads.
//
// The value v must be the EXACT float the unfused path would have quantised,
// so the callers hand back the value they just stored (the store/load round
// trip is exact) rather than a register copy taken before the store.
__device__ __forceinline__ void mla_q8_1_group_warp(float v, int group,
                                                    int8_t * __restrict__ qs,
                                                    float * __restrict__ ds) {
    const int lane = (int) threadIdx.x & 31;
    float ma = fabsf(v), mv = v;
    int   mi = lane;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float oa = __shfl_down(ma, off);
        const float ov = __shfl_down(mv, off);
        const int   oi = __shfl_down(mi, off);
        if (oa > ma || (oa == ma && oi < mi)) {
            ma = oa; mv = ov; mi = oi;
        }
    }
    const float d  = mv / 127.f;
    const float id = d != 0.f ? 1.0f / d : 0.f;
    if (lane == 0) {
        ds[group] = d;
    }
    const float id_b = __shfl(id, 0);
    qs[group * 32 + lane] = (int8_t) roundf(v * id_b);
}

// The epilogue itself: block `blk` covers outputs [blk*head_v, (blk+1)*head_v)
// of the [o_dim] wo activation. head_v is 128 = 4 whole 32-groups, so one block
// writes 4 complete quant groups and every group lives inside one block -- no
// cross-block reduction, only this warp butterfly. Requires one output per
// thread (blockDim.x >= head_v, true at both launch sites: 128 and 256), so
// warp w owns the 32 consecutive outputs [32w, 32w+32) -- exactly
// k_q8_1_quant's mapping.
__device__ __forceinline__ void mla_wo_q8_epilogue(float * __restrict__ orow,
                                                   int head_v, int blk,
                                                   int8_t * __restrict__ wo_qs,
                                                   float * __restrict__ wo_ds) {
    const int r = (int) threadIdx.x;
    if (r >= head_v) return;
    mla_q8_1_group_warp(orow[r], blk * (head_v / 32) + (r >> 5), wo_qs, wo_ds);
}

// ---------------------------------------------------------------------------
// Attention + wk_b absorb + wv_b de-absorb + head gate.
//
// One block per (head, sequence); the block loops over its n_tokens.
//   q_abs[r] = sum_i wk_b[(h*kv + r)*nope + i] * q[h*head_k + i]          r in [0, kv)
//   score[p] = (dot(q_abs, cache[p][0:kv]) + dot(q_pe, cache[p][kv:kv_a])) * scale
//   w        = softmax(score)
//   latent_o = sum_p w[p] * cache[p][0:kv]
//   out[r]   = sigmoid(gate) * sum_j wv_b[(h*head_v + r)*kv + j] * latent_o[j]
//
// LC1 SPEC4: wo_qs / wo_ds are the Q2-MLA epilogue's destination (see
// mla_q8_1_group_warp); nullptr keeps the pre-Q2 kernel exactly.
// ---------------------------------------------------------------------------
static __global__ void attn_kernel(const float * __restrict__ q,
                                   const float * __restrict__ gate,
                                   const float * __restrict__ cache,
                                   const float * __restrict__ wk_b,
                                   const float * __restrict__ wv_b,
                                   float * __restrict__ scores,
                                   float * __restrict__ out,
                                   int n_head, int head_k, int head_v,
                                   int nope, int rot, int kv, int kv_a_dim,
                                   int n_tokens, int n_pos, float scale,
                                   int64_t q_tok, int64_t q_seq,
                                   int64_t g_tok, int64_t g_seq,
                                   int64_t o_tok, int64_t o_seq,
                                   int64_t s_pos, int64_t s_seq,
                                   const int * pos_d,
                                   int8_t * __restrict__ wo_qs,
                                   float * __restrict__ wo_ds) {
    __shared__ float q_abs[512];
    __shared__ float latent_o[512];
    __shared__ float red[256];

    const int h = blockIdx.x;
    const int s = blockIdx.y;

    const float * qrow0 = q + (int64_t) h * head_k;                 // token 0 of this head
    float *       orow0 = out + (int64_t) h * head_v;
    const float * crow0 = cache + (int64_t) s * s_seq;
    // LS2 PHASE12: see cache_append_kernel. One device pointer, not two.
    const int npos = pos_d ? pos_d[0] + 1 : n_pos;

    float *       srow  = scores + ((int64_t) s * n_head + h) * npos;

    for (int t = 0; t < n_tokens; ++t) {
        const float * qh = qrow0 + (int64_t) t * q_tok + (int64_t) s * q_seq;
        const float * qp = qh + nope;

        // absorb q_nope into the latent space
        for (int r = threadIdx.x; r < kv; r += blockDim.x) {
            const float * wrow = wk_b + ((int64_t) h * kv + r) * nope;
            float acc = 0.0f;
            for (int i = 0; i < nope; ++i) {
                acc += wrow[i] * qh[i];
            }
            q_abs[r] = acc;
        }
        __syncthreads();

        // Causal window: token t sees cache positions [0, base + t], where
        // base = n_pos - n_tokens is where this call's tokens start. The fork masks
        // with a causal kq_mask; attending over all n_pos would leak future positions.
        const int p_limit = (npos - n_tokens) + t + 1;

        // scores over the visible cached positions
        for (int p = threadIdx.x; p < p_limit; p += blockDim.x) {
            const float * cp = crow0 + (int64_t) p * s_pos;
            float sum = 0.0f;
            for (int i = 0; i < kv; ++i) {
                sum += q_abs[i] * cp[i];
            }
            for (int i = 0; i < rot; ++i) {
                sum += qp[i] * cp[kv + i];
            }
            srow[p] = sum * scale;
        }
        __syncthreads();

        // softmax
        float lmax = -INFINITY;
        for (int p = threadIdx.x; p < p_limit; p += blockDim.x) {
            lmax = fmaxf(lmax, srow[p]);
        }
        lmax = block_reduce_max<256>(lmax, red);

        float lsum = 0.0f;
        for (int p = threadIdx.x; p < p_limit; p += blockDim.x) {
            const float e = expf(srow[p] - lmax);
            srow[p] = e;
            lsum += e;
        }
        lsum = block_reduce_sum<256>(lsum, red);
        const float inv = 1.0f / lsum;
        for (int p = threadIdx.x; p < p_limit; p += blockDim.x) {
            srow[p] *= inv;
        }
        __syncthreads();

        // weighted latent
        for (int j = threadIdx.x; j < kv; j += blockDim.x) {
            float acc = 0.0f;
            for (int p = 0; p < p_limit; ++p) {
                acc += srow[p] * crow0[(int64_t) p * s_pos + j];
            }
            latent_o[j] = acc;
        }
        __syncthreads();

        // de-absorb through wv_b and apply the head-wise gate
        const float g = sigmoid_f(gate[h + (int64_t) t * g_tok + (int64_t) s * g_seq]);
        float * orow = orow0 + (int64_t) t * o_tok + (int64_t) s * o_seq;
        for (int r = threadIdx.x; r < head_v; r += blockDim.x) {
            const float * wrow = wv_b + ((int64_t) h * head_v + r) * kv;
            float acc = 0.0f;
            for (int j = 0; j < kv; ++j) {
                acc += wrow[j] * latent_o[j];
            }
            orow[r] = g * acc;
        }
#if defined(LING3_MLA_DBG)
        if (h == 0 && s == 0 && threadIdx.x == 0) {
            printf("KERNEL t=%d p_limit=%d srow=%+.6f/%+.6f/%+.6f lat0=%+.6f out0=%+.6f\n",
                   t, p_limit, srow[0], p_limit > 1 ? srow[1] : 0.0f, p_limit > 2 ? srow[2] : 0.0f,
                   latent_o[0], orow[0]);
        }
        __syncthreads();
#endif
        __syncthreads();
    }

    // LC1 SPEC4 Q2-MLA: quantise the wo activation here instead of paying a
    // separate k_q8_1_quant launch (-1 per MLA layer, -7 per token). Decode
    // only (one token, one sequence -- n_seqs is the grid's y, so one sequence
    // means gridDim.y == 1): prefill and any multi-token call leave wo_qs null
    // and keep their own quantiser / GEMM. The values come back out of `out` --
    // the same bytes the unfused k_q8_1_quant would read -- after the loop's
    // trailing barrier.
    if (wo_qs != nullptr && n_tokens == 1 && gridDim.y == 1u) {
        __syncthreads();
        mla_wo_q8_epilogue(out + (int64_t) h * head_v, head_v, h, wo_qs, wo_ds);
    }
}

// ---------------------------------------------------------------------------
// host launchers
// ---------------------------------------------------------------------------
void mla_rope(const MlaShape & sh, float * x, int64_t s_head, int n_head,
              int n_tokens, int64_t s_tok, int64_t s_seq, int n_seqs,
              const int * pos, hipStream_t stream) {
#if defined(LING3_CHECK_DEVICE_PTRS)
    // `pos` is read by the kernel, so it must be DEVICE memory. A host address here is
    // valid on a CPU and a GCVM_L2_PROTECTION_FAULT on a GPU -- the exact bug this caught.
    LING3_CHECK_DEVICE_PTR(pos, (size_t) n_tokens * sizeof(int), "mla_rope pos");
    LING3_CHECK_DEVICE_PTR(x, sizeof(float), "mla_rope x");
#endif
    const int n_head_eff = n_head > 0 ? n_head : 1;
    const long total = (long) (sh.rot / 2) * n_head_eff * n_tokens * n_seqs;
    if (total <= 0) return;
    const dim3 grid((unsigned) ((total + 255) / 256), 1, 1);
    hipLaunchKernelGGL(rope_kernel, grid, 256, 0, stream, x, pos, sh.rope_theta, sh.rot,
                       n_tokens, n_head_eff, n_seqs, s_head, s_tok, s_seq, h_lvv_mrope);
    LING3_HIP_CHECK(hipGetLastError());
}

void mla_latent_norm(const MlaShape & sh, const MlaScalars & sc, const MlaWeights & w,
                     float * kv_a_inout, int n_tokens, int n_seqs,
                     int64_t s_tok, int64_t s_seq, hipStream_t stream) {
    const dim3 grid(n_tokens * n_seqs);
    hipLaunchKernelGGL(latent_norm_kernel, grid, 256, 0, stream,
                       kv_a_inout, w.kv_a_norm, sh.kv, n_tokens, n_seqs,
                       s_tok, s_seq, sc.rms_eps);
    LING3_HIP_CHECK(hipGetLastError());
}

void mla_cache_append(const MlaShape & sh, const MlaCache & cache,
                      const float * kv_a_in, int n_tokens, int64_t s_tok,
                      int n_seqs, int n_pos, hipStream_t stream, const int * pos_d) {
    const dim3 grid(n_tokens > 0 ? (unsigned) n_tokens : 1u, n_seqs);
    hipLaunchKernelGGL(cache_append_kernel, grid, 256, 0, stream,
                       kv_a_in, cache.kv, sh.kv_a(), n_tokens,
                       s_tok, s_tok * n_tokens,
                       cache.s_pos, cache.s_seq, n_seqs, n_pos, pos_d);
    LING3_HIP_CHECK(hipGetLastError());
}

// ---------------------------------------------------------------------------
// LC1 SPEC4 M1: the four stages above in one launch. Same arguments the four
// launchers take, collected -- including both `n_pos` and `pos_d`, because
// cache_append_kernel derives `npos` from whichever the CALL SITE used and the
// two block entries differ there (split-K passes no pos_d, the serial arm does).
// `pos` is DEVICE memory, as mla_rope's contract requires.
// ---------------------------------------------------------------------------
void mla_pre_fused(const MlaShape & sh, const MlaScalars & sc, const MlaWeights & w,
                   const MlaCache & cache, const MlaInputs & in,
                   const int * pos, int n_pos, const int * pos_d,
                   hipStream_t stream) {
#if defined(LING3_CHECK_DEVICE_PTRS)
    LING3_CHECK_DEVICE_PTR(pos, (size_t) (in.n_tokens > 0 ? in.n_tokens : 1) * sizeof(int),
                           "mla_pre_fused pos");
    LING3_CHECK_DEVICE_PTR(in.q.p, sizeof(float), "mla_pre_fused q");
    LING3_CHECK_DEVICE_PTR(in.kv_a.p, sizeof(float), "mla_pre_fused kv_a");
#endif
    if (pos == nullptr) {
        fprintf(stderr, "ling3_mla: mla_pre_fused needs a DEVICE pointer to the positions\n");
        abort();
    }
    const int T = in.n_tokens > 0 ? in.n_tokens : 0;
    const int S = in.n_seqs > 0 ? in.n_seqs : 1;
    if (T <= 0) return;
    const dim3 grid((unsigned) (T * S), 1, 1);
    hipLaunchKernelGGL(mla_pre_fused_kernel, grid, 256, 0, stream,
                       const_cast<float *>(in.q.p),
                       const_cast<float *>(in.kv_a.p),
                       w.kv_a_norm, cache.kv,
                       pos, pos_d, T, sh.n_head, S,
                       sh.nope, sh.rot, sh.kv, sh.kv_a(),
                       sh.head_k, in.q.s_tok, in.q.s_seq,
                       in.kv_a.s_tok, in.kv_a.s_seq,
                       cache.s_pos, cache.s_seq,
                       n_pos, sh.rope_theta, sc.rms_eps, h_lvv_mrope);
    LING3_HIP_CHECK(hipGetLastError());
    g_mla_pre_fused_launches += 1;
}

size_t mla_block_scratch_bytes(const MlaShape & sh, int n_seqs, int n_pos) {
    return (size_t) sh.n_head * (size_t) n_seqs * (size_t) n_pos * sizeof(float);
}

// ---------------------------------------------------------------------------
// LS3 2026-10-01: token-parallel attention for batched prefill.
// attn_kernel above runs one block per (head, sequence) and loops tokens
// serially inside -- 32 blocks for the whole chunk. This variant runs one
// WARP per (head, token): a block covers LS3_MLA_WPB tokens of one head
// (grid [n_head][ceil(tok_count/WPB)], 8 warps), so the cache tile staged
// into LDS is shared by 8 (head, token) pairs.
//   P2 rev 1 walked the cache in PT-position tiles staged by COALESCED loads
// with an online (flash-style) softmax, but kept one block per pair: every
// block still read the whole visible cache itself -- ~29 GB of cache reads
// plus ~4 GB each of wk_b / wv_b per layer per 512-token chunk.
//   P2 rev 2 (this kernel) shares the tile across 8 pairs, cutting that
// traffic 8x. Per pair the arithmetic is rev 1's: warp-lane-strided dots
// (i = lane + 32*jj ascending, then shfl_down 16..1 -- the same values as
// rev 1's score loop), the same online softmax bookkeeping in the same pi
// order, and the same per-j latent accumulation; the de-absorb publishes the
// latent to LDS and keeps attn_kernel's serial j-ascending dot (the first
// version warp-reduced the j-partitioned partial dots of different output
// elements -- wrong, and caught by the mla unit gate at t=0).
// Register layout per lane (kv = 512, rot = 64, head_v = 128 -- the model's
// shapes, same assumption rev 1's fixed 576-wide tile made):
//   q_abs[16]  r = lane + 32*jj   (the absorbed query)
//   q_pe[2]    i = kv + lane + 32*pe (the rope part)
//   lat[16]    j = lane + 32*jj   (the running weighted latent)
// ---------------------------------------------------------------------------
#define LS3_MLA_PT 8    // cache positions per LDS tile
#define LS3_MLA_WPB 8   // (head, token) pairs per block -- one warp each

static __global__ __launch_bounds__(256) void attn_tok_kernel(
        const float * __restrict__ q,
        const float * __restrict__ gate,
        const float * __restrict__ cache,
        const float * __restrict__ wk_b,
        const float * __restrict__ wv_b,
        float * __restrict__ scores,
        float * __restrict__ out,
        int n_head, int head_k, int head_v,
        int nope, int rot, int kv, int kv_a_dim,
        int tok_base, int tok_count, int chunk_tokens, int n_pos,
        float scale,
        int64_t q_tok, int64_t q_seq,
        int64_t g_tok, int64_t g_seq,
        int64_t o_tok, int64_t o_seq,
        int64_t s_pos, int64_t s_seq) {
    __shared__ float tile[LS3_MLA_PT][576 + 4];

    const int h = blockIdx.x;
    const int s = 0;
    const int kv_a = kv + rot;
    const int w = (int) threadIdx.x >> 5;    // warp = pair id 0..WPB-1
    const int lane = (int) threadIdx.x & 31;
    const int t = tok_base + (int) blockIdx.y * LS3_MLA_WPB + w;
    // The last block of a sub-batch carries up to WPB-1 dead pairs. They must
    // still reach both __syncthreads() per tile, so nothing here returns
    // early; `active` only guards the per-pair work (warp-uniform). Without
    // it a dead pair would write the NEXT sub-batch's out rows a second time.
    const bool active = t < tok_base + tok_count;

    const float * crow0 = cache + (int64_t) s * s_seq;
    // kept for the debug build's score dump only
    float * srow = scores + (((int64_t) blockIdx.y * n_head + h)) * n_pos;
    (void) srow;
    (void) kv_a_dim;

    // per-pair state, all in registers
    float q_abs[16];
    float q_pe[2];
    float lat[16];
    float m_run = -INFINITY;
    float lsum = 0.0f;

    if (active) {
        const float * qh = q + (int64_t) h * head_k + (int64_t) t * q_tok +
                           (int64_t) s * q_seq;
        const float * qp = qh + nope;
        // absorb q_nope into the latent space -- the same serial
        // i-ascending dot per r as rev 1, so the values are identical
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            const int r = lane + 32 * jj;
            const float * wrow = wk_b + ((int64_t) h * kv + r) * nope;
            float acc = 0.0f;
            for (int i = 0; i < nope; ++i) {
                acc += wrow[i] * qh[i];
            }
            q_abs[jj] = acc;
        }
#pragma unroll
        for (int pe = 0; pe < 2; ++pe) {
            q_pe[pe] = qp[lane + 32 * pe];
        }
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            lat[jj] = 0.0f;
        }
    }

    // Causal window: token t sees cache positions [0, base + t], where
    // base = n_pos - chunk_tokens is where the whole chunk starts. The chunk
    // size is an explicit parameter: deriving it from the grid would be wrong
    // for every sub-batch after the first (that bug leaked future positions
    // into the softmax; the model-level gate caught it).
    const int p_limit = active ? (n_pos - chunk_tokens) + t + 1 : 0;
    // The block stages tiles for its LAST active pair; every pair clips the
    // accumulation to its own p_limit. A per-pair staging bound would desync
    // the warps' tile phases (warp A on tile k while warp B still reads
    // tile k-1) -- the loop and the staging are block-uniform.
    const int t_hi = tok_base + (int) blockIdx.y * LS3_MLA_WPB + LS3_MLA_WPB - 1;
    const int t_last = t_hi < tok_base + tok_count - 1 ? t_hi
                                                      : tok_base + tok_count - 1;
    const int p_hi = (n_pos - chunk_tokens) + t_last + 1;

    for (int p0 = 0; p0 < p_hi; p0 += LS3_MLA_PT) {
        // ---- stage the tile coalesced: 8 positions x kv_a floats ----
        const int npt = p_hi - p0 < LS3_MLA_PT ? p_hi - p0 : LS3_MLA_PT;
        for (int e = (int) threadIdx.x; e < npt * kv_a; e += (int) blockDim.x) {
            const int pi = e / kv_a;
            const int i = e % kv_a;
            tile[pi][i] = crow0[(int64_t) (p0 + pi) * s_pos + i];
        }
        __syncthreads();

        if (active && p0 < p_limit) {
            const int own = p_limit - p0 < LS3_MLA_PT ? p_limit - p0 : LS3_MLA_PT;
            // ---- this pair's scores over the tile ----
            float sc[LS3_MLA_PT];
#pragma unroll
            for (int pi = 0; pi < LS3_MLA_PT; ++pi) {
                sc[pi] = -INFINITY;
            }
#pragma unroll
            for (int pi = 0; pi < LS3_MLA_PT; ++pi) {
                if (pi < own) {
                    float acc = 0.0f;
#pragma unroll
                    for (int jj = 0; jj < 18; ++jj) {   // 16 latent + 2 rope dims
                        const int i = lane + 32 * jj;
                        const float qv = (i < kv) ? q_abs[jj] : q_pe[jj - 16];
                        acc += qv * tile[pi][i];
                    }
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        acc += __shfl_down(acc, off, 32);   // width 32: correct under wave64 too
                    }
                    if (lane == 0) {
                        sc[pi] = acc * scale;
                    }
                }
            }
            // lane 0 holds the scores; every lane mirrors the bookkeeping
            // (identical values, no divergence)
#pragma unroll
            for (int pi = 0; pi < LS3_MLA_PT; ++pi) {
                sc[pi] = __shfl(sc[pi], 0, 32);
            }
            float m_new = m_run;
#pragma unroll
            for (int pi = 0; pi < LS3_MLA_PT; ++pi) {
                m_new = fmaxf(m_new, sc[pi]);
            }
            const float alpha = m_run == -INFINITY ? 0.0f : expf(m_run - m_new);
            float ladd = 0.0f;
#pragma unroll
            for (int pi = 0; pi < LS3_MLA_PT; ++pi) {
                sc[pi] = expf(sc[pi] - m_new);   // weights of this tile
                ladd += sc[pi];
            }
            lsum = lsum * alpha + ladd;
            m_run = m_new;
            // ---- rescale and accumulate the weighted latent ----
#pragma unroll
            for (int jj = 0; jj < 16; ++jj) {
                lat[jj] *= alpha;
#pragma unroll
                for (int pi = 0; pi < LS3_MLA_PT; ++pi) {
                    if (pi < own) {
                        lat[jj] += sc[pi] * tile[pi][lane + 32 * jj];
                    }
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        // Publish the running latent to the (now dead) tile buffer. The
        // de-absorb needs every j of the latent for each output element, but
        // lat is j-partitioned across the warp's lanes (j = lane + 32*jj) --
        // a warp-reduce there would sum the partial dots of DIFFERENT output
        // elements (the first version of this kernel did exactly that: the
        // shfl_down chain mixed lanes holding w-rows of different r, and only
        // orow[0/32/64/96] was ever written; the unit gate caught it as a
        // ~100 %-of-scale diff at t=0, where the softmax weight is exactly 1).
        float * latrow = tile[w];
        const float inv = 1.0f / lsum;
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            latrow[lane + 32 * jj] = lat[jj] * inv;
        }
    }
    __syncthreads();   // outside `active`: dead pairs must reach it
    if (active) {
        const float g = sigmoid_f(gate[h + (int64_t) t * g_tok + (int64_t) s * g_seq]);
        float * orow = out + (int64_t) h * head_v + (int64_t) t * o_tok +
                       (int64_t) s * o_seq;
        const float * latrow = tile[w];
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {   // head_v = 128 = 4 x 32
            const int r = lane + 32 * rr;
            const float * wrow = wv_b + ((int64_t) h * head_v + r) * kv;
            float acc = 0.0f;
            // serial j-ascending dot, the same statement order as
            // attn_kernel's de-absorb -- no cross-lane reduction needed
            for (int j = 0; j < kv; ++j) {
                acc += wrow[j] * latrow[j];
            }
            orow[r] = g * acc;
        }
    }
#if defined(LING3_MLA_DBG)
    if (h == 0 && blockIdx.y == 0 && w == 0 && lane == 0) {
        printf("TOK t=%d p_limit=%d lsum=%f\n", t, p_limit, lsum);
    }
#endif
}

void mla_attn_tok(const MlaShape & sh, const MlaScalars & sc, const MlaWeights & w,
                  const MlaCache & cache, int n_pos, const MlaInputs & in,
                  const View & out, float * scores,
                  int tok_base, int tok_count, int chunk_tokens,
                  hipStream_t stream) {
    const float kq_scale = sc.kq_scale != 0.0f ? sc.kq_scale
                                               : 1.0f / sqrtf((float) sh.head_k);
    const dim3 grid(sh.n_head,
                    tok_count > 0 ? (unsigned) ((tok_count + LS3_MLA_WPB - 1) /
                                                LS3_MLA_WPB)
                                  : 1u);
    hipLaunchKernelGGL(attn_tok_kernel, grid, 256, 0, stream,
                       in.q.p, in.gate.p, cache.kv, w.wk_b, w.wv_b, scores, out.p,
                       sh.n_head, sh.head_k, sh.head_v,
                       sh.nope, sh.rot, sh.kv, sh.kv_a(),
                       tok_base, tok_count, chunk_tokens, n_pos, kq_scale,
                       in.q.s_tok, in.q.s_seq,
                       in.gate.s_tok, in.gate.s_seq,
                       out.s_tok, out.s_seq,
                       cache.s_pos, cache.s_seq);
    LING3_HIP_CHECK(hipGetLastError());
}

void mla_attn(const MlaShape & sh, const MlaScalars & sc, const MlaWeights & w,
              const MlaCache & cache, int n_pos,
              const MlaInputs & in, const View & out,
              float * scores, hipStream_t stream, const int * pos_d,
              int8_t * wo_qs, float * wo_ds) {
    const float scale = sc.kq_scale != 0.0f ? sc.kq_scale
                                            : 1.0f / sqrtf((float) sh.head_k);

#if defined(LING3_CHECK_DEVICE_PTRS)
    LING3_CHECK_DEVICE_PTR(scores, mla_block_scratch_bytes(sh, in.n_seqs, n_pos), "mla_attn scores");
    LING3_CHECK_DEVICE_PTR(cache.kv, sizeof(float), "mla_attn cache.kv");
    LING3_CHECK_DEVICE_PTR(out.p, sizeof(float), "mla_attn out");
    LING3_CHECK_DEVICE_PTR(in.q.p, sizeof(float), "mla_attn q");
    LING3_CHECK_DEVICE_PTR(in.gate.p, sizeof(float), "mla_attn gate");
    LING3_CHECK_DEVICE_PTR(w.wk_b, sizeof(float), "mla_attn wk_b");
    LING3_CHECK_DEVICE_PTR(w.wv_b, sizeof(float), "mla_attn wv_b");
#endif

    // Q2-MLA: the epilogue owns one output per thread, which needs blockDim >=
    // head_v. Both launch sites below satisfy it (256 here, 128 on the split-K
    // reduce) for the model's head_v = 128; refuse rather than quantise wrong.
    if (wo_qs != nullptr && (sh.head_v > 256 || (sh.head_v % 32) != 0)) {
        fprintf(stderr, "ling3_mla: wo Q8_1 epilogue needs head_v <= 256 and a "
                        "multiple of 32, got %d\n", sh.head_v);
        abort();
    }
    if (wo_qs != nullptr) {
        g_mla_wo_q8_epilogues += 1;
    }

    const dim3 grid(sh.n_head, in.n_seqs);
    hipLaunchKernelGGL(attn_kernel, grid, 256, 0, stream,
                       in.q.p, in.gate.p, cache.kv, w.wk_b, w.wv_b, scores, out.p,
                       sh.n_head, sh.head_k, sh.head_v,
                       sh.nope, sh.rot, sh.kv, sh.kv_a(),
                       in.n_tokens, n_pos, scale,
                       in.q.s_tok, in.q.s_seq,
                       in.gate.s_tok, in.gate.s_seq,
                       out.s_tok, out.s_seq,
                       cache.s_pos, cache.s_seq, pos_d, wo_qs, wo_ds);
    LING3_HIP_CHECK(hipGetLastError());
}

void mla_block_scratch(const MlaShape & sh, const MlaScalars & sc, const MlaWeights & w,
                       const MlaCache & cache, const MlaInputs & in,
                       const int * pos, int n_pos, const View & out,
                       void * scratch, size_t scratch_bytes, hipStream_t stream,
                       const int * pos_d) {
    const size_t need = mla_block_scratch_bytes(sh, in.n_seqs, n_pos);
    if (scratch == nullptr || scratch_bytes < need) {
        fprintf(stderr, "ling3_mla: scratch %zu bytes given, %zu needed\n", scratch_bytes, need);
        abort();
    }
    mla_block(sh, sc, w, cache, in, pos, n_pos, out, (float *) scratch, stream, pos_d);
}

// ---------------------------------------------------------------------------
// SPEC6: row-transposed weight copies for the decode path.
// dst[h][i][r] = src[h][r][i]; one thread per (h, r) column. Values are
// copied, never recomputed, so the bits are exactly the source's.
// ---------------------------------------------------------------------------
static __global__ __launch_bounds__(256) void w_transpose_kernel(
        const float * __restrict__ src, float * __restrict__ dst,
        int n_head, int rows, int cols) {
    const int h = blockIdx.y;
    const int r0 = blockIdx.x * blockDim.x + threadIdx.x;
    for (int r = r0; r < rows; r += gridDim.x * blockDim.x) {
        const float * srow = src + ((int64_t) h * rows + r) * cols;
        float * dcol = dst + (int64_t) h * cols * rows + r;
        for (int i = 0; i < cols; ++i) {
            dcol[(int64_t) i * rows] = srow[i];
        }
    }
}

void mla_weights_transpose(const float * src, float * dst, int n_head,
                           int rows, int cols, hipStream_t stream) {
    const unsigned nb = (unsigned) ((rows + 255) / 256);
    hipLaunchKernelGGL(w_transpose_kernel, dim3(nb, (unsigned) n_head), 256, 0,
                       stream, src, dst, n_head, rows, cols);
    LING3_HIP_CHECK(hipGetLastError());
}

// ---------------------------------------------------------------------------
// LS3 PHASE4: split-K ("flash decoding") attention for the decode step.
//
// attn_kernel runs one block per head -- 32 blocks for the whole model --
// and walks the cache serially, so decode throughput falls as the context
// grows. Here each (head, slice) workgroup computes a partial
// (max, sum, weighted latent) over its slice of the cache and a reduce
// kernel combines the partials in FIXED ascending slice order (no atomics:
// deterministic run-to-run). Per-position arithmetic is attn_kernel's own:
// the same q_abs absorption, the same 576-wide score dot, the same
// j-ascending latent accumulation; the only difference is the flash-style
// per-slice max/denominator, which the reduce rescales away exactly.
// ---------------------------------------------------------------------------

// The absorbed q, hoisted out of the per-slice blocks: recomputing it in
// every (head, slice) workgroup re-reads wk_b (262 KB/head) k_slices times
// per token, which the first profile showed as a co-bottleneck at 13K
// positions.
static __global__ __launch_bounds__(256) void attn_sk_prep_kernel(
        const float * __restrict__ q,
        const float * __restrict__ wk_b,
        const float * __restrict__ wk_bT,
        float * __restrict__ q_abs_out,
        int head_k, int nope, int kv) {
    // SPEC6: blockIdx.x is a row group, blockIdx.y the head, so one thread
    // owns exactly one of the kv rows (no idle lanes, 2 groups of 256 at
    // kv = 512 instead of 32 blocks x 2 rows each).
    const int h = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= kv) return;
    const float * qh = q + (int64_t) h * head_k;   // decode: t = 0
    float a = 0.0f;
    if (wk_bT != nullptr) {
        // Coalesced: column i of the head lives 512 floats further on, so a
        // wave's 32 threads read 32 CONSECUTIVE floats at every i step.
        // Same statement, same ascending i, same values -- only w's address
        // changes, so the bits are the original kernel's.
        const float * wcol = wk_bT + (int64_t) h * nope * kv + r;
        for (int i = 0; i < nope; ++i) {
            a += wcol[i * kv] * qh[i];
        }
    } else {
        const float * wrow = wk_b + ((int64_t) h * kv + r) * nope;
        for (int i = 0; i < nope; ++i) {
            a += wrow[i] * qh[i];
        }
    }
    q_abs_out[(size_t) h * kv + r] = a;
}

static __global__ __launch_bounds__(256) void attn_sk_partial_kernel(
        const float * __restrict__ q,
        const float * __restrict__ cache,
        const float * __restrict__ q_abs_dev,
        float * __restrict__ part_m, float * __restrict__ part_l,
        float * __restrict__ part_acc,
        int n_head, int head_k, int nope, int rot, int kv, int kv_a_dim,
        int n_pos, float scale,
        int64_t q_tok, int64_t q_seq,
        int64_t s_pos, int64_t s_seq, int slice) {
    __shared__ float q_abs[512];
    __shared__ float qpe[64];
    __shared__ float srow[1024];
    __shared__ float acc[512];
    __shared__ float red[256];

    const int h = blockIdx.x;
    const int k = blockIdx.y;
    const int p0 = k * slice;
    const int p_cnt = n_pos - p0 < slice ? n_pos - p0 : slice;   // > 0 by grid
    const float * qh = q + (int64_t) h * head_k;                 // decode: t = 0
    const float * qp = qh + nope;
    const float * crow0 = cache;                                 // n_seqs == 1

    // the absorbed q comes precomputed from attn_sk_prep_kernel (identical
    // values: that kernel runs attn_kernel's own absorption loop)
    for (int r = threadIdx.x; r < kv; r += blockDim.x) {
        q_abs[r] = q_abs_dev[(size_t) h * kv + r];
    }
    for (int i = threadIdx.x; i < rot; i += blockDim.x) {
        qpe[i] = qp[i];
    }
    for (int j = threadIdx.x; j < kv; j += blockDim.x) {
        acc[j] = 0.0f;
    }
    __syncthreads();

    // scores, 8 positions per pass, 32 lanes each: consecutive lanes read
    // consecutive cache elements (coalesced), warp-reduced per position.
    for (int pb = 0; pb < p_cnt; pb += 8) {
        const int pl = pb + ((int) threadIdx.x >> 5);
        const int lane = (int) threadIdx.x & 31;
        if (pl < p_cnt) {
            const float * cp = crow0 + (int64_t) (p0 + pl) * s_pos;
            float a = 0.0f;
            for (int i = lane; i < kv_a_dim; i += 32) {
                a += i < kv ? q_abs[i] * cp[i] : qpe[i - kv] * cp[i];
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                a += __shfl_down(a, off, 32);   // width 32: correct under wave64
            }
            if (lane == 0) {
                srow[pl] = a * scale;
            }
        }
        __syncthreads();
    }

    // slice softmax statistics
    float lmax = -INFINITY;
    for (int p = threadIdx.x; p < p_cnt; p += blockDim.x) {
        lmax = fmaxf(lmax, srow[p]);
    }
    lmax = block_reduce_max<256>(lmax, red);
    float lsum = 0.0f;
    for (int p = threadIdx.x; p < p_cnt; p += blockDim.x) {
        const float e = expf(srow[p] - lmax);
        srow[p] = e;
        lsum += e;
    }
    lsum = block_reduce_sum<256>(lsum, red);

    // weighted latent over this slice, p ascending (attn_kernel's per-j order)
    for (int p = 0; p < p_cnt; ++p) {
        const float e = srow[p];
        const float * cp = crow0 + (int64_t) (p0 + p) * s_pos;
        for (int j = threadIdx.x; j < kv; j += blockDim.x) {
            acc[j] += e * cp[j];
        }
    }
    __syncthreads();

    // publish the partial
    const size_t sk = (size_t) h * gridDim.y + k;
    if (threadIdx.x == 0) {
        part_m[sk] = lmax;
        part_l[sk] = lsum;
    }
    for (int j = threadIdx.x; j < kv; j += blockDim.x) {
        part_acc[sk * (size_t) kv + j] = acc[j];
    }
}

// ---------------------------------------------------------------------------
// LS3 PHASE5: HEAD-SHARED partial. attn_sk_partial_kernel above runs one
// workgroup per (head, slice), so every cache slice is read n_head times per
// token; at 56K that traffic is ~76 ms of a ~108 ms token. This variant runs
// one workgroup per (group of 8 heads, slice): the group's absorbed q rows
// live in LDS, the slice streams through an 8-position LDS tile once, and all
// 8 heads' scores / online-softmax states / latent partials advance together.
// Each slice is then read 4x per token (n_head / 8) instead of 32x. The
// partial layout [head][slice] and the fixed-order reduce are unchanged, so
// attn_sk_reduce_kernel combines either producer identically. No atomics.
//
// Bounds (validated host-side before launch): slice*k < n_pos by grid
// construction; cache reads are (p0+pi)*s_pos + i with p0+pi < n_pos and
// i < kv_a; partial writes are (h*gridDim.y + k) and (h*k_slices + k)*kv + j
// with h < n_head, k < k_slices <= allocated slices, j < kv.
// ---------------------------------------------------------------------------
#define LS3_SK_HG 8    // heads per workgroup
#define LS3_SK_PT 8    // cache positions per LDS tile

static __global__ __launch_bounds__(256) void attn_sk_partial_hs_kernel(
        const float * __restrict__ q,
        const float * __restrict__ cache,
        const float * __restrict__ q_abs_dev,
        float * __restrict__ part_m, float * __restrict__ part_l,
        float * __restrict__ part_acc,
        int n_head, int head_k, int nope, int rot, int kv, int kv_a_dim,
        int n_pos, float scale,
        int64_t q_tok, int64_t q_seq,
        int64_t s_pos, int64_t s_seq, int slice) {
    __shared__ float qrow[LS3_SK_HG][576 + 4];   // absorbed q (+rope tail) per head
    __shared__ float tile[LS3_SK_PT][576 + 4];
    __shared__ float s_sc[LS3_SK_HG][LS3_SK_PT];
    __shared__ float s_m[LS3_SK_HG], s_l[LS3_SK_HG];
    __shared__ float s_ao[LS3_SK_HG], s_ak[LS3_SK_HG];

    const int hg = blockIdx.x;                 // head group
    const int k = blockIdx.y;                  // slice
    const int p0 = k * slice;
    const int p_cnt = n_pos - p0 < slice ? n_pos - p0 : slice;   // > 0 by grid
    const float * crow0 = cache;               // decode: n_seqs == 1

    // Stage this group's q rows: the precomputed absorbed part, then the rope
    // tail straight from q (32 heads x 576 floats total per group).
    for (int e = (int) threadIdx.x; e < LS3_SK_HG * kv_a_dim; e += 256) {
        const int hloc = e / kv_a_dim;
        const int i = e % kv_a_dim;
        const int h = hg * LS3_SK_HG + hloc;
        float v;
        if (i < kv) {
            v = q_abs_dev[(size_t) h * kv + i];
        } else {
            v = q[(int64_t) h * head_k + nope + (i - kv)];   // decode: t = 0
        }
        qrow[hloc][i] = v;
    }
    if (threadIdx.x < LS3_SK_HG) {
        s_m[threadIdx.x] = -INFINITY;
        s_l[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // 16 latent accumulators per thread: h_local = tid/32, j = (tid%32)*16..
    const int hloc = (int) threadIdx.x >> 5;
    const int j0 = ((int) threadIdx.x & 31) * 16;
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        acc[j] = 0.0f;
    }

    for (int pt = 0; pt < p_cnt; pt += LS3_SK_PT) {
        const int npt = p_cnt - pt < LS3_SK_PT ? p_cnt - pt : LS3_SK_PT;
        // ---- stage the tile coalesced ----
        for (int e = (int) threadIdx.x; e < npt * kv_a_dim; e += 256) {
            const int pi = e / kv_a_dim;
            const int i = e % kv_a_dim;
            tile[pi][i] = crow0[(int64_t) (p0 + pt + pi) * s_pos + i];
        }
        __syncthreads();

        // ---- scores: HG x PT dots, 4 lanes each ----
        {
            const int dot = (int) threadIdx.x >> 2;         // 0..63
            const int lane = (int) threadIdx.x & 3;
            const int h2 = dot / LS3_SK_PT;
            const int p2 = dot % LS3_SK_PT;
            float a = 0.0f;
            if (h2 < LS3_SK_HG && p2 < npt) {
                for (int i = lane * 144; i < (lane + 1) * 144; ++i) {
                    a += qrow[h2][i] * tile[p2][i];
                }
#pragma unroll
                for (int off = 2; off > 0; off >>= 1) {
                    a += __shfl_down(a, off, 4);
                }
                if (lane == 0) {
                    s_sc[h2][p2] = a * scale;
                }
            }
        }
        __syncthreads();

        // ---- online softmax per head (one thread decides, all rescale) ----
        if (threadIdx.x == 0) {
            for (int h2 = 0; h2 < LS3_SK_HG; ++h2) {
                float mn = s_m[h2];
                for (int p2 = 0; p2 < npt; ++p2) {
                    mn = fmaxf(mn, s_sc[h2][p2]);
                }
                const float ao = s_m[h2] == -INFINITY ? 0.0f : expf(s_m[h2] - mn);
                float ladd = 0.0f;
                for (int p2 = 0; p2 < npt; ++p2) {
                    const float e = expf(s_sc[h2][p2] - mn);
                    s_sc[h2][p2] = e;
                    ladd += e;
                }
                s_ao[h2] = ao;
                s_m[h2] = mn;
                s_l[h2] = s_l[h2] * ao + ladd;
            }
        }
        __syncthreads();

        // ---- rescale and accumulate the weighted latent ----
        const float ao = s_ao[hloc];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc[j] *= ao;
        }
        for (int p2 = 0; p2 < LS3_SK_PT; ++p2) {
            if (p2 < npt) {
                const float e = s_sc[hloc][p2];
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    acc[j] += e * tile[p2][j0 + j];
                }
            }
        }
        __syncthreads();
    }

    // ---- publish this group's partials (same layout as the per-head kernel) ----
    const int k_slices = gridDim.y;
    for (int h2 = 0; h2 < LS3_SK_HG; ++h2) {
        const int h = hg * LS3_SK_HG + h2;
        const size_t sk = (size_t) h * k_slices + k;
        if (threadIdx.x == 0) {
            part_m[sk] = s_m[h2];
            part_l[sk] = s_l[h2];
        }
    }
    const int h = hg * LS3_SK_HG + hloc;
    const size_t sk = (size_t) h * k_slices + k;
    for (int j = 0; j < 16; ++j) {
        part_acc[sk * (size_t) kv + j0 + j] = acc[j];
    }
}

// ---------------------------------------------------------------------------
// LS3 PHASE6: REGISTER-FUSED head-shared partial. The P5 LDS-tile version
// measured 57.4 ms of an 81.8 ms 56K token (70 %) at 15.8 GB/s -- 6.5 % of
// the 242 GB/s roof, and the SAME ~15.5 GB/s at 13.8K, i.e. a fixed
// structural cost, not bandwidth: the 38 KB of LDS (q rows + 8-position
// tile) left one block per CU, the tile/barrier pattern serialized every 8
// positions, and the online softmax ran on a single thread.
//
// This variant gives the WHOLE slice to ONE workgroup with ALL 32 heads in
// registers and NO LDS:
//   thread t -> head h = t / 8, i-slice = (t % 8) * 72 .. +72  (8 x 72 = 576)
//   per position p (serial over the slice):
//     - the thread loads ITS 72 cache floats once (consecutive lanes read
//       consecutive elements -- the block reads the row exactly once);
//     - partial score via a 8-lane butterfly (__shfl_xor, no barrier);
//     - online softmax state (m, l) lives redundantly in the head's 8 lanes,
//       updated identically -> still deterministic;
//     - the SAME 72 registers immediately accumulate e_p * c into the
//       thread's 72 latent partials: score and latent FUSED, one cache read
//       per slice per layer for all heads (the traffic minimum);
//   partials out: lane j-range -> part_acc, lane 0 of each head -> m/l.
// No LDS, no __syncthreads in the position loop, no single-thread sections;
// occupancy comes from block parallelism, which hides the load latency the
// one-block-per-CU P5 could not.
// ---------------------------------------------------------------------------

static __global__ __launch_bounds__(256) void attn_sk_partial_rf_kernel(
        const float * __restrict__ q,
        const float * __restrict__ cache,
        const float * __restrict__ q_abs_dev,
        float * __restrict__ part_m, float * __restrict__ part_l,
        float * __restrict__ part_acc,
        int n_head, int head_k, int nope, int rot, int kv, int kv_a_dim,
        int n_pos, float scale,
        int64_t q_tok, int64_t q_seq,
        int64_t s_pos, int64_t s_seq, int slice) {
    // 8 lanes per head x 72 elements = 576 = kv_a; n_head * 8 <= 256.
    constexpr int LANES = 8;
    constexpr int EI = 72;
    const int k = blockIdx.x;                       // slice (1-D grid)
    const int p0 = k * slice;
    const int p_cnt = n_pos - p0 < slice ? n_pos - p0 : slice;   // > 0 by grid

    const int h = (int) threadIdx.x / LANES;        // head
    const int lane = (int) threadIdx.x % LANES;     // i-slice lane
    const int i0 = lane * EI;

    // this thread's slice of the absorbed q (with the rope tail)
    float qw[EI];
#pragma unroll
    for (int j = 0; j < EI; ++j) {
        const int i = i0 + j;
        qw[j] = i < kv ? q_abs_dev[(size_t) h * kv + i]
                       : q[(int64_t) h * head_k + nope + (i - kv)];
    }

    float m = -INFINITY;
    float l = 0.0f;
    float acc[EI];
#pragma unroll
    for (int j = 0; j < EI; ++j) {
        acc[j] = 0.0f;
    }

    const float * crow = cache;                     // decode: n_seqs == 1
    // LVL SPEC2 item 2: the score dot below is a 72-long serial fp32 chain per
    // position and the loop is runtime-bounded, so the warp stalls on every
    // position's latency. Unrolling by 2 lets two positions' chains be in
    // flight together. Nothing is reordered: each position's score reduction
    // and softmax update still happen in the same order.
#pragma unroll 2
    for (int p = 0; p < p_cnt; ++p) {
        const float * cp = crow + (int64_t) (p0 + p) * s_pos;
        // partial score over this lane's 72 elements
        float a = 0.0f;
#pragma unroll
        for (int j = 0; j < EI; ++j) {
            a += qw[j] * cp[i0 + j];
        }
        // 8-lane butterfly: every lane ends with the full dot
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            a += __shfl_xor(a, off, LANES);
        }
        const float s_sc = a * scale;
        // online softmax (identical in all 8 lanes of this head)
        const float mn = fmaxf(m, s_sc);
        const float ao = m == -INFINITY ? 0.0f : expf(m - mn);
        const float e = expf(s_sc - mn);
        m = mn;
        l = l * ao + e;
        const float esc = e;
#pragma unroll
        for (int j = 0; j < EI; ++j) {
            acc[j] = acc[j] * ao + esc * cp[i0 + j];
        }
    }

    // publish: same [head][slice] layout the reduce kernel combines
    const int k_slices = gridDim.x;
    const size_t sk = (size_t) h * k_slices + k;
    if (lane == 0) {
        part_m[sk] = m;
        part_l[sk] = l;
    }
#pragma unroll
    for (int j = 0; j < EI; ++j) {
        // only the latent range [0, kv) is a partial: lanes covering the
        // rope tail (i >= kv) must NOT publish -- the row is kv wide and
        // writing past it lands in the next head's slot (the race the first
        // run of this kernel showed as det=N and ~1e-1 errors).
        if (i0 + j < kv) {
            part_acc[sk * (size_t) kv + i0 + j] = acc[j];
        }
    }
}

// PHASE7: the m-row variant of the register-fused partial. Identical body
// to attn_sk_partial_rf_kernel with (a) row = blockIdx.y selecting the query
// row (q offset by row * q_tok; gate row likewise in the reduce), and
// (b) a per-row causal limit: row j sees global positions < n_pos - m + j + 1.
// Partials indexed [row][head][slice].
static __global__ __launch_bounds__(256) void attn_sk_partial_rf_m_kernel(
        const float * __restrict__ q,
        const float * __restrict__ cache,
        const float * __restrict__ q_abs_dev,
        float * __restrict__ part_m, float * __restrict__ part_l,
        float * __restrict__ part_acc,
        int n_head, int head_k, int nope, int rot, int kv, int kv_a_dim,
        int n_pos, int m, float scale,
        int64_t q_tok, int64_t q_seq,
        int64_t s_pos, int64_t s_seq, int slice) {
    constexpr int LANES = 8;
    constexpr int EI = 72;
    const int k = blockIdx.x;                       // slice
    const int row = blockIdx.y;                     // query row [0, m)
    const int p0 = k * slice;
    const int p_cnt = n_pos - p0 < slice ? n_pos - p0 : slice;
    const int p_limit = n_pos - m + row + 1;        // row's causal bound
    const int p_lim = p0 + p_cnt < p_limit ? p0 + p_cnt : p_limit;   // clamp to slice

    const int h = (int) threadIdx.x / LANES;
    const int lane = (int) threadIdx.x % LANES;
    const int i0 = lane * EI;

    float qw[EI];
    const float * qrow = q + (int64_t) row * q_tok;
    const float * q_abs_row = q_abs_dev + (size_t) row * n_head * kv;
#pragma unroll
    for (int j = 0; j < EI; ++j) {
        const int i = i0 + j;
        qw[j] = i < kv ? q_abs_row[(size_t) h * kv + i]
                       : qrow[(int64_t) h * head_k + nope + (i - kv)];
    }

    float mst = -INFINITY;
    float l = 0.0f;
    float acc[EI];
#pragma unroll
    for (int j = 0; j < EI; ++j) {
        acc[j] = 0.0f;
    }

    const int p_lim_local = p_lim > p0 ? p_lim - p0 : 0;   // absolute -> local
    const float * crow = cache;                     // n_seqs == 1
    for (int p = 0; p < p_lim_local; ++p) {
        const float * cp = crow + (int64_t) (p0 + p) * s_pos;
        float a = 0.0f;
#pragma unroll
        for (int j = 0; j < EI; ++j) {
            a += qw[j] * cp[i0 + j];
        }
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            a += __shfl_xor(a, off, LANES);
        }
        const float s_sc = a * scale;
        const float mn = fmaxf(mst, s_sc);
        const float ao = mst == -INFINITY ? 0.0f : expf(mst - mn);
        const float e = expf(s_sc - mn);
        mst = mn;
        l = l * ao + e;
#pragma unroll
        for (int j = 0; j < EI; ++j) {
            acc[j] = acc[j] * ao + e * cp[i0 + j];
        }
    }

    const int k_slices = gridDim.x;
    const size_t sk = ((size_t) row * n_head + h) * k_slices + k;
    if (lane == 0) {
        part_m[sk] = p_lim > 0 ? mst : -INFINITY;
        part_l[sk] = l;
    }
#pragma unroll
    for (int j = 0; j < EI; ++j) {
        if (i0 + j < kv) {
            part_acc[sk * (size_t) kv + i0 + j] = acc[j];
        }
    }
}

// PHASE7: the reduce with a row axis (grid n_head x m); combines [row][h][*]
// in fixed ascending slice order and writes out row `row`.
static __global__ __launch_bounds__(256) void attn_sk_reduce_m_kernel(
        const float * __restrict__ gate,
        const float * __restrict__ wv_b,
        const float * __restrict__ wv_bT,
        const float * __restrict__ part_m, const float * __restrict__ part_l,
        const float * __restrict__ part_acc,
        float * __restrict__ out,
        int n_head, int k_slices, int head_v, int kv, int m,
        int64_t g_tok, int64_t g_seq, int64_t o_tok, int64_t o_seq) {
    __shared__ float lat[512];
    __shared__ float s_m, s_l, s_a_old, s_a_k;
    const int h = blockIdx.x;
    const int row = blockIdx.y;

    if (threadIdx.x == 0) {
        s_m = -INFINITY;
        s_l = 0.0f;
    }
    for (int j = threadIdx.x; j < kv; j += blockDim.x) {
        lat[j] = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < k_slices; ++k) {
        const size_t sk = ((size_t) row * n_head + h) * k_slices + k;
        if (threadIdx.x == 0) {
            const float mk = part_m[sk];
            const float mn = fmaxf(s_m, mk);
            s_a_old = s_m == -INFINITY ? 0.0f : expf(s_m - mn);
            s_a_k = mk == -INFINITY ? 0.0f : expf(mk - mn);
            s_m = mn;
            s_l = s_l * s_a_old + part_l[sk] * s_a_k;
        }
        __syncthreads();
        for (int j = threadIdx.x; j < kv; j += blockDim.x) {
            lat[j] = lat[j] * s_a_old + part_acc[sk * (size_t) kv + j] * s_a_k;
        }
        __syncthreads();
    }

    const float g = sigmoid_f(gate[h + (int64_t) row * g_tok]);
    float * orow = out + (int64_t) h * head_v + (int64_t) row * o_tok;
    const float inv = 1.0f / s_l;
    for (int r = threadIdx.x; r < head_v; r += blockDim.x) {
        float a = 0.0f;
        if (wv_bT != nullptr) {
            const float * wcol = wv_bT + (int64_t) h * kv * head_v + r;
            for (int j = 0; j < kv; ++j) {
                a += wcol[j * head_v] * lat[j];
            }
        } else {
            const float * wrow = wv_b + ((int64_t) h * head_v + r) * kv;
            for (int j = 0; j < kv; ++j) {
                a += wrow[j] * lat[j];
            }
        }
        orow[r] = g * (a * inv);
    }
}

static __global__ __launch_bounds__(256) void attn_sk_reduce_kernel(
        const float * __restrict__ gate,
        const float * __restrict__ wv_b,
        const float * __restrict__ wv_bT,
        const float * __restrict__ part_m, const float * __restrict__ part_l,
        const float * __restrict__ part_acc,
        float * __restrict__ out,
        int n_head, int k_slices, int head_v, int kv,
        int64_t g_tok, int64_t g_seq, int64_t o_tok, int64_t o_seq,
        int8_t * __restrict__ wo_qs, float * __restrict__ wo_ds) {
    __shared__ float lat[512];
    __shared__ float red[256];   // (kept for symmetry; unused by the math below)
    __shared__ float s_m, s_l, s_a_old, s_a_k;
    const int h = blockIdx.x;

    if (threadIdx.x == 0) {
        s_m = -INFINITY;
        s_l = 0.0f;
    }
    for (int j = threadIdx.x; j < kv; j += blockDim.x) {
        lat[j] = 0.0f;
    }
    __syncthreads();

    // combine the slices in FIXED ascending order -- the determinism gate
    for (int k = 0; k < k_slices; ++k) {
        const size_t sk = (size_t) h * k_slices + k;
        if (threadIdx.x == 0) {
            const float mk = part_m[sk];
            const float mn = fmaxf(s_m, mk);
            s_a_old = s_m == -INFINITY ? 0.0f : expf(s_m - mn);
            s_a_k = mk == -INFINITY ? 0.0f : expf(mk - mn);
            s_m = mn;
            s_l = s_l * s_a_old + part_l[sk] * s_a_k;
        }
        __syncthreads();
        for (int j = threadIdx.x; j < kv; j += blockDim.x) {
            lat[j] = lat[j] * s_a_old + part_acc[sk * (size_t) kv + j] * s_a_k;
        }
        __syncthreads();
    }

    // de-absorb through wv_b, head gate, divide by the combined sum
    const float g = sigmoid_f(gate[h]);   // decode: t = 0
    float * orow = out + (int64_t) h * head_v;
    const float inv = 1.0f / s_l;
    for (int r = threadIdx.x; r < head_v; r += blockDim.x) {
        float a = 0.0f;
        if (wv_bT != nullptr) {
            const float * wcol = wv_bT + (int64_t) h * kv * head_v + r;
            for (int j = 0; j < kv; ++j) {
                a += wcol[j * head_v] * lat[j];
            }
        } else {
            const float * wrow = wv_b + ((int64_t) h * head_v + r) * kv;
            for (int j = 0; j < kv; ++j) {
                a += wrow[j] * lat[j];
            }
        }
        orow[r] = g * (a * inv);
    }

    // LC1 SPEC4 Q2-MLA: the wo Q8_1 quantisation as this kernel's epilogue
    // (-1 launch per MLA layer on the split-K arm). One block per head covers
    // head_v = 128 outputs = 4 whole quant groups, so every group is complete
    // inside this block. Values are read back out of `out` -- the exact floats
    // the unfused k_q8_1_quant would see.
    if (wo_qs != nullptr) {
        __syncthreads();
        mla_wo_q8_epilogue(orow, head_v, h, wo_qs, wo_ds);
    }
    (void) red; (void) g_tok; (void) g_seq; (void) o_tok; (void) o_seq;
}

void mla_attn_splitk(const MlaShape & sh, const MlaScalars & sc, const MlaWeights & w,
                     const MlaCache & cache, int n_pos, const MlaInputs & in,
                     const View & out,
                     float * q_abs, float * part_m, float * part_l, float * part_acc,
                     int slice_cap, int head_shared, hipStream_t stream,
                     int8_t * wo_qs, float * wo_ds) {
    if (in.n_tokens != 1 || in.n_seqs != 1) {
        fprintf(stderr, "ling3_mla: mla_attn_splitk is the decode path (n_tokens == 1, "
                        "n_seqs == 1), got %d/%d\n", in.n_tokens, in.n_seqs);
        abort();
    }
    const int slice = slice_cap > 0 && slice_cap <= 1024 ? slice_cap : kMlaSplitkSlice;
    const int k_slices = (n_pos + slice - 1) / slice;
    const float scale = sc.kq_scale != 0.0f ? sc.kq_scale
                                            : 1.0f / sqrtf((float) sh.head_k);
    // Q2-MLA: see mla_attn. The reduce is launched with 128 threads.
    if (wo_qs != nullptr && (sh.head_v > 128 || (sh.head_v % 32) != 0)) {
        fprintf(stderr, "ling3_mla: wo Q8_1 epilogue needs head_v <= 128 and a "
                        "multiple of 32 on the split-K reduce, got %d\n", sh.head_v);
        abort();
    }
    if (wo_qs != nullptr) {
        g_mla_wo_q8_epilogues += 1;
    }
    hipLaunchKernelGGL(attn_sk_prep_kernel,
                       dim3((unsigned) ((sh.kv + 255) / 256),
                            (unsigned) sh.n_head),
                       256, 0, stream, in.q.p, w.wk_b, w.wk_bT, q_abs,
                       sh.head_k, sh.nope, sh.kv);
    LING3_HIP_CHECK(hipGetLastError());
    if (head_shared == 2 && sh.n_head * 8 <= 256 && sh.kv_a() == 8 * 72) {
        // P6 register-fused: one workgroup per slice, all heads, no LDS
        hipLaunchKernelGGL(attn_sk_partial_rf_kernel, dim3((unsigned) k_slices),
                            256, 0, stream,
                            in.q.p, cache.kv, q_abs, part_m, part_l, part_acc,
                            sh.n_head, sh.head_k, sh.nope, sh.rot, sh.kv, sh.kv_a(),
                            n_pos, scale,
                            in.q.s_tok, in.q.s_seq,
                            cache.s_pos, cache.s_seq, slice);
    } else if (head_shared != 0 && (sh.n_head % LS3_SK_HG) == 0) {
        const dim3 gridhs((unsigned) (sh.n_head / LS3_SK_HG), (unsigned) k_slices);
        hipLaunchKernelGGL(attn_sk_partial_hs_kernel, gridhs, 256, 0, stream,
                           in.q.p, cache.kv, q_abs, part_m, part_l, part_acc,
                           sh.n_head, sh.head_k, sh.nope, sh.rot, sh.kv, sh.kv_a(),
                           n_pos, scale,
                           in.q.s_tok, in.q.s_seq,
                           cache.s_pos, cache.s_seq, slice);
    } else {
        const dim3 gridp((unsigned) sh.n_head, (unsigned) k_slices);
        hipLaunchKernelGGL(attn_sk_partial_kernel, gridp, 256, 0, stream,
                           in.q.p, cache.kv, q_abs, part_m, part_l, part_acc,
                           sh.n_head, sh.head_k, sh.nope, sh.rot, sh.kv, sh.kv_a(),
                           n_pos, scale,
                           in.q.s_tok, in.q.s_seq,
                           cache.s_pos, cache.s_seq, slice);
    }
    LING3_HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(attn_sk_reduce_kernel, dim3((unsigned) sh.n_head), 128, 0,
                       stream, in.gate.p, w.wv_b, w.wv_bT,
                       part_m, part_l, part_acc, out.p,
                       sh.n_head, k_slices, sh.head_v, sh.kv,
                       in.gate.s_tok, in.gate.s_seq, out.s_tok, out.s_seq,
                       wo_qs, wo_ds);
    LING3_HIP_CHECK(hipGetLastError());
}

void mla_attn_splitk_m(const MlaShape & sh, const MlaScalars & sc, const MlaWeights & w,
                       const MlaCache & cache, int n_pos, const MlaInputs & in,
                       const View & out, int m,
                       float * q_abs, float * part_m, float * part_l, float * part_acc,
                       int slice_cap, int head_shared, hipStream_t stream) {
    if (m < 1 || m > 4 || in.n_tokens != m || in.n_seqs != 1) {
        fprintf(stderr, "ling3_mla: mla_attn_splitk_m needs 1 <= m <= 4 rows with "
                        "n_tokens == m, n_seqs == 1, got m=%d n_tokens=%d n_seqs=%d\n",
                m, in.n_tokens, in.n_seqs);
        abort();
    }
    if (m == 1) {
        mla_attn_splitk(sh, sc, w, cache, n_pos, in, out, q_abs, part_m, part_l,
                        part_acc, slice_cap, head_shared, stream);
        return;
    }
    if (n_pos < m) {
        fprintf(stderr, "ling3_mla: mla_attn_splitk_m needs n_pos >= m (the m rows "
                        "sit at positions n_pos-m..n_pos-1), got n_pos=%d m=%d\n",
                n_pos, m);
        abort();
    }
    // The m-row partial is the register-fused variant; it needs the rf shape.
    if (sh.n_head * 8 > 256 || sh.kv_a() != 8 * 72) {
        fprintf(stderr, "ling3_mla: mla_attn_splitk_m needs the rf shape "
                        "(n_head*8 <= 256, kv_a == 576), got n_head=%d kv_a=%d\n",
                sh.n_head, sh.kv_a());
        abort();
    }
    // Host-side audit before any launch (the 2026-10-03T04:43Z GCVM permission
    // fault class: an out-of-range partial access). The partials are indexed
    // [row][head][slice], so this call touches m * n_head * k_slices slots and
    // m * n_head * k_slices * kv accumulator floats: the caller must size the
    // buffers for m rows per the hpp contract. The launch is clamped to
    // kMlaSplitkMaxSlices slices so a legal-sized buffer is never overrun.
    int slice = slice_cap > 0 && slice_cap <= kMlaSplitkSlice ? slice_cap : kMlaSplitkSlice;
    int k_slices = (n_pos + slice - 1) / slice;
    if (k_slices > kMlaSplitkMaxSlices) {
        slice = mla_splitk_slice_auto(n_pos, kMlaSplitkMaxSlices);
        k_slices = (n_pos + slice - 1) / slice;
    }
    const float scale = sc.kq_scale != 0.0f ? sc.kq_scale
                                            : 1.0f / sqrtf((float) sh.head_k);
    // q_abs for every row: the prep kernel handles one row today, so run it
    // per row with the q view offset (same absorption loop, identical values).
    for (int r = 0; r < m; ++r) {
        MlaInputs rin = in;
        rin.q = ViewC{in.q.p + (int64_t) r * in.q.s_tok, in.q.s_tok, in.q.s_seq};
        rin.n_tokens = 1;
        hipLaunchKernelGGL(attn_sk_prep_kernel,
                           dim3((unsigned) ((sh.kv + 255) / 256),
                                (unsigned) sh.n_head),
                           256, 0, stream, rin.q.p, w.wk_b, w.wk_bT,
                           q_abs + (size_t) r * sh.n_head * sh.kv,
                           sh.head_k, sh.nope, sh.kv);
        LING3_HIP_CHECK(hipGetLastError());
    }
    const dim3 gridp((unsigned) k_slices, (unsigned) m);
    hipLaunchKernelGGL(attn_sk_partial_rf_m_kernel, gridp, 256, 0, stream,
                       in.q.p, cache.kv, q_abs, part_m, part_l, part_acc,
                       sh.n_head, sh.head_k, sh.nope, sh.rot, sh.kv, sh.kv_a(),
                       n_pos, m, scale,
                       in.q.s_tok, in.q.s_seq,
                       cache.s_pos, cache.s_seq, slice);
    LING3_HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(attn_sk_reduce_m_kernel,
                       dim3((unsigned) sh.n_head, (unsigned) m), 128, 0, stream,
                       in.gate.p, w.wv_b, w.wv_bT,
                       part_m, part_l, part_acc, out.p,
                       sh.n_head, k_slices, sh.head_v, sh.kv, m,
                       in.gate.s_tok, in.gate.s_seq, out.s_tok, out.s_seq);
    LING3_HIP_CHECK(hipGetLastError());
}

void mla_block_splitk(const MlaShape & sh, const MlaScalars & sc, const MlaWeights & w,
                      const MlaCache & cache, const MlaInputs & in,
                      const int * pos, int n_pos, const View & out,
                      float * q_abs, float * part_m, float * part_l, float * part_acc,
                      int slice_cap, int head_shared, hipStream_t stream) {
    if (pos == nullptr) {
        fprintf(stderr, "ling3_mla: mla_block_splitk needs a DEVICE pointer to the position\n");
        abort();
    }
    const int T = in.n_tokens;
    // LC1 SPEC4: M1 fuses the four stage launches (-3), Q2-MLA adds the wo Q8_1
    // epilogue (-1). This entry's cache_append used NO pos_d (npos = n_pos), so
    // the fused kernel is handed the same nullptr and computes the same npos.
    int8_t * wo_qs = nullptr;
    float *  wo_ds = nullptr;
    if (mla_decode_fused()) {
        mla_pre_fused(sh, sc, w, cache, in, pos, n_pos, /*pos_d=*/nullptr, stream);
        if (in.n_tokens == 1 && in.n_seqs == 1 && mla_wo_q8_prepare(sh.o_dim())) {
            wo_qs = g_wo_qs;
            wo_ds = g_wo_ds;
        }
    } else {
        mla_rope(sh, const_cast<float *>(in.q.p) + sh.nope,
                 /*s_head=*/sh.head_k, /*n_head=*/sh.n_head,
                 T, in.q.s_tok, in.q.s_seq, in.n_seqs, pos, stream);
        mla_rope(sh, const_cast<float *>(in.kv_a.p) + sh.kv,
                 /*s_head=*/0, /*n_head=*/1,
                 T, in.kv_a.s_tok, in.kv_a.s_seq, in.n_seqs, pos, stream);
        mla_latent_norm(sh, sc, w, const_cast<float *>(in.kv_a.p), T, in.n_seqs,
                        in.kv_a.s_tok, in.kv_a.s_seq, stream);
        mla_cache_append(sh, cache, in.kv_a.p, T, in.kv_a.s_tok, in.n_seqs, n_pos, stream);
    }
    mla_attn_splitk(sh, sc, w, cache, n_pos, in, out, q_abs, part_m, part_l,
                    part_acc, slice_cap, head_shared, stream, wo_qs, wo_ds);
}

void mla_block(const MlaShape & sh, const MlaScalars & sc, const MlaWeights & w,
               const MlaCache & cache, const MlaInputs & in,
               const int * pos, int n_pos, const View & out, hipStream_t stream,
               const int * pos_d) {
    const size_t need = mla_block_scratch_bytes(sh, in.n_seqs, n_pos);
    void * scratch = nullptr;
    LING3_HIP_CHECK(hipMalloc(&scratch, need));
    mla_block(sh, sc, w, cache, in, pos, n_pos, out, (float *) scratch, stream, pos_d);
    // The launches above are async. Freeing `scratch` while attn_kernel is still reading
    // its score rows is a use-after-free, which the GPU reports as a page fault
    // (GCVM_L2_PROTECTION_FAULT) rather than a clean error. Sync first.
    LING3_HIP_CHECK(hipStreamSynchronize(stream));
    LING3_HIP_CHECK(hipFree(scratch));
}

// the scratch-taking core
void mla_block(const MlaShape & sh, const MlaScalars & sc, const MlaWeights & w,
               const MlaCache & cache, const MlaInputs & in,
               const int * pos, int n_pos, const View & out,
               float * scores, hipStream_t stream, const int * pos_d) {
    const int T = in.n_tokens;
    if (pos == nullptr) {
        fprintf(stderr, "ling3_mla: mla_block needs a DEVICE pointer to the positions\n");
        abort();
    }

    // LC1 SPEC4: M1 + Q2-MLA, exactly as mla_block_splitk. This entry's
    // cache_append DID take pos_d, so the fused kernel gets it and computes
    // npos = pos_d[0] + 1 the same way.
    int8_t * wo_qs = nullptr;
    float *  wo_ds = nullptr;
    if (mla_decode_fused()) {
        mla_pre_fused(sh, sc, w, cache, in, pos, n_pos, pos_d, stream);
        if (in.n_tokens == 1 && in.n_seqs == 1 && mla_wo_q8_prepare(sh.o_dim())) {
            wo_qs = g_wo_qs;
            wo_ds = g_wo_ds;
        }
    } else {
        // 1. RoPE: q's rope section (one entry per head) and kv_a's key part (shared, n_head 1)
        mla_rope(sh, const_cast<float *>(in.q.p) + sh.nope,
                 /*s_head=*/sh.head_k, /*n_head=*/sh.n_head,
                 T, in.q.s_tok, in.q.s_seq, in.n_seqs, pos, stream);
        mla_rope(sh, const_cast<float *>(in.kv_a.p) + sh.kv,
                 /*s_head=*/0, /*n_head=*/1,
                 T, in.kv_a.s_tok, in.kv_a.s_seq, in.n_seqs, pos, stream);

        // 2. RMSNorm the latent
        mla_latent_norm(sh, sc, w, const_cast<float *>(in.kv_a.p), T, in.n_seqs,
                        in.kv_a.s_tok, in.kv_a.s_seq, stream);

        // 3. append to the cache
        mla_cache_append(sh, cache, in.kv_a.p, T, in.kv_a.s_tok, in.n_seqs, n_pos, stream, pos_d);
    }

    // 4. attention + absorption + de-absorption + gate
    mla_attn(sh, sc, w, cache, n_pos, in, out, scores, stream, pos_d, wo_qs, wo_ds);
}

void mla_block_decode(const MlaShape & sh, const MlaScalars & sc, const MlaWeights & w,
                      const MlaCache & cache, const MlaInputs & in,
                      const int * pos, int n_pos, const View & out, hipStream_t stream) {
    if (in.n_tokens != 1) {
        fprintf(stderr, "ling3_mla: mla_block_decode expects n_tokens == 1, got %d\n", in.n_tokens);
        abort();
    }
    if (pos == nullptr) {
        fprintf(stderr, "ling3_mla: mla_block_decode needs a DEVICE pointer to the position\n");
        abort();
    }
    mla_block(sh, sc, w, cache, in, pos, n_pos, out, stream);
}

const char * mla_version() { return "ling3-mla/1.0"; }

} // namespace ling3
