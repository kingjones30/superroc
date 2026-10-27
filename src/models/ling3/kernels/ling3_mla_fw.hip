// ling3_mla_fw.hip — LX1 (wave 3): fused WMMA flash-attention for the
// absorbed-MLA prefill. See ling3_mla_fw.hpp for the design and contracts.
//
// Layout of the path (per layer per 512-token chunk):
//   1. mla_w2_prep (ling3_mla_w2.hip, reused unchanged) stages the f32 cache
//      to f16 k16 [cap][576] (zero-padded to round64(n_pos)) and the
//      transposed latent vt [512][cap].
//   2. fw_qtrans_kernel stages q_nope into the [head][token][dim] layout the
//      pack GEMM's B fragments need, and writes q16's scaled rope tail.
//   3. fw_pack_kernel absorbs (RV3 C2): q16[r][0:512] =
//      f16(scale * wk_b[h] . q_nope) as ONE f16 WMMA GEMM per head with f32
//      accumulation. w2's serial-dot packq is latency-bound and costs
//      MEASURED 77 ms at the whole-chunk row count (16,384 rows); this is
//      ~0.3 ms.
//   4. fw_attn_kernel: the fused flash kernel — S = Q.K^T and O = P.V on the
//      matrix cores, online softmax in f32, nothing materialized in global
//      memory, ONE launch for the whole chunk. MEASURED 16.1 ms at the 8K
//      capture (the 20 ms target) in the 172-VGPR, zero-spill shape.
//   5. fw_deabsorb_kernel: out = sigmoid(gate) * wv_b[h] . (O/rowsum), also
//      one f16 WMMA GEMM per head (the serial-dot epilogue costs MEASURED
//      84 ms at this row count; this is ~0.3 ms).
//
// Index map of fw_attn_kernel (every access, enumerated — the host audit
// checks the extents these derive from before launch):
//   grid   = tok_count blocks; block b = local token lt = b
//   rows   = rb + gr, rb = lt * n_head (32 rows per block), gr in [0, 32)
//   q16    [(rb + gr) * kva + kd],  kd in [0, 576)          uint4 at kd%8==0
//   k16    [(c0 + p) * kva + kc + d], p in [0, 64),          uint4, d%8==0
//          c0 + 64 <= round64(n_pos) <= cap (w2 prep pads [n_pos, pad) with
//          zeros, so the tail chunk's loads read zeros, never poison)
//   Sf     [gr][c0-relative col] in LDS — 32 x 64, written once per chunk
//   Ps     [gr][col] in LDS — 32 x 64 f16, written once per chunk
//   vt     [(dim) * cap + c0 + ks*16 + q], dim in [0, 512), q in [0, 16)
//          uint4 at (c0 + ks*16) even-16 alignment; c0+64 <= cap as above
//   o      [(rb + gr) * kv + gd], gd in [0, 512)             scalar
//   rowsum [rb + gr]                                          scalar
//   pl     = (n_pos - chunk_tokens) + (tok_base + lt) + 1 in [1, n_pos]:
//          the chunk loop bounds c0 < pl, and every score beyond pl is
//          masked (-inf in the max, 0 in P), so real cache rows in
//          [pl, n_pos) can never reach the output.
//
// Index maps of the three helper kernels:
//   fw_qtrans:  one thread per row r = lt*n_head + h in [0, tok_count*32);
//              reads q[(tok_base+lt)*q_tok + h*head_k + i], i in [0, 192);
//              writes qT[(h*tok_count + lt)*nope + i] and
//              q16[r*(kv+rot) + kv + i], i in [0, 64).
//   fw_pack:    grid n_head x ceil(T/32); block = (h, t0). Reads
//              wk_b[(h*kv + j)*nope + i] and qT[(h*T + t)*nope + i]
//              (i < 128, staged 16 at a time); writes
//              q16[(t*n_head + h)*(kv+rot) + j], j in [0, 512), only for
//              t in [t0, t0 + live).
//   fw_deabsorb: grid n_head x ceil(T/64); block = (h, t0). Reads
//              wv_b[(h*head_v + ro)*kv + j], o[((t*n_head + h)*kv) + j]
//              and rowsum[t*n_head + h]; writes
//              out[(tok_base+t)*o_tok + h*head_v + ro] and gate at
//              (tok_base+t)*g_tok + h, only for live t.
//
// WMMA fragment maps: the wave32 f16 builtin with the maps measured by LW1
// (from superroc's qwen attention_wmma.hip) and LF1's impulse probe:
//   A frag: lane L is matrix row L%16, 16 contiguous K values.
//   B frag: lane L is matrix column L%16, 16 contiguous K values from an
//           [N][K]-major tile.
//   C frag: element i of lane L is row (2i + L/16), column L%16.

#include "ling3_mla_fw.hpp"

#include <cmath>
#include <cstdio>
#include <cstdint>

#define LING3_FW_HIP_CHECK(expr)                                                    \
    do {                                                                            \
        hipError_t _err = (expr);                                                   \
        if (_err != hipSuccess) {                                                   \
            fprintf(stderr, "ling3_mla_fw: %s failed at %s:%d: %s\n", #expr,        \
                    __FILE__, __LINE__, hipGetErrorString(_err));                    \
            abort();                                                                \
        }                                                                           \
    } while (0)

namespace ling3 {

namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f  = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// A/B fragments are 16 contiguous halves = 32 bytes; two 16-byte loads and a
// bitcast (w2's LoadFrag, verbatim — same proven fragment path).
__device__ __forceinline__ v16h LoadFrag(const __half * p) {
    union {
        v16h f;
        uint4 u[2];
    } cvt;
    cvt.u[0] = *reinterpret_cast<const uint4 *>(p);
    cvt.u[1] = *reinterpret_cast<const uint4 *>(p + 8);
    return cvt.f;
}

// ---------------------------------------------------------------------------
// Shape constants (audited host-side; the kernels bake them in).
// ---------------------------------------------------------------------------
constexpr int FW_THREADS  = 256;          // 8 wave32 warps
constexpr int FW_NHEAD    = 32;           // query rows per block (one token)
constexpr int FW_CHUNK   = 64;            // positions per iteration
constexpr int FW_KV      = 512;
constexpr int FW_KVA     = 576;
constexpr int FW_KSLICE  = 64;            // K dims staged per LDS slice
constexpr int FW_DPERW   = FW_KV / 8;      // 64 dims per warp in GEMM2

// LDS row strides: width + 8 halves (w2's padding rule) keeps the 16-lane
// fragment loads and the 8-thread softmax groups in the 2-way bank-conflict
// class w2 measured and accepted. Every stride is a multiple of 8 halves =
// 16 B, so all fragment loads stay 16-byte aligned.
constexpr int FW_QSTRIDE = FW_KVA + 8;    // 584 halves
constexpr int FW_KSTRIDE = FW_KSLICE + 8; // 72 halves
constexpr int FW_PSTRIDE = FW_CHUNK + 8;  // 72 halves
constexpr int FW_SSTRIDE = FW_CHUNK + 4;  // 68 words (16B-aligned rows, and
                                          // rows 0..3 of a 32-lane access hit
                                          // distinct bank quads)
constexpr int FW_QPIECES = FW_KVA / 8;    // 72 x 16B pieces per staged Q row
constexpr int FW_KPIECES = FW_KSLICE / 8; //  8 x 16B pieces per staged K row

struct FwSmem {
    __half Qs[FW_NHEAD][FW_QSTRIDE];      // 37,376 B
    __half Ks[FW_CHUNK][FW_KSTRIDE];      //  9,216 B
    float  Sf[FW_NHEAD][FW_SSTRIDE];      //  8,704 B
    __half Ps[FW_NHEAD][FW_PSTRIDE];      //  4,608 B
    float  m_run[FW_NHEAD];               //     128 B
    float  l_run[FW_NHEAD];               //     128 B
    float  mo[FW_NHEAD];                  //     128 B
    float  mn[FW_NHEAD];                  //     128 B
};
static_assert(sizeof(FwSmem) <= 61440, "LDS budget: keep the workgroup under "
              "60 KB of the 64 KB limit (1 block/CU, the fork's occupancy)");

// ---------------------------------------------------------------------------
// The fused flash kernel. Grid: tok_count; one block per token (its 32 heads).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(FW_THREADS) void fw_attn_kernel(
        const __half * __restrict__ q16,
        const __half * __restrict__ k16,
        const __half * __restrict__ vt,
        float * __restrict__ o,
        float * __restrict__ rowsum,
        int n_head, int kva, int kv, int cap,
        int base, int tok_base) {
    __shared__ __align__(16) FwSmem sm;

    const int tid  = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int sub  = lane & 15;
    const int hid  = lane >> 4;

    const int lt = (int) blockIdx.x;            // local token of the chunk
    const int pl = base + tok_base + lt + 1;    // uniform causal limit, [1, n_pos]
    const int rb = lt * n_head;                 // this block's first row

    // softmax state + Q staging (cooperative, 16B pieces)
    if (tid < FW_NHEAD) {
        sm.m_run[tid] = -1.0e30f;
        sm.l_run[tid] = 0.0f;
    }
    for (int e = tid; e < FW_NHEAD * FW_QPIECES; e += FW_THREADS) {
        const int row = e / FW_QPIECES;
        const int c8 = (e % FW_QPIECES) * 8;
        *(uint4 *) &sm.Qs[row][c8] =
            *(const uint4 *) &q16[(int64_t) (rb + row) * kva + c8];
    }
    __syncthreads();

    const int rt = warp & 1;   // GEMM1 row tile (0/1)
    const int pt = warp >> 1;  // GEMM1 position tile (0..3)
    const int d_base = warp * FW_DPERW;   // GEMM2 dim range [d_base, +64)

    // GEMM2 accumulator: warp w owns dims [64*w, 64*w + 64) x all 32 rows =
    // 2 row tiles x 4 dim tiles x 8 elements = 64 f32 per lane (8 chains —
    // LF1's measured WMMA sweet spot, far from the 256-VGPR spill regime).
    v8f acc[2][4];
    for (int r2 = 0; r2 < 2; ++r2) {
        for (int dt = 0; dt < 4; ++dt) acc[r2][dt] = {};
    }

    for (int c0 = 0; c0 < pl; c0 += FW_CHUNK) {
        // ---- GEMM1: S tile for (rt, pt) over the full 576 dims, K staged
        // through LDS in 64-dim slices (each K element is loaded from global
        // exactly once per block per chunk).
        v8f cs = {};
        for (int kc = 0; kc < FW_KVA; kc += FW_KSLICE) {
            for (int e = tid; e < FW_CHUNK * FW_KPIECES; e += FW_THREADS) {
                const int p  = e / FW_KPIECES;
                const int d8 = (e % FW_KPIECES) * 8;
                *(uint4 *) &sm.Ks[p][d8] =
                    *(const uint4 *) &k16[(int64_t) (c0 + p) * kva + kc + d8];
            }
            __syncthreads();
#pragma unroll
            for (int ks = 0; ks < FW_KSLICE / 16; ++ks) {
                cs = Wmma(LoadFrag(&sm.Qs[rt * 16 + sub][kc + ks * 16]),
                          LoadFrag(&sm.Ks[pt * 16 + sub][ks * 16]), cs);
            }
            __syncthreads();
        }

        // ---- S tile -> LDS (each element written by exactly one lane)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            sm.Sf[rt * 16 + 2 * i + hid][pt * 16 + sub] = cs[i];
        }
        __syncthreads();

        // ---- online softmax: one 8-lane group per row, f32, fixed order.
        // Group g = lanes [8g, 8g+8) of the block (rows never straddle a
        // warp); thread j of row r owns columns [8j, 8j+8) of the chunk.
        {
            const int row = tid >> 3;
            const int j   = tid & 7;
            const int cb  = j * 8;
            const float4 sa = *(const float4 *) &sm.Sf[row][cb];
            const float4 sb = *(const float4 *) &sm.Sf[row][cb + 4];
            const float vv[8] = {sa.x, sa.y, sa.z, sa.w,
                                 sb.x, sb.y, sb.z, sb.w};

            float mx = -1.0e30f;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                if (c0 + cb + q < pl) mx = fmaxf(mx, vv[q]);
            }
#pragma unroll
            for (int off = 1; off < 8; off <<= 1) {
                mx = fmaxf(mx, __shfl_xor(mx, off));
            }

            // H1 (RV17): the STANDARD online-softmax form. m_new =
            // fmaxf(m_old, mx) keeps every rescale factor <= 1, so a later
            // chunk whose max drops far below the running max (a > ~88 gap
            // overflows expf) can no longer send l_run and the O
            // accumulator to inf -> NaN in the de-absorb division. All 8
            // lanes read m_run[row] here, before the j == 0 update below;
            // the S-tile barrier above and the previous chunk's staging
            // barriers fence that read against the earlier update.
            const float m_old = sm.m_run[row];
            const float m_new = fmaxf(m_old, mx);

            float p8[8];
            float ssum = 0.0f;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const float s = (c0 + cb + q < pl) ? expf(vv[q] - m_new) : 0.0f;
                p8[q] = s;
                ssum += s;
            }
#pragma unroll
            for (int off = 1; off < 8; off <<= 1) {
                ssum += __shfl_xor(ssum, off);
            }

            // P -> f16 LDS tile (one 16B store per thread; the columns this
            // row owns beyond pl are exact zeros, so GEMM2 can round the
            // position loop up to the 64 boundary for free).
            union {
                __half h[8];
                uint4 u;
            } pc;
#pragma unroll
            for (int q = 0; q < 8; ++q) pc.h[q] = __float2half(p8[q]);
            *(uint4 *) &sm.Ps[row][cb] = pc.u;

            if (j == 0) {
                const float scale_old = expf(m_old - m_new);   // <= 1 always
                sm.l_run[row] = sm.l_run[row] * scale_old + ssum;
                sm.m_run[row] = m_new;
                sm.mo[row] = m_old;
                sm.mn[row] = m_new;
            }
        }
        __syncthreads();

        // ---- rescale the O accumulator by exp(mo - mn) per row. With the
        // standard running max (H1) mn >= mo, so the factor is always <= 1;
        // rows whose max did not move this chunk rescale by exactly 1.
#pragma unroll
        for (int r2 = 0; r2 < 2; ++r2) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int gr = r2 * 16 + 2 * i + hid;
                const float f = sm.mo[gr] == sm.mn[gr]
                                    ? 1.0f : expf(sm.mo[gr] - sm.mn[gr]);
#pragma unroll
                for (int dt = 0; dt < 4; ++dt) acc[r2][dt][i] *= f;
            }
        }

        // ---- GEMM2: O += P . V. A fragments from the LDS P tile, B
        // fragments straight from the staged vt (each warp's 64-dim range is
        // exclusively its own, so V is read once per block per chunk up to
        // the fragment's inherent 2-lane duplication).
#pragma unroll
        for (int ks = 0; ks < FW_CHUNK / 16; ++ks) {
            const v16h a0 = LoadFrag(&sm.Ps[sub][ks * 16]);
            const v16h a1 = LoadFrag(&sm.Ps[16 + sub][ks * 16]);
#pragma unroll
            for (int dt = 0; dt < 4; ++dt) {
                const v16h b = LoadFrag(
                    &vt[((int64_t) (d_base + dt * 16 + sub)) * cap +
                         c0 + ks * 16]);
                acc[0][dt] = Wmma(a0, b, acc[0][dt]);
                acc[1][dt] = Wmma(a1, b, acc[1][dt]);
            }
        }
        // The next chunk's K staging + its two barriers fence this chunk's P
        // tile reads before the next softmax overwrites Ps.
    }

    // ---- epilogue: unnormalised latent + rowsum. Every (row, dim) of the
    // block is written by exactly one lane.
#pragma unroll
    for (int r2 = 0; r2 < 2; ++r2) {
#pragma unroll
        for (int dt = 0; dt < 4; ++dt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int gr = r2 * 16 + 2 * i + hid;
                o[(int64_t) (rb + gr) * kv + d_base + dt * 16 + sub] =
                    acc[r2][dt][i];
            }
        }
    }
    if (warp == 0 && lane < FW_NHEAD) {
        rowsum[rb + lane] = sm.l_run[lane];
    }
}

// ---------------------------------------------------------------------------
// Absorb / de-absorb as WMMA GEMMs (RV3 C2). w2's serial-dot packq/out are
// latency-bound and cost MEASURED 77 + 84 ms at the whole-chunk row count
// (16,384 rows); the same arithmetic as f16 WMMA GEMMs with f32 accumulation
// is ~0.3 ms each.
// ---------------------------------------------------------------------------

// 1. q transpose + rope tail. One thread per (token, head) row.
__global__ __launch_bounds__(256) void fw_qtrans_kernel(
        const float * __restrict__ q, float * __restrict__ qT,
        __half * __restrict__ q16,
        int n_head, int head_k, int nope, int kv, int rot,
        int64_t q_tok, int tok_base, int tok_count, float scale) {
    const int r = blockIdx.x * 256 + (int) threadIdx.x;   // lt * n_head + h
    if (r >= tok_count * n_head) return;
    const int lt = r / n_head;
    const int h  = r % n_head;
    const int t  = tok_base + lt;
    const float * qh = q + (int64_t) t * q_tok + (int64_t) h * head_k;
    float * qt = qT + ((int64_t) h * tok_count + lt) * nope;
    for (int i = 0; i < nope; ++i) {   // one thread per row: the WHOLE row
        qt[i] = qh[i];
    }
    __half * tail = q16 + (int64_t) r * (kv + rot) + kv;
    for (int i = 0; i < rot; ++i) {
        tail[i] = __float2half(qh[nope + i] * scale);
    }
}

// 2. Pack: q16[r][j] = f16(scale * wk_b[h] . qT[h][t]).
// Grid: n_head x ceil(T / 32); block = (h, t0). Warp w owns the four j-tiles
// of [64w, 64w+64) x the two t-tiles of the block's 32 tokens = 4 x 2 C tiles
// (64 f32 acc per lane). K = nope (128) in 16-wide steps; the wk_b[h] slice
// and the qT slice stage through LDS as f16.
constexpr int FW_PK_T    = 32;   // tokens per block
constexpr int FW_PK_NT   = 2;    // t-tiles per warp (32 / 16)
constexpr int FW_PK_NJT  = 4;    // j-tiles per warp (64 / 16)
constexpr int FW_PK_STR  = 24;   // LDS row stride, halves (16 + 8): 48 B,
                                 // 16B-aligned rows, 2-way fragment conflicts

__global__ __launch_bounds__(256) void fw_pack_kernel(
        const float * __restrict__ wk_b, const float * __restrict__ qT,
        __half * __restrict__ q16,
        int n_head, int nope, int kv, int rot,
        int tok_base, int tok_count, float scale) {
    __shared__ __align__(16) __half As[FW_KV][FW_PK_STR];    // [j][i-slice]
    __shared__ __align__(16) __half Bs[FW_PK_T][FW_PK_STR];  // [t][i-slice]

    const int h  = (int) (blockIdx.x % (unsigned) n_head);
    const int t0 = (int) (blockIdx.x / (unsigned) n_head) * FW_PK_T;
    const int tid  = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int sub  = lane & 15;
    const int hid  = lane >> 4;

    const int nt = t0 + FW_PK_T <= tok_count ? FW_PK_T : tok_count - t0;
    const float * wh  = wk_b + (int64_t) h * kv * nope;              // [512][128]
    const float * qth = qT + ((int64_t) h * tok_count + t0) * nope;  // [T][128]

    v8f acc[FW_PK_NJT][FW_PK_NT];
    for (int a = 0; a < FW_PK_NJT; ++a) {
        for (int b = 0; b < FW_PK_NT; ++b) acc[a][b] = {};
    }

    for (int kc = 0; kc < nope; kc += 16) {
        for (int e = tid; e < FW_KV * 16; e += FW_THREADS) {
            const int j = e >> 4;
            const int i = e & 15;
            As[j][i] = __float2half(wh[(int64_t) j * nope + kc + i]);
        }
        for (int e = tid; e < FW_PK_T * 16; e += FW_THREADS) {
            const int t = e >> 4;
            const int i = e & 15;
            Bs[t][i] = t < nt ? __float2half(qth[(int64_t) t * nope + kc + i])
                              : __half(0.0f);
        }
        __syncthreads();
#pragma unroll
        for (int jt = 0; jt < FW_PK_NJT; ++jt) {
            const v16h a = LoadFrag(&As[(warp * 64 + jt * 16) + sub][0]);
#pragma unroll
            for (int tt = 0; tt < FW_PK_NT; ++tt) {
                const v16h b = LoadFrag(&Bs[tt * 16 + sub][0]);
                acc[jt][tt] = Wmma(a, b, acc[jt][tt]);
            }
        }
        __syncthreads();
    }

    // epilogue: C element i of lane L is j = 64w + jt*16 + 2i + hid at
    // token t = t0 + tt*16 + sub; q16 row r = t * n_head + h (tok_base's
    // offset is already inside qT's t index). nt is the block's LIVE token
    // count (from t0), so the guard is against t0 + nt, not nt.
#pragma unroll
    for (int jt = 0; jt < FW_PK_NJT; ++jt) {
#pragma unroll
        for (int tt = 0; tt < FW_PK_NT; ++tt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int t = t0 + tt * 16 + sub;
                if (t >= t0 + nt) continue;
                const int j = warp * 64 + jt * 16 + 2 * i + hid;
                q16[(int64_t) (t * n_head + h) * (kv + rot) + j] =
                    __float2half(acc[jt][tt][i] * scale);
            }
        }
    }
}

// 3. De-absorb: out[t][h][ro] = sigmoid(gate[t][h]) * wv_b[h] . (o[r]/rowsum).
// Grid: n_head x ceil(T / 64); block = (h, t0). Warp w is ONE r_out tile
// (16 rows) x the four t-tiles of the block's 64 tokens = 4 C tiles
// (32 f32 acc per lane). K = kv (512) in 16-wide steps.
constexpr int FW_DA_T   = 64;   // tokens per block
constexpr int FW_DA_NT  = 4;    // t-tiles per warp (64 / 16)
constexpr int FW_DA_STR = 24;   // LDS row stride, halves

__global__ __launch_bounds__(256) void fw_deabsorb_kernel(
        const float * __restrict__ wv_b, const float * __restrict__ o,
        const float * __restrict__ rowsum, const float * __restrict__ gate,
        float * __restrict__ out,
        int n_head, int head_v, int kv,
        int tok_base, int tok_count, int64_t g_tok, int64_t o_tok) {
    __shared__ __align__(16) __half As[/*head_v=*/128][FW_DA_STR];  // [ro][j]
    __shared__ __align__(16) __half Bs[FW_DA_T][FW_DA_STR];         // [t][j]
    __shared__ float inv[FW_DA_T];

    const int h  = (int) (blockIdx.x % (unsigned) n_head);
    const int t0 = (int) (blockIdx.x / (unsigned) n_head) * FW_DA_T;
    const int tid  = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int sub  = lane & 15;
    const int hid  = lane >> 4;

    const int nt = t0 + FW_DA_T <= tok_count ? FW_DA_T : tok_count - t0;
    const float * wh = wv_b + (int64_t) h * head_v * kv;    // [128][512]

    if (tid < FW_DA_T) {
        inv[tid] = tid < nt ? 1.0f / rowsum[(int64_t) (t0 + tid) * n_head + h]
                            : 0.0f;
    }
    __syncthreads();

    v8f acc[FW_DA_NT];
    for (int b = 0; b < FW_DA_NT; ++b) acc[b] = {};

    for (int kc = 0; kc < kv; kc += 16) {
        for (int e = tid; e < 128 * 16; e += FW_THREADS) {
            const int ro = e >> 4;
            const int j = e & 15;
            As[ro][j] = __float2half(wh[(int64_t) ro * kv + kc + j]);
        }
        for (int e = tid; e < FW_DA_T * 16; e += FW_THREADS) {
            const int t = e >> 4;
            const int j = e & 15;
            const int64_t r = (int64_t) (t0 + t) * n_head + h;
            Bs[t][j] = t < nt ? __float2half(o[r * kv + kc + j] * inv[t])
                              : __half(0.0f);
        }
        __syncthreads();
        const v16h a = LoadFrag(&As[(warp * 16) + sub][0]);
#pragma unroll
        for (int tt = 0; tt < FW_DA_NT; ++tt) {
            const v16h b = LoadFrag(&Bs[tt * 16 + sub][0]);
            acc[tt] = Wmma(a, b, acc[tt]);
        }
        __syncthreads();
    }

    // epilogue: C element i of lane L is ro = 16w + 2i + hid at token
    // t = t0 + tt*16 + sub (a LOCAL token; out/gate take the global one).
    // nt is the block's LIVE token count (from t0): guard against t0 + nt.
#pragma unroll
    for (int tt = 0; tt < FW_DA_NT; ++tt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int t = t0 + tt * 16 + sub;
            if (t >= t0 + nt) continue;
            const int ro = warp * 16 + 2 * i + hid;
            const int g = tok_base + t;
            const float sg = 1.0f / (1.0f + expf(-gate[(int64_t) g * g_tok + h]));
            out[(int64_t) g * o_tok + (int64_t) h * head_v + ro] =
                sg * acc[tt][i];
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// host side: audit + launch
// ---------------------------------------------------------------------------

// LF1's lf_audit model: enumerate every invariant the kernels rely on and
// refuse the launch (caller falls back) if any fails. No GPU fault may come
// from a bad index.
static bool fw_audit(const MlaShape & sh, const MlaWeights & w, int n_pos,
                     const MlaInputs & in, const View & out,
                     const MlaW2Bufs & w2b, const MlaFwBufs & bufs,
                     int tok_base, int tok_count, int chunk_tokens) {
    const char * bad = nullptr;
    if (sh.n_head != FW_NHEAD || sh.head_k != 192 || sh.nope != 128 ||
        sh.rot != 64 || sh.kv != FW_KV || sh.kv_a() != FW_KVA ||
        sh.head_v != 128)   // H2 (RV17): deabsorb bakes head_v = 128
        bad = "unsupported MlaShape (needs 32/192/128/64/512/576/head_v 128)";
    else if (in.n_seqs != 1) bad = "n_seqs != 1";
    else if (tok_base < 0 || tok_count <= 0) bad = "tok range";
    else if (tok_base + tok_count > chunk_tokens) bad = "tok range vs chunk";
    else if (chunk_tokens <= 0 || chunk_tokens > kMlaFwChunkTokens)
        bad = "chunk_tokens out of (0, 512]";
    else if (chunk_tokens > in.n_tokens)
        bad = "chunk_tokens > in.n_tokens";   // H3 (RV17), as w2 refuses
    else if (tok_count > kMlaFwChunkTokens) bad = "tok_count > 512";
    else if (n_pos <= 0 || n_pos > w2b.cap || (w2b.cap & 63) != 0)
        bad = "n_pos out of (0, cap] or cap not a multiple of 64";
    else if (n_pos < chunk_tokens) bad = "n_pos < chunk_tokens";
    else if (bufs.rows_cap < tok_count * sh.n_head) bad = "rows_cap too small";
    else if (bufs.q16 == nullptr || bufs.o == nullptr ||
             bufs.rowsum == nullptr || bufs.qt == nullptr ||
             w2b.k16 == nullptr || w2b.vt == nullptr)
        bad = "null staging buffer";
    else if (in.q.p == nullptr || in.gate.p == nullptr || out.p == nullptr ||
             w.wk_b == nullptr || w.wv_b == nullptr)
        bad = "null input";
    else if (in.q.s_tok != sh.q_dim() || in.gate.s_tok != sh.n_head ||
             out.s_tok != sh.o_dim())
        bad = "input/output strides";
    else if ((reinterpret_cast<std::uintptr_t>(bufs.q16) % 16) != 0 ||
             (reinterpret_cast<std::uintptr_t>(w2b.k16) % 16) != 0 ||
             (reinterpret_cast<std::uintptr_t>(w2b.vt) % 16) != 0 ||
             (reinterpret_cast<std::uintptr_t>(bufs.o) % 16) != 0)
        bad = "staging buffer not 16-byte aligned";
    if (bad != nullptr) {
        std::fprintf(stderr, "ling3_mla_fw: REFUSING launch: %s\n", bad);
        return false;
    }
    return true;
}

bool mla_fw_attn(const MlaShape & sh, const MlaScalars & sc, const MlaWeights & w,
                 int n_pos, const MlaInputs & in, const View & out,
                 const MlaW2Bufs & w2bufs, MlaFwBufs & bufs,
                 int tok_base, int tok_count, int chunk_tokens,
                 hipStream_t stream) {
    if (!fw_audit(sh, w, n_pos, in, out, w2bufs, bufs, tok_base, tok_count,
                  chunk_tokens)) {
        return false;
    }
    const int base = n_pos - chunk_tokens;
    const int R = tok_count * sh.n_head;
    const float scale = sc.kq_scale != 0.0f ? sc.kq_scale
                                            : 1.0f / sqrtf((float) sh.head_k);

    // 1. q transpose + q16 rope tail (one thread per row)
    hipLaunchKernelGGL(fw_qtrans_kernel,
                       dim3((unsigned) ((R + 255) / 256)), 256, 0, stream,
                       in.q.p, bufs.qt, bufs.q16,
                       sh.n_head, sh.head_k, sh.nope, sh.kv, sh.rot,
                       in.q.s_tok, tok_base, tok_count, scale);
    LING3_FW_HIP_CHECK(hipGetLastError());

    // 2. absorb: q16[:, 0:512] = f16(scale * wk_b . q_nope) (WMMA GEMM)
    hipLaunchKernelGGL(fw_pack_kernel,
                       dim3((unsigned) (sh.n_head *
                                         ((tok_count + FW_PK_T - 1) / FW_PK_T))),
                       256, 0, stream,
                       w.wk_b, bufs.qt, bufs.q16,
                       sh.n_head, sh.nope, sh.kv, sh.rot,
                       tok_base, tok_count, scale);
    LING3_FW_HIP_CHECK(hipGetLastError());

    // 3. the fused flash kernel — ONE launch for the whole chunk
    hipLaunchKernelGGL(fw_attn_kernel, dim3((unsigned) tok_count), FW_THREADS,
                       0, stream,
                       bufs.q16, w2bufs.k16, w2bufs.vt, bufs.o, bufs.rowsum,
                       sh.n_head, sh.kv_a(), sh.kv, w2bufs.cap,
                       base, tok_base);
    LING3_FW_HIP_CHECK(hipGetLastError());

    // 4. de-absorb + gate (WMMA GEMM)
    hipLaunchKernelGGL(fw_deabsorb_kernel,
                       dim3((unsigned) (sh.n_head *
                                        ((tok_count + FW_DA_T - 1) / FW_DA_T))),
                       256, 0, stream,
                       w.wv_b, bufs.o, bufs.rowsum, in.gate.p, out.p,
                       sh.n_head, sh.head_v, sh.kv,
                       tok_base, tok_count, in.gate.s_tok, out.s_tok);
    LING3_FW_HIP_CHECK(hipGetLastError());
    return true;
}

const char * mla_fw_version() { return "ling3-mla-fw/1.2-lx1"; }

} // namespace ling3
