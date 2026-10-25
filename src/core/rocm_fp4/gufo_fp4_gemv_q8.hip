// LS1 2026-10-01 — the decode-step GEMV as an integer dot against a Q8_1
// activation.
//
// Why this is faster than k_gemv_wide (which stays for the non-FP4 types):
// per 32 weights the float path issued 32 separate byte loads (the ROCmFP4 block
// is 17 bytes, so no wider load is legal at the block level) plus a ue4m3 decode
// and an FMA for each one. This path issues FOUR 32-bit loads of the block's 16
// nibble bytes, expands them to int8 through the fork's perm codebook, and does
// EIGHT sdot4. That is roughly a third of the instructions for the same bytes,
// and the FP4 scale is applied once per 32 instead of once per element.
//
// Numerics: the activation is quantised to int8 with ggml's quantize_row_q8_1
// recipe (max|x| over each 32, scale = max/127, round(x/scale)), and the FP4 code
// table is exactly gufo_fp4_rocmfp4_code (0,1,2,3,4,6,8,10 / -...), unscaled. This is
// the fork's / ggml's computation, not gufo's float one.
//
// There is deliberately NO cache of "which vector is currently quantised". A
// pointer-keyed one would be wrong: rocm_backend.cpp reuses st->normed for every
// layer, so the second layer's rms_norm overwrites the bytes behind the pointer
// the first layer quantised, and a pointer match would then mean the wrong data.
// Every entry point here either quantises the vector it was handed or is handed
// a buffer that the caller filled for that exact call. Slots exist only so the
// expert FFN can reuse one quantisation across several matvecs it owns itself.

#include "gufo_fp4_quant.hpp"

#include <hip/hip_runtime.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <condition_variable>
#include <map>
#include <mutex>
#include <set>
#include <vector>
namespace gufo_fp4_ks {

static int ls1_chk(const char * what) {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        fprintf(stderr, "ling3 q8 %s: %s\n", what, hipGetErrorString(e));
        return GUFO_FP4_ERR_HIP;
    }
    return GUFO_FP4_OK;
}

// ---------------------------------------------------------------------------
// FP4 nibble -> int8 codebook.
//
// Verbatim from the fork, ggml/rocmfp4/rocmfp4_hip_codebook.cuh
// (rocmfp4_get_int_from_codebook_16, its GGML_USE_HIP branch). Four nibbles go
// in; int2 of eight int8 comes out: .x is the LOW nibbles of bytes 0..3, which
// are elements 0..3 of the block, .y is the HIGH nibbles, elements 16..19. The
// table [0,1,2,3],[4,6,8,10],[0,-1,-2,-3],[-4,-6,-8,-10] is exactly
// gufo_fp4_rocmfp4_code(nib), and tools/gufo_fp4_gemv_q8_test.cpp checks all 65536
// four-nibble words against that predicate on the device.
//
// Two things about the expression that look wrong and are not, both confirmed on
// the GPU (a perm probe and the exhaustive selftest):
//   * the perm's LOW selector is its SECOND argument and its HIGH selector is its
//     FIRST (indices 0..3 from arg2, 4..7 from arg1), which is why the fork passes
//     (values1, values0) and not the other way round;
//   * `q_odd = q4 >> 4` shifts across byte boundaries, but both uses mask with
//     0x07070707 / 0x08080808, which land entirely inside the source byte, so the
//     nibbles it reads are still the high ones. v_odd_low / v_odd_high below are
//     computed and never used -- that is the fork's code, kept so the two stay
//     line-for-line comparable.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int2 ls1_fp4_codebook16(int q4) {
    constexpr uint32_t values0 = 0x03020100u;
    constexpr uint32_t values1 = 0x0a080604u;
    constexpr uint32_t values2 = 0xfdfeff00u;
    constexpr uint32_t values3 = 0xf6f8fafcu;

    const uint32_t q_even = (uint32_t) q4;
    const uint32_t q_odd  = (uint32_t) q4 >> 4;

    const uint32_t v_even_low  = __builtin_amdgcn_perm(values1, values0, q_even & 0x07070707u);
    const uint32_t v_odd_low   = __builtin_amdgcn_perm(values1, values0, q_odd  & 0x07070707u);
    const uint32_t v_even_high = __builtin_amdgcn_perm(values3, values2, q_even & 0x07070707u);
    const uint32_t v_odd_high  = __builtin_amdgcn_perm(values3, values2, q_odd  & 0x07070707u);

    const uint32_t mask_even = 0x03020100u | ((q_even & 0x08080808u) >> 1);
    const uint32_t mask_odd  = 0x03020100u | ((q_odd  & 0x08080808u) >> 1);

    return make_int2(
        __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even),
        __builtin_amdgcn_perm(v_odd_high,  v_odd_low,  mask_odd));
}

// Read the 4 nibble bytes at p as a little-endian 32-bit word.
//
// The fork does this with one direct `*(const int *)p` (its
// GGML_ROCMFP4_UNALIGNED_QS_DWORD_LOAD, on for HIP), which depends on the
// hardware accepting an unaligned 32-bit global load. The first revision of this
// file did exactly that and produced a GEMV wrong against the CPU reference by
// factors of 3 to 20 on every shape, with no fault and nothing in the log to
// point at it -- the compiler will not tell you whether it honoured the
// alignment, and the address is only 4-aligned for one block in four. Four byte
// loads instead, assembled explicitly; the compiler is free to fold them back
// into whatever it likes, and the byte order no longer depends on the pointer.
__device__ __forceinline__ int ls1_ld32(const uint8_t * p) {
    return (int) ((uint32_t) p[0] | ((uint32_t) p[1] << 8) |
                  ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24));
}

// One weight block, decoded once: the two packed int8 halves and the two FP4 scales.
//
// The codebook expansion (four v_perm) is the expensive half of a block, and in the
// small-m matvec below it is done ONCE per block and reused for every activation column.
// That sharing is the whole point of the m-column form: at m = 3 the weight bytes and
// the perms are paid for once and only the eight sudot4 per column are extra.
// FOUR dwords, not one: the block's 16 nibble bytes are four independent groups of four
// elements, so each dword carries its own codes. An earlier version of this struct held a
// single vx/vy expanded from bytes 0..3 and reused it for all four dwords, which silently
// scaled every row by ~4 on three quarters of the elements -- the phase-1 GEMV gate caught
// it immediately, and that is the only reason this comment exists.
struct ls1_blk32 {
    int vx[4], vy[4];    // per dword: codes of its 4 low-nibble and 4 high-nibble elements
    float slo, shi;
};

// Reads the block's 16 nibble bytes and its 1 or 2 scale bytes -- offsets 0..17 of a 17
// or 18 byte block, and nothing else, so a row touches exactly row_bytes bytes and can
// never read past the last row of a tightly sized tensor.
__device__ __forceinline__ ls1_blk32 ls1_load_blk32(int qtype, const uint8_t * blk) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    ls1_blk32 c;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int2 v = ls1_fp4_codebook16(ls1_ld32(blk + 4 * l));
        c.vx[l] = v.x;
        c.vy[l] = v.y;
    }
    c.slo = gufo_fp4_ue4m3_half(blk[16]);
    c.shi = (qtype == GUFO_FP4_Q4_0_ROCMFP4) ? gufo_fp4_ue4m3_half(blk[17]) : c.slo;
    return c;
}

// The decoded block dotted with one activation block. The codebook is unscaled, so the
// only factor left is the activation's own.
__device__ __forceinline__ float ls1_dot_blk32(const ls1_blk32 & c, const int8_t * xq32,
                                               float xd) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int * q8 = (const int *) xq32;
    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        // sudot4, not sdot4: this ROCm's clang gates sdot4 behind a target feature the
        // project's flags do not set, and the fork uses the same call (ggml_cuda_dp4a,
        // ggml/src/ggml-cuda/common.cuh:720, RDNA3/RDA4 branch). Verified on the device as
        // a bilinear form -- lane i pairs with lane i, both operands signed:
        // sudot4(true, 0x01020304, true, 0x04030201, 0, false) == 20 == 4 + 6 + 6 + 4.
        // .x against the first 16 elements of the block, .y against the last 16 --
        // which is why the second dot takes v.y and not v.x. Getting that wrong is
        // silent: it scales every row by roughly 4 and faults nothing.
        sumi0 = (int) __builtin_amdgcn_sudot4(true, c.vx[l], true, q8[l + 0], sumi0, false);
        sumi1 = (int) __builtin_amdgcn_sudot4(true, c.vy[l], true, q8[l + 4], sumi1, false);
    }
    return xd * (c.slo * (float) sumi0 + c.shi * (float) sumi1);
}

// One weight block (32 values, tsz bytes) dotted with one activation block: decode and
// dot in one call. This is the m = 1 path, and the small-m kernel below is the same maths
// with the decode hoisted out of the column loop.
__device__ __forceinline__ float ls1_dot_fp4_blk32_q8(int qtype, const uint8_t * blk,
                                                      const int8_t * xq32, float xd) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int * q8 = (const int *) xq32;
    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int2 v = ls1_fp4_codebook16(ls1_ld32(blk + 4 * l));
        // sudot4, not sdot4: this ROCm's clang gates sdot4 behind a target feature the
        // project's flags do not set, and the fork uses the same call (ggml_cuda_dp4a,
        // ggml/src/ggml-cuda/common.cuh:720, RDNA3/RDA4 branch). Verified on the device as
        // a bilinear form -- lane i pairs with lane i, both operands signed:
        // sudot4(true, 0x01020304, true, 0x04030201, 0, false) == 20 == 4 + 6 + 6 + 4.
        // .x against the first 16 elements of the block, .y against the last 16 --
        // which is why the second dot takes v.y and not v.x. Getting that wrong is
        // silent: it scales every row by roughly 4 and faults nothing.
        sumi0 = (int) __builtin_amdgcn_sudot4(true, v.x, true, q8[l + 0], sumi0, false);
        sumi1 = (int) __builtin_amdgcn_sudot4(true, v.y, true, q8[l + 4], sumi1, false);
    }
    const float slo = gufo_fp4_ue4m3_half(blk[16]);
    const float shi = (qtype == GUFO_FP4_Q4_0_ROCMFP4) ? gufo_fp4_ue4m3_half(blk[17]) : slo;
    // The codebook is unscaled, so the only factor left is the activation's own.
    return xd * (slo * (float) sumi0 + shi * (float) sumi1);
}

// ---------------------------------------------------------------------------
// Q8_1 activation buffers, one pair per slot.
//
//   qs[0 .. k)     int8, element i at qs[i]
//   ds[0 .. k/32)  float, the scale of qs[32*b .. 32*b+32)
//
// The ROCmFP4 weight block is also 32 values, so weight block b pairs with ds[b].
// ---------------------------------------------------------------------------
enum {
    GUFO_FP4_LS1_SLOT_GENERIC = 0,   // gufo_fp4_gemv_fp4()'s own scratch
    GUFO_FP4_LS1_SLOT_FFN_X  = 1,    // the expert FFN's input, shared by gate and up
    GUFO_FP4_LS1_SLOT_FFN_H  = 2,    // the expert FFN's SwiGLU output, input to down
    GUFO_FP4_LS1_SLOT_MOE_H  = 3,    // the grouped expert FFN's SwiGLU output, input to down
    GUFO_FP4_LS1_SLOT_ATTN   = 4,    // st->normed, once per layer, shared by its projections
    GUFO_FP4_LS1_NSLOT      = 5
};

namespace {

// Hard ceiling: 64 streams x ~462 KiB (27B) is ~29 MiB of scratch, and no seat
// runs that many sessions; past it the entry refuses, it never aliases.
enum { GUFO_FP4_LS1_MAX_SETS = 64 };

// ---------------------------------------------------------------------------
// SQ10: slot SETS, one per decode stream. Every buffer below used to be a
// single process-global pair, so two decode streams interleaving quantize ->
// GEMV read each other's activations and `-j 1` was the only safe seat. A set
// is the same buffers with an owner: streams attach 1:1 (serve pre-attaches
// each arena stream at load; any other stream attaches lazily on first use),
// and a session's helper (drafter) stream is aliased to its owner's set
// (SPEC2 R1). No free set -> the entry refuses at once with -GUFO_FP4_ERR_BUSY
// (SPEC2 R2) and the caller declines to its eager fallback; nothing in
// production waits. Growth stays per-set with the same kernels, so the `-j 1`
// path is bit-identical and unwarmed callers (tools, tests, the Flash-Next
// eager-first-pass) behave exactly as before; warmup pre-grows every set so
// no captured path allocates. alloc_count counts every slot hipMalloc and the
// test asserts it does not move across decode.
// SQ19-2: every load-time attach gets a set, configured or not. The load-time
// entry (gufo_fp4_slot_stream_attach_grow, from the executor constructor)
// reuses a free set and otherwise grows the table by one, up to the same
// ceiling; the caller warms the set's buffers before any step. Past the cap
// it refuses exactly like an in-launch attach. In-launch attach keeps
// refusing (R2) and is the only refusal left; configure() only pre-sizes.
// ---------------------------------------------------------------------------
struct GufoFp4SlotSet {
    int8_t * qs[GUFO_FP4_LS1_NSLOT] = {};
    float * ds[GUFO_FP4_LS1_NSLOT] = {};
    int cap[GUFO_FP4_LS1_NSLOT] = {};
    float * moe_hs = nullptr;
    int moe_hs_cap = 0;
    int * moe_rc = nullptr;
    int8_t * qs_m = nullptr;
    float * ds_m = nullptr;
    int cap_m = 0;
    float * moe_hs_m = nullptr;
    int moe_hs_m_cap = 0;
    int * moe_rc_m = nullptr;
    int8_t * qv = nullptr;
    float * dv = nullptr;
    int qv_cap = 0;
    // Super ROC QVO: the input the qv/dv pair holds, as quantised by the last
    // gufo_fp4_gemv_verify_m in this set (cleared when the pair is replaced).
    const float * vq_x = nullptr;
    int vq_m = 0;
    int vq_k = 0;
};

struct GufoFp4SlotMgr {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<GufoFp4SlotSet> sets = std::vector<GufoFp4SlotSet>(1);
    std::vector<char> busy = std::vector<char>(1, 0);
    std::map<hipStream_t, int> stream_set;
    // SQ10 SPEC2 (R1): helper stream -> owner (session) stream. A drafter or
    // helper stream of a session uses that session's set; the host alternates
    // the two streams (the helper's last launch is synced before the session's
    // next launch), so they never hold the set at once. One hop only: an
    // owner is never itself aliased.
    std::map<hipStream_t, hipStream_t> alias;
    // SQ10 SPEC2 (R2): streams already warned about a refusal, and the
    // lifetime refusal count. Production never waits; it refuses at once.
    std::set<hipStream_t> warned;
    unsigned long alloc_count = 0;
    unsigned long wait_count = 0;
    unsigned long wait_us = 0;
    unsigned long refuse_count = 0;
    bool configured = false;
};

inline GufoFp4SlotMgr & gufo_fp4_slots_mgr(void) {
    static GufoFp4SlotMgr mgr;
    return mgr;
}

// Counted malloc. Call with mgr.mu held.
static int gufo_fp4_slots_malloc(void ** p, size_t n) {
    if (hipMalloc(p, n) != hipSuccess) {
        *p = nullptr;
        return GUFO_FP4_ERR_HIP;
    }
    gufo_fp4_slots_mgr().alloc_count++;
    return GUFO_FP4_OK;
}

// Resolve stream -> set, attaching a free set when the stream is unseen. An
// aliased helper stream resolves to its owner's set (R1). When no set is
// free the stream is REFUSED at once with -GUFO_FP4_ERR_BUSY (R2): one WARN
// per stream on stderr plus a lifetime counter; production entries decline
// to their eager fallback exactly as for an unwarmed width. Host-side only
// (mutexes and map lookups), so calling it inside a captured step is legal;
// it performs no HIP call. SPEC3 (S3-5): the WARN names the declining
// launch (function + tensor width) so a fallback is diagnosable.
// SQ19-2: grow_at_load (the load-time attach entry only) grows the table by
// one set instead of refusing, configured or not. The new set's buffers are
// empty: the load-time caller warms them before any step, never here (a
// hipMalloc inside a captured step poisons the capture). In-launch callers
// pass false and keep refusing.
static int gufo_fp4_slots_for_stream_at(hipStream_t stream, const char * func, long width,
                                        bool grow_at_load) {
    const char * const who = (func != nullptr && func[0] != '\0') ? func : "?";
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::unique_lock<std::mutex> lk(mgr.mu);
    const std::map<hipStream_t, int>::const_iterator it = mgr.stream_set.find(stream);
    if (it != mgr.stream_set.end()) {
        return it->second;
    }
    const std::map<hipStream_t, hipStream_t>::const_iterator al = mgr.alias.find(stream);
    if (al != mgr.alias.end()) {
        const std::map<hipStream_t, int>::const_iterator own = mgr.stream_set.find(al->second);
        if (own != mgr.stream_set.end()) {
            return own->second;
        }
        // The owner is gone (destroyed out of order): fail closed. Detach
        // drops aliases pointing at a destroyed owner, so this is reachable
        // only across a destroy race; refusing is still the safe answer.
    }
    for (size_t i = 0; i < mgr.sets.size(); ++i) {
        if (!mgr.busy[i]) {
            mgr.busy[i] = 1;
            mgr.stream_set[stream] = (int) i;
            return (int) i;
        }
    }
    if (grow_at_load && mgr.sets.size() < (size_t) GUFO_FP4_LS1_MAX_SETS) {
        const int grown = (int) mgr.sets.size();
        mgr.sets.resize(mgr.sets.size() + 1);
        mgr.busy.push_back(1);
        mgr.stream_set[stream] = grown;
        return grown;
    }
    mgr.refuse_count++;
    const unsigned long refusals = mgr.refuse_count;
    const bool first = mgr.warned.insert(stream).second;
    lk.unlock();
    if (first) {
        fprintf(stderr, "GUFO_FP4_SLOT_REFUSE stream=%p func=%s width=%ld refusals=%lu\n",
                (const void *) stream, who, width, refusals);
    }
    return -GUFO_FP4_ERR_BUSY;
}

// The waiting resolver. Production never calls this (R2); it survives only
// for the unit test's exhaustion case, which proves the wait/wake machinery.
// Same resolution as above, but an unseen stream with no free set blocks on
// the condvar until a detach frees one. Waits are counted (and, with
// GUFO_FP4_SLOT_TRACE=1 in the environment, logged with their length).
static int gufo_fp4_slots_for_stream_wait(hipStream_t stream) {
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::unique_lock<std::mutex> lk(mgr.mu);
    const std::map<hipStream_t, int>::const_iterator it = mgr.stream_set.find(stream);
    if (it != mgr.stream_set.end()) {
        return it->second;
    }
    const std::map<hipStream_t, hipStream_t>::const_iterator al = mgr.alias.find(stream);
    if (al != mgr.alias.end()) {
        const std::map<hipStream_t, int>::const_iterator own = mgr.stream_set.find(al->second);
        if (own != mgr.stream_set.end()) {
            return own->second;
        }
    }
    for (size_t i = 0; i < mgr.sets.size(); ++i) {
        if (!mgr.busy[i]) {
            mgr.busy[i] = 1;
            mgr.stream_set[stream] = (int) i;
            return (int) i;
        }
    }
    static const bool trace = getenv("GUFO_FP4_SLOT_TRACE") != nullptr;
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        mgr.cv.wait(lk);
        for (size_t i = 0; i < mgr.sets.size(); ++i) {
            if (!mgr.busy[i]) {
                mgr.busy[i] = 1;
                mgr.stream_set[stream] = (int) i;
                struct timespec t1;
                clock_gettime(CLOCK_MONOTONIC, &t1);
                const unsigned long us =
                    (unsigned long) (t1.tv_sec - t0.tv_sec) * 1000000ul +
                    (unsigned long) (t1.tv_nsec - t0.tv_nsec) / 1000ul;
                mgr.wait_count++;
                mgr.wait_us += us;
                if (trace) {
                    fprintf(stderr, "GUFO_FP4_SLOT_WAIT set=%d us=%lu\n", (int) i, us);
                }
                return (int) i;
            }
        }
    }
}

// The four growth shapes, one place each (they used to be open-coded at every
// call site). Each syncs the stream before freeing, so a growth can never free
// a buffer a queued kernel still reads; growth never runs inside a capture
// because warmup pre-grows every set.
static int gufo_fp4_slots_grow_q8(int set, int slot, int k, hipStream_t stream,
                                  int8_t ** qs, float ** ds) {
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    GufoFp4SlotSet & S = mgr.sets[(size_t) set];
    if (k > S.cap[slot] || S.qs[slot] == nullptr || S.ds[slot] == nullptr) {
        if (hipStreamSynchronize(stream) != hipSuccess) return GUFO_FP4_ERR_HIP;
        if (S.qs[slot]) {
            if (hipFree(S.qs[slot]) != hipSuccess) return GUFO_FP4_ERR_HIP;
            S.qs[slot] = nullptr;
        }
        if (S.ds[slot]) {
            if (hipFree(S.ds[slot]) != hipSuccess) return GUFO_FP4_ERR_HIP;
            S.ds[slot] = nullptr;
        }
        S.cap[slot] = 0;
        // +GUFO_FP4_TAIL_PAD so a widened tail load still has readable bytes behind
        // the last element, the same contract every other activation has here.
        if (gufo_fp4_slots_malloc((void **) &S.qs[slot], (size_t) k + GUFO_FP4_TAIL_PAD) != GUFO_FP4_OK) {
            return GUFO_FP4_ERR_HIP;
        }
        if (gufo_fp4_slots_malloc((void **) &S.ds[slot],
                                  (size_t) (k / 32) * sizeof(float) + GUFO_FP4_TAIL_PAD) != GUFO_FP4_OK) {
            (void) hipFree(S.qs[slot]);
            S.qs[slot] = nullptr;
            return GUFO_FP4_ERR_HIP;
        }
        S.cap[slot] = k;
    }
    *qs = S.qs[slot];
    *ds = S.ds[slot];
    return GUFO_FP4_OK;
}

static int gufo_fp4_slots_get_q8(int set, int slot, int k,
                                 int8_t ** qs, float ** ds) {
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    GufoFp4SlotSet & S = mgr.sets[(size_t) set];
    if (S.qs[slot] == nullptr || S.cap[slot] < k) return GUFO_FP4_ERR_ARG;
    *qs = S.qs[slot];
    *ds = S.ds[slot];
    return GUFO_FP4_OK;
}

static int gufo_fp4_slots_grow_hs(int set, size_t nn, hipStream_t stream,
                                  float ** hs, int ** rc) {
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    GufoFp4SlotSet & S = mgr.sets[(size_t) set];
    if ((int) nn > S.moe_hs_cap || S.moe_hs == nullptr || S.moe_rc == nullptr) {
        if (hipStreamSynchronize(stream) != hipSuccess) return GUFO_FP4_ERR_HIP;
        if (S.moe_hs) {
            if (hipFree(S.moe_hs) != hipSuccess) return GUFO_FP4_ERR_HIP;
            S.moe_hs = nullptr;
        }
        if (S.moe_rc) {
            if (hipFree(S.moe_rc) != hipSuccess) return GUFO_FP4_ERR_HIP;
            S.moe_rc = nullptr;
        }
        S.moe_hs_cap = 0;
        if (gufo_fp4_slots_malloc((void **) &S.moe_hs, nn * sizeof(float) + GUFO_FP4_TAIL_PAD) != GUFO_FP4_OK) {
            return GUFO_FP4_ERR_HIP;
        }
        if (gufo_fp4_slots_malloc((void **) &S.moe_rc, sizeof(int)) != GUFO_FP4_OK) {
            (void) hipFree(S.moe_hs);
            S.moe_hs = nullptr;
            return GUFO_FP4_ERR_HIP;
        }
        S.moe_hs_cap = (int) nn;
        if (hipMemset(S.moe_rc, 0, sizeof(int)) != hipSuccess) return GUFO_FP4_ERR_HIP;
    }
    *hs = S.moe_hs;
    *rc = S.moe_rc;
    return GUFO_FP4_OK;
}

static int gufo_fp4_slots_grow_q8m(int set, int total, hipStream_t stream,
                                   int8_t ** qs, float ** ds) {
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    GufoFp4SlotSet & S = mgr.sets[(size_t) set];
    if (total > S.cap_m || S.qs_m == nullptr || S.ds_m == nullptr) {
        if (hipStreamSynchronize(stream) != hipSuccess) return GUFO_FP4_ERR_HIP;
        if (S.qs_m) {
            if (hipFree(S.qs_m) != hipSuccess) return GUFO_FP4_ERR_HIP;
            S.qs_m = nullptr;
        }
        if (S.ds_m) {
            if (hipFree(S.ds_m) != hipSuccess) return GUFO_FP4_ERR_HIP;
            S.ds_m = nullptr;
        }
        S.cap_m = 0;
        if (gufo_fp4_slots_malloc((void **) &S.qs_m, (size_t) total + GUFO_FP4_TAIL_PAD) != GUFO_FP4_OK) {
            return GUFO_FP4_ERR_HIP;
        }
        if (gufo_fp4_slots_malloc((void **) &S.ds_m,
                                  (size_t) (total / 32) * sizeof(float) + GUFO_FP4_TAIL_PAD) != GUFO_FP4_OK) {
            (void) hipFree(S.qs_m);
            S.qs_m = nullptr;
            return GUFO_FP4_ERR_HIP;
        }
        S.cap_m = total;
    }
    *qs = S.qs_m;
    *ds = S.ds_m;
    return GUFO_FP4_OK;
}

static int gufo_fp4_slots_grow_hsm(int set, size_t nn, hipStream_t stream,
                                   float ** hs, int ** rc) {
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    GufoFp4SlotSet & S = mgr.sets[(size_t) set];
    if ((int) nn > S.moe_hs_m_cap || S.moe_hs_m == nullptr || S.moe_rc_m == nullptr) {
        if (hipStreamSynchronize(stream) != hipSuccess) return GUFO_FP4_ERR_HIP;
        if (S.moe_hs_m) {
            if (hipFree(S.moe_hs_m) != hipSuccess) return GUFO_FP4_ERR_HIP;
            S.moe_hs_m = nullptr;
        }
        if (S.moe_rc_m) {
            if (hipFree(S.moe_rc_m) != hipSuccess) return GUFO_FP4_ERR_HIP;
            S.moe_rc_m = nullptr;
        }
        S.moe_hs_m_cap = 0;
        if (gufo_fp4_slots_malloc((void **) &S.moe_hs_m, nn * sizeof(float) + GUFO_FP4_TAIL_PAD) != GUFO_FP4_OK) {
            return GUFO_FP4_ERR_HIP;
        }
        if (gufo_fp4_slots_malloc((void **) &S.moe_rc_m, sizeof(int)) != GUFO_FP4_OK) {
            (void) hipFree(S.moe_hs_m);
            S.moe_hs_m = nullptr;
            return GUFO_FP4_ERR_HIP;
        }
        S.moe_hs_m_cap = (int) nn;
        if (hipMemset(S.moe_rc_m, 0, sizeof(int)) != hipSuccess) return GUFO_FP4_ERR_HIP;
    }
    *hs = S.moe_hs_m;
    *rc = S.moe_rc_m;
    return GUFO_FP4_OK;
}

// ggml's quantize_row_q8_1, one THREAD per 32 values.
//
// Deliberately the plainest possible transcription of the reference recipe:
// serial loop, no warp shuffle, no cross-lane anything. A revision that reduced
// the maximum with __shfl_down produced the right ds but codes that disagreed
// with the CPU reference on roughly a quarter of the elements, so the bug lived
// somewhere in the shuffle that a straight transcription cannot have. This
// kernel runs at most 9 times per decoded token over at most 10 KB, so the
// coalescing it gives up is worth far less than being wrong.
//
// The signed maximum (not the magnitude) sets the scale, exactly as ggml does,
// so a block that is entirely negative gets a negative scale and its codes come
// out positive. Which of two equal-magnitude elements wins the tie does not
// matter: the two choices give scales that differ by a sign and codes that
// differ by that same sign, so the product ds * qs is the same.
__global__ void gufo_fp4_k_q8_1_quant(const float * x, int8_t * qs, float * ds, int nb) {
    const int b = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (b >= nb) {
        return;
    }
    const float * xb = x + (size_t) b * 32;
    float amax = 0.f;
    float max = 0.f;
    for (int j = 0; j < 32; ++j) {
        const float v = xb[j];
        const float a = fabsf(v);
        if (a > amax) {
            amax = a;
            max = v;
        }
    }
    const float d = max / 127.f;
    const float id = d != 0.f ? 1.0f / d : 0.f;
    ds[b] = d;
    int8_t * qb = qs + (size_t) b * 32;
    for (int j = 0; j < 32; ++j) {
        qb[j] = (int8_t) roundf(xb[j] * id);
    }
}

}  // namespace

// Quantise x[0..k) into slot `slot` of the calling stream's set. Returns 0 on success.
extern "C" int gufo_fp4_quant_q8(const float * x, int k, int slot, hipStream_t stream) {
    if (x == nullptr || k <= 0 || (k % 32) != 0 || slot < 0 || slot >= GUFO_FP4_LS1_NSLOT) {
        return GUFO_FP4_ERR_ARG;
    }
    const int set = gufo_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // R2 refusal: the caller declines, never waits.
    int8_t * qs = nullptr;
    float * ds = nullptr;
    const int rc = gufo_fp4_slots_grow_q8(set, slot, k, stream, &qs, &ds);
    if (rc != GUFO_FP4_OK) return rc;
    const int nb = k / 32;
    const int threads = 256;
    const int blocks = (nb + threads - 1) / threads;
    gufo_fp4_k_q8_1_quant<<<blocks, threads, 0, stream>>>(x, qs, ds, nb);
    return ls1_chk("q8 quant");
}

// One row of the integer matvec, before the cross-lane reduction: thread `threadIdx.x` and
// its stride partners walk the row's 32-value blocks. Kept separate from the kernel so the
// grouped expert FFN can use the same dot with a different thread mapping -- and, in phase 3,
// so that mapping's reduction goes through the same ls1_block_sum rather than a 32-lane one.
__device__ __forceinline__ float ls1_dot_row_q8(int qtype, const uint8_t * w,
                                                const int8_t * xq, const float * xd,
                                                int k) {
    const int tsz = gufo_fp4_type_size(qtype);
    const int nblk = k / 32;
    float s = 0.f;
    for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
        s += ls1_dot_fp4_blk32_q8(qtype, w + (size_t) qb * (size_t) tsz,
                                  xq + (size_t) qb * 32, xd[qb]);
    }
    return s;
}

// ---------------------------------------------------------------------------
// The matvec itself.
//
// One 64-thread CTA per output row: 64x the parallelism of a thread per row, the
// same shape k_gemv_wide uses, so the only thing that changes is what happens
// inside the loop. Threads stride over the row's 32-value blocks, so the 64
// lanes of one step cover 64 consecutive blocks = 1088 contiguous bytes; each
// lane's four 32-bit loads land in those same lines, so nothing is fetched
// twice and nothing is fetched that is not used.
//
// The reduction is warp shuffles plus one LDS slot per warp, not the five
// __syncthreads of the LDS tree: with nblk small relative to the CTA the tree
// was costing more than the loop it was reducing.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float ls1_warp_sum(float s) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down(s, off);
    }
    return s;
}

// Sum over the whole CTA, in EVERY thread.
//
// ls1_warp_sum on its own is only a 32-lane reduction: __shfl_down with offsets 16..1
// brings lanes 0..31 together and thread 0 never sees lanes 32..63. That is fine inside
// k_gemv_wide_q8, which follows it with the LDS slots and a __syncthreads, and it is
// exactly what the two grouped MoE kernels were missing in phase 2 -- they are launched
// with 64-thread CTAs, so every row silently dropped half its work. At k = 2560 that is
// 80 blocks spread over all 64 lanes and the result was wrong by 3-6x with no fault;
// at k = 512 it is 16 blocks, all on lanes 0-15, and the bug is invisible. Hence this
// helper, and hence the test running n_used = 1 AND n_used = 8 at the SAME real shape.
//
// `red` needs blockDim.x/32 floats and is reusable across calls: the trailing
// __syncthreads() is what frees it. Both barriers are reached by every thread, so the
// caller must not return early around a call.
__device__ __forceinline__ float ls1_block_sum(float s, float * red) {
    const int nw = (int) blockDim.x >> 5;
    s = ls1_warp_sum(s);
    if ((int) (threadIdx.x & 31) == 0) {
        red[threadIdx.x >> 5] = s;
    }
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < nw; ++i) {
        t += red[i];
    }
    __syncthreads();
    return t;
}

// The phase-1 kernel, kept because it is what the unit test compares the warp-per-row
// one against: gufo_fp4_k_gemv_wide_q8_rows must be BIT-IDENTICAL to this.
__global__ void gufo_fp4_k_gemv_wide_q8_cta64(int qtype, const uint8_t * W, size_t row_bytes,
                               const int8_t * xq, const float * xd,
                               float * y, int k, int n) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int row = (int) blockIdx.x;
    if (row >= n) {
        return;
    }
    const uint8_t * w = W + (size_t) row * row_bytes;
    const int tsz = gufo_fp4_type_size(qtype);
    const int nblk = k / 32;
    __shared__ float red[32];

    float s = 0.f;
    for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
        s += ls1_dot_fp4_blk32_q8(qtype, w + (size_t) qb * (size_t) tsz,
                                  xq + (size_t) qb * 32, xd[qb]);
    }
    s = ls1_block_sum(s, red);
    if (threadIdx.x == 0) {
        y[row] = s;
    }
}

// ---------------------------------------------------------------------------
// PHASE6: one WARP per row, several consecutive rows per CTA.
//
// The measured shape of the two weak decode GEMVs (REPORT5 §3, from the build21
// profile) is a lane-utilisation problem, not a bandwidth one:
//
//   gufo_fp4_k_moe_down_weighted   n_ff = 768  -> 24 blocks of 32 values, spread over a 64-lane
//                                     CTA: 24 of 64 lanes load anything, 37.5 %.
//   k_gemv_wide_q8        k    = 2560 -> 80 blocks over 64 lanes: 80 of 128 lane-slots,
//                                     62.5 %, the tail wasted on a second iteration.
//
// One warp per row fixes both: 24/32 = 75 % and 80/96 = 83 % lane occupancy. It
// also drops the CTA-wide reduction entirely -- a warp reduction is a shuffle tree
// with no LDS and no __syncthreads -- and because a CTA's warps take CONSECUTIVE
// rows, its reads span 8 x 1360 = 10,880 contiguous bytes instead of 8 separate
// 1360-byte strided ones, which is what keeps the 64-byte lines fully used even
// though a 1360-byte row is not a multiple of 64 and so straddles lines.
//
// The row loop is a grid-stride, so the grid can be sized to the machine rather
// than to n. No CTA-wide barrier exists here, which is also why `row >= n` may
// return per warp without any risk of a hang -- the old CTA-per-row kernels had
// to keep that check uniform for the sake of their __syncthreads.
__global__ void gufo_fp4_k_gemv_wide_q8_rows(int qtype, const uint8_t * W, size_t row_bytes,
                                    const int8_t * xq, const float * xd,
                                    float * y, int k, int n, int grid_rows) {
    const int lane = threadIdx.x & 31;
    const int sub = (int) (threadIdx.x >> 5);
    const int nwarps = (int) blockDim.x >> 5;
    const int tsz = gufo_fp4_type_size(qtype);
    const int nblk = k / 32;
    const int stride = grid_rows * nwarps;
    for (int row = blockIdx.x * nwarps + sub; row < n; row += stride) {
        const uint8_t * w = W + (size_t) row * row_bytes;
        float s = 0.f;
        for (int qb = lane; qb < nblk; qb += 32) {
            s += ls1_dot_fp4_blk32_q8(qtype, w + (size_t) qb * (size_t) tsz,
                                      xq + (size_t) qb * 32, xd[qb]);
        }
        s = ls1_warp_sum(s);
        if (lane == 0) {
            y[row] = s;
        }
    }
}

// The same shape for the down projection: one warp per output row, the n_used
// slots looped in slot order with a warp reduction and no CTA barrier.
__global__ void gufo_fp4_k_moe_down_weighted_rows(int qtype, const uint8_t * down, size_t rb_ff,
                                         const int * ids, const float * weights,
                                         const int8_t * hq, const float * hd,
                                         float * y, int n_used, int n_expert,
                                         int k, int n_ff, int grid_rows, int * rc) {
    const int lane = threadIdx.x & 31;
    const int sub = (int) (threadIdx.x >> 5);
    const int nwarps = (int) blockDim.x >> 5;
    const int nb = n_ff / 32;
    const int stride = grid_rows * nwarps;
    for (int j = blockIdx.x * nwarps + sub; j < k; j += stride) {
        float acc = 0.f;
        for (int s = 0; s < n_used; ++s) {
            const int id = ids[s];
            const bool ok = (id >= 0 && id < n_expert);
            if (!ok && lane == 0) {
                *rc = GUFO_FP4_ERR_ARG;
            }
            const size_t base = ok ? (((size_t) id * (size_t) k + (size_t) j) * rb_ff) : 0;
            const uint8_t * drow = down + base;
            float part = 0.f;
            for (int qb = lane; qb < nb; qb += 32) {
                part += ls1_dot_fp4_blk32_q8(qtype, drow + (size_t) qb * (size_t) gufo_fp4_type_size(qtype),
                                            hq + (size_t) s * (size_t) n_ff + (size_t) qb * 32,
                                            hd[(size_t) s * (size_t) nb + (size_t) qb]);
            }
            part = ls1_warp_sum(part);
            if (lane == 0) {
                acc += weights[s] * part;
            }
        }
        if (lane == 0) {
            y[j] = acc;
        }
    }
}

// The LM head: n = 157184, so one CTA per row would be 157k CTAs of 64 threads
// for a 1360-byte job each and the scheduling cost would dominate. A fixed grid
// loops over rows, the way k_gemv_wide_gs does.
__global__ void gufo_fp4_k_gemv_wide_q8_gs(int qtype, const uint8_t * W, size_t row_bytes,
                                  const int8_t * xq, const float * xd,
                                  float * y, int k, int n) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int tsz = gufo_fp4_type_size(qtype);
    const int nblk = k / 32;
    __shared__ float red[32];

    for (int row = (int) blockIdx.x; row < n; row += (int) gridDim.x) {
        const uint8_t * w = W + (size_t) row * row_bytes;
        float s = 0.f;
        for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
            s += ls1_dot_fp4_blk32_q8(qtype, w + (size_t) qb * (size_t) tsz,
                                      xq + (size_t) qb * 32, xd[qb]);
        }
        // ls1_block_sum's own barriers handle red's reuse between rows.
        s = ls1_block_sum(s, red);
        if (threadIdx.x == 0) {
            y[row] = s;
        }
    }
}

// Below this many output rows a CTA per row still beats a grid-strided loop;
// above it there is already plenty of parallelism. Same cut as gufo_fp4_gemv.
#define GUFO_FP4_LS1_GEMV_WIDE_MAX 8192
#define GUFO_FP4_LS1_GEMV_GS_BLOCKS 2048

// Matvec against an activation the caller already quantised into `slot`.
// ---------------------------------------------------------------------------
// PHASE6 selection. Variant 0 is the phase-1 CTA-per-row kernel, unchanged; variant 1
// is the warp-per-row kernel. GUFO_FP4_GEMV_VARIANT picks it and DEFAULTS TO 0, so a
// build with no environment set behaves exactly as it did before this phase. The
// unit test runs both and requires them to agree bit-for-bit.
// ---------------------------------------------------------------------------
// LS1 PHASE7 item 4: the flag is SPLIT, because its two halves are not the same claim.
//
//   GUFO_FP4_GEMV_DOWN  (default ON)  -- the warp-per-row grouped expert FFN. MEASURED
//       bit-identical: max_rel_diff 0, 0 differing of 2560, in all three
//       configurations. There is no numerical risk in it.
//
//   GUFO_FP4_GEMV_GEMV  (default OFF) -- the warp-per-row plain matvec. It regroups a
//       row's float additions (float addition is not associative), so ~60 % of output
//       elements differ in their low bits and the greedy text diverges from build24 at
//       character 194. Its error against the CPU integer reference is the same ~1e-6
//       band as the kernel it replaces, and the perplexity gates pass (step 6.1014 vs
//       batched 6.1037 on positions 4096-8191; 4096-token step 9.2905 vs batched
//       9.3094) -- but end to end it bought 0.1 % (37.90/37.95 tok/s against 37.89/37.87
//       with it off), which does not pay for changing greedy output. Jay decides.
static int ls1_env_on(const char * name, int dflt) {
    const char * e = getenv(name);
    if (e == nullptr || e[0] == '\0') {
        return dflt;
    }
    return (e[0] == '1') ? 1 : 0;
}
static int ls1_variant_down() {
    static int v = -1;
    if (v < 0) {
        v = ls1_env_on("GUFO_FP4_GEMV_DOWN", 1);
        if (!v && ls1_env_on("GUFO_FP4_GEMV_VARIANT", 0)) {
            v = 1;   // the old single switch still turns both on
        }
    }
    return v;
}
static int ls1_variant_gemv() {
    static int v = -1;
    if (v < 0) {
        v = ls1_env_on("GUFO_FP4_GEMV_GEMV", 0);
        if (v == 0 && ls1_env_on("GUFO_FP4_GEMV_VARIANT", 0)) {
            v = 1;   // the old single switch still turns both on
        }
    }
    return v;
}

static int ls1_gemv_blocks(int n, int nwarps) {
    if (nwarps <= 0) return 1;
    const int want = (n + nwarps - 1) / nwarps;
    return want < 1 ? 1 : want;
}

// Variant 1 of the plain matvec, called directly so the test can time it against
// the CTA-per-row form at the same shapes. slot must already hold the Q8_1 vector.
extern "C" int gufo_fp4_gemv_fp4_q8_rows(int qtype, const void * W, int slot, float * y,
                                      int k, int n, int block, hipStream_t stream) {
    const size_t rb = gufo_fp4_row_bytes(qtype, k);
    if (!W || !y || rb == 0 || n <= 0 || (k % 32) != 0 || block < 32) return GUFO_FP4_ERR_ARG;
    if (slot < 0 || slot >= GUFO_FP4_LS1_NSLOT) return GUFO_FP4_ERR_ARG;
    const int set = gufo_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // R2 refusal: the caller declines, never waits.
    int8_t * qs = nullptr;
    float * ds = nullptr;
    if (gufo_fp4_slots_get_q8(set, slot, k, &qs, &ds) != GUFO_FP4_OK) return GUFO_FP4_ERR_ARG;
    const int nwarps = block >> 5;
    gufo_fp4_k_gemv_wide_q8_rows<<<ls1_gemv_blocks(n, nwarps), block, 0, stream>>>(
        qtype, (const uint8_t *) W, rb, qs, ds, y, k, n, 1 << 20);
    return ls1_chk("gemv rows");
}

// LS1 PHASE7 item 3: quantise into an explicitly named slot, for the call sites
// that run several matvecs against ONE activation. gufo_fp4_gemv_fp4 still quantises for
// itself; this is the opt-in that lets a caller pay once.
extern "C" int gufo_fp4_act_prepare(const float * x, int k, int slot, hipStream_t stream) {
    return gufo_fp4_quant_q8(x, k, slot, stream);
}

extern "C" int gufo_fp4_gemv_fp4_q8(int qtype, const void * W, int slot,
                                 float * y, int k, int n, hipStream_t stream) {
    size_t rb = gufo_fp4_row_bytes(qtype, k);
    if (W == nullptr || y == nullptr || rb == 0 || n <= 0 || (k % 32) != 0 ||
        slot < 0 || slot >= GUFO_FP4_LS1_NSLOT) {
        return GUFO_FP4_ERR_ARG;
    }
    const int set = gufo_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // R2 refusal: the caller declines, never waits.
    int8_t * qs = nullptr;
    float * ds = nullptr;
    if (gufo_fp4_slots_get_q8(set, slot, k, &qs, &ds) != GUFO_FP4_OK) {
        return GUFO_FP4_ERR_ARG;
    }
    if (ls1_variant_gemv() && n <= GUFO_FP4_LS1_GEMV_WIDE_MAX) {
        gufo_fp4_k_gemv_wide_q8_rows<<<ls1_gemv_blocks(n, 8), 256, 0, stream>>>(
            qtype, (const uint8_t *) W, rb, qs, ds, y, k, n, 1 << 20);
        return ls1_chk("gemv rows sel");
    }
    if (n <= GUFO_FP4_LS1_GEMV_WIDE_MAX) {
        gufo_fp4_k_gemv_wide_q8_cta64<<<n, 64, 0, stream>>>(qtype, (const uint8_t *) W, rb, qs, ds, y, k, n);
    } else {
        const int blocks = n < GUFO_FP4_LS1_GEMV_GS_BLOCKS ? n : GUFO_FP4_LS1_GEMV_GS_BLOCKS;
        gufo_fp4_k_gemv_wide_q8_gs<<<blocks, 64, 0, stream>>>(qtype, (const uint8_t *) W, rb, qs, ds, y, k, n);
    }
    return ls1_chk("gemv q8");
}

// Matvec that quantises the activation itself. One extra tiny kernel per call;
// the caller gets the right answer with no cooperation.
extern "C" int gufo_fp4_gemv_fp4(int qtype, const void * W, const float * x,
                              float * y, int k, int n, hipStream_t stream) {
    const int rc = gufo_fp4_quant_q8(x, k, GUFO_FP4_LS1_SLOT_GENERIC, stream);
    if (rc != GUFO_FP4_OK) {
        return rc;
    }
    return gufo_fp4_gemv_fp4_q8(qtype, W, GUFO_FP4_LS1_SLOT_GENERIC, y, k, n, stream);
}

// Copy a slot's quantised activation back to the host. Test-only: it exists so
// tools/gufo_fp4_gemv_q8_test.cpp can tell a wrong quantisation from a wrong dot
// without having to guess which one it is looking at.
extern "C" int gufo_fp4_q8_peek_set(int set, int slot, int8_t * qs_out, float * ds_out, int k) {
    int8_t * qs = nullptr;
    float * ds = nullptr;
    {
        GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        if (set < 0 || (size_t) set >= mgr.sets.size()) return GUFO_FP4_ERR_ARG;
        GufoFp4SlotSet & S = mgr.sets[(size_t) set];
        if (slot < 0 || slot >= GUFO_FP4_LS1_NSLOT || k <= 0 || S.qs[slot] == nullptr ||
            S.cap[slot] < k) {
            return GUFO_FP4_ERR_ARG;
        }
        qs = S.qs[slot];
        ds = S.ds[slot];
    }
    hipError_t e = hipMemcpy(qs_out, qs, (size_t) k, hipMemcpyDeviceToHost);
    if (e != hipSuccess) return GUFO_FP4_ERR_HIP;
    e = hipMemcpy(ds_out, ds, (size_t) (k / 32) * sizeof(float), hipMemcpyDeviceToHost);
    return e == hipSuccess ? GUFO_FP4_OK : GUFO_FP4_ERR_HIP;
}

extern "C" int gufo_fp4_q8_peek(int slot, int8_t * qs_out, float * ds_out, int k) {
    return gufo_fp4_q8_peek_set(0, slot, qs_out, ds_out, k);
}

// ---------------------------------------------------------------------------
// Exhaustively check that the perm codebook above really is
// gufo_fp4_rocmfp4_code, for all 65536 four-nibble words. Returns the number of
// byte lanes that disagree; 0 means the table is a faithful transcription. This
// is the one piece of the integer path that cannot be checked by comparing
// against a float reference, because a wrong code permutes the products instead
// of scaling them, and a float reference would just look like quantisation noise.
// (It is also the check that caught a 2x error in an earlier revision of this
// file, which a float comparison would have read as ordinary quantisation noise.)
// ---------------------------------------------------------------------------
__global__ void gufo_fp4_k_cb_selftest(int * bad) {
    const int w = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= 65536) {
        return;
    }
    const int2 v = ls1_fp4_codebook16(w);
    const int8_t * o = (const int8_t *) &v;
    const uint8_t * qb = (const uint8_t *) &w;
    int local = 0;
    for (int i = 0; i < 4; ++i) {
        // lane i of .x is the LOW nibble of byte i, which is element i of the block;
        // lane i of .y is the HIGH nibble of byte i, which is element 16 + i.
        if ((int) o[i] != gufo_fp4_rocmfp4_code(qb[i] & 0x0fu)) local++;
        if ((int) o[4 + i] != gufo_fp4_rocmfp4_code(qb[i] >> 4)) local++;
    }
    if (local != 0) {
        atomicAdd(bad, local);
    }
}

extern "C" int gufo_fp4_gemv_q8_codebook_check(void) {
    int * d_bad = nullptr;
    if (hipMalloc(&d_bad, sizeof(int)) != hipSuccess) {
        return -1;
    }
    hipMemset(d_bad, 0, sizeof(int));
    gufo_fp4_k_cb_selftest<<<65536 / 256, 256, 0, (hipStream_t) nullptr>>>(d_bad);
    int bad = -1;
    if (hipDeviceSynchronize() != hipSuccess || hipMemcpy(&bad, d_bad, sizeof(int),
                                                         hipMemcpyDeviceToHost) != hipSuccess) {
        bad = -1;
    }
    hipFree(d_bad);
    return bad;
}
// ---------------------------------------------------------------------------
// Grouped, device-routed expert FFN (LS1 phases 2 and 3).
//
// The per-expert host loop this replaces did, per MoE layer, two blocking D2H copies of
// the router's 8 ids and 8 weights -- 40 GPU syncs per token -- and then roughly 56 kernel
// launches, ~2,240 of them per token across the 40 MoE layers, each a full round trip
// through the scheduler for about 1.3 MB of work.
//
// This is three launches per layer and zero syncs:
//
//   1. gufo_fp4_k_moe_gate_up_swiglu -- one CTA per (selected expert slot, output row). The CTA
//      reads its slot's id from DEVICE memory, dots the gate row and the up row against
//      the ONE quantisation of the FFN input, and fuses SwiGLU into its epilogue.
//   2. gufo_fp4_k_q8_1_quant over hs -- the down projection needs Q8_1, and a 32-value group's
//      scale spans 32 different CTAs of step 1, so it cannot be formed there.
//   3. gufo_fp4_k_moe_down_weighted -- one CTA per output row, looping the n_used slots IN SLOT
//      ORDER and accumulating in a register: the weighted sum is in a fixed order every
//      run, with no float atomics and no dependence on scheduling.
//
// One quantisation of the FFN input serves every expert and the weight bytes read are
// unchanged: 8 experts x (2 x 768 x 2560 + 2560 x 768) values per layer, the same 25 MB,
// now in three launches instead of ~56.
// ---------------------------------------------------------------------------
__global__ void gufo_fp4_k_moe_gate_up_swiglu(int qtype, const uint8_t * gate, const uint8_t * up,
                                     size_t rb_k, const int * ids,
                                     const int8_t * xq, const float * xd,
                                     float * hs, float limit,
                                     int n_ff, int n_expert, int k, int * rc) {
    // One flat grid over (slot, row) rather than a 2D one, so the slot stays in the same
    // expression as the row. The division is once per CTA on ~6k CTAs.
    const int r = (int) blockIdx.x;
    const int s = r / n_ff;                  // which selected expert
    const int j = r - s * n_ff;              // output row inside that expert
    __shared__ float red[32];

    // Every thread reads the same ids[s], so `ok` is uniform across the CTA and every
    // thread reaches both barriers inside ls1_block_sum. A bad id still dots a real (if
    // meaningless) row rather than returning, because a divergent return around a
    // __syncthreads is a hang, and the status is reported through *rc instead.
    const int id = ids[s];
    const bool ok = (id >= 0 && id < n_expert);
    if (!ok && threadIdx.x == 0) {
        *rc = GUFO_FP4_ERR_ARG;
    }
    const size_t base = ok ? (((size_t) id * (size_t) n_ff + (size_t) j) * rb_k) : 0;

    const float g = ls1_block_sum(ls1_dot_row_q8(qtype, gate + base, xq, xd, k), red);
    const float u = ls1_block_sum(ls1_dot_row_q8(qtype, up + base, xq, xd, k), red);
    if (threadIdx.x == 0) {
        hs[(size_t) s * (size_t) n_ff + (size_t) j] = gufo_fp4_swiglu_elem(g, u, limit);
    }
}

__global__ void gufo_fp4_k_moe_down_weighted(int qtype, const uint8_t * down, size_t rb_ff,
                                    const int * ids, const float * weights,
                                    const int8_t * hq, const float * hd,
                                    float * y, int n_used, int n_expert,
                                    int k, int n_ff, int * rc) {
    const int j = (int) blockIdx.x;          // output row of the down projection
    __shared__ float red[32];
    float acc = 0.f;
    // Fixed slot order: this is what makes the sum deterministic. Do NOT reorder this loop
    // and do NOT replace acc with an atomicAdd -- both silently make the result depend on
    // which CTA wins, which is the property this path exists to have.
    for (int s = 0; s < n_used; ++s) {
        const int id = ids[s];
        const bool ok = (id >= 0 && id < n_expert);
        if (!ok && threadIdx.x == 0) {
            *rc = GUFO_FP4_ERR_ARG;
        }
        const size_t base = ok ? (((size_t) id * (size_t) k + (size_t) j) * rb_ff) : 0;
        const float part = ls1_block_sum(ls1_dot_row_q8(qtype, down + base,
                                                         hq + (size_t) s * (size_t) n_ff,
                                                         hd + (size_t) s * (size_t) (n_ff / 32),
                                                         n_ff),
                                          red);
        if (threadIdx.x == 0) {
            acc += weights[s] * part;
        }
    }
    if (threadIdx.x == 0) {
        y[j] = acc;
    }
}

// SQ10: the hs scratch and its status word live in the slot sets above (one per
// stream). The router's own status word stays a single registered pointer: it
// is owned by gufo_fp4_route, which cannot free it, so that ONE read covers
// the whole decode token. See gufo_fp4_moe_dev_check.
static int * ls1_route_rc_d = nullptr;

// Device-routed grouped expert FFN for ONE token. `ids` and `weights` are DEVICE pointers
// (what gufo_fp4_route wrote); nothing is read back to the host and NOTHING SYNCS. The status
// word is left on the device for gufo_fp4_moe_dev_check() to read once per token -- reading it
// here would put back exactly the per-layer drain this path exists to remove.
// ROCmFP4 (100/101) and a single token; anything else returns GUFO_FP4_ERR_ARG so the caller
// can fall back to gufo_fp4_expert_ffn.
// QSM: quantize_x == false reuses the FFN_X slot's Q8 copy of x (see the _xq entry point).
static int expert_ffn_dev_impl(
        int qtype, const void * gate, const void * up, const void * down,
        const float * x, const int * ids_d, const float * weights_d,
        int n_used, int n_expert, int k, int n_ff, float limit, float * y,
        hipStream_t stream, bool quantize_x) {
    const size_t rb_k = gufo_fp4_row_bytes(qtype, k);
    const size_t rb_ff = gufo_fp4_row_bytes(qtype, n_ff);
    if (!gate || !up || !down || !x || !ids_d || !weights_d || !y) return GUFO_FP4_ERR_ARG;
    if (qtype != GUFO_FP4_Q4_0_ROCMFP4 && qtype != GUFO_FP4_Q4_0_ROCMFP4_FAST) return GUFO_FP4_ERR_ARG;
    if (rb_k == 0 || rb_ff == 0 || n_used <= 0 || n_expert <= 0 || k <= 0 || n_ff <= 0) {
        return GUFO_FP4_ERR_ARG;
    }
    if ((k % 32) != 0 || (n_ff % 32) != 0) return GUFO_FP4_ERR_ARG;

    const int set = gufo_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // R2 refusal: the caller declines, never waits.
    const size_t nn = (size_t) n_used * (size_t) n_ff;
    float * moe_hs = nullptr;
    int * moe_rc = nullptr;
    int rc = gufo_fp4_slots_grow_hs(set, nn, stream, &moe_hs, &moe_rc);
    if (rc != GUFO_FP4_OK) return rc;

    if (quantize_x) {
        rc = gufo_fp4_quant_q8(x, k, GUFO_FP4_LS1_SLOT_FFN_X, stream);
        if (rc != GUFO_FP4_OK) return rc;
    }
    int8_t * xq = nullptr;
    float * xd = nullptr;
    if (gufo_fp4_slots_get_q8(set, GUFO_FP4_LS1_SLOT_FFN_X, k, &xq, &xd) != GUFO_FP4_OK) {
        return GUFO_FP4_ERR_HIP;
    }

    gufo_fp4_k_moe_gate_up_swiglu<<<(unsigned) ((n_ff + 63) / 64) * 64 * (unsigned) n_used, 64, 0, stream>>>(
        qtype, (const uint8_t *) gate, (const uint8_t *) up, rb_k,
        ids_d, xq, xd, moe_hs, limit, n_ff, n_expert, k, moe_rc);
    if ((rc = ls1_chk("moe gate/up")) != GUFO_FP4_OK) return rc;

    rc = gufo_fp4_quant_q8(moe_hs, (int) nn, GUFO_FP4_LS1_SLOT_MOE_H, stream);
    if (rc != GUFO_FP4_OK) return rc;
    int8_t * hq = nullptr;
    float * hd = nullptr;
    if (gufo_fp4_slots_get_q8(set, GUFO_FP4_LS1_SLOT_MOE_H, (int) nn, &hq, &hd) != GUFO_FP4_OK) {
        return GUFO_FP4_ERR_HIP;
    }

    // LS1 PHASE7 item 4: the down half defaults ON. It is MEASURED bit-identical
    // (max_rel_diff 0, 0 differing of 2560, all three configurations), so it carries no
    // numerical risk, and it is 1.22x faster on the isolated A/B.
    // (An earlier revision of this file had lost the branch here entirely, which is why
    // the first phase-7 "variant ON" numbers turned out to have exercised ONLY the
    // plain-GEMV half.)
    if (ls1_variant_down()) {
        const int nwarps = 8;
        gufo_fp4_k_moe_down_weighted_rows<<<ls1_gemv_blocks(k, nwarps), nwarps * 32, 0, stream>>>(
            qtype, (const uint8_t *) down, rb_ff, ids_d, weights_d,
            hq, hd, y, n_used, n_expert, k, n_ff,
            1 << 20, moe_rc);
        return ls1_chk("moe down rows");
    }
    gufo_fp4_k_moe_down_weighted<<<(unsigned) k, 64, 0, stream>>>(qtype, (const uint8_t *) down, rb_ff,
                                              ids_d, weights_d,
                                              hq, hd,
                                              y, n_used, n_expert, k, n_ff, moe_rc);
    return ls1_chk("moe down");
}

extern "C" int gufo_fp4_expert_ffn_dev(
        int qtype, const void * gate, const void * up, const void * down,
        const float * x, const int * ids_d, const float * weights_d,
        int n_used, int n_expert, int k, int n_ff, float limit, float * y,
        hipStream_t stream) {
    return expert_ffn_dev_impl(qtype, gate, up, down, x, ids_d, weights_d, n_used, n_expert,
                               k, n_ff, limit, y, stream, true);
}

extern "C" int gufo_fp4_expert_ffn_dev_xq(
        int qtype, const void * gate, const void * up, const void * down,
        const float * x, const int * ids_d, const float * weights_d,
        int n_used, int n_expert, int k, int n_ff, float limit, float * y,
        hipStream_t stream) {
    return expert_ffn_dev_impl(qtype, gate, up, down, x, ids_d, weights_d, n_used, n_expert,
                               k, n_ff, limit, y, stream, false);
}

// Called by gufo_fp4_route once, when it allocates its status word, so that the per-token
// check below can read it without gufo_fp4_route reading it on every one of the 40 layers.
extern "C" void gufo_fp4_moe_dev_register_route_rc(int * rc_d) {
    ls1_route_rc_d = rc_d;
}

// Read the decode step's device status words -- the grouped FFN's and the router's -- and
// clear them. Call ONCE per token, after the last MoE layer, never per layer: this is the
// only blocking read in the decode path, and 40 of them is exactly what phase 3 removed.
//
// Clearing here rather than at the start of the token is deliberate: a status word that is
// only ever written on error would otherwise carry one token's failure into every token
// after it, and clearing costs nothing extra because the read is already blocking.
extern "C" int gufo_fp4_moe_dev_check_set(int set);
extern "C" int gufo_fp4_moe_dev_check_set_on_stream(int set, hipStream_t stream) {
    if (stream == nullptr) return GUFO_FP4_ERR_ARG;
    if (hipStreamSynchronize(stream) != hipSuccess) return GUFO_FP4_ERR_HIP;
    return gufo_fp4_moe_dev_check_set(set);
}

extern "C" int gufo_fp4_moe_dev_check_set(int set) {
    int * rc_d = nullptr;
    {
        GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        if (set < 0 || (size_t) set >= mgr.sets.size()) return GUFO_FP4_ERR_ARG;
        rc_d = mgr.sets[(size_t) set].moe_rc;
    }
    int rc = GUFO_FP4_OK;
    int * src[2] = {rc_d, ls1_route_rc_d};
    for (int i = 0; i < 2; ++i) {
        if (src[i] == nullptr) {
            continue;
        }
        int word = GUFO_FP4_OK;
        if (hipMemcpy(&word, src[i], sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
            return GUFO_FP4_ERR_HIP;
        }
        if (word != GUFO_FP4_OK && rc == GUFO_FP4_OK) {
            rc = word;
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (src[i] != nullptr) {
            if (hipMemset(src[i], 0, sizeof(int)) != hipSuccess) return GUFO_FP4_ERR_HIP;
        }
    }
    return rc;
}

extern "C" int gufo_fp4_moe_dev_check(void) {
    // Aggregate over every set, for callers that do not name one (Flash-Next,
    // prefill, tools). The qwen decode path names its set precisely.
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    int n = 0;
    {
        std::lock_guard<std::mutex> lk(mgr.mu);
        n = (int) mgr.sets.size();
    }
    int rc = GUFO_FP4_OK;
    for (int s = 0; s < n; ++s) {
        const int r = gufo_fp4_moe_dev_check_set(s);
        if (r != GUFO_FP4_OK && rc == GUFO_FP4_OK) {
            rc = r;
        }
    }
    return rc;
}

// Test-only: copy the grouped FFN's hs scratch back to the host.
extern "C" int gufo_fp4_moe_hs_peek_set(int set, float * out, int n) {
    float * hs = nullptr;
    {
        GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        if (set < 0 || (size_t) set >= mgr.sets.size()) return -1;
        hs = mgr.sets[(size_t) set].moe_hs;
    }
    if (hs == nullptr || n <= 0) return -1;
    return hipMemcpy(out, hs, (size_t) n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
}

extern "C" int gufo_fp4_moe_hs_peek(float * out, int n) {
    return gufo_fp4_moe_hs_peek_set(0, out, n);
}

// ---------------------------------------------------------------------------
// LS1 phase 4, item 3: the small-m decode forward, for MTP verification.
//
// A speculative verify runs m = 2..4 rows of the SAME weights against m different
// activations. The m = 1 path reads every weight byte once per row, so m rows cost m times
// the bandwidth. This form reads each weight row ONCE and spends the extra work on m sets
// of dot products: the four v_perm of the codebook and the four 32-bit loads of the block
// are done once per 32 weights and reused for every column. That is llama.cpp's
// `mul_mat_vec_q` with ncols_dst <= 8, which is the model the fork uses too.
//
//   gufo_fp4_k_gemv_wide_q8_m  -- n CTAs, one per output row, m columns each, y is [m][n] token-major
//   gufo_fp4_k_moe_gu_m        -- the grouped expert FFN's gate+up+SwiGLU over (token x slot) pairs
//   gufo_fp4_k_moe_down_m      -- its down + weighted sum, m accumulators in one register set
//
// Both grouped kernels take the per-pair expert id from DEVICE memory and never sync, like
// their m = 1 counterparts. Nothing here is wired into Step; LS4 calls it.
enum { GUFO_FP4_LS1_MAX_M = 4 };

__global__ void gufo_fp4_k_gemv_wide_q8_m(int qtype, const uint8_t * W, size_t row_bytes,
                                 const int8_t * xq, const float * xd,
                                 float * y, int k, int n, int m) {
    const int row = (int) blockIdx.x;
    if (row >= n) {
        return;
    }
    const uint8_t * w = W + (size_t) row * row_bytes;
    const int tsz = gufo_fp4_type_size(qtype);
    const int nblk = k / 32;
    const int nb32 = k / 32;
    // One 32-float reduction slot per column: ls1_block_sum's trailing barrier makes red
    // reusable, so the columns are reduced one after another without extra sync.
    __shared__ float red[GUFO_FP4_LS1_MAX_M * 32];

    float s[GUFO_FP4_LS1_MAX_M];
    for (int t = 0; t < GUFO_FP4_LS1_MAX_M; ++t) {
        s[t] = 0.f;
    }
    for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
        // Decode the block ONCE; the m columns share it.
        const ls1_blk32 c = ls1_load_blk32(qtype, w + (size_t) qb * (size_t) tsz);
        for (int t = 0; t < m; ++t) {
            s[t] += ls1_dot_blk32(c, xq + (size_t) t * (size_t) k + (size_t) qb * 32,
                                  xd[(size_t) t * (size_t) nb32 + (size_t) qb]);
        }
    }
    for (int t = 0; t < m; ++t) {
        const float r = ls1_block_sum(s[t], red + t * 32);
        if (threadIdx.x == 0) {
            y[(size_t) t * (size_t) n + (size_t) row] = r;
        }
    }
}

// gate + up + SwiGLU for every (token, slot) pair. ids_d and weights_d are indexed by the
// flat slot s = token * n_used + slot, and are DEVICE pointers.
__global__ void gufo_fp4_k_moe_gu_m(int qtype, const uint8_t * gate, const uint8_t * up, size_t rb_k,
                           const int8_t * xq, const float * xd,
                           const int * ids_d, float * hs, float limit,
                           int n_used, int n_expert, int k, int n_ff, int n_slots,
                           int * rc) {
    const int r = (int) blockIdx.x;
    const int s = r / n_ff;                  // flat (token, slot)
    const int j = r - s * n_ff;              // output row inside that expert
    const int t = s / n_used;                // which token
    __shared__ float red[GUFO_FP4_LS1_MAX_M * 32];

    const int id = ids_d[s];
    const bool ok = (id >= 0 && id < n_expert);
    if (!ok && threadIdx.x == 0) {
        *rc = GUFO_FP4_ERR_ARG;
    }
    const size_t base = ok ? (((size_t) id * (size_t) n_ff + (size_t) j) * rb_k) : 0;
    const int8_t * xqt = xq + (size_t) t * (size_t) k;
    const float * xdt = xd + (size_t) t * (size_t) (k / 32);

    const float g = ls1_block_sum(ls1_dot_row_q8(qtype, gate + base, xqt, xdt, k), red);
    const float u = ls1_block_sum(ls1_dot_row_q8(qtype, up + base, xqt, xdt, k), red);
    if (threadIdx.x == 0) {
        hs[(size_t) s * (size_t) n_ff + (size_t) j] = gufo_fp4_swiglu_elem(g, u, limit);
    }
}

// down + the weighted sum over slots, m accumulators in one register set. The slot loop is
// in flat (token, slot) order and the tokens' accumulators are independent, so the order
// per token is fixed: the same determinism the m = 1 kernel has. The weight row is read
// ONCE per slot and the m tokens' hs rows each get their own dot, which is the same
// sharing the m-column matvec is built on.
__global__ void gufo_fp4_k_moe_down_m(int qtype, const uint8_t * down, size_t rb_ff,
                             const int * ids_d, const float * weights_d,
                             const int8_t * hq, const float * hd,
                             float * y, int n_used, int n_expert, int k, int n_ff,
                             int n_slots, int m, int * rc) {
    const int j = (int) blockIdx.x;          // output row of the down projection
    __shared__ float red[GUFO_FP4_LS1_MAX_M * 32];
    float acc[GUFO_FP4_LS1_MAX_M];
    for (int t = 0; t < GUFO_FP4_LS1_MAX_M; ++t) {
        acc[t] = 0.f;
    }
    for (int s = 0; s < n_slots; ++s) {
        // hs[s] belongs to ONE token, s / n_used, so its dot goes into that token's
        // accumulator and no other. Adding it to all m of them was the first version of
        // this kernel and every element came out wrong.
        const int t = s / n_used;
        const int id = ids_d[s];
        const bool ok = (id >= 0 && id < n_expert);
        if (!ok && threadIdx.x == 0) {
            *rc = GUFO_FP4_ERR_ARG;
        }
        const size_t base = ok ? (((size_t) id * (size_t) k + (size_t) j) * rb_ff) : 0;
        const float part = ls1_block_sum(
            ls1_dot_row_q8(qtype, down + base, hq + (size_t) s * (size_t) n_ff,
                           hd + (size_t) s * (size_t) (n_ff / 32), n_ff), red);
        if (threadIdx.x == 0) {
            acc[t] += weights_d[s] * part;
        }
    }
    for (int t = 0; t < m; ++t) {
        if (threadIdx.x == 0) {
            y[(size_t) t * (size_t) k + (size_t) j] = acc[t];
        }
    }
}

// LV1 R3: one expert, m rows. Each gate/up/down row is decoded once and dotted
// against every token. The per-thread block order, ls1_block_sum, SwiGLU and
// `acc += weight * part` match gufo_fp4_k_moe_gu_m / gufo_fp4_k_moe_down_m at n_used = 1.
__global__ void gufo_fp4_k_moe_gu_one_m(int qtype, const uint8_t * gate, const uint8_t * up,
                               size_t rb_k, const int8_t * xq, const float * xd,
                               const int * ids_d, float * hs, float limit,
                               int n_expert, int k, int n_ff, int m, int * rc) {
    const int j = (int) blockIdx.x;
    __shared__ float red[32];
    const int id = ids_d[0];
    bool ok = (id >= 0 && id < n_expert);
    for (int t = 1; t < m; ++t) {
        if (ids_d[t] != id) ok = false;
    }
    if (!ok && threadIdx.x == 0) {
        *rc = GUFO_FP4_ERR_ARG;
    }
    const size_t base = ok ? (((size_t) id * (size_t) n_ff + (size_t) j) * rb_k) : 0;
    const int tsz = gufo_fp4_type_size(qtype);
    const int nblk = k / 32;
    const int nb32 = k / 32;
    const uint8_t * grow = gate + base;
    const uint8_t * urow = up + base;
    float sg[GUFO_FP4_LS1_MAX_M], su[GUFO_FP4_LS1_MAX_M];
    for (int t = 0; t < GUFO_FP4_LS1_MAX_M; ++t) {
        sg[t] = 0.f;
        su[t] = 0.f;
    }
    for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
        const ls1_blk32 cg = ls1_load_blk32(qtype, grow + (size_t) qb * (size_t) tsz);
        const ls1_blk32 cu = ls1_load_blk32(qtype, urow + (size_t) qb * (size_t) tsz);
        for (int t = 0; t < m; ++t) {
            const int8_t * xqt = xq + (size_t) t * (size_t) k + (size_t) qb * 32;
            const float xdt = xd[(size_t) t * (size_t) nb32 + (size_t) qb];
            sg[t] += ls1_dot_blk32(cg, xqt, xdt);
            su[t] += ls1_dot_blk32(cu, xqt, xdt);
        }
    }
    for (int t = 0; t < m; ++t) {
        const float g = ls1_block_sum(sg[t], red);
        const float u = ls1_block_sum(su[t], red);
        if (threadIdx.x == 0) {
            hs[(size_t) t * (size_t) n_ff + (size_t) j] = gufo_fp4_swiglu_elem(g, u, limit);
        }
    }
}

__global__ void gufo_fp4_k_moe_down_one_m(int qtype, const uint8_t * down, size_t rb_ff,
                                 const int * ids_d, const float * weights_d,
                                 const int8_t * hq, const float * hd, float * y,
                                 int n_expert, int k, int n_ff, int m, int * rc) {
    const int j = (int) blockIdx.x;
    __shared__ float red[32];
    const int id = ids_d[0];
    bool ok = (id >= 0 && id < n_expert);
    for (int t = 1; t < m; ++t) {
        if (ids_d[t] != id) ok = false;
    }
    if (!ok && threadIdx.x == 0) {
        *rc = GUFO_FP4_ERR_ARG;
    }
    const size_t base = ok ? (((size_t) id * (size_t) k + (size_t) j) * rb_ff) : 0;
    const int tsz = gufo_fp4_type_size(qtype);
    const int nblk = n_ff / 32;
    const uint8_t * drow = down + base;
    float part[GUFO_FP4_LS1_MAX_M];
    float acc[GUFO_FP4_LS1_MAX_M];
    for (int t = 0; t < GUFO_FP4_LS1_MAX_M; ++t) {
        part[t] = 0.f;
        acc[t] = 0.f;
    }
    for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
        const ls1_blk32 c = ls1_load_blk32(qtype, drow + (size_t) qb * (size_t) tsz);
        for (int t = 0; t < m; ++t) {
            part[t] += ls1_dot_blk32(
                c, hq + (size_t) t * (size_t) n_ff + (size_t) qb * 32,
                hd[(size_t) t * (size_t) (n_ff / 32) + (size_t) qb]);
        }
    }
    for (int t = 0; t < m; ++t) {
        // n_used == 1, so slot s is the token and this is the only term.
        const float p = ls1_block_sum(part[t], red);
        if (threadIdx.x == 0) {
            acc[t] += weights_d[t] * p;
        }
    }
    if (threadIdx.x == 0) {
        for (int t = 0; t < m; ++t) {
            y[(size_t) t * (size_t) k + (size_t) j] = acc[t];
        }
    }
}

// SQ10: the small-m pair and hs live in the slot sets (one per stream).
// Quantise n_tokens * k floats (token-major) into the calling stream's set,
// handing the pair back.
static int ls1_quant_m(const float * x, int total, hipStream_t stream,
                       int8_t ** qs, float ** ds) {
    if (x == nullptr || total <= 0 || (total % 32) != 0) return GUFO_FP4_ERR_ARG;
    const int set = gufo_fp4_slots_for_stream_at(stream, __func__, (long) total, false);
    if (set < 0) return set;  // R2 refusal: the caller declines, never waits.
    const int rc = gufo_fp4_slots_grow_q8m(set, total, stream, qs, ds);
    if (rc != GUFO_FP4_OK) return rc;
    const int nb = total / 32;
    const int threads = 256;
    gufo_fp4_k_q8_1_quant<<<(nb + threads - 1) / threads, threads, 0, stream>>>(x, *qs, *ds, nb);
    return ls1_chk("q8 quant m");
}

// Matvec of n rows against m activation columns, activations already in Q8_1.
// y is [m][n] token-major: y[t * n + row].
extern "C" int gufo_fp4_gemv_q8_m(int qtype, const void * W, const int8_t * xq, const float * xd,
                               float * y, int k, int n, int m, hipStream_t stream) {
    size_t rb = gufo_fp4_row_bytes(qtype, k);
    if (!W || !xq || !xd || !y || rb == 0 || n <= 0) return GUFO_FP4_ERR_ARG;
    if (m < 1 || m > GUFO_FP4_LS1_MAX_M || (k % 32) != 0) return GUFO_FP4_ERR_ARG;
    if (qtype != GUFO_FP4_Q4_0_ROCMFP4 && qtype != GUFO_FP4_Q4_0_ROCMFP4_FAST) return GUFO_FP4_ERR_ARG;
    gufo_fp4_k_gemv_wide_q8_m<<<(unsigned) n, 64, 0, stream>>>(qtype, (const uint8_t *) W, rb, xq, xd, y, k, n, m);
    return ls1_chk("gemv q8 m");
}

// LM1 SPEC4 item 2: quantise-only entry. See the header for the sharing rule.
extern "C" int gufo_fp4_quant_m(const float * x, int n_tokens, int k, hipStream_t stream,
                             const int8_t ** xq, const float ** xd) {
    if (xq == nullptr || xd == nullptr) return GUFO_FP4_ERR_ARG;
    if (n_tokens < 1 || n_tokens > GUFO_FP4_LS1_MAX_M || k <= 0 || (k % 32) != 0) return GUFO_FP4_ERR_ARG;
    int8_t * qs = nullptr;
    float * ds = nullptr;
    const int rc = ls1_quant_m(x, n_tokens * k, stream, &qs, &ds);
    if (rc != GUFO_FP4_OK) return rc;
    *xq = qs;
    *xd = ds;
    return GUFO_FP4_OK;
}

// Same, but quantising m * k floats from float first.
extern "C" int gufo_fp4_gemv_m(int qtype, const void * W, const float * x, float * y,
                            int k, int n, int m, hipStream_t stream) {
    int8_t * qs = nullptr;
    float * ds = nullptr;
    const int rc = ls1_quant_m(x, m * k, stream, &qs, &ds);
    if (rc != GUFO_FP4_OK) return rc;
    return gufo_fp4_gemv_q8_m(qtype, W, qs, ds, y, k, n, m, stream);
}

// Grouped expert FFN over n_tokens * n_used (token, slot) pairs, for one token batch.
// ids_d / weights_d are DEVICE pointers, flat and indexed s = token * n_used + slot.
// y is [n_tokens][k] token-major. No sync, no host round trip.
extern "C" int gufo_fp4_expert_ffn_m(int qtype, const void * gate, const void * up,
                                 const void * down, const float * x,
                                 const int * ids_d, const float * weights_d,
                                 int n_tokens, int n_used, int n_expert,
                                 int k, int n_ff, float limit, float * y,
                                 hipStream_t stream) {
    const size_t rb_k = gufo_fp4_row_bytes(qtype, k);
    const size_t rb_ff = gufo_fp4_row_bytes(qtype, n_ff);
    if (!gate || !up || !down || !x || !ids_d || !weights_d || !y) return GUFO_FP4_ERR_ARG;
    if (qtype != GUFO_FP4_Q4_0_ROCMFP4 && qtype != GUFO_FP4_Q4_0_ROCMFP4_FAST) return GUFO_FP4_ERR_ARG;
    if (rb_k == 0 || rb_ff == 0) return GUFO_FP4_ERR_ARG;
    if (n_tokens < 1 || n_tokens > GUFO_FP4_LS1_MAX_M || n_used <= 0 || n_expert <= 0) return GUFO_FP4_ERR_ARG;
    if (k <= 0 || n_ff <= 0 || (k % 32) != 0 || (n_ff % 32) != 0) return GUFO_FP4_ERR_ARG;

    const int set = gufo_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // R2 refusal: the caller declines, never waits.
    const int n_slots = n_tokens * n_used;
    const size_t nn = (size_t) n_slots * (size_t) n_ff;
    float * moe_hs_m = nullptr;
    int * moe_rc_m = nullptr;
    int rc = gufo_fp4_slots_grow_hsm(set, nn, stream, &moe_hs_m, &moe_rc_m);
    if (rc != GUFO_FP4_OK) return rc;

    int8_t * qs = nullptr;
    float * ds = nullptr;
    rc = ls1_quant_m(x, n_tokens * k, stream, &qs, &ds);
    if (rc != GUFO_FP4_OK) return rc;

    gufo_fp4_k_moe_gu_m<<<(unsigned) (n_ff * n_slots), 64, 0, stream>>>(qtype, (const uint8_t *) gate,
                                                     (const uint8_t *) up, rb_k,
                                                     qs, ds, ids_d,
                                                     moe_hs_m, limit,
                                                     n_used, n_expert, k, n_ff, n_slots,
                                                     moe_rc_m);
    if ((rc = ls1_chk("moe m gate/up")) != GUFO_FP4_OK) return rc;

    int8_t * hq = nullptr;
    float * hd = nullptr;
    rc = ls1_quant_m(moe_hs_m, (int) nn, stream, &hq, &hd);
    if (rc != GUFO_FP4_OK) return rc;

    gufo_fp4_k_moe_down_m<<<(unsigned) k, 64, 0, stream>>>(qtype, (const uint8_t *) down, rb_ff,
                                       ids_d, weights_d, hq, hd, y,
                                       n_used, n_expert, k, n_ff, n_slots, n_tokens,
                                       moe_rc_m);
    return ls1_chk("moe m down");
}

// LV1 R3. n_tokens == 1 and n_used != 1 stay on gufo_fp4_expert_ffn_m, so Step's
// single-token expert and the routed verify (n_used = 8) do not change.
extern "C" int gufo_fp4_expert_ffn_shared_m(int qtype, const void * gate, const void * up,
                                        const void * down, const float * x,
                                        const int * ids_d, const float * weights_d,
                                        int n_tokens, int n_used, int n_expert,
                                        int k, int n_ff, float limit, float * y,
                                        hipStream_t stream) {
    if (n_used != 1 || n_tokens <= 1) {
        return gufo_fp4_expert_ffn_m(qtype, gate, up, down, x, ids_d, weights_d,
                                 n_tokens, n_used, n_expert, k, n_ff, limit, y, stream);
    }
    const size_t rb_k = gufo_fp4_row_bytes(qtype, k);
    const size_t rb_ff = gufo_fp4_row_bytes(qtype, n_ff);
    if (!gate || !up || !down || !x || !ids_d || !weights_d || !y) return GUFO_FP4_ERR_ARG;
    if (qtype != GUFO_FP4_Q4_0_ROCMFP4 && qtype != GUFO_FP4_Q4_0_ROCMFP4_FAST) return GUFO_FP4_ERR_ARG;
    if (rb_k == 0 || rb_ff == 0) return GUFO_FP4_ERR_ARG;
    if (n_tokens > GUFO_FP4_LS1_MAX_M || n_expert <= 0) return GUFO_FP4_ERR_ARG;
    if (k <= 0 || n_ff <= 0 || (k % 32) != 0 || (n_ff % 32) != 0) return GUFO_FP4_ERR_ARG;

    const int set = gufo_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // R2 refusal: the caller declines, never waits.
    const size_t nn = (size_t) n_tokens * (size_t) n_ff;
    float * moe_hs_m = nullptr;
    int * moe_rc_m = nullptr;
    int rc = gufo_fp4_slots_grow_hsm(set, nn, stream, &moe_hs_m, &moe_rc_m);
    if (rc != GUFO_FP4_OK) return rc;

    int8_t * qs = nullptr;
    float * ds = nullptr;
    rc = ls1_quant_m(x, n_tokens * k, stream, &qs, &ds);
    if (rc != GUFO_FP4_OK) return rc;

    gufo_fp4_k_moe_gu_one_m<<<(unsigned) n_ff, 64, 0, stream>>>(
        qtype, (const uint8_t *) gate, (const uint8_t *) up, rb_k,
        qs, ds, ids_d, moe_hs_m, limit, n_expert, k, n_ff, n_tokens,
        moe_rc_m);
    if ((rc = ls1_chk("shared m gate/up")) != GUFO_FP4_OK) return rc;

    int8_t * hq = nullptr;
    float * hd = nullptr;
    rc = ls1_quant_m(moe_hs_m, (int) nn, stream, &hq, &hd);
    if (rc != GUFO_FP4_OK) return rc;

    gufo_fp4_k_moe_down_one_m<<<(unsigned) k, 64, 0, stream>>>(
        qtype, (const uint8_t *) down, rb_ff, ids_d, weights_d, hq, hd, y,
        n_expert, k, n_ff, n_tokens, moe_rc_m);
    return ls1_chk("shared m down");
}

// Read the small-m path's status word and clear it. Blocking; one call per verify batch.
// Variant 1 of the grouped down projection. hs is already quantised into the MoE slot.
// The FULL grouped expert FFN with the warp-per-row down kernel, so the A/B in
// the test compares like with like: the previous version of that benchmark timed
// gufo_fp4_expert_ffn_dev (gate + up + quantise + down) against the down kernel
// alone and reported 0.26x, which is a comparison of three stages with one.
extern "C" int gufo_fp4_expert_ffn_dev_tuned(int qtype, const void * gate, const void * up,
                                         const void * down, const float * x,
                                         const int * ids_d, const float * weights_d,
                                         int n_used, int n_expert, int k, int n_ff,
                                         float limit, float * y, int block,
                                         hipStream_t stream) {
    const size_t rb_k = gufo_fp4_row_bytes(qtype, k);
    const size_t rb_ff = gufo_fp4_row_bytes(qtype, n_ff);
    if (!gate || !up || !down || !x || !ids_d || !weights_d || !y) return GUFO_FP4_ERR_ARG;
    if (qtype != GUFO_FP4_Q4_0_ROCMFP4 && qtype != GUFO_FP4_Q4_0_ROCMFP4_FAST) return GUFO_FP4_ERR_ARG;
    if (rb_k == 0 || rb_ff == 0) return GUFO_FP4_ERR_ARG;
    if (n_used <= 0 || n_expert <= 0 || k <= 0 || n_ff <= 0) return GUFO_FP4_ERR_ARG;
    if ((k % 32) != 0 || (n_ff % 32) != 0 || block < 32) return GUFO_FP4_ERR_ARG;

    const int set = gufo_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // R2 refusal: the caller declines, never waits.
    const size_t nn = (size_t) n_used * (size_t) n_ff;
    float * moe_hs = nullptr;
    int * moe_rc = nullptr;
    int rc = gufo_fp4_slots_grow_hs(set, nn, stream, &moe_hs, &moe_rc);
    if (rc != GUFO_FP4_OK) return rc;
    rc = gufo_fp4_quant_q8(x, k, GUFO_FP4_LS1_SLOT_FFN_X, stream);
    if (rc != GUFO_FP4_OK) return rc;
    int8_t * xq = nullptr;
    float * xd = nullptr;
    if (gufo_fp4_slots_get_q8(set, GUFO_FP4_LS1_SLOT_FFN_X, k, &xq, &xd) != GUFO_FP4_OK) {
        return GUFO_FP4_ERR_HIP;
    }

    gufo_fp4_k_moe_gate_up_swiglu<<<(unsigned) (((n_ff + 63) / 64) * 64 * n_used), 64, 0, stream>>>(
        qtype, (const uint8_t *) gate, (const uint8_t *) up, rb_k,
        ids_d, xq, xd, moe_hs, limit, n_ff, n_expert, k, moe_rc);
    if ((rc = ls1_chk("moe gate/up")) != GUFO_FP4_OK) return rc;
    rc = gufo_fp4_quant_q8(moe_hs, (int) nn, GUFO_FP4_LS1_SLOT_MOE_H, stream);
    if (rc != GUFO_FP4_OK) return rc;
    int8_t * hq = nullptr;
    float * hd = nullptr;
    if (gufo_fp4_slots_get_q8(set, GUFO_FP4_LS1_SLOT_MOE_H, (int) nn, &hq, &hd) != GUFO_FP4_OK) {
        return GUFO_FP4_ERR_HIP;
    }
    const int nwarps = block >> 5;
    gufo_fp4_k_moe_down_weighted_rows<<<ls1_gemv_blocks(k, nwarps), block, 0, stream>>>(
        qtype, (const uint8_t *) down, rb_ff, ids_d, weights_d,
        hq, hd, y, n_used, n_expert, k, n_ff,
        1 << 20, moe_rc);
    return ls1_chk("moe down rows tuned");
}

extern "C" int gufo_fp4_expert_ffn_dev_rows(int qtype, const void * down,
                                         const int * ids_d, const float * weights_d,
                                         int n_used, int n_expert, int k, int n_ff,
                                         float * y, int block, hipStream_t stream) {
    const size_t rb_ff = gufo_fp4_row_bytes(qtype, n_ff);
    if (!down || !ids_d || !weights_d || !y || rb_ff == 0 || block < 32) return GUFO_FP4_ERR_ARG;
    const int set = gufo_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // R2 refusal: the caller declines, never waits.
    int8_t * hq = nullptr;
    float * hd = nullptr;
    int * moe_rc = nullptr;
    {
        GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        GufoFp4SlotSet & S = mgr.sets[(size_t) set];
        if (S.qs[GUFO_FP4_LS1_SLOT_MOE_H] == nullptr || S.moe_rc == nullptr) {
            return GUFO_FP4_ERR_ARG;
        }
        hq = S.qs[GUFO_FP4_LS1_SLOT_MOE_H];
        hd = S.ds[GUFO_FP4_LS1_SLOT_MOE_H];
        moe_rc = S.moe_rc;
    }
    const int nwarps = block >> 5;
    gufo_fp4_k_moe_down_weighted_rows<<<ls1_gemv_blocks(k, nwarps), block, 0, stream>>>(
        qtype, (const uint8_t *) down, rb_ff, ids_d, weights_d,
        hq, hd, y, n_used, n_expert, k, n_ff,
        1 << 20, moe_rc);
    return ls1_chk("moe down rows");
}

extern "C" int gufo_fp4_expert_ffn_m_check_set(int set);
extern "C" int gufo_fp4_expert_ffn_m_check_set_on_stream(int set, hipStream_t stream) {
    if (stream == nullptr) return GUFO_FP4_ERR_ARG;
    if (hipStreamSynchronize(stream) != hipSuccess) return GUFO_FP4_ERR_HIP;
    return gufo_fp4_expert_ffn_m_check_set(set);
}

extern "C" int gufo_fp4_expert_ffn_m_check_set(int set) {
    int * rc_d = nullptr;
    {
        GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        if (set < 0 || (size_t) set >= mgr.sets.size()) return GUFO_FP4_ERR_ARG;
        rc_d = mgr.sets[(size_t) set].moe_rc_m;
    }
    if (rc_d == nullptr) return GUFO_FP4_OK;
    int rc = GUFO_FP4_OK;
    if (hipMemcpy(&rc, rc_d, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
        return GUFO_FP4_ERR_HIP;
    }
    if (hipMemset(rc_d, 0, sizeof(int)) != hipSuccess) return GUFO_FP4_ERR_HIP;
    return rc;
}

extern "C" int gufo_fp4_expert_ffn_m_check(void) {
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    int n = 0;
    {
        std::lock_guard<std::mutex> lk(mgr.mu);
        n = (int) mgr.sets.size();
    }
    int rc = GUFO_FP4_OK;
    for (int s = 0; s < n; ++s) {
        const int r = gufo_fp4_expert_ffn_m_check_set(s);
        if (r != GUFO_FP4_OK && rc == GUFO_FP4_OK) {
            rc = r;
        }
    }
    return rc;
}

// Test-only: the small-m path's hs scratch.
extern "C" int gufo_fp4_moe_hs_m_peek_set(int set, float * out, int n) {
    float * hs = nullptr;
    {
        GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        if (set < 0 || (size_t) set >= mgr.sets.size()) return -1;
        hs = mgr.sets[(size_t) set].moe_hs_m;
    }
    if (hs == nullptr || n <= 0) return -1;
    return hipMemcpy(out, hs, (size_t) n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
}

extern "C" int gufo_fp4_moe_hs_m_peek(float * out, int n) {
    return gufo_fp4_moe_hs_m_peek_set(0, out, n);
}

// lqw:spec6. Small-m verify GEMV. m = 2..8 columns, one load of each weight
// block, then the m=1 dot (ls1_dot_fp4_blk32_q8) and the same 64-thread
// ls1_block_sum. Launch geometry matches gufo_fp4_gemv_fp4_q8: one CTA per row
// at n <= 8192, otherwise the 2048-wide grid stride. Does not allocate.
// SQ10: the verify pair lives in the slot sets (one per stream).
extern "C" int gufo_fp4_verify_m_reserve(int total) {
    if (total <= 0 || (total % 32) != 0) return GUFO_FP4_ERR_ARG;
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    bool grew = false;
    for (size_t i = 0; i < mgr.sets.size(); ++i) {
        GufoFp4SlotSet & S = mgr.sets[i];
        if (S.qv_cap >= total && S.qv != nullptr && S.dv != nullptr) continue;
        grew = true;
    }
    if (!grew) return GUFO_FP4_OK;
    // Reserve takes no stream; drain the device before freeing, so no queued
    // kernel still reads a pair we replace. Load-time only in serve.
    if (hipDeviceSynchronize() != hipSuccess) return GUFO_FP4_ERR_HIP;
    for (size_t i = 0; i < mgr.sets.size(); ++i) {
        GufoFp4SlotSet & S = mgr.sets[i];
        if (S.qv_cap >= total && S.qv != nullptr && S.dv != nullptr) continue;
        int8_t * qs = nullptr;
        float * ds = nullptr;
        if (gufo_fp4_slots_malloc((void **) &qs, (size_t) total + GUFO_FP4_TAIL_PAD) != GUFO_FP4_OK) {
            return GUFO_FP4_ERR_HIP;
        }
        if (gufo_fp4_slots_malloc((void **) &ds,
                                  (size_t) (total / 32) * sizeof(float) + GUFO_FP4_TAIL_PAD) != GUFO_FP4_OK) {
            (void) hipFree(qs);
            return GUFO_FP4_ERR_HIP;
        }
        if (S.qv) {
            if (hipFree(S.qv) != hipSuccess) return GUFO_FP4_ERR_HIP;
        }
        if (S.dv) {
            if (hipFree(S.dv) != hipSuccess) return GUFO_FP4_ERR_HIP;
        }
        S.qv = qs;
        S.dv = ds;
        S.qv_cap = total;
        S.vq_x = nullptr;
        S.vq_m = 0;
        S.vq_k = 0;
    }
    return GUFO_FP4_OK;
}

// lqw:spec7. One K-block dotted against M columns for one or two rows.
// Same arithmetic as ls1_dot_fp4_blk32_q8 per (row, column): same decode,
// same sudot4 order, same float combine. has1=false skips the second row
// (the odd-n tail pair); the guard is CTA-uniform at every call site.
template <int M>
__device__ __forceinline__ void ls1_verify_dot2(int qtype, const uint8_t * w0, const uint8_t * w1,
                                                bool has1, const int8_t * xq, const float * xd,
                                                int k, int nblk, int tsz, int qb,
                                                float * s0, float * s1) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    {
        const ls1_blk32 c = ls1_load_blk32(qtype, w0 + (size_t) qb * (size_t) tsz);
        const size_t xo = (size_t) qb * 32;
#pragma unroll
        for (int t = 0; t < M; ++t) {
            s0[t] += ls1_dot_blk32(c, xq + (size_t) t * (size_t) k + xo,
                                   xd[(size_t) t * (size_t) nblk + (size_t) qb]);
        }
    }
    if (has1) {
        const ls1_blk32 c = ls1_load_blk32(qtype, w1 + (size_t) qb * (size_t) tsz);
        const size_t xo = (size_t) qb * 32;
#pragma unroll
        for (int t = 0; t < M; ++t) {
            s1[t] += ls1_dot_blk32(c, xq + (size_t) t * (size_t) k + xo,
                                   xd[(size_t) t * (size_t) nblk + (size_t) qb]);
        }
    }
}

// lqw:spec7. Two rows per CTA, two K-blocks per qb step (replaces the SPEC6 one-row form). Each row keeps
// the m = 1 arithmetic exactly: ascending qb order, ls1_dot_blk32 per block,
// the same 64-thread ls1_block_sum tree per column. Only geometry changes:
// the CTA walks two rows' blocks together (4x the independent work per step
// with the unroll), halving CTA count. Odd n ends in a predicated tail pair.
template <int M>
__global__ void gufo_fp4_k_gemv_verify_mt(int qtype, const uint8_t * W, size_t row_bytes,
                                          const int8_t * xq, const float * xd,
                                          float * y, int k, int n) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int tsz = gufo_fp4_type_size(qtype);
    const int nblk = k >> 5;
    const int bdim = (int) blockDim.x;
    const int tid = (int) threadIdx.x;
    const int npair = (n + 1) >> 1;
    __shared__ float red[32];
    float s0[M];
    float s1[M];
    for (int pair = (int) blockIdx.x; pair < npair; pair += (int) gridDim.x) {
        const int row0 = pair << 1;
        const int row1 = row0 + 1;
        const bool has1 = row1 < n;
        const uint8_t * w0 = W + (size_t) row0 * row_bytes;
        const uint8_t * w1 = W + (size_t) (has1 ? row1 : row0) * row_bytes;
#pragma unroll
        for (int t = 0; t < M; ++t) {
            s0[t] = 0.f;
            s1[t] = 0.f;
        }
        int qb = tid;
        for (; qb + bdim < nblk; qb += bdim << 1) {
            ls1_verify_dot2<M>(qtype, w0, w1, has1, xq, xd, k, nblk, tsz, qb, s0, s1);
            ls1_verify_dot2<M>(qtype, w0, w1, has1, xq, xd, k, nblk, tsz, qb + bdim, s0, s1);
        }
        // Loop exit guarantees qb + bdim >= nblk, so at most this one block
        // is left for the thread; qb order stays ascending.
        if (qb < nblk) {
            ls1_verify_dot2<M>(qtype, w0, w1, has1, xq, xd, k, nblk, tsz, qb, s0, s1);
        }
#pragma unroll
        for (int t = 0; t < M; ++t) {
            const float r0 = ls1_block_sum(s0[t], red);
            if (tid == 0) {
                y[(size_t) t * (size_t) n + (size_t) row0] = r0;
            }
            // Uniform: has1 depends only on pair and n.
            if (has1) {
                const float r1 = ls1_block_sum(s1[t], red);
                if (tid == 0) {
                    y[(size_t) t * (size_t) n + (size_t) row1] = r1;
                }
            }
        }
    }
}

// Super ROC QVO: quantize_x == false reuses the set's Q8 copy of x (the _xq
// entry point); it refuses unless that copy is of this x, m and k.
static int gemv_verify_m_impl(int qtype, const void * W, const float * x, float * y,
                              int k, int n, int m, hipStream_t stream, bool quantize_x) {
    // sq16 SPEC2 (S2-1 test hook): GUFO_FP4_FORCE_REFUSE=1 declines every
    // FP4 GEMV intercept (decode and verify alike) so the fallback paths
    // can be proven bit-identical. Off unless set.
    static const int force_refuse = [] {
        const char * e = getenv("GUFO_FP4_FORCE_REFUSE");
        return (e != nullptr && e[0] == '1') ? 1 : 0;
    }();
    if (force_refuse) return GUFO_FP4_ERR_ARG;
    if (W == nullptr || x == nullptr || y == nullptr || n <= 0 || k <= 0 || (k % 32) != 0 ||
        m < 2 || m > 8) {
        return GUFO_FP4_ERR_ARG;
    }
    if (qtype != GUFO_FP4_Q4_0_ROCMFP4 && qtype != GUFO_FP4_Q4_0_ROCMFP4_FAST) return GUFO_FP4_ERR_ARG;
    const int set = gufo_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // R2 refusal: the caller declines, never waits.
    int8_t * qv = nullptr;
    float * dv = nullptr;
    {
        GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        GufoFp4SlotSet & S = mgr.sets[(size_t) set];
        const long long need = (long long) m * (long long) k;
        if (need > S.qv_cap || S.qv == nullptr || S.dv == nullptr) {
            static int warned = 0;
            const int cap = S.qv_cap;
            if (warned++ < 4) {
                fprintf(stderr, "VERIFY_M_SHORT set=%d need=%lld cap=%d\n", set, need, cap);
            }
            return GUFO_FP4_ERR_ARG;
        }
        if (!quantize_x && (S.vq_x != x || S.vq_m != m || S.vq_k != k)) {
            return GUFO_FP4_ERR_ARG;
        }
        qv = S.qv;
        dv = S.dv;
    }
    const size_t rb = gufo_fp4_row_bytes(qtype, k);
    if (rb == 0) return GUFO_FP4_ERR_ARG;
    if (quantize_x) {
        const int nb = (m * k) / 32;
        const int threads = 256;
        gufo_fp4_k_q8_1_quant<<<(nb + threads - 1) / threads, threads, 0, stream>>>(
            x, qv, dv, nb);
        const int qrc = ls1_chk("verify quant");
        GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        GufoFp4SlotSet & S = mgr.sets[(size_t) set];
        S.vq_x = qrc == GUFO_FP4_OK ? x : nullptr;
        S.vq_m = m;
        S.vq_k = k;
        if (qrc != GUFO_FP4_OK) return GUFO_FP4_ERR_HIP;
    }
    using verify_kn = void (*)(int, const uint8_t *, size_t, const int8_t *,
                                const float *, float *, int, int);
    verify_kn kn = nullptr;
    // lqw:spec7. Two rows per CTA: the grid covers row pairs.
    const int npair = (n + 1) >> 1;
    const int grid =
        npair <= GUFO_FP4_LS1_GEMV_WIDE_MAX ? npair : GUFO_FP4_LS1_GEMV_GS_BLOCKS;
    switch (m) {
        case 2: kn = gufo_fp4_k_gemv_verify_mt<2>; break;
        case 3: kn = gufo_fp4_k_gemv_verify_mt<3>; break;
        case 4: kn = gufo_fp4_k_gemv_verify_mt<4>; break;
        case 5: kn = gufo_fp4_k_gemv_verify_mt<5>; break;
        case 6: kn = gufo_fp4_k_gemv_verify_mt<6>; break;
        case 7: kn = gufo_fp4_k_gemv_verify_mt<7>; break;
        case 8: kn = gufo_fp4_k_gemv_verify_mt<8>; break;
        default: return GUFO_FP4_ERR_ARG;
    }

    kn<<<(unsigned) grid, 64, 0, stream>>>(
        qtype, (const uint8_t *) W, rb, qv, dv, y, k, n);
    return ls1_chk("verify gemv");
}

extern "C" int gufo_fp4_gemv_verify_m(int qtype, const void * W, const float * x, float * y,
                                      int k, int n, int m, hipStream_t stream) {
    return gemv_verify_m_impl(qtype, W, x, y, k, n, m, stream, true);
}

extern "C" int gufo_fp4_gemv_verify_m_xq(int qtype, const void * W, const float * x, float * y,
                                         int k, int n, int m, hipStream_t stream) {
    return gemv_verify_m_impl(qtype, W, x, y, k, n, m, stream, false);
}

// ---------------------------------------------------------------------------
// SQ10: slot-set ownership API. See the header for the contract.
// (GUFO_FP4_LS1_MAX_SETS lives above the manager: the resolver needs it.)
// ---------------------------------------------------------------------------

extern "C" int gufo_fp4_slot_sets_configure(int n) {
    // SQ19-2: pre-sizing, never a refusal. Ensures at least n sets exist so
    // a server's sessions never grow one by one; grows only, so existing
    // owners keep their indices (never renumbers) and repeating with any n
    // is a no-op that still returns OK.
    if (n < 1) n = 1;
    if (n > GUFO_FP4_LS1_MAX_SETS) n = GUFO_FP4_LS1_MAX_SETS;
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    if (mgr.sets.size() < (size_t) n) {
        mgr.sets.resize((size_t) n);
        mgr.busy.resize((size_t) n, 0);
    }
    mgr.configured = true;
    return GUFO_FP4_OK;
}

extern "C" int gufo_fp4_slot_sets_count(void) {
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    return (int) mgr.sets.size();
}

extern "C" int gufo_fp4_slot_stream_attach(hipStream_t stream) {
    return gufo_fp4_slots_for_stream_at(stream, __func__, 0, false);
}

extern "C" int gufo_fp4_slot_stream_attach_at(hipStream_t stream, const char * func, long width) {
    return gufo_fp4_slots_for_stream_at(stream, func, width, false);
}

extern "C" int gufo_fp4_slot_stream_attach_grow(hipStream_t stream, const char * func, long width) {
    return gufo_fp4_slots_for_stream_at(stream, func, width, true);
}

extern "C" int gufo_fp4_slot_stream_attach_wait(hipStream_t stream) {
    return gufo_fp4_slots_for_stream_wait(stream);
}

// SPEC2 (R1): a helper stream of a session uses that session's set. The
// owner must already be attached (sessions pre-attach at load); the helper
// must not own a set of its own, and the owner must not itself be aliased
// (one hop only). Idempotent for the same pair. Returns the owner's set.
extern "C" int gufo_fp4_slot_stream_alias(hipStream_t helper, hipStream_t owner) {
    if (helper == nullptr || owner == nullptr || helper == owner) {
        return GUFO_FP4_ERR_ARG;
    }
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    const std::map<hipStream_t, int>::const_iterator own = mgr.stream_set.find(owner);
    if (own == mgr.stream_set.end()) {
        return GUFO_FP4_ERR_ARG;
    }
    const std::map<hipStream_t, hipStream_t>::const_iterator chained = mgr.alias.find(owner);
    if (chained != mgr.alias.end()) {
        return GUFO_FP4_ERR_ARG;
    }
    const std::map<hipStream_t, int>::const_iterator hs = mgr.stream_set.find(helper);
    if (hs != mgr.stream_set.end() && hs->second != own->second) {
        return GUFO_FP4_ERR_ARG;
    }
    const std::map<hipStream_t, hipStream_t>::const_iterator ha = mgr.alias.find(helper);
    if (ha != mgr.alias.end() && ha->second != owner) {
        return GUFO_FP4_ERR_ARG;
    }
    mgr.alias[helper] = owner;
    return own->second;
}

extern "C" int gufo_fp4_slot_stream_detach(hipStream_t stream) {
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    bool erased = false;
    {
        std::lock_guard<std::mutex> lk(mgr.mu);
        const std::map<hipStream_t, int>::iterator it = mgr.stream_set.find(stream);
        if (it != mgr.stream_set.end()) {
            mgr.busy[(size_t) it->second] = 0;
            mgr.stream_set.erase(it);
            erased = true;
        }
        // SPEC2 (R3): a destroyed executor detaches its stream AND its
        // aliases, both directions, so no destroyed handle stays in the map
        // (HIP may reuse the address for a later stream).
        if (mgr.alias.erase(stream) != 0) {
            erased = true;
        }
        for (std::map<hipStream_t, hipStream_t>::iterator al = mgr.alias.begin();
             al != mgr.alias.end();) {
            if (al->second == stream) {
                mgr.alias.erase(al++);
                erased = true;
            } else {
                ++al;
            }
        }
        mgr.warned.erase(stream);
    }
    if (!erased) return GUFO_FP4_ERR_ARG;
    mgr.cv.notify_all();
    return GUFO_FP4_OK;
}

extern "C" unsigned long gufo_fp4_slot_alloc_count(void) {
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    return mgr.alloc_count;
}

extern "C" void gufo_fp4_slot_count_alloc(void) {
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    mgr.alloc_count++;
}

extern "C" unsigned long gufo_fp4_slot_wait_count(void) {
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    return mgr.wait_count;
}

extern "C" unsigned long gufo_fp4_slot_wait_us(void) {
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    return mgr.wait_us;
}

extern "C" unsigned long gufo_fp4_slot_refuse_count(void) {
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    return mgr.refuse_count;
}

extern "C" int gufo_fp4_slot_debug_map_size(void) {
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    return (int) mgr.stream_set.size();
}

extern "C" int gufo_fp4_slot_debug_alias_size(void) {
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    return (int) mgr.alias.size();
}

// Pre-grow every set's five Q8 slots plus the verify pair, launching the same
// quant kernel warmup always launched. Runs at load, outside any capture.
extern "C" int gufo_fp4_slots_warm_q8(int k, hipStream_t stream) {
    if (k <= 0 || (k % 32) != 0) return GUFO_FP4_ERR_ARG;
    // lqw:spec6. The verify GEMV quantises m<=8 rows into a buffer grown here,
    // at load, so a captured step never hipMallocs. A reserve failure is
    // ignored exactly as before: the verify call falls back per call with
    // VERIFY_M_SHORT, while the warmed slots still serve.
    if (k <= INT_MAX / 8) {
        (void) gufo_fp4_verify_m_reserve(k * 8);
    }
    float * zeros = nullptr;
    if (hipMalloc(&zeros, (size_t) k * sizeof(float)) != hipSuccess) {
        return GUFO_FP4_ERR_HIP;
    }
    if (hipMemset(zeros, 0, (size_t) k * sizeof(float)) != hipSuccess) {
        (void) hipFree(zeros);
        return GUFO_FP4_ERR_HIP;
    }
    int rc = GUFO_FP4_OK;
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    int nsets = 0;
    {
        std::lock_guard<std::mutex> lk(mgr.mu);
        nsets = (int) mgr.sets.size();
    }
    for (int s = 0; s < nsets && rc == GUFO_FP4_OK; ++s) {
        for (int slot = 0; slot < GUFO_FP4_LS1_NSLOT && rc == GUFO_FP4_OK; ++slot) {
            int8_t * qs = nullptr;
            float * ds = nullptr;
            rc = gufo_fp4_slots_grow_q8(s, slot, k, stream, &qs, &ds);
            if (rc != GUFO_FP4_OK) break;
            const int nb = k / 32;
            const int threads = 256;
            const int blocks = (nb + threads - 1) / threads;
            gufo_fp4_k_q8_1_quant<<<blocks, threads, 0, stream>>>(zeros, qs, ds, nb);
            rc = ls1_chk("slot warm quant");
        }
    }
    if (rc == GUFO_FP4_OK && hipStreamSynchronize(stream) != hipSuccess) {
        rc = GUFO_FP4_ERR_HIP;
    }
    (void) hipFree(zeros);
    return rc;
}

// Pre-grow every set's single-token and small-m MoE scratch plus the m-pair.
// Zero sizes are a no-op (dense models have no experts). Alloc-only: the
// kernels fill hs on first use, exactly as the lazy path did.
extern "C" int gufo_fp4_slots_warm_moe(int hs_elems, int hs_m_elems, int m_total) {
    if (hs_elems < 0 || hs_m_elems < 0 || m_total < 0) return GUFO_FP4_ERR_ARG;
    if ((m_total % 32) != 0) return GUFO_FP4_ERR_ARG;
    GufoFp4SlotMgr & mgr = gufo_fp4_slots_mgr();
    int nsets = 0;
    {
        std::lock_guard<std::mutex> lk(mgr.mu);
        nsets = (int) mgr.sets.size();
    }
    for (int s = 0; s < nsets; ++s) {
        if (hs_elems > 0) {
            float * hs = nullptr;
            int * mrc = nullptr;
            const int rc = gufo_fp4_slots_grow_hs(s, (size_t) hs_elems, nullptr, &hs, &mrc);
            if (rc != GUFO_FP4_OK) return rc;
        }
        if (hs_m_elems > 0) {
            float * hs = nullptr;
            int * mrc = nullptr;
            const int rc = gufo_fp4_slots_grow_hsm(s, (size_t) hs_m_elems, nullptr, &hs, &mrc);
            if (rc != GUFO_FP4_OK) return rc;
        }
        if (m_total > 0) {
            int8_t * qs = nullptr;
            float * ds = nullptr;
            const int rc = gufo_fp4_slots_grow_q8m(s, m_total, nullptr, &qs, &ds);
            if (rc != GUFO_FP4_OK) return rc;
        }
    }
    return GUFO_FP4_OK;
}

}  // namespace gufo_fp4_ks
