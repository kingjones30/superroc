// The decode-step GEMV as an integer dot against a Q8_1
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
// table is exactly superroc_fp4_rocmfp4_code (0,1,2,3,4,6,8,10 / -...), unscaled. This is
// the fork's / ggml's computation, not superroc's float one.
//
// There is deliberately NO cache of "which vector is currently quantised". A
// pointer-keyed one would be wrong: rocm_backend.cpp reuses st->normed for every
// layer, so the second layer's rms_norm overwrites the bytes behind the pointer
// the first layer quantised, and a pointer match would then mean the wrong data.
// Every entry point here either quantises the vector it was handed or is handed
// a buffer that the caller filled for that exact call. Slots exist only so the
// expert FFN can reuse one quantisation across several matvecs it owns itself.

#include "superroc_fp4_quant.hpp"

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
#include "src/core/env.hpp"
namespace superroc_fp4_ks {

static int ls1_chk(const char * what) {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        fprintf(stderr, "ling3 q8 %s: %s\n", what, hipGetErrorString(e));
        return SUPERROC_FP4_ERR_HIP;
    }
    return SUPERROC_FP4_OK;
}

// ---------------------------------------------------------------------------
// FP4 nibble -> int8 codebook.
//
// Verbatim from the fork, ggml/rocmfp4/rocmfp4_hip_codebook.cuh
// (rocmfp4_get_int_from_codebook_16, its GGML_USE_HIP branch). Four nibbles go
// in; int2 of eight int8 comes out: .x is the LOW nibbles of bytes 0..3, which
// are elements 0..3 of the block, .y is the HIGH nibbles, elements 16..19. The
// table [0,1,2,3],[4,6,8,10],[0,-1,-2,-3],[-4,-6,-8,-10] is exactly
// superroc_fp4_rocmfp4_code(nib), and tools/superroc_fp4_gemv_q8_test.cpp checks all 65536
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
    c.slo = superroc_fp4_ue4m3_half(blk[16]);
    c.shi = (qtype == SUPERROC_FP4_Q4_0_ROCMFP4) ? superroc_fp4_ue4m3_half(blk[17]) : c.slo;
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
    const float slo = superroc_fp4_ue4m3_half(blk[16]);
    const float shi = (qtype == SUPERROC_FP4_Q4_0_ROCMFP4) ? superroc_fp4_ue4m3_half(blk[17]) : slo;
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
    SUPERROC_FP4_LS1_SLOT_GENERIC = 0,   // superroc_fp4_gemv_fp4()'s own scratch
    SUPERROC_FP4_LS1_SLOT_FFN_X  = 1,    // the expert FFN's input, shared by gate and up
    SUPERROC_FP4_LS1_SLOT_FFN_H  = 2,    // the expert FFN's SwiGLU output, input to down
    SUPERROC_FP4_LS1_SLOT_MOE_H  = 3,    // the grouped expert FFN's SwiGLU output, input to down
    SUPERROC_FP4_LS1_SLOT_ATTN   = 4,    // st->normed, once per layer, shared by its projections
    SUPERROC_FP4_LS1_NSLOT      = 5
};

namespace {

// Hard ceiling: 64 streams x ~462 KiB (27B) is ~29 MiB of scratch, and no seat
// runs that many sessions; past it the entry refuses, it never aliases.
enum { SUPERROC_FP4_LS1_MAX_SETS = 64 };

// ---------------------------------------------------------------------------
// Slot SETS, one per decode stream. Every buffer below used to be a
// single process-global pair, so two decode streams interleaving quantize ->
// GEMV read each other's activations and `-j 1` was the only safe seat. A set
// is the same buffers with an owner: streams attach 1:1 (serve pre-attaches
// each arena stream at load; any other stream attaches lazily on first use),
// and a session's helper (drafter) stream is aliased to its owner's set
//. No free set -> the entry refuses at once with -SUPERROC_FP4_ERR_BUSY
// and the caller declines to its eager fallback; nothing in
// production waits. Growth stays per-set with the same kernels, so the `-j 1`
// path is bit-identical and unwarmed callers (tools, tests, the Flash-Next
// eager-first-pass) behave exactly as before; warmup pre-grows every set so
// no captured path allocates. alloc_count counts every slot hipMalloc and the
// test asserts it does not move across decode.
// Every load-time attach gets a set, configured or not. The load-time
// entry (superroc_fp4_slot_stream_attach_grow, from the executor constructor)
// reuses a free set and otherwise grows the table by one, up to the same
// ceiling; the caller warms the set's buffers before any step. Past the cap
// it refuses exactly like an in-launch attach. In-launch attach keeps
// refusing and is the only refusal left; configure() only pre-sizes.
// ---------------------------------------------------------------------------
struct SuperROCFp4SlotSet {
    int8_t * qs[SUPERROC_FP4_LS1_NSLOT] = {};
    float * ds[SUPERROC_FP4_LS1_NSLOT] = {};
    int cap[SUPERROC_FP4_LS1_NSLOT] = {};
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
    // Super Roc QVO: the input the qv/dv pair holds, as quantised by the last
    // superroc_fp4_gemv_verify_m in this set (cleared when the pair is replaced).
    const float * vq_x = nullptr;
    int vq_m = 0;
    int vq_k = 0;
};

struct SuperROCFp4SlotMgr {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<SuperROCFp4SlotSet> sets = std::vector<SuperROCFp4SlotSet>(1);
    std::vector<char> busy = std::vector<char>(1, 0);
    std::map<hipStream_t, int> stream_set;
    // Helper stream -> owner (session) stream. A drafter or
    // helper stream of a session uses that session's set; the host alternates
    // the two streams (the helper's last launch is synced before the session's
    // next launch), so they never hold the set at once. One hop only: an
    // owner is never itself aliased.
    std::map<hipStream_t, hipStream_t> alias;
    // Streams already warned about a refusal, and the
    // lifetime refusal count. Production never waits; it refuses at once.
    std::set<hipStream_t> warned;
    unsigned long alloc_count = 0;
    unsigned long wait_count = 0;
    unsigned long wait_us = 0;
    unsigned long refuse_count = 0;
    bool configured = false;
};

inline SuperROCFp4SlotMgr & superroc_fp4_slots_mgr(void) {
    static SuperROCFp4SlotMgr mgr;
    return mgr;
}

// Counted malloc. Call with mgr.mu held.
static int superroc_fp4_slots_malloc(void ** p, size_t n) {
    if (hipMalloc(p, n) != hipSuccess) {
        *p = nullptr;
        return SUPERROC_FP4_ERR_HIP;
    }
    superroc_fp4_slots_mgr().alloc_count++;
    return SUPERROC_FP4_OK;
}

// Resolve stream -> set, attaching a free set when the stream is unseen. An
// aliased helper stream resolves to its owner's set. When no set is
// free the stream is REFUSED at once with -SUPERROC_FP4_ERR_BUSY: one WARN
// per stream on stderr plus a lifetime counter; production entries decline
// to their eager fallback exactly as for an unwarmed width. Host-side only
// (mutexes and map lookups), so calling it inside a captured step is legal;
// it performs no HIP call. The WARN names the declining
// launch (function + tensor width) so a fallback is diagnosable.
// grow_at_load (the load-time attach entry only) grows the table by
// one set instead of refusing, configured or not. The new set's buffers are
// empty: the load-time caller warms them before any step, never here (a
// hipMalloc inside a captured step poisons the capture). In-launch callers
// pass false and keep refusing.
static int superroc_fp4_slots_for_stream_at(hipStream_t stream, const char * func, long width,
                                        bool grow_at_load) {
    const char * const who = (func != nullptr && func[0] != '\0') ? func : "?";
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
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
    if (grow_at_load && mgr.sets.size() < (size_t) SUPERROC_FP4_LS1_MAX_SETS) {
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
        fprintf(stderr, "SUPERROC_FP4_SLOT_REFUSE stream=%p func=%s width=%ld refusals=%lu\n",
                (const void *) stream, who, width, refusals);
    }
    return -SUPERROC_FP4_ERR_BUSY;
}

// The waiting resolver. Production never calls this; it survives only
// for the unit test's exhaustion case, which proves the wait/wake machinery.
// Same resolution as above, but an unseen stream with no free set blocks on
// the condvar until a detach frees one. Waits are counted (and, with
// SUPERROC_FP4_SLOT_TRACE=1 in the environment, logged with their length).
static int superroc_fp4_slots_for_stream_wait(hipStream_t stream) {
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
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
    static const bool trace = ::superroc::core::GetEnv("SUPERROC_FP4_SLOT_TRACE") != nullptr;
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
                    fprintf(stderr, "SUPERROC_FP4_SLOT_WAIT set=%d us=%lu\n", (int) i, us);
                }
                return (int) i;
            }
        }
    }
}

// The four growth shapes, one place each (they used to be open-coded at every
// call site). Each syncs the stream first and never frees the buffers it
// replaces (captured decode graphs keep launching with those pointers);
// growth never runs inside a capture because warmup pre-grows every set.
static int superroc_fp4_slots_grow_q8(int set, int slot, int k, hipStream_t stream,
                                  int8_t ** qs, float ** ds) {
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    SuperROCFp4SlotSet & S = mgr.sets[(size_t) set];
    if (k > S.cap[slot] || S.qs[slot] == nullptr || S.ds[slot] == nullptr) {
        if (hipStreamSynchronize(stream) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
        // The prior pair stays alive, not freed: decode graphs capture those pointers. The new
        // pair replaces it only once both allocations succeeded, so a failed growth leaves the
        // slot as it was.
        // +SUPERROC_FP4_TAIL_PAD so a widened tail load still has readable bytes behind
        // the last element, the same contract every other activation has here.
        int8_t * qs_new = nullptr;
        float * ds_new = nullptr;
        if (superroc_fp4_slots_malloc((void **) &qs_new, (size_t) k + SUPERROC_FP4_TAIL_PAD) != SUPERROC_FP4_OK) {
            return SUPERROC_FP4_ERR_HIP;
        }
        if (superroc_fp4_slots_malloc((void **) &ds_new,
                                  (size_t) (k / 32) * sizeof(float) + SUPERROC_FP4_TAIL_PAD) != SUPERROC_FP4_OK) {
            (void) hipFree(qs_new);
            return SUPERROC_FP4_ERR_HIP;
        }
        S.qs[slot] = qs_new;
        S.ds[slot] = ds_new;
        S.cap[slot] = k;
    }
    *qs = S.qs[slot];
    *ds = S.ds[slot];
    return SUPERROC_FP4_OK;
}

static int superroc_fp4_slots_get_q8(int set, int slot, int k,
                                 int8_t ** qs, float ** ds) {
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    SuperROCFp4SlotSet & S = mgr.sets[(size_t) set];
    if (S.qs[slot] == nullptr || S.cap[slot] < k) return SUPERROC_FP4_ERR_ARG;
    *qs = S.qs[slot];
    *ds = S.ds[slot];
    return SUPERROC_FP4_OK;
}

static int superroc_fp4_slots_grow_hs(int set, size_t nn, hipStream_t stream,
                                  float ** hs, int ** rc) {
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    SuperROCFp4SlotSet & S = mgr.sets[(size_t) set];
    if ((int) nn > S.moe_hs_cap || S.moe_hs == nullptr || S.moe_rc == nullptr) {
        if (hipStreamSynchronize(stream) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
        // The prior pair stays alive, not freed: decode graphs capture those pointers. The new
        // pair replaces it only once it is allocated and its counter is zeroed.
        float * hs_new = nullptr;
        int * rc_new = nullptr;
        if (superroc_fp4_slots_malloc((void **) &hs_new, nn * sizeof(float) + SUPERROC_FP4_TAIL_PAD) != SUPERROC_FP4_OK) {
            return SUPERROC_FP4_ERR_HIP;
        }
        if (superroc_fp4_slots_malloc((void **) &rc_new, sizeof(int)) != SUPERROC_FP4_OK) {
            (void) hipFree(hs_new);
            return SUPERROC_FP4_ERR_HIP;
        }
        if (hipMemset(rc_new, 0, sizeof(int)) != hipSuccess) {
            (void) hipFree(hs_new);
            (void) hipFree(rc_new);
            return SUPERROC_FP4_ERR_HIP;
        }
        S.moe_hs = hs_new;
        S.moe_rc = rc_new;
        S.moe_hs_cap = (int) nn;
    }
    *hs = S.moe_hs;
    *rc = S.moe_rc;
    return SUPERROC_FP4_OK;
}

static int superroc_fp4_slots_grow_q8m(int set, int total, hipStream_t stream,
                                   int8_t ** qs, float ** ds) {
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    SuperROCFp4SlotSet & S = mgr.sets[(size_t) set];
    if (total > S.cap_m || S.qs_m == nullptr || S.ds_m == nullptr) {
        if (hipStreamSynchronize(stream) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
        // Keep prior qs/ds alive: decode graphs capture those pointers (same reason as hs/rc below).
        // One abandoned pair per size increase per set; sizes are model constants, so this is bounded.
        int8_t * qs_new = nullptr;
        float * ds_new = nullptr;
        if (superroc_fp4_slots_malloc((void **) &qs_new, (size_t) total + SUPERROC_FP4_TAIL_PAD) != SUPERROC_FP4_OK) {
            return SUPERROC_FP4_ERR_HIP;
        }
        if (superroc_fp4_slots_malloc((void **) &ds_new,
                                  (size_t) (total / 32) * sizeof(float) + SUPERROC_FP4_TAIL_PAD) != SUPERROC_FP4_OK) {
            (void) hipFree(qs_new);
            return SUPERROC_FP4_ERR_HIP;
        }
        S.qs_m = qs_new;
        S.ds_m = ds_new;
        S.cap_m = total;
    }
    *qs = S.qs_m;
    *ds = S.ds_m;
    return SUPERROC_FP4_OK;
}

static int superroc_fp4_slots_grow_hsm(int set, size_t nn, hipStream_t stream,
                                   float ** hs, int ** rc) {
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    SuperROCFp4SlotSet & S = mgr.sets[(size_t) set];
    if ((int) nn > S.moe_hs_m_cap || S.moe_hs_m == nullptr || S.moe_rc_m == nullptr) {
        if (hipStreamSynchronize(stream) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
        // Keep prior hs/rc alive: decode graphs capture those pointers. Freeing
        // them lets the next malloc reuse the bytes as moe_rc_m, and a replayed
        // hs store then fails m_check with a float bit pattern.
        float * hs_new = nullptr;
        int * rc_new = nullptr;
        if (superroc_fp4_slots_malloc((void **) &hs_new, nn * sizeof(float) + SUPERROC_FP4_TAIL_PAD) != SUPERROC_FP4_OK) {
            return SUPERROC_FP4_ERR_HIP;
        }
        if (superroc_fp4_slots_malloc((void **) &rc_new, sizeof(int)) != SUPERROC_FP4_OK) {
            (void) hipFree(hs_new);
            return SUPERROC_FP4_ERR_HIP;
        }
        if (hipMemset(rc_new, 0, sizeof(int)) != hipSuccess) {
            (void) hipFree(hs_new);
            (void) hipFree(rc_new);
            return SUPERROC_FP4_ERR_HIP;
        }
        S.moe_hs_m = hs_new;
        S.moe_rc_m = rc_new;
        S.moe_hs_m_cap = (int) nn;
    }
    *hs = S.moe_hs_m;
    *rc = S.moe_rc_m;
    return SUPERROC_FP4_OK;
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
__global__ void superroc_fp4_k_q8_1_quant(const float * x, int8_t * qs, float * ds, int nb) {
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
extern "C" int superroc_fp4_quant_q8(const float * x, int k, int slot, hipStream_t stream) {
    if (x == nullptr || k <= 0 || (k % 32) != 0 || slot < 0 || slot >= SUPERROC_FP4_LS1_NSLOT) {
        return SUPERROC_FP4_ERR_ARG;
    }
    const int set = superroc_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // Refused: the caller declines, never waits.
    int8_t * qs = nullptr;
    float * ds = nullptr;
    const int rc = superroc_fp4_slots_grow_q8(set, slot, k, stream, &qs, &ds);
    if (rc != SUPERROC_FP4_OK) return rc;
    const int nb = k / 32;
    const int threads = 256;
    const int blocks = (nb + threads - 1) / threads;
    superroc_fp4_k_q8_1_quant<<<blocks, threads, 0, stream>>>(x, qs, ds, nb);
    return ls1_chk("q8 quant");
}

// One row of the integer matvec, before the cross-lane reduction: thread `threadIdx.x` and
// its stride partners walk the row's 32-value blocks. Kept separate from the kernel so the
// grouped expert FFN can use the same dot with a different thread mapping -- and, in phase 3,
// so that mapping's reduction goes through the same ls1_block_sum rather than a 32-lane one.
__device__ __forceinline__ float ls1_dot_row_q8(int qtype, const uint8_t * w,
                                                const int8_t * xq, const float * xd,
                                                int k) {
    const int tsz = superroc_fp4_type_size(qtype);
    const int nblk = k / 32;
    float s = 0.f;
    for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
        s += ls1_dot_fp4_blk32_q8(qtype, w + (size_t) qb * (size_t) tsz,
                                  xq + (size_t) qb * 32, xd[qb]);
    }
    return s;
}

// ls1_dot_row_q8 for a row whose 32-value blocks are `stride` bytes apart.
__device__ __forceinline__ float ls1_dot_row_q8_stride(int qtype, const uint8_t * w,
                                                       const int8_t * xq, const float * xd,
                                                       int k, int stride) {
    const int nblk = k / 32;
    float s = 0.f;
    for (int qb = (int) threadIdx.x; qb < nblk; qb += (int) blockDim.x) {
        s += ls1_dot_fp4_blk32_q8(qtype, w + (size_t) qb * (size_t) stride,
                                  xq + (size_t) qb * 32, xd[qb]);
    }
    return s;
}

// Byte offset of row j of expert id when K blocks interleave inside groups of `group` rows (a power
// of two): inside a group a row's blocks are group * tsz bytes apart, and group = 1 is row-major.
__device__ __forceinline__ size_t ls1_grouped_row(int id, int n_rows, int j, size_t rb, int tsz,
                                                  int group) {
    return ((size_t) id * (size_t) n_rows + (size_t) (j & -group)) * rb +
           (size_t) (j & (group - 1)) * (size_t) tsz;
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

// The 32-lane tree of ls1_warp_sum with row permutes in place of the LDS shuffles: the same
// additions in the same order, and only lane 0 of the result means anything. Every lane must be
// active.
__device__ __forceinline__ float ls1_lane0_sum(float s) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    union Bits {
        float f;
        unsigned u;
    };
    Bits a;
    Bits b;
    a.f = s;
    b.u = __builtin_amdgcn_permlanex16(0u, a.u, 0x76543210u, 0xfedcba98u, false, false);
    a.f = a.f + b.f;
    b.u = __builtin_amdgcn_permlane16(0u, a.u, 0xfedcba98u, 0x76543210u, false, false);
    a.f = a.f + b.f;
    b.u = __builtin_amdgcn_permlane16(0u, a.u, 0xba987654u, 0x3210fedcu, false, false);
    a.f = a.f + b.f;
    b.u = __builtin_amdgcn_permlane16(0u, a.u, 0x98765432u, 0x10fedcbau, false, false);
    a.f = a.f + b.f;
    b.u = __builtin_amdgcn_permlane16(0u, a.u, 0x87654321u, 0x0fedcba9u, false, false);
    a.f = a.f + b.f;
    return a.f;
}

// A row's two trees summed the way ls1_block_sum sums its two waves. With 32 blocks or fewer no
// thread past 31 has work, so the second tree is a sum of +0: adding it changes nothing, since
// 0 + x is never -0.
__device__ __forceinline__ float ls1_row_sum(float even, float odd, int nblk) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    float r = 0.f + ls1_lane0_sum(even);
    if (nblk > 32) {
        r = r + ls1_lane0_sum(odd);
    }
    return r;
}

// Launch n rows against m columns in Q8_1, one wave per group. Defined after its kernel.
static int ls1_launch_rows_q8(int qtype, const uint8_t * W, size_t rb, const int8_t * xq,
                              const float * xd, float * y, int k, int n, int m,
                              hipStream_t stream);

// The phase-1 kernel, kept because it is what the unit test compares the warp-per-row
// one against: superroc_fp4_k_gemv_wide_q8_rows must be BIT-IDENTICAL to this.
__global__ void superroc_fp4_k_gemv_wide_q8_cta64(int qtype, const uint8_t * W, size_t row_bytes,
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
    const int tsz = superroc_fp4_type_size(qtype);
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
// One WARP per row, several consecutive rows per CTA.
//
// The measured shape of the two weak decode GEMVs (from an earlier profile)
// is a lane-utilisation problem, not a bandwidth one:
//
//   superroc_fp4_k_moe_down_weighted   n_ff = 768  -> 24 blocks of 32 values, spread over a 64-lane
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
__global__ void superroc_fp4_k_gemv_wide_q8_rows(int qtype, const uint8_t * W, size_t row_bytes,
                                    const int8_t * xq, const float * xd,
                                    float * y, int k, int n, int grid_rows) {
    const int lane = threadIdx.x & 31;
    const int sub = (int) (threadIdx.x >> 5);
    const int nwarps = (int) blockDim.x >> 5;
    const int tsz = superroc_fp4_type_size(qtype);
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
__global__ void superroc_fp4_k_moe_down_weighted_rows(int qtype, const uint8_t * down, size_t rb_ff,
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
                *rc = SUPERROC_FP4_ERR_ARG;
            }
            const size_t base = ok ? (((size_t) id * (size_t) k + (size_t) j) * rb_ff) : 0;
            const uint8_t * drow = down + base;
            float part = 0.f;
            for (int qb = lane; qb < nb; qb += 32) {
                part += ls1_dot_fp4_blk32_q8(qtype, drow + (size_t) qb * (size_t) superroc_fp4_type_size(qtype),
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
__global__ void superroc_fp4_k_gemv_wide_q8_gs(int qtype, const uint8_t * W, size_t row_bytes,
                                  const int8_t * xq, const float * xd,
                                  float * y, int k, int n) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int tsz = superroc_fp4_type_size(qtype);
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
// above it there is already plenty of parallelism. Same cut as superroc_fp4_gemv.
#define SUPERROC_FP4_LS1_GEMV_WIDE_MAX 8192
#define SUPERROC_FP4_LS1_GEMV_GS_BLOCKS 2048

// Matvec against an activation the caller already quantised into `slot`.
// ---------------------------------------------------------------------------
// PHASE6 selection. Variant 0 is the phase-1 CTA-per-row kernel, unchanged; variant 1
// is the warp-per-row kernel. SUPERROC_FP4_GEMV_VARIANT picks it and DEFAULTS TO 0, so a
// build with no environment set behaves exactly as it did before this phase. The
// unit test runs both and requires them to agree bit-for-bit.
// ---------------------------------------------------------------------------
// The flag is SPLIT, because its two halves are not the same claim.
//
//   SUPERROC_FP4_GEMV_DOWN  (default ON)  -- the warp-per-row grouped expert FFN. MEASURED
//       bit-identical: max_rel_diff 0, 0 differing of 2560, in all three
//       configurations. There is no numerical risk in it.
//
//   SUPERROC_FP4_GEMV_GEMV  (default OFF) -- the warp-per-row plain matvec. It regroups a
//       row's float additions (float addition is not associative), so ~60 % of
//       output elements differ in their low bits and the greedy text diverges
//       from the earlier kernel at character 194. Its error against the CPU
//       integer reference is the same ~1e-6 band as the kernel it replaces,
//       and the perplexity gates pass (step 6.1014 vs
//       batched 6.1037 on positions 4096-8191; 4096-token step 9.2905 vs batched
//       9.3094) -- but end to end it bought 0.1 % (37.90/37.95 tok/s against 37.89/37.87
//       with it off), which does not pay for changing greedy output.
static int ls1_env_on(const char * name, int dflt) {
    const char * e = ::superroc::core::GetEnv(name);
    if (e == nullptr || e[0] == '\0') {
        return dflt;
    }
    return (e[0] == '1') ? 1 : 0;
}
static int ls1_variant_down() {
    static int v = -1;
    if (v < 0) {
        v = ls1_env_on("SUPERROC_FP4_GEMV_DOWN", 1);
        if (!v && ls1_env_on("SUPERROC_FP4_GEMV_VARIANT", 0)) {
            v = 1;   // the old single switch still turns both on
        }
    }
    return v;
}
static int ls1_variant_gemv() {
    static int v = -1;
    if (v < 0) {
        v = ls1_env_on("SUPERROC_FP4_GEMV_GEMV", 0);
        if (v == 0 && ls1_env_on("SUPERROC_FP4_GEMV_VARIANT", 0)) {
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
extern "C" int superroc_fp4_gemv_fp4_q8_rows(int qtype, const void * W, int slot, float * y,
                                      int k, int n, int block, hipStream_t stream) {
    const size_t rb = superroc_fp4_row_bytes(qtype, k);
    if (!W || !y || rb == 0 || n <= 0 || (k % 32) != 0 || block < 32) return SUPERROC_FP4_ERR_ARG;
    if (slot < 0 || slot >= SUPERROC_FP4_LS1_NSLOT) return SUPERROC_FP4_ERR_ARG;
    const int set = superroc_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // Refused: the caller declines, never waits.
    int8_t * qs = nullptr;
    float * ds = nullptr;
    if (superroc_fp4_slots_get_q8(set, slot, k, &qs, &ds) != SUPERROC_FP4_OK) return SUPERROC_FP4_ERR_ARG;
    const int nwarps = block >> 5;
    superroc_fp4_k_gemv_wide_q8_rows<<<ls1_gemv_blocks(n, nwarps), block, 0, stream>>>(
        qtype, (const uint8_t *) W, rb, qs, ds, y, k, n, 1 << 20);
    return ls1_chk("gemv rows");
}

// Quantise into an explicitly named slot, for the call sites
// that run several matvecs against ONE activation. superroc_fp4_gemv_fp4 still quantises for
// itself; this is the opt-in that lets a caller pay once.
extern "C" int superroc_fp4_act_prepare(const float * x, int k, int slot, hipStream_t stream) {
    return superroc_fp4_quant_q8(x, k, slot, stream);
}

extern "C" int superroc_fp4_gemv_fp4_q8(int qtype, const void * W, int slot,
                                 float * y, int k, int n, hipStream_t stream) {
    size_t rb = superroc_fp4_row_bytes(qtype, k);
    if (W == nullptr || y == nullptr || rb == 0 || n <= 0 || (k % 32) != 0 ||
        slot < 0 || slot >= SUPERROC_FP4_LS1_NSLOT) {
        return SUPERROC_FP4_ERR_ARG;
    }
    const int set = superroc_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // Refused: the caller declines, never waits.
    int8_t * qs = nullptr;
    float * ds = nullptr;
    if (superroc_fp4_slots_get_q8(set, slot, k, &qs, &ds) != SUPERROC_FP4_OK) {
        return SUPERROC_FP4_ERR_ARG;
    }
    if (ls1_variant_gemv() && n <= SUPERROC_FP4_LS1_GEMV_WIDE_MAX) {
        superroc_fp4_k_gemv_wide_q8_rows<<<ls1_gemv_blocks(n, 8), 256, 0, stream>>>(
            qtype, (const uint8_t *) W, rb, qs, ds, y, k, n, 1 << 20);
        return ls1_chk("gemv rows sel");
    }
    const int launched = ls1_launch_rows_q8(qtype, (const uint8_t *) W, rb, qs, ds, y, k, n, 1, stream);
    if (launched != SUPERROC_FP4_OK) return launched;
    return ls1_chk("gemv q8");
}

// Matvec that quantises the activation itself. One extra tiny kernel per call;
// the caller gets the right answer with no cooperation.
extern "C" int superroc_fp4_gemv_fp4(int qtype, const void * W, const float * x,
                              float * y, int k, int n, hipStream_t stream) {
    const int rc = superroc_fp4_quant_q8(x, k, SUPERROC_FP4_LS1_SLOT_GENERIC, stream);
    if (rc != SUPERROC_FP4_OK) {
        return rc;
    }
    return superroc_fp4_gemv_fp4_q8(qtype, W, SUPERROC_FP4_LS1_SLOT_GENERIC, y, k, n, stream);
}

// Copy a slot's quantised activation back to the host. Test-only: it exists so
// tools/superroc_fp4_gemv_q8_test.cpp can tell a wrong quantisation from a wrong dot
// without having to guess which one it is looking at.
extern "C" int superroc_fp4_q8_peek_set(int set, int slot, int8_t * qs_out, float * ds_out, int k) {
    int8_t * qs = nullptr;
    float * ds = nullptr;
    {
        SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        if (set < 0 || (size_t) set >= mgr.sets.size()) return SUPERROC_FP4_ERR_ARG;
        SuperROCFp4SlotSet & S = mgr.sets[(size_t) set];
        if (slot < 0 || slot >= SUPERROC_FP4_LS1_NSLOT || k <= 0 || S.qs[slot] == nullptr ||
            S.cap[slot] < k) {
            return SUPERROC_FP4_ERR_ARG;
        }
        qs = S.qs[slot];
        ds = S.ds[slot];
    }
    hipError_t e = hipMemcpy(qs_out, qs, (size_t) k, hipMemcpyDeviceToHost);
    if (e != hipSuccess) return SUPERROC_FP4_ERR_HIP;
    e = hipMemcpy(ds_out, ds, (size_t) (k / 32) * sizeof(float), hipMemcpyDeviceToHost);
    return e == hipSuccess ? SUPERROC_FP4_OK : SUPERROC_FP4_ERR_HIP;
}

extern "C" int superroc_fp4_q8_peek(int slot, int8_t * qs_out, float * ds_out, int k) {
    return superroc_fp4_q8_peek_set(0, slot, qs_out, ds_out, k);
}

// ---------------------------------------------------------------------------
// Exhaustively check that the perm codebook above really is
// superroc_fp4_rocmfp4_code, for all 65536 four-nibble words. Returns the number of
// byte lanes that disagree; 0 means the table is a faithful transcription. This
// is the one piece of the integer path that cannot be checked by comparing
// against a float reference, because a wrong code permutes the products instead
// of scaling them, and a float reference would just look like quantisation noise.
// (It is also the check that caught a 2x error in an earlier revision of this
// file, which a float comparison would have read as ordinary quantisation noise.)
// ---------------------------------------------------------------------------
__global__ void superroc_fp4_k_cb_selftest(int * bad) {
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
        if ((int) o[i] != superroc_fp4_rocmfp4_code(qb[i] & 0x0fu)) local++;
        if ((int) o[4 + i] != superroc_fp4_rocmfp4_code(qb[i] >> 4)) local++;
    }
    if (local != 0) {
        atomicAdd(bad, local);
    }
}

extern "C" int superroc_fp4_gemv_q8_codebook_check(void) {
    int * d_bad = nullptr;
    if (hipMalloc(&d_bad, sizeof(int)) != hipSuccess) {
        return -1;
    }
    hipMemset(d_bad, 0, sizeof(int));
    superroc_fp4_k_cb_selftest<<<65536 / 256, 256, 0, (hipStream_t) nullptr>>>(d_bad);
    int bad = -1;
    if (hipDeviceSynchronize() != hipSuccess || hipMemcpy(&bad, d_bad, sizeof(int),
                                                         hipMemcpyDeviceToHost) != hipSuccess) {
        bad = -1;
    }
    hipFree(d_bad);
    return bad;
}
// ---------------------------------------------------------------------------
// Grouped, device-routed expert FFN.
//
// The per-expert host loop this replaces did, per MoE layer, two blocking D2H copies of
// the router's 8 ids and 8 weights -- 40 GPU syncs per token -- and then roughly 56 kernel
// launches, ~2,240 of them per token across the 40 MoE layers, each a full round trip
// through the scheduler for about 1.3 MB of work.
//
// This is three launches per layer and zero syncs:
//
//   1. superroc_fp4_k_moe_gate_up_swiglu -- one CTA per (selected expert slot, output row). The CTA
//      reads its slot's id from DEVICE memory, dots the gate row and the up row against
//      the ONE quantisation of the FFN input, and fuses SwiGLU into its epilogue.
//   2. superroc_fp4_k_q8_1_quant over hs -- the down projection needs Q8_1, and a 32-value group's
//      scale spans 32 different CTAs of step 1, so it cannot be formed there.
//   3. superroc_fp4_k_moe_down_weighted -- one CTA per output row, looping the n_used slots IN SLOT
//      ORDER and accumulating in a register: the weighted sum is in a fixed order every
//      run, with no float atomics and no dependence on scheduling.
//
// One quantisation of the FFN input serves every expert and the weight bytes read are
// unchanged: 8 experts x (2 x 768 x 2560 + 2560 x 768) values per layer, the same 25 MB,
// now in three launches instead of ~56.
// ---------------------------------------------------------------------------
__global__ void superroc_fp4_k_moe_gate_up_swiglu(int qtype, const uint8_t * gate, const uint8_t * up,
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
        *rc = SUPERROC_FP4_ERR_ARG;
    }
    const size_t base = ok ? (((size_t) id * (size_t) n_ff + (size_t) j) * rb_k) : 0;

    const float g = ls1_block_sum(ls1_dot_row_q8(qtype, gate + base, xq, xd, k), red);
    const float u = ls1_block_sum(ls1_dot_row_q8(qtype, up + base, xq, xd, k), red);
    if (threadIdx.x == 0) {
        hs[(size_t) s * (size_t) n_ff + (size_t) j] = superroc_fp4_swiglu_elem(g, u, limit);
    }
}

__global__ void superroc_fp4_k_moe_down_weighted(int qtype, const uint8_t * down, size_t rb_ff,
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
            *rc = SUPERROC_FP4_ERR_ARG;
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

// moebatch: superroc_fp4_k_moe_gate_up_swiglu with the row folded into the flat grid, so all rows of a layer share one launch. Same dots, same
// reduction, so every output is bit-identical to the one-row kernel.
__global__ void superroc_fp4_k_moe_gate_up_swiglu_batch(int qtype, const uint8_t * gate, const uint8_t * up,
                                     size_t rb_k, const int * ids,
                                     const int8_t * xq, const float * xd,
                                     float * hs, float limit,
                                     int n_ff, int n_used, int n_expert, int k, int * rc) {
    const int per_row = n_used * n_ff;
    const int t = (int) blockIdx.x / per_row;       // which row
    const int r = (int) blockIdx.x - t * per_row;
    const int s = r / n_ff;                         // which selected expert
    const int j = r - s * n_ff;                     // output row inside that expert
    __shared__ float red[32];
    const int id = ids[t * n_used + s];
    const bool ok = (id >= 0 && id < n_expert);
    if (!ok && threadIdx.x == 0) {
        *rc = SUPERROC_FP4_ERR_ARG;
    }
    const size_t base = ok ? (((size_t) id * (size_t) n_ff + (size_t) j) * rb_k) : 0;
    const int8_t * xq_t = xq + (size_t) t * (size_t) k;
    const float * xd_t = xd + (size_t) t * (size_t) (k / 32);
    const float g = ls1_block_sum(ls1_dot_row_q8(qtype, gate + base, xq_t, xd_t, k), red);
    const float u = ls1_block_sum(ls1_dot_row_q8(qtype, up + base, xq_t, xd_t, k), red);
    if (threadIdx.x == 0) {
        hs[(size_t) t * (size_t) per_row + (size_t) r] = superroc_fp4_swiglu_elem(g, u, limit);
    }
}

// moebatch: superroc_fp4_k_moe_down_weighted_rows over rows * k outputs in one launch. One warp per output, slots in slot order; bit-identical.
__global__ void superroc_fp4_k_moe_down_weighted_rows_batch(int qtype, const uint8_t * down, size_t rb_ff,
                                         const int * ids, const float * weights,
                                         const int8_t * hq, const float * hd,
                                         float * y, int rows, int n_used, int n_expert,
                                         int k, int n_ff, int * rc) {
    const int lane = threadIdx.x & 31;
    const int sub = (int) (threadIdx.x >> 5);
    const int nwarps = (int) blockDim.x >> 5;
    const int nb = n_ff / 32;
    const int jt = (int) blockIdx.x * nwarps + sub;
    if (jt >= rows * k) {
        return;
    }
    const int t = jt / k;
    const int j = jt - t * k;
    const int * ids_t = ids + (size_t) t * (size_t) n_used;
    const float * weights_t = weights + (size_t) t * (size_t) n_used;
    const size_t h0 = (size_t) t * (size_t) n_used;
    float acc = 0.f;
    for (int s = 0; s < n_used; ++s) {
        const int id = ids_t[s];
        const bool ok = (id >= 0 && id < n_expert);
        if (!ok && lane == 0) {
            *rc = SUPERROC_FP4_ERR_ARG;
        }
        const size_t base = ok ? (((size_t) id * (size_t) k + (size_t) j) * rb_ff) : 0;
        const uint8_t * drow = down + base;
        const size_t hs_slot = h0 + (size_t) s;
        float part = 0.f;
        for (int qb = lane; qb < nb; qb += 32) {
            part += ls1_dot_fp4_blk32_q8(qtype, drow + (size_t) qb * (size_t) superroc_fp4_type_size(qtype),
                                        hq + hs_slot * (size_t) n_ff + (size_t) qb * 32,
                                        hd[hs_slot * (size_t) nb + (size_t) qb]);
        }
        part = ls1_warp_sum(part);
        if (lane == 0) {
            acc += weights_t[s] * part;
        }
    }
    if (lane == 0) {
        y[jt] = acc;
    }
}

// one record per distinct routed expert of a verify round, first-seen order:
// id | cnt << 16 | f0 << 24 | f1 << 32 | f2 << 40 | f3 << 48, with f = row * n_used + slot. rec[0] holds the record count.
__global__ void superroc_fp4_k_moe_group_small(const int * ids, int n_slots, int n_expert,
                                           unsigned long long * rec, int * rc) {
    __shared__ int s_ids[64];
    __shared__ unsigned long long s_rec[65];
    const int tid = (int) threadIdx.x;
    if (tid < n_slots) {
        s_ids[tid] = ids[tid];
    }
    __syncthreads();
    if (tid == 0) {
        int n = 0;
        for (int f = 0; f < n_slots; ++f) {
            int id = s_ids[f];
            if (id < 0 || id >= n_expert) {
                *rc = SUPERROC_FP4_ERR_ARG;
                id = 0;
            }
            // A record holds 4 fan-out slots; only a record with room takes a
            // fifth occurrence, so the overflow drops no slot. A round is at
            // most 8 rows of 8 used experts, so f fits its byte, and rec is
            // sized 65 for 64 slots, so the record count cannot overflow.
            int g = -1;
            for (int i = 0; i < n; ++i) {
                const unsigned long long ri = s_rec[i + 1];
                if ((int) (ri & 0xffffULL) == id &&
                    (int) ((ri >> 16) & 0xffULL) < 4) {
                    g = i;
                    break;
                }
            }
            if (g < 0) {
                g = n++;
                s_rec[g + 1] = (unsigned long long) id;
            }
            const unsigned long long r = s_rec[g + 1];
            const int c = (int) ((r >> 16) & 0xffULL);
            s_rec[g + 1] = (r & ~(0xffULL << 16)) | ((unsigned long long) (c + 1) << 16) |
                           ((unsigned long long) f << (24 + 8 * c));
        }
        s_rec[0] = (unsigned long long) n;
    }
    __syncthreads();
    if (tid < 65) {
        rec[tid] = s_rec[tid];
    }
}

// gate + up + SwiGLU for every routed slot and the shared expert in one launch, each weight row read once. Per output the thread-level dot order, the warp
// sum and the (0.f + r0) + r1 sum match superroc_fp4_k_moe_gate_up_swiglu, so every value is bit-identical. hs layout: [rows * n_used][n_ff], then [rows][n_ff_s].
__global__ void superroc_fp4_k_moe_gate_up_all(int qtype, const uint8_t * gate, const uint8_t * up, size_t rb_k,
                                           const unsigned long long * recs,
                                           const uint8_t * gate_s, const uint8_t * up_s,
                                           const int8_t * xq, const float * xd, float * hs, float limit,
                                           int rows, int n_used, int n_ff, int n_ff_s, int k) {
    const int nr = rows * n_used * n_ff;
    const int b = (int) blockIdx.x;
    const int tid = (int) threadIdx.x;
    const int nw = (int) blockDim.x >> 5;
    const int tsz = superroc_fp4_type_size(qtype);
    const int nblk = k / 32;
    __shared__ float red[32];
    const uint8_t * grow;
    const uint8_t * urow;
    int cnt;
    unsigned long long rec = 0;
    int j;
    int part = 0;
    if (b < nr) {
        const int gi = b / n_ff;
        if (gi >= (int) recs[0]) {
            return;
        }
        j = b - gi * n_ff;
        rec = recs[gi + 1];
        const size_t base = ((size_t) (rec & 0xffffULL) * (size_t) n_ff + (size_t) j) * rb_k;
        grow = gate + base;
        urow = up + base;
        cnt = (int) ((rec >> 16) & 0xffULL);
    } else {
        // the shared expert runs in parts of 4 rows, so the per-thread accumulators stay 4 wide
        const int sb = b - nr;
        part = sb / n_ff_s;
        j = sb - part * n_ff_s;
        const size_t base = (size_t) j * rb_k;
        grow = gate_s + base;
        urow = up_s + base;
        cnt = min(rows - part * 4, 4);
    }
    float sg[4] = {0.f, 0.f, 0.f, 0.f};
    float su[4] = {0.f, 0.f, 0.f, 0.f};
    for (int qb = tid; qb < nblk; qb += (int) blockDim.x) {
        const ls1_blk32 cg = ls1_load_blk32(qtype, grow + (size_t) qb * (size_t) tsz);
        const ls1_blk32 cu = ls1_load_blk32(qtype, urow + (size_t) qb * (size_t) tsz);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m < cnt) {
                const int t = b < nr ? (int) ((rec >> (24 + 8 * m)) & 0xffULL) / n_used : part * 4 + m;
                const int8_t * xqt = xq + (size_t) t * (size_t) k + (size_t) qb * 32;
                const float xdt = xd[(size_t) t * (size_t) nblk + (size_t) qb];
                sg[m] += ls1_dot_blk32(cg, xqt, xdt);
                su[m] += ls1_dot_blk32(cu, xqt, xdt);
            }
        }
    }
    const int warp = tid >> 5;
    const bool lane0 = (tid & 31) == 0;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        if (m < cnt) {
            const float wg = ls1_warp_sum(sg[m]);
            const float wu = ls1_warp_sum(su[m]);
            if (lane0) {
                red[m * 2 * nw + warp] = wg;
                red[(m * 2 + 1) * nw + warp] = wu;
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        for (int m = 0; m < cnt; ++m) {
            float g = 0.f;
            float u = 0.f;
            for (int w = 0; w < nw; ++w) {
                g += red[m * 2 * nw + w];
            }
            for (int w = 0; w < nw; ++w) {
                u += red[(m * 2 + 1) * nw + w];
            }
            const size_t out = b < nr
                ? (size_t) ((rec >> (24 + 8 * m)) & 0xffULL) * (size_t) n_ff + (size_t) j
                : (size_t) nr + (size_t) (part * 4 + m) * (size_t) n_ff_s + (size_t) j;
            hs[out] = superroc_fp4_swiglu_elem(g, u, limit);
        }
    }
}

// down for every routed slot and the shared expert, the weighted sum in slot order and the combine, one warp per output row of all verify rows. Each weight row
// is read once. out[t * k + j] is bit-identical to moe_y + shexp_y of the per-row path.
__global__ void superroc_fp4_k_moe_down_all(int qtype, const uint8_t * down, size_t rb_ff,
                                        const unsigned long long * recs, const float * weights,
                                        const uint8_t * down_s, size_t rb_ff_s, const float * sig,
                                        const int8_t * hq, const float * hd, float * out,
                                        int rows, int n_used, int k, int n_ff, int n_ff_s) {
    const int lane = threadIdx.x & 31;
    const int sub = (int) (threadIdx.x >> 5);
    const int nwarps = (int) blockDim.x >> 5;
    const int j = (int) blockIdx.x * nwarps + sub;
    if (j >= k) {
        return;
    }
    __shared__ float s_parts[64 * 32];
    const int tsz = superroc_fp4_type_size(qtype);
    const int nb = n_ff / 32;
    const int nb_s = n_ff_s / 32;
    const int n_slots = rows * n_used;
    const int ngroups = (int) recs[0];
    for (int g = 0; g < ngroups; ++g) {
        const unsigned long long rec = recs[g + 1];
        const int cnt = (int) ((rec >> 16) & 0xffULL);
        const uint8_t * drow = down + ((size_t) (rec & 0xffffULL) * (size_t) k + (size_t) j) * rb_ff;
        float p[4] = {0.f, 0.f, 0.f, 0.f};
        for (int qb = lane; qb < nb; qb += 32) {
            const ls1_blk32 c = ls1_load_blk32(qtype, drow + (size_t) qb * (size_t) tsz);
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (m < cnt) {
                    const size_t f = (size_t) ((rec >> (24 + 8 * m)) & 0xffULL);
                    p[m] += ls1_dot_blk32(c, hq + f * (size_t) n_ff + (size_t) qb * 32,
                                          hd[f * (size_t) nb + (size_t) qb]);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m < cnt) {
                const float w = ls1_warp_sum(p[m]);
                if (lane == 0) {
                    s_parts[((int) ((rec >> (24 + 8 * m)) & 0xffULL)) * 32 + sub] = w;
                }
            }
        }
    }
    const size_t nr = (size_t) n_slots * (size_t) n_ff;
    const size_t nbr = (size_t) n_slots * (size_t) nb;
    const uint8_t * srow = down_s + (size_t) j * rb_ff_s;
    // the shared accumulator is one array of two 4-row parts (rows <= 8): a part loop that
    // folded each part into out fused the re-add into an fma and rounded differently, so the
    // combine below stays the per-row one, verbatim.
    float ps[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int qb = lane; qb < nb_s; qb += 32) {
        const ls1_blk32 c = ls1_load_blk32(qtype, srow + (size_t) qb * (size_t) tsz);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < rows) {
                ps[t] += ls1_dot_blk32(c, hq + nr + (size_t) t * (size_t) n_ff_s + (size_t) qb * 32,
                                       hd[nbr + (size_t) t * (size_t) nb_s + (size_t) qb]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (t < rows) {
            ps[t] = ls1_warp_sum(ps[t]);
        }
    }
    if (lane == 0) {
        for (int t = 0; t < rows; ++t) {
            float acc = 0.f;
            for (int s = 0; s < n_used; ++s) {
                const int f = t * n_used + s;
                acc += weights[f] * s_parts[f * 32 + sub];
            }
            float acc_s = 0.f;
            acc_s += sig[t] * ps[t];
            out[(size_t) t * (size_t) k + (size_t) j] = acc + acc_s;
        }
    }
}

// The hs scratch and its status word live in the slot sets above (one per
// stream). The router's own status word stays a single registered pointer: it
// is owned by superroc_fp4_route, which cannot free it, so that ONE read covers
// the whole decode token. See superroc_fp4_moe_dev_check.
static int * ls1_route_rc_d = nullptr;

// Device-routed grouped expert FFN for ONE token. `ids` and `weights` are DEVICE pointers
// (what superroc_fp4_route wrote); nothing is read back to the host and NOTHING SYNCS. The status
// word is left on the device for superroc_fp4_moe_dev_check() to read once per token -- reading it
// here would put back exactly the per-layer drain this path exists to remove.
// ROCmFP4 (100/101) and a single token; anything else returns SUPERROC_FP4_ERR_ARG so the caller
// can fall back to superroc_fp4_expert_ffn.
// quantize_x == false reuses the FFN_X slot's Q8 copy of x (see the _xq entry point).
static int expert_ffn_dev_impl(
        int qtype, const void * gate, const void * up, const void * down,
        const float * x, const int * ids_d, const float * weights_d,
        int n_used, int n_expert, int k, int n_ff, float limit, float * y,
        hipStream_t stream, bool quantize_x) {
    const size_t rb_k = superroc_fp4_row_bytes(qtype, k);
    const size_t rb_ff = superroc_fp4_row_bytes(qtype, n_ff);
    if (!gate || !up || !down || !x || !ids_d || !weights_d || !y) return SUPERROC_FP4_ERR_ARG;
    if (qtype != SUPERROC_FP4_Q4_0_ROCMFP4 && qtype != SUPERROC_FP4_Q4_0_ROCMFP4_FAST) return SUPERROC_FP4_ERR_ARG;
    if (rb_k == 0 || rb_ff == 0 || n_used <= 0 || n_expert <= 0 || k <= 0 || n_ff <= 0) {
        return SUPERROC_FP4_ERR_ARG;
    }
    if ((k % 32) != 0 || (n_ff % 32) != 0) return SUPERROC_FP4_ERR_ARG;

    const int set = superroc_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // Refused: the caller declines, never waits.
    const size_t nn = (size_t) n_used * (size_t) n_ff;
    float * moe_hs = nullptr;
    int * moe_rc = nullptr;
    int rc = superroc_fp4_slots_grow_hs(set, nn, stream, &moe_hs, &moe_rc);
    if (rc != SUPERROC_FP4_OK) return rc;

    if (quantize_x) {
        rc = superroc_fp4_quant_q8(x, k, SUPERROC_FP4_LS1_SLOT_FFN_X, stream);
        if (rc != SUPERROC_FP4_OK) return rc;
    }
    int8_t * xq = nullptr;
    float * xd = nullptr;
    if (superroc_fp4_slots_get_q8(set, SUPERROC_FP4_LS1_SLOT_FFN_X, k, &xq, &xd) != SUPERROC_FP4_OK) {
        return SUPERROC_FP4_ERR_HIP;
    }

    // One CTA per (slot, row): the kernel takes its slot from blockIdx / n_ff, so the
    // grid is exactly n_ff * n_used (n_ff need not be a multiple of 64).
    superroc_fp4_k_moe_gate_up_swiglu<<<(unsigned) n_ff * (unsigned) n_used, 64, 0, stream>>>(
        qtype, (const uint8_t *) gate, (const uint8_t *) up, rb_k,
        ids_d, xq, xd, moe_hs, limit, n_ff, n_expert, k, moe_rc);
    if ((rc = ls1_chk("moe gate/up")) != SUPERROC_FP4_OK) return rc;

    rc = superroc_fp4_quant_q8(moe_hs, (int) nn, SUPERROC_FP4_LS1_SLOT_MOE_H, stream);
    if (rc != SUPERROC_FP4_OK) return rc;
    int8_t * hq = nullptr;
    float * hd = nullptr;
    if (superroc_fp4_slots_get_q8(set, SUPERROC_FP4_LS1_SLOT_MOE_H, (int) nn, &hq, &hd) != SUPERROC_FP4_OK) {
        return SUPERROC_FP4_ERR_HIP;
    }

    // The down half defaults ON. It is MEASURED bit-identical
    // (max_rel_diff 0, 0 differing of 2560, all three configurations), so it carries no
    // numerical risk, and it is 1.22x faster on the isolated A/B.
    // (An earlier revision of this file had lost the branch here entirely, which is why
    // the first phase-7 "variant ON" numbers turned out to have exercised ONLY the
    // plain-GEMV half.)
    if (ls1_variant_down()) {
        const int nwarps = 8;
        superroc_fp4_k_moe_down_weighted_rows<<<ls1_gemv_blocks(k, nwarps), nwarps * 32, 0, stream>>>(
            qtype, (const uint8_t *) down, rb_ff, ids_d, weights_d,
            hq, hd, y, n_used, n_expert, k, n_ff,
            1 << 20, moe_rc);
        return ls1_chk("moe down rows");
    }
    superroc_fp4_k_moe_down_weighted<<<(unsigned) k, 64, 0, stream>>>(qtype, (const uint8_t *) down, rb_ff,
                                              ids_d, weights_d,
                                              hq, hd,
                                              y, n_used, n_expert, k, n_ff, moe_rc);
    return ls1_chk("moe down");
}

extern "C" int superroc_fp4_expert_ffn_dev(
        int qtype, const void * gate, const void * up, const void * down,
        const float * x, const int * ids_d, const float * weights_d,
        int n_used, int n_expert, int k, int n_ff, float limit, float * y,
        hipStream_t stream) {
    return expert_ffn_dev_impl(qtype, gate, up, down, x, ids_d, weights_d, n_used, n_expert,
                               k, n_ff, limit, y, stream, true);
}

extern "C" int superroc_fp4_expert_ffn_dev_xq(
        int qtype, const void * gate, const void * up, const void * down,
        const float * x, const int * ids_d, const float * weights_d,
        int n_used, int n_expert, int k, int n_ff, float limit, float * y,
        hipStream_t stream) {
    return expert_ffn_dev_impl(qtype, gate, up, down, x, ids_d, weights_d, n_used, n_expert,
                               k, n_ff, limit, y, stream, false);
}

// moebatch: buffers for superroc_fp4_expert_ffn_dev_batch only, one group per slot set. Separate from the Q8 slots on purpose: a captured decode graph holds
// those slots' pointers, and growing one would free memory the graph still reads. Kept for the life of the process.
struct MoeBatchBuffers {
    int8_t * xq = nullptr; float * xd = nullptr; int x_cap = 0;
    int8_t * hq = nullptr; float * hd = nullptr; int h_cap = 0;
    float * hs = nullptr; size_t hs_cap = 0;
    unsigned long long * rec = nullptr;
};
static std::mutex g_moe_batch_mu;
static std::map<int, MoeBatchBuffers> g_moe_batch;

static int moe_batch_grow_q8(int8_t ** q, float ** d, int * cap, int total, hipStream_t stream) {
    if (total <= *cap && *q != nullptr && *d != nullptr) return SUPERROC_FP4_OK;
    if (hipStreamSynchronize(stream) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
    if (*q != nullptr && hipFree(*q) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
    if (*d != nullptr && hipFree(*d) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
    *q = nullptr; *d = nullptr; *cap = 0;
    if (hipMalloc((void **) q, (size_t) total + SUPERROC_FP4_TAIL_PAD) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
    if (hipMalloc((void **) d, (size_t) (total / 32) * sizeof(float) + SUPERROC_FP4_TAIL_PAD) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
    *cap = total;
    return SUPERROC_FP4_OK;
}

// moebatch: every row of one MoE layer in one launch per stage. Bit-identical to
// expert_ffn_dev_impl called once per row: same dots, same 64-thread reduction, same slot
// order. Refused with SUPERROC_FP4_ERR_ARG (so the caller falls back to its per-row loop) when
// the geometry is not one this kernel covers, or the set has no moe_rc to report a bad id into.
extern "C" int superroc_fp4_expert_ffn_dev_batch(
        int qtype, const void * gate, const void * up, const void * down,
        const float * x, const int * ids_d, const float * weights_d,
        int rows, int n_used, int n_expert, int k, int n_ff, float limit, float * y,
        hipStream_t stream, int quantize_x) {
    const size_t rb_k = superroc_fp4_row_bytes(qtype, k);
    const size_t rb_ff = superroc_fp4_row_bytes(qtype, n_ff);
    if (!gate || !up || !down || !x || !ids_d || !weights_d || !y) return SUPERROC_FP4_ERR_ARG;
    if (qtype != SUPERROC_FP4_Q4_0_ROCMFP4 && qtype != SUPERROC_FP4_Q4_0_ROCMFP4_FAST) return SUPERROC_FP4_ERR_ARG;
    if (rb_k == 0 || rb_ff == 0 || rows < 2 || n_used <= 0 || n_expert <= 0 || k <= 0 || n_ff <= 0) return SUPERROC_FP4_ERR_ARG;
    if ((k % 32) != 0 || (n_ff % 64) != 0 || !ls1_variant_down()) return SUPERROC_FP4_ERR_ARG;
    const int set = superroc_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;
    int * moe_rc = nullptr;
    {
        SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        moe_rc = mgr.sets[(size_t) set].moe_rc;
    }
    if (moe_rc == nullptr) return SUPERROC_FP4_ERR_ARG;
    const size_t nn = (size_t) rows * (size_t) n_used * (size_t) n_ff;
    const int total_k = rows * k;
    int8_t * xq = nullptr; float * xd = nullptr; int8_t * hq = nullptr; float * hd = nullptr; float * hs = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_moe_batch_mu);
        MoeBatchBuffers & B = g_moe_batch[set];
        if (quantize_x) {
            int rc = moe_batch_grow_q8(&B.xq, &B.xd, &B.x_cap, total_k, stream);
            if (rc != SUPERROC_FP4_OK) return rc;
        } else if (B.xq == nullptr || B.x_cap < total_k) {
            return SUPERROC_FP4_ERR_ARG;
        }
        int rc = moe_batch_grow_q8(&B.hq, &B.hd, &B.h_cap, (int) nn, stream);
        if (rc != SUPERROC_FP4_OK) return rc;
        if (nn > B.hs_cap || B.hs == nullptr) {
            if (hipStreamSynchronize(stream) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
            if (B.hs != nullptr && hipFree(B.hs) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
            B.hs = nullptr; B.hs_cap = 0;
            if (hipMalloc((void **) &B.hs, nn * sizeof(float)) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
            B.hs_cap = nn;
        }
        xq = B.xq; xd = B.xd; hq = B.hq; hd = B.hd; hs = B.hs;
    }
    const int threads = 256;
    if (quantize_x) {
        const int nb = total_k / 32;
        superroc_fp4_k_q8_1_quant<<<(nb + threads - 1) / threads, threads, 0, stream>>>(x, xq, xd, nb);
        const int qrc = ls1_chk("moe batch quant x");
        if (qrc != SUPERROC_FP4_OK) return qrc;
    }
    superroc_fp4_k_moe_gate_up_swiglu_batch<<<(unsigned) nn, 64, 0, stream>>>(
        qtype, (const uint8_t *) gate, (const uint8_t *) up, rb_k,
        ids_d, xq, xd, hs, limit, n_ff, n_used, n_expert, k, moe_rc);
    int rc = ls1_chk("moe gate/up batch");
    if (rc != SUPERROC_FP4_OK) return rc;
    const int nbh = (int) (nn / 32);
    superroc_fp4_k_q8_1_quant<<<(nbh + threads - 1) / threads, threads, 0, stream>>>(hs, hq, hd, nbh);
    rc = ls1_chk("moe batch quant h");
    if (rc != SUPERROC_FP4_OK) return rc;
    const int nwarps = 8;
    superroc_fp4_k_moe_down_weighted_rows_batch<<<ls1_gemv_blocks(total_k, nwarps), nwarps * 32, 0, stream>>>(
        qtype, (const uint8_t *) down, rb_ff, ids_d, weights_d,
        hq, hd, y, rows, n_used, n_expert, k, n_ff, moe_rc);
    return ls1_chk("moe down rows batch");
}

// the routed experts and the shared expert of a verify round in one launch per stage, and the
// shared combine folded into the down launch. Refused with SUPERROC_FP4_ERR_ARG (the caller falls back to
// its per-row loop) unless the geometry is one this covers and the set has a moe_rc to report a bad id
// into. Bit-identical to superroc_fp4_expert_ffn_dev + superroc_fp4_expert_ffn_dev_xq + the combine.
extern "C" int superroc_fp4_expert_ffn_dev_batch_all(
        int qtype, const void * gate, const void * up, const void * down,
        const void * gate_s, const void * up_s, const void * down_s,
        const float * x, const int * ids_d, const float * weights_d, const float * sig_d,
        int rows, int n_used, int n_expert, int k, int n_ff, int n_ff_s, float limit, float * out,
        hipStream_t stream) {
    const size_t rb_k = superroc_fp4_row_bytes(qtype, k);
    const size_t rb_ff = superroc_fp4_row_bytes(qtype, n_ff);
    const size_t rb_ff_s = superroc_fp4_row_bytes(qtype, n_ff_s);
    if (!gate || !up || !down || !gate_s || !up_s || !down_s || !x || !ids_d || !weights_d || !sig_d || !out) return SUPERROC_FP4_ERR_ARG;
    if (qtype != SUPERROC_FP4_Q4_0_ROCMFP4 && qtype != SUPERROC_FP4_Q4_0_ROCMFP4_FAST) return SUPERROC_FP4_ERR_ARG;
    if (rb_k == 0 || rb_ff == 0 || rb_ff_s == 0 || rows < 2 || rows > 8 || n_used <= 0 || n_used > 8 || n_expert <= 0 || k <= 0 || n_ff <= 0 || n_ff_s <= 0) return SUPERROC_FP4_ERR_ARG;
    if ((k % 32) != 0 || (n_ff % 64) != 0 || (n_ff_s % 64) != 0 || !ls1_variant_down()) return SUPERROC_FP4_ERR_ARG;
    const int set = superroc_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;
    int * moe_rc = nullptr;
    {
        SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        moe_rc = mgr.sets[(size_t) set].moe_rc;
    }
    if (moe_rc == nullptr) return SUPERROC_FP4_ERR_ARG;
    const int n_slots = rows * n_used;
    const size_t nr = (size_t) n_slots * (size_t) n_ff;
    const size_t nn = nr + (size_t) rows * (size_t) n_ff_s;
    const int total_k = rows * k;
    int8_t * xq = nullptr; float * xd = nullptr; int8_t * hq = nullptr; float * hd = nullptr; float * hs = nullptr; unsigned long long * rec = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_moe_batch_mu);
        MoeBatchBuffers & B = g_moe_batch[set];
        int rc = moe_batch_grow_q8(&B.xq, &B.xd, &B.x_cap, total_k, stream);
        if (rc != SUPERROC_FP4_OK) return rc;
        rc = moe_batch_grow_q8(&B.hq, &B.hd, &B.h_cap, (int) nn, stream);
        if (rc != SUPERROC_FP4_OK) return rc;
        if (nn > B.hs_cap || B.hs == nullptr) {
            if (hipStreamSynchronize(stream) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
            if (B.hs != nullptr && hipFree(B.hs) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
            B.hs = nullptr; B.hs_cap = 0;
            if (hipMalloc((void **) &B.hs, nn * sizeof(float)) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
            B.hs_cap = nn;
        }
        if (B.rec == nullptr && hipMalloc((void **) &B.rec, 129 * sizeof(unsigned long long)) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
        xq = B.xq; xd = B.xd; hq = B.hq; hd = B.hd; hs = B.hs; rec = B.rec;
    }
    const int threads = 256;
    const int nbx = total_k / 32;
    superroc_fp4_k_q8_1_quant<<<(nbx + threads - 1) / threads, threads, 0, stream>>>(x, xq, xd, nbx);
    int rc = ls1_chk("moe all quant x");
    if (rc != SUPERROC_FP4_OK) return rc;
    superroc_fp4_k_moe_group_small<<<1, 128, 0, stream>>>(ids_d, n_slots, n_expert, rec, moe_rc);
    rc = ls1_chk("moe all group");
    if (rc != SUPERROC_FP4_OK) return rc;
    const int shared_parts = (rows + 3) / 4;
    superroc_fp4_k_moe_gate_up_all<<<(unsigned) (nr + (size_t) n_ff_s * (size_t) shared_parts), 64, 0, stream>>>(
        qtype, (const uint8_t *) gate, (const uint8_t *) up, rb_k, rec,
        (const uint8_t *) gate_s, (const uint8_t *) up_s, xq, xd, hs, limit,
        rows, n_used, n_ff, n_ff_s, k);
    rc = ls1_chk("moe all gate/up");
    if (rc != SUPERROC_FP4_OK) return rc;
    const int nbh = (int) (nn / 32);
    superroc_fp4_k_q8_1_quant<<<(nbh + threads - 1) / threads, threads, 0, stream>>>(hs, hq, hd, nbh);
    rc = ls1_chk("moe all quant h");
    if (rc != SUPERROC_FP4_OK) return rc;
    const int nwarps = 8;
    superroc_fp4_k_moe_down_all<<<ls1_gemv_blocks(k, nwarps), nwarps * 32, 0, stream>>>(
        qtype, (const uint8_t *) down, rb_ff, rec, weights_d,
        (const uint8_t *) down_s, rb_ff_s, sig_d, hq, hd, out, rows, n_used, k, n_ff, n_ff_s);
    return ls1_chk("moe all down");
}

// Called by superroc_fp4_route once, when it allocates its status word, so that the per-token
// check below can read it without superroc_fp4_route reading it on every one of the 40 layers.
extern "C" void superroc_fp4_moe_dev_register_route_rc(int * rc_d) {
    ls1_route_rc_d = rc_d;
}

// Read the decode step's device status words -- the grouped FFN's and the router's -- and
// clear them. Call ONCE per token, after the last MoE layer, never per layer: this is the
// only blocking read in the decode path, and 40 of them is exactly what phase 3 removed.
//
// Clearing here rather than at the start of the token is deliberate: a status word that is
// only ever written on error would otherwise carry one token's failure into every token
// after it, and clearing costs nothing extra because the read is already blocking.
extern "C" int superroc_fp4_moe_dev_check_set(int set);
extern "C" int superroc_fp4_moe_dev_check_set_on_stream(int set, hipStream_t stream) {
    if (stream == nullptr) return SUPERROC_FP4_ERR_ARG;
    if (hipStreamSynchronize(stream) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
    return superroc_fp4_moe_dev_check_set(set);
}

extern "C" int superroc_fp4_moe_dev_check_set(int set) {
    int * rc_d = nullptr;
    {
        SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        if (set < 0 || (size_t) set >= mgr.sets.size()) return SUPERROC_FP4_ERR_ARG;
        rc_d = mgr.sets[(size_t) set].moe_rc;
    }
    int rc = SUPERROC_FP4_OK;
    int * src[2] = {rc_d, ls1_route_rc_d};
    for (int i = 0; i < 2; ++i) {
        if (src[i] == nullptr) {
            continue;
        }
        int word = SUPERROC_FP4_OK;
        if (hipMemcpy(&word, src[i], sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
            return SUPERROC_FP4_ERR_HIP;
        }
        if (word != SUPERROC_FP4_OK && rc == SUPERROC_FP4_OK) {
            rc = word;
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (src[i] != nullptr) {
            if (hipMemset(src[i], 0, sizeof(int)) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
        }
    }
    return rc;
}

extern "C" int superroc_fp4_moe_dev_check(void) {
    // Aggregate over every set, for callers that do not name one (Flash-Next,
    // prefill, tools). The qwen decode path names its set precisely.
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    int n = 0;
    {
        std::lock_guard<std::mutex> lk(mgr.mu);
        n = (int) mgr.sets.size();
    }
    int rc = SUPERROC_FP4_OK;
    for (int s = 0; s < n; ++s) {
        const int r = superroc_fp4_moe_dev_check_set(s);
        if (r != SUPERROC_FP4_OK && rc == SUPERROC_FP4_OK) {
            rc = r;
        }
    }
    return rc;
}

// Test-only: copy the grouped FFN's hs scratch back to the host.
extern "C" int superroc_fp4_moe_hs_peek_set(int set, float * out, int n) {
    float * hs = nullptr;
    {
        SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        if (set < 0 || (size_t) set >= mgr.sets.size()) return -1;
        hs = mgr.sets[(size_t) set].moe_hs;
    }
    if (hs == nullptr || n <= 0) return -1;
    return hipMemcpy(out, hs, (size_t) n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
}

extern "C" int superroc_fp4_moe_hs_peek(float * out, int n) {
    return superroc_fp4_moe_hs_peek_set(0, out, n);
}

// One K-block dotted against M columns for one or two rows.
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

// The 64-thread walk of one pair of rows, played by one wave. At step c the wave holds
// qb = lane + 32c, and thread qb mod 64 is lane + 32(c & 1), so even and odd steps keep their own
// accumulators. Each lane still sees its blocks in ascending qb order.
template <int M>
__device__ __forceinline__ void ls1_wave_walk2(int qtype, const uint8_t * w0, const uint8_t * w1,
                                               bool has1, const int8_t * xq, const float * xd,
                                               int k, int nblk, int tsz, int lane, float * even0,
                                               float * even1, float * odd0, float * odd1) {
    const int nsteps = nblk >> 5;
    const int tail = nblk & 31;
    int c = 0;
    for (; c + 1 < nsteps; c += 2) {
        ls1_verify_dot2<M>(qtype, w0, w1, has1, xq, xd, k, nblk, tsz, lane + (c << 5), even0, even1);
        ls1_verify_dot2<M>(qtype, w0, w1, has1, xq, xd, k, nblk, tsz, lane + ((c + 1) << 5), odd0, odd1);
    }
    if (c < nsteps) {
        ls1_verify_dot2<M>(qtype, w0, w1, has1, xq, xd, k, nblk, tsz, lane + (c << 5), even0, even1);
        ++c;
    }
    // The blocks left over are the first ones of a step; c is uniform, tail too.
    if (tail != 0 && lane < tail) {
        if ((c & 1) == 0) {
            ls1_verify_dot2<M>(qtype, w0, w1, has1, xq, xd, k, nblk, tsz, lane + (c << 5), even0, even1);
        } else {
            ls1_verify_dot2<M>(qtype, w0, w1, has1, xq, xd, k, nblk, tsz, lane + (c << 5), odd0, odd1);
        }
    }
}

// ---------------------------------------------------------------------------
// The small-m decode forward, for MTP verification.
//
// A speculative verify runs m = 2..4 rows of the SAME weights against m different
// activations. The m = 1 path reads every weight byte once per row, so m rows cost m times
// the bandwidth. This form reads each weight row ONCE and spends the extra work on m sets
// of dot products: the four v_perm of the codebook and the four 32-bit loads of the block
// are done once per 32 weights and reused for every column. That is llama.cpp's
// `mul_mat_vec_q` with ncols_dst <= 8, which is the model the fork uses too.
//
//   superroc_fp4_k_gemv_verify_mt  -- one wave per pair of rows, m columns each, y is [m][n] token-major
//   superroc_fp4_k_moe_gu_m        -- the grouped expert FFN's gate+up+SwiGLU over (token x slot) pairs
//   superroc_fp4_k_moe_down_m      -- its down + weighted sum, m accumulators in one register set
//
// Both grouped kernels take the per-pair expert id from DEVICE memory and never sync, like
// their m = 1 counterparts. Nothing here is wired into Step; the multi-row verify calls it.
enum { SUPERROC_FP4_LS1_MAX_M = 8 };

// gate + up + SwiGLU for every (token, slot) pair. ids_d and weights_d are indexed by the
// flat slot s = token * n_used + slot, and are DEVICE pointers.
__global__ void superroc_fp4_k_moe_gu_m(int qtype, const uint8_t * gate, const uint8_t * up, size_t rb_k,
                           const int8_t * xq, const float * xd,
                           const int * ids_d, float * hs, float limit,
                           int n_used, int n_expert, int k, int n_ff, int n_slots,
                           int * rc, int group) {
    const int r = (int) blockIdx.x;
    const int s = r / n_ff;                  // flat (token, slot)
    const int j = r - s * n_ff;              // output row inside that expert
    const int t = s / n_used;                // which token
    __shared__ float red[32];

    const int id = ids_d[s];
    const bool ok = (id >= 0 && id < n_expert);
    if (!ok && threadIdx.x == 0) {
        *rc = SUPERROC_FP4_ERR_ARG;
    }
    const int tsz = superroc_fp4_type_size(qtype);
    const size_t base = ok ? ls1_grouped_row(id, n_ff, j, rb_k, tsz, group) : 0;
    const int8_t * xqt = xq + (size_t) t * (size_t) k;
    const float * xdt = xd + (size_t) t * (size_t) (k / 32);

    const float g = ls1_block_sum(ls1_dot_row_q8_stride(qtype, gate + base, xqt, xdt, k, tsz * group), red);
    const float u = ls1_block_sum(ls1_dot_row_q8_stride(qtype, up + base, xqt, xdt, k, tsz * group), red);
    if (threadIdx.x == 0) {
        hs[(size_t) s * (size_t) n_ff + (size_t) j] = superroc_fp4_swiglu_elem(g, u, limit);
    }
}

// down + the weighted sum over slots, one wave per pair of output rows. The slots stay in flat
// (token, slot) order and each token's sum is its own, so the order per token is fixed: the same
// determinism the m = 1 kernel has. hs[s] belongs to ONE token, s / n_used: adding its dot to all
// m sums was the first version of this kernel and every element came out wrong. A narrow n_ff
// leaves most of a 64-thread group idle (n_ff = 640 is 20 blocks), and each slot of each row
// reduced through two barriers, so this kernel ran at half the memory's rate; the two rows of a
// pair share their activation load and need neither.
__global__ void superroc_fp4_k_moe_down_m(int qtype, const uint8_t * down, size_t rb_ff,
                             const int * ids_d, const float * weights_d,
                             const int8_t * hq, const float * hd,
                             float * y, int n_used, int n_expert, int k, int n_ff,
                             int m, int * rc, int group) {
    const int lane = (int) threadIdx.x;
    const int tsz = superroc_fp4_type_size(qtype);
    const int nb = n_ff >> 5;
    const int npair = (k + 1) >> 1;
    for (int pair = (int) blockIdx.x; pair < npair; pair += (int) gridDim.x) {
        const int j0 = pair << 1;
        const int j1 = j0 + 1;
        const bool has1 = j1 < k;
        int s = 0;
        for (int t = 0; t < m; ++t) {
            float a0 = 0.f;
            float a1 = 0.f;
            for (int slot = 0; slot < n_used; ++slot, ++s) {
                const int id = ids_d[s];
                const bool ok = (id >= 0 && id < n_expert);
                if (!ok && lane == 0) {
                    *rc = SUPERROC_FP4_ERR_ARG;
                }
                const size_t base0 = ok ? ls1_grouped_row(id, k, j0, rb_ff, tsz, group) : 0;
                const size_t base1 = (has1 && ok) ? ls1_grouped_row(id, k, j1, rb_ff, tsz, group) : base0;
                float e0[1] = {0.f};
                float e1[1] = {0.f};
                float o0[1] = {0.f};
                float o1[1] = {0.f};
                ls1_wave_walk2<1>(qtype, down + base0, down + base1, has1,
                                  hq + (size_t) s * (size_t) n_ff, hd + (size_t) s * (size_t) nb,
                                  n_ff, nb, tsz * group, lane, e0, e1, o0, o1);
                const float p0 = ls1_row_sum(e0[0], o0[0], nb);
                const float p1 = has1 ? ls1_row_sum(e1[0], o1[0], nb) : 0.f;
                if (lane == 0) {
                    a0 += weights_d[s] * p0;
                    a1 += weights_d[s] * p1;
                }
            }
            if (lane == 0) {
                y[(size_t) t * (size_t) k + (size_t) j0] = a0;
                if (has1) {
                    y[(size_t) t * (size_t) k + (size_t) j1] = a1;
                }
            }
        }
    }
}

// One expert, m rows. Each gate/up/down row is decoded once and dotted
// against every token. The per-thread block order, ls1_block_sum, SwiGLU and
// `acc += weight * part` match superroc_fp4_k_moe_gu_m / superroc_fp4_k_moe_down_m at n_used = 1.
__global__ void superroc_fp4_k_moe_gu_one_m(int qtype, const uint8_t * gate, const uint8_t * up,
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
        *rc = SUPERROC_FP4_ERR_ARG;
    }
    const size_t base = ok ? (((size_t) id * (size_t) n_ff + (size_t) j) * rb_k) : 0;
    const int tsz = superroc_fp4_type_size(qtype);
    const int nblk = k / 32;
    const int nb32 = k / 32;
    const uint8_t * grow = gate + base;
    const uint8_t * urow = up + base;
    float sg[SUPERROC_FP4_LS1_MAX_M], su[SUPERROC_FP4_LS1_MAX_M];
    for (int t = 0; t < SUPERROC_FP4_LS1_MAX_M; ++t) {
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
            hs[(size_t) t * (size_t) n_ff + (size_t) j] = superroc_fp4_swiglu_elem(g, u, limit);
        }
    }
}

__global__ void superroc_fp4_k_moe_down_one_m(int qtype, const uint8_t * down, size_t rb_ff,
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
        *rc = SUPERROC_FP4_ERR_ARG;
    }
    const size_t base = ok ? (((size_t) id * (size_t) k + (size_t) j) * rb_ff) : 0;
    const int tsz = superroc_fp4_type_size(qtype);
    const int nblk = n_ff / 32;
    const uint8_t * drow = down + base;
    float part[SUPERROC_FP4_LS1_MAX_M];
    float acc[SUPERROC_FP4_LS1_MAX_M];
    for (int t = 0; t < SUPERROC_FP4_LS1_MAX_M; ++t) {
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

// The small-m pair and hs live in the slot sets (one per stream).
// Quantise n_tokens * k floats (token-major) into the calling stream's set,
// handing the pair back.
static int ls1_quant_m(const float * x, int total, hipStream_t stream,
                       int8_t ** qs, float ** ds) {
    if (x == nullptr || total <= 0 || (total % 32) != 0) return SUPERROC_FP4_ERR_ARG;
    const int set = superroc_fp4_slots_for_stream_at(stream, __func__, (long) total, false);
    if (set < 0) return set;  // Refused: the caller declines, never waits.
    const int rc = superroc_fp4_slots_grow_q8m(set, total, stream, qs, ds);
    if (rc != SUPERROC_FP4_OK) return rc;
    const int nb = total / 32;
    const int threads = 256;
    superroc_fp4_k_q8_1_quant<<<(nb + threads - 1) / threads, threads, 0, stream>>>(x, *qs, *ds, nb);
    return ls1_chk("q8 quant m");
}

// Matvec of n rows against m activation columns, activations already in Q8_1.
// y is [m][n] token-major: y[t * n + row].
extern "C" int superroc_fp4_gemv_q8_m(int qtype, const void * W, const int8_t * xq, const float * xd,
                               float * y, int k, int n, int m, hipStream_t stream) {
    size_t rb = superroc_fp4_row_bytes(qtype, k);
    if (!W || !xq || !xd || !y || rb == 0 || n <= 0) return SUPERROC_FP4_ERR_ARG;
    if (m < 1 || m > SUPERROC_FP4_LS1_MAX_M || (k % 32) != 0) return SUPERROC_FP4_ERR_ARG;
    if (qtype != SUPERROC_FP4_Q4_0_ROCMFP4 && qtype != SUPERROC_FP4_Q4_0_ROCMFP4_FAST) return SUPERROC_FP4_ERR_ARG;
    const int launched = ls1_launch_rows_q8(qtype, (const uint8_t *) W, rb, xq, xd, y, k, n, m, stream);
    if (launched != SUPERROC_FP4_OK) return launched;
    return ls1_chk("gemv q8 m");
}

// Quantise-only entry. See the header for the sharing rule.
extern "C" int superroc_fp4_quant_m(const float * x, int n_tokens, int k, hipStream_t stream,
                             const int8_t ** xq, const float ** xd) {
    if (xq == nullptr || xd == nullptr) return SUPERROC_FP4_ERR_ARG;
    if (n_tokens < 1 || n_tokens > SUPERROC_FP4_LS1_MAX_M || k <= 0 || (k % 32) != 0) return SUPERROC_FP4_ERR_ARG;
    int8_t * qs = nullptr;
    float * ds = nullptr;
    const int rc = ls1_quant_m(x, n_tokens * k, stream, &qs, &ds);
    if (rc != SUPERROC_FP4_OK) return rc;
    *xq = qs;
    *xd = ds;
    return SUPERROC_FP4_OK;
}

// Same, but quantising m * k floats from float first.
extern "C" int superroc_fp4_gemv_m(int qtype, const void * W, const float * x, float * y,
                            int k, int n, int m, hipStream_t stream) {
    int8_t * qs = nullptr;
    float * ds = nullptr;
    const int rc = ls1_quant_m(x, m * k, stream, &qs, &ds);
    if (rc != SUPERROC_FP4_OK) return rc;
    return superroc_fp4_gemv_q8_m(qtype, W, qs, ds, y, k, n, m, stream);
}

// Grouped expert FFN over n_tokens * n_used (token, slot) pairs, for one token batch.
// ids_d / weights_d are DEVICE pointers, flat and indexed s = token * n_used + slot.
// y is [n_tokens][k] token-major. No sync, no host round trip.
// Read each routed expert once per verify round: rows of a round share experts, so from four rows
// up this reads fewer weights than a slot-by-slot walk. One wave per record and pair of rows, no
// barrier in the walk. A record is id | cnt << 16 | f0 << 24 | f1 << 32 | f2 << 40 | f3 << 48 with
// f = row * n_used + slot; a fifth occurrence opens another record. rec[0] holds the count.
// Every value equals superroc_fp4_k_moe_gu_m and superroc_fp4_k_moe_down_m.
enum { SUPERROC_FP4_ONCE_MAX_SLOTS = 120, SUPERROC_FP4_ONCE_MAX_EXPERTS = 1024 };

// One record per distinct routed expert, first-seen order; s_open maps an expert to its record with room.
__global__ void superroc_fp4_k_moe_group_once(const int * ids, int n_slots, int n_expert, unsigned long long * rec, int * rc) {
    __shared__ int s_ids[SUPERROC_FP4_ONCE_MAX_SLOTS];
    __shared__ short s_open[SUPERROC_FP4_ONCE_MAX_EXPERTS];
    __shared__ unsigned long long s_rec[SUPERROC_FP4_ONCE_MAX_SLOTS + 1];
    const int tid = (int) threadIdx.x;
    if (tid < n_slots) {
        s_ids[tid] = ids[tid];
    }
    for (int i = tid; i < n_expert; i += 128) {
        s_open[i] = -1;
    }
    __syncthreads();
    if (tid == 0) {
        int n = 0;
        for (int f = 0; f < n_slots; ++f) {
            int id = s_ids[f];
            if (id < 0 || id >= n_expert) {
                *rc = SUPERROC_FP4_ERR_ARG;
                id = 0;
            }
            int g = s_open[id];
            if (g < 0) {
                g = n++;
                s_rec[g + 1] = (unsigned long long) id;
                s_open[id] = (short) g;
            }
            const unsigned long long r = s_rec[g + 1];
            const int c = (int) ((r >> 16) & 0xffULL);
            s_rec[g + 1] = (r & ~(0xffULL << 16)) | ((unsigned long long) (c + 1) << 16) |
                           ((unsigned long long) f << (24 + 8 * c));
            if (c == 3) {
                s_open[id] = -1;
            }
        }
        s_rec[0] = (unsigned long long) n;
    }
    __syncthreads();
    if (tid <= SUPERROC_FP4_ONCE_MAX_SLOTS) {
        rec[tid] = s_rec[tid];
    }
}

__device__ __forceinline__ void ls1_once_rec(const unsigned long long * p, unsigned & lo, unsigned & hi) {
    const unsigned long long r = *p;
    lo = __builtin_amdgcn_readfirstlane((unsigned) r);
    hi = __builtin_amdgcn_readfirstlane((unsigned) (r >> 32));
}

__device__ __forceinline__ void ls1_once_fans(unsigned lo, unsigned hi, int * f) {
    f[0] = (int) (lo >> 24);
    f[1] = (int) (hi & 0xffu);
    f[2] = (int) ((hi >> 8) & 0xffu);
    f[3] = (int) ((hi >> 16) & 0xffu);
}

__device__ __forceinline__ float ls1_once_dot(const ls1_blk32 & c, const int * q8, float xd) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        sumi0 = (int) __builtin_amdgcn_sudot4(true, c.vx[l], true, q8[l + 0], sumi0, false);
        sumi1 = (int) __builtin_amdgcn_sudot4(true, c.vy[l], true, q8[l + 4], sumi1, false);
    }
    return xd * (c.slo * (float) sumi0 + c.shi * (float) sumi1);
}

template <int F>
__device__ __forceinline__ void ls1_once_gu_step(int qtype, const uint8_t * g0, const uint8_t * u0, const uint8_t * g1,
                                           const uint8_t * u1, const int8_t * xq, const float * xd, int k,
                                           int nblk, int tsz, int qb, const int * t, float * sg0, float * su0,
                                           float * sg1, float * su1) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const size_t bo = (size_t) qb * (size_t) tsz;
    const ls1_blk32 cg0 = ls1_load_blk32(qtype, g0 + bo);
    const ls1_blk32 cu0 = ls1_load_blk32(qtype, u0 + bo);
    const ls1_blk32 cg1 = ls1_load_blk32(qtype, g1 + bo);
    const ls1_blk32 cu1 = ls1_load_blk32(qtype, u1 + bo);
#pragma unroll
    for (int m = 0; m < F; ++m) {
        const int * q8 = (const int *) (xq + (size_t) t[m] * (size_t) k + (size_t) qb * 32);
        int q[8];
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            q[l] = q8[l];
        }
        const float d = xd[(size_t) t[m] * (size_t) nblk + (size_t) qb];
        sg0[m] += ls1_once_dot(cg0, q, d);
        su0[m] += ls1_once_dot(cu0, q, d);
        sg1[m] += ls1_once_dot(cg1, q, d);
        su1[m] += ls1_once_dot(cu1, q, d);
    }
}

template <int F>
__device__ __forceinline__ void ls1_once_gu_body(int qtype, const uint8_t * gate, const uint8_t * up, size_t rb_k, int id,
                                           const int8_t * xq, const float * xd, float * hs, float limit, int n_used,
                                           int recip, int n_ff, int k, int j0, const int * f, int lane, int group) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int tsz = superroc_fp4_type_size(qtype);
    const int nblk = k >> 5;
    const int nsteps = nblk >> 5;
    const int tail = nblk & 31;
    const int stride = tsz * group;
    const size_t base0 = ls1_grouped_row(id, n_ff, j0, rb_k, tsz, group);
    const size_t base1 = ls1_grouped_row(id, n_ff, j0 + 1, rb_k, tsz, group);
    const uint8_t * g0 = gate + base0;
    const uint8_t * u0 = up + base0;
    const uint8_t * g1 = gate + base1;
    const uint8_t * u1 = up + base1;
    int t[F];
#pragma unroll
    for (int m = 0; m < F; ++m) {
        t[m] = (f[m] * recip) >> 16;
    }
    float eg0[F], eu0[F], eg1[F], eu1[F], og0[F], ou0[F], og1[F], ou1[F];
#pragma unroll
    for (int m = 0; m < F; ++m) {
        eg0[m] = eu0[m] = eg1[m] = eu1[m] = og0[m] = ou0[m] = og1[m] = ou1[m] = 0.f;
    }
    int c = 0;
    for (; c + 1 < nsteps; c += 2) {
        ls1_once_gu_step<F>(qtype, g0, u0, g1, u1, xq, xd, k, nblk, stride, lane + (c << 5), t, eg0, eu0, eg1, eu1);
        ls1_once_gu_step<F>(qtype, g0, u0, g1, u1, xq, xd, k, nblk, stride, lane + ((c + 1) << 5), t, og0, ou0, og1, ou1);
    }
    if (c < nsteps) {
        ls1_once_gu_step<F>(qtype, g0, u0, g1, u1, xq, xd, k, nblk, stride, lane + (c << 5), t, eg0, eu0, eg1, eu1);
        ++c;
    }
    if (tail != 0 && lane < tail) {
        if ((c & 1) == 0) {
            ls1_once_gu_step<F>(qtype, g0, u0, g1, u1, xq, xd, k, nblk, stride, lane + (c << 5), t, eg0, eu0, eg1, eu1);
        } else {
            ls1_once_gu_step<F>(qtype, g0, u0, g1, u1, xq, xd, k, nblk, stride, lane + (c << 5), t, og0, ou0, og1, ou1);
        }
    }
#pragma unroll
    for (int m = 0; m < F; ++m) {
        const float rg0 = ls1_row_sum(eg0[m], og0[m], nblk);
        const float ru0 = ls1_row_sum(eu0[m], ou0[m], nblk);
        const float rg1 = ls1_row_sum(eg1[m], og1[m], nblk);
        const float ru1 = ls1_row_sum(eu1[m], ou1[m], nblk);
        if (lane == 0) {
            const size_t out = (size_t) f[m] * (size_t) n_ff + (size_t) j0;
            hs[out] = superroc_fp4_swiglu_elem(rg0, ru0, limit);
            hs[out + 1] = superroc_fp4_swiglu_elem(rg1, ru1, limit);
        }
    }
}

// One wave per record and pair of rows.
__global__ void superroc_fp4_k_moe_gu_once(int qtype, const uint8_t * gate, const uint8_t * up, size_t rb_k,
                      const unsigned long long * recs, const int8_t * xq, const float * xd, float * hs, float limit,
                      int n_used, int recip, int n_ff, int k, int group) {
    unsigned lo, hi;
    ls1_once_rec(recs, lo, hi);
    const int g = (int) blockIdx.y;
    if (g >= (int) lo) {
        return;
    }
    ls1_once_rec(recs + g + 1, lo, hi);
    const int id = (int) (lo & 0xffffu);
    const int cnt = (int) ((lo >> 16) & 0xffu);
    int f[4];
    ls1_once_fans(lo, hi, f);
    const int lane = (int) threadIdx.x;
    const int j0 = (int) blockIdx.x << 1;
    if (cnt == 1) {
        ls1_once_gu_body<1>(qtype, gate, up, rb_k, id, xq, xd, hs, limit, n_used, recip, n_ff, k, j0, f, lane, group);
    } else if (cnt == 2) {
        ls1_once_gu_body<2>(qtype, gate, up, rb_k, id, xq, xd, hs, limit, n_used, recip, n_ff, k, j0, f, lane, group);
    } else if (cnt == 3) {
        ls1_once_gu_body<3>(qtype, gate, up, rb_k, id, xq, xd, hs, limit, n_used, recip, n_ff, k, j0, f, lane, group);
    } else {
        ls1_once_gu_body<4>(qtype, gate, up, rb_k, id, xq, xd, hs, limit, n_used, recip, n_ff, k, j0, f, lane, group);
    }
}

template <int F>
__device__ __forceinline__ void ls1_once_down_step(int qtype, const uint8_t * w0, const uint8_t * w1, bool has1,
                                             const int8_t * hq, const float * hd, int n_ff, int nb, int tsz, int qb,
                                             const int * f, float * s0, float * s1) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const size_t bo = (size_t) qb * (size_t) tsz;
    const ls1_blk32 c0 = ls1_load_blk32(qtype, w0 + bo);
    const ls1_blk32 c1 = has1 ? ls1_load_blk32(qtype, w1 + bo) : c0;
#pragma unroll
    for (int m = 0; m < F; ++m) {
        const int * q8 = (const int *) (hq + (size_t) f[m] * (size_t) n_ff + (size_t) qb * 32);
        int q[8];
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            q[l] = q8[l];
        }
        const float d = hd[(size_t) f[m] * (size_t) nb + (size_t) qb];
        s0[m] += ls1_once_dot(c0, q, d);
        if (has1) {
            s1[m] += ls1_once_dot(c1, q, d);
        }
    }
}

template <int F>
__device__ __forceinline__ void ls1_once_down_rec(int qtype, const uint8_t * w0, const uint8_t * w1, bool has1,
                                            const int8_t * hq, const float * hd, int n_ff, int nb, int tsz, int lane,
                                            const int * f, float * parts) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int nsteps = nb >> 5;
    const int tail = nb & 31;
    float e0[F], e1[F], o0[F], o1[F];
#pragma unroll
    for (int m = 0; m < F; ++m) {
        e0[m] = e1[m] = o0[m] = o1[m] = 0.f;
    }
    int c = 0;
    for (; c + 1 < nsteps; c += 2) {
        ls1_once_down_step<F>(qtype, w0, w1, has1, hq, hd, n_ff, nb, tsz, lane + (c << 5), f, e0, e1);
        ls1_once_down_step<F>(qtype, w0, w1, has1, hq, hd, n_ff, nb, tsz, lane + ((c + 1) << 5), f, o0, o1);
    }
    if (c < nsteps) {
        ls1_once_down_step<F>(qtype, w0, w1, has1, hq, hd, n_ff, nb, tsz, lane + (c << 5), f, e0, e1);
        ++c;
    }
    if (tail != 0 && lane < tail) {
        if ((c & 1) == 0) {
            ls1_once_down_step<F>(qtype, w0, w1, has1, hq, hd, n_ff, nb, tsz, lane + (c << 5), f, e0, e1);
        } else {
            ls1_once_down_step<F>(qtype, w0, w1, has1, hq, hd, n_ff, nb, tsz, lane + (c << 5), f, o0, o1);
        }
    }
#pragma unroll
    for (int m = 0; m < F; ++m) {
        const float p0 = ls1_row_sum(e0[m], o0[m], nb);
        const float p1 = has1 ? ls1_row_sum(e1[m], o1[m], nb) : 0.f;
        if (lane == 0) {
            parts[f[m]] = p0;
            parts[SUPERROC_FP4_ONCE_MAX_SLOTS + f[m]] = p1;
        }
    }
}

__global__ void superroc_fp4_k_moe_down_once(int qtype, const uint8_t * down, size_t rb_ff, const unsigned long long * recs,
                        const float * weights_d, const int8_t * hq, const float * hd, float * y, int n_used, int k,
                        int n_ff, int m, int group) {
    __shared__ float parts[2 * SUPERROC_FP4_ONCE_MAX_SLOTS];
    const int lane = (int) threadIdx.x;
    const int tsz = superroc_fp4_type_size(qtype);
    const int stride = tsz * group;
    const int nb = n_ff >> 5;
    const int npair = (k + 1) >> 1;
    unsigned lo, hi;
    ls1_once_rec(recs, lo, hi);
    const int ngroups = (int) lo;
    for (int pair = (int) blockIdx.x; pair < npair; pair += (int) gridDim.x) {
        const int j0 = pair << 1;
        const int j1 = j0 + 1;
        const bool has1 = j1 < k;
        for (int g = 0; g < ngroups; ++g) {
            ls1_once_rec(recs + g + 1, lo, hi);
            const int id = (int) (lo & 0xffffu);
            const uint8_t * w0 = down + ls1_grouped_row(id, k, j0, rb_ff, tsz, group);
            const uint8_t * w1 = down + ls1_grouped_row(id, k, j1, rb_ff, tsz, group);
            const int cnt = (int) ((lo >> 16) & 0xffu);
            int f[4];
            ls1_once_fans(lo, hi, f);
            if (cnt == 1) {
                ls1_once_down_rec<1>(qtype, w0, w1, has1, hq, hd, n_ff, nb, stride, lane, f, parts);
            } else if (cnt == 2) {
                ls1_once_down_rec<2>(qtype, w0, w1, has1, hq, hd, n_ff, nb, stride, lane, f, parts);
            } else if (cnt == 3) {
                ls1_once_down_rec<3>(qtype, w0, w1, has1, hq, hd, n_ff, nb, stride, lane, f, parts);
            } else {
                ls1_once_down_rec<4>(qtype, w0, w1, has1, hq, hd, n_ff, nb, stride, lane, f, parts);
            }
        }
        if (lane == 0) {
            int s = 0;
            for (int t = 0; t < m; ++t) {
                float a0 = 0.f;
                float a1 = 0.f;
                for (int slot = 0; slot < n_used; ++slot, ++s) {
                    const float wv = weights_d[s];
                    a0 += wv * parts[s];
                    a1 += wv * parts[SUPERROC_FP4_ONCE_MAX_SLOTS + s];
                }
                y[(size_t) t * (size_t) k + (size_t) j0] = a0;
                if (has1) {
                    y[(size_t) t * (size_t) k + (size_t) j1] = a1;
                }
            }
        }
    }
}

static int ls1_variant_once() {
    static int v = -1;
    if (v < 0) {
        v = ls1_env_on("SUPERROC_MOE_READ_ONCE", 1);
    }
    return v;
}

// SUPERROC_MOE_ONCE_MIN_ROWS: the fewest token rows that take the read-once
// grouping (default 4). Both paths give every value equal, so only the
// speed of 2- and 3-row verify batches depends on it.
static int ls1_once_min_rows() {
    static int v = -1;
    if (v < 0) {
        const char * e = ::superroc::core::GetEnv("SUPERROC_MOE_ONCE_MIN_ROWS");
        const int parsed = (e != nullptr && e[0] >= '1' && e[0] <= '9' && e[1] == '\0') ? e[0] - '0' : 4;
        v = parsed;
    }
    return v;
}

// Quantise into the caller's buffers, for an activation that other quantisations come between.
extern "C" int superroc_fp4_quant_to(const float * x, int n_tokens, int k, int8_t * xq, float * xd,
                              hipStream_t stream) {
    if (x == nullptr || xq == nullptr || xd == nullptr) return SUPERROC_FP4_ERR_ARG;
    if (n_tokens < 1 || n_tokens > SUPERROC_FP4_LS1_MAX_M || k <= 0 || (k % 32) != 0) return SUPERROC_FP4_ERR_ARG;
    const int nb = n_tokens * k / 32;
    superroc_fp4_k_q8_1_quant<<<(nb + 255) / 256, 256, 0, stream>>>(x, xq, xd, nb);
    return ls1_chk("q8 quant to");
}

static int ls1_expert_ffn_m_impl(int qtype, const void * gate, const void * up,
                                 const void * down, const float * x,
                                 const int * ids_d, const float * weights_d,
                                 int n_tokens, int n_used, int n_expert,
                                 int k, int n_ff, float limit, float * y,
                                 const int8_t * xq_ready, const float * xd_ready, int group,
                                 hipStream_t stream) {
    const size_t rb_k = superroc_fp4_row_bytes(qtype, k);
    const size_t rb_ff = superroc_fp4_row_bytes(qtype, n_ff);
    if (!gate || !up || !down || !x || !ids_d || !weights_d || !y) return SUPERROC_FP4_ERR_ARG;
    if (qtype != SUPERROC_FP4_Q4_0_ROCMFP4 && qtype != SUPERROC_FP4_Q4_0_ROCMFP4_FAST) return SUPERROC_FP4_ERR_ARG;
    if (rb_k == 0 || rb_ff == 0) return SUPERROC_FP4_ERR_ARG;
    if (n_tokens < 1 || n_tokens > SUPERROC_FP4_LS1_MAX_M || n_used <= 0 || n_expert <= 0) return SUPERROC_FP4_ERR_ARG;
    if (k <= 0 || n_ff <= 0 || (k % 32) != 0 || (n_ff % 32) != 0) return SUPERROC_FP4_ERR_ARG;

    const int set = superroc_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // Refused: the caller declines, never waits.
    const int n_slots = n_tokens * n_used;
    const size_t nn = (size_t) n_slots * (size_t) n_ff;
    // Size to the decode max (m=8) for this n_used so a later wider verify
    // does not realloc pointers a captured graph still launches.
    const size_t nn_cap = (size_t) SUPERROC_FP4_LS1_MAX_M * (size_t) n_used * (size_t) n_ff;
    float * moe_hs_m = nullptr;
    int * moe_rc_m = nullptr;
    int rc = superroc_fp4_slots_grow_hsm(set, nn_cap > nn ? nn_cap : nn, stream, &moe_hs_m, &moe_rc_m);
    if (rc != SUPERROC_FP4_OK) return rc;

    const int8_t * qs = xq_ready;
    const float * ds = xd_ready;
    if (qs == nullptr) {
        int8_t * q2 = nullptr;
        float * d2 = nullptr;
        rc = ls1_quant_m(x, n_tokens * k, stream, &q2, &d2);
        if (rc != SUPERROC_FP4_OK) return rc;
        qs = q2;
        ds = d2;
    }

    // Below four rows the slot walk is as fast: too few reads repeat to pay for the grouping.
    if (n_tokens >= ls1_once_min_rows() && n_slots <= SUPERROC_FP4_ONCE_MAX_SLOTS && n_expert <= SUPERROC_FP4_ONCE_MAX_EXPERTS &&
        n_used <= 16 && (n_ff % 64) == 0 && ls1_variant_once()) {
        unsigned long long * rec = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_moe_batch_mu);
            MoeBatchBuffers & B = g_moe_batch[set];
            if (B.rec == nullptr &&
                hipMalloc((void **) &B.rec, (SUPERROC_FP4_ONCE_MAX_SLOTS + 1) * sizeof(unsigned long long)) != hipSuccess) {
                return SUPERROC_FP4_ERR_HIP;
            }
            rec = B.rec;
        }
        superroc_fp4_k_moe_group_once<<<1, 128, 0, stream>>>(ids_d, n_slots, n_expert, rec, moe_rc_m);
        if ((rc = ls1_chk("moe once group")) != SUPERROC_FP4_OK) return rc;
        const int recip = (65536 + n_used - 1) / n_used;
        superroc_fp4_k_moe_gu_once<<<dim3((unsigned) (n_ff >> 1), (unsigned) n_slots), 32, 0, stream>>>(
            qtype, (const uint8_t *) gate, (const uint8_t *) up, rb_k, rec, qs, ds, moe_hs_m, limit, n_used,
            recip, n_ff, k, group);
        if ((rc = ls1_chk("moe once gate/up")) != SUPERROC_FP4_OK) return rc;
        int8_t * hq1 = nullptr;
        float * hd1 = nullptr;
        rc = ls1_quant_m(moe_hs_m, (int) nn, stream, &hq1, &hd1);
        if (rc != SUPERROC_FP4_OK) return rc;
        superroc_fp4_k_moe_down_once<<<(unsigned) ((k + 1) >> 1), 32, 0, stream>>>(
            qtype, (const uint8_t *) down, rb_ff, rec, weights_d, hq1, hd1, y, n_used, k, n_ff, n_tokens, group);
        return ls1_chk("moe once down");
    }
    superroc_fp4_k_moe_gu_m<<<(unsigned) (n_ff * n_slots), 64, 0, stream>>>(qtype, (const uint8_t *) gate,
                                                     (const uint8_t *) up, rb_k,
                                                     qs, ds, ids_d,
                                                     moe_hs_m, limit,
                                                     n_used, n_expert, k, n_ff, n_slots,
                                                     moe_rc_m, group);
    if ((rc = ls1_chk("moe m gate/up")) != SUPERROC_FP4_OK) return rc;

    int8_t * hq = nullptr;
    float * hd = nullptr;
    rc = ls1_quant_m(moe_hs_m, (int) nn, stream, &hq, &hd);
    if (rc != SUPERROC_FP4_OK) return rc;

    superroc_fp4_k_moe_down_m<<<(unsigned) ((k + 1) >> 1), 32, 0, stream>>>(qtype, (const uint8_t *) down, rb_ff,
                                       ids_d, weights_d, hq, hd, y,
                                       n_used, n_expert, k, n_ff, n_tokens,
                                       moe_rc_m, group);
    return ls1_chk("moe m down");
}

extern "C" int superroc_fp4_expert_ffn_m(int qtype, const void * gate, const void * up,
                                 const void * down, const float * x,
                                 const int * ids_d, const float * weights_d,
                                 int n_tokens, int n_used, int n_expert,
                                 int k, int n_ff, float limit, float * y,
                                 hipStream_t stream) {
    return ls1_expert_ffn_m_impl(qtype, gate, up, down, x, ids_d, weights_d, n_tokens, n_used, n_expert,
                                 k, n_ff, limit, y, nullptr, nullptr, 1, stream);
}

extern "C" int superroc_fp4_expert_ffn_m_preq(int qtype, const void * gate, const void * up,
                                      const void * down, const float * x,
                                      const int * ids_d, const float * weights_d,
                                      int n_tokens, int n_used, int n_expert,
                                      int k, int n_ff, float limit, float * y,
                                      const int8_t * xq, const float * xd,
                                      hipStream_t stream) {
    return ls1_expert_ffn_m_impl(qtype, gate, up, down, x, ids_d, weights_d, n_tokens, n_used, n_expert,
                                 k, n_ff, limit, y, xq, xd, 1, stream);
}

extern "C" int superroc_fp4_expert_ffn_grouped(int qtype, const void * gate, const void * up,
                                       const void * down, const float * x,
                                       const int * ids_d, const float * weights_d,
                                       int n_tokens, int n_used, int n_expert,
                                       int k, int n_ff, float limit, float * y,
                                       const int8_t * xq, const float * xd, int group,
                                       hipStream_t stream) {
    return ls1_expert_ffn_m_impl(qtype, gate, up, down, x, ids_d, weights_d, n_tokens, n_used, n_expert,
                                 k, n_ff, limit, y, xq, xd, group, stream);
}

// One row group per block: [row][block] to [block][row] through group memory, in place.
enum { SUPERROC_FP4_GROUP_LDS_BYTES = 32 * 1024 };

__global__ void superroc_fp4_k_group_rows(uint8_t * w, size_t group_bytes, int nb, int tsz, int group) {
    __shared__ __attribute__((aligned(16))) uint8_t lds[SUPERROC_FP4_GROUP_LDS_BYTES];
    uint8_t * g = w + (size_t) blockIdx.x * group_bytes;
    const int n = nb * group;
    const int step = (int) blockDim.x;
    for (int e = (int) threadIdx.x; e < n; e += step) {
        const int r = e / nb;
        const int b = e - r * nb;
        const uint8_t * from = g + (size_t) e * (size_t) tsz;
        uint8_t * to = lds + (b * group + r) * tsz;
        uint4 codes;
        __builtin_memcpy(&codes, from, 16);
        __builtin_memcpy(to, &codes, 16);
        to[16] = from[16];
        if (tsz == 18) {
            to[17] = from[17];
        }
    }
    __syncthreads();
    for (int e = (int) threadIdx.x; e < n; e += step) {
        const uint8_t * from = lds + e * tsz;
        uint8_t * to = g + (size_t) e * (size_t) tsz;
        uint4 codes;
        __builtin_memcpy(&codes, from, 16);
        __builtin_memcpy(to, &codes, 16);
        to[16] = from[16];
        if (tsz == 18) {
            to[17] = from[17];
        }
    }
}

extern "C" int superroc_fp4_group_rows(void * w, int qtype, size_t rows, int cols, int group,
                                   hipStream_t stream) {
    if (w == nullptr || (qtype != SUPERROC_FP4_Q4_0_ROCMFP4 && qtype != SUPERROC_FP4_Q4_0_ROCMFP4_FAST)) {
        return SUPERROC_FP4_ERR_ARG;
    }
    if (group < 2 || (group & (group - 1)) != 0 || cols <= 0 || (cols % 32) != 0 ||
        rows == 0 || rows % (size_t) group != 0) {
        return SUPERROC_FP4_ERR_ARG;
    }
    const int tsz = superroc_fp4_type_size(qtype);
    const int nb = cols / 32;
    const size_t group_bytes = (size_t) group * (size_t) nb * (size_t) tsz;
    if (group_bytes > SUPERROC_FP4_GROUP_LDS_BYTES) {
        return SUPERROC_FP4_ERR_ARG;
    }
    superroc_fp4_k_group_rows<<<(unsigned) (rows / (size_t) group), 64, 0, stream>>>(
        (uint8_t *) w, group_bytes, nb, tsz, group);
    return ls1_chk("group rows");
}

// n_tokens == 1 and n_used != 1 stay on superroc_fp4_expert_ffn_m, so Step's
// single-token expert and the routed verify (n_used = 8) do not change.
extern "C" int superroc_fp4_expert_ffn_shared_m(int qtype, const void * gate, const void * up,
                                        const void * down, const float * x,
                                        const int * ids_d, const float * weights_d,
                                        int n_tokens, int n_used, int n_expert,
                                        int k, int n_ff, float limit, float * y,
                                        hipStream_t stream) {
    if (n_used != 1 || n_tokens <= 1) {
        return superroc_fp4_expert_ffn_m(qtype, gate, up, down, x, ids_d, weights_d,
                                 n_tokens, n_used, n_expert, k, n_ff, limit, y, stream);
    }
    const size_t rb_k = superroc_fp4_row_bytes(qtype, k);
    const size_t rb_ff = superroc_fp4_row_bytes(qtype, n_ff);
    if (!gate || !up || !down || !x || !ids_d || !weights_d || !y) return SUPERROC_FP4_ERR_ARG;
    if (qtype != SUPERROC_FP4_Q4_0_ROCMFP4 && qtype != SUPERROC_FP4_Q4_0_ROCMFP4_FAST) return SUPERROC_FP4_ERR_ARG;
    if (rb_k == 0 || rb_ff == 0) return SUPERROC_FP4_ERR_ARG;
    if (n_tokens > SUPERROC_FP4_LS1_MAX_M || n_expert <= 0) return SUPERROC_FP4_ERR_ARG;
    if (k <= 0 || n_ff <= 0 || (k % 32) != 0 || (n_ff % 32) != 0) return SUPERROC_FP4_ERR_ARG;

    const int set = superroc_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // Refused: the caller declines, never waits.
    const size_t nn = (size_t) n_tokens * (size_t) n_ff;
    const size_t nn_cap = (size_t) SUPERROC_FP4_LS1_MAX_M * (size_t) n_ff;
    float * moe_hs_m = nullptr;
    int * moe_rc_m = nullptr;
    int rc = superroc_fp4_slots_grow_hsm(set, nn_cap > nn ? nn_cap : nn, stream, &moe_hs_m, &moe_rc_m);
    if (rc != SUPERROC_FP4_OK) return rc;

    int8_t * qs = nullptr;
    float * ds = nullptr;
    rc = ls1_quant_m(x, n_tokens * k, stream, &qs, &ds);
    if (rc != SUPERROC_FP4_OK) return rc;

    superroc_fp4_k_moe_gu_one_m<<<(unsigned) n_ff, 64, 0, stream>>>(
        qtype, (const uint8_t *) gate, (const uint8_t *) up, rb_k,
        qs, ds, ids_d, moe_hs_m, limit, n_expert, k, n_ff, n_tokens,
        moe_rc_m);
    if ((rc = ls1_chk("shared m gate/up")) != SUPERROC_FP4_OK) return rc;

    int8_t * hq = nullptr;
    float * hd = nullptr;
    rc = ls1_quant_m(moe_hs_m, (int) nn, stream, &hq, &hd);
    if (rc != SUPERROC_FP4_OK) return rc;

    superroc_fp4_k_moe_down_one_m<<<(unsigned) k, 64, 0, stream>>>(
        qtype, (const uint8_t *) down, rb_ff, ids_d, weights_d, hq, hd, y,
        n_expert, k, n_ff, n_tokens, moe_rc_m);
    return ls1_chk("shared m down");
}

// Read the small-m path's status word and clear it. Blocking; one call per verify batch.
// Variant 1 of the grouped down projection. hs is already quantised into the MoE slot.
// The FULL grouped expert FFN with the warp-per-row down kernel, so the A/B in
// the test compares like with like: the previous version of that benchmark timed
// superroc_fp4_expert_ffn_dev (gate + up + quantise + down) against the down kernel
// alone and reported 0.26x, which is a comparison of three stages with one.
extern "C" int superroc_fp4_expert_ffn_dev_tuned(int qtype, const void * gate, const void * up,
                                         const void * down, const float * x,
                                         const int * ids_d, const float * weights_d,
                                         int n_used, int n_expert, int k, int n_ff,
                                         float limit, float * y, int block,
                                         hipStream_t stream) {
    const size_t rb_k = superroc_fp4_row_bytes(qtype, k);
    const size_t rb_ff = superroc_fp4_row_bytes(qtype, n_ff);
    if (!gate || !up || !down || !x || !ids_d || !weights_d || !y) return SUPERROC_FP4_ERR_ARG;
    if (qtype != SUPERROC_FP4_Q4_0_ROCMFP4 && qtype != SUPERROC_FP4_Q4_0_ROCMFP4_FAST) return SUPERROC_FP4_ERR_ARG;
    if (rb_k == 0 || rb_ff == 0) return SUPERROC_FP4_ERR_ARG;
    if (n_used <= 0 || n_expert <= 0 || k <= 0 || n_ff <= 0) return SUPERROC_FP4_ERR_ARG;
    if ((k % 32) != 0 || (n_ff % 32) != 0 || block < 32) return SUPERROC_FP4_ERR_ARG;

    const int set = superroc_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // Refused: the caller declines, never waits.
    const size_t nn = (size_t) n_used * (size_t) n_ff;
    float * moe_hs = nullptr;
    int * moe_rc = nullptr;
    int rc = superroc_fp4_slots_grow_hs(set, nn, stream, &moe_hs, &moe_rc);
    if (rc != SUPERROC_FP4_OK) return rc;
    rc = superroc_fp4_quant_q8(x, k, SUPERROC_FP4_LS1_SLOT_FFN_X, stream);
    if (rc != SUPERROC_FP4_OK) return rc;
    int8_t * xq = nullptr;
    float * xd = nullptr;
    if (superroc_fp4_slots_get_q8(set, SUPERROC_FP4_LS1_SLOT_FFN_X, k, &xq, &xd) != SUPERROC_FP4_OK) {
        return SUPERROC_FP4_ERR_HIP;
    }

    // One CTA per (slot, row): the kernel takes its slot from blockIdx / n_ff, so the
    // grid is exactly n_ff * n_used (n_ff need not be a multiple of 64).
    superroc_fp4_k_moe_gate_up_swiglu<<<(unsigned) n_ff * (unsigned) n_used, 64, 0, stream>>>(
        qtype, (const uint8_t *) gate, (const uint8_t *) up, rb_k,
        ids_d, xq, xd, moe_hs, limit, n_ff, n_expert, k, moe_rc);
    if ((rc = ls1_chk("moe gate/up")) != SUPERROC_FP4_OK) return rc;
    rc = superroc_fp4_quant_q8(moe_hs, (int) nn, SUPERROC_FP4_LS1_SLOT_MOE_H, stream);
    if (rc != SUPERROC_FP4_OK) return rc;
    int8_t * hq = nullptr;
    float * hd = nullptr;
    if (superroc_fp4_slots_get_q8(set, SUPERROC_FP4_LS1_SLOT_MOE_H, (int) nn, &hq, &hd) != SUPERROC_FP4_OK) {
        return SUPERROC_FP4_ERR_HIP;
    }
    const int nwarps = block >> 5;
    superroc_fp4_k_moe_down_weighted_rows<<<ls1_gemv_blocks(k, nwarps), block, 0, stream>>>(
        qtype, (const uint8_t *) down, rb_ff, ids_d, weights_d,
        hq, hd, y, n_used, n_expert, k, n_ff,
        1 << 20, moe_rc);
    return ls1_chk("moe down rows tuned");
}

extern "C" int superroc_fp4_expert_ffn_dev_rows(int qtype, const void * down,
                                         const int * ids_d, const float * weights_d,
                                         int n_used, int n_expert, int k, int n_ff,
                                         float * y, int block, hipStream_t stream) {
    const size_t rb_ff = superroc_fp4_row_bytes(qtype, n_ff);
    if (!down || !ids_d || !weights_d || !y || rb_ff == 0 || block < 32) return SUPERROC_FP4_ERR_ARG;
    const int set = superroc_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // Refused: the caller declines, never waits.
    int8_t * hq = nullptr;
    float * hd = nullptr;
    int * moe_rc = nullptr;
    {
        SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        SuperROCFp4SlotSet & S = mgr.sets[(size_t) set];
        if (S.qs[SUPERROC_FP4_LS1_SLOT_MOE_H] == nullptr || S.moe_rc == nullptr) {
            return SUPERROC_FP4_ERR_ARG;
        }
        hq = S.qs[SUPERROC_FP4_LS1_SLOT_MOE_H];
        hd = S.ds[SUPERROC_FP4_LS1_SLOT_MOE_H];
        moe_rc = S.moe_rc;
    }
    const int nwarps = block >> 5;
    superroc_fp4_k_moe_down_weighted_rows<<<ls1_gemv_blocks(k, nwarps), block, 0, stream>>>(
        qtype, (const uint8_t *) down, rb_ff, ids_d, weights_d,
        hq, hd, y, n_used, n_expert, k, n_ff,
        1 << 20, moe_rc);
    return ls1_chk("moe down rows");
}

extern "C" int superroc_fp4_expert_ffn_m_check_set(int set);
extern "C" int superroc_fp4_expert_ffn_m_check_set_on_stream(int set, hipStream_t stream) {
    if (stream == nullptr) return SUPERROC_FP4_ERR_ARG;
    if (hipStreamSynchronize(stream) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
    return superroc_fp4_expert_ffn_m_check_set(set);
}

extern "C" int superroc_fp4_expert_ffn_m_check_set(int set) {
    int * rc_d = nullptr;
    {
        SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        if (set < 0 || (size_t) set >= mgr.sets.size()) return SUPERROC_FP4_ERR_ARG;
        rc_d = mgr.sets[(size_t) set].moe_rc_m;
    }
    if (rc_d == nullptr) return SUPERROC_FP4_OK;
    int rc = SUPERROC_FP4_OK;
    if (hipMemcpy(&rc, rc_d, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
        return SUPERROC_FP4_ERR_HIP;
    }
    if (hipMemset(rc_d, 0, sizeof(int)) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
    return rc;
}

extern "C" int superroc_fp4_expert_ffn_m_check(void) {
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    int n = 0;
    {
        std::lock_guard<std::mutex> lk(mgr.mu);
        n = (int) mgr.sets.size();
    }
    int rc = SUPERROC_FP4_OK;
    for (int s = 0; s < n; ++s) {
        const int r = superroc_fp4_expert_ffn_m_check_set(s);
        if (r != SUPERROC_FP4_OK && rc == SUPERROC_FP4_OK) {
            rc = r;
        }
    }
    return rc;
}

// Test-only: the small-m path's hs scratch.
extern "C" int superroc_fp4_moe_hs_m_peek_set(int set, float * out, int n) {
    float * hs = nullptr;
    {
        SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        if (set < 0 || (size_t) set >= mgr.sets.size()) return -1;
        hs = mgr.sets[(size_t) set].moe_hs_m;
    }
    if (hs == nullptr || n <= 0) return -1;
    return hipMemcpy(out, hs, (size_t) n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
}

extern "C" int superroc_fp4_moe_hs_m_peek(float * out, int n) {
    return superroc_fp4_moe_hs_m_peek_set(0, out, n);
}

// Small-m verify GEMV. m = 2..8 columns, one load of each weight
// block, then the m=1 dot (ls1_dot_fp4_blk32_q8) and the same 64-thread
// ls1_block_sum. Launch geometry matches superroc_fp4_gemv_fp4_q8: one CTA per row
// at n <= 8192, otherwise the 2048-wide grid stride. Does not allocate.
// The verify pair lives in the slot sets (one per stream).
extern "C" int superroc_fp4_verify_m_reserve(int total) {
    if (total <= 0 || (total % 32) != 0) return SUPERROC_FP4_ERR_ARG;
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    bool grew = false;
    for (size_t i = 0; i < mgr.sets.size(); ++i) {
        SuperROCFp4SlotSet & S = mgr.sets[i];
        if (S.qv_cap >= total && S.qv != nullptr && S.dv != nullptr) continue;
        grew = true;
    }
    if (!grew) return SUPERROC_FP4_OK;
    // Reserve takes no stream; drain the device before freeing, so no queued
    // kernel still reads a pair we replace. Load-time only in serve.
    if (hipDeviceSynchronize() != hipSuccess) return SUPERROC_FP4_ERR_HIP;
    for (size_t i = 0; i < mgr.sets.size(); ++i) {
        SuperROCFp4SlotSet & S = mgr.sets[i];
        if (S.qv_cap >= total && S.qv != nullptr && S.dv != nullptr) continue;
        int8_t * qs = nullptr;
        float * ds = nullptr;
        if (superroc_fp4_slots_malloc((void **) &qs, (size_t) total + SUPERROC_FP4_TAIL_PAD) != SUPERROC_FP4_OK) {
            return SUPERROC_FP4_ERR_HIP;
        }
        if (superroc_fp4_slots_malloc((void **) &ds,
                                  (size_t) (total / 32) * sizeof(float) + SUPERROC_FP4_TAIL_PAD) != SUPERROC_FP4_OK) {
            (void) hipFree(qs);
            return SUPERROC_FP4_ERR_HIP;
        }
        // Commit the new pair first: a failed free of the old one then loses memory,
        // never leaves the set pointing at a freed buffer.
        int8_t * const old_q = S.qv;
        float * const old_d = S.dv;
        S.qv = qs;
        S.dv = ds;
        S.qv_cap = total;
        S.vq_x = nullptr;
        S.vq_m = 0;
        S.vq_k = 0;
        if (old_q) (void) hipFree(old_q);
        if (old_d) (void) hipFree(old_d);
    }
    return SUPERROC_FP4_OK;
}

// One wave per group, no LDS and no barrier, two rows per group. Every row keeps the m = 1
// arithmetic of the 64-thread form exactly: ascending qb per thread, ls1_dot_blk32 per block, the
// same two 32-lane trees and the same sum of the two. The wave plays both waves of that group: at
// step c it holds qb = lane + 32c, and thread qb mod 64 is lane + 32(c & 1), so even and odd steps
// keep their own accumulators and their own trees.
//
// Why not a barrier per value: __syncthreads also runs buffer_gl0_inv, which drops every line this
// CU holds in L0, for every wave on the CU, and a reduction took two of them per row and column.
// Each one made the next wave read its activations from further away, so the cost of a verify row
// added to the cost of the weights instead of overlapping it. Without them the verify pass stays
// within a few percent of the memory limit from 2 rows to 8. Odd n ends in a predicated tail pair.
template <int M>
__global__ void superroc_fp4_k_gemv_verify_mt(int qtype, const uint8_t * W, size_t row_bytes,
                                          const int8_t * xq, const float * xd,
                                          float * y, int k, int n) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int tsz = superroc_fp4_type_size(qtype);
    const int nblk = k >> 5;
    const int lane = (int) threadIdx.x;
    float even0[M];
    float even1[M];
    float odd0[M];
    float odd1[M];
    // Up to six columns a full pair takes its own loop with has1 known at compile time: no branch
    // per block, about 3 % of a whole pass. At seven and eight that copy of the walk runs out of
    // registers and costs 28 %, so those widths share one loop.
    const int npair = M <= 6 ? (n >> 1) : ((n + 1) >> 1);
    for (int pair = (int) blockIdx.x; pair < npair; pair += (int) gridDim.x) {
        const int row0 = pair << 1;
        const int row1 = row0 + 1;
#pragma unroll
        for (int t = 0; t < M; ++t) {
            even0[t] = 0.f;
            even1[t] = 0.f;
            odd0[t] = 0.f;
            odd1[t] = 0.f;
        }
        const uint8_t * w0 = W + (size_t) row0 * row_bytes;
        if constexpr (M <= 6) {
            ls1_wave_walk2<M>(qtype, w0, w0 + row_bytes, true, xq, xd, k, nblk, tsz, lane, even0, even1, odd0, odd1);
        } else {
            const bool has1 = row1 < n;
            ls1_wave_walk2<M>(qtype, w0, has1 ? w0 + row_bytes : w0, has1, xq, xd, k, nblk, tsz, lane, even0, even1, odd0, odd1);
        }
#pragma unroll
        for (int t = 0; t < M; ++t) {
            const float r0 = (0.f + ls1_lane0_sum(even0[t])) + ls1_lane0_sum(odd0[t]);
            if (lane == 0) {
                y[(size_t) t * (size_t) n + (size_t) row0] = r0;
            }
            // Uniform: true for every pair of the first loop form.
            if (M <= 6 || row1 < n) {
                const float r1 = (0.f + ls1_lane0_sum(even1[t])) + ls1_lane0_sum(odd1[t]);
                if (lane == 0) {
                    y[(size_t) t * (size_t) n + (size_t) row1] = r1;
                }
            }
        }
    }
    // The odd row, in the group the pair loop would have given it.
    if constexpr (M <= 6) {
        const int last = n >> 1;
        if ((n & 1) != 0 && (last % (int) gridDim.x) == (int) blockIdx.x) {
#pragma unroll
            for (int t = 0; t < M; ++t) {
                even0[t] = 0.f;
                even1[t] = 0.f;
                odd0[t] = 0.f;
                odd1[t] = 0.f;
            }
            const int row0 = last << 1;
            const uint8_t * w0 = W + (size_t) row0 * row_bytes;
            ls1_wave_walk2<M>(qtype, w0, w0, false, xq, xd, k, nblk, tsz, lane, even0, even1, odd0, odd1);
#pragma unroll
            for (int t = 0; t < M; ++t) {
                const float r0 = (0.f + ls1_lane0_sum(even0[t])) + ls1_lane0_sum(odd0[t]);
                if (lane == 0) {
                    y[(size_t) t * (size_t) n + (size_t) row0] = r0;
                }
            }
        }
    }
}

// Launch n rows against m columns in Q8_1. Two rows per group, each pair past the grid wraps to
// another group, so a tall tensor launches at most SUPERROC_FP4_LS1_GEMV_WIDE_MAX waves.
static int ls1_launch_rows_q8(int qtype, const uint8_t * W, size_t rb, const int8_t * xq,
                              const float * xd, float * y, int k, int n, int m,
                              hipStream_t stream) {
    // One row has no read to overlap with: the 64-thread kernels stay 2 % faster over a whole pass.
    if (m == 1) {
        if (n <= SUPERROC_FP4_LS1_GEMV_WIDE_MAX) {
            superroc_fp4_k_gemv_wide_q8_cta64<<<n, 64, 0, stream>>>(qtype, W, rb, xq, xd, y, k, n);
        } else {
            const int blocks = n < SUPERROC_FP4_LS1_GEMV_GS_BLOCKS ? n : SUPERROC_FP4_LS1_GEMV_GS_BLOCKS;
            superroc_fp4_k_gemv_wide_q8_gs<<<blocks, 64, 0, stream>>>(qtype, W, rb, xq, xd, y, k, n);
        }
        return SUPERROC_FP4_OK;
    }
    using row_kn = void (*)(int, const uint8_t *, size_t, const int8_t *, const float *, float *,
                            int, int);
    row_kn kn = nullptr;
    switch (m) {
        case 2: kn = superroc_fp4_k_gemv_verify_mt<2>; break;
        case 3: kn = superroc_fp4_k_gemv_verify_mt<3>; break;
        case 4: kn = superroc_fp4_k_gemv_verify_mt<4>; break;
        case 5: kn = superroc_fp4_k_gemv_verify_mt<5>; break;
        case 6: kn = superroc_fp4_k_gemv_verify_mt<6>; break;
        case 7: kn = superroc_fp4_k_gemv_verify_mt<7>; break;
        case 8: kn = superroc_fp4_k_gemv_verify_mt<8>; break;
        default: return SUPERROC_FP4_ERR_ARG;
    }
    const int npair = (n + 1) >> 1;
    const int grid = npair < SUPERROC_FP4_LS1_GEMV_WIDE_MAX ? npair : SUPERROC_FP4_LS1_GEMV_WIDE_MAX;
    kn<<<(unsigned) grid, 32, 0, stream>>>(qtype, W, rb, xq, xd, y, k, n);
    return SUPERROC_FP4_OK;
}

// Super Roc QVO: quantize_x == false reuses the set's Q8 copy of x (the _xq
// entry point); it refuses unless that copy is of this x, m and k.
static int gemv_verify_m_impl(int qtype, const void * W, const float * x, float * y,
                              int k, int n, int m, hipStream_t stream, bool quantize_x) {
    // SUPERROC_FP4_FORCE_REFUSE=1 declines every
    // FP4 GEMV intercept (decode and verify alike) so the fallback paths
    // can be proven bit-identical. Off unless set.
    static const int force_refuse = [] {
        const char * e = ::superroc::core::GetEnv("SUPERROC_FP4_FORCE_REFUSE");
        return (e != nullptr && e[0] == '1') ? 1 : 0;
    }();
    if (force_refuse) return SUPERROC_FP4_ERR_ARG;
    if (W == nullptr || x == nullptr || y == nullptr || n <= 0 || k <= 0 || (k % 32) != 0 ||
        m < 2 || m > 8) {
        return SUPERROC_FP4_ERR_ARG;
    }
    if (qtype != SUPERROC_FP4_Q4_0_ROCMFP4 && qtype != SUPERROC_FP4_Q4_0_ROCMFP4_FAST) return SUPERROC_FP4_ERR_ARG;
    const int set = superroc_fp4_slots_for_stream_at(stream, __func__, (long) k, false);
    if (set < 0) return set;  // Refused: the caller declines, never waits.
    int8_t * qv = nullptr;
    float * dv = nullptr;
    {
        SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        SuperROCFp4SlotSet & S = mgr.sets[(size_t) set];
        const long long need = (long long) m * (long long) k;
        if (need > S.qv_cap || S.qv == nullptr || S.dv == nullptr) {
            static int warned = 0;
            const int cap = S.qv_cap;
            if (warned++ < 4) {
                fprintf(stderr, "VERIFY_M_SHORT set=%d need=%lld cap=%d\n", set, need, cap);
            }
            return SUPERROC_FP4_ERR_ARG;
        }
        if (!quantize_x && (S.vq_x != x || S.vq_m != m || S.vq_k != k)) {
            return SUPERROC_FP4_ERR_ARG;
        }
        qv = S.qv;
        dv = S.dv;
    }
    const size_t rb = superroc_fp4_row_bytes(qtype, k);
    if (rb == 0) return SUPERROC_FP4_ERR_ARG;
    if (quantize_x) {
        const int nb = (m * k) / 32;
        const int threads = 256;
        superroc_fp4_k_q8_1_quant<<<(nb + threads - 1) / threads, threads, 0, stream>>>(
            x, qv, dv, nb);
        const int qrc = ls1_chk("verify quant");
        SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
        std::lock_guard<std::mutex> lk(mgr.mu);
        SuperROCFp4SlotSet & S = mgr.sets[(size_t) set];
        S.vq_x = qrc == SUPERROC_FP4_OK ? x : nullptr;
        S.vq_m = m;
        S.vq_k = k;
        if (qrc != SUPERROC_FP4_OK) return SUPERROC_FP4_ERR_HIP;
    }
    const int launched = ls1_launch_rows_q8(qtype, (const uint8_t *) W, rb, qv, dv, y, k, n, m, stream);
    if (launched != SUPERROC_FP4_OK) return launched;
    return ls1_chk("verify gemv");
}

extern "C" int superroc_fp4_gemv_verify_m(int qtype, const void * W, const float * x, float * y,
                                      int k, int n, int m, hipStream_t stream) {
    return gemv_verify_m_impl(qtype, W, x, y, k, n, m, stream, true);
}

extern "C" int superroc_fp4_gemv_verify_m_xq(int qtype, const void * W, const float * x, float * y,
                                         int k, int n, int m, hipStream_t stream) {
    return gemv_verify_m_impl(qtype, W, x, y, k, n, m, stream, false);
}

// ---------------------------------------------------------------------------
// Slot-set ownership API. See the header for the contract.
// (SUPERROC_FP4_LS1_MAX_SETS lives above the manager: the resolver needs it.)
// ---------------------------------------------------------------------------

extern "C" int superroc_fp4_slot_sets_configure(int n) {
    // Pre-sizing, never a refusal. Ensures at least n sets exist so
    // a server's sessions never grow one by one; grows only, so existing
    // owners keep their indices (never renumbers) and repeating with any n
    // is a no-op that still returns OK.
    if (n < 1) n = 1;
    if (n > SUPERROC_FP4_LS1_MAX_SETS) n = SUPERROC_FP4_LS1_MAX_SETS;
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    if (mgr.sets.size() < (size_t) n) {
        mgr.sets.resize((size_t) n);
        mgr.busy.resize((size_t) n, 0);
    }
    mgr.configured = true;
    return SUPERROC_FP4_OK;
}

extern "C" int superroc_fp4_slot_sets_count(void) {
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    return (int) mgr.sets.size();
}

extern "C" int superroc_fp4_slot_stream_attach(hipStream_t stream) {
    return superroc_fp4_slots_for_stream_at(stream, __func__, 0, false);
}

extern "C" int superroc_fp4_slot_stream_attach_at(hipStream_t stream, const char * func, long width) {
    return superroc_fp4_slots_for_stream_at(stream, func, width, false);
}

extern "C" int superroc_fp4_slot_stream_attach_grow(hipStream_t stream, const char * func, long width) {
    return superroc_fp4_slots_for_stream_at(stream, func, width, true);
}

extern "C" int superroc_fp4_slot_stream_attach_wait(hipStream_t stream) {
    return superroc_fp4_slots_for_stream_wait(stream);
}

// A helper stream of a session uses that session's set. The
// owner must already be attached (sessions pre-attach at load); the helper
// must not own a set of its own, and the owner must not itself be aliased
// (one hop only). Idempotent for the same pair. Returns the owner's set.
extern "C" int superroc_fp4_slot_stream_alias(hipStream_t helper, hipStream_t owner) {
    if (helper == nullptr || owner == nullptr || helper == owner) {
        return SUPERROC_FP4_ERR_ARG;
    }
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    const std::map<hipStream_t, int>::const_iterator own = mgr.stream_set.find(owner);
    if (own == mgr.stream_set.end()) {
        return SUPERROC_FP4_ERR_ARG;
    }
    const std::map<hipStream_t, hipStream_t>::const_iterator chained = mgr.alias.find(owner);
    if (chained != mgr.alias.end()) {
        return SUPERROC_FP4_ERR_ARG;
    }
    const std::map<hipStream_t, int>::const_iterator hs = mgr.stream_set.find(helper);
    if (hs != mgr.stream_set.end() && hs->second != own->second) {
        return SUPERROC_FP4_ERR_ARG;
    }
    const std::map<hipStream_t, hipStream_t>::const_iterator ha = mgr.alias.find(helper);
    if (ha != mgr.alias.end() && ha->second != owner) {
        return SUPERROC_FP4_ERR_ARG;
    }
    mgr.alias[helper] = owner;
    return own->second;
}

extern "C" int superroc_fp4_slot_stream_detach(hipStream_t stream) {
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    bool erased = false;
    {
        std::lock_guard<std::mutex> lk(mgr.mu);
        const std::map<hipStream_t, int>::iterator it = mgr.stream_set.find(stream);
        if (it != mgr.stream_set.end()) {
            mgr.busy[(size_t) it->second] = 0;
            mgr.stream_set.erase(it);
            erased = true;
        }
        // A destroyed executor detaches its stream AND its
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
    if (!erased) return SUPERROC_FP4_ERR_ARG;
    mgr.cv.notify_all();
    return SUPERROC_FP4_OK;
}

extern "C" unsigned long superroc_fp4_slot_alloc_count(void) {
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    return mgr.alloc_count;
}

extern "C" void superroc_fp4_slot_count_alloc(void) {
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    mgr.alloc_count++;
}

extern "C" unsigned long superroc_fp4_slot_wait_count(void) {
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    return mgr.wait_count;
}

extern "C" unsigned long superroc_fp4_slot_wait_us(void) {
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    return mgr.wait_us;
}

extern "C" unsigned long superroc_fp4_slot_refuse_count(void) {
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    return mgr.refuse_count;
}

extern "C" int superroc_fp4_slot_debug_map_size(void) {
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    return (int) mgr.stream_set.size();
}

extern "C" int superroc_fp4_slot_debug_alias_size(void) {
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    std::lock_guard<std::mutex> lk(mgr.mu);
    return (int) mgr.alias.size();
}

// Pre-grow every set's five Q8 slots plus the verify pair, launching the same
// quant kernel warmup always launched. Runs at load, outside any capture.
extern "C" int superroc_fp4_slots_warm_q8(int k, hipStream_t stream) {
    if (k <= 0 || (k % 32) != 0) return SUPERROC_FP4_ERR_ARG;
    // The verify GEMV quantises m<=8 rows into a buffer grown here,
    // at load, so a captured step never hipMallocs. A reserve failure is
    // ignored exactly as before: the verify call falls back per call with
    // VERIFY_M_SHORT, while the warmed slots still serve.
    if (k <= INT_MAX / 8) {
        (void) superroc_fp4_verify_m_reserve(k * 8);
    }
    float * zeros = nullptr;
    if (hipMalloc(&zeros, (size_t) k * sizeof(float)) != hipSuccess) {
        return SUPERROC_FP4_ERR_HIP;
    }
    if (hipMemset(zeros, 0, (size_t) k * sizeof(float)) != hipSuccess) {
        (void) hipFree(zeros);
        return SUPERROC_FP4_ERR_HIP;
    }
    int rc = SUPERROC_FP4_OK;
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    int nsets = 0;
    {
        std::lock_guard<std::mutex> lk(mgr.mu);
        nsets = (int) mgr.sets.size();
    }
    for (int s = 0; s < nsets && rc == SUPERROC_FP4_OK; ++s) {
        for (int slot = 0; slot < SUPERROC_FP4_LS1_NSLOT && rc == SUPERROC_FP4_OK; ++slot) {
            int8_t * qs = nullptr;
            float * ds = nullptr;
            rc = superroc_fp4_slots_grow_q8(s, slot, k, stream, &qs, &ds);
            if (rc != SUPERROC_FP4_OK) break;
            const int nb = k / 32;
            const int threads = 256;
            const int blocks = (nb + threads - 1) / threads;
            superroc_fp4_k_q8_1_quant<<<blocks, threads, 0, stream>>>(zeros, qs, ds, nb);
            rc = ls1_chk("slot warm quant");
        }
    }
    if (rc == SUPERROC_FP4_OK && hipStreamSynchronize(stream) != hipSuccess) {
        rc = SUPERROC_FP4_ERR_HIP;
    }
    (void) hipFree(zeros);
    return rc;
}

// Pre-grow every set's single-token and small-m MoE scratch plus the m-pair.
// Zero sizes are a no-op (dense models have no experts). Alloc-only: the
// kernels fill hs on first use, exactly as the lazy path did.
extern "C" int superroc_fp4_slots_warm_moe(int hs_elems, int hs_m_elems, int m_total) {
    if (hs_elems < 0 || hs_m_elems < 0 || m_total < 0) return SUPERROC_FP4_ERR_ARG;
    if ((m_total % 32) != 0) return SUPERROC_FP4_ERR_ARG;
    SuperROCFp4SlotMgr & mgr = superroc_fp4_slots_mgr();
    int nsets = 0;
    {
        std::lock_guard<std::mutex> lk(mgr.mu);
        nsets = (int) mgr.sets.size();
    }
    for (int s = 0; s < nsets; ++s) {
        if (hs_elems > 0) {
            float * hs = nullptr;
            int * mrc = nullptr;
            const int rc = superroc_fp4_slots_grow_hs(s, (size_t) hs_elems, nullptr, &hs, &mrc);
            if (rc != SUPERROC_FP4_OK) return rc;
        }
        if (hs_m_elems > 0) {
            float * hs = nullptr;
            int * mrc = nullptr;
            const int rc = superroc_fp4_slots_grow_hsm(s, (size_t) hs_m_elems, nullptr, &hs, &mrc);
            if (rc != SUPERROC_FP4_OK) return rc;
        }
        if (m_total > 0) {
            int8_t * qs = nullptr;
            float * ds = nullptr;
            const int rc = superroc_fp4_slots_grow_q8m(s, m_total, nullptr, &qs, &ds);
            if (rc != SUPERROC_FP4_OK) return rc;
        }
    }
    return SUPERROC_FP4_OK;
}

}  // namespace superroc_fp4_ks
