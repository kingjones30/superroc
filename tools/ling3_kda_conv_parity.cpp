// tools/ling3_kda_conv_parity.cpp — LC1 (wave 3, 2026-10-03): is the token-parallel KDA
// conv BIT-IDENTICAL to build28g's serial one, and how much faster is it at n_tokens 512?
//
// The "old" side is a VERBATIM copy of build28g's conv_silu_kernel + conv_silu_body
// (gufo-ling-int-28g/src/models/ling3/kernels/ling3_kda.hip), compiled into this binary as
// the reference. The "new" side is ling3::kda_conv_silu_qkv / ling3::kda_conv_silu out of
// kernels/ling3_kda.hip — the code that ships. LG1's parity test called kda_conv_silu_qkv
// for BOTH sides, so once its change was in it compared the new path with itself
// (./LG1-parity-test-VACUOUS.cpp in the lane dir); this one also asserts
// ling3::kda_conv_silu_last_path(), so a silent fallback to the serial kernel cannot pass.
//
// Gates (every one must print IDENTICAL):
//   n_tokens  1, 2, 3, 4, 5, 64, 511, 512   (2 is the state-update case LG1 got wrong:
//             28g reads st[n_tokens + tap], LG1's pass 2 wrote st[tap])
//   d_conv    the model's — bailingmoe3.ssm.conv_kernel = 4 in the GGUF the seat serves,
//             which is what MakeKdaShape() hands the kernels — and 3/4/5 for the general
//             kernel
//   state     zeroed and random, with the q, k and v histories filled from DIFFERENT random
//             values (a shared history hides a missing which-slice offset — LG1's first bug)
//   n_seqs    1 and 2
//   every output float equal, and the whole conv state memcmp-equal, byte for byte
// then ms/call old vs new at n_tokens = 512 with the model's d_inner (n_head 32 * head_dim
// 128 = 4096).
//
// Every buffer is sized from the geometry it is indexed with — state bytes go into a
// state-sized host buffer, never a y-sized one (LG1's test did that and page-faulted the
// GPU at 15:28Z) — and the same index audit the launchers run is repeated on the host
// before each launch and printed.
//
// Build: target gufo_ling3_kda_conv_parity (src/models/ling3/CMakeLists.txt). gpu-run only.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#include "src/models/ling3/kernels/ling3_kda.hpp"

namespace {

int g_fail = 0;
int g_audit_prints = 0;   // the audit prints its first few passes, then goes quiet

#define HIPCK(expr)                                                             \
    do {                                                                        \
        hipError_t _e = (expr);                                                 \
        if (_e != hipSuccess) {                                                 \
            std::fprintf(stderr, "FAIL %s:%d %s: %s\n", __FILE__, __LINE__,     \
                         #expr, hipGetErrorString(_e));                         \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// deterministic fills — one stream per buffer, so q, k and v never share values
// ---------------------------------------------------------------------------
std::uint64_t g_rng = 0x243F6A8885A308D3ULL;
void Seed(std::uint64_t s) { g_rng = s; }
float Nf(float lo, float hi) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return lo + (hi - lo) * (float) (g_rng >> 11) / 9007199254740992.0f;
}
void Fill(float* p, std::size_t n, float lo, float hi) {
    for (std::size_t i = 0; i < n; ++i) p[i] = Nf(lo, hi);
}

// ---------------------------------------------------------------------------
// device buffers, each sized from the geometry it is indexed with
// ---------------------------------------------------------------------------
struct Buf {
    void* p = nullptr;
    std::size_t bytes = 0;
    explicit Buf(std::size_t b) : bytes(b == 0 ? 1 : b) {
        if (hipMalloc(&p, bytes) != hipSuccess) {
            std::fprintf(stderr, "FAIL hipMalloc(%zu)\n", bytes);
            std::abort();
        }
        if (hipMemset(p, 0, bytes) != hipSuccess) std::abort();
    }
    ~Buf() {
        if (p != nullptr && hipFree(p) != hipSuccess) {
            std::fprintf(stderr, "FAIL hipFree(%zu)\n", bytes);
        }
    }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
};

// ---------------------------------------------------------------------------
// VERBATIM copy of build28g's conv_silu_kernel + conv_silu_body (gufo-ling-int-28g,
// src/models/ling3/kernels/ling3_kda.hip, the conv_silu_body / conv_silu_kernel pair),
// silu_f included. This is the oracle: do not "fix" or reformat anything inside it. It
// sits in an anonymous namespace so its names do not collide with the ones in
// kernels/ling3_kda.hip.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ void conv_silu_body(
        const float * __restrict__ x, float * __restrict__ y,
        float * __restrict__ conv_state, const float * __restrict__ w,
        int d_conv, int d_inner, int n_tokens,
        int64_t x_tok, int64_t x_seq, int64_t y_tok, int64_t y_seq,
        int c, int s) {
    const int hist = d_conv - 1;
    float *       st = conv_state + (int64_t) s * 3 * hist * d_inner + (int64_t) c * hist;
    const float * xr = x + (int64_t) s * x_seq;
    float *       yr = y + (int64_t) s * y_seq;

    for (int t = 0; t < n_tokens; ++t) {
        float acc = 0.0f;
        for (int tap = 0; tap < d_conv; ++tap) {
            const int64_t p = (int64_t) t + tap;
            const float sv = p < hist ? st[p] : xr[c + (p - hist) * x_tok];
            acc += w[tap + (int64_t) c * d_conv] * sv;
        }
        yr[c + (int64_t) t * y_tok] = silu_f(acc);
    }
    for (int tap = 0; tap < hist; ++tap) {
        const int64_t p = (int64_t) n_tokens + tap;
        st[tap] = p < hist ? st[p] : xr[c + (p - hist) * x_tok];   // <-- st[p], the fix
    }
}

static __global__ void conv_silu_kernel(const float * __restrict__ xq,
                                        const float * __restrict__ xk,
                                        const float * __restrict__ xv,
                                        float * __restrict__ yq,
                                        float * __restrict__ yk,
                                        float * __restrict__ yv,
                                        float * __restrict__ conv_state,
                                        const float * __restrict__ wq,
                                        const float * __restrict__ wk,
                                        const float * __restrict__ wv,
                                        int d_conv, int d_inner, int n_tokens,
                                        int64_t qi_tok, int64_t ki_tok, int64_t vi_tok,
                                        int64_t qi_seq,
                                        int64_t qo_tok, int64_t ko_tok, int64_t vo_tok,
                                        int64_t qo_seq) {
    const int c = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= d_inner) return;                     // no __syncthreads in this kernel
    const int which = (int) blockIdx.z;          // 0 = q, 1 = k, 2 = v
    const int s = (int) blockIdx.y;
    const int hist = d_conv - 1;
    float * st = conv_state + (size_t) which * (size_t) hist * (size_t) d_inner;
    switch (which) {
        case 0: conv_silu_body(xq, yq, st, wq, d_conv, d_inner, n_tokens, qi_tok, qi_seq, qo_tok, qo_seq, c, s); break;
        case 1: conv_silu_body(xk, yk, st, wk, d_conv, d_inner, n_tokens, ki_tok, qi_seq, ko_tok, qo_seq, c, s); break;
        default: conv_silu_body(xv, yv, st, wv, d_conv, d_inner, n_tokens, vi_tok, qi_seq, vo_tok, qo_seq, c, s); break;
    }
}
// ---------------------------------------------------------------------------
// end of the verbatim copy
// ---------------------------------------------------------------------------

// The serial launch exactly as kda_conv_silu_qkv makes it (before LC1's split).
void RefQkvLaunch(const ::ling3::KdaShape& sh,
                  const ::ling3::ViewC& q, const ::ling3::ViewC& k, const ::ling3::ViewC& v,
                  const ::ling3::View& qo, const ::ling3::View& ko, const ::ling3::View& vo,
                  const float* wq, const float* wk, const float* wv,
                  float* conv_state, int n_tokens, int n_seqs, hipStream_t stream) {
    const int d_inner = sh.d_inner();
    const int bs = 256;
    const dim3 grid((d_inner + bs - 1) / bs, n_seqs, 3);
    hipLaunchKernelGGL(conv_silu_kernel, grid, bs, 0, stream,
                       q.p, k.p, v.p, qo.p, ko.p, vo.p, conv_state, wq, wk, wv,
                       sh.d_conv, d_inner, n_tokens,
                       q.s_tok, k.s_tok, v.s_tok, q.s_seq,
                       qo.s_tok, ko.s_tok, vo.s_tok, qo.s_seq);
    HIPCK(hipGetLastError());
}

// The one-stream serial launch exactly as kda_conv_silu makes it (gridDim.z == 1, so the
// kernel takes slice 0 of whatever st it is given).
void RefOneLaunch(const ::ling3::KdaShape& sh,
                  const ::ling3::ViewC& in, const ::ling3::View& out,
                  const float* w, float* st, int n_tokens, int n_seqs, hipStream_t stream) {
    const int d_inner = sh.d_inner();
    const int bs = 256;
    const dim3 grid((d_inner + bs - 1) / bs, n_seqs, 1);
    hipLaunchKernelGGL(conv_silu_kernel, grid, bs, 0, stream,
                       in.p, in.p, in.p, out.p, out.p, out.p, st, w, w, w,
                       sh.d_conv, d_inner, n_tokens,
                       in.s_tok, in.s_tok, in.s_tok, in.s_seq,
                       out.s_tok, out.s_tok, out.s_tok, out.s_seq);
    HIPCK(hipGetLastError());
}

// ---------------------------------------------------------------------------
// host-side index audit — the same extents conv_silu_stream_ok computes in
// kernels/ling3_kda.hip, checked here against the buffers this test owns.
// ---------------------------------------------------------------------------
bool HostAudit(const char* what, std::size_t alloc_bytes, int64_t n_elems) {
    const int64_t need = n_elems * (int64_t) sizeof(float);
    if (n_elems <= 0 || need <= 0 || (std::size_t) need > alloc_bytes) {
        std::printf("  AUDIT FAIL %s: %lld floats need %lld bytes, allocation is %zu\n",
                    what, (long long) n_elems, (long long) need, alloc_bytes);
        ++g_fail;
        return false;
    }
    if (g_audit_prints < 4) {
        std::printf("  AUDIT %s: %lld floats / %lld bytes <= %zu\n",
                    what, (long long) n_elems, (long long) need, alloc_bytes);
        ++g_audit_prints;
    }
    return true;
}

struct Case {
    int n_tokens = 1;
    int d_conv = 4;
    int n_seqs = 1;
    bool history = true;
};

// One parity case: the verbatim reference vs the shipped dispatcher.
void RunCase(const Case& cs, int d_inner, hipStream_t stream) {
    const int n_tokens = cs.n_tokens;
    const int d_conv = cs.d_conv;
    const int n_seqs = cs.n_seqs;
    const int hist = d_conv - 1;

    ::ling3::KdaShape sh;
    sh.n_head = 32;          // bailingmoe3.attention.head_count
    sh.head_dim = 128;       // bailingmoe3.kda.head_dim  ->  d_inner 4096
    sh.d_conv = d_conv;
    if (sh.d_inner() != d_inner) {
        std::printf("  FAIL: this test's d_inner %d != n_head*head_dim %d\n", d_inner, sh.d_inner());
        ++g_fail;
        return;
    }

    const std::size_t x_bytes = (std::size_t) d_inner * (std::size_t) n_tokens *
                                (std::size_t) n_seqs * sizeof(float);
    const std::size_t y_bytes = x_bytes;
    const std::size_t w_bytes = (std::size_t) d_conv * (std::size_t) d_inner * sizeof(float);
    const std::size_t s_bytes = (std::size_t) 3 * (std::size_t) hist * (std::size_t) d_inner *
                                (std::size_t) n_seqs * sizeof(float);
    const int64_t x_tok = d_inner;
    const int64_t y_tok = d_inner;
    const int64_t x_seq = (int64_t) d_inner * n_tokens;
    const int64_t y_seq = x_seq;

    // host-side audit of every index the kernels compute, against these sizes
    const int64_t xneed = (d_inner - 1) + (int64_t) (n_tokens - 1) * x_tok +
                          (int64_t) (n_seqs - 1) * x_seq + 1;
    const int64_t yneed = xneed;
    const int64_t wneed = (int64_t) d_conv * d_inner;
    const int64_t sneed = (int64_t) n_seqs * 3 * (int64_t) hist * d_inner;
    bool audited = true;
    audited = HostAudit("x", x_bytes, xneed) && audited;
    audited = HostAudit("y", y_bytes, yneed) && audited;
    audited = HostAudit("w", w_bytes, wneed) && audited;
    // the state layer is the SECOND of two in the arena: offset + extent must fit
    audited = HostAudit("state@+layer1", 2 * s_bytes,
                        (int64_t) (s_bytes / sizeof(float)) + sneed) && audited;
    if (!audited) return;

    Buf xq(x_bytes), xk(x_bytes), xv(x_bytes);
    Buf yq0(y_bytes), yk0(y_bytes), yv0(y_bytes);
    Buf yq1(y_bytes), yk1(y_bytes), yv1(y_bytes);
    // The conv state lives INSIDE a two-layer arena, like the engine's kda_conv_device (one
    // slice per layer), and both sides are handed the SECOND slice. That way the launchers'
    // audit sees a sub-pointer exactly as it does in the engine, and the first slice is a
    // guard: if either kernel writes outside the layer it owns, the guard moves.
    const std::size_t s_arena = 2 * s_bytes;
    Buf st0(s_arena), st1(s_arena);
    Buf wq(w_bytes), wk(w_bytes), wv(w_bytes);

    std::vector<float> hxq(x_bytes / sizeof(float)), hxk(x_bytes / sizeof(float)),
                       hxv(x_bytes / sizeof(float));
    std::vector<float> hwq(w_bytes / sizeof(float)), hwk(w_bytes / sizeof(float)),
                       hwv(w_bytes / sizeof(float));
    std::vector<float> hs(s_arena / sizeof(float));
    std::vector<float> hy0(y_bytes / sizeof(float)), hy1(y_bytes / sizeof(float));
    std::vector<float> hst0(s_arena / sizeof(float)), hst1(s_arena / sizeof(float));

    Seed(0xC0FFEE0DULL ^ ((std::uint64_t) n_tokens << 20) ^ ((std::uint64_t) d_conv << 8) ^
         (std::uint64_t) n_seqs);
    Fill(hxq.data(), hxq.size(), -1.0f, 1.0f);
    Fill(hxk.data(), hxk.size(), -1.2f, 0.8f);
    Fill(hxv.data(), hxv.size(), -0.7f, 1.3f);
    Fill(hwq.data(), hwq.size(), -0.5f, 0.5f);
    Fill(hwk.data(), hwk.size(), -0.6f, 0.4f);
    Fill(hwv.data(), hwv.size(), -0.4f, 0.6f);

    // guard layer first: a sentinel that would show any write outside the layer under test
    for (std::size_t i = 0; i < s_bytes / sizeof(float); ++i) hs[i] = 1234.5f;
    // q, k and v histories are DIFFERENT random values (and, when history is off, zero).
    const std::size_t slice = (std::size_t) hist * (std::size_t) d_inner;
    float* layer = hs.data() + s_bytes / sizeof(float);
    for (int s = 0; s < n_seqs; ++s) {
        float* row = layer + (std::size_t) s * 3 * slice;
        if (cs.history) {
            Fill(row + 0 * slice, slice, -0.50f, 0.50f);   // q
            Fill(row + 1 * slice, slice, 0.25f, 1.25f);    // k
            Fill(row + 2 * slice, slice, -1.25f, -0.25f);  // v
        } else {
            std::memset(row, 0, 3 * slice * sizeof(float));
        }
    }

    HIPCK(hipMemcpy(xq.p, hxq.data(), x_bytes, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(xk.p, hxk.data(), x_bytes, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(xv.p, hxv.data(), x_bytes, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(wq.p, hwq.data(), w_bytes, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(wk.p, hwk.data(), w_bytes, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(wv.p, hwv.data(), w_bytes, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(st0.p, hs.data(), s_arena, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(st1.p, hs.data(), s_arena, hipMemcpyHostToDevice));
    float* s0 = reinterpret_cast<float*>(static_cast<char*>(st0.p) + s_bytes);
    float* s1 = reinterpret_cast<float*>(static_cast<char*>(st1.p) + s_bytes);

    auto qc = ::ling3::ViewC{static_cast<const float*>(xq.p), x_tok, x_seq};
    auto kc = ::ling3::ViewC{static_cast<const float*>(xk.p), x_tok, x_seq};
    auto vc = ::ling3::ViewC{static_cast<const float*>(xv.p), x_tok, x_seq};
    auto qo0 = ::ling3::View{static_cast<float*>(yq0.p), y_tok, y_seq};
    auto ko0 = ::ling3::View{static_cast<float*>(yk0.p), y_tok, y_seq};
    auto vo0 = ::ling3::View{static_cast<float*>(yv0.p), y_tok, y_seq};
    auto qo1 = ::ling3::View{static_cast<float*>(yq1.p), y_tok, y_seq};
    auto ko1 = ::ling3::View{static_cast<float*>(yk1.p), y_tok, y_seq};
    auto vo1 = ::ling3::View{static_cast<float*>(yv1.p), y_tok, y_seq};

    // old: the verbatim reference. new: the shipped dispatcher. Both get the sub-pointer.
    RefQkvLaunch(sh, qc, kc, vc, qo0, ko0, vo0,
                 static_cast<const float*>(wq.p), static_cast<const float*>(wk.p),
                 static_cast<const float*>(wv.p),
                 s0, n_tokens, n_seqs, stream);
    HIPCK(hipStreamSynchronize(stream));

    ::ling3::kda_conv_silu_qkv(sh, qc, kc, vc, qo1, ko1, vo1,
                               static_cast<const float*>(wq.p),
                               static_cast<const float*>(wk.p),
                               static_cast<const float*>(wv.p),
                               s1, n_tokens, n_seqs, stream);
    HIPCK(hipStreamSynchronize(stream));

    const int want_path = n_tokens > 1 ? 1 : 0;
    const int got_path = ::ling3::kda_conv_silu_last_path();
    if (got_path != want_path) {
        std::printf("  FAIL path: n_tokens=%d ran path %d, expected %d "
                    "(the prefill pair did not run — this comparison would be vacuous)\n",
                    n_tokens, got_path, want_path);
        ++g_fail;
    }

    std::printf("n_tokens=%-4d d_conv=%d n_seqs=%d history=%d path=%d\n",
                n_tokens, d_conv, n_seqs, (int) cs.history, got_path);
    auto cmp = [&](const char* tag, void* a, void* b, std::size_t bytes) {
        HIPCK(hipMemcpy(hy0.data(), a, bytes, hipMemcpyDeviceToHost));
        HIPCK(hipMemcpy(hy1.data(), b, bytes, hipMemcpyDeviceToHost));
        const std::size_t n = bytes / sizeof(float);
        std::size_t bad = 0;
        std::size_t first = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (hy0[i] != hy1[i]) {
                if (bad == 0) first = i;
                ++bad;
            }
        }
        if (bad == 0) {
            std::printf("  %-3s IDENTICAL (%zu floats)\n", tag, n);
        } else {
            std::printf("  %-3s DIFF %zu/%zu floats, first [%zu] ref=%.9g new=%.9g\n",
                        tag, bad, n, first, (double) hy0[first], (double) hy1[first]);
            ++g_fail;
        }
    };
    cmp("yq", yq0.p, yq1.p, y_bytes);
    cmp("yk", yk0.p, yk1.p, y_bytes);
    cmp("yv", yv0.p, yv1.p, y_bytes);

    // the whole state arena, byte for byte, into arena-sized host buffers
    HIPCK(hipMemcpy(hst0.data(), st0.p, s_arena, hipMemcpyDeviceToHost));
    HIPCK(hipMemcpy(hst1.data(), st1.p, s_arena, hipMemcpyDeviceToHost));
    if (std::memcmp(hst0.data(), hst1.data(), s_arena) == 0) {
        std::printf("  st  IDENTICAL (%zu floats, memcmp)\n", s_bytes / sizeof(float));
    } else {
        const std::size_t n = s_arena / sizeof(float);
        std::size_t bad = 0, first = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (hst0[i] != hst1[i]) {
                if (bad == 0) first = i;
                ++bad;
            }
        }
        std::printf("  st  DIFF %zu/%zu floats, first [%zu] ref=%.9g new=%.9g\n",
                    bad, n, first, (double) hst0[first], (double) hst1[first]);
        ++g_fail;
    }
    // the guard layer must still read 1234.5 everywhere on BOTH sides
    std::size_t guard_bad = 0;
    for (std::size_t i = 0; i < s_bytes / sizeof(float); ++i) {
        guard_bad += (hst0[i] != 1234.5f) + (hst1[i] != 1234.5f);
    }
    if (guard_bad == 0) {
        std::printf("  gd  UNTOUCHED (%zu floats, sentinel 1234.5)\n", s_bytes / sizeof(float));
    } else {
        std::printf("  gd  MOVED %zu writes into the guard layer\n", guard_bad);
        ++g_fail;
    }
}

// The one-stream entry point, against the same reference with the same qkv state offset.
void RunOneStreamCase(int n_tokens, int d_conv, int n_seqs, int qkv, hipStream_t stream) {
    const int d_inner = 32 * 128;
    const int hist = d_conv - 1;
    ::ling3::KdaShape sh;
    sh.n_head = 32;
    sh.head_dim = 128;
    sh.d_conv = d_conv;

    const std::size_t x_bytes = (std::size_t) d_inner * (std::size_t) n_tokens *
                                (std::size_t) n_seqs * sizeof(float);
    const std::size_t w_bytes = (std::size_t) d_conv * (std::size_t) d_inner * sizeof(float);
    const std::size_t s_bytes = (std::size_t) 3 * (std::size_t) hist * (std::size_t) d_inner *
                                (std::size_t) n_seqs * sizeof(float);
    const int64_t x_tok = d_inner;
    const int64_t x_seq = (int64_t) d_inner * n_tokens;

    Buf x(x_bytes), y0(x_bytes), y1(x_bytes), st0(s_bytes), st1(s_bytes), w(w_bytes);
    std::vector<float> hx(x_bytes / sizeof(float)), hw(w_bytes / sizeof(float)),
                       hs(s_bytes / sizeof(float));
    std::vector<float> hy0(x_bytes / sizeof(float)), hy1(x_bytes / sizeof(float));
    std::vector<float> hst0(s_bytes / sizeof(float)), hst1(s_bytes / sizeof(float));
    Seed(0xBADC0DEULL ^ (std::uint64_t) n_tokens);
    Fill(hx.data(), hx.size(), -1.0f, 1.0f);
    Fill(hw.data(), hw.size(), -0.5f, 0.5f);
    Fill(hs.data(), hs.size(), -0.75f, 0.75f);
    HIPCK(hipMemcpy(x.p, hx.data(), x_bytes, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(w.p, hw.data(), w_bytes, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(st0.p, hs.data(), s_bytes, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(st1.p, hs.data(), s_bytes, hipMemcpyHostToDevice));

    auto in = ::ling3::ViewC{static_cast<const float*>(x.p), x_tok, x_seq};
    auto o0 = ::ling3::View{static_cast<float*>(y0.p), x_tok, x_seq};
    auto o1 = ::ling3::View{static_cast<float*>(y1.p), x_tok, x_seq};
    const std::size_t soff = (std::size_t) qkv * (std::size_t) hist * (std::size_t) d_inner *
                             sizeof(float);
    float* s0 = reinterpret_cast<float*>(static_cast<char*>(st0.p) + soff);

    RefOneLaunch(sh, in, o0, static_cast<const float*>(w.p), s0, n_tokens, n_seqs, stream);
    HIPCK(hipStreamSynchronize(stream));
    ::ling3::kda_conv_silu(sh, static_cast<const float*>(w.p), in, o1,
                           static_cast<float*>(st1.p), qkv, n_tokens, n_seqs, stream);
    HIPCK(hipStreamSynchronize(stream));

    const int want_path = n_tokens > 1 ? 1 : 0;
    const int got_path = ::ling3::kda_conv_silu_last_path();
    HIPCK(hipMemcpy(hy0.data(), y0.p, x_bytes, hipMemcpyDeviceToHost));
    HIPCK(hipMemcpy(hy1.data(), y1.p, x_bytes, hipMemcpyDeviceToHost));
    std::size_t bad = 0;
    for (std::size_t i = 0; i < hy0.size(); ++i) bad += (hy0[i] != hy1[i]);
    HIPCK(hipMemcpy(hst0.data(), st0.p, s_bytes, hipMemcpyDeviceToHost));
    HIPCK(hipMemcpy(hst1.data(), st1.p, s_bytes, hipMemcpyDeviceToHost));
    const bool st_ok = std::memcmp(hst0.data(), hst1.data(), s_bytes) == 0;
    std::printf("one-stream n_tokens=%d d_conv=%d n_seqs=%d qkv=%d path=%d: y %s (%zu floats), "
                "state %s (%zu floats, memcmp)\n",
                n_tokens, d_conv, n_seqs, qkv, got_path,
                bad == 0 ? "IDENTICAL" : "DIFF", hy0.size(),
                st_ok ? "IDENTICAL" : "DIFF", s_bytes / sizeof(float));
    if (bad != 0 || !st_ok || got_path != want_path) ++g_fail;
}

// ms per call, old (the verbatim serial reference) vs new (the shipped dispatcher), at the
// model's d_inner. Also prints the bytes each call moves and the bandwidth both reach, so
// "is this kernel bound by memory or by parallelism" is answered with numbers.
void Bench(int n_tokens, int d_conv, int d_inner, hipStream_t stream) {
    const int n_seqs = 1, hist = d_conv - 1;
    ::ling3::KdaShape sh;
    sh.n_head = 32;
    sh.head_dim = 128;
    sh.d_conv = d_conv;

    const std::size_t x_bytes = (std::size_t) d_inner * (std::size_t) n_tokens *
                                (std::size_t) n_seqs * sizeof(float);
    const std::size_t w_bytes = (std::size_t) d_conv * (std::size_t) d_inner * sizeof(float);
    const std::size_t s_bytes = (std::size_t) 3 * (std::size_t) hist * (std::size_t) d_inner *
                                (std::size_t) n_seqs * sizeof(float);
    const int64_t x_tok = d_inner, x_seq = (int64_t) d_inner * n_tokens;

    Buf xq(x_bytes), xk(x_bytes), xv(x_bytes);
    Buf yq(x_bytes), yk(x_bytes), yv(x_bytes);
    Buf st0(s_bytes), st1(s_bytes);
    Buf wq(w_bytes), wk(w_bytes), wv(w_bytes);
    std::vector<float> hz(x_bytes / sizeof(float), 1.0f);
    std::vector<float> hw(w_bytes / sizeof(float), 0.25f);
    std::vector<float> hs(s_bytes / sizeof(float), 0.125f);
    HIPCK(hipMemcpy(xq.p, hz.data(), x_bytes, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(xk.p, hz.data(), x_bytes, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(xv.p, hz.data(), x_bytes, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(wq.p, hw.data(), w_bytes, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(wk.p, hw.data(), w_bytes, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(wv.p, hw.data(), w_bytes, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(st0.p, hs.data(), s_bytes, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(st1.p, hs.data(), s_bytes, hipMemcpyHostToDevice));

    auto qc = ::ling3::ViewC{static_cast<const float*>(xq.p), x_tok, x_seq};
    auto kc = ::ling3::ViewC{static_cast<const float*>(xk.p), x_tok, x_seq};
    auto vc = ::ling3::ViewC{static_cast<const float*>(xv.p), x_tok, x_seq};
    auto qo = ::ling3::View{static_cast<float*>(yq.p), x_tok, x_seq};
    auto ko = ::ling3::View{static_cast<float*>(yk.p), x_tok, x_seq};
    auto vo = ::ling3::View{static_cast<float*>(yv.p), x_tok, x_seq};
    const float* wqp = static_cast<const float*>(wq.p);
    const float* wkq = static_cast<const float*>(wk.p);
    const float* wvq = static_cast<const float*>(wv.p);

    const int reps = 100;
    hipEvent_t t0, t1;
    HIPCK(hipEventCreate(&t0));
    HIPCK(hipEventCreate(&t1));
    float ms_old = 0, ms_new = 0;

    RefQkvLaunch(sh, qc, kc, vc, qo, ko, vo, wqp, wkq, wvq,
                 static_cast<float*>(st0.p), n_tokens, n_seqs, stream);
    HIPCK(hipStreamSynchronize(stream));
    HIPCK(hipEventRecord(t0, stream));
    for (int i = 0; i < reps; ++i) {
        RefQkvLaunch(sh, qc, kc, vc, qo, ko, vo, wqp, wkq, wvq,
                     static_cast<float*>(st0.p), n_tokens, n_seqs, stream);
    }
    HIPCK(hipEventRecord(t1, stream));
    HIPCK(hipEventSynchronize(t1));
    HIPCK(hipEventElapsedTime(&ms_old, t0, t1));

    ::ling3::kda_conv_silu_qkv(sh, qc, kc, vc, qo, ko, vo, wqp, wkq, wvq,
                               static_cast<float*>(st1.p), n_tokens, n_seqs, stream);
    HIPCK(hipStreamSynchronize(stream));
    HIPCK(hipEventRecord(t0, stream));
    for (int i = 0; i < reps; ++i) {
        ::ling3::kda_conv_silu_qkv(sh, qc, kc, vc, qo, ko, vo, wqp, wkq, wvq,
                                   static_cast<float*>(st1.p), n_tokens, n_seqs, stream);
    }
    HIPCK(hipEventRecord(t1, stream));
    HIPCK(hipEventSynchronize(t1));
    HIPCK(hipEventElapsedTime(&ms_new, t0, t1));

    // Bytes one call moves: d_conv x-reads + one y-write per output element (3 streams).
    // That is what the SERIAL reference really fetches; the prefill path reads each input
    // once, so the second line is its own traffic -- the roofline that bounds it.
    const double outs = 3.0 * d_inner * n_tokens;
    const double bytes = outs * (double) d_conv * 4.0 + outs * 4.0;
    const double one_read =
        3.0 * d_inner * (n_tokens + hist) * 4.0 + 3.0 * d_inner * n_tokens * 4.0;
    std::printf("BENCH n_tokens=%d d_inner=%d d_conv=%d n_seqs=1 reps=%d: "
                "old %.4f ms/call, new %.4f ms/call, speedup %.2fx "
                "(old %.1f us, new %.1f us) | %.1f MB/call -> old %.1f GB/s, new %.1f GB/s\n",
                n_tokens, d_inner, d_conv, reps,
                ms_old / reps, ms_new / reps, ms_old / ms_new,
                1000.0 * ms_old / reps, 1000.0 * ms_new / reps,
                bytes / 1e6, bytes / (ms_old / reps) / 1e6, bytes / (ms_new / reps) / 1e6);
    std::printf("BENCH real n_tokens=%d d_conv=%d: prefill path %.1f MB/call (each input read "
                "once + each output written once) -> %.1f GB/s new, %.1f GB/s old at that "
                "traffic; serial reference really moves %.1f MB/call\n",
                n_tokens, d_conv, one_read / 1e6,
                one_read / (ms_new / reps) / 1e6, one_read / (ms_old / reps) / 1e6,
                bytes / 1e6);

    // The host-side index audit is INSIDE the timed new path (10 hipMemGetAddressRange
    // queries per call). Measure one query so the split between audit and kernels is known.
    {
        void* base = nullptr;
        std::size_t bytes = 0;
        void* probe = st1.p;
        const int q = 1000;
        hipError_t eq = hipSuccess;
        auto t0c = std::chrono::steady_clock::now();
        for (int i = 0; i < q; ++i) {
            eq = hipMemGetAddressRange(&base, &bytes, probe);
        }
        auto t1c = std::chrono::steady_clock::now();
        const double us =
            std::chrono::duration<double, std::micro>(t1c - t0c).count() / q;
        std::printf("BENCH audit: %.2f us per hipMemGetAddressRange, ~%.1f us of the new "
                    "path's %.1f us is the 10-query audit (rc=%d base=%p size=%zu)\n",
                    us, 10.0 * us, 1000.0 * ms_new / reps, (int) eq, base, bytes);
    }
    HIPCK(hipEventDestroy(t0));
    HIPCK(hipEventDestroy(t1));
}

}  // namespace

int main() {
    const int d_inner = 32 * 128;  // the model's: bailingmoe3.attention.head_count * kda.head_dim
    std::printf("LC1 KDA-conv parity: token-parallel prefill vs build28g's serial conv_silu_kernel\n");
    std::printf("  model: bailingmoe3.ssm.conv_kernel=4 (GGUF), head_count=32, kda.head_dim=128, "
                "d_inner=%d\n", d_inner);

    hipStream_t stream;
    HIPCK(hipStreamCreate(&stream));

    for (int d_conv : {4, 3, 5}) {
        for (int n_tokens : {1, 2, 3, 4, 5, 64, 511, 512}) {
            for (int n_seqs : {1, 2}) {
                Case a{n_tokens, d_conv, n_seqs, true};
                RunCase(a, d_inner, stream);
                Case b{n_tokens, d_conv, n_seqs, false};
                RunCase(b, d_inner, stream);
            }
        }
    }
    std::printf("one-stream entry points:\n");
    RunOneStreamCase(1, 4, 1, 1, stream);
    RunOneStreamCase(2, 4, 1, 2, stream);
    RunOneStreamCase(5, 4, 2, 0, stream);
    RunOneStreamCase(512, 4, 1, 1, stream);

    std::printf("benchmark (model d_inner). n_tokens 2 is almost pure fixed cost (the state pass\n"
                "plus two launches), so the rows split the passes: T(n) = state-pass + slope*n.\n");
    // nt == 1 is every DECODE step: it keeps the serial kernel (so the same per-output
    // statement), which makes this row the A/B of the streaming y store on the decode path --
    // the one path a decode regression would hide in.
    for (int nt : {1, 2, 8, 64, 512}) Bench(nt, 4, d_inner, stream);
    // d_conv 2 halves the per-output window: it is the traffic probe (see REPORT).
    Bench(512, 2, d_inner, stream);

    HIPCK(hipStreamDestroy(stream));
    if (g_fail == 0) {
        std::printf("RESULT PASS\n");
    } else {
        std::printf("RESULT FAIL (%d)\n", g_fail);
    }
    return g_fail == 0 ? 0 : 2;
}
