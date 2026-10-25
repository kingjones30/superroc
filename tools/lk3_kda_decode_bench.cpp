// Decode KDA scan: bit-identity vs the reference kernel, and µs per call.
// Build: hipcc -O3 -ffast-math -fno-finite-math-only --offload-arch=gfx1151 -std=c++17
//        -I<kernels> ling3_kda.hip lk3_kda_decode_bench.cpp -o kda_bench
// Run inside gpu-run. Usage: kda_bench gate | time | replay <file>

#include "ling3_kda.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed) {}
    uint32_t next() {
        s = s * 1664525u + 1013904223u;
        return s;
    }
    float span(float a, float b) {
        const float u = (float) (next() >> 8) * (1.0f / 16777216.0f);
        return a + (b - a) * u;
    }
};

struct Stat {
    size_t nbad = 0;
    size_t first = 0;
    float max_abs = 0.f;
    float max_rel = 0.f;
};

Stat compare(const float * got, const float * ref, size_t n) {
    Stat st;
    for (size_t i = 0; i < n; ++i) {
        uint32_t ag = 0, ar = 0;
        std::memcpy(&ag, got + i, 4);
        std::memcpy(&ar, ref + i, 4);
        if (ag == ar) continue;
        if (st.nbad == 0) st.first = i;
        st.nbad += 1;
        const float d = std::fabs(got[i] - ref[i]);
        if (d > st.max_abs) st.max_abs = d;
        const float den = std::fabs(ref[i]);
        const float r = den > 0.f ? d / den : d;
        if (r > st.max_rel) st.max_rel = r;
    }
    return st;
}

void report(const char * name, const Stat & a, const Stat & b, int fast_delta) {
    const int bad = (a.nbad != 0 || b.nbad != 0 || fast_delta <= 0) ? 1 : 0;
    std::printf("GATE %s bad=%d fast_launches=+%d out_bad=%zu out_max_abs=%.3e out_max_rel=%.3e "
                "state_bad=%zu state_max_abs=%.3e state_max_rel=%.3e first_out=%zu first_state=%zu\n",
                name, bad, fast_delta, a.nbad, a.max_abs, a.max_rel, b.nbad, b.max_abs, b.max_rel,
                a.first, b.first);
    std::fflush(stdout);
}

struct Buf {
    int n_head = 32;
    int n_tokens = 1;
    int n_seqs = 1;
    int d_inner = 0;
    size_t n_q = 0;
    size_t n_beta = 0;
    size_t n_state = 0;
    std::vector<float> hq, hk, hv, hg, hb, hstate, hout;
    float *q = nullptr, *k = nullptr, *v = nullptr, *g = nullptr, *beta = nullptr;
    float *state = nullptr, *state_b = nullptr, *out = nullptr, *out_b = nullptr;
    hipStream_t stream = nullptr;

    explicit Buf(int nh, int nt, int ns) : n_head(nh), n_tokens(nt), n_seqs(ns) {
        d_inner = nh * 128;
        n_q = (size_t) d_inner * (size_t) nt * (size_t) ns;
        n_beta = (size_t) nh * (size_t) nt * (size_t) ns;
        n_state = (size_t) ns * (size_t) nh * 128u * 128u;
        hq.resize(n_q);
        hk.resize(n_q);
        hv.resize(n_q);
        hg.resize(n_q);
        hb.resize(n_beta);
        hstate.resize(n_state);
        hout.resize(n_q);
        auto alloc = [](size_t n) {
            float * p = nullptr;
            if (hipMalloc(&p, n * sizeof(float)) != hipSuccess) {
                std::fprintf(stderr, "hipMalloc %zu failed\n", n);
                std::exit(2);
            }
            return p;
        };
        q = alloc(n_q);
        k = alloc(n_q);
        v = alloc(n_q);
        g = alloc(n_q);
        beta = alloc(n_beta);
        state = alloc(n_state);
        state_b = alloc(n_state);
        out = alloc(n_q);
        out_b = alloc(n_q);
        if (hipStreamCreate(&stream) != hipSuccess) std::exit(2);
    }
    ~Buf() {
        hipFree(q);
        hipFree(k);
        hipFree(v);
        hipFree(g);
        hipFree(beta);
        hipFree(state);
        hipFree(state_b);
        hipFree(out);
        hipFree(out_b);
        if (stream) hipStreamDestroy(stream);
    }
    Buf(const Buf &) = delete;
    Buf & operator=(const Buf &) = delete;

    void h2d_inputs() {
        hipMemcpy(q, hq.data(), n_q * 4, hipMemcpyHostToDevice);
        hipMemcpy(k, hk.data(), n_q * 4, hipMemcpyHostToDevice);
        hipMemcpy(v, hv.data(), n_q * 4, hipMemcpyHostToDevice);
        hipMemcpy(g, hg.data(), n_q * 4, hipMemcpyHostToDevice);
        hipMemcpy(beta, hb.data(), n_beta * 4, hipMemcpyHostToDevice);
    }
    void h2d_state(float * dst) {
        hipMemcpy(dst, hstate.data(), n_state * 4, hipMemcpyHostToDevice);
    }
};

int audit_shape(int n_head, int n_seqs, int n_tokens) {
    const int HD = 128;
    const int d_inner = n_head * HD;
    const int64_t q_tok = d_inner;
    const int64_t q_seq = (int64_t) d_inner * n_tokens;
    const int64_t b_tok = n_head;
    const int64_t b_seq = (int64_t) n_head * n_tokens;
    const size_t n_q = (size_t) d_inner * (size_t) n_tokens * (size_t) n_seqs;
    const size_t n_beta = (size_t) n_head * (size_t) n_tokens * (size_t) n_seqs;
    const size_t n_state = (size_t) n_seqs * (size_t) n_head * (size_t) HD * HD;
    int64_t max_q = -1, max_b = -1, max_s = -1;
    for (int seq = 0; seq < n_seqs; ++seq) {
        for (int h = 0; h < n_head; ++h) {
            for (int t = 0; t < n_tokens; ++t) {
                const int64_t chan = (int64_t) h * HD;
                const int64_t iq = chan + (HD - 1) + (int64_t) t * q_tok + (int64_t) seq * q_seq;
                const int64_t ib = h + (int64_t) t * b_tok + (int64_t) seq * b_seq;
                if (iq > max_q) max_q = iq;
                if (ib > max_b) max_b = ib;
            }
            const int64_t hend = (((int64_t) seq * n_head + h) * HD + (HD - 1)) * HD + (HD - 1);
            if (hend > max_s) max_s = hend;
        }
    }
    bool f4 = true;
    int max_tile = -1;
    for (int m = 0; m < HD / 4; ++m) {
        for (int thr = 0; thr < HD; ++thr) {
            const int fidx = thr + m * HD;
            const int elem = fidx * 4;
            const int col = elem >> 7;
            const int i0 = elem & 127;
            if (col < 0 || col >= HD || i0 < 0 || i0 + 3 >= HD) f4 = false;
            for (int k = 0; k < 4; ++k) {
                const int tile = (i0 + k) * HD + col;
                if (tile < 0 || tile >= HD * HD) f4 = false;
                if (tile > max_tile) max_tile = tile;
            }
        }
    }
    bool t32ok = true;
    for (int z = 0; z < 4; ++z) {
        for (int lane = 0; lane < 32; ++lane) {
            const int col = z * 32 + lane;
            if (col < 0 || col >= HD) t32ok = false;
            if (lane * HD + (HD - 1) >= 32 * HD) t32ok = false;
        }
    }
    bool redok = true;
    for (int z = 0; z < 32; ++z) {
        for (int ty = 0; ty < 4; ++ty) {
            const int col = z * 4 + ty;
            if (col < 0 || col >= HD) redok = false;
            for (int lane = 0; lane < 32; ++lane) {
                for (int r = 0; r < 4; ++r) {
                    const int i = r * 32 + lane;
                    if (i < 0 || i >= HD) redok = false;
                }
            }
        }
    }
    const bool ok = max_q == (int64_t) n_q - 1 && max_b == (int64_t) n_beta - 1 &&
                    max_s == (int64_t) n_state - 1 && f4 && max_tile == HD * HD - 1 &&
                    t32ok && redok;
    std::printf("AUDIT n_head=%d n_seq=%d n_tok=%d max_q=%lld n_q=%zu max_b=%lld n_b=%zu "
                "max_state=%lld n_state=%zu max_tile=%d tile=%d %s\n",
                n_head, n_seqs, n_tokens, (long long) max_q, n_q, (long long) max_b, n_beta,
                (long long) max_s, n_state, max_tile, HD * HD, ok ? "ok" : "FAIL");
    std::fflush(stdout);
    return ok ? 0 : 1;
}

int run_pair(Buf & b, bool want_fast, Stat * out_st, Stat * state_st) {
    b.h2d_inputs();
    b.h2d_state(b.state);
    b.h2d_state(b.state_b);
    hipMemset(b.out, 0, b.n_q * 4);
    hipMemset(b.out_b, 0, b.n_q * 4);
    ling3::KdaShape sh;
    sh.n_head = b.n_head;
    sh.head_dim = 128;
    ling3::KdaScalars sc;
    sc.qk_scale = 1.0f / std::sqrt(128.0f);
    auto viewc = [&](const float * p) {
        return ling3::make_view(p, b.d_inner, b.n_tokens);
    };
    auto viewb = [&](const float * p) {
        return ling3::make_view(p, b.n_head, b.n_tokens);
    };
    // beta view rows = n_head, not d_inner. make_view(p, n_head, n_tokens) is correct.
    const int before = ling3::kda_decode_fast_launches();
    // Env is cached for the process, so this pair selects kernels by the test pick.
    // The variant name is the one the caller put in the env before this call.
    ling3::kda_decode_test_pick(1, nullptr);
    ling3::kda_scan(sh, sc, viewc(b.q), viewc(b.k), viewc(b.v), viewc(b.g), viewb(b.beta),
                    ling3::make_view(b.out, b.d_inner, b.n_tokens), b.state, b.n_tokens, b.n_seqs,
                    b.stream);
    const int mid = ling3::kda_decode_fast_launches();
    const char * var = std::getenv("LING3_KDA_DECODE_VARIANT");
    ling3::kda_decode_test_pick(2, (var != nullptr && var[0] != '\0') ? var : "t32");
    ling3::kda_scan(sh, sc, viewc(b.q), viewc(b.k), viewc(b.v), viewc(b.g), viewb(b.beta),
                    ling3::make_view(b.out_b, b.d_inner, b.n_tokens), b.state_b, b.n_tokens,
                    b.n_seqs, b.stream);
    const int after = ling3::kda_decode_fast_launches();
    hipError_t err = hipStreamSynchronize(b.stream);
    if (err != hipSuccess) {
        std::fprintf(stderr, "sync %s\n", hipGetErrorString(err));
        return -1;
    }
    std::vector<float> oref(b.n_q), ofast(b.n_q), sref(b.n_state), sfast(b.n_state);
    hipMemcpy(oref.data(), b.out, b.n_q * 4, hipMemcpyDeviceToHost);
    hipMemcpy(ofast.data(), b.out_b, b.n_q * 4, hipMemcpyDeviceToHost);
    hipMemcpy(sref.data(), b.state, b.n_state * 4, hipMemcpyDeviceToHost);
    hipMemcpy(sfast.data(), b.state_b, b.n_state * 4, hipMemcpyDeviceToHost);
    *out_st = compare(ofast.data(), oref.data(), b.n_q);
    *state_st = compare(sfast.data(), sref.data(), b.n_state);
    const int delta = after - before;
    const int ref_delta = mid - before;
    if (want_fast) {
        if (ref_delta != 0 || delta != 1) {
            std::printf("GATE path ref_launches=+%d fast_launches=+%d (want 0 and 1)\n", ref_delta, delta);
            return 2;
        }
        return delta;
    }
    if (ref_delta != 0 || delta != 0) {
        std::printf("GATE prefill path moved fast launches +%d / +%d\n", ref_delta, delta);
        return 2;
    }
    return 0;
}

int gate_one(const char * name, int n_head, int n_tok, int n_seq, int steps, uint32_t seed,
             float s_lo, float s_hi, bool zero_state, bool want_fast,
             float g_lo = -4.5f, float g_hi = -0.02f, float b_lo = 0.05f, float b_hi = 0.95f) {
    if (audit_shape(n_head, n_seq, n_tok) != 0) return 1;
    Buf b(n_head, n_tok, n_seq);
    Stat acc_o, acc_s;
    int fast_sum = 0;
    bool path_bad = false;
    for (int step = 0; step < steps; ++step) {
        Rng r(seed + 1009u * (uint32_t) (step + 1));
        const int d = b.d_inner;
        for (int s = 0; s < n_seq; ++s) {
            for (int t = 0; t < n_tok; ++t) {
                float * q = b.hq.data() + (size_t) t * d + (size_t) s * (size_t) d * n_tok;
                float * k = b.hk.data() + (size_t) t * d + (size_t) s * (size_t) d * n_tok;
                float * v = b.hv.data() + (size_t) t * d + (size_t) s * (size_t) d * n_tok;
                float * g = b.hg.data() + (size_t) t * d + (size_t) s * (size_t) d * n_tok;
                float * beta = b.hb.data() + (size_t) t * n_head + (size_t) s * (size_t) n_head * n_tok;
                for (int c = 0; c < d; ++c) {
                    q[c] = r.span(-1.f, 1.f);
                    k[c] = r.span(-1.f, 1.f);
                    v[c] = r.span(-0.5f, 0.5f);
                    g[c] = r.span(g_lo, g_hi);
                }
                for (int h = 0; h < n_head; ++h) beta[h] = r.span(b_lo, b_hi);
            }
        }
        if (step == 0) {
            for (size_t i = 0; i < b.n_state; ++i)
                b.hstate[i] = zero_state ? 0.f : r.span(s_lo, s_hi);
        } else if (hipMemcpy(b.hstate.data(), b.state, b.n_state * 4, hipMemcpyDeviceToHost) != hipSuccess) {
            std::fprintf(stderr, "state D2H failed\n");
            return 1;
        }
        Stat so, ss;
        const int fd = run_pair(b, want_fast, &so, &ss);
        if (fd < 0 || fd == 2) path_bad = true;
        else fast_sum += fd;
        if (so.nbad) acc_o = so;
        if (ss.nbad) acc_s = ss;
        if (path_bad || so.nbad || ss.nbad) break;
    }
    const bool ok = !path_bad && acc_o.nbad == 0 && acc_s.nbad == 0 &&
                    (want_fast ? fast_sum == steps : fast_sum == 0);
    std::printf("GATE %s bad=%d fast_launches=+%d out_bad=%zu out_max_abs=%.3e out_max_rel=%.3e "
                "state_bad=%zu state_max_abs=%.3e state_max_rel=%.3e first_out=%zu first_state=%zu\n",
                name, ok ? 0 : 1, fast_sum, acc_o.nbad, acc_o.max_abs, acc_o.max_rel, acc_s.nbad,
                acc_s.max_abs, acc_s.max_rel, acc_o.first, acc_s.first);
    std::fflush(stdout);
    return ok ? 0 : 1;
}

float time_scan(const char * variant, int iters) {
    Buf b(32, 1, 1);
    Rng r(7);
    for (size_t i = 0; i < b.n_state; ++i) b.hstate[i] = r.span(-0.25f, 0.25f);
    for (int c = 0; c < b.d_inner; ++c) {
        b.hq[c] = r.span(-1.f, 1.f);
        b.hk[c] = r.span(-1.f, 1.f);
        b.hv[c] = r.span(-0.5f, 0.5f);
        b.hg[c] = r.span(-3.f, -0.1f);
    }
    for (int h = 0; h < 32; ++h) b.hb[h] = r.span(0.1f, 0.8f);
    b.h2d_inputs();
    b.h2d_state(b.state);
    ling3::KdaShape sh;
    sh.n_head = 32;
    sh.head_dim = 128;
    ling3::KdaScalars sc;
    sc.qk_scale = 1.0f / std::sqrt(128.0f);
    if (std::strcmp(variant, "ref") == 0) ling3::kda_decode_test_pick(1, nullptr);
    else ling3::kda_decode_test_pick(2, variant);
    auto once = [&]() {
        ling3::kda_scan(sh, sc, ling3::make_view(b.q, b.d_inner, 1), ling3::make_view(b.k, b.d_inner, 1),
                        ling3::make_view(b.v, b.d_inner, 1), ling3::make_view(b.g, b.d_inner, 1),
                        ling3::make_view(b.beta, 32, 1), ling3::make_view(b.out, b.d_inner, 1), b.state, 1, 1,
                        b.stream);
    };
    for (int i = 0; i < 10; ++i) once();
    hipStreamSynchronize(b.stream);
    hipEvent_t a, c;
    hipEventCreate(&a);
    hipEventCreate(&c);
    hipEventRecord(a, b.stream);
    for (int i = 0; i < iters; ++i) once();
    hipEventRecord(c, b.stream);
    hipEventSynchronize(c);
    float ms = 0.f;
    hipEventElapsedTime(&ms, a, c);
    hipEventDestroy(a);
    hipEventDestroy(c);
    return (ms * 1000.f) / (float) iters;
}

float time_conv(int iters) {
    Buf b(32, 1, 1);
    const int d_conv = 4;
    const int d_inner = b.d_inner;
    const int hist = d_conv - 1;
    const size_t n_w = (size_t) d_conv * d_inner;
    const size_t n_st = (size_t) 3 * hist * d_inner;
    std::vector<float> hw(n_w * 3), hs(n_st);
    Rng r(9);
    for (auto & x : hw) x = r.span(-0.2f, 0.2f);
    for (auto & x : hs) x = r.span(-0.2f, 0.2f);
    for (int c = 0; c < d_inner; ++c) {
        b.hq[c] = r.span(-1.f, 1.f);
        b.hk[c] = r.span(-1.f, 1.f);
        b.hv[c] = r.span(-1.f, 1.f);
    }
    b.h2d_inputs();
    float *wq, *wk, *wv, *st;
    hipMalloc(&wq, n_w * 4);
    hipMalloc(&wk, n_w * 4);
    hipMalloc(&wv, n_w * 4);
    hipMalloc(&st, n_st * 4);
    hipMemcpy(wq, hw.data(), n_w * 4, hipMemcpyHostToDevice);
    hipMemcpy(wk, hw.data() + n_w, n_w * 4, hipMemcpyHostToDevice);
    hipMemcpy(wv, hw.data() + 2 * n_w, n_w * 4, hipMemcpyHostToDevice);
    hipMemcpy(st, hs.data(), n_st * 4, hipMemcpyHostToDevice);
    ling3::KdaShape sh;
    sh.n_head = 32;
    sh.head_dim = 128;
    sh.d_conv = 4;
    auto once = [&]() {
        ling3::kda_conv_silu_qkv(sh, ling3::make_view((const float *) b.q, d_inner, 1),
                                 ling3::make_view((const float *) b.k, d_inner, 1),
                                 ling3::make_view((const float *) b.v, d_inner, 1),
                                 ling3::make_view(b.out, d_inner, 1), ling3::make_view(b.out_b, d_inner, 1),
                                 ling3::make_view(b.state, d_inner, 1), wq, wk, wv, st, 1, 1, b.stream);
    };
    for (int i = 0; i < 5; ++i) once();
    hipStreamSynchronize(b.stream);
    hipEvent_t a, c;
    hipEventCreate(&a);
    hipEventCreate(&c);
    hipEventRecord(a, b.stream);
    for (int i = 0; i < iters; ++i) once();
    hipEventRecord(c, b.stream);
    hipEventSynchronize(c);
    float ms = 0.f;
    hipEventElapsedTime(&ms, a, c);
    hipEventDestroy(a);
    hipEventDestroy(c);
    hipFree(wq);
    hipFree(wk);
    hipFree(wv);
    hipFree(st);
    return (ms * 1000.f) / (float) iters;
}

int run_variant(const char * variant) {
    setenv("LING3_KDA_DECODE_VARIANT", variant, 1);
    int rc = 0;
    auto go = [&](const char * name, int nh, int nt, int ns, int steps, uint32_t seed,
                  float slo, float shi, bool zero, bool fast, float glo = -4.5f, float ghi = -0.02f,
                  float blo = 0.05f, float bhi = 0.95f) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s/%s", variant, name);
        rc |= gate_one(buf, nh, nt, ns, steps, seed, slo, shi, zero, fast, glo, ghi, blo, bhi);
    };
    go("small-1step", 1, 1, 1, 1, 1u, -0.5f, 0.5f, false, true);
    go("small-zero-chain8", 1, 1, 1, 8, 2u, 0.f, 0.f, true, true);
    go("h32-chain8", 32, 1, 1, 8, 3u, -0.5f, 0.5f, false, true);
    go("h32-wide-state", 32, 1, 1, 2, 4u, -8.f, 8.f, false, true);
    go("h32-seq2", 32, 1, 2, 2, 5u, -0.5f, 0.5f, false, true);
    go("prefill-t4", 32, 4, 1, 1, 6u, -0.5f, 0.5f, false, false);
    go("h32-beta0", 32, 1, 1, 1, 11u, -0.5f, 0.5f, false, true, -4.5f, -0.02f, 0.f, 0.f);
    go("h32-beta1", 32, 1, 1, 1, 12u, -0.5f, 0.5f, false, true, -4.5f, -0.02f, 1.f, 1.f);
    go("h32-gnear0", 32, 1, 1, 1, 13u, -0.5f, 0.5f, false, true, -1e-4f, -1e-7f, 0.05f, 0.95f);
    go("h32-gneg5", 32, 1, 1, 1, 14u, -0.5f, 0.5f, false, true, -5.f, -5.f, 0.05f, 0.95f);
    go("h32-tiny", 32, 1, 1, 1, 15u, 1e-30f, 1e-30f, false, true);
    go("h32-big", 32, 1, 1, 1, 16u, 1e3f, 1e3f, false, true);
    for (uint32_t s = 0; s < 16; ++s) {
        char name[32];
        std::snprintf(name, sizeof(name), "h32-seed%u", s);
        go(name, 32, 1, 1, 1, 100u + s, -1.f, 1.f, false, true);
    }
    std::printf("GATE variant=%s result=%s\n", variant, rc == 0 ? "PASS" : "FAIL");
    std::fflush(stdout);
    return rc;
}

int do_gate() {
    // vec is the ship default. t32 is the coalesced-tile attempt. red changes sum order.
    const int vec_rc = run_variant("vec");
    const int t32_rc = run_variant("t32");
    const int red_rc = run_variant("red");
    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, 0);
    int max_smem = 0;
    hipDeviceGetAttribute(&max_smem, hipDeviceAttributeMaxSharedMemoryPerBlock, 0);
    const int iters = 200;
    const float ref_us = time_scan("ref", iters);
    const float vec_us = time_scan("vec", iters);
    const float t32_us = time_scan("t32", iters);
    const float red_us = time_scan("red", iters);
    const float conv_us = time_conv(100);
    unsetenv("LING3_KDA_DECODE_REF");
    unsetenv("LING3_KDA_DECODE_VARIANT");
    std::printf("TIME device=%s max_shared=%d iters=%d scan_ref_us=%.3f scan_vec_us=%.3f "
                "scan_t32_us=%.3f scan_red_us=%.3f conv_us=%.3f fast_launches=%d\n",
                prop.name, max_smem, iters, ref_us, vec_us, t32_us, red_us, conv_us,
                ling3::kda_decode_fast_launches());
    std::fflush(stdout);
    std::printf("GATE t32=%s red=%s\n", t32_rc == 0 ? "PASS" : "FAIL", red_rc == 0 ? "PASS" : "FAIL");
    std::printf("GATE result=%s\n", vec_rc == 0 ? "PASS" : "FAIL");
    std::fflush(stdout);
    return vec_rc == 0 ? 0 : 1;
}

int do_time() {
    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, 0);
    int max_smem = 0;
    hipDeviceGetAttribute(&max_smem, hipDeviceAttributeMaxSharedMemoryPerBlock, 0);
    const int iters = 200;
    const float ref_us = time_scan("ref", iters);
    const float vec_us = time_scan("vec", iters);
    const float t32_us = time_scan("t32", iters);
    const float red_us = time_scan("red", iters);
    const float conv_us = time_conv(100);
    unsetenv("LING3_KDA_DECODE_REF");
    unsetenv("LING3_KDA_DECODE_VARIANT");
    std::printf("TIME device=%s max_shared=%d iters=%d scan_ref_us=%.3f scan_vec_us=%.3f "
                "scan_t32_us=%.3f scan_red_us=%.3f conv_us=%.3f fast_launches=%d\n",
                prop.name, max_smem, iters, ref_us, vec_us, t32_us, red_us, conv_us,
                ling3::kda_decode_fast_launches());
    std::fflush(stdout);
    return 0;
}

int do_replay(const char * path) {
    FILE * fp = std::fopen(path, "rb");
    if (!fp) {
        std::perror(path);
        return 1;
    }
    uint32_t magic = 0;
    int32_t version = 0, n_head = 0, head_dim = 0, n_seqs = 0, d_inner = 0, n_rec = 0, stride = 0;
    if (std::fread(&magic, 4, 1, fp) != 1 || std::fread(&version, 4, 1, fp) != 1 ||
        std::fread(&n_head, 4, 1, fp) != 1 || std::fread(&head_dim, 4, 1, fp) != 1 ||
        std::fread(&n_seqs, 4, 1, fp) != 1 || std::fread(&d_inner, 4, 1, fp) != 1 ||
        std::fread(&n_rec, 4, 1, fp) != 1 || std::fread(&stride, 4, 1, fp) != 1) {
        std::fprintf(stderr, "short header\n");
        return 1;
    }
    std::printf("REPLAY magic=%08x ver=%d n_head=%d head_dim=%d n_seq=%d d_inner=%d n_rec=%d stride=%d\n",
                magic, version, n_head, head_dim, n_seqs, d_inner, n_rec, stride);
    if (magic != 0x3141444Bu || version != 1 || head_dim != 128 || n_seqs != 1 || n_head < 1 ||
        d_inner != n_head * 128 || n_rec < 1) {
        std::fprintf(stderr, "bad capture header\n");
        return 1;
    }
    const size_t n_state = (size_t) n_head * 128u * 128u;
    const size_t body = sizeof(int) + sizeof(float) * ((size_t) d_inner * 4u + (size_t) n_head + n_state);
    std::vector<std::vector<char>> recs(n_rec);
    for (int i = 0; i < n_rec; ++i) {
        recs[i].resize(body);
        if (std::fread(recs[i].data(), 1, body, fp) != body) {
            std::fprintf(stderr, "short record %d\n", i);
            return 1;
        }
    }
    std::fclose(fp);

    if (audit_shape(n_head, 1, 1) != 0) return 1;
    Buf b(n_head, 1, 1);
    ling3::KdaShape sh;
    sh.n_head = n_head;
    sh.head_dim = 128;
    ling3::KdaScalars sc;
    sc.qk_scale = 1.0f / std::sqrt(128.0f);

    auto load_rec = [&](const std::vector<char> & rec, bool with_state) {
        const char * p = rec.data();
        int call = 0;
        std::memcpy(&call, p, 4);
        p += 4;
        std::memcpy(b.hq.data(), p, (size_t) d_inner * 4);
        p += (size_t) d_inner * 4;
        std::memcpy(b.hk.data(), p, (size_t) d_inner * 4);
        p += (size_t) d_inner * 4;
        std::memcpy(b.hv.data(), p, (size_t) d_inner * 4);
        p += (size_t) d_inner * 4;
        std::memcpy(b.hg.data(), p, (size_t) d_inner * 4);
        p += (size_t) d_inner * 4;
        std::memcpy(b.hb.data(), p, (size_t) n_head * 4);
        p += (size_t) n_head * 4;
        if (with_state) std::memcpy(b.hstate.data(), p, n_state * 4);
        return call;
    };

    int rc = 0;
    for (int i = 0; i < n_rec; ++i) {
        const int call = load_rec(recs[i], true);
        Stat so, ss;
        const int fd = run_pair(b, true, &so, &ss);
        char name[64];
        std::snprintf(name, sizeof(name), "replay-call%d", call);
        report(name, so, ss, fd < 0 ? 0 : fd);
        if (fd != 1 || so.nbad || ss.nbad) rc = 1;
    }

    // Chain records that are the same layer (call % stride == 0), in file order.
    std::vector<int> chain;
    for (int i = 0; i < n_rec; ++i) {
        int call = 0;
        std::memcpy(&call, recs[i].data(), 4);
        if (stride > 0 && call % stride == 0) chain.push_back(i);
    }
    if ((int) chain.size() >= 2) {
        load_rec(recs[chain[0]], true);
        Stat acc_o, acc_s;
        int fast_sum = 0;
        bool broke = false;
        for (size_t s = 0; s < chain.size(); ++s) {
            if (s > 0) {
                hipMemcpy(b.hstate.data(), b.state, b.n_state * 4, hipMemcpyDeviceToHost);
                load_rec(recs[chain[s]], false);
            }
            Stat so, ss;
            const int fd = run_pair(b, true, &so, &ss);
            if (fd > 0) fast_sum += fd;
            if (so.nbad) acc_o = so;
            if (ss.nbad) acc_s = ss;
            if (fd < 0 || so.nbad || ss.nbad) {
                broke = true;
                break;
            }
        }
        report("replay-chain", acc_o, acc_s, fast_sum);
        if (broke || fast_sum != (int) chain.size()) rc = 1;
    }
    std::printf("REPLAY result=%s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s gate|time|replay <file>\n", argv[0]);
        return 2;
    }
    const std::string cmd = argv[1];
    unsetenv("LING3_KDA_CAPTURE");
    if (cmd == "gate") return do_gate();
    if (cmd == "time") return do_time();
    if (cmd == "replay") {
        if (argc < 3) return 2;
        return do_replay(argv[2]);
    }
    std::fprintf(stderr, "unknown %s\n", argv[1]);
    return 2;
}
