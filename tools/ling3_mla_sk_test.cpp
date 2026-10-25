// ling3_mla_sk_test.cpp — LS3 PHASE4 unit gate for the split-K decode
// attention (mla_attn_splitk) against the serial mla_attn, which every
// earlier gate already tied to the CPU reference. Synthetic tensors at the
// model's real shapes; n_pos sweeps across slice boundaries (1024, 1025,
// 3000, 8192). Also proves run-to-run determinism (two runs byte-identical)
// and that the serial path is untouched. Run inside gpu-run.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ling3_mla.hpp"

namespace {

int g_fail = 0;
unsigned g_state = 20261002u;
unsigned Rnd() {
    g_state = g_state * 1664525u + 1013904223u;
    return g_state;
}
float Frnd(float lo, float hi) {
    return lo + (hi - lo) * ((float) (Rnd() & 0xffff) / 65535.0f);
}

#define HIPCK(expr)                                                            \
    do {                                                                       \
        hipError_t e_ = (expr);                                                \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "HIP %s: %s\n", #expr, hipGetErrorString(e_));\
            std::exit(9);                                                      \
        }                                                                      \
    } while (0)

void *DevAlloc(size_t bytes) {
    void *p = nullptr;
    HIPCK(hipMalloc(&p, bytes + 64));
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    std::setbuf(stdout, nullptr);
    const bool only_p7 = argc > 1 && std::strcmp(argv[1], "p7") == 0;
    // SPEC6: `bits` byte-compares the transposed and original weight layouts;
    // `bench` / `bench0` / `bench1` time mla_attn_splitk (both / original /
    // transposed) at decode depths.
    const bool mode_bits = argc > 1 && std::strcmp(argv[1], "bits") == 0;
    const int bench_mode = (argc > 1 && std::strcmp(argv[1], "bench") == 0)   ? 2
                         : (argc > 1 && std::strcmp(argv[1], "bench0") == 0)  ? 0
                         : (argc > 1 && std::strcmp(argv[1], "bench1") == 0)  ? 1
                                                                             : -1;
    // Real MLA geometry.
    ling3::MlaShape sh;
    sh.n_embd = 2560; sh.n_head = 32; sh.head_k = 192; sh.head_v = 128;
    sh.nope = 128; sh.rot = 64; sh.kv = 512; sh.rope_theta = 6000000.0f;
    ling3::MlaScalars sc;
    const int q_dim = sh.q_dim();      // 6144
    const int kv_a = sh.kv_a();        // 576
    const int o_dim = sh.o_dim();      // 4096
    const int max_pos = 30000;   // PHASE5: n_pos sweep extended to 30K
    const int slice = ling3::kMlaSplitkSlice;   // 1024
    const size_t ks_max = ling3::mla_splitk_slices(max_pos, slice);
    // PHASE7: partials sized for the FIXED slot cap; short slices make
    // k_slices up to kMlaSplitkMaxSlices, far beyond ceil(max_pos/1024).
    const size_t slots = (size_t) ling3::kMlaSplitkMaxSlices;

    // Weights (device): wk_b [nope, kv, n_head], wv_b [kv, head_v, n_head]
    std::vector<float> wk((size_t) sh.n_head * sh.kv * sh.nope);
    std::vector<float> wv((size_t) sh.n_head * sh.head_v * sh.kv);
    for (auto &v : wk) v = Frnd(-0.05f, 0.05f);
    for (auto &v : wv) v = Frnd(-0.05f, 0.05f);
    std::vector<float> q((size_t) q_dim), kv((size_t) kv_a), gate(sh.n_head);
    for (auto &v : q) v = Frnd(-0.5f, 0.5f);
    for (auto &v : kv) v = Frnd(-0.5f, 0.5f);
    for (auto &v : gate) v = Frnd(-2.f, 2.f);

    void *d_wk = DevAlloc(wk.size() * 4), *d_wv = DevAlloc(wv.size() * 4);
    void *d_q = DevAlloc(q.size() * 4), *d_kv = DevAlloc(kv.size() * 4);
    void *d_gate = DevAlloc(gate.size() * 4);
    HIPCK(hipMemcpy(d_wk, wk.data(), wk.size() * 4, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_wv, wv.data(), wv.size() * 4, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_q, q.data(), q.size() * 4, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_kv, kv.data(), kv.size() * 4, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_gate, gate.data(), gate.size() * 4, hipMemcpyHostToDevice));

    // SPEC6: row-transposed copies -- wk_bT[h][i][r] == wk_b[h][r][i] -- the
    // decode prep / de-absorb read these so a wave's threads hit consecutive
    // addresses. Built here exactly as the loader builds them.
    std::vector<float> wkT(wk.size()), wvT(wv.size());
    for (int h = 0; h < sh.n_head; ++h) {
        for (int r = 0; r < sh.kv; ++r) {
            for (int i = 0; i < sh.nope; ++i) {
                wkT[((size_t) h * sh.nope + i) * sh.kv + r] =
                    wk[((size_t) h * sh.kv + r) * sh.nope + i];
            }
        }
        for (int r = 0; r < sh.head_v; ++r) {
            for (int j = 0; j < sh.kv; ++j) {
                wvT[((size_t) h * sh.kv + j) * sh.head_v + r] =
                    wv[((size_t) h * sh.head_v + r) * sh.kv + j];
            }
        }
    }
    void *d_wkT = DevAlloc(wkT.size() * 4), *d_wvT = DevAlloc(wvT.size() * 4);
    HIPCK(hipMemcpy(d_wkT, wkT.data(), wkT.size() * 4, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_wvT, wvT.data(), wvT.size() * 4, hipMemcpyHostToDevice));

    ling3::MlaWeights w;
    w.wk_b = (const float *) d_wk;
    w.wv_b = (const float *) d_wv;
    w.kv_a_norm = nullptr;   // unused by mla_attn / mla_attn_splitk
    w.wk_bT = (const float *) d_wkT;
    w.wv_bT = (const float *) d_wvT;

    // Split-K partials sized for max_pos.
    void *d_pm2 = DevAlloc((size_t) sh.n_head * slots * 4);
    void *d_pl2 = DevAlloc((size_t) sh.n_head * slots * 4);
    void *d_pa2 = DevAlloc((size_t) sh.n_head * slots * (size_t) sh.kv * 4);
    void *d_qabs = DevAlloc((size_t) 4 * sh.n_head * (size_t) sh.kv * 4);
    void *d_pm = DevAlloc((size_t) sh.n_head * slots * 4);
    void *d_pl = DevAlloc((size_t) sh.n_head * slots * 4);
    void *d_pa = DevAlloc((size_t) sh.n_head * slots * (size_t) sh.kv * 4);
    // Serial score scratch, max size.
    void *d_sc = DevAlloc((size_t) sh.n_head * max_pos * 4);
    void *d_o1 = DevAlloc((size_t) o_dim * 4);
    void *d_o2 = DevAlloc((size_t) o_dim * 4);
    std::vector<float> o1(o_dim), o2(o_dim), o2b(o_dim);

    if (mode_bits || bench_mode >= 0) {
        if (mode_bits) {
            for (int n_pos : {1, 100, 1025, 1630, 6500, 13828}) {
                std::vector<float> cache((size_t) n_pos * kv_a);
                for (auto &v : cache) v = Frnd(-0.5f, 0.5f);
                void *d_c = DevAlloc(cache.size() * 4);
                HIPCK(hipMemcpy(d_c, cache.data(), cache.size() * 4, hipMemcpyHostToDevice));
                ling3::MlaCache ch;
                ch.kv = (float *) d_c; ch.s_pos = kv_a; ch.s_seq = 0; ch.max_pos = n_pos;
                ling3::MlaInputs in;
                in.q = ling3::make_view((float *) d_q, q_dim, 1);
                in.kv_a = ling3::make_view((float *) d_kv, kv_a, 1);
                in.gate = ling3::make_view((float *) d_gate, sh.n_head, 1);
                in.n_tokens = 1; in.n_seqs = 1;
                ling3::MlaWeights w0 = w; w0.wk_bT = nullptr; w0.wv_bT = nullptr;
                ling3::mla_attn_splitk(sh, sc, w0, ch, n_pos, in,
                                       ling3::make_view((float *) d_o1, o_dim, 1),
                                       (float *) d_qabs, (float *) d_pm,
                                       (float *) d_pl, (float *) d_pa, slice, 2, nullptr);
                HIPCK(hipDeviceSynchronize());
                HIPCK(hipMemcpy(o1.data(), d_o1, o_dim * 4, hipMemcpyDeviceToHost));
                ling3::mla_attn_splitk(sh, sc, w, ch, n_pos, in,
                                       ling3::make_view((float *) d_o2, o_dim, 1),
                                       (float *) d_qabs, (float *) d_pm,
                                       (float *) d_pl, (float *) d_pa, slice, 2, nullptr);
                HIPCK(hipDeviceSynchronize());
                HIPCK(hipMemcpy(o2.data(), d_o2, o_dim * 4, hipMemcpyDeviceToHost));
                const bool same = std::memcmp(o1.data(), o2.data(), o_dim * 4) == 0;
                std::printf("BITS n_pos=%d transposed_vs_original %s\n", n_pos,
                            same ? "IDENTICAL" : "DIFFERENT");
                if (!same) ++g_fail;
                hipFree(d_c);
            }
            std::printf("%s fail=%d\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
            return g_fail == 0 ? 0 : 1;
        }
        // bench: hipEvent time of mla_attn_splitk (whole call) per layout.
        hipEvent_t e0, e1;
        HIPCK(hipEventCreate(&e0)); HIPCK(hipEventCreate(&e1));
        const int reps = 20;
        for (int n_pos : {1025, 1630, 6500, 13828}) {
            for (int mode = 0; mode < 2; ++mode) {
                if (bench_mode < 2 && bench_mode != mode) continue;
                std::vector<float> cache((size_t) n_pos * kv_a);
                for (auto &v : cache) v = Frnd(-0.5f, 0.5f);
                void *d_c = DevAlloc(cache.size() * 4);
                HIPCK(hipMemcpy(d_c, cache.data(), cache.size() * 4, hipMemcpyHostToDevice));
                ling3::MlaCache ch;
                ch.kv = (float *) d_c; ch.s_pos = kv_a; ch.s_seq = 0; ch.max_pos = n_pos;
                ling3::MlaInputs in;
                in.q = ling3::make_view((float *) d_q, q_dim, 1);
                in.kv_a = ling3::make_view((float *) d_kv, kv_a, 1);
                in.gate = ling3::make_view((float *) d_gate, sh.n_head, 1);
                in.n_tokens = 1; in.n_seqs = 1;
                ling3::MlaWeights ww = w;
                if (mode == 0) { ww.wk_bT = nullptr; ww.wv_bT = nullptr; }
                // the shape the engine actually runs: ~120 workgroups, one
                // per slice (rocm_backend.cpp MlaSplitkSlice / LING3_MLA_SPLITK_WG)
                const int sl = ling3::mla_splitk_slice_auto(n_pos, 120);
                for (int warm = 0; warm < 5; ++warm) {
                    ling3::mla_attn_splitk(sh, sc, ww, ch, n_pos, in,
                                           ling3::make_view((float *) d_o1, o_dim, 1),
                                           (float *) d_qabs, (float *) d_pm,
                                           (float *) d_pl, (float *) d_pa,
                                           sl, 2, nullptr);
                }
                HIPCK(hipDeviceSynchronize());
                HIPCK(hipEventRecord(e0, nullptr));
                for (int it = 0; it < reps; ++it) {
                    ling3::mla_attn_splitk(sh, sc, ww, ch, n_pos, in,
                                           ling3::make_view((float *) d_o1, o_dim, 1),
                                           (float *) d_qabs, (float *) d_pm,
                                           (float *) d_pl, (float *) d_pa,
                                           sl, 2, nullptr);
                }
                HIPCK(hipEventRecord(e1, nullptr));
                HIPCK(hipEventSynchronize(e1));
                float ms = 0.f;
                HIPCK(hipEventElapsedTime(&ms, e0, e1));
                std::printf("BENCH n_pos=%d slice=%d slices=%d layout=%s  "
                            "mla_attn_splitk %.2f us/call (%d reps)\n",
                            n_pos, sl, (n_pos + sl - 1) / sl,
                            mode ? "transposed" : "original",
                            ms * 1000.0f / reps, reps);
                hipFree(d_c);
            }
        }
        hipEventDestroy(e0); hipEventDestroy(e1);
        std::printf("%s fail=%d\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
        return g_fail == 0 ? 0 : 1;
    }
    if (argc > 1 && std::strcmp(argv[1], "mrow") == 0) {
        // PHASE7 item 4: mla_attn_splitk_m vs m single-row calls, <= 1e-6
        // rel, run-twice determinism. Every buffer is sized for m_max rows
        // and the extents are printed before the first launch (the
        // 2026-10-03T04:43Z GCVM permission fault was an out-of-range
        // partial access from a one-row-sized harness buffer).
        const int m_max = 4;
        std::vector<float> qm((size_t) m_max * q_dim), gm((size_t) m_max * sh.n_head);
        for (auto &v : qm) v = Frnd(-0.5f, 0.5f);
        for (auto &v : gm) v = Frnd(-2.f, 2.f);
        void *d_qm = DevAlloc(qm.size() * 4);
        void *d_gm = DevAlloc(gm.size() * 4);
        void *d_qabs_m = DevAlloc((size_t) m_max * sh.n_head * sh.kv * 4);
        void *d_pm_m = DevAlloc((size_t) m_max * sh.n_head * slots * 4);
        void *d_pl_m = DevAlloc((size_t) m_max * sh.n_head * slots * 4);
        void *d_pa_m = DevAlloc((size_t) m_max * sh.n_head * slots * sh.kv * 4);
        void *d_om = DevAlloc((size_t) m_max * o_dim * 4);
        void *d_om2 = DevAlloc((size_t) m_max * o_dim * 4);
        HIPCK(hipMemcpy(d_qm, qm.data(), qm.size() * 4, hipMemcpyHostToDevice));
        HIPCK(hipMemcpy(d_gm, gm.data(), gm.size() * 4, hipMemcpyHostToDevice));
        std::printf("MROW audit: q=%zu gate=%zu q_abs=%zu pm/pl=%zu pa=%zu out=%zu floats "
                    "(m_max=%d slots=%zu kv=%d)\n",
                    qm.size(), gm.size(), (size_t) m_max * sh.n_head * sh.kv,
                    (size_t) m_max * sh.n_head * slots,
                    (size_t) m_max * sh.n_head * slots * sh.kv,
                    (size_t) m_max * o_dim, m_max, slots, sh.kv);
        for (int n_pos : {64, 100, 1025, 6500}) {
            std::vector<float> cache((size_t) n_pos * kv_a);
            for (auto &v : cache) v = Frnd(-0.5f, 0.5f);
            void *d_cache = DevAlloc(cache.size() * 4);
            HIPCK(hipMemcpy(d_cache, cache.data(), cache.size() * 4, hipMemcpyHostToDevice));
            ling3::MlaCache ch;
            ch.kv = (float *) d_cache; ch.s_pos = kv_a; ch.s_seq = 0; ch.max_pos = n_pos;
            for (int m = 2; m <= 4; ++m) {
                if (m > n_pos) continue;
                for (int sl : {64, 1024}) {
                    // reference: m single-row calls (row r sees n_pos-m+r+1)
                    for (int r = 0; r < m; ++r) {
                        ling3::MlaInputs in1;
                        in1.q = ling3::make_view((float *) d_qm + (size_t) r * q_dim, q_dim, 1);
                        in1.kv_a = ling3::make_view((float *) d_kv, kv_a, 1);
                        in1.gate = ling3::make_view((float *) d_gm + (size_t) r * sh.n_head, sh.n_head, 1);
                        in1.n_tokens = 1; in1.n_seqs = 1;
                        ling3::mla_attn_splitk(sh, sc, w, ch, n_pos - m + r + 1, in1,
                                              ling3::make_view((float *) d_om + (size_t) r * o_dim, o_dim, 1),
                                              (float *) d_qabs_m, (float *) d_pm_m, (float *) d_pl_m,
                                              (float *) d_pa_m, sl, 2, nullptr);
                    }
                    HIPCK(hipDeviceSynchronize());
                    std::vector<float> ref((size_t) m * o_dim), got((size_t) m * o_dim), got2((size_t) m * o_dim);
                    HIPCK(hipMemcpy(ref.data(), d_om, (size_t) m * o_dim * 4, hipMemcpyDeviceToHost));
                    for (int rep = 0; rep < 2; ++rep) {
                        ling3::MlaInputs inm;
                        inm.q = ling3::make_view((float *) d_qm, q_dim, 1);
                        inm.kv_a = ling3::make_view((float *) d_kv, kv_a, 1);
                        inm.gate = ling3::make_view((float *) d_gm, sh.n_head, 1);
                        inm.n_tokens = m; inm.n_seqs = 1;
                        ling3::mla_attn_splitk_m(sh, sc, w, ch, n_pos, inm,
                                                ling3::make_view((float *) d_om2, o_dim, 1), m,
                                                (float *) d_qabs_m, (float *) d_pm_m, (float *) d_pl_m,
                                                (float *) d_pa_m, sl, 2, nullptr);
                        HIPCK(hipDeviceSynchronize());
                        HIPCK(hipMemcpy(rep == 0 ? got.data() : got2.data(), d_om2,
                                        (size_t) m * o_dim * 4, hipMemcpyDeviceToHost));
                    }
                    double md = 0, sc3 = 1e-30;
                    for (size_t i = 0; i < ref.size(); ++i) {
                        md = std::fmax(md, std::fabs((double) ref[i] - (double) got[i]));
                        sc3 = std::fmax(sc3, std::fabs((double) ref[i]));
                    }
                    const bool det = std::memcmp(got.data(), got2.data(), got.size() * 4) == 0;
                    const bool ok = md <= 1e-6 * sc3 && det;
                    std::printf("MROW n_pos=%d m=%d slice=%d k_slices=%d %s max_abs=%.3e rel=%.3e det=%s\n",
                                n_pos, m, sl, (n_pos + sl - 1) / sl, ok ? "PASS" : "FAIL",
                                md, md / sc3, det ? "y" : "N");
                    if (!ok) ++g_fail;
                }
            }
            hipFree(d_cache);
        }
        hipFree(d_qm); hipFree(d_gm); hipFree(d_qabs_m); hipFree(d_pm_m);
        hipFree(d_pl_m); hipFree(d_pa_m); hipFree(d_om); hipFree(d_om2);
        std::printf("%s fail=%d\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
        return g_fail == 0 ? 0 : 1;
    }

    const std::vector<int> cases =
        only_p7 ? std::vector<int>{6500}
                : std::vector<int>{1,  2,   63,   64,    65,   100,  1023,
                                   1024, 1025, 3000, 6500, 8192, 30000};
    for (int n_pos : cases) {
        // random visible cache
        std::vector<float> cache((size_t) n_pos * kv_a);
        for (auto &v : cache) v = Frnd(-0.5f, 0.5f);
        void *d_cache = DevAlloc(cache.size() * 4);
        HIPCK(hipMemcpy(d_cache, cache.data(), cache.size() * 4, hipMemcpyHostToDevice));

        ling3::MlaCache ch;
        ch.kv = (float *) d_cache;
        ch.s_pos = kv_a;
        ch.s_seq = 0;
        ch.max_pos = n_pos;

        ling3::MlaInputs in;
        in.q = ling3::make_view((float *) d_q, q_dim, 1);
        in.kv_a = ling3::make_view((float *) d_kv, kv_a, 1);
        in.gate = ling3::make_view((float *) d_gate, sh.n_head, 1);
        in.n_tokens = 1;
        in.n_seqs = 1;

        ling3::mla_attn(sh, sc, w, ch, n_pos, in,
                        ling3::make_view((float *) d_o1, o_dim, 1),
                        (float *) d_sc, nullptr);
        HIPCK(hipDeviceSynchronize());
        HIPCK(hipMemcpy(o1.data(), d_o1, o_dim * 4, hipMemcpyDeviceToHost));

        double mabs[3] = {0, 0, 0}, between = 0, scale = 1e-30;
        bool det[3] = {false, false, false};
        for (int hs = 0; hs <= 2; ++hs) {
            std::printf("case n_pos=%d hs=%d enter\n", n_pos, hs); std::fflush(stdout);
            for (int rep = 0; rep < 2; ++rep) {
                ling3::mla_attn_splitk(sh, sc, w, ch, n_pos, in,
                                       ling3::make_view((float *) d_o2, o_dim, 1),
                                       (float *) d_qabs, (float *) d_pm, (float *) d_pl,
                                       (float *) d_pa, slice, hs, nullptr);
                HIPCK(hipDeviceSynchronize());
                HIPCK(hipMemcpy(rep == 0 ? o2.data() : o2b.data(), d_o2, o_dim * 4,
                                hipMemcpyDeviceToHost));
            }
            for (int i = 0; i < o_dim; ++i) {
                const double d = std::fabs((double) o1[i] - (double) o2[i]);
                if (d > mabs[hs]) mabs[hs] = d;
                scale = std::fmax(scale, std::fabs((double) o1[i]));
            }
            det[hs] = std::memcmp(o2.data(), o2b.data(), o_dim * 4) == 0;
        }
        // hs=2 vs hs=0 outputs directly
        std::vector<float> oh0(o_dim), oh2(o_dim);
        ling3::mla_attn_splitk(sh, sc, w, ch, n_pos, in,
                               ling3::make_view((float *) d_o2, o_dim, 1),
                               (float *) d_qabs, (float *) d_pm, (float *) d_pl,
                               (float *) d_pa, slice, 0, nullptr);
        HIPCK(hipDeviceSynchronize());
        HIPCK(hipMemcpy(oh0.data(), d_o2, o_dim * 4, hipMemcpyDeviceToHost));
        ling3::mla_attn_splitk(sh, sc, w, ch, n_pos, in,
                               ling3::make_view((float *) d_o2, o_dim, 1),
                               (float *) d_qabs, (float *) d_pm, (float *) d_pl,
                               (float *) d_pa, slice, 2, nullptr);
        HIPCK(hipDeviceSynchronize());
        HIPCK(hipMemcpy(oh2.data(), d_o2, o_dim * 4, hipMemcpyDeviceToHost));
        double scale2 = 1e-30;
        for (int i = 0; i < o_dim; ++i) {
            const double d = std::fabs((double) oh0[i] - (double) oh2[i]);
            if (d > between) between = d;
            scale2 = std::fmax(scale2, std::fabs((double) oh0[i]));
        }
        const bool ok = mabs[0] <= 1e-4 * scale && mabs[1] <= 1e-4 * scale &&
                        mabs[2] <= 1e-4 * scale && det[0] && det[1] && det[2] &&
                        between <= 1e-4 * scale2;
        std::printf("SK n_pos=%d slices=%d %s vs_serial(per-head=%.3e hs=%.3e rf=%.3e) "
                    "rf_vs_perhead=%.3e det=%s/%s/%s\n",
                    n_pos, (n_pos + slice - 1) / slice, ok ? "PASS" : "FAIL",
                    mabs[0], mabs[1], mabs[2], between, det[0] ? "y" : "N",
                    det[1] ? "y" : "N", det[2] ? "y" : "N");
        if (!ok) ++g_fail;
        // PHASE7 (in-loop): short slice lengths of the rf kernel at this
        // n_pos, on the same buffers as the passing calls above (o1 = the
        // case\x27s serial reference).
        if (n_pos == 6500) {
            for (int sl : {64, 128, 256, 384, 512}) {
                ling3::mla_attn_splitk(sh, sc, w, ch, n_pos, in,
                                       ling3::make_view((float *) d_o2, o_dim, 1),
                                       (float *) d_qabs, (float *) d_pm, (float *) d_pl,
                                       (float *) d_pa, sl, 2, nullptr);
                HIPCK(hipDeviceSynchronize());
                HIPCK(hipMemcpy(o2.data(), d_o2, o_dim * 4, hipMemcpyDeviceToHost));
                double sm = 0, sc2 = 1e-30;
                for (int i2 = 0; i2 < o_dim; ++i2) {
                    sm = std::fmax(sm, std::fabs((double) o1[i2] - (double) o2[i2]));
                    sc2 = std::fmax(sc2, std::fabs((double) o1[i2]));
                }
                const bool sok = sm <= 1e-4 * sc2;
                std::printf("SLICE len=%d n_pos=6500 %s max_abs=%.3e\n", sl,
                            sok ? "PASS" : "FAIL", sm);
                if (!sok) ++g_fail;
            }
        }
        hipFree(d_cache);
    }
    hipFree(d_wk); hipFree(d_wv); hipFree(d_wkT); hipFree(d_wvT);
    hipFree(d_q); hipFree(d_kv); hipFree(d_gate);
    hipFree(d_qabs); hipFree(d_pm); hipFree(d_pl); hipFree(d_pa); hipFree(d_sc);
    hipFree(d_o1); hipFree(d_o2);
    std::printf("%s fail=%d\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}
