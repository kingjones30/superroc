// ling3_mla_w2_test.cpp — LW1: unit gate for mla_w2_attn (the two-pass WMMA
// rewrite) against BOTH references — the serial mla_attn (the kernel that
// passed every phase-1 gate) and mla_attn_tok (today's batched-prefill
// kernel, f32 vector ALUs) — on synthetic tensors at the model's real shapes
// (32 heads, dk 192, dv 128, latent 512, kv_a 576). No model load: seconds on
// the GPU. Run inside gpu-run with no arguments.
//
// All three paths run over the SAME roped / latent-normed / cache-appended
// inputs, so any diff beyond fp-reorder + f16-input noise is a bug in the w2
// kernels. Cases cover the caller's real sub-batching (64 inside a chunk,
// chunk_tokens = whole-chunk T), the SPEC's T set (1, 2, 5, 63, 64, 65, 130,
// 512), tail sub-batches, and history lengths that are NOT multiples of the
// 64-tile geometry (57, 507), plus n_pos odd.
//
// Metrics printed per case: max_abs, the GATE metric max_rel = |d|/ref_scale
// (the house convention LS1/LS3 use — a per-element rel is noise-dominated at
// near-zero refs, reported separately as `elem`), and the p50/p99 of the
// per-element relative error. PASS bar (SPEC item 4): max_rel <= 2e-3 vs
// mla_attn (and the model gates decide the rest).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "ling3_mla.hpp"
#include "ling3_mla_w2.hpp"

namespace {

int g_fail = 0;

struct Buf {
    void* p = nullptr;
    explicit Buf(std::size_t bytes) { hipMalloc(&p, bytes); }
    ~Buf() {
        if (p != nullptr) hipFree(p);
    }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
};

#define MLA_TEST_CHECK(expr)                                                        \
    do {                                                                            \
        if ((expr) != hipSuccess) {                                                 \
            std::printf("FAIL hip: %s at %s:%d\n", #expr, __FILE__, __LINE__);      \
            return 2;                                                               \
        }                                                                           \
    } while (0)

struct Stats {
    double max_abs = 0.0;
    double max_rel = 0.0;          // GATE metric: |d| / ref_scale (house convention,
                                   // LS1/LS3 style — a per-element rel with a floor is
                                   // noise-dominated at near-zero refs)
    double max_rel_elem = 0.0;     // diagnostic: |d| / max(|ref|, 1e-6)
    double p50 = 0.0, p99 = 0.0;   // of |d| / max(|ref|, 1e-6)
};

Stats compare(const std::vector<float>& got, const std::vector<float>& ref) {
    Stats st;
    std::vector<double> rel;
    rel.reserve(ref.size());
    double ref_scale = 1e-30;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        ref_scale = std::max(ref_scale, std::fabs((double) ref[i]));
    }
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs((double) got[i] - (double) ref[i]);
        const double r = d / std::max(std::fabs((double) ref[i]), 1e-6);
        st.max_abs = std::max(st.max_abs, d);
        st.max_rel = std::max(st.max_rel, d / ref_scale);
        st.max_rel_elem = std::max(st.max_rel_elem, r);
        rel.push_back(r);
    }
    std::sort(rel.begin(), rel.end());
    const std::size_t n = rel.size();
    st.p50 = rel[n / 2];
    st.p99 = rel[std::min(n - 1, (std::size_t) (0.99 * n))];
    return st;
}

void report(const char* name, const char* ref, const Stats& st, double pass_bar) {
    const bool pass = st.max_rel <= pass_bar;
    if (!pass) ++g_fail;
    std::printf("  vs %-10s %-22s %s max_abs=%.3e max_rel=%.3e (elem=%.3e "
                "p50=%.1e p99=%.1e)\n",
                ref, name, pass ? "PASS" : "FAIL", st.max_abs, st.max_rel,
                st.max_rel_elem, st.p50, st.p99);
}

}  // namespace

int main() {
    using namespace ::ling3;
    const MlaShape sh;
    const MlaScalars sc;
    const int H = sh.n_head;       // 32
    const int nope = sh.nope;      // 128
    const int kv = sh.kv;          // 512
    const int kva = sh.kv_a();     // 576
    const int qd = sh.q_dim();     // 6144
    const int od = sh.o_dim();    // 4096
    const int gd = sh.n_head;      // gate width = 32
    const int hv = sh.head_v;      // 128
    const int max_pos = 1024;
    const int max_T = 512;
    const int kSubBatch = kMlaW2Batch;   // 64, the caller's sub-batch cap

    MLA_TEST_CHECK(hipSetDevice(0));
    hipStream_t stream = nullptr;

    std::mt19937 rng(20261002);
    auto rnd = [&](float lo, float hi) {
        std::uniform_real_distribution<float> d(lo, hi);
        return d(rng);
    };

    // ---- weights (host fill, device upload) ----
    const std::size_t wk_n = (std::size_t) H * kv * nope;
    const std::size_t wv_n = (std::size_t) H * hv * kv;
    std::vector<float> hwk(wk_n), hwv(wv_n), hnorm((std::size_t) kv);
    for (auto& v : hwk) v = rnd(-0.03f, 0.03f);
    for (auto& v : hwv) v = rnd(-0.05f, 0.05f);
    for (auto& v : hnorm) v = rnd(0.6f, 1.4f);
    Buf dwk(wk_n * sizeof(float)), dwv(wv_n * sizeof(float)),
        dnorm((std::size_t) kv * sizeof(float));
    MLA_TEST_CHECK(hipMemcpy(dwk.p, hwk.data(), wk_n * sizeof(float), hipMemcpyHostToDevice));
    MLA_TEST_CHECK(hipMemcpy(dwv.p, hwv.data(), wv_n * sizeof(float), hipMemcpyHostToDevice));
    MLA_TEST_CHECK(hipMemcpy(dnorm.p, hnorm.data(), kv * sizeof(float), hipMemcpyHostToDevice));

    MlaWeights w;
    w.wk_b = (const float*) dwk.p;
    w.wv_b = (const float*) dwv.p;
    w.kv_a_norm = (const float*) dnorm.p;

    // ---- per-case buffers, sized once for the biggest case ----
    std::vector<float> hq((std::size_t) max_T * qd), hkva((std::size_t) max_T * kva),
        hgate((std::size_t) max_T * gd), hcache((std::size_t) max_pos * kva);
    std::vector<float> href((std::size_t) max_T * od), htok((std::size_t) max_T * od),
        hw2((std::size_t) max_T * od);
    std::vector<int> hpos((std::size_t) max_T);
    Buf dq((std::size_t) max_T * qd * sizeof(float)),
        dkva((std::size_t) max_T * kva * sizeof(float)),
        dgate((std::size_t) max_T * gd * sizeof(float)),
        dcache((std::size_t) max_pos * kva * sizeof(float)),
        dpos((std::size_t) max_T * sizeof(int)),
        dout_ref((std::size_t) max_T * od * sizeof(float)),
        dout_tok((std::size_t) max_T * od * sizeof(float)),
        dout_w2((std::size_t) max_T * od * sizeof(float)),
        dscores((std::size_t) kSubBatch * H * max_pos * sizeof(float));

    // ---- w2 staging buffers (the production sizes; cap = max_pos) ----
    MlaW2Bufs bufs;
    bufs.cap = max_pos;
    Buf bk16((std::size_t) max_pos * kva * sizeof(__half)),
        bvt((std::size_t) kv * max_pos * sizeof(__half)),
        bq16((std::size_t) kMlaW2Rows * kva * sizeof(__half)),
        bp16((std::size_t) kMlaW2Rows * max_pos * sizeof(__half)),
        bo((std::size_t) kMlaW2Rows * kv * sizeof(float)),
        browsum((std::size_t) kMlaW2Rows * sizeof(float));
    bufs.k16 = (__half*) bk16.p;
    bufs.vt = (__half*) bvt.p;
    bufs.q16 = (__half*) bq16.p;
    bufs.p16 = (__half*) bp16.p;
    bufs.o = (float*) bo.p;
    bufs.rowsum = (float*) browsum.p;
    // poison the staging so unwritten reads would be loud (the kernels must
    // never read beyond what prep/softmax wrote)
    {
        const std::vector<__half> poison((std::size_t) max_pos * kva,
                                         __float2half(std::nanf("")));
        MLA_TEST_CHECK(hipMemcpy(bk16.p, poison.data(), poison.size() * sizeof(__half),
                                 hipMemcpyHostToDevice));
        MLA_TEST_CHECK(hipMemcpy(bvt.p, poison.data(),
                                 (std::size_t) kv * max_pos * sizeof(__half),
                                 hipMemcpyHostToDevice));
        MLA_TEST_CHECK(hipMemcpy(bq16.p, poison.data(),
                                 (std::size_t) kMlaW2Rows * kva * sizeof(__half),
                                 hipMemcpyHostToDevice));
        MLA_TEST_CHECK(hipMemcpy(bp16.p, poison.data(),
                                 (std::size_t) kMlaW2Rows * max_pos * sizeof(__half),
                                 hipMemcpyHostToDevice));
    }

    struct Case {
        int pos0;
        int T;
        const char* name;
    };
    const Case cases[] = {
        {0, 1, "T=1 fresh"},
        {0, 2, "T=2 fresh"},
        {0, 5, "T=5 fresh"},
        {57, 63, "T=63 tail history 57"},
        {0, 64, "T=64 one sub-batch"},
        {57, 65, "T=65 sub-batch tails"},
        {507, 130, "T=130 history 507 (odd)"},
        {8, 512, "T=512 full chunk with history"},
    };

    for (const Case& cs : cases) {
        const int T = cs.T;
        const int n_pos = cs.pos0 + T;

        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < qd; ++i) hq[(std::size_t) t * qd + i] = rnd(-1.f, 1.f);
            for (int i = 0; i < kva; ++i) hkva[(std::size_t) t * kva + i] = rnd(-2.f, 2.f);
            for (int i = 0; i < gd; ++i) hgate[(std::size_t) t * gd + i] = rnd(-4.f, 4.f);
            hpos[(std::size_t) t] = cs.pos0 + t;
        }
        for (int p = 0; p < cs.pos0; ++p) {
            for (int i = 0; i < kva; ++i) hcache[(std::size_t) p * kva + i] = rnd(-1.5f, 1.5f);
        }

        MLA_TEST_CHECK(hipMemcpy(dq.p, hq.data(), (std::size_t) T * qd * sizeof(float),
                                 hipMemcpyHostToDevice));
        MLA_TEST_CHECK(hipMemcpy(dkva.p, hkva.data(), (std::size_t) T * kva * sizeof(float),
                                 hipMemcpyHostToDevice));
        MLA_TEST_CHECK(hipMemcpy(dgate.p, hgate.data(), (std::size_t) T * gd * sizeof(float),
                                 hipMemcpyHostToDevice));
        MLA_TEST_CHECK(hipMemcpy(dpos.p, hpos.data(), (std::size_t) T * sizeof(int),
                                 hipMemcpyHostToDevice));
        MLA_TEST_CHECK(hipMemcpy(dcache.p, hcache.data(),
                                 (std::size_t) cs.pos0 * kva * sizeof(float),
                                 hipMemcpyHostToDevice));

        MlaInputs in;
        in.q = make_view((const float*) dq.p, qd, T);
        in.kv_a = make_view((const float*) dkva.p, kva, T);
        in.gate = make_view((const float*) dgate.p, gd, T);
        in.n_tokens = T;
        in.n_seqs = 1;

        MlaCache cache;
        cache.kv = (float*) dcache.p;
        cache.s_pos = kva;
        cache.s_seq = 0;
        cache.max_pos = max_pos;

        // ---- the shared stages, exactly as ForwardBatch calls them ----
        mla_rope(sh, (float*) dq.p + nope, /*s_head=*/sh.head_k, /*n_head=*/H,
                 T, qd, 0, 1, (const int*) dpos.p, stream);
        mla_rope(sh, (float*) dkva.p + kv, /*s_head=*/0, /*n_head=*/1,
                 T, kva, 0, 1, (const int*) dpos.p, stream);
        mla_latent_norm(sh, sc, w, (float*) dkva.p, T, 1, kva, 0, stream);
        mla_cache_append(sh, cache, (const float*) dkva.p, T, kva, 1, n_pos, stream);

        // ---- reference 1: the serial kernel ----
        mla_attn(sh, sc, w, cache, n_pos, in, make_view((float*) dout_ref.p, od, T),
                 (float*) dscores.p, stream);
        // ---- reference 2: today's batched-prefill kernel ----
        for (int tb = 0; tb < T; tb += kSubBatch) {
            const int tc = T - tb < kSubBatch ? T - tb : kSubBatch;
            mla_attn_tok(sh, sc, w, cache, n_pos, in,
                         make_view((float*) dout_tok.p, od, T),
                         (float*) dscores.p, tb, tc, T, stream);
        }
        // ---- candidate: the two-pass WMMA path, same sub-batching ----
        bool w2_ok = mla_w2_prep(sh, cache, n_pos, bufs, stream);
        if (w2_ok) {
            for (int tb = 0; tb < T; tb += kSubBatch) {
                const int tc = T - tb < kSubBatch ? T - tb : kSubBatch;
                if (!mla_w2_attn(sh, sc, w, n_pos, in,
                                 make_view((float*) dout_w2.p, od, T),
                                 (float*) dscores.p, bufs, tb, tc, T, stream)) {
                    w2_ok = false;
                    break;
                }
            }
        }
        if (!w2_ok) {
            std::printf("FAIL %-28s pos0=%d T=%d n_pos=%d : w2 audit refused\n",
                        cs.name, cs.pos0, T, n_pos);
            ++g_fail;
            continue;
        }
        MLA_TEST_CHECK(hipStreamSynchronize(stream));

        MLA_TEST_CHECK(hipMemcpy(href.data(), dout_ref.p, (std::size_t) T * od * sizeof(float),
                                 hipMemcpyDeviceToHost));
        MLA_TEST_CHECK(hipMemcpy(htok.data(), dout_tok.p, (std::size_t) T * od * sizeof(float),
                                 hipMemcpyDeviceToHost));
        MLA_TEST_CHECK(hipMemcpy(hw2.data(), dout_w2.p, (std::size_t) T * od * sizeof(float),
                                 hipMemcpyDeviceToHost));

        std::printf("%-28s pos0=%d T=%d n_pos=%d\n", cs.name, cs.pos0, T, n_pos);
        report(cs.name, "mla_attn", compare(hw2, href), 2e-3);
        report(cs.name, "mla_attn_t", compare(hw2, htok), 2e-3);
        report(cs.name, "tok_vs_ser", compare(htok, href), 2e-3);
        std::fflush(stdout);
    }

    std::printf("%s fail=%d\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}
