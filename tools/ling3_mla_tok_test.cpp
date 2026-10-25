// ling3_mla_tok_test.cpp — LS3 phase 2: unit gate for mla_attn_tok (the
// warp-pair tiled rewrite) against mla_attn (the serial kernel that passed
// every phase-1 and p2d gate), on synthetic tensors at the model's real
// shapes (32 heads, dk 192, dv 128, latent 512, kv_a 576). No model load:
// seconds on the GPU. Run inside gpu-run with no arguments.
//
// Both kernels run over the SAME roped / latent-normed / cache-appended
// inputs, so any diff beyond fp-reorder noise is a bug in the tiled kernel.
// Cases mirror the production caller: sub-batches of 64 inside a chunk,
// chunk_tokens = whole-chunk T, optional history before pos0.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "ling3_mla.hpp"

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
    const int od = sh.o_dim();     // 4096
    const int gd = sh.n_head;      // gate width = 32
    const int hv = sh.head_v;      // 128
    const int max_pos = 1024;
    const int max_T = 512;
    const int kSubBatch = 64;     // the caller's sub-batch cap

    MLA_TEST_CHECK(hipSetDevice(0));
    hipStream_t stream = nullptr;  // null stream, same as the production caller

    std::mt19937 rng(20261001);
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
    std::vector<float> href((std::size_t) max_T * od), htok((std::size_t) max_T * od);
    std::vector<int> hpos((std::size_t) max_T);
    Buf dq((std::size_t) max_T * qd * sizeof(float)),
        dkva((std::size_t) max_T * kva * sizeof(float)),
        dgate((std::size_t) max_T * gd * sizeof(float)),
        dcache((std::size_t) max_pos * kva * sizeof(float)),
        dpos((std::size_t) max_T * sizeof(int)),
        dout_ref((std::size_t) max_T * od * sizeof(float)),
        dout_tok((std::size_t) max_T * od * sizeof(float)),
        dscores((std::size_t) kSubBatch * H * max_pos * sizeof(float));

    struct Case {
        int pos0;
        int T;
        const char* name;
    };
    const Case cases[] = {
        {0, 1, "T=1 fresh"},
        {0, 5, "T=5 fresh"},
        {0, 64, "T=64 one sub-batch"},
        {0, 130, "T=130 sub-batch tails"},
        {500, 130, "T=130 with history"},
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
        // history rows only; the chunk's rows are written by cache_append
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

        // ---- reference: the serial kernel ----
        mla_attn(sh, sc, w, cache, n_pos, in, make_view((float*) dout_ref.p, od, T),
                 (float*) dscores.p, stream);

        // ---- candidate: the tiled kernel, sub-batched like the caller ----
        for (int tb = 0; tb < T; tb += kSubBatch) {
            const int tc = T - tb < kSubBatch ? T - tb : kSubBatch;
            mla_attn_tok(sh, sc, w, cache, n_pos, in, make_view((float*) dout_tok.p, od, T),
                         (float*) dscores.p, tb, tc, T, stream);
        }
        MLA_TEST_CHECK(hipStreamSynchronize(stream));

        MLA_TEST_CHECK(hipMemcpy(href.data(), dout_ref.p, (std::size_t) T * od * sizeof(float),
                                 hipMemcpyDeviceToHost));
        MLA_TEST_CHECK(hipMemcpy(htok.data(), dout_tok.p, (std::size_t) T * od * sizeof(float),
                                 hipMemcpyDeviceToHost));

        double maxd = 0.0, ref_scale = 0.0;
        int wt = 0, wh = 0, wr = 0;
        std::vector<double> per_t((std::size_t) T, 0.0);
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < H; ++h) {
                for (int r = 0; r < hv; ++r) {
                    const std::size_t idx = (std::size_t) t * od + (std::size_t) h * hv + r;
                    const double d = std::fabs((double) htok[idx] - (double) href[idx]);
                    if (d > maxd) {
                        maxd = d;
                        wt = t;
                        wh = h;
                        wr = r;
                    }
                    if (d > per_t[(std::size_t) t]) per_t[(std::size_t) t] = d;
                    const double a = std::fabs((double) href[idx]);
                    if (a > ref_scale) ref_scale = a;
                }
            }
        }

        const bool pass = maxd < 1e-3;
        if (!pass) ++g_fail;
        std::printf("%-28s pos0=%d T=%d n_pos=%d %s max_abs=%.3e (t=%d h=%d r=%d "
                    "ref_scale=%.2f)\n",
                    cs.name, cs.pos0, T, n_pos, pass ? "PASS" : "FAIL", maxd, wt, wh, wr,
                    ref_scale);
        if (!pass) {
            // worst 8 tokens, to localise where the divergence starts
            std::printf("  worst tokens:");
            for (int k = 0; k < 8 && k < T; ++k) {
                int bt = 0;
                for (int t = 0; t < T; ++t) {
                    if (per_t[(std::size_t) t] > per_t[(std::size_t) bt]) bt = t;
                }
                std::printf(" (t=%d %.2e)", bt, per_t[(std::size_t) bt]);
                per_t[(std::size_t) bt] = -1.0;
            }
            std::printf("\n");
        }
        std::fflush(stdout);
    }

    std::printf("%s fail=%d\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}
