// ling3_mla_prefill_test.cpp — LF1 SPEC3 unit gate for the PORT of the fork's
// tile flash-attention kernel (kernels/ling3_mla_tile.hip) to gufo's absorbed-
// MLA prefill. Same harness role as the wave-1/2 one, tile-only this time:
//
//   1. synthetic tensors at the real shapes: T = 1, 2, 5, 63, 64, 65, 130, 512,
//      n_pos not a multiple of the 64-row tile, and a long-history case;
//   2. a real captured q / cache / weights slice (LING3_MLA_CAPTURE=<dir>);
//   3. vs mla_attn_tok (the reference the model runs today) and, reported only,
//      the serial mla_attn.
//
// The tile kernel is NOT bit-identical to mla_attn_tok: its K/V are f16 (the
// fork's numerics), the online softmax reorders the position sum and the
// de-absorb is a separate batched GEMM. The gate is a RELATIVE tolerance
// (<= 2e-3 max_rel), plus a determinism check (two runs, byte-equal).
//
// Every index the tile kernel computes is enumerated here on the host against
// the test's own buffer extents before anything is launched (a GPU fault resets
// the box and every other lane's work on it).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ling3_mla.hpp"
#include "ling3_mla_tile.hpp"
// LX1 (wave 3): the w2 pair enters the bench and the fused WMMA flash
// candidate joins the gate — every case runs tok (reference), tile, w2 and
// fw; the bench times all four end to end.
#include "ling3_mla_w2.hpp"
#include "ling3_mla_fw.hpp"

namespace {

int g_fail = 0;
bool g_bench = false;

struct Buf {
    void* p = nullptr;
    std::size_t bytes = 0;
    explicit Buf(std::size_t b) : bytes(b) {
        if (b == 0) b = 1;
        if (hipMalloc(&p, b) != hipSuccess) p = nullptr;
    }
    ~Buf() {
        if (p != nullptr) hipFree(p);
    }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
};

#define MLA_TEST_CHECK(expr)                                                    \
    do {                                                                        \
        if ((expr) != hipSuccess) {                                             \
            std::printf("FAIL hip: %s at %s:%d\n", #expr, __FILE__, __LINE__);  \
            return 2;                                                           \
        }                                                                       \
    } while (0)

bool read_file(const std::string& path, void* dst, std::size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    const std::size_t n = std::fread(dst, 1, bytes, f);
    std::fclose(f);
    return n == bytes;
}

bool read_kv(const std::string& path, const char* key, long long* out) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) return false;
    char line[256];
    bool found = false;
    const std::size_t klen = std::strlen(key);
    while (std::fgets(line, sizeof line, f) != nullptr) {
        if (std::strncmp(line, key, klen) == 0 && line[klen] == ' ') {
            *out = std::strtoll(line + klen + 1, nullptr, 10);
            found = true;
            break;
        }
    }
    std::fclose(f);
    return found;
}

// ---------------------------------------------------------------------------
// Host-side index audit: enumerate exactly what attn_tile_kernel computes
// (2-token x 16-head blocks, 64-row position tiles, 64-dim K slices) and check
// every element against the extents this test allocated.
// ---------------------------------------------------------------------------
struct Extents {
    std::size_t q = 0, gate = 0, cache = 0, wk_b = 0, wv_b = 0, out = 0,
                latent = 0;
};

bool audit_indices(const ::ling3::MlaShape& sh, const Extents& ex, int n_pos,
                   int tok_base, int tok_count, int chunk_tokens, int n_tokens) {
    constexpr int kCols1 = 2, kCols2 = 16, kCols = kCols1 * kCols2;
    constexpr int kBatchFa = 64, kBatchK = 64;
    constexpr int kKva = 576, kKv = 512;
    long long checks = 0;
    bool ok = true;

    const int n_blocks_t = (tok_count + kCols1 - 1) / kCols1;
    const int n_blocks_h = sh.n_head / kCols2;
    if (n_blocks_h * kCols2 != sh.n_head) {
        std::printf("  audit: n_head=%d not a multiple of %d\n", sh.n_head,
                    kCols2);
        ok = false;
    }

    for (int bt = 0; bt < n_blocks_t; ++bt) {
        const int tok0 = tok_base + bt * kCols1;
        for (int bh = 0; bh < n_blocks_h; ++bh) {
            const int h0 = bh * kCols2;
            int k_sup = 0;
            for (int jc = 0; jc < kCols; ++jc) {
                const int t = tok0 + jc / kCols2;
                const int h = h0 + jc % kCols2;
                if (t < tok_base || t >= tok_base + tok_count) {
                    if (jc % kCols2 == 0 && (t < tok_base ||
                                             t >= tok_base + tok_count)) {
                        // tail block's spare token column: kernel must skip it
                        ++checks;
                    }
                    continue;
                }
                if (t < 0 || t >= n_tokens || h < 0 || h >= sh.n_head) {
                    std::printf("  audit: t=%d h=%d outside n_tokens=%d H=%d\n",
                                t, h, n_tokens, sh.n_head);
                    ok = false;
                }
                const int p_limit = (n_pos - chunk_tokens) + t + 1;
                if (p_limit < 1 || p_limit > n_pos) {
                    std::printf("  audit: p_limit=%d t=%d n_pos=%d\n", p_limit,
                                t, n_pos);
                    ok = false;
                }
                if (p_limit > k_sup) k_sup = p_limit;
                // Q row, wk_b rows, latent row, de-absorb row, gate, out
                const long long q_hi =
                    (long long) t * sh.q_dim() + (long long) h * sh.head_k +
                    sh.head_k;
                const long long wk_hi =
                    ((long long) h * kKv + (kKv - 1)) * sh.nope + sh.nope;
                const long long lat_hi =
                    (long long) t * ((long long) sh.n_head * kKv) +
                    (long long) h * kKv + kKv;
                const long long wv_hi =
                    ((long long) h * sh.head_v + (sh.head_v - 1)) * kKv + kKv;
                const long long o_hi =
                    (long long) t * sh.o_dim() + (long long) h * sh.head_v +
                    sh.head_v;
                const long long g_hi =
                    (long long) t * sh.n_head + h + 1;
                if ((std::size_t) q_hi > ex.q) {
                    std::printf("  audit: q t=%d h=%d\n", t, h);
                    ok = false;
                }
                if ((std::size_t) wk_hi > ex.wk_b) {
                    std::printf("  audit: wk_b h=%d\n", h);
                    ok = false;
                }
                if ((std::size_t) lat_hi > ex.latent) {
                    std::printf("  audit: latent t=%d h=%d\n", t, h);
                    ok = false;
                }
                if ((std::size_t) wv_hi > ex.wv_b) {
                    std::printf("  audit: wv_b h=%d\n", h);
                    ok = false;
                }
                if ((std::size_t) o_hi > ex.out) {
                    std::printf("  audit: out t=%d h=%d\n", t, h);
                    ok = false;
                }
                if ((std::size_t) g_hi > ex.gate) {
                    std::printf("  audit: gate t=%d h=%d\n", t, h);
                    ok = false;
                }
                checks += 6;
            }
            for (int p0 = 0; p0 < k_sup; p0 += kBatchFa) {
                const int npt = k_sup - p0 < kBatchFa ? k_sup - p0 : kBatchFa;
                if (npt < 1 || npt > kBatchFa || p0 + npt > n_pos) {
                    std::printf("  audit: tile p0=%d npt=%d\n", p0, npt);
                    ok = false;
                }
                if ((long long) (p0 + npt) * sh.kv_a() > (long long) ex.cache) {
                    std::printf("  audit: cache p0=%d npt=%d\n", p0, npt);
                    ok = false;
                }
                for (int k0 = 0; k0 < kKva; k0 += kBatchK) {
                    // K slice staged at KV_tmp[r*(32+4) + k0/2 + d2], d2 < 32
                    if (k0 + kBatchK > kKva) {
                        std::printf("  audit: k slice k0=%d\n", k0);
                        ok = false;
                    }
                    ++checks;
                }
                // P tile: kCols x kBatchFa f16 in LDS; V reads dims [0,512)
                checks += 2;
            }
        }
    }
    std::printf("  INDEX AUDIT  %s  (%lld checks; n_pos=%d tok %d+%d chunk=%d "
                "n_tokens=%d)\n",
                ok ? "PASS" : "FAIL", checks, n_pos, tok_base, tok_count,
                chunk_tokens, n_tokens);
    if (!ok) ++g_fail;
    return ok;
}

struct Metrics {
    double max_abs = 0.0, ref_scale = 0.0, max_rel = 0.0, max_elem_rel = 0.0;
    int wt = 0, wh = 0, wr = 0;
};

Metrics compare(const std::vector<float>& a, const std::vector<float>& b, int T,
                int H, int hv, std::size_t off = 0) {
    Metrics m;
    const std::size_t n = (std::size_t) T * (std::size_t) H * (std::size_t) hv;
    for (std::size_t k = 0; k < n; ++k) {
        const double av = (double) a[off + k], bv = (double) b[off + k];
        const double d = std::fabs(av - bv);
        const double s = std::fabs(bv);
        if (s > m.ref_scale) m.ref_scale = s;
        if (d > m.max_abs) {
            m.max_abs = d;
            m.wt = (int) (k / ((std::size_t) H * hv));
            m.wh = (int) ((k / (std::size_t) hv) % (std::size_t) H);
            m.wr = (int) (k % (std::size_t) hv);
        }
    }
    if (m.ref_scale > 0.0) m.max_rel = m.max_abs / m.ref_scale;
    for (std::size_t k = 0; k < n; ++k) {
        const double bv = (double) b[off + k];
        const double s = std::fabs(bv);
        if (s > 1e-3 * m.ref_scale) {
            const double er = std::fabs((double) a[k] - bv) / s;
            if (er > m.max_elem_rel) m.max_elem_rel = er;
        }
    }
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace ::ling3;
    // LX1: parse --bench BEFORE the case loop so the anchor cases bench too.
    const char* capdir = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) {
            g_bench = true;
        } else if (argv[i][0] != '-') {
            capdir = argv[i];
        }
    }
    const MlaShape sh;
    const MlaScalars sc;
    const int H = sh.n_head;    // 32
    const int nope = sh.nope;   // 128
    const int kv = sh.kv;       // 512
    const int kva = sh.kv_a();  // 576
    const int qd = sh.q_dim();  // 6144
    const int od = sh.o_dim();  // 4096
    const int gd = sh.n_head;   // 32
    const int hv = sh.head_v;   // 128
    const int kSubBatch = 64;   // the caller's mla_attn_tok sub-batch cap

    std::printf("ling3_mla_prefill unit gate  %s / %s  vs mla_attn_tok  "
                "(H=%d dk=%d dv=%d kv=%d kva=%d)\n",
                mla_tile_version(), mla_fw_version(), H, sh.head_k, hv, kv,
                kva);
    std::printf("NOTE: the tile kernel is NOT bit-identical to mla_attn_tok "
                "(f16 K/V,\n"
                "      reordered sums, de-absorb as a separate GEMM). Gate: "
                "max_rel <= 2e-3\n"
                "      and two runs byte-equal. Reference order is the 64-token\n"
                "      sub-batch schedule the model uses.\n");
    std::fflush(stdout);

    MLA_TEST_CHECK(hipSetDevice(0));
    hipStream_t stream = nullptr;

    std::mt19937 rng(20261003);
    auto rnd = [&](float lo, float hi) {
        std::uniform_real_distribution<float> d(lo, hi);
        return d(rng);
    };

    // ---- synthetic weights and latent-norm gamma ----
    const std::size_t wk_n = (std::size_t) H * kv * nope;
    const std::size_t wv_n = (std::size_t) H * hv * kv;
    std::vector<float> hwk(wk_n), hwv(wv_n), hnorm((std::size_t) kv);
    for (auto& v : hwk) v = rnd(-0.03f, 0.03f);
    for (auto& v : hwv) v = rnd(-0.05f, 0.05f);
    for (auto& v : hnorm) v = rnd(0.6f, 1.4f);
    Buf dwk(wk_n * sizeof(float)), dwv(wv_n * sizeof(float)),
        dnorm((std::size_t) kv * sizeof(float));
    MLA_TEST_CHECK(hipMemcpy(dwk.p, hwk.data(), wk_n * sizeof(float),
                             hipMemcpyHostToDevice));
    MLA_TEST_CHECK(hipMemcpy(dwv.p, hwv.data(), wv_n * sizeof(float),
                             hipMemcpyHostToDevice));
    MLA_TEST_CHECK(hipMemcpy(dnorm.p, hnorm.data(), kv * sizeof(float),
                             hipMemcpyHostToDevice));

    // LX1: deep enough for the bench anchors at n_pos 8,192 / 16,384 /
    // 32,768 / 57,344 (the spec's --bench points).
    const int max_pos = 57856;
    const int max_T = 1024;
    std::vector<float> hq((std::size_t) max_T * qd),
        hkva((std::size_t) max_T * kva), hgate((std::size_t) max_T * gd),
        hcache((std::size_t) max_pos * kva);
    std::vector<float> href((std::size_t) max_T * od),
        htile((std::size_t) max_T * od), htile2((std::size_t) max_T * od),
        hserial((std::size_t) max_T * od), hfw((std::size_t) max_T * od),
        hfw2((std::size_t) max_T * od), hw2((std::size_t) max_T * od);
    std::vector<int> hpos((std::size_t) max_T);

    Buf dq((std::size_t) max_T * qd * sizeof(float)),
        dkva((std::size_t) max_T * kva * sizeof(float)),
        dgate((std::size_t) max_T * gd * sizeof(float)),
        dcache((std::size_t) max_pos * kva * sizeof(float)),
        dpos((std::size_t) max_T * sizeof(int)),
        dout_ref((std::size_t) max_T * od * sizeof(float)),
        dout_tile((std::size_t) max_T * od * sizeof(float)),
        dout_tile2((std::size_t) max_T * od * sizeof(float)),
        dout_serial((std::size_t) max_T * od * sizeof(float)),
        dout_fw((std::size_t) max_T * od * sizeof(float)),
        dout_fw2((std::size_t) max_T * od * sizeof(float)),
        dout_w2((std::size_t) max_T * od * sizeof(float)),
        dscores((std::size_t) kSubBatch * H * max_pos * sizeof(float));
    // LX1: the w2 staging (k16/vt are shared with the fused path) and the
    // fused path's own full-chunk q16/o/rowsum, sized once for max_pos.
    const int w2cap = (max_pos + 63) / 64 * 64;
    Buf dw2k16((std::size_t) w2cap * kva * sizeof(__half)),
        dw2vt((std::size_t) kv * (std::size_t) w2cap * sizeof(__half)),
        dw2q16((std::size_t) kMlaW2Rows * kva * sizeof(__half)),
        dw2p16((std::size_t) kMlaW2Rows * (std::size_t) w2cap * sizeof(__half)),
        dw2o((std::size_t) kMlaW2Rows * kv * sizeof(float)),
        dw2sum((std::size_t) kMlaW2Rows * sizeof(float)),
        dfwq16((std::size_t) kMlaFwRows * kva * sizeof(__half)),
        dfwo((std::size_t) kMlaFwRows * kv * sizeof(float)),
        dfwsum((std::size_t) kMlaFwRows * sizeof(float)),
        dfwqt((std::size_t) kMlaFwRows * nope * sizeof(float));
    MlaW2Bufs w2b{};
    w2b.k16 = (__half*) dw2k16.p;
    w2b.vt = (__half*) dw2vt.p;
    w2b.q16 = (__half*) dw2q16.p;
    w2b.p16 = (__half*) dw2p16.p;
    w2b.o = (float*) dw2o.p;
    w2b.rowsum = (float*) dw2sum.p;
    w2b.cap = w2cap;
    MlaFwBufs fwb{};
    fwb.q16 = (__half*) dfwq16.p;
    fwb.o = (float*) dfwo.p;
    fwb.rowsum = (float*) dfwsum.p;
    fwb.qt = (float*) dfwqt.p;
    fwb.rows_cap = kMlaFwRows;
    if (dq.p == nullptr || dcache.p == nullptr || dout_ref.p == nullptr ||
        dscores.p == nullptr || dpos.p == nullptr) {
        std::printf("FAIL: device alloc (max_pos=%d max_T=%d)\n", max_pos, max_T);
        return 2;
    }

    MlaWeights w;
    w.wk_b = (const float*) dwk.p;
    w.wv_b = (const float*) dwv.p;
    w.kv_a_norm = (const float*) dnorm.p;

    // -------------------------------------------------------------------
    // One case = shared stages on the device (rope / latent norm / cache
    // append, exactly as ForwardBatch calls them) + the three attentions.
    // `cache_rows` is how many rows of `hcache` to preload as history; the
    // chunk's own rows are written by mla_cache_append when `stage` is true,
    // or taken straight from `hcache` for a captured case.
    // -------------------------------------------------------------------
    struct Args {
        const char* name;
        int pos0, T, n_pos, chunk_tokens, n_tokens, cache_rows;
        bool stage;
        const float* wk_src;
        const float* wv_src;
        std::size_t wk_floats, wv_floats, q_floats, gate_floats, cache_floats;
        // LX1 SPEC2 (RV17 1e/4): fw-only coverage — a partial chunk range
        // (the reference still computes the FULL chunk, fw is gated on rows
        // [tok_base, T)), and history score-shape probes (one history row
        // set to spike_val in all 576 dims: spiky softmax mass / the H1
        // chunk-max-drop > 88 regime).
        int tok_base = 0;
        int spike_pos = -1;
        float spike_val = 0.0f;
        // The H1 probe is OUT OF DISTRIBUTION by construction: a > 88
        // chunk-max drop needs |S| >= ~90, where f16 K/V rounding alone
        // (the class tile/w2/fw share) perturbs softmax mass past the
        // 2e-3 relative gate — the SHIPPING tile kernel measures 1.5e-2 on
        // this very case. Its gate is therefore finite + deterministic +
        // the regime PROVED (see the H1 PROOF line); max_rel is reported.
        bool h1_probe = false;
    };

    auto run_case = [&](const Args& a) -> int {
        const int n_pos = a.n_pos;
        const std::size_t q_bytes = (std::size_t) a.n_tokens * qd * sizeof(float);
        const std::size_t g_bytes = (std::size_t) a.n_tokens * gd * sizeof(float);
        MLA_TEST_CHECK(hipMemcpy(dq.p, hq.data(), q_bytes, hipMemcpyHostToDevice));
        MLA_TEST_CHECK(
            hipMemcpy(dgate.p, hgate.data(), g_bytes, hipMemcpyHostToDevice));
        MLA_TEST_CHECK(hipMemcpy(dcache.p, hcache.data(),
                                 (std::size_t) a.cache_rows * kva * sizeof(float),
                                 hipMemcpyHostToDevice));
        if (a.wk_src != nullptr) {
            MLA_TEST_CHECK(hipMemcpy(dwk.p, a.wk_src,
                                     a.wk_floats * sizeof(float),
                                     hipMemcpyHostToDevice));
            MLA_TEST_CHECK(hipMemcpy(dwv.p, a.wv_src,
                                     a.wv_floats * sizeof(float),
                                     hipMemcpyHostToDevice));
        }

        if (a.stage) {
            for (int t = 0; t < a.T; ++t) hpos[(std::size_t) t] = a.pos0 + t;
            MLA_TEST_CHECK(hipMemcpy(dpos.p, hpos.data(),
                                     (std::size_t) a.T * sizeof(int),
                                     hipMemcpyHostToDevice));
            mla_rope(sh, (float*) dq.p + nope, /*s_head=*/sh.head_k, /*n_head=*/H,
                     a.T, qd, 0, 1, (const int*) dpos.p, stream);
            MLA_TEST_CHECK(hipMemcpy(dkva.p, hkva.data(),
                                     (std::size_t) a.T * kva * sizeof(float),
                                     hipMemcpyHostToDevice));
            mla_rope(sh, (float*) dkva.p + kv, /*s_head=*/0, /*n_head=*/1, a.T,
                     kva, 0, 1, (const int*) dpos.p, stream);
            mla_latent_norm(sh, sc, w, (float*) dkva.p, a.T, 1, kva, 0, stream);
            MlaCache c0;
            c0.kv = (float*) dcache.p;
            c0.s_pos = kva;
            c0.s_seq = 0;
            c0.max_pos = max_pos;
            mla_cache_append(sh, c0, (const float*) dkva.p, a.T, kva, 1, n_pos,
                             stream);
        }
        MLA_TEST_CHECK(hipStreamSynchronize(stream));

        MlaInputs in;
        in.q = make_view((const float*) dq.p, qd, a.n_tokens);
        in.kv_a = make_view((const float*) dkva.p, kva, a.n_tokens);
        in.gate = make_view((const float*) dgate.p, gd, a.n_tokens);
        in.n_tokens = a.n_tokens;
        in.n_seqs = 1;

        MlaCache cache;
        cache.kv = (float*) dcache.p;
        cache.s_pos = kva;
        cache.s_seq = 0;
        cache.max_pos = max_pos;

        if (!mla_tile_supported(sh, cache)) {
            std::printf("%-28s FAIL  (shapes unsupported by the tile kernel)\n",
                        a.name);
            ++g_fail;
            return 0;
        }

        Extents ex;
        ex.q = a.q_floats;
        ex.gate = a.gate_floats;
        ex.cache = a.cache_floats;
        ex.wk_b = a.wk_floats;
        ex.wv_b = a.wv_floats;
        ex.out = (std::size_t) a.T * od;
        ex.latent = (std::size_t) a.n_tokens * (std::size_t) H * (std::size_t) kv;
        audit_indices(sh, ex, n_pos, 0, a.T, a.chunk_tokens, a.n_tokens);

        // reference: LS3's tiled kernel, sub-batched exactly like the caller
        for (int tb = 0; tb < a.T; tb += kSubBatch) {
            const int tc = a.T - tb < kSubBatch ? a.T - tb : kSubBatch;
            mla_attn_tok(sh, sc, w, cache, n_pos, in,
                         make_view((float*) dout_ref.p, od, a.n_tokens),
                         (float*) dscores.p, tb, tc, a.chunk_tokens, stream);
        }
        // candidate: the ported tile kernel, the whole call in one attention
        // launch (+ its f16 conversion and de-absorb GEMM)
        mla_attn_tile(sh, sc, w, cache, n_pos, in,
                      make_view((float*) dout_tile.p, od, a.n_tokens),
                      (float*) dscores.p, 0, a.T, a.chunk_tokens, stream);
        // cross-check: the serial kernel (reported, not gated; skipped at
        // the deep bench anchors, where it is minutes-slow and reported
        // nowhere)
        if (n_pos <= 8448) {
            mla_attn(sh, sc, w, cache, n_pos, in,
                     make_view((float*) dout_serial.p, od, a.n_tokens),
                     (float*) dscores.p, stream);
        }
        MLA_TEST_CHECK(hipStreamSynchronize(stream));

        const std::size_t ob = (std::size_t) a.T * od * sizeof(float);
        MLA_TEST_CHECK(
            hipMemcpy(href.data(), dout_ref.p, ob, hipMemcpyDeviceToHost));
        MLA_TEST_CHECK(
            hipMemcpy(htile.data(), dout_tile.p, ob, hipMemcpyDeviceToHost));
        MLA_TEST_CHECK(
            hipMemcpy(hserial.data(), dout_serial.p, ob, hipMemcpyDeviceToHost));

        const Metrics m = compare(htile, href, a.T, H, hv);
        const Metrics s = compare(htile, hserial, a.T, H, hv);
        // H1 probe: out-of-distribution scores (see Args) — gate tile on
        // finiteness instead of the 2e-3 tolerance, and report max_rel.
        int tl_nonfinite = -1;   // -1 = not scanned (not the H1 probe)
        if (a.h1_probe) {
            tl_nonfinite = 0;
            for (std::size_t k = 0; k < (std::size_t) a.T * od; ++k) {
                if (!std::isfinite(htile[k])) ++tl_nonfinite;
            }
            if (tl_nonfinite != 0) ++g_fail;
        }
        const bool pass = a.h1_probe ? tl_nonfinite == 0 : m.max_rel <= 2e-3;
        if (!pass) ++g_fail;

        // determinism: the same launch twice must be byte-equal
        mla_attn_tile(sh, sc, w, cache, n_pos, in,
                      make_view((float*) dout_tile2.p, od, a.n_tokens),
                      (float*) dscores.p, 0, a.T, a.chunk_tokens, stream);
        MLA_TEST_CHECK(hipStreamSynchronize(stream));
        MLA_TEST_CHECK(
            hipMemcpy(htile2.data(), dout_tile2.p, ob, hipMemcpyDeviceToHost));
        const bool det =
            std::memcmp(htile.data(), htile2.data(), ob) == 0;
        if (!det) ++g_fail;

        // LX1: the w2 pair (reported) and the fused WMMA flash candidate
        // (gated exactly like tile: max_rel <= 2e-3 + byte-equal twice).
        bool w2_ok = mla_w2_prep(sh, cache, n_pos, w2b, stream);
        if (w2_ok) {
            for (int tb = 0; tb < a.T; tb += kSubBatch) {
                const int tc = a.T - tb < kSubBatch ? a.T - tb : kSubBatch;
                w2_ok = mla_w2_attn(sh, sc, w, n_pos, in,
                                    make_view((float*) dout_w2.p, od,
                                              a.n_tokens),
                                    (float*) dscores.p, w2b, tb, tc,
                                    a.chunk_tokens, stream);
                if (!w2_ok) break;
            }
        }
        // LX1 SPEC2 (RV17 1e): fw may be asked for a PARTIAL chunk range —
        // rows [tok_base, T) only. The reference above computed the full
        // chunk, so the gate compares the written range at an offset.
        const int fw_tcount = a.T - a.tok_base;
        const std::size_t fw_off = (std::size_t) a.tok_base * od;
        const std::size_t fw_ob =
            (std::size_t) fw_tcount * od * sizeof(float);
        bool fw_ok = mla_w2_prep(sh, cache, n_pos, w2b, stream);
        fw_ok = fw_ok &&
                mla_fw_attn(sh, sc, w, n_pos, in,
                            make_view((float*) dout_fw.p, od, a.n_tokens),
                            w2b, fwb, a.tok_base, fw_tcount,
                            a.chunk_tokens, stream);
        MLA_TEST_CHECK(hipStreamSynchronize(stream));
        MLA_TEST_CHECK(
            hipMemcpy(hw2.data(), dout_w2.p, ob, hipMemcpyDeviceToHost));
        MLA_TEST_CHECK(hipMemcpy(hfw.data(), dout_fw.p, ob,
                                 hipMemcpyDeviceToHost));

        Metrics mf = compare(hfw, href, fw_tcount, H, hv, fw_off);
        // RV17 H1: any NaN/Inf in the fused path's written rows is a
        // failure in its own right (the pre-H1 form produced inf -> NaN
        // here when a later chunk's max dropped > ~88 below the running
        // max).
        int fw_nonfinite = 0;
        for (std::size_t k = fw_off; k < fw_off + fw_ob / sizeof(float); ++k) {
            if (!std::isfinite(hfw[k])) ++fw_nonfinite;
        }
        if (fw_nonfinite != 0) ++g_fail;
        bool fw_det = false;
        if (fw_ok) {
            mla_fw_attn(sh, sc, w, n_pos, in,
                        make_view((float*) dout_fw2.p, od, a.n_tokens),
                        w2b, fwb, a.tok_base, fw_tcount, a.chunk_tokens,
                        stream);
            MLA_TEST_CHECK(hipStreamSynchronize(stream));
            MLA_TEST_CHECK(hipMemcpy(hfw2.data(), dout_fw2.p, ob,
                                     hipMemcpyDeviceToHost));
            fw_det = std::memcmp(hfw.data() + fw_off, hfw2.data() + fw_off,
                                 fw_ob) == 0;
            if (!fw_det) ++g_fail;
        }
        // LX1 SPEC2 (RV17 4.5): PROVE the H1 probe exercises the regime.
        // attn_tok does not dump scores in release builds, so derive it
        // from the fused kernel's OWN q16 (f16, copied back): chunk 0
        // contains the spike row (spike_val in every dim, exactly
        // representable in f16), so row r's chunk-0 score there is
        // spike_val * sum(q16[r]); every OTHER cache row the row can see
        // is bounded (history rows |k_d| <= 1.5 by construction, the
        // chunk's own normed rows conservatively |k_d| < 4), so the
        // later-chunk max is at most 4 * sum(|q16[r]|). Rows where the
        // guaranteed chunk-0 max beats that bound by > 88 are exactly
        // where the pre-H1 rescale expf(m_old - mx) overflowed inf -> NaN.
        if (a.h1_probe) {
            std::vector<__half> hq16((std::size_t) kMlaFwRows * kva);
            MLA_TEST_CHECK(hipMemcpy(hq16.data(), fwb.q16,
                                     hq16.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost));
            long long regime = 0;
            for (int r = 0; r < a.T * H; ++r) {
                float s0 = 0.0f, abs_sum = 0.0f;
                for (int d = 0; d < kva; ++d) {
                    const float v =
                        (float) hq16[(std::size_t) r * kva + d];
                    s0 += v;
                    abs_sum += std::fabs(v);
                }
                if (a.spike_val * s0 - 4.0f * abs_sum > 88.0f) ++regime;
            }
            std::printf("H1 PROOF  %lld/%d rows with chunk-0 max provably "
                        "> 88 above every later chunk's max (pre-H1: NaN "
                        "there)\n",
                        regime, a.T * H);
            if (regime == 0) ++g_fail;   // the probe must hit the regime
        }
        const bool fw_pass = fw_ok && fw_det && fw_nonfinite == 0 &&
                             (a.h1_probe || mf.max_rel <= 2e-3);
        if (!fw_pass) ++g_fail;
        const Metrics mw2r = w2_ok ? compare(hw2, href, a.T, H, hv)
                                   : Metrics{};

        std::printf("%-28s pos0=%d T=%d n_pos=%d %s  vs tok  max_abs=%.3e "
                    "ref_scale=%.4f max_rel=%.3e elem_rel=%.3e (t=%d h=%d r=%d)"
                    "  | vs serial max_rel=%.3e  | det %s\n",
                    a.name, a.pos0, a.T, n_pos, pass ? "PASS" : "FAIL",
                    m.max_abs, m.ref_scale, m.max_rel, m.max_elem_rel, m.wt,
                    m.wh, m.wr, s.max_rel, det ? "PASS" : "FAIL");
        std::printf(
            "%-28s   fw %s (max_rel=%.3e (t=%d h=%d r=%d), det %s%s%s)"
            "  | w2 %s (max_rel=%.3e)\n",
            a.name, fw_pass ? "PASS" : "FAIL", mf.max_rel,
            mf.wt + a.tok_base, mf.wh, mf.wr, fw_det ? "PASS" : "FAIL",
            fw_nonfinite != 0 ? ", NONFINITE!" : "",
            a.h1_probe ? "; gate=finite+det (OOD tol exempt)" : "",
            w2_ok ? "ok" : "REFUSED", mw2r.max_rel);

        // --bench: device-side wall time of tok (64-token sub-batches) vs tile
        // (whole call) on this case. Same inputs, production launch shapes.
        if (g_bench) {
            constexpr int kIter = 5;
            hipEvent_t e0, e1;
            if (hipEventCreate(&e0) == hipSuccess &&
                hipEventCreate(&e1) == hipSuccess) {
                auto time_tok = [&]() -> float {
                    for (int it = 0; it < kIter; ++it) {
                        for (int tb = 0; tb < a.T; tb += kSubBatch) {
                            const int tc = a.T - tb < kSubBatch ? a.T - tb : kSubBatch;
                            mla_attn_tok(sh, sc, w, cache, n_pos, in,
                                         make_view((float*) dout_ref.p, od, a.n_tokens),
                                         (float*) dscores.p, tb, tc, a.chunk_tokens,
                                         stream);
                        }
                    }
                    hipEventRecord(e0, stream);
                    for (int it = 0; it < kIter; ++it) {
                        for (int tb = 0; tb < a.T; tb += kSubBatch) {
                            const int tc = a.T - tb < kSubBatch ? a.T - tb : kSubBatch;
                            mla_attn_tok(sh, sc, w, cache, n_pos, in,
                                         make_view((float*) dout_ref.p, od, a.n_tokens),
                                         (float*) dscores.p, tb, tc, a.chunk_tokens,
                                         stream);
                        }
                    }
                    hipEventRecord(e1, stream);
                    hipEventSynchronize(e1);
                    float ms = 0.f;
                    hipEventElapsedTime(&ms, e0, e1);
                    return ms / kIter;
                };
                auto time_tile = [&]() -> float {
                    for (int it = 0; it < kIter; ++it) {
                        mla_attn_tile(sh, sc, w, cache, n_pos, in,
                                      make_view((float*) dout_tile.p, od, a.n_tokens),
                                      (float*) dscores.p, 0, a.T, a.chunk_tokens,
                                      stream);
                    }
                    hipEventRecord(e0, stream);
                    for (int it = 0; it < kIter; ++it) {
                        mla_attn_tile(sh, sc, w, cache, n_pos, in,
                                      make_view((float*) dout_tile.p, od, a.n_tokens),
                                      (float*) dscores.p, 0, a.T, a.chunk_tokens,
                                      stream);
                    }
                    hipEventRecord(e1, stream);
                    hipEventSynchronize(e1);
                    float ms = 0.f;
                    hipEventElapsedTime(&ms, e0, e1);
                    return ms / kIter;
                };
                const float t_tok = time_tok();
                const float t_tl = time_tile();
                // LX1: the w2 pair and the fused flash, end to end (prep
                // included — that is what one layer-call pays).
                auto time_w2 = [&]() -> float {
                    for (int it = 0; it < kIter; ++it) {
                        mla_w2_prep(sh, cache, n_pos, w2b, stream);
                        for (int tb = 0; tb < a.T; tb += kSubBatch) {
                            const int tc =
                                a.T - tb < kSubBatch ? a.T - tb : kSubBatch;
                            mla_w2_attn(sh, sc, w, n_pos, in,
                                        make_view((float*) dout_w2.p, od,
                                                  a.n_tokens),
                                        (float*) dscores.p, w2b, tb, tc,
                                        a.chunk_tokens, stream);
                        }
                    }
                    hipEventRecord(e0, stream);
                    for (int it = 0; it < kIter; ++it) {
                        mla_w2_prep(sh, cache, n_pos, w2b, stream);
                        for (int tb = 0; tb < a.T; tb += kSubBatch) {
                            const int tc =
                                a.T - tb < kSubBatch ? a.T - tb : kSubBatch;
                            mla_w2_attn(sh, sc, w, n_pos, in,
                                        make_view((float*) dout_w2.p, od,
                                                  a.n_tokens),
                                        (float*) dscores.p, w2b, tb, tc,
                                        a.chunk_tokens, stream);
                        }
                    }
                    hipEventRecord(e1, stream);
                    hipEventSynchronize(e1);
                    float ms = 0.f;
                    hipEventElapsedTime(&ms, e0, e1);
                    return ms / kIter;
                };
                auto time_fw = [&]() -> float {
                    for (int it = 0; it < kIter; ++it) {
                        mla_w2_prep(sh, cache, n_pos, w2b, stream);
                        mla_fw_attn(sh, sc, w, n_pos, in,
                                    make_view((float*) dout_fw.p, od,
                                              a.n_tokens),
                                    w2b, fwb, a.tok_base, fw_tcount,
                                    a.chunk_tokens, stream);
                    }
                    hipEventRecord(e0, stream);
                    for (int it = 0; it < kIter; ++it) {
                        mla_w2_prep(sh, cache, n_pos, w2b, stream);
                        mla_fw_attn(sh, sc, w, n_pos, in,
                                    make_view((float*) dout_fw.p, od,
                                              a.n_tokens),
                                    w2b, fwb, a.tok_base, fw_tcount,
                                    a.chunk_tokens, stream);
                    }
                    hipEventRecord(e1, stream);
                    hipEventSynchronize(e1);
                    float ms = 0.f;
                    hipEventElapsedTime(&ms, e0, e1);
                    return ms / kIter;
                };
                const float t_w2 = time_w2();
                const float t_fw = time_fw();
                std::printf("  BENCH %s  n_pos=%d T=%d : mla_attn_tok %.2f ms   "
                            "mla_attn_tile %.2f ms   w2 %.2f ms   fw %.2f ms "
                            "(fw/tok %.2fx, fw/tile %.2fx)\n",
                            a.name, n_pos, a.T, t_tok, t_tl, t_w2, t_fw,
                            t_fw > 0.f ? t_tok / t_fw : 0.f,
                            t_fw > 0.f ? t_tl / t_fw : 0.f);
                hipEventDestroy(e0);
                hipEventDestroy(e1);
            }
        }
        std::fflush(stdout);
        return 0;
    };

    struct Case {
        int pos0, T;
        const char* name;
        int tok_base = 0;
        int spike_pos = -1;
        float spike_val = 0.0f;
        bool h1_probe = false;
    };
    // T = 1, 2, 5, 63, 64, 65, 130, 512; n_pos not a multiple of the 64-row
    // tile (5, 65, 130, 519 all leave a tail tile) and a long-history case.
    const Case cases[] = {
        {0, 1, "T=1 n_pos=1 (pos 1)"},
        {0, 2, "T=2 n_pos=2 (pos 1,2)"},
        {0, 3, "T=3 n_pos=3 (odd)"},
        {0, 5, "T=5 n_pos=5 tail tile"},
        {0, 7, "T=7 n_pos=7"},
        {0, 9, "T=9 n_pos=9 (odd)"},
        {0, 17, "T=17 n_pos=17 (odd)"},
        {0, 33, "T=33 n_pos=33 (odd)"},
        {0, 63, "T=63 n_pos=63"},
        {0, 64, "T=64 n_pos=64"},
        {0, 65, "T=65 n_pos=65"},
        {0, 127, "T=127 n_pos=127"},
        {0, 128, "T=128 n_pos=128"},
        {0, 129, "T=129 n_pos=129"},
        {0, 130, "T=130 n_pos=130 sub tails"},
        {0, 255, "T=255 n_pos=255"},
        {0, 256, "T=256 n_pos=256"},
        {0, 257, "T=257 n_pos=257"},
        {0, 511, "T=511 n_pos=511 odd full"},
        {500, 130, "T=130 pos0=500 history"},
        {7, 512, "T=512 n_pos=519 tail"},
        {0, 512, "T=512 n_pos=512 full chunk"},
        {4096, 512, "T=512 pos0=4096 deep history"},
        // LX1 SPEC2 (RV17 1e/4): fw partial-chunk coverage — the reference
        // computes the full 512-token chunk, fw is gated on rows [256,512)
        // and [511,512).
        {0, 512, "fw tok 512/256/256", /*tok_base=*/256},
        {0, 512, "fw tok 512/511/1", /*tok_base=*/511},
        // LX1 SPEC2 (RV17 2.4/4): score-shape probes. One history row is
        // set to spike_val in all 576 dims, so every query row's score at
        // that position is ~spike_val * sum(q16): a spiky softmax (mass
        // concentrates at one position; scores stay modest) and the H1
        // regime (chunk-0 max far above later chunks' — a drop > 88 that
        // overflowed the pre-H1 rescale to inf -> NaN).
        {512, 512, "spiky row (mass at pos 100)", 0, /*spike_pos=*/100,
         /*spike_val=*/25.0f},
        {512, 512, "H1 probe chunk-max drop>88", 0, /*spike_pos=*/0,
         /*spike_val=*/600.0f, /*h1_probe=*/true},
        // LX1: the spec's --bench anchors (T 512), each also gated.
        {7680, 512, "T=512 anchor n_pos=8192"},
        {15872, 512, "T=512 anchor n_pos=16384"},
        {32256, 512, "T=512 anchor n_pos=32768"},
        {56832, 512, "T=512 anchor n_pos=57344"},
    };

    for (const Case& cs : cases) {
        const int T = cs.T;
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < qd; ++i) hq[(std::size_t) t * qd + i] = rnd(-1.f, 1.f);
            for (int i = 0; i < kva; ++i)
                hkva[(std::size_t) t * kva + i] = rnd(-2.f, 2.f);
            for (int i = 0; i < gd; ++i)
                hgate[(std::size_t) t * gd + i] = rnd(-4.f, 4.f);
        }
        for (int p = 0; p < cs.pos0; ++p) {
            for (int i = 0; i < kva; ++i)
                hcache[(std::size_t) p * kva + i] = rnd(-1.5f, 1.5f);
        }
        if (cs.spike_pos >= 0 && cs.spike_pos < cs.pos0) {
            for (int i = 0; i < kva; ++i)
                hcache[(std::size_t) cs.spike_pos * kva + i] = cs.spike_val;
        }
        Args a;
        a.name = cs.name;
        a.pos0 = cs.pos0;
        a.T = T;
        a.n_pos = cs.pos0 + T;
        a.chunk_tokens = T;
        a.n_tokens = T;
        a.cache_rows = cs.pos0;
        a.stage = true;
        a.wk_src = nullptr;
        a.wv_src = nullptr;
        a.wk_floats = wk_n;
        a.wv_floats = wv_n;
        a.q_floats = (std::size_t) T * qd;
        a.gate_floats = (std::size_t) T * gd;
        a.cache_floats = (std::size_t) max_pos * kva;
        a.tok_base = cs.tok_base;
        a.spike_pos = cs.spike_pos;
        a.spike_val = cs.spike_val;
        a.h1_probe = cs.h1_probe;
        const int rc = run_case(a);
        if (rc != 0) return rc;
    }

    // -------------------------------------------------------------------
    // LX1 SPEC2 (RV17 1f/4): the REFUSAL gate — fw_audit must refuse each
    // of these BEFORE any launch (fail-closed; the caller falls back to its
    // proven path). Shapes are chosen so a wrongly-accepted launch would
    // stay inside the allocated extents wherever possible (head_v 64's
    // wv_b reads and out writes stay below the real strides; the
    // chunk_tokens=2 H3 case keeps every index in bounds) — the audit is
    // still the barrier under test. Buffers hold the last case's staged
    // data; nothing here should launch at all.
    // -------------------------------------------------------------------
    {
        MlaInputs rin;   // a small valid base call: T=1 at n_pos=64
        rin.q = make_view((const float*) dq.p, qd, 1);
        rin.kv_a = make_view((const float*) dkva.p, kva, 1);
        rin.gate = make_view((const float*) dgate.p, gd, 1);
        rin.n_tokens = 1;
        rin.n_seqs = 1;
        auto expect_refuse = [&](const char* why, const MlaShape& s2,
                                 const MlaInputs& i2, const View& o2,
                                 MlaFwBufs b2, int n_pos2, int tb, int tc,
                                 int ct) {
            const bool launched = mla_fw_attn(s2, sc, w, n_pos2, i2, o2, w2b,
                                              b2, tb, tc, ct, stream);
            const bool ok = !launched;
            if (!ok) ++g_fail;
            std::printf("REFUSE %-26s %s\n", why, ok ? "PASS" : "FAIL");
        };
        std::printf("-- fw refusal gate (fw_audit must return false)\n");
        {
            MlaInputs i2 = rin;
            i2.n_seqs = 2;
            expect_refuse("n_seqs=2", sh, i2,
                          make_view((float*) dout_fw.p, od, 1), fwb, 64, 0, 1,
                          1);
        }
        expect_refuse("chunk_tokens=513", sh, rin,
                      make_view((float*) dout_fw.p, od, 1), fwb, 1024, 0, 1,
                      513);
        expect_refuse("chunk>n_tokens (H3)", sh, rin,
                      make_view((float*) dout_fw.p, od, 1), fwb, 64, 0, 1, 2);
        {
            MlaShape s2 = sh;
            s2.head_v = 64;   // H2: deabsorb bakes head_v = 128
            expect_refuse("head_v=64 (H2)", s2, rin,
                          make_view((float*) dout_fw.p, 32 * 64, 1), fwb, 64,
                          0, 1, 1);
        }
        {
            MlaFwBufs b2 = fwb;
            b2.q16 = nullptr;
            expect_refuse("null q16", sh, rin,
                          make_view((float*) dout_fw.p, od, 1), b2, 64, 0, 1,
                          1);
        }
        {
            MlaInputs i2 = rin;
            i2.q = make_view((const float*) dq.p, qd + 8, 1);
            expect_refuse("bad q stride", sh, i2,
                          make_view((float*) dout_fw.p, od, 1), fwb, 64, 0, 1,
                          1);
        }
        expect_refuse("n_pos=cap+1", sh, rin,
                      make_view((float*) dout_fw.p, od, 1), fwb, w2b.cap + 1,
                      0, 1, 1);
        std::fflush(stdout);
    }

    // -------------------------------------------------------------------
    // Real captured q / cache / weights from a ~13.8 K prompt.
    // (capdir / g_bench were parsed at the top of main — LX1.)
    // -------------------------------------------------------------------
    if (capdir == nullptr || capdir[0] == '\0') {
        std::printf("REAL captured q/cache  FAIL  (no capture dir: run a real "
                    "prompt with LING3_MLA_CAPTURE=<dir>, then pass <dir> as "
                    "argv[1])\n");
        ++g_fail;
    } else {
        const std::string dir(capdir);
        const std::string hdr = dir + "/lf1_capture.txt";
        long long n_tokens = 0, n_pos = 0, tok_base = 0, tok_count = 0,
                  chunk_tokens = 0, q_floats = 0, cache_floats = 0,
                  wk_floats = 0, wv_floats = 0, gate_floats = 0, hh = 0, kkv = 0;
        const bool hdr_ok =
            read_kv(hdr, "n_head", &hh) && read_kv(hdr, "kv", &kkv) &&
            read_kv(hdr, "n_tokens", &n_tokens) && read_kv(hdr, "n_pos", &n_pos) &&
            read_kv(hdr, "tok_base", &tok_base) &&
            read_kv(hdr, "tok_count", &tok_count) &&
            read_kv(hdr, "chunk_tokens", &chunk_tokens) &&
            read_kv(hdr, "q_floats", &q_floats) &&
            read_kv(hdr, "cache_floats", &cache_floats) &&
            read_kv(hdr, "wk_b_floats", &wk_floats) &&
            read_kv(hdr, "wv_b_floats", &wv_floats) &&
            read_kv(hdr, "gate_floats", &gate_floats);
        std::printf("REAL captured q/cache  %s  n_tokens=%lld n_pos=%lld "
                    "tok %lld+%lld chunk=%lld q=%lld cache=%lld\n",
                    dir.c_str(), n_tokens, n_pos, tok_base, tok_count,
                    chunk_tokens, q_floats, cache_floats);
        if (!hdr_ok || hh != H || kkv != kv || n_tokens <= 0 || n_tokens > max_T ||
            n_pos <= 0 || n_pos > max_pos || tok_count <= 0 ||
            q_floats <= 0 || cache_floats <= 0) {
            std::printf("REAL captured q/cache  FAIL  (header unreadable or "
                        "outside the test's capacity)\n");
            ++g_fail;
        } else {
            std::vector<float> cq((std::size_t) q_floats),
                cc((std::size_t) cache_floats), cw((std::size_t) wk_floats),
                cv((std::size_t) wv_floats), cg((std::size_t) gate_floats);
            const bool loaded =
                read_file(dir + "/q.bin", cq.data(), cq.size() * sizeof(float)) &&
                read_file(dir + "/cache.bin", cc.data(),
                          cc.size() * sizeof(float)) &&
                read_file(dir + "/wk_b.bin", cw.data(),
                          cw.size() * sizeof(float)) &&
                read_file(dir + "/wv_b.bin", cv.data(),
                          cv.size() * sizeof(float)) &&
                read_file(dir + "/gate.bin", cg.data(),
                          cg.size() * sizeof(float));
            if (!loaded) {
                std::printf("REAL captured q/cache  FAIL  (binaries unreadable in "
                            "%s)\n",
                            dir.c_str());
                ++g_fail;
            } else {
                const int nT = (int) n_tokens;
                std::memcpy(hq.data(), cq.data(),
                            cq.size() * sizeof(float) < hq.size() * sizeof(float)
                                ? cq.size() * sizeof(float)
                                : hq.size() * sizeof(float));
                std::memcpy(hgate.data(), cg.data(),
                            cg.size() * sizeof(float) <
                                    hgate.size() * sizeof(float)
                                ? cg.size() * sizeof(float)
                                : hgate.size() * sizeof(float));
                std::memcpy(hcache.data(), cc.data(), cc.size() * sizeof(float));
                Args a;
                a.name = "REAL captured q/cache";
                a.pos0 = (int) tok_base;
                a.T = (int) tok_count;
                a.n_pos = (int) n_pos;
                a.chunk_tokens = (int) chunk_tokens;
                a.n_tokens = nT;
                a.cache_rows = (int) n_pos;
                a.stage = false;   // the capture is post-rope / post-append
                a.wk_src = cw.data();
                a.wv_src = cv.data();
                a.wk_floats = (std::size_t) wk_floats;
                a.wv_floats = (std::size_t) wv_floats;
                a.q_floats = (std::size_t) q_floats;
                a.gate_floats = (std::size_t) gate_floats;
                a.cache_floats = (std::size_t) cache_floats;
                const int rc = run_case(a);
                if (rc != 0) return rc;
            }
        }
    }

    std::printf("%s fail=%d\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}
