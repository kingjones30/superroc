// LKF: bit-identity of the decode fusion (K1 + K2 + Q2-KDA) against the unfused
// sequence. Every compared buffer is memcmp'd. 0 differing bytes is the gate.
//
// The Q8 oracle below is k_q8_1_quant from ling3_gemv_q8.hip, copied so this
// binary does not have to link the GEMV. Same butterfly, same d, same roundf.

#include "ling3_kda.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

using namespace ling3;

static int g_fail = 0;

#define CHECK(expr)                                                                  \
    do {                                                                             \
        hipError_t _e = (expr);                                                      \
        if (_e != hipSuccess) {                                                      \
            std::fprintf(stderr, "HIP %s: %s\n", #expr, hipGetErrorString(_e));      \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

static __global__ void oracle_q8_1_quant(const float * x, int8_t * qs, float * ds, int nb) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int b = (int) blockIdx.x * 8 + warp;
    if (b >= nb) {
        return;
    }
    const float * xb = x + (size_t) b * 32;
    const float v = xb[lane];
    float ma = fabsf(v), mv = v;
    int   mi = lane;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float oa = __shfl_down(ma, off);
        const float ov = __shfl_down(mv, off);
        const int   oi = __shfl_down(mi, off);
        if (oa > ma || (oa == ma && oi < mi)) {
            ma = oa;
            mv = ov;
            mi = oi;
        }
    }
    const float d  = mv / 127.f;
    const float id = d != 0.f ? 1.0f / d : 0.f;
    if (lane == 0) {
        ds[b] = d;
    }
    const float id_b = __shfl(id, 0);
    int8_t * qb = qs + (size_t) b * 32;
    qb[lane] = (int8_t) roundf(xb[lane] * id_b);
}

static void fill_u(float * h, size_t n, uint32_t seed, float lo, float hi) {
    uint32_t s = seed ? seed : 1u;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        const float u = (float) ((s >> 8) & 0xffff) / 65535.0f;
        h[i] = lo + (hi - lo) * u;
    }
}

static int diff_bytes(const void * a, const void * b, size_t n, const char * name) {
    const auto * x = static_cast<const unsigned char *>(a);
    const auto * y = static_cast<const unsigned char *>(b);
    int bad = 0;
    size_t first = 0;
    for (size_t i = 0; i < n; ++i) {
        if (x[i] != y[i]) {
            if (bad == 0) first = i;
            bad++;
        }
    }
    if (bad) {
        std::fprintf(stderr, "FAIL %s differing_bytes=%d / %zu first=%zu\n", name, bad, n, first);
        if (n % 4 == 0) {
            const auto * fx = static_cast<const float *>(a);
            const auto * fy = static_cast<const float *>(b);
            const size_t nf = n / 4;
            int shown = 0;
            int nfloat = 0;
            for (size_t i = 0; i < nf; ++i) {
                if (std::memcmp(fx + i, fy + i, 4) != 0) {
                    nfloat++;
                    if (shown < 4) {
                        unsigned ix = 0, iy = 0;
                        std::memcpy(&ix, fx + i, 4);
                        std::memcpy(&iy, fy + i, 4);
                        std::fprintf(stderr, "  float[%zu] ref=%a fused=%a hex %08x vs %08x\n", i, fx[i],
                                     fy[i], ix, iy);
                        shown++;
                    }
                }
            }
            std::fprintf(stderr, "  differing_floats=%d / %zu\n", nfloat, nf);
        }
        g_fail++;
    } else {
        std::fprintf(stderr, "ok %s 0 differing bytes (%zu)\n", name, n);
    }
    return bad;
}

struct Dev {
    float * q, * k, * v, * f, * g2, * beta;
    float * wq, * wk, * wv, * aexp, * dt, * onorm;
    float * conv, * ssm, * conv0, * ssm0;
    float * scratch, * out;
    int8_t * qs;
    float * ds;
    int d_inner, n_head, d_conv;
    size_t scratch_bytes, conv_n, ssm_n;
};

static void * must(size_t bytes) {
    void * p = nullptr;
    CHECK(hipMalloc(&p, bytes));
    CHECK(hipMemset(p, 0, bytes));
    return p;
}

static int run_case(Dev & d, const char * name, int pick, const char * variant,
                    bool expect_k2, uint32_t seed, bool zero_state, bool zero_act) {
    const int D = d.d_inner;
    const int H = d.n_head;
    const int CW = d.d_conv * D;
    std::vector<float> hq(D), hk(D), hv(D), hf(D), hg(D), hb(H);
    std::vector<float> hwq(CW), hwk(CW), hwv(CW), ha(H), hdt(D), ho(128);
    std::vector<float> hconv(d.conv_n), hssm(d.ssm_n);
    if (!zero_act) {
        fill_u(hq.data(), hq.size(), seed + 1, -1.0f, 1.0f);
        fill_u(hk.data(), hk.size(), seed + 2, -1.0f, 1.0f);
        fill_u(hv.data(), hv.size(), seed + 3, -1.0f, 1.0f);
        fill_u(hf.data(), hf.size(), seed + 4, -1.0f, 1.0f);
        fill_u(hg.data(), hg.size(), seed + 5, -1.0f, 1.0f);
        fill_u(hb.data(), hb.size(), seed + 6, -1.0f, 1.0f);
    }
    fill_u(hwq.data(), hwq.size(), seed + 7, -0.5f, 0.5f);
    fill_u(hwk.data(), hwk.size(), seed + 8, -0.5f, 0.5f);
    fill_u(hwv.data(), hwv.size(), seed + 9, -0.5f, 0.5f);
    fill_u(ha.data(), ha.size(), seed + 10, 0.2f, 1.5f);
    fill_u(hdt.data(), hdt.size(), seed + 11, -0.5f, 0.5f);
    fill_u(ho.data(), ho.size(), seed + 12, 0.5f, 1.5f);
    if (!zero_state) {
        fill_u(hconv.data(), hconv.size(), seed + 13, -0.2f, 0.2f);
        fill_u(hssm.data(), hssm.size(), seed + 14, -0.05f, 0.05f);
    }

    auto up = [](void * dst, const void * src, size_t n) {
        CHECK(hipMemcpy(dst, src, n, hipMemcpyHostToDevice));
    };
    up(d.q, hq.data(), hq.size() * 4);
    up(d.k, hk.data(), hk.size() * 4);
    up(d.v, hv.data(), hv.size() * 4);
    up(d.f, hf.data(), hf.size() * 4);
    up(d.g2, hg.data(), hg.size() * 4);
    up(d.beta, hb.data(), hb.size() * 4);
    up(d.wq, hwq.data(), hwq.size() * 4);
    up(d.wk, hwk.data(), hwk.size() * 4);
    up(d.wv, hwv.data(), hwv.size() * 4);
    up(d.aexp, ha.data(), ha.size() * 4);
    up(d.dt, hdt.data(), hdt.size() * 4);
    up(d.onorm, ho.data(), ho.size() * 4);
    up(d.conv0, hconv.data(), hconv.size() * 4);
    up(d.ssm0, hssm.data(), hssm.size() * 4);

    KdaShape sh;
    sh.n_head = H;
    sh.head_dim = 128;
    sh.d_conv = d.d_conv;
    KdaScalars sc;
    KdaWeights w;
    w.conv_wq = d.wq;
    w.conv_wk = d.wk;
    w.conv_wv = d.wv;
    w.a_exp = d.aexp;
    w.dt_bias = d.dt;
    w.o_norm = d.onorm;
    KdaInputs in;
    in.q = make_view(d.q, D, 1);
    in.k = make_view(d.k, D, 1);
    in.v = make_view(d.v, D, 1);
    in.f = make_view(d.f, D, 1);
    in.g2 = make_view(d.g2, D, 1);
    in.beta = make_view(d.beta, H, 1);
    in.n_tokens = 1;
    in.n_seqs = 1;
    KdaState st;
    st.conv_state = d.conv;
    st.ssm_state = d.ssm;
    st.n_seqs = 1;
    View out = make_view(d.out, D, 1);

    const size_t scratch_f = d.scratch_bytes / sizeof(float);
    std::vector<float> scratch_ref(scratch_f), scratch_new(scratch_f);
    std::vector<float> out_ref(D), out_new(D);
    std::vector<float> conv_ref(d.conv_n), conv_new(d.conv_n);
    std::vector<float> ssm_ref(d.ssm_n), ssm_new(d.ssm_n);
    std::vector<int8_t> qs_ref(D), qs_new(D);
    std::vector<float> ds_ref(D / 32), ds_new(D / 32);

    auto one = [&](bool fused, bool with_q8, std::vector<float> & scratch_h, std::vector<float> & out_h,
                   std::vector<float> & conv_h, std::vector<float> & ssm_h, std::vector<int8_t> & qs_h,
                   std::vector<float> & ds_h, int & dk1, int & dk2) {
        kda_decode_test_pick(pick, variant);
        kda_decode_fused_test_pick(fused ? 2 : 1);
        CHECK(hipMemcpy(d.conv, d.conv0, d.conv_n * sizeof(float), hipMemcpyDeviceToDevice));
        CHECK(hipMemcpy(d.ssm, d.ssm0, d.ssm_n * sizeof(float), hipMemcpyDeviceToDevice));
        CHECK(hipMemset(d.scratch, 0, d.scratch_bytes));
        CHECK(hipMemset(d.out, 0, D * sizeof(float)));
        CHECK(hipMemset(d.qs, 0x7f, D));
        CHECK(hipMemset(d.ds, 0, (D / 32) * sizeof(float)));
        const int k1b = kda_decode_fused_k1_launches();
        const int k2b = kda_decode_fused_k2_launches();
        kda_forward_scratch(sh, sc, w, in, st, out, d.scratch, d.scratch_bytes, nullptr,
                            (fused && with_q8) ? d.qs : nullptr,
                            (fused && with_q8) ? d.ds : nullptr);
        CHECK(hipDeviceSynchronize());
        dk1 = kda_decode_fused_k1_launches() - k1b;
        dk2 = kda_decode_fused_k2_launches() - k2b;
        CHECK(hipMemcpy(scratch_h.data(), d.scratch, d.scratch_bytes, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(out_h.data(), d.out, D * sizeof(float), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(conv_h.data(), d.conv, d.conv_n * sizeof(float), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(ssm_h.data(), d.ssm, d.ssm_n * sizeof(float), hipMemcpyDeviceToHost));
        if (!fused && with_q8) {
            const int nb = D / 32;
            oracle_q8_1_quant<<<(nb + 7) / 8, 256>>>(d.out, d.qs, d.ds, nb);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
        }
        if (with_q8) {
            CHECK(hipMemcpy(qs_h.data(), d.qs, D, hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(ds_h.data(), d.ds, (D / 32) * sizeof(float), hipMemcpyDeviceToHost));
        }
    };

    int dk1_off = 0, dk2_off = 0, dk1_on = 0, dk2_on = 0;
    one(false, expect_k2, scratch_ref, out_ref, conv_ref, ssm_ref, qs_ref, ds_ref, dk1_off, dk2_off);
    one(true, expect_k2, scratch_new, out_new, conv_new, ssm_new, qs_new, ds_new, dk1_on, dk2_on);

    std::fprintf(stderr, "== %s k1_off=%d k2_off=%d k1_on=%d k2_on=%d\n", name, dk1_off, dk2_off,
                 dk1_on, dk2_on);
    if (dk1_off != 0 || dk2_off != 0 || dk1_on != 1 || dk2_on != (expect_k2 ? 1 : 0)) {
        std::fprintf(stderr, "FAIL %s fused-launch counts\n", name);
        g_fail++;
    } else {
        std::fprintf(stderr, "ok %s fused path launch counts\n", name);
    }
    diff_bytes(scratch_ref.data(), scratch_new.data(), d.scratch_bytes, (std::string(name) + " scratch").c_str());
    diff_bytes(out_ref.data(), out_new.data(), D * sizeof(float), (std::string(name) + " out").c_str());
    diff_bytes(conv_ref.data(), conv_new.data(), d.conv_n * sizeof(float), (std::string(name) + " conv_state").c_str());
    diff_bytes(ssm_ref.data(), ssm_new.data(), d.ssm_n * sizeof(float), (std::string(name) + " ssm_state").c_str());
    if (expect_k2) {
        diff_bytes(qs_ref.data(), qs_new.data(), (size_t) D, (std::string(name) + " q8.qs").c_str());
        diff_bytes(ds_ref.data(), ds_new.data(), (D / 32) * sizeof(float), (std::string(name) + " q8.ds").c_str());
    }
    return g_fail;
}

// Capture record: see kda_capture_decode. g2 and o_norm are not in the file; they
// are a fixed vector so both paths see the same epilogue inputs. EVERY record is
// replayed, not just the first: record 0 is call 0 (the first decode step, zero ssm
// state) while the later records carry the real accumulated state -- which is where a
// fused scan can diverge (the rejected tiled K2 broke on a random state, not a zero one).
static int run_capture(const char * path) {
    FILE * fp = std::fopen(path, "rb");
    if (fp == nullptr) {
        std::fprintf(stderr, "capture: %s not readable, skipped\n", path);
        return 0;
    }
    uint32_t magic = 0;
    int32_t version = 0, n_head = 0, head_dim = 0, n_seqs = 0, d_inner = 0, nrec = 0, stride = 0;
    if (std::fread(&magic, 4, 1, fp) != 1 || magic != 0x3141444Bu ||
        std::fread(&version, 4, 1, fp) != 1 || std::fread(&n_head, 4, 1, fp) != 1 ||
        std::fread(&head_dim, 4, 1, fp) != 1 || std::fread(&n_seqs, 4, 1, fp) != 1 ||
        std::fread(&d_inner, 4, 1, fp) != 1 || std::fread(&nrec, 4, 1, fp) != 1 ||
        std::fread(&stride, 4, 1, fp) != 1) {
        std::fprintf(stderr, "FAIL capture header\n");
        std::fclose(fp);
        g_fail++;
        return 1;
    }
    if (nrec < 1 || head_dim != 128 || n_head < 1 || d_inner != n_head * 128) {
        std::fprintf(stderr, "capture: no usable record (nrec=%d head_dim=%d), skipped\n", nrec, head_dim);
        std::fclose(fp);
        return 0;
    }
    const size_t n_state = (size_t) n_head * 128u * 128u;
    const size_t rec = sizeof(int) + sizeof(float) * ((size_t) d_inner * 4u + (size_t) n_head + n_state);
    std::vector<char> raw(rec);

    std::vector<float> hg((size_t) d_inner), ho(128);
    fill_u(hg.data(), hg.size(), 99, -1.0f, 1.0f);
    fill_u(ho.data(), ho.size(), 100, 0.5f, 1.5f);

    float * q = (float *) must((size_t) d_inner * 4);
    float * k = (float *) must((size_t) d_inner * 4);
    float * v = (float *) must((size_t) d_inner * 4);
    float * g1 = (float *) must((size_t) d_inner * 4);
    float * beta = (float *) must((size_t) n_head * 4);
    float * g2 = (float *) must((size_t) d_inner * 4);
    float * onorm = (float *) must(128 * 4);
    float * ssm = (float *) must(n_state * 4);
    float * ssm0 = (float *) must(n_state * 4);
    float * out = (float *) must((size_t) d_inner * 4);
    int8_t * qs = (int8_t *) must((size_t) d_inner + 64);
    float * ds = (float *) must((size_t) (d_inner / 32) * 4 + 64);
    CHECK(hipMemcpy(g2, hg.data(), (size_t) d_inner * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(onorm, ho.data(), 128 * 4, hipMemcpyHostToDevice));

    KdaShape sh;
    sh.n_head = n_head;
    sh.head_dim = 128;
    sh.d_conv = 4;
    KdaScalars sc;
    auto launch = [&](bool fused) {
        kda_decode_test_pick(2, "t32");
        kda_decode_fused_test_pick(fused ? 2 : 1);
        CHECK(hipMemcpy(ssm, ssm0, n_state * 4, hipMemcpyDeviceToDevice));
        CHECK(hipMemset(out, 0, (size_t) d_inner * 4));
        CHECK(hipMemset(qs, 0x7f, (size_t) d_inner));
        const int k2b = kda_decode_fused_k2_launches();
        const int ran = kda_decode_scan_out(
            sh, sc, onorm, make_view(q, d_inner, 1), make_view(k, d_inner, 1),
            make_view(v, d_inner, 1), make_view(g1, d_inner, 1), make_view(beta, n_head, 1),
            make_view(g2, d_inner, 1), make_view(out, d_inner, 1), ssm,
            fused ? qs : nullptr, fused ? ds : nullptr, nullptr);
        CHECK(hipDeviceSynchronize());
        const int dk2 = kda_decode_fused_k2_launches() - k2b;
        if (fused) {
            if (ran != 1 || dk2 != 1) {
                std::fprintf(stderr, "FAIL capture fused did not run ran=%d dk2=%d\n", ran, dk2);
                g_fail++;
            }
        } else if (ran != 0 || dk2 != 0) {
            std::fprintf(stderr, "FAIL capture unfused ran K2 ran=%d dk2=%d\n", ran, dk2);
            g_fail++;
        }
        if (!fused) {
            const int nb = d_inner / 32;
            oracle_q8_1_quant<<<(nb + 7) / 8, 256>>>(out, qs, ds, nb);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
        }
    };

    std::vector<float> out_ref((size_t) d_inner), out_new((size_t) d_inner);
    std::vector<float> ssm_ref(n_state), ssm_new(n_state);
    std::vector<int8_t> qs_ref((size_t) d_inner), qs_new((size_t) d_inner);
    std::vector<float> ds_ref((size_t) d_inner / 32), ds_new((size_t) d_inner / 32);
    std::fprintf(stderr, "== capture %s n_head=%d nrec=%d stride=%d\n", path, n_head, nrec, stride);
    for (int r = 0; r < nrec; ++r) {
        if (std::fread(raw.data(), 1, rec, fp) != rec) {
            std::fprintf(stderr, "FAIL capture record %d short\n", r);
            std::fclose(fp);
            g_fail++;
            return 1;
        }
        const float * hp = reinterpret_cast<const float *>(raw.data() + sizeof(int));
        const float * cq = hp;
        const float * ck = cq + d_inner;
        const float * cv = ck + d_inner;
        const float * cg = cv + d_inner;
        const float * cb = cg + d_inner;
        const float * cs = cb + n_head;
        CHECK(hipMemcpy(q, cq, (size_t) d_inner * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(k, ck, (size_t) d_inner * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(v, cv, (size_t) d_inner * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(g1, cg, (size_t) d_inner * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(beta, cb, (size_t) n_head * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(ssm0, cs, n_state * 4, hipMemcpyHostToDevice));

        launch(false);
        CHECK(hipMemcpy(out_ref.data(), out, (size_t) d_inner * 4, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(ssm_ref.data(), ssm, n_state * 4, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(qs_ref.data(), qs, (size_t) d_inner, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(ds_ref.data(), ds, (d_inner / 32) * 4, hipMemcpyDeviceToHost));
        launch(true);
        CHECK(hipMemcpy(out_new.data(), out, (size_t) d_inner * 4, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(ssm_new.data(), ssm, n_state * 4, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(qs_new.data(), qs, (size_t) d_inner, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(ds_new.data(), ds, (d_inner / 32) * 4, hipMemcpyDeviceToHost));

        char nm[72];
        std::snprintf(nm, sizeof(nm), "capture[%d] out", r);
        diff_bytes(out_ref.data(), out_new.data(), (size_t) d_inner * 4, nm);
        std::snprintf(nm, sizeof(nm), "capture[%d] ssm", r);
        diff_bytes(ssm_ref.data(), ssm_new.data(), n_state * 4, nm);
        std::snprintf(nm, sizeof(nm), "capture[%d] q8.qs", r);
        diff_bytes(qs_ref.data(), qs_new.data(), (size_t) d_inner, nm);
        std::snprintf(nm, sizeof(nm), "capture[%d] q8.ds", r);
        diff_bytes(ds_ref.data(), ds_new.data(), ((size_t) d_inner / 32) * sizeof(float), nm);
    }
    std::fclose(fp);
    return g_fail;
}

// ---------------------------------------------------------------------------
// Bisect helpers. The fused K2 body is a copy of scan_kernel's statements, while
// the unfused reference runs scan_kernel_decode_t32. These two checks separate
// "the two shipped scans already disagree" from "the out_norm/Q8 epilogue differs".
// ---------------------------------------------------------------------------

// Do the two SHIPPED scan kernels agree bit-for-bit on the same inputs?
// t32 (32 threads, shared tile) vs scan_kernel (one thread per column).
static int run_scan_cmp(Dev & d, uint32_t seed, bool zero_state, bool zero_act) {
    const int D = d.d_inner;
    const int H = d.n_head;
    std::vector<float> hq(D), hk(D), hv(D), hg(D), hb(H);
    std::vector<float> hssm(d.ssm_n);
    if (!zero_act) {
        fill_u(hq.data(), hq.size(), seed + 1, -1.0f, 1.0f);
        fill_u(hk.data(), hk.size(), seed + 2, -1.0f, 1.0f);
        fill_u(hv.data(), hv.size(), seed + 3, -1.0f, 1.0f);
        fill_u(hg.data(), hg.size(), seed + 5, -1.0f, 1.0f);
        fill_u(hb.data(), hb.size(), seed + 6, -1.0f, 1.0f);
    }
    if (!zero_state) {
        fill_u(hssm.data(), hssm.size(), seed + 14, -0.05f, 0.05f);
    }
    auto up = [](void * dst, const void * src, size_t n) {
        CHECK(hipMemcpy(dst, src, n, hipMemcpyHostToDevice));
    };
    float * g1 = (float *) must((size_t) D * 4);
    up(d.q, hq.data(), hq.size() * 4);
    up(d.k, hk.data(), hk.size() * 4);
    up(d.v, hv.data(), hv.size() * 4);
    up(g1, hg.data(), hg.size() * 4);
    up(d.beta, hb.data(), hb.size() * 4);
    up(d.ssm0, hssm.data(), hssm.size() * 4);

    KdaShape sh;
    sh.n_head = H;
    sh.head_dim = 128;
    sh.d_conv = d.d_conv;
    KdaScalars sc;

    std::vector<float> out_a(D), out_b(D), ssm_a(d.ssm_n), ssm_b(d.ssm_n);
    auto one = [&](int pick, const char * variant, std::vector<float> & oh, std::vector<float> & shh) {
        kda_decode_test_pick(pick, variant);
        kda_decode_fused_test_pick(1);
        CHECK(hipMemcpy(d.ssm, d.ssm0, d.ssm_n * sizeof(float), hipMemcpyDeviceToDevice));
        CHECK(hipMemset(d.out, 0, D * sizeof(float)));
        kda_scan(sh, sc, make_view(d.q, D, 1), make_view(d.k, D, 1), make_view(d.v, D, 1),
                 make_view(g1, D, 1), make_view(d.beta, H, 1), make_view(d.out, D, 1), d.ssm,
                 /*n_tokens*/ 1, /*n_seqs*/ 1, nullptr);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(oh.data(), d.out, D * sizeof(float), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(shh.data(), d.ssm, d.ssm_n * sizeof(float), hipMemcpyDeviceToHost));
    };
    one(2, "t32", out_a, ssm_a);
    one(1, nullptr, out_b, ssm_b);
    std::fprintf(stderr, "== scancmp seed=%u zero_state=%d\n", seed, zero_state ? 1 : 0);
    diff_bytes(out_a.data(), out_b.data(), D * sizeof(float), "scancmp out (t32 vs ref)");
    diff_bytes(ssm_a.data(), ssm_b.data(), d.ssm_n * sizeof(float), "scancmp ssm (t32 vs ref)");
    return g_fail;
}

// out_norm with o_norm == 1.0 and g2 == 0.0 gives y = nscale * x * 0.5. Both the
// 1.0 and the 0.5 multiplies are exact, so ANY grouping of the per-element
// expression yields the same bits. A difference here can only come from the scan
// output x or from the out_norm reduction (nscale) -- not from `* w * sigmoid`.
static int run_iso(Dev & d, uint32_t seed, bool zero_state, bool zero_act) {
    const int D = d.d_inner;
    const int H = d.n_head;
    std::vector<float> hq(D), hk(D), hv(D), hf(D), hb(H);
    std::vector<float> hwq(d.d_conv * D), hwk(d.d_conv * D), hwv(d.d_conv * D), ha(H), hdt(D);
    std::vector<float> ho(128), hzero(D);
    std::vector<float> hconv(d.conv_n), hssm(d.ssm_n);
    std::fill(ho.begin(), ho.end(), 1.0f);      // o_norm = 1  -> exact
    std::fill(hzero.begin(), hzero.end(), 0.0f); // g2 = 0      -> sigmoid(0) = 0.5 exactly
    if (!zero_act) {
        fill_u(hq.data(), hq.size(), seed + 1, -1.0f, 1.0f);
        fill_u(hk.data(), hk.size(), seed + 2, -1.0f, 1.0f);
        fill_u(hv.data(), hv.size(), seed + 3, -1.0f, 1.0f);
        fill_u(hf.data(), hf.size(), seed + 4, -1.0f, 1.0f);
        fill_u(hb.data(), hb.size(), seed + 6, -1.0f, 1.0f);
    }
    fill_u(hwq.data(), hwq.size(), seed + 7, -0.5f, 0.5f);
    fill_u(hwk.data(), hwk.size(), seed + 8, -0.5f, 0.5f);
    fill_u(hwv.data(), hwv.size(), seed + 9, -0.5f, 0.5f);
    fill_u(ha.data(), ha.size(), seed + 10, 0.2f, 1.5f);
    fill_u(hdt.data(), hdt.size(), seed + 11, -0.5f, 0.5f);
    if (!zero_state) {
        fill_u(hconv.data(), hconv.size(), seed + 13, -0.2f, 0.2f);
        fill_u(hssm.data(), hssm.size(), seed + 14, -0.05f, 0.05f);
    }
    auto up = [](void * dst, const void * src, size_t n) {
        CHECK(hipMemcpy(dst, src, n, hipMemcpyHostToDevice));
    };
    up(d.q, hq.data(), hq.size() * 4);
    up(d.k, hk.data(), hk.size() * 4);
    up(d.v, hv.data(), hv.size() * 4);
    up(d.f, hf.data(), hf.size() * 4);
    up(d.g2, hzero.data(), hzero.size() * 4);
    up(d.beta, hb.data(), hb.size() * 4);
    up(d.wq, hwq.data(), hwq.size() * 4);
    up(d.wk, hwk.data(), hwk.size() * 4);
    up(d.wv, hwv.data(), hwv.size() * 4);
    up(d.aexp, ha.data(), ha.size() * 4);
    up(d.dt, hdt.data(), hdt.size() * 4);
    up(d.onorm, ho.data(), ho.size() * 4);
    up(d.conv0, hconv.data(), hconv.size() * 4);
    up(d.ssm0, hssm.data(), hssm.size() * 4);

    KdaShape sh;
    sh.n_head = H;
    sh.head_dim = 128;
    sh.d_conv = d.d_conv;
    KdaScalars sc;
    KdaWeights w;
    w.conv_wq = d.wq;
    w.conv_wk = d.wk;
    w.conv_wv = d.wv;
    w.a_exp = d.aexp;
    w.dt_bias = d.dt;
    w.o_norm = d.onorm;
    KdaInputs in;
    in.q = make_view(d.q, D, 1);
    in.k = make_view(d.k, D, 1);
    in.v = make_view(d.v, D, 1);
    in.f = make_view(d.f, D, 1);
    in.g2 = make_view(d.g2, D, 1);
    in.beta = make_view(d.beta, H, 1);
    in.n_tokens = 1;
    in.n_seqs = 1;
    KdaState st;
    st.conv_state = d.conv;
    st.ssm_state = d.ssm;
    st.n_seqs = 1;
    View out = make_view(d.out, D, 1);

    std::vector<float> out_ref(D), out_new(D), ssm_ref(d.ssm_n), ssm_new(d.ssm_n);
    auto one = [&](bool fused, std::vector<float> & oh, std::vector<float> & shh) {
        kda_decode_test_pick(2, "t32");
        kda_decode_fused_test_pick(fused ? 2 : 1);
        CHECK(hipMemcpy(d.conv, d.conv0, d.conv_n * sizeof(float), hipMemcpyDeviceToDevice));
        CHECK(hipMemcpy(d.ssm, d.ssm0, d.ssm_n * sizeof(float), hipMemcpyDeviceToDevice));
        CHECK(hipMemset(d.scratch, 0, d.scratch_bytes));
        CHECK(hipMemset(d.out, 0, D * sizeof(float)));
        kda_forward_scratch(sh, sc, w, in, st, out, d.scratch, d.scratch_bytes, nullptr, nullptr, nullptr);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(oh.data(), d.out, D * sizeof(float), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(shh.data(), d.ssm, d.ssm_n * sizeof(float), hipMemcpyDeviceToHost));
    };
    one(false, out_ref, ssm_ref);
    one(true, out_new, ssm_new);
    std::fprintf(stderr, "== iso seed=%u zero_state=%d (o_norm=1, g2=0 -> y = nscale*x*0.5)\n", seed,
                 zero_state ? 1 : 0);
    diff_bytes(ssm_ref.data(), ssm_new.data(), d.ssm_n * sizeof(float), "iso ssm");
    diff_bytes(out_ref.data(), out_new.data(), D * sizeof(float), "iso out");
    return g_fail;
}

// ---------------------------------------------------------------------------
// Which grouping of (nscale*x) * w * sigmoid(g) does out_norm_gate_kernel emit?
//
// With g2 == 0 the gate is sigmoid(0) = 0.5 exactly, so the whole probe is f32
// multiplies and needs no expf on the host. Two runs share one scan output:
//   A: w = 1.0  ->  out_A = P * 0.5        (every grouping agrees, P = nscale*x)
//   B: w = 1.5  ->  out_B = the grouping we want to identify
//   C1 = ((2*out_A) * 1.5) * 0.5   is  ((P*w)*sig)     source order
//   C2 = (2*out_A) * (1.5 * 0.5)   is  (P*(w*sig))
//   C3 = (out_A) * 1.5             is  ((P*sig)*w)     (P*0.5 == out_A exactly)
// Whichever candidate is memcmp-identical to out_B is the order to hard-code.
// ---------------------------------------------------------------------------
static int run_group_probe(Dev & d, uint32_t seed) {
    const int D = d.d_inner;
    const int H = d.n_head;
    std::vector<float> hq(D), hk(D), hv(D), hg(D), hb(H), hssm(d.ssm_n);
    fill_u(hq.data(), hq.size(), seed + 1, -1.0f, 1.0f);
    fill_u(hk.data(), hk.size(), seed + 2, -1.0f, 1.0f);
    fill_u(hv.data(), hv.size(), seed + 3, -1.0f, 1.0f);
    fill_u(hg.data(), hg.size(), seed + 5, -1.0f, 1.0f);
    fill_u(hb.data(), hb.size(), seed + 6, -1.0f, 1.0f);
    fill_u(hssm.data(), hssm.size(), seed + 14, -0.05f, 0.05f);
    auto up = [](void * dst, const void * src, size_t n) {
        CHECK(hipMemcpy(dst, src, n, hipMemcpyHostToDevice));
    };
    float * g1 = (float *) must((size_t) D * 4);
    float * g2z = (float *) must((size_t) D * 4); // all zero -> sigmoid(0) = 0.5
    up(d.q, hq.data(), hq.size() * 4);
    up(d.k, hk.data(), hk.size() * 4);
    up(d.v, hv.data(), hv.size() * 4);
    up(g1, hg.data(), hg.size() * 4);
    up(d.beta, hb.data(), hb.size() * 4);
    up(d.ssm0, hssm.data(), hssm.size() * 4);

    KdaShape sh;
    sh.n_head = H;
    sh.head_dim = 128;
    sh.d_conv = d.d_conv;
    KdaScalars sc;
    KdaWeights w;
    w.o_norm = d.onorm;

    std::vector<float> ho1(128), ho2(128), out_a(D), out_b(D);
    std::fill(ho1.begin(), ho1.end(), 1.0f);
    std::fill(ho2.begin(), ho2.end(), 1.5f);

    auto one = [&](const std::vector<float> & on, std::vector<float> & oh) {
        kda_decode_test_pick(2, "t32");
        kda_decode_fused_test_pick(1);
        up(d.onorm, on.data(), on.size() * 4);
        CHECK(hipMemcpy(d.ssm, d.ssm0, d.ssm_n * sizeof(float), hipMemcpyDeviceToDevice));
        CHECK(hipMemset(d.out, 0, D * sizeof(float)));
        kda_scan(sh, sc, make_view(d.q, D, 1), make_view(d.k, D, 1), make_view(d.v, D, 1),
                 make_view(g1, D, 1), make_view(d.beta, H, 1), make_view(d.out, D, 1), d.ssm,
                 1, 1, nullptr);
        kda_out_norm_gate(sh, sc, w, ViewC(make_view(d.out, D, 1)), ViewC(make_view(g2z, D, 1)),
                          make_view(d.out, D, 1), 1, 1, nullptr);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(oh.data(), d.out, D * sizeof(float), hipMemcpyDeviceToHost));
    };
    one(ho1, out_a);
    one(ho2, out_b);

    std::vector<float> c1(D), c2(D), c3(D);
    for (int i = 0; i < D; ++i) {
        const float P = 2.0f * out_a[(size_t) i];
        c1[(size_t) i] = (P * 1.5f) * 0.5f;
        c2[(size_t) i] = P * (1.5f * 0.5f);
        c3[(size_t) i] = out_a[(size_t) i] * 1.5f;
    }
    std::fprintf(stderr, "== group-probe seed=%u (w=1.5, sig=0.5). Which grouping is out_norm_gate?\n",
                 seed);
    const int b1 = diff_bytes(c1.data(), out_b.data(), D * sizeof(float), "group ((P*w)*sig) source-order");
    const int b2 = diff_bytes(c2.data(), out_b.data(), D * sizeof(float), "group (P*(w*sig))");
    const int b3 = diff_bytes(c3.data(), out_b.data(), D * sizeof(float), "group ((P*sig)*w)");
    // The probe is diagnostic: only the real fused-vs-unfused memcmp is the gate,
    // so do not count a probe mismatch as a failure.
    g_fail -= (b1 > 0) + (b2 > 0) + (b3 > 0);
    std::fprintf(stderr, "group-probe verdict: source=%s w-first=%s sig-first=%s\n",
                 b1 == 0 ? "MATCH" : "no", b2 == 0 ? "MATCH" : "no", b3 == 0 ? "MATCH" : "no");
    return g_fail;
}

int main(int argc, char ** argv) {
    std::string cap_own;
    if (const char * env = std::getenv("LING3_KDA_CAPTURE")) {
        if (env[0] != '\0') cap_own = env;
    }
    unsetenv("LING3_KDA_CAPTURE");
    const char * cap = argc > 1 ? argv[1] : (cap_own.empty() ? nullptr : cap_own.c_str());

    Dev d{};
    d.d_inner = 4096;
    d.n_head = 32;
    d.d_conv = 4;
    KdaShape sh;
    sh.n_head = 32;
    sh.head_dim = 128;
    sh.d_conv = 4;
    d.scratch_bytes = kda_forward_scratch_bytes(sh, 1, 1);
    d.conv_n = (size_t) sh.embd_r();
    d.ssm_n = (size_t) sh.embd_s();
    const int D = d.d_inner;
    const int CW = d.d_conv * D;
    d.q = (float *) must(D * 4);
    d.k = (float *) must(D * 4);
    d.v = (float *) must(D * 4);
    d.f = (float *) must(D * 4);
    d.g2 = (float *) must(D * 4);
    d.beta = (float *) must(d.n_head * 4);
    d.wq = (float *) must(CW * 4);
    d.wk = (float *) must(CW * 4);
    d.wv = (float *) must(CW * 4);
    d.aexp = (float *) must(d.n_head * 4);
    d.dt = (float *) must(D * 4);
    d.onorm = (float *) must(128 * 4);
    d.conv = (float *) must(d.conv_n * 4);
    d.ssm = (float *) must(d.ssm_n * 4);
    d.conv0 = (float *) must(d.conv_n * 4);
    d.ssm0 = (float *) must(d.ssm_n * 4);
    d.scratch = (float *) must(d.scratch_bytes);
    d.out = (float *) must(D * 4);
    d.qs = (int8_t *) must((size_t) D + 64);
    d.ds = (float *) must((D / 32) * sizeof(float) + 64);

    // Bisect first: these decide where a mismatch lives before the full cases run.
    run_scan_cmp(d, 20, false, false);
    run_scan_cmp(d, 1, true, true);
    run_iso(d, 20, false, false);
    run_iso(d, 40, false, false);
    run_group_probe(d, 20);

    run_case(d, "zero-state", 2, "t32", true, 1, true, true);
    run_case(d, "rand-a", 2, "t32", true, 20, false, false);
    run_case(d, "rand-b", 2, "t32", true, 40, false, false);
    run_case(d, "red", 2, "red", false, 60, false, false);
    run_case(d, "ref", 1, nullptr, false, 80, false, false);
    if (cap != nullptr && cap[0] != '\0') run_capture(cap);
    else std::fprintf(stderr, "capture: LING3_KDA_CAPTURE unset, skipped\n");

    if (g_fail) {
        std::fprintf(stderr, "LKF FUSION FAIL %d\n", g_fail);
        return 1;
    }
    std::fprintf(stderr, "LKF FUSION PASS\n");
    return 0;
}
