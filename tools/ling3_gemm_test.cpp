// ling3_gemm_test.cpp — LS3's unit gate for kernels/ling3_gemm.hip.
// Compares the tiled grouped GEMM and the batched router against the CPU
// references (ling3_gemm_cpu / ling3_route_core) on SMALL shapes, so the first
// full-model run never carries an untested kernel (the brief's rule after H1's
// GPU reset). Run inside gpu-run: ./superroc_ling3_gemm_test
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ling3_gemm.hpp"
#include "ling3_moe.hpp"   // header-only CPU references via ling3_cpu.hpp

namespace {

int g_fail = 0;

unsigned g_state = 20261001u;
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
    HIPCK(hipMalloc(&p, bytes + LING3_TAIL_PAD));
    return p;
}

const char *TypeName(int qtype) {
    switch (qtype) {
        case LING3_F32: return "F32";
        case LING3_Q8_0: return "Q8_0";
        case LING3_Q6_K: return "Q6_K";
        case LING3_Q4_0_ROCMFP4: return "FP4";
        case LING3_Q4_0_ROCMFP4_FAST: return "FP4F";
        default: return "?";
    }
}

// Random weight bytes; for F32 interpret them as floats in [-1,1].
std::vector<uint8_t> MakeW(int qtype, size_t bytes) {
    std::vector<uint8_t> w(bytes);
    if (qtype == LING3_F32) {
        float *f = reinterpret_cast<float *>(w.data());
        for (size_t i = 0; i < bytes / 4; ++i) {
            f[i] = Frnd(-1.f, 1.f);
        }
    } else {
        for (size_t i = 0; i < bytes; ++i) {
            w[i] = (uint8_t) (Rnd() & 0xff);
        }
    }
    return w;
}

// ---------------------------------------------------------------------------
// dense GEMM: ling3_gemm_fast vs ling3_gemm_cpu
// ---------------------------------------------------------------------------
void TestDense(int qtype, int k, int n, int m) {
    const size_t rb = ling3_row_bytes(qtype, k);
    if (rb == 0) {
        std::printf("DENSE %-4s k=%d n=%d m=%d SKIP (bad k for type)\n", TypeName(qtype), k, n, m);
        return;
    }
    std::vector<uint8_t> w = MakeW(qtype, rb * (size_t) n);
    std::vector<float> x((size_t) m * k), yref((size_t) m * n), ydev((size_t) m * n, -777.f);
    for (auto &v : x) v = Frnd(-1.f, 1.f);
    int rc = ling3_gemm_cpu(qtype, w.data(), x.data(), yref.data(), k, n, m);
    if (rc != LING3_OK) {
        std::printf("DENSE %-4s k=%d n=%d m=%d FAIL cpu rc=%d\n", TypeName(qtype), k, n, m, rc);
        ++g_fail;
        return;
    }
    void *dw = DevAlloc(w.size());
    void *dx = DevAlloc(x.size() * sizeof(float));
    void *dy = DevAlloc(ydev.size() * sizeof(float));
    HIPCK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
    rc = ling3_gemm_fast(qtype, dw, (const float *) dx, (float *) dy, k, n, m);
    HIPCK(hipDeviceSynchronize());
    HIPCK(hipMemcpy(ydev.data(), dy, ydev.size() * sizeof(float), hipMemcpyDeviceToHost));
    if (rc != LING3_OK) {
        std::printf("DENSE %-4s k=%d n=%d m=%d FAIL dev rc=%d\n", TypeName(qtype), k, n, m, rc);
        ++g_fail;
    } else {
        double maxabs = 0, scale = 1e-30;
        for (size_t i = 0; i < yref.size(); ++i) {
            const double d = std::fabs((double) yref[i] - (double) ydev[i]);
            if (d > maxabs) maxabs = d;
            scale = std::fmax(scale, std::fabs((double) yref[i]));
        }
        const bool ok = maxabs <= 1e-4 * scale;
        std::printf("DENSE %-4s k=%d n=%d m=%d %s max_abs=%.3e (ref scale %.2f)\n",
                    TypeName(qtype), k, n, m, ok ? "PASS" : "FAIL", maxabs, scale);
        if (!ok) ++g_fail;
    }
    hipFree(dw); hipFree(dx); hipFree(dy);
}

// ---------------------------------------------------------------------------
// grouped GEMM: random (expert, token) slots vs a per-slot CPU reference
// ---------------------------------------------------------------------------
void TestGrouped(int qtype, int k, int n, int n_expert, int n_slots) {
    const size_t rb = ling3_row_bytes(qtype, k);
    if (rb == 0) {
        std::printf("GROUP %-4s SKIP\n", TypeName(qtype));
        return;
    }
    const int n_tokens = 13;
    const size_t e_stride = (size_t) n * rb;
    std::vector<uint8_t> w = MakeW(qtype, e_stride * n_expert);
    std::vector<float> x((size_t) n_tokens * k);
    for (auto &v : x) v = Frnd(-1.f, 1.f);

    // slot j: (expert, token, scale)
    std::vector<int> slot_e(n_slots), slot_tok(n_slots), slot_off(n_expert + 1, 0);
    std::vector<float> slot_w(n_slots);
    std::vector<int> counts(n_expert + 1, 0);
    for (int j = 0; j < n_slots; ++j) {
        slot_e[j] = (int) (Rnd() % (unsigned) n_expert);
        slot_tok[j] = (int) (Rnd() % (unsigned) n_tokens);
        slot_w[j] = Frnd(0.05f, 0.95f);
        counts[slot_e[j] + 1]++;
    }
    for (int e = 0; e < n_expert; ++e) counts[e + 1] += counts[e];
    std::vector<int> fill(counts.begin(), counts.end() - 1);
    std::vector<int> g_slot_tok(n_slots);
    std::vector<float> g_slot_w(n_slots);
    for (int j = 0; j < n_slots; ++j) {
        const int p = fill[slot_e[j]]++;
        g_slot_tok[p] = slot_tok[j];
        g_slot_w[p] = slot_w[j];
    }
    std::vector<int> grp_e, grp_off, grp_cnt;
    int max_cnt = 0;
    for (int e = 0; e < n_expert; ++e) {
        const int c = counts[e + 1] - counts[e];
        if (c > 0) {
            grp_e.push_back(e);
            grp_off.push_back(counts[e]);
            grp_cnt.push_back(c);
            max_cnt = std::max(max_cnt, c);
        }
    }
    // CPU reference: y[token] += scale * dot(W_e_row, x_token), slot order.
    std::vector<float> yref((size_t) n_tokens * n, 0.f), ydev((size_t) n_tokens * n, -777.f);
    std::vector<float> srow(n);
    for (int j = 0; j < n_slots; ++j) {
        const int rc = ling3_gemv_cpu(qtype, w.data() + (size_t) slot_e[j] * e_stride,
                                      x.data() + (size_t) slot_tok[j] * k, srow.data(), k, n);
        if (rc != LING3_OK) {
            std::printf("GROUP %-4s FAIL cpu rc=%d\n", TypeName(qtype), rc);
            ++g_fail;
            return;
        }
        for (int r = 0; r < n; ++r) {
            yref[(size_t) slot_tok[j] * n + r] += slot_w[j] * srow[r];
        }
    }
    void *dw = DevAlloc(w.size());
    void *dx = DevAlloc(x.size() * sizeof(float));
    // y must hold both the token-major scatter (n_tokens rows) and the
    // slot-major store rows of the bit-identity gate (n_slots rows).
    void *dy = DevAlloc(std::max(ydev.size(), (size_t) n_slots * n) * sizeof(float));
    void *d_gt = DevAlloc(g_slot_tok.size() * sizeof(int));
    void *d_gw = DevAlloc(g_slot_w.size() * sizeof(float));
    void *d_ge = DevAlloc(grp_e.size() * sizeof(int));
    void *d_go = DevAlloc(grp_off.size() * sizeof(int));
    void *d_gc = DevAlloc(grp_cnt.size() * sizeof(int));
    HIPCK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_gt, g_slot_tok.data(), g_slot_tok.size() * sizeof(int), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_gw, g_slot_w.data(), g_slot_w.size() * sizeof(float), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_ge, grp_e.data(), grp_e.size() * sizeof(int), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_go, grp_off.data(), grp_off.size() * sizeof(int), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(d_gc, grp_cnt.data(), grp_cnt.size() * sizeof(int), hipMemcpyHostToDevice));
    HIPCK(hipMemset(dy, 0, ydev.size() * sizeof(float)));
    {
        const hipError_t pre = hipGetLastError();
        if (pre != hipSuccess) {
            std::printf("GROUP %-4s PENDING hip error before value run: %s\n",
                        TypeName(qtype), hipGetErrorString(pre));
        }
    }
    const int rc = ling3_gemm_grp(qtype, dw, e_stride, (const int *) d_ge,
                                  (const int *) d_go, (const int *) d_gc,
                                  (int) grp_e.size(), (const float *) dx,
                                  (const int *) d_gt, (float *) dy, (const int *) d_gt,
                                  (const float *) d_gw, 1, k, n, max_cnt, nullptr, 0);
    HIPCK(hipDeviceSynchronize());
    HIPCK(hipMemcpy(ydev.data(), dy, ydev.size() * sizeof(float), hipMemcpyDeviceToHost));

    // P3 step 1 gate: the compacted grid (grp_zp prefix) must be
    // BIT-IDENTICAL to the legacy worst-case grid on the same inputs.
    // Deterministic comparison only: the batched production path stores
    // (accumulate=0), where each (slot, row) is written by exactly one
    // block. The accumulate=1 branch above scatters through atomicAdd, whose
    // float sum order varies between ANY two launches, grid or not, so it
    // cannot serve as the identity gate.
    std::vector<int> zp(grp_e.size() + 1, 0);
    int tacc = 0;
    for (size_t gi = 0; gi < grp_e.size(); ++gi) {
        zp[gi] = tacc;
        tacc += (grp_cnt[gi] + 15) / 16;
    }
    zp[grp_e.size()] = tacc;
    void *d_zp = DevAlloc(zp.size() * sizeof(int));
    HIPCK(hipMemcpy(d_zp, zp.data(), zp.size() * sizeof(int), hipMemcpyHostToDevice));
    std::vector<int> ident(g_slot_tok.size());
    for (size_t i = 0; i < ident.size(); ++i) ident[i] = (int) i;
    void *d_id = DevAlloc(ident.size() * sizeof(int));
    HIPCK(hipMemcpy(d_id, ident.data(), ident.size() * sizeof(int), hipMemcpyHostToDevice));
    std::vector<float> yslot((size_t) g_slot_tok.size() * n, -779.f);
    std::vector<float> yslot2(yslot.size(), -780.f);
    HIPCK(hipMemset(dy, 0, yslot.size() * sizeof(float)));
    const int rcs = ling3_gemm_grp(qtype, dw, e_stride, (const int *) d_ge,
                                   (const int *) d_go, (const int *) d_gc,
                                   (int) grp_e.size(), (const float *) dx,
                                   (const int *) d_gt, (float *) dy, (const int *) d_id,
                                   (const float *) d_gw, 0, k, n, max_cnt, nullptr, 0);
    HIPCK(hipDeviceSynchronize());
    HIPCK(hipMemcpy(yslot.data(), dy, yslot.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIPCK(hipMemset(dy, 0, yslot.size() * sizeof(float)));
    const int rcs2 = ling3_gemm_grp(qtype, dw, e_stride, (const int *) d_ge,
                                    (const int *) d_go, (const int *) d_gc,
                                    (int) grp_e.size(), (const float *) dx,
                                    (const int *) d_gt, (float *) dy, (const int *) d_id,
                                    (const float *) d_gw, 0, k, n, max_cnt,
                                    (const int *) d_zp, tacc);
    HIPCK(hipDeviceSynchronize());
    HIPCK(hipMemcpy(yslot2.data(), dy, yslot2.size() * sizeof(float), hipMemcpyDeviceToHost));
    bool zp_identical = rcs == LING3_OK && rcs2 == LING3_OK;
    for (size_t i = 0; i < yslot.size(); ++i) {
        zp_identical = zp_identical && (yslot[i] == yslot2[i]);
    }
    if (!zp_identical) {
        std::printf("GROUP %-4s compacted-grid BIT-IDENTITY FAIL\n", TypeName(qtype));
        ++g_fail;
    }
    if (rc != LING3_OK) {
        std::printf("GROUP %-4s n_expert=%d slots=%d FAIL dev rc=%d rcs=%d rcs2=%d\n",
                    TypeName(qtype), n_expert, n_slots, rc, rcs, rcs2);
        ++g_fail;
    } else {
        double maxabs = 0, scale = 1e-30;
        for (size_t i = 0; i < yref.size(); ++i) {
            const double d = std::fabs((double) yref[i] - (double) ydev[i]);
            if (d > maxabs) maxabs = d;
            scale = std::fmax(scale, std::fabs((double) yref[i]));
        }
        const bool ok = maxabs <= 1e-4 * scale;
        std::printf("GROUP %-4s k=%d n=%d experts=%d slots=%d %s max_abs=%.3e (ref scale %.2f) zp=%s\n",
                    TypeName(qtype), k, n, n_expert, n_slots, ok ? "PASS" : "FAIL", maxabs, scale,
                    zp_identical ? "identical" : "DIFFER");
        if (!ok) ++g_fail;
    }
    hipFree(dw); hipFree(dx); hipFree(dy); hipFree(d_gt); hipFree(d_gw);
    hipFree(d_ge); hipFree(d_go); hipFree(d_gc); hipFree(d_zp); hipFree(d_id);
}

// ---------------------------------------------------------------------------
// batched router vs ling3_route_cpu per token
// ---------------------------------------------------------------------------
void TestRoute(int n_tokens) {
    const int k = 96, n_expert = 16, n_used = 4, n_groups = 4, n_group_used = 2;
    const float w_scale = 1.7f;
    const int norm_w = 1;
    std::vector<float> x((size_t) n_tokens * k), gate((size_t) n_expert * k), bias(n_expert);
    for (auto &v : x) v = Frnd(-1.f, 1.f);
    for (auto &v : gate) v = Frnd(-1.f, 1.f);
    for (auto &v : bias) v = Frnd(-0.2f, 0.2f);

    std::vector<int> ids_ref((size_t) n_tokens * n_used, -1);
    std::vector<float> w_ref((size_t) n_tokens * n_used);
    for (int t = 0; t < n_tokens; ++t) {
        int rc = ling3_route_cpu(x.data() + (size_t) t * k, gate.data(), bias.data(), k,
                                 n_expert, n_used, n_groups, n_group_used, w_scale, norm_w,
                                 ids_ref.data() + (size_t) t * n_used,
                                 w_ref.data() + (size_t) t * n_used);
        if (rc != LING3_OK) {
            std::printf("ROUTE T=%d FAIL cpu rc=%d\n", n_tokens, rc);
            ++g_fail;
            return;
        }
    }
    std::vector<int> ids_dev((size_t) n_tokens * n_used, -1);
    std::vector<float> w_dev((size_t) n_tokens * n_used, -1.f);
    std::vector<int> rc_dev(n_tokens, -1);
    void *dx = DevAlloc(x.size() * sizeof(float));
    void *dg = DevAlloc(gate.size() * sizeof(float));
    void *db = DevAlloc(bias.size() * sizeof(float));
    void *di = DevAlloc(ids_dev.size() * sizeof(int));
    void *dw = DevAlloc(w_dev.size() * sizeof(float));
    void *dr = DevAlloc(rc_dev.size() * sizeof(int));
    HIPCK(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(dg, gate.data(), gate.size() * sizeof(float), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(db, bias.data(), bias.size() * sizeof(float), hipMemcpyHostToDevice));
    const int rc = ling3_route_batch((const float *) dx, (const float *) dg,
                                     (const float *) db, k, n_tokens, n_expert, n_used,
                                     n_groups, n_group_used, w_scale, norm_w, (int *) di,
                                     (float *) dw, (int *) dr);
    HIPCK(hipDeviceSynchronize());
    HIPCK(hipMemcpy(ids_dev.data(), di, ids_dev.size() * sizeof(int), hipMemcpyDeviceToHost));
    HIPCK(hipMemcpy(w_dev.data(), dw, w_dev.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIPCK(hipMemcpy(rc_dev.data(), dr, rc_dev.size() * sizeof(int), hipMemcpyDeviceToHost));
    bool ok = rc == LING3_OK;
    double maxw = 0;
    int bad_ids = 0, bad_rc = 0;
    for (int t = 0; t < n_tokens; ++t) {
        if (rc_dev[t] != LING3_OK) ++bad_rc;
        for (int s = 0; s < n_used; ++s) {
            if (ids_dev[(size_t) t * n_used + s] != ids_ref[(size_t) t * n_used + s]) ++bad_ids;
            maxw = std::fmax(maxw, (double) std::fabs(w_dev[(size_t) t * n_used + s] -
                                                      w_ref[(size_t) t * n_used + s]));
        }
    }
    ok = ok && bad_ids == 0 && bad_rc == 0 && maxw <= 1e-6;
    std::printf("ROUTE T=%d %s bad_ids=%d bad_rc=%d max_w_diff=%.3e\n", n_tokens,
                ok ? "PASS" : "FAIL", bad_ids, bad_rc, maxw);
    if (!ok) ++g_fail;
    hipFree(dx); hipFree(dg); hipFree(db); hipFree(di); hipFree(dw); hipFree(dr);
}

// ---------------------------------------------------------------------------
// P2-C: token-major gather-sum of slot partials vs a host sum
// ---------------------------------------------------------------------------
void TestGather(int n_tokens, int n_used, int k) {
    std::vector<float> part((size_t) n_tokens * n_used * k);
    std::vector<int> slot((size_t) n_tokens * n_used);
    for (auto &v : part) v = Frnd(-1.f, 1.f);
    for (int t = 0; t < n_tokens; ++t) {
        for (int s = 0; s < n_used; ++s) {
            slot[(size_t) t * n_used + s] = (int) (Rnd() % (unsigned) (n_tokens * n_used));
        }
    }
    std::vector<float> yref((size_t) n_tokens * k, 0.f), ydev((size_t) n_tokens * k, -777.f);
    for (int t = 0; t < n_tokens; ++t) {
        for (int r = 0; r < k; ++r) {
            float a = 0.f;
            for (int s = 0; s < n_used; ++s) {
                a += part[(size_t) slot[(size_t) t * n_used + s] * k + r];
            }
            yref[(size_t) t * k + r] = a;
        }
    }
    void *dp = DevAlloc(part.size() * sizeof(float));
    void *ds = DevAlloc(slot.size() * sizeof(int));
    void *dy = DevAlloc(ydev.size() * sizeof(float));
    HIPCK(hipMemcpy(dp, part.data(), part.size() * sizeof(float), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(ds, slot.data(), slot.size() * sizeof(int), hipMemcpyHostToDevice));
    const int rc = ling3_slot_gather_sum((const float *) dp, (const int *) ds,
                                         (float *) dy, n_used, k, n_tokens);
    HIPCK(hipDeviceSynchronize());
    HIPCK(hipMemcpy(ydev.data(), dy, ydev.size() * sizeof(float), hipMemcpyDeviceToHost));
    double maxabs = 0;
    for (size_t i = 0; i < yref.size(); ++i) {
        maxabs = std::fmax(maxabs, std::fabs((double) yref[i] - (double) ydev[i]));
    }
    const bool ok = rc == LING3_OK && maxabs == 0.0;
    std::printf("GATHER T=%d n_used=%d k=%d %s max_abs=%.3e\n", n_tokens, n_used, k,
                ok ? "PASS" : "FAIL", maxabs);
    if (!ok) ++g_fail;
    hipFree(dp); hipFree(ds); hipFree(dy);
}

}  // namespace

int main() {
    std::printf("ling3_gemm_test %s\n", ling3_gemm_version());
    // Dense: both FP4 variants at the model's real (k, n) shapes plus edge
    // tiles (n and m not multiples of the tile, m=1, k tail for F32).
    const int qts[] = {LING3_F32, LING3_Q8_0, LING3_Q6_K, LING3_Q4_0_ROCMFP4,
                       LING3_Q4_0_ROCMFP4_FAST};
    for (int q : qts) {
        TestDense(q, 64, 64, 1);
        TestDense(q, 64, 64, 3);
        TestDense(q, 128, 96, 33);    // n < TR edge + m crossing one token tile
        TestDense(q, 96, 128, 17);
        TestDense(q, 256, 192, 2);
    }
    TestDense(LING3_Q4_0_ROCMFP4_FAST, 2560, 4096, 1);    // wq shape, decode-size
    TestDense(LING3_Q4_0_ROCMFP4_FAST, 2560, 4096, 5);
    TestDense(LING3_Q4_0_ROCMFP4_FAST, 2560, 4096, 32);
    TestDense(LING3_Q4_0_ROCMFP4, 2560, 4096, 32);
    TestDense(LING3_Q4_0_ROCMFP4_FAST, 768, 2560, 32);    // expert gate/up shape
    TestDense(LING3_Q4_0_ROCMFP4_FAST, 2560, 768, 32);    // expert down shape
    TestDense(LING3_F32, 100, 70, 3);                     // k/m tails, non-tile k
    // Grouped: the routed-expert access pattern.
    TestGrouped(LING3_Q4_0_ROCMFP4_FAST, 256, 96, 8, 20);
    TestGrouped(LING3_Q4_0_ROCMFP4, 128, 64, 5, 13);
    TestGrouped(LING3_F32, 64, 64, 3, 7);
    // Groups crossing the small(<=8)/tiled(>8) kernel boundary, and a group
    // large enough to need two 32-token tiles.
    TestGrouped(LING3_Q4_0_ROCMFP4_FAST, 192, 128, 3, 26);
    TestGrouped(LING3_F32, 64, 64, 1, 40);
    // Router.
    TestRoute(1);
    TestRoute(9);
    TestRoute(33);
    // Gather-sum (P2-C).
    TestGather(5, 8, 64);
    TestGather(33, 8, 2560);
    TestGather(1, 1, 96);
    std::printf("%s fail=%d\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}
