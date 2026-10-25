// ling3_wmma_test.cpp — LS3 phase 3's unit gate for kernels/ling3_wmma.hip.
// Run inside gpu-run BEFORE the first full-model run (the brief's rule after
// H1's GPU reset): every kernel here is small-shape, synthetic-tensor, and
// checked against CPU transcriptions of the SAME algorithms.
//
//   1. ling3_wmma_codebook_check: the FP4->int8 perm codebook + ue4m3 half
//      scales, exhaustively on-device (every 16-bit four-nibble pattern,
//      both halves, both block types, both blocks of a row) against
//      ling3_rocmfp4_code / ling3_ue4m3_half.
//   2. the Q8_1 quantiser vs a CPU transcription (exact bytes + scales), for
//      identity and group-remapped destinations.
//   3. the grouped W8A8 WMMA GEMM vs the FORK-SEMANTICS oracle: an exact
//      int32 transcription of vec_dot_rocmfp4_q8_1 (same Q8_1 activation
//      quantisation, Codebook10 integer dot, per-half block scales, one
//      float epilogue per 32-block) — this catches every index, stride,
//      tile-prefix and lane-ownership bug, and it is the number PHASE3
//      asks for as "kernel-level error vs the fork's vec_dot_rocmfp4_q8_1".
//   4. the same outputs vs the F32 dequant oracle (ling3_gemv_cpu) — the
//      quantisation-cost number, normalised by the reference RMS.
//   5. the dense W8A8 WMMA GEMM at m = 1/5/64/256.
//   6. optional REAL weight rows: `gufo_ling3_wmma_test <raw-rows-file> <k>`
//      treats the file as consecutive FP4 rows and runs check 3/4 on them.
//
// Weights are synthetic but not kind: every FP4 code appears many times in
// every block and the scale bytes cover the whole ue4m3 range including the
// 0x7f guard.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ling3_wmma.hpp"
#include "ling3_moe.hpp"  // ling3_gemv_cpu

namespace {

int g_fail = 0;
std::FILE * g_dump = nullptr;  // LS3 PHASE8: raw ydev bytes per case (A/B bit gate)

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
    return qtype == LING3_Q4_0_ROCMFP4 ? "FP4" : "FP4F";
}

// --------------------------------------------------------------------------
// CPU transcription of the Q8_1 quantiser (ling3_wmma_quant): exact.
// Returns per (row, block): d and the 32 int8 codes.
// --------------------------------------------------------------------------
struct Q8Block {
    float d;
    int8_t q[32];
};

Q8Block QuantBlock(const float *xtok, int k, int blk) {
    Q8Block b;
    float max_abs = 0.f;
    for (int l = 0; l < 32; ++l) {
        const float v = xtok[blk * 32 + l];
        max_abs = fmaxf(max_abs, fabsf(v));
    }
    const float d = max_abs / 127.0f;
    const float id = (d != 0.f) ? (1.f / d) : 0.f;
    for (int l = 0; l < 32; ++l) {
        b.q[l] = (int8_t) roundf(xtok[blk * 32 + l] * id);
    }
    b.d = d;
    return b;
}

// --------------------------------------------------------------------------
// CPU transcription of the W8A8 fork semantics for one (row, token) pair:
//   sum over 32-blocks of db * (e0 * sumi0 + e1 * sumi1)
// accumulated like the kernel: one float term per 16-element half per block.
// --------------------------------------------------------------------------
double ForkDot(int qtype, const uint8_t *wrow, const float *xtok, int k) {
    const int tsz = ling3_type_size(qtype);
    const int nblk = k / 32;
    double acc = 0.0;
    for (int blk = 0; blk < nblk; ++blk) {
        const Q8Block qb = QuantBlock(xtok, k, blk);
        const uint8_t *wblk = wrow + (size_t) blk * tsz;
        const float e0 = ling3_ue4m3_half(wblk[16]);
        const float e1 = (qtype == LING3_Q4_0_ROCMFP4)
                             ? ling3_ue4m3_half(wblk[17])
                             : e0;
        long long sumi0 = 0, sumi1 = 0;
        for (int l = 0; l < 16; ++l) {
            const uint8_t b = wblk[l & 15];
            const int nib0 = (int) (b & 0x0f);
            const int nib1 = (int) (b >> 4);
            sumi0 += (long long) ling3_rocmfp4_code((uint32_t) nib0) * qb.q[l];
            sumi1 += (long long) ling3_rocmfp4_code((uint32_t) nib1) * qb.q[16 + l];
        }
        acc += (double) ((double) e0 * (double) qb.d) * (double) sumi0 +
               (double) ((double) e1 * (double) qb.d) * (double) sumi1;
    }
    return acc;
}

// RMS-normalised error report (LS1's lesson: not per-row relative).
void Report(const char *tag, const char *sub, const std::vector<float> &ref,
            const std::vector<float> &dev, double tol_rms) {
    double maxabs = 0, rms = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs((double) ref[i] - (double) dev[i]);
        if (d > maxabs) maxabs = d;
        rms += (double) ref[i] * (double) ref[i];
    }
    rms = std::sqrt(rms / (double) ref.size());
    const double rel = maxabs / (rms > 1e-30 ? rms : 1e-30);
    const bool ok = rel <= tol_rms;
    std::printf("%s %-4s %s max_abs=%.3e rms=%.3e err/rms=%.2e %s\n", tag, sub,
                ok ? "PASS" : "FAIL", maxabs, rms, rel, ok ? "" : "(over tolerance)");
    if (!ok) ++g_fail;
}

// --------------------------------------------------------------------------
// 2. quantiser: exact bytes + scales, identity and remapped destinations
// --------------------------------------------------------------------------
void TestQuant(int qtype, int n_rows, int k) {
    const int nblk = k / 32;
    std::vector<float> x((size_t) n_rows * k * 2);
    for (auto &v : x) v = Frnd(-2.f, 2.f);
    const int tiles = (n_rows + 15) / 16;
    std::vector<uint8_t> host_q8((size_t) tiles * nblk * 576, 0xAB);

    // remapped destination: reverse the tile order, rotate lanes.
    std::vector<int> dest(n_rows);
    for (int r = 0; r < n_rows; ++r) {
        const int tile = (r >> 4);
        dest[r] = ((tiles - 1 - tile) << 4) | (r & 15);
    }

    void *dx = DevAlloc(x.size() * sizeof(float));
    void *dd = DevAlloc(dest.size() * sizeof(int));
    void *dq = DevAlloc(host_q8.size());
    HIPCK(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(dd, dest.data(), dest.size() * sizeof(int), hipMemcpyHostToDevice));
    for (int mode = 0; mode < 2; ++mode) {
        std::fill(host_q8.begin(), host_q8.end(), 0xAB);
        HIPCK(hipMemcpy(dq, host_q8.data(), host_q8.size(), hipMemcpyHostToDevice));
        const int rc = ling3_wmma_quant((const float *) dx, nullptr,
                                        mode ? (const int *) dd : nullptr, dq, n_rows, k);
        HIPCK(hipDeviceSynchronize());
        HIPCK(hipMemcpy(host_q8.data(), dq, host_q8.size(), hipMemcpyDeviceToHost));
        if (rc != LING3_OK) {
            std::printf("QUANT %s FAIL rc=%d\n", mode ? "remap" : "ident", rc);
            ++g_fail;
            continue;
        }
        // The kernel file compiles with -ffast-math, which lowers the block
        // scale's `max/127` to a reciprocal multiply on gfx11 (measured: the
        // GPU byte equals max*(1/127) exactly) -- the same compiled form as
        // gufo's quantizer and LS1's shipped ling3_gemv_q8 under the same
        // flags. The gate therefore pins every byte against the UNION of
        // the two exact CPU lowerings; anything outside that union is an
        // indexing/layout bug, which is what the gate exists to catch.
        long long bad = 0;
        for (int r = 0; r < n_rows; ++r) {
            const int dp = mode ? dest[r] : r;
            const int tile = dp >> 4, tl = dp & 15;
            for (int blk = 0; blk < nblk; ++blk) {
                float mx = 0.f;
                for (int l = 0; l < 32; ++l) {
                    mx = fmaxf(mx, fabsf(x[(size_t) r * k + blk * 32 + l]));
                }
                const float ddiv = mx / 127.0f;
                const float drec = mx * (1.0f / 127.0f);
                const float scales[2] = {ddiv, drec};
                const uint8_t *t = host_q8.data() +
                                   ((size_t) (tile * nblk + blk)) * 576;
                const float dgpu = *reinterpret_cast<const float *>(t + 512 + tl * 4);
                const uint32_t ug = *(const uint32_t *) &dgpu;
                if (ug != *(const uint32_t *) &ddiv && ug != *(const uint32_t *) &drec) {
                    ++bad;
                }
                for (int l = 0; l < 32; ++l) {
                    const int half = l >> 4, pos = l & 15;
                    const uint8_t got = t[half * 256 + tl * 16 + pos];
                    bool ok = false;
                    for (int si = 0; si < 2 && !ok; ++si) {
                        const float d = scales[si];
                        const float id = (d != 0.f) ? (1.f / d) : 0.f;
                        const int8_t q =
                            (int8_t) roundf(x[(size_t) r * k + blk * 32 + l] * id);
                        ok = got == (uint8_t) q;
                    }
                    if (!ok) ++bad;
                }
            }
        }
        std::printf("QUANT %-4s rows=%d k=%d %s (%lld bad bytes, dest=%s)\n",
                    TypeName(qtype), n_rows, k, bad == 0 ? "PASS" : "FAIL", bad,
                    mode ? "remap" : "ident");
        if (bad != 0) ++g_fail;
    }
    hipFree(dx); hipFree(dd); hipFree(dq);
}

// --------------------------------------------------------------------------
// 3/4. grouped GEMM vs fork-semantics + F32 oracles
// --------------------------------------------------------------------------
void RunGrouped(int qtype, int k, int n, int n_expert, int n_tokens,
                const std::vector<int> &cnts, const char *tag) {
    const size_t rb = ling3_row_bytes(qtype, k);
    if (rb == 0) return;
    const size_t e_stride = (size_t) n * rb;
    std::vector<uint8_t> w(e_stride * n_expert);
    for (auto &v : w) v = (uint8_t) Rnd();
    std::vector<float> x((size_t) n_tokens * k);
    for (auto &v : x) v = Frnd(-1.5f, 1.5f);

    // groups: expert i gets cnts[i] consecutive slots (cap-free here; the
    // model's cap is 64 and test cnts stay <= 40).
    std::vector<int> grp_e, grp_off, grp_cnt;
    int slots = 0;
    for (int e = 0; e < n_expert; ++e) {
        if (cnts[e] <= 0) continue;
        grp_e.push_back(e);
        grp_off.push_back(slots);
        grp_cnt.push_back(cnts[e]);
        slots += cnts[e];
    }
    if (slots == 0) return;
    std::vector<int> slot_tok(slots), slot_dest(slots);
    std::vector<float> slot_w(slots);
    std::vector<int> zp(grp_e.size() + 1, 0);
    int tacc = 0;
    for (size_t g = 0; g < grp_e.size(); ++g) {
        zp[g] = tacc;
        tacc += (grp_cnt[g] + 15) / 16;
        for (int c = 0; c < grp_cnt[g]; ++c) {
            const int j = grp_off[g] + c;
            slot_tok[j] = (int) (Rnd() % (unsigned) n_tokens);
            slot_w[j] = Frnd(0.05f, 0.95f);
            slot_dest[j] = ((zp[g] + (c >> 4)) << 4) | (c & 15);
        }
    }
    zp[grp_e.size()] = tacc;

    // CPU references, slot-major y (the WMMA contract).
    std::vector<float> yfork((size_t) slots * n), yf32((size_t) slots * n),
        ydev((size_t) slots * n, -777.f);
    std::vector<float> srow(n);
    for (int j = 0; j < slots; ++j) {
        const int tok = slot_tok[j];
        int expert = -1;
        for (int g = 0; g < (int) grp_e.size(); ++g) {
            if (j >= grp_off[g] && j < grp_off[g] + grp_cnt[g]) {
                expert = grp_e[g];
                break;
            }
        }
        const uint8_t *wrow = w.data() + (size_t) expert * e_stride;
        for (int r = 0; r < n; ++r) {
            yfork[(size_t) j * n + r] =
                (float) (slot_w[j] * ForkDot(qtype, wrow + (size_t) r * rb,
                                             x.data() + (size_t) tok * k, k));
        }
        const int rc32 = ling3_gemv_cpu(qtype, wrow, x.data() + (size_t) tok * k,
                                        srow.data(), k, n);
        if (rc32 != LING3_OK) {
            std::printf("%s F32 oracle rc=%d\n", tag, rc32);
            ++g_fail;
            return;
        }
        for (int r = 0; r < n; ++r) {
            yf32[(size_t) j * n + r] = slot_w[j] * srow[r];
        }
    }

    void *dw = DevAlloc(w.size());
    void *dx = DevAlloc(x.size() * sizeof(float));
    void *dst = DevAlloc(slot_tok.size() * sizeof(int));
    void *dsd = DevAlloc(slot_dest.size() * sizeof(int));
    void *dsw = DevAlloc(slot_w.size() * sizeof(float));
    void *dge = DevAlloc(grp_e.size() * sizeof(int));
    void *dgo = DevAlloc(grp_off.size() * sizeof(int));
    void *dgc = DevAlloc(grp_cnt.size() * sizeof(int));
    void *dzp = DevAlloc(zp.size() * sizeof(int));
    void *dq8 = DevAlloc((size_t) tacc * (k / 32) * 576);
    void *dy = DevAlloc(ydev.size() * sizeof(float));
    HIPCK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(dst, slot_tok.data(), slot_tok.size() * sizeof(int), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(dsd, slot_dest.data(), slot_dest.size() * sizeof(int), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(dsw, slot_w.data(), slot_w.size() * sizeof(float), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(dge, grp_e.data(), grp_e.size() * sizeof(int), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(dgo, grp_off.data(), grp_off.size() * sizeof(int), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(dgc, grp_cnt.data(), grp_cnt.size() * sizeof(int), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(dzp, zp.data(), zp.size() * sizeof(int), hipMemcpyHostToDevice));

    int rc = ling3_wmma_quant((const float *) dx, (const int *) dst,
                              (const int *) dsd, dq8, slots, k);
    HIPCK(hipDeviceSynchronize());
    if (rc == LING3_OK) {
        rc = ling3_wmma_gemm_grp(qtype, dw, e_stride, (const int *) dge,
                                 (const int *) dgo, (const int *) dgc,
                                 (const int *) dzp, (int) grp_e.size(), tacc, dq8,
                                 (float *) dy, (const float *) dsw, k, n);
        HIPCK(hipDeviceSynchronize());
    }
    HIPCK(hipMemcpy(ydev.data(), dy, ydev.size() * sizeof(float), hipMemcpyDeviceToHost));
    if (rc != LING3_OK) {
        std::printf("%s %-4s FAIL rc=%d\n", tag, TypeName(qtype), rc);
        ++g_fail;
    } else {
        if (g_dump != nullptr) {  // LS3 PHASE8: raw bytes for the BK A/B gate
            std::fprintf(g_dump, "CASE %s %s k=%d n=%d slots=%d\n", tag,
                         TypeName(qtype), k, n, slots);
            std::fwrite(ydev.data(), sizeof(float), ydev.size(), g_dump);
        }
        char sub[64];
        std::snprintf(sub, sizeof sub, "k=%d n=%d g=%zu slots=%d", k, n,
                      grp_e.size(), slots);
        Report(tag, sub, yfork, ydev, 3e-4);   // float32 vs double epilogue
        Report("Q-COST", sub, yf32, ydev, 0.05);  // int8 activation cost
    }
    hipFree(dw); hipFree(dx); hipFree(dst); hipFree(dsd); hipFree(dsw);
    hipFree(dge); hipFree(dgo); hipFree(dgc); hipFree(dzp); hipFree(dq8); hipFree(dy);
}

void TestGrouped(int qtype) {
    // real shapes: gate/up (k=2560 -> n=768), down (k=768 -> n=2560);
    // plus a tiny shape and the 1..40-slot sweep over 1..8 experts.
    std::vector<int> c1 = {1};
    RunGrouped(qtype, 2560, 768, 1, 1, c1, "GRP1 ");
    std::vector<int> c16 = {16};
    RunGrouped(qtype, 2560, 768, 1, 16, c16, "GRP16");
    std::vector<int> c17 = {17};
    RunGrouped(qtype, 768, 2560, 1, 5, c17, "GRP17");
    std::vector<int> c40 = {40, 33, 7, 1, 21, 16, 2, 40};
    RunGrouped(qtype, 768, 2560, 8, 37, c40, "GRP40");
    std::vector<int> csm = {3, 5, 2};
    RunGrouped(qtype, 64, 32, 3, 6, csm, "GRPSM");
    // uneven alignment: offsets 0,3,8,16,17 (prefix of 3 then 5 then 2)
    std::vector<int> cu = {3, 5, 2, 40};
    RunGrouped(qtype, 2560, 768, 4, 50, cu, "GRPUA");
}

// --------------------------------------------------------------------------
// 5. dense GEMM at m = 1/5/64/256
// --------------------------------------------------------------------------
void TestDense(int qtype, int k, int n, int m) {
    const size_t rb = ling3_row_bytes(qtype, k);
    if (rb == 0) return;
    std::vector<uint8_t> w(rb * (size_t) n);
    for (auto &v : w) v = (uint8_t) Rnd();
    std::vector<float> x((size_t) m * k);
    for (auto &v : x) v = Frnd(-1.5f, 1.5f);
    std::vector<float> yfork((size_t) m * n), yf32((size_t) m * n),
        ydev((size_t) m * n, -777.f);
    std::vector<float> srow(n);
    for (int t = 0; t < m; ++t) {
        for (int r = 0; r < n; ++r) {
            yfork[(size_t) t * n + r] =
                (float) ForkDot(qtype, w.data() + (size_t) r * rb,
                                x.data() + (size_t) t * k, k);
        }
        const int rc32 = ling3_gemv_cpu(qtype, w.data(), x.data() + (size_t) t * k,
                                        srow.data(), k, n);
        if (rc32 != LING3_OK) {
            std::printf("DENSE F32 oracle rc=%d\n", rc32);
            ++g_fail;
            return;
        }
        for (int r = 0; r < n; ++r) yf32[(size_t) t * n + r] = srow[r];
    }
    void *dw = DevAlloc(w.size());
    void *dx = DevAlloc(x.size() * sizeof(float));
    void *dq8 = DevAlloc((size_t) ((m + 15) / 16) * (k / 32) * 576);
    void *dy = DevAlloc(ydev.size() * sizeof(float));
    HIPCK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
    int rc = ling3_wmma_quant((const float *) dx, nullptr, nullptr, dq8, m, k);
    if (rc == LING3_OK) {
        rc = ling3_wmma_gemm_dense(qtype, dw, dq8, (float *) dy, k, n, m);
        HIPCK(hipDeviceSynchronize());
    }
    HIPCK(hipMemcpy(ydev.data(), dy, ydev.size() * sizeof(float), hipMemcpyDeviceToHost));
    if (rc != LING3_OK) {
        std::printf("DENSE %-4s k=%d n=%d m=%d FAIL rc=%d\n", TypeName(qtype), k, n, m, rc);
        ++g_fail;
    } else {
        char sub[64];
        std::snprintf(sub, sizeof sub, "k=%d n=%d m=%d", k, n, m);
        Report("DENSE", sub, yfork, ydev, 3e-4);
        Report("Q-COST", sub, yf32, ydev, 0.05);
    }
    hipFree(dw); hipFree(dx); hipFree(dq8); hipFree(dy);
}

// --------------------------------------------------------------------------
// 6. real weight rows from a raw file (extracted from the GGUF on the box)
// --------------------------------------------------------------------------
void TestRealRows(const char *path, int k) {
    FILE *f = std::fopen(path, "rb");
    if (f == nullptr) {
        std::printf("REAL-ROWS open %s FAILED\n", path);
        ++g_fail;
        return;
    }
    std::vector<uint8_t> w;
    uint8_t buf[65536];
    size_t got;
    while ((got = std::fread(buf, 1, sizeof buf, f)) > 0) {
        w.insert(w.end(), buf, buf + got);
    }
    std::fclose(f);
    const size_t rb = (size_t) (k / 32) * 17;
    if (w.size() < rb) {
        std::printf("REAL-ROWS too short (%zu)\n", w.size());
        ++g_fail;
        return;
    }
    // both types: the rows are FAST (101); also decode them as dual (100) so
    // the e[1] path is exercised on real bytes.
    for (int qi = 0; qi < 2; ++qi) {
        const int qtype = qi ? LING3_Q4_0_ROCMFP4 : LING3_Q4_0_ROCMFP4_FAST;
        const size_t rbq = ling3_row_bytes(qtype, k);
        if (rbq == 0 || w.size() < rbq) continue;
        const int n = (int) (w.size() / rbq);
        const int m = 1;
        std::vector<float> x((size_t) m * k);
        for (auto &v : x) v = Frnd(-1.5f, 1.5f);
        std::vector<float> yfork((size_t) m * n), yf32((size_t) m * n),
            ydev((size_t) m * n, -777.f);
        std::vector<float> srow(n);
        for (int r = 0; r < n; ++r) {
            yfork[r] = (float) ForkDot(qtype, w.data() + (size_t) r * rbq,
                                       x.data(), k);
        }
        const int rc32 = ling3_gemv_cpu(qtype, w.data(), x.data(), srow.data(), k, n);
        if (rc32 != LING3_OK) {
            std::printf("REAL-ROWS F32 oracle rc=%d\n", rc32);
            ++g_fail;
            return;
        }
        yf32.assign(srow.begin(), srow.end());
        void *dw = DevAlloc(w.size());
        void *dx = DevAlloc(x.size() * sizeof(float));
        void *dq8 = DevAlloc((size_t) ((m + 15) / 16) * (k / 32) * 576);
        void *dy = DevAlloc(ydev.size() * sizeof(float));
        HIPCK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
        HIPCK(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
        int rc = ling3_wmma_quant((const float *) dx, nullptr, nullptr, dq8, m, k);
        if (rc == LING3_OK) {
            rc = ling3_wmma_gemm_dense(qtype, dw, dq8, (float *) dy, k, n, m);
            HIPCK(hipDeviceSynchronize());
        }
        HIPCK(hipMemcpy(ydev.data(), dy, ydev.size() * sizeof(float), hipMemcpyDeviceToHost));
        if (rc != LING3_OK) {
            std::printf("REAL-ROWS %-4s FAIL rc=%d\n", TypeName(qtype), rc);
            ++g_fail;
        } else {
            char sub[64];
            std::snprintf(sub, sizeof sub, "k=%d rows=%d", k, n);
            Report("REAL", sub, yfork, ydev, 3e-4);
            Report("REAL-Q", sub, yf32, ydev, 0.05);
        }
        hipFree(dw); hipFree(dx); hipFree(dq8); hipFree(dy);
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--dump") == 0) {
        g_dump = std::fopen(argv[2], "wb");
        std::printf("DUMP -> %s (%s)\n", argv[2],
                    g_dump != nullptr ? "open" : "FAILED");
        if (g_dump == nullptr) ++g_fail;
    }
    if (argc >= 3 && std::strcmp(argv[1], "--dump") != 0) {
        TestRealRows(argv[1], std::atoi(argv[2]));
    }
    const int cb = ling3_wmma_codebook_check();
    std::printf("CODEBOOK %s\n", cb == LING3_OK ? "PASS" : "FAIL");
    if (cb != LING3_OK) ++g_fail;

    for (int qi = 0; qi < 2; ++qi) {
        const int qtype = qi ? LING3_Q4_0_ROCMFP4 : LING3_Q4_0_ROCMFP4_FAST;
        TestQuant(qtype, 37, 128);
        TestGrouped(qtype);
        for (int m : {1, 5, 64, 256}) {
            TestDense(qtype, 2560, 768, m);
        }
        TestDense(qtype, 768, 2560, 64);
        TestDense(qtype, 640, 192, 3);   // odd tile tails
    }
    if (g_dump != nullptr) {
        std::fclose(g_dump);
    }
    std::printf("%s fail=%d\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}
