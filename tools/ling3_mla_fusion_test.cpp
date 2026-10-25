// ling3_mla_fusion_test.cpp — LC1 SPEC4 unit gate for the RV25 MLA decode
// fusions against the exact unfused sequence they replace:
//
//   M1   mla_pre_fused   = rope(q) + rope(kv) + latent_norm + cache_append
//   Q2   the wo Q8_1 quantisation as the attention kernel's epilogue
//
// Both decode arms, at the model's real MLA shapes:
//   n_pos   100 -> the SERIAL arm (attn_kernel, mla_block)
//   n_pos 1630 -> the SPLIT-K arm (attn_sk_reduce_kernel, mla_block_splitk)
//
// Every comparison is a whole-buffer memcmp, not a tolerance: the fusions claim
// bit-identity, and a 1-ulp drift is a failure. Compared per case: the attention
// output [4096], the whole KV cache the append touches, the Q8_1 codes [4096]
// and the Q8_1 scales [128] against k_q8_1_quant's own output, and (at the
// model's wo shape 4096 -> 2560) the wo GEMV's result through the fused handoff
// against the old quant+cta64 pair.
//
// It also asserts the fused path actually RAN (the launch counters), so a
// LING3_MLA_DECODE_FUSED=0 environment cannot make this gate pass vacuously.
// Synthetic tensors; no model load. Run inside gpu-run.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ling3_gemv_q8.hpp"
#include "ling3_mla.hpp"
#include "ling3_quant.hpp"

namespace {

int g_fail = 0;
unsigned g_state = 20261004u;

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

void * DevAlloc(size_t bytes) {
    void * p = nullptr;
    HIPCK(hipMalloc(&p, bytes + 64));
    return p;
}

void CheckMem(const char * what, const void * a, const void * b, size_t n) {
    const bool same = std::memcmp(a, b, n) == 0;
    std::printf("    %-24s %8zu B  %s\n", what, n, same ? "IDENTICAL" : "DIFFERENT");
    if (!same) {
        ++g_fail;
        const unsigned char * pa = (const unsigned char *) a;
        const unsigned char * pb = (const unsigned char *) b;
        for (size_t i = 0; i < n; ++i) {
            if (pa[i] != pb[i]) {
                std::printf("      first diff at byte %zu: %02x vs %02x\n", i, pa[i], pb[i]);
                break;
            }
        }
    }
}

void CheckTrue(const char * what, bool ok) {
    std::printf("    %-24s %s\n", what, ok ? "OK" : "FAILED");
    if (!ok) ++g_fail;
}

/// FP4 weight rows, the way tools/ling3_gemv_q8_test.cpp makes them: every one
/// of the sixteen codes appears many times per block and the scale bytes span
/// the whole ue4m3 range.
std::vector<std::uint8_t> MakeWeights(int qtype, int k, int n) {
    const std::size_t rb = ling3_row_bytes(qtype, k);
    std::vector<std::uint8_t> w(n * rb);
    const int nb = k / 32;
    const std::size_t tsz = ling3_type_size(qtype);
    for (int r = 0; r < n; ++r) {
        std::uint8_t * row = w.data() + (std::size_t) r * rb;
        for (int b = 0; b < nb; ++b) {
            std::uint8_t * blk = row + (std::size_t) b * tsz;
            for (int i = 0; i < 16; ++i) {
                blk[i] = (std::uint8_t) (Rnd() & 0xFF);
            }
            blk[16] = (std::uint8_t) (Rnd() % 256);
            if (qtype == LING3_Q4_0_ROCMFP4) {
                blk[17] = (std::uint8_t) (Rnd() % 256);
            }
        }
    }
    return w;
}

struct Shared {
    // weights
    void * d_wk = nullptr, * d_wv = nullptr, * d_knorm = nullptr;
    void * d_wkT = nullptr, * d_wvT = nullptr;
    ling3::MlaWeights w;
    // split-K partials (sized for the FIXED slot cap, like the engine)
    void * d_qabs = nullptr, * d_pm = nullptr, * d_pl = nullptr, * d_pa = nullptr;
    // the wo matrix, for the handoff check
    void * d_wo = nullptr;
};

}  // namespace

int main(int argc, char ** argv) {
    std::setbuf(stdout, nullptr);
    (void) argc;
    (void) argv;

    // Real MLA geometry (ling3_mla.hpp's defaults are the model's).
    ling3::MlaShape sh;
    ling3::MlaScalars sc;
    const int q_dim = sh.q_dim();      // 6144
    const int kv_a  = sh.kv_a();       // 576
    const int o_dim = sh.o_dim();      // 4096
    const int k_wo  = o_dim;           // wo: [4096 -> 2560]
    const int n_wo  = sh.n_embd;
    const int slice = ling3::kMlaSplitkSlice;
    const size_t slots = (size_t) ling3::kMlaSplitkMaxSlices;

    std::printf("ling3_mla_fusion_test  n_head=%d head_k=%d head_v=%d nope=%d rot=%d "
                "kv=%d q_dim=%d kv_a=%d o_dim=%d wo=[%d -> %d]\n",
                sh.n_head, sh.head_k, sh.head_v, sh.nope, sh.rot, sh.kv,
                q_dim, kv_a, o_dim, k_wo, n_wo);
    CheckTrue("knob LING3_MLA_DECODE_FUSED", ling3::mla_decode_fused());

    // ---- weights ----------------------------------------------------------
    std::vector<float> wk((size_t) sh.n_head * sh.kv * sh.nope);
    std::vector<float> wv((size_t) sh.n_head * sh.head_v * sh.kv);
    std::vector<float> knorm((size_t) sh.kv);
    for (auto & v : wk) v = Frnd(-0.05f, 0.05f);
    for (auto & v : wv) v = Frnd(-0.05f, 0.05f);
    for (auto & v : knorm) v = 0.5f + Frnd(0.0f, 1.0f);

    // Row-transposed copies, exactly as the loader builds them (the decode
    // prep / de-absorb read these).
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

    Shared S;
    S.d_wk = DevAlloc(wk.size() * 4); S.d_wv = DevAlloc(wv.size() * 4);
    S.d_knorm = DevAlloc(knorm.size() * 4);
    S.d_wkT = DevAlloc(wkT.size() * 4); S.d_wvT = DevAlloc(wvT.size() * 4);
    HIPCK(hipMemcpy(S.d_wk, wk.data(), wk.size() * 4, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(S.d_wv, wv.data(), wv.size() * 4, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(S.d_knorm, knorm.data(), knorm.size() * 4, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(S.d_wkT, wkT.data(), wkT.size() * 4, hipMemcpyHostToDevice));
    HIPCK(hipMemcpy(S.d_wvT, wvT.data(), wvT.size() * 4, hipMemcpyHostToDevice));
    S.w.wk_b = (const float *) S.d_wk;
    S.w.wv_b = (const float *) S.d_wv;
    S.w.kv_a_norm = (const float *) S.d_knorm;
    S.w.wk_bT = (const float *) S.d_wkT;
    S.w.wv_bT = (const float *) S.d_wvT;

    S.d_qabs = DevAlloc((size_t) sh.n_head * sh.kv * 4);
    S.d_pm   = DevAlloc((size_t) sh.n_head * slots * 4);
    S.d_pl   = DevAlloc((size_t) sh.n_head * slots * 4);
    S.d_pa   = DevAlloc((size_t) sh.n_head * slots * (size_t) sh.kv * 4);

    // The wo matrix the handoff check projects against.
    {
        const int qt = LING3_Q4_0_ROCMFP4_FAST;
        std::vector<std::uint8_t> w_wo = MakeWeights(qt, k_wo, n_wo);
        S.d_wo = DevAlloc(w_wo.size());
        HIPCK(hipMemcpy(S.d_wo, w_wo.data(), w_wo.size(), hipMemcpyHostToDevice));
    }

    // ---- the two decode arms ---------------------------------------------
    //   n_pos 100   -> serial attn_kernel   (MlaSplitkMin is 1024)
    //   n_pos 1630  -> split-K attn_sk_reduce_kernel
    for (int n_pos : {100, 1630}) {
        const bool splitk = n_pos >= 1024;
        std::printf("\n== case n_pos=%d  arm=%s ==\n", n_pos,
                    splitk ? "split-K (attn_sk_reduce_kernel)" : "serial (attn_kernel)");

        // Inputs: one token, one sequence, at position n_pos-1.
        std::vector<float> q((size_t) q_dim), kv((size_t) kv_a), gate((size_t) sh.n_head);
        for (auto & v : q) v = Frnd(-0.5f, 0.5f);
        for (auto & v : kv) v = Frnd(-0.5f, 0.5f);
        for (auto & v : gate) v = Frnd(-2.f, 2.f);
        // The cache: rows [0, n_pos-1) are history, row n_pos-1 is what this
        // call appends over.
        std::vector<float> cache((size_t) n_pos * kv_a);
        for (auto & v : cache) v = Frnd(-0.5f, 0.5f);
        const int pos_val = n_pos - 1;

        // ---- reference: the UNFUSED sequence, exactly what the block entry
        // launched before LC1 (four stages + attention without the epilogue),
        // then k_q8_1_quant for the wo activation.
        void * d_q  = DevAlloc(q.size() * 4);
        void * d_kv = DevAlloc(kv.size() * 4);
        void * d_g  = DevAlloc(gate.size() * 4);
        void * d_c  = DevAlloc(cache.size() * 4);
        void * d_pos = DevAlloc(sizeof(int));
        void * d_out = DevAlloc((size_t) o_dim * 4);
        void * d_scores = DevAlloc((size_t) sh.n_head * (size_t) n_pos * 4);
        HIPCK(hipMemcpy(d_q, q.data(), q.size() * 4, hipMemcpyHostToDevice));
        HIPCK(hipMemcpy(d_kv, kv.data(), kv.size() * 4, hipMemcpyHostToDevice));
        HIPCK(hipMemcpy(d_g, gate.data(), gate.size() * 4, hipMemcpyHostToDevice));
        HIPCK(hipMemcpy(d_c, cache.data(), cache.size() * 4, hipMemcpyHostToDevice));
        HIPCK(hipMemcpy(d_pos, &pos_val, sizeof(int), hipMemcpyHostToDevice));

        ling3::MlaInputs in;
        in.q    = ling3::make_view((float *) d_q, q_dim, 1);
        in.kv_a = ling3::make_view((float *) d_kv, kv_a, 1);
        in.gate = ling3::make_view((float *) d_g, sh.n_head, 1);
        in.n_tokens = 1;
        in.n_seqs   = 1;
        ling3::MlaCache ch;
        ch.kv = (float *) d_c; ch.s_pos = kv_a; ch.s_seq = 0; ch.max_pos = n_pos;
        const ling3::View outv = ling3::make_view((float *) d_out, o_dim, 1);

        ling3::mla_fusion_stats_reset();

        // The four stage launches, in the order both block entries used.
        ling3::mla_rope(sh, (float *) d_q + sh.nope, sh.head_k, sh.n_head,
                        1, in.q.s_tok, in.q.s_seq, 1, (const int *) d_pos, nullptr);
        ling3::mla_rope(sh, (float *) d_kv + sh.kv, 0, 1,
                        1, in.kv_a.s_tok, in.kv_a.s_seq, 1, (const int *) d_pos, nullptr);
        ling3::mla_latent_norm(sh, sc, S.w, (float *) d_kv, 1, 1,
                               in.kv_a.s_tok, in.kv_a.s_seq, nullptr);
        if (splitk) {
            // mla_block_splitk passed NO pos_d to cache_append (npos = n_pos).
            ling3::mla_cache_append(sh, ch, in.kv_a.p, 1, in.kv_a.s_tok, 1,
                                    n_pos, nullptr);
            ling3::mla_attn_splitk(sh, sc, S.w, ch, n_pos, in, outv,
                                   (float *) S.d_qabs, (float *) S.d_pm,
                                   (float *) S.d_pl, (float *) S.d_pa,
                                   slice, 2, nullptr);
        } else {
            // mla_block passed pos_d to cache_append (npos = pos_d[0] + 1).
            ling3::mla_cache_append(sh, ch, in.kv_a.p, 1, in.kv_a.s_tok, 1,
                                    n_pos, nullptr, (const int *) d_pos);
            ling3::mla_attn(sh, sc, S.w, ch, n_pos, in, outv,
                            (float *) d_scores, nullptr, (const int *) d_pos);
        }
        HIPCK(hipDeviceSynchronize());

        std::vector<float> out_ref((size_t) o_dim), cache_ref(cache.size());
        std::vector<std::int8_t> qs_ref((size_t) o_dim);
        std::vector<float> ds_ref((size_t) o_dim / 32);
        HIPCK(hipMemcpy(out_ref.data(), d_out, (size_t) o_dim * 4, hipMemcpyDeviceToHost));
        HIPCK(hipMemcpy(cache_ref.data(), d_c, cache.size() * 4, hipMemcpyDeviceToHost));
        // The unfused wo quantiser: k_q8_1_quant's own output is the reference
        // the epilogue has to reproduce byte for byte.
        CheckTrue("unfused quant rc",
                  ling3_quant_q8((const float *) d_out, o_dim,
                                 LING3_Q8_SLOT_GENERIC, nullptr) == LING3_OK);
        CheckTrue("unfused peek rc",
                  ling3_q8_peek(LING3_Q8_SLOT_GENERIC, qs_ref.data(),
                                ds_ref.data(), o_dim) == LING3_OK);

        const int ref_pre = ling3::mla_pre_fused_launches();
        const int ref_wo  = ling3::mla_wo_q8_epilogues();
        CheckTrue("unfused sequence left M1 alone", ref_pre == 0);
        CheckTrue("unfused sequence left Q2 alone", ref_wo == 0);

        // ---- fused: the block entry, which now runs mla_pre_fused + the
        // epilogue because LING3_MLA_DECODE_FUSED is on.
        HIPCK(hipMemcpy(d_q, q.data(), q.size() * 4, hipMemcpyHostToDevice));
        HIPCK(hipMemcpy(d_kv, kv.data(), kv.size() * 4, hipMemcpyHostToDevice));
        HIPCK(hipMemcpy(d_g, gate.data(), gate.size() * 4, hipMemcpyHostToDevice));
        HIPCK(hipMemcpy(d_c, cache.data(), cache.size() * 4, hipMemcpyHostToDevice));
        HIPCK(hipMemset(d_out, 0, (size_t) o_dim * 4));

        if (splitk) {
            ling3::mla_block_splitk(sh, sc, S.w, ch, in, (const int *) d_pos,
                                    n_pos, outv,
                                    (float *) S.d_qabs, (float *) S.d_pm,
                                    (float *) S.d_pl, (float *) S.d_pa,
                                    slice, 2, nullptr);
        } else {
            ling3::mla_block(sh, sc, S.w, ch, in, (const int *) d_pos,
                             n_pos, outv, (float *) d_scores, nullptr,
                             (const int *) d_pos);
        }
        HIPCK(hipDeviceSynchronize());

        std::vector<float> out_new((size_t) o_dim), cache_new(cache.size());
        std::vector<std::int8_t> qs_new((size_t) o_dim);
        std::vector<float> ds_new((size_t) o_dim / 32);
        HIPCK(hipMemcpy(out_new.data(), d_out, (size_t) o_dim * 4, hipMemcpyDeviceToHost));
        HIPCK(hipMemcpy(cache_new.data(), d_c, cache.size() * 4, hipMemcpyDeviceToHost));
        CheckTrue("fused path ran (M1)", ling3::mla_pre_fused_launches() > 0);
        CheckTrue("fused path ran (Q2)", ling3::mla_wo_q8_epilogues() > 0);
        CheckTrue("wo Q8_1 pair ready",
                  ling3::mla_wo_q8_qs() != nullptr && ling3::mla_wo_q8_ds() != nullptr);
        if (ling3::mla_wo_q8_qs() != nullptr && ling3::mla_wo_q8_ds() != nullptr) {
            HIPCK(hipMemcpy(qs_new.data(), ling3::mla_wo_q8_qs(), (size_t) o_dim,
                            hipMemcpyDeviceToHost));
            HIPCK(hipMemcpy(ds_new.data(), ling3::mla_wo_q8_ds(),
                            (size_t) (o_dim / 32) * 4, hipMemcpyDeviceToHost));
        }

        CheckMem("out [4096]", out_ref.data(), out_new.data(), (size_t) o_dim * 4);
        CheckMem("cache [n_pos*576]", cache_ref.data(), cache_new.data(),
                 cache.size() * 4);
        CheckMem("wo q8 codes [4096]", qs_ref.data(), qs_new.data(), (size_t) o_dim);
        CheckMem("wo q8 scales [128]", ds_ref.data(), ds_new.data(),
                 (size_t) (o_dim / 32) * 4);

        // ---- the wo handoff itself, at the model's shape 4096 -> 2560:
        // ling3_gemv's old quant+cta64 pair against the fused pair through
        // ling3_gemv_q8_m(m = 1), which is the call Step now makes.
        {
            const int qt = LING3_Q4_0_ROCMFP4_FAST;
            std::vector<float> y_ref((size_t) n_wo), y_new((size_t) n_wo);
            void * d_yr = DevAlloc(y_ref.size() * 4);
            void * d_yn = DevAlloc(y_new.size() * 4);
            CheckTrue("handoff: unfused quant rc",
                      ling3_quant_q8((const float *) d_out, k_wo,
                                     LING3_Q8_SLOT_GENERIC, nullptr) == LING3_OK);
            CheckTrue("handoff: unfused gemv rc",
                      ling3_gemv_fp4_q8(qt, S.d_wo, LING3_Q8_SLOT_GENERIC,
                                        (float *) d_yr, k_wo, n_wo, nullptr) == LING3_OK);
            CheckTrue("handoff: fused gemv rc",
                      ling3_gemv_q8_m(qt, S.d_wo, ling3::mla_wo_q8_qs(),
                                      ling3::mla_wo_q8_ds(), (float *) d_yn,
                                      k_wo, n_wo, 1, nullptr) == LING3_OK);
            HIPCK(hipDeviceSynchronize());
            HIPCK(hipMemcpy(y_ref.data(), d_yr, y_ref.size() * 4, hipMemcpyDeviceToHost));
            HIPCK(hipMemcpy(y_new.data(), d_yn, y_new.size() * 4, hipMemcpyDeviceToHost));
            CheckMem("wo gemv out [2560]", y_ref.data(), y_new.data(),
                     y_ref.size() * 4);
            hipFree(d_yr); hipFree(d_yn);
        }

        hipFree(d_q); hipFree(d_kv); hipFree(d_g); hipFree(d_c);
        hipFree(d_pos); hipFree(d_out); hipFree(d_scores);
    }

    std::printf("\n%s fail=%d\n", g_fail == 0 ? "RESULT PASS" : "RESULT FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
