// ling3_batch_test.cpp — LS3's model-level gate for batched prefill.
// Loads the real Ling-3.0-flash GGUF on the ROCm backend and compares
// ForwardBatch against the token-at-a-time Step on the same weights:
//   * a length-1 batch (T=1) vs Step: max |logit diff|  -- separates the
//     batched kernels from the wiring, the experiment H1 never ran;
//   * per-position teacher-forced argmax agreement (>= 99% is the gate),
//     using the all-position logits mode (LING3_BATCH_ALL_LOGITS=1);
//   * final-position argmax + top-5 set equality on prompts of several
//     lengths, including one that crosses the 512-token chunk boundary;
//   * an 8-token greedy continuation off the batched state (the mixed
//     batched-prefill -> scalar-decode path serve actually runs).
// Run inside gpu-run with the model path as argv[1] (or $LING3_GGUF).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "src/models/ling3/engine.hpp"

namespace ling3 {
void kda_decode_test_pick(int pick, const char * variant);
}

// Jay 2026-10-03: FNV-1a-64 over raw float bytes, folded 8 bytes at a time (bit-identity fingerprint).
static unsigned long long JayHash(unsigned long long h, const float* p, std::size_t n_floats) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(p);
    const std::size_t nb = n_floats * sizeof(float);
    std::size_t i = 0;
    for (; i + 8 <= nb; i += 8) {
        unsigned long long w;
        std::memcpy(&w, b + i, 8);
        h ^= w;
        h *= 1099511628211ULL;
    }
    for (; i < nb; ++i) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

namespace {

namespace l3 = ::superroc::models::ling3;
using l3::Model;
using l3::ModelOptions;
using l3::BackendKind;
using SessionMode = ::superroc::core::SessionMode;

int ArgmaxOf(const float* v, std::size_t n) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (v[i] > v[best]) best = i;
    }
    return (int) best;
}

void Top5Of(const float* v, std::size_t n, int* out_idx) {
    float val[5];
    for (int s = 0; s < 5; ++s) {
        val[s] = -1e30f;
        out_idx[s] = -1;
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (v[i] > val[4]) {
            int p = 4;
            while (p > 0 && val[p - 1] < v[i]) {
                val[p] = val[p - 1];
                out_idx[p] = out_idx[p - 1];
                --p;
            }
            val[p] = v[i];
            out_idx[p] = (int) i;
        }
    }
}

double NllOf(const float* v, std::size_t n, int target) {
    float mx = v[0];
    for (std::size_t i = 1; i < n; ++i) mx = std::max(mx, v[i]);
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) sum += std::exp((double) v[i] - (double) mx);
    return std::log(sum) + (double) mx - (double) v[target];
}

template <class ModelPtr>
int KdaAbWalk(ModelPtr& model, std::size_t V, const std::vector<std::int32_t>& ids) {
    const char* path = std::getenv("LING3_BATCH_KDA_AB");
    std::string err;
    void* s_t32 = model->NewState(&err);
    void* s_red = model->NewState(&err);
    if (s_t32 == nullptr || s_red == nullptr) {
        std::printf("AB STATE FAIL: %s\n", err.c_str());
        if (s_t32 != nullptr) model->DestroyState(s_t32);
        if (s_red != nullptr) model->DestroyState(s_red);
        return 1;
    }
    std::FILE* fp = std::fopen(path, "w");
    if (fp == nullptr) {
        std::printf("AB FAIL cannot open %s\n", path);
        model->DestroyState(s_t32);
        model->DestroyState(s_red);
        return 1;
    }
    std::fprintf(fp, "step max_abs argmax_t32 argmax_red\n");
    std::vector<float> a(V), b(V);
    double max_of = 0;
    std::size_t at = 0;
    int first = -1;
    int n_diff = 0;
    double sum = 0;
    const std::size_t n = ids.size();
    for (std::size_t i = 0; i < n; ++i) {
        ling3::kda_decode_test_pick(2, "t32");
        if (!model->Step(ids[i], s_t32, std::span<float>(a), &err)) {
            std::printf("AB FAIL t32 step %zu: %s\n", i, err.c_str());
            std::fclose(fp);
            model->DestroyState(s_t32);
            model->DestroyState(s_red);
            return 1;
        }
        ling3::kda_decode_test_pick(2, "red");
        if (!model->Step(ids[i], s_red, std::span<float>(b), &err)) {
            std::printf("AB FAIL red step %zu: %s\n", i, err.c_str());
            std::fclose(fp);
            model->DestroyState(s_t32);
            model->DestroyState(s_red);
            return 1;
        }
        double mx = 0;
        for (std::size_t j = 0; j < V; ++j) {
            const double d = std::fabs((double) a[j] - (double) b[j]);
            if (d > mx) mx = d;
        }
        const int am_a = ArgmaxOf(a.data(), V);
        const int am_b = ArgmaxOf(b.data(), V);
        if (am_a != am_b) {
            if (first < 0) first = (int) i;
            n_diff += 1;
        }
        if (mx > max_of) {
            max_of = mx;
            at = i;
        }
        sum += mx;
        std::fprintf(fp, "%zu %.9e %d %d\n", i, mx, am_a, am_b);
        if ((i + 1) % 512 == 0 || i + 1 == n) {
            std::printf("AB step %zu/%zu max_so_far=%.4e first_argmax_diff=%d\n",
                        i + 1, n, max_of, first);
            std::fflush(stdout);
            std::fflush(fp);
        }
    }
    std::fclose(fp);
    model->DestroyState(s_t32);
    model->DestroyState(s_red);
    ling3::kda_decode_test_pick(0, nullptr);
    std::printf("LOGITDIFF n=%zu first_argmax_diff=%d n_argmax_diff=%d max_abs=%.9e at_step=%zu mean_max_abs=%.9e file=%s\n",
                n, first, n_diff, max_of, at, n ? sum / (double) n : 0.0, path);
    return 0;
}

int g_fail = 0;

}  // namespace

int main(int argc, char** argv) {
    std::setbuf(stdout, nullptr);   // abort() must not eat the progress log
    std::string path = argc > 1 ? argv[1] : "";
    if (path.empty()) {
        const char* e = std::getenv("LING3_GGUF");
        path = e != nullptr ? e
                            : "/path/to/models/swap/ling-raul/"
                              "Ling-3.0-flash-ROCmFP4-STRIX-MTP-Q4_0-00001-of-00002.gguf";
    }
    ModelOptions opt;
    opt.max_context = 4096;
    if (const char* mc = std::getenv("LING3_BATCH_TEST_MAX_TOKENS"); mc != nullptr && *mc != 0) {
        using T = decltype(opt.max_context);
        const T want = static_cast<T>(std::strtoull(mc, nullptr, 10) + 64);
        if (want > opt.max_context) opt.max_context = want;
    }
    opt.backend = BackendKind::kRocm;
    opt.prefill_batch = 1;
    opt.stream_experts = false;
    std::string err;
    std::printf("loading %s ...\n", path.c_str());
    std::fflush(stdout);
    auto model = Model::Load(path, opt, &err);
    if (!model) {
        std::printf("LOAD FAIL: %s\n", err.c_str());
        return 2;
    }
    if (!model->PrefersBatchPrefill()) {
        std::printf("LOAD FAIL: batched prefill not available on this load\n");
        return 2;
    }
    const std::size_t V = model->config().vocab_size;
    const std::uint32_t chunk = l3::PrefillBatchChunk();
    void* state = model->NewState(&err);
    if (state == nullptr) {
        std::printf("STATE FAIL: %s\n", err.c_str());
        return 2;
    }
    std::printf("loaded. vocab=%zu chunk=%u\n", V, chunk);

    // Prompts: several lengths; the last two cross the 512-token boundary so
    // chunked prefill exercises more than one ForwardBatch call.
    const auto rep = [](const char* p, int k) {
        std::string s;
        for (int i = 0; i < k; ++i) s += p;
        return s;
    };
    std::vector<std::string> prompts = {
        "Hello",
        "The capital of France is",
        "def quicksort(items):\n    if len(items) <= 1:\n        return items\n"
        "    pivot = items[0]\n    left = [x for x in items[1:] if x < pivot]\n"
        "    right = [x for x in items[1:] if x >= pivot]\n"
        "    return quicksort(left) + [pivot] + quicksort(right)\n\n"
        "Explain this code, then write a unit test for it.",
        rep("Ling-3.0-flash is a hybrid recurrent model. ", 22),
        rep("The history of computing is long. ", 160),   // ~1.1k tokens: crosses the 512 chunk boundary
    };
    // Jay 2026-10-01: LING3_BATCH_TEST_PROMPT_FILE runs only the prompt read from that file (a long real document
    // exercises many 512-token chunks); LING3_BATCH_TEST_MAX_TOKENS raises the 2000-token cap.
    if (const char* pf = std::getenv("LING3_BATCH_TEST_PROMPT_FILE"); pf != nullptr && *pf != 0) {
        std::FILE* f = std::fopen(pf, "rb");
        if (f == nullptr) {
            std::printf("FAIL cannot open %s\n", pf);
            return 1;
        }
        std::string text;
        char buf[65536];
        for (std::size_t r; (r = std::fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, r);
        std::fclose(f);
        prompts = {text};
        std::printf("prompt file %s: %zu bytes\n", pf, text.size());
    }
    std::size_t max_tokens = 2000;
    if (const char* mt = std::getenv("LING3_BATCH_TEST_MAX_TOKENS"); mt != nullptr && *mt != 0) {
        max_tokens = static_cast<std::size_t>(std::strtoull(mt, nullptr, 10));
    }

    if (const char* ab = std::getenv("LING3_BATCH_KDA_AB"); ab != nullptr && ab[0] != '\0') {
        if (prompts.empty()) {
            std::printf("AB FAIL no prompt\n");
            model->DestroyState(state);
            return 1;
        }
        std::vector<std::int32_t> ids = model->tokenizer().Encode(prompts[0]);
        if (ids.size() > max_tokens) ids.resize(max_tokens);
        std::printf("AB prompt_tokens=%zu vocab=%zu file=%s\n", ids.size(), V, ab);
        model->DestroyState(state);
        return KdaAbWalk(model, V, ids);
    }

    std::vector<float> logits(V), all;
    int prompt_idx = 0;
    for (const std::string& text : prompts) {
        ++prompt_idx;
        std::vector<std::int32_t> ids = model->tokenizer().Encode(text);
        if (ids.size() > max_tokens) ids.resize(max_tokens);
        const std::size_t n = ids.size();
        if (n == 0) continue;

        // ---- scalar reference: one Step per token, argmax after each ----
        model->ResetState(state);
        std::printf("PROMPT %d len=%zu: scalar run\n", prompt_idx, n);
        std::vector<int> ref_argmax;
        unsigned long long scalar_hash = 1469598103934665603ULL;
        std::vector<double> ref_nll;
        ref_argmax.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (!model->Step(ids[i], state, std::span<float>(logits), &err)) {
                std::printf("PROMPT %d FAIL scalar step %zu: %s\n", prompt_idx, i,
                            err.c_str());
                ++g_fail;
                break;
            }
            ref_argmax.push_back(ArgmaxOf(logits.data(), V));
            scalar_hash = JayHash(scalar_hash, logits.data(), V);
            if (i + 1 < n) ref_nll.push_back(NllOf(logits.data(), V, ids[i + 1]));
        }
        std::vector<float> ref_last(logits);
        int ref_final = ref_argmax.empty() ? -1 : ref_argmax.back();
        int ref_top5[5];
        Top5Of(ref_last.data(), V, ref_top5);

        // ---- batched prefill, all-position logits ----
        model->ResetState(state);
        std::printf("PROMPT %d: batched run\n", prompt_idx);
        all.assign(n * V, 0.f);
        bool ok = true;
        for (std::size_t off = 0; off < n && ok; off += chunk) {
            const std::size_t take = std::min<std::size_t>(chunk, n - off);
            if (!model->StepBatch(ids.data() + off, (int) take, state,
                                  std::span<float>(
                                      all.data() + static_cast<std::ptrdiff_t>(off * V),
                                      take * V),
                                  std::span<float>(), std::span<float>(), &err)) {
                std::printf("PROMPT %d FAIL batched chunk at %zu: %s\n", prompt_idx,
                            off, err.c_str());
                ++g_fail;
                ok = false;
                break;
            }
        }
        if (!ok) continue;
        std::printf("HASH prompt=%d n=%zu scalar_logits=%016llx batched_logits=%016llx\n", prompt_idx, n,
                    scalar_hash, JayHash(1469598103934665603ULL, all.data(), n * V));

        // gates: per-position argmax agreement, final top-5, final max diff
        std::size_t agree = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (ArgmaxOf(all.data() + i * V, V) == static_cast<std::size_t>(ref_argmax[i])) {
                ++agree;
            }
        }
        if (std::getenv("LING3_BATCH_TEST_PER_CHUNK") != nullptr) {
            double tot_s = 0, tot_b = 0;
            std::size_t tot_n = 0;
            for (std::size_t c0 = 0; c0 < n; c0 += chunk) {
                const std::size_t c1 = std::min<std::size_t>(c0 + chunk, n);
                std::size_t ag = 0, m = 0;
                double ns = 0, nb = 0;
                for (std::size_t i = c0; i < c1; ++i) {
                    if (ArgmaxOf(all.data() + i * V, V) == ref_argmax[i]) ++ag;
                    if (i + 1 < n && i < ref_nll.size()) {
                        ns += ref_nll[i];
                        nb += NllOf(all.data() + i * V, V, ids[i + 1]);
                        ++m;
                    }
                }
                tot_s += ns;
                tot_b += nb;
                tot_n += m;
                std::printf("CHUNK %zu pos %zu-%zu agree=%.2f%% nll scalar=%.4f batched=%.4f\n", c0 / chunk, c0, c1 - 1,
                            100.0 * (double) ag / (double) (c1 - c0), m ? ns / (double) m : 0.0, m ? nb / (double) m : 0.0);
            }
            if (tot_n > 0) {
                std::printf("PPL scalar=%.4f batched=%.4f over %zu predictions\n", std::exp(tot_s / (double) tot_n),
                            std::exp(tot_b / (double) tot_n), tot_n);
            }
        }
        double maxd = 0;
        for (std::size_t i = 0; i < V; ++i) {
            const double d = std::fabs((double) all[(n - 1) * V + i] - (double) ref_last[i]);
            if (d > maxd) maxd = d;
        }
        int bat_top5[5];
        Top5Of(all.data() + (n - 1) * V, V, bat_top5);
        bool top5_same = true;
        for (int s = 0; s < 5; ++s) {
            bool found = false;
            for (int t = 0; t < 5; ++t) {
                if (bat_top5[t] == ref_top5[s]) found = true;
            }
            if (!found) top5_same = false;
        }
        const bool argmax_same = ArgmaxOf(all.data() + (n - 1) * V, V) ==
                                 static_cast<std::size_t>(ref_final);
        const double agree_pct = 100.0 * (double) agree / (double) n;

        // ---- greedy continuation off the batched state (mixed path) ----
        std::vector<std::int32_t> cont;
        std::int32_t next = static_cast<std::int32_t>(
            ArgmaxOf(all.data() + (n - 1) * V, V));
        cont.push_back(next);
        for (int g = 0; g < 7; ++g) {
            if (!model->Step(next, state, std::span<float>(logits), &err)) {
                std::printf("PROMPT %d FAIL greedy step: %s\n", prompt_idx, err.c_str());
                ++g_fail;
                break;
            }
            next = static_cast<std::int32_t>(ArgmaxOf(logits.data(), V));
            cont.push_back(next);
        }
        std::string cont_text = model->tokenizer().Decode(cont);

        const bool gate = argmax_same && top5_same && agree_pct >= 99.0;
        if (!gate) ++g_fail;
        std::printf(
            "PROMPT %d len=%zu chunks=%zu %s agree=%zu/%zu (%.2f%%) final_argmax=%s "
            "top5=%s max_last_logit_diff=%.4e\n",
            prompt_idx, n, (n + chunk - 1) / chunk, gate ? "PASS" : "FAIL", agree, n,
            agree_pct, argmax_same ? "same" : "DIFF", top5_same ? "same" : "DIFF", maxd);
        std::printf("  greedy(batched): %s\n", cont_text.substr(0, 120).c_str());
        std::fflush(stdout);
    }

    // ---- T=1: one token through StepBatch vs Step, same fresh state ----
    {
        model->ResetState(state);
        const std::vector<std::int32_t> one = model->tokenizer().Encode("The capital of France is");
        for (std::size_t i = 0; i + 1 < one.size(); ++i) {
            if (!model->Step(one[i], state, std::span<float>(logits), &err)) {
                std::printf("T1 FAIL prefix: %s\n", err.c_str());
                ++g_fail;
                break;
            }
        }
        const std::int32_t last = one.back();
        if (!model->Step(last, state, std::span<float>(logits), &err)) {
            std::printf("T1 FAIL scalar: %s\n", err.c_str());
            ++g_fail;
        } else {
            std::vector<float> scalar(logits);
            model->ResetState(state);
            for (std::size_t i = 0; i + 1 < one.size(); ++i) {
                if (!model->Step(one[i], state, std::span<float>(logits), &err)) {
                    std::printf("T1 FAIL prefix2: %s\n", err.c_str());
                    ++g_fail;
                    break;
                }
            }
            if (!model->StepBatch(&last, 1, state, std::span<float>(logits), std::span<float>(), std::span<float>(), &err)) {
                std::printf("T1 FAIL batched: %s\n", err.c_str());
                ++g_fail;
            } else {
                double maxd = 0;
                std::size_t at = 0;
                for (std::size_t i = 0; i < V; ++i) {
                    const double d = std::fabs((double) logits[i] - (double) scalar[i]);
                    if (d > maxd) {
                        maxd = d;
                        at = i;
                    }
                }
                const int am_s = ArgmaxOf(scalar.data(), V);
                const int am_b = ArgmaxOf(logits.data(), V);
                std::printf("T1 max_logit_diff=%.4e (at %zu) argmax scalar=%d batched=%d %s\n",
                            maxd, at, am_s, am_b, am_s == am_b ? "same" : "DIFF");
                if (am_s != am_b) ++g_fail;
            }
        }
    }

    model->DestroyState(state);
    std::printf("%s fail=%d\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}
