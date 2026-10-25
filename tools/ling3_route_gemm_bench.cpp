// ling3-route-gemm-bench — where does the 2D register-tiled router GEMM overtake the kernel
// it replaces, as a function of the number of tokens in the chunk?
//
// PHASE9 item 3. The profile says `route` falls 148.5 -> 53.0 ms on a 512-token chunk, but
// the same profile shows a 1-token chunk going 4.1 -> 27.4 ms. That is not noise: the GEMM's
// grid is ceil(E/BM) x ceil(T/BN), so at T=1 it launches 8 CTAs on a 20-CU part while the
// kernel it replaces launches E = 512. This sweeps T to find the crossover, which is the
// number that decides where the dispatch has to switch.
//
// Transcribes k_route_batch_logits (ling3_gemm.hip) for the "ref" arm -- the same
// transcription the correctness gate uses, so the two agree on what "ref" means.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/models/ling3/kernels/ling3_moe_group.hpp"
#include "src/models/ling3/kernels/ling3_quant.hpp"

namespace {

__global__ void k_route_batch_logits_ref(const float * __restrict__ x,
                                         const float * __restrict__ gate,
                                         int k, int n_expert,
                                         float * __restrict__ logits) {
    const int e = blockIdx.x;
    const int t = blockIdx.y;
    const float * row = gate + (size_t) e * (size_t) k;
    const float * xt = x + (size_t) t * (size_t) k;
    float s = 0.f;
    for (int i = (int) threadIdx.x; i < k; i += (int) blockDim.x) {
        s += row[i] * xt[i];
    }
    __shared__ float red[64];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int off = 32; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) {
            red[threadIdx.x] += red[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        logits[(size_t) t * (size_t) n_expert + e] = red[0];
    }
}

// Best of `reps`, in ms. Best-of rather than mean so a single scheduling hiccup on a
// shared box cannot invent or hide a regression.
template <typename F>
float BestMs(F && fn, int reps) {
    float best = 1e30f;
    for (int i = 0; i < reps; ++i) {
        hipEvent_t a, b;
        hipEventCreate(&a);
        hipEventCreate(&b);
        hipDeviceSynchronize();
        hipEventRecord(a);
        fn();
        hipEventRecord(b);
        hipEventSynchronize(b);
        float ms = 0.f;
        hipEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
        hipEventDestroy(a);
        hipEventDestroy(b);
    }
    return best;
}

}  // namespace

int main() {
    const int E = getenv("GEMM_E") ? atoi(getenv("GEMM_E")) : 512;
    const int K = getenv("GEMM_K") ? atoi(getenv("GEMM_K")) : 2048;
    const int reps = getenv("GEMM_REPS") ? atoi(getenv("GEMM_REPS")) : 7;
    const size_t pad = LING3_TAIL_PAD;

    // x is sized for the largest T swept; smaller T use a prefix, so the ref arm reads the
    // same bytes from the same buffers it would in production.
    const int tlist[] = {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};
    const int nt = (int)(sizeof(tlist) / sizeof(tlist[0]));
    const int Tmax = tlist[nt - 1];

    std::vector<float> hx((std::size_t) Tmax * K);
    for (std::size_t i = 0; i < hx.size(); ++i) {
        hx[i] = (float)((int)(i % 17) - 8) * 0.125f;   // real values, no RNG, no fakes
    }
    std::vector<float> hg((std::size_t) E * K);
    for (std::size_t i = 0; i < hg.size(); ++i) {
        hg[i] = (float)((int)(i % 13) - 6) * 0.0625f;
    }
    float * d_x = nullptr; float * d_g = nullptr;
    const std::size_t lg = (std::size_t) Tmax * E * sizeof(float);
    float * d_out = nullptr;
    if (hipMalloc(&d_x, (hx.size() + pad) * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_g, (hg.size() + pad) * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_out, lg + pad) != hipSuccess) {
        std::printf("FAIL: device allocation\n");
        return 1;
    }
    hipMemcpy(d_x, hx.data(), hx.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_g, hg.data(), hg.size() * sizeof(float), hipMemcpyHostToDevice);

    std::printf("== router GEMM crossover sweep  E=%d K=%d reps=%d (best of, ms)\n", E, K, reps);
    std::printf("  %6s %8s %10s %10s %9s %10s %8s\n", "T", "CTAs_ref", "CTAs_gemm",
                "ref_ms", "gemm_ms", "speedup", "winner");
    for (int i = 0; i < nt; ++i) {
        const int T = tlist[i];
        const float ref_ms = BestMs([&] {
            k_route_batch_logits_ref<<<dim3(E, T), 64>>>(d_x, d_g, K, E, d_out);
        }, reps);
        const float gemm_ms = BestMs([&] {
            ling3_route_logits_gemm_launch(d_x, d_g, K, E, T, d_out);
        }, reps);
        const long long ctas_ref = (long long) E * T;
        const long long ctas_gemm =
            (long long)((E + 63) / 64) * (long long)((T + 31) / 32);
        std::printf("  %6d %8lld %10lld %10.4f %10.4f %8.2fx %10s\n", T, ctas_ref, ctas_gemm,
                    ref_ms, gemm_ms, ref_ms / gemm_ms,
                    gemm_ms < ref_ms ? "gemm" : "REF");
    }
    hipFree(d_x); hipFree(d_g); hipFree(d_out);
    std::printf("done\n");
    return 0;
}
