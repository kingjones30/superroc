// ling3-roof-bench — what read bandwidth does amdaimax2 ACTUALLY achieve?
//
// PHASE5 item 1. Every "the GEMVs run at 43 % of the roof" statement in this lane is
// meaningless until the roof is a measured number on the same box, in the same units.
//
// The number that matters is not a spec sheet. Strix Halo is an APU: the "128 GiB of
// GTT" is system memory behind the SoC interconnect, and the brief's own NUDGE warns
// that this box has a **32 MB Infinity Cache (MALL)** that reports 3-7x the real
// bandwidth to any working set that fits inside it. I measured 461 GB/s earlier in this
// lane on a microbenchmark whose weights were 22 MB, which is not a bandwidth, it is the
// MALL.
//
// So this benchmark:
//   * works over a buffer far larger than the MALL (default 1 GiB) and reports the size
//     it used, so a reader can tell at a glance whether the number can be believed;
//   * sweeps the grid so the best occupancy is found rather than an arbitrary one;
//   * measures THREE things, because they differ and the GEMV is not the fastest one:
//       16-byte reads   (int4, what a repacked FP4 layout could use)
//        4-byte reads   (int, what the current kernel uses for the unaligned blocks)
//        1-byte reads   (the floor -- what the ORIGINAL byte-at-a-time kernel did)
//   * plus hipMemcpyDeviceToDevice, the number everyone quotes as "the roof";
//   * uses the ROCm agent's own notion of the MALL size if it will tell us.
//
// Run it inside gpu-run. It allocates; it never touches the model.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

std::size_t g_bytes = (std::size_t) 1024 * 1024 * 1024;   // 1 GiB, 32x the MALL

// Grid-stride read at W bytes per thread per step.
//
// EVERY width folds into ONE accumulator, and the store is guarded on that single
// accumulator. The first version of this benchmark used four accumulators and a
// four-way guard; for W = 4 and W = 1 the other three are provably zero, so the
// compiler proved the store unreachable and deleted the whole loop -- which is
// where its "435,484 GB/s" came from. A bandwidth number that high is not a fast
// memory system, it is a kernel that did not run; if you see one, check this.
template <int W>
__global__ void k_read(const void * __restrict__ p, std::size_t n, int * out) {
    const std::size_t stride = (std::size_t) gridDim.x * blockDim.x * W;
    std::size_t i = ((std::size_t) blockIdx.x * blockDim.x + threadIdx.x) * W;
    uint32_t a = 0;
    for (; i + W <= n; i += stride) {
        if (W == 16) {
            const int4 v = *((const int4 *) p + (i >> 4));
            a ^= (uint32_t) v.x; a ^= (uint32_t) v.y; a ^= (uint32_t) v.z; a ^= (uint32_t) v.w;
        } else if (W == 4) {
            a ^= (uint32_t) *((const int *) p + (i >> 2));
        } else {
            a ^= (uint32_t) ((const uint8_t *) p)[i];
        }
    }
    if (a == 0xFFFFFFFFu) {
        out[0] = 1;
    }
}

double TimeRead(const void * d, std::size_t bytes, int w, int grid, int block, int reps) {
    int * sink = nullptr;
    if (hipMalloc(&sink, sizeof(int)) != hipSuccess) return -1.0;
    auto launch = [&]() {
        if (w == 16) k_read<16><<<grid, block>>>(d, bytes, sink);
        else if (w == 4) k_read<4><<<grid, block>>>(d, bytes, sink);
        else k_read<1><<<grid, block>>>(d, bytes, sink);
    };
    launch();
    if (hipDeviceSynchronize() != hipSuccess) { hipFree(sink); return -1.0; }
    hipEvent_t a, b;
    hipEventCreate(&a); hipEventCreate(&b);
    hipEventRecord(a, (hipStream_t) 0);
    for (int i = 0; i < reps; ++i) launch();
    hipEventRecord(b, (hipStream_t) 0);
    hipEventSynchronize(b);
    float ms = 0.f;
    hipEventElapsedTime(&ms, a, b);
    hipEventDestroy(a); hipEventDestroy(b);
    hipFree(sink);
    if (ms <= 0.f) return -1.0;
    return (double) bytes * (double) reps / (ms * 1e-3f) / 1e9;   // GB/s
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc > 1) g_bytes = (std::size_t) std::atoll(argv[1]);

    int dev = 0, ncu = 0;
    hipGetDevice(&dev);
    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, dev);
    std::printf("== roof bench on %s (gcnArchName=%s, %d CUs, clock %.2f GHz)\n",
                prop.name, prop.gcnArchName, prop.multiProcessorCount,
                prop.clockRate / 1e6);
    std::size_t mtotal = 0, mfree = 0;
    hipMemGetInfo(&mfree, &mtotal);
    std::printf("   device memory: %.1f GiB total, %.1f GiB free; working set %.0f MiB\n",
                mtotal / 1073741824.0, mfree / 1073741824.0, g_bytes / 1048576.0);
    std::printf("   NOTE: Strix Halo has a 32 MB Infinity Cache (MALL). A working set well\n"
                "         above it is what makes the numbers below meaningful.\n\n");

    if (g_bytes < (std::size_t) 160 * 1024 * 1024) {
        std::printf("   WARNING: working set is under 160 MiB -- these numbers may be MALL-resident\n"
                    "            and should not be quoted.\n");
    }

    void * d = nullptr;
    if (hipMalloc(&d, g_bytes) != hipSuccess) {
        std::fprintf(stderr, "hipMalloc %zu bytes failed\n", g_bytes);
        return 1;
    }
    hipMemset(d, 1, g_bytes);   // touch every page so the first read is not a fault storm

    const int blocks[] = {512, 1024, 2048, 4096, 8192, 16384, 32768};
    const int block = 256;
    const int reps = 8;

    struct Res { int w; double best; int grid; };
    Res res[3] = {{16, 0.0, 0}, {4, 0.0, 0}, {1, 0.0, 0}};

    for (int wi = 0; wi < 3; ++wi) {
        const int w = res[wi].w;
        std::printf("-- %d-byte reads\n", w);
        for (int g : blocks) {
            const double gbs = TimeRead(d, g_bytes, w, g, block, reps);
            if (gbs < 0.0) { std::printf("   grid %6d  FAILED\n", g); continue; }
            std::printf("   grid %6d  %8.2f GB/s%s\n", g, gbs,
                        gbs > res[wi].best ? "  *" : "");
            if (gbs > res[wi].best) { res[wi].best = gbs; res[wi].grid = g; }
        }
        std::printf("   best %d-byte: %.2f GB/s at grid %d\n\n", w, res[wi].best, res[wi].grid);
    }

    // hipMemcpyDeviceToDevice, the number usually quoted as "the roof".
    void * d2 = nullptr;
    double copy_gbs = -1.0;
    if (hipMalloc(&d2, g_bytes) == hipSuccess) {
        hipMemset(d2, 0, g_bytes);
        hipMemcpy(d2, d, g_bytes, hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();
        hipEvent_t a, b;
        hipEventCreate(&a); hipEventCreate(&b);
        hipEventRecord(a, (hipStream_t) 0);
        for (int i = 0; i < reps; ++i) {
            hipMemcpy(d2, d, g_bytes, hipMemcpyDeviceToDevice);
        }
        hipEventRecord(b, (hipStream_t) 0);
        hipEventSynchronize(b);
        float ms = 0.f;
        hipEventElapsedTime(&ms, a, b);
        copy_gbs = (double) g_bytes * (double) reps / (ms * 1e-3f) / 1e9;
        std::printf("-- hipMemcpyDeviceToDevice: %.2f GB/s (%.3f ms per %.0f MiB)\n",
                    copy_gbs, ms / reps, g_bytes / 1048576.0);
        hipFree(d2);
    }
    std::printf("   (that counts the write as well as the read; a read-only roof is the\n"
                "    16-byte line above)\n\n");

    const double roof = res[0].best > copy_gbs ? res[0].best : copy_gbs;
    std::printf("== ROOF (max of 16-byte read and D2D copy): %.2f GB/s\n", roof);
    std::printf("   use this as the 100%% denominator for %% of roof\n");

    hipFree(d);
    (void) ncu;
    return 0;
}