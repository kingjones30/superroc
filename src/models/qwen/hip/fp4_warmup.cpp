// lqw:fp4-warmup. The copied Ling FP4 GEMV slots (superroc_fp4_quant_q8) grow
// lazily via hipFree/hipMalloc on first use per width, and the fused-SwiGLU
// scratch in fp4_launch.hpp does the same. The first FP4 decode step runs
// inside a captured hipGraph, where an allocation poisons the capture (fault
// 2026-10-04: "ling3 q8 q8 quant: operation failed due to a previous error
// during capture"). WarmupRocmFp4Slots() pre-grows every slot at model load,
// outside any capture; the inline guards in fp4_launch.hpp then route any
// width beyond the warmed size to the A4 sub16 fallback instead of letting an
// allocation happen mid-capture. Single TU on purpose: the warm mark is read
// from every kernel TU through these definitions.
#include <climits>

#include "src/models/qwen/hip/fp4_launch.hpp"

namespace superroc::hip {
namespace {
struct WarmMark {
  std::size_t k = 0;
  int sets = 0;
};
WarmMark g_warm;
}  // namespace

std::size_t RocmFp4WarmedK() noexcept {
  return g_warm.k;
}

bool WarmupRocmFp4Slots(std::size_t k, hipStream_t stream) {
  if (k == 0 || (k % 32) != 0 || k > static_cast<std::size_t>(INT_MAX)) {
    return false;
  }
  // SQ19: the mark covers a width only for the sets that existed when it was
  // set. A grown table (another executor attached, or configure() pre-sized)
  // re-warms so the new sets' slots exist before any captured step; without
  // this the first captured decode on a new set hipMallocs mid-capture.
  if (g_warm.k >= k && g_warm.sets >= superroc_fp4_slot_sets_count()) {
    return true;
  }
  // SQ10: pre-grow every set's five Q8 slots (the verify pair rides inside,
  // as before). The MoE scratch rides in model_loader beside this call.
  const bool ok =
      superroc_fp4_slots_warm_q8(static_cast<int>(k), stream) == SUPERROC_FP4_OK;
  if (ok) {
    if (k > g_warm.k) {
      g_warm.k = k;
    }
    g_warm.sets = superroc_fp4_slot_sets_count();
  }
  return ok;
}

// lqw:rv27-fix2. Drops the warm mark after a failed load-time SwiGLU scratch
// grow (see QwenGpuExecutor::Create): every RocmFp4ShapeOk then fails, the
// A4 sub16 fallback serves the model, and no FP4 intercept - and therefore
// no lazy allocation - can run inside a captured decode graph.
void DiscardRocmFp4Warmup() noexcept {
  g_warm.k = 0;
}

}  // namespace superroc::hip
