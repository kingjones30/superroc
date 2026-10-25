#ifndef GUFO_MODELS_QWEN_HIP_FP4_LAUNCH_HPP_
#define GUFO_MODELS_QWEN_HIP_FP4_LAUNCH_HPP_

// Host-side intercept onto the renamed Ling ROCmFP4 GEMV (src/core/rocm_fp4).
// A4's sub16 decoder stays the fallback when a shape is rejected.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/rocm_fp4/gufo_fp4_gemv_q8.hpp"
#include "src/core/rocm_fp4/gufo_fp4_quant.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

[[nodiscard]] inline bool IsRocmFp4(core::GgmlType type) noexcept {
  return type == core::GgmlType::kQ4_0_ROCMFP4 ||
         type == core::GgmlType::kQ4_0_ROCMFP4_FAST;
}

// lqw:capture-warmup. gufo_fp4_quant_q8 grows its slot buffers lazily
// (hipFree/hipMalloc); inside a captured decode graph that poisons the
// capture. WarmupRocmFp4Slots() pre-grows every slot at model load; the
// warmed-width checks below then reject (falling back to the A4 sub16 path)
// rather than allocate mid-capture. Definitions live in fp4_warmup.cpp so the
// mark is process-wide, not one copy per including TU.
[[nodiscard]] std::size_t RocmFp4WarmedK() noexcept;
bool WarmupRocmFp4Slots(std::size_t k, hipStream_t stream);

// lqw:rv27-fix2: clears the warm mark (see fp4_warmup.cpp). Call when a
// load-time warmup step failed, so no FP4 intercept can run afterwards.
void DiscardRocmFp4Warmup() noexcept;

[[nodiscard]] inline bool RocmFp4ShapeOk(std::size_t n,
                                         std::size_t k) noexcept {
  return n > 0 && k > 0 && (k % 32) == 0 && k <= RocmFp4WarmedK() &&
         n <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
         k <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

[[nodiscard]] inline int RocmFp4Qtype(core::GgmlType type) noexcept {
  return static_cast<int>(type);
}

// sq16 SPEC2 (S2-1 test hook): GUFO_FP4_FORCE_REFUSE=1 makes every FP4 GEMV
// intercept decline (decode and verify alike), so the fallback paths can be
// proven bit-identical. The MoE expert FFN is untouched. Off unless set.
// SPEC3 (2): says so once per process on stderr when it takes effect (read
// once, inside this lambda; the hot path still pays one cached-bool branch).
[[nodiscard]] inline bool GufoFp4ForceRefuse() {
  static const bool on = [] {
    const char* env = std::getenv("GUFO_FP4_FORCE_REFUSE");
    const bool refuse = env != nullptr && env[0] == '1';
    if (refuse) {
      std::fprintf(stderr,
                   "GUFO_FP4_FORCE_REFUSE=1: test hook, every dense FP4 "
                   "intercept declines (fp32 fallback, much slower)\n");
    }
    return refuse;
  }();
  return on;
}

// One matrix. Quantises x into the generic slot inside gufo_fp4_gemv_fp4.
inline bool TryLaunchRocmFp4Gemv(core::GgmlType type, const void* weights,
                                 const float* x, float* y, std::size_t n,
                                 std::size_t k, hipStream_t stream) {
  if (GufoFp4ForceRefuse()) {
    return false;
  }
  if (!IsRocmFp4(type) || !RocmFp4ShapeOk(n, k) || weights == nullptr ||
      x == nullptr || y == nullptr) {
    return false;
  }
  return gufo_fp4_gemv_fp4(RocmFp4Qtype(type), weights, x, y,
                           static_cast<int>(k), static_cast<int>(n),
                           stream) == GUFO_FP4_OK;
}

// Several matrices against one activation, quantised once into `slot`.
inline bool TryLaunchRocmFp4GemvSlot(core::GgmlType type, const void* weights,
                                     int slot, float* y, std::size_t n,
                                     std::size_t k, hipStream_t stream) {
  if (GufoFp4ForceRefuse()) {
    return false;
  }
  if (!IsRocmFp4(type) || !RocmFp4ShapeOk(n, k) || weights == nullptr ||
      y == nullptr) {
    return false;
  }
  return gufo_fp4_gemv_fp4_q8(RocmFp4Qtype(type), weights, slot, y,
                              static_cast<int>(k), static_cast<int>(n),
                              stream) == GUFO_FP4_OK;
}

inline bool PrepareRocmFp4Act(const float* x, std::size_t k, int slot,
                              hipStream_t stream) {
  if (GufoFp4ForceRefuse()) {
    return false;
  }
  if (x == nullptr || k == 0 || (k % 32) != 0 || k > RocmFp4WarmedK() ||
      k > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  return gufo_fp4_act_prepare(x, static_cast<int>(k), slot, stream) ==
         GUFO_FP4_OK;
}

// SQ10: gate and up scratch for the fused SwiGLU intercept, one pair per slot
// set. Streams attach 1:1 under the same rule as the Q8 slots (see
// gufo_fp4_gemv_q8.hpp); the intercept names its stream, so two decode streams
// can never share a pair. Pairs are never freed (process lifetime, one box
// job). lqw:capture-warmup: pre-grow every pair at model load
// (RocmFp4SwigluScratchWarm, beside WarmupRocmFp4Slots) so the first captured
// decode step cannot allocate.
struct RocmFp4SwigluSets {
  std::vector<float*> gate;
  std::vector<float*> up;
  std::vector<std::size_t> cap;
  std::mutex mu;
};

inline RocmFp4SwigluSets& RocmFp4SwigluSetsOf() noexcept {
  static RocmFp4SwigluSets sets;
  return sets;
}

inline void RocmFp4SwigluSetsEnsure(int n) {
  RocmFp4SwigluSets& sets = RocmFp4SwigluSetsOf();
  std::lock_guard<std::mutex> lk(sets.mu);
  if (static_cast<int>(sets.gate.size()) < n) {
    sets.gate.resize(static_cast<std::size_t>(n), nullptr);
    sets.up.resize(static_cast<std::size_t>(n), nullptr);
    sets.cap.resize(static_cast<std::size_t>(n), 0);
  }
}

// lqw:rv27-fix2. Readiness probe: reports whether the load-time grow already
// covers n WITHOUT allocating, so the SwiGLU intercept can decline (and the
// A4 fallback serve the step) instead of hipMallocing mid-capture.
inline bool RocmFp4SwigluScratchReady(std::size_t n, hipStream_t stream) {
  const int count = gufo_fp4_slot_sets_count();
  RocmFp4SwigluSetsEnsure(count);
  const int set =
      gufo_fp4_slot_stream_attach_at(stream, __func__, static_cast<long>(n));
  if (set < 0)
    return false;
  RocmFp4SwigluSets& sets = RocmFp4SwigluSetsOf();
  std::lock_guard<std::mutex> lk(sets.mu);
  return set < static_cast<int>(sets.cap.size()) &&
         sets.cap[static_cast<std::size_t>(set)] >= n;
}

inline bool RocmFp4SwigluScratch(std::size_t n, float*& gate, float*& up,
                                 hipStream_t stream) {
  const int count = gufo_fp4_slot_sets_count();
  RocmFp4SwigluSetsEnsure(count);
  const int set =
      gufo_fp4_slot_stream_attach_at(stream, __func__, static_cast<long>(n));
  if (set < 0)
    return false;
  RocmFp4SwigluSets& sets = RocmFp4SwigluSetsOf();
  std::lock_guard<std::mutex> lk(sets.mu);
  if (set >= static_cast<int>(sets.cap.size()))
    return false;
  if (sets.cap[static_cast<std::size_t>(set)] < n) {
    if (hipStreamSynchronize(stream) != hipSuccess)
      return false;
    float* g = nullptr;
    float* u = nullptr;
    if (hipMalloc(&g, n * sizeof(float)) != hipSuccess) {
      return false;
    }
    gufo_fp4_slot_count_alloc();
    if (hipMalloc(&u, n * sizeof(float)) != hipSuccess) {
      (void)hipFree(g);
      return false;
    }
    gufo_fp4_slot_count_alloc();
    if (sets.gate[static_cast<std::size_t>(set)] != nullptr) {
      (void)hipFree(sets.gate[static_cast<std::size_t>(set)]);
    }
    if (sets.up[static_cast<std::size_t>(set)] != nullptr) {
      (void)hipFree(sets.up[static_cast<std::size_t>(set)]);
    }
    sets.gate[static_cast<std::size_t>(set)] = g;
    sets.up[static_cast<std::size_t>(set)] = u;
    sets.cap[static_cast<std::size_t>(set)] = n;
  }
  gate = sets.gate[static_cast<std::size_t>(set)];
  up = sets.up[static_cast<std::size_t>(set)];
  return gate != nullptr && up != nullptr;
}

// Pre-grow every set's pair to n at load. Runs beside WarmupRocmFp4Slots,
// outside any capture; zero is a no-op.
inline bool RocmFp4SwigluScratchWarm(std::size_t n) {
  if (n == 0)
    return true;
  const int count = gufo_fp4_slot_sets_count();
  RocmFp4SwigluSetsEnsure(count);
  RocmFp4SwigluSets& sets = RocmFp4SwigluSetsOf();
  std::lock_guard<std::mutex> lk(sets.mu);
  for (int s = 0; s < count; ++s) {
    if (sets.cap[static_cast<std::size_t>(s)] >= n)
      continue;
    float* g = nullptr;
    float* u = nullptr;
    if (hipMalloc(&g, n * sizeof(float)) != hipSuccess) {
      return false;
    }
    gufo_fp4_slot_count_alloc();
    if (hipMalloc(&u, n * sizeof(float)) != hipSuccess) {
      (void)hipFree(g);
      return false;
    }
    gufo_fp4_slot_count_alloc();
    if (sets.gate[static_cast<std::size_t>(s)] != nullptr) {
      (void)hipFree(sets.gate[static_cast<std::size_t>(s)]);
    }
    if (sets.up[static_cast<std::size_t>(s)] != nullptr) {
      (void)hipFree(sets.up[static_cast<std::size_t>(s)]);
    }
    sets.gate[static_cast<std::size_t>(s)] = g;
    sets.up[static_cast<std::size_t>(s)] = u;
    sets.cap[static_cast<std::size_t>(s)] = n;
  }
  return true;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_FP4_LAUNCH_HPP_
