// Defaults for the GPU runtime's own environment switches, set before the
// runtime starts.
//
// On an integrated Radeon the runtime registers ordinary process memory with
// the GPU driver in two places: its own host buffers (anonymous memory, by
// default), and any buffer over 128 MiB that is copied to or from the device
// (registered in 32 MiB windows, a few of which it keeps for reuse). When the
// kernel later moves or unmaps a page in such a range (memory compaction,
// swap, a free), the driver stops every GPU queue of the process until the
// range is mapped again. A load or a first request then pauses with no GPU
// kernel running: up to 0.9 s per load and 0.5 s on a first request were
// measured on a Radeon 8060S with the page cache full.
//
// HSA_USERPTR_FOR_PAGED_MEM=0 makes the runtime take its host buffers from
// the driver, which the kernel does not move. A GPU_PINNED_MIN_XFER_SIZE (in
// MiB) no copy reaches makes it copy large buffers through its staging
// buffer and register nothing.
#ifndef SUPERROC_CORE_GPU_RUNTIME_ENVIRONMENT_HPP_
#define SUPERROC_CORE_GPU_RUNTIME_ENVIRONMENT_HPP_

#include <cstdlib>

namespace superroc::core {

inline constexpr const char* kRuntimeHostBuffersKey =
    "HSA_USERPTR_FOR_PAGED_MEM";
inline constexpr const char* kRuntimeHostBuffersFromDriver = "0";
inline constexpr const char* kRuntimePinnedCopyKey = "GPU_PINNED_MIN_XFER_SIZE";
// 1 TiB, in MiB.
inline constexpr const char* kRuntimePinnedCopyNever = "1048576";

// Call before the first GPU runtime call. A value already in the environment
// is kept, so either switch can be set back from outside
// (HSA_USERPTR_FOR_PAGED_MEM=1, GPU_PINNED_MIN_XFER_SIZE=128).
inline void ApplyGpuRuntimeDefaults() {
  (void)::setenv(kRuntimeHostBuffersKey, kRuntimeHostBuffersFromDriver, 0);
  (void)::setenv(kRuntimePinnedCopyKey, kRuntimePinnedCopyNever, 0);
}

}  // namespace superroc::core
#endif
