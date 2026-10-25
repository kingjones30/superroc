#pragma once

// FLASHNEXT: host-side kernel-launch counter for LFN_DECODE_PROFILE.
// Armed only while a profiled decode token is launching. Replay of a
// captured graph does not execute this (the seat path leaves it disarmed).

#include <atomic>

namespace gufo::models::qwen38_flash_next::rocm {

inline std::atomic<int>& LfnLaunchArmed() {
  static std::atomic<int> v{0};
  return v;
}

inline std::atomic<int>& LfnLaunchTotal() {
  static std::atomic<int> v{0};
  return v;
}

inline void LfnNoteLaunch() {
  if (LfnLaunchArmed().load(std::memory_order_relaxed) != 0) {
    LfnLaunchTotal().fetch_add(1, std::memory_order_relaxed);
  }
}

inline void LfnLaunchReset() {
  LfnLaunchTotal().store(0, std::memory_order_relaxed);
}

inline void LfnLaunchArm(bool on) {
  LfnLaunchArmed().store(on ? 1 : 0, std::memory_order_relaxed);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
