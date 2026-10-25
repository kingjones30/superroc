#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_VERIFY_DEBUG_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_VERIFY_DEBUG_HPP_

// sq16: environment-gated per-stage snapshots that compare the single-row
// decode path against the multi-row verification path on the same tokens.
// Off unless GUFO_VERIFY_DEBUG=1; when off, SnapVerifyStage is one static
// branch and no device traffic.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace gufo::hip::detail {

inline constexpr std::uint32_t kVerifyDebugNoLayer = 0xFFFFFFFFu;

struct VerifyDebugStage {
  std::string tag;
  std::uint32_t layer{kVerifyDebugNoLayer};
  std::size_t floats_per_row{0};
  std::size_t row_count{0};
  std::vector<float> data;  // row-major, row_count * floats_per_row
};

[[nodiscard]] inline bool VerifyDebugEnabled() {
  static const bool on = [] {
    const char* env = std::getenv("GUFO_VERIFY_DEBUG");
    return env != nullptr && env[0] == '1';
  }();
  return on;
}

// sq16 SPEC2 (S2-3): the snapshot buffer is bounded (default 512 MiB per
// arena; GUFO_VERIFY_DEBUG_MAX_BYTES overrides for tests). Recording stops
// with one warning when the next stage would exceed it.
[[nodiscard]] inline std::size_t VerifyDebugMaxBytes() {
  static const std::size_t cap = [] {
    constexpr std::size_t kDefault = 512ULL << 20;
    const char* env = std::getenv("GUFO_VERIFY_DEBUG_MAX_BYTES");
    if (env == nullptr || env[0] == '\0') {
      return kDefault;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end == env || parsed == 0) {
      return kDefault;
    }
    return static_cast<std::size_t>(parsed);
  }();
  return cap;
}

}  // namespace gufo::hip::detail

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_VERIFY_DEBUG_HPP_
