// Flint split-K decode attention (fp32 head-major, head_dim 128, sinks).
#ifndef SUPERROC_MODELS_FLINT_HIP_FLINT_ATTENTION_HPP_
#define SUPERROC_MODELS_FLINT_HIP_FLINT_ATTENTION_HPP_

#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace superroc::hip {

struct FlintLaunchStatus {
  bool ok{false};
  const char* error{nullptr};
};

// FLINT_SPLITK=0 declines the split-K decode attention; unset and every
// other value take it.
inline bool FlintSplitKEnabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("FLINT_SPLITK");
    return env == nullptr || std::strcmp(env, "0") != 0;
  }();
  return enabled;
}

// Flint: split-K flash-decoding attention over the windowed range
// [begin, *d_pos+1). Sinks enter the combine once (not per split).
[[nodiscard]] FlintLaunchStatus LaunchFlintDecodeAttention(
    const float* q, const float* k_cache, const float* v_cache, float* out,
    std::uint32_t layer, const std::uint32_t* d_pos, std::uint32_t window,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, std::uint32_t n_splits,
    float* partial, const float* sinks, hipStream_t stream);

}  // namespace superroc::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // SUPERROC_MODELS_FLINT_HIP_FLINT_ATTENTION_HPP_
