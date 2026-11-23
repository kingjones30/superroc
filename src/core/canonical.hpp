// SUPERROC_CANONICAL_CACHE=1: a reply must not depend on how the conversation
// before it reached the model: turn by turn through the prompt cache, or in
// one request. For runners with prompt snapshots the server then keeps only
// prompt-pass state after a request; dense Qwen prompt passes take one
// kernel route for every pass size; Flash-Next prompts are cut only on
// positions where a cut does not change the state.
#pragma once

#include <cstring>

#include "src/core/env.hpp"

namespace superroc::core {

// Read once from the environment.
inline bool CanonicalCacheOn() {
  static const bool on = [] {
    const char* env = GetEnv("SUPERROC_CANONICAL_CACHE");
    return env != nullptr && std::strcmp(env, "1") == 0;
  }();
  return on;
}

}  // namespace superroc::core
