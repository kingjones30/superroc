#ifndef SUPERROC_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_
#define SUPERROC_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_

#include <cstddef>
#include <stdexcept>
#include <string>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/managed_alloc.hpp"

namespace superroc::hip::detail {

[[nodiscard]] inline void* AllocateDevice(std::size_t bytes) {
  void* pointer = nullptr;
  const auto error = superroc::hip::AllocateDevice(&pointer, bytes);
  if (error != hipSuccess) {
    throw std::runtime_error(std::string("MTP allocation failed: ") +
                             hipGetErrorString(error));
  }
  return pointer;
}

}  // namespace superroc::hip::detail
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // SUPERROC_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_
