#ifndef SUPERROC_MODELS_QWEN_HIP_MTP_PRIME_ROUTE_HPP_
#define SUPERROC_MODELS_QWEN_HIP_MTP_PRIME_ROUTE_HPP_

#include <cstddef>

#include "src/core/gguf_reader.hpp"

namespace superroc::hip {

/// The weight type and GEMM shape rule for running an MTP prime projection
/// through the target's tiled prefill route
/// (LaunchBatchedQuantGEMMPreQuantized) instead of the eight-row loop. Q8_0
/// only needs more rows than the small-batch exact kernel handles; the FP4
/// formats need a wave-tile shape the prefill kernel is tuned for. Everything
/// else has no tiled route.
[[nodiscard]] constexpr bool MtpPrimeTiledShapeOk(
    core::GgmlType type, std::size_t batch, std::size_t rows_out,
    std::size_t cols_in) noexcept {
  const bool fp4 = type == core::GgmlType::kQ4_0_ROCMFP4 ||
                   type == core::GgmlType::kQ4_0_ROCMFP4_FAST;
  const bool shape_ok = type == core::GgmlType::kQ8_0
                            ? batch > 16
                            : fp4 && batch >= 96 && rows_out >= 1024 &&
                                  cols_in >= 1024 && cols_in % 256 == 0;
  return shape_ok && cols_in % 32 == 0;
}

}  // namespace superroc::hip

#endif  // SUPERROC_MODELS_QWEN_HIP_MTP_PRIME_ROUTE_HPP_
