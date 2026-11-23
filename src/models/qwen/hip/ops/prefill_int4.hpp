#ifndef SUPERROC_MODELS_QWEN_HIP_OPS_PREFILL_INT4_HPP_
#define SUPERROC_MODELS_QWEN_HIP_OPS_PREFILL_INT4_HPP_
#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
namespace superroc::hip {
// Prefill FFN gate and up, attn_qkv and attn_gate through the 4-bit matrix
// unit. Weights are packed int4 with one scale per row, read from a route
// weights file. Inputs are packed int4 with one scale per token and a power of
// two per group of columns; columns are sorted so that a group is a run of K
// steps, and the accumulators shift left between runs.
constexpr std::size_t kInt4MaxShifts = 16;
constexpr int kInt4BlockSteps = 4;
constexpr std::uint32_t kInt4NoColumn = 0xFFFFFFFFU;
// A column with this bit set is the second, finer pass over its weights.
constexpr std::uint32_t kInt4FinePass = 0x80000000U;
// A column with this bit set is the second, finer pass over its input.
constexpr std::uint32_t kInt4FineInput = 0x40000000U;
constexpr std::uint32_t kInt4ColumnMask = 0x3FFFFFFFU;
struct Int4Shifts {
  std::int32_t count;
  std::int32_t
      at[kInt4MaxShifts];  // K step (16 columns each) that starts a run
  std::int32_t bits[kInt4MaxShifts];  // left shift applied before that step
};
// `order[position]` names the source column of every packed column, marked with
// kInt4FinePass or kInt4FineInput on its finer passes, or kInt4NoColumn for
// padding. `k_in` is the source width, `k_out` the packed one.
// `inverse[position]` is one over the power of two of that packed column.
void LaunchInt4PackHalf(const void* x, std::uint8_t* out, float* token_scale,
                        const std::uint32_t* order, const float* inverse,
                        float base, std::size_t batch, std::size_t k_in,
                        std::size_t k_out, float clip, hipStream_t stream);
// One matrix; y holds floats, token major.
void LaunchInt4GemmStore(const std::uint8_t* w, const std::uint8_t* x,
                         const float* row_scale, const float* token_scale,
                         const Int4Shifts& shifts, float* y, std::size_t batch,
                         std::size_t m, std::size_t k_out, hipStream_t stream);
// Gate and up for the same rows with SwiGLU applied; y holds half floats, token
// major.
void LaunchInt4GemmPair(const std::uint8_t* gate, const std::uint8_t* up,
                        const std::uint8_t* x, const float* gate_scale,
                        const float* up_scale, const float* token_scale,
                        const Int4Shifts& shifts, void* y, std::size_t batch,
                        std::size_t m, std::size_t k_out, hipStream_t stream);
}  // namespace superroc::hip
#endif
#endif  // SUPERROC_MODELS_QWEN_HIP_OPS_PREFILL_INT4_HPP_
