// Flint decode step: one-token forward over GQA + sigmoid MoE + ungated shexp.
#ifndef SUPERROC_MODELS_FLINT_HIP_FLINT_DECODE_HPP_
#define SUPERROC_MODELS_FLINT_HIP_FLINT_DECODE_HPP_

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace superroc::models {
struct QwenModelWeights;
namespace qwen::vision {
class DeviceInput;
}
}  // namespace superroc::models

namespace superroc::hip {

class QwenGpuArena;

/// Flint: one decode step. Throws before any kernel if an index is illegal.
/// Does not enter HIP graph capture.
void ExecuteFlintDecodeStep(QwenGpuArena& arena,
                            const models::QwenModelWeights& weights,
                            tokenization::TokenId token_id, std::uint32_t pos,
                            bool compute_logits,
                            models::qwen::vision::DeviceInput* image_input);

}  // namespace superroc::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // SUPERROC_MODELS_FLINT_HIP_FLINT_DECODE_HPP_
