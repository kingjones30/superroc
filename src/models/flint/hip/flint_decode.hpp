// Flint decode step: one-token forward over GQA + sigmoid MoE + ungated shexp.
#ifndef GUFO_MODELS_FLINT_HIP_FLINT_DECODE_HPP_
#define GUFO_MODELS_FLINT_HIP_FLINT_DECODE_HPP_

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::models {
struct QwenModelWeights;
namespace qwen::vision {
class DeviceInput;
}
}  // namespace gufo::models

namespace gufo::hip {

class QwenGpuArena;

/// Flint: one decode step. Throws before any kernel if an index is illegal.
/// Does not enter HIP graph capture.
void ExecuteFlintDecodeStep(QwenGpuArena& arena,
                            const models::QwenModelWeights& weights,
                            tokenization::TokenId token_id, std::uint32_t pos,
                            bool compute_logits,
                            models::qwen::vision::DeviceInput* image_input);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_FLINT_HIP_FLINT_DECODE_HPP_
