#ifndef SUPERROC_MODELS_LLAMA_HIP_LLAMA_DECODE_HPP_
#define SUPERROC_MODELS_LLAMA_HIP_LLAMA_DECODE_HPP_

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace superroc::models {
struct QwenModelWeights;
namespace qwen::vision {
class DeviceInput;
}
}  // namespace superroc::models

namespace superroc::hip {

class QwenGpuArena;

void ExecuteLlamaDecodeStep(QwenGpuArena& arena,
                            const models::QwenModelWeights& weights,
                            tokenization::TokenId token_id, std::uint32_t pos,
                            bool compute_logits,
                            models::qwen::vision::DeviceInput* image_input);

}  // namespace superroc::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // SUPERROC_MODELS_LLAMA_HIP_LLAMA_DECODE_HPP_
