#ifndef GUFO_MODELS_LLAMA_HIP_LLAMA_DECODE_HPP_
#define GUFO_MODELS_LLAMA_HIP_LLAMA_DECODE_HPP_

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace gufo::models {
struct QwenModelWeights;
namespace qwen::vision {
class DeviceInput;
}
}  // namespace gufo::models

namespace gufo::hip {

class QwenGpuArena;

void ExecuteLlamaDecodeStep(QwenGpuArena& arena,
                            const models::QwenModelWeights& weights,
                            tokenization::TokenId token_id, std::uint32_t pos,
                            bool compute_logits,
                            models::qwen::vision::DeviceInput* image_input);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_LLAMA_HIP_LLAMA_DECODE_HPP_
