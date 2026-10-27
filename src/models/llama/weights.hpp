#ifndef SUPERROC_MODELS_LLAMA_WEIGHTS_HPP_
#define SUPERROC_MODELS_LLAMA_WEIGHTS_HPP_

#include <optional>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/core/model_config.hpp"
#include "src/models/qwen/state.hpp"

namespace superroc::models {

/// Bind a dense Llama GGUF onto the Qwen chassis weight struct. Every layer is
/// full-attention SwiGLU; q/k norms, sinks, and MoE tensors are refused.
[[nodiscard]] std::optional<QwenModelWeights> BindLlamaWeights(
    const core::GgufReader& reader, const core::ModelConfig& config,
    std::string* error_msg = nullptr);

}  // namespace superroc::models

#endif  // SUPERROC_MODELS_LLAMA_WEIGHTS_HPP_
