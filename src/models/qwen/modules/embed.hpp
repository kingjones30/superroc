#ifndef SUPERROC_MODELS_QWEN_MODULES_EMBED_HPP_
#define SUPERROC_MODELS_QWEN_MODULES_EMBED_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/state.hpp"

namespace superroc::models::qwen {

/// Token embedding lookup: copies the embedding row for token_id into out.
void EmbedForward(const CpuModuleContext& ctx, std::uint32_t token_id,
                  const QwenTensorRef& token_embd, std::size_t hidden_size,
                  std::span<float> out) noexcept;

}  // namespace superroc::models::qwen

#endif  // SUPERROC_MODELS_QWEN_MODULES_EMBED_HPP_
