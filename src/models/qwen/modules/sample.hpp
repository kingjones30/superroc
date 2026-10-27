#ifndef SUPERROC_MODELS_QWEN_MODULES_SAMPLE_HPP_
#define SUPERROC_MODELS_QWEN_MODULES_SAMPLE_HPP_

#include <cstdint>
#include <span>

#include "src/models/qwen/modules/module_ctx.hpp"

namespace superroc::models::qwen {

/// Greedy argmax over a logit distribution (sampling-policy module seam).
std::uint32_t SampleForward(const CpuModuleContext& ctx,
                            std::span<const float> logits) noexcept;

}  // namespace superroc::models::qwen

#endif  // SUPERROC_MODELS_QWEN_MODULES_SAMPLE_HPP_
