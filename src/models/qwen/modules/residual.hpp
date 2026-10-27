#ifndef SUPERROC_MODELS_QWEN_MODULES_RESIDUAL_HPP_
#define SUPERROC_MODELS_QWEN_MODULES_RESIDUAL_HPP_

#include <span>

#include "src/models/qwen/modules/module_ctx.hpp"

namespace superroc::models::qwen {

/// Residual add: dst = dst + src (element-wise). In-place on `dst`.
///
/// The fused residual+norm route is a composition-layer concern (owned by the
/// per-layer `ExecuteStep`), not a module-local one, so the module is the
/// standalone add only.
void ResidualAdd(const CpuModuleContext& ctx, std::span<float> dst,
                 std::span<const float> src) noexcept;
void ResidualAdd(const HipModuleContext& ctx, std::span<float> dst,
                 std::span<const float> src) noexcept;

}  // namespace superroc::models::qwen

#endif  // SUPERROC_MODELS_QWEN_MODULES_RESIDUAL_HPP_
