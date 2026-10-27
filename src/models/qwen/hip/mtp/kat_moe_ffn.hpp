#ifndef SUPERROC_MODELS_QWEN_HIP_MTP_KAT_MOE_FFN_HPP_
#define SUPERROC_MODELS_QWEN_HIP_MTP_KAT_MOE_FFN_HPP_

#include <cstddef>
#include <cstdint>

#include "src/core/model_config.hpp"
#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace superroc::hip {

struct KatMtpMoeScratch {
  float* logits{nullptr};
  int* ids{nullptr};
  float* weights{nullptr};
  float* sig{nullptr};
  float* gate{nullptr};
  float* up{nullptr};
  float* act{nullptr};
  float* hidden{nullptr};
};

// KAT-MTP: one-token MoE FFN for the embedded nextn block. Same math as the
// trunk's MoE (the top-k router and the sigmoid-gated shared expert) but a
// different kernel: a per-expert Q8 GEMM driven from a host loop, with one
// D2H readback and stream sync per draft token, so ulp differences vs the
// trunk are expected. Writes `out` (it does not add into a residual).
void LaunchKatMtpMoeFfn(const models::QwenLayerWeights& layer,
                        const core::ModelConfig& config, const float* normed,
                        float* out, const KatMtpMoeScratch& scratch,
                        hipStream_t stream);

// KAT-MTP: pack already-normed, already-RoPE'd K/V into the decode cache.
// Index formula matches WriteKVCacheKernel (layer 0).
void LaunchKatMtpWriteKv(const float* k, const float* v, float* k_cache,
                         float* v_cache, void* k_cache_f16, void* v_cache_f16,
                         std::uint32_t start_pos, std::size_t batch,
                         std::uint32_t max_context, std::uint32_t num_kv_heads,
                         std::uint32_t head_dim, hipStream_t stream);

}  // namespace superroc::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // SUPERROC_MODELS_QWEN_HIP_MTP_KAT_MOE_FFN_HPP_
