#ifndef SUPERROC_MODELS_QWEN_HIP_MTP_KAT_MTP_TYPES_HPP_
#define SUPERROC_MODELS_QWEN_HIP_MTP_KAT_MTP_TYPES_HPP_

#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/mtp_reference.hpp"

namespace superroc::hip {

// KAT-MTP: the only types the one-token expert GEMV runs, for a routed or a
// shared nextn expert (hip/mtp/kat_moe_ffn.hip, GemvQ8).
[[nodiscard]] inline bool KatMtpExpertGemmTypeSupported(
    core::GgmlType type) noexcept {
  return type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ8_K;
}

// KAT-MTP: the only types RunPrimeGemm (hip/mtp/batch.cpp) executes. Q8_0 and
// Q8_K go straight to the quantized GEMM, Q4_0 is dequantized to BF16, and
// BF16 is read as BF16. Anything else would be reinterpreted as BF16 and
// would fill the draft KV with garbage.
[[nodiscard]] inline bool KatMtpPrimeGemmTypeSupported(
    core::GgmlType type) noexcept {
  switch (type) {
    case core::GgmlType::kQ4_0:
    case core::GgmlType::kQ8_0:
    case core::GgmlType::kQ8_K:
    case core::GgmlType::kBF16:
      return true;
    default:
      return false;
  }
}

// KAT-MTP F2 (RV44): the one load-time type gate for the embedded nextn
// block. QwenMtpGpuModel::CreateFromEmbedded (hip/mtp/model.cpp) calls this
// before it builds anything, so an unsupported tensor type is refused at load
// instead of throwing on the first draft step (an HTTP 500 in serve) or being
// silently read as BF16 (a garbage draft KV, acceptance collapse).
// Header-only and HIP-free, so the CPU unit test
// tests/models/qwen/kat_mtp_types_test.cpp drives this exact function with a
// synthetic block.
[[nodiscard]] inline bool KatMtpValidateEmbeddedBlockTypes(
    const speculative::QwenMtpWeights& weights, std::string* error_msg) {
  const auto reject = [error_msg](const char* message) {
    if (error_msg != nullptr) {
      *error_msg = message;
    }
    return false;
  };

  const auto& layer = weights.layer;
  if (!KatMtpExpertGemmTypeSupported(layer.ffn_gate_exps.type) ||
      !KatMtpExpertGemmTypeSupported(layer.ffn_up_exps.type) ||
      !KatMtpExpertGemmTypeSupported(layer.ffn_down_exps.type)) {
    return reject("KAT MTP nextn expert matrices must be Q8_0 or Q8_K");
  }

  // The shared expert is optional, but if any of its three matrices is bound
  // the other two must be as well (ExpertSwiglu reads all three).
  const bool shared_present = !layer.ffn_gate_shexp.empty() ||
                              !layer.ffn_up_shexp.empty() ||
                              !layer.ffn_down_shexp.empty();
  if (shared_present &&
      (!KatMtpExpertGemmTypeSupported(layer.ffn_gate_shexp.type) ||
       !KatMtpExpertGemmTypeSupported(layer.ffn_up_shexp.type) ||
       !KatMtpExpertGemmTypeSupported(layer.ffn_down_shexp.type))) {
    return reject("KAT MTP nextn shared-expert matrices must be Q8_0 or Q8_K");
  }

  if (!KatMtpPrimeGemmTypeSupported(weights.fusion_projection.type) ||
      !KatMtpPrimeGemmTypeSupported(layer.attn_k.type) ||
      !KatMtpPrimeGemmTypeSupported(layer.attn_v.type)) {
    return reject("KAT MTP prime weights must be Q4_0, Q8_0, Q8_K or BF16");
  }
  return true;
}

}  // namespace superroc::hip

#endif  // SUPERROC_MODELS_QWEN_HIP_MTP_KAT_MTP_TYPES_HPP_
