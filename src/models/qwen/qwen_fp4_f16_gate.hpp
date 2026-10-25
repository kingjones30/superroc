#ifndef GUFO_MODELS_QWEN_QWEN_FP4_F16_GATE_HPP_
#define GUFO_MODELS_QWEN_QWEN_FP4_F16_GATE_HPP_

// FRZA SPEC2 (fail closed): a kernel cell that has never run on hardware is
// not taken by default. Type-100 (kQ4_0_ROCMFP4) tensors on the fused-epilogue
// prefill cells (ffn_up kSwiGLU; ffn_down, ssm_out, attn_output kResidual) are
// the REPORT1 candidate (a): bench-proven only on kStore, never on an epilogue
// with type 100. This header is the whole new decision, pure and host-only, so
// the engine and the host test share it exactly. No HIP, no device.

#include <cstddef>
#include <span>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen {

// Values of GUFO_QWEN_FP4_F16_PREFILL.
constexpr int kQwenFp4F16PrefillOff = 0;    // kill switch: pre-f85008b dispatch
constexpr int kQwenFp4F16PrefillGated = 1;  // fail-closed gate (default)
constexpr int kQwenFp4F16PrefillForce = 2;  // tag behaviour: F16 for every FP4

constexpr bool IsQwenFp4GateType(core::GgmlType type) noexcept {
  return type == core::GgmlType::kQ4_0_ROCMFP4 ||
         type == core::GgmlType::kQ4_0_ROCMFP4_FAST;
}

// Unset/empty/unknown -> gated; a leading '0' -> off (exactly today's off
// condition); exact "2" -> force. Unknown values fail closed.
constexpr int ParseQwenFp4F16PrefillMode(const char* env) noexcept {
  if (env == nullptr || env[0] == '\0')
    return kQwenFp4F16PrefillGated;
  if (env[0] == '0')
    return kQwenFp4F16PrefillOff;
  if (env[0] == '2' && env[1] == '\0')
    return kQwenFp4F16PrefillForce;
  return kQwenFp4F16PrefillGated;
}

// One layer's checked tensors, plain GGUF type codes. Same checked set as
// UseQwen27bFp16Prefill: FFN always, mixer per layer kind.
struct QwenFp4F16LayerTypes {
  bool is_full_attention = false;
  core::GgmlType ffn_gate = core::GgmlType::kF32;
  core::GgmlType ffn_up = core::GgmlType::kF32;
  core::GgmlType ffn_down = core::GgmlType::kF32;
  core::GgmlType attn_q = core::GgmlType::kF32;
  core::GgmlType attn_k = core::GgmlType::kF32;
  core::GgmlType attn_v = core::GgmlType::kF32;
  core::GgmlType attn_output = core::GgmlType::kF32;
  core::GgmlType attn_qkv = core::GgmlType::kF32;
  core::GgmlType attn_gate = core::GgmlType::kF32;
  core::GgmlType ssm_alpha = core::GgmlType::kF32;
  core::GgmlType ssm_beta = core::GgmlType::kF32;
  core::GgmlType ssm_out = core::GgmlType::kF32;
};

enum class QwenFp4F16Route : std::uint8_t {
  kLegacy,  // exactly the pre-f85008b dispatch (half_prefill=false)
  kF16,     // the tag engine's F16 route (capability checks still apply)
};

struct QwenFp4F16Decision {
  QwenFp4F16Route route = QwenFp4F16Route::kLegacy;
  bool has_fp4 = false;
  bool refused_by_gate = false;
  const char* family = "";
  int layer = -1;
};

// The route decision. Type 101 anywhere never refuses; files with no FP4
// tensor report has_fp4=false so the caller keeps its old logic verbatim.
constexpr QwenFp4F16Decision DecideQwenFp4F16Prefill(
    int mode, std::span<const QwenFp4F16LayerTypes> layers) noexcept {
  QwenFp4F16Decision decision;
  for (const auto& layer : layers) {
    if (IsQwenFp4GateType(layer.ffn_gate) || IsQwenFp4GateType(layer.ffn_up) ||
        IsQwenFp4GateType(layer.ffn_down)) {
      decision.has_fp4 = true;
      break;
    }
    if (layer.is_full_attention) {
      if (IsQwenFp4GateType(layer.attn_q) || IsQwenFp4GateType(layer.attn_k) ||
          IsQwenFp4GateType(layer.attn_v) ||
          IsQwenFp4GateType(layer.attn_output)) {
        decision.has_fp4 = true;
        break;
      }
    } else if (IsQwenFp4GateType(layer.attn_qkv) ||
               IsQwenFp4GateType(layer.attn_gate) ||
               IsQwenFp4GateType(layer.ssm_alpha) ||
               IsQwenFp4GateType(layer.ssm_beta) ||
               IsQwenFp4GateType(layer.ssm_out)) {
      decision.has_fp4 = true;
      break;
    }
  }
  if (!decision.has_fp4 || mode == kQwenFp4F16PrefillOff)
    return decision;
  if (mode == kQwenFp4F16PrefillForce) {
    decision.route = QwenFp4F16Route::kF16;
    return decision;
  }
  // Gated (mode 1, and any unknown mode): type 100 on a fused-epilogue cell
  // refuses the F16 route. Scan order is layer-ascending, ffn_up, ffn_down,
  // then the layer kind's output projection; the first hit names the log line.
  // Plain-store cells (kStore: ffn_gate, attn_q/k/v, attn_qkv, attn_gate,
  // ssm_alpha/beta) never refuse, whatever their type.
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const auto& layer = layers[i];
    const char* hit = nullptr;
    if (layer.ffn_up == core::GgmlType::kQ4_0_ROCMFP4) {
      hit = "ffn_up";
    } else if (layer.ffn_down == core::GgmlType::kQ4_0_ROCMFP4) {
      hit = "ffn_down";
    } else if (layer.is_full_attention) {
      if (layer.attn_output == core::GgmlType::kQ4_0_ROCMFP4)
        hit = "attn_output";
    } else if (layer.ssm_out == core::GgmlType::kQ4_0_ROCMFP4) {
      hit = "ssm_out";
    }
    if (hit != nullptr) {
      decision.refused_by_gate = true;
      decision.family = hit;
      decision.layer = static_cast<int>(i);
      return decision;
    }
  }
  decision.route = QwenFp4F16Route::kF16;
  return decision;
}

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_QWEN_FP4_F16_GATE_HPP_
