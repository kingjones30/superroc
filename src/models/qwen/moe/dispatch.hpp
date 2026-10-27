#ifndef SUPERROC_MODELS_QWEN_MOE_DISPATCH_HPP_
#define SUPERROC_MODELS_QWEN_MOE_DISPATCH_HPP_

#include <string_view>

#include "src/core/model_family.hpp"

namespace superroc::models::qwen {

// KAT: serve maps TextSpeculativeBackend by enumerator name
// (kDisabled, kDFlash, kDSpark, kMtp), not by casting the integer.
// kOff is TextSpeculativeBackend::kDisabled.
enum class KatSpecKind : int {
  kOff = 0,
  kDFlash = 1,
  kDSpark = 2,
  kMtp = 3,
};

enum class KatDispatch : int {
  kNotKat = 0,
  kAccept = 1,
  kRefuseDSpark = 2,
  kRefuseSidecar = 3,
  kRefuseVision = 4,
  kRefuseOther = 5,
};

/// True when a dflash sidecar is the MiniCPM/DSpark markov kind: markov_w1
/// present and the dflash2 selector codebook absent.
inline bool DflashSidecarIsMarkovKind(bool has_markov_w1,
                                      bool has_selector_predecessor) noexcept {
  return has_markov_w1 && !has_selector_predecessor;
}

/// KAT: qwen35moe policy. MTP is embedded (no sidecar). The enumerator
/// `--speculative dspark` stays refused (DeepSeek-only). `--speculative
/// dflash2` is accepted only when the sidecar GGUF is the markov kind.
inline KatDispatch ClassifyKatDispatch(
    std::string_view arch, KatSpecKind spec, bool draft_path_empty,
    bool has_vision, bool dflash_markov_sidecar = false) noexcept {
  if (!superroc::core::IsModelFamily(arch, superroc::core::ModelFamilyId::kQwen35Moe)) {
    return KatDispatch::kNotKat;
  }
  if (has_vision) {
    return KatDispatch::kRefuseVision;
  }
  if (spec == KatSpecKind::kDSpark) {
    return KatDispatch::kRefuseDSpark;
  }
  if (spec == KatSpecKind::kDFlash) {
    if (dflash_markov_sidecar && !draft_path_empty) {
      return KatDispatch::kAccept;
    }
    return KatDispatch::kRefuseOther;
  }
  if (spec == KatSpecKind::kMtp && !draft_path_empty) {
    return KatDispatch::kRefuseSidecar;
  }
  if (spec != KatSpecKind::kOff && spec != KatSpecKind::kMtp) {
    return KatDispatch::kRefuseOther;
  }
  return KatDispatch::kAccept;
}

}  // namespace superroc::models::qwen

#endif  // SUPERROC_MODELS_QWEN_MOE_DISPATCH_HPP_
