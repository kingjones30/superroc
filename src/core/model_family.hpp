// Super ROC M2 R1: the model-family registry.
//
// Every architecture name the engine recognises is spelled once, here. Code
// asks FindModelFamily(arch) / IsModelFamily(arch, id) instead of comparing
// `general.architecture` strings. Adding a family = one row in this table plus
// the family's own package (see docs/king/SUPERROC-REGISTRY.md).
#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace gufo::core {

enum class ModelFamilyId : std::uint8_t {
  kQwen35,       // qwen35 and every other "qwen*" GGUF: Qwen hybrid chassis
  kQwen35Moe,    // KAT: MoE on the Qwen chassis
  kQwen4Exp,     // Qwen3.8-Flash-Next package
  kDeepSeek4,    // DeepSeek-V4-Flash package
  kBailingMoe3,  // Ling-3.0-flash package
  kMuseGlimmer,  // attention-only chassis on the Qwen config
  kLfm2Moe,      // LFM2.5: short-conv + GQA MoE on the Qwen chassis
  kLfm2,         // LFM2.5 dense: short-conv + GQA, SwiGLU FFN in every layer
  kFlint,  // RecursiveSI/flint: Qwen-chassis MoE, sliding/NoPE, attention sinks
};

struct ModelFamily {
  ModelFamilyId id;
  std::string_view arch;  // the GGUF `general.architecture` value
  bool prefix_match;      // true: matches every arch that starts with `arch`
  // Non-empty: the HTTP server refuses --mmproj with exactly this text.
  std::string_view vision_refusal;
  // Non-empty: the HTTP server refuses every speculative backend with exactly
  // this text.
  std::string_view speculative_refusal;
};

inline constexpr std::array<ModelFamily, 9> kModelFamilies{{
    {ModelFamilyId::kQwen35Moe, "qwen35moe", false, "", ""},
    {ModelFamilyId::kQwen4Exp, "qwen4exp", false, "", ""},
    {ModelFamilyId::kDeepSeek4, "deepseek4", false, "", ""},
    {ModelFamilyId::kBailingMoe3, "bailingmoe3", false, "", ""},
    {ModelFamilyId::kMuseGlimmer, "muse-glimmer", false,
     "muse-glimmer does not support vision",
     "muse-glimmer speculative decoding is not in S0/S1 (no MTP)"},
    // LFM2 families accept only DSpark drafting via --speculative dflash2;
    // the serve loader enforces that backend-by-backend (there is no MTP
    // head), so the blanket refusal stays empty.
    {ModelFamilyId::kLfm2Moe, "lfm2moe", false,
     "lfm2moe does not support vision", ""},
    {ModelFamilyId::kLfm2, "lfm2", false, "lfm2 does not support vision", ""},
    {ModelFamilyId::kFlint, "flint", false, "flint does not support vision",
     "flint has no speculative decoding yet"},
    {ModelFamilyId::kQwen35, "qwen", true, "", ""},
}};

/// Metadata reads: pass `GetMetadataString(...).value_or("")`; a missing
/// value is "", which is no family.
/// Exact entries first, then prefix entries, in table order.
/// nullptr = unsupported architecture (callers fail closed).
constexpr const ModelFamily* FindModelFamily(std::string_view arch) noexcept {
  for (const ModelFamily& family : kModelFamilies) {
    if (!family.prefix_match && family.arch == arch) {
      return &family;
    }
  }
  for (const ModelFamily& family : kModelFamilies) {
    if (family.prefix_match && arch.starts_with(family.arch)) {
      return &family;
    }
  }
  return nullptr;
}

constexpr bool IsModelFamily(std::string_view arch, ModelFamilyId id) noexcept {
  const ModelFamily* family = FindModelFamily(arch);
  return family != nullptr && family->id == id;
}

/// lfm2 (dense) and lfm2moe share the LFM2 runtime (short-conv + GQA, LFM2
/// chat).
constexpr bool IsLfm2Family(std::string_view arch) noexcept {
  return IsModelFamily(arch, ModelFamilyId::kLfm2) ||
         IsModelFamily(arch, ModelFamilyId::kLfm2Moe);
}

/// The canonical `general.architecture` spelling of an exact family.
constexpr std::string_view ModelFamilyArch(ModelFamilyId id) noexcept {
  for (const ModelFamily& family : kModelFamilies) {
    if (family.id == id && !family.prefix_match) {
      return family.arch;
    }
  }
  return {};
}

}  // namespace gufo::core
