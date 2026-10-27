// superroc-quant recipes — the seven fork ROCmFP4 recipes as DATA.
//
// The per-tensor type for a recipe is a tensor-class -> type table plus a few
// layer-position predicates. This reproduces llama-quant.cpp's
// `llama_tensor_get_type_impl` ladder for exactly the 7 ROC ftypes
// (100..106). Ported-worthy logic is described in SPEC-LOCK.md §2; this is a
// fresh, data-driven reimplementation of that ladder (not a copy of ggml).

#ifndef SUPERROC_QUANT_RECIPE_HPP_
#define SUPERROC_QUANT_RECIPE_HPP_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gguf.hpp"

namespace superroc::quant_tool {

enum class Cat {
  kOutput,
  kTokenEmbd,
  kAttnQkv,
  kAttnKvB,
  kAttnV,
  kAttnK,
  kAttnQ,
  kAttnOut,
  kFfnUp,
  kFfnGate,
  kFfnDown,
  kOther,
};

[[nodiscard]] inline std::string_view CatName(Cat c) {
  switch (c) {
    case Cat::kOutput: return "output";
    case Cat::kTokenEmbd: return "token_embd";
    case Cat::kAttnQkv: return "attn_qkv";
    case Cat::kAttnKvB: return "attn_kv_b";
    case Cat::kAttnV: return "attn_v";
    case Cat::kAttnK: return "attn_k";
    case Cat::kAttnQ: return "attn_q";
    case Cat::kAttnOut: return "attn_out";
    case Cat::kFfnUp: return "ffn_up";
    case Cat::kFfnGate: return "ffn_gate";
    case Cat::kFfnDown: return "ffn_down";
    case Cat::kOther: return "other";
  }
  return "?";
}

inline bool EndsWith(const std::string& name, const std::string& suf) {
  return name.size() >= suf.size() &&
         name.compare(name.size() - suf.size(), suf.size(), suf) == 0;
}

// tensor_allows_quantization subset: 2D+, name ends "weight", not norm/router,
// not conv/positional/etc. Excludes the exclusions the fork applies.
[[nodiscard]] inline bool TensorAllowsQuantization(const std::string& name,
                                                   bool quantize_output) {
  if (name.size() < 6 || name.compare(name.size() - 6, 6, "weight") != 0)
    return false;
  if (name.find("_norm.weight") != std::string::npos) return false;
  if (!quantize_output && name == "output.weight") return false;
  if (name.find("ffn_gate_inp.weight") != std::string::npos) return false;
  if (name.find("ffn_gate_inp_shexp.weight") != std::string::npos) return false;
  if (name.find("altup") != std::string::npos) return false;
  if (name.find("laurel") != std::string::npos) return false;
  if (name.find("per_layer_model_proj") != std::string::npos) return false;
  if (name.find("ssm_conv1d") != std::string::npos) return false;
  if (name.find("shortconv.conv.weight") != std::string::npos) return false;
  if (name.find(".position_embd") != std::string::npos) return false;
  if (name.find(".rel_pos") != std::string::npos) return false;
  if (name.find(".patch_embd") != std::string::npos) return false;
  if (name.find(".patch_merger") != std::string::npos) return false;
  return true;
}

/// Returns true when the tensor must stay at its source precision (not quantized).
[[nodiscard]] inline bool IsProtected(const std::string& name) {
  // exact names
  if (name == "output.weight") return false;  // handled by allows_quantization
  if (EndsWith(name, "weight") && name.find("ffn_gate_inp.weight") != std::string::npos) return true;
  if (name.find("_norm.weight") != std::string::npos) return true;
  return false;
}

[[nodiscard]] inline Cat Categorize(const std::string& name) {
  if (name == "output.weight") return Cat::kOutput;
  if (name == "token_embd.weight" || name == "per_layer_token_embd.weight")
    return Cat::kTokenEmbd;
  if (name.find("nextn.shared_head_head") != std::string::npos) return Cat::kOutput;
  if (name.find("nextn.embed_tokens") != std::string::npos) return Cat::kTokenEmbd;
  if (name.find("attn_qkv.weight") != std::string::npos) return Cat::kAttnQkv;
  if (name.find("attn_kv_b.weight") != std::string::npos) return Cat::kAttnKvB;
  if (name.find("attn_v.weight") != std::string::npos) return Cat::kAttnV;
  if (name.find("attn_k.weight") != std::string::npos) return Cat::kAttnK;
  if (name.find("attn_q.weight") != std::string::npos) return Cat::kAttnQ;
  if (name.find("attn_output.weight") != std::string::npos) return Cat::kAttnOut;
  if (name.find("ffn_up") != std::string::npos) return Cat::kFfnUp;
  if (name.find("ffn_gate") != std::string::npos) return Cat::kFfnGate;
  if (name.find("ffn_down") != std::string::npos) return Cat::kFfnDown;
  return Cat::kOther;
}

/// Parse layer index from `blk.<N>.` (returns -1 if absent).
[[nodiscard]] inline int LayerFromName(const std::string& name, int n_layer) {
  constexpr const char* pat = "blk.";
  auto p = name.find(pat);
  if (p == std::string::npos) return 0;
  size_t q = p + 4;
  std::string digits;
  while (q < name.size() && name[q] >= '0' && name[q] <= '9') {
    digits += name[q];
    ++q;
    if (digits.size() > 6) break;
  }
  if (digits.empty()) return 0;
  int layer = std::stoi(digits);
  if (layer < 0 || layer >= n_layer) layer = 0;
  return layer;
}

inline bool UseMoreBits(int i_layer, int n_layers) {
  return i_layer < n_layers / 8 || i_layer >= 7 * n_layers / 8 ||
         (i_layer - n_layers / 8) % 3 == 2;
}

// Draft-sensitive projection -> Q8_0 floor.
[[nodiscard]] inline bool IsDraftSensitiveProjection(const std::string& name) {
  return name == "fc.weight" ||
         name.find("nextn.eh_proj") != std::string::npos ||
         name.find("nextn.e_proj") != std::string::npos ||
         name.find("nextn.h_proj") != std::string::npos ||
         name.find("mtp.pre_projection") != std::string::npos ||
         name.find("mtp.post_projection") != std::string::npos;
}

/// The 7 fork recipes. ftype = GGUF file_type value (100..106).
/// The two EVEN recipes map to ftype 100/103 with --pure (see DECISIONS/SPEC2
/// item 6 and the fork's quantize.cpp, which sets params.pure for
/// Q4_0_ROCMFP4_EVEN/FAST_EVEN).
/// G4 recipes (product track) are superroc-native layout variants (120/121).
enum class Recipe {
  kRocmFp4 = 100,
  kLean = 101,
  kCoherent = 102,
  kFast = 103,
  kFastCoherent = 104,
  kStrix = 105,
  kStrixLean = 106,
  // superroc-native split-scale-plane recipes (RV20 §3). file_type stays 105/106
  // (G4 files carry the g4 recipe in superroc.quant.recipe and format_version 2).
  kG4Strix = 200,
  kG4StrixLean = 201,
};

inline Recipe RecipeFromFtype(std::uint32_t ftype) {
  switch (ftype) {
    case 100: return Recipe::kRocmFp4;
    case 101: return Recipe::kLean;
    case 102: return Recipe::kCoherent;
    case 103: return Recipe::kFast;
    case 104: return Recipe::kFastCoherent;
    case 105: return Recipe::kStrix;
    case 106: return Recipe::kStrixLean;
  }
  throw std::runtime_error("unknown ROC recipe ftype");
}

[[nodiscard]] inline Type RecipeDefault(Recipe r) {
  switch (r) {
    case Recipe::kRocmFp4: case Recipe::kLean: case Recipe::kCoherent:
      return (Type)100;
    case Recipe::kG4Strix:
      return (Type)121;
    case Recipe::kG4StrixLean:
      return (Type)121;
    default:
      return (Type)101;
  }
}

[[nodiscard]] inline bool IsG4(Recipe r) {
  return r == Recipe::kG4Strix || r == Recipe::kG4StrixLean;
}

// The G4 recipes are built as a post-transform of the STRIX recipe: run the
// STRIX selection, then map 100->120 and 101->121 (split scale plane). The
// g4-strix recipe additionally keeps shared experts (ffn_*_shexp) and
// attention K/V on the dual-scale 120 layout (RV20 §3.3).
[[nodiscard]] inline Type G4Map(Type t) {
  if (t == (Type)100) return (Type)120;
  if (t == (Type)101) return (Type)121;
  return t;
}
[[nodiscard]] inline bool IsSharedExpert(const std::string& name) {
  return name.find("_shexp") != std::string::npos;
}

inline bool InLeanFamily(Recipe r) {
  return r == Recipe::kLean || r == Recipe::kCoherent ||
         r == Recipe::kFastCoherent || r == Recipe::kStrix ||
         r == Recipe::kStrixLean;
}

inline bool IsStrix(Recipe r) {
  return r == Recipe::kStrix || r == Recipe::kStrixLean;
}

inline bool IsLeanOrStrixLean(Recipe r) {
  return r == Recipe::kLean || r == Recipe::kStrixLean;
}

// Size helpers (BlockBytes/BlockElems) live in gguf.hpp; this header relies on
// them via the shared namespace.

/// Shape-fallback: only when ne[0] is NOT divisible by the target's block size
/// map to a compatible type, exactly as tensor_type_fallback does.
[[nodiscard]] inline Type FallbackFor(Type t, std::uint64_t ne0) {
  if (ne0 % BlockElems(t) == 0) return t;   // no fallback needed
  Type fb = t;
  switch (t) {
    case (Type)100: case (Type)101: fb = (Type)2; break;   // Q4_0
    case (Type)120: case (Type)121: fb = (Type)2; break;   // G4 -> Q4_0 (behaves like 100/101)
    case (Type)13: fb = (Type)7; break;                     // Q5_1 (ggml type 7)
    case (Type)14: fb = (Type)8; break;                     // Q8_0
    default: break;  // no ROC fallback defined -> leave (other types)
  }
  if (ne0 % BlockElems(fb) != 0) fb = (Type)1;  // F16 catch-all
  return fb;
}

/// Select the quantized target type for a tensor, mirroring the fork ladder.
/// `tied_to_output`: when true, TOKEN_EMBD is treated as OUTPUT (fork logic).
[[nodiscard]] inline Type SelectType(Recipe r, const std::string& name, Cat cat,
                                     Type default_type, std::uint64_t ne0,
                                     int n_layer, bool has_expert,
                                     bool tied_to_output) {
  // G4 recipes run the STRIX ladder (base = 105/106 depending on which), then
  // map 100->120 / 101->121 to the split-scale-plane layout. Shared experts
  // stay dual (120) even where STRIX would use single-scale fast 121.
  if (IsG4(r)) {
    Recipe base = (r == Recipe::kG4Strix) ? Recipe::kStrix : Recipe::kStrixLean;
    Type bt = SelectType(base, name, cat, (Type)101, ne0, n_layer, has_expert, tied_to_output);
    Type mapped = G4Map(bt);
    if (IsSharedExpert(name) && (mapped == (Type)121)) mapped = (Type)120;
    // draft-sensitive projections must keep their Q8_0 floor (G4 not applied)
    if (IsDraftSensitiveProjection(name) && (bt == (Type)8)) mapped = (Type)8;
    return mapped;
  }

  Type new_type = default_type;
  bool is_draft = IsDraftSensitiveProjection(name);

  // OUTPUT (and tied token_embd)
  if (cat == Cat::kOutput || (tied_to_output && cat == Cat::kTokenEmbd)) {
    if (r == Recipe::kFast) {
      new_type = (Type)101;
    } else if (InLeanFamily(r)) {
      if (cat == Cat::kTokenEmbd) {
        new_type = IsLeanOrStrixLean(r) ? (Type)13 /*Q5_K*/ : (Type)14 /*Q6_K*/;
      } else {
        new_type = (r == Recipe::kFastCoherent || IsStrix(r)) ? (Type)101 : (Type)100;
      }
    } else {
      // pure ROCMFP4 (100): falls through to Q6_K
      if (ne0 % 256 != 0) new_type = (Type)8;  // Falcon-like fallback
      else new_type = (Type)14;                 // Q6_K
    }
  }
  // TOKEN_EMBD
  else if (cat == Cat::kTokenEmbd) {
    if (InLeanFamily(r)) {
      new_type = IsLeanOrStrixLean(r) ? (Type)13 : (Type)14;
    } else {
      new_type = default_type;  // pure-100 stays 100
    }
  }
  // ATTENTION_V / QKV / KV_B
  else if (cat == Cat::kAttnV || cat == Cat::kAttnQkv || cat == Cat::kAttnKvB) {
    if (IsStrix(r)) new_type = (Type)100;
    else if (r == Recipe::kRocmFp4) new_type = (Type)13;  // Q5_K
    else new_type = default_type;
  }
  // ATTENTION_K
  else if (cat == Cat::kAttnK) {
    if (IsStrix(r)) new_type = (Type)100;
    else new_type = default_type;
  }
  // ATTENTION_Q
  else if (cat == Cat::kAttnQ) {
    new_type = default_type;
  }
  // ATTENTION_OUTPUT
  else if (cat == Cat::kAttnOut) {
    if (r == Recipe::kRocmFp4) new_type = (Type)13;  // Q5_K
    else new_type = default_type;
  }
  // FFN_DOWN
  else if (cat == Cat::kFfnDown) {
    int i_layer = LayerFromName(name, n_layer);
    if (r == Recipe::kRocmFp4) {
      new_type = (n_layer > 0 && i_layer >= (2 * n_layer) / 3)
                     ? (Type)13 /*Q5_K*/ : (Type)14 /*Q6_K*/;
    } else {
      new_type = default_type;
    }
  }
  // FFN_GATE
  else if (cat == Cat::kFfnGate) {
    int i_layer = LayerFromName(name, n_layer);
    if (r == Recipe::kRocmFp4 && UseMoreBits(i_layer, n_layer))
      new_type = (Type)13;  // Q5_K
    else
      new_type = default_type;
  }
  // FFN_UP
  else if (cat == Cat::kFfnUp) {
    new_type = default_type;
  }
  // OTHER (incl. MTP/NextN, unless draft-sensitive -> Q8_0)
  else {
    new_type = default_type;
  }

  // MTP/draft-sensitive projection floor -> Q8_0
  if (is_draft && (new_type == (Type)100 || new_type == (Type)101 ||
                   new_type == (Type)2)) {
    new_type = (Type)8;
  }

  // shape fallback
  new_type = FallbackFor(new_type, ne0);
  return new_type;
}

}  // namespace superroc::quant_tool

#endif  // SUPERROC_QUANT_RECIPE_HPP_
