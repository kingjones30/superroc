// Super ROC M2 R1: the model-family registry table.
#include "src/core/model_family.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string_view>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << '\n';
    ++g_failures;
  }
}

}  // namespace

int main() {
  using gufo::core::FindModelFamily;
  using gufo::core::IsModelFamily;
  using gufo::core::ModelFamilyArch;
  using gufo::core::ModelFamilyId;

  // Every exact family resolves to itself and spells its arch once.
  Check(IsModelFamily("qwen35moe", ModelFamilyId::kQwen35Moe), "qwen35moe");
  Check(IsModelFamily("qwen4exp", ModelFamilyId::kQwen4Exp), "qwen4exp");
  Check(IsModelFamily("deepseek4", ModelFamilyId::kDeepSeek4), "deepseek4");
  Check(IsModelFamily("bailingmoe3", ModelFamilyId::kBailingMoe3),
        "bailingmoe3");
  Check(IsModelFamily("muse-glimmer", ModelFamilyId::kMuseGlimmer),
        "muse-glimmer");
  Check(IsModelFamily("lfm2moe", ModelFamilyId::kLfm2Moe), "lfm2moe");
  Check(IsModelFamily("flint", ModelFamilyId::kFlint), "flint");
  Check(ModelFamilyArch(ModelFamilyId::kFlint) == "flint", "arch flint");
  Check(!IsModelFamily("flint", ModelFamilyId::kQwen35),
        "flint is not the qwen prefix");
  Check(ModelFamilyArch(ModelFamilyId::kLfm2Moe) == "lfm2moe", "arch lfm2moe");
  Check(ModelFamilyArch(ModelFamilyId::kQwen4Exp) == "qwen4exp",
        "arch qwen4exp");
  Check(ModelFamilyArch(ModelFamilyId::kQwen35Moe) == "qwen35moe",
        "arch qwen35moe");
  Check(ModelFamilyArch(ModelFamilyId::kQwen35).empty(),
        "prefix family has no canonical arch");

  // The old server gate accepted every arch that starts with "qwen".
  Check(IsModelFamily("qwen35", ModelFamilyId::kQwen35), "qwen35");
  Check(IsModelFamily("qwen3", ModelFamilyId::kQwen35), "qwen3 prefix");
  Check(IsModelFamily("qwen2", ModelFamilyId::kQwen35), "qwen2 prefix");
  // Exact entries win over the prefix entry.
  Check(!IsModelFamily("qwen35moe", ModelFamilyId::kQwen35),
        "qwen35moe is not the prefix family");
  Check(!IsModelFamily("qwen4exp", ModelFamilyId::kQwen35),
        "qwen4exp is not the prefix family");

  // Unknown and empty fail closed; near misses do not match.
  Check(FindModelFamily("") == nullptr, "empty");
  Check(IsModelFamily("llama", ModelFamilyId::kLlama), "llama");
  Check(ModelFamilyArch(ModelFamilyId::kLlama) == "llama", "arch llama");
  Check(!IsModelFamily("llama", ModelFamilyId::kQwen35),
        "llama is not the qwen prefix");
  Check(IsModelFamily("lfm2", ModelFamilyId::kLfm2), "lfm2 dense");
  Check(!IsModelFamily("lfm2", ModelFamilyId::kLfm2Moe), "lfm2 is not lfm2moe");
  Check(gufo::core::IsLfm2Family("lfm2") &&
            gufo::core::IsLfm2Family("lfm2moe") &&
            !gufo::core::IsLfm2Family("qwen35"),
        "lfm2 family");
  Check(ModelFamilyArch(ModelFamilyId::kLfm2) == "lfm2", "arch lfm2");
  Check(FindModelFamily("Qwen35") == nullptr, "case sensitive");
  Check(FindModelFamily("deepseek4x") == nullptr, "no suffix match");

  // A missing metadata value reaches the registry as "": no family.
  Check(!IsModelFamily(std::optional<std::string_view>{}.value_or(""),
                       ModelFamilyId::kLfm2Moe),
        "missing arch");

  // The server refusal texts are byte-identical to the pre-registry ones.
  const auto* glimmer = FindModelFamily("muse-glimmer");
  const auto* lfm2 = FindModelFamily("lfm2moe");
  const auto* kat = FindModelFamily("qwen35moe");
  Check(glimmer != nullptr &&
            glimmer->vision_refusal == "muse-glimmer does not support vision",
        "glimmer vision text");
  Check(glimmer != nullptr &&
            glimmer->speculative_refusal ==
                "muse-glimmer speculative decoding is not in S0/S1 (no MTP)",
        "glimmer speculative text");
  Check(lfm2 != nullptr &&
            lfm2->vision_refusal == "lfm2moe does not support vision",
        "lfm2 vision text");
  Check(lfm2 != nullptr && lfm2->speculative_refusal.empty(),
        "lfm2 speculative text");
  Check(kat != nullptr && kat->vision_refusal.empty() &&
            kat->speculative_refusal.empty(),
        "kat refusals stay in its own dispatch");
  const auto* flint = FindModelFamily("flint");
  Check(flint != nullptr &&
            flint->vision_refusal == "flint does not support vision",
        "flint vision text");
  Check(flint != nullptr && flint->speculative_refusal ==
                                "flint has no speculative decoding yet",
        "flint speculative text");
  const auto* llama = FindModelFamily("llama");
  Check(llama != nullptr &&
            llama->vision_refusal == "llama does not support vision",
        "llama vision text");
  // NBFAST T1: llama gained DFlash2 drafting; like LFM2, the blanket
  // refusal stays empty and the serve loader refuses other backends.
  Check(llama != nullptr && llama->speculative_refusal.empty(),
        "llama speculative text");

  // Compile-time use (package constants are initialised from the table).
  static_assert(gufo::core::IsModelFamily("lfm2moe", ModelFamilyId::kLfm2Moe));
  static_assert(ModelFamilyArch(ModelFamilyId::kBailingMoe3) == "bailingmoe3");

  if (g_failures != 0) {
    std::cerr << g_failures << " failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "model_family_test: ALL OK\n";
  return EXIT_SUCCESS;
}
