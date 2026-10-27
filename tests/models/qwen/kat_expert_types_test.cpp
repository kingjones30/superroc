// HGC audit J7: BindKatMoe require_fp4 trunk gate. CPU only.
// QWEN_LEAN_GGUF must load (ROCmFP4 trunk), QWEN_STOCK_GGUF must fail with
// "must be ROCmFP4", QWEN_MTP_GGUF (a qwen35moe file with an MTP block) must
// still load (draft experts stay Q8_0). Exit 0 pass, 1 fail, 77 skipped
// (LEAN/STOCK unset or unreadable; MTP only skips its own case).
#include <cstdlib>
#include <iostream>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/state.hpp"

namespace {

int fails = 0;

void Expect(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL " << msg << "\n";
    ++fails;
  } else {
    std::cout << "PASS " << msg << "\n";
  }
}

bool IsFp4(superroc::core::GgmlType type) {
  return type == superroc::core::GgmlType::kQ4_0_ROCMFP4 ||
         type == superroc::core::GgmlType::kQ4_0_ROCMFP4_FAST;
}

}  // namespace

int main() {
  const char* lean_env = std::getenv("QWEN_LEAN_GGUF");
  const char* stock_env = std::getenv("QWEN_STOCK_GGUF");
  const char* mtp_env = std::getenv("QWEN_MTP_GGUF");
  if (lean_env == nullptr || stock_env == nullptr) {
    std::cerr << "SKIP kat_expert_types_test: QWEN_LEAN_GGUF and "
                 "QWEN_STOCK_GGUF are not both set\n";
    return 77;
  }

  std::string error;
  auto lean_reader = superroc::core::GgufReader::OpenFile(lean_env, &error);
  if (lean_reader == nullptr) {
    std::cerr << "SKIP kat_expert_types_test: " << lean_env
              << " unreadable: " << error << "\n";
    return 77;
  }
  auto stock_reader = superroc::core::GgufReader::OpenFile(stock_env, &error);
  if (stock_reader == nullptr) {
    std::cerr << "SKIP kat_expert_types_test: " << stock_env
              << " unreadable: " << error << "\n";
    return 77;
  }

  error.clear();
  auto lean =
      superroc::models::QwenModelWeights::LoadFromGguf(*lean_reader, &error);
  Expect(lean.has_value(), "lean loads: " + error);
  if (lean.has_value() && !lean->layers.empty()) {
    Expect(IsFp4(lean->layers[0].ffn_gate_exps.type), "lean trunk is ROCmFP4");
  }

  error.clear();
  auto stock =
      superroc::models::QwenModelWeights::LoadFromGguf(*stock_reader, &error);
  Expect(!stock.has_value(), "stock refused");
  Expect(
      !stock.has_value() && error.find("must be ROCmFP4") != std::string::npos,
      "stock error names ROCmFP4: " + error);

  if (mtp_env == nullptr) {
    std::cout << "SKIP mtp case: QWEN_MTP_GGUF is not set\n";
  } else {
    auto mtp_reader = superroc::core::GgufReader::OpenFile(mtp_env, &error);
    if (mtp_reader == nullptr) {
      std::cout << "SKIP mtp case: " << mtp_env << " unreadable: " << error
                << "\n";
    } else {
      error.clear();
      auto mtp =
          superroc::models::QwenModelWeights::LoadFromGguf(*mtp_reader, &error);
      Expect(mtp.has_value(), "mtp loads: " + error);
      if (mtp.has_value()) {
        Expect(mtp->mtp.has_value(), "mtp block bound");
        if (mtp->mtp.has_value()) {
          Expect(mtp->mtp->ffn_gate_exps.type == superroc::core::GgmlType::kQ8_0,
                 "mtp draft experts stay Q8_0");
        }
      }
    }
  }
  return fails == 0 ? 0 : 1;
}
