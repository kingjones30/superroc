// KAT-MTP F2: the load-time type gate for the embedded nextn block.
//
// A full QwenMtpGpuModel::CreateFromEmbedded call needs a GPU-resident
// QwenGpuModel (loaded weights + a GGUF), which the CPU test target cannot
// build. This test drives the exact gate function CreateFromEmbedded calls
// (KatMtpValidateEmbeddedBlockTypes) with a synthetic block, so it can sweep
// every tensor type instead of only the one the shipped GGUF happens to carry.
#include "src/models/qwen/hip/mtp/kat_mtp_types.hpp"

#include <iostream>
#include <string>

namespace {

int fails = 0;

void Expect(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "FAIL " << what << "\n";
    ++fails;
  }
}

int dummy = 0;

gufo::models::QwenTensorRef Present(gufo::core::GgmlType type) {
  gufo::models::QwenTensorRef ref;
  ref.data = &dummy;
  ref.type = type;
  ref.num_elements = 1;
  return ref;
}

// The shipped KAT-Coder-V2.5 block: Q8_0 routed and shared experts, Q8_0
// nextn.eh_proj, Q8_0 attention K and V (verified against the GGUF).
gufo::speculative::QwenMtpWeights KatLikeBlock() {
  using gufo::core::GgmlType;
  gufo::speculative::QwenMtpWeights weights;
  weights.config.is_moe = true;
  weights.config.expert_count = 256;
  weights.config.expert_used_count = 8;
  weights.fusion_projection = Present(GgmlType::kQ8_0);
  weights.layer.is_moe = true;
  weights.layer.attn_k = Present(GgmlType::kQ8_0);
  weights.layer.attn_v = Present(GgmlType::kQ8_0);
  weights.layer.ffn_gate_exps = Present(GgmlType::kQ8_0);
  weights.layer.ffn_up_exps = Present(GgmlType::kQ8_0);
  weights.layer.ffn_down_exps = Present(GgmlType::kQ8_0);
  weights.layer.ffn_gate_shexp = Present(GgmlType::kQ8_0);
  weights.layer.ffn_up_shexp = Present(GgmlType::kQ8_0);
  weights.layer.ffn_down_shexp = Present(GgmlType::kQ8_0);
  return weights;
}

}  // namespace

int main() {
  using gufo::core::GgmlType;
  using gufo::hip::KatMtpExpertGemmTypeSupported;
  using gufo::hip::KatMtpPrimeGemmTypeSupported;
  using gufo::hip::KatMtpValidateEmbeddedBlockTypes;

  // The expert GEMV accepts Q8_0 and Q8_K only.
  Expect(KatMtpExpertGemmTypeSupported(GgmlType::kQ8_0), "expert Q8_0");
  Expect(KatMtpExpertGemmTypeSupported(GgmlType::kQ8_K), "expert Q8_K");
  Expect(!KatMtpExpertGemmTypeSupported(GgmlType::kQ4_0),
         "expert Q4_0 refused");
  Expect(!KatMtpExpertGemmTypeSupported(GgmlType::kBF16),
         "expert BF16 refused");
  Expect(!KatMtpExpertGemmTypeSupported(GgmlType::kQ4_0_ROCMFP4),
         "expert ROCMFP4 refused");
  Expect(!KatMtpExpertGemmTypeSupported(GgmlType::kF32), "expert F32 refused");

  // RunPrimeGemm runs Q4_0, Q8_0, Q8_K and BF16; everything else is refused.
  Expect(KatMtpPrimeGemmTypeSupported(GgmlType::kQ4_0), "prime Q4_0");
  Expect(KatMtpPrimeGemmTypeSupported(GgmlType::kQ8_0), "prime Q8_0");
  Expect(KatMtpPrimeGemmTypeSupported(GgmlType::kQ8_K), "prime Q8_K");
  Expect(KatMtpPrimeGemmTypeSupported(GgmlType::kBF16), "prime BF16");
  Expect(!KatMtpPrimeGemmTypeSupported(GgmlType::kF32), "prime F32 refused");
  Expect(!KatMtpPrimeGemmTypeSupported(GgmlType::kF16), "prime F16 refused");
  Expect(!KatMtpPrimeGemmTypeSupported(GgmlType::kQ6_K), "prime Q6_K refused");

  // The shipped KAT block passes, and an accepted block leaves the message be.
  {
    const auto weights = KatLikeBlock();
    std::string error = "sentinel";
    Expect(KatMtpValidateEmbeddedBlockTypes(weights, &error),
           "KAT-like block accepted");
    Expect(error == "sentinel", "accepted block leaves error_msg untouched");
    Expect(KatMtpValidateEmbeddedBlockTypes(weights, nullptr),
           "accepted block tolerates a null error_msg");
  }

  // An unsupported routed-expert type is refused at load, naming the rule.
  {
    auto weights = KatLikeBlock();
    weights.layer.ffn_gate_exps = Present(GgmlType::kQ4_0);
    std::string error;
    Expect(!KatMtpValidateEmbeddedBlockTypes(weights, &error),
           "Q4_0 routed expert refused");
    Expect(error.find("Q8_0") != std::string::npos,
           "routed-expert refusal names Q8_0");
  }
  {
    auto weights = KatLikeBlock();
    weights.layer.ffn_down_exps = Present(GgmlType::kQ4_0_ROCMFP4);
    std::string error;
    Expect(!KatMtpValidateEmbeddedBlockTypes(weights, &error),
           "ROCMFP4 routed expert refused");
  }
  {
    auto weights = KatLikeBlock();
    weights.layer.ffn_up_exps = Present(GgmlType::kBF16);
    std::string error;
    Expect(!KatMtpValidateEmbeddedBlockTypes(weights, &error),
           "BF16 routed expert refused");
  }
  // A shared expert with an unsupported type is refused...
  {
    auto weights = KatLikeBlock();
    weights.layer.ffn_up_shexp = Present(GgmlType::kF16);
    std::string error;
    Expect(!KatMtpValidateEmbeddedBlockTypes(weights, &error),
           "F16 shared expert refused");
  }
  // ...but an entirely absent shared expert is fine.
  {
    auto weights = KatLikeBlock();
    weights.layer.ffn_gate_shexp = gufo::models::QwenTensorRef{};
    weights.layer.ffn_up_shexp = gufo::models::QwenTensorRef{};
    weights.layer.ffn_down_shexp = gufo::models::QwenTensorRef{};
    std::string error;
    Expect(KatMtpValidateEmbeddedBlockTypes(weights, &error),
           "absent shared expert accepted");
  }

  // A prime weight type that RunPrimeGemm cannot run is refused at load.
  {
    auto weights = KatLikeBlock();
    weights.fusion_projection = Present(GgmlType::kF32);
    std::string error;
    Expect(!KatMtpValidateEmbeddedBlockTypes(weights, &error),
           "F32 nextn.eh_proj refused");
    Expect(error.find("BF16") != std::string::npos,
           "prime-weight refusal names BF16");
  }
  {
    auto weights = KatLikeBlock();
    weights.layer.attn_k = Present(GgmlType::kQ6_K);
    std::string error;
    Expect(!KatMtpValidateEmbeddedBlockTypes(weights, &error),
           "Q6_K attn_k refused");
  }
  {
    auto weights = KatLikeBlock();
    weights.layer.attn_v = Present(GgmlType::kIQ4_XS);
    std::string error;
    Expect(!KatMtpValidateEmbeddedBlockTypes(weights, &error),
           "IQ4_XS attn_v refused");
  }

  if (fails != 0) {
    std::cerr << "KAT_MTP_TYPES FAIL " << fails << "\n";
    return 1;
  }
  std::cout << "KAT_MTP_TYPES PASS\n";
  return 0;
}
