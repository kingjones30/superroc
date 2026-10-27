#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/moe/dispatch.hpp"
#include "src/models/qwen/state.hpp"

namespace {

void Expect(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL " << msg << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  using superroc::models::qwen::ClassifyKatDispatch;
  using superroc::models::qwen::KatDispatch;
  using superroc::models::qwen::KatSpecKind;

  Expect(ClassifyKatDispatch("qwen35", KatSpecKind::kOff, true, false) ==
             KatDispatch::kNotKat,
         "dense arch is not KAT");
  Expect(ClassifyKatDispatch("qwen35moe", KatSpecKind::kOff, true, false) ==
             KatDispatch::kAccept,
         "spec-off accepted");
  Expect(ClassifyKatDispatch("qwen35moe", KatSpecKind::kMtp, true, false) ==
             KatDispatch::kAccept,
         "embedded MTP accepted");
  Expect(ClassifyKatDispatch("qwen35moe", KatSpecKind::kMtp, false, false) ==
             KatDispatch::kRefuseSidecar,
         "sidecar MTP refused");
  Expect(ClassifyKatDispatch("qwen35moe", KatSpecKind::kDSpark, true, false) ==
             KatDispatch::kRefuseDSpark,
         "DSpark refused");
  Expect(ClassifyKatDispatch("qwen35moe", KatSpecKind::kDFlash, true, false) ==
             KatDispatch::kRefuseOther,
         "DFlash refused");
  Expect(ClassifyKatDispatch("qwen35moe", KatSpecKind::kOff, true, true) ==
             KatDispatch::kRefuseVision,
         "vision refused");

  const char* path_env = std::getenv("KAT_GGUF");
  const std::string path =
      path_env != nullptr ? path_env
                          : "/path/to/models/swap/kat-coder/"
                            "KAT-Coder-V2.5-Dev-Q4_0_ROCMFP4_STRIX-MTP.gguf";
  if (!std::filesystem::exists(path)) {
    std::cout << "SKIP: KAT GGUF not found: " << path << " (set KAT_GGUF)\n";
    return 77;
  }
  std::string error;
  auto reader = superroc::core::GgufReader::OpenFile(path, &error);
  Expect(reader != nullptr, "open GGUF: " + error);

  auto weights = superroc::models::QwenModelWeights::LoadFromGguf(*reader, &error);
  Expect(weights.has_value(), "bind: " + error);
  const auto& cfg = weights->config;
  Expect(cfg.architecture == "qwen35moe", "arch");
  Expect(cfg.is_moe, "is_moe");
  Expect(cfg.num_layers == 40, "trunk layers");
  Expect(cfg.mtp_num_layers == 1, "mtp layers");
  Expect(cfg.hidden_size == 2048, "hidden");
  Expect(cfg.num_attention_heads == 16, "heads");
  Expect(cfg.num_key_value_heads == 2, "kv heads");
  Expect(cfg.head_dim == 256, "head dim");
  Expect(cfg.rotary_dim == 64, "rotary");
  Expect(cfg.vocab_size == 248320, "vocab");
  Expect(cfg.expert_count == 256, "experts");
  Expect(cfg.expert_used_count == 8, "top-k");
  Expect(cfg.expert_feed_forward_length == 512, "expert ff");
  Expect(cfg.expert_shared_feed_forward_length == 512, "shexp ff");
  Expect(cfg.full_attention_interval == 4, "interval");
  Expect(cfg.ssm_conv_kernel == 4 && cfg.ssm_state_size == 128 &&
             cfg.ssm_group_count == 16 && cfg.ssm_time_step_rank == 32 &&
             cfg.ssm_inner_size == 4096,
         "ssm hparams");
  Expect(cfg.IsValidQwen(), "IsValidQwen");
  Expect(
      !reader->GetMetadataUint32("qwen35moe.feed_forward_length").has_value(),
      "dense ff key absent");
  Expect(!reader->GetMetadataUint32("qwen35moe.expert_group_count").has_value(),
         "no expert groups");

  Expect(weights->layers.size() == 40, "trunk vector excludes MTP");
  Expect(weights->mtp.has_value(), "MTP bound");
  Expect(weights->mtp_block_index == 40, "MTP index");
  Expect(weights->mtp_head_tied_to_output, "tied MTP head");
  Expect(weights->mtp_embed_tied_to_token, "tied MTP embed");
  Expect(!weights->output_tied_to_token_embd, "trunk head is its own tensor");
  Expect(weights->output.type == superroc::core::GgmlType::kQ4_0_ROCMFP4_FAST,
         "output type 101");
  Expect(weights->token_embd.type == superroc::core::GgmlType::kQ6_K,
         "token embd type 14");
  Expect(weights->output.data != weights->token_embd.data, "head not embd");
  Expect(reader->FindTensor("blk.40.nextn.embed_tokens.weight") == nullptr,
         "no nextn embed");
  Expect(reader->FindTensor("blk.40.nextn.shared_head_head.weight") == nullptr,
         "no nextn head");

  int n_attn = 0;
  int n_ssm = 0;
  for (std::uint32_t i = 0; i < weights->layers.size(); ++i) {
    const auto& layer = weights->layers[i];
    const bool expect_attn = ((i + 1) % 4) == 0;
    Expect(layer.is_full_attention == expect_attn,
           "hybrid map " + std::to_string(i));
    Expect(layer.is_moe, "trunk moe " + std::to_string(i));
    Expect(layer.nextn_eh_proj.empty(), "trunk has no nextn");
    Expect(layer.ffn_gate_inp.type == superroc::core::GgmlType::kF32, "router f32");
    Expect(layer.ffn_gate_inp.num_elements == 2048U * 256U, "router numel");
    Expect(layer.ffn_gate_inp_shexp.type == superroc::core::GgmlType::kF32,
           "shexp gate f32");
    Expect(layer.ffn_gate_inp_shexp.num_elements == 2048, "shexp gate numel");
    Expect(layer.ffn_gate_exps.type == superroc::core::GgmlType::kQ4_0_ROCMFP4_FAST,
           "expert type 101");
    Expect(layer.ffn_down_exps.type == superroc::core::GgmlType::kQ4_0_ROCMFP4_FAST,
           "down expert type");
    Expect(
        layer.ffn_gate_shexp.type == superroc::core::GgmlType::kQ4_0_ROCMFP4_FAST,
        "shexp type 101");
    Expect(layer.ffn_gate.empty() && layer.ffn_up.empty() &&
               layer.ffn_down.empty(),
           "no dense ffn");
    if (expect_attn) {
      ++n_attn;
    } else {
      ++n_ssm;
    }
  }
  Expect(n_attn == 10 && n_ssm == 30, "30 SSM + 10 attention");

  const auto* gate0 = reader->FindTensor("blk.0.ffn_gate_exps.weight");
  Expect(gate0 != nullptr && gate0->dimensions.size() == 3, "gate dims");
  Expect(gate0->dimensions[0] == 2048 && gate0->dimensions[1] == 512 &&
             gate0->dimensions[2] == 256,
         "gate shape");
  const auto* down0 = reader->FindTensor("blk.0.ffn_down_exps.weight");
  Expect(down0 != nullptr && down0->dimensions[0] == 512 &&
             down0->dimensions[1] == 2048 && down0->dimensions[2] == 256,
         "down shape");
  const auto* router0 = reader->FindTensor("blk.0.ffn_gate_inp.weight");
  Expect(router0 != nullptr && router0->dimensions.size() == 2 &&
             router0->dimensions[0] == 2048 && router0->dimensions[1] == 256,
         "router shape");

  const auto& mtp = *weights->mtp;
  Expect(mtp.is_full_attention && mtp.is_moe, "MTP is full-attn MoE");
  Expect(mtp.ffn_gate_exps.type == superroc::core::GgmlType::kQ8_0,
         "MTP experts Q8_0");
  Expect(mtp.ffn_gate_inp.type == superroc::core::GgmlType::kF32, "MTP router F32");
  Expect(mtp.ffn_gate_inp_shexp.type == superroc::core::GgmlType::kF32,
         "MTP shexp gate");
  Expect(mtp.nextn_eh_proj.type == superroc::core::GgmlType::kQ8_0, "eh_proj Q8_0");
  Expect(mtp.nextn_eh_proj.num_elements == 4096U * 2048U, "eh_proj numel");
  Expect(mtp.nextn_enorm.type == superroc::core::GgmlType::kF32 &&
             mtp.nextn_hnorm.type == superroc::core::GgmlType::kF32 &&
             mtp.nextn_shared_head_norm.type == superroc::core::GgmlType::kF32,
         "MTP norms F32");
  Expect(mtp.nextn_eh_proj.data != weights->layers[0].ffn_gate_inp.data,
         "MTP storage is not a trunk layer");

  // Dormant: the trunk vector stops at 39. blk.40 is only in weights.mtp.
  Expect(weights->layers.size() == cfg.num_layers, "dormant trunk size");
  Expect(weights->mtp_block_index == weights->layers.size(),
         "dormant MTP sits one past the trunk");

  std::cout << "KAT_BIND PASS layers=40 attn=" << n_attn << " ssm=" << n_ssm
            << " mtp_blk=" << weights->mtp_block_index
            << " experts=256 topk=8 ff=512 type_trunk=101 type_mtp=8"
            << " tied_head=1 tied_embd=1 output_type=101\n";
  return 0;
}
