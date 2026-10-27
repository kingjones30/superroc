// Flint: bind the COHERENT-PAS GGUF through GgufReader + ExtractModelConfig.
// Asserts config, tensor presence/shape/type, and prints per-type byte totals.
// Exits 77 (SKIP) when the GGUF is absent.

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/model_config.hpp"
#include "src/models/qwen/state.hpp"

namespace {

namespace fs = std::filesystem;

void Fail(const std::string& msg) {
  std::cerr << "FAIL flint_bind_test: " << msg << "\n";
  std::exit(1);
}

void Expect(bool ok, const std::string& msg) {
  if (!ok) {
    Fail(msg);
  }
}

const superroc::core::GgufTensorInfo* Require(const superroc::core::GgufReader& reader,
                                          const std::string& name) {
  const auto* t = reader.FindTensor(name);
  if (t == nullptr) {
    Fail("missing tensor " + name);
  }
  return t;
}

void ExpectDims(const superroc::core::GgufTensorInfo& t,
                const std::vector<std::uint64_t>& dims) {
  if (t.dimensions != dims) {
    std::string got = t.name.data();
    got += " dims {";
    for (std::size_t i = 0; i < t.dimensions.size(); ++i) {
      if (i) {
        got += ",";
      }
      got += std::to_string(t.dimensions[i]);
    }
    got += "} expected {";
    for (std::size_t i = 0; i < dims.size(); ++i) {
      if (i) {
        got += ",";
      }
      got += std::to_string(dims[i]);
    }
    got += "}";
    Fail(got);
  }
}

void ExpectType(const superroc::core::GgufTensorInfo& t, superroc::core::GgmlType ty) {
  if (t.type != ty) {
    Fail(std::string(t.name) + " type " +
         std::string(superroc::core::ToString(t.type)) + " expected " +
         std::string(superroc::core::ToString(ty)));
  }
}

constexpr std::array<int, 14> kFull = {0,  1,  5,  9,  13, 17, 21,
                                       25, 29, 33, 37, 41, 45, 48};

}  // namespace

int main() {
  const char* gguf_env = std::getenv("FLINT_GGUF");
  fs::path gguf;
  if (gguf_env != nullptr && gguf_env[0] != '\0') {
    gguf = fs::path(gguf_env);
  } else if (const char* home = std::getenv("HOME");
             home != nullptr && home[0] != '\0') {
    gguf = fs::path(home) / "models" / "flint" /
           "Flint-40B-A2.5B-Q4_0-ROCmFP4-COHERENT-PAS.gguf";
  }
  if (gguf.empty() || !fs::exists(gguf)) {
    std::cout << "SKIP flint_bind_test: GGUF is absent\n";
    return 77;
  }

  std::string error;
  const auto reader = superroc::core::GgufReader::OpenFile(gguf, &error);
  Expect(reader != nullptr, "open GGUF: " + error);
  const auto cfg_opt = reader->ExtractModelConfig(&error);
  Expect(cfg_opt.has_value(), "ExtractModelConfig: " + error);
  const auto& cfg = *cfg_opt;

  Expect(cfg.architecture == "flint", "arch flint");
  Expect(cfg.flint, "flint flag");
  Expect(cfg.attn_sinks, "attn_sinks");
  Expect(cfg.moe_sigmoid_bias, "moe_sigmoid_bias");
  Expect(cfg.num_layers == 49, "49 layers");
  Expect(cfg.hidden_size == 2048, "hidden 2048");
  Expect(cfg.intermediate_size == 6144, "ff 6144");
  Expect(cfg.num_attention_heads == 32, "32 heads");
  Expect(cfg.num_key_value_heads == 4, "4 kv");
  Expect(cfg.head_dim == 128, "head_dim 128");
  Expect(cfg.rotary_dim == 128, "rotary_dim 128");
  Expect(cfg.swa_window == 513, "window 513");
  Expect(cfg.dense_lead_layers == 2, "dense lead 2");
  Expect(cfg.expert_count == 256, "256 experts");
  Expect(cfg.expert_used_count == 6, "top-6");
  Expect(cfg.expert_feed_forward_length == 512, "expert ff 512");
  Expect(cfg.expert_shared_feed_forward_length == 1024, "shexp ff 1024");
  Expect(cfg.vocab_size == 150144, "vocab 150144");
  Expect(cfg.rope_variant == superroc::core::RopeVariant::kNeox, "kNeox");
  Expect(cfg.IsValidQwen(), "IsValidQwen");

  std::uint32_t n_full = 0;
  std::uint32_t n_slide = 0;
  for (std::uint32_t i = 0; i < cfg.num_layers; ++i) {
    if (cfg.use_rope[i] == 0) {
      ++n_full;
    } else {
      ++n_slide;
    }
  }
  Expect(n_full == 14 && n_slide == 35,
         "14 full / 35 sliding, got full=" + std::to_string(n_full) +
             " slide=" + std::to_string(n_slide));
  for (int layer : kFull) {
    Expect(cfg.use_rope[static_cast<std::size_t>(layer)] == 0,
           "full layer " + std::to_string(layer) + " must be NoPE");
  }

  using superroc::core::GgmlType;
  const auto* embd = Require(*reader, "token_embd.weight");
  ExpectDims(*embd, {2048, 150144});
  ExpectType(*embd, GgmlType::kQ8_0);
  const auto* onorm = Require(*reader, "output_norm.weight");
  ExpectDims(*onorm, {2048});
  ExpectType(*onorm, GgmlType::kF32);
  const auto* output = Require(*reader, "output.weight");
  ExpectDims(*output, {2048, 150144});
  ExpectType(*output, GgmlType::kQ8_0);

  for (std::uint32_t i = 0; i < 49; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    ExpectDims(*Require(*reader, p + "attn_norm.weight"), {2048});
    ExpectType(*Require(*reader, p + "attn_norm.weight"), GgmlType::kF32);
    ExpectDims(*Require(*reader, p + "ffn_norm.weight"), {2048});
    ExpectType(*Require(*reader, p + "ffn_norm.weight"), GgmlType::kF32);
    ExpectDims(*Require(*reader, p + "attn_q.weight"), {2048, 4096});
    ExpectType(*Require(*reader, p + "attn_q.weight"), GgmlType::kQ8_0);
    ExpectDims(*Require(*reader, p + "attn_k.weight"), {2048, 512});
    ExpectType(*Require(*reader, p + "attn_k.weight"), GgmlType::kQ8_0);
    ExpectDims(*Require(*reader, p + "attn_v.weight"), {2048, 512});
    ExpectType(*Require(*reader, p + "attn_v.weight"), GgmlType::kQ8_0);
    ExpectDims(*Require(*reader, p + "attn_output.weight"), {4096, 2048});
    ExpectType(*Require(*reader, p + "attn_output.weight"), GgmlType::kQ8_0);
    ExpectDims(*Require(*reader, p + "attn_q_norm.weight"), {128});
    ExpectType(*Require(*reader, p + "attn_q_norm.weight"), GgmlType::kF32);
    ExpectDims(*Require(*reader, p + "attn_k_norm.weight"), {128});
    ExpectType(*Require(*reader, p + "attn_k_norm.weight"), GgmlType::kF32);
    ExpectDims(*Require(*reader, p + "attn_sinks"), {32});
    ExpectType(*Require(*reader, p + "attn_sinks"), GgmlType::kF32);
    if (i < 2) {
      ExpectDims(*Require(*reader, p + "ffn_gate.weight"), {2048, 6144});
      ExpectType(*Require(*reader, p + "ffn_gate.weight"), GgmlType::kQ8_0);
      ExpectDims(*Require(*reader, p + "ffn_up.weight"), {2048, 6144});
      ExpectType(*Require(*reader, p + "ffn_up.weight"), GgmlType::kQ8_0);
      ExpectDims(*Require(*reader, p + "ffn_down.weight"), {6144, 2048});
      ExpectType(*Require(*reader, p + "ffn_down.weight"), GgmlType::kQ8_0);
    } else {
      ExpectDims(*Require(*reader, p + "ffn_gate_inp.weight"), {2048, 256});
      ExpectType(*Require(*reader, p + "ffn_gate_inp.weight"), GgmlType::kF32);
      ExpectDims(*Require(*reader, p + "exp_probs_b.bias"), {256});
      ExpectType(*Require(*reader, p + "exp_probs_b.bias"), GgmlType::kF32);
      ExpectDims(*Require(*reader, p + "ffn_gate_exps.weight"),
                 {2048, 512, 256});
      ExpectType(*Require(*reader, p + "ffn_gate_exps.weight"),
                 GgmlType::kQ4_0_ROCMFP4);
      ExpectDims(*Require(*reader, p + "ffn_up_exps.weight"), {2048, 512, 256});
      ExpectType(*Require(*reader, p + "ffn_up_exps.weight"),
                 GgmlType::kQ4_0_ROCMFP4);
      ExpectDims(*Require(*reader, p + "ffn_down_exps.weight"),
                 {512, 2048, 256});
      ExpectType(*Require(*reader, p + "ffn_down_exps.weight"),
                 GgmlType::kQ4_0_ROCMFP4);
      ExpectDims(*Require(*reader, p + "ffn_gate_shexp.weight"), {2048, 1024});
      ExpectType(*Require(*reader, p + "ffn_gate_shexp.weight"),
                 GgmlType::kQ8_0);
      ExpectDims(*Require(*reader, p + "ffn_up_shexp.weight"), {2048, 1024});
      ExpectType(*Require(*reader, p + "ffn_up_shexp.weight"), GgmlType::kQ8_0);
      ExpectDims(*Require(*reader, p + "ffn_down_shexp.weight"), {1024, 2048});
      ExpectType(*Require(*reader, p + "ffn_down_shexp.weight"),
                 GgmlType::kQ8_0);
    }
  }

  Expect(reader->GetTensorCount() == 826,
         "tensor count " + std::to_string(reader->GetTensorCount()));

  std::map<std::string, std::uint64_t> bytes;
  std::map<std::string, std::uint64_t> count;
  for (const auto& t : reader->GetTensors()) {
    const std::string ty{superroc::core::ToString(t.type)};
    bytes[ty] += t.size_bytes;
    count[ty] += 1;
  }
  std::cout << "flint_bind_test tensors=" << reader->GetTensorCount()
            << " layers=49 full=14 sliding=35 window=513 experts=256x6 "
               "dense_lead=2\n";
  for (const auto& [ty, n] : count) {
    std::cout << "type " << ty << " tensors=" << n << " bytes=" << bytes[ty]
              << "\n";
  }
  std::string bind_error;
  const auto weights_opt =
      superroc::models::QwenModelWeights::LoadFromGguf(*reader, &bind_error);
  Expect(weights_opt.has_value(), "LoadFromGguf: " + bind_error);
  Expect(weights_opt->layers.size() == 49, "LoadFromGguf 49 layers");
  for (std::uint32_t i = 0; i < 49; ++i) {
    const auto& layer = weights_opt->layers[i];
    Expect(layer.is_full_attention,
           "layer " + std::to_string(i) + " is_full_attention");
    Expect(!layer.attn_sinks.empty() &&
               layer.attn_sinks.type == GgmlType::kF32 &&
               layer.attn_sinks.num_elements == 32,
           "layer " + std::to_string(i) + " attn_sinks F32[32]");
    if (i < 2) {
      Expect(!layer.is_moe, "layer " + std::to_string(i) + " dense");
    } else {
      Expect(layer.is_moe, "layer " + std::to_string(i) + " moe");
      Expect(!layer.ffn_gate_shexp.empty() && !layer.ffn_up_shexp.empty() &&
                 !layer.ffn_down_shexp.empty(),
             "layer " + std::to_string(i) + " shexp refs");
    }
  }

  std::cout << "LoadFromGguf ok layers=" << weights_opt->layers.size()
            << " dense=0-1 moe=2-48 sinks=49 F32 shexp=47\n";
  std::cout << "PASS flint_bind_test\n";
  return 0;
}
