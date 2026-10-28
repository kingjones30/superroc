// LFM2 config + binder test. CPU only. Plain main, no framework.
// Env: LFM2_A_GGUF (A-LEAN), LFM2_BF16_GGUF, LFM2_Q4KM_GGUF, QWEN_GGUF.
// Exit 0 pass, 1 fail, 77 skipped (an env var is unset).

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/state.hpp"

namespace {

int g_fail = 0;
std::vector<std::unique_ptr<superroc::core::GgufReader>> g_readers;

void Check(bool ok, const std::string& name, const std::string& why = "") {
  if (ok) {
    std::cout << "PASS " << name << '\n';
  } else {
    std::cout << "FAIL " << name << ": " << why << '\n';
    ++g_fail;
  }
}

const superroc::core::GgufReader* Open(const std::string& path,
                                       std::string* err,
                                       const std::string& tag) {
  auto reader = superroc::core::GgufReader::OpenFile(path, err);
  if (reader == nullptr) {
    Check(false, tag, path + " unreadable: " + *err);
    return nullptr;
  }
  g_readers.push_back(std::move(reader));
  return g_readers.back().get();
}

}  // namespace

int main() {
  const char* a_path = std::getenv("LFM2_A_GGUF");     // LFM2
  const char* b_path = std::getenv("LFM2_BF16_GGUF");  // LFM2
  const char* q_path = std::getenv("LFM2_Q4KM_GGUF");  // LFM2
  const char* w_path = std::getenv("QWEN_GGUF");       // LFM2
  if (a_path == nullptr || b_path == nullptr || q_path == nullptr ||
      w_path == nullptr) {
    std::cout << "SKIP lfm2_config_test: LFM2_A_GGUF, LFM2_BF16_GGUF, "
                 "LFM2_Q4KM_GGUF and QWEN_GGUF must all be set\n";
    return 77;
  }

  // ---- A-LEAN: full config + bind ----
  {
    std::string err;
    const auto* reader = Open(a_path, &err, "A-open");
    if (reader != nullptr) {
      const auto config = reader->ExtractModelConfig(&err);
      Check(config.has_value(), "A-config", err);
      if (config.has_value()) {
        const auto& c = *config;
        Check(c.lfm2, "A-lfm2-flag", "lfm2 is false");
        const bool geom =
            c.num_layers == 40 && c.hidden_size == 2048 &&
            c.num_attention_heads == 32 && c.num_key_value_heads == 8 &&
            c.head_dim == 64 && c.intermediate_size == 11776 &&
            c.expert_count == 64 && c.expert_used_count == 4 &&
            c.expert_feed_forward_length == 1536 &&
            c.expert_shared_feed_forward_length == 0 &&
            c.dense_lead_layers == 2 && c.shortconv_l_cache == 3 &&
            c.moe_sigmoid_bias && c.vocab_size == 128000;
        Check(geom, "A-geometry",
              "layers=" + std::to_string(c.num_layers) +
                  " hidden=" + std::to_string(c.hidden_size) +
                  " heads=" + std::to_string(c.num_attention_heads) +
                  " kv=" + std::to_string(c.num_key_value_heads) +
                  " headdim=" + std::to_string(c.head_dim) +
                  " inter=" + std::to_string(c.intermediate_size) +
                  " experts=" + std::to_string(c.expert_count) + "/" +
                  std::to_string(c.expert_used_count) + "/" +
                  std::to_string(c.expert_feed_forward_length) + "/" +
                  std::to_string(c.expert_shared_feed_forward_length) +
                  " dense_lead=" + std::to_string(c.dense_lead_layers) +
                  " lcache=" + std::to_string(c.shortconv_l_cache) +
                  " sig=" + std::to_string(c.moe_sigmoid_bias) +
                  " vocab=" + std::to_string(c.vocab_size));
        Check(std::fabs(c.norm_eps_pre - 1.0e-5F) < 1.0e-9F, "A-eps",
              std::to_string(c.norm_eps_pre));
        Check(std::fabs(c.rope_theta - 1.0e7F) / 1.0e7F < 1.0e-6F,
              "A-rope-theta", std::to_string(c.rope_theta));
        Check(c.FullAttentionLayerCount() == 10, "A-attn-count",
              std::to_string(c.FullAttentionLayerCount()));
        Check(c.SsmLayerCount() == 30, "A-ssm-count",
              std::to_string(c.SsmLayerCount()));
        bool table_ok = true;
        for (std::uint32_t i = 0; i < 40; ++i) {
          const bool want_attn =
              (i == 2 || i == 6 || i == 10 || i == 14 || i == 18 || i == 22 ||
               i == 26 || i == 30 || i == 34 || i == 38);
          if ((c.lfm2_attn[i] != 0) != want_attn) {
            table_ok = false;
          }
        }
        Check(table_ok, "A-lfm2-table", "attn pattern mismatch");
        Check(c.Lfm2KindIndex(2) == 0 && c.Lfm2KindIndex(38) == 9 &&
                  c.Lfm2KindIndex(0) == 0 && c.Lfm2KindIndex(1) == 1 &&
                  c.Lfm2KindIndex(3) == 2 && c.Lfm2KindIndex(39) == 29,
              "A-kind-index",
              std::to_string(c.Lfm2KindIndex(2)) + "," +
                  std::to_string(c.Lfm2KindIndex(38)) + "," +
                  std::to_string(c.Lfm2KindIndex(0)) + "," +
                  std::to_string(c.Lfm2KindIndex(1)) + "," +
                  std::to_string(c.Lfm2KindIndex(3)) + "," +
                  std::to_string(c.Lfm2KindIndex(39)));
        Check(c.Lfm2ConvStateFloats() == 4096, "A-conv-floats",
              std::to_string(c.Lfm2ConvStateFloats()));
        Check(c.IsValidQwen(), "A-valid", "IsValidQwen false");
      }
      auto weights =
          superroc::models::QwenModelWeights::LoadFromGguf(*reader, &err);
      Check(weights.has_value(), "A-bind", err);
      if (weights.has_value()) {
        Check(weights->output_tied_to_token_embd, "A-tied", "head is not tied");
        const auto& l0 = weights->layers[0];
        const auto& l1 = weights->layers[1];
        Check(!l0.is_moe && !l1.is_moe && !l0.ffn_down.empty() &&
                  !l1.ffn_down.empty(),
              "A-dense-layers", "layers 0/1 are not dense SwiGLU");
        const auto& l2 = weights->layers[2];
        Check(l2.is_full_attention && l2.is_moe &&
                  l2.exp_probs_b.num_elements == 64,
              "A-layer2",
              "is_full=" + std::to_string(l2.is_full_attention) +
                  " is_moe=" + std::to_string(l2.is_moe) +
                  " probs=" + std::to_string(l2.exp_probs_b.num_elements));
        const auto& l3 = weights->layers[3];
        Check(l3.is_shortconv && l3.is_moe &&
                  l3.shortconv_conv.type == superroc::core::GgmlType::kF32 &&
                  l3.shortconv_conv.num_elements == 3U * 2048U,
              "A-layer3",
              "is_conv=" + std::to_string(l3.is_shortconv) +
                  " is_moe=" + std::to_string(l3.is_moe) +
                  " elems=" + std::to_string(l3.shortconv_conv.num_elements));
        std::uint32_t n_conv = 0;
        for (const auto& l : weights->layers) {
          n_conv += l.is_shortconv ? 1U : 0U;
        }
        Check(n_conv == 30, "A-conv-count", std::to_string(n_conv));
      }
    }
  }

  // ---- BF16 source: config ok, bind must refuse non-FP4 experts ----
  {
    std::string err;
    const auto* reader = Open(b_path, &err, "B-open");
    if (reader != nullptr) {
      const auto config = reader->ExtractModelConfig(&err);
      Check(config.has_value(), "B-config", err);
      auto weights =
          superroc::models::QwenModelWeights::LoadFromGguf(*reader, &err);
      Check(!weights.has_value() &&
                err.find("must be ROCmFP4") != std::string::npos,
            "B-bind-fails-fp4", err);
    }
  }

  // ---- Stock Q4_K_M: bind must refuse non-FP4 experts ----
  {
    std::string err;
    const auto* reader = Open(q_path, &err, "Q-open");
    if (reader != nullptr) {
      auto weights =
          superroc::models::QwenModelWeights::LoadFromGguf(*reader, &err);
      Check(!weights.has_value() &&
                err.find("must be ROCmFP4") != std::string::npos,
            "Q-bind-fails-fp4", err);
    }
  }

  // ---- Qwen regression (minicpm LEAN): F02/F09 must not break qwen35moe ----
  {
    std::string err;
    const auto* reader = Open(w_path, &err, "QWEN-open");
    if (reader != nullptr) {
      const auto config = reader->ExtractModelConfig(&err);
      Check(config.has_value(), "QWEN-config", err);
      if (config.has_value()) {
        Check(!config->lfm2, "QWEN-not-lfm2", "lfm2 is true");
        Check(config->IsValidQwen(), "QWEN-valid", "IsValidQwen false");
      }
      auto weights =
          superroc::models::QwenModelWeights::LoadFromGguf(*reader, &err);
      Check(weights.has_value(), "QWEN-bind", err);
    }
  }

  if (g_fail == 0) {
    std::cout << "LFM2 PASS\n";
    return 0;
  }
  return 1;
}
