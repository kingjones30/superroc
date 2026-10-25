// LLAMA: read the Nanbeige4.3-33B-Base GGUF header through GgufReader and
// assert every architectural value the runtime depends on. Exits 77 (SKIP)
// when NANBEIGE_GGUF is unset or the file is absent.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/model_config.hpp"

namespace {

namespace fs = std::filesystem;

void Fail(const std::string& msg) {
  std::cerr << "FAIL llama_config_test: " << msg << "\n";
  std::exit(1);
}

void Expect(bool ok, const std::string& msg) {
  if (!ok) {
    Fail(msg);
  }
}

template<typename T>
void ExpectEq(const T& got, const T& want, const std::string& what) {
  if (!(got == want)) {
    Fail(what + ": got " + std::to_string(got) + " want " +
         std::to_string(want));
  }
}

}  // namespace

int main() {
  const char* gguf_env = std::getenv("NANBEIGE_GGUF");
  if (gguf_env == nullptr || gguf_env[0] == '\0') {
    std::cout << "SKIP llama_config_test: NANBEIGE_GGUF is not set\n";
    return 77;
  }
  const fs::path gguf{gguf_env};
  if (!fs::exists(gguf)) {
    std::cout << "SKIP llama_config_test: " << gguf << " is absent\n";
    return 77;
  }

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(gguf, &error);
  Expect(reader != nullptr, "open GGUF: " + error);

  const auto cfg_opt = reader->ExtractModelConfig(&error);
  Expect(cfg_opt.has_value(), "ExtractModelConfig: " + error);
  const auto& cfg = *cfg_opt;

  Expect(cfg.architecture == "llama", "arch llama, got " + cfg.architecture);
  Expect(cfg.llama, "llama flag");
  Expect(!cfg.flint, "flint flag stays false");
  Expect(!cfg.lfm2, "lfm2 flag stays false");
  Expect(!cfg.attention_only, "attention_only stays false");
  Expect(!cfg.is_moe, "dense (is_moe false)");
  Expect(cfg.is_text_only, "text-only");

  ExpectEq<std::uint32_t>(cfg.num_layers, 52, "block_count");
  ExpectEq<std::uint32_t>(cfg.hidden_size, 7168, "embedding_length");
  ExpectEq<std::uint32_t>(cfg.intermediate_size, 22016, "feed_forward_length");
  ExpectEq<std::uint32_t>(cfg.num_attention_heads, 56, "head_count");
  ExpectEq<std::uint32_t>(cfg.num_key_value_heads, 8, "head_count_kv");
  ExpectEq<std::uint32_t>(cfg.head_dim, 128, "key/value length");
  ExpectEq<std::uint32_t>(cfg.rotary_dim, 128, "rope dimension_count");
  ExpectEq<std::uint32_t>(cfg.context_length, 131072, "context_length");
  ExpectEq<std::uint32_t>(cfg.vocab_size, 166912, "vocab_size");
  ExpectEq<std::uint32_t>(cfg.mtp_num_layers, 0, "mtp layers");
  ExpectEq<std::uint32_t>(cfg.full_attention_interval, 1,
                          "full_attention_interval");

  Expect(cfg.rope_theta == 10000000.0F,
         "rope.freq_base got " + std::to_string(cfg.rope_theta));
  Expect(cfg.norm_eps_pre == 1.0e-5F,
         "layer_norm_rms_epsilon got " + std::to_string(cfg.norm_eps_pre));
  Expect(cfg.rope_variant == gufo::core::RopeVariant::kNorm,
         "adjacent-pair (NORM) RoPE");
  Expect(cfg.attn_gate_mode == gufo::core::AttnGateMode::kFused,
         "fused attention gate mode");
  Expect(cfg.AttentionSize() == 56U * 128U, "attention size 7168");
  Expect(cfg.FullAttentionLayerCount() == 52U, "every layer full attention");
  Expect(cfg.swa_window == 0U, "no sliding window");
  Expect(!cfg.attn_sinks, "no attention sinks");
  for (std::uint32_t layer = 0; layer < cfg.num_layers; ++layer) {
    Expect(cfg.LayerUsesRope(layer),
           "layer " + std::to_string(layer) + " uses RoPE");
    Expect(cfg.LayerSwaWindow(layer) == 0U,
           "layer " + std::to_string(layer) + " is unwindowed");
  }
  Expect(cfg.IsValidQwen(), "IsValidQwen");

  Expect(!reader->HasMtpTensors(), "no MTP tensors");
  Expect(!reader->HasVisionTensors(), "no vision tensors");

  // Tokenizer contract (T2 consumes it at runtime).
  const auto model = reader->GetMetadataString("tokenizer.ggml.model");
  Expect(model.has_value() && *model == "llama", "tokenizer.ggml.model llama");
  const auto pre = reader->GetMetadataString("tokenizer.ggml.pre");
  Expect(pre.has_value() && *pre == "default", "tokenizer.ggml.pre default");
  ExpectEq<std::size_t>(
      reader->GetMetadataStringArray("tokenizer.ggml.tokens").size(),
      cfg.vocab_size, "tokenizer.ggml.tokens length");
  const auto* scores = reader->FindMetadata("tokenizer.ggml.scores");
  const auto* score_values =
      scores != nullptr ? std::get_if<std::vector<double>>(&scores->value)
                        : nullptr;
  Expect(score_values != nullptr && score_values->size() == cfg.vocab_size,
         "tokenizer.ggml.scores length");
  ExpectEq<std::size_t>(
      reader->GetMetadataUintArray("tokenizer.ggml.token_type").size(),
      cfg.vocab_size, "tokenizer.ggml.token_type length");
  const auto bos = reader->GetMetadataUint32("tokenizer.ggml.bos_token_id");
  const auto eos = reader->GetMetadataUint32("tokenizer.ggml.eos_token_id");
  const auto pad = reader->GetMetadataUint32("tokenizer.ggml.padding_token_id");
  Expect(bos.has_value() && *bos == 1U, "bos 1");
  Expect(eos.has_value() && *eos == 2U, "eos 2");
  Expect(pad.has_value() && *pad == 0U, "pad 0");
  const auto add_bos = reader->GetMetadataBool("tokenizer.ggml.add_bos_token");
  const auto add_eos = reader->GetMetadataBool("tokenizer.ggml.add_eos_token");
  Expect(add_bos.has_value() && *add_bos, "add_bos true");
  Expect(add_eos.has_value() && !*add_eos, "add_eos false");
  const auto chat = reader->GetMetadataString("tokenizer.chat_template");
  Expect(chat.has_value() &&
             *chat ==
                 "{% for message in messages %}{% if loop.first %}<s>{% endif "
                 "%}{{ message['content'] }}{% endfor %}",
         "chat template (BOS + concatenated contents)");

  ExpectEq<std::uint64_t>(reader->GetTensorCount(), 471, "tensor count");

  std::cout << "llama_config_test arch=" << cfg.architecture
            << " layers=" << cfg.num_layers << " hidden=" << cfg.hidden_size
            << " ff=" << cfg.intermediate_size
            << " heads=" << cfg.num_attention_heads
            << " kv=" << cfg.num_key_value_heads << " head_dim=" << cfg.head_dim
            << " rope=" << cfg.rotary_dim << " theta=" << cfg.rope_theta
            << " eps=" << cfg.norm_eps_pre << " ctx=" << cfg.context_length
            << " vocab=" << cfg.vocab_size
            << " tensors=" << reader->GetTensorCount() << "\n";
  std::cout << "PASS llama_config_test\n";
  return 0;
}
