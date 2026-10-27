// LLAMA: read the Nanbeige4.3-33B-Base GGUF header through GgufReader and
// assert every architectural value the runtime depends on. Exits 77 (SKIP)
// when NANBEIGE_GGUF is unset or the file is absent.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/model_config.hpp"
#include "src/models/qwen/hip/executor.hpp"

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

namespace {

// F3: a llama QwenGpuModel carries no Qwen tokenizer. GetTokenizer() must
// throw std::logic_error instead of dereferencing null (previously UB).
// Runs without NANBEIGE_GGUF.
void TestLlamaGetTokenizerThrows() {
  superroc::models::QwenModelWeights weights;
  weights.config.llama = true;
  const superroc::hip::QwenGpuModel model(nullptr, std::move(weights), nullptr, {},
                                      {}, nullptr);
  bool threw = false;
  try {
    model.GetTokenizer();
  } catch (const std::logic_error&) {
    threw = true;
  }
  Expect(threw, "llama QwenGpuModel::GetTokenizer must throw");
}

// F4: QwenGpuExecutor::Create must resolve the llama fp32-KV policy
// centrally (serve, prompt and bench all create through here). Runs without
// NANBEIGE_GGUF and without a GPU: it exercises ApplyLlamaExecutionPolicy,
// the exact override Create applies, since building an executor needs a fully
// loaded model.
void TestLlamaCreateResolvesFp32KvPolicy() {
  auto policy = superroc::hip::QwenExecutionPolicy::Production();
  superroc::hip::ApplyLlamaExecutionPolicy(policy);
  Expect(!policy.UsesFp16AttentionKv(), "llama policy keeps fp32 KV cache");
  Expect(!policy.fuse_qk_norm_rope_kv, "llama policy unfused QK-RoPE-KV");
  Expect(policy.glimmer_fp16_kv, "llama policy glimmer fp16 KV plane");
}

}  // namespace

// Minimal metadata-only llama GGUF writer for the F9/F11 config tests.
void AppendMetaPod(std::vector<std::uint8_t>& buf, std::uint64_t value,
                   std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    buf.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  }
}

void AppendMetaString(std::vector<std::uint8_t>& buf, std::string_view text) {
  AppendMetaPod(buf, text.size(), 8);
  buf.insert(buf.end(), text.begin(), text.end());
}

void AppendMetaStringValue(std::vector<std::uint8_t>& buf, std::string_view key,
                           std::string_view value) {
  AppendMetaString(buf, key);
  AppendMetaPod(buf, 8, 4);  // kString
  AppendMetaString(buf, value);
}

void AppendMetaUint32Value(std::vector<std::uint8_t>& buf, std::string_view key,
                           std::uint32_t value) {
  AppendMetaString(buf, key);
  AppendMetaPod(buf, 4, 4);  // kUint32
  AppendMetaPod(buf, value, 4);
}

void AppendMetaFloat32Value(std::vector<std::uint8_t>& buf,
                            std::string_view key, float value) {
  AppendMetaString(buf, key);
  AppendMetaPod(buf, 6, 4);  // kFloat32
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  AppendMetaPod(buf, bits, 4);
}

// A valid 2-layer llama config: head_dim 128, GQA group 2. Extra entries
// (sliding_window, rope scaling) are appended by the caller when needed.
std::vector<std::uint8_t> BuildSyntheticLlamaConfig(
    std::uint32_t head_dim, std::uint32_t heads, std::uint32_t kv_heads,
    const std::vector<std::uint8_t>& extra = {}) {
  std::vector<std::uint8_t> buf;
  buf.insert(buf.end(), {'G', 'G', 'U', 'F'});
  AppendMetaPod(buf, 3, 4);  // version
  AppendMetaPod(buf, 0, 8);  // tensor count
  const std::vector<std::pair<std::string_view, std::uint32_t>> uints = {
      {"llama.block_count", 2},
      {"llama.embedding_length", 128},
      {"llama.feed_forward_length", 256},
      {"llama.attention.head_count", heads},
      {"llama.attention.head_count_kv", kv_heads},
      {"llama.attention.key_length", head_dim},
      {"llama.attention.value_length", head_dim},
      {"llama.rope.dimension_count", head_dim},
      {"llama.context_length", 1024},
      {"llama.vocab_size", 512},
  };
  AppendMetaPod(buf, 3 + uints.size() + (extra.empty() ? 0 : 1), 8);
  AppendMetaStringValue(buf, "general.architecture", "llama");
  for (const auto& [key, value] : uints) {
    AppendMetaUint32Value(buf, key, value);
  }
  AppendMetaFloat32Value(buf, "llama.attention.layer_norm_rms_epsilon",
                         0.00001F);
  AppendMetaFloat32Value(buf, "llama.rope.freq_base", 500000.0F);
  buf.insert(buf.end(), extra.begin(), extra.end());
  // ParseHeaders aligns the tensor payload base to the 32-byte GGUF
  // alignment; a real file always carries tensor bytes after the headers, so
  // pad this header-only buffer the same way.
  while (buf.size() % 32 != 0) {
    buf.push_back(0);
  }
  return buf;
}

std::optional<superroc::core::ModelConfig> ParseSyntheticConfig(
    const std::vector<std::uint8_t>& file, std::string& error) {
  const auto reader =
      superroc::core::GgufReader::OpenMemory(file.data(), file.size(), &error);
  Expect(reader != nullptr, "open synthetic llama GGUF: " + error);
  return reader->ExtractModelConfig(&error);
}

// F9: sliding-window and scaled-RoPE llama files must be refused at load.
void TestLlamaUnsupportedGeometryRefused() {
  std::string error;
  std::vector<std::uint8_t> swa_extra;
  AppendMetaString(swa_extra, "llama.attention.sliding_window");
  AppendMetaPod(swa_extra, 4, 4);  // kUint32
  AppendMetaPod(swa_extra, 4096, 4);
  Expect(!ParseSyntheticConfig(BuildSyntheticLlamaConfig(128, 4, 2, swa_extra),
                               error)
              .has_value(),
         "llama sliding_window must be refused: " + error);
  std::vector<std::uint8_t> scaling_extra;
  AppendMetaString(scaling_extra, "llama.rope.scaling.type");
  AppendMetaPod(scaling_extra, 8, 4);  // kString
  AppendMetaString(scaling_extra, "yarn");
  Expect(!ParseSyntheticConfig(
              BuildSyntheticLlamaConfig(128, 4, 2, scaling_extra), error)
              .has_value(),
         "llama rope scaling must be refused: " + error);
  Expect(ParseSyntheticConfig(BuildSyntheticLlamaConfig(128, 4, 2), error)
             .has_value(),
         "valid llama config must load: " + error);
}

// F11: only head_dim 128 and GQA groups <= 8 reach Flint decode intact.
void TestLlamaDecodeGeometryValidated() {
  std::string error;
  Expect(!ParseSyntheticConfig(BuildSyntheticLlamaConfig(64, 4, 2), error)
              .has_value(),
         "llama head_dim != 128 must be refused: " + error);
  Expect(!ParseSyntheticConfig(BuildSyntheticLlamaConfig(128, 16, 1), error)
              .has_value(),
         "llama GQA group over 8 must be refused: " + error);
}

int main() {
  TestLlamaGetTokenizerThrows();
  TestLlamaCreateResolvesFp32KvPolicy();
  TestLlamaUnsupportedGeometryRefused();
  TestLlamaDecodeGeometryValidated();
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
  const auto reader = superroc::core::GgufReader::OpenFile(gguf, &error);
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
  Expect(cfg.rope_variant == superroc::core::RopeVariant::kNorm,
         "adjacent-pair (NORM) RoPE");
  Expect(cfg.attn_gate_mode == superroc::core::AttnGateMode::kFused,
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
