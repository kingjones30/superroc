// LLAMA: bind the Nanbeige4.3-33B-Base GGUF and print a per-type census.
// All 471 tensors must be accounted for. Exits 77 when NANBEIGE_GGUF is unset.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/model_config.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/llama/weights.hpp"
#include "src/models/qwen/state.hpp"

namespace {

namespace fs = std::filesystem;

void Fail(const std::string& msg) {
  std::cerr << "FAIL llama_bind_test: " << msg << "\n";
  std::exit(1);
}

void Expect(bool ok, const std::string& msg) {
  if (!ok) {
    Fail(msg);
  }
}

// Minimal in-memory GGUF writer: header plus F32 tensors, enough for the
// F6 bind-type test via OpenMemory.
void AppendBytes(std::vector<std::uint8_t>& buf, const void* data,
                 std::size_t size) {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  buf.insert(buf.end(), bytes, bytes + size);
}

template<typename T>
void AppendPod(std::vector<std::uint8_t>& buf, T value) {
  AppendBytes(buf, &value, sizeof(value));
}

void AppendString(std::vector<std::uint8_t>& buf, std::string_view text) {
  AppendPod(buf, static_cast<std::uint64_t>(text.size()));
  AppendBytes(buf, text.data(), text.size());
}

void AppendTensor(std::vector<std::uint8_t>& header,
                  std::vector<std::uint8_t>& payload, std::string_view name,
                  superroc::core::GgmlType type,
                  std::initializer_list<std::uint64_t> dims,
                  std::uint64_t& offset) {
  AppendString(header, name);
  AppendPod(header, static_cast<std::uint32_t>(dims.size()));
  std::size_t elements = 1;
  for (const auto dim : dims) {
    AppendPod(header, dim);
    elements *= dim;
  }
  AppendPod(header, static_cast<std::uint32_t>(type));
  AppendPod(header, offset);
  const std::size_t bytes =
      type == superroc::core::GgmlType::kF32
          ? elements * 4
          : superroc::quant::EncodedSizeBytes(type, elements);
  Expect(bytes > 0, "tensor payload size must be positive");
  payload.resize(payload.size() + bytes, 0);
  offset += bytes;
}

void AppendF32Tensor(std::vector<std::uint8_t>& header,
                     std::vector<std::uint8_t>& payload, std::string_view name,
                     std::initializer_list<std::uint64_t> dims,
                     std::uint64_t& offset) {
  AppendTensor(header, payload, name, superroc::core::GgmlType::kF32, dims,
               offset);
}

// F6: an F32 projection (output.weight) must be refused at bind time: the
// batched prefill FP16 GEMM throws on F32/BF16/Q8_K. Runs without
// NANBEIGE_GGUF.
void TestF32ProjectionRefused() {
  constexpr std::uint64_t kHidden = 32;
  constexpr std::uint64_t kVocab = 64;
  std::vector<std::uint8_t> header;
  std::vector<std::uint8_t> payload;
  std::uint64_t offset = 0;
  AppendF32Tensor(header, payload, "token_embd.weight", {kHidden, kVocab},
                  offset);
  AppendF32Tensor(header, payload, "output_norm.weight", {kHidden}, offset);
  AppendF32Tensor(header, payload, "output.weight", {kHidden, kVocab}, offset);
  std::vector<std::uint8_t> file;
  AppendBytes(file, "GGUF", 4);
  AppendPod(file, static_cast<std::uint32_t>(3));  // version
  AppendPod(file, static_cast<std::uint64_t>(3));  // tensor count
  AppendPod(file, static_cast<std::uint64_t>(1));  // metadata count
  AppendString(file, "general.architecture");
  AppendPod(file,
            static_cast<std::uint32_t>(superroc::core::GgufValueType::kString));
  AppendString(file, "llama");
  file.insert(file.end(), header.begin(), header.end());
  while (file.size() % 32 != 0) {
    file.push_back(0);
  }
  file.insert(file.end(), payload.begin(), payload.end());
  std::string error;
  const auto reader =
      superroc::core::GgufReader::OpenMemory(file.data(), file.size(), &error);
  Expect(reader != nullptr, "open synthetic GGUF: " + error);
  superroc::core::ModelConfig config;
  config.llama = true;
  config.num_layers = 1;
  config.hidden_size = static_cast<std::uint32_t>(kHidden);
  config.intermediate_size = 64;
  config.num_attention_heads = 2;
  config.num_key_value_heads = 2;
  config.head_dim = 16;
  config.rotary_dim = 16;
  config.context_length = 128;
  config.vocab_size = static_cast<std::uint32_t>(kVocab);
  const auto weights =
      superroc::models::BindLlamaWeights(*reader, config, &error);
  Expect(!weights.has_value(), "F32 output.weight must not bind");
  Expect(error.find("Unsupported llama tensor type") != std::string::npos,
         "F32 projection refused for its type, got: " + error);
}

// F6 allowlist + F15: an all-quant 1-layer file binds, and binding prints
// nothing on stdout (the prompt CLI owns stdout as the generation stream).
void TestQuantFileBindsQuietly() {
  constexpr std::uint64_t kHidden = 32;
  constexpr std::uint64_t kVocab = 64;
  constexpr std::uint64_t kInter = 64;
  constexpr std::uint64_t kAttn = 32;
  constexpr auto kQ8 = superroc::core::GgmlType::kQ8_0;
  std::vector<std::uint8_t> header;
  std::vector<std::uint8_t> payload;
  std::uint64_t offset = 0;
  std::uint64_t tensor_count = 0;
  const auto add = [&](std::string_view name, superroc::core::GgmlType type,
                       std::initializer_list<std::uint64_t> dims) {
    AppendTensor(header, payload, name, type, dims, offset);
    ++tensor_count;
  };
  add("token_embd.weight", superroc::core::GgmlType::kF32, {kHidden, kVocab});
  add("output_norm.weight", superroc::core::GgmlType::kF32, {kHidden});
  add("output.weight", kQ8, {kHidden, kVocab});
  const std::string prefix = "blk.0.";
  add(prefix + "attn_norm.weight", superroc::core::GgmlType::kF32, {kHidden});
  add(prefix + "ffn_norm.weight", superroc::core::GgmlType::kF32, {kHidden});
  add(prefix + "attn_q.weight", kQ8, {kAttn, kHidden});
  add(prefix + "attn_k.weight", kQ8, {kAttn, kHidden});
  add(prefix + "attn_v.weight", kQ8, {kAttn, kHidden});
  add(prefix + "attn_output.weight", kQ8, {kHidden, kAttn});
  add(prefix + "ffn_gate.weight", kQ8, {kInter, kHidden});
  add(prefix + "ffn_up.weight", kQ8, {kInter, kHidden});
  add(prefix + "ffn_down.weight", kQ8, {kHidden, kInter});
  std::vector<std::uint8_t> file;
  AppendBytes(file, "GGUF", 4);
  AppendPod(file, static_cast<std::uint32_t>(3));  // version
  AppendPod(file, tensor_count);
  AppendPod(file, static_cast<std::uint64_t>(1));  // metadata count
  AppendString(file, "general.architecture");
  AppendPod(file,
            static_cast<std::uint32_t>(superroc::core::GgufValueType::kString));
  AppendString(file, "llama");
  file.insert(file.end(), header.begin(), header.end());
  while (file.size() % 32 != 0) {
    file.push_back(0);
  }
  file.insert(file.end(), payload.begin(), payload.end());
  std::string error;
  const auto reader =
      superroc::core::GgufReader::OpenMemory(file.data(), file.size(), &error);
  Expect(reader != nullptr, "open synthetic quant GGUF: " + error);
  superroc::core::ModelConfig config;
  config.llama = true;
  config.num_layers = 1;
  config.hidden_size = static_cast<std::uint32_t>(kHidden);
  config.intermediate_size = static_cast<std::uint32_t>(kInter);
  config.num_attention_heads = 2;
  config.num_key_value_heads = 2;
  config.head_dim = 16;
  config.rotary_dim = 16;
  config.context_length = 128;
  config.vocab_size = static_cast<std::uint32_t>(kVocab);
  std::stringstream captured;
  auto* const old_cout = std::cout.rdbuf(captured.rdbuf());
  const auto weights =
      superroc::models::BindLlamaWeights(*reader, config, &error);
  std::cout.rdbuf(old_cout);
  Expect(weights.has_value(), "quant llama file must bind: " + error);
  Expect(captured.str().empty(), "bind must not print on stdout");
}

}  // namespace

int main() {
  TestF32ProjectionRefused();
  TestQuantFileBindsQuietly();
  const char* gguf_env = std::getenv("NANBEIGE_GGUF");
  if (gguf_env == nullptr || gguf_env[0] == '\0') {
    std::cout << "SKIP llama_bind_test: NANBEIGE_GGUF is not set\n";
    return 77;
  }
  const fs::path gguf{gguf_env};
  if (!fs::exists(gguf)) {
    std::cout << "SKIP llama_bind_test: " << gguf << " is absent\n";
    return 77;
  }

  std::string error;
  const auto reader = superroc::core::GgufReader::OpenFile(gguf, &error);
  Expect(reader != nullptr, "open GGUF: " + error);
  Expect(reader->GetTensorCount() == 471,
         "tensor count " + std::to_string(reader->GetTensorCount()));

  std::map<std::string, std::uint64_t> count;
  std::uint64_t accounted = 0;
  for (const auto& tensor : reader->GetTensors()) {
    count[std::string(superroc::core::ToString(tensor.type))] += 1;
    ++accounted;
  }
  Expect(accounted == 471, "accounted " + std::to_string(accounted));
  std::cout << "llama_bind_test tensors=" << accounted;
  for (const auto& [ty, n] : count) {
    std::cout << " " << ty << "=" << n;
  }
  std::cout << "\n";

  const auto weights =
      superroc::models::QwenModelWeights::LoadFromGguf(*reader, &error);
  Expect(weights.has_value(), "LoadFromGguf: " + error);
  Expect(weights->config.llama, "config.llama");
  Expect(weights->layers.size() == 52, "52 layers");
  Expect(!weights->output.empty(), "untied output");
  Expect(!weights->output_tied_to_token_embd, "output not tied");
  for (std::uint32_t i = 0; i < 52; ++i) {
    const auto& layer = weights->layers[i];
    Expect(layer.is_full_attention,
           "layer " + std::to_string(i) + " full attn");
    Expect(!layer.is_moe, "layer " + std::to_string(i) + " dense");
    Expect(layer.attn_sinks.empty(),
           "layer " + std::to_string(i) + " no sinks");
    Expect(layer.attn_q_norm.empty() && layer.attn_k_norm.empty(),
           "layer " + std::to_string(i) + " no q/k norm");
    Expect(!layer.attn_q.empty() && !layer.attn_k.empty() &&
               !layer.attn_v.empty() && !layer.attn_output.empty(),
           "layer " + std::to_string(i) + " qkv/o");
    Expect(!layer.ffn_gate.empty() && !layer.ffn_up.empty() &&
               !layer.ffn_down.empty(),
           "layer " + std::to_string(i) + " swiglu");
  }
  std::cout << "LoadFromGguf ok layers=" << weights->layers.size()
            << " hidden=" << weights->config.hidden_size
            << " vocab=" << weights->config.vocab_size << "\n";
  std::cout << "PASS llama_bind_test\n";
  return 0;
}
