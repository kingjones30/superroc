// LLAMA: bind the Nanbeige4.3-33B-Base GGUF and print a per-type census.
// All 471 tensors must be accounted for. Exits 77 when NANBEIGE_GGUF is unset.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>

#include "src/core/gguf_reader.hpp"
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

}  // namespace

int main() {
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
  const auto reader = gufo::core::GgufReader::OpenFile(gguf, &error);
  Expect(reader != nullptr, "open GGUF: " + error);
  Expect(reader->GetTensorCount() == 471,
         "tensor count " + std::to_string(reader->GetTensorCount()));

  std::map<std::string, std::uint64_t> count;
  std::uint64_t accounted = 0;
  for (const auto& tensor : reader->GetTensors()) {
    count[std::string(gufo::core::ToString(tensor.type))] += 1;
    ++accounted;
  }
  Expect(accounted == 471, "accounted " + std::to_string(accounted));
  std::cout << "llama_bind_test tensors=" << accounted;
  for (const auto& [ty, n] : count) {
    std::cout << " " << ty << "=" << n;
  }
  std::cout << "\n";

  const auto weights =
      gufo::models::QwenModelWeights::LoadFromGguf(*reader, &error);
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
