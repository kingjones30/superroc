#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/dflash_weights.hpp"
#include "src/core/env_compat.hpp"

namespace {

constexpr int kSkipped = 77;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void PutU32(std::vector<std::uint8_t>& o, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    o.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
  }
}

void PutU64(std::vector<std::uint8_t>& o, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    o.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
  }
}

void PutStr(std::vector<std::uint8_t>& o, std::string_view s) {
  PutU64(o, s.size());
  o.insert(o.end(), s.begin(), s.end());
}

void PutKey(std::vector<std::uint8_t>& o, std::string_view key) {
  PutStr(o, key);
}

void AddStr(std::vector<std::uint8_t>& o, std::string_view key,
            std::string_view value) {
  PutKey(o, key);
  PutU32(o, 8);
  PutStr(o, value);
}

void AddU32(std::vector<std::uint8_t>& o, std::string_view key,
            std::uint32_t value) {
  PutKey(o, key);
  PutU32(o, 4);
  PutU32(o, value);
}

void AddF32(std::vector<std::uint8_t>& o, std::string_view key, float value) {
  PutKey(o, key);
  PutU32(o, 6);
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, 4);
  PutU32(o, bits);
}

void AddBool(std::vector<std::uint8_t>& o, std::string_view key, bool value) {
  PutKey(o, key);
  PutU32(o, 7);
  o.push_back(value ? 1 : 0);
}

void AddU32Array(std::vector<std::uint8_t>& o, std::string_view key,
                 const std::vector<std::uint32_t>& values) {
  PutKey(o, key);
  PutU32(o, 9);
  PutU32(o, 4);
  PutU64(o, values.size());
  for (auto v : values) {
    PutU32(o, v);
  }
}

struct Tensor {
  std::string name;
  std::vector<std::uint64_t> dims;
  std::vector<std::uint8_t> raw;
};

Tensor F32Tensor(std::string name, std::vector<std::uint64_t> dims) {
  Tensor t;
  t.name = std::move(name);
  t.dims = std::move(dims);
  std::uint64_t n = 1;
  for (auto d : t.dims) {
    n *= d;
  }
  t.raw.assign(static_cast<std::size_t>(n) * 4, 0);
  return t;
}

std::vector<std::uint8_t> WriteGguf(const std::vector<std::uint8_t>& kv,
                                    std::uint64_t kv_count,
                                    const std::vector<Tensor>& ts) {
  constexpr std::uint32_t kAlign = 32;
  std::vector<std::uint8_t> o;
  o.push_back('G');
  o.push_back('G');
  o.push_back('U');
  o.push_back('F');
  PutU32(o, 3);
  PutU64(o, ts.size());
  PutU64(o, kv_count);
  o.insert(o.end(), kv.begin(), kv.end());
  std::uint64_t run = 0;
  std::vector<std::uint64_t> offs(ts.size());
  for (std::size_t i = 0; i < ts.size(); ++i) {
    offs[i] = run;
    const auto nb = static_cast<std::uint64_t>(ts[i].raw.size());
    run += (nb + kAlign - 1) / kAlign * kAlign;
  }
  for (std::size_t i = 0; i < ts.size(); ++i) {
    PutStr(o, ts[i].name);
    PutU32(o, static_cast<std::uint32_t>(ts[i].dims.size()));
    for (auto d : ts[i].dims) {
      PutU64(o, d);
    }
    PutU32(o, 0);
    PutU64(o, offs[i]);
  }
  while (o.size() % kAlign) {
    o.push_back(0);
  }
  for (const auto& t : ts) {
    o.insert(o.end(), t.raw.begin(), t.raw.end());
    while (o.size() % kAlign) {
      o.push_back(0);
    }
  }
  return o;
}

void TestDflash2HeaderStillLoads() {
  constexpr std::uint32_t kHidden = 32;
  constexpr std::uint32_t kFf = 64;
  constexpr std::uint32_t kHeads = 2;
  constexpr std::uint32_t kKv = 1;
  constexpr std::uint32_t kHeadDim = 16;
  constexpr std::uint32_t kVocab = 16;
  constexpr std::uint32_t kRank = 8;
  constexpr std::uint32_t kTopK = 4;
  constexpr std::uint32_t kConvK = 2;
  constexpr std::uint32_t kConvG = 16;
  constexpr std::uint32_t kBlock = 4;
  constexpr std::uint32_t kWindow = 32;
  const std::uint32_t groups = kHidden / kConvG;
  const std::uint32_t dynamic = 2U * kConvK * groups;
  const std::uint32_t attn = kHeads * kHeadDim;
  const std::uint32_t kv = kKv * kHeadDim;

  std::vector<std::uint8_t> kv_bytes;
  std::uint64_t kv_count = 0;
  const auto add = [&](auto fn) {
    fn();
    ++kv_count;
  };
  add([&] { AddStr(kv_bytes, "general.architecture", "dflash"); });
  add([&] { AddU32(kv_bytes, "dflash.embedding_length", kHidden); });
  add([&] { AddU32(kv_bytes, "dflash.feed_forward_length", kFf); });
  add([&] { AddU32(kv_bytes, "dflash.attention.head_count", kHeads); });
  add([&] { AddU32(kv_bytes, "dflash.attention.head_count_kv", kKv); });
  add([&] { AddU32(kv_bytes, "dflash.attention.key_length", kHeadDim); });
  add([&] { AddU32(kv_bytes, "dflash.rope.dimension_count", kHeadDim); });
  add([&] { AddU32(kv_bytes, "dflash.context_length", 128); });
  add([&] { AddU32(kv_bytes, "dflash.block_count", 1); });
  add([&] { AddF32(kv_bytes, "dflash.rope.freq_base", 10000.0F); });
  add([&] { AddU32(kv_bytes, "dflash.block_size", kBlock); });
  add([&] { AddU32(kv_bytes, "dflash.conv_kernel_size", kConvK); });
  add([&] { AddU32(kv_bytes, "dflash.conv_group_size", kConvG); });
  add([&] { AddU32(kv_bytes, "dflash.selector_rank", kRank); });
  add([&] { AddU32(kv_bytes, "dflash.selector_top_k", kTopK); });
  add([&] { AddU32(kv_bytes, "dflash.attention.sliding_window", kWindow); });
  add([&] { AddU32(kv_bytes, "tokenizer.ggml.mask_token_id", 1); });
  add([&] { AddBool(kv_bytes, "dflash.attention.causal", false); });
  add([&] { AddU32Array(kv_bytes, "dflash.target_layers", {2}); });

  std::vector<Tensor> ts;
  ts.push_back(F32Tensor("selector_predecessor.weight", {kRank, kVocab}));
  ts.push_back(F32Tensor("selector_successor.weight", {kRank, kVocab}));
  ts.push_back(F32Tensor("selector_hidden.weight", {kHidden, kRank}));
  ts.push_back(F32Tensor("fc.weight", {kHidden, kHidden}));
  ts.push_back(F32Tensor("enc.output_norm.weight", {kHidden}));
  ts.push_back(F32Tensor("output_norm.weight", {kHidden}));
  ts.push_back(F32Tensor("blk.0.attn_norm.weight", {kHidden}));
  ts.push_back(F32Tensor("blk.0.attn_q.weight", {kHidden, attn}));
  ts.push_back(F32Tensor("blk.0.attn_k.weight", {kHidden, kv}));
  ts.push_back(F32Tensor("blk.0.attn_v.weight", {kHidden, kv}));
  ts.push_back(F32Tensor("blk.0.attn_output.weight", {attn, kHidden}));
  ts.push_back(F32Tensor("blk.0.attn_q_norm.weight", {kHeadDim}));
  ts.push_back(F32Tensor("blk.0.attn_k_norm.weight", {kHeadDim}));
  ts.push_back(F32Tensor("blk.0.ffn_norm.weight", {kHidden}));
  ts.push_back(F32Tensor("blk.0.ffn_gate.weight", {kHidden, kFf}));
  ts.push_back(F32Tensor("blk.0.ffn_up.weight", {kHidden, kFf}));
  ts.push_back(F32Tensor("blk.0.ffn_down.weight", {kFf, kHidden}));
  ts.push_back(F32Tensor("blk.0.attn_conv_base", {kHidden, kConvK, 2}));
  ts.push_back(F32Tensor("blk.0.attn_conv_proj.weight", {kHidden, dynamic}));
  ts.push_back(F32Tensor("blk.0.ffn_conv_base", {kHidden, kConvK, 2}));
  ts.push_back(F32Tensor("blk.0.ffn_conv_proj.weight", {kHidden, dynamic}));

  const auto bytes = WriteGguf(kv_bytes, kv_count, ts);
  const auto path =
      (std::filesystem::temp_directory_path() / "pdk-dflash2-synth.gguf")
          .string();
  {
    std::ofstream out(path, std::ios::binary);
    Expect(out.good(), "open synth dflash2 gguf");
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    Expect(out.good(), "write synth dflash2 gguf");
  }

  std::string error;
  auto reader = superroc::core::GgufReader::OpenFile(path, &error);
  Expect(reader != nullptr, "open synth reader: " + error);
  auto weights =
      superroc::speculative::QwenDFlashWeights::LoadFromGguf(*reader, &error);
  Expect(weights.has_value(), "dflash2-style header must still load: " + error);
  Expect(weights->dflash_config.has_selector, "dflash2 has_selector");
  Expect(weights->dflash_config.has_dynamic_conv, "dflash2 has_dynamic_conv");
  Expect(!weights->dflash_config.has_confidence_head,
         "dflash2 has no confidence head");
  Expect(weights->config.hidden_size == kHidden, "dflash2 hidden");
  Expect(weights->config.vocab_size == kVocab, "dflash2 vocab from selector");
  Expect(weights->dflash_config.selector_rank == kRank,
         "dflash2 selector_rank");
  Expect(weights->dflash_config.block_size == kBlock, "dflash2 block_size");
  Expect(weights->dflash_config.target_layer_ids.size() == 1 &&
             weights->dflash_config.target_layer_ids[0] == 1,
         "dflash2 still normalizes GGUF target_layers N to N-1");
  std::filesystem::remove(path);
  std::cout << "PASS dflash2-style synthetic header still loads the old way\n";
}

void TestRealDsparkFile(const char* path) {
  std::string error;
  auto reader = superroc::core::GgufReader::OpenFile(path, &error);
  Expect(reader != nullptr, "open dspark: " + error);
  auto weights =
      superroc::speculative::QwenDFlashWeights::LoadFromGguf(*reader, &error);
  Expect(weights.has_value(), "DSpark LoadFromGguf: " + error);
  const auto& cfg = weights->config;
  const auto& df = weights->dflash_config;
  Expect(cfg.hidden_size == 2048, "hidden 2048");
  Expect(df.num_layers == 5, "5 blocks");
  Expect(df.block_size == 15, "block_size 15");
  Expect(df.target_layer_ids.size() == 5, "5 taps");
  const char* tap_minus1 = ::superroc::core::GetEnvCompat("SUPERROC_DSPARK_TAP_MINUS1");
  const bool want_minus1 = tap_minus1 != nullptr && tap_minus1[0] != '\0' &&
                           !(tap_minus1[0] == '0' && tap_minus1[1] == '\0');
  if (want_minus1) {
    Expect(df.target_layer_ids[0] == 3 && df.target_layer_ids[1] == 11 &&
               df.target_layer_ids[2] == 19 && df.target_layer_ids[3] == 27 &&
               df.target_layer_ids[4] == 35,
           "taps 3 11 19 27 35 with SUPERROC_DSPARK_TAP_MINUS1");
  } else {
    Expect(df.target_layer_ids[0] == 4 && df.target_layer_ids[1] == 12 &&
               df.target_layer_ids[2] == 20 && df.target_layer_ids[3] == 28 &&
               df.target_layer_ids[4] == 36,
           "taps 4 12 20 28 36");
  }
  Expect(df.markov_rank == 256, "markov rank 256");
  Expect(df.has_confidence_head, "confidence head present");
  Expect(!df.has_selector, "no dflash2 selector");
  Expect(!df.has_dynamic_conv, "no dynamic conv");
  Expect(df.sample_from_anchor, "sample_from_anchor");
  Expect(cfg.vocab_size == 248144, "vocab from markov_w1");
  Expect(!weights->conf_proj.empty() && !weights->conf_proj_bias.empty(),
         "conf_proj bound");
  Expect(!weights->markov_w1.empty() && !weights->markov_w2.empty(),
         "markov tensors bound");
  std::cout << "PASS real DSpark loader"
            << " hidden=" << cfg.hidden_size << " layers=" << df.num_layers
            << " block_size=" << df.block_size
            << " markov_rank=" << df.markov_rank << " vocab=" << cfg.vocab_size
            << " taps=";
  for (std::size_t i = 0; i < df.target_layer_ids.size(); ++i) {
    if (i != 0) {
      std::cout << ',';
    }
    std::cout << df.target_layer_ids[i];
  }
  std::cout << "\n";
}

}  // namespace

int main() {
  try {
    TestDflash2HeaderStillLoads();
    const char* path = ::superroc::core::GetEnvCompat("SUPERROC_DSPARK_MODEL");
    if (path == nullptr || path[0] == '\0') {
      std::cout << "SKIP real DSpark file (SUPERROC_DSPARK_MODEL unset)\n";
      return 0;
    }
    if (access(path, R_OK) != 0) {
      std::cout << "SKIP real DSpark file missing: " << path << "\n";
      return kSkipped;
    }
    TestRealDsparkFile(path);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "FAIL " << ex.what() << "\n";
    return 1;
  }
}
