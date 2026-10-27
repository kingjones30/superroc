// GLIMMER G1. CPU only. Args: the two muse-glimmer GGUF paths, read in place.
// Exit 0 pass, 1 gate fail, 2 usage, 77 skipped (a file is not named/absent).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/ggml_gemm.hpp"
#include "src/models/glimmer/glimmer_chat.hpp"
#include "src/models/glimmer/glimmer_oracle.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

int Fail(const std::string& msg) {
  std::cerr << "FAIL " << msg << '\n';
  return 1;
}

std::span<const float> CSpan(const std::vector<float>& values) {
  return {values.data(), values.size()};
}

std::span<float> MSpan(std::vector<float>& values) {
  return {values.data(), values.size()};
}

constexpr std::int8_t kTable[16] = {0, 1,  2,  3,  4,  6,  8,  10,
                                    0, -1, -2, -3, -4, -6, -8, -10};

float TableScale(std::uint8_t e) {
  if (e > 0x7e) {
    return 0.0F;
  }
  if (e < 8) {
    return static_cast<float>(e) * 0x1p-10F;
  }
  const int exponent = static_cast<int>(e / 8);
  const float mantissa = static_cast<float>(8 + (e & 7));
  float scale = 0x1p-11F;
  for (int i = 0; i < exponent; ++i) {
    scale *= 2.0F;
  }
  return mantissa * scale;
}

float IndependentDot(superroc::core::GgmlType type, const std::uint8_t* raw,
                     const float* x, std::size_t k) {
  const std::size_t stride =
      type == superroc::core::GgmlType::kQ4_0_ROCMFP4 ? 18U : 17U;
  double acc = 0.0;
  for (std::size_t i = 0; i < k; ++i) {
    const std::uint8_t* blk = raw + (i / 32U) * stride;
    const int local = static_cast<int>(i % 32U);
    const std::uint8_t byte = blk[local & 15];
    const unsigned nib = local < 16 ? (byte & 0x0FU) : (byte >> 4);
    const std::uint8_t eb =
        (type == superroc::core::GgmlType::kQ4_0_ROCMFP4 && local >= 16) ? blk[17]
                                                                     : blk[16];
    acc += static_cast<double>(kTable[nib]) * TableScale(eb) * x[i];
  }
  return static_cast<float>(acc);
}

bool ConfigOk(const superroc::core::ModelConfig& config, std::string* why) {
  auto bad = [&](const char* msg) {
    *why = msg;
    return false;
  };
  if (config.architecture != "muse-glimmer" || !config.attention_only) {
    return bad("architecture");
  }
  if (config.num_layers != 52 || config.hidden_size != 6656 ||
      config.intermediate_size != 19968 || config.num_attention_heads != 32 ||
      config.num_key_value_heads != 2 || config.head_dim != 128 ||
      config.vocab_size != 202048 || config.context_length != 131072) {
    return bad("geometry");
  }
  if (config.full_attention_interval != 1 || config.swa_window != 2048 ||
      config.rotary_dim != 128) {
    return bad("window");
  }
  const float expect_scale = 0.1961161345243454F;
  const float rel =
      std::fabs(config.logit_scale - expect_scale) / std::fabs(expect_scale);
  if (!(rel < 1.0e-6F) || config.final_softcap != 20.0F) {
    return bad("logit scale or softcap");
  }
  if (!(std::fabs(config.norm_eps_pre - 1.0e-5F) < 1.0e-12F) ||
      config.norm_eps_post != 1.0e-8F) {
    return bad("eps");
  }
  int swa = 0;
  for (std::uint32_t i = 0; i < 52; ++i) {
    const bool expect = (i % 4U) != 3U;
    if (config.LayerUsesRope(i) != expect) {
      return bad("pattern");
    }
    swa += expect ? 1 : 0;
  }
  if (swa != 39) {
    return bad("swa count");
  }
  return true;
}

bool MathOk(const superroc::models::QwenModelWeights& weights, std::string* why) {
  const float x[4] = {1.0F, -2.0F, 0.5F, 0.25F};
  const float w[4] = {1.0F, 1.0F, 1.0F, 1.0F};
  float got[4];
  superroc::models::GlimmerRmsNorm(x, w, 1.0e-5F, got);
  double sum = 0.0;
  for (float v : x) {
    sum += static_cast<double>(v) * v;
  }
  const double inv = 1.0 / std::sqrt(sum / 4.0 + 1.0e-5);
  float ref[4];
  for (int i = 0; i < 4; ++i) {
    ref[i] = static_cast<float>(x[i] * inv);
  }
  if (superroc::models::GlimmerCosine(got, ref) < 0.999999) {
    *why = "rmsnorm";
    return false;
  }
  float bare[4];
  superroc::models::GlimmerRmsNorm(x, {}, 1.0e-5F, bare);
  if (superroc::models::GlimmerCosine(bare, ref) < 0.999999) {
    *why = "unweighted rmsnorm";
    return false;
  }

  const auto qn = weights.layers[0].attn_q_norm.AsFloatSpan();
  if (qn.size() != 128) {
    *why = "q_norm span";
    return false;
  }
  std::vector<float> head(128), normed(128), local(128);
  for (int i = 0; i < 128; ++i) {
    head[i] = static_cast<float>((i % 9) - 4) * 0.1F;
  }
  superroc::models::GlimmerRmsNorm(CSpan(head), qn, 1.0e-5F, MSpan(normed));
  double hsum = 0.0;
  for (float v : head) {
    hsum += static_cast<double>(v) * v;
  }
  const double hinv = 1.0 / std::sqrt(hsum / 128.0 + 1.0e-5);
  for (int i = 0; i < 128; ++i) {
    local[i] =
        static_cast<float>(head[i] * hinv * qn[static_cast<std::size_t>(i)]);
  }
  if (superroc::models::GlimmerCosine(CSpan(normed), CSpan(local)) < 0.999999) {
    *why = "real q_norm";
    return false;
  }

  std::vector<float> rope = head;
  superroc::models::GlimmerNormRope(MSpan(rope), 128, 7, 500000.0F);
  std::vector<float> rope_ref = head;
  for (std::size_t i = 0; i < 64; ++i) {
    const double freq =
        std::pow(500000.0, -2.0 * static_cast<double>(i) / 128.0);
    const double ang = 7.0 * freq;
    const double c = std::cos(ang);
    const double s = std::sin(ang);
    const double x0 = head[2 * i];
    const double x1 = head[2 * i + 1];
    rope_ref[2 * i] = static_cast<float>(x0 * c - x1 * s);
    rope_ref[2 * i + 1] = static_cast<float>(x0 * s + x1 * c);
  }
  if (superroc::models::GlimmerCosine(CSpan(rope), CSpan(rope_ref)) < 0.999999) {
    *why = "norm rope";
    return false;
  }
  if (superroc::models::GlimmerSwaBegin(10, 2048) != 0 ||
      superroc::models::GlimmerSwaBegin(2048, 2048) != 1 ||
      superroc::models::GlimmerSwaBegin(3, 0) != 0) {
    *why = "swa begin";
    return false;
  }

  float logits[3] = {40.0F, -40.0F, 1.0F};
  const float scale = weights.config.logit_scale;
  float expect[3];
  for (int i = 0; i < 3; ++i) {
    const float y = logits[i] * scale;
    expect[i] = 20.0F * std::tanh(y / 20.0F);
  }
  superroc::models::GlimmerScaleSoftcap(logits, scale, 20.0F);
  if (superroc::models::GlimmerCosine(logits, expect) < 0.999999) {
    *why = "softcap";
    return false;
  }
  return true;
}

bool SampleTensor(const superroc::core::GgufTensorInfo& tensor, double* min_float,
                  double* min_q8, std::string* why) {
  if (tensor.dimensions.size() < 2 || tensor.data == nullptr) {
    return true;
  }
  const std::size_t k = static_cast<std::size_t>(tensor.dimensions[0]);
  std::size_t rows = static_cast<std::size_t>(tensor.dimensions[1]);
  for (std::size_t d = 2; d < tensor.dimensions.size(); ++d) {
    rows *= static_cast<std::size_t>(tensor.dimensions[d]);
  }
  if (k == 0 || (k % 32U) != 0U || rows == 0) {
    *why = std::string(tensor.name) + " bad shape";
    return false;
  }
  const std::size_t row_bytes = superroc::quant::QuantizedRowBytes(tensor.type, k);
  if (row_bytes == 0) {
    *why = std::string(tensor.name) + " row bytes";
    return false;
  }
  constexpr int kRows = 32;
  std::vector<float> rnd(k), ramp(k), dot_a(kRows), ind_a(kRows), q8_a(kRows);
  std::vector<float> dot_b(kRows), ind_b(kRows), q8_b(kRows);
  std::mt19937 rng(0x6C676C31U);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  for (std::size_t i = 0; i < k; ++i) {
    rnd[i] = dist(rng);
    ramp[i] = (static_cast<float>(i % 32U) - 15.5F) / 16.0F;
  }
  const auto* base = static_cast<const std::uint8_t*>(tensor.data);
  for (int s = 0; s < kRows; ++s) {
    const std::size_t row = (static_cast<std::size_t>(s) * rows) / kRows;
    const auto* raw = base + row * row_bytes;
    dot_a[s] = superroc::quant::Dot(tensor.type, raw, CSpan(rnd), k);
    ind_a[s] = IndependentDot(tensor.type, raw, rnd.data(), k);
    q8_a[s] = superroc::models::GlimmerQ8Dot(tensor.type, raw, CSpan(rnd), k);
    dot_b[s] = superroc::quant::Dot(tensor.type, raw, CSpan(ramp), k);
    ind_b[s] = IndependentDot(tensor.type, raw, ramp.data(), k);
    q8_b[s] = superroc::models::GlimmerQ8Dot(tensor.type, raw, CSpan(ramp), k);
  }
  const double c0 = superroc::models::GlimmerCosine(CSpan(dot_a), CSpan(ind_a));
  const double c1 = superroc::models::GlimmerCosine(CSpan(dot_b), CSpan(ind_b));
  const double q0 = superroc::models::GlimmerCosine(CSpan(q8_a), CSpan(ind_a));
  const double q1 = superroc::models::GlimmerCosine(CSpan(q8_b), CSpan(ind_b));
  *min_float = std::min(*min_float, std::min(c0, c1));
  *min_q8 = std::min(*min_q8, std::min(q0, q1));
  if (!(c0 >= 0.999999) || !(c1 >= 0.999999)) {
    *why = std::string(tensor.name) + " float cosine";
    return false;
  }
  if (!(q0 >= 0.9999) || !(q1 >= 0.9999)) {
    *why = std::string(tensor.name) + " q8 cosine " +
           std::to_string(std::min(q0, q1));
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 1 && argc != 3) {
    std::cerr << "usage: glimmer_g1_test <103.gguf> <fast.gguf>\n";
    return 2;
  }
  // The two muse-glimmer files are named on the command line or, under ctest,
  // by GLIMMER_G1_GGUF and GLIMMER_G1_FAST_GGUF. If either is not named, or is
  // named but absent, the job skips (ctest reports Skipped, exit 77) instead
  // of failing, so a machine without the weights still builds green.
  std::array<std::string, 2> paths;
  if (argc == 3) {
    paths = {argv[1], argv[2]};
  } else {
    const char* primary = std::getenv("GLIMMER_G1_GGUF");
    const char* fast = std::getenv("GLIMMER_G1_FAST_GGUF");
    if (primary == nullptr || fast == nullptr) {
      std::cerr << "SKIP glimmer_g1_test: GLIMMER_G1_GGUF and "
                   "GLIMMER_G1_FAST_GGUF are not both set\n";
      return 77;
    }
    paths = {primary, fast};
  }
  double min_float = 1.0;
  double min_q8 = 1.0;
  int fp4_tensors = 0;
  for (int arg = 0; arg < 2; ++arg) {
    const std::string path = paths[arg];
    std::string err;
    auto reader = superroc::core::GgufReader::OpenFile(path, &err);
    if (reader == nullptr) {
      std::cerr << "SKIP glimmer_g1_test: " << path << " unreadable: " << err
                << '\n';
      return 77;
    }
    const auto config = reader->ExtractModelConfig(&err);
    if (!config.has_value()) {
      return Fail(path + " config " + err);
    }
    std::string why;
    if (!ConfigOk(*config, &why)) {
      return Fail(path + " " + why);
    }
    auto weights = superroc::models::QwenModelWeights::LoadFromGguf(*reader, &err);
    if (!weights.has_value()) {
      return Fail(path + " bind " + err);
    }
    const auto& layer0 = weights->layers[0];
    if (layer0.attn_post_norm.data == nullptr ||
        layer0.attn_post_norm.data == layer0.ffn_norm.data ||
        layer0.ffn_post_norm.data == layer0.ffn_norm.data ||
        layer0.attn_q.num_elements != 4096ULL * 6656ULL) {
      return Fail(path + " post_attention_norm alias or attn_q shape");
    }
    auto tokenizer =
        superroc::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
    if (tokenizer == nullptr || !tokenizer->UsesLlama4PreTokenizer() ||
        !tokenizer->AddBosToken() || tokenizer->GetBosTokenId() != 200000 ||
        tokenizer->GetEosTokenId() != 200001 ||
        tokenizer->GetPadTokenId() != 200018) {
      return Fail(path + " tokenizer " + err);
    }
    const auto hello = tokenizer->PreTokenizePieces("Hello world");
    const auto digits = tokenizer->PreTokenizePieces("12345");
    const auto dont = tokenizer->PreTokenizePieces("don\'t");
    const std::string long_upper(300, 'A');
    const auto mixed = tokenizer->PreTokenizePieces(long_upper + "abc");
    if (hello.size() != 2 || hello[0] != "Hello" || hello[1] != " world" ||
        digits.size() != 2 || digits[0] != "123" || digits[1] != "45" ||
        dont.size() != 1 || dont[0] != "don\'t" || mixed.size() != 1 ||
        mixed[0] != long_upper + "abc") {
      std::cerr << "pieces:";
      for (const auto& piece : hello) {
        std::cerr << " [" << piece << "]";
      }
      std::cerr << " digits";
      for (const auto& piece : digits) {
        std::cerr << " [" << piece << "]";
      }
      std::cerr << " dont=" << dont.size() << " mixed=" << mixed.size();
      if (!mixed.empty()) {
        std::cerr << " mixed0=" << mixed[0].size();
      }
      std::cerr << '\n';
      return Fail(path + " llama4 pieces");
    }
    auto chat =
        superroc::tokenization::QwenChatTemplate::CreateFromGguf(*reader, &err);
    if (chat == nullptr ||
        chat->GetTemplateSha256() !=
            superroc::tokenization::QwenChatTemplate::MuseGlimmerTemplateSha256() ||
        chat->GetProfile() !=
            superroc::tokenization::QwenChatTemplate::Profile::kMuseGlimmer ||
        !superroc::tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                                    &err)) {
      return Fail(path + " chat sha " + err);
    }
    superroc::tokenization::ChatMessage user{superroc::tokenization::ChatRole::kUser,
                                         "Hello", "", ""};
    std::string chat_err;
    const auto rendered = superroc::tokenization::GlimmerRender(
        *tokenizer, std::span<const superroc::tokenization::ChatMessage>(&user, 1),
        {}, {}, &chat_err);
    if (!rendered.has_value() ||
        rendered->find("<|im_start|>") != std::string::npos ||
        rendered->find("<|start|>user<|message|>Hello<|eot|>") ==
            std::string::npos ||
        rendered->find("Reasoning strength: high.") == std::string::npos) {
      return Fail(path + " render " + chat_err);
    }
    // Tool DEFINITIONS are rendered: the pinned template's render_tool_defs
    // macro (namespaces, JSON schemas, the ATEM function-call instructions, the
    // recipient namespace) is emitted into the system block, so the model can
    // call a tool. This is the contract serve relies on
    // (inference_backend.cpp "including tool definitions"). Only tool *history*
    // -- tool-role messages and assistant tool_calls -- fails closed.
    superroc::tokenization::ChatTool tool;
    tool.name = "now";
    std::string tool_err;
    const auto with_tool = superroc::tokenization::GlimmerRender(
        *tokenizer, std::span<const superroc::tokenization::ChatMessage>(&user, 1),
        std::span<const superroc::tokenization::ChatTool>(&tool, 1), {}, &tool_err);
    if (!with_tool.has_value() ||
        with_tool->find("\"name\": \"now\"") == std::string::npos ||
        with_tool->find("atem:function_calls") == std::string::npos ||
        with_tool->find("# Valid recipients: \"self\", \"now.*\", \"user\".") ==
            std::string::npos) {
      return Fail(path + " tool definition not rendered " + tool_err);
    }
    superroc::tokenization::ChatMessage tool_result{
        superroc::tokenization::ChatRole::kTool, "result", "now", ""};
    std::string result_err;
    const auto history_blocked = superroc::tokenization::GlimmerRender(
        *tokenizer,
        std::span<const superroc::tokenization::ChatMessage>(&tool_result, 1), {},
        {}, &result_err);
    if (history_blocked.has_value() ||
        result_err.find("are not rendered") == std::string::npos) {
      return Fail(path + " tool-result history was rendered");
    }
    superroc::tokenization::ChatMessage assistant_call{
        superroc::tokenization::ChatRole::kAssistant, "on it", "", ""};
    assistant_call.tool_calls.push_back(
        superroc::tokenization::ChatMessage::ToolCall{"call-1", "now", {}});
    std::string call_err;
    const auto call_blocked = superroc::tokenization::GlimmerRender(
        *tokenizer,
        std::span<const superroc::tokenization::ChatMessage>(&assistant_call, 1),
        {}, {}, &call_err);
    if (call_blocked.has_value() ||
        call_err.find("are not rendered") == std::string::npos) {
      return Fail(path + " assistant tool_calls were rendered");
    }

    using superroc::hip::detail::AttentionSupportParams;
    using superroc::hip::detail::IsSplitKDecodeAttentionSupported;
    using superroc::hip::detail::IsTiledAttentionSupported;
    AttentionSupportParams params;
    params.batch_size = 4096;
    params.num_heads = 32;
    params.num_kv_heads = 2;
    params.head_dim = 128;
    if (IsTiledAttentionSupported(params) ||
        IsSplitKDecodeAttentionSupported(4096, 32, 2, 128)) {
      return Fail(path + " 256-head policy selected head_dim 128");
    }

    if (!MathOk(*weights, &why)) {
      return Fail(path + " math " + why);
    }

    for (const auto& tensor : reader->GetTensors()) {
      if (tensor.type != superroc::core::GgmlType::kQ4_0_ROCMFP4 &&
          tensor.type != superroc::core::GgmlType::kQ4_0_ROCMFP4_FAST) {
        continue;
      }
      if (!SampleTensor(tensor, &min_float, &min_q8, &why)) {
        return Fail(path + " " + why);
      }
      ++fp4_tensors;
    }

    superroc::models::QwenScratchArena arena(*config);
    superroc::models::QwenKvCache cache(
        config->num_layers, config->num_key_value_heads, 8, config->head_dim);
    superroc::models::ForwardEmbedding(1, weights->token_embd, config->hidden_size,
                                   arena.hidden);
    superroc::models::GlimmerRmsNorm(arena.hidden.first(config->hidden_size), {},
                                 config->norm_eps_pre,
                                 arena.normed.first(config->hidden_size));
    std::copy_n(arena.normed.begin(), config->hidden_size,
                arena.hidden.begin());
    for (const std::uint32_t layer_index : {0U, 3U}) {
      superroc::models::GlimmerForwardLayer(arena.hidden,
                                        weights->layers[layer_index], *config,
                                        cache, layer_index, 0, arena);
      double energy = 0.0;
      for (std::size_t i = 0; i < config->hidden_size; ++i) {
        if (!std::isfinite(arena.hidden[i])) {
          return Fail(path + " non-finite layer " +
                      std::to_string(layer_index));
        }
        energy += std::fabs(arena.hidden[i]);
      }
      if (!(energy > 0.0)) {
        return Fail(path + " zero layer " + std::to_string(layer_index));
      }
    }
    std::cout << path << " bind_ok swa=39 fp4_running=" << fp4_tensors << '\n';
  }
  std::cout << "G1 float_cosine_min=" << min_float
            << " q8_cosine_min=" << min_q8 << " fp4_tensors=" << fp4_tensors
            << " rows=32 acts=2 (random + ramp, no calibration corpus)\n";
  if (!(min_float >= 0.999999) || !(min_q8 >= 0.9999)) {
    return Fail("cosine gate");
  }
  std::cout << "G1 PASS\n";
  return 0;
}
