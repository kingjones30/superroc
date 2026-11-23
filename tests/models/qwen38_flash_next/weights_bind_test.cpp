// FLASHNEXT: the trunk binder's LM-head format rule. A ROCmFP4 head (ggml
// types 100/101) must bind; the embedding table keeps the dense-projection
// list. Each case builds a metadata-plus-tensor-header GGUF image in memory,
// binds it, and inspects the binder's error, which is the first tensor it
// rejected or the first one it could not find.
//
// A full trunk cannot be synthesised here (48 layers of routed experts), so
// the pass signal for an accepted head is that the binder moves past both
// head tensors and stops at the output mixer: "missing tensor
// output_hc_norm.weight". The binder binds token_embd and output before that
// mixer, so reaching it proves the head's shape and format were accepted.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace qfn = superroc::models::qwen38_flash_next;
namespace core = superroc::core;

namespace {

using Metadata = std::map<
    std::string,
    std::variant<std::uint32_t, float, std::string, std::vector<std::uint64_t>,
                 std::vector<std::int64_t>, std::vector<double>>>;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

struct TensorSpec {
  std::string name;
  core::GgmlType type;
  std::uint64_t cols;
  std::uint64_t rows;
};

std::size_t EncodedRowBytes(const TensorSpec& spec) {
  qfn::TensorRef ref;
  ref.type = spec.type;
  ref.cols = spec.cols;
  ref.rows = spec.rows;
  return ref.RowBytes();
}

// GGUF v3 header with `fields` as metadata and one tensor-info entry per
// spec. Payload bytes are zero: nothing in this test dequantises them.
std::vector<std::uint8_t> BuildGguf(const Metadata& fields,
                                    const std::vector<TensorSpec>& tensors) {
  std::vector<std::uint8_t> bytes;
  const auto pod = [&]<class T>(T value) {
    const auto begin = bytes.size();
    bytes.resize(begin + sizeof(value));
    std::memcpy(bytes.data() + begin, &value, sizeof(value));
  };
  const auto text = [&](const std::string& value) {
    pod(static_cast<std::uint64_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(static_cast<std::uint64_t>(tensors.size()));
  pod(static_cast<std::uint64_t>(fields.size()));
  for (const auto& [key, value] : fields) {
    text(key);
    std::visit(
        [&](const auto& item) {
          using T = std::decay_t<decltype(item)>;
          if constexpr (std::is_same_v<T, std::uint32_t>) {
            pod(std::uint32_t{4});
            pod(item);
          } else if constexpr (std::is_same_v<T, float>) {
            pod(std::uint32_t{6});
            pod(item);
          } else if constexpr (std::is_same_v<T, std::string>) {
            pod(std::uint32_t{8});
            text(item);
          } else {
            pod(std::uint32_t{9});
            using E = typename T::value_type;
            pod(std::uint32_t{std::is_same_v<E, std::uint64_t>  ? 10U
                              : std::is_same_v<E, std::int64_t> ? 11U
                                                                : 12U});
            pod(static_cast<std::uint64_t>(item.size()));
            for (const auto entry : item) {
              pod(entry);
            }
          }
        },
        value);
  }
  std::uint64_t offset = 0;
  for (const auto& spec : tensors) {
    text(spec.name);
    pod(std::uint32_t{2});
    pod(spec.cols);
    pod(spec.rows);
    pod(static_cast<std::uint32_t>(spec.type));
    pod(offset);
    const std::size_t size = EncodedRowBytes(spec) * spec.rows;
    offset = (offset + size + 31) / 32 * 32;
  }
  // The tensor payload starts at the next 32-byte boundary of the image.
  bytes.resize((bytes.size() + 31) / 32 * 32);
  bytes.resize(bytes.size() + offset);
  return bytes;
}

// Binds the synthesised trunk and returns the error the binder reported. The
// full trunk is absent by design, so a bind that succeeds is a bug here.
std::string BindError(const Metadata& fields,
                      const std::vector<TensorSpec>& tensors) {
  auto bytes = BuildGguf(fields, tensors);
  std::string open_error;
  const auto reader =
      core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &open_error);
  Require(reader != nullptr, "GGUF image rejected: " + open_error);
  std::string bind_error;
  const auto weights = qfn::ModelWeights::Bind(*reader, &bind_error);
  Require(!weights.has_value(), "a trunk without layers bound successfully");
  return bind_error;
}

Metadata ValidMetadata() {
  Metadata fields{{"general.architecture", std::string{"qwen4exp"}}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 48},
           {"embedding_length", 2560},
           {"context_length", 262144},
           {"hyper_connection.count", 4},
           {"hyper_connection.low_rank", 320},
           {"full_attention_interval", 4},
           {"attention.head_count", 24},
           {"attention.head_count_kv", 2},
           {"attention.key_length", 256},
           {"rope.dimension_count", 64},
           {"attention.indexer.head_count", 4},
           {"attention.indexer.key_length", 128},
           {"attention.indexer.top_k", 2048},
           {"ssm.conv_kernel", 4},
           {"ssm.state_size", 128},
           {"ssm.group_count", 16},
           {"ssm.time_step_rank", 48},
           {"ssm.inner_size", 6144},
           {"expert_count", 512},
           {"expert_used_count", 10},
           {"expert_feed_forward_length", 640},
           {"expert_shared_feed_forward_length", 640}}) {
    fields[std::string{"qwen4exp."} + key] = value;
  }
  fields["qwen4exp.attention.layer_norm_rms_epsilon"] = 1e-6F;
  fields["qwen4exp.rope.freq_base"] = 1e7F;
  fields["qwen4exp.rope.dimension_sections"] =
      std::vector<std::uint64_t>{11, 11, 10, 0};
  std::vector<std::uint64_t> ratios(48);
  for (std::size_t i = 3; i < ratios.size(); i += 4) {
    ratios[i] = 4;
  }
  fields["qwen4exp.attention.compress_ratios"] = ratios;
  return fields;
}

constexpr std::uint64_t kHidden = 2560;
constexpr std::uint64_t kVocab = 64;
// The first tensor the binder asks for after the two head tensors.
const char* const kHeadBound = "missing tensor output_hc_norm.weight";

std::vector<TensorSpec> HeadTensors(core::GgmlType embedding,
                                    core::GgmlType head,
                                    bool with_head = true) {
  std::vector<TensorSpec> tensors{
      TensorSpec{"token_embd.weight", embedding, kHidden, kVocab}};
  if (with_head) {
    tensors.push_back(TensorSpec{"output.weight", head, kHidden, kVocab});
  }
  return tensors;
}

void CheckHeadFormats() {
  const auto fields = ValidMetadata();
  // Control: the format the shipped artifact carries today still binds.
  Require(BindError(fields, HeadTensors(core::GgmlType::kQ5_K,
                                        core::GgmlType::kQ6_K)) == kHeadBound,
          "a Q6_K head stopped binding");
  // The new configuration: the head stored as either ROCmFP4 form.
  Require(BindError(fields, HeadTensors(core::GgmlType::kQ5_K,
                                        core::GgmlType::kQ4_0_ROCMFP4_FAST)) ==
              kHeadBound,
          "a ROCmFP4_FAST head was rejected");
  Require(BindError(fields, HeadTensors(core::GgmlType::kQ5_K,
                                        core::GgmlType::kQ4_0_ROCMFP4)) ==
              kHeadBound,
          "a dual-scale ROCmFP4 head was rejected");
  // F32 stays on the shared list.
  Require(BindError(fields, HeadTensors(core::GgmlType::kQ5_K,
                                        core::GgmlType::kF32)) == kHeadBound,
          "an F32 head was rejected");
  // A routed-expert format is still not a valid head.
  Require(BindError(fields, HeadTensors(core::GgmlType::kQ5_K,
                                        core::GgmlType::kQ4_K)) ==
              "tensor output.weight has unsupported format Q4_K",
          "a Q4_K head was accepted");
}

void CheckEmbeddingList() {
  const auto fields = ValidMetadata();
  // token_embd keeps the dense-projection list: neither FP4 form may reach
  // the embedding role.
  Require(BindError(fields, HeadTensors(core::GgmlType::kQ4_0_ROCMFP4_FAST,
                                        core::GgmlType::kQ6_K)) ==
              "tensor token_embd.weight has unsupported format "
              "Q4_0_ROCMFP4_FAST",
          "a ROCmFP4_FAST embedding table was accepted");
  Require(BindError(fields, HeadTensors(core::GgmlType::kQ4_0_ROCMFP4,
                                        core::GgmlType::kQ6_K)) ==
              "tensor token_embd.weight has unsupported format "
              "Q4_0_ROCMFP4",
          "a dual-scale ROCmFP4 embedding table was accepted");
}

void CheckTiedFallback() {
  // Without output.weight the head is the embedding table; the fallback must
  // survive the change to the output list.
  const auto fields = ValidMetadata();
  Require(BindError(fields,
                    HeadTensors(core::GgmlType::kQ5_K, core::GgmlType::kQ5_K,
                                /*with_head=*/false)) == kHeadBound,
          "a tied head stopped binding");
}

void CheckHeadRowGeometry() {
  // The head's device row size decides the per-token read and the draft
  // table's stride; both are keyed off TensorRef::RowBytes.
  qfn::TensorRef fast;
  fast.type = core::GgmlType::kQ4_0_ROCMFP4_FAST;
  fast.cols = 2560;
  fast.rows = 248320;
  Require(fast.RowBytes() == 1360, "ROCmFP4_FAST head row is not 1360 bytes");
  Require(fast.SizeBytes() == std::size_t{337715200},
          "ROCmFP4_FAST head extent changed");
  qfn::TensorRef dual;
  dual.type = core::GgmlType::kQ4_0_ROCMFP4;
  dual.cols = 2560;
  dual.rows = 248320;
  Require(dual.RowBytes() == 1440, "ROCmFP4 head row is not 1440 bytes");
  Require(dual.SizeBytes() == std::size_t{357580800},
          "ROCmFP4 head extent changed");
}

}  // namespace

int main() {
  CheckHeadFormats();
  CheckEmbeddingList();
  CheckTiedFallback();
  CheckHeadRowGeometry();
  std::cout << "Flash-Next head binding checks passed.\n";
  return 0;
}
