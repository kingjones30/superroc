#ifndef SUPERROC_MODELS_QWEN_HIP_DETAIL_INT4_ROUTE_FILE_HPP_
#define SUPERROC_MODELS_QWEN_HIP_DETAIL_INT4_ROUTE_FILE_HPP_
#include <sys/types.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// The int4 prefill route weights file (SUPERROC_INT4_WEIGHTS): its layout and
// the reader of its tables. Host code only, so the checks run in the CPU tests.
namespace superroc::hip::detail {
// The tensors of a layer the route can hold, and the input set each reads: set
// 0 is the FFN input, set 1 the attention input.
enum Int4Tensor : std::uint32_t {
  kInt4FfnGate = 0,
  kInt4FfnUp = 1,
  kInt4AttnQkv = 2,
  kInt4AttnGate = 3,
  kInt4AttnQ = 4,
  kInt4TensorKinds = 5
};
constexpr std::uint32_t kInt4Sets = 2;
constexpr std::uint32_t Int4SetOf(std::uint32_t tensor) {
  return tensor >= kInt4AttnQkv ? 1U : 0U;
}
inline const char* Int4TensorName(std::uint32_t tensor) {
  constexpr const char* kNames[kInt4TensorKinds] = {
      "ffn_gate", "ffn_up", "attn_qkv", "attn_gate", "attn_q"};
  return kNames[tensor];
}

// The Flash-Next tensors of a layer the route can hold. Names are the GGUF
// families the Flash-Next executor registers; `attn_qkv` is the attention
// stack, `ssm_qkv`/`ssm_gate` the linear layers' projections, `*_shexp` the
// shared expert and `*_exps` the routed experts. Sets match the qwen ones:
// set 0 is the FFN input, set 1 the mixer input.
enum FnInt4Tensor : std::uint32_t {
  kFnInt4SsmQkv = 0,
  kFnInt4SsmGate = 1,
  kFnInt4AttnQkv = 2,
  kFnInt4ShexpGate = 3,
  kFnInt4ShexpUp = 4,
  kFnInt4GateExps = 5,
  kFnInt4UpExps = 6,
  kFnInt4TensorKinds = 7
};
constexpr std::uint32_t FnInt4SetOf(std::uint32_t tensor) {
  return tensor >= kFnInt4ShexpGate ? 0U : 1U;
}
inline const char* FnInt4TensorName(std::uint32_t tensor) {
  constexpr const char* kNames[kFnInt4TensorKinds] = {
      "ssm_qkv",      "ssm_gate",      "attn_qkv",   "ffn_gate_shexp",
      "ffn_up_shexp", "ffn_gate_exps", "ffn_up_exps"};
  return kNames[tensor];
}

// A route weights file.
// After the header: layers * kInt4TensorKinds uint64 offsets (0 for a tensor
// not in the file); kInt4Sets * layers * columns float32 input scales; per
// tensor its packed weights, then one float32 step per row. Tensors follow
// the tables and each other with no gap.
// each other. `reserved` names the kind table the offsets column holds:
// 0 = the qwen tensors above (kInt4TensorKinds), 1 = the Flash-Next
// tensors (kFnInt4TensorKinds).
struct Int4WeightsHeader {
  char magic[4] = {0, 0, 0, 0};
  std::uint32_t layers = 0;
  std::uint32_t columns = 0;
  std::uint32_t fine_columns = 0;
  float clip = 0.F;
  std::uint32_t reserved = 0;
};

// The tables at the head of a route weights file.
struct Int4RouteTables {
  Int4WeightsHeader head;
  std::vector<std::uint64_t> weights_offset;  // layers * kinds
  std::vector<float> sigma;                   // kInt4Sets * layers * columns
  // The kind table the head names: kInt4TensorKinds or kFnInt4TensorKinds.
  std::uint32_t kinds = kInt4TensorKinds;
  std::uint64_t file_bytes = 0;
};

// The most the header and the two tables may take.
constexpr std::uint64_t kInt4MaxTablesBytes = std::uint64_t{1} << 30;

// Reads and checks the tables of a route weights file. The file comes with a
// model download: nothing is sized from its header before the header's tables
// are known to fit in the file, every input scale is a finite, non-negative
// number (the plan sorts and groups columns by them), and every tensor starts
// after the tables and inside the file. Returns nullptr when the tables are
// good, else what is wrong with the file.
inline const char* ReadInt4RouteTables(std::FILE* file, Int4RouteTables& out) {
  Int4WeightsHeader& head = out.head;
  if (file == nullptr || fseeko(file, 0, SEEK_SET) != 0 ||
      std::fread(&head, sizeof(head), 1, file) != 1 ||
      std::memcmp(head.magic, "A4W3", 4) != 0 || head.layers == 0 ||
      head.columns == 0 || head.columns > 65536 ||
      head.fine_columns > head.columns || !std::isfinite(head.clip) ||
      head.clip <= 0.F || head.reserved > 1) {
    return "SUPERROC_INT4_WEIGHTS is not a route weights file";
  }
  // `reserved` names the kind table: 0 = the qwen tensors, 1 = the
  // Flash-Next ones (one offsets column per kind in that table).
  const std::uint32_t kinds =
      head.reserved == 1 ? static_cast<std::uint32_t>(kFnInt4TensorKinds)
                         : static_cast<std::uint32_t>(kInt4TensorKinds);
  const std::uint64_t offsets_bytes = std::uint64_t{head.layers} * kinds * 8;
  const std::uint64_t sigma_bytes =
      std::uint64_t{kInt4Sets} * head.layers * head.columns * sizeof(float);
  const std::uint64_t tables_end = sizeof(head) + offsets_bytes + sigma_bytes;
  // The tables of a real model are a few megabytes; a header that asks for
  // more than this is not one, however large the file is.
  if (tables_end > kInt4MaxTablesBytes) {
    return "SUPERROC_INT4_WEIGHTS is not a route weights file";
  }
  off_t file_bytes = -1;
  if (fseeko(file, 0, SEEK_END) == 0) {
    file_bytes = ftello(file);
  }
  if (file_bytes < 0 || tables_end > static_cast<std::uint64_t>(file_bytes) ||
      fseeko(file, static_cast<off_t>(sizeof(head)), SEEK_SET) != 0) {
    return "SUPERROC_INT4_WEIGHTS is short";
  }
  out.kinds = kinds;
  out.weights_offset.resize(std::size_t{head.layers} * kinds);
  out.sigma.resize(std::size_t{kInt4Sets} * head.layers * head.columns);
  if (std::fread(out.weights_offset.data(), 8, out.weights_offset.size(),
                 file) != out.weights_offset.size() ||
      std::fread(out.sigma.data(), sizeof(float), out.sigma.size(), file) !=
          out.sigma.size()) {
    return "SUPERROC_INT4_WEIGHTS is short";
  }
  for (const float scale : out.sigma) {
    if (!std::isfinite(scale) || scale < 0.F) {
      return "SUPERROC_INT4_WEIGHTS holds an input scale that is not a "
             "finite, non-negative number";
    }
  }
  for (const std::uint64_t offset : out.weights_offset) {
    if (offset != 0 && (offset < tables_end ||
                        offset >= static_cast<std::uint64_t>(file_bytes))) {
      return "SUPERROC_INT4_WEIGHTS names weights outside the file";
    }
  }
  out.file_bytes = static_cast<std::uint64_t>(file_bytes);
  return nullptr;
}

// The bytes the file holds for the tensor that starts at `offset`: up to the
// next tensor, or to the end of the file. A model's tensor has rows x (packed
// width / 2) weight bytes and one float per row; a file whose tensor has
// another size was made for another model.
inline std::uint64_t Int4TensorExtent(
    const std::vector<std::uint64_t>& weights_offset, std::uint64_t file_bytes,
    std::uint64_t offset) {
  std::uint64_t end = file_bytes;
  for (const std::uint64_t other : weights_offset) {
    if (other > offset && other < end) {
      end = other;
    }
  }
  return end > offset ? end - offset : 0;
}
}  // namespace superroc::hip::detail
#endif  // SUPERROC_MODELS_QWEN_HIP_DETAIL_INT4_ROUTE_FILE_HPP_
