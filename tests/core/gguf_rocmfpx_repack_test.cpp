#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_identity.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace {

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

// Minimal GGUF image builder (same layout as tests/core/gguf_reader_test.cpp).
class GgufBuilder {
public:
  GgufBuilder() {
    const char magic[4] = {'G', 'G', 'U', 'F'};
    AppendBytes(magic, 4);
    AppendPod(static_cast<std::uint32_t>(3));
    tensor_count_pos_ = buffer_.size();
    AppendPod(static_cast<std::uint64_t>(0));
    metadata_count_pos_ = buffer_.size();
    AppendPod(static_cast<std::uint64_t>(0));
  }

  void AddMetadataString(std::string_view key, std::string_view val) {
    AppendString(key);
    AppendPod(
        static_cast<std::uint32_t>(superroc::core::GgufValueType::kString));
    AppendString(val);
    metadata_count_++;
  }

  void AddTensor(std::string_view name, const std::vector<std::uint64_t>& dims,
                 superroc::core::GgmlType type, std::uint64_t offset) {
    tensors_to_write_.push_back({std::string(name), dims, type, offset});
  }

  std::vector<std::uint8_t> Build(std::size_t payload_bytes) {
    std::memcpy(buffer_.data() + metadata_count_pos_, &metadata_count_,
                sizeof(metadata_count_));
    const std::uint64_t tensor_count = tensors_to_write_.size();
    std::memcpy(buffer_.data() + tensor_count_pos_, &tensor_count,
                sizeof(tensor_count));
    for (const auto& t : tensors_to_write_) {
      AppendString(t.name);
      AppendPod(static_cast<std::uint32_t>(t.dims.size()));
      for (auto d : t.dims) {
        AppendPod(d);
      }
      AppendPod(static_cast<std::uint32_t>(t.type));
      AppendPod(t.offset);
    }
    std::size_t rem = buffer_.size() % 32;
    if (rem != 0) {
      buffer_.resize(buffer_.size() + (32 - rem), 0);
    }
    for (const auto& tensor : tensors_to_write_) {
      std::size_t elements = 1;
      for (auto dimension : tensor.dims)
        elements *= dimension;
      payload_bytes = std::max(
          payload_bytes,
          static_cast<std::size_t>(tensor.offset) +
              superroc::quant::EncodedSizeBytes(tensor.type, elements));
    }
    buffer_.resize(buffer_.size() + payload_bytes, 0xAB);
    return buffer_;
  }

private:
  struct TensorRecord {
    std::string name;
    std::vector<std::uint64_t> dims;
    superroc::core::GgmlType type;
    std::uint64_t offset;
  };

  void AppendBytes(const void* data, std::size_t len) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    buffer_.insert(buffer_.end(), p, p + len);
  }

  template<typename T>
  void AppendPod(T val) {
    AppendBytes(&val, sizeof(T));
  }

  void AppendString(std::string_view s) {
    const std::uint64_t len = s.size();
    AppendPod(len);
    AppendBytes(s.data(), len);
  }

  std::vector<std::uint8_t> buffer_;
  std::size_t tensor_count_pos_{0};
  std::size_t metadata_count_pos_{0};
  std::uint64_t metadata_count_{0};
  std::vector<TensorRecord> tensors_to_write_;
};

// (a) Repack is exact: Q8_0 dequant of the repacked block == Q8_0_ROCMFPX
// dequant, bit for bit, for every exponent byte.
void TestRepackExactAllExponents() {
  for (int e = 0; e < 256; ++e) {
    std::uint8_t src[33];
    for (int i = 0; i < 32; ++i) {
      src[i] = static_cast<std::uint8_t>((i * 37 + e) % 256);
    }
    src[32] = static_cast<std::uint8_t>(e);
    std::uint8_t repacked[34] = {};
    superroc::quant::RepackQ8_0_ROCMFPXToQ8_0(src, repacked, 1);
    float direct[32] = {};
    float via_q8_0[32] = {};
    superroc::quant::DequantizeQ8_0_ROCMFPX(src, direct, 32);
    superroc::quant::DequantizeQ8_0(repacked, via_q8_0, 32);
    if (std::memcmp(direct, via_q8_0, sizeof(direct)) != 0) {
      std::cerr << "Mismatch at e=" << e << "\n";
      std::exit(1);
    }
    // The repacked scale must be the fp16 of the UE4M3 scale, with the int8
    // payload copied verbatim.
    const std::uint16_t want_d = superroc::quant::FloatToFp16(
        superroc::quant::Ue4m3ScaleToFloat(static_cast<std::uint8_t>(e)));
    std::uint16_t got_d = 0;
    std::memcpy(&got_d, repacked, 2);
    Expect(got_d == want_d, "repacked d is FloatToFp16(Ue4M3(e))");
    Expect(std::memcmp(repacked + 2, src, 32) == 0, "qs copied verbatim");
  }
}

// (b) End-to-end reader repack of an in-memory image.
void TestReaderRepackMemoryImage() {
  using superroc::core::GgmlType;
  constexpr std::uint64_t kF32Elems = 64 * 2;
  constexpr std::uint64_t kFpxElems = 64 * 3;  // 6 blocks
  constexpr std::size_t kF32Bytes = kF32Elems * 4;
  constexpr std::size_t kFpxBytes = 6 * 33;
  constexpr std::size_t kRepackedBytes = 6 * 34;  // 3*2*34

  GgufBuilder builder;
  builder.AddMetadataString("general.architecture", "qwen35");
  builder.AddTensor("blk.0.attn_q.weight", {64, 2}, GgmlType::kF32, 0);
  builder.AddTensor("blk.0.ffn_up.weight", {64, 3}, GgmlType::kQ8_0_ROCMFPX,
                    kF32Bytes);
  auto image = builder.Build(kF32Bytes + kFpxBytes);

  const std::size_t data_base = image.size() - (kF32Bytes + kFpxBytes);
  std::vector<float> f32_pattern(kF32Elems);
  for (std::size_t i = 0; i < kF32Elems; ++i) {
    f32_pattern[i] = static_cast<float>(i) * 0.5F - 3.0F;
  }
  std::memcpy(image.data() + data_base, f32_pattern.data(), kF32Bytes);
  std::vector<std::uint8_t> fpx_pattern(kFpxBytes);
  for (int b = 0; b < 6; ++b) {
    for (int i = 0; i < 32; ++i) {
      fpx_pattern[static_cast<std::size_t>(b * 33 + i)] =
          static_cast<std::uint8_t>((i * 37 + b * 41) % 256);
    }
    fpx_pattern[static_cast<std::size_t>(b * 33 + 32)] =
        static_cast<std::uint8_t>(b * 41);
  }
  std::memcpy(image.data() + data_base + kF32Bytes, fpx_pattern.data(),
              kFpxBytes);

  std::string error;
  auto reader = superroc::core::GgufReader::OpenMemory(image.data(),
                                                       image.size(), &error);
  Expect(reader != nullptr, "OpenMemory of 103 image succeeds: " + error);

  // Snapshot expectations before the repack drops the source pages.
  std::vector<std::uint8_t> want_repacked(kRepackedBytes);
  superroc::quant::RepackQ8_0_ROCMFPXToQ8_0(fpx_pattern.data(),
                                            want_repacked.data(), 6);

  auto repacked =
      superroc::core::GgufReader::RepackRocmFpx(std::move(reader), &error);
  Expect(repacked != nullptr, "RepackRocmFpx succeeds: " + error);

  const auto* f32 = repacked->FindTensor("blk.0.attn_q.weight");
  Expect(f32 != nullptr && f32->type == GgmlType::kF32, "F32 tensor kept");
  Expect(f32->size_bytes == kF32Bytes, "F32 size kept");
  Expect(std::memcmp(f32->data, f32_pattern.data(), kF32Bytes) == 0,
         "F32 bytes identical after repack");

  const auto* fpx = repacked->FindTensor("blk.0.ffn_up.weight");
  Expect(fpx != nullptr && fpx->type == GgmlType::kQ8_0, "103 -> Q8_0");
  Expect(fpx->size_bytes == kRepackedBytes, "repacked size 3*2*34");
  Expect(std::memcmp(fpx->data, want_repacked.data(), kRepackedBytes) == 0,
         "repacked bytes match direct repack");

  const auto regions = repacked->GetMappedRegions();
  Expect(regions.size() == 1, "one anonymous mapped region");
  Expect(regions[0].file_descriptor == -1, "region fd is -1");
  Expect(regions[0].anonymous, "region is anonymous");
  for (const auto* tensor : {f32, fpx}) {
    const auto begin = reinterpret_cast<std::uintptr_t>(tensor->data);
    const auto base = reinterpret_cast<std::uintptr_t>(regions[0].data);
    Expect(
        begin >= base && tensor->size_bytes <= regions[0].size - (begin - base),
        "tensor data inside the anonymous region");
  }
  Expect(repacked->GetMetadataString("general.architecture") == "qwen35",
         "metadata still readable after repack");
}

// (c) A file with no 103 tensor returns the same reader.
void TestNoFpxReturnsSameReader() {
  GgufBuilder builder;
  builder.AddMetadataString("general.architecture", "qwen35");
  builder.AddTensor("blk.0.attn_q.weight", {64, 2},
                    superroc::core::GgmlType::kF32, 0);
  auto image = builder.Build(64 * 2 * 4);

  std::string error;
  auto reader = superroc::core::GgufReader::OpenMemory(image.data(),
                                                       image.size(), &error);
  Expect(reader != nullptr, "OpenMemory of plain image succeeds: " + error);
  const auto* before = reader.get();
  auto same =
      superroc::core::GgufReader::RepackRocmFpx(std::move(reader), &error);
  Expect(same.get() == before, "no-103 file returns the same reader");
}

// (d) repacked readers hash the source bytes with a repack tag; plain
// readers expose mapped regions as their identity regions.
void TestIdentityRegionsRepackTag() {
  using superroc::core::GgmlType;
  constexpr std::uint64_t kF32Elems = 64 * 2;
  constexpr std::uint64_t kFpxElems = 64 * 3;  // 6 blocks
  constexpr std::size_t kF32Bytes = kF32Elems * 4;
  constexpr std::size_t kFpxBytes = 6 * 33;

  GgufBuilder builder;
  builder.AddMetadataString("general.architecture", "qwen35");
  builder.AddTensor("blk.0.attn_q.weight", {64, 2}, GgmlType::kF32, 0);
  builder.AddTensor("blk.0.ffn_up.weight", {64, 3}, GgmlType::kQ8_0_ROCMFPX,
                    kF32Bytes);
  auto image = builder.Build(kF32Bytes + kFpxBytes);

  std::string error;
  auto base = superroc::core::GgufReader::OpenMemory(image.data(), image.size(),
                                                     &error);
  Expect(base != nullptr, "OpenMemory of 103 image succeeds: " + error);
  auto repacked =
      superroc::core::GgufReader::RepackRocmFpx(std::move(base), &error);
  Expect(repacked != nullptr, "RepackRocmFpx succeeds: " + error);
  Expect(repacked->IsRocmFpxRepacked(), "repacked flag is set");
  const auto identity = repacked->GetIdentityRegions();
  Expect(identity.size() == 1, "identity has one source region");
  Expect(identity[0].data == image.data(), "identity data is image pointer");
  Expect(identity[0].size == image.size(), "identity size is image size");

  auto plain = superroc::core::GgufReader::OpenMemory(image.data(),
                                                      image.size(), &error);
  Expect(plain != nullptr, "second OpenMemory succeeds: " + error);
  Expect(!plain->IsRocmFpxRepacked(), "unrepacked flag is clear");
  const std::string repacked_hex = superroc::core::GgufIdentityHex(*repacked);
  const std::string plain_hex = superroc::core::GgufIdentityHex(*plain);
  Expect(repacked_hex != plain_hex, "repack tag changes the identity");

  GgufBuilder nobuilder;
  nobuilder.AddMetadataString("general.architecture", "qwen35");
  nobuilder.AddTensor("blk.0.attn_q.weight", {64, 2}, GgmlType::kF32, 0);
  auto noimage = nobuilder.Build(64 * 2 * 4);
  auto noreader = superroc::core::GgufReader::OpenMemory(
      noimage.data(), noimage.size(), &error);
  Expect(noreader != nullptr, "OpenMemory of plain image succeeds: " + error);
  Expect(!noreader->IsRocmFpxRepacked(), "plain reader flag is clear");
  const auto no_mapped = noreader->GetMappedRegions();
  const auto no_identity = noreader->GetIdentityRegions();
  Expect(no_identity.size() == no_mapped.size(),
         "plain identity size matches mapped");
  for (std::size_t i = 0; i < no_mapped.size(); ++i) {
    Expect(no_identity[i].data == no_mapped[i].data,
           "plain identity data matches mapped");
    Expect(no_identity[i].size == no_mapped[i].size,
           "plain identity size matches mapped");
  }
}

// (e) a kMetadata open of an image with a 103 tensor keeps type 103
// with no repack (metadata/tensor-table only).
void TestMetadataRoleSkipsRepack() {
  using superroc::core::GgmlType;
  constexpr std::uint64_t kF32Elems = 64 * 2;
  constexpr std::size_t kF32Bytes = kF32Elems * 4;
  constexpr std::size_t kFpxBytes = 2 * 33;  // 64 elems, 2 blocks

  GgufBuilder builder;
  builder.AddMetadataString("general.architecture", "qwen35");
  builder.AddTensor("blk.0.attn_q.weight", {64, 2}, GgmlType::kF32, 0);
  builder.AddTensor("blk.0.ffn_up.weight", {64, 1}, GgmlType::kQ8_0_ROCMFPX,
                    kF32Bytes);
  auto image = builder.Build(kF32Bytes + kFpxBytes);

  const auto path =
      std::filesystem::temp_directory_path() / "fpxrepack_kmetadata_test.gguf";
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    Expect(out.good(), "temp GGUF file opens for write");
    out.write(reinterpret_cast<const char*>(image.data()),
              static_cast<std::streamsize>(image.size()));
    Expect(out.good(), "temp GGUF file writes fully");
  }

  std::string error;
  auto reader = superroc::core::GgufReader::OpenFile(
      path, &error, superroc::core::GgufReader::FileRole::kMetadata);
  std::error_code remove_error;
  std::filesystem::remove(path, remove_error);
  Expect(reader != nullptr, "kMetadata open of 103 file succeeds: " + error);
  Expect(remove_error.value() == 0, "temp GGUF file removed");

  const auto* fpx = reader->FindTensor("blk.0.ffn_up.weight");
  Expect(fpx != nullptr, "103 tensor present in kMetadata reader");
  Expect(fpx->type == GgmlType::kQ8_0_ROCMFPX,
         "kMetadata open keeps type 103 (no repack)");
  Expect(fpx->size_bytes == kFpxBytes, "kMetadata open keeps 103 size");
  Expect(!reader->IsRocmFpxRepacked(), "kMetadata reader repack flag is clear");
  Expect(reader->GetMetadataString("general.architecture") == "qwen35",
         "kMetadata reader exposes metadata");
  const auto identity = reader->GetIdentityRegions();
  const auto mapped = reader->GetMappedRegions();
  Expect(identity.size() == mapped.size(),
         "kMetadata identity size matches mapped");
  for (std::size_t i = 0; i < mapped.size(); ++i) {
    Expect(identity[i].data == mapped[i].data,
           "kMetadata identity data matches mapped");
    Expect(identity[i].size == mapped[i].size,
           "kMetadata identity size matches mapped");
  }
}

}  // namespace

int main() {
  TestRepackExactAllExponents();
  TestReaderRepackMemoryImage();
  TestNoFpxReturnsSameReader();
  TestIdentityRegionsRepackTag();
  TestMetadataRoleSkipsRepack();
  std::cout << "All ROCmFPX repack tests passed successfully!\n";
  return 0;
}
