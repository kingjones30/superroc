// Unit tests for the reader of the int4 prefill route weights file
// (SUPERROC_INT4_WEIGHTS). The file comes with a model download, so every
// field of it is checked before use. CPU only. Plain main, PASS/FAIL lines,
// nonzero exit on any failure.

#include "src/models/qwen/hip/detail/int4_route_file.hpp"

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

using superroc::hip::detail::Int4RouteTables;
using superroc::hip::detail::Int4TensorExtent;
using superroc::hip::detail::Int4WeightsHeader;
using superroc::hip::detail::kInt4Sets;
using superroc::hip::detail::kInt4TensorKinds;
using superroc::hip::detail::ReadInt4RouteTables;

int failures = 0;

void Check(bool condition, const std::string& message) {
  std::cout << (condition ? "PASS " : "FAIL ") << message << "\n";
  if (!condition) {
    ++failures;
  }
}

// A route file in memory: the header, the two tables, then `tail` bytes.
struct RouteImage {
  Int4WeightsHeader head;
  std::vector<std::uint64_t> offsets;
  std::vector<float> sigma;
  std::size_t tail = 0;

  RouteImage(std::uint32_t layers, std::uint32_t columns) {
    std::memcpy(head.magic, "A4W3", 4);
    head.layers = layers;
    head.columns = columns;
    head.fine_columns = columns / 2;
    head.clip = 3.F;
    offsets.assign(std::size_t{layers} * kInt4TensorKinds, 0);
    sigma.assign(std::size_t{kInt4Sets} * layers * columns, 0.5F);
  }
  [[nodiscard]] std::uint64_t TablesEnd() const {
    return sizeof(head) + offsets.size() * 8 + sigma.size() * sizeof(float);
  }
  [[nodiscard]] std::string Bytes() const {
    std::string out(reinterpret_cast<const char*>(&head), sizeof(head));
    out.append(reinterpret_cast<const char*>(offsets.data()),
               offsets.size() * 8);
    out.append(reinterpret_cast<const char*>(sigma.data()),
               sigma.size() * sizeof(float));
    out.append(tail, '\0');
    return out;
  }
};

std::filesystem::path temp_dir;

// Writes `bytes` to a file and runs the reader on it.
const char* Read(const std::string& bytes, Int4RouteTables& tables) {
  const auto path = temp_dir / "route.a4w";
  std::FILE* out = std::fopen(path.c_str(), "wb");
  if (out == nullptr ||
      std::fwrite(bytes.data(), 1, bytes.size(), out) != bytes.size()) {
    std::cout << "FAIL cannot write the test file\n";
    ++failures;
  }
  if (out != nullptr) {
    std::fclose(out);
  }
  std::FILE* in = std::fopen(path.c_str(), "rb");
  const char* wrong = ReadInt4RouteTables(in, tables);
  if (in != nullptr) {
    std::fclose(in);
  }
  return wrong;
}

bool Refused(const std::string& bytes, std::string_view message) {
  Int4RouteTables tables;
  const char* wrong = Read(bytes, tables);
  return wrong != nullptr &&
         std::string_view(wrong).find(message) != std::string_view::npos;
}

void TestGoodFile() {
  RouteImage image(2, 8);
  image.tail = 96;
  image.offsets[0] = image.TablesEnd();
  image.offsets[1] = image.TablesEnd() + 40;
  image.offsets[7] = image.TablesEnd() + 64;
  Int4RouteTables tables;
  const char* wrong = Read(image.Bytes(), tables);
  Check(wrong == nullptr, std::string("a well-formed file is read: ") +
                              (wrong != nullptr ? wrong : "ok"));
  Check(tables.head.layers == 2 && tables.head.columns == 8 &&
            tables.head.fine_columns == 4 && tables.head.clip == 3.F,
        "the header fields are returned");
  Check(tables.weights_offset.size() == 10 && tables.sigma.size() == 32,
        "the tables have the header's sizes");
  Check(tables.file_bytes == image.TablesEnd() + 96,
        "the file size is returned");
  Check(Int4TensorExtent(tables.weights_offset, tables.file_bytes,
                         image.offsets[0]) == 40,
        "a tensor ends where the next one starts");
  Check(Int4TensorExtent(tables.weights_offset, tables.file_bytes,
                         image.offsets[1]) == 24,
        "the next tensor is the nearest larger offset, in any table order");
  Check(Int4TensorExtent(tables.weights_offset, tables.file_bytes,
                         image.offsets[7]) == 32,
        "the last tensor ends at the end of the file");
  Check(Int4TensorExtent(tables.weights_offset, tables.file_bytes,
                         tables.file_bytes + 8) == 0,
        "an offset past the file has no bytes");
}

void TestHeader() {
  const auto with = [](auto change) {
    RouteImage image(2, 8);
    change(image);
    return image.Bytes();
  };
  const std::string_view not_route = "is not a route weights file";
  Check(Refused(with([](RouteImage& i) { i.head.magic[3] = '2'; }), not_route),
        "another magic is refused");
  Check(Refused(with([](RouteImage& i) { i.head.layers = 0; }), not_route),
        "zero layers are refused");
  Check(Refused(with([](RouteImage& i) { i.head.columns = 0; }), not_route),
        "zero columns are refused");
  Check(Refused(with([](RouteImage& i) { i.head.columns = 65537; }), not_route),
        "more than 65536 columns are refused");
  Check(
      Refused(with([](RouteImage& i) { i.head.fine_columns = 9; }), not_route),
      "more fine columns than columns are refused");
  for (const float clip : {std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::infinity(), 0.F, -3.F}) {
    Check(
        Refused(with([clip](RouteImage& i) { i.head.clip = clip; }), not_route),
        "a clip of " + std::to_string(clip) + " is refused");
  }
  Check(Refused(std::string("A4W3"), not_route),
        "a file shorter than the header is refused");
  Int4RouteTables tables;
  Check(ReadInt4RouteTables(nullptr, tables) != nullptr,
        "a file that did not open is refused");
}

void TestTablesMustFit() {
  const auto header_only = [](std::uint32_t layers, std::uint32_t columns) {
    RouteImage image(1, 1);
    image.head.layers = layers;
    image.head.columns = columns;
    image.head.fine_columns = 64;
    std::string bytes(reinterpret_cast<const char*>(&image.head),
                      sizeof(image.head));
    bytes.append(64, '\0');
    return bytes;
  };
  // 4 billion layers of 65536 columns would be petabytes of tables, and 4096
  // layers 2 GiB: both are over the limit for tables, whatever the file holds.
  Check(Refused(header_only(0xFFFFFFFFU, 65536), "is not a route weights file"),
        "a header that asks for petabytes of tables is refused");
  Check(Refused(header_only(4096, 65536), "is not a route weights file"),
        "tables over the size limit are refused");
  // A real model's shape on an 88-byte file: refused before anything is sized.
  Check(Refused(header_only(64, 5120), "is short"),
        "tables larger than the file are refused before they are sized");
  RouteImage cut(2, 8);
  std::string short_bytes = cut.Bytes();
  short_bytes.resize(short_bytes.size() - 1);
  Check(Refused(short_bytes, "is short"), "tables one byte short are refused");
}

void TestScales() {
  for (const float scale : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(), -1.F}) {
    RouteImage image(2, 8);
    image.sigma[17] = scale;
    Check(Refused(image.Bytes(), "not a finite, non-negative number"),
          "an input scale of " + std::to_string(scale) + " is refused");
  }
  RouteImage zero(2, 8);
  zero.sigma[3] = 0.F;
  Int4RouteTables tables;
  Check(Read(zero.Bytes(), tables) == nullptr,
        "an input scale of zero is accepted");
}

void TestOffsets() {
  const std::string_view outside = "names weights outside the file";
  RouteImage past(2, 8);
  past.tail = 64;
  past.offsets[4] = past.TablesEnd() + 64;
  Check(Refused(past.Bytes(), outside),
        "an offset at the end of the file is refused");
  past.offsets[4] = past.TablesEnd() + (std::uint64_t{1} << 40);
  Check(Refused(past.Bytes(), outside), "an offset far past the file");
  RouteImage inside(2, 8);
  inside.tail = 64;
  inside.offsets[9] = sizeof(Int4WeightsHeader);
  Check(Refused(inside.Bytes(), outside),
        "an offset inside the tables is refused");
  inside.offsets[9] = inside.TablesEnd() - 1;
  Check(Refused(inside.Bytes(), outside),
        "an offset one byte before the end of the tables is refused");
  inside.offsets[9] = inside.TablesEnd();
  Int4RouteTables tables;
  Check(Read(inside.Bytes(), tables) == nullptr,
        "an offset right after the tables is accepted");
}

}  // namespace

int main() {
  temp_dir =
      std::filesystem::temp_directory_path() /
      ("superroc-int4-route-" + std::to_string(static_cast<long>(getpid())));
  std::filesystem::create_directories(temp_dir);
  TestGoodFile();
  TestHeader();
  TestTablesMustFit();
  TestScales();
  TestOffsets();
  std::filesystem::remove_all(temp_dir);
  if (failures != 0) {
    std::cout << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "all int4 route file checks passed\n";
  return 0;
}
