// CPU oracle for ROCmFP4 types 100/101. Synthetic blocks are checked against
// an independent codebook table (not quant::'s own arithmetic). Optional
// argv[1] is a GGUF: every type-100/101 tensor's first and last row is
// checked the same way, then QwenModelWeights::LoadFromGguf is attempted.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/ggml_gemm.hpp"
#include "src/models/qwen/gemm_route.hpp"
#include "src/models/qwen/state.hpp"

namespace {

int Fail(const std::string& msg) {
  std::cerr << "FAIL " << msg << '\n';
  std::exit(1);
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

void FillDual(std::uint8_t* blk, std::uint8_t e0, std::uint8_t e1) {
  for (int j = 0; j < 16; ++j) {
    const unsigned lo = static_cast<unsigned>(j) & 0x0Fu;
    const unsigned hi = static_cast<unsigned>(15 - j) & 0x0Fu;
    blk[j] = static_cast<std::uint8_t>(lo | (hi << 4));
  }
  blk[16] = e0;
  blk[17] = e1;
}

void FillFast(std::uint8_t* blk, std::uint8_t e) {
  for (int j = 0; j < 16; ++j) {
    blk[j] = static_cast<std::uint8_t>((j * 3) & 0xFF);
  }
  blk[16] = e;
}

void ExpectRow(superroc::core::GgmlType type, const std::uint8_t* raw,
               std::size_t k, const std::string& label) {
  const std::size_t stride =
      type == superroc::core::GgmlType::kQ4_0_ROCMFP4 ? 18 : 17;
  std::vector<float> got(k);
  superroc::quant::Dequantize(type, raw, got.data(), k);
  std::vector<float> x(k);
  float acc = 0.0F;
  for (std::size_t i = 0; i < k; ++i) {
    const std::size_t b = i / 32;
    const int local = static_cast<int>(i % 32);
    const std::uint8_t* blk = raw + b * stride;
    const std::uint8_t byte = blk[local & 15];
    const unsigned nib = local < 16 ? (byte & 0x0Fu) : (byte >> 4);
    const std::uint8_t eb =
        (type == superroc::core::GgmlType::kQ4_0_ROCMFP4 && local >= 16)
            ? blk[17]
            : blk[16];
    const float expect = static_cast<float>(kTable[nib]) * TableScale(eb);
    if (got[i] != expect) {
      std::cerr << "FAIL " << label << " i=" << i << " got=" << got[i]
                << " expect=" << expect << '\n';
      std::exit(1);
    }
    x[i] = static_cast<float>((static_cast<int>(i) % 7) - 3);
    acc += expect * x[i];
  }
  const float dot = superroc::quant::Dot(type, raw, x, k);
  if (dot != acc) {
    std::cerr << "FAIL " << label << " dot got=" << dot << " acc=" << acc
              << '\n';
    std::exit(1);
  }
  if (!superroc::quant::IsSupported(type)) {
    Fail("IsSupported " + label);
  }
}

void Synthetic() {
  std::vector<std::uint8_t> dual(18 * 4);
  FillDual(dual.data(), 0x00, 0x01);
  FillDual(dual.data() + 18, 0x7e, 0x7f);
  FillDual(dual.data() + 36, 0x08, 0x10);
  FillDual(dual.data() + 54, 0x7d, 0x04);
  ExpectRow(superroc::core::GgmlType::kQ4_0_ROCMFP4, dual.data(), 128, "dual");

  std::vector<std::uint8_t> fast(17 * 4);
  FillFast(fast.data(), 0x00);
  FillFast(fast.data() + 17, 0x7f);
  FillFast(fast.data() + 34, 0x7e);
  FillFast(fast.data() + 51, 0x3c);
  ExpectRow(superroc::core::GgmlType::kQ4_0_ROCMFP4_FAST, fast.data(), 128,
            "fast");

  using superroc::models::qwen::DescribeQwenGemmFormat;
  using superroc::models::qwen::QwenGemmMode;
  using superroc::models::qwen::ResolveQwenGemmRoute;
  for (auto type : {superroc::core::GgmlType::kQ4_0_ROCMFP4,
                    superroc::core::GgmlType::kQ4_0_ROCMFP4_FAST}) {
    if (superroc::quant::QuantizedBlockElements(type) != 32) {
      Fail("block elements");
    }
    const std::size_t stride =
        type == superroc::core::GgmlType::kQ4_0_ROCMFP4 ? 18 : 17;
    if (superroc::quant::QuantizedRowBytes(type, 128) != stride * 4) {
      Fail("row bytes");
    }
    const auto fmt = DescribeQwenGemmFormat(type);
    if (!fmt.quantized || !fmt.cpu_direct || !fmt.hip_decode_direct ||
        !fmt.hip_prefill_direct || fmt.block_elements != 32) {
      std::cerr << "FAIL route caps for type " << static_cast<int>(type)
                << '\n';
      std::exit(1);
    }
    const auto cpu = ResolveQwenGemmRoute({.type = type,
                                           .batch_size = 1,
                                           .m = 4,
                                           .k = 64,
                                           .mode = QwenGemmMode::kCpu});
    const auto dec = ResolveQwenGemmRoute({.type = type,
                                           .batch_size = 1,
                                           .m = 4,
                                           .k = 64,
                                           .mode = QwenGemmMode::kHipDecode});
    const auto pre = ResolveQwenGemmRoute({.type = type,
                                           .batch_size = 2,
                                           .m = 4,
                                           .k = 64,
                                           .mode = QwenGemmMode::kHipPrefill});
    if (!cpu.accepted() || !dec.accepted() || !pre.accepted()) {
      std::cerr << "FAIL route resolve type " << static_cast<int>(type) << '\n';
      std::exit(1);
    }
  }
  std::cout << "SYNTHETIC PASS dual+fast 128 els, route accepts 100/101\n";
}

void ScanGguf(const std::string& path) {
  std::string err;
  auto reader = superroc::core::GgufReader::OpenFile(path, &err);
  if (reader == nullptr) {
    std::cerr << "FAIL open " << path << " " << err << '\n';
    std::exit(1);
  }
  std::size_t n100 = 0;
  std::size_t n101 = 0;
  std::size_t rows = 0;
  for (const auto& tensor : reader->GetTensors()) {
    if (tensor.type != superroc::core::GgmlType::kQ4_0_ROCMFP4 &&
        tensor.type != superroc::core::GgmlType::kQ4_0_ROCMFP4_FAST) {
      continue;
    }
    if (tensor.type == superroc::core::GgmlType::kQ4_0_ROCMFP4) {
      ++n100;
    } else {
      ++n101;
    }
    const auto& dims = tensor.dimensions;
    if (dims.empty() || dims[0] == 0 || (dims[0] % 32) != 0) {
      std::cerr << "FAIL shape " << tensor.name << '\n';
      std::exit(1);
    }
    const std::size_t k = static_cast<std::size_t>(dims[0]);
    const std::size_t stride =
        superroc::quant::QuantizedRowBytes(tensor.type, k);
    if (stride == 0 || tensor.data == nullptr) {
      std::cerr << "FAIL row bytes " << tensor.name << '\n';
      std::exit(1);
    }
    std::size_t nrows = 1;
    for (std::size_t d = 1; d < dims.size(); ++d) {
      nrows *= static_cast<std::size_t>(dims[d]);
    }
    if (tensor.size_bytes < nrows * stride) {
      std::cerr << "FAIL size " << tensor.name << " bytes=" << tensor.size_bytes
                << " need=" << (nrows * stride) << '\n';
      std::exit(1);
    }
    const auto* base = static_cast<const std::uint8_t*>(tensor.data);
    ExpectRow(tensor.type, base, k, std::string(tensor.name) + " first");
    if (nrows > 1) {
      ExpectRow(tensor.type, base + (nrows - 1) * stride, k,
                std::string(tensor.name) + " last");
      rows += 2;
    } else {
      rows += 1;
    }
  }
  std::cout << "G1 PASS file=" << path << " tensors100=" << n100
            << " tensors101=" << n101 << " rows_checked=" << rows
            << " cosine=1 bitexact=1\n";
  std::string load_err;
  auto weights =
      superroc::models::QwenModelWeights::LoadFromGguf(*reader, &load_err);
  if (!weights.has_value()) {
    std::cout << "BIND FAIL " << load_err << '\n';
    std::exit(2);
  }
  std::cout << "BIND PASS layers=" << weights->config.num_layers
            << " hidden=" << weights->config.hidden_size << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  Synthetic();
  if (argc > 1) {
    ScanGguf(argv[1]);
  }
  return 0;
}
