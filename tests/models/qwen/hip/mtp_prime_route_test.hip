// MTP prime route: the shape rule and the tiled prefill GEMM it selects.
//   shape: MtpPrimeTiledShapeOk accepts exactly the Q8_0/ROCmFP4 shapes
//          RunPrimeGemm may route through LaunchBatchedQuantGEMMPreQuantized,
//          and rejects every other type and every off-rule shape
//   gemm:  for every accepted case the tiled GEMM writes every output (no NaN
//          survives) without touching the 1 MiB 0xA5 guard tails behind the
//          Q8_1 activation scratch and the output buffer, and stays within 3%
//          of the reference RMS against the eight-row reference loop
//   bf16:  the BF16 entry (LaunchFloatToBfloat16 + LaunchBatchedQuantGEMM) on
//          the shape that overran its prime scratch runs clean, same bound
// Weights are random and seeded: ROCmFP4 blocks are packed with the layout and
// decoders of src/core/rocm_fp4/superroc_fp4_quant.hpp, Q8_0 rows with the
// block_q8_0 format of src/core/quant. Activations are random floats.
#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/rocm_fp4/superroc_fp4_quant.hpp"
#include "src/models/qwen/hip/mtp/prime_route.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using superroc::core::GgmlType;
using superroc::hip::LaunchBatchedQuantGEMM;
using superroc::hip::LaunchBatchedQuantGEMMPreQuantized;
using superroc::hip::LaunchBatchedQuantGEMMFp32;
using superroc::hip::LaunchFloatToBfloat16;
using superroc::hip::LaunchQuantizeActivationQ8_1FromFp32;
using superroc::hip::MtpPrimeTiledShapeOk;
using superroc::hip::QuantizedActivationBytes;

namespace {

void Expect(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL " << msg << "\n";
    std::exit(1);
  }
}

#define HIP_REQUIRE(expr)                                                      \
  do {                                                                         \
    const hipError_t err_ = (expr);                                            \
    if (err_ != hipSuccess) {                                                  \
      std::cerr << "FAIL hip " << hipGetErrorString(err_) << "\n";             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct DeviceBuf {
  void* p = nullptr;
  explicit DeviceBuf(std::size_t bytes) { HIP_REQUIRE(hipMalloc(&p, bytes)); }
  ~DeviceBuf() {
    const hipError_t e = hipFree(p);
    (void)e;
  }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
  float* f32() const { return static_cast<float*>(p); }
};

constexpr std::uint32_t kBaseSeed = 0x50A1u;
// Guard tail behind every buffer a launch may write.
constexpr std::size_t kTailBytes = 1024u * 1024u;

std::uint32_t Mix(std::uint32_t x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

std::uint32_t Next(std::uint32_t& rng) {
  rng = Mix(rng);
  return rng;
}

float NextValue(std::uint32_t& rng) {
  return static_cast<float>(static_cast<int>(Next(rng) % 2001) - 1000) /
         1000.0F;
}

void Upload(DeviceBuf& dst, const void* src, std::size_t bytes) {
  HIP_REQUIRE(hipMemcpy(dst.p, src, bytes, hipMemcpyHostToDevice));
}

void FetchBytes(std::vector<std::uint8_t>& dst, const DeviceBuf& src) {
  HIP_REQUIRE(hipMemcpy(dst.data(), src.p, dst.size(), hipMemcpyDeviceToHost));
}

void Fetch(std::vector<float>& dst, const DeviceBuf& src) {
  HIP_REQUIRE(hipMemcpy(dst.data(), src.p, dst.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
}

// ggml quantize_row_q8_0_ref: fp16 scale amax/127, round(x * 127/amax).
void EncodeRowQ8_0(const float* x, std::size_t k, std::uint8_t* out) {
  for (std::size_t off = 0; off < k; off += 32) {
    float amax = 0.0F;
    for (std::size_t j = 0; j < 32; ++j) {
      amax = std::max(amax, std::fabs(x[off + j]));
    }
    const float d = amax / 127.0F;
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    superroc::quant::block_q8_0 blk;
    blk.d = __half_as_ushort(__float2half(d));
    for (std::size_t j = 0; j < 32; ++j) {
      blk.qs[j] = static_cast<std::int8_t>(std::lroundf(x[off + j] * id));
    }
    std::memcpy(out + (off / 32) * sizeof(blk), &blk, sizeof(blk));
  }
}

// Fork rocmfp4_decode: magnitude m<=4 ? m : 2m-4, sign in bit 3.
int Rocmfp4CodeValue(std::uint8_t nib) {
  const std::uint32_t mag3 = nib & 7u;
  const int mag = mag3 <= 4u ? static_cast<int>(mag3)
                             : static_cast<int>(2u * mag3 - 4u);
  return (nib & 8u) != 0 ? -mag : mag;
}

// One row of random float weights quantized block by block the way the fork's
// encoder does it: exhaustive UE4M3 half-scale byte search on the block MSE,
// then nearest decoded level per element. Type 100 carries one scale per
// 16-element half (bytes 16 and 17), type 101 one scale for the block
// (byte 16); both share the low/high nibble packing.
void EncodeRowRocmfp4(GgmlType type, std::size_t k, std::uint32_t& rng,
                      std::uint8_t* out) {
  const bool fast = type == GgmlType::kQ4_0_ROCMFP4_FAST;
  const std::size_t block_bytes = fast ? 17u : 18u;
  std::vector<float> target(k);
  for (std::size_t i = 0; i < k; ++i) {
    target[i] = NextValue(rng);
  }
  std::vector<float> expected(k);
  for (std::size_t off = 0; off < k; off += 32) {
    std::uint8_t* blk = out + (off / 32) * block_bytes;
    std::memset(blk, 0, block_bytes);
    for (std::size_t half = 0; half < (fast ? 1u : 2u); ++half) {
      const std::size_t base = off + half * 16;
      float best_err = std::numeric_limits<float>::max();
      std::uint8_t best_scale = 0;
      std::uint8_t best_codes[16] = {};
      for (std::uint32_t e = 0x01u; e <= 0x7eu; ++e) {
        const float scale_f = superroc_fp4_ue4m3_half(static_cast<std::uint8_t>(e));
        float err = 0.0F;
        std::uint8_t codes[16] = {};
        for (std::size_t j = 0; j < 16; ++j) {
          const float x = target[base + j];
          std::uint8_t best_code = 0;
          float best_mag_err = std::numeric_limits<float>::max();
          for (std::uint8_t nib = 0u; nib < 16u; ++nib) {
            const float mag =
                static_cast<float>(Rocmfp4CodeValue(nib)) * scale_f;
            const float d = mag - x;
            const float e2 = d * d;
            if (e2 < best_mag_err) {
              best_mag_err = e2;
              best_code = nib;
            }
          }
          codes[j] = best_code;
          err += best_mag_err;
        }
        if (err < best_err) {
          best_err = err;
          best_scale = static_cast<std::uint8_t>(e);
          std::memcpy(best_codes, codes, sizeof(best_codes));
        }
      }
      const float scale_f = superroc_fp4_ue4m3_half(best_scale);
      for (std::size_t j = 0; j < 16; ++j) {
        const std::size_t elem = half * 16 + j;
        if (elem < 16) {
          blk[elem] |= best_codes[j];
        } else {
          blk[elem - 16] |= static_cast<std::uint8_t>(best_codes[j] << 4);
        }
        expected[base + j] =
            static_cast<float>(Rocmfp4CodeValue(best_codes[j])) * scale_f;
      }
      blk[16 + half] = best_scale;
    }
  }
  // The packed bytes must read back through the shared decoder exactly as
  // packed, otherwise both GEMMs below consume a row nobody intended.
  std::vector<float> decoded(k);
  superroc_fp4_dequant_row(static_cast<int>(type), out, decoded.data(),
                       static_cast<int>(k));
  for (std::size_t i = 0; i < k; ++i) {
    Expect(decoded[i] == expected[i], "fp4 row decodes as packed");
  }
}

std::vector<std::uint8_t> BuildWeights(GgmlType type, std::size_t m,
                                       std::size_t k, std::uint32_t seed) {
  const std::size_t row_bytes = superroc::quant::QuantizedRowBytes(type, k);
  Expect(row_bytes > 0, "weight row bytes");
  std::uint32_t rng = seed;
  std::vector<std::uint8_t> w(row_bytes * m);
  for (std::size_t row = 0; row < m; ++row) {
    std::uint8_t* out = w.data() + row * row_bytes;
    if (type == GgmlType::kQ8_0) {
      std::vector<float> x(k);
      for (std::size_t i = 0; i < k; ++i) {
        x[i] = NextValue(rng);
      }
      EncodeRowQ8_0(x.data(), k, out);
      if (row == 0) {
        // The row must round-trip through the reader the GEMMs share.
        std::vector<float> decoded(k);
        superroc::quant::DequantizeQ8_0(out, decoded.data(), k);
        float err = 0.0F;
        float amax = 0.0F;
        for (std::size_t i = 0; i < k; ++i) {
          err = std::max(err, std::fabs(decoded[i] - x[i]));
          amax = std::max(amax, std::fabs(x[i]));
        }
        Expect(err <= 1.5F * amax / 127.0F, "q8_0 row decodes as packed");
      }
    } else if (type == GgmlType::kQ4_0_ROCMFP4 ||
               type == GgmlType::kQ4_0_ROCMFP4_FAST) {
      EncodeRowRocmfp4(type, k, rng, out);
    } else {
      Expect(false, "unexpected weight type");
    }
  }
  return w;
}

void ExpectTailA5(const std::vector<std::uint8_t>& raw, std::size_t body_bytes,
                  const std::string& what) {
  for (std::size_t i = body_bytes; i < raw.size(); ++i) {
    if (raw[i] != 0xA5) {
      std::cerr << "FAIL " << what << " tail byte " << (i - body_bytes)
                << " = 0x" << std::hex
                << static_cast<int>(raw[i]) << std::dec << "\n";
      std::exit(1);
    }
  }
}

void ExpectNoNaN(const std::vector<std::uint8_t>& raw, std::size_t elems,
                 const std::string& what) {
  const auto* f = reinterpret_cast<const float*>(raw.data());
  for (std::size_t i = 0; i < elems; ++i) {
    if (std::isnan(f[i])) {
      std::cerr << "FAIL " << what << " element " << i << " is NaN\n";
      std::exit(1);
    }
  }
}

// The eight-row reference loop RunPrimeGemm falls back to.
void FillReference(const DeviceBuf& w, const DeviceBuf& x, const DeviceBuf& ref,
                   GgmlType type, std::size_t batch, std::size_t rows_out,
                   std::size_t cols_in) {
  std::size_t done = 0;
  while (done < batch) {
    const std::size_t rows = std::min<std::size_t>(8, batch - done);
    LaunchBatchedQuantGEMMFp32(type, w.p, x.f32() + done * cols_in,
                               ref.f32() + done * rows_out, rows, rows_out,
                               cols_in, nullptr);
    done += rows;
  }
}

double MaxAbsOverRms(const std::vector<std::uint8_t>& out_raw,
                     const std::vector<float>& ref, std::size_t elems,
                     double* rms_out) {
  const auto* out = reinterpret_cast<const float*>(out_raw.data());
  double max_abs = 0.0;
  double sum_sq = 0.0;
  for (std::size_t i = 0; i < elems; ++i) {
    const double diff = std::fabs(static_cast<double>(out[i]) -
                                  static_cast<double>(ref[i]));
    max_abs = std::max(max_abs, diff);
    sum_sq += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  const double rms = std::sqrt(sum_sq / static_cast<double>(elems));
  *rms_out = rms;
  return rms > 0.0 ? max_abs / rms : max_abs;
}

// Every measured case reports before the bound is judged, so one bad case
// still leaves the full per-case table behind it.
std::vector<std::pair<std::string, double>> g_rels;

void RecordRel(const std::string& name, double rel) {
  g_rels.emplace_back(name, rel);
}

void RunTiledCase(const char* label, GgmlType type, std::size_t batch,
                  std::size_t rows_out, std::size_t cols_in,
                  std::uint32_t seed) {
  Expect(MtpPrimeTiledShapeOk(type, batch, rows_out, cols_in),
         std::string(label) + " shape gate");
  const std::vector<std::uint8_t> w = BuildWeights(type, rows_out, cols_in, seed);
  std::vector<float> x(batch * cols_in);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = NextValue(seed);
  }
  const std::size_t q8_bytes = QuantizedActivationBytes(batch, cols_in);
  const std::size_t out_bytes = batch * rows_out * sizeof(float);

  DeviceBuf d_w(w.size());
  Upload(d_w, w.data(), w.size());
  DeviceBuf d_x(x.size() * sizeof(float));
  Upload(d_x, x.data(), x.size() * sizeof(float));
  std::vector<std::uint8_t> q8_host(q8_bytes + kTailBytes, 0xA5);
  DeviceBuf d_q8(q8_host.size());
  Upload(d_q8, q8_host.data(), q8_host.size());
  std::vector<std::uint8_t> out_host(out_bytes + kTailBytes, 0xA5);
  {
    auto* f = reinterpret_cast<float*>(out_host.data());
    for (std::size_t i = 0; i < batch * rows_out; ++i) {
      f[i] = std::numeric_limits<float>::quiet_NaN();
    }
  }
  DeviceBuf d_out(out_host.size());
  Upload(d_out, out_host.data(), out_host.size());

  LaunchQuantizeActivationQ8_1FromFp32(d_x.f32(), d_q8.p, batch, cols_in,
                                       nullptr);
  LaunchBatchedQuantGEMMPreQuantized(type, d_w.p, d_q8.p, d_out.f32(), batch,
                                     rows_out, cols_in, nullptr);
  HIP_REQUIRE(hipStreamSynchronize(nullptr));

  std::vector<std::uint8_t> q8_raw(q8_host.size());
  FetchBytes(q8_raw, d_q8);
  std::vector<std::uint8_t> out_raw(out_host.size());
  FetchBytes(out_raw, d_out);
  ExpectTailA5(q8_raw, q8_bytes, std::string(label) + " q8 scratch");
  ExpectTailA5(out_raw, out_bytes, std::string(label) + " output");
  ExpectNoNaN(out_raw, batch * rows_out, std::string(label) + " output");

  std::vector<std::uint8_t> ref_host(out_bytes, 0xA5);
  {
    auto* f = reinterpret_cast<float*>(ref_host.data());
    for (std::size_t i = 0; i < batch * rows_out; ++i) {
      f[i] = std::numeric_limits<float>::quiet_NaN();
    }
  }
  DeviceBuf d_ref(ref_host.size());
  Upload(d_ref, ref_host.data(), ref_host.size());
  FillReference(d_w, d_x, d_ref, type, batch, rows_out, cols_in);
  HIP_REQUIRE(hipStreamSynchronize(nullptr));
  std::vector<float> ref(batch * rows_out);
  Fetch(ref, d_ref);
  {
    std::vector<std::uint8_t> ref_raw(ref_host.size());
    FetchBytes(ref_raw, d_ref);
    ExpectNoNaN(ref_raw, batch * rows_out, std::string(label) + " reference");
  }
  double rms = 0.0;
  const double rel = MaxAbsOverRms(out_raw, ref, batch * rows_out, &rms);
  std::cout << "prime_route " << label << " batch=" << batch
            << " m=" << rows_out << " k=" << cols_in << std::scientific
            << " max_abs=" << rel * rms << " rms=" << rms << " rel=" << rel
            << "\n";
  RecordRel(std::string(label) + " batch=" + std::to_string(batch) + " m=" +
                std::to_string(rows_out) + " k=" + std::to_string(cols_in),
            rel);
}

// Every measured case reports before the bound is judged, so one bad case
// still leaves the full per-case table behind it.
void RunBf16EntryCase(std::uint32_t seed) {
  constexpr std::size_t kBatch = 4096;
  constexpr std::size_t kRows = 1024;
  constexpr std::size_t kCols = 2048;
  const std::vector<std::uint8_t> w =
      BuildWeights(GgmlType::kQ8_0, kRows, kCols, seed);
  std::vector<float> x(kBatch * kCols);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = NextValue(seed);
  }
  const std::size_t bf16_bytes = kBatch * kCols * sizeof(std::uint16_t);
  const std::size_t out_bytes = kBatch * kRows * sizeof(float);

  DeviceBuf d_w(w.size());
  Upload(d_w, w.data(), w.size());
  DeviceBuf d_x(x.size() * sizeof(float));
  Upload(d_x, x.data(), x.size() * sizeof(float));
  std::vector<std::uint8_t> bf16_host(bf16_bytes + kTailBytes, 0xA5);
  DeviceBuf d_bf16(bf16_host.size());
  Upload(d_bf16, bf16_host.data(), bf16_host.size());
  std::vector<std::uint8_t> out_host(out_bytes + kTailBytes, 0xA5);
  {
    auto* f = reinterpret_cast<float*>(out_host.data());
    for (std::size_t i = 0; i < kBatch * kRows; ++i) {
      f[i] = std::numeric_limits<float>::quiet_NaN();
    }
  }
  DeviceBuf d_out(out_host.size());
  Upload(d_out, out_host.data(), out_host.size());

  LaunchFloatToBfloat16(d_x.f32(), d_bf16.p, kBatch * kCols, nullptr);
  LaunchBatchedQuantGEMM(GgmlType::kQ8_0, d_w.p, d_bf16.p, d_out.f32(),
                         kBatch, kRows, kCols, nullptr);
  HIP_REQUIRE(hipStreamSynchronize(nullptr));

  std::vector<std::uint8_t> bf16_raw(bf16_host.size());
  FetchBytes(bf16_raw, d_bf16);
  std::vector<std::uint8_t> out_raw(out_host.size());
  FetchBytes(out_raw, d_out);
  ExpectTailA5(bf16_raw, bf16_bytes, "bf16 entry input");
  ExpectTailA5(out_raw, out_bytes, "bf16 entry output");
  ExpectNoNaN(out_raw, kBatch * kRows, "bf16 entry output");

  std::vector<std::uint8_t> ref_host(out_bytes, 0xA5);
  {
    auto* f = reinterpret_cast<float*>(ref_host.data());
    for (std::size_t i = 0; i < kBatch * kRows; ++i) {
      f[i] = std::numeric_limits<float>::quiet_NaN();
    }
  }
  DeviceBuf d_ref(ref_host.size());
  Upload(d_ref, ref_host.data(), ref_host.size());
  FillReference(d_w, d_x, d_ref, GgmlType::kQ8_0, kBatch, kRows, kCols);
  HIP_REQUIRE(hipStreamSynchronize(nullptr));
  std::vector<float> ref(kBatch * kRows);
  Fetch(ref, d_ref);
  {
    std::vector<std::uint8_t> ref_raw(ref_host.size());
    FetchBytes(ref_raw, d_ref);
    ExpectNoNaN(ref_raw, kBatch * kRows, "bf16 entry reference");
  }
  double rms = 0.0;
  const double rel = MaxAbsOverRms(out_raw, ref, kBatch * kRows, &rms);
  std::cout << "prime_route bf16_entry batch=" << kBatch << " m=" << kRows
            << " k=" << kCols << std::scientific << " max_abs=" << rel * rms
            << " rms=" << rms << " rel=" << rel << "\n";
  RecordRel("bf16_entry batch=4096 m=1024 k=2048", rel);
}

void ExpectShapeRejected(const char* label, GgmlType type,
                         std::size_t batch, std::size_t rows_out,
                         std::size_t cols_in) {
  Expect(!MtpPrimeTiledShapeOk(type, batch, rows_out, cols_in),
         std::string(label) + " must be rejected");
  std::cout << "prime_route shape " << label << " rejected OK\n";
}

}  // namespace

int main() {
  // The shape rule rejects the off-rule shapes first, cheaply.
  ExpectShapeRejected("q8_0 batch=16", GgmlType::kQ8_0, 16, 1024, 2048);
  ExpectShapeRejected("fp4 batch=95", GgmlType::kQ4_0_ROCMFP4, 95, 1024,
                      2048);
  ExpectShapeRejected("fp4 k=2080", GgmlType::kQ4_0_ROCMFP4, 96, 1024,
                      2080);
  ExpectShapeRejected("q4_0", GgmlType::kQ4_0, 4096, 1024, 2048);
  ExpectShapeRejected("bf16", GgmlType::kBF16, 4096, 1024, 2048);
  ExpectShapeRejected("q8_k", GgmlType::kQ8_K, 4096, 1024, 2048);

  RunTiledCase("q8_0", GgmlType::kQ8_0, 17, 1024, 2048, kBaseSeed + 1u);
  RunTiledCase("q8_0", GgmlType::kQ8_0, 95, 1024, 2048, kBaseSeed + 2u);
  RunTiledCase("q8_0", GgmlType::kQ8_0, 96, 1024, 2048, kBaseSeed + 3u);
  RunTiledCase("q8_0", GgmlType::kQ8_0, 1501, 1024, 2048, kBaseSeed + 4u);
  RunTiledCase("q8_0", GgmlType::kQ8_0, 4096, 1024, 2048, kBaseSeed + 5u);
  RunTiledCase("q8_0", GgmlType::kQ8_0, 4096, 2048, 4096, kBaseSeed + 6u);
  RunTiledCase("q8_0", GgmlType::kQ8_0, 4096, 5120, 10240,
               kBaseSeed + 7u);
  RunTiledCase("fp4", GgmlType::kQ4_0_ROCMFP4, 96, 1024, 2048,
               kBaseSeed + 8u);
  RunTiledCase("fp4", GgmlType::kQ4_0_ROCMFP4, 1501, 1024, 2048,
               kBaseSeed + 9u);
  RunTiledCase("fp4", GgmlType::kQ4_0_ROCMFP4, 4096, 2048, 4096,
               kBaseSeed + 10u);
  RunTiledCase("fp4_fast", GgmlType::kQ4_0_ROCMFP4_FAST, 96, 1024, 2048,
               kBaseSeed + 11u);
  RunTiledCase("fp4_fast", GgmlType::kQ4_0_ROCMFP4_FAST, 1501, 1024,
               2048, kBaseSeed + 12u);
  RunTiledCase("fp4_fast", GgmlType::kQ4_0_ROCMFP4_FAST, 4096, 2048, 4096,
               kBaseSeed + 13u);
  RunBf16EntryCase(kBaseSeed + 14u);

  for (const auto& [name, rel] : g_rels) {
    Expect(rel <= 0.03, name + " relative difference " +
                            std::to_string(rel));
  }

  std::cout << "MTP_PRIME_ROUTE PASS\n";
  return 0;
}
