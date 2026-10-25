// LM1 SPEC11 item 2: bit-exact KV equality for the KAT-MTP batched prime pack.
//
//   LaunchKatMtpWriteKv packs a BATCH of already-normed, already-RoPE'd K/V
//   rows into the draft head's dense KV cache for the embedded nextn block
//   (layer 0). It is the batched-prime replacement for the tile pack, which
//   cannot run on KAT (2 KV heads; the tile kernel needs 4). Its whole reason
//   to exist is that the MTP head's plain LaunchAttention must read exactly the
//   bytes WriteKVCacheKernel would have written per position — otherwise the
//   draft head attends over a corrupt cache and acceptance silently degrades,
//   or (worse) a wrong offset writes outside the buffer.
//
//   This test drives BOTH kernels on the GPU over identical inputs and compares
//   the WHOLE cache buffers byte-for-byte, for both dense layouts:
//     * fp32  k_cache / v_cache : ((layer_idx * kv_heads + kv_h) * max_context
//                                  + pos) * head_dim + tid
//     * fp16  k_cache_f16 / v_cache_f16 : (layer_idx * max_context + pos)
//                                  * kv_width + kv_h * head_dim + tid
//   at layer_idx 0, over several start positions including the 512-token prime
//   chunk boundary (509+4, 511+1, 512+4, 1023+3) and the context tail.
//
//   Both caches are pre-filled with a sentinel and compared in full, so a write
//   outside the intended (kv_h, pos) slots fails too — that is the class of
//   out-of-range access that faulted the GPU in REPORT9.
//
//   Runs on a live ROCm device; deliberately NOT part of the default CTest
//   suite on a CPU box (same policy as rocmfp4_gpu_gemv_test).
#include "src/models/qwen/hip/mtp/kat_moe_ffn.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace gufo::hip {

// Defined at namespace scope in
// src/models/qwen/hip/kernels/attention_decode.hip, which is compiled into
// gufo_core. Declared here so the test can drive the reference kernel at
// layer_idx 0 with the same arguments the production decode path passes.
__global__ void WriteKVCacheKernel(
    const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    std::uint32_t layer_idx, std::uint32_t pos, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim);

}  // namespace gufo::hip

namespace {

// KAT-Coder V2.5 attention shape (16 heads / 2 KV heads / head_dim 256). The
// pack is exercised with the real KV head count and head dim; max_context is
// kept small so the sentinel comparison stays cheap.
constexpr std::uint32_t kMaxContext = 4096;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kHeadDim = 256;
constexpr unsigned char kSentinel = 0x5A;

void Fail(const std::string& message) {
  std::cerr << "FAIL " << message << "\n";
  std::exit(1);
}

#define HIP_REQUIRE(expr)                                                      \
  do {                                                                         \
    const hipError_t err_ = (expr);                                            \
    if (err_ != hipSuccess) {                                                  \
      Fail(std::string("hip ") + hipGetErrorString(err_) + " at " #expr);      \
    }                                                                          \
  } while (0)

class DeviceBuf {
public:
  explicit DeviceBuf(std::size_t bytes) : bytes_(bytes) {
    if (bytes == 0)
      return;
    HIP_REQUIRE(hipMalloc(&p_, bytes));
  }
  ~DeviceBuf() {
    if (p_ != nullptr) {
      const hipError_t e = hipFree(p_);
      (void)e;
    }
  }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;

  void* raw() const { return p_; }
  float* f32() const { return static_cast<float*>(p_); }
  __half* f16() const { return static_cast<__half*>(p_); }
  std::size_t bytes() const { return bytes_; }

  // Device-side sentinel fill, so an unwritten slot is distinguishable.
  void FillSentinel() {
    if (p_ != nullptr)
      HIP_REQUIRE(hipMemset(p_, kSentinel, bytes_));
  }

  void ReadInto(std::vector<unsigned char>* out) const {
    out->resize(bytes_);
    if (bytes_ > 0)
      HIP_REQUIRE(hipMemcpy(out->data(), p_, bytes_, hipMemcpyDeviceToHost));
  }

private:
  void* p_{nullptr};
  std::size_t bytes_{0};
};

struct Case {
  std::uint32_t start_pos;
  std::uint32_t batch;
  const char* note;
};

// First index of a byte-level mismatch over the sentinel, or -1 when identical.
long FirstMismatch(const std::vector<unsigned char>& a,
                   const std::vector<unsigned char>& b) {
  if (a.size() != b.size())
    return 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i])
      return static_cast<long>(i);
  }
  return -1;
}

void Compare(const char* label, const DeviceBuf& produced,
             const DeviceBuf& reference, std::size_t element_bytes,
             std::size_t element_count) {
  std::vector<unsigned char> got;
  std::vector<unsigned char> want;
  produced.ReadInto(&got);
  reference.ReadInto(&want);
  const long at = FirstMismatch(got, want);
  if (at < 0)
    return;
  const std::size_t element = static_cast<std::size_t>(at) / element_bytes;
  std::fprintf(stderr,
               "  %s: first mismatch at byte %ld (element %zu of %zu)\n",
               label, at, element, element_count);
  Fail(std::string(label) + " is not byte-identical to WriteKVCacheKernel");
}

// Runs one (start_pos, batch) case. `fp32_caches` selects whether the fp32
// dense caches are present; the fp16 caches are always exercised.
void RunCase(const Case& c, bool fp32_caches) {
  const std::size_t rows =
      static_cast<std::size_t>(c.batch) * kKvHeads * kHeadDim;
  const std::size_t dense = static_cast<std::size_t>(kKvHeads) * kMaxContext *
                            kHeadDim;
  const std::size_t f16_elems =
      static_cast<std::size_t>(kMaxContext) * kKvHeads * kHeadDim;

  std::vector<float> h_k(rows);
  std::vector<float> h_v(rows);
  std::uint32_t seed = 0x1234567u + c.start_pos * 131u + c.batch;
  const auto next = [&seed]() {
    seed = seed * 1664525u + 1013904223u;
    return static_cast<float>((seed >> 8) & 0x00FFFFFFu) / 16777216.0F - 0.5F;
  };
  for (float& value : h_k)
    value = next();
  for (float& value : h_v)
    value = next();

  DeviceBuf d_k(rows * sizeof(float));
  DeviceBuf d_v(rows * sizeof(float));
  DeviceBuf mtp_kc(dense * sizeof(float));
  DeviceBuf mtp_vc(dense * sizeof(float));
  DeviceBuf ref_kc(dense * sizeof(float));
  DeviceBuf ref_vc(dense * sizeof(float));
  DeviceBuf mtp_kf(f16_elems * sizeof(__half));
  DeviceBuf mtp_vf(f16_elems * sizeof(__half));
  DeviceBuf ref_kf(f16_elems * sizeof(__half));
  DeviceBuf ref_vf(f16_elems * sizeof(__half));

  HIP_REQUIRE(hipMemcpy(d_k.raw(), h_k.data(), rows * sizeof(float),
                        hipMemcpyHostToDevice));
  HIP_REQUIRE(hipMemcpy(d_v.raw(), h_v.data(), rows * sizeof(float),
                        hipMemcpyHostToDevice));
  for (DeviceBuf* buffer : {&mtp_kc, &mtp_vc, &ref_kc, &ref_vc, &mtp_kf,
                            &mtp_vf, &ref_kf, &ref_vf}) {
    buffer->FillSentinel();
  }

  hipStream_t stream = nullptr;
  HIP_REQUIRE(hipStreamCreate(&stream));

  gufo::hip::LaunchKatMtpWriteKv(
      d_k.f32(), d_v.f32(), fp32_caches ? mtp_kc.f32() : nullptr,
      fp32_caches ? mtp_vc.f32() : nullptr, mtp_kf.raw(), mtp_vf.raw(),
      c.start_pos, static_cast<std::size_t>(c.batch), kMaxContext, kKvHeads,
      kHeadDim, stream);

  // The production decode path writes one position per call through this
  // kernel; drive it once per token over the same K/V rows.
  const std::size_t row_stride =
      static_cast<std::size_t>(kKvHeads) * kHeadDim;
  for (std::uint32_t token = 0; token < c.batch; ++token) {
    gufo::hip::WriteKVCacheKernel<<<kKvHeads, 256, 0, stream>>>(
        d_k.f32() + static_cast<std::size_t>(token) * row_stride,
        d_v.f32() + static_cast<std::size_t>(token) * row_stride,
        fp32_caches ? ref_kc.f32() : nullptr,
        fp32_caches ? ref_vc.f32() : nullptr, ref_kf.f16(), ref_vf.f16(),
        /*layer_idx=*/0, c.start_pos + token, kMaxContext, kKvHeads, kHeadDim);
  }

  HIP_REQUIRE(hipStreamSynchronize(stream));
  HIP_REQUIRE(hipStreamDestroy(stream));

  Compare("k_cache_f16", mtp_kf, ref_kf, sizeof(__half), f16_elems);
  Compare("v_cache_f16", mtp_vf, ref_vf, sizeof(__half), f16_elems);
  if (fp32_caches) {
    Compare("k_cache", mtp_kc, ref_kc, sizeof(float), dense);
    Compare("v_cache", mtp_vc, ref_vc, sizeof(float), dense);
  }
}

}  // namespace

int main() {
  int device_count = 0;
  HIP_REQUIRE(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    Fail("no ROCm device");
  }
  HIP_REQUIRE(hipSetDevice(0));

  hipDeviceProp_t props{};
  HIP_REQUIRE(hipGetDeviceProperties(&props, 0));
  std::cout << "device " << props.name << " gfx" << props.gcnArchName
            << std::endl;

  const Case cases[] = {
      {0, 1, "single token at 0"},
      {0, 4, "first prime chunk"},
      {511, 1, "last token of chunk 0"},
      {509, 4, "crosses the 512 chunk boundary"},
      {512, 4, "first chunk of the second slab"},
      {1023, 3, "crosses the 1024 boundary"},
      {4092, 4, "context tail"},
  };

  for (const bool fp32_caches : {true, false}) {
    for (const Case& c : cases) {
      std::cout << "case start_pos=" << c.start_pos << " batch=" << c.batch
                << " fp32_caches=" << (fp32_caches ? 1 : 0) << " (" << c.note
                << ")" << std::endl;
      RunCase(c, fp32_caches);
    }
  }

  std::cout << "OK kat_mtp_writekv: LaunchKatMtpWriteKv == WriteKVCacheKernel "
               "bit-exact on "
            << (sizeof(cases) / sizeof(cases[0])) * 2
            << " case/representation combinations" << std::endl;
  return 0;
}
