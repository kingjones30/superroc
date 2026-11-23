#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = superroc::models::qwen38_flash_next::rocm;
namespace {

// The model's full-attention geometry as Executor::Attention passes it
// (config.cpp pins these for the package): 24 query heads over 2 KV heads,
// head_dim 256, rotary 64 of the 256 lanes, theta 1e7, rms eps 1e-6.
constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kDim = 256;
constexpr std::uint32_t kRotary = 64;
constexpr float kTheta = 1e7F;
constexpr float kEps = 1e-6F;
constexpr std::uint32_t kQWidth = kHeads * kDim;     // AttentionQDim()
constexpr std::uint32_t kKvWidth = kKvHeads * kDim;  // AttentionKvDim()
constexpr std::uint32_t kQgStride = 2 * kQWidth;     // 2 * AttentionQDim()
// Every half of an untouched cache row reads 0x5A5A (about 2.9e15), far from
// anything the preparation writes (inputs within [-4, 4], gamma near 1), so a
// write outside the token range cannot hide behind a plausible value.
constexpr std::uint8_t kSentinelByte = 0x5A;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t count) : count_(count) {
    void* allocation = nullptr;
    CheckHip(hipMalloc(&allocation, bytes()), "hipMalloc");
    CheckHip(hipMemset(allocation, 0, bytes()), "hipMemset");
    data_ = static_cast<T*>(allocation);
  }
  ~HipBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  HipBuffer(HipBuffer&&) = delete;
  HipBuffer& operator=(HipBuffer&&) = delete;

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] const T* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

// Deterministic xorshift values in [-scale, scale].
std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
                              float scale) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = scale *
            static_cast<float>(static_cast<int>(NextRandom(&seed) & 0xFFFFU) -
                               32768) /
            32768.0F;
  }
  return values;
}

template<typename T>
void Upload(HipBuffer<T>* destination, const std::vector<T>& source) {
  if (source.size() * sizeof(T) != destination->bytes()) {
    throw std::runtime_error("upload size mismatch");
  }
  CheckHip(hipMemcpy(destination->get(), source.data(), destination->bytes(),
                     hipMemcpyHostToDevice),
           "upload");
}

template<typename T>
std::vector<T> Download(const HipBuffer<T>& source) {
  std::vector<T> values(source.bytes() / sizeof(T));
  CheckHip(hipMemcpy(values.data(), source.get(), source.bytes(),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

template<typename T>
std::uint64_t Bits(const T& value) noexcept {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(T));
  return bits;
}

// memcmp over the whole buffer; on the first differing element print its
// index and both bit patterns and fail the run.
template<typename T>
void ExpectSame(const std::vector<T>& expected, const std::vector<T>& actual,
                const char* name, std::uint32_t n, std::uint32_t start) {
  if (expected.size() != actual.size()) {
    throw std::runtime_error(std::string(name) + " length mismatch");
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::memcmp(&expected[i], &actual[i], sizeof(T)) == 0) {
      continue;
    }
    std::cerr << "mismatch n=" << n << " start=" << start << " " << name
              << " index=" << i << " reference=0x" << std::hex
              << Bits(expected[i]) << " fused=0x" << Bits(actual[i]) << std::dec
              << '\n';
    throw std::runtime_error(std::string("split preparation differs in ") +
                             name);
  }
}

// Every half of every row outside [start, start + n) must still be the
// sentinel pattern: neither route may touch a cache position it does not own.
void ExpectSentinel(const std::vector<__half>& cache, const char* name,
                    std::uint32_t n, std::uint32_t start) {
  const std::size_t rows = cache.size() / kKvWidth;
  for (std::size_t row = 0; row < rows; ++row) {
    if (row >= start && row < static_cast<std::size_t>(start) + n) {
      continue;
    }
    for (std::uint32_t i = 0; i < kKvWidth; ++i) {
      const __half value = cache[row * kKvWidth + i];
      std::uint16_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      if (bits != static_cast<std::uint16_t>(kSentinelByte * 0x0101U)) {
        std::cerr << "cache " << name << " n=" << n << " start=" << start
                  << " row=" << row << " column=" << i << " was modified: 0x"
                  << std::hex << bits << std::dec << '\n';
        throw std::runtime_error(std::string("split preparation wrote ") +
                                 name + " outside the token range");
      }
    }
  }
}

// One (n_tokens, start_pos) case: launch the fused split preparation, then
// the separate sequence Executor::Attention runs when the
// SUPERROC_FN_PREP_ATTN_SPLIT switch is off, and require byte-identical
// outputs.
void RunCase(std::uint32_t n, std::uint32_t start, std::uint32_t seed) {
  const std::size_t q_count = static_cast<std::size_t>(n) * kQWidth;
  const std::size_t kv_count = static_cast<std::size_t>(n) * kKvWidth;
  const std::size_t cache_rows = start + n + 2;
  const std::size_t cache_count = cache_rows * kKvWidth;

  // Projections in [-4, 4]: the interleaved [q|gate] rows and the separate
  // K and V rows, plus per-head gammas around 1.
  std::vector<float> qg =
      MakeValues(static_cast<std::size_t>(n) * kQgStride, seed, 4.0F);
  std::vector<float> k_values = MakeValues(kv_count, seed + 1, 4.0F);
  std::vector<float> v_values = MakeValues(kv_count, seed + 2, 4.0F);
  std::vector<float> q_gamma = MakeValues(kDim, seed + 3, 0.25F);
  std::vector<float> k_gamma = MakeValues(kDim, seed + 4, 0.25F);
  for (float& g : q_gamma)
    g += 1.0F;
  for (float& g : k_gamma)
    g += 1.0F;

  HipBuffer<float> qg_dev(qg.size());
  // One pristine K/V copy per route: the executor norm- and rope-s the K
  // row in place, so the fused launch must not observe the reference's
  // overwritten buffer.
  HipBuffer<float> k_fused(kv_count), v_fused(kv_count);
  HipBuffer<float> k_ref(kv_count), v_ref(kv_count);
  HipBuffer<float> q_gamma_dev(kDim), k_gamma_dev(kDim);
  HipBuffer<float> q_fused(q_count), gate_fused(q_count);
  HipBuffer<float> q_ref(q_count), gate_ref(q_count);
  HipBuffer<__half> k_cache_fused(cache_count), v_cache_fused(cache_count);
  HipBuffer<__half> k_cache_ref(cache_count), v_cache_ref(cache_count);
  HipBuffer<std::uint32_t> pos(1);

  Upload(&qg_dev, qg);
  Upload(&k_fused, k_values);
  Upload(&v_fused, v_values);
  Upload(&k_ref, k_values);
  Upload(&v_ref, v_values);
  Upload(&q_gamma_dev, q_gamma);
  Upload(&k_gamma_dev, k_gamma);
  Upload(&pos, std::vector<std::uint32_t>{start});
  for (HipBuffer<__half>* cache :
       {&k_cache_fused, &v_cache_fused, &k_cache_ref, &v_cache_ref}) {
    std::vector<__half> sentinel(cache_count);
    std::memset(sentinel.data(), kSentinelByte,
                sentinel.size() * sizeof(__half));
    Upload(cache, sentinel);
  }

  // Fused route: kernels.hpp:478 PrepareAttentionSplit, one launch
  // (PrepareAttentionKernel<1> with k_in/v_in).
  if (!q::PrepareAttentionSplit(
          qg_dev.get(), kQgStride, k_fused.get(), v_fused.get(),
          q_gamma_dev.get(), k_gamma_dev.get(), q_fused.get(), gate_fused.get(),
          k_cache_fused.get(), v_cache_fused.get(), n, kHeads, kKvHeads, kDim,
          kRotary, pos.get(), kTheta, kEps, nullptr)) {
    throw std::runtime_error("PrepareAttentionSplit rejected the geometry");
  }
  CheckHip(hipDeviceSynchronize(), "fused preparation");

  // Reference route, the exact calls and arguments of Executor::Attention
  // with the switch off (src/models/qwen38_flash_next/kernels/rocm/
  // executor.cpp):
  //   3954-3955  the K and V projections land in their own buffers; here
  //              the test's k_ref/v_ref stand in for the Dense(attn_k)/
  //              Dense(attn_v) writes into s_.k/s_.v;
  //   3968-3969  UnpackQGate(s_.qg, 2 * AttentionQDim(), s_.q, s_.attn_gate,
  //              nullptr, nullptr, n, heads, head_dim, 0): k and v stay
  //              null because the split projections wrote them already;
  //   3973-3976  RmsNormRows of q with attn_q_norm and of k with attn_k_norm,
  //              in place, rows n * heads / n * kv_heads, dim 256, eps;
  //   3977-3980  Rope of q (heads) and k (kv_heads) at the device position;
  //   3981-3982  StoreKv of k and v into the caches, row dim AttentionKvDim().
  q::UnpackQGate(qg_dev.get(), kQgStride, q_ref.get(), gate_ref.get(), nullptr,
                 nullptr, n, kHeads, kDim, 0, nullptr);
  q::RmsNormRows(q_ref.get(), q_gamma_dev.get(), q_ref.get(), n * kHeads, kDim,
                 1, kEps, nullptr);
  q::RmsNormRows(k_ref.get(), k_gamma_dev.get(), k_ref.get(), n * kKvHeads,
                 kDim, 1, kEps, nullptr);
  q::Rope(q_ref.get(), n, kHeads, kDim, kRotary, pos.get(), kTheta, nullptr);
  q::Rope(k_ref.get(), n, kKvHeads, kDim, kRotary, pos.get(), kTheta, nullptr);
  q::StoreKv(k_ref.get(), k_cache_ref.get(), n, kKvWidth, pos.get(), nullptr);
  q::StoreKv(v_ref.get(), v_cache_ref.get(), n, kKvWidth, pos.get(), nullptr);
  CheckHip(hipDeviceSynchronize(), "reference preparation");

  ExpectSame(Download(q_ref), Download(q_fused), "queries", n, start);
  ExpectSame(Download(gate_ref), Download(gate_fused), "gates", n, start);
  ExpectSame(Download(k_cache_ref), Download(k_cache_fused), "key cache", n,
             start);
  ExpectSame(Download(v_cache_ref), Download(v_cache_fused), "value cache", n,
             start);
  ExpectSentinel(Download(k_cache_fused), "fused key cache", n, start);
  ExpectSentinel(Download(v_cache_fused), "fused value cache", n, start);
  ExpectSentinel(Download(k_cache_ref), "reference key cache", n, start);
  ExpectSentinel(Download(v_cache_ref), "reference value cache", n, start);
  std::cout << "split attention preparation n=" << n << " start=" << start
            << ": q, gate, key cache and value cache byte-identical\n";
}

}  // namespace

int main() {
  try {
    const std::pair<std::uint32_t, std::uint32_t> cases[] = {
        {1, 0},    {2, 0},    {3, 0},    {7, 0},    {8, 0},    {64, 0},
        {1, 1000}, {2, 1000}, {3, 1000}, {7, 1000}, {8, 1000}, {64, 1000}};
    std::uint32_t seed = 0x5EED0B5CU;
    for (const auto& [n, start] : cases) {
      RunCase(n, start, seed);
      seed += 16;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
