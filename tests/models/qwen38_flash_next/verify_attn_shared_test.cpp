#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = superroc::models::qwen38_flash_next::rocm;
namespace {

// The model's full-attention geometry: 24 query heads over 2 KV heads,
// head_dim 256. The shared-row kernel runs one query head per block and one
// block per split, so this geometry is also the fused kernel's dispatch case.
constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kDim = 256;
constexpr std::uint32_t kRatio = 4;
constexpr std::uint32_t kQWidth = kHeads * kDim;
constexpr std::uint32_t kKvWidth = kKvHeads * kDim;

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
  CheckHip(hipMemcpy(destination->get(), source.data(), destination->bytes(),
                     hipMemcpyHostToDevice),
           "upload");
}

std::vector<float> Download(HipBuffer<float>* source, std::size_t count) {
  std::vector<float> values(count);
  CheckHip(hipMemcpy(values.data(), source->get(), source->bytes(),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

/// Runs the per-row kernel and the shared-row kernel over the same cache and
/// requires their outputs to be bit for bit equal, in the single-block form
/// and split eight ways (the decode geometry). `start_pos` is the anchor
/// row's position; the batch holds `rows` consecutive positions, so the last
/// rows see keys the first ones do not.
void CompareRowKernels(std::uint32_t start_pos, std::uint32_t rows,
                       std::uint32_t seed) {
  const std::uint32_t n_kv = start_pos + rows;
  const std::size_t q_count = static_cast<std::size_t>(rows) * kQWidth;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv) * kKvWidth;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  std::vector<__half> kh(kv_count);
  std::vector<__half> vh(kv_count);
  {
    const auto kf = MakeValues(kv_count, seed ^ 0xAAAAU, 1.0F);
    const auto vf = MakeValues(kv_count, seed ^ 0x3333U, 1.0F);
    for (std::size_t i = 0; i < kv_count; ++i) {
      kh[i] = __float2half(kf[i]);
      vh[i] = __float2half(vf[i]);
    }
  }

  HipBuffer<float> d_q(q_count);
  HipBuffer<__half> d_k(kv_count);
  HipBuffer<__half> d_v(kv_count);
  HipBuffer<std::uint32_t> d_pos(1);
  Upload(&d_q, qv);
  Upload(&d_k, kh);
  Upload(&d_v, vh);
  Upload(&d_pos, std::vector<std::uint32_t>{start_pos});

  std::cout << "shared-row attention rows=" << rows << " start=" << start_pos
            << " keys=" << n_kv;
  for (const std::uint32_t splits : {0U, 8U}) {
    const std::uint32_t z = std::max(splits, 1U);
    HipBuffer<float> d_partials(static_cast<std::size_t>(rows) * kHeads * z *
                                (kDim + 2));
    HipBuffer<float> d_per_row(q_count);
    HipBuffer<float> d_shared(q_count);
    float* partials = splits == 0 ? nullptr : d_partials.get();
    // The dense sweep: no selection mask. The shared kernel is opt-in, so the
    // same call with the flag off must still take the per-row kernel.
    q::Attention(d_q.get(), d_k.get(), d_v.get(), nullptr, 0, d_per_row.get(),
                 partials, splits, rows, d_pos.get(), kHeads, kKvHeads, kDim,
                 kRatio, nullptr);
    q::Attention(d_q.get(), d_k.get(), d_v.get(), nullptr, 0, d_shared.get(),
                 partials, splits, rows, d_pos.get(), kHeads, kKvHeads, kDim,
                 kRatio, nullptr, nullptr, nullptr, 0, true);
    CheckHip(hipDeviceSynchronize(), "attention synchronization");
    const auto per_row = Download(&d_per_row, q_count);
    const auto shared = Download(&d_shared, q_count);
    for (std::size_t i = 0; i < q_count; ++i) {
      if (!std::isfinite(shared[i])) {
        throw std::runtime_error("shared-row attention output is not finite");
      }
    }
    if (std::memcmp(per_row.data(), shared.data(), q_count * sizeof(float)) !=
        0) {
      std::size_t first = q_count;
      for (std::size_t i = 0; i < q_count; ++i) {
        if (std::memcmp(&per_row[i], &shared[i], sizeof(float)) != 0) {
          first = i;
          break;
        }
      }
      std::cerr << " rows=" << rows << " start=" << start_pos
                << " splits=" << splits << " first differing index=" << first
                << " row=" << first / kQWidth
                << " head=" << (first % kQWidth) / kDim
                << " dim=" << first % kDim << " per-row=" << std::hexfloat
                << per_row[first] << " shared=" << shared[first]
                << std::defaultfloat << '\n';
      throw std::runtime_error("shared-row attention is not bitwise equal");
    }
    std::cout << (splits == 0 ? " single" : " split8") << " bitwise-equal";
  }
  std::cout << '\n';
}

}  // namespace

int main() {
  try {
    // 1K, 3K and 32K positions, every verify width the engine launches.
    for (const std::uint32_t depth : {1000U, 3000U, 33000U}) {
      for (std::uint32_t rows = 2; rows <= 6; ++rows) {
        CompareRowKernels(depth, rows, depth * 17U + rows * 101U);
      }
    }
    // A batch that straddles a 256-key tile boundary: the earlier rows stop
    // at the previous tile, which the shared kernel still walks as an exact
    // no-op for them.
    for (const std::uint32_t depth : {254U, 509U, 767U}) {
      for (std::uint32_t rows = 2; rows <= 6; ++rows) {
        CompareRowKernels(depth, rows, depth * 7U + rows * 13U);
      }
    }
    // The widest instantiations of the kernel.
    CompareRowKernels(1000, 7, 7001);
    CompareRowKernels(1000, 8, 8001);
    std::cout << "verify shared-row attention: all comparisons bitwise-equal\n";
  } catch (const std::exception& error) {
    std::cerr << "verify shared-row attention: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
