// The 4-bit FFN prefill kernels against a host reference: host-packed weights,
// packed activations, and gate + up with SwiGLU, on sizes with partial tiles
// and with columns whose scale is far above the rest.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/models/qwen/hip/detail/int4_plan.hpp"
#include "src/models/qwen/hip/ops/prefill_int4.hpp"

namespace {
using superroc::core::GgmlType;
using superroc::hip::kInt4ColumnMask;
using superroc::hip::kInt4FineInput;
using superroc::hip::kInt4FinePass;
using superroc::hip::kInt4NoColumn;
using superroc::hip::detail::Int4Plan;
using superroc::hip::detail::MakeInt4Plan;

int g_failed = 0;
#define CHECK(cond, ...)                 \
  do {                                   \
    if (!(cond)) {                       \
      std::printf("FAIL: " __VA_ARGS__); \
      std::printf("\n");                 \
      ++g_failed;                        \
    }                                    \
  } while (0)
#define HIP(call)                                  \
  do {                                             \
    if ((call) != hipSuccess) {                    \
      std::printf("hip call failed: %s\n", #call); \
      std::exit(1);                                \
    }                                              \
  } while (0)

float Ue4m3(std::uint32_t e) {
  if (e > 0x7EU)
    return 0.F;
  const std::uint32_t exp = (e >> 3U) & 15U, man = e & 7U;
  if (exp == 0U)
    return static_cast<float>(man) / 1024.F;
  return std::ldexp(1.F + static_cast<float>(man) / 8.F,
                    static_cast<int>(exp) - 8);
}
float Code(std::uint32_t nibble) {
  const std::uint32_t m = nibble & 7U;
  const float v =
      m <= 4U ? static_cast<float>(m) : static_cast<float>((m - 2U) * 2U);
  return (nibble & 8U) != 0U ? -v : v;
}
float Fp4Value(const std::uint8_t* row, std::uint32_t column,
               std::uint32_t stride) {
  const std::uint32_t j = column % 32U;
  const std::uint8_t* block = row + std::size_t{column / 32U} * stride;
  const std::uint32_t byte = block[j % 16U];
  const std::uint32_t nibble = j < 16U ? byte & 15U : byte >> 4U;
  return Code(nibble) * Ue4m3(block[(stride == 17U || j < 16U) ? 16U : 17U]);
}
int Nibble(std::uint64_t packed, int j) {
  const int v = static_cast<int>((packed >> (4 * j)) & 15U);
  return v >= 8 ? v - 16 : v;
}
template<typename T>
T* Upload(const std::vector<T>& v) {
  T* d = nullptr;
  HIP(hipMalloc(&d, v.size() * sizeof(T)));
  HIP(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
  return d;
}
template<typename T>
std::vector<T> Download(const T* d, std::size_t n) {
  std::vector<T> v(n);
  HIP(hipMemcpy(v.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
  return v;
}

std::vector<std::uint8_t> RandomFp4(std::mt19937& rng, std::size_t m,
                                    std::size_t k, std::uint32_t stride) {
  std::vector<std::uint8_t> w(m * (k / 32) * stride);
  std::uniform_int_distribution<int> byte(0, 255), scale(0x18, 0x48);
  for (std::size_t i = 0; i < w.size(); i += stride) {
    for (std::uint32_t j = 0; j < 16; ++j)
      w[i + j] = static_cast<std::uint8_t>(byte(rng));
    for (std::uint32_t j = 16; j < stride; ++j)
      w[i + j] = static_cast<std::uint8_t>(scale(rng));
  }
  return w;
}

struct Encoded {
  std::vector<std::uint64_t> packed;
  std::vector<float> scale;
  std::uint8_t* device = nullptr;
  float* device_scale = nullptr;
};

Encoded HostEncode(GgmlType type, const std::vector<std::uint8_t>& host,
                   const Int4Plan& plan, std::size_t m, std::size_t k) {
  const std::uint32_t stride = type == GgmlType::kQ4_0_ROCMFP4_FAST ? 17U : 18U;
  const std::size_t row_bytes = (k / 32) * stride;
  const std::size_t steps = plan.k_out / 16;
  Encoded e;
  e.packed.assign(m * steps, 0);
  e.scale.assign(m, 1e-12F / 7.F);
  for (std::size_t r = 0; r < m; ++r) {
    const std::uint8_t* src = host.data() + r * row_bytes;
    float peak = 1e-12F;
    for (std::size_t c = 0; c < k; ++c) {
      const float v = Fp4Value(src, static_cast<std::uint32_t>(c), stride);
      peak = std::max(peak, std::fabs(v));
    }
    const float step = peak / 7.F;
    e.scale[r] = step;
    for (std::size_t p = 0; p < plan.k_out; ++p) {
      const std::uint32_t entry = plan.order[p];
      int want = 0;
      if (entry != kInt4NoColumn) {
        const float v = Fp4Value(src, entry & kInt4ColumnMask, stride);
        const float coarse = std::clamp(std::floor(v / step + 0.5F), -7.F, 7.F);
        if ((entry & kInt4FinePass) != 0U)
          want = static_cast<int>(
              std::clamp(std::floor((v - coarse * step) / (step / 16.F) + 0.5F),
                         -7.F, 7.F));
        else
          want = static_cast<int>(coarse);
      }
      e.packed[r * steps + p / 16] |=
          static_cast<std::uint64_t>(static_cast<std::uint32_t>(want) & 15U)
          << (4U * (p % 16));
    }
  }
  HIP(hipMalloc(&e.device, m * (plan.k_out / 2)));
  HIP(hipMalloc(&e.device_scale, m * sizeof(float)));
  HIP(hipMemcpy(e.device, e.packed.data(),
                e.packed.size() * sizeof(std::uint64_t),
                hipMemcpyHostToDevice));
  HIP(hipMemcpy(e.device_scale, e.scale.data(), e.scale.size() * sizeof(float),
                hipMemcpyHostToDevice));
  return e;
}

void RunCase(std::size_t k, std::size_t m, std::size_t batch, GgmlType type,
             std::size_t fine_columns, unsigned seed) {
  std::mt19937 rng(seed);
  hipStream_t stream = nullptr;
  HIP(hipStreamCreate(&stream));
  std::lognormal_distribution<float> spread(0.F, 0.3F);
  std::vector<float> sigma(k);
  for (auto& s : sigma)
    s = 250.F * spread(rng);
  sigma[k / 3] = 14000.F;  // A column far above the rest.
  sigma[k / 2] = 2500.F;
  sigma[k / 2 + 1] = 2100.F;
  sigma[k / 4] = 3e-30F;  // A column that never fires.
  const Int4Plan plan = MakeInt4Plan(sigma.data(), k, fine_columns);
  CHECK(plan.k_out % 64 == 0 && plan.k_out >= k, "k_out %zu", plan.k_out);
  CHECK(plan.fine_columns >= fine_columns, "fine columns %zu of %zu",
        plan.fine_columns, fine_columns);
  std::vector<int> plain(k, 0), fine(k, 0), fine_input(k, 0);
  for (const auto c : plan.order) {
    if (c == kInt4NoColumn)
      continue;
    if ((c & kInt4FinePass) != 0U)
      ++fine[c & kInt4ColumnMask];
    else if ((c & kInt4FineInput) != 0U)
      ++fine_input[c & kInt4ColumnMask];
    else
      ++plain[c];
  }
  CHECK(std::all_of(plain.begin(), plain.end(), [](int n) { return n == 1; }),
        "every column appears once");
  CHECK(std::all_of(fine.begin(), fine.end(), [](int n) { return n <= 1; }),
        "a fine pass appears at most once");
  CHECK(std::all_of(fine_input.begin(), fine_input.end(),
                    [](int n) { return n <= 1; }),
        "a fine input pass appears at most once");
  std::printf("k %zu -> %zu, %d shifts, %zu fine columns\n", k, plan.k_out,
              plan.shifts.count, plan.fine_columns);

  const std::uint32_t stride = type == GgmlType::kQ4_0_ROCMFP4_FAST ? 17U : 18U;
  const float clip = 3.F;
  const auto gate_host = RandomFp4(rng, m, k, stride);
  const auto up_host = RandomFp4(rng, m, k, stride);
  Encoded gate = HostEncode(type, gate_host, plan, m, k);
  Encoded up = HostEncode(type, up_host, plan, m, k);

  std::normal_distribution<float> normal(0.F, 1.F);
  std::vector<__half> x_host(batch * k);
  for (std::size_t t = 0; t < batch; ++t) {
    for (std::size_t c = 0; c < k; ++c)
      x_host[t * k + c] =
          __float2half(std::fmin(normal(rng) * sigma[c] * 1e-3F, 30000.F));
  }
  auto* x = Upload(x_host);
  auto* order = Upload(plan.order);
  auto* inverse = Upload(plan.inverse);
  std::uint8_t* x4 = nullptr;
  float* ts_device = nullptr;
  HIP(hipMalloc(&x4, batch * (plan.k_out / 2)));
  HIP(hipMalloc(&ts_device, batch * sizeof(float)));
  superroc::hip::LaunchInt4PackHalf(x, x4, ts_device, order, inverse, plan.base,
                                    batch, k, plan.k_out, clip, stream);
  HIP(hipStreamSynchronize(stream));
  const auto x_packed = Download(reinterpret_cast<const std::uint64_t*>(x4),
                                 batch * (plan.k_out / 16));
  const auto ts = Download(ts_device, batch);
  std::size_t wrong = 0, total = 0;
  for (std::size_t t = 0; t < batch; ++t) {
    double sum = 0.0;
    for (std::size_t p = 0; p < plan.k_out; ++p) {
      if (plan.order[p] <= kInt4ColumnMask) {
        const double v =
            static_cast<double>(__half2float(x_host[t * k + plan.order[p]])) *
            plan.inverse[p];
        sum += v * v;
      }
    }
    const float step =
        std::max(static_cast<float>(std::sqrt(sum / static_cast<double>(k))) *
                     clip / 7.F,
                 1e-12F);
    CHECK(std::fabs(step * plan.base - ts[t]) <= 1e-4F * ts[t],
          "token %zu scale", t);
    const float gpu_step = ts[t] / plan.base;
    for (std::size_t p = 0; p < plan.k_out; ++p) {
      const int got = Nibble(x_packed[t * (plan.k_out / 16) + p / 16],
                             static_cast<int>(p % 16));
      int want = 0;
      const std::uint32_t entry = plan.order[p];
      if (entry != kInt4NoColumn) {
        const float v =
            __half2float(x_host[t * k + (entry & kInt4ColumnMask)]) *
            plan.inverse[p];
        const float coarse =
            std::clamp(std::floor(v / gpu_step + 0.5F), -7.F, 7.F);
        want = static_cast<int>(
            (entry & kInt4FineInput) != 0U
                ? std::clamp(
                      std::floor((v - coarse * gpu_step) / (gpu_step / 16.F) +
                                 0.5F),
                      -7.F, 7.F)
                : coarse);
      }
      wrong += got != want;
      ++total;
    }
  }
  CHECK(wrong * 10000 <= total, "pack: %zu of %zu nibbles differ", wrong,
        total);
  std::printf("pack: %zu of %zu nibbles differ\n", wrong, total);

  const std::size_t guard = 4096;
  std::vector<__half> y_init(batch * m + guard, __float2half(123.F));
  auto* y = Upload(y_init);
  superroc::hip::LaunchInt4GemmPair(
      gate.device, up.device, x4, gate.device_scale, up.device_scale, ts_device,
      plan.shifts, y, batch, m, plan.k_out, stream);
  HIP(hipStreamSynchronize(stream));
  const auto got = Download(y, batch * m + guard);
  // The reference runs the same Horner sum over the packed values, with the
  // same shifts.
  const std::size_t steps = plan.k_out / 16;
  double worst = 0.0;
  std::size_t off = 0, overflowed = 0;
  // Against the real product, from the unrounded weights and inputs: rounding
  // noise is small next to the signal, and a wrong shift, order or scale in the
  // plan is not.
  double err2 = 0.0, sig2 = 0.0;
  const std::size_t gate_row_bytes = (k / 32) * stride;
  for (std::size_t t = 0; t < batch; ++t) {
    for (std::size_t r = 0; r < m; ++r) {
      std::int64_t ag = 0, au = 0;
      int next = 0;
      for (std::size_t s = 0; s < steps; ++s) {
        if (next < plan.shifts.count &&
            static_cast<int>(s) == plan.shifts.at[next]) {
          ag <<= plan.shifts.bits[next];
          au <<= plan.shifts.bits[next];
          ++next;
        }
        const std::uint64_t xv = x_packed[t * steps + s],
                            gv = gate.packed[r * steps + s],
                            uv = up.packed[r * steps + s];
        for (int j = 0; j < 16; ++j) {
          ag += Nibble(xv, j) * Nibble(gv, j);
          au += Nibble(xv, j) * Nibble(uv, j);
        }
      }
      const float g = static_cast<float>(ag) * gate.scale[r] * ts[t];
      if (t % 7 == 0 && r % 5 == 0) {
        double truth = 0.0;
        for (std::size_t c = 0; c < k; ++c)
          truth += static_cast<double>(
                       Fp4Value(gate_host.data() + r * gate_row_bytes,
                                static_cast<std::uint32_t>(c), stride)) *
                   __half2float(x_host[t * k + c]);
        err2 += (truth - g) * (truth - g);
        sig2 += truth * truth;
      }
      const float u = static_cast<float>(au) * up.scale[r] * ts[t];
      const float want = g * (1.F / (1.F + std::exp(-g))) * u;
      const float value = __half2float(got[t * m + r]);
      // The store is a half float, so a product past its range reads as
      // infinity on both sides.
      const float want_half = __half2float(__float2half(want));
      if (std::isinf(value) && value == want_half) {
        ++overflowed;
        continue;
      }
      const double err = std::fabs(static_cast<double>(value) - want_half) /
                         (std::fabs(want_half) + 1e-2);
      worst = std::max(worst, err);
      off += err > 2e-2;
    }
  }
  CHECK(off == 0, "pair: %zu values off, worst relative %g", off, worst);
  std::printf(
      "pair: %zu values off, worst relative %g, %zu past the half range\n", off,
      worst, overflowed);
  const double relative = std::sqrt(err2 / std::max(sig2, 1e-30));
  CHECK(relative < 0.5, "gate against the real product: relative rms %g",
        relative);
  std::printf("gate against the real product: relative rms %.3f\n", relative);
  for (std::size_t i = 0; i < guard; ++i)
    CHECK(__half2float(got[batch * m + i]) == 123.F,
          "wrote past the end at %zu", i);

  // One matrix, stored as floats.
  std::vector<float> y_float_init(batch * m + guard, 7.F);
  auto* y_float = Upload(y_float_init);
  superroc::hip::LaunchInt4GemmStore(gate.device, x4, gate.device_scale,
                                     ts_device, plan.shifts, y_float, batch, m,
                                     plan.k_out, stream);
  HIP(hipStreamSynchronize(stream));
  const auto stored = Download(y_float, batch * m + guard);
  double worst_store = 0.0;
  for (std::size_t t = 0; t < batch; ++t) {
    for (std::size_t r = 0; r < m; ++r) {
      std::int64_t ag = 0;
      int next = 0;
      for (std::size_t s = 0; s < steps; ++s) {
        if (next < plan.shifts.count &&
            static_cast<int>(s) == plan.shifts.at[next]) {
          ag <<= plan.shifts.bits[next];
          ++next;
        }
        const std::uint64_t xv = x_packed[t * steps + s],
                            gv = gate.packed[r * steps + s];
        for (int j = 0; j < 16; ++j)
          ag += Nibble(xv, j) * Nibble(gv, j);
      }
      const float want = static_cast<float>(ag) * gate.scale[r] * ts[t];
      worst_store =
          std::max(worst_store,
                   std::fabs(static_cast<double>(stored[t * m + r]) - want) /
                       (std::fabs(want) + 1e-2));
    }
  }
  CHECK(worst_store < 1e-5, "store: worst relative %g", worst_store);
  for (std::size_t i = 0; i < guard; ++i)
    CHECK(stored[batch * m + i] == 7.F, "store wrote past the end at %zu", i);
  std::printf("store: worst relative %g\n", worst_store);
  HIP(hipFree(y_float));

  // With a fine pass, the two nibbles of an input column must rebuild its
  // value within a sixteenth of a step, token by token.
  if (fine_columns != 0) {
    std::size_t far_input = 0, checked_input = 0;
    for (std::size_t t = 0; t < batch; ++t) {
      const float gpu_step = ts[t] / plan.base;
      for (std::size_t p = 0; p < plan.k_out; ++p) {
        const std::uint32_t entry = plan.order[p];
        if (entry == kInt4NoColumn || (entry & kInt4FineInput) == 0U)
          continue;
        const std::uint32_t c = entry & kInt4ColumnMask;
        int coarse = 0;
        for (std::size_t q = 0; q < plan.k_out; ++q) {
          if (plan.order[q] == c)
            coarse =
                Nibble(x_packed[t * steps + q / 16], static_cast<int>(q % 16));
        }
        const int fine_nibble =
            Nibble(x_packed[t * steps + p / 16], static_cast<int>(p % 16));
        const float truth = __half2float(x_host[t * k + c]) * plan.inverse[p];
        const float rebuilt = (static_cast<float>(coarse) +
                               static_cast<float>(fine_nibble) / 16.F) *
                              gpu_step;
        const bool clipped = std::fabs(truth) > 7.5F * gpu_step;
        far_input += !clipped && std::fabs(rebuilt - truth) > gpu_step / 16.F;
        ++checked_input;
      }
    }
    CHECK(far_input == 0, "fine input pass: %zu of %zu inputs not rebuilt",
          far_input, checked_input);
    std::printf("fine input pass: %zu of %zu inputs not rebuilt\n", far_input,
                checked_input);
  }
  HIP(hipFree(x));
  HIP(hipFree(order));
  HIP(hipFree(inverse));
  HIP(hipFree(x4));
  HIP(hipFree(ts_device));
  HIP(hipFree(y));
  HIP(hipFree(gate.device));
  HIP(hipFree(gate.device_scale));
  HIP(hipFree(up.device));
  HIP(hipFree(up.device_scale));
  HIP(hipStreamDestroy(stream));
}
}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("no GPU\n");
    return 77;
  }
  RunCase(512, 300, 560, GgmlType::kQ4_0_ROCMFP4_FAST, 0, 11);
  RunCase(512, 128, 256, GgmlType::kQ4_0_ROCMFP4, 16, 12);
  RunCase(1024, 257, 513, GgmlType::kQ4_0_ROCMFP4_FAST, 32, 13);
  std::printf("%s\n", g_failed == 0 ? "PASS" : "FAILED");
  return g_failed == 0 ? 0 : 1;
}
#else
int main() {
  return 77;
}
#endif
