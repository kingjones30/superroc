// LFM2 (SPEC3) fp16 token-major KV GPU test. The fp16 kernels convert each
// element to fp32 on load and otherwise run the fp32 kernels' math, so on a
// cache whose fp32 values are exactly the fp16 values they must match the
// fp32 kernels bit for bit; both are also checked against a CPU double
// reference. The writes must store round-to-nearest-even fp16 bits at
// ((layer*max_ctx+pos)*kv_heads+kvh)*64+d and touch nothing else.
// Exit 0 pass, 1 fail, 77 no HIP device.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/models/lfm2/hip/lfm2_attention.hpp"
#include "src/models/lfm2/hip/lfm2_kernels.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

#if !defined(ENGINE_ENABLE_HIP)
int main() {
  std::cerr << "FAIL lfm2_kv_f16_test built without HIP\n";
  return 1;
}
#else
#include <hip/hip_runtime.h>

namespace {

int g_fail = 0;

void Report(const std::string& cas, bool pass, const std::string& numbers) {
  std::cout << (pass ? "PASS " : "FAIL ") << cas << ": " << numbers << "\n";
  if (!pass) {
    g_fail = 1;
  }
}

bool HipOk(hipError_t err, const char* what) {
  if (err == hipSuccess) {
    return true;
  }
  std::cerr << "FAIL hip " << what << " code=" << static_cast<int>(err) << "\n";
  g_fail = 1;
  return false;
}

bool CheckOk(const gufo::hip::Lfm2LaunchStatus& st, const char* what,
             const std::string& cas) {
  if (st.ok) {
    return true;
  }
  std::cout << "FAIL " << cas << ": " << what << " "
            << (st.error != nullptr ? st.error : "?") << "\n";
  g_fail = 1;
  return false;
}

constexpr std::uint32_t kHeads = 32;
constexpr std::uint32_t kKvHeads = 8;
constexpr std::uint32_t kHeadDim = 64;
constexpr std::uint32_t kKv = kKvHeads * kHeadDim;
constexpr std::uint32_t kMaxCtx = 4096;
constexpr std::uint32_t kLayers = 2;
constexpr std::uint32_t kMaxSplits = 64;
constexpr double kTol = 2e-5;

// Host fp16 round to nearest even (clang _Float16), as bits and back.
std::uint16_t HalfBits(float x) {
  return std::bit_cast<std::uint16_t>(static_cast<_Float16>(x));
}
float HalfValue(std::uint16_t bits) {
  return static_cast<float>(std::bit_cast<_Float16>(bits));
}

std::size_t HeadMajor(std::uint32_t layer, std::uint32_t kvh, std::uint32_t p,
                      std::uint32_t d) {
  return ((static_cast<std::size_t>(layer) * kKvHeads + kvh) * kMaxCtx + p) *
             kHeadDim +
         d;
}
std::size_t TokenMajor(std::uint32_t layer, std::uint32_t kvh, std::uint32_t p,
                       std::uint32_t d) {
  return ((static_cast<std::size_t>(layer) * kMaxCtx + p) * kKvHeads + kvh) *
             kHeadDim +
         d;
}

// CPU double attention of q head h (row of q at q_row) over positions 0..last.
void CpuHead(const float* qh, const float* kc, const float* vc,
             std::uint32_t layer, std::uint32_t kvh, std::uint32_t last,
             float* out) {
  const double scale = 1.0 / std::sqrt(64.0);
  std::vector<double> s(last + 1);
  double m = -1e300;
  for (std::uint32_t p = 0; p <= last; ++p) {
    double dot = 0.0;
    for (std::uint32_t d = 0; d < kHeadDim; ++d) {
      dot += static_cast<double>(qh[d]) *
             static_cast<double>(kc[HeadMajor(layer, kvh, p, d)]);
    }
    s[p] = dot * scale;
    m = std::max(m, s[p]);
  }
  double l = 0.0;
  for (std::uint32_t p = 0; p <= last; ++p) {
    s[p] = std::exp(s[p] - m);
    l += s[p];
  }
  for (std::uint32_t d = 0; d < kHeadDim; ++d) {
    double a = 0.0;
    for (std::uint32_t p = 0; p <= last; ++p) {
      a += s[p] * static_cast<double>(vc[HeadMajor(layer, kvh, p, d)]);
    }
    out[d] = static_cast<float>(a / l);
  }
}

double MaxRatio(const float* out, const float* ref, std::size_t n) {
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double diff =
        std::fabs(static_cast<double>(out[i]) - static_cast<double>(ref[i]));
    worst =
        std::max(worst, diff / (1.0 + std::fabs(static_cast<double>(ref[i]))));
  }
  return worst;
}

template<typename T>
bool Upload(T* dst, const std::vector<T>& src, const char* what) {
  return HipOk(
      hipMemcpy(dst, src.data(), src.size() * sizeof(T), hipMemcpyHostToDevice),
      what);
}

template<typename T>
bool Download(std::vector<T>& dst, const T* src, const char* what) {
  return HipOk(
      hipMemcpy(dst.data(), src, dst.size() * sizeof(T), hipMemcpyDeviceToHost),
      what);
}

}  // namespace

int main() {
  const int gate = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "lfm2_kv_f16_test");
  if (gate != 0) {
    return gate;
  }
  std::mt19937 rng(20261015);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

  const std::size_t plane =
      static_cast<std::size_t>(kLayers) * kKvHeads * kMaxCtx * kHeadDim;
  // Rounded fp32 head-major cache and its fp16 token-major image.
  std::vector<float> h_kr(plane), h_vr(plane);
  std::vector<std::uint16_t> h_k16(plane), h_v16(plane);
  for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
    for (std::uint32_t kvh = 0; kvh < kKvHeads; ++kvh) {
      for (std::uint32_t p = 0; p < kMaxCtx; ++p) {
        for (std::uint32_t d = 0; d < kHeadDim; ++d) {
          const std::uint16_t kb = HalfBits(dist(rng));
          const std::uint16_t vb = HalfBits(dist(rng));
          h_kr[HeadMajor(layer, kvh, p, d)] = HalfValue(kb);
          h_vr[HeadMajor(layer, kvh, p, d)] = HalfValue(vb);
          h_k16[TokenMajor(layer, kvh, p, d)] = kb;
          h_v16[TokenMajor(layer, kvh, p, d)] = vb;
        }
      }
    }
  }

  float *d_kr = nullptr, *d_vr = nullptr, *d_q = nullptr, *d_o32 = nullptr,
        *d_o16 = nullptr, *d_partial = nullptr, *d_k = nullptr, *d_v = nullptr;
  std::uint16_t *d_k16 = nullptr, *d_v16 = nullptr;
  std::uint32_t* d_pos = nullptr;
  constexpr std::uint32_t kMaxBatch = 300;
  const std::size_t q_max =
      static_cast<std::size_t>(kMaxBatch) * kHeads * kHeadDim;
  const std::size_t partial_elems =
      static_cast<std::size_t>(kHeads) * kMaxSplits * (kHeadDim + 2);
  if (!HipOk(hipMalloc(&d_kr, plane * sizeof(float)), "malloc kr") ||
      !HipOk(hipMalloc(&d_vr, plane * sizeof(float)), "malloc vr") ||
      !HipOk(hipMalloc(&d_k16, plane * sizeof(std::uint16_t)), "malloc k16") ||
      !HipOk(hipMalloc(&d_v16, plane * sizeof(std::uint16_t)), "malloc v16") ||
      !HipOk(hipMalloc(&d_q, q_max * sizeof(float)), "malloc q") ||
      !HipOk(hipMalloc(&d_o32, q_max * sizeof(float)), "malloc o32") ||
      !HipOk(hipMalloc(&d_o16, q_max * sizeof(float)), "malloc o16") ||
      !HipOk(hipMalloc(&d_partial, partial_elems * sizeof(float)),
             "malloc partial") ||
      !HipOk(hipMalloc(&d_k, kMaxBatch * kKv * sizeof(float)), "malloc k") ||
      !HipOk(hipMalloc(&d_v, kMaxBatch * kKv * sizeof(float)), "malloc v") ||
      !HipOk(hipMalloc(&d_pos, sizeof(std::uint32_t)), "malloc pos")) {
    return 1;
  }
  if (!Upload(d_kr, h_kr, "up kr") || !Upload(d_vr, h_vr, "up vr") ||
      !Upload(d_k16, h_k16, "up k16") || !Upload(d_v16, h_v16, "up v16")) {
    return 1;
  }

  // 1. Decode attention: fp16 == fp32-on-rounded bitwise, and vs CPU double.
  const std::size_t q1 = static_cast<std::size_t>(kHeads) * kHeadDim;
  std::vector<float> h_q(q1), h_o32(q1), h_o16(q1), h_ref(q1);
  const std::uint32_t kPos[] = {0, 1, 31, 32, 33, 1000, 4095};
  const std::uint32_t kSplits[] = {1, 8, 32, 64};
  for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
    for (std::uint32_t pos : kPos) {
      for (float& x : h_q) {
        x = dist(rng);
      }
      if (!HipOk(hipMemcpy(d_q, h_q.data(), q1 * sizeof(float),
                           hipMemcpyHostToDevice),
                 "up q") ||
          !HipOk(hipMemcpy(d_pos, &pos, sizeof(pos), hipMemcpyHostToDevice),
                 "up pos")) {
        return 1;
      }
      for (std::uint32_t h = 0; h < kHeads; ++h) {
        CpuHead(h_q.data() + static_cast<std::size_t>(h) * kHeadDim,
                h_kr.data(), h_vr.data(), layer, h / (kHeads / kKvHeads), pos,
                h_ref.data() + static_cast<std::size_t>(h) * kHeadDim);
      }
      for (std::uint32_t ns : kSplits) {
        char cas[128];
        std::snprintf(cas, sizeof(cas), "decode f16 layer=%u pos=%u splits=%u",
                      layer, pos, ns);
        if (!CheckOk(gufo::hip::LaunchLfm2DecodeAttention(
                         d_q, d_kr, d_vr, d_o32, layer, d_pos, kMaxCtx, kHeads,
                         kKvHeads, kHeadDim, ns, d_partial, nullptr),
                     "fp32", cas) ||
            !HipOk(hipDeviceSynchronize(), "sync32") ||
            !Download(h_o32, d_o32, "down o32") ||
            !CheckOk(gufo::hip::LaunchLfm2DecodeAttentionF16(
                         d_q, d_k16, d_v16, d_o16, layer, d_pos, kMaxCtx,
                         kHeads, kKvHeads, kHeadDim, ns, d_partial, nullptr),
                     "fp16", cas) ||
            !HipOk(hipDeviceSynchronize(), "sync16") ||
            !Download(h_o16, d_o16, "down o16")) {
          return 1;
        }
        const bool same =
            std::memcmp(h_o32.data(), h_o16.data(), q1 * sizeof(float)) == 0;
        const double r = MaxRatio(h_o16.data(), h_ref.data(), q1);
        char nums[128];
        std::snprintf(nums, sizeof(nums), "bitwise_fp32=%d cpu_ratio=%.3g",
                      same ? 1 : 0, r);
        Report(cas, same && r <= kTol, nums);
      }
    }
  }

  // 2. Prefill attention: fp16 == fp32-on-rounded bitwise, and vs CPU double.
  const std::uint32_t kBatch[] = {1, 63, 64, 65, 300};
  const std::uint32_t kStart[] = {0, 100, 3796};
  for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
    for (std::uint32_t start : kStart) {
      for (std::uint32_t batch : kBatch) {
        if (start + batch > kMaxCtx) {
          continue;
        }
        char cas[128];
        std::snprintf(cas, sizeof(cas),
                      "prefill f16 layer=%u start=%u batch=%u", layer, start,
                      batch);
        const std::size_t qn =
            static_cast<std::size_t>(batch) * kHeads * kHeadDim;
        std::vector<float> q(qn), o32(qn), o16(qn), ref(qn);
        for (float& x : q) {
          x = dist(rng);
        }
        if (!HipOk(hipMemcpy(d_q, q.data(), qn * sizeof(float),
                             hipMemcpyHostToDevice),
                   "up pq")) {
          return 1;
        }
        if (!CheckOk(gufo::hip::LaunchLfm2PrefillAttention(
                         d_q, d_kr, d_vr, d_o32, layer, start, batch, kMaxCtx,
                         kHeads, kKvHeads, nullptr),
                     "fp32", cas) ||
            !HipOk(hipDeviceSynchronize(), "psync32") ||
            !Download(o32, d_o32, "down p32") ||
            !CheckOk(gufo::hip::LaunchLfm2PrefillAttentionF16(
                         d_q, d_k16, d_v16, d_o16, layer, start, batch, kMaxCtx,
                         kHeads, kKvHeads, nullptr),
                     "fp16", cas) ||
            !HipOk(hipDeviceSynchronize(), "psync16") ||
            !Download(o16, d_o16, "down p16")) {
          return 1;
        }
        for (std::uint32_t r = 0; r < batch; ++r) {
          for (std::uint32_t h = 0; h < kHeads; ++h) {
            const std::size_t off =
                (static_cast<std::size_t>(r) * kHeads + h) * kHeadDim;
            CpuHead(q.data() + off, h_kr.data(), h_vr.data(), layer,
                    h / (kHeads / kKvHeads), start + r, ref.data() + off);
          }
        }
        const bool same =
            std::memcmp(o32.data(), o16.data(), qn * sizeof(float)) == 0;
        const double rr = MaxRatio(o16.data(), ref.data(), qn);
        char nums[128];
        std::snprintf(nums, sizeof(nums), "bitwise_fp32=%d cpu_ratio=%.3g",
                      same ? 1 : 0, rr);
        Report(cas, same && rr <= kTol, nums);
      }
    }
  }

  // 3. Writes: exact RNE bits in the right rows, nothing else touched.
  std::vector<float> h_k(static_cast<std::size_t>(kMaxBatch) * kKv),
      h_v(static_cast<std::size_t>(kMaxBatch) * kKv);
  std::vector<std::uint16_t> got_k(plane), got_v(plane), want_k(plane),
      want_v(plane);
  struct WriteCase {
    std::uint32_t layer, start, batch;
    bool single;  // LaunchLfm2WriteKvDevF16 (device pos) vs batched
  };
  const WriteCase kWrites[] = {{0, 0, 1, true},      {1, 1000, 1, true},
                               {1, 4095, 1, true},   {0, 0, 300, false},
                               {1, 100, 300, false}, {1, 3796, 300, false},
                               {0, 4095, 1, false}};
  for (const WriteCase& w : kWrites) {
    char cas[128];
    std::snprintf(cas, sizeof(cas), "write f16 %s layer=%u start=%u batch=%u",
                  w.single ? "dev" : "batched", w.layer, w.start, w.batch);
    for (float& x : h_k) {
      x = 4.0F * dist(rng);
    }
    for (float& x : h_v) {
      x = 4.0F * dist(rng);
    }
    std::fill(want_k.begin(), want_k.end(), std::uint16_t{0x7e00});
    std::fill(want_v.begin(), want_v.end(), std::uint16_t{0x7e00});
    for (std::uint32_t r = 0; r < w.batch; ++r) {
      for (std::uint32_t i = 0; i < kKv; ++i) {
        const std::size_t dst =
            (static_cast<std::size_t>(w.layer) * kMaxCtx + w.start + r) * kKv +
            i;
        want_k[dst] = HalfBits(h_k[static_cast<std::size_t>(r) * kKv + i]);
        want_v[dst] = HalfBits(h_v[static_cast<std::size_t>(r) * kKv + i]);
      }
    }
    // Fill the planes with an fp16 NaN pattern (0x7e00) so stray writes show.
    std::vector<std::uint16_t> nan_plane(plane, std::uint16_t{0x7e00});
    if (!Upload(d_k16, nan_plane, "nan k") ||
        !Upload(d_v16, nan_plane, "nan v") ||
        !HipOk(
            hipMemcpy(d_k, h_k.data(),
                      static_cast<std::size_t>(w.batch) * kKv * sizeof(float),
                      hipMemcpyHostToDevice),
            "up wk") ||
        !HipOk(
            hipMemcpy(d_v, h_v.data(),
                      static_cast<std::size_t>(w.batch) * kKv * sizeof(float),
                      hipMemcpyHostToDevice),
            "up wv") ||
        !HipOk(
            hipMemcpy(d_pos, &w.start, sizeof(w.start), hipMemcpyHostToDevice),
            "up wpos")) {
      return 1;
    }
    const gufo::hip::Lfm2LaunchStatus st =
        w.single
            ? gufo::hip::LaunchLfm2WriteKvDevF16(d_k, d_v, d_k16, d_v16,
                                                 w.layer, d_pos, kMaxCtx,
                                                 kKvHeads, kHeadDim, nullptr)
            : gufo::hip::LaunchLfm2BatchedWriteKvF16(d_k, d_v, d_k16, d_v16,
                                                     w.layer, w.start, w.batch,
                                                     kMaxCtx, kKv, nullptr);
    if (!CheckOk(st, "launch", cas) ||
        !HipOk(hipDeviceSynchronize(), "wsync") ||
        !Download(got_k, d_k16, "down wk") ||
        !Download(got_v, d_v16, "down wv")) {
      return 1;
    }
    const bool k_eq = got_k == want_k;
    const bool v_eq = got_v == want_v;
    Report(cas, k_eq && v_eq,
           std::string("k_eq=") + (k_eq ? "1" : "0") +
               " v_eq=" + (v_eq ? "1" : "0"));
  }

  // 3b. HKV: both writes saturate at +-65504 (inf and out-of-range values),
  // keep NaN a NaN, and leave in-range values bit-exact.
  {
    const float kSpecial[] = {1.0e6F,   -1.0e6F,   70000.0F, -70000.0F,
                              65519.0F, 65520.0F,  65504.0F, -65504.0F,
                              INFINITY, -INFINITY, NAN,      1.5F};
    constexpr std::size_t kNs = sizeof(kSpecial) / sizeof(kSpecial[0]);
    for (const bool single : {true, false}) {
      const std::uint32_t layer = single ? 0U : 1U;
      const std::uint32_t start = single ? 7U : 9U;
      for (std::size_t i = 0; i < kKv; ++i) {
        h_k[i] = i < kNs ? kSpecial[i] : 0.25F;
        h_v[i] = i < kNs ? -kSpecial[i] : -0.25F;
      }
      std::vector<std::uint16_t> nan_plane(plane, std::uint16_t{0x7e00});
      if (!Upload(d_k16, nan_plane, "sat nan k") ||
          !Upload(d_v16, nan_plane, "sat nan v") ||
          !HipOk(hipMemcpy(d_k, h_k.data(), kKv * sizeof(float),
                           hipMemcpyHostToDevice),
                 "sat up k") ||
          !HipOk(hipMemcpy(d_v, h_v.data(), kKv * sizeof(float),
                           hipMemcpyHostToDevice),
                 "sat up v") ||
          !HipOk(hipMemcpy(d_pos, &start, sizeof(start), hipMemcpyHostToDevice),
                 "sat up pos")) {
        return 1;
      }
      const gufo::hip::Lfm2LaunchStatus st =
          single
              ? gufo::hip::LaunchLfm2WriteKvDevF16(d_k, d_v, d_k16, d_v16,
                                                   layer, d_pos, kMaxCtx,
                                                   kKvHeads, kHeadDim, nullptr)
              : gufo::hip::LaunchLfm2BatchedWriteKvF16(d_k, d_v, d_k16, d_v16,
                                                       layer, start, 1, kMaxCtx,
                                                       kKv, nullptr);
      const char* cas =
          single ? "write f16 dev saturates" : "write f16 batched saturates";
      if (!CheckOk(st, "launch", cas) ||
          !HipOk(hipDeviceSynchronize(), "sat sync") ||
          !Download(got_k, d_k16, "sat down k") ||
          !Download(got_v, d_v16, "sat down v")) {
        return 1;
      }
      bool ok = true;
      for (std::size_t i = 0; i < kKv; ++i) {
        const std::size_t dst =
            (static_cast<std::size_t>(layer) * kMaxCtx + start) * kKv + i;
        for (const bool is_k : {true, false}) {
          const float x = is_k ? h_k[i] : h_v[i];
          const std::uint16_t got = is_k ? got_k[dst] : got_v[dst];
          if (std::isnan(x)) {
            ok = ok && (got & 0x7c00U) == 0x7c00U && (got & 0x03ffU) != 0U;
          } else {
            ok = ok && got == HalfBits(std::clamp(x, -65504.0F, 65504.0F));
          }
        }
      }
      Report(cas, ok,
             ok ? "inf/out-of-range -> +-65504, NaN kept" : "mismatch");
    }
  }

  // 4. Refusals: batched write past the context end.
  {
    const gufo::hip::Lfm2LaunchStatus st =
        gufo::hip::LaunchLfm2BatchedWriteKvF16(d_k, d_v, d_k16, d_v16, 0, 4000,
                                               300, kMaxCtx, kKv, nullptr);
    Report("write f16 batched refuses range", !st.ok,
           st.error != nullptr ? st.error : "(accepted)");
  }

  std::cout << (g_fail == 0 ? "ALL LFM2_KV_F16 OK" : "LFM2_KV_F16 FAILED")
            << "\n";
  return g_fail == 0 ? 0 : 1;
}
#endif
