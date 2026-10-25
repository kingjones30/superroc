#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/moe/forward_moe.hpp"
#include "src/models/qwen/moe/qwen_moe_ffn.hpp"
#include "src/models/qwen/moe/qwen_moe_route.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

void Expect(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL " << msg << "\n";
    std::exit(1);
  }
}

#define HIP_REQUIRE(expr)                                                    \
  do {                                                                       \
    const hipError_t err_ = (expr);                                          \
    if (err_ != hipSuccess) {                                                \
      std::cerr << "FAIL hip " << hipGetErrorString(err_) << "\n";         \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

struct DeviceF32 {
  float* p = nullptr;
  explicit DeviceF32(std::size_t n) {
    HIP_REQUIRE(hipMalloc(&p, n * sizeof(float)));
  }
  ~DeviceF32() {
    const hipError_t e = hipFree(p);
    (void)e;
  }
  DeviceF32(const DeviceF32&) = delete;
  DeviceF32& operator=(const DeviceF32&) = delete;
};

struct DeviceI32 {
  int* p = nullptr;
  explicit DeviceI32(std::size_t n) {
    HIP_REQUIRE(hipMalloc(&p, n * sizeof(int)));
  }
  ~DeviceI32() {
    const hipError_t e = hipFree(p);
    (void)e;
  }
};

float Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  return static_cast<float>(dot / std::sqrt(na * nb));
}

void Compare(const std::vector<float>& x, const std::vector<float>& gate, int k,
             int n_expert, const char* name) {
  const int n_used = 8;
  std::vector<float> logits(static_cast<std::size_t>(n_expert));
  for (int e = 0; e < n_expert; ++e) {
    float acc = 0.0F;
    const float* row = gate.data() + static_cast<std::size_t>(e) * k;
    for (int i = 0; i < k; ++i) {
      acc += row[i] * x[static_cast<std::size_t>(i)];
    }
    logits[static_cast<std::size_t>(e)] = acc;
  }
  std::vector<int> cpu_ids(n_used);
  std::vector<float> cpu_w(n_used);
  Expect(gufo::models::KatRouteSoftmaxTopK(logits, n_used, cpu_ids, cpu_w),
         "cpu route");

  DeviceF32 d_x(x.size());
  DeviceF32 d_gate(gate.size());
  DeviceF32 d_logits(static_cast<std::size_t>(n_expert));
  DeviceF32 d_w(n_used);
  DeviceI32 d_ids(n_used);
  HIP_REQUIRE(hipMemcpy(d_x.p, x.data(), x.size() * sizeof(float),
                        hipMemcpyHostToDevice));
  HIP_REQUIRE(hipMemcpy(d_gate.p, gate.data(), gate.size() * sizeof(float),
                        hipMemcpyHostToDevice));
  HIP_REQUIRE(hipMemset(d_ids.p, 0, sizeof(int) * n_used));
  Expect(LaunchQwenMoeRoute(nullptr, d_gate.p, k, n_expert, n_used, d_logits.p,
                            d_ids.p, d_w.p, nullptr) == 1,
         "null x rejected");
  Expect(LaunchQwenMoeRoute(d_x.p, d_gate.p, k, n_expert, n_used, d_logits.p,
                            d_ids.p, d_w.p, nullptr) == 0,
         "launch");
  HIP_REQUIRE(hipDeviceSynchronize());
  std::vector<float> got_logits(static_cast<std::size_t>(n_expert));
  std::vector<int> got_ids(n_used);
  std::vector<float> got_w(n_used);
  HIP_REQUIRE(hipMemcpy(got_logits.data(), d_logits.p,
                        got_logits.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
  HIP_REQUIRE(hipMemcpy(got_ids.data(), d_ids.p, sizeof(int) * n_used,
                        hipMemcpyDeviceToHost));
  HIP_REQUIRE(hipMemcpy(got_w.data(), d_w.p, sizeof(float) * n_used,
                        hipMemcpyDeviceToHost));
  // KAT: the device dot contracts to FMA (gfx1151, clang default
  // fp-contract=fast) while this host reference is plain mul+add on baseline
  // x86-64, so logits may differ by a few ulp. G1 gates the id SET bit-exact
  // and the weight cosine; the logits are gated at cosine >= 0.999999 plus a
  // bounded max-abs delta, and both numbers are printed.
  double logit_max_abs = 0.0;
  double logit_ref_absmax = 0.0;
  for (int e = 0; e < n_expert; ++e) {
    const double d =
        std::fabs(static_cast<double>(got_logits[static_cast<std::size_t>(e)]) -
                  static_cast<double>(logits[static_cast<std::size_t>(e)]));
    logit_max_abs = std::max(logit_max_abs, d);
    logit_ref_absmax = std::max(
        logit_ref_absmax,
        std::fabs(static_cast<double>(logits[static_cast<std::size_t>(e)])));
  }
  const float logit_cosine = Cosine(got_logits, logits);
  Expect(logit_cosine >= 0.999999F, std::string(name) + " logit cosine");
  Expect(logit_max_abs <= 1.0e-3 * std::max(1.0, logit_ref_absmax),
         std::string(name) + " logit max abs");
  std::vector<int> cpu_set = cpu_ids;
  std::vector<int> gpu_set = got_ids;
  std::sort(cpu_set.begin(), cpu_set.end());
  std::sort(gpu_set.begin(), gpu_set.end());
  Expect(cpu_set == gpu_set, std::string(name) + " id set");
  const float cosine = Cosine(cpu_w, got_w);
  float cpu_sum = 0.0F;
  float gpu_sum = 0.0F;
  for (int i = 0; i < n_used; ++i) {
    cpu_sum += cpu_w[static_cast<std::size_t>(i)];
    gpu_sum += got_w[static_cast<std::size_t>(i)];
  }
  std::cout << "KAT_ROUTE " << name << " k=" << k << " e=" << n_expert
            << " logit_cosine=" << logit_cosine
            << " logit_max_abs=" << logit_max_abs
            << " ids_match=1 weight_cosine=" << cosine
            << " weight_sum_cpu=" << cpu_sum
            << " weight_sum_gpu=" << gpu_sum << "\n";
  Expect(cosine >= 0.999999F, std::string(name) + " weight cosine");
  // Old unnormalised weights sum to the top-8 mass, not 1. Both ends divide
  // by their own rank-order sum, so each sum is 1 even when the probs differ
  // by a few ulp.
  Expect(std::fabs(cpu_sum - 1.0F) < 1.0e-4F, std::string(name) + " cpu renorm");
  Expect(std::fabs(gpu_sum - 1.0F) < 1.0e-3F, std::string(name) + " gpu renorm");
}

// Super ROC QRH: host reference of the one-wave head (same rule as the
// device: bigger logit first, equal logits -> lower id, NaN read as -inf;
// weights = softmax over the picked logits in rank order).
void QrhRefHead(const float* l, int n_expert, int n_used, int* ids,
                float* w) {
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<std::pair<float, int>> c(static_cast<std::size_t>(n_expert));
  for (int e = 0; e < n_expert; ++e) {
    const float x = l[e];
    c[static_cast<std::size_t>(e)] = {std::isnan(x) ? -inf : x, e};
  }
  std::sort(c.begin(), c.end(), [](const std::pair<float, int>& a,
                                   const std::pair<float, int>& b) {
    return a.first > b.first || (a.first == b.first && a.second < b.second);
  });
  const float top = c[0].first;
  float ex[8];
  float sum = 0.0F;
  for (int r = 0; r < n_used; ++r) {
    const float v = c[static_cast<std::size_t>(r)].first;
    ex[r] = std::isinf(top) ? (v == top ? 1.0F : 0.0F) : std::exp(v - top);
    sum += ex[r];
    ids[r] = c[static_cast<std::size_t>(r)].second;
  }
  for (int r = 0; r < n_used; ++r) {
    w[r] = ex[r] / sum;
  }
}

// 420 logit vectors over six shapes, with exact ties, all-equal, -inf, NaN
// and +inf cases. The batch head (8 tokens per block) and the single-token
// head must agree bit for bit, ids must equal the reference exactly and
// weights within 2e-6 relative (device vs host expf).
void QrhHeadCases() {
  struct Shape {
    int n_expert;
    int n_used;
  };
  const Shape shapes[] = {{256, 8}, {128, 8}, {64, 4},
                          {60, 8},  {256, 1}, {33, 8}};
  constexpr int kPer = 70;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::uint32_t s = 0x9E3779B9u;
  auto next = [&s]() {
    s = s * 1664525u + 1013904223u;
    return s;
  };
  int vectors = 0;
  int mismatched = 0;
  double worst_rel = 0.0;
  for (const Shape& sh : shapes) {
    const int ne = sh.n_expert;
    const int nu = sh.n_used;
    std::vector<float> lg(static_cast<std::size_t>(kPer) * ne);
    for (int v = 0; v < kPer; ++v) {
      float* l = lg.data() + static_cast<std::size_t>(v) * ne;
      for (int e = 0; e < ne; ++e) {
        const float u = static_cast<float>(next() % 40001u) / 1000.0F - 20.0F;
        switch (v % 7) {
          case 0:
            l[e] = std::round(u * 4.0F) / 4.0F;  // many exact ties
            break;
          case 1:
            l[e] = 1.5F;  // all equal
            break;
          case 4:
            l[e] = -inf;  // all -inf
            break;
          default:
            l[e] = u;
            break;
        }
      }
      if (v % 7 == 2) {
        l[next() % static_cast<std::uint32_t>(ne)] = nan;
        l[next() % static_cast<std::uint32_t>(ne)] = -inf;
        l[next() % static_cast<std::uint32_t>(ne)] = nan;
      }
      if (v % 7 == 3) {
        l[next() % static_cast<std::uint32_t>(ne)] = inf;
      }
      if (v % 7 == 5) {
        for (int e = 0; e < ne; ++e) {
          l[e] = nan;
        }
        l[next() % static_cast<std::uint32_t>(ne)] = 0.5F;
      }
    }
    const std::size_t n_out = static_cast<std::size_t>(kPer) * nu;
    std::vector<int> rid(n_out);
    std::vector<float> rw(n_out);
    for (int v = 0; v < kPer; ++v) {
      QrhRefHead(lg.data() + static_cast<std::size_t>(v) * ne, ne, nu,
                 rid.data() + static_cast<std::size_t>(v) * nu,
                 rw.data() + static_cast<std::size_t>(v) * nu);
    }
    DeviceF32 d_l(lg.size());
    DeviceI32 d_ib(n_out);
    DeviceF32 d_wb(n_out);
    DeviceI32 d_i1(static_cast<std::size_t>(nu));
    DeviceF32 d_w1(static_cast<std::size_t>(nu));
    HIP_REQUIRE(hipMemcpy(d_l.p, lg.data(), lg.size() * sizeof(float),
                          hipMemcpyHostToDevice));
    Expect(LaunchQwenMoeRouteHeadForTest(d_l.p, kPer, ne, nu, d_ib.p, d_wb.p,
                                         nullptr) == 0,
           "qrh batch launch");
    HIP_REQUIRE(hipDeviceSynchronize());
    std::vector<int> bid(n_out);
    std::vector<float> bw(n_out);
    HIP_REQUIRE(hipMemcpy(bid.data(), d_ib.p, n_out * sizeof(int),
                          hipMemcpyDeviceToHost));
    HIP_REQUIRE(hipMemcpy(bw.data(), d_wb.p, n_out * sizeof(float),
                          hipMemcpyDeviceToHost));
    for (int v = 0; v < kPer; ++v) {
      const std::size_t o = static_cast<std::size_t>(v) * nu;
      Expect(LaunchQwenMoeRouteHeadForTest(
                 d_l.p + static_cast<std::size_t>(v) * ne, 0, ne, nu, d_i1.p,
                 d_w1.p, nullptr) == 0,
             "qrh single launch");
      HIP_REQUIRE(hipDeviceSynchronize());
      std::vector<int> sid(static_cast<std::size_t>(nu));
      std::vector<float> sw(static_cast<std::size_t>(nu));
      HIP_REQUIRE(hipMemcpy(sid.data(), d_i1.p, sid.size() * sizeof(int),
                            hipMemcpyDeviceToHost));
      HIP_REQUIRE(hipMemcpy(sw.data(), d_w1.p, sw.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
      bool ok = std::memcmp(sid.data(), bid.data() + o, sid.size() * sizeof(int)) == 0 &&
                std::memcmp(sw.data(), bw.data() + o, sw.size() * sizeof(float)) == 0;
      for (int r = 0; r < nu; ++r) {
        const float g = sw[static_cast<std::size_t>(r)];
        const float w = rw[o + static_cast<std::size_t>(r)];
        if (sid[static_cast<std::size_t>(r)] != rid[o + static_cast<std::size_t>(r)]) {
          ok = false;
        }
        if (w == 0.0F) {
          ok = ok && g == 0.0F;
        } else {
          const double rel = std::fabs(static_cast<double>(g) - w) / std::fabs(static_cast<double>(w));
          worst_rel = std::max(worst_rel, rel);
          ok = ok && rel <= 2.0e-6;
        }
      }
      if (!ok) {
        ++mismatched;
        if (mismatched <= 3) {
          std::cerr << "QRH mismatch e=" << ne << " used=" << nu << " v=" << v;
          for (int r = 0; r < nu; ++r) {
            std::cerr << " [" << sid[static_cast<std::size_t>(r)] << " "
                      << sw[static_cast<std::size_t>(r)] << " | ref "
                      << rid[o + static_cast<std::size_t>(r)] << " "
                      << rw[o + static_cast<std::size_t>(r)] << "]";
          }
          std::cerr << "\n";
        }
      }
      ++vectors;
    }
  }
  std::cout << "QRH head vs reference: " << vectors << " vectors, "
            << mismatched << " mismatched, worst_weight_rel=" << worst_rel
            << "\n";
  Expect(mismatched == 0, "qrh head vs reference");
}

// Super ROC QVB: LaunchQwenMoeRouteRows (1..4 rows, with and without the
// shared-expert gate) must match per-row LaunchQwenMoeRoute +
// LaunchQwenMoeShexpSigmoid bit for bit on real KAT layer-0 router weights.
void QvbRowsCases() {
  const char* path_env = std::getenv("KAT_GGUF");
  const std::string path =
      path_env != nullptr
          ? path_env
          : "/path/to/models/swap/kat-coder/"
            "KAT-Coder-V2.5-Dev-Q4_0_ROCMFP4_STRIX-MTP.gguf";
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  Expect(reader != nullptr, "qvb open: " + error);
  auto weights = gufo::models::QwenModelWeights::LoadFromGguf(*reader, &error);
  Expect(weights.has_value(), "qvb bind: " + error);
  const auto& gate = weights->layers[0].ffn_gate_inp;
  const auto& shexp = weights->layers[0].ffn_gate_inp_shexp;
  constexpr int k = 2048;
  constexpr int n_expert = 256;
  constexpr int n_used = 8;
  Expect(gate.type == gufo::core::GgmlType::kF32 &&
             gate.num_elements == static_cast<std::size_t>(k) * n_expert,
         "qvb router f32");
  Expect(shexp.type == gufo::core::GgmlType::kF32 &&
             shexp.num_elements == static_cast<std::size_t>(k),
         "qvb shexp gate f32");
  std::vector<float> g(gate.num_elements);
  std::memcpy(g.data(), gate.data, g.size() * sizeof(float));
  std::vector<float> sv(shexp.num_elements);
  std::memcpy(sv.data(), shexp.data, sv.size() * sizeof(float));
  constexpr int kRows = 4;
  std::vector<float> x(static_cast<std::size_t>(kRows) * k);
  std::uint32_t s = 0xC0FFEEu;
  for (float& v : x) {
    s = s * 1664525u + 1013904223u;
    v = static_cast<float>(static_cast<int>(s % 2001) - 1000) / 1000.0F;
  }
  DeviceF32 d_x(x.size());
  DeviceF32 d_gate(g.size());
  DeviceF32 d_shexp(sv.size());
  HIP_REQUIRE(hipMemcpy(d_x.p, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_REQUIRE(hipMemcpy(d_gate.p, g.data(), g.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_REQUIRE(hipMemcpy(d_shexp.p, sv.data(), sv.size() * sizeof(float), hipMemcpyHostToDevice));
  // Per-row reference.
  DeviceF32 d_lr(static_cast<std::size_t>(kRows) * n_expert);
  DeviceF32 d_wr(static_cast<std::size_t>(kRows) * n_used);
  DeviceI32 d_ir(static_cast<std::size_t>(kRows) * n_used);
  DeviceF32 d_sr(kRows);
  for (int r = 0; r < kRows; ++r) {
    Expect(LaunchQwenMoeRoute(d_x.p + static_cast<std::size_t>(r) * k, d_gate.p, k,
                              n_expert, n_used,
                              d_lr.p + static_cast<std::size_t>(r) * n_expert,
                              d_ir.p + static_cast<std::size_t>(r) * n_used,
                              d_wr.p + static_cast<std::size_t>(r) * n_used,
                              nullptr) == 0,
           "qvb ref route");
    Expect(LaunchQwenMoeShexpSigmoid(d_x.p + static_cast<std::size_t>(r) * k,
                                     d_shexp.p, k, 1, d_sr.p + r, nullptr) == 0,
           "qvb ref sigmoid");
  }
  HIP_REQUIRE(hipDeviceSynchronize());
  std::vector<float> lr(static_cast<std::size_t>(kRows) * n_expert);
  std::vector<float> wr(static_cast<std::size_t>(kRows) * n_used);
  std::vector<int> ir(static_cast<std::size_t>(kRows) * n_used);
  std::vector<float> sr(kRows);
  HIP_REQUIRE(hipMemcpy(lr.data(), d_lr.p, lr.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_REQUIRE(hipMemcpy(wr.data(), d_wr.p, wr.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_REQUIRE(hipMemcpy(ir.data(), d_ir.p, ir.size() * sizeof(int), hipMemcpyDeviceToHost));
  HIP_REQUIRE(hipMemcpy(sr.data(), d_sr.p, sr.size() * sizeof(float), hipMemcpyDeviceToHost));
  int mismatched = 0;
  int cases = 0;
  for (int rows = 1; rows <= kRows; ++rows) {
    for (const bool with_shexp : {true, false}) {
      DeviceF32 d_l(static_cast<std::size_t>(rows) * n_expert);
      DeviceF32 d_w(static_cast<std::size_t>(rows) * n_used);
      DeviceI32 d_i(static_cast<std::size_t>(rows) * n_used);
      DeviceF32 d_s(rows);
      Expect(LaunchQwenMoeRouteRows(d_x.p, d_gate.p, k, n_expert, rows, n_used,
                                    d_l.p, d_i.p, d_w.p,
                                    with_shexp ? d_shexp.p : nullptr,
                                    with_shexp ? d_s.p : nullptr, nullptr) == 0,
             "qvb rows launch");
      HIP_REQUIRE(hipDeviceSynchronize());
      std::vector<float> l(static_cast<std::size_t>(rows) * n_expert);
      std::vector<float> w(static_cast<std::size_t>(rows) * n_used);
      std::vector<int> id(static_cast<std::size_t>(rows) * n_used);
      std::vector<float> sg(rows);
      HIP_REQUIRE(hipMemcpy(l.data(), d_l.p, l.size() * sizeof(float), hipMemcpyDeviceToHost));
      HIP_REQUIRE(hipMemcpy(w.data(), d_w.p, w.size() * sizeof(float), hipMemcpyDeviceToHost));
      HIP_REQUIRE(hipMemcpy(id.data(), d_i.p, id.size() * sizeof(int), hipMemcpyDeviceToHost));
      bool ok = std::memcmp(l.data(), lr.data(), l.size() * sizeof(float)) == 0 &&
                std::memcmp(w.data(), wr.data(), w.size() * sizeof(float)) == 0 &&
                std::memcmp(id.data(), ir.data(), id.size() * sizeof(int)) == 0;
      if (with_shexp) {
        HIP_REQUIRE(hipMemcpy(sg.data(), d_s.p, sg.size() * sizeof(float), hipMemcpyDeviceToHost));
        ok = ok && std::memcmp(sg.data(), sr.data(), sg.size() * sizeof(float)) == 0;
      }
      if (!ok) {
        ++mismatched;
        std::cerr << "QVB mismatch rows=" << rows << " shexp=" << with_shexp << "\n";
      }
      ++cases;
    }
  }
  Expect(LaunchQwenMoeRouteRows(d_x.p, d_gate.p, k, n_expert, 5, n_used, d_lr.p,
                                d_ir.p, d_wr.p, nullptr, nullptr, nullptr) == 1,
         "qvb rows > 4 rejected");
  Expect(LaunchQwenMoeRouteRows(d_x.p, d_gate.p, k, n_expert, 1, n_used, d_lr.p,
                                d_ir.p, d_wr.p, d_shexp.p, nullptr, nullptr) == 1,
         "qvb shexp without sig rejected");
  std::cout << "QVB rows vs per-row: " << cases << " cases, " << mismatched
            << " mismatched\n";
  Expect(mismatched == 0, "qvb rows bit-identical");
}

}  // namespace

int main() {
  Expect(LaunchQwenMoeRoute(nullptr, nullptr, 32, 32, 8, nullptr, nullptr,
                            nullptr, nullptr) == 1,
         "null rejected before launch");
  Expect(LaunchQwenMoeRouteHeadForTest(nullptr, 0, 32, 8, nullptr, nullptr,
                                       nullptr) == 1,
         "qrh null rejected");
  QrhHeadCases();
  QvbRowsCases();

  constexpr int k_small = 32;
  constexpr int e_small = 32;
  std::vector<float> x_small(k_small);
  std::vector<float> g_small(static_cast<std::size_t>(e_small) * k_small);
  for (int i = 0; i < k_small; ++i) {
    x_small[static_cast<std::size_t>(i)] =
        static_cast<float>((i * 5) % 7) / 3.0F - 1.0F;
  }
  for (int e = 0; e < e_small; ++e) {
    for (int i = 0; i < k_small; ++i) {
      g_small[static_cast<std::size_t>(e) * k_small + i] =
          static_cast<float>((e * 17 + i * 3) % 100) / 50.0F - 1.0F;
    }
  }
  // KAT: exact tie on both sides — experts 5 and 20 get bitwise-identical gate
  // rows, so their logits tie within each side (GPU FMA and host mul+add each
  // stay self-consistent). The fork's bitonic DESC strict-compare rule must
  // resolve the tie to the same top-8 set as the CPU reference.
  for (int i = 0; i < k_small; ++i) {
    g_small[static_cast<std::size_t>(20) * k_small + i] =
        g_small[static_cast<std::size_t>(5) * k_small + i];
  }
  Compare(x_small, g_small, k_small, e_small, "synthetic_tie");

  const char* path_env = std::getenv("KAT_GGUF");
  const std::string path =
      path_env != nullptr
          ? path_env
          : "/path/to/models/swap/kat-coder/"
            "KAT-Coder-V2.5-Dev-Q4_0_ROCMFP4_STRIX-MTP.gguf";
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  Expect(reader != nullptr, "open: " + error);
  auto weights = gufo::models::QwenModelWeights::LoadFromGguf(*reader, &error);
  Expect(weights.has_value(), "bind: " + error);
  const auto& gate = weights->layers[0].ffn_gate_inp;
  Expect(gate.type == gufo::core::GgmlType::kF32, "router f32");
  constexpr int k = 2048;
  constexpr int n_expert = 256;
  Expect(gate.num_elements == static_cast<std::size_t>(k) * n_expert,
         "router numel");
  std::vector<float> x(k);
  std::uint32_t s = 0x51A7u;
  for (float& v : x) {
    s = s * 1664525u + 1013904223u;
    v = static_cast<float>(static_cast<int>(s % 2001) - 1000) / 1000.0F;
  }
  std::vector<float> g(gate.num_elements);
  std::memcpy(g.data(), gate.data, g.size() * sizeof(float));
  Compare(x, g, k, n_expert, "layer0");

  DeviceF32 d_x(x.size());
  DeviceF32 d_gate(g.size());
  DeviceF32 d_logits(n_expert);
  DeviceF32 d_w(8);
  DeviceI32 d_ids(8);
  HIP_REQUIRE(hipMemcpy(d_x.p, x.data(), x.size() * sizeof(float),
                        hipMemcpyHostToDevice));
  HIP_REQUIRE(hipMemcpy(d_gate.p, g.data(), g.size() * sizeof(float),
                        hipMemcpyHostToDevice));
  HIP_REQUIRE(LaunchQwenMoeRoute(d_x.p, d_gate.p, k, n_expert, 8, d_logits.p,
                                 d_ids.p, d_w.p, nullptr) == 0
                 ? hipSuccess
                 : hipErrorInvalidValue);
  HIP_REQUIRE(hipDeviceSynchronize());
  constexpr int k_warm = 3;
  constexpr int k_iters = 20;
  for (int i = 0; i < k_warm; ++i) {
    Expect(LaunchQwenMoeRoute(d_x.p, d_gate.p, k, n_expert, 8, d_logits.p,
                              d_ids.p, d_w.p, nullptr) == 0,
           "warm");
  }
  HIP_REQUIRE(hipDeviceSynchronize());
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  HIP_REQUIRE(hipEventCreate(&start));
  HIP_REQUIRE(hipEventCreate(&stop));
  HIP_REQUIRE(hipEventRecord(start, nullptr));
  for (int i = 0; i < k_iters; ++i) {
    Expect(LaunchQwenMoeRoute(d_x.p, d_gate.p, k, n_expert, 8, d_logits.p,
                              d_ids.p, d_w.p, nullptr) == 0,
           "bench");
  }
  HIP_REQUIRE(hipEventRecord(stop, nullptr));
  HIP_REQUIRE(hipEventSynchronize(stop));
  float ms = 0.0F;
  HIP_REQUIRE(hipEventElapsedTime(&ms, start, stop));
  const double us = (static_cast<double>(ms) * 1000.0) / k_iters;
  std::cout << "KAT_ROUTE microbench shape=k2048_e256_top8 layer=0 iters="
            << k_iters << " us_per_call=" << us << "\n";
  std::cout << "KAT_ROUTE PASS\n";
  const hipError_t e1 = hipEventDestroy(start);
  const hipError_t e2 = hipEventDestroy(stop);
  (void)e1;
  (void)e2;
  return 0;
}
