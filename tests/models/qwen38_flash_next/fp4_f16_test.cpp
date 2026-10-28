// LFP item 4 (+ SPEC2 item 6): (a) FP4->F16 decode exactness vs the
// superroc_fp4_elem CPU oracle (all 16 codes x all 256 scale bytes x types
// 100/101, bitwise + max_abs); (b) Fp4DenseF16Gemm vs an F64 reference on
// every dense (m, k) at 16/96/1024/2048 tokens, next to DenseF16Gemm/Q8_0's
// own error, plus bitwise determinism; 6(a) FP4->int8 code exactness;
// 6(b) Fp4Int8Gemm vs F64 on 320x10240 and 2560x6144; mixer F16-vs-F32
// equivalence. Run under gpu-run. (LANGUAGE HIP.)
#include "src/models/qwen38_flash_next/kernels/rocm/fp4_f16.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/rocm_fp4/superroc_fp4_quant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {
namespace rocm = superroc::models::qwen38_flash_next::rocm;
namespace fp4 = rocm::fp4_f16;

int failures = 0;

// 20:02 fault ruling: a dead context must FAIL the test. Every HIP call is
// checked and every launch is followed by hipGetLastError + a checked sync;
// any error aborts with the case name (never a tainted PASS).
const char* g_hip_case = "startup";
[[noreturn]] void LfpHipDie(hipError_t e, const char* what, int line) {
  std::printf("HIP FATAL [%s] %s:%d: %s (%d)\n", g_hip_case, what, line,
              hipGetErrorString(e), static_cast<int>(e));
  std::fflush(stdout);
  std::exit(1);
}
inline void LfpHipCheck(hipError_t e, const char* what, int line) {
  if (e != hipSuccess) {
    LfpHipDie(e, what, line);
  }
}
#define LFP_HIP_CHECK(x) LfpHipCheck((x), #x, __LINE__)
#define LFP_LAUNCHED() \
  LfpHipCheck(hipGetLastError(), "hipGetLastError", __LINE__)
void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAIL %s\n", what.c_str());
    ++failures;
  }
}

std::uint64_t rng = 0xFEDCBA987654321ULL;
std::uint64_t Next() {
  rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return rng >> 11;
}

// One thread decodes one block with the exact device function the kernel uses.
__global__ void DecodeCheckKernel(const std::uint8_t* blocks,
                                  std::uint16_t* out, int n_blocks,
                                  int stride) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n_blocks) {
    return;
  }
  const std::uint8_t* blk = blocks + static_cast<std::size_t>(i) * stride;
  std::uint32_t code4[4];
  __builtin_memcpy(code4, blk, 16);
  __half2 h[16];
  const bool fast = stride == 17;
  fp4::Fp4DecodeBlock(code4, blk[16], fast ? blk[16] : blk[17], h);
  __builtin_memcpy(out + static_cast<std::size_t>(i) * 32, h, 64);
}

// 4(a): every nibble value in both halves, every scale byte, both types.
// Type 100 runs the full 256x256 (s0, s1) cross product.
void DecodeExactness() {
  g_hip_case = __func__;
  for (int type = 100; type <= 101; ++type) {
    const bool fast = type == 101;
    const int stride = fast ? 17 : 18;
    const int n_blocks = fast ? 256 : 65536;
    std::vector<std::uint8_t> blocks(static_cast<std::size_t>(n_blocks) *
                                     stride);
    for (int b = 0; b < n_blocks; ++b) {
      std::uint8_t* blk = &blocks[static_cast<std::size_t>(b) * stride];
      for (int i = 0; i < 16; ++i) {
        blk[i] = static_cast<std::uint8_t>(i | ((15 - i) << 4));
      }
      if (fast) {
        blk[16] = static_cast<std::uint8_t>(b);
      } else {
        blk[16] = static_cast<std::uint8_t>(b & 0xFF);
        blk[17] = static_cast<std::uint8_t>((b >> 8) & 0xFF);
      }
    }
    std::uint8_t* d_blocks = nullptr;
    std::uint16_t* d_out = nullptr;
    LFP_HIP_CHECK(hipMalloc(&d_blocks, blocks.size()));
    LFP_HIP_CHECK(
        hipMalloc(&d_out, static_cast<std::size_t>(n_blocks) * 32 * 2));
    LFP_HIP_CHECK(hipMemcpy(d_blocks, blocks.data(), blocks.size(),
                            hipMemcpyHostToDevice));
    const int threads = 256;
    DecodeCheckKernel<<<(n_blocks + threads - 1) / threads, threads, 0,
                        nullptr>>>(d_blocks, d_out, n_blocks, stride);
    LFP_LAUNCHED();
    LFP_HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint16_t> out(static_cast<std::size_t>(n_blocks) * 32);
    LFP_HIP_CHECK(
        hipMemcpy(out.data(), d_out, out.size() * 2, hipMemcpyDeviceToHost));
    double max_abs = 0.0;
    long bit_bad = 0;
    for (int b = 0; b < n_blocks; ++b) {
      const std::uint8_t* blk = &blocks[static_cast<std::size_t>(b) * stride];
      for (int j = 0; j < 32; ++j) {
        const float ref = superroc_fp4_elem(type, blk, j);
        const std::uint16_t got = out[static_cast<std::size_t>(b) * 32 + j];
        __half gh;
        __builtin_memcpy(&gh, &got, 2);
        const double d = std::fabs(static_cast<double>(__half2float(gh)) - ref);
        if (d > max_abs) {
          max_abs = d;
        }
        const __half rh = __float2half_rn(ref);
        std::uint16_t want = 0;
        __builtin_memcpy(&want, &rh, 2);
        // Signed zeros: the oracle's F32 (-x)*(+0) is -0, while the
        // kernel's hfma(code, scale, +0) normalizes every signed zero to
        // +0 (including code 8). Value-equal (|+0 - -0| == 0) and harmless
        // to every gate (comparisons, exp/log, argmax ties). Only a
        // non-sign-zero diff may fail.
        const bool both_zero = (got & 0x7FFF) == 0 && (want & 0x7FFF) == 0;
        if (got != want && !both_zero) {
          ++bit_bad;
        }
      }
    }
    std::printf("4a type=%d blocks=%d max_abs=%.3g bitwise_bad=%ld\n", type,
                n_blocks, max_abs, bit_bad);
    Check(max_abs == 0.0, "4a max_abs != 0 type " + std::to_string(type));
    Check(bit_bad == 0, "4a bitwise mismatch type " + std::to_string(type));
    LFP_HIP_CHECK(hipFree(d_blocks));
    LFP_HIP_CHECK(hipFree(d_out));
  }
  // The host scale reference matches the oracle on all 256 bytes (value).
  for (int e = 0; e < 256; ++e) {
    __half h;
    const std::uint16_t bits =
        fp4::Fp4ScaleByteToHalf(static_cast<std::uint8_t>(e));
    __builtin_memcpy(&h, &bits, 2);
    const float ref = superroc_fp4_ue4m3_half(static_cast<std::uint8_t>(e));
    if (static_cast<double>(__half2float(h)) != static_cast<double>(ref)) {
      Check(false, "host scale table mismatch e=" + std::to_string(e));
      break;
    }
  }
}

struct Dev {
  void* p = nullptr;
  explicit Dev(std::size_t n) { LFP_HIP_CHECK(hipMalloc(&p, n ? n : 1)); }
  ~Dev() { LFP_HIP_CHECK(hipFree(p)); }
};

std::vector<std::uint8_t> MakeFp4(std::size_t m, std::size_t k, bool fast) {
  const std::size_t nb = k / 32;
  const std::size_t bb = fast ? 17 : 18;
  std::vector<std::uint8_t> w(m * nb * bb);
  for (std::size_t i = 0; i < w.size(); ++i) {
    w[i] = static_cast<std::uint8_t>(Next() & 0xFF);
  }
  return w;
}

std::vector<std::uint8_t> MakeQ80(std::size_t m, std::size_t k) {
  const std::size_t nb = k / 32;
  std::vector<std::uint8_t> w(m * nb * 34);
  for (std::size_t r = 0; r < m; ++r) {
    for (std::size_t b = 0; b < nb; ++b) {
      std::uint8_t* blk = &w[(r * nb + b) * 34];
      const __half d = __float2half(0.02f);
      __builtin_memcpy(blk, &d, 2);
      for (int i = 0; i < 32; ++i) {
        blk[2 + i] = static_cast<std::uint8_t>(Next() & 0xFF);
      }
    }
  }
  return w;
}

// F64 dot of one (token, row) over K-ascending exact dequantized weights.
double RefDot(const std::uint8_t* wrow, int qtype, const __half* xrow,
              std::size_t k) {
  double acc = 0.0;
  for (std::size_t i = 0; i < k; ++i) {
    const double wv = superroc_fp4_elem(qtype, wrow, static_cast<int>(i));
    __half h = xrow[i];
    acc += wv * static_cast<double>(__half2float(h));
  }
  return acc;
}

double RefDotQ80(const std::uint8_t* wrow, const __half* xrow, std::size_t k) {
  double acc = 0.0;
  for (std::size_t i = 0; i < k; ++i) {
    const std::uint8_t* blk = wrow + (i / 32) * 34;
    __half d;
    __builtin_memcpy(&d, blk, 2);
    const auto q = static_cast<const std::int8_t*>(
        static_cast<const void*>(blk + 2))[i % 32];
    __half h = xrow[i];
    acc += static_cast<double>(__half2float(d)) * q *
           static_cast<double>(__half2float(h));
  }
  return acc;
}

void GemmCase(std::size_t batch, std::size_t m, std::size_t k, bool fast) {
  g_hip_case = __func__;
  const std::size_t bb = fast ? 17 : 18;
  const auto w = MakeFp4(m, k, fast);
  const auto wq = MakeQ80(m, k);
  std::vector<__half> xh(batch * k);
  for (__half& v : xh) {
    v = __float2half(static_cast<float>(Next() & 0xFFFF) / 65535.f - 0.5f);
  }
  Dev d_w(w.size()), d_wq(wq.size()), d_x(xh.size() * 2), d_o1(batch * m * 4),
      d_o2(batch * m * 4), d_oq(batch * m * 4);
  LFP_HIP_CHECK(hipMemcpy(d_w.p, w.data(), w.size(), hipMemcpyHostToDevice));
  LFP_HIP_CHECK(hipMemcpy(d_wq.p, wq.data(), wq.size(), hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_x.p, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
  hipStream_t st = nullptr;
  LFP_HIP_CHECK(hipStreamCreate(&st));
  const bool ok1 =
      rocm::Fp4DenseF16Gemm(d_w.p, static_cast<const __half*>(d_x.p),
                            static_cast<float*>(d_o1.p), batch, m, k, fast, st);
  LFP_LAUNCHED();
  const bool ok2 =
      rocm::Fp4DenseF16Gemm(d_w.p, static_cast<const __half*>(d_x.p),
                            static_cast<float*>(d_o2.p), batch, m, k, fast, st);
  LFP_LAUNCHED();
  const bool okq =
      rocm::DenseF16Gemm(d_wq.p, static_cast<const __half*>(d_x.p),
                         static_cast<float*>(d_oq.p), batch, m, k, st);
  LFP_LAUNCHED();
  LFP_HIP_CHECK(hipStreamSynchronize(st));
  Check(ok1 && ok2 && okq, "launcher returned false");
  std::vector<float> o1(batch * m), o2(batch * m), oq(batch * m);
  LFP_HIP_CHECK(
      hipMemcpy(o1.data(), d_o1.p, o1.size() * 4, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(
      hipMemcpy(o2.data(), d_o2.p, o2.size() * 4, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(
      hipMemcpy(oq.data(), d_oq.p, oq.size() * 4, hipMemcpyDeviceToHost));
  Check(std::memcmp(o1.data(), o2.data(), o1.size() * 4) == 0,
        "nondeterministic GEMM");
  // Sampled F64 reference: all rows/tokens when small, else strided.
  const std::size_t n_rt = m <= 640 ? m : 256;
  const std::size_t n_tt = batch <= 96 ? batch : 64;
  double max_abs = 0.0, max_abs_q = 0.0, dot = 0.0, n1 = 0.0, n2 = 0.0,
         dot_q = 0.0, n1_q = 0.0, n2_q = 0.0, max_ref = 0.0;
  bool finite = true;
  for (float v : o1) {
    if (!std::isfinite(v)) {
      finite = false;
    }
  }
  Check(finite, "non-finite output");
  for (std::size_t a = 0; a < n_rt; ++a) {
    const std::size_t r = m <= 640 ? a : a * m / n_rt;
    const std::uint8_t* wrow = &w[r * (k / 32) * bb];
    const std::uint8_t* wqrow = &wq[r * (k / 32) * 34];
    for (std::size_t b = 0; b < n_tt; ++b) {
      const std::size_t t = batch <= 96 ? b : b * batch / n_tt;
      const double ref = RefDot(wrow, fast ? 101 : 100, &xh[t * k], k);
      const double got = o1[t * m + r];
      const double refq = RefDotQ80(wqrow, &xh[t * k], k);
      const double gotq = oq[t * m + r];
      max_abs = std::max(max_abs, std::fabs(got - ref));
      max_ref = std::max(max_ref, std::fabs(ref));
      max_abs_q = std::max(max_abs_q, std::fabs(gotq - refq));
      dot += got * ref;
      n1 += got * got;
      n2 += ref * ref;
      dot_q += gotq * refq;
      n1_q += gotq * gotq;
      n2_q += refq * refq;
    }
  }
  const double cos = dot / std::sqrt(n1 * n2);
  const double cos_q = dot_q / std::sqrt(n1_q * n2_q);
  std::printf(
      "4b %s batch=%4zu m=%5zu k=%5zu max_abs=%.3g (%.1e of %.3g) cos=%.9f | "
      "Q8_0 max_abs=%.3g cos=%.9f\n",
      fast ? "101" : "100", batch, m, k, max_abs, max_abs / max_ref, max_ref,
      cos, max_abs_q, cos_q);
  Check(cos > 0.9999, "low cosine");
  LFP_HIP_CHECK(hipStreamDestroy(st));
}

// One thread decodes one nibble dword with the int8 device function.
__global__ void Int8CheckKernel(const std::uint32_t* nibs, std::uint32_t* out,
                                int n) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) {
    return;
  }
  fp4::Fp4NibblesToInt8(nibs[i], out[i]);
}

// 6(a): int8 code bytes match superroc_fp4_rocmfp4_code on every nibble, in
// every lane, with garbage upper bits set (the contract ignores them).
void Int8Exactness() {
  g_hip_case = __func__;
  std::vector<std::uint32_t> nibs;
  for (int b = 0; b < 256; ++b) {
    nibs.push_back(static_cast<std::uint32_t>(b * 0x01010101u));
  }
  for (int r = 0; r < 256; ++r) {
    nibs.push_back(static_cast<std::uint32_t>(Next() | 0xF0F0F0F0u));
  }
  const int n = static_cast<int>(nibs.size());
  Dev d_n(n * 4), d_o(n * 4);
  LFP_HIP_CHECK(hipMemcpy(d_n.p, nibs.data(), n * 4, hipMemcpyHostToDevice));
  const int threads = 256;
  Int8CheckKernel<<<(n + threads - 1) / threads, threads, 0, nullptr>>>(
      static_cast<const std::uint32_t*>(d_n.p),
      static_cast<std::uint32_t*>(d_o.p), n);
  LFP_LAUNCHED();
  LFP_HIP_CHECK(hipDeviceSynchronize());
  std::vector<std::uint32_t> out(n);
  LFP_HIP_CHECK(hipMemcpy(out.data(), d_o.p, n * 4, hipMemcpyDeviceToHost));
  long bad = 0;
  for (int i = 0; i < n; ++i) {
    for (int lane = 0; lane < 4; ++lane) {
      const int nib = (nibs[i] >> (lane * 8)) & 0xF;
      const auto got = static_cast<std::int8_t>((out[i] >> (lane * 8)) & 0xFF);
      const int want =
          superroc_fp4_rocmfp4_code(static_cast<std::uint32_t>(nib));
      if (got != want) {
        ++bad;
      }
    }
  }
  std::printf("6a nibbles=%d bad=%ld\n", n * 4, bad);
  Check(bad == 0, "6a int8 code mismatch");
  // The host float-scale twin matches the oracle on all 256 bytes, exactly.
  for (int e = 0; e < 256; ++e) {
    const float got = fp4::Fp4ScaleByteToFloat(static_cast<std::uint8_t>(e));
    const float ref = superroc_fp4_ue4m3_half(static_cast<std::uint8_t>(e));
    if (got != ref) {
      Check(false, "host float scale mismatch e=" + std::to_string(e));
      break;
    }
  }
}

// Tiled-Q8 host parse (mirrors QuantizeQ8TiledKernel's layout: 576-byte
// tiles, two 256-byte code halves of 16 tokens x 16 codes, 16 float
// scales at byte 512).
float TiledQ8At(const std::uint8_t* tiled, std::size_t k, std::size_t t,
                std::size_t i) {
  const std::size_t nb = k / 32;
  const std::size_t kb = i / 32, lane = i % 32;
  const std::size_t half = lane >> 4, pos = lane & 15;
  const std::size_t tt = t / 16, tl = t % 16;
  const std::uint8_t* tile = tiled + ((tt * nb + kb) * 576);
  const auto q = static_cast<std::int8_t>(tile[half * 256 + tl * 16 + pos]);
  float d = 0;
  __builtin_memcpy(&d, tile + 512 + tl * 4, 4);
  return q * d;
}

// 6(b): Fp4Int8Gemm vs F64 references (FAST only): against the QUANTIZED
// activations (isolates kernel error: int32-exact math plus F32 scale
// products) and against the float activations (total path error).
void Int8GemmCase(std::size_t batch, std::size_t m, std::size_t k) {
  g_hip_case = __func__;
  // Realistic scales (exp <= 3): edge scales would overflow downstream
  // F16 products; the kernel math is scale-agnostic and 6(a) covers the
  // full scale range exactly.
  auto w = MakeFp4(m, k, true);
  const std::size_t nb = k / 32;
  for (std::size_t r = 0; r < m; ++r) {
    for (std::size_t b = 0; b < nb; ++b) {
      w[(r * nb + b) * 17 + 16] &= 0x1F;
    }
  }
  std::vector<float> xf(batch * k);
  for (float& v : xf) {
    v = static_cast<float>(Next() & 0xFFFF) / 65535.f - 0.5f;
  }
  const std::size_t qb = rocm::Q8TiledBytes(batch, k);
  Dev d_w(w.size()), d_x(xf.size() * 4), d_q(qb), d_o1(batch * m * 4),
      d_o2(batch * m * 4);
  LFP_HIP_CHECK(hipMemcpy(d_w.p, w.data(), w.size(), hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_x.p, xf.data(), xf.size() * 4, hipMemcpyHostToDevice));
  hipStream_t st = nullptr;
  LFP_HIP_CHECK(hipStreamCreate(&st));
  rocm::QuantizeQ8Tiled(static_cast<const float*>(d_x.p), d_q.p, batch, k, st);
  LFP_LAUNCHED();
  const bool ok1 = rocm::Fp4Int8Gemm(d_w.p, d_q.p, static_cast<float*>(d_o1.p),
                                     batch, m, k, true, st);
  LFP_LAUNCHED();
  const bool ok2 = rocm::Fp4Int8Gemm(d_w.p, d_q.p, static_cast<float*>(d_o2.p),
                                     batch, m, k, true, st);
  LFP_LAUNCHED();
  LFP_HIP_CHECK(hipStreamSynchronize(st));
  Check(ok1 && ok2, "int8 launcher returned false");
  std::vector<float> o1(batch * m), o2(batch * m);
  std::vector<std::uint8_t> q(qb);
  LFP_HIP_CHECK(
      hipMemcpy(o1.data(), d_o1.p, o1.size() * 4, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(
      hipMemcpy(o2.data(), d_o2.p, o2.size() * 4, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(hipMemcpy(q.data(), d_q.p, qb, hipMemcpyDeviceToHost));
  Check(std::memcmp(o1.data(), o2.data(), o1.size() * 4) == 0,
        "nondeterministic int8 GEMM");
  bool finite = true;
  for (float v : o1) {
    if (!std::isfinite(v)) {
      finite = false;
    }
  }
  Check(finite, "non-finite int8 output");
  const std::size_t n_rt = m <= 640 ? m : 64;
  const std::size_t n_tt = batch <= 96 ? batch : 32;
  double max_k = 0.0, max_ref = 0.0, dot = 0.0, n1 = 0.0, n2 = 0.0;
  double max_t = 0.0, dott = 0.0, n1t = 0.0, n2t = 0.0;
  for (std::size_t a = 0; a < n_rt; ++a) {
    const std::size_t r = m <= 640 ? a : a * m / n_rt;
    const std::uint8_t* wrow = &w[r * (k / 32) * 17];
    for (std::size_t b = 0; b < n_tt; ++b) {
      const std::size_t t = batch <= 96 ? b : b * batch / n_tt;
      double ref = 0.0, reft = 0.0;
      for (std::size_t i = 0; i < k; ++i) {
        const double wv = superroc_fp4_elem(101, wrow, static_cast<int>(i));
        ref += wv * TiledQ8At(q.data(), k, t, i);
        reft += wv * xf[t * k + i];
      }
      const double got = o1[t * m + r];
      max_k = std::max(max_k, std::fabs(got - ref));
      max_ref = std::max(max_ref, std::fabs(ref));
      dot += got * ref;
      n1 += got * got;
      n2 += ref * ref;
      max_t = std::max(max_t, std::fabs(got - reft));
      dott += got * reft;
      n1t += got * got;
      n2t += reft * reft;
    }
  }
  const double cos = dot / std::sqrt(n1 * n2);
  const double cost = dott / std::sqrt(n1t * n2t);
  std::printf(
      "6b batch=%4zu m=%5zu k=%5zu kernel rel=%.2e cos=%.9f | total rel=%.2e "
      "cos=%.9f\n",
      batch, m, k, max_k / max_ref, cos, max_t / max_ref, cost);
  Check(max_k / max_ref < 2e-6 && cos > 0.999999, "int8 kernel error");
  Check(max_t / max_ref < 2e-2 && cost > 0.999, "int8 total error");
  LFP_HIP_CHECK(hipStreamDestroy(st));
}

// SPEC2 item 1 proof: the F16 mixer route against the F32 route on
// identical inputs. Both parts must agree within F16 rounding.
void MixerEquiv() {
  g_hip_case = __func__;
  const std::uint32_t tok = 128, hid = 2560, used = 10, streams = 4;
  const std::uint32_t hcd = hid * streams;
  const float eps = 1e-5f;
  std::vector<__half> eh(static_cast<std::size_t>(tok) * used * hid);
  for (__half& v : eh) {
    v = __float2half(static_cast<float>(Next() % 2000) / 1000.f - 1.f);
  }
  std::vector<float> wt(static_cast<std::size_t>(tok) * used);
  for (float& v : wt) {
    v = static_cast<float>(Next() % 1000) / 1000.f * 0.3f;
  }
  std::vector<float> sh(static_cast<std::size_t>(tok) * hid);
  for (float& v : sh) {
    v = static_cast<float>(Next() % 4000) / 1000.f - 2.f;
  }
  std::vector<float> gate(static_cast<std::size_t>(tok) * 513);
  for (float& v : gate) {
    v = static_cast<float>(Next() % 2000) / 1000.f - 1.f;
  }
  std::vector<float> inj(static_cast<std::size_t>(tok) * streams);
  for (float& v : inj) {
    v = static_cast<float>(Next() % 2000) / 1000.f - 1.f;
  }
  std::vector<float> gamma(hcd);
  for (float& v : gamma) {
    v = 0.9f + static_cast<float>(Next() % 1000) / 1000.f * 0.2f;
  }
  std::vector<float> res(static_cast<std::size_t>(tok) * hcd);
  for (float& v : res) {
    v = static_cast<float>(Next() % 2000) / 1000.f - 1.f;
  }
  const std::size_t xnqb = rocm::Q8TiledBytes(tok, hcd);
  Dev d_eh(eh.size() * 2), d_wt(wt.size() * 4), d_sh(sh.size() * 4),
      d_g(gate.size() * 4), d_in(inj.size() * 4), d_ga(gamma.size() * 4),
      d_rA(res.size() * 4), d_rB(res.size() * 4), d_xnh(res.size() * 2),
      d_xnq(xnqb), d_bo(sh.size() * 4), d_xnf(res.size() * 4);
  LFP_HIP_CHECK(
      hipMemcpy(d_eh.p, eh.data(), eh.size() * 2, hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_wt.p, wt.data(), wt.size() * 4, hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_sh.p, sh.data(), sh.size() * 4, hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_g.p, gate.data(), gate.size() * 4, hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_in.p, inj.data(), inj.size() * 4, hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_ga.p, gamma.data(), gamma.size() * 4, hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_rA.p, res.data(), res.size() * 4, hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_rB.p, res.data(), res.size() * 4, hipMemcpyHostToDevice));
  hipStream_t st = nullptr;
  LFP_HIP_CHECK(hipStreamCreate(&st));
  const bool ok = rocm::HcCombineMoeF16(
      static_cast<float*>(d_rA.p), static_cast<const __half*>(d_eh.p),
      static_cast<const float*>(d_wt.p), static_cast<const float*>(d_sh.p),
      static_cast<const float*>(d_g.p), 513, used,
      static_cast<const float*>(d_in.p), 1, static_cast<const float*>(d_ga.p),
      static_cast<__half*>(d_xnh.p), d_xnq.p, tok, hid, streams, eps, st);
  LFP_LAUNCHED();
  Check(ok, "moe-f16 geometry declined");
  rocm::MoeEpilogueVec4F16(
      static_cast<const __half*>(d_eh.p), static_cast<const float*>(d_wt.p),
      static_cast<const float*>(d_sh.p), static_cast<const float*>(d_g.p), 513,
      static_cast<float*>(d_bo.p), tok, used, hid, st);
  LFP_LAUNCHED();
  rocm::HcCombine(
      static_cast<float*>(d_rB.p), static_cast<const float*>(d_bo.p),
      static_cast<const float*>(d_in.p), 1, static_cast<const float*>(d_ga.p),
      static_cast<float*>(d_xnf.p), tok, hid, streams, eps, st);
  LFP_LAUNCHED();
  LFP_HIP_CHECK(hipStreamSynchronize(st));
  std::vector<float> rA(res.size()), rB(res.size()), xnf(res.size());
  std::vector<__half> xnh(res.size());
  LFP_HIP_CHECK(
      hipMemcpy(rA.data(), d_rA.p, rA.size() * 4, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(
      hipMemcpy(rB.data(), d_rB.p, rB.size() * 4, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(
      hipMemcpy(xnh.data(), d_xnh.p, xnh.size() * 2, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(
      hipMemcpy(xnf.data(), d_xnf.p, xnf.size() * 4, hipMemcpyDeviceToHost));
  double max_r = 0.0, ref_r = 0.0, max_x = 0.0, ref_x = 0.0;
  for (std::size_t i = 0; i < res.size(); ++i) {
    max_r = std::max(max_r, std::fabs(static_cast<double>(rA[i] - rB[i])));
    ref_r = std::max(ref_r, std::fabs(static_cast<double>(rB[i])));
    const double xh = __half2float(xnh[i]);
    max_x = std::max(max_x, std::fabs(xh - xnf[i]));
    ref_x = std::max(ref_x, std::fabs(static_cast<double>(xnf[i])));
  }
  std::printf("mixer combine res rel=%.2e xn rel=%.2e\n", max_r / ref_r,
              max_x / ref_x);
  Check(max_r / ref_r < 2e-3, "mixer res mismatch");
  Check(max_x / ref_x < 2e-3, "mixer xn mismatch");
  // Mix epilogue: F16-xn input vs F32-xn input (same gate, no inject/Q8).
  std::vector<float> xnf2(res.size()), gf(res.size());
  for (float& v : xnf2) {
    v = static_cast<float>(Next() % 4000) / 1000.f - 2.f;
  }
  for (float& v : gf) {
    v = static_cast<float>(Next() % 4000) / 1000.f - 2.f;
  }
  std::vector<__half> xnh2(res.size());
  for (std::size_t i = 0; i < res.size(); ++i) {
    xnh2[i] = __float2half(xnf2[i]);
  }
  Dev d_xf(xnf2.size() * 4), d_xh(xnh2.size() * 2), d_gf(gf.size() * 4),
      d_mA(sh.size() * 4), d_mB(sh.size() * 4);
  LFP_HIP_CHECK(
      hipMemcpy(d_xf.p, xnf2.data(), xnf2.size() * 4, hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_xh.p, xnh2.data(), xnh2.size() * 2, hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_gf.p, gf.data(), gf.size() * 4, hipMemcpyHostToDevice));
  rocm::HcMixEpilogueVec4F16(static_cast<const __half*>(d_xh.p),
                             static_cast<const float*>(d_gf.p), nullptr,
                             static_cast<float*>(d_mA.p), nullptr, nullptr,
                             nullptr, tok, hid, st);
  LFP_LAUNCHED();
  rocm::HcMixEpilogueVec4(
      static_cast<const float*>(d_xf.p), static_cast<const float*>(d_gf.p),
      nullptr, static_cast<float*>(d_mB.p), nullptr, tok, hid, streams, st);
  LFP_LAUNCHED();
  LFP_HIP_CHECK(hipStreamSynchronize(st));
  std::vector<float> mA(sh.size()), mB(sh.size());
  LFP_HIP_CHECK(
      hipMemcpy(mA.data(), d_mA.p, mA.size() * 4, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(
      hipMemcpy(mB.data(), d_mB.p, mB.size() * 4, hipMemcpyDeviceToHost));
  double max_m = 0.0, ref_m = 0.0;
  for (std::size_t i = 0; i < sh.size(); ++i) {
    max_m = std::max(max_m, std::fabs(static_cast<double>(mA[i] - mB[i])));
    ref_m = std::max(ref_m, std::fabs(static_cast<double>(mB[i])));
  }
  std::printf("mixer epilogue mixed rel=%.2e\n", max_m / ref_m);
  Check(max_m / ref_m < 2e-3, "mixer epilogue mismatch");
  LFP_HIP_CHECK(hipStreamDestroy(st));
}

// Host twin of SsmConv4Value (same fmaf order; silu matches to ~1ulp).
float HostSsmConv(const float* w, float x0, float x1, float x2, float x3) {
  float acc = std::fmaf(w[0], x0, w[1] * x1);
  acc = std::fmaf(w[2], x2, acc);
  acc = std::fmaf(w[3], x3, acc);
  return acc / (1.0f + std::exp(-acc));
}

// SPEC2 item 1b(a): Fp4SsmF16Gemm vs two separate Fp4DenseF16Gemm runs
// (raw rows must be BITWISE identical: same tile, same K order) and the
// convolved rows vs a host 4-tap reference (silu/exp to ~1ulp).
void SsmFusedCase(std::size_t batch) {
  g_hip_case = __func__;
  const std::size_t k = 2560, qkv = 10240, gate = 6144, m = 16384;
  auto wq = MakeFp4(qkv, k, false);
  auto wg = MakeFp4(gate, k, true);
  std::vector<__half> xh(batch * k);
  for (__half& v : xh) {
    v = __float2half(static_cast<float>(Next() & 0xFFFF) / 65535.f - 0.5f);
  }
  std::vector<float> convw(qkv * 4), hist(3 * qkv);
  for (float& v : convw) {
    v = static_cast<float>(Next() & 0xFFFF) / 65535.f - 0.5f;
  }
  for (float& v : hist) {
    v = static_cast<float>(Next() & 0xFFFF) / 65535.f - 0.5f;
  }
  Dev d_wq(wq.size()), d_wg(wg.size()), d_x(xh.size() * 2),
      d_cw(convw.size() * 4), d_hi(hist.size() * 4), d_qkvz(batch * m * 4),
      d_conv(batch * qkv * 4), d_raw1(batch * qkv * 4),
      d_raw2(batch * gate * 4), d_qkvz2(batch * m * 4),
      d_conv2(batch * qkv * 4);
  LFP_HIP_CHECK(hipMemcpy(d_wq.p, wq.data(), wq.size(), hipMemcpyHostToDevice));
  LFP_HIP_CHECK(hipMemcpy(d_wg.p, wg.data(), wg.size(), hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_x.p, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_cw.p, convw.data(), convw.size() * 4, hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_hi.p, hist.data(), hist.size() * 4, hipMemcpyHostToDevice));
  hipStream_t st = nullptr;
  LFP_HIP_CHECK(hipStreamCreate(&st));
  const bool ok = rocm::Fp4SsmF16Gemm(
      d_wq.p, d_wg.p, static_cast<const __half*>(d_x.p),
      static_cast<const float*>(d_cw.p), static_cast<const float*>(d_hi.p),
      static_cast<float*>(d_qkvz.p), static_cast<float*>(d_conv.p),
      static_cast<std::uint32_t>(batch), static_cast<std::uint32_t>(m),
      static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(qkv), 4, st);
  LFP_LAUNCHED();
  Check(ok, "ssm launcher returned false");
  const bool ok1 = rocm::Fp4DenseF16Gemm(
      d_wq.p, static_cast<const __half*>(d_x.p), static_cast<float*>(d_raw1.p),
      batch, qkv, k, false, st);
  LFP_LAUNCHED();
  const bool ok2 = rocm::Fp4DenseF16Gemm(
      d_wg.p, static_cast<const __half*>(d_x.p), static_cast<float*>(d_raw2.p),
      batch, gate, k, true, st);
  LFP_LAUNCHED();
  Check(ok1 && ok2, "ssm ref launchers returned false");
  const bool okr = rocm::Fp4SsmF16Gemm(
      d_wq.p, d_wg.p, static_cast<const __half*>(d_x.p),
      static_cast<const float*>(d_cw.p), static_cast<const float*>(d_hi.p),
      static_cast<float*>(d_qkvz2.p), static_cast<float*>(d_conv2.p),
      static_cast<std::uint32_t>(batch), static_cast<std::uint32_t>(m),
      static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(qkv), 4, st);
  LFP_LAUNCHED();
  Check(okr, "ssm replay failed");
  LFP_HIP_CHECK(hipStreamSynchronize(st));
  std::vector<float> qkvz(batch * m), conv(batch * qkv), raw1(batch * qkv),
      raw2(batch * gate), qkvz2(batch * m), conv2(batch * qkv);
  LFP_HIP_CHECK(
      hipMemcpy(qkvz.data(), d_qkvz.p, qkvz.size() * 4, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(
      hipMemcpy(conv.data(), d_conv.p, conv.size() * 4, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(
      hipMemcpy(raw1.data(), d_raw1.p, raw1.size() * 4, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(
      hipMemcpy(raw2.data(), d_raw2.p, raw2.size() * 4, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(hipMemcpy(qkvz2.data(), d_qkvz2.p, qkvz2.size() * 4,
                          hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(hipMemcpy(conv2.data(), d_conv2.p, conv2.size() * 4,
                          hipMemcpyDeviceToHost));
  Check(std::memcmp(qkvz.data(), qkvz2.data(), qkvz.size() * 4) == 0,
        "nondeterministic ssm raw");
  Check(std::memcmp(conv.data(), conv2.data(), conv.size() * 4) == 0,
        "nondeterministic ssm conv");
  // Raw rows: gate rows are always written; qkv rows only on tile
  // boundaries (tok_l < 3 or >= 29) and the batch tail (tok + 3 >= batch).
  // Interior qkv raw rows are NEVER written — stock's contract (the
  // convolved path reads them from conv_out instead) — so they are
  // excluded here. tok_l == t % 32: t0 is a multiple of 32 (t_block is
  // 128-wide, j steps by 2) and tok = t0 + tok_l with tok_l < 32.
  long raw_bad = 0;
  for (std::size_t t = 0; t < batch; ++t) {
    if (std::memcmp(&qkvz[t * m + qkv], &raw2[t * gate], gate * 4) != 0) {
      ++raw_bad;
    }
    const std::size_t tok_l = t % 32;
    if (tok_l < 3 || tok_l >= 29 || t + 3 >= batch) {
      if (std::memcmp(&qkvz[t * m], &raw1[t * qkv], qkv * 4) != 0) {
        ++raw_bad;
      }
    }
  }
  std::printf("ssm batch=%4zu raw bitwise_bad=%ld\n", batch, raw_bad);
  Check(raw_bad == 0, "ssm raw mismatch");
  // Convolved rows vs the host 4-tap reference (all tokens x all channels
  // would take minutes; stride to ~2M cells).
  double max_abs = 0.0, max_ref = 0.0;
  const std::size_t t_step = 1, c_step = 5;
  for (std::size_t t = 0; t < batch; t += t_step) {
    for (std::size_t c = 0; c < qkv; c += c_step) {
      float v[4];
      for (int j = 0; j < 4; ++j) {
        const long src = static_cast<long>(t) - 3 + j;
        v[j] = src < 0 ? hist[(src + 3) * qkv + c] : raw1[src * qkv + c];
      }
      const float ref = HostSsmConv(&convw[c * 4], v[0], v[1], v[2], v[3]);
      const double got = conv[t * qkv + c];
      max_abs = std::max(max_abs, std::fabs(got - ref));
      max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref)));
    }
  }
  std::printf("ssm batch=%4zu conv max_abs=%.3g rel=%.2e\n", batch, max_abs,
              max_abs / max_ref);
  Check(max_abs / max_ref < 1e-5, "ssm conv mismatch");
  LFP_HIP_CHECK(hipStreamDestroy(st));
}

// SPEC2 item 1b(b): Fp4SplitF16Gemm vs Fp4DenseF16Gemm + UnpackQGate.
// Same tile and K order feed the same scatter, so q|gate must be
// BITWISE identical to the two-kernel reference (24 heads x d256).
void SplitCase(std::size_t batch) {
  g_hip_case = __func__;
  const std::size_t m = 12288, k = 2560, width = m / 2;
  auto w = MakeFp4(m, k, true);
  std::vector<__half> xh(batch * k);
  for (__half& v : xh) {
    v = __float2half(static_cast<float>(Next() & 0xFFFF) / 65535.f - 0.5f);
  }
  Dev d_w(w.size()), d_x(xh.size() * 2), d_q(batch * width * 4),
      d_g(batch * width * 4), d_qg(batch * m * 4), d_qr(batch * width * 4),
      d_gr(batch * width * 4), d_q2(batch * width * 4), d_g2(batch * width * 4);
  LFP_HIP_CHECK(hipMemcpy(d_w.p, w.data(), w.size(), hipMemcpyHostToDevice));
  LFP_HIP_CHECK(
      hipMemcpy(d_x.p, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
  hipStream_t st = nullptr;
  LFP_HIP_CHECK(hipStreamCreate(&st));
  const bool ok = rocm::Fp4SplitF16Gemm(
      d_w.p, static_cast<const __half*>(d_x.p), static_cast<float*>(d_q.p),
      static_cast<float*>(d_g.p), batch, m, k, true, st);
  LFP_LAUNCHED();
  Check(ok, "split launcher returned false");
  const bool okr =
      rocm::Fp4DenseF16Gemm(d_w.p, static_cast<const __half*>(d_x.p),
                            static_cast<float*>(d_qg.p), batch, m, k, true, st);
  LFP_LAUNCHED();
  Check(okr, "split ref launcher returned false");
  rocm::UnpackQGate(static_cast<const float*>(d_qg.p),
                    static_cast<std::uint32_t>(m), static_cast<float*>(d_qr.p),
                    static_cast<float*>(d_gr.p), nullptr, nullptr,
                    static_cast<std::uint32_t>(batch), 24, 256, 0, st);
  LFP_LAUNCHED();
  const bool ok2 = rocm::Fp4SplitF16Gemm(
      d_w.p, static_cast<const __half*>(d_x.p), static_cast<float*>(d_q2.p),
      static_cast<float*>(d_g2.p), batch, m, k, true, st);
  LFP_LAUNCHED();
  Check(ok2, "split replay failed");
  LFP_HIP_CHECK(hipStreamSynchronize(st));
  std::vector<float> q(batch * width), g(batch * width), qr(batch * width),
      gr(batch * width), q2(batch * width), g2(batch * width);
  LFP_HIP_CHECK(
      hipMemcpy(q.data(), d_q.p, q.size() * 4, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(
      hipMemcpy(g.data(), d_g.p, g.size() * 4, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(
      hipMemcpy(qr.data(), d_qr.p, qr.size() * 4, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(
      hipMemcpy(gr.data(), d_gr.p, gr.size() * 4, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(
      hipMemcpy(q2.data(), d_q2.p, q2.size() * 4, hipMemcpyDeviceToHost));
  LFP_HIP_CHECK(
      hipMemcpy(g2.data(), d_g2.p, g2.size() * 4, hipMemcpyDeviceToHost));
  Check(std::memcmp(q.data(), q2.data(), q.size() * 4) == 0,
        "nondeterministic split q");
  Check(std::memcmp(g.data(), g2.data(), g.size() * 4) == 0,
        "nondeterministic split gate");
  const bool q_ok = std::memcmp(q.data(), qr.data(), q.size() * 4) == 0;
  const bool g_ok = std::memcmp(g.data(), gr.data(), g.size() * 4) == 0;
  std::printf("split batch=%4zu q_bitwise=%d gate_bitwise=%d\n", batch, q_ok,
              g_ok);
  Check(q_ok && g_ok, "split mismatch");
  LFP_HIP_CHECK(hipStreamDestroy(st));
}

}  // namespace

int main() {
  g_hip_case = __func__;
  DecodeExactness();
  Check(!rocm::Fp4DenseF16Gemm(nullptr, nullptr, nullptr, 1, 1, 31, true,
                               nullptr),
        "k%32 guard missing");
  Check(!rocm::Fp4DenseF16Gemm(nullptr, nullptr, nullptr, 0, 8, 32, true,
                               nullptr),
        "batch==0 guard missing");
  // 96x2560 = the stacked ssm_alpha_beta (device_model Stack).
  const std::size_t shapes[][2] = {
      {10240, 2560}, {6144, 2560}, {2560, 6144}, {96, 2560},   {48, 2560},
      {12288, 2560}, {512, 2560},  {128, 2560},  {10240, 320}, {320, 10240},
      {4, 10240},    {640, 2560},  {2560, 640},  {2560, 2560}};
  for (std::size_t batch : {16, 96, 1024, 2048}) {
    for (const auto& s : shapes) {
      for (bool fast : {true, false}) {
        GemmCase(batch, s[0], s[1], fast);
      }
    }
  }
  Int8Exactness();
  Check(!rocm::Fp4Int8Gemm(nullptr, nullptr, nullptr, 1, 1, 31, true, nullptr),
        "int8 k%32 guard missing");
  Check(!rocm::Fp4Int8Gemm(nullptr, nullptr, nullptr, 1, 320, 10240, false,
                           nullptr),
        "int8 fast guard missing");
  Int8GemmCase(2048, 320, 10240);
  Int8GemmCase(96, 320, 10240);
  Int8GemmCase(2048, 2560, 6144);
  Int8GemmCase(96, 2560, 6144);
  MixerEquiv();
  SsmFusedCase(1024);
  SsmFusedCase(2048);
  Check(!rocm::Fp4SplitF16Gemm(nullptr, nullptr, nullptr, nullptr, 95, 12288,
                               2560, true, nullptr),
        "split batch guard missing");
  Check(!rocm::Fp4SplitF16Gemm(nullptr, nullptr, nullptr, nullptr, 96, 12287,
                               2560, true, nullptr),
        "split geometry guard missing");
  SplitCase(2048);
  SplitCase(96);
  if (failures == 0) {
    std::puts("FP4_F16_TEST PASS");
  } else {
    std::printf("FP4_F16_TEST %d FAILURES\n", failures);
  }
  return failures ? 1 : 0;
}
