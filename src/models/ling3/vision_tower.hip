// Ling-VL vision tower for King's gufo.
// Qwen3-VL graph from llama.cpp clip_graph_qwen3vl: temporal conv sum,
// 2x2 reorder, absolute position embeddings, LayerNorm + GELU ViT,
// merger mm.0 / GELU / mm.2. GEMMs run on the GPU through hipBLAS.
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define CHECK_HIP(expr)                                                       \
  do {                                                                        \
    hipError_t _e = (expr);                                                   \
    if (_e != hipSuccess) {                                                   \
      throw std::runtime_error(std::string("hip ") + hipGetErrorString(_e) +  \
                               " @ " #expr);                                  \
    }                                                                         \
  } while (0)

#define CHECK_BLAS(expr)                                                      \
  do {                                                                        \
    hipblasStatus_t _e = (expr);                                              \
    if (_e != HIPBLAS_STATUS_SUCCESS) {                                       \
      throw std::runtime_error(std::string("hipblas ") +                      \
                               std::to_string(static_cast<int>(_e)) +         \
                               " @ " #expr);                                  \
    }                                                                         \
  } while (0)

#include <cstdio>
#include <memory>
#include <mutex>

#include "src/models/ling3/vision_tower.hpp"

namespace {

constexpr int kPatch = 16;
constexpr int kMerge = 2;
constexpr int kEmb = 1152;
constexpr int kHeads = 16;
constexpr int kD = kEmb / kHeads;  // 72
constexpr int kLayers = 27;
constexpr int kFfn = 4304;
constexpr int kProj = 2560;
constexpr float kEps = 1e-6f;
constexpr float kMean[3] = {0.48145467042922974f, 0.45782750844955444f,
                            0.40821072459220886f};
constexpr float kStd[3] = {0.2686295509338379f, 0.2613025903701782f,
                           0.27577710151672363f};

struct Tensor {
  std::vector<int64_t> dims;  // ggml order, ne0 fastest
  std::vector<float> data;
  int64_t n0() const { return dims.empty() ? 1 : dims[0]; }
  int64_t n1() const { return dims.size() < 2 ? 1 : dims[1]; }
};

using Weights = std::unordered_map<std::string, Tensor>;

void ReadExact(std::ifstream& in, void* dst, size_t n) {
  in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
  if (!in) throw std::runtime_error("gguf truncated");
}

uint64_t ReadU64(std::ifstream& in) {
  uint64_t v;
  ReadExact(in, &v, 8);
  return v;
}

std::string ReadStr(std::ifstream& in) {
  const uint64_t n = ReadU64(in);
  std::string s(n, '\0');
  if (n) ReadExact(in, s.data(), n);
  return s;
}

void SkipVal(std::ifstream& in, uint32_t typ);

void SkipVal(std::ifstream& in, uint32_t typ) {
  const size_t sz[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
  if (typ == 8) {
    ReadStr(in);
    return;
  }
  if (typ == 9) {
    uint32_t et;
    ReadExact(in, &et, 4);
    const uint64_t n = ReadU64(in);
    for (uint64_t i = 0; i < n; ++i) SkipVal(in, et);
    return;
  }
  if (typ > 12) throw std::runtime_error("bad gguf type");
  in.seekg(static_cast<std::streamoff>(sz[typ]), std::ios::cur);
}

Weights LoadGguf(const std::string& path, std::string* arch_out = nullptr,
                   std::string* type_out = nullptr,
                   std::size_t* host_bytes = nullptr) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("open " + path);
  std::string arch, gtype;
  char magic[4];
  ReadExact(in, magic, 4);
  if (std::memcmp(magic, "GGUF", 4) != 0) throw std::runtime_error("not gguf");
  uint32_t version;
  ReadExact(in, &version, 4);
  const uint64_t n_tensors = ReadU64(in);
  const uint64_t n_kv = ReadU64(in);
  uint64_t alignment = 32;
  for (uint64_t i = 0; i < n_kv; ++i) {
    const std::string key = ReadStr(in);
    uint32_t typ;
    ReadExact(in, &typ, 4);
    if (key == "general.alignment" && typ == 4) {
      uint32_t a;
      ReadExact(in, &a, 4);
      alignment = a;
    } else if (typ == 8 &&
               (key == "general.architecture" || key == "general.type")) {
      const std::string value = ReadStr(in);
      if (key == "general.architecture") arch = value;
      else gtype = value;
    } else {
      SkipVal(in, typ);
    }
  }
  if ((!arch.empty() && arch != "clip") ||
      (!gtype.empty() && gtype != "mmproj")) {
    throw std::runtime_error(
        "mmproj sidecar is not a clip mmproj (architecture=" + arch +
        " type=" + gtype + ")");
  }
  struct Info {
    std::string name;
    std::vector<int64_t> dims;
    uint32_t type;
    uint64_t offset;
  };
  std::vector<Info> infos;
  infos.reserve(n_tensors);
  for (uint64_t i = 0; i < n_tensors; ++i) {
    Info info;
    info.name = ReadStr(in);
    uint32_t nd;
    ReadExact(in, &nd, 4);
    info.dims.resize(nd);
    for (uint32_t d = 0; d < nd; ++d) info.dims[d] = static_cast<int64_t>(ReadU64(in));
    ReadExact(in, &info.type, 4);
    info.offset = ReadU64(in);
    infos.push_back(std::move(info));
  }
  const auto pos = static_cast<uint64_t>(in.tellg());
  const uint64_t data_off = (pos + alignment - 1) / alignment * alignment;
  Weights out;
  for (const Info& info : infos) {
    int64_t n = 1;
    for (int64_t d : info.dims) n *= d;
    Tensor tensor;
    tensor.dims = info.dims;
    tensor.data.resize(static_cast<size_t>(n));
    in.seekg(static_cast<std::streamoff>(data_off + info.offset));
    if (info.type == 0) {
      ReadExact(in, tensor.data.data(), static_cast<size_t>(n) * 4);
    } else if (info.type == 30) {
      std::vector<uint16_t> raw(static_cast<size_t>(n));
      ReadExact(in, raw.data(), raw.size() * 2);
      for (int64_t i = 0; i < n; ++i) {
        uint32_t bits = static_cast<uint32_t>(raw[static_cast<size_t>(i)]) << 16;
        float f;
        std::memcpy(&f, &bits, 4);
        tensor.data[static_cast<size_t>(i)] = f;
      }
    } else {
      throw std::runtime_error("unsupported tensor type " + info.name);
    }
    out.emplace(info.name, std::move(tensor));
  }
  if (arch_out != nullptr) *arch_out = arch;
  if (type_out != nullptr) *type_out = gtype;
  if (host_bytes != nullptr) {
    std::size_t host = 0;
    for (const auto& item : out) host += item.second.data.size() * sizeof(float);
    *host_bytes = host;
  }
  return out;
}

void Stat(const char* name, const std::vector<float>& a) {
  double sum = 0;
  for (float v : a) sum += static_cast<double>(v);
  std::cerr << "STAT " << name << " n=" << a.size() << " sum=" << sum << " head=";
  const int k = std::min<int>(8, static_cast<int>(a.size()));
  for (int i = 0; i < k; ++i) std::cerr << a[static_cast<size_t>(i)] << ' ';
  std::cerr << "\n";
}

__device__ float BlockSum(float v) {
  constexpr int kThreads = 256;
  __shared__ float sm[kThreads];
  const int tid = threadIdx.x;
  sm[tid] = v;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (tid < s) sm[tid] += sm[tid + s];
    __syncthreads();
  }
  return sm[0];
}

__global__ void LnKernel(const float* x, const float* w, const float* b, float* y,
                         int d, float eps) {
  const int t = blockIdx.x;
  const float* row = x + static_cast<int64_t>(t) * d;
  float* out = y + static_cast<int64_t>(t) * d;
  float sum = 0.f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) sum += row[i];
  const float mean = BlockSum(sum) / static_cast<float>(d);
  float var = 0.f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    const float v = row[i] - mean;
    var += v * v;
  }
  var = BlockSum(var) / static_cast<float>(d);
  const float inv = rsqrtf(var + eps);
  for (int i = threadIdx.x; i < d; i += blockDim.x)
    out[i] = (row[i] - mean) * inv * w[i] + b[i];
}

__global__ void BiasKernel(float* y, const float* b, int n, int d) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * d) return;
  y[i] += b[i % d];
}

__global__ void AddKernel(float* y, const float* x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += x[i];
}

__global__ void GeluKernel(float* y, const float* x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = x[i];
  const float inner = 0.7978845608028654f * v * (1.f + 0.044715f * v * v);
  y[i] = 0.5f * v * (1.f + tanhf(inner));
}

// ggml_compute_fp32_to_bf16, then back to f32. Block GEMMs dot BF16
// activations against BF16 weights; the patch conv stays f32.
__global__ void RoundBf16Kernel(float* x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t u = __float_as_uint(x[i]);
  uint32_t bits;
  if ((u & 0x7fffffffu) > 0x7f800000u) {
    bits = (u >> 16) | 64u;
  } else {
    bits = (u + (0x7fffu + ((u >> 16) & 1u))) >> 16;
  }
  x[i] = __uint_as_float(bits << 16);
}

__global__ void SplitQkvKernel(const float* qkv, float* q, float* k, float* v,
                               int n_pos) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_pos * kHeads * kD;
  if (idx >= total) return;
  const int di = idx % kD;
  const int h = (idx / kD) % kHeads;
  const int t = idx / (kD * kHeads);
  const int src = t * (3 * kEmb) + h * kD + di;
  q[(h * n_pos + t) * kD + di] = qkv[src];
  k[(h * n_pos + t) * kD + di] = qkv[src + kEmb];
  v[(h * n_pos + t) * kD + di] = qkv[src + 2 * kEmb];
}

__global__ void RopeKernel(float* q, const int* pos_y, const int* pos_x,
                           const float* freq, int n_pos) {
  const int pair = threadIdx.x;  // 0..35
  if (pair >= kD / 2) return;
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  if (t >= n_pos) return;
  const float theta =
      (pair < 18 ? static_cast<float>(pos_y[t]) : static_cast<float>(pos_x[t])) *
      freq[pair < 18 ? pair : pair - 18];
  const float c = cosf(theta);
  const float s = sinf(theta);
  float* row = q + (static_cast<int64_t>(h) * n_pos + t) * kD;
  const float x0 = row[pair];
  const float x1 = row[pair + kD / 2];
  row[pair] = x0 * c - x1 * s;
  row[pair + kD / 2] = x0 * s + x1 * c;
}

__global__ void QkKernel(const float* q, const float* k, float* scores, int n_pos,
                         float scale) {
  const int h = blockIdx.z;
  const int t = blockIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n_pos) return;
  const float* qq = q + (static_cast<int64_t>(h) * n_pos + t) * kD;
  const float* kk = k + (static_cast<int64_t>(h) * n_pos + j) * kD;
  float dot = 0.f;
  for (int i = 0; i < kD; ++i) dot += qq[i] * kk[i];
  scores[(static_cast<int64_t>(h) * n_pos + t) * n_pos + j] = dot * scale;
}

__global__ void SoftmaxKernel(float* scores, int n_pos) {
  const int h = blockIdx.y;
  const int t = blockIdx.x;
  float* row = scores + (static_cast<int64_t>(h) * n_pos + t) * n_pos;
  float m = -1e30f;
  for (int j = threadIdx.x; j < n_pos; j += blockDim.x) m = fmaxf(m, row[j]);
  __shared__ float sm[256];
  sm[threadIdx.x] = m;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
    __syncthreads();
  }
  m = sm[0];
  float sum = 0.f;
  for (int j = threadIdx.x; j < n_pos; j += blockDim.x) {
    const float e = expf(row[j] - m);
    row[j] = e;
    sum += e;
  }
  sm[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
    __syncthreads();
  }
  sum = sm[0];
  const float inv = 1.f / sum;
  for (int j = threadIdx.x; j < n_pos; j += blockDim.x) row[j] *= inv;
}

__global__ void AvKernel(const float* scores, const float* v, float* ctx, int n_pos) {
  const int h = blockIdx.y;
  const int t = blockIdx.x;
  const float* att = scores + (static_cast<int64_t>(h) * n_pos + t) * n_pos;
  float* out = ctx + (static_cast<int64_t>(h) * n_pos + t) * kD;
  for (int di = threadIdx.x; di < kD; di += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < n_pos; ++j) {
      acc += att[j] * v[(static_cast<int64_t>(h) * n_pos + j) * kD + di];
    }
    out[di] = acc;
  }
}

__global__ void MergeHeadsKernel(const float* ctx, float* y, int n_pos) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_pos * kEmb;
  if (idx >= total) return;
  const int di = idx % kD;
  const int h = (idx / kD) % kHeads;
  const int t = idx / kEmb;
  y[t * kEmb + h * kD + di] = ctx[(static_cast<int64_t>(h) * n_pos + t) * kD + di];
}

__global__ void ReorderKernel(const float* src, float* dst, int pw, int ph) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = pw * ph;
  if (idx >= n * kEmb) return;
  const int c = idx % kEmb;
  const int dst_t = idx / kEmb;
  const int hw = pw / 2;
  const int group = dst_t / 4;
  const int intra = dst_t % 4;
  const int ty = group / hw;
  const int tx = group % hw;
  const int dy = intra / 2;
  const int dx = intra % 2;
  const int src_t = (ty * 2 + dy) * pw + (tx * 2 + dx);
  dst[dst_t * kEmb + c] = src[src_t * kEmb + c];
}

__global__ void Pack4Kernel(const float* src, float* dst, int n_pos) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int rows = n_pos / 4;
  const int width = kEmb * 4;
  if (idx >= rows * width) return;
  dst[idx] = src[idx];  // groups of 4 tokens are already contiguous
}

struct GpuBuf {
  float* p{nullptr};
  explicit GpuBuf(size_t n = 0) {
    if (n) CHECK_HIP(hipMalloc(&p, n * sizeof(float)));
  }
  ~GpuBuf() {
    if (p) (void)hipFree(p);
  }
  GpuBuf(const GpuBuf&) = delete;
  GpuBuf& operator=(const GpuBuf&) = delete;
  GpuBuf(GpuBuf&& o) noexcept : p(o.p) { o.p = nullptr; }
  GpuBuf& operator=(GpuBuf&& o) noexcept {
    if (this != &o) {
      if (p) (void)hipFree(p);
      p = o.p;
      o.p = nullptr;
    }
    return *this;
  }
  void Upload(const float* src, size_t n) {
    CHECK_HIP(hipMemcpy(p, src, n * sizeof(float), hipMemcpyHostToDevice));
  }
  void Download(float* dst, size_t n) const {
    CHECK_HIP(hipMemcpy(dst, p, n * sizeof(float), hipMemcpyDeviceToHost));
  }
};

void Gemm(hipblasHandle_t blas, const float* W, int n_out, int n_in, const float* X,
          int n_tok, float* Y) {
  const float alpha = 1.f, beta = 0.f;
  CHECK_BLAS(hipblasSgemm(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, n_out, n_tok, n_in, &alpha,
                          W, n_in, X, n_in, &beta, Y, n_out));
}

void LaunchLn(const float* x, const float* w, const float* b, float* y, int n, int d) {
  hipLaunchKernelGGL(LnKernel, dim3(n), dim3(256), 0, 0, x, w, b, y, d, kEps);
  CHECK_HIP(hipGetLastError());
}

void LaunchBias(float* y, const float* b, int n, int d) {
  const int m = n * d;
  hipLaunchKernelGGL(BiasKernel, dim3((m + 255) / 256), dim3(256), 0, 0, y, b, n, d);
  CHECK_HIP(hipGetLastError());
}

void LaunchAdd(float* y, const float* x, int n) {
  hipLaunchKernelGGL(AddKernel, dim3((n + 255) / 256), dim3(256), 0, 0, y, x, n);
  CHECK_HIP(hipGetLastError());
}

void LaunchGelu(float* y, const float* x, int n) {
  hipLaunchKernelGGL(GeluKernel, dim3((n + 255) / 256), dim3(256), 0, 0, y, x, n);
  CHECK_HIP(hipGetLastError());
}

void RoundBf16(float* x, int n) {
  hipLaunchKernelGGL(RoundBf16Kernel, dim3((n + 255) / 256), dim3(256), 0, 0, x, n);
  CHECK_HIP(hipGetLastError());
}

std::vector<float> Im2Col(const std::vector<float>& chw, int width, int height) {
  const int pw = width / kPatch;
  const int ph = height / kPatch;
  std::vector<float> col(static_cast<size_t>(pw * ph) * 3 * kPatch * kPatch);
  for (int oy = 0; oy < ph; ++oy) {
    for (int ox = 0; ox < pw; ++ox) {
      float* dst = col.data() + static_cast<size_t>(oy * pw + ox) * 768;
      for (int ic = 0; ic < 3; ++ic) {
        for (int kh = 0; kh < kPatch; ++kh) {
          for (int kw = 0; kw < kPatch; ++kw) {
            const int x = ox * kPatch + kw;
            const int y = oy * kPatch + kh;
            const float pix = chw[(static_cast<size_t>(ic) * height + y) * width + x];
            dst[kw + kh * kPatch + ic * kPatch * kPatch] = pix;
          }
        }
      }
    }
  }
  return col;
}

void BuildPositions(int pw, int ph, std::vector<int>* y, std::vector<int>* x) {
  const int n = pw * ph;
  y->assign(n, 0);
  x->assign(n, 0);
  int ptr = 0;
  for (int yy = 0; yy < ph; yy += 2) {
    for (int xx = 0; xx < pw; xx += 2) {
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          (*y)[ptr] = yy + dy;
          (*x)[ptr] = xx + dx;
          ++ptr;
        }
      }
    }
  }
}

// Antialiased bilinear, ggml layout src (side, side, C) with x fastest in the
// first index of a host array shaped [x][y][c] stored as c + y*C + x*C*side? 
// We pass src as (side, side, C) with index ((x * side) + y) * C + c, i.e. x major.
std::vector<float> InterpPos(const Tensor& pos, int pw, int ph) {
  const int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(pos.n1()))));
  // ggml [C, side*side], pos_index = x + y*side, data[c + pos*C]
  auto at = [&](int x, int y, int c) -> float {
    return pos.data[static_cast<size_t>(c + (x + y * side) * kEmb)];
  };
  std::vector<float> raster(static_cast<size_t>(pw * ph) * kEmb);
  if (pw == side && ph == side) {
    for (int y = 0; y < ph; ++y)
      for (int x = 0; x < pw; ++x)
        for (int c = 0; c < kEmb; ++c)
          raster[(static_cast<size_t>(y * pw + x) * kEmb) + c] = at(x, y, c);
    return raster;
  }
  const float sf0 = static_cast<float>(pw) / static_cast<float>(side);
  const float sf1 = static_cast<float>(ph) / static_cast<float>(side);
  const float pixel_offset = 0.5f;
  const float support0 = std::max(1.f, 1.f / sf0);
  const float support1 = std::max(1.f, 1.f / sf1);
  const float inv0 = 1.f / support0;
  const float inv1 = 1.f / support1;
  auto tri = [](float v) { return std::max(1.f - std::fabs(v), 0.f); };
  auto trunc_i = [](float v) { return static_cast<int>(v); };
  for (int i1 = 0; i1 < ph; ++i1) {
    const float y = (static_cast<float>(i1) + pixel_offset) / sf1;
    const int y_min = std::max(trunc_i(y - support1 + pixel_offset), 0);
    const int y_max = std::min(trunc_i(y + support1 + pixel_offset), side);
    for (int i0 = 0; i0 < pw; ++i0) {
      const float x = (static_cast<float>(i0) + pixel_offset) / sf0;
      const int x_min = std::max(trunc_i(x - support0 + pixel_offset), 0);
      const int x_max = std::min(trunc_i(x + support0 + pixel_offset), side);
      std::vector<double> acc(kEmb, 0.0);
      double total = 0.0;
      for (int sy = y_min; sy < y_max; ++sy) {
        const float wy = tri((static_cast<float>(sy) - y + pixel_offset) * inv1);
        if (wy <= 0.f) continue;
        for (int sx = x_min; sx < x_max; ++sx) {
          const float wx = tri((static_cast<float>(sx) - x + pixel_offset) * inv0);
          const float weight = wx * wy;
          if (weight <= 0.f) continue;
          for (int c = 0; c < kEmb; ++c)
            acc[c] += static_cast<double>(at(sx, sy, c)) * weight;
          total += weight;
        }
      }
      float* dst = raster.data() + (static_cast<size_t>(i1 * pw + i0) * kEmb);
      if (total > 0.0) {
        for (int c = 0; c < kEmb; ++c)
          dst[c] = static_cast<float>(acc[c] / total);
      }
    }
  }
  return raster;
}

const Tensor& Need(const Weights& w, const std::string& name) {
  const auto it = w.find(name);
  if (it == w.end()) throw std::runtime_error("missing " + name);
  return it->second;
}

}  // namespace

std::vector<float> RunEncode(const Weights& weights, const unsigned char* rgb,
                             int height, int width) {
    if (rgb == nullptr || width <= 0 || height <= 0 || (width % 32) != 0 ||
        (height % 32) != 0 || width > 1024 || height > 1024) {
      throw std::runtime_error("image must be a positive multiple of 32 and at most 1024");
    }
    const int pw = width / kPatch;
    const int ph = height / kPatch;
    const int n_pos = pw * ph;
    const int n_out = n_pos / 4;
    std::vector<float> chw(static_cast<size_t>(3) * height * width);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < 3; ++c) {
          const float p = rgb[(static_cast<size_t>(y) * width + x) * 3 + c];
          chw[(static_cast<size_t>(c) * height + y) * width + x] =
              (p / 255.f - kMean[c]) / kStd[c];
        }
      }
    }

    const Tensor& pe0 = Need(weights, "v.patch_embd.weight");
    const Tensor& pe1 = Need(weights, "v.patch_embd.weight.1");
    std::vector<float> kernel(static_cast<size_t>(kEmb) * 768);
    for (size_t i = 0; i < kernel.size(); ++i) kernel[i] = pe0.data[i] + pe1.data[i];
    std::vector<float> col = Im2Col(chw, width, height);
    std::vector<float> pos_raster = InterpPos(Need(weights, "v.position_embd.weight"), pw, ph);

    hipblasHandle_t blas;
    CHECK_BLAS(hipblasCreate(&blas));

    GpuBuf d_kernel(kernel.size());
    d_kernel.Upload(kernel.data(), kernel.size());
    GpuBuf d_col(col.size());
    d_col.Upload(col.data(), col.size());
    GpuBuf d_raster(static_cast<size_t>(n_pos) * kEmb);
    Gemm(blas, d_kernel.p, kEmb, 768, d_col.p, n_pos, d_raster.p);
    CHECK_HIP(hipDeviceSynchronize());

    GpuBuf d_tok(static_cast<size_t>(n_pos) * kEmb);
    hipLaunchKernelGGL(ReorderKernel, dim3((n_pos * kEmb + 255) / 256), dim3(256), 0, 0,
                       d_raster.p, d_tok.p, pw, ph);
    CHECK_HIP(hipGetLastError());
    GpuBuf d_bias(kEmb);
    d_bias.Upload(Need(weights, "v.patch_embd.bias").data.data(), kEmb);
    LaunchBias(d_tok.p, d_bias.p, n_pos, kEmb);
    CHECK_HIP(hipDeviceSynchronize());
    {
      std::vector<float> host(static_cast<size_t>(n_pos) * kEmb);
      d_tok.Download(host.data(), host.size());
      Stat("patch_bias", host);
    }

    GpuBuf d_pos(pos_raster.size());
    // reorder position embeddings the same way as patches
    GpuBuf d_pos_src(pos_raster.size());
    d_pos_src.Upload(pos_raster.data(), pos_raster.size());
    hipLaunchKernelGGL(ReorderKernel, dim3((n_pos * kEmb + 255) / 256), dim3(256), 0, 0,
                       d_pos_src.p, d_pos.p, pw, ph);
    CHECK_HIP(hipGetLastError());
    LaunchAdd(d_tok.p, d_pos.p, n_pos * kEmb);
    CHECK_HIP(hipDeviceSynchronize());
    {
      std::vector<float> host(static_cast<size_t>(n_pos) * kEmb);
      d_tok.Download(host.data(), host.size());
      Stat("inp_pos_emb", host);
    }

    std::vector<int> pos_y, pos_x;
    BuildPositions(pw, ph, &pos_y, &pos_x);
    int *d_y = nullptr, *d_x = nullptr;
    CHECK_HIP(hipMalloc(&d_y, pos_y.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_x, pos_x.size() * sizeof(int)));
    CHECK_HIP(hipMemcpy(d_y, pos_y.data(), pos_y.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_x, pos_x.data(), pos_x.size() * sizeof(int), hipMemcpyHostToDevice));
    float freq[18];
    const float theta_scale = std::pow(10000.f, -2.f / 36.f);
    freq[0] = 1.f;
    for (int i = 1; i < 18; ++i) freq[i] = freq[i - 1] * theta_scale;
    GpuBuf d_freq(18);
    d_freq.Upload(freq, 18);

    const float scale = 1.f / std::sqrt(static_cast<float>(kD));
    GpuBuf d_n1(static_cast<size_t>(n_pos) * kEmb);
    GpuBuf d_qkv(static_cast<size_t>(n_pos) * 3 * kEmb);
    GpuBuf d_q(static_cast<size_t>(kHeads) * n_pos * kD);
    GpuBuf d_k(static_cast<size_t>(kHeads) * n_pos * kD);
    GpuBuf d_v(static_cast<size_t>(kHeads) * n_pos * kD);
    GpuBuf d_scores(static_cast<size_t>(kHeads) * n_pos * n_pos);
    GpuBuf d_ctx(static_cast<size_t>(kHeads) * n_pos * kD);
    GpuBuf d_merged(static_cast<size_t>(n_pos) * kEmb);
    GpuBuf d_attn(static_cast<size_t>(n_pos) * kEmb);
    GpuBuf d_up(static_cast<size_t>(n_pos) * kFfn);
    GpuBuf d_gelu(static_cast<size_t>(n_pos) * kFfn);
    GpuBuf d_down(static_cast<size_t>(n_pos) * kEmb);

    auto upload_w = [&](const std::string& name) {
      const Tensor& t = Need(weights, name);
      GpuBuf buf(t.data.size());
      buf.Upload(t.data.data(), t.data.size());
      return buf;
    };

    for (int il = 0; il < kLayers; ++il) {
      const std::string p = "v.blk." + std::to_string(il) + ".";
      GpuBuf ln1_w = upload_w(p + "ln1.weight");
      GpuBuf ln1_b = upload_w(p + "ln1.bias");
      GpuBuf qkv_w = upload_w(p + "attn_qkv.weight");
      GpuBuf qkv_b = upload_w(p + "attn_qkv.bias");
      GpuBuf o_w = upload_w(p + "attn_out.weight");
      GpuBuf o_b = upload_w(p + "attn_out.bias");
      GpuBuf ln2_w = upload_w(p + "ln2.weight");
      GpuBuf ln2_b = upload_w(p + "ln2.bias");
      GpuBuf up_w = upload_w(p + "ffn_up.weight");
      GpuBuf up_b = upload_w(p + "ffn_up.bias");
      GpuBuf dn_w = upload_w(p + "ffn_down.weight");
      GpuBuf dn_b = upload_w(p + "ffn_down.bias");

      LaunchLn(d_tok.p, ln1_w.p, ln1_b.p, d_n1.p, n_pos, kEmb);
      RoundBf16(d_n1.p, n_pos * kEmb);
      Gemm(blas, qkv_w.p, 3 * kEmb, kEmb, d_n1.p, n_pos, d_qkv.p);
      LaunchBias(d_qkv.p, qkv_b.p, n_pos, 3 * kEmb);
      hipLaunchKernelGGL(SplitQkvKernel, dim3((n_pos * kEmb + 255) / 256), dim3(256), 0, 0,
                         d_qkv.p, d_q.p, d_k.p, d_v.p, n_pos);
      CHECK_HIP(hipGetLastError());
      hipLaunchKernelGGL(RopeKernel, dim3(n_pos, kHeads), dim3(kD / 2), 0, 0, d_q.p, d_y, d_x,
                         d_freq.p, n_pos);
      hipLaunchKernelGGL(RopeKernel, dim3(n_pos, kHeads), dim3(kD / 2), 0, 0, d_k.p, d_y, d_x,
                         d_freq.p, n_pos);
      CHECK_HIP(hipGetLastError());
      hipLaunchKernelGGL(QkKernel, dim3((n_pos + 63) / 64, n_pos, kHeads), dim3(64), 0, 0, d_q.p,
                         d_k.p, d_scores.p, n_pos, scale);
      CHECK_HIP(hipGetLastError());
      hipLaunchKernelGGL(SoftmaxKernel, dim3(n_pos, kHeads), dim3(256), 0, 0, d_scores.p, n_pos);
      CHECK_HIP(hipGetLastError());
      hipLaunchKernelGGL(AvKernel, dim3(n_pos, kHeads), dim3(72), 0, 0, d_scores.p, d_v.p, d_ctx.p,
                         n_pos);
      CHECK_HIP(hipGetLastError());
      hipLaunchKernelGGL(MergeHeadsKernel, dim3((n_pos * kEmb + 255) / 256), dim3(256), 0, 0,
                         d_ctx.p, d_merged.p, n_pos);
      CHECK_HIP(hipGetLastError());
      RoundBf16(d_merged.p, n_pos * kEmb);
      Gemm(blas, o_w.p, kEmb, kEmb, d_merged.p, n_pos, d_attn.p);
      LaunchBias(d_attn.p, o_b.p, n_pos, kEmb);
      LaunchAdd(d_tok.p, d_attn.p, n_pos * kEmb);

      LaunchLn(d_tok.p, ln2_w.p, ln2_b.p, d_n1.p, n_pos, kEmb);
      RoundBf16(d_n1.p, n_pos * kEmb);
      Gemm(blas, up_w.p, kFfn, kEmb, d_n1.p, n_pos, d_up.p);
      LaunchBias(d_up.p, up_b.p, n_pos, kFfn);
      LaunchGelu(d_gelu.p, d_up.p, n_pos * kFfn);
      RoundBf16(d_gelu.p, n_pos * kFfn);
      Gemm(blas, dn_w.p, kEmb, kFfn, d_gelu.p, n_pos, d_down.p);
      LaunchBias(d_down.p, dn_b.p, n_pos, kEmb);
      LaunchAdd(d_tok.p, d_down.p, n_pos * kEmb);
      CHECK_HIP(hipDeviceSynchronize());
      if (il == 0 || il == kLayers - 1) {
        std::vector<float> host(static_cast<size_t>(n_pos) * kEmb);
        d_tok.Download(host.data(), host.size());
        Stat(il == 0 ? "layer_out-0" : "layer_out-26", host);
      }
    }

    GpuBuf post_w = upload_w("v.post_ln.weight");
    GpuBuf post_b = upload_w("v.post_ln.bias");
    LaunchLn(d_tok.p, post_w.p, post_b.p, d_n1.p, n_pos, kEmb);
    GpuBuf d_pack(static_cast<size_t>(n_out) * kEmb * 4);
    // n1 is [n_pos, kEmb] and groups of 4 consecutive tokens are the merger rows.
    CHECK_HIP(hipMemcpy(d_pack.p, d_n1.p, static_cast<size_t>(n_pos) * kEmb * sizeof(float),
                        hipMemcpyDeviceToDevice));
    GpuBuf mm0_w = upload_w("mm.0.weight");
    GpuBuf mm0_b = upload_w("mm.0.bias");
    GpuBuf mm2_w = upload_w("mm.2.weight");
    GpuBuf mm2_b = upload_w("mm.2.bias");
    GpuBuf d_h1(static_cast<size_t>(n_out) * kProj);
    GpuBuf d_h1g(static_cast<size_t>(n_out) * kProj);
    GpuBuf d_emb(static_cast<size_t>(n_out) * kProj);
    RoundBf16(d_pack.p, n_out * kEmb * 4);
    Gemm(blas, mm0_w.p, kProj, kEmb * 4, d_pack.p, n_out, d_h1.p);
    LaunchBias(d_h1.p, mm0_b.p, n_out, kProj);
    LaunchGelu(d_h1g.p, d_h1.p, n_out * kProj);
    RoundBf16(d_h1g.p, n_out * kProj);
    Gemm(blas, mm2_w.p, kProj, kProj, d_h1g.p, n_out, d_emb.p);
    LaunchBias(d_emb.p, mm2_b.p, n_out, kProj);
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<float> emb(static_cast<size_t>(n_out) * kProj);
    d_emb.Download(emb.data(), emb.size());
    Stat("embeddings", emb);
    (void)hipFree(d_y);
    (void)hipFree(d_x);
    hipblasDestroy(blas);
    return emb;
}

std::size_t ScratchFor(int height, int width) {
    const int n_pos = (height / kPatch) * (width / kPatch);
    const int n_out = n_pos / 4;
    std::size_t bytes = 0;
    auto add = [&](std::size_t n) { bytes += n * sizeof(float); };
    add(static_cast<std::size_t>(n_pos) * 768);
    add(static_cast<std::size_t>(n_pos) * kEmb * 14);
    add(static_cast<std::size_t>(n_pos) * 3 * kEmb);
    add(static_cast<std::size_t>(kHeads) * n_pos * static_cast<std::size_t>(n_pos));
    add(static_cast<std::size_t>(n_pos) * kFfn * 2);
    add(static_cast<std::size_t>(n_out) * kEmb * 4);
    add(static_cast<std::size_t>(n_out) * kProj * 3);
    add(static_cast<std::size_t>(kEmb) * 768);
    add(static_cast<std::size_t>(3) * kEmb * kEmb);
    add(static_cast<std::size_t>(kFfn) * kEmb * 2);
    add(static_cast<std::size_t>(kProj) * kEmb * 4);
    add(static_cast<std::size_t>(kProj) * kProj);
    add(1u << 20);
    return bytes;
}

struct TowerSlot {
  Weights weights;
  std::size_t host_bytes{0};
  std::mutex mu;
};
std::vector<std::unique_ptr<TowerSlot>> g_slots;

#ifndef LVV_NO_MAIN
int main(int argc, char** argv) {
  try {
    if (argc < 5) {
      std::cerr << "usage: lvv_tower <mmproj> <rgb> <h> <w> [ref.bin]\n";
      return 2;
    }
    const std::string mmproj = argv[1];
    const std::string rgb_path = argv[2];
    const int height = std::atoi(argv[3]);
    const int width = std::atoi(argv[4]);
    const char* ref_path = argc >= 6 ? argv[5] : nullptr;
    std::ifstream rin(rgb_path, std::ios::binary);
    std::vector<unsigned char> rgb(static_cast<size_t>(height) * width * 3);
    rin.read(reinterpret_cast<char*>(rgb.data()), rgb.size());
    if (!rin) throw std::runtime_error("rgb read");
    std::cerr << "load weights\n";
    Weights weights = LoadGguf(mmproj);
    std::vector<float> emb = RunEncode(weights, rgb.data(), height, width);
    const int n_out = (width / kPatch) * (height / kPatch) / 4;
    const char* env_out = std::getenv("LVV_EMB_OUT");
    const std::string out_path =
        (env_out != nullptr && env_out[0] != '\0') ? env_out
                                                    : "/path/to/lvv/gufo-emb.bin";
    std::ofstream out(out_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(emb.data()), emb.size() * sizeof(float));
    std::cerr << "wrote " << out_path << " rows=" << n_out << " dim=" << kProj << "\n";

    if (ref_path != nullptr) {
      std::ifstream ref(ref_path, std::ios::binary);
      std::vector<float> r(emb.size());
      ref.read(reinterpret_cast<char*>(r.data()), r.size() * sizeof(float));
      if (!ref) throw std::runtime_error("ref bin short");
      double min_cos = 1.0;
      double max_abs = 0.0;
      int worst = 0;
      for (int row = 0; row < n_out; ++row) {
        double dot = 0, na = 0, nb = 0, ma = 0;
        for (int c = 0; c < kProj; ++c) {
          const double a = emb[static_cast<size_t>(row) * kProj + c];
          const double b = r[static_cast<size_t>(row) * kProj + c];
          dot += a * b;
          na += a * a;
          nb += b * b;
          ma = std::max(ma, std::fabs(a - b));
        }
        const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
        if (cos < min_cos) {
          min_cos = cos;
          worst = row;
        }
        max_abs = std::max(max_abs, ma);
      }
      std::cerr << "COMPARE rows=" << n_out << " min_cosine=" << min_cos
                << " worst_row=" << worst << " max_abs=" << max_abs << "\n";
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "ERR " << ex.what() << "\n";
    return 1;
  }
}
#endif

namespace gufo::models::ling3::vision {

struct VisionTower::Impl {
  int slot{-1};
};

VisionTower::VisionTower() = default;
VisionTower::~VisionTower() = default;

std::shared_ptr<VisionTower> VisionTower::Open(const std::string& path,
                                               std::string* error) {
  try {
    std::string arch;
    std::string gtype;
    std::size_t host = 0;
    Weights weights = LoadGguf(path, &arch, &gtype, &host);
    if (arch != "clip" || gtype != "mmproj") {
      if (error != nullptr) {
        *error = "mmproj sidecar is not a clip mmproj (architecture=" + arch +
                 " type=" + gtype + ")";
      }
      return {};
    }
    const Tensor& mm2 = Need(weights, "mm.2.weight");
    if (mm2.n0() != 2560 || mm2.n1() != 2560) {
      if (error != nullptr) {
        *error = "mmproj projector width does not match Ling hidden size 2560";
      }
      return {};
    }
    auto slot = std::make_unique<TowerSlot>();
    slot->host_bytes = host;
    slot->weights = std::move(weights);
    g_slots.push_back(std::move(slot));
    auto tower = std::shared_ptr<VisionTower>(new VisionTower());
    tower->impl_ = std::make_unique<Impl>();
    tower->impl_->slot = static_cast<int>(g_slots.size()) - 1;
    std::fprintf(stderr, "LVV_VISION host_bytes=%zu scratch_bytes=%zu\n",
                 host, ScratchFor(1024, 1024));
    return tower;
  } catch (const std::exception& ex) {
    if (error != nullptr) *error = ex.what();
    return {};
  }
}

std::vector<float> VisionTower::Encode(const std::uint8_t* rgb, int height,
                                       int width, std::string* error) {
  if (impl_ == nullptr || impl_->slot < 0 ||
      static_cast<std::size_t>(impl_->slot) >= g_slots.size()) {
    if (error != nullptr) *error = "vision tower is not open";
    return {};
  }
  TowerSlot& slot = *g_slots[static_cast<std::size_t>(impl_->slot)];
  std::lock_guard<std::mutex> lock(slot.mu);
  try {
    if (error != nullptr) error->clear();
    return RunEncode(slot.weights, rgb, height, width);
  } catch (const std::exception& ex) {
    if (error != nullptr) *error = ex.what();
    return {};
  }
}

std::size_t VisionTower::HostBytes() const noexcept {
  if (impl_ == nullptr || impl_->slot < 0) return 0;
  return g_slots[static_cast<std::size_t>(impl_->slot)]->host_bytes;
}

std::size_t VisionTower::ScratchBytes() const noexcept {
  return ScratchFor(1024, 1024);
}

}  // namespace gufo::models::ling3::vision
