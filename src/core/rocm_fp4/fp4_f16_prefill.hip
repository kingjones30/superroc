// SQ1 (Super ROC wave, A1a; rd1 DESIGN.md section 2.2.1): the Flash-Next
// FP4->F16 prefill kernels, promoted byte-exact from
// src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp at 1194fea
// (Fp4Int8GemmKernel, Fp4DenseF16GEMMKernel, Fp4SplitF16GemmKernel,
// Fp4SsmF16GemmKernel + the four launchers). Namespaces and code are
// unchanged, so kernel symbols and numerics are identical; the qwen
// family calls the same launchers for ROC FP4 prefill. Compile flags
// must match kernels.hip.cpp (-ffast-math -fno-finite-math-only; set in
// src/models/qwen/CMakeLists.txt next to this file's registration).
#include "src/core/rocm_fp4/fp4_f16_prefill.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

// Launch counting (LFN_DECODE_PROFILE) shared with kernels.hip.cpp.
#include "src/models/qwen38_flash_next/kernels/rocm/lfn_launch.hpp"

#undef hipLaunchKernelGGL
#define hipLaunchKernelGGL(kernelName, numBlocks, numThreads, memPerBlock, streamId, ...) \
  do {                                                                                    \
    ::gufo::models::qwen38_flash_next::rocm::LfnNoteLaunch();                             \
    kernelName<<<(numBlocks), (numThreads), (memPerBlock), (streamId)>>>(__VA_ARGS__);   \
  } while (0)

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

// ---- copied deps (SQ1): these stay in kernels.hip.cpp too. They are
// TU-local (anonymous namespace) one-paragraph helpers the moved kernels
// need; copying keeps the Flash-Next TU untouched. ----
constexpr unsigned kThreads = 256;

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float SiluF(float x) {
  return x * SigmoidF(x);
}

/// The W8A8 GEMM's activation tiles: 16 tokens x one 32-wide K block,
/// codes in fragment order (two 256-byte halves of 16 tokens x 16 codes)
/// followed by the sixteen per-token scales.
constexpr std::size_t kQ8ActTileTokens = 16;
constexpr std::size_t kQ8ActTileBytes = 576;
constexpr std::size_t kQ8ActScaleOffset = 512;

// F16 WMMA fragments (wave32): sixteen halves per lane, eight F32
// accumulators per lane.
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

__device__ __forceinline__ int32x8_t WmmaI8(int32x4_t a, int32x4_t b,
                                            int32x8_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
}

__device__ __forceinline__ float HcScaledSilu(float value) {
  float scaled = value * 0.25F;
  asm volatile("" : "+v"(scaled));
  float result = SiluF(scaled);
  asm volatile("" : "+v"(result));
  return result;
}

__device__ __forceinline__ float SsmConv4Value(float4 w, float x0, float x1,
                                               float x2, float x3) {
  float acc = __fmaf_rn(w.x, x0, __fmul_rn(w.y, x1));
  acc = __fmaf_rn(w.z, x2, acc);
  acc = __fmaf_rn(w.w, x3, acc);
  return SiluF(acc);
}
constexpr unsigned kSsmProjectionTileTokens = 32;
constexpr std::uint32_t kSsmConvTaps = 4;

// The fused projection leaves each 32-token tile's first and last three
// raw rows in qkv. Only the first three convolutions need another tile or
// the previous chunk's history; all other rows are produced in LDS.
__global__ void SsmConvBoundaryKernel(const float* qkv, const float* w,
                                      const float* history, float* out,
                                      std::uint32_t n_tokens,
                                      std::uint32_t channels,
                                      std::uint32_t stride) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t t = blockIdx.y * kSsmProjectionTileTokens + blockIdx.z;
  if (c >= channels || t >= n_tokens) {
    return;
  }
  const float4 taps = *reinterpret_cast<const float4*>(w + c * 4);
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int src = static_cast<int>(t) - 3 + j;
    v[j] = src < 0 ? history[static_cast<std::size_t>(src + 3) * channels + c]
                   : qkv[static_cast<std::size_t>(src) * stride + c];
  }
  out[static_cast<std::size_t>(t) * channels + c] =
      SsmConv4Value(taps, v[0], v[1], v[2], v[3]);
}

inline unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}


/// W8A8 WMMA GEMM over FP4 FAST weights (LFP SPEC2 item 3): the same block/
/// grid plan, LDS layout, activation side and epilogue as
/// W8A8BlockedWmmaGEMMKernel, with the weight fetch swapped. One 17-byte
/// block decodes to 32 int8 codes (exact: |code| <= 10) plus the ue4m3
/// block scale as the per-block float scale, so the int8 math matches the
/// F32 oracle up to the activation quantization, exactly like the Q8_0
/// route. FAST only: one dw per block cannot express type 100's two
/// scales (the launcher enforces it). y is [batch][m].
template<int BM, int BN, int BK, int WM, int WN, bool kFast,
         bool kHcDown = false>
__launch_bounds__(256) __global__ void Fp4Int8GemmKernel(
    const void* __restrict__ w, const void* __restrict__ x_blocks,
    std::conditional_t<kHcDown, __half, float>* __restrict__ y,
    std::size_t batch, std::size_t m, std::size_t k) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  static_assert(BN / 16 <= 8,
                "the activation stage assigns one wave per token subtile");
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;

  // One LDS plane: codes and scales of both operands; the epilogue reuses
  // the first 16 KB to transpose 16 x 32 result tiles per wave.
  constexpr int kABytes = BK * kRowTiles * 32 * 16;
  constexpr int kDwBytes = BK * kRowTiles * 16 * 4;
  constexpr int kBBytes = BK * kTokTiles * 32 * 16;
  constexpr int kDxBytes = BK * kTokTiles * 16 * 4;
  constexpr int kLdsBytes = kABytes + kDwBytes + kBBytes + kDxBytes;
  static_assert(kLdsBytes >= 8 * 512 * 4, "epilogue transposes 16 KB");
  static_assert(kWaveRowTiles % 2 == 0, "the epilogue pairs row tiles");
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_a = reinterpret_cast<int32x4_t(*)[kRowTiles][32]>(lds);
  auto* s_dw = reinterpret_cast<float (*)[kRowTiles][16]>(lds + kABytes);
  auto* s_b =
      reinterpret_cast<int32x4_t(*)[kTokTiles][32]>(lds + kABytes + kDwBytes);
  auto* s_dx = reinterpret_cast<float (*)[kTokTiles][16]>(lds + kABytes +
                                                          kDwBytes + kBBytes);

  static_assert(kFast, "one dw per block cannot express type 100");
  constexpr std::size_t kBlockBytes = 17;  // FAST blocks only (see above)
  const auto* w_bytes = static_cast<const std::uint8_t*>(w);
  const std::size_t num_blocks = k / 32;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;

  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;
  const std::size_t t_block = static_cast<std::size_t>(blockIdx.x) * BN;
  const std::size_t tt_block = t_block / kQ8ActTileTokens;

  float acc[kWaveRowTiles][kWaveTokTiles][8];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        acc[i][j][l] = 0.0F;
      }
    }
  }

  // Staging registers for the next K stage, fetched one stage ahead so the
  // scattered 34-byte block reads overlap the WMMA work.
  constexpr int kPrefetch = (BM * BK) / 256;
  int32x4_t r_q0[kPrefetch];
  int32x4_t r_q1[kPrefetch];
  float r_dw[kPrefetch];
  int32x4_t r_b[BK];
  float r_dx[BK];
  const int b_tile = wave_id;  // one wave per token subtile

  const int num_kb = static_cast<int>(num_blocks);
  const int m_i = static_cast<int>(m);
  const std::uint8_t* w_row[kPrefetch];
  float row_live[kPrefetch];
#pragma unroll
  for (int p = 0; p < kPrefetch; ++p) {
    const int r = static_cast<int>(r_block) + (((p * 256) + tid) / BK);
    const int r_clamped = (r < m_i) ? r : (m_i - 1);
    w_row[p] = w_bytes + (static_cast<std::size_t>(r_clamped) * num_blocks *
                          kBlockBytes);
    row_live[p] = (r < m_i) ? 1.0F : 0.0F;
  }
  const auto* b_base = static_cast<const std::int8_t*>(x_blocks) +
                       ((tt_block + static_cast<std::size_t>(b_tile)) *
                        num_blocks * kQ8ActTileBytes);
  const bool b_live = b_tile < kTokTiles;

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
      // One 17-byte FAST block -> 32 int8 codes + the float block scale.
      // Dead units read the last block with scale byte 0x00 = +0.
      const bool live = row_live[p] > 0.5F && kb < num_kb;
      const std::uint8_t* blk =
          w_row[p] + static_cast<std::size_t>(kb_clamped) * kBlockBytes;
      std::uint32_t packed[4];
      __builtin_memcpy(packed, blk, 16);
      std::uint32_t* q0 = reinterpret_cast<std::uint32_t*>(&r_q0[p]);
      std::uint32_t* q1 = reinterpret_cast<std::uint32_t*>(&r_q1[p]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        fp4_f16::Fp4NibblesToInt8(packed[j], q0[j]);
        fp4_f16::Fp4NibblesToInt8(packed[j] >> 4, q1[j]);
      }
      r_dw[p] = fp4_f16::Fp4ScaleByteToFloat(live ? blk[16] : 0);
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        const int kb = kb0 + i;
        const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
        const auto* tile =
            b_base + (static_cast<std::size_t>(kb_clamped) * kQ8ActTileBytes);
        r_b[i] = reinterpret_cast<const int32x4_t*>(tile)[lane_id];
        r_dx[i] = ((kb < num_kb) ? 1.0F : 0.0F) *
                  reinterpret_cast<const float*>(
                      tile + kQ8ActScaleOffset)[lane_id & 15];
      }
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * 256) + tid;
      const int rr = idx / BK;
      const int kk = idx % BK;
      const int rs = rr / 16;
      const int rl = rr % 16;
      s_a[kk][rs][rl] = r_q0[p];
      s_a[kk][rs][16 + rl] = r_q1[p];
      s_dw[kk][rs][(rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2))] = r_dw[p];
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][b_tile][lane_id] = r_b[i];
        if (lane_id < 16) {
          s_dx[i][b_tile][lane_id] = r_dx[i];
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }

#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t a0[kWaveRowTiles];
      int32x4_t a1[kWaveRowTiles];
      float dw[kWaveRowTiles][8];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int rs = (wave_row * kWaveRowTiles) + i;
        a0[i] = s_a[kb][rs][sub_lane];
        a1[i] = s_a[kb][rs][16 + sub_lane];
        const float4 lo =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][half_id * 8]);
        const float4 up =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][(half_id * 8) + 4]);
        dw[i][0] = lo.x;
        dw[i][1] = lo.y;
        dw[i][2] = lo.z;
        dw[i][3] = lo.w;
        dw[i][4] = up.x;
        dw[i][5] = up.y;
        dw[i][6] = up.z;
        dw[i][7] = up.w;
      }
      // Token-tile operands are read one tile at a time to keep the live
      // register set small enough for the BK=2 stage.
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int ts = (wave_tok * kWaveTokTiles) + j;
        const int32x4_t b0 = s_b[kb][ts][sub_lane];
        const int32x4_t b1 = s_b[kb][ts][16 + sub_lane];
        const float dx = s_dx[kb][ts][sub_lane];
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
          c = WmmaI8(a0[i], b0, c);
          c = WmmaI8(a1[i], b1, c);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc[i][j][l] += (dw[i][l] * dx) * static_cast<float>(c[l]);
          }
        }
      }
#if __clang_major__ >= 23
      // LLVM 23 hoists the next K block's operands above this one's WMMAs,
      // which spills registers on gfx1151. ROCm 7.2.3 (LLVM 22) does not, and
      // the fence costs it about 3%.
      __builtin_amdgcn_sched_barrier(0);
#endif
    }
    __syncthreads();
  }

  // Transpose the result through LDS, two row tiles at a time, so every
  // global store covers 32 consecutive rows of one token: a full 128-byte
  // line (half lines cost a read-modify-write on the fabric).
  __syncthreads();
  constexpr unsigned kOutputStride = kHcDown ? 36 : 32;
  static_assert(8 * 16 * kOutputStride * sizeof(float) <= sizeof(lds));
  float* tile_scratch =
      reinterpret_cast<float*>(lds) + wave_id * 16 * kOutputStride;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; i += 2) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      // scratch[token][row] over 16 tokens x 32 rows.
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * kOutputStride) + (2 * l) + half_id] =
            acc[i][j][l];
        tile_scratch[(sub_lane * kOutputStride) + 16 + (2 * l) + half_id] =
            acc[i + 1][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          r_block +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 =
          t_block +
          static_cast<std::size_t>((((wave_tok * kWaveTokTiles) + j) * 16));
      // Lane pair (2p, 2p + 1) stores token p's 32 rows as eight float4.
      const int tok_l = lane_id >> 1;
      const int row_l = (lane_id & 1) * 16;
      const std::size_t tok = t0 + static_cast<std::size_t>(tok_l);
      const auto* src = reinterpret_cast<const float4*>(
          tile_scratch + (tok_l * kOutputStride) + row_l);
      if (tok < batch && r0 + 32 <= m) {
        if constexpr (kHcDown) {
          auto* dst = y + tok * m + r0 + static_cast<std::size_t>(row_l);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float4 v = src[q];
            *reinterpret_cast<__half2*>(dst + 4 * q) =
                __floats2half2_rn(HcScaledSilu(v.x), HcScaledSilu(v.y));
            *reinterpret_cast<__half2*>(dst + 4 * q + 2) =
                __floats2half2_rn(HcScaledSilu(v.z), HcScaledSilu(v.w));
          }
        } else {
          auto* dst = reinterpret_cast<float4*>(
              y + (tok * m) + r0 + static_cast<std::size_t>(row_l));
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            dst[q] = src[q];
          }
        }
      } else if (tok < batch) {
#pragma unroll
        for (int q = 0; q < 16; ++q) {
          const std::size_t r = r0 + static_cast<std::size_t>(row_l + q);
          if (r < m) {
            const float value =
                tile_scratch[(tok_l * kOutputStride) + row_l + q];
            if constexpr (kHcDown)
              y[(tok * m) + r] = __float2half_rn(HcScaledSilu(value));
            else
              y[(tok * m) + r] = value;
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

}  // namespace

// Dense/split/SSM kernels at plain rocm:: scope, exactly as in base
// (base kernels.hip.cpp:30-4731 is the only anonymous block; the int8
// kernel at :3715 is inside it, dense/split/SSM/boundary are outside).
// This keeps every moved kernel symbol byte-identical (symdiff-clean).
/// Dense F16 WMMA GEMM over ROCmFP4 weights (LFP): the same block/grid plan,
/// LDS layout, swizzle, K16 product order and epilogue as DenseF16GEMMKernel,
/// with the weight fetch/decode swapped. One 32-element block is 16 code
/// bytes plus 1 (kFast, type 101) or 2 (type 100) ue4m3 scale bytes; rows are
/// 17/18 bytes per block, so block loads are memcpy (byte-granular safe on
/// unaligned type-101 rows). Fp4DecodeBlock decodes to F16 exactly as the
/// weights are committed to LDS; activations are F16 rows, accumulation F32,
/// launched on the caller's stream. Plain GEMM only: no HcMix/SsmConv/
/// attention fusions and no row groups (every FP4 mixer/attention projection
/// calls the plain Dense leg). y is [batch][m].
template<int BM, int BN, int BK, int WM, int WN, bool kFast, int kRowGroup = 1>
__launch_bounds__(256) __global__ void Fp4DenseF16GEMMKernel(
    const void* __restrict__ w, const __half* __restrict__ x,
    float* __restrict__ y, std::size_t batch, std::size_t m, std::size_t k) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;
  // Weight and activation K blocks staged per thread; a unit past the
  // stage's block count is idle.
  constexpr int kAUnits = BM * BK;
  constexpr int kBUnits = BN * BK;
  constexpr int kAPer = (kAUnits + 255) / 256;
  constexpr int kBPer = (kBUnits + 255) / 256;

  // One K block of one row or token is four 16-byte chunks; the chunks of
  // nearby rows are permuted so a fragment read (one row per lane, 64-byte
  // stride) covers all bank groups.
  constexpr int kLdsChunks = BK * (BM + BN) * 4;
  static_assert(kLdsChunks * 16 >= 8 * 512 * 4, "epilogue transposes 16 KB");
  constexpr int kTransposeChunks = 8 * 16 * 36 * sizeof(float) / sizeof(uint4);
  constexpr int kLdsStorage =
      kLdsChunks < kTransposeChunks ? kTransposeChunks : kLdsChunks;
  __shared__ __attribute__((aligned(16))) uint4 s_lds[kLdsStorage];
  auto* s_a = reinterpret_cast<uint4(*)[BM][4]>(s_lds);
  auto* s_b = reinterpret_cast<uint4(*)[BN][4]>(s_lds + (BK * BM * 4));
  const auto swizzle = [](int row, int c) { return c ^ ((row >> 1) & 3); };

  constexpr std::size_t kBlockBytes = kFast ? 17 : 18;
  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const auto* w_bytes = static_cast<const std::uint8_t*>(w);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;
  // #421 row-group launch order, same as DenseF16GEMMKernel: the tile
  // grid's row count is divisible by kRowGroup; every dot product keeps
  // its K order, so this is numerics-neutral.
  const unsigned row_group = blockIdx.y / kRowGroup;
  const unsigned within = (blockIdx.y % kRowGroup) * gridDim.x + blockIdx.x;
  const int r_block = (row_group * kRowGroup + within % kRowGroup) * BM;
  const int t_block = (within / kRowGroup) * BN;

  // Weight fetch unit p of a thread: row (p * 256 + tid) / BK, K block
  // (p * 256 + tid) % BK of the stage; rows past m read the last row with
  // a zero scale (scale byte 0x00 decodes to +0, and code 8's -0 times +0
  // plus +0 is +0, so dead units contribute exact zeros).
  const std::uint8_t* a_ptr[kAPer];
  bool a_live[kAPer];
#pragma unroll
  for (int p = 0; p < kAPer; ++p) {
    const int idx = (p * 256) + tid;
    const int r = r_block + (idx / BK);
    a_live[p] = idx < kAUnits && r < m_i;
    a_ptr[p] = w_bytes +
               static_cast<std::size_t>(a_live[p] ? r : (m_i - 1)) *
                   static_cast<std::size_t>(num_kb) * kBlockBytes;
  }
  const __half* b_ptr[kBPer];
#pragma unroll
  for (int p = 0; p < kBPer; ++p) {
    const int idx = (p * 256) + tid;
    const int t = t_block + (idx / BK);
    b_ptr[p] = idx < kBUnits && t < static_cast<int>(batch)
                   ? x + (static_cast<std::size_t>(t) * k)
                   : nullptr;
  }
  std::uint32_t a_packed[kAPer][4];
  std::uint8_t a_s0[kAPer];
  std::uint8_t a_s1[kAPer];
  uint4 b_data[kBPer][4];

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const bool live = a_live[p] && kb < num_kb;
      const std::uint8_t* blk =
          a_ptr[p] +
          (static_cast<std::size_t>(live ? kb : (num_kb - 1)) * kBlockBytes);
      __builtin_memcpy(a_packed[p], blk, 16);
      a_s0[p] = live ? blk[16] : 0;
      a_s1[p] = live ? (kFast ? blk[16] : blk[17]) : 0;
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      if (b_ptr[p] != nullptr && kb < num_kb) {
        const auto* src = reinterpret_cast<const uint4*>(b_ptr[p] + (kb * 32));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = src[c];
        }
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kAUnits) {
        break;
      }
      const int row = idx / BK;
      const int kk = idx % BK;
      __half2 h[16];
      fp4_f16::Fp4DecodeBlock(a_packed[p], a_s0[p], a_s1[p], h);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        uint4 v;
        __builtin_memcpy(&v, &h[4 * c], 16);
        s_a[kk][row][swizzle(row, c)] = v;
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kBUnits) {
        break;
      }
      const int t = idx / BK;
      const int kk = idx % BK;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        s_b[kk][t][swizzle(t, c)] = b_data[p][c];
      }
    }
  };

  v8f acc[kWaveRowTiles][kWaveTokTiles];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      acc[i][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    // Keeps the next stage's loads ahead of the matrix work: scheduled
    // freely, the compiler sinks them below it and the commit waits on
    // the full memory latency.
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int row = (((wave_row * kWaveRowTiles) + i) * 16) + sub_lane;
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          c[q] = s_a[kb][row][swizzle(row, q)];
        }
        __builtin_memcpy(&a_lo[i], &c[0], 32);
        __builtin_memcpy(&a_hi[i], &c[2], 32);
      }
      // SPL: fence the a-load / b-load / WMMA phases so the scheduler
      // cannot overlap their LDS-read temporaries and fragment pairs
      // across the unrolled j copies (that overlap spilled 12-13 VGPRs
      // to scratch at 256 VGPR). Scheduling only: the operations and
      // their order per output element are unchanged.
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int t = (((wave_tok * kWaveTokTiles) + j) * 16) + sub_lane;
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          c[q] = s_b[kb][t][swizzle(t, q)];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &c[0], 32);
        __builtin_memcpy(&b_hi, &c[2], 32);
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
          acc[i][j] = Wmma(a_hi[i], b_hi, acc[i][j]);
        }
        __builtin_amdgcn_sched_barrier(0);
      }
    }
    __syncthreads();
  }

  // Transpose the result through LDS, two row tiles at a time, so every
  // global store covers 32 consecutive rows of one token: a full 128-byte
  // line (half lines cost a read-modify-write on the fabric).
  // Four padding floats reduce scatter bank conflicts and retain float4
  // alignment for both output stores and the fused convolution's reads.
  constexpr unsigned kOutputStride = 36;
  constexpr unsigned kOutputTokens = 16;
  static_assert(8 * kOutputTokens * kOutputStride * sizeof(float) <=
                sizeof(s_lds));
  static_assert(kWaveRowTiles % 2 == 0, "the epilogue pairs row tiles");
  float* tile_scratch =
      reinterpret_cast<float*>(s_lds) + wave_id * kOutputTokens * kOutputStride;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; i += 2) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      // scratch[token][row], with 32 output rows per token.
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * kOutputStride) + (2 * l) + half_id] =
            acc[i][j][l];
        tile_scratch[(sub_lane * kOutputStride) + 16 + (2 * l) + half_id] =
            acc[i + 1][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          static_cast<std::size_t>(r_block) +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 =
          static_cast<std::size_t>(t_block) +
          static_cast<std::size_t>((((wave_tok * kWaveTokTiles) + j) * 16));
      // Lane pair (2p, 2p + 1) stores token p's 32 rows as eight float4.
      const int tok_l = lane_id >> 1;
      const int row_l = (lane_id & 1) * 16;
      const std::size_t tok = t0 + static_cast<std::size_t>(tok_l);
      const auto* src = reinterpret_cast<const float4*>(
          tile_scratch + (tok_l * kOutputStride) + row_l);
      if (tok < batch && r0 + 32 <= m && ((tok * m) % 4 == 0)) {
        auto* dst = reinterpret_cast<float4*>(
            y + (tok * m) + r0 + static_cast<std::size_t>(row_l));
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          dst[q] = src[q];
        }
      } else if (tok < batch) {
#pragma unroll
        for (int q = 0; q < 16; ++q) {
          const std::size_t r = r0 + static_cast<std::size_t>(row_l + q);
          if (r < m) {
            y[(tok * m) + r] =
                tile_scratch[(tok_l * kOutputStride) + row_l + q];
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

/// Split-output FP4 GEMM (LFP SPEC2 item 1b): the same tile, fetch and
/// transpose as Fp4DenseF16GEMMKernel, but rows [h*512, h*512+256) store to
/// `q` and rows [h*512+256, h*512+512) to `gate` (UnpackQGateKernel's
/// per-head interleave), so the separate unpack disappears. 16-row halves
/// never straddle the 256-row boundary (r0+row_l is 16-aligned), so each
/// half stores coalesced through one pointer + offset. q/gate are
/// [batch][m/2]. (8,1) tile only.
// Split-output addressing for Fp4SplitF16GemmKernel (device only).
__device__ __forceinline__ float* Fp4SplitOut(float* q, float* gate,
                                             std::size_t tok, std::size_t r,
                                             std::size_t width) {
  const std::size_t h = r / 512;
  const std::size_t j = r % 512;
  const std::size_t off =
      tok * width + h * 256 + (j < 256 ? j : j - 256);
  return (j < 256 ? q : gate) + off;
}

template<int BM, int BN, int BK, int WM, int WN, bool kFast>
__launch_bounds__(256) __global__ void Fp4SplitF16GemmKernel(
    const void* __restrict__ w, const __half* __restrict__ x,
    float* __restrict__ q, float* __restrict__ gate, std::size_t batch,
    std::size_t m, std::size_t k) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;
  // Weight and activation K blocks staged per thread; a unit past the
  // stage's block count is idle.
  constexpr int kAUnits = BM * BK;
  constexpr int kBUnits = BN * BK;
  constexpr int kAPer = (kAUnits + 255) / 256;
  constexpr int kBPer = (kBUnits + 255) / 256;

  // One K block of one row or token is four 16-byte chunks; the chunks of
  // nearby rows are permuted so a fragment read (one row per lane, 64-byte
  // stride) covers all bank groups.
  constexpr int kLdsChunks = BK * (BM + BN) * 4;
  static_assert(kLdsChunks * 16 >= 8 * 512 * 4, "epilogue transposes 16 KB");
  constexpr int kTransposeChunks = 8 * 16 * 36 * sizeof(float) / sizeof(uint4);
  constexpr int kLdsStorage =
      kLdsChunks < kTransposeChunks ? kTransposeChunks : kLdsChunks;
  __shared__ __attribute__((aligned(16))) uint4 s_lds[kLdsStorage];
  auto* s_a = reinterpret_cast<uint4(*)[BM][4]>(s_lds);
  auto* s_b = reinterpret_cast<uint4(*)[BN][4]>(s_lds + (BK * BM * 4));
  const auto swizzle = [](int row, int c) { return c ^ ((row >> 1) & 3); };

  constexpr std::size_t kBlockBytes = kFast ? 17 : 18;
  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const auto* w_bytes = static_cast<const std::uint8_t*>(w);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;
  const int r_block = static_cast<int>(blockIdx.y) * BM;
  const int t_block = static_cast<int>(blockIdx.x) * BN;

  // Weight fetch unit p of a thread: row (p * 256 + tid) / BK, K block
  // (p * 256 + tid) % BK of the stage; rows past m read the last row with
  // a zero scale (scale byte 0x00 decodes to +0, and code 8's -0 times +0
  // plus +0 is +0, so dead units contribute exact zeros).
  const std::uint8_t* a_ptr[kAPer];
  bool a_live[kAPer];
#pragma unroll
  for (int p = 0; p < kAPer; ++p) {
    const int idx = (p * 256) + tid;
    const int r = r_block + (idx / BK);
    a_live[p] = idx < kAUnits && r < m_i;
    a_ptr[p] = w_bytes +
               static_cast<std::size_t>(a_live[p] ? r : (m_i - 1)) *
                   static_cast<std::size_t>(num_kb) * kBlockBytes;
  }
  const __half* b_ptr[kBPer];
#pragma unroll
  for (int p = 0; p < kBPer; ++p) {
    const int idx = (p * 256) + tid;
    const int t = t_block + (idx / BK);
    b_ptr[p] = idx < kBUnits && t < static_cast<int>(batch)
                   ? x + (static_cast<std::size_t>(t) * k)
                   : nullptr;
  }
  std::uint32_t a_packed[kAPer][4];
  std::uint8_t a_s0[kAPer];
  std::uint8_t a_s1[kAPer];
  uint4 b_data[kBPer][4];

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const bool live = a_live[p] && kb < num_kb;
      const std::uint8_t* blk =
          a_ptr[p] +
          (static_cast<std::size_t>(live ? kb : (num_kb - 1)) * kBlockBytes);
      __builtin_memcpy(a_packed[p], blk, 16);
      a_s0[p] = live ? blk[16] : 0;
      a_s1[p] = live ? (kFast ? blk[16] : blk[17]) : 0;
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      if (b_ptr[p] != nullptr && kb < num_kb) {
        const auto* src = reinterpret_cast<const uint4*>(b_ptr[p] + (kb * 32));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = src[c];
        }
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kAUnits) {
        break;
      }
      const int row = idx / BK;
      const int kk = idx % BK;
      __half2 h[16];
      fp4_f16::Fp4DecodeBlock(a_packed[p], a_s0[p], a_s1[p], h);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        uint4 v;
        __builtin_memcpy(&v, &h[4 * c], 16);
        s_a[kk][row][swizzle(row, c)] = v;
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kBUnits) {
        break;
      }
      const int t = idx / BK;
      const int kk = idx % BK;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        s_b[kk][t][swizzle(t, c)] = b_data[p][c];
      }
    }
  };

  v8f acc[kWaveRowTiles][kWaveTokTiles];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      acc[i][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    // Keeps the next stage's loads ahead of the matrix work: scheduled
    // freely, the compiler sinks them below it and the commit waits on
    // the full memory latency.
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int row = (((wave_row * kWaveRowTiles) + i) * 16) + sub_lane;
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          c[q] = s_a[kb][row][swizzle(row, q)];
        }
        __builtin_memcpy(&a_lo[i], &c[0], 32);
        __builtin_memcpy(&a_hi[i], &c[2], 32);
      }
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int t = (((wave_tok * kWaveTokTiles) + j) * 16) + sub_lane;
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          c[q] = s_b[kb][t][swizzle(t, q)];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &c[0], 32);
        __builtin_memcpy(&b_hi, &c[2], 32);
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
          acc[i][j] = Wmma(a_hi[i], b_hi, acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  // Transpose the result through LDS, two row tiles at a time, so every
  // global store covers 32 consecutive rows of one token: a full 128-byte
  // line (half lines cost a read-modify-write on the fabric).
  // Four padding floats reduce scatter bank conflicts and retain float4
  // alignment for both output stores and the fused convolution's reads.
  constexpr unsigned kOutputStride = 36;
  constexpr unsigned kOutputTokens = 16;
  static_assert(8 * kOutputTokens * kOutputStride * sizeof(float) <=
                sizeof(s_lds));
  static_assert(kWaveRowTiles % 2 == 0, "the epilogue pairs row tiles");
  float* tile_scratch =
      reinterpret_cast<float*>(s_lds) + wave_id * kOutputTokens * kOutputStride;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; i += 2) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      // scratch[token][row], with 32 output rows per token.
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * kOutputStride) + (2 * l) + half_id] =
            acc[i][j][l];
        tile_scratch[(sub_lane * kOutputStride) + 16 + (2 * l) + half_id] =
            acc[i + 1][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          static_cast<std::size_t>(r_block) +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 =
          static_cast<std::size_t>(t_block) +
          static_cast<std::size_t>((((wave_tok * kWaveTokTiles) + j) * 16));
      // Lane pair (2p, 2p + 1) stores token p's 32 rows as eight float4.
      const int tok_l = lane_id >> 1;
      const int row_l = (lane_id & 1) * 16;
      const std::size_t tok = t0 + static_cast<std::size_t>(tok_l);
      const std::size_t width = m / 2;
      const auto* src = reinterpret_cast<const float4*>(
          tile_scratch + (tok_l * kOutputStride) + row_l);
      if (tok < batch && r0 + 32 <= m) {
        // Each lane stores its 16-row half (the pair covers 32 rows); a
        // 16-row half never straddles the 256-row q|gate boundary (ro is
        // 16-aligned), so one pointer + offset stays coalesced and all
        // offsets are multiples of 16 (float4-aligned).
        const std::size_t ro = r0 + row_l;
        auto* f0 = reinterpret_cast<float4*>(
            Fp4SplitOut(q, gate, tok, ro, width));
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          f0[v] = src[v];
        }
      } else if (tok < batch) {
        for (int v = 0; v < 16; ++v) {
          const std::size_t r = r0 + row_l + v;
          if (r >= m) {
            continue;
          }
          const float value =
              tile_scratch[(tok_l * kOutputStride) + row_l + v];
          *Fp4SplitOut(q, gate, tok, r, width) = value;
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

/// Fused SSM projection over split FP4 weights (LFP SPEC2 item 1b): rows
/// [0,10240) read the qkv tensor, rows [10240,16384) the gate tensor — one
/// activation pass, like the stacked Q8_0 ssm_in — with the SsmConv epilogue
/// copied from DenseF16GEMMKernel, so the separate SsmConv4Kernel
/// disappears. Otherwise the same block/grid plan, LDS layout, swizzle,
/// K16 product order as Fp4DenseF16GEMMKernel. y is [batch][m] (raw rows),
/// conv_out is [batch][10240] (convolved qkv). (8,1) tile only.
template<int BM, int BN, int BK, int WM, int WN, bool kFastQkv,
         bool kFastGate>
__launch_bounds__(256) __global__ void Fp4SsmF16GemmKernel(
    const void* __restrict__ w0, const void* __restrict__ w1,
    const __half* __restrict__ x, float* __restrict__ y, std::size_t batch,
    std::size_t m, std::size_t k, const float* __restrict__ conv_w,
    float* __restrict__ conv_out) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;
  // Weight and activation K blocks staged per thread; a unit past the
  // stage's block count is idle.
  constexpr int kAUnits = BM * BK;
  constexpr int kBUnits = BN * BK;
  constexpr int kAPer = (kAUnits + 255) / 256;
  constexpr int kBPer = (kBUnits + 255) / 256;

  // One K block of one row or token is four 16-byte chunks; the chunks of
  // nearby rows are permuted so a fragment read (one row per lane, 64-byte
  // stride) covers all bank groups.
  constexpr int kLdsChunks = BK * (BM + BN) * 4;
  static_assert(kLdsChunks * 16 >= 8 * 512 * 4, "epilogue transposes 16 KB");
  constexpr int kTransposeChunks = 8 * 16 * 36 * sizeof(float) / sizeof(uint4);
  constexpr int kLdsStorage =
      kLdsChunks < kTransposeChunks ? kTransposeChunks : kLdsChunks;
  __shared__ __attribute__((aligned(16))) uint4 s_lds[kLdsStorage];
  auto* s_a = reinterpret_cast<uint4(*)[BM][4]>(s_lds);
  auto* s_b = reinterpret_cast<uint4(*)[BN][4]>(s_lds + (BK * BM * 4));
  const auto swizzle = [](int row, int c) { return c ^ ((row >> 1) & 3); };

  static_assert(BM == 256 && BN == 128 && BK == 2 && WM == 8 && WN == 1,
                "the SSM fusion instantiates the (8,1) tile only");
  constexpr std::size_t kSplit = 10240;  // qkv rows; gate follows
  constexpr std::size_t kTszQkv = kFastQkv ? 17 : 18;
  constexpr std::size_t kTszGate = kFastGate ? 17 : 18;
  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const auto* w0_bytes = static_cast<const std::uint8_t*>(w0);
  const auto* w1_bytes = static_cast<const std::uint8_t*>(w1);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;
  const int r_block = static_cast<int>(blockIdx.y) * BM;
  const int t_block = static_cast<int>(blockIdx.x) * BN;

  // Weight fetch unit p of a thread: row (p * 256 + tid) / BK, K block
  // (p * 256 + tid) % BK of the stage; rows past m read the last row with
  // a zero scale (scale byte 0x00 decodes to +0, and code 8's -0 times +0
  // plus +0 is +0, so dead units contribute exact zeros).
  const std::uint8_t* a_ptr[kAPer];
  bool a_live[kAPer];
  bool a_qkv[kAPer];
#pragma unroll
  for (int p = 0; p < kAPer; ++p) {
    const int idx = (p * 256) + tid;
    const int r = r_block + (idx / BK);
    a_live[p] = idx < kAUnits && r < m_i;
    const int rc = a_live[p] ? r : (m_i - 1);
    a_qkv[p] = static_cast<std::size_t>(rc) < kSplit;
    const std::size_t ro = a_qkv[p] ? static_cast<std::size_t>(rc)
                                    : static_cast<std::size_t>(rc) - kSplit;
    a_ptr[p] = (a_qkv[p] ? w0_bytes : w1_bytes) + ro *
                   static_cast<std::size_t>(num_kb) *
                   (a_qkv[p] ? kTszQkv : kTszGate);
  }
  const __half* b_ptr[kBPer];
#pragma unroll
  for (int p = 0; p < kBPer; ++p) {
    const int idx = (p * 256) + tid;
    const int t = t_block + (idx / BK);
    b_ptr[p] = idx < kBUnits && t < static_cast<int>(batch)
                   ? x + (static_cast<std::size_t>(t) * k)
                   : nullptr;
  }
  std::uint32_t a_packed[kAPer][4];
  std::uint8_t a_s0[kAPer];
  std::uint8_t a_s1[kAPer];
  uint4 b_data[kBPer][4];

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const bool live = a_live[p] && kb < num_kb;
      const std::size_t tsz = a_qkv[p] ? kTszQkv : kTszGate;
      const std::uint8_t* blk =
          a_ptr[p] + (static_cast<std::size_t>(live ? kb : (num_kb - 1)) * tsz);
      __builtin_memcpy(a_packed[p], blk, 16);
      a_s0[p] = live ? blk[16] : 0;
      const int s1off = (a_qkv[p] ? kFastQkv : kFastGate) ? 16 : 17;
      a_s1[p] = live ? blk[s1off] : 0;
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      if (b_ptr[p] != nullptr && kb < num_kb) {
        const auto* src = reinterpret_cast<const uint4*>(b_ptr[p] + (kb * 32));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = src[c];
        }
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kAUnits) {
        break;
      }
      const int row = idx / BK;
      const int kk = idx % BK;
      __half2 h[16];
      fp4_f16::Fp4DecodeBlock(a_packed[p], a_s0[p], a_s1[p], h);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        uint4 v;
        __builtin_memcpy(&v, &h[4 * c], 16);
        s_a[kk][row][swizzle(row, c)] = v;
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kBUnits) {
        break;
      }
      const int t = idx / BK;
      const int kk = idx % BK;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        s_b[kk][t][swizzle(t, c)] = b_data[p][c];
      }
    }
  };

  v8f acc[kWaveRowTiles][kWaveTokTiles];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      acc[i][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    // Keeps the next stage's loads ahead of the matrix work: scheduled
    // freely, the compiler sinks them below it and the commit waits on
    // the full memory latency.
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int row = (((wave_row * kWaveRowTiles) + i) * 16) + sub_lane;
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          c[q] = s_a[kb][row][swizzle(row, q)];
        }
        __builtin_memcpy(&a_lo[i], &c[0], 32);
        __builtin_memcpy(&a_hi[i], &c[2], 32);
      }
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int t = (((wave_tok * kWaveTokTiles) + j) * 16) + sub_lane;
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          c[q] = s_b[kb][t][swizzle(t, q)];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &c[0], 32);
        __builtin_memcpy(&b_hi, &c[2], 32);
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
          acc[i][j] = Wmma(a_hi[i], b_hi, acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  // Transpose the result through LDS, two row tiles at a time, so every
  // global store covers 32 consecutive rows of one token: a full 128-byte
  // line (half lines cost a read-modify-write on the fabric).
  // Four padding floats reduce scatter bank conflicts and retain float4
  // alignment for both output stores and the fused convolution's reads.
  constexpr unsigned kOutputStride = 36;
  constexpr unsigned kOutputTokens = kSsmProjectionTileTokens;
  constexpr unsigned kOutputGroups = kOutputTokens / 16;
  static_assert(8 * kOutputTokens * kOutputStride * sizeof(float) <=
                sizeof(s_lds));
  static_assert(kWaveRowTiles % 2 == 0, "the epilogue pairs row tiles");
  float* tile_scratch =
      reinterpret_cast<float*>(s_lds) + wave_id * kOutputTokens * kOutputStride;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; i += 2) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; j += kOutputGroups) {
#pragma unroll
      for (unsigned group = 0; group < kOutputGroups; ++group) {
        // scratch[token][row], with 32 output rows per token.
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          tile_scratch[((sub_lane + group * 16) * kOutputStride) + (2 * l) +
                       half_id] = acc[i][j + group][l];
          tile_scratch[((sub_lane + group * 16) * kOutputStride) + 16 +
                       (2 * l) + half_id] = acc[i + 1][j + group][l];
        }
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          static_cast<std::size_t>(r_block) +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 =
          static_cast<std::size_t>(t_block) +
          static_cast<std::size_t>((((wave_tok * kWaveTokTiles) + j) * 16));
#pragma unroll
      for (unsigned group = 0; group < kOutputGroups; ++group) {
        // Lane pair (2p, 2p + 1) stores token p's 32 rows as eight float4.
        const int tok_l = (lane_id >> 1) + group * 16;
        const int row_l = (lane_id & 1) * 16;
        const std::size_t tok = t0 + static_cast<std::size_t>(tok_l);
        const auto* src = reinterpret_cast<const float4*>(
            tile_scratch + (tok_l * kOutputStride) + row_l);
        constexpr std::uint32_t channels = 10240;
        if (tok < batch) {
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            const unsigned row = r0 + row_l + v * 4;
            if (row >= m)
              continue;
            const float4 current = src[v];
            if (row >= channels || tok_l < 3 || tok_l >= kOutputTokens - 3 ||
                tok + 3 >= batch) {
              *reinterpret_cast<float4*>(y + tok * m + row) = current;
            }
            if (row < channels && tok_l >= 3) {
              const float4 x0 = *reinterpret_cast<const float4*>(
                  tile_scratch + (tok_l - 3) * kOutputStride + row_l + v * 4);
              const float4 x1 = *reinterpret_cast<const float4*>(
                  tile_scratch + (tok_l - 2) * kOutputStride + row_l + v * 4);
              const float4 x2 = *reinterpret_cast<const float4*>(
                  tile_scratch + (tok_l - 1) * kOutputStride + row_l + v * 4);
              const float4 w0 =
                  *reinterpret_cast<const float4*>(conv_w + (row + 0) * 4);
              const float4 w1 =
                  *reinterpret_cast<const float4*>(conv_w + (row + 1) * 4);
              const float4 w2 =
                  *reinterpret_cast<const float4*>(conv_w + (row + 2) * 4);
              const float4 w3 =
                  *reinterpret_cast<const float4*>(conv_w + (row + 3) * 4);
              const float4 value{
                  SsmConv4Value(w0, x0.x, x1.x, x2.x, current.x),
                  SsmConv4Value(w1, x0.y, x1.y, x2.y, current.y),
                  SsmConv4Value(w2, x0.z, x1.z, x2.z, current.z),
                  SsmConv4Value(w3, x0.w, x1.w, x2.w, current.w)};
              *reinterpret_cast<float4*>(conv_out + tok * channels + row) =
                  value;
            }
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

bool Fp4Int8Gemm(const void* w, const void* x_tiled, float* out,
                 std::size_t batch, std::size_t m, std::size_t k, bool fast,
                 hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || k % 32 != 0 || !fast) {
    return false;  // FAST (type 101) only: one scale per block
  }
  // The same tile plan as W8A8Gemm (the activation side is identical).
  constexpr int kBM = 128;
  if (m <= 512 && batch >= 96) {
    constexpr int kBN = 128;
    constexpr int kNarrowBM = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kNarrowBM - 1) / kNarrowBM));
    hipLaunchKernelGGL((Fp4Int8GemmKernel<kNarrowBM, kBN, 4, 2, 4, true>),
                       grid, dim3(kThreads), 0, stream, w, x_tiled, out, batch,
                       m, k);
  } else if (batch >= 96) {
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((Fp4Int8GemmKernel<kBM, kBN, 2, 4, 2, true>), grid,
                       dim3(kThreads), 0, stream, w, x_tiled, out, batch, m, k);
  } else {
    constexpr int kBN = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((Fp4Int8GemmKernel<kBM, kBN, 4, 4, 2, true>), grid,
                       dim3(kThreads), 0, stream, w, x_tiled, out, batch, m, k);
  }
  return true;
}

template<int BM, int BN, int BK, int WM, int WN, int kRowGroup = 1>
bool LaunchFp4DenseF16(const void* w, const __half* x, float* out,
                       std::size_t batch, std::size_t m, std::size_t k,
                       bool fast, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned int>((batch + BN - 1) / BN),
                  static_cast<unsigned int>((m + BM - 1) / BM));
  if (fast) {
    hipLaunchKernelGGL((Fp4DenseF16GEMMKernel<BM, BN, BK, WM, WN, true, kRowGroup>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  } else {
    hipLaunchKernelGGL((Fp4DenseF16GEMMKernel<BM, BN, BK, WM, WN, false, kRowGroup>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  }
  return true;
}

bool Fp4DenseF16Gemm(const void* w, const __half* x, float* out,
                     std::size_t batch, std::size_t m, std::size_t k, bool fast,
                     hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || k % 32 != 0) {
    return false;
  }
  // LFP final (bench ~/ls/lfp/out-round3.txt): the (8, 1) variant hides the
  // heavier FP4 decode best on every wide shape (6144x2560 1.45x on (4, 2)
  // vs 1.00-1.06x on (8, 1); 2560x2560 1.29x vs 0.99x), so it covers all
  // wide FP4 shapes from batch >= 96; the narrow tile covers m <= 1024
  // (640x2560 1.47x wide vs 1.13x narrow; narrow is also faster than Q8_0's
  // wide choice there). The shallow-K 10240x320 stays on (8, 1) as a
  // backstop (1.35-1.47x; the executor keeps it on the W8A8 route, which
  // wins). Plain row mapping throughout (no HcMix interleave: the FP4 mixer
  // calls the plain Dense leg).
  // #421 (stock DenseF16Gemm): row-group launch order on the two m==2560
  // prefill shapes (output projection group 5, shared-expert down group 2).
  // Same tiles as the wide branch, only the order changes.
  if (batch >= 1024 && m == 2560 && k == 6144) {
    return LaunchFp4DenseF16<256, 128, 2, 8, 1, 5>(w, x, out, batch, m, k,
                                                   fast, stream);
  }
  if (batch >= 1024 && m == 2560) {
    return LaunchFp4DenseF16<256, 128, 1, 4, 2, 2>(w, x, out, batch, m, k,
                                                   fast, stream);
  }
  if (m <= 1024 && batch >= 96) {
    return LaunchFp4DenseF16<64, 128, 2, 2, 4>(w, x, out, batch, m, k, fast,
                                               stream);
  }
  if (batch >= 96) {
    if (m >= 2560 && (k == 320 || k == 640 || k == 2560 || k == 6144)) {
      // Eight row groups reuse each weight fragment across all token tiles
      // and keep fewer weight fragments live. K accumulation is unchanged.
      return LaunchFp4DenseF16<256, 128, 2, 8, 1>(w, x, out, batch, m, k, fast,
                                                  stream);
    }
    return LaunchFp4DenseF16<256, 128, 1, 4, 2>(w, x, out, batch, m, k, fast,
                                                stream);
  }
  return LaunchFp4DenseF16<128, 64, 4, 4, 2>(w, x, out, batch, m, k, fast,
                                             stream);
}



bool Fp4SplitF16Gemm(const void* w, const __half* x, float* q, float* gate,
                     std::size_t batch, std::size_t m, std::size_t k,
                     bool fast, hipStream_t stream) {
  if (batch < 96 || m != 12288 || k != 2560) {
    return false;
  }
  const dim3 grid(static_cast<unsigned int>((batch + 127) / 128),
                  static_cast<unsigned int>(m / 256));
  if (fast) {
    hipLaunchKernelGGL((Fp4SplitF16GemmKernel<256, 128, 2, 8, 1, true>), grid,
                       dim3(kThreads), 0, stream, w, x, q, gate, batch, m, k);
  } else {
    hipLaunchKernelGGL((Fp4SplitF16GemmKernel<256, 128, 2, 8, 1, false>), grid,
                       dim3(kThreads), 0, stream, w, x, q, gate, batch, m, k);
  }
  return true;
}

bool Fp4SsmF16Gemm(const void* w0, const void* w1, const __half* x,
                   const float* conv_w, const float* history, float* qkvz,
                   float* convolved, std::uint32_t n_tokens, std::uint32_t m,
                   std::uint32_t k, std::uint32_t channels,
                   std::uint32_t kernel, hipStream_t stream) {
  if (n_tokens < 1024 || m != 16384 || k != 2560 || channels != 10240 ||
      kernel != kSsmConvTaps) {
    return false;
  }
  hipLaunchKernelGGL(
      (Fp4SsmF16GemmKernel<256, 128, 2, 8, 1, false, true>),
      dim3((n_tokens + 127) / 128, m / 256), dim3(kThreads), 0, stream, w0,
      w1, x, qkvz, n_tokens, m, k, conv_w, convolved);
  hipLaunchKernelGGL(
      SsmConvBoundaryKernel,
      dim3(Blocks(channels),
           (n_tokens + kSsmProjectionTileTokens - 1) / kSsmProjectionTileTokens,
           3),
      dim3(kThreads), 0, stream, qkvz, conv_w, history, convolved, n_tokens,
      channels, m);
  return true;
}


// SQ1 occupancy proof (rd1 section 2.4): every instantiation must hold >= 2
// waves/SIMD (one wave issues WMMA at half rate). 256-thread blocks are 8
// wave32 waves; a gfx1151 CU has 4 SIMDs; waves/SIMD = blocks * 2.
int Fp4F16PrefillMinWavesPerSimd(char* report, std::size_t report_bytes) {
  struct Entry {
    const char* name;
    const void* func;
  };
  const Entry entries[] = {
      {"dense<64,128,2,2,4,f>", (const void*)Fp4DenseF16GEMMKernel<64, 128, 2, 2, 4, false>},
      {"dense<64,128,2,2,4,t>", (const void*)Fp4DenseF16GEMMKernel<64, 128, 2, 2, 4, true>},
      {"dense<256,128,2,8,1,f>", (const void*)Fp4DenseF16GEMMKernel<256, 128, 2, 8, 1, false>},
      {"dense<256,128,2,8,1,t>", (const void*)Fp4DenseF16GEMMKernel<256, 128, 2, 8, 1, true>},
      {"dense<256,128,1,4,2,f>", (const void*)Fp4DenseF16GEMMKernel<256, 128, 1, 4, 2, false>},
      {"dense<256,128,1,4,2,t>", (const void*)Fp4DenseF16GEMMKernel<256, 128, 1, 4, 2, true>},
      {"dense<128,64,4,4,2,f>", (const void*)Fp4DenseF16GEMMKernel<128, 64, 4, 4, 2, false>},
      {"dense<128,64,4,4,2,t>", (const void*)Fp4DenseF16GEMMKernel<128, 64, 4, 4, 2, true>},
      {"dense<256,128,2,8,1,5,f>", (const void*)Fp4DenseF16GEMMKernel<256, 128, 2, 8, 1, false, 5>},
      {"dense<256,128,2,8,1,5,t>", (const void*)Fp4DenseF16GEMMKernel<256, 128, 2, 8, 1, true, 5>},
      {"dense<256,128,1,4,2,2,f>", (const void*)Fp4DenseF16GEMMKernel<256, 128, 1, 4, 2, false, 2>},
      {"dense<256,128,1,4,2,2,t>", (const void*)Fp4DenseF16GEMMKernel<256, 128, 1, 4, 2, true, 2>},
      {"split<256,128,2,8,1,f>", (const void*)Fp4SplitF16GemmKernel<256, 128, 2, 8, 1, false>},
      {"split<256,128,2,8,1,t>", (const void*)Fp4SplitF16GemmKernel<256, 128, 2, 8, 1, true>},
      {"ssm<256,128,2,8,1,f,t>", (const void*)Fp4SsmF16GemmKernel<256, 128, 2, 8, 1, false, true>},
      {"ssm-boundary", (const void*)SsmConvBoundaryKernel},
      {"int8<64,128,4,2,4>", (const void*)Fp4Int8GemmKernel<64, 128, 4, 2, 4, true>},
      {"int8<128,128,2,4,2>", (const void*)Fp4Int8GemmKernel<128, 128, 2, 4, 2, true>},
      {"int8<128,64,4,4,2>", (const void*)Fp4Int8GemmKernel<128, 64, 4, 4, 2, true>},
  };
  if (report != nullptr && report_bytes > 0) report[0] = '\0';
  int min_waves = 1000000;
  for (const Entry& e : entries) {
    int blocks = 0;
    const hipError_t rc = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, e.func, 256, 0);
    if (rc != hipSuccess || blocks <= 0) {
      if (report != nullptr && report_bytes > 0) {
        std::snprintf(report, report_bytes, "OCCUPANCY-QUERY-FAIL %s rc=%d\n",
                      e.name, static_cast<int>(rc));
      }
      return -1;
    }
    const int waves = blocks * 200;  // blocks * 8 waves / 4 SIMD, x100
    if (waves < min_waves) min_waves = waves;
    if (report != nullptr && report_bytes > 0) {
      char line[128];
      std::snprintf(line, sizeof(line), "%s blocks=%d waves_per_simd=%.2f\n",
                    e.name, blocks, waves / 100.0);
      std::strncat(report, line, report_bytes - std::strlen(report) - 1);
    }
  }
  return min_waves;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
