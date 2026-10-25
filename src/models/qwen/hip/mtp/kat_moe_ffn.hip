#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/mtp/kat_moe_ffn.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/models/qwen/hip/detail/kv_f16_store.hpp"  // QKS

#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/moe/forward_moe.hpp"
#include "src/models/qwen/moe/qwen_moe_ffn.hpp"
#include "src/models/qwen/moe/qwen_moe_route.hpp"
#include "src/core/hip_launch_check.hpp"

namespace gufo::hip {
namespace {

void CheckLaunch(int rc, const char* what) {
  if (rc != 0) {
    throw std::runtime_error(std::string("KAT-MTP: ") + what +
                             " launch failed");
  }
}

__global__ void KatMtpSwigluKernel(const float* __restrict__ gate,
                                   const float* __restrict__ up,
                                   float* __restrict__ act, int n) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= n) {
    return;
  }
  const float g = gate[index];
  act[index] = (g / (1.0F + expf(-g))) * up[index];
}

__global__ void KatMtpAxpyKernel(const float* __restrict__ x, float scale,
                                 float* __restrict__ out, int n) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= n) {
    return;
  }
  out[index] += scale * x[index];
}

__global__ void KatMtpScaleAddKernel(const float* __restrict__ x,
                                     const float* __restrict__ scale,
                                     float* __restrict__ out, int n) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= n) {
    return;
  }
  out[index] += scale[0] * x[index];
}

void LaunchSwiglu(const float* gate, const float* up, float* act, int n,
                  hipStream_t stream) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  KatMtpSwigluKernel<<<blocks, threads, 0, stream>>>(gate, up, act, n);
  GUFO_CHECK_LAUNCH("KatMtpSwigluKernel");
}

void LaunchAxpy(const float* x, float scale, float* out, int n,
                hipStream_t stream) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  KatMtpAxpyKernel<<<blocks, threads, 0, stream>>>(x, scale, out, n);
  GUFO_CHECK_LAUNCH("KatMtpAxpyKernel");
}

// KDG (SPEC1 step 2): per-slot axpy reading the weight on device.
__global__ void KatMtpAxpySlotKernel(const float* __restrict__ x,
                                     const float* __restrict__ d_weights,
                                     int slot, float* __restrict__ out,
                                     int n) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= n) {
    return;
  }
  out[index] += d_weights[slot] * x[index];
}

void LaunchAxpySlot(const float* x, const float* d_weights, int slot,
                    float* out, int n, hipStream_t stream) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  KatMtpAxpySlotKernel<<<blocks, threads, 0, stream>>>(x, d_weights, slot, out,
                                                       n);
  GUFO_CHECK_LAUNCH("KatMtpAxpySlotKernel");
}

// KDG (SPEC1 step 2): device-routed Q8_0 GEMV. Bit-identical to
// SmallBatchQ8_0ExactFp32VecGEMMKernel<16, 1, 2, true, true> (the batch=1
// Q8_0 route LaunchBatchedQuantGEMMFp32 takes in
// kernels/prefill_quant_gemm.hip) on the same expert bytes: same LDS
// staging, same scalar block-dot order, same scale fold-in, same
// ReduceKQuantWave tail. The only difference is the weight row address,
// which adds the routed expert's offset on device: base + id *
// expert_stride_blocks + row * num_blocks + block, with id =
// d_ids[slot]. expert_stride_blocks (= expert_bytes / 34) comes from the
// host-side KatExpertRowBytes math, so the slice matches KatExpertSlice
// exactly. An out-of-range id writes zeros for its rows (the step cannot
// sync to throw); LaunchQwenMoeRoute guarantees 0 <= id < n_expert.
__launch_bounds__(512, 1) __global__
    void KatMtpIndirectQ8GemmKernel(const Q8_0Block* __restrict__ base,
                                    const int* __restrict__ d_ids, int slot,
                                    std::uint32_t expert_stride_blocks,
                                    std::uint32_t n_expert,
                                    const float* __restrict__ x,
                                    float* __restrict__ y, std::uint32_t m,
                                    std::uint32_t k) {
  constexpr std::uint32_t kBlocksPerTile = 32;
  constexpr std::uint32_t kVectorsPerBlock = kQ8_0BlockSize / 4;
  constexpr std::uint32_t kStride = kQ8_0BlockSize;
  constexpr std::uint32_t kTileStride = kBlocksPerTile * kStride;
  __shared__ float staged_x[kTileStride];

  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const int id = d_ids[slot];
  const bool valid = (id >= 0 && static_cast<std::uint32_t>(id) < n_expert);
  const std::uint32_t safe_id = valid ? static_cast<std::uint32_t>(id) : 0u;
  const std::uint32_t row_base = (blockIdx.x * 16u + warp_id) * 2u;
  const std::uint32_t num_blocks = k / kQ8_0BlockSize;
  const std::size_t expert_base =
      static_cast<std::size_t>(safe_id) * expert_stride_blocks;
  float sums[2][1] = {};

  for (std::uint32_t tile_base = 0; tile_base < num_blocks;
       tile_base += kBlocksPerTile) {
    for (std::uint32_t flat = threadIdx.x; flat < 256u; flat += blockDim.x) {
      const std::uint32_t block = flat / kVectorsPerBlock;
      const std::uint32_t vector = flat % kVectorsPerBlock;
      const std::uint32_t source_block = tile_base + block;
      float4 value = {0.0F, 0.0F, 0.0F, 0.0F};
      if (source_block < num_blocks) {
        value = *reinterpret_cast<const float4*>(
            x + (source_block * kQ8_0BlockSize) + (vector * 4));
      }
      *reinterpret_cast<float4*>(staged_x + (vector * 128 + (block ^ vector) * 4)) =
          value;
    }
    SyncKQuantTile();

    const std::uint32_t block = tile_base + lane;
    if (block < num_blocks) {
      const Q8_0Block* rows[2];
      float scale[2];
#pragma unroll
      for (std::uint32_t r = 0; r < 2; ++r) {
        const std::uint32_t row = row_base + r;
        const std::uint32_t safe_row = row < m ? row : m - 1;
        rows[r] = &base[expert_base + (safe_row * num_blocks) + block];
        scale[r] = __half2float(rows[r]->d);
      }
      std::uint32_t packed[2][1];
      float block_dots[2][1] = {};
#pragma unroll
      for (std::uint32_t group = 0; group < kVectorsPerBlock; ++group) {
#pragma unroll
        for (std::uint32_t r = 0; r < 2; ++r) {
          __builtin_memcpy(&packed[r][0], rows[r]->qs + group * 4,
                           sizeof(std::uint32_t));
        }
        const float4 xv = *reinterpret_cast<const float4*>(
            staged_x + (group * 128 + (lane ^ group) * 4));
#pragma unroll
        for (std::uint32_t r = 0; r < 2; ++r) {
          const std::uint32_t p = packed[r][0];
          block_dots[r][0] +=
              static_cast<float>(static_cast<std::int8_t>(p & 0xFFU)) * xv.x;
          block_dots[r][0] +=
              static_cast<float>(
                  static_cast<std::int8_t>((p >> 8U) & 0xFFU)) *
              xv.y;
          block_dots[r][0] +=
              static_cast<float>(
                  static_cast<std::int8_t>((p >> 16U) & 0xFFU)) *
              xv.z;
          block_dots[r][0] +=
              static_cast<float>(
                  static_cast<std::int8_t>((p >> 24U) & 0xFFU)) *
              xv.w;
        }
      }
#pragma unroll
      for (std::uint32_t r = 0; r < 2; ++r) {
        sums[r][0] += scale[r] * block_dots[r][0];
      }
    }
    SyncKQuantTile();
  }

#pragma unroll
  for (std::uint32_t r = 0; r < 2; ++r) {
    sums[r][0] = ReduceKQuantWave(sums[r][0]);
  }
  if (lane == 0) {
#pragma unroll
    for (std::uint32_t r = 0; r < 2; ++r) {
      const std::uint32_t row = row_base + r;
      if (row >= m) {
        continue;
      }
      y[row] = valid ? sums[r][0] : 0.0F;
    }
  }
}

// KDG (SPEC1 step 2): device-routed Q8_K GEMV. Bit-identical to
// SmallBatchQ8KFp32GEMMKernel<8, 1> (the batch=1 Q8_K route
// LaunchBatchedQuantGEMMFp32 takes) on the same expert bytes; only the
// row address adds the on-device routed offset, as above.
__launch_bounds__(256, 4) __global__
    void KatMtpIndirectQ8KGemmKernel(const Q8KBlock* __restrict__ base,
                                     const int* __restrict__ d_ids, int slot,
                                     std::size_t expert_stride_blocks,
                                     int n_expert,
                                     const float* __restrict__ x,
                                     float* __restrict__ y, std::size_t m,
                                     std::size_t k) {
  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp_id = threadIdx.x >> 5u;
  const std::size_t row = (blockIdx.x * 8u) + warp_id;
  if (row >= m) {
    return;
  }
  const int id = d_ids[slot];
  const bool valid = (id >= 0 && id < n_expert);
  const std::size_t safe_id = valid ? static_cast<std::size_t>(id) : 0u;

  const std::size_t num_blocks = k / kQ8KBlockSize;
  const Q8KBlock* row_blocks =
      base + (safe_id * expert_stride_blocks) + (row * num_blocks);
  float sums[1] = {};

  for (std::size_t block = 0; block < num_blocks; ++block) {
    const Q8KBlock& weights = row_blocks[block];
    float weight_values[8];
#pragma unroll
    for (std::size_t element = 0; element < 8; ++element) {
      const std::size_t index = lane + (32 * element);
      weight_values[element] = static_cast<float>(weights.qs[index]);
    }

    float dot = 0.0F;
    {
      const float* input = x + (block * kQ8KBlockSize);
#pragma unroll
      for (std::size_t element = 0; element < 8; ++element) {
        const std::size_t index = lane + (32 * element);
        dot += weight_values[element] * input[index];
      }
    }
    sums[0] += weights.d * dot;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    sums[0] += __shfl_xor(sums[0], offset);
  }

  if (lane == 0) {
    y[row] = valid ? sums[0] : 0.0F;
  }
}

void GemvQ8Device(core::GgmlType type, const void* base, const int* d_ids,
                  int slot, std::size_t expert_bytes, int n_expert,
                  const float* x, float* y, int rows, int cols,
                  hipStream_t stream) {
  if (type == core::GgmlType::kQ8_0) {
    if (expert_bytes % sizeof(Q8_0Block) != 0) {
      throw std::runtime_error("KAT-MTP: Q8_0 expert stride is not aligned");
    }
    KatMtpIndirectQ8GemmKernel<<<static_cast<unsigned int>((rows + 31) / 32),
                                 512, 0, stream>>>(
        static_cast<const Q8_0Block*>(base), d_ids, slot,
        static_cast<std::uint32_t>(expert_bytes / sizeof(Q8_0Block)),
        static_cast<std::uint32_t>(n_expert), x, y,
        static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols));
    GUFO_CHECK_LAUNCH("KatMtpIndirectQ8GemmKernel");
    return;
  }
  if (type == core::GgmlType::kQ8_K) {
    if (expert_bytes % sizeof(Q8KBlock) != 0) {
      throw std::runtime_error("KAT-MTP: Q8_K expert stride is not aligned");
    }
    KatMtpIndirectQ8KGemmKernel<<<static_cast<unsigned int>((rows + 7) / 8),
                                  256, 0, stream>>>(
        static_cast<const Q8KBlock*>(base), d_ids, slot,
        expert_bytes / sizeof(Q8KBlock), n_expert, x, y,
        static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
    GUFO_CHECK_LAUNCH("KatMtpIndirectQ8KGemmKernel");
    return;
  }
  throw std::runtime_error("KAT-MTP: expert matrix must be Q8_0 or Q8_K");
}

// KDG (SPEC1 step 2): device-routed ExpertSwiglu variant. Takes the
// router's device id array plus the slot instead of a host expert id; each
// GEMV below resolves base + d_ids[slot] * expert_bytes on device, so the
// step launches with no D2H readback and no stream sync. Strides use
// KatExpertRowBytes (the same row math KatExpertSlice applies on the old
// host-loop path), and the operation order matches that path exactly.
void ExpertSwiglu(const models::QwenLayerWeights& layer, const int* d_ids,
                  int slot, const float* x, float* gate, float* up, float* act,
                  float* hidden, int hidden_size, int ff, int n_expert,
                  hipStream_t stream) {
  if (d_ids == nullptr) {
    throw std::runtime_error("KAT-MTP: device expert ids are missing");
  }
  if (layer.ffn_gate_exps.data == nullptr ||
      layer.ffn_up_exps.data == nullptr ||
      layer.ffn_down_exps.data == nullptr) {
    throw std::runtime_error("KAT-MTP: expert tensor storage is missing");
  }
  const std::size_t rb_gate =
      models::KatExpertRowBytes(layer.ffn_gate_exps.type, hidden_size);
  const std::size_t rb_up =
      models::KatExpertRowBytes(layer.ffn_up_exps.type, hidden_size);
  const std::size_t rb_down =
      models::KatExpertRowBytes(layer.ffn_down_exps.type, ff);
  if (rb_gate == 0 || rb_up == 0 || rb_down == 0) {
    throw std::runtime_error("KAT-MTP: device expert slice stride is zero");
  }
  GemvQ8Device(layer.ffn_gate_exps.type, layer.ffn_gate_exps.data, d_ids, slot,
                rb_gate * static_cast<std::size_t>(ff), n_expert, x, gate, ff,
                hidden_size, stream);
  GemvQ8Device(layer.ffn_up_exps.type, layer.ffn_up_exps.data, d_ids, slot,
                rb_up * static_cast<std::size_t>(ff), n_expert, x, up, ff,
                hidden_size, stream);
  LaunchSwiglu(gate, up, act, ff, stream);
  GemvQ8Device(layer.ffn_down_exps.type, layer.ffn_down_exps.data, d_ids, slot,
                rb_down * static_cast<std::size_t>(hidden_size), n_expert, act,
                hidden, hidden_size, ff, stream);
}

void LaunchScaleAdd(const float* x, const float* scale, float* out, int n,
                    hipStream_t stream) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  KatMtpScaleAddKernel<<<blocks, threads, 0, stream>>>(x, scale, out, n);
  GUFO_CHECK_LAUNCH("KatMtpScaleAddKernel");
}

void GemvQ8(const models::QwenTensorRef& weight, const float* x, float* y,
            int rows, int cols, hipStream_t stream) {
  if (weight.type != core::GgmlType::kQ8_0 &&
      weight.type != core::GgmlType::kQ8_K) {
    throw std::runtime_error(
        "KAT-MTP: expert matrix must be Q8_0 or Q8_K");
  }
  LaunchBatchedQuantGEMMFp32(weight.type, weight.data, x, y,
                             /*batch=*/1, static_cast<std::size_t>(rows),
                             static_cast<std::size_t>(cols), stream);
}

void ExpertSwiglu(const models::QwenLayerWeights& layer, int expert,
                  const float* x, float* gate, float* up, float* act,
                  float* hidden, int hidden_size, int ff, hipStream_t stream) {
  const auto gate_w = models::KatExpertSlice(
      expert < 0 ? layer.ffn_gate_shexp : layer.ffn_gate_exps, expert, ff,
      hidden_size);
  const auto up_w = models::KatExpertSlice(
      expert < 0 ? layer.ffn_up_shexp : layer.ffn_up_exps, expert, ff,
      hidden_size);
  const auto down_w = models::KatExpertSlice(
      expert < 0 ? layer.ffn_down_shexp : layer.ffn_down_exps, expert,
      hidden_size, ff);
  GemvQ8(gate_w, x, gate, ff, hidden_size, stream);
  GemvQ8(up_w, x, up, ff, hidden_size, stream);
  LaunchSwiglu(gate, up, act, ff, stream);
  GemvQ8(down_w, act, hidden, hidden_size, ff, stream);
}

__global__ void KatMtpWriteKvKernel(
    const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    std::uint32_t start_pos, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::uint32_t kv_h = blockIdx.x;
  const std::uint32_t token = blockIdx.y;
  const std::uint32_t tid = threadIdx.x;
  if (kv_h >= num_kv_heads || tid >= head_dim) {
    return;
  }
  const std::uint32_t pos = start_pos + token;
  const std::size_t src =
      (static_cast<std::size_t>(token) * num_kv_heads + kv_h) * head_dim + tid;
  // Same layout as WriteKVCacheKernel, layer_idx 0.
  const std::size_t kv_offset =
      ((static_cast<std::size_t>(kv_h) * max_context + pos) * head_dim) + tid;
  const float key = k[src];
  const float val = v[src];
  if (k_cache != nullptr) {
    k_cache[kv_offset] = key;
  }
  if (v_cache != nullptr) {
    v_cache[kv_offset] = val;
  }
  if (k_cache_f16 != nullptr || v_cache_f16 != nullptr) {
    const std::size_t kv_width =
        static_cast<std::size_t>(num_kv_heads) * head_dim;
    const std::size_t f16_offset =
        (static_cast<std::size_t>(pos) * kv_width) +
        (static_cast<std::size_t>(kv_h) * head_dim) + tid;
    if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_offset] = QwenKvF16(key);
    }
    if (v_cache_f16 != nullptr) {
      v_cache_f16[f16_offset] = QwenKvF16(val);
    }
  }
}

}  // namespace

void LaunchKatMtpWriteKv(const float* k, const float* v, float* k_cache,
                         float* v_cache, void* k_cache_f16, void* v_cache_f16,
                         std::uint32_t start_pos, std::size_t batch,
                         std::uint32_t max_context, std::uint32_t num_kv_heads,
                         std::uint32_t head_dim, hipStream_t stream) {
  if (k == nullptr || v == nullptr || batch == 0 || num_kv_heads == 0 ||
      head_dim == 0 || head_dim > 1024) {
    throw std::runtime_error("KAT-MTP: KV pack shape is invalid");
  }
  const dim3 grid(num_kv_heads, static_cast<unsigned int>(batch));
  KatMtpWriteKvKernel<<<grid, head_dim, 0, stream>>>(
      k, v, k_cache, v_cache, static_cast<__half*>(k_cache_f16),
      static_cast<__half*>(v_cache_f16), start_pos, max_context, num_kv_heads,
      head_dim);
  GUFO_CHECK_LAUNCH("KatMtpWriteKvKernel");
}

void LaunchKatMtpMoeFfn(const models::QwenLayerWeights& layer,
                        const core::ModelConfig& config, const float* normed,
                        float* out, const KatMtpMoeScratch& scratch,
                        hipStream_t stream) {
  const int n_expert = static_cast<int>(config.expert_count);
  const int n_used = static_cast<int>(config.expert_used_count);
  const int hidden = static_cast<int>(config.hidden_size);
  const int ff = static_cast<int>(config.expert_feed_forward_length);
  const int ff_s = static_cast<int>(config.expert_shared_feed_forward_length);
  if (normed == nullptr || out == nullptr || scratch.logits == nullptr ||
      scratch.ids == nullptr || scratch.weights == nullptr ||
      scratch.sig == nullptr || scratch.gate == nullptr ||
      scratch.up == nullptr || scratch.act == nullptr ||
      scratch.hidden == nullptr || hidden <= 0 || ff <= 0 || n_expert <= 0 ||
      n_used <= 0 || n_used > 8) {
    throw std::runtime_error("KAT-MTP: MoE scratch is incomplete");
  }
  if (hipMemsetAsync(out, 0, static_cast<std::size_t>(hidden) * sizeof(float),
                     stream) != hipSuccess) {
    throw std::runtime_error("KAT-MTP: MoE output clear failed");
  }
  CheckLaunch(LaunchQwenMoeRoute(
                  normed,
                  static_cast<const float*>(layer.ffn_gate_inp.data), hidden,
                  n_expert, n_used, scratch.logits, scratch.ids,
                  scratch.weights, stream),
              "route");
  // KDG (SPEC1 step 2): device-side expert dispatch. The slot kernels below
  // read the routed ids and weights on device (scratch.ids[slot],
  // scratch.weights[slot]); the host loop only fixes the launch order
  // (slot 0..n_used-1 ascending), so there is no D2H readback and no stream
  // sync on this path.
  for (int slot = 0; slot < n_used; ++slot) {
    ExpertSwiglu(layer, scratch.ids, slot, normed, scratch.gate, scratch.up,
                 scratch.act, scratch.hidden, hidden, ff, n_expert, stream);
    LaunchAxpySlot(scratch.hidden, scratch.weights, slot, out, hidden, stream);
  }
  if (!layer.ffn_up_shexp.empty()) {
    if (ff_s <= 0) {
      throw std::runtime_error("KAT-MTP: shared expert width is zero");
    }
    CheckLaunch(LaunchQwenMoeShexpSigmoid(
                    normed,
                    static_cast<const float*>(layer.ffn_gate_inp_shexp.data),
                    hidden, 1, scratch.sig, stream),
                "shared-expert sigmoid");
    ExpertSwiglu(layer, /*expert=*/-1, normed, scratch.gate, scratch.up,
                 scratch.act, scratch.hidden, hidden, ff_s, stream);
    LaunchScaleAdd(scratch.hidden, scratch.sig, out, hidden, stream);
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
