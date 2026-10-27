#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/dflash/dspark_sampling.hpp"
#include "src/core/hip_launch_check.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace superroc::hip::kernels {
namespace {

constexpr std::uint32_t kDsparkThreads = 256;
constexpr std::uint32_t kDsparkMaxTopK = 16;
constexpr std::uint32_t kDsparkTokensPerBlock = 2048;

__device__ __forceinline__ float DsparkBf16(hip_bfloat16 value) {
  return static_cast<float>(value);
}

__device__ __forceinline__ float DsparkDot(const float* state,
                                           const hip_bfloat16* cand,
                                           std::uint32_t rank) {
  float acc = 0.0F;
  std::uint32_t r = 0;
  for (; r + 4U <= rank; r += 4U) {
    acc += state[r] * DsparkBf16(cand[r]);
    acc += state[r + 1U] * DsparkBf16(cand[r + 1U]);
    acc += state[r + 2U] * DsparkBf16(cand[r + 2U]);
    acc += state[r + 3U] * DsparkBf16(cand[r + 3U]);
  }
  for (; r < rank; ++r) {
    acc += state[r] * DsparkBf16(cand[r]);
  }
  return acc;
}

__device__ __forceinline__ void InsertCandidate(float score, std::uint32_t id,
                                                float* scores,
                                                std::uint32_t* ids,
                                                std::uint32_t top_k) {
  if (!(score > scores[top_k - 1U])) {
    return;
  }
  std::uint32_t slot = top_k - 1U;
  while (slot > 0U && score > scores[slot - 1U]) {
    scores[slot] = scores[slot - 1U];
    ids[slot] = ids[slot - 1U];
    --slot;
  }
  scores[slot] = score;
  ids[slot] = id;
}

__device__ __forceinline__ void MergeLists(float* shared_scores,
                                           std::uint32_t* shared_ids,
                                           std::uint32_t top_k) {
  for (std::uint32_t stride = kDsparkThreads / 2U; stride > 0U;
       stride >>= 1U) {
    __syncthreads();
    if (threadIdx.x < stride) {
      const std::uint32_t src = (threadIdx.x + stride) * kDsparkMaxTopK;
      const std::uint32_t dst = threadIdx.x * kDsparkMaxTopK;
      float merged_scores[kDsparkMaxTopK];
      std::uint32_t merged_ids[kDsparkMaxTopK];
#pragma unroll
      for (std::uint32_t i = 0; i < kDsparkMaxTopK; ++i) {
        merged_scores[i] = shared_scores[dst + i];
        merged_ids[i] = shared_ids[dst + i];
      }
      for (std::uint32_t i = 0; i < top_k; ++i) {
        InsertCandidate(shared_scores[src + i], shared_ids[src + i],
                        merged_scores, merged_ids, top_k);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < kDsparkMaxTopK; ++i) {
        shared_scores[dst + i] = merged_scores[i];
        shared_ids[dst + i] = merged_ids[i];
      }
    }
  }
  __syncthreads();
}

__global__ void dspark_markov_partial_kernel(
    const float* __restrict__ logits, const hip_bfloat16* __restrict__ w1,
    const hip_bfloat16* __restrict__ w2, const std::uint32_t* __restrict__ tokens,
    float* __restrict__ partial_scores, std::uint32_t* __restrict__ partial_ids,
    std::uint32_t vocab_size, std::uint32_t rank, std::uint32_t top_k) {
  const std::uint32_t row = blockIdx.y;
  const std::uint32_t prev = tokens[row];
  extern __shared__ float shared[];
  float* state = shared;
  float* shared_scores = shared + rank;
  std::uint32_t* shared_ids =
      reinterpret_cast<std::uint32_t*>(shared_scores + kDsparkThreads * kDsparkMaxTopK);

  const hip_bfloat16* prev_row = w1 + static_cast<std::uint64_t>(prev) * rank;
  for (std::uint32_t i = threadIdx.x; i < rank; i += blockDim.x) {
    state[i] = DsparkBf16(prev_row[i]);
  }
  __syncthreads();

  float local_scores[kDsparkMaxTopK];
  std::uint32_t local_ids[kDsparkMaxTopK];
#pragma unroll
  for (std::uint32_t i = 0; i < kDsparkMaxTopK; ++i) {
    local_scores[i] = -INFINITY;
    local_ids[i] = 0;
  }

  const std::uint32_t n_tiles =
      (vocab_size + kDsparkTokensPerBlock - 1U) / kDsparkTokensPerBlock;
  const float* row_logits =
      logits + static_cast<std::uint64_t>(row) * vocab_size;
  for (std::uint32_t tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
    const std::uint32_t begin = tile * kDsparkTokensPerBlock;
    const std::uint32_t end = min(begin + kDsparkTokensPerBlock, vocab_size);
    for (std::uint32_t token = begin + threadIdx.x; token < end;
         token += blockDim.x) {
      const hip_bfloat16* cand = w2 + static_cast<std::uint64_t>(token) * rank;
      InsertCandidate(row_logits[token] + DsparkDot(state, cand, rank), token,
                      local_scores, local_ids, top_k);
    }
  }

  const std::uint32_t off = threadIdx.x * kDsparkMaxTopK;
#pragma unroll
  for (std::uint32_t i = 0; i < kDsparkMaxTopK; ++i) {
    shared_scores[off + i] = local_scores[i];
    shared_ids[off + i] = local_ids[i];
  }
  MergeLists(shared_scores, shared_ids, top_k);

  const std::uint32_t out = (row * gridDim.x + blockIdx.x) * top_k;
  for (std::uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
    partial_scores[out + i] = shared_scores[i];
    partial_ids[out + i] = shared_ids[i];
  }
}

__global__ void dspark_markov_finalize_kernel(
    const float* __restrict__ partial_scores,
    const std::uint32_t* __restrict__ partial_ids,
    const float* __restrict__ hidden, const hip_bfloat16* __restrict__ w1,
    const float* __restrict__ conf_proj, const float* __restrict__ conf_bias,
    std::uint32_t* __restrict__ tokens, float* __restrict__ confidences,
    const float* __restrict__ uniforms, std::uint32_t* __restrict__ candidate_ids,
    float* __restrict__ candidate_probabilities, std::uint32_t hidden_size,
    std::uint32_t rank, std::uint32_t top_k, std::uint32_t partial_count,
    float temperature) {
  const std::uint32_t row = blockIdx.x;
  __shared__ float shared_scores[kDsparkThreads * kDsparkMaxTopK];
  __shared__ std::uint32_t shared_ids[kDsparkThreads * kDsparkMaxTopK];
  __shared__ float conf_partial[kDsparkThreads];

  float local_scores[kDsparkMaxTopK];
  std::uint32_t local_ids[kDsparkMaxTopK];
#pragma unroll
  for (std::uint32_t i = 0; i < kDsparkMaxTopK; ++i) {
    local_scores[i] = -INFINITY;
    local_ids[i] = 0;
  }
  const std::uint32_t candidate_count = partial_count * top_k;
  const std::uint32_t base = row * candidate_count;
  for (std::uint32_t i = threadIdx.x; i < candidate_count; i += blockDim.x) {
    InsertCandidate(partial_scores[base + i], partial_ids[base + i],
                    local_scores, local_ids, top_k);
  }
  const std::uint32_t off = threadIdx.x * kDsparkMaxTopK;
#pragma unroll
  for (std::uint32_t i = 0; i < kDsparkMaxTopK; ++i) {
    shared_scores[off + i] = local_scores[i];
    shared_ids[off + i] = local_ids[i];
  }
  MergeLists(shared_scores, shared_ids, top_k);

  const std::uint32_t prev = tokens[row];
  float sum = 0.0F;
  if (conf_proj != nullptr) {
    const hip_bfloat16* prev_row = w1 + static_cast<std::uint64_t>(prev) * rank;
    const float* hid = hidden + static_cast<std::uint64_t>(row) * hidden_size;
    const std::uint32_t width = hidden_size + rank;
    for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
      const float value = i < hidden_size ? hid[i] : DsparkBf16(prev_row[i - hidden_size]);
      sum += value * conf_proj[i];
    }
  }
  conf_partial[threadIdx.x] = sum;
  __syncthreads();
  for (std::uint32_t stride = kDsparkThreads / 2U; stride > 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      conf_partial[threadIdx.x] += conf_partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x != 0U) {
    return;
  }

  float best_score = -INFINITY;
  std::uint32_t best_index = 0;
  for (std::uint32_t i = 0; i < top_k; ++i) {
    if (shared_scores[i] > best_score) {
      best_score = shared_scores[i];
      best_index = i;
    }
  }

  if (candidate_ids != nullptr) {
    for (std::uint32_t i = 0; i < top_k; ++i) {
      candidate_ids[row * top_k + i] = shared_ids[i];
    }
  }

  std::uint32_t chosen = best_index;
  if (temperature > 0.0F && uniforms != nullptr) {
    const float divisor = fmaxf(temperature, FLT_MIN);
    float denominator = 0.0F;
    float weights[kDsparkMaxTopK];
    for (std::uint32_t i = 0; i < top_k; ++i) {
      weights[i] = expf((shared_scores[i] - best_score) / divisor);
      denominator += weights[i];
    }
    const float inverse = 1.0F / fmaxf(denominator, 1e-30F);
    float threshold =
        fminf(fmaxf(uniforms[row], 0.0F), 0.99999994F) * denominator;
    chosen = top_k - 1U;
    for (std::uint32_t i = 0; i < top_k; ++i) {
      if (candidate_probabilities != nullptr) {
        candidate_probabilities[row * top_k + i] = weights[i] * inverse;
      }
      threshold -= weights[i];
      if (threshold < 0.0F && chosen == top_k - 1U) {
        chosen = i;
      }
    }
  } else if (candidate_probabilities != nullptr) {
    for (std::uint32_t i = 0; i < top_k; ++i) {
      candidate_probabilities[row * top_k + i] = i == best_index ? 1.0F : 0.0F;
    }
  }

  tokens[row + 1U] = shared_ids[chosen];
  if (confidences != nullptr) {
    if (conf_proj != nullptr) {
      const float bias = conf_bias != nullptr ? *conf_bias : 0.0F;
      confidences[row] = 1.0F / (1.0F + expf(-(conf_partial[0] + bias)));
    } else {
      confidences[row] = 1.0F;
    }
  }
}

}  // namespace

void LaunchDSparkMarkovSelect(
    const float* logits, const float* hidden, const void* markov_w1_bf16,
    const void* markov_w2_bf16, const float* conf_proj,
    const float* conf_bias, std::uint32_t* tokens, float* confidences,
    float* partial_scores, std::uint32_t* partial_ids, const float* uniforms,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::uint32_t rows, std::uint32_t vocab_size, std::uint32_t hidden_size,
    std::uint32_t markov_rank, std::uint32_t top_k, float temperature,
    hipStream_t stream) {
  if (logits == nullptr || hidden == nullptr || markov_w1_bf16 == nullptr ||
      markov_w2_bf16 == nullptr || tokens == nullptr ||
      partial_scores == nullptr || partial_ids == nullptr || rows == 0 ||
      vocab_size == 0 || hidden_size == 0 || markov_rank == 0 || top_k == 0 ||
      top_k > kDsparkMaxTopK || top_k > vocab_size ||
      !std::isfinite(temperature) || temperature < 0.0F ||
      (temperature > 0.0F && uniforms == nullptr)) {
    throw std::invalid_argument("invalid DSpark markov selector shape");
  }
  const std::uint32_t n_tiles =
      (vocab_size + kDsparkTokensPerBlock - 1U) / kDsparkTokensPerBlock;
  const std::size_t shared_bytes =
      (markov_rank + kDsparkThreads * kDsparkMaxTopK) * sizeof(float) +
      kDsparkThreads * kDsparkMaxTopK * sizeof(std::uint32_t);
  static std::uint32_t cached_grid = 0;
  static std::uint32_t cached_tiles = 0;
  if (cached_grid == 0 || cached_tiles != n_tiles) {
    int blocks_per_sm = 0;
    int sm_count = 0;
    (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, dspark_markov_partial_kernel, kDsparkThreads,
        shared_bytes);
    (void)hipDeviceGetAttribute(
        &sm_count, hipDeviceAttributeMultiprocessorCount, 0);
    const int occupancy_blocks =
        std::max(1, blocks_per_sm) * std::max(1, sm_count);
    cached_grid = static_cast<std::uint32_t>(
        std::max(1, std::min(static_cast<int>(n_tiles), occupancy_blocks)));
    cached_tiles = n_tiles;
    std::fprintf(stderr,
                 "[DsparkProf] markov_grid=%u tiles=%u bpsm=%d sm=%d\n",
                 cached_grid, n_tiles, blocks_per_sm, sm_count);
    std::fflush(stderr);
  }
  const std::uint32_t grid_x = cached_grid;
  dspark_markov_partial_kernel<<<dim3(grid_x, rows), kDsparkThreads,
                                 shared_bytes, stream>>>(
      logits, static_cast<const hip_bfloat16*>(markov_w1_bf16),
      static_cast<const hip_bfloat16*>(markov_w2_bf16), tokens, partial_scores,
      partial_ids, vocab_size, markov_rank, top_k);
  SUPERROC_CHECK_LAUNCH("dspark_markov_partial_kernel");
  dspark_markov_finalize_kernel<<<rows, kDsparkThreads, 0, stream>>>(
      partial_scores, partial_ids, hidden,
      static_cast<const hip_bfloat16*>(markov_w1_bf16), conf_proj, conf_bias,
      tokens, confidences, uniforms, candidate_ids, candidate_probabilities,
      hidden_size, markov_rank, top_k, grid_x, temperature);
  SUPERROC_CHECK_LAUNCH("dspark_markov_finalize_kernel");
}

__device__ __forceinline__ void DsparkGridBarrier(int* barrier,
                                                  int n_blocks) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const int sense = atomicAdd(barrier + 1, 0);
    __threadfence();
    const int arrived = atomicAdd(barrier, 1);
    if (arrived == n_blocks - 1) {
      atomicExch(barrier, 0);
      __threadfence();
      atomicExch(barrier + 1, 1 - sense);
    } else {
      while (atomicAdd(barrier + 1, 0) == sense) {
      }
    }
  }
  __syncthreads();
}

__global__ void dspark_markov_chain_kernel(
    const float* __restrict__ logits, const float* __restrict__ hidden,
    const hip_bfloat16* __restrict__ w1, const hip_bfloat16* __restrict__ w2,
    const float* __restrict__ conf_proj, const float* __restrict__ conf_bias,
    std::uint32_t* __restrict__ tokens, float* __restrict__ confidences,
    float* __restrict__ partial_scores, std::uint32_t* __restrict__ partial_ids,
    const float* __restrict__ uniforms, std::uint32_t* __restrict__ candidate_ids,
    float* __restrict__ candidate_probabilities, int* __restrict__ barrier,
    std::uint32_t n_slots, std::uint32_t vocab_size, std::uint32_t hidden_size,
    std::uint32_t rank, std::uint32_t top_k, std::uint32_t partial_count,
    float temperature) {
  extern __shared__ float shared[];
  float* state = shared;
  float* shared_scores = shared + rank;
  float* conf_partial =
      shared_scores + kDsparkThreads * kDsparkMaxTopK;
  std::uint32_t* shared_ids = reinterpret_cast<std::uint32_t*>(
      conf_partial + kDsparkThreads);

  for (std::uint32_t slot = 0; slot < n_slots; ++slot) {
    const std::uint32_t prev = tokens[slot];
    const hip_bfloat16* prev_row =
        w1 + static_cast<std::uint64_t>(prev) * rank;
    for (std::uint32_t i = threadIdx.x; i < rank; i += blockDim.x) {
      state[i] = DsparkBf16(prev_row[i]);
    }
    __syncthreads();

    float local_scores[kDsparkMaxTopK];
    std::uint32_t local_ids[kDsparkMaxTopK];
#pragma unroll
    for (std::uint32_t i = 0; i < kDsparkMaxTopK; ++i) {
      local_scores[i] = -INFINITY;
      local_ids[i] = 0;
    }

    const std::uint32_t begin = blockIdx.x * kDsparkTokensPerBlock;
    const std::uint32_t end = min(begin + kDsparkTokensPerBlock, vocab_size);
    const float* row_logits =
        logits + static_cast<std::uint64_t>(slot) * vocab_size;
    for (std::uint32_t token = begin + threadIdx.x; token < end;
         token += blockDim.x) {
      const hip_bfloat16* cand = w2 + static_cast<std::uint64_t>(token) * rank;
      float acc = 0.0F;
      for (std::uint32_t r = 0; r < rank; ++r) {
        acc += state[r] * DsparkBf16(cand[r]);
      }
      InsertCandidate(row_logits[token] + acc, token, local_scores, local_ids,
                      top_k);
    }

    const std::uint32_t off = threadIdx.x * kDsparkMaxTopK;
#pragma unroll
    for (std::uint32_t i = 0; i < kDsparkMaxTopK; ++i) {
      shared_scores[off + i] = local_scores[i];
      shared_ids[off + i] = local_ids[i];
    }
    MergeLists(shared_scores, shared_ids, top_k);

    const std::uint32_t out = blockIdx.x * top_k;
    for (std::uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
      partial_scores[out + i] = shared_scores[i];
      partial_ids[out + i] = shared_ids[i];
    }
    __threadfence();
    DsparkGridBarrier(barrier, static_cast<int>(gridDim.x));

    if (blockIdx.x == 0) {
      float merge_scores[kDsparkMaxTopK];
      std::uint32_t merge_ids[kDsparkMaxTopK];
#pragma unroll
      for (std::uint32_t i = 0; i < kDsparkMaxTopK; ++i) {
        merge_scores[i] = -INFINITY;
        merge_ids[i] = 0;
      }
      const std::uint32_t candidate_count = partial_count * top_k;
      for (std::uint32_t i = threadIdx.x; i < candidate_count;
           i += blockDim.x) {
        InsertCandidate(partial_scores[i], partial_ids[i], merge_scores,
                        merge_ids, top_k);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < kDsparkMaxTopK; ++i) {
        shared_scores[off + i] = merge_scores[i];
        shared_ids[off + i] = merge_ids[i];
      }
      MergeLists(shared_scores, shared_ids, top_k);

      float sum = 0.0F;
      if (conf_proj != nullptr) {
        const hip_bfloat16* conf_prev =
            w1 + static_cast<std::uint64_t>(prev) * rank;
        const float* hid =
            hidden + static_cast<std::uint64_t>(slot) * hidden_size;
        const std::uint32_t width = hidden_size + rank;
        for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
          const float value =
              i < hidden_size ? hid[i] : DsparkBf16(conf_prev[i - hidden_size]);
          sum += value * conf_proj[i];
        }
      }
      conf_partial[threadIdx.x] = sum;
      __syncthreads();
      for (std::uint32_t stride = kDsparkThreads / 2U; stride > 0U;
           stride >>= 1U) {
        if (threadIdx.x < stride) {
          conf_partial[threadIdx.x] += conf_partial[threadIdx.x + stride];
        }
        __syncthreads();
      }

      if (threadIdx.x == 0U) {
        float best_score = -INFINITY;
        std::uint32_t best_index = 0;
        for (std::uint32_t i = 0; i < top_k; ++i) {
          if (shared_scores[i] > best_score) {
            best_score = shared_scores[i];
            best_index = i;
          }
        }
        if (candidate_ids != nullptr) {
          for (std::uint32_t i = 0; i < top_k; ++i) {
            candidate_ids[slot * top_k + i] = shared_ids[i];
          }
        }
        std::uint32_t chosen = best_index;
        if (temperature > 0.0F && uniforms != nullptr) {
          const float divisor = fmaxf(temperature, FLT_MIN);
          float denominator = 0.0F;
          float weights[kDsparkMaxTopK];
          for (std::uint32_t i = 0; i < top_k; ++i) {
            weights[i] = expf((shared_scores[i] - best_score) / divisor);
            denominator += weights[i];
          }
          const float inverse = 1.0F / fmaxf(denominator, 1e-30F);
          float threshold =
              fminf(fmaxf(uniforms[slot], 0.0F), 0.99999994F) * denominator;
          chosen = top_k - 1U;
          for (std::uint32_t i = 0; i < top_k; ++i) {
            if (candidate_probabilities != nullptr) {
              candidate_probabilities[slot * top_k + i] = weights[i] * inverse;
            }
            threshold -= weights[i];
            if (threshold < 0.0F && chosen == top_k - 1U) {
              chosen = i;
            }
          }
        } else if (candidate_probabilities != nullptr) {
          for (std::uint32_t i = 0; i < top_k; ++i) {
            candidate_probabilities[slot * top_k + i] =
                i == best_index ? 1.0F : 0.0F;
          }
        }
        tokens[slot + 1U] = shared_ids[chosen];
        if (confidences != nullptr) {
          if (conf_proj != nullptr) {
            const float bias = conf_bias != nullptr ? *conf_bias : 0.0F;
            confidences[slot] =
                1.0F / (1.0F + expf(-(conf_partial[0] + bias)));
          } else {
            confidences[slot] = 1.0F;
          }
        }
      }
    }
    DsparkGridBarrier(barrier, static_cast<int>(gridDim.x));
  }
}

void LaunchDSparkMarkovChain(
    const float* logits, const float* hidden, const void* markov_w1_bf16,
    const void* markov_w2_bf16, const float* conf_proj,
    const float* conf_bias, std::uint32_t* tokens, float* confidences,
    float* partial_scores, std::uint32_t* partial_ids, const float* uniforms,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::uint32_t n_slots, std::uint32_t vocab_size, std::uint32_t hidden_size,
    std::uint32_t markov_rank, std::uint32_t top_k, float temperature,
    hipStream_t stream) {
  if (n_slots == 0) {
    return;
  }
  if (logits == nullptr || hidden == nullptr || markov_w1_bf16 == nullptr ||
      markov_w2_bf16 == nullptr || tokens == nullptr ||
      partial_scores == nullptr || partial_ids == nullptr || vocab_size == 0 ||
      hidden_size == 0 || markov_rank == 0 || top_k == 0 ||
      top_k > kDsparkMaxTopK || top_k > vocab_size ||
      !std::isfinite(temperature) || temperature < 0.0F ||
      (temperature > 0.0F && uniforms == nullptr)) {
    throw std::invalid_argument("invalid DSpark markov chain shape");
  }
  const std::uint32_t partial_count =
      (vocab_size + kDsparkTokensPerBlock - 1U) / kDsparkTokensPerBlock;
  int* barrier = reinterpret_cast<int*>(partial_ids + partial_count * top_k);
  const auto zero = hipMemsetAsync(barrier, 0, 2U * sizeof(int), stream);
  if (zero != hipSuccess) {
    throw std::runtime_error("DSpark markov barrier reset failed");
  }
  const std::size_t shared_bytes =
      (markov_rank + kDsparkThreads * kDsparkMaxTopK + kDsparkThreads) *
          sizeof(float) +
      kDsparkThreads * kDsparkMaxTopK * sizeof(std::uint32_t);
  dspark_markov_chain_kernel<<<partial_count, kDsparkThreads, shared_bytes,
                               stream>>>(
      logits, hidden, static_cast<const hip_bfloat16*>(markov_w1_bf16),
      static_cast<const hip_bfloat16*>(markov_w2_bf16), conf_proj, conf_bias,
      tokens, confidences, partial_scores, partial_ids, uniforms, candidate_ids,
      candidate_probabilities, barrier, n_slots, vocab_size, hidden_size,
      markov_rank, top_k, partial_count, temperature);
  SUPERROC_CHECK_LAUNCH("dspark_markov_chain_kernel");
}

}  // namespace superroc::hip::kernels
#endif  // defined(ENGINE_ENABLE_HIP)
