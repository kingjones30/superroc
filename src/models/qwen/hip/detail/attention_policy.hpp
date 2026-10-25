#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_ATTENTION_POLICY_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_ATTENTION_POLICY_HPP_

#include <cstddef>
#include <cstdint>
#include <utility>

#include "src/models/qwen/hip/execution_policy.hpp"

namespace gufo::hip::detail {

inline constexpr std::size_t kOptimizedAttentionMinBatch{1024};
inline constexpr std::uint32_t kTiledAttentionQueryHeads{24};
inline constexpr std::uint32_t kTiledAttentionKvHeads{4};
inline constexpr std::uint32_t kTiledAttentionHeadDim{256};
// KAT-ATTN: KAT-Coder full-attention head config (GQA 8). The WMMA prefill
// kernel instantiates for (24,4), (16,2) and (16,4); head_dim stays 256 for
// all three.
inline constexpr std::uint32_t kKatFullAttentionQueryHeads{16};
inline constexpr std::uint32_t kKatFullAttentionKvHeads{2};
// SQ17: MiMo-9B full-attention head config (GQA 4). Third WMMA prefill
// instantiation, (16,4); head_dim stays 256. Without it MiMo falls back to
// the scalar baseline kernel, whose per-block LDS grows with depth.
inline constexpr std::uint32_t kMimoFullAttentionQueryHeads{16};
inline constexpr std::uint32_t kMimoFullAttentionKvHeads{4};
// LGA: Muse Glimmer 30B prefill head config (GQA 16, head_dim 128). The
// WMMA prefill kernel instantiates for (32,2,128) windowed and windowless.
inline constexpr std::uint32_t kGlimmerQueryHeads{32};
inline constexpr std::uint32_t kGlimmerKvHeads{2};
inline constexpr std::uint32_t kGlimmerHeadDim{128};
inline constexpr std::uint32_t kFlintQueryHeads{32};
inline constexpr std::uint32_t kFlintKvHeads{4};
inline constexpr std::uint32_t kFlintHeadDim{128};
inline constexpr std::uint32_t kLlamaQueryHeads{56};
inline constexpr std::uint32_t kLlamaKvHeads{8};
inline constexpr std::uint32_t kLlamaHeadDim{128};
inline constexpr std::size_t kSplitKDecodeAttentionMinContext{128};
inline constexpr std::uint32_t kSplitKDecodeAttentionMaxSplits{32};
// gfx1151, from rocminfo on this box: 40 CUs, 32 waves/CU, wavefront 32.
// The KAT decode block is 256 threads = 8 waves, so a CU holds 4 blocks
// before the wave cap. Grid.x is the 2 KV heads, so the split axis needs
// 40 * 4 / 2 = 80 blocks to put 4 blocks on every CU. p1k and p4k stay on
// the historical 32; 8192 is the first length where an 80-way split still
// leaves each partition more than 100 tokens.
inline constexpr std::uint32_t kGfx1151ComputeUnits{40};
inline constexpr std::uint32_t kGfx1151MaxWavesPerCu{32};
inline constexpr std::uint32_t kKatGqa8WavesPerBlock{8};
inline constexpr std::uint32_t kKatGqa8FillSplits{
    kGfx1151ComputeUnits * (kGfx1151MaxWavesPerCu / kKatGqa8WavesPerBlock) /
    kKatFullAttentionKvHeads};
inline constexpr std::uint32_t kKatGqa8MaxSplits{kKatGqa8FillSplits};
inline constexpr std::size_t kKatGqa8FillMinContext{8192};
static_assert(kKatGqa8FillSplits == 80);
inline constexpr std::uint32_t kFusedQkNormMaxHeadDim{256};

struct AttentionSupportParams {
  std::size_t batch_size{0};
  std::uint32_t start_pos{0};
  std::uint32_t max_context{0};
  std::uint32_t num_heads{0};
  std::uint32_t num_kv_heads{0};
  std::uint32_t head_dim{0};
  bool has_k_cache_f16{false};
  bool has_v_cache_f16{false};
};

[[nodiscard]] constexpr bool ShouldAttemptOptimizedAttention(
    std::size_t visible_context) noexcept {
  return visible_context >= kOptimizedAttentionMinBatch;
}

// opt-c010-qk-rope-kv: fuse per-head Q/K RMSNorm, RoPE, and the KV-cache write
// into a single kernel per token (decode) / per token row (prefill). Flip to
// false to revert to the unfused chain (PerHeadRMSNorm x2 + RoPE +
// WriteKVCache*), which stays wired as the independent reference.
[[nodiscard]] constexpr bool ShouldFuseQKNormRoPEKvWrite(
    const QwenExecutionPolicy& policy) noexcept {
  return policy.fuse_qk_norm_rope_kv;
}

[[nodiscard]] constexpr bool ShouldFuseQKNormRoPEKvWrite() noexcept {
  return ShouldFuseQKNormRoPEKvWrite(QwenExecutionPolicy::Production());
}

[[nodiscard]] constexpr std::uint32_t SelectDecodeAttentionSplitCount(
    std::size_t sequence_length) noexcept {
  if (sequence_length < kSplitKDecodeAttentionMinContext) {
    return 1;
  }
  return kSplitKDecodeAttentionMaxSplits;
}

// KAT 16/2 only. 27B keeps SelectDecodeAttentionSplitCount (32). Below
// kKatGqa8FillMinContext the historical 32 is returned, so p1k and p4k do
// not move. At and above it, the split count is the CU fill count.
[[nodiscard]] constexpr std::uint32_t SelectKatGqa8SplitCount(
    std::size_t sequence_length) noexcept {
  const std::uint32_t base = SelectDecodeAttentionSplitCount(sequence_length);
  if (base <= 1 || sequence_length < kKatGqa8FillMinContext) {
    return base;
  }
  return kKatGqa8MaxSplits;
}

[[nodiscard]] constexpr std::uint32_t DecodeAttentionScratchSplits(
    std::uint32_t num_heads) noexcept {
  // The KAT GQA-8 kernel indexes partials with kKatGqa8MaxSplits. Every
  // other head count, including 27B's 24, keeps the historical stride.
  return num_heads == kKatFullAttentionQueryHeads
             ? kKatGqa8MaxSplits
             : kSplitKDecodeAttentionMaxSplits;
}

[[nodiscard]] constexpr bool IsFusedQkNormSupported(
    std::uint32_t head_dim) noexcept {
  return head_dim != 0 && head_dim <= kFusedQkNormMaxHeadDim;
}

[[nodiscard]] constexpr bool IsSplitKDecodeAttentionSupported(
    std::size_t sequence_length, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) noexcept {
  return SelectDecodeAttentionSplitCount(sequence_length) > 1 &&
         num_heads != 0 && num_kv_heads != 0 &&
         (num_heads % num_kv_heads) == 0 && head_dim == 256;
}

[[nodiscard]] constexpr std::size_t DecodeAttentionScratchElements(
    std::uint32_t num_heads, std::uint32_t head_dim) noexcept {
  return static_cast<std::size_t>(num_heads) *
         DecodeAttentionScratchSplits(num_heads) *
         (static_cast<std::size_t>(head_dim) + 2);
}

[[nodiscard]] constexpr bool IsTiledAttentionSupported(
    const AttentionSupportParams& params) noexcept {
  return params.batch_size != 0 &&
         params.num_heads == kTiledAttentionQueryHeads &&
         params.num_kv_heads == kTiledAttentionKvHeads &&
         params.head_dim == kTiledAttentionHeadDim &&
         static_cast<std::size_t>(params.start_pos) + params.batch_size <=
             params.max_context &&
         params.has_k_cache_f16 && params.has_v_cache_f16;
}

// KAT-ATTN: WMMA-only eligibility for one instantiated head config. This does
// NOT widen IsTiledAttentionSupported (which also gates the still-(24,4)
// tiled kernel and its pack launcher); only LaunchQwenWmmaAttention's
// dispatcher checks it, routing (24,4) and (16,2) to their instantiations.
// LGA: kHeadDim (default 256) extends the gate to Glimmer's (32,2,128).
template<std::uint32_t kHq, std::uint32_t kHkv,
         std::uint32_t kHeadDim = kTiledAttentionHeadDim>
[[nodiscard]] constexpr bool IsWmmaAttentionSupportedFor(
    const AttentionSupportParams& params) noexcept {
  return params.batch_size != 0 && params.num_heads == kHq &&
         params.num_kv_heads == kHkv && params.head_dim == kHeadDim &&
         static_cast<std::size_t>(params.start_pos) + params.batch_size <=
             params.max_context &&
         params.has_k_cache_f16 && params.has_v_cache_f16;
}

/// The row-split recurrence needs two scratch planes and a 128 x 128 state
/// tile.
[[nodiscard]] constexpr bool ShouldUseSsmRowSplitRecurrence(
    bool shape_supported, bool scratch_ready) noexcept {
  return shape_supported && scratch_ready;
}

/// Executes the existing prefill attention fallback chain without virtual
/// dispatch: tiled, then the baseline implementation.
template<typename TiledLauncher, typename BaselineLauncher>
inline void DispatchPrefillAttention(std::size_t visible_context,
                                     TiledLauncher&& launch_tiled,
                                     BaselineLauncher&& launch_baseline) {
  if (ShouldAttemptOptimizedAttention(visible_context)) {
    if (std::forward<TiledLauncher>(launch_tiled)()) {
      return;
    }
  }
  std::forward<BaselineLauncher>(launch_baseline)();
}

}  // namespace gufo::hip::detail

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_ATTENTION_POLICY_HPP_
