#ifndef GUFO_MODELS_GLIMMER_HIP_GLIMMER_ATTENTION_HPP_
#define GUFO_MODELS_GLIMMER_HIP_GLIMMER_ATTENTION_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/glimmer/hip/glimmer_decode.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

// LGA: the fp16 KV plane layout. The fp32 cache stays head-major
// ((layer*nkv+kv)*maxctx+pos)*hd+d (decode and the reference read it). The
// fp16 plane is TOKEN-major ((layer*maxctx+pos)*nkv+kv)*hd+d, K then V, so
// the WMMA canonical and packed paths read it directly. Same element count;
// only the index changes.
/// LGA: GLIMMER_ATTN=0 keeps the fp32 reference path. Unset or 1 uses the
/// WMMA kernel first, then the shared kernel. Fixed for the process.
[[nodiscard]] bool GlimmerSharedAttnOn() noexcept;

/// LGA: cast the fp32 KV reference rows [0, valid_rows) into the token-major
/// fp16 plane. Used after a snapshot restore, which writes only the fp32
/// plane. Rows past valid_rows are stale in both planes until a dual-write
/// overwrites them.
/// PGA: `layer_base` casts a single layer of a many-layer plane, and
/// `plane_layers` says how many layers the WHOLE plane holds -- the K/V half
/// stride is the full per-half plane, not the block being cast, because V
/// always starts one whole K plane past K. Passing 0 for `plane_layers` keeps
/// LGA's original behaviour (stride = `layers`), so arena.cpp's two
/// snapshot-restore calls, which pass neither, are byte-for-byte unchanged.
[[nodiscard]] GlimmerLaunchStatus LaunchGlimmerCastKvFp16(
    const float* src, void* dst, std::uint32_t layers,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, std::uint32_t valid_rows, hipStream_t stream,
    std::uint32_t layer_base = 0, std::uint32_t plane_layers = 0);

/// Same remap, restricted to the row range [row_begin, valid_rows): callers
/// that keep an fp16 mirror of the fp32 cache convert only the rows written
/// since their previous cast.
[[nodiscard]] GlimmerLaunchStatus LaunchGlimmerCastKvFp16Range(
    const float* src, void* dst, std::uint32_t layers,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, std::uint32_t row_begin, std::uint32_t valid_rows,
    hipStream_t stream, std::uint32_t layer_base = 0,
    std::uint32_t plane_layers = 0);

/// LGA: GQA-shared windowed attention over the token-major fp16 plane. One
/// block owns one KV head and up to 4 query rows, reads each fp16 K/V row
/// once, and serves all 16 query heads of that KV head. Q stays fp32.
/// fp32 accumulate, online softmax. Window 0 is full attention. Refuses any
/// shape other than head_dim 128 and 16 query heads per KV head, before
/// launch. This is the fallback when the WMMA path refuses a shape.
[[nodiscard]] GlimmerLaunchStatus LaunchGlimmerSharedAttention(
    const float* q, const float* gate, const void* k_f16, const void* v_f16,
    float* out, std::uint32_t layer, std::uint32_t start_pos,
    std::uint32_t batch, std::uint32_t window, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::size_t q_elems, std::size_t gate_elems, std::size_t cache_elems,
    std::size_t out_elems, hipStream_t stream);

/// LGA: WMMA matrix-core prefill attention over the token-major fp16 plane.
/// The KV cache is assumed already written (the prefill dual-write did it).
/// Window 0 selects the full-attention instantiation, nonzero the windowed
/// one. Returns ok=false (before any launch) when the shape is unsupported
/// so the caller can fall back. The packed workspaces are the arena's idle
/// FFN scratch; they must not alias live inputs, outputs or each other.
[[nodiscard]] GlimmerLaunchStatus LaunchGlimmerWmmaPrefillAttention(
    const float* q, const float* gate, const void* k_f16, const void* v_f16,
    float* out, std::uint32_t layer, std::uint32_t start_pos,
    std::uint32_t batch, std::uint32_t window, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::size_t q_elems, std::size_t gate_elems, std::size_t cache_elems,
    std::size_t out_elems, std::span<float> packed_k_workspace,
    std::span<float> packed_v_workspace, hipStream_t stream);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_GLIMMER_HIP_GLIMMER_ATTENTION_HPP_
