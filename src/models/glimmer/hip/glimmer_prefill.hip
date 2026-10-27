#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/executor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/models/glimmer/hip/glimmer_attention.hpp"  // LGA
#include "src/models/glimmer/hip/glimmer_decode.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/hip/ops/token.hpp"

namespace superroc::hip {
namespace {

void CheckGpu(const char* what) {
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error(what);
  }
}

std::size_t WeightScratchElems(const core::ModelConfig& config) {
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection = 2 * attention;
  return hidden * std::max<std::size_t>(
                      {q_projection, kv, attention, config.intermediate_size,
                       config.SsmQkvSize(), config.ssm_inner_size,
                       config.ssm_time_step_rank});
}

// GLM: GLIMMER_W8A8=0 keeps the BF16 dequant path. Anything else, including
// unset, uses the 27B prequantized W8A8 route.
bool UseW8A8() {
  static const bool on = [] {
    const char* env = std::getenv("GLIMMER_W8A8");
    return !(env != nullptr && env[0] == '0' && env[1] == '\0');
  }();
  return on;
}

// GLM: SPEC4 item 2. The W8A8 WMMA route wins on short prefill chunks and
// loses once a chunk approaches the arena batch. Measured on amdaimax2 with
// GLIMMER_PROF per-chunk GEMM ms, 103 GGUF (chunk rows -> BF16 / W8A8):
//   1447 -> 4221.8 / 3602.9 (-15%), 1777 -> 4872.8 / 4274.7 (-12%),
//   4096 -> 9253.7..9496.7 / 9756.9..10120.6 (+5..7%),
//   4056 -> 9472.6 / 11512.6 (+22%).
// The cost is flat in start position, so route on chunk rows alone.
// GLIMMER_W8A8_MAX_BATCH overrides the threshold; GLIMMER_W8A8=0 still wins.
std::size_t W8A8MaxBatch() {
  static const std::size_t rows = [] {
    const char* env = std::getenv("GLIMMER_W8A8_MAX_BATCH");
    if (env != nullptr && env[0] != '\0') {
      return static_cast<std::size_t>(std::strtoul(env, nullptr, 10));
    }
    return std::size_t{2048};
  }();
  return rows;
}

bool UseW8A8ForChunk(std::size_t batch) {
  return UseW8A8() && batch <= W8A8MaxBatch();
}

bool ProfOn() {
  static const bool on = [] {
    const char* env = std::getenv("GLIMMER_PROF");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
  }();
  return on;
}

// Same byte count arena.cpp reserves for d_scratch_q8_act at this batch.
std::size_t ArenaQ8ScratchBytes(std::size_t batch, std::size_t row_elems) {
  const std::size_t q8_rows = ((batch + 15) / 16) * 16;
  return ((((q8_rows * row_elems) + 31) / 32) * sizeof(float) * 10) + 4096;
}

std::uint64_t CausalPairs(std::uint32_t start, std::uint32_t n,
                          std::uint32_t window) {
  std::uint64_t pairs = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    const std::uint32_t keys = start + i + 1;
    pairs += window == 0 ? keys : std::min(keys, window);
  }
  return pairs;
}

struct ProfSpan {
  hipEvent_t start{};
  hipEvent_t stop{};
};

struct ProfAcc {
  double gemm_ms = 0;
  double attn_swa_ms = 0;
  double attn_full_ms = 0;
  double wall_ms = 0;
  std::uint64_t pairs_swa = 0;
  std::uint64_t pairs_full = 0;
};

double FlopPerPair(const core::ModelConfig& config) {
  return static_cast<double>(config.num_attention_heads) * 4.0 *
         static_cast<double>(config.head_dim);
}

double NanosPerPair(std::uint64_t pairs, double ms) {
  if (pairs == 0 || ms <= 0) {
    return 0;
  }
  return ms * 1.0e6 / static_cast<double>(pairs);
}

double Teraflops(std::uint64_t pairs, double ms, double flop_per_pair) {
  if (pairs == 0 || ms <= 0) {
    return 0;
  }
  const double flops = static_cast<double>(pairs) * flop_per_pair;
  return flops / (ms * 1.0e-3) / 1.0e12;
}

// GPU spans. attn_* starts after QK-norm and ends at o_proj, so it includes
// RoPE and the KV write as well as attention. Those two are linear in the
// chunk; the quadratic cost is the attention kernel.
struct Prof {
  bool on = false;
  hipStream_t stream = nullptr;
  std::vector<ProfSpan> gemm;
  std::vector<ProfSpan> swa;
  std::vector<ProfSpan> full;
  hipEvent_t wall0{};
  hipEvent_t wall1{};
  std::uint64_t pairs_swa = 0;
  std::uint64_t pairs_full = 0;

  void Arm(hipStream_t in_stream) {
    on = ProfOn();
    stream = in_stream;
    if (!on) {
      return;
    }
    if (hipEventCreate(&wall0) != hipSuccess ||
        hipEventCreate(&wall1) != hipSuccess ||
        hipEventRecord(wall0, stream) != hipSuccess) {
      throw std::runtime_error("glimmer prof arm failed");
    }
  }

  void Begin(std::vector<ProfSpan>& bucket) {
    if (!on) {
      return;
    }
    ProfSpan span;
    if (hipEventCreate(&span.start) != hipSuccess ||
        hipEventCreate(&span.stop) != hipSuccess ||
        hipEventRecord(span.start, stream) != hipSuccess) {
      throw std::runtime_error("glimmer prof begin failed");
    }
    bucket.push_back(span);
  }

  void End(std::vector<ProfSpan>& bucket) {
    if (!on) {
      return;
    }
    if (hipEventRecord(bucket.back().stop, stream) != hipSuccess) {
      throw std::runtime_error("glimmer prof end failed");
    }
  }

  void MarkWall() {
    if (!on) {
      return;
    }
    if (hipEventRecord(wall1, stream) != hipSuccess) {
      throw std::runtime_error("glimmer prof wall failed");
    }
  }

  static double Drain(std::vector<ProfSpan>& bucket) {
    double ms = 0;
    for (auto& span : bucket) {
      if (hipEventSynchronize(span.stop) != hipSuccess) {
        throw std::runtime_error("glimmer prof sync failed");
      }
      float dt = 0;
      if (hipEventElapsedTime(&dt, span.start, span.stop) != hipSuccess) {
        throw std::runtime_error("glimmer prof elapsed failed");
      }
      ms += static_cast<double>(dt);
      hipEventDestroy(span.start);
      hipEventDestroy(span.stop);
    }
    bucket.clear();
    return ms;
  }

  void Finish(ProfAcc& total, std::uint32_t start, std::size_t batch,
              bool last, bool w8, double flop_per_pair) {
    if (!on) {
      return;
    }
    if (hipEventSynchronize(wall1) != hipSuccess) {
      throw std::runtime_error("glimmer prof wall sync failed");
    }
    float wall_dt = 0;
    if (hipEventElapsedTime(&wall_dt, wall0, wall1) != hipSuccess) {
      throw std::runtime_error("glimmer prof wall elapsed failed");
    }
    const double gemm_ms = Drain(gemm);
    const double swa_ms = Drain(swa);
    const double full_ms = Drain(full);
    const double wall_ms = static_cast<double>(wall_dt);
    const double rest_ms = wall_ms - gemm_ms - swa_ms - full_ms;
    hipEventDestroy(wall0);
    hipEventDestroy(wall1);
    total.gemm_ms += gemm_ms;
    total.attn_swa_ms += swa_ms;
    total.attn_full_ms += full_ms;
    total.wall_ms += wall_ms;
    total.pairs_swa += pairs_swa;
    total.pairs_full += pairs_full;
    std::fprintf(
        stderr,
        "GLIMMER_PROF chunk start=%u batch=%zu w8a8=%d gemm_ms=%.3f "
        "attn_swa_ms=%.3f attn_full_ms=%.3f rest_ms=%.3f wall_ms=%.3f "
        "pairs_swa=%llu pairs_full=%llu swa_ns=%.2f full_ns=%.2f "
        "swa_tflops=%.2f full_tflops=%.2f note=attn_includes_rope_kv\n",
        start, batch, w8 ? 1 : 0, gemm_ms, swa_ms, full_ms, rest_ms, wall_ms,
        static_cast<unsigned long long>(pairs_swa),
        static_cast<unsigned long long>(pairs_full),
        NanosPerPair(pairs_swa, swa_ms), NanosPerPair(pairs_full, full_ms),
        Teraflops(pairs_swa, swa_ms, flop_per_pair),
        Teraflops(pairs_full, full_ms, flop_per_pair));
    if (last) {
      const double total_rest =
          total.wall_ms - total.gemm_ms - total.attn_swa_ms - total.attn_full_ms;
      std::fprintf(
          stderr,
          "GLIMMER_PROF total w8a8=%d gemm_ms=%.3f attn_swa_ms=%.3f "
          "attn_full_ms=%.3f rest_ms=%.3f wall_ms=%.3f pairs_swa=%llu "
          "pairs_full=%llu swa_ns=%.2f full_ns=%.2f swa_tflops=%.2f "
          "full_tflops=%.2f note=attn_includes_rope_kv\n",
          w8 ? 1 : 0, total.gemm_ms, total.attn_swa_ms, total.attn_full_ms,
          total_rest, total.wall_ms,
          static_cast<unsigned long long>(total.pairs_swa),
          static_cast<unsigned long long>(total.pairs_full),
          NanosPerPair(total.pairs_swa, total.attn_swa_ms),
          NanosPerPair(total.pairs_full, total.attn_full_ms),
          Teraflops(total.pairs_swa, total.attn_swa_ms, flop_per_pair),
          Teraflops(total.pairs_full, total.attn_full_ms, flop_per_pair));
    }
    std::fflush(stderr);
    pairs_swa = 0;
    pairs_full = 0;
  }
};

// Batch 1 uses the decode GEMV so a one-token chunk matches ForwardToken.
// Longer chunks either dequantize FP4 to BF16 and run hipBLAS, or (the
// default) quantize the activation once and run the 27B W8A8 WMMA GEMM.
void Project(QwenGpuArena& arena, const models::QwenTensorRef& weight,
             const float* fp32, const void* bf16, float* y, std::size_t batch,
             std::size_t m, std::size_t k, std::size_t weight_elems) {
  if (m == 0 || k == 0 || batch == 0) {
    throw std::runtime_error("glimmer prefill projection is empty");
  }
  if (batch == 1) {
    LaunchGEMV(weight.data, weight.type, fp32, y, m, k, arena.stream);
    CheckGpu("glimmer prefill gemv failed");
    return;
  }
  if (m * k > weight_elems) {
    throw std::runtime_error("glimmer prefill weight scratch is short");
  }
  LaunchDequantizeToBf16(weight.type, weight.data, arena.d_weights_bf16, m * k,
                         arena.stream);
  CheckGpu("glimmer prefill dequant failed");
  bool ran = false;
  if (arena.hipblaslt_gemm != nullptr && m >= 1024 && k >= 1024) {
    ran = arena.hipblaslt_gemm->RunBf16(arena.d_weights_bf16, bf16, y, batch, m,
                                        k, arena.stream);
  }
  if (!ran) {
    LaunchHipblasGEMMBF16(arena.hipblas_handle, arena.d_weights_bf16, bf16, y,
                          batch, m, k, arena.stream);
  }
  CheckGpu("glimmer prefill bf16 gemm failed");
}

// GLM: prequantized activation, no weight dequant.
void ProjectQ8(QwenGpuArena& arena, const models::QwenTensorRef& weight,
               const void* q8, float* y, std::size_t batch, std::size_t m,
               std::size_t k) {
  if (m == 0 || k == 0 || batch == 0 || q8 == nullptr) {
    throw std::runtime_error("glimmer prefill w8a8 projection is empty");
  }
  if (!detail::IsNativeWmmaQuant(weight.type)) {
    throw std::runtime_error("glimmer prefill w8a8 weight is not native quant");
  }
  LaunchBatchedQuantGEMMPreQuantized(weight.type, weight.data, q8, y, batch, m,
                                     k, arena.stream);
  CheckGpu("glimmer prefill w8a8 gemm failed");
}

}  // namespace

tokenization::TokenId QwenGpuExecutor::ForwardGlimmerPromptChunk(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  CheckReset();
  if (vision_input_.Bytes() != 0) {
    throw std::runtime_error("muse-glimmer vision is not supported");
  }
  const auto& config = weights_.config;
  if (!config.attention_only) {
    throw std::runtime_error("glimmer prefill called for a qwen config");
  }
  if (arena_.d_kv_cache == nullptr) {
    throw std::runtime_error("muse-glimmer requires fp32 KV");
  }
  const std::size_t batch = prompt_tokens.size();
  if (batch == 0) {
    return 0;
  }
  if (batch > arena_.GetMaxBatch()) {
    throw std::length_error("prompt chunk exceeds the GPU batch length");
  }
  const std::size_t end_pos = static_cast<std::size_t>(start_pos) + batch;
  if (end_pos > arena_.GetMaxContext()) {
    throw std::length_error("prompt exceeds the GPU context length");
  }
  // One token is the decode step, so ForwardPromptBatch of one id matches
  // ForwardToken. The fused SwiGLU GEMV is not the same kernel as two GEMVs.
  if (batch == 1) {
    last_hidden_offset_ = 0;
    ExecuteGlimmerDecodeStep(arena_, weights_, prompt_tokens[0], start_pos,
                             compute_logits, nullptr);
    if (!compute_logits) {
      if (hipStreamSynchronize(arena_.stream) != hipSuccess) {
        throw std::runtime_error("glimmer prefill synchronize failed");
      }
      return 0;
    }
    std::uint32_t next_token_id = 0;
    if (hipMemcpyAsync(&next_token_id, arena_.d_alpha_buf, sizeof(std::uint32_t),
                       hipMemcpyDeviceToHost, arena_.stream) != hipSuccess ||
        hipStreamSynchronize(arena_.stream) != hipSuccess) {
      throw std::runtime_error("glimmer prefill token read failed");
    }
    return CheckedSampleToken(next_token_id, config.vocab_size);
  }

  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv = static_cast<std::size_t>(config.num_key_value_heads) *
                         config.head_dim;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t weight_elems = WeightScratchElems(config);
  const std::size_t plane = arena_.GetAttentionKvPlaneElements();
  float* v_cache = arena_.d_kv_cache + plane;
  auto* bf16 = static_cast<hip_bfloat16*>(arena_.d_scratch_bf16);
  const bool w8 =
      UseW8A8ForChunk(batch) && !weights_.layers.empty() &&
      detail::IsNativeWmmaQuant(weights_.layers.front().attn_q.type);
  if (w8) {
    const std::size_t q_projection = 2 * attention;
    const std::size_t ssm_qkv = config.SsmQkvSize();
    const std::size_t projection_width = std::max(q_projection, ssm_qkv);
    const std::size_t row = std::max<std::size_t>(
        {intermediate, hidden, projection_width,
         ssm_qkv + config.ssm_inner_size +
             (2 * static_cast<std::size_t>(config.ssm_time_step_rank))});
    const std::size_t have = ArenaQ8ScratchBytes(arena_.GetMaxBatch(), row);
    const std::size_t need =
        QuantizedActivationBytes(arena_.GetMaxBatch(), intermediate);
    if (need > have) {
      throw std::runtime_error("glimmer prefill q8 scratch is short");
    }
  }
  static ProfAcc prof_total;
  if (start_pos == 0) {
    prof_total = {};
  }
  Prof prof;
  prof.Arm(arena_.stream);
  const double flop_per_pair = FlopPerPair(config);

  std::vector<std::uint32_t> host_tokens(prompt_tokens.begin(),
                                         prompt_tokens.end());
  if (hipMemcpyAsync(arena_.d_prompt_tokens, host_tokens.data(),
                     batch * sizeof(std::uint32_t), hipMemcpyHostToDevice,
                     arena_.stream) != hipSuccess) {
    throw std::runtime_error("glimmer prefill token copy failed");
  }
  LaunchBatchedEmbeddingLookup(weights_.token_embd.data, weights_.token_embd.type,
                               arena_.d_prompt_tokens, arena_.d_hidden, batch,
                               hidden, arena_.stream);
  CheckGpu("glimmer prefill embedding failed");
  LaunchBatchedRMSNorm(arena_.d_hidden, nullptr, arena_.d_hidden, bf16, batch,
                       hidden, config.norm_eps_pre, arena_.stream);
  CheckGpu("glimmer prefill embed norm failed");

  for (std::uint32_t layer_index = 0; layer_index < config.num_layers;
       ++layer_index) {
    PrefillLayerCheckpoint(layer_index);
    const auto& layer = weights_.layers[layer_index];
    if (layer.attn_q_norm.data == nullptr || layer.attn_k_norm.data == nullptr ||
        layer.attn_post_norm.data == nullptr ||
        layer.ffn_post_norm.data == nullptr) {
      throw std::runtime_error("glimmer prefill layer is missing a norm tensor");
    }
    LaunchBatchedRMSNorm(arena_.d_hidden,
                         static_cast<const float*>(layer.attn_norm.data),
                         arena_.d_normed, bf16, batch, hidden,
                         config.norm_eps_pre, arena_.stream);
    CheckGpu("glimmer prefill attn norm failed");
    // GLM: one Q8_1 activation shared by q/k/v/gate.
    prof.Begin(prof.gemm);
    if (w8) {
      LaunchQuantizeActivationQ8_1(bf16, arena_.d_scratch_q8_act, batch, hidden,
                                   arena_.stream);
      CheckGpu("glimmer prefill attn q8 failed");
      ProjectQ8(arena_, layer.attn_q, arena_.d_scratch_q8_act, arena_.d_q, batch,
                attention, hidden);
      ProjectQ8(arena_, layer.attn_k, arena_.d_scratch_q8_act, arena_.d_k, batch,
                kv, hidden);
      ProjectQ8(arena_, layer.attn_v, arena_.d_scratch_q8_act, arena_.d_v, batch,
                kv, hidden);
      ProjectQ8(arena_, layer.attn_gate, arena_.d_scratch_q8_act,
                arena_.d_ssm_gate, batch, attention, hidden);
    } else {
      Project(arena_, layer.attn_q, arena_.d_normed, bf16, arena_.d_q, batch,
              attention, hidden, weight_elems);
      Project(arena_, layer.attn_k, arena_.d_normed, bf16, arena_.d_k, batch, kv,
              hidden, weight_elems);
      Project(arena_, layer.attn_v, arena_.d_normed, bf16, arena_.d_v, batch, kv,
              hidden, weight_elems);
      Project(arena_, layer.attn_gate, arena_.d_normed, bf16, arena_.d_ssm_gate,
              batch, attention, hidden, weight_elems);
    }
    prof.End(prof.gemm);
    LaunchBatchedPerHeadRMSNorm(
        arena_.d_q, static_cast<const float*>(layer.attn_q_norm.data),
        arena_.d_q, batch, config.num_attention_heads, config.head_dim,
        config.norm_eps_pre, arena_.stream);
    LaunchBatchedPerHeadRMSNorm(
        arena_.d_k, static_cast<const float*>(layer.attn_k_norm.data),
        arena_.d_k, batch, config.num_key_value_heads, config.head_dim,
        config.norm_eps_pre, arena_.stream);
    CheckGpu("glimmer prefill qk norm failed");
    // GLM: attention region. The launch below is LX1's call site and is unchanged.
    const bool full_layer = config.LayerSwaWindow(layer_index) == 0;
    const std::uint64_t layer_pairs = CausalPairs(
        start_pos, static_cast<std::uint32_t>(batch),
        full_layer ? 0U : config.LayerSwaWindow(layer_index));
    if (full_layer) {
      prof.pairs_full += layer_pairs;
      prof.Begin(prof.full);
    } else {
      prof.pairs_swa += layer_pairs;
      prof.Begin(prof.swa);
    }
    if (config.LayerUsesRope(layer_index)) {
      const auto q_rope = LaunchGlimmerBatchedNormRope(
          arena_.d_q, config.num_attention_heads, config.head_dim,
          config.rotary_dim, start_pos, static_cast<std::uint32_t>(batch),
          config.rope_theta, arena_.stream);
      const auto k_rope = LaunchGlimmerBatchedNormRope(
          arena_.d_k, config.num_key_value_heads, config.head_dim,
          config.rotary_dim, start_pos, static_cast<std::uint32_t>(batch),
          config.rope_theta, arena_.stream);
      if (!q_rope.ok || !k_rope.ok) {
        throw std::runtime_error(q_rope.ok ? k_rope.error : q_rope.error);
      }
    }
    const auto wrote = LaunchGlimmerBatchedWriteKv(
        arena_.d_k, arena_.d_v, arena_.d_kv_cache, v_cache, layer_index,
        start_pos, static_cast<std::uint32_t>(batch), arena_.GetMaxContext(),
        config.num_key_value_heads, config.head_dim, batch * kv, plane,
        arena_.stream);
    if (!wrote.ok) {
      throw std::runtime_error(wrote.error);
    }
    // LGA: the token-major fp16 plane sits beside the fp32 cache. V starts
    // one plane past K, mirroring the fp32 v_cache above.
    void* k_f16 = arena_.d_glimmer_kv_f16;
    void* v_f16 =
        k_f16 == nullptr ? nullptr : static_cast<std::uint16_t*>(k_f16) + plane;
    // PGA: c20cf22 has no fp16 dual-write in LaunchGlimmerBatchedWriteKv --
    // lane LX1's never landed and glimmer_decode.{hpp,hip} are outside this
    // lane's zone. LGA's own LaunchGlimmerCastKvFp16 does exactly the remap
    // LX1's dual-write did (head-major fp32 rows [0, valid_rows) -> token-major
    // fp16), so the prefill cast is issued here, per layer, right after the
    // fp32 write. [0, start_pos+batch) is precisely the causal span the
    // attention below reads, and the fp32 plane stays the decode path's
    // single source of truth, so GlimmerAttendQuery is unchanged.
    // layer_base = layer_index casts this layer only, and plane_layers tells
    // the cast how many layers the whole plane holds, so its K/V half stride
    // stays the FULL per-half plane. Both halves are [all layers], V one plane
    // past K -- exactly the layout the WMMA launcher assumes when it walks
    // layer_k / layer_v off k_f16 / v_f16 (attention_wmma.hip,
    // layer_cache_offset = layer_idx * max_context * kKvWidth).
    // PGA MEASURED: without plane_layers the stride was layer_index+1 layers,
    // so every layer except the last wrote V into the wrong place; PPL went to
    // 337811 and the greedy text stopped following the prompt.
    const bool fp16_plane = GlimmerSharedAttnOn() && k_f16 != nullptr;
    if (fp16_plane) {
      const std::size_t per_layer =
          static_cast<std::size_t>(arena_.GetMaxContext()) *
          config.num_key_value_heads * config.head_dim;
      const std::uint32_t plane_layers =
          per_layer == 0
              ? 0U
              : static_cast<std::uint32_t>(plane / per_layer);
      const auto cast = LaunchGlimmerCastKvFp16(
          arena_.d_kv_cache, k_f16, /*layers=*/1, arena_.GetMaxContext(),
          config.num_key_value_heads, config.head_dim,
          static_cast<std::uint32_t>(end_pos), arena_.stream, layer_index,
          plane_layers);
      if (!cast.ok) {
        throw std::runtime_error(
            cast.error != nullptr ? cast.error
                                  : "glimmer prefill fp16 kv cast failed");
      }
    }
    // LGA: WMMA matrix cores first, the GQA-shared fp16 kernel second, the
    // fp32 reference last. GLIMMER_ATTN=0 keeps the fp32 reference.
    GlimmerLaunchStatus attn{false, nullptr};
    if (fp16_plane) {
      // FFN scratch is idle during attention; the packed path borrows it.
      const auto wmma_workspace =
          arena_.GetScratchView(arena_.GetMaxBatch()).ffn;
      attn = LaunchGlimmerWmmaPrefillAttention(
          arena_.d_q, arena_.d_ssm_gate, k_f16, v_f16, arena_.d_ssm_out,
          layer_index, start_pos, static_cast<std::uint32_t>(batch),
          config.LayerSwaWindow(layer_index), arena_.GetMaxContext(),
          config.num_attention_heads, config.num_key_value_heads,
          config.head_dim, batch * attention, batch * attention, plane,
          batch * attention, wmma_workspace.gate, wmma_workspace.up,
          arena_.stream);
      if (!attn.ok) {
        attn = LaunchGlimmerSharedAttention(
            arena_.d_q, arena_.d_ssm_gate, k_f16, v_f16, arena_.d_ssm_out,
            layer_index, start_pos, static_cast<std::uint32_t>(batch),
            config.LayerSwaWindow(layer_index), arena_.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, batch * attention, batch * attention, plane,
            batch * attention, arena_.stream);
      }
    } else {
      attn = LaunchGlimmerPrefillAttention(
          arena_.d_q, arena_.d_ssm_gate, arena_.d_kv_cache, v_cache,
          arena_.d_ssm_out, layer_index, start_pos,
          static_cast<std::uint32_t>(batch), config.LayerSwaWindow(layer_index),
          arena_.GetMaxContext(), config.num_attention_heads,
          config.num_key_value_heads, config.head_dim, batch * attention,
          batch * attention, plane, batch * attention, arena_.stream);
    }
    if (!attn.ok) {
      throw std::runtime_error(attn.error);
    }
    prof.End(full_layer ? prof.full : prof.swa);
    if (batch == 1) {
      Project(arena_, layer.attn_output, arena_.d_ssm_out, bf16, arena_.d_attn_out,
              batch, hidden, attention, weight_elems);
    } else if (w8) {
      // GLM: quantize the fp32 attention output. No bf16 cast.
      prof.Begin(prof.gemm);
      LaunchQuantizeActivationQ8_1FromFp32(arena_.d_ssm_out, arena_.d_scratch_q8_act,
                                           batch, attention, arena_.stream);
      CheckGpu("glimmer prefill attn-out q8 failed");
      ProjectQ8(arena_, layer.attn_output, arena_.d_scratch_q8_act,
                arena_.d_attn_out, batch, hidden, attention);
      prof.End(prof.gemm);
    } else {
      prof.Begin(prof.gemm);
      LaunchFloatToBfloat16(arena_.d_ssm_out, bf16, batch * attention,
                            arena_.stream);
      CheckGpu("glimmer prefill attn bf16 cast failed");
      Project(arena_, layer.attn_output, arena_.d_ssm_out, bf16, arena_.d_attn_out,
              batch, hidden, attention, weight_elems);
      prof.End(prof.gemm);
    }
    LaunchBatchedRMSNorm(arena_.d_attn_out,
                         static_cast<const float*>(layer.attn_post_norm.data),
                         arena_.d_normed, nullptr, batch, hidden,
                         config.norm_eps_post, arena_.stream);
    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_normed, arena_.d_hidden,
                             batch, hidden, arena_.stream);
    CheckGpu("glimmer prefill attn residual failed");
    LaunchBatchedRMSNorm(arena_.d_hidden,
                         static_cast<const float*>(layer.ffn_norm.data),
                         arena_.d_normed, bf16, batch, hidden,
                         config.norm_eps_pre, arena_.stream);
    CheckGpu("glimmer prefill ffn norm failed");
    // GLM: one Q8_1 activation, dual gate/up, fused SwiGLU-quantize, then down.
    prof.Begin(prof.gemm);
    if (w8) {
      LaunchQuantizeActivationQ8_1(bf16, arena_.d_scratch_q8_act, batch, hidden,
                                   arena_.stream);
      CheckGpu("glimmer prefill ffn q8 failed");
      if (layer.ffn_gate.type == layer.ffn_up.type) {
        LaunchBatchedDualQuantGEMMPreQuantized(
            layer.ffn_gate.type, layer.ffn_gate.data, layer.ffn_up.data,
            arena_.d_scratch_q8_act, arena_.d_ffn_gate, arena_.d_ffn_up, batch,
            intermediate, hidden, arena_.stream);
        CheckGpu("glimmer prefill dual ffn gemm failed");
      } else {
        ProjectQ8(arena_, layer.ffn_gate, arena_.d_scratch_q8_act,
                  arena_.d_ffn_gate, batch, intermediate, hidden);
        ProjectQ8(arena_, layer.ffn_up, arena_.d_scratch_q8_act, arena_.d_ffn_up,
                  batch, intermediate, hidden);
      }
      LaunchBatchedFusedSwiGLUQuantizeQ8_1(
          arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_scratch_q8_act, batch,
          intermediate, arena_.stream);
      CheckGpu("glimmer prefill swiglu q8 failed");
      ProjectQ8(arena_, layer.ffn_down, arena_.d_scratch_q8_act, arena_.d_ffn_out,
                batch, hidden, intermediate);
    } else {
      Project(arena_, layer.ffn_gate, arena_.d_normed, bf16, arena_.d_ffn_gate,
              batch, intermediate, hidden, weight_elems);
      Project(arena_, layer.ffn_up, arena_.d_normed, bf16, arena_.d_ffn_up, batch,
              intermediate, hidden, weight_elems);
      LaunchBatchedSwiGLUActivation(arena_.d_ffn_gate, arena_.d_ffn_up,
                                    arena_.d_ffn_act,
                                    batch == 1 ? nullptr : static_cast<void*>(bf16),
                                    batch * intermediate, arena_.stream);
      CheckGpu("glimmer prefill swiglu failed");
      Project(arena_, layer.ffn_down, arena_.d_ffn_act, bf16, arena_.d_ffn_out,
              batch, hidden, intermediate, weight_elems);
    }
    prof.End(prof.gemm);
    LaunchBatchedRMSNorm(arena_.d_ffn_out,
                         static_cast<const float*>(layer.ffn_post_norm.data),
                         arena_.d_normed, nullptr, batch, hidden,
                         config.norm_eps_post, arena_.stream);
    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_normed, arena_.d_hidden,
                             batch, hidden, arena_.stream);
    CheckGpu("glimmer prefill ffn residual failed");
  }

  last_hidden_offset_ = (batch - 1) * hidden;
  if (!compute_logits) {
    prof.MarkWall();
    if (hipStreamSynchronize(arena_.stream) != hipSuccess) {
      throw std::runtime_error("glimmer prefill synchronize failed");
    }
    prof.Finish(prof_total, start_pos, batch, false, w8, flop_per_pair);
    return 0;
  }
  const float* final_hidden = arena_.d_hidden + last_hidden_offset_;
  LaunchRMSNorm(final_hidden, static_cast<const float*>(weights_.output_norm.data),
                arena_.d_normed, hidden, config.norm_eps_pre, arena_.stream);
  LaunchGEMV(weights_.output.data, weights_.output.type, arena_.d_normed,
             arena_.d_logits, config.vocab_size, hidden, arena_.stream);
  const auto cap = LaunchGlimmerScaleSoftcap(
      arena_.d_logits, config.vocab_size, config.logit_scale, config.final_softcap,
      arena_.stream);
  if (!cap.ok) {
    throw std::runtime_error(cap.error);
  }
  LaunchBatchedGPUArgmax(
      arena_.d_logits, reinterpret_cast<std::uint32_t*>(arena_.d_alpha_buf), 1,
      config.vocab_size,
      std::span<float>(arena_.d_ffn_out, arena_.GetMaxBatch() * hidden),
      arena_.stream);
  std::uint32_t next_token_id = 0;
  prof.MarkWall();
  if (hipMemcpyAsync(&next_token_id, arena_.d_alpha_buf, sizeof(std::uint32_t),
                     hipMemcpyDeviceToHost, arena_.stream) != hipSuccess ||
      hipStreamSynchronize(arena_.stream) != hipSuccess) {
    throw std::runtime_error("glimmer prefill token read failed");
  }
  prof.Finish(prof_total, start_pos, batch, true, w8, flop_per_pair);
  return CheckedSampleToken(next_token_id, config.vocab_size);
}

void QwenGpuExecutor::MaterializeGlimmerRowLogits(std::size_t row) {
  CheckReset();
  const auto& config = weights_.config;
  if (!config.attention_only) {
    throw std::runtime_error("glimmer row logits called for a qwen config");
  }
  if (row >= arena_.GetMaxBatch()) {
    throw std::runtime_error("glimmer row logits row is past the batch");
  }
  const std::size_t hidden = config.hidden_size;
  const float* src = arena_.d_hidden + row * hidden;
  LaunchRMSNorm(src, static_cast<const float*>(weights_.output_norm.data),
                arena_.d_normed, hidden, config.norm_eps_pre, arena_.stream);
  LaunchGEMV(weights_.output.data, weights_.output.type, arena_.d_normed,
             arena_.d_logits, config.vocab_size, hidden, arena_.stream);
  const auto cap = LaunchGlimmerScaleSoftcap(
      arena_.d_logits, config.vocab_size, config.logit_scale, config.final_softcap,
      arena_.stream);
  if (!cap.ok) {
    throw std::runtime_error(cap.error);
  }
  if (hipStreamSynchronize(arena_.stream) != hipSuccess) {
    throw std::runtime_error("glimmer row logits synchronize failed");
  }
}

}  // namespace superroc::hip
#endif
