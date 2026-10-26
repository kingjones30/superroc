#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/rocm_fp4/fp4_f16_prefill.hpp"   // SQ1: ported FP4->F16 GEMMs
#include "src/core/rocm_fp4/gufo_fp4_gemv_q8.hpp"  // KAT:
#include "src/core/rocm_fp4/gufo_fp4_quant.hpp"    // KAT:
#include "src/models/qwen/gemm_route.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "src/models/qwen/moe/qwen_moe_ffn.hpp"      // KAT:
#include "src/models/qwen/moe/qwen_moe_prefill.hpp"  // KAT:
#include "src/models/qwen/moe/qwen_moe_route.hpp"    // KAT:
#include "src/models/qwen/qwen_fp4_f16_gate.hpp"

namespace gufo::hip {
// SQ1: the promoted FP4->F16 prefill launchers (Flash-Next namespace kept
// for kernel symbol stability; see fp4_f16_prefill.hpp).
namespace fp4w = ::gufo::models::qwen38_flash_next::rocm;
namespace {

// KAT: default ON only when the grouped path can run the shape
// (n_expert <= 256 and n_used <= 8). Otherwise, and when KAT_MOE_GROUP=0,
// the caller keeps the m <= 4 expert loop. No throw on a wider MoE.
bool KatMoeGrouped(int n_expert, int n_used);

// KAT: env-gated prefill stage timers. KAT_PREFILL_PROF=1 records HIP events
// on the arena stream and prints one line per chunk. Off, Arm/Take are empty.
struct KatPrefillProf {
  bool on = false;
  hipStream_t stream = nullptr;
  hipEvent_t a = nullptr;
  hipEvent_t b = nullptr;
  bool armed = false;
  double attn_ms = 0, ssm_ms = 0, route_ms = 0, routed_ms = 0, shared_ms = 0;
  double sigmoid_ms = 0, combine_ms = 0, norm_ms = 0, head_ms = 0;
  int attn_layers = 0, ssm_layers = 0;

  explicit KatPrefillProf(hipStream_t s) : stream(s) {
    const char* e = std::getenv("KAT_PREFILL_PROF");
    on = e != nullptr && e[0] == '1' && e[1] == '\0';
    if (!on)
      return;
    if (hipEventCreate(&a) != hipSuccess || hipEventCreate(&b) != hipSuccess) {
      on = false;
    }
  }
  ~KatPrefillProf() {
    if (a)
      hipEventDestroy(a);
    if (b)
      hipEventDestroy(b);
  }
  KatPrefillProf(const KatPrefillProf&) = delete;
  void Arm() {
    if (!on)
      return;
    hipEventRecord(a, stream);
    armed = true;
  }
  double Take() {
    if (!on || !armed)
      return 0;
    hipEventRecord(b, stream);
    hipEventSynchronize(b);
    float ms = 0;
    hipEventElapsedTime(&ms, a, b);
    armed = false;
    return static_cast<double>(ms);
  }
  void AddMixer(bool full_attn) {
    const double ms = Take();
    if (full_attn) {
      attn_ms += ms;
      ++attn_layers;
    } else {
      ssm_ms += ms;
      ++ssm_layers;
    }
  }
  void Dump(std::size_t batch, std::uint32_t start,
            const models::QwenModelWeights& weights) const {
    if (!on)
      return;
    const auto& cfg = weights.config;
    double routed_b = 0, shared_b = 0, router_b = 0;
    const int tokens = static_cast<int>(batch);
    const int nexp_cfg = static_cast<int>(cfg.expert_count);
    const int used_cfg = static_cast<int>(cfg.expert_used_count);
    const bool grouped = KatMoeGrouped(nexp_cfg, used_cfg);
    if (!weights.layers.empty() && weights.layers[0].is_moe && tokens > 0) {
      const auto& layer = weights.layers[0];
      const int qt = static_cast<int>(layer.ffn_gate_exps.type);
      const int qts = static_cast<int>(layer.ffn_gate_shexp.type);
      const int hidden = static_cast<int>(cfg.hidden_size);
      const int ff = static_cast<int>(cfg.expert_feed_forward_length);
      const int ffs = static_cast<int>(cfg.expert_shared_feed_forward_length);
      const int used = static_cast<int>(cfg.expert_used_count);
      const int nexp = static_cast<int>(cfg.expert_count);
      const double expert =
          2.0 * ff * static_cast<double>(gufo_fp4_row_bytes(qt, hidden)) +
          static_cast<double>(hidden) * gufo_fp4_row_bytes(qt, ff);
      const double shared_w =
          2.0 * ffs * static_cast<double>(gufo_fp4_row_bytes(qts, hidden)) +
          static_cast<double>(hidden) * gufo_fp4_row_bytes(qts, ffs);
      if (!grouped) {
        routed_b = static_cast<double>(tokens) * used * expert;
        shared_b = static_cast<double>((tokens + 3) / 4) * shared_w;
      } else {
        for (int t0 = 0; t0 < tokens; t0 += 1024) {
          const int m = std::min(1024, tokens - t0);
          const int tiles = (m * used + nexp * 16 - 1) / std::max(nexp * 16, 1);
          routed_b += static_cast<double>(nexp) * tiles * expert;
        }
        shared_b = static_cast<double>((tokens + 63) / 64) * shared_w;
      }
      router_b = static_cast<double>(nexp) * hidden * sizeof(float);
      // Stage timers sum every layer. The byte model above is one layer.
      const double layers = static_cast<double>(cfg.num_layers);
      routed_b *= layers;
      shared_b *= layers;
      router_b *= layers;
    }
    const double positions = static_cast<double>(batch) *
                             (2.0 * start + static_cast<double>(batch) + 1.0) /
                             2.0;
    const double attn_kv = static_cast<double>(attn_layers) * positions * 2.0 *
                           cfg.num_key_value_heads * cfg.head_dim *
                           sizeof(float);
    const auto gbps = [](double bytes, double ms) {
      return ms > 0.0 ? bytes / ms / 1.0e6 : 0.0;
    };
    const auto tps = [](double n, double ms) {
      return ms > 0.0 ? n / ms * 1.0e3 : 0.0;
    };
    const double tok = static_cast<double>(batch);
    std::printf(
        "KAT_PREFILL_PROF chunk=%zu start=%u group=%d attn_layers=%d "
        "ssm_layers=%d attn_ms=%.3f ssm_ms=%.3f route_ms=%.3f sigmoid_ms=%.3f "
        "routed_ms=%.3f shared_ms=%.3f combine_ms=%.3f norm_ms=%.3f "
        "head_ms=%.3f\n",
        batch, start, grouped ? 1 : 0, attn_layers, ssm_layers, attn_ms, ssm_ms,
        route_ms, sigmoid_ms, routed_ms, shared_ms, combine_ms, norm_ms,
        head_ms);
    // attn_kv_model_bytes is a byte estimate, not traffic the WMMA kernel
    // moves, so it is not reported as a GB/s.
    std::printf(
        "KAT_PREFILL_BYTES chunk=%zu routed=%.0f shared=%.0f router=%.0f "
        "attn_kv_model_bytes=%.0f routed_gbps=%.2f shared_gbps=%.2f "
        "route_gbps=%.2f routed_tok_s=%.1f shared_tok_s=%.1f attn_tok_s=%.1f "
        "ssm_tok_s=%.1f\n",
        batch, routed_b, shared_b, router_b, attn_kv, gbps(routed_b, routed_ms),
        gbps(shared_b, shared_ms), gbps(router_b, route_ms),
        tps(tok, routed_ms), tps(tok, shared_ms), tps(tok, attn_ms),
        tps(tok, ssm_ms));
    std::fflush(stdout);
  }
};

// KAT: default ON only for the shape the grouped kernels accept.
// KAT_MOE_GROUP=0 keeps the m <= 4 expert loop. A wider MoE falls back
// to that loop instead of throwing.
bool KatMoeGrouped(int n_expert, int n_used) {
  static const bool on = [] {
    const char* e = std::getenv("KAT_MOE_GROUP");
    return e == nullptr || e[0] != '0';
  }();
  return on && n_expert > 0 && n_expert <= 256 && n_used > 0 && n_used <= 8;
}

// SQ1: ROC FP4 (100/101) prefill on the promoted F16 WMMA kernels.
bool IsQwenFp4(core::GgmlType type) {
  return type == core::GgmlType::kQ4_0_ROCMFP4 ||
         type == core::GgmlType::kQ4_0_ROCMFP4_FAST;
}

// FRZA SPEC2: GUFO_QWEN_FP4_F16_PREFILL is tri-state now. Unset/1 = the
// fail-closed gate, 0 = today's kill switch (unchanged), 2 = force the tag
// engine's F16 route for every FP4 file (one controlled experiment later).
int QwenFp4F16PrefillMode() {
  static const int mode = models::qwen::ParseQwenFp4F16PrefillMode(
      std::getenv("GUFO_QWEN_FP4_F16_PREFILL"));
  return mode;
}

// SQ1: GUFO_QWEN_FP4_PORTED (default off) routes FP4 tensors through the
// SPEC1 ported kernels instead of the stock F16 kernel. The ported path is
// kept for A/B only; the shippable FP4 route is the F16 kernel (SPEC2 §2).
bool QwenFp4PortedOn() {
  static const bool on = [] {
    const char* e = std::getenv("GUFO_QWEN_FP4_PORTED");
    return e != nullptr && e[0] != '0';
  }();
  return on;
}

bool UseQwen27bFp16Prefill(const models::QwenModelWeights& weights,
                           std::size_t batch) {
  const auto& config = weights.config;
  // Large Qwen27B low-bit prefill is qualified independently from the Q8
  // target and from the small-batch decode/verification kernels.
  if (batch < 1024)
    return false;
  const auto supported = [](const models::QwenTensorRef& tensor) {
    // lqw:fp16-prefill. Types 100/101 stay off the fp16 prefill path.
    return (tensor.type == core::GgmlType::kQ8_0 ||
            detail::IsNativeWmmaQuant(tensor.type)) &&
           tensor.type != core::GgmlType::kQ4_0_ROCMFP4 &&
           tensor.type != core::GgmlType::kQ4_0_ROCMFP4_FAST;
  };
  // SQ1: the checked tensor set is unchanged (FFN always; mixer per layer
  // kind). Files with no FP4 tensor keep today's logic verbatim below, so
  // stock files are bit-identical with the switch on or off. MoE layers
  // keep the grouped expert path (KAT is already 4.9x the fork there).
  for (const auto& layer : weights.layers) {
    if (layer.is_moe)
      return false;
  }
  // FRZA SPEC2: the route gate is pure (host-tested in
  // tests/models/qwen/fp4_f16_gate_test.cpp) and shared with the test.
  const int fp4_mode = QwenFp4F16PrefillMode();
  std::vector<models::qwen::QwenFp4F16LayerTypes> fp4_mix;
  fp4_mix.reserve(weights.layers.size());
  for (const auto& layer : weights.layers) {
    models::qwen::QwenFp4F16LayerTypes mix;
    mix.is_full_attention = layer.is_full_attention;
    mix.ffn_gate = layer.ffn_gate.type;
    mix.ffn_up = layer.ffn_up.type;
    mix.ffn_down = layer.ffn_down.type;
    mix.attn_q = layer.attn_q.type;
    mix.attn_k = layer.attn_k.type;
    mix.attn_v = layer.attn_v.type;
    mix.attn_output = layer.attn_output.type;
    mix.attn_qkv = layer.attn_qkv.type;
    mix.attn_gate = layer.attn_gate.type;
    mix.ssm_alpha = layer.ssm_alpha.type;
    mix.ssm_beta = layer.ssm_beta.type;
    mix.ssm_out = layer.ssm_out.type;
    fp4_mix.push_back(mix);
  }
  const models::qwen::QwenFp4F16Decision fp4_gate =
      models::qwen::DecideQwenFp4F16Prefill(fp4_mode, fp4_mix);
  if (!fp4_gate.has_fp4 || fp4_mode == models::qwen::kQwenFp4F16PrefillOff) {
    if (config.hidden_size != 5120 || config.intermediate_size != 17408 ||
        config.num_layers != 64 || config.AttentionSize() != 6144 ||
        config.ssm_inner_size != 6144)
      return false;
    bool low_bit_ffn = false;
    for (const auto& layer : weights.layers) {
      if (!supported(layer.ffn_gate) || !supported(layer.ffn_up) ||
          !supported(layer.ffn_down))
        return false;
      low_bit_ffn |= detail::IsNativeWmmaQuant(layer.ffn_gate.type) ||
                     detail::IsNativeWmmaQuant(layer.ffn_up.type) ||
                     detail::IsNativeWmmaQuant(layer.ffn_down.type);
      if (layer.is_full_attention) {
        if (!supported(layer.attn_q) || !supported(layer.attn_k) ||
            !supported(layer.attn_v) || !supported(layer.attn_output))
          return false;
      } else if (!supported(layer.attn_qkv) || !supported(layer.attn_gate) ||
                 !supported(layer.ssm_alpha) || !supported(layer.ssm_beta) ||
                 !supported(layer.ssm_out))
        return false;
    }
    return low_bit_ffn;
  }
  if (fp4_gate.route == models::qwen::QwenFp4F16Route::kLegacy) {
    // FRZA SPEC2: fail-closed refusal (gated mode only). Returning false
    // takes exactly the pre-f85008b dispatch below (half_prefill=false).
    // One line per process, on the first refusing chunk.
    static std::atomic<bool> frza_refusal_logged{false};
    if (!frza_refusal_logged.exchange(true)) {
      std::fprintf(stderr,
                   "frza: GUFO_QWEN_FP4_F16_PREFILL=1 refuses the F16 prefill "
                   "route: type-100 on %s at layer %d; using the pre-f85008b "
                   "dispatch\n",
                   fp4_gate.family, fp4_gate.layer);
    }
    return false;
  }
  // SQ1 capability for files with FP4 tensors: every checked tensor keeps a
  // supported route (FP4 -> the F16 launchers, or the ported kernels when
  // GUFO_QWEN_FP4_PORTED=1; the rest -> the F16 launchers), every K is
  // non-zero and a multiple of 32 (the ported kernels' contract; any batch
  // and m), and K % 256 == 0 holds wherever a tensor needs DirectGemm's
  // contract (SQ1: all 27B/MiMo K dims qualify, so FP4 always takes F16).
  const std::size_t kdims[] = {config.hidden_size, config.intermediate_size,
                               config.AttentionSize(), config.ssm_inner_size};
  for (std::size_t k : kdims) {
    if (k == 0 || k % 32 != 0)
      return false;
  }
  bool non_fp4_ok = true;
  for (std::size_t k : kdims) {
    if (k % 256 != 0)
      non_fp4_ok = false;
  }
  for (const auto& layer : weights.layers) {
    const models::QwenTensorRef* ffn[] = {&layer.ffn_gate, &layer.ffn_up,
                                          &layer.ffn_down};
    for (const auto* t : ffn) {
      if (IsQwenFp4(t->type))
        continue;
      if (!supported(*t) || !non_fp4_ok)
        return false;
    }
    if (layer.is_full_attention) {
      const models::QwenTensorRef* ts[] = {&layer.attn_q, &layer.attn_k,
                                           &layer.attn_v, &layer.attn_output};
      for (const auto* t : ts) {
        if (IsQwenFp4(t->type))
          continue;
        if (!supported(*t) || !non_fp4_ok)
          return false;
      }
    } else {
      const models::QwenTensorRef* ts[] = {&layer.attn_qkv, &layer.attn_gate,
                                           &layer.ssm_alpha, &layer.ssm_beta,
                                           &layer.ssm_out};
      for (const auto* t : ts) {
        if (IsQwenFp4(t->type))
          continue;
        if (!supported(*t) || !non_fp4_ok)
          return false;
      }
    }
  }
  return true;
}
}  // namespace

tokenization::TokenId QwenGpuExecutor::ForwardPromptChunk(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  CheckReset();
  const auto& config = weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t batch_size = prompt_tokens.size();
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;
  const float eps = 1e-6F;

  if (batch_size == 0) {
    return 0;
  }
  if (batch_size > arena_.GetMaxBatch()) {
    throw std::length_error("prompt chunk exceeds the GPU batch length");
  }
  // GLIMMER: sandwich norms, separate gate, NORM RoPE, SWA. Not the qwen path.
  if (config.attention_only) {
    return ForwardGlimmerPromptChunk(prompt_tokens, start_pos, compute_logits);
  }
  if (config.lfm2) {  // LFM2
    return ForwardLfm2PromptChunk(prompt_tokens, start_pos, compute_logits);
  }
  if (config.flint) {
    return ForwardFlintPromptChunk(prompt_tokens, start_pos, compute_logits);
  }
  if (config.llama) {
    return ForwardLlamaPromptChunk(prompt_tokens, start_pos, compute_logits);
  }
  auto scratch = arena_.GetScratchView(batch_size);
  const auto attention_workspace =
      arena_.GetScratchView(arena_.GetMaxBatch()).ffn;
  const std::size_t target_layer_count = arena_.GetTargetLayerCapture().size();
  const std::size_t prompt_capture_offset = h_prompt_hidden_.size();
  if (capture_prompt_hidden_ && target_layer_count > 0) {
    h_prompt_hidden_.resize(prompt_capture_offset +
                            batch_size * target_layer_count * hidden_size);
  }

  // 1. Copy prompt token IDs to GPU
  std::vector<std::uint32_t> host_tokens(prompt_tokens.begin(),
                                         prompt_tokens.end());
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                           host_tokens.data(),
                           batch_size * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, arena_.stream));

  // 2. Batched Embedding lookup: d_hidden [B, hidden_size]
  LaunchBatchedEmbeddingLookup(
      weights_.token_embd.data, weights_.token_embd.type,
      scratch.decode.prompt_tokens.data(), scratch.decode.hidden.data(),
      batch_size, hidden_size, arena_.stream);

  vision_input_.Inject(scratch.decode.hidden.data(), start_pos, batch_size,
                       hidden_size, 1, arena_.stream);

  const bool half_prefill = UseQwen27bFp16Prefill(weights_, batch_size);
  const auto gemm_weight = [&](const models::QwenTensorRef& w,
                               const void* bf16_input, const float* fp32_input,
                               float* output, std::size_t m, std::size_t k,
                               const void* q8_act = nullptr) {
    if (half_prefill) {
      if (IsQwenFp4(w.type) && QwenFp4PortedOn()) {
        // SQ1: ROC FP4 prefill on the promoted F16 WMMA kernels. bf16_input
        // holds F16 here (LaunchBatchedRMSNormFp16 / the F16 mixer outputs).
        // SQ1: ported only when GUFO_QWEN_FP4_PORTED=1; otherwise FP4 falls
        // into the stock F16 launcher below (DirectGemm FP4 cases).
        const bool fast = w.type == core::GgmlType::kQ4_0_ROCMFP4_FAST;
        if (!fp4w::Fp4DenseF16Gemm(
                w.data, static_cast<const __half*>(bf16_input), output,
                batch_size, m, k, fast, arena_.stream)) {
          std::fprintf(stderr,
                       "sq1: Fp4DenseF16Gemm declined batch=%zu m=%zu k=%zu\n",
                       batch_size, m, k);
          std::abort();
        }
        return;
      }
      LaunchBatchedQuantGEMMFp16(w.type, w.data, bf16_input, output, batch_size,
                                 m, k, arena_.stream);
      return;
    }
    const auto resolution = models::qwen::ResolveQwenGemmRoute(
        {.type = w.type,
         .batch_size = batch_size,
         .m = m,
         .k = k,
         .mode = models::qwen::QwenGemmMode::kHipPrefill});
    if (!resolution.accepted()) {
      std::abort();
    }
    switch (resolution.route) {
      case models::qwen::QwenGemmRoute::kHipPrefillBf16Fp32:
        // Preserve FP32 activation precision and a fixed reduction order.
        LaunchExactBf16GEMMFp32SmallBatch(w.data, fp32_input, output,
                                          batch_size, m, k, arena_.stream);
        return;
      case models::qwen::QwenGemmRoute::kHipPrefillF32Blas:
        LaunchHipblasGEMM(arena_.hipblas_handle, w.data, false, fp32_input,
                          output, batch_size, m, k, arena_.d_scratch_bf16,
                          arena_.stream);
        return;
      case models::qwen::QwenGemmRoute::kHipPrefillQuantDirect: {
        if (w.type == core::GgmlType::kQ8_0 ||
            detail::IsNativeWmmaQuant(w.type)) {
          // Route Q8_0 and every K-quant/IQ format the UD-Q4_K_XL shard uses
          // through the native W-quant x A8 WMMA matrix-core kernel with zero
          // scratch dequantization. Without the second half of this condition
          // the K-quants fell into the branch below, which dequantizes the
          // whole weight matrix to a BF16 scratch buffer and then runs a BF16
          // hipBLASLt GEMM -- paying the dequant AND giving up the 2x matrix
          // -core throughput of the int8 path.
          if (q8_act != nullptr) {
            LaunchBatchedQuantGEMMPreQuantized(w.type, w.data, q8_act, output,
                                               batch_size, m, k, arena_.stream);
          } else {
            LaunchBatchedQuantGEMM(w.type, w.data, bf16_input, output,
                                   batch_size, m, k, arena_.stream);
          }
        } else if (batch_size > 1) {
          LaunchDequantizeToBf16(w.type, w.data, arena_.d_weights_bf16, m * k,
                                 arena_.stream);
          if (arena_.hipblaslt_gemm != nullptr && m >= 1024 && k >= 1024 &&
              arena_.hipblaslt_gemm->RunBf16(arena_.d_weights_bf16, bf16_input,
                                             output, batch_size, m, k,
                                             arena_.stream)) {
            // hipBLASLt executed successfully
          } else {
            LaunchHipblasGEMMBF16(arena_.hipblas_handle, arena_.d_weights_bf16,
                                  bf16_input, output, batch_size, m, k,
                                  arena_.stream);
          }
        } else {
          LaunchBatchedQuantGEMM(w.type, w.data, bf16_input, output, batch_size,
                                 m, k, arena_.stream);
        }
        return;
      }
      default:
        std::abort();
    }
  };

  KatPrefillProf kat_prof(arena_.stream);  // KAT:
  // 3. Layer stack
  // 3. Layer stack. One recurrent pass per chunk.
  const auto recurrent_state = arena_.BeginRecurrentWrite();
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    PrefillLayerCheckpoint(l);
    const auto& layer = weights_.layers[l];
    const auto route_resolution = ResolveQwenLayerRouteWithReasons(
        policy_, QwenExecutionMode::kPrefill, layer.is_full_attention);
    const auto& route_plan = route_resolution.plan;
    detail::EmitQwenRouteResolution(
        "prefill", l, layer.is_full_attention ? "attention" : "ssm",
        route_plan.Fingerprint(),
        static_cast<std::uint32_t>(route_resolution.rejected));

    // opt-q4kxl: "reads the tiled Q8_1 activation" is the property every gate
    // below actually cares about, and it is no longer synonymous with Q8_0.
    // The K-quant/IQ formats in the UD-Q4_K_XL shard take the same
    // pre-quantized WMMA route, so they must drive the same fusion decisions --
    // a gate left at `== kQ8_0` would leave the activation buffer unwritten
    // under a route that reads it.
    const auto reads_q8_act = [&](const models::QwenTensorRef& w) {
      return !half_prefill && (w.type == core::GgmlType::kQ8_0 ||
                               detail::IsNativeWmmaQuant(w.type));
    };
    const auto is_q8 = reads_q8_act;
    // opt-c173-norm-quant: when every projection reading a norm is Q8_0 the
    // FP32 normed row and the BF16 staging copy are both dead, so the norm can
    // write the tiled Q8_1 activation directly and skip two round trips.
    const bool norm_feeds_q8_only =
        IsFusedRMSNormQuantizeQ8_1Supported(hidden_size) &&
        (layer.is_full_attention
             ? (is_q8(layer.attn_q) && is_q8(layer.attn_k) &&
                is_q8(layer.attn_v))
             : (is_q8(layer.attn_qkv) && is_q8(layer.attn_gate) &&
                is_q8(layer.ssm_alpha) && is_q8(layer.ssm_beta)));
    // Where these hold, the BF16 and FP32 buffers passed to gemm_weight below
    // are stale: the route for a Q8_0 tensor always reads the quantized
    // activation, and the flags are exactly the condition that every consumer
    // is Q8_0, so no route can reach them.
    const bool ffn_feeds_q8_only =
        IsFusedRMSNormQuantizeQ8_1Supported(hidden_size) &&
        is_q8(layer.ffn_gate) && is_q8(layer.ffn_up);

    kat_prof.Arm();  // KAT:
    // Pre-layer RMSNorm (generates BF16 into d_scratch_bf16 directly)
    if (half_prefill) {
      LaunchBatchedRMSNormFp16(arena_.d_hidden, nullptr,
                               static_cast<const float*>(layer.attn_norm.data),
                               nullptr, arena_.d_scratch_bf16, batch_size,
                               hidden_size, eps, arena_.stream);
    } else if (norm_feeds_q8_only) {
      LaunchBatchedFusedRMSNormQuantizeQ8_1(
          arena_.d_hidden, /*residual=*/nullptr,
          static_cast<const float*>(layer.attn_norm.data), /*sum_out=*/nullptr,
          arena_.d_scratch_q8_act, batch_size, hidden_size, eps, arena_.stream);
    } else {
      LaunchBatchedRMSNorm(arena_.d_hidden,
                           static_cast<const float*>(layer.attn_norm.data),
                           arena_.d_normed, arena_.d_scratch_bf16, batch_size,
                           hidden_size, eps, arena_.stream);
    }

    if (layer.is_full_attention) {
      // The quantized activation is read only by Q8_0 projections; when q/k/v
      // are all BF16 (as in the Q8_K_XL artifact) nothing consumes it, and when
      // they are all Q8_0 the fused norm already wrote it.
      if (!norm_feeds_q8_only &&
          (reads_q8_act(layer.attn_q) || reads_q8_act(layer.attn_k) ||
           reads_q8_act(layer.attn_v))) {
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     hidden_size, arena_.stream);
      }
      gemm_weight(layer.attn_q, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ssm_qkv, q_projection_size, hidden_size,
                  arena_.d_scratch_q8_act);
      gemm_weight(layer.attn_k, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_k, kv_size, hidden_size, arena_.d_scratch_q8_act);
      gemm_weight(layer.attn_v, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_v, kv_size, hidden_size, arena_.d_scratch_q8_act);

      LaunchBatchedUnpackQG(arena_.d_ssm_qkv, arena_.d_q, arena_.d_ssm_gate,
                            batch_size, config.num_attention_heads,
                            config.head_dim, arena_.stream);

      const std::size_t total_k = arena_.GetAttentionKvPlaneElements();
      const std::uint32_t attn_layer_idx = l / config.full_attention_interval;

      // QK-Norm + RoPE + KV-cache write fused into one kernel
      // (opt-c010-qk-rope-kv). The unfused chain stays wired behind the policy
      // toggle as the independent reference.
      const bool fused_qknorm_rope_kv =
          route_plan.fuse_qk_norm_rope_kv &&
          detail::IsFusedQkNormSupported(config.head_dim);
      if (fused_qknorm_rope_kv) {
        LaunchBatchedFusedQKNormRoPEKvWrite(
            arena_.d_q, arena_.d_k, arena_.d_v,
            static_cast<const float*>(layer.attn_q_norm.data),
            static_cast<const float*>(layer.attn_k_norm.data), arena_.d_q,
            arena_.d_k, arena_.d_kv_cache,
            OffsetIfPresent(arena_.d_kv_cache, total_k),
            arena_.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(arena_.d_attention_kv_f16),
                total_k),
            attn_layer_idx, start_pos, batch_size, arena_.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, config.rotary_dim, config.rope_theta, eps,
            arena_.stream, vision_input_.rope());
      } else {
        if (!layer.attn_q_norm.empty()) {
          LaunchBatchedPerHeadRMSNorm(
              arena_.d_q, static_cast<const float*>(layer.attn_q_norm.data),
              arena_.d_q, batch_size, config.num_attention_heads,
              config.head_dim, eps, arena_.stream);
        }
        if (!layer.attn_k_norm.empty()) {
          LaunchBatchedPerHeadRMSNorm(
              arena_.d_k, static_cast<const float*>(layer.attn_k_norm.data),
              arena_.d_k, batch_size, config.num_key_value_heads,
              config.head_dim, eps, arena_.stream);
        }

        LaunchBatchedRoPE(
            arena_.d_q, arena_.d_k, batch_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, config.rotary_dim,
            start_pos, config.rope_theta, arena_.stream, vision_input_.rope());
      }

      const std::size_t visible_context =
          static_cast<std::size_t>(start_pos) + batch_size;
      // One masked WMMA pass covers the visible prefix and causal diagonal.
      // Unsupported shapes use the tiled or scalar fallback below.
      // opt-c180-kv-resync: the fused QK-norm/RoPE kernel above already wrote
      // this chunk's K and V into the selected canonical cache plane. Earlier
      // chunks and all three decode paths maintain that same plane. The
      // attention launcher's own pack pass would therefore rewrite identical
      // bytes, and an FP16 prefix sync would reconvert a prefix that already
      // matches -- work that grows linearly with depth. The unfused fallback
      // below only applies RoPE in place and never touches the cache, so it
      // still needs the pack.
      const bool kv_already_written = fused_qknorm_rope_kv;
      const bool wmma_attention = LaunchQwenWmmaAttention(
          arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
          arena_.d_kv_cache, OffsetIfPresent(arena_.d_kv_cache, total_k),
          arena_.d_attention_kv_f16,
          OffsetIfPresent(
              static_cast<std::uint16_t*>(arena_.d_attention_kv_f16), total_k),
          arena_.d_ssm_out, attn_layer_idx, start_pos, batch_size,
          arena_.GetMaxContext(), config.num_attention_heads,
          config.num_key_value_heads, config.head_dim, arena_.stream,
          /*lse_out=*/nullptr, /*key_begin=*/0,
          /*skip_kv_write=*/kv_already_written, attention_workspace.gate,
          attention_workspace.up);
      if (wmma_attention) {
        detail::EmitAttentionDispatch("prefill_wmma", "");
      }

      if (!wmma_attention) {
        bool tiled_attention = false;
        detail::DispatchPrefillAttention(
            visible_context,
            [&] {
              tiled_attention = LaunchBatchedAttentionTile(
                  arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
                  arena_.d_kv_cache,
                  OffsetIfPresent(arena_.d_kv_cache, total_k),
                  arena_.d_attention_kv_f16,
                  OffsetIfPresent(
                      static_cast<std::uint16_t*>(arena_.d_attention_kv_f16),
                      total_k),
                  arena_.d_ssm_out, attn_layer_idx, start_pos, batch_size,
                  arena_.GetMaxContext(), config.num_attention_heads,
                  config.num_key_value_heads, config.head_dim, arena_.stream);
              return tiled_attention;
            },
            [&] {
              LaunchBatchedAttention(
                  arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
                  arena_.d_kv_cache,
                  OffsetIfPresent(arena_.d_kv_cache, total_k),
                  arena_.d_attention_kv_f16,
                  OffsetIfPresent(
                      static_cast<std::uint16_t*>(arena_.d_attention_kv_f16),
                      total_k),
                  arena_.d_ssm_out, attn_layer_idx, start_pos, batch_size,
                  arena_.GetMaxContext(), config.num_attention_heads,
                  config.num_key_value_heads, config.head_dim, arena_.stream,
                  fused_qknorm_rope_kv);
            });
        if (tiled_attention) {
          detail::EmitAttentionDispatch("prefill_tiled", "");
        } else {
          detail::EmitAttentionDispatch(
              "prefill_baseline",
              detail::ShouldAttemptOptimizedAttention(visible_context)
                  ? "prefill_tiled: rejected"
                  : "prefill_tiled: below_threshold");
        }
      }

      // opt-c172-fp32-quant: the Q8_0 route reads only the quantized
      // activation, so the BF16 staging buffer is dead. Quantizing straight
      // from FP32 drops one launch and one round trip (about 75 MB per layer
      // at batch 2048) and keeps the activation's full precision going into the
      // Q8_1 codes instead of rounding to BF16 first.
      if (half_prefill) {
        LaunchFloatToFp16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                          batch_size * attention_size, arena_.stream);
      } else if (reads_q8_act(layer.attn_output)) {
        LaunchQuantizeActivationQ8_1FromFp32(
            arena_.d_ssm_out, arena_.d_scratch_q8_act, batch_size,
            attention_size, arena_.stream);
      } else {
        LaunchFloatToBfloat16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                              batch_size * attention_size, arena_.stream);
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     attention_size, arena_.stream);
      }
      if (half_prefill) {
        if (IsQwenFp4(layer.attn_output.type) && QwenFp4PortedOn()) {
          // SQ1: ported GEMM into the attention-output buffer, then the
          // residual add the non-half path does (d_attn_out is dead in
          // half mode otherwise). SQ1: F16 path takes the fused Residual
          // epilogue in the else arm (DirectGemm FP4 cases).
          gemm_weight(layer.attn_output, arena_.d_scratch_bf16, nullptr,
                      arena_.d_attn_out, hidden_size, attention_size);
          LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_attn_out,
                                   arena_.d_hidden, batch_size, hidden_size,
                                   arena_.stream);
        } else {
          LaunchBatchedQuantGEMMResidualFp16(
              layer.attn_output.type, layer.attn_output.data,
              arena_.d_scratch_bf16, arena_.d_hidden, batch_size, hidden_size,
              attention_size, arena_.stream);
        }
      } else {
        gemm_weight(layer.attn_output, arena_.d_scratch_bf16, arena_.d_ssm_out,
                    arena_.d_attn_out, hidden_size, attention_size,
                    arena_.d_scratch_q8_act);
      }
    } else {
      if (!half_prefill && !norm_feeds_q8_only) {
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     hidden_size, arena_.stream);
      }
      gemm_weight(layer.attn_qkv, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ssm_qkv, ssm_qkv_size, hidden_size,
                  arena_.d_scratch_q8_act);
      gemm_weight(layer.attn_gate, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ssm_gate, ssm_inner_size, hidden_size,
                  arena_.d_scratch_q8_act);
      gemm_weight(layer.ssm_alpha, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_alpha_buf, time_step_rank, hidden_size,
                  arena_.d_scratch_q8_act);
      gemm_weight(layer.ssm_beta, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_beta_buf, time_step_rank, hidden_size,
                  arena_.d_scratch_q8_act);
      if (arena_.IsSsmReplayCaptureActive()) {
        LaunchCaptureBatchedSsmReplay(
            arena_.d_ssm_qkv, arena_.d_alpha_buf, arena_.d_beta_buf,
            arena_.GetSsmReplayCapture(), config.SsmLayerIndex(l), start_pos,
            batch_size, ssm_qkv_size, time_step_rank, arena_.stream);
      }

      // The row-split recurrence parallelizes independent state rows. Its
      // epilogue emits the activation consumed by ssm_out directly.
      const bool ssm_epilogue_q8 = reads_q8_act(layer.ssm_out) &&
                                   IsFusedSSMEpilogueQuantizeQ8_1Supported(
                                       config.SsmValueSize(), ssm_inner_size);
      const bool ssm_row_split = detail::ShouldUseSsmRowSplitRecurrence(
          IsDeltaNetRowSplitSupported(config.ssm_state_size,
                                      config.SsmValueSize()),
          arena_.d_ssm_kq_scales != nullptr &&
              arena_.d_ssm_alpha_beta != nullptr);
      if (ssm_row_split) {
        LaunchBatchedSSMConvRecurrenceRowSplit(
            arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
            recurrent_state, arena_.d_conv_out, arena_.d_alpha_buf,
            arena_.d_beta_buf, static_cast<const float*>(layer.ssm_a.data),
            static_cast<const float*>(layer.ssm_dt.data),
            static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
            arena_.d_ssm_out,
            ssm_epilogue_q8 ? arena_.d_scratch_q8_act : nullptr,
            arena_.d_ssm_kq_scales, arena_.d_ssm_alpha_beta,
            config.SsmLayerIndex(l), batch_size, ssm_qkv_size,
            config.ssm_group_count, config.ssm_time_step_rank,
            config.ssm_state_size, config.SsmValueSize(), arena_.stream,
            arena_.GetRecurrentStateStorage(),
            half_prefill ? arena_.d_scratch_bf16 : nullptr);
      } else {
        LaunchBatchedSSMConvRecurrence(
            arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
            recurrent_state, arena_.d_conv_out, arena_.d_alpha_buf,
            arena_.d_beta_buf, static_cast<const float*>(layer.ssm_a.data),
            static_cast<const float*>(layer.ssm_dt.data),
            static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
            arena_.d_ssm_out, config.SsmLayerIndex(l), batch_size, ssm_qkv_size,
            config.ssm_group_count, config.ssm_time_step_rank,
            config.ssm_state_size, config.SsmValueSize(), arena_.stream,
            arena_.GetRecurrentStateStorage());
      }

      if (half_prefill) {
        if (!ssm_row_split) {
          LaunchFloatToFp16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                            batch_size * ssm_inner_size, arena_.stream);
        }
      } else if (ssm_row_split && ssm_epilogue_q8) {
        // The recurrence epilogue already wrote the quantized activation.
      } else if (reads_q8_act(layer.ssm_out)) {
        LaunchQuantizeActivationQ8_1FromFp32(
            arena_.d_ssm_out, arena_.d_scratch_q8_act, batch_size,
            ssm_inner_size, arena_.stream);
      } else {
        LaunchFloatToBfloat16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                              batch_size * ssm_inner_size, arena_.stream);
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     ssm_inner_size, arena_.stream);
      }
      if (half_prefill) {
        if (IsQwenFp4(layer.ssm_out.type) && QwenFp4PortedOn()) {
          // SQ1: ported GEMM into the attention-output buffer, then the
          // residual add the non-half path does (d_attn_out is dead in
          // half mode otherwise). SQ1: F16 path takes the fused Residual
          // epilogue in the else arm (DirectGemm FP4 cases).
          gemm_weight(layer.ssm_out, arena_.d_scratch_bf16, nullptr,
                      arena_.d_attn_out, hidden_size, ssm_inner_size);
          LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_attn_out,
                                   arena_.d_hidden, batch_size, hidden_size,
                                   arena_.stream);
        } else {
          LaunchBatchedQuantGEMMResidualFp16(
              layer.ssm_out.type, layer.ssm_out.data, arena_.d_scratch_bf16,
              arena_.d_hidden, batch_size, hidden_size, ssm_inner_size,
              arena_.stream);
        }
      } else {
        gemm_weight(layer.ssm_out, arena_.d_scratch_bf16, arena_.d_ssm_out,
                    arena_.d_attn_out, hidden_size, ssm_inner_size,
                    arena_.d_scratch_q8_act);
      }
    }

    kat_prof.AddMixer(layer.is_full_attention);  // KAT:
    if (layer.is_moe) {
      // KAT S6: grouped expert prefill. KAT_MOE_GROUP=0 keeps the m <= 4 loop
      // (each call re-reads up to 32 experts). The grouped path sorts the
      // chunk on device and reads each expert once per 16 of its tokens.
      kat_prof.Arm();
      LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_attn_out,
                               arena_.d_hidden, batch_size, hidden_size,
                               arena_.stream);
      LaunchBatchedRMSNorm(arena_.d_hidden,
                           static_cast<const float*>(layer.ffn_norm.data),
                           arena_.d_normed, arena_.d_scratch_bf16, batch_size,
                           hidden_size, eps, arena_.stream);
      kat_prof.norm_ms += kat_prof.Take();
      const int moe_expert = static_cast<int>(config.expert_count);
      const int moe_used = static_cast<int>(config.expert_used_count);
      const int moe_ff = static_cast<int>(config.expert_feed_forward_length);
      const int moe_ffs =
          static_cast<int>(config.expert_shared_feed_forward_length);
      const int moe_hidden = static_cast<int>(hidden_size);
      const int moe_tokens = static_cast<int>(batch_size);
      kat_prof.Arm();
      if (LaunchQwenMoeRouteBatch(
              arena_.d_normed,
              static_cast<const float*>(layer.ffn_gate_inp.data), moe_hidden,
              moe_expert, moe_tokens, moe_used, arena_.d_moe_logits,
              arena_.d_moe_ids, arena_.d_moe_weights, arena_.stream) != 0) {
        throw std::runtime_error("qwen35moe prefill route launch failed");
      }
      kat_prof.route_ms += kat_prof.Take();
      kat_prof.Arm();
      if (LaunchQwenMoeShexpSigmoid(
              arena_.d_normed,
              static_cast<const float*>(layer.ffn_gate_inp_shexp.data),
              moe_hidden, moe_tokens, arena_.d_moe_sig, arena_.stream) != 0) {
        throw std::runtime_error("qwen35moe prefill sigmoid launch failed");
      }
      kat_prof.sigmoid_ms += kat_prof.Take();
      if (KatMoeGrouped(moe_expert, moe_used)) {
        kat_prof.Arm();
        if (LaunchKatMoeRoutedGrouped(
                static_cast<int>(layer.ffn_gate_exps.type),
                layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
                layer.ffn_down_exps.data, arena_.d_normed, arena_.d_moe_ids,
                arena_.d_moe_weights, moe_tokens, moe_used, moe_expert,
                moe_hidden, moe_ff, arena_.d_moe_y, arena_.stream) != 0) {
          throw std::runtime_error("qwen35moe grouped prefill experts failed");
        }
        kat_prof.routed_ms += kat_prof.Take();
        kat_prof.Arm();
        if (LaunchKatMoeSharedDense(
                static_cast<int>(layer.ffn_gate_shexp.type),
                layer.ffn_gate_shexp.data, layer.ffn_up_shexp.data,
                layer.ffn_down_shexp.data, arena_.d_normed, arena_.d_moe_sig,
                moe_tokens, moe_hidden, moe_ffs, arena_.d_moe_shexp_y,
                arena_.stream) != 0) {
          throw std::runtime_error("qwen35moe grouped prefill shared failed");
        }
        kat_prof.shared_ms += kat_prof.Take();
      } else {
        for (int t0 = 0; t0 < moe_tokens; t0 += 4) {
          const int m = std::min(4, moe_tokens - t0);
          const std::size_t off = static_cast<std::size_t>(t0) * hidden_size;
          kat_prof.Arm();
          if (gufo_fp4_expert_ffn_m(
                  static_cast<int>(layer.ffn_gate_exps.type),
                  layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
                  layer.ffn_down_exps.data, arena_.d_normed + off,
                  arena_.d_moe_ids + static_cast<std::size_t>(t0) * moe_used,
                  arena_.d_moe_weights +
                      static_cast<std::size_t>(t0) * moe_used,
                  m, moe_used, moe_expert, moe_hidden, moe_ff, 0.0F,
                  arena_.d_moe_y + off, arena_.stream) != GUFO_FP4_OK) {
            throw std::runtime_error("qwen35moe prefill experts failed");
          }
          kat_prof.routed_ms += kat_prof.Take();
          kat_prof.Arm();
          if (gufo_fp4_expert_ffn_shared_m(
                  static_cast<int>(layer.ffn_gate_shexp.type),
                  layer.ffn_gate_shexp.data, layer.ffn_up_shexp.data,
                  layer.ffn_down_shexp.data, arena_.d_normed + off,
                  arena_.d_moe_zero_ids + t0, arena_.d_moe_sig + t0, m, 1, 1,
                  moe_hidden, moe_ffs, 0.0F, arena_.d_moe_shexp_y + off,
                  arena_.stream) != GUFO_FP4_OK) {
            throw std::runtime_error("qwen35moe prefill shared expert failed");
          }
          kat_prof.shared_ms += kat_prof.Take();
        }
        // SQ10 SPEC3 (S3-2): stream-ordered per-set check. The bare aggregate
        // read could precede these kernels and miss a bad-router-id error
        // (the R6 stale class); it also cleared every other session's word.
        // A refusal fails loudly, as on the decode sites.
        const int slot_set = gufo_fp4_slot_stream_attach_at(
            arena_.stream, __func__, static_cast<long>(moe_hidden));
        if (slot_set < 0) {
          throw std::runtime_error("qwen35moe prefill slot set refused");
        }
        if (gufo_fp4_expert_ffn_m_check_set_on_stream(
                slot_set, arena_.stream) != GUFO_FP4_OK) {
          throw std::runtime_error("qwen35moe prefill expert device status");
        }
      }
      kat_prof.Arm();
      if (LaunchQwenMoeCombine(arena_.d_moe_y, arena_.d_moe_shexp_y, moe_hidden,
                               moe_tokens, arena_.d_ffn_out,
                               arena_.stream) != 0) {
        throw std::runtime_error("qwen35moe prefill combine failed");
      }
      LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_ffn_out,
                               arena_.d_hidden, batch_size, hidden_size,
                               arena_.stream);
      kat_prof.combine_ms += kat_prof.Take();
    } else {
      if (half_prefill) {
        LaunchBatchedRMSNormFp16(arena_.d_hidden, nullptr,
                                 static_cast<const float*>(layer.ffn_norm.data),
                                 nullptr, arena_.d_scratch_bf16, batch_size,
                                 hidden_size, eps, arena_.stream);
      } else if (ffn_feeds_q8_only) {
        // The post-attention residual add folds into the norm: one pass reads
        // the hidden state and the attention output, writes the updated hidden
        // state for the next residual link, and emits the Q8_1 activation.
        LaunchBatchedFusedRMSNormQuantizeQ8_1(
            arena_.d_hidden, arena_.d_attn_out,
            static_cast<const float*>(layer.ffn_norm.data), arena_.d_hidden,
            arena_.d_scratch_q8_act, batch_size, hidden_size, eps,
            arena_.stream);
      } else {
        LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_attn_out,
                                 arena_.d_hidden, batch_size, hidden_size,
                                 arena_.stream);

        // FFN RMSNorm (generates BF16 into d_scratch_bf16 directly)
        LaunchBatchedRMSNorm(arena_.d_hidden,
                             static_cast<const float*>(layer.ffn_norm.data),
                             arena_.d_normed, arena_.d_scratch_bf16, batch_size,
                             hidden_size, eps, arena_.stream);
      }

      // Which buffer holds the Q8_1 form of the SwiGLU output ffn_down
      // consumes. The fused epilogue below moves it out of the shared
      // activation scratch.
      const void* ffn_down_q8_act = arena_.d_scratch_q8_act;
      const void* ffn_down_input = arena_.d_scratch_bf16;

      {
        if (!half_prefill && !ffn_feeds_q8_only) {
          LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                       arena_.d_scratch_q8_act, batch_size,
                                       hidden_size, arena_.stream);
        }
        // opt-c192-swiglu-epilogue: when ffn_down also reads the Q8_1
        // activation, the whole gate -> up -> SwiGLU -> quantize chain
        // collapses into the two projections, because the up projection can
        // apply SwiGLU against the already-written gate and emit the quantized
        // activation from its own accumulator. The FP32 up intermediate is then
        // dead, so its allocation is where the Q8_1 result goes -- reusing it
        // keeps resident memory flat and avoids writing the [batch, k]
        // activation the same kernel is reading.
        const bool fused_dual_swiglu =
            reads_q8_act(layer.ffn_gate) && reads_q8_act(layer.ffn_up) &&
            layer.ffn_gate.type == layer.ffn_up.type &&
            reads_q8_act(layer.ffn_down) &&
            IsFusedSwiGluGemmEpilogueSupported(
                layer.ffn_gate.type, batch_size, intermediate_size,
                batch_size * intermediate_size * sizeof(float));
        if (half_prefill) {
          if ((IsQwenFp4(layer.ffn_gate.type) ||
               IsQwenFp4(layer.ffn_up.type)) &&
              QwenFp4PortedOn()) {
            // SQ1: the dual-FP16 path cannot consume FP4 weights: run both
            // projections through the ported kernels (F32 out), then SwiGLU
            // (F32) and narrow to F16. d_scratch_bf16's norm input is dead
            // past the two GEMMs (same reuse the non-half path does), and
            // d_ffn_up's F32 content is dead past SwiGLU, so the F16 result
            // lands where the paired path puts it. (A fused SwiGLU-to-F16
            // would save the F32 round trip; follow-up, not this lane.)
            // SQ1: ported only when GUFO_QWEN_FP4_PORTED=1; otherwise the
            // else arm runs (dual declines FP4, singles + SwiGLU epilogue).
            gemm_weight(layer.ffn_gate, arena_.d_scratch_bf16, nullptr,
                        arena_.d_ffn_gate, intermediate_size, hidden_size);
            gemm_weight(layer.ffn_up, arena_.d_scratch_bf16, nullptr,
                        arena_.d_ffn_up, intermediate_size, hidden_size);
            LaunchBatchedSwiGLUActivation(
                arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_ffn_act,
                arena_.d_scratch_bf16, batch_size * intermediate_size,
                arena_.stream);
            LaunchFloatToFp16(arena_.d_ffn_act, arena_.d_ffn_up,
                              batch_size * intermediate_size, arena_.stream);
            ffn_down_input = arena_.d_ffn_up;
          } else {
            const bool paired = TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
                layer.ffn_gate.type, layer.ffn_up.type, layer.ffn_gate.data,
                layer.ffn_up.data, arena_.d_scratch_bf16, arena_.d_ffn_up,
                batch_size, intermediate_size, hidden_size, arena_.stream);
            if (!paired) {
              gemm_weight(layer.ffn_gate, arena_.d_scratch_bf16,
                          arena_.d_normed, arena_.d_ffn_gate, intermediate_size,
                          hidden_size);
              LaunchBatchedQuantGEMMSwiGLUFp16(
                  layer.ffn_up.type, layer.ffn_up.data, arena_.d_scratch_bf16,
                  arena_.d_ffn_gate, arena_.d_ffn_up, batch_size,
                  intermediate_size, hidden_size, arena_.stream);
            }
            ffn_down_input = arena_.d_ffn_up;
          }
        } else if (fused_dual_swiglu) {
          LaunchBatchedDualQuantGEMMSwiGLUQuantizeQ8_1(
              layer.ffn_gate.type, layer.ffn_gate.data, layer.ffn_up.data,
              arena_.d_scratch_q8_act, arena_.d_ffn_gate, arena_.d_ffn_up,
              batch_size, intermediate_size, hidden_size, arena_.stream);
          ffn_down_q8_act = arena_.d_ffn_up;
        } else if (reads_q8_act(layer.ffn_gate) && reads_q8_act(layer.ffn_up) &&
                   layer.ffn_gate.type == layer.ffn_up.type) {
          LaunchBatchedDualQuantGEMMPreQuantized(
              layer.ffn_gate.type, layer.ffn_gate.data, layer.ffn_up.data,
              arena_.d_scratch_q8_act, arena_.d_ffn_gate, arena_.d_ffn_up,
              batch_size, intermediate_size, hidden_size, arena_.stream);
        } else {
          gemm_weight(layer.ffn_gate, arena_.d_scratch_bf16, arena_.d_normed,
                      arena_.d_ffn_gate, intermediate_size, hidden_size,
                      arena_.d_scratch_q8_act);

          gemm_weight(layer.ffn_up, arena_.d_scratch_bf16, arena_.d_normed,
                      arena_.d_ffn_up, intermediate_size, hidden_size,
                      arena_.d_scratch_q8_act);
        }

        // opt-c164-swiglu-quant: when ffn_down reads Q8_0 the only consumer of
        // the activation is the quantized buffer, so SwiGLU can write it
        // directly. That drops the FP32 activation and BF16 scratch round trips
        // (about 500 MB per layer at batch 2048) and one kernel launch. Other
        // ffn_down formats still need the FP32/BF16 forms, so they keep the
        // unfused chain.
        if (half_prefill || fused_dual_swiglu) {
          // The up projection's epilogue already wrote the quantized
          // activation.
        } else if (reads_q8_act(layer.ffn_down)) {
          LaunchBatchedFusedSwiGLUQuantizeQ8_1(
              arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_scratch_q8_act,
              batch_size, intermediate_size, arena_.stream);
        } else {
          LaunchBatchedSwiGLUActivation(arena_.d_ffn_gate, arena_.d_ffn_up,
                                        arena_.d_ffn_act, arena_.d_scratch_bf16,
                                        batch_size * intermediate_size,
                                        arena_.stream);
          LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                       arena_.d_scratch_q8_act, batch_size,
                                       intermediate_size, arena_.stream);
        }
      }

      if (half_prefill) {
        if (IsQwenFp4(layer.ffn_down.type) && QwenFp4PortedOn()) {
          // SQ1: ported GEMM into the FFN-output buffer, then the residual
          // add the non-half path does (d_ffn_out is dead in half mode
          // otherwise). SQ1: F16 path takes the fused Residual epilogue in
          // the else arm (DirectGemm FP4 cases).
          gemm_weight(layer.ffn_down, ffn_down_input, nullptr, arena_.d_ffn_out,
                      hidden_size, intermediate_size);
          LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_ffn_out,
                                   arena_.d_hidden, batch_size, hidden_size,
                                   arena_.stream);
        } else {
          LaunchBatchedQuantGEMMResidualFp16(
              layer.ffn_down.type, layer.ffn_down.data, ffn_down_input,
              arena_.d_hidden, batch_size, hidden_size, intermediate_size,
              arena_.stream);
        }
      } else {
        gemm_weight(layer.ffn_down, ffn_down_input, arena_.d_ffn_act,
                    arena_.d_ffn_out, hidden_size, intermediate_size,
                    ffn_down_q8_act);
        LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_ffn_out,
                                 arena_.d_hidden, batch_size, hidden_size,
                                 arena_.stream);
      }
    }  // KAT S4: end of the dense-FFN-only else

    if (capture_prompt_hidden_) {
      if (const auto tap_index = arena_.GetTargetLayerCaptureIndex(l);
          tap_index.has_value()) {
        float* destination = h_prompt_hidden_.data() + prompt_capture_offset +
                             (*tap_index * hidden_size);
        HIP_CHECK(hipMemcpy2DAsync(
            destination, target_layer_count * hidden_size * sizeof(float),
            scratch.decode.hidden.data(), hidden_size * sizeof(float),
            hidden_size * sizeof(float), batch_size, hipMemcpyDeviceToHost,
            arena_.stream));
      }
    }
  }

  if (capture_prompt_hidden_) {
    if (target_layer_count == 0) {
      const std::size_t old_size = h_prompt_hidden_.size();
      const std::size_t chunk_elements = batch_size * hidden_size;
      h_prompt_hidden_.resize(old_size + chunk_elements);
      HIP_CHECK(hipMemcpyAsync(h_prompt_hidden_.data() + old_size,
                               scratch.decode.hidden.data(),
                               chunk_elements * sizeof(float),
                               hipMemcpyDeviceToHost, arena_.stream));
    }
    HIP_CHECK(hipStreamSynchronize(arena_.stream));
  }
  last_hidden_offset_ = (batch_size - 1) * hidden_size;

  if (!compute_logits) {
    kat_prof.Dump(batch_size, start_pos, weights_);  // KAT:
    return 0;
  }
  kat_prof.Arm();  // KAT:

  // 4. Output Norm for final token
  const float* final_hidden =
      scratch.decode.hidden.data() + ((batch_size - 1) * hidden_size);
  LaunchRMSNorm(final_hidden,
                static_cast<const float*>(weights_.output_norm.data),
                scratch.decode.normed.data(), hidden_size, eps, arena_.stream);

  // 5. LM Head Logits GEMV on final token
  LaunchGEMV(weights_.output.data, weights_.output.type,
             scratch.decode.normed.data(), scratch.decode.logits.data(),
             vocab_size, hidden_size, arena_.stream);

  // 6. GPU Argmax. The sampled-token view enters its documented alias epoch
  // only after all SSM layer uses of alpha have completed.
  auto* d_out_token = scratch.decode.sampled_token.data();
  LaunchGPUArgmax(scratch.decode.logits.data(), d_out_token, vocab_size,
                  arena_.stream);

  std::uint32_t next_token_id = 0;
  HIP_CHECK(hipMemcpyAsync(&next_token_id, d_out_token, sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  kat_prof.head_ms += kat_prof.Take();             // KAT:
  kat_prof.Dump(batch_size, start_pos, weights_);  // KAT:

  return CheckedSampleToken(next_token_id, config.vocab_size);
}

std::vector<tokenization::TokenId> QwenGpuExecutor::ForwardVerificationChunk(
    std::span<const tokenization::TokenId> candidate_tokens,
    std::uint32_t start_pos, bool capture_logits) {
  // NBFAST T1: llama verifies through its own prefill path (adjacent-pair
  // RoPE, fp32 KV); the decode-equivalent stack below is Qwen-shaped.
  if (weights_.config.llama) {
    return ForwardLlamaVerificationChunk(candidate_tokens, start_pos,
                                         capture_logits);
  }
  if (!weights_.config.lfm2) {
    return ForwardDecodeEquivalentVerificationChunk(candidate_tokens, start_pos,
                                                    capture_logits);
  }
  // LFM2: the batched verifier kernels assume the Qwen DeltaNet layer
  // stack. Verify with the exact single-token decode arithmetic instead;
  // acceptance still decides which of these rows stay committed, and the
  // conv-state rollback in SaveState/RestoreState undoes the rest.
  CheckReset();
  h_verification_hidden_.clear();
  h_verification_logits_.clear();
  last_verification_rows_ = candidate_tokens.size();
  if (candidate_tokens.empty()) {
    return {};
  }
  EnsureVerificationLogits(candidate_tokens.size());
  const std::size_t vocab_size = weights_.config.vocab_size;
  std::vector<tokenization::TokenId> predictions;
  predictions.reserve(candidate_tokens.size());
  for (std::size_t row = 0; row < candidate_tokens.size(); ++row) {
    predictions.push_back(
        ForwardToken(candidate_tokens[row],
                     start_pos + static_cast<std::uint32_t>(row), true));
    if (capture_prompt_hidden_) {
      const auto hidden = CopyLastHidden();
      h_verification_hidden_.insert(h_verification_hidden_.end(),
                                    hidden.begin(), hidden.end());
    }
    auto scratch = arena_.GetScratchView();
    HIP_CHECK(hipMemcpyAsync(d_verification_logits_ + (row * vocab_size),
                             scratch.decode.logits.data(),
                             vocab_size * sizeof(float),
                             hipMemcpyDeviceToDevice, arena_.stream));
    if (capture_logits) {
      const auto logits = CopyLastLogits();
      h_verification_logits_.insert(h_verification_logits_.end(),
                                    logits.begin(), logits.end());
    }
  }
  last_hidden_offset_ = 0;
  return predictions;
}

void QwenGpuExecutor::CommitVerificationChunk(
    std::span<const tokenization::TokenId> committed_tokens,
    std::uint32_t start_pos) {
  if (committed_tokens.empty()) {
    return;
  }
  // NBFAST T1: llama has no recurrent state to rebuild, and its verification
  // forward already wrote the committed rows' KV at their final positions
  // (the same state a fully-accepted chunk keeps). Rejected rows sit beyond
  // the committed position and are overwritten by the next forward, which
  // writes KV before reading it. Re-forwarding the committed prefix through
  // single-token decode would cost one full weight pass per token and erase
  // the speculation gain, so commit is bookkeeping only here.
  if (weights_.config.llama) {
    replaying_ssm_state_ = false;
    arena_.DisableSsmReplayCapture();
    HIP_CHECK(hipStreamSynchronize(arena_.stream));
    return;
  }
  std::size_t replayed = 0;
  // LFM2 rows never enter the DeltaNet replay log; they re-run below.
  if (replaying_ssm_state_ && !weights_.config.lfm2) {
    while (replayed < committed_tokens.size() &&
           arena_.CanReplaySsmPosition(start_pos +
                                       static_cast<std::uint32_t>(replayed)))
      ++replayed;
    if (replayed != 0)
      ReplaySsmState(start_pos, static_cast<std::uint32_t>(replayed));
  }
  replaying_ssm_state_ = false;
  // SQ10 SPEC3 (S3-1): the commit replays below queue on arena_.stream with
  // no drain of their own (ForwardToken(false) only drains on MoE, via the
  // status check), and AcceptFeedback's drafter launches on the aliased set
  // next. Drain once here so the session -> drafter edge holds on every
  // architecture; a verify step already pays several syncs, so this adds no
  // new pipeline stage.
  if (replayed == committed_tokens.size()) {
    HIP_CHECK(hipStreamSynchronize(arena_.stream));
    return;
  }
  arena_.DisableSsmReplayCapture();

  const bool capture_hidden = capture_prompt_hidden_;
  capture_prompt_hidden_ = false;
  try {
    // Unrecorded rows must use the original decode arithmetic. Prefill
    // projections can round differently and leave a different recurrent state.
    for (std::size_t row = replayed; row < committed_tokens.size(); ++row)
      (void)ForwardToken(committed_tokens[row],
                         start_pos + static_cast<std::uint32_t>(row), false);
    HIP_CHECK(hipStreamSynchronize(arena_.stream));
  } catch (...) {
    capture_prompt_hidden_ = capture_hidden;
    throw;
  }
  capture_prompt_hidden_ = capture_hidden;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
