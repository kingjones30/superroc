#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/core/env.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/flint/hip/flint_attention.hpp"  // Flint
#include "src/models/glimmer/hip/glimmer_decode.hpp"  // LFM2 (fp32 KV + attention)
#include "src/models/lfm2/hip/lfm2_attention.hpp"
#include "src/models/lfm2/hip/lfm2_kernels.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/detail/decode_step.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/fp4_launch.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/moe/qwen_moe_ffn.hpp"    // KAT-MTP:
#include "src/models/qwen/moe/qwen_moe_route.hpp"  // KAT-MTP:
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace superroc::hip {
namespace {

constexpr std::size_t kMaxDecodeBatch = 8;
static_assert(kQwenMaxDraftTokens + 1 == kMaxDecodeBatch,
              "a verification item is the anchor token plus the drafts");

// GPU spans inside one verification forward.
struct VerifyProf {
  bool on{false};
  hipEvent_t a{};
  hipEvent_t b{};
  double proj_ms{0};
  double attn_ms{0};
  double ssm_ms{0};
  double ffn_ms{0};
  double head_ms{0};
  double sample_ms{0};
  std::size_t proj_b{0};
  std::size_t attn_b{0};
  std::size_t ssm_b{0};
  std::size_t ffn_b{0};
  std::size_t head_b{0};

  void Begin(hipStream_t stream) const {
    if (on) {
      (void)hipEventRecord(a, stream);
    }
  }

  void End(hipStream_t stream, double& slot) const {
    if (!on) {
      return;
    }
    (void)hipEventRecord(b, stream);
    (void)hipEventSynchronize(b);
    float ms = 0.0F;
    (void)hipEventElapsedTime(&ms, a, b);
    slot += static_cast<double>(ms);
  }
};

[[nodiscard]] constexpr bool SupportsExactSharedProjection(
    core::GgmlType type) noexcept {
  // These formats preserve scalar FP32 arithmetic while sharing weight loads.
  return type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ8_K ||
         detail::IsNativeWmmaQuant(type);
}

[[nodiscard]] std::size_t VerifyWeightBytes(core::GgmlType type,
                                            std::size_t elements) noexcept {
  if (elements == 0) {
    return 0;
  }
  switch (type) {
    case core::GgmlType::kF32:
      return elements * sizeof(float);
    case core::GgmlType::kBF16:
    case core::GgmlType::kF16:
      return elements * sizeof(std::uint16_t);
    case core::GgmlType::kQ4_0:
    case core::GgmlType::kQ4_0_ROCMFP4:
      return (elements / 32) * 18;
    case core::GgmlType::kQ4_0_ROCMFP4_FAST:
      return (elements / 32) * 17;
    case core::GgmlType::kQ8_0:
      return (elements / 32) * 34;
    default:
      return 0;
  }
}

[[nodiscard]] std::size_t VerifyTrafficBytes(
    const models::QwenTensorRef& tensor, std::size_t batch) noexcept {
  const std::size_t once = VerifyWeightBytes(tensor.type, tensor.num_elements);
  if (once == 0 || batch == 0) {
    return once;
  }
  if (SupportsExactSharedProjection(tensor.type) ||
      tensor.type == core::GgmlType::kBF16) {
    return once;
  }
  // The small-m FP4 path reads each weight tile once.
  if (IsRocmFp4(tensor.type) && batch >= 2 && batch <= 8) {
    return once;
  }
  return once * batch;
}

// Chunked-vs-per-row switch for the >8-row integer path.
// Read per call (only multi-row verify with batch > 8 reaches it): the test
// toggles it mid-process to compare both forms bit for bit. One
// stderr line per process the first time it is seen on (the read stays per
// call; only the announcement is once).
[[nodiscard]] bool VerifyNoChunk() {
  const char* env = ::superroc::core::GetEnv("SUPERROC_VERIFY_NO_CHUNK");
  const bool on = env != nullptr && env[0] == '1';
  if (on) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      std::fprintf(stderr,
                   "SUPERROC_VERIFY_NO_CHUNK=1: test hook, >8-row verify runs "
                   "per-row integer GEMV (slower)\n");
    }
  }
  return on;
}

// Batch > 8 as <= 8-row verify_m calls (bit-identical to
// the one-row arithmetic), with a single tail row via the
// per-row integer GEMV (likewise identical). False when any chunk refuses;
// the caller then runs the same fallback as for any refused integer launch.
bool TryLaunchChunkedVerifyM(core::GgmlType type, const void* weights,
                             const float* input, float* output,
                             std::size_t batch_size, std::size_t output_size,
                             std::size_t input_size, hipStream_t stream) {
  std::size_t row = 0;
  while (row < batch_size) {
    const std::size_t left = batch_size - row;
    if (left < 2) {
      return TryLaunchRocmFp4Gemv(type, weights, input + row * input_size,
                                  output + row * output_size, output_size,
                                  input_size, stream);
    }
    const auto m = static_cast<int>(std::min<std::size_t>(8, left));
    if (superroc_fp4_gemv_verify_m(
            RocmFp4Qtype(type), weights, input + row * input_size,
            output + row * output_size, static_cast<int>(input_size),
            static_cast<int>(output_size), m, stream) != SUPERROC_FP4_OK) {
      return false;
    }
    row += static_cast<std::size_t>(m);
  }
  return true;
}

// The integer attempt for one FP4 tensor. Precondition:
// IsRocmFp4(weight.type) && RocmFp4ShapeOk(output_size, input_size). True
// when the integer launches were accepted; false on any refusal, in which
// case the caller runs the path decode would use. Never throws: under
// sq10-3 a refused scratch set is a normal event that means "fall back".
bool TryLaunchVerifyInteger(const models::QwenTensorRef& weight,
                            const float* input, float* output,
                            std::size_t batch_size, std::size_t output_size,
                            std::size_t input_size, hipStream_t stream) {
  // Intercept m = 2..8 first, including the LM head (n > 8192
  // takes the 2048-wide grid stride).
  if (batch_size >= 2 && batch_size <= 8) {
    const int rc = superroc_fp4_gemv_verify_m(
        RocmFp4Qtype(weight.type), weight.data, input, output,
        static_cast<int>(input_size), static_cast<int>(output_size),
        static_cast<int>(batch_size), stream);
    if (rc == SUPERROC_FP4_OK) {
      return true;
    }
    static int nwarn = 0;
    if (nwarn++ < 6) {
      std::fprintf(stderr,
                   "VERIFY_M_FALLBACK rc=%d n=%zu k=%zu warm=%zu batch=%zu\n",
                   rc, output_size, input_size, RocmFp4WarmedK(), batch_size);
    }
  } else if (batch_size > 8 && !VerifyNoChunk()) {
    if (TryLaunchChunkedVerifyM(weight.type, weight.data, input, output,
                                batch_size, output_size, input_size, stream)) {
      return true;
    }
    static int nwarn_chunk = 0;
    if (nwarn_chunk++ < 6) {
      std::fprintf(stderr,
                   "VERIFY_M_CHUNK_FALLBACK n=%zu k=%zu warm=%zu batch=%zu\n",
                   output_size, input_size, RocmFp4WarmedK(), batch_size);
    }
  }
  // Width 1, NO_CHUNK, or verify_m refused: the per-row integer GEMV,
  // exactly what decode's LaunchGEMV runs for this tensor.
  for (std::size_t row = 0; row < batch_size; ++row) {
    if (!TryLaunchRocmFp4Gemv(
            weight.type, weight.data, input + (row * input_size),
            output + (row * output_size), output_size, input_size, stream)) {
      return false;
    }
  }
  return true;
}

// Super Roc QVO: the members of a verify projection
// group read one input, but each superroc_fp4_gemv_verify_m call quantised it
// again (SSM group 4x, QKV 3x, dense FFN gate/up 2x per layer). The first
// member quantises; the rest reuse that Q8 copy (_xq), which is bit-identical.
// On any refusal the members rerun through TryLaunchVerifyInteger exactly as
// before (same short-circuit), so the outputs can only be today's.
// SUPERROC_VERIFY_Q8_REUSE=0 turns the reuse off for A/B runs.
struct VerifyGroupMember {
  const models::QwenTensorRef* weight;
  float* output;
  std::size_t output_size;
};

bool VerifyQ8ReuseOn() {
  static const bool on = [] {
    const char* e = ::superroc::core::GetEnv("SUPERROC_VERIFY_Q8_REUSE");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

bool TryLaunchVerifyIntegerGroup(
    std::initializer_list<VerifyGroupMember> members, const float* input,
    std::size_t batch_size, std::size_t input_size, hipStream_t stream) {
  if (VerifyQ8ReuseOn() && batch_size >= 2 && batch_size <= 8) {
    bool first = true;
    bool ok = true;
    for (const auto& member : members) {
      const auto launch =
          first ? superroc_fp4_gemv_verify_m : superroc_fp4_gemv_verify_m_xq;
      ok = launch(RocmFp4Qtype(member.weight->type), member.weight->data, input,
                  member.output, static_cast<int>(input_size),
                  static_cast<int>(member.output_size),
                  static_cast<int>(batch_size), stream) == SUPERROC_FP4_OK;
      first = false;
      if (!ok)
        break;
    }
    if (ok)
      return true;
  }
  for (const auto& member : members) {
    if (!TryLaunchVerifyInteger(*member.weight, input, member.output,
                                batch_size, member.output_size, input_size,
                                stream)) {
      return false;
    }
  }
  return true;
}

void LaunchVerifyProjection(const models::QwenTensorRef& weight,
                            const float* input, float* output,
                            std::size_t batch_size, std::size_t output_size,
                            std::size_t input_size, hipStream_t stream,
                            bool integer_path) {
  // sq16: decode's fused kernels (LaunchFusedSSMInputProjections,
  // LaunchFusedQKVProjections) take the integer Q8 path only when EVERY
  // tensor in the fused group is FP4; otherwise every tensor in the group
  // takes the fp32 dot. A per-tensor choice cannot match both, so the
  // caller passes integer_path: true when decode would run this tensor
  // through the integer GEMV (all-FP4 fused groups and every per-tensor
  // LaunchGEMV), false when decode runs the fused fp32 dot. Anything else
  // changes accepted drafts' logits and flips greedy output (v1-LEAN
  // DIFF@218, layer-0 qkv 3e-2 on ~21).
  // A launch failure is retried on the path decode would use,
  // never thrown: under sq10-3 a refused scratch set is normal and means
  // "fall back". Fused-group members do not come here with integer_path
  // true; the group wrappers below keep the group atomic instead.
  if (IsRocmFp4(weight.type)) {
    if (integer_path && RocmFp4ShapeOk(output_size, input_size)) {
      if (TryLaunchVerifyInteger(weight, input, output, batch_size, output_size,
                                 input_size, stream)) {
        return;
      }
      // Runtime refusal: decode's own per-tensor call, per row. It declines
      // the intercept on the same refusal and serves its fallback, so this
      // is bit-identical by construction to decode under the same refusal.
      for (std::size_t row = 0; row < batch_size; ++row) {
        LaunchGEMV(weight.data, weight.type, input + (row * input_size),
                   output + (row * output_size), output_size, input_size,
                   stream);
      }
      return;
    }
    if (integer_path) {
      // Static shape fallback (!ShapeOk): decode's LaunchGEMV declines the
      // intercept on the same condition and serves the same fallback.
      for (std::size_t row = 0; row < batch_size; ++row) {
        LaunchGEMV(weight.data, weight.type, input + (row * input_size),
                   output + (row * output_size), output_size, input_size,
                   stream);
      }
      return;
    }
    // Mixed fused group: decode runs the fp32 dot. Types 100/101 are
    // IsNativeWmmaQuant, so the branch below is the FP32 K-quant GEMM, which
    // preserves the fused dot's arithmetic.
  }

  if (SupportsExactSharedProjection(weight.type)) {
    LaunchBatchedQuantGEMMFp32(weight.type, weight.data, input, output,
                               batch_size, output_size, input_size, stream);
    return;
  }
  if (weight.type == core::GgmlType::kBF16) {
    LaunchExactBf16GEMMFp32SmallBatch(weight.data, input, output, batch_size,
                                      output_size, input_size, stream);
    return;
  }

  for (std::size_t row = 0; row < batch_size; ++row) {
    LaunchGEMV(weight.data, weight.type, input + (row * input_size),
               output + (row * output_size), output_size, input_size, stream);
  }
}

void LaunchFfnActivation(const models::QwenLayerWeights& layer,
                         const QwenGpuScratchView& scratch,
                         std::size_t batch_size, std::size_t intermediate_size,
                         std::size_t hidden_size, hipStream_t stream) {
  const auto& gate = layer.ffn_gate;
  const auto& up = layer.ffn_up;
  // Decode runs gate/up through the fused SwiGLU GEMV,
  // which takes the integer path only when both are FP4 with an acceptable
  // shape and ready scratch (swiglu.hip). Mirror it atomically: both
  // integer, or decode's literal fused call per row (which declines exactly
  // as decode would). A partial integer pair would diverge from decode's
  // all-fp32 fused dot.
  if (IsRocmFp4(gate.type) && IsRocmFp4(up.type)) {
    if (RocmFp4ShapeOk(intermediate_size, hidden_size) &&
        RocmFp4SwigluScratchReady(intermediate_size, stream) &&
        TryLaunchVerifyIntegerGroup(
            {{&gate, scratch.ffn.gate.data(), intermediate_size},
             {&up, scratch.ffn.up.data(), intermediate_size}},
            scratch.decode.normed.data(), batch_size, hidden_size, stream)) {
      LaunchBatchedSwiGLUActivation(scratch.ffn.gate.data(),
                                    scratch.ffn.up.data(),
                                    scratch.ffn.activation.data(), nullptr,
                                    batch_size * intermediate_size, stream);
      return;
    }
    for (std::size_t row = 0; row < batch_size; ++row) {
      LaunchFusedSwiGLUGEMV(
          gate.data, gate.type, up.data, up.type,
          scratch.decode.normed.data() + row * hidden_size,
          scratch.ffn.activation.data() + row * intermediate_size,
          intermediate_size, hidden_size, stream);
    }
    return;
  }
  const bool packed_format = gate.type == core::GgmlType::kQ4_K ||
                             gate.type == core::GgmlType::kQ5_K ||
                             gate.type == core::GgmlType::kQ6_K ||
                             gate.type == core::GgmlType::kIQ4_XS ||
                             gate.type == core::GgmlType::kQ8_0;
  if (batch_size >= 2 && intermediate_size == 17408 && hidden_size == 5120 &&
      packed_format && gate.type == up.type &&
      gate.num_elements == intermediate_size * hidden_size &&
      up.num_elements == gate.num_elements) {
    const std::size_t matrix_bytes = gate.EncodedSizeBytes();
    const std::size_t packed_elements = 2 * batch_size * intermediate_size;
    if (gate.available_bytes >= 2 * matrix_bytes &&
        static_cast<const std::byte*>(gate.data) + matrix_bytes == up.data) {
      if (TryLaunchPackedQuantSwiGLUFp32(
              gate.type, gate.data, scratch.decode.normed.data(),
              scratch.ffn.activation.data(), batch_size, intermediate_size,
              hidden_size, stream))
        return;
      if (scratch.decode.weight_bf16.size_bytes() >=
          packed_elements * sizeof(float)) {
        // Adjacent GGUF tensors form one matrix with twice as many output rows.
        // Exact projections do not need dequantization scratch; reuse it until
        // SwiGLU has consumed the packed gate/up rows on this stream.
        auto* const packed =
            reinterpret_cast<float*>(scratch.decode.weight_bf16.data());
        LaunchVerifyProjection(gate, scratch.decode.normed.data(), packed,
                               batch_size, 2 * intermediate_size, hidden_size,
                               stream, true);
        LaunchPackedSwiGLUActivation(packed, scratch.ffn.activation.data(),
                                     batch_size, intermediate_size, stream);
        return;
      }
    }
  }
  // Both-FP4 returned above, so at most one member is FP4 here; decode's
  // fused kernel serves fp32 unless both are FP4, so no member may take
  // integer (for non-FP4 the flag is a no-op, as before).
  LaunchVerifyProjection(gate, scratch.decode.normed.data(),
                         scratch.ffn.gate.data(), batch_size, intermediate_size,
                         hidden_size, stream, false);
  LaunchVerifyProjection(up, scratch.decode.normed.data(),
                         scratch.ffn.up.data(), batch_size, intermediate_size,
                         hidden_size, stream, false);
  LaunchBatchedSwiGLUActivation(scratch.ffn.gate.data(), scratch.ffn.up.data(),
                                scratch.ffn.activation.data(), nullptr,
                                batch_size * intermediate_size, stream);
}

// KAT-MTP: verification FFN. A qwen35moe layer keeps its dense ffn_gate/up/down
// projections empty and routes through the experts instead, so the dense SwiGLU
// above would read an empty weight and fault the GPU. This runs EXACTLY the
// decode_step.cpp per-token MoE math once per verification row (QVB batches
// the route, the shared-expert gate and the combine across rows without
// changing a bit), so row 0's logits are bit-identical to what a plain
// single-row Step would produce -- anything else flips greedy output on a
// near-tie. `normed` is the post-ffn-norm
// hidden, `out` the FFN result.
void LaunchVerifyFfn(const models::QwenLayerWeights& layer,
                     const core::ModelConfig& config, const float* normed,
                     float* out, float* moe_logits, int* moe_ids,
                     float* moe_weights, float* moe_sig, int* moe_zero_ids,
                     float* moe_y, float* moe_shexp_y, std::size_t batch_size,
                     std::size_t hidden_size, hipStream_t stream,
                     float* moe_gate_e = nullptr, float* moe_slot_y = nullptr,
                     float* moe_shexp_h = nullptr) {
  const int moe_expert = static_cast<int>(config.expert_count);
  const int moe_used = static_cast<int>(config.expert_used_count);
  const int moe_ff = static_cast<int>(config.expert_feed_forward_length);
  const int moe_ffs =
      static_cast<int>(config.expert_shared_feed_forward_length);
  const int moe_hidden = static_cast<int>(hidden_size);
  const int gate_type = static_cast<int>(layer.ffn_gate_exps.type);
  const int shexp_type = static_cast<int>(layer.ffn_gate_shexp.type);
  // The route and the shared-expert gate for up to 4 rows per launch
  // pair (the router read once per chunk), the experts per row as before,
  // one combine for all rows. Every row's outputs are bit-identical to the
  // former per-row sequence, so row 0 still matches a plain Step.
  for (std::size_t r0 = 0; r0 < batch_size; r0 += 4) {
    const int rows =
        static_cast<int>(std::min<std::size_t>(4, batch_size - r0));
    if (LaunchQwenMoeRouteRows(
            normed + r0 * hidden_size,
            static_cast<const float*>(layer.ffn_gate_inp.data), moe_hidden,
            moe_expert, rows, moe_used,
            moe_logits + r0 * static_cast<std::size_t>(moe_expert),
            moe_ids + r0 * static_cast<std::size_t>(moe_used),
            moe_weights + r0 * static_cast<std::size_t>(moe_used),
            static_cast<const float*>(layer.ffn_gate_inp_shexp.data),
            moe_sig + r0, stream,
            static_cast<int>(config.ExpertsInFile())) != 0) {
      throw std::runtime_error("qwen35moe verify route launch failed");
    }
  }
  if (layer.moe_kquant) {
    // K-quant verify path. Gated in <=8-row chunks (the entry
    // point's row limit), down in one call, shexp per row via dense Q8_0
    // GEMV, weighted combine in LaunchQwenMoeCombineSlots.
    if (moe_gate_e == nullptr || moe_slot_y == nullptr ||
        moe_shexp_h == nullptr) {
      throw std::runtime_error("qwen35moe verify K-quant scratch missing");
    }
    if (batch_size > 512) {
      throw std::runtime_error("qwen35moe verify batch exceeds K-quant R");
    }
    const int b = static_cast<int>(batch_size);
    for (std::size_t r0 = 0; r0 < batch_size; r0 += 8) {
      const int r = static_cast<int>(std::min<std::size_t>(8, batch_size - r0));
      if (qfn_mmq_moe_gated_vec(
              gate_type, layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
              normed + r0 * hidden_size,
              moe_ids + r0 * static_cast<std::size_t>(moe_used),
              moe_gate_e + r0 * static_cast<std::size_t>(moe_used) * moe_ff,
              moe_ff, moe_hidden, r, moe_expert, moe_used, stream) != 0) {
        throw std::runtime_error("qwen35moe verify experts failed");
      }
    }
    if (qfn_mmq_moe_vec(static_cast<int>(layer.ffn_down_exps.type),
                        layer.ffn_down_exps.data, moe_gate_e, moe_ids,
                        moe_slot_y, moe_hidden, moe_ff, b * moe_used,
                        moe_expert, 1, stream) != 0) {
      throw std::runtime_error("qwen35moe verify experts failed");
    }
    for (std::size_t row = 0; row < batch_size; ++row) {
      LaunchFusedSwiGLUGEMV(
          layer.ffn_gate_shexp.data, layer.ffn_gate_shexp.type,
          layer.ffn_up_shexp.data, layer.ffn_up_shexp.type,
          normed + row * hidden_size,
          moe_shexp_h + row * static_cast<std::size_t>(moe_ffs),
          static_cast<std::size_t>(moe_ffs),
          static_cast<std::size_t>(moe_hidden), stream);
      LaunchGEMV(layer.ffn_down_shexp.data, layer.ffn_down_shexp.type,
                 moe_shexp_h + row * static_cast<std::size_t>(moe_ffs),
                 moe_shexp_y + row * hidden_size,
                 static_cast<std::size_t>(moe_hidden),
                 static_cast<std::size_t>(moe_ffs), stream);
    }
    if (LaunchQwenMoeCombineSlots(moe_slot_y, moe_weights, moe_shexp_y, moe_sig,
                                  moe_used, moe_hidden, b, out, stream) != 0) {
      throw std::runtime_error("qwen35moe verify combine failed");
    }
    return;
  }
  // Routed and shared experts in one launch per stage, each weight row read
  // once; writes `out` directly, the combine included. Bit-identical to the
  // paths below.
  static const bool all_on = [] {
    const char* e = ::superroc::core::GetEnv("SUPERROC_MOE_VERIFY_ALL");
    return e == nullptr || e[0] != '0';
  }();
  if (all_on && batch_size >= 2 && batch_size <= 8 && gate_type == shexp_type &&
      superroc_fp4_expert_ffn_dev_batch_all(
          gate_type, layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
          layer.ffn_down_exps.data, layer.ffn_gate_shexp.data,
          layer.ffn_up_shexp.data, layer.ffn_down_shexp.data, normed, moe_ids,
          moe_weights, moe_sig, static_cast<int>(batch_size), moe_used,
          moe_expert, moe_hidden, moe_ff, moe_ffs, 0.0F, out,
          stream) == SUPERROC_FP4_OK) {
    return;
  }
  // moebatch: every row of this layer in one launch per stage. Bit-identical to
  // the loop below, which stays as the fallback (a refused batch has queued
  // nothing that is read).
  static const bool batch_on = [] {
    const char* e = ::superroc::core::GetEnv("SUPERROC_MOE_VERIFY_BATCH");
    return e == nullptr || e[0] != '0';
  }();
  bool experts_done = false;
  if (batch_on && batch_size > 1) {
    experts_done =
        superroc_fp4_expert_ffn_dev_batch(
            gate_type, layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
            layer.ffn_down_exps.data, normed, moe_ids, moe_weights,
            static_cast<int>(batch_size), moe_used, moe_expert, moe_hidden,
            moe_ff, 0.0F, moe_y, stream, 1) == SUPERROC_FP4_OK &&
        superroc_fp4_expert_ffn_dev_batch(
            shexp_type, layer.ffn_gate_shexp.data, layer.ffn_up_shexp.data,
            layer.ffn_down_shexp.data, normed, moe_zero_ids, moe_sig,
            static_cast<int>(batch_size), 1, 1, moe_hidden, moe_ffs, 0.0F,
            moe_shexp_y, stream, 0) == SUPERROC_FP4_OK;
  }
  for (std::size_t row = 0; !experts_done && row < batch_size; ++row) {
    const float* const x = normed + row * hidden_size;
    int* const ids = moe_ids + row * static_cast<std::size_t>(moe_used);
    float* const wts = moe_weights + row * static_cast<std::size_t>(moe_used);
    float* const sig = moe_sig + row;
    int* const zero = moe_zero_ids + row;
    float* const y = moe_y + row * hidden_size;
    float* const sy = moe_shexp_y + row * hidden_size;
    if (superroc_fp4_expert_ffn_dev(
            gate_type, layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
            layer.ffn_down_exps.data, x, ids, wts, moe_used, moe_expert,
            moe_hidden, moe_ff, 0.0F, y, stream) != SUPERROC_FP4_OK) {
      throw std::runtime_error("qwen35moe verify experts failed");
    }
    // Reuses the routed call's Q8 copy of this row's x (same stream).
    if (superroc_fp4_expert_ffn_dev_xq(
            shexp_type, layer.ffn_gate_shexp.data, layer.ffn_up_shexp.data,
            layer.ffn_down_shexp.data, x, zero, sig, 1, 1, moe_hidden, moe_ffs,
            0.0F, sy, stream) != SUPERROC_FP4_OK) {
      throw std::runtime_error("qwen35moe verify shared expert failed");
    }
  }
  if (LaunchQwenMoeCombine(moe_y, moe_shexp_y, moe_hidden,
                           static_cast<int>(batch_size), out, stream) != 0) {
    throw std::runtime_error("qwen35moe verify combine failed");
  }
}

struct SsmControls {
  const float* alpha;
  const float* beta;
  std::size_t row_stride;
};

SsmControls LaunchSsmControls(const models::QwenLayerWeights& layer,
                              const QwenGpuScratchView& scratch,
                              std::size_t batch_size,
                              std::size_t time_step_rank,
                              std::size_t hidden_size, hipStream_t stream,
                              bool integer_path) {
  const auto& alpha = layer.ssm_alpha;
  const auto& beta = layer.ssm_beta;
  if (time_step_rank == 48 && hidden_size == 5120 &&
      alpha.type == core::GgmlType::kQ8_0 && beta.type == alpha.type &&
      alpha.num_elements == time_step_rank * hidden_size &&
      beta.num_elements == alpha.num_elements) {
    const std::size_t matrix_bytes = alpha.EncodedSizeBytes();
    const std::size_t row_stride = 2 * time_step_rank;
    if (alpha.available_bytes >= 2 * matrix_bytes &&
        static_cast<const std::byte*>(alpha.data) + matrix_bytes == beta.data &&
        scratch.decode.weight_bf16.size_bytes() >=
            batch_size * row_stride * sizeof(float)) {
      // Adjacent control matrices share one launch. Recurrence already accepts
      // a row stride, so it can consume the interleaved output directly.
      auto* const packed =
          reinterpret_cast<float*>(scratch.decode.weight_bf16.data());
      LaunchVerifyProjection(alpha, scratch.decode.normed.data(), packed,
                             batch_size, row_stride, hidden_size, stream,
                             integer_path);
      return {packed, packed + time_step_rank, row_stride};
    }
  }
  LaunchVerifyProjection(alpha, scratch.decode.normed.data(),
                         scratch.ssm.alpha.data(), batch_size, time_step_rank,
                         hidden_size, stream, integer_path);
  LaunchVerifyProjection(beta, scratch.decode.normed.data(),
                         scratch.ssm.beta.data(), batch_size, time_step_rank,
                         hidden_size, stream, integer_path);
  return {scratch.ssm.alpha.data(), scratch.ssm.beta.data(), time_step_rank};
}

// sq16: decode's integer-path conditions, mirrored from
// LaunchFusedSSMInputProjections / LaunchFusedQKVProjections (the type and
// shape checks; slot-launch success is decode's own run-time decision).
[[nodiscard]] bool DecodeSsmIntegerPath(const models::QwenLayerWeights& layer,
                                        std::size_t ssm_qkv_size,
                                        std::size_t ssm_inner_size,
                                        std::size_t time_step_rank,
                                        std::size_t hidden_size) {
  return IsRocmFp4(layer.attn_qkv.type) && IsRocmFp4(layer.attn_gate.type) &&
         IsRocmFp4(layer.ssm_alpha.type) && IsRocmFp4(layer.ssm_beta.type) &&
         RocmFp4ShapeOk(ssm_qkv_size, hidden_size) &&
         RocmFp4ShapeOk(ssm_inner_size, hidden_size) &&
         RocmFp4ShapeOk(time_step_rank, hidden_size);
}

[[nodiscard]] bool DecodeQkvIntegerPath(const models::QwenLayerWeights& layer,
                                        std::size_t q_projection_size,
                                        std::size_t kv_size,
                                        std::size_t hidden_size) {
  return IsRocmFp4(layer.attn_q.type) && IsRocmFp4(layer.attn_k.type) &&
         IsRocmFp4(layer.attn_v.type) &&
         RocmFp4ShapeOk(q_projection_size, hidden_size) &&
         RocmFp4ShapeOk(kv_size, hidden_size);
}

// Decode's per-tensor integer condition (LaunchGEMV in
// gemv.hip, reached via QuantGemm for attn_output and directly for ssm_out,
// ffn_down and the head): FP4 with an acceptable shape. The launch itself
// may still refuse; LaunchVerifyProjection retries that on decode's path.
[[nodiscard]] bool DecodeSingleIntegerPath(const models::QwenTensorRef& weight,
                                           std::size_t n, std::size_t k) {
  return IsRocmFp4(weight.type) && RocmFp4ShapeOk(n, k);
}

// Group-atomic integer attempt for a decode-fused QKV group: every member
// integer, or decode's literal fused call per row (which falls back exactly
// as decode would). A partial integer group would diverge from decode's
// all-fp32 fused dot, so the members never fall back one by one.
void LaunchVerifyQkvGroup(const models::QwenLayerWeights& layer,
                          const float* input, float* q_out, float* k_out,
                          float* v_out, std::size_t batch_size,
                          std::size_t q_projection_size, std::size_t kv_size,
                          std::size_t hidden_size, hipStream_t stream,
                          bool integer_path) {
  if (integer_path &&
      TryLaunchVerifyIntegerGroup({{&layer.attn_q, q_out, q_projection_size},
                                   {&layer.attn_k, k_out, kv_size},
                                   {&layer.attn_v, v_out, kv_size}},
                                  input, batch_size, hidden_size, stream)) {
    return;
  }
  if (!integer_path) {
    LaunchVerifyProjection(layer.attn_q, input, q_out, batch_size,
                           q_projection_size, hidden_size, stream, false);
    LaunchVerifyProjection(layer.attn_k, input, k_out, batch_size, kv_size,
                           hidden_size, stream, false);
    LaunchVerifyProjection(layer.attn_v, input, v_out, batch_size, kv_size,
                           hidden_size, stream, false);
    return;
  }
  for (std::size_t row = 0; row < batch_size; ++row) {
    LaunchFusedQKVProjections(
        layer.attn_q.data, layer.attn_q.type, layer.attn_k.data,
        layer.attn_k.type, layer.attn_v.data, layer.attn_v.type,
        input + row * hidden_size, q_out + row * q_projection_size,
        k_out + row * kv_size, v_out + row * kv_size, q_projection_size,
        kv_size, hidden_size, stream);
  }
}

// Same atomicity for a decode-fused SSM group. The packed control layout
// needs Q8_0 alpha/beta, which contradicts an all-FP4 integer group, so it
// only exists on the shared (!integer_path) side, inside LaunchSsmControls.
SsmControls LaunchVerifySsmGroup(
    const models::QwenLayerWeights& layer, const QwenGpuScratchView& scratch,
    std::size_t batch_size, std::size_t ssm_qkv_size,
    std::size_t ssm_inner_size, std::size_t time_step_rank,
    std::size_t hidden_size, hipStream_t stream, bool integer_path) {
  const float* input = scratch.decode.normed.data();
  if (integer_path &&
      TryLaunchVerifyIntegerGroup(
          {{&layer.attn_qkv, scratch.ssm.qkv.data(), ssm_qkv_size},
           {&layer.attn_gate, scratch.ssm.gate.data(), ssm_inner_size},
           {&layer.ssm_alpha, scratch.ssm.alpha.data(), time_step_rank},
           {&layer.ssm_beta, scratch.ssm.beta.data(), time_step_rank}},
          input, batch_size, hidden_size, stream)) {
    return {scratch.ssm.alpha.data(), scratch.ssm.beta.data(), time_step_rank};
  }
  if (!integer_path) {
    LaunchVerifyProjection(layer.attn_qkv, input, scratch.ssm.qkv.data(),
                           batch_size, ssm_qkv_size, hidden_size, stream,
                           false);
    LaunchVerifyProjection(layer.attn_gate, input, scratch.ssm.gate.data(),
                           batch_size, ssm_inner_size, hidden_size, stream,
                           false);
    return LaunchSsmControls(layer, scratch, batch_size, time_step_rank,
                             hidden_size, stream, false);
  }
  for (std::size_t row = 0; row < batch_size; ++row) {
    LaunchFusedSSMInputProjections(
        layer.attn_qkv.data, layer.attn_qkv.type, layer.attn_gate.data,
        layer.attn_gate.type, layer.ssm_alpha.data, layer.ssm_alpha.type,
        layer.ssm_beta.data, layer.ssm_beta.type, input + row * hidden_size,
        scratch.ssm.qkv.data() + row * ssm_qkv_size,
        scratch.ssm.gate.data() + row * ssm_inner_size,
        scratch.ssm.alpha.data() + row * time_step_rank,
        scratch.ssm.beta.data() + row * time_step_rank, hidden_size,
        ssm_qkv_size, ssm_inner_size, time_step_rank, stream);
  }
  return {scratch.ssm.alpha.data(), scratch.ssm.beta.data(), time_step_rank};
}

}  // namespace

std::vector<tokenization::TokenId> QwenGpuExecutor::ForwardTokenBatch(
    std::span<const QwenGpuBatchItem> items) {
  for (const auto& item : items)
    if (item.executor)
      item.executor->CheckReset();
  if (items.size() < 2 || items.size() > kMaxDecodeBatch) {
    throw std::invalid_argument(
        "Qwen GPU decode batch requires two to eight sessions");
  }

  QwenGpuExecutor* coordinator = items.front().executor;
  if (coordinator == nullptr) {
    throw std::invalid_argument(
        "Qwen GPU decode batch contains a null session");
  }
  const auto* shared_model = coordinator->model_.get();
  const std::uint64_t shared_policy = coordinator->policy_.Fingerprint();
  for (std::size_t index = 0; index < items.size(); ++index) {
    QwenGpuExecutor* executor = items[index].executor;
    if (executor == nullptr || executor->model_.get() != shared_model ||
        executor->policy_.Fingerprint() != shared_policy) {
      throw std::invalid_argument(
          "Qwen GPU decode batch sessions are not execution-compatible");
    }
    if (items[index].position >= executor->arena_.GetMaxContext()) {
      throw std::length_error(
          "token position exceeds a batched Qwen session context");
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (items[previous].executor == executor) {
        throw std::invalid_argument(
            "Qwen GPU decode batch contains a duplicate session");
      }
    }
  }
  if (items.size() > coordinator->arena_.GetMaxBatch()) {
    throw std::length_error(
        "Qwen GPU decode width exceeds the coordinator arena capacity");
  }

  bool capture_replay = false;
  for (const auto& item : items) {
    auto& executor = *item.executor;
    if (executor.replaying_ssm_state_) {
      executor.replaying_ssm_state_ = false;
      executor.arena_.DisableSsmReplayCapture();
    }
    // Replay and capture-flag updates may still be queued on a session's
    // stream. The coordinator takes ownership only after they finish.
    if (&executor != coordinator) {
      HIP_CHECK(hipStreamSynchronize(executor.arena_.stream));
    }
    capture_replay |= executor.arena_.IsSsmReplayCaptureActive();
    executor.arena_.MarkSsmReplayPosition(item.position);
    executor.last_hidden_offset_ = 0;
  }

  const std::size_t batch_size = items.size();
  const auto& weights = coordinator->weights_;
  const auto& config = weights.config;
  auto& arena = coordinator->arena_;
  const auto scratch = arena.GetScratchView(batch_size);
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;

  std::array<std::uint32_t, kMaxDecodeBatch> host_tokens{};
  std::array<std::uint32_t, kMaxDecodeBatch> host_positions{};
  std::array<SsmSequenceState, kMaxDecodeBatch> ssm_sequences{};
  for (std::size_t row = 0; row < batch_size; ++row) {
    host_tokens[row] = items[row].token_id;
    host_positions[row] = items[row].position;
    auto& state_arena = items[row].executor->arena_;
    auto replay = state_arena.GetSsmReplayCapture();
    replay.position = scratch.decode.prompt_tokens.data() + row;
    ssm_sequences[row] = {state_arena.BeginRecurrentWrite(), replay,
                          static_cast<std::uint32_t>(row), 1};
  }
  HIP_CHECK(hipMemcpyAsync(
      scratch.decode.prompt_tokens.data(), host_tokens.data(),
      batch_size * sizeof(std::uint32_t), hipMemcpyHostToDevice, arena.stream));
  LaunchBatchedEmbeddingLookup(weights.token_embd.data, weights.token_embd.type,
                               scratch.decode.prompt_tokens.data(),
                               scratch.decode.hidden.data(), batch_size,
                               hidden_size, arena.stream);
  for (std::size_t row = 0; row < items.size(); ++row) {
    items[row].executor->vision_input_.Inject(
        scratch.decode.hidden.data() + row * hidden_size, items[row].position,
        1, hidden_size, 1, arena.stream);
  }
  if (capture_replay) {
    // Embedding has consumed the token IDs. Reuse the buffer for the actual
    // per-session positions until every recurrent layer has captured its row.
    HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                             host_positions.data(),
                             batch_size * sizeof(std::uint32_t),
                             hipMemcpyHostToDevice, arena.stream));
  }

  EmitDecodeRouteTelemetry(weights, coordinator->policy_);
  for (std::uint32_t layer_index = 0; layer_index < config.num_layers;
       ++layer_index) {
    const auto& layer = weights.layers[layer_index];
    LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                         static_cast<const float*>(layer.attn_norm.data),
                         scratch.decode.normed.data(), nullptr, batch_size,
                         hidden_size, 1e-6F, arena.stream);

    if (layer.is_full_attention) {
      const bool qkv_integer =
          DecodeQkvIntegerPath(layer, q_projection_size, kv_size, hidden_size);
      LaunchVerifyQkvGroup(
          layer, scratch.decode.normed.data(), scratch.ssm.qkv.data(),
          scratch.attention.k.data(), scratch.attention.v.data(), batch_size,
          q_projection_size, kv_size, hidden_size, arena.stream, qkv_integer);
      LaunchBatchedUnpackQG(scratch.ssm.qkv.data(), scratch.attention.q.data(),
                            scratch.ssm.gate.data(), batch_size,
                            config.num_attention_heads, config.head_dim,
                            arena.stream);

      if (!layer.attn_q_norm.empty()) {
        LaunchBatchedPerHeadRMSNorm(
            scratch.attention.q.data(),
            static_cast<const float*>(layer.attn_q_norm.data),
            scratch.attention.q.data(), batch_size, config.num_attention_heads,
            config.head_dim, 1e-6F, arena.stream);
      }
      if (!layer.attn_k_norm.empty()) {
        LaunchBatchedPerHeadRMSNorm(
            scratch.attention.k.data(),
            static_cast<const float*>(layer.attn_k_norm.data),
            scratch.attention.k.data(), batch_size, config.num_key_value_heads,
            config.head_dim, 1e-6F, arena.stream);
      }

      const std::uint32_t attention_layer =
          layer_index / config.full_attention_interval;
      for (std::size_t row = 0; row < batch_size; ++row) {
        auto& state_arena = items[row].executor->arena_;
        const std::size_t total_k = state_arena.GetAttentionKvPlaneElements();
        float* query = scratch.attention.q.data() + (row * attention_size);
        float* key = scratch.attention.k.data() + (row * kv_size);
        float* value = scratch.attention.v.data() + (row * kv_size);
        float* gate = scratch.ssm.gate.data() + (row * attention_size);
        float* context = scratch.ssm.out.data() + (row * attention_size);
        const std::uint32_t position = items[row].position;

        LaunchRoPE(query, key, config.num_attention_heads,
                   config.num_key_value_heads, config.head_dim,
                   config.rotary_dim, position, config.rope_theta, arena.stream,
                   items[row].executor->vision_input_.rope());
        const bool use_split_k = detail::IsSplitKDecodeAttentionSupported(
            static_cast<std::size_t>(position) + 1, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim);
        LaunchAttention(
            query, key, value, gate, state_arena.d_kv_cache,
            OffsetIfPresent(state_arena.d_kv_cache, total_k),
            state_arena.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(state_arena.d_attention_kv_f16),
                total_k),
            context, attention_layer, position, state_arena.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, arena.stream,
            use_split_k ? arena.d_split_k_attention : nullptr);
      }
      LaunchVerifyProjection(
          layer.attn_output, scratch.ssm.out.data(),
          scratch.attention.output.data(), batch_size, hidden_size,
          attention_size, arena.stream,
          DecodeSingleIntegerPath(layer.attn_output, hidden_size,
                                  attention_size));
    } else {
      const bool ssm_integer = DecodeSsmIntegerPath(
          layer, ssm_qkv_size, ssm_inner_size, time_step_rank, hidden_size);
      const auto controls = LaunchVerifySsmGroup(
          layer, scratch, batch_size, ssm_qkv_size, ssm_inner_size,
          time_step_rank, hidden_size, arena.stream, ssm_integer);

      LaunchSSMConvRecurrenceBatch(
          scratch.ssm.qkv.data(),
          static_cast<const float*>(layer.ssm_conv1d.data),
          scratch.ssm.conv_out.data(), controls.alpha, controls.beta,
          static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data),
          static_cast<const float*>(layer.ssm_norm.data),
          scratch.ssm.gate.data(), scratch.ssm.out.data(),
          std::span(ssm_sequences).first(batch_size),
          config.SsmLayerIndex(layer_index), ssm_qkv_size,
          config.ssm_group_count, config.ssm_time_step_rank,
          config.ssm_state_size, config.SsmValueSize(), controls.row_stride,
          ssm_inner_size, arena.stream, arena.GetRecurrentStateStorage());

      LaunchVerifyProjection(
          layer.ssm_out, scratch.ssm.out.data(),
          scratch.attention.output.data(), batch_size, hidden_size,
          ssm_inner_size, arena.stream,
          DecodeSingleIntegerPath(layer.ssm_out, hidden_size, ssm_inner_size));
    }

    LaunchBatchedResidualAdd(
        scratch.decode.hidden.data(), scratch.attention.output.data(),
        scratch.decode.hidden.data(), batch_size, hidden_size, arena.stream);
    LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                         static_cast<const float*>(layer.ffn_norm.data),
                         scratch.decode.normed.data(), nullptr, batch_size,
                         hidden_size, 1e-6F, arena.stream);

    if (layer.is_moe) {
      LaunchVerifyFfn(layer, config, scratch.decode.normed.data(),
                      scratch.ffn.out.data(), arena.d_moe_logits,
                      arena.d_moe_ids, arena.d_moe_weights, arena.d_moe_sig,
                      arena.d_moe_zero_ids, arena.d_moe_y, arena.d_moe_shexp_y,
                      batch_size, hidden_size, arena.stream, arena.d_moe_gate_e,
                      arena.d_moe_slot_y, arena.d_moe_shexp_h);
    } else {
      LaunchFfnActivation(layer, scratch, batch_size, intermediate_size,
                          hidden_size, arena.stream);
      LaunchVerifyProjection(
          layer.ffn_down, scratch.ffn.activation.data(), scratch.ffn.out.data(),
          batch_size, hidden_size, intermediate_size, arena.stream,
          DecodeSingleIntegerPath(layer.ffn_down, hidden_size,
                                  intermediate_size));
    }
    LaunchBatchedResidualAdd(
        scratch.decode.hidden.data(), scratch.ffn.out.data(),
        scratch.decode.hidden.data(), batch_size, hidden_size, arena.stream);

    for (std::size_t row = 0; row < batch_size; ++row) {
      auto& state_arena = items[row].executor->arena_;
      if (const auto tap = state_arena.GetTargetLayerCaptureIndex(layer_index);
          tap.has_value()) {
        HIP_CHECK(hipMemcpyAsync(
            state_arena.d_target_layer_features + (*tap * hidden_size),
            scratch.decode.hidden.data() + (row * hidden_size),
            hidden_size * sizeof(float), hipMemcpyDeviceToDevice,
            arena.stream));
      }
    }
  }

  // Name the coordinator stream's set, so concurrent sessions check
  // theirs. A refusal fails the step loudly (unreachable in serve).
  if (config.is_moe) {
    const int slot_set = superroc_fp4_slot_stream_attach_at(
        arena.stream, __func__, static_cast<long>(hidden_size));
    if (slot_set < 0) {
      throw std::runtime_error("qwen35moe verify slot set refused");
    }
    // Stream-ordered check (the bare read proved stale on the
    // box: STALE_STATUS missed 1/200).
    if (superroc_fp4_moe_dev_check_set_on_stream(slot_set, arena.stream) !=
        SUPERROC_FP4_OK) {
      throw std::runtime_error("qwen35moe verify expert device status");
    }
  }

  LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                       static_cast<const float*>(weights.output_norm.data),
                       scratch.decode.normed.data(), nullptr, batch_size,
                       hidden_size, 1e-6F, arena.stream);
  coordinator->EnsureVerificationLogits(batch_size);
  LaunchVerifyProjection(
      weights.output, scratch.decode.normed.data(),
      coordinator->d_verification_logits_, batch_size, vocab_size, hidden_size,
      arena.stream,
      DecodeSingleIntegerPath(weights.output, vocab_size, hidden_size));
  LaunchBatchedGPUArgmax(coordinator->d_verification_logits_,
                         scratch.decode.prompt_tokens.data(), batch_size,
                         vocab_size, scratch.ffn.out, arena.stream);

  std::array<std::uint32_t, kMaxDecodeBatch> host_frontiers{};
  HIP_CHECK(hipMemcpyAsync(
      host_frontiers.data(), scratch.decode.prompt_tokens.data(),
      batch_size * sizeof(std::uint32_t), hipMemcpyDeviceToHost, arena.stream));
  for (std::size_t row = 0; row < batch_size; ++row) {
    auto& state_arena = items[row].executor->arena_;
    if (&state_arena != &arena || row != 0) {
      HIP_CHECK(hipMemcpyAsync(
          state_arena.d_hidden,
          scratch.decode.hidden.data() + (row * hidden_size),
          hidden_size * sizeof(float), hipMemcpyDeviceToDevice, arena.stream));
    }
    HIP_CHECK(hipMemcpyAsync(
        state_arena.d_logits,
        coordinator->d_verification_logits_ + (row * vocab_size),
        vocab_size * sizeof(float), hipMemcpyDeviceToDevice, arena.stream));
  }
  HIP_CHECK(hipStreamSynchronize(arena.stream));

  for (std::size_t row = 0; row < batch_size; ++row)
    (void)CheckedSampleToken(host_frontiers[row], vocab_size);
  return {host_frontiers.begin(),
          host_frontiers.begin() + static_cast<std::ptrdiff_t>(batch_size)};
}

std::vector<tokenization::TokenId>
QwenGpuExecutor::ForwardDecodeEquivalentVerificationChunk(
    std::span<const tokenization::TokenId> candidate_tokens,
    std::uint32_t start_pos, bool capture_logits) {
  if (candidate_tokens.empty()) {
    h_verification_hidden_.clear();
    h_verification_logits_.clear();
    last_verification_rows_ = 0;
    return {};
  }
  const QwenGpuVerificationItem item{this, candidate_tokens, start_pos,
                                     capture_logits};
  auto predictions = ForwardVerificationBatch(std::span(&item, 1));
  return std::move(predictions.front());
}

std::vector<std::vector<tokenization::TokenId>>
QwenGpuExecutor::ForwardVerificationBatch(
    std::span<const QwenGpuVerificationItem> items) {
  for (const auto& item : items)
    if (item.executor)
      item.executor->CheckReset();
  if (items.empty() || items.size() > kMaxDecodeBatch ||
      items.front().executor == nullptr) {
    throw std::invalid_argument(
        "Qwen verification requires one to eight sessions");
  }
  auto& coordinator = *items.front().executor;
  auto& arena_ = coordinator.arena_;
  const auto& weights_ = coordinator.weights_;
  const auto& config = weights_.config;
  const bool capture_prompt_hidden_ = coordinator.capture_prompt_hidden_;
  const auto policy = coordinator.policy_.Fingerprint();
  constexpr std::size_t kMaxRows = kMaxDecodeBatch * kMaxDecodeBatch;
  std::array<std::size_t, kMaxDecodeBatch> offsets{};
  std::size_t batch_size = 0;
  for (std::size_t index = 0; index < items.size(); ++index) {
    const auto& item = items[index];
    const auto* executor = item.executor;
    if (executor == nullptr || executor->model_ != coordinator.model_ ||
        executor->policy_.Fingerprint() != policy ||
        executor->capture_prompt_hidden_ != capture_prompt_hidden_ ||
        !std::ranges::equal(executor->arena_.GetTargetLayerCapture(),
                            arena_.GetTargetLayerCapture()) ||
        item.tokens.empty() || item.tokens.size() > kMaxDecodeBatch) {
      throw std::invalid_argument(
          "Qwen verification has incompatible sessions or token counts");
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (items[previous].executor == executor)
        throw std::invalid_argument("Qwen verification repeats a session");
    }
    if (item.position > executor->arena_.GetMaxContext() ||
        item.tokens.size() > executor->arena_.GetMaxContext() - item.position) {
      throw std::length_error("Qwen verification exceeds a session context");
    }
    offsets[index] = batch_size;
    batch_size += item.tokens.size();
  }
  if (batch_size > arena_.GetMaxBatch()) {
    throw std::length_error(
        "Qwen verification exceeds coordinator scratch capacity");
  }
  for (const auto& item : items) {
    auto& executor = *item.executor;
    if (&executor != &coordinator)
      HIP_CHECK(hipStreamSynchronize(executor.arena_.stream));
    executor.replaying_ssm_state_ = false;
    executor.last_verification_rows_ = item.tokens.size();
    executor.h_verification_hidden_.clear();
    executor.h_verification_logits_.clear();
  }
  const auto scratch = arena_.GetScratchView(batch_size);
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;
  const std::size_t captured_layer_count =
      arena_.GetTargetLayerCapture().size();
  const std::size_t feature_width = captured_layer_count * hidden_size;
  const std::size_t feature_elements =
      capture_prompt_hidden_ ? batch_size * feature_width : 0;
  const std::size_t feature_rows =
      feature_elements / vocab_size + (feature_elements % vocab_size != 0);
  // The last FFN releases this scratch before the vocabulary projection.
  // Reuse it for the combined rows; each session retains only its own logits.
  const auto full_gate = arena_.GetScratchView(arena_.GetMaxBatch()).ffn.gate;
  const bool reuse_gate =
      items.size() > 1 && full_gate.size() >= batch_size * vocab_size;
  coordinator.EnsureVerificationLogits(std::max(
      reuse_gate ? items.front().tokens.size() : batch_size, feature_rows));
  auto* const feature_buffer = coordinator.d_verification_logits_;
  auto* const logits_buffer =
      reuse_gate ? full_gate.data() : coordinator.d_verification_logits_;
  for (const auto& item : items) {
    if (item.executor != &coordinator)
      item.executor->EnsureVerificationLogits(item.tokens.size());
    if (capture_prompt_hidden_)
      item.executor->h_verification_hidden_.resize(
          item.tokens.size() * (feature_width ? feature_width : hidden_size));
  }
  std::array<std::uint32_t, kMaxRows> host_tokens{};
  std::array<std::uint32_t, kMaxRows> host_positions{};
  std::array<SsmSequenceState, kMaxDecodeBatch> ssm_sequences{};
  for (std::size_t index = 0; index < items.size(); ++index) {
    const auto& item = items[index];
    auto& state_arena = item.executor->arena_;
    auto replay = state_arena.GetSsmReplayCapture();
    replay.position = scratch.decode.prompt_tokens.data() + offsets[index];
    ssm_sequences[index] = {state_arena.BeginRecurrentWrite(), replay,
                            static_cast<std::uint32_t>(offsets[index]),
                            static_cast<std::uint32_t>(item.tokens.size())};
    for (std::size_t row = 0; row < item.tokens.size(); ++row) {
      const auto position = item.position + static_cast<std::uint32_t>(row);
      host_tokens[offsets[index] + row] = item.tokens[row];
      host_positions[offsets[index] + row] = position;
      item.executor->arena_.MarkSsmReplayPosition(position);
    }
  }
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                           host_tokens.data(),
                           batch_size * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, arena_.stream));
  LaunchBatchedEmbeddingLookup(
      weights_.token_embd.data, weights_.token_embd.type,
      scratch.decode.prompt_tokens.data(), scratch.decode.hidden.data(),
      batch_size, hidden_size, arena_.stream);
  arena_.SnapVerifyStage("emb", detail::kVerifyDebugNoLayer,
                         scratch.decode.hidden.data(),
                         scratch.decode.hidden.size() / batch_size, batch_size);
  for (std::size_t index = 0; index < items.size(); ++index) {
    items[index].executor->vision_input_.Inject(
        scratch.decode.hidden.data() + offsets[index] * hidden_size,
        items[index].position, items[index].tokens.size(), hidden_size, 1,
        arena_.stream);
  }
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                           host_positions.data(),
                           batch_size * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, arena_.stream));
  EmitDecodeRouteTelemetry(weights_, coordinator.policy_);
  static hipEvent_t prof_a = nullptr;
  static hipEvent_t prof_b = nullptr;
  static const bool prof_on = [] {
    const char* env = ::superroc::core::GetEnv("SUPERROC_MTP_PROFILE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  if (prof_on && prof_a == nullptr) {
    (void)hipEventCreate(&prof_a);
    (void)hipEventCreate(&prof_b);
  }
  VerifyProf prof;
  prof.on = prof_on;
  prof.a = prof_a;
  prof.b = prof_b;
  for (std::uint32_t layer_index = 0; layer_index < config.num_layers;
       ++layer_index) {
    const auto& layer = weights_.layers[layer_index];
    const auto route_plan =
        ResolveQwenLayerRoute(coordinator.policy_, QwenExecutionMode::kDecode,
                              layer.is_full_attention);

    LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                         static_cast<const float*>(layer.attn_norm.data),
                         scratch.decode.normed.data(), nullptr, batch_size,
                         hidden_size, 1e-6F, arena_.stream);
    arena_.SnapVerifyStage(
        "attn_norm", layer_index, scratch.decode.normed.data(),
        scratch.decode.normed.size() / batch_size, batch_size);

    if (layer.is_full_attention) {
      prof.Begin(arena_.stream);
      const bool qkv_integer =
          DecodeQkvIntegerPath(layer, q_projection_size, kv_size, hidden_size);
      LaunchVerifyQkvGroup(
          layer, scratch.decode.normed.data(), scratch.ssm.qkv.data(),
          scratch.attention.k.data(), scratch.attention.v.data(), batch_size,
          q_projection_size, kv_size, hidden_size, arena_.stream, qkv_integer);
      LaunchBatchedUnpackQG(scratch.ssm.qkv.data(), scratch.attention.q.data(),
                            scratch.ssm.gate.data(), batch_size,
                            config.num_attention_heads, config.head_dim,
                            arena_.stream);
      arena_.SnapVerifyStage("q", layer_index, scratch.attention.q.data(),
                             scratch.attention.q.size() / batch_size,
                             batch_size);
      arena_.SnapVerifyStage("k", layer_index, scratch.attention.k.data(),
                             scratch.attention.k.size() / batch_size,
                             batch_size);
      arena_.SnapVerifyStage("v", layer_index, scratch.attention.v.data(),
                             scratch.attention.v.size() / batch_size,
                             batch_size);
      arena_.SnapVerifyStage("gate", layer_index, scratch.ssm.gate.data(),
                             scratch.ssm.gate.size() / batch_size, batch_size);
      prof.End(arena_.stream, prof.proj_ms);
      prof.proj_b += VerifyTrafficBytes(layer.attn_q, batch_size) +
                     VerifyTrafficBytes(layer.attn_k, batch_size) +
                     VerifyTrafficBytes(layer.attn_v, batch_size);

      prof.Begin(arena_.stream);
      const std::uint32_t attention_layer =
          layer_index / config.full_attention_interval;
      const bool fused_qknorm_rope_kv =
          route_plan.fuse_qk_norm_rope_kv &&
          detail::IsFusedQkNormSupported(config.head_dim);
      if (!fused_qknorm_rope_kv) {
        if (!layer.attn_q_norm.empty()) {
          LaunchBatchedPerHeadRMSNorm(
              scratch.attention.q.data(),
              static_cast<const float*>(layer.attn_q_norm.data),
              scratch.attention.q.data(), batch_size,
              config.num_attention_heads, config.head_dim, 1e-6F,
              arena_.stream);
        }
        if (!layer.attn_k_norm.empty()) {
          LaunchBatchedPerHeadRMSNorm(
              scratch.attention.k.data(),
              static_cast<const float*>(layer.attn_k_norm.data),
              scratch.attention.k.data(), batch_size,
              config.num_key_value_heads, config.head_dim, 1e-6F,
              arena_.stream);
        }
      }

      if (!fused_qknorm_rope_kv) {
        arena_.SnapVerifyStage("qkn_q", layer_index, scratch.attention.q.data(),
                               scratch.attention.q.size() / batch_size,
                               batch_size);
        arena_.SnapVerifyStage("qkn_k", layer_index, scratch.attention.k.data(),
                               scratch.attention.k.size() / batch_size,
                               batch_size);
      }
      for (std::size_t index = 0; index < items.size(); ++index) {
        const auto& item = items[index];
        auto& state_arena = item.executor->arena_;
        const std::size_t offset = offsets[index];
        const std::size_t sequence_size = item.tokens.size();
        const std::uint32_t start_pos = item.position;
        const std::size_t total_k = state_arena.GetAttentionKvPlaneElements();
        if (fused_qknorm_rope_kv) {
          LaunchBatchedFusedQKNormRoPEKvWrite(
              (scratch.attention.q.data() + offset * attention_size),
              (scratch.attention.k.data() + offset * kv_size),
              (scratch.attention.v.data() + offset * kv_size),
              static_cast<const float*>(layer.attn_q_norm.data),
              static_cast<const float*>(layer.attn_k_norm.data),
              (scratch.attention.q.data() + offset * attention_size),
              (scratch.attention.k.data() + offset * kv_size),
              state_arena.d_kv_cache,
              OffsetIfPresent(state_arena.d_kv_cache, total_k),
              state_arena.d_attention_kv_f16,
              OffsetIfPresent(
                  static_cast<std::uint16_t*>(state_arena.d_attention_kv_f16),
                  total_k),
              attention_layer, start_pos, sequence_size,
              state_arena.GetMaxContext(), config.num_attention_heads,
              config.num_key_value_heads, config.head_dim, config.rotary_dim,
              config.rope_theta, 1e-6F, arena_.stream,
              item.executor->vision_input_.rope());
          // Exact verification projections do not use weight dequantization
          // scratch. Reuse it for independent split-K rows without allocating
          // another buffer or changing the scalar decode workspace.
          auto attention_scratch = scratch.attention.split_k;
          if (scratch.decode.weight_bf16.size_bytes() >=
              sequence_size * attention_scratch.size_bytes()) {
            attention_scratch = {
                reinterpret_cast<float*>(scratch.decode.weight_bf16.data()),
                sequence_size * attention_scratch.size()};
          }
          LaunchCausalDecodeAttention(
              (scratch.attention.q.data() + offset * attention_size),
              (scratch.ssm.gate.data() + offset * attention_size),
              state_arena.d_kv_cache,
              OffsetIfPresent(state_arena.d_kv_cache, total_k),
              state_arena.d_attention_kv_f16,
              OffsetIfPresent(
                  static_cast<std::uint16_t*>(state_arena.d_attention_kv_f16),
                  total_k),
              (scratch.ssm.out.data() + offset * attention_size),
              attention_layer, start_pos, sequence_size,
              state_arena.GetMaxContext(), config.num_attention_heads,
              config.num_key_value_heads, config.head_dim, arena_.stream,
              attention_scratch);
        } else {
          for (std::size_t row = 0; row < sequence_size; ++row) {
            float* const query =
                (scratch.attention.q.data() + offset * attention_size) +
                (row * attention_size);
            float* const key = (scratch.attention.k.data() + offset * kv_size) +
                               (row * kv_size);
            const std::uint32_t position = host_positions[offset + row];
            LaunchRoPE(query, key, config.num_attention_heads,
                       config.num_key_value_heads, config.head_dim,
                       config.rotary_dim, position, config.rope_theta,
                       arena_.stream, item.executor->vision_input_.rope());
            LaunchAttention(
                query, key,
                (scratch.attention.v.data() + offset * kv_size) + row * kv_size,
                (scratch.ssm.gate.data() + offset * attention_size) +
                    row * attention_size,
                state_arena.d_kv_cache,
                OffsetIfPresent(state_arena.d_kv_cache, total_k),
                state_arena.d_attention_kv_f16,
                OffsetIfPresent(
                    static_cast<std::uint16_t*>(state_arena.d_attention_kv_f16),
                    total_k),
                (scratch.ssm.out.data() + offset * attention_size) +
                    row * attention_size,
                attention_layer, position, state_arena.GetMaxContext(),
                config.num_attention_heads, config.num_key_value_heads,
                config.head_dim, arena_.stream,
                state_arena.d_split_k_attention);
          }
        }
      }
      if (!fused_qknorm_rope_kv) {
        arena_.SnapVerifyStage(
            "rope_q", layer_index, scratch.attention.q.data(),
            scratch.attention.q.size() / batch_size, batch_size);
        arena_.SnapVerifyStage(
            "rope_k", layer_index, scratch.attention.k.data(),
            scratch.attention.k.size() / batch_size, batch_size);
      } else {
        arena_.SnapVerifyStage(
            "qkrope_q", layer_index, scratch.attention.q.data(),
            scratch.attention.q.size() / batch_size, batch_size);
        arena_.SnapVerifyStage(
            "qkrope_k", layer_index, scratch.attention.k.data(),
            scratch.attention.k.size() / batch_size, batch_size);
      }
      arena_.SnapVerifyStage("attn_out", layer_index, scratch.ssm.out.data(),
                             scratch.ssm.out.size() / batch_size, batch_size);
      prof.End(arena_.stream, prof.attn_ms);
      if (prof.on) {
        for (std::size_t index = 0; index < items.size(); ++index) {
          const std::size_t seq = items[index].tokens.size();
          const std::size_t pos = items[index].position;
          prof.attn_b +=
              seq * 2 * kv_size * (pos + seq) * sizeof(std::uint16_t);
        }
      }
      prof.Begin(arena_.stream);
      LaunchVerifyProjection(
          layer.attn_output, scratch.ssm.out.data(),
          scratch.attention.output.data(), batch_size, hidden_size,
          attention_size, arena_.stream,
          DecodeSingleIntegerPath(layer.attn_output, hidden_size,
                                  attention_size));
      arena_.SnapVerifyStage(
          "mix_proj", layer_index, scratch.attention.output.data(),
          scratch.attention.output.size() / batch_size, batch_size);
      prof.End(arena_.stream, prof.proj_ms);
      prof.proj_b += VerifyTrafficBytes(layer.attn_output, batch_size);
    } else {
      prof.Begin(arena_.stream);
      const bool ssm_integer = DecodeSsmIntegerPath(
          layer, ssm_qkv_size, ssm_inner_size, time_step_rank, hidden_size);
      const auto controls = LaunchVerifySsmGroup(
          layer, scratch, batch_size, ssm_qkv_size, ssm_inner_size,
          time_step_rank, hidden_size, arena_.stream, ssm_integer);
      arena_.SnapVerifyStage("qkv", layer_index, scratch.ssm.qkv.data(),
                             scratch.ssm.qkv.size() / batch_size, batch_size);
      arena_.SnapVerifyStage("gate", layer_index, scratch.ssm.gate.data(),
                             scratch.ssm.gate.size() / batch_size, batch_size);
      if (controls.row_stride == time_step_rank) {
        arena_.SnapVerifyStage("alpha", layer_index, controls.alpha,
                               time_step_rank, batch_size);
        arena_.SnapVerifyStage("beta", layer_index, controls.beta,
                               time_step_rank, batch_size);
      } else {
        arena_.SnapVerifyStage("alpha_pack", layer_index, controls.alpha,
                               controls.row_stride, batch_size);
      }
      prof.End(arena_.stream, prof.proj_ms);
      prof.proj_b += VerifyTrafficBytes(layer.attn_qkv, batch_size) +
                     VerifyTrafficBytes(layer.attn_gate, batch_size);
      prof.Begin(arena_.stream);
      LaunchSSMConvRecurrenceBatch(
          scratch.ssm.qkv.data(),
          static_cast<const float*>(layer.ssm_conv1d.data),
          scratch.ssm.conv_out.data(), controls.alpha, controls.beta,
          static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data),
          static_cast<const float*>(layer.ssm_norm.data),
          scratch.ssm.gate.data(), scratch.ssm.out.data(),
          std::span(ssm_sequences).first(items.size()),
          config.SsmLayerIndex(layer_index), ssm_qkv_size,
          config.ssm_group_count, config.ssm_time_step_rank,
          config.ssm_state_size, config.SsmValueSize(), controls.row_stride,
          ssm_inner_size, arena_.stream, arena_.GetRecurrentStateStorage());
      arena_.SnapVerifyStage(
          "conv_out", layer_index, scratch.ssm.conv_out.data(),
          scratch.ssm.conv_out.size() / batch_size, batch_size);
      arena_.SnapVerifyStage("ssm_out", layer_index, scratch.ssm.out.data(),
                             scratch.ssm.out.size() / batch_size, batch_size);
      prof.End(arena_.stream, prof.ssm_ms);
      prof.ssm_b += VerifyWeightBytes(
          core::GgmlType::kF32,
          config.ssm_state_size * ssm_inner_size * batch_size);
      prof.Begin(arena_.stream);
      LaunchVerifyProjection(
          layer.ssm_out, scratch.ssm.out.data(),
          scratch.attention.output.data(), batch_size, hidden_size,
          ssm_inner_size, arena_.stream,
          DecodeSingleIntegerPath(layer.ssm_out, hidden_size, ssm_inner_size));
      arena_.SnapVerifyStage(
          "mix_proj", layer_index, scratch.attention.output.data(),
          scratch.attention.output.size() / batch_size, batch_size);
      prof.End(arena_.stream, prof.proj_ms);
      prof.proj_b += VerifyTrafficBytes(layer.ssm_out, batch_size);
    }

    prof.Begin(arena_.stream);
    LaunchBatchedResidualAdd(
        scratch.decode.hidden.data(), scratch.attention.output.data(),
        scratch.decode.hidden.data(), batch_size, hidden_size, arena_.stream);
    LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                         static_cast<const float*>(layer.ffn_norm.data),
                         scratch.decode.normed.data(), nullptr, batch_size,
                         hidden_size, 1e-6F, arena_.stream);
    arena_.SnapVerifyStage(
        "ffn_norm", layer_index, scratch.decode.normed.data(),
        scratch.decode.normed.size() / batch_size, batch_size);

    if (layer.is_moe) {
      LaunchVerifyFfn(
          layer, config, scratch.decode.normed.data(), scratch.ffn.out.data(),
          arena_.d_moe_logits, arena_.d_moe_ids, arena_.d_moe_weights,
          arena_.d_moe_sig, arena_.d_moe_zero_ids, arena_.d_moe_y,
          arena_.d_moe_shexp_y, batch_size, hidden_size, arena_.stream,
          arena_.d_moe_gate_e, arena_.d_moe_slot_y, arena_.d_moe_shexp_h);
    } else {
      LaunchFfnActivation(layer, scratch, batch_size, intermediate_size,
                          hidden_size, arena_.stream);
      LaunchVerifyProjection(
          layer.ffn_down, scratch.ffn.activation.data(), scratch.ffn.out.data(),
          batch_size, hidden_size, intermediate_size, arena_.stream,
          DecodeSingleIntegerPath(layer.ffn_down, hidden_size,
                                  intermediate_size));
      arena_.SnapVerifyStage(
          "ffn_act", layer_index, scratch.ffn.activation.data(),
          scratch.ffn.activation.size() / batch_size, batch_size);
    }
    arena_.SnapVerifyStage("ffn_out", layer_index, scratch.ffn.out.data(),
                           scratch.ffn.out.size() / batch_size, batch_size);
    LaunchBatchedResidualAdd(
        scratch.decode.hidden.data(), scratch.ffn.out.data(),
        scratch.decode.hidden.data(), batch_size, hidden_size, arena_.stream);
    arena_.SnapVerifyStage("hidden", layer_index, scratch.decode.hidden.data(),
                           scratch.decode.hidden.size() / batch_size,
                           batch_size);
    prof.End(arena_.stream, prof.ffn_ms);
    prof.ffn_b += VerifyTrafficBytes(layer.ffn_gate, batch_size) +
                  VerifyTrafficBytes(layer.ffn_up, batch_size) +
                  VerifyTrafficBytes(layer.ffn_down, batch_size);
    if (capture_prompt_hidden_) {
      if (const auto tap = arena_.GetTargetLayerCaptureIndex(layer_index);
          tap.has_value()) {
        float* const destination = feature_buffer + (*tap * hidden_size);
        HIP_CHECK(hipMemcpy2DAsync(destination, feature_width * sizeof(float),
                                   scratch.decode.hidden.data(),
                                   hidden_size * sizeof(float),
                                   hidden_size * sizeof(float), batch_size,
                                   hipMemcpyDeviceToDevice, arena_.stream));
      }
    }
  }

  for (std::size_t index = 0; index < items.size(); ++index) {
    const auto& item = items[index];
    auto& executor = *item.executor;
    const std::size_t offset = offsets[index];
    if (capture_prompt_hidden_ && captured_layer_count > 0) {
      HIP_CHECK(
          hipMemcpyAsync(executor.h_verification_hidden_.data(),
                         feature_buffer + offset * feature_width,
                         item.tokens.size() * feature_width * sizeof(float),
                         hipMemcpyDeviceToHost, arena_.stream));
      HIP_CHECK(hipMemcpyAsync(
          executor.arena_.d_target_layer_features,
          feature_buffer + (offset + item.tokens.size() - 1) * feature_width,
          feature_width * sizeof(float), hipMemcpyDeviceToDevice,
          arena_.stream));
    } else if (capture_prompt_hidden_) {
      HIP_CHECK(
          hipMemcpyAsync(executor.h_verification_hidden_.data(),
                         scratch.decode.hidden.data() + offset * hidden_size,
                         item.tokens.size() * hidden_size * sizeof(float),
                         hipMemcpyDeviceToHost, arena_.stream));
    }
    if (&executor == &coordinator) {
      executor.last_hidden_offset_ = (item.tokens.size() - 1) * hidden_size;
    } else {
      HIP_CHECK(hipMemcpyAsync(
          executor.arena_.d_hidden,
          scratch.decode.hidden.data() +
              (offset + item.tokens.size() - 1) * hidden_size,
          hidden_size * sizeof(float), hipMemcpyDeviceToDevice, arena_.stream));
      executor.last_hidden_offset_ = 0;
    }
  }
  // Name the coordinator stream's set, so concurrent sessions check
  // theirs. A refusal fails the step loudly (unreachable in serve).
  if (config.is_moe) {
    const int slot_set = superroc_fp4_slot_stream_attach_at(
        arena_.stream, __func__, static_cast<long>(hidden_size));
    if (slot_set < 0) {
      throw std::runtime_error("qwen35moe verify slot set refused");
    }
    // Stream-ordered check (the bare read proved stale on the
    // box: STALE_STATUS missed 1/200).
    if (superroc_fp4_moe_dev_check_set_on_stream(slot_set, arena_.stream) !=
        SUPERROC_FP4_OK) {
      throw std::runtime_error("qwen35moe verify expert device status");
    }
  }
  prof.Begin(arena_.stream);
  LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                       static_cast<const float*>(weights_.output_norm.data),
                       scratch.decode.normed.data(), nullptr, batch_size,
                       hidden_size, 1e-6F, arena_.stream);
  arena_.SnapVerifyStage("final_norm", detail::kVerifyDebugNoLayer,
                         scratch.decode.normed.data(),
                         scratch.decode.normed.size() / batch_size, batch_size);
  LaunchVerifyProjection(
      weights_.output, scratch.decode.normed.data(), logits_buffer, batch_size,
      vocab_size, hidden_size, arena_.stream,
      DecodeSingleIntegerPath(weights_.output, vocab_size, hidden_size));
  arena_.SnapVerifyStage("logits", detail::kVerifyDebugNoLayer, logits_buffer,
                         vocab_size, batch_size);
  prof.End(arena_.stream, prof.head_ms);
  prof.head_b += VerifyTrafficBytes(weights_.output, batch_size);
  prof.Begin(arena_.stream);
  auto* const d_out_tokens = scratch.decode.prompt_tokens.data();
  LaunchBatchedGPUArgmax(logits_buffer, d_out_tokens, batch_size, vocab_size,
                         scratch.ffn.out, arena_.stream);
  std::array<tokenization::TokenId, kMaxRows> host_predictions{};
  HIP_CHECK(hipMemcpyAsync(host_predictions.data(), d_out_tokens,
                           batch_size * sizeof(tokenization::TokenId),
                           hipMemcpyDeviceToHost, arena_.stream));
  for (std::size_t index = 0; index < items.size(); ++index) {
    const auto& item = items[index];
    auto& executor = *item.executor;
    const std::size_t count = item.tokens.size() * vocab_size;
    const auto* logits = logits_buffer + offsets[index] * vocab_size;
    if (logits != executor.d_verification_logits_) {
      HIP_CHECK(hipMemcpyAsync(executor.d_verification_logits_, logits,
                               count * sizeof(float), hipMemcpyDeviceToDevice,
                               arena_.stream));
    }
    if (item.capture_logits) {
      executor.h_verification_logits_.resize(count);
      HIP_CHECK(hipMemcpyAsync(executor.h_verification_logits_.data(), logits,
                               count * sizeof(float), hipMemcpyDeviceToHost,
                               arena_.stream));
    }
  }
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  prof.End(arena_.stream, prof.sample_ms);
  if (prof.on) {
    const auto gbps = [](std::size_t bytes, double ms) {
      return ms > 0.0 ? static_cast<double>(bytes) / (ms * 1.0e6) : 0.0;
    };
    std::fprintf(stderr,
                 "VERIFY_PROF pos=%u batch=%zu proj_ms=%.3f attn_ms=%.3f "
                 "ssm_ms=%.3f ffn_ms=%.3f head_ms=%.3f sample_ms=%.3f "
                 "proj_B=%zu attn_B=%zu ssm_B=%zu ffn_B=%zu head_B=%zu "
                 "proj_GBps=%.1f attn_GBps=%.1f ssm_GBps=%.1f ffn_GBps=%.1f "
                 "head_GBps=%.1f\n",
                 items.front().position, batch_size, prof.proj_ms, prof.attn_ms,
                 prof.ssm_ms, prof.ffn_ms, prof.head_ms, prof.sample_ms,
                 prof.proj_b, prof.attn_b, prof.ssm_b, prof.ffn_b, prof.head_b,
                 gbps(prof.proj_b, prof.proj_ms),
                 gbps(prof.attn_b, prof.attn_ms), gbps(prof.ssm_b, prof.ssm_ms),
                 gbps(prof.ffn_b, prof.ffn_ms),
                 gbps(prof.head_b, prof.head_ms));
  }
  for (std::size_t row = 0; row < batch_size; ++row)
    (void)CheckedSampleToken(host_predictions[row], vocab_size);
  std::vector<std::vector<tokenization::TokenId>> predictions;
  predictions.reserve(items.size());
  for (std::size_t index = 0; index < items.size(); ++index) {
    const auto& item = items[index];
    predictions.emplace_back(
        host_predictions.begin() + offsets[index],
        host_predictions.begin() + offsets[index] + item.tokens.size());
  }
  return predictions;
}

namespace {

// LFM2: lfm2_decode.hip's Require pattern, for both launcher status types.
void Require(const Lfm2LaunchStatus& status) {
  if (!status.ok) {
    throw std::runtime_error(status.error != nullptr ? status.error
                                                     : "lfm2moe launch failed");
  }
}

void Require(const GlimmerLaunchStatus& status) {
  if (!status.ok) {
    throw std::runtime_error(status.error != nullptr ? status.error
                                                     : "glimmer launch failed");
  }
}

void Require(const FlintLaunchStatus& status) {
  if (!status.ok) {
    throw std::runtime_error(status.error != nullptr ? status.error
                                                     : "flint launch failed");
  }
}

}  // namespace

// LFM2: speculative verification for lfm2moe, rows 2..8. The projections run
// once for all rows through the verify_m intercepts (shared weight reads);
// the short-conv decode, the KV write and the attention stay per row and in
// row order, because each row reads the conv columns and the K/V entries the
// row before it wrote. Every kernel and its arguments are the row's exact
// decode launches (ExecuteLfm2DecodeStep), so logits and states are
// bit-identical to walking the chunk token by token.
std::vector<tokenization::TokenId> QwenGpuExecutor::ForwardLfm2VerificationRows(
    std::span<const tokenization::TokenId> tokens, std::uint32_t start_pos,
    bool capture_logits) {
  CheckReset();
  if (vision_input_.Bytes() != 0) {
    throw std::runtime_error("lfm2moe vision is not supported");
  }
  const auto& config = weights_.config;
  if (!config.lfm2) {
    throw std::runtime_error("lfm2moe verify called for a non-lfm2 config");
  }
  if (arena_.d_kv_cache == nullptr && arena_.d_attention_kv_f16 == nullptr) {
    throw std::runtime_error("lfm2moe requires a KV cache");  // LFM2
  }
  if (arena_.d_ssm_conv_state == nullptr) {
    throw std::runtime_error("lfm2moe requires ssm conv state");
  }
  const std::size_t rows = tokens.size();
  if (start_pos > arena_.GetMaxContext() ||
      rows > arena_.GetMaxContext() - start_pos) {
    throw std::length_error("lfm2moe verify exceeds the GPU context length");
  }
  last_verification_rows_ = rows;
  h_verification_hidden_.clear();
  h_verification_logits_.clear();
  replaying_ssm_state_ = false;
  arena_.DisableSsmReplayCapture();
  (void)arena_.BeginRecurrentWrite();
  EnsureVerificationLogits(rows);

  const std::size_t H = config.hidden_size;
  const std::size_t attn =
      static_cast<std::size_t>(config.num_attention_heads) * config.head_dim;
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t I = config.intermediate_size;
  const float eps = config.norm_eps_pre;
  const auto scratch = arena_.GetScratchView(rows);
  if (scratch.decode.hidden.size() < rows * H ||
      scratch.decode.normed.size() < rows * H ||
      scratch.attention.q.size() < rows * attn ||
      scratch.attention.k.size() < rows * kv ||
      scratch.attention.v.size() < rows * kv ||
      scratch.ssm.out.size() < rows * std::max(H, attn) ||
      scratch.attention.output.size() < rows * H ||
      scratch.ffn.activation.size() < rows * std::max(I, 3 * H) ||
      scratch.ffn.out.size() < rows * H ||
      scratch.decode.prompt_tokens.size() < rows) {
    throw std::runtime_error("lfm2 verify scratch is short");
  }

  const std::size_t vocab_size = config.vocab_size;
  const std::size_t captured_layer_count =
      arena_.GetTargetLayerCapture().size();
  const std::size_t feature_width = captured_layer_count * H;
  const std::size_t feature_elements =
      capture_prompt_hidden_ ? rows * feature_width : 0;
  const std::size_t feature_rows =
      feature_elements / vocab_size + (feature_elements % vocab_size != 0);
  // The head writes the rows' logits into the same buffer that held the
  // drafter features, exactly as the single-session ForwardVerificationBatch
  // does: the features leave for the host and the tap plane before it runs.
  if (capture_prompt_hidden_) {
    EnsureVerificationLogits(std::max(rows, feature_rows));
    h_verification_hidden_.resize(rows * (feature_width ? feature_width : H));
  }
  auto* const feature_buffer = d_verification_logits_;
  auto* const logits_buffer = d_verification_logits_;

  // LFM2: one saved conv state per row (the size QwenGpuArena::SaveState
  // copies), so CommitVerificationChunk can accept any prefix of the rows.
  const std::size_t conv_floats = config.Lfm2ConvStateFloats();
  const std::size_t total_conv =
      static_cast<std::size_t>(config.SsmLayerCount()) * config.SsmQkvSize() *
      config.ssm_conv_kernel;
  if (d_lfm2_row_conv_ == nullptr) {
    HIP_CHECK(hipMalloc(&d_lfm2_row_conv_, 8 * total_conv * sizeof(float)));
  }

  std::vector<std::uint32_t> host_rows(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    host_rows[r] = static_cast<std::uint32_t>(tokens[r]);
  }
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                           host_rows.data(), rows * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, arena_.stream));
  LaunchBatchedEmbeddingLookup(
      weights_.token_embd.data, weights_.token_embd.type,
      scratch.decode.prompt_tokens.data(), scratch.decode.hidden.data(), rows,
      H, arena_.stream);
  // LFM2: NO norm after the embedding. The positions replace the token ids
  // so every row carries its own d_pos word, as the decode step's pair does.
  for (std::size_t r = 0; r < rows; ++r) {
    host_rows[r] = start_pos + static_cast<std::uint32_t>(r);
  }
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                           host_rows.data(), rows * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, arena_.stream));
  const std::uint32_t* d_pos = scratch.decode.prompt_tokens.data();

  const std::uint32_t max_ctx = arena_.GetMaxContext();
  const std::uint32_t heads = config.num_attention_heads;
  const std::uint32_t kv_heads = config.num_key_value_heads;
  const std::uint32_t head_dim = config.head_dim;
  const std::uint32_t rotary_dim = config.rotary_dim;
  const float rope_theta = config.rope_theta;
  const std::uint32_t n_expert = config.expert_count;
  const std::uint32_t n_used = config.expert_used_count;
  const std::uint32_t expert_ff = config.expert_feed_forward_length;
  const std::uint32_t l_cache = config.shortconv_l_cache;
  const std::size_t plane = arena_.GetAttentionKvPlaneElements();
  // LFM2: fp16 token-major plane when the arena has one, else fp32.
  const bool kv_f16 = arena_.d_attention_kv_f16 != nullptr;
  float* v_cache = kv_f16 ? nullptr : arena_.d_kv_cache + plane;
  auto* k_f16 = static_cast<std::uint16_t*>(arena_.d_attention_kv_f16);
  std::uint16_t* v_f16 = kv_f16 ? k_f16 + plane : nullptr;
  // LFM2: normed holds attn_norm(hidden) at the top of every layer.
  // Layer 0 norms here; every residual add below is fused with the RMSNorm
  // that follows it (ffn_norm, the next layer's attn_norm, output_norm).
  for (std::size_t r = 0; r < rows; ++r) {
    LaunchRMSNorm(scratch.decode.hidden.data() + r * H,
                  static_cast<const float*>(weights_.layers[0].attn_norm.data),
                  scratch.decode.normed.data() + r * H,
                  static_cast<std::uint32_t>(H), eps, arena_.stream);
  }
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const auto& layer = weights_.layers[l];
    if (layer.is_shortconv) {
      // Rows B | C | X of the in-projection land in ffn.activation [3H].
      LaunchVerifyProjection(
          layer.shortconv_in_proj, scratch.decode.normed.data(),
          scratch.ffn.activation.data(), rows, 3 * H, H, arena_.stream,
          DecodeSingleIntegerPath(layer.shortconv_in_proj, 3 * H, H));
      const std::size_t kind = config.Lfm2KindIndex(l);
      for (std::size_t r = 0; r < rows; ++r) {
        Require(LaunchLfm2ShortConvDecode(
            scratch.ffn.activation.data() + r * 3 * H,
            static_cast<const float*>(layer.shortconv_conv.data),
            arena_.d_ssm_conv_state + kind * conv_floats,
            scratch.ssm.out.data() + r * H, static_cast<std::uint32_t>(H),
            l_cache, arena_.stream));
        HIP_CHECK(hipMemcpyAsync(
            d_lfm2_row_conv_ + r * total_conv + kind * conv_floats,
            arena_.d_ssm_conv_state + kind * conv_floats,
            conv_floats * sizeof(float), hipMemcpyDeviceToDevice,
            arena_.stream));
      }
      LaunchVerifyProjection(
          layer.shortconv_out_proj, scratch.ssm.out.data(),
          scratch.attention.output.data(), rows, H, H, arena_.stream,
          DecodeSingleIntegerPath(layer.shortconv_out_proj, H, H));
    } else {  // LFM2 attention layer
      // LFM2: with ROCmFP4 q/k/v the group quantises normed once and
      // shares that Q8 copy, the same quant and matvec kernels the decode
      // step's slot path runs; any refusal falls back per tensor to the
      // launches LaunchGEMV serves decode.
      const bool shared_qkv = IsRocmFp4(layer.attn_q.type) &&
                              IsRocmFp4(layer.attn_k.type) &&
                              IsRocmFp4(layer.attn_v.type) &&
                              RocmFp4ShapeOk(attn, H) && RocmFp4ShapeOk(kv, H);
      if (!shared_qkv ||
          !TryLaunchVerifyIntegerGroup(
              {{&layer.attn_q, scratch.attention.q.data(), attn},
               {&layer.attn_k, scratch.attention.k.data(), kv},
               {&layer.attn_v, scratch.attention.v.data(), kv}},
              scratch.decode.normed.data(), rows, H, arena_.stream)) {
        LaunchVerifyProjection(layer.attn_q, scratch.decode.normed.data(),
                               scratch.attention.q.data(), rows, attn, H,
                               arena_.stream,
                               DecodeSingleIntegerPath(layer.attn_q, attn, H));
        LaunchVerifyProjection(layer.attn_k, scratch.decode.normed.data(),
                               scratch.attention.k.data(), rows, kv, H,
                               arena_.stream,
                               DecodeSingleIntegerPath(layer.attn_k, kv, H));
        LaunchVerifyProjection(layer.attn_v, scratch.decode.normed.data(),
                               scratch.attention.v.data(), rows, kv, H,
                               arena_.stream,
                               DecodeSingleIntegerPath(layer.attn_v, kv, H));
      }
      // Row order matters: row r reads the K/V rows the rows before it wrote.
      for (std::size_t r = 0; r < rows; ++r) {
        float* q = scratch.attention.q.data() + r * attn;
        float* k = scratch.attention.k.data() + r * kv;
        float* v = scratch.attention.v.data() + r * kv;
        const std::uint32_t* row_pos = d_pos + r;
        LaunchPerHeadRMSNorm(q,
                             static_cast<const float*>(layer.attn_q_norm.data),
                             q, heads, head_dim, eps, arena_.stream);
        LaunchPerHeadRMSNorm(k,
                             static_cast<const float*>(layer.attn_k_norm.data),
                             k, kv_heads, head_dim, eps, arena_.stream);
        LaunchRoPE(q, k, heads, kv_heads, head_dim, rotary_dim, row_pos,
                   rope_theta, arena_.stream, nullptr);  // LFM2 (NEOX)
        if (kv_f16) {                                    // LFM2
          Require(LaunchLfm2WriteKvDevF16(
              k, v, k_f16, v_f16, config.Lfm2KindIndex(l), row_pos, max_ctx,
              kv_heads, head_dim, arena_.stream));
          Require(LaunchLfm2DecodeAttentionF16(
              q, k_f16, v_f16, scratch.ssm.out.data() + r * attn,
              config.Lfm2KindIndex(l), row_pos, max_ctx, heads, kv_heads,
              head_dim, 32, arena_.d_lfm2_attn_partial, arena_.stream));
        } else {
          Require(LaunchLfm2WriteKvDev(
              k, v, arena_.d_kv_cache, v_cache, config.Lfm2KindIndex(l),
              row_pos, max_ctx, kv_heads, head_dim, arena_.stream));
          Require(LaunchLfm2DecodeAttention(
              q, arena_.d_kv_cache, v_cache, scratch.ssm.out.data() + r * attn,
              config.Lfm2KindIndex(l), row_pos, max_ctx, heads, kv_heads,
              head_dim, 32, arena_.d_lfm2_attn_partial, arena_.stream));
        }
      }
      LaunchVerifyProjection(
          layer.attn_output, scratch.ssm.out.data(),
          scratch.attention.output.data(), rows, H, attn, arena_.stream,
          DecodeSingleIntegerPath(layer.attn_output, H, attn));
    }
    for (std::size_t r = 0; r < rows; ++r) {
      Require(LaunchLfm2AddRMSNorm(  // LFM2
          scratch.decode.hidden.data() + r * H,
          scratch.attention.output.data() + r * H,
          static_cast<const float*>(layer.ffn_norm.data),
          scratch.decode.normed.data() + r * H, static_cast<std::uint32_t>(H),
          eps, arena_.stream));
    }
    if (!layer.is_moe) {  // LFM2 dense lead layer
      LaunchFfnActivation(layer, scratch, rows, I, H, arena_.stream);
      LaunchVerifyProjection(layer.ffn_down, scratch.ffn.activation.data(),
                             scratch.ffn.out.data(), rows, H, I, arena_.stream,
                             DecodeSingleIntegerPath(layer.ffn_down, H, I));
    } else {  // LFM2 MoE layer
      for (std::size_t r = 0; r < rows; ++r) {
        Require(LaunchLfm2MoeRoute(
            scratch.decode.normed.data() + r * H,
            static_cast<const float*>(layer.ffn_gate_inp.data),
            static_cast<const float*>(layer.exp_probs_b.data),
            static_cast<std::uint32_t>(H), n_expert, n_used,
            arena_.d_moe_logits + r * n_expert, arena_.d_moe_ids + r * n_used,
            arena_.d_moe_weights + r * n_used, arena_.stream));
      }
      if (superroc_fp4_expert_ffn_dev_batch(
              static_cast<int>(layer.ffn_gate_exps.type),
              layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
              layer.ffn_down_exps.data, scratch.decode.normed.data(),
              arena_.d_moe_ids, arena_.d_moe_weights, static_cast<int>(rows),
              static_cast<int>(n_used), static_cast<int>(n_expert),
              static_cast<int>(H), static_cast<int>(expert_ff), 0.0F,
              scratch.ffn.out.data(), arena_.stream) != SUPERROC_FP4_OK) {
        // Same per-row launches the decode step runs, on the same refusal.
        for (std::size_t r = 0; r < rows; ++r) {
          if (superroc_fp4_expert_ffn_dev(
                  static_cast<int>(layer.ffn_gate_exps.type),
                  layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
                  layer.ffn_down_exps.data,
                  scratch.decode.normed.data() + r * H,
                  arena_.d_moe_ids + r * n_used,
                  arena_.d_moe_weights + r * n_used, static_cast<int>(n_used),
                  static_cast<int>(n_expert), static_cast<int>(H),
                  static_cast<int>(expert_ff), 0.0F,
                  scratch.ffn.out.data() + r * H,
                  arena_.stream) != SUPERROC_FP4_OK) {
            throw std::runtime_error("lfm2moe decode experts failed");
          }
        }
      }
    }
    // LFM2: residual + the next RMSNorm (next layer's attn_norm, or
    // output_norm after the last layer).
    const float* next_norm =
        l + 1U < config.num_layers
            ? static_cast<const float*>(weights_.layers[l + 1U].attn_norm.data)
            : static_cast<const float*>(weights_.output_norm.data);
    for (std::size_t r = 0; r < rows; ++r) {
      Require(LaunchLfm2AddRMSNorm(
          scratch.decode.hidden.data() + r * H, scratch.ffn.out.data() + r * H,
          next_norm, scratch.decode.normed.data() + r * H,
          static_cast<std::uint32_t>(H), eps, arena_.stream));
    }
    // DSpark tap: the layer output is the hidden state the drafter's
    // feature-fusion encoder consumes (same point as the Qwen paths).
    if (capture_prompt_hidden_) {
      if (const auto tap = arena_.GetTargetLayerCaptureIndex(l);
          tap.has_value()) {
        float* const destination = feature_buffer + (*tap * H);
        HIP_CHECK(hipMemcpy2DAsync(destination, feature_width * sizeof(float),
                                   scratch.decode.hidden.data(),
                                   H * sizeof(float), H * sizeof(float), rows,
                                   hipMemcpyDeviceToDevice, arena_.stream));
      }
    }
  }

  if (capture_prompt_hidden_) {  // LFM2 (feature rows leave before the head)
    if (captured_layer_count > 0) {
      HIP_CHECK(hipMemcpyAsync(h_verification_hidden_.data(), feature_buffer,
                               rows * feature_width * sizeof(float),
                               hipMemcpyDeviceToHost, arena_.stream));
      HIP_CHECK(hipMemcpyAsync(arena_.d_target_layer_features,
                               feature_buffer + (rows - 1) * feature_width,
                               feature_width * sizeof(float),
                               hipMemcpyDeviceToDevice, arena_.stream));
    } else {
      HIP_CHECK(hipMemcpyAsync(
          h_verification_hidden_.data(), scratch.decode.hidden.data(),
          rows * H * sizeof(float), hipMemcpyDeviceToHost, arena_.stream));
    }
  }
  // LFM2: normed already holds output_norm(hidden) of every row.
  LaunchVerifyProjection(
      weights_.output, scratch.decode.normed.data(), logits_buffer, rows,
      vocab_size, H, arena_.stream,
      DecodeSingleIntegerPath(weights_.output, vocab_size, H));
  auto* const d_out_tokens = scratch.decode.prompt_tokens.data();
  LaunchBatchedGPUArgmax(logits_buffer, d_out_tokens, rows, vocab_size,
                         scratch.ffn.out, arena_.stream);
  std::vector<tokenization::TokenId> host_predictions(rows, 0);
  HIP_CHECK(hipMemcpyAsync(host_predictions.data(), d_out_tokens,
                           rows * sizeof(tokenization::TokenId),
                           hipMemcpyDeviceToHost, arena_.stream));
  if (capture_logits) {
    h_verification_logits_.resize(rows * vocab_size);
    HIP_CHECK(hipMemcpyAsync(h_verification_logits_.data(), logits_buffer,
                             rows * vocab_size * sizeof(float),
                             hipMemcpyDeviceToHost, arena_.stream));
  }
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  std::vector<tokenization::TokenId> predictions;
  predictions.reserve(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    predictions.push_back(
        CheckedSampleToken(host_predictions[r], config.vocab_size));
  }
  last_hidden_offset_ = (rows - 1) * H;
  lfm2_rows_verified_ = true;
  return predictions;
}

// Flint: speculative verification for flint, rows 2..8. The projections run
// once for all rows through the verify_m intercepts (shared weight reads);
// the KV write and the attention stay per row and in row order, because each
// row reads the K/V entries the row before it wrote. Every kernel and its
// arguments are the row's exact decode launches (ExecuteFlintDecodeStep), so
// logits are bit-identical to walking the chunk token by token.
std::vector<tokenization::TokenId>
QwenGpuExecutor::ForwardFlintVerificationRows(
    std::span<const tokenization::TokenId> tokens, std::uint32_t start_pos,
    bool capture_logits) {
  CheckReset();
  if (vision_input_.Bytes() != 0) {
    throw std::runtime_error("flint vision is not supported");
  }
  const auto& config = weights_.config;
  if (!config.flint) {
    throw std::runtime_error("flint verify called for a non-flint config");
  }
  if (arena_.d_kv_cache == nullptr) {
    throw std::runtime_error("flint verify requires the fp32 KV cache");
  }
  const std::size_t rows = tokens.size();
  if (start_pos > arena_.GetMaxContext() ||
      rows > arena_.GetMaxContext() - start_pos) {
    throw std::length_error("flint verify exceeds the GPU context length");
  }
  last_verification_rows_ = rows;
  h_verification_hidden_.clear();
  h_verification_logits_.clear();
  replaying_ssm_state_ = false;
  arena_.DisableSsmReplayCapture();
  EnsureVerificationLogits(rows);

  const std::size_t H = config.hidden_size;
  const std::size_t attn =
      static_cast<std::size_t>(config.num_attention_heads) * config.head_dim;
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t I = config.intermediate_size;
  const std::size_t S = config.expert_shared_feed_forward_length;
  const float eps = config.norm_eps_pre;
  const auto scratch = arena_.GetScratchView(rows);
  if (scratch.decode.hidden.size() < rows * H ||
      scratch.decode.normed.size() < rows * H ||
      scratch.attention.q.size() < rows * attn ||
      scratch.attention.k.size() < rows * kv ||
      scratch.attention.v.size() < rows * kv ||
      scratch.ssm.out.size() < rows * std::max(H, attn) ||
      scratch.attention.output.size() < rows * H ||
      scratch.ffn.activation.size() < rows * std::max(I, S) ||
      scratch.ffn.out.size() < rows * H ||
      scratch.decode.prompt_tokens.size() < rows) {
    throw std::runtime_error("flint verify scratch is short");
  }

  const std::size_t vocab_size = config.vocab_size;
  const std::size_t captured_layer_count =
      arena_.GetTargetLayerCapture().size();
  const std::size_t feature_width = captured_layer_count * H;
  const std::size_t feature_elements =
      capture_prompt_hidden_ ? rows * feature_width : 0;
  const std::size_t feature_rows =
      feature_elements / vocab_size + (feature_elements % vocab_size != 0);
  // The head writes the rows' logits into the same buffer that held the
  // drafter features, exactly as the single-session ForwardVerificationBatch
  // does: the features leave for the host and the tap plane before it runs.
  if (capture_prompt_hidden_) {
    EnsureVerificationLogits(std::max(rows, feature_rows));
    h_verification_hidden_.resize(rows * (feature_width ? feature_width : H));
  }
  auto* const feature_buffer = d_verification_logits_;
  auto* const logits_buffer = d_verification_logits_;

  std::vector<std::uint32_t> host_rows(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    host_rows[r] = static_cast<std::uint32_t>(tokens[r]);
  }
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                           host_rows.data(), rows * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, arena_.stream));
  LaunchBatchedEmbeddingLookup(
      weights_.token_embd.data, weights_.token_embd.type,
      scratch.decode.prompt_tokens.data(), scratch.decode.hidden.data(), rows,
      H, arena_.stream);
  // Flint: the decode step norms the embedded rows before layer 0, unlike
  // LFM2. The positions below replace the token ids so every row carries its
  // own d_pos word, as the decode step's pair does.
  if (config.num_layers != 0) {
    for (std::size_t r = 0; r < rows; ++r) {
      LaunchRMSNorm(
          scratch.decode.hidden.data() + r * H,
          static_cast<const float*>(weights_.layers[0].attn_norm.data),
          scratch.decode.normed.data() + r * H, static_cast<std::uint32_t>(H),
          eps, arena_.stream);
    }
  }
  for (std::size_t r = 0; r < rows; ++r) {
    host_rows[r] = start_pos + static_cast<std::uint32_t>(r);
  }
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                           host_rows.data(), rows * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, arena_.stream));
  const std::uint32_t* d_pos = scratch.decode.prompt_tokens.data();

  const std::uint32_t max_ctx = arena_.GetMaxContext();
  const std::uint32_t heads = config.num_attention_heads;
  const std::uint32_t kv_heads = config.num_key_value_heads;
  const std::uint32_t head_dim = config.head_dim;
  const std::uint32_t rotary_dim = config.rotary_dim;
  const float rope_theta = config.rope_theta;
  const std::uint32_t n_expert = config.expert_count;
  const std::uint32_t n_used = config.expert_used_count;
  const std::uint32_t expert_ff = config.expert_feed_forward_length;
  const std::size_t plane = arena_.GetAttentionKvPlaneElements();
  // Flint: fp32 KV only.
  float* v_cache = arena_.d_kv_cache + plane;
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const auto& layer = weights_.layers[l];
    // Flint: with ROCmFP4 q/k/v the group quantises normed once and shares
    // that Q8 copy, the same quant and matvec kernels the decode step's slot
    // path runs; any refusal falls back per tensor to the launches LaunchGEMV
    // serves decode.
    const bool shared_qkv = IsRocmFp4(layer.attn_q.type) &&
                            IsRocmFp4(layer.attn_k.type) &&
                            IsRocmFp4(layer.attn_v.type) &&
                            RocmFp4ShapeOk(attn, H) && RocmFp4ShapeOk(kv, H);
    if (!shared_qkv ||
        !TryLaunchVerifyIntegerGroup(
            {{&layer.attn_q, scratch.attention.q.data(), attn},
             {&layer.attn_k, scratch.attention.k.data(), kv},
             {&layer.attn_v, scratch.attention.v.data(), kv}},
            scratch.decode.normed.data(), rows, H, arena_.stream)) {
      LaunchVerifyProjection(layer.attn_q, scratch.decode.normed.data(),
                             scratch.attention.q.data(), rows, attn, H,
                             arena_.stream,
                             DecodeSingleIntegerPath(layer.attn_q, attn, H));
      LaunchVerifyProjection(layer.attn_k, scratch.decode.normed.data(),
                             scratch.attention.k.data(), rows, kv, H,
                             arena_.stream,
                             DecodeSingleIntegerPath(layer.attn_k, kv, H));
      LaunchVerifyProjection(layer.attn_v, scratch.decode.normed.data(),
                             scratch.attention.v.data(), rows, kv, H,
                             arena_.stream,
                             DecodeSingleIntegerPath(layer.attn_v, kv, H));
    }
    // Row order matters: row r reads the K/V rows the rows before it wrote.
    const float* sinks = static_cast<const float*>(layer.attn_sinks.data);
    for (std::size_t r = 0; r < rows; ++r) {
      float* q = scratch.attention.q.data() + r * attn;
      float* k = scratch.attention.k.data() + r * kv;
      float* v = scratch.attention.v.data() + r * kv;
      const std::uint32_t* row_pos = d_pos + r;
      const std::uint32_t pos = start_pos + static_cast<std::uint32_t>(r);
      LaunchPerHeadRMSNorm(q, static_cast<const float*>(layer.attn_q_norm.data),
                           q, heads, head_dim, eps, arena_.stream);
      LaunchPerHeadRMSNorm(k, static_cast<const float*>(layer.attn_k_norm.data),
                           k, kv_heads, head_dim, eps, arena_.stream);
      if (config.LayerUsesRope(l)) {
        LaunchRoPE(q, k, heads, kv_heads, head_dim, rotary_dim, row_pos,
                   rope_theta, arena_.stream, nullptr);
      }
      Require(LaunchGlimmerWriteKv(k, v, arena_.d_kv_cache, v_cache, l, pos,
                                   max_ctx, kv_heads, head_dim, kv, plane,
                                   arena_.stream));
      if (FlintSplitKEnabled()) {
        const std::uint32_t n_splits =
            config.LayerSwaWindow(l) == 0U ? 32U : 16U;
        Require(LaunchFlintDecodeAttention(
            q, arena_.d_kv_cache, v_cache, scratch.ssm.out.data() + r * attn, l,
            row_pos, config.LayerSwaWindow(l), max_ctx, heads, kv_heads,
            head_dim, n_splits, arena_.d_lfm2_attn_partial, sinks,
            arena_.stream));
      } else {
        Require(LaunchGlimmerWindowAttention(
            q, nullptr, arena_.d_kv_cache, v_cache,
            scratch.ssm.out.data() + r * attn, l, pos, config.LayerSwaWindow(l),
            max_ctx, heads, kv_heads, head_dim, attn, 0, plane,
            scratch.ssm.out.size() - r * attn, arena_.stream, sinks));
      }
    }
    LaunchVerifyProjection(layer.attn_output, scratch.ssm.out.data(),
                           scratch.attention.output.data(), rows, H, attn,
                           arena_.stream,
                           DecodeSingleIntegerPath(layer.attn_output, H, attn));
    for (std::size_t r = 0; r < rows; ++r) {
      Require(LaunchLfm2AddRMSNorm(  // Flint: residual + ffn_norm
          scratch.decode.hidden.data() + r * H,
          scratch.attention.output.data() + r * H,
          static_cast<const float*>(layer.ffn_norm.data),
          scratch.decode.normed.data() + r * H, static_cast<std::uint32_t>(H),
          eps, arena_.stream));
    }
    if (!layer.is_moe) {  // Flint dense layer
      LaunchFfnActivation(layer, scratch, rows, I, H, arena_.stream);
      LaunchVerifyProjection(layer.ffn_down, scratch.ffn.activation.data(),
                             scratch.ffn.out.data(), rows, H, I, arena_.stream,
                             DecodeSingleIntegerPath(layer.ffn_down, H, I));
    } else {  // Flint MoE layer
      // The ungated shared expert stays per row: the decode step's launches.
      for (std::size_t r = 0; r < rows; ++r) {
        LaunchFusedSwiGLUGEMV(
            layer.ffn_gate_shexp.data, layer.ffn_gate_shexp.type,
            layer.ffn_up_shexp.data, layer.ffn_up_shexp.type,
            scratch.decode.normed.data() + r * H,
            scratch.ffn.activation.data() + r * S, S, H, arena_.stream);
        LaunchGEMV(layer.ffn_down_shexp.data, layer.ffn_down_shexp.type,
                   scratch.ffn.activation.data() + r * S,
                   scratch.attention.output.data() + r * H, H, S,
                   arena_.stream);
      }
      for (std::size_t r = 0; r < rows; ++r) {
        Require(LaunchLfm2MoeRoute(
            scratch.decode.normed.data() + r * H,
            static_cast<const float*>(layer.ffn_gate_inp.data),
            static_cast<const float*>(layer.exp_probs_b.data),
            static_cast<std::uint32_t>(H), n_expert, n_used,
            arena_.d_moe_logits + r * n_expert, arena_.d_moe_ids + r * n_used,
            arena_.d_moe_weights + r * n_used, arena_.stream));
      }
      if (superroc_fp4_expert_ffn_dev_batch(
              static_cast<int>(layer.ffn_gate_exps.type),
              layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
              layer.ffn_down_exps.data, scratch.decode.normed.data(),
              arena_.d_moe_ids, arena_.d_moe_weights, static_cast<int>(rows),
              static_cast<int>(n_used), static_cast<int>(n_expert),
              static_cast<int>(H), static_cast<int>(expert_ff), 0.0F,
              scratch.ffn.out.data(), arena_.stream) != SUPERROC_FP4_OK) {
        // Same per-row launches the decode step runs, on the same refusal.
        for (std::size_t r = 0; r < rows; ++r) {
          if (superroc_fp4_expert_ffn_dev(
                  static_cast<int>(layer.ffn_gate_exps.type),
                  layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
                  layer.ffn_down_exps.data,
                  scratch.decode.normed.data() + r * H,
                  arena_.d_moe_ids + r * n_used,
                  arena_.d_moe_weights + r * n_used, static_cast<int>(n_used),
                  static_cast<int>(n_expert), static_cast<int>(H),
                  static_cast<int>(expert_ff), 0.0F,
                  scratch.ffn.out.data() + r * H,
                  arena_.stream) != SUPERROC_FP4_OK) {
            throw std::runtime_error("flint decode experts failed");
          }
        }
      }
      for (std::size_t r = 0; r < rows; ++r) {
        LaunchResidualAdd(scratch.ffn.out.data() + r * H,
                          scratch.attention.output.data() + r * H,
                          scratch.ffn.out.data() + r * H, H, arena_.stream);
      }
    }
    // Flint: residual + the next RMSNorm (next layer's attn_norm, or
    // output_norm after the last layer).
    const float* next_norm =
        l + 1U < config.num_layers
            ? static_cast<const float*>(weights_.layers[l + 1U].attn_norm.data)
            : static_cast<const float*>(weights_.output_norm.data);
    for (std::size_t r = 0; r < rows; ++r) {
      Require(LaunchLfm2AddRMSNorm(
          scratch.decode.hidden.data() + r * H, scratch.ffn.out.data() + r * H,
          next_norm, scratch.decode.normed.data() + r * H,
          static_cast<std::uint32_t>(H), eps, arena_.stream));
    }
    // DSpark tap: the layer output is the hidden state the drafter's
    // feature-fusion encoder consumes (same point as the Qwen paths).
    if (capture_prompt_hidden_) {
      if (const auto tap = arena_.GetTargetLayerCaptureIndex(l);
          tap.has_value()) {
        float* const destination = feature_buffer + (*tap * H);
        HIP_CHECK(hipMemcpy2DAsync(destination, feature_width * sizeof(float),
                                   scratch.decode.hidden.data(),
                                   H * sizeof(float), H * sizeof(float), rows,
                                   hipMemcpyDeviceToDevice, arena_.stream));
      }
    }
  }

  if (capture_prompt_hidden_) {  // Flint (feature rows leave before the head)
    if (captured_layer_count > 0) {
      HIP_CHECK(hipMemcpyAsync(h_verification_hidden_.data(), feature_buffer,
                               rows * feature_width * sizeof(float),
                               hipMemcpyDeviceToHost, arena_.stream));
      HIP_CHECK(hipMemcpyAsync(arena_.d_target_layer_features,
                               feature_buffer + (rows - 1) * feature_width,
                               feature_width * sizeof(float),
                               hipMemcpyDeviceToDevice, arena_.stream));
    } else {
      HIP_CHECK(hipMemcpyAsync(
          h_verification_hidden_.data(), scratch.decode.hidden.data(),
          rows * H * sizeof(float), hipMemcpyDeviceToHost, arena_.stream));
    }
  }
  // Flint: normed already holds output_norm(hidden) of every row.
  LaunchVerifyProjection(
      weights_.output, scratch.decode.normed.data(), logits_buffer, rows,
      vocab_size, H, arena_.stream,
      DecodeSingleIntegerPath(weights_.output, vocab_size, H));
  auto* const d_out_tokens = scratch.decode.prompt_tokens.data();
  LaunchBatchedGPUArgmax(logits_buffer, d_out_tokens, rows, vocab_size,
                         scratch.ffn.out, arena_.stream);
  std::vector<tokenization::TokenId> host_predictions(rows, 0);
  HIP_CHECK(hipMemcpyAsync(host_predictions.data(), d_out_tokens,
                           rows * sizeof(tokenization::TokenId),
                           hipMemcpyDeviceToHost, arena_.stream));
  if (capture_logits) {
    h_verification_logits_.resize(rows * vocab_size);
    HIP_CHECK(hipMemcpyAsync(h_verification_logits_.data(), logits_buffer,
                             rows * vocab_size * sizeof(float),
                             hipMemcpyDeviceToHost, arena_.stream));
  }
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  std::vector<tokenization::TokenId> predictions;
  predictions.reserve(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    predictions.push_back(
        CheckedSampleToken(host_predictions[r], config.vocab_size));
  }
  last_hidden_offset_ = (rows - 1) * H;
  return predictions;
}

}  // namespace superroc::hip
#endif  // defined(ENGINE_ENABLE_HIP)
