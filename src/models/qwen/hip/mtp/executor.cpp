#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/snapshot_transfer.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen/hip/mtp/detail/allocation.hpp"
#include "src/models/qwen/hip/mtp/kat_moe_ffn.hpp"
#include "src/models/qwen/hip/mtp/kat_mtp_pmin.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/core/env_compat.hpp"

namespace superroc::hip {
namespace {

template<typename T>
void AllocateBuffer(T*& pointer, std::size_t elements) {
  pointer = static_cast<T*>(detail::AllocateDevice(elements * sizeof(T)));
}

// lqw:spec5-prof. One speculative draft forward, split with HIP events.
// SUPERROC_MTP_PROFILE=1. Events are recorded on the draft stream and read after
// the step's existing sync, so the spans are GPU time.
struct DraftStepProf {
  bool on{false};
  hipEvent_t ev[8]{};

  DraftStepProf() {
    const char* env = ::superroc::core::GetEnvCompat("SUPERROC_MTP_PROFILE");
    on = env != nullptr && env[0] != '\0' && env[0] != '0';
    if (!on) {
      return;
    }
    for (int i = 0; i < 8; ++i) {
      if (hipEventCreate(&ev[i]) != hipSuccess) {
        on = false;
        return;
      }
    }
  }

  void Mark(int index, hipStream_t stream) const {
    if (on) {
      (void)hipEventRecord(ev[index], stream);
    }
  }

  [[nodiscard]] float Span(int index) const {
    float ms = 0.0F;
    if (on) {
      (void)hipEventElapsedTime(&ms, ev[index], ev[index + 1]);
    }
    return ms;
  }
};

[[nodiscard]] std::size_t DraftWeightBytes(core::GgmlType type,
                                           std::size_t rows,
                                           std::size_t cols) noexcept {
  const std::size_t n = rows * cols;
  switch (type) {
    case core::GgmlType::kF32:
      return n * sizeof(float);
    case core::GgmlType::kBF16:
    case core::GgmlType::kF16:
      return n * sizeof(std::uint16_t);
    case core::GgmlType::kQ4_0:
    case core::GgmlType::kQ4_0_ROCMFP4:
      return (cols / 32) * 18 * rows;
    case core::GgmlType::kQ4_0_ROCMFP4_FAST:
      return (cols / 32) * 17 * rows;
    case core::GgmlType::kQ8_0:
      return (cols / 32) * 34 * rows;
    // Super ROC DVT: K-quant and IQ row sizes, the same block sizes
    // quant_ops.hpp static_asserts (Q6_K is a common LM head type).
    case core::GgmlType::kQ8_K:
      return (cols / 256) * 292 * rows;
    case core::GgmlType::kQ6_K:
      return (cols / 256) * 210 * rows;
    case core::GgmlType::kQ5_K:
      return (cols / 256) * 176 * rows;
    case core::GgmlType::kQ4_K:
      return (cols / 256) * 144 * rows;
    case core::GgmlType::kQ3_K:
    case core::GgmlType::kIQ3_S:
      return (cols / 256) * 110 * rows;
    case core::GgmlType::kIQ4_XS:
      return (cols / 256) * 136 * rows;
    case core::GgmlType::kIQ4_NL:
      return (cols / 32) * 18 * rows;
    default:
      return 0;
  }
}

}  // namespace

QwenMtpGpuExecutor::QwenMtpGpuExecutor(
    std::shared_ptr<const QwenMtpGpuModel> model, std::uint32_t max_context)
    : model_(std::move(model)),
      max_context_(max_context),
      h_last_hidden_(model_->GetConfig().hidden_size),
      h_logits_(model_->GetConfig().vocab_size) {
  try {
    Allocate();
    ConfigureDraftVocab();
    Reset();
  } catch (...) {
    Free();
    throw;
  }
}

// SQ10 SPEC2 (R3): declared locally (not via the rocm_fp4 header) so this
// TU's include block stays untouched.
extern "C" int superroc_fp4_slot_stream_detach(hipStream_t stream);

QwenMtpGpuExecutor::~QwenMtpGpuExecutor() {
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
    // Remove this helper stream's alias (and any set it owns) before
    // Free() destroys the stream.
    (void)superroc_fp4_slot_stream_detach(stream_);
  }
  Free();
}

// Super ROC DVT. The head is row-major (row r at r * row_bytes for every type
// DraftWeightBytes knows), so a row prefix and a row tail are themselves
// valid matrices for the same GEMV. Ids below N cover 99.8 % of English,
// C++, Python and Markdown tokens at N = 98304 on the Qwen3.5 vocab (Jay's
// count, 602,738 tokens); the special tokens (chat, think, tool call) sit at
// the top of the vocab, so the tail keeps all of them.
void QwenMtpGpuExecutor::ConfigureDraftVocab() {
  draft_head_rows_ = 0;
  draft_tail_start_ = 0;
  draft_rows_ = 0;
  // Default N = 98304 (lane DVT, KAT on MAX1: draft counts unchanged, decode
  // +8-11 %). SUPERROC_MTP_DRAFT_VOCAB=0 keeps the full head; another N trims to N.
  const char* env = ::superroc::core::GetEnvCompat("SUPERROC_MTP_DRAFT_VOCAB");
  const bool env_set = env != nullptr && env[0] != '\0';
  const auto& weights = model_->GetWeights();
  const std::size_t vocab = weights.config.vocab_size;
  const std::size_t hidden = weights.config.hidden_size;
  // HT2 (Muse Wonder W2 DVT-1): the default was measured on the Qwen3.5 BPE
  // vocab of 248,320 tokens only (frequent tokens at low ids); any other vocab
  // trims only when SUPERROC_MTP_DRAFT_VOCAB asks for it.
  if (!env_set && vocab != 248320) {
    std::fprintf(stderr,
                 "MTP draft vocab: full head (vocab %zu; the default trim is "
                 "measured on the 248320 Qwen3.5 vocab only)\n",
                 vocab);
    return;
  }
  // HT2 (W2 DVT-2, W6 D-2): p-min and the TF dump read the draft
  // distribution, so either keeps the full head, and wins over an explicit
  // SUPERROC_MTP_DRAFT_VOCAB. Same parser as the p-min gate (kat_mtp_pmin.hpp).
  const bool tf_dump = ::superroc::core::GetEnvCompat("SUPERROC_MTP_TF_DUMP") != nullptr;
  if (MtpPMinFromEnv() > 0.0F || tf_dump) {
    std::fprintf(
        stderr,
        "MTP draft vocab: full head (%s reads the draft distribution)\n",
        tf_dump ? "SUPERROC_MTP_TF_DUMP" : "SUPERROC_MTP_PMIN");
    return;
  }
  long requested = 98304L;
  if (env_set) {
    char* parsed_end = nullptr;
    requested = std::strtol(env, &parsed_end, 10);
    if (parsed_end == env || *parsed_end != '\0') {
      std::fprintf(stderr,
                   "MTP draft vocab: SUPERROC_MTP_DRAFT_VOCAB='%s' is not a whole "
                   "number; full head\n",
                   env);
      return;
    }
  }
  const std::size_t row_bytes =
      DraftWeightBytes(weights.output.type, 1, hidden);
  if (requested <= 0 || static_cast<std::size_t>(requested) >= vocab) {
    return;
  }
  // HT2 (W2 DVT-3): a whole number of blocks per row for every type the row
  // size table knows (256 covers the K/IQ super-blocks and the 32-blocks).
  const char* refused = nullptr;
  if (requested < 1024) {
    refused = "below 1024 rows";
  } else if (row_bytes == 0 || weights.output.data == nullptr) {
    refused = "the LM head type has no row size here";
  } else if ((hidden % 256) != 0) {
    refused = "hidden size is not a multiple of 256";
  }
  if (refused != nullptr) {
    std::fprintf(stderr, "MTP draft vocab: %s %ld refused (%s); full head\n",
                 env_set ? "SUPERROC_MTP_DRAFT_VOCAB" : "default", requested,
                 refused);
    return;
  }
  // Rows in multiples of 256 (a whole number of GEMV row groups).
  const std::size_t head =
      (static_cast<std::size_t>(requested) + 255) / 256 * 256;
  std::size_t tail = vocab;
  for (const auto& entry :
       model_->GetTargetModel().GetTokenizer().SpecialTokens()) {
    const auto id = static_cast<std::size_t>(entry.second);
    if (id >= head && id < vocab) {
      tail = std::min(tail, id / 256 * 256);
    }
  }
  if (tail < head) {
    tail = head;
  }
  if (head >= vocab || head + (vocab - tail) >= vocab) {
    return;  // nothing saved
  }
  draft_head_rows_ = static_cast<std::uint32_t>(head);
  draft_tail_start_ = static_cast<std::uint32_t>(tail);
  draft_rows_ = static_cast<std::uint32_t>(head + (vocab - tail));
  draft_row_bytes_ = row_bytes;
  h_draft_logits_.assign(draft_rows_, 0.0F);
  std::fprintf(
      stderr,
      "MTP draft vocab: rows [0, %zu) + [%zu, %zu) = %u of %zu "
      "(%.1f %%)\n",
      head, tail, vocab, draft_rows_, vocab,
      100.0 * static_cast<double>(draft_rows_) / static_cast<double>(vocab));
}

std::unique_ptr<QwenMtpGpuExecutor> QwenMtpGpuExecutor::Create(
    std::shared_ptr<const QwenMtpGpuModel> model, std::uint32_t max_context,
    std::string* error_msg) {
  if (model == nullptr || max_context == 0 ||
      max_context > model->GetConfig().context_length) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GPU executor context is invalid";
    }
    return nullptr;
  }
  try {
    return std::unique_ptr<QwenMtpGpuExecutor>(
        new QwenMtpGpuExecutor(std::move(model), max_context));
  } catch (const std::exception& exception) {
    if (error_msg != nullptr) {
      *error_msg = exception.what();
    }
    return nullptr;
  }
}

void QwenMtpGpuExecutor::Allocate() {
  const auto& config = model_->GetConfig();
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context_ *
      config.head_dim;

  const auto stream_error = hipStreamCreate(&stream_);
  if (stream_error != hipSuccess) {
    throw std::runtime_error(std::string("MTP stream creation failed: ") +
                             hipGetErrorString(stream_error));
  }
  HIPBLAS_CHECK(hipblasCreate(&hipblas_));
  HIPBLAS_CHECK(hipblasSetStream(hipblas_, stream_));
  AllocateBuffer(d_target_hidden_, hidden);
  AllocateBuffer(d_embedding_, hidden);
  AllocateBuffer(d_fusion_, 2 * hidden);
  AllocateBuffer(d_hidden_, hidden);
  AllocateBuffer(d_normed_, hidden);
  AllocateBuffer(d_qg_, 2 * attention);
  AllocateBuffer(d_q_, attention);
  AllocateBuffer(d_k_, kv);
  AllocateBuffer(d_v_, kv);
  AllocateBuffer(d_gate_, attention);
  AllocateBuffer(d_context_, attention);
  AllocateBuffer(d_attn_out_, hidden);
  AllocateBuffer(d_ffn_act_, intermediate);
  AllocateBuffer(d_ffn_out_, hidden);
  AllocateBuffer(d_feedback_hidden_, hidden);
  AllocateBuffer(d_logits_, config.vocab_size);
  AllocateBuffer(d_kv_cache_, 2 * total_kv);
  d_kv_cache_f16_ =
      detail::AllocateDevice(2 * total_kv * sizeof(std::uint16_t));
  const std::size_t split_k_elements = detail::DecodeAttentionScratchElements(
      config.num_attention_heads, config.head_dim);
  AllocateBuffer(d_split_k_scratch_, split_k_elements);
  AllocateBuffer(d_out_token_, 1);
  // RV44 cosmetic: the draft-owned MoE scratch below is per-step scratch too,
  // so it is counted in scratch_bytes_ at the end of this function.
  std::size_t moe_scratch_bytes = 0;
  if (model_->GetWeights().layer.is_moe) {
    // KAT-MTP: one-token MoE scratch. The dense SwiGLU buffers stay allocated
    // from intermediate_size; this block is what the FFN actually runs.
    const std::size_t n_expert = config.expert_count;
    const std::size_t n_used = config.expert_used_count;
    const std::size_t ff = std::max(config.expert_feed_forward_length,
                                    config.expert_shared_feed_forward_length);
    if (n_expert == 0 || n_used == 0 || n_used > 8 || ff == 0) {
      throw std::runtime_error("KAT-MTP: MoE shape is invalid");
    }
    AllocateBuffer(d_moe_logits_, n_expert);
    AllocateBuffer(d_moe_ids_, n_used);
    AllocateBuffer(d_moe_w_, n_used);
    AllocateBuffer(d_moe_sig_, 1);
    AllocateBuffer(d_moe_gate_, ff);
    AllocateBuffer(d_moe_up_, ff);
    AllocateBuffer(d_moe_act_, ff);
    AllocateBuffer(d_moe_h_, hidden);
    // d_moe_logits_ float[n_expert], d_moe_ids_ int[n_used], d_moe_w_
    // float[n_used], d_moe_sig_ float[1], d_moe_{gate,up,act}_ float[ff],
    // d_moe_h_ float[hidden].
    moe_scratch_bytes =
        (n_expert + n_used + 1 + 3 * ff + hidden) * sizeof(float) +
        n_used * sizeof(int);
  }

  const std::size_t rows = kPrimeBatchRows;
  AllocateBuffer(d_b_tokens_, rows);
  AllocateBuffer(d_b_h_, rows * hidden);
  AllocateBuffer(d_b_embed_, rows * hidden);
  AllocateBuffer(d_b_fusion_, rows * 2 * hidden);
  AllocateBuffer(d_b_hidden_, rows * hidden);
  AllocateBuffer(d_b_normed_, rows * hidden);
  AllocateBuffer(d_b_qg_, rows * 2 * attention);
  AllocateBuffer(d_b_q_, rows * attention);
  AllocateBuffer(d_b_gate_, rows * attention);
  AllocateBuffer(d_b_k_, rows * kv);
  AllocateBuffer(d_b_v_, rows * kv);
  AllocateBuffer(d_b_context_, rows * attention);
  AllocateBuffer(d_b_attn_out_, rows * hidden);
  AllocateBuffer(d_b_ffn_act_, rows * intermediate);
  AllocateBuffer(d_b_ffn_up_, rows * intermediate);
  AllocateBuffer(d_b_ffn_out_, rows * hidden);
  const std::size_t widest =
      std::max(std::max(2 * hidden, intermediate), attention);
  b_bf16_elems_ = rows * widest;
  AllocateBuffer(d_b_bf16_, b_bf16_elems_);
  // Largest prime weight is an FFN matrix (intermediate x hidden).
  q4_bf16_elems_ = intermediate * hidden;
  AllocateBuffer(d_q4_bf16_, q4_bf16_elems_);

  // Account what Allocate() just reserved, in the same terms the serving layer
  // reports: the two context-scaling KV copies are retained state, everything
  // else is per-step scratch that is reused on every drafted token.
  state_bytes_ = 2 * total_kv * (sizeof(float) + sizeof(std::uint16_t));
  scratch_bytes_ =
      (5 * hidden + 2 * hidden + 2 * attention + 3 * attention + 2 * kv +
       2 * hidden + intermediate + config.vocab_size + 1) *
          sizeof(float) +
      split_k_elements * sizeof(float) + moe_scratch_bytes;
}

void QwenMtpGpuExecutor::Free() noexcept {
  const auto free_buffer = [](auto*& pointer) {
    if (pointer != nullptr) {
      (void)hipFree(pointer);
      pointer = nullptr;
    }
  };
  free_buffer(d_target_hidden_);
  free_buffer(d_embedding_);
  free_buffer(d_fusion_);
  free_buffer(d_hidden_);
  free_buffer(d_normed_);
  free_buffer(d_qg_);
  free_buffer(d_q_);
  free_buffer(d_k_);
  free_buffer(d_v_);
  free_buffer(d_gate_);
  free_buffer(d_context_);
  free_buffer(d_attn_out_);
  free_buffer(d_ffn_act_);
  free_buffer(d_ffn_out_);
  free_buffer(d_feedback_hidden_);
  free_buffer(d_logits_);
  free_buffer(d_kv_cache_);
  free_buffer(d_kv_cache_f16_);
  free_buffer(d_split_k_scratch_);
  free_buffer(d_out_token_);
  free_buffer(d_moe_logits_);
  free_buffer(d_moe_ids_);
  free_buffer(d_moe_w_);
  free_buffer(d_moe_sig_);
  free_buffer(d_moe_gate_);
  free_buffer(d_moe_up_);
  free_buffer(d_moe_act_);
  free_buffer(d_moe_h_);
  free_buffer(d_b_tokens_);
  free_buffer(d_b_h_);
  free_buffer(d_b_embed_);
  free_buffer(d_b_fusion_);
  free_buffer(d_b_hidden_);
  free_buffer(d_b_normed_);
  free_buffer(d_b_qg_);
  free_buffer(d_b_q_);
  free_buffer(d_b_gate_);
  free_buffer(d_b_k_);
  free_buffer(d_b_v_);
  free_buffer(d_b_context_);
  free_buffer(d_b_attn_out_);
  free_buffer(d_b_ffn_act_);
  free_buffer(d_b_ffn_up_);
  free_buffer(d_b_ffn_out_);
  free_buffer(d_b_bf16_);
  b_bf16_elems_ = 0;
  free_buffer(d_q4_bf16_);
  q4_bf16_elems_ = 0;
  if (hipblas_ != nullptr) {
    (void)hipblasDestroy(hipblas_);
    hipblas_ = nullptr;
  }
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

void QwenMtpGpuExecutor::Reset() noexcept {
  // Attention only reads the valid prefix; new rows overwrite stale KV.
  next_position_ = 0;
  f16_prefix_packed_ = 0;
}

void QwenMtpGpuExecutor::Rewind(std::uint32_t position) {
  if (position > next_position_) {
    throw std::out_of_range("MTP rewind position exceeds executed context");
  }
  next_position_ = position;
  if (f16_prefix_packed_ > position) {
    f16_prefix_packed_ = position;
  }
}

void QwenMtpGpuExecutor::Seek(std::uint32_t position) {
  if (position > max_context_) {
    throw std::out_of_range("MTP seek position exceeds the context");
  }
  next_position_ = position;
}

void QwenMtpGpuExecutor::Sync() {
  if (stream_ == nullptr) {
    return;
  }
  if (hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP stream synchronization failed");
  }
}

tokenization::TokenId QwenMtpGpuExecutor::ForwardTargetHidden(
    tokenization::TokenId input_token, std::span<const float> target_hidden,
    std::uint32_t position, bool compute_logits) {
  if (target_hidden.size() != model_->GetConfig().hidden_size) {
    throw std::invalid_argument("MTP target hidden width is invalid");
  }
  const auto copy_error = hipMemcpyAsync(d_target_hidden_, target_hidden.data(),
                                         target_hidden.size_bytes(),
                                         hipMemcpyHostToDevice, stream_);
  if (copy_error != hipSuccess) {
    throw std::runtime_error(std::string("MTP hidden copy failed: ") +
                             hipGetErrorString(copy_error));
  }
  return Run(input_token, d_target_hidden_, position, compute_logits);
}

tokenization::TokenId QwenMtpGpuExecutor::ForwardFeedback(
    tokenization::TokenId input_token, std::uint32_t position,
    bool compute_logits) {
  return Run(input_token, d_feedback_hidden_, position, compute_logits);
}

tokenization::TokenId QwenMtpGpuExecutor::Run(tokenization::TokenId input_token,
                                              const float* hidden_input,
                                              std::uint32_t position,
                                              bool compute_logits) {
  if (position != next_position_ || position >= max_context_) {
    throw std::out_of_range("MTP GPU position is not sequential");
  }
  const auto& weights = model_->GetWeights();
  const auto& config = weights.config;
  const auto& layer = weights.layer;
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t total_kv =
      static_cast<std::size_t>(config.num_key_value_heads) * max_context_ *
      config.head_dim;
  static DraftStepProf prof;
  prof.Mark(0, stream_);

  LaunchEmbeddingLookup(weights.token_embedding.data,
                        weights.token_embedding.type, input_token, d_embedding_,
                        hidden, stream_);
  if (vision_input_)
    vision_input_->Inject(d_embedding_, position + 1, 1, hidden, 1, stream_);
  LaunchRMSNorm(d_embedding_,
                static_cast<const float*>(weights.embedding_norm.data),
                d_fusion_, hidden, 1.0e-6F, stream_);
  LaunchRMSNorm(hidden_input,
                static_cast<const float*>(weights.hidden_norm.data),
                d_fusion_ + hidden, hidden, 1.0e-6F, stream_);
  LaunchGEMV(weights.fusion_projection.data, weights.fusion_projection.type,
             d_fusion_, d_hidden_, hidden, 2 * hidden, stream_,
             models::qwen::QwenGemmMode::kHipMtp);
  prof.Mark(1, stream_);

  LaunchRMSNorm(d_hidden_, static_cast<const float*>(layer.attn_norm.data),
                d_normed_, hidden, 1.0e-6F, stream_);
  LaunchFusedQKVProjections(
      layer.attn_q.data, layer.attn_q.type, layer.attn_k.data,
      layer.attn_k.type, layer.attn_v.data, layer.attn_v.type, d_normed_, d_qg_,
      d_k_, d_v_, 2 * attention, kv, hidden, stream_);
  LaunchUnpackQG(d_qg_, d_q_, d_gate_, config.num_attention_heads,
                 config.head_dim, stream_);
  LaunchPerHeadRMSNorm(d_q_, static_cast<const float*>(layer.attn_q_norm.data),
                       d_q_, config.num_attention_heads, config.head_dim,
                       1.0e-6F, stream_);
  LaunchPerHeadRMSNorm(d_k_, static_cast<const float*>(layer.attn_k_norm.data),
                       d_k_, config.num_key_value_heads, config.head_dim,
                       1.0e-6F, stream_);
  LaunchRoPE(d_q_, d_k_, config.num_attention_heads, config.num_key_value_heads,
             config.head_dim, config.rotary_dim, position, config.rope_theta,
             stream_, vision_input_ ? vision_input_->rope() : nullptr);
  LaunchAttention(
      d_q_, d_k_, d_v_, d_gate_, d_kv_cache_, d_kv_cache_ + total_kv,
      d_kv_cache_f16_, static_cast<std::uint16_t*>(d_kv_cache_f16_) + total_kv,
      d_context_, 0, position, max_context_, config.num_attention_heads,
      config.num_key_value_heads, config.head_dim, stream_, d_split_k_scratch_);
  LaunchGEMV(layer.attn_output.data, layer.attn_output.type, d_context_,
             d_attn_out_, hidden, attention, stream_,
             models::qwen::QwenGemmMode::kHipMtp);
  LaunchResidualAdd(d_hidden_, d_attn_out_, d_hidden_, hidden, stream_);
  prof.Mark(2, stream_);

  LaunchRMSNorm(d_hidden_, static_cast<const float*>(layer.ffn_norm.data),
                d_normed_, hidden, 1.0e-6F, stream_);
  if (layer.is_moe) {
    // KAT-MTP: routed experts + sigmoid-gated shared expert. The dense
    // SwiGLU weights are empty on this block.
    KatMtpMoeScratch scratch;
    scratch.logits = d_moe_logits_;
    scratch.ids = d_moe_ids_;
    scratch.weights = d_moe_w_;
    scratch.sig = d_moe_sig_;
    scratch.gate = d_moe_gate_;
    scratch.up = d_moe_up_;
    scratch.act = d_moe_act_;
    scratch.hidden = d_moe_h_;
    LaunchKatMtpMoeFfn(layer, config, d_normed_, d_ffn_out_, scratch, stream_);
  } else {
    LaunchFusedSwiGLUGEMV(layer.ffn_gate.data, layer.ffn_gate.type,
                          layer.ffn_up.data, layer.ffn_up.type, d_normed_,
                          d_ffn_act_, config.intermediate_size, hidden,
                          stream_);
    LaunchGEMV(layer.ffn_down.data, layer.ffn_down.type, d_ffn_act_, d_ffn_out_,
               hidden, config.intermediate_size, stream_,
               models::qwen::QwenGemmMode::kHipMtp);
  }
  LaunchResidualAdd(d_hidden_, d_ffn_out_, d_hidden_, hidden, stream_);
  prof.Mark(3, stream_);
  LaunchRMSNorm(d_hidden_,
                static_cast<const float*>(weights.shared_head_norm.data),
                d_feedback_hidden_, hidden, 1.0e-6F, stream_);
  prof.Mark(4, stream_);
  ++next_position_;

  // SQ10 SPEC3 (S3-1): the drafter stream is aliased to its session's FP4
  // slot set, so no entry may return with work still queued: drain here too,
  // not just on the logits path below. The logits path is unchanged (it
  // already pays exactly one sync); this adds one sync per no-logits draft
  // forward (prime/replay), zero per proposed token.
  if (!compute_logits) {
    if (hipStreamSynchronize(stream_) != hipSuccess) {
      throw std::runtime_error("MTP GPU result synchronization failed");
    }
    return 0;
  }
  if (draft_rows_ == 0) {
    LaunchGEMV(weights.output.data, weights.output.type, d_feedback_hidden_,
               d_logits_, config.vocab_size, hidden, stream_,
               models::qwen::QwenGemmMode::kHipMtp);
  } else {
    LaunchGEMV(weights.output.data, weights.output.type, d_feedback_hidden_,
               d_logits_, draft_head_rows_, hidden, stream_,
               models::qwen::QwenGemmMode::kHipMtp);
    if (draft_tail_start_ < config.vocab_size) {
      LaunchGEMV(
          static_cast<const std::uint8_t*>(weights.output.data) +
              static_cast<std::size_t>(draft_tail_start_) * draft_row_bytes_,
          weights.output.type, d_feedback_hidden_, d_logits_ + draft_head_rows_,
          config.vocab_size - draft_tail_start_, hidden, stream_,
          models::qwen::QwenGemmMode::kHipMtp);
    }
  }
  prof.Mark(5, stream_);
  // QSM: the target decode's two-phase tiled argmax (same answer as the
  // one-block scan: lowest index of the largest finite logit). d_ffn_out_
  // (hidden floats) was consumed by the residual add and holds the partials.
  LaunchBatchedGPUArgmax(d_logits_, d_out_token_, 1,
                         draft_rows_ == 0 ? config.vocab_size : draft_rows_,
                         std::span<float>(d_ffn_out_, hidden), stream_);
  tokenization::TokenId result = 0;
  const auto copy_error = hipMemcpyAsync(&result, d_out_token_, sizeof(result),
                                         hipMemcpyDeviceToHost, stream_);
  prof.Mark(6, stream_);
  if (copy_error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP GPU result synchronization failed");
  }
  if (draft_rows_ != 0 && result >= draft_head_rows_) {
    result = static_cast<tokenization::TokenId>(draft_tail_start_ +
                                                (result - draft_head_rows_));
  }
  if (prof.on) {
    const float eh = prof.Span(0);
    const float attn = prof.Span(1);
    const float ffn = prof.Span(2);
    const float hnorm = prof.Span(3);
    const float head = prof.Span(4);
    const float sample = prof.Span(5);
    const std::size_t eh_b =
        DraftWeightBytes(weights.fusion_projection.type, hidden, 2 * hidden);
    const std::size_t attn_b =
        DraftWeightBytes(layer.attn_q.type, 2 * attention, hidden) +
        DraftWeightBytes(layer.attn_k.type, kv, hidden) +
        DraftWeightBytes(layer.attn_v.type, kv, hidden) +
        DraftWeightBytes(layer.attn_output.type, hidden, attention);
    const std::size_t ffn_b =
        DraftWeightBytes(layer.ffn_gate.type, config.intermediate_size,
                         hidden) +
        DraftWeightBytes(layer.ffn_up.type, config.intermediate_size, hidden) +
        DraftWeightBytes(layer.ffn_down.type, hidden, config.intermediate_size);
    const std::size_t head_b = DraftWeightBytes(
        weights.output.type, draft_rows_ == 0 ? config.vocab_size : draft_rows_,
        hidden);
    const auto gbps = [](std::size_t bytes, float ms) {
      return ms > 0.0F ? (static_cast<double>(bytes) /
                          (static_cast<double>(ms) * 1.0e6))
                       : 0.0;
    };
    std::fprintf(
        stderr,
        "DRAFT_PROF pos=%u eh_ms=%.3f attn_ms=%.3f ffn_ms=%.3f "
        "hnorm_ms=%.3f head_ms=%.3f sample_ms=%.3f eh_B=%zu attn_B=%zu "
        "ffn_B=%zu head_B=%zu eh_GBps=%.1f attn_GBps=%.1f "
        "ffn_GBps=%.1f head_GBps=%.1f\n",
        position, eh, attn, ffn, hnorm, head, sample, eh_b, attn_b, ffn_b,
        head_b, gbps(eh_b, eh), gbps(attn_b, attn), gbps(ffn_b, ffn),
        gbps(head_b, head));
  }
  return CheckedSampleToken(result, config.vocab_size);
}

std::span<const float> QwenMtpGpuExecutor::CopyLastHidden() {
  const auto error = hipMemcpyAsync(h_last_hidden_.data(), d_feedback_hidden_,
                                    h_last_hidden_.size() * sizeof(float),
                                    hipMemcpyDeviceToHost, stream_);
  if (error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP hidden synchronization failed");
  }
  return h_last_hidden_;
}

std::span<const float> QwenMtpGpuExecutor::CopyLastLogits() {
  if (draft_rows_ != 0) {
    // DVT: unpack head + tail into full-vocab order; rows the draft did not
    // score read -inf, so callers indexing by token id stay correct.
    const auto error = hipMemcpyAsync(h_draft_logits_.data(), d_logits_,
                                      h_draft_logits_.size() * sizeof(float),
                                      hipMemcpyDeviceToHost, stream_);
    if (error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
      throw std::runtime_error("MTP logit synchronization failed");
    }
    std::fill(h_logits_.begin(), h_logits_.end(),
              -std::numeric_limits<float>::infinity());
    std::copy(h_draft_logits_.begin(),
              h_draft_logits_.begin() + draft_head_rows_, h_logits_.begin());
    std::copy(h_draft_logits_.begin() + draft_head_rows_, h_draft_logits_.end(),
              h_logits_.begin() + draft_tail_start_);
    return h_logits_;
  }
  const auto error = hipMemcpyAsync(h_logits_.data(), d_logits_,
                                    h_logits_.size() * sizeof(float),
                                    hipMemcpyDeviceToHost, stream_);
  if (error != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("MTP logit synchronization failed");
  }
  return h_logits_;
}

namespace {

std::size_t MtpCheckedMultiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("MTP snapshot size overflows");
  }
  return left * right;
}

std::size_t MtpCheckedSum(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("MTP snapshot size overflows");
  }
  return left + right;
}

}  // namespace

namespace {

constexpr std::array<std::uint8_t, 8> kMtpGpuPersistentMagic = {
    'G', 'M', 'T', 'P', 'G', 'P', '0', '1'};
constexpr std::uint32_t kMtpGpuPersistentVersion = 1;
constexpr std::size_t kMtpGpuPersistentHeaderBytes = 64;

template<typename T>
  requires(std::is_unsigned_v<T>)
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     T value) {
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    throw std::length_error("MTP GPU persistent header is truncated");
  }
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename T>
  requires(std::is_unsigned_v<T>)
[[nodiscard]] T GetLittleEndian(std::span<const std::uint8_t> source,
                                std::size_t offset) {
  if (offset > source.size() || sizeof(T) > source.size() - offset) {
    throw std::invalid_argument("MTP GPU persistent header is truncated");
  }
  T value = 0;
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    value |= static_cast<T>(source[offset + byte]) << (byte * 8U);
  }
  return value;
}

[[nodiscard]] std::size_t PersistentSizeFromU64(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("MTP GPU persistent size overflows");
  }
  return static_cast<std::size_t>(value);
}

}  // namespace

QwenMtpGpuSnapshot::~QwenMtpGpuSnapshot() {
  if (d_k_f32_ != nullptr) {
    (void)hipFree(d_k_f32_);
  }
  if (d_v_f32_ != nullptr) {
    (void)hipFree(d_v_f32_);
  }
  if (d_k_f16_ != nullptr) {
    (void)hipFree(d_k_f16_);
  }
  if (d_v_f16_ != nullptr) {
    (void)hipFree(d_v_f16_);
  }
}

std::size_t QwenMtpGpuExecutor::SnapshotPayloadBytes() const {
  const auto& config = model_->GetConfig();
  if (next_position_ > max_context_ || config.num_key_value_heads == 0 ||
      config.head_dim == 0) {
    throw std::logic_error("MTP snapshot boundary is invalid");
  }
  const std::size_t kv =
      MtpCheckedMultiply(config.num_key_value_heads, config.head_dim);
  const std::size_t rows = MtpCheckedMultiply(next_position_, kv);
  return MtpCheckedSum(MtpCheckedMultiply(rows, 2 * sizeof(float)),
                       MtpCheckedMultiply(rows, 2 * sizeof(std::uint16_t)));
}

std::unique_ptr<QwenMtpGpuSnapshot> QwenMtpGpuExecutor::SaveSnapshot() const {
  const auto& config = model_->GetConfig();
  const std::uint32_t valid = next_position_;
  const std::uint32_t heads = config.num_key_value_heads;
  const std::uint32_t dim = config.head_dim;
  if (valid > max_context_ || heads == 0 || dim == 0) {
    throw std::logic_error("MTP snapshot boundary is invalid");
  }
  const std::size_t kv = MtpCheckedMultiply(heads, dim);
  const std::size_t total_kv =
      MtpCheckedMultiply(MtpCheckedMultiply(heads, max_context_), dim);

  auto snapshot = std::unique_ptr<QwenMtpGpuSnapshot>(new QwenMtpGpuSnapshot());
  snapshot->num_kv_heads_ = heads;
  snapshot->head_dim_ = dim;
  snapshot->max_context_ = max_context_;
  snapshot->valid_context_ = valid;
  snapshot->payload_bytes_ = SnapshotPayloadBytes();
  if (valid == 0) {
    return snapshot;
  }
  // Each buffer is assigned before the next allocation, so a failed hipMalloc
  // frees what succeeded through the snapshot dtor (a capture skip, not a
  // leak). Copies run after all four allocations succeed.
  const std::size_t f32_plane =
      MtpCheckedMultiply(MtpCheckedMultiply(valid, kv), sizeof(float));
  const std::size_t f16_plane =
      MtpCheckedMultiply(MtpCheckedMultiply(valid, kv), sizeof(std::uint16_t));
  snapshot->d_k_f32_ = detail::AllocateDevice(f32_plane);
  snapshot->d_v_f32_ = detail::AllocateDevice(f32_plane);
  snapshot->d_k_f16_ = detail::AllocateDevice(f16_plane);
  snapshot->d_v_f16_ = detail::AllocateDevice(f16_plane);

  // FP32 planes are head-major [head][pos][dim]; FP16 planes are token-major
  // [pos][kv_width] (attention_decode.hip WriteKVCacheKernel).
  const std::size_t row_width =
      MtpCheckedMultiply(MtpCheckedMultiply(valid, dim), sizeof(float));
  const std::size_t src_pitch =
      MtpCheckedMultiply(MtpCheckedMultiply(max_context_, dim), sizeof(float));
  const auto* src_k = static_cast<const float*>(d_kv_cache_);
  const auto* src_v = static_cast<const float*>(d_kv_cache_) + total_kv;
  HIP_CHECK(hipMemcpy2DAsync(snapshot->d_k_f32_, row_width, src_k, src_pitch,
                             row_width, heads, hipMemcpyDeviceToDevice,
                             stream_));
  HIP_CHECK(hipMemcpy2DAsync(snapshot->d_v_f32_, row_width, src_v, src_pitch,
                             row_width, heads, hipMemcpyDeviceToDevice,
                             stream_));
  const auto* src_k16 = static_cast<const std::uint8_t*>(d_kv_cache_f16_);
  HIP_CHECK(hipMemcpyAsync(snapshot->d_k_f16_, src_k16, f16_plane,
                           hipMemcpyDeviceToDevice, stream_));
  HIP_CHECK(hipMemcpyAsync(
      snapshot->d_v_f16_,
      src_k16 + MtpCheckedMultiply(total_kv, sizeof(std::uint16_t)), f16_plane,
      hipMemcpyDeviceToDevice, stream_));
  HIP_CHECK(hipStreamSynchronize(stream_));
  return snapshot;
}

void QwenMtpGpuExecutor::RestoreSnapshot(const QwenMtpGpuSnapshot& snapshot) {
  const auto& config = model_->GetConfig();
  const std::uint32_t heads = config.num_key_value_heads;
  const std::uint32_t dim = config.head_dim;
  const std::uint32_t valid = snapshot.valid_context_;
  if (snapshot.num_kv_heads_ != heads || snapshot.head_dim_ != dim ||
      snapshot.max_context_ != max_context_ || valid > max_context_) {
    throw std::invalid_argument(
        "MTP snapshot is incompatible with the executor");
  }
  const std::size_t kv = MtpCheckedMultiply(heads, dim);
  const std::size_t rows = MtpCheckedMultiply(valid, kv);
  const std::size_t expected =
      MtpCheckedSum(MtpCheckedMultiply(rows, 2 * sizeof(float)),
                    MtpCheckedMultiply(rows, 2 * sizeof(std::uint16_t)));
  if (snapshot.payload_bytes_ != expected) {
    throw std::invalid_argument("MTP snapshot payload size is inconsistent");
  }
  if (valid == 0) {
    next_position_ = 0;
    f16_prefix_packed_ = 0;
    return;
  }
  if (snapshot.d_k_f32_ == nullptr || snapshot.d_v_f32_ == nullptr ||
      snapshot.d_k_f16_ == nullptr || snapshot.d_v_f16_ == nullptr) {
    throw std::invalid_argument("MTP snapshot payload is incomplete");
  }
  const std::size_t total_kv =
      MtpCheckedMultiply(MtpCheckedMultiply(heads, max_context_), dim);
  const std::size_t f16_plane =
      MtpCheckedMultiply(MtpCheckedMultiply(valid, kv), sizeof(std::uint16_t));
  const std::size_t row_width =
      MtpCheckedMultiply(MtpCheckedMultiply(valid, dim), sizeof(float));
  const std::size_t dst_pitch =
      MtpCheckedMultiply(MtpCheckedMultiply(max_context_, dim), sizeof(float));
  auto* dst_k = static_cast<float*>(d_kv_cache_);
  auto* dst_v = static_cast<float*>(d_kv_cache_) + total_kv;
  HIP_CHECK(hipMemcpy2DAsync(dst_k, dst_pitch, snapshot.d_k_f32_, row_width,
                             row_width, heads, hipMemcpyDeviceToDevice,
                             stream_));
  HIP_CHECK(hipMemcpy2DAsync(dst_v, dst_pitch, snapshot.d_v_f32_, row_width,
                             row_width, heads, hipMemcpyDeviceToDevice,
                             stream_));
  auto* dst_k16 = static_cast<std::uint8_t*>(d_kv_cache_f16_);
  HIP_CHECK(hipMemcpyAsync(dst_k16, snapshot.d_k_f16_, f16_plane,
                           hipMemcpyDeviceToDevice, stream_));
  HIP_CHECK(hipMemcpyAsync(
      dst_k16 + MtpCheckedMultiply(total_kv, sizeof(std::uint16_t)),
      snapshot.d_v_f16_, f16_plane, hipMemcpyDeviceToDevice, stream_));
  HIP_CHECK(hipStreamSynchronize(stream_));
  next_position_ = valid;
  f16_prefix_packed_ = valid;
}

std::size_t QwenMtpGpuSnapshot::PersistentPayloadBytes() const {
  return MtpCheckedSum(kMtpGpuPersistentHeaderBytes, payload_bytes_);
}

std::size_t QwenMtpGpuSnapshot::SerializePersistent(
    std::span<std::uint8_t> destination) const {
  const std::size_t expected_bytes = PersistentPayloadBytes();
  if (destination.size() != expected_bytes) {
    throw std::invalid_argument(
        "MTP GPU persistent destination size is invalid");
  }
  if (num_kv_heads_ == 0 || head_dim_ == 0 || valid_context_ > max_context_ ||
      head_dim_ > std::numeric_limits<std::size_t>::max() / num_kv_heads_) {
    throw std::invalid_argument("MTP GPU snapshot metadata is malformed");
  }
  const std::size_t kv = static_cast<std::size_t>(num_kv_heads_) * head_dim_;
  const std::size_t f32_plane =
      MtpCheckedMultiply(MtpCheckedMultiply(valid_context_, kv), sizeof(float));
  const std::size_t f16_plane = MtpCheckedMultiply(
      MtpCheckedMultiply(valid_context_, kv), sizeof(std::uint16_t));
  const std::size_t expected_payload = MtpCheckedSum(
      MtpCheckedMultiply(f32_plane, 2), MtpCheckedMultiply(f16_plane, 2));
  if (payload_bytes_ != expected_payload ||
      ((f32_plane != 0 || f16_plane != 0) &&
       (d_k_f32_ == nullptr || d_v_f32_ == nullptr || d_k_f16_ == nullptr ||
        d_v_f16_ == nullptr))) {
    throw std::invalid_argument("MTP GPU snapshot payload is malformed");
  }

  // The plane copies overwrite the payload; do not clear those pages.
  std::fill_n(destination.begin(), kMtpGpuPersistentHeaderBytes,
              std::uint8_t{0});
  std::copy(kMtpGpuPersistentMagic.begin(), kMtpGpuPersistentMagic.end(),
            destination.begin());
  PutLittleEndian<std::uint32_t>(destination, 8, kMtpGpuPersistentVersion);
  PutLittleEndian<std::uint32_t>(
      destination, 12,
      static_cast<std::uint32_t>(kMtpGpuPersistentHeaderBytes));
  PutLittleEndian<std::uint32_t>(destination, 16, num_kv_heads_);
  PutLittleEndian<std::uint32_t>(destination, 20, head_dim_);
  PutLittleEndian<std::uint32_t>(destination, 24, max_context_);
  PutLittleEndian<std::uint32_t>(destination, 28, valid_context_);
  PutLittleEndian<std::uint64_t>(destination, 32,
                                 static_cast<std::uint64_t>(f32_plane));
  PutLittleEndian<std::uint64_t>(destination, 40,
                                 static_cast<std::uint64_t>(f16_plane));
  PutLittleEndian<std::uint64_t>(destination, 48,
                                 static_cast<std::uint64_t>(payload_bytes_));
  PutLittleEndian<std::uint64_t>(destination, 56,
                                 static_cast<std::uint64_t>(expected_bytes));

  if (f32_plane != 0) {
    SnapshotTransfer transfer;
    transfer.Copy(destination.data() + kMtpGpuPersistentHeaderBytes, d_k_f32_,
                  f32_plane);
    transfer.Copy(destination.data() + kMtpGpuPersistentHeaderBytes + f32_plane,
                  d_v_f32_, f32_plane);
  }
  if (f16_plane != 0) {
    SnapshotTransfer transfer;
    transfer.Copy(destination.data() + kMtpGpuPersistentHeaderBytes +
                      f32_plane + f32_plane,
                  d_k_f16_, f16_plane);
    transfer.Copy(destination.data() + kMtpGpuPersistentHeaderBytes +
                      f32_plane + f32_plane + f16_plane,
                  d_v_f16_, f16_plane);
  }
  return destination.size();
}

void QwenMtpGpuExecutor::RestorePersistentSnapshot(
    std::span<const std::uint8_t> payload) {
  if (payload.size() < kMtpGpuPersistentHeaderBytes ||
      !std::equal(kMtpGpuPersistentMagic.begin(), kMtpGpuPersistentMagic.end(),
                  payload.begin()) ||
      GetLittleEndian<std::uint32_t>(payload, 8) != kMtpGpuPersistentVersion ||
      GetLittleEndian<std::uint32_t>(payload, 12) !=
          kMtpGpuPersistentHeaderBytes) {
    throw std::invalid_argument("MTP GPU persistent header is invalid");
  }
  const std::uint32_t heads = GetLittleEndian<std::uint32_t>(payload, 16);
  const std::uint32_t dim = GetLittleEndian<std::uint32_t>(payload, 20);
  const std::uint32_t max_context = GetLittleEndian<std::uint32_t>(payload, 24);
  const std::uint32_t valid = GetLittleEndian<std::uint32_t>(payload, 28);
  const std::size_t f32_plane =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 32));
  const std::size_t f16_plane =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 40));
  const std::size_t payload_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 48));
  const std::size_t total_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 56));

  const auto& config = model_->GetConfig();
  if (config.num_key_value_heads == 0 || config.head_dim == 0 ||
      heads != config.num_key_value_heads || dim != config.head_dim ||
      max_context != max_context_ || valid > max_context_) {
    throw std::invalid_argument("MTP GPU persistent metadata is incompatible");
  }
  const std::size_t kv = MtpCheckedMultiply(heads, dim);
  const std::size_t expected_f32 =
      MtpCheckedMultiply(MtpCheckedMultiply(valid, kv), sizeof(float));
  const std::size_t expected_f16 =
      MtpCheckedMultiply(MtpCheckedMultiply(valid, kv), sizeof(std::uint16_t));
  if (f32_plane != expected_f32 || f16_plane != expected_f16 ||
      payload_bytes != MtpCheckedSum(MtpCheckedMultiply(expected_f32, 2),
                                     MtpCheckedMultiply(expected_f16, 2)) ||
      total_bytes != payload.size() ||
      MtpCheckedSum(kMtpGpuPersistentHeaderBytes, payload_bytes) !=
          payload.size()) {
    throw std::invalid_argument("MTP GPU persistent payload size is invalid");
  }
  if (valid == 0) {
    next_position_ = 0;
    f16_prefix_packed_ = 0;
    return;
  }
  const std::size_t total_kv =
      MtpCheckedMultiply(MtpCheckedMultiply(heads, max_context_), dim);
  const std::size_t row_width =
      MtpCheckedMultiply(MtpCheckedMultiply(valid, dim), sizeof(float));
  const std::size_t dst_pitch =
      MtpCheckedMultiply(MtpCheckedMultiply(max_context_, dim), sizeof(float));
  const std::uint8_t* planes = payload.data() + kMtpGpuPersistentHeaderBytes;
  auto* dst_k = static_cast<float*>(d_kv_cache_);
  auto* dst_v = static_cast<float*>(d_kv_cache_) + total_kv;
  HIP_CHECK(hipMemcpy2DAsync(dst_k, dst_pitch, planes, row_width, row_width,
                             heads, hipMemcpyHostToDevice, stream_));
  HIP_CHECK(hipMemcpy2DAsync(dst_v, dst_pitch, planes + f32_plane, row_width,
                             row_width, heads, hipMemcpyHostToDevice, stream_));
  auto* dst_k16 = static_cast<std::uint8_t*>(d_kv_cache_f16_);
  HIP_CHECK(hipMemcpyAsync(dst_k16, planes + f32_plane + f32_plane, f16_plane,
                           hipMemcpyHostToDevice, stream_));
  HIP_CHECK(hipMemcpyAsync(
      dst_k16 + MtpCheckedMultiply(total_kv, sizeof(std::uint16_t)),
      planes + f32_plane + f32_plane + f16_plane, f16_plane,
      hipMemcpyHostToDevice, stream_));
  HIP_CHECK(hipStreamSynchronize(stream_));
  next_position_ = valid;
  f16_prefix_packed_ = valid;
}

}  // namespace superroc::hip
#endif  // defined(ENGINE_ENABLE_HIP)
