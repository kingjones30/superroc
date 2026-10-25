#ifndef GUFO_MODELS_QWEN_HIP_MTP_HPP_
#define GUFO_MODELS_QWEN_HIP_MTP_HPP_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/mtp_reference.hpp"
#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Immutable GPU-visible MTP layer weights. Current GGUF K-quant matrices are
/// expanded once to BF16; the tied embedding and LM head remain shared with
/// the target model.
class QwenMtpGpuModel final {
public:
  ~QwenMtpGpuModel();

  QwenMtpGpuModel(const QwenMtpGpuModel&) = delete;
  QwenMtpGpuModel& operator=(const QwenMtpGpuModel&) = delete;
  QwenMtpGpuModel(QwenMtpGpuModel&&) = delete;
  QwenMtpGpuModel& operator=(QwenMtpGpuModel&&) = delete;

  [[nodiscard]] static std::shared_ptr<const QwenMtpGpuModel> Create(
      std::shared_ptr<const core::GgufReader> mtp_reader,
      std::shared_ptr<const QwenGpuModel> target_model,
      std::string* error_msg = nullptr);

  // KAT-MTP: the nextn block already resides in the target model (blk.40).
  // No sidecar file. Tensors are aliased, not copied.
  [[nodiscard]] static std::shared_ptr<const QwenMtpGpuModel>
  CreateFromEmbedded(std::shared_ptr<const QwenGpuModel> target_model,
                     std::string* error_msg = nullptr);

  [[nodiscard]] const speculative::QwenMtpWeights& GetWeights() const noexcept {
    return weights_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }
  [[nodiscard]] std::size_t GetPackedWeightBytes() const noexcept {
    return packed_weight_bytes_;
  }
  [[nodiscard]] double GetPackTimeSeconds() const noexcept {
    return pack_time_seconds_;
  }
  /// Host copy of the target's final output_norm weights. The MTP head
  /// consumes the trunk's POST-final-norm hidden state (fork parity: the
  /// reference feeds ggml's t_h_pre_norm, which qwen35.cpp assigns after
  /// model.output_norm); the backend normalizes the captured pre-norm
  /// features with these weights as it ingests them.
  [[nodiscard]] std::span<const float> GetTargetOutputNorm() const noexcept {
    return target_output_norm_;
  }
  /// Super ROC DVT: the target model (its tokenizer names the special
  /// tokens the trimmed draft head must keep).
  [[nodiscard]] const QwenGpuModel& GetTargetModel() const noexcept {
    return *target_model_;
  }

private:
  QwenMtpGpuModel(std::shared_ptr<const core::GgufReader> mtp_reader,
                  std::shared_ptr<const QwenGpuModel> target_model,
                  speculative::QwenMtpWeights weights,
                  std::vector<void*> allocations,
                  std::size_t packed_weight_bytes, double pack_time_seconds,
                  std::vector<float> target_output_norm);

  std::shared_ptr<const core::GgufReader> mtp_reader_;
  std::shared_ptr<const QwenGpuModel> target_model_;
  speculative::QwenMtpWeights weights_;
  std::vector<void*> allocations_;
  std::size_t packed_weight_bytes_{0};
  double pack_time_seconds_{0.0};
  std::vector<float> target_output_norm_;
};

class QwenMtpGpuSnapshot;

/// Mutable single-session executor for the Qwen3.8 layer-64 MTP graph.
class QwenMtpGpuExecutor final {
public:
  ~QwenMtpGpuExecutor();

  QwenMtpGpuExecutor(const QwenMtpGpuExecutor&) = delete;
  QwenMtpGpuExecutor& operator=(const QwenMtpGpuExecutor&) = delete;
  QwenMtpGpuExecutor(QwenMtpGpuExecutor&&) = delete;
  QwenMtpGpuExecutor& operator=(QwenMtpGpuExecutor&&) = delete;

  [[nodiscard]] static std::unique_ptr<QwenMtpGpuExecutor> Create(
      std::shared_ptr<const QwenMtpGpuModel> model, std::uint32_t max_context,
      std::string* error_msg = nullptr);

  void SetVisionInput(models::qwen::vision::DeviceInput* input) noexcept {
    vision_input_ = input;
  }
  void Reset() noexcept;
  void Rewind(std::uint32_t position);
  void Seek(std::uint32_t position);
  void Sync();

  [[nodiscard]] tokenization::TokenId ForwardTargetHidden(
      tokenization::TokenId input_token, std::span<const float> target_hidden,
      std::uint32_t position, bool compute_logits = true);

  [[nodiscard]] tokenization::TokenId ForwardFeedback(
      tokenization::TokenId input_token, std::uint32_t position,
      bool compute_logits = true);

  /// SPEC4 item 2: batched prompt priming. One MTP-block forward for a whole
  /// chunk of cells [start_pos, start_pos + tokens.size()); hidden_rows are
  /// the ALREADY-normalized target features for the same rows (pair p uses
  /// h_{p-1}). Writes the MTP KV identically to the serial path.
  void ForwardTargetBatch(std::span<const tokenization::TokenId> tokens,
                          std::span<const float> hidden_rows,
                          std::uint32_t start_pos);

  /// Exports the F32 KV planes (K then V) for the serial-vs-batch equality
  /// gate.
  std::vector<float> CopyKvCacheF32();

  /// Maximum rows per ForwardTargetBatch call.
  static constexpr std::size_t kPrimeBatchRows = 4096;

  [[nodiscard]] std::span<const float> CopyLastHidden();
  [[nodiscard]] std::span<const float> CopyLastLogits();

  [[nodiscard]] std::uint32_t GetNextPosition() const noexcept {
    return next_position_;
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return max_context_;
  }
  [[nodiscard]] std::size_t GetHiddenSize() const noexcept {
    return model_->GetConfig().hidden_size;
  }
  [[nodiscard]] bool IsMoe() const noexcept {
    return model_->GetWeights().layer.is_moe;
  }
  /// SQ10 SPEC2 (R1): this helper's stream. The session aliases it to its
  /// own set at backend creation; the dtor removes the alias (R3).
  [[nodiscard]] hipStream_t Stream() const noexcept { return stream_; }

  /// Retained KV state and per-step scratch, in the same terms the DFlash
  /// executor reports them. The MTP draft keeps its own full-size KV cache, so
  /// this is the number a serving layer must budget for a second cache.
  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const noexcept {
    return {state_bytes_, scratch_bytes_};
  }

  /// Exact device bytes SaveSnapshot will allocate for the valid prefix.
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const;
  /// Copies the valid KV prefix [0, next_position_) to a device snapshot.
  [[nodiscard]] std::unique_ptr<QwenMtpGpuSnapshot> SaveSnapshot() const;
  /// Restores a snapshot taken from this configuration and rewinds the draft
  /// position to its boundary. Both planes are fully valid after a restore,
  /// so the FP16 pack hint is set to the boundary (never a stale value).
  void RestoreSnapshot(const QwenMtpGpuSnapshot& snapshot);
  void RestorePersistentSnapshot(std::span<const std::uint8_t> payload);

private:
  QwenMtpGpuExecutor(std::shared_ptr<const QwenMtpGpuModel> model,
                     std::uint32_t max_context);

  [[nodiscard]] tokenization::TokenId Run(tokenization::TokenId input_token,
                                          const float* hidden_input,
                                          std::uint32_t position,
                                          bool compute_logits);
  void Allocate();
  void Free() noexcept;
  /// Super ROC DVT (draft vocab trim), read once from GUFO_MTP_DRAFT_VOCAB.
  void ConfigureDraftVocab();

  std::shared_ptr<const QwenMtpGpuModel> model_;
  std::uint32_t max_context_;
  std::uint32_t next_position_{0};
  /// F16 KV rows [0, f16_prefix_packed_) already match the F32 cache.
  /// Later prime chunks skip the prefix reconvert.
  std::uint32_t f16_prefix_packed_{0};
  hipStream_t stream_{nullptr};

  float* d_target_hidden_{nullptr};
  float* d_embedding_{nullptr};
  float* d_fusion_{nullptr};
  float* d_hidden_{nullptr};
  float* d_normed_{nullptr};
  float* d_qg_{nullptr};
  float* d_q_{nullptr};
  float* d_k_{nullptr};
  float* d_v_{nullptr};
  float* d_gate_{nullptr};
  float* d_context_{nullptr};
  float* d_attn_out_{nullptr};
  float* d_ffn_act_{nullptr};
  float* d_ffn_out_{nullptr};
  float* d_feedback_hidden_{nullptr};
  float* d_logits_{nullptr};
  float* d_kv_cache_{nullptr};
  void* d_kv_cache_f16_{nullptr};
  float* d_split_k_scratch_{nullptr};

  /// Batched-prime scratch (kPrimeBatchRows rows).
  std::uint32_t* d_b_tokens_{nullptr};
  float* d_b_h_{nullptr};
  float* d_b_embed_{nullptr};
  float* d_b_fusion_{nullptr};
  float* d_b_hidden_{nullptr};
  float* d_b_normed_{nullptr};
  float* d_b_qg_{nullptr};
  float* d_b_q_{nullptr};
  float* d_b_gate_{nullptr};
  float* d_b_k_{nullptr};
  float* d_b_v_{nullptr};
  float* d_b_context_{nullptr};
  float* d_b_attn_out_{nullptr};
  float* d_b_ffn_act_{nullptr};
  float* d_b_ffn_up_{nullptr};
  float* d_b_ffn_out_{nullptr};
  std::uint32_t* d_out_token_{nullptr};
  // KAT-MTP: one-token MoE scratch. Null unless the embedded block is MoE.
  float* d_moe_logits_{nullptr};
  int* d_moe_ids_{nullptr};
  float* d_moe_w_{nullptr};
  float* d_moe_sig_{nullptr};
  float* d_moe_gate_{nullptr};
  float* d_moe_up_{nullptr};
  float* d_moe_act_{nullptr};
  float* d_moe_h_{nullptr};
  /// BF16 activation scratch for the hipBLAS prime GEMM. One weight read
  /// per chunk; sized to kPrimeBatchRows times the widest K.
  std::uint16_t* d_b_bf16_{nullptr};
  std::size_t b_bf16_elems_{0};
  /// BF16 image of one Q4_0 prime weight. Decode itself stays on the Q4_0 GEMV.
  std::uint16_t* d_q4_bf16_{nullptr};
  std::size_t q4_bf16_elems_{0};
  hipblasHandle_t hipblas_{nullptr};

  /// Prime-chunk GEMM. Wider than 16 rows goes through hipBLAS (BF16
  /// activations, FP32 accumulate). The tail stays on the exact 16-row GEMM.
  void RunPrimeGemm(const void* weight, core::GgmlType type, const float* input,
                    float* output, std::size_t batch, std::size_t rows_out,
                    std::size_t cols_in, hipStream_t stream);

  models::qwen::vision::DeviceInput* vision_input_{
      nullptr};  // owned by the target session
  std::size_t state_bytes_{0};
  std::size_t scratch_bytes_{0};
  std::vector<float> h_last_hidden_;
  std::vector<float> h_logits_;

  /// Super ROC DVT. GUFO_MTP_DRAFT_VOCAB=N (N >= 1024) scores only the
  /// head's rows [0, N) and the tail [draft_tail_start_, vocab) that holds
  /// the special tokens; the draft logits buffer is packed head then tail.
  /// Default N = 98304; GUFO_MTP_DRAFT_VOCAB=0 (draft_rows_ == 0) keeps the
  /// full head. Drafts are proposals only:
  /// the target verifies every token with its full head, so greedy output
  /// does not depend on this.
  std::uint32_t draft_head_rows_{0};
  std::uint32_t draft_tail_start_{0};
  std::uint32_t draft_rows_{0};
  std::size_t draft_row_bytes_{0};
  std::vector<float> h_draft_logits_;
};

/// Device copy of the MTP draft KV prefix [0, valid_context): the FP32
/// head-major planes plus the canonical FP16 token-major planes. The draft
/// block is append-only, so a prefix copy restores exactly; per-step scratch
/// (logits, feedback hidden, batched-prime buffers) is not retained.
class QwenMtpGpuSnapshot final {
public:
  ~QwenMtpGpuSnapshot();

  QwenMtpGpuSnapshot(const QwenMtpGpuSnapshot&) = delete;
  QwenMtpGpuSnapshot& operator=(const QwenMtpGpuSnapshot&) = delete;
  QwenMtpGpuSnapshot(QwenMtpGpuSnapshot&&) = delete;
  QwenMtpGpuSnapshot& operator=(QwenMtpGpuSnapshot&&) = delete;

  [[nodiscard]] std::size_t PayloadBytes() const noexcept {
    return payload_bytes_;
  }
  [[nodiscard]] std::uint32_t ValidContext() const noexcept {
    return valid_context_;
  }
  [[nodiscard]] std::size_t PersistentPayloadBytes() const;
  [[nodiscard]] std::size_t SerializePersistent(
      std::span<std::uint8_t> destination) const;

private:
  QwenMtpGpuSnapshot() = default;

  void* d_k_f32_{nullptr};
  void* d_v_f32_{nullptr};
  void* d_k_f16_{nullptr};
  void* d_v_f16_{nullptr};
  std::uint32_t num_kv_heads_{0};
  std::uint32_t head_dim_{0};
  std::uint32_t max_context_{0};
  std::uint32_t valid_context_{0};
  std::size_t payload_bytes_{0};

  friend class QwenMtpGpuExecutor;
};

struct QwenMtpGpuDraftConfig {
  std::uint32_t max_context{4096};
  std::uint32_t max_draft_tokens{4};
  models::qwen::vision::DeviceInput* vision_input{nullptr};
};

/// Stateful MTP draft provider backed by the production HIP executor.
class QwenMtpGpuDraftBackend final : public speculative::IDraftBackend {
public:
  [[nodiscard]] static std::unique_ptr<QwenMtpGpuDraftBackend> Create(
      std::shared_ptr<const QwenMtpGpuModel> model,
      QwenMtpGpuDraftConfig config = {}, std::string* error_msg = nullptr);

  [[nodiscard]] static std::unique_ptr<QwenMtpGpuDraftBackend> CreateFromGguf(
      std::string_view model_path,
      std::shared_ptr<const QwenGpuModel> target_model,
      QwenMtpGpuDraftConfig config = {}, std::string* error_msg = nullptr);

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "QwenMtpGpuDraftBackend";
  }

  /// SQ10 SPEC2 (R1): the draft executor's stream, for aliasing to the
  /// owning session's set at creation.
  [[nodiscard]] hipStream_t DraftStream() const noexcept {
    return executor_ != nullptr ? executor_->Stream() : nullptr;
  }

  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }

  [[nodiscard]] bool PrimeTargetContext(
      const speculative::DraftTargetContext& context) override;
  [[nodiscard]] bool AppendTargetContext(
      const speculative::DraftTargetContext& context,
      std::uint32_t position) override;

  [[nodiscard]] speculative::DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;

  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override;

  void UpdateTargetHidden(std::span<const float> hidden) override;

  /// Drops target features that a cancelled verification step left queued and
  /// rewinds the draft's own position to match the target's. Mirrors
  /// QwenDFlashGpuDraftBackend::DiscardPendingTargetContext; the MTP draft has
  /// no injected-context graph to keep in step, so this only has to clear the
  /// queue and set next_position_.
  void DiscardPendingTargetContext(std::uint32_t position);

  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const noexcept {
    return executor_->GetMemoryUsage();
  }

  /// F7: the draft executor's next position, for the capture-site check that
  /// the draft prefix covers exactly the target's committed context.
  [[nodiscard]] std::uint32_t DraftPosition() const noexcept {
    return executor_ != nullptr ? executor_->GetNextPosition() : 0;
  }

  /// The draft KV prefix plus the pending normalized target hidden row.
  /// Snapshots persist the draft KV prefix and hidden row next to the target
  /// state, so post-restore drafts match an uninterrupted run.
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const override;
  [[nodiscard]] std::unique_ptr<speculative::IDraftBackendSnapshot> Snapshot()
      const override;
  void RestoreSnapshot(
      const speculative::IDraftBackendSnapshot& snapshot) override;
  void RestorePersistentSnapshot(
      std::span<const std::uint8_t> payload) override;

  void Reset() noexcept override;
  ~QwenMtpGpuDraftBackend() override;

  [[nodiscard]] const std::string& GetLastError() const noexcept {
    return last_error_;
  }

private:
  struct TfDraftRecord {
    std::uint32_t cell;
    std::uint32_t index;
    tokenization::TokenId token;
    float probability;
  };

  QwenMtpGpuDraftBackend(std::unique_ptr<QwenMtpGpuExecutor> executor,
                         QwenMtpGpuDraftConfig config,
                         std::vector<float> target_output_norm);

  /// RMSNorms a captured pre-final-norm target feature row with the
  /// target's output_norm weights (in place into normed_hidden_).
  std::span<const float> NormalizeTargetHidden(std::span<const float> hidden);
  void TfRecord(tokenization::TokenId token, std::uint32_t cell,
                std::uint32_t index);
  void TfFlush(std::size_t accepted);

  std::unique_ptr<QwenMtpGpuExecutor> executor_;
  QwenMtpGpuDraftConfig config_;
  std::vector<float> output_norm_;
  std::vector<float> normed_hidden_;
  std::vector<float> zero_hidden_;
  bool norm_at_ingest_{true};
  bool batch_prime_{true};
  std::vector<float> target_hidden_;
  std::vector<float> proposal_target_hidden_;
  std::vector<float> committed_target_hidden_;
  std::vector<tokenization::TokenId> proposed_tokens_;
  tokenization::TokenId proposal_input_{0};
  std::uint32_t proposal_checkpoint_{0};
  bool primed_{false};
  bool proposal_active_{false};
  std::string last_error_;
  std::string tf_path_;
  std::FILE* tf_file_{nullptr};
  std::FILE* tf_logits_file_{nullptr};
  std::uint32_t tf_round_{0};
  std::string kv_dump_path_;
  std::uint32_t kv_dump_count_{0};
  std::uint32_t tf_logits_rows_{0};
  std::vector<TfDraftRecord> tf_pending_;
};

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_MTP_HPP_
