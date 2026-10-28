#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/core/env_compat.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen/hip/mtp/kat_mtp_pmin.hpp"
#include "src/models/qwen/hip/mtp/snapshot_guard.hpp"

namespace superroc::hip {

QwenMtpGpuDraftBackend::QwenMtpGpuDraftBackend(
    std::unique_ptr<QwenMtpGpuExecutor> executor, QwenMtpGpuDraftConfig config,
    std::vector<float> target_output_norm)
    : executor_(std::move(executor)),
      config_(config),
      output_norm_(std::move(target_output_norm)),
      normed_hidden_(executor_->GetHiddenSize()),
      zero_hidden_(executor_->GetHiddenSize(), 0.0F),
      target_hidden_(executor_->GetHiddenSize()) {
  if (output_norm_.size() != target_hidden_.size()) {
    throw std::invalid_argument(
        "MTP target output_norm width does not match the hidden size");
  }
  if (const char* kv_dump =
          ::superroc::core::GetEnvCompat("SUPERROC_MTP_KV_DUMP");
      kv_dump != nullptr && *kv_dump != '\0') {
    kv_dump_path_ = kv_dump;
  }
  if (const char* serial =
          ::superroc::core::GetEnvCompat("SUPERROC_MTP_PRIME_SERIAL");
      serial != nullptr && *serial != '\0' && std::atoi(serial) != 0) {
    batch_prime_ = false;
  }
  if (const char* prenorm =
          ::superroc::core::GetEnvCompat("SUPERROC_MTP_PRENORM_H");
      prenorm != nullptr && *prenorm != '\0' && std::atoi(prenorm) != 0) {
    norm_at_ingest_ = false;
  }
  if (const char* path = ::superroc::core::GetEnvCompat("SUPERROC_MTP_TF_DUMP");
      path != nullptr && *path != '\0') {
    tf_path_ = path;
    tf_file_ = std::fopen(path, "w");
    if (tf_file_ == nullptr) {
      throw std::runtime_error("SUPERROC_MTP_TF_DUMP path is not writable: " +
                               tf_path_);
    }
    std::fprintf(tf_file_,
                 "# TF cell=MTP KV position of the input pair; prediction "
                 "target position is cell+idx+1\n");
    if (const char* logits =
            ::superroc::core::GetEnvCompat("SUPERROC_MTP_TF_LOGITS");
        logits != nullptr && *logits != '\0' && std::atoi(logits) != 0) {
      tf_logits_file_ = std::fopen((tf_path_ + ".bin").c_str(), "wb");
      if (tf_logits_file_ == nullptr) {
        throw std::runtime_error("SUPERROC_MTP_TF_LOGITS path is not writable");
      }
    }
  }
}

QwenMtpGpuDraftBackend::~QwenMtpGpuDraftBackend() {
  if (tf_file_ != nullptr) {
    std::fclose(tf_file_);
  }
  if (tf_logits_file_ != nullptr) {
    std::fclose(tf_logits_file_);
  }
}

std::span<const float> QwenMtpGpuDraftBackend::NormalizeTargetHidden(
    std::span<const float> hidden) {
  if (hidden.size() != output_norm_.size()) {
    throw std::invalid_argument("MTP target hidden-state shape is invalid");
  }
  if (!norm_at_ingest_) {
    return hidden;
  }
  float sum = 0.0F;
  for (const float value : hidden) {
    sum += value * value;
  }
  const float scale =
      1.0F / std::sqrt(sum / static_cast<float>(hidden.size()) + 1.0e-6F);
  for (std::size_t index = 0; index < hidden.size(); ++index) {
    normed_hidden_[index] = hidden[index] * scale * output_norm_[index];
  }
  return normed_hidden_;
}

void QwenMtpGpuDraftBackend::TfRecord(tokenization::TokenId token,
                                      std::uint32_t cell, std::uint32_t index) {
  if (tf_file_ == nullptr) {
    return;
  }
  float probability = 0.0F;
  const auto logits = executor_->CopyLastLogits();
  if (!logits.empty()) {
    float peak = logits.front();
    for (const float value : logits) {
      peak = value > peak ? value : peak;
    }
    double denominator = 0.0;
    for (const float value : logits) {
      denominator += std::exp(static_cast<double>(value - peak));
    }
    if (const auto chosen = static_cast<std::size_t>(token);
        chosen < logits.size()) {
      probability = static_cast<float>(
          std::exp(static_cast<double>(logits[chosen] - peak)) /
          (denominator > 0.0 ? denominator : 1.0));
    }
    if (index == 0 && tf_logits_file_ != nullptr && tf_logits_rows_ < 24) {
      std::fwrite(logits.data(), sizeof(float), logits.size(), tf_logits_file_);
      ++tf_logits_rows_;
    }
  }
  tf_pending_.push_back(TfDraftRecord{cell, index, token, probability});
}

void QwenMtpGpuDraftBackend::TfFlush(std::size_t accepted) {
  if (tf_file_ == nullptr) {
    return;
  }
  for (std::size_t index = 0; index < tf_pending_.size(); ++index) {
    const auto& record = tf_pending_[index];
    std::fprintf(tf_file_, "TF round=%u cell=%u idx=%u tok=%u p=%.6f acc=%d\n",
                 tf_round_, record.cell, record.index,
                 static_cast<unsigned>(record.token), record.probability,
                 index < accepted ? 1 : 0);
  }
  ++tf_round_;
  tf_pending_.clear();
  std::fflush(tf_file_);
}

std::unique_ptr<QwenMtpGpuDraftBackend> QwenMtpGpuDraftBackend::Create(
    std::shared_ptr<const QwenMtpGpuModel> model, QwenMtpGpuDraftConfig config,
    std::string* error_msg) {
  if (model == nullptr || config.max_draft_tokens == 0) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GPU draft configuration is invalid";
    }
    return nullptr;
  }
  std::vector<float> target_output_norm(model->GetTargetOutputNorm().begin(),
                                        model->GetTargetOutputNorm().end());
  auto executor = QwenMtpGpuExecutor::Create(std::move(model),
                                             config.max_context, error_msg);
  if (executor == nullptr) {
    return nullptr;
  }
  executor->SetVisionInput(config.vision_input);
  return std::unique_ptr<QwenMtpGpuDraftBackend>(new QwenMtpGpuDraftBackend(
      std::move(executor), config, std::move(target_output_norm)));
}

std::unique_ptr<QwenMtpGpuDraftBackend> QwenMtpGpuDraftBackend::CreateFromGguf(
    std::string_view model_path,
    std::shared_ptr<const QwenGpuModel> target_model,
    QwenMtpGpuDraftConfig config, std::string* error_msg) {
  if (model_path.empty() || target_model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GGUF path and target GPU model are required";
    }
    return nullptr;
  }
  auto reader_owner =
      core::GgufReader::OpenFile(std::string(model_path), error_msg);
  if (reader_owner == nullptr) {
    return nullptr;
  }
  std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  auto model = QwenMtpGpuModel::Create(std::move(reader),
                                       std::move(target_model), error_msg);
  if (model == nullptr) {
    return nullptr;
  }
  return Create(std::move(model), config, error_msg);
}

bool QwenMtpGpuDraftBackend::PrimeTargetContext(
    const speculative::DraftTargetContext& context) {
  Reset();
  if (context.prompt_tokens.empty() || context.hidden_size == 0 ||
      context.hidden_size != target_hidden_.size() ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * context.hidden_size) {
    last_error_ = "MTP prompt hidden-state shape is invalid";
    return false;
  }

  try {
    // F3, fork parity: MTP cell p holds the pair (x_p, h_{p-1}) at
    // RoPE/KV position p; cell 0 is (x_0, zeros), exactly the
    // reference draft-mtp feed. F1: every captured feature row is
    // normalized with the target's output_norm at ingest (the head
    // consumes post-final-norm hidden states).
    (void)executor_->ForwardTargetHidden(context.prompt_tokens[0], zero_hidden_,
                                         0, false);
    // KAT-MTP: a full MoE step per prompt token is the expensive path, so
    // batch the KV-only prime for every prompt longer than one token. The
    // dense sidecar keeps the old floor.
    const std::size_t batch_floor =
        executor_->IsMoe() ? 1 : (1 + QwenMtpGpuExecutor::kPrimeBatchRows / 16);
    if (batch_prime_ && context.prompt_tokens.size() > batch_floor) {
      // SPEC4 item 2: prime in batched chunks (one MTP-block forward per
      // chunk) instead of one forward per prompt token. Falls back to the
      // serial loop below when the batched path rejects the configuration.
      try {
        const std::size_t pair_count = context.prompt_tokens.size() - 1;
        std::vector<float> normed_rows;
        std::size_t offset = 0;
        while (offset < pair_count) {
          const std::size_t count = std::min(
              QwenMtpGpuExecutor::kPrimeBatchRows, pair_count - offset);
          normed_rows.resize(count * context.hidden_size);
          for (std::size_t row = 0; row < count; ++row) {
            const auto hidden = context.prompt_hidden_states.subspan(
                (offset + row) * context.hidden_size, context.hidden_size);
            std::ranges::copy(
                NormalizeTargetHidden(hidden),
                normed_rows.begin() +
                    static_cast<std::ptrdiff_t>(row * context.hidden_size));
          }
          executor_->ForwardTargetBatch(
              context.prompt_tokens.subspan(offset + 1, count), normed_rows,
              static_cast<std::uint32_t>(offset + 1));
          offset += count;
        }
      } catch (const std::exception&) {
        Reset();
        (void)executor_->ForwardTargetHidden(context.prompt_tokens[0],
                                             zero_hidden_, 0, false);
        for (std::size_t index = 1; index < context.prompt_tokens.size();
             ++index) {
          const auto hidden = context.prompt_hidden_states.subspan(
              (index - 1) * context.hidden_size, context.hidden_size);
          (void)executor_->ForwardTargetHidden(
              context.prompt_tokens[index], NormalizeTargetHidden(hidden),
              static_cast<std::uint32_t>(index), false);
        }
      }
    } else {
      for (std::size_t index = 1; index < context.prompt_tokens.size();
           ++index) {
        const auto hidden = context.prompt_hidden_states.subspan(
            (index - 1) * context.hidden_size, context.hidden_size);
        (void)executor_->ForwardTargetHidden(
            context.prompt_tokens[index], NormalizeTargetHidden(hidden),
            static_cast<std::uint32_t>(index), false);
      }
    }
    const std::size_t final_offset =
        (context.prompt_tokens.size() - 1) * context.hidden_size;
    const auto final_hidden =
        context.prompt_hidden_states.subspan(final_offset, context.hidden_size);
    std::ranges::copy(NormalizeTargetHidden(final_hidden),
                      target_hidden_.begin());
    primed_ = true;
    if (!kv_dump_path_.empty()) {
      const auto kv = executor_->CopyKvCacheF32();
      const std::string path =
          kv_dump_path_ + "." + std::to_string(kv_dump_count_++) + ".bin";
      if (std::FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
        std::fwrite(kv.data(), sizeof(float), kv.size(), file);
        std::fclose(file);
      }
    }
    return true;
  } catch (const std::exception& exception) {
    last_error_ = exception.what();
    Reset();
    return false;
  }
}

bool QwenMtpGpuDraftBackend::AppendTargetContext(
    const speculative::DraftTargetContext& context, std::uint32_t position) {
  if (!primed_ || proposal_active_ ||
      executor_->GetNextPosition() != position ||
      context.hidden_size != target_hidden_.size() ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * context.hidden_size) {
    return false;
  }
  if (batch_prime_ && !context.prompt_tokens.empty()) {
    // SPEC4 item 2: the extend path (prefill chunks after the first, and any
    // long append) batches exactly like priming. Row r pairs (x_{pos+r},
    // h_{pos+r-1}): row 0 uses target_hidden_ (already normalized), rows >= 1
    // use the captured features normalized here.
    try {
      const std::size_t rows = context.prompt_tokens.size();
      std::vector<float> h_rows(rows * context.hidden_size);
      std::ranges::copy(target_hidden_, h_rows.begin());
      for (std::size_t row = 1; row < rows; ++row) {
        std::ranges::copy(
            NormalizeTargetHidden(context.prompt_hidden_states.subspan(
                (row - 1) * context.hidden_size, context.hidden_size)),
            h_rows.begin() +
                static_cast<std::ptrdiff_t>(row * context.hidden_size));
      }
      executor_->ForwardTargetBatch(context.prompt_tokens, h_rows, position);
      std::ranges::copy(
          NormalizeTargetHidden(context.prompt_hidden_states.subspan(
              (rows - 1) * context.hidden_size, context.hidden_size)),
          target_hidden_.begin());
      return true;
    } catch (const std::exception&) {
      // next_position_ only advances on success, so the serial loop below can
      // still take over (e.g. a non-batchable embedding table).
    }
  }
  for (std::size_t row = 0; row < context.prompt_tokens.size(); ++row) {
    (void)executor_->ForwardTargetHidden(
        context.prompt_tokens[row], target_hidden_,
        position + static_cast<std::uint32_t>(row), false);
    UpdateTargetHidden(context.prompt_hidden_states.subspan(
        row * context.hidden_size, context.hidden_size));
  }
  return true;
}

speculative::DraftProposal QwenMtpGpuDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  if (!primed_ || prompt_tokens.empty() || current_pos == 0) {
    throw std::logic_error("MTP GPU draft backend is not primed");
  }
  if (proposal_active_) {
    throw std::logic_error("MTP GPU proposal feedback is pending");
  }
  if (executor_->GetNextPosition() != current_pos) {
    throw std::logic_error("MTP GPU draft position is inconsistent");
  }

  speculative::DraftProposal proposal;
  proposal.start_pos = current_pos;
  const std::uint32_t count = std::min(max_tokens, config_.max_draft_tokens);
  if (count == 0) {
    return proposal;
  }

  proposal_checkpoint_ = executor_->GetNextPosition();
  proposal_input_ = prompt_tokens.back();
  proposal_target_hidden_ = target_hidden_;
  committed_target_hidden_.clear();
  proposed_tokens_.clear();
  proposed_tokens_.reserve(count);

  // KAT-MTP: fork draft-mtp p-min. Default 0 keeps every draft.
  // SUPERROC_MTP_PMIN is clamped to [0, 1], matching
  // common_speculative_effective_p_min.
  const float p_min = MtpPMinFromEnv();

  auto token = executor_->ForwardTargetHidden(proposal_input_, target_hidden_,
                                              proposal_checkpoint_, true);
  auto keep = [&](tokenization::TokenId drafted, std::uint32_t index) {
    if (p_min > 0.0F) {
      const float probability =
          KatMtpTop1Probability(executor_->CopyLastLogits());
      if (probability < p_min) {
        // KAT-MTP position invariant. A rejected draft leaves proposed_tokens_
        // short (possibly empty); the verifier then commits the target's own
        // token, and AdvanceCommittedToken -> AppendTargetContext requires
        // GetNextPosition() == that position, so the head must be rewound to
        // proposal_checkpoint_ + index first. It then walks the head forward
        // one row itself. Without this rewind AppendTargetContext returns
        // false and the step throws "draft failed to append committed target
        // token" (measured: HTTP 500 when the rewind was guarded off).
        executor_->Rewind(proposal_checkpoint_ + index);
        return false;
      }
    }
    TfRecord(drafted, proposal_checkpoint_ + index, index);
    proposed_tokens_.push_back(drafted);
    return true;
  };
  if (!keep(token, 0)) {
    return proposal;
  }
  for (std::uint32_t index = 1; index < count; ++index) {
    token =
        executor_->ForwardFeedback(token, proposal_checkpoint_ + index, true);
    if (!keep(token, index)) {
      break;
    }
  }
  proposal.tokens = proposed_tokens_;
  proposal_active_ = true;
  return proposal;
}

void QwenMtpGpuDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  (void)correction_token;
  if (!proposal_active_ || accepted.size() > proposed_tokens_.size()) {
    throw std::logic_error("MTP GPU proposal feedback is invalid");
  }
  if (committed_target_hidden_.size() !=
      (accepted.size() + 1) * target_hidden_.size()) {
    throw std::logic_error("MTP feedback requires each committed target row");
  }

  // lqw:spec6. Verify already stored the committed trunk hiddens. Propose
  // wrote MTP KV for the drafted positions; the fork's accept() does not
  // re-decode them. Re-running that layer here is what inflated the host draft
  // bucket. SUPERROC_MTP_REPLAY=1 keeps the old logits-off replay.
  const char* replay_env =
      ::superroc::core::GetEnvCompat("SUPERROC_MTP_REPLAY");
  const bool replay = replay_env != nullptr && replay_env[0] == '1';
  const char* prof_env =
      ::superroc::core::GetEnvCompat("SUPERROC_MTP_REPLAY_PROFILE");
  const bool prof = prof_env != nullptr && prof_env[0] == '1';
  const auto t0 = std::chrono::steady_clock::now();
  std::size_t replay_forwards = 0;
  if (replay) {
    executor_->Rewind(proposal_checkpoint_);
    (void)executor_->ForwardTargetHidden(
        proposal_input_, proposal_target_hidden_, proposal_checkpoint_, false);
    ++replay_forwards;
    for (std::size_t index = 0; index < accepted.size(); ++index) {
      (void)executor_->ForwardTargetHidden(
          accepted[index],
          std::span<const float>(committed_target_hidden_)
              .subspan(index * target_hidden_.size(), target_hidden_.size()),
          proposal_checkpoint_ + static_cast<std::uint32_t>(index) + 1, false);
      ++replay_forwards;
    }
  } else {
    executor_->Rewind(proposal_checkpoint_);
    const auto accepted_n = static_cast<std::uint32_t>(accepted.size());
    const std::uint32_t end = proposal_checkpoint_ + accepted_n + 1;
    if (!accepted.empty() && accepted.size() == proposed_tokens_.size()) {
      const std::size_t index = accepted.size() - 1;
      const std::uint32_t pos =
          proposal_checkpoint_ + static_cast<std::uint32_t>(index) + 1;
      executor_->Seek(pos);
      (void)executor_->ForwardTargetHidden(
          accepted[index],
          std::span<const float>(committed_target_hidden_)
              .subspan(index * target_hidden_.size(), target_hidden_.size()),
          pos, false);
      ++replay_forwards;
    } else {
      executor_->Seek(end);
    }
  }
  if (prof) {
    executor_->Sync();
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    std::fprintf(
        stderr, "REPLAY_PROF mode=%s ms=%.3f forwards=%zu accepted=%zu\n",
        replay ? "replay" : "skip", ms, replay_forwards, accepted.size());
  }
  TfFlush(accepted.size());
  proposal_active_ = false;
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
}

void QwenMtpGpuDraftBackend::UpdateTargetHidden(std::span<const float> hidden) {
  if (hidden.size() != target_hidden_.size()) {
    throw std::invalid_argument("MTP target hidden-state shape is invalid");
  }
  const auto normed = NormalizeTargetHidden(hidden);
  if (proposal_active_) {
    committed_target_hidden_.insert(committed_target_hidden_.end(),
                                    normed.begin(), normed.end());
  }
  std::ranges::copy(normed, target_hidden_.begin());
}

void QwenMtpGpuDraftBackend::DiscardPendingTargetContext(
    std::uint32_t position) {
  if (proposal_active_) {
    throw std::logic_error("MTP cancellation frontier is not retained");
  }
  committed_target_hidden_.clear();
  proposal_target_hidden_.clear();
  proposed_tokens_.clear();
  tf_pending_.clear();
  executor_->Rewind(position);
}

namespace {

constexpr std::array<std::uint8_t, 8> kMtpDraftPersistentMagic = {
    'G', 'M', 'T', 'P', 'D', 'R', '0', '1'};
constexpr std::uint32_t kMtpDraftPersistentVersion = 1;
constexpr std::size_t kMtpDraftPersistentHeaderBytes = 64;
constexpr std::uint32_t kMtpDraftPrimedFlag = 1U << 0U;

template<typename T>
  requires(std::is_unsigned_v<T>)
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     T value) {
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    throw std::length_error("MTP draft persistent header is truncated");
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
    throw std::invalid_argument("MTP draft persistent header is truncated");
  }
  T value = 0;
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    value |= static_cast<T>(source[offset + byte]) << (byte * 8U);
  }
  return value;
}

[[nodiscard]] std::size_t CheckedPersistentAdd(std::size_t left,
                                               std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("MTP draft persistent size overflows");
  }
  return left + right;
}

[[nodiscard]] std::size_t PersistentSizeFromU64(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("MTP draft persistent size overflows");
  }
  return static_cast<std::size_t>(value);
}

class QwenMtpDraftSnapshot final : public speculative::IDraftBackendSnapshot {
public:
  QwenMtpDraftSnapshot(std::unique_ptr<QwenMtpGpuSnapshot> gpu_snapshot,
                       std::vector<float> target_hidden,
                       std::uint32_t max_context, bool primed)
      : gpu_snapshot(std::move(gpu_snapshot)),
        target_hidden(std::move(target_hidden)),
        max_context(max_context),
        primed(primed) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return (gpu_snapshot != nullptr ? gpu_snapshot->PayloadBytes() : 0) +
           target_hidden.size() * sizeof(float);
  }

  [[nodiscard]] std::size_t PersistentPayloadBytes() const override {
    if (gpu_snapshot == nullptr ||
        target_hidden.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
      throw std::invalid_argument(
          "MTP draft snapshot persistent metadata is invalid");
    }
    return CheckedPersistentAdd(
        CheckedPersistentAdd(kMtpDraftPersistentHeaderBytes,
                             gpu_snapshot->PersistentPayloadBytes()),
        target_hidden.size() * sizeof(std::uint32_t));
  }

  [[nodiscard]] std::size_t SerializePersistent(
      std::span<std::uint8_t> destination) const override {
    const std::size_t expected_bytes = PersistentPayloadBytes();
    if (destination.size() != expected_bytes || target_hidden.empty() ||
        target_hidden.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument(
          "MTP draft persistent destination is invalid");
    }
    const std::size_t gpu_payload_bytes =
        gpu_snapshot->PersistentPayloadBytes();
    const std::size_t hidden_bytes =
        target_hidden.size() * sizeof(std::uint32_t);
    // The nested serializer and hidden row overwrite the entire payload.
    std::fill_n(destination.begin(), kMtpDraftPersistentHeaderBytes,
                std::uint8_t{0});
    std::copy(kMtpDraftPersistentMagic.begin(), kMtpDraftPersistentMagic.end(),
              destination.begin());
    PutLittleEndian<std::uint32_t>(destination, 8, kMtpDraftPersistentVersion);
    PutLittleEndian<std::uint32_t>(
        destination, 12,
        static_cast<std::uint32_t>(kMtpDraftPersistentHeaderBytes));
    PutLittleEndian<std::uint32_t>(destination, 16,
                                   primed ? kMtpDraftPrimedFlag : 0U);
    PutLittleEndian<std::uint32_t>(
        destination, 20, static_cast<std::uint32_t>(target_hidden.size()));
    PutLittleEndian<std::uint64_t>(
        destination, 24, static_cast<std::uint64_t>(target_hidden.size()));
    PutLittleEndian<std::uint64_t>(
        destination, 32, static_cast<std::uint64_t>(gpu_payload_bytes));
    PutLittleEndian<std::uint64_t>(destination, 40,
                                   static_cast<std::uint64_t>(expected_bytes));
    PutLittleEndian<std::uint32_t>(destination, 48,
                                   gpu_snapshot->ValidContext());
    PutLittleEndian<std::uint32_t>(destination, 52, max_context);

    const std::size_t gpu_offset = kMtpDraftPersistentHeaderBytes;
    const std::size_t hidden_offset =
        CheckedPersistentAdd(gpu_offset, gpu_payload_bytes);
    const std::size_t written = gpu_snapshot->SerializePersistent(
        destination.subspan(gpu_offset, gpu_payload_bytes));
    if (written != gpu_payload_bytes) {
      throw std::runtime_error(
          "MTP GPU persistent serializer returned the wrong byte count");
    }
    for (std::size_t index = 0; index < target_hidden.size(); ++index) {
      PutLittleEndian<std::uint32_t>(
          destination, hidden_offset + index * sizeof(std::uint32_t),
          std::bit_cast<std::uint32_t>(target_hidden[index]));
    }
    if (CheckedPersistentAdd(hidden_offset, hidden_bytes) !=
        destination.size()) {
      throw std::logic_error("MTP draft persistent serializer size mismatch");
    }
    return destination.size();
  }

  std::unique_ptr<QwenMtpGpuSnapshot> gpu_snapshot;
  std::vector<float> target_hidden;
  std::uint32_t max_context{0};
  bool primed{false};
};

}  // namespace

std::size_t QwenMtpGpuDraftBackend::SnapshotPayloadBytes() const {
  const std::size_t gpu_bytes = executor_->SnapshotPayloadBytes();
  if (target_hidden_.size() >
      (std::numeric_limits<std::size_t>::max() - gpu_bytes) / sizeof(float)) {
    throw std::overflow_error("MTP snapshot size overflows");
  }
  return gpu_bytes + target_hidden_.size() * sizeof(float);
}

std::unique_ptr<speculative::IDraftBackendSnapshot>
QwenMtpGpuDraftBackend::Snapshot() const {
  if (proposal_active_) {
    throw std::logic_error(
        "MTP snapshot requires a committed proposal boundary");
  }
  if (!primed_) {
    throw std::logic_error("MTP snapshot requires a primed draft backend");
  }
  return std::make_unique<QwenMtpDraftSnapshot>(
      executor_->SaveSnapshot(), target_hidden_, executor_->GetMaxContext(),
      primed_);
}

void QwenMtpGpuDraftBackend::RestoreSnapshot(
    const speculative::IDraftBackendSnapshot& snapshot) {
  const auto* mtp_snapshot =
      dynamic_cast<const QwenMtpDraftSnapshot*>(&snapshot);
  if (mtp_snapshot == nullptr || mtp_snapshot->gpu_snapshot == nullptr) {
    throw std::invalid_argument(
        "MTP draft snapshot is incompatible with the backend");
  }
  if (mtp_snapshot->target_hidden.size() != target_hidden_.size()) {
    throw std::invalid_argument(
        "MTP draft snapshot hidden width does not match the backend");
  }
  executor_->RestoreSnapshot(*mtp_snapshot->gpu_snapshot);
  target_hidden_ = mtp_snapshot->target_hidden;
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
  proposal_input_ = 0;
  proposal_checkpoint_ = 0;
  primed_ = mtp_snapshot->primed;
  proposal_active_ = false;
}

void QwenMtpGpuDraftBackend::RestorePersistentSnapshot(
    std::span<const std::uint8_t> payload) {
  if (payload.size() < kMtpDraftPersistentHeaderBytes ||
      !std::equal(kMtpDraftPersistentMagic.begin(),
                  kMtpDraftPersistentMagic.end(), payload.begin()) ||
      GetLittleEndian<std::uint32_t>(payload, 8) !=
          kMtpDraftPersistentVersion ||
      GetLittleEndian<std::uint32_t>(payload, 12) !=
          kMtpDraftPersistentHeaderBytes ||
      GetLittleEndian<std::uint64_t>(payload, 56) != 0) {
    throw std::invalid_argument("MTP draft persistent header is invalid");
  }
  const std::uint32_t flags = GetLittleEndian<std::uint32_t>(payload, 16);
  const std::size_t feature_width = GetLittleEndian<std::uint32_t>(payload, 20);
  const std::size_t hidden_count =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 24));
  const std::size_t gpu_payload_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 32));
  const std::size_t total_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 40));
  const std::uint32_t valid_context =
      GetLittleEndian<std::uint32_t>(payload, 48);
  const std::uint32_t max_context = GetLittleEndian<std::uint32_t>(payload, 52);

  if ((flags & ~kMtpDraftPrimedFlag) != 0 ||
      feature_width != target_hidden_.size() || hidden_count != feature_width ||
      hidden_count >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t) ||
      max_context != config_.max_context ||
      valid_context > config_.max_context || total_bytes != payload.size()) {
    throw std::invalid_argument(
        "MTP draft persistent metadata is incompatible");
  }
  const bool primed = (flags & kMtpDraftPrimedFlag) != 0;
  if (!primed && (valid_context != 0 || hidden_count != 0)) {
    throw std::invalid_argument("MTP draft unprimed payload contains state");
  }
  const std::size_t gpu_offset = kMtpDraftPersistentHeaderBytes;
  const std::size_t hidden_offset =
      CheckedPersistentAdd(gpu_offset, gpu_payload_bytes);
  const std::size_t hidden_bytes = hidden_count * sizeof(std::uint32_t);
  if (CheckedPersistentAdd(hidden_offset, hidden_bytes) != payload.size()) {
    throw std::invalid_argument("MTP draft persistent payload size is invalid");
  }

  std::vector<float> target_hidden;
  target_hidden.reserve(hidden_count);
  for (std::size_t index = 0; index < hidden_count; ++index) {
    target_hidden.push_back(std::bit_cast<float>(GetLittleEndian<std::uint32_t>(
        payload, hidden_offset + index * sizeof(std::uint32_t))));
  }

  executor_->RestorePersistentSnapshot(
      payload.subspan(gpu_offset, gpu_payload_bytes));
  // F7: the restored draft prefix must cover exactly the stored boundary.
  CheckMtpSnapshotPosition(DraftPosition(), valid_context);
  target_hidden_ = std::move(target_hidden);
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
  proposal_input_ = 0;
  proposal_checkpoint_ = 0;
  primed_ = primed;
  proposal_active_ = false;
}

void QwenMtpGpuDraftBackend::Reset() noexcept {
  executor_->Reset();
  std::ranges::fill(target_hidden_, 0.0F);
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
  proposal_input_ = 0;
  proposal_checkpoint_ = 0;
  primed_ = false;
  proposal_active_ = false;
  last_error_.clear();
  tf_round_ = 0;
  tf_pending_.clear();
  if (tf_file_ != nullptr) {
    std::fprintf(tf_file_, "# reset\n");
    std::fflush(tf_file_);
  }
}

}  // namespace superroc::hip
#endif  // defined(ENGINE_ENABLE_HIP)
