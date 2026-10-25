#include "src/models/ling3/engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <utility>

#include "src/models/ling3/mtp_gpu.hpp"
#include "src/models/ling3/reference.hpp"
#include "src/models/ling3/vision_prompt.hpp"

#if defined(LING3_WITH_ROCM)
#include "src/models/ling3/rocm_backend.hpp"
#endif

namespace gufo::models::ling3 {
namespace {

/// LS4 phase 3: how many prompt positions to prime the MTP head over.
/// LING3_MTP_PRIME overrides; 0 disables priming entirely.
int PrimeBudget() {
  static const int v = [] {
    const char* e = std::getenv("LING3_MTP_PRIME");
    if (e == nullptr || e[0] == '\0')
      return 512;
    const int n = std::atoi(e);
    return n < 0 ? 0 : n;
  }();
  return v;
}

}  // namespace

// LS3: the batched-prefill knobs. Read once; the same env vars size the ROCm
// backend's chunk scratch (rocm_backend.cpp reads LING3_PREFILL_CHUNK itself
// so buffer sizing and chunking cannot drift apart).
bool PrefillBatchEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("LING3_PREFILL_BATCH");
    return !(e != nullptr && e[0] == '0' && e[1] == '\0');
  }();
  return on;
}

std::uint32_t PrefillBatchChunk() {
  static const std::uint32_t chunk = [] {
    const char* e = std::getenv("LING3_PREFILL_CHUNK");
    long v = e != nullptr ? std::strtol(e, nullptr, 10) : 512;
    if (v < 1)
      v = 1;
    if (v > static_cast<long>(Session::kPrefillChunk)) {
      v = static_cast<long>(Session::kPrefillChunk);
    }
    return static_cast<std::uint32_t>(v);
  }();
  return chunk;
}

/// The per-sequence state the scalar reference needs. KDA conv + SSM state and
/// the MLA latent cache all live inside ReferenceModel, so this is thin.
struct Model::CpuState {
  ReferenceModel model;
  explicit CpuState(const ModelWeights& w, std::uint32_t ctx) : model(w, ctx) {}
};

Model::~Model() {
  delete cpu_state_;
}

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  if (!m->Init(model_path, options, error_msg)) {
    return nullptr;
  }
  return m;
}

bool Model::Init(const std::string& model_path, const ModelOptions& options,
                 std::string* error_msg) {
  options_ = options;
  auto reader_owner = core::GgufReader::OpenFile(model_path, error_msg);
  if (!reader_owner) {
    return false;
  }
  reader_ = std::shared_ptr<core::GgufReader>(std::move(reader_owner));
  auto weights = ModelWeights::Bind(*reader_, error_msg);
  if (!weights.has_value()) {
    return false;
  }
  weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  auto tokenizer = LingTokenizer::FromGguf(*reader_, error_msg);
  if (!tokenizer.has_value()) {
    return false;
  }
  tokenizer_ = std::make_unique<LingTokenizer>(std::move(*tokenizer));
  max_context_ = options_.max_context != 0 ? options_.max_context
                                           : weights_->config.context_length;
  if (max_context_ < 2 || max_context_ > weights_->config.context_length) {
    if (error_msg != nullptr) {
      *error_msg = "requested context does not fit the model";
    }
    return false;
  }

  switch (options_.backend) {
    case BackendKind::kCpuReference:
      cpu_state_ = new CpuState(*weights_, max_context_);
      break;
#if defined(LING3_WITH_ROCM)
    case BackendKind::kRocm: {
      std::string rocm_error;
      const rocm::ExpertPlacement placement{
          .stream = options_.stream_experts,
      };
      if (!rocm::Init(*weights_, *reader_, placement, &resident_bytes_,
                      &rocm_error)) {
        if (error_msg != nullptr) {
          *error_msg = "ROCm backend: " + rocm_error;
        }
        return false;
      }
      if (!rocm::ExecutorReady()) {
        if (error_msg != nullptr) {
          *error_msg = "Ling-3.0-flash ROCm backend: " + rocm::MissingPieces() +
                       ". Use the CPU reference backend "
                       "(BackendKind::kCpuReference), which is the oracle "
                       "the GPU path is checked against.";
        }
        return false;
      }
      // LS4: the MTP draft head. Two switches reach it, and both must load it:
      //   LING3_MTP=1        -- speculative decoding (phase 2)
      //   LING3_MTP_PROBE=1  -- the phase-1 acceptance probe
      // With neither set this is a no-op and nothing extra is resident.
      // Init() is idempotent, so asking for both costs one upload.
      if (mtp::SpeculativeEnabled()) {
        std::size_t head_bytes = 0;
        std::string head_error;
        if (!mtp::Init(*weights_, mtp::HeadNorm::kOutputNorm, &head_bytes,
                       &head_error)) {
          // Speculation is a speed feature, never a correctness one: refuse to
          // load rather than serve something subtly wrong.
          if (error_msg != nullptr) {
            *error_msg = "Ling-3.0-flash MTP head: " + head_error;
          }
          return false;
        }
        if (resident_bytes_ != 0) {
          resident_bytes_ += head_bytes;
        }
        mtp::InstallSpecReporter();
        std::fprintf(stderr,
                     "mtp speculative decoding: ON (head %.1f MiB resident)\n",
                     static_cast<double>(head_bytes) / 1048576.0);
      }
      mtp::ProbeInit(*weights_, max_context_, error_msg);
      break;
    }
#else
    case BackendKind::kRocm:
      if (error_msg != nullptr) {
        *error_msg =
            "this build has no ROCm backend for ling3; use the CPU reference";
      }
      return false;
#endif
  }
  return true;
}

std::shared_ptr<Session> Model::CreateSession(core::SessionMode mode,
                                              std::uint32_t max_context,
                                              std::string* error_msg) {
  (void)mode;
  if (max_context != 0 && max_context > max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "session context exceeds the model context";
    }
    return nullptr;
  }
  std::shared_ptr<Session> s(new Session(shared_from_this(), max_context_));
  s->state_ = NewState(error_msg);
  if (s->state_ == nullptr) {
    return nullptr;
  }
  return s;
}

void* Model::NewState(std::string* error_msg) const {
  switch (options_.backend) {
    case BackendKind::kCpuReference:
      (void)error_msg;
      return new CpuState(*weights_, max_context_);
#if defined(LING3_WITH_ROCM)
    case BackendKind::kRocm:
      // NewState returns the backend's own State*; Model::Step takes a void*.
      // max_context_ is REQUIRED, not optional: the ROCm latent cache is sized
      // by it (mla_block is called with n_pos = pos + 1, so the cache has to
      // hold max_ctx positions per layer, not one).
      return static_cast<void*>(rocm::NewState(max_context_));
#else
    case BackendKind::kRocm:
      if (error_msg != nullptr) {
        *error_msg = "no ROCm backend in this build";
      }
      return nullptr;
#endif
  }
  return nullptr;
}

void* Model::ProbeState(const Session* session) const noexcept {
  // Model is a friend of Session, so the private blob is reachable here and
  // nowhere else. Owned by the Session: never free it.
  if (session == nullptr || session->model_.get() != this) {
    return nullptr;
  }
  return session->state_;
}

void Model::DestroyState(void* state) const noexcept {
  if (state == nullptr) {
    return;
  }
#if defined(LING3_WITH_ROCM)
  if (options_.backend == BackendKind::kRocm) {
    rocm::DestroyState(static_cast<rocm::State*>(state));
    return;
  }
#endif
  delete static_cast<CpuState*>(state);
}

void Model::ResetState(void* state) const {
  if (state == nullptr) {
    return;
  }
#if defined(LING3_WITH_ROCM)
  if (options_.backend == BackendKind::kRocm) {
    // The backend owns the layout of this blob, so it clears it. Rewinding
    // `pos` alone left the conv history, the SSM state and the latent cache
    // holding the PREVIOUS prompt's values, so every prompt change started from
    // stale recurrent state and produced plausible-looking wrong tokens.
    rocm::ResetState(static_cast<rocm::State*>(state));
    return;
  }
#endif
  if (auto* cpu = static_cast<CpuState*>(state)) {
    cpu->model.Reset();
  }
}

bool Model::Step(std::int32_t token, void* state, std::span<float> logits,
                 std::string* error_msg) {
  if (state == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "session state is null";
    }
    return false;
  }
  if (options_.backend == BackendKind::kCpuReference) {
    auto* cpu = static_cast<CpuState*>(state);
    // The reference sizes its own logits vector; copy out the prefix asked for.
    std::vector<float> full;
    if (!cpu->model.Step(token, full, error_msg)) {
      return false;
    }
    const std::size_t n = std::min<std::size_t>(logits.size(), full.size());
    if (n > 0) {
      std::copy_n(full.data(), n, logits.data());
    }
    return true;
  }
#if defined(LING3_WITH_ROCM)
  return rocm::Step(token, static_cast<rocm::State*>(state), logits, error_msg);
#else
  if (error_msg != nullptr) {
    *error_msg = "no ROCm backend in this build";
  }
  return false;
#endif
}

bool Model::PrefersBatchPrefill() const noexcept {
#if defined(LING3_WITH_ROCM)
  return options_.backend == BackendKind::kRocm && !options_.stream_experts &&
         rocm::BatchCapable();
#else
  return false;
#endif
}

bool Model::StepBatch(const std::int32_t* tokens, int n, void* state,
                      std::span<float> logits, std::span<float> hidden_out,
                      std::span<float> hidden_rows_out,
                      std::string* error_msg) {
  if (state == nullptr || tokens == nullptr || n <= 0) {
    if (error_msg != nullptr) {
      *error_msg = "session state is null or empty batch";
    }
    return false;
  }
  if (options_.backend == BackendKind::kCpuReference) {
    // The oracle stays scalar: single-token steps, keeping only the last
    // position's logits (what EvaluateBatch's contract exposes).
    auto* cpu = static_cast<CpuState*>(state);
    std::vector<float> full;
    for (int i = 0; i < n; ++i) {
      if (!cpu->model.Step(tokens[i], full, error_msg)) {
        return false;
      }
    }
    const std::size_t take = std::min<std::size_t>(logits.size(), full.size());
    if (take > 0) {
      std::copy_n(full.data(), take, logits.data());
    }
    return true;
  }
#if defined(LING3_WITH_ROCM)
  // LS4: logit_rows MUST stay 0 here. Prefill wants the LAST position logits,
  // which is LS3s span-size-driven path; passing a non-zero logit_rows would
  // switch ForwardBatch to its per-row branch and return ROW 0s logits
  // instead -- a silently wrong prompt representation and wrong text. The
  // hidden request is independent of that choice and costs nothing.
  return rocm::ForwardBatch(tokens, n, static_cast<rocm::State*>(state), logits,
                            error_msg, 0, hidden_out, hidden_rows_out);
#else
  (void)logits;
  if (error_msg != nullptr) {
    *error_msg = "no ROCm backend in this build";
  }
  return false;
#endif
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->TokenText(token);
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return tokenizer_->IsStop(token);
}

std::string Model::ModelName() const {
  return "Ling-3.0-flash (bailingmoe3)";
}

std::size_t Model::ResidentBytes() const noexcept {
  if (resident_bytes_ != 0) {
    return resident_bytes_;
  }
  // CPU backend: the GGUF stays mmapped, so residency is the mapped payload.
  std::size_t total = 0;
  for (const LayerWeights& l : weights_->layers) {
    total += l.attn_norm.tensor.SizeBytes() + l.ffn_norm.tensor.SizeBytes();
  }
  return total;
}

std::size_t Model::SessionBytes(core::SessionMode,
                                std::uint32_t) const noexcept {
  // Recurrent state and the latent cache scale with the context, so an exact
  // figure would need a walk; the reference reports a per-sequence constant
  // for the recurrent part and the caller bounds the cache separately.
  if (options_.backend == BackendKind::kRocm) {
    return sizeof(CpuState);  // device-side; see rocm::ResidentBytesFor
  }
  return sizeof(CpuState);
}

// ---------------------------------------------------------------------------

Session::Session(std::shared_ptr<Model> model, std::uint32_t max_context)
    : model_(std::move(model)), max_context_(max_context) {}

void Session::FlushSpecReport() {
  // steps == 0 means we are between prefill CHUNKS of the same request (no
  // decode has run yet), not at a request boundary -- keep accumulating the
  // prompt length and report nothing. Only a flush with real decode stats ends
  // a request, and that is what zeroes the length for the next one.
  if (spec_ == nullptr || spec_stats_.steps == 0) {
    return;
  }
  ++spec_report_req_;
  mtp::ReportSpecSession(spec_stats_, mtp::SpecDraftLength(spec_),
                         spec_report_req_, spec_report_prompt_tokens_);
  spec_stats_ = {};
  spec_report_prompt_tokens_ = 0;
}

Session::~Session() {
  // The last request has no following EvaluateBatch to flush it.
  FlushSpecReport();
  mtp::DestroySpecState(spec_);
  spec_ = nullptr;
  if (state_ != nullptr && model_) {
    model_->DestroyState(state_);
  }
}

int Session::SpecDraftLength() const noexcept {
  return mtp::SpecDraftLength(spec_);
}

bool Session::CheckCancel(std::string* error_msg) const {
  if (cancel_ && cancel_()) {
    if (error_msg != nullptr) {
      *error_msg = "cancelled";
    }
    return true;
  }
  return false;
}

void Session::Reset() {
  vision::Clear();
  vision_guard_ = false;
  // A reset ends the request that was in flight (nothing calls this on the
  // serve path today -- Session::Reset has only test callers -- but if it is
  // ever wired up the row must still land).
  FlushSpecReport();
  tokens_.clear();
  logits_.clear();
  position_ = 0;
  pending_h_.clear();
  pending_h_valid_ = false;
  if (spec_ != nullptr) {
    // The head's cache and the policy window are both per-sequence: a new
    // sequence must not inherit the previous one's latent cache or its
    // acceptance history. RV4 (Jay build28f): rewind both in place -- the
    // head's allocations are identical every request, so destroying and
    // re-creating them per request was pure churn.
    mtp::ResetSpecState(spec_);
  }
  if (state_ != nullptr && model_) {
    model_->ResetState(state_);
  }
}

bool Session::EvaluateSpeculative(std::int32_t base,
                                  std::vector<std::int32_t>* consumed,
                                  std::string* error_msg) {
  if (consumed != nullptr)
    consumed->clear();
  if (position_ >= max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "context length exceeded";
    }
    return false;
  }

  // Speculation needs a position with a hidden state to draft from, an
  // initialised head, and a batch-capable trunk. Anything else is a plain
  // Evaluate(base), which is always correct.
  bool can_spec = mtp::SpeculativeEnabled() && position_ > 0 &&
                  model_->options_.backend == BackendKind::kRocm;
#if defined(LING3_WITH_ROCM)
  can_spec = can_spec && rocm::BatchCapable() && mtp::Ready();
  if (can_spec && spec_ == nullptr) {
    spec_ = mtp::NewSpecState(max_context_, spec_policy_, error_msg);
    can_spec = spec_ != nullptr;
  }
#endif

  if (!can_spec) {
    if (!Evaluate(base, error_msg))
      return false;
    if (consumed != nullptr)
      consumed->assign(1, base);
    return true;
  }

  std::vector<std::int32_t> block;
  block.reserve(static_cast<std::size_t>(spec_policy_.n_max) + 1);
  std::vector<float> next_logits(model_->config().vocab_size);
  std::string spec_error;
  // Pass the stop predicate down so a block that straddles EOS is rewound to
  // end at it rather than leaving the checkpoint ahead of the published tokens.
  const auto is_stop = [this](std::int32_t t) {
    return model_->IsStopToken(t);
  };
  if (!mtp::SpecStep(state_, spec_, base, pending_h_, pending_h_valid_, &block,
                     &next_logits, &pending_h_, &spec_stats_, is_stop,
                     &spec_error)) {
    pending_h_valid_ = false;
    // Fail soft. SpecStep leaves the trunk untouched when it cannot run, so a
    // plain decode of `base` is still exactly the right thing to do.
    if (error_msg != nullptr && std::getenv("LING3_MTP_VERBOSE") != nullptr) {
      *error_msg = "spec step fell back to a plain decode: " + spec_error;
    }
    if (!Evaluate(base, error_msg))
      return false;
    if (consumed != nullptr)
      consumed->assign(1, base);
    return true;
  }

  for (const std::int32_t t : block) {
    ++position_;
    tokens_.push_back(t);
  }
  if (consumed != nullptr)
    *consumed = block;
  logits_ = std::move(next_logits);
  if (logits_.size() < model_->config().vocab_size) {
    logits_.resize(model_->config().vocab_size);
  }
  return true;
}

std::size_t Session::AllocatedBytes() const noexcept {
  return tokens_.capacity() * sizeof(std::int32_t) +
         logits_.capacity() * sizeof(float) +
         (state_ != nullptr && model_
              ? model_->SessionBytes(core::SessionMode::kAutoregressive, 0)
              : 0);
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  // Keep the longest common prefix; rewind the state, then replay the rest.
  std::size_t common = 0;
  while (common < tokens_.size() && common < prompt.size() &&
         tokens_[common] == prompt[common]) {
    ++common;
  }
  in_prefill_ = true;
  if (common != tokens_.size()) {
    if (common == 0) {
      Reset();
    } else {
      // Rewinding means replaying from scratch: the recurrent state has no
      // cheap truncate, and speed is not a gate tonight.
      Reset();
      common = 0;
    }
  }
  for (std::size_t i = common; i < prompt.size(); ++i) {
    if (CheckCancel(error_msg)) {
      in_prefill_ = false;
      return false;
    }
    if (!Evaluate(prompt[i], error_msg)) {
      in_prefill_ = false;
      return false;
    }
  }
  in_prefill_ = false;
  return true;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  if (position_ >= max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "context length exceeded";
    }
    return false;
  }
#if defined(LING3_WITH_ROCM)
  // ---- LS4 MTP probe (LING3_MTP_PROBE=1) --------------------------------
  // This has to run BEFORE Step, and that is the whole subtlety.
  //
  // The draft batch is { token = the token about to be fed, pos = its own
  // position, embd = the target's hidden state at pos-1 } -- the fork's
  // { id_last, p_next, pending_h } (speculative.cpp:1766-1793). At the top of
  // Evaluate all three are on hand: `token` and `position_` as arguments, and
  // `state->t`, which the PREVIOUS Step left holding h(position_-1).
  //
  // Running it after Step instead, with position_-1 and the token just
  // consumed, is off by one in BOTH the position and the token. That was the
  // first cut here and it MEASURED 8.3 % acceptance against the oracle's
  // 85-97.5 % -- the block was right, the call site was not. Replay through
  // mtp_gpu_check (37/37 exact) is unaffected: it feeds the triples directly.
  //
  // Prefill is included on purpose: the MTP block keeps its own latent cache
  // and only ever sees positions a draft wrote, so skipping the prompt would
  // leave it attending over an empty cache. This mirrors the fork, where
  // process() replays the whole verify batch through ctx_dft.
  //
  // The draft is NEVER fed back, so generation is identical with the probe on
  // or off -- which is what makes the acceptance number trustworthy.
  if (mtp::ProbeEnabled() && state_ != nullptr) {
    std::vector<float> hidden;
    const bool have_hidden =
        position_ > 0 &&
        (hidden.resize(model_->config().hidden_size),
         rocm::ReadHidden(static_cast<rocm::State*>(state_), hidden, nullptr));
    mtp::ProbeStep(state_, token,
                   std::span<const float>(have_hidden ? hidden.data() : nullptr,
                                          have_hidden ? hidden.size() : 0),
                   position_, have_hidden, !in_prefill_);
  }
#endif

  // Size before handing over the span: Step writes vocab floats through it and
  // a default-constructed vector would give it an empty range.
  logits_.resize(model_->config().vocab_size);
  if (!model_->Step(token, state_, std::span<float>(logits_), error_msg)) {
    logits_.clear();
    return false;
  }
  ++position_;
  tokens_.push_back(token);

#if defined(LING3_WITH_ROCM)
  // LS4: keep pending_h_ in step with the batched path. Step() leaves the final
  // hidden in state->t, so ReadHidden is valid here.
  if (mtp::SpeculativeEnabled()) {
    if (pending_h_.size() != model_->config().hidden_size) {
      pending_h_.resize(model_->config().hidden_size);
    }
    pending_h_valid_ = rocm::ReadHidden(
        static_cast<rocm::State*>(state_),
        std::span<float>(pending_h_.data(), pending_h_.size()), nullptr);
  }
#endif
  return true;
}

bool Session::EvaluateBatch(std::span<const std::int32_t> tokens,
                            std::string* error_msg) {
  if (position_ == 0 && !vision_guard_) {
    vision::Clear();
  }
  // SPEC6 item 4: the per-REQUEST boundary on the serve path. The scheduler
  // keeps ONE Session resident for the whole server -- ~Session runs once at
  // shutdown and Reset is never called -- so the only thing that fires once per
  // prompt is this call. The row that flushes here is the PREVIOUS prompt's,
  // which is why the label is captured below and printed before it is
  // overwritten.
  FlushSpecReport();
  // This call is one prefill CHUNK. Accumulate so the row that eventually
  // flushes carries the request's whole prompt length (SPEC6 item 4: "LM1_COST
  // per length" -- the length is the row's identity).
  spec_report_prompt_tokens_ += static_cast<int>(tokens.size());
  if (tokens.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "empty batch";
    }
    return false;
  }
  if (position_ + tokens.size() > max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "context length exceeded";
    }
    return false;
  }
  logits_.resize(model_->config().vocab_size);
  const std::size_t H = model_->config().hidden_size;
  const std::size_t n_rows = tokens.size();

  // LS4 phase 3: prime the MTP head over the prompt.
  //
  // The head is the only writer of its own latent cache, so without this its
  // attention at the first speculative step runs over a cache whose prompt
  // rows are zeros. The fork fills them by replaying the verify batch through
  // ctx_dft (`process()`); here the prefill forward already normalised every
  // row, so all that is needed is to ask for them and feed them through the
  // head. PrimeNext skips the LM head -- the token is discarded and only the
  // hidden state matters.
  //
  // Bounded by LING3_MTP_PRIME (default 512, one prefill chunk): only a chunk
  // that fits entirely in the budget is primed, so the cost is capped and a
  // long prompt primes its TAIL, which is where the context that matters is.
  // h_prev for the chunk's first row is the carried pending_h_, and for every
  // later row it is the previous row's hidden out of this same batch.
  // The head state is created here rather than lazily in
  // EvaluateSpeculative: priming happens during prefill, which runs first.
  // Creating it twice would throw away a cache that was just primed.
  if (spec_ == nullptr && mtp::SpeculativeEnabled() && mtp::Ready() &&
      model_->options_.backend == BackendKind::kRocm) {
    spec_ = mtp::NewSpecState(max_context_, spec_policy_, nullptr);
  }
  // LM1 (DECISIONS 00:30): LING3_MTP_PRIME_BATCH=1 primes the WHOLE chunk with
  // one batched head forward instead of PrimeNext per row. It ignores the
  // LING3_MTP_PRIME budget on purpose -- the point is that priming the entire
  // prompt is now cheap enough to always do. Default off until the gates land.
  const int prime_budget = PrimeBudget();
  const bool prime_batch = spec_ != nullptr && mtp::PrimeBatchEnabled() &&
                           model_->options_.backend == BackendKind::kRocm;
  const bool prime =
      prime_batch || (prime_budget > 0 && spec_ != nullptr &&
                      n_rows <= static_cast<std::size_t>(prime_budget));
  spec_rows_.resize(prime ? H * n_rows : H);
  std::string step_error;
  const bool ok = model_->StepBatch(
      tokens.data(), static_cast<int>(n_rows), state_,
      std::span<float>(logits_), std::span<float>(spec_rows_.data(), H),
      prime ? std::span<float>(spec_rows_.data(), spec_rows_.size())
            : std::span<float>(),
      &step_error);
  if (!ok) {
    logits_.clear();
    pending_h_.clear();
    pending_h_valid_ = false;
    if (error_msg != nullptr)
      *error_msg = step_error;
    return false;
  }

  if (prime) {
    if (prime_batch) {
      // One batched forward over the chunk.
      //
      // RV5 fix 2: the head pairs token t with the TRUNK hidden at t-1, which
      // is what the fork does (common/speculative.cpp:1578-1585 -- it shifts
      // the target embeddings one to the right before decoding the draft). So
      // row 0 of a chunk sees the PREVIOUS chunk's last trunk row, and row r
      // sees this chunk's trunk row r-1. Position 0 has no t-1 and sees zeros,
      // which is exactly the fork's `pos_needed < 0` branch. The first version
      // passed spec_rows_ as-is, so every token saw ITS OWN hidden instead of
      // its predecessor's.
      //
      // pending_h_ still holds the PREVIOUS chunk's last trunk row here: it is
      // reassigned from spec_rows_ at the bottom of this function, after this
      // call. That ordering is load-bearing.
      std::vector<float> h_shifted(H * n_rows, 0.0F);
      if (position_ > 0 && pending_h_valid_ && pending_h_.size() >= H) {
        std::copy(pending_h_.begin(),
                  pending_h_.begin() + static_cast<std::ptrdiff_t>(H),
                  h_shifted.begin());
      }
      if (n_rows > 1) {
        // h_shifted rows 1..n_rows-1 <- spec_rows_ rows 0..n_rows-2.
        std::copy(
            spec_rows_.begin(),
            spec_rows_.begin() + static_cast<std::ptrdiff_t>(H * (n_rows - 1)),
            h_shifted.begin() + static_cast<std::ptrdiff_t>(H));
      }
      std::string prime_error;
      if (!mtp::PrimeBatch(mtp::SpecHead(spec_), tokens.data(),
                           h_shifted.data(), static_cast<int>(n_rows),
                           position_, &prime_error)) {
        // Priming is an optimisation for the drafts, never a correctness
        // requirement: a failure leaves the head unprimed exactly as
        // LING3_MTP_PRIME=0 would, and the run continues. PrimeBatch has
        // already cleared the window base, so nothing half-filled is attended.
        (void)prime_error;
      }
    } else {
      // Row-by-row fallback. Same RV5 fix 2 pairing: feed each row the trunk
      // hidden at the PREVIOUS position, not the head's own output (which is
      // what chaining h_next did, and which the fork never does during
      // prefill). Row 0 takes pending_h_ / zeros exactly as above.
      std::vector<float> h_prev_row(H, 0.0F);
      if (position_ > 0 && pending_h_valid_ && pending_h_.size() >= H) {
        std::copy(pending_h_.begin(),
                  pending_h_.begin() + static_cast<std::ptrdiff_t>(H),
                  h_prev_row.begin());
      }
      std::vector<float> h_next(H);
      for (std::size_t r = 0; r < n_rows; ++r) {
        if (!mtp::PrimeNext(mtp::SpecHead(spec_), tokens[r],
                            std::span<const float>(h_prev_row.data(), H),
                            static_cast<std::uint32_t>(position_ + r),
                            std::span<float>(h_next.data(), H), nullptr)) {
          break;
        }
        // Advance the pairing: the next row sees this row's trunk hidden.
        if (r + 1 < n_rows) {
          std::copy(
              spec_rows_.begin() + static_cast<std::ptrdiff_t>(r * H),
              spec_rows_.begin() + static_cast<std::ptrdiff_t>((r + 1) * H),
              h_prev_row.begin());
        }
      }
    }
  }

  // Take the LAST row that was actually staged. When prime is false spec_rows_
  // holds exactly one row (out_hidden), so the offset is 0 -- slicing it as
  // [H*(n_rows-1), H*n_rows) read past the end and SEGFAULTED the first chat
  // request with drafting off. Found by running the drafting-OFF control, which
  // is the only run that exercises this path.
  const std::size_t last_row = prime ? (n_rows - 1) : 0;
  pending_h_.assign(
      spec_rows_.begin() + static_cast<std::ptrdiff_t>(H * last_row),
      spec_rows_.begin() + static_cast<std::ptrdiff_t>(H * (last_row + 1)));
  pending_h_valid_ = true;
  position_ += static_cast<std::uint32_t>(tokens.size());
  tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
  return true;
}

std::int32_t Session::Argmax() const noexcept {
  // Ties resolve to the lowest index, matching the device Argmax and the
  // fork's greedy path.
  std::int32_t best = 0;
  for (std::size_t i = 1; i < logits_.size(); ++i) {
    if (logits_[i] > logits_[static_cast<std::size_t>(best)]) {
      best = static_cast<std::int32_t>(i);
    }
  }
  return best;
}

void Session::BindVision(const vision::PreparedPrompt& prepared) {
  vision_guard_ = true;
  vision::Arm(prepared);
}

void Session::UnbindVision() {
  vision_guard_ = false;
  vision::Clear();
}

bool Session::PrefillVision(const vision::PreparedPrompt& prepared,
                            std::string* error_msg) {
  if (prepared.tokens.empty()) {
    if (error_msg != nullptr)
      *error_msg = "empty vision prompt";
    return false;
  }
  BindVision(prepared);
  bool ok = true;
  const std::size_t n = prepared.tokens.size();
  for (std::size_t off = 0; off < n;) {
    const std::size_t take = std::min<std::size_t>(512, n - off);
    if (!EvaluateBatch(
            std::span<const std::int32_t>(prepared.tokens.data() + off, take),
            error_msg)) {
      ok = false;
      break;
    }
    off += take;
  }
  vision_guard_ = false;
  return ok;
}

}  // namespace gufo::models::ling3
