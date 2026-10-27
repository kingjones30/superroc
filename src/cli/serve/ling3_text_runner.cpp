#include "src/cli/serve/ling3_text_runner.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/image.hpp"
#include "src/models/ling3/chat_template.hpp"
#include "src/models/ling3/mtp_gpu.hpp"
#include "src/models/ling3/vision_tower.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/vision/prompt.hpp"

namespace superroc::server {
namespace {

namespace lgchat = superroc::models::ling3::chat;

/// Stability identity for a saved Ling continuation. The recurrent state and
/// the MLA latent cache are both layout-sensitive, so this covers the model
/// family, the tokenizer, the template and the payload version.
constexpr std::string_view kLing3StateAbi = "ling3-bailingmoe3-session-v1";
constexpr std::uint32_t kLing3PayloadVersion = 1;

struct LingImagePlan final : TextPromptContext {
  std::vector<std::vector<std::uint8_t>> rgb;
  std::vector<int> width;
  std::vector<int> height;
  std::vector<std::pair<int, int>> grids;
  models::ling3::vision::PreparedPrompt prepared;
};

std::string ToLingRoleName(tokenization::ChatRole role) {
  switch (role) {
    case tokenization::ChatRole::kSystem:
    case tokenization::ChatRole::kDeveloper:
      return "system";
    case tokenization::ChatRole::kUser:
      return "user";
    case tokenization::ChatRole::kAssistant:
      return "assistant";
    case tokenization::ChatRole::kTool:
      return "tool";
  }
  return "user";
}

}  // namespace

TextRunnerMeasuredResources Ling3RunnerState::MeasuredResources()
    const noexcept {
  TextRunnerMeasuredResources r;
  r.per_request_state_bytes = session_ ? session_->AllocatedBytes() : 0;
  return r;
}

void Ling3RunnerState::Invalidate() noexcept {
  vision_.reset();
  if (session_) {
    session_->Reset();
    position_ = 0;
  }
}

Ling3TextRunner::Ling3TextRunner(
    std::shared_ptr<models::ling3::Model> model, std::uint32_t max_context,
    std::string disk_cache_directory,
    std::shared_ptr<models::ling3::vision::VisionTower> vision)
    : model_(std::move(model)),
      max_context_(max_context != 0 ? max_context : model_->MaxContext()),
      disk_cache_directory_(std::move(disk_cache_directory)),
      vision_(std::move(vision)) {}

Ling3TextRunner::~Ling3TextRunner() = default;

TextRunnerDescriptor Ling3TextRunner::Descriptor() const {
  TextRunnerDescriptor d;
  d.model_id = model_->ModelName();
  d.state_abi = std::string(kLing3StateAbi);
  d.max_context = max_context_;
  d.capabilities.incremental_prefill = true;
  // The snapshot family in TextModelRunner throws unless the runner implements
  // it, so this stays false rather than advertising a capability that would
  // then fault on use.
  d.capabilities.snapshot = false;
  d.capabilities.fork = false;
  // LS4 phase 3: prompt-prefix reuse is turned OFF while speculative decoding
  // is on. A speculative step advances the runner position by 1..n per
  // scheduler step, and the reuse path validates that position against
  // prompt+generated on the NEXT request; with drafting on it can land outside
  // that history and the request dies with "checkpoint is outside executed
  // token history" (observed on gate (a) prompt 3). Turning the capability off
  // is the safe resolution -- it only costs prompt-cache reuse, and only in the
  // drafting-on configuration. With drafting off nothing changes.
  d.capabilities.prefix_reuse = !models::ling3::mtp::SpeculativeEnabled();
  // LS4 (LING3_MTP=1): the scheduler only calls DecodeStep -- the hook the MTP
  // head implements -- when this capability is advertised. Left false by
  // default, so the default build decodes one token at a time exactly as
  // before. batched_multi_token_decode stays false: this runner does not
  // override DecodeBatch, and advertising it would let the scheduler take a
  // path that has no implementation here.
  d.capabilities.multi_token_decode = models::ling3::mtp::SpeculativeEnabled();
  // Jay 2026-10-01: no persistence -- superroc's ValidateRunner refuses persistence
  // without snapshot+fork ("text runner persistence requires snapshot and fork
  // support"), which is exactly what this runner declares false above. Leave it
  // std::nullopt.
  return d;
}

TextRunnerResourceClaim Ling3TextRunner::ResourceClaim() const {
  TextRunnerResourceClaim c;
  std::size_t resident = model_->ResidentBytes();
  std::size_t scratch = model_->DeferredScratchBytes();
  if (vision_) {
    resident += vision_->HostBytes();
    scratch += vision_->ScratchBytes();
  }
  c.resident_weights_bytes = resident;
  // Jay 2026-10-01: 0 made ValidateRunner throw "request-state claim exceeds
  // state capacity" (capacity / per_request = 0 < 1 state). std::nullopt = "not
  // yet measurable": the pool skips the check and CreateState fails loudly if
  // memory is really short.
  c.state_capacity_bytes = std::nullopt;
  c.per_request_state_bytes =
      model_->SessionBytes(core::SessionMode::kAutoregressive, max_context_);
  c.temporary_scratch_bytes = scratch;
  c.retained_snapshot_capacity_bytes = std::size_t{0};
  c.requires_device_runtime_lock =
      model_->backend() == models::ling3::BackendKind::kRocm;
  return c;
}

std::vector<TextExecutionPlan> Ling3TextRunner::SupportedPlans() const {
  std::vector<TextExecutionPlan> plans;
  plans.push_back(TextExecutionPlan{TextExecutionPlanKind::kSerial, 1});
  return plans;
}

std::vector<TextRunnerToken> Ling3TextRunner::Tokenize(
    std::string_view text) const {
  const auto ids = model_->tokenizer().Encode(text);
  return std::vector<TextRunnerToken>(ids.begin(), ids.end());
}

std::optional<std::vector<TextRunnerToken>> Ling3TextRunner::RenderAndTokenize(
    const ChatRequest& request) const {
  std::vector<lgchat::Message> messages;
  messages.reserve(request.messages.size());
  for (const auto& m : request.messages) {
    lgchat::Message out;
    out.role = ToLingRoleName(m.role);
    out.content = m.content;
    out.reasoning = m.thought;
    messages.push_back(std::move(out));
  }

  std::vector<lgchat::Tool> tools;
  tools.reserve(request.tools.size());
  for (const auto& t : request.tools) {
    lgchat::Tool tool;
    tool.name = t.name;
    tool.description = t.description;
    // definition_json preserves the client's field order; fall back to the
    // structured fields when the client did not supply it.
    tool.parameters_json = t.parameters_json.empty() ? "{}" : t.parameters_json;
    tools.push_back(std::move(tool));
  }

  std::string error;
  auto rendered = lgchat::RenderChat(messages, tools,
                                     /*enable_thinking=*/false,
                                     /*add_generation_prompt=*/true, &error);
  if (!rendered.has_value()) {
    return std::nullopt;
  }
  const auto ids = model_->tokenizer().Encode(*rendered);
  return std::vector<TextRunnerToken>(ids.begin(), ids.end());
}

std::string Ling3TextRunner::Decode(
    std::span<const TextRunnerToken> tokens) const {
  const std::vector<std::int32_t> ids(tokens.begin(), tokens.end());
  return model_->tokenizer().Decode(ids);
}

std::unique_ptr<TextRunnerState> Ling3TextRunner::CreateState() const {
  std::string error;
  auto session = model_->CreateSession(core::SessionMode::kAutoregressive,
                                       max_context_, &error);
  if (!session) {
    throw std::runtime_error("ling3: cannot create a session: " + error);
  }
  return std::make_unique<Ling3RunnerState>(std::move(session));
}

TextPrefillStep Ling3TextRunner::Prefill(
    TextRunnerState& state, std::span<const TextRunnerToken> prompt,
    std::size_t offset, std::size_t max_input_tokens) const {
  Ling3RunnerState& s = RequireLing3State(state);
  models::ling3::Session& session = s.session();

  if (offset == 0) {
    session.Reset();
    s.set_position(0);
    // Reset drops the M-RoPE arm. A cache miss prefills from the start, so the
    // image rows and the position table have to be published again before the
    // first embedding lookup.
    if (const auto* vis = s.Vision()) {
      session.BindVision(*vis);
    }
  } else if (offset != session.Position()) {
    throw std::logic_error(
        "ling3: prefill offset does not match the session position");
  }

  // LS3 2026-10-01: prefill in ForwardBatch chunks (up to the batch chunk
  // cap) when the model supports it; LING3_PREFILL_BATCH=0 falls back to the
  // token-at-a-time path (the decode step, unchanged).
  const bool use_batch =
      models::ling3::PrefillBatchEnabled() && model_->PrefersBatchPrefill();
  const std::size_t cap =
      use_batch ? std::min<std::size_t>(max_input_tokens,
                                        models::ling3::PrefillBatchChunk())
                : std::min<std::size_t>(max_input_tokens,
                                        models::ling3::Session::kPrefillChunk);
  const std::size_t end = std::min(prompt.size(), offset + cap);
  if (use_batch) {
    for (std::size_t i = offset; i < end;) {
      const std::size_t take = std::min<std::size_t>(cap, end - i);
      std::vector<std::int32_t> chunk(
          prompt.begin() + static_cast<std::ptrdiff_t>(i),
          prompt.begin() + static_cast<std::ptrdiff_t>(i + take));
      std::string error;
      if (!session.EvaluateBatch(chunk, &error)) {
        std::fprintf(stderr,
                     "LS4 Prefill: EvaluateBatch failed: %s (n=%zu pos=%u)\n",
                     error.c_str(), take, session.Position());
        throw std::runtime_error("ling3 prefill failed: " + error);
      }
      i += take;
    }
  } else {
    for (std::size_t i = offset; i < end; ++i) {
      std::string error;
      if (!session.Evaluate(static_cast<std::int32_t>(prompt[i]), &error)) {
        std::fprintf(stderr, "LS4 Prefill: Evaluate failed: %s\n",
                     error.c_str());
        throw std::runtime_error("ling3 prefill failed: " + error);
      }
    }
  }
  s.set_position(session.Position());

  TextPrefillStep step;
  step.consumed_tokens = end - offset;
  // Decode-ready only at the prompt frontier: the scheduler requires
  // decode_ready == (offset + consumed == prompt.size()). Jay 2026-10-01: the
  // old "any token evaluated" rule made every prompt longer than kPrefillChunk
  // (16) fail with "text runner returned an inconsistent prefill boundary" --
  // i.e. every /v1/chat/completions request (the chat template alone is > 16
  // tokens).
  step.decode_ready =
      end == prompt.size() && end > offset && !session.Logits().empty();
  return step;
}

TextDecodeSelection Ling3TextRunner::SelectNext(
    TextRunnerState& state, sampling::SamplerState& sampler) const {
  Ling3RunnerState& s = RequireLing3State(state);
  const auto logits = s.session().Logits();
  if (logits.empty()) {
    std::fprintf(stderr, "LS4 SelectNext: no logits (pos=%u)\n",
                 RequireLing3State(state).session().Position());
    throw std::runtime_error("ling3: SelectNext with no logits");
  }
  const auto token = static_cast<TextRunnerToken>(sampler.Sample(logits));
  if (s.stop_at_eos() &&
      model_->IsStopToken(static_cast<std::int32_t>(token))) {
    return TextDecodeSelection{true, 0, {}};
  }
  return TextDecodeSelection{
      false, token, model_->TokenText(static_cast<std::int32_t>(token))};
}

std::optional<TextDecodeSelection> Ling3TextRunner::PreviewFirstToken(
    TextRunnerState& state, sampling::SamplerState& sampler) const {
  // SelectNext only samples the logits the prompt's last forward left in the
  // session; it advances nothing, and `sampler` is the scheduler's copy.
  return SelectNext(state, sampler);
}

void Ling3TextRunner::Advance(TextRunnerState& state,
                              TextRunnerToken token) const {
  Ling3RunnerState& s = RequireLing3State(state);
  std::string error;
  if (!s.session().Evaluate(static_cast<std::int32_t>(token), &error)) {
    throw std::runtime_error("ling3 advance failed: " + error);
  }
  s.set_position(s.session().Position());
}

TextDecodeStep Ling3TextRunner::DecodeStep(
    TextRunnerState& state, std::size_t max_tokens,
    sampling::SamplerState& sampler) const {
  // The first token is always the model's own sample of the current logits,
  // exactly as SelectNext does it.
  const auto first = SelectNext(state, sampler);
  if (first.stop) {
    return TextDecodeStep{.selections = {},
                          .draft_tokens = 0,
                          .draft_accepted_tokens = 0,
                          .stop = true};
  }

  Ling3RunnerState& s = RequireLing3State(state);
  auto& session = s.session();

  // Only speculate when the whole block fits in what is left of the budget. The
  // extra tokens are already consumed by the time this returns, so returning
  // more than the budget would silently drop text -- the tail of a request must
  // decode one token at a time.
  const int n_max = static_cast<int>(session.SpecPolicy().n_max);
  const bool have_room = max_tokens >= static_cast<std::size_t>(n_max) + 1;
  // SPEC8: a short answer does not probe an unproven depth. Position() here is
  // the prompt length; max_tokens is what is left of this step's budget.
  const bool depth_ok = models::ling3::mtp::DepthDraftOk(
      static_cast<int>(session.Position()), static_cast<int>(max_tokens));
  if (!models::ling3::mtp::SpeculativeEnabled() || !have_room || !depth_ok) {
    Advance(state, first.token);
    return TextDecodeStep{.selections = {first}};
  }

  const std::size_t before = session.SpecStats().drafted_total;
  std::vector<std::int32_t> consumed;
  std::string error;
  if (!session.EvaluateSpeculative(static_cast<std::int32_t>(first.token),
                                   &consumed, &error)) {
    std::fprintf(stderr,
                 "LS4 DecodeStep: EvaluateSpeculative failed: %s "
                 "(pos=%u tokens=%zu)\n",
                 error.c_str(), session.Position(), consumed.size());
    throw std::runtime_error("ling3 speculative advance failed: " + error);
  }
  s.set_position(session.Position());

  const std::size_t drafted_now = session.SpecStats().drafted_total - before;

  TextDecodeStep step;
  step.selections.push_back(first);
  step.stop = false;
  for (std::size_t i = 1; i < consumed.size(); ++i) {
    const auto t = static_cast<TextRunnerToken>(consumed[i]);
    if (s.stop_at_eos() && model_->IsStopToken(consumed[i])) {
      // A stop token inside an accepted block ends the request here. The
      // remaining accepted tokens were still consumed by the model -- that is
      // what the non-speculative path would have done too -- but they are not
      // emitted.
      step.stop = true;
      break;
    }
    step.selections.push_back(
        TextDecodeSelection{false, t, model_->TokenText(consumed[i])});
  }
  // `drafted_now` counts what the head produced; the accepted ones are the
  // selections after the first that we actually emitted.
  step.draft_tokens = drafted_now;
  step.draft_accepted_tokens = consumed.empty() ? 0 : consumed.size() - 1;
  return step;
}

std::size_t Ling3TextRunner::CheckpointPosition(
    const TextRunnerState& state) const {
  return RequireLing3State(state).session().Position();
}

Ling3RunnerState& RequireLing3State(TextRunnerState& state) {
  auto* s = dynamic_cast<Ling3RunnerState*>(&state);
  if (s == nullptr) {
    throw std::logic_error("text runner state is not Ling-3.0-flash");
  }
  return *s;
}

const Ling3RunnerState& RequireLing3State(const TextRunnerState& state) {
  const auto* s = dynamic_cast<const Ling3RunnerState*>(&state);
  if (s == nullptr) {
    throw std::logic_error("text runner state is not Ling-3.0-flash");
  }
  return *s;
}

namespace {

void HashImage(crypto::Sha256Hasher& hasher, int width, int height,
               std::span<const std::uint8_t> pixels) {
  const std::uint32_t w = static_cast<std::uint32_t>(width);
  const std::uint32_t h = static_cast<std::uint32_t>(height);
  const std::uint8_t wh[8] = {
      static_cast<std::uint8_t>(w),       static_cast<std::uint8_t>(w >> 8),
      static_cast<std::uint8_t>(w >> 16), static_cast<std::uint8_t>(w >> 24),
      static_cast<std::uint8_t>(h),       static_cast<std::uint8_t>(h >> 8),
      static_cast<std::uint8_t>(h >> 16), static_cast<std::uint8_t>(h >> 24)};
  hasher.Update(wh);
  hasher.Update(pixels);
}

}  // namespace

std::optional<TextPreparedPrompt> Ling3TextRunner::PreparePrompt(
    const ChatRequest& request) const {
  bool any_image = false;
  for (const auto& message : request.messages) {
    if (!message.images.empty())
      any_image = true;
  }
  if (!any_image)
    return TextModelRunner::PreparePrompt(request);
  if (!vision_) {
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  }

  auto plan = std::make_shared<LingImagePlan>();
  std::vector<lgchat::Message> messages;
  messages.reserve(request.messages.size());
  for (const auto& message : request.messages) {
    if (!message.images.empty() &&
        message.role != tokenization::ChatRole::kUser) {
      throw std::invalid_argument("Ling-VL images belong on user messages");
    }
    std::vector<const tokenization::ChatMessage::ImagePart*> parts;
    parts.reserve(message.images.size());
    for (const auto& image : message.images)
      parts.push_back(&image);
    std::sort(parts.begin(), parts.end(), [](const auto* a, const auto* b) {
      return a->offset < b->offset;
    });
    std::string content = message.content;
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
      const auto& image = **it;
      if (!image.bytes || image.bytes->empty()) {
        throw std::invalid_argument("image part is empty");
      }
      if (image.offset > content.size()) {
        throw std::invalid_argument("image offset is outside the message");
      }
      core::Image decoded = core::DecodeImage(*image.bytes);
      core::Image resized = models::qwen::vision::ResizeImage(decoded);
      if (resized.width == 0 || resized.height == 0 ||
          resized.width % 32 != 0 || resized.height % 32 != 0 ||
          resized.width > 1024 || resized.height > 1024) {
        throw std::invalid_argument(
            "image exceeds the Ling vision tower limit (1024 pixels on a side, "
            "multiple of 32)");
      }
      plan->rgb.push_back(resized.pixels);
      plan->width.push_back(static_cast<int>(resized.width));
      plan->height.push_back(static_cast<int>(resized.height));
      plan->grids.push_back({static_cast<int>(resized.width / 32),
                             static_cast<int>(resized.height / 32)});
      content.insert(image.offset,
                     "<|vision_start|><|image_pad|><|vision_end|>");
    }
    // Recorded back-to-front; the prompt walks images in offset order.
    if (!parts.empty()) {
      const std::size_t n = parts.size();
      const std::size_t base = plan->grids.size() - n;
      std::reverse(plan->rgb.begin() + static_cast<std::ptrdiff_t>(base),
                   plan->rgb.end());
      std::reverse(plan->width.begin() + static_cast<std::ptrdiff_t>(base),
                   plan->width.end());
      std::reverse(plan->height.begin() + static_cast<std::ptrdiff_t>(base),
                   plan->height.end());
      std::reverse(plan->grids.begin() + static_cast<std::ptrdiff_t>(base),
                   plan->grids.end());
    }
    lgchat::Message out;
    out.role = ToLingRoleName(message.role);
    out.content = std::move(content);
    out.reasoning = message.thought;
    messages.push_back(std::move(out));
  }

  std::vector<lgchat::Tool> tools;
  for (const auto& tool : request.tools) {
    lgchat::Tool out;
    out.name = tool.name;
    out.description = tool.description;
    out.parameters_json =
        tool.parameters_json.empty() ? "{}" : tool.parameters_json;
    tools.push_back(std::move(out));
  }
  std::string error;
  auto rendered = lgchat::RenderChat(messages, tools, /*enable_thinking=*/false,
                                     /*add_generation_prompt=*/true, &error);
  if (!rendered.has_value()) {
    throw std::invalid_argument(error.empty() ? "Ling chat render failed"
                                              : error);
  }
  const auto ids = model_->tokenizer().Encode(*rendered);
  std::int32_t pad = 0;
  std::string token_error;
  const auto pad_ids = model_->tokenizer().Encode("<|image_pad|>", false, true);
  if (pad_ids.size() != 1) {
    throw std::invalid_argument("Ling-VL tokenizer has no image placeholder");
  }
  pad = pad_ids[0];
  (void)token_error;
  std::vector<std::int32_t> expanded;
  expanded.reserve(ids.size() + 64);
  std::size_t seen = 0;
  for (const std::int32_t id : ids) {
    if (id != pad) {
      expanded.push_back(id);
      continue;
    }
    if (seen >= plan->grids.size()) {
      throw std::invalid_argument(
          "prompt has more image placeholders than images");
    }
    const int count = plan->grids[seen].first * plan->grids[seen].second;
    ++seen;
    for (int k = 0; k < count; ++k)
      expanded.push_back(pad);
  }
  if (seen != plan->grids.size()) {
    throw std::invalid_argument("prompt is missing an image placeholder");
  }
  models::ling3::vision::PreparedPrompt prepared;
  if (!models::ling3::vision::FillImageRope(expanded, pad, plan->grids,
                                            &prepared, &error)) {
    throw std::invalid_argument(error.empty() ? "image position plan failed"
                                              : error);
  }

  crypto::Sha256Hasher hasher;
  const std::uint8_t tag[] = {'l', 'v', 'v', '2'};
  hasher.Update(tag);
  std::vector<std::uint8_t> running;
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < plan->grids.size(); ++i) {
    const std::size_t phys =
        static_cast<std::size_t>(prepared.image_phys.at(cursor));
    plan->cache_prefixes.push_back({phys, running});
    const int count = plan->grids[i].first * plan->grids[i].second;
    cursor += static_cast<std::size_t>(count);
    HashImage(hasher, plan->width[i], plan->height[i], plan->rgb[i]);
    const auto digest = hasher.Digest();
    running.assign(digest.begin(), digest.end());
  }
  plan->cache_identity = running;
  plan->prepared = std::move(prepared);

  TextPreparedPrompt out;
  out.tokens.assign(plan->prepared.tokens.begin(), plan->prepared.tokens.end());
  out.context = plan;
  return out;
}

void Ling3TextRunner::SetPromptContext(
    TextRunnerState& state,
    std::shared_ptr<const TextPromptContext> context) const {
  Ling3RunnerState& ling = RequireLing3State(state);
  if (!context) {
    ling.SetVision({});
    ling.session().UnbindVision();
    return;
  }
  const auto* plan = dynamic_cast<const LingImagePlan*>(context.get());
  if (plan == nullptr) {
    throw std::invalid_argument("invalid Ling prompt context");
  }
  if (!vision_) {
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  }
  auto prepared = plan->prepared;
  prepared.image_emb.clear();
  std::string error;
  for (std::size_t i = 0; i < plan->rgb.size(); ++i) {
    auto rows = vision_->Encode(plan->rgb[i].data(), plan->width[i],
                                plan->height[i], &error);
    const std::size_t need = static_cast<std::size_t>(plan->grids[i].first) *
                             static_cast<std::size_t>(plan->grids[i].second) *
                             2560u;
    if (rows.size() != need) {
      throw std::invalid_argument(
          error.empty() ? "vision tower returned an unexpected row count"
                        : error);
    }
    prepared.image_emb.insert(prepared.image_emb.end(), rows.begin(),
                              rows.end());
  }
  ling.session().BindVision(prepared);
  ling.SetVision(std::make_shared<const models::ling3::vision::PreparedPrompt>(
      std::move(prepared)));
}

}  // namespace superroc::server
