#ifndef SUPERROC_MODELS_LING3_MTP_GPU_HPP_
#define SUPERROC_MODELS_LING3_MTP_GPU_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "src/models/ling3/weights.hpp"

/// The Ling-3.0-flash MTP draft head (block 42) on the GPU.
///
/// One MTP step is a whole decoder block plus the LM head -- one block in 43,
/// but with the *whole* vocabulary projection, so it is a real cost and has to
/// be measured rather than assumed cheap. See MTP_NOTES.md.
///
///   DraftNext(h_prev, x_next, pos) -> (x_next2, h_new)
///
/// `h_prev` is the target's post-`output_norm` hidden state at `pos - 1` (the
/// fork's `pending_h`, i.e. `llama_get_embeddings_pre_norm(ctx_tgt)`), and
/// `x_next` is the token sitting at `pos`. The head is a shifted predictor: it
/// emits the token for `pos + 1`, and `h_new` is its own post-final-norm
/// hidden, which pairs with that token on the next draft step.
///
/// This file owns NO kernel source. It calls the existing entry points in
/// ling3_ops.hip / ling3_mla.hip / ling3_moe.hip read-only, which stay LS1's
/// and LS2's.
namespace superroc::models::ling3::mtp {

/// Which vector normalises the head's final hidden state. See MTP_NOTES.md §2.
enum class HeadNorm {
  /// The main `output_norm` -- what `bailing-hybrid.cpp:1011-1014` actually
  /// does on Ling. Default, because the fork is the reference.
  kOutputNorm,
  /// `blk.42.layer_output_norm`, which `bailingmoe3.cpp:718-720` prefers.
  kLayerOutNorm,
};

/// Uploads the MTP block's own weights and prepares the shared views the head
/// borrows from the trunk. `rocm::Init` must already have run. Idempotent.
[[nodiscard]] bool Init(const ModelWeights& weights, HeadNorm head_norm,
                        std::size_t* resident_bytes = nullptr,
                        std::string* error_msg = nullptr);

/// Bytes `Init` will add to the resident plan.
[[nodiscard]] std::size_t ResidentBytesFor(const ModelWeights& weights);

[[nodiscard]] bool Ready() noexcept;
/// Empty when ready; otherwise the reason.
[[nodiscard]] std::string MissingPieces();

struct State;

/// `max_ctx` sizes the MTP block's own MLA latent cache (one [512 + 64] row
/// per position, plus the tail pad). Refuses 0.
[[nodiscard]] State* NewState(std::uint32_t max_ctx);
void ResetState(State* state) noexcept;
void DestroyState(State* state) noexcept;

/// Bytes one state holds, for a pre-flight check.
[[nodiscard]] std::size_t StateBytes(std::uint32_t max_ctx);

/// One priming pass: writes the head's latent cache at `pos` from
/// `(h_prev, x_next)` and returns `h_out` for the next position. Identical to
/// `DraftNext` except that it SKIPS the LM head and the argmax -- the draft
/// token is discarded during prefill and only the hidden state is wanted, and
/// the LM head is the most expensive part of the head (214 MB of weight read).
///
/// `State` must be windowed with base 0 (the default for a fresh state), so the
/// positions written line up with the model's absolute positions.
/// LM1 (DECISIONS 00:30): the batched version of PrimeNext. Fills the head's
/// MLA latent cache over a whole prefill chunk -- rows = the chunk's tokens,
/// h_prev = the trunk hidden at the PREVIOUS position (EvaluateBatch pairs
/// token t with trunk hidden t-1, as the fork does) -- so priming the ENTIRE
/// prompt costs one layer's projection traffic instead of one block per token.
/// Only the cache is written: the attention output, the MoE FFN and h_out are
/// not produced, because prefill discards the head's own output (the fork's
/// process() does the same) and nothing downstream of mla_cache_append touches
/// the cache. That is also why the head's Q8_0 weights never reach the
/// ROCmFP4-only small-m kernels -- see the note above PrimeBatch.
/// LING3_MTP_PRIME_BATCH=1 turns it on; default off.
[[nodiscard]] bool PrimeBatchEnabled();
[[nodiscard]] bool PrimeBatch(State* state, const std::int32_t* tokens,
                              const float* h_prev, int n, std::uint32_t pos0,
                              std::string* error_msg);
[[nodiscard]] bool PrimeNext(State* state, std::int32_t x_next,
                             std::span<const float> h_prev, std::uint32_t pos,
                             std::span<float> h_out,
                             std::string* error_msg = nullptr);

/// Makes the head's latent cache hold only the positions from the first draft
/// onward, instead of indexing it by the model's absolute position. Speculation
/// needs this: the head is the only writer of its cache, so the prompt rows
/// would otherwise stay zeros and dilute its attention (MEASURED: acceptance
/// 0/3 without it). RoPE still uses the true position.
void SetWindowed(State* state, bool windowed) noexcept;

/// LM1 SPEC7 item 2: `LING3_MTP_WINDOW=W` -- the head ATTENDS only to the last
/// W rows of its own latent cache (a sliding-window draft head). The cache is
/// still written at every position (priming fills the prompt), but the MLA
/// attention reads a W-row tail of it, so the draft forward stops growing with
/// the context. MEASURED (REPORT6 §5): at 56K one draft forward was 47.8 ms
/// against 7.5 ms at 3K -- the difference is attention over the prompt rows.
///
/// rows <= 0 (the default) attends over the whole cache, i.e. the pre-SPEC7
/// behaviour. The verify still decides every emitted token, so this can only
/// change which drafts are proposed, never the text.
void SetAttnWindow(State* state, int rows) noexcept;

/// Runs one draft step.
///
/// `h_prev` is host, `hidden_size` floats. `draft_token` and `h_out` are host.
/// `pos` is the position of `x_next` and must be below the state's `max_ctx`;
/// the call advances the MTP block's latent cache to `pos + 1`.
///
/// Fails rather than running on out-of-range positions or token ids.
[[nodiscard]] bool DraftNext(State* state, std::int32_t x_next,
                             std::span<const float> h_prev, std::uint32_t pos,
                             std::int32_t* draft_token, std::span<float> h_out,
                             std::string* error_msg = nullptr,
                             float* top1_p = nullptr);
// `top1_p`, when non-null, receives the drafted token's top-1 softmax
// probability over the full vocab (1 / sum_i exp(x_i - x_max), exact to fp32).
// It is what the p_min confidence gate reads (SPEC6 item 3 / RV14 3.2). Null
// skips the exponent pass -- the priming path never needs it.

// ---------------------------------------------------------------------------
// Phase-1 acceptance probe. Off unless LING3_MTP_PROBE=1 is in the
// environment, and then it only observes -- the draft is never fed back into
// the model, so generation is bit-identical with the probe on or off and the
// acceptance number it reports is the real one.
//
// `LING3_MTP_HEAD_NORM=layer_out_norm` selects blk.42.layer_output_norm for the
// head's final norm instead of the main output_norm (MTP_NOTES.md §2).
// ---------------------------------------------------------------------------

[[nodiscard]] bool ProbeEnabled() noexcept;

/// Loads the head and allocates the probe's single state. `max_context` must be
/// the same one the sessions were created with. Refuses if `Init` fails, and
/// says why.
void ProbeInit(const ModelWeights& weights, std::uint32_t max_context,
               std::string* error_msg = nullptr);

/// Called once per position, at the TOP of the decode step, PREFILL INCLUDED.
///
/// `pos` is the position of `x_next` in the target's numbering, `h_prev` the
/// target's post-`output_norm` hidden state at `pos - 1`, and `have_hidden`
/// says whether `h_prev` is real (it is not at `pos == 0`, which is also where
/// the cache is wiped). `scored` is false during prefill, where the token that
/// arrives is a prompt token rather than something the model chose.
///
/// The draft made here predicts the token at `pos + 1`, so it is scored
/// against the `x_next` of the NEXT call. That is why this runs before Step:
/// `x_next` and `pos` are exactly the draft batch's `id_last` and `p_next`
/// (speculative.cpp:1766-1793).
void ProbeStep(void* backend_state, std::int32_t x_next,
               std::span<const float> h_prev, std::uint32_t pos,
               bool have_hidden, bool scored);

/// How many drafts have been scored so far, and how many were accepted.
void ProbeTally(int* compared, int* accepted, double* total_ms);

/// Frees the probe state. Safe to call when the probe never initialised.
void ProbeShutdown();

// ===========================================================================
// Phase 2 — speculative decoding: draft k, verify 1+k in one batched forward,
// accept the greedy-matching prefix, roll the trunk's KDA state back.
//
// The contract that makes this safe at temperature 0:
//
//   after SpecStep returns, `trunk` is in EXACTLY the state it would be in if
//   the tokens in `out` had been fed one at a time, and `next_logits` is the
//   model's own distribution for the position right after them.
//
// Both come from the model's own argmax rows, never from a draft, so greedy
// decoding emits byte-identical text with drafting on or off.
// ===========================================================================

/// Acceptance-adaptive draft length. Never a fixed n: a fixed n = 7 fell to
/// 18 % acceptance on gfx1151, and the fork's own drafting LOSES to no drafting
/// on prose at temperature 0 (31.7 vs 34.3 tok/s). So the cap moves with the
/// measured acceptance and can fall to 0, which turns the step back into an
/// ordinary single-token decode.
struct SpecPolicy {
  // LM1 SPEC6 item 3 (RV14 3.3): default 2, up from 1. The SPEC4 numbers that
  // argued for 1 were measured with NO per-token confidence gate, so a weak
  // step always paid a 3-row verify. With p_min the second draft only exists
  // when the head is already confident about the first, and RV14 4 shows the
  // gated k<=2 mix beats fixed k=1 above ~0.80 acceptance (1.171 vs 1.165 at
  // a = 0.89). Override with LING3_MTP_NMAX.
  int n_max{2};
  /// SPEC6 item 3: stop the draft chain after any candidate whose top-1
  /// softmax probability is below this, and DISCARD that candidate -- the
  /// fork's rule (fork/speculative.cpp:1867-1871). Replaces the one-strike
  /// shrink below entirely. Override with LING3_MTP_PMIN.
  double p_min{0.5};
  /// Kept for LING3_MTP_ADAPTIVE env-compat only; the shrink/grow branches
  /// that read these are gone (RV14 3: "delete rather than repair" -- its
  /// window mixed token and step units). The p_min gate does their job per
  /// step, so a weak step verifies 1 row or 0, never 3.
  double min_accept{0.55};
  /// See min_accept.
  double resume_accept{0.80};
  /// See min_accept.
  int window{8};
};

/// LM1 SPEC3: LING3_MTP_MAX_POS.
///
/// Drafting is a loss at depth: each verify row pays MLA attention over the
/// whole context and the MTP head attends over it too, so cost grows with
/// position while the win per step does not. Measured on build25 at --context
/// 32768, temperature 0: a 6,540-token prompt decodes 22.76 tok/s with drafting
/// OFF and 17.70 ON (-22 %).
///
/// This is the cap. Once a session's position reaches `LING3_MTP_MAX_POS`,
/// SpecStep runs plain Step instead of drafting -- the same code path
/// drafting-OFF takes, so the output is byte-identical by construction. The
/// default is the measured breakeven rounded down (see REPORT3); it is a
/// tunable because LS3's split-K MLA moves the breakeven. 0 or unset disables
/// the cap.
[[nodiscard]] bool MaxPosAllows(std::uint32_t pos);

/// Counters for one step and for the process, all measured.
struct SpecStats {
  int steps{0};
  int drafted_total{0};
  int accepted_total{0};
  int full_accept_steps{0};
  int no_draft_steps{0};
  double draft_ms{0.0};
  double verify_ms{0.0};
  double rollback_ms{0.0};
  /// Depth-1..n_max acceptance, for the per-length breakdown the report needs.
  long accept_at_len[4]{0, 0, 0, 0};
  long drafted_at_len[4]{0, 0, 0, 0};
  // SPEC6 item 3: the k-distribution the p_min gate produces, and how often it
  // shortens the chain. k0_steps are the steps where the FIRST candidate was
  // already under p_min, so the step ran as a plain Step. gate_skips counts
  // discarded candidates and equals k0 + k1: the gate stops at the first
  // failure, so a k0 step throws away its one candidate, a k1 step throws away
  // the second, and a k2 step kept both and discards nothing (measured:
  // tooltest 027a5ae9 -> skips=42 = k1 42 + k0 0).
  long gate_skips{0};
  long k0_steps{0};
  long k1_steps{0};
  long k2_steps{0};
  // SPEC6 item 4: LM1_COST per length. The process-wide LM1_COST at exit mixes
  // every request into one average; these are the same three tallies, reset
  // with the session, so ReportSpecSession can print one row per request and
  // the curve can price each prompt. plain_n is no_draft_steps above (every
  // plain step counted there took the same run_plain_step path, timed the same
  // way); the only steps that skip it are the ones whose hipEvent pair
  // returned 0, which the process-wide plain_n skips too.
  double plain_ms_sum{0.0};
  double acc_ms_sum{0.0};
  long acc_n_sum{0};
  double rej_ms_sum{0.0};
  long rej_n_sum{0};
  double rej_tok_sum{0.0};
};

/// Per-session speculative state: the head's own latent cache, the KDA
/// snapshot used to rewind the trunk, and the rolling acceptance window.
class SpecState;

[[nodiscard]] SpecState* NewSpecState(std::uint32_t max_ctx,
                                      const SpecPolicy& policy,
                                      std::string* error_msg = nullptr);
void DestroySpecState(SpecState* spec) noexcept;
/// RV4 (Jay build28f): rewinds a spec state for a new sequence without
/// releasing its allocations -- the head's cache, the rolling acceptance window
/// and the cost guard are per-sequence state a new prompt must not inherit.
void ResetSpecState(SpecState* spec) noexcept;
/// The head state inside `spec`, for the prefill priming in
/// Session::EvaluateBatch. A free function because SpecState is deliberately
/// only forward-declared in this header.
[[nodiscard]] State* SpecHead(SpecState* spec) noexcept;
/// Bytes one SpecState holds (the KDA snapshot dominates).
[[nodiscard]] std::size_t SpecStateBytes(std::uint32_t max_ctx);

/// Runs one speculative decode step.
///
/// `trunk` is the session's ROCm state, positioned at `base`'s position.
/// `base` is the token about to be fed (the model's own argmax from the
/// previous step). `out` receives the tokens the trunk has now consumed --
/// `base` plus whichever drafts were accepted. `next_logits` receives the
/// model's distribution for the position after them.
///
/// Never fails soft: if anything is unavailable (no HIP, no head, a position
/// past the context, a batch the trunk cannot serve) it returns false and
/// leaves `trunk` untouched, and the caller falls back to a plain decode.
///
/// `h_prev` / `have_h_prev` are the target's post-`output_norm` hidden at
/// `pos - 1` and whether it is real. It is passed IN rather than read from the
/// trunk because ForwardBatch normalises into its own scratch -- only the
/// single-token Step leaves the final hidden in `state->t`, so ReadHidden would
/// hand back a stale row after any batched forward. `h_prev_out` receives the
/// hidden at the new frontier, for the next step.
///
/// `is_stop` decides whether a token ends generation. If one sits inside the
/// block and is not its last token, the trunk is rewound to end exactly at it
/// and only the tokens up to and including it are reported -- otherwise the
/// caller's checkpoint would run ahead of the tokens it published and the
/// scheduler refuses the step.
[[nodiscard]] bool SpecStep(void* trunk, SpecState* spec, std::int32_t base,
                            const std::vector<float>& h_prev, bool have_h_prev,
                            std::vector<std::int32_t>* out,
                            std::vector<float>* next_logits,
                            std::vector<float>* h_prev_out, SpecStats* stats,
                            const std::function<bool(std::int32_t)>& is_stop,
                            std::string* error_msg = nullptr);

/// The draft length the adaptive policy currently allows, for logging.
[[nodiscard]] int SpecDraftLength(const SpecState* spec) noexcept;

/// True when the environment asked for speculative decoding (LING3_MTP=1).
[[nodiscard]] bool SpeculativeEnabled() noexcept;

/// SPEC8. Unset LING3_MTP_DEPTH uses the fixed rule. A generation
/// shorter than depth::kUnprovenMinTokens does not open an unproven
/// depth (the probe cannot fit in a short answer). A request that
/// already started drafting keeps going as the remainder shrinks.
/// LING3_MTP_DEPTH=0 always returns true.
[[nodiscard]] bool DepthDraftOk(int pos, int remaining_tokens) noexcept;

/// Installs a process-exit reporter for `SpecStats`. Called once from
/// NewSpecState; harmless to call again.
void InstallSpecReporter();

/// Merge a session's counters into the process totals and print the report on
/// exit. The scheduler destroys sessions freely, so the totals live here.
/// Flush one REQUEST's row to stderr and fold it into the process-wide
/// aggregate. `req` and `prompt_tokens` label the row: the serve path keeps one
/// Session resident for the whole server, and Session::FlushSpecReport fires
/// when a request's decode is done (EvaluateBatch runs once per 512-row prefill
/// CHUNK, so the prompt length is summed across a request's chunks before it is
/// reported). The curve's LM1_COST rows therefore arrive in prompt order and
/// each carries the prompt length that produced it -- "LM1_COST per length"
/// (SPEC6 item 4) readable straight out of the serve log.
void ReportSpecSession(const SpecStats& s, int final_draft_len, int req,
                       int prompt_tokens);

}  // namespace superroc::models::ling3::mtp

#endif  // SUPERROC_MODELS_LING3_MTP_GPU_HPP_
