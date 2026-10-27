#ifndef SUPERROC_LING3_ROCM_BACKEND_HPP_
#define SUPERROC_LING3_ROCM_BACKEND_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling3/weights.hpp"

/// The ROCm execution path for Ling-3.0-flash.
///
/// STATUS, 20:45 local. The executor is WRITTEN and type-checks on the Mac
/// against LK's hipstub. It has **not been run**: LK's and LX's device
/// numerics were still unverified when this was written (their held jobs had
/// not produced a passing result), and nothing has been through
/// `ling-lg-gpu` yet. Treat the token output as unproven until the GPU gates
/// have run against `superroc serve`.
///
/// The per-token shape, per block:
///   ops_rms_norm -> (KDA via LK kda_forward | MLA via LK mla_block)
///                 -> residual -> ops_rms_norm
///                 -> (dense | MoE via LX) -> residual
/// then ops_final_norm, the LM head GEMV, ops_argmax.
///
/// Three things that are easy to get wrong and are deliberate here:
///   * `ling3_route` writes **device** ids and weights, but `ling3_expert_ffn`
///     takes **host** ones -- 8 ints and 8 floats are copied back per layer per
///     token;
///   * the routed experts cannot be resident: 512 of them is ~64 GiB against
///     ~24 MiB for the 8 selected, so each is staged into a small device buffer
///     at offset 0 and passed with `n_used = 1, n_expert = 1`, accumulating
///     into y (LX's expert_ffn does `y[i] += w * p[i]`);
///   * LK's `MlaWeights` wants `wk_b` / `wv_b` as `float*` and they are
///     ROCmFP4 on disk, so they are dequantized once at load (~8.4 MB each per
///     layer, ~117 MB total).
///
/// LK's entry points are in `namespace ling3`; LX's are C-linkage globals with
/// no namespace at all. Both collide by name with this package's own
/// `superroc::models::ling3`, so LK's are always spelled `::ling3::`.
namespace superroc::models::ling3::rocm {

/// Per-sequence device state: the KDA conv + SSM state for every KDA layer
/// (3*(d_conv-1)*d_inner + head_dim*head_dim*n_head floats each), the MLA
/// latent cache for every MLA layer, and the activation buffers.
struct State {
  // LG1 W3 stage (a): the host mirrors of the per-layer caches are never
  // uploaded and never read -- only their element counts are (for the device
  // alloc and the mtp keep image sizing). Replace the std::vectors with plain
  // size_t fields; keep every byte-size computation identical.
  std::size_t kda_conv_elems{0}, kda_ssm_elems{0};
  std::size_t mla_cache_elems{0};
  std::vector<std::size_t> kda_conv_off, kda_ssm_off;
  // mla_off holds 7 entries: one per MLA layer, indexed by g.mla_ord[li].
  // KDA layers never read or write the MLA cache.
  std::vector<std::size_t> mla_off;
  int pos{0};
  /// RV25 wave1 F5: how many decode Steps this state has run. Only used to
  /// space out ling3_moe_dev_check() -- one check per LING3_MOE_DEV_CHECK_EVERY
  /// tokens instead of one per token. Never reset (it is a phase, not a count
  /// of anything observable).
  std::uint64_t step_n{0};
  /// Context length this state was built for. The MLA latent cache is
  /// [max_ctx][kv_lora_rank + rotary_dim] per layer because mla_block is called
  /// with n_pos = pos + 1, so the cache MUST be sized for the context and not
  /// for one token.
  std::uint32_t max_ctx{0};
  float *x{nullptr}, *normed{nullptr}, *attn_out{nullptr}, *ffn_in{nullptr},
      *t{nullptr}, *y{nullptr};
  float *q{nullptr}, *k{nullptr}, *v{nullptr}, *f{nullptr}, *g2{nullptr},
      *beta{nullptr}, *tmp{nullptr}, *gate{nullptr};
  float* logits{nullptr};
  int* tok_in{nullptr};
  int* tok_out{nullptr};
  /// DEVICE position for mla_block's `const int * pos`. rope_kernel reads it on
  /// the GPU, so a host address here is a GCVM fault.
  int* pos_d{nullptr};
  /// LS3: lazily-allocated chunk-sized batch-prefill scratch (a BatchBufs*,
  /// opaque here because it is private to rocm_backend.cpp). Every buffer in it
  /// is sized for the chunk cap at creation and ForwardBatch refuses any n
  /// larger than that cap -- no buffer sized for n = 1 is ever handed T tokens
  /// (the class of bug that faulted the GPU in H1's PHASE7).
  void* batch{nullptr};
  /// LS2 PHASE7: per-session scratch for kda_forward_scratch /
  /// mla_block_scratch. The
  /// allocating wrappers hipMalloc + hipStreamSynchronize + hipFree on EVERY
  /// call, and Step calls them 40 + 7 times per token -- 47 full device drains
  /// per token, during which the host cannot run ahead and queue the next
  /// layer. The *_scratch entry points take the same buffers from us and launch
  /// the same kernels in the same order. Sized in NewState for n_tokens ==
  /// n_seqs == 1 and n_pos == max_ctx; Step falls back to the allocating
  /// wrapper if either needs more.
  void* kda_scratch{nullptr};
  std::size_t kda_scratch_bytes{0};
  void* mla_scratch{nullptr};
  std::size_t mla_scratch_bytes{0};
  /// LS3 PHASE4: split-K decode-attention partials (m / l are
  /// [heads][k_slices], acc is [heads][k_slices][kv]), preallocated for
  /// max_ctx so the decode step never mallocs or syncs (HIP-graph safe).
  float* mla_sk_q{nullptr};  // [heads][kv] hoisted absorbed q
  float* mla_sk_m{nullptr};
  float* mla_sk_l{nullptr};
  float* mla_sk_acc{nullptr};
  /// LS2 PHASE7: PINNED host staging for the one H2D that fills pos_d. Was a
  /// blocking pageable copy inside the MLA branch -- 7 per token. May be null
  /// (then Step does the blocking copy).
  int* pos_h{nullptr};
  void* kda_conv_device{nullptr};
  void* kda_ssm_device{nullptr};
  void* mla_cache_device{nullptr};
  std::vector<void*> allocs;
};

/// How the 512 routed experts per MoE layer are held.
struct ExpertPlacement {
  /// false (the default, and what the seat wants): all 512 experts of every MoE
  /// layer are uploaded once and stay resident -- ~59.8 GiB for this model,
  /// which fits the 128 GiB GTT. One `ling3_expert_ffn` call per layer, no
  /// staging, no per-token PCIe traffic.
  ///
  /// true: only the 8 selected experts are staged per layer per token
  /// (~24 MiB of working set) at the cost of ~3 MB of host-to-device copies per
  /// expert. Kept for boxes without the memory.
  bool stream{false};
};

/// Uploads the resident weights, allocates the routed experts according to
/// `placement`, and checks the total against the device before committing.
[[nodiscard]] bool Init(const ModelWeights& weights,
                        const core::GgufReader& reader,
                        const ExpertPlacement& placement,
                        std::size_t* resident_bytes, std::string* error_msg);

/// Bytes the resident plan needs, for a pre-flight memory check.
[[nodiscard]] std::size_t ResidentBytesFor(const ModelWeights& weights,
                                           const ExpertPlacement& placement);

/// True once Init has uploaded the weights and the staging buffers.
[[nodiscard]] bool ExecutorReady() noexcept;

/// Empty when ready; otherwise the reason, for the load-time message.
[[nodiscard]] std::string MissingPieces();

/// `max_ctx` is the context length the session will run to; it sizes the MLA
/// latent cache. Refuses (nullptr) if `max_ctx` is 0.
[[nodiscard]] State* NewState(std::uint32_t max_ctx);

/// Clears the recurrent state and the latent cache and rewinds `pos`. The
/// device arrays are owned by the backend, so this has to happen here.
void ResetState(State* state) noexcept;
void DestroyState(State* state) noexcept;

/// Refuses, with the reason.
[[nodiscard]] bool Step(std::int32_t token, State* state,
                        std::span<float> logits, std::string* error_msg);

/// LS3 batched prefill: runs `n` tokens starting at the state's current
/// position in one pass per layer (projections as tiled GEMMs, the KDA
/// recurrence and MLA causal attention with n_tokens = n, batched MoE routing
/// with per-expert grouped GEMMs) and advances the position by n.
///
/// `out_logits` takes the LAST position's logits when its size >= vocab, or
/// the per-position logits for all n tokens ([token][vocab]) when its size >=
/// n * vocab -- the teacher-forcing comparison the batch gate runs on. The
/// all-positions device buffer only exists when LING3_BATCH_ALL_LOGITS=1 was
/// set when the state was created (a ~322 MB test-only scratch at chunk 512).
///
/// Fails closed: refuses n above the chunk cap the buffers were sized for, a
/// position past max_ctx, streamed-expert loads, or any launch error -- the
/// caller falls back to token-at-a-time prefill.
///
/// LS4: `logit_rows > 0` asks for the logits of the FIRST `logit_rows` rows of
/// the batch -- [logit_rows][vocab] into `out_logits` -- through a small
/// fixed-size buffer instead of the chunk-sized all-positions scratch. The
/// speculative verify needs every row of a `1 + n_max` batch, and
/// LING3_BATCH_ALL_LOGITS=1 would cost 322 MB for a handful of rows.
/// `logit_rows` is capped at 8 rows (~5 MiB). Pass 0 for LS3's original
/// behaviour, which is unchanged.
///
/// LS4: `out_hidden`, when its size is exactly [hidden], receives the LAST
/// row's post-`output_norm` hidden state. ForwardBatch normalises into `B->t`;
/// only the single-token `Step` leaves the final hidden in `state->t`, so
/// `ReadHidden` CANNOT see it after a batched forward. The speculative draft
/// needs it -- it is the MTP head's `h_prev` for the next step -- which is what
/// this parameter is for. Any other size is refused.
///
/// LS4 phase 3: `out_hidden_rows`, when non-empty, receives the LAST
/// `out_hidden_rows` rows' post-`output_norm` hidden ([rows][hidden]). Priming
/// the MTP head over the prompt needs the hidden of every prompt position, and
/// the prefill path already normalised them -- it just did not hand them back.
[[nodiscard]] bool ForwardBatch(const std::int32_t* tokens, int n, State* state,
                                std::span<float> out_logits,
                                std::string* error_msg, int logit_rows = 0,
                                std::span<float> out_hidden = {},
                                std::span<float> out_hidden_rows = {});

/// LS4: the MTP draft head borrows these three from the trunk -- Ling ships no
/// nextn.embed_tokens and no nextn.shared_head_head, so the head reuses the
/// main token_embd / output / output_norm.
struct SharedTensors {
  const void* token_embd{nullptr};  ///< ggml type in `token_embd_type`
  int token_embd_type{0};           ///< Q6_K (text file) or Q5_K (VL file)
  const void* output{nullptr};  ///< ggml type in `output_type`; [hidden, vocab]
  int output_type{0};
  const float* output_norm{nullptr};  ///< F32 [hidden]
  int hidden{0};
  int vocab{0};
};
/// False until Init() has uploaded them.
[[nodiscard]] bool SharedTensorsView(SharedTensors* out) noexcept;

/// Copies the post-`output_norm` hidden state of the last Step() -- the fork's
/// `llama_get_embeddings_pre_norm(ctx_tgt)` row. Valid only immediately after
/// a successful Step(); ForwardBatch normalises into its own scratch, so this
/// CANNOT see it after a batched forward -- use ForwardBatch's out_hidden.
[[nodiscard]] bool ReadHidden(const State* state, std::span<float> out,
                              std::string* error_msg);

/// LS4 phase 4: `m` sequential `Step`s, one per row, each writing its own
/// [vocab] logits into `out_logits`.
///
/// Byte-identical to feeding the rows one at a time BY CONSTRUCTION: every row
/// runs the same op sequence, with the same kernels and the same inputs, that
/// `Step` runs for one token. That is the whole point -- speculative decoding
/// emits tokens the model chose, so the verify has to reproduce the model's own
/// argmax exactly, and the batched path's arithmetic does not (REPORT3 §3).
///
/// Being m sequential steps, it is also no faster than m sequential steps. The
/// speed is meant to arrive at `StepMultiDispatch`'s single dispatch point.
///
/// On failure the position is rewound by the rows already advanced, so the call
/// leaves the state exactly as it found it.
/// LM1 SPEC4 item 1: the trunk state a speculative verify must leave behind so
/// a rejected draft can be unwound WITHOUT replaying `Step`.
///
/// `Step`'s only cross-token state is (a) the KDA conv + SSM state of every KDA
/// layer and (b) `state->t`, the post-`output_norm` hidden that `ReadHidden`
/// hands the draft head. The MLA latent cache is NOT in that set: it is
/// position-indexed, a verify appends rows [pos0, pos0+m) and the next `Step`
/// at `pos0+a+1` appends row `pos0+a+1` and then attends over the first
/// `pos0+a+2`, so the rows a reject abandons are overwritten before they can
/// ever be read. (mla_block and mla_block_splitk both append at `n_pos -
/// n_tokens` and attend over the first `n_pos`; `n_pos` comes from the
/// position.)
///
/// One image per row, rows 0..m-2. A rejection of `a` drafts installs the image
/// for row `a`; a full accept installs nothing and keeps the block. Row m-1 is
/// never needed -- that is the full-accept case.
/// LV1 R6: install one imaged row's KDA conv/ssm slices. MLA slices are not
/// copied. `conv_row` / `ssm_row` point at that row of the image (already
/// advanced by row * bytes).
[[nodiscard]] bool InstallKdaSlices(State* trunk, const void* conv_row,
                                    const void* ssm_row);

struct MultiRowKeep {
  void* kda_conv{nullptr};  ///< [m-1] images of the WHOLE conv array
  void* kda_ssm{nullptr};   ///< [m-1] images of the WHOLE ssm array
  float* hidden{nullptr};   ///< [m-1][hidden], post-output_norm
  std::size_t conv_bytes{0};
  std::size_t ssm_bytes{0};
  int rows{0};
};

[[nodiscard]] bool StepMulti(const std::int32_t* tokens, int m, State* state,
                             std::span<float> out_logits,
                             std::string* error_msg,
                             MultiRowKeep* keep = nullptr);

/// LS4 phase 4: THE dispatch point for a speculative verify.
///
/// With `LING3_MTP_MULTI=0` (the default, and the only mode that compiles until
/// LS1's multi-row kernels exist) this is exactly `StepMulti`. With
/// `LING3_MTP_MULTI=1` and the kernels present it runs the whole block through
/// `ling3_gemv_multi` / `ling3_expert_ffn_multi`, which read each weight once
/// for all m activation columns. LS1 owns those kernels; nothing outside this
/// function changes when they land.
[[nodiscard]] bool StepMultiDispatch(const std::int32_t* tokens, int m,
                                     State* state, std::span<float> out_logits,
                                     std::string* error_msg,
                                     MultiRowKeep* keep = nullptr);

/// True when this load can serve ForwardBatch: weights resident (the routed
/// experts staged per token have no batched path) and Init succeeded.
[[nodiscard]] bool BatchCapable() noexcept;

}  // namespace superroc::models::ling3::rocm

#endif  // SUPERROC_LING3_ROCM_BACKEND_HPP_
