#include "src/models/ling3/mtp_gpu.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(LING3_WITH_ROCM)

#include <hip/hip_runtime.h>

#include "src/models/ling3/depth_rule.hpp"
#include "src/models/ling3/kernels/ling3_gemm.hpp"
#include "src/models/ling3/kernels/ling3_gemv_q8.hpp"
#include "src/models/ling3/kernels/ling3_mla.hpp"
#include "src/models/ling3/kernels/ling3_moe.hpp"
#include "src/models/ling3/kernels/ling3_ops.hpp"
#include "src/models/ling3/rocm_backend.hpp"

namespace superroc::models::ling3::mtp {
namespace {

constexpr std::size_t kMaxExperts = 8;
/// ling3_moe.hpp: every weight and activation allocation needs this many
/// readable bytes past the last element (the kernels widen a tail load).
constexpr std::size_t kPad = LING3_TAIL_PAD;

// ---------------------------------------------------------------------------
// Device helpers. Mirrors rocm_backend.cpp's, deliberately: hipMemset is
// absent from LK's Mac type-check stub, and a big device memset is a launch
// we do not need.
// ---------------------------------------------------------------------------
void* DevAlloc(std::size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  void* p = nullptr;
  if (hipMalloc(&p, bytes) != hipSuccess) {
    return nullptr;
  }
  return p;
}

// `d` is `const void*` because the weight structs hold `const void*` device
// pointers; nothing here writes through it before the cast.
bool DevCopy(const void* d, const void* s, std::size_t n) {
  return n == 0 || hipMemcpy(const_cast<void*>(d), s, n,
                             hipMemcpyHostToDevice) == hipSuccess;
}

bool DevCopyF(const float** slot, const void* src, std::size_t n) {
  float* p = static_cast<float*>(DevAlloc(n));
  if (p == nullptr || !DevCopy(p, src, n)) {
    return false;
  }
  *slot = p;
  return true;
}

void DevZero(void* p, std::size_t bytes) {
  if (p == nullptr || bytes == 0) {
    return;
  }
  static const std::vector<float> kZeros(1 << 16, 0.0F);
  std::size_t left = bytes;
  const std::uint8_t* src =
      reinterpret_cast<const std::uint8_t*>(kZeros.data());
  while (left > 0) {
    const std::size_t chunk = left < kZeros.size() * sizeof(float)
                                  ? left
                                  : kZeros.size() * sizeof(float);
    if (hipMemcpy(p, src, chunk, hipMemcpyHostToDevice) != hipSuccess) {
      return;
    }
    p = static_cast<std::uint8_t*>(p) + chunk;
    left -= chunk;
  }
}

// ---------------------------------------------------------------------------
// Plan + device weights for the MTP block only.
// ---------------------------------------------------------------------------
struct W {
  const void* d{nullptr};
  int type{0};
  int rows{0};
  int cols{0};
  int experts{1};
  std::size_t bytes{0};
};

struct Mtp {
  int hidden{0}, vocab{0}, heads{0};
  int expert_ff{0}, n_expert{0}, n_used{0}, n_groups{0}, n_group_used{0};
  float w_scale{1.0F}, eps{1e-6F};
  float rope_theta{6000000.0F};
  float clamp_exp{0.0F}, clamp_shexp{0.0F};

  // Borrowed from the trunk's ROCm Init -- the MTP head ships no
  // nextn.embed_tokens and no nextn.shared_head_head.
  const void* token_embd{nullptr};
  int token_embd_type{0};
  const void* output{nullptr};
  int output_type{0};
  const float* output_norm{nullptr};

  W eh_proj, attn_q, attn_kv_a_mqa, attn_gate, attn_output;
  W router, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
  W ex_gate, ex_up, ex_down;  // resident [*, *, 512] stacks, Q8_0
  const float* attn_norm{nullptr};
  const float* ffn_norm{nullptr};
  const float* kv_a_norm{nullptr};
  const float* probs_bias{nullptr};
  const float* enorm{nullptr};
  const float* hnorm{nullptr};
  const float* head_norm{nullptr};
  /// wk_b / wv_b are F32 mirrors: MlaWeights wants float*, and on disk the MTP
  /// block's are Q8_0.
  const float* wk_b_f32{nullptr};
  const float* wv_b_f32{nullptr};
  int* d_ids{nullptr};
  float* d_weights{nullptr};
  std::size_t resident{0};
  bool ready{false};
};

Mtp g;

std::string Missing;

bool Upload(const TensorRef& t, W* w, std::string* error_msg) {
  // core::GgmlType and the kernels' enum share their numeric values (0=F32,
  // 8=Q8_0, 14=Q6_K, 100/101 ROCmFP4); rocm_backend.cpp:339 relies on it too.
  w->type = static_cast<int>(t.type);
  w->rows = static_cast<int>(t.rows);
  w->cols = static_cast<int>(t.cols);
  w->experts = static_cast<int>(t.experts);
  w->bytes = t.SizeBytes();
  w->d = DevAlloc(w->bytes);
  if (w->d == nullptr || !DevCopy(w->d, t.data, w->bytes)) {
    if (error_msg != nullptr) {
      *error_msg = "mtp: upload failed for " + std::string(t.name);
    }
    return false;
  }
  g.resident += w->bytes + kPad;
  return true;
}

bool UploadVec(const VectorRef& v, const float** slot, std::string* error_msg) {
  if (v.empty()) {
    if (error_msg != nullptr)
      *error_msg = "mtp: missing norm vector";
    return false;
  }
  const std::size_t n = v.tensor.SizeBytes();
  if (!DevCopyF(slot, v.tensor.data, n)) {
    if (error_msg != nullptr)
      *error_msg = "mtp: norm upload failed";
    return false;
  }
  g.resident += n + kPad;
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------

struct State {
  float* mla_cache{nullptr};
  std::size_t mla_bytes{0};
  int pos{0};
  std::uint32_t max_ctx{0};
  /// Windowing. `windowed` makes the latent cache hold only the positions from
  /// `base_pos` onward, instead of indexing it by the model's absolute
  /// position. Used by the speculative path, where the prompt rows would
  /// otherwise be zeros the head never filled.
  bool windowed{false};
  int base_pos{0};
  bool base_set{false};
  /// SPEC7 item 2: the ATTENTION window, in cache rows. > 0 => the head reads
  /// only the last `attn_win` rows of its cache (LING3_MTP_WINDOW=W); 0 => the
  /// whole cache. Independent of `windowed`, which is about how rows are
  /// ADDRESSED, not how many are read.
  int attn_win{0};
  /// Persistent softmax-score scratch for the head's attention
  /// (mla_block_scratch_bytes(sh, 1, rows) floats). The non-scratch
  /// `mla_block` overload hipMallocs that per call and then stream-syncs before
  /// freeing it, which is a per-draft alloc plus a pipeline drain; the buffer
  /// is reused instead. `attn_scores_rows` is its capacity in cache rows.
  float* attn_scores{nullptr};
  int attn_scores_rows{0};

  float *h_prev{nullptr}, *e_raw{nullptr}, *enorm{nullptr}, *hnorm{nullptr},
      *concat{nullptr}, *cur{nullptr}, *normed{nullptr}, *q{nullptr},
      *k{nullptr}, *gate{nullptr}, *attn_out{nullptr}, *ffn_in{nullptr},
      *t{nullptr}, *y{nullptr}, *f{nullptr}, *h_out{nullptr}, *logits{nullptr};
  int *tok_in{nullptr}, *tok_out{nullptr}, *pos_d{nullptr};
  std::vector<void*> allocs;

  ~State() { Release(); }
  /// Frees the device buffers. DestroyState() is this plus `delete st`, so it
  /// must NOT be called from the destructor -- that would delete `this` while
  /// it is still being destroyed.
  void Release() noexcept {
    for (void* p : allocs) {
      if (p != nullptr)
        hipFree(p);
    }
    allocs.clear();
    // SPEC7 item 2: the score scratch is grown on demand (the window is fixed
    // before the first call, so in practice it is one alloc), so it is NOT in
    // `allocs` -- that would double-free a buffer replaced on grow.
    if (attn_scores != nullptr) {
      (void)hipFree(attn_scores);
      attn_scores = nullptr;
    }
    attn_scores_rows = 0;
  }
};

namespace {

constexpr std::size_t F(std::size_t n) {
  return n * sizeof(float);
}

bool AllocF(float** slot, std::size_t n) {
  float* p = static_cast<float*>(DevAlloc(F(n)));
  if (p == nullptr)
    return false;
  DevZero(p, F(n));
  *slot = p;
  return true;
}

}  // namespace

std::size_t StateBytes(std::uint32_t max_ctx) {
  // The latent cache dominates: one [kv_lora_rank + rotary_dim] row per
  // position.
  // + the activation scratch below, and + the attention score scratch
  // (n_head rows of n_pos floats -- mla_block_scratch_bytes(sh, 1, n_pos)).
  // Worst case: the whole cache is visible. With LING3_MTP_WINDOW=W set it is
  // W rows instead, so this over-states a windowed state.
  return static_cast<std::size_t>(max_ctx) * (512 + 64) * sizeof(float) +
         (64 * 1024 * sizeof(float)) +
         static_cast<std::size_t>(max_ctx) * 32 * sizeof(float);
}

std::size_t ResidentBytesFor(const ModelWeights& w) {
  std::size_t n = 0;
  const auto add = [&n](const TensorRef& t) { n += t.SizeBytes() + kPad; };
  add(w.mtp_head.eh_proj);
  add(w.mtp.mla_weights.wq);
  add(w.mtp.mla_weights.wkv_a_mqa);
  add(w.mtp.mla_weights.wqkv_gate);
  add(w.mtp.mla_weights.wo);
  add(w.mtp.moe.gate_inp);
  add(w.mtp.moe.gate_shexp);
  add(w.mtp.moe.up_shexp);
  add(w.mtp.moe.down_shexp);
  add(w.mtp.moe.gate_exps);
  add(w.mtp.moe.up_exps);
  add(w.mtp.moe.down_exps);
  // wk_b / wv_b are uploaded packed, dequantized to F32, then the packed copy
  // is freed -- but the F32 mirror is what stays resident.
  const std::size_t wk = static_cast<std::size_t>(w.mtp.mla_weights.wk_b.rows) *
                         w.mtp.mla_weights.wk_b.cols *
                         w.mtp.mla_weights.wk_b.experts * sizeof(float);
  const std::size_t wv = static_cast<std::size_t>(w.mtp.mla_weights.wv_b.rows) *
                         w.mtp.mla_weights.wv_b.cols *
                         w.mtp.mla_weights.wv_b.experts * sizeof(float);
  n += wk + kPad + wv + kPad;
  for (const VectorRef* v :
       {&w.mtp.attn_norm, &w.mtp.ffn_norm, &w.mtp.mla_weights.kv_a_norm,
        &w.mtp.moe.probs_bias, &w.mtp_head.enorm, &w.mtp_head.hnorm}) {
    n += v->tensor.SizeBytes() + kPad;
  }
  return n;
}

bool Init(const ModelWeights& w, HeadNorm head_norm,
          std::size_t* resident_bytes, std::string* error_msg) {
  if (g.ready) {
    if (resident_bytes != nullptr)
      *resident_bytes = g.resident;
    return true;
  }
  auto fail = [&](const std::string& m) {
    Missing = m;
    if (error_msg != nullptr)
      *error_msg = m;
    return false;
  };

  // The head borrows main token_embd / output / output_norm, so the trunk's
  // ROCm Init must already have uploaded them.
  rocm::SharedTensors shared;
  if (!rocm::SharedTensorsView(&shared) || shared.token_embd == nullptr ||
      shared.output == nullptr || shared.output_norm == nullptr) {
    return fail(
        "mtp: rocm::Init has not run, or the trunk's token_embd / output / "
        "output_norm are not on the device");
  }

  const Config& c = w.config;
  g.hidden = static_cast<int>(c.hidden_size);
  g.vocab = static_cast<int>(c.vocab_size);
  g.heads = static_cast<int>(c.num_heads);
  g.expert_ff = static_cast<int>(c.expert_ff);
  g.n_expert = static_cast<int>(c.num_experts);
  g.n_used = static_cast<int>(c.num_experts_used);
  g.n_groups = static_cast<int>(c.num_expert_groups);
  g.n_group_used = static_cast<int>(c.num_group_used);
  g.w_scale = c.expert_weights_scale;
  g.eps = c.rms_eps;
  g.rope_theta = c.rope_theta;
  g.token_embd = shared.token_embd;
  g.token_embd_type = shared.token_embd_type;
  g.output = shared.output;
  g.output_type = shared.output_type;

  // Block 42's SwiGLU clamps. swiglu_clamp_* has block_count entries, so index
  // 42 is in range (0..42) -- the MTP block is not "past the end".
  const std::uint32_t blk = c.block_count - 1;
  g.clamp_exp =
      blk < c.swiglu_clamp_exp.size() ? c.swiglu_clamp_exp[blk] : 0.0F;
  g.clamp_shexp =
      blk < c.swiglu_clamp_shexp.size() ? c.swiglu_clamp_shexp[blk] : 0.0F;

  if (w.mtp_head.eh_proj.empty() || w.mtp_head.enorm.empty() ||
      w.mtp_head.hnorm.empty()) {
    return fail("mtp: the artifact has no nextn.eh_proj / enorm / hnorm");
  }

  const MlaWeights& mw = w.mtp.mla_weights;
  const MoeWeights& mo = w.mtp.moe;
  if (!Upload(w.mtp_head.eh_proj, &g.eh_proj, error_msg) ||
      !Upload(mw.wq, &g.attn_q, error_msg) ||
      !Upload(mw.wkv_a_mqa, &g.attn_kv_a_mqa, error_msg) ||
      !Upload(mw.wqkv_gate, &g.attn_gate, error_msg) ||
      !Upload(mw.wo, &g.attn_output, error_msg) ||
      !Upload(mo.gate_inp, &g.router, error_msg) ||
      !Upload(mo.gate_shexp, &g.ffn_gate_shexp, error_msg) ||
      !Upload(mo.up_shexp, &g.ffn_up_shexp, error_msg) ||
      !Upload(mo.down_shexp, &g.ffn_down_shexp, error_msg) ||
      !Upload(mo.gate_exps, &g.ex_gate, error_msg) ||
      !Upload(mo.up_exps, &g.ex_up, error_msg) ||
      !Upload(mo.down_exps, &g.ex_down, error_msg) ||
      !UploadVec(w.mtp.attn_norm, &g.attn_norm, error_msg) ||
      !UploadVec(w.mtp.ffn_norm, &g.ffn_norm, error_msg) ||
      !UploadVec(mw.kv_a_norm, &g.kv_a_norm, error_msg) ||
      !UploadVec(mo.probs_bias, &g.probs_bias, error_msg) ||
      !UploadVec(w.mtp_head.enorm, &g.enorm, error_msg) ||
      !UploadVec(w.mtp_head.hnorm, &g.hnorm, error_msg)) {
    Missing = error_msg != nullptr ? *error_msg : "mtp: weight upload failed";
    return false;
  }

  if (head_norm == HeadNorm::kLayerOutNorm) {
    if (w.mtp_head.layer_out_norm.empty()) {
      return fail("mtp: blk.42.layer_output_norm is absent; cannot A/B it");
    }
    if (!UploadVec(w.mtp_head.layer_out_norm, &g.head_norm, error_msg)) {
      Missing =
          error_msg != nullptr ? *error_msg : "mtp: head norm upload failed";
      return false;
    }
  } else {
    // Not an upload: it is the trunk's, borrowed.
    g.head_norm = shared.output_norm;
  }
  g.output_norm = shared.output_norm;

  // wk_b / wv_b: MlaWeights wants float*, so mirror through ling3_dequant.
  // The packed copy is released afterwards -- only the F32 mirror stays.
  const auto mirror = [&](const TensorRef& t, const float** slot) {
    W packed;
    if (!Upload(t, &packed, error_msg))
      return false;
    const int rows_all = packed.rows * packed.experts;
    const std::size_t bytes =
        static_cast<std::size_t>(rows_all) * packed.cols * sizeof(float);
    float* f = static_cast<float*>(DevAlloc(bytes));
    if (f == nullptr || ling3_dequant(packed.type, packed.d, f, packed.cols,
                                      rows_all) != LING3_OK) {
      if (error_msg != nullptr)
        *error_msg = "mtp: wk_b/wv_b F32 mirror failed";
      return false;
    }
    g.resident += bytes + kPad;
    hipFree(const_cast<void*>(packed.d));
    g.resident -= packed.bytes + kPad;  // the packed copy is gone again
    *slot = f;
    return true;
  };
  if (!mirror(mw.wk_b, &g.wk_b_f32) || !mirror(mw.wv_b, &g.wv_b_f32)) {
    Missing = error_msg != nullptr ? *error_msg : "mtp: F32 mirror failed";
    return false;
  }

  g.d_ids = static_cast<int*>(DevAlloc(kMaxExperts * sizeof(int)));
  g.d_weights = static_cast<float*>(DevAlloc(kMaxExperts * sizeof(float)));
  if (g.d_ids == nullptr || g.d_weights == nullptr) {
    return fail("mtp: router scratch allocation");
  }
  g.resident += kMaxExperts * (sizeof(int) + sizeof(float)) + 2 * kPad;

  if (resident_bytes != nullptr)
    *resident_bytes = g.resident;
  Missing.clear();
  g.ready = true;
  return true;
}

bool Ready() noexcept {
  return g.ready;
}

std::string MissingPieces() {
  return g.ready ? std::string() : Missing;
}

// ---------------------------------------------------------------------------

State* NewState(std::uint32_t max_ctx) {
  if (!g.ready || max_ctx == 0) {
    return nullptr;
  }
  State* st = new State();
  st->max_ctx = max_ctx;

  const std::size_t H = static_cast<std::size_t>(g.hidden);
  const int kv_a = 512 + 64;
  st->mla_bytes = static_cast<std::size_t>(max_ctx) *
                  static_cast<std::size_t>(kv_a) * sizeof(float);
  st->mla_cache = static_cast<float*>(DevAlloc(st->mla_bytes));
  if (st->mla_cache == nullptr) {
    DestroyState(st);
    return nullptr;
  }
  st->allocs.push_back(st->mla_cache);

  // Scratch. k is sized for the raw MLA output (o_dim = heads * head_v = 4096),
  // NOT hidden: writing 4096 floats into a 2560-float buffer is the GPU fault
  // rocm_backend.cpp:780 records.
  const int o_dim = g.heads * 128;
  const int q_dim = g.heads * 192;
  const bool ok =
      AllocF(&st->h_prev, H) && AllocF(&st->e_raw, H) &&
      AllocF(&st->enorm, H) && AllocF(&st->hnorm, H) &&
      AllocF(&st->concat, 2 * H) && AllocF(&st->cur, H) &&
      AllocF(&st->normed, H) &&
      AllocF(&st->q, static_cast<std::size_t>(q_dim) + kv_a + g.heads) &&
      AllocF(&st->k, o_dim > 0 ? o_dim : 1) && AllocF(&st->gate, g.heads) &&
      AllocF(&st->attn_out, H) && AllocF(&st->ffn_in, H) && AllocF(&st->t, H) &&
      AllocF(&st->y, H) && AllocF(&st->f, H) && AllocF(&st->h_out, H) &&
      AllocF(&st->logits, g.vocab);
  if (!ok) {
    DestroyState(st);
    return nullptr;
  }
  for (float* p : {st->h_prev, st->e_raw, st->enorm, st->hnorm, st->concat,
                   st->cur, st->normed, st->q, st->k, st->gate, st->attn_out,
                   st->ffn_in, st->t, st->y, st->f, st->h_out, st->logits}) {
    st->allocs.push_back(p);
  }
  st->tok_in = static_cast<int*>(DevAlloc(sizeof(int)));
  // int[2]: [argmax, float-bits of p_top1]. One 8-byte readback gets both
  // (SPEC6 item 3) -- the draft loop pays one sync per token already.
  st->tok_out = static_cast<int*>(DevAlloc(2 * sizeof(int)));
  st->pos_d = static_cast<int*>(DevAlloc(sizeof(int)));
  if (st->tok_in == nullptr || st->tok_out == nullptr || st->pos_d == nullptr) {
    DestroyState(st);
    return nullptr;
  }
  st->allocs.push_back(st->tok_in);
  st->allocs.push_back(st->tok_out);
  st->allocs.push_back(st->pos_d);
  return st;
}

void ResetState(State* st) noexcept {
  if (st == nullptr)
    return;
  DevZero(st->mla_cache, st->mla_bytes);
  st->pos = 0;
  st->base_pos = 0;
  st->base_set = false;
}

/// LS4 phase 2: window the cache to positions the head actually writes.
void SetWindowed(State* st, bool windowed) noexcept {
  if (st == nullptr)
    return;
  st->windowed = windowed;
  st->base_pos = 0;
  st->base_set = false;
}

/// SPEC7 item 2: LING3_MTP_WINDOW=W. rows <= 0 attends over the whole cache.
void SetAttnWindow(State* st, int rows) noexcept {
  if (st == nullptr)
    return;
  st->attn_win = rows > 0 ? rows : 0;
}

void DestroyState(State* st) noexcept {
  if (st == nullptr)
    return;
  st->Release();
  delete st;
}

// ---------------------------------------------------------------------------

/// The MTP block, without the LM head when `draft_token` is null.
static bool HeadBlock(State* st, std::int32_t x_next,
                      std::span<const float> h_prev, std::uint32_t pos,
                      std::int32_t* draft_token, std::span<float> h_out,
                      std::string* error_msg, float* top1_p);

bool DraftNext(State* st, std::int32_t x_next, std::span<const float> h_prev,
               std::uint32_t pos, std::int32_t* draft_token,
               std::span<float> h_out, std::string* error_msg, float* top1_p) {
  return HeadBlock(st, x_next, h_prev, pos, draft_token, h_out, error_msg,
                   top1_p);
}

bool PrimeNext(State* st, std::int32_t x_next, std::span<const float> h_prev,
               std::uint32_t pos, std::span<float> h_out,
               std::string* error_msg) {
  // top1_p = nullptr: the row-by-row priming fallback and the probe path never
  // ask for a confidence, so the exponent pass is skipped (SPEC6 item 3).
  return HeadBlock(st, x_next, h_prev, pos, nullptr, h_out, error_msg, nullptr);
}

// ---------------------------------------------------------------------------
// LM1: batched priming (DECISIONS 00:30; RV5 change 1 + SPEC5, 2026-10-03).
//
// WHY THIS IS NOT THE SMALL-M KERNELS. SPEC5 item 2 said to strip every
// LS1_MAX_M-capped call into rows of <= 4. That cannot work on this artifact:
// the ENTIRE MTP head (blk.42) is Q8_0 (13 Q8_0 tensors -- eh_proj, attn_q,
// attn_kv_a_mqa, attn_gate, attn_output, the three exps stacks and the three
// shexp matrices) while the trunk is ROCmFP4. All four of those kernels reject
// a non-ROCmFP4 qtype at the door, INDEPENDENT of m:
//     ling3_gemv_q8_m   : ling3_gemv_q8.hip:1028  qtype != ROCmFP4 -> ERR_ARG
//     ling3_quant_m     : :1037                  (feeds only the above)
//     ling3_expert_ffn_m: :1068                  qtype != ROCmFP4 -> ERR_ARG
//     ling3_gemv_m      : wraps ling3_gemv_q8_m, same rejection
// So PrimeBatch returned fail("eh_proj") on EVERY chunk -- MEASURED at
// prompt_tokens=5, i.e. m=5 and at the m<=4 strips too. The m cap was never
// the binding constraint; the qtype gate is. ls1_load_blk32 decodes the FP4
// codebook unconditionally, so there is no small-m Q8_0 path to strip into.
//
// WHAT PRIMING ACTUALLY NEEDS. The only thing prefill must leave behind is the
// head's own MLA latent cache. mla_block writes it in step 3 (mla_cache_append)
// from kv_a, and step 4 (mla_attn) only READS the cache to produce `out`. The
// attention output, the MoE FFN and the head norm exist purely to produce
// h_out, which prefill DISCARDS -- the fork's process() throws the head's
// output away too, and EvaluateBatch never asked for it. So priming is
// the chain
//     embed -> enorm/hnorm -> eh_proj -> attn_norm -> attn_kv_a_mqa
//     -> rope(kv key) -> latent_norm -> cache_append
// and nothing else. Dropping the tail removes the whole expert FFN (n_used
// weight matrices per row) and with it the router readbacks.
//
// KERNELS. The head is Q8_0. The first cut used per-row ling3_gemv (the exact
// call HeadBlock makes, bit-identical cache rows) and cost 8.5-20 % of prefill;
// the second called LX's ling3_gemm and cost 5x MORE than that (see the call
// site). Both re-read every weight row once per PROMPT TOKEN. The projections
// now go through ling3_gemm_fast, LS3's tiled batched kernel: one block covers
// 64 weight rows x 64 tokens x 64 k, and each weight element is dequantized
// into LDS once and reused by every token in the tile, so each weight is read
// from global ONCE PER CHUNK. mla_rope / mla_latent_norm / mla_cache_append
// all take n_tokens, so they run once over the whole chunk.
//
// Priming writes ONLY the head's own cache. No trunk KDA, no trunk MLA state,
// no emitted token: greedy text stays the trunk argmax by construction.
// ---------------------------------------------------------------------------
namespace {

/// Row workspace for PrimeBatch, sized once for kMaxPrimeBatch rows.
struct PrimeScratch {
  float* e_raw{nullptr};   // [m][H]   embed
  float* concat{nullptr};  // [m][2H]  [e_norm ; h_norm]
  float* h_in{nullptr};    // [m][H]   h_prev staging (stride H source)
  float* cur{nullptr};     // [m][H]   eh_proj output
  float* normed{nullptr};  // [m][H]   attn_norm output
  float* kv_a{nullptr};    // [m][KVA] the latent that fills the cache
  std::int32_t* tok_in{nullptr};
  std::int32_t* pos_d{nullptr};
  void* pool{nullptr};
  std::size_t pool_bytes{0};
  int rows{0};
};
PrimeScratch g_pr;
constexpr int kMaxPrimeBatch = 512;

/// How much of the pool a chunk of `m` rows needs, matching the carve-out in
/// PrimeScratchReady exactly. Used by the host-side audit so a later edit to
/// the carve-out cannot silently walk off the allocation.
std::size_t PrimePoolBytes(int m) {
  const std::size_t H = static_cast<std::size_t>(g.hidden);
  const std::size_t twoH = 2 * H;
  const std::size_t KVA = 576;
  const std::size_t M = static_cast<std::size_t>(m < 1 ? 1 : m);
  // Float side: e_raw H, concat 2H, h_in H, cur H, normed H, kv_a KVA.
  // Int side: tok_in, pos_d.
  const std::size_t floats = M * (H + twoH + H + H + H + KVA);
  const std::size_t ints = M * 2;
  return floats * sizeof(float) + ints * sizeof(std::int32_t) + 256;
}

bool PrimeScratchReady(int m) {
  if (g_pr.rows >= m && g_pr.pool != nullptr)
    return true;
  if (g_pr.pool != nullptr) {
    (void)hipFree(g_pr.pool);
    g_pr.pool = nullptr;
  }
  const std::size_t H = static_cast<std::size_t>(g.hidden);
  const std::size_t twoH = 2 * H;
  const std::size_t KVA = 576;
  const std::size_t M = static_cast<std::size_t>(m < 1 ? 1 : m);
  const std::size_t bytes = PrimePoolBytes(m);
  g_pr.pool = DevAlloc(bytes);
  if (g_pr.pool == nullptr)
    return false;
  char* base = static_cast<char*>(g_pr.pool);
  char* const end = base + bytes;
  auto f = [&](std::size_t n) {
    float* r = reinterpret_cast<float*>(static_cast<void*>(base));
    base += n * sizeof(float);
    return r;
  };
  auto i32 = [&](std::size_t n) {
    std::int32_t* r = reinterpret_cast<std::int32_t*>(static_cast<void*>(base));
    base += n * sizeof(std::int32_t);
    return r;
  };
  g_pr.e_raw = f(M * H);
  g_pr.concat = f(M * twoH);
  g_pr.h_in = f(M * H);
  g_pr.cur = f(M * H);
  g_pr.normed = f(M * H);
  g_pr.kv_a = f(M * KVA);
  g_pr.tok_in = i32(M);
  g_pr.pos_d = i32(M);
  // Host-side audit of the carve-out itself: every buffer must end inside the
  // allocation. LF1's lf_audit is the model -- indices are checked on the host
  // BEFORE a launch, not discovered as a GPU page fault after it.
  if (base > end) {
    (void)hipFree(g_pr.pool);
    g_pr.pool = nullptr;
    g_pr.rows = 0;
    return false;
  }
  // LM1 M1, same reason as the trunk image pool: start the pool clean so
  // nothing uninitialised can reach a live buffer.
  (void)hipMemset(g_pr.pool, 0, bytes);
  g_pr.pool_bytes = bytes;
  g_pr.rows = m < 1 ? 1 : m;
  return true;
}

}  // namespace

bool PrimeBatchEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("LING3_MTP_PRIME_BATCH");
    return e != nullptr && e[0] != '\0' && std::string(e) != "0";
  }();
  return on;
}

bool PrimeBatch(State* st, const std::int32_t* tokens, const float* h_prev,
                int n, std::uint32_t pos0, std::string* error_msg) {
  // Failures are logged ONCE per process -- not once per row, and not once per
  // chunk (a long prompt would otherwise print the same line 27 times).
  static bool logged = false;
  const auto fail = [&](const char* what) {
    if (error_msg != nullptr) {
      *error_msg = std::string("mtp: PrimeBatch ") + what;
    }
    // SPEC5: any failure leaves the head EXACTLY as un-primed -- no
    // half-filled window. Clearing base_set makes the next DraftNext re-base at
    // its own position and write cache row 0, so the rows this call did write
    // are overwritten before they can ever be attended over. Keeping base_set
    // would leave DraftNext attending from 0 over a cache that was never filled
    // (RV5: the zero-prompt dilution).
    if (st != nullptr) {
      st->base_set = false;
      st->base_pos = 0;
    }
    if (!logged) {
      logged = true;
      const char* v = std::getenv("LING3_MTP_VERBOSE");
      if (v != nullptr && v[0] != '\0' && std::string(v) != "0") {
        std::fprintf(stderr, "ling3 mtp: PrimeBatch %s (logged once)\n", what);
      }
    }
    return false;
  };
  if (!g.ready || st == nullptr || tokens == nullptr || h_prev == nullptr ||
      n <= 0) {
    return fail("bad arguments");
  }
  if (n > kMaxPrimeBatch)
    return fail("chunk larger than the batch scratch");
  if (static_cast<std::uint32_t>(n) + pos0 > st->max_ctx) {
    return fail("position past the context");
  }
  if (!PrimeScratchReady(n))
    return fail("scratch allocation");

  const std::size_t H = static_cast<std::size_t>(g.hidden);
  const std::size_t twoH = 2 * H;
  const std::size_t KVA = 576;
  const int mm = n;
  hipStream_t stream = nullptr;

  // The window base this chunk will use, computed but NOT committed. It is
  // written to `st` only after the cache rows land (RV5 fix 3).
  const int base =
      (st->windowed && !st->base_set) ? static_cast<int>(pos0) : st->base_pos;
  if (st->windowed && static_cast<int>(pos0) < base) {
    return fail("position went backwards before the window base");
  }

  // ---- Host-side index audit, before any launch. ----
  // Every token id, every absolute position and every cache row the kernels
  // below will touch is range-checked here. The pool carve-out is audited in
  // PrimeScratchReady. Nothing below may index outside what this block proved.
  if (g_pr.pool_bytes < PrimePoolBytes(mm)) {
    return fail("scratch pool smaller than the carve-out needs");
  }
  for (int t = 0; t < mm; ++t) {
    if (tokens[t] < 0 || static_cast<std::uint32_t>(tokens[t]) >= g.vocab) {
      return fail("token id out of the vocabulary");
    }
    const std::uint32_t p = pos0 + static_cast<std::uint32_t>(t);
    if (p >= st->max_ctx)
      return fail("row position past the context");
    const int row =
        st->windowed ? static_cast<int>(p) - base : static_cast<int>(p);
    if (row < 0 || row >= static_cast<int>(st->max_ctx)) {
      return fail("window row out of range");
    }
    if (g_pr.e_raw == nullptr || g_pr.concat == nullptr ||
        g_pr.h_in == nullptr || g_pr.cur == nullptr || g_pr.normed == nullptr ||
        g_pr.kv_a == nullptr || g_pr.tok_in == nullptr ||
        g_pr.pos_d == nullptr) {
      return fail("scratch buffer is null");
    }
  }
  // The cache rows this chunk will occupy: [row_first, row_last].
  const int row_first =
      st->windowed ? static_cast<int>(pos0) - base : static_cast<int>(pos0);
  const int row_last = row_first + mm - 1;

  std::vector<std::int32_t> hs(static_cast<std::size_t>(mm));
  std::vector<std::int32_t> ps(static_cast<std::size_t>(mm));
  for (int t = 0; t < mm; ++t) {
    hs[static_cast<std::size_t>(t)] = tokens[t];
    ps[static_cast<std::size_t>(t)] = static_cast<std::int32_t>(pos0) + t;
  }
  if (hipMemcpy(g_pr.tok_in, hs.data(),
                static_cast<std::size_t>(mm) * sizeof(std::int32_t),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(g_pr.pos_d, ps.data(),
                static_cast<std::size_t>(mm) * sizeof(std::int32_t),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(g_pr.h_in, h_prev,
                static_cast<std::size_t>(mm) * H * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess) {
    return fail("host->device copy");
  }

  // Steps 0-2: embed + RMSNorm both halves. concat is [m][2H]: e_norm lands at
  // stride 2H and h_norm at +H with the same stride. The contiguous
  // ops_rms_norm convenience assumes stride == n, which is wrong here -- pass
  // the strides explicitly (the first run died on an invalid-argument launch at
  // ling3_ops.hip:249 from the contiguous form). h_in holds h_prev at stride H
  // and is consumed here; nothing overwrites it before this launch runs.
  ::ling3::ops_embedding_row(g.token_embd_type, g.token_embd, g.hidden, g.vocab,
                             g_pr.tok_in, mm, g_pr.e_raw, stream);
  ::ling3::ops_rms_norm(g_pr.e_raw, g.enorm, g_pr.concat, g.hidden, mm, g.eps,
                        static_cast<std::int64_t>(H),
                        static_cast<std::int64_t>(twoH), stream);
  ::ling3::ops_rms_norm(g_pr.h_in, g.hnorm, g_pr.concat + H, g.hidden, mm,
                        g.eps, static_cast<std::int64_t>(H),
                        static_cast<std::int64_t>(twoH), stream);

  // Steps 3-5: eh_proj, then the attention norm. ONE batched GEMM per
  // projection over the whole chunk instead of one ling3_gemv per row (SPEC6
  // item 2).
  //
  // WHICH kernel matters more than batching. LX's ling3_gemm (ling3_moe.hip
  // k_gemm) is one thread per (row, token), each running a serial
  // ling3_dot_row, so it re-reads every weight row once per token AND loses
  // k_gemv_wide's 64-way tree: prefill fell from 408.7 to 78.2 tok/s at 1.6K
  // (job lm1-h1a, binary 654c077f). ling3_gemm.hpp:5-8 documents that exact
  // failure ("one-thread-per-output-row with a serial k dot ... 5x SLOWER
  // than the scalar path") and points at ling3_gemm_fast as the fix. Its tile
  // is 64 weight rows x 64 tokens x 64 k (ling3_gemm.hip:5-8): every weight
  // element is dequantized into LDS once per k-tile and reused by all 64
  // tokens, i.e. each weight is read from global once per CHUNK -- the
  // property item 2 asks for. Q8_0 is not an FP4 type, so it takes the
  // ling3_elem decode into the same LDS tile (ling3_gemm.hip:258).
  //
  // Bit-identity with row-by-row PrimeNext is NOT required (Jay DECISIONS
  // 12:38): the head's cache only affects acceptance, and the trunk verify
  // decides every emitted token, so text ON == OFF holds by construction. The
  // tile accumulates k-ascending, one += per element, in ling3_dot_row's
  // statement order (ling3_gemm.hip:14-16), so the rows are near but not bit
  // identical to the per-row path -- REPORT6 reports acceptance before/after.
  //
  // Both entry points launch on the NULL stream (ling3_gemm.hpp:23-24), and
  // every stream in this file is nullptr, so ordering with the ops below is
  // unchanged.
  //
  // These are the only two projections the cache needs; attn_q and attn_gate
  // feed mla_attn alone, and priming does not run the attention.
  if (ling3_gemm_fast(g.eh_proj.type, g.eh_proj.d, g_pr.concat, g_pr.cur,
                      g.eh_proj.cols, g.eh_proj.rows, mm) != LING3_OK) {
    return fail("eh_proj");
  }
  ::ling3::ops_rms_norm(g_pr.cur, g.attn_norm, g_pr.normed, g.hidden, mm, g.eps,
                        stream);

  // Step 6: the latent the cache stores. Batched the same way (HeadBlock
  // stages it at q + q_dim; priming has its own kv_a buffer).
  if (ling3_gemm_fast(g.attn_kv_a_mqa.type, g.attn_kv_a_mqa.d, g_pr.normed,
                      g_pr.kv_a, g.hidden, static_cast<int>(KVA),
                      mm) != LING3_OK) {
    return fail("attn_kv_a_mqa");
  }

  // Steps 7-9: rope the latent's key section, RMSNorm the 512-dim latent in
  // place, append the chunk to the cache. All three take n_tokens, so one call
  // each covers the whole chunk -- this is mla_block's steps 1b/2/3 verbatim,
  // which is all the cache write needs. mla_block's step 4 (mla_attn) is
  // skipped on purpose: it READS the cache to build the attention output that
  // prefill throws away, so running it would be pure cost.
  {
    ::ling3::MlaShape ms;
    ms.n_embd = g.hidden;
    ms.n_head = g.heads;
    ms.head_k = 192;
    ms.head_v = 128;
    ms.nope = 128;
    ms.rot = 64;
    ms.kv = 512;
    ms.rope_theta = g.rope_theta;
    ::ling3::MlaScalars msc;
    msc.rms_eps = g.eps;
    msc.kq_scale = 0.0F;
    ::ling3::MlaWeights mw;
    mw.kv_a_norm = g.kv_a_norm;
    ::ling3::MlaCache cache;
    cache.kv = st->mla_cache;
    cache.s_pos = KVA;
    cache.s_seq = 0;
    cache.max_pos = static_cast<int>(st->max_ctx);

    // make_view(p, rows, n_tokens) -> s_tok = rows, s_seq = rows * n_tokens.
    const std::int64_t s_tok = static_cast<std::int64_t>(KVA);
    const std::int64_t s_seq = s_tok * static_cast<std::int64_t>(mm);
    // RoPE over the key rope section only: s_head 0, n_head 1 (the shared key
    // part), exactly as mla_block step 1 does for the kv half.
    ::ling3::mla_rope(ms, g_pr.kv_a + ms.kv, /*s_head=*/0, /*n_head=*/1, mm,
                      s_tok, s_seq, /*n_seqs=*/1, g_pr.pos_d, stream);
    ::ling3::mla_latent_norm(ms, msc, mw, g_pr.kv_a, mm, /*n_seqs=*/1, s_tok,
                             s_seq, stream);
    // Append rows [row_first, row_last]. cache_append computes its base as
    // n_pos - n_tokens when pos_d is null, so n_pos = row_last + 1 writes
    // exactly this chunk's rows and nothing else.
    //
    // CAUTION: this signature is (sh, cache, kv_a, n_tokens, s_tok, n_seqs,
    // n_pos, stream, pos_d) -- it takes NO s_seq; the sequence stride comes
    // from cache.s_seq. Passing s_seq here lands in the n_seqs slot and
    // launches dim3(n_tokens, s_tok*n_tokens) blocks against a kv_a buffer of
    // only n_tokens rows. cache_append_kernel guards `t >= n_tokens` but has
    // no `s` guard, so the extra blocks read kv_a + s*a_seq off the end -- a
    // linear over-read, GCVM_L2_PROTECTION_FAULT. n_seqs is 1 here.
    ::ling3::mla_cache_append(ms, cache, g_pr.kv_a, mm, s_tok, /*n_seqs=*/1,
                              row_last + 1, stream);
  }

  // Every launch above is queued on the null stream in order and there is
  // nothing left to fail, so committing the window base here is the same as
  // committing it after the rows land. RV5 fix 3 wants it set only once the
  // chunk's MLA rows are written -- that is now.
  if (st->windowed && !st->base_set) {
    st->base_pos = base;
    st->base_set = true;
  }
  return true;
}
static bool HeadBlock(State* st, std::int32_t x_next,
                      std::span<const float> h_prev, std::uint32_t pos,
                      std::int32_t* draft_token, std::span<float> h_out,
                      std::string* error_msg, float* top1_p) {
  if (!g.ready || st == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "mtp: not ready";
    return false;
  }
  // A null draft_token means "priming": the LM head and the argmax are
  // skipped, because the token is thrown away and only h_out is wanted.
  const bool want_token = draft_token != nullptr;
  const std::size_t H = static_cast<std::size_t>(g.hidden);
  if (x_next < 0 || static_cast<std::uint32_t>(x_next) >= g.vocab) {
    if (error_msg != nullptr)
      *error_msg = "mtp: token id out of vocabulary";
    return false;
  }
  if (h_prev.size() < H) {
    if (error_msg != nullptr)
      *error_msg = "mtp: h_prev too short";
    return false;
  }
  if (h_out.size() < H) {
    if (error_msg != nullptr)
      *error_msg = "mtp: h_out too short";
    return false;
  }
  // One past the last writable cache row: pos + 1 rows must exist.
  if (pos >= st->max_ctx) {
    if (error_msg != nullptr)
      *error_msg = "mtp: position past the context";
    return false;
  }

  // Window the cache to the first position this state was asked to draft.
  if (st->windowed && !st->base_set) {
    st->base_pos = static_cast<int>(pos);
    st->base_set = true;
  }
  const int row = st->windowed ? static_cast<int>(pos) - st->base_pos
                               : static_cast<int>(pos);
  if (row < 0) {
    if (error_msg != nullptr) {
      *error_msg = "mtp: position went backwards before the window base";
    }
    return false;
  }
  const int n_pos_abs = row + 1;
  // SPEC7 item 2: the attention window. `n_pos_abs` cache rows exist from the
  // window base to `pos`; with LING3_MTP_WINDOW=W the head reads only the last
  // W of them. mla_block indexes rows from cache.kv and takes the row count as
  // an argument, so an offset base plus a clamped count is exactly a sliding
  // window: no data movement, no kernel change. The append is consistent with
  // that base -- it writes at relative row n_pos - n_tokens, which is absolute
  // row attn_start + n_pos - n_tokens == n_pos_abs - n_tokens, the row this
  // call always wrote. RoPE still uses the TRUE position (pos_d), and the
  // cache rows hold keys rotated at their own true positions, so the relative
  // geometry the scores see is unchanged.
  int attn_start = 0;
  if (st->attn_win > 0 && n_pos_abs > st->attn_win) {
    attn_start = n_pos_abs - st->attn_win;
  }
  const int n_pos = n_pos_abs - attn_start;

  hipStream_t stream = nullptr;
  const int tok = static_cast<int>(x_next);
  if (!DevCopy(st->tok_in, &tok, sizeof(int)) ||
      !DevCopy(st->h_prev, h_prev.data(), F(H))) {
    if (error_msg != nullptr)
      *error_msg = "mtp: host->device copy";
    return false;
  }

  // Steps 0-2: embed x_next, RMSNorm both halves.
  // `token_embd` is the trunk's K-quant table (borrowed).
  ::ling3::ops_embedding_row(g.token_embd_type, g.token_embd, g.hidden, g.vocab,
                             st->tok_in, 1, st->e_raw, stream);
  ::ling3::ops_rms_norm(st->e_raw, g.enorm, st->concat, g.hidden, 1, g.eps,
                        stream);
  ::ling3::ops_rms_norm(st->h_prev, g.hnorm, st->concat + H, g.hidden, 1, g.eps,
                        stream);

  // Steps 3-4: [e_norm ; h_norm] -> eh_proj. inpSA is the eh_proj output,
  // BEFORE the attention norm -- that is the residual the fork adds back.
  ling3_gemv(g.eh_proj.type, g.eh_proj.d, st->concat, st->cur, g.eh_proj.cols,
             g.eh_proj.rows);
  if (!DevCopy(st->t, st->cur, F(H))) {
    if (error_msg != nullptr)
      *error_msg = "mtp: inpSA snapshot";
    return false;
  }

  // Steps 6-19: the gated-MLA block, on the MTP block's own latent cache.
  ::ling3::ops_rms_norm(st->cur, g.attn_norm, st->normed, g.hidden, 1, g.eps,
                        stream);
  const int q_dim = g.heads * 192;
  const int kv_a = 512 + 64;
  ling3_gemv(g.attn_q.type, g.attn_q.d, st->normed, st->q, g.hidden, q_dim);
  ling3_gemv(g.attn_kv_a_mqa.type, g.attn_kv_a_mqa.d, st->normed, st->q + q_dim,
             g.hidden, kv_a);
  ling3_gemv(g.attn_gate.type, g.attn_gate.d, st->normed, st->gate, g.hidden,
             g.heads);

  ::ling3::MlaShape ms;
  ms.n_embd = g.hidden;
  ms.n_head = g.heads;
  ms.head_k = 192;
  ms.head_v = 128;
  ms.nope = 128;
  ms.rot = 64;
  ms.kv = 512;
  ms.rope_theta = g.rope_theta;
  ::ling3::MlaScalars msc;
  msc.rms_eps = g.eps;
  msc.kq_scale = 0.0F;  // 0 => 1/sqrt(head_k)

  ::ling3::MlaWeights mw;
  mw.wk_b = g.wk_b_f32;
  mw.wv_b = g.wv_b_f32;
  mw.kv_a_norm = g.kv_a_norm;

  ::ling3::MlaCache cache;
  // SPEC7 item 2: the base slides with the attention window. Rows are laid out
  // [kv_a] apart from st->mla_cache, so a row offset is a pointer offset.
  cache.kv = st->mla_cache + static_cast<std::size_t>(attn_start) *
                                 static_cast<std::size_t>(kv_a);
  cache.s_pos = kv_a;
  cache.s_seq = 0;
  // The window never exceeds the context, so the allocation is unchanged.
  cache.max_pos = static_cast<int>(st->max_ctx);

  ::ling3::MlaInputs in;
  in.q = ::ling3::make_view(st->q, q_dim, 1);
  in.kv_a = ::ling3::make_view(st->q + q_dim, kv_a, 1);
  in.gate = ::ling3::make_view(st->gate, g.heads, 1);
  in.n_tokens = 1;
  in.n_seqs = 1;

  // `pos` MUST be device memory: mla_block's RoPE kernel dereferences it.
  const int pos_host = static_cast<int>(pos);
  if (!DevCopy(st->pos_d, &pos_host, sizeof(int))) {
    if (error_msg != nullptr)
      *error_msg = "mtp: position upload";
    return false;
  }
  // Persistent softmax-score scratch. The no-scratch `mla_block` overload
  // hipMallocs one per call and then stream-syncs before freeing it (an alloc
  // and a pipeline drain on EVERY draft step); this buffer is reused instead.
  // Sized for the widest row count this state can ask for -- the window when
  // one is set, otherwise the whole cache.
  const int win_cap =
      st->attn_win > 0 ? st->attn_win : static_cast<int>(st->max_ctx);
  int score_rows = win_cap < static_cast<int>(st->max_ctx)
                       ? win_cap
                       : static_cast<int>(st->max_ctx);
  // mla_block_scratch ABORTS the process on a short scratch, so the capacity is
  // never below the row count this call will ask for.
  if (score_rows < n_pos)
    score_rows = n_pos;
  if (st->attn_scores != nullptr && st->attn_scores_rows < score_rows) {
    (void)hipFree(st->attn_scores);
    st->attn_scores = nullptr;
    st->attn_scores_rows = 0;
  }
  if (st->attn_scores == nullptr) {
    const std::size_t bytes =
        ::ling3::mla_block_scratch_bytes(ms, 1, score_rows);
    st->attn_scores = static_cast<float*>(DevAlloc(bytes));
    if (st->attn_scores == nullptr) {
      if (error_msg != nullptr)
        *error_msg = "mtp: attention score scratch";
      return false;
    }
    st->attn_scores_rows = score_rows;
  }
  // Raw MLA output lands in st->k (o_dim = 4096 floats), then attn_output
  // projects it into st->attn_out. Do NOT project into a hidden-sized buffer.
  // `st->pos_d` carries the TRUE position (RoPE); `n_pos` is the window row
  // count (SPEC7 item 2: at most attn_win rows, from attn_start on). mla_block
  // indexes the cache from cache.kv and takes RoPE from pos[].
  ::ling3::mla_block(ms, msc, mw, cache, in, st->pos_d, n_pos,
                     ::ling3::make_view(st->k, ms.o_dim(), 1), st->attn_scores,
                     stream);
  ling3_gemv(g.attn_output.type, g.attn_output.d, st->k, st->attn_out,
             ms.o_dim(), g.hidden);

  // Step 20: ffn_in = inpSA + attn_out.
  ::ling3::ops_residual_add(st->t, st->attn_out, st->ffn_in, g.hidden, 1,
                            stream);

  // Steps 21-24: FFN norm, MoE + shared expert, residual.
  ::ling3::ops_rms_norm(st->ffn_in, g.ffn_norm, st->normed, g.hidden, 1, g.eps,
                        stream);
  DevZero(st->y, F(H));
  ling3_route(st->normed, static_cast<const float*>(g.router.d), g.probs_bias,
              g.hidden, g.n_expert, g.n_used, g.n_groups, g.n_group_used,
              g.w_scale, /*norm_w=*/1, g.d_ids, g.d_weights);
  int ids[kMaxExperts];
  float wts[kMaxExperts];
  if (hipMemcpy(ids, g.d_ids, sizeof(ids), hipMemcpyDeviceToHost) !=
          hipSuccess ||
      hipMemcpy(wts, g.d_weights, sizeof(wts), hipMemcpyDeviceToHost) !=
          hipSuccess) {
    if (error_msg != nullptr)
      *error_msg = "mtp: router readback";
    return false;
  }
  ling3_expert_ffn(g.ex_gate.type, g.ex_gate.d, g.ex_up.d, g.ex_down.d,
                   st->normed, ids, wts, 1, g.n_used, g.n_expert, g.hidden,
                   g.expert_ff, g.clamp_exp, st->y);
  // ling3_expert_ffn OVERWRITES y, so the shared expert goes to its own buffer
  // and is added (rocm_backend.cpp:890 records this bug).
  ling3_dense_ffn(g.ffn_gate_shexp.type, g.ffn_gate_shexp.d, g.ffn_up_shexp.d,
                  g.ffn_down_shexp.d, st->normed, 1, g.hidden, g.expert_ff,
                  g.clamp_shexp, st->f);
  ling3_vec_add(st->y, st->f, st->y, g.hidden);
  ::ling3::ops_residual_add(st->ffn_in, st->y, st->cur, g.hidden, 1, stream);

  // Step 25: the final norm, then step 26-27 through the shared LM head.
  ::ling3::ops_rms_norm(st->cur, g.head_norm, st->h_out, g.hidden, 1, g.eps,
                        stream);

  int best = -1;
  float best_p = 1.0F;
  if (want_token) {
    ling3_gemv(g.output_type, g.output, st->h_out, st->logits, g.hidden,
               g.vocab);
    // SPEC6 item 3 (RV14 3.2): argmax and its top-1 softmax probability in one
    // launch, one readback. p = 1 / sum_i exp(x_i - x_max), exact to fp32.
    if (top1_p != nullptr) {
      ::ling3::ops_argmax_prob(st->logits, g.vocab, 1, st->tok_out,
                               reinterpret_cast<float*>(st->tok_out + 1),
                               g.vocab, stream);
      std::int32_t got[2] = {0, 0};
      if (hipMemcpy(got, st->tok_out, sizeof(got), hipMemcpyDeviceToHost) !=
          hipSuccess) {
        if (error_msg != nullptr)
          *error_msg = "mtp: argmax readback";
        return false;
      }
      best = static_cast<int>(got[0]);
      std::memcpy(&best_p, &got[1], sizeof(float));
    } else {
      ::ling3::ops_argmax(st->logits, g.vocab, 1, st->tok_out, stream);
      if (hipMemcpy(&best, st->tok_out, sizeof(int), hipMemcpyDeviceToHost) !=
          hipSuccess) {
        if (error_msg != nullptr)
          *error_msg = "mtp: argmax readback";
        return false;
      }
    }
  }
  if (hipMemcpy(h_out.data(), st->h_out, F(H), hipMemcpyDeviceToHost) !=
      hipSuccess) {
    if (error_msg != nullptr)
      *error_msg = "mtp: hidden readback";
    return false;
  }
  if (want_token) {
    if (best < 0 || best >= g.vocab) {
      if (error_msg != nullptr)
        *error_msg = "mtp: argmax out of vocabulary";
      return false;
    }
    *draft_token = static_cast<std::int32_t>(best);
    if (top1_p != nullptr)
      *top1_p = best_p;
  }
  st->pos = static_cast<int>(pos) + 1;
  return true;
}

// ---------------------------------------------------------------------------
// Phase-1 acceptance probe.
// ---------------------------------------------------------------------------

namespace {

/// One probe state, because the serve layer is single-sequence (-j 1 in
/// bench.sh). A second concurrent sequence would need its own; the counters
/// below are global for the same reason and say so in the report line.
struct Probe {
  bool enabled{false};
  State* state{nullptr};
  std::uint32_t max_ctx{0};
  int compared{0};
  int accepted{0};
  double total_ms{0.0};
  int drafted{0};
  /// The draft made at the previous position, and whether it was a decode
  /// draft worth scoring against the token produced at this position.
  std::int32_t prev_draft{-1};
  bool prev_scored{false};
  hipEvent_t ev_a{nullptr}, ev_b{nullptr};
};

Probe& P() {
  static Probe p;
  return p;
}

}  // namespace

bool ProbeEnabled() noexcept {
  return P().enabled;
}

void ProbeInit(const ModelWeights& w, std::uint32_t max_context,
               std::string* error_msg) {
  Probe& p = P();
  if (p.enabled) {
    return;
  }
  const char* on = std::getenv("LING3_MTP_PROBE");
  if (on == nullptr || on[0] == '\0' || std::string(on) == "0") {
    return;
  }
  const char* hn = std::getenv("LING3_MTP_HEAD_NORM");
  const HeadNorm norm = (hn != nullptr && std::string(hn) == "layer_out_norm")
                            ? HeadNorm::kLayerOutNorm
                            : HeadNorm::kOutputNorm;
  std::size_t bytes = 0;
  if (!Init(w, norm, &bytes, error_msg)) {
    return;
  }
  p.state = NewState(max_context);
  if (p.state == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "mtp probe: could not allocate the draft state";
    }
    return;
  }
  p.max_ctx = max_context;
  // The head needs the target's post-output_norm hidden, which comes from the
  // trunk's Step(). Both the model load and this must have succeeded first.
  if (hipEventCreate(&p.ev_a) != hipSuccess ||
      hipEventCreate(&p.ev_b) != hipSuccess) {
    if (error_msg != nullptr)
      *error_msg = "mtp probe: hipEventCreate failed";
    ProbeShutdown();
    return;
  }
  p.enabled = true;
  std::fprintf(
      stderr,
      "mtp probe: ON  head_norm=%s  resident=%.1f MiB  state=%.1f MiB\n",
      norm == HeadNorm::kLayerOutNorm ? "blk.42.layer_output_norm"
                                      : "output_norm",
      static_cast<double>(bytes) / 1048576.0,
      static_cast<double>(StateBytes(max_context)) / 1048576.0);
  // The tally has to survive the server exiting, so print it from atexit rather
  // than from a call site that may never run.
  std::atexit([] {
    const Probe& q = P();
    if (q.compared == 0) {
      std::fprintf(stderr,
                   "MTP_PROBE drafted=%d compared=0 (no scored decode steps)\n",
                   q.drafted);
      return;
    }
    const double acc =
        static_cast<double>(q.accepted) / static_cast<double>(q.compared);
    std::fprintf(stderr,
                 "MTP_MEASURED drafted=%d compared=%d accepted=%d "
                 "acceptance=%.1f%% draft_ms_per_step=%.3f\n",
                 q.drafted, q.compared, q.accepted, 100.0 * acc,
                 q.drafted > 0 ? q.total_ms / q.drafted : 0.0);
  });
}

void ProbeStep(void* backend_state, std::int32_t x_next,
               std::span<const float> h_prev, std::uint32_t pos,
               bool have_hidden, bool scored) {
  Probe& p = P();
  (void)backend_state;
  if (!p.enabled || p.state == nullptr) {
    return;
  }
  const std::size_t H = static_cast<std::size_t>(g.hidden);

  // Position 0 is where a new sequence starts, and it is also where the MTP
  // cache must be wiped: the bench makes several requests on one server and
  // Reset() rewinds the trunk without the probe knowing, so a stale cache
  // would silently poison every draft after the first request.
  //
  // Row 0 itself is left unwritten -- there is no hidden state at position
  // -1 to condition on. Both this path and the CPU oracle skip position 0, so
  // they read the same zeroed row and agree; see MTP_NOTES.md.
  if (pos == 0) {
    // A new sequence starts here. Report the PREVIOUS one first: one pooled
    // number over five prompts (four short, one 1,387-token) is not comparable
    // with the per-prompt acceptance the CPU oracle reports, and the long
    // prompt is exactly the one that would drag it down.
    if (p.compared > 0) {
      std::fprintf(stderr,
                   "MTP_REQUEST compared=%d accepted=%d acceptance=%.1f%% "
                   "draft_ms_per_step=%.3f\n",
                   p.compared, p.accepted,
                   100.0 * static_cast<double>(p.accepted) /
                       static_cast<double>(p.compared),
                   p.drafted > 0 ? p.total_ms / p.drafted : 0.0);
    }
    p.compared = 0;
    p.accepted = 0;
    p.drafted = 0;
    p.total_ms = 0.0;
    ResetState(p.state);
    p.prev_scored = false;
    return;
  }

  // The draft made at pos-1 predicted the token that sits at `pos`, which is
  // exactly the token arriving here.
  if (p.prev_scored) {
    ++p.compared;
    if (p.prev_draft == x_next) {
      ++p.accepted;
    }
  }

  if (!have_hidden) {
    p.prev_draft = -1;
    p.prev_scored = false;
    return;
  }

  std::int32_t draft = -1;
  std::vector<float> h_new(H);
  hipEventRecord(p.ev_a);
  const bool ok =
      DraftNext(p.state, x_next, h_prev, pos, &draft,
                std::span<float>(h_new.data(), h_new.size()), nullptr);
  hipEventRecord(p.ev_b);
  if (hipEventSynchronize(p.ev_b) == hipSuccess) {
    float ms = 0.0F;
    if (hipEventElapsedTime(&ms, p.ev_a, p.ev_b) == hipSuccess) {
      p.total_ms += static_cast<double>(ms);
      ++p.drafted;
    }
  }
  p.prev_draft = ok ? draft : -1;
  p.prev_scored = ok && scored;
}

void ProbeTally(int* compared, int* accepted, double* total_ms) {
  const Probe& p = P();
  if (compared != nullptr)
    *compared = p.compared;
  if (accepted != nullptr)
    *accepted = p.accepted;
  if (total_ms != nullptr)
    *total_ms = p.total_ms;
}

void ProbeShutdown() {
  Probe& p = P();
  if (p.ev_a != nullptr)
    hipEventDestroy(p.ev_a);
  if (p.ev_b != nullptr)
    hipEventDestroy(p.ev_b);
  p.ev_a = p.ev_b = nullptr;
  if (p.state != nullptr) {
    DestroyState(p.state);
    p.state = nullptr;
  }
  p.enabled = false;
}

// ===========================================================================
// Phase 2 — speculative decoding.
// ===========================================================================

bool SpeculativeEnabled() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("LING3_MTP");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
  }();
  return on;
}

/// One session's speculative state. The head keeps its own plain MLA latent
/// cache (never rewound -- it is overwritten next step, which is exactly why
/// Ling's head is cheap to speculate with). What DOES need rewinding is the
/// trunk's KDA recurrence, so that is what the snapshot holds.
class SpecState {
public:
  State* head{nullptr};
  SpecPolicy policy;
  std::uint32_t max_ctx{0};

  // LM1 SPEC4 item 1: `kda_saved` is now the pool for the per-row trunk-state
  // IMAGES the verify fills (rocm::MultiRowKeep), not a single before-shot. A
  // rejected draft installs the image for the row it kept and never replays a
  // Step. One image per row of the block, rows 0..m-2.
  void* kda_saved{nullptr};
  std::size_t keep_cap{0};
  rocm::MultiRowKeep keep;
  std::size_t keep_hidden_span{0};
  std::size_t kda_conv_bytes{0};
  std::size_t kda_ssm_bytes{0};

  // Rolling acceptance over the last `policy.window` steps.
  int window{8};
  int win_len{0};
  int win_sum{0};
  int draft_len{2};

  hipEvent_t ev_a{nullptr}, ev_b{nullptr};

  // LM1 SPEC3: the cost guard. `plain_ms` is the rolling cost of one plain Step
  // at the current depth, sampled from the k <= 0 branch below; `spec_pt` is
  // what the speculative path costs per EMITTED token. Drafting stops for the
  // rest of the session the first time a speculative step costs more per token
  // than a plain step would have. That is what makes drafting free on the
  // prompts the head is bad at (prose, lists) -- where the acceptance policy
  // needs two steps to notice and both of them are paid at full verify price.
  // Byte-identical by construction: the guard only turns drafting off, and k <=
  // 0 is plain Step.
  double plain_ms{0.0};
  int plain_n{0};
  double spec_ms_sum{0.0};
  double spec_tok_sum{0.0};
  int spec_n{0};
  bool cost_stop{false};

  // LM1 SPEC4 item 1: the cost table. `acc_*` are the steps where every draft
  // was accepted (no state install), `rej_*` the ones that were not.
  // Accumulated process-wide because the ~Session reporter never fires --
  // sessions are pooled, so g_total stays zero no matter how much speculation
  // ran (LS4 REPORT3 §6.4; still true).
  double acc_ms{0.0};
  long acc_n{0};
  double rej_ms{0.0};
  long rej_n{0};
  double rej_tok{0.0};

  // SPEC6 item 3: the cost guard's sliding window, a ring of whole STEPS. The
  // old guard accumulated over every speculative step since the session began
  // and could only ever turn drafting OFF, so a request that started unlucky
  // ran all 64 tokens plain. This window is bounded and reversible; when it has
  // stopped drafting it re-opens one speculative sample every RetryInterval()
  // plain steps (SPEC7 item 3: depth-scaled) so the window can learn the run
  // improved. The shallow default that interval replaces was 8.
  static constexpr int kCostWin = 8;
  double cost_win_ms[kCostWin]{};
  double cost_win_tok[kCostWin]{};
  int cost_win_n{0};
  int cost_win_next{0};
  int cost_retry_n{0};
};

namespace {

double MsBetween(hipEvent_t a, hipEvent_t b) {
  float ms = 0.0F;
  if (hipEventSynchronize(b) != hipSuccess)
    return 0.0;
  if (hipEventElapsedTime(&ms, a, b) != hipSuccess)
    return 0.0;
  return static_cast<double>(ms);
}

/// LM1 SPEC4 item 1: install row `row`'s image of the trunk state. This is the
/// whole of the "rewind" now -- no Step is replayed. The MLA latent cache is
/// deliberately NOT restored: it is position-indexed, the verify appended rows
/// [pos0, pos0+m) and the next Step at `pos0+row+1` appends that row before it
/// attends, so every row the reject abandons is overwritten before it can be
/// read (mla_block / mla_block_splitk append at `n_pos - n_tokens` and attend
/// over the first `n_pos`, and `n_pos` comes from the position).
bool KeepInstall(rocm::State* trunk, const SpecState* spec, int row) {
  const rocm::MultiRowKeep& K = spec->keep;
  if (K.kda_conv == nullptr || trunk == nullptr || row < 0 || row >= K.rows) {
    return false;
  }
  const std::size_t r = static_cast<std::size_t>(row);
  // LM1 00:05: async on the null stream. The image writes were issued on the
  // null stream too, so this orders after them, and it no longer blocks the
  // host on a GPU drain per copy. KeepInstall is followed by an explicit D2H of
  // the hidden (B1), which is what actually synchronises. LV1 R6: KDA slices
  // only, the same bytes KeepImageLayer wrote. The 7 MLA slices are zeros on
  // both sides of the copy.
  const char* conv_row =
      static_cast<const char*>(K.kda_conv) + r * K.conv_bytes;
  const char* ssm_row = static_cast<const char*>(K.kda_ssm) + r * K.ssm_bytes;
  return rocm::InstallKdaSlices(trunk, conv_row, ssm_row) &&
         hipMemcpyAsync(trunk->t, K.hidden + r * spec->keep_hidden_span,
                        spec->keep_hidden_span * sizeof(float),
                        hipMemcpyDeviceToDevice) == hipSuccess;
}

void RecordAcceptance(SpecState* spec, int accepted, int drafted) {
  // SPEC6 item 3 (RV14 3): report-only tally. The one-strike shrink/grow that
  // lived here is DELETED, not repaired -- its window mixed units (win_len grew
  // by `drafted` tokens but the trim subtracted 1) and at n_max 1 the first
  // rejection set draft_len to 0, which made the grow-back branch unreachable,
  // so a single unlucky first draft switched drafting off for the whole
  // request. The p_min gate does that job per step instead: a weak step
  // verifies 1 row or 0, never 3. draft_len is fixed at policy.n_max, and
  // LING3_MTP_ADAPTIVE is a no-op kept for env-compat.
  spec->win_sum += accepted;
  spec->win_len += drafted;
  if (spec->win_len > spec->window) {
    spec->win_len -= (drafted > 0 ? drafted : 1);
    spec->win_sum -= accepted;
    if (spec->win_sum < 0)
      spec->win_sum = 0;
    if (spec->win_len < 0)
      spec->win_len = 0;
  }
}

std::int32_t Argmax(const float* v, std::size_t n) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < n; ++i) {
    if (v[i] > v[best])
      best = i;
  }
  return static_cast<std::int32_t>(best);
}

}  // namespace

std::size_t SpecStateBytes(std::uint32_t max_ctx) {
  // The head's latent cache plus the KDA snapshot. The snapshot is sized from
  // the trunk's own layout at run time; this is only the head's part.
  return StateBytes(max_ctx);
}

namespace {

int EnvInt(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0')
    return fallback;
  return std::atoi(v);
}

bool EnvOff(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && std::string(v) == "0";
}

}  // namespace

// Default LING3_MTP_MAX_POS, from the measured breakeven (REPORT3 has the
// table), rounded down. The breakeven is content-dependent and the gate is
// "drafting must never cost speed", so this is the WORST case, not the average:
//   code   wins at 1.6K (+2.1 %) and 3K (+6.2 %), loses at 6.5K (-2.3 %)
//   prose  loses at 1.6K (-1.5 %) and 3K (-2.6 %) -- one rejected draft is 2.4x
//   a
//          plain step at that depth, and at 96 tokens that alone is -1.5 %
// So the breakeven for "never costs" is below the 1.6K point: 1024. The bench's
// 41/58-token prompts sit far under it and keep the full drafting win. Raise
// the knob if the seat's turns are code and short; LS3's split-K MLA moves it.
// 0 disables the cap.
constexpr std::uint32_t kDefaultMaxPos = 1024;

bool MaxPosAllows(std::uint32_t pos) {
  static const long cap = [] {
    const char* e = std::getenv("LING3_MTP_MAX_POS");
    if (e == nullptr || e[0] == '\0')
      return static_cast<long>(kDefaultMaxPos);
    const long v = std::strtol(e, nullptr, 10);
    return v <= 0 ? 0L : v;  // 0 (or negative) explicitly disables the cap
  }();
  return cap == 0 || static_cast<long>(pos) < cap;
}

// ---------------------------------------------------------------------------
// SPEC8: drafting that never locks out.
//
// (a) structural, in depth_rule.hpp. A full-accept block is the best case at
// this depth. Cold blocks (more than 20% above the cheapest full-accept in the
// ring) do not vote. A deny needs four warm samples from two requests, then
// skips a backoff counted in requests. The probe request is allowed to draft,
// and its samples can lift the deny. A request that was skipped does not
// refresh the backoff, or the deny would never re-open.
//
// (b) observed mix is NOT here. Acceptance is the text, not the depth. The
// per-request cost ring stops a request whose window is a clear loss
// (kCostStopBand above a plain step). A window inside the band keeps
// drafting: the seat heat-pump reply opened at 1.07x and freezing there
// left the rest of the request on the plain path, under the off decode.
// ResetSpecState still clears the ring, so the next request starts allowed.
// A clear loss re-opens inside the request only under 16K. Under 8K the
// interval stays 8; 8K-16K stays 32. Buckets at 27.7K and 56K do not
// re-open inside the request (REPORT6's leak).
//
// LING3_MTP_DEPTH unset selects this rule. LING3_MTP_DEPTH=0 turns it off.
// ---------------------------------------------------------------------------
namespace {

constexpr int kDepthBuckets = depth::kBuckets;

int DepthBucket(int pos) {
  return depth::BucketOf(pos);
}

depth::Bucket g_depth[kDepthBuckets];
int g_open_bucket = -1;
bool g_skip_logged = false;

// Unset is the fixed rule (on). =0 is the control arm.
const bool g_depth_rule = [] {
  const char* e = std::getenv("LING3_MTP_DEPTH");
  return e == nullptr || e[0] == '\0' || std::string(e) != "0";
}();

/// SPEC7 interval. Under 8K this is 8. Buckets 4 and 5 do not use it while
/// the depth rule is on (the 27.7K/56K leak). Shallow buckets do.
constexpr double kCostStopBand = 1.12;
int RetryInterval(int depth_bucket) {
  if (depth_bucket <= 2)
    return 8;
  if (depth_bucket <= 4)
    return 32;
  return 64;
}

double GlobalPlainPt() {
  double ms = 0.0;
  long n = 0;
  for (int b = 0; b < kDepthBuckets; ++b) {
    ms += g_depth[b].plain_ms;
    n += g_depth[b].plain_n;
  }
  return n > 0 ? ms / static_cast<double>(n) : 0.0;
}

void CloseOpenBucket() {
  if (g_open_bucket < 0)
    return;
  depth::EndRequest(g_depth[g_open_bucket], GlobalPlainPt());
  g_open_bucket = -1;
}

bool DepthAllows(int pos) {
  if (!g_depth_rule)
    return true;
  const int b = DepthBucket(pos);
  if (g_open_bucket != b) {
    if (g_open_bucket >= 0) {
      depth::EndRequest(g_depth[g_open_bucket], GlobalPlainPt());
    }
    g_open_bucket = b;
    depth::BeginRequest(g_depth[b]);
  }
  if (!depth::DeepOpen(g_depth[b], b, GlobalPlainPt()))
    return false;
  return depth::Allows(g_depth[b]);
}

/// Log-only. Must not Begin a request (the exit printer used to call
/// DepthAllows and would have opened a bucket as a side effect).
bool DepthVerdict(int pos) {
  if (!g_depth_rule)
    return true;
  const int b = DepthBucket(pos);
  return depth::DeepOpen(g_depth[b], b, GlobalPlainPt()) &&
         depth::Allows(g_depth[b]);
}

void DepthLog(int b, double* best, double* spec, double* plain, long* warm_n,
              long* spec_n, long* plain_n, int* warm_req, int* deny_a,
              int* probe) {
  const depth::Bucket& d = g_depth[b];
  long wn = 0;
  *best = depth::BestPt(d, &wn);
  *spec = depth::SpecPt(d);
  const double local =
      d.plain_n > 0 ? d.plain_ms / static_cast<double>(d.plain_n) : 0.0;
  *plain = local > 0.0 ? local : GlobalPlainPt();
  *warm_n = wn;
  *spec_n = d.req_spec_n;
  *plain_n = d.plain_n;
  *warm_req = d.warm_requests;
  *deny_a = d.deny_a ? 1 : 0;
  *probe = d.last_probe ? 1 : 0;
}

}  // namespace

int ShortFloor() {
  const char* e = std::getenv("LING3_MTP_SHORT_FLOOR");
  if (e == nullptr || e[0] == '\0')
    return depth::kUnprovenMinTokens;
  return static_cast<int>(std::strtol(e, nullptr, 10));
}

bool DepthDraftOk(int pos, int remaining_tokens) noexcept {
  if (!g_depth_rule)
    return true;
  const int b = depth::BucketOf(pos);
  if (!depth::DeepOpen(g_depth[b], b, GlobalPlainPt())) {
    if (!g_skip_logged) {
      std::fprintf(stderr, "MTP_DEPTH skip_deep pos=%d bucket=%d\n", pos, b);
      g_skip_logged = true;
    }
    return false;
  }
  if (depth::DraftShort(g_depth[b], remaining_tokens, GlobalPlainPt(),
                        ShortFloor())) {
    return true;
  }
  if (!g_skip_logged) {
    std::fprintf(stderr, "MTP_DEPTH skip_short pos=%d remaining=%d bucket=%d\n",
                 pos, remaining_tokens, b);
    g_skip_logged = true;
  }
  return false;
}

SpecState* NewSpecState(std::uint32_t max_ctx, const SpecPolicy& policy_in,
                        std::string* error_msg) {
  static const int depth_logged = [] {
    const char* e = std::getenv("LING3_MTP_DEPTH");
    const bool unset = (e == nullptr || e[0] == '\0');
    const char* fe = std::getenv("LING3_MTP_SHORT_FLOOR");
    const bool floor_unset = (fe == nullptr || fe[0] == '\0');
    std::fprintf(stderr,
                 "MTP_DEPTH mode=%s source=%s min_tokens=%d floor_source=%s\n",
                 g_depth_rule ? "on" : "off", unset ? "unset" : e, ShortFloor(),
                 floor_unset ? "unset" : fe);
    return 1;
  }();
  (void)depth_logged;
  // LING3_MTP_NMAX pins the draft length and LING3_MTP_ADAPTIVE=0 turns the
  // rolling stop off. Both exist so acceptance can be MEASURED at a fixed k:
  // with the adaptive stop live, two rejections switch drafting off and the
  // run reports three samples, which says nothing.
  SpecPolicy policy = policy_in;
  policy.n_max = EnvInt("LING3_MTP_NMAX", policy.n_max);
  // SPEC6 item 3: the p_min confidence gate (RV14 3.3). Clamped to [0,1]; 0
  // disables the gate (every candidate is kept, i.e. fixed n_max).
  {
    const char* e = std::getenv("LING3_MTP_PMIN");
    double v = (e != nullptr && e[0] != '\0') ? std::atof(e) : policy.p_min;
    if (v < 0.0)
      v = 0.0;
    if (v > 1.0)
      v = 1.0;
    policy.p_min = v;
  }
  // LING3_MTP_ADAPTIVE is now a NO-OP (SPEC6 item 3): the shrink/grow branches
  // it used to disable are deleted. The knob is read and ignored so existing
  // seat env keeps working.
  (void)EnvOff("LING3_MTP_ADAPTIVE");
  if (policy.n_max < 0)
    policy.n_max = 0;
  if (policy.n_max > 3)
    policy.n_max = 3;
  if (!g.ready) {
    if (error_msg != nullptr)
      *error_msg = "mtp: head not initialised";
    return nullptr;
  }
  SpecState* s = new SpecState();
  s->policy = policy;
  s->max_ctx = max_ctx;
  s->window = policy.window > 0 ? policy.window : 8;
  s->draft_len = policy.n_max;
  s->head = NewState(max_ctx);
  if (s->head != nullptr) {
    // SPEC7 item 2 redefines LING3_MTP_WINDOW as a ROW COUNT (it used to be a
    // boolean for the cache REBASE). > 0 is the sliding attention window the
    // head reads; 0 / unset is the whole cache, which is the pre-SPEC7
    // behaviour and what an unset seat env gets. A negative value keeps the old
    // diagnostic: absolute cache positions, no rebase.
    const int win_rows = EnvInt("LING3_MTP_WINDOW", 0);
    SetAttnWindow(s->head, win_rows > 0 ? win_rows : 0);
    SetWindowed(s->head, /*windowed=*/win_rows >= 0);
  }
  if (s->head == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "mtp: head state allocation failed";
    delete s;
    return nullptr;
  }
  // Sized generously for any Ling-like model: the snapshot is allocated once.
  // The exact size comes from the trunk's layout when a step first runs, so
  // allocate the worst case this model can need and fail loudly if it does
  // not fit.
  const std::size_t bytes = 128u << 20;  // 128 MiB covers 35 KDA layers here
  s->kda_saved = DevAlloc(bytes);
  if (s->kda_saved == nullptr || hipEventCreate(&s->ev_a) != hipSuccess ||
      hipEventCreate(&s->ev_b) != hipSuccess) {
    if (error_msg != nullptr)
      *error_msg = "mtp: spec state allocation failed";
    DestroySpecState(s);
    return nullptr;
  }
  return s;
}

State* SpecHead(SpecState* spec) noexcept {
  return spec == nullptr ? nullptr : spec->head;
}

void DestroySpecState(SpecState* spec) noexcept {
  if (spec == nullptr)
    return;
  if (spec->ev_a != nullptr)
    (void)hipEventDestroy(spec->ev_a);
  if (spec->ev_b != nullptr)
    (void)hipEventDestroy(spec->ev_b);
  if (spec->kda_saved != nullptr)
    (void)hipFree(spec->kda_saved);
  if (spec->head != nullptr)
    DestroyState(spec->head);
  delete spec;
}

void ResetSpecState(SpecState* spec) noexcept {
  if (spec == nullptr)
    return;
  // The head's latent cache is per-sequence; ResetState zeroes it (~150 MB)
  // and rewinds pos / base_pos / base_set. `windowed` is kept.
  if (spec->head != nullptr)
    ResetState(spec->head);
  // Rolling acceptance history and the cost guard are per-sequence too
  // (initial values as in NewSpecState). acc_* / rej_* are process-wide
  // diagnostics and stay cumulative.
  spec->win_len = 0;
  spec->win_sum = 0;
  spec->draft_len = spec->policy.n_max;
  spec->plain_ms = 0.0;
  spec->plain_n = 0;
  spec->spec_ms_sum = 0.0;
  spec->spec_tok_sum = 0.0;
  spec->spec_n = 0;
  spec->cost_stop = false;
  // RV18 F2: the sliding window and the re-probe phase are per-sequence too.
  // Without this a later request inherits the previous one's ring and the guard
  // decides on its first drafting step from stale entries.
  spec->cost_win_n = 0;
  spec->cost_win_next = 0;
  spec->cost_retry_n = 0;
  for (int i = 0; i < SpecState::kCostWin; ++i) {
    spec->cost_win_ms[i] = 0.0;
    spec->cost_win_tok[i] = 0.0;
  }
  g_skip_logged = false;
}

int SpecDraftLength(const SpecState* spec) noexcept {
  return spec == nullptr ? 0 : spec->draft_len;
}

namespace {

/// Process-wide totals: the scheduler destroys a session per request, so a
/// per-session counter would only ever describe the last one.
SpecStats g_total;
std::atomic<int> g_reported{0};
double g_cost_plain_ms = 0.0;
long g_cost_plain_n = 0;
double g_cost_acc_ms = 0.0;
long g_cost_acc_n = 0;
double g_cost_rej_ms = 0.0;
long g_cost_rej_n = 0;
double g_cost_rej_tok = 0.0;

}  // namespace

/// One MTP_SPEC_MEASURED line for `s`. `tag` is a label prefix that already
/// ends in a space ("req=3 "), or "" for the process-wide aggregate -- the
/// aggregate keeps its exact historical format so nothing that greps
/// MTP_SPEC_MEASURED has to care that per-request rows now exist.
void PrintMeasured(const SpecStats& s, const char* tag) {
  if (s.steps == 0) {
    std::fprintf(stderr, "MTP_SPEC %ssteps=0 (speculation never ran)\n", tag);
    return;
  }
  const double acc = s.drafted_total > 0
                         ? 100.0 * static_cast<double>(s.accepted_total) /
                               static_cast<double>(s.drafted_total)
                         : 0.0;
  std::fprintf(stderr,
               "MTP_SPEC_MEASURED %ssteps=%d no_draft=%d full_accept=%d "
               "drafted=%d accepted=%d acceptance=%.1f%% "
               "tokens_per_step=%.3f draft_ms=%.3f verify_ms=%.3f "
               "rollback_ms=%.3f | per_len:",
               tag, s.steps, s.no_draft_steps, s.full_accept_steps,
               s.drafted_total, s.accepted_total, acc,
               static_cast<double>(s.drafted_total + s.steps) /
                   static_cast<double>(s.steps),
               s.draft_ms / static_cast<double>(s.steps),
               s.verify_ms / static_cast<double>(s.steps),
               s.rollback_ms / static_cast<double>(s.steps));
  for (int k = 0; k < 4; ++k) {
    if (s.drafted_at_len[k] == 0)
      continue;
    std::fprintf(stderr, " k=%d %ld/%ld=%.1f%%", k, s.accept_at_len[k],
                 s.drafted_at_len[k],
                 100.0 * static_cast<double>(s.accept_at_len[k]) /
                     static_cast<double>(s.drafted_at_len[k]));
  }
  // SPEC6 item 3: the p_min gate's k-distribution. k0 steps are the ones the
  // gate shortened to zero drafts (they ran the plain path); gate_skips is
  // the number of discarded candidates, so a step that stopped at k=1 after
  // one confident draft contributes k1_steps 1 and gate_skips 1.
  std::fprintf(stderr, " | gate: k0=%ld k1=%ld k2=%ld skips=%ld", s.k0_steps,
               s.k1_steps, s.k2_steps, s.gate_skips);
  std::fprintf(stderr, "\n");
}

void InstallSpecReporter() {
  std::atexit([] {
    CloseOpenBucket();
    // LM1 SPEC4 item 1: the cost table. g_total is never fed (see above), so
    // this reads the process-wide tally SpecStep writes. Output:
    //   plain   = one Step, from the k <= 0 branch
    //   acc/rej = one speculative step, by whether the block fully accepted
    // all in ms/step; rej/acc also give ms per emitted token.
    std::fprintf(stderr,
                 "LM1_COST plain_ms=%.3f plain_n=%ld acc_ms=%.3f acc_n=%ld "
                 "rej_ms=%.3f rej_n=%ld rej_tok=%.0f\n",
                 g_cost_plain_ms, g_cost_plain_n, g_cost_acc_ms, g_cost_acc_n,
                 g_cost_rej_ms, g_cost_rej_n, g_cost_rej_tok);
    PrintMeasured(g_total, "");
    // SPEC7 item 3: the measured cost model per n_pos bucket, in full.
    // best_pt is the full-accept block's cost per emitted token (the structural
    // case (a) tests); spec_pt is the mix that actually ran (case (b)).
    for (int b = 0; b < kDepthBuckets; ++b) {
      double best_pt = 0, spec_pt = 0, plain_pt = 0;
      long warm_n = 0, spec_n = 0, plain_n = 0;
      int warm_req = 0, deny_a = 0, probe_flag = 0;
      DepthLog(b, &best_pt, &spec_pt, &plain_pt, &warm_n, &spec_n, &plain_n,
               &warm_req, &deny_a, &probe_flag);
      const int probe = b == 0 ? 1024 : (1 << b) * 1500;
      std::fprintf(
          stderr,
          "MTP_DEPTH bucket=%d best_pt=%.3f spec_pt=%.3f plain_pt=%.3f "
          "acc_n=%ld spec_n=%ld plain_n=%ld warm_req=%d deny_a=%d "
          "probe=%d verdict=%s\n",
          b, best_pt, spec_pt, plain_pt, warm_n, spec_n, plain_n, warm_req,
          deny_a, probe_flag, DepthVerdict(probe) ? "allow" : "deny");
    }
  });
}

void ReportSpecSession(const SpecStats& s, int final_draft_len, int req,
                       int prompt_tokens) {
  // SPEC8: close the bucket before the row, so verdict is what the
  // NEXT request will see and req_* still holds this request.
  CloseOpenBucket();
  // SPEC6 item 4: LM1_COST and the drafts counters PER LENGTH, one row per
  // request. `req` counts prompts in arrival order and `prompt_tokens` is the
  // length of the prompt that produced the row, so the six curve points sort
  // themselves. The process-wide LM1_COST / MTP_SPEC_MEASURED at exit is
  // unchanged and remains the aggregate. `final_draft_len` is the policy's
  // n_max (SpecDraftLength), printed on the depth row below.
  std::fprintf(stderr,
               "LM1_COST req=%d prompt_tokens=%d plain_ms=%.3f plain_n=%d "
               "acc_ms=%.3f acc_n=%ld rej_ms=%.3f rej_n=%ld rej_tok=%.0f\n",
               req, prompt_tokens, s.plain_ms_sum, s.no_draft_steps,
               s.acc_ms_sum, s.acc_n_sum, s.rej_ms_sum, s.rej_n_sum,
               s.rej_tok_sum);
  // SPEC7 item 3: the depth model this request sat in, as it stands AFTER the
  // request (the buckets are process-wide, so the row is cumulative -- that is
  // the point: the next prompt at a new depth sees what the earlier ones paid).
  {
    const int b = DepthBucket(prompt_tokens);
    double best_pt = 0, spec_pt = 0, plain_pt = 0;
    long warm_n = 0, spec_n = 0, plain_n = 0;
    int warm_req = 0, deny_a = 0, probe_flag = 0;
    DepthLog(b, &best_pt, &spec_pt, &plain_pt, &warm_n, &spec_n, &plain_n,
             &warm_req, &deny_a, &probe_flag);
    const depth::Bucket& d = g_depth[b];
    const double req_pt =
        d.req_spec_tok > 0.0 ? d.req_spec_ms / d.req_spec_tok : 0.0;
    std::fprintf(stderr,
                 "MTP_DEPTH req=%d prompt_tokens=%d bucket=%d n_max=%d "
                 "best_pt=%.3f spec_pt=%.3f plain_pt=%.3f acc_n=%ld spec_n=%ld "
                 "plain_n=%ld warm_req=%d deny_a=%d probe=%d req_pt=%.3f "
                 "verdict=%s\n",
                 req, prompt_tokens, b, final_draft_len, best_pt, spec_pt,
                 plain_pt, warm_n, spec_n, plain_n, warm_req, deny_a,
                 probe_flag, req_pt,
                 DepthVerdict(prompt_tokens) ? "allow" : "deny");
  }
  {
    char tag[48];
    std::snprintf(tag, sizeof(tag), "req=%d prompt_tokens=%d ", req,
                  prompt_tokens);
    PrintMeasured(s, tag);
  }
  g_total.steps += s.steps;
  g_total.drafted_total += s.drafted_total;
  g_total.accepted_total += s.accepted_total;
  g_total.full_accept_steps += s.full_accept_steps;
  g_total.no_draft_steps += s.no_draft_steps;
  g_total.draft_ms += s.draft_ms;
  g_total.verify_ms += s.verify_ms;
  g_total.rollback_ms += s.rollback_ms;
  for (int k = 0; k < 4; ++k) {
    g_total.accept_at_len[k] += s.accept_at_len[k];
    g_total.drafted_at_len[k] += s.drafted_at_len[k];
  }
  g_total.gate_skips += s.gate_skips;
  g_total.k0_steps += s.k0_steps;
  g_total.k1_steps += s.k1_steps;
  g_total.k2_steps += s.k2_steps;
}

bool SpecStep(void* trunk_ptr, SpecState* spec, std::int32_t base,
              const std::vector<float>& h_prev_in, bool have_h_prev,
              std::vector<std::int32_t>* out, std::vector<float>* next_logits,
              std::vector<float>* h_prev_out, SpecStats* stats,
              const std::function<bool(std::int32_t)>& is_stop,
              std::string* error_msg) {
  if (!g.ready || spec == nullptr || spec->head == nullptr ||
      trunk_ptr == nullptr || out == nullptr || next_logits == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "mtp: SpecStep not ready";
    return false;
  }
  auto* trunk = static_cast<rocm::State*>(trunk_ptr);
  const std::size_t V = static_cast<std::size_t>(g.vocab);
  const std::size_t H = static_cast<std::size_t>(g.hidden);

  // LM1 SPEC3: past LING3_MTP_MAX_POS this becomes 0, which is the plain-Step
  // branch right below -- byte-identical by construction, and the only thing
  // the cap does. Checked on the trunk position BEFORE the step, so `cap` is
  // the position at which drafting stops.
  int k = spec->draft_len;  // 0 => ordinary single-token decode
  // SPEC7 item 3: the depth bucket this step is priced in. Captured ONCE here,
  // before any Step advances the trunk, so the plain baseline and the block
  // that follows it land in the same bucket (it is what MaxPosAllows checked).
  const int depth_bucket = DepthBucket(static_cast<int>(trunk->pos));
  if (k > 0 && !MaxPosAllows(static_cast<std::uint32_t>(trunk->pos))) {
    k = 0;
  }
  // SPEC7 item 3: the depth rule. Drafting runs only where the measured cost
  // model says a block pays at this depth; LING3_MTP_MAX_POS is the blunt
  // override and LING3_MTP_DEPTH=0 turns the model off.
  if (k > 0 && !DepthAllows(static_cast<int>(trunk->pos))) {
    k = 0;
  }
  // LM1 SPEC3 cost guard: no baseline yet -> take one plain step to get one
  // (it is what sets h_prev anyway), and a step that lost per emitted token ->
  // plain Step for the rest of the session.
  if (k > 0 && (spec->cost_stop || spec->plain_n == 0)) {
    k = 0;
  }
  // SPEC6 item 3: the sliding window the cost guard decides on, and the
  // reversible decision itself, in one place so every step that took the
  // DRAFTING path is priced the same way. That includes a k0 step, which paid
  // the draft head's forward and THEN a plain Step for a single emitted token
  // -- the most expensive thing the gate can do, and the case the guard exists
  // to stop. Without this the ring only ever saw fully-drafted steps, so a
  // gate-heavy stretch could run the whole request at draft-forward-plus-Step
  // cost and the guard would never look at it. A plain step taken because
  // drafting was ALREADY off (the k <= 0 branch) does NOT record: that is the
  // baseline the window is priced against, not the thing being priced.
  //
  // The decision is reversible in both directions: a window at or under a plain
  // step clears cost_stop again, and while it is set every RetryInterval(depth)
  // -th plain step re-opens drafting for one sample so the window can observe a
  // run that improved. Gate before firing -- no baseline yet, or the window not
  // yet full, means there is nothing to judge.
  double plain_step_ms = 0.0;
  const auto record_cost = [&](double ms, double tok, bool full_accept) {
    // SPEC7 item 3: the same sample goes into the depth model. `ms` is a whole
    // block (or a k0 probe: draft forward + the plain step it fell back to).
    // `full_accept` selects the acc_* bucket DepthAllows decides on -- the best
    // case this depth can produce.
    if (g_depth_rule) {
      // SPEC8: every block is recorded. Cold ones fall out of the warm band
      // once a cheaper full-accept arrives; a constant skip cannot see a
      // whole cold first reply (the "Say OK." then prose case).
      depth::RecordBlock(g_depth[depth_bucket], ms, tok, full_accept);
    }
    spec->cost_win_ms[spec->cost_win_next] = ms;
    spec->cost_win_tok[spec->cost_win_next] = tok;
    spec->cost_win_next = (spec->cost_win_next + 1) % SpecState::kCostWin;
    if (spec->cost_win_n < SpecState::kCostWin)
      ++spec->cost_win_n;
    if (spec->plain_n > 0 && spec->cost_win_n >= SpecState::kCostWin) {
      double w_ms = 0.0, w_tok = 0.0;
      for (int i = 0; i < SpecState::kCostWin; ++i) {
        w_ms += spec->cost_win_ms[i];
        w_tok += spec->cost_win_tok[i];
      }
      if (w_tok > 0.0) {
        const double pt = w_ms / w_tok;
        // SPEC8: inside the band is noise around a paying reply, not a stop.
        spec->cost_stop = g_depth_rule ? (pt > spec->plain_ms * kCostStopBand)
                                       : (pt > spec->plain_ms);
        // The first drafting request under 8K is the cold seat open. G1's
        // heat-pump froze at 1.12x and finished under the off decode
        // (41.49 vs 44.14); warm, the same prompt drafted 67/77 at 46.54.
        // Do not freeze that first short-context request. At 8K and deeper
        // a losing window may still stop.
        if (g_depth_rule && depth_bucket <= 2 &&
            g_depth[depth_bucket].requests <= 1) {
          spec->cost_stop = false;
        }
      }
    }
  };

  // The plain single-token path. Reached when drafting is off up front (cost
  // guard, position cap, no baseline yet) AND when the p_min gate shortens a
  // chain to zero drafts (SPEC6 item 3) -- same code either way, so a
  // gate-skipped step is byte-identical to drafting being off.
  const auto run_plain_step = [&]() -> bool {
    hipEventRecord(spec->ev_a);
    std::vector<float> step_hidden(H);
    if (!rocm::Step(base, trunk,
                    std::span<float>(next_logits->data(), next_logits->size()),
                    error_msg)) {
      return false;
    }
    hipEventRecord(spec->ev_b);
    const double step_ms = MsBetween(spec->ev_a, spec->ev_b);
    plain_step_ms = step_ms;
    if (step_ms > 0.0) {
      spec->plain_ms =
          spec->plain_n == 0 ? step_ms : 0.75 * spec->plain_ms + 0.25 * step_ms;
      ++spec->plain_n;
      g_cost_plain_ms += step_ms;
      ++g_cost_plain_n;
      if (stats != nullptr)
        stats->plain_ms_sum += step_ms;
      // SPEC7 item 3: the depth model's plain baseline. Every path that runs a
      // plain Step lands here -- including a k0 probe's fallback, which is a
      // real plain step at this depth and a valid sample of what drafting is
      // priced against.
      if (g_depth_rule) {
        depth::RecordPlain(g_depth[depth_bucket], step_ms);
      }
    }
    if (h_prev_out != nullptr &&
        rocm::ReadHidden(trunk, std::span<float>(step_hidden.data(), H),
                         nullptr)) {
      *h_prev_out = step_hidden;
    }
    if (next_logits->size() < V)
      next_logits->resize(V);
    out->assign(1, base);
    if (stats != nullptr) {
      ++stats->steps;
      ++stats->no_draft_steps;
      stats->verify_ms += step_ms;
    }
    // SPEC6 item 3: while the guard has drafting stopped, every re-probe
    // interval re-opens it for ONE speculative sample. Without this the
    // window can never observe that the run improved, and "can re-enable"
    // would be a claim rather than a mechanism. SPEC7 item 3 scales that
    // interval with DEPTH (RetryInterval): shallow it stays at the 8 steps
    // SPEC6 measured; at 27.7K/56K each re-probe was re-opening a block that
    // cannot pay and running until the ring refilled (REPORT6 §5's leak).
    // Jay 13:21: keep SPEC7's in-request re-probe at shallow depth so a
    // reply that opens in prose can still draft the code block. Under 8K
    // the interval is 8; bucket 3 (8K-16K) stays 32. The leak was the
    // re-open at 27.7K and 56K (buckets 4 and 5): those wait for the next
    // request. ResetSpecState still clears cost_stop either way.
    const bool reprobe = !g_depth_rule || depth_bucket <= 3;
    if (spec->cost_stop && reprobe) {
      ++spec->cost_retry_n;
      const int interval = RetryInterval(depth_bucket);
      if (spec->cost_retry_n >= interval) {
        spec->cost_retry_n = 0;
        spec->cost_stop = false;
        std::fprintf(stderr, "MTP_DEPTH reprobe bucket=%d interval=%d\n",
                     depth_bucket, interval);
      }
    }
    return true;
  };

  if (k <= 0) {
    return run_plain_step();
  }

  const std::size_t n = static_cast<std::size_t>(k) + 1;
  if (trunk->pos <= 0 || static_cast<std::size_t>(trunk->pos) + n >
                             static_cast<std::size_t>(trunk->max_ctx)) {
    if (error_msg != nullptr)
      *error_msg = "mtp: position out of range";
    return false;
  }
  // Where the trunk stands BEFORE the verify: it has consumed 0..pos0-1. The
  // KDA snapshot below is exactly this state, so rewinding means restoring the
  // snapshot AND putting the position back here -- not one token further on.
  const int pos0 = trunk->pos;

  // LM1 SPEC4 item 1: allocate the per-row trunk-state images the verify fills.
  // There is no before-shot any more -- a rejected draft installs the image for
  // the row it kept, so nothing is replayed and nothing is restored twice.
  if (spec->kda_conv_bytes == 0) {
    spec->kda_conv_bytes = trunk->kda_conv_elems * sizeof(float);
    spec->kda_ssm_bytes = trunk->kda_ssm_elems * sizeof(float);
    spec->keep_hidden_span = H;
    if (spec->kda_conv_bytes + spec->kda_ssm_bytes > (512u << 20)) {
      if (error_msg != nullptr) {
        *error_msg = "mtp: KDA state larger than the image pool";
      }
      return false;
    }
  }
  const int images = n - 1 > 0 ? n - 1 : 1;
  const std::size_t need =
      static_cast<std::size_t>(images) *
      (spec->kda_conv_bytes + spec->kda_ssm_bytes + H * sizeof(float));
  if (spec->keep_cap < need) {
    if (spec->kda_saved != nullptr)
      (void)hipFree(spec->kda_saved);
    spec->kda_saved = DevAlloc(need);
    spec->keep_cap = spec->kda_saved != nullptr ? need : 0;
    // LM1 M1 (01:22): zero the pool ONCE here. KeepImageLayer images only the
    // KDA layers' slices -- the MLA layers never touch theirs (zeroed at
    // NewState and left alone) -- so without this those slices are
    // uninitialised pool bytes and KeepInstall copies them straight back into
    // live trunk state. Nothing reads them today, which is why it never showed;
    // it is garbage in live state either way. Zero on alloc only: every image
    // is fully written before use.
    if (spec->kda_saved != nullptr) {
      (void)hipMemset(spec->kda_saved, 0, need);
    }
  }
  if (spec->kda_saved == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "mtp: trunk-state image allocation failed";
    }
    return false;
  }
  {
    char* base = static_cast<char*>(spec->kda_saved);
    spec->keep.kda_conv = base;
    spec->keep.kda_ssm =
        base + static_cast<std::size_t>(images) * spec->kda_conv_bytes;
    spec->keep.hidden = reinterpret_cast<float*>(
        base + static_cast<std::size_t>(images) *
                   (spec->kda_conv_bytes + spec->kda_ssm_bytes));
    spec->keep.conv_bytes = spec->kda_conv_bytes;
    spec->keep.ssm_bytes = spec->kda_ssm_bytes;
    spec->keep.rows = images;
    spec->keep_hidden_span = H;
  }

  // Draft: the head is chained -- each draft's own h_new conditions the next.
  std::vector<std::int32_t> tokens;
  tokens.reserve(n);
  tokens.push_back(base);
  if (!have_h_prev || h_prev_in.size() < H) {
    if (error_msg != nullptr)
      *error_msg = "mtp: no hidden state to draft from";
    return false;
  }
  // SPEC6 item 3: the k-distribution counters live in SpecStats (the reporter
  // reads that), but `stats` is nullable and the k0 path returns early, so they
  // are accumulated locally and flushed at both exits.
  int gate_skips = 0;
  int k0 = 0, k1 = 0, k2 = 0;
  const auto flush_k = [&]() {
    if (stats != nullptr) {
      stats->gate_skips += gate_skips;
      stats->k0_steps += k0;
      stats->k1_steps += k1;
      stats->k2_steps += k2;
    }
  };

  std::vector<float> h_prev(h_prev_in.begin(), h_prev_in.begin() + H);
  std::vector<float> h_next(H);
  hipEventRecord(spec->ev_a);
  // SPEC6 item 3 (RV14 3.3): the p_min confidence gate. `k` is the MAX width;
  // `k_this` is how many drafts survive the gate this step (0, 1 or 2). A
  // candidate under p_min is DISCARDED and the chain stops, exactly like the
  // fork (fork/speculative.cpp:1867-1871) -- it never reaches the verify batch.
  int k_this = 0;
  for (int i = 0; i < k; ++i) {
    std::int32_t d = -1;
    float d_p = 0.0F;
    if (!DraftNext(spec->head, tokens[i],
                   std::span<const float>(h_prev.data(), H),
                   static_cast<std::uint32_t>(trunk->pos) +
                       static_cast<std::uint32_t>(i),
                   &d, std::span<float>(h_next.data(), H), nullptr, &d_p)) {
      if (error_msg != nullptr)
        *error_msg = "mtp: draft failed";
      return false;
    }
    if (d_p < static_cast<float>(spec->policy.p_min)) {
      ++gate_skips;
      break;
    }
    tokens.push_back(d);
    h_prev.swap(h_next);
    ++k_this;
  }
  hipEventRecord(spec->ev_b);
  const double draft_ms = MsBetween(spec->ev_a, spec->ev_b);

  if (k_this == 0) {
    // The gate rejected the very first candidate: there is nothing to verify.
    // The draft head already advanced its own cache for this position -- the
    // same approximation a rejected draft leaves behind, and harmless because
    // the cache only affects future acceptance while the trunk verify owns
    // every emitted token (RV14 3 invariant).
    ++k0;
    flush_k();
    const bool ok = run_plain_step();
    // It still paid the draft forward, so it is a DRAFTING-path step and goes
    // in the window at one emitted token (see record_cost).
    if (ok && plain_step_ms > 0.0) {
      record_cost(draft_ms + plain_step_ms, 1.0, /*full_accept=*/false);
    }
    // RV18 F3: price the probe. run_plain_step has already booked the Step into
    // plain_ms_sum / no_draft_steps, so only the draft forward goes on the rej
    // tally, at 1.0 emitted token -- adding the whole block would count the
    // plain half twice.
    if (ok && stats != nullptr) {
      stats->draft_ms += draft_ms;
      stats->rej_ms_sum += draft_ms;
      stats->rej_tok_sum += 1.0;
      ++stats->rej_n_sum;
    }
    return ok;
  }

  // Verify: ONE dispatch of 1 + k rows through StepMultiDispatch, which today
  // runs the same per-row op sequence `Step` runs -- so every argmax here is an
  // argmax the model would itself have produced.
  std::vector<float> rows(n * V);
  hipEventRecord(spec->ev_a);
  // The verify width is 1 + k_this, not 1 + k: the gate may have shortened the
  // chain, and a shorter verify is strictly cheaper and safer (a weak step
  // verifies 1 row, never 3).
  const int n_this = k_this + 1;
  if (!rocm::StepMultiDispatch(tokens.data(), n_this, trunk,
                               std::span<float>(rows.data(), rows.size()),
                               error_msg, &spec->keep)) {
    if (error_msg != nullptr) {
      *error_msg = "mtp: verify dispatch failed: " + *error_msg;
    }
    return false;
  }
  hipEventRecord(spec->ev_b);
  const double verify_ms = MsBetween(spec->ev_a, spec->ev_b);

  // Accept the longest greedy-matching prefix of the drafts.
  int accepted = 0;
  for (int i = 0; i < k_this; ++i) {
    const std::int32_t r =
        Argmax(rows.data() + static_cast<std::size_t>(i) * V, V);
    if (tokens[static_cast<std::size_t>(i) + 1] == r) {
      ++accepted;
    } else {
      break;
    }
  }

  out->assign(tokens.begin(),
              tokens.begin() + static_cast<std::size_t>(accepted) + 1);

  // A stop token inside the block ends generation there. If it is not the last
  // token we must put the trunk back to exactly that point, or the caller's
  // checkpoint runs ahead of the tokens it published -- which the scheduler
  // rejects with "checkpoint is outside executed token history". That is what
  // failed gate (a) prompt 3 on the first primed run.
  int stop_at = -1;
  if (is_stop) {
    for (std::size_t i = 0; i < out->size(); ++i) {
      if (is_stop((*out)[i])) {
        stop_at = static_cast<int>(i);
        break;
      }
    }
  }

  // The next step drafts from the hidden of the token that ends this block.
  // `Step` leaves that in state->t, so ReadHidden is valid here -- which it is
  // NOT after a ForwardBatch, whose normalising target is its own scratch.
  std::vector<float> hidden_last(H);
  if (h_prev_out != nullptr) {
    rocm::ReadHidden(trunk, std::span<float>(hidden_last.data(), H), nullptr);
    *h_prev_out = hidden_last;
  }

  double rollback_ms = 0.0;
  // SPEC6 item 3 / RV18 F1: keyed on k_this, the width actually verified, not
  // on k the max width. `rows` holds 1 + k_this real rows (StepMultiDispatch
  // was told n_this), so the last row is index k_this; row k is unwritten
  // whenever the gate shortened the chain and would install stale trunk state.
  if (accepted == k_this) {
    // The batch's last row is exactly the distribution the trunk would have
    // produced for the next position. No rewind.
    std::copy_n(rows.data() + static_cast<std::size_t>(k_this) * V, V,
                next_logits->begin());
  } else {
    // LM1 SPEC4 item 1: NO REPLAY. The verify imaged the trunk state after row
    // `accepted`, and row `accepted`'s logits are already in `rows` -- the
    // corrected token's distribution never needed recomputing. Install the
    // image, move the position to just past the tokens we are keeping, and
    // stop. The cost of a rejection is now the image install (~1 ms), not a
    // full Step.
    hipEventRecord(spec->ev_a);
    if (!KeepInstall(trunk, spec, accepted)) {
      if (error_msg != nullptr)
        *error_msg = "mtp: state image install failed";
      return false;
    }
    trunk->pos = pos0 + accepted + 1;
    hipEventRecord(spec->ev_b);
    rollback_ms = MsBetween(spec->ev_a, spec->ev_b);
    std::copy_n(rows.data() + static_cast<std::size_t>(accepted) * V, V,
                next_logits->begin());
    if (h_prev_out != nullptr) {
      // LM1 B1 (01:22): `keep.hidden` lives inside `spec->kda_saved`, a
      // hipMalloc'd DEVICE buffer. `assign(hk, hk + H)` reads it on the host --
      // it does not fault on gfx1151 (Jay's 01:31 probe: the APU host-maps
      // hipMalloc memory) but it is unsynchronised with the image writes and
      // faults on any non-APU GPU. Explicit ordered D2H, same as int27's
      // ReadHidden after the replay.
      const float* hk = spec->keep.hidden + static_cast<std::size_t>(accepted) *
                                                spec->keep_hidden_span;
      h_prev_out->resize(H);
      if (hipMemcpy(h_prev_out->data(), hk, H * sizeof(float),
                    hipMemcpyDeviceToHost) != hipSuccess) {
        if (error_msg != nullptr)
          *error_msg = "mtp: h_prev image readback failed";
        return false;
      }
    }
  }
  if (stop_at >= 0 && stop_at + 1 < static_cast<int>(out->size())) {
    const int keep = stop_at + 1;
    // LM1 SPEC4 item 1: same, no replay -- row `keep-1`'s image and its already
    // computed logits are exactly what a stop-token rewind needs.
    hipEventRecord(spec->ev_a);
    if (!KeepInstall(trunk, spec, keep - 1)) {
      if (error_msg != nullptr) {
        *error_msg = "mtp: state image install for the stop";
      }
      return false;
    }
    trunk->pos = pos0 + keep;
    hipEventRecord(spec->ev_b);
    rollback_ms += MsBetween(spec->ev_a, spec->ev_b);
    out->resize(static_cast<std::size_t>(keep));
    std::copy_n(rows.data() + static_cast<std::size_t>(keep - 1) * V, V,
                next_logits->begin());
    if (h_prev_out != nullptr) {
      // LM1 B1: same -- device memory, explicit ordered D2H (see above).
      const float* hk = spec->keep.hidden + static_cast<std::size_t>(keep - 1) *
                                                spec->keep_hidden_span;
      h_prev_out->resize(H);
      if (hipMemcpy(h_prev_out->data(), hk, H * sizeof(float),
                    hipMemcpyDeviceToHost) != hipSuccess) {
        if (error_msg != nullptr) {
          *error_msg = "mtp: h_prev image readback for the stop failed";
        }
        return false;
      }
    }
  }
  if (next_logits->size() < V)
    next_logits->resize(V);

  // LM1 SPEC3: cost guard. `pt` is what this step cost per token it EMITTED,
  // against what a plain step at the same depth has been costing. The first
  // step that loses turns drafting off for the session, so the worst case is
  // one wasted step rather than the two the acceptance policy needs to notice.
  {
    const double emitted = static_cast<double>(accepted) + 1.0;
    const double pt = (draft_ms + verify_ms + rollback_ms) / emitted;
    // CUMULATIVE mean over the speculative steps so far, not an EMA. An EMA
    // with 0.25 weight barely recovers from one rejected draft: on the bench's
    // 58-token code prompt (95 % acceptance, five wins and one rejection in the
    // first six steps) the EMA was still above a plain step at step 4 and the
    // guard fired at `drafts=5/6`, taking +19 % to a wash. A cumulative mean
    // puts the same six steps at 26.6 ms/token against 27.8 and correctly stays
    // quiet.
    spec->spec_ms_sum += draft_ms + verify_ms + rollback_ms;
    spec->spec_tok_sum += emitted;
    ++spec->spec_n;
    // SPEC6 item 3: this step into the sliding window, then the guard (both in
    // record_cost, which the k0 path shares so every drafting-path step is
    // priced the same way).
    record_cost(draft_ms + verify_ms + rollback_ms, emitted,
                /*full_accept=*/accepted == k_this);
    // LM1 SPEC4 item 1: the cost table, by whether the block was fully
    // accepted.
    if (accepted == k_this) {
      spec->acc_ms += draft_ms + verify_ms + rollback_ms;
      ++spec->acc_n;
    } else {
      spec->rej_ms += draft_ms + verify_ms + rollback_ms;
      spec->rej_tok += emitted;
      ++spec->rej_n;
    }
    {
      const double block_ms = draft_ms + verify_ms + rollback_ms;
      if (accepted == k_this) {
        g_cost_acc_ms += block_ms;
        ++g_cost_acc_n;
        if (stats != nullptr) {
          stats->acc_ms_sum += block_ms;
          ++stats->acc_n_sum;
        }
      } else {
        g_cost_rej_ms += block_ms;
        g_cost_rej_tok += emitted;
        ++g_cost_rej_n;
        if (stats != nullptr) {
          stats->rej_ms_sum += block_ms;
          stats->rej_tok_sum += emitted;
          ++stats->rej_n_sum;
        }
      }
    }
    // ROLLING MEAN, not per step and not "two losses in a row". Both earlier
    // rules were wrong in opposite directions:
    //   * one step  -> fired at `drafts=5/6` on the bench's 58-token code
    //   prompt
    //                  (a rejected draft alone is 2.4x a plain step) and took
    //                  the +19 % win to a wash;
    //   * two in a row -> never fired on sampled prose, which ran 46 proposals
    //   at
    //                  74 % acceptance and still lost 3.5 %, because at k = 1
    //                  the cheap wins and the expensive rejections alternate
    //                  and no two losses are ever adjacent.
    // The loss lives in the MEAN. At k = 1 the breakeven acceptance is ~79 %:
    // pt = (draft + verify + (1-a)*replay) / (1+a) vs one plain step. Above it
    // the mean is below a plain step and the guard stays quiet; below it the
    // mean is above and it fires. Four steps is enough to see that without
    // charging much for the steps that go into the window. Eight steps is
    // enough to separate "the head is bad here" from "this run had a
    // rejection", and cheap enough to charge: at 74 % acceptance those eight
    // steps cost ~0.3 % of a 160-token generation before the guard stops them.
  }
  RecordAcceptance(spec, accepted, k_this);
  if (stats != nullptr) {
    ++stats->steps;
    // SPEC6 item 3: the k-distribution the p_min gate produced. k0_steps are
    // counted where the gate fell through to the plain path.
    if (k_this == 1)
      ++k1;
    else if (k_this >= 2)
      ++k2;
    flush_k();
    stats->drafted_total += k_this;
    stats->accepted_total += accepted;
    if (accepted == k_this)
      ++stats->full_accept_steps;
    stats->draft_ms += draft_ms;
    stats->verify_ms += verify_ms;
    stats->rollback_ms += rollback_ms;
    const int idx = k_this < 4 ? k_this : 3;
    stats->accept_at_len[idx] += accepted;
    stats->drafted_at_len[idx] += k_this;
  }
  return true;
}

}  // namespace superroc::models::ling3::mtp

#else  // !LING3_WITH_ROCM

namespace superroc::models::ling3::mtp {

bool Init(const ModelWeights&, HeadNorm, std::size_t*, std::string* error_msg) {
  if (error_msg != nullptr) {
    *error_msg = "this build has no HIP backend, so there is no MTP GPU head";
  }
  return false;
}
std::size_t ResidentBytesFor(const ModelWeights&) {
  return 0;
}
bool Ready() noexcept {
  return false;
}
std::string MissingPieces() {
  return "built without HIP (LING3_WITH_ROCM unset)";
}

struct State {};

State* NewState(std::uint32_t) {
  return nullptr;
}
void ResetState(State*) noexcept {}
void DestroyState(State*) noexcept {}
void SetWindowed(State*, bool) noexcept {}
void SetAttnWindow(State*, int) noexcept {}
bool PrimeNext(State*, std::int32_t, std::span<const float>, std::uint32_t,
               std::span<float>, std::string* error_msg) {
  if (error_msg != nullptr) {
    *error_msg = "mtp: built without HIP, so there is no MTP GPU head";
  }
  return false;
}
std::size_t StateBytes(std::uint32_t) {
  return 0;
}

bool DraftNext(State*, std::int32_t, std::span<const float>, std::uint32_t,
               std::int32_t*, std::span<float>, std::string* error_msg) {
  if (error_msg != nullptr) {
    *error_msg = "this build has no HIP backend, so there is no MTP GPU head";
  }
  return false;
}

bool ProbeEnabled() noexcept {
  return false;
}
void ProbeInit(const ModelWeights&, std::uint32_t, std::string* error_msg) {
  if (error_msg != nullptr) {
    *error_msg = "mtp probe: built without HIP";
  }
}
void ProbeStep(void*, std::int32_t, std::span<const float>, std::uint32_t,
               std::int32_t, bool) {}
void ProbeTally(int* compared, int* accepted, double* total_ms) {
  if (compared != nullptr)
    *compared = 0;
  if (accepted != nullptr)
    *accepted = 0;
  if (total_ms != nullptr)
    *total_ms = 0.0;
}
void ProbeShutdown() {}

bool SpeculativeEnabled() noexcept {
  return false;
}
bool DepthDraftOk(int, int) noexcept {
  return true;
}
class SpecState {};
State* SpecHead(SpecState*) noexcept {
  return nullptr;
}
SpecState* NewSpecState(std::uint32_t, const SpecPolicy&,
                        std::string* error_msg) {
  if (error_msg != nullptr) {
    *error_msg = "mtp: built without HIP, so there is no speculative decode";
  }
  return nullptr;
}
void DestroySpecState(SpecState*) noexcept {}
void ResetSpecState(SpecState*) noexcept {}
std::size_t SpecStateBytes(std::uint32_t) {
  return 0;
}
int SpecDraftLength(const SpecState*) noexcept {
  return 0;
}
bool SpecStep(void*, SpecState*, std::int32_t, const std::vector<float>&, bool,
              std::vector<std::int32_t>*, std::vector<float>*,
              std::vector<float>*, SpecStats*,
              const std::function<bool(std::int32_t)>&,
              std::string* error_msg) {
  if (error_msg != nullptr) {
    *error_msg = "mtp: built without HIP, so there is no speculative decode";
  }
  return false;
}

}  // namespace superroc::models::ling3::mtp

#endif  // LING3_WITH_ROCM
