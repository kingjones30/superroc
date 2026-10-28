// ling3_kda.hpp — KDA (Kimi Delta Attention) block of a Ling-3.0 `bailingmoe3`
// layer. Lane: LK (MiMo) — Ling-on-superroc overnight build. Do not confuse
// with LG's drafts. Standalone deliverable of lane LK for the Ling-on-superroc
// build.
//
// The fork's code is the spec:
//   src/models/bailingmoe3.cpp          (layer wiring, safe gate, L2 norm,
//   output gate) src/models/delta-net-base.cpp       (delta-net dispatch)
//   ggml/src/ggml-cuda/gated_delta_net.cu   (the arithmetic that actually runs
//   on ROCm) ggml/src/ggml-cuda/norm.cu          (rms_norm / l2_norm)
//   ggml/src/ggml-cuda/unary.cuh        (silu / sigmoid)
//
// Scope (exactly the four points of the LK spec): this module owns
//   (1) causal short conv (kernel 4) + SiLU on q, k, v, with conv-state carry
//   for decode (2) the fine-grained decay gate from ssm_f: ssm_a / A_log,
//   ssm_dt.bias, the "safe gate"
//       with its lower bound (GGUF bailingmoe3.kda.gate_lower_bound = -5), plus
//       beta = sigmoid
//   (3) the gated delta-rule recurrence (chunked prefill / recurrent decode),
//   state
//       32 heads x 128 x 128 in F32
//   (4) RMSNorm with ssm_norm, gated by sigmoid(ssm_g)
//
// The q/k/v, ssm_f, ssm_g and ssm_beta projections are GEMMs the CALLER does;
// their outputs are the inputs here. The output projection (attn_output.weight)
// is also the caller's.
//
// Numerics: F32 activations/state (the fork's state is F32), with reductions
// accumulated in F32 the same way ggml-cuda does. The FP64 CPU reference in
// ling3_kda_ref.hpp is the accuracy oracle; kda_test.hip reports the deviation
// of these kernels from it.

#ifndef LING3_KDA_HPP
#define LING3_KDA_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ling3 {

// ---------------------------------------------------------------------------
// Geometry of one Ling-3.0-flash KDA layer (GGUF bailingmoe3.*)
//   n_embd 2560 · n_head 32 · kda.head_dim 128 · d_inner 4096
//   ssm.conv_kernel 4 · kda.gate_lower_bound -5
//   attention.layer_norm_rms_epsilon 1e-6
// ---------------------------------------------------------------------------
struct KdaShape {
  int n_head = 32;
  int head_dim = 128;
  int d_conv = 4;

  int d_inner() const { return n_head * head_dim; }  // 4096
  int embd_s() const {
    return head_dim * head_dim * n_head;
  }  // 524288  one seq's SSM state
  int embd_r() const {
    return 3 * (d_conv - 1) * d_inner();
  }  // 36864   one seq's conv state
};

// ---------------------------------------------------------------------------
// Activation views. Layout is [rows, n_tokens, n_seqs] when contiguous
// (element (r, t, s) at r + t*s_tok + s*s_seq), which is what ggml produces for
// these tensors. Strides are in ELEMENTS, so pass make_view() for the usual
// case.
// ---------------------------------------------------------------------------
struct View {
  float* p = nullptr;
  int64_t s_tok = 0;
  int64_t s_seq = 0;
};

struct ViewC {
  const float* p = nullptr;
  int64_t s_tok = 0;  // elements between tokens of one sequence
  int64_t s_seq = 0;  // elements between sequences

  ViewC() = default;
  ViewC(const float* p_, int64_t tok, int64_t seq)
      : p(p_), s_tok(tok), s_seq(seq) {}
  ViewC(const View& v)
      : p(v.p), s_tok(v.s_tok), s_seq(v.s_seq) {}  // View -> ViewC
};

inline ViewC make_view(const float* p, int rows, int n_tokens) {
  return ViewC(p, rows, (int64_t)rows * n_tokens);
}
inline View make_view(float* p, int rows, int n_tokens) {
  return {p, rows, (int64_t)rows * n_tokens};
}

// ---------------------------------------------------------------------------
// Weights (device pointers, caller-owned, must outlive the launch)
//   conv_w{q,k,v} : [d_conv, d_inner] F32, w[tap + c*d_conv]
//                   (GGUF blk.N.ssm_conv1d_{q,k,v}.weight, shape [d_conv, 1,
//                   d_inner])
//   a_exp         : [n_head] F32 = exp(A_log)  (GGUF blk.N.ssm_a — the
//   conversion script
//                   already applied -exp(), the fork multiplies by this value
//                   as-is)
//   dt_bias       : [d_inner] F32 (GGUF blk.N.ssm_dt.bias)
//   o_norm        : [head_dim] F32 (GGUF blk.N.ssm_norm.weight)
// ---------------------------------------------------------------------------
struct KdaWeights {
  const float* conv_wq = nullptr;
  const float* conv_wk = nullptr;
  const float* conv_wv = nullptr;
  const float* a_exp = nullptr;
  const float* dt_bias = nullptr;
  const float* o_norm = nullptr;
};

struct KdaScalars {
  float gate_lower_bound = -5.0f;  // bailingmoe3.kda.gate_lower_bound
  float rms_eps = 1e-6f;  // bailingmoe3.attention.layer_norm_rms_epsilon
  float l2_eps = 1e-6f;   // same key in the fork (ggml_l2_norm / l2_norm_f32)
  float qk_scale = 0.0f;  // 0 => 1/sqrt(head_dim), the delta-net scale
};

// ---------------------------------------------------------------------------
// Caller-owned state buffers (device, F32). One row per sequence, in the fork's
// recurrent-cache row layout so a recurrent cache can be handed over directly.
//
// conv_state : [n_seqs] rows of shape.embd_r() = 3*(d_conv-1)*d_inner floats
//     row s = [Q | K | V]; inside each slice the tap index is fastest:
//       conv_state[s*embd_r() + qkv*(d_conv-1)*d_inner() + c*(d_conv-1) + tap]
//     = the last (d_conv-1) samples of that channel's projection stream, oldest
//     first.
//
// ssm_state  : [n_seqs] rows of shape.embd_s() = head_dim*head_dim*n_head
// floats
//       ssm_state[s*embd_s() + h*head_dim*head_dim + j*head_dim + i]
//     i = key channel (fastest), j = value channel. This is the fork's
//     transposed state M[col][i] = S[i][col] with S[key, value].
// ---------------------------------------------------------------------------
struct KdaState {
  float* conv_state = nullptr;
  float* ssm_state = nullptr;
  int n_seqs = 1;
};

// ---------------------------------------------------------------------------
// Inputs — the outputs of the caller's GEMMs, before this module's elementwise
// math.
//   q, k, v : [d_inner, n_tokens, n_seqs]  (attn_q/k/v projection output)
//   f       : [d_inner, n_tokens, n_seqs]  (ssm_f projection output; dt_bias is
//   added here) g2      : [d_inner, n_tokens, n_seqs]  (ssm_g projection
//   output; sigmoid applied here) beta    : [n_head,  n_tokens, n_seqs]
//   (ssm_beta projection output; sigmoid applied here)
// ---------------------------------------------------------------------------
struct KdaInputs {
  ViewC q, k, v, f, g2, beta;
  int n_tokens = 0;
  int n_seqs = 1;
};

// ---------------------------------------------------------------------------
// The four pieces. Each takes the same shape/scalars and a stream; each is
// independently callable so superroc can schedule/fuse around them.
// ---------------------------------------------------------------------------

// (1) Causal conv (kernel d_conv) + SiLU on one projected stream, with state
// carry.
//   in  : [d_inner, n_tokens, n_seqs] raw projection output (the conv state
//   stores THIS) out : [d_inner, n_tokens, n_seqs] silu(conv(stream)) — may
//   alias `in` conv_state : whole buffer (stride/offset derived from `qkv`),
//   updated in place qkv : 0 = Q, 1 = K, 2 = V
void kda_conv_silu_qkv(const KdaShape& sh, const ViewC& q, const ViewC& k,
                       const ViewC& v, const View& qo, const View& ko,
                       const View& vo, const float* wq, const float* wk,
                       const float* wv, float* conv_state, int n_tokens,
                       int n_seqs, hipStream_t stream);

void kda_conv_silu(const KdaShape& sh, const float* conv_w, const ViewC& in,
                   const View& out, float* conv_state, int qkv, int n_tokens,
                   int n_seqs, hipStream_t stream);

// LC1 (wave 3): which path the last conv call took -- 0 = the serial kernel, 1
// = the token-parallel prefill pair (taken for n_tokens > 1). Diagnostic: the
// parity test reads it so it cannot compare the serial path with itself (LG1's
// test did exactly that).
int kda_conv_silu_last_path();

// (2) Decay gate + beta.
//   g1_out   : [d_inner, n_tokens, n_seqs] = gate_lower_bound * sigmoid((f +
//   dt_bias) * a_exp) beta_out : [n_head,  n_tokens, n_seqs] = sigmoid(beta)
void kda_gate_beta(const KdaShape& sh, const KdaScalars& sc,
                   const KdaWeights& w, const ViewC& f, const ViewC& beta,
                   const View& g1_out, const View& beta_out, int n_tokens,
                   int n_seqs, hipStream_t stream);

// L2-normalise q (or k) along head_dim, in place if out aliases in.
//   scale = rsqrtf(fmaxf(sum(x*x), l2_eps*l2_eps))   (ggml-cuda l2_norm_f32)
void kda_l2_norm(const KdaShape& sh, float l2_eps, const ViewC& in,
                 const View& out, int n_tokens, int n_seqs, hipStream_t stream);

// (3) Gated delta-rule scan over n_tokens tokens.
//   q, k : [d_inner, n_tokens, n_seqs] convolved + SiLU + L2-normalised
//   v    : [d_inner, n_tokens, n_seqs] convolved + SiLU
//   g1   : [d_inner, n_tokens, n_seqs] decay gate from (2), in
//   (gate_lower_bound, 0) beta : [n_head,  n_tokens, n_seqs] from (2) out  :
//   [head_dim, n_head, n_tokens, n_seqs] == [d_inner, n_tokens, n_seqs]
//   ssm_state updated in place.
//   n_tokens == 1 is the decode path (fast scan unless LING3_KDA_DECODE_REF=1).
//   Default decode kernel is t32 (bit-identical coalesced 32-column tile).
//   LING3_KDA_DECODE_VARIANT=vec|coal|fma|red selects another kernel. red
//   changes the sum order. n_tokens > 1 is the prefill path and always runs the
//   reference scan.
void kda_scan(const KdaShape& sh, const KdaScalars& sc, const ViewC& q,
              const ViewC& k, const ViewC& v, const ViewC& g1,
              const ViewC& beta, const View& out, float* ssm_state,
              int n_tokens, int n_seqs, hipStream_t stream);

// How many decode launches took the fast scan in this process.
int kda_decode_fast_launches();

// Test-only. pick 0 follows the cached env, 1 forces the reference scan,
// 2 forces `variant`. LING3_KDA_DECODE_REF and LING3_KDA_DECODE_VARIANT are
// each read once per process. Serve does not call this.
void kda_decode_test_pick(int pick, const char* variant);

// (4) output gate: RMSNorm(o, o_norm) * sigmoid(g2)
//   o_in  : [d_inner, n_tokens, n_seqs] from kda_scan
//   g2    : [d_inner, n_tokens, n_seqs] raw ssm_g projection
//   out   : [d_inner, n_tokens, n_seqs] — may alias o_in
void kda_out_norm_gate(const KdaShape& sh, const KdaScalars& sc,
                       const KdaWeights& w, const ViewC& o_in, const ViewC& g2,
                       const View& out, int n_tokens, int n_seqs,
                       hipStream_t stream);

// Zero-initialise the two state buffers (fresh sequence).
void kda_state_zero(const KdaShape& sh, float* conv_state, float* ssm_state,
                    int n_seqs, hipStream_t stream);

// ---------------------------------------------------------------------------
// Umbrella: the whole KDA block in one call. `out` is the gated output, i.e.
// what the caller feeds to attn_output.weight. Conv/SSM state is read and
// written.
// ---------------------------------------------------------------------------
void kda_forward(const KdaShape& sh, const KdaScalars& sc, const KdaWeights& w,
                 const KdaInputs& in, const KdaState& st, const View& out,
                 hipStream_t stream, int8_t* wo_qs = nullptr,
                 float* wo_ds = nullptr);

// Decode convenience: identical to kda_forward with n_tokens == 1.
void kda_forward_decode(const KdaShape& sh, const KdaScalars& sc,
                        const KdaWeights& w, const KdaInputs& in,
                        const KdaState& st, const View& out,
                        hipStream_t stream);

// Scratch bytes kda_forward needs for its intermediate buffers (g1, beta, and
// the three conv outputs). `kda_forward` allocates this from the current device
// if you pass nullptr; superroc can pass its own arena instead. LKF (wave 3). 0
// = LING3_KDA_DECODE_FUSED (default on, read once). 1 = force off. 2 = force
// on. Serve does not call this.
void kda_decode_fused_test_pick(int pick);
int kda_decode_fused_k1_launches();
int kda_decode_fused_k2_launches();
// True when Step should skip its own wo quant: the fused t32 epilogue writes
// the slot.
bool kda_wo_q8_fused();

// Decode scan + out_norm (+ Q8 when wo_qs != nullptr and the t32 fused path is
// on). q/k are post-L2, g1/beta post-gate, which is what LING3_KDA_CAPTURE
// records. Returns 1 if the fused K2 kernel was launched.
int kda_decode_scan_out(const KdaShape& sh, const KdaScalars& sc,
                        const float* o_norm, const ViewC& q, const ViewC& k,
                        const ViewC& v, const ViewC& g1, const ViewC& beta,
                        const ViewC& g2, const View& out, float* ssm_state,
                        int8_t* wo_qs, float* wo_ds, hipStream_t stream);

size_t kda_forward_scratch_bytes(const KdaShape& sh, int n_tokens, int n_seqs);

void kda_forward_scratch(const KdaShape& sh, const KdaScalars& sc,
                         const KdaWeights& w, const KdaInputs& in,
                         const KdaState& st, const View& out, void* scratch,
                         size_t scratch_bytes, hipStream_t stream,
                         int8_t* wo_qs = nullptr, float* wo_ds = nullptr);

const char* kda_version();  // "ling3-kda/1.0"

}  // namespace ling3

#endif  // LING3_KDA_HPP
