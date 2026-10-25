// Compiled in EVERY configuration, but the body only exists when the HIP
// backend is on: a CPU build must not need hip/hip_runtime.h. The stub below
// refuses, which is what `gufo serve` reports on a CPU-only build.
// Unconditional: the header itself pulls only gufo's GGUF reader and the weight
// structs, no HIP.
#include "src/models/ling3/rocm_backend.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(LING3_WITH_ROCM)
// LK's and LX's kernels live in the GLOBAL namespace `ling3`, which
// collides with this package's own `gufo::models::ling3`; every
// reference below is therefore `::ling3::`.
#include "src/models/ling3/kernels/ling3_gemv_q8.hpp"
#include "src/models/ling3/kernels/ling3_kda.hpp"
#include "src/models/ling3/kernels/ling3_mla.hpp"
#include "src/models/ling3/kernels/ling3_mla_tile.hpp"
#include "src/models/ling3/kernels/ling3_moe.hpp"
#include "src/models/ling3/kernels/ling3_ops.hpp"
#include "src/models/ling3/kernels/ling3_quant.hpp"
// LS3: the tiled grouped GEMM + batched router for batched prefill.
#include "src/models/ling3/kernels/ling3_gemm.hpp"
#include "src/models/ling3/kernels/ling3_moe_group.hpp"
// LS3 P3: the W8A8 int8-WMMA grouped GEMM (gufo skeleton, fork numerics).
#include "src/models/ling3/kernels/ling3_wmma.hpp"
// LW1 (wave 2): the two-pass WMMA prefill attention, LING3_MLA_WMMA2=1.
#include "src/models/ling3/kernels/ling3_mla_w2.hpp"
// LX1 (wave 3): the fused WMMA flash prefill attention, LING3_MLA_FW=1.
#include "src/models/ling3/kernels/ling3_mla_fw.hpp"
#include "src/models/ling3/vision_prompt.hpp"
#endif  // LING3_WITH_ROCM

#if defined(LING3_WITH_ROCM)

#if defined(LING3_MAC_HIPSTUB)
// LK's Mac type-check stub (../lk/mactest/hipstub) does not declare
// hipMemGetInfo, so the device query below is compiled out for that check only.
// A real build always runs it.
#define LING3_SKIP_DEVICE_MEMORY_CHECK 1
#endif

namespace gufo::models::ling3::rocm {
namespace {

/// LX's mandatory readable pad past the last byte of any allocation: the
/// 17-byte ROCmFP4_FAST block leaves the last row unaligned and the kernels
/// widen a tail load.
constexpr std::size_t kPad = LING3_TAIL_PAD;
constexpr std::size_t kMaxExperts = 8;

/// Zeroes a device buffer with hipMemcpy rather than hipMemset: hipMemset is
/// absent from LK's Mac type-check stub, and a device memset of several GiB is
/// an extra kernel launch we do not need -- only the recurrent state and the
/// activation scratch must start at zero, and those are few MiB.
void DevZero(void* p, std::size_t bytes) {
  if (p == nullptr || bytes == 0) {
    return;
  }
  static const std::vector<float> kZeros(1 << 20, 0.0F);
  std::size_t left = bytes;
  const std::uint8_t* src =
      reinterpret_cast<const std::uint8_t*>(kZeros.data());
  while (left > 0) {
    const std::size_t chunk = left < kZeros.size() * sizeof(float)
                                  ? left
                                  : kZeros.size() * sizeof(float);
    (void)hipMemcpy(p, src, chunk, hipMemcpyHostToDevice);
    p = static_cast<std::uint8_t*>(p) + chunk;
    // Jay 2026-10-01: no "src += chunk" -- the source is the same 4 MiB zero
    // block every time. Advancing it walked off the end of kZeros as soon as a
    // buffer was larger than 4 MiB (H1's F7 made DevZero clear real sizes):
    // SIGSEGV in hipMemcpy at load.
    left -= chunk;
  }
}

void* DevAlloc(std::size_t bytes) {
  void* p = nullptr;
  if (hipMalloc(&p, bytes + kPad) != hipSuccess) {
    return nullptr;
  }
  return p;
}

// LQ3: LING3_POISON_ALLOC=zero|nan fills every freshly-allocated SCRATCH buffer
// on the batched-prefill path. hipMalloc does not zero: a buffer a kernel reads
// before anyone wrote it holds whatever the previous process left there, which
// is exactly the run-to-run nondeterminism this lane is chasing. Poisoning
// makes that read reproducible (zero) or loudly visible (NaN -> the first
// tensor that turns NaN is the one whose kernel read memory nobody wrote). It
// is NOT applied to weight loads: DevCopy overwrites those immediately and
// filling 65 GB of them would stall the load.
int PoisonMode() {
  static const int mode = [] {
    const char* e = std::getenv("LING3_POISON_ALLOC");
    if (e == nullptr || e[0] == '\0')
      return 0;
    if (e[0] == 'n' || e[0] == 'N')
      return 2;  // nan
    return 1;    // zero (or anything else)
  }();
  return mode;
}

void DevPoison(void* p, std::size_t bytes) {
  const int mode = PoisonMode();
  if (p == nullptr || bytes == 0 || mode == 0) {
    return;
  }
  static const std::vector<float> kZero(1 << 20, 0.0F);
  static const std::vector<float> kNan(1 << 20,
                                       std::numeric_limits<float>::quiet_NaN());
  const std::vector<float>& fill = mode == 2 ? kNan : kZero;
  std::size_t left = bytes;
  auto* dst = static_cast<std::uint8_t*>(p);
  const auto* src = reinterpret_cast<const std::uint8_t*>(fill.data());
  while (left > 0) {
    const std::size_t chunk =
        left < fill.size() * sizeof(float) ? left : fill.size() * sizeof(float);
    (void)hipMemcpy(dst, src, chunk, hipMemcpyHostToDevice);
    dst += chunk;
    left -= chunk;
  }
  // LQ3 positive control, once: read the fill back so the log PROVES the poison
  // reached the device. Without this a clean run means either "no uninitialised
  // read" or "the knob never fired", and the two are indistinguishable.
  static bool checked = false;
  if (!checked && bytes >= 4) {
    checked = true;
    unsigned char probe[4] = {0, 0, 0, 0};
    (void)hipMemcpy(probe, p, sizeof(probe), hipMemcpyDeviceToHost);
    std::fprintf(stderr,
                 "LQ3-POISON mode=%d first4=%02x%02x%02x%02x "
                 "(nan -> 0000c07f, zero -> 00000000)\n",
                 mode, probe[0], probe[1], probe[2], probe[3]);
  }
}

/// `d` is taken as const because most call sites hold a `const void*` device
/// pointer (the weights are only ever read); hipMemcpy just needs a non-const
/// destination type.
bool DevCopy(const void* d, const void* s, std::size_t n) {
  // Jay 2026-10-01 (hotfix, for LG to adopt): a plain hipMemcpy straight out of
  // the GGUF mmap stalled the 65 GB Ling load at ~45 GB of GTT on amdaimax2 --
  // main thread spinning inside libhsa, GPU 0 % busy, no new page faults, `gufo
  // serve` never answered /health in 10 min. HIP pins pageable sources for the
  // DMA; pinning tens of GB of file-backed pages on a 128 GB unified-memory box
  // while the device copy grows is what wedged it. Copy through ONE pinned 64
  // MiB bounce buffer instead, so HIP never pins the mmap: the 65 GB load then
  // completes in ~30 s.
  if (n == 0) {
    return true;
  }
  constexpr std::size_t kChunk = std::size_t{64} << 20;
  static void* bounce = nullptr;
  if (bounce == nullptr &&
      hipHostMalloc(&bounce, kChunk, hipHostMallocDefault) != hipSuccess) {
    bounce = nullptr;
    return hipMemcpy(const_cast<void*>(d), s, n, hipMemcpyHostToDevice) ==
           hipSuccess;
  }
  auto* dst = static_cast<std::uint8_t*>(const_cast<void*>(d));
  const auto* src = static_cast<const std::uint8_t*>(s);
  for (std::size_t off = 0; off < n; off += kChunk) {
    const std::size_t c = n - off < kChunk ? n - off : kChunk;
    std::memcpy(bounce, src + off, c);
    if (hipMemcpy(dst + off, bounce, c, hipMemcpyHostToDevice) != hipSuccess) {
      return false;
    }
    // (no madvise: DevCopy is also fed host-heap buffers -- converted norms,
    // F32 copies -- and MADV_DONTNEED on heap pages zeroes them and the malloc
    // headers sharing those pages; that is the likeliest cause of the SIGSEGV
    // right after load in serve-test3.)
  }
  return true;
}

/// Convenience for the `const float**` the norm upload helpers take.
bool DevCopyF(const float** slot, const void* src, std::size_t n) {
  float* p = static_cast<float*>(DevAlloc(n));
  if (p == nullptr || !DevCopy(p, src, n)) {
    return false;
  }
  *slot = p;
  return true;
}

/// Everything the executor needs, resolved once at load.
struct Plan {
  int hidden{0}, vocab{0}, heads{0}, inner{0}, hd{0}, kc{0};
  int expert_ff{0}, n_expert{0}, n_used{0}, n_groups{0}, n_group_used{0};
  int ffn_len{0}, dense_lead{0};
  int mla_kv{0}, mla_head_k{0}, mla_head_v{0}, mla_rot{0}, mla_o_dim{0};
  float rope_theta{0.0F}, w_scale{1.0F}, eps{1e-6F};
  int norm_w{1};
  std::vector<float> clamp_exp, clamp_shexp;
  std::vector<bool> is_mla;
  // LG1 W3: MLA ordinal map, -1 for KDA layers, 0..6 for the 7 MLA layers
  // (5, 11, 17, 23, 29, 35, 41 in build28f). Built at Init beside is_mla.
  std::vector<int> mla_ord;
  std::size_t resident{0};
  bool ready{false};
};

Plan g;

struct Layer {
  // norms
  const float* attn_norm{nullptr};
  const float* ffn_norm{nullptr};
  // projections: {device ptr, ggml type, rows, cols}
  struct W {
    const void* d{nullptr};
    int type{0}, rows{0}, cols{0};
    const void* host{nullptr};
    std::size_t host_off{0}, host_bytes{0};
    int experts{1};
  };
  W wq, wk, wv, wf, wg, wbeta, wo;
  const float* conv_q{nullptr};
  const float* conv_k{nullptr};
  const float* conv_v{nullptr};
  const float* a_exp{nullptr};
  const float* dt_bias{nullptr};
  const float* o_norm{nullptr};
  // MLA
  W mq, m_kv_a_mqa, m_gate, m_wo;
  W m_wk_b_packed, m_wv_b_packed;
  const float* m_wk_b_f32{nullptr};
  const float* m_wv_b_f32{nullptr};
  // SPEC6: row-transposed copies for the decode prep / de-absorb kernels.
  const float* m_wk_bT{nullptr};
  const float* m_wv_bT{nullptr};
  const float* m_kv_a_norm{nullptr};
  // FFN
  bool dense{false};
  W ffn_gate, ffn_up, ffn_down;  // dense blocks, or the shared expert
  W router;
  const float* probs_bias{nullptr};
  const void* host_gate_exps{nullptr};
  const void* host_up_exps{nullptr};
  const void* host_down_exps{nullptr};
  std::size_t rb_k{0}, rb_ff{0};  // expert row bytes at k and at n_ff
  int ex_type{0};
  float clamp_exp{0.0F}, clamp_shexp{0.0F};
};

struct Device {
  const void* token_embd{nullptr};
  int token_embd_type{0};  // ggml type of token_embd (Q6_K text, Q5_K VL)
  const void* output{nullptr};
  int output_type{0}, output_rows{0}, output_cols{0};
  const float* output_norm{nullptr};
  std::vector<Layer> layers;
  // per-expert staging for the streamed routed experts
  void* s_gate{nullptr};
  void* s_up{nullptr};
  void* s_down{nullptr};
  std::size_t sb_gate{0}, sb_up{0}, sb_down{0};
  // Resident routed experts: one device copy of the WHOLE [*, *, 512] stack
  // per MoE layer. ~1.49 GiB each, ~59.8 GiB over the 40 MoE layers.
  struct ResidentExperts {
    const void* gate{nullptr};
    const void* up{nullptr};
    const void* down{nullptr};
    int type{0};
    std::size_t rows{0};       // expert_ff, gate/up
    std::size_t cols{0};       // hidden
    std::size_t down_rows{0};  // hidden
    std::size_t down_cols{0};  // expert_ff
  };
  bool stream_experts{true};
  std::vector<ResidentExperts> experts;
  // device-side router outputs
  int* d_ids{nullptr};
  float* d_weights{nullptr};
};

Device g_dev;

}  // namespace

// ---------------------------------------------------------------------------

std::size_t ResidentBytesFor(const ModelWeights& weights,
                             const ExpertPlacement& placement) {
  std::size_t total = 0;
  const auto add = [&total](const TensorRef& t) { total += t.SizeBytes(); };
  add(weights.token_embd);
  add(weights.output);
  add(weights.output_norm.tensor);
  for (const LayerWeights& l : weights.layers) {
    add(l.attn_norm.tensor);
    add(l.ffn_norm.tensor);
    if (l.mla) {
      add(l.mla_weights.wq);
      add(l.mla_weights.wkv_a_mqa);
      add(l.mla_weights.kv_a_norm.tensor);
      add(l.mla_weights.wk_b);
      add(l.mla_weights.wv_b);
      add(l.mla_weights.wqkv_gate);
      add(l.mla_weights.wo);
    } else {
      add(l.kda.wq);
      add(l.kda.wk);
      add(l.kda.wv);
      add(l.kda.conv_q);
      add(l.kda.conv_k);
      add(l.kda.conv_v);
      add(l.kda.w_f);
      add(l.kda.w_g);
      add(l.kda.w_beta);
      add(l.kda.a_log);
      add(l.kda.dt_bias.tensor);
      add(l.kda.o_norm.tensor);
      add(l.kda.wo);
    }
    if (!l.ffn_gate.empty()) {
      add(l.ffn_gate);
      add(l.ffn_up);
      add(l.ffn_down);
    } else {
      add(l.moe.gate_inp);
      add(l.moe.probs_bias.tensor);
      add(l.moe.gate_shexp);
      add(l.moe.up_shexp);
      add(l.moe.down_shexp);
      if (!placement.stream) {
        // The whole routed stack, resident.
        add(l.moe.gate_exps);
        add(l.moe.up_exps);
        add(l.moe.down_exps);
      }
    }
  }
  return total;
}

bool Init(const ModelWeights& weights, const core::GgufReader& reader,
          const ExpertPlacement& placement, std::size_t* resident_bytes,
          std::string* error_msg) {
  (void)reader;
  const Config& c = weights.config;
  g.hidden = static_cast<int>(c.hidden_size);
  g.vocab = static_cast<int>(c.vocab_size);
  g.heads = static_cast<int>(c.num_heads);
  g.hd = static_cast<int>(c.kda_head_dim);
  g.inner = g.heads * g.hd;
  g.kc = static_cast<int>(c.kda_conv_kernel);
  g.eps = c.rms_eps;
  g.rope_theta = c.rope_theta;
  g.expert_ff = static_cast<int>(c.expert_ff);
  g.n_expert = static_cast<int>(c.num_experts);
  g.n_used = static_cast<int>(c.num_experts_used);
  g.n_groups = static_cast<int>(c.num_expert_groups);
  g.n_group_used = static_cast<int>(c.num_group_used);
  g.ffn_len = static_cast<int>(c.ffn_length);
  g.dense_lead = static_cast<int>(c.dense_lead_layers);
  g.w_scale = c.expert_weights_scale;
  g.norm_w = c.expert_weights_norm ? 1 : 0;
  g.mla_kv = static_cast<int>(c.mla_kv_lora_rank);
  g.mla_head_k = static_cast<int>(c.mla_head_dim_k);
  g.mla_head_v = static_cast<int>(c.mla_head_dim_v);
  g.mla_rot = static_cast<int>(c.rotary_dim);
  g.mla_o_dim = g.heads * g.mla_head_v;
  g.clamp_exp = c.swiglu_clamp_exp;
  g.clamp_shexp = c.swiglu_clamp_shexp;

  const auto fail = [&](std::string m) {
    if (error_msg != nullptr) {
      *error_msg = "ling3 ROCm load: " + m;
    }
    return false;
  };

  // Pre-flight the memory BEFORE committing to it: a 60 GiB allocation that
  // fails halfway leaves the driver wedged, and the box may still have a seat
  // loaded. Ask the device, and refuse with the shortfall spelled out.
  const std::size_t need = ResidentBytesFor(weights, placement);
  g_dev.stream_experts = placement.stream;
#if !defined(LING3_SKIP_DEVICE_MEMORY_CHECK)
  {
    std::size_t free_bytes = 0, total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
      return fail("hipMemGetInfo failed; cannot check device memory");
    }
    if (need + (512u << 20) > free_bytes) {
      char msg[320];
      std::snprintf(
          msg, sizeof(msg),
          "needs %.1f GiB but only %.1f GiB is free on the device. "
          "Load with ModelOptions{.stream_experts = true} to stage "
          "the 8 selected experts per layer per token instead of "
          "holding all 512 resident.",
          static_cast<double>(need) / (1024.0 * 1024.0 * 1024.0),
          static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0));
      return fail(msg);
    }
  }
#endif  // !LING3_SKIP_DEVICE_MEMORY_CHECK

  // Embeddings and the LM head. `output.weight` is ROCmFP4_FAST and is read
  // straight out of the packed upload by ling3_gemv.
  {
    g.resident = 0;
    const TensorRef& e = weights.token_embd;
    g_dev.token_embd = DevAlloc(e.SizeBytes());
    if (!DevCopy(g_dev.token_embd, e.data, e.SizeBytes())) {
      return fail("token_embd upload");
    }
    g_dev.token_embd_type = static_cast<int>(e.type);
    g.resident += e.SizeBytes() + kPad;
    const TensorRef& o = weights.output;
    g_dev.output = DevAlloc(o.SizeBytes());
    if (!DevCopy(g_dev.output, o.data, o.SizeBytes())) {
      return fail("output upload");
    }
    g.resident += o.SizeBytes() + kPad;
    g_dev.output_type = static_cast<int>(o.type);
    g_dev.output_rows = static_cast<int>(o.rows);
    g_dev.output_cols = static_cast<int>(o.cols);
    const VectorRef& on = weights.output_norm;
    if (DevCopyF(&g_dev.output_norm, on.tensor.data, on.tensor.SizeBytes())) {
      g.resident += on.tensor.SizeBytes() + kPad;
    }
  }

  g_dev.layers.assign(weights.layers.size(), Layer{});
  g.is_mla.assign(weights.layers.size(), false);
  g.mla_ord.assign(weights.layers.size(), -1);
  {
    int mla_count = 0;
    for (std::size_t i = 0; i < weights.layers.size(); ++i) {
      if (weights.layers[i].mla) {
        g.mla_ord[i] = mla_count++;
      }
    }
  }
  for (std::size_t i = 0; i < weights.layers.size(); ++i) {
    const LayerWeights& l = weights.layers[i];
    Layer& d = g_dev.layers[i];
    g.is_mla[i] = l.mla;
    d.clamp_exp = i < g.clamp_exp.size() ? g.clamp_exp[i] : 0.0F;
    d.clamp_shexp = i < g.clamp_shexp.size() ? g.clamp_shexp[i] : 0.0F;

    const auto up = [&](const TensorRef& t, Layer::W* w) {
      w->type = static_cast<int>(t.type);
      w->rows = static_cast<int>(t.rows);
      w->cols = static_cast<int>(t.cols);
      w->experts = static_cast<int>(t.experts);
      w->host = t.data;
      w->host_bytes = t.SizeBytes();
      w->d = DevAlloc(w->host_bytes);
      if (w->d == nullptr || !DevCopy(w->d, t.data, w->host_bytes)) {
        return false;
      }
      g.resident += w->host_bytes + kPad;
      return true;
    };
    const auto upv = [&](const VectorRef& v, const float** slot) {
      float* p = static_cast<float*>(DevAlloc(v.tensor.SizeBytes()));
      if (p == nullptr || !DevCopy(p, v.tensor.data, v.tensor.SizeBytes())) {
        return false;
      }
      *slot = p;
      g.resident += v.tensor.SizeBytes() + kPad;
      return true;
    };
    // The conv weights are already F32 in the artifact, so no mirror is needed.
    const auto upf = [&](const TensorRef& t, const float** slot) {
      float* p = static_cast<float*>(DevAlloc(t.SizeBytes()));
      if (p == nullptr || !DevCopy(p, t.data, t.SizeBytes())) {
        return false;
      }
      *slot = p;
      g.resident += t.SizeBytes() + kPad;
      return true;
    };

    if (!upv(l.attn_norm, &d.attn_norm) || !upv(l.ffn_norm, &d.ffn_norm)) {
      return fail("norm upload at layer " + std::to_string(i));
    }
    if (l.mla) {
      if (!up(l.mla_weights.wq, &d.mq) ||
          !up(l.mla_weights.wkv_a_mqa, &d.m_kv_a_mqa) ||
          !up(l.mla_weights.wqkv_gate, &d.m_gate) ||
          !up(l.mla_weights.wo, &d.m_wo) ||
          !up(l.mla_weights.wk_b, &d.m_wk_b_packed) ||
          !up(l.mla_weights.wv_b, &d.m_wv_b_packed) ||
          !upv(l.mla_weights.kv_a_norm, &d.m_kv_a_norm)) {
        return fail("MLA upload at layer " + std::to_string(i));
      }
      // LK's MlaWeights takes wk_b / wv_b as float*, and on disk they are
      // ROCmFP4. Dequantize once at load: 8.4 MB each per layer.
      const auto mirror = [&](const Layer::W& p, const float** slot) {
        const int rows_all = p.rows * p.experts;
        float* f = static_cast<float*>(DevAlloc(
            static_cast<std::size_t>(rows_all) * p.cols * sizeof(float)));
        if (f == nullptr) {
          return false;
        }
        const std::size_t bytes =
            static_cast<std::size_t>(rows_all) * p.cols * sizeof(float);
        g.resident += bytes + kPad;
        // Jay 2026-10-01: the mirror was never stored -- `slot` stayed nullptr,
        // so MlaWeights.wk_b / wv_b reached mla_block as null and the first MLA
        // layer faulted the GPU (ling3_mla.hip:378, PERMISSION_FAULTS 0x3, two
        // GPU resets).
        *slot = f;
        return ling3_dequant(p.type, p.d, f, p.cols, rows_all) == LING3_OK;
      };
      // SPEC6: remember the shapes before the packed copies go away.
      const int wk_rows = d.m_wk_b_packed.rows * d.m_wk_b_packed.experts;
      const int wk_cols = d.m_wk_b_packed.cols;
      const int wv_rows = d.m_wv_b_packed.rows * d.m_wv_b_packed.experts;
      const int wv_cols = d.m_wv_b_packed.cols;
      if (!mirror(d.m_wk_b_packed, &d.m_wk_b_f32) ||
          !mirror(d.m_wv_b_packed, &d.m_wv_b_f32)) {
        return fail("MLA F32 mirror at layer " + std::to_string(i));
      }
      // SPEC6: one row-transposed copy each, made once at load, for the
      // decode path's prep and de-absorb kernels (coalesced reads). The
      // originals stay as they are: the prefill kernels read wk_b / wv_b.
      {
        const ::ling3::MlaShape sx;  // the model's MLA geometry (its defaults)
        if (wk_rows != sx.n_head * sx.kv || wk_cols != sx.nope ||
            wv_rows != sx.n_head * sx.head_v || wv_cols != sx.kv) {
          return fail("MLA weight shape does not match MlaShape at layer " +
                      std::to_string(i));
        }
        float* t = static_cast<float*>(DevAlloc(
            static_cast<std::size_t>(wk_rows) * wk_cols * sizeof(float)));
        if (t == nullptr)
          return fail("MLA wk_bT alloc at layer " + std::to_string(i));
        ::ling3::mla_weights_transpose(d.m_wk_b_f32, t, sx.n_head, sx.kv,
                                       sx.nope, nullptr);
        d.m_wk_bT = t;
        g.resident +=
            static_cast<std::size_t>(wk_rows) * wk_cols * sizeof(float) + kPad;
        t = static_cast<float*>(DevAlloc(static_cast<std::size_t>(wv_rows) *
                                         wv_cols * sizeof(float)));
        if (t == nullptr)
          return fail("MLA wv_bT alloc at layer " + std::to_string(i));
        ::ling3::mla_weights_transpose(d.m_wv_b_f32, t, sx.n_head, sx.head_v,
                                       sx.kv, nullptr);
        d.m_wv_bT = t;
        g.resident +=
            static_cast<std::size_t>(wv_rows) * wv_cols * sizeof(float) + kPad;
      }
      hipFree(const_cast<void*>(d.m_wk_b_packed.d));
      hipFree(const_cast<void*>(d.m_wv_b_packed.d));
      d.m_wk_b_packed.d = nullptr;
      d.m_wv_b_packed.d = nullptr;
    } else {
      if (!up(l.kda.wq, &d.wq) || !up(l.kda.wk, &d.wk) ||
          !up(l.kda.wv, &d.wv) || !up(l.kda.w_f, &d.wf) ||
          !up(l.kda.w_g, &d.wg) || !up(l.kda.w_beta, &d.wbeta) ||
          !up(l.kda.wo, &d.wo) || !upf(l.kda.conv_q, &d.conv_q) ||
          !upf(l.kda.conv_k, &d.conv_k) || !upf(l.kda.conv_v, &d.conv_v) ||
          !upf(l.kda.a_log, &d.a_exp) || !upv(l.kda.dt_bias, &d.dt_bias) ||
          !upv(l.kda.o_norm, &d.o_norm)) {
        return fail("KDA upload at layer " + std::to_string(i));
      }
    }

    if (!l.ffn_gate.empty()) {
      d.dense = true;
      if (!up(l.ffn_gate, &d.ffn_gate) || !up(l.ffn_up, &d.ffn_up) ||
          !up(l.ffn_down, &d.ffn_down)) {
        return fail("dense FFN upload at layer " + std::to_string(i));
      }
    } else {
      if (!up(l.moe.gate_inp, &d.router) ||
          !upv(l.moe.probs_bias, &d.probs_bias) ||
          !up(l.moe.gate_shexp, &d.ffn_gate) ||
          !up(l.moe.up_shexp, &d.ffn_up) ||
          !up(l.moe.down_shexp, &d.ffn_down)) {
        return fail("MoE upload at layer " + std::to_string(i));
      }
      d.ex_type = d.ffn_gate.type;
      d.rb_k = ling3_row_bytes(d.ex_type, g.hidden);
      d.rb_ff = ling3_row_bytes(d.ex_type, g.expert_ff);
      d.host_gate_exps = l.moe.gate_exps.data;
      d.host_up_exps = l.moe.up_exps.data;
      d.host_down_exps = l.moe.down_exps.data;
      if (!placement.stream) {
        Device::ResidentExperts re;
        re.type = d.ex_type;
        re.rows = static_cast<std::size_t>(l.moe.gate_exps.rows);
        re.cols = static_cast<std::size_t>(l.moe.gate_exps.cols);
        re.down_rows = static_cast<std::size_t>(l.moe.down_exps.rows);
        re.down_cols = static_cast<std::size_t>(l.moe.down_exps.cols);
        const TensorRef* stack[3] = {&l.moe.gate_exps, &l.moe.up_exps,
                                     &l.moe.down_exps};
        void* dst[3] = {nullptr, nullptr, nullptr};
        for (int e = 0; e < 3; ++e) {
          const std::size_t bytes = stack[e]->SizeBytes();
          dst[e] = DevAlloc(bytes);
          if (dst[e] == nullptr || !DevCopy(dst[e], stack[e]->data, bytes)) {
            return fail("routed expert upload at layer " + std::to_string(i));
          }
          g.resident += bytes + kPad;
        }
        re.gate = dst[0];
        re.up = dst[1];
        re.down = dst[2];
        g_dev.experts.push_back(re);
      }
    }
  }

  // Per-expert staging: the routed experts cannot be resident (512 of them is
  // ~64 GiB), so the 8 selected per token are copied in one at a time.
  {
    std::size_t rb_k = 0, rb_ff = 0;
    for (const Layer& l : g_dev.layers) {
      if (!l.dense && l.ex_type != 0) {
        rb_k = l.rb_k;
        rb_ff = l.rb_ff;
        break;
      }
    }
    if (rb_k == 0) {
      return fail("no MoE layer found to size the expert staging buffers");
    }
    g_dev.sb_gate = static_cast<std::size_t>(g.expert_ff) * rb_k;
    g_dev.sb_up = g_dev.sb_gate;
    g_dev.sb_down = static_cast<std::size_t>(g.hidden) * rb_ff;
    g_dev.s_gate = DevAlloc(g_dev.sb_gate);
    g_dev.s_up = DevAlloc(g_dev.sb_up);
    g_dev.s_down = DevAlloc(g_dev.sb_down);
    if (!g_dev.s_gate || !g_dev.s_up || !g_dev.s_down) {
      return fail("expert staging allocation");
    }
  }
  g_dev.d_ids = static_cast<int*>(DevAlloc(kMaxExperts * sizeof(int)));
  g_dev.d_weights = static_cast<float*>(DevAlloc(kMaxExperts * sizeof(float)));
  if (!g_dev.d_ids || !g_dev.d_weights) {
    return fail("router output allocation");
  }

  if (resident_bytes != nullptr) {
    *resident_bytes = g.resident;
  }
  g.ready = true;
  return true;
}

bool ExecutorReady() noexcept {
  return g.ready;
}
std::string MissingPieces() {
  return g.ready ? std::string() : std::string("executor not initialised");
}

// ---------------------------------------------------------------------------
// Per-session state.
// ---------------------------------------------------------------------------
namespace {

::ling3::KdaShape MakeKdaShape() {
  ::ling3::KdaShape s;
  s.n_head = g.heads;
  s.head_dim = g.hd;
  s.d_conv = g.kc;
  return s;
}

// Jay 2026-10-01: LING3_GPU_TRACE=<pos> prints the same per-block "LG-TRACE
// <block> <name> first3 last3" lines as the CPU oracle
// (tools/ling3_ref_main.cpp --trace) for the token at that position, so the
// first diverging tensor names the bug.
long GpuTracePos() {
  static const long pos = [] {
    const char* e = std::getenv("LING3_GPU_TRACE");
    return e != nullptr ? std::strtol(e, nullptr, 10) : -1L;
  }();
  return pos;
}
void GpuTrace(const State* st, std::uint32_t block, const char* name,
              const void* d, std::size_t n) {
  if (GpuTracePos() < 0 || st == nullptr ||
      static_cast<long>(st->pos) != GpuTracePos() || d == nullptr || n == 0) {
    return;
  }
  (void)hipDeviceSynchronize();
  std::vector<float> h(n);
  if (hipMemcpy(h.data(), d, n * sizeof(float), hipMemcpyDeviceToHost) !=
      hipSuccess) {
    std::printf("LG-TRACE %u %s COPY-FAILED\n", block, name);
    return;
  }
  std::printf("LG-TRACE %u %s", block, name);
  const std::size_t lead = n < 3 ? n : 3;
  for (std::size_t r = 0; r < lead; ++r)
    std::printf(" %.6f", static_cast<double>(h[r]));
  if (n > 6)
    for (std::size_t r = n - 3; r < n; ++r)
      std::printf(" %.6f", static_cast<double>(h[r]));
  std::printf("\n");
  std::fflush(stdout);
}

// Router ids are int32 on the device; the fork's debug printer renders its
// ffn_moe_topk (also i32) as floats, so print them the same way to stay
// comparable. Only reached when LING3_GPU_TRACE matches this position.
void GpuTraceIds(const State* st, std::uint32_t block, const char* name,
                 const int* d, std::size_t n) {
  if (GpuTracePos() < 0 || st == nullptr ||
      static_cast<long>(st->pos) != GpuTracePos() || d == nullptr || n == 0) {
    return;
  }
  (void)hipDeviceSynchronize();
  std::vector<int> h(n);
  if (hipMemcpy(h.data(), d, n * sizeof(int), hipMemcpyDeviceToHost) !=
      hipSuccess) {
    std::printf("LG-TRACE %u %s COPY-FAILED\n", block, name);
    return;
  }
  std::printf("LG-TRACE %u %s", block, name);
  const std::size_t lead = n < 3 ? n : 3;
  for (std::size_t r = 0; r < lead; ++r) {
    std::printf(" %.6f", static_cast<double>(static_cast<float>(h[r])));
  }
  if (n > 6) {
    for (std::size_t r = n - 3; r < n; ++r) {
      std::printf(" %.6f", static_cast<double>(static_cast<float>(h[r])));
    }
  }
  std::printf("\n");
  std::fflush(stdout);
}

::ling3::MlaShape MakeMlaShape() {
  ::ling3::MlaShape s;
  s.n_embd = g.hidden;
  s.n_head = g.heads;
  s.head_k = g.mla_head_k;
  s.head_v = g.mla_head_v;
  s.nope = g.mla_head_k - g.mla_rot;
  s.rot = g.mla_rot;
  s.kv = g.mla_kv;
  s.rope_theta = g.rope_theta;
  return s;
}

}  // namespace

State* NewState(std::uint32_t max_ctx) {
  if (!g.ready || max_ctx == 0) {
    return nullptr;
  }
  auto* st = new State();
  st->max_ctx = max_ctx;
  const ::ling3::KdaShape ks = MakeKdaShape();
  const ::ling3::MlaShape ms = MakeMlaShape();
  const std::size_t conv_n = static_cast<std::size_t>(ks.embd_r());
  const std::size_t ssm_n = static_cast<std::size_t>(ks.embd_s());
  // Jay 2026-10-01: H1's F4 (REPORT3 §3) said "mla_n = kv_a() * max_ctx" but
  // its patch only set cache.max_pos = max_ctx -- the allocation stayed ONE
  // position per layer, so the kernel would write rows 1..max_ctx-1 out of
  // bounds. Size it for max_ctx here.
  const std::size_t mla_n =
      static_cast<std::size_t>(ms.kv_a()) * static_cast<std::size_t>(max_ctx);

  st->kda_conv_elems = g_dev.layers.size() * conv_n;
  st->kda_ssm_elems = g_dev.layers.size() * ssm_n;
  // LG1 W3 stage (a): the 35 KDA slots were pure waste (never read, never
  // written). The MLA cache holds 7 entries, one per MLA layer, indexed by
  // g.mla_ord. Bit-identical to the old layout: KDA layers never touched
  // mla_cache (RV6 sec 1.1), so deleting their slots changes no byte any
  // kernel reads; each MLA layer's rows keep the same in-slot offsets.
  {
    std::size_t mla_count = 0;
    for (std::size_t i = 0; i < g_dev.layers.size(); ++i) {
      if (g.mla_ord[i] >= 0)
        ++mla_count;
    }
    st->mla_cache_elems = mla_count * mla_n;
  }
  for (std::size_t i = 0; i < g_dev.layers.size(); ++i) {
    st->kda_conv_off.push_back(i * conv_n);
    st->kda_ssm_off.push_back(i * ssm_n);
  }
  for (std::size_t j = 0; j < st->mla_cache_elems / mla_n; ++j) {
    st->mla_off.push_back(j * mla_n);
  }

  // The recurrent state and the cache live in one host-mirrored array each so
  // a session is a single allocation; upload them once and keep them there for
  // the life of the session.
  void* conv_d = DevAlloc(st->kda_conv_elems * sizeof(float));
  void* ssm_d = DevAlloc(st->kda_ssm_elems * sizeof(float));
  void* mla_d = DevAlloc(st->mla_cache_elems * sizeof(float));
  if (!conv_d || !ssm_d || !mla_d) {
    delete st;
    return nullptr;
  }
  st->allocs = {conv_d, ssm_d, mla_d};
  // Remember the device pointers by borrowing kda_conv's slot order.
  DevZero(conv_d, st->kda_conv_elems * sizeof(float));
  DevZero(ssm_d, st->kda_ssm_elems * sizeof(float));
  // LQ3: mla_d was the one state buffer this function never cleared. The
  // device side held whatever the previous process left in it. It happens to
  // be written by mla_cache_append before the attention reads it in the
  // normal chunked flow, and Session::Reset/ResetState clears it per request
  // -- but nothing enforced that, and a fresh hipMalloc on the prefill path
  // is exactly the class of read that is invisible until the allocator
  // changes. Clear it like its two siblings.
  DevZero(mla_d, st->mla_cache_elems * sizeof(float));
  st->kda_conv_device = conv_d;
  st->kda_ssm_device = ssm_d;
  st->mla_cache_device = mla_d;

  const auto act = [&](float** slot, std::size_t n) {
    float* p = static_cast<float*>(DevAlloc(n * sizeof(float)));
    *slot = p;
    return p != nullptr;
  };
  const std::size_t H = static_cast<std::size_t>(g.hidden);
  const std::size_t I = static_cast<std::size_t>(g.inner);
  // The MLA path needs more than d_inner in two buffers: the q projection is
  // [q_dim] floats and lands in tmp, and kv_a is packed at offset q_dim of q.
  // q_dim = heads*mla_head_dim_k (6144) > d_inner = heads*kda_head_dim (4096),
  // and kv_a adds another 576 on top.
  const std::size_t QMAX =
      static_cast<std::size_t>(g.heads) *
          static_cast<std::size_t>(std::max(g.hd, g.mla_head_k)) +
      static_cast<std::size_t>(g.mla_kv) + static_cast<std::size_t>(g.mla_rot);
  const bool ok =
      act(&st->x, H) && act(&st->normed, H) && act(&st->attn_out, H) &&
      act(&st->ffn_in, H) && act(&st->t, H) && act(&st->y, H) &&
      act(&st->q, QMAX) && act(&st->k, I) && act(&st->v, I) && act(&st->f, I) &&
      act(&st->g2, I) && act(&st->beta, static_cast<std::size_t>(g.heads)) &&
      act(&st->tmp, QMAX) &&
      act(&st->gate, static_cast<std::size_t>(g.heads)) &&
      act(&st->logits, static_cast<std::size_t>(g.vocab));
  // Zero for real: the old DevZero(a, 0) passed zero BYTES and returned
  // immediately, so nothing was cleared. Every buffer is written before it is
  // read today, but that is an invariant nothing enforces.
  DevZero(st->x, H * sizeof(float));
  DevZero(st->normed, H * sizeof(float));
  DevZero(st->attn_out, H * sizeof(float));
  DevZero(st->ffn_in, H * sizeof(float));
  DevZero(st->t, H * sizeof(float));
  DevZero(st->y, H * sizeof(float));
  DevZero(st->q, QMAX * sizeof(float));
  DevZero(st->k, I * sizeof(float));
  DevZero(st->v, I * sizeof(float));
  DevZero(st->f, I * sizeof(float));
  DevZero(st->g2, I * sizeof(float));
  DevZero(st->beta, static_cast<std::size_t>(g.heads) * sizeof(float));
  DevZero(st->tmp, QMAX * sizeof(float));
  DevZero(st->gate, static_cast<std::size_t>(g.heads) * sizeof(float));
  DevZero(st->logits, static_cast<std::size_t>(g.vocab) * sizeof(float));
  st->tok_in = static_cast<int*>(DevAlloc(sizeof(int)));
  st->tok_out = static_cast<int*>(DevAlloc(sizeof(int)));
  st->pos_d = static_cast<int*>(DevAlloc(sizeof(int)));
  if (hipHostMalloc(reinterpret_cast<void**>(&st->pos_h), sizeof(int),
                    hipHostMallocDefault) != hipSuccess) {
    st->pos_h = nullptr;
  }
  st->allocs.push_back(st->pos_d);
  if (!ok || !st->tok_in || !st->tok_out || !st->pos_d) {
    for (void* a : st->allocs)
      hipFree(a);
    delete st;
    return nullptr;
  }
  st->kda_scratch_bytes = ::ling3::kda_forward_scratch_bytes(ks, 1, 1);
  st->mla_scratch_bytes =
      ::ling3::mla_block_scratch_bytes(ms, 1, static_cast<int>(max_ctx));
  st->kda_scratch = DevAlloc(st->kda_scratch_bytes);
  st->mla_scratch = DevAlloc(st->mla_scratch_bytes);
  if (st->kda_scratch == nullptr || st->mla_scratch == nullptr) {
    if (st->kda_scratch != nullptr)
      (void)hipFree(st->kda_scratch);
    if (st->mla_scratch != nullptr)
      (void)hipFree(st->mla_scratch);
    st->kda_scratch = nullptr;
    st->mla_scratch = nullptr;
  }
  // LS3 PHASE4: split-K decode-attention partials. P7: sized for a FIXED
  // slot count (kMlaSplitkMaxSlices) instead of max_ctx/1024, so the
  // adaptive short slices cannot over-run the buffers; q_abs holds 4 rows
  // for LM1's multi-row verify entry.
  {
    const std::size_t ks =
        static_cast<std::size_t>(::ling3::kMlaSplitkMaxSlices);
    const std::size_t nhl = static_cast<std::size_t>(ms.n_head) * ks;
    st->mla_sk_q = static_cast<float*>(DevAlloc(
        static_cast<std::size_t>(4) * static_cast<std::size_t>(ms.n_head) *
        static_cast<std::size_t>(ms.kv) * sizeof(float)));
    st->mla_sk_m = static_cast<float*>(DevAlloc(nhl * sizeof(float)));
    st->mla_sk_l = static_cast<float*>(DevAlloc(nhl * sizeof(float)));
    st->mla_sk_acc = static_cast<float*>(
        DevAlloc(nhl * static_cast<std::size_t>(ms.kv) * sizeof(float)));
    if (st->mla_sk_q == nullptr || st->mla_sk_m == nullptr ||
        st->mla_sk_l == nullptr || st->mla_sk_acc == nullptr) {
      if (st->mla_sk_q != nullptr)
        (void)hipFree(st->mla_sk_q);
      if (st->mla_sk_m != nullptr)
        (void)hipFree(st->mla_sk_m);
      if (st->mla_sk_l != nullptr)
        (void)hipFree(st->mla_sk_l);
      if (st->mla_sk_acc != nullptr)
        (void)hipFree(st->mla_sk_acc);
      st->mla_sk_q = nullptr;
      st->mla_sk_m = st->mla_sk_l = st->mla_sk_acc = nullptr;
    }
  }
  return st;
}

// ---------------------------------------------------------------------------
// LS3: batched-prefill scratch.
//
// One set of chunk-capacity buffers per State, created lazily by the first
// ForwardBatch and freed by DestroyState. Every buffer here is sized for the
// chunk cap at creation; ForwardBatch refuses any n above the cap. This is the
// direct answer to H1's PHASE7 fault class (token ids and positions read out
// of one-token buffers) and to the `b_cap` growth-on-demand pattern its post-
// mortem listed as an open candidate: there is no growth, and no buffer sized
// for n = 1 is ever handed T tokens.
// ---------------------------------------------------------------------------
namespace {

/// Same knob the runner uses (engine.cpp's PrefillBatchChunk), read here so
/// the buffers and the chunking stay consistent without a header cycle.
int BatchChunkCap() {
  static const int cap = [] {
    const char* e = std::getenv("LING3_PREFILL_CHUNK");
    long v = e != nullptr ? std::strtol(e, nullptr, 10) : 512;
    if (v < 1)
      v = 1;
    if (v > 512)
      v = 512;
    return static_cast<int>(v);
  }();
  return cap;
}

bool BatchAllLogits() {
  static const bool on = [] {
    const char* e = std::getenv("LING3_BATCH_ALL_LOGITS");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on;
}

/// LING3_BATCH_PROFILE=1: per-section device-synchronised timings, one line
/// per chunk. Diagnosis only -- each measurement adds a device sync.
/// LS3 PHASE4 knobs for the split-K decode attention.
bool MlaSplitkOn() {
  static const bool on = [] {
    const char* e = std::getenv("LING3_MLA_SPLITK");
    return !(e != nullptr && e[0] == '0' && e[1] == '\0');
  }();
  return on;
}
/// LS3 PHASE5: 1 = head-shared partial kernel (each cache slice read once per
/// 8-head group instead of once per head). Gates passed (unit <= 1.3e-8 vs the
/// serial kernel at n_pos <= 30000, 13K PPL A/B 0.093 %, trace 298/5), so ON
/// by default; LING3_MLA_SPLITK_HEADS=0 restores the per-head kernel.
int MlaSplitkHeads() {
  static const int on = [] {
    const char* e = std::getenv("LING3_MLA_SPLITK_HEADS");
    // P6: 2 = register-fused partial (one workgroup per slice, all heads,
    // fused score+latent, no LDS). Unit-gated like the others and the curve
    // winner (31.95/31.36/30.28 tok/s at 13.8K/27.7K/56K vs the LDS-tile
    // kernel's 24.24/17.98/11.88), so it is the default. 1 = the P5
    // LDS-tile kernel, 0 = the P4 per-head kernel.
    return e != nullptr ? std::atoi(e) : 2;
  }();
  return on;
}

/// P7: the adaptive slice knobs. LING3_MLA_SPLITK_SLICE forces a slice
/// length (64..1024); 0 = auto, sized so the partial grid is ~target
/// workgroups (LING3_MLA_SPLITK_WG, default 120 = 3x the 40 CUs).
int MlaSplitkSlice(int n_pos) {
  static const int forced = [] {
    const char* e = std::getenv("LING3_MLA_SPLITK_SLICE");
    long v = e != nullptr ? std::strtol(e, nullptr, 10) : 0;
    if (v < 0)
      v = 0;
    if (v > 1024)
      v = 0;
    return static_cast<int>(v);
  }();
  if (forced > 0) {
    // clamp up if a forced slice would exceed the partial slots
    if ((n_pos + forced - 1) / forced > ::ling3::kMlaSplitkMaxSlices) {
      return ::ling3::mla_splitk_slice_auto(n_pos,
                                            ::ling3::kMlaSplitkMaxSlices);
    }
    return forced;
  }
  static const int wg = [] {
    const char* e = std::getenv("LING3_MLA_SPLITK_WG");
    long v = e != nullptr ? std::strtol(e, nullptr, 10) : 120;
    if (v < 8)
      v = 8;
    if (v > ::ling3::kMlaSplitkMaxSlices)
      v = ::ling3::kMlaSplitkMaxSlices;
    return static_cast<int>(v);
  }();
  return ::ling3::mla_splitk_slice_auto(n_pos, wg);
}

/// Jay build28 glue (RV1 B2): the split-K partials hold kMlaSplitkMaxSlices
/// slots per head; above n_pos 196,608 even a 1,024-position slice needs more,
/// so those positions take the serial path instead of over-running them.
bool MlaSplitkSlicesFit(int n_pos) {
  const int s = MlaSplitkSlice(n_pos);
  return s > 0 && (n_pos + s - 1) / s <= ::ling3::kMlaSplitkMaxSlices;
}

int MlaSplitkMin() {
  static const int min = [] {
    const char* e = std::getenv("LING3_MLA_SPLITK_MIN");
    // P7: 1024, the measured serial/split-K crossover. MEASURED on MAX-1
    // (out-p7chain.txt bench arms, sha 6cbab1eaea0df5f2): at ~300 decode
    // positions (bench.sh short prompts) the serial 32-block kernel wins
    // (39.20/39.17 tok/s vs split-K's 38.92/38.91); from ~1.4K positions
    // split-K wins big (38.53 vs 34.69 on the 1,387-token bench prompt) and
    // the adaptive slice (MlaSplitkSlice, ~120 workgroups) keeps it ahead at
    // every depth: 38.6/38.3/36.9/35.9/34.2/30.8 at 1.6K/3K/6.5K/13.8K/27.7K/
    // 56K vs the serial fallback's 33.6/30.4/23.3/33.3 (ls3-b27) and the
    // fork no-draft's 35.0/34.8/34.3/33.4/31.8/29.0 (fork3-m1).
    long v = e != nullptr ? std::strtol(e, nullptr, 10) : 1024;
    if (v < 1)
      v = 1;
    return static_cast<int>(v);
  }();
  return min;
}

bool BatchProfile() {
  static const bool on = [] {
    const char* e = std::getenv("LING3_BATCH_PROFILE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on;
}

/// P3: LING3_PREFILL_W8A8=0 opts the MoE prefill back onto the phase-2 float
/// tiled GEMM (still on the compacted grid). Default ON: the W8A8 path is
/// the deliverable; the knob is the escape hatch, like LING3_PREFILL_BATCH.
/// Types other than ROCmFP4 fall back regardless (checked at the call site).
bool PrefillW8A8() {
  static const bool on = [] {
    const char* e = std::getenv("LING3_PREFILL_W8A8");
    return e == nullptr || e[0] == '\0' || e[0] != '0';
  }();
  return on;
}

/// LF1 SPEC3: LING3_MLA_TILE=1 routes batched-prefill MLA attention through
/// the port of the fork's tile flash-attention kernel
/// (kernels/ling3_mla_tile.hip) instead of LS3's mla_attn_tok. Default OFF.
/// Unsupported shapes or misaligned buffers fall back to mla_attn_tok inside
/// the launcher, so the knob can never leave a chunk without attention.
bool MlaTile() {
  static const bool on = [] {
    const char* e = std::getenv("LING3_MLA_TILE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on;
}

/// Jay build28d glue: LING3_MLA_TILE_MAX caps the tile path by depth. Measured
/// on amdaimax2 (build28c vs build28b curves): the tile kernel wins at
/// 1.6K-27.7K (317 vs 282 tok/s at 27.7K) but loses at 56K (197 vs 213) to
/// the LW1 two-pass WMMA, so chunks ending beyond the cap take the else-branch
/// (w2 when LING3_MLA_WMMA2 is set, else mla_attn_tok). 0 / unset = no cap.
bool MlaTileFits(int n_pos) {
  static const long cap = [] {
    const char* e = std::getenv("LING3_MLA_TILE_MAX");
    long v = e != nullptr ? std::strtol(e, nullptr, 10) : 0;
    return v < 0 ? 0L : v;
  }();
  return cap == 0 || n_pos <= cap;
}

/// LX1 SPEC: LING3_MLA_FW=1 routes batched-prefill MLA attention through
/// the fused WMMA flash kernel (kernels/ling3_mla_fw.hip): ONE attention
/// launch for the whole 512-token chunk, absorb/de-absorb as batched
/// kernels around it, nothing materialized in global memory. Default OFF;
/// any audit refusal falls back to the proven paths (fail-closed).
bool MlaFw() {
  static const bool on = [] {
    const char* e = std::getenv("LING3_MLA_FW");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on;
}

/// LX1: the fused path's depth window — chunks ending at >= min positions
/// take the fused kernel, and <= max if max > 0 (0 = every depth, the spec's
/// default: no 16,384-style cap). MEASURED on this box (unit bench, the
/// four spec anchors + shallow cases): the fused path is faster than the
/// tile kernel at EVERY measured depth from n_pos 130 up (5.15 vs 7.06 ms
/// at 512, 21.2 vs 61.3 at 8K, 158.0 vs 517.0 at 57K), so the default min
/// is one full prefill chunk; see lx1/REPORT.md.
int MlaFwMin() {
  static const int v = [] {
    const char* e = std::getenv("LING3_MLA_FW_MIN");
    long v = e != nullptr ? std::strtol(e, nullptr, 10) : 512;
    return v < 0 ? 0 : (int)v;
  }();
  return v;
}
int MlaFwMax() {
  static const int v = [] {
    const char* e = std::getenv("LING3_MLA_FW_MAX");
    long v = e != nullptr ? std::strtol(e, nullptr, 10) : 0;
    return v < 0 ? 0 : (int)v;
  }();
  return v;
}

/// Section ids for the profiler. Order matters only for the printed line.
enum BatchSection {
  kSecEmbed = 0,
  kSecAttnNorm,
  kSecKdaProj,
  kSecKda,
  kSecMlaProj,
  kSecMla,
  kSecAttnOut,  // wo / m_wo gemms
  kSecFfnNorm,
  kSecRoute,
  kSecMoeGemm,
  kSecShared,
  kSecResidual,
  kSecFinal,
  kSecMax
};

/// Token sub-batch for the token-parallel MLA attention (also sizes its
/// score scratch: kMlaTokBatchCap * heads * max_ctx floats).
constexpr int kMlaTokBatchCap = 64;

/// Max tokens per routed-expert sub-group (host-side split, P2-A).
constexpr int kMoeGroupCap = 64;
const char* const kSecNames[kSecMax] = {
    "embed",    "attn_norm", "kda_proj", "kda",    "mla_proj", "mla",  "attn_o",
    "ffn_norm", "route",     "moe",      "shared", "residual", "final"};

struct BatchProfiler {
  double ms[kSecMax] = {0};
  std::chrono::steady_clock::time_point t0;
  void begin() { t0 = std::chrono::steady_clock::now(); }
  double end() {
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
};

struct BatchBufs {
  int cap{0};
  // [H * cap]
  float *x{nullptr}, *normed{nullptr}, *attn{nullptr}, *ffn_in{nullptr},
      *t{nullptr}, *y{nullptr}, *fsh{nullptr};
  // [I * cap]
  float *q{nullptr}, *kk{nullptr}, *vv{nullptr}, *fp{nullptr}, *g2{nullptr},
      *kdaout{nullptr};
  // [G * cap]
  float *beta{nullptr}, *gate{nullptr};
  // MLA per cap
  float* mq{nullptr};      // [q_dim * cap]
  float* kva{nullptr};     // [kv_a() * cap]
  float* mlaout{nullptr};  // [o_dim() * cap]
  float* kda_scratch{nullptr};
  std::size_t kda_scratch_bytes{0};
  // mla_attn_tok score scratch: [kMlaTokBatch][heads][max_ctx] -- one row per
  // (sub-batched token, head), laid out as the kernel indexes it.
  float* mla_scores{nullptr};
  // LW1: two-pass WMMA prefill attention staging (LING3_MLA_WMMA2=1).
  ::ling3::MlaW2Bufs w2{};
  // LX1: fused WMMA flash prefill staging (LING3_MLA_FW=1); k16/vt are
  // shared with w2 above, these are the fused path's own q16/o/rowsum.
  ::ling3::MlaFwBufs fw{};
  // MoE per cap: routed slots [cap * 8] x expert_ff, shared/dense [cap] x
  // n_ff_max
  float *eg{nullptr}, *eu{nullptr}, *eh{nullptr};
  float *sg{nullptr}, *su{nullptr}, *sh{nullptr};
  // P2-C: slot-major routed partials [cap*8 x hidden] + token->slot map; the
  // down-projection plain-stores here and one gather-sum folds the 8 slots
  // per token (no atomicAdd, deterministic slot order).
  float* y2{nullptr};
  int* token_slots{nullptr};
  int* slot_tok{nullptr};
  float* slot_w{nullptr};
  int* ident{nullptr};  // [cap * 8] = {0,1,...}: slot-row identity for gathers
  int *grp_e{nullptr}, *grp_off{nullptr}, *grp_cnt{nullptr};
  // LS1 PHASE8 item 2: capacity of grp_e / grp_off / grp_cnt, recorded where
  // they are allocated so the device grouping kernel can be handed the SAME
  // bound the host code used, instead of one guessed at the call site.
  int grp_e_cap{0};
  // P3: 16-token-tile prefix over the groups (compacted grids; the WMMA
  // activation tiles are numbered by it too) and the per-slot quantiser
  // destination (tile<<4 | lane, group-local).
  int* grp_zp{nullptr};
  int* slot_dest{nullptr};
  // P3: tiled Q8_1 activations for the W8A8 WMMA path. q8_t holds the
  // gate/up input (B->t rows, slot order, k = hidden), q8_h the down input
  // (B->eh rows, k = expert_ff). Both sized for the worst-case tile count.
  void *q8_t{nullptr}, *q8_h{nullptr};
  // P3 step 3: dense-path activation tiles. q8_hidden serves the attn
  // projections (B->normed) and, later in the same layer, the FFN gate/up
  // input (B->t) — the two uses never overlap. q8_odim serves the two
  // attention output projections (B->mlaout / B->kdaout, both inner dims),
  // q8_ffn the FFN down input (B->sh, k = n_ff).
  void *q8_hidden{nullptr}, *q8_odim{nullptr}, *q8_ffn{nullptr};
  // per-chunk inputs / router outputs (device)
  int *tok{nullptr}, *pos{nullptr};
  int *ids{nullptr}, *rc{nullptr};
  float* wts{nullptr};
  // all-position logits, test gate only (LING3_BATCH_ALL_LOGITS=1)
  float* logits_all{nullptr};
  // LS4: the speculative verify needs the logits of the FIRST few rows of a
  // 3-row batch. logit_rows routes those through here -- at most
  // kSpeculativeRows rows, ~5 MiB -- instead of the chunk-sized logits_all
  // scratch, which is 322 MB at chunk 512 and exists only so the
  // teacher-forcing gate can read every position of a full prefill chunk.
  float* logits_spec{nullptr};
  std::vector<void*> allocs;
};

// LS4: the most rows of one ForwardBatch whose logits the speculative verify
// ever reads -- the sampled token plus n_max drafts. 8 rows x 157184 floats is
// 5.03 MB, against 322 MB for the chunk-sized all-position scratch.
constexpr int kSpeculativeRows = 8;

void FreeBatchBufs(BatchBufs* b) noexcept;

BatchBufs* NewBatchBufs(std::uint32_t max_ctx) {
  const ::ling3::KdaShape ks = MakeKdaShape();
  const ::ling3::MlaShape ms = MakeMlaShape();
  auto* b = new BatchBufs();
  b->cap = BatchChunkCap();
  const int cap = b->cap;
  const std::size_t H = static_cast<std::size_t>(g.hidden);
  const std::size_t I = static_cast<std::size_t>(g.inner);
  const std::size_t G = static_cast<std::size_t>(g.heads);
  const std::size_t QD = static_cast<std::size_t>(ms.q_dim());
  const std::size_t KVA = static_cast<std::size_t>(ms.kv_a());
  const std::size_t OD = static_cast<std::size_t>(ms.o_dim());
  const std::size_t NFF = static_cast<std::size_t>(
      g.expert_ff > g.ffn_len ? g.expert_ff : g.ffn_len);
  const std::size_t CAP = static_cast<std::size_t>(cap);
  const std::size_t SLOTS = CAP * kMaxExperts;
  const std::size_t E = static_cast<std::size_t>(g.n_expert);
  // P3: worst-case group-local 16-token tiles the W8A8 path can address:
  // every group pads to less than one tile beyond its slots, and there are
  // at most E + SLOTS/kMoeGroupCap groups (kMoeGroupCap splits).
  const std::size_t kQ8Tiles = SLOTS / 16 + E + SLOTS / 64 + 2;

  const auto af = [&](float** slot, std::size_t floats) {
    float* p = static_cast<float*>(DevAlloc(floats * sizeof(float)));
    *slot = p;
    if (p != nullptr) {
      b->allocs.push_back(p);
      DevPoison(p, floats * sizeof(float));  // LQ3
    }
    return p != nullptr;
  };
  const auto ai = [&](int** slot, std::size_t ints) {
    int* p = static_cast<int*>(DevAlloc(ints * sizeof(int)));
    *slot = p;
    if (p != nullptr) {
      b->allocs.push_back(p);
      DevPoison(p, ints * sizeof(int));  // LQ3
    }
    return p != nullptr;
  };
  const auto ab = [&](void** slot, std::size_t bytes) {
    void* p = DevAlloc(bytes);
    *slot = p;
    if (p != nullptr) {
      b->allocs.push_back(p);
      DevPoison(p, bytes);  // LQ3
    }
    return p != nullptr;
  };

  b->kda_scratch_bytes = ::ling3::kda_forward_scratch_bytes(ks, cap, 1);
  // LW1: WMMA prefill staging caps (kernels/ling3_mla_w2.hpp). max_ctx is
  // rounded up to the 64-position tile geometry the GEMM fragments use.
  const std::size_t W2CAP = ((std::size_t)max_ctx + 63) / 64 * 64;
  const std::size_t W2KV = static_cast<std::size_t>(ms.kv);
  const std::size_t W2KVA = static_cast<std::size_t>(ms.kv_a());
  const std::size_t W2ROWS = static_cast<std::size_t>(::ling3::kMlaW2Rows);
  b->w2.cap = static_cast<int>(W2CAP);
  const bool ok =
      af(&b->x, H * CAP) && af(&b->normed, H * CAP) && af(&b->attn, H * CAP) &&
      af(&b->ffn_in, H * CAP) && af(&b->t, H * CAP) && af(&b->y, H * CAP) &&
      af(&b->fsh, H * CAP) && af(&b->q, I * CAP) && af(&b->kk, I * CAP) &&
      af(&b->vv, I * CAP) && af(&b->fp, I * CAP) && af(&b->g2, I * CAP) &&
      af(&b->kdaout, I * CAP) && af(&b->beta, G * CAP) &&
      af(&b->gate, G * CAP) && af(&b->mq, QD * CAP) && af(&b->kva, KVA * CAP) &&
      af(&b->mlaout, OD * CAP) &&
      af(&b->kda_scratch, b->kda_scratch_bytes / sizeof(float)) &&
      af(&b->mla_scores, static_cast<std::size_t>(kMlaTokBatchCap) *
                             static_cast<std::size_t>(g.heads) *
                             static_cast<std::size_t>(max_ctx)) &&
      af(&b->eg, SLOTS * static_cast<std::size_t>(g.expert_ff)) &&
      af(&b->eu, SLOTS * static_cast<std::size_t>(g.expert_ff)) &&
      af(&b->eh, SLOTS * static_cast<std::size_t>(g.expert_ff)) &&
      af(&b->y2, SLOTS * H) && ai(&b->token_slots, SLOTS) &&
      af(&b->sg, NFF * CAP) && af(&b->su, NFF * CAP) && af(&b->sh, NFF * CAP) &&
      ai(&b->slot_tok, SLOTS) && ai(&b->ident, SLOTS) &&
      af(&b->slot_w, SLOTS) &&
      // +split sub-groups for fat experts (P2-A), with headroom.
      ((b->grp_e_cap = E + SLOTS / 8 + 2), ai(&b->grp_e, b->grp_e_cap) &&
                                               ai(&b->grp_off, b->grp_e_cap) &&
                                               ai(&b->grp_cnt, b->grp_e_cap)) &&
      // P3: the tile prefix needs one entry more than the groups; the Q8_1
      // buffers take the worst-case group-local tile count (every group pads
      // to <1 tile beyond its slots: SLOTS/16 + one per group).
      ai(&b->grp_zp, E + SLOTS / 8 + 3) && ai(&b->slot_dest, SLOTS) &&
      ab(&b->q8_t, kQ8Tiles * (H / 32) * 576) &&
      ab(&b->q8_h, kQ8Tiles * (NFF / 32) * 576) &&
      // P3 step 3 dense tiles: ceil(cap/16) identity tiles per k.
      ab(&b->q8_hidden, ((CAP + 15) / 16) * (H / 32) * 576) &&
      ab(&b->q8_odim, ((CAP + 15) / 16) * ((OD > I ? OD : I) / 32) * 576) &&
      ab(&b->q8_ffn, ((CAP + 15) / 16) * (NFF / 32) * 576) &&
      ai(&b->tok, CAP) && ai(&b->pos, CAP) && ai(&b->ids, SLOTS) &&
      ai(&b->rc, CAP) && af(&b->wts, SLOTS) &&
      af(&b->logits_spec, static_cast<std::size_t>(kSpeculativeRows) *
                              static_cast<std::size_t>(g.vocab)) &&
      // LW1: the WMMA prefill attention's staging buffers.
      ab(reinterpret_cast<void**>(&b->w2.k16),
         W2CAP * W2KVA * sizeof(__half)) &&
      ab(reinterpret_cast<void**>(&b->w2.vt), W2KV * W2CAP * sizeof(__half)) &&
      ab(reinterpret_cast<void**>(&b->w2.q16),
         W2ROWS * W2KVA * sizeof(__half)) &&
      ab(reinterpret_cast<void**>(&b->w2.p16),
         W2ROWS * W2CAP * sizeof(__half)) &&
      ab(reinterpret_cast<void**>(&b->w2.o), W2ROWS * W2KV * sizeof(float)) &&
      ab(reinterpret_cast<void**>(&b->w2.rowsum), W2ROWS * sizeof(float));
  // LX1: the fused WMMA flash path's own staging (LING3_MLA_FW=1 only —
  // q16/o are full-chunk-sized, 52 MB together). k16/vt above are shared.
  if (ok && std::getenv("LING3_MLA_FW") != nullptr) {
    const std::size_t FWR = static_cast<std::size_t>(::ling3::kMlaFwRows);
    const std::size_t FQT =
        static_cast<std::size_t>(32) *
        static_cast<std::size_t>(::ling3::kMlaFwChunkTokens) *
        static_cast<std::size_t>(128);
    if (!(ab(reinterpret_cast<void**>(&b->fw.q16),
             FWR * W2KVA * sizeof(__half)) &&
          ab(reinterpret_cast<void**>(&b->fw.o), FWR * W2KV * sizeof(float)) &&
          ab(reinterpret_cast<void**>(&b->fw.rowsum), FWR * sizeof(float)) &&
          ab(reinterpret_cast<void**>(&b->fw.qt), FQT * sizeof(float)))) {
      FreeBatchBufs(b);
      return nullptr;
    }
    b->fw.rows_cap = ::ling3::kMlaFwRows;
  }
  if (!ok) {
    FreeBatchBufs(b);
    return nullptr;
  }
  // slot-row identity, written once: eg/eu rows live at their global slot.
  std::vector<int> ident(SLOTS);
  for (std::size_t j = 0; j < SLOTS; ++j)
    ident[j] = static_cast<int>(j);
  if (hipMemcpy(b->ident, ident.data(), SLOTS * sizeof(int),
                hipMemcpyHostToDevice) != hipSuccess ||
      (BatchAllLogits() && ((b->logits_all = static_cast<float*>(DevAlloc(
                                 CAP * static_cast<std::size_t>(g.vocab) *
                                 sizeof(float)))) == nullptr ||
                            (b->allocs.push_back(b->logits_all), false)))) {
    FreeBatchBufs(b);
    return nullptr;
  }
  return b;
}

void FreeBatchBufs(BatchBufs* b) noexcept {
  if (b == nullptr)
    return;
  for (void* p : b->allocs) {
    if (p != nullptr)
      (void)hipFree(p);
  }
  delete b;
}

}  // namespace

void DestroyState(State* state) noexcept {
  if (state == nullptr)
    return;
  // Free every device allocation this session made. Previously only the three
  // big state arrays were tracked, so each Session leaked 15 activation buffers
  // plus tok_in/tok_out -- invisible in a short test, permanent in a seat.
  for (float* a :
       {state->x, state->normed, state->attn_out, state->ffn_in, state->t,
        state->y, state->q, state->k, state->v, state->f, state->g2,
        state->beta, state->tmp, state->gate, state->logits}) {
    if (a != nullptr)
      (void)hipFree(a);
  }
  if (state->tok_in != nullptr)
    (void)hipFree(state->tok_in);
  if (state->tok_out != nullptr)
    (void)hipFree(state->tok_out);
  if (state->pos_h != nullptr)
    (void)hipFreeHost(state->pos_h);
  if (state->kda_scratch != nullptr)
    (void)hipFree(state->kda_scratch);
  if (state->mla_scratch != nullptr)
    (void)hipFree(state->mla_scratch);
  if (state->mla_sk_q != nullptr)
    (void)hipFree(state->mla_sk_q);
  if (state->mla_sk_m != nullptr)
    (void)hipFree(state->mla_sk_m);
  if (state->mla_sk_l != nullptr)
    (void)hipFree(state->mla_sk_l);
  if (state->mla_sk_acc != nullptr)
    (void)hipFree(state->mla_sk_acc);
  for (void* a : state->allocs) {
    if (a != nullptr)
      (void)hipFree(a);
  }
  // LS3: the chunk-sized batch-prefill scratch, if the session ever batched.
  if (state->batch != nullptr) {
    FreeBatchBufs(static_cast<BatchBufs*>(state->batch));
    state->batch = nullptr;
  }
  delete state;
}

void ResetState(State* state) noexcept {
  if (state == nullptr)
    return;
  // Winding the position back to 0 is NOT enough for the RECURRENT state: the
  // conv history and the SSM matrix carry the previous prompt's values into the
  // next one, producing plausible-looking wrong tokens. Clear those two.
  if (state->kda_conv_device != nullptr)
    DevZero(state->kda_conv_device, state->kda_conv_elems * sizeof(float));
  if (state->kda_ssm_device != nullptr)
    DevZero(state->kda_ssm_device, state->kda_ssm_elems * sizeof(float));
  // RV4 (Jay build28f): the latent cache is position-indexed and written before
  // it is read -- mla_block, mla_block_splitk, ForwardBatch and StepMulti all
  // append rows [n_pos - n_tokens, n_pos) ahead of the attention that reads the
  // first n_pos rows, and every reader is bounded by n_pos. Rewinding `pos` is
  // the whole reset; zeroing it copied ~6 GiB per request (~90 ms, twice per
  // request). NewState still zeroes it once at session creation.
  state->pos = 0;
}

// ---------------------------------------------------------------------------
// One token.
// ---------------------------------------------------------------------------
bool Step(std::int32_t token, State* st, std::span<float> out_logits,
          std::string* error_msg) {
  if (!g.ready || st == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "ling3 ROCm: no state";
    return false;
  }
  const std::size_t H = static_cast<std::size_t>(g.hidden);
  const std::size_t I = static_cast<std::size_t>(g.inner);
  const std::size_t G = static_cast<std::size_t>(g.heads);
  const std::size_t V = static_cast<std::size_t>(g.vocab);

  hipStream_t stream =
      nullptr;  // the null stream; LK's and LX's kernels use it

  // token_embd is Q6_K (text file) or Q5_K (VL file); its type travels in
  // g_dev.token_embd_type and ops_embedding_row decodes whichever it is.
  const int tok = static_cast<int>(token);
  if (hipMemcpy(st->tok_in, &tok, sizeof(int), hipMemcpyHostToDevice) !=
      hipSuccess) {
    if (error_msg)
      *error_msg = "ling3 ROCm: token upload";
    return false;
  }
  ::ling3::ops_embedding_row(g_dev.token_embd_type, g_dev.token_embd, g.hidden,
                             g.vocab, st->tok_in, 1, st->x, stream);
  GpuTrace(st, static_cast<std::uint32_t>(-1), "inp_embd", st->x, H);

  gufo::models::ling3::vision::OnHidden(st->x, static_cast<int>(st->pos), 1,
                                        static_cast<int>(H), stream);

  const ::ling3::KdaShape ks = MakeKdaShape();
  const ::ling3::KdaScalars ksc;
  const ::ling3::MlaShape ms = MakeMlaShape();
  const ::ling3::MlaScalars msc;

  // LS2 PHASE7: push the position ONCE per token from pinned staging, instead
  // of a blocking pageable copy inside the MLA branch (7 per token). Safe to
  // rewrite pos_h here: the previous Step ended with a blocking D2H, so the GPU
  // has finished everything that read it.
  if (st->pos_h != nullptr) {
    *st->pos_h = static_cast<int>(st->pos);
    if (hipMemcpyAsync(st->pos_d, st->pos_h, sizeof(int), hipMemcpyHostToDevice,
                       stream) != hipSuccess) {
      if (error_msg)
        *error_msg = "ling3 ROCm: position upload";
      return false;
    }
  }
  // g_dev.experts only holds MoE layers, so index it with its own counter
  // LS1 PHASE7 item 3: six (MLA: three) projections in a layer all read the
  // SAME st->normed, and st->normed is re-quantised by every one of them.
  // Quantise it once into a named slot and route them through that. The
  // fallback is deliberate: the MTP block is Q8_0, not ROCmFP4, and for a
  // non-FP4 type ling3_gemv uses the float kernel and quantises nothing -- so
  // the FP4-only shortcut must not be taken blindly.
  const int k_attn_slot = LING3_Q8_SLOT_ATTN_NORM;
  auto gemv_pre = [](int qtype, const void* W, const float* xsrc, int slot,
                     float* y, int k, int n) {
    if (qtype == LING3_Q4_0_ROCMFP4 || qtype == LING3_Q4_0_ROCMFP4_FAST) {
      return ling3_gemv_fp4_q8(qtype, W, slot, y, k, n);
    }
    return ling3_gemv(qtype, W, xsrc, y, k, n);
  };

  // rather than the block number.
  std::size_t expert_slot = 0;
  const std::size_t n_layers = g_dev.layers.size();
  for (std::size_t li = 0; li < n_layers; ++li) {
    Layer& L = g_dev.layers[li];

    // RV25 wave1 F4: layer 0's attn_norm has no residual in front of it (its
    // input is the embedding row), so it stays a plain rms_norm. Every later
    // layer gets its attn_norm from the fused residual#2 at the END of the
    // previous layer, which already holds a[i]+b[i] in registers and so skips
    // the round trip through st->x.
    if (li == 0) {
      ::ling3::ops_rms_norm(st->x, L.attn_norm, st->normed, g.hidden, 1, g.eps,
                            stream);
    }
    GpuTrace(st, static_cast<std::uint32_t>(li), "attn_norm", st->normed, H);
    // One quantisation of st->normed for every projection in this layer that
    // consumes it. Named slot, no pointer cache: st->normed is about to be
    // overwritten by the next layer's rms_norm, and a pointer-keyed cache would
    // hand back this layer's vector for the next one.
    ling3_act_prepare(st->normed, g.hidden, k_attn_slot);

    if (g.is_mla[li]) {
      const std::size_t q_dim = static_cast<std::size_t>(ms.q_dim());
      // q [6144] and kv_a [576] and gate [32] are gathered into one scratch.
      gemv_pre(L.mq.type, L.mq.d, st->normed, k_attn_slot, st->tmp, g.hidden,
               static_cast<int>(q_dim));
      gemv_pre(L.m_kv_a_mqa.type, L.m_kv_a_mqa.d, st->normed, k_attn_slot,
               st->q + q_dim, g.hidden, ms.kv_a());
      gemv_pre(L.m_gate.type, L.m_gate.d, st->normed, k_attn_slot, st->gate,
               g.hidden, static_cast<int>(G));

      ::ling3::MlaInputs in;
      in.q = ::ling3::make_view(st->tmp, static_cast<int>(q_dim), 1);
      in.kv_a = ::ling3::make_view(st->q + q_dim, ms.kv_a(), 1);
      in.gate = ::ling3::make_view(st->gate, static_cast<int>(G), 1);
      in.n_tokens = 1;
      in.n_seqs = 1;

      ::ling3::MlaCache cache;
      // LG1 W3: reindex mla_off through the MLA ordinal map. All three
      // construction sites are inside `if (g.is_mla[li])`, so mla_ord[li] >= 0
      // here; the abort is a safety net for any future invariant break.
      if (g.mla_ord[li] < 0) {
        std::fprintf(stderr, "lg1: mla_ord[%zu] < 0 at Step MlaCache build\n",
                     li);
        std::abort();
      }
      cache.kv = static_cast<float*>(st->mla_cache_device) +
                 st->mla_off[g.mla_ord[li]];
      cache.s_pos = ms.kv_a();
      cache.s_seq = 0;
      cache.max_pos = static_cast<int>(st->max_ctx);

      ::ling3::MlaWeights mw;
      mw.wk_b = L.m_wk_b_f32;
      mw.wv_b = L.m_wv_b_f32;
      mw.wk_bT = L.m_wk_bT;
      mw.wv_bT = L.m_wv_bT;
      mw.kv_a_norm = L.m_kv_a_norm;

      // `pos` must be DEVICE memory: mla_block's contract is `const int * pos`
      // and rope_kernel dereferences it on the GPU (const float angle = pos[t]
      // * freq). A host stack array here is a GCVM_L2_PROTECTION_FAULT on the
      // first MLA layer of the first token.
      if (st->pos_h == nullptr) {
        hipMemcpy(st->pos_d, &st->pos, sizeof(int), hipMemcpyHostToDevice);
      }
      // out is [4096] = heads * head_v
      // LS3 PHASE4: split-K ("flash decoding") attention once the visible
      // cache is long enough to need it -- the serial mla_attn runs 32
      // blocks total and its throughput falls linearly with the context.
      // LING3_MLA_SPLITK=0 forces the serial path back on.
      if (MlaSplitkOn() && st->mla_sk_acc != nullptr &&
          st->pos + 1 <= static_cast<int>(st->max_ctx) &&
          st->pos + 1 >= MlaSplitkMin() && MlaSplitkSlicesFit(st->pos + 1)) {
        ::ling3::mla_block_splitk(
            ms, msc, mw, cache, in, st->pos_d, st->pos + 1,
            ::ling3::make_view(st->k, ms.o_dim(), 1), st->mla_sk_q,
            st->mla_sk_m, st->mla_sk_l, st->mla_sk_acc,
            MlaSplitkSlice(st->pos + 1), MlaSplitkHeads(), stream);
      } else if (st->mla_scratch != nullptr &&
                 st->pos + 1 <= static_cast<int>(st->max_ctx) &&
                 st->mla_scratch_bytes >=
                     ::ling3::mla_block_scratch_bytes(ms, 1, st->pos + 1)) {
        ::ling3::mla_block_scratch(
            ms, msc, mw, cache, in, st->pos_d, st->pos + 1,
            ::ling3::make_view(st->k, ms.o_dim(), 1), st->mla_scratch,
            st->mla_scratch_bytes, stream, st->pos_d);
      } else {
        ::ling3::mla_block(ms, msc, mw, cache, in, st->pos_d, st->pos + 1,
                           ::ling3::make_view(st->k, ms.o_dim(), 1), stream,
                           st->pos_d);
      }
      // Jay 2026-10-01: the MLA output is o_dim = heads*head_v = 4096 floats;
      // attn_out is allocated hidden = 2560, so writing it there overran by
      // 1536 floats (the GPU fault on the first request, ling3_mla.hip:378).
      // And the projection went to `normed` while the residual add below reads
      // attn_out. Raw MLA output -> st->k (4096 floats, unused on the MLA
      // path), projection -> attn_out (hidden), exactly as the KDA branch and
      // reference.cpp:370 do. LC1 SPEC4 Q2-MLA: the wo Q8_1 quantisation is the
      // decode attention's epilogue, so this projection reads that activation
      // instead of paying a second k_q8_1_quant (-1 launch per MLA layer, -7
      // per token). Bit-identical end to end: the epilogue is k_q8_1_quant's
      // recipe verbatim (same signed amax with the lowest-index tie-break, same
      // d = mv / 127, same id, same roundf(x * id)) and the matvec is
      // k_gemv_wide_q8_m at m = 1, whose per-row dot (ls1_load_blk32 +
      // ls1_dot_blk32) and ls1_block_sum reduction are k_gemv_wide_q8_cta64's
      // own, in the same order. LING3_MLA_DECODE_FUSED=0, a non-FP4 m_wo or a
      // failed allocation keeps the old quant+GEMV pair below, unchanged --
      // that is the A/B the bench gate measures.
      bool wo_fused = false;
      if (::ling3::mla_decode_fused() && ::ling3::mla_wo_q8_qs() != nullptr &&
          ::ling3::mla_wo_q8_ds() != nullptr &&
          (L.m_wo.type == LING3_Q4_0_ROCMFP4 ||
           L.m_wo.type == LING3_Q4_0_ROCMFP4_FAST)) {
        wo_fused =
            ling3_gemv_q8_m(L.m_wo.type, L.m_wo.d, ::ling3::mla_wo_q8_qs(),
                            ::ling3::mla_wo_q8_ds(), st->attn_out, ms.o_dim(),
                            g.hidden, 1) == LING3_OK;
      }
      if (!wo_fused) {
        ling3_gemv(L.m_wo.type, L.m_wo.d, st->k, st->attn_out, ms.o_dim(),
                   g.hidden);
      }
      GpuTrace(st, static_cast<std::uint32_t>(li), "mla_q", st->tmp, q_dim);
      GpuTrace(st, static_cast<std::uint32_t>(li), "mla_attn_out", st->k,
               static_cast<std::size_t>(ms.o_dim()));
      GpuTrace(st, static_cast<std::uint32_t>(li), "attn_out", st->attn_out, H);
      // attn_out now holds the attention output projection result.
      (void)msc;
    } else {
      gemv_pre(L.wq.type, L.wq.d, st->normed, k_attn_slot, st->q, g.hidden,
               static_cast<int>(I));
      gemv_pre(L.wk.type, L.wk.d, st->normed, k_attn_slot, st->k, g.hidden,
               static_cast<int>(I));
      gemv_pre(L.wv.type, L.wv.d, st->normed, k_attn_slot, st->v, g.hidden,
               static_cast<int>(I));
      gemv_pre(L.wf.type, L.wf.d, st->normed, k_attn_slot, st->f, g.hidden,
               static_cast<int>(I));
      gemv_pre(L.wg.type, L.wg.d, st->normed, k_attn_slot, st->g2, g.hidden,
               static_cast<int>(I));
      gemv_pre(L.wbeta.type, L.wbeta.d, st->normed, k_attn_slot, st->beta,
               g.hidden, static_cast<int>(G));

      ::ling3::KdaWeights kw;
      kw.conv_wq = L.conv_q;
      kw.conv_wk = L.conv_k;
      kw.conv_wv = L.conv_v;
      kw.a_exp = L.a_exp;
      kw.dt_bias = L.dt_bias;
      kw.o_norm = L.o_norm;

      ::ling3::KdaInputs ki;
      ki.q = ::ling3::make_view(st->q, static_cast<int>(I), 1);
      ki.k = ::ling3::make_view(st->k, static_cast<int>(I), 1);
      ki.v = ::ling3::make_view(st->v, static_cast<int>(I), 1);
      ki.f = ::ling3::make_view(st->f, static_cast<int>(I), 1);
      ki.g2 = ::ling3::make_view(st->g2, static_cast<int>(I), 1);
      ki.beta = ::ling3::make_view(st->beta, static_cast<int>(G), 1);
      ki.n_tokens = 1;
      ki.n_seqs = 1;

      ::ling3::KdaState ks_state;
      ks_state.conv_state =
          static_cast<float*>(st->kda_conv_device) + st->kda_conv_off[li];
      ks_state.ssm_state =
          static_cast<float*>(st->kda_ssm_device) + st->kda_ssm_off[li];
      ks_state.n_seqs = 1;

      // LKF: when the t32 decode fusion is on, K2's epilogue already wrote the
      // wo Q8 slot. Skip k_q8_1_quant and matvec that slot. Other variants keep
      // ling3_gemv, which quantises st->tmp itself.
      int8_t* wo_qs = nullptr;
      float* wo_ds = nullptr;
      const bool wo_fused =
          ::ling3::kda_wo_q8_fused() && (L.wo.type == LING3_Q4_0_ROCMFP4 ||
                                         L.wo.type == LING3_Q4_0_ROCMFP4_FAST);
      if (wo_fused &&
          ling3_q8_slot_device(LING3_Q8_SLOT_GENERIC, static_cast<int>(I),
                               &wo_qs, &wo_ds) != LING3_OK) {
        if (error_msg)
          *error_msg = "ling3 ROCm: kda wo q8 slot";
        return false;
      }
      if (st->kda_scratch != nullptr &&
          st->kda_scratch_bytes >=
              ::ling3::kda_forward_scratch_bytes(ks, 1, 1)) {
        ::ling3::kda_forward_scratch(
            ks, ksc, kw, ki, ks_state,
            ::ling3::make_view(st->tmp, static_cast<int>(I), 1),
            st->kda_scratch, st->kda_scratch_bytes, stream, wo_qs, wo_ds);
      } else {
        ::ling3::kda_forward(
            ks, ksc, kw, ki, ks_state,
            ::ling3::make_view(st->tmp, static_cast<int>(I), 1), stream, wo_qs,
            wo_ds);
      }
      // Jay 2026-10-01: attn_output is [4096 -> 2560]; the input length is
      // d_inner (I), not hidden (reference.cpp:254 MatVec(k.wo,
      // attn_[0:inner])).
      if (wo_fused) {
        ling3_gemv_fp4_q8(L.wo.type, L.wo.d, LING3_Q8_SLOT_GENERIC,
                          st->attn_out, static_cast<int>(I), g.hidden);
      } else {
        ling3_gemv(L.wo.type, L.wo.d, st->tmp, st->attn_out,
                   static_cast<int>(I), g.hidden);
      }
      GpuTrace(st, static_cast<std::uint32_t>(li), "kda_gated", st->tmp, I);
      GpuTrace(st, static_cast<std::uint32_t>(li), "kda_out", st->attn_out, H);
    }

    // LS2 PHASE7: residual add + the norm that consumes it are one kernel.
    // ffn_in is still written -- the second residual add reads it and the trace
    // hook prints it as ffn_inp.
    ::ling3::ops_add_rms_norm(st->x, st->attn_out, L.ffn_norm, st->t,
                              st->ffn_in, g.hidden, 1, g.eps, stream);
    GpuTrace(st, static_cast<std::uint32_t>(li), "ffn_inp", st->ffn_in, H);
    GpuTrace(st, static_cast<std::uint32_t>(li), "ffn_norm", st->t, H);

    // ---- feed forward ----
    // LS2 PHASE7 devzero: removed. Redundant -- ling3_expert_ffn hipMemset()s y
    // at entry, and it is a blocking pageable H2D on every layer (~40/token).
    if (L.dense) {
      // RV25 wave1 F2/F3. This is an n_used == 1, w == 1.0 expert whose output
      // IS st->y. xq_ready = 0: there is no grouped call in front of it, so its
      // FFN_X quant is NOT a duplicate and has to run.
      if (ling3_dense_ffn_w1(L.ffn_gate.type, L.ffn_gate.d, L.ffn_up.d,
                             L.ffn_down.d, st->t, 1, g.hidden, g.ffn_len,
                             L.clamp_shexp, st->y, 0, stream) != LING3_OK) {
        if (error_msg)
          *error_msg = "ling3 ROCm: dense FFN (w1)";
        return false;
      }
    } else {
      // ling3_route writes DEVICE ids and weights. LS1 phase 3: when the
      // experts are resident the grouped FFN reads them WHERE THEY ARE, so the
      // two blocking D2H copies that used to sit here -- 40 GPU syncs per
      // token, each draining the pipeline -- are gone from this branch. The
      // streaming branch still needs host ids, because it stages one expert at
      // a time, so it keeps the readback.
      ling3_route(st->t, static_cast<const float*>(L.router.d), L.probs_bias,
                  g.hidden, g.n_expert, g.n_used, g.n_groups, g.n_group_used,
                  g.w_scale, g.norm_w, g_dev.d_ids, g_dev.d_weights);
      // SPEC2: the router's top-k expert ids and the weights it produced, so
      // the fork's ffn_moe_topk / ffn_moe_weights_scaled can be compared per
      // layer.
      GpuTraceIds(st, static_cast<std::uint32_t>(li), "ffn_moe_topk",
                  g_dev.d_ids, static_cast<std::size_t>(g.n_used));
      GpuTrace(st, static_cast<std::uint32_t>(li), "ffn_moe_weights_scaled",
               g_dev.d_weights, static_cast<std::size_t>(g.n_used));
      if (!g_dev.stream_experts) {
        // All 512 are resident: ONE grouped call covers the whole selected set,
        // reading the ids and weights the router just wrote on the device. 3
        // launches, 0 syncs, and no status readback -- ling3_moe_dev_check()
        // below does that once for the whole token.
        const Device::ResidentExperts& re = g_dev.experts[expert_slot];
        if (ling3_expert_ffn_dev(re.type, re.gate, re.up, re.down, st->t,
                                 g_dev.d_ids, g_dev.d_weights, g.n_used,
                                 g.n_expert, g.hidden, g.expert_ff, L.clamp_exp,
                                 st->y, stream) != LING3_OK) {
          if (error_msg)
            *error_msg = "ling3 ROCm: grouped expert FFN";
          return false;
        }
      } else {
        int ids[kMaxExperts];
        float wts[kMaxExperts];
        if (hipMemcpy(ids, g_dev.d_ids, sizeof(int) * kMaxExperts,
                      hipMemcpyDeviceToHost) != hipSuccess ||
            hipMemcpy(wts, g_dev.d_weights, sizeof(float) * kMaxExperts,
                      hipMemcpyDeviceToHost) != hipSuccess) {
          if (error_msg)
            *error_msg = "ling3 ROCm: router readback";
          return false;
        }
        for (int j = 0; j < g.n_used; ++j) {
          const int e = ids[j];
          if (e < 0 || e >= g.n_expert)
            continue;
          // Stage this expert's three matrices; all 512 are not resident.
          DevCopy(g_dev.s_gate,
                  static_cast<const std::uint8_t*>(L.host_gate_exps) +
                      static_cast<std::size_t>(e) * g.expert_ff * L.rb_k,
                  g_dev.sb_gate);
          DevCopy(g_dev.s_up,
                  static_cast<const std::uint8_t*>(L.host_up_exps) +
                      static_cast<std::size_t>(e) * g.expert_ff * L.rb_k,
                  g_dev.sb_up);
          DevCopy(g_dev.s_down,
                  static_cast<const std::uint8_t*>(L.host_down_exps) +
                      static_cast<std::size_t>(e) * g.hidden * L.rb_ff,
                  g_dev.sb_down);
          const int id0 = 0;  // the staged expert sits at offset 0
          const float w0 = wts[j];
          ling3_expert_ffn(L.ex_type, g_dev.s_gate, g_dev.s_up, g_dev.s_down,
                           st->t, &id0, &w0, 1, 1, 1, g.hidden, g.expert_ff,
                           L.clamp_exp, st->y, stream);
        }
      }
      ++expert_slot;
      // The shared expert is never routed and never scaled.
      // Jay 2026-10-01: LX's ling3_expert_ffn (and ling3_dense_ffn, which wraps
      // it) OVERWRITES y (hipMemset at entry), so writing the shared expert
      // into st->y wiped the routed experts' sum -- the GPU trace matched the
      // CPU oracle through block 2's ffn_norm and then ffn_out == ffn_shexp
      // alone. Shared expert -> st->f (d_inner >= hidden floats, idle in the
      // FFN), then add.
      //
      // RV25 wave1 F2/F3 (verified against the code, not taken on trust):
      //   F2 -- ling3_expert_ffn_dev above quantised THIS SAME st->t, of this
      //   same
      //        length, into THIS SAME LING3_Q8_SLOT_FFN_X
      //        (ling3_gemv_q8.hip:812), and everything it ran after that only
      //        READ the slot (the grouped gate/up) or wrote a different one
      //        (LS1_SLOT_MOE_H). So the quant ling3_dense_ffn used to do here
      //        rewrote identical bytes over themselves -> xq_ready = 1, one
      //        launch gone per MoE layer.
      //   F3 -- the memset + axpy of the n_used == 1 path are dropped; the down
      //        projection writes st->f itself as f[i] = 0.0f + acc, bitwise
      //        what memset(f,0) + k_axpy(1.0f, p, f) produced. k_add below is
      //        untouched (regrouping (a+b)+c would NOT be bit-identical -- RV25
      //        §5).
      if (ling3_dense_ffn_w1(L.ffn_gate.type, L.ffn_gate.d, L.ffn_up.d,
                             L.ffn_down.d, st->t, 1, g.hidden, g.expert_ff,
                             L.clamp_shexp, st->f, 1, stream) != LING3_OK) {
        if (error_msg)
          *error_msg = "ling3 ROCm: shared expert FFN (w1)";
        return false;
      }
      ling3_vec_add(st->y, st->f, st->y, g.hidden, stream);
    }

    // RV25 wave1 F4: residual#2 fused with the norm that consumes it. The
    // add_rms_norm_kernel computes the same xi = a[i] + b[i] in the same
    // thread-strided order, writes it to rsum (which IS the st->x the next
    // block reads, so the GpuTrace `l_out` tensor is unchanged) and normalises
    // from it. Bit-identical to ops_residual_add followed by ops_rms_norm --
    // the LS2 PHASE7 argument, applied one step further along. The last layer
    // fuses with the FINAL norm instead. ops_final_norm hardcodes LING3_RMS_EPS
    // (it does not take g.eps), so the fused tail must too.
    if (li + 1 < n_layers) {
      ::ling3::ops_add_rms_norm(st->ffn_in, st->y,
                                g_dev.layers[li + 1].attn_norm, st->normed,
                                st->x, g.hidden, 1, g.eps, stream);
    } else {
      ::ling3::ops_add_rms_norm(st->ffn_in, st->y, g_dev.output_norm, st->t,
                                st->x, g.hidden, 1, ::ling3::LING3_RMS_EPS,
                                stream);
    }
    GpuTrace(st, static_cast<std::uint32_t>(li), "ffn_out", st->y, H);
    GpuTrace(st, static_cast<std::uint32_t>(li), "l_out", st->x, H);
  }

  // LS1 phase 3: the grouped expert FFN leaves its status on the device and
  // never reads it back per layer -- that readback is a GPU drain, and 40 of
  // them is what this phase is about. One blocking read here covers every MoE
  // layer of the token.
  //
  // RV25 wave1 F5: that read is a DEBUG check, not a correctness dependency --
  // the status words are only ever written on error (k_route also stamps
  // LING3_OK), and every consumer of the FFN output is downstream of here. So
  // it runs once per LING3_MOE_DEV_CHECK_EVERY decode tokens instead of once
  // per token (1 = today's behaviour, which is also what a
  // missing/empty/zero/negative value means). A sticky error survives the
  // skipped tokens -- the clear is exactly what would drop it -- and is
  // reported at the next check.
  {
    static int every = -1;
    if (every < 0) {
      const char* e = std::getenv("LING3_MOE_DEV_CHECK_EVERY");
      every = ((e != nullptr && e[0] != '\0') ? std::atoi(e) : 64);
      if (every < 1)
        every = 1;
    }
    const std::uint64_t nth = st->step_n++;
    if (every == 1 || (nth % static_cast<std::uint64_t>(every)) == 0) {
      if (ling3_moe_dev_check() != LING3_OK) {
        if (error_msg)
          *error_msg = "ling3 ROCm: grouped expert FFN reported an error";
        return false;
      }
    }
  }

  // RV25 wave1 F4: the final norm is fused into the last layer's residual#2
  // (see the end of the layer loop); it wrote st->t there.
  GpuTrace(st, static_cast<std::uint32_t>(-1), "result_norm_input", st->x, H);
  ling3_gemv(g_dev.output_type, g_dev.output, st->t, st->logits, g.hidden,
             g.vocab);
  // RV25 wave1 F1: the device argmax pair and its blocking 4-byte D2H are gone.
  // The result was `(void)next`-discarded -- sampling is the host
  // Session::Argmax over the logits this step copies out (engine.cpp), never
  // this value. The kernels themselves are still used by StepMultiMultiRow's
  // MTP verify, so only this call site changed. Bit-identical: nothing
  // downstream read the answer. st->tok_out is left allocated (4 bytes, freed
  // by DestroyState) rather than reshaping NewState for no gain.

  if (out_logits.size() >= V &&
      hipMemcpy(out_logits.data(), st->logits, V * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) {
    if (error_msg)
      *error_msg = "ling3 ROCm: logits readback";
    return false;
  }
  ++st->pos;
  return true;
}

// ---------------------------------------------------------------------------
// LS3: batched prefill.
//
// Same per-block shape as Step -- rms_norm -> (KDA | MLA) -> residual ->
// rms_norm -> (dense | MoE) -> residual -- with every projection on the tiled
// grouped GEMM, kda_forward / mla_block at n_tokens = n (their own prefill
// paths: the KDA recurrence stays token-sequential inside the kernel, MLA
// masks causally over the chunk), and the MoE routed for all n tokens in one
// batched router call, grouped per expert into three GEMM launches per layer.
// Step() above is untouched (LS2 owns it).
//
// Numerics vs the scalar path: the projections change accumulation shape
// (tiled k-ascending vs LX's block-wide tree), and the routed-expert sum is
// an atomicAdd in completion order rather than slot order -- both are
// quantisation-noise-level reorderings of the same arithmetic, gated by the
// T=1 and teacher-forcing comparisons in tools/ling3_batch_test.cpp.
// ---------------------------------------------------------------------------
bool ForwardBatch(const std::int32_t* tokens, int n, State* st,
                  std::span<float> out_logits, std::string* error_msg,
                  int logit_rows, std::span<float> out_hidden,
                  std::span<float> out_hidden_rows) {
  const auto fail = [&](std::string m) {
    if (error_msg != nullptr) {
      *error_msg = "ling3 ROCm ForwardBatch: " + m;
    }
    return false;
  };
  if (!g.ready || st == nullptr || tokens == nullptr || n <= 0) {
    return fail("no state or empty batch");
  }
  if (g_dev.stream_experts) {
    return fail("streamed-expert loads have no batched path");
  }
  if (static_cast<std::size_t>(st->pos) + static_cast<std::size_t>(n) >
      static_cast<std::size_t>(st->max_ctx)) {
    return fail("context length exceeded");
  }
  const std::size_t H = static_cast<std::size_t>(g.hidden);
  const std::size_t I = static_cast<std::size_t>(g.inner);
  const std::size_t G = static_cast<std::size_t>(g.heads);
  const std::size_t V = static_cast<std::size_t>(g.vocab);
  // LS4: logit_rows > 0 asks for the logits of the FIRST logit_rows rows into
  // the small logits_spec buffer. 0 keeps LS3's two original modes exactly.
  int spec_rows = 0;
  if (logit_rows > 0) {
    if (logit_rows > n || logit_rows > kSpeculativeRows) {
      return fail("logit_rows " + std::to_string(logit_rows) + " exceeds n (" +
                  std::to_string(n) + ") or the " +
                  std::to_string(kSpeculativeRows) + "-row cap");
    }
    if (out_logits.size() < static_cast<std::size_t>(logit_rows) * V) {
      return fail("logits span too small for logit_rows");
    }
    spec_rows = logit_rows;
  }
  const bool want_all =
      spec_rows == 0 && out_logits.size() >= static_cast<std::size_t>(n) * V;
  if (spec_rows == 0 && !want_all && out_logits.size() < V) {
    return fail("logits span too small");
  }
  // LS4 phase 3: per-row hidden, so the MTP head can be primed over the prompt.
  int hidden_rows = 0;
  if (!out_hidden_rows.empty()) {
    if (out_hidden_rows.size() % H != 0) {
      return fail("out_hidden_rows must be a whole number of hidden rows");
    }
    hidden_rows = static_cast<int>(out_hidden_rows.size() / H);
    if (hidden_rows > n) {
      return fail("out_hidden_rows asks for more rows than the batch has");
    }
  }
  const bool want_hidden = !out_hidden.empty();
  if (want_hidden && out_hidden.size() != H) {
    return fail("out_hidden must be exactly one row of hidden floats, got " +
                std::to_string(out_hidden.size()));
  }

  if (st->batch == nullptr) {
    st->batch = NewBatchBufs(st->max_ctx);
  }
  BatchBufs* B = static_cast<BatchBufs*>(st->batch);
  if (B == nullptr) {
    return fail("cannot allocate batch scratch");
  }
  if (n > B->cap) {
    return fail("batch of " + std::to_string(n) + " exceeds the chunk cap " +
                std::to_string(B->cap));
  }
  if (want_all && B->logits_all == nullptr) {
    return fail(
        "all-position logits need LING3_BATCH_ALL_LOGITS=1 at state creation");
  }
  if (spec_rows > 0 && B->logits_spec == nullptr) {
    return fail("the per-row logits buffer is missing");
  }

  hipStream_t stream = nullptr;  // the null stream, like Step
  const int pos0 = st->pos;
  const int tok_n = n;

  const bool prof = BatchProfile();
  BatchProfiler P;
  BatchSection cur_sec = kSecEmbed;
  const auto sec = [&](BatchSection s) {
    cur_sec = s;
    if (prof) {
      (void)hipDeviceSynchronize();
      P.begin();
    }
  };
  const auto eend = [&]() {
    if (prof) {
      (void)hipDeviceSynchronize();
      P.ms[cur_sec] += P.end();
    }
  };

  // Token ids and positions: chunk-sized device buffers (the 1-int Step
  // buffers must never be handed n tokens -- see the H1 fault note above).
  std::vector<int> htok(static_cast<std::size_t>(n));
  std::vector<int> hpos(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    htok[static_cast<std::size_t>(i)] = static_cast<int>(tokens[i]);
    hpos[static_cast<std::size_t>(i)] = pos0 + i;
  }
  if (hipMemcpy(B->tok, htok.data(), static_cast<std::size_t>(n) * sizeof(int),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(B->pos, hpos.data(), static_cast<std::size_t>(n) * sizeof(int),
                hipMemcpyHostToDevice) != hipSuccess) {
    return fail("token/position upload");
  }

  sec(kSecEmbed);
  ::ling3::ops_embedding_row(g_dev.token_embd_type, g_dev.token_embd, g.hidden,
                             g.vocab, B->tok, n, B->x, stream);
  eend();

  gufo::models::ling3::vision::OnHidden(B->x, pos0, n, static_cast<int>(H),
                                        stream);

  // LING3_GPU_TRACE hook: same tensor names as Step, taken at the traced
  // position's row, so trace.sh diffs against the CPU oracle unchanged.
  const long tpos = GpuTracePos();
  const long trow =
      (tpos >= static_cast<long>(pos0) && tpos < static_cast<long>(pos0) + n)
          ? tpos - static_cast<long>(pos0)
          : -1L;
  const auto TR = [&](std::uint32_t block, const char* name, const void* d,
                      std::size_t cnt) {
    if (trow < 0 || d == nullptr || cnt == 0) {
      return;
    }
    (void)hipDeviceSynchronize();
    std::vector<float> h(cnt);
    if (hipMemcpy(h.data(), d, cnt * sizeof(float), hipMemcpyDeviceToHost) !=
        hipSuccess) {
      std::printf("LG-TRACE %u %s COPY-FAILED\n", block, name);
      return;
    }
    std::printf("LG-TRACE %u %s", block, name);
    const std::size_t lead = cnt < 3 ? cnt : 3;
    for (std::size_t r = 0; r < lead; ++r) {
      std::printf(" %.6f", static_cast<double>(h[r]));
    }
    if (cnt > 6) {
      for (std::size_t r = cnt - 3; r < cnt; ++r) {
        std::printf(" %.6f", static_cast<double>(h[r]));
      }
    }
    std::printf("\n");
    // LQ3: a whole-row fingerprint on its OWN line (never on the LG-TRACE line
    // -- trace.sh parses f[3:] of that as floats and a stray field would make
    // it count the tensor as bad). first3/last3 can miss a difference in the
    // middle of a 2560-wide row; these change if ANY element of the row does.
    // Same summation order both runs, so two runs are comparable.
    double s = 0.0, s2 = 0.0;
    std::size_t nn = 0;
    for (std::size_t r = 0; r < cnt; ++r) {
      const double v = static_cast<double>(h[r]);
      if (v != v) {
        ++nn;
        continue;
      }
      s += v;
      s2 += v * v;
    }
    std::printf("LG-FP %u %s n=%zu sum=%.9g sumsq=%.9g nan=%zu\n", block, name,
                cnt, s, s2, nn);
    std::fflush(stdout);
  };
  const std::size_t tr = static_cast<std::size_t>(trow < 0 ? 0 : trow);
  TR(static_cast<std::uint32_t>(-1), "inp_embd", B->x + tr * H, H);

  const ::ling3::KdaShape ks = MakeKdaShape();
  const ::ling3::KdaScalars ksc;
  const ::ling3::MlaShape ms = MakeMlaShape();
  const ::ling3::MlaScalars msc;

  // MoE routing readback + expert grouping scratch, reused across layers.
  std::vector<int> hid(static_cast<std::size_t>(n) * kMaxExperts);
  std::vector<float> hw(static_cast<std::size_t>(n) * kMaxExperts);
  std::vector<int> hrc(static_cast<std::size_t>(n));
  std::vector<int> hslot_tok(static_cast<std::size_t>(n) * kMaxExperts);
  std::vector<float> hslot_w(static_cast<std::size_t>(n) * kMaxExperts);
  std::vector<int> hslot_pos(static_cast<std::size_t>(n) * kMaxExperts);
  std::vector<int> counts(static_cast<std::size_t>(g.n_expert) + 1);
  std::vector<int> fill(static_cast<std::size_t>(g.n_expert));
  std::vector<int> hgrp_e, hgrp_off, hgrp_cnt;
  // P3: 16-token-tile prefix (compacted grid + WMMA activation tile
  // numbering) and the per-slot quantiser destination (tile<<4|lane).
  std::vector<int> hgrp_zp, hslot_dest;
  int moe_max_cnt = 0, moe_big = 0, moe_groups = 0, moe_tiles = 0;

  // P3 step 3: dense W8A8 WMMA GEMM with its own quantisation of x into the
  // given tile scratch. Falls back to the phase-2 float kernel when the
  // knob is off, the tensor is not ROCmFP4, or the row count is below the
  // WMMA block height (128) — the float path is already fine at that size.
  const auto gemm_dense_batch = [&](int type, const void* w, const float* x,
                                    float* y, int k, int n_rows,
                                    void* q8) -> bool {
    if (PrefillW8A8() && n_rows >= 128 && (k % 32) == 0 &&
        (type == LING3_Q4_0_ROCMFP4 || type == LING3_Q4_0_ROCMFP4_FAST) &&
        ling3_wmma_quant(x, nullptr, nullptr, q8, n, k) == LING3_OK) {
      return ling3_wmma_gemm_dense(type, w, q8, y, k, n_rows, n) == LING3_OK;
    }
    return ling3_gemm_fast(type, w, x, y, k, n_rows, n) == LING3_OK;
  };

  // Dense SwiGLU FFN: t [n x hidden] -> y [n x hidden]. On the WMMA path the
  // gate/up quantisation of B->t is shared by passing the pre-quantised
  // tiles in via the q8_hidden scratch the caller filled for projections
  // (the projection reads are done by then), so only the down input
  // quantises here.
  const auto dense_ffn_batch = [&](int gt, const void* wg, const void* wu,
                                   const void* wd, int n_ff, float limit,
                                   float* y) {
    const bool t_w8 =
        PrefillW8A8() && n_ff >= 128 && (n_ff % 32) == 0 &&
        (gt == LING3_Q4_0_ROCMFP4 || gt == LING3_Q4_0_ROCMFP4_FAST) &&
        ling3_wmma_quant(B->t, nullptr, nullptr, B->q8_hidden, n, g.hidden) ==
            LING3_OK;
    const bool gu_ok =
        t_w8 ? (ling3_wmma_gemm_dense(gt, wg, B->q8_hidden, B->sg, g.hidden,
                                      n_ff, n) == LING3_OK &&
                ling3_wmma_gemm_dense(gt, wu, B->q8_hidden, B->su, g.hidden,
                                      n_ff, n) == LING3_OK)
             : (ling3_gemm_fast(gt, wg, B->t, B->sg, g.hidden, n_ff, n) ==
                    LING3_OK &&
                ling3_gemm_fast(gt, wu, B->t, B->su, g.hidden, n_ff, n) ==
                    LING3_OK);
    return gu_ok &&
           ling3_swiglu(B->sg, B->su, B->sh,
                        static_cast<int>(static_cast<std::size_t>(n) *
                                         static_cast<std::size_t>(n_ff)),
                        limit) == LING3_OK &&
           (t_w8 ? (ling3_wmma_quant(B->sh, nullptr, nullptr, B->q8_ffn, n,
                                     n_ff) == LING3_OK &&
                    ling3_wmma_gemm_dense(gt, wd, B->q8_ffn, y, n_ff, g.hidden,
                                          n) == LING3_OK)
                 : ling3_gemm_fast(gt, wd, B->sh, y, n_ff, g.hidden, n) ==
                       LING3_OK);
  };

  std::size_t expert_slot = 0;
  for (std::size_t li = 0; li < g_dev.layers.size(); ++li) {
    Layer& L = g_dev.layers[li];

    sec(kSecAttnNorm);
    ::ling3::ops_rms_norm(B->x, L.attn_norm, B->normed, g.hidden, n, g.eps,
                          stream);
    eend();
    TR(static_cast<std::uint32_t>(li), "attn_norm", B->normed + tr * H, H);

    // P3 step 3: one quantisation of B->normed serves every attention
    // projection of this layer (up to nine GEMMs). Tensors that are not
    // ROCmFP4 or rows below the WMMA block height stay on the float path,
    // which reads the F32 rows directly.
    const bool normed_w8 =
        PrefillW8A8() && (g.hidden % 32) == 0 &&
        ling3_wmma_quant(B->normed, nullptr, nullptr, B->q8_hidden, n,
                         g.hidden) == LING3_OK;
    const auto proj_gemm = [&](int type, const void* w, float* y,
                               int n_rows) -> bool {
      if (normed_w8 && n_rows >= 128 &&
          (type == LING3_Q4_0_ROCMFP4 || type == LING3_Q4_0_ROCMFP4_FAST)) {
        return ling3_wmma_gemm_dense(type, w, B->q8_hidden, y, g.hidden, n_rows,
                                     n) == LING3_OK;
      }
      return ling3_gemm_fast(type, w, B->normed, y, g.hidden, n_rows, n) ==
             LING3_OK;
    };

    if (g.is_mla[li]) {
      const std::size_t q_dim = static_cast<std::size_t>(ms.q_dim());
      sec(kSecMlaProj);
      const bool mla_proj_ok =
          proj_gemm(L.mq.type, L.mq.d, B->mq, static_cast<int>(q_dim)) &&
          proj_gemm(L.m_kv_a_mqa.type, L.m_kv_a_mqa.d, B->kva,
                    static_cast<int>(ms.kv_a())) &&
          proj_gemm(L.m_gate.type, L.m_gate.d, B->gate, static_cast<int>(G));
      eend();
      if (!mla_proj_ok) {
        return fail("MLA projection gemm at layer " + std::to_string(li));
      }

      ::ling3::MlaInputs in;
      in.q = ::ling3::make_view(B->mq, static_cast<int>(q_dim), n);
      in.kv_a = ::ling3::make_view(B->kva, ms.kv_a(), n);
      in.gate = ::ling3::make_view(B->gate, static_cast<int>(G), n);
      in.n_tokens = n;
      in.n_seqs = 1;

      ::ling3::MlaCache cache;
      // LG1 W3: reindex mla_off through the MLA ordinal map. All three
      // construction sites are inside `if (g.is_mla[li])`, so mla_ord[li] >= 0
      // here; the abort is a safety net for any future invariant break.
      if (g.mla_ord[li] < 0) {
        std::fprintf(stderr, "lg1: mla_ord[%zu] < 0 at Step MlaCache build\n",
                     li);
        std::abort();
      }
      cache.kv = static_cast<float*>(st->mla_cache_device) +
                 st->mla_off[g.mla_ord[li]];
      cache.s_pos = ms.kv_a();
      cache.s_seq = 0;
      cache.max_pos = static_cast<int>(st->max_ctx);

      ::ling3::MlaWeights mw;
      mw.wk_b = L.m_wk_b_f32;
      mw.wv_b = L.m_wv_b_f32;
      mw.wk_bT = L.m_wk_bT;
      mw.wv_bT = L.m_wv_bT;
      mw.kv_a_norm = L.m_kv_a_norm;

      // B->pos already holds pos0..pos0+n-1 on the device.
      sec(kSecMla);
      {
        // LS3: the same four stages as mla_block, but the attention runs
        // token-parallel (mla_attn_tok) in sub-batches of 64 -- mla_attn's
        // serial token loop left the whole chunk on 32 blocks (profile:
        // 0.75-1.6 s per 512-token chunk, growing with position).
        ::ling3::mla_rope(ms, const_cast<float*>(in.q.p) + ms.nope,
                          /*s_head=*/ms.head_k, ms.n_head, n, in.q.s_tok,
                          in.q.s_seq, 1, B->pos, stream);
        ::ling3::mla_rope(ms, const_cast<float*>(in.kv_a.p) + ms.kv,
                          /*s_head=*/0, /*n_head=*/1, n, in.kv_a.s_tok,
                          in.kv_a.s_seq, 1, B->pos, stream);
        ::ling3::mla_latent_norm(ms, msc, mw, const_cast<float*>(in.kv_a.p), n,
                                 1, in.kv_a.s_tok, in.kv_a.s_seq, stream);
        ::ling3::mla_cache_append(ms, cache, in.kv_a.p, n, in.kv_a.s_tok, 1,
                                  pos0 + n, stream);
        // LS3 P2 diagnostic knob: LING3_MLA_SERIAL=1 runs the attention through
        // the proven serial mla_attn instead of the tiled mla_attn_tok, so a
        // gate failure bisects between the two kernels without a rebuild.
        static const bool mla_serial =
            std::getenv("LING3_MLA_SERIAL") != nullptr;
        if (mla_serial) {
          ::ling3::mla_attn(ms, msc, mw, cache, pos0 + n, in,
                            ::ling3::make_view(B->mlaout, ms.o_dim(), n),
                            B->mla_scores, stream);
        } else if (MlaFw() && static_cast<int>(pos0 + n) >= MlaFwMin() &&
                   (MlaFwMax() == 0 ||
                    static_cast<int>(pos0 + n) <= MlaFwMax()) &&
                   ::ling3::mla_w2_prep(ms, cache, pos0 + n, B->w2, stream) &&
                   ::ling3::mla_fw_attn(
                       ms, msc, mw, pos0 + n, in,
                       ::ling3::make_view(B->mlaout, ms.o_dim(), n), B->w2,
                       B->fw, /*tok_base=*/0, /*tok_count=*/n,
                       /*chunk_tokens=*/n, stream)) {
          // LX1: the fused WMMA flash kernel handled the whole chunk —
          // prep stages the f16 cache once per layer per chunk (shared
          // with the w2 path), then absorb -> flash -> de-absorb/gate, one
          // attention launch, no materialized S/P. Any refusal in the
          // condition falls through to the proven paths below.
        } else if (MlaTile() && MlaTileFits(pos0 + n)) {
          // LF1 SPEC3: the ported fork tile kernel. One attention launch for
          // the whole chunk (its f16 cache conversion and the de-absorb GEMM
          // run inside the same call), tok_base 0 / tok_count n / chunk n.
          ::ling3::mla_attn_tile(ms, msc, mw, cache, pos0 + n, in,
                                 ::ling3::make_view(B->mlaout, ms.o_dim(), n),
                                 B->mla_scores, /*tok_base=*/0,
                                 /*tok_count=*/n, /*chunk_tokens=*/n, stream);
        } else {
          constexpr int kMlaTokBatch = kMlaTokBatchCap;
          // LW1 (wave 2): the two-pass WMMA attention behind
          // LING3_MLA_WMMA2=1 (default OFF). prep stages the f16 cache once
          // per layer per chunk; per sub-batch, any audit refusal falls back
          // to mla_attn_tok (fail-closed). Sub-batches are independent, so a
          // mixed chunk stays correct.
          static const bool mla_w2 = std::getenv("LING3_MLA_WMMA2") != nullptr;
          // LW1: below ~4K positions the six-kernel chain is latency-bound and
          // loses to mla_attn_tok end to end (measured on this box: 1.6K 338
          // vs 393 tok/s, 3K 340 vs ~402; from 6.5K up w2 wins: 311 vs 294,
          // 262 vs 188 at 13.8K, 200 vs 112 at 27.7K). The spec's "path used
          // only above a measured threshold": chunks ending under 4096
          // positions take the proven fallback.
          const bool w2_ok =
              mla_w2 && pos0 + n >= 4096 &&
              ::ling3::mla_w2_prep(ms, cache, pos0 + n, B->w2, stream);
          for (int tb = 0; tb < n; tb += kMlaTokBatch) {
            const int tc = n - tb < kMlaTokBatch ? n - tb : kMlaTokBatch;
            if (w2_ok && ::ling3::mla_w2_attn(
                             ms, msc, mw, pos0 + n, in,
                             ::ling3::make_view(B->mlaout, ms.o_dim(), n),
                             B->mla_scores, B->w2, tb, tc, n, stream)) {
              continue;
            }
            ::ling3::mla_attn_tok(ms, msc, mw, cache, pos0 + n, in,
                                  ::ling3::make_view(B->mlaout, ms.o_dim(), n),
                                  B->mla_scores, tb, tc, n, stream);
          }
        }
      }
      eend();
      sec(kSecAttnOut);
      const bool mla_out_ok =
          gemm_dense_batch(L.m_wo.type, L.m_wo.d, B->mlaout, B->attn,
                           static_cast<int>(ms.o_dim()), g.hidden, B->q8_odim);
      eend();
      if (!mla_out_ok) {
        return fail("MLA output gemm at layer " + std::to_string(li));
      }
      TR(static_cast<std::uint32_t>(li), "mla_q", B->mq + tr * q_dim, q_dim);
      TR(static_cast<std::uint32_t>(li), "mla_attn_out",
         B->mlaout + tr * static_cast<std::size_t>(ms.o_dim()),
         static_cast<std::size_t>(ms.o_dim()));
      TR(static_cast<std::uint32_t>(li), "attn_out", B->attn + tr * H, H);
    } else {
      sec(kSecKdaProj);
      const bool kda_proj_ok =
          proj_gemm(L.wq.type, L.wq.d, B->q, static_cast<int>(I)) &&
          proj_gemm(L.wk.type, L.wk.d, B->kk, static_cast<int>(I)) &&
          proj_gemm(L.wv.type, L.wv.d, B->vv, static_cast<int>(I)) &&
          proj_gemm(L.wf.type, L.wf.d, B->fp, static_cast<int>(I)) &&
          proj_gemm(L.wg.type, L.wg.d, B->g2, static_cast<int>(I)) &&
          proj_gemm(L.wbeta.type, L.wbeta.d, B->beta, static_cast<int>(G));
      eend();
      if (!kda_proj_ok) {
        return fail("KDA projection gemm at layer " + std::to_string(li));
      }

      ::ling3::KdaWeights kw;
      kw.conv_wq = L.conv_q;
      kw.conv_wk = L.conv_k;
      kw.conv_wv = L.conv_v;
      kw.a_exp = L.a_exp;
      kw.dt_bias = L.dt_bias;
      kw.o_norm = L.o_norm;

      ::ling3::KdaInputs ki;
      ki.q = ::ling3::make_view(B->q, static_cast<int>(I), n);
      ki.k = ::ling3::make_view(B->kk, static_cast<int>(I), n);
      ki.v = ::ling3::make_view(B->vv, static_cast<int>(I), n);
      ki.f = ::ling3::make_view(B->fp, static_cast<int>(I), n);
      ki.g2 = ::ling3::make_view(B->g2, static_cast<int>(I), n);
      ki.beta = ::ling3::make_view(B->beta, static_cast<int>(G), n);
      ki.n_tokens = n;
      ki.n_seqs = 1;

      ::ling3::KdaState kst;
      kst.conv_state =
          static_cast<float*>(st->kda_conv_device) + st->kda_conv_off[li];
      kst.ssm_state =
          static_cast<float*>(st->kda_ssm_device) + st->kda_ssm_off[li];
      kst.n_seqs = 1;

      sec(kSecKda);
      ::ling3::kda_forward_scratch(
          ks, ksc, kw, ki, kst,
          ::ling3::make_view(B->kdaout, static_cast<int>(I), n), B->kda_scratch,
          B->kda_scratch_bytes, stream);
      eend();
      sec(kSecAttnOut);
      const bool kda_out_ok =
          gemm_dense_batch(L.wo.type, L.wo.d, B->kdaout, B->attn,
                           static_cast<int>(I), g.hidden, B->q8_odim);
      eend();
      if (!kda_out_ok) {
        return fail("KDA output gemm at layer " + std::to_string(li));
      }
      TR(static_cast<std::uint32_t>(li), "kda_gated", B->kdaout + tr * I, I);
      TR(static_cast<std::uint32_t>(li), "kda_out", B->attn + tr * H, H);
    }

    sec(kSecFfnNorm);
    ::ling3::ops_residual_add(B->x, B->attn, B->ffn_in, g.hidden, n, stream);
    ::ling3::ops_rms_norm(B->ffn_in, L.ffn_norm, B->t, g.hidden, n, g.eps,
                          stream);
    eend();
    TR(static_cast<std::uint32_t>(li), "ffn_inp", B->ffn_in + tr * H, H);
    TR(static_cast<std::uint32_t>(li), "ffn_norm", B->t + tr * H, H);

    // ---- feed forward ----
    if (L.dense) {
      sec(kSecShared);
      const bool dense_ok =
          dense_ffn_batch(L.ffn_gate.type, L.ffn_gate.d, L.ffn_up.d,
                          L.ffn_down.d, g.ffn_len, L.clamp_shexp, B->y);
      eend();
      if (!dense_ok) {
        return fail("dense FFN gemm at layer " + std::to_string(li));
      }
    } else {
      // Route all n tokens at once, read the results back once.
      sec(kSecRoute);
      const bool route_ok =
          ling3_route_batch(B->t, static_cast<const float*>(L.router.d),
                            L.probs_bias, g.hidden, n, g.n_expert, g.n_used,
                            g.n_groups, g.n_group_used, g.w_scale, g.norm_w,
                            B->ids, B->wts, B->rc) == LING3_OK;
      eend();
      if (!route_ok) {
        return fail("router at layer " + std::to_string(li));
      }
      const std::size_t slots = static_cast<std::size_t>(n) * kMaxExperts;
      // LS1 PHASE8 item 2: the grouping is now done ON THE DEVICE, in the same
      // order, by the same algorithm -- see kernels/ling3_moe_group.hip. This
      // replaces three blocking D2H copies, a host counting sort with its
      // sub-group split and tile prefix, and eight H2D uploads, per MoE layer
      // per chunk. The four words the GEMM's launch config needs (n_groups,
      // max_cnt, tiles, status) come back in one 16-byte read; everything else
      // stays on the device. Stability inside an expert is preserved exactly,
      // because the device sort walks slots in the same ascending order the
      // host loop did.
      int ls1_hdr[LS1_HDR_WORDS] = {0, 0, 0, 0};
      const int grp_cap = (int)B->grp_e_cap;
      if (ling3_moe_group_batch(B->ids, B->wts, B->rc, n, kMaxExperts, g.n_used,
                                g.n_expert, kMoeGroupCap, B->slot_tok,
                                B->slot_w, B->token_slots, B->grp_e, B->grp_off,
                                B->grp_cnt, B->grp_zp, B->slot_dest, grp_cap,
                                (int)slots, ls1_hdr) != LING3_OK) {
        fprintf(stderr,
                "LS1-GROUP layer=%zu n=%d slots=%d n_expert=%d n_used=%d "
                "cap=%d grp_cap=%d\n",
                li, n, (int)slots, g.n_expert, g.n_used, (int)kMoeGroupCap,
                grp_cap);
        return fail("expert grouping on device at layer " + std::to_string(li));
      }
      if (ls1_hdr[LS1_HDR_STATUS] != LING3_OK) {
        // The old code had loud failures here for a router rc and for an expert
        // id out of range. The device kernel cannot call fail() itself, so it
        // reports through one sticky status word and this turns that back into
        // the same loud failure.
        return fail("router/grouping rejected at layer " + std::to_string(li) +
                    " (a router rc, or an expert id outside [0," +
                    std::to_string(g.n_expert) + "))");
      }
      const int n_groups = ls1_hdr[LS1_HDR_NGROUPS];
      const int max_cnt = ls1_hdr[LS1_HDR_MAXCNT];
      const int tile_acc = ls1_hdr[LS1_HDR_TILES];
      moe_max_cnt = std::max(moe_max_cnt, max_cnt);
      moe_big += max_cnt > 32 ? 1 : 0;
      moe_groups += n_groups;
      moe_tiles = std::max(moe_tiles, tile_acc);
      const Device::ResidentExperts& re = g_dev.experts[expert_slot];
      const std::size_t stride_k =
          static_cast<std::size_t>(g.expert_ff) * L.rb_k;
      const std::size_t stride_ff =
          static_cast<std::size_t>(g.hidden) * L.rb_ff;
      sec(kSecMoeGemm);
      // P3: the W8A8 int8-WMMA path (gufo's kernel skeleton, fork numerics:
      // Q8_1 activations, FP4->int8 weights, per-half block scales in the
      // epilogue). LING3_PREFILL_W8A8=0 keeps the phase-2 float tiled GEMM
      // -- now on the compacted per-group grid (same arithmetic, so both
      // paths share the determinism property; the unit gate pins them
      // bit-identical to the legacy grid).
      const bool w8a8 = PrefillW8A8() && (re.type == LING3_Q4_0_ROCMFP4 ||
                                          re.type == LING3_Q4_0_ROCMFP4_FAST);
      // gate / up write each slot's row at its global slot index (identity),
      // swiglu is flat over all slots, and down plain-stores its scaled
      // partials at the slot rows (P2-C); one gather-sum then folds each
      // token's 8 slots in slot order -- no atomics, deterministic, the same
      // sum order as the CPU reference.
      bool moe_ok;
      if (w8a8) {
        // One quantisation of B->t serves gate AND up (same input rows).
        moe_ok =
            ling3_wmma_quant(B->t, B->slot_tok, B->slot_dest, B->q8_t,
                             static_cast<int>(slots), g.hidden) == LING3_OK &&
            ling3_wmma_gemm_grp(re.type, re.gate, stride_k, B->grp_e,
                                B->grp_off, B->grp_cnt, B->grp_zp, n_groups,
                                tile_acc, B->q8_t, B->eg, nullptr, g.hidden,
                                g.expert_ff) == LING3_OK &&
            ling3_wmma_gemm_grp(re.type, re.up, stride_k, B->grp_e, B->grp_off,
                                B->grp_cnt, B->grp_zp, n_groups, tile_acc,
                                B->q8_t, B->eu, nullptr, g.hidden,
                                g.expert_ff) == LING3_OK &&
            ling3_swiglu(
                B->eg, B->eu, B->eh,
                static_cast<int>(slots * static_cast<std::size_t>(g.expert_ff)),
                L.clamp_exp) == LING3_OK &&
            ling3_wmma_quant(B->eh, nullptr, B->slot_dest, B->q8_h,
                             static_cast<int>(slots),
                             g.expert_ff) == LING3_OK &&
            ling3_wmma_gemm_grp(re.type, re.down, stride_ff, B->grp_e,
                                B->grp_off, B->grp_cnt, B->grp_zp, n_groups,
                                tile_acc, B->q8_h, B->y2, B->slot_w,
                                g.expert_ff, g.hidden) == LING3_OK &&
            ling3_slot_gather_sum(B->y2, B->token_slots, B->y, kMaxExperts,
                                  g.hidden, n) == LING3_OK;
      } else {
        moe_ok =
            ling3_gemm_grp(re.type, re.gate, stride_k, B->grp_e, B->grp_off,
                           B->grp_cnt, n_groups, B->t, B->slot_tok, B->eg,
                           B->ident, nullptr, 0, g.hidden, g.expert_ff, max_cnt,
                           B->grp_zp, tile_acc) == LING3_OK &&
            ling3_gemm_grp(re.type, re.up, stride_k, B->grp_e, B->grp_off,
                           B->grp_cnt, n_groups, B->t, B->slot_tok, B->eu,
                           B->ident, nullptr, 0, g.hidden, g.expert_ff, max_cnt,
                           B->grp_zp, tile_acc) == LING3_OK &&
            ling3_swiglu(
                B->eg, B->eu, B->eh,
                static_cast<int>(slots * static_cast<std::size_t>(g.expert_ff)),
                L.clamp_exp) == LING3_OK &&
            ling3_gemm_grp(re.type, re.down, stride_ff, B->grp_e, B->grp_off,
                           B->grp_cnt, n_groups, B->eh, B->ident, B->y2,
                           B->ident, B->slot_w, 0, g.expert_ff, g.hidden,
                           max_cnt, B->grp_zp, tile_acc) == LING3_OK &&
            ling3_slot_gather_sum(B->y2, B->token_slots, B->y, kMaxExperts,
                                  g.hidden, n) == LING3_OK;
      }
      eend();
      if (!moe_ok) {
        return fail("routed expert gemm at layer " + std::to_string(li));
      }
      ++expert_slot;
      // The shared expert is never routed and never scaled (same as Step).
      sec(kSecShared);
      const bool shared_ok =
          dense_ffn_batch(L.ffn_gate.type, L.ffn_gate.d, L.ffn_up.d,
                          L.ffn_down.d, g.expert_ff, L.clamp_shexp, B->fsh) &&
          ling3_vec_add(B->y, B->fsh, B->y,
                        static_cast<int>(H * static_cast<std::size_t>(n))) ==
              LING3_OK;
      eend();
      if (!shared_ok) {
        return fail("shared expert gemm at layer " + std::to_string(li));
      }
    }

    sec(kSecResidual);
    ::ling3::ops_residual_add(B->ffn_in, B->y, B->x, g.hidden, n, stream);
    eend();
    TR(static_cast<std::uint32_t>(li), "ffn_out", B->y + tr * H, H);
    TR(static_cast<std::uint32_t>(li), "l_out", B->x + tr * H, H);
  }

  sec(kSecFinal);
  int norm_rows_for_hidden = 1;
  {
    // LS4: how many rows to final-norm, which logits to produce, and which
    // rows' hidden to return are three independent questions. Folding the
    // hidden request into the want_all branch made every batched prefill take
    // the chunk-sized logits_all path, whose buffer only exists under
    // LING3_BATCH_ALL_LOGITS=1, so every request 500'd.
    //
    // hidden_rows > 1 needs every row normalised, so it forces norm_rows = n.
    int norm_rows = (spec_rows > 0) ? spec_rows : (want_all) ? n : 1;
    if (hidden_rows > 1)
      norm_rows = n;
    norm_rows_for_hidden = norm_rows;
    const float* norm_src =
        (norm_rows == n) ? B->x : B->x + static_cast<std::size_t>(n - 1) * H;
    ::ling3::ops_final_norm(norm_src, g_dev.output_norm, B->t, g.hidden,
                            norm_rows, stream);
    TR(static_cast<std::uint32_t>(-1), "result_norm_input", B->x + tr * H, H);

    if (spec_rows > 0) {
      // The speculative verify: logits for rows [0, spec_rows) through the
      // small fixed-size buffer.
      if (ling3_gemm_fast(g_dev.output_type, g_dev.output, B->t, B->logits_spec,
                          g.hidden, g.vocab, spec_rows) != LING3_OK ||
          hipMemcpy(out_logits.data(), B->logits_spec,
                    static_cast<std::size_t>(spec_rows) * V * sizeof(float),
                    hipMemcpyDeviceToHost) != hipSuccess) {
        return fail("per-row logits");
      }
    } else if (want_all) {
      if (ling3_gemm_fast(g_dev.output_type, g_dev.output, B->t, B->logits_all,
                          g.hidden, g.vocab, n) != LING3_OK ||
          hipMemcpy(out_logits.data(), B->logits_all,
                    static_cast<std::size_t>(n) * V * sizeof(float),
                    hipMemcpyDeviceToHost) != hipSuccess) {
        return fail("all-position logits");
      }
    } else {
      // LS3's original path: the last position's logits. B->t holds
      // norm_rows rows, so the last one is row norm_rows-1 -- NOT row 0.
      // Reading B->t directly returned the FIRST position's logits whenever
      // anything forced norm_rows above 1 (LS4's per-row hidden), which
      // silently changed generation: the answer began with the chat
      // template's "SYSTEM" marker.
      const float* last_norm =
          B->t + static_cast<std::size_t>(norm_rows - 1) * H;
      if (ling3_gemm_fast(g_dev.output_type, g_dev.output, last_norm,
                          st->logits, g.hidden, g.vocab, 1) != LING3_OK ||
          hipMemcpy(out_logits.data(), st->logits, V * sizeof(float),
                    hipMemcpyDeviceToHost) != hipSuccess) {
        return fail("logits");
      }
    }
  }
  if (want_hidden) {
    const float* last =
        B->t + static_cast<std::size_t>(norm_rows_for_hidden - 1) * H;
    if (hipMemcpy(out_hidden.data(), last, H * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      return fail("hidden rows");
    }
  }
  if (hidden_rows > 0) {
    const std::size_t off = static_cast<std::size_t>(n - hidden_rows) * H;
    if (hipMemcpy(out_hidden_rows.data(), B->t + off,
                  static_cast<std::size_t>(hidden_rows) * H * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      return fail("hidden rows");
    }
  }
  eend();
  st->pos += tok_n;
  if (prof) {
    std::printf("LS3-PROFILE n=%d", tok_n);
    for (int s = 0; s < kSecMax; ++s) {
      std::printf(" %s=%.1f", kSecNames[s], P.ms[s]);
    }
    std::printf(" | moe_groups=%d big>32=%d max_cnt=%d tiles=%d\n", moe_groups,
                moe_big, moe_max_cnt, moe_tiles);
    std::fflush(stdout);
  }
  return true;
}

bool BatchCapable() noexcept {
  return g.ready && !g_dev.stream_experts;
}
// ---------------------------------------------------------------------------
// LS4 phase 4 -- StepMulti / StepMultiDispatch.
//
// StepMulti runs the SAME per-row op sequence Step runs, in order, with the
// same kernels and the same inputs. Byte-identity with the non-speculative
// path is therefore structural rather than measured-and-hoped-for: the
// verify cannot produce an argmax the model would not have produced,
// because it IS the model, one token at a time.
//
// Being m sequential steps, it is also no faster than m sequential steps.
// That is the honest cost today; the speed is meant to arrive at
// StepMultiDispatch, the single dispatch point every weight matmul goes
// through. When LS1's ling3_gemv_multi / ling3_expert_ffn_multi land,
// LING3_MTP_MULTI=1 routes the block through them and nothing else changes.

// LM1 SPEC4 item 1: copy the trunk's whole KDA arrays + the post-norm hidden of
// row `row` into `keep`. Used by the token-outer StepMulti; the layer-outer
// StepMultiMultiRow fills the same images one layer slice at a time instead
// (see KeepImageLayer), because that is the only point at which a given layer
// has finished row `row` while the others have not. Layer-outer counterpart:
// image ONE layer's KDA slice for row `row`. This is the only point at which
// layer `li` has finished row `row` while the later layers have not, which is
// why the verify cannot do a whole-array copy per row the way the token-outer
// StepMulti does. The MLA layers never touch their slice (it is zeroed at
// NewState and stays that way), so imaging it here is still exact.
bool KeepImageLayer(State* st, MultiRowKeep* keep, std::size_t li, int row) {
  if (keep == nullptr || st == nullptr || row < 0 || row >= keep->rows) {
    return false;
  }
  const ::ling3::KdaShape ks = MakeKdaShape();
  const std::size_t cn = static_cast<std::size_t>(ks.embd_r());
  const std::size_t sn = static_cast<std::size_t>(ks.embd_s());
  const std::size_t coff = st->kda_conv_off[li];
  const std::size_t soff = st->kda_ssm_off[li];
  const std::size_t r = static_cast<std::size_t>(row);
  // LM1 00:05: this runs 2 per KDA layer per row -- 72 per verify at m = 2 --
  // and a synchronous hipMemcpy blocks the host until the GPU drains each time,
  // which serialised the verify around ~70 host round trips. Async on the null
  // stream: same ordering, no host block.
  return hipMemcpyAsync(static_cast<char*>(keep->kda_conv) +
                            r * keep->conv_bytes + coff * sizeof(float),
                        static_cast<const char*>(st->kda_conv_device) +
                            coff * sizeof(float),
                        cn * sizeof(float),
                        hipMemcpyDeviceToDevice) == hipSuccess &&
         hipMemcpyAsync(static_cast<char*>(keep->kda_ssm) +
                            r * keep->ssm_bytes + soff * sizeof(float),
                        static_cast<const char*>(st->kda_ssm_device) +
                            soff * sizeof(float),
                        sn * sizeof(float),
                        hipMemcpyDeviceToDevice) == hipSuccess;
}

// LV1 R6: the reverse of KeepImageLayer, KDA layers only. MLA slices stay the
// zeros NewState wrote; the image's copies of those slices are zeros too.
bool InstallKdaSlices(State* trunk, const void* conv_row, const void* ssm_row) {
  if (trunk == nullptr || conv_row == nullptr || ssm_row == nullptr ||
      trunk->kda_conv_device == nullptr || trunk->kda_ssm_device == nullptr) {
    return false;
  }
  const ::ling3::KdaShape ks = MakeKdaShape();
  const std::size_t cn = static_cast<std::size_t>(ks.embd_r());
  const std::size_t sn = static_cast<std::size_t>(ks.embd_s());
  const std::size_t n = g.is_mla.size();
  if (n == 0 || trunk->kda_conv_off.size() != n ||
      trunk->kda_ssm_off.size() != n) {
    return false;
  }
  for (std::size_t li = 0; li < n; ++li) {
    if (g.is_mla[li])
      continue;
    const std::size_t coff = trunk->kda_conv_off[li];
    const std::size_t soff = trunk->kda_ssm_off[li];
    if (hipMemcpyAsync(
            static_cast<char*>(trunk->kda_conv_device) + coff * sizeof(float),
            static_cast<const char*>(conv_row) + coff * sizeof(float),
            cn * sizeof(float), hipMemcpyDeviceToDevice) != hipSuccess ||
        hipMemcpyAsync(
            static_cast<char*>(trunk->kda_ssm_device) + soff * sizeof(float),
            static_cast<const char*>(ssm_row) + soff * sizeof(float),
            sn * sizeof(float), hipMemcpyDeviceToDevice) != hipSuccess) {
      return false;
    }
  }
  return true;
}

bool KeepImageWhole(State* st, MultiRowKeep* keep, int row) {
  if (keep == nullptr || st == nullptr || row < 0 || row >= keep->rows) {
    return false;
  }
  auto* cd = static_cast<char*>(keep->kda_conv) +
             static_cast<std::size_t>(row) * keep->conv_bytes;
  auto* sd = static_cast<char*>(keep->kda_ssm) +
             static_cast<std::size_t>(row) * keep->ssm_bytes;
  // LM1 00:05: async on the null stream -- same stream as the kernels, so the
  // order is kept and the host is not blocked on a GPU drain per copy. The
  // verify's own D2H of the logits is what synchronises before anyone reads.
  return hipMemcpyAsync(cd, st->kda_conv_device, keep->conv_bytes,
                        hipMemcpyDeviceToDevice) == hipSuccess &&
         hipMemcpyAsync(sd, st->kda_ssm_device, keep->ssm_bytes,
                        hipMemcpyDeviceToDevice) == hipSuccess &&
         hipMemcpyAsync(keep->hidden + static_cast<std::size_t>(row) *
                                           static_cast<std::size_t>(g.hidden),
                        st->t,
                        static_cast<std::size_t>(g.hidden) * sizeof(float),
                        hipMemcpyDeviceToDevice) == hipSuccess;
}

bool StepMulti(const std::int32_t* tokens, int m, State* st,
               std::span<float> out_logits, std::string* error_msg,
               MultiRowKeep* keep) {
  if (!g.ready || st == nullptr || tokens == nullptr || m <= 0) {
    if (error_msg != nullptr)
      *error_msg = "ling3 ROCm StepMulti: bad arguments";
    return false;
  }
  const std::size_t V = static_cast<std::size_t>(g.vocab);
  if (out_logits.size() < static_cast<std::size_t>(m) * V) {
    if (error_msg != nullptr) {
      *error_msg = "ling3 ROCm StepMulti: logits span too small for m rows";
    }
    return false;
  }
  if (static_cast<std::size_t>(st->pos) + static_cast<std::size_t>(m) >
      static_cast<std::size_t>(st->max_ctx)) {
    if (error_msg != nullptr) {
      *error_msg = "ling3 ROCm StepMulti: context length exceeded";
    }
    return false;
  }
  const int pos0 = st->pos;
  for (int r = 0; r < m; ++r) {
    const std::size_t off = static_cast<std::size_t>(r) * V;
    if (!Step(tokens[r], st, std::span<float>(out_logits.data() + off, V),
              error_msg)) {
      // Put the state back where we found it: a failed verify must leave the
      // session usable for a plain decode.
      st->pos = pos0;
      return false;
    }
    // LM1 SPEC4 item 1: image the trunk state after this row so a rejection can
    // be unwound without replaying. Token-outer here, so one whole-array copy
    // per row is the right shape. Rows 0..m-2 only -- row m-1 is the
    // full-accept case.
    if (keep != nullptr && r + 1 < m) {
      if (!KeepImageWhole(st, keep, r)) {
        st->pos = pos0;
        return false;
      }
    }
  }
  return true;
}

#if defined(LING3_HAVE_MULTI_ROW_KERNELS)
// ---------------------------------------------------------------------------
// LM1 -- StepMultiMultiRow: the multi-row verify.
//
// LS4's StepMulti is byte-identical to m sequential Step[s] BY CONSTRUCTION and
// costs exactly that: every weight matrix is read m times. This is the same
// verify with the weight reads amortised. Every projection is one
// ling3_gemv_m over m activation columns, the routed experts of one layer are
// one ling3_expert_ffn_m over the m * n_used (token, slot) pairs with the
// per-pair expert id read on the device, and the norms, the residual adds, the
// router and the argmax are launched ONCE for all m rows. KDA and MLA are still
// stepped per row, in token order -- their state is sequential, so that is the
// model, not an optimisation left on the table.
//
// Byte-identity with Step is structural, the same argument StepMulti makes, and
// it holds only because every kernel below reduces in the same order over the
// same per-column inputs as its m = 1 counterpart:
//
//   ling3_gemv_m / k_gemv_wide_q8_m   vs  ling3_gemv / k_gemv_wide_q8
//      ls1_dot_blk32(ls1_load_blk32(..)) IS ls1_dot_fp4_blk32_q8(..) -- that is
//      how the latter is defined -- the per-thread qb stride is the same, and
//      ls1_block_sum is the same reduction on the same partials.
//   ling3_expert_ffn_m / k_moe_*_m    vs  ling3_expert_ffn_dev / k_moe_*
//      LS1's own gate: bit-exact per column against the m = 1 grouped kernel.
//   ling3_expert_ffn_m at n_used = 1  vs  ling3_dense_ffn / ling3_expert_ffn
//      the same dots, the same ling3_swiglu_elem, and weight 1.0 into a zero
//      accumulator is exact, so the dense / shared expert sum is the same
//      float.
//   ling3_expert_ffn_shared_m         vs  that n_used = 1 call (LV1 R3)
//      same quant, same per-thread block dots with the weight decoded once,
//      same swiglu, same acc += weight * part. Routed experts stay on
//      ling3_expert_ffn_m.
//   k_route_batch_logits_m            vs  k_route_batch_logits (LV1 R2)
//      n_tokens 2..4 only: same per-thread stride, same 64-wide tree.
//   ling3_route_batch                 vs  ling3_route
//      k_route_batch is k_route copied verbatim: same sigmoid+bias scores, same
//      group mask, same bitonic argsort with the same strict-compare tie rule,
//      same weight normalisation. The two argsort device functions are copies
//      of each other.
//
// The ops_* entry points already take a row count and are the same calls Step
// makes with rows = 1. LS1's kernels sit behind LING3_HAVE_MULTI_ROW_KERNELS
// (detected in CMakeLists); without them this function does not compile and
// StepMultiDispatch keeps StepMulti. Anything this path cannot cover --
// streamed experts, a non-ROCmFP4 weight, m above LS1_MAX_M -- falls back to
// StepMulti, which is byte-identical too: a slower verify, never a wrong one.
// ---------------------------------------------------------------------------
namespace {

// LS1's small-m kernels take m = 1..LS1_MAX_M (4 in ling3_gemv_q8.hip). One set
// of m-row scratch for the executor, allocated on first use: it is pure
// workspace, every buffer is written before it is read, so it belongs to the
// executor and not to a State -- the same choice g_dev.d_ids already makes.
// Layout is token-major [m][n] throughout, which is what ling3_gemv_m /
// ling3_expert_ffn_m and the ops_* row loops both want, and what makes row 0 of
// every buffer the one GpuTrace prints (row 0 is always at offset 0).
constexpr int kMultiMax = 4;

struct MultiScratch {
  float* x{nullptr};         // [m][H] residual stream
  float* normed{nullptr};    // [m][H]
  float* attn_out{nullptr};  // [m][H]
  float* ffn_in{nullptr};    // [m][H]
  float* t{nullptr};         // [m][H] post-ffn_norm / post-final_norm
  float* y{nullptr};         // [m][H] routed experts
  float* sh{nullptr};        // [m][H] dense / shared expert
  float* mla_q{nullptr};     // [m][q_dim]
  float* mla_kv_a{nullptr};  // [m][kv_a]
  float* mla_gate{nullptr};  // [m][heads]
  float* mla_out{nullptr};   // [m][o_dim]
  float* q{nullptr};         // [m][I]
  float* k{nullptr};
  float* v{nullptr};
  float* f{nullptr};
  float* g2{nullptr};
  float* beta{nullptr};     // [m][heads]
  float* kda_out{nullptr};  // [m][I]
  float* logits{nullptr};   // [m][V]
  std::int32_t* tok_in{nullptr};
  std::int32_t* tok_out{nullptr};
  std::int32_t* pos_d{nullptr};     // [m] the row's absolute position
  std::int32_t* ids{nullptr};       // [m][n_used] the router's output
  float* wts{nullptr};              // [m][n_used]
  std::int32_t* route_rc{nullptr};  // [m]
  std::int32_t* zero_ids{nullptr};  // [m] all 0 -- the dense / shared expert id
  float* one_wts{nullptr};          // [m] all 1 -- its weight
  std::vector<void*> allocs;
  bool ready{false};
};

// LM1: hipEvent timing of the verify itself, printed at process exit.
//
// LS4's MTP_SPEC reporter is fed from ~Session and sessions are pooled rather
// than destroyed, so it prints "steps=0" no matter how much speculation ran
// (LS4 REPORT3 §6.4 flagged this; it is still true on int-24). This one is fed
// from the function that does the work, so it cannot miss it. The end-of-call
// logit readback already synchronises the device, so recording the stop event
// and asking for its elapsed time costs no extra drain.
//
// `LM1_FORCE_FALLBACK=1` routes the SAME calls through StepMulti -- m
// sequential Step[s] -- under the same timer. That is how "one Step" is
// measured here: the fallback's per-row figure IS one Step, taken with the same
// instrument.
struct MultiTimer {
  hipEvent_t a{nullptr}, b{nullptr};
  double ms{0.0};
  long calls{0};
  long n_by_m[5] = {0, 0, 0, 0, 0};
  double ms_by_m[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  int fallback{0};
};
MultiTimer g_mt;
bool g_mt_ready = false;
bool g_mt_force_fallback = false;

void MultiTimerInit() {
  if (g_mt_ready)
    return;
  g_mt_ready = true;
  const char* e = std::getenv("LM1_FORCE_FALLBACK");
  g_mt_force_fallback = e != nullptr && e[0] != '\0' && std::string(e) != "0";
  if (hipEventCreate(&g_mt.a) != hipSuccess ||
      hipEventCreate(&g_mt.b) != hipSuccess) {
    g_mt_ready = false;
    return;
  }
  std::atexit([] {
    if (g_mt.calls == 0) {
      std::fprintf(stderr, "LM1_MULTIROW calls=0 (never ran)\n");
      return;
    }
    std::fprintf(
        stderr, "LM1_MULTIROW calls=%ld fallback=%d total_ms=%.1f mean_ms=%.3f",
        g_mt.calls, g_mt.fallback, g_mt.ms,
        g_mt.ms / static_cast<double>(g_mt.calls));
    for (int m = 1; m <= 4; ++m) {
      if (g_mt.n_by_m[m] == 0)
        continue;
      const double mean = g_mt.ms_by_m[m] / static_cast<double>(g_mt.n_by_m[m]);
      std::fprintf(stderr, " | m=%d n=%ld mean_ms=%.3f ms_per_row=%.3f", m,
                   g_mt.n_by_m[m], mean, mean / static_cast<double>(m));
    }
    std::fprintf(stderr, "\n");
  });
}

MultiScratch g_ms;

bool MultiAlloc(std::string* error_msg) {
  if (g_ms.ready) {
    return true;
  }
  const std::size_t M = static_cast<std::size_t>(kMultiMax);
  const std::size_t H = static_cast<std::size_t>(g.hidden);
  const std::size_t I = static_cast<std::size_t>(g.inner);
  const std::size_t G = static_cast<std::size_t>(g.heads);
  const std::size_t V = static_cast<std::size_t>(g.vocab);
  const ::ling3::MlaShape ms = MakeMlaShape();
  const std::size_t QD = static_cast<std::size_t>(ms.q_dim());
  const std::size_t KVA = static_cast<std::size_t>(ms.kv_a());
  const std::size_t OD = static_cast<std::size_t>(ms.o_dim());
  const std::size_t NU = static_cast<std::size_t>(g.n_used);
  bool ok = true;
  const auto take = [&](void* p) {
    if (p == nullptr) {
      ok = false;
    } else {
      g_ms.allocs.push_back(p);
    }
    return p;
  };
  const auto act = [&](std::size_t n) {
    return take(DevAlloc(n * sizeof(float)));
  };
  const auto acti = [&](std::size_t n) {
    return take(DevAlloc(n * sizeof(std::int32_t)));
  };
  g_ms.x = static_cast<float*>(act(M * H));
  g_ms.normed = static_cast<float*>(act(M * H));
  g_ms.attn_out = static_cast<float*>(act(M * H));
  g_ms.ffn_in = static_cast<float*>(act(M * H));
  g_ms.t = static_cast<float*>(act(M * H));
  g_ms.y = static_cast<float*>(act(M * H));
  g_ms.sh = static_cast<float*>(act(M * H));
  g_ms.mla_q = static_cast<float*>(act(M * QD));
  g_ms.mla_kv_a = static_cast<float*>(act(M * KVA));
  g_ms.mla_gate = static_cast<float*>(act(M * G));
  g_ms.mla_out = static_cast<float*>(act(M * OD));
  g_ms.q = static_cast<float*>(act(M * I));
  g_ms.k = static_cast<float*>(act(M * I));
  g_ms.v = static_cast<float*>(act(M * I));
  g_ms.f = static_cast<float*>(act(M * I));
  g_ms.g2 = static_cast<float*>(act(M * I));
  g_ms.beta = static_cast<float*>(act(M * G));
  g_ms.kda_out = static_cast<float*>(act(M * I));
  g_ms.logits = static_cast<float*>(act(M * V));
  g_ms.tok_in = static_cast<std::int32_t*>(acti(M));
  g_ms.tok_out = static_cast<std::int32_t*>(acti(M));
  g_ms.pos_d = static_cast<std::int32_t*>(acti(M));
  g_ms.ids = static_cast<std::int32_t*>(acti(M * NU));
  g_ms.wts = static_cast<float*>(act(M * NU));
  g_ms.route_rc = static_cast<std::int32_t*>(acti(M));
  g_ms.zero_ids = static_cast<std::int32_t*>(acti(M));
  g_ms.one_wts = static_cast<float*>(act(M));
  if (!ok) {
    if (error_msg != nullptr) {
      *error_msg = "ling3 ROCm StepMultiMultiRow: scratch allocation";
    }
    return false;
  }
  // ling3_dense_ffn is one expert, one slot, weight 1 -- which is what
  // ling3_expert_ffn_m sees with these. They never change, so fill them once.
  std::vector<std::int32_t> zeros(M, 0);
  std::vector<float> ones(M, 1.0F);
  if (hipMemcpy(g_ms.zero_ids, zeros.data(), M * sizeof(std::int32_t),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(g_ms.one_wts, ones.data(), M * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          "ling3 ROCm StepMultiMultiRow: dense-expert constants upload";
    }
    return false;
  }
  g_ms.ready = true;
  return true;
}

// Every weight this path hands the small-m kernels must be ROCmFP4 (100/101):
// ling3_gemv_m and ling3_expert_ffn_m refuse anything else. Ling-3.0-flash
// decodes nothing else -- LS1 counted 383 ROCmFP4 tensors and no other quant
// type in the decode path -- but a different model would otherwise fail the
// verify at runtime instead of falling back, so check once and cache it.
bool MultiWeightsSupported() {
  static const bool ok = [] {
    const auto fp4 = [](int t) {
      return t == LING3_Q4_0_ROCMFP4 || t == LING3_Q4_0_ROCMFP4_FAST;
    };
    if (!fp4(g_dev.output_type)) {
      return false;
    }
    std::size_t slot = 0;
    for (std::size_t li = 0; li < g_dev.layers.size(); ++li) {
      const Layer& L = g_dev.layers[li];
      if (!fp4(L.ffn_gate.type) || !fp4(L.ffn_up.type) ||
          !fp4(L.ffn_down.type)) {
        return false;
      }
      if (li >= g.is_mla.size()) {
        return false;
      }
      if (g.is_mla[li]) {
        if (!fp4(L.mq.type) || !fp4(L.m_kv_a_mqa.type) || !fp4(L.m_gate.type) ||
            !fp4(L.m_wo.type)) {
          return false;
        }
      } else {
        for (int tt : {L.wq.type, L.wk.type, L.wv.type, L.wf.type, L.wg.type,
                       L.wbeta.type, L.wo.type}) {
          if (!fp4(tt)) {
            return false;
          }
        }
      }
      if (!L.dense) {
        if (g_dev.stream_experts) {
          return false;
        }
        if (slot >= g_dev.experts.size() || !fp4(g_dev.experts[slot].type)) {
          return false;
        }
        ++slot;
      }
    }
    return true;
  }();
  return ok;
}

}  // namespace

bool StepMultiMultiRow(const std::int32_t* tokens, int m, State* st,
                       std::span<float> out_logits, std::string* error_msg,
                       MultiRowKeep* keep) {
  const auto fail = [&](const char* what) {
    if (error_msg != nullptr) {
      *error_msg = std::string("ling3 ROCm StepMultiMultiRow: ") + what;
    }
    return false;
  };
  if (!g.ready || st == nullptr || tokens == nullptr || m <= 0) {
    return fail("bad arguments");
  }
  const std::size_t V = static_cast<std::size_t>(g.vocab);
  if (out_logits.size() < static_cast<std::size_t>(m) * V) {
    return fail("logits span too small for m rows");
  }
  if (static_cast<std::size_t>(st->pos) + static_cast<std::size_t>(m) >
      static_cast<std::size_t>(st->max_ctx)) {
    return fail("context length exceeded");
  }
  // Cases the small-m kernels cannot cover. StepMulti is byte-identical to
  // m sequential Step[s] by construction, so the fallback is never wrong.
  // LM1_FORCE_FALLBACK=1 takes it deliberately: same timer, and its per-row
  // figure is one Step by construction.
  MultiTimerInit();
  const bool timed = g_mt_ready;
  if (timed) {
    hipEventRecord(g_mt.a);
  }
  const auto stop_timer = [&]() {
    if (!timed)
      return;
    hipEventRecord(g_mt.b);
    (void)hipEventSynchronize(g_mt.b);
    float ms = 0.0f;
    if (hipEventElapsedTime(&ms, g_mt.a, g_mt.b) == hipSuccess) {
      const int mi = (m >= 1 && m <= 4) ? m : 0;
      g_mt.ms += ms;
      ++g_mt.calls;
      g_mt.n_by_m[mi] += 1;
      g_mt.ms_by_m[mi] += ms;
      if (g_mt_force_fallback)
        g_mt.fallback = 1;
    }
  };
  if (m > kMultiMax || g_dev.stream_experts || !MultiWeightsSupported() ||
      g_mt_force_fallback) {
    const bool ok = StepMulti(tokens, m, st, out_logits, error_msg, keep);
    stop_timer();
    return ok;
  }
  if (!MultiAlloc(error_msg)) {
    return false;
  }
  MultiScratch& S = g_ms;
  hipStream_t stream = nullptr;  // the null stream, as Step uses
  const int pos0 = st->pos;
  const int mm = m;
  const std::size_t H = static_cast<std::size_t>(g.hidden);
  const std::size_t I = static_cast<std::size_t>(g.inner);
  const std::size_t G = static_cast<std::size_t>(g.heads);
  const ::ling3::KdaShape ks = MakeKdaShape();
  const ::ling3::KdaScalars ksc;
  const ::ling3::MlaShape ms = MakeMlaShape();
  const ::ling3::MlaScalars msc;
  const int q_dim = ms.q_dim();
  const int kv_a = ms.kv_a();
  const int o_dim = ms.o_dim();

  // ---- positions and token ids, one H2D for all m rows -----------------
  {
    std::vector<std::int32_t> hs(static_cast<std::size_t>(mm));
    std::vector<std::int32_t> ps(static_cast<std::size_t>(mm));
    for (int t = 0; t < mm; ++t) {
      hs[static_cast<std::size_t>(t)] = static_cast<std::int32_t>(tokens[t]);
      ps[static_cast<std::size_t>(t)] = pos0 + t;
    }
    if (hipMemcpy(S.tok_in, hs.data(),
                  static_cast<std::size_t>(mm) * sizeof(std::int32_t),
                  hipMemcpyHostToDevice) != hipSuccess) {
      return fail("token upload");
    }
    if (hipMemcpy(S.pos_d, ps.data(),
                  static_cast<std::size_t>(mm) * sizeof(std::int32_t),
                  hipMemcpyHostToDevice) != hipSuccess) {
      return fail("position upload");
    }
  }

  // ---- embeddings: ONE launch for all m rows ---------------------------
  ::ling3::ops_embedding_row(g_dev.token_embd_type, g_dev.token_embd, g.hidden,
                             g.vocab, S.tok_in, mm, S.x, stream);
  GpuTrace(st, static_cast<std::uint32_t>(-1), "inp_embd", S.x, H);

  gufo::models::ling3::vision::OnHidden(S.x, pos0, mm, static_cast<int>(H),
                                        stream);

  // g_dev.experts holds MoE layers only, so index it with its own counter
  // rather than the block number -- the same rule Step follows.
  std::size_t expert_slot = 0;
  for (std::size_t li = 0; li < g_dev.layers.size(); ++li) {
    Layer& L = g_dev.layers[li];

    // ---- attention norm: ONE launch for all m rows ---------------------
    ::ling3::ops_rms_norm(S.x, L.attn_norm, S.normed, g.hidden, mm, g.eps,
                          stream);
    GpuTrace(st, static_cast<std::uint32_t>(li), "attn_norm", S.normed, H);

    if (g.is_mla[li]) {
      // Three projections, each reading its weights ONCE for all m columns.
      // LM1 SPEC4 item 2: ONE quantisation of S.normed for all three
      // projections, exactly as Step does with ling3_act_prepare. Bit-identical
      // either way -- k_q8_1_quant is deterministic per element, so
      // re-quantising would produce the same bytes -- but three quantises
      // become one launch.
      const int8_t* nq = nullptr;
      const float* nd = nullptr;
      if (ling3_quant_m(S.normed, mm, g.hidden, stream, &nq, &nd) != LING3_OK) {
        return fail("normed quantise");
      }
      if (ling3_gemv_q8_m(L.mq.type, L.mq.d, nq, nd, S.mla_q, g.hidden, q_dim,
                          mm) != LING3_OK ||
          ling3_gemv_q8_m(L.m_kv_a_mqa.type, L.m_kv_a_mqa.d, nq, nd, S.mla_kv_a,
                          g.hidden, kv_a, mm) != LING3_OK ||
          ling3_gemv_q8_m(L.m_gate.type, L.m_gate.d, nq, nd, S.mla_gate,
                          g.hidden, static_cast<int>(G), mm) != LING3_OK) {
        return fail("mla projections");
      }
      GpuTrace(st, static_cast<std::uint32_t>(li), "mla_q", S.mla_q,
               static_cast<std::size_t>(q_dim));
      // MLA is token-sequential through its latent cache: one block per row, in
      // token order, each appending its own position and attending over n_pos.
      for (int t = 0; t < mm; ++t) {
        const std::size_t tq =
            static_cast<std::size_t>(t) * static_cast<std::size_t>(q_dim);
        const std::size_t tk =
            static_cast<std::size_t>(t) * static_cast<std::size_t>(kv_a);
        const std::size_t tg = static_cast<std::size_t>(t) * G;
        const std::size_t to =
            static_cast<std::size_t>(t) * static_cast<std::size_t>(o_dim);
        ::ling3::MlaInputs in;
        in.q = ::ling3::make_view(S.mla_q + tq, q_dim, 1);
        in.kv_a = ::ling3::make_view(S.mla_kv_a + tk, kv_a, 1);
        in.gate = ::ling3::make_view(S.mla_gate + tg, static_cast<int>(G), 1);
        in.n_tokens = 1;
        in.n_seqs = 1;

        ::ling3::MlaCache cache;
        // LG1 W3: reindex mla_off through the MLA ordinal map. All three
        // construction sites are inside `if (g.is_mla[li])`, so mla_ord[li] >=
        // 0 here; the abort is a safety net for any future invariant break.
        if (g.mla_ord[li] < 0) {
          std::fprintf(stderr, "lg1: mla_ord[%zu] < 0 at Step MlaCache build\n",
                       li);
          std::abort();
        }
        cache.kv = static_cast<float*>(st->mla_cache_device) +
                   st->mla_off[g.mla_ord[li]];
        cache.s_pos = ms.kv_a();
        cache.s_seq = 0;
        cache.max_pos = static_cast<int>(st->max_ctx);

        ::ling3::MlaWeights mw;
        mw.wk_b = L.m_wk_b_f32;
        mw.wv_b = L.m_wv_b_f32;
        mw.wk_bT = L.m_wk_bT;
        mw.wv_bT = L.m_wv_bT;
        mw.kv_a_norm = L.m_kv_a_norm;

        float* out_t = S.mla_out + to;
        // n_pos is the count of VALID positions including this row, and this
        // row appends at absolute position pos0 + t -- the same arithmetic
        // Step does with st->pos and one row.
        const int n_pos = pos0 + t + 1;
        // LM1 SPEC4 item 3: split-K ("flash decoding") attention inside the
        // verify, gated by the SAME guard Step uses. Before this the verify was
        // the only MLA left on the serial path, so at depth the verify's rows
        // scaled with the context while a plain Step's did not -- which is a
        // large part of why a rejected draft was 2.4 plain steps. Per row here;
        // LS3's m-row entry replaces the loop when it lands.
        if (MlaSplitkOn() && st->mla_sk_acc != nullptr &&
            n_pos <= static_cast<int>(st->max_ctx) && n_pos >= MlaSplitkMin() &&
            MlaSplitkSlicesFit(n_pos)) {
          ::ling3::mla_block_splitk(ms, msc, mw, cache, in, S.pos_d + t, n_pos,
                                    ::ling3::make_view(out_t, o_dim, 1),
                                    st->mla_sk_q, st->mla_sk_m, st->mla_sk_l,
                                    st->mla_sk_acc, MlaSplitkSlice(n_pos),
                                    MlaSplitkHeads(), stream);
        } else if (st->mla_scratch != nullptr &&
                   n_pos <= static_cast<int>(st->max_ctx) &&
                   st->mla_scratch_bytes >=
                       ::ling3::mla_block_scratch_bytes(ms, 1, n_pos)) {
          ::ling3::mla_block_scratch(ms, msc, mw, cache, in, S.pos_d + t, n_pos,
                                     ::ling3::make_view(out_t, o_dim, 1),
                                     st->mla_scratch, st->mla_scratch_bytes,
                                     stream);
        } else {
          ::ling3::mla_block(ms, msc, mw, cache, in, S.pos_d + t, n_pos,
                             ::ling3::make_view(out_t, o_dim, 1), stream);
        }
      }
      GpuTrace(st, static_cast<std::uint32_t>(li), "mla_attn_out", S.mla_out,
               static_cast<std::size_t>(o_dim));
      // Output projection, one weight read for all m columns.
      if (ling3_quant_m(S.mla_out, mm, o_dim, stream, &nq, &nd) != LING3_OK ||
          ling3_gemv_q8_m(L.m_wo.type, L.m_wo.d, nq, nd, S.attn_out, o_dim,
                          g.hidden, mm) != LING3_OK) {
        return fail("mla output projection");
      }
      GpuTrace(st, static_cast<std::uint32_t>(li), "attn_out", S.attn_out, H);
      (void)msc;
    } else {
      // LM1 SPEC4 item 2: one quantisation of S.normed for all six.
      const int8_t* nq = nullptr;
      const float* nd = nullptr;
      if (ling3_quant_m(S.normed, mm, g.hidden, stream, &nq, &nd) != LING3_OK) {
        return fail("normed quantise");
      }
      if (ling3_gemv_q8_m(L.wq.type, L.wq.d, nq, nd, S.q, g.hidden,
                          static_cast<int>(I), mm) != LING3_OK ||
          ling3_gemv_q8_m(L.wk.type, L.wk.d, nq, nd, S.k, g.hidden,
                          static_cast<int>(I), mm) != LING3_OK ||
          ling3_gemv_q8_m(L.wv.type, L.wv.d, nq, nd, S.v, g.hidden,
                          static_cast<int>(I), mm) != LING3_OK ||
          ling3_gemv_q8_m(L.wf.type, L.wf.d, nq, nd, S.f, g.hidden,
                          static_cast<int>(I), mm) != LING3_OK ||
          ling3_gemv_q8_m(L.wg.type, L.wg.d, nq, nd, S.g2, g.hidden,
                          static_cast<int>(I), mm) != LING3_OK ||
          ling3_gemv_q8_m(L.wbeta.type, L.wbeta.d, nq, nd, S.beta, g.hidden,
                          static_cast<int>(G), mm) != LING3_OK) {
        return fail("kda projections");
      }
      // KDA's conv + SSM state is sequential: one forward per row, in token
      // order, each advancing the same state the single-token path advances.
      for (int t = 0; t < mm; ++t) {
        const std::size_t off_i = static_cast<std::size_t>(t) * I;
        const std::size_t off_g = static_cast<std::size_t>(t) * G;
        ::ling3::KdaWeights kw;
        kw.conv_wq = L.conv_q;
        kw.conv_wk = L.conv_k;
        kw.conv_wv = L.conv_v;
        kw.a_exp = L.a_exp;
        kw.dt_bias = L.dt_bias;
        kw.o_norm = L.o_norm;

        ::ling3::KdaInputs ki;
        ki.q = ::ling3::make_view(S.q + off_i, static_cast<int>(I), 1);
        ki.k = ::ling3::make_view(S.k + off_i, static_cast<int>(I), 1);
        ki.v = ::ling3::make_view(S.v + off_i, static_cast<int>(I), 1);
        ki.f = ::ling3::make_view(S.f + off_i, static_cast<int>(I), 1);
        ki.g2 = ::ling3::make_view(S.g2 + off_i, static_cast<int>(I), 1);
        ki.beta = ::ling3::make_view(S.beta + off_g, static_cast<int>(G), 1);
        ki.n_tokens = 1;
        ki.n_seqs = 1;

        ::ling3::KdaState kst;
        kst.conv_state =
            static_cast<float*>(st->kda_conv_device) + st->kda_conv_off[li];
        kst.ssm_state =
            static_cast<float*>(st->kda_ssm_device) + st->kda_ssm_off[li];
        kst.n_seqs = 1;

        float* out_t = S.kda_out + off_i;
        if (st->kda_scratch != nullptr &&
            st->kda_scratch_bytes >=
                ::ling3::kda_forward_scratch_bytes(ks, 1, 1)) {
          ::ling3::kda_forward_scratch(
              ks, ksc, kw, ki, kst,
              ::ling3::make_view(out_t, static_cast<int>(I), 1),
              st->kda_scratch, st->kda_scratch_bytes, stream);
        } else {
          ::ling3::kda_forward(
              ks, ksc, kw, ki, kst,
              ::ling3::make_view(out_t, static_cast<int>(I), 1), stream);
        }
        // LM1 SPEC4 item 1: this layer has now finished row t; image its slice.
        if (keep != nullptr && t + 1 < mm && !KeepImageLayer(st, keep, li, t)) {
          return fail("keep image (kda)");
        }
      }
      GpuTrace(st, static_cast<std::uint32_t>(li), "kda_gated", S.kda_out, I);
      if (ling3_quant_m(S.kda_out, mm, static_cast<int>(I), stream, &nq, &nd) !=
              LING3_OK ||
          ling3_gemv_q8_m(L.wo.type, L.wo.d, nq, nd, S.attn_out,
                          static_cast<int>(I), g.hidden, mm) != LING3_OK) {
        return fail("kda output projection");
      }
      GpuTrace(st, static_cast<std::uint32_t>(li), "kda_out", S.attn_out, H);
    }

    // ---- residual add + ffn norm, once for all m rows ------------------
    // NOT ops_add_rms_norm. Its rsum output is written at rsum[i] with NO row
    // stride (ling3_ops.hip, add_rms_norm_kernel: "if (rsum) rsum[i] = xi;"
    // while yr IS row-strided), so at rows > 1 every row overwrites the first n
    // floats of ffn_in and the other rows are never written at all. Invisible
    // at rows == 1, the only shape its other caller (Step) uses, and
    // ForwardBatch deliberately does not use it -- it splits the op in two. The
    // two-kernel form below is the same arithmetic LS2 fused: the residual sum
    // is a[i]+b[i] in both, and the norm consumes exactly that. Byte-identical
    // to Step.
    ::ling3::ops_residual_add(S.x, S.attn_out, S.ffn_in, g.hidden, mm, stream);
    ::ling3::ops_rms_norm(S.ffn_in, L.ffn_norm, S.t, g.hidden, mm, g.eps,
                          stream);
    GpuTrace(st, static_cast<std::uint32_t>(li), "ffn_inp", S.ffn_in, H);
    GpuTrace(st, static_cast<std::uint32_t>(li), "ffn_norm", S.t, H);

    if (L.dense) {
      // ling3_dense_ffn is ling3_expert_ffn with n_used = 1, n_expert = 1,
      // id 0, weight 1 -- exactly what this call sees. y is WRITTEN, not
      // accumulated, which is what the m = 1 grouped kernel does too.
      if (ling3_expert_ffn_shared_m(L.ffn_gate.type, L.ffn_gate.d, L.ffn_up.d,
                                    L.ffn_down.d, S.t, S.zero_ids, S.one_wts,
                                    mm, 1, 1, g.hidden, g.ffn_len,
                                    L.clamp_shexp, S.y) != LING3_OK) {
        return fail("dense expert FFN");
      }
    } else {
      // Router for all m rows in ONE launch. ids / weights land token-major at
      // [t][n_used], which is the flat s = t * n_used + slot layout
      // ling3_expert_ffn_m reads -- no repack, no host round trip.
      if (ling3_route_batch(S.t, static_cast<const float*>(L.router.d),
                            L.probs_bias, g.hidden, mm, g.n_expert, g.n_used,
                            g.n_groups, g.n_group_used, g.w_scale, g.norm_w,
                            S.ids, S.wts, S.route_rc) != LING3_OK) {
        return fail("router");
      }
      GpuTraceIds(st, static_cast<std::uint32_t>(li), "ffn_moe_topk", S.ids,
                  static_cast<std::size_t>(g.n_used));
      GpuTrace(st, static_cast<std::uint32_t>(li), "ffn_moe_weights_scaled",
               S.wts, static_cast<std::size_t>(g.n_used));
      // All 512 experts are resident (stream_experts falls back above), so ONE
      // grouped call covers every selected expert of every row: 3 launches for
      // the whole layer instead of 3 per row.
      const Device::ResidentExperts& re = g_dev.experts[expert_slot];
      if (ling3_expert_ffn_m(re.type, re.gate, re.up, re.down, S.t, S.ids,
                             S.wts, mm, g.n_used, g.n_expert, g.hidden,
                             g.expert_ff, L.clamp_exp, S.y) != LING3_OK) {
        return fail("grouped expert FFN");
      }
      ++expert_slot;
      // The shared expert is never routed and never scaled: one more grouped
      // call at n_used = 1, id 0, weight 1, then add it in. (Step writes it to
      // st->f because ling3_expert_ffn overwrites y; here the two live in
      // separate buffers and the add is explicit either way.)
      if (ling3_expert_ffn_shared_m(L.ffn_gate.type, L.ffn_gate.d, L.ffn_up.d,
                                    L.ffn_down.d, S.t, S.zero_ids, S.one_wts,
                                    mm, 1, 1, g.hidden, g.expert_ff,
                                    L.clamp_shexp, S.sh) != LING3_OK) {
        return fail("shared expert FFN");
      }
      ling3_vec_add(S.y, S.sh, S.y, g.hidden * mm);
    }

    ::ling3::ops_residual_add(S.ffn_in, S.y, S.x, g.hidden, mm, stream);
    GpuTrace(st, static_cast<std::uint32_t>(li), "ffn_out", S.y, H);
    GpuTrace(st, static_cast<std::uint32_t>(li), "l_out", S.x, H);
  }

  // ONE blocking read for the whole verify: the expert FFN's status word and
  // the router's per-row words. Step does the same after its layer loop.
  if (ling3_expert_ffn_m_check() != LING3_OK) {
    return fail("grouped expert FFN reported an error");
  }
  {
    std::vector<std::int32_t> rc(static_cast<std::size_t>(mm), LING3_OK);
    if (hipMemcpy(rc.data(), S.route_rc,
                  static_cast<std::size_t>(mm) * sizeof(std::int32_t),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      return fail("router status readback");
    }
    for (int t = 0; t < mm; ++t) {
      if (rc[static_cast<std::size_t>(t)] != LING3_OK) {
        return fail("router reported an error");
      }
    }
  }

  // ---- final norm + LM head + argmax, ONE launch each for all m rows ---
  ::ling3::ops_final_norm(S.x, g_dev.output_norm, S.t, g.hidden, mm, stream);
  // LM1 SPEC4 item 1: row t's post-output_norm hidden, for the same reason --
  // ReadHidden hands the draft head its h_prev from state->t. One contiguous
  // copy of the first m-1 rows; S.t is [m][hidden] token-major.
  if (keep != nullptr && mm > 1 &&
      hipMemcpyAsync(keep->hidden, S.t,
                     static_cast<std::size_t>(mm - 1) * H * sizeof(float),
                     hipMemcpyDeviceToDevice) != hipSuccess) {
    return fail("keep image (hidden)");
  }
  GpuTrace(st, static_cast<std::uint32_t>(-1), "result_norm_input", S.x, H);
  if (ling3_gemv_m(g_dev.output_type, g_dev.output, S.t, S.logits, g.hidden,
                   g.vocab, mm) != LING3_OK) {
    return fail("lm head");
  }
  ::ling3::ops_argmax(S.logits, g.vocab, mm, S.tok_out, stream);
  if (hipMemcpy(out_logits.data(), S.logits,
                static_cast<std::size_t>(mm) * V * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) {
    return fail("logits readback");
  }
  // `ReadHidden` hands the MTP head its h_prev from state->t, which Step leaves
  // holding the LAST row's post-output_norm hidden. Do the same: without this
  // the next draft would run on whatever the previous call left in state->t and
  // acceptance would collapse -- LS4 phase 2's stale-hidden bug, in a different
  // buffer. One H-sized D2D copy, against the ~150 MB of weights this call just
  // amortised.
  if (hipMemcpy(st->t, S.t + static_cast<std::size_t>(mm - 1) * H,
                H * sizeof(float), hipMemcpyDeviceToDevice) != hipSuccess) {
    return fail("hidden copy for the draft head");
  }
  st->pos += mm;
  stop_timer();
  return true;
}
#endif  // LING3_HAVE_MULTI_ROW_KERNELS

bool StepMultiDispatch(const std::int32_t* tokens, int m, State* state,
                       std::span<float> out_logits, std::string* error_msg,
                       MultiRowKeep* keep) {
#if defined(LING3_HAVE_MULTI_ROW_KERNELS)
  // LS1 phase 4 slot. Gated on LING3_MTP_MULTI=1 because per-column
  // bit-exactness against m = 1 is LS1's gate, not ours: if a multi-row kernel
  // ever disagreed with Step on a single column, drafting would silently stop
  // being byte-identical, which is the one property this path exists to give.
  static const bool use_multi = [] {
    const char* v = std::getenv("LING3_MTP_MULTI");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
  }();
  if (use_multi) {
    return StepMultiMultiRow(tokens, m, state, out_logits, error_msg, keep);
  }
#endif
  return StepMulti(tokens, m, state, out_logits, error_msg, keep);
}

bool SharedTensorsView(SharedTensors* out) noexcept {
  if (out == nullptr || !g.ready) {
    return false;
  }
  out->token_embd = g_dev.token_embd;
  out->token_embd_type = g_dev.token_embd_type;
  out->output = g_dev.output;
  out->output_type = g_dev.output_type;
  out->output_norm = g_dev.output_norm;
  out->hidden = g.hidden;
  out->vocab = g.vocab;
  return true;
}

// `st->t` is left holding the post-`output_norm` hidden of the LAST token by
// the ops_final_norm at the end of Step(). ForwardBatch normalises into B->t,
// so this is only valid on the single-token path.
bool ReadHidden(const State* state, std::span<float> out,
                std::string* error_msg) {
  if (state == nullptr || !g.ready) {
    if (error_msg != nullptr)
      *error_msg = "ling3 ROCm: no state";
    return false;
  }
  const std::size_t need = static_cast<std::size_t>(g.hidden);
  if (out.size() < need) {
    if (error_msg != nullptr)
      *error_msg = "ling3 ROCm: hidden span too short";
    return false;
  }
  if (hipMemcpy(out.data(), state->t, need * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) {
    if (error_msg != nullptr)
      *error_msg = "ling3 ROCm: hidden readback";
    return false;
  }
  return true;
}

}  // namespace gufo::models::ling3::rocm

#else  // !LING3_WITH_ROCM

namespace gufo::models::ling3::rocm {
namespace {
struct Plan {
  bool ready{false};
};
Plan g;
}  // namespace

bool Init(const ModelWeights&, const core::GgufReader&, std::size_t* resident,
          std::string* error) {
  if (resident != nullptr) {
    *resident = 0;
  }
  if (error != nullptr) {
    *error =
        "this build has no ROCm backend for ling3; use "
        "BackendKind::kCpuReference";
  }
  return false;
}

bool ExecutorReady() noexcept {
  return false;
}
std::string MissingPieces() {
  return "this build has no ROCm backend for ling3";
}
State* NewState() {
  return nullptr;
}
void DestroyState(State*) noexcept {}
bool Step(std::int32_t, State*, std::span<float>, std::string*) {
  return false;
}
// LS3: batched prefill refuses without the HIP backend.
bool ForwardBatch(const std::int32_t*, int, State*, std::span<float>,
                  std::string*, int, std::span<float>, std::span<float>) {
  return false;
}
bool StepMulti(const std::int32_t*, int, State*, std::span<float>,
               std::string*) {
  return false;
}
bool StepMultiDispatch(const std::int32_t*, int, State*, std::span<float>,
                       std::string*) {
  return false;
}
bool SharedTensorsView(SharedTensors*) noexcept {
  return false;
}
bool ReadHidden(const State*, std::span<float>, std::string*) {
  return false;
}
bool BatchCapable() noexcept {
  return false;
}

}  // namespace gufo::models::ling3::rocm

#endif  // LING3_WITH_ROCM
