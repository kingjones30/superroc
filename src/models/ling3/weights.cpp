#include "src/models/ling3/weights.hpp"

#include <string>
#include <utility>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::ling3 {
namespace {

using core::GgmlType;

/// Storage types this runtime knows how to decode. Ling's weights are
/// ROCmFP4 (100/101), F32 for the router and the KDA gate vectors, and Q8_0
/// for the MTP block. The two wide tensors are artifact-specific: the text
/// file carries a ROCmFP4 LM head and a Q6_K token_embd, while the VL file
/// carries a Q6_K LM head and a Q5_K token_embd, so both K-quants are bound
/// here and the exact type travels with the tensor.
bool IsSupportedType(GgmlType type) {
  switch (type) {
    case GgmlType::kF32:
    case GgmlType::kF16:
    case GgmlType::kBF16:
    case GgmlType::kQ8_0:
    case GgmlType::kQ5_K:
    case GgmlType::kQ6_K:
    case GgmlType::kQ4_0_ROCMFP4:
    case GgmlType::kQ4_0_ROCMFP4_FAST:
      return true;
    default:
      return false;
  }
}

struct Binder {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  /// Binds `name` and requires it to have exactly the given dimensions, with
  /// the innermost dimension first (GGUF ne order).
  TensorRef Get(std::string_view name,
                std::initializer_list<std::uint64_t> ne) {
    const auto* info = gguf.FindTensor(name);
    if (info == nullptr) {
      Fail("missing tensor " + std::string(name));
      return {};
    }
    if (!IsSupportedType(info->type)) {
      Fail(std::string(name) + " has unsupported type " +
           std::string(core::ToString(info->type)));
      return {};
    }
    if (info->dimensions.size() != ne.size()) {
      Fail(std::string(name) + " has " +
           std::to_string(info->dimensions.size()) + " dimensions, expected " +
           std::to_string(ne.size()));
      return {};
    }
    std::uint64_t i = 0;
    for (const auto want : ne) {
      if (info->dimensions[i] != want) {
        Fail(std::string(name) + " dimension " + std::to_string(i) + " is " +
             std::to_string(info->dimensions[i]) + ", expected " +
             std::to_string(want));
        return {};
      }
      ++i;
    }
    TensorRef ref;
    ref.data = info->data;
    ref.type = info->type;
    ref.cols = info->dimensions[0];
    ref.rows = ne.size() > 1 ? info->dimensions[1] : 1;
    ref.experts = ne.size() > 2 ? info->dimensions[2] : 1;
    ref.name = info->name;
    return ref;
  }

  /// Binds a 1-D parameter.
  VectorRef Vec(std::string_view name, std::uint64_t n) {
    VectorRef v;
    v.tensor = Get(name, {n});
    return v;
  }

  /// Re-views a validated [kc, 1, inner] conv weight as the 2-D [kc, inner]
  /// matrix it already is, so `DequantizeRow(t, 0, d, taps)` yields the kc taps
  /// for channel d and taps[c] is conv_w[c][d].
  ///
  /// GGUF ne order puts ne[0] fastest, so element (c, 0, d) of [kc,1,inner] is
  /// at flat index c + 4*d -- which is exactly row d of a [kc, inner] matrix.
  /// Reading it the other way round (cols=inner) would both transpose the taps
  /// and, because RowBytes depends on cols, stride the rows wrongly.
  static TensorRef AsConvTaps(TensorRef t, std::uint64_t kc) {
    const std::uint64_t inner = t.ElementCount() / kc;
    t.cols = kc;
    t.rows = inner;
    t.experts = 1;
    return t;
  }

  /// Re-views a validated multi-dimensional tensor as a flat 1-D vector of
  /// `n` elements. Used for ssm_a, which is stored as [1, n_head].
  static TensorRef AsFlatVector(TensorRef t, std::uint64_t n) {
    t.cols = n;
    t.rows = 1;
    t.experts = 1;
    return t;
  }
};

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  // EncodedSizeBytes covers the dense types (F32/F16/BF16) directly and falls
  // through to QuantizedRowBytes for the block formats, returning 0 when the
  // element count is not block aligned.
  return quant::EncodedSizeBytes(type, static_cast<std::size_t>(cols));
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& gguf,
                                               std::string* error_msg) {
  auto config = Config::FromGguf(gguf, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  const Config& c = *config;
  Binder b{gguf, error_msg, true};

  const std::uint64_t H = c.hidden_size;
  const std::uint64_t inner = c.KdaInnerSize();
  const std::uint64_t hd = c.kda_head_dim;
  const std::uint64_t heads = c.num_heads;
  const std::uint64_t kv = c.mla_kv_lora_rank;
  const std::uint64_t dk = c.mla_head_dim_k;
  const std::uint64_t dv = c.mla_head_dim_v;
  const std::uint64_t nope = c.MlaQkNopeDim();
  const std::uint64_t E = c.num_experts;
  const std::uint64_t ff = c.expert_ff;

  ModelWeights w;
  w.config = c;
  w.token_embd = b.Get("token_embd.weight", {H, c.vocab_size});
  w.output_norm = b.Vec("output_norm.weight", H);
  w.output = b.Get("output.weight", {H, c.vocab_size});

  auto bind_layer = [&](std::uint32_t i, bool mla, LayerWeights* out) {
    const std::string p = "blk." + std::to_string(i) + ".";
    out->mla = mla;
    out->attn_norm = b.Vec(p + "attn_norm.weight", H);
    out->ffn_norm = b.Vec(p + "ffn_norm.weight", H);

    if (mla) {
      MlaWeights& m = out->mla_weights;
      m.wq = b.Get(p + "attn_q.weight", {H, heads * dk});
      m.wkv_a_mqa = b.Get(p + "attn_kv_a_mqa.weight", {H, c.MlaKvAOutDim()});
      m.kv_a_norm = b.Vec(p + "attn_kv_a_norm.weight", kv);
      m.wk_b = b.Get(p + "attn_k_b.weight", {nope, kv, heads});
      m.wv_b = b.Get(p + "attn_v_b.weight", {kv, dv, heads});
      m.wqkv_gate = b.Get(p + "attn_gate.weight", {H, heads});
      m.wo = b.Get(p + "attn_output.weight", {heads * dv, H});
    } else {
      KdaWeights& k = out->kda;
      k.wq = b.Get(p + "attn_q.weight", {H, inner});
      k.wk = b.Get(p + "attn_k.weight", {H, inner});
      k.wv = b.Get(p + "attn_v.weight", {H, inner});
      // The converter writes the conv weight as [kernel, 1, inner]. The 2-D
      // fallback the fork also accepts is not produced by this converter and
      // has a transposed memory layout, so it is rejected rather than guessed.
      auto conv = [&](std::string_view name) {
        TensorRef t =
            b.Get(p + std::string(name), {c.kda_conv_kernel, 1, inner});
        if (t.empty()) {
          b.Fail(std::string(name) + " must have shape [" +
                 std::to_string(c.kda_conv_kernel) + ", 1, " +
                 std::to_string(inner) + "]; the 2-D form is not supported");
          return TensorRef{};
        }
        return Binder::AsConvTaps(t, c.kda_conv_kernel);
      };
      k.conv_q = conv("ssm_conv1d_q.weight");
      k.conv_k = conv("ssm_conv1d_k.weight");
      k.conv_v = conv("ssm_conv1d_v.weight");
      k.w_f = b.Get(p + "ssm_f.weight", {H, inner});
      k.w_g = b.Get(p + "ssm_g.weight", {H, inner});
      k.w_beta = b.Get(p + "ssm_beta.weight", {H, heads});
      // ssm_a is stored [1, n_head]; flatten it so one row reads all heads.
      {
        TensorRef a = b.Get(p + "ssm_a", {1, heads});
        if (a.empty()) {
          a = b.Get(p + "ssm_a", {heads});
        }
        k.a_log = a.empty() ? TensorRef{} : Binder::AsFlatVector(a, heads);
      }
      k.dt_bias = b.Vec(p + "ssm_dt.bias", inner);
      k.o_norm = b.Vec(p + "ssm_norm.weight", hd);
      k.wo = b.Get(p + "attn_output.weight", {inner, H});
    }

    if (i < c.dense_lead_layers) {
      out->ffn_gate = b.Get(p + "ffn_gate.weight", {H, c.ffn_length});
      out->ffn_up = b.Get(p + "ffn_up.weight", {H, c.ffn_length});
      out->ffn_down = b.Get(p + "ffn_down.weight", {c.ffn_length, H});
    } else {
      MoeWeights& m = out->moe;
      m.gate_inp = b.Get(p + "ffn_gate_inp.weight", {H, E});
      m.probs_bias = b.Vec(p + "exp_probs_b.bias", E);
      m.gate_exps = b.Get(p + "ffn_gate_exps.weight", {H, ff, E});
      m.up_exps = b.Get(p + "ffn_up_exps.weight", {H, ff, E});
      m.down_exps = b.Get(p + "ffn_down_exps.weight", {ff, H, E});
      m.gate_shexp =
          b.Get(p + "ffn_gate_shexp.weight", {H, c.shared_expert_ff});
      m.up_shexp = b.Get(p + "ffn_up_shexp.weight", {H, c.shared_expert_ff});
      m.down_shexp =
          b.Get(p + "ffn_down_shexp.weight", {c.shared_expert_ff, H});
    }
  };

  w.layers.resize(c.num_layers);
  for (std::uint32_t i = 0; i < c.num_layers; ++i) {
    bind_layer(i, c.IsMla(i), &w.layers[i]);
  }
  // The MTP block is the last one and is full attention; bind it so the
  // package can report a complete artifact, and bind its nextn.* glue so the
  // draft head in mtp_gpu.cpp can run. The trunk Step() never executes it.
  const std::uint32_t mtp_blk = c.block_count - 1;
  bind_layer(mtp_blk, true, &w.mtp);

  {
    const std::string p = "blk." + std::to_string(mtp_blk) + ".";
    MtpHeadWeights& mh = w.mtp_head;
    mh.eh_proj = b.Get(p + "nextn.eh_proj.weight", {2 * H, H});
    mh.enorm = b.Vec(p + "nextn.enorm.weight", H);
    mh.hnorm = b.Vec(p + "nextn.hnorm.weight", H);
    if (gguf.FindTensor(p + "layer_output_norm.weight") != nullptr) {
      mh.layer_out_norm = b.Vec(p + "layer_output_norm.weight", H);
    }
  }

  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

}  // namespace gufo::models::ling3
