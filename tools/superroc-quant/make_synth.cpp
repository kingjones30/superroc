// make_synth — build a synthetic BF16 GGUF source with a full model tensor
// roster (separate attn_q/k/v/output, ffn_up/gate/down, token_embd, output,
// norms, MTP eh_proj, router ffn_gate_inp) across `n_layer` layers, so the
// recipe ladder's layer-position branches (use_more_bits, ffn_down 2/3 split,
// embed Q5_K/Q6_K, MTP->Q8_0, experts) are all exercised. KV (arch, hparams)
// is copied from a template GGUF; block_count is overridden to n_layer.
// Tensor data is synthetic BF16: normal, zeros, denormals, huge values, and
// exact ties (to probe the nearest-code tie-break and UE4M3 saturation).
//
// Usage: make_synth template.gguf out.gguf n_layer [seed]

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "gguf.hpp"

namespace gqt = superroc::quant_tool;

static std::uint32_t g_rand_state = 12345;
static std::uint32_t rng() {
  g_rand_state = g_rand_state * 1664525u + 1013904223u;
  return g_rand_state;
}
static float rf() {  // ~normal-ish in [-4,4]
  float s = 0; for (int i = 0; i < 4; ++i) s += (rng() % 1000) / 1000.0f - 0.5f;
  return s * 6.0f;
}

static void fill(std::vector<std::uint8_t>& raw, std::uint64_t nelem, std::uint64_t seed,
                 int kind) {
  (void)seed;
  raw.resize(nelem * 2);
  std::uint16_t* q = (std::uint16_t*)raw.data();
  for (std::uint64_t i = 0; i < nelem; ++i) {
    float v = 0;
    switch (kind) {
      case 0: v = rf(); break;
      case 1: v = 0.0f; break;                                              // zeros
      case 2: v = (i % 2) ? 1.0f : -1.0f; break;                            // exact ties / symmetric
      case 3: v = (i % 3) ? rf() : (float)((i % 11) + 1) * 1.0e3f; break;  // huge + normal
      case 4: v = (i % 5) ? rf() : 1.0e-42f * ((i % 2) ? 1 : -1); break;    // denormals
      default: v = rf(); break;
    }
    // BF16: store as u32 << 16 truncated
    std::uint32_t u; std::memcpy(&u, &v, 4);
    std::uint32_t b = u & 0xFFFF0000u;                    // round-to-zero to BF16
    std::uint16_t h = (std::uint16_t)(b >> 16);
    q[i] = h;
  }
}

gqt::GgufTensor MakeTensor(const std::string& name, std::uint64_t ne0, std::uint64_t ne1,
                           std::uint64_t ne3 /*experts*/, int kind) {
  gqt::GgufTensor t;
  t.name = name;
  t.type = (gqt::Type)30;  // BF16
  t.dims = {ne0, ne1, 1, ne3};
  std::uint64_t nelem = ne0 * ne1 * ne3;
  fill(t.raw, nelem, name.size(), kind);
  t.size_bytes = t.raw.size();
  return t;
}

int main(int argc, char** argv) {
  if (argc < 4) { std::cerr << "usage: make_synth template.gguf out.gguf n_layer [seed]\n"; return 2; }
  std::string tpl = argv[1], out = argv[2];
  int n_layer = std::atoi(argv[3]);
  if (argc > 4) g_rand_state = (std::uint32_t)std::atoi(argv[4]);

  gqt::GgufFile src = gqt::LoadGguf(tpl);

  gqt::GgufFile o;
  o.alignment = src.alignment;
  // copy KV, override block_count to n_layer, drop split/quant kv
  for (auto& [k, v] : src.kv) {
    if (k == "general.split.no" || k == "general.split.count" ||
        k == "general.split.tensors_count" || k == "general.quantization_version" ||
        k == "split.no" || k == "split.count" || k == "split.tensors.count") continue;
    if (k.find("block_count") != std::string::npos) {
      // override value to n_layer
      gqt::KvValue nv = v;
      if (nv.scalar_bytes.size() >= 4) { std::uint32_t x = (std::uint32_t)n_layer; nv.scalar_bytes.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 4); }
      o.kv.emplace_back(k, nv);
      continue;
    }
    o.kv.emplace_back(k, v);
  }

  // ---- full roster across n_layer layers ----
  std::uint64_t n_embd = 256, n_head_kv = 8, n_ff = 1024, n_expert = 4;
  for (int l = 0; l < n_layer; ++l) {
    std::string pre = "blk." + std::to_string(l) + ".";
    o.tensors.push_back(MakeTensor(pre + "attn_norm.weight", n_embd, 1, 1, 0));
    o.tensors.push_back(MakeTensor(pre + "attn_q_norm.weight", 32, 1, 1, 0));
    o.tensors.push_back(MakeTensor(pre + "attn_k_norm.weight", 32, 1, 1, 0));
    o.tensors.push_back(MakeTensor(pre + "attn_q.weight", n_embd, n_embd, 1, 0));
    o.tensors.push_back(MakeTensor(pre + "attn_k.weight", n_embd, n_head_kv * 32, 1, 0));
    o.tensors.push_back(MakeTensor(pre + "attn_v.weight", n_embd, n_head_kv * 32, 1, 0));
    o.tensors.push_back(MakeTensor(pre + "attn_output.weight", n_embd, n_embd, 1, 0));
    o.tensors.push_back(MakeTensor(pre + "post_attention_norm.weight", n_embd, 1, 1, 0));
    // ffn (with use_more_bits layer variation -> different type selection)
    o.tensors.push_back(MakeTensor(pre + "ffn_gate_inp.weight", n_embd, n_expert, 1, 0));  // router stays F32? BF16 here
    o.tensors.push_back(MakeTensor(pre + "ffn_up.weight", n_embd, n_ff, 1, (l % 3) + 1));
    o.tensors.push_back(MakeTensor(pre + "ffn_gate.weight", n_embd, n_ff, 1, (l % 3)));
    o.tensors.push_back(MakeTensor(pre + "ffn_down.weight", n_ff, n_embd, 1, (l % 4)));
  }
  o.tensors.push_back(MakeTensor("token_embd.weight", n_embd, 128, 1, 0));
  o.tensors.push_back(MakeTensor("output_norm.weight", n_embd, 1, 1, 0));
  o.tensors.push_back(MakeTensor("output.weight", n_embd, 128, 1, 0));
  // MTP projection (draft-sensitive -> Q8_0 floor)
  o.tensors.push_back(MakeTensor("blk.0.nextn.eh_proj.weight", n_embd, 16, 1, 0));

  std::vector<gqt::Type> types(o.tensors.size(), (gqt::Type)30);
  std::vector<std::vector<std::uint8_t>> payloads(o.tensors.size());
  for (size_t i = 0; i < o.tensors.size(); ++i) { types[i] = o.tensors[i].type; payloads[i] = o.tensors[i].raw; }
  std::vector<gqt::ExtraKv> extra;
  gqt::WriteGguf(out, o, types, payloads, 32, extra);
  std::cout << "wrote " << out << " with " << o.tensors.size() << " tensors, " << n_layer << " layers\n";
  return 0;
}
