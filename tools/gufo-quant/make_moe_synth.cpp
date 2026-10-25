// make_moe_synth — build a synthetic BF16 GGUF source that carries the REAL
// KV header and tensor roster of an MoE GGUF (read-only), so the recipe
// ladder's MoE-specific categories (ffn_*_exps 3-D, ffn_*_shexp, ffn_gate_inp
// router, MLA attn latent tensors, KDA tensors, MTP/NextN, hc_*, indexer,
// ple_*, ...) are all exercised as they are in King's real models.
//
// SPEC2 item 5: prove all 7 recipes MATCH the fork on two such fixtures:
//   (a) King's Ling-3.0-flash (arch bailingmoe3, 512 experts)
//   (b) Qwen3.8-Flash-Next (the gufo qwen38_flash_next / qwen4exp family)
//
// The fixture keeps the real tensor NAMES and the real ne[0] (so the recipe
// ladder's shape-fallback/divisibility behaves exactly as the full model),
// but shrinks ne[1]/ne[2] to a cap so the file stays small on a 99%-full disk.
// The fork's ROC recipe ladder does NOT branch on ne[1]/ne[2] or n_expert
// (SPEC-LOCK §2 — the ROC branches never read n_expert; only non-ROC ftypes
// do), so the per-tensor chosen type is identical to the full model. We keep
// n_expert at the real count (512) because some non-ROC branches test ==8.
//
// KV: copied from the real header verbatim except block_count is overridden to
// the kept layer count (SPEC item 5: "keep the hparams consistent — block_count
// = the layers you keep; per-layer arrays trimmed to match") and split/quant
// provenance keys are dropped. Tensors: global (non-blk) kept as-is (dims
// capped), blk.N kept for N in [0, n_layer), dims capped.
//
// Usage:
//   make_moe_synth source.gguf out-bf16.gguf n_layer [ne1_cap] [ne2_cap] [seed]

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "gguf.hpp"

namespace gqt = gufo::quant_tool;

static std::uint32_t g_rand_state = 12345;
static std::uint32_t rng() { g_rand_state = g_rand_state * 1664525u + 1013904223u; return g_rand_state; }
static float rf() {
  float s = 0; for (int i = 0; i < 4; ++i) s += (rng() % 1000) / 1000.0f - 0.5f;
  return s * 6.0f;
}
static void fill(std::vector<std::uint8_t>& raw, std::uint64_t nelem, int kind) {
  raw.resize(nelem * 2);
  std::uint16_t* q = (std::uint16_t*)raw.data();
  for (std::uint64_t i = 0; i < nelem; ++i) {
    float v = 0;
    switch (kind) {
      case 0: v = rf(); break;
      case 1: v = 0.0f; break;
      case 2: v = (i % 2) ? 1.0f : -1.0f; break;
      case 3: v = (i % 3) ? rf() : (float)((i % 11) + 1) * 1.0e3f; break;
      case 4: v = (i % 5) ? rf() : 1.0e-42f * ((i % 2) ? 1 : -1); break;
      default: v = rf(); break;
    }
    std::uint32_t u; std::memcpy(&u, &v, 4);
    std::uint16_t h = (std::uint16_t)((u & 0xFFFF0000u) >> 16);
    q[i] = h;
  }
}

// Parse `blk.<N>.` from a tensor name; -1 if none.
static int layer_idx(const std::string& name) {
  const std::string pat = "blk.";
  auto p = name.find(pat);
  if (p == std::string::npos) return -1;
  size_t q = p + 4; long v = 0; bool any = false;
  while (q < name.size() && name[q] >= '0' && name[q] <= '9') { v = v * 10 + (name[q] - '0'); any = true; ++q; }
  return any ? (int)v : -1;
}

int main(int argc, char** argv) {
  // usage: make_moe_synth out-bf16.gguf n_layer [ne1_cap] [ne2_cap] [seed] src1 [src2 ...]
  // (Sources may be multi-part splits; src1 supplies the KV, all supply tensors.)
  if (argc < 6) { std::cerr << "usage: make_moe_synth out-bf16.gguf n_layer [ne1_cap] [ne2_cap] [seed] src1 [src2 ...]\n"; return 2; }
  std::string out = argv[1];
  int n_layer = std::atoi(argv[2]);
  std::uint64_t ne1_cap = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 16;
  std::uint64_t ne2_cap = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 8;
  std::vector<std::string> srcs(argv + 6, argv + argc);

  // Header-only: the real MoE GGUFs are multi-GB (42G Ling, 99G Qwen); only the
  // roster + KV are needed, never the payload bytes. First source = KV template.
  // load_sizes=false: some real tensors carry types we don't need to nail down
  // (e.g. per_layer_token_embd type 7); we force BF16 sources anyway.
  gqt::GgufFile src = gqt::LoadGgufHeaderOnly(srcs[0], /*load_sizes=*/false);

  gqt::GgufFile o;
  o.alignment = 32;
  // Find the SOURCE's real block_count (for per-layer array trim below) before
  // we override it. Match only the exact `<arch>.block_count` leaf (last
  // dot-segment == "block_count"), NOT leading_dense_block_count / similar.
  auto is_block_count_leaf = [](const std::string& k) {
    auto d = k.rfind('.');
    return d != std::string::npos && k.compare(d + 1, std::string::npos, "block_count") == 0;
  };
  std::uint32_t src_block_count = 0;
  for (auto& [k, v] : src.kv) {
    if (is_block_count_leaf(k) && v.type == gqt::ValType::kUint32 &&
        v.scalar_bytes.size() >= 4) {
      std::memcpy(&src_block_count, v.scalar_bytes.data(), 4);
    }
  }
  // Copy KV, dropping split/quant provenance; override block_count keys.
  bool have_block_count = false;
  for (auto& [k, v] : src.kv) {
    if (k == "general.split.no" || k == "general.split.count" ||
        k == "general.split.tensors_count" || k == "general.quantization_version" ||
        k == "split.no" || k == "split.count" || k == "split.tensors.count") continue;
    if (is_block_count_leaf(k)) {
      gqt::KvValue nv = v;
      if (nv.scalar_bytes.size() >= 4) { std::uint32_t x = (std::uint32_t)n_layer; nv.scalar_bytes.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 4); }
      o.kv.emplace_back(k, nv);
      have_block_count = true;
      continue;
    }
    // Per-layer ARRAY KV (attention.head_count_kv, swiglu_clamp_exp, etc.) must
    // be trimmed to the kept layer count, or the fork's loader rejects the
    // fixture ("wrong array length; expected <n_layer>, got <block_count>").
    if (v.type == gqt::ValType::kArray && src_block_count > 0 &&
        (int)v.array_elems.size() == (int)src_block_count) {
      gqt::KvValue nv = v;
      nv.array_elems.resize((size_t)n_layer);
      o.kv.emplace_back(k, nv);
      continue;
    }
    o.kv.emplace_back(k, v);
  }
  if (!have_block_count) {
    gqt::KvValue nv; nv.type = gqt::ValType::kUint32;
    std::uint32_t x = (std::uint32_t)n_layer; nv.scalar_bytes.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 4);
    o.kv.emplace_back("general.block_count", nv);
  }

  // Build the tensor list: keep global tensors + blk.[0,n_layer) tensors,
  // cap ne[1]/ne[2], keep ne[0] exactly (drives the shape fallback like the
  // full model). Fill BF16 edge-value data (kind varies by rank/name).
  std::vector<gqt::Type> types;
  std::vector<std::vector<std::uint8_t>> payloads;
  auto cap_dims = [&](const std::vector<std::uint64_t>& dims, std::vector<std::uint64_t>& out) {
    out = dims;
    if (out.size() >= 2) out[1] = std::min<std::uint64_t>(out[1], ne1_cap);
    if (out.size() >= 3) out[2] = std::min<std::uint64_t>(out[2], ne2_cap);
    if (out.size() >= 4) out[3] = 1;
    // 1-D tensors (norms/router shexp etc.) keep ne[0] (or small).
  };
  std::uint64_t total_elems = 0;
  std::set<std::string> seen;
  auto handle_tensor = [&](const gqt::GgufTensor& t) {
    if (!seen.insert(t.name).second) return;  // dedup across split parts
    int li = layer_idx(t.name);
    if (li < 0) {
      // global tensor: keep
    } else if (li >= n_layer) {
      return;  // drop layers beyond the fixture's block_count
    }
    std::vector<std::uint64_t> dims;
    cap_dims(t.dims, dims);
    gqt::GgufTensor nt;
    nt.name = t.name;
    nt.type = (gqt::Type)30;  // BF16 source
    nt.dims = dims;
    std::uint64_t nelem = 1;
    for (auto d : dims) nelem *= d;
    // Add block_count to patch: no per-layer KV array trim needed for Ling/Qwen
    // (their block-related KV are scalars; the only per-array ones are tensors
    // which we trim above).
    int kind = 0;
    if (dims.size() == 1) kind = 1;               // norms / 1-d
    else if (nt.name.find("_exps") != std::string::npos) kind = 0;
    else if (nt.name.find("_shexp") != std::string::npos) kind = 2;
    else if (nt.name.find("ssm_") != std::string::npos) kind = 4;
    fill(nt.raw, nelem, kind);
    nt.size_bytes = nt.raw.size();
    total_elems += nelem;
    types.push_back((gqt::Type)30);
    payloads.push_back(nt.raw);
    o.tensors.push_back(std::move(nt));
  };
  // Iterate all source parts (merged roster), dedup by name.
  for (auto& sp : srcs) {
    gqt::GgufFile part = gqt::LoadGgufHeaderOnly(sp, /*load_sizes=*/false);
    for (auto& t : part.tensors) handle_tensor(t);
  }

  std::vector<gqt::ExtraKv> extra;
  gqt::WriteGguf(out, o, types, payloads, 32, extra);
  std::cout << "wrote " << out << " with " << o.tensors.size() << " tensors, "
            << n_layer << " layers, " << (total_elems * 2 / 1024 / 1024) << " MB\n";
  return 0;
}
