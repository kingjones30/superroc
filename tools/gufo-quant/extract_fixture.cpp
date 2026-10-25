// extract_fixture — copy KV + a chosen SMALL subset of tensors out of a GGUF
// into a new small non-split GGUF (preserving fork-required metadata).
//
// Keeps: token_embd, output, norms, routers (small), plus the smallest
// attention tensors per layer so the recipe ladder has real categories. Drops
// huge FFN/expert tensors and split metadata. Usage:
//   extract_fixture in.gguf out.gguf N_MB

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "gguf.hpp"

namespace gqt = gufo::quant_tool;

int main(int argc, char** argv) {
  if (argc < 4) { std::cerr << "usage: extract_fixture in.gguf out.gguf N_MB\n"; return 2; }
  std::string in = argv[1], out = argv[2];
  std::uint64_t budget_mb = (std::uint64_t)std::atoll(argv[3]);
  gqt::GgufFile f = gqt::LoadGguf(in);
  std::cout << "input tensors: " << f.tensors.size() << "\n";

  // strip split KV
  gqt::GgufFile o;
  o.alignment = f.alignment;
  for (auto& [k, v] : f.kv) {
    if (k == "general.split.no" || k == "general.split.count" ||
        k == "general.split.tensors_count" || k == "split.no" || k == "split.count" ||
        k == "split.tensors.count") continue;
    o.kv.emplace_back(k, v);
  }
  // remove general.file_type? keep (it's the source type). The fork resets it.

  // pick tensors: small ones first, prefer meaningful categories
  struct E { size_t idx; std::uint64_t bytes; std::string name; };
  std::vector<E> cand;
  for (size_t i = 0; i < f.tensors.size(); ++i) {
    auto& t = f.tensors[i];
    cand.push_back({i, t.size_bytes, t.name});
  }
  std::sort(cand.begin(), cand.end(), [](const E& a, const E& b) { return a.bytes < b.bytes; });

  std::uint64_t total = 0;
  std::set<std::string> kept;
  int layer0_ffn = 0;
  for (auto& e : cand) {
    // always keep specials; keep blk tensors that are attention (small)
    bool is_special = e.name == "token_embd.weight" || e.name == "output.weight" ||
                      e.name == "output_norm.weight" || e.name == "token_embd_norm.weight" ||
                      e.name.find("ffn_gate_inp") != std::string::npos;
    bool is_attn = e.name.find("attn_") != std::string::npos;
    if (!is_special && !is_attn) continue;  // skip norms/ffn/expert within blk
    // keep it if under budget
    if (total + e.bytes > budget_mb * 1024 * 1024) continue;
    kept.insert(e.name);
    total += e.bytes;
  }
  // ensure layer 0 also has a couple FFN tensors for category coverage
  for (auto& e : cand) {
    if (kept.size() >= 40) break;
    if (kept.count(e.name)) continue;
    if (e.name.find("blk.0.ffn") == std::string::npos) continue;
    if (total + e.bytes > budget_mb * 1024 * 1024) continue;
    kept.insert(e.name); total += e.bytes;
  }

  for (auto& t : f.tensors) {
    if (kept.count(t.name)) o.tensors.push_back(t);
  }
  std::cout << "kept tensors: " << o.tensors.size() << " (" << (total / 1024 / 1024) << " MB)\n";

  std::vector<gqt::Type> types(o.tensors.size());
  std::vector<std::vector<std::uint8_t>> payloads(o.tensors.size());
  for (size_t i = 0; i < o.tensors.size(); ++i) {
    types[i] = o.tensors[i].type;
    payloads[i] = o.tensors[i].raw;
  }
  std::vector<gqt::ExtraKv> extra;
  gqt::WriteGguf(out, o, types, payloads, 32, extra);
  std::cout << "wrote " << out << "\n";
  return 0;
}
