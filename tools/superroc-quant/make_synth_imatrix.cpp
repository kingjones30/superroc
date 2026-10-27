// make_synth_imatrix — build a proper fork-format imatrix GGUF (tensors
// `<name>.in_sum2` + `<name>.counts`, the format llama-imatrix writes and the
// fork's load_imatrix reads) for every quantizable weight tensor in a source
// GGUF, with deterministic synthetic importance. Used to validate superroc-quant's
// weighted encoder against the fork's `llama-quantize --imatrix`.
//
// For each weight tensor [ne0, ne1, ne2]: writes a sums tensor of ne0*ne2 f32
// and a counts tensor of ne2 f32 (count = 1 per expert) so weights are
// sums/1. The synthetic sum for column i of expert j is a deterministic
// pseudo-random positive value, so weighting is active and stable.
//
// Usage: make_synth_imatrix in.gguf out-imatrix.gguf [--uniform]
//   Entries ALWAYS use the base tensor name from the source (no .in_sum2/
//   .counts), matching load_imatrix. --uniform makes all importance = 1.0
//   (for isolating the fork's energy-weighting behavior).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "gguf.hpp"

namespace gqt = superroc::quant_tool;

// Deterministic synthetic importance in [0.25, 4.0] per column.
static float synth_importance(std::uint64_t seed) {
  std::uint64_t x = seed * 2654435761u + 0x9e3779b9u;
  x ^= x >> 17; x *= 0x85ebca6bu; x ^= x >> 13;
  const float r = (float)(x & 0xffffff) / (float)0xffffff;  // [0,1]
  return 0.25f + 3.75f * r;
}

int main(int argc, char** argv) {
  bool uniform = false;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--uniform") uniform = true;
    else pos.push_back(argv[i]);
  }
  if (pos.size() < 2) { std::cerr << "usage: make_synth_imatrix in.gguf out.gguf [--uniform]\n"; return 2; }
  gqt::GgufFile src = gqt::LoadGguf(pos[0]);
  gqt::GgufFile o;
  o.alignment = src.alignment;
  // Reuse the source KV (general.* etc. so the fork loader is happy) with no
  // quantize imatrix keys; we write tensors instead (the t2 GGUF format).
  for (auto& [k, v] : src.kv) o.kv.push_back({k, v});
  // Add imatrix metadata (the loader requires them).
  auto add_kv = [&](const std::string& k, std::uint32_t v) {
    gqt::KvValue kv; kv.type = gqt::ValType::kUint32;
    kv.scalar_bytes.assign((std::uint8_t*)&v, (std::uint8_t*)&v + 4);
    o.kv.push_back({k, kv});
  };
  add_kv("imatrix.chunk_size", 512);
  add_kv("imatrix.chunk_count", 1);
  {
    gqt::KvValue kv; kv.type = gqt::ValType::kArray;
    kv.array_type = gqt::ValType::kString;
    kv.array_strs.push_back("synthetic-calibration");
    o.kv.push_back({"imatrix.datasets", kv});
  }

  std::uint64_t seed = 0;
  for (auto& t : src.tensors) {
    const std::string& nm = t.name;
    if (nm.find("weight") == std::string::npos) continue;
    if (t.dims.empty() || t.dims[0] == 0) continue;
    const std::uint64_t ne0 = t.dims[0];
    const std::uint64_t ne2 = t.dims.size() > 2 ? t.dims[2] : 1;
    const std::uint64_t len = ne0 * ne2;

    // sums tensor: ne0 x ne2 f32
    gqt::GgufTensor sums;
    sums.name = nm + ".in_sum2";
    sums.dims = {ne0, ne2};
    sums.type = (gqt::Type)0;  // F32
    sums.raw.resize(len * 4);
    // counts tensor: ne2 f32 (=1 each)
    gqt::GgufTensor counts;
    counts.name = nm + ".counts";
    counts.dims = {ne2};
    counts.type = (gqt::Type)0;
    counts.raw.resize(ne2 * 4);

    for (std::uint64_t e = 0; e < ne2; ++e) {
      for (std::uint64_t i = 0; i < ne0; ++i) {
        float v = uniform ? 1.0f : synth_importance(seed++);
        float cnt = 1.0f;
        std::memcpy(sums.raw.data() + (e * ne0 + i) * 4, &v, 4);
        if (counts.raw.size() >= 4) std::memcpy(counts.raw.data(), &cnt, 4);
      }
      float cnt = 1.0f;
      std::memcpy(counts.raw.data() + e * 4, &cnt, 4);
    }
    o.tensors.push_back(std::move(sums));
    o.tensors.push_back(std::move(counts));
  }

  std::vector<gqt::Type> types(o.tensors.size(), (gqt::Type)0);
  std::vector<std::vector<std::uint8_t>> payloads;
  for (auto& t : o.tensors) payloads.push_back(t.raw);
  std::vector<gqt::ExtraKv> extra;
  gqt::WriteGguf(pos[1], o, types, payloads, 32, extra);
  std::cout << "wrote imatrix " << pos[1] << " (" << o.tensors.size() << " tensors)\n";
  return 0;
}
