// make_imatrix — build a minimal GGUF imatrix carrying one per-tensor row for
// every weight tensor in a source GGUF, so llama-quantize sees a valid imatrix
// (keys "imatrix.<tensor>", array of f32, size ne0*ne2). Used to prove that an
// imatrix changes no byte for the ROC recipes.
// Usage: make_imatrix in.gguf out-imatrix.gguf

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "gguf.hpp"
#include "vendored_quantize.hpp"  // not needed for float fill

namespace gqt = gufo::quant_tool;

int main(int argc, char** argv) {
  if (argc < 3) { std::cerr << "usage: make_imatrix in.gguf out.gguf\n"; return 2; }
  gqt::GgufFile src = gqt::LoadGguf(argv[1]);
  gqt::GgufFile o;
  o.alignment = src.alignment;
  // copy general KV
  for (auto& [k, v] : src.kv) o.kv.push_back({k, v});
  // add an "imatrix.<tensor>" array key per weight tensor
  for (auto& t : src.tensors) {
    if (t.name.find("weight") == std::string::npos) continue;
    std::uint64_t ne0 = t.dims[0];
    std::uint64_t len = t.dims.size() >= 3 ? ne0 * t.dims[2] : ne0;
    gqt::KvValue kv;
    kv.type = gqt::ValType::kArray;
    kv.array_type = gqt::ValType::kFloat32;
    for (std::uint64_t i = 0; i < len; ++i) {
      std::uint8_t fb[4]; float v = 1.0f; std::memcpy(fb, &v, 4);
      kv.array_elems.push_back(std::vector<std::uint8_t>(fb, fb + 4));
    }
    o.kv.push_back({"imatrix." + t.name, kv});
  }
  // imatrix GGUF has no tensors; just KV. Write with 0 tensors.
  std::vector<gqt::Type> types;
  std::vector<std::vector<std::uint8_t>> payloads;
  std::vector<gqt::ExtraKv> extra;
  gqt::WriteGguf(argv[2], o, types, payloads, 32, extra);
  std::cout << "wrote imatrix " << argv[2] << " with " << o.kv.size() << " kv\n";
  return 0;
}
