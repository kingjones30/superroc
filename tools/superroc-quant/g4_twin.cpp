// g4_twin — build a G4 (120/121) twin of a ROCmFP4 (100/101) GGUF and verify
// that every tensor decodes to bit-identical f32 (SPEC2 item 3 proof).
//
// The G4 twin keeps the same codes/scales as its 100/101 source, so
// decode(G4 twin) == decode(100/101) bit-for-bit. This tool:
//   1. reads a 100/101 quantized GGUF
//   2. writes its G4 twin (100->120, 101->121) using the split-scale-plane
//      layout, carrying codes/scales byte-for-byte
//   3. decodes both files tensor-by-tensor and compares f32 bit patterns
//      (memcmp of the f32 output), reporting any mismatch
//
// Usage: g4_twin in.gguf out-twin.gguf   (in must contain only 100/101 tensors)

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "g4_encode.hpp"
#include "gguf.hpp"

namespace gqt = superroc::quant_tool;

int main(int argc, char** argv) {
  if (argc < 3) { std::cerr << "usage: g4_twin in.gguf out-twin.gguf\n"; return 2; }
  gqt::GgufFile f = gqt::LoadGguf(argv[1]);
  std::string out = argv[2];

  std::vector<gqt::Type> ntypes;
  std::vector<std::vector<std::uint8_t>> payloads;
  for (auto& t : f.tensors) {
    gqt::Type src = t.type;
    gqt::Type dst;
    bool is_roc = (src == (gqt::Type)100 || src == (gqt::Type)101);
    if (src == (gqt::Type)100) dst = (gqt::Type)120;
    else if (src == (gqt::Type)101) dst = (gqt::Type)121;
    else { dst = src; }  // non-ROC tensor: keep type + bytes as-is (skipped in twin check)
    ntypes.push_back(dst);

    if (!is_roc) {
      payloads.push_back(t.raw);
      continue;
    }
    // Rebuild the payload in the split layout from the source blocks.
    std::uint64_t ne0 = t.dims[0], nrows = 1;
    for (size_t d = 1; d < t.dims.size(); ++d) nrows *= t.dims[d];
    std::uint64_t nb = ne0 / 32;
    std::uint64_t rowb = (src == (gqt::Type)100) ? nb * 18 : nb * 17;
    std::uint64_t new_rowb = (src == (gqt::Type)100) ? nb * 16 + nb * 2 : nb * 16 + nb * 1;
    std::vector<std::uint8_t> nrow(new_rowb);
    payloads.emplace_back();
    payloads.back().resize(new_rowb * nrows);
    for (std::uint64_t r = 0; r < nrows; ++r) {
      const std::uint8_t* srow = t.raw.data() + r * rowb;
      std::uint8_t* drow = payloads.back().data() + r * new_rowb;
      // split plane
      if (src == (gqt::Type)100) {
        for (std::uint64_t ib = 0; ib < nb; ++ib) {
          const gqt::block_rocmfp4* b = (const gqt::block_rocmfp4*)(srow + ib * 18);
          std::memcpy(drow + ib * 16, b->qs, 16);
          std::memcpy(drow + nb * 16 + ib * 2, b->e, 2);
        }
      } else {
        for (std::uint64_t ib = 0; ib < nb; ++ib) {
          const gqt::block_rocmfp4_fast* b = (const gqt::block_rocmfp4_fast*)(srow + ib * 17);
          std::memcpy(drow + ib * 16, b->qs, 16);
          std::memcpy(drow + nb * 16 + ib, &b->e, 1);
        }
      }
    }
  }

  std::vector<gqt::ExtraKv> extra;
  gqt::WriteGguf(out, f, ntypes, payloads, 105, extra);

  // ---- decode both and compare ----
  gqt::GgufFile tf = gqt::LoadGguf(out);
  bool ok = true;
  std::uint64_t twin_count = 0;
  for (size_t i = 0; i < f.tensors.size(); ++i) {
    auto& sa = f.tensors[i];
    auto& tb = tf.tensors[i];
    bool is_roc = (sa.type == (gqt::Type)100 || sa.type == (gqt::Type)101);
    if (!is_roc) continue;  // non-ROC tensors not part of the twin proof
    ++twin_count;
    std::uint64_t ne0 = sa.dims[0], nrows = 1;
    for (size_t d = 1; d < sa.dims.size(); ++d) nrows *= sa.dims[d];
    std::uint64_t nelem = ne0 * nrows;
    std::vector<float> da(nelem), db(nelem);
    for (std::uint64_t r = 0; r < nrows; ++r) {
      gqt::DecodeRowRocmFp4(sa.raw.data() + r * (sa.size_bytes / nrows), sa.type, (std::int64_t)ne0, da.data() + r * ne0);
      gqt::DecodeRowG4(tb.raw.data() + r * (tb.size_bytes / nrows), tb.type, (std::int64_t)ne0, db.data() + r * ne0);
    }
    if (std::memcmp(da.data(), db.data(), nelem * sizeof(float)) != 0) {
      std::cout << "MISMATCH " << sa.name << " (" << gqt::TypeName(sa.type) << "->" << gqt::TypeName(tb.type) << ")\n";
      ok = false;
    }
  }
  if (ok) {
    std::cout << "G4 TWIN OK: " << twin_count << " ROC tensors decode bit-identical to their 100/101 twins\n";
    return 0;
  }
  if (ok) {
    std::cout << "G4 TWIN OK: " << f.tensors.size() << " tensors decode bit-identical\n";
    return 0;
  }
  std::cout << "G4 TWIN MISMATCHES FOUND\n";
  return 1;
}
