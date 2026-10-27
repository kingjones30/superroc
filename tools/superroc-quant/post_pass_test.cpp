// post_pass_test — SPEC3B item 3 unit test: on every ROC block of a source GGUF,
// assert that the stored bytes of the weighted + post-pass encode have
// mse-weighted error <= the stored bytes of the weighted (fork) encode, and
// report how many blocks strictly improved.
//
// This is the empirical companion to RV29 §4 / the code-sweep being a provable
// no-op: with D8 the fork's exhaustive weighted scale search is the global
// argmin of the mse-weighted objective (per-element-separable nearest codes at
// the winning scale), so the post-pass (LS refit + code sweep) can never
// strictly improve and MUST tie. The test asserts the invariant from the ACTUAL
// STORED BYTES (SPEC3B item 3: "computed FROM THE STORED BYTES"), not from a
// re-derivation, and reports the improved-block count.
//
// Usage: post_pass_test <src.gguf> <imatrix.gguf> [recipe]
//   recipe defaults to STRIX (the four-arm quality recipe). Only ROC 100/101
//   tensors are scored. Exit 0 = invariant held (post_pass_error <= fork_error
//   on every block); exit 1 = a block violated it.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gguf.hpp"
#include "recipe.hpp"
#include "rocmfp4_encode.hpp"

namespace gqt = superroc::quant_tool;

namespace {

// BF16 (type 30) -> f32: BF16 keeps the top 16 bits of the f32 pattern.
inline float Bf16ToF32(std::uint16_t h) {
  std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Dequantize one tensor's payload to a row-major f32 vector. Supports the
// source types a BF16-export extract holds: F32 (0) and BF16 (30).
std::vector<float> DequantSrc(const gqt::GgufTensor& t, std::uint64_t nelem) {
  std::vector<float> out(nelem);
  if (t.type == (gqt::Type)30) {
    const std::uint16_t* p = (const std::uint16_t*)t.raw.data();
    for (std::uint64_t i = 0; i < nelem; ++i) out[i] = Bf16ToF32(p[i]);
  } else if (t.type == (gqt::Type)0) {
    std::memcpy(out.data(), t.raw.data(), nelem * sizeof(float));
  } else {
    throw std::runtime_error("post_pass_test: unsupported source type " +
                             std::to_string((int)t.type) + " for " + t.name);
  }
  return out;
}

// ROC codebook decode level for a nibble (rocmfp4_decode; SPEC-LOCK §1a).
inline float DecodeNibble(std::uint8_t nib) noexcept {
  const int q = nib & 7;
  const float mag = (q <= 4) ? (float)q : (float)(2 * q - 4);  // 0,1,2,3,4,6,8,10
  return (nib & 8) ? -mag : mag;
}

// Weighted MSE of a stored type-100 block computed FROM its stored bytes:
// decode qs under the stored e[0]/e[1] and compare to x, weighted by w.
inline float StoredErrType100(const float* __restrict x, const float* __restrict w,
                              const gqt::block_rocmfp4* b) noexcept {
  const float s0 = gqt::ScaleFromByte(b->e[0]);
  const float s1 = gqt::ScaleFromByte(b->e[1]);
  float err = 0.0f;
  for (int j = 0; j < 16; ++j) {
    const float d0 = x[j] - s0 * DecodeNibble(b->qs[j] & 0x0F);
    err += w[j] * d0 * d0;
    const float d1 = x[j + 16] - s1 * DecodeNibble((b->qs[j] >> 4) & 0x0F);
    err += w[j + 16] * d1 * d1;
  }
  return err;
}

// Weighted MSE of a stored type-101 block computed FROM its stored bytes.
inline float StoredErrType101(const float* __restrict x, const float* __restrict w,
                              const gqt::block_rocmfp4_fast* b) noexcept {
  const float s = gqt::ScaleFromByte(b->e);
  float err = 0.0f;
  for (int j = 0; j < 16; ++j) {
    const float d0 = x[j] - s * DecodeNibble(b->qs[j] & 0x0F);
    err += w[j] * d0 * d0;
    const float d1 = x[j + 16] - s * DecodeNibble((b->qs[j] >> 4) & 0x0F);
    err += w[j + 16] * d1 * d1;
  }
  return err;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: post_pass_test <src.gguf> <imatrix.gguf> [recipe]\n");
    return 2;
  }
  const std::string src_path = argv[1];
  const std::string im_path = argv[2];
  std::string recipe_name = argc > 3 ? argv[3] : "STRIX";

  gqt::GgufFile f = gqt::LoadGguf(src_path);

  // Recipe: built by name (mirror superroc_quant_main.cpp RecipeNames()).
  struct RN { const char* n; gqt::Recipe r; };
  static const RN RNS[] = {
      {"ROCmFP4", gqt::Recipe::kRocmFp4}, {"ROCmFP4_EVEN", gqt::Recipe::kRocmFp4},
      {"LEAN", gqt::Recipe::kLean}, {"COHERENT", gqt::Recipe::kCoherent},
      {"FAST", gqt::Recipe::kFast}, {"FAST_EVEN", gqt::Recipe::kFast},
      {"FAST_COHERENT", gqt::Recipe::kFastCoherent}, {"STRIX", gqt::Recipe::kStrix},
      {"STRIX_LEAN", gqt::Recipe::kStrixLean},
  };
  gqt::Recipe recipe = gqt::Recipe::kStrix;
  bool found = false;
  for (auto& rn : RNS)
    if (recipe_name == rn.n) { recipe = rn.r; found = true; break; }
  if (!found) {
    std::fprintf(stderr, "unknown recipe %s\n", recipe_name.c_str());
    return 2;
  }

  // Imatrix (fork-verbatim loader semantics).
  // Use the shared helper by replicating main's LoadImatrix: this test reads
  // the same files the quantizer does, so map names via in_sum2/counts pairs.
  struct Im { std::vector<float> e; };
  // Load imatrix the same way superroc_quant_main.cpp::LoadImatrix does.
  gqt::GgufFile imf = gqt::LoadGguf(im_path);
  std::map<std::string, std::pair<const gqt::GgufTensor*, const gqt::GgufTensor*>> pairs;
  for (auto& t : imf.tensors) {
    std::string nm = t.name;
    const std::string ss{".in_sum2"}, sc{".counts"};
    if (nm.size() > ss.size() && nm.compare(nm.size() - ss.size(), ss.size(), ss) == 0) {
      nm.resize(nm.size() - ss.size()); pairs[nm].first = &t;
    } else if (nm.size() > sc.size() && nm.compare(nm.size() - sc.size(), sc.size(), sc) == 0) {
      nm.resize(nm.size() - sc.size()); pairs[nm].second = &t;
    }
  }
  std::map<std::string, std::vector<float>> imatrix;
  for (auto& [name, sc] : pairs) {
    const gqt::GgufTensor* sums = sc.first, *cnts = sc.second;
    if (!sums || !cnts) throw std::runtime_error("imatrix mismatched for " + name);
    const std::uint64_t ne0 = sums->dims[0];
    const std::uint64_t ne1 = sums->dims.size() > 1 ? sums->dims[1] : 1;
    std::vector<float> e(ne0 * ne1);
    const float* sd = (const float*)sums->raw.data();
    const float* cd = (const float*)cnts->raw.data();
    for (std::uint64_t j = 0; j < ne1; ++j) {
      const float c = cd[j];
      for (std::uint64_t i = 0; i < ne0; ++i)
        e[j * ne0 + i] = c > 0.0f ? sd[j * ne0 + i] / c : 1.0f;
    }
    for (float v : e) if (!std::isfinite(v))
      throw std::runtime_error("imatrix non-finite for " + name);
    imatrix[name] = std::move(e);
  }

  // Replicate main's per-tensor inputs for SelectType.
  bool has_expert = false;
  for (auto& t : f.tensors) if (t.dims.size() >= 3 && t.dims[2] > 1) { has_expert = true; break; }
  bool tied = true;
  for (auto& t : f.tensors) if (t.name == "output.weight") { tied = false; break; }
  int n_layer = 1;
  for (auto& [k, v] : f.kv)
    if (k == "qwen35.block_count" || k.find("block_count") != std::string::npos) {
      if (v.scalar_bytes.size() >= 4)
        n_layer = (int)(std::uint32_t)((std::uint8_t)v.scalar_bytes[0] |
            ((std::uint8_t)v.scalar_bytes[1] << 8) |
            ((std::uint8_t)v.scalar_bytes[2] << 16) |
            ((std::uint8_t)v.scalar_bytes[3] << 24));
    }
  gqt::Type default_type = gqt::RecipeDefault(recipe);

  std::uint64_t total_blocks = 0, improved = 0, tied_blocks = 0, violated = 0;
  std::uint64_t roc_tensors = 0;

  for (auto& t : f.tensors) {
    std::string name = t.name;
    gqt::Cat cat = gqt::Categorize(name);
    std::uint64_t ne0 = t.dims.empty() ? 0 : t.dims[0];
    int ndims = 0;
    if (!t.dims.empty()) {
      int last_non1 = -1;
      for (int di = 0; di < (int)t.dims.size(); ++di) if (t.dims[di] > 1) last_non1 = di;
      ndims = last_non1 + 1;
    }
    bool allow = gqt::TensorAllowsQuantization(name, true) && ndims >= 2;
    if (!allow) continue;
    gqt::Type dt = gqt::SelectType(recipe, name, cat, default_type, ne0, n_layer,
                                   has_expert, tied);
    if (dt != (gqt::Type)100 && dt != (gqt::Type)101) continue;  // ROC-only
    roc_tensors++;

    std::uint64_t nrows = 1;
    for (size_t d = 1; d < t.dims.size(); ++d) nrows *= t.dims[d];
    if (nrows == 0) continue;
    std::uint64_t nelem = ne0 * nrows;
    std::uint64_t ne1 = t.dims.size() > 1 ? t.dims[1] : 1;
    std::uint64_t ne2 = t.dims.size() > 2 ? t.dims[2] : 1;

    auto it = imatrix.find(name);
    const float* irow = (it != imatrix.end() && it->second.size() == (size_t)(ne0 * ne2))
                            ? it->second.data() : nullptr;
    auto row_weights = [&](std::uint64_t r) -> const float* {
      if (!irow) return nullptr;
      std::uint64_t expert = (ne1 > 0) ? (r / ne1) : 0;
      return irow + expert * ne0;
    };

    const bool is101 = (dt == (gqt::Type)101);
    const std::vector<float> src = DequantSrc(t, nelem);
    std::vector<std::uint8_t> out_fork(gqt::RowSize(dt, ne0) * nrows);
    std::vector<std::uint8_t> out_pp(out_fork.size());
    std::vector<float> mse_buf(ne0);

    // Quantize the whole tensor twice (same x, same weights): post_pass off
    // (the fork reference) and on.
    for (std::uint64_t r = 0; r < nrows; ++r) {
      const float* x = src.data() + r * ne0;
      const float* qw = row_weights(r);
      if (is101) {
        gqt::QuantizeRowRocmFp4FastEx(x, (gqt::block_rocmfp4_fast*)out_fork.data(),
                                      (std::int64_t)ne0, qw, false);
        gqt::QuantizeRowRocmFp4FastEx(x, (gqt::block_rocmfp4_fast*)out_pp.data(),
                                      (std::int64_t)ne0, qw, true);
      } else {
        gqt::QuantizeRowRocmFp4Ex(x, (gqt::block_rocmfp4*)out_fork.data(), (std::int64_t)ne0,
                                  qw, false);
        gqt::QuantizeRowRocmFp4Ex(x, (gqt::block_rocmfp4*)out_pp.data(), (std::int64_t)ne0,
                                  qw, true);
      }
    }

    if (!irow) {
      // No imatrix weights for this tensor -> the encoder used the UNWEIGHTED
      // !quant_weights arms (qw=nullptr), which ignore post_pass, so both our
      // out_fork/out_pp calls produced identical bytes (no post-pass variant
      // exists for unweighted). ROC parity covers these tensors; skip scoring.
      continue;
    }

    const std::uint64_t nb = ne0 / 32;
    for (std::uint64_t r = 0; r < nrows; ++r) {
      const float* x = src.data() + r * ne0;
      const float* qw = row_weights(r);
      // sigma2 over the full row (same as encoder).
      float sum_x2 = 0.0f;
      for (std::uint64_t i = 0; i < ne0; ++i) sum_x2 += x[i] * x[i];
      const float sigma2 = sum_x2 / (float)ne0;
      // prepared mse weights for this row (matches ChooseScaleWeightedEx).
      float max_abs, max_abs_weight; bool all_finite;
      gqt::PrepareMseWeights(x, (int)ne0, qw, sigma2, mse_buf.data(), max_abs,
                             max_abs_weight, all_finite);

      for (std::uint64_t ib = 0; ib < nb; ++ib) {
        const float* xb = x + ib * 32;
        const float* wb = mse_buf.data() + ib * 32;
        float ef, ep;
        if (is101) {
          const gqt::block_rocmfp4_fast* fk = (const gqt::block_rocmfp4_fast*)out_fork.data();
          const gqt::block_rocmfp4_fast* pp = (const gqt::block_rocmfp4_fast*)out_pp.data();
          ef = StoredErrType101(xb, wb, fk + ib);
          ep = StoredErrType101(xb, wb, pp + ib);
        } else {
          const gqt::block_rocmfp4* fk = (const gqt::block_rocmfp4*)out_fork.data();
          const gqt::block_rocmfp4* pp = (const gqt::block_rocmfp4*)out_pp.data();
          ef = StoredErrType100(xb, wb, fk + ib);
          ep = StoredErrType100(xb, wb, pp + ib);
        }
        total_blocks++;
        if (ep < ef) improved++;
        else if (ep == ef) tied_blocks++;
        else violated++;
        if (ep > ef && !(ep == ef)) {
          std::fprintf(stderr, "VIOLATION %s row %llu blk %llu: fork=%.6e post=%.6e\n",
                       name.c_str(), (unsigned long long)r, (unsigned long long)ib, ef, ep);
        }
      }
    }
  }

  std::printf("recipe=%s roc_tensors=%llu blocks=%llu improved=%llu tied=%llu violated=%llu\n",
              recipe_name.c_str(), (unsigned long long)roc_tensors,
              (unsigned long long)total_blocks, (unsigned long long)improved,
              (unsigned long long)tied_blocks, (unsigned long long)violated);
  if (violated == 0) {
    std::printf("RESULT: PASS (post_pass stored error <= fork weighted on every block)\n");
    return 0;
  }
  std::printf("RESULT: FAIL\n");
  return 1;
}
