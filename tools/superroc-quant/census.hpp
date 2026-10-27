// In-tree GGUF census for superroc-quant v0.5 (LQP SPEC1 item 4).
//
// `superroc-quant census <file.gguf> [--json]`: the header census — arch,
// hyperparameters, tensor count, 2-D types by family with shapes, 1-D summary,
// provenance keys, bytes, BPW. A C++ port of `~/ls/kit/gguf_census.py` (same
// output fields for the parts the script has) with the receipt fields the
// formula needs added (1-D summary, provenance keys, bytes, BPW).
//
// Family normalization is exactly the script's:
//   re.sub(r"^blk\.\d+\.", "blk.N.", name) then re.sub(r"\.\d+\.", ".#.", ...)
// including the substitution quirk that a consumed trailing dot is not
// re-scanned ("a.1.2.b" -> "a.#.2.b").

#ifndef SUPERROC_QUANT_CENSUS_HPP_
#define SUPERROC_QUANT_CENSUS_HPP_

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gguf.hpp"

namespace superroc::quant_tool {

// Type id -> short name. The census reads ARBITRARY files (BF16 sources,
// winner files, fork files), so this covers the full GGUF enum, not just the
// types superroc-quant can write (TypeName() in gguf.hpp covers only those).
inline std::string CensusTypeName(Type t) {
  switch (t) {
    case (Type)0: return "F32";
    case (Type)1: return "F16";
    case (Type)2: return "Q4_0";
    case (Type)3: return "Q4_1";
    case (Type)6: return "Q5_0";
    case (Type)7: return "Q5_1";
    case (Type)8: return "Q8_0";
    case (Type)9: return "Q8_1";
    case (Type)10: return "Q2_K";
    case (Type)11: return "Q3_K";
    case (Type)12: return "Q4_K";
    case (Type)13: return "Q5_K";
    case (Type)14: return "Q6_K";
    case (Type)15: return "Q8_K";
    case (Type)16: return "IQ2_XXS";
    case (Type)17: return "IQ2_XS";
    case (Type)18: return "IQ3_XXS";
    case (Type)19: return "IQ1_S";
    case (Type)20: return "IQ4_NL";
    case (Type)21: return "IQ3_S";
    case (Type)22: return "IQ2_S";
    case (Type)23: return "IQ4_XS";
    case (Type)24: return "I8";
    case (Type)25: return "I16";
    case (Type)26: return "I32";
    case (Type)27: return "I64";
    case (Type)28: return "F64";
    case (Type)29: return "IQ1_M";
    case (Type)30: return "BF16";
    case (Type)31: return "Q8_0_ROCMFPX";
    // v0.8 (RV-LQP2 F3): type 32 is NOT named MXFP4: this project's ggml enum
    // (qfn_ggml_stubs.h) jumps from BF16=30 to MXFP4=39, and no header in the
    // 1194fea tree defines id 32, so the old name was a guess. It falls
    // through to the generic T32 below and is sized by offset differencing.
    case (Type)39: return "MXFP4";
    case (Type)40: return "NVFP4";
    case (Type)41: return "Q1_0";
    case (Type)100: return "ROCMFP4";
    case (Type)101: return "ROCMFP4_FAST";
    case (Type)120: return "G4_Q4";
    case (Type)121: return "G4_Q4_FAST";
  }
  return "T" + std::to_string(t);
}

// "blk.12.attn_q.weight" -> "blk.N.attn_q.weight"; later ".<digits>." runs ->
// ".#.". Faithful to gguf_census.py's two re.sub calls (left-to-right,
// consumed text not re-scanned).
inline std::string CensusFamily(const std::string& name) {
  std::string s = name;
  // ^blk\.\d+\.
  if (s.compare(0, 4, "blk.") == 0) {
    std::size_t i = 4;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    if (i > 4 && i < s.size() && s[i] == '.') s = "blk.N." + s.substr(i + 1);
  }
  // \.\d+\. -> .#.  (all non-overlapping occurrences, left to right)
  std::string o;
  for (std::size_t i = 0; i < s.size();) {
    if (s[i] == '.') {
      std::size_t j = i + 1;
      while (j < s.size() && s[j] >= '0' && s[j] <= '9') ++j;
      if (j > i + 1 && j < s.size() && s[j] == '.') {
        o += ".#.";
        i = j + 1;
        continue;
      }
    }
    o += s[i];
    ++i;
  }
  return o;
}

struct CensusKv {
  std::string key;
  std::string value;  // human rendering (ints decimal, floats %g, strings raw)
};

struct CensusFamilyRow {
  std::string family;
  std::map<std::string, std::uint64_t> types;  // type name -> count
  std::vector<std::uint64_t> shape;            // first-seen dims
  std::uint64_t ndims = 0;                     // stored dim count
};

struct Census {
  std::string arch = "?";
  std::string name, basename, finetune;
  std::uint64_t ntensors = 0;
  std::vector<CensusKv> hyper;                 // arch.* keys of interest
  std::map<std::string, std::uint64_t> d2_overall;  // 2-D+ type -> count
  std::map<std::string, std::uint64_t> d1_overall;  // 1-D type -> count
  std::vector<CensusFamilyRow> families;       // sorted by family
  std::vector<CensusKv> provenance;            // superroc.quant.* + quantize.*
  std::uint64_t bytes = 0;                     // sum of stored data bytes
  std::uint64_t elems = 0;                     // sum of dim products
  double bpw = 0.0;                            // 8*bytes/elems
  // v0.8 (F3): names of tensors sized by offset differencing (unknown layout;
  // the size includes alignment padding). Empty for fully-known files.
  std::vector<std::string> offset_diff_sized;
};

inline std::string CensusKvStr(const KvValue& v) {
  char tmp[64];
  switch (v.type) {
    case ValType::kUint8:
      std::snprintf(tmp, sizeof tmp, "%u", v.scalar_bytes[0]);
      return tmp;
    case ValType::kInt8:
      std::snprintf(tmp, sizeof tmp, "%d", (int)(std::int8_t)v.scalar_bytes[0]);
      return tmp;
    case ValType::kUint16: {
      std::uint16_t x = 0;
      std::memcpy(&x, v.scalar_bytes.data(), 2);
      std::snprintf(tmp, sizeof tmp, "%u", x);
      return tmp;
    }
    case ValType::kInt16: {
      std::int16_t x = 0;
      std::memcpy(&x, v.scalar_bytes.data(), 2);
      std::snprintf(tmp, sizeof tmp, "%d", (int)x);
      return tmp;
    }
    case ValType::kUint32: {
      std::uint32_t x = 0;
      std::memcpy(&x, v.scalar_bytes.data(), 4);
      std::snprintf(tmp, sizeof tmp, "%u", x);
      return tmp;
    }
    case ValType::kInt32: {
      std::int32_t x = 0;
      std::memcpy(&x, v.scalar_bytes.data(), 4);
      std::snprintf(tmp, sizeof tmp, "%d", (int)x);
      return tmp;
    }
    case ValType::kFloat32: {
      float x = 0;
      std::memcpy(&x, v.scalar_bytes.data(), 4);
      std::snprintf(tmp, sizeof tmp, "%g", (double)x);
      return tmp;
    }
    case ValType::kBool:
      return v.scalar_bytes[0] ? "true" : "false";
    case ValType::kString:
      return v.str;
    case ValType::kUint64: {
      std::uint64_t x = 0;
      std::memcpy(&x, v.scalar_bytes.data(), 8);
      std::snprintf(tmp, sizeof tmp, "%llu", (unsigned long long)x);
      return tmp;
    }
    case ValType::kInt64: {
      std::int64_t x = 0;
      std::memcpy(&x, v.scalar_bytes.data(), 8);
      std::snprintf(tmp, sizeof tmp, "%lld", (long long)x);
      return tmp;
    }
    case ValType::kFloat64: {
      double x = 0;
      std::memcpy(&x, v.scalar_bytes.data(), 8);
      std::snprintf(tmp, sizeof tmp, "%g", x);
      return tmp;
    }
    case ValType::kArray:
      if (v.array_type == ValType::kString) {
        std::string s = "[";
        for (std::size_t i = 0; i < v.array_strs.size(); ++i) {
          if (i) s += ",";
          s += v.array_strs[i];
        }
        return s + "]";
      }
      return "[array " +
             std::to_string(v.array_elems.size()) + "]";
  }
  return "?";
}

// gguf_census.py hyperparam filter: keys starting "<arch>." matching
// block_count|embedding_length|expert|hc|ssm|feed_forward|head_count$|indexer|ple
inline bool CensusIsHyper(const std::string& arch, const std::string& key) {
  if (key.size() <= arch.size() + 1) return false;
  if (key.compare(0, arch.size(), arch) != 0 || key[arch.size()] != '.') return false;
  auto has = [&](const char* sub) { return key.find(sub) != std::string::npos; };
  if (has("block_count") || has("embedding_length") || has("expert") ||
      has("hc") || has("ssm") || has("feed_forward") || has("indexer") ||
      has("ple"))
    return true;
  const std::string tail = "head_count";
  return key.size() >= tail.size() &&
         key.compare(key.size() - tail.size(), tail.size(), tail) == 0;
}

// Data bytes of one tensor for the census, which reads ARBITRARY files (stock
// quants, fork files, BF16 sources) — so this covers the full GGUF type enum,
// not just the types superroc-quant can write. Sets *known=false for types with
// no public layout (fork MXFP4-family); the caller then sizes those by tensor
// offset differencing (padded region size, exact up to alignment slack).
inline std::uint64_t CensusTensorBytes(Type t, const std::vector<std::uint64_t>& dims,
                                       bool* known) {
  std::uint64_t be = 0, bb = 0;  // block elems, block bytes
  switch (t) {
    case (Type)0: be = 1; bb = 4; break;    // F32
    case (Type)1: be = 1; bb = 2; break;    // F16
    case (Type)2: be = 32; bb = 18; break;  // Q4_0
    case (Type)3: be = 32; bb = 20; break;  // Q4_1
    case (Type)6: be = 32; bb = 22; break;  // Q5_0
    case (Type)7: be = 32; bb = 24; break;  // Q5_1
    case (Type)8: be = 32; bb = 34; break;  // Q8_0
    case (Type)9: be = 32; bb = 36; break;  // Q8_1
    case (Type)10: be = 256; bb = 84; break;   // Q2_K
    case (Type)11: be = 256; bb = 110; break;  // Q3_K
    case (Type)12: be = 256; bb = 144; break;  // Q4_K
    case (Type)13: be = 256; bb = 176; break;  // Q5_K
    case (Type)14: be = 256; bb = 210; break;  // Q6_K
    case (Type)15: be = 256; bb = 292; break;  // Q8_K
    case (Type)16: be = 256; bb = 66; break;   // IQ2_XXS
    case (Type)17: be = 256; bb = 74; break;   // IQ2_XS
    case (Type)18: be = 256; bb = 98; break;   // IQ3_XXS
    case (Type)19: be = 256; bb = 50; break;   // IQ1_S
    case (Type)20: be = 32; bb = 18; break;    // IQ4_NL
    case (Type)21: be = 256; bb = 110; break;  // IQ3_S
    case (Type)22: be = 256; bb = 82; break;   // IQ2_S
    case (Type)23: be = 256; bb = 136; break;  // IQ4_XS
    case (Type)24: be = 1; bb = 1; break;      // I8
    case (Type)25: be = 1; bb = 2; break;      // I16
    case (Type)26: be = 1; bb = 4; break;      // I32
    case (Type)27: be = 1; bb = 8; break;      // I64
    case (Type)28: be = 1; bb = 8; break;      // F64
    case (Type)29: be = 256; bb = 56; break;   // IQ1_M
    case (Type)30: be = 1; bb = 2; break;      // BF16
    // v0.8 (RV-LQP2 F3): ggml MXFP4-family layouts from qfn_ggml_stubs.h
    // (ggml_type_size :254-256, ggml_blck_size :279-297).
    case (Type)39: be = 32; bb = 17; break;    // MXFP4
    case (Type)40: be = 64; bb = 18; break;    // NVFP4
    case (Type)41: be = 256; bb = 36; break;   // Q1_0
    case (Type)100: be = 32; bb = 18; break;   // ROCMFP4
    case (Type)101: be = 32; bb = 17; break;   // ROCMFP4_FAST
    case (Type)120: be = 32; bb = 18; break;   // G4_Q4
    case (Type)121: be = 32; bb = 17; break;   // G4_Q4_FAST
    default: break;
  }
  if (be == 0) {
    if (known) *known = false;
    return 0;
  }
  if (known) *known = true;
  std::uint64_t ne0 = dims.empty() ? 0 : dims[0];
  std::uint64_t nrows = 1;
  for (std::size_t d = 1; d < dims.size(); ++d) nrows *= dims[d];
  if (be == 1) return ne0 * bb * nrows;
  if (ne0 % be != 0) {  // corrupt or exotic: caller falls back to diffing
    if (known) *known = false;
    return 0;
  }
  return (ne0 / be) * bb * nrows;
}

// Load one file's header + per-tensor sizes for the census. Parses WITHOUT
// sizing (TensorNBytes throws on the non-writable types stock quants carry),
// then sizes every tensor from the census's own full-enum table, with offset
// differencing against the file size for unknown-layout types — so this never
// throws on a well-formed header. NOTE: reads the given file only; for a
// split model pass each shard separately.
inline GgufFile LoadCensusFile(const std::string& path) {
  std::ifstream fs(path, std::ios::binary);
  if (!fs) throw std::runtime_error("cannot open " + path);
  HeaderResult hr = LoadHeaderStream(fs, /*load_sizes=*/false);
  fs.seekg(0, std::ios::end);
  std::uint64_t file_size = (std::uint64_t)fs.tellg();
  GgufFile f;
  f.alignment = hr.alignment;
  f.kv = std::move(hr.kv);
  f.tensors = std::move(hr.tensors);
  for (std::size_t i = 0; i < f.tensors.size(); ++i) {
    bool known = false;
    std::uint64_t nb = CensusTensorBytes(f.tensors[i].type, f.tensors[i].dims, &known);
    if (!known) {
      // Padded region size: next tensor's offset, or EOF for the last one.
      if (i + 1 < f.tensors.size())
        nb = f.tensors[i + 1].offset - f.tensors[i].offset;
      else
        nb = file_size - hr.data_base - f.tensors[i].offset;
    }
    f.tensors[i].size_bytes = nb;
  }
  return f;
}

inline Census ComputeCensus(const GgufFile& f) {
  Census c;
  auto kv_str = [&](const std::string& key) -> std::string {
    for (auto& [k, v] : f.kv)
      if (k == key && v.type == ValType::kString) return v.str;
    return "";
  };
  {
    std::string a = kv_str("general.architecture");
    if (!a.empty()) c.arch = a;
  }
  c.name = kv_str("general.name");
  c.basename = kv_str("general.basename");
  c.finetune = kv_str("general.finetune");
  c.ntensors = f.tensors.size();
  for (auto& [k, v] : f.kv) {
    if (CensusIsHyper(c.arch, k)) c.hyper.push_back({k, CensusKvStr(v)});
    if (k.rfind("superroc.quant.", 0) == 0 || k.rfind("quantize.", 0) == 0)
      c.provenance.push_back({k, CensusKvStr(v)});
  }
  std::map<std::string, CensusFamilyRow> fam;
  for (auto& t : f.tensors) {
    const std::string tn = CensusTypeName(t.type);
    bool layout_known = false;
    (void)CensusTensorBytes(t.type, t.dims, &layout_known);
    if (!layout_known) c.offset_diff_sized.push_back(t.name);
    const std::uint64_t nd = t.dims.size();
    if (nd >= 2)
      c.d2_overall[tn]++;
    else
      c.d1_overall[tn]++;
    c.bytes += t.size_bytes;
    std::uint64_t e = 1;
    for (auto d : t.dims) e *= d;
    c.elems += e;
    const std::string fn = CensusFamily(t.name);
    auto& row = fam[fn];
    row.family = fn;
    row.types[tn]++;
    if (row.shape.empty()) {
      row.shape = t.dims;
      row.ndims = nd;
    }
  }
  for (auto& [_, row] : fam) c.families.push_back(row);  // map order = sorted
  if (c.elems) c.bpw = 8.0 * (double)c.bytes / (double)c.elems;
  return c;
}

inline std::string CensusText(const Census& c, const std::string& basename) {
  std::ostringstream o;
  o << "\n##### " << basename << "\n";
  o << "arch=" << c.arch << "  name=" << c.name << "  basename=" << c.basename
    << "  finetune=" << c.finetune << "  tensors=" << c.ntensors << "\n";
  o << "  ";
  for (std::size_t i = 0; i < c.hyper.size(); ++i) {
    if (i) o << "  ";
    std::string k = c.hyper[i].key;
    auto dot = k.find('.');
    if (dot != std::string::npos) k = k.substr(dot + 1);
    o << k << "=" << c.hyper[i].value;
  }
  o << "\n  2-D+ tensor types overall: {";
  {
    bool first = true;
    // most-common order, like the script's Counter.most_common()
    std::vector<std::pair<std::string, std::uint64_t>> v(c.d2_overall.begin(),
                                                          c.d2_overall.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
      if (a.second != b.second) return a.second > b.second;
      return a.first < b.first;
    });
    for (auto& [t, n] : v) {
      if (!first) o << ", ";
      first = false;
      o << "'" << t << "': " << n;
    }
  }
  o << "}\n  1-D tensor types overall: {";
  {
    bool first = true;
    for (auto& [t, n] : c.d1_overall) {
      if (!first) o << ", ";
      first = false;
      o << "'" << t << "': " << n;
    }
  }
  o << "}\n  families (2-D+), name: types  [shape]\n";
  char line[512];
  for (auto& row : c.families) {
    if (row.ndims < 2) continue;
    std::string ts;
    {
      std::vector<std::pair<std::string, std::uint64_t>> v(row.types.begin(),
                                                            row.types.end());
      std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
      });
      for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) ts += ",";
        ts += v[i].first + " x" + std::to_string(v[i].second);
      }
    }
    std::string sh = "[";
    for (std::size_t i = 0; i < row.shape.size(); ++i) {
      if (i) sh += ", ";
      sh += std::to_string(row.shape[i]);
    }
    sh += "]";
    std::snprintf(line, sizeof line, "    %-44s %-34s %s", row.family.c_str(),
                  ts.c_str(), sh.c_str());
    o << line << "\n";
  }
  o << "  provenance:";
  if (c.provenance.empty()) {
    o << " (none)\n";
  } else {
    o << "\n";
    for (auto& p : c.provenance) o << "    " << p.key << " = " << p.value << "\n";
  }
  {
    char b[128];
    std::snprintf(b, sizeof b, "  bytes=%llu elems=%llu BPW=%.3f\n",
                  (unsigned long long)c.bytes, (unsigned long long)c.elems,
                  c.bpw);
    o << b;
  }
  // v0.8 (F3): say when a size came from offset differencing (the padded
  // region size, alignment slack included) instead of the type layout.
  if (!c.offset_diff_sized.empty()) {
    o << "  sizes from offset differencing (alignment padding included):";
    for (auto& n : c.offset_diff_sized) o << " " << n;
    o << "\n";
  }
  return o.str();
}

inline std::string JsonEscape(const std::string& s) {
  std::string o;
  for (char ch : s) {
    switch (ch) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if ((unsigned char)ch < 0x20) {
          char tmp[8];
          std::snprintf(tmp, sizeof tmp, "\\u%04x", ch);
          o += tmp;
        } else {
          o += ch;
        }
    }
  }
  return o;
}

inline std::string CensusJson(const Census& c, const std::string& path) {
  std::ostringstream o;
  char b[128];
  o << "{\n";
  o << "  \"file\": \"" << JsonEscape(path) << "\",\n";
  o << "  \"arch\": \"" << JsonEscape(c.arch) << "\",\n";
  o << "  \"name\": \"" << JsonEscape(c.name) << "\",\n";
  o << "  \"basename\": \"" << JsonEscape(c.basename) << "\",\n";
  o << "  \"finetune\": \"" << JsonEscape(c.finetune) << "\",\n";
  o << "  \"tensors\": " << c.ntensors << ",\n";
  o << "  \"hyperparameters\": {";
  for (std::size_t i = 0; i < c.hyper.size(); ++i) {
    if (i) o << ",";
    o << "\n    \"" << JsonEscape(c.hyper[i].key)
      << "\": \"" << JsonEscape(c.hyper[i].value) << "\"";
  }
  o << (c.hyper.empty() ? "},\n" : "\n  },\n");
  auto typemap = [&](const std::map<std::string, std::uint64_t>& m,
                     const char* key) {
    o << "  \"" << key << "\": {";
    bool first = true;
    for (auto& [t, n] : m) {
      if (!first) o << ",";
      first = false;
      o << "\n    \"" << JsonEscape(t) << "\": " << n;
    }
    o << (first ? "},\n" : "\n  },\n");
  };
  typemap(c.d2_overall, "types_2d");
  typemap(c.d1_overall, "types_1d");
  o << "  \"families\": {";
  for (std::size_t i = 0; i < c.families.size(); ++i) {
    auto& row = c.families[i];
    if (i) o << ",";
    o << "\n    \"" << JsonEscape(row.family) << "\": {\"types\": {";
    bool first = true;
    for (auto& [t, n] : row.types) {
      if (!first) o << ",";
      first = false;
      o << "\"" << JsonEscape(t) << "\": " << n;
    }
    o << "}, \"shape\": [";
    for (std::size_t j = 0; j < row.shape.size(); ++j) {
      if (j) o << ", ";
      o << row.shape[j];
    }
    o << "], \"ndims\": " << row.ndims << "}";
  }
  o << (c.families.empty() ? "},\n" : "\n  },\n");
  o << "  \"provenance\": {";
  for (std::size_t i = 0; i < c.provenance.size(); ++i) {
    if (i) o << ",";
    o << "\n    \"" << JsonEscape(c.provenance[i].key)
      << "\": \"" << JsonEscape(c.provenance[i].value) << "\"";
  }
  o << (c.provenance.empty() ? "},\n" : "\n  },\n");
  std::snprintf(b, sizeof b, "  \"bytes\": %llu,\n",
                (unsigned long long)c.bytes);
  o << b;
  std::snprintf(b, sizeof b, "  \"elems\": %llu,\n", (unsigned long long)c.elems);
  o << b;
  std::snprintf(b, sizeof b, "  \"bpw\": %.4f\n", c.bpw);
  o << b;
  o << "}\n";
  return o.str();
}

}  // namespace superroc::quant_tool

#endif  // SUPERROC_QUANT_CENSUS_HPP_
