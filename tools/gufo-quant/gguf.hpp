// gufo-quant GGUF reader/writer.
//
// Reader loads a GGUF (v3) into an ordered KV list + tensors (name, dims,
// type, and the RAW payload bytes for each tensor). Writer reproduces the
// fork's gguf.cpp layout byte-for-byte so gufo-quant output == fork output
// except the added `gufo.quant.*` KV keys.

#ifndef GUFO_QUANT_GGUF_HPP_
#define GUFO_QUANT_GGUF_HPP_

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "sha256.hpp"

namespace gufo::quant_tool {

inline constexpr std::uint32_t kGgufVersion = 3;
inline constexpr std::uint32_t kDefaultAlignment = 32;

using Type = std::uint32_t;

enum class ValType : std::uint8_t {
  kUint8 = 0, kInt8 = 1, kUint16 = 2, kInt16 = 3, kUint32 = 4, kInt32 = 5,
  kFloat32 = 6, kBool = 7, kString = 8, kArray = 9, kUint64 = 10, kInt64 = 11,
  kFloat64 = 12,
};

struct KvValue {
  ValType type{ValType::kString};
  std::vector<std::uint8_t> scalar_bytes;          // scalar raw LE bytes
  ValType array_type{ValType::kString};            // for ARRAY
  std::vector<std::vector<std::uint8_t>> array_elems;  // for non-string ARRAY
  std::vector<std::string> array_strs;             // for STRING array
  std::string str;                                 // for STRING
};

struct GgufTensor {
  std::string name;
  std::vector<std::uint64_t> dims;
  Type type{0};
  std::uint64_t offset{0};
  std::uint64_t size_bytes{0};
  int shard{0};                    // split shard index this tensor lives in
  bool raw_loaded{false};
  std::vector<std::uint8_t> raw;   // raw payload (for dense: raw bytes)
};

struct GgufFile {
  std::vector<std::pair<std::string, KvValue>> kv;
  std::vector<GgufTensor> tensors;
  std::uint32_t alignment{kDefaultAlignment};
  std::vector<std::uint8_t> whole;  // whole input file bytes (for reference)
  // split support: per-shard source files + their per-shard tensor data bases
  std::vector<std::string> shard_paths;
  std::vector<std::uint64_t> shard_databases;
  bool is_split{false};
};

inline std::vector<std::uint8_t> ReadFileBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(f)),
                                   std::istreambuf_iterator<char>());
}
inline void WriteFileBytes(const std::string& path, const std::vector<std::uint8_t>& b) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.write((const char*)b.data(), b.size());
}

// ---- block/type helpers (element/byte counts per storage block) ----
inline std::uint64_t BlockBytes(Type t) {
  switch (t) {
    case (Type)0: return 4;   // F32
    case (Type)1: return 2;   // F16
    case (Type)30: return 2;  // BF16
    case (Type)2: return 18;  // Q4_0
    case (Type)8: return 34;  // Q8_0
    case (Type)7: return 24;  // Q5_1 (f16 d + f16 m + 4 qh + 16 qs)
    case (Type)13: return 176;// Q5_K
    case (Type)14: return 210;// Q6_K
    case (Type)100: return 18;// Q4_0_ROCMFP4
    case (Type)101: return 17;// Q4_0_ROCMFP4_FAST
    case (Type)120: return 18;// G4_Q4 (split scale plane; same bytes as 100)
    case (Type)121: return 17;// G4_Q4_FAST (split scale plane; same bytes as 101)
    default: throw std::runtime_error("BlockBytes: unsupported type " + std::to_string(t));
  }
}
inline std::uint64_t BlockElems(Type t) {
  switch (t) {
    case (Type)2: case (Type)7: case (Type)8:
    case (Type)100: case (Type)101: case (Type)120: case (Type)121: return 32;
    case (Type)13: case (Type)14: return 256;
    default: return 1;
  }
}
// "TypeBlockBytes" = block bytes (valid for quant); for dense returns elem bytes
inline std::uint64_t TypeBlockBytes(Type t) { return BlockBytes(t); }
inline std::string TypeName(Type t) {
  switch (t) {
    case (Type)0: return "F32";
    case (Type)1: return "F16";
    case (Type)30: return "BF16";
    case (Type)2: return "Q4_0";
    case (Type)7: return "Q5_1";
    case (Type)8: return "Q8_0";
    case (Type)13: return "Q5_K";
    case (Type)14: return "Q6_K";
    case (Type)100: return "Q4_0_ROCMFP4";
    case (Type)101: return "Q4_0_ROCMFP4_FAST";
    case (Type)120: return "G4_Q4";
    case (Type)121: return "G4_Q4_FAST";
  }
  return "UNKNOWN";
}

// Row size (bytes) of one row (ne0 elements) for type t.
inline std::uint64_t RowSize(Type t, std::uint64_t ne0) {
  const std::uint64_t be = BlockElems(t);
  if (be == 1) return ne0 * BlockBytes(t);
  if (ne0 % be != 0) throw std::runtime_error("RowSize: ne0 not divisible by block");
  return (ne0 / be) * BlockBytes(t);
}
inline std::uint64_t TensorNBytes(Type t, const std::vector<std::uint64_t>& dims) {
  std::uint64_t nrows = 1;
  for (size_t d = 1; d < dims.size(); ++d) nrows *= dims[d];
  return RowSize(t, dims[0]) * nrows;
}

// ---- low-level read ----
inline std::uint64_t R_U8(const std::vector<std::uint8_t>& b, std::uint64_t& o) {
  // v0.8: split across lines (was one line; gcc -Wmisleading-indentation).
  if (o + 1 > b.size()) throw std::runtime_error("gguf short");
  return b[o++];
}
inline std::uint64_t R_U16(const std::vector<std::uint8_t>& b, std::uint64_t& o) {
  if (o + 2 > b.size()) throw std::runtime_error("gguf short");
  std::uint64_t v; std::memcpy(&v, &b[o], 2); o += 2; return v;
}
inline std::uint64_t R_U32(const std::vector<std::uint8_t>& b, std::uint64_t& o) {
  if (o + 4 > b.size()) throw std::runtime_error("gguf short");
  std::uint64_t v; std::memcpy(&v, &b[o], 4); o += 4; return v;
}
inline std::uint64_t R_U64(const std::vector<std::uint8_t>& b, std::uint64_t& o) {
  if (o + 8 > b.size()) throw std::runtime_error("gguf short");
  std::uint64_t v; std::memcpy(&v, &b[o], 8); o += 8; return v;
}
inline std::string R_STR(const std::vector<std::uint8_t>& b, std::uint64_t& o) {
  std::uint64_t n = R_U64(b, o);
  if (o + n > b.size()) throw std::runtime_error("gguf short str");
  std::string s((const char*)&b[o], n); o += n; return s;
}

// ---- load (header only: KV + tensor metadata, NO payload bytes) ----
// Reads only the GGUF header so huge (multi-GB) model files can be used as
// roster/KV templates without loading their tensor data.
inline GgufFile LoadGgufHeaderOnly(const std::string& path, bool load_sizes = true) {
  std::ifstream fs(path, std::ios::binary);
  if (!fs) throw std::runtime_error("cannot open " + path);
  auto rd = [&](void* dst, std::size_t n) { fs.read((char*)dst, n); if (!fs) throw std::runtime_error("gguf short read"); };
  char mag[4]; rd(mag, 4);
  if (std::memcmp(mag, "GGUF", 4) != 0) throw std::runtime_error("not a GGUF");
  std::uint32_t ver; rd(&ver, 4);
  std::uint64_t n_tensors, n_kv; rd(&n_tensors, 8); rd(&n_kv, 8);

  GgufFile f;
  f.kv.reserve(n_kv);
  auto rs = [&]() -> std::string {
    std::uint64_t n; rd(&n, 8);
    std::string s(n, '\0');
    if (n) rd(&s[0], n);
    return s;
  };
  auto read_scalar = [&](ValType t, std::vector<std::uint8_t>& out) {
    switch (t) {
      case ValType::kUint8: case ValType::kInt8: case ValType::kBool: {
        std::uint8_t x; rd(&x, 1); out = {x}; break;
      }
      case ValType::kUint16: case ValType::kInt16: {
        std::uint16_t x; rd(&x, 2); out.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 2); break;
      }
      case ValType::kUint32: case ValType::kInt32: case ValType::kFloat32: {
        std::uint32_t x; rd(&x, 4); out.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 4); break;
      }
      case ValType::kUint64: case ValType::kInt64: case ValType::kFloat64: {
        std::uint64_t x; rd(&x, 8); out.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 8); break;
      }
      default: throw std::runtime_error("unexpected scalar type in read");
    }
  };
  for (std::uint64_t i = 0; i < n_kv; ++i) {
    std::string key = rs();
    ValType vt = (ValType)[&]{ std::uint32_t t; rd(&t, 4); return t; }();
    KvValue v; v.type = vt;
    if (vt == ValType::kArray) {
      std::uint32_t at; rd(&at, 4);
      v.array_type = (ValType)at;
      std::uint64_t n; rd(&n, 8);
      if (v.array_type == ValType::kString) {
        for (std::uint64_t a = 0; a < n; ++a) v.array_strs.push_back(rs());
      } else {
        for (std::uint64_t a = 0; a < n; ++a) { std::vector<std::uint8_t> ev; read_scalar(v.array_type, ev); v.array_elems.push_back(ev); }
      }
    } else if (vt == ValType::kString) {
      v.str = rs();
    } else {
      read_scalar(vt, v.scalar_bytes);
    }
    if (key == "general.alignment" && v.scalar_bytes.size() >= 4) {
      std::uint32_t al = 0; std::memcpy(&al, v.scalar_bytes.data(), 4);
      f.alignment = al ? al : kDefaultAlignment;
    }
    f.kv.emplace_back(key, std::move(v));
  }
  for (std::uint64_t i = 0; i < n_tensors; ++i) {
    GgufTensor t;
    t.name = rs();
    std::uint32_t nd; rd(&nd, 4);
    for (std::uint32_t d = 0; d < nd; ++d) { std::uint64_t x; rd(&x, 8); t.dims.push_back(x); }
    rd(&t.type, 4);
    rd(&t.offset, 8);
    if (load_sizes) t.size_bytes = TensorNBytes(t.type, t.dims);
    f.tensors.push_back(std::move(t));
  }
  return f;
}

// ---- low-level helpers for streaming / split load ----
// Parse just the header (KV + tensor metadata) of one GGUF file from a stream.
struct HeaderResult {
  std::uint32_t alignment{kDefaultAlignment};
  std::vector<std::pair<std::string, KvValue>> kv;
  std::vector<GgufTensor> tensors;
  std::uint64_t data_base{0};
  std::uint64_t tensor_count{0};
};

// load_sizes=false leaves size_bytes at 0 for every tensor (mirrors
// LoadGgufHeaderOnly's flag above): the census reads arbitrary files with
// types TensorNBytes cannot size, so it must parse first and size later.
inline HeaderResult LoadHeaderStream(std::ifstream& fs, bool load_sizes = true) {
  auto rd = [&](void* dst, std::size_t n) { fs.read((char*)dst, n); if (!fs) throw std::runtime_error("gguf short read"); };
  char mag[4]; rd(mag, 4);
  if (std::memcmp(mag, "GGUF", 4) != 0) throw std::runtime_error("not a GGUF");
  std::uint32_t ver; rd(&ver, 4);
  std::uint64_t n_tensors, n_kv; rd(&n_tensors, 8); rd(&n_kv, 8);
  HeaderResult hr;
  hr.tensor_count = n_tensors;
  hr.kv.reserve(n_kv);
  auto rs = [&]() -> std::string {
    std::uint64_t n; rd(&n, 8);
    std::string s(n, 0); if (n) rd(&s[0], n); return s;
  };
  auto read_scalar = [&](ValType t, std::vector<std::uint8_t>& out) {
    switch (t) {
      case ValType::kUint8: case ValType::kInt8: case ValType::kBool: {
        std::uint8_t x; rd(&x, 1); out = {x}; break; }
      case ValType::kUint16: case ValType::kInt16: {
        std::uint16_t x; rd(&x, 2); out.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 2); break; }
      case ValType::kUint32: case ValType::kInt32: case ValType::kFloat32: {
        std::uint32_t x; rd(&x, 4); out.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 4); break; }
      case ValType::kUint64: case ValType::kInt64: case ValType::kFloat64: {
        std::uint64_t x; rd(&x, 8); out.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 8); break; }
      default: throw std::runtime_error("unexpected scalar type in read");
    }
  };
  for (std::uint64_t i = 0; i < n_kv; ++i) {
    std::string key = rs();
    ValType vt = (ValType)[&]{ std::uint32_t t; rd(&t, 4); return t; }();
    KvValue v; v.type = vt;
    if (vt == ValType::kArray) {
      std::uint32_t at; rd(&at, 4); v.array_type = (ValType)at;
      std::uint64_t n; rd(&n, 8);
      if (v.array_type == ValType::kString) { for (std::uint64_t a = 0; a < n; ++a) v.array_strs.push_back(rs()); }
      else { for (std::uint64_t a = 0; a < n; ++a) { std::vector<std::uint8_t> ev; read_scalar(v.array_type, ev); v.array_elems.push_back(ev); } }
    } else if (vt == ValType::kString) { v.str = rs(); }
    else { read_scalar(vt, v.scalar_bytes); }
    if (key == "general.alignment" && v.scalar_bytes.size() >= 4) {
      std::uint32_t al = 0; std::memcpy(&al, v.scalar_bytes.data(), 4);
      hr.alignment = al ? al : kDefaultAlignment;
    }
    hr.kv.emplace_back(key, std::move(v));
  }
  for (std::uint64_t i = 0; i < n_tensors; ++i) {
    GgufTensor t; t.name = rs();
    std::uint32_t nd; rd(&nd, 4);
    for (std::uint32_t d = 0; d < nd; ++d) { std::uint64_t x; rd(&x, 8); t.dims.push_back(x); }
    rd(&t.type, 4); rd(&t.offset, 8);
    if (load_sizes) t.size_bytes = TensorNBytes(t.type, t.dims);
    hr.tensors.push_back(std::move(t));
  }
  std::uint64_t o = (std::uint64_t)fs.tellg();
  std::uint64_t pad = (hr.alignment - o % hr.alignment) % hr.alignment;
  hr.data_base = o + pad;
  return hr;
}

inline std::string SplitPathFor(const std::string& path0, int split_no, int split_count) {
  char f[32]; snprintf(f, sizeof(f), "-%05d-of-%05d.gguf", split_no + 1, split_count);
  char g[32]; snprintf(g, sizeof(g), "-%05d-of-%05d.gguf", 1, split_count);
  std::string suffix = g;
  if (path0.size() <= suffix.size() || path0.compare(path0.size() - suffix.size(), suffix.size(), suffix) != 0)
    throw std::runtime_error("not a split file name: " + path0);
  return path0.substr(0, path0.size() - suffix.size()) + f;
}

inline GgufFile LoadGgufStreaming(const std::string& path) {
  std::ifstream fs(path, std::ios::binary);
  if (!fs) throw std::runtime_error("cannot open " + path);
  HeaderResult hr = LoadHeaderStream(fs);
  GgufFile f;
  f.alignment = hr.alignment; f.kv = hr.kv; f.tensors = hr.tensors;
  f.shard_paths.push_back(path); f.shard_databases.push_back(hr.data_base);
  f.is_split = false;
  std::uint64_t split_count = 1;
  for (auto& [k, v] : f.kv) if (k == "split.count" && v.scalar_bytes.size() >= 2) {
    std::uint16_t c; std::memcpy(&c, v.scalar_bytes.data(), 2); split_count = c; }
  if (split_count > 1) {
    f.is_split = true;
    int idx0 = 0;
    for (auto& [k, v] : f.kv) if (k == "split.no" && v.scalar_bytes.size() >= 2) {
      std::uint16_t n; std::memcpy(&n, v.scalar_bytes.data(), 2); idx0 = n; }
    if (idx0 != 0) throw std::runtime_error("split load: first file is not split #0");
    std::uint64_t total = f.tensors.size();
    for (std::uint64_t sp = 1; sp < split_count; ++sp) {
      std::string spath = SplitPathFor(path, (int)sp, (int)split_count);
      std::ifstream sfs(spath, std::ios::binary);
      if (!sfs) throw std::runtime_error("cannot open split " + spath);
      HeaderResult shr = LoadHeaderStream(sfs);
      for (auto& [k, v] : shr.kv) if (k == "split.no" && v.scalar_bytes.size() >= 2) {
        std::uint16_t n; std::memcpy(&n, v.scalar_bytes.data(), 2);
        if ((int)n != (int)sp) throw std::runtime_error("split idx mismatch in " + spath); }
      for (auto& t : shr.tensors) { t.shard = (int)sp; f.tensors.push_back(std::move(t)); }
      f.shard_paths.push_back(spath); f.shard_databases.push_back(shr.data_base);
      total += shr.tensors.size();
    }
    for (auto& [k, v] : f.kv) if (k == "split.tensors.count" && v.scalar_bytes.size() >= 4) {
      std::uint32_t n; std::memcpy(&n, v.scalar_bytes.data(), 4);
      if ((std::uint64_t)n != total) throw std::runtime_error("split.tensors.count mismatch"); }
  }
  return f;
}

inline void FetchTensorData(GgufFile& f, size_t i) {
  GgufTensor& t = f.tensors[i];
  if (t.raw_loaded) return;
  if (t.shard < 0 || t.shard >= (int)f.shard_paths.size())
    throw std::runtime_error("bad shard index for " + t.name);
  std::ifstream fs(f.shard_paths[t.shard], std::ios::binary);
  if (!fs) throw std::runtime_error("cannot open " + f.shard_paths[t.shard]);
  std::uint64_t pos = f.shard_databases[t.shard] + t.offset;
  t.raw.resize(t.size_bytes);
  fs.seekg((std::streamoff)pos, std::ios::beg);
  if (!fs) throw std::runtime_error("gguf seek fail " + t.name);
  fs.read((char*)t.raw.data(), (std::streamsize)t.size_bytes);
  if ((std::uint64_t)fs.gcount() != t.size_bytes) throw std::runtime_error("gguf short tensor read " + t.name);
  t.raw_loaded = true;
}

// ---- load (full: header + payload bytes) ----
inline GgufFile LoadGguf(const std::string& path) {
  auto b = ReadFileBytes(path);
  std::uint64_t o = 0;
  if (b.size() < 4 || b[0] != 'G' || b[1] != 'G' || b[2] != 'U' || b[3] != 'F')
    throw std::runtime_error("not a GGUF");
  o = 4;
  (void)R_U32(b, o);  // version
  std::uint64_t n_tensors = R_U64(b, o);
  std::uint64_t n_kv = R_U64(b, o);

  GgufFile f;
  f.whole = b;
  f.kv.reserve(n_kv);

  auto read_scalar = [&](ValType t, std::vector<std::uint8_t>& out) {
    switch (t) {
      case ValType::kUint8: case ValType::kInt8: case ValType::kBool: {
        std::uint8_t x = (std::uint8_t)R_U8(b, o); out = {x}; break;
      }
      case ValType::kUint16: case ValType::kInt16: {
        std::uint64_t x = R_U16(b, o); out.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 2); break;
      }
      case ValType::kUint32: case ValType::kInt32: case ValType::kFloat32: {
        std::uint64_t x = R_U32(b, o); out.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 4); break;
      }
      case ValType::kUint64: case ValType::kInt64: case ValType::kFloat64: {
        std::uint64_t x = R_U64(b, o); out.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 8); break;
      }
      default: throw std::runtime_error("unexpected scalar type in read");
    }
  };

  for (std::uint64_t i = 0; i < n_kv; ++i) {
    std::string key = R_STR(b, o);
    ValType vt = (ValType)R_U32(b, o);
    KvValue v; v.type = vt;
    if (vt == ValType::kArray) {
      v.array_type = (ValType)R_U32(b, o);
      std::uint64_t n = R_U64(b, o);
      if (v.array_type == ValType::kString) {
        for (std::uint64_t a = 0; a < n; ++a) v.array_strs.push_back(R_STR(b, o));
      } else {
        for (std::uint64_t a = 0; a < n; ++a) {
          std::vector<std::uint8_t> ev;
          read_scalar(v.array_type, ev);
          v.array_elems.push_back(ev);
        }
      }
    } else if (vt == ValType::kString) {
      v.str = R_STR(b, o);
    } else {
      read_scalar(vt, v.scalar_bytes);
    }
    if (key == "general.alignment" && v.scalar_bytes.size() >= 4) {
      std::uint32_t al = 0; std::memcpy(&al, v.scalar_bytes.data(), 4);
      f.alignment = al ? al : kDefaultAlignment;
    }
    f.kv.emplace_back(key, std::move(v));
  }

  // tensor infos
  for (std::uint64_t i = 0; i < n_tensors; ++i) {
    GgufTensor t;
    t.name = R_STR(b, o);
    std::uint32_t nd = (std::uint32_t)R_U32(b, o);
    for (std::uint32_t d = 0; d < nd; ++d) t.dims.push_back(R_U64(b, o));
    t.type = (Type)R_U32(b, o);
    t.offset = R_U64(b, o);
    f.tensors.push_back(std::move(t));
  }
  // compute sizes (for parser we need type->block; use our BlockBytes)
  for (auto& t : f.tensors) t.size_bytes = TensorNBytes(t.type, t.dims);

  // data base
  std::uint64_t pad = (f.alignment - o % f.alignment) % f.alignment;
  std::uint64_t data_base = o + pad;
  for (auto& t : f.tensors) {
    if (t.offset + t.size_bytes > b.size()) throw std::runtime_error("gguf tensor out of range");
    t.raw.assign(b.begin() + data_base + t.offset, b.begin() + data_base + t.offset + t.size_bytes);
  }
  return f;
}

// ---- write ----
// Serializes a scalar into `out` (little-endian type bytes, no length prefix).
inline void PutScalar(ValType t, const std::vector<std::uint8_t>& bytes,
                      std::vector<std::uint8_t>& out) {
  (void)t;
  out.insert(out.end(), bytes.begin(), bytes.end());
}
inline void PutStr(const std::string& s, std::vector<std::uint8_t>& out) {
  std::uint64_t n = s.size();
  out.insert(out.end(), (std::uint8_t*)&n, (std::uint8_t*)&n + 8);
  out.insert(out.end(), s.begin(), s.end());
}
inline void PutU32(std::uint32_t v, std::vector<std::uint8_t>& out) {
  out.insert(out.end(), (std::uint8_t*)&v, (std::uint8_t*)&v + 4);
}
inline void PutU64(std::uint64_t v, std::vector<std::uint8_t>& out) {
  out.insert(out.end(), (std::uint8_t*)&v, (std::uint8_t*)&v + 8);
}

/// Write an output GGUF for f with new per-tensor types/payloads and extra KV.
/// `extra_kv`: additional KV pairs appended AFTER the input KV and the two
/// required fork keys are already added by caller via param flags.
struct ExtraKv { std::string key; KvValue v; };

/// A serialized output header: the padded header bytes plus the per-tensor
/// data offsets the tensor infos carry. v0.6 splits header construction out
/// of WriteGguf so the quantizer can write the header first (output sizes are
/// a pure function of input shapes + chosen types) and stream each payload as
/// its encode finishes, instead of holding every payload in RAM.
struct GgufOutHeader {
  std::vector<std::uint8_t> bytes;      // header incl. alignment padding
  std::vector<std::uint64_t> offsets;   // per-tensor data offset
};

inline GgufOutHeader BuildGgufHeader(const GgufFile& f,
                                     const std::vector<Type>& new_types,
                                     const std::vector<std::uint64_t>& sizes,
                                     std::uint32_t ftype,
                                     const std::vector<ExtraKv>& extra) {
  GgufOutHeader h;
  std::vector<std::uint8_t> o;
  o.push_back('G'); o.push_back('G'); o.push_back('U'); o.push_back('F');
  PutU32(kGgufVersion, o);

  // Build output KV: copy input kv, then replace-or-append the two required
  // keys (git fork replaces in place via gguf_set_kv), then append extras.
  struct Item { std::string key; KvValue v; };
  std::vector<Item> kv;
  for (auto& [k, v] : f.kv) kv.push_back({k, v});
  auto set_or_append = [&](const std::string& key, const KvValue& nv) {
    for (auto& it : kv) if (it.key == key) { it.v = nv; return; }
    kv.push_back({key, nv});
  };
  { KvValue qv; qv.type = ValType::kUint32; std::uint32_t x = 2;
    qv.scalar_bytes.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 4);
    set_or_append("general.quantization_version", qv); }
  { KvValue fv; fv.type = ValType::kUint32; std::uint32_t x = ftype;
    fv.scalar_bytes.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 4);
    set_or_append("general.file_type", fv); }
  for (auto& e : extra) { Item it; it.key = e.key; it.v = e.v; kv.push_back(it); }

  PutU64(f.tensors.size(), o);   // n_tensors
  PutU64(kv.size(), o);          // n_kv

  auto put_kv = [&](const std::string& key, const KvValue& v) {
    PutStr(key, o);
    if (v.type == ValType::kArray) {
      PutU32((std::uint32_t)ValType::kArray, o);
      PutU32((std::uint32_t)v.array_type, o);
      if (v.array_type == ValType::kString) {
        PutU64(v.array_strs.size(), o);
        for (auto& s : v.array_strs) PutStr(s, o);
      } else {
        PutU64(v.array_elems.size(), o);
        for (auto& e : v.array_elems) PutScalar(v.array_type, e, o);
      }
    } else if (v.type == ValType::kString) {
      PutU32((std::uint32_t)ValType::kString, o);
      PutStr(v.str, o);
    } else {
      PutU32((std::uint32_t)v.type, o);
      PutScalar(v.type, v.scalar_bytes, o);
    }
  };
  for (auto& it : kv) put_kv(it.key, it.v);

  // tensor infos with updated offsets
  std::uint32_t align = f.alignment;
  if (sizes.size() != f.tensors.size())
    throw std::runtime_error("BuildGgufHeader: sizes/tensors mismatch");
  std::vector<std::uint64_t> offsets(f.tensors.size(), 0);
  {
    std::uint64_t run = 0;
    for (size_t i = 0; i < f.tensors.size(); ++i) {
      offsets[i] = run;
      std::uint64_t nb = sizes[i];
      run += ((nb + align - 1) / align) * align;
    }
  }
  auto trimmed = [](const std::vector<std::uint64_t>& d) {
    // ggml_n_dims: write only the leading dims up to the last non-1 dimension.
    // ggml always returns >= 1 (a tensor has at least one dimension), so floor
    // at 1 rather than collapsing all-1 shapes to 0 dims (used e.g. by sparse
    // imatrix counts tensors of a single expert).
    std::size_t n = d.size();
    while (n > 1 && d[n - 1] == 1) --n;
    return n;
  };
  for (size_t i = 0; i < f.tensors.size(); ++i) {
    auto& t = f.tensors[i];
    PutStr(t.name, o);
    std::size_t nd = trimmed(t.dims);
    PutU32((std::uint32_t)nd, o);
    for (std::size_t j = 0; j < nd; ++j) PutU64(t.dims[j], o);
    PutU32(new_types[i], o);
    PutU64(offsets[i], o);
  }

  // pad to alignment
  while (o.size() % align) o.push_back(0);
  h.bytes = std::move(o);
  h.offsets = std::move(offsets);
  return h;
}

/// Incremental output writer: the header bytes go out at construction, then
/// one WritePayload per tensor in index order (each payload + its alignment
/// padding, exactly the layout BuildGgufHeader's offsets describe). When
/// `hash` is non-null every byte written -- header, payloads and padding --
/// is fed to it, so the caller gets the file's sha256 as a side stream of
/// the write with no second pass. Single-threaded by design: one writer.
class GgufStreamWriter {
 public:
  GgufStreamWriter(const std::string& out_path,
                   const std::vector<std::uint8_t>& header_bytes,
                   std::uint32_t alignment, Sha256* hash)
      : align_(alignment ? alignment : 32),
        hash_(hash),
        pos_(0),
        path_(out_path) {
    fo_.open(out_path, std::ios::binary);
    if (!fo_) throw std::runtime_error("cannot write " + out_path);
    WriteRaw(header_bytes.data(), header_bytes.size());
    if (!fo_) throw std::runtime_error("write failed: " + out_path);
  }

  GgufStreamWriter(const GgufStreamWriter&) = delete;
  GgufStreamWriter& operator=(const GgufStreamWriter&) = delete;

  void WritePayload(const void* data, std::uint64_t size) {
    if (size && data) WriteRaw(data, size);
    std::uint64_t pad = (align_ - (pos_ % align_)) % align_;
    if (pad) {
      // Same zero padding WriteGguf emits (and hashes, when hashing).
      static const char kZeros[256] = {};
      std::uint64_t left = pad;
      while (left > 0) {
        std::uint64_t n = left > sizeof(kZeros) ? sizeof(kZeros) : left;
        WriteRaw(kZeros, n);
        left -= n;
      }
    }
    if (!fo_) throw std::runtime_error("write failed: " + path_);
  }

  void Close() {
    fo_.close();
    if (!fo_) throw std::runtime_error("write failed: " + path_);
  }

 private:
  void WriteRaw(const void* data, std::uint64_t size) {
    fo_.write((const char*)data, (std::streamsize)size);
    // v0.8 (F1): a failed write feeds NEITHER the hash NOR the position --
    // the hash stays the digest of the bytes actually accepted, and the pad
    // math cannot drift. Callers throw on !fo_ after the payload/padding.
    if (!fo_) return;
    if (hash_) hash_->Update(data, size);
    pos_ += size;
  }

  std::ofstream fo_;
  std::uint32_t align_;
  Sha256* hash_;  // nullable, owned by the caller
  std::uint64_t pos_;
  std::string path_;  // for error text only
};

inline void WriteGguf(const std::string& out_path, const GgufFile& f,
                      const std::vector<Type>& new_types,
                      const std::vector<std::vector<std::uint8_t>>& payloads,
                      std::uint32_t ftype, const std::vector<ExtraKv>& extra) {
  // Tensor data is streamed straight to disk. Accumulating it into one vector
  // made the vector realloc (capacity doubling) on every growth step, so an
  // 8 GB quant peaked at ~25 GB RSS and a 14 GB one at far more; the byte
  // layout written here is identical (header padded to alignment, then each
  // payload padded). v0.6 routes through the same header builder + streaming
  // writer the quantizer's incremental path uses; the pinned stream-test sha
  // proves the bytes did not move.
  std::vector<std::uint64_t> sizes;
  sizes.reserve(payloads.size());
  for (auto& p : payloads) sizes.push_back(p.size());
  GgufOutHeader h = BuildGgufHeader(f, new_types, sizes, ftype, extra);
  GgufStreamWriter w(out_path, h.bytes, f.alignment, /*hash=*/nullptr);
  for (size_t i = 0; i < f.tensors.size(); ++i) {
    const auto& p = payloads[i];
    w.WritePayload(p.empty() ? nullptr : p.data(), p.size());
  }
  w.Close();
}

}  // namespace gufo::quant_tool

#endif  // GUFO_QUANT_GGUF_HPP_
