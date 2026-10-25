// compare_gguf — byte-compare two GGUF files: every KV (types/values) and every
// tensor (name, dims, type, payload bytes). Ignores keys starting `gufo.quant.`
// (the additive provenance keys). Reports first mismatch. Exit 0 = identical.

#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "gguf.hpp"

namespace gqt = gufo::quant_tool;

static std::string ValStr(const gqt::KvValue& v) {
  std::string s;
  switch (v.type) {
    case gqt::ValType::kUint8: case gqt::ValType::kInt8:
    case gqt::ValType::kUint16: case gqt::ValType::kInt16:
    case gqt::ValType::kUint32: case gqt::ValType::kInt32:
    case gqt::ValType::kFloat32: case gqt::ValType::kUint64:
    case gqt::ValType::kInt64: case gqt::ValType::kFloat64: case gqt::ValType::kBool: {
      for (auto b : v.scalar_bytes) { char tmp[8]; std::snprintf(tmp, sizeof tmp, "%02x", b); s += tmp; }
      return s;
    }
    case gqt::ValType::kString: return "S:" + v.str;
    case gqt::ValType::kArray: {
      s = "A[";
      if (v.array_type == gqt::ValType::kString) {
        for (auto& st : v.array_strs) s += "S:" + st + ";";
      } else {
        for (auto& e : v.array_elems) {
          for (auto b : e) { char t[8]; std::snprintf(t, sizeof t, "%02x", b); s += t; }
          s += ";";
        }
      }
      return s + "]";
    }
  }
  return "?";
}

int main(int argc, char** argv) {
  if (argc < 3) { std::cerr << "usage: compare_gguf a.gguf b.gguf\n"; return 2; }
  auto a = gqt::LoadGguf(argv[1]);
  auto b = gqt::LoadGguf(argv[2]);

  // KV compare (skip tool-provenance keys in both: gufo-quant adds gufo.quant.*
  // and the fork adds quantize.imatrix.* (+ quantize.*); ignore order, keys unique)
  auto skip = [](const std::string& k) {
    return k.rfind("gufo.quant.", 0) == 0 || k.rfind("quantize.", 0) == 0;
  };
  std::map<std::string, std::string> ka, kb;
  for (auto& [k, v] : a.kv) if (!skip(k)) ka[k] = ValStr(v);
  for (auto& [k, v] : b.kv) if (!skip(k)) kb[k] = ValStr(v);
  for (auto& [k, val] : ka) {
    if (!kb.count(k)) { std::cout << "KV only in A: " << k << "\n"; return 1; }
    if (kb[k] != val) { std::cout << "KV DIFF " << k << ":\n  A=" << val << "\n  B=" << kb[k] << "\n"; return 1; }
  }
  for (auto& [k, _] : kb) {
    if (!ka.count(k)) { std::cout << "KV only in B: " << k << "\n"; return 1; }
  }
  std::cout << "KV match (" << ka.size() << " shared keys)\n";

  if (a.tensors.size() != b.tensors.size()) {
    std::cout << "tensor count diff: A=" << a.tensors.size() << " B=" << b.tensors.size() << "\n"; return 1;
  }
  for (size_t i = 0; i < a.tensors.size(); ++i) {
    auto& ta = a.tensors[i], tb = b.tensors[i];
    if (ta.name != tb.name) { std::cout << "tensor name diff at " << i << ": " << ta.name << " vs " << tb.name << "\n"; return 1; }
    if (ta.type != tb.type) { std::cout << "tensor type diff " << ta.name << ": A=" << gqt::TypeName(ta.type) << " B=" << gqt::TypeName(tb.type) << "\n"; return 1; }
    if (ta.dims != tb.dims) { std::cout << "tensor dims diff " << ta.name << "\n"; return 1; }
    if (ta.raw.size() != tb.raw.size()) { std::cout << "tensor size diff " << ta.name << ": A=" << ta.raw.size() << " B=" << tb.raw.size() << "\n"; return 1; }
    if (std::memcmp(ta.raw.data(), tb.raw.data(), ta.raw.size()) != 0) {
      // find first differing byte
      std::uint64_t first = 0;
      while (first < ta.raw.size() && ta.raw[first] == tb.raw[first]) first++;
      std::cout << "tensor DATA diff " << ta.name << " at byte " << first << " (" << gqt::TypeName(ta.type) << ")\n";
      return 1;
    }
  }
  std::cout << "ALL " << a.tensors.size() << " tensors byte-identical\n";
  std::cout << "RESULT: MATCH\n";
  return 0;
}
