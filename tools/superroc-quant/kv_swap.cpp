// kv_swap — write a GGUF whose KV/metadata block is copied verbatim from a
// template file and whose tensor payloads come verbatim from another file.
//
// SPEC6 item 1: King's 27B seat should change ONLY the weights. The weighted
// 27B differs from the served file in its metadata too (chat template,
// add_bos_token, padding_token_id, general.*/base_model.*), so a drop-in is
// the served file's KV block with the weighted tensor bytes.
//
// Usage:
//   kv_swap <template.gguf> <data.gguf> <out.gguf> [--no-provenance]
//
// The two inputs must have the same tensors in the same order, with the same
// names, dims and types; only the payload bytes differ. The output header is
// re-serialized from the template's parsed KV and tensor infos with the same
// layout superroc-quant's WriteGguf uses (which reproduces the fork's gguf.cpp),
// and the tool self-checks that this re-serialization is byte-identical to the
// template's own header before writing anything. Tensor data is streamed from
// the data file's data region, so no multi-GB buffer is held in memory.
//
// superroc.quant.* provenance keys are appended (the comparator ignores them);
// --no-provenance writes the template header with nothing added.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf.hpp"

namespace gqt = superroc::quant_tool;

// ---- same KV serialization as gguf.hpp WriteGguf::put_kv ----
static void PutKv(const std::string& key, const gqt::KvValue& v,
                  std::vector<std::uint8_t>& o) {
  gqt::PutStr(key, o);
  if (v.type == gqt::ValType::kArray) {
    gqt::PutU32((std::uint32_t)gqt::ValType::kArray, o);
    gqt::PutU32((std::uint32_t)v.array_type, o);
    if (v.array_type == gqt::ValType::kString) {
      gqt::PutU64(v.array_strs.size(), o);
      for (auto& s : v.array_strs) gqt::PutStr(s, o);
    } else {
      gqt::PutU64(v.array_elems.size(), o);
      for (auto& e : v.array_elems) gqt::PutScalar(v.array_type, e, o);
    }
  } else if (v.type == gqt::ValType::kString) {
    gqt::PutU32((std::uint32_t)gqt::ValType::kString, o);
    gqt::PutStr(v.str, o);
  } else {
    gqt::PutU32((std::uint32_t)v.type, o);
    gqt::PutScalar(v.type, v.scalar_bytes, o);
  }
}

static std::size_t TrimmedN(const std::vector<std::uint64_t>& d) {
  std::size_t n = d.size();
  while (n > 1 && d[n - 1] == 1) --n;
  return n;
}

// Serialize magic + version + counts + KV + tensor infos + alignment pad.
// `extra` KV pairs are appended after the template KV (empty for the check).
static std::vector<std::uint8_t> SerializeHeader(
    const gqt::HeaderResult& hr,
    const std::vector<std::pair<std::string, gqt::KvValue>>& extra) {
  std::vector<std::uint8_t> o;
  o.push_back('G'); o.push_back('G'); o.push_back('U'); o.push_back('F');
  gqt::PutU32(gqt::kGgufVersion, o);
  gqt::PutU64(hr.tensors.size(), o);
  gqt::PutU64(hr.kv.size() + extra.size(), o);
  for (auto& [k, v] : hr.kv) PutKv(k, v, o);
  for (auto& [k, v] : extra) PutKv(k, v, o);
  std::uint64_t run = 0;
  for (auto& t : hr.tensors) {
    gqt::PutStr(t.name, o);
    std::size_t nd = TrimmedN(t.dims);
    gqt::PutU32((std::uint32_t)nd, o);
    for (std::size_t j = 0; j < nd; ++j) gqt::PutU64(t.dims[j], o);
    gqt::PutU32(t.type, o);
    gqt::PutU64(run, o);
    std::uint64_t nb = t.size_bytes;
    run += ((nb + hr.alignment - 1) / hr.alignment) * hr.alignment;
  }
  while (o.size() % hr.alignment) o.push_back(0);
  return o;
}

static std::uint64_t FileSize(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + p);
  return (std::uint64_t)f.tellg();
}

int main(int argc, char** argv) {
  std::vector<std::string> args(argv + 1, argv + argc);
  bool provenance = true;
  std::string tpl, dat, out;
  for (auto& a : args) {
    if (a == "--no-provenance") { provenance = false; continue; }
    if (!a.empty() && a[0] == '-') { std::cerr << "unknown option: " << a << "\n"; return 2; }
    if (tpl.empty()) tpl = a;
    else if (dat.empty()) dat = a;
    else if (out.empty()) out = a;
    else { std::cerr << "unexpected extra arg: " << a << "\n"; return 2; }
  }
  if (tpl.empty() || dat.empty() || out.empty()) {
    std::cerr << "usage: kv_swap <template.gguf> <data.gguf> <out.gguf> "
                 "[--no-provenance]\n";
    return 2;
  }

  // ---- parse both headers (no payload bytes) ----
  std::ifstream ft(tpl, std::ios::binary), fd(dat, std::ios::binary);
  if (!ft) { std::cerr << "cannot open " << tpl << "\n"; return 3; }
  if (!fd) { std::cerr << "cannot open " << dat << "\n"; return 3; }
  gqt::HeaderResult ht = gqt::LoadHeaderStream(ft);
  gqt::HeaderResult hd = gqt::LoadHeaderStream(fd);

  if (ht.tensors.size() != hd.tensors.size()) {
    std::cerr << "tensor count differs: " << ht.tensors.size() << " vs "
              << hd.tensors.size() << "\n";
    return 4;
  }
  for (std::size_t i = 0; i < ht.tensors.size(); ++i) {
    const auto& a = ht.tensors[i];
    const auto& b = hd.tensors[i];
    // Report which attribute differs: a dims mismatch between same-named,
    // same-typed tensors would otherwise print an identical-looking line.
    if (a.name != b.name) {
      std::cerr << "tensor " << i << " name differs: " << a.name << " vs "
                << b.name << "\n";
      return 4;
    }
    if (a.dims != b.dims) {
      std::cerr << "tensor " << i << " (" << a.name << ") dims differ: [";
      for (std::size_t d = 0; d < a.dims.size(); ++d) std::cerr << (d ? "," : "") << a.dims[d];
      std::cerr << "] vs [";
      for (std::size_t d = 0; d < b.dims.size(); ++d) std::cerr << (d ? "," : "") << b.dims[d];
      std::cerr << "]\n";
      return 4;
    }
    if (a.type != b.type) {
      std::cerr << "tensor " << i << " (" << a.name << ") type differs: " << a.type
                << " vs " << b.type << "\n";
      return 4;
    }
  }
  std::uint64_t st = FileSize(tpl), sd = FileSize(dat);
  if (st < ht.data_base || sd < hd.data_base) { std::cerr << "bad data base\n"; return 4; }
  std::uint64_t dt = st - ht.data_base, dd = sd - hd.data_base;
  if (dt != dd) {
    std::cerr << "data region differs: " << dt << " vs " << dd << "\n";
    return 4;
  }
  std::cerr << "tensors=" << ht.tensors.size() << " data_region=" << dt
            << " alignment=" << ht.alignment << "\n";

  // ---- self-check: re-serialized template header must be byte-identical ----
  std::vector<std::uint8_t> hdr_check = SerializeHeader(ht, {});
  std::vector<std::uint8_t> tpl_head(ht.data_base);
  ft.clear(); ft.seekg(0, std::ios::beg);
  ft.read((char*)tpl_head.data(), (std::streamsize)ht.data_base);
  bool verbatim = (hdr_check.size() == tpl_head.size()) &&
                  std::memcmp(hdr_check.data(), tpl_head.data(), hdr_check.size()) == 0;
  std::cerr << "HEADER_VERBATIM=" << (verbatim ? "yes" : "no") << "\n";
  if (!verbatim) {
    std::size_t n = std::min(hdr_check.size(), tpl_head.size());
    std::size_t first = n;
    for (std::size_t i = 0; i < n; ++i) if (hdr_check[i] != tpl_head[i]) { first = i; break; }
    std::cerr << "  sizes " << hdr_check.size() << " vs " << tpl_head.size()
              << " first_diff=" << first << "\n";
    return 5;
  }

  // ---- provenance (appended after the template KV) ----
  std::vector<std::pair<std::string, gqt::KvValue>> extra;
  auto str_kv = [](const std::string& s) {
    gqt::KvValue v; v.type = gqt::ValType::kString; v.str = s; return v;
  };
  if (provenance) {
    extra.emplace_back("superroc.quant.tool", str_kv("superroc-quant 0.1 kv_swap"));
    extra.emplace_back("superroc.quant.kv_template", str_kv(tpl));
    extra.emplace_back("superroc.quant.tensor_source", str_kv(dat));
    { gqt::KvValue v; v.type = gqt::ValType::kUint32; std::uint32_t x = 1;
      v.scalar_bytes.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 4);
      extra.emplace_back("superroc.quant.format_version", v); }
  }
  std::vector<std::uint8_t> hdr = SerializeHeader(ht, extra);

  // ---- write header + streamed data region ----
  {
    std::ofstream fo(out, std::ios::binary);
    if (!fo) { std::cerr << "cannot write " << out << "\n"; return 6; }
    fo.write((const char*)hdr.data(), (std::streamsize)hdr.size());
    fd.clear(); fd.seekg((std::streamoff)hd.data_base, std::ios::beg);
    std::vector<char> buf(64u << 20);
    std::uint64_t left = dd, done = 0;
    while (left > 0) {
      std::size_t want = (std::size_t)std::min<std::uint64_t>(left, buf.size());
      fd.read(buf.data(), (std::streamsize)want);
      if ((std::size_t)fd.gcount() != want) { std::cerr << "short read\n"; return 7; }
      fo.write(buf.data(), (std::streamsize)want);
      if (!fo) { std::cerr << "write failed\n"; return 7; }
      left -= want; done += want;
    }
    fo.flush();
    std::cerr << "copied " << done << " data bytes\n";
  }
  std::cerr << "wrote " << out << "\n";
  return 0;
}
