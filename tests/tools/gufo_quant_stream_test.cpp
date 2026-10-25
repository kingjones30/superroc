// gufo_quant_stream_test — SPEC7 item 2(a): the streaming GGUF writer.
//
// `tools/gufo-quant/gguf.hpp` WriteGguf used to build the whole output file in
// one std::vector and write it in a single call; v0.4 (gq6) streams tensor
// payloads straight to disk, keeping only the header in memory. The byte layout
// written is unchanged. This test pins that writer:
//
//   Part A  a fixed-payload fixture is written with the tree's WriteGguf and
//           checked against (i) the exact documented layout (re-parsed from the
//           file: KV order/values, tensor infos, per-tensor offsets, padding,
//           data-region bytes) and (ii) a pinned sha256 of the whole file.
//           The fixture's payloads are literal bytes, so the pin is a pure
//           function of the writer's layout -- no floating point, no encoder,
//           identical on every architecture.
//   Part B  the tree's gufo-quant binary is run at 1, 4 and 8 threads on a
//           deterministic in-process BF16 synth source; all outputs must be
//           byte-identical (the writer is deterministic, the chunk tasks +
//           pool move no byte at any thread count).
//   Part C  SHA-256 known answers (empty / "abc" / 1M 'a') through the tool's
//           own sha256.hpp class, exercising whichever compression block the
//           runtime picked (scalar, or SHA-NI where the CPU has it).
//
// The end-to-end proof that this streaming writer reproduces the pre-streaming
// writer byte-for-byte on make_synth / make_moe_synth / 2-shard fixtures is in
// REPORT7.md (hand-run, `cmp`): the two binaries differ only in this function.
//
// Usage: gufo_quant_stream_test <path-to-gufo-quant> [workdir]

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gguf.hpp"
#include "sha256.hpp"

namespace gqt = gufo::quant_tool;

namespace {

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

// ---- SHA-256 (FIPS 180-4): the tool's own implementation (sha256.hpp), the
// one proven equal to `shasum`. v0.7 consolidates what used to be a local
// duplicate of the whole hash into this include (single source of truth);
// the Part A pin is UNCHANGED, which also proves the old local table and
// the tool's table agree bit-for-bit. ----
using gqt::Sha256Hex;

gqt::KvValue KvStr(const std::string& s) {
  gqt::KvValue v;
  v.type = gqt::ValType::kString;
  v.str = s;
  return v;
}
gqt::KvValue KvU32(std::uint32_t x) {
  gqt::KvValue v;
  v.type = gqt::ValType::kUint32;
  v.scalar_bytes.assign((const std::uint8_t*)&x, (const std::uint8_t*)&x + 4);
  return v;
}
gqt::KvValue KvBool(bool b) {
  gqt::KvValue v;
  v.type = gqt::ValType::kBool;
  v.scalar_bytes = {(std::uint8_t)(b ? 1 : 0)};
  return v;
}
gqt::KvValue KvU32Array(const std::vector<std::uint32_t>& xs) {
  gqt::KvValue v;
  v.type = gqt::ValType::kArray;
  v.array_type = gqt::ValType::kUint32;
  for (std::uint32_t x : xs)
    v.array_elems.push_back({(std::uint8_t)(x), (std::uint8_t)(x >> 8),
                             (std::uint8_t)(x >> 16), (std::uint8_t)(x >> 24)});
  return v;
}
gqt::KvValue KvStrArray(const std::vector<std::string>& xs) {
  gqt::KvValue v;
  v.type = gqt::ValType::kArray;
  v.array_type = gqt::ValType::kString;
  v.array_strs = xs;
  return v;
}

// Deterministic literal payload: byte k of tensor i.
std::vector<std::uint8_t> Payload(std::size_t i, std::size_t n) {
  std::vector<std::uint8_t> p(n);
  for (std::size_t k = 0; k < n; ++k)
    p[k] = (std::uint8_t)(3 + 11 * k + 37 * i);
  return p;
}

std::uint64_t Aligned(std::uint64_t n, std::uint64_t a) {
  return ((n + a - 1) / a) * a;
}

// ggml_n_dims: the writer stores only the leading dims up to the last non-1
// one.
std::vector<std::uint64_t> Trimmed(std::vector<std::uint64_t> d) {
  std::size_t n = d.size();
  while (n > 1 && d[n - 1] == 1)
    --n;
  d.resize(n);
  return d;
}

// ---- Part A: fixed-payload fixture through WriteGguf -----------------------
void PartA(const std::filesystem::path& dir) {
  const std::uint64_t align = 32;
  gqt::GgufFile f;
  f.alignment = (std::uint32_t)align;
  f.kv.emplace_back("general.architecture", KvStr("gufo-test"));
  f.kv.emplace_back("gufo.test.number", KvU32(7));
  f.kv.emplace_back("gufo.test.flag", KvBool(true));
  f.kv.emplace_back("gufo.test.int_array", KvU32Array({1, 2, 3}));
  f.kv.emplace_back("gufo.test.str_array", KvStrArray({"a", "bb"}));

  auto add = [&](const std::string& name, std::vector<std::uint64_t> dims,
                 std::uint32_t type) {
    gqt::GgufTensor t;
    t.name = name;
    t.dims = std::move(dims);
    t.type = type;
    f.tensors.push_back(std::move(t));
  };
  add("blk.0.attn_q.weight", {256, 2, 1, 1},
      13);  // Q5_K: 176 B / 256 elems -> 352 B
  add("blk.0.attn_v.weight", {64, 3, 1, 1},
      101);  // FAST: 17 B / 32 elems -> 34 B/row -> 102 B
  add("output_norm.weight", {5, 1, 1, 1},
      0);  // F32  -> 20 B (ndims trims to 1)
  add("output.weight", {32, 2, 1, 1},
      8);  // Q8_0: 34 B / 32 elems -> 34 B/row -> 68 B

  std::vector<gqt::Type> types;
  std::vector<std::vector<std::uint8_t>> payloads;
  std::vector<std::uint64_t> expect_offsets;
  {
    std::uint64_t run = 0;
    for (std::size_t i = 0; i < f.tensors.size(); ++i) {
      const auto& t = f.tensors[i];
      const std::uint64_t nb = gqt::TensorNBytes(t.type, t.dims);
      expect_offsets.push_back(run);
      run += Aligned(nb, align);
      types.push_back(t.type);
      payloads.push_back(Payload(i, nb));
    }
  }
  std::vector<gqt::ExtraKv> extra;
  {
    gqt::ExtraKv e;
    e.key = "gufo.quant.tool";
    e.v = KvStr("gufo-quant-stream-test");
    extra.push_back(e);
  }

  const std::string path = (dir / "writer-fixed.gguf").string();
  gqt::WriteGguf(path, f, types, payloads, /*ftype=*/105, extra);

  // ---- re-parse and verify every structural property of the layout ----
  std::ifstream in(path, std::ios::binary);
  Expect((bool)in, "Part A: output missing");
  gqt::HeaderResult hr = gqt::LoadHeaderStream(in);
  Expect(hr.alignment == align, "Part A: alignment not preserved");
  Expect(hr.tensors.size() == f.tensors.size(), "Part A: tensor count");
  for (std::size_t i = 0; i < f.tensors.size(); ++i) {
    Expect(hr.tensors[i].name == f.tensors[i].name,
           "Part A: tensor name order");
    Expect(hr.tensors[i].dims == Trimmed(f.tensors[i].dims),
           "Part A: tensor dims (trimmed)");
    Expect(hr.tensors[i].type == f.tensors[i].type, "Part A: tensor type");
    Expect(hr.tensors[i].offset == expect_offsets[i],
           "Part A: tensor offset not 32-byte aligned pack");
  }
  // KV: input KV in order, then quantization_version + file_type, then extras.
  const std::vector<std::pair<std::string, gqt::ValType>> expect_kv_types = {
      {"general.architecture", gqt::ValType::kString},
      {"gufo.test.number", gqt::ValType::kUint32},
      {"gufo.test.flag", gqt::ValType::kBool},
      {"gufo.test.int_array", gqt::ValType::kArray},
      {"gufo.test.str_array", gqt::ValType::kArray},
      {"general.quantization_version", gqt::ValType::kUint32},
      {"general.file_type", gqt::ValType::kUint32},
      {"gufo.quant.tool", gqt::ValType::kString},
  };
  Expect(hr.kv.size() == expect_kv_types.size(), "Part A: KV count");
  for (std::size_t i = 0; i < expect_kv_types.size(); ++i) {
    Expect(hr.kv[i].first == expect_kv_types[i].first,
           "Part A: KV key order at " + std::to_string(i));
    Expect(hr.kv[i].second.type == expect_kv_types[i].second,
           "Part A: KV type at " + std::to_string(i));
  }
  {
    std::uint32_t qv = 0, ft = 0;
    std::memcpy(&qv, hr.kv[5].second.scalar_bytes.data(), 4);
    std::memcpy(&ft, hr.kv[6].second.scalar_bytes.data(), 4);
    Expect(qv == 2, "Part A: general.quantization_version must be 2");
    Expect(ft == 105, "Part A: general.file_type must be the recipe's");
    Expect(hr.kv[1].second.scalar_bytes == KvU32(7).scalar_bytes,
           "Part A: uint32 KV value");
    Expect(hr.kv[3].second.array_elems == KvU32Array({1, 2, 3}).array_elems,
           "Part A: uint32 array KV");
    Expect(hr.kv[4].second.array_strs == std::vector<std::string>({"a", "bb"}),
           "Part A: string array KV");
  }
  // Data region: payload bytes, each padded with zeros to the alignment.
  const std::vector<std::uint8_t> whole = gqt::ReadFileBytes(path);
  std::uint64_t expect_size = hr.data_base;
  for (std::size_t i = 0; i < payloads.size(); ++i)
    expect_size += Aligned(payloads[i].size(), align);
  Expect(whole.size() == expect_size,
         "Part A: file size != header + aligned payloads");
  for (std::size_t i = 0; i < payloads.size(); ++i) {
    const std::uint64_t off = hr.data_base + expect_offsets[i];
    Expect(std::memcmp(whole.data() + off, payloads[i].data(),
                       payloads[i].size()) == 0,
           "Part A: payload bytes at tensor " + std::to_string(i));
    for (std::uint64_t k = payloads[i].size();
         k < Aligned(payloads[i].size(), align); ++k)
      Expect(whole[off + k] == 0,
             "Part A: padding not zero at tensor " + std::to_string(i));
  }

  // ---- pinned sha256 (layout-only fixture: no encoder, no FP) ----
  const std::string sha = Sha256Hex(whole);
  // Pinned on 2026-10-05 from the fixed-payload fixture above (1216 bytes).
  // The fixture has literal payload bytes and an integer-only layout, so this
  // value is toolchain- and architecture-independent; it changed only if
  // WriteGguf's byte layout changed. Unchanged through the v0.7 hash
  // consolidation (same digest from sha256.hpp).
  const std::string kPinnedSha256 =
      "5d95729f7b39aefd743ed856d8d23937f53d38d818fb6f614b35238cc14eb2d8";
  std::cout << "Part A writer sha256 = " << sha << " (" << whole.size()
            << " bytes)\n";
  Expect(sha == kPinnedSha256, "Part A: streaming writer output changed; got " +
                                   sha + " want " + kPinnedSha256);
}

// ---- Part B: tree's gufo-quant binary on a deterministic synth source ------
void PartB(const std::string& gufo_quant, const std::filesystem::path& dir) {
  gqt::GgufFile src;
  src.alignment = 32;
  src.kv.emplace_back("general.architecture", KvStr("qwen35"));
  src.kv.emplace_back("general.block_count", KvU32(2));
  auto add = [&](const std::string& name, std::vector<std::uint64_t> dims) {
    gqt::GgufTensor t;
    t.name = name;
    t.dims = std::move(dims);
    t.type = 30;  // BF16 source
    std::uint64_t nelem = 1;
    for (auto d : t.dims)
      nelem *= d;
    for (std::uint64_t k = 0; k < nelem;
         ++k) {  // deterministic BF16-ish pattern
      std::uint16_t h = (std::uint16_t)(((k * 2654435761u) >> 16) & 0x7FFFu);
      t.raw.push_back((std::uint8_t)(h & 0xFF));
      t.raw.push_back((std::uint8_t)(h >> 8));
    }
    t.size_bytes = t.raw.size();
    src.tensors.push_back(std::move(t));
  };
  add("blk.0.attn_q.weight", {128, 128, 1, 1});
  add("blk.0.ffn_up.weight", {128, 256, 1, 1});
  add("blk.0.ffn_down.weight", {256, 128, 1, 1});
  add("token_embd.weight", {128, 32, 1, 1});
  add("output_norm.weight", {128, 1, 1, 1});
  add("output.weight", {128, 32, 1, 1});
  add("blk.1.ffn_big.weight", {1024, 2048, 1, 1});  // 2M elems: multi-chunk

  std::vector<gqt::Type> types(src.tensors.size(), 30);
  std::vector<std::vector<std::uint8_t>> payloads;
  payloads.reserve(src.tensors.size());
  for (auto& t : src.tensors)
    payloads.push_back(t.raw);
  const std::string src_path = (dir / "synth-bf16.gguf").string();
  gqt::WriteGguf(src_path, src, types, payloads, /*ftype=*/0, {});

  const std::string out1 = (dir / "synth-strix-1.gguf").string();
  const std::string out4 = (dir / "synth-strix-4.gguf").string();
  const std::string out8 = (dir / "synth-strix-8.gguf").string();
  const std::string log = (dir / "quant.log").string();
  auto run = [&](const std::string& out, int n) {
    const std::string cmd = gufo_quant + " " + src_path + " " + out +
                            " STRIX " + std::to_string(n) + " > " + log +
                            " 2>&1";
    return std::system(cmd.c_str());
  };
  Expect(run(out1, 1) == 0,
         "Part B: gufo-quant run 1 failed (see " + log + ")");
  Expect(run(out4, 4) == 0,
         "Part B: gufo-quant run 2 failed (see " + log + ")");
  Expect(run(out8, 8) == 0,
         "Part B: gufo-quant run 3 failed (see " + log + ")");

  const std::vector<std::uint8_t> b1 = gqt::ReadFileBytes(out1);
  const std::vector<std::uint8_t> b4 = gqt::ReadFileBytes(out4);
  const std::vector<std::uint8_t> b8 = gqt::ReadFileBytes(out8);
  Expect(!b1.empty(), "Part B: run 1 wrote an empty file");
  Expect(b1 == b4,
         "Part B: the quantizer + streaming writer is not deterministic");
  // v0.6: chunk tasks + the persistent pool must not move a byte at any
  // thread count (the big tensor splits into chunks, the small ones steal).
  Expect(b1 == b8, "Part B: output depends on thread count (1 vs 8)");
  std::ifstream in(out1, std::ios::binary);
  gqt::HeaderResult hr = gqt::LoadHeaderStream(in);
  Expect(hr.tensors.size() == src.tensors.size(),
         "Part B: output tensor count");
  const std::uint64_t region = (std::uint64_t)b1.size() - hr.data_base;
  Expect(region > 0 && region % hr.alignment == 0,
         "Part B: data region not aligned");
  std::cout << "Part B gufo-quant STRIX: threads 1/4/8 byte-identical, "
            << b1.size() << " bytes, tensors=" << hr.tensors.size() << "\n";
}

// ---- Part C: SHA-256 known answers (FIPS 180-4 / NIST examples) ---------
// v0.7: --manifest hashes whole input shards through this class; these
// vectors pin the implementation — and its SHA-NI block where the CPU has
// it — no matter which compression path the runtime dispatch picked.
void PartC() {
  gqt::Sha256 e;
  Expect(e.Hex() ==
             "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
         "Part C: empty-message KAT");

  gqt::Sha256 abc;
  abc.Update("abc", 3);
  Expect(abc.Hex() ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "Part C: 'abc' KAT");

  // one million 'a' fed in awkward chunk sizes (1, 7, 64, 1000) so the
  // 64-byte buffering sees partial blocks at every boundary case
  gqt::Sha256 mil;
  const std::string kMilWant =
      "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
  const std::uint64_t sizes[] = {1, 7, 64, 1000};
  std::uint64_t fed = 0;
  std::size_t si = 0;
  while (fed < 1000000) {
    std::uint64_t take = sizes[si++ % 4];
    if (fed + take > 1000000)
      take = 1000000 - fed;
    const std::string a(take, 'a');
    mil.Update(a.data(), a.size());
    fed += take;
  }
  Expect(mil.Hex() == kMilWant, "Part C: 1M-'a' KAT (buffered chunks)");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr
        << "usage: gufo_quant_stream_test <path-to-gufo-quant> [workdir]\n";
    return 2;
  }
  const std::string gufo_quant = argv[1];
  std::filesystem::path dir;
  if (argc > 2) {
    dir = argv[2];
  } else {
    dir = std::filesystem::temp_directory_path() /
          ("gufo-quant-stream-test-" + std::to_string((long)getpid()));
  }
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir, ec);
  Expect(!ec, "cannot create workdir " + dir.string());

  PartA(dir);
  PartB(gufo_quant, dir);
  PartC();

  std::filesystem::remove_all(dir, ec);
  std::cout << "gufo_quant_stream_test PASS\n";
  return 0;
}
