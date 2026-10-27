// kv_swap_test — SPEC7 item 2(b): tests for the kv_swap tool.
//
// kv_swap (tools/superroc-quant/kv_swap.cpp) writes a GGUF whose KV/metadata block
// comes verbatim from a template file and whose tensor payloads come verbatim
// from a data file. It is what produced both v0.4 `-servedkv` drop-ins.
//
// This test covers:
//   * one success case: output KV == template, output data == data file,
//     the template header re-serialization self-check passes, tensor infos are
//     the template's, and the data region is byte-identical to the data file's;
//   * --no-provenance: the output is exactly the template header + data region;
//   * five refusal cases, each of which must exit non-zero with a clear message
//     and write NO output file: tensor count mismatch, name/order mismatch,
//     dims mismatch, type mismatch, data-region length mismatch.
//
// Usage: kv_swap_test <path-to-kv_swap> [workdir]

#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gguf.hpp"

namespace gqt = superroc::quant_tool;

namespace {

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

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

struct TensorSpec {
  std::string name;
  std::vector<std::uint64_t> dims;
  std::uint32_t type;
  std::uint8_t byte_base;  // payload byte generator
};

struct Fixture {
  std::vector<std::pair<std::string, gqt::KvValue>> kv;
  std::vector<TensorSpec> tensors;
};

// Write a GGUF fixture with payloads generated from each tensor's byte_base.
// `dims_override`/`type_override`/`count_drop` mutate it for the refusal cases.
std::string WriteFixture(const std::filesystem::path& path, const Fixture& fx,
                         std::uint8_t kv_marker) {
  gqt::GgufFile f;
  f.alignment = 32;
  f.kv.emplace_back("general.architecture", KvStr("superroc-kvswap-test"));
  f.kv.emplace_back("superroc.test.marker", KvU32(kv_marker));
  for (auto& [k, v] : fx.kv)
    f.kv.emplace_back(k, v);

  std::vector<gqt::Type> types;
  std::vector<std::vector<std::uint8_t>> payloads;
  for (auto& ts : fx.tensors) {
    gqt::GgufTensor t;
    t.name = ts.name;
    t.dims = ts.dims;
    t.type = ts.type;
    const std::uint64_t n = gqt::TensorNBytes(ts.type, ts.dims);
    std::vector<std::uint8_t> p(n);
    for (std::uint64_t k = 0; k < n; ++k)
      p[k] = (std::uint8_t)(ts.byte_base + k * 7);
    t.raw = p;
    t.size_bytes = n;
    f.tensors.push_back(std::move(t));
    types.push_back(ts.type);
    payloads.push_back(std::move(p));
  }
  gqt::WriteGguf(path.string(), f, types, payloads, /*ftype=*/0, {});
  return path.string();
}

int RunTool(const std::string& tool, const std::vector<std::string>& args,
            const std::string& log_path) {
  std::string cmd = tool;
  for (auto& a : args)
    cmd += " " + a;
  cmd += " > " + log_path + " 2>&1";
  int rc = std::system(cmd.c_str());
  return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

std::string ReadText(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

// Run an expected-refusal: non-zero exit, no output file, and an error message
// that names the actual mismatch.
void ExpectRefusal(const std::string& tool, const std::string& tpl,
                   const std::string& dat, const std::string& out,
                   const std::string& log, const std::string& what,
                   const std::string& expect_substr) {
  std::error_code ec;
  std::filesystem::remove(out, ec);
  const int rc = RunTool(tool, {tpl, dat, out}, log);
  const std::string msg = ReadText(log);
  Expect(rc != 0, "refusal [" + what + "]: expected non-zero exit, got 0");
  Expect(!std::filesystem::exists(out),
         "refusal [" + what + "]: wrote an output file");
  Expect(!msg.empty(), "refusal [" + what + "]: no error message on stderr");
  Expect(msg.find(expect_substr) != std::string::npos,
         "refusal [" + what + "]: message does not say \"" + expect_substr +
             "\": " + msg);
  std::cout << "  refusal [" << what << "] rc=" << rc
            << " msg=" << msg.substr(0, 96) << (msg.size() > 96 ? "..." : "")
            << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: kv_swap_test <path-to-kv_swap> [workdir]\n";
    return 2;
  }
  const std::string tool = argv[1];
  std::filesystem::path dir;
  if (argc > 2) {
    dir = argv[2];
  } else {
    dir = std::filesystem::temp_directory_path() /
          ("superroc-kv-swap-test-" + std::to_string((long)getpid()));
  }
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir, ec);
  Expect(!ec, "cannot create workdir " + dir.string());

  const std::string log = (dir / "kvswap.log").string();

  const Fixture tpl_fx = {
      {{"superroc.test.tpl_only", KvStr("template-side")}},
      {{"blk.0.attn_q.weight", {64, 2, 1, 1}, 100, 11},
       {"blk.0.attn_v.weight", {64, 2, 1, 1}, 101, 29},
       {"output_norm.weight", {8, 1, 1, 1}, 0, 53}},
  };
  const Fixture dat_fx = {
      {{"superroc.test.data_only",
        KvStr("data-side")}},  // must NOT appear in the output
      {{"blk.0.attn_q.weight", {64, 2, 1, 1}, 100, 200},
       {"blk.0.attn_v.weight", {64, 2, 1, 1}, 101, 201},
       {"output_norm.weight", {8, 1, 1, 1}, 0, 202}},
  };

  const std::string tpl = WriteFixture(dir / "template.gguf", tpl_fx, 1);
  const std::string dat = WriteFixture(dir / "data.gguf", dat_fx, 2);

  gqt::HeaderResult ht;
  {
    std::ifstream f(tpl, std::ios::binary);
    ht = gqt::LoadHeaderStream(f);
  }
  const std::vector<std::uint8_t> tpl_bytes = gqt::ReadFileBytes(tpl);
  const std::vector<std::uint8_t> dat_bytes = gqt::ReadFileBytes(dat);
  const std::uint64_t tpl_region =
      (std::uint64_t)tpl_bytes.size() - ht.data_base;
  const std::uint64_t dat_region =
      (std::uint64_t)dat_bytes.size() - ht.data_base;
  Expect(tpl_region == dat_region && tpl_region > 0,
         "fixture data regions must match");

  // ---- success case ----
  {
    const std::string out = (dir / "swapped.gguf").string();
    const int rc = RunTool(tool, {tpl, dat, out}, log);
    const std::string msg = ReadText(log);
    Expect(rc == 0, "success: kv_swap failed: " + msg);
    Expect(msg.find("HEADER_VERBATIM=yes") != std::string::npos,
           "success: template header self-check did not report "
           "HEADER_VERBATIM=yes");
    Expect(std::filesystem::exists(out), "success: no output file");

    const std::vector<std::uint8_t> outb = gqt::ReadFileBytes(out);
    std::ifstream in(out, std::ios::binary);
    gqt::HeaderResult ho = gqt::LoadHeaderStream(in);

    // KV = template KV + the four provenance keys, in order.
    Expect(ho.kv.size() == ht.kv.size() + 4, "success: output KV count");
    for (std::size_t i = 0; i < ht.kv.size(); ++i)
      Expect(ho.kv[i].first == ht.kv[i].first,
             "success: template KV order at " + std::to_string(i));
    const std::vector<std::string> prov = {
        "superroc.quant.tool", "superroc.quant.kv_template", "superroc.quant.tensor_source",
        "superroc.quant.format_version"};
    for (std::size_t i = 0; i < prov.size(); ++i)
      Expect(ho.kv[ht.kv.size() + i].first == prov[i],
             "success: provenance key at +" + std::to_string(i));
    // The data file's own KV must not leak in.
    for (auto& [k, v] : ho.kv)
      Expect(k != "superroc.test.data_only", "success: data-file KV leaked");

    // Tensor infos are the template's, payloads are the data file's.
    Expect(ho.tensors.size() == ht.tensors.size(), "success: tensor count");
    for (std::size_t i = 0; i < ht.tensors.size(); ++i) {
      Expect(ho.tensors[i].name == ht.tensors[i].name, "success: tensor name");
      Expect(ho.tensors[i].dims == ht.tensors[i].dims, "success: tensor dims");
      Expect(ho.tensors[i].type == ht.tensors[i].type, "success: tensor type");
      Expect(ho.tensors[i].offset == ht.tensors[i].offset,
             "success: tensor offset");
    }
    Expect(outb.size() == ho.data_base + dat_region,
           "success: output size != header + data region");
    Expect(std::memcmp(outb.data() + ho.data_base,
                       dat_bytes.data() + ht.data_base, dat_region) == 0,
           "success: output data region != data file data region");
    std::cout << "  success: rc=0 HEADER_VERBATIM=yes, KV=template+provenance, "
                 "data==data file\n";
  }

  // ---- --no-provenance: output == template header bytes + data region ----
  {
    const std::string out = (dir / "swapped-noprov.gguf").string();
    const int rc = RunTool(tool, {tpl, dat, out, "--no-provenance"}, log);
    Expect(rc == 0, "no-provenance: rc != 0: " + ReadText(log));
    const std::vector<std::uint8_t> outb = gqt::ReadFileBytes(out);
    Expect(outb.size() == ht.data_base + dat_region,
           "no-provenance: output size");
    Expect(std::memcmp(outb.data(), tpl_bytes.data(), ht.data_base) == 0,
           "no-provenance: header is not the template header byte-for-byte");
    Expect(std::memcmp(outb.data() + ht.data_base,
                       dat_bytes.data() + ht.data_base, dat_region) == 0,
           "no-provenance: data region != data file data region");
    std::cout << "  no-provenance: output == template header + data region\n";
  }

  // ---- refusal cases (each must write no output) ----
  {
    Fixture bad = dat_fx;
    bad.tensors.resize(2);  // tensor count mismatch
    const std::string b = WriteFixture(dir / "bad-count.gguf", bad, 2);
    ExpectRefusal(tool, tpl, b, (dir / "r1.gguf").string(), log, "tensor count",
                  "tensor count differs");
  }
  {
    Fixture bad = dat_fx;
    bad.tensors[1].name = "blk.0.renamed.weight";  // name/order mismatch
    const std::string b = WriteFixture(dir / "bad-name.gguf", bad, 2);
    ExpectRefusal(tool, tpl, b, (dir / "r2.gguf").string(), log, "name/order",
                  "name differs");
  }
  {
    Fixture bad = dat_fx;
    bad.tensors[0].dims = {64, 4, 1, 1};  // dims mismatch
    const std::string b = WriteFixture(dir / "bad-dims.gguf", bad, 2);
    ExpectRefusal(tool, tpl, b, (dir / "r3.gguf").string(), log, "dims",
                  "dims differ");
  }
  {
    Fixture bad = dat_fx;
    bad.tensors[0].type = 101;  // type mismatch (both are valid for ne0=64)
    const std::string b = WriteFixture(dir / "bad-type.gguf", bad, 2);
    ExpectRefusal(tool, tpl, b, (dir / "r4.gguf").string(), log, "type",
                  "type differs");
  }
  {
    const std::string b =
        (dir / "bad-region.gguf").string();  // valid data + 32 extra bytes
    std::vector<std::uint8_t> longer = dat_bytes;
    longer.insert(longer.end(), 32, 0);
    gqt::WriteFileBytes(b, longer);
    ExpectRefusal(tool, tpl, b, (dir / "r5.gguf").string(), log,
                  "data-region length", "data region differs");
  }

  std::filesystem::remove_all(dir, ec);
  std::cout << "kv_swap_test PASS\n";
  return 0;
}
