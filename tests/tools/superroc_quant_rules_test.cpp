// superroc_quant_rules_test — LQP SPEC1 item 6: superroc-quant v0.5 rules,
// protect files, fork head flags, census, and manifest.
//
//   Part 1  in-process rule matching: exact/anchor/glob/substring + kinds.
//   Part 2  in-process protect-file parsing: comments, `RULE TYPE # why`,
//           unknown-type and malformed-line errors.
//   Part 3  in-process census helpers: family normalization (pinned to
//           gguf_census.py, quirk included) and per-type byte sizes.
//   Part 4  the built superroc-quant binary on a tiny deterministic BF16 synth:
//           =exact vs substring, first-match-wins, protect precedence and the
//           zero-match error (+ --protect-allow-unmatched), the fork head
//           flags (incl. tied-model no-op and --leave-output-tensor gating),
//           --tensor-type-trace, census text/JSON, and --manifest receipts.
//
// Usage: superroc_quant_rules_test <path-to-superroc-quant> [workdir]

#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "census.hpp"
#include "gguf.hpp"
#include "rules.hpp"
#include "sha256.hpp"

namespace gqt = superroc::quant_tool;

namespace {

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  Expect(!!f, "cannot open " + path);
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

void WriteFile(const std::string& path, const std::string& s) {
  std::ofstream f(path, std::ios::binary);
  Expect(!!f, "cannot write " + path);
  f << s;
}

// ---- Part 1: rule matching ----------------------------------------------
void Part1() {
  auto R = [](const std::string& spec) {
    return gqt::ParseRule(spec, (gqt::Type)8, "cli");
  };
  // The motivating case (SPEC1 item 1): substring `output.weight` hits
  // blk.N.attn_output.weight; `=output.weight` does not.
  gqt::Rule sub = R("output.weight");
  Expect(sub.kind == gqt::RuleKind::kSubstring, "P1: plain is substring");
  Expect(sub.Matches("output.weight"), "P1: substring hits output");
  Expect(sub.Matches("blk.0.attn_output.weight"),
         "P1: substring hits attn_output");
  gqt::Rule ex = R("=output.weight");
  Expect(ex.kind == gqt::RuleKind::kExact, "P1: = is exact");
  Expect(ex.Matches("output.weight"), "P1: exact hits output");
  Expect(!ex.Matches("blk.0.attn_output.weight"),
         "P1: exact misses attn_output");
  Expect(!ex.Matches("output.weightx"), "P1: exact misses prefix");
  // Anchors.
  gqt::Rule pre = R("^blk.0.");
  Expect(pre.kind == gqt::RuleKind::kPrefix, "P1: ^ is prefix");
  Expect(pre.Matches("blk.0.attn_q.weight"), "P1: prefix hits blk.0");
  Expect(!pre.Matches("blk.10.attn_q.weight"), "P1: prefix misses blk.10");
  gqt::Rule suf = R("attn_k.weight$");
  Expect(suf.kind == gqt::RuleKind::kSuffix, "P1: $ is suffix");
  Expect(suf.Matches("blk.3.attn_k.weight"), "P1: suffix hits");
  Expect(!suf.Matches("blk.3.attn_k.weightx"), "P1: suffix misses");
  Expect(!suf.Matches("blk.3.xattn_k.weighty"), "P1: suffix misses 2");
  // Globs: '*' is any run within the one name (may cross dots, may be empty).
  gqt::Rule g1 = R("blk.*.ssm_alpha.weight");
  Expect(g1.kind == gqt::RuleKind::kGlob, "P1: * is glob");
  Expect(g1.Matches("blk.3.ssm_alpha.weight"), "P1: glob hits blk.3");
  Expect(g1.Matches("blk.12.ssm_alpha.weight"), "P1: glob hits blk.12");
  Expect(!g1.Matches("blk.3.ssm_beta.weight"), "P1: glob misses beta");
  Expect(!g1.Matches("blk.3.ssm_alpha.weightx"), "P1: glob is full-match");
  Expect(!g1.Matches("xblk.3.ssm_alpha.weight"), "P1: glob anchored front");
  gqt::Rule g2 = R("*ffn_gate_inp.weight");
  Expect(g2.Matches("blk.0.ffn_gate_inp.weight"), "P1: lead-star hits router");
  Expect(!g2.Matches("blk.0.ffn_gate_inp_shexp.weight"),
         "P1: lead-star misses shexp");
  gqt::Rule g3 = R("blk.*.attn_*");
  Expect(g3.Matches("blk.7.attn_qkv.weight"), "P1: two-star hits");
  Expect(!g3.Matches("blk.7.ffn_up.weight"), "P1: two-star misses");
  // Degenerate: '*' and '' match everything ('' is the pre-v0.5 `=Q8_0` form).
  Expect(R("*").Matches("anything.at.all"), "P1: star matches all");
  Expect(R("").Matches("anything.at.all"), "P1: empty matches all");
  // Anchors around a glob core are accepted (redundant: globs full-match).
  gqt::Rule g4 = R("^blk.*.weight$");
  Expect(g4.kind == gqt::RuleKind::kGlob, "P1: anchored glob is glob");
  Expect(g4.Matches("blk.3.attn_q.weight"), "P1: anchored glob hits");
  // '=' disables globbing.
  gqt::Rule g5 = R("=blk.*.weight");
  Expect(g5.kind == gqt::RuleKind::kExact, "P1: = disables glob");
  Expect(!g5.Matches("blk.3.attn_q.weight"), "P1: literal star misses");
  std::cout << "Part 1 rule matching PASS\n";
}

// ---- Part 2: protect files -----------------------------------------------
void Part2(const std::filesystem::path& dir) {
  const std::string pp = (dir / "p.txt").string();
  WriteFile(pp,
            "# qwen35 v0 (test)\n"
            "\n"
            "=output.weight Q6_K  # head: exact, never 4-bit\n"
            "*ssm_alpha.weight q8_0\n"
            "  *ssm_beta.weight\tQ8_0  # tab separated\n");
  auto rules = gqt::ParseProtectFile(pp);
  Expect(rules.size() == 3, "P2: three rules");
  Expect(rules[0].kind == gqt::RuleKind::kExact, "P2: rule 0 exact");
  Expect(rules[0].type == (gqt::Type)14, "P2: rule 0 Q6_K");
  Expect(rules[0].source == "protect:" + pp + ":3",
         "P2: rule 0 source, got " + rules[0].source);
  Expect(rules[1].kind == gqt::RuleKind::kGlob, "P2: rule 1 glob");
  Expect(rules[1].type == (gqt::Type)8, "P2: rule 1 q8_0 lower");
  Expect(rules[2].text == "*ssm_beta.weight", "P2: rule 2 text");
  const std::string bad = (dir / "bad.txt").string();
  WriteFile(bad, "=output.weight Q9_Z\n");
  bool threw = false;
  try {
    (void)gqt::ParseProtectFile(bad);
  } catch (const std::exception& e) {
    threw = true;
    Expect(std::string(e.what()).find("unknown type") != std::string::npos,
           "P2: unknown-type message");
  }
  Expect(threw, "P2: unknown type throws");
  WriteFile(bad, "lonelyrule\n");
  threw = false;
  try {
    (void)gqt::ParseProtectFile(bad);
  } catch (...) {
    threw = true;
  }
  Expect(threw, "P2: malformed line throws");
  std::cout << "Part 2 protect parsing PASS\n";
}

// ---- Part 3: census helpers -----------------------------------------------
void Part3() {
  Expect(gqt::CensusFamily("blk.12.attn_q.weight") == "blk.N.attn_q.weight",
         "P3: blk family");
  Expect(gqt::CensusFamily("token_embd.weight") == "token_embd.weight",
         "P3: plain family");
  // gguf_census.py quirk: the consumed trailing dot is not re-scanned.
  Expect(gqt::CensusFamily("a.1.2.b") == "a.#.2.b", "P3: numeric quirk");
  // Every CensusTensorBytes enum row, pinned against ggml's GGML_TYPE_SIZES
  // (repo ground truth: qfn_ggml_stubs.h). Each row: type, block elems, block
  // bytes. Probed with 3 rows so the row multiply is pinned too.
  struct TypeRow {
    gqt::Type t;
    const char* name;
    std::uint64_t be, bb;
  };
  const TypeRow kRows[] = {
      {0, "F32", 1, 4},
      {1, "F16", 1, 2},
      {2, "Q4_0", 32, 18},
      {3, "Q4_1", 32, 20},
      {6, "Q5_0", 32, 22},
      {7, "Q5_1", 32, 24},
      {8, "Q8_0", 32, 34},
      {9, "Q8_1", 32, 36},
      {10, "Q2_K", 256, 84},
      {11, "Q3_K", 256, 110},
      {12, "Q4_K", 256, 144},
      {13, "Q5_K", 256, 176},
      {14, "Q6_K", 256, 210},
      {15, "Q8_K", 256, 292},
      {16, "IQ2_XXS", 256, 66},
      {17, "IQ2_XS", 256, 74},
      {18, "IQ3_XXS", 256, 98},
      {19, "IQ1_S", 256, 50},
      {20, "IQ4_NL", 32, 18},
      {21, "IQ3_S", 256, 110},
      {22, "IQ2_S", 256, 82},
      {23, "IQ4_XS", 256, 136},
      {24, "I8", 1, 1},
      {25, "I16", 1, 2},
      {26, "I32", 1, 4},
      {27, "I64", 1, 8},
      {28, "F64", 1, 8},
      {29, "IQ1_M", 256, 56},
      {30, "BF16", 1, 2},
      {100, "ROCMFP4", 32, 18},
      {101, "ROCMFP4_FAST", 32, 17},
      {120, "G4_Q4", 32, 18},
      {121, "G4_Q4_FAST", 32, 17},
      // v0.8 (F3): ggml MXFP4-family layouts (qfn_ggml_stubs.h).
      {39, "MXFP4", 32, 17},
      {40, "NVFP4", 64, 18},
      {41, "Q1_0", 256, 36},
  };
  for (const auto& r : kRows) {
    bool known = false;
    const std::vector<std::uint64_t> dims = {r.be, 3};
    const std::uint64_t got = gqt::CensusTensorBytes(r.t, dims, &known);
    Expect(known && got == r.bb * 3,
           std::string("P3: ") + r.name + " bytes, got " + std::to_string(got));
    Expect(gqt::CensusTypeName(r.t) == r.name,
           std::string("P3: ") + r.name + " name");
  }
  bool known = false;
  (void)gqt::CensusTensorBytes(31, {32, 8}, &known);
  Expect(!known, "P3: Q8_0_ROCMFPX unknown layout");
  // v0.8 (F3): id 32 is not MXFP4 (no header in 1194fea defines it) -- it is
  // the generic T32 and keeps the offset-differencing fallback.
  (void)gqt::CensusTensorBytes(32, {32, 8}, &known);
  Expect(!known, "P3: T32 unknown layout");
  Expect(gqt::CensusTypeName((gqt::Type)32) == "T32", "P3: T32 name");
  (void)gqt::CensusTensorBytes(2, {33, 8}, &known);
  Expect(!known, "P3: unaligned ne0 falls back to diffing");
  std::cout << "Part 3 census helpers PASS\n";
}

// ---- Part 4: the binary on a tiny deterministic synth ----------------------
gqt::KvValue KvStr(const std::string& s) {
  gqt::KvValue v;
  v.type = gqt::ValType::kString;
  v.str = s;
  return v;
}
gqt::KvValue KvU32(std::uint32_t x) {
  gqt::KvValue v;
  v.type = gqt::ValType::kUint32;
  v.scalar_bytes.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 4);
  return v;
}

// Tiny BF16 source. ne0=256 for the head pair (Q6_K/Q5_K need %256==0 when a
// head flag or rule selects them: no shape fallback there, like the fork);
// ne0=32 everywhere else (ROC/Q8_0 need %32==0).
std::string MakeSynth(const std::filesystem::path& dir, const std::string& leaf,
                      bool with_output) {
  gqt::GgufFile src;
  src.alignment = 32;
  src.kv.emplace_back("general.architecture", KvStr("qwen35"));
  src.kv.emplace_back("qwen35.block_count", KvU32(2));
  auto add = [&](const std::string& name, std::vector<std::uint64_t> dims) {
    gqt::GgufTensor t;
    t.name = name;
    t.dims = std::move(dims);
    t.type = 30;  // BF16 source
    std::uint64_t nelem = 1;
    for (auto d : t.dims)
      nelem *= d;
    for (std::uint64_t k = 0; k < nelem; ++k) {
      // Deterministic sane values in [-1, 1): the k-quant encoders assert on
      // huge magnitudes, and garbage BF16 patterns trip them.
      float v = (float)((k * 2654435761u) % 2000) / 1000.0f - 1.0f + 0.0005f;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      std::uint16_t h = (std::uint16_t)(u >> 16);
      t.raw.push_back((std::uint8_t)(h & 0xFF));
      t.raw.push_back((std::uint8_t)(h >> 8));
    }
    t.size_bytes = t.raw.size();
    src.tensors.push_back(std::move(t));
  };
  add("blk.0.attn_q.weight", {32, 32});
  add("blk.0.attn_output.weight", {32, 32});
  add("blk.0.ssm_alpha.weight", {32, 8});
  add("blk.1.ssm_alpha.weight", {32, 8});
  add("blk.0.ffn_gate_inp.weight", {32, 8});  // router: 2-D, gate-blocked
  add("token_embd.weight", {256, 8});
  if (with_output)
    add("output.weight", {256, 8});
  add("output_norm.weight", {32});  // 1-D: never quantized
  std::vector<gqt::Type> types(src.tensors.size(), 30);
  std::vector<std::vector<std::uint8_t>> payloads;
  for (auto& t : src.tensors)
    payloads.push_back(t.raw);
  const std::string p = (dir / leaf).string();
  gqt::WriteGguf(p, src, types, payloads, /*ftype=*/0, {});
  return p;
}

std::map<std::string, gqt::Type> OutTypes(const std::string& path) {
  std::map<std::string, gqt::Type> m;
  gqt::GgufFile f = gqt::LoadGgufHeaderOnly(path);
  for (auto& t : f.tensors)
    m[t.name] = t.type;
  return m;
}

int Run(const std::string& cmd) {
  return std::system(cmd.c_str());
}

// Decoded exit code (-1 when killed by a signal): distinguishes exit 2 from
// an abort, which plain `system()` nonzero cannot.
int ExitCode(const std::string& cmd) {
  const int st = std::system(cmd.c_str());
  if (WIFEXITED(st))
    return WEXITSTATUS(st);
  return -1;
}

// A synth carrying a non-writable type (Q4_K), like the stock reference
// quants the formula censuses. Payload content is filler: the census reads
// the header only, but the size must be the true layout (144 B/block).
std::string MakeQ4KSynth(const std::filesystem::path& dir,
                         const std::string& leaf) {
  gqt::GgufFile src;
  src.alignment = 32;
  src.kv.emplace_back("general.architecture", KvStr("qwen35"));
  src.kv.emplace_back("qwen35.block_count", KvU32(2));
  gqt::GgufTensor t;
  t.name = "blk.0.attn_q.weight";
  t.dims = {256, 4};
  t.type = 30;
  src.tensors.push_back(t);
  std::vector<std::vector<std::uint8_t>> payloads(1);
  payloads[0].assign(4 * 144, 0);
  const std::string p = (dir / leaf).string();
  gqt::WriteGguf(p, src, {(gqt::Type)12}, payloads, /*ftype=*/0, {});
  return p;
}

void Part4(const std::string& superroc_quant,
           const std::filesystem::path& dir) {
  const std::string Q = "\"" + superroc_quant + "\"";
  const std::string src = MakeSynth(dir, "synth-bf16.gguf", true);
  const std::string log = (dir / "q.log").string();
  auto q = [&](const std::string& out, const std::string& opts) {
    return Run(Q + " " + opts + " \"" + src + "\" \"" + out +
               "\" STRIX 2 > \"" + log + "\" 2>&1");
  };

  // =exact hits output only; the recipe keeps attn_output at 101 on STRIX.
  {
    const std::string o = (dir / "exact.gguf").string();
    Expect(q(o, "--tensor-type =output.weight=Q6_K") == 0, "P4: exact run");
    auto t = OutTypes(o);
    Expect(t["output.weight"] == (gqt::Type)14, "P4: exact output Q6_K");
    Expect(t["blk.0.attn_output.weight"] == (gqt::Type)101,
           "P4: exact leaves attn_output at recipe");
  }
  // Plain substring keeps the legacy meaning: output.weight hits both.
  {
    const std::string o = (dir / "sub.gguf").string();
    Expect(q(o, "--tensor-type output.weight=Q8_0") == 0, "P4: substring run");
    auto t = OutTypes(o);
    Expect(t["output.weight"] == (gqt::Type)8, "P4: substring output Q8_0");
    Expect(t["blk.0.attn_output.weight"] == (gqt::Type)8,
           "P4: substring also hits attn_output");
  }
  // First matching rule wins (CLI order).
  {
    const std::string o = (dir / "first.gguf").string();
    Expect(
        q(o, "--tensor-type attn_q=Q8_0 --tensor-type attn=Q4_0 --dry-run") ==
            0,
        "P4: first-match run");
    // dry-run writes nothing; re-run for real to read types.
    Expect(q(o, "--tensor-type attn_q=Q8_0 --tensor-type attn=Q4_0") == 0,
           "P4: first-match run 2");
    auto t = OutTypes(o);
    Expect(t["blk.0.attn_q.weight"] == (gqt::Type)8, "P4: first rule wins");
    Expect(t["blk.0.attn_output.weight"] == (gqt::Type)2,
           "P4: second rule takes the rest");
  }
  // v0.6: --dry-run prints a real run's table exactly (sizes only, no
  // encode), and writes nothing.
  {
    const std::string o = (dir / "dryeq.gguf").string();
    const std::string dry_tab = (dir / "dryeq-dry.txt").string();
    const std::string real_tab = (dir / "dryeq-real.txt").string();
    const std::string opts =
        "--tensor-type attn_q=Q8_0 --tensor-type attn=Q4_0";
    Expect(Run(Q + " " + opts + " --dry-run \"" + src + "\" \"" + o +
               "\" STRIX 2 > \"" + dry_tab + "\" 2> \"" + log + "\"") == 0,
           "P4: dry-run table run");
    Expect(!std::filesystem::exists(o), "P4: dry-run writes nothing");
    Expect(Run(Q + " " + opts + " \"" + src + "\" \"" + o + "\" STRIX 2 > \"" +
               real_tab + "\" 2> \"" + log + "\"") == 0,
           "P4: dry-run real run");
    Expect(ReadFile(dry_tab) == ReadFile(real_tab),
           "P4: dry-run table == real-run table");
  }
  // v0.6 (SPEC2 DECISIONS 12:54): --dry-run is header-only (no tensor
  // reads, no hashing), so with --manifest it records sizes without sha256
  // and says so in the JSON.
  {
    const std::string o = (dir / "drym.gguf").string();
    const std::string dm = (dir / "drym.json").string();
    Expect(Run(Q + " --dry-run --manifest \"" + dm + "\" \"" + src + "\" \"" +
               o + "\" STRIX 2 > \"" + log + "\" 2>&1") == 0,
           "P4: dry-manifest run");
    Expect(!std::filesystem::exists(o), "P4: dry-manifest writes no GGUF");
    const std::string mj = ReadFile(dm);
    Expect(mj.find("\"sha256\": null") != std::string::npos,
           "P4: dry-manifest null sha");
    Expect(mj.find("\"hash_note\"") != std::string::npos,
           "P4: dry-manifest note");
    Expect(mj.find("\"hash_seconds\": 0.00") != std::string::npos,
           "P4: dry-manifest zero hash");
    Expect(mj.find("\"output\": null") != std::string::npos,
           "P4: dry-manifest null output");
    Expect(mj.find("\"bytes\": " + std::to_string(std::filesystem::file_size(
                                       src))) != std::string::npos,
           "P4: dry-manifest stat size");
  }
  // --protect applies BEFORE --tensor-type; globs; router unlock.
  {
    const std::string pf = (dir / "p.txt").string();
    WriteFile(pf,
              "=output.weight Q8_0  # beats the CLI rule below\n"
              "*ssm_alpha.weight Q8_0\n"
              "*ffn_gate_inp.weight Q8_0\n");
    const std::string o = (dir / "prot.gguf").string();
    Expect(
        q(o, "--protect \"" + pf + "\" --tensor-type =output.weight=Q6_K") == 0,
        "P4: protect run: " + ReadFile(log));
    auto t = OutTypes(o);
    Expect(t["output.weight"] == (gqt::Type)8, "P4: protect beats CLI");
    Expect(t["blk.0.ssm_alpha.weight"] == (gqt::Type)8 &&
               t["blk.1.ssm_alpha.weight"] == (gqt::Type)8,
           "P4: protect glob hits both ssm_alpha");
    Expect(t["blk.0.ffn_gate_inp.weight"] == (gqt::Type)8,
           "P4: protect unlocks the router");
    Expect(t["output_norm.weight"] == (gqt::Type)30, "P4: 1-D kept");
  }
  // A protect rule matching zero tensors is an error; --protect-allow-
  // unmatched relaxes it with a printed notice.
  {
    const std::string pf = (dir / "pz.txt").string();
    WriteFile(pf, "*no_such_family.weight Q8_0\n");
    const std::string o = (dir / "pz.gguf").string();
    Expect(q(o, "--protect \"" + pf + "\"") != 0,
           "P4: unmatched protect fails");
    Expect(ReadFile(log).find("matched 0 tensors") != std::string::npos,
           "P4: unmatched message");
    Expect(q(o, "--protect \"" + pf + "\" --protect-allow-unmatched") == 0,
           "P4: unmatched allowed");
    Expect(ReadFile(log).find("--protect-allow-unmatched") != std::string::npos,
           "P4: allowed notice");
  }
  // Fork head flags: exact names, recorded provenance, trace text.
  {
    const std::string o = (dir / "head.gguf").string();
    Expect(q(o,
             "--output-tensor-type Q6_K --token-embedding-type Q8_0 "
             "--tensor-type-trace --manifest \"" +
                 (dir / "m.json").string() + "\"") == 0,
           "P4: head run: " + ReadFile(log));
    auto t = OutTypes(o);
    Expect(t["output.weight"] == (gqt::Type)14, "P4: head output Q6_K");
    Expect(t["token_embd.weight"] == (gqt::Type)8, "P4: head embd Q8_0");
    Expect(t["blk.0.attn_output.weight"] == (gqt::Type)101,
           "P4: head flag misses attn_output");
    const std::string tab = ReadFile(log);
    Expect(tab.find("output.weight") != std::string::npos &&
               tab.find("via --output-tensor-type") != std::string::npos,
           "P4: trace shows the head flag");
    Expect(tab.find("via recipe") != std::string::npos,
           "P4: trace shows recipe");
    gqt::GgufFile f = gqt::LoadGgufHeaderOnly(o);
    bool found_out = false, found_tok = false;
    for (auto& [k, v] : f.kv) {
      if (k == "superroc.quant.output_tensor_type" && v.str == "Q6_K")
        found_out = true;
      if (k == "superroc.quant.token_embedding_type" && v.str == "Q8_0")
        found_tok = true;
    }
    Expect(found_out && found_tok, "P4: head provenance keys");
    // Manifest: inputs/output sha + bytes, rules, families, wall time.
    const std::string mj = ReadFile((dir / "m.json").string());
    std::uint64_t inb = 0;
    const std::string insha = gqt::Sha256File(src, &inb);
    std::uint64_t outb = 0;
    const std::string outsha = gqt::Sha256File(o, &outb);
    Expect(mj.find(insha) != std::string::npos, "P4: manifest input sha");
    Expect(mj.find(outsha) != std::string::npos, "P4: manifest output sha");
    Expect(mj.find("\"bytes\": " + std::to_string(inb)) != std::string::npos,
           "P4: manifest input bytes");
    Expect(mj.find("\"recipe\": \"STRIX\"") != std::string::npos,
           "P4: manifest recipe");
    Expect(mj.find("\"wall_seconds\"") != std::string::npos,
           "P4: manifest wall");
    Expect(mj.find("\"blk.N.ssm_alpha.weight\"") != std::string::npos,
           "P4: manifest families");
  }
  // Head flag beats rules (fork order); --leave-output-tensor gates the head
  // flag (fork-exact) while a CLI rule still bypasses the gate (superroc).
  {
    const std::string o = (dir / "h2.gguf").string();
    Expect(
        q(o, "--output-tensor-type Q5_K --tensor-type =output.weight=Q6_K") ==
            0,
        "P4: head-beats-rule run");
    Expect(OutTypes(o)["output.weight"] == (gqt::Type)13,
           "P4: head beats rule");
    const std::string o2 = (dir / "h3.gguf").string();
    Expect(q(o2, "--leave-output-tensor --output-tensor-type Q6_K") == 0,
           "P4: leave run");
    Expect(OutTypes(o2)["output.weight"] == (gqt::Type)30,
           "P4: leave gates the head flag");
    const std::string o3 = (dir / "h4.gguf").string();
    Expect(
        q(o3, "--leave-output-tensor --tensor-type =output.weight=Q6_K") == 0,
        "P4: leave+rule run");
    Expect(OutTypes(o3)["output.weight"] == (gqt::Type)14,
           "P4: CLI rule bypasses leave");
  }
  // Tied model (no output.weight): --output-tensor-type warns and is a no-op;
  // --token-embedding-type still applies.
  {
    const std::string tied = MakeSynth(dir, "tied-bf16.gguf", false);
    const std::string o = (dir / "tied.gguf").string();
    const std::string cmd = Q + " --output-tensor-type Q8_0 \"" + tied +
                            "\" \"" + o + "\" LEAN 2 > \"" + log + "\" 2>&1";
    Expect(Run(cmd) == 0, "P4: tied run");
    Expect(ReadFile(log).find("no-op") != std::string::npos,
           "P4: tied warning");
    auto t = OutTypes(o);
    Expect(t["token_embd.weight"] == (gqt::Type)13,
           "P4: tied embd keeps recipe Q5_K, got " +
               gqt::TypeName(t["token_embd.weight"]));
    const std::string cmd2 = Q +
                             " --output-tensor-type Q8_0 "
                             "--token-embedding-type Q8_0 \"" +
                             tied + "\" \"" + o + "\" LEAN 2 > \"" + log +
                             "\" 2>&1";
    Expect(Run(cmd2) == 0, "P4: tied run 2");
    Expect(OutTypes(o)["token_embd.weight"] == (gqt::Type)8,
           "P4: tied embd takes the token flag");
  }
  // Census: text fields and JSON keys (prot.gguf has the Q8_0 ssm pair,
  // head.gguf the Q6_K output).
  {
    const std::string clog = (dir / "c.log").string();
    Expect(Run(Q + " census \"" + (dir / "prot.gguf").string() + "\" > \"" +
               clog + "\" 2>&1") == 0,
           "P4: census run");
    const std::string c = ReadFile(clog);
    Expect(c.find("arch=qwen35") != std::string::npos, "P4: census arch");
    Expect(c.find("tensors=8") != std::string::npos, "P4: census count");
    Expect(c.find("block_count=2") != std::string::npos, "P4: census hyper");
    Expect(c.find("blk.N.ssm_alpha.weight") != std::string::npos &&
               c.find("Q8_0 x2") != std::string::npos,
           "P4: census family row");
    Expect(Run(Q + " census \"" + (dir / "head.gguf").string() + "\" > \"" +
               clog + "\" 2>&1") == 0,
           "P4: census run 2");
    const std::string c2 = ReadFile(clog);
    Expect(c2.find("output.weight") != std::string::npos &&
               c2.find("Q6_K x1") != std::string::npos,
           "P4: census head row");
    Expect(c2.find("superroc.quant.tool = superroc-quant 0.9.1") !=
               std::string::npos,
           "P4: census provenance");
    Expect(c2.find("BPW=") != std::string::npos, "P4: census BPW");
    Expect(Run(Q + " census \"" + (dir / "head.gguf").string() +
               "\" --json > \"" + clog + "\" 2>&1") == 0,
           "P4: census json run");
    const std::string j = ReadFile(clog);
    Expect(j.find("\"bpw\"") != std::string::npos, "P4: json bpw");
    Expect(j.find("\"types_2d\"") != std::string::npos, "P4: json types");
    Expect(j.find("\"provenance\"") != std::string::npos, "P4: json prov");
  }
  // RV-LQP1 F1: census of a file with a non-writable type (Q4_K) succeeds
  // and sizes it at 144 B/block; F5: a missing file exits 2, never aborts.
  {
    const std::string qs = MakeQ4KSynth(dir, "q4k.gguf");
    const std::string qlog = (dir / "q4k.log").string();
    // NOTE: run first, THEN read the log (C++ argument order is unspecified,
    // so ReadFile must not sit in the same Expect as the command itself).
    const int qrc =
        ExitCode(Q + " census \"" + qs + "\" > \"" + qlog + "\" 2>&1");
    Expect(qrc == 0, "P4: census Q4_K rc 0: " + ReadFile(qlog));
    const std::string c = ReadFile(qlog);
    Expect(c.find("Q4_K x1") != std::string::npos, "P4: census Q4_K row");
    Expect(c.find("bytes=576") != std::string::npos,
           "P4: census Q4_K 144 B/block");
    const std::string mlog = (dir / "q4k-miss.log").string();
    Expect(ExitCode(Q + " census \"" + (dir / "nope.gguf").string() +
                    "\" > \"" + mlog + "\" 2>&1") == 2,
           "P4: census missing rc 2");
    Expect(ReadFile(mlog).find("cannot open") != std::string::npos,
           "P4: census missing message");
  }
  // v0.8 (F4 note): Q5_1 is nameable end to end (the encoder and the recipe
  // fallback already knew it; only the CLI table did not).
  {
    const std::string o = (dir / "q51.gguf").string();
    // NOTE: run first, THEN read the log (C++ argument order is unspecified).
    const int q51rc = q(o, "--tensor-type =blk.0.attn_q.weight=Q5_1");
    Expect(q51rc == 0, "P4: Q5_1 run: " + ReadFile(log));
    Expect(OutTypes(o)["blk.0.attn_q.weight"] == (gqt::Type)7,
           "P4: Q5_1 named type applied");
  }
  // v0.8 (F3): the census text says when a size came from offset
  // differencing (unknown-layout id 32, now the generic T32); a fully-known
  // file carries no such line.
  {
    gqt::GgufFile src;
    src.alignment = 32;
    src.kv.emplace_back("general.architecture", KvStr("qwen35"));
    src.kv.emplace_back("qwen35.block_count", KvU32(2));
    gqt::GgufTensor t;
    t.name = "blk.0.mystery.weight";
    t.dims = {32, 8};
    t.type = 30;
    src.tensors.push_back(t);
    std::vector<std::vector<std::uint8_t>> payloads(1);
    payloads[0].assign(64, 0);
    const std::string tp = (dir / "t32.gguf").string();
    gqt::WriteGguf(tp, src, {(gqt::Type)32}, payloads, /*ftype=*/0, {});
    const std::string tlog = (dir / "t32.log").string();
    // NOTE: run first, THEN read the log (C++ argument order is unspecified;
    // gcc evaluates ReadFile first and fails on the missing file).
    const int t32rc =
        ExitCode(Q + " census \"" + tp + "\" > \"" + tlog + "\" 2>&1");
    Expect(t32rc == 0, "P4: census T32 rc 0: " + ReadFile(tlog));
    const std::string c = ReadFile(tlog);
    Expect(c.find("T32 x1") != std::string::npos, "P4: census T32 row");
    Expect(c.find("sizes from offset differencing (alignment padding "
                  "included): blk.0.mystery.weight") != std::string::npos,
           "P4: census offset-diff line");
    const std::string hlog = (dir / "t32-head.log").string();
    Expect(Run(Q + " census \"" + (dir / "head.gguf").string() + "\" > \"" +
               hlog + "\" 2>&1") == 0,
           "P4: census head rerun");
    Expect(ReadFile(hlog).find("offset differencing") == std::string::npos,
           "P4: no offset-diff line for fully-known files");
  }
  std::cout << "Part 4 binary end-to-end PASS\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: superroc_quant_rules_test <path-to-superroc-quant> "
                 "[workdir]\n";
    return 2;
  }
  const std::string superroc_quant = argv[1];
  std::filesystem::path dir;
  if (argc > 2) {
    dir = argv[2];
  } else {
    dir = std::filesystem::temp_directory_path() /
          ("superroc-quant-rules-test-" + std::to_string((long)getpid()));
  }
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir, ec);
  Expect(!ec, "cannot create workdir " + dir.string());

  Part1();
  Part2(dir);
  Part3();
  Part4(superroc_quant, dir);

  std::filesystem::remove_all(dir, ec);
  std::cout << "superroc_quant_rules_test PASS\n";
  return 0;
}
