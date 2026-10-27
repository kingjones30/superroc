// Tensor-type rules for superroc-quant v0.5 (LQP SPEC1 items 1-2).
//
// A rule is `RULE -> TYPE`, from `--tensor-type RULE=TYPE` on the command line
// or a `RULE TYPE` line in a `--protect FILE`. RULE syntax:
//
//   `=name`      exact full-name match (`=output.weight` hits ONLY output.weight)
//   `^prefix`    prefix match
//   `suffix$`    suffix match
//   `a*b*c`      glob full-name match (`*` = any run of chars, may be empty;
//                everything else is literal). E.g. `blk.*.ssm_alpha.weight`,
//                `*ffn_gate_inp.weight`.
//   `plain`      substring match (the pre-v0.5 meaning; kept for compat)
//
// A leading `=` disables globbing (the rest is literal). A leading `^` and/or
// trailing `$` may wrap a glob core; globs already match the full name so the
// anchors are redundant there but accepted. Precedence across rules is purely
// positional: --protect rules (file order) first, then --tensor-type rules
// (command-line order); the FIRST matching rule wins for each tensor.
//
// This header is dependency-free (gguf.hpp's Type only) so the CTest suite can
// exercise it without the tool binary.

#ifndef SUPERROC_QUANT_RULES_HPP_
#define SUPERROC_QUANT_RULES_HPP_

#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf.hpp"

namespace superroc::quant_tool {

enum class RuleKind {
  kSubstring,  // plain text: name.find(core) (pre-v0.5 meaning)
  kExact,      // =name
  kPrefix,     // ^prefix
  kSuffix,     // suffix$
  kGlob,       // full-name glob, '*' = any run
};

struct Rule {
  std::string text;    // as written (for --tensor-type-trace and --manifest)
  std::string source;  // "cli" or "protect:<file>:<line>"
  RuleKind kind = RuleKind::kSubstring;
  std::string core;    // pattern after stripping =, ^, $
  Type type = 0;

  [[nodiscard]] bool Matches(const std::string& name) const {
    switch (kind) {
      case RuleKind::kExact:
        return name == core;
      case RuleKind::kPrefix:
        return name.size() >= core.size() &&
               name.compare(0, core.size(), core) == 0;
      case RuleKind::kSuffix:
        return name.size() >= core.size() &&
               name.compare(name.size() - core.size(), core.size(), core) == 0;
      case RuleKind::kGlob:
        return GlobMatch(core, name);
      case RuleKind::kSubstring:
      default:
        return name.find(core) != std::string::npos;
    }
  }

  // Full-match glob where only '*' is special (any run, possibly empty).
  static bool GlobMatch(const std::string& pat, const std::string& s) {
    std::size_t px = 0, sx = 0, star = std::string::npos, ss = 0;
    while (sx < s.size()) {
      if (px < pat.size() && (pat[px] == s[sx])) {
        ++px;
        ++sx;
      } else if (px < pat.size() && pat[px] == '*') {
        star = px++;
        ss = sx;
      } else if (star != std::string::npos) {
        px = star + 1;
        sx = ++ss;
      } else {
        return false;
      }
    }
    while (px < pat.size() && pat[px] == '*') ++px;
    return px == pat.size();
  }
};

// Upper and (fork-style) lower-case names both accepted, e.g. the fork's
// `--tensor-type ...=q8_0` and our `=Q8_0`. Same table TypeNames() in
// superroc_quant_main.cpp used before v0.5 (moved here so tests share it).
inline const std::map<std::string, Type>& TypeNameTable() {
  static const std::map<std::string, Type> m = {
      {"F32", (Type)0}, {"f32", (Type)0}, {"F16", (Type)1}, {"f16", (Type)1},
      {"Q4_0", (Type)2}, {"q4_0", (Type)2}, {"Q8_0", (Type)8}, {"q8_0", (Type)8},
      // v0.8 (RV-LQP2 F4 note): Q5_1 is nameable. The encoder, BlockBytes and
      // the recipe's Q5_K shape fallback already knew it; only the CLI table
      // did not, so no new byte path is introduced by naming it.
      {"Q5_1", (Type)7}, {"q5_1", (Type)7},
      {"Q5_K", (Type)13}, {"q5_k", (Type)13}, {"Q6_K", (Type)14}, {"q6_k", (Type)14},
      {"BF16", (Type)30}, {"bf16", (Type)30},
      {"Q4_0_ROCMFP4", (Type)100}, {"Q4_0_ROCMFP4_FAST", (Type)101},
      {"G4_Q4", (Type)120}, {"G4_Q4_FAST", (Type)121},
  };
  return m;
}

// Parse one RULE spec (the left side of `--tensor-type RULE=TYPE`, or the
// first column of a protect line) into a Rule. `source` is recorded verbatim.
inline Rule ParseRule(const std::string& spec, Type type, const std::string& source) {
  Rule r;
  r.text = spec;
  r.source = source;
  r.type = type;
  std::string s = spec;
  if (!s.empty() && s[0] == '=') {
    r.kind = RuleKind::kExact;
    r.core = s.substr(1);
    return r;
  }
  bool anchored_pre = false, anchored_suf = false;
  if (!s.empty() && s[0] == '^') {
    anchored_pre = true;
    s = s.substr(1);
  }
  if (!s.empty() && s.back() == '$') {
    anchored_suf = true;
    s.pop_back();
  }
  r.core = s;
  if (s.find('*') != std::string::npos) {
    r.kind = RuleKind::kGlob;  // full-name match; ^/$ redundant but accepted
  } else if (anchored_pre) {
    r.kind = RuleKind::kPrefix;
  } else if (anchored_suf) {
    r.kind = RuleKind::kSuffix;
  } else {
    r.kind = RuleKind::kSubstring;
  }
  return r;
}

[[nodiscard]] inline std::string RuleKindName(RuleKind k) {
  switch (k) {
    case RuleKind::kExact: return "exact";
    case RuleKind::kPrefix: return "prefix";
    case RuleKind::kSuffix: return "suffix";
    case RuleKind::kGlob: return "glob";
    case RuleKind::kSubstring: return "substring";
  }
  return "?";
}

// Parse a --protect FILE: `RULE TYPE  # why` per line. Blank lines and `#`
// comments are skipped. Throws std::runtime_error naming file:line on an
// unknown type or a malformed line. Returned rules carry
// source "protect:<file>:<line>" in file order.
inline std::vector<Rule> ParseProtectFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open --protect file " + path);
  std::vector<Rule> out;
  std::string line;
  int lineno = 0;
  auto trim = [](std::string s) {
    std::size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return std::string();
    std::size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
  };
  while (std::getline(f, line)) {
    ++lineno;
    auto h = line.find('#');
    if (h != std::string::npos) line = line.substr(0, h);
    line = trim(line);
    if (line.empty()) continue;
    std::string::size_type sp = line.find_first_of(" \t");
    if (sp == std::string::npos)
      throw std::runtime_error("--protect " + path + ":" +
                               std::to_string(lineno) + ": want `RULE TYPE`, got `" +
                               line + "`");
    std::string rule = trim(line.substr(0, sp));
    std::string type = trim(line.substr(sp));
    // The TYPE column is one token; anything after it is a malformed line
    // (a `#` comment would already have been stripped above).
    if (type.find_first_of(" \t") != std::string::npos)
      throw std::runtime_error("--protect " + path + ":" +
                               std::to_string(lineno) + ": want `RULE TYPE`, got `" +
                               line + "`");
    auto it = TypeNameTable().find(type);
    if (it == TypeNameTable().end())
      throw std::runtime_error("--protect " + path + ":" +
                               std::to_string(lineno) + ": unknown type `" + type +
                               "`");
    out.push_back(ParseRule(rule, it->second,
                            "protect:" + path + ":" + std::to_string(lineno)));
  }
  return out;
}

}  // namespace superroc::quant_tool

#endif  // SUPERROC_QUANT_RULES_HPP_
