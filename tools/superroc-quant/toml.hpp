// Minimal TOML reader for superroc-quant recipe files. Handles the subset our
// recipes use: `key = value` pairs, `[section]` headers, integer/string values
// and `#` comments. Real production TOML would use a library; this keeps
// superroc-quant dependency-free (matches SPEC: "or link nothing external").

#ifndef SUPERROC_QUANT_TOML_HPP_
#define SUPERROC_QUANT_TOML_HPP_

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace superroc::quant_tool {

using TomlTable = std::map<std::string, std::string>;  // flattened "section.key" -> value

inline TomlTable ParseToml(const std::string& path) {
  TomlTable out;
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open toml " + path);
  std::string section;
  std::string line;
  while (std::getline(f, line)) {
    // strip comment
    auto h = line.find('#');
    if (h != std::string::npos) line = line.substr(0, h);
    if (line.empty()) continue;
    // trim
    auto b = line.find_first_not_of(" \t\r");
    if (b == std::string::npos) continue;
    line = line.substr(b);
    if (line.empty() || line[0] == '\n') continue;
    if (line[0] == '[') {
      // [section]
      auto close = line.find(']');
      if (close != std::string::npos) section = line.substr(1, close - 1);
      continue;
    }
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = line.substr(0, eq);
    auto kb = key.find_first_not_of(" \t"), ke = key.find_last_not_of(" \t");
    key = key.substr(kb, ke - kb + 1);
    std::string val = line.substr(eq + 1);
    auto vb = val.find_first_not_of(" \t\r"), ve = val.find_last_not_of(" \t\r\n");
    val = val.substr(vb, ve - vb + 1);
    // strip surrounding quotes for strings
    if (val.size() >= 2 && ((val.front() == '"' && val.back() == '"') ||
                            (val.front() == '\'' && val.back() == '\'')))
      val = val.substr(1, val.size() - 2);
    std::string fq = section.empty() ? key : section + "." + key;
    out[fq] = val;
  }
  return out;
}

}  // namespace superroc::quant_tool

#endif  // SUPERROC_QUANT_TOML_HPP_
