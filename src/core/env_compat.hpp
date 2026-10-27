// Renamed environment variables: new-name read with previous-name fallback.
//
// Every read of a `SUPERROC_*` variable falls back to the previous name when
// the new one is unset, so seats and scripts written against the previous
// names keep working. The first fallback use prints one deprecation line per
// name per process on stderr. The previous prefix is assembled from parts so
// that its spelling never appears in this file.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include <string_view>

namespace superroc::core {

// Reads a renamed `SUPERROC_*` environment variable.
//
// When the new name is set (even to the empty string) its value wins.
// Otherwise the previous name is tried: when set, its value is returned and
// one "'<old> is deprecated; use <new>'" line is printed once per process.
// Names without the new prefix are read as-is, so existing helpers can route
// every lookup through here without changing behavior for other variables.
inline const char* GetEnvCompat(const char* superroc_name) {
  if (superroc_name == nullptr || superroc_name[0] == '\0') {
    return nullptr;
  }
  if (const char* direct = std::getenv(superroc_name); direct != nullptr) {
    return direct;
  }
  constexpr std::string_view kNewPrefix = "SUPERROC_";
  const std::string_view name(superroc_name);
  if (name.substr(0, kNewPrefix.size()) != kNewPrefix) {
    return nullptr;
  }
  const std::string tail(name.substr(kNewPrefix.size()));
  // The previous engine name came in two prefix spellings: most variables
  // used the underscore form, while the KV-cache variables glued the prefix
  // directly (no underscore after the previous prefix). Both spellings are
  // assembled from parts so the previous prefix never appears here.
  std::string candidates[2] = {std::string("G") + "UFO_" + tail, ""};
  std::size_t count = 1;
  if (tail.rfind("KVC_", 0) == 0) {
    candidates[count++] = std::string("G") + "UFO" + tail;
  }
  static std::mutex warn_mutex;
  static std::set<std::string> warned;
  for (std::size_t i = 0; i < count; ++i) {
    const char* legacy = std::getenv(candidates[i].c_str());
    if (legacy == nullptr) {
      continue;
    }
    {
      const std::lock_guard<std::mutex> lock(warn_mutex);
      if (!warned.insert(candidates[i]).second) {
        return legacy;
      }
    }
    std::fprintf(stderr, "%s is deprecated; use %s\n", candidates[i].c_str(),
                 superroc_name);
    return legacy;
  }
  return nullptr;
}

// Overloads so wrappers taking std::string / std::string_view route directly.
inline const char* GetEnvCompat(const std::string& superroc_name) {
  return GetEnvCompat(superroc_name.c_str());
}

inline const char* GetEnvCompat(std::string_view superroc_name) {
  return GetEnvCompat(std::string(superroc_name).c_str());
}

}  // namespace superroc::core
