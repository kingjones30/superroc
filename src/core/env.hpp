// Reads the engine's environment switches.
#pragma once

#include <cstdlib>
#include <string>
#include <string_view>

namespace superroc::core {

// The variable's value, or nullptr when it is unset or the name is empty.
inline const char* GetEnv(const char* name) {
  if (name == nullptr || name[0] == '\0') {
    return nullptr;
  }
  return std::getenv(name);
}

inline const char* GetEnv(const std::string& name) {
  return GetEnv(name.c_str());
}

inline const char* GetEnv(std::string_view name) {
  return GetEnv(std::string(name).c_str());
}

}  // namespace superroc::core
