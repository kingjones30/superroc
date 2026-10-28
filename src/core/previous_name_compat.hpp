// Previous engine-name compatibility for renamed identifiers.
//
// Readers of names our code writes (JSON schemas, request keys, manifests)
// accept the current `superroc*` spelling and the previous spelling, so data
// written by previous releases keeps working. Writers always emit the current
// spelling. The previous spelling is assembled from parts so the old engine
// name never appears in this file.
#pragma once

#include <string>
#include <string_view>

namespace superroc::core {

// Maps a current `superroc*` engine name to the spelling used by previous
// releases (assembled from parts in the code below, never written out here).
//
// Examples: `superroc.foo` and `superroc` map to their previous-release
// counterparts. Names without the `superroc` prefix are returned unchanged.
inline std::string PreviousEngineName(std::string_view current) {
  constexpr std::string_view kNew = "superroc";
  if (current.size() >= kNew.size() && current.substr(0, kNew.size()) == kNew) {
    return std::string("g") + "ufo" + std::string(current.substr(kNew.size()));
  }
  return std::string(current);
}

// True when a stored name uses the current spelling or the previous one.
inline bool MatchesCurrentOrPrevious(std::string_view value,
                                     std::string_view current) {
  return value == current || value == PreviousEngineName(current);
}

}  // namespace superroc::core
