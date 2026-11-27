#pragma once

#include <filesystem>
#include <string_view>
#include <system_error>

namespace superroc::cli {

// Locates a data file that the install step places under <prefix>/share.
// A build tree uses its source copy. An installed or unpacked binary uses the
// copy beside it (<executable>/../share/<share_relative>), so a release package
// works wherever it is unpacked. The configured install path is the fallback.
inline std::filesystem::path FindDataFile(
    const std::filesystem::path& source_path,
    const std::filesystem::path& installed_path,
    std::string_view share_relative) {
  std::error_code error;
  if (std::filesystem::is_regular_file(source_path, error)) {
    return source_path;
  }
  const std::filesystem::path executable =
      std::filesystem::read_symlink("/proc/self/exe", error);
  if (!error) {
    const std::filesystem::path beside =
        executable.parent_path().parent_path() / "share" /
        std::filesystem::path(share_relative);
    if (std::filesystem::is_regular_file(beside, error)) {
      return beside;
    }
  }
  return installed_path;
}

}  // namespace superroc::cli
