#ifndef SUPERROC_SERVER_ENVIRONMENT_HPP_
#define SUPERROC_SERVER_ENVIRONMENT_HPP_

#include <cstdint>
#include <string>
#include <string_view>

namespace superroc::server {

// One /sys/class/kfd/kfd/topology/nodes/<id>/properties file, parsed.
struct KfdNodeProps {
  std::uint64_t simd_count{0};
  std::uint64_t cwsr_size{0};
  std::uint64_t ctl_stack_size{0};
  std::uint64_t capability{0};
};

/// A compute (GPU) KFD node carries shader engines; the CPU node has none.
[[nodiscard]] inline bool IsKfdGpuNode(const KfdNodeProps& props) noexcept {
  return props.simd_count > 0;
}

/// Lemonade Halogen's CWSR rule (check_kfd_node): capability bit 0x08000000
/// set and cwsr_size == ctl_stack_size + (simd_count/2)*479232. Applies to
/// GPU nodes only; the caller skips CPU nodes with IsKfdGpuNode.
[[nodiscard]] inline bool CheckKfdCwsrOk(const KfdNodeProps& props) noexcept {
  constexpr std::uint64_t kCapabilityCwsr = 0x08000000ULL;
  constexpr std::uint64_t kCwsrPerSimdPair = 479232ULL;
  if ((props.capability & kCapabilityCwsr) == 0) {
    return false;
  }
  return props.cwsr_size ==
         props.ctl_stack_size + (props.simd_count / 2) * kCwsrPerSimdPair;
}

namespace detail {
[[nodiscard]] inline std::uint64_t ParseKfdU64(
    std::string_view value) noexcept {
  std::uint64_t out = 0;
  for (const char c : value) {
    if (c < '0' || c > '9') {
      return 0;
    }
    out = out * 10 + static_cast<std::uint64_t>(c - '0');
  }
  return out;
}
}  // namespace detail

/// Parses a KFD topology properties file (`KEY VALUE` per line; unknown keys
/// and malformed lines are ignored).
[[nodiscard]] inline KfdNodeProps ParseKfdNodeProperties(
    std::string_view contents) {
  KfdNodeProps props;
  std::size_t pos = 0;
  while (pos < contents.size()) {
    std::size_t end = contents.find('\n', pos);
    if (end == std::string_view::npos) {
      end = contents.size();
    }
    const std::string_view line = contents.substr(pos, end - pos);
    pos = end + 1;
    const std::size_t space = line.find(' ');
    if (space == std::string_view::npos) {
      continue;
    }
    const std::string_view key = line.substr(0, space);
    const std::string_view value = line.substr(space + 1);
    if (key == "simd_count") {
      props.simd_count = detail::ParseKfdU64(value);
    } else if (key == "cwsr_size") {
      props.cwsr_size = detail::ParseKfdU64(value);
    } else if (key == "ctl_stack_size") {
      props.ctl_stack_size = detail::ParseKfdU64(value);
    } else if (key == "capability") {
      props.capability = detail::ParseKfdU64(value);
    }
  }
  return props;
}

// The GET /props "environment" object, collected once per server process.
struct ServerEnvironment {
  std::string gpu_arch = "unknown";
  std::string rocm_version = "unknown";
  std::string kernel = "unknown";
  bool cwsr_ok{false};
  bool hip_launch_blocking{false};
};

/// Reads the live box (sysfs KFD topology, uname, HIP runtime, environment).
/// Never throws: every unreadable source degrades to "unknown"/false.
ServerEnvironment CollectServerEnvironment() noexcept;

}  // namespace superroc::server

#endif  // SUPERROC_SERVER_ENVIRONMENT_HPP_
