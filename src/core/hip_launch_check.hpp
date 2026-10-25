#ifndef GUFO_CORE_HIP_LAUNCH_CHECK_HPP_
#define GUFO_CORE_HIP_LAUNCH_CHECK_HPP_
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>
namespace gufo::hip_check {
// audit F04: a launch whose configuration is invalid never enqueues;
// hipGetLastError reports it here instead of later stale data.
inline void ThrowIfLaunchFailed(const char* what, const char* file, int line) {
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) {
    throw std::runtime_error(std::string("HIP launch failed: ") + what + " (" +
                             hipGetErrorString(e) + ") at " + file + ":" +
                             std::to_string(line));
  }
}
[[nodiscard]] inline bool LaunchFailed() noexcept {
  return hipGetLastError() != hipSuccess;
}
}  // namespace gufo::hip_check
#define GUFO_CHECK_LAUNCH(what) \
  ::gufo::hip_check::ThrowIfLaunchFailed((what), __FILE__, __LINE__)
#endif
#endif
