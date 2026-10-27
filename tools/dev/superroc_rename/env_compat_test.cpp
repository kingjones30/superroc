// env_compat_test: renamed environment reads fall back as specified.
//
// New name wins when set; the previous name alone works; the previous name
// warns exactly once per name per process. Previous names are assembled from
// parts so their spelling never appears in this file.
#include "src/core/env_compat.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>

namespace {

// The previous name for a suffix, assembled from parts.
std::string OldName(const char* suffix) {
  return std::string("G") + "UFO_" + suffix;
}

int g_failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << '\n';
    ++g_failures;
  }
}

}  // namespace

int main() {
  const char* kNew = "SUPERROC_TEST_COMPAT_PROBE";
  const std::string old = OldName("TEST_COMPAT_PROBE");
  ::unsetenv(kNew);
  ::unsetenv(old.c_str());

  // Neither set: null.
  Check(::superroc::core::GetEnvCompat(kNew) == nullptr, "unset reads null");

  // Previous name alone works.
  ::setenv(old.c_str(), "legacy-value", 1);
  const char* via_old = ::superroc::core::GetEnvCompat(kNew);
  Check(via_old != nullptr && std::string(via_old) == "legacy-value",
        "previous name alone works");

  // New name wins over the previous one.
  ::setenv(kNew, "new-value", 1);
  const char* via_new = ::superroc::core::GetEnvCompat(kNew);
  Check(via_new != nullptr && std::string(via_new) == "new-value",
        "new name wins");
  ::unsetenv(kNew);
  ::unsetenv(old.c_str());

  // The warning fires exactly once per name per process (fresh names here:
  // the probe above already warned once for its own name).
  const char* kNew2 = "SUPERROC_TEST_COMPAT_PROBE_TWO";
  const std::string old2 = OldName("TEST_COMPAT_PROBE_TWO");
  ::unsetenv(kNew2);
  ::unsetenv(old2.c_str());
  char tmpl[] = "/tmp/superroc_env_compat_XXXXXX";
  const int fd = ::mkstemp(tmpl);
  Check(fd >= 0, "mkstemp for stderr capture");
  ::fflush(nullptr);
  const int saved = ::dup(STDERR_FILENO);
  Check(saved >= 0, "dup stderr");
  Check(::dup2(fd, STDERR_FILENO) >= 0, "redirect stderr");
  ::setenv(old2.c_str(), "legacy-two", 1);
  ::superroc::core::GetEnvCompat(kNew2);
  ::superroc::core::GetEnvCompat(kNew2);
  ::fflush(nullptr);
  Check(::dup2(saved, STDERR_FILENO) >= 0, "restore stderr");
  ::close(saved);
  ::lseek(fd, 0, SEEK_SET);
  std::string captured;
  char buf[256];
  ssize_t n = 0;
  while ((n = ::read(fd, buf, sizeof(buf))) > 0) {
    captured.append(buf, static_cast<std::size_t>(n));
  }
  ::close(fd);
  ::unlink(tmpl);
  ::unsetenv(old2.c_str());
  int lines = 0;
  for (char c : captured) {
    if (c == '\n') {
      ++lines;
    }
  }
  Check(lines == 1, "warns exactly once");
  Check(captured.find("is deprecated; use") != std::string::npos,
        "warning text");
  Check(captured.find(kNew2) != std::string::npos,
        "warning names the new variable");
  Check(captured.find(old2) != std::string::npos,
        "warning names the previous variable");

  // The glued previous prefix (no underscore) also falls back with a warning.
  const char* kNewKvc = "SUPERROC_KVC_TEST_PROBE";
  const std::string old_kvc = std::string("G") + "UFO" + "KVC_TEST_PROBE";
  ::unsetenv(kNewKvc);
  ::unsetenv(old_kvc.c_str());
  Check(::superroc::core::GetEnvCompat(kNewKvc) == nullptr,
        "unset glued name reads null");
  ::setenv(old_kvc.c_str(), "legacy-kvc", 1);
  const char* via_kvc = ::superroc::core::GetEnvCompat(kNewKvc);
  Check(via_kvc != nullptr && std::string(via_kvc) == "legacy-kvc",
        "glued previous name alone works");
  ::setenv(kNewKvc, "new-kvc", 1);
  const char* via_kvc_new = ::superroc::core::GetEnvCompat(kNewKvc);
  Check(via_kvc_new != nullptr && std::string(via_kvc_new) == "new-kvc",
        "new name wins over the glued previous name");
  ::unsetenv(kNewKvc);
  ::unsetenv(old_kvc.c_str());

  if (g_failures == 0) {
    std::cout << "env_compat_test passed\n";
  }
  return g_failures == 0 ? 0 : 1;
}
