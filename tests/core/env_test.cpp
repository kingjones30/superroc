// env_test: basic reads of the engine's environment switches.
#include "src/core/env.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

int main() {
  const char* kProbe = "SUPERROC_ENV_TEST_PROBE";
  ::unsetenv(kProbe);

  ::setenv(kProbe, "on", 1);
  const char* direct = ::superroc::core::GetEnv(kProbe);
  if (direct == nullptr || std::strcmp(direct, "on") != 0) {
    std::printf("FAIL set returns on\n");
    return 1;
  }

  ::unsetenv(kProbe);
  if (::superroc::core::GetEnv(kProbe) != nullptr) {
    std::printf("FAIL unset returns null\n");
    return 1;
  }

  if (::superroc::core::GetEnv("") != nullptr) {
    std::printf("FAIL empty returns null\n");
    return 1;
  }

  ::setenv(kProbe, "on", 1);
  const char* via_view = ::superroc::core::GetEnv(std::string_view(kProbe));
  if (via_view == nullptr || std::strcmp(via_view, "on") != 0) {
    std::printf("FAIL string_view returns on\n");
    return 1;
  }
  ::unsetenv(kProbe);
  return 0;
}
