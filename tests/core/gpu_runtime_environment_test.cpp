// gpu_runtime_environment_test: the runtime defaults are set once and never
// replace a value that is already in the environment.
#include "src/core/gpu_runtime_environment.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int failures = 0;

void Expect(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL %s\n", what);
    ++failures;
  }
}

bool Is(const char* key, const char* value) {
  const char* got = std::getenv(key);
  return got != nullptr && std::strcmp(got, value) == 0;
}

}  // namespace

int main() {
  using namespace superroc::core;
  ::unsetenv(kRuntimeHostBuffersKey);
  ::unsetenv(kRuntimePinnedCopyKey);
  ApplyGpuRuntimeDefaults();
  Expect(Is(kRuntimeHostBuffersKey, "0"), "host buffers default");
  Expect(Is(kRuntimePinnedCopyKey, "1048576"), "pinned copy default");

  ::setenv(kRuntimeHostBuffersKey, "1", 1);
  ::setenv(kRuntimePinnedCopyKey, "128", 1);
  ApplyGpuRuntimeDefaults();
  Expect(Is(kRuntimeHostBuffersKey, "1"), "a set host buffers value is kept");
  Expect(Is(kRuntimePinnedCopyKey, "128"), "a set pinned copy value is kept");

  ::setenv(kRuntimePinnedCopyKey, "", 1);
  ApplyGpuRuntimeDefaults();
  Expect(Is(kRuntimePinnedCopyKey, ""), "an empty value counts as set");
  if (failures != 0) {
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
