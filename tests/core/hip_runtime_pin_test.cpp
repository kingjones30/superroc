// hip_runtime_pin_test: which mappings count as registered by the GPU runtime,
// and that such a mapping is pinned once (no GPU needed: the mark the runtime
// leaves is MADV_DONTFORK, which the test sets itself).
#include <sys/mman.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/hip/host_allocations.hpp"

namespace {

using superroc::hip::HostMapping;

int failures = 0;

void Expect(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL %s\n", what);
    ++failures;
  }
}

// VmPin of this process, in bytes.
std::size_t PinnedBytes() {
  std::FILE* file = std::fopen("/proc/self/status", "re");
  if (file == nullptr) {
    return 0;
  }
  char line[256];
  std::size_t kilobytes = 0;
  while (std::fgets(line, sizeof(line), file) != nullptr) {
    if (std::sscanf(line, "VmPin: %zu kB", &kilobytes) == 1) {
      break;
    }
  }
  std::fclose(file);
  return kilobytes * 1024;
}

void TestSelection() {
  constexpr std::string_view kSmaps =
      "7ad19ea00000-7ad19fc56000 rw-p 00000000 00:00 0 \n"
      "Size:              18776 kB\n"
      "Rss:               18776 kB\n"
      "VmFlags: rd wr mr mw me dc ac sd \n"
      // The engine's own registered backing: opted out of huge pages.
      "7ad100000000-7ad188000000 rw-p 00000000 00:00 0 \n"
      "Size:            2228224 kB\n"
      "VmFlags: rd wr mr mw me dc ac sd nh \n"
      // Driver memory mapped from the render node.
      "7ad2b0000000-7ad2b0400000 rw-s 123000 00:06 512   /dev/dri/renderD128\n"
      "Size:               4096 kB\n"
      "VmFlags: rd wr sh mr mw me ms pf io dc de dd sd \n"
      // Ordinary heap and a thread stack.
      "55d0c1000000-55d0c1800000 rw-p 00000000 00:00 0   [heap]\n"
      "VmFlags: rd wr mr mw me ac sd \n"
      "7f0000401000-7f0000c01000 rw-p 00000000 00:00 0 \n"
      "VmFlags: rd wr mr mw me ac sd \n"
      // Executable runtime code segment.
      "7ad2c0000000-7ad2c0200000 r-xp 00000000 00:00 0 \n"
      "VmFlags: rd ex mr mw me dc ac sd \n"
      // Shared anonymous memory is not process-private.
      "7ad2d0000000-7ad2d0200000 rw-s 00000000 00:01 9   /dev/zero (deleted)\n"
      "VmFlags: rd wr sh mr mw me ms dc sd \n"
      // Larger than anything the runtime registers by itself.
      "7b0000000000-7b0080000000 rw-p 00000000 00:00 0 \n"
      "VmFlags: rd wr mr mw me dc ac sd \n"
      "7ad2e0000000-7ad2e0001000 rw-p 00000000 00:00 0 \n"
      "VmFlags: rd wr mr mw me ac dc";
  const std::vector<HostMapping> expected = {
      {0x7ad19ea00000, 0x7ad19fc56000},
      {0x7ad2c0000000, 0x7ad2c0200000},
      {0x7ad2e0000000, 0x7ad2e0001000},
  };
  Expect(superroc::hip::RuntimeRegisteredMappings(kSmaps) == expected,
         "anonymous private do-not-fork mappings without the huge-page "
         "opt-out");
  Expect(superroc::hip::RuntimeRegisteredMappings("").empty(), "empty text");
  // An address that does not fit, and a header cut off before its inode.
  Expect(superroc::hip::RuntimeRegisteredMappings(
             "17ad19ea000000000-7ad19fc56000 rw-p 00000000 00:00 0 \n"
             "VmFlags: rd wr mr mw me dc ac sd \n")
             .empty(),
         "an address wider than the type is not a mapping");
  Expect(superroc::hip::RuntimeRegisteredMappings(
             "7ad19ea00000-7ad19fc56000 rw-p\n"
             "VmFlags: rd wr mr mw me dc ac sd \n")
             .empty(),
         "a cut-off header is not taken for an anonymous mapping");
  Expect(superroc::hip::RuntimeRegisteredMappings("VmFlags: dc\n").empty(),
         "flags without a mapping");
}

void TestPinsOnce() {
  constexpr std::size_t kBytes = 1 << 20;
  void* memory = mmap(nullptr, kBytes, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  Expect(memory != MAP_FAILED, "mmap");
  if (memory == MAP_FAILED) {
    return;
  }
  std::memset(memory, 1, kBytes);
  Expect(madvise(memory, kBytes, MADV_DONTFORK) == 0, "mark do-not-fork");
  const std::size_t before = PinnedBytes();
  const auto first = superroc::hip::PinRuntimeRegisteredMemory();
  Expect(first.ranges >= 1, "the marked mapping is found");
  Expect(first.pinned_bytes + first.huge_bytes + first.refused_bytes >= kBytes,
         "every byte of it is accounted for");
  if (superroc::hip::LongTermPinsAvailable()) {
    Expect(first.refused_ranges == 0, "nothing refused where pins work");
    Expect(first.pinned_bytes >= kBytes, "pinned bytes cover the mapping");
    Expect(PinnedBytes() >= before + kBytes, "the kernel counts the pin");
    const std::size_t pinned = PinnedBytes();
    const auto second = superroc::hip::PinRuntimeRegisteredMemory();
    Expect(second.refused_ranges == 0 && second.pinned_bytes >= kBytes,
           "a second call reports the range as pinned");
    Expect(PinnedBytes() == pinned, "and does not pin it again");
    // Once the mapping is gone the next call gives its pin back.
    Expect(munmap(memory, kBytes) == 0, "munmap");
    (void)superroc::hip::PinRuntimeRegisteredMemory();
    Expect(PinnedBytes() + kBytes <= pinned, "an unmapped range is unpinned");
    return;
  } else {
    // io_uring unavailable or a small locked-memory limit: a mapping below
    // one huge page cannot be held either, which is reported, and it stays
    // usable.
    std::printf("note: no long-term pins here; %zu bytes refused\n",
                first.refused_bytes);
    Expect(first.pinned_bytes == 0, "nothing reported as pinned");
    Expect(first.refused_bytes > 0, "a refusal counts its bytes");
  }
  static_cast<char*>(memory)[kBytes - 1] = 2;
  Expect(static_cast<char*>(memory)[0] == 1, "the mapping keeps its contents");
}

// The fallback for memory that cannot be pinned: whole 2 MiB units end up in
// huge pages where the kernel offers them, and the contents survive.
void TestHoldInPlace() {
  constexpr std::size_t kHuge = 2 << 20;
  constexpr std::size_t kBytes = 3 * kHuge;
  auto* const base =
      static_cast<char*>(mmap(nullptr, kBytes + kHuge, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  Expect(base != MAP_FAILED, "mmap");
  if (base == MAP_FAILED) {
    return;
  }
  // Start half a unit past a boundary: one whole unit lies after the first
  // boundary and before the last.
  const auto address = reinterpret_cast<std::uintptr_t>(base);
  char* const data = base + (kHuge - address % kHuge) % kHuge + kHuge / 2;
  const std::size_t bytes = 2 * kHuge;
  for (std::size_t i = 0; i < bytes; i += 4096) {
    data[i] = static_cast<char>(i >> 12);
  }
  bool locked = true;
  const std::size_t huge =
      superroc::hip::HoldHostPagesInPlace(data, bytes, &locked);
  Expect(huge == 0 || huge == kHuge, "only whole 2 MiB units count as huge");
  std::printf("note: lock %s\n", locked ? "granted" : "refused");
  bool untouched = true;
  Expect(superroc::hip::HoldHostPagesInPlace(nullptr, 0, &untouched) == 0 &&
             !untouched,
         "an empty range reports no lock");
  bool intact = true;
  for (std::size_t i = 0; i < bytes; i += 4096) {
    intact = intact && data[i] == static_cast<char>(i >> 12);
  }
  Expect(intact, "the contents survive");
  Expect(superroc::hip::HoldHostPagesInPlace(nullptr, 0) == 0, "empty range");
  std::printf("note: %zu of %zu bytes in huge pages\n", huge, bytes);
  Expect(munmap(base, kBytes + kHuge) == 0, "munmap");
}

}  // namespace

int main() {
  TestSelection();
  TestPinsOnce();
  TestHoldInPlace();
  if (failures != 0) {
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
