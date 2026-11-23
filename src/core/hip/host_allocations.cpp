#include "src/core/hip/host_allocations.hpp"

#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <mutex>
#include <new>
#include <string>

namespace superroc::hip {

namespace {

// Larger than any range the runtime registers by itself; the engine's own
// registered memory is far bigger and pinned where it is allocated.
constexpr std::size_t kMaxRuntimeRangeBytes = 1ULL << 30;

bool HasFlag(std::string_view flags, std::string_view flag) {
  std::size_t pos = 0;
  while (pos < flags.size()) {
    while (pos < flags.size() && flags[pos] == ' ') {
      ++pos;
    }
    std::size_t end = flags.find(' ', pos);
    if (end == std::string_view::npos) {
      end = flags.size();
    }
    if (flags.substr(pos, end - pos) == flag) {
      return true;
    }
    pos = end;
  }
  return false;
}

}  // namespace

std::vector<HostMapping> RuntimeRegisteredMappings(std::string_view smaps) {
  std::vector<HostMapping> mappings;
  HostMapping current;
  bool candidate = false;
  std::size_t pos = 0;
  while (pos < smaps.size()) {
    std::size_t end = smaps.find('\n', pos);
    if (end == std::string_view::npos) {
      end = smaps.size();
    }
    const std::string_view line = smaps.substr(pos, end - pos);
    pos = end + 1;
    // A mapping starts with `begin-end perms offset dev inode [name]`; every
    // other line of its block is `Key: value`.
    const std::size_t dash = line.find('-');
    const std::size_t space = line.find(' ');
    HostMapping header;
    // A number that does not fit is not an address of this process.
    const auto parsed = [](const char* first, const char* last,
                           std::uintptr_t* value) {
      const auto result = std::from_chars(first, last, *value, 16);
      return result.ec == std::errc{} && result.ptr == last;
    };
    if (dash != std::string_view::npos && space != std::string_view::npos &&
        dash < space && line.size() >= space + 5 &&
        parsed(line.data(), line.data() + dash, &header.begin) &&
        parsed(line.data() + dash + 1, line.data() + space, &header.end)) {
      current = header;
      const std::string_view perms = line.substr(space + 1, 4);
      // Offset, device, inode; a name after them means a file or a special
      // mapping. A line that ends before the inode is not a whole header.
      std::string_view rest = line.substr(space + 5);
      bool whole = true;
      for (int field = 0; field < 3; ++field) {
        while (!rest.empty() && rest.front() == ' ') {
          rest.remove_prefix(1);
        }
        if (rest.empty()) {
          whole = false;
          break;
        }
        const std::size_t gap = rest.find(' ');
        rest = gap == std::string_view::npos ? std::string_view{}
                                             : rest.substr(gap);
      }
      while (!rest.empty() && rest.front() == ' ') {
        rest.remove_prefix(1);
      }
      candidate = whole && rest.empty() && perms[0] == 'r' && perms[3] == 'p' &&
                  current.end > current.begin &&
                  current.end - current.begin <= kMaxRuntimeRangeBytes;
      continue;
    }
    constexpr std::string_view kFlags = "VmFlags:";
    if (candidate && line.starts_with(kFlags)) {
      const std::string_view flags = line.substr(kFlags.size());
      if (HasFlag(flags, "dc") && !HasFlag(flags, "nh") &&
          !HasFlag(flags, "io") && !HasFlag(flags, "pf")) {
        mappings.push_back(current);
      }
      candidate = false;
    }
  }
  return mappings;
}

bool LongTermPinsAvailable() noexcept {
  static const bool available = [] {
    constexpr std::size_t kPage = 4096;
    void* page = mmap(nullptr, kPage, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) {
      return false;
    }
    const int ring = PinHostPages(page, kPage);
    UnpinHostPages(ring);
    munmap(page, kPage);
    return ring >= 0;
  }();
  return available;
}

std::size_t HoldHostPagesInPlace(void* data, std::size_t bytes,
                                 bool* locked) noexcept {
  if (locked != nullptr) {
    *locked = false;
  }
  if (data == nullptr || bytes == 0) {
    return 0;
  }
  constexpr std::uintptr_t kHuge = 2ULL << 20;
  const auto begin = reinterpret_cast<std::uintptr_t>(data);
  const std::uintptr_t first = (begin + kHuge - 1) / kHuge * kHuge;
  const std::uintptr_t last = (begin + bytes) / kHuge * kHuge;
  std::size_t huge = 0;
  if (madvise(data, bytes, MADV_HUGEPAGE) == 0 && last > first) {
#if defined(MADV_COLLAPSE)
    // Moves the range once, now, instead of at any later time.
    if (madvise(reinterpret_cast<void*>(first), last - first, MADV_COLLAPSE) ==
        0) {
      huge = last - first;
    }
#endif
  }
#if defined(MLOCK_ONFAULT)
  const bool is_locked = syscall(SYS_mlock2, data, bytes, MLOCK_ONFAULT) == 0;
  if (locked != nullptr) {
    *locked = is_locked;
  }
#endif
  return huge;
}

RuntimePinReport PinRuntimeRegisteredMemory() noexcept {
  static std::mutex mutex;
  struct Held {
    HostMapping mapping;
    // The ring that pins the range, or -1 for a range held in huge pages.
    int ring;
    std::size_t huge_bytes;
    bool locked;
    // Passes that could not hold the range at all.
    int refusals;
  };
  // A range that no pass could hold is not tried a fourth time: every try
  // walks its page tables under the address-space lock.
  constexpr int kMaxRefusals = 3;
  static std::vector<Held>* const held = new (std::nothrow) std::vector<Held>;
  RuntimePinReport report;
  if (held == nullptr) {
    return report;
  }
  try {
    std::string smaps;
    if (std::FILE* file = std::fopen("/proc/self/smaps", "re")) {
      char buffer[1 << 16];
      for (;;) {
        const std::size_t got = std::fread(buffer, 1, sizeof(buffer), file);
        if (got == 0) {
          break;
        }
        smaps.append(buffer, got);
      }
      std::fclose(file);
    }
    const std::vector<HostMapping> mappings = RuntimeRegisteredMappings(smaps);
    const std::lock_guard lock(mutex);
    // A range the runtime has unmapped gives its pin back, and a later
    // mapping at the same address is handled as the new range it is.
    std::erase_if(*held, [&](const Held& entry) {
      if (std::find(mappings.begin(), mappings.end(), entry.mapping) !=
          mappings.end()) {
        return false;
      }
      UnpinHostPages(entry.ring);
      return true;
    });
    held->reserve(held->size() + mappings.size());
    for (const HostMapping& mapping : mappings) {
      const std::size_t bytes = mapping.end - mapping.begin;
      ++report.ranges;
      const auto account = [&](const Held& entry) {
        if (entry.ring >= 0) {
          report.pinned_bytes += bytes;
          return;
        }
        report.huge_bytes += entry.huge_bytes;
        if (!entry.locked) {
          report.unlocked_bytes += entry.huge_bytes;
        }
        if (entry.huge_bytes < bytes) {
          ++report.refused_ranges;
          report.refused_bytes += bytes - entry.huge_bytes;
        }
      };
      auto known = std::find_if(
          held->begin(), held->end(),
          [&](const Held& entry) { return entry.mapping == mapping; });
      if (known != held->end() && (known->ring >= 0 || known->huge_bytes != 0 ||
                                   known->refusals >= kMaxRefusals)) {
        account(*known);
        continue;
      }
      if (known == held->end()) {
        held->push_back({mapping, -1, 0, false, 0});
        known = std::prev(held->end());
      }
      auto* const data = reinterpret_cast<void*>(mapping.begin);
      // The ring is kept open on purpose: closing it would drop the pin.
      if (LongTermPinsAvailable()) {
        known->ring = PinHostPages(data, bytes);
      }
      if (known->ring < 0) {
        known->huge_bytes = HoldHostPagesInPlace(data, bytes, &known->locked);
        if (known->huge_bytes == 0) {
          ++known->refusals;
        }
      }
      account(*known);
    }
  } catch (...) {
    // Out of memory while reading the map: what is pinned stays pinned.
  }
  return report;
}

int PinHostPages(void* data, std::size_t bytes) noexcept {
  if (!data || !bytes)
    return -1;
  // Bound each kernel pinning range so request-side CPU page faults can
  // progress between ranges. One ring still owns the complete allocation.
  constexpr std::size_t max_buffer = 8ULL << 20;
  std::vector<iovec> ranges;
  try {
    for (std::size_t offset = 0; offset < bytes;) {
      const auto length = std::min(max_buffer, bytes - offset);
      ranges.push_back({static_cast<std::byte*>(data) + offset, length});
      offset += length;
    }
  } catch (const std::bad_alloc&) {
    return -1;
  }
  io_uring_params parameters{};
  const int ring = syscall(SYS_io_uring_setup, 1, &parameters);
  if (ring < 0)
    return -1;
  if (syscall(SYS_io_uring_register, ring, IORING_REGISTER_BUFFERS,
              ranges.data(), ranges.size()) == 0)
    return ring;
  close(ring);
  return -1;
}

void UnpinHostPages(int ring) noexcept {
  if (ring >= 0) {
    (void)syscall(SYS_io_uring_register, ring, IORING_UNREGISTER_BUFFERS,
                  nullptr, 0);
    close(ring);
  }
}

HostAllocations::~HostAllocations() {
  Reset();
}

hipError_t HostAllocations::Allocate(void** pointer, std::size_t bytes,
                                     unsigned flags) {
  if (!pointer)
    return hipErrorInvalidValue;
  *pointer = nullptr;
  void* data{};
  const auto status = hipHostMalloc(&data, bytes, flags);
  if (status != hipSuccess)
    return status;
  if (!data)
    return hipSuccess;
  const int pin = PinHostPages(data, bytes);
  try {
    allocations_.push_back({data, bytes, pin});
  } catch (const std::bad_alloc&) {
    UnpinHostPages(pin);
    (void)hipHostFree(data);
    return hipErrorOutOfMemory;
  }
  *pointer = data;
  return hipSuccess;
}

void HostAllocations::Reset() noexcept {
  for (const auto& allocation : allocations_) {
    UnpinHostPages(allocation.pin);
    (void)hipHostFree(allocation.data);
  }
  allocations_.clear();
}

std::size_t HostAllocations::PinnedBytes() const noexcept {
  std::size_t bytes{};
  for (const auto& allocation : allocations_)
    if (allocation.pin >= 0)
      bytes += allocation.bytes;
  return bytes;
}

}  // namespace superroc::hip
