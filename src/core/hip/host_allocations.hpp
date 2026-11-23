#ifndef SUPERROC_CORE_HIP_HOST_ALLOCATIONS_HPP_
#define SUPERROC_CORE_HIP_HOST_ALLOCATIONS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace superroc::hip {

// Best-effort Linux long-term pins prevent CPU compaction from moving GPU
// registrations. A restricted io_uring/memlock limit retains ordinary HIP
// registration. The caller must unpin before freeing the host allocation.
[[nodiscard]] int PinHostPages(void* data, std::size_t bytes) noexcept;
void UnpinHostPages(int ring) noexcept;

struct HostMapping {
  std::uintptr_t begin{};
  std::uintptr_t end{};
  friend bool operator==(const HostMapping&, const HostMapping&) = default;
};

// The anonymous private mappings in a /proc/<pid>/smaps text that carry the
// do-not-fork mark and are not opted out of huge pages. The GPU runtime puts
// that mark on process memory it registers with the driver on its own, such
// as the loaded GPU code; the engine's registered backing opts out of huge
// pages and holds its own pins.
[[nodiscard]] std::vector<HostMapping> RuntimeRegisteredMappings(
    std::string_view smaps);

// Whether this process can take long-term pins at all: io_uring is reachable
// and the locked-memory limit has room. Containers commonly filter io_uring
// out. Probed once.
[[nodiscard]] bool LongTermPinsAvailable() noexcept;

// For memory that must stay where it is but cannot be pinned: asks for huge
// pages, gathers the whole 2 MiB units of the range into them and locks the
// range. Memory compaction does not move huge pages, and locked pages are not
// swapped. Returns the bytes that are in huge-page units; everything here is
// best effort (no huge pages on this kernel, a small locked-memory limit).
// `locked` reports whether the lock was granted.
std::size_t HoldHostPagesInPlace(void* data, std::size_t bytes,
                                 bool* locked = nullptr) noexcept;

struct RuntimePinReport {
  std::size_t ranges{};
  std::size_t pinned_bytes{};
  // Not pinned, but held in huge pages (HoldHostPagesInPlace).
  std::size_t huge_bytes{};
  // The part of huge_bytes whose lock was refused: compaction leaves it in
  // place, swap does not.
  std::size_t unlocked_bytes{};
  std::size_t refused_ranges{};
  std::size_t refused_bytes{};
};

// Keeps the mappings RuntimeRegisteredMappings finds in this process where
// they are while they exist: a long-term pin where pins are available, huge
// pages otherwise. The kernel otherwise moves those pages when it compacts or
// swaps memory, and every move of a page the driver tracks stops the
// process's GPU queues until the range is mapped again. A range an earlier
// call handled is skipped, one that is gone gives its pin back, and one that
// three calls could not hold is not tried again; refused bytes are in neither
// a pin nor a huge page. Thread safe.
RuntimePinReport PinRuntimeRegisteredMemory() noexcept;

// Owns HIP host buffers and their long-term pins together. Allocations retain
// HIP's original coherence and huge-page policy. Drain readers and GPU work
// before destroying the owner.
class HostAllocations {
public:
  HostAllocations() = default;
  ~HostAllocations();
  HostAllocations(const HostAllocations&) = delete;
  HostAllocations& operator=(const HostAllocations&) = delete;

  [[nodiscard]] hipError_t Allocate(void** pointer, std::size_t bytes,
                                    unsigned flags = hipHostMallocDefault);
  template<typename T>
  [[nodiscard]] hipError_t Allocate(T** pointer, std::size_t bytes,
                                    unsigned flags = hipHostMallocDefault) {
    return Allocate(reinterpret_cast<void**>(pointer), bytes, flags);
  }
  void Reset() noexcept;
  [[nodiscard]] std::size_t PinnedBytes() const noexcept;

private:
  struct Allocation {
    void* data;
    std::size_t bytes;
    int pin;
  };
  std::vector<Allocation> allocations_;
};

}  // namespace superroc::hip
#endif
