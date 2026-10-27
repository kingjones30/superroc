#ifndef SUPERROC_CORE_HIP_STAGING_HPP_
#define SUPERROC_CORE_HIP_STAGING_HPP_

#include "src/cache/streaming.hpp"

namespace superroc::hip {
// Initialization-only pinned allocation for StreamedStore. Admission of the
// staging bytes occurs in StreamedStore before this allocator is invoked.
[[nodiscard]] cache::StagingAllocator PinnedStagingAllocator(
    cache::ResourceLedger&);
}  // namespace superroc::hip
#endif
