#ifndef SUPERROC_MODELS_QWEN_HIP_MTP_SNAPSHOT_GUARD_HPP_
#define SUPERROC_MODELS_QWEN_HIP_MTP_SNAPSHOT_GUARD_HPP_

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace superroc::hip {

// F7: the MTP draft KV prefix must cover exactly the target's committed
// context at capture. Draft and target advance in lockstep at every commit
// boundary (prime, append, proposal feedback, cancellation rewind), so a
// mismatch means the draft lagged: restoring it would resume drafting at the
// wrong position and every later Propose would throw (an HTTP 500 in serve).
// Refuse the capture instead; the serving path turns the throw into a
// snapshot skip, exactly like the other capture-time refusals.
// Header-only and HIP-free, so the CPU unit test
// (tests/models/qwen/hip/mtp_snapshot_guard_test.cpp) drives this exact
// function without a GPU or a model file.
inline void CheckMtpSnapshotPosition(std::uint32_t draft_position,
                                     std::size_t target_position) {
  if (draft_position != target_position) {
    throw std::logic_error("MTP snapshot refused: draft position " +
                           std::to_string(draft_position) +
                           " differs from target position " +
                           std::to_string(target_position));
  }
}

}  // namespace superroc::hip

#endif  // SUPERROC_MODELS_QWEN_HIP_MTP_SNAPSHOT_GUARD_HPP_
