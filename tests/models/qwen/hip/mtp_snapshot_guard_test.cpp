// F7: the MTP capture-site position check.
//
// A full capture needs a GPU-resident target executor plus a primed MTP draft
// backend, which the CPU test target cannot build. This test drives the exact
// guard function the capture path calls (CheckMtpSnapshotPosition) with
// synthetic positions, so both directions of mismatch are covered instead of
// only the matching case the live probe exercises.
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "src/models/qwen/hip/mtp/snapshot_guard.hpp"

namespace {

int fails = 0;

void Expect(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "FAIL " << what << "\n";
    ++fails;
  }
}

void ExpectThrows(std::uint32_t draft, std::size_t target, const char* what) {
  try {
    gufo::hip::CheckMtpSnapshotPosition(draft, target);
  } catch (const std::logic_error& error) {
    const std::string message = error.what();
    Expect(message.find(std::to_string(draft)) != std::string::npos &&
               message.find(std::to_string(target)) != std::string::npos,
           what);
    return;
  }
  std::cerr << "FAIL " << what << " (no throw)\n";
  ++fails;
}

void ExpectPasses(std::uint32_t draft, std::size_t target, const char* what) {
  try {
    gufo::hip::CheckMtpSnapshotPosition(draft, target);
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << what << " (threw: " << error.what() << ")\n";
    ++fails;
  }
}

}  // namespace

int main() {
  // The live invariant: draft and target advance in lockstep, so equal
  // positions (including an empty session and a full-context session) pass.
  ExpectPasses(0, 0, "empty session captures");
  ExpectPasses(13376, 13376, "matching positions capture");
  ExpectPasses(0xFFFFFFFFU, std::size_t{0xFFFFFFFFU},
               "full-range equal positions capture");

  // Either direction of drift refuses the capture with a diagnosable reason.
  ExpectThrows(13375, 13376, "lagging draft refuses and names both positions");
  ExpectThrows(13377, 13376, "leading draft refuses and names both positions");
  ExpectThrows(0, 1, "unprimed draft against a live target refuses");
  ExpectThrows(1, 0, "live draft against an empty target refuses");

  if (fails == 0) {
    std::cout << "qwen_mtp_snapshot_guard_test: passed\n";
  }
  return fails == 0 ? 0 : 1;
}
