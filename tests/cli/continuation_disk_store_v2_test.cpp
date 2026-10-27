// Tests for the version-2 on-disk continuation format: a per-chunk SHA-256
// table verified on threads at restore time, a header-only startup index, mmap
// staging instead of an intermediate heap copy, and version-1 files as clean
// misses. CPU only: no GPU, no network, no model files.
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/continuation_disk_store.hpp"
#include "src/core/crypto/sha256.hpp"

namespace {

using superroc::server::ChatRequest;
using superroc::server::ContinuationDiskEvent;
using superroc::server::ContinuationDiskEventAction;
using superroc::server::ContinuationDiskEventReason;
using superroc::server::ContinuationDiskStore;
using superroc::server::ContinuationDiskStoreOptions;
using superroc::server::TextDecodeSelection;
using superroc::server::TextExecutionPlan;
using superroc::server::TextExecutionPlanKind;
using superroc::server::TextModelRunner;
using superroc::server::TextPrefillStep;
using superroc::server::TextRunnerCapabilities;
using superroc::server::TextRunnerDescriptor;
using superroc::server::TextRunnerPersistenceDescriptor;
using superroc::server::TextRunnerResourceClaim;
using superroc::server::TextRunnerSnapshot;
using superroc::server::TextRunnerState;
using superroc::server::TextRunnerToken;

constexpr std::size_t kChunkBytes = std::size_t{4} * 1024U * 1024U;
constexpr std::size_t kDiskHeaderBytes = 96;
constexpr std::size_t kChecksumBytes = 64;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           "superroc-continuation-disk-v2-XXXXXX")
                              .string();
    char* created = ::mkdtemp(pattern.data());
    if (created == nullptr) {
      throw std::runtime_error("failed to create temporary directory");
    }
    path_ = created;
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

template<typename Integer>
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     Integer value) {
  for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename Integer>
Integer GetLittleEndian(std::span<const std::uint8_t> source,
                        std::size_t offset) {
  Integer result = 0;
  for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
    result |= static_cast<Integer>(source[offset + byte]) << (byte * 8U);
  }
  return result;
}

class PatternState final : public TextRunnerState {
public:
  void Invalidate() noexcept override { value = 0; }
  std::uint64_t value{0};
};

class PatternSnapshot final : public TextRunnerSnapshot {
public:
  PatternSnapshot(std::size_t payload_bytes, std::uint64_t value)
      : bytes_(payload_bytes) {
    for (std::size_t index = 0; index < bytes_.size(); ++index) {
      bytes_[index] = static_cast<std::uint8_t>((index * 31U + 7U) % 251U);
    }
    PutLittleEndian<std::uint64_t>(bytes_, 0, value);
  }

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return bytes_.size();
  }
  [[nodiscard]] const std::vector<std::uint8_t>& bytes() const noexcept {
    return bytes_;
  }

private:
  std::vector<std::uint8_t> bytes_;
};

class PatternRunner final : public TextModelRunner {
public:
  PatternRunner(std::string identity, std::size_t payload_bytes)
      : identity_(identity.begin(), identity.end()),
        payload_bytes_(payload_bytes) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = "pattern-model",
        .state_abi = "pattern-state-v1",
        .max_context = 65536,
        .capabilities = TextRunnerCapabilities{.snapshot = true, .fork = true},
        .persistence =
            TextRunnerPersistenceDescriptor{
                .compatibility_identity = identity_,
                .payload_version = 11,
            },
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {};
  }
  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {{.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
  }
  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    std::vector<TextRunnerToken> tokens;
    for (const char value : text) {
      tokens.push_back(static_cast<unsigned char>(value));
    }
    return tokens;
  }
  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest&) const override {
    return std::nullopt;
  }
  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken>) const override {
    return {};
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<PatternState>();
  }
  [[nodiscard]] TextPrefillStep Prefill(TextRunnerState&,
                                        std::span<const TextRunnerToken>,
                                        std::size_t,
                                        std::size_t) const override {
    throw std::logic_error("pattern runner does not prefill");
  }
  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState&, superroc::sampling::SamplerState&) const override {
    throw std::logic_error("pattern runner does not decode");
  }
  void Advance(TextRunnerState&, TextRunnerToken) const override {
    throw std::logic_error("pattern runner does not advance");
  }
  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState&) const override {
    return 0;
  }
  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const TextRunnerState&) const override {
    return payload_bytes_;
  }
  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    return std::make_unique<PatternSnapshot>(
        payload_bytes_, dynamic_cast<const PatternState&>(state).value);
  }
  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    state.Invalidate();
    const auto* pattern = dynamic_cast<const PatternSnapshot*>(&snapshot);
    if (pattern != nullptr && !pattern->bytes().empty()) {
      dynamic_cast<PatternState&>(state).value =
          GetLittleEndian<std::uint64_t>(pattern->bytes(), 0);
    }
  }
  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    return dynamic_cast<const PatternSnapshot&>(snapshot).PayloadBytes();
  }
  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto& pattern = dynamic_cast<const PatternSnapshot&>(snapshot);
    if (destination.size() != pattern.bytes().size()) {
      throw std::invalid_argument("pattern destination size mismatch");
    }
    std::copy(pattern.bytes().begin(), pattern.bytes().end(),
              destination.begin());
    return destination.size();
  }
  // Deliberately ragged sink calls: chunk accumulation must be independent of
  // how the model chooses to split its serial stream.
  void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                const SnapshotSink& sink) const override {
    const auto& pattern = dynamic_cast<const PatternSnapshot&>(snapshot);
    const auto bytes = std::span<const std::uint8_t>(pattern.bytes());
    const std::size_t step = kChunkBytes / 3U;
    for (std::size_t offset = 0; offset < bytes.size(); offset += step) {
      sink(bytes.subspan(offset, std::min(step, bytes.size() - offset)));
    }
  }
  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    if (payload.size() != payload_bytes_) {
      throw std::invalid_argument("pattern payload size mismatch");
    }
    // Touch every mapped byte, so a mapping that was never populated would
    // fault here rather than pass silently.
    std::uint64_t sum = 0;
    for (const std::uint8_t value : payload) {
      sum += value;
    }
    restored_bytes_ = sum;
    dynamic_cast<PatternState&>(state).value =
        GetLittleEndian<std::uint64_t>(payload, 0);
  }

  [[nodiscard]] std::size_t payload_bytes() const noexcept {
    return payload_bytes_;
  }
  [[nodiscard]] std::uint64_t restored_bytes() const noexcept {
    return restored_bytes_;
  }
  [[nodiscard]] std::size_t identity_bytes() const noexcept {
    return identity_.size();
  }

private:
  std::vector<std::uint8_t> identity_;
  std::size_t payload_bytes_{0};
  mutable std::uint64_t restored_bytes_{0};
};

ContinuationDiskStoreOptions StoreOptions(
    const std::filesystem::path& directory,
    std::size_t capacity = 256U * 1024U * 1024U,
    std::size_t staging = 256U * 1024U * 1024U) {
  return {
      .directory = directory,
      .capacity_bytes = capacity,
      .staging_capacity_bytes = staging,
  };
}

std::vector<std::filesystem::path> CacheFiles(
    const std::filesystem::path& directory) {
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.path().extension() == ".kvc") {
      files.push_back(entry.path());
    }
  }
  return files;
}

std::vector<std::uint8_t> ReadWholeFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input),
                                   std::istreambuf_iterator<char>());
}

void WriteWholeFile(const std::filesystem::path& path,
                    std::span<const std::uint8_t> bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.close();
  // The store only accepts owner-private regular files, so fixtures written
  // by hand have to match what PublishImage produces.
  if (::chmod(path.c_str(), S_IRUSR | S_IWUSR) != 0) {
    throw std::runtime_error("failed to make fixture private");
  }
}

std::size_t ChunkCount(std::size_t payload_bytes) {
  return payload_bytes / kChunkBytes + (payload_bytes % kChunkBytes != 0U);
}

/// Where the chunk table starts for a file written by `runner`.
std::size_t TableOffset(const PatternRunner& runner, std::size_t token_count) {
  return kDiskHeaderBytes + runner.identity_bytes() +
         token_count * sizeof(std::uint32_t);
}

/// Where the payload starts, and therefore the file size for known inputs.
std::size_t FileBytes(const PatternRunner& runner, std::size_t token_count,
                      std::size_t payload_bytes) {
  return kDiskHeaderBytes + runner.identity_bytes() +
         token_count * sizeof(std::uint32_t) +
         ChunkCount(payload_bytes) * kChecksumBytes + payload_bytes;
}

void FlipByte(const std::filesystem::path& path, std::size_t offset) {
  auto bytes = ReadWholeFile(path);
  if (offset >= bytes.size()) {
    throw std::out_of_range("flip offset past end of file");
  }
  bytes[offset] ^= 0x5AU;
  WriteWholeFile(path, bytes);
}

ContinuationDiskStore::SaveResult SavePrefix(
    ContinuationDiskStore& store, const PatternRunner& runner,
    std::vector<TextRunnerToken> tokens, std::uint64_t value) {
  auto state = runner.CreateState();
  dynamic_cast<PatternState&>(*state).value = value;
  return store.Save(runner, tokens, *runner.Snapshot(*state));
}

bool RestorePrefix(ContinuationDiskStore& store, const PatternRunner& runner,
                   std::vector<TextRunnerToken> tokens, std::uint64_t* value) {
  auto state = runner.CreateState();
  const auto result = store.RestoreLongestPrefix(runner, *state, tokens);
  if (result.restored && value != nullptr) {
    *value = dynamic_cast<PatternState&>(*state).value;
  }
  return result.restored;
}

/// A file written by the version-1 writer: header, identity, tokens, payload,
/// and one whole-file SHA-256 with the digest field zeroed. The version-2
/// reader must never parse this and must never delete it.
std::vector<std::uint8_t> MakeVersionOneImage(
    const PatternRunner& runner, std::span<const TextRunnerToken> tokens,
    std::string_view identity, std::size_t payload_bytes) {
  const std::size_t total = kDiskHeaderBytes + identity.size() +
                            tokens.size() * sizeof(std::uint32_t) +
                            payload_bytes;
  std::vector<std::uint8_t> image(total);
  const std::array<std::uint8_t, 8> magic = {'G', 'U', 'F', 'O',
                                             'K', 'V', 'C', '1'};
  std::copy(magic.begin(), magic.end(), image.begin());
  PutLittleEndian<std::uint32_t>(image, 8, 1);
  PutLittleEndian<std::uint32_t>(image, 12, 11);
  PutLittleEndian<std::uint32_t>(image, 16,
                                 static_cast<std::uint32_t>(identity.size()));
  PutLittleEndian<std::uint32_t>(image, 20,
                                 static_cast<std::uint32_t>(tokens.size()));
  PutLittleEndian<std::uint64_t>(image, 24, payload_bytes);
  std::copy(identity.begin(), identity.end(), image.begin() + kDiskHeaderBytes);
  std::size_t offset = kDiskHeaderBytes + identity.size();
  for (const TextRunnerToken token : tokens) {
    PutLittleEndian<std::uint32_t>(image, offset, token);
    offset += sizeof(std::uint32_t);
  }
  for (std::size_t index = 0; index < payload_bytes; ++index) {
    image[offset + index] = static_cast<std::uint8_t>(index % 251U);
  }
  (void)runner;
  const std::string digest = superroc::crypto::Sha256Hex(image);
  std::copy(digest.begin(), digest.end(), image.begin() + 32);
  return image;
}

// ---------------------------------------------------------------------------

void TestVersionOneFileIsACleanMiss() {
  TemporaryDirectory directory;
  PatternRunner runner("v1-identity", 4096);
  const std::vector<TextRunnerToken> tokens{1, 2, 3};
  const auto image = MakeVersionOneImage(runner, tokens, "v1-identity", 4096);
  const auto path = directory.path() / "legacy.kvc";
  WriteWholeFile(path, image);

  std::vector<ContinuationDiskEvent> events;
  {
    ContinuationDiskStore store(
        StoreOptions(directory.path()),
        [&](const ContinuationDiskEvent& event) { events.push_back(event); });
    // The file is accounted for retention (one placeholder entry) but is
    // never inserted into the prefix index, so no lookup can reach it.
    Expect(store.entry_count() == 1,
           "a version-1 image is accounted for, never indexed");
    Expect(CacheFiles(directory.path()).size() == 1,
           "a version-1 image is left on disk, not deleted");
    Expect(store.retained_bytes() == image.size(),
           "a version-1 image is still accounted for retention");
    Expect(std::ranges::any_of(
               events,
               [](const ContinuationDiskEvent& event) {
                 return event.action == ContinuationDiskEventAction::kSkipped &&
                        event.reason ==
                            ContinuationDiskEventReason::kUnsupported;
               }),
           "startup reports a version-1 image as unsupported, not corrupt");
    std::uint64_t value = 0;
    Expect(!RestorePrefix(store, runner, {1, 2, 3, 4}, &value),
           "a version-1 image is a clean miss for lookup");
  }
  Expect(std::ranges::none_of(
             events,
             [](const ContinuationDiskEvent& event) {
               return event.reason ==
                      ContinuationDiskEventReason::kChecksumMismatch;
             }),
         "a version-1 image is never parsed as version 2");
  {
    // The ordinary retention path, not the reader, is what removes it.
    ContinuationDiskStore small(
        StoreOptions(directory.path(), 1024, 8U * 1024U * 1024U));
    Expect(small.entry_count() == 0 && CacheFiles(directory.path()).empty(),
           "retention eviction removes a stale version-1 image");
  }
}

void TestHeaderOnlyIndexIgnoresDamagedPayloads() {
  TemporaryDirectory directory;
  PatternRunner runner("header-only", 2 * kChunkBytes - 5);
  const std::vector<TextRunnerToken> tokens{5, 6, 7};
  {
    ContinuationDiskStore store(StoreOptions(directory.path()));
    Expect(SavePrefix(store, runner, tokens, 0x1122334455667788ULL).stored,
           "multi-chunk entry is persisted");
  }
  const auto files = CacheFiles(directory.path());
  Expect(files.size() == 1, "one multi-chunk file was published");
  const std::size_t payload_offset =
      TableOffset(runner, tokens.size()) +
      ChunkCount(runner.payload_bytes()) * kChecksumBytes;
  FlipByte(files.front(), payload_offset + kChunkBytes / 2);

  std::vector<ContinuationDiskEvent> events;
  {
    ContinuationDiskStore store(
        StoreOptions(directory.path()),
        [&](const ContinuationDiskEvent& event) { events.push_back(event); });
    Expect(store.entry_count() == 1,
           "startup indexes an entry without reading one payload byte");
    Expect(std::ranges::none_of(
               events,
               [](const ContinuationDiskEvent& event) {
                 return event.reason ==
                        ContinuationDiskEventReason::kChecksumMismatch;
               }),
           "startup reports no checksum problem it never looked for");
    std::uint64_t value = 0;
    Expect(!RestorePrefix(store, runner, {5, 6, 7, 8}, &value),
           "the damaged payload is refused when a restore needs it");
    Expect(store.entry_count() == 0 && CacheFiles(directory.path()).empty(),
           "the refused entry is removed from index and disk");
  }
  Expect(std::ranges::any_of(
             events,
             [](const ContinuationDiskEvent& event) {
               return event.action == ContinuationDiskEventAction::kMiss &&
                      event.reason ==
                          ContinuationDiskEventReason::kChecksumMismatch;
             }),
         "the restore miss reports a checksum mismatch");
}

void TestEveryChunkIsCoveredExactlyOnce() {
  const std::size_t payload_bytes = 2 * kChunkBytes + 11;
  const std::size_t chunk_count = ChunkCount(payload_bytes);
  Expect(chunk_count == 3, "fixture spans three chunks");
  for (std::size_t damaged = 0; damaged < chunk_count; ++damaged) {
    TemporaryDirectory directory;
    PatternRunner runner("cover", payload_bytes);
    const std::vector<TextRunnerToken> tokens{9, 10};
    {
      ContinuationDiskStore store(StoreOptions(directory.path()));
      Expect(SavePrefix(store, runner, tokens, 0xABCDEFULL).stored,
             "coverage fixture stores");
    }
    const auto files = CacheFiles(directory.path());
    const std::size_t payload_offset =
        TableOffset(runner, tokens.size()) + chunk_count * kChecksumBytes;
    FlipByte(files.front(), payload_offset + damaged * kChunkBytes + 3);

    ContinuationDiskStore store(StoreOptions(directory.path()));
    std::uint64_t value = 0;
    Expect(!RestorePrefix(store, runner, {9, 10, 11}, &value),
           "a flip inside any payload chunk is refused");
  }
  // And the same image with every chunk intact restores.
  TemporaryDirectory directory;
  PatternRunner runner("cover", payload_bytes);
  const std::vector<TextRunnerToken> tokens{9, 10};
  {
    ContinuationDiskStore store(StoreOptions(directory.path()));
    Expect(SavePrefix(store, runner, tokens, 0xABCDEFULL).stored,
           "clean coverage fixture stores");
  }
  ContinuationDiskStore store(StoreOptions(directory.path()));
  std::uint64_t value = 0;
  Expect(
      RestorePrefix(store, runner, {9, 10, 11}, &value) && value == 0xABCDEFULL,
      "the intact multi-chunk image restores exactly");
}

void TestMetadataDigestCoversTheChunkTable() {
  TemporaryDirectory directory;
  PatternRunner runner("table-digest", kChunkBytes + 3);
  const std::vector<TextRunnerToken> tokens{21, 22};
  {
    ContinuationDiskStore store(StoreOptions(directory.path()));
    Expect(SavePrefix(store, runner, tokens, 77).stored,
           "table digest fixture stores");
  }
  const auto files = CacheFiles(directory.path());
  const std::size_t table_offset = TableOffset(runner, tokens.size());
  FlipByte(files.front(), table_offset + 5);

  std::vector<ContinuationDiskEvent> events;
  ContinuationDiskStore store(
      StoreOptions(directory.path()),
      [&](const ContinuationDiskEvent& event) { events.push_back(event); });
  Expect(store.entry_count() == 0 && CacheFiles(directory.path()).empty() &&
             std::ranges::any_of(
                 events,
                 [](const ContinuationDiskEvent& event) {
                   return event.reason ==
                          ContinuationDiskEventReason::kChecksumMismatch;
                 }),
         "a damaged chunk table is caught by the metadata digest at startup");
}

void TestThreadedVerificationIsDeterministic() {
  const std::size_t payload_bytes = 2 * kChunkBytes + 1024;
  const char* const thread_counts[] = {"1", "2", "8", "16"};
  for (const char* const threads : thread_counts) {
    ::setenv("SUPERROC_KVC_VERIFY_THREADS", threads, 1);
    for (const bool corrupt : {false, true}) {
      TemporaryDirectory directory;
      PatternRunner runner("threaded", payload_bytes);
      const std::vector<TextRunnerToken> tokens{31, 32, 33};
      {
        ContinuationDiskStore store(StoreOptions(directory.path()));
        Expect(SavePrefix(store, runner, tokens, 4242).stored,
               "threaded fixture stores");
      }
      if (corrupt) {
        const auto files = CacheFiles(directory.path());
        const std::size_t payload_offset =
            TableOffset(runner, tokens.size()) +
            ChunkCount(payload_bytes) * kChecksumBytes;
        FlipByte(files.front(), payload_offset + kChunkBytes + 17);
      }
      ContinuationDiskStore store(StoreOptions(directory.path()));
      std::uint64_t value = 0;
      const bool restored =
          RestorePrefix(store, runner, {31, 32, 33, 34}, &value);
      Expect(restored == !corrupt,
             corrupt ? "any thread count refuses the same damaged chunk"
                     : "any thread count accepts the same clean image");
      if (!corrupt) {
        Expect(value == 4242,
               "thread count does not change the restored value");
      }
    }
  }
  ::unsetenv("SUPERROC_KVC_VERIFY_THREADS");
}

void TestMmapAndLegacyStagingAgree() {
  const std::size_t payload_bytes = kChunkBytes + 4096;
  TemporaryDirectory directory;
  PatternRunner runner("staging-modes", payload_bytes);
  const std::vector<TextRunnerToken> tokens{41, 42};
  {
    ContinuationDiskStore store(StoreOptions(directory.path()));
    Expect(SavePrefix(store, runner, tokens, 0x51525354ULL).stored,
           "staging-mode fixture stores");
  }
  std::uint64_t mapped_value = 0;
  std::uint64_t legacy_value = 0;
  {
    ::unsetenv("SUPERROC_KVC_LEGACY_STAGING");
    ContinuationDiskStore store(StoreOptions(directory.path()));
    Expect(RestorePrefix(store, runner, {41, 42, 43}, &mapped_value),
           "the mapped staging path restores");
  }
  {
    ::setenv("SUPERROC_KVC_LEGACY_STAGING", "1", 1);
    ContinuationDiskStore store(StoreOptions(directory.path()));
    Expect(RestorePrefix(store, runner, {41, 42, 43}, &legacy_value),
           "the legacy heap staging path restores");
    ::unsetenv("SUPERROC_KVC_LEGACY_STAGING");
  }
  Expect(mapped_value == legacy_value && mapped_value == 0x51525354ULL,
         "both staging paths restore identical bytes");
}

void TestTimestampAndMissingFileHazards() {
  TemporaryDirectory directory;
  PatternRunner runner("hazards", 4096);
  // Two entries with different token counts, so the file that belongs to the
  // first one is identifiable by size.
  const std::vector<TextRunnerToken> first{1, 1};
  const std::vector<TextRunnerToken> second{2, 2, 2};
  ContinuationDiskStore store(StoreOptions(directory.path()));
  Expect(SavePrefix(store, runner, first, 1).stored &&
             SavePrefix(store, runner, second, 2).stored,
         "hazard fixture stores two entries");
  const auto files = CacheFiles(directory.path());
  Expect(files.size() == 2, "two files exist before the hazard");

  // Simulate an entry removed behind this store's back (another instance, or
  // an operator). Lookup must clean up and keep serving, never loop or crash.
  const std::uintmax_t first_bytes =
      FileBytes(runner, first.size(), runner.payload_bytes());
  std::size_t removed = 0;
  for (std::size_t index = 0; index < files.size(); ++index) {
    if (std::filesystem::file_size(files[index]) == first_bytes) {
      removed = index;
    }
  }
  std::filesystem::remove(files[removed]);
  std::uint64_t first_value = 0;
  std::uint64_t second_value = 0;
  const bool first_restored =
      RestorePrefix(store, runner, {1, 1, 3}, &first_value);
  const bool second_restored =
      RestorePrefix(store, runner, {2, 2, 2, 3}, &second_value);
  Expect(!first_restored && second_restored && second_value == 2,
         "a vanished entry misses once and the store keeps serving");
  Expect(store.entry_count() == 1,
         "the vanished entry leaves the index exactly once");
}

}  // namespace

int main() {
  TestVersionOneFileIsACleanMiss();
  TestHeaderOnlyIndexIgnoresDamagedPayloads();
  TestEveryChunkIsCoveredExactlyOnce();
  TestMetadataDigestCoversTheChunkTable();
  TestThreadedVerificationIsDeterministic();
  TestMmapAndLegacyStagingAgree();
  TestTimestampAndMissingFileHazards();
  std::cout << "continuation disk store v2 tests passed\n";
  return 0;
}
