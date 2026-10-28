#include "src/cli/serve/continuation_disk_store.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <semaphore>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/env_compat.hpp"

namespace superroc::server {
namespace {

TextRunnerDescriptor DescriptorForInput(
    const TextModelRunner& runner,
    std::span<const std::uint8_t> input_identity) {
  auto descriptor = runner.Descriptor();
  if (input_identity.empty() || !descriptor.persistence)
    return descriptor;
  auto& identity = descriptor.persistence->compatibility_identity;
  const auto model_size = identity.size();
  identity.insert(identity.begin(), {'I', 'M', 'G', 1});
  for (unsigned shift = 0; shift < 64; shift += 8) {
    identity.push_back(
        static_cast<std::uint8_t>(std::uint64_t{model_size} >> shift));
  }
  identity.insert(identity.end(), input_identity.begin(), input_identity.end());
  return descriptor;
}

template<class Visit>
void VisitInputRanges(std::size_t token_count,
                      std::span<const std::uint8_t> identity,
                      std::span<const ContinuationInputPrefix> prefixes,
                      Visit&& visit) {
  std::size_t begin = 0;
  for (const auto& prefix : prefixes) {
    const auto end = std::min(prefix.token_count, token_count);
    if (end < begin)
      throw std::invalid_argument("input identity boundaries are invalid");
    visit(begin, end, std::span<const std::uint8_t>(prefix.identity));
    if (end == token_count)
      return;
    begin = end + 1;
  }
  visit(begin, token_count, identity);
}

constexpr std::array<std::uint8_t, 8> kMagic = {'S', 'R', 'O', 'C',
                                                'K', 'V', 'C', '1'};
/// Previous engine-name magic, kept in char form so the old spelling never
/// appears here; readers still accept files written by previous releases.
constexpr std::array<std::uint8_t, 8> kPreviousMagic = {'G', 'U', 'F', 'O',
                                                        'K', 'V', 'C', '1'};
// Format version 2. Version 1 hashed the whole image with one serial SHA-256
// stream and stored the digest where version 2 keeps the table digest. A
// version-1 file is a clean miss for this reader: it is never parsed as
// version 2 and is left for the ordinary LRU/retention eviction to remove.
constexpr std::uint32_t kFileVersion = 2;
constexpr std::uint32_t kLegacyFileVersion = 1;
constexpr std::size_t kChecksumBytes = 64;
constexpr std::size_t kMagicOffset = 0;
constexpr std::size_t kFileVersionOffset = 8;
constexpr std::size_t kPayloadVersionOffset = 12;
constexpr std::size_t kIdentityBytesOffset = 16;
constexpr std::size_t kTokenCountOffset = 20;
constexpr std::size_t kPayloadBytesOffset = 24;
constexpr std::size_t kChecksumOffset = 32;
constexpr std::size_t kHeaderBytes = kChecksumOffset + kChecksumBytes;
constexpr std::size_t kMaxIdentityBytes = std::size_t{64} * 1024U;
constexpr std::size_t kMaxTokenCount = std::size_t{16} * 1024U * 1024U;
// Version 2 splits the payload into fixed-size chunks, each with its own
// SHA-256 in a header table; the payload byte range is covered exactly once.
constexpr std::size_t kChunkBytes = std::size_t{4} * 1024U * 1024U;
constexpr std::size_t kMaxVerifyThreads = 64;
constexpr std::string_view kFileSuffix = ".kvc";
constexpr std::string_view kTemporaryPrefix = ".tmp-";

[[nodiscard]] constexpr std::size_t ChunkCount(std::size_t payload_bytes) {
  return payload_bytes / kChunkBytes +
         (payload_bytes % kChunkBytes != 0 ? 1U : 0U);
}

[[nodiscard]] constexpr std::size_t TableBytes(std::size_t payload_bytes) {
  return ChunkCount(payload_bytes) * kChecksumBytes;
}

/// Chunk verification threads: SUPERROC_KVC_VERIFY_THREADS wins, else the
/// machine's parallelism capped at 8. Tests pin it to 1, 2, 8 and 16.
[[nodiscard]] std::size_t VerifyThreads() noexcept {
  if (const char* text =
          ::superroc::core::GetEnvCompat("SUPERROC_KVC_VERIFY_THREADS")) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (end != text && parsed >= 1 && parsed <= kMaxVerifyThreads)
      return parsed;
  }
  const unsigned hardware = std::thread::hardware_concurrency();
  return std::clamp<std::size_t>(hardware == 0 ? 1U : hardware, 1,
                                 kMaxVerifyThreads < 8 ? kMaxVerifyThreads : 8);
}

[[nodiscard]] bool TimingEnabled() noexcept {
  const char* text = ::superroc::core::GetEnvCompat("SUPERROC_KVC_TIMING");
  return text != nullptr && text[0] != '\0' && text[0] != '0';
}

/// Measurement aid only: SUPERROC_KVC_LEGACY_STAGING=1 restores through the
/// previous "read the whole image into a heap buffer" staging path, with the
/// verification and runner calls unchanged, so the mmap change can be measured
/// from one binary. Unset in production.
[[nodiscard]] bool LegacyStagingEnabled() noexcept {
  const char* text =
      ::superroc::core::GetEnvCompat("SUPERROC_KVC_LEGACY_STAGING");
  return text != nullptr && text[0] != '\0' && text[0] != '0';
}

class ScopedFileDescriptor {
public:
  ScopedFileDescriptor() = default;
  explicit ScopedFileDescriptor(int descriptor) noexcept
      : descriptor_(descriptor) {}
  ~ScopedFileDescriptor() {
    if (descriptor_ >= 0) {
      ::close(descriptor_);
    }
  }

  ScopedFileDescriptor(const ScopedFileDescriptor&) = delete;
  ScopedFileDescriptor& operator=(const ScopedFileDescriptor&) = delete;

  ScopedFileDescriptor(ScopedFileDescriptor&& other) noexcept
      : descriptor_(std::exchange(other.descriptor_, -1)) {}
  ScopedFileDescriptor& operator=(ScopedFileDescriptor&& other) noexcept {
    if (this != &other) {
      if (descriptor_ >= 0) {
        ::close(descriptor_);
      }
      descriptor_ = std::exchange(other.descriptor_, -1);
    }
    return *this;
  }

  [[nodiscard]] int get() const noexcept { return descriptor_; }
  [[nodiscard]] int release() noexcept {
    return std::exchange(descriptor_, -1);
  }
  [[nodiscard]] explicit operator bool() const noexcept {
    return descriptor_ >= 0;
  }

private:
  int descriptor_{-1};
};

class ScopedOperationPermit {
public:
  explicit ScopedOperationPermit(std::binary_semaphore& gate, bool wait = true)
      : gate_(gate) {
    if (wait) {
      gate_.acquire();
      acquired_ = true;
    } else {
      acquired_ = gate_.try_acquire();
    }
  }
  ~ScopedOperationPermit() {
    if (acquired_)
      gate_.release();
  }
  explicit operator bool() const noexcept { return acquired_; }
  void Unlock() {
    gate_.release();
    acquired_ = false;
  }
  void Lock() {
    gate_.acquire();
    acquired_ = true;
  }

  ScopedOperationPermit(const ScopedOperationPermit&) = delete;
  ScopedOperationPermit& operator=(const ScopedOperationPermit&) = delete;
  ScopedOperationPermit(ScopedOperationPermit&&) = delete;
  ScopedOperationPermit& operator=(ScopedOperationPermit&&) = delete;

private:
  std::binary_semaphore& gate_;
  bool acquired_{false};
};

class ScopedFileLock {
public:
  ScopedFileLock(int descriptor, int operation) noexcept
      : descriptor_(descriptor) {
    int result;
    do {
      result = ::flock(descriptor_, operation);
    } while (result != 0 && errno == EINTR);
    acquired_ = result == 0;
  }

  ~ScopedFileLock() {
    if (acquired_) {
      while (::flock(descriptor_, LOCK_UN) != 0 && errno == EINTR) {
      }
    }
  }

  explicit operator bool() const noexcept { return acquired_; }

  ScopedFileLock(const ScopedFileLock&) = delete;
  ScopedFileLock& operator=(const ScopedFileLock&) = delete;

private:
  int descriptor_;
  bool acquired_{false};
};

template<typename Integer>
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     Integer value) {
  static_assert(std::is_unsigned_v<Integer>);
  if (offset > destination.size() ||
      sizeof(Integer) > destination.size() - offset) {
    throw std::out_of_range("continuation disk header write overflow");
  }
  for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename Integer>
bool GetLittleEndian(std::span<const std::uint8_t> source, std::size_t offset,
                     Integer* value) noexcept {
  static_assert(std::is_unsigned_v<Integer>);
  if (value == nullptr || offset > source.size() ||
      sizeof(Integer) > source.size() - offset) {
    return false;
  }
  Integer result = 0;
  for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
    result |= static_cast<Integer>(source[offset + byte]) << (byte * 8U);
  }
  *value = result;
  return true;
}

bool IsLowerHexDigest(std::string_view digest) noexcept {
  if (digest.size() != kChecksumBytes) {
    return false;
  }
  return std::ranges::all_of(digest, [](char value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
  });
}

std::string DefaultKeyHash(std::span<const std::uint8_t> bytes) {
  return crypto::Sha256Hex(bytes);
}

std::vector<std::uint8_t> MakeKeyBytes(
    const TextRunnerPersistenceDescriptor& persistence,
    std::span<const TextRunnerToken> tokens) {
  if (persistence.compatibility_identity.size() >
          std::numeric_limits<std::uint32_t>::max() ||
      tokens.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("continuation disk key is too large");
  }
  const std::size_t identity_bytes = persistence.compatibility_identity.size();
  if (tokens.size() >
      (std::numeric_limits<std::size_t>::max() - 12U - identity_bytes) /
          sizeof(std::uint32_t)) {
    throw std::overflow_error("continuation disk key size overflows");
  }
  std::vector<std::uint8_t> key(12U + identity_bytes +
                                tokens.size() * sizeof(std::uint32_t));
  PutLittleEndian<std::uint32_t>(key, 0,
                                 static_cast<std::uint32_t>(identity_bytes));
  PutLittleEndian<std::uint32_t>(key, 4, persistence.payload_version);
  PutLittleEndian<std::uint32_t>(key, 8,
                                 static_cast<std::uint32_t>(tokens.size()));
  std::copy(persistence.compatibility_identity.begin(),
            persistence.compatibility_identity.end(), key.begin() + 12);
  std::size_t offset = 12U + identity_bytes;
  for (const TextRunnerToken token : tokens) {
    PutLittleEndian<std::uint32_t>(key, offset, token);
    offset += sizeof(std::uint32_t);
  }
  return key;
}

std::size_t CheckedFileBytes(std::size_t identity_bytes,
                             std::size_t token_count,
                             std::size_t payload_bytes) {
  if (identity_bytes > kMaxIdentityBytes || token_count > kMaxTokenCount) {
    throw std::invalid_argument(
        "continuation disk metadata exceeds format limits");
  }
  if (token_count > (std::numeric_limits<std::size_t>::max() - kHeaderBytes -
                     identity_bytes) /
                        sizeof(std::uint32_t)) {
    throw std::overflow_error("continuation disk metadata size overflows");
  }
  std::size_t metadata_bytes =
      kHeaderBytes + identity_bytes + token_count * sizeof(std::uint32_t);
  const std::size_t table_bytes = TableBytes(payload_bytes);
  if (table_bytes > std::numeric_limits<std::size_t>::max() - metadata_bytes) {
    throw std::overflow_error("continuation disk file size overflows");
  }
  metadata_bytes += table_bytes;
  if (payload_bytes >
      std::numeric_limits<std::size_t>::max() - metadata_bytes) {
    throw std::overflow_error("continuation disk file size overflows");
  }
  return metadata_bytes + payload_bytes;
}

/// Byte offset of the payload for a given metadata extent.
[[nodiscard]] constexpr std::size_t PayloadOffset(std::size_t identity_bytes,
                                                  std::size_t token_count,
                                                  std::size_t payload_bytes) {
  return kHeaderBytes + identity_bytes + token_count * sizeof(std::uint32_t) +
         TableBytes(payload_bytes);
}

std::string ErrnoMessage(std::string_view operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

bool WriteAll(int descriptor, std::span<const std::uint8_t> bytes) noexcept {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t written =
        ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (written == 0) {
      return false;
    }
    offset += static_cast<std::size_t>(written);
  }
  return true;
}

bool ReadAll(int descriptor, std::span<std::uint8_t> bytes) noexcept {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count =
        ::read(descriptor, bytes.data() + offset, bytes.size() - offset);
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (count == 0) {
      return false;
    }
    offset += static_cast<std::size_t>(count);
  }
  return true;
}

bool HasSuffix(std::string_view value, std::string_view suffix) noexcept {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

/// Regular, owner-only, non-symlink check for one directory entry. Never
/// follows the final path component.
bool SafeRegularFileAt(int directory_fd, std::string_view filename,
                       struct stat* status) noexcept {
  if (directory_fd < 0 || filename.empty() ||
      filename.find('/') != std::string_view::npos ||
      filename.find("..") != std::string_view::npos || status == nullptr) {
    return false;
  }
  if (::fstatat(directory_fd, std::string(filename).c_str(), status,
                AT_SYMLINK_NOFOLLOW) != 0) {
    return false;
  }
  return S_ISREG(status->st_mode) && !S_ISLNK(status->st_mode) &&
         status->st_uid == ::geteuid() &&
         (status->st_mode & (S_IRWXG | S_IRWXO)) == 0 &&
         (status->st_mode & S_IRUSR) != 0;
}

/// Read-only mapping of one published cache file.
///
/// The payload used to be copied into a staging heap buffer before it could be
/// handed to the runner; at 56K context that copy was a 4 GiB anonymous
/// allocation plus a full read into it. Mapping instead lets the runner's
/// device copies read the page cache directly. The mapping is safe to keep
/// after the directory entry is unlinked (unlink does not invalidate an open
/// mapping) and nothing in this file ever truncates a published image, which
/// would raise SIGBUS on a later touch.
class MappedImage {
public:
  MappedImage() = default;
  MappedImage(const MappedImage&) = delete;
  MappedImage& operator=(const MappedImage&) = delete;
  MappedImage(MappedImage&& other) noexcept
      : file_(std::move(other.file_)),
        mapping_(std::exchange(other.mapping_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}
  MappedImage& operator=(MappedImage&& other) noexcept {
    if (this != &other) {
      Reset();
      file_ = std::move(other.file_);
      mapping_ = std::exchange(other.mapping_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }
  ~MappedImage() { Reset(); }

  void Reset() noexcept {
    if (mapping_ != nullptr) {
      (void)::munmap(mapping_, size_);
      mapping_ = nullptr;
    }
    size_ = 0;
    file_ = ScopedFileDescriptor();
  }

  [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept {
    return {static_cast<const std::uint8_t*>(mapping_), size_};
  }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] explicit operator bool() const noexcept {
    return mapping_ != nullptr && size_ != 0;
  }

  /// Maps `filename` under `directory_fd`. `max_bytes` is the staging budget;
  /// `file_bytes` receives the stat size when the entry is a safe regular file.
  static bool Open(int directory_fd, std::string_view filename,
                   std::size_t max_bytes, MappedImage* out,
                   ContinuationDiskEventReason* failure_reason,
                   std::size_t* file_bytes = nullptr) {
    if (out == nullptr || failure_reason == nullptr) {
      return false;
    }
    struct stat status{};
    if (!SafeRegularFileAt(directory_fd, filename, &status)) {
      *failure_reason = ContinuationDiskEventReason::kUnsafeFile;
      return false;
    }
    if (file_bytes != nullptr && status.st_size > 0) {
      *file_bytes = static_cast<std::size_t>(status.st_size);
    }
    if (status.st_size <= 0 || static_cast<std::uint64_t>(status.st_size) >
                                   static_cast<std::uint64_t>(max_bytes)) {
      *failure_reason = status.st_size > 0
                            ? ContinuationDiskEventReason::kStagingCapacity
                            : ContinuationDiskEventReason::kCorrupt;
      return false;
    }
    ScopedFileDescriptor file(::openat(directory_fd,
                                       std::string(filename).c_str(),
                                       O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (!file) {
      *failure_reason = ContinuationDiskEventReason::kIoFailure;
      return false;
    }
    // A racing rename must never hand this mapping a different inode than the
    // one that passed the safety check above.
    struct stat opened{};
    if (::fstat(file.get(), &opened) != 0 || opened.st_dev != status.st_dev ||
        opened.st_ino != status.st_ino || opened.st_size != status.st_size) {
      *failure_reason = ContinuationDiskEventReason::kIoFailure;
      return false;
    }
    const std::size_t size = static_cast<std::size_t>(status.st_size);
    void* mapping =
        ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file.get(), 0);
    if (mapping == MAP_FAILED) {
      *failure_reason = ContinuationDiskEventReason::kIoFailure;
      return false;
    }
    // Ask the kernel for readahead now: verification touches every page next,
    // so the I/O overlaps with the checksum pass instead of stalling it.
    (void)::madvise(mapping, size, MADV_WILLNEED);
    out->file_ = std::move(file);
    out->mapping_ = mapping;
    out->size_ = size;
    return true;
  }

private:
  ScopedFileDescriptor file_;
  void* mapping_{nullptr};
  std::size_t size_{0};
};

enum class ParseFailure : std::uint8_t {
  kNone,
  kCorrupt,
  kChecksum,
  /// A well-formed image of a format version this reader does not understand
  /// (version 1). Never parsed; the caller must treat it as a clean miss.
  kLegacy,
};

struct ParsedImage {
  TextRunnerPersistenceDescriptor persistence;
  std::vector<TextRunnerToken> tokens;
  std::size_t payload_offset{0};
  std::size_t table_offset{0};
  std::size_t table_bytes{0};
  std::size_t payload_bytes{0};
};

struct VerifyStats {
  double metadata_ms{0.0};
  double chunks_ms{0.0};
  std::size_t chunk_count{0};
  std::size_t threads{0};
};

[[nodiscard]] double MillisSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

/// The header fields every reader needs, before any byte of the payload is
/// looked at. Everything here comes from the first `kHeaderBytes` bytes plus
/// arithmetic on them, which is what makes a header-only startup index cheap.
struct ImageFields {
  std::uint32_t payload_version{0};
  std::size_t identity_bytes{0};
  std::size_t token_count{0};
  std::size_t payload_bytes{0};
  std::size_t payload_offset{0};
  std::size_t expected_bytes{0};
};

ParseFailure DecodeHeader(std::span<const std::uint8_t> header,
                          ImageFields* fields) {
  if (fields == nullptr || header.size() < kHeaderBytes ||
      (!std::equal(kMagic.begin(), kMagic.end(),
                   header.begin() + kMagicOffset) &&
       !std::equal(kPreviousMagic.begin(), kPreviousMagic.end(),
                   header.begin() + kMagicOffset))) {
    return ParseFailure::kCorrupt;
  }
  std::uint32_t file_version = 0;
  std::uint32_t payload_version = 0;
  std::uint32_t identity_bytes_u32 = 0;
  std::uint32_t token_count_u32 = 0;
  std::uint64_t payload_bytes_u64 = 0;
  if (!GetLittleEndian(header, kFileVersionOffset, &file_version) ||
      !GetLittleEndian(header, kPayloadVersionOffset, &payload_version) ||
      !GetLittleEndian(header, kIdentityBytesOffset, &identity_bytes_u32) ||
      !GetLittleEndian(header, kTokenCountOffset, &token_count_u32) ||
      !GetLittleEndian(header, kPayloadBytesOffset, &payload_bytes_u64)) {
    return ParseFailure::kCorrupt;
  }
  if (file_version != kFileVersion) {
    return file_version == kLegacyFileVersion ? ParseFailure::kLegacy
                                              : ParseFailure::kCorrupt;
  }
  if (payload_version == 0 || identity_bytes_u32 == 0 ||
      identity_bytes_u32 > kMaxIdentityBytes || token_count_u32 == 0 ||
      token_count_u32 > kMaxTokenCount || payload_bytes_u64 == 0 ||
      payload_bytes_u64 >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return ParseFailure::kCorrupt;
  }
  fields->payload_version = payload_version;
  fields->identity_bytes = identity_bytes_u32;
  fields->token_count = token_count_u32;
  fields->payload_bytes = static_cast<std::size_t>(payload_bytes_u64);
  try {
    fields->expected_bytes = CheckedFileBytes(
        fields->identity_bytes, fields->token_count, fields->payload_bytes);
    fields->payload_offset = PayloadOffset(
        fields->identity_bytes, fields->token_count, fields->payload_bytes);
  } catch (...) {
    return ParseFailure::kCorrupt;
  }
  return ParseFailure::kNone;
}

/// Verifies the metadata region `[0, payload_offset)` of a version-2 image and
/// decodes identity and tokens. The region must be exactly the metadata extent;
/// the caller has already confirmed the whole file is `fields.expected_bytes`
/// long. The metadata digest covers the header, the identity, the tokens and
/// the chunk table, so a damaged table is caught here, before any payload I/O.
ParseFailure VerifyMetadataRegion(std::span<const std::uint8_t> image,
                                  const ImageFields& fields,
                                  ParsedImage* parsed) {
  if (parsed == nullptr || image.size() != fields.payload_offset) {
    return ParseFailure::kCorrupt;
  }
  const std::string stored_checksum(
      reinterpret_cast<const char*>(image.data() + kChecksumOffset),
      kChecksumBytes);
  if (!IsLowerHexDigest(stored_checksum)) {
    return ParseFailure::kCorrupt;
  }
  const std::array<std::uint8_t, kChecksumBytes> zeros{};
  crypto::Sha256Hasher metadata_hasher;
  metadata_hasher.Update(image.first(kChecksumOffset));
  metadata_hasher.Update(zeros);
  metadata_hasher.Update(
      image.subspan(kHeaderBytes, image.size() - kHeaderBytes));
  if (metadata_hasher.FinishHex() != stored_checksum) {
    return ParseFailure::kChecksum;
  }
  parsed->persistence.payload_version = fields.payload_version;
  parsed->persistence.compatibility_identity.resize(fields.identity_bytes);
  std::memcpy(parsed->persistence.compatibility_identity.data(),
              image.data() + kHeaderBytes, fields.identity_bytes);
  parsed->tokens.resize(fields.token_count);
  std::size_t token_offset = kHeaderBytes + fields.identity_bytes;
  for (TextRunnerToken& token : parsed->tokens) {
    if (!GetLittleEndian(image, token_offset, &token)) {
      return ParseFailure::kCorrupt;
    }
    token_offset += sizeof(std::uint32_t);
  }
  parsed->payload_offset = fields.payload_offset;
  parsed->table_bytes = TableBytes(fields.payload_bytes);
  parsed->table_offset = fields.payload_offset - parsed->table_bytes;
  parsed->payload_bytes = fields.payload_bytes;
  return ParseFailure::kNone;
}

/// Verifies each payload chunk's SHA-256 against the header table. The verdict
/// depends only on the bytes: any mismatch fails regardless of thread count or
/// completion order, and a clean table always passes.
bool VerifyChunkTable(std::span<const std::uint8_t> payload,
                      std::span<const std::uint8_t> table,
                      std::size_t threads) {
  const std::size_t count = ChunkCount(payload.size());
  if (table.size() != count * kChecksumBytes) {
    return false;
  }
  if (count == 0) {
    return true;
  }
  std::atomic<std::size_t> next{0};
  std::atomic<bool> ok{true};
  const auto worker = [&] {
    std::size_t index = 0;
    while (ok.load(std::memory_order_relaxed) &&
           (index = next.fetch_add(1, std::memory_order_relaxed)) < count) {
      const std::size_t begin = index * kChunkBytes;
      const std::size_t length = std::min(kChunkBytes, payload.size() - begin);
      const std::string digest =
          crypto::Sha256Hex(payload.subspan(begin, length));
      const std::string_view stored(
          reinterpret_cast<const char*>(table.data() + index * kChecksumBytes),
          kChecksumBytes);
      if (digest != stored) {
        ok.store(false, std::memory_order_relaxed);
        return;
      }
    }
  };
  const std::size_t workers = std::clamp<std::size_t>(threads, 1, count);
  if (workers <= 1) {
    worker();
  } else {
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (std::size_t index = 0; index < workers; ++index) {
      pool.emplace_back(worker);
    }
    for (std::thread& thread : pool) {
      thread.join();
    }
  }
  return ok.load(std::memory_order_relaxed);
}

/// Parses and fully verifies one version-2 image mapped in memory. `stats` is
/// optional and only fills timing counters.
ParseFailure ParseAndVerifyImage(std::span<const std::uint8_t> image,
                                 ParsedImage* parsed,
                                 VerifyStats* stats = nullptr) {
  ImageFields fields;
  const ParseFailure header_failure =
      DecodeHeader(image.first(std::min(image.size(), kHeaderBytes)), &fields);
  if (header_failure != ParseFailure::kNone) {
    return header_failure;
  }
  if (fields.expected_bytes != image.size()) {
    return ParseFailure::kCorrupt;
  }
  const auto metadata_start = std::chrono::steady_clock::now();
  const ParseFailure metadata_failure =
      VerifyMetadataRegion(image.first(fields.payload_offset), fields, parsed);
  if (stats != nullptr) {
    stats->metadata_ms = MillisSince(metadata_start);
  }
  if (metadata_failure != ParseFailure::kNone) {
    return metadata_failure;
  }
  const auto chunk_start = std::chrono::steady_clock::now();
  const bool chunks_ok = VerifyChunkTable(
      image.subspan(parsed->payload_offset, parsed->payload_bytes),
      image.subspan(parsed->table_offset, parsed->table_bytes),
      VerifyThreads());
  if (stats != nullptr) {
    stats->chunks_ms = MillisSince(chunk_start);
    stats->chunk_count = ChunkCount(parsed->payload_bytes);
    stats->threads = VerifyThreads();
  }
  return chunks_ok ? ParseFailure::kNone : ParseFailure::kChecksum;
}

/// What a startup index read decided about one directory entry.
enum class IndexDisposition : std::uint8_t {
  /// Header valid, metadata verified: safe to index.
  kUsable,
  /// Not this reader's format version (version 1): leave the file untouched.
  kKeep,
  /// Corrupt or unsafe: delete the entry.
  kRemove,
  /// Larger than the staging budget: keep and account, but never index
  /// unverified tokens so a later restart with enough staging can reuse it.
  kStaging,
};

struct IndexReadResult {
  IndexDisposition disposition{IndexDisposition::kRemove};
  ContinuationDiskEventReason reason{ContinuationDiskEventReason::kCorrupt};
  ImageFields fields;
  std::vector<std::uint8_t> metadata;
  std::size_t file_bytes{0};
};

std::vector<std::uint8_t> BuildHeader(
    const TextRunnerPersistenceDescriptor& persistence,
    std::span<const TextRunnerToken> tokens, std::size_t payload_bytes) {
  const std::size_t file_bytes = CheckedFileBytes(
      persistence.compatibility_identity.size(), tokens.size(), payload_bytes);
  std::vector<std::uint8_t> image(file_bytes - payload_bytes);
  std::copy(kMagic.begin(), kMagic.end(), image.begin() + kMagicOffset);
  PutLittleEndian<std::uint32_t>(image, kFileVersionOffset, kFileVersion);
  PutLittleEndian<std::uint32_t>(image, kPayloadVersionOffset,
                                 persistence.payload_version);
  PutLittleEndian<std::uint32_t>(
      image, kIdentityBytesOffset,
      static_cast<std::uint32_t>(persistence.compatibility_identity.size()));
  PutLittleEndian<std::uint32_t>(image, kTokenCountOffset,
                                 static_cast<std::uint32_t>(tokens.size()));
  PutLittleEndian<std::uint64_t>(image, kPayloadBytesOffset,
                                 static_cast<std::uint64_t>(payload_bytes));
  std::copy(persistence.compatibility_identity.begin(),
            persistence.compatibility_identity.end(),
            image.begin() + kHeaderBytes);
  std::size_t token_offset =
      kHeaderBytes + persistence.compatibility_identity.size();
  for (const TextRunnerToken token : tokens) {
    PutLittleEndian<std::uint32_t>(image, token_offset, token);
    token_offset += sizeof(std::uint32_t);
  }
  return image;
}

}  // namespace

struct ContinuationDiskStore::Impl {
  struct Entry {
    std::string filename;
    std::string key_hash;
    TextRunnerPersistenceDescriptor persistence;
    std::vector<TextRunnerToken> tokens;
    std::size_t file_bytes{0};
    std::size_t payload_bytes{0};
    std::filesystem::file_time_type last_access;
  };

  using EntryIterator = std::list<Entry>::iterator;

  // Compressed token edges keep lookup proportional to the prompt length.
  // Full compatibility bytes and tokens remain the keys, never only a hash.
  struct PrefixNode {
    std::vector<TextRunnerToken> edge;
    std::optional<EntryIterator> entry;
    std::map<TextRunnerToken, std::unique_ptr<PrefixNode>> children;

    void Insert(std::span<const TextRunnerToken> tokens, EntryIterator value,
                bool overwrite = true) {
      if (tokens.empty()) {
        if (overwrite || !entry.has_value())
          entry = value;
        return;
      }
      auto& child = children[tokens.front()];
      if (!child) {
        child = std::make_unique<PrefixNode>();
        child->edge.assign(tokens.begin(), tokens.end());
        child->entry = value;
        return;
      }
      const auto mismatch = std::ranges::mismatch(child->edge, tokens);
      const auto shared =
          static_cast<std::size_t>(mismatch.in1 - child->edge.begin());
      if (shared < child->edge.size()) {
        auto branch = std::make_unique<PrefixNode>();
        branch->edge.assign(child->edge.begin(), child->edge.begin() + shared);
        child->edge.erase(child->edge.begin(), child->edge.begin() + shared);
        const auto key = child->edge.front();
        branch->children.emplace(key, std::move(child));
        child = std::move(branch);
      }
      child->Insert(tokens.subspan(shared), value, overwrite);
    }

    void Erase(std::span<const TextRunnerToken> tokens) noexcept {
      if (tokens.empty()) {
        entry.reset();
        return;
      }
      auto found = children.find(tokens.front());
      if (found == children.end())
        return;
      auto& child = *found->second;
      if (tokens.size() < child.edge.size() ||
          !std::ranges::equal(child.edge, tokens.first(child.edge.size()))) {
        return;
      }
      child.Erase(tokens.subspan(child.edge.size()));
      if (!child.entry && child.children.empty())
        children.erase(found);
    }

    void ErasePending(std::span<const TextRunnerToken> tokens,
                      EntryIterator pending) noexcept {
      if (tokens.empty()) {
        if (entry == pending)
          entry.reset();
        return;
      }
      auto found = children.find(tokens.front());
      if (found == children.end())
        return;
      auto& child = *found->second;
      if (tokens.size() < child.edge.size() ||
          !std::ranges::equal(child.edge, tokens.first(child.edge.size()))) {
        return;
      }
      child.ErasePending(tokens.subspan(child.edge.size()), pending);
      if (!child.entry && child.children.empty())
        children.erase(found);
    }

    struct Match {
      std::optional<EntryIterator> longest;
      std::vector<std::size_t> shared_boundaries;
    };

    [[nodiscard]] Match Find(std::span<const TextRunnerToken> prompt,
                             bool boundaries, EntryIterator skip) const {
      Match result;
      const PrefixNode* node = this;
      std::size_t offset = 0;
      while (true) {
        if (node->entry && node->entry != skip)
          result.longest = node->entry;
        if (offset == prompt.size())
          break;
        const auto next = node->children.find(prompt[offset]);
        if (boundaries && !node->entry && offset != 0 &&
            (node->children.size() > 1 ||
             (!node->children.empty() && next == node->children.end()))) {
          result.shared_boundaries.push_back(offset);
        }
        if (next == node->children.end())
          break;
        const auto& child = *next->second;
        const auto mismatch =
            std::ranges::mismatch(child.edge, prompt.subspan(offset));
        const auto shared =
            static_cast<std::size_t>(mismatch.in1 - child.edge.begin());
        offset += shared;
        if (shared != child.edge.size()) {
          if (boundaries && offset != 0 && offset < prompt.size()) {
            result.shared_boundaries.push_back(offset);
          }
          break;
        }
        node = &child;
      }
      return result;
    }
  };

  using PersistenceKey = std::pair<std::uint32_t, std::vector<std::uint8_t>>;
  static PersistenceKey PrefixKey(
      const TextRunnerPersistenceDescriptor& descriptor) {
    return {descriptor.payload_version, descriptor.compatibility_identity};
  }

  struct PendingSave {
    std::shared_ptr<const TextModelRunner> runner;
    std::vector<TextRunnerToken> tokens;
    std::shared_ptr<const TextRunnerSnapshot> snapshot;
    std::vector<std::uint8_t> identity;
    std::size_t retained_bytes;
    bool shared_prefix{false};
    bool mark_published{false};
  };

  Impl(ContinuationDiskStoreOptions store_options, EventSink sink,
       KeyHashFunction hasher)
      : options(std::move(store_options)),
        event_sink(std::move(sink)),
        key_hash(std::move(hasher)) {
    ValidateOptions();
    InitializeDirectory();
    IndexExistingFiles();
    EvictToCapacity();
    writer = std::thread([this] { PersistQueued(); });
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  ~Impl() {
    {
      std::lock_guard lock(queue_mutex);
      stopping = true;
    }
    queue_changed.notify_all();
    if (writer.joinable())
      writer.join();
    if (directory_fd >= 0) {
      ::close(directory_fd);
    }
  }

  void PersistQueued() noexcept {
    for (;;) {
      PendingSave job;
      {
        std::unique_lock lock(queue_mutex);
        queue_changed.wait(lock,
                           [this] { return stopping || !pending.empty(); });
        if (pending.empty())
          return;
        job = std::move(pending.front());
        pending.pop_front();
      }
      try {
        (void)Save(*job.runner, job.tokens, *job.snapshot, job.identity,
                   job.shared_prefix, /*clear_pending=*/true,
                   /*publish_mark=*/!job.mark_published);
      } catch (...) {
        Emit(ContinuationDiskEventAction::kSkipped,
             ContinuationDiskEventReason::kIoFailure, 0, 0, job.tokens.size());
      }
      const auto released = job.retained_bytes;
      // Release GPU/host snapshot storage before making its budget available.
      job = {};
      {
        std::lock_guard lock(queue_mutex);
        queued_bytes -= released;
      }
      queue_changed.notify_all();
    }
  }

  void ValidateOptions() {
    if (options.directory.empty()) {
      throw std::invalid_argument(
          "continuation disk directory must not be empty");
    }
    if (options.capacity_bytes == 0) {
      throw std::invalid_argument("continuation disk capacity must be nonzero");
    }
    if (options.staging_capacity_bytes == 0) {
      // The host snapshot budget already leaves half of available RAM free.
      // Disk staging gets a quarter of that budget. It allocates nothing
      // upfront, so a fixed cap would only refuse checkpoints larger than it.
      options.staging_capacity_bytes =
          std::min(options.capacity_bytes, HostSnapshotBudgetBytes() / 4);
    }
    if (options.staging_capacity_bytes < kHeaderBytes) {
      throw std::invalid_argument(
          "continuation disk staging capacity is too small");
    }
    if (!key_hash) {
      key_hash = DefaultKeyHash;
    }
  }

  void InitializeDirectory() {
    std::error_code error;
    const bool existed = std::filesystem::exists(options.directory, error);
    if (error) {
      throw std::runtime_error("failed to inspect continuation disk directory");
    }
    if (!existed) {
      std::filesystem::create_directories(options.directory, error);
      if (error) {
        throw std::runtime_error(
            "failed to create continuation disk directory");
      }
    }

    struct stat status{};
    if (::lstat(options.directory.c_str(), &status) != 0) {
      throw std::runtime_error(
          ErrnoMessage("failed to inspect continuation disk directory"));
    }
    if (!S_ISDIR(status.st_mode) || S_ISLNK(status.st_mode) ||
        status.st_uid != ::geteuid()) {
      throw std::runtime_error(
          "continuation disk directory is not a private owned directory");
    }
    if (::chmod(options.directory.c_str(), S_IRWXU) != 0) {
      throw std::runtime_error(
          ErrnoMessage("failed to secure continuation disk directory"));
    }

    directory_fd = ::open(options.directory.c_str(),
                          O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (directory_fd < 0) {
      throw std::runtime_error(
          ErrnoMessage("failed to open continuation disk directory"));
    }
  }

  void Emit(ContinuationDiskEventAction action,
            ContinuationDiskEventReason reason, std::size_t file_bytes,
            std::size_t payload_bytes, std::size_t token_count,
            double elapsed_ms = 0.0,
            std::size_t staging_used_bytes = 0) const noexcept {
    if (!event_sink) {
      return;
    }
    try {
      event_sink({
          .action = action,
          .reason = reason,
          .file_bytes = file_bytes,
          .payload_bytes = payload_bytes,
          .token_count = token_count,
          .retained_bytes = retained,
          .capacity_bytes = options.capacity_bytes,
          .staging_capacity_bytes = options.staging_capacity_bytes,
          .staging_used_bytes = staging_used_bytes,
          .elapsed_ms = elapsed_ms,
      });
    } catch (...) {
      return;
    }
  }

  /// Stage breakdown for the disk-restore critical path, off unless
  /// SUPERROC_KVC_TIMING is set. Every field is wall-clock milliseconds
  /// measured inside this store; the one field that covers work done by the
  /// runner is called out by name (runner_ms) because the host-to-device copies
  /// and their stream synchronisation happen behind RestorePersistentSnapshot.
  static void EmitTiming(std::string_view line) noexcept {
    if (!TimingEnabled()) {
      return;
    }
    static std::mutex timing_mutex;
    const std::lock_guard lock(timing_mutex);
    std::fprintf(stderr, "sq14-timing %.*s\n", static_cast<int>(line.size()),
                 line.data());
    std::fflush(stderr);
  }

  /// Called with queue_mutex held. Report capacity rejection before allocation
  /// or enqueue, where it would otherwise look like a successful cache capture.
  [[nodiscard]] bool CanStage(std::size_t bytes, std::size_t payload_bytes,
                              std::size_t token_count) const {
    if (stopping)
      return false;
    const auto used =
        queued_bytes + capture_bytes->load(std::memory_order_relaxed);
    if (bytes > options.capacity_bytes) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kByteCapacity, bytes, payload_bytes,
           token_count, 0.0, used);
      return false;
    }
    const auto limit =
        std::min(options.capacity_bytes, options.staging_capacity_bytes);
    if (bytes > limit || used > limit - bytes) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kStagingCapacity, bytes, payload_bytes,
           token_count, 0.0, used);
      return false;
    }
    return true;
  }

  [[nodiscard]] std::string Hash(std::span<const std::uint8_t> bytes) const {
    const std::string digest = key_hash(bytes);
    if (!IsLowerHexDigest(digest)) {
      throw std::runtime_error(
          "continuation disk key hash must be 64 lowercase hex characters");
    }
    return digest;
  }

  [[nodiscard]] std::string HashKey(
      const TextRunnerPersistenceDescriptor& persistence,
      std::span<const TextRunnerToken> tokens) const {
    return Hash(MakeKeyBytes(persistence, tokens));
  }

  [[nodiscard]] bool SafeRegularFile(std::string_view filename,
                                     struct stat* status) const noexcept {
    return SafeRegularFileAt(directory_fd, filename, status);
  }

  /// Startup index read. Reads the header, identity, tokens and chunk table
  /// and stops at the first payload byte, so starting up beside N files costs
  /// N metadata extents instead of N whole snapshots. The payload is not
  /// touched here: it is verified at restore time, where it is needed anyway.
  [[nodiscard]] IndexReadResult ReadIndexMetadata(
      std::string_view filename) const {
    IndexReadResult result;
    struct stat status{};
    if (!SafeRegularFile(filename, &status)) {
      result.disposition = IndexDisposition::kRemove;
      result.reason = ContinuationDiskEventReason::kUnsafeFile;
      return result;
    }
    if (status.st_size > 0) {
      result.file_bytes = static_cast<std::size_t>(status.st_size);
    }
    if (status.st_size <= 0 ||
        static_cast<std::uint64_t>(status.st_size) >
            static_cast<std::uint64_t>(options.staging_capacity_bytes)) {
      result.disposition = status.st_size > 0 ? IndexDisposition::kStaging
                                              : IndexDisposition::kRemove;
      result.reason = status.st_size > 0
                          ? ContinuationDiskEventReason::kStagingCapacity
                          : ContinuationDiskEventReason::kCorrupt;
      return result;
    }
    const std::size_t size = static_cast<std::size_t>(status.st_size);
    const ScopedFileDescriptor file(
        ::openat(directory_fd, std::string(filename).c_str(),
                 O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (!file) {
      result.disposition = IndexDisposition::kRemove;
      result.reason = ContinuationDiskEventReason::kIoFailure;
      return result;
    }
    result.metadata.resize(std::min<std::size_t>(size, kHeaderBytes));
    if (!ReadAll(file.get(), result.metadata)) {
      result.disposition = IndexDisposition::kRemove;
      result.reason = ContinuationDiskEventReason::kIoFailure;
      return result;
    }
    const ParseFailure header_failure =
        DecodeHeader(result.metadata, &result.fields);
    if (header_failure == ParseFailure::kLegacy) {
      // A version-1 image is a clean miss: never parsed as version 2, never
      // deleted here. Ordinary retention/LRU eviction removes it in time.
      result.disposition = IndexDisposition::kKeep;
      result.reason = ContinuationDiskEventReason::kUnsupported;
      return result;
    }
    if (header_failure != ParseFailure::kNone ||
        result.fields.expected_bytes != size) {
      result.disposition = IndexDisposition::kRemove;
      result.reason = ContinuationDiskEventReason::kCorrupt;
      return result;
    }
    result.metadata.resize(result.fields.payload_offset);
    std::size_t offset = kHeaderBytes;
    if (offset > result.metadata.size()) {
      offset = result.metadata.size();
    }
    while (offset < result.metadata.size()) {
      const ssize_t count =
          ::pread(file.get(), result.metadata.data() + offset,
                  result.metadata.size() - offset, static_cast<off_t>(offset));
      if (count < 0) {
        if (errno == EINTR) {
          continue;
        }
        result.disposition = IndexDisposition::kRemove;
        result.reason = ContinuationDiskEventReason::kIoFailure;
        return result;
      }
      if (count == 0) {
        result.disposition = IndexDisposition::kRemove;
        result.reason = ContinuationDiskEventReason::kIoFailure;
        return result;
      }
      offset += static_cast<std::size_t>(count);
    }
    result.disposition = IndexDisposition::kUsable;
    result.reason = ContinuationDiskEventReason::kNotFound;
    return result;
  }

  /// One restore's image bytes plus the stage timings that produced them.
  struct StagedImage {
    MappedImage mapping;
    std::vector<std::uint8_t> buffer;
    std::span<const std::uint8_t> bytes;
    bool legacy{false};
    double open_ms{0.0};
    double alloc_ms{0.0};
    double read_ms{0.0};
  };

  /// Loads the image for a restore. Production maps the published file, so
  /// there is no staging copy at all; SUPERROC_KVC_LEGACY_STAGING=1 keeps the
  /// previous read-into-heap path for measurement.
  [[nodiscard]] bool StageImage(
      std::string_view filename, StagedImage* staged,
      ContinuationDiskEventReason* failure_reason) const {
    if (staged == nullptr || failure_reason == nullptr) {
      return false;
    }
    if (!LegacyStagingEnabled()) {
      const auto started = std::chrono::steady_clock::now();
      if (!MappedImage::Open(directory_fd, filename,
                             options.staging_capacity_bytes, &staged->mapping,
                             failure_reason)) {
        return false;
      }
      staged->open_ms = MillisSince(started);
      staged->bytes = staged->mapping.bytes();
      return true;
    }
    struct stat status{};
    const auto open_started = std::chrono::steady_clock::now();
    if (!SafeRegularFile(filename, &status)) {
      *failure_reason = ContinuationDiskEventReason::kUnsafeFile;
      return false;
    }
    if (status.st_size <= 0 ||
        static_cast<std::uint64_t>(status.st_size) >
            static_cast<std::uint64_t>(options.staging_capacity_bytes)) {
      *failure_reason = status.st_size > 0
                            ? ContinuationDiskEventReason::kStagingCapacity
                            : ContinuationDiskEventReason::kCorrupt;
      return false;
    }
    const ScopedFileDescriptor file(
        ::openat(directory_fd, std::string(filename).c_str(),
                 O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (!file) {
      *failure_reason = ContinuationDiskEventReason::kIoFailure;
      return false;
    }
    staged->open_ms = MillisSince(open_started);
    const auto alloc_started = std::chrono::steady_clock::now();
    staged->buffer.resize(static_cast<std::size_t>(status.st_size));
    staged->alloc_ms = MillisSince(alloc_started);
    const auto read_started = std::chrono::steady_clock::now();
    if (!ReadAll(file.get(), staged->buffer)) {
      *failure_reason = ContinuationDiskEventReason::kIoFailure;
      return false;
    }
    std::uint8_t trailing = 0;
    while (true) {
      const ssize_t count = ::read(file.get(), &trailing, 1);
      if (count < 0 && errno == EINTR) {
        continue;
      }
      if (count != 0) {
        *failure_reason = count < 0 ? ContinuationDiskEventReason::kIoFailure
                                    : ContinuationDiskEventReason::kCorrupt;
        return false;
      }
      break;
    }
    // Restore uses the in-memory copy; drop the page-cache one.
    (void)::posix_fadvise(file.get(), 0, 0, POSIX_FADV_DONTNEED);
    staged->read_ms = MillisSince(read_started);
    staged->legacy = true;
    staged->bytes = staged->buffer;
    return true;
  }

  void RemoveFileOnly(std::string_view filename) const noexcept {
    if (!filename.empty()) {
      (void)::unlinkat(directory_fd, std::string(filename).c_str(), 0);
    }
  }

  void IndexExistingFiles() {
    std::error_code error;
    const std::filesystem::directory_iterator iterator(
        options.directory,
        std::filesystem::directory_options::skip_permission_denied, error);
    if (error) {
      throw std::runtime_error(
          "failed to enumerate continuation disk directory");
    }
    for (const auto& directory_entry : iterator) {
      const std::string filename = directory_entry.path().filename().string();
      if (filename.starts_with(kTemporaryPrefix)) {
        // Publishers hold a shared directory lock from before temporary-file
        // creation through rename. Defer orphan cleanup while any is active;
        // readers must not wait for another store's potentially long write.
        const ScopedFileLock cleanup_lock(directory_fd, LOCK_EX | LOCK_NB);
        if (cleanup_lock)
          RemoveFileOnly(filename);
        continue;
      }
      if (!HasSuffix(filename, kFileSuffix)) {
        continue;
      }

      // Header-only read: no payload byte is touched at startup.
      const auto read_started = std::chrono::steady_clock::now();
      IndexReadResult read = ReadIndexMetadata(filename);
      ++index_stats.files_seen;
      index_stats.bytes_read += read.metadata.size();
      index_stats.metadata_ms += MillisSince(read_started);
      if (read.disposition == IndexDisposition::kStaging) {
        // A smaller RAM budget must not destroy a previously valid cache.
        // Account the file for LRU/retention, but never index unverified
        // tokens. A restart with enough staging can validate and reuse it.
        auto access = directory_entry.last_write_time(error);
        if (error) {
          error.clear();
          access = std::filesystem::file_time_type::min();
        }
        entries.push_back({.filename = filename,
                           .key_hash = {},
                           .persistence = {},
                           .tokens = {},
                           .file_bytes = read.file_bytes,
                           .last_access = access});
        retained += read.file_bytes;
        retained_entries.fetch_add(1, std::memory_order_relaxed);
        ++index_stats.skipped;
        Emit(ContinuationDiskEventAction::kSkipped, read.reason,
             read.file_bytes, 0, 0);
        continue;
      }
      if (read.disposition == IndexDisposition::kKeep) {
        // A version-1 image: a clean miss for this reader. Do not parse it and
        // do not delete it; retention eviction will remove it in time.
        auto access = directory_entry.last_write_time(error);
        if (error) {
          error.clear();
          access = std::filesystem::file_time_type::min();
        }
        entries.push_back({.filename = filename,
                           .key_hash = {},
                           .persistence = {},
                           .tokens = {},
                           .file_bytes = read.file_bytes,
                           .last_access = access});
        retained += read.file_bytes;
        retained_entries.fetch_add(1, std::memory_order_relaxed);
        ++index_stats.skipped;
        Emit(ContinuationDiskEventAction::kSkipped, read.reason,
             read.file_bytes, 0, 0);
        continue;
      }
      if (read.disposition == IndexDisposition::kRemove) {
        RemoveFileOnly(filename);
        ++index_stats.removed;
        Emit(ContinuationDiskEventAction::kRemoved, read.reason,
             read.file_bytes, 0, 0);
        continue;
      }

      ParsedImage parsed;
      const ParseFailure parse_failure =
          VerifyMetadataRegion(read.metadata, read.fields, &parsed);
      if (parse_failure != ParseFailure::kNone) {
        RemoveFileOnly(filename);
        ++index_stats.removed;
        Emit(ContinuationDiskEventAction::kRemoved,
             parse_failure == ParseFailure::kChecksum
                 ? ContinuationDiskEventReason::kChecksumMismatch
                 : ContinuationDiskEventReason::kCorrupt,
             read.metadata.size(), 0, 0);
        continue;
      }

      std::filesystem::file_time_type last_access =
          directory_entry.last_write_time(error);
      if (error) {
        error.clear();
        last_access = std::filesystem::file_time_type::min();
      }
      const std::string digest = HashKey(parsed.persistence, parsed.tokens);
      const EntryIterator duplicate =
          FindExact(digest, parsed.persistence, parsed.tokens);
      if (duplicate != entries.end()) {
        if (duplicate->last_access >= last_access) {
          RemoveFileOnly(filename);
          ++index_stats.removed;
          Emit(ContinuationDiskEventAction::kRemoved,
               ContinuationDiskEventReason::kExactReplacement, read.file_bytes,
               parsed.payload_bytes, parsed.tokens.size());
          continue;
        }
        RemoveEntry(duplicate, ContinuationDiskEventReason::kExactReplacement);
      }
      entries.push_back({
          .filename = filename,
          .key_hash = digest,
          .persistence = std::move(parsed.persistence),
          .tokens = std::move(parsed.tokens),
          .file_bytes = read.file_bytes,
          .payload_bytes = parsed.payload_bytes,
          .last_access = last_access,
      });
      const EntryIterator added = std::prev(entries.end());
      index.emplace(added->key_hash, added);
      prefixes[PrefixKey(added->persistence)].Insert(added->tokens, added);
      retained += added->file_bytes;
      retained_entries.fetch_add(1, std::memory_order_relaxed);
      ++index_stats.indexed;
    }
    if (TimingEnabled()) {
      std::ostringstream line;
      line << "event=disk_index files=" << index_stats.files_seen
           << " indexed=" << index_stats.indexed
           << " removed=" << index_stats.removed
           << " skipped=" << index_stats.skipped
           << " bytes_read=" << index_stats.bytes_read
           << " index_ms=" << std::fixed << std::setprecision(3)
           << index_stats.metadata_ms << " retained_bytes=" << retained;
      EmitTiming(line.str());
    }
  }

  [[nodiscard]] EntryIterator FindExact(
      const std::string& digest,
      const TextRunnerPersistenceDescriptor& persistence,
      std::span<const TextRunnerToken> tokens) {
    const auto [begin, end] = index.equal_range(digest);
    for (auto current = begin; current != end; ++current) {
      const EntryIterator entry = current->second;
      if (entry->persistence == persistence &&
          std::ranges::equal(entry->tokens, tokens)) {
        return entry;
      }
    }
    return entries.end();
  }

  void EraseIndex(EntryIterator entry) {
    const auto prefix = prefixes.find(PrefixKey(entry->persistence));
    if (prefix != prefixes.end()) {
      prefix->second.Erase(entry->tokens);
      if (prefix->second.children.empty() && !prefix->second.entry) {
        prefixes.erase(prefix);
      }
    }
    const auto [begin, end] = index.equal_range(entry->key_hash);
    for (auto current = begin; current != end; ++current) {
      if (current->second == entry) {
        index.erase(current);
        return;
      }
    }
  }

  bool RemoveEntry(EntryIterator entry,
                   ContinuationDiskEventReason reason) noexcept {
    if (entry == entries.end()) {
      return true;
    }
    if (::unlinkat(directory_fd, entry->filename.c_str(), 0) != 0 &&
        errno != ENOENT) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kIoFailure, entry->file_bytes,
           entry->payload_bytes, entry->tokens.size());
      return false;
    }
    const std::size_t file_bytes = entry->file_bytes;
    const std::size_t payload_bytes = entry->payload_bytes;
    const std::size_t token_count = entry->tokens.size();
    EraseIndex(entry);
    retained -= file_bytes;
    entries.erase(entry);
    retained_entries.fetch_sub(1, std::memory_order_relaxed);
    Emit(ContinuationDiskEventAction::kRemoved, reason, file_bytes,
         payload_bytes, token_count);
    return true;
  }

  [[nodiscard]] EntryIterator LeastRecentlyUsed() {
    EntryIterator selected = entries.end();
    for (auto current = entries.begin(); current != entries.end(); ++current) {
      if (selected == entries.end() ||
          current->last_access < selected->last_access ||
          (current->last_access == selected->last_access &&
           current->filename < selected->filename)) {
        selected = current;
      }
    }
    return selected;
  }

  void EvictToCapacity() {
    while (retained > options.capacity_bytes) {
      const EntryIterator victim = LeastRecentlyUsed();
      if (victim == entries.end() ||
          !RemoveEntry(victim, ContinuationDiskEventReason::kLru)) {
        break;
      }
    }
  }

  bool MakeCapacity(std::size_t file_bytes) {
    if (file_bytes > options.capacity_bytes) {
      return false;
    }
    while (retained > options.capacity_bytes - file_bytes) {
      const EntryIterator victim = LeastRecentlyUsed();
      if (victim == entries.end() ||
          !RemoveEntry(victim, ContinuationDiskEventReason::kLru)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] std::string UniqueSuffix() {
    const std::uint64_t random_value =
        (static_cast<std::uint64_t>(random_device()) << 32U) ^
        static_cast<std::uint64_t>(random_device()) ^
        static_cast<std::uint64_t>(++unique_counter) ^
        static_cast<std::uint64_t>(::getpid());
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << random_value;
    return output.str();
  }

  [[nodiscard]] std::string NewFinalFilename(const std::string& digest) {
    for (int attempt = 0; attempt < 64; ++attempt) {
      const std::string filename =
          digest + "-" + UniqueSuffix() + std::string(kFileSuffix);
      struct stat status{};
      if (::fstatat(directory_fd, filename.c_str(), &status,
                    AT_SYMLINK_NOFOLLOW) != 0 &&
          errno == ENOENT) {
        return filename;
      }
    }
    throw std::runtime_error("failed to allocate a continuation disk filename");
  }

  [[nodiscard]] bool PublishImage(std::string_view final_filename,
                                  std::span<const std::uint8_t> header,
                                  const TextModelRunner& runner,
                                  const TextRunnerSnapshot& snapshot,
                                  std::size_t payload_bytes) {
    // Save serializes publishers in this instance with write_mutex, and
    // startup indexing completes before its worker starts. Thus no two lock
    // owners in this instance can share directory_fd's open file description.
    const ScopedFileLock publication_lock(directory_fd, LOCK_SH);
    if (!publication_lock)
      return false;
    const std::string temporary_filename =
        std::string(kTemporaryPrefix) + UniqueSuffix();
    ScopedFileDescriptor temporary(
        ::openat(directory_fd, temporary_filename.c_str(),
                 O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                 S_IRUSR | S_IWUSR));
    if (!temporary) {
      return false;
    }
    bool valid = true;
    try {
      // `header` already carries the metadata extent, with the chunk table
      // still zeroed as a placeholder, so the payload lands sequentially at
      // exactly the offset the reader expects.
      if (!WriteAll(temporary.get(), header)) {
        throw std::runtime_error("continuation disk header write failed");
      }
      std::vector<std::string> chunk_digests;
      chunk_digests.reserve(ChunkCount(payload_bytes));
      std::vector<std::uint8_t> chunk;
      chunk.reserve(kChunkBytes);
      std::size_t written = 0;
      const auto flush_chunk = [&] {
        if (chunk.empty()) {
          return;
        }
        if (!WriteAll(temporary.get(), chunk)) {
          throw std::runtime_error("continuation disk write failed");
        }
        chunk_digests.push_back(crypto::Sha256Hex(chunk));
        chunk.clear();
      };
      runner.StreamPersistentSnapshot(snapshot, [&](auto bytes) {
        if (bytes.size() > payload_bytes - written) {
          throw std::runtime_error("snapshot serialization overflow");
        }
        written += bytes.size();
        // Fixed-size chunks, digesting and writing each byte exactly once.
        while (!bytes.empty()) {
          const std::size_t take =
              std::min(bytes.size(), kChunkBytes - chunk.size());
          chunk.insert(chunk.end(), bytes.begin(), bytes.begin() + take);
          bytes = bytes.subspan(take);
          if (chunk.size() == kChunkBytes) {
            flush_chunk();
          }
        }
      });
      if (written != payload_bytes) {
        throw std::runtime_error("snapshot serialization truncated");
      }
      flush_chunk();
      if (chunk_digests.size() != ChunkCount(payload_bytes)) {
        throw std::runtime_error("snapshot chunk count mismatch");
      }
      const std::size_t table_bytes = TableBytes(payload_bytes);
      const std::size_t table_offset = header.size() - table_bytes;
      std::string table;
      table.reserve(chunk_digests.size() * kChecksumBytes);
      for (const std::string& digest : chunk_digests) {
        table += digest;
      }
      if (::lseek(temporary.get(), static_cast<off_t>(table_offset), SEEK_SET) <
              0 ||
          !WriteAll(temporary.get(),
                    {reinterpret_cast<const std::uint8_t*>(table.data()),
                     table.size()})) {
        throw std::runtime_error("snapshot table write failed");
      }
      // Metadata digest over the header, identity, tokens and the real table.
      const std::array<std::uint8_t, kChecksumBytes> zeros{};
      crypto::Sha256Hasher metadata_hasher;
      metadata_hasher.Update(header.first(kChecksumOffset));
      metadata_hasher.Update(zeros);
      metadata_hasher.Update(
          header.subspan(kHeaderBytes, table_offset - kHeaderBytes));
      metadata_hasher.Update(
          {reinterpret_cast<const std::uint8_t*>(table.data()), table.size()});
      const std::string checksum = metadata_hasher.FinishHex();
      if (::lseek(temporary.get(), kChecksumOffset, SEEK_SET) < 0 ||
          !WriteAll(temporary.get(),
                    {reinterpret_cast<const std::uint8_t*>(checksum.data()),
                     checksum.size()})) {
        throw std::runtime_error("snapshot checksum write failed");
      }
    } catch (...) {
      valid = false;
    }
    valid = valid && ::fsync(temporary.get()) == 0;
    if (valid) {
      // The image is durable now. Drop its clean pages so multi-GB snapshots
      // don't accumulate in the page cache.
      (void)::posix_fadvise(temporary.get(), 0, 0, POSIX_FADV_DONTNEED);
    }
    const int raw_descriptor = temporary.release();
    if (::close(raw_descriptor) != 0) {
      valid = false;
    }
    if (!valid) {
      RemoveFileOnly(temporary_filename);
      return false;
    }
    if (::renameat(directory_fd, temporary_filename.c_str(), directory_fd,
                   std::string(final_filename).c_str()) != 0) {
      RemoveFileOnly(temporary_filename);
      return false;
    }
    if (::fsync(directory_fd) != 0) {
      RemoveFileOnly(final_filename);
      (void)::fsync(directory_fd);
      return false;
    }
    return true;
  }

  [[nodiscard]] SaveResult Save(
      const TextModelRunner& runner,
      std::span<const TextRunnerToken> checkpoint_tokens,
      const TextRunnerSnapshot& snapshot,
      std::span<const std::uint8_t> input_identity, bool shared_prefix = false,
      bool clear_pending = false, bool publish_mark = false) {
    // Serialize writers, but keep existing entries readable during payload
    // serialization, hashing and filesystem durability operations.
    const std::lock_guard write_lock(write_mutex);
    ScopedOperationPermit metadata_lock(operation_gate);
    const auto descriptor = DescriptorForInput(runner, input_identity);
    if (!descriptor.persistence.has_value()) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kUnsupported, 0, 0,
           checkpoint_tokens.size());
      return {};
    }
    const std::optional<PersistenceKey> pending_key =
        clear_pending
            ? std::optional<PersistenceKey>(PrefixKey(*descriptor.persistence))
            : std::nullopt;
    struct PendingMarkGuard {
      Impl* store;
      std::optional<PersistenceKey> key;
      std::span<const TextRunnerToken> tokens;
      ~PendingMarkGuard() {
        if (!key.has_value())
          return;
        const auto root = store->prefixes.find(*key);
        if (root != store->prefixes.end())
          root->second.ErasePending(tokens, store->entries.end());
      }
      void dismiss() { key.reset(); }
    } pending_mark_guard{this, pending_key, checkpoint_tokens};
    if (checkpoint_tokens.empty()) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kSerializationFailure, 0, 0, 0);
      return {};
    }
    // The same prefix is offered again whenever a restored continuation is
    // committed or several conversations share one system prompt. The stored
    // file already represents these tokens; refresh its recency and skip the
    // serialization and write.
    const std::string digest =
        HashKey(*descriptor.persistence, checkpoint_tokens);
    const EntryIterator existing =
        FindExact(digest, *descriptor.persistence, checkpoint_tokens);
    if (existing != entries.end()) {
      TouchEntry(existing);
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kExactReplacement, existing->file_bytes,
           existing->payload_bytes, checkpoint_tokens.size());
      return {};
    }

    // A continuation advances by a few hundred tokens per turn, so persisting
    // every turn rewrites a largely identical snapshot. Skipping writes that
    // barely move past a stored prefix trades that write for re-prefilling the
    // short gap, which the caller does anyway when no checkpoint matches.
    //
    // Shared prefixes are exempt: they exist to give a new conversation a place
    // to start from, and they sit only a few tokens past a deeper entry by
    // construction.
    if (!shared_prefix &&
        WithinCheckpointStep(runner, checkpoint_tokens, input_identity)) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kMinStep, 0, 0,
           checkpoint_tokens.size());
      return {};
    }

    std::size_t payload_bytes = 0;
    const auto started = std::chrono::steady_clock::now();
    std::vector<std::uint8_t> header;
    std::size_t file_bytes = 0;
    try {
      payload_bytes = runner.PersistentSnapshotPayloadBytes(snapshot);
      if (payload_bytes == 0) {
        throw std::runtime_error("persistent snapshot payload is empty");
      }
      file_bytes = CheckedFileBytes(
          descriptor.persistence->compatibility_identity.size(),
          checkpoint_tokens.size(), payload_bytes);
      if (file_bytes > options.staging_capacity_bytes) {
        Emit(ContinuationDiskEventAction::kSkipped,
             ContinuationDiskEventReason::kStagingCapacity, file_bytes,
             payload_bytes, checkpoint_tokens.size());
        return {};
      }
      if (file_bytes > options.capacity_bytes) {
        Emit(ContinuationDiskEventAction::kSkipped,
             ContinuationDiskEventReason::kByteCapacity, file_bytes,
             payload_bytes, checkpoint_tokens.size());
        return {};
      }
      header = BuildHeader(*descriptor.persistence, checkpoint_tokens,
                           payload_bytes);
    } catch (...) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kSerializationFailure, file_bytes,
           payload_bytes, checkpoint_tokens.size());
      return {};
    }

    // The enqueue could not publish the pending mark (a reader held the
    // gate); publish it now, under the gate, before the long write below.
    if (publish_mark) {
      prefixes[PrefixKey(*descriptor.persistence)].Insert(
          checkpoint_tokens, entries.end(), /*overwrite=*/false);
    }
    const std::string filename = NewFinalFilename(digest);
    metadata_lock.Unlock();
    const bool published =
        PublishImage(filename, header, runner, snapshot, payload_bytes);
    metadata_lock.Lock();
    if (!published) {
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kIoFailure, file_bytes, payload_bytes,
           checkpoint_tokens.size());
      return {};
    }
    // No visible entry is evicted until its replacement is fully durable.
    if (!MakeCapacity(file_bytes)) {
      RemoveFileOnly(filename);
      Emit(ContinuationDiskEventAction::kSkipped,
           ContinuationDiskEventReason::kByteCapacity, file_bytes,
           payload_bytes, checkpoint_tokens.size());
      return {};
    }

    pending_mark_guard.dismiss();
    entries.push_back({
        .filename = filename,
        .key_hash = digest,
        .persistence = *descriptor.persistence,
        .tokens = std::vector<TextRunnerToken>(checkpoint_tokens.begin(),
                                               checkpoint_tokens.end()),
        .file_bytes = file_bytes,
        .payload_bytes = payload_bytes,
        .last_access = std::filesystem::file_time_type::clock::now(),
    });
    const EntryIterator added = std::prev(entries.end());
    index.emplace(digest, added);
    prefixes[PrefixKey(added->persistence)].Insert(added->tokens, added);
    retained += file_bytes;
    retained_entries.fetch_add(1, std::memory_order_relaxed);
    Emit(ContinuationDiskEventAction::kStored,
         ContinuationDiskEventReason::kSaved, file_bytes, payload_bytes,
         checkpoint_tokens.size(),
         std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - started)
             .count());
    return {
        .stored = true,
        .file_bytes = file_bytes,
        .payload_bytes = payload_bytes,
    };
  }

  [[nodiscard]] EntryIterator FindLongestCandidate(
      const TextRunnerPersistenceDescriptor& persistence,
      std::span<const TextRunnerToken> prompt) {
    const auto root = prefixes.find(PrefixKey(persistence));
    if (root == prefixes.end())
      return entries.end();
    return root->second.Find(prompt, false, entries.end())
        .longest.value_or(entries.end());
  }

  [[nodiscard]] EntryIterator FindLongestInputCandidate(
      const TextModelRunner& runner, std::span<const TextRunnerToken> prompt,
      std::span<const std::uint8_t> input_identity,
      std::span<const ContinuationInputPrefix> input_prefixes) {
    auto best = entries.end();
    VisitInputRanges(
        prompt.size(), input_identity, input_prefixes,
        [&](auto begin, auto end, auto identity) {
          const auto descriptor = DescriptorForInput(runner, identity);
          const auto candidate =
              FindLongestCandidate(*descriptor.persistence, prompt.first(end));
          if (candidate != entries.end() && candidate->tokens.size() >= begin &&
              (best == entries.end() ||
               candidate->tokens.size() > best->tokens.size()))
            best = candidate;
        });
    return best;
  }

  [[nodiscard]] RestoreResult RestoreLongestPrefix(
      const TextModelRunner& runner, TextRunnerState& state,
      std::span<const TextRunnerToken> prompt,
      std::span<const std::uint8_t> input_identity,
      std::size_t stable_prefix_tokens,
      std::span<const ContinuationInputPrefix> input_prefixes) {
    const auto descriptor = DescriptorForInput(runner, input_identity);
    if (!descriptor.persistence.has_value()) {
      Emit(ContinuationDiskEventAction::kMiss,
           ContinuationDiskEventReason::kUnsupported, 0, 0, 0);
      return {};
    }

    while (true) {
      const EntryIterator candidate = FindLongestInputCandidate(
          runner, prompt, input_identity, input_prefixes);
      if (candidate == entries.end()) {
        Emit(ContinuationDiskEventAction::kMiss,
             ContinuationDiskEventReason::kNotFound, 0, 0, 0);
        return {};
      }
      if (stable_prefix_tokens != 0 &&
          candidate->tokens.size() > stable_prefix_tokens &&
          FindLongestInputCandidate(runner, prompt.first(stable_prefix_tokens),
                                    input_identity,
                                    input_prefixes) == entries.end()) {
        Emit(ContinuationDiskEventAction::kMiss,
             ContinuationDiskEventReason::kNotFound, 0, 0, 0);
        return {};
      }

      const auto restore_started = std::chrono::steady_clock::now();
      StagedImage image;
      ContinuationDiskEventReason failure_reason =
          ContinuationDiskEventReason::kCorrupt;
      if (!StageImage(candidate->filename, &image, &failure_reason)) {
        const std::size_t file_bytes = candidate->file_bytes;
        const std::size_t payload_bytes = candidate->payload_bytes;
        const std::size_t token_count = candidate->tokens.size();
        const bool removed = RemoveEntry(candidate, failure_reason);
        Emit(ContinuationDiskEventAction::kMiss, failure_reason, file_bytes,
             payload_bytes, token_count);
        if (!removed)
          return {};
        continue;
      }

      ParsedImage parsed;
      VerifyStats verify;
      const ParseFailure parse_failure =
          ParseAndVerifyImage(image.bytes, &parsed, &verify);
      if (parse_failure == ParseFailure::kLegacy) {
        // Unreachable through the index, which never exposes a version-1
        // file. Report a clean miss and leave the file in place.
        Emit(ContinuationDiskEventAction::kMiss,
             ContinuationDiskEventReason::kUnsupported, image.bytes.size(),
             candidate->payload_bytes, candidate->tokens.size());
        return {};
      }
      if (parse_failure != ParseFailure::kNone ||
          parsed.persistence != candidate->persistence ||
          parsed.tokens != candidate->tokens ||
          parsed.payload_bytes != candidate->payload_bytes) {
        const ContinuationDiskEventReason reason =
            parse_failure == ParseFailure::kChecksum
                ? ContinuationDiskEventReason::kChecksumMismatch
                : ContinuationDiskEventReason::kCorrupt;
        const std::size_t file_bytes = candidate->file_bytes;
        const std::size_t payload_bytes = candidate->payload_bytes;
        const std::size_t token_count = candidate->tokens.size();
        const bool removed = RemoveEntry(candidate, reason);
        Emit(ContinuationDiskEventAction::kMiss, reason, file_bytes,
             payload_bytes, token_count);
        if (!removed)
          return {};
        continue;
      }

      const auto runner_started = std::chrono::steady_clock::now();
      try {
        runner.RestorePersistentSnapshot(
            state,
            image.bytes.subspan(parsed.payload_offset, parsed.payload_bytes));
      } catch (...) {
        state.Invalidate();
        const std::size_t file_bytes = candidate->file_bytes;
        const std::size_t payload_bytes = candidate->payload_bytes;
        const std::size_t token_count = candidate->tokens.size();
        (void)RemoveEntry(candidate,
                          ContinuationDiskEventReason::kRestoreFailure);
        Emit(ContinuationDiskEventAction::kMiss,
             ContinuationDiskEventReason::kRestoreFailure, file_bytes,
             payload_bytes, token_count);
        return {};
      }
      const double runner_ms = MillisSince(runner_started);

      TouchEntry(candidate);
      if (TimingEnabled()) {
        const double total_ms = MillisSince(restore_started);
        std::ostringstream line;
        line << "event=disk_restore file_bytes=" << image.bytes.size()
             << " payload_bytes=" << parsed.payload_bytes
             << " token_count=" << candidate->tokens.size()
             << " staged=" << (image.legacy ? "heap" : "mmap")
             << " stage_open_ms=" << std::fixed << std::setprecision(3)
             << image.open_ms << " stage_alloc_ms=" << image.alloc_ms
             << " stage_read_ms=" << image.read_ms
             << " checksum_ms=" << verify.chunks_ms
             << " parse_ms=" << verify.metadata_ms
             << " chunks=" << verify.chunk_count
             << " verify_threads=" << verify.threads
             << " runner_ms=" << runner_ms << " total_ms=" << total_ms;
        EmitTiming(line.str());
      }
      Emit(ContinuationDiskEventAction::kRestored,
           ContinuationDiskEventReason::kHit, candidate->file_bytes,
           candidate->payload_bytes, candidate->tokens.size());
      return {
          .restored = true,
          .token_count = candidate->tokens.size(),
          .file_bytes = candidate->file_bytes,
          .payload_bytes = candidate->payload_bytes,
      };
    }
  }

  void TouchEntry(EntryIterator entry) {
    entry->last_access = std::filesystem::file_time_type::clock::now();
    (void)::utimensat(directory_fd, entry->filename.c_str(), nullptr,
                      AT_SYMLINK_NOFOLLOW);
  }

  [[nodiscard]] std::vector<std::size_t> SharedPrefixBoundaries(
      const TextModelRunner& runner, std::span<const TextRunnerToken> prompt,
      std::size_t min_tokens, std::size_t max_boundaries,
      std::span<const std::uint8_t> input_identity,
      std::span<const ContinuationInputPrefix> input_prefixes) {
    const auto descriptor = DescriptorForInput(runner, input_identity);
    if (!descriptor.persistence.has_value() || max_boundaries == 0) {
      return {};
    }
    std::vector<std::size_t> boundaries;
    VisitInputRanges(
        prompt.size(), input_identity, input_prefixes,
        [&](auto begin, auto end, auto identity) {
          const auto scoped = DescriptorForInput(runner, identity);
          const auto root = prefixes.find(PrefixKey(*scoped.persistence));
          if (root == prefixes.end())
            return;
          for (const auto count :
               root->second.Find(prompt.first(end), true, entries.end())
                   .shared_boundaries) {
            if (count >= begin && count < prompt.size())
              boundaries.push_back(count);
          }
        });
    std::ranges::sort(boundaries);
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                     boundaries.end());
    std::erase_if(boundaries,
                  [min_tokens](auto length) { return length < min_tokens; });
    // Keep the longest ones: they save the most prefill when they hit.
    if (boundaries.size() > max_boundaries) {
      boundaries.erase(
          boundaries.begin(),
          boundaries.end() - static_cast<std::ptrdiff_t>(max_boundaries));
    }
    return boundaries;
  }

  [[nodiscard]] bool WithinCheckpointStep(
      const TextModelRunner& runner, std::span<const TextRunnerToken> tokens,
      std::span<const std::uint8_t> input_identity) {
    if (options.min_checkpoint_step_tokens == 0 ||
        !DescriptorForInput(runner, input_identity).persistence.has_value())
      return false;
    const auto base =
        FindLongestInputCandidate(runner, tokens, input_identity, {});
    return base != entries.end() && base->tokens.size() < tokens.size() &&
           tokens.size() - base->tokens.size() <
               options.min_checkpoint_step_tokens;
  }

  [[nodiscard]] bool Touch(const TextModelRunner& runner,
                           std::span<const TextRunnerToken> tokens,
                           std::span<const std::uint8_t> input_identity) {
    const auto descriptor = DescriptorForInput(runner, input_identity);
    if (!descriptor.persistence.has_value()) {
      return false;
    }
    const std::string digest = HashKey(*descriptor.persistence, tokens);
    const EntryIterator entry =
        FindExact(digest, *descriptor.persistence, tokens);
    if (entry == entries.end()) {
      return false;
    }
    TouchEntry(entry);
    return true;
  }

  /// Startup-index counters, reported once through EmitTiming.
  struct IndexStats {
    std::size_t files_seen{0};
    std::size_t indexed{0};
    std::size_t removed{0};
    std::size_t skipped{0};
    std::size_t bytes_read{0};
    double metadata_ms{0.0};
  };

  ContinuationDiskStoreOptions options;
  EventSink event_sink;
  KeyHashFunction key_hash;
  int directory_fd{-1};
  mutable std::binary_semaphore operation_gate{1};
  std::mutex write_mutex;
  std::list<Entry> entries;
  std::unordered_multimap<std::string, EntryIterator> index;
  std::map<PersistenceKey, PrefixNode> prefixes;
  std::atomic<std::size_t> retained{0};
  std::atomic<std::size_t> retained_entries{0};
  IndexStats index_stats;
  std::random_device random_device;
  std::uint64_t unique_counter{0};
  mutable std::mutex queue_mutex;
  std::condition_variable queue_changed;
  std::deque<PendingSave> pending;
  std::size_t queued_bytes{0};
  std::shared_ptr<std::atomic<std::size_t>> capture_bytes =
      std::make_shared<std::atomic<std::size_t>>(0);
  bool stopping{false};
  std::thread writer;
};

ContinuationDiskStore::ContinuationDiskStore(
    ContinuationDiskStoreOptions options, EventSink event_sink,
    KeyHashFunction key_hash)
    : impl_(std::make_unique<Impl>(std::move(options), std::move(event_sink),
                                   std::move(key_hash))) {}

ContinuationDiskStore::~ContinuationDiskStore() = default;

std::size_t ContinuationDiskStore::FileBytesFor(std::size_t identity_bytes,
                                                std::size_t token_count,
                                                std::size_t payload_bytes) {
  return CheckedFileBytes(identity_bytes, token_count, payload_bytes);
}

ContinuationDiskStore::CaptureReservation::~CaptureReservation() {
  counter_->fetch_sub(bytes_, std::memory_order_relaxed);
}

std::unique_ptr<ContinuationDiskStore::CaptureReservation>
ContinuationDiskStore::ReserveCapture(
    const TextModelRunner& runner, std::size_t token_count,
    std::size_t snapshot_bytes, std::span<const std::uint8_t> input_identity) {
  const auto descriptor = DescriptorForInput(runner, input_identity);
  if (!descriptor.persistence || token_count == 0 || snapshot_bytes == 0)
    return {};
  const auto bytes =
      CheckedFileBytes(descriptor.persistence->compatibility_identity.size(),
                       token_count, snapshot_bytes);
  std::lock_guard lock(impl_->queue_mutex);
  if (!impl_->CanStage(bytes, snapshot_bytes, token_count))
    return {};
  auto reservation = std::unique_ptr<CaptureReservation>(
      new CaptureReservation(impl_->capture_bytes, bytes));
  impl_->capture_bytes->fetch_add(bytes, std::memory_order_relaxed);
  return reservation;
}

bool ContinuationDiskStore::CanSave(
    const TextModelRunner& runner, std::size_t token_count,
    std::size_t snapshot_bytes,
    std::span<const std::uint8_t> input_identity) const {
  const auto descriptor = DescriptorForInput(runner, input_identity);
  if (!descriptor.persistence || token_count == 0 || snapshot_bytes == 0) {
    return false;
  }
  const auto bytes =
      CheckedFileBytes(descriptor.persistence->compatibility_identity.size(),
                       token_count, snapshot_bytes);
  std::lock_guard lock(impl_->queue_mutex);
  return impl_->CanStage(bytes, snapshot_bytes, token_count);
}

std::size_t ContinuationDiskStore::SaveAsync(
    std::shared_ptr<const TextModelRunner> runner,
    std::vector<TextRunnerToken> checkpoint_tokens,
    std::shared_ptr<const TextRunnerSnapshot> snapshot,
    std::vector<std::uint8_t> input_identity,
    std::unique_ptr<CaptureReservation> reservation, bool shared_prefix) {
  if (!runner || !snapshot || checkpoint_tokens.empty())
    return 0;
  const auto descriptor = DescriptorForInput(*runner, input_identity);
  if (!descriptor.persistence)
    return 0;
  const auto file_bytes =
      CheckedFileBytes(descriptor.persistence->compatibility_identity.size(),
                       checkpoint_tokens.size(),
                       runner->PersistentSnapshotPayloadBytes(*snapshot));
  const auto charge = CheckedFileBytes(
      descriptor.persistence->compatibility_identity.size(),
      checkpoint_tokens.size(),
      std::max(snapshot->PayloadBytes(),
               runner->PersistentSnapshotPayloadBytes(*snapshot)));
  {
    std::lock_guard lock(impl_->queue_mutex);
    if (reservation) {
      if (reservation->counter_ != impl_->capture_bytes)
        return 0;
      reservation.reset();
    }
    if (!impl_->CanStage(charge, snapshot->PayloadBytes(),
                         checkpoint_tokens.size()))
      return 0;
    impl_->pending.push_back({std::move(runner), std::move(checkpoint_tokens),
                              std::move(snapshot), std::move(input_identity),
                              charge, shared_prefix});
    impl_->queued_bytes += charge;
    // E2B/E2C: the queued tokens already teach the shared boundary, so
    // publish them to the prefix tree now, marked pending (entries.end()).
    // A real entry for the same tokens is never replaced by the mark; the
    // writer replaces the mark on success, erases it on failure, and
    // publishes it itself when a reader held the gate here.
    {
      ScopedOperationPermit pending_mark(impl_->operation_gate, false);
      if (pending_mark) {
        impl_->prefixes[Impl::PrefixKey(*descriptor.persistence)].Insert(
            impl_->pending.back().tokens, impl_->entries.end(),
            /*overwrite=*/false);
        impl_->pending.back().mark_published = true;
      }
    }
  }
  impl_->queue_changed.notify_one();
  return file_bytes;
}

void ContinuationDiskStore::Flush() {
  std::unique_lock lock(impl_->queue_mutex);
  impl_->queue_changed.wait(lock, [this] { return impl_->queued_bytes == 0; });
}

ContinuationDiskStore::SaveResult ContinuationDiskStore::Save(
    const TextModelRunner& runner,
    std::span<const TextRunnerToken> checkpoint_tokens,
    const TextRunnerSnapshot& snapshot,
    std::span<const std::uint8_t> input_identity) {
  return impl_->Save(runner, checkpoint_tokens, snapshot, input_identity);
}

ContinuationDiskStore::RestoreResult
ContinuationDiskStore::RestoreLongestPrefix(
    const TextModelRunner& runner, TextRunnerState& state,
    std::span<const TextRunnerToken> prompt,
    std::span<const std::uint8_t> input_identity,
    std::size_t stable_prefix_tokens,
    std::span<const ContinuationInputPrefix> input_prefixes) {
  if (stable_prefix_tokens > prompt.size())
    throw std::invalid_argument("stable cache prefix exceeds prompt length");
  const ScopedOperationPermit permit(impl_->operation_gate, false);
  if (!permit) {
    impl_->Emit(ContinuationDiskEventAction::kMiss,
                ContinuationDiskEventReason::kBusy, 0, 0, prompt.size());
    return {};
  }
  return impl_->RestoreLongestPrefix(runner, state, prompt, input_identity,
                                     stable_prefix_tokens, input_prefixes);
}

std::vector<std::size_t> ContinuationDiskStore::SharedPrefixBoundaries(
    const TextModelRunner& runner, std::span<const TextRunnerToken> prompt,
    std::size_t min_tokens, std::size_t max_boundaries,
    std::span<const std::uint8_t> input_identity,
    std::span<const ContinuationInputPrefix> input_prefixes) {
  const ScopedOperationPermit permit(impl_->operation_gate, false);
  if (!permit)
    return {};
  return impl_->SharedPrefixBoundaries(runner, prompt, min_tokens,
                                       max_boundaries, input_identity,
                                       input_prefixes);
}

bool ContinuationDiskStore::Touch(
    const TextModelRunner& runner, std::span<const TextRunnerToken> tokens,
    std::span<const std::uint8_t> input_identity) {
  const ScopedOperationPermit permit(impl_->operation_gate, false);
  if (!permit)
    return false;
  return impl_->Touch(runner, tokens, input_identity);
}

bool ContinuationDiskStore::WithinCheckpointStep(
    const TextModelRunner& runner, std::span<const TextRunnerToken> tokens,
    std::span<const std::uint8_t> input_identity) {
  const ScopedOperationPermit permit(impl_->operation_gate, false);
  if (!permit)
    return false;
  return impl_->WithinCheckpointStep(runner, tokens, input_identity);
}

std::size_t ContinuationDiskStore::entry_count() const noexcept {
  return impl_->retained_entries.load(std::memory_order_relaxed);
}

std::size_t ContinuationDiskStore::retained_bytes() const noexcept {
  return impl_->retained.load(std::memory_order_relaxed);
}

std::size_t ContinuationDiskStore::capacity_bytes() const noexcept {
  return impl_->options.capacity_bytes;
}

std::size_t ContinuationDiskStore::staging_capacity_bytes() const noexcept {
  return impl_->options.staging_capacity_bytes;
}

}  // namespace superroc::server
