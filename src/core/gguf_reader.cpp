#include "src/core/gguf_reader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "src/core/model_family.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::core {

namespace {

constexpr std::array<char, 4> kGgufMagic = {'G', 'G', 'U', 'F'};

template<typename T>
bool ReadPod(const std::uint8_t* data, std::size_t size, std::size_t& offset,
             T& out) {
  if (offset > size || sizeof(T) > size - offset) {
    return false;
  }
  std::memcpy(&out, data + offset, sizeof(T));
  offset += sizeof(T);
  return true;
}

bool ReadString(const std::uint8_t* data, std::size_t size, std::size_t& offset,
                std::string_view& out) {
  std::uint64_t len = 0;
  if (!ReadPod(data, size, offset, len)) {
    return false;
  }
  if (offset > size || len > size - offset) {
    return false;
  }
  out = std::string_view(reinterpret_cast<const char*>(data + offset), len);
  offset += len;
  return true;
}

}  // namespace

GgufReader::~GgufReader() {
  if (owns_mmap_ && mmap_addr_ != nullptr && size_ > 0) {
    munmap(mmap_addr_, size_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
}

GgufReader::GgufReader(GgufReader&& other) noexcept
    : data_(other.data_),
      mmap_addr_(other.mmap_addr_),
      size_(other.size_),
      fd_(other.fd_),
      owns_mmap_(other.owns_mmap_),
      version_(other.version_),
      alignment_(other.alignment_),
      metadata_(std::move(other.metadata_)),
      tensors_(std::move(other.tensors_)),
      tensor_index_(std::move(other.tensor_index_)),
      mapped_regions_(std::move(other.mapped_regions_)),
      shards_(std::move(other.shards_)) {
  other.data_ = nullptr;
  other.mmap_addr_ = nullptr;
  other.size_ = 0;
  other.fd_ = -1;
  other.owns_mmap_ = false;
}

GgufReader& GgufReader::operator=(GgufReader&& other) noexcept {
  if (this != &other) {
    if (owns_mmap_ && mmap_addr_ != nullptr && size_ > 0) {
      munmap(mmap_addr_, size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }

    data_ = other.data_;
    mmap_addr_ = other.mmap_addr_;
    size_ = other.size_;
    fd_ = other.fd_;
    owns_mmap_ = other.owns_mmap_;
    version_ = other.version_;
    alignment_ = other.alignment_;
    metadata_ = std::move(other.metadata_);
    tensors_ = std::move(other.tensors_);
    tensor_index_ = std::move(other.tensor_index_);
    mapped_regions_ = std::move(other.mapped_regions_);
    shards_ = std::move(other.shards_);

    other.data_ = nullptr;
    other.mmap_addr_ = nullptr;
    other.size_ = 0;
    other.fd_ = -1;
    other.owns_mmap_ = false;
  }
  return *this;
}

std::unique_ptr<GgufReader> GgufReader::OpenFile(
    const std::filesystem::path& path, std::string* error_msg) {
  auto first = OpenSingleFile(path, error_msg);
  if (!first) {
    return nullptr;
  }

  const auto split_count = first->GetMetadataUint32("split.count").value_or(1U);
  if (first->FindMetadata("split.count") &&
      (!first->GetMetadataUint32("split.count") || split_count == 0 ||
       split_count > 65535)) {
    if (error_msg)
      *error_msg = "Invalid GGUF split count";
    return nullptr;
  }
  if (split_count <= 1) {
    return first;
  }
  return OpenSplitFileSet(path, std::move(first), error_msg);
}

std::unique_ptr<GgufReader> GgufReader::OpenSingleFile(
    const std::filesystem::path& path, std::string* error_msg) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    if (error_msg != nullptr) {
      *error_msg = "Failed to open file: " + path.string();
    }
    return nullptr;
  }

  struct stat sb{};
  if (fstat(fd, &sb) != 0 || sb.st_size <= 0) {
    close(fd);
    if (error_msg != nullptr) {
      *error_msg = "Invalid or empty file: " + path.string();
    }
    return nullptr;
  }

  const auto size = static_cast<std::size_t>(sb.st_size);
  void* const addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (addr == MAP_FAILED) {
    close(fd);
    if (error_msg != nullptr) {
      *error_msg = "Failed to mmap file: " + path.string();
    }
    return nullptr;
  }
  (void)madvise(addr, size, MADV_SEQUENTIAL);

  auto reader = std::unique_ptr<GgufReader>(new GgufReader());
  reader->mmap_addr_ = addr;
  reader->data_ = static_cast<const std::uint8_t*>(addr);
  reader->size_ = size;
  reader->fd_ = fd;
  reader->owns_mmap_ = true;

  try {
    if (!reader->ParseHeaders(error_msg)) {
      if (error_msg && error_msg->empty())
        *error_msg = "Malformed or truncated GGUF";
      return nullptr;
    }
  } catch (const std::bad_alloc&) {
    if (error_msg)
      *error_msg = "Insufficient memory for GGUF descriptors";
    return nullptr;
  }
  reader->mapped_regions_.push_back(
      {reader->data_, reader->size_, reader->fd_});
  return reader;
}

std::unique_ptr<GgufReader> GgufReader::OpenSplitFileSet(
    const std::filesystem::path& path, std::unique_ptr<GgufReader> first,
    std::string* error_msg) {
  const auto split_count = first->GetMetadataUint32("split.count").value_or(1U);
  const auto split_no = first->GetMetadataUint32("split.no");
  const auto total_tensor_count =
      first->GetMetadataUint64("split.tensors.count");
  if (!split_no.has_value() || *split_no >= split_count ||
      !total_tensor_count.has_value()) {
    if (error_msg != nullptr) {
      *error_msg = "Invalid split GGUF metadata in " + path.string();
    }
    return nullptr;
  }

  const std::string filename = path.filename().string();
  const std::regex split_pattern{R"(^(.*)-([0-9]+)-of-([0-9]+)\.gguf$)"};
  std::smatch match;
  if (!std::regex_match(filename, match, split_pattern)) {
    if (error_msg != nullptr) {
      *error_msg =
          "Split GGUF filename does not match "
          "<name>-00001-of-00002.gguf: " +
          filename;
    }
    return nullptr;
  }

  const std::string prefix = match[1].str();
  const std::size_t index_width = match[2].str().size();
  const std::size_t count_width = match[3].str().size();
  const auto build_path = [&](std::uint32_t index) {
    std::ostringstream name;
    name << prefix << '-' << std::setfill('0')
         << std::setw(static_cast<int>(index_width)) << (index + 1) << "-of-"
         << std::setw(static_cast<int>(count_width)) << split_count << ".gguf";
    return path.parent_path() / name.str();
  };

  std::vector<std::unique_ptr<GgufReader>> shards(split_count);
  shards[*split_no] = std::move(first);
  for (std::uint32_t i = 0; i < split_count; ++i) {
    if (shards[i]) {
      continue;
    }
    const auto shard_path = build_path(i);
    shards[i] = OpenSingleFile(shard_path, error_msg);
    if (!shards[i]) {
      if (error_msg != nullptr && error_msg->empty()) {
        *error_msg = "Failed to open split GGUF shard: " + shard_path.string();
      }
      return nullptr;
    }
  }

  const auto architecture =
      shards[0]->GetMetadataString("general.architecture");
  for (std::uint32_t i = 0; i < split_count; ++i) {
    const auto shard_no = shards[i]->GetMetadataUint32("split.no");
    const auto shard_count = shards[i]->GetMetadataUint32("split.count");
    const auto shard_architecture =
        shards[i]->GetMetadataString("general.architecture");
    if (!shard_no.has_value() || *shard_no != i || !shard_count.has_value() ||
        *shard_count != split_count ||
        shards[i]->GetVersion() != shards[0]->GetVersion() ||
        (shard_architecture.has_value() &&
         shard_architecture != architecture)) {
      if (error_msg != nullptr) {
        *error_msg =
            "Inconsistent split GGUF shard metadata: " + build_path(i).string();
      }
      return nullptr;
    }
  }

  auto combined = std::unique_ptr<GgufReader>(new GgufReader());
  combined->version_ = shards[0]->version_;
  combined->alignment_ = shards[0]->alignment_;
  combined->metadata_ = shards[0]->metadata_;
  std::size_t actual_count = 0;
  for (const auto& shard : shards) {
    if (shard->tensors_.size() >
            std::numeric_limits<std::size_t>::max() - actual_count ||
        shard->size_ >
            std::numeric_limits<std::size_t>::max() - combined->size_)
      return nullptr;
    actual_count += shard->tensors_.size();
    combined->size_ += shard->size_;
  }
  if (actual_count != *total_tensor_count) {
    if (error_msg)
      *error_msg = "Split GGUF tensor count mismatch";
    return nullptr;
  }
  combined->tensors_.reserve(actual_count);
  combined->tensor_index_.reserve(
      static_cast<std::size_t>(*total_tensor_count));
  combined->mapped_regions_.reserve(split_count);

  for (const auto& shard : shards) {
    combined->mapped_regions_.push_back(
        {shard->data_, shard->size_, shard->fd_});
    for (const auto& tensor : shard->tensors_) {
      if (combined->tensor_index_.contains(tensor.name)) {
        if (error_msg != nullptr) {
          *error_msg = "Duplicate tensor across split GGUF shards: " +
                       std::string(tensor.name);
        }
        return nullptr;
      }
      combined->tensor_index_[tensor.name] = combined->tensors_.size();
      combined->tensors_.push_back(tensor);
    }
  }

  if (combined->tensors_.size() != *total_tensor_count) {
    if (error_msg != nullptr) {
      *error_msg = "Split GGUF tensor count mismatch: expected " +
                   std::to_string(*total_tensor_count) + ", found " +
                   std::to_string(combined->tensors_.size());
    }
    return nullptr;
  }

  combined->shards_ = std::move(shards);
  return combined;
}

std::unique_ptr<GgufReader> GgufReader::OpenMemory(const void* data,
                                                   std::size_t size,
                                                   std::string* error_msg) {
  if (data == nullptr || size < 24) {
    if (error_msg != nullptr) {
      *error_msg = "Invalid memory buffer for GGUF";
    }
    return nullptr;
  }

  auto reader = std::unique_ptr<GgufReader>(new GgufReader());
  reader->mmap_addr_ = nullptr;
  reader->data_ = static_cast<const std::uint8_t*>(data);
  reader->size_ = size;
  reader->fd_ = -1;
  reader->owns_mmap_ = false;

  try {
    if (!reader->ParseHeaders(error_msg)) {
      if (error_msg && error_msg->empty())
        *error_msg = "Malformed or truncated GGUF";
      return nullptr;
    }
  } catch (const std::bad_alloc&) {
    if (error_msg)
      *error_msg = "Insufficient memory for GGUF descriptors";
    return nullptr;
  }
  reader->mapped_regions_.push_back(
      {reader->data_, reader->size_, reader->fd_});
  return reader;
}

bool GgufReader::ParseHeaders(std::string* error_msg) {
  if (size_ < 24) {
    if (error_msg != nullptr) {
      *error_msg = "File too small for GGUF header";
    }
    return false;
  }

  if (std::memcmp(data_, kGgufMagic.data(), 4) != 0) {
    if (error_msg != nullptr) {
      *error_msg = "Invalid GGUF magic header";
    }
    return false;
  }

  std::size_t offset = 4;
  if (!ReadPod(data_, size_, offset, version_)) {
    if (error_msg != nullptr) {
      *error_msg = "Failed to read GGUF version";
    }
    return false;
  }

  if (version_ != 2 && version_ != 3) {
    if (error_msg != nullptr) {
      *error_msg = "Unsupported GGUF version: " + std::to_string(version_);
    }
    return false;
  }

  std::uint64_t tensor_count = 0;
  std::uint64_t metadata_count = 0;
  if (!ReadPod(data_, size_, offset, tensor_count) ||
      !ReadPod(data_, size_, offset, metadata_count)) {
    if (error_msg != nullptr) {
      *error_msg = "Failed to read tensor/metadata counts";
    }
    return false;
  }

  // Bound host-side descriptor allocations independently of mapped weight size.
  std::size_t budget = 128 * 1024 * 1024;
  const auto charge = [&](std::uint64_t count, std::size_t item_bytes) {
    if (count > budget / item_bytes)
      return false;
    budget -= static_cast<std::size_t>(count) * item_bytes;
    return true;
  };
  if (metadata_count > (size_ - offset) / 13 ||
      tensor_count > (size_ - offset) / 32 ||
      !charge(metadata_count, sizeof(GgufMetadataValue) + 64) ||
      !charge(tensor_count, sizeof(GgufTensorInfo) + 128)) {
    if (error_msg)
      *error_msg = "GGUF descriptor counts exceed file or memory limits";
    return false;
  }
  metadata_.reserve(metadata_count);

  // Parse metadata key-value pairs
  for (std::uint64_t i = 0; i < metadata_count; ++i) {
    std::string_view key;
    if (!ReadString(data_, size_, offset, key)) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to read metadata key";
      }
      return false;
    }

    if (key.empty() || key.size() > 65535 || !charge(key.size(), 1)) {
      if (error_msg)
        *error_msg = "GGUF metadata key exceeds limits";
      return false;
    }
    std::uint32_t val_type_raw = 0;
    if (!ReadPod(data_, size_, offset, val_type_raw)) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to read metadata type for key " + std::string(key);
      }
      return false;
    }

    if (val_type_raw > static_cast<std::uint32_t>(GgufValueType::kFloat64))
      return false;
    const auto val_type = static_cast<GgufValueType>(val_type_raw);
    GgufMetadataValue meta_val;
    meta_val.type = val_type;

    switch (val_type) {
      case GgufValueType::kUint8: {
        std::uint8_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::uint64_t>(v);
        break;
      }
      case GgufValueType::kInt8: {
        std::int8_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::int64_t>(v);
        break;
      }
      case GgufValueType::kUint16: {
        std::uint16_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::uint64_t>(v);
        break;
      }
      case GgufValueType::kInt16: {
        std::int16_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::int64_t>(v);
        break;
      }
      case GgufValueType::kUint32: {
        std::uint32_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::uint64_t>(v);
        break;
      }
      case GgufValueType::kInt32: {
        std::int32_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::int64_t>(v);
        break;
      }
      case GgufValueType::kFloat32: {
        float v = 0.0F;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<double>(v);
        break;
      }
      case GgufValueType::kBool: {
        std::uint8_t v = 0;
        if (!ReadPod(data_, size_, offset, v) || v > 1) {
          return false;
        }
        meta_val.value = (v != 0);
        break;
      }
      case GgufValueType::kString: {
        std::string_view s;
        if (!ReadString(data_, size_, offset, s)) {
          return false;
        }
        if (!charge(s.size(), 1))
          return false;
        meta_val.value = s;
        break;
      }
      case GgufValueType::kUint64: {
        std::uint64_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = v;
        break;
      }
      case GgufValueType::kInt64: {
        std::int64_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = v;
        break;
      }
      case GgufValueType::kFloat64: {
        double v = 0.0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = v;
        break;
      }
      case GgufValueType::kArray: {
        std::uint32_t item_type_raw = 0;
        std::uint64_t array_len = 0;
        if (!ReadPod(data_, size_, offset, item_type_raw) ||
            !ReadPod(data_, size_, offset, array_len)) {
          return false;
        }
        if (item_type_raw >
                static_cast<std::uint32_t>(GgufValueType::kFloat64) ||
            item_type_raw == static_cast<std::uint32_t>(GgufValueType::kArray))
          return false;
        constexpr std::size_t item_bytes[] = {1, 1, 2, 2, 4, 4, 4,
                                              1, 8, 0, 8, 8, 8};
        if (offset > size_ ||
            array_len > (size_ - offset) / item_bytes[item_type_raw] ||
            !charge(array_len,
                    item_type_raw == 8 ? sizeof(std::string_view) : 8))
          return false;
        const auto item_type = static_cast<GgufValueType>(item_type_raw);
        if (item_type == GgufValueType::kString) {
          std::vector<std::string_view> str_arr;
          str_arr.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            std::string_view s;
            if (!ReadString(data_, size_, offset, s)) {
              return false;
            }
            if (!charge(s.size(), 1))
              return false;
            str_arr.push_back(s);
          }
          meta_val.value = std::move(str_arr);
        } else if (item_type == GgufValueType::kFloat32 ||
                   item_type == GgufValueType::kFloat64) {
          const auto bytes = item_type == GgufValueType::kFloat32 ? 4U : 8U;
          if (offset > size_ || array_len > (size_ - offset) / bytes)
            return false;
          std::vector<double> values;
          values.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            double value = 0;
            if (item_type == GgufValueType::kFloat32) {
              float scalar = 0;
              if (!ReadPod(data_, size_, offset, scalar))
                return false;
              value = scalar;
            } else if (!ReadPod(data_, size_, offset, value)) {
              return false;
            }
            values.push_back(value);
          }
          meta_val.value = std::move(values);
        } else if (item_type == GgufValueType::kUint8 ||
                   item_type == GgufValueType::kUint16 ||
                   item_type == GgufValueType::kUint32 ||
                   item_type == GgufValueType::kBool ||
                   item_type == GgufValueType::kUint64) {
          // Writers pick the narrowest integer type; widen every width.
          std::vector<std::uint64_t> u_arr;
          u_arr.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            std::uint64_t val = 0;
            if (item_type == GgufValueType::kUint8) {
              std::uint8_t val8 = 0;
              if (!ReadPod(data_, size_, offset, val8))
                return false;
              val = val8;
            } else if (item_type == GgufValueType::kUint16) {
              std::uint16_t val16 = 0;
              if (!ReadPod(data_, size_, offset, val16))
                return false;
              val = val16;
            } else if (item_type == GgufValueType::kUint32) {
              std::uint32_t val32 = 0;
              if (!ReadPod(data_, size_, offset, val32)) {
                return false;
              }
              val = val32;
            } else if (item_type == GgufValueType::kBool) {
              std::uint8_t boolean = 0;
              if (!ReadPod(data_, size_, offset, boolean) || boolean > 1)
                return false;
              val = boolean;
            } else {
              if (!ReadPod(data_, size_, offset, val)) {
                return false;
              }
            }
            u_arr.push_back(val);
          }
          meta_val.value = std::move(u_arr);
        } else if (item_type == GgufValueType::kInt8 ||
                   item_type == GgufValueType::kInt16 ||
                   item_type == GgufValueType::kInt32 ||
                   item_type == GgufValueType::kInt64) {
          std::vector<std::int64_t> i_arr;
          i_arr.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            std::int64_t val = 0;
            if (item_type == GgufValueType::kInt8) {
              std::int8_t val8 = 0;
              if (!ReadPod(data_, size_, offset, val8))
                return false;
              val = val8;
            } else if (item_type == GgufValueType::kInt16) {
              std::int16_t val16 = 0;
              if (!ReadPod(data_, size_, offset, val16))
                return false;
              val = val16;
            } else if (item_type == GgufValueType::kInt32) {
              std::int32_t val32 = 0;
              if (!ReadPod(data_, size_, offset, val32)) {
                return false;
              }
              val = val32;
            } else if (!ReadPod(data_, size_, offset, val)) {
              return false;
            }
            i_arr.push_back(val);
          }
          meta_val.value = std::move(i_arr);
        } else {
          // Every scalar item type is decoded above; never store an array
          // entry without its values.
          return false;
        }
        break;
      }
    }

    if (key == "general.alignment") {
      const auto* value = std::get_if<std::uint64_t>(&meta_val.value);
      if (!value || !std::has_single_bit(*value) || *value > UINT32_MAX) {
        if (error_msg)
          *error_msg = "Invalid GGUF alignment";
        return false;
      }
      alignment_ = *value;
    }
    if (!metadata_.emplace(key, std::move(meta_val)).second) {
      if (error_msg)
        *error_msg = "Duplicate GGUF metadata key";
      return false;
    }
  }

  // Parse tensor infos
  tensors_.reserve(tensor_count);
  tensor_index_.reserve(tensor_count);

  for (std::uint64_t i = 0; i < tensor_count; ++i) {
    GgufTensorInfo info;
    if (!ReadString(data_, size_, offset, info.name)) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to read tensor name";
      }
      return false;
    }

    if (info.name.empty() || info.name.size() > 65535 ||
        !charge(info.name.size(), 1))
      return false;
    std::uint32_t n_dims = 0;
    if (!ReadPod(data_, size_, offset, n_dims) || n_dims == 0 || n_dims > 8) {
      if (error_msg != nullptr) {
        *error_msg = "Invalid tensor dimensions count";
      }
      return false;
    }

    info.dimensions.resize(n_dims);
    for (std::uint32_t d = 0; d < n_dims; ++d) {
      if (!ReadPod(data_, size_, offset, info.dimensions[d])) {
        return false;
      }
    }

    std::uint32_t type_raw = 0;
    if (!ReadPod(data_, size_, offset, type_raw)) {
      return false;
    }
    if (type_raw > UINT16_MAX)
      return false;
    info.type = static_cast<GgmlType>(type_raw);
    const auto elements = info.ElementCount();
    const auto block = quant::QuantizedBlockElements(info.type);
    info.size_bytes = quant::EncodedSizeBytes(info.type, elements);
    if (!info.size_bytes || (block && info.dimensions.front() % block != 0)) {
      if (error_msg)
        *error_msg = "Invalid GGUF tensor shape or unsupported storage type: " +
                     std::string(info.name);
      return false;
    }

    if (!ReadPod(data_, size_, offset, info.offset)) {
      return false;
    }

    if (!tensor_index_.emplace(info.name, tensors_.size()).second) {
      if (error_msg)
        *error_msg = "Duplicate GGUF tensor name";
      return false;
    }
    tensors_.push_back(std::move(info));
  }

  // Align data payload base to alignment boundary
  const auto padding = (alignment_ - offset % alignment_) % alignment_;
  if (offset > size_ || padding > size_ - offset)
    return false;
  const std::size_t data_base = offset + padding;
  for (auto& tensor : tensors_) {
    if (tensor.offset % alignment_ != 0 || tensor.offset > size_ - data_base ||
        tensor.size_bytes > size_ - data_base - tensor.offset) {
      if (error_msg != nullptr) {
        *error_msg = "Tensor extent or alignment is invalid: " +
                     std::string(tensor.name);
      }
      return false;
    }
    tensor.data = data_ + data_base + tensor.offset;
  }

  return true;
}

const GgufMetadataValue* GgufReader::FindMetadata(
    std::string_view key) const noexcept {
  auto it = metadata_.find(key);
  if (it != metadata_.end()) {
    return &it->second;
  }
  return nullptr;
}

std::optional<std::string_view> GgufReader::GetMetadataString(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr &&
      std::holds_alternative<std::string_view>(meta->value)) {
    return std::get<std::string_view>(meta->value);
  }
  return std::nullopt;
}

std::optional<std::uint32_t> GgufReader::GetMetadataUint32(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta == nullptr) {
    return std::nullopt;
  }
  if (const auto* value = std::get_if<std::uint64_t>(&meta->value)) {
    if (std::in_range<std::uint32_t>(*value)) {
      return static_cast<std::uint32_t>(*value);
    }
  } else if (const auto* value = std::get_if<std::int64_t>(&meta->value)) {
    if (std::in_range<std::uint32_t>(*value)) {
      return static_cast<std::uint32_t>(*value);
    }
  }
  return std::nullopt;
}

std::optional<std::uint64_t> GgufReader::GetMetadataUint64(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta == nullptr) {
    return std::nullopt;
  }
  if (const auto* value = std::get_if<std::uint64_t>(&meta->value)) {
    return *value;
  }
  if (const auto* value = std::get_if<std::int64_t>(&meta->value)) {
    if (std::in_range<std::uint64_t>(*value)) {
      return static_cast<std::uint64_t>(*value);
    }
  }
  return std::nullopt;
}

std::optional<float> GgufReader::GetMetadataFloat32(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr && std::holds_alternative<double>(meta->value)) {
    return static_cast<float>(std::get<double>(meta->value));
  }
  return std::nullopt;
}

std::optional<bool> GgufReader::GetMetadataBool(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr && std::holds_alternative<bool>(meta->value)) {
    return std::get<bool>(meta->value);
  }
  return std::nullopt;
}

std::vector<std::string_view> GgufReader::GetMetadataStringArray(
    std::string_view key) const {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr &&
      std::holds_alternative<std::vector<std::string_view>>(meta->value)) {
    return std::get<std::vector<std::string_view>>(meta->value);
  }
  return {};
}

std::vector<std::uint64_t> GgufReader::GetMetadataUintArray(
    std::string_view key) const {
  const auto* meta = FindMetadata(key);
  if (meta == nullptr) {
    return {};
  }
  if (const auto* values =
          std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
    return *values;
  }
  if (const auto* values =
          std::get_if<std::vector<std::int64_t>>(&meta->value)) {
    std::vector<std::uint64_t> out;
    out.reserve(values->size());
    for (const std::int64_t value : *values) {
      if (value < 0) {
        return {};
      }
      out.push_back(static_cast<std::uint64_t>(value));
    }
    return out;
  }
  return {};
}

const GgufTensorInfo* GgufReader::FindTensor(
    std::string_view name) const noexcept {
  auto it = tensor_index_.find(name);
  if (it != tensor_index_.end()) {
    return &tensors_[it->second];
  }
  return nullptr;
}

bool GgufReader::HasTensor(std::string_view name) const noexcept {
  return tensor_index_.contains(name);
}

std::string GgufReader::GetQuantizationLabel() const {
  // `general.file_type` is deliberately not consulted. The Unsloth UD-*_K_XL
  // artifacts this repo targets carry a stale one -- UD-Q8_K_XL records ftype
  // 15 (Q4_K_M), which is what llama-bench prints for it -- so the recorded
  // value is not evidence about the weights. The tensor table always is.
  //
  // Weighted by element count, not by tensor count: an artifact carries
  // hundreds of tiny F32 norms that would otherwise outvote every projection.
  // Element count also avoids needing each format's byte geometry here, which
  // lives in quant::QuantizedRowBytes and would invert this file's dependency
  // direction.
  std::unordered_map<std::uint16_t, std::uint64_t> elements_by_type;
  std::uint64_t total_elements = 0;
  for (const auto& tensor : tensors_) {
    const std::uint64_t count = tensor.ElementCount();
    elements_by_type[static_cast<std::uint16_t>(tensor.type)] += count;
    total_elements += count;
  }
  if (elements_by_type.empty() || total_elements == 0) {
    return "unknown";
  }
  const auto dominant = std::ranges::max_element(
      elements_by_type, {}, [](const auto& entry) { return entry.second; });
  std::string label(ToString(static_cast<GgmlType>(dominant->first)));
  // The UD-*_K_XL targets spread seven formats across the projections, so
  // naming only the largest would read as a claim about the whole file. Four
  // fifths is the line between "this artifact is that format" and "this is a
  // mix".
  if (dominant->second * 5 < total_elements * 4) {
    label += " mixed";
  }
  return label;
}

bool GgufReader::HasMtpTensors() const noexcept {
  return std::ranges::any_of(tensors_, [](const auto& t) {
    return t.name.starts_with("mtp.") ||
           t.name.find(".mtp.") != std::string_view::npos ||
           t.name.find(".nextn.") != std::string_view::npos;
  });
}

bool GgufReader::HasVisionTensors() const noexcept {
  return std::ranges::any_of(tensors_, [](const auto& t) {
    return t.name.starts_with("model.visual") ||
           t.name.starts_with("visual.") || t.name.starts_with("v.");
  });
}

std::optional<ModelConfig> GgufReader::ExtractModelConfig(
    std::string* error_msg) const {
  ModelConfig config;

  // 1. Check architecture prefix (e.g. "qwen2", "qwen3", "qwen35", "llama")
  auto arch_str = GetMetadataString("general.architecture").value_or("qwen");
  config.architecture = std::string(arch_str);
  const std::string prefix = config.architecture + ".";

  // 2. Read standard architectural hyperparameters
  const auto total_layers =
      GetMetadataUint32(prefix + "block_count").value_or(config.num_layers);
  if (auto val = GetMetadataUint32(prefix + "embedding_length")) {
    config.hidden_size = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "feed_forward_length")) {
    config.intermediate_size = *val;
  }
  // KAT: qwen35moe expert metadata. Dense qwen35 leaves these at 0.
  if (auto val = GetMetadataUint32(prefix + "expert_count")) {
    config.expert_count = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "expert_used_count")) {
    config.expert_used_count = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "expert_feed_forward_length")) {
    config.expert_feed_forward_length = *val;
  }
  if (auto val =
          GetMetadataUint32(prefix + "expert_shared_feed_forward_length")) {
    config.expert_shared_feed_forward_length = *val;
  }
  config.is_moe = config.expert_count > 0;
  if (auto val = GetMetadataUint32(prefix + "attention.head_count")) {
    config.num_attention_heads = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "attention.head_count_kv")) {
    config.num_key_value_heads = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "attention.key_length")) {
    config.head_dim = *val;
  } else if (config.num_attention_heads > 0) {
    config.head_dim = config.hidden_size / config.num_attention_heads;
  }
  if (auto val = GetMetadataUint32(prefix + "context_length")) {
    config.context_length = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "full_attention_interval")) {
    config.full_attention_interval = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "ssm.conv_kernel")) {
    config.ssm_conv_kernel = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "ssm.state_size")) {
    config.ssm_state_size = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "ssm.group_count")) {
    config.ssm_group_count = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "ssm.time_step_rank")) {
    config.ssm_time_step_rank = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "ssm.inner_size")) {
    config.ssm_inner_size = *val;
  }
  if (auto val = GetMetadataFloat32(prefix + "rope.freq_base")) {
    config.rope_theta = *val;
  }
  if (auto val = GetMetadataUint32(prefix + "rope.dimension_count")) {
    config.rotary_dim = *val;
  } else {
    config.rotary_dim = config.head_dim;
  }

  // 3. MTP speculative layers are included in qwen35.block_count but are not
  // part of the main autoregressive transformer stack.
  const auto embedded_mtp_layers =
      GetMetadataUint32(prefix + "nextn_predict_layers");
  config.mtp_num_layers =
      embedded_mtp_layers.value_or(HasMtpTensors() ? 1U : 0U);
  const std::uint32_t layers_in_block_count = embedded_mtp_layers.value_or(0U);
  if (layers_in_block_count > total_layers) {
    if (error_msg != nullptr) {
      *error_msg = "MTP layer count exceeds qwen block count";
    }
    return std::nullopt;
  }
  config.num_layers = total_layers - layers_in_block_count;

  // 4. Assert text-only contract
  if (HasVisionTensors()) {
    config.is_text_only = false;
    if (error_msg != nullptr) {
      *error_msg =
          "Model contains vision encoder tensors; text-only contract violation";
    }
    return std::nullopt;
  }
  config.is_text_only = true;

  // 5. Model name
  if (auto name = GetMetadataString("general.name")) {
    config.model_name = std::string(*name);
  }
  if (const auto* token_embd = FindTensor("token_embd.weight");
      token_embd != nullptr && config.hidden_size > 0) {
    const auto elements = token_embd->ElementCount();
    if ((elements % config.hidden_size) == 0) {
      config.vocab_size =
          static_cast<std::uint32_t>(elements / config.hidden_size);
    }
  }

  // GLIMMER: muse-glimmer is an attention-only chassis on the qwen config.
  // Interval 1 makes FullAttentionLayerCount() == num_layers and the existing
  // decode index `l / interval` equal the layer index. SSM scalars stay at
  // their defaults so SsmValueSize() does not divide by zero; SsmLayerCount()
  // is 0 because every layer is attention.
  if (gufo::core::IsModelFamily(config.architecture,
                                gufo::core::ModelFamilyId::kMuseGlimmer)) {
    if (HasMtpTensors() || embedded_mtp_layers.value_or(0U) != 0U) {
      if (error_msg != nullptr) {
        *error_msg = "muse-glimmer does not support MTP";
      }
      return std::nullopt;
    }
    config.num_layers = total_layers;
    config.mtp_num_layers = 0;
    config.attention_only = true;
    config.attn_gate_mode = AttnGateMode::kSeparate;
    config.rope_variant = RopeVariant::kNorm;
    config.full_attention_interval = 1;
    config.norm_eps_post = 1.0e-8F;

    const auto eps =
        GetMetadataFloat32(prefix + "attention.layer_norm_rms_epsilon");
    if (!eps.has_value() || !(*eps > 0.0F)) {
      if (error_msg != nullptr) {
        *error_msg = "muse-glimmer is missing attention.layer_norm_rms_epsilon";
      }
      return std::nullopt;
    }
    config.norm_eps_pre = *eps;

    const auto window = GetMetadataUint32(prefix + "attention.sliding_window");
    if (!window.has_value() || *window != 2048U) {
      if (error_msg != nullptr) {
        *error_msg = "muse-glimmer sliding_window must be 2048";
      }
      return std::nullopt;
    }
    config.swa_window = *window;

    const auto scale = GetMetadataFloat32(prefix + "logit_scale");
    if (!scale.has_value() || *scale == 0.0F) {
      if (error_msg != nullptr) {
        *error_msg = "muse-glimmer is missing logit_scale";
      }
      return std::nullopt;
    }
    config.logit_scale = *scale;

    const auto softcap = GetMetadataFloat32(prefix + "final_logit_softcapping");
    if (!softcap.has_value() || !(*softcap > 0.0F)) {
      if (error_msg != nullptr) {
        *error_msg = "muse-glimmer is missing final_logit_softcapping";
      }
      return std::nullopt;
    }
    config.final_softcap = *softcap;

    const auto pattern =
        GetMetadataUintArray(prefix + "attention.sliding_window_pattern");
    if (pattern.size() != config.num_layers ||
        config.num_layers > config.use_rope.size()) {
      if (error_msg != nullptr) {
        *error_msg =
            "muse-glimmer sliding_window_pattern length must equal block_count";
      }
      return std::nullopt;
    }
    for (std::uint32_t i = 0; i < config.num_layers; ++i) {
      if (pattern[i] > 1U) {
        if (error_msg != nullptr) {
          *error_msg = "muse-glimmer sliding_window_pattern is not boolean";
        }
        return std::nullopt;
      }
      config.use_rope[i] = pattern[i] == 0U ? 0U : 1U;
    }
  }

  if (gufo::core::IsLfm2Family(config.architecture)) {  // LFM2
    if (HasMtpTensors()) {
      if (error_msg != nullptr) {
        *error_msg = "lfm2moe does not support MTP";
      }
      return std::nullopt;
    }
    config.num_layers = total_layers;
    config.mtp_num_layers = 0;
    config.lfm2 = true;
    const bool lfm2_dense = gufo::core::IsModelFamily(
        config.architecture, gufo::core::ModelFamilyId::kLfm2);
    if (total_layers == 0 || total_layers > config.lfm2_attn.size()) {
      if (error_msg != nullptr) {
        *error_msg = "lfm2moe block_count out of range";
      }
      return std::nullopt;
    }
    const auto kv = GetMetadataUintArray(prefix + "attention.head_count_kv");
    if (kv.size() != total_layers) {
      if (error_msg != nullptr) {
        *error_msg =
            "lfm2moe attention.head_count_kv must be a per-layer array";
      }
      return std::nullopt;
    }
    std::uint64_t kv_first = 0;
    std::uint32_t attn_seen = 0;
    std::uint32_t conv_seen = 0;
    for (std::uint32_t i = 0; i < total_layers; ++i) {
      if (kv[i] != 0) {
        if (kv_first == 0) {
          kv_first = kv[i];
        } else if (kv[i] != kv_first) {
          if (error_msg != nullptr) {
            *error_msg = "lfm2moe head_count_kv is not uniform";
          }
          return std::nullopt;
        }
        config.lfm2_attn[i] = 1;
        config.lfm2_kind_index[i] = static_cast<std::uint8_t>(attn_seen);
        ++attn_seen;
      } else {
        config.lfm2_attn[i] = 0;
        config.lfm2_kind_index[i] = static_cast<std::uint8_t>(conv_seen);
        ++conv_seen;
      }
    }
    if (kv_first == 0) {
      if (error_msg != nullptr) {
        *error_msg = "lfm2moe has no attention layer";
      }
      return std::nullopt;
    }
    config.num_key_value_heads = static_cast<std::uint32_t>(kv_first);
    if (lfm2_dense) {
      // Super ROC: dense lfm2 — every layer is a SwiGLU FFN, no router.
      if (config.expert_count != 0) {
        if (error_msg != nullptr) {
          *error_msg = "lfm2 (dense) must not carry experts";
        }
        return std::nullopt;
      }
      config.dense_lead_layers = total_layers;
      config.moe_sigmoid_bias = false;
    } else {
      const auto dense_lead =
          GetMetadataUint32(prefix + "leading_dense_block_count");
      if (!dense_lead.has_value()) {
        if (error_msg != nullptr) {
          *error_msg = "lfm2moe is missing leading_dense_block_count";
        }
        return std::nullopt;
      }
      config.dense_lead_layers = *dense_lead;
      const auto gating = GetMetadataUint32(prefix + "expert_gating_func");
      if (!gating.has_value() || *gating != 2U) {
        if (error_msg != nullptr) {
          *error_msg = "lfm2moe expert_gating_func must be 2 (sigmoid)";
        }
        return std::nullopt;
      }
      config.moe_sigmoid_bias = true;
    }
    const auto l_cache = GetMetadataUint32(prefix + "shortconv.l_cache");
    if (!l_cache.has_value()) {
      if (error_msg != nullptr) {
        *error_msg = "lfm2moe is missing shortconv.l_cache";
      }
      return std::nullopt;
    }
    config.shortconv_l_cache = *l_cache;
    const auto eps =
        GetMetadataFloat32(prefix + "attention.layer_norm_rms_epsilon");
    if (!eps.has_value() || !std::isfinite(*eps) || !(*eps > 0.0F)) {
      if (error_msg != nullptr) {
        *error_msg = "lfm2moe is missing attention.layer_norm_rms_epsilon";
      }
      return std::nullopt;
    }
    config.norm_eps_pre = *eps;
    if (!std::isfinite(config.rope_theta) || !(config.rope_theta > 0.0F)) {
      if (error_msg != nullptr) {
        *error_msg = "lfm2moe rope.freq_base is invalid";
      }
      return std::nullopt;
    }
    config.full_attention_interval = 4;                     // LFM2 chassis
    config.ssm_conv_kernel = config.shortconv_l_cache - 1;  // LFM2 chassis
    config.ssm_group_count = 0;                             // LFM2 chassis
    config.ssm_state_size = 0;                              // LFM2 chassis
    config.ssm_inner_size = config.hidden_size;             // LFM2 chassis
    config.ssm_time_step_rank = 1;                          // LFM2 chassis
    config.rope_variant = RopeVariant::kNeox;               // LFM2 chassis
    config.norm_eps_post = config.norm_eps_pre;             // LFM2 chassis
  }  // LFM2

  if (gufo::core::IsModelFamily(config.architecture,
                                gufo::core::ModelFamilyId::kFlint)) {
    if (HasMtpTensors() || embedded_mtp_layers.value_or(0U) != 0U) {
      if (error_msg != nullptr) {
        *error_msg = "flint does not support MTP";
      }
      return std::nullopt;
    }
    config.flint = true;
    config.attn_sinks = true;
    config.num_layers = total_layers;
    config.mtp_num_layers = 0;
    config.full_attention_interval = 1;
    config.rope_variant = RopeVariant::kNeox;
    config.moe_sigmoid_bias = true;

    const auto eps =
        GetMetadataFloat32(prefix + "attention.layer_norm_rms_epsilon");
    if (!eps.has_value() || !std::isfinite(*eps) || !(*eps > 0.0F)) {
      if (error_msg != nullptr) {
        *error_msg = "flint is missing attention.layer_norm_rms_epsilon";
      }
      return std::nullopt;
    }
    config.norm_eps_pre = *eps;
    config.norm_eps_post = *eps;

    const auto window = GetMetadataUint32(prefix + "attention.sliding_window");
    if (!window.has_value() || *window != 513U) {
      if (error_msg != nullptr) {
        *error_msg = "flint sliding_window must be 513";
      }
      return std::nullopt;
    }
    config.swa_window = *window;

    const auto dense_lead =
        GetMetadataUint32(prefix + "leading_dense_block_count");
    if (!dense_lead.has_value() || *dense_lead == 0U) {
      if (error_msg != nullptr) {
        *error_msg = "flint is missing leading_dense_block_count";
      }
      return std::nullopt;
    }
    config.dense_lead_layers = *dense_lead;

    const auto gating = GetMetadataUint32(prefix + "expert_gating_func");
    if (!gating.has_value() || *gating != 2U) {
      if (error_msg != nullptr) {
        *error_msg = "flint expert_gating_func must be 2 (sigmoid)";
      }
      return std::nullopt;
    }

    if (auto vocab = GetMetadataUint32(prefix + "vocab_size")) {
      config.vocab_size = *vocab;
    }

    const auto pattern =
        GetMetadataUintArray(prefix + "attention.sliding_window_pattern");
    if (pattern.size() != config.num_layers ||
        config.num_layers > config.use_rope.size()) {
      if (error_msg != nullptr) {
        *error_msg =
            "flint sliding_window_pattern length must equal block_count";
      }
      return std::nullopt;
    }
    for (std::uint32_t i = 0; i < config.num_layers; ++i) {
      if (pattern[i] > 1U) {
        if (error_msg != nullptr) {
          *error_msg = "flint sliding_window_pattern is not boolean";
        }
        return std::nullopt;
      }
      config.use_rope[i] = pattern[i] == 0U ? 0U : 1U;
    }
  }

  if (gufo::core::IsModelFamily(config.architecture,
                                gufo::core::ModelFamilyId::kLlama)) {
    // LLAMA: a plain dense transformer. Every architectural key is required:
    // the struct defaults are Qwen's, so a missing key must never be inherited.
    if (HasMtpTensors() || embedded_mtp_layers.value_or(0U) != 0U) {
      if (error_msg != nullptr) {
        *error_msg = "llama does not support MTP";
      }
      return std::nullopt;
    }
    if (config.is_moe) {
      if (error_msg != nullptr) {
        *error_msg = "llama does not support MoE (expert_count must be 0)";
      }
      return std::nullopt;
    }
    const auto fail = [error_msg](std::string_view key) {
      if (error_msg != nullptr) {
        *error_msg = "llama is missing or has an invalid " + std::string(key);
      }
      return std::nullopt;
    };
    const auto block_count = GetMetadataUint32(prefix + "block_count");
    const auto embedding = GetMetadataUint32(prefix + "embedding_length");
    const auto feed_forward = GetMetadataUint32(prefix + "feed_forward_length");
    const auto heads = GetMetadataUint32(prefix + "attention.head_count");
    const auto kv_heads = GetMetadataUint32(prefix + "attention.head_count_kv");
    const auto key_len = GetMetadataUint32(prefix + "attention.key_length");
    const auto value_len = GetMetadataUint32(prefix + "attention.value_length");
    const auto rope_dim = GetMetadataUint32(prefix + "rope.dimension_count");
    const auto context = GetMetadataUint32(prefix + "context_length");
    const auto vocab = GetMetadataUint32(prefix + "vocab_size");
    const auto eps =
        GetMetadataFloat32(prefix + "attention.layer_norm_rms_epsilon");
    const auto theta = GetMetadataFloat32(prefix + "rope.freq_base");
    if (!block_count.has_value() || *block_count == 0U) {
      return fail("block_count");
    }
    if (!embedding.has_value() || *embedding == 0U) {
      return fail("embedding_length");
    }
    if (!feed_forward.has_value() || *feed_forward == 0U) {
      return fail("feed_forward_length");
    }
    if (!heads.has_value() || *heads == 0U) {
      return fail("attention.head_count");
    }
    if (!kv_heads.has_value() || *kv_heads == 0U || *kv_heads > *heads ||
        (*heads % *kv_heads) != 0U) {
      return fail("attention.head_count_kv");
    }
    if (!key_len.has_value() || *key_len == 0U || (*key_len % 2U) != 0U ||
        !value_len.has_value() || *value_len != *key_len) {
      return fail("attention.key_length / attention.value_length");
    }
    if (!rope_dim.has_value() || *rope_dim != *key_len) {
      return fail("rope.dimension_count");
    }
    if (!context.has_value() || *context == 0U) {
      return fail("context_length");
    }
    if (!vocab.has_value() || *vocab == 0U) {
      return fail("vocab_size");
    }
    if (!eps.has_value() || !std::isfinite(*eps) || !(*eps > 0.0F)) {
      return fail("attention.layer_norm_rms_epsilon");
    }
    if (!theta.has_value() || !std::isfinite(*theta) || !(*theta > 0.0F)) {
      return fail("rope.freq_base");
    }
    if (const auto* token_embd = FindTensor("token_embd.weight");
        token_embd != nullptr &&
        token_embd->ElementCount() !=
            static_cast<std::uint64_t>(*vocab) * *embedding) {
      return fail("token_embd.weight shape vs vocab_size * embedding_length");
    }

    config.llama = true;
    config.num_layers = *block_count;
    config.hidden_size = *embedding;
    config.intermediate_size = *feed_forward;
    config.num_attention_heads = *heads;
    config.num_key_value_heads = *kv_heads;
    config.head_dim = *key_len;
    config.rotary_dim = *rope_dim;
    config.context_length = *context;
    config.vocab_size = *vocab;
    config.norm_eps_pre = *eps;
    config.norm_eps_post = *eps;
    config.rope_theta = *theta;
    // Every layer is full causal attention with adjacent-pair (NORM) RoPE.
    config.full_attention_interval = 1;
    config.rope_variant = gufo::core::RopeVariant::kNorm;
    config.attn_gate_mode = gufo::core::AttnGateMode::kFused;
    config.mtp_num_layers = 0;
    config.is_moe = false;
    config.attn_sinks = false;
    config.swa_window = 0;
    if (config.num_layers > config.use_rope.size()) {
      return fail("block_count");
    }
    for (std::uint32_t i = 0; i < config.num_layers; ++i) {
      config.use_rope[i] = 1U;
    }
  }

  // 6. Validate configuration against Qwen3.5/3.8 structural rules
  if (!config.IsValidQwen()) {
    if (error_msg != nullptr) {
      *error_msg = "Model config failed Qwen structural validation (head_dim=" +
                   std::to_string(config.head_dim) +
                   ", layers=" + std::to_string(config.num_layers) + ")";
    }
    return std::nullopt;
  }

  return config;
}

}  // namespace gufo::core
