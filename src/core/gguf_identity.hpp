#ifndef SUPERROC_CORE_GGUF_IDENTITY_HPP_
#define SUPERROC_CORE_GGUF_IDENTITY_HPP_

#include <string>
#include <string_view>

#include "src/core/gguf_reader.hpp"

namespace superroc::core {

/// Scheme tag for full-content GGUF identities.
inline constexpr std::string_view kGgufIdentityScheme = "gguf-sha256-v1";

/// Hashes every byte of every shard, including metadata and tensor padding.
/// File digests are cached privately by inode, size, mtime and ctime; memory
/// images are always hashed. Copies have identical identities. The initial
/// scan reads the full artifact; later launches reuse its cached full digest.
/// Throws if the mapped artifact changes while its identity is computed.
[[nodiscard]] std::string GgufIdentityHex(const GgufReader& reader);

}  // namespace superroc::core

#endif  // SUPERROC_CORE_GGUF_IDENTITY_HPP_
