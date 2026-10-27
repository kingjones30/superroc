#ifndef SUPERROC_CORE_SESSION_MODE_HPP_
#define SUPERROC_CORE_SESSION_MODE_HPP_

namespace superroc::core {

/// Per-session execution choice, independent of resident model capabilities.
enum class SessionMode { kAutoregressive, kSpeculative };

}  // namespace superroc::core

#endif  // SUPERROC_CORE_SESSION_MODE_HPP_
