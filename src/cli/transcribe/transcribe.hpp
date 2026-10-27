#ifndef SUPERROC_CLI_TRANSCRIBE_HPP_
#define SUPERROC_CLI_TRANSCRIBE_HPP_

#include <span>
#include <string_view>

namespace superroc::cli {

void PrintTranscribeHelp(std::string_view program_name);
[[nodiscard]] int RunTranscribe(std::span<const char* const> args);

}  // namespace superroc::cli

#endif  // SUPERROC_CLI_TRANSCRIBE_HPP_
