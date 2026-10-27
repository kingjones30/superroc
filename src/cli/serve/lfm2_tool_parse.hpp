// LFM2: pythonic tool-call parser for the pinned LFM2.5 template.
#ifndef SUPERROC_SERVER_LFM2_TOOL_PARSE_HPP_
#define SUPERROC_SERVER_LFM2_TOOL_PARSE_HPP_

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen/chat_template.hpp"

namespace superroc::server {

// LFM2: one pythonic tool call parsed from the text between
// <|tool_call_start|> and <|tool_call_end|>.
struct Lfm2Call {
  std::string name;
  std::vector<tokenization::ChatMessage::ToolArgument> arguments;
};

// LFM2: parses the body BETWEEN <|tool_call_start|> and <|tool_call_end|>.
// Grammar: optional spaces, '[', zero or more calls separated by ',' plus
// optional spaces, ']', optional spaces. Any syntax error yields nullopt
// (the whole block is then left as content: fail closed, never partial).
std::optional<std::vector<Lfm2Call>> ParseLfm2PythonicCalls(
    std::string_view body);

}  // namespace superroc::server

#endif  // SUPERROC_SERVER_LFM2_TOOL_PARSE_HPP_
