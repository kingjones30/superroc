#ifndef GUFO_MODELS_QWEN_CHAT_JSON_HPP_
#define GUFO_MODELS_QWEN_CHAT_JSON_HPP_

// LFM2: shared JSON helpers for the Qwen chat renderers. Moved verbatim from
// chat_template.cpp's anonymous namespace so the LFM2 renderer reuses the
// exact Qwen spacing/quoting.

#include <string>
#include <string_view>

namespace gufo::tokenization {

inline std::string PythonJsonSpacing(std::string_view value) {
  std::string output;
  output.reserve(value.size() + value.size() / 8);
  bool in_string = false;
  bool escaped = false;
  for (const char character : value) {
    output.push_back(character);
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        in_string = false;
      }
      continue;
    }
    if (character == '"') {
      in_string = true;
    } else if (character == ':' || character == ',') {
      output.push_back(' ');
    }
  }
  return output;
}

inline void AppendJsonString(std::string& output, std::string_view value) {
  output.push_back('"');
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output.append("\\\"");
        break;
      case '\\':
        output.append("\\\\");
        break;
      case '\b':
        output.append("\\b");
        break;
      case '\f':
        output.append("\\f");
        break;
      case '\n':
        output.append("\\n");
        break;
      case '\r':
        output.append("\\r");
        break;
      case '\t':
        output.append("\\t");
        break;
      default:
        output.push_back(static_cast<char>(character));
        break;
    }
  }
  output.push_back('"');
}

}  // namespace gufo::tokenization

#endif  // GUFO_MODELS_QWEN_CHAT_JSON_HPP_
