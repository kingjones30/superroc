// LFM2: pythonic tool-call parser for the pinned LFM2.5 template.
// Format (from the pinned chat template): '[' + calls joined by ', ' + ']',
// a call = name(arg=value, ...), a value = a Python literal: a single-quoted
// string (escapes \\ \' \n \r \t), a double-quoted string, an int/float,
// True/False/None, or a JSON/Python object or list.
#include "src/cli/serve/lfm2_tool_parse.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace gufo::server {
namespace {

bool IsAlpha(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool IsDigit(char c) {
  return c >= '0' && c <= '9';
}

bool IsAlnum(char c) {
  return IsAlpha(c) || IsDigit(c);
}

// LFM2: a nested object/array literal held for conversion to JSON text.
struct NestedValue {
  enum class Kind { kNull, kBool, kNumber, kString, kArray, kObject };
  Kind kind = Kind::kNull;
  bool boolean = false;
  std::string text;  // kNumber: as written; kString: unescaped.
  std::vector<NestedValue> items;
  std::vector<std::pair<std::string, NestedValue>> members;
};

// LFM2: recursive-descent reader over the block body. Any syntax error
// returns false up the chain and the whole block yields nullopt.
class Lfm2Reader {
public:
  explicit Lfm2Reader(std::string_view body) : body_(body) {}

  bool AtEnd() {
    SkipSpaces();
    return pos_ == body_.size();
  }

  bool Consume(char c) {
    SkipSpaces();
    if (pos_ < body_.size() && body_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }

  // LFM2: a name is [A-Za-z_][A-Za-z0-9_.-]*.
  bool ParseName(std::string* out) {
    SkipSpaces();
    const std::size_t start = pos_;
    if (pos_ < body_.size() && (IsAlpha(body_[pos_]) || body_[pos_] == '_')) {
      ++pos_;
      while (pos_ < body_.size() &&
             (IsAlnum(body_[pos_]) || body_[pos_] == '_' ||
              body_[pos_] == '.' || body_[pos_] == '-')) {
        ++pos_;
      }
      out->assign(body_.substr(start, pos_ - start));
      return true;
    }
    return false;
  }

  bool ParseCall(Lfm2Call* out) {
    if (!ParseName(&out->name))
      return false;
    if (!Consume('('))
      return false;
    if (Consume(')'))
      return true;
    while (true) {
      std::string key;
      if (!ParseName(&key))
        return false;
      if (!Consume('='))
        return false;
      tokenization::ChatMessage::ToolArgument argument;
      if (!ParseValue(&argument, key))
        return false;
      out->arguments.push_back(std::move(argument));
      if (Consume(','))
        continue;
      if (Consume(')'))
        return true;
      return false;
    }
  }

private:
  void SkipSpaces() {
    while (pos_ < body_.size() &&
           (body_[pos_] == ' ' || body_[pos_] == '\t' || body_[pos_] == '\r' ||
            body_[pos_] == '\n')) {
      ++pos_;
    }
  }

  bool ParseValue(tokenization::ChatMessage::ToolArgument* out,
                  const std::string& key) {
    SkipSpaces();
    if (pos_ >= body_.size())
      return false;
    const char c = body_[pos_];
    out->name = key;
    if (c == '\'' || c == '"') {
      std::string text;
      if (!ParseQuoted(&text))
        return false;
      out->value = std::move(text);
      out->is_string = true;
      return true;
    }
    if (c == '{' || c == '[') {
      NestedValue nested;
      if (!ParseNested(&nested))
        return false;
      NestedToJson(&out->value, nested);
      out->is_string = false;
      return true;
    }
    if (c == 'T' && TakeWord("True")) {
      out->value = "true";
      out->is_string = false;
      return true;
    }
    if (c == 'F' && TakeWord("False")) {
      out->value = "false";
      out->is_string = false;
      return true;
    }
    if (c == 'N' && TakeWord("None")) {
      out->value = "null";
      out->is_string = false;
      return true;
    }
    // LFM2 (Super ROC): JSON spellings, admitted by the tool grammar too.
    if ((c == 't' && TakeWord("true")) || (c == 'f' && TakeWord("false")) ||
        (c == 'n' && TakeWord("null"))) {
      out->value = c == 't' ? "true" : (c == 'f' ? "false" : "null");
      out->is_string = false;
      return true;
    }
    if (c == '-' || IsDigit(c)) {
      std::string number;
      if (!ParseNumber(&number))
        return false;
      out->value = std::move(number);
      out->is_string = false;
      return true;
    }
    return false;
  }

  // LFM2: single- or double-quoted string; unescapes \\ \' \" \n \r \t \uXXXX.
  bool ParseQuoted(std::string* out) {
    const char quote = body_[pos_];
    ++pos_;
    while (true) {
      if (pos_ >= body_.size())
        return false;
      const char c = body_[pos_++];
      if (c == quote)
        return true;
      if (c != '\\') {
        out->push_back(c);
        continue;
      }
      if (pos_ >= body_.size())
        return false;
      const char escape = body_[pos_++];
      switch (escape) {
        case '\\':
          out->push_back('\\');
          break;
        case '\'':
          out->push_back('\'');
          break;
        case '"':
          out->push_back('"');
          break;
        case 'n':
          out->push_back('\n');
          break;
        case 'r':
          out->push_back('\r');
          break;
        case 't':
          out->push_back('\t');
          break;
        case '/':  // LFM2 (Super ROC): JSON escapes
          out->push_back('/');
          break;
        case 'b':
          out->push_back('\b');
          break;
        case 'f':
          out->push_back('\f');
          break;
        case 'u': {
          std::uint32_t point = 0;
          if (!ParseHex4(&point))
            return false;
          if (point >= 0xD800 && point <= 0xDBFF) {
            if (pos_ + 1 >= body_.size() || body_[pos_] != '\\' ||
                body_[pos_ + 1] != 'u') {
              return false;
            }
            pos_ += 2;
            std::uint32_t low = 0;
            if (!ParseHex4(&low))
              return false;
            if (low < 0xDC00 || low > 0xDFFF)
              return false;
            point = 0x10000 + ((point - 0xD800) << 10) + (low - 0xDC00);
          } else if (point >= 0xDC00 && point <= 0xDFFF) {
            return false;
          }
          AppendUtf8(out, point);
          break;
        }
        default:
          return false;
      }
    }
  }

  bool ParseHex4(std::uint32_t* out) {
    if (pos_ + 4 > body_.size())
      return false;
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = body_[pos_ + i];
      value <<= 4;
      if (c >= '0' && c <= '9') {
        value |= static_cast<std::uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        value |= static_cast<std::uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        value |= static_cast<std::uint32_t>(c - 'A' + 10);
      } else {
        return false;
      }
    }
    pos_ += 4;
    *out = value;
    return true;
  }

  static void AppendUtf8(std::string* out, std::uint32_t point) {
    if (point < 0x80) {
      out->push_back(static_cast<char>(point));
    } else if (point < 0x800) {
      out->push_back(static_cast<char>(0xC0 | (point >> 6)));
      out->push_back(static_cast<char>(0x80 | (point & 0x3F)));
    } else if (point < 0x10000) {
      out->push_back(static_cast<char>(0xE0 | (point >> 12)));
      out->push_back(static_cast<char>(0x80 | ((point >> 6) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | (point & 0x3F)));
    } else {
      out->push_back(static_cast<char>(0xF0 | (point >> 18)));
      out->push_back(static_cast<char>(0x80 | ((point >> 12) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | ((point >> 6) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | (point & 0x3F)));
    }
  }

  // LFM2: keyword with an identifier boundary (NoneX is not None).
  bool TakeWord(std::string_view word) {
    if (!body_.substr(pos_).starts_with(word))
      return false;
    const std::size_t after = pos_ + word.size();
    if (after < body_.size() &&
        (IsAlnum(body_[after]) || body_[after] == '_')) {
      return false;
    }
    pos_ = after;
    return true;
  }

  // LFM2: JSON number grammar, no leading '+'.
  bool ParseNumber(std::string* out) {
    const std::size_t start = pos_;
    if (pos_ < body_.size() && body_[pos_] == '-')
      ++pos_;
    if (pos_ >= body_.size()) {
      pos_ = start;
      return false;
    }
    if (body_[pos_] == '0') {
      ++pos_;
    } else if (IsDigit(body_[pos_])) {
      while (pos_ < body_.size() && IsDigit(body_[pos_]))
        ++pos_;
    } else {
      pos_ = start;
      return false;
    }
    if (pos_ < body_.size() && body_[pos_] == '.') {
      ++pos_;
      if (pos_ >= body_.size() || !IsDigit(body_[pos_])) {
        pos_ = start;
        return false;
      }
      while (pos_ < body_.size() && IsDigit(body_[pos_]))
        ++pos_;
    }
    if (pos_ < body_.size() && (body_[pos_] == 'e' || body_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < body_.size() && (body_[pos_] == '+' || body_[pos_] == '-')) {
        ++pos_;
      }
      if (pos_ >= body_.size() || !IsDigit(body_[pos_])) {
        pos_ = start;
        return false;
      }
      while (pos_ < body_.size() && IsDigit(body_[pos_]))
        ++pos_;
    }
    out->assign(body_.substr(start, pos_ - start));
    return true;
  }

  // LFM2: nested literal; strings may be single- or double-quoted,
  // True/False/None map to JSON, object keys may be quoted either way.
  bool ParseNested(NestedValue* out) {
    // HTC (RVW C3): nesting is capped, so a runaway or hostile body cannot
    // recurse until the stack overflows; deeper input is a parse failure
    // (the call is dropped, fail closed). The required-call grammar emits at
    // most 4 levels; 64 leaves room for free-form JSON arguments.
    const DepthGuard guard(&depth_);
    if (depth_ > kMaxNestDepth)
      return false;
    SkipSpaces();
    if (pos_ >= body_.size())
      return false;
    const char c = body_[pos_];
    if (c == '\'' || c == '"') {
      out->kind = NestedValue::Kind::kString;
      return ParseQuoted(&out->text);
    }
    if (c == '[') {
      ++pos_;
      out->kind = NestedValue::Kind::kArray;
      SkipSpaces();
      if (pos_ < body_.size() && body_[pos_] == ']') {
        ++pos_;
        return true;
      }
      while (true) {
        NestedValue item;
        if (!ParseNested(&item))
          return false;
        out->items.push_back(std::move(item));
        SkipSpaces();
        if (pos_ >= body_.size())
          return false;
        if (body_[pos_] == ',') {
          ++pos_;
          continue;
        }
        if (body_[pos_] == ']') {
          ++pos_;
          return true;
        }
        return false;
      }
    }
    if (c == '{') {
      ++pos_;
      out->kind = NestedValue::Kind::kObject;
      SkipSpaces();
      if (pos_ < body_.size() && body_[pos_] == '}') {
        ++pos_;
        return true;
      }
      while (true) {
        SkipSpaces();
        if (pos_ >= body_.size())
          return false;
        std::string key;
        if (body_[pos_] == '\'' || body_[pos_] == '"') {
          if (!ParseQuoted(&key))
            return false;
        } else {
          if (!ParseName(&key))
            return false;
        }
        SkipSpaces();
        if (pos_ >= body_.size() || body_[pos_] != ':')
          return false;
        ++pos_;
        NestedValue value;
        if (!ParseNested(&value))
          return false;
        out->members.emplace_back(std::move(key), std::move(value));
        SkipSpaces();
        if (pos_ >= body_.size())
          return false;
        if (body_[pos_] == ',') {
          ++pos_;
          continue;
        }
        if (body_[pos_] == '}') {
          ++pos_;
          return true;
        }
        return false;
      }
    }
    if (c == 'T' && TakeWord("True")) {
      out->kind = NestedValue::Kind::kBool;
      out->boolean = true;
      return true;
    }
    if (c == 'F' && TakeWord("False")) {
      out->kind = NestedValue::Kind::kBool;
      out->boolean = false;
      return true;
    }
    if (c == 'N' && TakeWord("None")) {
      out->kind = NestedValue::Kind::kNull;
      return true;
    }
    // LFM2 (Super ROC): JSON spellings, admitted by the tool grammar too.
    if (c == 't' && TakeWord("true")) {
      out->kind = NestedValue::Kind::kBool;
      out->boolean = true;
      return true;
    }
    if (c == 'f' && TakeWord("false")) {
      out->kind = NestedValue::Kind::kBool;
      out->boolean = false;
      return true;
    }
    if (c == 'n' && TakeWord("null")) {
      out->kind = NestedValue::Kind::kNull;
      return true;
    }
    if (c == '-' || IsDigit(c)) {
      out->kind = NestedValue::Kind::kNumber;
      return ParseNumber(&out->text);
    }
    return false;
  }

  static void AppendJsonString(std::string* out, std::string_view text) {
    out->push_back('"');
    for (const char ch : text) {
      switch (ch) {
        case '"':
          out->append("\\\"");
          break;
        case '\\':
          out->append("\\\\");
          break;
        case '\n':
          out->append("\\n");
          break;
        case '\r':
          out->append("\\r");
          break;
        case '\t':
          out->append("\\t");
          break;
        case '\b':
          out->append("\\b");
          break;
        case '\f':
          out->append("\\f");
          break;
        default:
          if (static_cast<unsigned char>(ch) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
            out->append(buf);
          } else {
            out->push_back(ch);
          }
          break;
      }
    }
    out->push_back('"');
  }

  static void NestedToJson(std::string* out, const NestedValue& value) {
    switch (value.kind) {
      case NestedValue::Kind::kNull:
        out->append("null");
        break;
      case NestedValue::Kind::kBool:
        out->append(value.boolean ? "true" : "false");
        break;
      case NestedValue::Kind::kNumber:
        out->append(value.text);
        break;
      case NestedValue::Kind::kString:
        AppendJsonString(out, value.text);
        break;
      case NestedValue::Kind::kArray: {
        out->push_back('[');
        for (std::size_t i = 0; i < value.items.size(); ++i) {
          if (i > 0)
            out->append(", ");
          NestedToJson(out, value.items[i]);
        }
        out->push_back(']');
        break;
      }
      case NestedValue::Kind::kObject: {
        out->push_back('{');
        for (std::size_t i = 0; i < value.members.size(); ++i) {
          if (i > 0)
            out->append(", ");
          AppendJsonString(out, value.members[i].first);
          out->append(": ");
          NestedToJson(out, value.members[i].second);
        }
        out->push_back('}');
        break;
      }
    }
  }

  static constexpr int kMaxNestDepth = 64;
  struct DepthGuard {
    explicit DepthGuard(int* depth) : depth_(depth) { ++*depth_; }
    ~DepthGuard() { --*depth_; }
    DepthGuard(const DepthGuard&) = delete;
    DepthGuard& operator=(const DepthGuard&) = delete;
    int* depth_;
  };

  std::string_view body_;
  std::size_t pos_ = 0;
  int depth_ = 0;
};

}  // namespace

std::optional<std::vector<Lfm2Call>> ParseLfm2PythonicCalls(
    std::string_view body) {
  Lfm2Reader reader(body);
  if (!reader.Consume('['))
    return std::nullopt;
  std::vector<Lfm2Call> calls;
  if (reader.Consume(']')) {
    if (!reader.AtEnd())
      return std::nullopt;
    return calls;
  }
  while (true) {
    Lfm2Call call;
    if (!reader.ParseCall(&call))
      return std::nullopt;
    calls.push_back(std::move(call));
    if (reader.Consume(','))
      continue;
    if (reader.Consume(']'))
      break;
    return std::nullopt;
  }
  if (!reader.AtEnd())
    return std::nullopt;
  return calls;
}

}  // namespace gufo::server
