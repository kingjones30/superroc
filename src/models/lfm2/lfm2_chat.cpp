#include "src/models/lfm2/lfm2_chat.hpp"

#include <unicode/uchar.h>
#include <unicode/utf8.h>

#include <cstdint>
#include <string_view>

#include "src/core/json.hpp"
#include "src/models/qwen/chat_json.hpp"

namespace gufo::tokenization {
namespace {

// Message text is text. Content spans are tokenized without special matching,
// so a vocabulary token a client sends reaches the model as the characters it
// contains (same pattern as chat_template.cpp's AppendContent).
void AppendContent(std::string& output, std::vector<ContentSpan>* spans,
                   std::string_view text) {
  const auto begin = output.size();
  output.append(text);
  if (spans != nullptr && !text.empty()) {
    spans->push_back({begin, output.size() - begin});
  }
}

// The template's `| trim` (Python str.strip).
std::string_view TrimView(std::string_view value) {
  const auto whitespace = [](UChar32 cp) {
    return u_isUWhiteSpace(cp) || (cp >= 0x1c && cp <= 0x1f);
  };
  std::int32_t first = 0;
  auto last = static_cast<std::int32_t>(value.size());
  while (first < last) {
    auto next = first;
    UChar32 cp;
    U8_NEXT(value.data(), next, last, cp);
    if (!whitespace(cp)) {
      break;
    }
    first = next;
  }
  while (first < last) {
    auto previous = last;
    UChar32 cp;
    U8_PREV(value.data(), first, previous, cp);
    if (!whitespace(cp)) {
      break;
    }
    last = previous;
  }
  return value.substr(first, last - first);
}

// A tool's JSON: the spaced definition when present, else the same
// {"type": "function", "function": {...}} shape AppendToolsPrompt builds.
void AppendToolJson(std::string& output, const ChatTool& tool) {
  if (!tool.definition_json.empty()) {
    output.append(PythonJsonSpacing(tool.definition_json));
    return;
  }
  output.append("{\"type\": \"function\", \"function\": {\"name\": ");
  AppendJsonString(output, tool.name);
  output.append(", \"description\": ");
  AppendJsonString(output, tool.description);
  output.append(", \"parameters\": ");
  output.append(tool.parameters_json.empty()
                    ? "{}"
                    : PythonJsonSpacing(tool.parameters_json));
  output.append("}}");
}

// Python repr() of a JSON string: '"' when it holds ' and no '"', else '\''.
void AppendPythonReprString(std::string& output, std::string_view value) {
  char quote = '\'';
  if (value.find('\'') != std::string_view::npos &&
      value.find('"') == std::string_view::npos) {
    quote = '"';
  }
  output.push_back(quote);
  for (const char ch : value) {
    const auto c = static_cast<unsigned char>(ch);
    if (c == '\\') {
      output.append("\\\\");
    } else if (c == static_cast<unsigned char>(quote)) {
      output.push_back('\\');
      output.push_back(quote);
    } else if (c == '\n') {
      output.append("\\n");
    } else if (c == '\r') {
      output.append("\\r");
    } else if (c == '\t') {
      output.append("\\t");
    } else if (c < 0x20 || c == 0x7f) {
      static constexpr char kHex[] = "0123456789abcdef";
      output.append("\\x");
      output.push_back(kHex[(c >> 4) & 0xf]);
      output.push_back(kHex[c & 0xf]);
    } else {
      output.push_back(ch);
    }
  }
  output.push_back(quote);
}

// NOLINTNEXTLINE(misc-no-recursion)
// Python repr() of a JSON value for sequence tool arguments.
void AppendPythonRepr(std::string& output, const gufo::json::Value& v) {
  if (v.is_string()) {
    AppendPythonReprString(output, v.str());
    return;
  }
  if (v.is_bool()) {
    output.append(v.as_bool() ? "True" : "False");
    return;
  }
  if (v.is_null()) {
    output.append("None");
    return;
  }
  if (v.is_number()) {
    output.append(v.dump());
    return;
  }
  if (v.is_array()) {
    output.push_back('[');
    bool first = true;
    for (const auto& item : v.items()) {
      if (!first) {
        output.append(", ");
      }
      first = false;
      AppendPythonRepr(output, item);
    }
    output.push_back(']');
    return;
  }
  if (v.is_object()) {
    output.push_back('{');
    bool first = true;
    for (const auto& [key, val] : v.members()) {
      if (!first) {
        output.append(", ");
      }
      first = false;
      AppendPythonReprString(output, key);
      output.append(": ");
      AppendPythonRepr(output, val);
    }
    output.push_back('}');
    return;
  }
  output.append(v.dump());
}

// The template's format_arg_value macro: strings are single-quoted (with
// backslash, quote, LF, CR escapes when the variant escapes string args,
// else raw); true/false/null become True/False/None; objects keep JSON
// spacing; sequences use Python repr under repr variants, else JSON spacing;
// numbers stay as is.
void AppendArgValue(std::string& output, const ChatMessage::ToolArgument& arg,
                    const Lfm2TemplateVariant& variant) {
  if (arg.is_string) {
    output.push_back('\'');
    if (variant.escape_string_args) {
      for (const char character : arg.value) {
        switch (character) {
          case '\\':
            output.append("\\\\");
            break;
          case '\'':
            output.append("\\'");
            break;
          case '\n':
            output.append("\\n");
            break;
          case '\r':
            output.append("\\r");
            break;
          default:
            output.push_back(character);
            break;
        }
      }
    } else {
      output.append(arg.value);
    }
    output.push_back('\'');
    return;
  }
  if (arg.value == "true") {
    output.append("True");
  } else if (arg.value == "false") {
    output.append("False");
  } else if (arg.value == "null") {
    output.append("None");
  } else if (!arg.value.empty() && arg.value.front() == '{') {
    output.append(PythonJsonSpacing(arg.value));
  } else if (!arg.value.empty() && arg.value.front() == '[') {
    if (variant.python_repr_sequences) {
      try {
        AppendPythonRepr(output, gufo::json::parse(arg.value));
      } catch (const std::exception&) {
        output.append(PythonJsonSpacing(arg.value));
      }
    } else {
      output.append(PythonJsonSpacing(arg.value));
    }
  } else {
    output.append(arg.value);
  }
}

std::string_view Lfm2RoleName(ChatRole role) noexcept {
  switch (role) {
    case ChatRole::kSystem:
    case ChatRole::kDeveloper:
      return "system";
    case ChatRole::kUser:
      return "user";
    case ChatRole::kAssistant:
      return "assistant";
    case ChatRole::kTool:
      return "tool";
  }
  return "user";
}

}  // namespace

const Lfm2TemplateVariant* FindLfm2TemplateVariant(
    std::string_view sha256) noexcept {
  for (const auto& variant : kLfm2TemplateVariants) {
    if (variant.sha256 == sha256) {
      return &variant;
    }
  }
  return nullptr;
}

const Lfm2TemplateVariant* Lfm2TemplateVariantForGguf(
    const core::GgufReader& reader) {
  std::string err;
  auto tpl = QwenChatTemplate::CreateFromGguf(reader, &err);
  if (tpl == nullptr) {
    return nullptr;
  }
  return FindLfm2TemplateVariant(tpl->GetTemplateSha256());
}

std::optional<std::string> Lfm2Render(std::span<const ChatMessage> messages,
                                      std::span<const ChatTool> tools,
                                      const ChatTemplateOptions& options,
                                      std::string* error_msg,
                                      std::vector<ContentSpan>* content_spans,
                                      const Lfm2TemplateVariant& variant) {
  if (content_spans != nullptr) {
    content_spans->clear();
  }
  for (std::size_t i = 0; i < messages.size(); ++i) {
    if (!messages[i].images.empty()) {
      if (error_msg != nullptr) {
        *error_msg = "lfm2 vision is not supported";
      }
      return std::nullopt;
    }
    if (messages[i].role == ChatRole::kDeveloper && i != 0) {
      if (error_msg != nullptr) {
        *error_msg =
            "lfm2 does not support a developer message after the first "
            "message";
      }
      return std::nullopt;
    }
  }

  std::string out = "<|startoftext|>";
  std::span<const ChatMessage> rest(messages);
  std::string sys;
  if (!rest.empty() && (rest[0].role == ChatRole::kSystem ||
                        rest[0].role == ChatRole::kDeveloper)) {
    sys = rest[0].content;
    rest = rest.subspan(1);
  }
  if (!tools.empty()) {
    if (!sys.empty()) {
      sys.push_back('\n');
    }
    sys.append("List of tools: [");
    bool first_tool = true;
    for (const ChatTool& tool : tools) {
      if (!first_tool) {
        sys.append(", ");
      }
      first_tool = false;
      AppendToolJson(sys, tool);
    }
    sys.push_back(']');
    if (options.require_tool_call) {
      // LFM2: gufo extension, the pinned template has no "required"; same
      // sentence as the Qwen renderer.
      sys.append(
          "\nYou must call at least one available function. Do not answer the "
          "user directly.");
    }
  }
  if (!sys.empty()) {
    out.append("<|im_start|>system\n");
    // HTC (RVW C4): the tool list is client text too, so the whole system
    // turn body is content: a special-token spelling inside a tool
    // description stays characters. It holds no framing token of its own, so
    // a normal render tokenizes exactly as before.
    AppendContent(out, content_spans, sys);
    out.append("<|im_end|>\n");
  }

  long last_user = -1;
  long last_assistant = -1;
  for (std::size_t i = 0; i < rest.size(); ++i) {
    if (rest[i].role == ChatRole::kUser) {
      last_user = static_cast<long>(i);
    }
    if (rest[i].role == ChatRole::kAssistant) {
      last_assistant = static_cast<long>(i);
    }
  }

  for (std::size_t i = 0; i < rest.size(); ++i) {
    const ChatMessage& message = rest[i];
    out.append("<|im_start|>");
    out.append(Lfm2RoleName(message.role));
    out.push_back('\n');
    if (message.role == ChatRole::kAssistant) {
      // The template's preserve_thinking defaults to false: only an
      // assistant after the last user turn keeps its thinking.
      const bool keep =
          variant.keep_thinking == Lfm2KeepThinking::kAfterLastUser
              ? static_cast<long>(i) > last_user
              : static_cast<long>(i) == last_assistant;
      if (keep && !message.thought.empty()) {
        out.append("<think>");
        AppendContent(out, content_spans, message.thought);
        out.append("</think>");
      }
      std::string content = message.content;
      if (!keep) {
        const auto cut = content.rfind("</think>");
        if (cut != std::string::npos) {
          content = std::string(TrimView(content.substr(cut + 8)));
        }
      }
      AppendContent(out, content_spans, content);
      if (!message.tool_calls.empty()) {
        out.append("<|tool_call_start|>[");
        bool first_call = true;
        for (const auto& call : message.tool_calls) {
          if (!first_call) {
            out.append(", ");
          }
          first_call = false;
          // HTC (RVW C4): a replayed call (name, keys, values) is client
          // history, so it is content; only <|tool_call_start|>/end frame.
          std::string call_text = call.name;
          call_text.push_back('(');
          bool first_arg = true;
          for (const auto& arg : call.arguments) {
            if (!first_arg) {
              call_text.append(", ");
            }
            first_arg = false;
            call_text.append(arg.name);
            call_text.push_back('=');
            AppendArgValue(call_text, arg, variant);
          }
          call_text.push_back(')');
          AppendContent(out, content_spans, call_text);
        }
        out.push_back(']');
        out.append("<|tool_call_end|>");
      }
      out.append("<|im_end|>\n");
      continue;
    }
    AppendContent(out, content_spans, message.content);
    out.append("<|im_end|>\n");
  }

  if (options.add_generation_prompt) {
    if (options.require_tool_call) {
      // LFM2: gufo extension, the pinned template has no "required"; prefill
      // past thinking into a tool call so a required call cannot be skipped.
      // Super ROC: stop at <|tool_call_start|>. The vocabulary merges "[" and
      // "(" with the following letters ("[get", "[search", "(x"), so a
      // prefilled "[" or "name(" left the model on a token boundary it never
      // saw and it picked the wrong tool; the tool grammar writes them now.
      // Super ROC: and close a think block only where the template opens one
      // (24B): an empty "<think></think>" in front of the call made the 1.2B
      // (whose template never opens it) call the same tool for everything.
      out.append("<|im_start|>assistant\n");
      if (variant.generation_opens_think) {
        out.append("<think></think>");
      }
      out.append("<|tool_call_start|>");
    } else {
      if (variant.generation_opens_think) {
        out.append("<|im_start|>assistant\n<think>");
        if (!options.enable_thinking) {
          // gufo extension: the pinned template has no think-off switch.
          out.append("</think>");
        }
      } else {
        out.append("<|im_start|>assistant\n");
        if (!options.enable_thinking) {
          out.append("<think></think>");
        }
      }
    }
  }
  return out;
}

std::optional<std::vector<TokenId>> Lfm2RenderAndTokenize(
    const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
    std::span<const ChatTool> tools, const ChatTemplateOptions& options,
    std::string* error_msg, const Lfm2TemplateVariant& variant) {
  std::vector<ContentSpan> spans;
  auto rendered =
      Lfm2Render(messages, tools, options, error_msg, &spans, variant);
  if (!rendered.has_value()) {
    return std::nullopt;
  }
  TokenizerOptions tok_opts;
  tok_opts.add_bos = false;
  tok_opts.add_eos = false;
  return QwenChatTemplate::EncodeRendered(tokenizer, *rendered, 0,
                                          rendered->size(), spans, tok_opts);
}

}  // namespace gufo::tokenization
