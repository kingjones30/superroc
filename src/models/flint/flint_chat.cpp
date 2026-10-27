#include "src/models/flint/flint_chat.hpp"

#include <algorithm>
#include <utility>

#include "src/core/json.hpp"

namespace superroc::tokenization {
namespace {

constexpr std::string_view kToolsHeader =
    "<|system|>\n"
    "# Tools\n"
    "\n"
    "You may call one or more functions to assist with the user query.\n"
    "\n"
    "You are provided with function signatures within <tools></tools> XML "
    "tags:\n"
    "<tools>\n";

constexpr std::string_view kToolsFooter =
    "</tools>\n"
    "\n"
    "For each function call, output the function name and arguments within the "
    "following XML format:\n"
    "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{"
    "arg-value-1}</arg_value>...</tool_call>\n";

void AppendContent(std::string& out, std::vector<ContentSpan>* spans,
                   std::string_view text) {
  const auto begin = out.size();
  out.append(text);
  if (spans != nullptr && !text.empty()) {
    spans->push_back({begin, out.size() - begin});
  }
}

void ReplaceAll(std::string* s, std::string_view from, std::string_view to) {
  std::size_t pos = 0;
  while ((pos = s->find(from, pos)) != std::string::npos) {
    s->replace(pos, from.size(), to);
    pos += to.size();
  }
}

// flint_text macro: strip NULs and think tags, HTML-escape role markers.
std::string FlintText(std::string_view text) {
  std::string out(text);
  ReplaceAll(&out, std::string_view("\0", 1), "");
  ReplaceAll(&out, "<think>", "");
  ReplaceAll(&out, "</think>", "");
  ReplaceAll(&out, "<|begin_of_text|>", "&lt;|begin_of_text|&gt;");
  ReplaceAll(&out, "<|end_of_text|>", "&lt;|end_of_text|&gt;");
  ReplaceAll(&out, "<|system|>", "&lt;|system|&gt;");
  ReplaceAll(&out, "<|developer|>", "&lt;|developer|&gt;");
  ReplaceAll(&out, "<|user|>", "&lt;|user|&gt;");
  ReplaceAll(&out, "<|assistant|>", "&lt;|assistant|&gt;");
  ReplaceAll(&out, "<|observation|>", "&lt;|observation|&gt;");
  return out;
}

void AppendToolFnJson(std::string& out, const ChatTool& tool) {
  if (!tool.definition_json.empty()) {
    try {
      const json::Value parsed = json::parse(tool.definition_json);
      if (const json::Value* fn = parsed.find("function"); fn != nullptr) {
        out.append(fn->tojson());
        return;
      }
      out.append(parsed.tojson());
      return;
    } catch (const std::exception&) {
    }
  }
  json::Value fn = json::Value::object();
  fn.append_member("name", json::Value(tool.name));
  fn.append_member("description", json::Value(tool.description));
  if (tool.parameters_json.empty()) {
    fn.append_member("parameters", json::Value::object());
  } else {
    try {
      fn.append_member("parameters", json::parse(tool.parameters_json));
    } catch (const std::exception&) {
      fn.append_member("parameters", json::Value::object());
    }
  }
  out.append(fn.tojson());
}

void AppendArgValue(std::string& out, const ChatMessage::ToolArgument& arg) {
  if (arg.is_string) {
    out.append(arg.value);
    return;
  }
  try {
    out.append(json::parse(arg.value).tojson());
  } catch (const std::exception&) {
    out.append(arg.value);
  }
}

void AppendToolPreamble(std::string& out, std::span<const ChatTool> tools) {
  if (tools.empty()) {
    return;
  }
  out.append(kToolsHeader);
  for (const ChatTool& tool : tools) {
    AppendToolFnJson(out, tool);
    out.push_back('\n');
  }
  out.append(kToolsFooter);
}

}  // namespace

std::optional<std::string> FlintRender(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options, std::string* error_msg,
    std::vector<ContentSpan>* content_spans) {
  if (content_spans != nullptr) {
    content_spans->clear();
  }
  for (const ChatMessage& message : messages) {
    if (!message.images.empty()) {
      if (error_msg != nullptr) {
        *error_msg = "flint vision is not supported";
      }
      return std::nullopt;
    }
  }

  std::string out;
  out.reserve(1024);
  bool tools_pending = true;
  for (const ChatMessage& message : messages) {
    if (tools_pending && message.role != ChatRole::kSystem) {
      AppendToolPreamble(out, tools);
      tools_pending = false;
    }
    switch (message.role) {
      case ChatRole::kSystem:
        out.append("<|system|>");
        AppendContent(out, content_spans, FlintText(message.content));
        break;
      case ChatRole::kDeveloper:
        out.append("<|developer|>");
        AppendContent(out, content_spans, FlintText(message.content));
        break;
      case ChatRole::kUser:
        out.append("<|user|>");
        AppendContent(out, content_spans, FlintText(message.content));
        break;
      case ChatRole::kAssistant:
        out.append("<|assistant|>");
        if (!message.thought.empty()) {
          out.append("<think>");
          AppendContent(out, content_spans, message.thought);
          out.append("</think>");
        }
        if (!message.content.empty()) {
          AppendContent(out, content_spans, message.content);
        }
        for (const auto& call : message.tool_calls) {
          out.append("<tool_call>");
          AppendContent(out, content_spans, call.name);
          for (const auto& arg : call.arguments) {
            out.append("<arg_key>");
            AppendContent(out, content_spans, arg.name);
            out.append("</arg_key><arg_value>");
            std::string value;
            AppendArgValue(value, arg);
            AppendContent(out, content_spans, value);
            out.append("</arg_value>");
          }
          out.append("</tool_call>");
        }
        break;
      case ChatRole::kTool:
        out.append("<|observation|><tool_response>");
        AppendContent(out, content_spans, FlintText(message.content));
        out.append("</tool_response>");
        break;
    }
  }
  if (tools_pending) {
    AppendToolPreamble(out, tools);
  }
  if (options.add_generation_prompt) {
    out.append("<|assistant|>");
  }
  return out;
}

std::optional<std::vector<TokenId>> FlintRenderAndTokenize(
    const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
    std::span<const ChatTool> tools, const ChatTemplateOptions& options,
    std::string* error_msg) {
  std::vector<ContentSpan> spans;
  const auto rendered =
      FlintRender(messages, tools, options, error_msg, &spans);
  if (!rendered.has_value()) {
    return std::nullopt;
  }
  TokenizerOptions encode;
  encode.add_bos = false;
  encode.add_eos = false;
  encode.parse_special_tokens = true;
  return QwenChatTemplate::EncodeRendered(tokenizer, *rendered, 0,
                                          rendered->size(), spans, encode);
}

FlintToolCallScan ScanFlintToolCall(std::string_view text) {
  FlintToolCallScan result;
  constexpr std::string_view kOpen = "<tool_call>";
  constexpr std::string_view kClose = "</tool_call>";
  constexpr std::string_view kKeyOpen = "<arg_key>";
  constexpr std::string_view kKeyClose = "</arg_key>";
  constexpr std::string_view kValOpen = "<arg_value>";
  constexpr std::string_view kValClose = "</arg_value>";
  if (!text.starts_with(kOpen)) {
    return result;
  }
  result.matched = true;
  std::string_view body = text.substr(kOpen.size());
  const auto close = body.find(kClose);
  if (close == std::string_view::npos) {
    return result;
  }
  const auto name_end = std::min(body.find(kKeyOpen), close);
  result.name = std::string(body.substr(0, name_end));
  std::string_view rest = body.substr(name_end, close - name_end);
  while (rest.starts_with(kKeyOpen)) {
    rest.remove_prefix(kKeyOpen.size());
    const auto key_end = rest.find(kKeyClose);
    if (key_end == std::string_view::npos) {
      return result;
    }
    ChatMessage::ToolArgument arg;
    arg.name = std::string(rest.substr(0, key_end));
    rest.remove_prefix(key_end + kKeyClose.size());
    if (!rest.starts_with(kValOpen)) {
      return result;
    }
    rest.remove_prefix(kValOpen.size());
    const auto val_end = rest.find(kValClose);
    if (val_end == std::string_view::npos) {
      return result;
    }
    arg.value = std::string(rest.substr(0, val_end));
    arg.is_string = true;
    result.arguments.push_back(std::move(arg));
    rest.remove_prefix(val_end + kValClose.size());
  }
  result.complete = true;
  result.end = kOpen.size() + close + kClose.size();
  return result;
}

}  // namespace superroc::tokenization
