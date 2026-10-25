#include "src/models/glimmer/glimmer_chat.hpp"

#include <algorithm>
#include <ctime>
#include <string_view>
#include <vector>

namespace gufo::tokenization {
namespace {

constexpr std::string_view kToolError =
    "muse-glimmer tool-result and assistant tool-call history are not rendered";

void AppendJsonString(std::string* out, std::string_view value) {
  out->push_back('"');
  for (const unsigned char c : value) {
    switch (c) {
      case '"':
      case '\\':
        out->push_back('\\');
        out->push_back(static_cast<char>(c));
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
      default:
        out->push_back(static_cast<char>(c));
        break;
    }
  }
  out->push_back('"');
}

// First dotted component, matching the pinned template's fn.name.split('.')[0].
std::string ToolNamespace(std::string_view name) {
  const auto dot = name.find('.');
  return std::string(name.substr(0, dot));
}

std::vector<std::string> ToolNamespaces(std::span<const ChatTool> tools) {
  std::vector<std::string> namespaces;
  for (const ChatTool& tool : tools) {
    std::string ns = ToolNamespace(tool.name);
    if (std::ranges::find(namespaces, ns) == namespaces.end()) {
      namespaces.push_back(std::move(ns));
    }
  }
  return namespaces;
}

// Text of the pinned template's render_tool_defs macro.
void AppendToolDefs(std::string* out, std::span<const ChatTool> tools) {
  out->append(
      "In this environment you have access to a set of tools you can use to "
      "answer the user's question.\n\n"
      "You can invoke a function by writing a \"<atem:function_calls>\" block "
      "like the following:\n"
      "<atem:function_calls>\n"
      "<atem:invoke name=\"$FUNCTION_NAME\">\n"
      "<atem:parameter "
      "name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</atem:parameter>\n"
      "...\n"
      "</atem:invoke>\n"
      "</atem:function_calls>\n\n"
      "String and scalar parameters should be specified as is, while lists and "
      "objects should use JSON format. Note that spaces for string values are "
      "not stripped. The output is not expected to be valid XML and is parsed "
      "with regular expressions.\n"
      "Here are the functions available in JSONSchema format:\n"
      "// Tool metadata\n");
  for (const std::string& ns : ToolNamespaces(tools)) {
    out->append("{\"name\": ");
    AppendJsonString(out, ns);
    out->append(", \"description\": \"\"}\n");
  }
  out->append("// Function schemas");
  for (const ChatTool& tool : tools) {
    out->append("\n{\"name\": ");
    AppendJsonString(out, tool.name);
    out->append(", \"description\": ");
    AppendJsonString(out, tool.description);
    out->append(", \"parameters\": ");
    out->append(tool.parameters_json.empty() ? "{}" : tool.parameters_json);
    out->push_back('}');
  }
  out->append(
      "\n\nHere's an example of how to call a function in the tool set:\n"
      "(If the tool namespace is not specified, invoke the function directly "
      "as `example_function_name` rather than "
      "`example_tool_name.example_function_name`)\n\n"
      "to=example_tool_name.example_function_name\n\n"
      "<atem:function_calls>\n"
      "<atem:invoke name=\"example_tool_name.example_function_name\">\n"
      "<atem:parameter name=\"example_parameter_1\">value_1</atem:parameter>\n"
      "<atem:parameter name=\"example_parameter_2\">This is the value for the "
      "second parameter\n"
      "that can span\n"
      "\"multiple\" lines\n"
      "</atem:parameter>\n"
      "</atem:invoke>\n"
      "</atem:function_calls>");
}

std::string_view ReasoningStrength(QwenReasoningEffort effort) noexcept {
  switch (effort) {
    case QwenReasoningEffort::kLow:
      return "low";
    case QwenReasoningEffort::kMedium:
      return "medium";
    case QwenReasoningEffort::kXHigh:
      // The pinned template's default is "high". Qwen's xhigh is that default.
      return "high";
  }
  return "high";
}

// GLM: llama.cpp's strftime_now('%Y-%m-%d') on the default system message.
void AppendCurrentDate(std::string* out) {
  const std::time_t now = std::time(nullptr);
  std::tm local{};
  if (localtime_r(&now, &local) == nullptr) {
    return;
  }
  char date[16];
  if (std::strftime(date, sizeof(date), "%Y-%m-%d", &local) == 0) {
    return;
  }
  out->append("\nCurrent date: ");
  out->append(date);
  out->push_back('.');
}

void AppendSystemTail(std::string* out, const ChatTemplateOptions& options,
                      std::span<const ChatTool> tools) {
  out->append("\n\n");
  out->append("Reasoning strength: ");
  // GLM: the pinned template uses reasoning_strength, else "high".
  if (!options.reasoning_strength.empty()) {
    out->append(options.reasoning_strength);
  } else {
    out->append(ReasoningStrength(options.reasoning_effort));
  }
  out->push_back('.');
  if (!tools.empty()) {
    out->append("\n\n");
    AppendToolDefs(out, tools);
  }
  out->append("\n\n# Valid recipients: \"self\"");
  for (const std::string& ns : ToolNamespaces(tools)) {
    out->append(", \"");
    out->append(ns);
    out->append(".*\"");
  }
  out->append(", \"user\".");
  out->append("<|eot|>");
}

// audit J1: message text is content (spelled control tokens stay text),
// tokenized without special matching -- same pattern as AppendContent in
// chat_template.cpp. Framing strings are unchanged.
void AppendContent(std::string& out, std::vector<ContentSpan>* spans,
                   std::string_view text) {
  const auto begin = out.size();
  out.append(text);
  if (spans != nullptr && !text.empty()) {
    spans->push_back({begin, out.size() - begin});
  }
}

}  // namespace

std::optional<std::string> GlimmerRender(  // audit J1
    const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
    std::span<const ChatTool> tools, const ChatTemplateOptions& options,
    std::string* error_msg, std::vector<ContentSpan>* content_spans) {
  for (const ChatMessage& message : messages) {
    if (!message.tool_calls.empty() || message.role == ChatRole::kTool ||
        !message.images.empty()) {
      if (error_msg != nullptr) {
        *error_msg = std::string(kToolError);
      }
      return std::nullopt;
    }
  }

  const TokenId bos = tokenizer.GetBosTokenId();
  if (bos == kInvalidTokenId) {
    if (error_msg != nullptr) {
      *error_msg = "muse-glimmer tokenizer has no bos token";
    }
    return std::nullopt;
  }

  std::string out;
  out.reserve(1024);
  out.append(tokenizer.DecodeToken(bos));

  bool has_system = false;
  for (const ChatMessage& message : messages) {
    if (message.role == ChatRole::kSystem ||
        message.role == ChatRole::kDeveloper) {
      has_system = true;
      break;
    }
  }
  if (!has_system) {
    out.append(
        "<|start|>system<|message|>You are a helpful AI assistant.\n"
        "Knowledge cutoff: 2026-01-04.");
    // GLM: the pinned template inserts today's date before the strength line.
    AppendCurrentDate(&out);
    AppendSystemTail(&out, options, tools);
  }

  for (const ChatMessage& message : messages) {
    if (message.role == ChatRole::kSystem ||
        message.role == ChatRole::kDeveloper) {
      out.append("<|start|>system<|message|>");
      AppendContent(out, content_spans, message.content);  // audit J1
      AppendSystemTail(&out, options, tools);
      continue;
    }
    if (message.role == ChatRole::kUser) {
      out.append("<|start|>user<|message|>");
      AppendContent(out, content_spans, message.content);  // audit J1
      out.append("<|eot|>");
      continue;
    }
    if (message.role == ChatRole::kAssistant) {
      if (!message.thought.empty()) {
        out.append("<|start|>assistant to=self<|message|>");
        AppendContent(out, content_spans, message.thought);  // audit J1
        out.append("<|eom|>");
      }
      out.append("<|start|>assistant to=user<|message|>");
      AppendContent(out, content_spans, message.content);  // audit J1
      out.append("<|eot|>");
    }
  }
  if (options.add_generation_prompt) {
    out.append("<|start|>assistant");
  }
  return out;
}

std::optional<std::vector<TokenId>> GlimmerRenderAndTokenize(
    const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
    std::span<const ChatTool> tools, const ChatTemplateOptions& options,
    std::string* error_msg) {
  std::vector<ContentSpan> spans;  // audit J1
  const auto rendered =
      GlimmerRender(tokenizer, messages, tools, options, error_msg, &spans);
  if (!rendered.has_value()) {
    return std::nullopt;
  }
  TokenizerOptions tok_opts;
  tok_opts.add_bos = false;
  tok_opts.add_eos = false;
  tok_opts.parse_special_tokens = true;
  return QwenChatTemplate::EncodeRendered(tokenizer, *rendered, 0,
                                          rendered->size(), spans, tok_opts);
}

}  // namespace gufo::tokenization
