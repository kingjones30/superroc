#include "src/models/ling3/chat_template.hpp"

#include <string>
#include <utility>

namespace superroc::models::ling3::chat {
namespace {

/// The fixed instruction block the template emits between </tools> and the
/// thinking-option line. Verbatim from the model template.
constexpr const char* kToolsPreamble =
    "# Tools\n\nYou may call one or more functions to assist with the user "
    "query.\n\nYou are provided with function signatures within "
    "<tools></tools> "
    "XML tags:\n<tools>";
constexpr const char* kToolsPostamble =
    "\n</tools>\n\nIf none of the functions can be used, point it out. If the "
    "given question lacks the parameters required by the function, also point "
    "it out.\nIf you need to use a function, for each function call, output "
    "the function name and arguments within the following XML format:\n"
    "<\xE2\x80\x8Btool_call>{function-name}\n"
    "<arg_key>{arg-key-1}</arg_key>\n"
    "<arg_value>{arg-value-1}</arg_value>\n...\n"
    "</\xE2\x80\x8Btool_call>\n";

/// The system turn is only recognised when it already states the thinking mode.
bool StatesThinking(const std::string& content) {
  return content.find("detailed thinking on") != std::string::npos ||
         content.find("detailed thinking off") != std::string::npos;
}

std::string Escaper(const std::string& s) {
  return s;
}

}  // namespace

std::string Tool::to_json() const {
  // Hand-built rather than pulled from superroc's JSON writer so the key order
  // is fixed: the template embeds this verbatim, and a different order would
  // change the token ids of the prompt.
  std::string out = "{\"type\": \"function\", \"function\": {\"name\": \"";
  out += Escaper(name);
  out += "\", \"description\": \"";
  out += Escaper(description);
  out += "\", \"parameters\": ";
  out += parameters_json.empty() ? std::string("{}") : parameters_json;
  out += "}}";
  return out;
}

std::optional<std::string> RenderChat(const std::vector<Message>& messages,
                                      const std::vector<Tool>& tools,
                                      bool enable_thinking,
                                      bool add_generation_prompt,
                                      std::string* error) {
  const std::string thinking = enable_thinking ? "on" : "off";
  std::string out;

  auto fail = [&](std::string message) {
    if (error != nullptr) {
      *error = std::move(message);
    }
    return std::nullopt;
  };

  // The template emits <role>SYSTEM</role> unconditionally, even with no system
  // message at all, and then appends the first message's content only when it
  // really is a system turn. Reproducing that is not cosmetic: it decides
  // whether the prompt starts with the tools block at all.
  out += "<role>SYSTEM</role>";
  const bool has_system = !messages.empty() && messages[0].role == "system";
  if (has_system) {
    out += messages[0].content;
  }
  out += '\n';
  if (!tools.empty()) {
    out += kToolsPreamble;
    for (const Tool& t : tools) {
      out += '\n';
      out += t.to_json();
    }
    out += kToolsPostamble;
  }
  if (has_system && StatesThinking(messages[0].content)) {
    out += markers::kRoleEnd;
  } else {
    out += "detailed thinking ";
    out += thinking;
    out += markers::kRoleEnd;
  }

  for (std::size_t i = has_system ? 1 : 0; i < messages.size(); ++i) {
    const Message& m = messages[i];

    if (m.role == "system") {
      // A later system turn is rendered inline; the *first* one is not,
      // because the header below already consumed it.
      if (i != 0) {
        out += "<role>SYSTEM</role>";
        out += m.content;
        out += markers::kRoleEnd;
      }
      continue;
    }

    if (m.role == "user") {
      out += "<role>HUMAN</role>";
      out += m.content;
      out += markers::kRoleEnd;
      continue;
    }

    if (m.role == "assistant") {
      // The template preserves a prior reasoning block when one is present,
      // either explicitly or recovered from <think>...</think> inside content.
      std::string reasoning = m.reasoning;
      std::string content = m.content;
      if (reasoning.empty()) {
        const auto close = content.find(markers::kThinkClose);
        if (close != std::string::npos) {
          const auto open = content.rfind(markers::kThinkOpen, close);
          const std::string inner = content.substr(
              open == std::string::npos ? 0 : open + 7,
              close - (open == std::string::npos ? 0 : open + 7));
          std::string trimmed = inner;
          while (!trimmed.empty() &&
                 (trimmed.front() == '\n' || trimmed.front() == '\r')) {
            trimmed.erase(trimmed.begin());
          }
          while (!trimmed.empty() &&
                 (trimmed.back() == '\n' || trimmed.back() == '\r')) {
            trimmed.pop_back();
          }
          reasoning = trimmed;
          content = content.substr(close + 8);
          while (!content.empty() && (content.front() == '\n')) {
            content.erase(content.begin());
          }
        }
      }

      out += "<role>ASSISTANT</role>";
      if (!reasoning.empty()) {
        std::string body = reasoning;
        while (!body.empty() &&
               (body.front() == '\n' || body.front() == '\r')) {
          body.erase(body.begin());
        }
        while (!body.empty() && (body.back() == '\n' || body.back() == '\r')) {
          body.pop_back();
        }
        out += '\n';
        out += markers::kThinkOpen;
        out += body;
        out += markers::kThinkClose;
        while (!content.empty() && content.front() == '\n') {
          content.erase(content.begin());
        }
        out += content;
      } else {
        out += '\n';
        out += markers::kThinkOpen;
        out += markers::kThinkClose;
        out += content;
      }

      for (std::size_t c = 0; c < m.tool_calls.size(); ++c) {
        const ToolCall& tc = m.tool_calls[c];
        if (tc.name.empty()) {
          return fail("assistant tool call has no function name");
        }
        if ((c == 0 && !content.empty()) || c != 0) {
          out += '\n';
        }
        out += markers::kToolCallOpen;
        out += tc.name;
        for (const ToolCallArgument& a : tc.arguments) {
          out += "<arg_key>";
          out += a.key;
          out += "</arg_key>";
          out += "\n<arg_value>";
          out += a.value;
          out += "</arg_value>";
        }
        out += '\n';
        out += markers::kToolCallClose;
      }
      out += markers::kRoleEnd;
      continue;
    }

    if (m.role == "tool") {
      // Consecutive tool results share one OBSERVATION turn.
      const bool first = i == 0 || messages[i - 1].role != "tool";
      const bool last =
          i + 1 == messages.size() || messages[i + 1].role != "tool";
      if (first) {
        out += "<role>OBSERVATION</role>";
      }
      out += "\n<tool_response>\n";
      out += m.content;
      out += "\n</tool_response>";
      if (last) {
        out += markers::kRoleEnd;
      }
      continue;
    }

    return fail("unsupported chat role: '" + m.role + "'");
  }

  if (add_generation_prompt) {
    out += "<role>ASSISTANT</role>";
    if (enable_thinking) {
      out += '\n';
      out += markers::kThinkOpen;
    } else {
      out += '\n';
      out += markers::kThinkOpen;
      out += markers::kThinkClose;
    }
  }

  return out;
}

}  // namespace superroc::models::ling3::chat
