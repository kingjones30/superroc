#ifndef GUFO_MODELS_LING3_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_LING3_CHAT_TEMPLATE_HPP_

#include <optional>
#include <string>
#include <vector>

/// C++ rendering of Ling-3.0's Bailing-V3 chat template.
///
/// Transcribed from the model's own `tokenizer.chat_template` (and the matching
/// `Ling-3.0-flash-*.jinja`), because gufo does not embed a Jinja engine. Only
/// the paths this package needs are implemented; anything else is an explicit
/// error rather than a silent approximation.
///
/// Two details that are easy to get wrong and are byte-significant here:
///   * the tool-call open/close markers contain a **zero-width space** (U+200B)
///     -- `<tool_call>` is not `<tool_call>`. The vocabulary has the
///     zero-width form as a single USER_DEFINED token (156896), so the plain
///     ASCII spelling would not tokenize to it at all;
///   * `<think></think>` is emitted literally when thinking is off, and the
///     system turn always ends with `detailed thinking on|off`.
namespace gufo::models::ling3::chat {

/// A tool exposed to the model. `parameters_json` is the raw JSON of the
/// parameter object; the renderer embeds it verbatim.
struct Tool {
  std::string name;
  std::string description;
  std::string parameters_json;
  /// The OpenAI-style `{"type":"function","function":{...}}` object the
  /// template serialises into <tools>. Built from the fields above when empty.
  std::string to_json() const;
};

struct ToolCallArgument {
  std::string key;
  std::string value;  ///< Emitted as a string; JSON-encode numbers yourself.
};

struct ToolCall {
  std::string name;
  std::vector<ToolCallArgument> arguments;
};

struct Message {
  /// "system", "user", "assistant" or "tool". Anything else is rejected.
  std::string role;
  std::string content;
  /// Only read for assistant turns; empty means "no preserved thinking".
  std::string reasoning;
  std::vector<ToolCall> tool_calls;
};

/// Renders `messages` into the prompt string the model expects.
///
/// `enable_thinking` defaults to true, matching the template's own default
/// (`thinking_option = 'on'`). `add_generation_prompt` appends the assistant
/// opening turn.
///
/// Returns nullopt and fills `error` for a role this template does not define,
/// or for a tool call with no name.
[[nodiscard]] std::optional<std::string> RenderChat(
    const std::vector<Message>& messages, const std::vector<Tool>& tools = {},
    bool enable_thinking = true, bool add_generation_prompt = true,
    std::string* error = nullptr);

/// The marker strings, exposed because the tokenizer needs to recognise them
/// and because they are the easiest thing to corrupt.
namespace markers {
inline constexpr const char* kRoleEnd = "<|role_end|>";
inline constexpr const char* kToolCallOpen =
    "<\xE2\x80\x8Btool_call>";  // U+200B
inline constexpr const char* kToolCallClose =
    "</\xE2\x80\x8Btool_call>";  // U+200B
inline constexpr const char* kThinkOpen = "<think>";
inline constexpr const char* kThinkClose = "</think>";
}  // namespace markers

}  // namespace gufo::models::ling3::chat

#endif  // GUFO_MODELS_LING3_CHAT_TEMPLATE_HPP_
