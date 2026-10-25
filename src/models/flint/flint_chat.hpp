#ifndef GUFO_MODELS_FLINT_FLINT_CHAT_HPP_
#define GUFO_MODELS_FLINT_FLINT_CHAT_HPP_

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen/chat_template.hpp"

namespace gufo::tokenization {

/// Pinned sha256 of ref/chat_template.jinja (RecursiveSI/flint).
inline constexpr std::string_view kFlintChatTemplateSha256 =
    "7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466";

/// Flint: roles <|system|> <|developer|> <|user|> <|assistant|>
/// <|observation|>, reasoning in <think>, GLM-style
/// <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>.
[[nodiscard]] std::optional<std::string> FlintRender(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options = {}, std::string* error_msg = nullptr,
    std::vector<ContentSpan>* content_spans = nullptr);

[[nodiscard]] std::optional<std::vector<TokenId>> FlintRenderAndTokenize(
    const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
    std::span<const ChatTool> tools, const ChatTemplateOptions& options = {},
    std::string* error_msg = nullptr);

/// One GLM-style Flint tool call scanned from the start of `text`.
struct FlintToolCallScan {
  bool matched{false};
  std::size_t end{0};
  std::string name;
  std::vector<ChatMessage::ToolArgument> arguments;
  bool complete{false};
};

[[nodiscard]] FlintToolCallScan ScanFlintToolCall(std::string_view text);

}  // namespace gufo::tokenization

#endif  // GUFO_MODELS_FLINT_FLINT_CHAT_HPP_
