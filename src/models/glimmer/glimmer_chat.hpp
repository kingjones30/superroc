#ifndef GUFO_MODELS_GLIMMER_GLIMMER_CHAT_HPP_
#define GUFO_MODELS_GLIMMER_GLIMMER_CHAT_HPP_

#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen/chat_template.hpp"

namespace gufo::tokenization {

/// Pinned sha256 of tokenizer.chat_template in both Muse Glimmer GGUFs.
inline constexpr std::string_view kGlimmerChatTemplateSha256 =
    "114f55ebdc1804c1af371197b9fdf2d6bb925966c9dfe46b73782a71bc07965e";

/// GLIMMER: Onyx ATEM render. Tool definitions are rendered (the pinned
/// template's render_tool_defs macro); tool-role messages and assistant
/// tool_calls fail closed, since their history is not rendered.
/// Encode must keep add_bos false: the template already emits bos_token.
[[nodiscard]] std::optional<std::string> GlimmerRender(  // audit J1
    const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
    std::span<const ChatTool> tools, const ChatTemplateOptions& options = {},
    std::string* error_msg = nullptr,
    std::vector<ContentSpan>* content_spans = nullptr);

[[nodiscard]] std::optional<std::vector<TokenId>> GlimmerRenderAndTokenize(
    const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
    std::span<const ChatTool> tools, const ChatTemplateOptions& options = {},
    std::string* error_msg = nullptr);

}  // namespace gufo::tokenization

#endif  // GUFO_MODELS_GLIMMER_GLIMMER_CHAT_HPP_
