#ifndef SUPERROC_MODELS_QWEN_LFM2_CHAT_HPP_
#define SUPERROC_MODELS_QWEN_LFM2_CHAT_HPP_

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace superroc::tokenization {

// LFM2: pinned sha256 of tokenizer.chat_template in the LFM2.5 GGUFs.
inline constexpr std::string_view kLfm2ChatTemplateSha256 =
    "ea663864491de7ade391839479860ca95541f892f72665c73251fbd4643b1bef";

/// Super ROC: LFM2 chat-template variants rendered natively, keyed by the
/// GGUF template's SHA-256. Each switch is a difference Jay found between
/// the three templates (verified against Jinja references).
enum class Lfm2KeepThinking : std::uint8_t {
  kAfterLastUser,      // assistants after the last user turn keep thinking
  kLastAssistantOnly,  // only the last assistant message keeps thinking
};
struct Lfm2TemplateVariant {
  std::string_view sha256;
  std::string_view name;
  bool escape_string_args;      // strings: escape \ ' LF CR inside '...'
  bool python_repr_sequences;   // lists (and dicts inside them): Python repr
  bool generation_opens_think;  // generation prompt ends "assistant\n<think>"
  Lfm2KeepThinking keep_thinking;
};
inline constexpr std::array<Lfm2TemplateVariant, 3> kLfm2TemplateVariants{{
    {kLfm2ChatTemplateSha256, "lfm2.5-24b-a2b", true, false, true,
     Lfm2KeepThinking::kAfterLastUser},
    {"c5b67247c9f736f6c40eaa91ae1ef27dc67f5b041eedaadb97a1bea5030fb0ff",
     "lfm2.5-8b-a1b", false, true, false, Lfm2KeepThinking::kAfterLastUser},
    {"ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6",
     "lfm2.5-1.2b-instruct", true, false, false,
     Lfm2KeepThinking::kLastAssistantOnly},
}};
/// nullptr = not an LFM2 template Super ROC renders.
[[nodiscard]] const Lfm2TemplateVariant* FindLfm2TemplateVariant(
    std::string_view sha256) noexcept;
/// The variant of a GGUF whose template passes the LFM2 pin (nullptr if none).
[[nodiscard]] const Lfm2TemplateVariant* Lfm2TemplateVariantForGguf(
    const core::GgufReader& reader);

/// LFM2: render of the pinned lfm2moe chat template. Message text, the tool
/// list and replayed tool calls are content (spelled control tokens stay
/// text); turn and tool-call markers are framing.
/// Encode must keep add_bos/add_eos false: the render already starts with the
/// BOS text.
[[nodiscard]] std::optional<std::string> Lfm2Render(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options = {}, std::string* error_msg = nullptr,
    std::vector<ContentSpan>* content_spans = nullptr,
    const Lfm2TemplateVariant& variant = kLfm2TemplateVariants[0]);

[[nodiscard]] std::optional<std::vector<TokenId>> Lfm2RenderAndTokenize(
    const QwenTokenizer& tokenizer, std::span<const ChatMessage> messages,
    std::span<const ChatTool> tools, const ChatTemplateOptions& options = {},
    std::string* error_msg = nullptr,
    const Lfm2TemplateVariant& variant = kLfm2TemplateVariants[0]);

}  // namespace superroc::tokenization

#endif  // SUPERROC_MODELS_QWEN_LFM2_CHAT_HPP_
