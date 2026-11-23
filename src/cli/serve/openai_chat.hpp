#ifndef SUPERROC_SERVER_OPENAI_CHAT_HPP_
#define SUPERROC_SERVER_OPENAI_CHAT_HPP_

#include "src/cli/serve/http_server.hpp"
#include "src/core/image.hpp"

namespace superroc::server {

/// Model-name policy for text requests. Off (default): a request whose
/// "model" differs from the served model ID gets 404 model_not_found. On
/// (`serve llm --accept-any-model-name`): any "model" string is accepted, as
/// llama-server does, for launchers that forward their own model names.
void SetAcceptAnyModelName(bool accept);
[[nodiscard]] bool AcceptAnyModelName();

/// Handles the supported OpenAI Chat Completions subset. Streaming responses
/// consume scheduler-published token pieces from HttpResponse::streaming_body;
/// socket writes never own or execute model state.
HttpResponse HandleOpenAiChat(const HttpRequest& request,
                              TextGenerationBackend& backend);

/// Request fields the Chat Completions endpoint accepts, mirroring
/// ParseRequest (plus the sampling fields ParseSamplingConfig reads). The
/// /props support list and the parser-acceptance test share it.
std::vector<std::string_view> ChatAcceptedFields();

/// Responses nests output schema under text.format and effort under reasoning.
/// Shared controls stay validated as in Chat Completions, but the Responses
/// adapter tolerates standard Responses fields with no native effect (hosted
/// tool types, include, reasoning.summary, text.verbosity) rather than reject
/// the whole request, because conforming Responses clients routinely send them.
std::optional<HttpResponse> ParseOpenAiResponseControls(const json::Value& body,
                                                        ChatRequest* chat);
bool ParseOpenAiResponseMessage(const json::Value& item,
                                tokenization::ChatMessage* message,
                                core::ImageReadBudget& budget,
                                std::string* error);
/// Shared effort names; each API applies its own thinking and alias rules.
std::optional<ReasoningEffort> ParseReasoningEffortName(std::string_view value);

/// Messages tools and tool_choice, mapped onto the Chat tool declarations so
/// framing, schema constraints and replayed tool turns match Chat.
std::optional<HttpResponse> ParseAnthropicToolControls(const json::Value& body,
                                                       ChatRequest* chat);
/// One Messages image block (base64 or HTTPS URL source), appended to the
/// message at the current end of its text.
bool ParseAnthropicImage(const json::Value& block,
                         tokenization::ChatMessage* message,
                         core::ImageReadBudget& budget, std::string* error);
/// One Messages turn with tool_use or tool_result blocks. Each tool_result
/// becomes a tool message, with its images; text and images in the same user
/// turn follow as a user message.
bool ParseAnthropicToolMessage(const json::Value& item,
                               std::vector<tokenization::ChatMessage>* messages,
                               core::ImageReadBudget& budget,
                               std::string* error);
/// Responses text output uses the same reasoning/UTF-8 filter and scheduler
/// as Chat Completions, including streaming cancellation and cache retention.
HttpResponse CreateOpenAiResponse(const HttpRequest& request,
                                  TextGenerationBackend& backend,
                                  const ChatRequest& chat,
                                  std::size_t max_tokens,
                                  const sampling::SamplingConfig& sampling,
                                  bool stream);
/// Messages output through the same path: content blocks when buffered,
/// Anthropic SSE events when streamed.
HttpResponse CreateAnthropicMessage(const HttpRequest& request,
                                    TextGenerationBackend& backend,
                                    const ChatRequest& chat,
                                    std::size_t max_tokens,
                                    const sampling::SamplingConfig& sampling,
                                    bool stream);

}  // namespace superroc::server

#endif  // SUPERROC_SERVER_OPENAI_CHAT_HPP_
