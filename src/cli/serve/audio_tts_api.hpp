#ifndef SUPERROC_SERVER_AUDIO_TTS_API_HPP_
#define SUPERROC_SERVER_AUDIO_TTS_API_HPP_

#include <string_view>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/tts_service.hpp"

namespace superroc::server {

inline constexpr std::string_view kAudioTtsApiSchema =
    "superroc.audio-tts-api.v1";

[[nodiscard]] bool IsAudioTtsApiPath(std::string_view path) noexcept;
[[nodiscard]] HttpResponse HandleAudioTtsApiRequest(const HttpRequest& request,
                                                    TtsService& service);
[[nodiscard]] HttpResponse ValidateAudioTtsRequest(const HttpRequest& request,
                                                   TtsService& service);

}  // namespace superroc::server

#endif  // SUPERROC_SERVER_AUDIO_TTS_API_HPP_
