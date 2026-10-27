#ifndef SUPERROC_SERVER_AUDIO_WEBSOCKET_HPP_
#define SUPERROC_SERVER_AUDIO_WEBSOCKET_HPP_

#include "src/cli/serve/http_server.hpp"

namespace superroc::server {
HttpResponse HandleTtsWebSocket(const HttpRequest& request,
                                TtsService& service);
HttpResponse HandleAsrWebSocket(const HttpRequest& request,
                                AsrService& service);
}  // namespace superroc::server

#endif  // SUPERROC_SERVER_AUDIO_WEBSOCKET_HPP_
