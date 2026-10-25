// ling3-chat-template test — runs on the Mac, no gufo_core needed.
//
// Renders the gate-(b) prompt and a few shapes, and checks the rendering
// against the expected strings derived from the model's own
// tokenizer.chat_template. The point is the byte-significant parts: the
// zero-width space inside the tool-call markers, and `<think></think>` when
// thinking is off.

#include <cstdio>
#include <string>
#include <vector>

#include "src/models/ling3/chat_template.hpp"

namespace chat = gufo::models::ling3::chat;

namespace {

int failures = 0;

void Check(const char* name, const std::string& got, const std::string& want) {
  if (got == want) {
    std::printf("ok    %s\n", name);
    return;
  }
  ++failures;
  std::printf("FAIL  %s\n  got : [%s]\n  want: [%s]\n", name, got.c_str(),
              want.c_str());
}

bool Contains(const char* name, const std::string& hay, const std::string& needle) {
  if (hay.find(needle) != std::string::npos) {
    std::printf("ok    %s (contains %s)\n", name, needle.c_str());
    return true;
  }
  ++failures;
  std::printf("FAIL  %s: missing [%s]\n  in [%s]\n", name, needle.c_str(),
              hay.c_str());
  return false;
}

}  // namespace

int main() {
  // --- the gate (b) prompt: one tool, thinking off -------------------------
  std::vector<chat::Tool> tools;
  chat::Tool weather;
  weather.name = "get_weather";
  weather.description = "Get the current weather for a city";
  weather.parameters_json =
      "{\"type\": \"object\", \"properties\": {\"city\": {\"type\": "
      "\"string\", \"description\": \"The city name\"}}, \"required\": "
      "[\"city\"]}";
  tools.push_back(weather);
  std::vector<chat::Message> msgs;
  msgs.push_back({"user", "weather in Paris?", "", {}});

  std::string err;
  auto rendered = chat::RenderChat(msgs, tools, /*enable_thinking=*/false,
                                   /*add_generation_prompt=*/true, &err);
  if (!rendered.has_value()) {
    std::printf("FAIL  gate (b) render: %s\n", err.c_str());
    return 1;
  }
  const std::string p = *rendered;

  // The whole thing must be exactly:
  //   <role>SYSTEM</role>\n# Tools\n\n...<tools>\n{json}\n</tools>\n\n...
  //   detailed thinking off<|role_end|><role>HUMAN</role>weather in Paris?
  //   <|role_end|><role>ASSISTANT</role>\n<think></think>
  Check("gate(b) begins with SYSTEM + newline",
        p.substr(0, std::string("<role>SYSTEM</role>\n").size()),
        "<role>SYSTEM</role>\n");
  Contains("gate(b) tools preamble", p, "# Tools\n\nYou may call one or more");
  Contains("gate(b) tool json", p,
           "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\"");
  Contains("gate(b) thinking off", p, "detailed thinking off<|role_end|>");
  Contains("gate(b) human turn", p,
           "<role>HUMAN</role>weather in Paris?<|role_end|>");
  Contains("gate(b) generation prompt (thinking off)", p,
           "<role>ASSISTANT</role>\n<think></think>");
  Contains("gate(b) zero-width tool_call marker", p,
           "<\xE2\x80\x8Btool_call>{function-name}");
  Contains("gate(b) zero-width /tool_call marker", p,
           "</\xE2\x80\x8Btool_call>");
  if (p.find("<tool_call>") != std::string::npos) {
    ++failures;
    std::printf("FAIL  gate(b): a plain (no zero-width space) tool_call marker "
                "leaked in\n");
  } else {
    std::printf("ok    gate(b): no plain tool_call marker\n");
  }

  // --- thinking on ----------------------------------------------------------
  auto on = chat::RenderChat(msgs, tools, /*enable_thinking=*/true, true, &err);
  if (!on.has_value()) {
    std::printf("FAIL  thinking-on render: %s\n", err.c_str());
    ++failures;
  } else {
    Contains("thinking on: generation prompt", *on,
             "<role>ASSISTANT</role>\n<think>");
    Contains("thinking on: system line", *on, "detailed thinking on<|role_end|>");
    if (on->find("<think></think>") != std::string::npos) {
      ++failures;
      std::printf("FAIL  thinking on: emitted an empty <think></think>\n");
    }
  }

  // --- plain chat, no tools, no generation prompt --------------------------
  std::vector<chat::Message> simple = {{"user", "hi", "", {}},
                                       {"assistant", "hello", "", {}}};
  auto s1 = chat::RenderChat(simple, {}, false, false, &err);
  if (!s1.has_value()) {
    std::printf("FAIL  simple render: %s\n", err.c_str());
    ++failures;
  } else {
    // No system message, but the template still opens a SYSTEM turn and closes
    // it with the thinking option.
    Check("simple two-turn",
          *s1,
          "<role>SYSTEM</role>\ndetailed thinking off<|role_end|>"
          "<role>HUMAN</role>hi<|role_end|>"
          "<role>ASSISTANT</role>\n<think></think>hello<|role_end|>");
  }

  // --- assistant tool call --------------------------------------------------
  std::vector<chat::Message> withcall = {{"user", "weather in Paris?", "", {}}};
  chat::Message a;
  a.role = "assistant";
  a.content = "";
  a.tool_calls.push_back({"get_weather", {{"city", "Paris"}}});
  withcall.push_back(a);
  auto s2 = chat::RenderChat(withcall, tools, false, true, &err);
  if (!s2.has_value()) {
    std::printf("FAIL  tool-call render: %s\n", err.c_str());
    ++failures;
  } else {
    Contains("assistant tool_call open", *s2,
             "<\xE2\x80\x8Btool_call>get_weather");
    Contains("assistant arg_key", *s2, "<arg_key>city</arg_key>");
    Contains("assistant arg_value", *s2, "\n<arg_value>Paris</arg_value>");
    Contains("assistant tool_call close", *s2,
             "\n</\xE2\x80\x8Btool_call><|role_end|>");
  }

  // --- tool observation -----------------------------------------------------
  std::vector<chat::Message> obs = {{"user", "weather in Paris?", "", {}},
                                    {"tool", "{\"temp\": 18}", "", {}}};
  auto s3 = chat::RenderChat(obs, {}, false, true, &err);
  if (!s3.has_value()) {
    std::printf("FAIL  observation render: %s\n", err.c_str());
    ++failures;
  } else {
    Contains("tool observation", *s3,
             "<role>OBSERVATION</role>\n<tool_response>\n{\"temp\": 18}"
             "\n</tool_response><|role_end|>");
  }

  // --- an unknown role must fail closed ------------------------------------
  std::vector<chat::Message> bad = {{"wizard", "hi", "", {}}};
  err.clear();
  auto s4 = chat::RenderChat(bad, {}, false, true, &err);
  if (s4.has_value()) {
    ++failures;
    std::printf("FAIL  unknown role: RenderChat returned a value instead of "
                "failing closed\n");
  } else {
    std::printf("ok    unknown role fails closed: %s\n", err.c_str());
  }

  std::printf("\n%s: %d failures\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
