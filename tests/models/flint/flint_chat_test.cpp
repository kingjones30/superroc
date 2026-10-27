// Flint: chat render equality against fixtures/render.json (jinja2 reference).

#include "src/models/flint/flint_chat.hpp"

#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/json.hpp"

namespace {

namespace json = superroc::json;
using superroc::tokenization::ChatMessage;
using superroc::tokenization::ChatRole;
using superroc::tokenization::ChatTemplateOptions;
using superroc::tokenization::ChatTool;

std::string ReadFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

bool RoleFromString(const std::string& role, ChatRole* out) {
  if (role == "system") {
    *out = ChatRole::kSystem;
  } else if (role == "developer") {
    *out = ChatRole::kDeveloper;
  } else if (role == "user") {
    *out = ChatRole::kUser;
  } else if (role == "assistant") {
    *out = ChatRole::kAssistant;
  } else if (role == "tool") {
    *out = ChatRole::kTool;
  } else {
    return false;
  }
  return true;
}

ChatMessage::ToolArgument MakeArgument(const std::string& name,
                                       const json::Value& value) {
  ChatMessage::ToolArgument arg;
  arg.name = name;
  if (value.is_string()) {
    arg.is_string = true;
    arg.value = value.str();
  } else {
    arg.is_string = false;
    arg.value = value.dump();
  }
  return arg;
}

bool BuildMessage(const json::Value& node, ChatMessage* out,
                  std::string* error) {
  if (!RoleFromString(node.member_str("role"), &out->role)) {
    *error = std::string("unknown role: ") + node.member_str("role");
    return false;
  }
  out->content = node.member_str("content");
  out->thought = node.member_str("reasoning_content");
  const json::Value* calls = node.find("tool_calls");
  if (calls != nullptr && calls->is_array()) {
    for (const json::Value& entry : calls->items()) {
      ChatMessage::ToolCall call;
      call.id = entry.member_str("id");
      const json::Value* fn = entry.find("function");
      const json::Value& func = (fn != nullptr) ? *fn : entry;
      call.name = func.member_str("name");
      const json::Value* args = func.find("arguments");
      if (args != nullptr && args->is_object()) {
        for (const auto& [key, value] : args->members()) {
          call.arguments.push_back(MakeArgument(key, value));
        }
      }
      out->tool_calls.push_back(std::move(call));
    }
  }
  return true;
}

ChatTool BuildTool(const json::Value& node) {
  ChatTool tool;
  tool.definition_json = node.dump();
  const json::Value* fn = node.find("function");
  if (fn != nullptr) {
    tool.name = fn->member_str("name");
    tool.description = fn->member_str("description");
    const json::Value* params = fn->find("parameters");
    if (params != nullptr) {
      tool.parameters_json = params->dump();
    }
  }
  return tool;
}

void PrintDiffContext(const std::string& expected, const std::string& actual,
                      std::size_t offset) {
  const std::size_t first = offset >= 60 ? offset - 60 : 0;
  std::cout << "  first diff at byte " << offset << " (expected "
            << expected.size() << " bytes, actual " << actual.size()
            << " bytes)\n";
  std::cout << "  expected[" << first << "..]: " << expected.substr(first, 120)
            << "\n";
  std::cout << "  actual  [" << first << "..]: " << actual.substr(first, 120)
            << "\n";
}

}  // namespace

int main() {
#ifndef FLINT_RENDER
  std::cout << "FAIL flint_chat_test: FLINT_RENDER is unset\n";
  return 1;
#else
  json::Value cases;
  try {
    cases = json::parse(ReadFile(FLINT_RENDER));
  } catch (const std::exception& e) {
    std::cout << "FAIL flint_chat_test: cannot parse render.json: " << e.what()
              << "\n";
    return 1;
  }
  if (!cases.is_array() || cases.items().size() != 6) {
    std::cout << "FAIL flint_chat_test: expected 6 cases\n";
    return 1;
  }

  int failures = 0;
  int count = 0;
  for (const json::Value& entry : cases.items()) {
    ++count;
    const json::Value* messages_node = entry.find("messages");
    if (messages_node == nullptr || !messages_node->is_array()) {
      std::cout << "FAIL case " << count << ": no messages array\n";
      ++failures;
      continue;
    }
    std::vector<ChatMessage> messages;
    std::string error;
    bool bad = false;
    for (const json::Value& node : messages_node->items()) {
      ChatMessage message;
      if (!BuildMessage(node, &message, &error)) {
        std::cout << "FAIL case " << count << ": " << error << "\n";
        bad = true;
        break;
      }
      messages.push_back(std::move(message));
    }
    if (bad) {
      ++failures;
      continue;
    }
    std::vector<ChatTool> tools;
    const json::Value* tools_node = entry.find("tools");
    if (tools_node != nullptr && tools_node->is_array()) {
      for (const json::Value& node : tools_node->items()) {
        tools.push_back(BuildTool(node));
      }
    }
    ChatTemplateOptions options;
    options.add_generation_prompt = true;
    const auto rendered = superroc::tokenization::FlintRender(
        std::span<const ChatMessage>(messages.data(), messages.size()),
        std::span<const ChatTool>(tools.data(), tools.size()), options, &error);
    if (!rendered.has_value()) {
      std::cout << "FAIL case " << count << ": render failed: " << error
                << "\n";
      ++failures;
      continue;
    }
    const std::string expected = entry.member_str("text");
    if (*rendered != expected) {
      std::size_t offset = 0;
      while (offset < rendered->size() && offset < expected.size() &&
             (*rendered)[offset] == expected[offset]) {
        ++offset;
      }
      std::cout << "FAIL case " << count << ": render differs\n";
      PrintDiffContext(expected, *rendered, offset);
      ++failures;
      continue;
    }
    std::cout << "PASS case " << count << ": " << rendered->size()
              << " bytes match\n";
  }

  if (failures != 0) {
    std::cout << "FAIL flint_chat_test: " << failures << " of " << count
              << " cases differ\n";
    return 1;
  }
  std::cout << "PASS flint_chat_test: all " << count << " cases match\n";
  return 0;
#endif
}
