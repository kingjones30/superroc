// LFM2: chat render equality against ~/ls/lfm/ref_render/<name>.txt.
// Builds ChatMessage/ChatTool values from LFM2_CASES JSON, calls Lfm2Render
// and compares byte for byte. Case c09 additionally checks tokenization keeps
// spelled specials as text. Exits 77 (SKIP) when LFM2_GGUF is unset.

#include "src/models/lfm2/lfm2_chat.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"

namespace {

namespace fs = std::filesystem;
namespace json = superroc::json;
using superroc::tokenization::ChatMessage;
using superroc::tokenization::ChatRole;
using superroc::tokenization::ChatTemplateOptions;
using superroc::tokenization::ChatTool;

std::string ReadFile(const fs::path& path) {
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
  const char* gguf = std::getenv("LFM2_GGUF");
  if (gguf == nullptr || gguf[0] == '\0') {
    std::cout << "SKIP lfm2_chat_test: LFM2_GGUF is unset\n";
    return 77;
  }
  const char* cases_path = std::getenv("LFM2_CASES");
  if (cases_path == nullptr || cases_path[0] == '\0') {
    std::cout << "SKIP lfm2_chat_test: LFM2_CASES is unset\n";
    return 77;
  }
  fs::path ref_dir;
  if (const char* env = std::getenv("LFM2_REF_DIR");
      env != nullptr && env[0] != '\0') {
    ref_dir = fs::path(env);
  } else if (const char* home = std::getenv("HOME");
             home != nullptr && home[0] != '\0') {
    ref_dir = fs::path(home) / "ls" / "lfm" / "ref_render";
  } else {
    std::cout << "FAIL lfm2_chat_test: no ref dir (set LFM2_REF_DIR)\n";
    return 1;
  }

  std::string error;
  const auto reader = superroc::core::GgufReader::OpenFile(fs::path(gguf), &error);
  if (reader == nullptr) {
    std::cout << "FAIL lfm2_chat_test: cannot open GGUF: " << error << "\n";
    return 1;
  }
  auto tokenizer =
      superroc::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (tokenizer == nullptr) {
    std::cout << "FAIL lfm2_chat_test: cannot load tokenizer: " << error
              << "\n";
    return 1;
  }
  const auto* variant = superroc::tokenization::Lfm2TemplateVariantForGguf(*reader);
  if (variant == nullptr) {
    std::cout << "FAIL lfm2_chat_test: unsupported template\n";
    return 1;
  }
  std::cout << "variant " << variant->name << "\n";

  json::Value cases;
  try {
    cases = json::parse(ReadFile(fs::path(cases_path)));
  } catch (const std::exception& e) {
    std::cout << "FAIL lfm2_chat_test: cannot parse cases: " << e.what()
              << "\n";
    return 1;
  }
  if (!cases.is_array() || cases.empty()) {
    std::cout << "FAIL lfm2_chat_test: cases file holds no cases\n";
    return 1;
  }

  int failures = 0;
  int count = 0;
  for (const json::Value& entry : cases.items()) {
    const std::string name = entry.member_str("name");
    ++count;
    const json::Value* messages_node = entry.find("messages");
    if (name.empty() || messages_node == nullptr ||
        !messages_node->is_array()) {
      std::cout << "FAIL " << name << ": case has no messages array\n";
      ++failures;
      continue;
    }
    std::vector<ChatMessage> messages;
    bool bad = false;
    for (const json::Value& node : messages_node->items()) {
      ChatMessage message;
      if (!BuildMessage(node, &message, &error)) {
        std::cout << "FAIL " << name << ": " << error << "\n";
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
    if (const json::Value* gen = entry.find("add_generation_prompt");
        gen != nullptr && gen->is_bool()) {
      options.add_generation_prompt = gen->as_bool(true);
    }
    const auto rendered = superroc::tokenization::Lfm2Render(
        std::span<const ChatMessage>(messages.data(), messages.size()),
        std::span<const ChatTool>(tools.data(), tools.size()), options, &error,
        nullptr, *variant);
    if (!rendered.has_value()) {
      std::cout << "FAIL " << name << ": render failed: " << error << "\n";
      ++failures;
      continue;
    }
    const std::string expected = ReadFile(ref_dir / (name + ".txt"));
    if (*rendered != expected) {
      std::size_t offset = 0;
      while (offset < rendered->size() && offset < expected.size() &&
             (*rendered)[offset] == expected[offset]) {
        ++offset;
      }
      std::cout << "FAIL " << name << ": render differs\n";
      PrintDiffContext(expected, *rendered, offset);
      ++failures;
      continue;
    }
    std::cout << "PASS " << name << ": " << rendered->size()
              << " bytes match\n";
    if (name == "c05_tools") {
      // LFM2: tools + require_tool_call renders the c05 reference with
      // the required sentence inserted after the tool list's closing ']'.
      ChatTemplateOptions required_options = options;
      required_options.require_tool_call = true;
      const auto required_rendered = superroc::tokenization::Lfm2Render(
          std::span<const ChatMessage>(messages.data(), messages.size()),
          std::span<const ChatTool>(tools.data(), tools.size()),
          required_options, &error, nullptr, *variant);
      if (!required_rendered.has_value()) {
        std::cout << "FAIL c05_tools+required: render failed: " << error
                  << "\n";
        ++failures;
      } else {
        std::string required_expected = expected;
        const std::string close = "}]<|im_end|>";
        const std::size_t at = required_expected.find(close);
        if (at == std::string::npos) {
          std::cout << "FAIL c05_tools+required: c05 ref has no tool list"
                       " close\n";
          ++failures;
        } else {
          required_expected.insert(
              at + 2,
              "\nYou must call at least one available function. Do not "
              "answer the user directly.");
          // Step 3 prefill: require_tool_call ends the generation prompt
          // with `<think></think><|tool_call_start|>[` instead of `<think>`.
          const std::string gen_prompt = variant->generation_opens_think
                                             ? "<|im_start|>assistant\n<think>"
                                             : "<|im_start|>assistant\n";
          const bool has_gen_prompt =
              required_expected.size() >= gen_prompt.size() &&
              required_expected.compare(
                  required_expected.size() - gen_prompt.size(),
                  gen_prompt.size(), gen_prompt) == 0;
          if (!has_gen_prompt) {
            std::cout << "FAIL c05_tools+required: c05 ref has no generation "
                         "prompt\n";
            ++failures;
          } else {
            required_expected.replace(
                required_expected.size() - gen_prompt.size(), gen_prompt.size(),
                std::string("<|im_start|>assistant\n<think></"
                            "think><|tool_call_start|>[") +
                    (tools.size() == 1 ? tools[0].name + "(" : std::string()));
          }
          if (has_gen_prompt && *required_rendered != required_expected) {
            std::size_t offset = 0;
            while (offset < required_rendered->size() &&
                   offset < required_expected.size() &&
                   (*required_rendered)[offset] == required_expected[offset]) {
              ++offset;
            }
            std::cout << "FAIL c05_tools+required: render differs\n";
            PrintDiffContext(required_expected, *required_rendered, offset);
            ++failures;
          } else {
            std::cout << "PASS c05_tools+required: "
                      << required_rendered->size() << " bytes match\n";
          }
        }
      }
    }

    if (name == "c09_special_in_content") {
      if (variant->name != "lfm2.5-1.2b-instruct") {
        const auto tokens = superroc::tokenization::Lfm2RenderAndTokenize(
            *tokenizer,
            std::span<const ChatMessage>(messages.data(), messages.size()),
            std::span<const ChatTool>(tools.data(), tools.size()), options,
            &error);
        if (!tokens.has_value()) {
          std::cout << "FAIL c09: tokenize failed: " << error << "\n";
          ++failures;
          continue;
        }
        const auto im_start = tokenizer->FindSpecialToken("<|im_start|>");
        if (!im_start.has_value()) {
          std::cout << "FAIL c09: no <|im_start|> token\n";
          ++failures;
          continue;
        }
        const std::size_t bos =
            std::count(tokens->begin(), tokens->end(), 124894);
        const std::size_t im_end =
            std::count(tokens->begin(), tokens->end(), 124900);
        const std::size_t im_start_count =
            std::count(tokens->begin(), tokens->end(), *im_start);
        if (bos != 1 || im_end != 1 || im_start_count != 2) {
          std::cout << "FAIL c09: BOS x" << bos << " <|im_end|> x" << im_end
                    << " <|im_start|> x" << im_start_count
                    << " (want 1/1/2) in " << tokens->size() << " tokens\n";
          ++failures;
          continue;
        }
        std::cout << "PASS c09: BOS x1 <|im_end|> x1 <|im_start|> x2 in "
                  << tokens->size() << " tokens\n";
      } else {
        std::cout << "SKIP c09 token ids: 1.2B vocabulary\n";
      }
    }
  }

  if (failures != 0) {
    std::cout << "FAIL lfm2_chat_test: " << failures << " of " << count
              << " cases differ\n";
    return 1;
  }
  std::cout << "PASS lfm2_chat_test: all " << count << " cases match\n";
  return 0;
}
