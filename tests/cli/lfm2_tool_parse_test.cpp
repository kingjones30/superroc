// LFM2: unit test for ParseLfm2PythonicCalls (plain main, PASS/FAIL lines).
#include "src/cli/serve/lfm2_tool_parse.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
  if (condition) {
    std::cout << "PASS " << message << "\n";
  } else {
    std::cout << "FAIL " << message << "\n";
    ++failures;
  }
}

using superroc::server::ParseLfm2PythonicCalls;

void TestSingleStringArg() {
  const auto calls = ParseLfm2PythonicCalls("[get_weather(city='Paris')]");
  Check(calls.has_value(), "single call parses");
  if (!calls.has_value())
    return;
  Check(calls->size() == 1, "single call count == 1");
  Check((*calls)[0].name == "get_weather", "single call name");
  Check((*calls)[0].arguments.size() == 1, "single call argc == 1");
  if ((*calls)[0].arguments.size() == 1) {
    Check((*calls)[0].arguments[0].name == "city", "single call arg name");
    Check((*calls)[0].arguments[0].value == "Paris", "single call arg value");
    Check((*calls)[0].arguments[0].is_string, "single call arg is_string");
  }
}

void TestScalars() {
  const auto calls = ParseLfm2PythonicCalls(
      "[f(a=1, b=2.5, c=True, d=None, e='it\\'s\\nok')]");
  Check(calls.has_value(), "scalars parse");
  if (!calls.has_value())
    return;
  Check(calls->size() == 1, "scalars count == 1");
  const auto& args = (*calls)[0].arguments;
  Check(args.size() == 5, "scalars argc == 5");
  if (args.size() != 5)
    return;
  Check(args[0].value == "1" && !args[0].is_string, "int 1");
  Check(args[1].value == "2.5" && !args[1].is_string, "float 2.5");
  Check(args[2].value == "true" && !args[2].is_string, "True -> true");
  Check(args[3].value == "null" && !args[3].is_string, "None -> null");
  Check(args[4].value == std::string("it's\nok") && args[4].is_string,
        "escaped string");
}

void TestNested() {
  const auto calls = ParseLfm2PythonicCalls("[f(x={'k': [1, 'two', False]})]");
  Check(calls.has_value(), "nested parses");
  if (!calls.has_value())
    return;
  Check((*calls)[0].arguments.size() == 1, "nested argc == 1");
  if ((*calls)[0].arguments.empty())
    return;
  Check((*calls)[0].arguments[0].value == "{\"k\": [1, \"two\", false]}",
        "nested JSON text");
  Check(!(*calls)[0].arguments[0].is_string, "nested is_string false");
}

void TestTwoCalls() {
  const auto calls = ParseLfm2PythonicCalls("[f(), g(y=\"dq\")]");
  Check(calls.has_value(), "two calls parse");
  if (!calls.has_value())
    return;
  Check(calls->size() == 2, "two calls count == 2");
  if (calls->size() != 2)
    return;
  Check((*calls)[0].name == "f" && (*calls)[0].arguments.empty(),
        "first call empty args");
  Check((*calls)[1].name == "g", "second call name");
  Check((*calls)[1].arguments.size() == 1 &&
            (*calls)[1].arguments[0].value == "dq" &&
            (*calls)[1].arguments[0].is_string,
        "second call double-quoted arg");
}

void TestSpaces() {
  const auto calls = ParseLfm2PythonicCalls("[ f(a='x') ]");
  Check(calls.has_value(), "spaces parse");
  if (!calls.has_value())
    return;
  Check(calls->size() == 1 && (*calls)[0].arguments.size() == 1 &&
            (*calls)[0].arguments[0].value == "x",
        "spaces call content");
}

void TestJsonLiterals() {
  // LFM2 (Super ROC): JSON spellings and escapes the tool grammar admits.
  const auto calls = ParseLfm2PythonicCalls(
      "[f(a=true, b=false, c=null, d={\"k\": [true, null]}, e='x\\/y\\b')]");
  Check(calls.has_value(), "json literals parse");
  if (!calls.has_value())
    return;
  const auto& args = (*calls)[0].arguments;
  Check(args.size() == 5, "json literals argc == 5");
  if (args.size() != 5)
    return;
  Check(args[0].value == "true" && !args[0].is_string, "true -> true");
  Check(args[1].value == "false" && !args[1].is_string, "false -> false");
  Check(args[2].value == "null" && !args[2].is_string, "null -> null");
  Check(args[3].value == "{\"k\": [true, null]}", "nested json literals");
  Check(args[4].value == std::string("x/y\b") && args[4].is_string,
        "escapes / and b");
  Check(!ParseLfm2PythonicCalls("[f(a=nullx)]").has_value(),
        "error: null needs a word boundary");
}

void TestErrors() {
  Check(!ParseLfm2PythonicCalls("[f(a='x')").has_value(),
        "error: missing bracket");
  Check(!ParseLfm2PythonicCalls("[f(a=)]").has_value(), "error: empty value");
  Check(!ParseLfm2PythonicCalls("[f(a='x' b=1)]").has_value(),
        "error: missing comma");
  Check(!ParseLfm2PythonicCalls("[1bad()]").has_value(), "error: bad name");
  Check(!ParseLfm2PythonicCalls("").has_value(), "error: empty string");
}

// HTC (RVW C3): deep nesting fails closed instead of exhausting the stack;
// nesting within the cap still parses.
void TestNestingCap() {
  const std::string deep = "[f(a=" + std::string(100000, '[') + "1" +
                           std::string(100000, ']') + ")]";
  Check(!ParseLfm2PythonicCalls(deep).has_value(), "nesting 100000 refused");
  const std::string ok =
      "[f(a=" + std::string(60, '[') + "1" + std::string(60, ']') + ")]";
  const auto calls = ParseLfm2PythonicCalls(ok);
  Check(calls.has_value() && calls->size() == 1, "nesting 60 parses");
  const std::string over =
      "[f(a=" + std::string(70, '[') + "1" + std::string(70, ']') + ")]";
  Check(!ParseLfm2PythonicCalls(over).has_value(), "nesting 70 refused");
}

}  // namespace

int main() {
  TestSingleStringArg();
  TestScalars();
  TestNested();
  TestTwoCalls();
  TestSpaces();
  TestJsonLiterals();
  TestErrors();
  TestNestingCap();
  if (failures == 0) {
    std::cout << "lfm2_tool_parse_test: passed\n";
    return 0;
  }
  std::cout << "lfm2_tool_parse_test: " << failures << " failures\n";
  return 1;
}
