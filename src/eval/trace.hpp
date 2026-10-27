#ifndef SUPERROC_EVAL_TRACE_HPP_
#define SUPERROC_EVAL_TRACE_HPP_

#include <cstdint>
#include <string>
#include <vector>

namespace superroc::eval {

struct EvalCaseTrace {
  std::string case_id;
  std::string prompt;
  std::string expected_answer;
  std::string extracted_answer;
  std::vector<std::uint32_t> generated_tokens;
  bool passed{false};
  double elapsed_ms{0.0};
};

struct EvalTraceReport {
  std::string model_id;
  std::string route_identity;
  std::string suite_hash;
  std::vector<EvalCaseTrace> cases;
};

}  // namespace superroc::eval

#endif  // SUPERROC_EVAL_TRACE_HPP_
