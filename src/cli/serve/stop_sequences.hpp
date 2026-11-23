#ifndef SUPERROC_SERVER_STOP_SEQUENCES_HPP_
#define SUPERROC_SERVER_STOP_SEQUENCES_HPP_

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/json.hpp"

namespace superroc::server {

inline constexpr std::size_t kMaxStopSequences = 64;
/// OpenAI documents up to 4 `stop` strings; up to 16 are accepted so harness
/// requests meant for native-window servers also run (lemonade/lm-eval GSM8K).
inline constexpr std::size_t kMaxOpenAiStopSequences = 16;
inline constexpr std::size_t kMaxStopSequenceBytes = 4096;
inline constexpr std::size_t kMaxStopTotalBytes = 16384;

inline void ValidateStopSequences(const std::vector<std::string>& sequences,
                                  std::size_t max_count = kMaxStopSequences) {
  if (sequences.size() > max_count)
    throw std::invalid_argument("too many stop sequences (maximum " +
                                std::to_string(max_count) + ")");
  std::size_t bytes = 0;
  for (const auto& sequence : sequences) {
    if (sequence.empty() || sequence.size() > kMaxStopSequenceBytes)
      throw std::invalid_argument("stop sequences must contain 1 to " +
                                  std::to_string(kMaxStopSequenceBytes) +
                                  " UTF-8 bytes");
    bytes += sequence.size();
    if (bytes > kMaxStopTotalBytes)
      throw std::invalid_argument("stop sequences exceed the total byte limit");
  }
}

enum class StopSequenceFormat { kOpenAi, kAnthropic };

inline std::optional<std::string> ParseStopSequences(
    const json::Value* value, StopSequenceFormat format,
    std::vector<std::string>* output) {
  output->clear();
  if (value == nullptr ||
      (format == StopSequenceFormat::kOpenAi && value->is_null()))
    return std::nullopt;
  if (format == StopSequenceFormat::kOpenAi && value->is_string()) {
    output->push_back(value->str());
  } else if (value->is_array()) {
    for (const auto& item : value->items()) {
      if (!item.is_string())
        return "stop sequences must be strings";
      output->push_back(item.str());
    }
  } else {
    return format == StopSequenceFormat::kOpenAi
               ? "'stop' must be null, a string, or an array of strings"
               : "'stop_sequences' must be an array of strings";
  }
  try {
    ValidateStopSequences(*output, format == StopSequenceFormat::kOpenAi
                                       ? kMaxOpenAiStopSequences
                                       : kMaxStopSequences);
  } catch (const std::invalid_argument& error) {
    return error.what();
  }
  return std::nullopt;
}

/// Matches raw accepted output, retaining only a possible stop prefix.
/// The first completed match wins; ties at the same end prefer the longer
/// sequence. This is independent of how token pieces are split into chunks.
///
/// With `reasoning_ends`, the output starts inside a reasoning span that ends
/// at the first of those markers. A stop sequence is then matched in the text
/// after the marker only: the client wrote it for the answer it reads, and a
/// match inside the reasoning would end the reply before any answer exists.
/// A sequence that contains a marker names the boundary itself and is matched
/// everywhere.
class StopSequenceFilter {
public:
  explicit StopSequenceFilter(std::vector<std::string> sequences = {},
                              std::vector<std::string> reasoning_ends = {})
      : sequences_(std::move(sequences)),
        reasoning_ends_(std::move(reasoning_ends)) {
    ValidateStopSequences(sequences_);
    std::erase_if(reasoning_ends_,
                  [](const std::string& marker) { return marker.empty(); });
    if (sequences_.empty())
      reasoning_ends_.clear();
    everywhere_.reserve(sequences_.size());
    for (const auto& sequence : sequences_) {
      everywhere_.push_back(
          std::ranges::any_of(reasoning_ends_, [&](const std::string& marker) {
            return sequence.find(marker) != std::string::npos;
          }));
    }
    if (!reasoning_ends_.empty())
      answer_begin_ = std::string::npos;
  }

  [[nodiscard]] bool enabled() const noexcept { return !sequences_.empty(); }
  [[nodiscard]] bool stopped() const noexcept { return match_.has_value(); }
  [[nodiscard]] const std::string& matched_sequence() const {
    return sequences_.at(match_.value());
  }

  std::string Push(std::string_view piece) {
    if (stopped())
      return {};
    pending_.append(piece);
    if (answer_begin_ == std::string::npos) {
      // The reasoning ends at the marker that starts first.
      std::size_t first = std::string::npos;
      for (const auto& marker : reasoning_ends_) {
        const auto position = pending_.find(marker);
        if (position < first) {
          first = position;
          answer_begin_ = position + marker.size();
        }
      }
    }
    std::size_t end = std::string::npos;
    std::size_t start = std::string::npos;
    for (std::size_t i = 0; i < sequences_.size(); ++i) {
      const std::size_t from = everywhere_[i] ? 0 : answer_begin_;
      if (from == std::string::npos)
        continue;
      const auto position = pending_.find(sequences_[i], from);
      if (position == std::string::npos)
        continue;
      const auto candidate_end = position + sequences_[i].size();
      if (candidate_end < end || (candidate_end == end && position < start)) {
        end = candidate_end;
        start = position;
        match_ = i;
      }
    }
    if (stopped()) {
      auto safe = pending_.substr(0, start);
      pending_.clear();
      return safe;
    }
    // Hold back the longest tail that may still grow into a match: a prefix
    // of a sequence (within the text it is matched in) or, during reasoning,
    // of an end marker.
    std::size_t keep = 0;
    const auto hold = [&](std::string_view pattern, std::size_t reach) {
      for (auto length = std::min(reach, pattern.size() - 1); length > keep;
           --length) {
        if (std::string_view(pending_).ends_with(pattern.substr(0, length))) {
          keep = length;
          break;
        }
      }
    };
    for (std::size_t i = 0; i < sequences_.size(); ++i) {
      if (everywhere_[i]) {
        hold(sequences_[i], pending_.size());
      } else if (answer_begin_ != std::string::npos) {
        hold(sequences_[i], pending_.size() - answer_begin_);
      }
    }
    if (answer_begin_ == std::string::npos) {
      for (const auto& marker : reasoning_ends_)
        hold(marker, pending_.size());
    }
    auto safe = pending_.substr(0, pending_.size() - keep);
    pending_.erase(0, safe.size());
    if (answer_begin_ != std::string::npos)
      answer_begin_ -= std::min(answer_begin_, safe.size());
    return safe;
  }

  std::string Finish() { return std::exchange(pending_, {}); }

private:
  std::vector<std::string> sequences_;
  std::vector<std::string> reasoning_ends_;
  // Per sequence: matched in the reasoning as well.
  std::vector<bool> everywhere_;
  std::string pending_;
  // Offset in pending_ where the answer starts; npos while the reasoning has
  // not ended.
  std::size_t answer_begin_{0};
  std::optional<std::size_t> match_;
};

}  // namespace superroc::server

#endif  // SUPERROC_SERVER_STOP_SEQUENCES_HPP_
