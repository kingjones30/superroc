#ifndef GUFO_MODELS_LING3_VISION_TOWER_HPP_
#define GUFO_MODELS_LING3_VISION_TOWER_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gufo::models::ling3::vision {

/// Qwen3-VL mmproj (BF16 clip) kept resident for `gufo serve`.
/// Encode takes HWC RGB8 whose edges are multiples of 32 and at most 1024.
class VisionTower {
public:
  static std::shared_ptr<VisionTower> Open(const std::string& path,
                                           std::string* error);
  [[nodiscard]] std::vector<float> Encode(const std::uint8_t* rgb, int height,
                                          int width, std::string* error);
  [[nodiscard]] std::size_t HostBytes() const noexcept;
  /// Activation scratch for the largest image this tower accepts.
  [[nodiscard]] std::size_t ScratchBytes() const noexcept;
  [[nodiscard]] int OutputWidth() const noexcept { return 2560; }
  ~VisionTower();

  VisionTower(const VisionTower&) = delete;
  VisionTower& operator=(const VisionTower&) = delete;

private:
  struct Impl;
  VisionTower();
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::ling3::vision

#endif
