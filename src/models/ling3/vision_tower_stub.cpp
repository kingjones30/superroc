#include "src/models/ling3/vision_tower.hpp"

namespace superroc::models::ling3::vision {

struct VisionTower::Impl {};

VisionTower::VisionTower() = default;
VisionTower::~VisionTower() = default;

std::shared_ptr<VisionTower> VisionTower::Open(const std::string&,
                                               std::string* error) {
  if (error != nullptr) {
    *error = "Ling vision tower requires the HIP backend";
  }
  return {};
}

std::vector<float> VisionTower::Encode(const std::uint8_t*, int, int,
                                       std::string* error) {
  if (error != nullptr) {
    *error = "Ling vision tower requires the HIP backend";
  }
  return {};
}

std::size_t VisionTower::HostBytes() const noexcept {
  return 0;
}
std::size_t VisionTower::ScratchBytes() const noexcept {
  return 0;
}

}  // namespace superroc::models::ling3::vision
