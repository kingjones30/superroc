// mtp_stub.cpp: HIP-off build stub for the Ling-3 MTP draft head.
// HIP-off build: the MTP draft head lives in mtp_gpu.cpp (HIP only);
// these definitions keep engine.cpp linking and make the CPU engine report
// no speculative support. Every query reports unavailable; every state
// pointer is null; every mutating call is a no-op. Do NOT extend: HIP builds
// compile mtp_gpu.cpp instead and must never list this file.

#include "src/models/ling3/mtp_gpu.hpp"

namespace superroc::models::ling3::mtp {

bool SpeculativeEnabled() noexcept {
  return {};
}

State* SpecHead([[maybe_unused]] SpecState* spec) noexcept {
  return {};
}

int SpecDraftLength([[maybe_unused]] const SpecState* spec) noexcept {
  return {};
}

bool Ready() noexcept {
  return {};
}

SpecState* NewSpecState([[maybe_unused]] std::uint32_t max_ctx,
                        [[maybe_unused]] const SpecPolicy& policy,
                        std::string* error_msg) {
  if (error_msg)
    *error_msg = "ling3 MTP draft head needs the HIP build";
  return {};
}

bool SpecStep([[maybe_unused]] void* trunk, [[maybe_unused]] SpecState* spec,
              [[maybe_unused]] std::int32_t base,
              [[maybe_unused]] const std::vector<float>& h_prev,
              [[maybe_unused]] bool have_h_prev,
              [[maybe_unused]] std::vector<std::int32_t>* out,
              [[maybe_unused]] std::vector<float>* next_logits,
              [[maybe_unused]] std::vector<float>* h_prev_out,
              [[maybe_unused]] SpecStats* stats,
              [[maybe_unused]] const std::function<bool(std::int32_t)>& is_stop,
              std::string* error_msg) {
  if (error_msg)
    *error_msg = "ling3 MTP draft head needs the HIP build";
  return {};
}

void ResetSpecState([[maybe_unused]] SpecState* spec) noexcept {}

void ReportSpecSession([[maybe_unused]] const SpecStats& s,
                       [[maybe_unused]] int final_draft_len,
                       [[maybe_unused]] int req,
                       [[maybe_unused]] int prompt_tokens) {}

void ProbeStep([[maybe_unused]] void* backend_state,
               [[maybe_unused]] std::int32_t x_next,
               [[maybe_unused]] std::span<const float> h_prev,
               [[maybe_unused]] std::uint32_t pos,
               [[maybe_unused]] bool have_hidden,
               [[maybe_unused]] bool scored) {}

void ProbeInit([[maybe_unused]] const ModelWeights& weights,
               [[maybe_unused]] std::uint32_t max_context,
               std::string* error_msg) {
  if (error_msg)
    *error_msg = "ling3 MTP draft head needs the HIP build";
}

bool ProbeEnabled() noexcept {
  return {};
}

bool PrimeNext([[maybe_unused]] State* state,
               [[maybe_unused]] std::int32_t x_next,
               [[maybe_unused]] std::span<const float> h_prev,
               [[maybe_unused]] std::uint32_t pos,
               [[maybe_unused]] std::span<float> h_out,
               std::string* error_msg) {
  if (error_msg)
    *error_msg = "ling3 MTP draft head needs the HIP build";
  return {};
}

bool PrimeBatchEnabled() {
  return {};
}

bool PrimeBatch([[maybe_unused]] State* state,
                [[maybe_unused]] const std::int32_t* tokens,
                [[maybe_unused]] const float* h_prev, [[maybe_unused]] int n,
                [[maybe_unused]] std::uint32_t pos0, std::string* error_msg) {
  if (error_msg)
    *error_msg = "ling3 MTP draft head needs the HIP build";
  return {};
}

void InstallSpecReporter() {}

bool Init([[maybe_unused]] const ModelWeights& weights,
          [[maybe_unused]] HeadNorm head_norm,
          [[maybe_unused]] std::size_t* resident_bytes,
          std::string* error_msg) {
  if (error_msg)
    *error_msg = "ling3 MTP draft head needs the HIP build";
  return {};
}

void DestroySpecState([[maybe_unused]] SpecState* spec) noexcept {}

bool DepthDraftOk([[maybe_unused]] int pos,
                  [[maybe_unused]] int remaining_tokens) noexcept {
  return {};
}

}  // namespace superroc::models::ling3::mtp
