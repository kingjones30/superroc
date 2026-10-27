#ifndef SUPERROC_MODELS_LING3_VISION_PROMPT_HPP_
#define SUPERROC_MODELS_LING3_VISION_PROMPT_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/ling3/tokenizer.hpp"

namespace superroc::models::ling3::vision {

/// One prompt after the Bailing V3 image placeholder has been expanded.
///
/// `tokens[i]` is the physical sequence the language model walks. An image
/// placeholder becomes `nx * ny` slots (the `<|image_pad|>` id, overwritten
/// with projector rows). `rope` is three ints per physical token, (t, h, w),
/// matching llama.cpp mtmd M-RoPE for sections [8, 12, 12, 0]:
///   text token: t = h = w = the running logical position
///   image token k (row-major): t = pos0, h = pos0 + row, w = pos0 + col
///   pos0 is the logical position of the first image token
///   the logical counter then advances by max(nx, ny), not by nx*ny
/// `next_logical` is the logical position of the first token after this prompt.
struct PreparedPrompt {
  std::vector<std::int32_t> tokens;
  std::vector<std::int32_t> rope;
  std::vector<float> image_emb;
  std::vector<std::int32_t> image_phys;
  std::int32_t next_logical{0};
  int hidden{2560};
  bool bind_rope{false};
};

/// Host-only check of the position rule. Returns false and fills `error`.
[[nodiscard]] bool RopeSelfCheck(std::string* error);

/// `embeddings` is row-major [nx*ny, hidden] from the vision tower. Empty when
/// `with_image` is false. The rendered string is the VL jinja template with
/// thinking on (its default): the user text is preceded by
/// `<|vision_start|><|image_pad|><|vision_end|>` when an image is present,
/// which is what mtmd produces after it replaces `<__media__>`.
/// `grids[i]` is (nx, ny) merged tokens for the i-th `<|image_pad|>` run.
/// `tokens` must already be expanded. An empty grid list is a text prompt.
[[nodiscard]] bool FillImageRope(const std::vector<std::int32_t>& tokens,
                                 std::int32_t pad,
                                 const std::vector<std::pair<int, int>>& grids,
                                 PreparedPrompt* out, std::string* error);

[[nodiscard]] bool Prepare(const LingTokenizer& tokenizer,
                           std::string_view user_text, bool with_image,
                           int image_w, int image_h,
                           std::span<const float> embeddings,
                           PreparedPrompt* out, std::string* error);

/// Same tokens, but a rope table where every section equals the physical
/// index. Binding it must not change a text-only forward.
[[nodiscard]] PreparedPrompt IdentityOf(std::span<const std::int32_t> tokens);

void Arm(const PreparedPrompt& prompt);
void Clear();
[[nodiscard]] bool Armed();

/// Called from the ROCm backend after the token embedding lookup, before the
/// first layer. No-op unless Arm() is set. Overwrites image rows and publishes
/// the M-RoPE table the MLA rope kernels read.
void OnHidden(float* hidden, int pos0, int n, int width, void* stream);

}  // namespace superroc::models::ling3::vision

#endif
