#include "src/models/ling3/vision_prompt.hpp"

#include <algorithm>
#include <string>
#include <utility>

namespace gufo::models::ling3::vision {
namespace {

constexpr int kHidden = 2560;
constexpr int kPatch = 16;
constexpr int kMerge = 2;
constexpr int kAlign = kPatch * kMerge;  // 32

bool LookupOne(const LingTokenizer& tokenizer, std::string_view text,
               std::int32_t* id, std::string* error) {
  const std::vector<std::int32_t> ids = tokenizer.Encode(text, false, true);
  if (ids.size() != 1) {
    if (error != nullptr) {
      *error = "expected one special token for " + std::string(text);
    }
    return false;
  }
  *id = ids[0];
  return true;
}

// One contiguous pad run per image, in order. Text tokens take one logical
// position; an image of nx by ny advances the logical counter by max(nx, ny).
bool AssignRopes(const std::vector<std::int32_t>& tokens, std::int32_t pad,
                 const std::vector<std::pair<int, int>>& grids,
                 PreparedPrompt* out, std::string* error) {
  if (out == nullptr)
    return false;
  struct Run {
    int start;
    int len;
    int nx;
    int ny;
  };
  std::vector<Run> runs;
  const int n = static_cast<int>(tokens.size());
  for (int i = 0; i < n;) {
    if (tokens[static_cast<std::size_t>(i)] != pad) {
      ++i;
      continue;
    }
    const int start = i;
    while (i < n && tokens[static_cast<std::size_t>(i)] == pad)
      ++i;
    if (runs.size() >= grids.size()) {
      if (error)
        *error = "prompt has more image placeholders than images";
      return false;
    }
    const auto [nx, ny] = grids[runs.size()];
    const int need = nx * ny;
    if (nx <= 0 || ny <= 0 || i - start != need) {
      if (error)
        *error = "image pad run length does not match the grid";
      return false;
    }
    runs.push_back({start, need, nx, ny});
  }
  if (runs.size() != grids.size()) {
    if (error)
      *error = grids.empty() ? "unexpected image pad in a text prompt"
                             : "prompt is missing an image placeholder";
    return false;
  }

  out->tokens = tokens;
  out->rope.assign(static_cast<std::size_t>(n) * 3, 0);
  out->image_phys.clear();
  int logical = 0;
  std::size_t run_i = 0;
  for (int i = 0; i < n;) {
    if (run_i < runs.size() && i == runs[run_i].start) {
      const Run& run = runs[run_i];
      const int pos0 = logical;
      for (int k = 0; k < run.len; ++k) {
        const int row = k / run.nx;
        const int col = k % run.nx;
        const std::size_t base = static_cast<std::size_t>(i + k) * 3;
        out->rope[base] = pos0;
        out->rope[base + 1] = pos0 + row;
        out->rope[base + 2] = pos0 + col;
        out->image_phys.push_back(i + k);
      }
      logical += std::max(run.nx, run.ny);
      i += run.len;
      ++run_i;
    } else {
      const std::size_t base = static_cast<std::size_t>(i) * 3;
      out->rope[base] = out->rope[base + 1] = out->rope[base + 2] = logical;
      ++logical;
      ++i;
    }
  }
  out->next_logical = logical;
  out->hidden = kHidden;
  out->bind_rope = !grids.empty();
  if (error)
    error->clear();
  return true;
}

bool AssignRope(const std::vector<std::int32_t>& tokens, std::int32_t pad,
                int nx, int ny, bool expect_image, PreparedPrompt* out,
                std::string* error) {
  std::vector<std::pair<int, int>> grids;
  if (expect_image)
    grids.emplace_back(nx, ny);
  return AssignRopes(tokens, pad, grids, out, error);
}

}  // namespace

bool FillImageRope(const std::vector<std::int32_t>& tokens, std::int32_t pad,
                   const std::vector<std::pair<int, int>>& grids,
                   PreparedPrompt* out, std::string* error) {
  return AssignRopes(tokens, pad, grids, out, error);
}

bool RopeSelfCheck(std::string* error) {
  // 2 text tokens, a 2x2 image, one trailing text token. Pad id is 7.
  const std::vector<std::int32_t> tokens = {1, 2, 7, 7, 7, 7, 3};
  PreparedPrompt got;
  if (!AssignRope(tokens, 7, 2, 2, true, &got, error))
    return false;
  const int expect[] = {
      0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 3, 3, 4, 4, 4,
  };
  // Two text positions, the image advances by max(nx, ny)=2, then the
  // trailing text token takes the next logical slot. next_logical is the
  // position after that token.
  if (got.rope.size() != 21 || got.next_logical != 5 ||
      got.image_phys.size() != 4) {
    if (error) {
      *error = "rope self-check shape rope=" + std::to_string(got.rope.size()) +
               " next=" + std::to_string(got.next_logical) +
               " images=" + std::to_string(got.image_phys.size());
    }
    return false;
  }
  for (int i = 0; i < 21; ++i) {
    if (got.rope[static_cast<std::size_t>(i)] != expect[i]) {
      if (error)
        *error = "rope self-check value at " + std::to_string(i);
      return false;
    }
  }
  if (error)
    error->clear();
  return true;
}

bool Prepare(const LingTokenizer& tokenizer, std::string_view user_text,
             bool with_image, int image_w, int image_h,
             std::span<const float> embeddings, PreparedPrompt* out,
             std::string* error) {
  if (out == nullptr)
    return false;
  std::int32_t pad = 0;
  if (!LookupOne(tokenizer, "<|image_pad|>", &pad, error))
    return false;
  std::int32_t vision_start = 0;
  std::int32_t vision_end = 0;
  if (!LookupOne(tokenizer, "<|vision_start|>", &vision_start, error) ||
      !LookupOne(tokenizer, "<|vision_end|>", &vision_end, error)) {
    return false;
  }
  (void)vision_start;
  (void)vision_end;

  int nx = 0;
  int ny = 0;
  if (with_image) {
    if (image_w <= 0 || image_h <= 0 || image_w % kAlign || image_h % kAlign) {
      if (error)
        *error = "image edge must be a positive multiple of 32";
      return false;
    }
    nx = image_w / kAlign;
    ny = image_h / kAlign;
    const std::size_t rows = static_cast<std::size_t>(nx * ny);
    if (embeddings.size() != rows * static_cast<std::size_t>(kHidden)) {
      if (error)
        *error = "embedding file is not nx*ny*2560 floats";
      return false;
    }
  }

  // VL jinja default (enable_thinking true, no system turn): no newline between
  // <role>SYSTEM</role> and the thinking instruction. mtmd then replaces the
  // media marker with vision_start / embeddings / vision_end.
  std::string rendered =
      "<role>SYSTEM</role>detailed thinking on<|role_end|>"
      "<role>HUMAN</role>";
  if (with_image) {
    rendered += "<|vision_start|><|image_pad|><|vision_end|>";
  }
  rendered += user_text;
  rendered += "<|role_end|><role>ASSISTANT</role>\n<think>";

  std::vector<std::int32_t> ids = tokenizer.Encode(rendered, false, true);
  if (with_image) {
    std::vector<std::int32_t> expanded;
    expanded.reserve(ids.size() + static_cast<std::size_t>(nx * ny));
    int pads = 0;
    for (const std::int32_t id : ids) {
      if (id == pad) {
        ++pads;
        for (int k = 0; k < nx * ny; ++k)
          expanded.push_back(pad);
      } else {
        expanded.push_back(id);
      }
    }
    if (pads != 1) {
      if (error)
        *error = "prompt does not contain exactly one image pad";
      return false;
    }
    ids.swap(expanded);
  }
  if (!AssignRope(ids, pad, nx, ny, with_image, out, error))
    return false;
  if (with_image) {
    out->image_emb.assign(embeddings.begin(), embeddings.end());
  }
  return true;
}

PreparedPrompt IdentityOf(std::span<const std::int32_t> tokens) {
  PreparedPrompt out;
  out.tokens.assign(tokens.begin(), tokens.end());
  out.rope.resize(tokens.size() * 3);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const int v = static_cast<int>(i);
    out.rope[i * 3] = out.rope[i * 3 + 1] = out.rope[i * 3 + 2] = v;
  }
  out.next_logical = static_cast<std::int32_t>(tokens.size());
  out.hidden = kHidden;
  out.bind_rope = true;
  return out;
}

}  // namespace gufo::models::ling3::vision
