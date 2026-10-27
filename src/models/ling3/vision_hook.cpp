#include <cstdio>
#include <cstring>
#include <vector>

#include "src/models/ling3/vision_prompt.hpp"

#if defined(LING3_WITH_ROCM)
#include <hip/hip_runtime.h>

#include "src/models/ling3/kernels/ling3_mla.hpp"
#endif

namespace superroc::models::ling3::vision {
namespace {

struct ArmedPlan {
  bool on{false};
  std::vector<std::int32_t> rope;
  std::vector<float> image_emb;
  std::vector<std::int32_t> image_phys;
  std::int32_t n_prompt{0};
  std::int32_t next_logical{0};
  int hidden{2560};
  bool bind_rope{false};
};

ArmedPlan& Plan() {
  static ArmedPlan plan;
  return plan;
}

#if defined(LING3_WITH_ROCM)
int* g_dev = nullptr;
int g_cap = 0;

void Ensure(int n_phys) {
  if (n_phys <= g_cap)
    return;
  int cap = g_cap > 0 ? g_cap : 4096;
  while (cap < n_phys)
    cap *= 2;
  int* fresh = nullptr;
  if (hipMalloc(&fresh, static_cast<std::size_t>(cap) * 3 * sizeof(int)) !=
      hipSuccess) {
    std::fprintf(stderr, "lvv: mrope alloc failed\n");
    return;
  }
  if (g_dev != nullptr) {
    hipMemcpy(fresh, g_dev, static_cast<std::size_t>(g_cap) * 3 * sizeof(int),
              hipMemcpyDeviceToDevice);
    hipFree(g_dev);
  }
  g_dev = fresh;
  g_cap = cap;
  ::ling3::mla_set_mrope(g_dev);
}
#endif

}  // namespace

void Arm(const PreparedPrompt& prompt) {
  ArmedPlan& plan = Plan();
  plan.on = true;
  plan.rope = prompt.rope;
  plan.image_emb = prompt.image_emb;
  plan.image_phys = prompt.image_phys;
  plan.n_prompt = static_cast<std::int32_t>(prompt.tokens.size());
  plan.next_logical = prompt.next_logical;
  plan.hidden = prompt.hidden;
  plan.bind_rope = prompt.bind_rope;
}

void Clear() {
  Plan() = ArmedPlan{};
#if defined(LING3_WITH_ROCM)
  ::ling3::mla_set_mrope(nullptr);
#endif
}

bool Armed() {
  return Plan().on;
}

void OnHidden(float* hidden, int pos0, int n, int width, void* stream) {
  ArmedPlan& plan = Plan();
  if (!plan.on || hidden == nullptr || n <= 0)
    return;
#if !defined(LING3_WITH_ROCM)
  (void)width;
  (void)stream;
  return;
#else
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  if (width == plan.hidden && !plan.image_emb.empty()) {
    const int rows = static_cast<int>(plan.image_phys.size());
    for (int r = 0; r < rows; ++r) {
      const int phys = plan.image_phys[static_cast<std::size_t>(r)];
      if (phys < pos0 || phys >= pos0 + n)
        continue;
      const int local = phys - pos0;
      hipMemcpyAsync(
          hidden +
              static_cast<std::size_t>(local) * static_cast<std::size_t>(width),
          plan.image_emb.data() +
              static_cast<std::size_t>(r) * static_cast<std::size_t>(width),
          static_cast<std::size_t>(width) * sizeof(float),
          hipMemcpyHostToDevice, hip_stream);
    }
  }
  if (!plan.bind_rope)
    return;
  const int end = pos0 + n;
  Ensure(end);
  if (g_dev == nullptr || end > g_cap)
    return;
  std::vector<int> host(static_cast<std::size_t>(n) * 3);
  for (int i = 0; i < n; ++i) {
    const int phys = pos0 + i;
    int t;
    int h;
    int w;
    if (phys >= 0 && phys < plan.n_prompt) {
      const std::size_t base = static_cast<std::size_t>(phys) * 3;
      t = plan.rope[base];
      h = plan.rope[base + 1];
      w = plan.rope[base + 2];
    } else {
      const int logical = plan.next_logical + (phys - plan.n_prompt);
      t = h = w = logical;
    }
    host[static_cast<std::size_t>(i) * 3] = t;
    host[static_cast<std::size_t>(i) * 3 + 1] = h;
    host[static_cast<std::size_t>(i) * 3 + 2] = w;
  }
  // Synchronous: `host` dies when this function returns, and the rope kernel
  // reads the table on the same stream immediately afterwards.
  if (hipMemcpy(g_dev + static_cast<std::size_t>(pos0) * 3, host.data(),
                host.size() * sizeof(int),
                hipMemcpyHostToDevice) != hipSuccess) {
    std::fprintf(stderr, "lvv: mrope upload failed\n");
    return;
  }
  ::ling3::mla_set_mrope(g_dev);
#endif
}

}  // namespace superroc::models::ling3::vision
