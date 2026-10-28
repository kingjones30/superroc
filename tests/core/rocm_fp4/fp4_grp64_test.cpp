// MG64 SPEC1 step 5: 64-token grouped FP4 GEMM must be bit-identical to the
// 16-token path. Random ROCmFP4 weights (n = 1792, k = 2048, 8 experts) and
// int8 activations via superroc_fp4_wmma_quant; group arrays built exactly like
// superroc_fp4_moe_group_batch makes them (counting-sort slot order, grp_zp
// from the (cnt+15)/16 prefix). For each layout: gemm_grp_bk(..., 4) vs
// gemm_grp64 on the same inputs, memcmp the outputs.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "src/core/rocm_fp4/superroc_fp4_moe_group.hpp"
#include "src/core/rocm_fp4/superroc_fp4_quant.hpp"
#include "src/core/rocm_fp4/superroc_fp4_wmma.hpp"

namespace {

int g_failures = 0;

#define MG64_CHECK_HIP(e)                                    \
  do {                                                       \
    const hipError_t _e = (e);                               \
    if (_e != hipSuccess) {                                  \
      std::printf("FAIL HIP %s:%d %s\n", __FILE__, __LINE__, \
                  hipGetErrorString(_e));                    \
      return 1;                                              \
    }                                                        \
  } while (0)

#define MG64_CHECK_RC(rc, what)                 \
  do {                                          \
    if ((rc) != SUPERROC_FP4_OK) {              \
      std::printf("FAIL %s rc=%d\n", what, rc); \
      return 1;                                 \
    }                                           \
  } while (0)

std::uint32_t NextRandom(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

int RunLayout(const std::vector<int>& counts, std::uint32_t seed) {
  const int kN = 1792;
  const int kK = 2048;
  const int kExperts = 8;
  const int kQtype = SUPERROC_FP4_Q4_0_ROCMFP4;
  const int n_groups = static_cast<int>(counts.size());
  int slots = 0;
  for (int c : counts)
    slots += c;

  // Group arrays exactly like the builder: counting-sort slot order.
  std::vector<int> grp_e(n_groups), grp_off(n_groups), grp_cnt(n_groups);
  std::vector<int> grp_zp(n_groups + 1);
  int off_acc = 0, tile_acc = 0;
  for (int g = 0; g < n_groups; ++g) {
    grp_e[g] = g % kExperts;
    grp_off[g] = off_acc;
    grp_cnt[g] = counts[g];
    grp_zp[g] = tile_acc;
    off_acc += counts[g];
    tile_acc += (counts[g] + 15) / 16;
  }
  grp_zp[n_groups] = tile_acc;
  const int total_tiles = tile_acc;
  int total_blocks = 0;
  for (int c : counts)
    total_blocks += (c + 63) / 64;

  std::vector<int> slot_dest(slots);
  for (int g = 0; g < n_groups; ++g) {
    for (int l = 0; l < counts[g]; ++l) {
      slot_dest[grp_off[g] + l] = ((grp_zp[g] + l / 16) << 4) | (l & 15);
    }
  }

  // Random weights: any byte is a valid FP4 nibble / ue4m3 scale byte.
  const std::size_t rb = superroc_fp4_row_bytes(kQtype, kK);
  const std::size_t stride = static_cast<std::size_t>(kN) * rb;
  std::vector<std::uint8_t> weights(static_cast<std::size_t>(kExperts) *
                                    stride);
  for (std::uint8_t& b : weights)
    b = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFFu);

  // Random activations.
  std::vector<float> x(static_cast<std::size_t>(slots) * kK);
  for (float& v : x) {
    v = static_cast<float>(static_cast<int>(NextRandom(&seed) & 0xFFFFu) -
                           32768) /
        32768.0f * 2.0f;
  }

  const std::size_t num_kb = static_cast<std::size_t>(kK / 32);
  const std::size_t q8_bytes =
      static_cast<std::size_t>(total_tiles) * num_kb * 576;
  const std::size_t y_bytes =
      static_cast<std::size_t>(slots) * kN * sizeof(float);

  std::uint8_t* d_w = nullptr;
  float* d_x = nullptr;
  int* d_grp_e = nullptr;
  int* d_grp_off = nullptr;
  int* d_grp_cnt = nullptr;
  int* d_grp_zp = nullptr;
  int* d_grp_zb = nullptr;
  int* d_slot_dest = nullptr;
  std::uint8_t* d_q8 = nullptr;
  float* d_y_ref = nullptr;
  float* d_y_tst = nullptr;
  MG64_CHECK_HIP(hipMalloc(&d_w, weights.size()));
  MG64_CHECK_HIP(hipMalloc(&d_x, x.size() * sizeof(float)));
  MG64_CHECK_HIP(hipMalloc(&d_grp_e, n_groups * sizeof(int)));
  MG64_CHECK_HIP(hipMalloc(&d_grp_off, n_groups * sizeof(int)));
  MG64_CHECK_HIP(hipMalloc(&d_grp_cnt, n_groups * sizeof(int)));
  MG64_CHECK_HIP(hipMalloc(&d_grp_zp, (n_groups + 1) * sizeof(int)));
  MG64_CHECK_HIP(hipMalloc(&d_grp_zb, (n_groups + 1) * sizeof(int)));
  MG64_CHECK_HIP(hipMalloc(&d_slot_dest, slots * sizeof(int)));
  MG64_CHECK_HIP(hipMalloc(&d_q8, q8_bytes));
  MG64_CHECK_HIP(hipMalloc(&d_y_ref, y_bytes));
  MG64_CHECK_HIP(hipMalloc(&d_y_tst, y_bytes));

  MG64_CHECK_HIP(
      hipMemcpy(d_w, weights.data(), weights.size(), hipMemcpyHostToDevice));
  MG64_CHECK_HIP(hipMemcpy(d_x, x.data(), x.size() * sizeof(float),
                           hipMemcpyHostToDevice));
  MG64_CHECK_HIP(hipMemcpy(d_grp_e, grp_e.data(), n_groups * sizeof(int),
                           hipMemcpyHostToDevice));
  MG64_CHECK_HIP(hipMemcpy(d_grp_off, grp_off.data(), n_groups * sizeof(int),
                           hipMemcpyHostToDevice));
  MG64_CHECK_HIP(hipMemcpy(d_grp_cnt, grp_cnt.data(), n_groups * sizeof(int),
                           hipMemcpyHostToDevice));
  MG64_CHECK_HIP(hipMemcpy(d_grp_zp, grp_zp.data(),
                           (n_groups + 1) * sizeof(int),
                           hipMemcpyHostToDevice));
  MG64_CHECK_HIP(hipMemcpy(d_slot_dest, slot_dest.data(), slots * sizeof(int),
                           hipMemcpyHostToDevice));

  // The real block prefix from the device builder.
  MG64_CHECK_RC(
      superroc_fp4_moe_group_blocks(d_grp_cnt, n_groups, 64, d_grp_zb, nullptr),
      "blocks");
  std::vector<int> grp_zb(n_groups + 1, -1);
  MG64_CHECK_HIP(hipMemcpy(grp_zb.data(), d_grp_zb,
                           (n_groups + 1) * sizeof(int),
                           hipMemcpyDeviceToHost));
  int host_acc = 0;
  for (int g = 0; g < n_groups; ++g) {
    if (grp_zb[g] != host_acc) {
      std::printf("FAIL blocks layout {");
      for (int h = 0; h < n_groups; ++h)
        std::printf("%s%d", h ? "," : "", counts[h]);
      std::printf("}: grp_zb[%d]=%d want %d\n", g, grp_zb[g], host_acc);
      return 1;
    }
    host_acc += (counts[g] + 63) / 64;
  }
  if (grp_zb[n_groups] != host_acc) {
    std::printf("FAIL blocks total: got %d want %d\n", grp_zb[n_groups],
                host_acc);
    return 1;
  }

  MG64_CHECK_RC(
      superroc_fp4_wmma_quant(d_x, nullptr, d_slot_dest, d_q8, slots, kK),
      "quant");
  MG64_CHECK_HIP(hipMemset(d_y_ref, 0, y_bytes));
  MG64_CHECK_HIP(hipMemset(d_y_tst, 0, y_bytes));
  MG64_CHECK_RC(
      superroc_fp4_wmma_gemm_grp_bk(kQtype, d_w, stride, d_grp_e, d_grp_off,
                                    d_grp_cnt, d_grp_zp, n_groups, total_tiles,
                                    d_q8, d_y_ref, nullptr, kK, kN, /*bk=*/4),
      "gemm_grp_bk");
  MG64_CHECK_RC(
      superroc_fp4_wmma_gemm_grp64(kQtype, d_w, stride, d_grp_e, d_grp_off,
                                   d_grp_cnt, d_grp_zp, n_groups, total_tiles,
                                   d_q8, d_y_tst, nullptr, kK, kN, d_grp_zb),
      "gemm_grp64");

  std::vector<float> y_ref(slots * kN), y_tst(slots * kN);
  MG64_CHECK_HIP(
      hipMemcpy(y_ref.data(), d_y_ref, y_bytes, hipMemcpyDeviceToHost));
  MG64_CHECK_HIP(
      hipMemcpy(y_tst.data(), d_y_tst, y_bytes, hipMemcpyDeviceToHost));

  hipFree(d_w);
  hipFree(d_x);
  hipFree(d_grp_e);
  hipFree(d_grp_off);
  hipFree(d_grp_cnt);
  hipFree(d_grp_zp);
  hipFree(d_grp_zb);
  hipFree(d_slot_dest);
  hipFree(d_q8);
  hipFree(d_y_ref);
  hipFree(d_y_tst);

  const bool same = std::memcmp(y_ref.data(), y_tst.data(), y_bytes) == 0;
  std::printf("%s layout {", same ? "PASS" : "FAIL");
  for (int g = 0; g < n_groups; ++g)
    std::printf("%s%d", g ? "," : "", counts[g]);
  std::printf("}: slots=%d groups=%d tiles=%d blocks=%d %s\n", slots, n_groups,
              total_tiles, total_blocks, same ? "byte-identical" : "MISMATCH");
  if (!same)
    ++g_failures;
  return same ? 0 : 1;
}

}  // namespace

int main() {
  const std::vector<std::vector<int>> layouts = {
      {3}, {16}, {17}, {64}, {65}, {200, 5, 64, 1}, {128, 128, 128}};
  std::uint32_t seed = 0x6D3640u;
  for (const auto& counts : layouts) {
    if (RunLayout(counts, seed) != 0)
      return 1;
    seed ^= 0x9E3779B9u;
  }
  if (g_failures != 0) {
    std::printf("FAIL %d layouts mismatched\n", g_failures);
    return 1;
  }
  std::printf("PASS all %d layouts byte-identical\n",
              static_cast<int>(layouts.size()));
  return 0;
}
