// KPF2 SPEC1 step 4: the 32k multi-block grouper is bit-identical to the
// serial oracle. For slots in {4096, 8192, 8193, 16384, 24576, 32768} with
// three seeded random routings each (ids in [0, 256), top-8 per token,
// random weights), run the oracle (GUFO_FP4_MOE_GROUP_SERIAL=1 path) and
// the new batch32k path, and memcmp every output: slot_tok, slot_w,
// token_slots, slot_dest, grp_e, grp_off, grp_cnt, grp_zp and the 4 header
// words.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/core/rocm_fp4/gufo_fp4_moe_group.hpp"
#include "src/core/rocm_fp4/gufo_fp4_quant.hpp"

namespace {

int g_failures = 0;

#define KPF2_CHECK_HIP(e)                                    \
  do {                                                       \
    const hipError_t _e = (e);                               \
    if (_e != hipSuccess) {                                  \
      std::printf("FAIL HIP %s:%d %s\n", __FILE__, __LINE__, \
                  hipGetErrorString(_e));                    \
      return 1;                                              \
    }                                                        \
  } while (0)

#define KPF2_CHECK_RC(rc, what)                 \
  do {                                          \
    if ((rc) != GUFO_FP4_OK) {                  \
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

int RunSlots(int slots, std::uint32_t seed_in) {
  const int kExperts = 256;
  const int kUsed = 8;
  const int kCap = 256;
  // Floor: the oracle audits n*kMaxExperts <= slots and places only
  // j < n*kMaxExperts (its i,s loops); batch32k mirrors that bound.
  const int n = slots / kUsed;
  const int grp_capacity = kExperts + slots / kCap + 2;
  std::uint32_t seed = seed_in;

  std::vector<int> ids(slots);
  std::vector<float> wts(slots);
  for (int j = 0; j < slots; ++j) {
    ids[j] = (int)(NextRandom(&seed) % (std::uint32_t)kExperts);
    wts[j] = (float)((int)(NextRandom(&seed) % 2001u) - 1000) / 1000.0f;
  }

  int* d_ids = nullptr;
  float* d_wts = nullptr;
  int* d_slot_tok = nullptr;
  float* d_slot_w = nullptr;
  int* d_token_slots = nullptr;
  int* d_slot_dest = nullptr;
  int* d_grp_e = nullptr;
  int* d_grp_off = nullptr;
  int* d_grp_cnt = nullptr;
  int* d_grp_zp = nullptr;
  int* d_hdr = nullptr;
  KPF2_CHECK_HIP(hipMalloc(&d_ids, (std::size_t)slots * sizeof(int)));
  KPF2_CHECK_HIP(hipMalloc(&d_wts, (std::size_t)slots * sizeof(float)));
  KPF2_CHECK_HIP(hipMalloc(&d_slot_tok, (std::size_t)slots * sizeof(int)));
  KPF2_CHECK_HIP(hipMalloc(&d_slot_w, (std::size_t)slots * sizeof(float)));
  KPF2_CHECK_HIP(hipMalloc(&d_token_slots, (std::size_t)slots * sizeof(int)));
  KPF2_CHECK_HIP(hipMalloc(&d_slot_dest, (std::size_t)slots * sizeof(int)));
  KPF2_CHECK_HIP(hipMalloc(&d_grp_e, (std::size_t)grp_capacity * sizeof(int)));
  KPF2_CHECK_HIP(
      hipMalloc(&d_grp_off, (std::size_t)grp_capacity * sizeof(int)));
  KPF2_CHECK_HIP(
      hipMalloc(&d_grp_cnt, (std::size_t)grp_capacity * sizeof(int)));
  KPF2_CHECK_HIP(
      hipMalloc(&d_grp_zp, (std::size_t)(grp_capacity + 1) * sizeof(int)));
  KPF2_CHECK_HIP(hipMalloc(&d_hdr, 4 * sizeof(int)));
  KPF2_CHECK_HIP(hipMemcpy(d_ids, ids.data(), (std::size_t)slots * sizeof(int),
                           hipMemcpyHostToDevice));
  KPF2_CHECK_HIP(hipMemcpy(d_wts, wts.data(),
                           (std::size_t)slots * sizeof(float),
                           hipMemcpyHostToDevice));

  std::vector<int> ref_slot_tok(slots), tst_slot_tok(slots);
  std::vector<float> ref_slot_w(slots), tst_slot_w(slots);
  std::vector<int> ref_token_slots(slots), tst_token_slots(slots);
  std::vector<int> ref_slot_dest(slots), tst_slot_dest(slots);
  std::vector<int> ref_grp_e(grp_capacity), tst_grp_e(grp_capacity);
  std::vector<int> ref_grp_off(grp_capacity), tst_grp_off(grp_capacity);
  std::vector<int> ref_grp_cnt(grp_capacity), tst_grp_cnt(grp_capacity);
  std::vector<int> ref_grp_zp(grp_capacity + 1), tst_grp_zp(grp_capacity + 1);
  std::vector<int> ref_hdr(4, -1), tst_hdr(4, -2);

  // Zero output buffers before each run so unplaced trailing slots (only
  // possible when slots % 8 != 0) compare deterministically.
  KPF2_CHECK_HIP(hipMemset(d_slot_tok, 0, (std::size_t)slots * sizeof(int)));
  KPF2_CHECK_HIP(hipMemset(d_slot_w, 0, (std::size_t)slots * sizeof(float)));
  KPF2_CHECK_HIP(hipMemset(d_token_slots, 0, (std::size_t)slots * sizeof(int)));
  KPF2_CHECK_HIP(hipMemset(d_slot_dest, 0, (std::size_t)slots * sizeof(int)));
  KPF2_CHECK_HIP(
      hipMemset(d_grp_e, 0, (std::size_t)grp_capacity * sizeof(int)));
  KPF2_CHECK_HIP(
      hipMemset(d_grp_off, 0, (std::size_t)grp_capacity * sizeof(int)));
  KPF2_CHECK_HIP(
      hipMemset(d_grp_cnt, 0, (std::size_t)grp_capacity * sizeof(int)));
  KPF2_CHECK_HIP(
      hipMemset(d_grp_zp, 0, (std::size_t)(grp_capacity + 1) * sizeof(int)));

  // Oracle: the serial body via the env knob (read fresh on every call).
  setenv("GUFO_FP4_MOE_GROUP_SERIAL", "1", 1);
  KPF2_CHECK_RC(gufo_fp4_moe_group_batch(d_ids, d_wts, nullptr, n, kUsed, kUsed,
                                         kExperts, kCap, d_slot_tok, d_slot_w,
                                         d_token_slots, d_grp_e, d_grp_off,
                                         d_grp_cnt, d_grp_zp, d_slot_dest,
                                         grp_capacity, slots, ref_hdr.data()),
                "oracle batch");
  unsetenv("GUFO_FP4_MOE_GROUP_SERIAL");
  KPF2_CHECK_HIP(hipMemcpy(ref_slot_tok.data(), d_slot_tok,
                           (std::size_t)slots * sizeof(int),
                           hipMemcpyDeviceToHost));
  KPF2_CHECK_HIP(hipMemcpy(ref_slot_w.data(), d_slot_w,
                           (std::size_t)slots * sizeof(float),
                           hipMemcpyDeviceToHost));
  KPF2_CHECK_HIP(hipMemcpy(ref_token_slots.data(), d_token_slots,
                           (std::size_t)slots * sizeof(int),
                           hipMemcpyDeviceToHost));
  KPF2_CHECK_HIP(hipMemcpy(ref_slot_dest.data(), d_slot_dest,
                           (std::size_t)slots * sizeof(int),
                           hipMemcpyDeviceToHost));
  KPF2_CHECK_HIP(hipMemcpy(ref_grp_e.data(), d_grp_e,
                           (std::size_t)grp_capacity * sizeof(int),
                           hipMemcpyDeviceToHost));
  KPF2_CHECK_HIP(hipMemcpy(ref_grp_off.data(), d_grp_off,
                           (std::size_t)grp_capacity * sizeof(int),
                           hipMemcpyDeviceToHost));
  KPF2_CHECK_HIP(hipMemcpy(ref_grp_cnt.data(), d_grp_cnt,
                           (std::size_t)grp_capacity * sizeof(int),
                           hipMemcpyDeviceToHost));
  KPF2_CHECK_HIP(hipMemcpy(ref_grp_zp.data(), d_grp_zp,
                           (std::size_t)(grp_capacity + 1) * sizeof(int),
                           hipMemcpyDeviceToHost));

  // New path: header pre-zeroed (OK is 0), no host readback inside.
  KPF2_CHECK_HIP(hipMemset(d_slot_tok, 0, (std::size_t)slots * sizeof(int)));
  KPF2_CHECK_HIP(hipMemset(d_slot_w, 0, (std::size_t)slots * sizeof(float)));
  KPF2_CHECK_HIP(hipMemset(d_token_slots, 0, (std::size_t)slots * sizeof(int)));
  KPF2_CHECK_HIP(hipMemset(d_slot_dest, 0, (std::size_t)slots * sizeof(int)));
  KPF2_CHECK_HIP(
      hipMemset(d_grp_e, 0, (std::size_t)grp_capacity * sizeof(int)));
  KPF2_CHECK_HIP(
      hipMemset(d_grp_off, 0, (std::size_t)grp_capacity * sizeof(int)));
  KPF2_CHECK_HIP(
      hipMemset(d_grp_cnt, 0, (std::size_t)grp_capacity * sizeof(int)));
  KPF2_CHECK_HIP(
      hipMemset(d_grp_zp, 0, (std::size_t)(grp_capacity + 1) * sizeof(int)));
  KPF2_CHECK_HIP(hipMemset(d_hdr, 0, 4 * sizeof(int)));
  KPF2_CHECK_RC(
      gufo_fp4_moe_group_batch32k(
          d_ids, d_wts, nullptr, n, kUsed, kUsed, kExperts, kCap, d_slot_tok,
          d_slot_w, d_token_slots, d_grp_e, d_grp_off, d_grp_cnt, d_grp_zp,
          d_slot_dest, grp_capacity, slots, d_hdr),
      "batch32k");
  KPF2_CHECK_HIP(hipMemcpy(tst_slot_tok.data(), d_slot_tok,
                           (std::size_t)slots * sizeof(int),
                           hipMemcpyDeviceToHost));
  KPF2_CHECK_HIP(hipMemcpy(tst_slot_w.data(), d_slot_w,
                           (std::size_t)slots * sizeof(float),
                           hipMemcpyDeviceToHost));
  KPF2_CHECK_HIP(hipMemcpy(tst_token_slots.data(), d_token_slots,
                           (std::size_t)slots * sizeof(int),
                           hipMemcpyDeviceToHost));
  KPF2_CHECK_HIP(hipMemcpy(tst_slot_dest.data(), d_slot_dest,
                           (std::size_t)slots * sizeof(int),
                           hipMemcpyDeviceToHost));
  KPF2_CHECK_HIP(hipMemcpy(tst_grp_e.data(), d_grp_e,
                           (std::size_t)grp_capacity * sizeof(int),
                           hipMemcpyDeviceToHost));
  KPF2_CHECK_HIP(hipMemcpy(tst_grp_off.data(), d_grp_off,
                           (std::size_t)grp_capacity * sizeof(int),
                           hipMemcpyDeviceToHost));
  KPF2_CHECK_HIP(hipMemcpy(tst_grp_cnt.data(), d_grp_cnt,
                           (std::size_t)grp_capacity * sizeof(int),
                           hipMemcpyDeviceToHost));
  KPF2_CHECK_HIP(hipMemcpy(tst_grp_zp.data(), d_grp_zp,
                           (std::size_t)(grp_capacity + 1) * sizeof(int),
                           hipMemcpyDeviceToHost));
  KPF2_CHECK_HIP(
      hipMemcpy(tst_hdr.data(), d_hdr, 4 * sizeof(int), hipMemcpyDeviceToHost));

  hipFree(d_ids);
  hipFree(d_wts);
  hipFree(d_slot_tok);
  hipFree(d_slot_w);
  hipFree(d_token_slots);
  hipFree(d_slot_dest);
  hipFree(d_grp_e);
  hipFree(d_grp_off);
  hipFree(d_grp_cnt);
  hipFree(d_grp_zp);
  hipFree(d_hdr);

  bool same = std::memcmp(ref_hdr.data(), tst_hdr.data(), 4 * sizeof(int)) == 0;
  if (!same) {
    std::printf("FAIL slots=%d header: ref=[%d %d %d %d] tst=[%d %d %d %d]\n",
                slots, ref_hdr[0], ref_hdr[1], ref_hdr[2], ref_hdr[3],
                tst_hdr[0], tst_hdr[1], tst_hdr[2], tst_hdr[3]);
    ++g_failures;
    return 1;
  }
  const int n_groups = ref_hdr[0];
  const bool arrays_same =
      std::memcmp(ref_slot_tok.data(), tst_slot_tok.data(),
                  (std::size_t)slots * sizeof(int)) == 0 &&
      std::memcmp(ref_slot_w.data(), tst_slot_w.data(),
                  (std::size_t)slots * sizeof(float)) == 0 &&
      std::memcmp(ref_token_slots.data(), tst_token_slots.data(),
                  (std::size_t)slots * sizeof(int)) == 0 &&
      std::memcmp(ref_slot_dest.data(), tst_slot_dest.data(),
                  (std::size_t)slots * sizeof(int)) == 0 &&
      std::memcmp(ref_grp_e.data(), tst_grp_e.data(),
                  (std::size_t)n_groups * sizeof(int)) == 0 &&
      std::memcmp(ref_grp_off.data(), tst_grp_off.data(),
                  (std::size_t)n_groups * sizeof(int)) == 0 &&
      std::memcmp(ref_grp_cnt.data(), tst_grp_cnt.data(),
                  (std::size_t)n_groups * sizeof(int)) == 0 &&
      std::memcmp(ref_grp_zp.data(), tst_grp_zp.data(),
                  (std::size_t)(n_groups + 1) * sizeof(int)) == 0;
  std::printf("%s slots=%d seed=0x%08x groups=%d tiles=%d %s\n",
              arrays_same ? "PASS" : "FAIL", slots, seed_in, n_groups,
              ref_hdr[2], arrays_same ? "byte-identical" : "MISMATCH");
  if (!arrays_same)
    ++g_failures;
  return arrays_same ? 0 : 1;
}

}  // namespace

int main() {
  const int slots_list[] = {4096, 8192, 8193, 16384, 24576, 32768};
  std::uint32_t seed = 0x2F9A10u;
  for (int s = 0; s < 6; ++s) {
    for (int r = 0; r < 3; ++r) {
      if (RunSlots(slots_list[s], seed) != 0)
        return 1;
      seed = seed * 1664525u + 1013904223u;  // LCG: 18 distinct routings
    }
  }
  if (g_failures != 0) {
    std::printf("FAIL %d cases mismatched\n", g_failures);
    return 1;
  }
  std::printf("PASS all 18 cases byte-identical\n");
  return 0;
}
