// ling3-moe-group-test — is the DEVICE expert grouping bit-identical to the host
// counting sort it replaces?
//
// PHASE8 item 2. The grouped GEMM's result depends on the grouping being not merely
// equivalent but IDENTICAL, including the order of slots inside an expert: the counting
// sort is stable by construction, and ling3_slot_gather_sum folds a token's eight slots in
// slot order. So this test compares every output array element for element, and it also
// checks the two ways the old host loop could fail loudly -- a router status word and an
// expert id outside [0, n_expert) -- because the device kernel cannot call fail() itself
// and reports through one sticky status word instead.
//
// It also runs first at its smallest shape, per the lane's rule after the 2026-10-02 fault.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>

#include "src/models/ling3/kernels/ling3_moe_group.hpp"
#include "src/models/ling3/kernels/ling3_quant.hpp"

namespace {

int failures = 0;
#define CHECK(cond, msg)                                                        \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (msg));       \
      ++failures;                                                               \
    }                                                                           \
  } while (0)

std::uint64_t rng = 0x9E3779B97F4A7C15ULL;
std::uint32_t Next() {
  rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<std::uint32_t>(rng >> 32);
}

struct Host {
  std::vector<int> slot_tok, token_slots, grp_e, grp_off, grp_cnt, grp_zp, slot_dest;
  std::vector<float> slot_w;
  int n_groups = 0, max_cnt = 0, tiles = 0, status = 0;
  // A transcription of the loop rocm_backend.cpp used to run, for comparison.
  void Run(const std::vector<int> & ids, const std::vector<float> & wts,
           const std::vector<int> & rc, int n, int kMaxExperts, int n_expert, int cap,
           int grp_capacity) {
    const int slots = n * kMaxExperts;
    slot_tok.assign(slots, -1); token_slots.assign(slots, -1);
    slot_w.assign(slots, 0.f); slot_dest.assign(slots, -1);
    grp_e.clear(); grp_off.clear(); grp_cnt.clear();
    for (int i = 0; i < n; ++i) {
      if (!rc.empty() && rc[i] != 0) { status = 1; return; }
    }
    std::vector<int> counts(n_expert + 1, 0), fill(n_expert, 0);
    for (int j = 0; j < slots; ++j) {
      const int e = ids[j];
      if (e < 0 || e >= n_expert) { status = 1; return; }
      counts[e + 1]++;
    }
    for (int e = 0; e < n_expert; ++e) { counts[e + 1] += counts[e]; fill[e] = counts[e]; }
    for (int i = 0; i < n; ++i) {
      for (int s = 0; s < kMaxExperts; ++s) {
        const int j = i * kMaxExperts + s;
        const int p = fill[ids[j]]++;
        slot_tok[p] = i; slot_w[p] = wts[j]; token_slots[j] = p;
      }
    }
    int tile_acc = 0;
    for (int e = 0; e < n_expert; ++e) {
      const int c = counts[e + 1] - counts[e];
      for (int o = 0; o < c; o += cap) {
        const int take = (c - o) < cap ? (c - o) : cap;
        if ((int) grp_e.size() >= grp_capacity) { status = 1; return; }
        grp_e.push_back(e); grp_off.push_back(counts[e] + o); grp_cnt.push_back(take);
        if (take > max_cnt) max_cnt = take;
        tile_acc += (take + 15) / 16;
      }
    }
    n_groups = (int) grp_e.size();
    tiles = tile_acc;
    grp_zp.assign(n_groups + 1, 0);
    for (int gi = 0; gi < n_groups; ++gi) {
      int acc = 0;
      for (int h = 0; h < gi; ++h) acc += (grp_cnt[h] + 15) / 16;
      grp_zp[gi] = acc;
    }
    grp_zp[n_groups] = tiles;
    for (int gi = 0; gi < n_groups; ++gi) {
      for (int c = 0; c < grp_cnt[gi]; ++c) {
        slot_dest[grp_off[gi] + c] = ((grp_zp[gi] + (c >> 4)) << 4) | (c & 15);
      }
    }
  }
};

void RunCase(const char * label, int n, int kMaxExperts, int n_expert, int cap,
             int skew, int poison_id, int poison_rc) {
  const int slots = n * kMaxExperts;
  // Mirror the PRODUCTION allocation (n_expert + SLOTS/8 + 2) when cap is Ling's 64,
  // which is what the host formula was sized for. This fixture first ran with cap=4
  // and kept the slots/8 term, which under-allocates at that cap and made the
  // kernel's audit refuse a call the host would have accepted -- the audit was right
  // and the fixture was wrong.
  const int grp_capacity = n_expert + slots / (cap < 8 ? cap : 8) + 3;
  std::vector<int> ids(slots), rc(n, 0);
  std::vector<float> wts(slots);
  for (int j = 0; j < slots; ++j) {
    // `skew` sends most slots to one expert, which is what real routing does (Jay
    // measured one expert taking 488 of 512 tokens) and is what makes kMoeGroupCap bite.
    ids[j] = (skew > 0 && (j % 100) < skew) ? 7 : (int) (Next() % (std::uint32_t) n_expert);
    wts[j] = 0.05f + 0.2f * (float) Next() / 4294967296.0F;
  }
  if (poison_id >= 0) {
    ids[slots / 2] = n_expert + 5;   // must be REJECTED, never indexed
  }
  if (poison_rc) {
    rc[n / 2] = 1;                    // the router said this token failed
  }

  Host h;
  h.Run(ids, wts, rc, n, kMaxExperts, n_expert, cap, grp_capacity);

  int * d_ids = nullptr; float * d_wts = nullptr; int * d_rc = nullptr;
  int * d_slot_tok = nullptr; float * d_slot_w = nullptr; int * d_tok_slots = nullptr;
  int * d_ge = nullptr; int * d_go = nullptr; int * d_gc = nullptr;
  int * d_gz = nullptr; int * d_sd = nullptr;
  const size_t pad = LING3_TAIL_PAD;
  bool a = hipMalloc(&d_ids, (slots + pad) * sizeof(int)) == hipSuccess;
  a = a && hipMalloc(&d_wts, (slots + pad) * sizeof(float)) == hipSuccess;
  a = a && hipMalloc(&d_rc, ((size_t) n + pad) * sizeof(int)) == hipSuccess;
  a = a && hipMalloc(&d_slot_tok, (slots + pad) * sizeof(int)) == hipSuccess;
  a = a && hipMalloc(&d_slot_w, (slots + pad) * sizeof(float)) == hipSuccess;
  a = a && hipMalloc(&d_tok_slots, (slots + pad) * sizeof(int)) == hipSuccess;
  a = a && hipMalloc(&d_ge, (grp_capacity + pad) * sizeof(int)) == hipSuccess;
  a = a && hipMalloc(&d_go, (grp_capacity + pad) * sizeof(int)) == hipSuccess;
  a = a && hipMalloc(&d_gc, (grp_capacity + pad) * sizeof(int)) == hipSuccess;
  a = a && hipMalloc(&d_gz, (grp_capacity + 1 + pad) * sizeof(int)) == hipSuccess;
  a = a && hipMalloc(&d_sd, (slots + pad) * sizeof(int)) == hipSuccess;
  CHECK(a, "device allocations");
  if (!a) return;

  hipMemcpy(d_ids, ids.data(), slots * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(d_wts, wts.data(), slots * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_rc, rc.data(), (size_t) n * sizeof(int), hipMemcpyHostToDevice);
  // Poison the outputs so an unwritten element cannot pass by being zero.
  hipMemset(d_slot_tok, 0x7f, slots * sizeof(int));
  hipMemset(d_tok_slots, 0x7f, slots * sizeof(int));
  hipMemset(d_sd, 0x7f, slots * sizeof(int));
  hipMemset(d_ge, 0x7f, grp_capacity * sizeof(int));
  hipMemset(d_gz, 0x7f, (grp_capacity + 1) * sizeof(int));

  int hdr[LS1_HDR_WORDS] = {0, 0, 0, 0};
  const int rc_h = ling3_moe_group_batch(d_ids, d_wts, d_rc, n, kMaxExperts, 8, n_expert,
                                         cap, d_slot_tok, d_slot_w, d_tok_slots,
                                         d_ge, d_go, d_gc, d_gz, d_sd,
                                         grp_capacity, slots, hdr);
  CHECK(rc_h == LING3_OK, "ling3_moe_group_batch rc");
  CHECK(hipDeviceSynchronize() == hipSuccess, "sync");
  CHECK(hdr[LS1_HDR_STATUS] == (h.status ? 1 : 0), "status word disagrees with the host");

  if (!h.status) {
    std::vector<int> gt(slots), gs(slots), gd(slots), ge(grp_capacity), go(grp_capacity),
        gc(grp_capacity), gz(grp_capacity + 1);
    std::vector<float> gw(slots);
    hipMemcpy(gt.data(), d_slot_tok, slots * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(gw.data(), d_slot_w, slots * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(gs.data(), d_tok_slots, slots * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(gd.data(), d_sd, slots * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(ge.data(), d_ge, grp_capacity * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(go.data(), d_go, grp_capacity * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(gc.data(), d_gc, grp_capacity * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(gz.data(), d_gz, (grp_capacity + 1) * sizeof(int), hipMemcpyDeviceToHost);

    int bad = 0;
    for (int i = 0; i < slots; ++i) {
      if (gt[i] != h.slot_tok[i]) ++bad;
      if (gw[i] != h.slot_w[i]) ++bad;
      if (gs[i] != h.token_slots[i]) ++bad;
      if (gd[i] != h.slot_dest[i]) ++bad;
    }
    int gbad = 0;
    for (int gi = 0; gi < h.n_groups; ++gi) {
      if (ge[gi] != h.grp_e[gi] || go[gi] != h.grp_off[gi] || gc[gi] != h.grp_cnt[gi]) ++gbad;
    }
    for (int gi = 0; gi <= h.n_groups; ++gi) {
      if (gz[gi] != h.grp_zp[gi]) ++gbad;
    }
    CHECK(bad == 0, "per-slot arrays differ from the host counting sort");
    CHECK(gbad == 0, "group arrays differ from the host sort");
    CHECK(hdr[LS1_HDR_NGROUPS] == h.n_groups, "n_groups differs");
    CHECK(hdr[LS1_HDR_MAXCNT] == h.max_cnt, "max_cnt differs");
    CHECK(hdr[LS1_HDR_TILES] == h.tiles, "tile count differs");
    std::printf("  GROUP %-28s n=%4d slots=%5d groups=%5d max_cnt=%3d tiles=%5d  "
                "slot_diffs=%d group_diffs=%d status=%d\n",
                label, n, slots, h.n_groups, h.max_cnt, h.tiles, bad, gbad, hdr[LS1_HDR_STATUS]);
  } else {
    std::printf("  GROUP %-28s n=%4d slots=%5d  rejected as the host would "
                "(status=%d)\n", label, n, slots, hdr[LS1_HDR_STATUS]);
  }

  hipFree(d_ids); hipFree(d_wts); hipFree(d_rc); hipFree(d_slot_tok); hipFree(d_slot_w);
  hipFree(d_tok_slots); hipFree(d_ge); hipFree(d_go); hipFree(d_gc); hipFree(d_gz); hipFree(d_sd);
}

// LK1: both kernels, memcmp of every output, including error paths.

extern "C" int ling3_moe_group_debug_counts(int * host, int n_out);

struct DevOut {
    std::vector<int> slot_tok, token_slots, grp_e, grp_off, grp_cnt, grp_zp, slot_dest, counts;
    std::vector<float> slot_w;
    int hdr[LS1_HDR_WORDS]{};
    int batch_rc = -1;
};

int DiffInts(const char * name, const std::vector<int> & a, const std::vector<int> & b) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "DIFF %s size %zu vs %zu\n", name, a.size(), b.size());
        return 1;
    }
    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            if (bad < 4) {
                std::fprintf(stderr, "DIFF %s [%zu] serial=%d parallel=%d\n", name, i, a[i], b[i]);
            }
            ++bad;
        }
    }
    if (bad) std::fprintf(stderr, "DIFF %s total %d\n", name, bad);
    return bad ? 1 : 0;
}

int DiffFloats(const char * name, const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() ||
        std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
        int shown = 0;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n && shown < 4; ++i) {
            if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
                std::fprintf(stderr, "DIFF %s [%zu] bits\n", name, i);
                ++shown;
            }
        }
        std::fprintf(stderr, "DIFF %s\n", name);
        return 1;
    }
    return 0;
}

bool AllocCopy(const std::vector<int> & ids, const std::vector<float> & wts,
               const std::vector<int> & rc, int n, int kMax, int n_expert, int cap,
               int grp_capacity, int serial, DevOut * out) {
    const int slots = n * kMax;
    const size_t pad = LING3_TAIL_PAD;
    int *d_ids = nullptr, *d_rc = nullptr, *d_slot_tok = nullptr, *d_tok = nullptr;
    int *d_ge = nullptr, *d_go = nullptr, *d_gc = nullptr, *d_gz = nullptr, *d_sd = nullptr;
    float *d_wts = nullptr, *d_slot_w = nullptr;
    auto ok = [&](hipError_t e) { return e == hipSuccess; };
    bool a = ok(hipMalloc(&d_ids, (slots + pad) * sizeof(int)));
    a = a && ok(hipMalloc(&d_wts, (slots + pad) * sizeof(float)));
    a = a && ok(hipMalloc(&d_rc, ((size_t)n + pad) * sizeof(int)));
    a = a && ok(hipMalloc(&d_slot_tok, (slots + pad) * sizeof(int)));
    a = a && ok(hipMalloc(&d_slot_w, (slots + pad) * sizeof(float)));
    a = a && ok(hipMalloc(&d_tok, (slots + pad) * sizeof(int)));
    a = a && ok(hipMalloc(&d_ge, (grp_capacity + pad) * sizeof(int)));
    a = a && ok(hipMalloc(&d_go, (grp_capacity + pad) * sizeof(int)));
    a = a && ok(hipMalloc(&d_gc, (grp_capacity + pad) * sizeof(int)));
    a = a && ok(hipMalloc(&d_gz, (grp_capacity + 1 + pad) * sizeof(int)));
    a = a && ok(hipMalloc(&d_sd, (slots + pad) * sizeof(int)));
    if (!a) {
        std::fprintf(stderr, "alloc failed\n");
        return false;
    }
    hipMemcpy(d_ids, ids.data(), slots * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_wts, wts.data(), slots * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_rc, rc.data(), (size_t)n * sizeof(int), hipMemcpyHostToDevice);
    hipMemset(d_slot_tok, 0x7f, slots * sizeof(int));
    hipMemset(d_slot_w, 0x7f, slots * sizeof(float));
    hipMemset(d_tok, 0x7f, slots * sizeof(int));
    hipMemset(d_sd, 0x7f, slots * sizeof(int));
    hipMemset(d_ge, 0x7f, grp_capacity * sizeof(int));
    hipMemset(d_go, 0x7f, grp_capacity * sizeof(int));
    hipMemset(d_gc, 0x7f, grp_capacity * sizeof(int));
    hipMemset(d_gz, 0x7f, (grp_capacity + 1) * sizeof(int));

    setenv("LING3_MOE_GROUP_SERIAL", serial ? "1" : "0", 1);
    int hdr[LS1_HDR_WORDS] = {0, 0, 0, 0};
    out->batch_rc = ling3_moe_group_batch(
        d_ids, d_wts, d_rc, n, kMax, /*n_used=*/8, n_expert, cap,
        d_slot_tok, d_slot_w, d_tok, d_ge, d_go, d_gc, d_gz, d_sd,
        grp_capacity, slots, hdr);
    std::memcpy(out->hdr, hdr, sizeof(hdr));
    out->slot_tok.resize(slots);
    out->slot_w.resize(slots);
    out->token_slots.resize(slots);
    out->slot_dest.resize(slots);
    out->grp_e.resize(grp_capacity);
    out->grp_off.resize(grp_capacity);
    out->grp_cnt.resize(grp_capacity);
    out->grp_zp.resize(grp_capacity + 1);
    out->counts.assign(n_expert + 1, 0);
    hipMemcpy(out->slot_tok.data(), d_slot_tok, slots * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(out->slot_w.data(), d_slot_w, slots * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(out->token_slots.data(), d_tok, slots * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(out->slot_dest.data(), d_sd, slots * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(out->grp_e.data(), d_ge, grp_capacity * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(out->grp_off.data(), d_go, grp_capacity * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(out->grp_cnt.data(), d_gc, grp_capacity * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(out->grp_zp.data(), d_gz, (grp_capacity + 1) * sizeof(int), hipMemcpyDeviceToHost);
    if (ling3_moe_group_debug_counts(out->counts.data(), n_expert + 1) != LING3_OK) {
        std::fprintf(stderr, "counts copy failed\n");
        a = false;
    }
    hipFree(d_ids); hipFree(d_wts); hipFree(d_rc);
    hipFree(d_slot_tok); hipFree(d_slot_w); hipFree(d_tok);
    hipFree(d_ge); hipFree(d_go); hipFree(d_gc); hipFree(d_gz); hipFree(d_sd);
    return a;
}

int CompareCase(const char * label, int n, int n_expert, int cap, int mode, int poison) {
    const int kMax = 8;
    const int slots = n * kMax;
    int grp_capacity = n_expert + slots / (cap < 8 ? cap : 8) + 3;
    const bool overflow = mode == 3;
    if (overflow) {
        grp_capacity = 2;
        setenv("LING3_MOE_GROUP_SKIP_AUDIT", "1", 1);
    } else {
        setenv("LING3_MOE_GROUP_SKIP_AUDIT", "0", 1);
    }
    std::vector<int> ids(slots), rc(n, 0);
    std::vector<float> wts(slots);
    for (int j = 0; j < slots; ++j) {
        if (mode == 1) {
            ids[j] = 0;                              // one expert
        } else if (mode == 2) {
            ids[j] = j % 8;                          // eight experts, duplicates inside a token
        } else if (mode == 4) {
            // 8 distinct experts per token
            int base = (j / kMax) * 17;
            int s = j % kMax;
            ids[j] = (base + s * 3) % n_expert;
            // repair collisions inside the token
            for (int t = 0; t < s; ++t) {
                if (ids[j] == ids[(j / kMax) * kMax + t]) {
                    ids[j] = (ids[j] + 1) % n_expert;
                    t = -1;
                }
            }
        } else {
            ids[j] = (int)(Next() % (std::uint32_t)n_expert);
        }
        // Distinct bit patterns so a swapped slot fails the float memcmp.
        wts[j] = (float)((j * 17 + 3) % 10007) * 0.001f + 0.125f;
    }
    if (poison == 1 && slots > 0) ids[slots / 2] = n_expert + 3;
    if (poison == 2) rc[n / 2] = 1;

    DevOut a, b;
    bool oa = AllocCopy(ids, wts, rc, n, kMax, n_expert, cap, grp_capacity, 1, &a);
    bool ob = AllocCopy(ids, wts, rc, n, kMax, n_expert, cap, grp_capacity, 0, &b);
    int bad = 0;
    if (!oa || !ob) bad = 1;
    if (a.batch_rc != LING3_OK || b.batch_rc != LING3_OK) {
        std::fprintf(stderr, "FAIL %s batch rc serial=%d parallel=%d\n", label, a.batch_rc, b.batch_rc);
        bad = 1;
    }
    bad |= DiffInts("slot_tok", a.slot_tok, b.slot_tok);
    bad |= DiffFloats("slot_w", a.slot_w, b.slot_w);
    bad |= DiffInts("token_slots", a.token_slots, b.token_slots);
    bad |= DiffInts("slot_dest", a.slot_dest, b.slot_dest);
    bad |= DiffInts("grp_e", a.grp_e, b.grp_e);
    bad |= DiffInts("grp_off", a.grp_off, b.grp_off);
    bad |= DiffInts("grp_cnt", a.grp_cnt, b.grp_cnt);
    bad |= DiffInts("grp_zp", a.grp_zp, b.grp_zp);
    bad |= DiffInts("counts", a.counts, b.counts);
    if (std::memcmp(a.hdr, b.hdr, sizeof(a.hdr)) != 0) {
        std::fprintf(stderr, "DIFF %s hdr serial %d %d %d %d parallel %d %d %d %d\n", label,
                     a.hdr[0], a.hdr[1], a.hdr[2], a.hdr[3],
                     b.hdr[0], b.hdr[1], b.hdr[2], b.hdr[3]);
        bad = 1;
    }
    // Second parallel run must match the first (atomics are sums only).
    DevOut c;
    bool oc = AllocCopy(ids, wts, rc, n, kMax, n_expert, cap, grp_capacity, 0, &c);
    if (!oc || DiffInts("slot_tok rerun", b.slot_tok, c.slot_tok) ||
        DiffFloats("slot_w rerun", b.slot_w, c.slot_w) ||
        DiffInts("token_slots rerun", b.token_slots, c.token_slots) ||
        std::memcmp(b.hdr, c.hdr, sizeof(b.hdr)) != 0) {
        std::fprintf(stderr, "DIFF %s parallel rerun\n", label);
        bad = 1;
    }
    if (bad) {
        std::fprintf(stderr, "FAIL %s\n", label);
        ++failures;
    } else {
        std::printf("  BOTH %-32s n=%4d E=%4d cap=%3d status=%d groups=%d\n",
                    label, n, n_expert, cap, a.hdr[LS1_HDR_STATUS], a.hdr[LS1_HDR_NGROUPS]);
    }
    setenv("LING3_MOE_GROUP_SKIP_AUDIT", "0", 1);
    setenv("LING3_MOE_GROUP_SERIAL", "0", 1);
    return bad;
}

double TimeUs(int serial, int n, int n_expert, int cap, int iters) {
    const int kMax = 8;
    const int slots = n * kMax;
    const int grp_capacity = n_expert + slots / 8 + 3;
    std::vector<int> ids(slots), rc(n, 0);
    std::vector<float> wts(slots);
    for (int j = 0; j < slots; ++j) {
        // Production-like skew: one expert takes most slots, the rest spread.
        ids[j] = ((j % 100) < 90) ? 3 : (int)(Next() % (std::uint32_t)n_expert);
        wts[j] = 0.5f;
    }
    const size_t pad = LING3_TAIL_PAD;
    int *d_ids, *d_rc, *d_st, *d_ts, *d_ge, *d_go, *d_gc, *d_gz, *d_sd;
    float *d_w, *d_sw;
    hipMalloc(&d_ids, (slots + pad) * sizeof(int));
    hipMalloc(&d_w, (slots + pad) * sizeof(float));
    hipMalloc(&d_rc, ((size_t)n + pad) * sizeof(int));
    hipMalloc(&d_st, (slots + pad) * sizeof(int));
    hipMalloc(&d_sw, (slots + pad) * sizeof(float));
    hipMalloc(&d_ts, (slots + pad) * sizeof(int));
    hipMalloc(&d_ge, (grp_capacity + pad) * sizeof(int));
    hipMalloc(&d_go, (grp_capacity + pad) * sizeof(int));
    hipMalloc(&d_gc, (grp_capacity + pad) * sizeof(int));
    hipMalloc(&d_gz, (grp_capacity + 1 + pad) * sizeof(int));
    hipMalloc(&d_sd, (slots + pad) * sizeof(int));
    hipMemcpy(d_ids, ids.data(), slots * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_w, wts.data(), slots * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_rc, rc.data(), (size_t)n * sizeof(int), hipMemcpyHostToDevice);
    setenv("LING3_MOE_GROUP_SERIAL", serial ? "1" : "0", 1);
    setenv("LING3_MOE_GROUP_SKIP_AUDIT", "0", 1);
    int hdr[4];
    for (int i = 0; i < 3; ++i) {
        ling3_moe_group_batch(d_ids, d_w, d_rc, n, kMax, 8, n_expert, cap,
                              d_st, d_sw, d_ts, d_ge, d_go, d_gc, d_gz, d_sd,
                              grp_capacity, slots, hdr);
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        ling3_moe_group_batch(d_ids, d_w, d_rc, n, kMax, 8, n_expert, cap,
                              d_st, d_sw, d_ts, d_ge, d_go, d_gc, d_gz, d_sd,
                              grp_capacity, slots, hdr);
    }
    auto t1 = std::chrono::steady_clock::now();
    hipFree(d_ids); hipFree(d_w); hipFree(d_rc); hipFree(d_st); hipFree(d_sw);
    hipFree(d_ts); hipFree(d_ge); hipFree(d_go); hipFree(d_gc); hipFree(d_gz); hipFree(d_sd);
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::setvbuf(stderr, nullptr, _IOLBF, 0);
  setenv("LING3_MOE_GROUP_SERIAL", "0", 1);
  setenv("LING3_MOE_GROUP_SKIP_AUDIT", "0", 1);
  int n = 8, kmax = 8, nex = 16, cap = 4;
  if (const char * e = getenv("GROUP_N")) n = atoi(e);
  if (const char * e = getenv("GROUP_NEXPERT")) nex = atoi(e);
  if (const char * e = getenv("GROUP_CAP")) cap = atoi(e);
  const int skew = getenv("GROUP_SKEW") ? atoi(getenv("GROUP_SKEW")) : 0;
  std::printf("== moe grouping test: n=%d kMaxExperts=%d n_expert=%d cap=%d skew=%d\n",
              n, kmax, nex, cap, skew);
  RunCase("smallest", n, kmax, nex, cap, skew, -1, 0);
  RunCase("smallest, id out of range", n, kmax, nex, cap, skew, 1, 0);
  RunCase("smallest, router rc set", n, kmax, nex, cap, skew, -1, 1);
  if (n >= 512 && nex >= 512) {
    RunCase("real shape", 512, 8, 512, 64, skew, -1, 0);
    RunCase("real shape, one fat expert", 512, 8, 512, 64, 95, -1, 0);
  }

  std::printf("== LK1 serial vs parallel\n");
  const int ns[] = {1, 2, 5, 64, 511, 512};
  for (int ntok : ns) {
    char buf[80];
    std::snprintf(buf, sizeof(buf), "random n=%d", ntok);
    CompareCase(buf, ntok, 512, 64, 0, 0);
    std::snprintf(buf, sizeof(buf), "one-expert n=%d", ntok);
    CompareCase(buf, ntok, 512, 64, 1, 0);
    std::snprintf(buf, sizeof(buf), "eight n=%d", ntok);
    CompareCase(buf, ntok, 512, 64, 2, 0);
    std::snprintf(buf, sizeof(buf), "distinct n=%d", ntok);
    CompareCase(buf, ntok, 512, 64, 4, 0);
  }
  CompareCase("id out of range", 64, 512, 64, 0, 1);
  CompareCase("router rc", 64, 512, 64, 0, 2);
  CompareCase("group cap overflow", 32, 64, 4, 3, 0);
  CompareCase("two-group fat", 16, 32, 64, 1, 0);
  const double us_s = TimeUs(1, 512, 512, 64, 40);
  const double us_p = TimeUs(0, 512, 512, 64, 40);
  const double us_s1 = TimeUs(1, 1, 512, 64, 40);
  const double us_p1 = TimeUs(0, 1, 512, 64, 40);
  std::printf("TIME n=512 serial %.2f us parallel %.2f us speedup %.2fx\n",
              us_s, us_p, us_p > 0 ? us_s / us_p : 0.0);
  std::printf("TIME n=1 serial %.2f us parallel %.2f us\n", us_s1, us_p1);
  setenv("LING3_MOE_GROUP_SERIAL", "0", 1);

  if (failures) {
    std::printf("FAIL: %d check(s)\n", failures);
    return 1;
  }
  std::printf("PASS: device grouping is bit-identical to the host counting sort\n");
  return 0;
}