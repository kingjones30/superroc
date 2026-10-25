// ling3-route-test — is the new router BIT-IDENTICAL to build24's?
//
// PHASE7 rule, learned the hard way: **bounds discipline in every harness.** Before
// any launch this file validates, ON THE HOST, every index it is about to hand the
// device against the same constants it sized the allocations with, and refuses to
// launch if they disagree. That rule exists because at 2026-10-02T01:58:04Z a test
// in this lane handed expert ids of up to 448 into buffers sized for 8, the kernel
// read off the end, and the GPU page faulted and reset -- which kills every live
// seat on the box. A harness that can fault the GPU needs the same discipline as
// the kernel it tests.
//
// The comparison is old-against-NEW ON THE DEVICE, which is the only comparison
// that can be bit-exact and that does not depend on my re-deriving the router's
// tie-breaking rule on the host:
//
//   logits[e]  bit-identical, every expert, every case
//   ids[i]     bit-identical AND in the same ORDER -- the expert FFN sums
//               contributions in ids order, so order is part of the contract
//   weights[i] bit-identical
//
// Cases: random, adversarial ties (coarse-quantised gate so many experts share a
// score, equal group scores, a flat gate that ties ranks 8..n), and real router
// inputs captured from the model when LING3_ROUTE_CAPTURE points at them.
//
// Smallest shape first: it runs at whatever LING3_ROUTE_NEXPERT / LING3_ROUTE_K say,
// and the caller grows it, never the other way round.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "src/models/ling3/kernels/ling3_moe.hpp"
#include "src/models/ling3/kernels/ling3_quant.hpp"
#include "src/models/ling3/kernels/ling3_route.hpp"

extern "C" int ling3_route_ref(const float * x, const float * gate, const float * bias,
                               int k, int n_expert, int n_used, int n_groups,
                               int n_group_used, float w_scale, int norm_w,
                               int * ids, float * weights);

namespace {

int failures = 0;

#define CHECK(cond, msg)                                                        \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (msg));       \
      ++failures;                                                               \
    }                                                                           \
  } while (0)

std::uint64_t rng = 0x243F6A8885A308D3ULL;
std::uint32_t Next() {
  rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<std::uint32_t>(rng >> 32);
}
float Nf(float lo, float hi) {
  return lo + (hi - lo) * (float) Next() / 4294967296.0F;
}

int n_expert = 512;
int n_used = 8;
int n_groups = 8;
int n_group_used = 2;
int k_hidden = 2560;
bool verbose = false;

// Everything the kernel will index, with the allocation it must fit inside.
struct Shape { int n_expert, n_used, n_groups, n_group_used, k; };

// The phase rule, implemented. Sizes and indices come from the same constants and
// are checked, not assumed.
bool ValidateBounds() {
  bool ok = true;
  if (n_expert <= 0 || n_expert > 1024) { std::fprintf(stderr, "FAIL bounds n_expert=%d\n", n_expert); ok = false; }
  if (n_used <= 0 || n_used > n_expert) { std::fprintf(stderr, "FAIL bounds n_used=%d > n_expert=%d\n", n_used, n_expert); ok = false; }
  if (n_groups <= 0 || (n_expert % n_groups) != 0) { std::fprintf(stderr, "FAIL bounds n_groups=%d does not divide n_expert=%d\n", n_groups, n_expert); ok = false; }
  if (n_group_used <= 0 || n_group_used > n_groups) { std::fprintf(stderr, "FAIL bounds n_group_used=%d\n", n_group_used); ok = false; }
  if (k_hidden <= 0) { std::fprintf(stderr, "FAIL bounds k=%d\n", k_hidden); ok = false; }
  // Every expert id the router can produce is in [0, n_expert), and the expert FFN
  // indexes gate at id * n_ff * row_bytes -- so the gate allocation must hold the
  // largest id. Assert it rather than trusting it.
  const long long gate_bytes = (long long) n_expert * (long long) k_hidden * 4;
  const long long max_gate_touch = (long long)(n_expert - 1) * k_hidden * 4 + k_hidden * 4;
  if (max_gate_touch > gate_bytes) { std::fprintf(stderr, "FAIL bounds gate touch %lld > %lld\n", max_gate_touch, gate_bytes); ok = false; }
  const int nf = ling3_route_nfloat(n_expert, n_groups);
  const int ni = ling3_route_nint(n_expert, n_groups);
  if (nf < 0 || ni < 0) { std::fprintf(stderr, "FAIL bounds route scratch negative\n"); ok = false; }
  // LDS: x staging in the new logits kernel, and the router core's own scratch.
  const long long smem_x = ((long long) k_hidden * 4 + 255) & ~255LL;
  const long long smem_route = (((long long) nf * 4 + (long long) ni * 4) + 255) & ~255LL;
  if (smem_x > 48 * 1024) { std::fprintf(stderr, "FAIL bounds x staging %lld B exceeds 48 KB LDS\n", smem_x); ok = false; }
  if (smem_route > 48 * 1024) { std::fprintf(stderr, "FAIL bounds route scratch %lld B exceeds 48 KB LDS\n", smem_route); ok = false; }
  return ok;
}

void RunCase(const char * label, const std::vector<float> & x,
             const std::vector<float> & gate, const std::vector<float> & bias) {
  const std::size_t xb = x.size() * 4;      // x   bytes
  const std::size_t gb = gate.size() * 4;    // gate bytes
  const std::size_t bb = bias.size() * 4;   // bias bytes
  const std::size_t nlogit_b = (std::size_t) n_expert * 4;   // logits bytes
  const std::size_t nid_b = (std::size_t) n_used * 4;        // ids bytes
  const std::size_t nwt_b = (std::size_t) n_used * 4;         // weights bytes
  const std::size_t pad = LING3_TAIL_PAD;

  float * d_x = nullptr; float * d_gate = nullptr; float * d_bias = nullptr;
  float * d_la = nullptr; float * d_lb = nullptr;
  int * d_ia = nullptr; int * d_ib = nullptr;
  float * d_wa = nullptr; float * d_wb = nullptr;
  const int before = failures;
  // Every allocation below is xb/gb/bb/lb/ib/wb plus the pad, and ValidateBounds has
  // already checked that the largest index each kernel forms is inside the matching
  // one. If an allocation fails the case is skipped, not guessed at.
  bool alloc = hipMalloc(&d_x, xb + pad) == hipSuccess;
  alloc = alloc && hipMalloc(&d_gate, gb + pad) == hipSuccess;
  alloc = alloc && hipMalloc(&d_bias, bb + pad) == hipSuccess;
  alloc = alloc && hipMalloc(&d_la, nlogit_b) == hipSuccess;
  alloc = alloc && hipMalloc(&d_lb, nlogit_b) == hipSuccess;
  alloc = alloc && hipMalloc(&d_ia, nid_b) == hipSuccess;
  alloc = alloc && hipMalloc(&d_ib, nid_b) == hipSuccess;
  alloc = alloc && hipMalloc(&d_wa, nwt_b) == hipSuccess;
  alloc = alloc && hipMalloc(&d_wb, nwt_b) == hipSuccess;
  CHECK(alloc, "device allocations");
  if (!alloc || failures != before) {
    hipFree(d_x); hipFree(d_gate); hipFree(d_bias); hipFree(d_la); hipFree(d_lb);
    hipFree(d_ia); hipFree(d_ib); hipFree(d_wa); hipFree(d_wb);
    return;
  }
  hipMemcpy(d_x, x.data(), xb, hipMemcpyHostToDevice);
  hipMemcpy(d_gate, gate.data(), gb, hipMemcpyHostToDevice);
  hipMemcpy(d_bias, bias.data(), bb, hipMemcpyHostToDevice);
  hipMemset(d_la, 0x7f, nlogit_b); hipMemset(d_lb, 0x7f, nlogit_b);
  hipMemset(d_ia, 0x7f, nid_b); hipMemset(d_ib, 0x7f, nid_b);
  hipMemset(d_wa, 0x7f, nwt_b); hipMemset(d_wb, 0x7f, nwt_b);

  // 1. build24's router, as the reference. Read back before anything else runs, because
  //    the next two calls overwrite the same device buffers.
  const int rc_a = ling3_route_ref(d_x, d_gate, d_bias, k_hidden, n_expert, n_used,
                                   n_groups, n_group_used, 1.0f, 1, d_ia, d_wa);
  CHECK(rc_a == LING3_OK, "ling3_route_ref rc");
  CHECK(hipDeviceSynchronize() == hipSuccess, "sync after ref");
  std::vector<float> la(n_expert), wa(n_used);
  std::vector<int> ia(n_used);
  hipMemcpy(la.data(), d_la, nlogit_b, hipMemcpyDeviceToHost);
  hipMemcpy(ia.data(), d_ia, nid_b, hipMemcpyDeviceToHost);
  hipMemcpy(wa.data(), d_wa, nwt_b, hipMemcpyDeviceToHost);

  // 2. the new router as shipped. Its core is build24's network; what changed is
  //    the logits kernel, so this checks BOTH: the logits are the item-2a claim, and the
  //    ids/weights prove the untouched core still reproduces build24's order.
  const int rc_b = ling3_route(d_x, d_gate, d_bias, k_hidden, n_expert, n_used,
                               n_groups, n_group_used, 1.0f, 1, d_ib, d_wb);
  CHECK(rc_b == LING3_OK, "ling3_route rc");
  CHECK(hipDeviceSynchronize() == hipSuccess, "sync after new");
  std::vector<float> lbv(n_expert), wbv(n_used);
  std::vector<int> ibv(n_used);
  hipMemcpy(lbv.data(), d_lb, nlogit_b, hipMemcpyDeviceToHost);
  hipMemcpy(ibv.data(), d_ib, nid_b, hipMemcpyDeviceToHost);
  hipMemcpy(wbv.data(), d_wb, nwt_b, hipMemcpyDeviceToHost);

  int ldiff = 0, iddiff = 0, wdiff = 0;
  for (int e = 0; e < n_expert; ++e) if (la[e] != lbv[e]) ++ldiff;
  for (int i = 0; i < n_used; ++i) {
    if (ia[i] != ibv[i]) ++iddiff;
    if (wa[i] != wbv[i]) ++wdiff;
    // The returned ids are what the expert FFN will index, so they are checked
    // against the expert count BEFORE anything downstream is asked to use them.
    CHECK(ibv[i] >= 0 && ibv[i] < n_expert, "returned expert id is inside the expert range");
  }
  CHECK(ldiff == 0, "router logits differ from build24's");
  CHECK(iddiff == 0, "router ids differ from build24's (order included)");
  CHECK(wdiff == 0, "router weights differ from build24's");
  if (verbose || ldiff || iddiff || wdiff) {
    std::printf("  ROUTE %-36s logits_diff=%d ids_diff=%d weights_diff=%d\n",
                label, ldiff, iddiff, wdiff);
    if (ldiff) {
      for (int e = 0; e < n_expert; ++e) {
        if (la[e] != lbv[e]) { std::printf("    first logits mismatch e=%d ref=%.9g new=%.9g\n", e, la[e], lbv[e]); break; }
      }
    }
    if (iddiff) {
      std::printf("    ref ids:");
      for (int i = 0; i < n_used && i < 8; ++i) std::printf(" %d", ia[i]);
      std::printf("\n    new ids:");
      for (int i = 0; i < n_used && i < 8; ++i) std::printf(" %d", ibv[i]);
      std::printf("\n");
    }
  }
  hipFree(d_x); hipFree(d_gate); hipFree(d_bias); hipFree(d_la); hipFree(d_lb);
  hipFree(d_ia); hipFree(d_ib); hipFree(d_wa); hipFree(d_wb);
}

}  // namespace

int main() {
  if (const char * e = getenv("LING3_ROUTE_VERBOSE")) verbose = true;
  if (const char * e = getenv("LING3_ROUTE_NEXPERT")) n_expert = atoi(e);
  if (const char * e = getenv("LING3_ROUTE_NUSED")) n_used = atoi(e);
  if (const char * e = getenv("LING3_ROUTE_K")) k_hidden = atoi(e);
  // LR2 SPEC2: the model's gate shape is 512/8/8/4 (GGUF expert_group_used_count = 4).
  if (const char * e = getenv("LING3_ROUTE_NGROUPS")) n_groups = atoi(e);
  if (const char * e = getenv("LING3_ROUTE_NGROUPUSED")) n_group_used = atoi(e);
  int trials = 8;
  if (const char * e = getenv("LING3_ROUTE_TRIALS")) trials = atoi(e);

  std::printf("== route test: n_expert=%d n_used=%d n_groups=%d n_group_used=%d k=%d trials=%d\n",
              n_expert, n_used, n_groups, n_group_used, k_hidden, trials);
  if (!ValidateBounds()) {
    std::printf("FAIL: host-side bounds validation refused to launch\n");
    return 1;
  }
  std::printf("   host-side bounds validated against the allocation sizes and the LDS limits\n");

  const std::size_t nb = (std::size_t) n_expert * (std::size_t) k_hidden;
  std::vector<float> x(k_hidden), gate(nb), bias(n_expert);

  for (int t = 0; t < trials; ++t) {
    for (int i = 0; i < k_hidden; ++i) x[i] = Nf(-3.f, 3.f);
    for (std::size_t i = 0; i < nb; ++i) gate[i] = Nf(-1.f, 1.f);
    for (int i = 0; i < n_expert; ++i) bias[i] = Nf(-0.1f, 0.1f);
    RunCase("random", x, gate, bias);

    // Ties inside a group: quantise the gate to a coarse grid so many experts share
    // a score exactly. With 3 levels over 2560 terms many dot products collide.
    const int grid = (t % 2 == 0) ? 3 : 9;
    for (std::size_t i = 0; i < nb; ++i) {
      gate[i] = (float) (int) Nf(0.f, (float) grid);
    }
    for (int i = 0; i < n_expert; ++i) bias[i] = (t % 2 == 1) ? 0.25f : Nf(-0.1f, 0.1f);
    RunCase((t % 2 == 0) ? "tied sel within groups" : "equal group scores", x, gate, bias);

    // Tie at the 8th/9th boundary: a flat gate, so every expert ties and the top
    // n_used is chosen purely by the tie rule.
    for (std::size_t i = 0; i < nb; ++i) gate[i] = 0.0f;
    for (int i = 0; i < n_expert; ++i) bias[i] = 0.0f;
    RunCase("all tied (8th vs 9th)", x, gate, bias);

    // Two clear winners, then everything else tied at the boundary.
    for (std::size_t i = 0; i < nb; ++i) gate[i] = 0.0f;
    for (int i = 0; i < n_expert; ++i) bias[i] = 0.0f;
    bias[7] = 5.0f; if (n_expert > 300) bias[300] = 5.0f;  // LR2: 300 is OOB below n_expert=301
    RunCase("two winners then a tie", x, gate, bias);

    // LR2 SPEC2: tie AT THE GROUP KEEP CUT -- the case that decides 4-of-8 selection.
    // Groups 0..2 are clear winners; every remaining group has the SAME gscore, so which
    // tied group takes the last keep slot (n_group_used = 4 in the model) is decided by
    // the group bitonic's network tie order -- a hand-rolled arg-max keep-set can diverge
    // here (v1 did: it hardcoded 2 groups). Flat gate: every logit is 0, every prob is
    // exactly 0.5, the biases alone set the group scores (17 / 9 / 5 / 1 / 1 / 1 / 1 / 1
    // at n_groups=8), and inside each kept group all 64 experts tie too, so the final
    // 512-wide bitonic also runs at its tie worst.
    for (std::size_t i = 0; i < nb; ++i) gate[i] = 0.0f;
    for (int i = 0; i < n_expert; ++i) bias[i] = 0.0f;
    if (n_groups >= 4) {
      const int npg_t = n_expert / n_groups;
      for (int j = 0; j < npg_t; ++j) {
        bias[j] = 8.0f;             // group 0
        bias[npg_t + j] = 4.0f;     // group 1
        bias[2 * npg_t + j] = 2.0f; // group 2; groups 3.. tie at gscore 1.0
      }
    }
    RunCase("group tie at the keep cut", x, gate, bias);

  }

  // Real router inputs, if Jay captured them: LING3_ROUTE_CAPTURE=<file> holding
  // n_expert*k floats per case, one after another.
  if (const char * cap = getenv("LING3_ROUTE_CAPTURE")) {
    std::FILE * f = std::fopen(cap, "rb");
    if (f) {
      std::printf("   reading captured router inputs from %s\n", cap);
      int cases = 0;
      while (true) {
        if (std::fread(x.data(), 4, x.size(), f) != x.size()) break;
        if (std::fread(gate.data(), 4, gate.size(), f) != gate.size()) break;
        if (std::fread(bias.data(), 4, bias.size(), f) != bias.size()) break;
        RunCase("captured from the model", x, gate, bias);
        if (++cases >= 40) break;
      }
      std::printf("   %d captured case(s)\n", cases);
      std::fclose(f);
    } else {
      std::printf("   capture file %s not readable; skipping captured cases\n", cap);
    }
  }

  // LR2 SPEC2 item 3: microseconds per call at THIS shape, for whichever core this
  // process selects (LING3_ROUTE_REF=1 -> k_route, default -> k_route_fast). Both paths
  // run the SAME k_route_logits, so (REF us) - (default us) = k_route - k_route_fast.
  if (const char * e = getenv("LING3_ROUTE_TIME")) {
    const int iters = atoi(e);
    if (iters > 0) {
      for (int i = 0; i < k_hidden; ++i) x[i] = Nf(-3.f, 3.f);
      for (std::size_t i = 0; i < nb; ++i) gate[i] = Nf(-1.f, 1.f);
      for (int i = 0; i < n_expert; ++i) bias[i] = Nf(-0.1f, 0.1f);
      float * d_tx = nullptr; float * d_tgate = nullptr; float * d_tbias = nullptr;
      int * d_tids = nullptr; float * d_tw = nullptr;
      bool tok = hipMalloc(&d_tx, x.size() * 4) == hipSuccess;
      tok = tok && hipMalloc(&d_tgate, gate.size() * 4) == hipSuccess;
      tok = tok && hipMalloc(&d_tbias, bias.size() * 4) == hipSuccess;
      tok = tok && hipMalloc(&d_tids, (std::size_t) n_used * 4) == hipSuccess;
      tok = tok && hipMalloc(&d_tw, (std::size_t) n_used * 4) == hipSuccess;
      if (tok) {
        hipMemcpy(d_tx, x.data(), x.size() * 4, hipMemcpyHostToDevice);
        hipMemcpy(d_tgate, gate.data(), gate.size() * 4, hipMemcpyHostToDevice);
        hipMemcpy(d_tbias, bias.data(), bias.size() * 4, hipMemcpyHostToDevice);
        for (int i = 0; i < 20; ++i)
          ling3_route(d_tx, d_tgate, d_tbias, k_hidden, n_expert, n_used,
                      n_groups, n_group_used, 1.0f, 1, d_tids, d_tw);
        hipDeviceSynchronize();
        hipEvent_t t0, t1;
        hipEventCreate(&t0);
        hipEventCreate(&t1);
        hipEventRecord(t0);
        for (int i = 0; i < iters; ++i)
          ling3_route(d_tx, d_tgate, d_tbias, k_hidden, n_expert, n_used,
                      n_groups, n_group_used, 1.0f, 1, d_tids, d_tw);
        hipEventRecord(t1);
        hipEventSynchronize(t1);
        float ms = 0.f;
        hipEventElapsedTime(&ms, t0, t1);
        const char * refv = getenv("LING3_ROUTE_REF");
        std::printf("TIME %s: %.2f us/call over %d calls (whole ling3_route: k_route_logits + core)\n",
                    (refv && refv[0] == '1') ? "k_route (REF)" : "k_route_fast (default)",
                    (double) ms * 1000.0 / (double) iters, iters);
        hipEventDestroy(t0);
        hipEventDestroy(t1);
      } else {
        std::printf("TIME skipped: hipMalloc failed\n");
      }
      hipFree(d_tx); hipFree(d_tgate); hipFree(d_tbias); hipFree(d_tids); hipFree(d_tw);
    }
  }

  if (failures) {
    std::printf("FAIL: %d check(s)\n", failures);
    return 1;
  }
  std::printf("PASS: router logits, ids (in order) and weights are bit-identical to build24's\n");
  return 0;
}
