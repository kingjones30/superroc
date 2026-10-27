// Host test for the SPEC8 depth rule. No HIP. Built and run on amdaimax2
// through gpu-run (see ~/ls/lm1/h-build.sh).
#include "depth_rule.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace dr = superroc::models::ling3::mtp::depth;

static int g_fails = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_fails;                                                           \
    }                                                                      \
  } while (0)

static void AddPlain(dr::Bucket& d, double pt, int n) {
  for (int i = 0; i < n; ++i)
    dr::RecordPlain(d, pt);
}

static void AddFull(dr::Bucket& d, double pt, int n) {
  // k=1 full accept emits 2 tokens. pt is ms per token.
  for (int i = 0; i < n; ++i)
    dr::RecordBlock(d, pt * 2.0, 2.0, true);
}

static void AddReject(dr::Bucket& d, double pt, int n) {
  for (int i = 0; i < n; ++i)
    dr::RecordBlock(d, pt, 1.0, false);
}

static void Finish(dr::Bucket& d) {
  dr::EndRequest(d, /*global_plain=*/0.0);
}

// The production lock-out: one low-acceptance reply's observed mix must not
// deny the next request when full-accepts are under a plain step.
static void TestContentDoesNotStick() {
  dr::Bucket d;
  dr::BeginRequest(d);
  AddPlain(d, 22.7, 8);
  AddFull(d, 19.6, 5);    // structural case pays
  AddReject(d, 40.0, 8);  // observed mix does not
  CHECK(dr::SpecPt(d) > 22.7);
  Finish(d);
  CHECK(!d.deny_a);
  dr::BeginRequest(d);
  CHECK(dr::Allows(d));
  Finish(d);
}

// oldenv: the first reply's full-accepts are cold (27 ms vs a later 19).
// They must not deny the next request, and the warm samples must drop them
// out of the band.
static void TestColdDoesNotDeny() {
  dr::Bucket d;
  dr::BeginRequest(d);
  AddPlain(d, 24.0, 4);
  AddFull(d, 27.3, 5);
  Finish(d);
  CHECK(d.warm_requests == 1);
  CHECK(!d.deny_a);
  dr::BeginRequest(d);
  CHECK(dr::Allows(d));
  AddPlain(d, 23.0, 2);
  AddFull(d, 19.6, 6);
  Finish(d);
  long wn = 0;
  const double best = dr::BestPt(d, &wn);
  CHECK(wn >= 4);
  CHECK(best < 21.0);
  CHECK(best <= 23.0);
  CHECK(!d.deny_a);
  dr::BeginRequest(d);
  CHECK(dr::Allows(d));
}

// Two requests of warm full-accepts that lose to plain: deny, skip the
// backoff, then a cheap probe lifts it. A skipped request must not refresh
// the backoff (that was the "never re-opens" bug in miniature).
static void TestStructuralDenyReprobes() {
  dr::Bucket d;
  dr::BeginRequest(d);
  AddPlain(d, 24.0, 4);
  AddFull(d, 30.0, 4);
  Finish(d);
  CHECK(!d.deny_a);

  dr::BeginRequest(d);
  AddPlain(d, 24.0, 2);
  AddFull(d, 30.0, 4);
  Finish(d);
  CHECK(d.deny_a);
  CHECK(d.skip_left == dr::kBackoffStart);

  const int backoff = d.skip_left;
  for (int i = 0; i < backoff; ++i) {
    dr::BeginRequest(d);
    CHECK(!dr::Allows(d));
    Finish(d);
    CHECK(d.deny_a);
  }
  dr::BeginRequest(d);
  CHECK(d.probe_open);
  CHECK(dr::Allows(d));
  AddPlain(d, 24.0, 2);
  AddFull(d, 18.0, 6);
  Finish(d);
  CHECK(!d.deny_a);
  dr::BeginRequest(d);
  CHECK(dr::Allows(d));
}

// A probe that is still expensive backs off further instead of probing every
// following request.
static void TestFailedProbeBacksOff() {
  dr::Bucket d;
  dr::BeginRequest(d);
  AddPlain(d, 24.0, 2);
  AddFull(d, 30.0, 4);
  Finish(d);
  dr::BeginRequest(d);
  AddFull(d, 30.0, 4);
  Finish(d);
  CHECK(d.deny_a);
  const int first = d.backoff;
  for (int i = 0; i < first; ++i) {
    dr::BeginRequest(d);
    Finish(d);
  }
  dr::BeginRequest(d);
  CHECK(d.probe_open);
  AddFull(d, 30.0, 4);
  Finish(d);
  CHECK(d.deny_a);
  CHECK(d.backoff == std::min(first * 2, dr::kBackoffCap));
  CHECK(d.skip_left == d.backoff);
}

static void TestShortDraftFloor() {
  dr::Bucket d;
  CHECK(!dr::DraftShort(d, 32, 0.0));
  CHECK(!dr::DraftShort(d, 128, 0.0));
  CHECK(dr::DraftShort(d, 144, 0.0));
  CHECK(dr::DraftShort(d, 160, 0.0));
  // floor <= 0 is the G3 arm that disables kUnprovenMinTokens.
  CHECK(dr::DraftShort(d, 32, 0.0, /*floor=*/0));
  // Once the request has opened, a shrinking remainder must not cut it off.
  dr::BeginRequest(d);
  CHECK(dr::DraftShort(d, 32, 0.0));
  // A proven bucket may draft a short answer.
  dr::Bucket proven;
  dr::BeginRequest(proven);
  AddPlain(proven, 22.0, 4);
  AddFull(proven, 19.0, 4);
  Finish(proven);
  CHECK(dr::Pays(proven, 0.0));
  CHECK(dr::DraftShort(proven, 32, 0.0));
}

// Jay 13:21: a low-acceptance prose request, then a code request in the same
// bucket. The code request drafts on its first eligible step. The observed
// mix does not stick, and one prose request is not a structural deny.
static void TestProseThenCode() {
  dr::Bucket d;
  dr::BeginRequest(d);
  AddPlain(d, 22.66, 8);
  AddFull(d, 19.58, 5);
  AddReject(d, 48.0, 6);
  CHECK(dr::SpecPt(d) > 22.66);
  Finish(d);
  CHECK(!d.deny_a);
  CHECK(d.skip_left == 0);
  CHECK(!d.probe_open);

  dr::BeginRequest(d);
  CHECK(dr::Allows(d));
  CHECK(!d.probe_open);
  CHECK(dr::DraftShort(d, 160, 22.66));
  AddFull(d, 19.5, 1);
  CHECK(d.req_spec_n == 1);
  CHECK(d.req_acc_n == 1);
  CHECK(dr::Allows(d));
  Finish(d);
  CHECK(!d.deny_a);
  dr::BeginRequest(d);
  CHECK(dr::Allows(d));
}

// 11K full-accepts at 22.9 ms vs a 23.35 ms plain step do not clear the
// margin, so the bucket stays shut. A 19 ms full accept against 23 ms does.
static void TestDeepMargin() {
  dr::Bucket cold;
  CHECK(!dr::DeepOpen(cold, 3, 23.0));
  CHECK(!dr::DeepOpen(cold, 5, 23.0));
  CHECK(dr::DeepOpen(cold, 0, 23.0));
  CHECK(dr::DeepOpen(cold, 2, 23.0));

  dr::Bucket thin;
  dr::BeginRequest(thin);
  AddPlain(thin, 23.35, 4);
  AddFull(thin, 22.9, 4);
  Finish(thin);
  CHECK(!dr::DeepOpen(thin, 3, 0.0));

  dr::Bucket wide;
  dr::BeginRequest(wide);
  AddPlain(wide, 23.0, 4);
  AddFull(wide, 19.0, 4);
  Finish(wide);
  CHECK(dr::DeepOpen(wide, 3, 0.0));
}

int main() {
  TestContentDoesNotStick();
  TestColdDoesNotDeny();
  TestStructuralDenyReprobes();
  TestFailedProbeBacksOff();
  TestShortDraftFloor();
  TestProseThenCode();
  TestDeepMargin();
  if (g_fails != 0) {
    std::fprintf(stderr, "%d checks failed\n", g_fails);
    return 1;
  }
  std::printf("depth_rule_test ok\n");
  return 0;
}
