// SPEC8 depth rule. Header-only so a host test can run the state machine
// without HIP. mtp_gpu.cpp is the only production caller.
//
// Two decisions, and they are not the same kind of fact:
//
// (a) structural. A full-accept block is the best a depth can do. If warm
//     full-accepts still cost more per token than a plain step, drafting
//     cannot pay there. That deny is allowed to stick, but only after two
//     requests agree, and only on samples near the cheapest full-accept seen
//     (cold blocks sit above that band and do not vote). A stuck deny re-opens
//     on a backoff counted in requests, and the probe's own samples can lift
//     it.
//
// (b) observed mix. Acceptance depends on the text. It stops the rest of THIS
//     request via the per-request cost ring (reset in ResetSpecState) and is
//     not stored here. A low-acceptance reply must not deny the next request.
//
// LING3_MTP_DEPTH unset uses this rule. LING3_MTP_DEPTH=0 skips it.

#ifndef GUFO_MODELS_LING3_DEPTH_RULE_HPP_
#define GUFO_MODELS_LING3_DEPTH_RULE_HPP_

#include <algorithm>

namespace gufo::models::ling3::mtp::depth {

constexpr int kBuckets = 6;
constexpr int kAccRing = 16;
constexpr int kSpecRing = 16;
// A full-accept counts toward the structural price only when it is within this
// factor of the cheapest full-accept in the ring. The first blocks after load
// are several times a warm block; once a warm one lands they fall out of the
// band and can no longer deny the bucket.
constexpr double kWarmBand = 1.20;
constexpr long kWarmMinSamples = 4;
constexpr int kConfirmRequests = 2;
constexpr int kBackoffStart = 2;
constexpr int kBackoffCap = 16;
// An unproven depth does not open on a short answer. The n=32 bench item lost
// 6% to plain decode because the probe was the whole answer; a 128-token agent
// turn has the same shape. Seat replies are 160 tokens, which clears this, so
// those still draft. A depth that has already shown a warm full-accept under a
// plain step (Pays) may draft a short answer too.
constexpr int kUnprovenMinTokens = 144;
// 8K and deeper stays closed until warm full-accepts beat a plain step by
// this much. At 11K a full accept is only ~2% under plain, and the client
// loses as soon as one draft is rejected (G3). The margin is the host work
// around the block, which the block timer does not see. Bucket 0 stays open.
constexpr double kDeepMargin = 0.95;

inline int BucketOf(int pos) {
  if (pos < 2048)
    return 0;
  if (pos < 4096)
    return 1;
  if (pos < 8192)
    return 2;
  if (pos < 16384)
    return 3;
  if (pos < 32768)
    return 4;
  return 5;
}

struct Bucket {
  double plain_ms{0.0};
  long plain_n{0};

  double acc_ms[kAccRing]{};
  double acc_tok[kAccRing]{};
  int acc_n{0};
  int acc_i{0};

  double spec_ms[kSpecRing]{};
  double spec_tok[kSpecRing]{};
  int spec_n{0};
  int spec_i{0};

  double req_acc_ms{0.0};
  double req_acc_tok{0.0};
  int req_acc_n{0};
  double req_spec_ms{0.0};
  double req_spec_tok{0.0};
  long req_spec_n{0};
  double req_min_pt{0.0};

  int warm_requests{0};
  int requests{0};
  bool deny_a{false};
  int skip_left{0};
  int backoff{kBackoffStart};
  bool probe_open{false};
  bool last_probe{false};  // this request was opened as a structural re-probe
  bool started{false};
};

inline void BeginRequest(Bucket& d) {
  if (d.started)
    return;
  d.started = true;
  ++d.requests;
  d.req_acc_ms = 0.0;
  d.req_acc_tok = 0.0;
  d.req_acc_n = 0;
  d.req_spec_ms = 0.0;
  d.req_spec_tok = 0.0;
  d.req_spec_n = 0;
  d.req_min_pt = 0.0;
  d.probe_open = false;
  d.last_probe = false;
  if (d.deny_a) {
    if (d.skip_left > 0) {
      --d.skip_left;
    } else {
      d.probe_open = true;
      d.last_probe = true;
    }
  }
}

inline void RecordPlain(Bucket& d, double ms) {
  if (ms <= 0.0)
    return;
  d.plain_ms += ms;
  ++d.plain_n;
}

inline void RecordBlock(Bucket& d, double ms, double tok, bool full_accept) {
  if (tok <= 0.0)
    return;
  if (d.spec_n < kSpecRing) {
    d.spec_ms[d.spec_n] = ms;
    d.spec_tok[d.spec_n] = tok;
    ++d.spec_n;
  } else {
    d.spec_ms[d.spec_i] = ms;
    d.spec_tok[d.spec_i] = tok;
    d.spec_i = (d.spec_i + 1) % kSpecRing;
  }
  d.req_spec_ms += ms;
  d.req_spec_tok += tok;
  ++d.req_spec_n;
  if (!full_accept)
    return;
  if (d.acc_n < kAccRing) {
    d.acc_ms[d.acc_n] = ms;
    d.acc_tok[d.acc_n] = tok;
    ++d.acc_n;
  } else {
    d.acc_ms[d.acc_i] = ms;
    d.acc_tok[d.acc_i] = tok;
    d.acc_i = (d.acc_i + 1) % kAccRing;
  }
  d.req_acc_ms += ms;
  d.req_acc_tok += tok;
  ++d.req_acc_n;
  const double pt = ms / tok;
  if (d.req_min_pt <= 0.0 || pt < d.req_min_pt)
    d.req_min_pt = pt;
}

inline double RingMinPt(const Bucket& d) {
  double m = 0.0;
  const int n = d.acc_n < kAccRing ? d.acc_n : kAccRing;
  for (int i = 0; i < n; ++i) {
    if (d.acc_tok[i] <= 0.0)
      continue;
    const double pt = d.acc_ms[i] / d.acc_tok[i];
    if (m <= 0.0 || pt < m)
      m = pt;
  }
  return m;
}

inline void WarmTotals(const Bucket& d, double* ms, double* tok, long* n) {
  *ms = 0.0;
  *tok = 0.0;
  *n = 0;
  const double min_pt = RingMinPt(d);
  if (min_pt <= 0.0)
    return;
  const int cnt = d.acc_n < kAccRing ? d.acc_n : kAccRing;
  for (int i = 0; i < cnt; ++i) {
    if (d.acc_tok[i] <= 0.0)
      continue;
    const double pt = d.acc_ms[i] / d.acc_tok[i];
    if (pt <= min_pt * kWarmBand) {
      *ms += d.acc_ms[i];
      *tok += d.acc_tok[i];
      ++*n;
    }
  }
}

inline double SpecPt(const Bucket& d) {
  double ms = 0.0;
  double tok = 0.0;
  const int cnt = d.spec_n < kSpecRing ? d.spec_n : kSpecRing;
  for (int i = 0; i < cnt; ++i) {
    ms += d.spec_ms[i];
    tok += d.spec_tok[i];
  }
  return tok > 0.0 ? ms / tok : 0.0;
}

inline double BestPt(const Bucket& d, long* warm_n) {
  double ms = 0.0;
  double tok = 0.0;
  long n = 0;
  WarmTotals(d, &ms, &tok, &n);
  if (warm_n != nullptr)
    *warm_n = n;
  return tok > 0.0 ? ms / tok : 0.0;
}

// Content never sticks. A structural deny sticks until a probe lifts it, and
// the probe request itself is allowed through.
inline bool Allows(const Bucket& d) {
  if (!d.deny_a)
    return true;
  return d.probe_open;
}

// Buckets 0-2 are measured on the request. Buckets 3-5 (8K and up) draft only
// after warm full-accepts clear kDeepMargin. An empty deep bucket stays shut,
// so the first short answer at 11K is a plain step.
inline bool DeepOpen(const Bucket& d, int bucket, double global_plain) {
  if (bucket < 3)
    return true;
  long wn = 0;
  const double best = BestPt(d, &wn);
  if (wn < kWarmMinSamples || d.warm_requests < 1 || best <= 0.0)
    return false;
  const double local =
      d.plain_n > 0 ? d.plain_ms / static_cast<double>(d.plain_n) : 0.0;
  const double plain = local > 0.0 ? local : global_plain;
  return plain > 0.0 && best <= plain * kDeepMargin;
}

inline void EndRequest(Bucket& d, double global_plain) {
  if (!d.started)
    return;
  const double min_pt = RingMinPt(d);
  const bool ran = d.req_acc_n > 0 || d.req_spec_n > 0;
  if (ran && d.req_min_pt > 0.0 && min_pt > 0.0 &&
      d.req_min_pt <= min_pt * kWarmBand) {
    ++d.warm_requests;
  }
  long wn = 0;
  const double best = BestPt(d, &wn);
  const double local =
      d.plain_n > 0 ? d.plain_ms / static_cast<double>(d.plain_n) : 0.0;
  const double plain = local > 0.0 ? local : global_plain;
  const bool confirmed_pay = ran && wn >= kWarmMinSamples &&
                             d.warm_requests >= 1 && best > 0.0 &&
                             plain > 0.0 && best <= plain;
  const bool confirmed_loss = ran && wn >= kWarmMinSamples &&
                              d.warm_requests >= kConfirmRequests &&
                              best > 0.0 && plain > 0.0 && best > plain;
  if (confirmed_pay) {
    d.deny_a = false;
    d.backoff = kBackoffStart;
    d.skip_left = 0;
  } else if (confirmed_loss || (ran && d.probe_open)) {
    // A failed probe backs off harder. A denied request that did not run
    // (ran == false) must not refresh skip_left, or the backoff never elapses.
    d.deny_a = true;
    if (d.probe_open)
      d.backoff = std::min(d.backoff * 2, kBackoffCap);
    d.skip_left = d.backoff;
  }
  d.probe_open = false;
  d.started = false;
}

inline bool Pays(const Bucket& d, double global_plain) {
  if (d.deny_a && !d.probe_open)
    return false;
  long wn = 0;
  const double best = BestPt(d, &wn);
  if (wn < kWarmMinSamples || d.warm_requests < 1 || best <= 0.0)
    return false;
  const double local =
      d.plain_n > 0 ? d.plain_ms / static_cast<double>(d.plain_n) : 0.0;
  const double plain = local > 0.0 ? local : global_plain;
  return plain > 0.0 && best <= plain;
}

// remaining >= floor drafts immediately. The default floor is
// kUnprovenMinTokens (144). floor <= 0 disables it: real clients send no
// max_tokens or 2048+, so a floor keyed on the budget does not see a short
// natural answer, and G3 measures that arm with the floor off.
// A request that already opened the bucket keeps drafting after the remainder
// drops below the floor (the runner passes the tokens still left).
inline bool DraftShort(const Bucket& d, int remaining, double global_plain,
                       int floor = kUnprovenMinTokens) {
  if (floor <= 0 || remaining >= floor)
    return true;
  if (d.started)
    return true;
  return Pays(d, global_plain);
}

}  // namespace gufo::models::ling3::mtp::depth

#endif  // GUFO_MODELS_LING3_DEPTH_RULE_HPP_
