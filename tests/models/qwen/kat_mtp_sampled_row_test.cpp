#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/models/qwen/hip/mtp/mtp_sampled_row.hpp"

namespace {

int fails = 0;

void Expect(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "FAIL " << what << "\n";
    ++fails;
  }
}

bool Near(double got, double want, double tol) {
  return std::fabs(got - want) <= tol;
}

// 24 rows: eight spread scores, a flat tail far below them and two -inf rows
// that must never reach the reported row.
std::vector<float> FixedLogits() {
  std::vector<float> logits(24, -50.0F);
  logits[4] = 6.0F;
  logits[17] = 4.0F;
  logits[0] = 2.5F;
  logits[9] = 1.0F;
  logits[12] = 0.0F;
  logits[21] = -1.5F;
  logits[7] = -3.0F;
  logits[14] = -5.0F;
  logits[3] = -std::numeric_limits<float>::infinity();
  logits[19] = -std::numeric_limits<float>::infinity();
  return logits;
}

}  // namespace

int main() {
  using superroc::hip::DrawSampledRow;
  using superroc::tokenization::TokenId;

  const std::vector<float> logits = FixedLogits();
  const std::size_t k = 8;
  const float temperature = 0.7F;
  std::uint64_t rng = 0x243F6A8885A308D3ULL;

  std::vector<TokenId> ids;
  std::vector<float> probs;
  (void)DrawSampledRow(std::span<const float>(logits), temperature, k, &rng,
                       ids, probs);

  double total = 0.0;
  for (const float probability : probs) {
    total += probability;
  }
  Expect(Near(total, 1.0, 1e-5), "probabilities sum to 1");
  Expect(ids.size() == k && probs.size() == k, "row width is k");

  bool distinct = true;
  bool finite_only = true;
  bool ordered = true;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    for (std::size_t j = i + 1; j < ids.size(); ++j) {
      if (ids[i] == ids[j]) {
        distinct = false;
      }
    }
    if (!std::isfinite(logits[ids[i]])) {
      finite_only = false;
    }
    if (i > 0 && !(logits[ids[i - 1]] > logits[ids[i]])) {
      ordered = false;
    }
  }
  Expect(distinct, "ids are distinct");
  Expect(finite_only, "ids are finite-logit rows only");
  Expect(ordered, "ids are ordered by logit");

  // probs[i] / probs[j] is exp((logit_i - logit_j) / T) for any two rows of
  // the same renormalized top-K set, so the top pair alone checks it.
  const double gap = static_cast<double>(logits[ids[0]] - logits[ids[1]]);
  Expect(Near(static_cast<double>(probs[0]) / probs[1],
              std::exp(gap / static_cast<double>(temperature)), 1e-3),
         "top-two probability ratio is exp(gap/T)");

  // 200000 draws must track the reported row: the row describes the
  // distribution actually sampled. probs is in descending logit order, so
  // key the reported value by token id before comparing.
  std::vector<double> reported(logits.size(), 0.0);
  for (std::size_t index = 0; index < ids.size(); ++index) {
    reported[static_cast<std::size_t>(ids[index])] = probs[index];
  }
  std::vector<std::size_t> counts(logits.size(), 0);
  const int draws = 200000;
  for (int index = 0; index < draws; ++index) {
    ids.clear();
    probs.clear();
    const TokenId token = DrawSampledRow(std::span<const float>(logits),
                                         temperature, k, &rng, ids, probs);
    ++counts[static_cast<std::size_t>(token)];
  }
  for (std::size_t index = 0; index < counts.size(); ++index) {
    const double frequency =
        static_cast<double>(counts[index]) / static_cast<double>(draws);
    Expect(Near(frequency, reported[index], 0.01),
           "drawn frequency matches reported probability");
  }

  // Two -inf rows leave 22 scored rows: asking for more must throw rather
  // than report a short row.
  bool threw = false;
  try {
    std::vector<TokenId> extra_ids;
    std::vector<float> extra_probs;
    (void)DrawSampledRow(std::span<const float>(logits), temperature, 25, &rng,
                         extra_ids, extra_probs);
  } catch (const std::exception&) {
    threw = true;
  }
  Expect(threw, "too few scored rows throws");

  if (fails != 0) {
    std::cerr << "KAT_MTP_SAMPLED_ROW FAIL " << fails << "\n";
    return 1;
  }
  std::cout << "KAT_MTP_SAMPLED_ROW PASS\n";
  return 0;
}
