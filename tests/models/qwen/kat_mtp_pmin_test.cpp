#include "src/models/qwen/hip/mtp/kat_mtp_pmin.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {

int fails = 0;

void Expect(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "FAIL " << what << "\n";
    ++fails;
  }
}

bool Near(float got, float want, float tol) {
  return std::fabs(got - want) <= tol;
}

}  // namespace

int main() {
  using superroc::hip::MtpTop1Probability;
  Expect(MtpTop1Probability({}) == 0.0F, "empty");
  Expect(Near(MtpTop1Probability(std::vector<float>{3.0F}), 1.0F, 1e-6F),
         "one logit");

  const std::vector<float> equal(10, 1.0F);
  Expect(Near(MtpTop1Probability(equal), 0.1F, 1e-5F), "ten equal");

  std::vector<float> peaked(10, -20.0F);
  peaked[3] = 5.0F;
  Expect(Near(MtpTop1Probability(peaked), 1.0F, 1e-4F), "one peak");

  // Two equal peaks anywhere in the row share the mass.
  std::vector<float> two(4096, -40.0F);
  two[0] = 8.0F;
  two[4095] = 8.0F;
  Expect(Near(MtpTop1Probability(two), 0.5F, 1e-4F), "two peaks");

  // Every row counts: 99 rows one unit below the peak take most of the mass.
  std::vector<float> crowd(100, 1.0F);
  crowd[7] = 2.0F;
  const float expected_crowd = 1.0F / (1.0F + (99.0F * std::exp(-1.0F)));
  Expect(Near(MtpTop1Probability(crowd), expected_crowd, 1e-5F),
         "the whole row is summed");

  // Logits that are not finite add nothing, and never become the peak.
  std::vector<float> odd = peaked;
  odd.push_back(-std::numeric_limits<float>::infinity());
  odd.push_back(std::nanf(""));
  odd.push_back(std::numeric_limits<float>::infinity());
  Expect(Near(MtpTop1Probability(odd), MtpTop1Probability(peaked), 1e-6F),
         "non-finite logits are ignored");
  const std::vector<float> none{std::nanf(""),
                                std::numeric_limits<float>::infinity()};
  Expect(MtpTop1Probability(none) == 0.0F, "no finite logit");

  // An unparsable SUPERROC_MTP_PMIN must not silently disable the gate.
  unsetenv("SUPERROC_MTP_PMIN");
  Expect(Near(superroc::hip::MtpPMinFromEnv(), 0.0F, 0.0F), "unset p-min is 0");
  setenv("SUPERROC_MTP_PMIN", "abc", 1);
  Expect(Near(superroc::hip::MtpPMinFromEnv(), 0.0F, 0.0F),
         "unparsable p-min falls back to 0");
  setenv("SUPERROC_MTP_PMIN", "0.5", 1);
  Expect(Near(superroc::hip::MtpPMinFromEnv(), 0.5F, 1e-6F), "0.5 parses");
  setenv("SUPERROC_MTP_PMIN", "5", 1);
  Expect(Near(superroc::hip::MtpPMinFromEnv(), 1.0F, 1e-6F), "5 clamps to 1");
  setenv("SUPERROC_MTP_PMIN", "-2", 1);
  Expect(Near(superroc::hip::MtpPMinFromEnv(), 0.0F, 1e-6F), "-2 clamps to 0");
  setenv("SUPERROC_MTP_PMIN", "nan", 1);
  Expect(Near(superroc::hip::MtpPMinFromEnv(), 0.0F, 0.0F),
         "nan falls back to 0");
  unsetenv("SUPERROC_MTP_PMIN");

  // Unset: a dense block gates at its default, a MoE block does not gate.
  using superroc::hip::MtpEffectivePMin;
  Expect(
      Near(MtpEffectivePMin(false), superroc::hip::kMtpDensePMinDefault, 0.0F),
      "dense default");
  Expect(MtpEffectivePMin(true) == 0.0F, "MoE default is no gate");
  setenv("SUPERROC_MTP_PMIN", "0", 1);
  Expect(MtpEffectivePMin(false) == 0.0F, "0 turns the dense gate off");
  setenv("SUPERROC_MTP_PMIN", "0.7", 1);
  Expect(Near(MtpEffectivePMin(false), 0.7F, 1e-6F) &&
             Near(MtpEffectivePMin(true), 0.7F, 1e-6F),
         "a set value applies to both kinds");
  unsetenv("SUPERROC_MTP_PMIN");

  if (fails != 0) {
    std::cerr << "KAT_MTP_PMIN FAIL " << fails << "\n";
    return 1;
  }
  std::cout << "KAT_MTP_PMIN PASS\n";
  return 0;
}
