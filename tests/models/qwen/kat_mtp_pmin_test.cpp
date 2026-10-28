#include "src/models/qwen/hip/mtp/kat_mtp_pmin.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
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
  using superroc::hip::KatMtpTop1Probability;
  Expect(KatMtpTop1Probability({}) == 0.0F, "empty");
  Expect(Near(KatMtpTop1Probability(std::vector<float>{3.0F}), 1.0F, 1e-6F),
         "one logit");

  const std::vector<float> equal(10, 1.0F);
  Expect(Near(KatMtpTop1Probability(equal), 0.1F, 1e-5F), "ten equal");

  std::vector<float> peaked(10, -20.0F);
  peaked[3] = 5.0F;
  Expect(Near(KatMtpTop1Probability(peaked), 1.0F, 1e-4F), "one peak");

  // The 11th logit is another peak. Top-10 keeps both peaks, so the max
  // probability is about 1/2, not 1.
  std::vector<float> two(11, -40.0F);
  two[0] = 8.0F;
  two[10] = 8.0F;
  Expect(Near(KatMtpTop1Probability(two), 0.5F, 1e-4F), "two peaks in top-10");

  // A value below the current 10th largest (-20 here) must not move the max
  // probability. (-1.0 would be the second largest and would change it.)
  std::vector<float> ignored = peaked;
  ignored.push_back(-100.0F);
  Expect(Near(KatMtpTop1Probability(ignored), KatMtpTop1Probability(peaked),
              1e-6F),
         "11th logit below top-10 does not move the max");

  // RV44: an unparsable SUPERROC_MTP_PMIN must not silently disable the gate.
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

  if (fails != 0) {
    std::cerr << "KAT_MTP_PMIN FAIL " << fails << "\n";
    return 1;
  }
  std::cout << "KAT_MTP_PMIN PASS\n";
  return 0;
}
