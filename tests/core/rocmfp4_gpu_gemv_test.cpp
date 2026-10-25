// Small-shape GPU check of the copied FP4 GEMV. x is all ones, so the Q8_1
// activation is exact (127 * 1/127). Every nibble is code +1 and every scale
// is 1/1024, so each of the 128 weights contributes 1/1024 and y is 0.125.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

extern "C" int gufo_fp4_gemv_fp4(int qtype, const void* W, const float* x,
                                 float* y, int k, int n, hipStream_t stream);
extern "C" int gufo_fp4_gemv_q8_codebook_check(void);

int main() {
  const int bad = gufo_fp4_gemv_q8_codebook_check();
  hipDeviceSynchronize();
  if (bad != 0) {
    std::cerr << "FAIL codebook bad=" << bad << '\n';
    return 1;
  }
  std::cout << "CODEBOOK PASS\n";

  const int k = 128;
  const int n = 2;
  const float expect = 0.125f;
  for (const int qtype : {100, 101}) {
    const int stride = qtype == 100 ? 18 : 17;
    const std::size_t row_bytes = static_cast<std::size_t>(stride) * (k / 32);
    const std::size_t bytes = static_cast<std::size_t>(n) * row_bytes + 256;
    std::vector<std::uint8_t> host(bytes, 0);
    for (int row = 0; row < n; ++row) {
      std::uint8_t* base =
          host.data() + static_cast<std::size_t>(row) * row_bytes;
      for (int block = 0; block < k / 32; ++block) {
        std::uint8_t* blk = base + static_cast<std::size_t>(block) * stride;
        for (int i = 0; i < 16; ++i) {
          blk[i] = 0x11;
        }
        blk[16] = 0x01;
        if (qtype == 100) {
          blk[17] = 0x01;
        }
      }
    }
    std::vector<float> x(static_cast<std::size_t>(k), 1.f);
    void* weights = nullptr;
    void* xd = nullptr;
    void* yd = nullptr;
    if (hipMalloc(&weights, bytes) != hipSuccess ||
        hipMalloc(&xd, x.size() * sizeof(float)) != hipSuccess ||
        hipMalloc(&yd, static_cast<std::size_t>(n) * sizeof(float)) !=
            hipSuccess) {
      std::cerr << "FAIL hipMalloc type " << qtype << '\n';
      return 1;
    }
    hipMemcpy(weights, host.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(xd, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(yd, 0, static_cast<std::size_t>(n) * sizeof(float));
    const int rc =
        gufo_fp4_gemv_fp4(qtype, weights, static_cast<const float*>(xd),
                          static_cast<float*>(yd), k, n, nullptr);
    hipDeviceSynchronize();
    if (rc != 0) {
      std::cerr << "FAIL gemv rc=" << rc << " type=" << qtype << '\n';
      return 1;
    }
    std::vector<float> y(static_cast<std::size_t>(n));
    hipMemcpy(y.data(), yd, y.size() * sizeof(float), hipMemcpyDeviceToHost);
    for (int row = 0; row < n; ++row) {
      const float err = std::fabs(y[static_cast<std::size_t>(row)] - expect);
      std::cout << "type " << qtype << " row " << row
                << " y=" << y[static_cast<std::size_t>(row)]
                << " expect=" << expect << " abs=" << err << '\n';
      if (!(err < 1e-4f)) {
        std::cerr << "FAIL value type=" << qtype << " row=" << row << '\n';
        return 1;
      }
    }
    hipFree(weights);
    hipFree(xd);
    hipFree(yd);
  }
  std::cout << "GPU GEMV PASS k=128 n=2 dual+fast\n";
  return 0;
}
