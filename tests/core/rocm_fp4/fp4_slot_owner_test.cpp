// SQ10: FP4 slot ownership. Two host threads on two streams run interleaved
// quantize -> GEMV on different inputs; each stream's outputs must equal its
// single-stream reference bit-for-bit. On 1194fea's process-global slots the
// second stream's quantize overwrites the first's activation, so this test
// FAILS there (the bug); with per-stream slot sets it passes.
//
// The interleaved workload uses only the long-stable entry points, so this
// file compiles against 1194fea too (the ownership-API checks are ifdef'd).
#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__unix__)
#include <unistd.h>
#endif

#include "src/core/rocm_fp4/superroc_fp4_gemv_q8.hpp"
#include "src/core/rocm_fp4/superroc_fp4_quant.hpp"
#ifdef SUPERROC_FP4_SLOT_OWNER
#include "src/models/qwen/hip/fp4_launch.hpp"
#endif

namespace {

int g_failures = 0;

#define SQ10_CHECK_HIP(e)                                             \
  do {                                                                \
    const hipError_t _e = (e);                                        \
    if (_e != hipSuccess) {                                           \
      std::fprintf(stderr, "FAIL %s:%d HIP %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(_e));                            \
      ++g_failures;                                                   \
      return 1;                                                       \
    }                                                                 \
  } while (0)

#define SQ10_EXPECT(cond, msg)                                          \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, msg); \
      ++g_failures;                                                     \
      return 1;                                                         \
    }                                                                   \
  } while (0)

// Two-party barrier a failing thread can trip: the peer wakes and returns
// instead of hanging at the barrier (a pthread_barrier would deadlock the
// fail path, since exactly one side mismatches per clobbered step).
struct AbortBarrier {
  explicit AbortBarrier(int p) : parties(p) {}
  bool wait(std::atomic<bool>& stop) {
    std::unique_lock<std::mutex> lk(mu);
    if (stop.load()) {
      cv.notify_all();
      return false;
    }
    const int g = gen;
    if (++count == parties) {
      count = 0;
      ++gen;
      cv.notify_all();
      return !stop.load();
    }
    cv.wait(lk, [&] { return stop.load() || gen != g; });
    return !stop.load();
  }
  void trip(std::atomic<bool>& stop) {
    std::lock_guard<std::mutex> lk(mu);
    stop.store(true);
    cv.notify_all();
  }

  std::mutex mu;
  std::condition_variable cv;
  const int parties;
  int count = 0;
  int gen = 0;
};

// SPEC2 (RV 4.1): bounded wait on a flag with a deadline, not a bare sleep.
// Returns true when the flag set before the deadline.
bool WaitFlagUntil(const std::atomic<bool>& flag,
                   std::chrono::milliseconds deadline) {
  const auto t0 = std::chrono::steady_clock::now();
  while (!flag.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() - t0 > deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  return true;
}

struct InterleaveArgs {
  hipStream_t stream;
  const float* dx;
  float* dy;
  const float* ref;
  const void* dw;
  int k;
  int n;
  int qtype;
  int steps;
  AbortBarrier* barrier;
  std::atomic<int>* mismatches;
  std::atomic<bool>* stop;
};

void InterleaveWorker(InterleaveArgs* a) {
  std::vector<float> host(static_cast<std::size_t>(a->n));
  for (int t = 0; t < a->steps && !a->stop->load(); ++t) {
    if (superroc_fp4_quant_q8(a->dx, a->k, SUPERROC_FP4_Q8_SLOT_GENERIC,
                              a->stream) != SUPERROC_FP4_OK) {
      std::fprintf(stderr, "FAIL worker quant step %d\n", t);
      a->mismatches->fetch_add(1);
      a->barrier->trip(*a->stop);
      return;
    }
    // Both quants are queued before either GEMV: on shared slots the GEMVs
    // read whichever quantize landed last, so at least one side is wrong.
    if (!a->barrier->wait(*a->stop))
      return;
    if (superroc_fp4_gemv_fp4_q8(a->qtype, a->dw, SUPERROC_FP4_Q8_SLOT_GENERIC,
                                 a->dy, a->k, a->n,
                                 a->stream) != SUPERROC_FP4_OK) {
      std::fprintf(stderr, "FAIL worker gemv step %d\n", t);
      a->mismatches->fetch_add(1);
      a->barrier->trip(*a->stop);
      return;
    }
    if (hipStreamSynchronize(a->stream) != hipSuccess) {
      std::fprintf(stderr, "FAIL worker sync step %d\n", t);
      a->mismatches->fetch_add(1);
      a->barrier->trip(*a->stop);
      return;
    }
    if (hipMemcpy(host.data(), a->dy,
                  static_cast<std::size_t>(a->n) * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      std::fprintf(stderr, "FAIL worker copy step %d\n", t);
      a->mismatches->fetch_add(1);
      a->barrier->trip(*a->stop);
      return;
    }
    if (std::memcmp(host.data(), a->ref,
                    static_cast<std::size_t>(a->n) * sizeof(float)) != 0) {
      const int count = a->mismatches->fetch_add(1);
      if (count < 4) {
        std::fprintf(stderr, "FAIL worker mismatch step %d\n", t);
      }
      a->barrier->trip(*a->stop);
      return;
    }
    if (!a->barrier->wait(*a->stop))
      return;
  }
}

int RunInterleave() {
#ifdef SUPERROC_FP4_SLOT_OWNER
  if (superroc_fp4_slot_sets_configure(2) != SUPERROC_FP4_OK) {
    std::fprintf(stderr, "FAIL configure(2)\n");
    return 1;
  }
#endif
  const int k = 2048;
  const int n = 256;
  const int qtype = SUPERROC_FP4_Q4_0_ROCMFP4_FAST;
  const std::size_t rb = superroc_fp4_row_bytes(qtype, k);
  if (rb == 0) {
    std::fprintf(stderr, "FAIL row_bytes\n");
    return 1;
  }
  std::vector<std::uint8_t> w(static_cast<std::size_t>(n) * rb);
  for (std::size_t i = 0; i < w.size(); ++i) {
    w[i] = static_cast<std::uint8_t>((i * 2654435761u) >> 13);
  }
  std::vector<float> hx0(static_cast<std::size_t>(k));
  std::vector<float> hx1(static_cast<std::size_t>(k));
  std::srand(12345);
  for (int i = 0; i < k; ++i) {
    hx0[static_cast<std::size_t>(i)] =
        static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 4.0f -
        2.0f;
  }
  std::srand(987654321);
  for (int i = 0; i < k; ++i) {
    hx1[static_cast<std::size_t>(i)] =
        static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 4.0f -
        2.0f;
  }

  hipStream_t s0 = nullptr;
  hipStream_t s1 = nullptr;
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
  void* dw = nullptr;
  float* dx0 = nullptr;
  float* dx1 = nullptr;
  float* dy0 = nullptr;
  float* dy1 = nullptr;
  SQ10_CHECK_HIP(hipMalloc(&dw, static_cast<std::size_t>(n) * rb));
  SQ10_CHECK_HIP(hipMalloc(&dx0, static_cast<std::size_t>(k) * sizeof(float)));
  SQ10_CHECK_HIP(hipMalloc(&dx1, static_cast<std::size_t>(k) * sizeof(float)));
  SQ10_CHECK_HIP(hipMalloc(&dy0, static_cast<std::size_t>(n) * sizeof(float)));
  SQ10_CHECK_HIP(hipMalloc(&dy1, static_cast<std::size_t>(n) * sizeof(float)));
  SQ10_CHECK_HIP(hipMemcpy(dw, w.data(), static_cast<std::size_t>(n) * rb,
                           hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(hipMemcpy(dx0, hx0.data(),
                           static_cast<std::size_t>(k) * sizeof(float),
                           hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(hipMemcpy(dx1, hx1.data(),
                           static_cast<std::size_t>(k) * sizeof(float),
                           hipMemcpyHostToDevice));

  // Single-stream references, sequential and synced: correct on every tree.
  std::vector<float> ref0(static_cast<std::size_t>(n));
  std::vector<float> ref1(static_cast<std::size_t>(n));
  SQ10_EXPECT(superroc_fp4_quant_q8(dx0, k, SUPERROC_FP4_Q8_SLOT_GENERIC, s0) ==
                  SUPERROC_FP4_OK,
              "ref quant 0");
  SQ10_EXPECT(superroc_fp4_gemv_fp4_q8(qtype, dw, SUPERROC_FP4_Q8_SLOT_GENERIC,
                                       dy0, k, n, s0) == SUPERROC_FP4_OK,
              "ref gemv 0");
  SQ10_CHECK_HIP(hipStreamSynchronize(s0));
  SQ10_CHECK_HIP(hipMemcpy(ref0.data(), dy0,
                           static_cast<std::size_t>(n) * sizeof(float),
                           hipMemcpyDeviceToHost));
  SQ10_EXPECT(superroc_fp4_quant_q8(dx1, k, SUPERROC_FP4_Q8_SLOT_GENERIC, s0) ==
                  SUPERROC_FP4_OK,
              "ref quant 1");
  SQ10_EXPECT(superroc_fp4_gemv_fp4_q8(qtype, dw, SUPERROC_FP4_Q8_SLOT_GENERIC,
                                       dy1, k, n, s0) == SUPERROC_FP4_OK,
              "ref gemv 1");
  SQ10_CHECK_HIP(hipStreamSynchronize(s0));
  SQ10_CHECK_HIP(hipMemcpy(ref1.data(), dy1,
                           static_cast<std::size_t>(n) * sizeof(float),
                           hipMemcpyDeviceToHost));
  if (std::memcmp(ref0.data(), ref1.data(),
                  static_cast<std::size_t>(n) * sizeof(float)) == 0) {
    std::fprintf(stderr, "FAIL refs identical (inputs must differ)\n");
    return 1;
  }

  AbortBarrier barrier(2);
  std::atomic<int> mismatches(0);
  std::atomic<bool> stop(false);
  InterleaveArgs a0{s0, dx0,   dy0,   ref0.data(), dw,          k,
                    n,  qtype, 10000, &barrier,    &mismatches, &stop};
  InterleaveArgs a1{s1, dx1,   dy1,   ref1.data(), dw,          k,
                    n,  qtype, 10000, &barrier,    &mismatches, &stop};
  std::thread t0(InterleaveWorker, &a0);
  std::thread t1(InterleaveWorker, &a1);
  t0.join();
  t1.join();
  SQ10_EXPECT(mismatches.load() == 0, "interleave bit-identical");
  std::printf("INTERLEAVE PASS (2 streams x 10000 steps, bit-identical)\n");

#ifdef SUPERROC_FP4_SLOT_OWNER
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(s0) == SUPERROC_FP4_OK,
              "detach s0");
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(s1) == SUPERROC_FP4_OK,
              "detach s1");
#endif
  SQ10_CHECK_HIP(hipStreamDestroy(s0));
  SQ10_CHECK_HIP(hipStreamDestroy(s1));
  SQ10_CHECK_HIP(hipFree(dw));
  SQ10_CHECK_HIP(hipFree(dx0));
  SQ10_CHECK_HIP(hipFree(dx1));
  SQ10_CHECK_HIP(hipFree(dy0));
  SQ10_CHECK_HIP(hipFree(dy1));
  return 0;
}

#ifdef SUPERROC_FP4_SLOT_OWNER
int RunExhaustion() {
  hipStream_t t0 = nullptr;
  hipStream_t t1 = nullptr;
  hipStream_t t2 = nullptr;
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&t0, hipStreamNonBlocking));
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&t1, hipStreamNonBlocking));
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&t2, hipStreamNonBlocking));
  const int a0 = superroc_fp4_slot_stream_attach(t0);
  const int a1 = superroc_fp4_slot_stream_attach(t1);
  SQ10_EXPECT(a0 >= 0 && a1 >= 0, "attach t0/t1");
  SQ10_EXPECT(a0 != a1, "no alias while held");
  SQ10_EXPECT(superroc_fp4_slot_stream_attach(t0) == a0, "attach idempotent");
#ifdef SUPERROC_FP4_SLOT_HAS_ALIAS
  // SPEC2 (R2): the N+1-th stream is refused at once, not parked.
  const unsigned long r0 = superroc_fp4_slot_refuse_count();
  SQ10_EXPECT(superroc_fp4_slot_stream_attach(t2) == -SUPERROC_FP4_ERR_BUSY,
              "N+1-th attach refuses");
  SQ10_EXPECT(superroc_fp4_slot_refuse_count() == r0 + 1, "refusal counted");
  SQ10_EXPECT(superroc_fp4_slot_stream_attach(t2) == -SUPERROC_FP4_ERR_BUSY,
              "repeat refusal");
  SQ10_EXPECT(superroc_fp4_slot_refuse_count() == r0 + 2, "repeat counted");
  // A resolving entry on the refused stream declines the same way.
  float* dx = nullptr;
  SQ10_CHECK_HIP(hipMalloc(&dx, 64 * sizeof(float)));
  SQ10_EXPECT(superroc_fp4_quant_q8(dx, 64, SUPERROC_FP4_Q8_SLOT_GENERIC, t2) ==
                  -SUPERROC_FP4_ERR_BUSY,
              "entry declines on refused stream");
  SQ10_CHECK_HIP(hipFree(dx));
  // The waiting attach survives as a separately named test-only entry:
  // it blocks while both sets are held and wakes on detach. Proven with a
  // deadline flag (RV 4.1), not a bare sleep.
  std::atomic<bool> done(false);
  int a2 = -1;
  std::thread waiter([&] {
    a2 = superroc_fp4_slot_stream_attach_wait(t2);
    done.store(true, std::memory_order_release);
  });
  SQ10_EXPECT(!WaitFlagUntil(done, std::chrono::milliseconds(300)),
              "wait entry holds");
  const unsigned long w0 = superroc_fp4_slot_wait_count();
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(t0) == SUPERROC_FP4_OK,
              "detach t0");
  SQ10_EXPECT(WaitFlagUntil(done, std::chrono::milliseconds(5000)),
              "waiter wakes");
  waiter.join();
  SQ10_EXPECT(a2 == a0, "waiter reuses the freed set");
  SQ10_EXPECT(superroc_fp4_slot_wait_count() == w0 + 1, "wait counted");
  std::printf("wait_count=%lu wait_us=%lu refuse_count=%lu\n",
              superroc_fp4_slot_wait_count(), superroc_fp4_slot_wait_us(),
              superroc_fp4_slot_refuse_count());
#else
  // sq10-1 legacy: attach waits (deadline-flagged, RV 4.1 backport).
  std::atomic<bool> done(false);
  int a2 = -1;
  std::thread waiter([&] {
    a2 = superroc_fp4_slot_stream_attach(t2);
    done.store(true, std::memory_order_release);
  });
  SQ10_EXPECT(!WaitFlagUntil(done, std::chrono::milliseconds(2000)),
              "N+1-th attach waits");
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(t0) == SUPERROC_FP4_OK,
              "detach t0");
  SQ10_EXPECT(WaitFlagUntil(done, std::chrono::milliseconds(5000)),
              "waiter wakes");
  waiter.join();
  SQ10_EXPECT(a2 == a0, "waiter reuses the freed set");
  SQ10_EXPECT(superroc_fp4_slot_wait_count() >= 1, "wait counted");
  std::printf("wait_count=%lu wait_us=%lu\n", superroc_fp4_slot_wait_count(),
              superroc_fp4_slot_wait_us());
#endif
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(t1) == SUPERROC_FP4_OK,
              "detach t1");
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(t2) == SUPERROC_FP4_OK,
              "detach t2");
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(t2) == SUPERROC_FP4_ERR_ARG,
              "double detach is loud");
  SQ10_CHECK_HIP(hipStreamDestroy(t0));
  SQ10_CHECK_HIP(hipStreamDestroy(t1));
  SQ10_CHECK_HIP(hipStreamDestroy(t2));
#ifdef SUPERROC_FP4_SLOT_HAS_ALIAS
  std::printf("EXHAUSTION PASS (refuses at once, never aliases)\n");
#else
  std::printf("EXHAUSTION PASS (waits, never aliases)\n");
#endif
  return 0;
}

int RunZeroAlloc() {
  const int k = 2048;
  const int n = 256;
  const int qtype = SUPERROC_FP4_Q4_0_ROCMFP4_FAST;
  hipStream_t c0 = nullptr;
  hipStream_t c1 = nullptr;
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&c0, hipStreamNonBlocking));
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&c1, hipStreamNonBlocking));
  const int s0 = superroc_fp4_slot_stream_attach(c0);
  const int s1 = superroc_fp4_slot_stream_attach(c1);
  SQ10_EXPECT(s0 >= 0 && s1 >= 0 && s0 != s1, "attach c0/c1");
  SQ10_EXPECT(superroc_fp4_slots_warm_q8(k, c0) == SUPERROC_FP4_OK, "warm q8");
  SQ10_EXPECT(superroc_fp4_slots_warm_moe(512, 0, 4096) == SUPERROC_FP4_OK,
              "warm moe");
  SQ10_EXPECT(superroc::hip::RocmFp4SwigluScratchWarm(1024), "warm swiglu");
  // The two streams own different SwiGLU pairs.
  float *g0 = nullptr, *u0 = nullptr, *g1 = nullptr, *u1 = nullptr;
  SQ10_EXPECT(superroc::hip::RocmFp4SwigluScratchReady(1024, c0),
              "swiglu ready 0");
  SQ10_EXPECT(superroc::hip::RocmFp4SwigluScratchReady(1024, c1),
              "swiglu ready 1");
  SQ10_EXPECT(superroc::hip::RocmFp4SwigluScratch(1024, g0, u0, c0),
              "swiglu 0");
  SQ10_EXPECT(superroc::hip::RocmFp4SwigluScratch(1024, g1, u1, c1),
              "swiglu 1");
  SQ10_EXPECT(g0 != nullptr && u0 != nullptr && g1 != nullptr && u1 != nullptr,
              "swiglu non-null");
  SQ10_EXPECT(g0 != g1 && u0 != u1, "swiglu pairs differ");

  const std::size_t rb = superroc_fp4_row_bytes(qtype, k);
  std::vector<std::uint8_t> w(static_cast<std::size_t>(n) * rb, 0x11);
  std::vector<float> hx(static_cast<std::size_t>(k), 0.5f);
  void* dw = nullptr;
  float* dx = nullptr;
  float* dy = nullptr;
  SQ10_CHECK_HIP(hipMalloc(&dw, static_cast<std::size_t>(n) * rb));
  SQ10_CHECK_HIP(hipMalloc(&dx, static_cast<std::size_t>(k) * sizeof(float)));
  SQ10_CHECK_HIP(hipMalloc(&dy, static_cast<std::size_t>(n) * sizeof(float)));
  SQ10_CHECK_HIP(hipMemcpy(dw, w.data(), static_cast<std::size_t>(n) * rb,
                           hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(hipMemcpy(dx, hx.data(),
                           static_cast<std::size_t>(k) * sizeof(float),
                           hipMemcpyHostToDevice));
  // One synthetic expert for the grouped FFN leg.
  const int n_ff = 512;
  const std::size_t rb_k = superroc_fp4_row_bytes(qtype, k);
  const std::size_t rb_ff = superroc_fp4_row_bytes(qtype, n_ff);
  std::vector<std::uint8_t> gate(static_cast<std::size_t>(n_ff) * rb_k, 0x22);
  std::vector<std::uint8_t> up(static_cast<std::size_t>(n_ff) * rb_k, 0x33);
  std::vector<std::uint8_t> down(static_cast<std::size_t>(k) * rb_ff, 0x44);
  void *d_gate = nullptr, *d_up = nullptr, *d_down = nullptr;
  int* d_ids = nullptr;
  float* d_wts = nullptr;
  float* d_moe_y = nullptr;
  SQ10_CHECK_HIP(hipMalloc(&d_gate, gate.size()));
  SQ10_CHECK_HIP(hipMalloc(&d_up, up.size()));
  SQ10_CHECK_HIP(hipMalloc(&d_down, down.size()));
  SQ10_CHECK_HIP(hipMalloc(&d_ids, sizeof(int)));
  SQ10_CHECK_HIP(hipMalloc(&d_wts, sizeof(float)));
  SQ10_CHECK_HIP(
      hipMalloc(&d_moe_y, static_cast<std::size_t>(k) * sizeof(float)));
  SQ10_CHECK_HIP(
      hipMemcpy(d_gate, gate.data(), gate.size(), hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(hipMemcpy(d_up, up.data(), up.size(), hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(
      hipMemcpy(d_down, down.data(), down.size(), hipMemcpyHostToDevice));
  const int zero_id = 0;
  const float one_w = 1.0f;
  SQ10_CHECK_HIP(
      hipMemcpy(d_ids, &zero_id, sizeof(int), hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(
      hipMemcpy(d_wts, &one_w, sizeof(float), hipMemcpyHostToDevice));

  const unsigned long c0alloc = superroc_fp4_slot_alloc_count();
  SQ10_EXPECT(superroc_fp4_quant_q8(dx, k, SUPERROC_FP4_Q8_SLOT_GENERIC, c0) ==
                  SUPERROC_FP4_OK,
              "decode quant c0");
  SQ10_EXPECT(superroc_fp4_gemv_fp4_q8(qtype, dw, SUPERROC_FP4_Q8_SLOT_GENERIC,
                                       dy, k, n, c0) == SUPERROC_FP4_OK,
              "decode gemv c0");
  SQ10_EXPECT(superroc_fp4_quant_q8(dx, k, SUPERROC_FP4_Q8_SLOT_GENERIC, c1) ==
                  SUPERROC_FP4_OK,
              "decode quant c1");
  SQ10_EXPECT(superroc_fp4_gemv_fp4_q8(qtype, dw, SUPERROC_FP4_Q8_SLOT_GENERIC,
                                       dy, k, n, c1) == SUPERROC_FP4_OK,
              "decode gemv c1");
  SQ10_EXPECT(superroc_fp4_expert_ffn_dev(qtype, d_gate, d_up, d_down, dx,
                                          d_ids, d_wts, 1, 1, k, n_ff, 0.0f,
                                          d_moe_y, c0) == SUPERROC_FP4_OK,
              "decode expert");
  SQ10_EXPECT(superroc_fp4_moe_dev_check_set(s0) == SUPERROC_FP4_OK,
              "decode check");
  SQ10_CHECK_HIP(hipStreamSynchronize(c0));
  SQ10_CHECK_HIP(hipStreamSynchronize(c1));
  const unsigned long c1alloc = superroc_fp4_slot_alloc_count();
  SQ10_EXPECT(c0alloc == c1alloc, "zero slot allocs across decode");

  SQ10_EXPECT(superroc_fp4_slot_stream_detach(c0) == SUPERROC_FP4_OK,
              "detach c0");
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(c1) == SUPERROC_FP4_OK,
              "detach c1");
  SQ10_CHECK_HIP(hipStreamDestroy(c0));
  SQ10_CHECK_HIP(hipStreamDestroy(c1));
  SQ10_CHECK_HIP(hipFree(dw));
  SQ10_CHECK_HIP(hipFree(dx));
  SQ10_CHECK_HIP(hipFree(dy));
  SQ10_CHECK_HIP(hipFree(d_gate));
  SQ10_CHECK_HIP(hipFree(d_up));
  SQ10_CHECK_HIP(hipFree(d_down));
  SQ10_CHECK_HIP(hipFree(d_ids));
  SQ10_CHECK_HIP(hipFree(d_wts));
  SQ10_CHECK_HIP(hipFree(d_moe_y));
  std::printf("ZEROALLOC PASS (counter %lu -> %lu)\n", c0alloc, c1alloc);
  return 0;
}

// SPEC2 gate 1: the production shape. Configure once, two sessions attach
// (both sets busy, as under -j 2), then a helper stream launches an FP4
// matrix. On sq10-1 (no alias API: the ifdef below compiles out) the helper
// waits for a set nobody ever frees, so this case FAILS on the deadline --
// the KAT+MTP freeze in miniature. With the fix the helper is aliased to
// its session's set and the launch completes.
struct HelperArgs {
  hipStream_t stream;
  const float* dx;
  float* dy;
  const float* ref;
  const void* dw;
  int k;
  int n;
  int qtype;
  std::atomic<bool>* done;
  std::atomic<int>* status;
};

void HelperWorker(HelperArgs* a) {
  std::vector<float> host(static_cast<std::size_t>(a->n));
  int st = 1;
  if (superroc_fp4_gemv_fp4(a->qtype, a->dw, a->dx, a->dy, a->k, a->n,
                            a->stream) == SUPERROC_FP4_OK &&
      hipStreamSynchronize(a->stream) == hipSuccess &&
      hipMemcpy(host.data(), a->dy,
                static_cast<std::size_t>(a->n) * sizeof(float),
                hipMemcpyDeviceToHost) == hipSuccess &&
      std::memcmp(host.data(), a->ref,
                  static_cast<std::size_t>(a->n) * sizeof(float)) == 0) {
    st = 0;
  }
  a->status->store(st, std::memory_order_release);
  a->done->store(true, std::memory_order_release);
}

int RunProductionHelper() {
  if (superroc_fp4_slot_sets_configure(2) != SUPERROC_FP4_OK) {
    std::fprintf(stderr, "FAIL production configure(2)\n");
    return 1;
  }
  const int k = 2048;
  const int n = 256;
  const int qtype = SUPERROC_FP4_Q4_0_ROCMFP4_FAST;
  const std::size_t rb = superroc_fp4_row_bytes(qtype, k);
  SQ10_EXPECT(rb != 0, "production row_bytes");
  std::vector<std::uint8_t> w(static_cast<std::size_t>(n) * rb);
  for (std::size_t i = 0; i < w.size(); ++i) {
    w[i] = static_cast<std::uint8_t>((i * 2654435761u) >> 11);
  }
  std::vector<float> hx(static_cast<std::size_t>(k));
  std::srand(424242);
  for (int i = 0; i < k; ++i) {
    hx[static_cast<std::size_t>(i)] =
        static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 4.0f -
        2.0f;
  }
  hipStream_t s0 = nullptr;
  hipStream_t s1 = nullptr;
  hipStream_t h = nullptr;
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&h, hipStreamNonBlocking));
  void* dw = nullptr;
  float* dx = nullptr;
  float* dy = nullptr;
  SQ10_CHECK_HIP(hipMalloc(&dw, static_cast<std::size_t>(n) * rb));
  SQ10_CHECK_HIP(hipMalloc(&dx, static_cast<std::size_t>(k) * sizeof(float)));
  SQ10_CHECK_HIP(hipMalloc(&dy, static_cast<std::size_t>(n) * sizeof(float)));
  SQ10_CHECK_HIP(hipMemcpy(dw, w.data(), static_cast<std::size_t>(n) * rb,
                           hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(hipMemcpy(dx, hx.data(),
                           static_cast<std::size_t>(k) * sizeof(float),
                           hipMemcpyHostToDevice));
  // Two sessions take both sets; the serial reference runs on s0.
  const int a0 = superroc_fp4_slot_stream_attach(s0);
  const int a1 = superroc_fp4_slot_stream_attach(s1);
  SQ10_EXPECT(a0 >= 0 && a1 >= 0 && a0 != a1, "production sessions attach");
  std::vector<float> ref(static_cast<std::size_t>(n));
  SQ10_EXPECT(
      superroc_fp4_gemv_fp4(qtype, dw, dx, dy, k, n, s0) == SUPERROC_FP4_OK,
      "production ref launch");
  SQ10_CHECK_HIP(hipStreamSynchronize(s0));
  SQ10_CHECK_HIP(hipMemcpy(ref.data(), dy,
                           static_cast<std::size_t>(n) * sizeof(float),
                           hipMemcpyDeviceToHost));
#ifdef SUPERROC_FP4_SLOT_HAS_ALIAS
  SQ10_EXPECT(superroc_fp4_slot_stream_alias(h, s0) == a0, "production alias");
  const unsigned long r0 = superroc_fp4_slot_refuse_count();
#endif
  std::atomic<bool> done(false);
  std::atomic<int> status(1);
  HelperArgs args{h, dx, dy, ref.data(), dw, k, n, qtype, &done, &status};
  std::thread worker(HelperWorker, &args);
  if (!WaitFlagUntil(done, std::chrono::milliseconds(15000))) {
    std::fprintf(stderr,
                 "FAIL production: helper launch hung 15 s (sq10-1 freeze)\n");
    // The worker waits on a set nobody frees; normal teardown hangs with
    // it (T2 rc=124), so exit without running static destructors.
    std::fflush(nullptr);
#if defined(__unix__)
    _exit(1);
#else
    std::exit(1);
#endif
  }
  worker.join();
  SQ10_EXPECT(status.load() == 0, "production helper bit-identical");
#ifdef SUPERROC_FP4_SLOT_HAS_ALIAS
  SQ10_EXPECT(superroc_fp4_slot_refuse_count() == r0, "production no refusal");
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(h) == SUPERROC_FP4_OK,
              "production detach helper");
  SQ10_EXPECT(superroc_fp4_slot_debug_alias_size() == 0,
              "production alias flat");
#endif
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(s0) == SUPERROC_FP4_OK,
              "production detach s0");
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(s1) == SUPERROC_FP4_OK,
              "production detach s1");
  SQ10_CHECK_HIP(hipStreamDestroy(s0));
  SQ10_CHECK_HIP(hipStreamDestroy(s1));
  SQ10_CHECK_HIP(hipStreamDestroy(h));
  SQ10_CHECK_HIP(hipFree(dw));
  SQ10_CHECK_HIP(hipFree(dx));
  SQ10_CHECK_HIP(hipFree(dy));
  std::printf("PRODUCTION PASS (helper shares its session's set)\n");
  return 0;
}

#ifdef SUPERROC_FP4_SLOT_HAS_ALIAS
// SPEC2 gate 5: the alias contract. A helper resolves to its owner's set
// and launches bit-identically there; bad registrations are loud; detach
// removes aliases in both directions.
int RunAlias() {
  hipStream_t s0 = nullptr;
  hipStream_t h = nullptr;
  hipStream_t x = nullptr;
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&h, hipStreamNonBlocking));
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&x, hipStreamNonBlocking));
  const int a0 = superroc_fp4_slot_stream_attach(s0);
  SQ10_EXPECT(a0 >= 0, "alias owner attach");
  SQ10_EXPECT(superroc_fp4_slot_stream_alias(h, s0) == a0, "alias registers");
  SQ10_EXPECT(superroc_fp4_slot_stream_alias(h, s0) == a0, "alias idempotent");
  SQ10_EXPECT(superroc_fp4_slot_stream_attach(h) == a0,
              "helper resolves owner");
  SQ10_EXPECT(superroc_fp4_slot_debug_alias_size() == 1, "alias size 1");
  // Bad registrations are loud.
  SQ10_EXPECT(superroc_fp4_slot_stream_alias(x, x) == SUPERROC_FP4_ERR_ARG,
              "self alias refused");
  SQ10_EXPECT(
      superroc_fp4_slot_stream_alias(nullptr, s0) == SUPERROC_FP4_ERR_ARG,
      "null helper refused");
  SQ10_EXPECT(superroc_fp4_slot_stream_alias(h, x) == SUPERROC_FP4_ERR_ARG,
              "unattached owner refused");
  const int ax = superroc_fp4_slot_stream_attach(x);
  SQ10_EXPECT(ax >= 0 && ax != a0, "alias second owner attach");
  SQ10_EXPECT(superroc_fp4_slot_stream_alias(h, x) == SUPERROC_FP4_ERR_ARG,
              "re-alias refused");
  SQ10_EXPECT(superroc_fp4_slot_stream_alias(x, s0) == SUPERROC_FP4_ERR_ARG,
              "owning helper refused");
  // The helper launches bit-identically on the owner's set.
  const int k = 1024;
  const int n = 128;
  const int qtype = SUPERROC_FP4_Q4_0_ROCMFP4_FAST;
  const std::size_t rb = superroc_fp4_row_bytes(qtype, k);
  std::vector<std::uint8_t> w(static_cast<std::size_t>(n) * rb, 0x5a);
  std::vector<float> hx(static_cast<std::size_t>(k), 0.25f);
  void* dw = nullptr;
  float* dx = nullptr;
  float* dy = nullptr;
  SQ10_CHECK_HIP(hipMalloc(&dw, static_cast<std::size_t>(n) * rb));
  SQ10_CHECK_HIP(hipMalloc(&dx, static_cast<std::size_t>(k) * sizeof(float)));
  SQ10_CHECK_HIP(hipMalloc(&dy, static_cast<std::size_t>(n) * sizeof(float)));
  SQ10_CHECK_HIP(hipMemcpy(dw, w.data(), static_cast<std::size_t>(n) * rb,
                           hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(hipMemcpy(dx, hx.data(),
                           static_cast<std::size_t>(k) * sizeof(float),
                           hipMemcpyHostToDevice));
  std::vector<float> ref(static_cast<std::size_t>(n));
  std::vector<float> got(static_cast<std::size_t>(n));
  SQ10_EXPECT(
      superroc_fp4_gemv_fp4(qtype, dw, dx, dy, k, n, s0) == SUPERROC_FP4_OK,
      "alias ref launch");
  SQ10_CHECK_HIP(hipStreamSynchronize(s0));
  SQ10_CHECK_HIP(hipMemcpy(ref.data(), dy,
                           static_cast<std::size_t>(n) * sizeof(float),
                           hipMemcpyDeviceToHost));
  SQ10_EXPECT(
      superroc_fp4_gemv_fp4(qtype, dw, dx, dy, k, n, h) == SUPERROC_FP4_OK,
      "alias helper launch");
  SQ10_CHECK_HIP(hipStreamSynchronize(h));
  SQ10_CHECK_HIP(hipMemcpy(got.data(), dy,
                           static_cast<std::size_t>(n) * sizeof(float),
                           hipMemcpyDeviceToHost));
  SQ10_EXPECT(std::memcmp(ref.data(), got.data(),
                          static_cast<std::size_t>(n) * sizeof(float)) == 0,
              "alias helper bit-identical");
  // Detach removes the alias as helper; detaching the owner drops aliases
  // pointing at it, and the ex-helper reattaches cleanly afterwards.
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(h) == SUPERROC_FP4_OK,
              "detach helper removes alias");
  SQ10_EXPECT(superroc_fp4_slot_debug_alias_size() == 0, "alias flat");
  SQ10_EXPECT(superroc_fp4_slot_stream_alias(h, s0) == a0, "re-alias");
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(s0) == SUPERROC_FP4_OK,
              "detach owner");
  SQ10_EXPECT(superroc_fp4_slot_debug_alias_size() == 0,
              "owner detach drops it");
  SQ10_EXPECT(superroc_fp4_slot_stream_attach(h) == a0, "ex-helper reattaches");
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(h) == SUPERROC_FP4_OK,
              "detach h");
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(x) == SUPERROC_FP4_OK,
              "detach x");
  SQ10_EXPECT(superroc_fp4_slot_debug_map_size() == 0, "map flat");
  SQ10_CHECK_HIP(hipStreamDestroy(s0));
  SQ10_CHECK_HIP(hipStreamDestroy(h));
  SQ10_CHECK_HIP(hipStreamDestroy(x));
  SQ10_CHECK_HIP(hipFree(dw));
  SQ10_CHECK_HIP(hipFree(dx));
  SQ10_CHECK_HIP(hipFree(dy));
  std::printf("ALIAS PASS\n");
  return 0;
}

// SPEC2 gate 5: create/destroy 1,000 attach/alias pairs; the maps stay flat.
int RunLifecycle() {
  SQ10_EXPECT(superroc_fp4_slot_debug_map_size() == 0,
              "lifecycle map starts 0");
  SQ10_EXPECT(superroc_fp4_slot_debug_alias_size() == 0,
              "lifecycle alias starts 0");
  for (int i = 0; i < 1000; ++i) {
    hipStream_t a = nullptr;
    hipStream_t b = nullptr;
    if (hipStreamCreateWithFlags(&a, hipStreamNonBlocking) != hipSuccess ||
        hipStreamCreateWithFlags(&b, hipStreamNonBlocking) != hipSuccess) {
      std::fprintf(stderr, "FAIL lifecycle create %d\n", i);
      return 1;
    }
    const int sa = superroc_fp4_slot_stream_attach(a);
    if (sa < 0 || superroc_fp4_slot_stream_alias(b, a) != sa ||
        superroc_fp4_slot_stream_detach(b) != SUPERROC_FP4_OK ||
        superroc_fp4_slot_stream_detach(a) != SUPERROC_FP4_OK) {
      std::fprintf(stderr, "FAIL lifecycle pair %d\n", i);
      return 1;
    }
    if (hipStreamDestroy(a) != hipSuccess ||
        hipStreamDestroy(b) != hipSuccess) {
      std::fprintf(stderr, "FAIL lifecycle destroy %d\n", i);
      return 1;
    }
  }
  SQ10_EXPECT(superroc_fp4_slot_debug_map_size() == 0, "lifecycle map flat");
  SQ10_EXPECT(superroc_fp4_slot_debug_alias_size() == 0,
              "lifecycle alias flat");
  // Functional proof: two attach, the third refuses, both detach.
  hipStream_t f0 = nullptr;
  hipStream_t f1 = nullptr;
  hipStream_t f2 = nullptr;
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&f0, hipStreamNonBlocking));
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&f1, hipStreamNonBlocking));
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&f2, hipStreamNonBlocking));
  const int b0 = superroc_fp4_slot_stream_attach(f0);
  const int b1 = superroc_fp4_slot_stream_attach(f1);
  SQ10_EXPECT(b0 >= 0 && b1 >= 0 && b0 != b1, "lifecycle functional attach");
  SQ10_EXPECT(superroc_fp4_slot_stream_attach(f2) == -SUPERROC_FP4_ERR_BUSY,
              "lifecycle functional refuse");
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(f0) == SUPERROC_FP4_OK,
              "detach f0");
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(f1) == SUPERROC_FP4_OK,
              "detach f1");
  SQ10_CHECK_HIP(hipStreamDestroy(f0));
  SQ10_CHECK_HIP(hipStreamDestroy(f1));
  SQ10_CHECK_HIP(hipStreamDestroy(f2));
  std::printf("LIFECYCLE PASS (1000 pairs, maps flat)\n");
  return 0;
}
#endif  // SUPERROC_FP4_SLOT_HAS_ALIAS

// SPEC2 gate 5 (RV 4.2): all five scratch groups interleaved on two
// streams -- GENERIC quant+GEMV, the m-pair, the single-token MoE scratch,
// the verify pair, and the SwiGLU pair. Both workers queue every launch,
// meet at a barrier, then sync and compare against serial references.
struct GroupFixture {
  hipStream_t stream;
  int set;
  const void* w1;
  const float* x1;
  float* y1;
  int k1;
  int n1;
  const float* r1;
  const void* w2;
  const float* x2;
  float* y2;
  int k2;
  int n2;
  int m2;
  const float* r2;
  const void* gate;
  const void* up;
  const void* down;
  const float* x3;
  const int* ids;
  const float* wts;
  float* y3;
  int k3;
  int n_ff;
  const float* r3;
  const void* w4;
  const float* x4;
  float* y4;
  int k4;
  int n4;
  int mv;
  const float* r4;
  float* sw_g;
  float* sw_u;
  unsigned char mark_g;
  unsigned char mark_u;
};

struct GroupArgs {
  GroupFixture* f;
  int qtype;
  int steps;
  AbortBarrier* barrier;
  std::atomic<int>* mismatches;
  std::atomic<bool>* stop;
};

#define SQ10_GSTEP(call, msg)                        \
  do {                                               \
    if ((call) != SUPERROC_FP4_OK) {                 \
      std::fprintf(stderr, "FAIL groups %s\n", msg); \
      a->mismatches->fetch_add(1);                   \
      a->barrier->trip(*a->stop);                    \
      return;                                        \
    }                                                \
  } while (0)

void GroupWorker(GroupArgs* a) {
  GroupFixture* f = a->f;
  std::vector<float> h1(static_cast<std::size_t>(f->n1));
  std::vector<float> h2(static_cast<std::size_t>(f->m2) *
                        static_cast<std::size_t>(f->n2));
  std::vector<float> h3(static_cast<std::size_t>(f->k3));
  std::vector<float> h4(static_cast<std::size_t>(f->mv) *
                        static_cast<std::size_t>(f->n4));
  for (int t = 0; t < a->steps && !a->stop->load(); ++t) {
    SQ10_GSTEP(superroc_fp4_quant_q8(f->x1, f->k1, SUPERROC_FP4_Q8_SLOT_GENERIC,
                                     f->stream),
               "quant");
    SQ10_GSTEP(
        superroc_fp4_gemv_fp4_q8(a->qtype, f->w1, SUPERROC_FP4_Q8_SLOT_GENERIC,
                                 f->y1, f->k1, f->n1, f->stream),
        "gemv");
    SQ10_GSTEP(superroc_fp4_gemv_m(a->qtype, f->w2, f->x2, f->y2, f->k2, f->n2,
                                   f->m2, f->stream),
               "gemv_m");
    SQ10_GSTEP(superroc_fp4_expert_ffn_dev(a->qtype, f->gate, f->up, f->down,
                                           f->x3, f->ids, f->wts, 1, 2, f->k3,
                                           f->n_ff, 0.0f, f->y3, f->stream),
               "expert");
    SQ10_GSTEP(superroc_fp4_gemv_verify_m(a->qtype, f->w4, f->x4, f->y4, f->k4,
                                          f->n4, f->mv, f->stream),
               "verify");
    if (hipMemsetAsync(f->sw_g, f->mark_g, 64 * sizeof(float), f->stream) !=
            hipSuccess ||
        hipMemsetAsync(f->sw_u, f->mark_u, 64 * sizeof(float), f->stream) !=
            hipSuccess) {
      std::fprintf(stderr, "FAIL groups swiglu memset\n");
      a->mismatches->fetch_add(1);
      a->barrier->trip(*a->stop);
      return;
    }
    // Every launch on both streams is queued before either side reads.
    if (!a->barrier->wait(*a->stop))
      return;
    if (hipStreamSynchronize(f->stream) != hipSuccess) {
      std::fprintf(stderr, "FAIL groups sync\n");
      a->mismatches->fetch_add(1);
      a->barrier->trip(*a->stop);
      return;
    }
    bool ok = hipMemcpy(h1.data(), f->y1,
                        static_cast<std::size_t>(f->n1) * sizeof(float),
                        hipMemcpyDeviceToHost) == hipSuccess &&
              hipMemcpy(h2.data(), f->y2, h2.size() * sizeof(float),
                        hipMemcpyDeviceToHost) == hipSuccess &&
              hipMemcpy(h3.data(), f->y3,
                        static_cast<std::size_t>(f->k3) * sizeof(float),
                        hipMemcpyDeviceToHost) == hipSuccess &&
              hipMemcpy(h4.data(), f->y4, h4.size() * sizeof(float),
                        hipMemcpyDeviceToHost) == hipSuccess &&
              std::memcmp(h1.data(), f->r1, h1.size() * sizeof(float)) == 0 &&
              std::memcmp(h2.data(), f->r2, h2.size() * sizeof(float)) == 0 &&
              std::memcmp(h3.data(), f->r3, h3.size() * sizeof(float)) == 0 &&
              std::memcmp(h4.data(), f->r4, h4.size() * sizeof(float)) == 0 &&
              superroc_fp4_moe_dev_check_set(f->set) == SUPERROC_FP4_OK;
    unsigned char sg = 0;
    unsigned char su = 0;
    ok = ok &&
         hipMemcpy(&sg, f->sw_g, 1, hipMemcpyDeviceToHost) == hipSuccess &&
         hipMemcpy(&su, f->sw_u, 1, hipMemcpyDeviceToHost) == hipSuccess &&
         sg == f->mark_g && su == f->mark_u;
    if (!ok) {
      const int count = a->mismatches->fetch_add(1);
      if (count < 4) {
        std::fprintf(stderr, "FAIL groups mismatch step %d\n", t);
      }
      a->barrier->trip(*a->stop);
      return;
    }
    if (!a->barrier->wait(*a->stop))
      return;
  }
}

int RunInterleaveGroups() {
  const int qtype = SUPERROC_FP4_Q4_0_ROCMFP4_FAST;
  const int k1 = 1024;
  const int n1 = 128;
  const int k2 = 512;
  const int n2 = 128;
  const int m2 = 4;
  const int k3 = 512;
  const int n_ff = 256;
  const int k4 = 512;
  const int n4 = 128;
  const int mv = 4;
  SQ10_EXPECT(superroc_fp4_verify_m_reserve(mv * k4) == SUPERROC_FP4_OK,
              "reserve");
  SQ10_EXPECT(superroc::hip::RocmFp4SwigluScratchWarm(512), "swiglu warm");
  hipStream_t s0 = nullptr;
  hipStream_t s1 = nullptr;
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
  const int set0 = superroc_fp4_slot_stream_attach(s0);
  const int set1 = superroc_fp4_slot_stream_attach(s1);
  SQ10_EXPECT(set0 >= 0 && set1 >= 0 && set0 != set1, "groups attach");
  float *g0 = nullptr, *u0 = nullptr, *g1 = nullptr, *u1 = nullptr;
  SQ10_EXPECT(superroc::hip::RocmFp4SwigluScratch(512, g0, u0, s0), "swiglu 0");
  SQ10_EXPECT(superroc::hip::RocmFp4SwigluScratch(512, g1, u1, s1), "swiglu 1");
  SQ10_EXPECT(g0 != g1 && u0 != u1, "swiglu pairs differ");

  GroupFixture f[2];
  std::vector<float> refs[2][4];
  std::srand(777);
  for (int i = 0; i < 2; ++i) {
    GroupFixture* p = &f[i];
    p->stream = i == 0 ? s0 : s1;
    p->set = i == 0 ? set0 : set1;
    p->k1 = k1;
    p->n1 = n1;
    p->k2 = k2;
    p->n2 = n2;
    p->m2 = m2;
    p->k3 = k3;
    p->n_ff = n_ff;
    p->k4 = k4;
    p->n4 = n4;
    p->mv = mv;
    p->sw_g = i == 0 ? g0 : g1;
    p->sw_u = i == 0 ? u0 : u1;
    p->mark_g = static_cast<unsigned char>(0xa0 + i);
    p->mark_u = static_cast<unsigned char>(0xb0 + i);
    const std::size_t rb1 = superroc_fp4_row_bytes(qtype, k1);
    const std::size_t rb2 = superroc_fp4_row_bytes(qtype, k2);
    const std::size_t rb3 = superroc_fp4_row_bytes(qtype, k3);
    const std::size_t rb3f = superroc_fp4_row_bytes(qtype, n_ff);
    const std::size_t rb4 = superroc_fp4_row_bytes(qtype, k4);
    std::vector<std::uint8_t> w1(static_cast<std::size_t>(n1) * rb1);
    std::vector<std::uint8_t> w2(static_cast<std::size_t>(n2) * rb2);
    std::vector<std::uint8_t> gate(static_cast<std::size_t>(n_ff) * rb3);
    std::vector<std::uint8_t> upw(static_cast<std::size_t>(n_ff) * rb3);
    std::vector<std::uint8_t> downw(static_cast<std::size_t>(k3) * rb3f);
    std::vector<std::uint8_t> w4(static_cast<std::size_t>(n4) * rb4);
    std::vector<float> hx1(static_cast<std::size_t>(k1));
    std::vector<float> hx2(static_cast<std::size_t>(m2) *
                           static_cast<std::size_t>(k2));
    std::vector<float> hx3(static_cast<std::size_t>(k3));
    std::vector<float> hx4(static_cast<std::size_t>(mv) *
                           static_cast<std::size_t>(k4));
    for (auto& v : {&w1, &w2, &gate, &upw, &downw, &w4}) {
      for (std::size_t j = 0; j < v->size(); ++j) {
        (*v)[j] = static_cast<std::uint8_t>(std::rand() >> 7);
      }
    }
    for (auto* v : {&hx1, &hx2, &hx3, &hx4}) {
      for (std::size_t j = 0; j < v->size(); ++j) {
        (*v)[j] = static_cast<float>(std::rand()) /
                      static_cast<float>(RAND_MAX) * 2.0f -
                  1.0f;
      }
    }
    void* dw1 = nullptr;
    void* dw2 = nullptr;
    void* dgate = nullptr;
    void* dup = nullptr;
    void* ddown = nullptr;
    void* dw4 = nullptr;
    float* dx1 = nullptr;
    float* dx2 = nullptr;
    float* dx3 = nullptr;
    float* dx4 = nullptr;
    float* dy1 = nullptr;
    float* dy2 = nullptr;
    float* dy3 = nullptr;
    float* dy4 = nullptr;
    int* dids = nullptr;
    float* dwts = nullptr;
    SQ10_CHECK_HIP(hipMalloc(&dw1, w1.size()));
    SQ10_CHECK_HIP(hipMalloc(&dw2, w2.size()));
    SQ10_CHECK_HIP(hipMalloc(&dgate, gate.size()));
    SQ10_CHECK_HIP(hipMalloc(&dup, upw.size()));
    SQ10_CHECK_HIP(hipMalloc(&ddown, downw.size()));
    SQ10_CHECK_HIP(hipMalloc(&dw4, w4.size()));
    SQ10_CHECK_HIP(hipMalloc(&dx1, hx1.size() * sizeof(float)));
    SQ10_CHECK_HIP(hipMalloc(&dx2, hx2.size() * sizeof(float)));
    SQ10_CHECK_HIP(hipMalloc(&dx3, hx3.size() * sizeof(float)));
    SQ10_CHECK_HIP(hipMalloc(&dx4, hx4.size() * sizeof(float)));
    SQ10_CHECK_HIP(
        hipMalloc(&dy1, static_cast<std::size_t>(n1) * sizeof(float)));
    SQ10_CHECK_HIP(
        hipMalloc(&dy2, static_cast<std::size_t>(m2 * n2) * sizeof(float)));
    SQ10_CHECK_HIP(
        hipMalloc(&dy3, static_cast<std::size_t>(k3) * sizeof(float)));
    SQ10_CHECK_HIP(
        hipMalloc(&dy4, static_cast<std::size_t>(mv * n4) * sizeof(float)));
    SQ10_CHECK_HIP(hipMalloc(&dids, sizeof(int)));
    SQ10_CHECK_HIP(hipMalloc(&dwts, sizeof(float)));
    SQ10_CHECK_HIP(hipMemcpy(dw1, w1.data(), w1.size(), hipMemcpyHostToDevice));
    SQ10_CHECK_HIP(hipMemcpy(dw2, w2.data(), w2.size(), hipMemcpyHostToDevice));
    SQ10_CHECK_HIP(
        hipMemcpy(dgate, gate.data(), gate.size(), hipMemcpyHostToDevice));
    SQ10_CHECK_HIP(
        hipMemcpy(dup, upw.data(), upw.size(), hipMemcpyHostToDevice));
    SQ10_CHECK_HIP(
        hipMemcpy(ddown, downw.data(), downw.size(), hipMemcpyHostToDevice));
    SQ10_CHECK_HIP(hipMemcpy(dw4, w4.data(), w4.size(), hipMemcpyHostToDevice));
    SQ10_CHECK_HIP(hipMemcpy(dx1, hx1.data(), hx1.size() * sizeof(float),
                             hipMemcpyHostToDevice));
    SQ10_CHECK_HIP(hipMemcpy(dx2, hx2.data(), hx2.size() * sizeof(float),
                             hipMemcpyHostToDevice));
    SQ10_CHECK_HIP(hipMemcpy(dx3, hx3.data(), hx3.size() * sizeof(float),
                             hipMemcpyHostToDevice));
    SQ10_CHECK_HIP(hipMemcpy(dx4, hx4.data(), hx4.size() * sizeof(float),
                             hipMemcpyHostToDevice));
    const int zero_id = 0;
    const float one_w = 1.0f;
    SQ10_CHECK_HIP(
        hipMemcpy(dids, &zero_id, sizeof(int), hipMemcpyHostToDevice));
    SQ10_CHECK_HIP(
        hipMemcpy(dwts, &one_w, sizeof(float), hipMemcpyHostToDevice));
    p->w1 = dw1;
    p->x1 = dx1;
    p->y1 = dy1;
    p->w2 = dw2;
    p->x2 = dx2;
    p->y2 = dy2;
    p->gate = dgate;
    p->up = dup;
    p->down = ddown;
    p->x3 = dx3;
    p->ids = dids;
    p->wts = dwts;
    p->y3 = dy3;
    p->w4 = dw4;
    p->x4 = dx4;
    p->y4 = dy4;
    // Serial references on s0, synced per op: correct on every tree.
    refs[i][0].resize(static_cast<std::size_t>(n1));
    refs[i][1].resize(static_cast<std::size_t>(m2 * n2));
    refs[i][2].resize(static_cast<std::size_t>(k3));
    refs[i][3].resize(static_cast<std::size_t>(mv * n4));
    SQ10_EXPECT(superroc_fp4_quant_q8(dx1, k1, SUPERROC_FP4_Q8_SLOT_GENERIC,
                                      s0) == SUPERROC_FP4_OK,
                "groups ref quant");
    SQ10_EXPECT(
        superroc_fp4_gemv_fp4_q8(qtype, dw1, SUPERROC_FP4_Q8_SLOT_GENERIC, dy1,
                                 k1, n1, s0) == SUPERROC_FP4_OK,
        "groups ref gemv");
    SQ10_EXPECT(superroc_fp4_gemv_m(qtype, dw2, dx2, dy2, k2, n2, m2, s0) ==
                    SUPERROC_FP4_OK,
                "groups ref gemv_m");
    SQ10_EXPECT(superroc_fp4_expert_ffn_dev(qtype, dgate, dup, ddown, dx3, dids,
                                            dwts, 1, 2, k3, n_ff, 0.0f, dy3,
                                            s0) == SUPERROC_FP4_OK,
                "groups ref expert");
    SQ10_EXPECT(superroc_fp4_gemv_verify_m(qtype, dw4, dx4, dy4, k4, n4, mv,
                                           s0) == SUPERROC_FP4_OK,
                "groups ref verify");
    SQ10_CHECK_HIP(hipStreamSynchronize(s0));
    SQ10_CHECK_HIP(hipMemcpy(refs[i][0].data(), dy1,
                             refs[i][0].size() * sizeof(float),
                             hipMemcpyDeviceToHost));
    SQ10_CHECK_HIP(hipMemcpy(refs[i][1].data(), dy2,
                             refs[i][1].size() * sizeof(float),
                             hipMemcpyDeviceToHost));
    SQ10_CHECK_HIP(hipMemcpy(refs[i][2].data(), dy3,
                             refs[i][2].size() * sizeof(float),
                             hipMemcpyDeviceToHost));
    SQ10_CHECK_HIP(hipMemcpy(refs[i][3].data(), dy4,
                             refs[i][3].size() * sizeof(float),
                             hipMemcpyDeviceToHost));
    p->r1 = refs[i][0].data();
    p->r2 = refs[i][1].data();
    p->r3 = refs[i][2].data();
    p->r4 = refs[i][3].data();
  }
  if (std::memcmp(refs[0][0].data(), refs[1][0].data(),
                  refs[0][0].size() * sizeof(float)) == 0) {
    std::fprintf(stderr, "FAIL groups refs identical (inputs must differ)\n");
    return 1;
  }
  AbortBarrier barrier(2);
  std::atomic<int> mismatches(0);
  std::atomic<bool> stop(false);
  GroupArgs a0{&f[0], qtype, 2000, &barrier, &mismatches, &stop};
  GroupArgs a1{&f[1], qtype, 2000, &barrier, &mismatches, &stop};
  std::thread t0(GroupWorker, &a0);
  std::thread t1(GroupWorker, &a1);
  t0.join();
  t1.join();
  SQ10_EXPECT(mismatches.load() == 0, "groups bit-identical");
  std::printf(
      "GROUPS PASS (2 streams x 2000 steps x 5 groups, bit-identical)\n");
  SQ10_CHECK_HIP(hipStreamSynchronize(s0));
  SQ10_CHECK_HIP(hipStreamSynchronize(s1));
  // SPEC3 (S3-4): the SwiGLU grow path. Both sets were pre-warmed to 512;
  // asking s0's set for 2048 must sync, malloc under the sets mutex, free
  // the old pair and hand back a working new one, leaving s1's pair alone.
  float* gg = nullptr;
  float* gu = nullptr;
  SQ10_EXPECT(superroc::hip::RocmFp4SwigluScratch(2048, gg, gu, s0),
              "groups grow");
  SQ10_EXPECT(gg != nullptr && gu != nullptr, "groups grown pair");
  SQ10_EXPECT(gg != g0 && gu != u0, "groups grow reallocates");
  SQ10_CHECK_HIP(hipMemset(gg, 0xc0, 2048 * sizeof(float)));
  SQ10_CHECK_HIP(hipMemset(gu, 0xc1, 2048 * sizeof(float)));
  SQ10_CHECK_HIP(hipMemset(g1, 0xd0, 512 * sizeof(float)));
  SQ10_CHECK_HIP(hipMemset(u1, 0xd1, 512 * sizeof(float)));
  unsigned char b0g = 0, b0u = 0, b1g = 0, b1u = 0;
  SQ10_CHECK_HIP(hipMemcpy(&b0g, gg, 1, hipMemcpyDeviceToHost));
  SQ10_CHECK_HIP(hipMemcpy(&b0u, gu, 1, hipMemcpyDeviceToHost));
  SQ10_CHECK_HIP(hipMemcpy(&b1g, g1, 1, hipMemcpyDeviceToHost));
  SQ10_CHECK_HIP(hipMemcpy(&b1u, u1, 1, hipMemcpyDeviceToHost));
  SQ10_EXPECT(b0g == 0xc0 && b0u == 0xc1 && b1g == 0xd0 && b1u == 0xd1,
              "groups grown pair works, other pair intact");
  std::printf("GROUPS_GROW PASS (512 -> 2048 reallocates; pairs isolated)\n");
#ifdef SUPERROC_FP4_SLOT_HAS_ALIAS
  // SPEC3 (S3-4): alias + groups. s1 drops its set and aliases to s0's, and
  // both sides run the four FP4 groups in ordered serial phases (the
  // production-with-drains shape: each side syncs before the other
  // launches), bit-identical to the serial references. The SwiGLU pair is
  // one buffer per set, so two aliased streams cannot hold different marks
  // in it; the FP4 groups are the separable check.
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(s1) == SUPERROC_FP4_OK,
              "alias-groups detach s1");
  SQ10_EXPECT(superroc_fp4_slot_stream_alias(s1, s0) == set0,
              "alias-groups alias");
  std::vector<float> ah1(static_cast<std::size_t>(n1));
  std::vector<float> ah2(static_cast<std::size_t>(m2) *
                         static_cast<std::size_t>(n2));
  std::vector<float> ah3(static_cast<std::size_t>(k3));
  std::vector<float> ah4(static_cast<std::size_t>(mv) *
                         static_cast<std::size_t>(n4));
  for (int t = 0; t < 200; ++t) {
    for (int i = 0; i < 2; ++i) {
      GroupFixture* p = (i == 0) ? &f[0] : &f[1];
      hipStream_t st = (i == 0) ? s0 : s1;
      SQ10_EXPECT(
          superroc_fp4_quant_q8(p->x1, p->k1, SUPERROC_FP4_Q8_SLOT_GENERIC,
                                st) == SUPERROC_FP4_OK,
          "alias-groups quant");
      SQ10_EXPECT(
          superroc_fp4_gemv_fp4_q8(qtype, p->w1, SUPERROC_FP4_Q8_SLOT_GENERIC,
                                   p->y1, p->k1, p->n1, st) == SUPERROC_FP4_OK,
          "alias-groups gemv");
      SQ10_EXPECT(superroc_fp4_gemv_m(qtype, p->w2, p->x2, p->y2, p->k2, p->n2,
                                      p->m2, st) == SUPERROC_FP4_OK,
                  "alias-groups gemv_m");
      SQ10_EXPECT(superroc_fp4_expert_ffn_dev(
                      qtype, p->gate, p->up, p->down, p->x3, p->ids, p->wts, 1,
                      2, p->k3, p->n_ff, 0.0f, p->y3, st) == SUPERROC_FP4_OK,
                  "alias-groups expert");
      SQ10_EXPECT(
          superroc_fp4_gemv_verify_m(qtype, p->w4, p->x4, p->y4, p->k4, p->n4,
                                     p->mv, st) == SUPERROC_FP4_OK,
          "alias-groups verify");
      SQ10_CHECK_HIP(hipStreamSynchronize(st));
      SQ10_CHECK_HIP(hipMemcpy(ah1.data(), p->y1, ah1.size() * sizeof(float),
                               hipMemcpyDeviceToHost));
      SQ10_CHECK_HIP(hipMemcpy(ah2.data(), p->y2, ah2.size() * sizeof(float),
                               hipMemcpyDeviceToHost));
      SQ10_CHECK_HIP(hipMemcpy(ah3.data(), p->y3, ah3.size() * sizeof(float),
                               hipMemcpyDeviceToHost));
      SQ10_CHECK_HIP(hipMemcpy(ah4.data(), p->y4, ah4.size() * sizeof(float),
                               hipMemcpyDeviceToHost));
      SQ10_EXPECT(
          std::memcmp(ah1.data(), p->r1, ah1.size() * sizeof(float)) == 0 &&
              std::memcmp(ah2.data(), p->r2, ah2.size() * sizeof(float)) == 0 &&
              std::memcmp(ah3.data(), p->r3, ah3.size() * sizeof(float)) == 0 &&
              std::memcmp(ah4.data(), p->r4, ah4.size() * sizeof(float)) == 0,
          "alias-groups bit-identical");
      SQ10_EXPECT(superroc_fp4_moe_dev_check_set(set0) == SUPERROC_FP4_OK,
                  "alias-groups check");
    }
  }
  std::printf(
      "GROUPS_ALIAS PASS (200 ordered steps x 4 groups x 2 sides, "
      "bit-identical)\n");
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(s1) == SUPERROC_FP4_OK,
              "alias-groups detach alias");
  const int set1b = superroc_fp4_slot_stream_attach(s1);
  SQ10_EXPECT(set1b >= 0 && set1b != set0, "alias-groups re-attach s1");
#endif
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(s0) == SUPERROC_FP4_OK,
              "detach s0");
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(s1) == SUPERROC_FP4_OK,
              "detach s1");
  SQ10_CHECK_HIP(hipStreamDestroy(s0));
  SQ10_CHECK_HIP(hipStreamDestroy(s1));
  for (int i = 0; i < 2; ++i) {
    SQ10_CHECK_HIP(hipFree(const_cast<void*>(f[i].w1)));
    SQ10_CHECK_HIP(hipFree(const_cast<float*>(f[i].x1)));
    SQ10_CHECK_HIP(hipFree(f[i].y1));
    SQ10_CHECK_HIP(hipFree(const_cast<void*>(f[i].w2)));
    SQ10_CHECK_HIP(hipFree(const_cast<float*>(f[i].x2)));
    SQ10_CHECK_HIP(hipFree(f[i].y2));
    SQ10_CHECK_HIP(hipFree(const_cast<void*>(f[i].gate)));
    SQ10_CHECK_HIP(hipFree(const_cast<void*>(f[i].up)));
    SQ10_CHECK_HIP(hipFree(const_cast<void*>(f[i].down)));
    SQ10_CHECK_HIP(hipFree(const_cast<float*>(f[i].x3)));
    SQ10_CHECK_HIP(hipFree(const_cast<int*>(f[i].ids)));
    SQ10_CHECK_HIP(hipFree(const_cast<float*>(f[i].wts)));
    SQ10_CHECK_HIP(hipFree(f[i].y3));
    SQ10_CHECK_HIP(hipFree(const_cast<void*>(f[i].w4)));
    SQ10_CHECK_HIP(hipFree(const_cast<float*>(f[i].x4)));
    SQ10_CHECK_HIP(hipFree(f[i].y4));
  }
  return 0;
}

// SPEC2 (R6): is the MoE status-word read stale? A bad expert id is safe
// (the kernel clamps to row 0 and writes SUPERROC_FP4_ERR_ARG to the word), so
// force one on a non-blocking stream and check at once, 200 times. This is
// main's behaviour, not the patch's: the case reports CONFIRMED or REFUTED
// and never fails -- the box decides whether the smallest fix ships.
int RunStaleStatus() {
  const int qtype = SUPERROC_FP4_Q4_0_ROCMFP4_FAST;
  const int k = 512;
  const int n_ff = 256;
  const int n_expert = 2;
  hipStream_t s = nullptr;
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  const int set = superroc_fp4_slot_stream_attach(s);
  SQ10_EXPECT(set >= 0, "stale attach");
  const std::size_t rb_k = superroc_fp4_row_bytes(qtype, k);
  const std::size_t rb_ff = superroc_fp4_row_bytes(qtype, n_ff);
  std::vector<std::uint8_t> gate(static_cast<std::size_t>(n_ff) * rb_k, 0x22);
  std::vector<std::uint8_t> up(static_cast<std::size_t>(n_ff) * rb_k, 0x33);
  std::vector<std::uint8_t> down(static_cast<std::size_t>(k) * rb_ff, 0x44);
  std::vector<float> hx(static_cast<std::size_t>(k), 0.5f);
  void *d_gate = nullptr, *d_up = nullptr, *d_down = nullptr;
  float *dx = nullptr, *dy = nullptr;
  int* dids = nullptr;
  float* dwts = nullptr;
  SQ10_CHECK_HIP(hipMalloc(&d_gate, gate.size()));
  SQ10_CHECK_HIP(hipMalloc(&d_up, up.size()));
  SQ10_CHECK_HIP(hipMalloc(&d_down, down.size()));
  SQ10_CHECK_HIP(hipMalloc(&dx, hx.size() * sizeof(float)));
  SQ10_CHECK_HIP(hipMalloc(&dy, hx.size() * sizeof(float)));
  SQ10_CHECK_HIP(hipMalloc(&dids, sizeof(int)));
  SQ10_CHECK_HIP(hipMalloc(&dwts, sizeof(float)));
  SQ10_CHECK_HIP(
      hipMemcpy(d_gate, gate.data(), gate.size(), hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(hipMemcpy(d_up, up.data(), up.size(), hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(
      hipMemcpy(d_down, down.data(), down.size(), hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(hipMemcpy(dx, hx.data(), hx.size() * sizeof(float),
                           hipMemcpyHostToDevice));
  const int good_id = 0;
  const float one_w = 1.0f;
  SQ10_CHECK_HIP(hipMemcpy(dids, &good_id, sizeof(int), hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(hipMemcpy(dwts, &one_w, sizeof(float), hipMemcpyHostToDevice));
  // Settle (allocations sync) and clear the word.
  SQ10_EXPECT(superroc_fp4_expert_ffn_dev(qtype, d_gate, d_up, d_down, dx, dids,
                                          dwts, 1, n_expert, k, n_ff, 0.0f, dy,
                                          s) == SUPERROC_FP4_OK,
              "stale settle");
  SQ10_CHECK_HIP(hipStreamSynchronize(s));
  SQ10_EXPECT(superroc_fp4_moe_dev_check_set(set) == SUPERROC_FP4_OK,
              "stale clear");
  // Force the bad id and check at once, with no stream sync between.
  const int bad_id = n_expert;
  SQ10_CHECK_HIP(hipMemcpy(dids, &bad_id, sizeof(int), hipMemcpyHostToDevice));
  int reported = 0;
  int missed = 0;
  for (int t = 0; t < 200; ++t) {
    if (superroc_fp4_expert_ffn_dev(qtype, d_gate, d_up, d_down, dx, dids, dwts,
                                    1, n_expert, k, n_ff, 0.0f, dy,
                                    s) != SUPERROC_FP4_OK) {
      std::fprintf(stderr, "FAIL stale launch %d\n", t);
      return 1;
    }
    if (superroc_fp4_moe_dev_check_set(set) != SUPERROC_FP4_OK) {
      ++reported;
    } else {
      ++missed;
    }
  }
  SQ10_CHECK_HIP(hipStreamSynchronize(s));
  // Non-vacuous: the word really is written by a bad id.
  SQ10_EXPECT(superroc_fp4_moe_dev_check_set(set) != SUPERROC_FP4_OK,
              "stale final reports (else vacuous)");
  // The bare loop only reports: its race window is timing-dependent by nature;
  // STALE_ORDERED below is the gate.
  std::printf("STALE_STATUS reported=%d missed=%d/200 verdict=%s\n", reported,
              missed, missed > 0 ? "CONFIRMED" : "REFUTED");
#ifdef SUPERROC_FP4_SLOT_HAS_STREAM_CHECK
  // SPEC2 (R6): the stream-ordered check must never miss. This loop is the
  // failing test for the fix (the bare loop above misses on main).
  int fixed_missed = 0;
  for (int t = 0; t < 200; ++t) {
    if (superroc_fp4_expert_ffn_dev(qtype, d_gate, d_up, d_down, dx, dids, dwts,
                                    1, n_expert, k, n_ff, 0.0f, dy,
                                    s) != SUPERROC_FP4_OK) {
      std::fprintf(stderr, "FAIL stale-ordered launch %d\n", t);
      return 1;
    }
    if (superroc_fp4_moe_dev_check_set_on_stream(set, s) != SUPERROC_FP4_OK) {
      ++reported;
    } else {
      ++fixed_missed;
    }
  }
  SQ10_EXPECT(fixed_missed == 0, "ordered check never misses");
  std::printf("STALE_ORDERED missed=%d/200\n", fixed_missed);
#endif
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(s) == SUPERROC_FP4_OK,
              "stale detach");
  SQ10_CHECK_HIP(hipStreamDestroy(s));
  SQ10_CHECK_HIP(hipFree(d_gate));
  SQ10_CHECK_HIP(hipFree(d_up));
  SQ10_CHECK_HIP(hipFree(d_down));
  SQ10_CHECK_HIP(hipFree(dx));
  SQ10_CHECK_HIP(hipFree(dy));
  SQ10_CHECK_HIP(hipFree(dids));
  SQ10_CHECK_HIP(hipFree(dwts));
  return 0;
}

#ifdef SUPERROC_FP4_SLOT_HAS_ALIAS
// SPEC3 (S3-1): the alias race, then the ordered pattern. An aliased helper
// and its owner share one slot set with no ordering edge between their
// streams. Queue the helper's quant + a long memset stall + the helper's
// gemv, then launch the owner's quant+gemv at once over the same slots,
// both streams really in flight: without a drain the owner's quant lands in
// the shared slot while the helper's gemv is still queued, and the helper
// reads the wrong activation. With drain == false this case shows the
// unordered pattern FAILING (it fails if the window never opens); with
// drain == true it runs the S3-1 production pattern (the drafter entry
// drains its stream before the session's next launch) and must stay
// bit-identical to the serial reference. An event (queried, never waited
// on) times the owner's launch into the helper's stall; the streams
// themselves are never ordered.
int RunAliasOrder(bool drain) {
  SQ10_EXPECT(superroc_fp4_slot_sets_configure(2) == SUPERROC_FP4_OK,
              "alias-order configure");
  const int k = 2048;
  const int n = 256;
  const int qtype = SUPERROC_FP4_Q4_0_ROCMFP4_FAST;
  const std::size_t rb = superroc_fp4_row_bytes(qtype, k);
  SQ10_EXPECT(rb != 0, "alias-order row_bytes");
  std::vector<std::uint8_t> w(static_cast<std::size_t>(n) * rb);
  for (std::size_t i = 0; i < w.size(); ++i) {
    w[i] = static_cast<std::uint8_t>((i * 2654435761u) >> 13);
  }
  std::vector<float> hxh(static_cast<std::size_t>(k));
  std::vector<float> hxo(static_cast<std::size_t>(k));
  std::srand(31337);
  for (int i = 0; i < k; ++i) {
    hxh[static_cast<std::size_t>(i)] =
        static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 4.0f -
        2.0f;
    hxo[static_cast<std::size_t>(i)] =
        static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 4.0f -
        2.0f;
  }
  hipStream_t o = nullptr;
  hipStream_t h = nullptr;
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&o, hipStreamNonBlocking));
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&h, hipStreamNonBlocking));
  const int set = superroc_fp4_slot_stream_attach(o);
  SQ10_EXPECT(set >= 0, "alias-order owner attach");
  SQ10_EXPECT(superroc_fp4_slot_stream_alias(h, o) == set, "alias-order alias");
  void* dw = nullptr;
  float* dxh = nullptr;
  float* dxo = nullptr;
  float* dyh = nullptr;
  float* dyo = nullptr;
  void* delay = nullptr;
  static constexpr std::size_t kDelayBytes = std::size_t{512} << 20;
  SQ10_CHECK_HIP(hipMalloc(&dw, static_cast<std::size_t>(n) * rb));
  SQ10_CHECK_HIP(hipMalloc(&dxh, static_cast<std::size_t>(k) * sizeof(float)));
  SQ10_CHECK_HIP(hipMalloc(&dxo, static_cast<std::size_t>(k) * sizeof(float)));
  SQ10_CHECK_HIP(hipMalloc(&dyh, static_cast<std::size_t>(n) * sizeof(float)));
  SQ10_CHECK_HIP(hipMalloc(&dyo, static_cast<std::size_t>(n) * sizeof(float)));
  SQ10_CHECK_HIP(hipMalloc(&delay, kDelayBytes));
  SQ10_CHECK_HIP(hipMemcpy(dw, w.data(), static_cast<std::size_t>(n) * rb,
                           hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(hipMemcpy(dxh, hxh.data(),
                           static_cast<std::size_t>(k) * sizeof(float),
                           hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(hipMemcpy(dxo, hxo.data(),
                           static_cast<std::size_t>(k) * sizeof(float),
                           hipMemcpyHostToDevice));
  // Serial references on the owner stream, synced: correct on every tree.
  std::vector<float> ref_h(static_cast<std::size_t>(n));
  std::vector<float> ref_o(static_cast<std::size_t>(n));
  SQ10_EXPECT(superroc_fp4_quant_q8(dxh, k, SUPERROC_FP4_Q8_SLOT_GENERIC, o) ==
                  SUPERROC_FP4_OK,
              "alias-order ref h quant");
  SQ10_EXPECT(superroc_fp4_gemv_fp4_q8(qtype, dw, SUPERROC_FP4_Q8_SLOT_GENERIC,
                                       dyh, k, n, o) == SUPERROC_FP4_OK,
              "alias-order ref h gemv");
  SQ10_CHECK_HIP(hipStreamSynchronize(o));
  SQ10_CHECK_HIP(hipMemcpy(ref_h.data(), dyh,
                           static_cast<std::size_t>(n) * sizeof(float),
                           hipMemcpyDeviceToHost));
  SQ10_EXPECT(superroc_fp4_quant_q8(dxo, k, SUPERROC_FP4_Q8_SLOT_GENERIC, o) ==
                  SUPERROC_FP4_OK,
              "alias-order ref o quant");
  SQ10_EXPECT(superroc_fp4_gemv_fp4_q8(qtype, dw, SUPERROC_FP4_Q8_SLOT_GENERIC,
                                       dyo, k, n, o) == SUPERROC_FP4_OK,
              "alias-order ref o gemv");
  SQ10_CHECK_HIP(hipStreamSynchronize(o));
  SQ10_CHECK_HIP(hipMemcpy(ref_o.data(), dyo,
                           static_cast<std::size_t>(n) * sizeof(float),
                           hipMemcpyDeviceToHost));
  SQ10_EXPECT(std::memcmp(ref_h.data(), ref_o.data(),
                          static_cast<std::size_t>(n) * sizeof(float)) != 0,
              "alias-order refs differ (else vacuous)");
  hipEvent_t ev = nullptr;
  SQ10_CHECK_HIP(hipEventCreate(&ev));
  int yh_mm = 0;
  int yo_mm = 0;
  std::vector<float> got(static_cast<std::size_t>(n));
  for (int t = 0; t < 200; ++t) {
    SQ10_EXPECT(superroc_fp4_quant_q8(dxh, k, SUPERROC_FP4_Q8_SLOT_GENERIC,
                                      h) == SUPERROC_FP4_OK,
                "alias-order h quant");
    for (int m = 0; m < 4; ++m) {
      SQ10_CHECK_HIP(hipMemsetAsync(delay, t + m, kDelayBytes, h));
      if (m == 0) {
        SQ10_CHECK_HIP(hipEventRecord(ev, h));
      }
    }
    SQ10_EXPECT(
        superroc_fp4_gemv_fp4_q8(qtype, dw, SUPERROC_FP4_Q8_SLOT_GENERIC, dyh,
                                 k, n, h) == SUPERROC_FP4_OK,
        "alias-order h gemv");
    if (drain) {
      SQ10_CHECK_HIP(hipStreamSynchronize(h));
    } else {
      while (hipEventQuery(ev) != hipSuccess) {
      }
    }
    SQ10_EXPECT(superroc_fp4_quant_q8(dxo, k, SUPERROC_FP4_Q8_SLOT_GENERIC,
                                      o) == SUPERROC_FP4_OK,
                "alias-order o quant");
    SQ10_EXPECT(
        superroc_fp4_gemv_fp4_q8(qtype, dw, SUPERROC_FP4_Q8_SLOT_GENERIC, dyo,
                                 k, n, o) == SUPERROC_FP4_OK,
        "alias-order o gemv");
    SQ10_CHECK_HIP(hipStreamSynchronize(o));
    SQ10_CHECK_HIP(hipStreamSynchronize(h));
    SQ10_CHECK_HIP(hipMemcpy(got.data(), dyh,
                             static_cast<std::size_t>(n) * sizeof(float),
                             hipMemcpyDeviceToHost));
    if (std::memcmp(got.data(), ref_h.data(),
                    static_cast<std::size_t>(n) * sizeof(float)) != 0) {
      ++yh_mm;
    }
    SQ10_CHECK_HIP(hipMemcpy(got.data(), dyo,
                             static_cast<std::size_t>(n) * sizeof(float),
                             hipMemcpyDeviceToHost));
    if (std::memcmp(got.data(), ref_o.data(),
                    static_cast<std::size_t>(n) * sizeof(float)) != 0) {
      ++yo_mm;
    }
  }
  SQ10_CHECK_HIP(hipEventDestroy(ev));
  if (drain) {
    SQ10_EXPECT(yh_mm == 0 && yo_mm == 0, "ordered alias stays bit-identical");
    std::printf("ALIAS_ORDERED yh_mm=%d/200 yo_mm=%d/200\n", yh_mm, yo_mm);
  } else {
    SQ10_EXPECT(yh_mm > 0, "unordered alias must corrupt (window shut?)");
    std::printf("ALIAS_RACE yh_mm=%d/200 yo_mm=%d/200\n", yh_mm, yo_mm);
  }
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(h) == SUPERROC_FP4_OK,
              "alias-order detach helper");
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(o) == SUPERROC_FP4_OK,
              "alias-order detach owner");
  SQ10_CHECK_HIP(hipStreamDestroy(o));
  SQ10_CHECK_HIP(hipStreamDestroy(h));
  SQ10_CHECK_HIP(hipFree(dw));
  SQ10_CHECK_HIP(hipFree(dxh));
  SQ10_CHECK_HIP(hipFree(dxo));
  SQ10_CHECK_HIP(hipFree(dyh));
  SQ10_CHECK_HIP(hipFree(dyo));
  SQ10_CHECK_HIP(hipFree(delay));
  return 0;
}
#endif  // SUPERROC_FP4_SLOT_HAS_ALIAS

// SPEC3 (S3-2): the prefill-path stale read. superroc_fp4_expert_ffn_m runs on
// the non-blocking arena stream; the bare m_check reads the status word on
// the default stream with no drain, so it can precede the kernels and miss
// a bad-router-id error on a prefill row -- the same stale class R6 fixed
// on decode. Force a bad id on row 1 of 2 and check at once, 200 times.
int RunStaleMStatus() {
  const int qtype = SUPERROC_FP4_Q4_0_ROCMFP4_FAST;
  const int k = 512;
  const int n_ff = 256;
  const int n_expert = 2;
  const int m = 2;
  const int n_used = 1;
  hipStream_t s = nullptr;
  SQ10_CHECK_HIP(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  const int set = superroc_fp4_slot_stream_attach(s);
  SQ10_EXPECT(set >= 0, "mstale attach");
  const std::size_t rb_k = superroc_fp4_row_bytes(qtype, k);
  const std::size_t rb_ff = superroc_fp4_row_bytes(qtype, n_ff);
  std::vector<std::uint8_t> gate(static_cast<std::size_t>(n_ff) * rb_k, 0x22);
  std::vector<std::uint8_t> up(static_cast<std::size_t>(n_ff) * rb_k, 0x33);
  std::vector<std::uint8_t> down(static_cast<std::size_t>(k) * rb_ff, 0x44);
  std::vector<float> hx(
      static_cast<std::size_t>(m) * static_cast<std::size_t>(k), 0.5f);
  void *d_gate = nullptr, *d_up = nullptr, *d_down = nullptr;
  float *dx = nullptr, *dy = nullptr;
  int* dids = nullptr;
  float* dwts = nullptr;
  SQ10_CHECK_HIP(hipMalloc(&d_gate, gate.size()));
  SQ10_CHECK_HIP(hipMalloc(&d_up, up.size()));
  SQ10_CHECK_HIP(hipMalloc(&d_down, down.size()));
  SQ10_CHECK_HIP(hipMalloc(&dx, hx.size() * sizeof(float)));
  SQ10_CHECK_HIP(hipMalloc(&dy, hx.size() * sizeof(float)));
  SQ10_CHECK_HIP(hipMalloc(&dids, static_cast<std::size_t>(m) * sizeof(int)));
  SQ10_CHECK_HIP(hipMalloc(&dwts, static_cast<std::size_t>(m) * sizeof(float)));
  SQ10_CHECK_HIP(
      hipMemcpy(d_gate, gate.data(), gate.size(), hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(hipMemcpy(d_up, up.data(), up.size(), hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(
      hipMemcpy(d_down, down.data(), down.size(), hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(hipMemcpy(dx, hx.data(), hx.size() * sizeof(float),
                           hipMemcpyHostToDevice));
  const int good[2] = {0, 0};
  const float one_w[2] = {1.0f, 1.0f};
  SQ10_CHECK_HIP(hipMemcpy(dids, good, sizeof(good), hipMemcpyHostToDevice));
  SQ10_CHECK_HIP(hipMemcpy(dwts, one_w, sizeof(one_w), hipMemcpyHostToDevice));
  // Settle (allocations sync) and clear the word.
  SQ10_EXPECT(superroc_fp4_expert_ffn_m(qtype, d_gate, d_up, d_down, dx, dids,
                                        dwts, m, n_used, n_expert, k, n_ff,
                                        0.0f, dy, s) == SUPERROC_FP4_OK,
              "mstale settle");
  SQ10_CHECK_HIP(hipStreamSynchronize(s));
  SQ10_EXPECT(superroc_fp4_expert_ffn_m_check_set(set) == SUPERROC_FP4_OK,
              "mstale clear");
  // Force the bad id on row 1 and check at once, with no stream sync between.
  const int bad_id = n_expert;
  SQ10_CHECK_HIP(
      hipMemcpy(dids + 1, &bad_id, sizeof(int), hipMemcpyHostToDevice));
  int reported = 0;
  int missed = 0;
  for (int t = 0; t < 200; ++t) {
    if (superroc_fp4_expert_ffn_m(qtype, d_gate, d_up, d_down, dx, dids, dwts,
                                  m, n_used, n_expert, k, n_ff, 0.0f, dy,
                                  s) != SUPERROC_FP4_OK) {
      std::fprintf(stderr, "FAIL mstale launch %d\n", t);
      return 1;
    }
    if (superroc_fp4_expert_ffn_m_check_set(set) != SUPERROC_FP4_OK) {
      ++reported;
    } else {
      ++missed;
    }
  }
  SQ10_CHECK_HIP(hipStreamSynchronize(s));
  // Non-vacuous: the word really is written by a bad id on a prefill row.
  SQ10_EXPECT(superroc_fp4_expert_ffn_m_check_set(set) != SUPERROC_FP4_OK,
              "mstale final reports (else vacuous)");
  // The bare loop only reports: its race window is timing-dependent by nature;
  // MSTALE_ORDERED below is the gate.
  std::printf("MSTALE_STATUS reported=%d missed=%d/200 verdict=%s\n", reported,
              missed, missed > 0 ? "CONFIRMED" : "REFUTED");
#ifdef SUPERROC_FP4_SLOT_HAS_STREAM_MCHECK
  int fixed_missed = 0;
  for (int t = 0; t < 200; ++t) {
    if (superroc_fp4_expert_ffn_m(qtype, d_gate, d_up, d_down, dx, dids, dwts,
                                  m, n_used, n_expert, k, n_ff, 0.0f, dy,
                                  s) != SUPERROC_FP4_OK) {
      std::fprintf(stderr, "FAIL mstale-ordered launch %d\n", t);
      return 1;
    }
    if (superroc_fp4_expert_ffn_m_check_set_on_stream(set, s) !=
        SUPERROC_FP4_OK) {
      ++reported;
    } else {
      ++fixed_missed;
    }
  }
  SQ10_EXPECT(fixed_missed == 0, "ordered m-check never misses");
  std::printf("MSTALE_ORDERED missed=%d/200\n", fixed_missed);
#endif
  SQ10_EXPECT(superroc_fp4_slot_stream_detach(s) == SUPERROC_FP4_OK,
              "mstale detach");
  SQ10_CHECK_HIP(hipStreamDestroy(s));
  SQ10_CHECK_HIP(hipFree(d_gate));
  SQ10_CHECK_HIP(hipFree(d_up));
  SQ10_CHECK_HIP(hipFree(d_down));
  SQ10_CHECK_HIP(hipFree(dx));
  SQ10_CHECK_HIP(hipFree(dy));
  SQ10_CHECK_HIP(hipFree(dids));
  SQ10_CHECK_HIP(hipFree(dwts));
  return 0;
}
#endif  // SUPERROC_FP4_SLOT_OWNER

}  // namespace

int main(int argc, char** argv) {
  const bool only_production =
      argc > 1 && std::strcmp(argv[1], "production") == 0;
  const bool skip_production = argc > 1 && std::strcmp(argv[1], "suite") == 0;
#ifdef SUPERROC_FP4_SLOT_OWNER
  if (!skip_production && RunProductionHelper() != 0) {
    std::fprintf(stderr, "PRODUCTION FAIL\n");
    return 1;
  }
  if (only_production) {
    return g_failures == 0 ? 0 : 1;
  }
  if (RunInterleave() != 0) {
    std::fprintf(stderr, "INTERLEAVE FAIL\n");
    return 1;
  }
  if (RunInterleaveGroups() != 0) {
    std::fprintf(stderr, "GROUPS FAIL\n");
    return 1;
  }
  if (RunExhaustion() != 0) {
    std::fprintf(stderr, "EXHAUSTION FAIL\n");
    return 1;
  }
#ifdef SUPERROC_FP4_SLOT_HAS_ALIAS
  if (RunAlias() != 0) {
    std::fprintf(stderr, "ALIAS FAIL\n");
    return 1;
  }
  if (RunLifecycle() != 0) {
    std::fprintf(stderr, "LIFECYCLE FAIL\n");
    return 1;
  }
#else
  std::printf("SKIP alias/lifecycle (sq10-1: no alias API)\n");
#endif
  if (RunZeroAlloc() != 0) {
    std::fprintf(stderr, "ZEROALLOC FAIL\n");
    return 1;
  }
  if (RunStaleStatus() != 0) {
    std::fprintf(stderr, "STALE_STATUS FAIL\n");
    return 1;
  }
  if (RunStaleMStatus() != 0) {
    std::fprintf(stderr, "STALE_MSTATUS FAIL\n");
    return 1;
  }
#ifdef SUPERROC_FP4_SLOT_HAS_ALIAS
  if (RunAliasOrder(false) != 0) {
    std::fprintf(stderr, "ALIAS_RACE FAIL\n");
    return 1;
  }
  if (RunAliasOrder(true) != 0) {
    std::fprintf(stderr, "ALIAS_ORDERED FAIL\n");
    return 1;
  }
#else
  std::printf("SKIP alias-order cases (sq10-1: no alias API)\n");
#endif
#else
  (void)only_production;
  (void)skip_production;
  if (RunInterleave() != 0) {
    std::fprintf(stderr, "INTERLEAVE FAIL\n");
    return 1;
  }
  std::printf("SKIP ownership cases (1194fea: no ownership API)\n");
#endif
  std::printf("ALL PASS\n");
  return g_failures == 0 ? 0 : 1;
}
