#include "src/models/qwen/moe/qwen_moe_prefill.hpp"

#include "src/core/rocm_fp4/superroc_fp4_gemm.hpp"
#include "src/core/rocm_fp4/superroc_fp4_moe_group.hpp"
#include "src/core/rocm_fp4/superroc_fp4_quant.hpp"
#include "src/core/rocm_fp4/superroc_fp4_wmma.hpp"

#include <algorithm>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include "src/core/env_compat.hpp"

namespace {

constexpr int kThreads = 256;
// 1024 * top-8 = 8192 slots, the largest shape superroc_fp4_moe_group_batch runs
// on its parallel kernel (slots <= 8192 and dynamic smem <= 48 KB).
// KPF: default for the SUPERROC_KAT_SUB_TOKENS knob (1024 | 2048 | 4096);
// larger sub-batches run the grouper's serial kernel.
constexpr int kSubTokens = 4096;  // KPF2: 32k grouper handles slots <= 32768.
constexpr int kGroupCap = 256;
constexpr int kActTileBytes = 576;
constexpr int kMaxTokens = 4096;
constexpr int kMaxUsed = 8;
constexpr int kMaxExpert = 256;

// i in [0, n). Reads g[i], u[i] and writes h[i] only. n is the host-checked
// product slots * n_ff (or n_tokens * n_ff for the dense shared expert).
__global__ void KatSwigluKernel(const float* __restrict__ g,
                                const float* __restrict__ u,
                                float* __restrict__ h, int n) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    h[i] = superroc_fp4_swiglu_elem(g[i], u[i], 0.0F);
  }
}

// i in [0, n_tokens * hidden). y[i] *= sig[i / hidden]. sig has n_tokens
// entries. hidden > 0 is checked on the host.
__global__ void KatScaleRowsKernel(float* __restrict__ y,
                                   const float* __restrict__ sig, int n_tokens,
                                   int hidden) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int n = n_tokens * hidden;
  if (i < n) {
    y[i] *= sig[i / hidden];
  }
}

int KatLaunchSwiglu(const float* g, const float* u, float* h, int n,
                    hipStream_t stream) {
  if (g == nullptr || u == nullptr || h == nullptr || n <= 0) {
    return SUPERROC_FP4_ERR_ARG;
  }
  const int blocks = (n + kThreads - 1) / kThreads;
  KatSwigluKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(g, u,
                                                                           h, n);
  return hipGetLastError() == hipSuccess ? SUPERROC_FP4_OK : SUPERROC_FP4_ERR_HIP;
}

int KatLaunchScale(float* y, const float* sig, int n_tokens, int hidden,
                   hipStream_t stream) {
  if (y == nullptr || sig == nullptr || n_tokens <= 0 || hidden <= 0) {
    return SUPERROC_FP4_ERR_ARG;
  }
  const long long n = static_cast<long long>(n_tokens) * hidden;
  if (n > 0x7fffffffLL) {
    return SUPERROC_FP4_ERR_ARG;
  }
  const int blocks =
      (static_cast<int>(n) + kThreads - 1) / kThreads;
  KatScaleRowsKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      y, sig, n_tokens, hidden);
  return hipGetLastError() == hipSuccess ? SUPERROC_FP4_OK : SUPERROC_FP4_ERR_HIP;
}

// superroc_fp4_wmma_gemm_dense launches BN=64 blocks, and each block reads four
// 16-token activation tiles. Round the buffer up so that last block stays
// inside the allocation. The epilogue writes only tok < m.
int DenseQ8Tiles(int m) {
  return ((m + 63) / 64) * 4;
}

int GroupQ8Tiles(int slots, int n_expert) {
  return slots / 16 + slots / kGroupCap + n_expert + 2;
}

std::size_t Q8Bytes(int tiles, int k) {
  return static_cast<std::size_t>(std::max(tiles, 1)) *
         static_cast<std::size_t>(std::max(k / 32, 1)) * kActTileBytes;
}

// KPF SPEC1 step 2: routed sub-batch token knob, read once. 1024 | 2048 |
// 4096; anything else (including unset) falls back to kSubTokens with one
// stderr line. Larger sub-batches make slots = sub * n_used exceed the
// parallel grouper's 8192-slot cap; superroc_fp4_moe_group_batch then runs its
// serial kernel (its oracle, no cap), so no grouper change is needed.
int KatSubTokens() {
  static const int sub = [] {
    const char* e = ::superroc::core::GetEnvCompat("SUPERROC_KAT_SUB_TOKENS");
    if (e != nullptr && (std::strcmp(e, "1024") == 0 ||
                         std::strcmp(e, "2048") == 0 ||
                         std::strcmp(e, "4096") == 0)) {
      return std::atoi(e);
    }
    std::fprintf(stderr, "KAT sub-tokens rejected '%s', using %d\n",
                 e != nullptr ? e : "(unset)", kSubTokens);
    return kSubTokens;
  }();
  return sub;
}

// KPF SPEC1 step 3: when SUPERROC_KAT_MOE_DRAIN=0, skip the Drain at the top of
// LaunchKatMoeRoutedGrouped (the one inside Ensure on reallocation stays).
// Default (unset or anything but "0") keeps the drain.
bool KatMoeDrain() {
  static const bool drain = [] {
    const char* e = ::superroc::core::GetEnvCompat("SUPERROC_KAT_MOE_DRAIN");
    return e == nullptr || std::strcmp(e, "0") != 0;
  }();
  return drain;
}

// KPF SPEC1 step 4: sync attribution behind KAT_SYNC_PROF=1. drain counts
// Drain() calls (one hipStreamSynchronize each), order counts OrderEvent()
// calls, ensure counts KatBuf (re)allocations (the Ensure realloc path),
// other counts the OnNull hipEventRecord + hipStreamWaitEvent ops. Syncs
// inside superroc_fp4_* callees and KatPrefillProf::Take live outside this file
// and are not counted here; the rocprofv3 runtime-trace reports true totals.
// The one summary line goes to the same stdout as KAT_PREFILL_PROF, printed
// once at process exit (prompt-profile runs are a single chunk).
struct KatSyncProf {
  int chunk = 0;
  long drain = 0, order = 0, ensure = 0, other = 0;
};

KatSyncProf& KatSync() {
  static KatSyncProf s;
  return s;
}

void KatSyncProfDump() {
  const KatSyncProf& c = KatSync();
  std::printf(
      "KAT_SYNC_PROF chunk=%d drain=%ld order=%ld ensure=%ld other=%ld\n",
      c.chunk, c.drain, c.order, c.ensure, c.other);
  std::fflush(stdout);
}

bool KatSyncProfOn() {
  static const bool on = [] {
    const char* e = std::getenv("KAT_SYNC_PROF");
    const bool v = e != nullptr && e[0] == '1' && e[1] == '\0';
    if (v) std::atexit(KatSyncProfDump);
    return v;
  }();
  return on;
}

struct KatBuf {
  void* p = nullptr;
  std::size_t bytes = 0;

  int Need(std::size_t n) {
    if (p != nullptr && bytes >= n) {
      return SUPERROC_FP4_OK;
    }
    if (n > 0 && KatSyncProfOn()) KatSync().ensure++;
    if (p != nullptr) {
      hipFree(p);
      p = nullptr;
      bytes = 0;
    }
    if (n == 0) {
      return SUPERROC_FP4_OK;
    }
    if (hipMalloc(&p, n) != hipSuccess) {
      p = nullptr;
      bytes = 0;
      return SUPERROC_FP4_ERR_HIP;
    }
    bytes = n;
    return SUPERROC_FP4_OK;
  }

  template <typename T>
  T* As() {
    return static_cast<T*>(p);
  }
};

struct KatScratch {
  hipEvent_t order = nullptr;
  KatBuf slot_tok, slot_w, token_slots, slot_dest;
  KatBuf grp_e, grp_off, grp_cnt, grp_zp, grp_zb;
  KatBuf hdr;  // KPF2: device header (4 ints) for batch32k; zeroed once per chunk.
  KatBuf gate_e, up_e, hs, down_e;
  KatBuf q8_e, q8_h;

  int OrderEvent() {
    if (KatSyncProfOn()) KatSync().order++;
    if (order != nullptr) {
      return SUPERROC_FP4_OK;
    }
    if (hipEventCreateWithFlags(&order, hipEventDisableTiming) != hipSuccess) {
      order = nullptr;
      return SUPERROC_FP4_ERR_HIP;
    }
    return SUPERROC_FP4_OK;
  }

  // Sized for one routed sub-batch (SUPERROC_KAT_SUB_TOKENS tokens, default
  // 1024) and for a dense shared expert over the whole chunk
  // (n_tokens <= 4096). Each buffer tracks its own size.
  int Ensure(int n_tokens, int n_used, int n_expert, int hidden, int n_ff) {
    const int sub = std::min(n_tokens, KatSubTokens());
    const int slots = sub * n_used;
    const int groups = n_expert + slots / kGroupCap + 2;
    const int gate_n = std::max(slots * n_ff, n_tokens * n_ff);
    const int down_n = slots * hidden;
    const int tiles =
        std::max(GroupQ8Tiles(slots, n_expert), DenseQ8Tiles(n_tokens));
    int rc = OrderEvent();
    if (rc != SUPERROC_FP4_OK) return rc;
    const auto need = [&](KatBuf& b, std::size_t n) {
      return rc == SUPERROC_FP4_OK ? (rc = b.Need(n)) : rc;
    };
    need(slot_tok, static_cast<std::size_t>(slots) * sizeof(int));
    need(slot_w, static_cast<std::size_t>(slots) * sizeof(float));
    need(token_slots, static_cast<std::size_t>(slots) * sizeof(int));
    need(slot_dest, static_cast<std::size_t>(slots) * sizeof(int));
    need(grp_e, static_cast<std::size_t>(groups) * sizeof(int));
    need(grp_off, static_cast<std::size_t>(groups) * sizeof(int));
    need(grp_cnt, static_cast<std::size_t>(groups) * sizeof(int));
    need(grp_zp, static_cast<std::size_t>(groups + 1) * sizeof(int));
    need(grp_zb, static_cast<std::size_t>(groups + 1) * sizeof(int));
    need(hdr, static_cast<std::size_t>(SUPERROC_FP4_LS1_HDR_WORDS) * sizeof(int));
    need(gate_e, static_cast<std::size_t>(gate_n) * sizeof(float));
    need(up_e, static_cast<std::size_t>(gate_n) * sizeof(float));
    need(hs, static_cast<std::size_t>(gate_n) * sizeof(float));
    need(down_e, static_cast<std::size_t>(down_n) * sizeof(float));
    need(q8_e, Q8Bytes(tiles, hidden));
    need(q8_h, Q8Bytes(tiles, n_ff));
    return rc;
  }
};

KatScratch& Scratch() {
  static KatScratch s;
  return s;
}

int Drain(hipStream_t stream) {
  if (KatSyncProfOn()) KatSync().drain++;
  if (hipStreamSynchronize(stream) != hipSuccess) {
    return SUPERROC_FP4_ERR_HIP;
  }
  return SUPERROC_FP4_OK;
}

template <typename F>
int OnNull(hipStream_t stream, F&& body) {
  int rc = Drain(stream);
  if (rc != SUPERROC_FP4_OK) return rc;
  rc = body();
  if (rc != SUPERROC_FP4_OK) return rc;
  hipEvent_t order = Scratch().order;
  if (hipEventRecord(order, nullptr) != hipSuccess) return SUPERROC_FP4_ERR_HIP;
  if (KatSyncProfOn()) KatSync().other++;
  if (hipStreamWaitEvent(stream, order, 0) != hipSuccess) {
    return SUPERROC_FP4_ERR_HIP;
  }
  if (KatSyncProfOn()) KatSync().other++;
  return SUPERROC_FP4_OK;
}

int CheckShape(int qtype, int n_tokens, int hidden, int n_ff) {
  if (qtype != SUPERROC_FP4_Q4_0_ROCMFP4 && qtype != SUPERROC_FP4_Q4_0_ROCMFP4_FAST) {
    return SUPERROC_FP4_ERR_TYPE;
  }
  if (n_tokens <= 0 || n_tokens > kMaxTokens || hidden <= 0 || n_ff <= 0) {
    return SUPERROC_FP4_ERR_ARG;
  }
  if ((hidden % 32) != 0 || (n_ff % 32) != 0) {
    return SUPERROC_FP4_ERR_ARG;
  }
  if (superroc_fp4_row_bytes(qtype, hidden) == 0 ||
      superroc_fp4_row_bytes(qtype, n_ff) == 0) {
    return SUPERROC_FP4_ERR_TYPE;
  }
  return SUPERROC_FP4_OK;
}

// MG64: 64-token grouped GEMM gate, read once. SUPERROC_FP4_GRP64 zero
// keeps the 16-token path; unset or anything else takes the 64-token
// path when the shape gate (average >= 32 tokens per group) passes.
bool Grp64Enabled() {
  static const bool on = [] {
    const char* e = ::superroc::core::GetEnvCompat("SUPERROC_FP4_GRP64");
    return e == nullptr || std::strcmp(e, "0") != 0;
  }();
  return on;
}

int RoutedSub(int qtype, const void* gate, const void* up, const void* down,
              const float* x, const int* ids, const float* weights, int m,
              int n_used, int n_expert, int hidden, int n_ff, float* y,
              hipStream_t stream) {
  KatScratch& s = Scratch();
  const int slots = m * n_used;
  const int grp_capacity = n_expert + slots / kGroupCap + 2;
  if (s.slot_tok.bytes < static_cast<std::size_t>(slots) * sizeof(int) ||
      s.grp_e.bytes < static_cast<std::size_t>(grp_capacity) * sizeof(int)) {
    return SUPERROC_FP4_ERR_ARG;
  }
  const std::size_t stride_k =
      static_cast<std::size_t>(n_ff) * superroc_fp4_row_bytes(qtype, hidden);
  const std::size_t stride_ff =
      static_cast<std::size_t>(hidden) * superroc_fp4_row_bytes(qtype, n_ff);
  // KPF2: the header lives on the device now (s.hdr, 4 ints, zeroed once per
  // chunk by the caller). No D2H here: n_groups/total_tiles come out of
  // hdr_dev inside the GEMM kernels, which launch over the host-known bound
  // max_tiles = GroupQ8Tiles(slots, n_expert) (>= the true tile count, the
  // same bound Ensure sized the q8 scratch with).
  int * hdr_dev = s.hdr.As<int>();
  const int max_tiles = GroupQ8Tiles(slots, n_expert);
  int* slot_tok = s.slot_tok.As<int>();
  float* slot_w = s.slot_w.As<float>();
  int* token_slots = s.token_slots.As<int>();
  int* slot_dest = s.slot_dest.As<int>();
  int* grp_e = s.grp_e.As<int>();
  int* grp_off = s.grp_off.As<int>();
  int* grp_cnt = s.grp_cnt.As<int>();
  int* grp_zp = s.grp_zp.As<int>();
  int* grp_zb = s.grp_zb.As<int>();
  float* gate_e = s.gate_e.As<float>();
  float* up_e = s.up_e.As<float>();
  float* hs = s.hs.As<float>();
  float* down_e = s.down_e.As<float>();
  // Host-side audits from host-known bounds only (the true header stays on
  // the device; STATUS is checked once per chunk by the caller). grp arrays
  // hold grp_capacity groups by construction; the q8 scratch holds max_tiles
  // by Ensure.
  if (Q8Bytes(max_tiles, hidden) > s.q8_e.bytes ||
      Q8Bytes(max_tiles, n_ff) > s.q8_h.bytes) {
    std::fprintf(stderr, "KAT grouped tiles rejected slots=%d tokens=%d\n",
                 slots, m);
    return SUPERROC_FP4_ERR_ARG;
  }
  int rc = OnNull(stream, [&] {
    return superroc_fp4_moe_group_batch32k(
        ids, weights, nullptr, m, n_used, n_used, n_expert, kGroupCap, slot_tok,
        slot_w, token_slots, grp_e, grp_off, grp_cnt, grp_zp, slot_dest,
        grp_capacity, slots, hdr_dev);
  });
  if (rc != SUPERROC_FP4_OK) return rc;
  const int hs_n = slots * n_ff;
  if (s.gate_e.bytes < static_cast<std::size_t>(hs_n) * sizeof(float) ||
      s.down_e.bytes < static_cast<std::size_t>(slots) * hidden * sizeof(float)) {
    return SUPERROC_FP4_ERR_ARG;
  }
  // MG64: 64-token blocks when the average group holds >= 32 tokens. KPF2:
  // the true tiles/groups need the header D2H this lane removes, so the
  // gate runs on the host-known bounds (both GEMM paths are
  // bit-identical; this only moves the crossover). max_tiles >=
  // 2 * grp_capacity first holds at slots >= 4404 for 256 experts.
  const bool use64 = Grp64Enabled() && max_tiles >= 2 * grp_capacity;
  if (use64) {
    rc = OnNull(stream, [&] {
      return superroc_fp4_moe_group_blocks_dev(grp_cnt, hdr_dev, 64, grp_zb, stream);
    });
    if (rc != SUPERROC_FP4_OK) return rc;
  }
  rc = OnNull(stream, [&] {
    if (superroc_fp4_wmma_quant(x, slot_tok, slot_dest, s.q8_e.p, slots, hidden) !=
        SUPERROC_FP4_OK) {
      return SUPERROC_FP4_ERR_HIP;
    }
    // KAT grouped prefill asks for BK=4 on this call. Ling's gemm_grp
    // still reads SUPERROC_FP4_MOE_V2 and stays at BK=2 when that env is unset.
    if (use64) {
      if (superroc_fp4_wmma_gemm_grp64_dev(qtype, gate, stride_k, grp_e, grp_off, grp_cnt,
                                grp_zp, hdr_dev, max_tiles, s.q8_e.p, gate_e,
                                nullptr, hidden, n_ff, grp_zb) != SUPERROC_FP4_OK) {
        return SUPERROC_FP4_ERR_HIP;
      }
    } else if (superroc_fp4_wmma_gemm_grp_bk_dev(qtype, gate, stride_k, grp_e, grp_off, grp_cnt,
                                      grp_zp, hdr_dev, max_tiles, s.q8_e.p, gate_e,
                                      nullptr, hidden, n_ff, /*bk=*/4) != SUPERROC_FP4_OK) {
      return SUPERROC_FP4_ERR_HIP;
    }
    if (use64) {
      if (superroc_fp4_wmma_gemm_grp64_dev(qtype, up, stride_k, grp_e, grp_off, grp_cnt,
                                grp_zp, hdr_dev, max_tiles, s.q8_e.p, up_e,
                                nullptr, hidden, n_ff, grp_zb) != SUPERROC_FP4_OK) {
        return SUPERROC_FP4_ERR_HIP;
      }
    } else if (superroc_fp4_wmma_gemm_grp_bk_dev(qtype, up, stride_k, grp_e, grp_off, grp_cnt,
                                      grp_zp, hdr_dev, max_tiles, s.q8_e.p, up_e,
                                      nullptr, hidden, n_ff, /*bk=*/4) != SUPERROC_FP4_OK) {
      return SUPERROC_FP4_ERR_HIP;
    }
    return SUPERROC_FP4_OK;
  });
  if (rc != SUPERROC_FP4_OK) return rc;
  rc = KatLaunchSwiglu(gate_e, up_e, hs, hs_n, stream);
  if (rc != SUPERROC_FP4_OK) return rc;
  rc = OnNull(stream, [&] {
    if (superroc_fp4_wmma_quant(hs, nullptr, slot_dest, s.q8_h.p, slots, n_ff) !=
        SUPERROC_FP4_OK) {
      return SUPERROC_FP4_ERR_HIP;
    }
    if (use64) {
      if (superroc_fp4_wmma_gemm_grp64_dev(qtype, down, stride_ff, grp_e, grp_off, grp_cnt,
                                grp_zp, hdr_dev, max_tiles, s.q8_h.p, down_e,
                                slot_w, n_ff, hidden, grp_zb) != SUPERROC_FP4_OK) {
        return SUPERROC_FP4_ERR_HIP;
      }
    } else if (superroc_fp4_wmma_gemm_grp_bk_dev(qtype, down, stride_ff, grp_e, grp_off, grp_cnt,
                                      grp_zp, hdr_dev, max_tiles, s.q8_h.p, down_e,
                                      slot_w, n_ff, hidden, /*bk=*/4) != SUPERROC_FP4_OK) {
      return SUPERROC_FP4_ERR_HIP;
    }
    if (superroc_fp4_slot_gather_sum(down_e, token_slots, y, n_used, hidden, m) !=
        SUPERROC_FP4_OK) {
      return SUPERROC_FP4_ERR_HIP;
    }
    return SUPERROC_FP4_OK;
  });
  return rc;
}

}  // namespace

int LaunchKatMoeRoutedGrouped(int qtype, const void* gate, const void* up,
                              const void* down, const float* x, const int* ids,
                              const float* weights, int n_tokens, int n_used,
                              int n_expert, int hidden, int n_ff, float* y,
                              hipStream_t stream) {
  if (gate == nullptr || up == nullptr || down == nullptr || x == nullptr ||
      ids == nullptr || weights == nullptr || y == nullptr) {
    return SUPERROC_FP4_ERR_ARG;
  }
  int rc = CheckShape(qtype, n_tokens, hidden, n_ff);
  if (rc != SUPERROC_FP4_OK) return rc;
  if (n_used <= 0 || n_used > kMaxUsed || n_expert <= 0 ||
      n_expert > kMaxExpert) {
    return SUPERROC_FP4_ERR_ARG;
  }
  if (static_cast<long long>(n_tokens) * n_used * n_ff > 0x7fffffffLL) {
    return SUPERROC_FP4_ERR_ARG;
  }
  if (KatSyncProfOn()) KatSync().chunk = n_tokens;
  if (KatMoeDrain()) {
    rc = Drain(stream);
    if (rc != SUPERROC_FP4_OK) return rc;
  }
  rc = Scratch().Ensure(n_tokens, n_used, n_expert, hidden, n_ff);
  if (rc != SUPERROC_FP4_OK) return rc;
  // KPF2: one zeroing of the device header per chunk; batch32k only ever
  // sets STATUS (sticky ERR), so a single D2H after the sub-batch loop sees
  // every sub-batch's status. Synchronous (16 bytes), hence ordered before
  // the default-stream grouper kernels.
  if (hipMemset(Scratch().hdr.As<int>(), 0,
                SUPERROC_FP4_LS1_HDR_WORDS * sizeof(int)) != hipSuccess) {
    return SUPERROC_FP4_ERR_HIP;
  }
  const int sub_tok = KatSubTokens();
  for (int t0 = 0; t0 < n_tokens; t0 += sub_tok) {
    const int m = std::min(sub_tok, n_tokens - t0);
    const std::size_t tok = static_cast<std::size_t>(t0);
    rc = RoutedSub(qtype, gate, up, down, x + tok * hidden,
                   ids + tok * static_cast<std::size_t>(n_used),
                   weights + tok * static_cast<std::size_t>(n_used), m, n_used,
                   n_expert, hidden, n_ff, y + tok * hidden, stream);
    if (rc != SUPERROC_FP4_OK) return rc;
  }
  // KPF2: the single host read of the chunk's header. A non-OK status fails
  // the chunk with the existing message.
  int hdr_host[SUPERROC_FP4_LS1_HDR_WORDS] = {0, 0, 0, 0};
  if (hipMemcpy(hdr_host, Scratch().hdr.As<int>(),
                SUPERROC_FP4_LS1_HDR_WORDS * sizeof(int),
                hipMemcpyDeviceToHost) != hipSuccess) {
    return SUPERROC_FP4_ERR_HIP;
  }
  if (hdr_host[SUPERROC_FP4_LS1_HDR_STATUS] != SUPERROC_FP4_OK) {
    std::fprintf(stderr, "KAT grouped routing rejected status=%d tokens=%d\n",
                 hdr_host[SUPERROC_FP4_LS1_HDR_STATUS], n_tokens);
    return SUPERROC_FP4_ERR_ARG;
  }
  return SUPERROC_FP4_OK;
}

int LaunchKatMoeSharedDense(int qtype, const void* gate, const void* up,
                            const void* down, const float* x, const float* sig,
                            int n_tokens, int hidden, int n_ff, float* y,
                            hipStream_t stream) {
  if (gate == nullptr || up == nullptr || down == nullptr || x == nullptr ||
      sig == nullptr || y == nullptr) {
    return SUPERROC_FP4_ERR_ARG;
  }
  int rc = CheckShape(qtype, n_tokens, hidden, n_ff);
  if (rc != SUPERROC_FP4_OK) return rc;
  if (static_cast<long long>(n_tokens) * n_ff > 0x7fffffffLL) {
    return SUPERROC_FP4_ERR_ARG;
  }
  rc = Drain(stream);
  if (rc != SUPERROC_FP4_OK) return rc;
  // n_used/n_expert only size the routed half of the scratch. The dense
  // path needs the q8 pad and the [n_tokens, n_ff] SwiGLU rows.
  rc = Scratch().Ensure(n_tokens, /*n_used=*/1, /*n_expert=*/1, hidden, n_ff);
  if (rc != SUPERROC_FP4_OK) return rc;
  KatScratch& s = Scratch();
  const int hs_n = n_tokens * n_ff;
  if (s.gate_e.bytes < static_cast<std::size_t>(hs_n) * sizeof(float)) {
    return SUPERROC_FP4_ERR_ARG;
  }
  const int dense_tiles = DenseQ8Tiles(n_tokens);
  if (Q8Bytes(dense_tiles, hidden) > s.q8_e.bytes ||
      Q8Bytes(dense_tiles, n_ff) > s.q8_h.bytes) {
    return SUPERROC_FP4_ERR_ARG;
  }
  float* gate_e = s.gate_e.As<float>();
  float* up_e = s.up_e.As<float>();
  float* hs = s.hs.As<float>();
  rc = OnNull(stream, [&] {
    if (superroc_fp4_wmma_quant(x, nullptr, nullptr, s.q8_e.p, n_tokens, hidden) !=
        SUPERROC_FP4_OK) {
      return SUPERROC_FP4_ERR_HIP;
    }
    if (superroc_fp4_wmma_gemm_dense(qtype, gate, s.q8_e.p, gate_e, hidden, n_ff,
                                 n_tokens) != SUPERROC_FP4_OK) {
      return SUPERROC_FP4_ERR_HIP;
    }
    if (superroc_fp4_wmma_gemm_dense(qtype, up, s.q8_e.p, up_e, hidden, n_ff,
                                 n_tokens) != SUPERROC_FP4_OK) {
      return SUPERROC_FP4_ERR_HIP;
    }
    return SUPERROC_FP4_OK;
  });
  if (rc != SUPERROC_FP4_OK) return rc;
  rc = KatLaunchSwiglu(gate_e, up_e, hs, hs_n, stream);
  if (rc != SUPERROC_FP4_OK) return rc;
  rc = OnNull(stream, [&] {
    if (superroc_fp4_wmma_quant(hs, nullptr, nullptr, s.q8_h.p, n_tokens, n_ff) !=
        SUPERROC_FP4_OK) {
      return SUPERROC_FP4_ERR_HIP;
    }
    if (superroc_fp4_wmma_gemm_dense(qtype, down, s.q8_h.p, y, n_ff, hidden,
                                 n_tokens) != SUPERROC_FP4_OK) {
      return SUPERROC_FP4_ERR_HIP;
    }
    return SUPERROC_FP4_OK;
  });
  if (rc != SUPERROC_FP4_OK) return rc;
  return KatLaunchScale(y, sig, n_tokens, hidden, stream);
}
