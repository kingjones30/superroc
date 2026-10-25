#pragma once
// Host launch API for Ling-3.0-flash quant GEMV/GEMM, the bailingmoe3 router,
// and the clamped SwiGLU expert FFN.
//
// Weight layout is ggml's: ne[0] = K is contiguous and quantized. Row r of a
// matrix is at byte offset r * ling3_row_bytes(qtype, K). An expert tensor
// {K, n_rows, n_expert} stores expert e at byte offset
// e * n_rows * ling3_row_bytes(qtype, K).
//
// Activations and outputs are F32, token-major:
//   x[token * K + k],  y[token * N + n]
//
// Device pointers are the weight / activation / output buffers. ids and
// router weights passed to ling3_expert_ffn are HOST pointers (n_used is 8).
// ling3_route writes DEVICE ids and weights.
//
// Every device allocation of a weight or activation buffer needs
// LING3_TAIL_PAD (64) extra readable bytes after the last element. Quant
// blocks are not padded (the fast FP4 block is 17 bytes) and the kernel
// reads only bytes that belong to the row; the pad covers a compiler-widened
// tail load on the last row.
//
// CPU entry points (ling3_*_cpu) are inline in ling3_cpu.hpp, included below.
// Device entry points are defined in ling3_moe.hip and synchronize before
// returning.

#include "ling3_cpu.hpp"

// LI1: LS2 phase-14 dropped-stream fixes. The entry points below take an
// optional launch stream, defaulted so every existing call site is unchanged.
// hipStream_t is needed only for those declarations; ling3_ops.hpp includes
// this header for the same reason.
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

int ling3_dequant(int qtype, const void* rows, float* y, int k, int n,
                  hipStream_t stream = nullptr);
int ling3_gemv(int qtype, const void* W, const float* x, float* y, int k, int n,
               hipStream_t stream = nullptr);
int ling3_gemm(int qtype, const void* W, const float* x, float* y, int k, int n,
               int m, hipStream_t stream = nullptr);

/// Test-only: force the router's tie-fallback network (PHASE7 item 2b).
void ling3_route_set_force_network(int on);

int ling3_route(const float* x, const float* gate, const float* bias, int k,
                int n_expert, int n_used, int n_groups, int n_group_used,
                float w_scale, int norm_w, int* ids, float* weights,
                hipStream_t stream = nullptr);

int ling3_swiglu(const float* gate, const float* up, float* h, int n,
                 float limit, hipStream_t stream = nullptr);
int ling3_vec_add(const float* a, const float* b, float* y, int n,
                  hipStream_t stream = nullptr);

int ling3_expert_ffn(int qtype, const void* gate, const void* up,
                     const void* down, const float* x, const int* ids,
                     const float* weights, int n_tokens, int n_used,
                     int n_expert, int k, int n_ff, float limit, float* y,
                     hipStream_t stream = nullptr);

int ling3_dense_ffn(int qtype, const void* gate, const void* up,
                    const void* down, const float* x, int n_tokens, int k,
                    int n_ff, float limit, float* y,
                    hipStream_t stream = nullptr);

/// RV25 wave1 F2+F3: ling3_dense_ffn for the n_used == 1, w == 1.0 case (the
/// shared expert of a MoE layer, the whole FFN of a dense layer), decode only
/// (n_tokens == 1), ROCmFP4. Two changes, both bit-identical:
///   xq_ready  1 = `x` is ALREADY quantised into LING3_Q8_SLOT_FFN_X (the
///   resident
///             grouped call did it moments ago to the same buffer) -- skip the
///             repeat.
///   the down projection writes `y` directly with y[i] = 0.0f + acc, which is
///   the exact value of the unfused hipMemset(y,0) + k_axpy(1.0f, p, y).
/// Returns LING3_ERR_ARG (before launching anything) when the fused form does
/// not apply; the caller then falls back to ling3_dense_ffn.
int ling3_dense_ffn_w1(int qtype, const void* gate, const void* up,
                       const void* down, const float* x, int n_tokens, int k,
                       int n_ff, float limit, float* y, int xq_ready,
                       hipStream_t stream = nullptr);

#ifdef __cplusplus
}
#endif
