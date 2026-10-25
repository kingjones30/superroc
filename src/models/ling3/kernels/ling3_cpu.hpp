#pragma once
// Host reference for the launch API. Same row kernel as the device: one
// sequential dot per output, fp-contract off. Not a device function.

#include <stdlib.h>
#include <string.h>

#include "ling3_route.hpp"

static inline int ling3_gemv_cpu(int qtype, const void* W, const float* x,
                                 float* y, int k, int n) {
  size_t rb = ling3_row_bytes(qtype, k);
  if (!W || !x || !y || rb == 0 || n <= 0) {
    return LING3_ERR_ARG;
  }
  const uint8_t* base = (const uint8_t*)W;
  for (int row = 0; row < n; ++row) {
    y[row] = ling3_dot_row(qtype, base + (size_t)row * rb, x, k);
  }
  return LING3_OK;
}

// x: [m, k] token-major. y: [m, n] token-major. W: n rows of k.
static inline int ling3_gemm_cpu(int qtype, const void* W, const float* x,
                                 float* y, int k, int n, int m) {
  if (m <= 0) {
    return LING3_ERR_ARG;
  }
  for (int t = 0; t < m; ++t) {
    int rc = ling3_gemv_cpu(qtype, W, x + (size_t)t * (size_t)k,
                            y + (size_t)t * (size_t)n, k, n);
    if (rc != LING3_OK) {
      return rc;
    }
  }
  return LING3_OK;
}

static inline int ling3_route_cpu(const float* x, const float* gate,
                                  const float* bias, int k, int n_expert,
                                  int n_used, int n_groups, int n_group_used,
                                  float w_scale, int norm_w, int* ids,
                                  float* weights) {
  int nf = ling3_route_nfloat(n_expert, n_groups);
  int ni = ling3_route_nint(n_expert, n_groups);
  if (nf < 0 || ni < 0) {
    return LING3_ERR_ARG;
  }
  float* sf = (float*)malloc((size_t)nf * sizeof(float));
  int* si = (int*)malloc((size_t)ni * sizeof(int));
  if (!sf || !si) {
    free(sf);
    free(si);
    return LING3_ERR_ARG;
  }
  int rc =
      ling3_route_core(x, gate, bias, k, n_expert, n_used, n_groups,
                       n_group_used, w_scale, norm_w, ids, weights, sf, si);
  free(sf);
  free(si);
  return rc;
}

// gate, up: each expert is n_ff rows of k. down: each expert is k rows of n_ff.
// Expert e begins at e * n_rows * row_bytes. ids/weights are host, length
// n_tokens * n_used, token-major. x and y are [n_tokens, k].
// y is the weighted sum of the routed experts (fork: weight after down).
static inline int ling3_expert_ffn_cpu(int qtype, const void* gate,
                                       const void* up, const void* down,
                                       const float* x, const int* ids,
                                       const float* weights, int n_tokens,
                                       int n_used, int n_expert, int k,
                                       int n_ff, float limit, float* y) {
  size_t rb_k = ling3_row_bytes(qtype, k);
  size_t rb_ff = ling3_row_bytes(qtype, n_ff);
  if (!gate || !up || !down || !x || !ids || !weights || !y) {
    return LING3_ERR_ARG;
  }
  if (rb_k == 0 || rb_ff == 0 || n_tokens <= 0 || n_used <= 0 ||
      n_expert <= 0 || n_ff <= 0) {
    return LING3_ERR_ARG;
  }
  float* g = (float*)malloc((size_t)n_ff * sizeof(float));
  float* u = (float*)malloc((size_t)n_ff * sizeof(float));
  float* h = (float*)malloc((size_t)n_ff * sizeof(float));
  float* p = (float*)malloc((size_t)k * sizeof(float));
  if (!g || !u || !h || !p) {
    free(g);
    free(u);
    free(h);
    free(p);
    return LING3_ERR_ARG;
  }
  memset(y, 0, (size_t)n_tokens * (size_t)k * sizeof(float));
  int rc = LING3_OK;
  for (int t = 0; t < n_tokens && rc == LING3_OK; ++t) {
    const float* xt = x + (size_t)t * (size_t)k;
    float* yt = y + (size_t)t * (size_t)k;
    for (int s = 0; s < n_used; ++s) {
      int id = ids[t * n_used + s];
      if (id < 0 || id >= n_expert) {
        rc = LING3_ERR_ARG;
        break;
      }
      float w = weights[t * n_used + s];
      const uint8_t* gb =
          (const uint8_t*)gate + (size_t)id * (size_t)n_ff * rb_k;
      const uint8_t* ub = (const uint8_t*)up + (size_t)id * (size_t)n_ff * rb_k;
      const uint8_t* db = (const uint8_t*)down + (size_t)id * (size_t)k * rb_ff;
      for (int i = 0; i < n_ff; ++i) {
        g[i] = ling3_dot_row(qtype, gb + (size_t)i * rb_k, xt, k);
        u[i] = ling3_dot_row(qtype, ub + (size_t)i * rb_k, xt, k);
        h[i] = ling3_swiglu_elem(g[i], u[i], limit);
      }
      for (int o = 0; o < k; ++o) {
        p[o] = ling3_dot_row(qtype, db + (size_t)o * rb_ff, h, n_ff);
        yt[o] += w * p[o];
      }
    }
  }
  free(g);
  free(u);
  free(h);
  free(p);
  return rc;
}

// Dense blocks 0-1 and the shared expert. One matrix each, no router weight.
// limit is swiglu_clamp_shexp[layer] (0 disables the clamp, including blocks
// 0-1).
static inline int ling3_dense_ffn_cpu(int qtype, const void* gate,
                                      const void* up, const void* down,
                                      const float* x, int n_tokens, int k,
                                      int n_ff, float limit, float* y) {
  if (n_tokens <= 0) {
    return LING3_ERR_ARG;
  }
  int* ids = (int*)malloc((size_t)n_tokens * sizeof(int));
  float* w = (float*)malloc((size_t)n_tokens * sizeof(float));
  if (!ids || !w) {
    free(ids);
    free(w);
    return LING3_ERR_ARG;
  }
  for (int t = 0; t < n_tokens; ++t) {
    ids[t] = 0;
    w[t] = 1.f;
  }
  int rc = ling3_expert_ffn_cpu(qtype, gate, up, down, x, ids, w, n_tokens, 1,
                                1, k, n_ff, limit, y);
  free(ids);
  free(w);
  return rc;
}
