#pragma once
// MoE router for bailingmoe3, matching llm_graph_context::build_moe_ffn
// (llama-graph.cpp) and the fork's GPU argsort (ggml-cuda/argsort.cu bitonic,
// DESC). Selection scores are sigmoid(logits) + bias. Group score is the sum
// of the top-2 selection scores in the group (DeepSeek-V3 / the fork). The
// top n_group_used groups stay; the rest are masked to -inf. Then top
// n_used experts. Weights are the unbiased sigmoids of those experts,
// divided by max(sum, 2^-14) when norm_w is set, then multiplied by w_scale
// when w_scale is not 0 and not 1 (the fork skips both of those).

#include "gufo_fp4_quant.hpp"

GUFO_FP4_HD inline int gufo_fp4_pow2_ge(int n) {
  if (n <= 0 || n > 1024) {
    return -1;
  }
  int p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

// work_a and work_b each have length gufo_fp4_pow2_ge(ncols). dst has length
// ncols. Copied from k_argsort_f32_i32<GGML_SORT_ORDER_DESC>, including the
// padding rule (indices >= ncols sort to the end) and strict < / > (ties do not
// swap).
GUFO_FP4_HD inline int gufo_fp4_bitonic_argsort_desc(const float* x, int* dst,
                                                     int* work_a, int* work_b,
                                                     int ncols) {
  int pad = gufo_fp4_pow2_ge(ncols);
  if (pad < 0 || !x || !dst || !work_a || !work_b) {
    return GUFO_FP4_ERR_ARG;
  }
  for (int i = 0; i < pad; ++i) {
    work_a[i] = i;
  }
  for (int k = 2; k <= pad; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int col = 0; col < pad; ++col) {
        work_b[col] = work_a[col];
      }
      for (int col = 0; col < pad; ++col) {
        int ixj = col ^ j;
        if (ixj > col) {
          int do_swap = 0;
          if ((col & k) == 0) {
            if (work_a[col] >= ncols ||
                (work_a[ixj] < ncols && x[work_a[col]] < x[work_a[ixj]])) {
              do_swap = 1;
            }
          } else {
            if (work_a[ixj] >= ncols ||
                (work_a[col] < ncols && x[work_a[col]] > x[work_a[ixj]])) {
              do_swap = 1;
            }
          }
          if (do_swap) {
            work_b[col] = work_a[ixj];
            work_b[ixj] = work_a[col];
          }
        }
      }
      for (int col = 0; col < pad; ++col) {
        work_a[col] = work_b[col];
      }
    }
  }
  for (int i = 0; i < ncols; ++i) {
    dst[i] = work_a[i];
  }
  return GUFO_FP4_OK;
}

GUFO_FP4_HD inline int gufo_fp4_route_nfloat(int n_expert, int n_groups) {
  if (n_expert <= 0 || n_groups <= 0) {
    return -1;
  }
  return 3 * n_expert + n_groups;
}

GUFO_FP4_HD inline int gufo_fp4_route_nint(int n_expert, int n_groups) {
  int pad_e = gufo_fp4_pow2_ge(n_expert);
  int g = n_groups < 1 ? 1 : n_groups;
  int pad_g = gufo_fp4_pow2_ge(g);
  if (pad_e < 0 || pad_g < 0) {
    return -1;
  }
  return 2 * pad_e + n_expert + 2 * pad_g + g;
}

// scratch_f: gufo_fp4_route_nfloat floats. scratch_i: gufo_fp4_route_nint ints.
// gate is F32, n_expert rows of k (ggml {k, n_expert}). bias may be null.
// ids and weights each have length n_used.
GUFO_FP4_HD inline int gufo_fp4_route_core(
    const float* x, const float* gate, const float* bias, int k, int n_expert,
    int n_used, int n_groups, int n_group_used, float w_scale, int norm_w,
    int* ids, float* weights, float* scratch_f, int* scratch_i,
    int logits_ready = 0) {
  if (!x || !gate || !ids || !weights || !scratch_f || !scratch_i) {
    return GUFO_FP4_ERR_ARG;
  }
  if (k <= 0 || n_expert <= 0 || n_expert > 1024 || n_used <= 0 ||
      n_used > n_expert) {
    return GUFO_FP4_ERR_ARG;
  }
  if (n_groups <= 0 || n_group_used <= 0 || n_group_used > n_groups) {
    return GUFO_FP4_ERR_ARG;
  }
  if ((n_expert % n_groups) != 0) {
    return GUFO_FP4_ERR_ARG;
  }
  int npg = n_expert / n_groups;
  if (n_groups > 1 && npg < 2) {
    return GUFO_FP4_ERR_ARG;
  }
  if (gufo_fp4_row_bytes(GUFO_FP4_F32, k) == 0) {
    return GUFO_FP4_ERR_ARG;
  }

  int pad_e = gufo_fp4_pow2_ge(n_expert);
  int pad_g = gufo_fp4_pow2_ge(n_groups);
  float* logits = scratch_f;
  float* probs = scratch_f + n_expert;
  float* sel = scratch_f + 2 * n_expert;
  float* gscore = scratch_f + 3 * n_expert;
  int* ba = scratch_i;
  int* bb = scratch_i + pad_e;
  int* order = scratch_i + 2 * pad_e;
  int* ga = order + n_expert;
  int* gb = ga + pad_g;
  int* gorder = gb + pad_g;

  size_t rb = (size_t)k * sizeof(float);
  for (int e = 0; e < n_expert; ++e) {
    // Jay 2026-10-01: logits_ready = 1 means a parallel kernel already wrote
    // logits[0, n_expert) (k_route_logits); the single-thread dot over 512 x
    // 2560 was ~0.4 s per call on the GPU, 40 calls per token.
    if (!logits_ready) {
      const float* row = (const float*)((const uint8_t*)gate + (size_t)e * rb);
      logits[e] = gufo_fp4_dot_row(GUFO_FP4_F32, row, x, k);
    }
    probs[e] = 1.0f / (1.0f + expf(-logits[e]));
    float b = bias ? bias[e] : 0.f;
    sel[e] = probs[e] + b;
  }

  if (n_groups > 1) {
    for (int g = 0; g < n_groups; ++g) {
      int rc = gufo_fp4_bitonic_argsort_desc(sel + g * npg, order, ba, bb, npg);
      if (rc != GUFO_FP4_OK) {
        return rc;
      }
      gscore[g] = sel[g * npg + order[0]] + sel[g * npg + order[1]];
    }
    int rc = gufo_fp4_bitonic_argsort_desc(gscore, gorder, ga, gb, n_groups);
    if (rc != GUFO_FP4_OK) {
      return rc;
    }
    for (int g = 0; g < n_groups; ++g) {
      int keep = 0;
      for (int i = 0; i < n_group_used; ++i) {
        if (gorder[i] == g) {
          keep = 1;
        }
      }
      if (!keep) {
        for (int j = 0; j < npg; ++j) {
          sel[g * npg + j] = gufo_fp4_neg_inf();
        }
      }
    }
  }

  int rc = gufo_fp4_bitonic_argsort_desc(sel, order, ba, bb, n_expert);
  if (rc != GUFO_FP4_OK) {
    return rc;
  }

  float sum = 0.f;
  for (int i = 0; i < n_used; ++i) {
    ids[i] = order[i];
    weights[i] = probs[ids[i]];
    sum += weights[i];
  }
  if (norm_w) {
    float den = sum < GUFO_FP4_WEIGHT_SUM_MIN ? GUFO_FP4_WEIGHT_SUM_MIN : sum;
    for (int i = 0; i < n_used; ++i) {
      weights[i] = weights[i] / den;
    }
  }
  if (w_scale != 0.f && w_scale != 1.f) {
    for (int i = 0; i < n_used; ++i) {
      weights[i] *= w_scale;
    }
  }
  return GUFO_FP4_OK;
}
