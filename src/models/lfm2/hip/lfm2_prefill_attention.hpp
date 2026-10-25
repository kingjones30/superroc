// LFM2 tiled fp32 flash-attention prefill for head_dim 64 / GQA (SPEC2, lane
// LFS). // LFM2
#ifndef GUFO_MODELS_LFM2_HIP_LFM2_PREFILL_ATTENTION_HPP_
#define GUFO_MODELS_LFM2_HIP_LFM2_PREFILL_ATTENTION_HPP_

#include "src/models/lfm2/hip/lfm2_kernels.hpp"  // LFM2 (Lfm2LaunchStatus)

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>  // LFM2

// LFM2: the launcher is declared in lfm2_kernels.hpp (next to the other LFM2 //
// LFM2 launchers) so lfm2_prefill.hip sees it through its existing includes. //
// LFM2 Contract (SPEC2 step 1): head_dim 64 fixed, group = heads/kv_heads == 4;
// // LFM2 q/out are [batch][heads][64], the KV cache is fp32 head-major // LFM2
// (k_cache[((layer*kv_heads+kvh)*max_ctx+pos)*64+d], v_cache = k_cache+plane);
// // LFM2 query row r sits at absolute position start_pos+r and attends
// positions // LFM2 [0, start_pos+r] with scale 1/sqrt(64); out = acc/l. //
// LFM2

#endif  // defined(ENGINE_ENABLE_HIP)  // LFM2

#endif  // GUFO_MODELS_LFM2_HIP_LFM2_PREFILL_ATTENTION_HPP_  // LFM2
