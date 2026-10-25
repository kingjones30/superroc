target_sources(gufo_core PRIVATE
  ${PROJECT_SOURCE_DIR}/src/models/qwen/vision/image_cache.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/vision/prompt.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/tokenizer.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/chat_template.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/oracles.cpp
  ${PROJECT_SOURCE_DIR}/src/models/glimmer/glimmer_oracle.cpp
  ${PROJECT_SOURCE_DIR}/src/models/glimmer/glimmer_chat.cpp
  ${PROJECT_SOURCE_DIR}/src/models/lfm2/lfm2_chat.cpp  # LFM2
  ${PROJECT_SOURCE_DIR}/src/models/flint/flint_chat.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/state.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/weights.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/ssm.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/forward.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/moe/forward_moe.cpp  # KAT:
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/attention.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/embed.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/ffn.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/norm.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/quant_gemm.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/residual.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/rope.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/sample.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/ssm.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/unembed.cpp
   ${PROJECT_SOURCE_DIR}/src/models/qwen/generator.cpp
   ${PROJECT_SOURCE_DIR}/src/models/qwen/mtp_reference.cpp
   ${PROJECT_SOURCE_DIR}/src/models/qwen/dflash_weights.cpp)

if (ENGINE_ENABLE_HIP)
  find_package(OpenSSL REQUIRED COMPONENTS Crypto)
  target_link_libraries(gufo_core PRIVATE OpenSSL::Crypto)
  target_sources(gufo_core PRIVATE
    ${PROJECT_SOURCE_DIR}/src/models/qwen/vision/encoder.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/vision/device_input.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/attention.hip
    ${PROJECT_SOURCE_DIR}/src/models/glimmer/hip/glimmer_decode.hip  # GLIMMER
    ${PROJECT_SOURCE_DIR}/src/models/lfm2/hip/lfm2_kernels.hip  # LFM2
    ${PROJECT_SOURCE_DIR}/src/models/lfm2/hip/lfm2_decode.hip  # LFM2
    ${PROJECT_SOURCE_DIR}/src/models/lfm2/hip/lfm2_prefill.hip  # LFM2
    ${PROJECT_SOURCE_DIR}/src/models/flint/hip/flint_decode.hip
    ${PROJECT_SOURCE_DIR}/src/models/flint/hip/flint_attention.hip
    ${PROJECT_SOURCE_DIR}/src/models/flint/hip/flint_prefill.hip
    ${PROJECT_SOURCE_DIR}/src/models/lfm2/hip/lfm2_prefill_attention.hip  # LFM2
    ${PROJECT_SOURCE_DIR}/src/models/lfm2/hip/lfm2_attention.hip  # LFM2
    ${PROJECT_SOURCE_DIR}/src/models/glimmer/hip/glimmer_prefill.hip  # GLIMMER
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/batched_ssm.hip
    ${PROJECT_SOURCE_DIR}/src/models/glimmer/hip/glimmer_attention.hip  # LGA
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_embed.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_gemm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_quant_gemm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_fp16.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_quant_wave64.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/small_batch_wave64.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/small_batch_quant16_wave64.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_norm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_residual.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_rope.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_ssm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_swiglu.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_unpack.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/embed.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/mtp_batch.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/residual.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/unpack.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/rope.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/sample.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/norm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/gemv.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/gemv_quant.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/fused.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/qkv.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/ssm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/ssm_decode_recurrence.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/ssm_recurrence.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/ssm_row_split.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/swiglu.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_batched.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_decode.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_decode_verify_wave64.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_decode_graph.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_tile.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_wmma.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/hipblas_gemm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/hipblaslt_gemm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/dflash_kernels.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/arena.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/batched_decode.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/decode.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/decode_step.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/executor.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/fp4_warmup.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/model_loader.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/prefill.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/prefill_chunk.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/ssm_replay.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/mtp/model.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/mtp/executor.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/mtp/backend.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/mtp/batch.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/mtp/kat_moe_ffn.hip  # KAT-MTP:
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/dflash/model.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/dflash/executor.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/dflash/batch.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/dflash/backend.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/dflash/dspark_sampling.hip)
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/dflash/dspark_sampling.hip
    TARGET_DIRECTORY gufo_core
    PROPERTIES LANGUAGE HIP)

  # HIP helpers must share the kernel's wave size, so compile this unit
  # separately while the other Qwen kernels retain wave32.
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/small_batch_wave64.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_quant_wave64.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_decode_verify_wave64.hip
    TARGET_DIRECTORY gufo_core
    PROPERTIES COMPILE_OPTIONS -mwavefrontsize64)
  # Iterative ILP improves Q4/Q5 groups of sixteen without changing arithmetic.
  # Keep it isolated: other widths regress. Line tables retain source profiling
  # while avoiding an LLVM register-pressure crash with full debug information.
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/small_batch_quant16_wave64.hip
    TARGET_DIRECTORY gufo_core
    PROPERTIES COMPILE_OPTIONS
    "-mwavefrontsize64;-gline-tables-only;-Xarch_device;-mllvm=-amdgpu-sched-strategy=iterative-ilp")
  # lqw:cmake-hip. Renamed Ling FP4 kernels. wmma stays wave64; the others stay wave32.
  target_sources(gufo_core PRIVATE
    ${PROJECT_SOURCE_DIR}/src/core/rocm_fp4/gufo_fp4_gemv_q8.hip
    ${PROJECT_SOURCE_DIR}/src/core/rocm_fp4/gufo_fp4_gemm.hip
    ${PROJECT_SOURCE_DIR}/src/core/rocm_fp4/gufo_fp4_wmma.hip
    ${PROJECT_SOURCE_DIR}/src/core/rocm_fp4/fp4_f16_prefill.hip  # SQ1:
    ${PROJECT_SOURCE_DIR}/src/core/rocm_fp4/gufo_fp4_moe_group.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/moe/qwen_moe_route.hip  # KAT:
    ${PROJECT_SOURCE_DIR}/src/models/qwen/moe/qwen_moe_ffn.hip  # KAT:
    ${PROJECT_SOURCE_DIR}/src/models/qwen/moe/qwen_moe_prefill.hip)  # KAT:
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/core/rocm_fp4/gufo_fp4_gemv_q8.hip
    ${PROJECT_SOURCE_DIR}/src/core/rocm_fp4/gufo_fp4_gemm.hip
    ${PROJECT_SOURCE_DIR}/src/core/rocm_fp4/gufo_fp4_moe_group.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/moe/qwen_moe_route.hip  # KAT:
    ${PROJECT_SOURCE_DIR}/src/models/qwen/moe/qwen_moe_ffn.hip  # KAT:
    ${PROJECT_SOURCE_DIR}/src/models/qwen/moe/qwen_moe_prefill.hip  # KAT:
    TARGET_DIRECTORY gufo_core
    PROPERTIES LANGUAGE HIP)
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/core/rocm_fp4/gufo_fp4_wmma.hip
    TARGET_DIRECTORY gufo_core
    PROPERTIES LANGUAGE HIP COMPILE_OPTIONS -mwavefrontsize64)
  # SQ1: the promoted FP4->F16 prefill kernels. Wave32 like kernels.hip.cpp;
  # -ffast-math -fno-finite-math-only match that TU's codegen exactly, so the
  # moved kernels are bit-identical to the Flash-Next build. The two -Wno-
  # flags match its warning behaviour too: gufo_core builds with -Wall -Wextra
  # but the Flash-Next TU has neither, so without them the byte-exact moved
  # code warns (-Wsign-compare in the SSM epilogue, -Wunneeded-internal for the
  # HcScaledSilu dep whose only uses here are dead template branches).
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/core/rocm_fp4/fp4_f16_prefill.hip
    TARGET_DIRECTORY gufo_core
    PROPERTIES LANGUAGE HIP COMPILE_OPTIONS
    "-ffast-math;-fno-finite-math-only;-Wno-sign-compare;-Wno-unneeded-internal-declaration")
  # Teacher-forced perplexity for G3. Not part of the default test build.
  add_executable(glimmer_ppl EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/glimmer_ppl.cpp)
  target_link_libraries(glimmer_ppl PRIVATE gufo_core)
  # LFM2: teacher-forced perplexity on the LFM2 token-loop decode path.
  add_executable(lfm2_ppl EXCLUDE_FROM_ALL  # LFM2
    ${PROJECT_SOURCE_DIR}/tools/lfm2_ppl.cpp)
  target_link_libraries(lfm2_ppl PRIVATE gufo_core)
  add_executable(flint_ppl EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/flint_ppl.cpp)
  target_link_libraries(flint_ppl PRIVATE gufo_core)
  add_executable(flint_parity EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/flint_parity.cpp)
  target_link_libraries(flint_parity PRIVATE gufo_core)

  # SQ1: ported-kernel unit test + 27B-shape microbench. GPU-only tools,
  # EXCLUDE_FROM_ALL like the other qualification tools; run via gpu-run.
  add_executable(qwen_fp4_f16_prefill_test EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tests/models/qwen/hip/fp4_f16_prefill_test.cpp)
  # The tiled-Q8 producer and the Q8_0 cross-check kernel stay Flash-Next
  # resident; the ported launchers come from gufo_core.
  target_link_libraries(qwen_fp4_f16_prefill_test PRIVATE gufo_core
    gufo_qwen38_flash_next)
  # LANGUAGE HIP: the int8 cell calls fp4_f16::Fp4ScaleByteToFloat, which
  # lives in the header's device section (same precedent as fp4_f16_test).
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/tests/models/qwen/hip/fp4_f16_prefill_test.cpp
    TARGET_DIRECTORY qwen_fp4_f16_prefill_test
    PROPERTIES LANGUAGE HIP)
  add_executable(qwen_gemm_bench EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/qwen_gemm_bench.cpp)
  target_link_libraries(qwen_gemm_bench PRIVATE gufo_core)
  # SLP vectorization puts the K-quant scales and offsets of the blocked
  # prefill GEMMs into wide register tuples. With clang 23, the Q3_K and Q6_K
  # 128x128 tiles then spill. Disable SLP for the two units on all compilers.
  set_property(
    SOURCE
      ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_quant_gemm.hip
      ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_quant_wave64.hip
    TARGET_DIRECTORY gufo_core
    APPEND PROPERTY COMPILE_OPTIONS -fno-slp-vectorize)
endif()

# lqw:qwen35_ppl. Teacher-forced perplexity on the GPU executor over a prompt
# file (the gufo-side leg of the fork llama-perplexity comparison).
add_executable(qwen35_ppl EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/qwen35_ppl.cpp)
target_link_libraries(qwen35_ppl PRIVATE gufo_core)

# lqw:spec6. Bitwise small-m FP4 GEMV against real 27B layers.
add_executable(qwen35_fp4_m_test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/qwen35_fp4_m_test.cpp)
target_link_libraries(qwen35_fp4_m_test PRIVATE gufo_core)
