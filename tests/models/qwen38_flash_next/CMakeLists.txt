# Small operator checks are the default. Full-model qualification and
# diagnostic probes are built explicitly so routine iterations stay focused.
add_custom_target(qwen38_flash_next_tests)

function(add_qwen38_flash_next_test name library labels)
  set(target qwen38_flash_next_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME qwen38_flash_next.${name} COMMAND ${target})
  set_tests_properties(qwen38_flash_next.${name} PROPERTIES
    LABELS "qwen38_flash_next;${labels}")
  add_dependencies(qwen38_flash_next_tests ${target})
endfunction()

add_qwen38_flash_next_test(config superroc_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(continuation_layout superroc_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(ngram superroc_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(mtp_sampling superroc_core "cpu")
add_qwen38_flash_next_test(draft_vocab superroc_core "cpu")
add_qwen38_flash_next_test(prompt_lookup superroc_core "cpu")

add_executable(qwen38_flash_next_reference_probe EXCLUDE_FROM_ALL reference_probe.cpp)
target_link_libraries(qwen38_flash_next_reference_probe PRIVATE
  superroc_qwen38_flash_next_reference)
set_source_files_properties(reference_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")

# FLASHNEXT G1-CPU: binder + oracle qualification against the real ROCmFP4
# artifact. EXCLUDE_FROM_ALL and unregistered with ctest: it mmaps the 105 GB
# seat file, whose path is a command-line argument, and is run explicitly
# through gpu-run.
add_executable(qwen38_flash_next_rocmfp4_oracle_test EXCLUDE_FROM_ALL
  rocmfp4_oracle_test.cpp)
target_link_libraries(qwen38_flash_next_rocmfp4_oracle_test PRIVATE
  superroc_qwen38_flash_next_reference)
set_source_files_properties(rocmfp4_oracle_test.cpp PROPERTIES
  COMPILE_OPTIONS "-O2")

# FLASHNEXT G1-GPU: S6 dense type-mix qualification against the real ROC
# artifact. EXCLUDE_FROM_ALL and unregistered with ctest: it needs the GPU
# and the HIP half of the model library, and is run explicitly through
# gpu-run.
if (ENGINE_ENABLE_HIP)
  add_executable(qwen38_flash_next_rocmfp4_gpu_test EXCLUDE_FROM_ALL
    rocmfp4_gpu_test.cpp)
  target_link_libraries(qwen38_flash_next_rocmfp4_gpu_test PRIVATE
    superroc_qwen38_flash_next_reference)
  set_source_files_properties(rocmfp4_gpu_test.cpp PROPERTIES
    COMPILE_OPTIONS "-O2")
endif()

if (ENGINE_ENABLE_HIP)
  foreach(name IN ITEMS mtp_ops blaslt_ops hc_mix_ops gdn_ops
      attention_ops moe_epilogue_ops moe_ids_ops projection_ops routed_wmma_ops
      select_ops snapshot_copy)
    add_qwen38_flash_next_test(${name} superroc_qwen38_flash_next "hip;gpu;gfx1151")
  endforeach()

  add_executable(qwen38_flash_next_session_test EXCLUDE_FROM_ALL session_test.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_disk_store.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/component_text_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_model_runner.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_generation_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/inference_backend.cpp)
  target_link_libraries(qwen38_flash_next_session_test PRIVATE superroc_logging
    superroc_qwen38_flash_next superroc_deepseek_v4_flash superroc_cache_hip hip::host)
  add_executable(qwen38_flash_next_snapshot_test EXCLUDE_FROM_ALL snapshot_test.cpp
    device_headroom_fault.cpp)
  target_link_libraries(qwen38_flash_next_snapshot_test PRIVATE
    superroc_qwen38_flash_next ${CMAKE_DL_LIBS})
  add_executable(qwen38_flash_next_continuation_adapter_test EXCLUDE_FROM_ALL
    continuation_adapter_test.cpp reset_submission_fault.cpp)
  target_link_libraries(qwen38_flash_next_continuation_adapter_test PRIVATE
    superroc_qwen38_flash_next ${CMAKE_DL_LIBS})
  add_executable(qwen38_flash_next_continuation_probe EXCLUDE_FROM_ALL
    continuation_probe.cpp)
  target_link_libraries(qwen38_flash_next_continuation_probe PRIVATE
    superroc_qwen38_flash_next)
  add_custom_target(qwen38_flash_next_model_tests
    DEPENDS qwen38_flash_next_session_test qwen38_flash_next_snapshot_test
      qwen38_flash_next_continuation_adapter_test)

  add_executable(qwen38_flash_next_dense_gemm_bench EXCLUDE_FROM_ALL
    dense_gemm_bench.cpp)
  target_link_libraries(qwen38_flash_next_dense_gemm_bench PRIVATE
    superroc_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_attn_bench EXCLUDE_FROM_ALL attn_bench.cpp)
  target_link_libraries(qwen38_flash_next_attn_bench PRIVATE
    superroc_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_decode_attn_bench EXCLUDE_FROM_ALL
    decode_attn_bench.cpp)
  target_link_libraries(qwen38_flash_next_decode_attn_bench PRIVATE
    superroc_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_routed_gemm_bench EXCLUDE_FROM_ALL
    routed_gemm_bench.cpp)
  target_link_libraries(qwen38_flash_next_routed_gemm_bench PRIVATE
    superroc_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_gpu_probe EXCLUDE_FROM_ALL gpu_probe.cpp
    mtp_audit.cpp)
  target_link_libraries(qwen38_flash_next_gpu_probe PRIVATE
    superroc_qwen38_flash_next_reference)
endif()

# LFP item 1: per-shape dense GEMM microbench (timing only, random weights).
# EXCLUDE_FROM_ALL, run explicitly through gpu-run.
if (ENGINE_ENABLE_HIP)
  add_executable(qwen38_flash_next_fp4_dense_bench EXCLUDE_FROM_ALL
    fp4_dense_bench.cpp)
  target_link_libraries(qwen38_flash_next_fp4_dense_bench PRIVATE
    superroc_qwen38_flash_next)
endif()

# LFP item 4: FP4->F16 decode exactness + GEMM qualification. LANGUAGE HIP: the
# 4a kernel calls fp4_f16::Fp4DecodeBlock directly.
if (ENGINE_ENABLE_HIP)
  add_qwen38_flash_next_test(fp4_f16 superroc_qwen38_flash_next "hip;gpu;gfx1151")
  set_source_files_properties(fp4_f16_test.cpp
    PROPERTIES LANGUAGE HIP)
endif()
