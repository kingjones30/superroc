function(add_flint_cpu_test target source)
  add_executable(${target} ${PROJECT_SOURCE_DIR}/${source})
  target_link_libraries(${target} PRIVATE superroc_core)
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES LABELS "cpu;models;flint")
endfunction()

add_flint_cpu_test(flint_tokenizer_test
  tests/models/flint/flint_tokenizer_test.cpp)
target_compile_definitions(flint_tokenizer_test PRIVATE
  FLINT_TOKREF="${PROJECT_SOURCE_DIR}/tests/models/flint/fixtures/tokref.jsonl")
set_tests_properties(flint_tokenizer_test PROPERTIES SKIP_RETURN_CODE 77)

add_flint_cpu_test(flint_chat_test
  tests/models/flint/flint_chat_test.cpp)
target_compile_definitions(flint_chat_test PRIVATE
  FLINT_RENDER="${PROJECT_SOURCE_DIR}/tests/models/flint/fixtures/render.json")

add_flint_cpu_test(flint_bind_test
  tests/models/flint/flint_bind_test.cpp)
set_tests_properties(flint_bind_test PROPERTIES SKIP_RETURN_CODE 77)

if (ENGINE_ENABLE_HIP)
  function(add_flint_hip_test target source labels)
    add_executable(${target} ${PROJECT_SOURCE_DIR}/${source})
    target_link_libraries(${target} PRIVATE
      superroc_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES
      LABELS "${labels}"
      SKIP_RETURN_CODE 77)
  endfunction()

  add_flint_hip_test(flint_sink_attention_test
    tests/models/flint/hip/flint_sink_attention_test.cpp
    "hip;gpu;kernels;flint")
  add_flint_hip_test(flint_splitk_attention_test
    tests/models/flint/hip/flint_splitk_attention_test.cpp
    "hip;gpu;kernels;flint")
  add_flint_hip_test(flint_wmma_prefill_test
    tests/models/flint/hip/flint_wmma_prefill_test.cpp
    "hip;gpu;kernels;flint")
endif()
