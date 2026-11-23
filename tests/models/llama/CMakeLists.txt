function(add_llama_cpu_test target source)
  add_executable(${target} ${PROJECT_SOURCE_DIR}/${source})
  target_link_libraries(${target} PRIVATE superroc_core)
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES LABELS "cpu;models;llama")
endfunction()

# llama_config_test builds the HIP model object (superroc::hip), so it exists
# only in HIP builds. Reads the real GGUF named by NANBEIGE_GGUF when the
# developer sets it; a missing file is a skip (77). No path is hardcoded:
# public checkouts skip.
if (ENGINE_ENABLE_HIP)
  add_executable(llama_config_test
    ${PROJECT_SOURCE_DIR}/tests/models/llama/llama_config_test.cpp)
  target_link_libraries(llama_config_test PRIVATE superroc_core)
  add_test(NAME llama_config_test COMMAND llama_config_test)
  set_tests_properties(llama_config_test PROPERTIES
    LABELS "gpu;models;llama" SKIP_RETURN_CODE 77)
endif()

add_llama_cpu_test(llama_tokenizer_test
  tests/models/llama/llama_tokenizer_test.cpp)
target_compile_definitions(llama_tokenizer_test PRIVATE
  NBG_TOKREF="${PROJECT_SOURCE_DIR}/tests/models/llama/fixtures/tokref.json")
set_tests_properties(llama_tokenizer_test PROPERTIES SKIP_RETURN_CODE 77)

add_llama_cpu_test(llama_bind_test
  tests/models/llama/llama_bind_test.cpp)
set_tests_properties(llama_bind_test PROPERTIES SKIP_RETURN_CODE 77)

# SUPERROC_CANONICAL_CACHE=1: the prompt state is the same bits wherever
# earlier requests cut the prompt (llama twin of qwen27b_prompt_split_test;
# the Qwen test cannot tokenize llama files).
if (ENGINE_ENABLE_HIP)
  add_executable(llama_prompt_split_test
    ${PROJECT_SOURCE_DIR}/tests/models/llama/prompt_split_test.cpp)
  target_link_libraries(llama_prompt_split_test PRIVATE superroc_core)
  add_test(NAME llama_prompt_split_test COMMAND llama_prompt_split_test)
  set_tests_properties(llama_prompt_split_test PROPERTIES
    LABELS "llama;gpu;hip;cache;external-model" SKIP_RETURN_CODE 77)
endif()
