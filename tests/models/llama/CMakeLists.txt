function(add_llama_cpu_test target source)
  add_executable(${target} ${PROJECT_SOURCE_DIR}/${source})
  target_link_libraries(${target} PRIVATE superroc_core)
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES LABELS "cpu;models;llama")
endfunction()

add_llama_cpu_test(llama_config_test
  tests/models/llama/llama_config_test.cpp)
# Reads the real GGUF named by NANBEIGE_GGUF when the developer sets it; a
# missing file is a skip (77). No path is hardcoded: public checkouts skip.
set_tests_properties(llama_config_test PROPERTIES SKIP_RETURN_CODE 77)

add_llama_cpu_test(llama_tokenizer_test
  tests/models/llama/llama_tokenizer_test.cpp)
target_compile_definitions(llama_tokenizer_test PRIVATE
  NBG_TOKREF="${PROJECT_SOURCE_DIR}/tests/models/llama/fixtures/tokref.json")
set_tests_properties(llama_tokenizer_test PROPERTIES SKIP_RETURN_CODE 77)

add_llama_cpu_test(llama_bind_test
  tests/models/llama/llama_bind_test.cpp)
set_tests_properties(llama_bind_test PROPERTIES SKIP_RETURN_CODE 77)
