function(add_llama_cpu_test target source)
  add_executable(${target} ${PROJECT_SOURCE_DIR}/${source})
  target_link_libraries(${target} PRIVATE gufo_core)
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES LABELS "cpu;models;llama")
endfunction()

add_llama_cpu_test(llama_config_test
  tests/models/llama/llama_config_test.cpp)
# Reads the real GGUF named by NANBEIGE_GGUF; a missing file is a skip (77).
set_tests_properties(llama_config_test PROPERTIES
  SKIP_RETURN_CODE 77
  ENVIRONMENT
    "NANBEIGE_GGUF=/home/kingjones/models/nanbeige/nanbeige4.3-33b-base-q4_k_m.gguf")

add_llama_cpu_test(llama_tokenizer_test
  tests/models/llama/llama_tokenizer_test.cpp)
target_compile_definitions(llama_tokenizer_test PRIVATE
  NBG_TOKREF="${PROJECT_SOURCE_DIR}/tests/models/llama/fixtures/tokref.json")
set_tests_properties(llama_tokenizer_test PROPERTIES
  SKIP_RETURN_CODE 77
  ENVIRONMENT
    "NANBEIGE_GGUF=/home/kingjones/models/nanbeige/nanbeige4.3-33b-base-q4_k_m.gguf")

add_llama_cpu_test(llama_bind_test
  tests/models/llama/llama_bind_test.cpp)
set_tests_properties(llama_bind_test PROPERTIES
  SKIP_RETURN_CODE 77
  ENVIRONMENT
    "NANBEIGE_GGUF=/home/kingjones/models/nanbeige/nanbeige4.3-33b-base-q4_k_m.gguf")
