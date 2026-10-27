# Ling-3.0-flash (bailingmoe3) model package.
#
# The scalar oracle is built only by explicit validation targets, not by
# production or routine checks: it decodes every weight on the fly and is
# unusable at speed.
add_library(superroc_ling3 STATIC
  chat_template.cpp
  rocm_backend.cpp
  config.cpp
  rocmfp4.cpp
  tokenizer.cpp
  weights.cpp
  vision_prompt.cpp
  vision_hook.cpp)

target_include_directories(superroc_ling3 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(superroc_ling3 PUBLIC superroc_core Threads::Threads)

add_library(superroc_ling3_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(superroc_ling3_reference PUBLIC superroc_ling3)

# The reference decodes every weight on the fly; unoptimized it is far too
# slow even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

# The scalar oracle walks ~3e9 weight elements per token; parallelising over
# output rows only keeps every row's accumulation order intact.
find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
  target_link_libraries(superroc_ling3_reference PUBLIC OpenMP::OpenMP_CXX)
endif()

# Verification target: checks the ROCmFP4 decoder bit-for-bit against the
# fork's own, dlopen'd from the caller's libggml-base.so. Never built by
# default; it needs a path that only exists on the port box.
add_executable(superroc_ling3_rocmfp4_test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/ling3_rocmfp4_test.cpp)
target_link_libraries(superroc_ling3_rocmfp4_test PRIVATE superroc_ling3 ${CMAKE_DL_LIBS})

# The oracle driver. Prints greedy token ids and can dump per-step top-5 so a
# divergence is diagnosable without re-running.
add_executable(superroc_ling3_ref EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/ling3_ref_main.cpp)
target_link_libraries(superroc_ling3_ref PRIVATE superroc_ling3_reference)
set_source_files_properties(${PROJECT_SOURCE_DIR}/tools/ling3_ref_main.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

# LS4: replays mtp_ref_test's dump through mtp::DraftNext on the GPU and
# compares the draft tokens against the CPU oracle's. Needs the HIP build.
if(ENGINE_ENABLE_HIP)
  add_executable(mtp_gpu_check EXCLUDE_FROM_ALL mtp_gpu_check.cpp)
  target_link_libraries(mtp_gpu_check PRIVATE superroc_ling3 ${CMAKE_DL_LIBS})
  set_source_files_properties(mtp_gpu_check.cpp PROPERTIES COMPILE_OPTIONS "-O2")
endif()

# LS4: drives the MTP draft head (block 42) on the CPU oracle and MEASURES
# draft acceptance. CPU only, links superroc_ling3_reference -- no HIP, no GPU.
add_executable(mtp_ref_test EXCLUDE_FROM_ALL
  mtp_ref_test.cpp)
target_link_libraries(mtp_ref_test PRIVATE superroc_ling3_reference)
set_source_files_properties(mtp_ref_test.cpp PROPERTIES COMPILE_OPTIONS "-O2")

# Checks superroc's Q6_K / Q8_0 row decoders against the fork's, bit for bit.
add_executable(superroc_ling3_dequant_parity EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/ling3_dequant_parity.cpp)
target_link_libraries(superroc_ling3_dequant_parity PRIVATE superroc_core
  superroc_ling3 ${CMAKE_DL_LIBS})

# Chat-template renderer check. Self-contained (no superroc_core), so it builds and
# runs anywhere; it pins the byte-significant markers.
add_executable(superroc_ling3_chat_template_test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/ling3_chat_template_test.cpp)
target_link_libraries(superroc_ling3_chat_template_test PRIVATE superroc_ling3)

# Standalone semantics tests. These depend on nothing -- not on superroc_core, not on
# a model file, not on the GPU -- because they compare LG's arithmetic against a
# second, independent transcription of the fork's kernels and graph. Two real
# bugs were found this way (the shared KDA conv state and the MLA softmax indexed
# by head instead of position), both of which a single-token smoke test cannot
# see. Build them by name; they are never part of a normal build.
foreach(lg3_test kda_math_test kda_layer_test mla_layer_test moe_router_test)
  add_executable(${lg3_test} EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/${lg3_test}.cpp)
  target_compile_features(${lg3_test} PRIVATE cxx_std_20)
  set_source_files_properties(${PROJECT_SOURCE_DIR}/tools/${lg3_test}.cpp
    PROPERTIES COMPILE_OPTIONS "-O2")
endforeach()

# Runs every standalone semantics test and reports one verdict.
add_custom_target(superroc_ling3_semantics
  COMMAND kda_math_test
  COMMAND kda_layer_test
  COMMAND mla_layer_test
  COMMAND moe_router_test
  DEPENDS kda_math_test kda_layer_test mla_layer_test moe_router_test
  COMMENT "ling3 semantics: KDA conv/l2norm/scan, KDA layer, MLA layer, MoE router")

# Structural wiring check: Step over a synthetic 3-block model (KDA, MLA, MoE).
# It links the real reference.cpp, so it needs the quant entry points too; the
# test stubs them because its synthetic tensors are all F32.
# Deliberately NOT linked against superroc_ling3: that would pull in weights.cpp,
# which defines TensorRef::RowBytes() -- the very symbol the test defines itself
# so it does not have to drag the whole GgufReader in behind it. Compiling the
# two sources it does need keeps the target self-contained.
# reference.cpp is listed explicitly: ReferenceModel lives there, and linking
# superroc_ling3_reference instead would drag in superroc_ling3 -> weights.cpp and
# collide with the test's own TensorRef::RowBytes().
add_executable(ling3_step_test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/ling3_step_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reference.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu_ops.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rocmfp4.cpp)
target_include_directories(ling3_step_test PRIVATE ${PROJECT_SOURCE_DIR})
set_source_files_properties(${PROJECT_SOURCE_DIR}/tools/ling3_step_test.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

# ---------------------------------------------------------------------------
# ROCm backend. engine.cpp is compiled either way; the kernels and LK's / LX's
# sources only with HIP, and LING3_WITH_ROCM is what switches Model::Load
# between the two backends.
# ---------------------------------------------------------------------------
set(LING3_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

if(ENGINE_ENABLE_HIP)
  # LG's own: elementwise/reduction/rotary plus the decode GEMV.
  # LK's and LX's kernel sources, copied into kernels/. A missing one is
  # skipped with a message rather than a configure error, so a partial set
  # degrades to the refusal rocm_backend.cpp already produces.
  # LS3 2026-10-01: ling3_gemm.hip (batched-prefill GEMM + router) follows the
  # same rule.
  set(LING3_HIP_SOURCES "")
  # LS3 P3: ling3_wmma.hip holds only WaveSize-64 kernels (the
  # __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64 path needs the wavefrontsize64
  # target feature; superroc compiles its own wave64 kernel this way too). The
  # other kernels retain wave32. Its 32-lane reductions pin the width
  # explicitly (see the file).
  set_source_files_properties(${LING3_KERNEL_DIR}/ling3_wmma.hip
    PROPERTIES COMPILE_OPTIONS -mwavefrontsize64)
  foreach(lg3_src ling3_ops.hip ling3_mla.hip ling3_mla_tile.hip ling3_kda.hip ling3_moe.hip ling3_gemm.hip ling3_wmma.hip ling3_gemv_q8.hip ling3_mla_w2.hip ling3_mla_fw.hip ling3_moe_group.hip)
    if(EXISTS ${LING3_KERNEL_DIR}/${lg3_src})
      list(APPEND LING3_HIP_SOURCES ${LING3_KERNEL_DIR}/${lg3_src})
    else()
      message(STATUS "ling3: ${lg3_src} is not present; the ROCm executor "
                     "cannot link without it")
    endif()
  endforeach()
  # LK's KDA block and LX's quant/GEMV/router/expert-FFN, copied into this
  # package rather than linked across trees.
  # mtp_gpu.cpp is LS4's MTP draft head. It is listed only in the HIP build:
  # it has its own #else stub, but there is nothing to compile without HIP.
  target_sources(superroc_ling3 PRIVATE engine.cpp mtp_gpu.cpp ${LING3_HIP_SOURCES})
  set_source_files_properties(${LING3_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
  set_target_properties(superroc_ling3 PROPERTIES HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(superroc_ling3 PRIVATE LING3_WITH_ROCM=1)
  # LM1 2026-10-02: LS1's small-m decode kernels (ling3_gemv_m /
  # ling3_gemv_q8_m / ling3_expert_ffn_m in kernels/ling3_gemv_q8.{hpp,hip})
  # let a speculative verify read each weight ONCE for m activation columns.
  # Detect them and tell rocm_backend.cpp so StepMultiDispatch can route
  # through StepMultiMultiRow; without them the tree still builds and the
  # verify stays m sequential Step[s] (StepMultiDispatch keeps its #if
  # fallback).
  if(EXISTS ${LING3_KERNEL_DIR}/ling3_gemv_q8.hpp)
    file(READ ${LING3_KERNEL_DIR}/ling3_gemv_q8.hpp _ling3_gemv_q8_hpp)
    string(FIND "${_ling3_gemv_q8_hpp}" "ling3_gemv_q8_m" _ling3_have_multi_row)
    if(NOT _ling3_have_multi_row EQUAL -1)
      target_compile_definitions(superroc_ling3 PRIVATE LING3_HAVE_MULTI_ROW_KERNELS=1)
      message(STATUS "ling3: small-m decode kernels present; "
                     "StepMultiMultiRow enabled behind LING3_MTP_MULTI=1")
    endif()
    unset(_ling3_gemv_q8_hpp)
    unset(_ling3_have_multi_row)
  endif()
  target_include_directories(superroc_ling3 SYSTEM PUBLIC ${LING3_KERNEL_DIR})
  target_compile_options(superroc_ling3 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(superroc_ling3 PUBLIC hip::host)

  # LVV: the vision tower is its own HIP library so it does not inherit the
  # language kernels' -ffast-math. The projector math stays the one the
  # cosine check measured.
  add_library(superroc_ling3_vision STATIC vision_tower_lib.hip)
  set_source_files_properties(vision_tower_lib.hip PROPERTIES LANGUAGE HIP)
  set_target_properties(superroc_ling3_vision PROPERTIES HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_include_directories(superroc_ling3_vision PUBLIC ${PROJECT_SOURCE_DIR})
  target_link_libraries(superroc_ling3_vision PUBLIC hip::host roc::hipblas)
  target_link_libraries(superroc_ling3 PUBLIC superroc_ling3_vision)
  add_executable(lvv_describe EXCLUDE_FROM_ALL vision_describe.cpp)
  target_link_libraries(lvv_describe PRIVATE superroc_ling3 superroc_ling3_reference)


  # LC1 (wave 3, 2026-10-03): the token-parallel KDA conv against a VERBATIM copy of
  # build28g's serial conv_silu_kernel that lives inside the test -- the "old" side is not
  # the new dispatcher. Bit-identical or it did not happen. Compiled as HIP: the test
  # defines the reference __global__ kernel.
  add_executable(superroc_ling3_kda_conv_parity EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/ling3_kda_conv_parity.cpp
    ${LING3_KERNEL_DIR}/ling3_kda.hip)
  set_source_files_properties(${PROJECT_SOURCE_DIR}/tools/ling3_kda_conv_parity.cpp
    ${LING3_KERNEL_DIR}/ling3_kda.hip PROPERTIES LANGUAGE HIP)
  set_target_properties(superroc_ling3_kda_conv_parity PROPERTIES HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_include_directories(superroc_ling3_kda_conv_parity PRIVATE ${LING3_KERNEL_DIR})
  target_compile_options(superroc_ling3_kda_conv_parity PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(superroc_ling3_kda_conv_parity PRIVATE hip::host)

  # LS3: unit gate for the batched kernels, against the CPU reference, on
  # small shapes. EXCLUDE_FROM_ALL: built by name, run inside gpu-run.
  add_executable(superroc_ling3_gemm_test EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/ling3_gemm_test.cpp
    ${LING3_KERNEL_DIR}/ling3_gemm.hip)
  set_source_files_properties(${LING3_KERNEL_DIR}/ling3_gemm.hip PROPERTIES LANGUAGE HIP)
  set_target_properties(superroc_ling3_gemm_test PROPERTIES HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_include_directories(superroc_ling3_gemm_test PRIVATE ${LING3_KERNEL_DIR})
  target_compile_options(superroc_ling3_gemm_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(superroc_ling3_gemm_test PRIVATE hip::host)

  # LK3: decode KDA scan vs the reference kernel. EXCLUDE_FROM_ALL, run inside gpu-run.
  add_executable(superroc_ling3_kda_decode_bench EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/lk3_kda_decode_bench.cpp
    ${LING3_KERNEL_DIR}/ling3_kda.hip)
  set_source_files_properties(${LING3_KERNEL_DIR}/ling3_kda.hip PROPERTIES LANGUAGE HIP)
  set_target_properties(superroc_ling3_kda_decode_bench PROPERTIES HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_include_directories(superroc_ling3_kda_decode_bench PRIVATE ${LING3_KERNEL_DIR})
  target_compile_options(superroc_ling3_kda_decode_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(superroc_ling3_kda_decode_bench PRIVATE hip::host)

  # LKF (wave 3): fused decode KDA vs the unfused sequence, memcmp exact.
  add_executable(ling3_kda_fusion_test EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/ling3_kda_fusion_test.cpp
    ${LING3_KERNEL_DIR}/ling3_kda.hip)
  set_source_files_properties(${PROJECT_SOURCE_DIR}/tools/ling3_kda_fusion_test.cpp
    ${LING3_KERNEL_DIR}/ling3_kda.hip PROPERTIES LANGUAGE HIP)
  set_target_properties(ling3_kda_fusion_test PROPERTIES HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_include_directories(ling3_kda_fusion_test PRIVATE ${LING3_KERNEL_DIR})
  target_compile_options(ling3_kda_fusion_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(ling3_kda_fusion_test PRIVATE hip::host)


  # LS3 P3: unit gate for the W8A8 int8-WMMA grouped GEMM (fork-semantics
  # oracle + F32 oracle, small shapes, real shapes; optional real GGUF rows).
  add_executable(superroc_ling3_wmma_test EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/ling3_wmma_test.cpp
    ${LING3_KERNEL_DIR}/ling3_wmma.hip)
  set_source_files_properties(${LING3_KERNEL_DIR}/ling3_wmma.hip PROPERTIES LANGUAGE HIP)
  set_target_properties(superroc_ling3_wmma_test PROPERTIES HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_include_directories(superroc_ling3_wmma_test PRIVATE ${LING3_KERNEL_DIR})
  target_compile_options(superroc_ling3_wmma_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(superroc_ling3_wmma_test PRIVATE hip::host)

  # LS3 P2: unit gate for mla_attn_tok (tiled, warp pairs) vs the proven
  # serial mla_attn, on synthetic tensors at the real shapes. No model load.
  add_executable(superroc_ling3_mla_tok_test EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/ling3_mla_tok_test.cpp
    ${LING3_KERNEL_DIR}/ling3_mla.hip)
  set_source_files_properties(${LING3_KERNEL_DIR}/ling3_mla.hip PROPERTIES LANGUAGE HIP)
  set_target_properties(superroc_ling3_mla_tok_test PROPERTIES HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_include_directories(superroc_ling3_mla_tok_test PRIVATE ${LING3_KERNEL_DIR})
  target_compile_options(superroc_ling3_mla_tok_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(superroc_ling3_mla_tok_test PRIVATE hip::host)

  # LW1 (wave 2): unit gate for the two-pass WMMA prefill attention vs the
  # serial mla_attn and vs mla_attn_tok, synthetic tensors at real shapes.
  add_executable(superroc_ling3_mla_w2_test EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/ling3_mla_w2_test.cpp
    ${LING3_KERNEL_DIR}/ling3_mla.hip
    ${LING3_KERNEL_DIR}/ling3_mla_w2.hip)
  set_source_files_properties(${LING3_KERNEL_DIR}/ling3_mla.hip
    ${LING3_KERNEL_DIR}/ling3_mla_w2.hip PROPERTIES LANGUAGE HIP)
  set_target_properties(superroc_ling3_mla_w2_test PROPERTIES HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_include_directories(superroc_ling3_mla_w2_test PRIVATE ${LING3_KERNEL_DIR})
  target_compile_options(superroc_ling3_mla_w2_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(superroc_ling3_mla_w2_test PRIVATE hip::host)

  # LF1 SPEC3: unit gate for the ported fork tile kernel
  # (ling3_mla_tile.hip) vs mla_attn_tok, on synthetic tensors at the real
  # shapes plus an optional real captured q/cache. EXCLUDE_FROM_ALL: built by
  # name, run inside gpu-run.
  add_executable(superroc_ling3_mla_prefill_test EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/ling3_mla_prefill_test.cpp
    ${LING3_KERNEL_DIR}/ling3_mla.hip
    ${LING3_KERNEL_DIR}/ling3_mla_tile.hip
    ${LING3_KERNEL_DIR}/ling3_mla_w2.hip
    ${LING3_KERNEL_DIR}/ling3_mla_fw.hip)
  set_source_files_properties(
    ${LING3_KERNEL_DIR}/ling3_mla.hip
    ${LING3_KERNEL_DIR}/ling3_mla_tile.hip
    ${LING3_KERNEL_DIR}/ling3_mla_w2.hip
    ${LING3_KERNEL_DIR}/ling3_mla_fw.hip PROPERTIES LANGUAGE HIP)
  set_target_properties(superroc_ling3_mla_prefill_test PROPERTIES HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_include_directories(superroc_ling3_mla_prefill_test PRIVATE ${LING3_KERNEL_DIR})
  target_compile_options(superroc_ling3_mla_prefill_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(superroc_ling3_mla_prefill_test PRIVATE hip::host)

  # LS3 PHASE6: minimal decode runner for rocprof kernel traces.
  add_executable(superroc_ling3_trace_decode EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/ling3_trace_decode.cpp)
  target_link_libraries(superroc_ling3_trace_decode PRIVATE superroc_ling3
    superroc_ling3_reference)

  # LS3 PHASE4: split-K decode-attention gate vs the serial mla_attn.
  add_executable(superroc_ling3_mla_sk_test EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/ling3_mla_sk_test.cpp
    ${LING3_KERNEL_DIR}/ling3_mla.hip)
  set_target_properties(superroc_ling3_mla_sk_test PROPERTIES HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_include_directories(superroc_ling3_mla_sk_test PRIVATE ${LING3_KERNEL_DIR})
  target_compile_options(superroc_ling3_mla_sk_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(superroc_ling3_mla_sk_test PRIVATE hip::host)

  # LC1 SPEC4 (wave 3): unit gate for the RV25 MLA decode fusions -- M1's
  # mla_pre_fused and the Q2-MLA wo Q8_1 epilogue -- against the unfused
  # sequence they replace, on BOTH decode arms (n_pos 100 serial, 1630
  # split-K) at the model's real MLA shapes. Whole-buffer memcmp, not a
  # tolerance. EXCLUDE_FROM_ALL: built by name, run inside gpu-run.
  add_executable(superroc_ling3_mla_fusion_test EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/ling3_mla_fusion_test.cpp
    ${LING3_KERNEL_DIR}/ling3_mla.hip
    ${LING3_KERNEL_DIR}/ling3_gemv_q8.hip)
  set_source_files_properties(${LING3_KERNEL_DIR}/ling3_mla.hip
    ${LING3_KERNEL_DIR}/ling3_gemv_q8.hip PROPERTIES LANGUAGE HIP)
  set_target_properties(superroc_ling3_mla_fusion_test PROPERTIES HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_include_directories(superroc_ling3_mla_fusion_test PRIVATE ${LING3_KERNEL_DIR})
  target_compile_options(superroc_ling3_mla_fusion_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(superroc_ling3_mla_fusion_test PRIVATE hip::host)

  # LS3: model-level batched-prefill gate (scalar-vs-batched logits, teacher
  # forcing, greedy continuation) on the real GGUF. Needs the whole ling3
  # package (weights + reader + kernels). Run inside gpu-run.
  add_executable(superroc_ling3_batch_test EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tools/ling3_batch_test.cpp)
  target_link_libraries(superroc_ling3_batch_test PRIVATE superroc_ling3
    superroc_ling3_reference)
else()
  # CPU build: the engine still loads and runs, on the scalar reference.
  target_sources(superroc_ling3 PRIVATE engine.cpp vision_tower_stub.cpp mtp_stub.cpp)
  target_compile_options(superroc_ling3 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2>)
endif()

# Checks LG's ROCmFP4 matvec against ggml's own CPU kernel. ggml quantizes the
# activation to Q8_0 and runs an integer dot, so decode-and-multiply in float is
# a different computation -- this is what tells the two apart.
add_executable(superroc_ling3_dot_parity EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/ling3_dot_parity.cpp)
target_link_libraries(superroc_ling3_dot_parity PRIVATE superroc_ling3_reference
  ${CMAKE_DL_LIBS})

# GPU-only test and benchmark tools: defined only in HIP builds (a CPU-only
# build has no HIP language).
if(ENGINE_ENABLE_HIP)
# LS1 2026-10-01: the integer (Q8_1 activation) ROCmFP4 GEMV against the CPU
# dequant reference, on synthetic rows of BOTH block types and on every shape
# Ling decodes. Needs the GPU; run it inside gpu-run.
add_executable(superroc_ling3_gemv_q8_test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/ling3_gemv_q8_test.cpp)
target_link_libraries(superroc_ling3_gemv_q8_test PRIVATE superroc_ling3 ${CMAKE_DL_LIBS})

# LA1 2026-10-03: the two-phase argmax, byte-for-byte vs the CPU reference on
# random rows, crafted ties, all -INF, NaNs, slice-boundary n, and stride rows.
# Needs the GPU; run it inside gpu-run. Small and fast.
add_executable(superroc_ling3_argmax_test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/ling3_argmax_test.cpp)
target_link_libraries(superroc_ling3_argmax_test PRIVATE superroc_ling3 ${CMAKE_DL_LIBS})

# LF1 2026-10-03: RV25 wave-1 fusion gate. F4 (add_rms_norm vs residual_add +
# rms_norm) memcmp-exact at H=2560, and F3 (the 0.0f + down store) memcmp-exact
# against memset + k_axpy including a -0.0 case, plus ling3_dense_ffn vs
# ling3_dense_ffn_w1 end to end on synthetic ROCmFP4 weights. Needs the GPU; run
# it inside gpu-run. Small and fast.
add_executable(superroc_ling3_fusion_w1_test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/ling3_fusion_w1_test.cpp)
target_link_libraries(superroc_ling3_fusion_w1_test PRIVATE superroc_ling3 ${CMAKE_DL_LIBS})
# It DEFINES the __global__ epilogue kernels the -0.0 trap needs, so it must be
# compiled as HIP rather than as the host compiler the other test tools use.
set_source_files_properties(${PROJECT_SOURCE_DIR}/tools/ling3_fusion_w1_test.cpp
  PROPERTIES LANGUAGE HIP)


# LS1 2026-10-02: the memory ROOF of this box -- achieved read bandwidth for large
# reads, MALL-cold. Every "% of roof" claim needs this number measured on the same
# machine. Allocates its own buffer; never touches the model. Run it inside gpu-run.
add_executable(superroc_ling3_roof_bench EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/ling3_roof_bench.cpp)
target_link_libraries(superroc_ling3_roof_bench PRIVATE superroc_ling3)
# It DEFINES the __global__ read kernels, so it has to be compiled as HIP rather
# than as the host compiler the other tools use.
set_source_files_properties(${PROJECT_SOURCE_DIR}/tools/ling3_roof_bench.cpp
  PROPERTIES LANGUAGE HIP)

# LS1 2026-10-02: the router. Runs build24's kernels against the new ones on the
# device and requires bit-identical logits, ids (in order) and weights, on random
# inputs, adversarial ties and captured real ones. Validates every index it hands
# the device on the HOST first. Run it inside gpu-run, smallest shape first.
add_executable(superroc_ling3_route_test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/ling3_route_test.cpp)
target_link_libraries(superroc_ling3_route_test PRIVATE superroc_ling3)

# LS1 2026-10-02: the device MoE expert grouping, against a host transcription of the
# counting sort it replaces. Bit-exact per element, including the order of slots inside an
# expert. Runs inside gpu-run, smallest shape first.
add_executable(superroc_ling3_moe_group_test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/ling3_moe_group_test.cpp)
target_link_libraries(superroc_ling3_moe_group_test PRIVATE superroc_ling3)

# LS1 2026-10-03: the TILED router logits kernel, against a transcription of the
# one it replaces. Bit-exact per element -- Ling's router has a block-2 near-tie that a
# 1-ulp change flips, so anything less is not a pass. Smallest shape first.
add_executable(superroc_ling3_route_logits_test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/ling3_route_logits_test.cpp)
target_link_libraries(superroc_ling3_route_logits_test PRIVATE superroc_ling3)
# It DEFINES the reference __global__ kernel, so it must be compiled as HIP, not as the
# host compiler the other test tools use.
set_source_files_properties(${PROJECT_SOURCE_DIR}/tools/ling3_route_logits_test.cpp
  PROPERTIES LANGUAGE HIP)

# LS1 2026-10-03: the 2D register-tiled router-logits GEMM. NOT bit-identical by
# design (DECISIONS 2026-10-03 00:55) -- this reports max_abs / max_rel against the kernel
# it replaces and how many top-8 picks move per token, which is what can move a text.
# Compiled as HIP: it defines the reference __global__ kernel.
add_executable(superroc_ling3_route_gemm_test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/ling3_route_gemm_test.cpp)
target_link_libraries(superroc_ling3_route_gemm_test PRIVATE superroc_ling3)
set_source_files_properties(${PROJECT_SOURCE_DIR}/tools/ling3_route_gemm_test.cpp
  PROPERTIES LANGUAGE HIP)

# LS1 PHASE9 item 3: the token-count crossover between the 2D register-tiled router
# GEMM and the kernel it replaces. EXCLUDE_FROM_ALL: built by name.
add_executable(superroc_ling3_route_gemm_bench EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/ling3_route_gemm_bench.cpp)
target_link_libraries(superroc_ling3_route_gemm_bench PRIVATE superroc_ling3)
set_source_files_properties(${PROJECT_SOURCE_DIR}/tools/ling3_route_gemm_bench.cpp
  PROPERTIES LANGUAGE HIP)
endif()

# LS/LQ1 2026-10-01: tokenizer parity probe. Reads the GGUF vocabulary and
# nothing else -- no weights, no engine, no GPU -- and prints the ids
# LingTokenizer::Encode produces for a text file. Compiles the tokenizer and the
# GGUF reader directly instead of linking superroc_ling3, so the target never pulls
# in the ROCm backend or the HIP kernels. EXCLUDE_FROM_ALL: built by name.
add_executable(superroc_ling3_tokenize EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tools/ling3_tokenize_dump.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer.cpp
  ${PROJECT_SOURCE_DIR}/src/core/gguf_reader.cpp
  ${PROJECT_SOURCE_DIR}/src/core/quant/ggml_dequant.cpp)
target_include_directories(superroc_ling3_tokenize PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(superroc_ling3_tokenize PRIVATE ICU::uc)

# LM1: host test for the Ling MTP depth rule (depth_rule.hpp). CPU only.
add_executable(depth_rule_test EXCLUDE_FROM_ALL depth_rule_test.cpp)
target_compile_options(depth_rule_test PRIVATE -Wall -Wextra -Werror)
