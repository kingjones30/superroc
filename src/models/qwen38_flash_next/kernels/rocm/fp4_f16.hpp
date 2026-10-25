// SQ1: the FP4->F16 decode helpers moved to src/core/rocm_fp4/
// fp4_f16_prefill.hpp. This shim keeps existing Flash-Next includes
// working with no other change.
#pragma once
#include "src/core/rocm_fp4/fp4_f16_prefill.hpp"
