// ling3_all.hpp — the helper kernels this package uses, in the one include
// order that works.
//
// LK's ling3_mla.hpp uses ViewC and View without declaring them: they come from
// ling3_kda.hpp, so that header has to be included first. Verified on the Mac
// with LK's hipstub -- including ling3_mla.hpp on its own fails with
// "unknown type name 'ViewC'".
//
// Sources are copied into this package rather than linked across trees, per the
// build brief. They are LK's (KDA, MLA, ops, GEMV) and LX's (quant, router,
// expert FFN); both are referenced in the README.

#ifndef GUFO_LING3_ALL_HPP
#define GUFO_LING3_ALL_HPP

// The order matters: ling3_mla.hpp uses View and ViewC without declaring
// them, and they come from ling3_kda.hpp. Verified on the Mac with LK's
// hipstub -- including ling3_mla.hpp on its own fails with
// "unknown type name 'ViewC'"; this order makes it compile.
//
// NOTE: ling3_gemm.hpp is LK's and is currently absent (see
// MISSING_FROM_LK.md), so this header does not compile until it is restored.
// The ROCm backend is a stub that does not include it, so the package itself
// still builds in both configurations.

#include "ling3_gemm.hpp"
#include "ling3_kda.hpp"  // must precede ling3_mla.hpp: it defines View/ViewC
#include "ling3_mla.hpp"
#include "ling3_ops.hpp"

#endif  // GUFO_LING3_ALL_HPP
