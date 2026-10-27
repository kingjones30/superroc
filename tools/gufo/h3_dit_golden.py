"""Import-compat shim: `gufo.h3_dit_golden` moved to the new engine name.

Kept so out-of-tree code importing the previous top-level package keeps
working. New code must import `superroc.h3_dit_golden` directly.
"""
import warnings as _warnings

_warnings.warn(
    "the gufo.h3_dit_golden module moved to superroc.h3_dit_golden; import superroc.h3_dit_golden instead",
    DeprecationWarning, stacklevel=2)
from superroc.h3_dit_golden import *
