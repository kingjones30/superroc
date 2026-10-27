"""Import-compat shim: `gufo.h3_rng` moved to the new engine name.

Kept so out-of-tree code importing the previous top-level package keeps
working. New code must import `superroc.h3_rng` directly.
"""
import warnings as _warnings

_warnings.warn(
    "the gufo.h3_rng module moved to superroc.h3_rng; import superroc.h3_rng instead",
    DeprecationWarning, stacklevel=2)
from superroc.h3_rng import *
