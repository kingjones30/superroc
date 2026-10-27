"""Import-compat shim: `gufo.h3_quality` moved to the new engine name.

Kept so out-of-tree code importing the previous top-level package keeps
working. New code must import `superroc.h3_quality` directly.
"""
import warnings as _warnings

_warnings.warn(
    "the gufo.h3_quality module moved to superroc.h3_quality; import superroc.h3_quality instead",
    DeprecationWarning, stacklevel=2)
from superroc.h3_quality import *
