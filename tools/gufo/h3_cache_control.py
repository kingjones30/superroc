"""Import-compat shim: `gufo.h3_cache_control` moved to the new engine name.

Kept so out-of-tree code importing the previous top-level package keeps
working. New code must import `superroc.h3_cache_control` directly.
"""
import warnings as _warnings

_warnings.warn(
    "the gufo.h3_cache_control module moved to superroc.h3_cache_control; import superroc.h3_cache_control instead",
    DeprecationWarning, stacklevel=2)
from superroc.h3_cache_control import *
