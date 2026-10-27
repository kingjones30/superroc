"""Import-compat shim: `gufo.model_bench.__init__` moved to the new engine name.

Kept so out-of-tree code importing the previous top-level package keeps
working. New code must import `superroc.model_bench.__init__` directly.
"""
import warnings as _warnings

_warnings.warn(
    "the gufo.model_bench.__init__ module moved to superroc.model_bench.__init__; import superroc.model_bench.__init__ instead",
    DeprecationWarning, stacklevel=2)
from superroc.model_bench.__init__ import *
