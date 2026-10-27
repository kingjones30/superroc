"""Import-compat shim: `gufo.model_bench.render` moved to the new engine name.

Kept so out-of-tree code importing the previous top-level package keeps
working. New code must import `superroc.model_bench.render` directly.
"""
import warnings as _warnings

_warnings.warn(
    "the gufo.model_bench.render module moved to superroc.model_bench.render; import superroc.model_bench.render instead",
    DeprecationWarning, stacklevel=2)
from superroc.model_bench.render import *
