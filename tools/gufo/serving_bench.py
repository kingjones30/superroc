"""Import-compat shim: `gufo.serving_bench` moved to the new engine name.

Kept so out-of-tree code importing the previous top-level package keeps
working. New code must import `superroc.serving_bench` directly.
"""
import warnings as _warnings

_warnings.warn(
    "the gufo.serving_bench module moved to superroc.serving_bench; import superroc.serving_bench instead",
    DeprecationWarning, stacklevel=2)
from superroc.serving_bench import *
