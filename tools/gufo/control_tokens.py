"""Import-compat shim: `gufo.control_tokens` moved to the new engine name.

Kept so out-of-tree code importing the previous top-level package keeps
working. New code must import `superroc.control_tokens` directly.
"""
import warnings as _warnings

_warnings.warn(
    "the gufo.control_tokens module moved to superroc.control_tokens; import superroc.control_tokens instead",
    DeprecationWarning, stacklevel=2)
from superroc.control_tokens import *
