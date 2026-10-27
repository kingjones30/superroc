"""Import-compat shim: `gufo` moved to the new engine name.

Kept so out-of-tree code importing the previous top-level package keeps
working. New code must import `superroc` directly.
"""
import warnings as _warnings

_warnings.warn(
    "the gufo module moved to superroc; import superroc instead",
    DeprecationWarning, stacklevel=2)
from superroc import *
