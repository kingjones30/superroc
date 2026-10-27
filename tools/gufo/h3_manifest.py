"""Import-compat shim: `gufo.h3_manifest` moved to the new engine name.

Kept so out-of-tree code importing the previous top-level package keeps
working. New code must import `superroc.h3_manifest` directly.
"""
import warnings as _warnings

_warnings.warn(
    "the gufo.h3_manifest module moved to superroc.h3_manifest; import superroc.h3_manifest instead",
    DeprecationWarning, stacklevel=2)
from superroc.h3_manifest import *
