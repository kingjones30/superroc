"""Import-compat shim: `gufo.h3_diffusers_forward` moved to the new engine name.

Kept so out-of-tree code importing the previous top-level package keeps
working. New code must import `superroc.h3_diffusers_forward` directly.
"""
import warnings as _warnings

_warnings.warn(
    "the gufo.h3_diffusers_forward module moved to superroc.h3_diffusers_forward; import superroc.h3_diffusers_forward instead",
    DeprecationWarning, stacklevel=2)
from superroc.h3_diffusers_forward import *
