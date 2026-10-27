"""Import-compat shim: `gufo.safetensors` moved to the new engine name.

Kept so out-of-tree code importing the previous top-level package keeps
working. New code must import `superroc.safetensors` directly.
"""
import warnings as _warnings

_warnings.warn(
    "the gufo.safetensors module moved to superroc.safetensors; import superroc.safetensors instead",
    DeprecationWarning, stacklevel=2)
from superroc.safetensors import *
