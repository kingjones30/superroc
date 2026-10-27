"""Import-compat shim: `gufo.gguf` moved to the new engine name.

Kept so out-of-tree code importing the previous top-level package keeps
working. New code must import `superroc.gguf` directly.
"""
import warnings as _warnings

_warnings.warn(
    "the gufo.gguf module moved to superroc.gguf; import superroc.gguf instead",
    DeprecationWarning, stacklevel=2)
from superroc.gguf import *
