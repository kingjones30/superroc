"""Import-compat shim: `gufo.h3_profile_report` moved to the new engine name.

Kept so out-of-tree code importing the previous top-level package keeps
working. New code must import `superroc.h3_profile_report` directly.
"""
import warnings as _warnings

_warnings.warn(
    "the gufo.h3_profile_report module moved to superroc.h3_profile_report; import superroc.h3_profile_report instead",
    DeprecationWarning, stacklevel=2)
from superroc.h3_profile_report import *
