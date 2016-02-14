import sys

_pyver = sys.version_info.major
if _pyver == 3:
    xrange = range
