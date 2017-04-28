from __future__ import print_function
import sys, os, warnings, numpy as np

# cv2
try:
    import cv2
except ImportError:
    cv2 = None
    warnings.warn('cv2 not detected, expect strange behaviours.')

# Versions
_pyver = sys.version_info.major
if cv2:
    CV_VERSION = int(cv2.__version__[0])

# Types
if _pyver == 3:
    xrange = range
    PF_str_types = (str,)
elif _pyver == 2:
    PF_str_types = (str, unicode)
    input = raw_input
PF_numeric_types = tuple([int, float] + np.sctypes['float'] + np.sctypes['int'] + np.sctypes['uint'])
PF_list_types = (list, np.ndarray, tuple)
PF_pyver = _pyver

"""
# User settings
def _parse_settings(stgs):
    stgs = stgs.split('\n')
    stgs = [s.strip(' ') for s in stgs]
    stgs = [s for s in stgs if not s.startswith('#')]
    stgs = [s for s in stgs if ':' in s]
    stgs = [s.split(':') for s in stgs]
    stgs = [s for s in stgs if len(s) == 2]
    stgs = [[si.strip(' ') for si in s] for s in stgs]
    stgs = {'PF_'+s[0]:s[1] for s in stgs}
    return stgs
_home_dir = os.getenv('HOME', default='.')
_rc = os.path.join(_home_dir, '.pyfluorc')
if not os.path.exists(_rc):
    with open(_rc, 'a') as _f:
        pass
with open(_rc, 'r') as _f:
    _settings = _f.read()
_settings = _parse_settings(_settings)
globals().update(_settings)
"""
