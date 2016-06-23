import sys, cv2, os, numpy as np

# Versions
_pyver = sys.version_info.major
CV_VERSION = int(cv2.__version__[0])

# Types
if _pyver == 3:
    xrange = range
    PF_str_types = [str]
elif _pyver == 2:
    PF_str_types = [str, unicode]
PF_numeric_types = [int, float] + np.sctypes['float'] + np.sctypes['int'] + np.sctypes['uint']

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
