import sys, cv2, numpy as np

_pyver = sys.version_info.major

if _pyver == 3:
    xrange = range
    PF_str_types = [str]

elif _pyver == 2:
    PF_str_types = [str, unicode]

CV_VERSION = int(cv2.__version__[0])

PF_numeric_types = [int, float, np.float16, np.float32, np.float64, np.float128, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
