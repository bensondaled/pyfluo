import numpy as np, pandas as pd
import warnings, sys

from .util import ProgressBar
from .config import *

def compute_dff(data, percentile=8., window=1.):
    # TODO: incorporate option for 3d np array (using old method, pandas breaks down for this)

    _window = int(np.round(window / data.Ts))
    def f_dff(x):
        f0 = np.percentile(x, percentile)
        return (x[-1]-f0)/f0
    dff = pd.rolling_apply(data, window=_window, func=f_dff, min_periods=1)
    return dff

