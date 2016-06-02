import numpy as np, pandas as pd
import warnings, sys

from .util import ProgressBar
from .util import sliding_window
from .config import *

def compute_dff(data, percentile=8., window_size=1., step_size=None, method='pd', pad_kwargs=dict(mode='edge'), root_f=False, return_f0=False, verbose=True):
    """
    method: 'pd' / 'np'
    """

    if not any([isinstance(data, p) for p in [pd.Series, pd.DataFrame]]):
        method = 'np'

    if method=='pd':
        _window = int(np.round(window_size / data.Ts))
        def f_dff(x):
            f0 = np.percentile(x, percentile)
            return f0
        f0 = data.rolling(window=_window, min_periods=1).apply(func=f_dff)
        dff = (data-f0)/f0
        return dff
    
    elif method=='np':
        if step_size == None:
            step_size = data.Ts

        window_size = int(window_size/data.Ts)
        step_size = int(step_size/data.Ts)

        if window_size<1:
            warnings.warn('Requested a window size smaller than sampling interval. Using sampling interval.')
            window_size = 1.
        if step_size<1:
            warnings.warn('Requested a step size smaller than sampling interval. Using sampling interval.')
            step_size = 1.

        pad_size = window_size - 1
        pad = ((pad_size,0),) + tuple([(0,0) for _ in xrange(data.ndim-1)])
        padded = np.pad(data, pad, **pad_kwargs)

        out_size = ((len(padded) - window_size) // step_size) + 1

        if verbose:
            pbar = ProgressBar(maxval=out_size).start()
        f0 = []
        for idx,win in enumerate(sliding_window(padded, ws=window_size, ss=step_size)):
            f0.append(np.percentile(win, percentile, axis=0))
            if verbose:
                pbar.update(idx)
        f0 = np.repeat(f0, step_size, axis=0)[:len(data)]
        if verbose:
            pbar.finish()

        if not root_f:
            bl = f0
        elif root_f:
            bl = np.sqrt(f0)

        ret = (data-f0)/bl

        if return_f0:
            return ( ret, f0 )
        else:
            return ret
