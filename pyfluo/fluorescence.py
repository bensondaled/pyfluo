import numpy as np, pandas as pd
from scipy.stats.mstats import mode
from scipy.ndimage.measurements import label
import warnings, sys

from .util import ProgressBar
from .util import sliding_window
from .config import *

def compute_dff(data, percentile=5., window_size=3., step_size=None, method='pd', Ts=None, pad_kwargs=dict(mode='edge'), root_f=False, return_f0=False, verbose=True):
    """
    method: 'pd' / 'np'
    """
    Ts = Ts or data.Ts

    if not any([isinstance(data, p) for p in [pd.Series, pd.DataFrame]]):
        method = 'np'

    if method=='pd':
        _window = int(np.round(window_size / Ts))
        def f_dff(x):
            #xmode,_ = mode(x, axis=0)
            #return xmode[0]
            return np.percentile(x, percentile)
        f0 = data.rolling(window=_window, min_periods=1).apply(func=f_dff)
        dff = (data-f0)/f0
        return dff
    
    elif method=='np':
        if step_size == None:
            step_size = Ts

        window_size = int(window_size/Ts)
        step_size = int(step_size/Ts)

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
            #f0.append(np.percentile(win, percentile, axis=0))
            f0.append(mode(win, axis=0))
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

def detect_transients(sig, main_thresh=0.0001, peak_thresh=1.5, drift_window=10., width_lims=(0.040, 3.00), Ts=None):
    """
    TODO: this is not properly implemented, some things will fall through cracks, other will be false positive
    sig is a 1d signal
    width_lims in units of sig's Ts
    """

    Ts = Ts or sig.Ts

    if isinstance(sig, pd.DataFrame):
        res = sig.apply(detect_transients, axis=0, Ts=Ts)
        result = sig.copy() # trick to maintain other properties
        result[:] = res[:]
        return result
    
    wmin,wmax = np.round(np.asarray(width_lims) / Ts).astype(int)
    dwin = int(np.round(drift_window/Ts))

    sig = pd.Series(np.squeeze(np.asarray(sig)))

    def pin(x):
        return np.max(x)

    # find significant samples
    mean_signal = sig.rolling(window=dwin, min_periods=1).mean()
    thresh1 = mean_signal + main_thresh * sig.std(axis=0)
    thresh2 = mean_signal + peak_thresh * sig.std(axis=0)
    above_thresh1 = sig>thresh1
    peak_hoods = sig.rolling(window=wmax, min_periods=1).apply(pin).values # this step only speeds things up, not necessary
    potential = (above_thresh1) & (peak_hoods>thresh2)

    # label potential transients
    labelled,nlab = label(potential.values)

    # width and peak restrictions
    dummies = np.arange(1,nlab+1)
    ns_per_label = np.array([np.sum(labelled==l) for l in dummies])
    peaks_ok = np.array([np.max(sig[labelled==l])>=np.max(thresh2[labelled==l]) for l in dummies])
    lab_ids = dummies[(ns_per_label>=wmin) & (ns_per_label<=wmax) & (peaks_ok)]
    valid = pd.Series(labelled).isin(lab_ids).values

    new_sig = np.zeros_like(sig)
    new_sig[valid] = sig[valid]

    return new_sig
