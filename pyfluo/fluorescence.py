import numpy as np, pandas as pd
from scipy.stats.mstats import mode
from scipy.ndimage.measurements import label
from scipy.ndimage.filters import median_filter
import warnings, sys

from .series import Series
from .util import ProgressBar
from .util import sliding_window
from .config import *

def causal_median_filter(sig, size):
    # works, but be careful, b/c using a causal filter can make pseudo-shifts in the apparent time course of the signal
    from scipy.signal import medfilt
    kernel = [size] + [1]*(sig.ndim-1)
    filt = medfilt(sig, kernel)
    filt = filt[size//2:-size//2+1]
    pad = sig[:size-1]
    return np.concatenate([pad, filt])

def compute_dff(data, window_size=5., filter_size=1., step_size=None, Ts=None, pad_kwargs=dict(mode='edge'), root_f=False, return_f0=False, verbose=True):
    """
    pad_kwargs can be an array to use as left-pad instead of using np.pad. in this case, give the *raw data* to use as left pad
    """
    Ts = Ts or data.Ts

    was_pd = any([isinstance(data, t) for t in [pd.Series, pd.DataFrame]])
    if was_pd:
        orig_type = type(data)
        data = data.values.squeeze()

    if step_size is None:
        step_size = Ts

    window_size = int(window_size/Ts)
    step_size = int(step_size/Ts)
    filter_size = int(filter_size/Ts)
    if filter_size % 2 == 0:
        filter_size += 1
    filter_kernel = [filter_size] + [1]*(data.ndim-1)

    if window_size<1:
        warnings.warn('Requested a window size smaller than sampling interval. Using sampling interval.')
        window_size = 1.
    if step_size<1:
        warnings.warn('Requested a step size smaller than sampling interval. Using sampling interval.')
        step_size = 1.

    assert window_size < len(data), 'Window size is >= data size'
    assert step_size < len(data), 'Step size is >= data size'

    if isinstance(pad_kwargs, dict):
        pad_size = window_size - 1
        pad = ((pad_size,0),) + tuple([(0,0) for _ in range(data.ndim-1)])
        padded = np.pad(data, pad, **pad_kwargs)
    elif any([isinstance(pad_kwargs, dt) for dt in [np.ndarray, pd.Series, pd.DataFrame]]):
        if any([isinstance(pad_kwargs, t) for t in [pd.Series, pd.DataFrame]]):
            pad_kwargs = pad_kwargs.values.squeeze()
        assert len(pad_kwargs) >= window_size, 'Not enough padding was supplied.'
        pad = pad_kwargs[-window_size+1:]
        padded = np.concatenate([pad, data])

    out_size = ((len(padded) - window_size) // step_size) + 1
    
    if verbose:
        pbar = ProgressBar(maxval=out_size).start()

    padded = median_filter(padded, filter_kernel)

    f0 = []
    for idx,win in enumerate(sliding_window(padded, ws=window_size, ss=step_size)):
        f0.append(np.min(win, axis=0))
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

    if was_pd:
        ret = orig_type(ret, Ts=Ts)

    if return_f0:
        return ( ret, f0 )
    else:
        return ret

def detect_transients(sig, baseline_thresh=0.0, peak_thresh=2.5, drift_window=10., width_lims=(0.100, 3.00), Ts=None):
    """
    width_lims in units of sig's Ts

    baseline_thresh: how far down the signal must fall, specified as number of std's above or below mean
    peak_thresh: how high the signal must peak, even if just for 1 frame, also as n std's above mean
    drift_window: the sliding window to determine local means and stds for above thresholds
    """

    Ts = Ts or sig.Ts

    if isinstance(sig, pd.DataFrame):
        res = sig.apply(detect_transients, axis=0, Ts=Ts, baseline_thresh=baseline_thresh, peak_thresh=peak_thresh, drift_window=drift_window, width_lims=width_lims)
        result = sig.copy() # trick to maintain other properties
        result[:] = res[:]
        return result

    wmin,wmax = np.round(np.asarray(width_lims) / Ts).astype(int)
    dwin = int(np.round(drift_window/Ts))

    sig = pd.Series(np.squeeze(np.asarray(sig)))
    mean_signal = sig.rolling(window=dwin, min_periods=1).median().values
    std_signal = sig.rolling(window=dwin, min_periods=1).std().values

    bl_thresh = mean_signal + baseline_thresh*std_signal
    pk_thresh = mean_signal + peak_thresh*std_signal

    above = sig>pk_thresh
    below = sig<=bl_thresh

    labelled,nlab = label(above)

    borders = []
    for l in np.arange(1,nlab):
        aw = np.argwhere(labelled==l).T[0]
        first,last = aw[0],aw[-1]

        downbefore = np.argwhere(below[:first][::-1])
        if len(downbefore)==0:
            continue
        downbefore = first-np.min(downbefore)
        
        downafter = np.argwhere(below[last+1:])
        if len(downafter)==0:
            continue
        downafter = last+1+np.min(downafter)

        borders.append((downbefore, downafter))

    ret = np.zeros(sig.shape)
    for b in borders:
        if b[1]-b[0]<wmin or b[1]-b[0]>wmax:
            continue
        ret[b[0]:b[1]] = sig[b[0]:b[1]]
    return ret
