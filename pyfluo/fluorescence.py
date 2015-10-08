#TODO: choose a df/f method

import numpy as np
import warnings
from util import sliding_window as sw
from util import ProgressBar

def compute_dff(data, percentile=8., window_size=1., step_size=None, root_f=False, subtract_minimum=False, pad_mode='edge', in_place=False, return_f0=False, prog_bar=True):
    """Compute delta-f-over-f

    Computes the percentile-based delta-f-over-f along the 0th axis of the supplied data.

    Parameters
    ----------
    data : np.ndarray
        n-dimensional data (DFF is taken over axis 0)
    percentile : float 
        percentile of data window to be taken as F0
    window_size : float
        size of window to determine F0, in seconds
    step_size : float
        size of steps used to determine F0, in seconds. Defaults to sampling interval.
    root_f : bool:
        normalize by sqrt(f0) as opposed to f0
    subtract_minimum : bool
        subtract minimum value from data before computing
    pad_mode : str 
        mode argument for np.pad, used to specify F0 determination at start of data
    in_place : bool
        perform operation in place on supplied array
    return_f0 : bool
        return f0 as second return value
    prog_bar : bool
        show progress
        
    Returns
    -------
    Data of the same shape as input, transformed to DFF
    F0 (if return_f0==True)
    """
    if not in_place:
        data = data.copy()

    if step_size == None:
        step_size = data.Ts

    window_size = int(window_size*data.fs)
    step_size = int(step_size*data.fs)
    
    if window_size<1:
        warnings.warn('Requested a window size smaller than sampling interval. Using sampling interval.')
        window_size = 1.
    if step_size<1:
        warnings.warn('Requested a step size smaller than sampling interval. Using sampling interval.')
        step_size = 1.

    if subtract_minimum:
        data -= data.min()
     
    pad_size = window_size - 1
    pad = ((pad_size,0),) + tuple([(0,0) for _ in xrange(data.ndim-1)])
    padded = np.pad(data, pad, mode=pad_mode)

    out_size = ((len(padded) - window_size) // step_size) + 1
    if prog_bar:    pbar = ProgressBar(maxval=out_size).start()
    f0 = []
    for idx,win in enumerate(sw(padded, ws=window_size, ss=step_size)):
        f0.append(np.percentile(win, percentile, axis=0))
        if prog_bar:    pbar.update(idx)
    f0 = np.repeat(f0, step_size, axis=0)[:len(data)]
    if prog_bar:    pbar.finish()
   
    if not root_f:
        bl = f0
    elif root_f:
        bl = np.sqrt(f0)

    ret = (data-f0)/bl
    if return_f0:
        return ( ret, f0 )
    else:
        return ret

