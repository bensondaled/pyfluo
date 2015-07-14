import numpy as np
import warnings
from util import sliding_window as sw
from util import ProgressBar

def compute_dff(data, percentile=8., window_size=1., step_size=.025, subtract_minimum=True, pad_mode='symmetric'):
    """Compute delta-f-over-f

    Computes the percentile-based delta-f-over-f along the 0th axis of the supplied data.

    Parameters
    ----------
    data (np.ndarray): n-dimensional data (DFF is taken over axis 0)
    percentile (float): percentile of data window to be taken as F0
    window_size (float): size of window to determine F0, in seconds
    step_size (float): size of steps used to determine F0, in seconds
    subtract_minimum (bool): substract minimum value from data before computing
    pad_mode (str): mode argument for np.pad, used to specify F0 determination at start of data

    Returns
    -------
    Data of the same shape as input, transformed to DFF
    """
    data = data.copy()

    window_size = int(window_size*data.fs)
    step_size = int(step_size*data.fs)

    if step_size<1:
        warnings.warn('Requested a step size smaller than sampling interval. Using sampling interval.')
        step_size = 1.

    if subtract_minimum:
        data -= data.min()
     
    pad_size = window_size - 1
    padded = np.pad(data, ((pad_size,0),(0,0),(0,0)), mode=pad_mode)

    out_size = ((len(padded) - window_size) // step_size) + 1
    pbar = ProgressBar(maxval=out_size).start()
    f0 = []
    for idx,win in enumerate(sw(padded, ws=window_size, ss=step_size)):
        f0.append(np.percentile(win, percentile, axis=0))
        pbar.update(idx)
    f0 = np.repeat(f0, step_size, axis=0)[:len(data)]
    pbar.finish()

    return (data-f0)/f0 
