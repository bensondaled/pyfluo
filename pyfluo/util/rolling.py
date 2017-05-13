import numpy as np

from .window import sliding_window
from . import ProgressBar
                
def rolling_correlation(arr, win, verbose=True):
    """
    Computes the rolling correlation along axis 0 of `arr`, in steps of size `win`
    Returns a single value for each element along axis 0 of `arr`
    This value is the mean of the triangle of the correlation matrix (coefficient if elements are 2xN, else average coefficient)
    """
    sliding = sliding_window(arr, win)

    def cc(a):
        cmat = np.corrcoef(a.T)
        mask = np.tril(np.ones_like(cmat))==0
        return np.mean(cmat[mask])

    if verbose:
        pbar = ProgressBar(maxval=len(sliding)).start()

    _rollcor = []
    for idx,i in enumerate(sliding):
        _rollcor.append(cc(i))
        if verbose:
            pbar.update(idx)

    if verbose:
        pbar.finish()

    _rollcor = np.array(_rollcor)
    return _rollcor

