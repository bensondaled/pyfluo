#http://www.johnvinyard.com/blog/?p=268
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

def sliding_window(a,ws,ss=1):
    """Generate sliding window version of an array

    Parameters
    ----------
    a (np.ndarray): input array
    ws (int): window size
    ss (int): step size

    Returns
    -------
    Array in which iteration along the 0th dimension provides requested data windows

    """
    # sliding window along 0'th axis of a, using stride tricks
    # ws: window size
    # ss: step size
    l = a.shape[0]
    n_slices = ((l - ws) // ss) + 1
    newshape = (n_slices,ws) + a.shape[1:] 
    newstrides = (a.strides[0]*ss,) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    return strided
