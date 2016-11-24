#http://www.johnvinyard.com/blog/?p=268
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

def sliding_window(a, ws, ss=1, pad=True, pos='center', pad_kw=dict(mode='constant', constant_values=np.nan)):
    """Generate sliding window version of an array

    * note that padding and positioning are not implemented for step size (ss) other than 1

    Parameters
    ----------
    a (np.ndarray): input array
    ws (int): window size
    ss (int): step size
    pad (bool): maintain size of supplied array by padding resulting array
    pos (str): center / right / left, applies only if pad==True
    pad_kw (dict): kwargs for np.pad

    Returns
    -------
    Array in which iteration along the 0th dimension provides requested data windows

    """
    if pad:
        npad = ws-1
        if pos=='right':
            pad_size = (npad, 0)
        elif pos=='left':
            pad_size = (0, npad)
        elif pos=='center':
            np2 = npad/2.
            pad_size = (int(np.ceil(np2)), int(np.floor(np2))) # is there a more principled way to choose which end takes the bigger pad in the even window size scenario?
        pad_size = [pad_size] + [(0,0) for i in range(len(a.shape)-1)]
        a = np.pad(a, pad_size, **pad_kw)

    l = a.shape[0]
    n_slices = ((l - ws) // ss) + 1
    newshape = (n_slices,ws) + a.shape[1:] 
    newstrides = (a.strides[0]*ss,) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)

    return strided
