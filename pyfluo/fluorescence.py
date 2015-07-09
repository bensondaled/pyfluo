from scipy import stats
from scipy.ndimage import zoom
from scipy.signal import resample
import numpy as np


def compute_dff(sig, window_size=5., quantile=8., subtract_minimum=False):
    
    sig = sig.copy()

    if subtract_minimum:
        sig -= sig.min()

    n_frames,n_rows,n_cols = sig.shape
    out_len = int(sig.shape[0]*sig.fs/window_size)
    
    #elm_missing = int(np.ceil(n_frames*1./resamp_factor)*resamp_factor-n_frames)
    #pad = [np.floor(elm_missing/2.), np.ceil(elm_missing/2.)]
    #pad_param = pad + [0 for _ in sig.shape[1:]]
    #sig = np.pad(sig, pad_param, mode='reflect')
    #n_frames_padded,n_rows_padded,n_cols_padded = sig.shape
    
    sig = resample(sig, out_len, axis=0)
    sig = np.percentile(sig, quantile, axis=0);
    f0 = zoom(sig, [downsampfact ,1, 1],order=0, mode='constant', cval=0.0, prefilter=False)
    sig = (sig-f0)/f0
    return sig 
