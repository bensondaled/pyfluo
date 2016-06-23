import numpy as np
import h5py

class HDF5(object):
    """An object to store hdf5 movie data
    Currently does not take advantage of memory mapping, so loads entire mov field. I.e. don't supply large movies

    Parameters
    ----------
    file_path : str 
        path to hdf5 file containing at least the following 2 fields:
            mov : 3d array of frames
            ts : time stamps

    Attributes
    ----------
    data : np.ndarray
        the data, with shape (n,y,x)

    Notes
    -----
    Currently assumes greyscale images, using only 1 of 3 channels when loading
    """
    def __init__(self, file_path, reslice=slice(None,None)):
        self.file_path = file_path
        vc = h5py.File(self.file_path, 'r')

        try:
            mov = vc['mov'][reslice]
        except:
            raise Exception('\'mov\' and/or \'ts\' fields not found in HDF5 file.')

        try:
            ts = vc['ts'][reslice]
        except:
            ts = np.arange(len(mov))

        ts = np.asarray(ts)
        if ts.ndim == 2:
            ts = ts[:,0]
        self.Ts = np.mean(np.diff(ts))
        if np.std(np.diff(ts)) / self.Ts > 1.0: # if coeffcient of variation is greater than threshold (arbitrarily 1 right now), frame rate is inconsistent
            warnings.warn('Frame rate is inconsistent.')
        self.data = np.asarray(mov)
        vc.close()
