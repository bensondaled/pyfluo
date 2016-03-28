import numpy as np
import h5py

class HDF5(object):
    """An object to store hdf5 movie data

    Parameters
    ----------
    file_path : str 
        path to hdf5 file

    Attributes
    ----------
    data : np.ndarray
        the data, with shape (n,y,x)

    Notes
    -----
    Currently assumes greyscale images, using only 1 of 3 channels when loading
    """
    def __init__(self, file_path, pbar=True):
        self.file_path = file_path
        vc = h5py.File(self.file_path, 'r')

        try:
            mov = vc['mov']
            ts = vc['ts']
        except:
            raise Exception('\'mov\' and/or \'ts\' fields not found in HDF5 file.')

        self.data = np.asarray(mov)
        vc.close()
