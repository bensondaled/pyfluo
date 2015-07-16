import pims
import numpy as np

class Tiff(object):
    """An object to store tiff file data

    Parameters
    ----------
    file_path : str 
        path to tiff file

    Attributes
    ----------
    data : np.ndarray
        the data, with shape (n,y,x)

    """
    def __init__(self, file_path):
        self.file_path = file_path
        with pims.open(self.file_path) as tif_file:
            self.data = np.swapaxes(np.array(tif_file, dtype=np.float32), 1, 2)[:,:,::-1]
