import tifffile
import numpy as np

class Tiff(object):
    """An object to store tiff file data

    Parameters
    ----------
    file_path : str 
        path to tiff file, or list thereof

    Attributes
    ----------
    data : np.ndarray
        the data, with shape (n,y,x)

    """
    def __init__(self, file_path):
        self.file_path = file_path
        if isinstance(file_path, str):
            self.data = tifffile.imread(self.file_path)
        elif any([isinstance(file_path, t) for t in [np.ndarray,list]]):
            data = [tifffile.imread(f) for f in self.file_path if 'tif' in f]
            self.data = np.concatenate([d if d.ndim==3 else [d] for d in data], axis=0)
