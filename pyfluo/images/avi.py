import numpy as np
import cv2

class AVI(object):
    """An object to store avi file data

    Parameters
    ----------
    file_path : str 
        path to avi file

    Attributes
    ----------
    data : np.ndarray
        the data, with shape (n,y,x)

    Notes
    -----
    Currently assumes greyscale images, using only 1 of 3 channels when loading
    """
    def __init__(self, file_path):
        self.file_path = file_path
        frs = []
        vc = cv2.VideoCapture(self.file_path)
        valid,fr = vc.read()
        while valid:
            frs.append(fr[:,:,0])
            valid,fr = vc.read()
        vc.release()
        self.data = np.array(frs)
