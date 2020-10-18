import numpy as np
from ..config import cv2
from ..util import ProgressBar

class AVI(object):
    """An object to store avi file data

    Parameters
    ----------
    file_path : str 
        path to avi file
    pbar : bool
        progress bar for loading

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
        vc = cv2.VideoCapture(self.file_path)

        self.n_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.data = np.empty([self.n_frames, self.height, self.width], dtype=np.uint8)

        if pbar:
            pbar = ProgressBar(maxval=self.n_frames).start()

        idx = -1
        while True:
            valid,fr = vc.read()
            if not valid:
                break
            idx += 1
            self.data[idx] = fr[:,:,0]
            if pbar:
                pbar.update(idx)
        pbar.finish()
        assert idx == self.n_frames-1
        vc.release()
