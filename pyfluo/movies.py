# External imports
import pandas as pd, numpy as np
import warnings, cv2
# Internal imports
from images import Tiff, AVI
from config import *

class Movie(pd.DataFrame):
    """Movie object
    """

    _metadata = ['Ts', 'frame_shape']
    _metadata_defaults = [None, None]

    def __init__(self, data, *args, **kwargs):

        # Set custom fields
        for md,dmd in zip(self._metadata, self._metadata_defaults):
            setattr(self, md, kwargs.pop(md, dmd))

        # Assess data type
        if isinstance(data, np.ndarray) and data.ndim==3:
            # 3d array passed as data, flatten frames
            if self.frame_shape is not None:
                warnings.warn('Frame shape parameter ignored because frame shape can be inferred from data.')
            self.frame_shape = data.shape[1:]
            data = data.reshape([data.shape[0], data.shape[1]*data.shape[2]])
        elif isinstance(data, Tiff):
            # Tiff object passed as data, extract a copy of its data attribute
            data = data.data.copy()

        # Prepare index if Ts supplied
        if self.Ts is not None:
            if 'index' in kwargs:
                warnings.warn('Ts parameter ignored because index was also supplied.')
            else:
                kwargs['index'] = self.Ts*np.arange(0, len(data))

        # Init object
        super(Movie, self).__init__(data, *args, **kwargs)

        # Custom property validation
        # Note that this is completely useless when a previously existing instance is being manipulated; the properties will be overwritten after this. But it serves as a validator for initial instantiation
        if self.Ts is None:
            self.Ts = np.mean(np.diff(self.index))
        if self.frame_shape is None:
            self.frame_shape = [1, self.shape[-1]]
        self.frame_shape = np.asarray(self.frame_shape)

    @property
    def _constructor(self):
        return Movie
    
    @property
    def _constructor_sliced(self):
        return pd.Series

    def project(self, method=np.mean, axis=0):
        proj = self.apply(method, axis=axis)
        pass

    def play(self, contrast=1.0, scale=1.0, show_time=True, font_size=1, fps=None):
        minn,maxx = self.values.min(),self.values.max()
        size = tuple((scale*self.frame_shape[::-1]).astype(int))
        if fps==None:
            fps = 1/self.Ts
        fpms = fps / 1000.

        for t,fr in self.iterrows():
            fr = fr.reshape(self.frame_shape)
            fr = contrast * (fr-minn)/(maxx-minn)
            fr[fr>1.0] = 1.0 #clips; this should become a parameter
            fr[fr<0.0] = 0.0 #clips; this should become a parameter
            fr = cv2.resize(fr,size)
            if show_time:
                cv2.putText(fr, '%0.3f'%(t), (5,int(30*font_size)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (120,100,80), thickness=1)
            cv2.imshow('Movie', fr)
            k=cv2.waitKey(int(1./fpms))
            if k == ord('q'):
                break
        cv2.destroyWindow('Movie')

if __name__ == '__main__':
    import tifffile
    data = tifffile.imread('/Users/ben/phd/data/2p/mov.tif')
    m = Movie(data, Ts=1/64)

