# External imports
import pandas as pd, numpy as np, matplotlib.pyplot as pl
import warnings, cv2

# Internal imports
from .config import *

class Series(pd.DataFrame):
    """Series object
    """

    _metadata = ['Ts']
    _metadata_defaults = [1.0]

    def __init__(self, data, *args, **kwargs):

        # Set custom fields
        for md,dmd in zip(self._metadata, self._metadata_defaults):
            setattr(self, md, kwargs.pop(md, dmd))

        # Init object
        super(Series, self).__init__(data, *args, **kwargs)

    @property
    def _constructor(self):
        return Series
    
    @property
    def _constructor_sliced(self):
        return pd.Series

    def plot(self, *args, gap=0.1, **kwargs):

        # Overwrite default cmap
        if 'cmap' not in kwargs:
            kwargs['cmap'] = pl.cm.viridis

        # Overwrite meaning of "stacked," b/c something other than pandas implementation is desired
        stacked = kwargs.pop('stacked', False)
        if stacked:
            to_plot = self - self.min(axis=0)
            tops = (to_plot.max(axis=0)).cumsum()
            to_add = pd.Series(0).append( tops[:-1] ).reset_index(drop=True) + gap*np.arange(to_plot.shape[1])
            to_plot = to_plot + to_add
        else:
            to_plot = self

        super(Series, to_plot).plot(*args, **kwargs)

if __name__ == '__main__':
    pass

