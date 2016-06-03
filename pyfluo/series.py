# External imports
import pandas as pd, numpy as np, matplotlib.pyplot as pl
import warnings, cv2

# Internal imports
from .config import *

class Series(pd.DataFrame):
    """Series object
    """

    _metadata = ['Ts']

    def __init__(self, data, *args, **kwargs):

        Ts = kwargs.pop('Ts', None)

        # Init object
        super(Series, self).__init__(data, *args, **kwargs)

        if hasattr(self, 'Ts'):
            self.set_index(self.Ts*np.arange(len(self)), inplace=True)
        elif Ts is not None:
            self.Ts = Ts
            self.set_index(self.Ts*np.arange(len(self)), inplace=True)

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
        kwargs['legend'] = kwargs.get('legend', False)

        super(Series, to_plot).plot(*args, **kwargs)

    def heat(self, color_id_map=pl.cm.viridis, **kwargs):
        x = np.append(np.asarray(self.index), self.index[-1]+self.Ts)
        true_y = np.arange(self.shape[1])
        y = np.arange(self.shape[1]+1)-0.5
        res = pl.pcolormesh(x, y, self.T, **kwargs)
        pl.hlines(y, x[0], x[-1], color='w')
        pl.xlim([x[0], x[-1]])
        pl.ylim([y[0], y[-1]])
        pl.yticks(true_y, [str(int(i)) for i in true_y], ha='right')
        for i,c in zip(pl.gca().get_yticklabels(), color_id_map(np.linspace(0,1,self.shape[1]))):
            i.set_color(c)
        return res

if __name__ == '__main__':
    pass

