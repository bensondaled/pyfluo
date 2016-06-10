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
        else:
            self.Ts = 1.0

    def _wrapped_pandas_method(self, mtd, *args, **kwargs):
        val = getattr(super(Series, self), mtd)(*args, **kwargs)
        if isinstance(val, pd.Series) or isinstance(val, Series):
            val.__class__ = Series
            for name in self._metadata:
                setattr(val, name, getattr(self, name))
        return val

    def __add__(self, arg):
        return self._wrapped_pandas_method('__add__', arg)
    def __radd__(self, arg):
        return self._wrapped_pandas_method('__radd__', arg)
    def __sub__(self, arg):
        return self._wrapped_pandas_method('__sub__', arg)
    def __rsub__(self, arg):
        return self._wrapped_pandas_method('__rsub__', arg)
    def __div__(self, arg):
        return self._wrapped_pandas_method('__div__', arg)
    def __rdiv__(self, arg):
        return self._wrapped_pandas_method('__rdiv__', arg)
    def __mul__(self, arg):
        return self._wrapped_pandas_method('__mul__', arg)
    def __rmul__(self, arg):
        return self._wrapped_pandas_method('__rmul__', arg)

    @property
    def _constructor(self):
        return Series
    
    def __finalize__(self, other, method=None, **kwargs):
        for name in self._metadata:
            object.__setattr__(self, name, getattr(other, name, None))
        return self

    def plot(self, *args, gap=0.1, **kwargs):

        # Overwrite default cmap
        if 'cmap' not in kwargs:
            kwargs['cmap'] = pl.cm.viridis
        kwargs['legend'] = kwargs.get('legend', False)

        # Overwrite meaning of "stacked," b/c something other than pandas implementation is desired
        stacked = kwargs.pop('stacked', False)
        if stacked:
            to_plot = self - self.min(axis=0)
            tops = (to_plot.max(axis=0)).cumsum()
            to_add = pd.Series(0).append( tops[:-1] ).reset_index(drop=True) + gap*np.arange(to_plot.shape[1])
            to_plot = to_plot + to_add
            yticks = np.asarray(to_plot.mean(axis=0))
            yticklab = [str(i) for i in np.arange(self.shape[1])]
        else:
            to_plot = self

        ax = super(Series, to_plot).plot(*args, **kwargs)
        
        if stacked:
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklab)

        return ax

    def heat(self, order=None, color_id_map=pl.cm.viridis, ax=None, **kwargs):
        if ax is None:
            ax = pl.gca()
        if order is None:
            order = np.arange(self.shape[1])
        x = np.append(np.asarray(self.index), self.index[-1]+self.Ts)
        true_y = np.arange(self.shape[1])
        y = np.arange(self.shape[1]+1)-0.5
        ylab = [str(int(i)) for i in true_y[order]]
        ycolors = color_id_map(np.linspace(0,1,self.shape[1]))[order]

        res = ax.pcolormesh(x, y, self.T.iloc[order,:], **kwargs)

        ax.hlines(y, x[0], x[-1], color='w')
        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim([y[0], y[-1]])
        ax.set_yticks(true_y)
        ax.set_yticklabels(ylab, ha='right')
        for i,c in zip(ax.get_yticklabels(), ycolors):
            i.set_color(c)
        return res

if __name__ == '__main__':
    pass

