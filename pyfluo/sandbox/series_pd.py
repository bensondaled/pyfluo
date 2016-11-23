# External imports
import pandas as pd, numpy as np, matplotlib.pyplot as pl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import warnings

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

    def reset_time(self, offset=0, **kwargs):
        self.set_index(self.Ts*np.arange(len(self)) + offset, **kwargs)
    
    def plot(self, gap=0.1, order=None, names=None, cmap=pl.cm.viridis, legend=False, ax=None, color=None, stacked=True, binary_label=None, **kwargs):
        """
            gap : float
            order : list-like
            names : list-like
            cmap : mpl cmap
            stacked : bool
            binary_label : trues/falses of same shape as data
        """

        if ax is None:
            ax = pl.gca()
        
        if order is None:
            order = np.arange(self.shape[1])
        
        ycolors = cmap(np.linspace(0,1,self.shape[1]))[order]
        if color is not None:
            ycolors = color
            if isinstance(ycolors, PF_str_types):
                ycolors = [ycolors] * self.shape[1]

        if binary_label is not None:
            binary_label = binary_label.T.iloc[order,:].T.astype(bool).values

        to_plot = self.T.iloc[order,:].T.values
        if self.shape[1] == 1:
            stacked=False
        if stacked:
            to_plot = to_plot - to_plot.min(axis=0)
            tops = (to_plot.max(axis=0)).cumsum()
            to_add = np.append(0, tops[:-1]) + gap*np.arange(to_plot.shape[1])
            to_plot += to_add
            yticks = np.asarray(to_plot.mean(axis=0))
            if names is None:
                yticklab = np.array([str(i) for i in np.arange(self.shape[1])])[order]
            else:
                yticklab = names

        for idx,tp,color in zip(np.arange(to_plot.shape[1]),to_plot.T, ycolors):
            if binary_label is not None:
                dfx = np.asarray(self.index)
                dfy = tp
                points = np.array([dfx, dfy]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lcmap = ListedColormap([color,'r'])
                norm = BoundaryNorm([-1, -0.5, 0.5, 1], lcmap.N)
                lc = LineCollection(segments, cmap=lcmap, norm=norm)
                blab = binary_label.T[idx]
                lc.set_array((blab[1:] | blab[:-1]))
                ax.add_collection(lc)
            else:
                ax.plot(self.index, tp, color=color, **kwargs)

        if stacked:
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklab, ha='right')
            for i,c in zip(ax.get_yticklabels(), ycolors):
                i.set_color(c)

        ax.set_xlim([self.index.min(), self.index.max()])
        ax.set_ylim([to_plot.min(), to_plot.max()])

        return ax

    def heat(self, order=None, color_id_map=pl.cm.viridis, labels=True, ax=None, yfontsize=15, hlines=True, **kwargs):
        """
        labels: True, False/None, or list of labels, one for each column in data
        """
        if ax is None:
            ax = pl.gca()
        if order is None:
            order = np.arange(self.shape[1])
        if 'cmap' not in kwargs:
            kwargs['cmap'] = pl.cm.jet
        x = np.append(np.asarray(self.index), self.index[-1]+self.Ts)
        true_y = np.arange(self.shape[1])
        y = np.arange(self.shape[1]+1)-0.5

        if labels is True:
            ylab = [str(int(i)) for i in true_y[order]]
        elif isinstance(labels,list) or isinstance(labels,np.ndarray):
            ylab = labels
        elif not labels:
            ylab = None
        ycolors = color_id_map(np.linspace(0,1,self.shape[1]))[order]

        res = ax.pcolormesh(x, y, self.T.iloc[order,:], **kwargs)

        if hlines:
            ax.hlines(y, x[0], x[-1], color='w')
        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim([y[0], y[-1]])
        if ylab is not None:
            ax.set_yticks(true_y)
            ax.set_yticklabels(ylab, ha='right')
            for i,c in zip(ax.get_yticklabels(), ycolors):
                i.set_color(c)
                i.set_fontsize(yfontsize)
        return res

    def normalize(self, axis=0):
        copy = self.copy()
        copy = (copy-copy.min(axis=axis))/(copy.max(axis=axis)-copy.min(axis=axis))
        return copy


if __name__ == '__main__':
    pass

