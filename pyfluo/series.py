# External imports
import pandas as pd, numpy as np, matplotlib.pyplot as pl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import warnings

# Internal imports
from .config import *

class Series(np.ndarray):
    """Series object
    """

    __array_priority__ = 1. #ensures that ufuncs return ROI class instead of np.ndarrays
    _custom_attrs = {'Ts':1., 't0':0.}

    def __new__(cls, data, Ts=1., t0=0., **kwargs):

        obj = np.asarray(data, **kwargs).view(cls)
        if obj.ndim > 2:
            obj = np.squeeze(obj)
            if obj.ndim > 2:
                raise Exception('Series can be at most 2d.')

        obj.Ts = Ts
        obj.t0 = t0

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        for ca in self._custom_attrs:
            setattr(self, ca, getattr(obj, ca, self._custom_attrs[ca]))

    def __array_wrap__(self, out, context=None):
        return np.ndarray.__array_wrap__(self, out, context)

    def __getitem__(self,idx):

        # special handling of lists of slices: this is not normally allowed in numpy, but is commonly desired in this class
        # there are 2 instances that are being custom allowed that wouldn't otherwise be:
            # (1): myseries[[slice(), slice(), slice()]] -- normally this would be too many axes, but it will instead interpret it as multiple slices of first axis
            # (2): myseries[(slice(), slice(),), 0] -- normally this is not allowed, but will be here. only applies to 2d series
                # (2a) the one above
                # (2b) myseries[(slice(), slice()), 0:10]
                # (2c) myseries[(slice(), slice()), [0,5,9]]
        # importantly, case (1) *must* use list or ndarray, not tuple
            # (reason is:) with tuple, it's underdetermined; if that tuple contains 2 slices, it is indistinguishible from a normal attempt to access the 2 axes of this array
            # (this arises from the fact that python cannot distinguish x[1,2] from x[(1,2)])
        # (case (1) is actually a special case of (2b), where the second index is inferred to be ":")
        # conditions: must supply a list-like, must all be slices, and none can be boundless slices (i.e. where None is one of the bounds)

        # case (1)
        if isinstance(idx, (list,np.ndarray)) and all([isinstance(item,slice) for item in idx]) and not any([s.start is None or s.stop is None for s in idx]):
            slice_len_0 = idx[0].stop - idx[0].start
            if not all([s.stop-s.start == slice_len_0 for s in idx]):
                raise Exception('Supplied list of slices to index Series, but slices are not all of the same length.')
            result = np.array([self[sli] for sli in idx]) # now 3d so no longer in the purview of this class
            return result

        # case (2) -- only relevant to 2d Series
        if self.ndim==2 and isinstance(idx, tuple) and isinstance(idx[0], PF_list_types) and all([isinstance(item,slice) for item in idx[0]]) and not any([s.start is None or s.stop is None for s in idx[0]]):
            sls = idx[0] # these are the time slices requested
            # (2a)
            if isinstance(idx[1], PF_numeric_types):
                i1 = idx[1] # this is the specific subseries requested
                result = Series(np.array([self[sl,i1] for sl in sls]).T)
                for ca in self._custom_attrs:
                    setattr(result, ca, getattr(self, ca, self._custom_attrs[ca]))
                return result
            # (2b)
            elif isinstance(idx[1], slice):
                i1s = idx[1] # this is the slice of desired subseries
                result = np.array([self[sl,i1s] for sl in sls]) # now 3d, out of purview of this class
                return result
            # (2c)
            elif isinstance(idx[1], PF_list_types):
                i1s = idx[1] # this is the list of desired subseries
                result = np.array([self[sl,i1s] for sl in sls]) # now 3d, out of purview of this class
                return result
        
        # special handling of attempts to treat a 1d series as 2d -- this is another custom-allowed feature
        if self.ndim==1 and isinstance(idx, tuple):
            if len(idx) > 2:
                pass # let the superclass handle it
            elif len(idx) == 2:
                # first dimension is the time index, second refers to our nonexistent second axis
                # if the second item is None, then it's the special numpy convention, pass it through
                if idx[1] is None:
                    pass
                else:
                    # now the only acceptable indices for the second position are "0" or ":"
                    if idx[1] not in [0, slice(None,None,None)]:
                        raise Exception('This series object is 1D and thus index exceeds bounds.')
                    idx = idx[0]

        out = super(Series,self).__getitem__(idx)

        # cases where a pf Series should not be returned
        if not isinstance(out,Series):
            return out
        if isinstance(idx, PF_numeric_types):
            return out.view(np.ndarray)

        ## when it used to be that all pf series are by definition 2d
        #if out.ndim == 1:
        #    out = np.atleast_2d(out).T

        return out

    @property
    def index(self):
        return np.arange(len(self)) * self.Ts + self.t0

    def as_2d(self):
        if self.ndim==2:
            return self
        elif self.ndim==1:
            return np.atleast_2d(self).T

    def _cls_chk(self, res):
        if isinstance(res, np.ndarray) and res.ndim>0:
            res = Series(res)
            for ca in self._custom_attrs:
                setattr(res, ca, getattr(self, ca, self._custom_attrs[ca]))
        else:
            pass
        return res

    def mean(self, **kwargs):
        return self._cls_chk( np.nanmean(self.view(np.ndarray), **kwargs) )
    def std(self, **kwargs):
        return self._cls_chk( np.nanstd(self.view(np.ndarray), **kwargs) )
    def median(self, **kwargs):
        return self._cls_chk( np.nanmedian(self.view(np.ndarray), **kwargs) )
    def sem(self, **kwargs):
        axis = kwargs.get('axis', None)
        if axis is None:
            n = np.size(self)
        else:
            n = self.shape[axis]
        return self.std(**kwargs) / np.sqrt(n)

    def plot(self, gap=0.1, order=None, names=None, cmap=pl.cm.viridis, legend=False, ax=None, color=None, stacked=True, binary_label=None, use_index=True, **kwargs):
        """
            gap : float
            order : list-like
            names : list-like
            cmap : mpl cmap
            stacked : bool
            binary_label : trues/falses of same shape as data
            use_index : use Ts to determine index
        """

        to_plot = self.as_2d()
        if use_index:
            tp_idx = self.index
        else:
            tp_idx = np.arange(len(to_plot))

        if ax is None:
            ax = pl.gca()
        
        if order is None:
            order = np.arange(to_plot.shape[1])
        
        ycolors = cmap(np.linspace(0,1,to_plot.shape[1]))[order]
        if color is not None:
            ycolors = color
            if isinstance(ycolors, PF_str_types):
                ycolors = [ycolors] * to_plot.shape[1]

        if to_plot.shape[1] == 1: # use default color cycle if only plotting one line
            ycolor = None 

        if binary_label is not None:
            binary_label = binary_label[:,order].astype(bool)

        to_plot = to_plot[:,order]
        if to_plot.shape[1] == 1:
            stacked=False
        if stacked:
            to_plot = to_plot - to_plot.min(axis=0)
            tops = (to_plot.max(axis=0)).cumsum()
            to_add = np.append(0, tops[:-1]) + gap*np.arange(to_plot.shape[1])
            to_plot += to_add
            yticks = np.asarray(to_plot.mean(axis=0))
            if names is None:
                yticklab = np.array([str(i) for i in np.arange(to_plot.shape[1])])[order]
            else:
                yticklab = names

        for idx,tp,color in zip(np.arange(to_plot.shape[1]),to_plot.T, ycolors):
            if binary_label is not None:
                dfx = np.asarray(tp_idx)
                dfy = tp
                points = np.array([dfx, dfy]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lcmap = ListedColormap([color,'r'])
                norm = BoundaryNorm([-1, -0.5, 0.5, 1], lcmap.N)
                lc = LineCollection(segments, cmap=lcmap, norm=norm)
                blab = binary_label[:,idx]
                lc.set_array((blab[1:] | blab[:-1]))
                ax.add_collection(lc)
            else:
                ax.plot(tp_idx, tp, color=color, **kwargs)

        if stacked:
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklab, ha='right')
            for i,c in zip(ax.get_yticklabels(), ycolors):
                i.set_color(c)

        ax.set_xlim([tp_idx.min(), tp_idx.max()])
        ax.set_ylim([to_plot.min(), to_plot.max()])

        return ax

    def heat(self, order=None, color_id_map=pl.cm.viridis, labels=True, ax=None, yfontsize=15, hlines=False, **kwargs):
        """
        labels: True, False/None, or list of labels, one for each column in data
        """
        to_plot = self.as_2d()

        if ax is None:
            ax = pl.gca()
        if order is None:
            order = np.arange(to_plot.shape[1])
        if 'cmap' not in kwargs:
            kwargs['cmap'] = pl.cm.inferno
        x = np.append(np.asarray(self.index), self.index[-1]+self.Ts)
        true_y = np.arange(to_plot.shape[1])
        y = np.arange(to_plot.shape[1]+1)-0.5

        if labels is True:
            ylab = [str(int(i)) for i in true_y[order]]
        elif isinstance(labels,list) or isinstance(labels,np.ndarray):
            ylab = labels
        elif not labels:
            ylab = None
        ycolors = color_id_map(np.linspace(0,1,to_plot.shape[1]))[order]

        res = ax.pcolormesh(x, y, to_plot.T[order,:], **kwargs)

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

