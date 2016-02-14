from ts_base import TSBase
import numpy as np
import pylab as pl
import warnings, sys

_pyver = sys.version_info.major
if _pyver == 3:
    xrange = range

class Trace(TSBase):
    """An object to hold one or more data vectors along with a single time vector

    Parameters
    ----------
    data : np.ndarray
        input data, either 1d or 2d. Details below
    time : [optional] np.ndarray, list 
        time vector with n elements
    Ts : [optional] float
        sampling period
    info : [optional] list, np.ndarray
        info vector with n elements

    The data in Trace objects is stored following the standard pyfluo convention in which the 0th axis corresponds to time. For example, trace[0] corresponds to the data at time 0. 

    This object can be used to store multiple time series of data, for example the traces corresponding to multiple ROIs in a movie. In this case, the above convention still holds, and trace[0] will supply an n-length vector corresponding to n series of data at time 0.

    It should be noted that error checking with regard to the time vector is still under progress. All tested operations are functional, but thorough testing has not yet been performed.
    """
    def __new__(cls, data, **kwargs):
        return super(Trace, cls).__new__(cls, data, _ndim=[1,2], class_name='Trace', **kwargs)

    def take(self, *args, **kwargs):
        res = super(Trace, self).take(*args, **kwargs)

        # for special case of pulling out multiple segments from a single trace, return a new trace
        if len(np.unique(map(len,res))) == 1 and all([i.squeeze().ndim==1 for i in res]) and isinstance(res[0],self.__class__):
            t = res[0].time
            t -= t[0]
            Ts = res[0].Ts
            return self.__class__(np.asarray(res).squeeze().T, time=t.copy(), Ts=Ts)

        else:
            return res

    def as2d(self):
        """Return 2d version of object

        Useful because the object can in principle be 1d or 2d
        """
        if self.ndim == 1:
            return np.rollaxis(np.atleast_2d(self),-1)
        else:
            return self

    def normalize(self, minmax=(0., 1.), axis=0):
        """Normalize the data
        
        Parameters
        ----------
        minmax : list, tuple 
            ``[post_normalizaton_data_min, max]``
        axis : int 
            axis over which to normalize
            
        Returns
        -------
        A new Trace object, normalized
        """
        newmin,newmax = minmax
        omin,omax = self.min(axis=axis),self.max(axis=axis)
        newdata = (self-omin)/(omax-omin) * (newmax-newmin) + newmin
        return self.__class__(np.asarray(newdata),time=self.time,Ts=self.Ts,info=self.info)

    def plot(self, stacked=True, subtract_minimum=False, cmap=pl.cm.jet, **kwargs):
        """Plot the data
        
        Parameters
        ----------
        stacked : bool 
            for multiple columns of data, stack instead of overlaying
        subtract_minimum : bool
            subtract minimum from each individual trace
        cmap : matplotlib.LinearSegmentedColormap
            color map for display. Options are found in pl.colormaps(), and are accessed as pl.cm.my_favourite_map
        kwargs : dict
            any arguments accepted by matplotlib.plot

        Returns
        -------
        The matplotlib axes object corresponding to the data plot
        """
        d = self.copy()
        n = 1 #number of traces
        if len(d.shape)>1:
            n = d.shape[1]

        ax = pl.gca()

        if kwargs.get('color', None) is None:
            colors = cmap(np.linspace(0, 1, n))
            if _pyver == 2:
                ax.set_color_cycle(colors)
            elif _pyver == 3:
                from cycler import cycler
                ax.set_prop_cycle(cycler('c',colors))
        else:
            colors = np.array([kwargs['color'] for _ in range(n)])

        if subtract_minimum:
            d -= d.min(axis=0)
        if stacked and n>1:
            d += np.append(0, np.cumsum(d.max(axis=0))[:-1])
        ax.plot(self.time, d, **kwargs)
       
        # display trace labels along right
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(np.atleast_1d(d.mean(axis=0)))
        ax2.set_yticklabels([str(i) for i in xrange(n)], weight='bold')
        [l.set_color(c) for l,c in zip(ax2.get_yticklabels(), colors)]

        pl.gcf().canvas.draw()

        return ax
