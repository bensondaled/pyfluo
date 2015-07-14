from ts_base import TSBase
import numpy as np
import pylab as pl
import warnings

class Trace(TSBase):
    """An object to hold one or more data vectors along with a single time vector

    Parameters
    ----------
    data (np.ndarray): input data, either 1d or 2d. Details below
    time [optional] (np.ndarray / list): time vector with n elements
    Ts [optional] (float): sampling period
    info [optional] (list / np.ndarray): info vector with n elements

    The data in Trace objects is stored following the standard pyfluo convention in which the 0th axis corresponds to time. For example, trace[0] corresponds to the data at time 0. 

    This object can be used to store multiple time series of data, for example the traces corresponding to multiple ROIs in a movie. In this case, the above convention still holds, and trace[0] will supply an n-length vector corresponding to n series of data at time 0.

    It should be noted that error checking with regard to the time vector is still under progress. All tested operations are functional, but thorough testing has not yet been performed.
    """
    def __new__(cls, data, **kwargs):
        return super(Trace, cls).__new__(cls, data, n_dims=[1,2], **kwargs)

    def as2d(self):
        """Return 2d version of object

        Useful because the object can in principle be 1d or 2d
        """
        if self.ndim == 1:
            return np.rollaxis(np.atleast_2d(self),-1)
        else:
            return self

    def normalize(self, minmax=(0., 1.), axis=None):
        """Normalize the data
        
        Parameters
        ----------
        minmax (list / tuple): ``[post_normalizaton_data_min, max]``
        axis (int): axis over which to normalize
            
        Returns
        -------
        A new Trace object, normalized
        """
        newmin,newmax = minmax
        omin,omax = self.min(axis=axis),self.max(axis=axis)
        new_obj = self.copy() 
        new_obj.data = (self-omin)/(omax-omin) * (newmax-newmin) + newmin
        return new_obj

    def plot(self, stacked=True, gap_fraction=0.08, **kwargs):
        """Plot the data
        
        Parameters
        ----------
        stacked (bool): for multiple series of data, stack instead of overlaying.
        gap_fraction (float): if ``stacked==True``, specifies the spacing between curves as a fraction of the average range of the curves
        kwargs: any arguments accepted by matplotlib.plot
        """
        gap = gap_fraction * np.mean(np.ptp(self, axis=0))
        d = np.atleast_2d(self)

        ax = pl.gca()
        series_ticks = []
        colors = pl.cm.jet(np.linspace(0, 1, d.shape[1]))
        
        last_max = 0.
        for idx,series in enumerate(d.T):
            data = np.ma.masked_array(series,np.isnan(series))
            data = data-np.ma.min(data) + stacked*last_max
            if stacked: series_ticks.append(np.ma.mean(data))
            else:   series_ticks.append(data[-1])
            
            ax.plot(self.time, data.filled(np.nan), color=colors[idx], **kwargs)
            last_max = np.ma.max(data) + stacked*gap
        
        ax2 = ax.twinx()
        pl.yticks(series_ticks, [str(i) for i in xrange(len(series_ticks))], weight='bold')
        [l.set_color(col) for col,l in zip(colors,ax2.get_yticklabels())]
        pl.ylim(ax.get_ylim())
            
        pl.xlabel('Time')   
