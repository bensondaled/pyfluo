from ts_base import TSBase
import numpy as np
import pylab as pl
import warnings

class Trace(TSBase):
    """
    Important to use this convention: with 2d data, data[0] refers to time-point 0 of all contained traces (as opposed to all time points of trace 0). This allows it to follow the same logic as movies, in which movie[0] refers to a frame at time 0, not a zero'th pixel along all time.
    """
    def __new__(cls, data, **kwargs):
        return super(Trace, cls).__new__(cls, data, n_dims=[1,2], **kwargs)

    def normalize(self, minmax=(0., 1.), axis=None):
        """Normalize the trace object.
        
        **Parameters:**
            * **minmax** (*list*): ``[post_normalizaton_data_min, max]``
            * **by_series** (*bool*): normalize each data row individually
            
        **Returns:**
            A new *Trace* object, normalized.
            
        Example::
        
            >>> ts.data = [ [1, 2, 3],
            >>>             [4, 5, 6],
            >>>             [7, 8, 9] ]
            
            >>> ts.normalize()
            
            >>> ts.data
            
            [ [0, 0.5, 1],
              [0, 0.5, 1],
              [0, 0.5, 1] ]
        """
        newmin,newmax = minmax
        omin,omax = self.min(axis=axis),self.max(axis=axis)
        new_obj = self.copy() 
        new_obj.data = (self-omin)/(omax-omin) * (newmax-newmin) + newmin
        return new_obj
    def plot(self, stacked=True, gap_fraction=0.08, **kwargs):
        """Plot the time series.
        
        **Parameters:**
            * **stacked** (*bool*): for multiple rows of data, stack instead of overlaying.
            * **gap_fraction** (*float*): if ``stacked==True``, specifies the spacing between curves as a fraction of the average range of the curves.
            * ** **kwargs:** any arguments accepted by *matplotlib.plot*
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
