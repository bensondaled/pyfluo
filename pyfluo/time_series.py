from pyfluo.ts_base import TSBase
import numpy as np
import pylab as pl
import matplotlib.cm as mpl_cm
import copy
import time as pytime
import pickle
    
class TimeSeries(TSBase):
    """A data structure that holds a one or more one-dimensional arrays of values, associated with a single time vector. 
    
    Note that because this object can store multiple data arrays as separate ``series'', most of its methods perform operations in a parallel manner on each series. Unless otherwise noted, this is the default action of the class methods.
    
    **Attributes:**
        * **data** (*np.ndarray*): the data array of the time series.
        
        * **time** (*np.ndarray*): the time array of the time series.
        
        * **info** (*np.ndarray*): an array of meta-information associated with time points.
        
        * **n_series** (*int*): the number of time series stored in the object (i.e. the number of rows in *data*).
        
        * **Ts** (*float*): the sampling period of the time series.
        
        * **fs** (*float*): the sampling frequency of the time series.
        
        * **tunit** (*str*): the unit of measurement in which the object's time is stored.
        
        * **name** (*str*): a unique name generated for the object when instantiated
        
    **Parameters:**
            * **data** (*list* / *np.ndarray* / *TimeSeries*): the time series data. See below for details on parameter format.
            
            * **time** (*list* / *np.ndarray*): list of uniformly spaced numerical values, identical in length to the number of columns in *data*. If ``None``, defaults to ``range(len(data))``.
            
            * **info** (*list* / *np.ndarray*): list of meta-information associated with each time point.
            
            * **tunit** (*str*): the unit of measurement in which the object's time is stored.
            
            * **merge_method_time** (*def*): the function to be used to merge time vector when necessary. See notes on *data* parameter for details.
            
            * **merge_method_data** (*def*): the function to be used to merge the data vector when necessary. See notes on *data* parameter for details.
        
        
    **Notes:**
        For the *data* parameter, the only strict constraint is that all input contains the same number of samples. There are multple types of input that are supported:
        
        #. The data of one or multiple time series. In the case of multiple time series, supply a list of lists, where each list is one of the time series. This can also be a 2D np.ndarray of the same structure. For example::
            
            >>> data = [ 31.6, 81.5, 51.8, 93.7, 12.2, 16.1 ]   
                        
            >>> data = [ [5.5, 6.1, 3.3],
                         [2.8, 1.6, 9.8],
                         [1.2, 3.4, 5.2] ]
            
            
        #. A list of TimeSeries objects. This situation is handled in various ways depending on the nature of the input:
        
            **(a)** If the supplied TimeSeries objects all contain just single time series (i.e. ``TimeSeries.n_series == 1``), they are concatenated and the resulting TimeSeries will contain multiple rows of data corresponding to those inputs. The time vectors of the series are merged using the function given by the *merge_method_time* argument in the constructor.
        
            **(b)** If the supplied TimeSeries objects contain varying numbers of time series (i.e. ``TimeSeries.n_series`` varies), the course of action is the same as (a), concatenating all the series from each TimeSeries object into one large TimeSeries object. Again, time vectors are merged as in (a).
        
            **(c)** If the supplied TimeSeries objects contain multiple time series (i.e. two-dimensional data vectors), and *every one has the same number of series*, it is assumed that there is a relationship between the series, and they are merged. Specifically, the first series (or data row) from every supplied TimeSeries is merged into one, then the second, then third, etc. The merge is performed by the *merge_method_data* constructor argument. This course of action can be overridden by setting the *merge_method_data* constructor argument to None, in which case the input will be handled as in (b).
            
    **Usage Tips:**
        The *TimeSeries* object is in many ways like a numpy ndarray. In general it is safe to assume that operations performed on the object will be performed on the *data* attribute. Here are some basic pointers for using *TimeSeries*:
        
        * Indexing::
                
            # create a time series
            >>> ts = TimeSeries(data = [2.0, 4.0, 8.0, 16.0, 32.0])
            
            # basic indexing
            >>> ts[0]
            2.0
            
            # slicing
            >>> ts[2:4]
            np.array([8.0, 16.0])
            
            # create a TimeSeries with multiple series
            >>> ts = TimeSeries(data = [ [2.0, 4.0, 6.0],
                                         [1.0, 2.0, 3.0],
                                         [5.0, 10.0, 15.0] ])
            
            # regular indexing
            >>> ts[1]
            np.array([ 4.0,
                       2.0,
                       10.0 ])
            
            # index a series using a string
            >>> ts['1']
            TimeSeries(data: [1.0, 2.0, 3.0])
            
            
        * Arithmetic::
        
            # create some time series
            >>> ts1 = TimeSeries(data = [2.0, 4.0, 8.0, 16.0])
            >>> ts2 = TimeSeries(data = [10.0, 20.0, 30.0, 40.0])
        
            # add TimeSeries
            >>> ts1 + ts2
            TimeSeries(data: [12.0, 24.0, 38.0 56.0])
            
    """
    def __init__(self, data, time=None, info=None, tunit='s', merge_method_data=np.mean, merge_method_time=np.mean):
        super(TimeSeries, self).__init__()
        self.tunit = tunit
        self.fs = None
        self.Ts = None
        self.time = time
    
        self._data = data
        if type(self._data) == TimeSeries:
            self.time = self._data.time
            self._data = self._data.data
        elif type(self._data)==list and type(self._data[0]) == TimeSeries:
            if not all([len(s)==len(self._data[0]) for s in self._data]):
                raise Exception('Series of varying duration cannot be joined in a new TimeSeries object.')

            self.time = merge_method_time([s.time for s in self._data], axis=0)
            # Concatenate the TimeSeries objects if:
            # a) they all only contain one series 
            # b) they contain varying numbers of series
            # c) the concatenate argument is given as True
            if all([s.n_series==1 for s in self._data]) or ( not all([s.n_series==self._data[0].n_series for s in self._data]) ) or merge_method_data==None:
                self._data = np.concatenate([ts.data for ts in self._data])
            # Merge the TimeSeries objects if: they all contain the same number (>1) of series.
            elif all([s.n_series==self._data[0].n_series for s in self._data]):
                self._data = merge_method_data(np.dstack([s.data for s in self._data]), axis=2)
                
        self.data = self._data

        if self.time != None:
            self.time = np.asarray(self.time).astype(np.float64)
        else:
            self.time = np.array(range(len(self))).astype(np.float64)
        
        self.info = info
        if self.info == None:
            self.info = [None for i in self.time]
        
        if len(np.shape(self.data)) > 2:
            raise Exception("Data contains too many dimensions.")
        if len(self) != len(self.time):
            raise Exception("Data series and time vectors contain a different number of samples.")
        if not all(self.time[i] <= self.time[i+1] for i in xrange(len(self.time)-1)):
            raise Exception("Time vector is not sorted.")
    
        self._update()
    
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, data):
        self._data = np.atleast_2d(data).astype(np.float64)

    # Public Methods
    def get_series(self, idx):
        """ Get one row of data (one series).
        
        **Parameters:**
            * **idx** (*int*): the index of the desired series (i.e. row of data).
            
        **Returns:**
            A TimeSeries object.
            
        .. note:: Note that this is equivalent to the simpler string-based indexing. For example, ``ts.get_series(1)`` is equivalent to ``ts['1']``.
        
        """
        if type(idx)==int:
            return self[str(idx)]
        elif type(idx)==list:
            return self[[str(i) for i in idx]]
    def append(self, item):
        """ Append data to the series.
        
        **Parameters:**
            * **item** (*int* / *float* / *np.ndarray* / *list*): the data to be appended to the series.
            
            If a single value is supplied, the value is added to the end of the data. In the case that multiple data rows already exist in the data, the new entry is added to the end of each series.
            If a list or array is supplied, it must be the same length as the number of series (i.e. rows of data) in the object's data. In this case, the first item in the list is added to the end of the first series, the second to the second, and so on.
            
        Examples::
            
            # example 1
            
            >>> ts.data = [ [1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9] ]
             
            >>> ts.append(42)
             
            >>> ts.data
            
            [ [1, 2, 3, 42],
              [4, 5, 6, 42],
              [7, 8, 9, 42] ]
            
            # example 2
                    
            >>> ts.data = [ [1, 2],
                            [3, 4],
                            [5, 6] ]
                    
            >>> ts.append([66, 77, 88])
            
            >>> ts.data
            
            [ [1, 2, 66],
              [3, 4, 77],
              [5, 6, 88] ]          
            
        
        """
        self.data = np.insert(self.data,np.shape(self.data)[-1],item,axis=len(np.shape(self.data))-1)
        added_time = self.time[-1] + (np.array(range(len(np.shape(item))))+1)*self.Ts
        self.time = np.append(self.time, added_time)
        self._update()
    def append_series(self, item, merge_method_time=np.mean):
        """ Append a series, or row of data, to the object.

        **Parameters:**
            * **item** (*np.ndarray* / *list* / *TimeSeries*): array of data to be added as a new row, or series. This must be the same length as the number of time points (i.e. columns) in the object's data.
            * **merge_method_time** (*def*): function to be used to merge time vectors (applies when *item* is a *TimeSeries*).
        
        Example::
            
            >>> ts.data = [ [1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9] ]
                    
            >>> ts.append_series([10, 11, 12])

            >>> ts.data 
            
            [ [1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],    
              [10, 11, 12] ]
        """
        if type(item) == TimeSeries:
            item = TimeSeries.data
            self.time = merge_method_time([self.time, item.time], axis=0)
        
        self.data = np.insert(self.data,np.shape(self.data)[0],item,axis=0)
        self._update()
    def normalize(self, minmax=(0., 1.), by_series=True, in_place=False):
        """Normalize the time series object.
        
        **Parameters:**
            * **minmax** (*list*): ``[post_normalizaton_data_min, max]``
            * **by_series** (*bool*): normalize each data row individually
            * **in_place**: apply the normalization to *this* instance of the object.
            
        **Returns:**
            A new *TimeSeries* object, normalized.
            
        Example::
        
            >>> ts.data = [ [1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9] ]
            
            >>> ts.normalize()
            
            >>> ts.data
            
            [ [0, 0.5, 1],
              [0, 0.5, 1],
              [0, 0.5, 1] ]
        """
        newmin = minmax[0]
        newmax = minmax[1]
        if by_series:
            omin = np.min(self.data, axis=1)[:,None]
            omax = np.max(self.data, axis=1)[:,None]
        else:
            omin,omax = [np.min(self.data), np.max(self.data)]
        newdata = (self.data-omin)/(omax-omin) * (newmax-newmin) + newmin
        new = self.copy()
        new.data = newdata
        new.time = self.time
        if in_place:
            self.data = new.data
            self.time = new.time
            self._update()
        return new
    def merge(self, method=np.mean):
        """Merge the series (data rows) of the object.
        
        **Parameters:**
            * **method** (*def*): the function with which to perform the merge.
            
        **Returns:**
            A new *TimeSeries* object with just one row of data, the merged result.
            
        Example::
        
            >>> ts.data = [ [1, 3, 5],
                            [2, 4, 6],
                            [9, 2, 7] ]
            
            >>> ts = ts.merge(method=np.mean)
            
            >>> ts.data
            
            [ [4, 3, 6] ]
        """
        series = TimeSeries(data=method(self.data, axis=0), time=self.time, tunit=self.tunit)
        return series
    def take(self, time_range, pad=(0., 0.), reset_time=True, merge_method_time=np.mean, merge_method_data=np.mean, **kwargs):
        """Extract a range of data values using time indices.
        
        **Parameters:**
            * **time_range** (*list*): ``[start_time, end_time]``
            * **pad** (*list*): used optionally to expand the specified time range on either end. The first value is subtracted from the start time, and the second value is added to the end time.
            * **reset_time** (*bool*): shift the time vector of the resultant time series, such that time zero corresponds to the data originally located at ``time_range[0]``.
            * **merge_method_time** (*def*): specifies how to merge time when multiple time ranges are extracted from multiple data series. See note below for details.
            * **merge_method_data** (*def*): specifies how to merge data when multiple time ranges are extracted from multiple data series. See note below for details.
            
        **Returns:**
            A new *TimeSeries* object with data corresponding to desired time range.
        
        **Notes:**
            * The *time_range* parameter can be either a pair of values ``(start_time, end_time)``, or a list of such pairs. In the latter case, this method is applied to each pair of times, and the result is a *TimeSeries* object with multiple rows of data.
            
            * If values in *time_range* lie outside the bounds of the time series, or if the padding causes this to be true, the time vector is extrapolated accordingly, and the data for all non-existent points is given as ``None``. 
            
            * Concerning merging of data: this method first extracts the desired time range/s from each row individually. If the original object contains multiple rows of data, there are two different results that should be expected: 
                #. If only one time range is supplied, the resulting TimeSeries then contains multiple rows, each corresponding to the rows of the original object, as would be expected.
                #. If multiple time ranges are supplied, this method extracts a set of time ranges from each of the objects' series. Since TimeSeries objects are only capable of holding two-dimensional data, creation of the resulting TimeSeries will, by action of the constructor, merge this data. Because the multiple rows of the original series likely had individual significance, the default behaviour for this merge is to preserve the separation of the series. 
            
                For example, consider a TimeSeries with 3 rows of data corresponding to the fluorescence traces of three regions of interest. One may use this *take* method to align multiple points of stimulation (say, 5) in each of these traces all at once. In the resulting *TimeSeries*, there will then be just 3 rows of data, corresponding to the 3 regions of interest. These will consist of the merged data from the 5 stimulations. 
                
                The merge is performed by the function supplied in the *merge_method* parameter. Importantly, this merge can be overriden, as described in the class constructor, by supplying ``merge_method=None``. In that case, all time segments from all data series are concatenated into rows and set as the data of the resulting *TimeSeries*.
            
        """
        
        if type(time_range[0]) not in (list, tuple):
            time_range = [time_range]
        stims = TimeSeries(data=[self._take(st, pad=pad, reset_time=reset_time, take_axis=1, **kwargs) for st in time_range], merge_method_time=merge_method_time, merge_method_data=merge_method_data)
        return stims
    def plot(self, stim=None, stacked=True, gap_fraction=0.08, use_idxs=False, normalize=False, show=True, color=None, **kwargs):
        """Plot the time series.
        
        **Parameters:**
            * **stim** (*pyfluo.StimSeries*): stimulation to be plotted over the data.
            * **stacked** (*bool*): for multiple rows of data, stack instead of overlaying.
            * **gap_fraction** (*float*): if ``stacked==True``, specifies the spacing between curves as a fraction of the average range of the curves.
            * **use_idxs** (*bool*): ignore time and instead use vector indices as x coordinate.
            * **normalize** (*bool*): normalize the data before plotting.
            * **show** (*bool*): show the plot immediately.
            * ** **kwargs:** any arguments accepted by *matplotlib.plot*
        """
        if use_idxs:    t = range(len(self.data))
        else:   t = self.time
            
        if normalize:   d = self.normalize(normalize).data
        else:   d = self.data
       
        gap = gap_fraction * np.mean(np.ptp(d, axis=1))

        ax = pl.gca()
        series_ticks = []
        if color==None: colors = mpl_cm.jet(np.linspace(0, 1, self.n_series))
        else: colors = [color for i in range(self.n_series)]
        
        if self.n_series==1:
            data = d[0]
            ax.plot(t, data, color=colors[0], **kwargs)
            if stim:
                stim.plot(normalize=(np.min(data), np.max(data)), color='black', ls='dotted')
        elif self.n_series>1:       
            last_max = 0.
            for idx,series in enumerate(d):
                data = np.ma.masked_array(series,np.isnan(series))
                smin = np.ma.min(data)
                data = data-smin + stacked*last_max
                if stacked: series_ticks.append(np.ma.mean(data))
                else:   series_ticks.append(data[-1])
            
                ax.plot(t, data.filled(np.nan), color=colors[idx], **kwargs)
                last_max = np.ma.max(data) + stacked*gap
            
            if stim and not stacked:
                stim.plot(normalize=(0., np.max(data)), color='black', ls='dashed')
            elif stim and stacked:
                stim.plot(normalize=(0., last_max), color='black', ls='dotted')
            
            ax2 = ax.twinx()
            pl.yticks(series_ticks, [str(i) for i in range(len(series_ticks))], weight='bold')
            [l.set_color(col) for col,l in zip(colors,ax2.get_yticklabels())]
            pl.ylim(ax.get_ylim())
            
        pl.xlabel('Time (%s)'%self.tunit)   
        if show:    pl.show()
        
    # Special Methods
    def __len__(self):
        return np.shape(self.data)[1]
    def __contains__(self, item):
        return item in self.data
    def __getitem__(self, idx):
        if type(idx) in (int, slice):
            return self.data[...,idx]
        elif type(idx) == str:
            return TimeSeries(data=self.data[int(idx)], time=self.time, info=self.info, tunit=self.tunit)
        elif type(idx) in (list, tuple) and type(idx[0])==str:
            return TimeSeries(data=[self[i] for i in idx], time=self.time, info=self.info, tunit=self.tunit)
    def __str__(self):
        return '\n'.join([
        'TimeSeries object.',
        'Number of series: %i'%self.n_series,
        "Length: %i samples."%len(self),
        "Duration: %f seconds."%(self.time[-1]-self.time[0]+self.Ts),
        ])
    def __repr__(self):
        return "Time: %s\nData: %s\n"%(self.time.__repr__(),self.data.__repr__())
    def _update(self):
        if len(self) > 1:           
            self.Ts = self.time[1] - self.time[0]
            self.fs = 1/self.Ts
        self.n_series = np.shape(self.data)[0]

            #The following should be implemented, however it was causing problems:
            # if not all(round(self.time[i+1]-self.time[i],10) == round(self.Ts,10) for i in xrange(len(self.time)-1)):
            #   raise Exception("Time vector does not have a consistent sampling period to within 10 decimal places.")
