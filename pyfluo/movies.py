from pyfluo.ts_base import TSBase
from pyfluo.pf_base import pfBase
from pyfluo.time_series import TimeSeries
from pyfluo.stimulation import StimSeries
from pyfluo.fluorescence import subtract_background as sub_bg
from pyfluo.roi import ROI, ROISet
import matplotlib.animation as animation
import numpy as np
import pylab as pl
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import os
import json
import pickle


class Movie(TSBase):
    """An object that holds a movie: a series of images each with a timestamp. Specficially designed for two-photon microscopy data stored in tiffs.
    
    **Attributes:**
        * **data** (*np.array*): a 3D matrix storing the movie data as a series of frames.
        
        * **info** (*list*): a list storing any relevant data associated with each frame in the movie.
        
        * **time** (*np.array*): a one-dimensional array storing the timestamp of each frame.
        
        * **width** (*int*): width of the movie in pixels.
        
        * **height** (*int*): height of the movie in pixels.
        
        * **pixel_duration** (*float*): the duration of time (in seconds) associated with one pixel.
        
        * **frame_duration** (*float*): the duration of time (in seconds) associated with one frame.
        
        * **Ts** (*float*): sampling period (in seconds) with regard to frames. Equivalent to 1/*fs*.
        
        * **fs** (*float*): sampling frequency (in Hz) with regard to frames. Equivalent to 1/*Ts*.
        
        * **rois** (*ROISet*): contains any selected regions of interest that were stored in association with the Movie object.

        * **name** (*str*): a unique name generated for the object when instantiated

    """
    def __init__(self, data, time=None, info=None, skip=(0,0,0)):
        """Initialize a Movie object.
        
        **Parameters:**
            * **data** (*np.array*): a 3D matrix storing the movie data as a series of frames. Dimensions are (n,height,width) where n is the number frames in the movie.
            * **time** (*np.array*): the timestamps of each frame in *data*. If ``None``, uses *info* to extract a sampling rate and builds time based on that.
            * **info** (*list*): a list storing any relevant data associated with each frame in the movie. Defaults to a list of ``None``.
            * **skip** (*list*): ``[number_of_frames_to_ignore_at_beginning, end, interval_on_which_to_ignore]``
        """
        super(Movie, self).__init__()
            
        self.data = data
        self.info = info
        if self.info==None:
            self.info = [None for i in range(len(self))]
            
        skip_beginning = skip[0]
        skip_end = skip[1]
        skip_interval = skip[2]
        if skip_beginning:
            self.data = self.data[skip_beginning:]
            self.info = self.info[skip_beginning:]
        if skip_end:
            self.data = self.data[:-skip_end]
            self.info = self.info[:-skip_end]
        if skip_interval:
            self.data = self.data[[i for i in range(len(self.data)) if (i+1)%skip_interval],...]
        
        self.ex_info = self.info[0]
        lpf = float(self.ex_info['state.acq.linesPerFrame'])
        ppl = float(self.ex_info['state.acq.pixelsPerLine'])
        mspl = float(self.ex_info['state.acq.msPerLine'])
        self.pixel_duration = mspl / ppl / 1000. #seconds
        self.frame_duration = self.pixel_duration * ppl * lpf #seconds
        self.Ts = self.frame_duration
        self.fs = 1/self.Ts
        
        self.time = np.arange(len(self))*self.frame_duration
                
        self.rois = ROISet()

        self.visual_aspect = 1.0
    
    # Special Calls
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return np.shape(self.data)[0]
    def __str__(self):
        return '\n'.join([
        'Movie object.',
        "Length: %i frames."%len(self),
        "Frame Dimensions: %i x %i"%(np.shape(self.data)[0], np.shape(self.data)[1]),
        "Duration: %f seconds."%(self.time[-1]-self.time[0]+self.Ts),
        ])
    
    # Public Methods
    
    # Modifying data
    def append(self, movies):
        """Append another Movie object to this Movie.
        """
        if type(movies) == Movie:
            movies = [movies]
        if type(movies) != list:
            raise Exception('Not a valid data type to append to a Movie object (should be Movie or list of Movies).')
        for m in movies:
            if m.frame_duration != self.frame_duration:
                raise Exception('Frame rates of movies to be appended do not match.')
            self.data = np.append(self.data, m.data, axis=0)
            self.info += m.info
        self.time = np.arange(len(self))*self.frame_duration
            
    # Extracting/Reshaping data 
    def take(self, *args, **kwargs):
        """Extract a range of frames from the movie.
      
        **Parameters:**
            * **time_range** (*list*): the start and end times of the range desired.
            * **merge_method** (*def*): the method used to merge results if more than one time range is supplied. If ``None``, returns a list of movies.
            * **pad** (*list*): a list of 2 values specifying the padding to be inserted around specified time range. The first value is subtracted from the start time, and the second value is added to the end time.
            * **reset_time** (*bool*): set the first element of the resultant time series to time 0.
            
        **Returns:**
            Movie between supplied time range
            
        **Notes:**  
            If values in *time_range* lie outside the bounds of the movie time, or if the padding causes this to be true, the time vector is extrapolated accordingly, and the data for all non-existent points is given as ``None``.
        """
        time_range = kwargs.pop('time_range', list(args).pop(0))
        if type(time_range) != list:
            raise Exception('Improper time range supplied for take().')
        merge_method = kwargs.pop('merge_method', None)
        if type(time_range[0]) != list:
            time_range = [time_range]

        movs = [self._take(st, take_axis=0, **kwargs) for st in time_range]
        if len(movs)>1 and merge_method != None:
            mov_data = merge_method([m.data for m in movs], axis=0)
        elif len(movs)>1 and merge_method == None:
            return movs
        elif len(movs)==1:
            mov_data = movs[0].data
        return self.__class__(data=mov_data, info=movs[0].info)
        
    def flatten(self, destination_class=StimSeries, **kwargs):
        """Flatten the values in *data* to a linear series.
        
        To be used when the movie-capturing apparatus was used to capture a signal whose natural shape is linear. For example, capturing trigger data in a tiff file.
        
        **Parameters:**
            * **destination_class** (*type*): the class in which to store and return the flattened data. Ideal options are *TimeSeries* or *StimSeries*.

        **Returns:**
            An object of type *destination_class*, the flattened movie.
        """
        flat_data = self.data.flatten()
        t = np.arange(len(flat_data))*self.pixel_duration
        return destination_class(data=flat_data, time=t, **kwargs)      
    
    # ROI analysis
    def select_roi(self, n=1, store=True):
        """Select any number of regions of interest (ROI) in the movie.
        
        **Parameters:**
            * **n** (*int*): number of ROIs to select.
            * **store** (*bool*): store the selected ROI(s) as attributes of this movie instance.
            
        **Returns:**
            *ROISet* object storing the selected ROIs (if >1 ROIs selected)
            or
            *ROI* object of selected ROI (if 1 ROI selected).
        """
        rois = []
        for q in range(n):
            pl.clf()
            zp = self.z_project(show=True, rois=True)
            roi = None
            pts = pl.ginput(0, timeout=0)
            if pts:
                roi = ROI(np.shape(zp), pts)
                if store:
                    self.rois.add(roi)
            rois.append(roi)
        pl.close()
        return ROISet(rois)
    def extract_by_roi(self, rois=None, method=np.mean, subtract_background=True):
        """Extract a time series consisting of one value for each movie frame, attained by performing an operation over the regions of interest (ROI) supplied.
        
        **Parameters:**
            * **rois** (*ROISet* / *list*): the ROI(s) over which to extract data. If None, uses the object attribute *rois*.
            * **method** (*def*): the function by which to convert the data within an ROI to a single value.
            * **subtract_background** (*bool*): subtract the background and noise using *pyfluo.fluorescence.subtract_background*
            
        **Returns:**
            *TimeSeries* object, with multiple rows corresponding to multiple ROIs.
        """
        series = None
        if rois == None:
            rois = self.rois
        if type(rois) not in (list, tuple, ROISet):
            rois = [rois]
        for roi in rois:
            if type(roi)==int:
                roi = self.rois[roi]
            data = self.data[:,~roi.mask]
            if subtract_background:
                data = sub_bg(data)
            ser = method(data, axis=1) 

            if series == None:
                series = TimeSeries(data=ser, time=self.time)
            else:
                series.append_series(ser)
        return series
    
    # Visualizing data
    def z_project(self, method=np.mean, show=False, rois=False, aspect='equal'):
        """Flatten/project the movie data across all frames (z-axis).
        
        **Parameters:**
            * **method** (*def*): function to apply across the z-axis of the data.
            * **show** (*bool*): display the result.
            * **rois** (*bool*): display this object's stored regions of interest, as dictated by the class attribute *rois*.
            
        **Returns:**
            The z-projected image (same width and height as any movie frame).
        """
        zp = method(self.data,axis=0)
        if show:
            ax = pl.gca()
            ax.margins(0.)
            pl.imshow(np.atleast_2d(zp), cmap=mpl_cm.Greys_r, aspect=aspect)
            if rois:
                self.rois.show(mode='pts',labels=True)
        return zp
    def play(self, loop=False, fps=None, **kwargs):
        """Play the movie.
        
        **Parameters:**
            * **loop** (*bool*): repeat playback upon finishing.
            * **fps** (*float*): playback rate in frames per second.
            
        """
        if fps==None:
            fps = self.fs
        fpms = fps / 1000.
        
        flag = pl.isinteractive()
        pl.ioff()
        fig = pl.figure()
        ims = [ [pl.imshow(np.atleast_2d(i), cmap=mpl_cm.Greys_r, aspect=self.visual_aspect)] for i in self ]
        
        ani = animation.ArtistAnimation(fig, ims, interval=1./fpms, blit=False, repeat=loop, **kwargs)
        pl.show()
        if flag:    pl.ion()


class LineScan(Movie):
    """An object that holds a line scan:  a series of 1-dimensional images each with a timestamp. Specficially designed for two-photon microscopy data stored in tiffs.
   
    This object is effectively identical to the *Movie* class (it is a subclass of it), but some its methods are adjusted to work with line scan data.
    """
    def __init__(self, *args, **kwargs):
        skip = kwargs.pop('skip', (0,0,0))
        super(LineScan, self).__init__(*args, **kwargs)

        if len(np.shape(self.data))==3:
            self.info = list(np.repeat(self.info, np.shape(self.data)[1]))
            self.data = self.data.reshape((np.shape(self.data)[0]*np.shape(self.data)[1], np.shape(self.data)[2]))
        if len(np.shape(self.data))!=2:
            raise Exception('Data given to LineScan was not of a parseable shape.')
        
        skip_beginning = skip[0]
        skip_end = skip[1]
        skip_interval = skip[2]
        if skip_beginning:
            self.data = self.data[skip_beginning:]
            self.info = self.info[skip_beginning:]
        if skip_end:
            self.data = self.data[:-skip_end]
            self.info = self.info[:-skip_end]
        if skip_interval:
            self.data = self.data[[i for i in range(len(self.data)) if (i+1)%skip_interval],...]

        line_duration = self.pixel_duration * np.shape(self.data)[1]
        self.time = np.arange(len(self))*line_duration

        self.Ts = line_duration
        self.fs = 1./self.Ts

        self.visual_aspect = 20
    def z_project(self, *args, **kwargs):
        aspect = kwargs.pop('aspect', self.visual_aspect)
        return super(LineScan, self).z_project(*args, aspect=aspect, **kwargs)
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return np.shape(self.data)[0]
    def __str__(self):
        return '\n'.join([
        'LineScan object.',
        "Length: %i lines."%len(self),
        "Pixels per line: %i"%(np.shape(self.data)[1]),
        "Duration: %f seconds."%(self.time[-1]-self.time[0]+self.Ts),
        ])
