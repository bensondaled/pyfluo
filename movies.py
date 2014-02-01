from pyfluo.ts_base import TSBase
from pyfluo.tiff import WangLabScanImageTiff
from pyfluo.time_series import TimeSeries
from pyfluo.stimulation import StimSeries
from pyfluo.roi import ROI, ROISet
import matplotlib.animation as animation
import numpy as np
import pylab as pl
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import os
import time as pytime
import json
import pickle
from pyfluo.util import *

class MultiChannelMovie(object):
	"""An object that holds multiple *Movie* objects as channels.
	
	This class is currently exclusively for creation from WangLabScanImageTiff's. Its main goal is to circumvent the need to load a multi-channel tiff file more than once in order to attain movies from its multiple channels.
	
	Attributes:
		movies (list): list of Movie objects.
		
		name (str): a unique name generated for the object when instantiated
		
	"""
	def __init__(self, raw, skip=(0,0)):
		"""Initialize a MultiChannelMovie object.
		
		Args:
			raw (str or WangLabScanImageTiff or list thereof): list of movies.
			skip (list): a two-item list specifying how many frames to skip from the start (first item) and end (second item) of each movie.
		"""
		self.name = pytime.strftime("MultiChannelMovie-%Y%m%d_%H%M%S")
		
		self.movies = []
		
		if type(raw) != list:
			raw = [raw]
		widgets=[' Loading tiffs:', Percentage(), Bar()]
		pbar = ProgressBar(widgets=widgets, maxval=len(raw)).start()
		for idx,item in enumerate(raw):
			if type(item) == str:
				raw[idx] = WangLabScanImageTiff(item)
				pbar.update(idx+1)
			elif type(item) != WangLabScanImageTiff:
				raise Exception('Invalid input for movie. Should be WangLabScanImageTiff or tiff filename.')
		tiffs = raw
		pbar.finish()
				
		n_channels = tiffs[0].n_channels
	
		for ch in range(n_channels):	
			data = None
			info = []
		
			for item in tiffs:
				if item.n_channels != n_channels:
					raise Exception('Channel number inconsistent among provided tiffs.')
				
				chan = item[ch]
				dat = chan['data']
				info += (chan['info'])
				if data == None:
					data = dat
				else:
					if np.shape(dat)[:2] != np.shape(data)[:2]:
						raise Exception('Movies are not of the same dimensions.')
					data = np.append(data, dat, axis=2)	
			mov = Movie(data=data, info=info, skip=skip)
			self.movies.append(mov)
			
	def get_channel(self, i):
		return self.movies[i]
	def __getitem__(self, i):
		return self.get_channel(i)
	def __len__(self):
		return len(self.movies)

class Movie(TSBase):
	"""An object that holds a movie: a series of images each with a timestamp. Specficially designed for two-photon microscopy data stored in tiffs.
	
	Attributes:
		data (np.array): a 3D matrix storing the movie data as a series of frames.
		
		info (list): a list storing any relevant data associated with each frame in the movie.
		
		time (np.array): a one-dimensional array storing the timestamp of each frame.
		
		width (int): width of the movie in pixels.
		
		height (int): height of the movie in pixels.
		
		pixel_duration (float): the duration of time (in seconds) associated with one pixel.
		
		frame_duration (float): the duration of time (in seconds) associated with one frame.
		
		Ts (float): sampling period (in seconds) with regard to frames. Equivalent to 1/fs.
		
		fs (float): sampling frequency (in Hz) with regard to frames. Equivalent to 1/Ts.
		
		rois (ROISet): contains any selected regions of interest that were stored in association with the Movie object.

		name (str): a unique name generated for the object when instantiated

	"""
	def __init__(self, data, time=None, info=None, skip=(0,0)):
		"""Initialize a Movie object.
		
		Args:
			data (np.array): a 3D matrix storing the movie data as a series of frames. Dimensions are (n,height,width) where n is the number frames in the movie.
			time (np.array): a one-dimensional array storing the timestamps of each frame in *data*. If None, uses *info* to extract a sampling rate, and builds time based on that.
			info (list): a list storing any relevant data associated with each frame in the movie. Defaults to a list of None.
			skip (list): a two-item list specifying how many frames to skip from the start (first item) and end (second item) of *data*.
		"""
		self.name = pytime.strftime("Movie-%Y%m%d_%H%M%S")
			
		self.data = data
		self.info = info
		if self.info==None:
			self.info = [None for i in range(len(self))]
			
		skip_beginning = skip[0]
		skip_end = skip[1]
		if skip_beginning:
			self.data = self.data[:,:,skip_beginning:]
			self.info = self.info[skip_beginning:]
		if skip_end:
			self.data = self.data[:,:,:-skip_end]
			self.info = self.info[:-skip_end]
		
		self.ex_info = self.info[0]
		lpf = float(self.ex_info['state.acq.linesPerFrame'])
		ppl = float(self.ex_info['state.acq.pixelsPerLine'])
		mspl = float(self.ex_info['state.acq.msPerLine'])
		self.pixel_duration = mspl / ppl / 1000. #seconds
		self.frame_duration = self.pixel_duration * ppl * lpf #seconds
		self.Ts = self.frame_duration
		self.fs = 1/self.Ts
		
		self.time = np.arange(len(self))*self.frame_duration
		self.width = np.shape(self.data)[1]
		self.height = np.shape(self.data)[0]
				
		self.rois = ROISet()
	
	# Special Calls
	def __getitem__(self, idx):
		return self.data[:,:,idx]
	def __len__(self):
		return np.shape(self.data)[2]
	def __str__(self):
		return '\n'.join([
		'Movie object.',
		"Length: %i frames."%len(self),
		"Frame Dimensions: %i x %i"%(np.shape(self.data)[0], np.shape(self.data)[1]),
		"Duration: %f seconds."%(self.time[-1]-self.time[0]+self.frame_duration),
		])
	
	# Public Methods
	
	# Modifying data
	def append_movie(self, movies):
		if type(movies) == Movie:
			movies = [movies]
		if type(movies) != list:
			raise Exception('Not a valid data type to append to a Movie object (should be Movie or list of Movies).')
		for m in movies:
			if m.frame_duration != self.frame_duration:
				raise Exception('Frame rates of movies to be appended do not match.')
			self.data = np.append(self.data, m.data, axis=2)
		self.time = np.arange(len(self))*self.frame_duration
			
	# Extracting/Reshaping data	
	def take(self, *args, **kwargs):
		"""Extract a range of frames from the movie.
		
		Args:
			time_range (list): the start and end times of the range desired.
			pad (list): a list of 2 values specifying the padding to be inserted around specified time range. The first value is subtracted from the start time, and the second value is added to the end time.
			reset_time (bool): set the first element of the resultant time series to time 0.
			
			If values in *time_range* lie outside the bounds of the time series, or if the padding causes this to be true, the time vector is extrapolated accordingly, and the data for all non-existent points is given as None. 
			
		Returns:
			Movie between supplied time range
		"""
		try:
			time_range = kwargs.pop('time_range')
		except KeyError:
			time_range = args.pop(0)
				
		if type(time_range[0]) != list:
			time_range = [time_range]
		movs = [self._take(st, *args, **kwargs) for st in time_range]
		if len(movs)>1:
			mov_data = np.mean([m.data for m in movs], axis=0)
		elif len(movs)==1:
			mov_data = movs[0].data
		return Movie(data=mov_data, info=movs[0].info)
		
	def flatten(self, destination_class=StimSeries):
		"""Flatten the values in *data* to a linear series.
		
		To be used when the movie-capturing apparatus was used to capture a signal whose natural shape is linear.

		Returns:
			A TimeSeries object, the flattened movie.
		"""
		flat_data = np.transpose(self.data,[2,0,1]).flatten()
		t = np.arange(len(flat_data))*self.pixel_duration
		return destination_class(data=flat_data, time=t)		
	
	# ROI analysis
	def select_roi(self, n=1, store=True):
		"""Select any number of regions of interest (ROI) in the movie.
		
		Args:
			n (int): number of ROIs to select.
			store (bool): store the selected ROI(s) as attributes of this movie instance.
			
		Returns:
			ROISet object sotring the selected ROIs (if >1 ROIs selected)
			or
			ROI object of selected ROI (if 1 ROI selected).
		"""
		rois = []
		for q in range(n):
			zp = self.z_project(show=True, rois=True)
			roi = None
			pts = pl.ginput(0)
			if pts:
				roi = ROI(np.shape(zp), pts)
				if store:
					self.rois.add(roi)
			rois.append(roi)
		pl.close()
		if len(rois)==1:
			rois = rois[0]
		return ROISet(rois)
	def extract_by_roi(self, rois=None, method=np.ma.mean):
		"""Extract a time series consisting of one value for each movie frame, attained by performing an operation over the regions of interest (ROI) supplied.
		
		Args:
			rois (ROISet or list): the ROI(s) over which to extract data. If None, uses the object attribute *rois*.
			method (def): the function by which to convert the data within an ROI to a single value.
			
		Returns:
			TimeSeries object (if single ROI supplied)
			or
			TimeSeriesCollection object (if multiple ROIs supplied).
		"""
		series = None
		if rois == None:
			rois = self.rois
		if type(rois) not in (list, tuple, ROISet):
			rois = [rois]
		for roi in rois:
			if type(roi)==int:
				roi = self.rois[roi]
			roi_stack = np.dstack([roi.mask for i in self])
			masked = np.ma.masked_array(self.data, mask=roi_stack)
			ser = method(method(masked,axis=0),axis=0)
			if series == None:
				series = TimeSeries(data=ser, time=self.time)
			else:
				series.append_series(ser)
		return series
	
	# Visualizing data
	def z_project(self, method=np.mean, show=False, rois=False):
		"""Flatten/project the movie data across all frames (z-axis).
		
		Args:
			method (def): function to apply across the z-axis of the data.
			show (bool): display the result.
			rois (bool): display this object's stored regions of interest, as dictated by the class attribute *rois*.
			
		Returns:
			The z-projected image (same width and height as any movie frame).
		"""
		zp = method(self.data,2)
		if show:
			pl.imshow(zp, cmap=mpl_cm.Greys_r)
			if rois:
				self.rois.show(mode='pts',labels=True)
		return zp
	def play(self, loop=False, fps=None, **kwargs):
		"""Play the movie.
		
		Args:
			loop (bool): repeat playback upon finishing.
			fps (float): playback frames per second.
			
		"""
		if fps==None:
			fps = self.fs
		fpms = fps / 1000.
		
		flag = pl.isinteractive()
		pl.ioff()
		fig = pl.figure()
		ims = [ [pl.imshow(i, cmap=mpl_cm.Greys_r)] for i in self ]
		
		ani = animation.ArtistAnimation(fig, ims, interval=1./fpms, blit=False, repeat=loop, **kwargs)
		pl.show()
		if flag:	pl.ion()