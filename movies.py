from pyfluo.ts_base import TSBase
from pyfluo.tiff import WangLabScanImageTiff
from pyfluo.stimulation import StimSeries
from pyfluo.time_series import TimeSeries, TimeSeriesCollection
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

class MultiChannelMovie(object):
	"""Currently exclusively for creation from WangLabScanImageTiff's, this class holds multiple Movie objects as channels. Its main goal is to circumvent the need to load a multi-channel tiff file more than once in order to attain movies from its multiple channels.
	"""
	def __init__(self, raw, skip_beginning=0, skip_end=0):
		self.name = pytime.strftime("MultiChannelMovie-%Y%m%d_%H%M%S")
		
		self.movies = []
		
		if type(raw) != list:
			raw = [raw]
		for idx,item in enumerate(raw):
			if type(item) == str:
				raw[idx] = WangLabScanImageTiff(item)
			elif type(item) != WangLabScanImageTiff:
				raise Exception('Invalid input for movie. Should be WangLabScanImageTiff or tiff filename.')
		tiffs = raw
				
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
					
			mov = Movie(data=data, info=info, skip_beginning=skip_beginning, skip_end=skip_end)
			self.movies.append(mov)
			
	def get_channel(self, i):
		return self.movies[i]
	def __getitem__(self, i):
		return self.get_channel(i)
	def __len__(self):
		return len(self.movies)

class Movie(TSBase):
	def __init__(self, data, time=None, info=None, skip_beginning=0, skip_end=0):
		self.name = pytime.strftime("Movie-%Y%m%d_%H%M%S")
			
		self.data = data
		self.info = info
		if self.info==None:
			self.info = [None for i in range(len(self))]
			
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
		
	def as_series(self):
		flat_data = np.transpose(self.data,[2,0,1]).flatten()
		t = np.arange(len(flat_data))*self.pixel_duration
		return Series1D(data=flat_data, time=t)		
	def as_stim_series(self):
		flat_data = np.transpose(self.data,[2,0,1]).flatten()
		t = np.arange(len(flat_data))*self.pixel_duration
		return StimSeries(data=flat_data, time=t)
	
	# ROI analysis
	def select_roi(self, n=1, store=True):
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
		return rois
	def extract_by_roi(self, rois=None, method=np.ma.mean):
		series = []
		if rois == None:
			rois = self.rois
		if type(rois) == int:
			rois = [self.rois[rois]]
		elif type(rois) == ROI:
			roiset = ROISet(rois)
		for roi in rois:
			if type(roi)==int:
				roi = self.rois[roi]
			roi_stack = np.dstack([roi.mask for i in self])
			masked = np.ma.masked_array(self.data, mask=roi_stack)
			ser = method(method(masked,axis=0),axis=0)
			ser = TimeSeries(ser, time=self.time)
			series.append(ser)
		series = TimeSeriesCollection(series)
		
		if len(series) == 1:
			return series[0]
		else:
			return series
	
	# Visualizing data
	def z_project(self, method=np.mean, show=False, rois=False):
		zp = method(self.data,2)
		if show:
			pl.imshow(zp, cmap=mpl_cm.Greys_r)
			if rois:
				self.rois.show(mode='pts',labels=True)
		return zp
	def play(self, repeat=False, fps=None, **kwargs):
		if fps==None:
			fps = self.fs
		fpms = fps / 1000.
		
		flag = pl.isinteractive()
		pl.ioff()
		fig = pl.figure()
		ims = [ [pl.imshow(i, cmap=mpl_cm.Greys_r)] for i in self ]
		
		ani = animation.ArtistAnimation(fig, ims, interval=1./fpms, blit=False, repeat=repeat, **kwargs)
		pl.show()
		if flag:	pl.ion()