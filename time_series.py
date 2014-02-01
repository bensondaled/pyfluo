from pyfluo.ts_base import TSBase
import numpy as np
import pylab as pl
import copy
import time as pytime
import pickle
	
class TimeSeries(TSBase):
	"""A data structure that holds a one-dimensional array of values each associated with a particular time index.
	
	Attributes:
		data (np.ndarray): the data array of the time series.
		
		time (np.ndarray): the time array of the time series.
		
		Ts (float): the sampling period of the time series.
		
		fs (float): the sampling frequency of the time series.
		
		tunit (str): the unit of measurement in which the object's time is stored.
		
		name (str): a unique name generated for the object when instantiated
		
	"""
	def __init__(self, data, time=None, info=None, tunit='s', merge_method=np.mean):
		"""Initialize a TimeSeries object.
		
		Args:
			data (list or np.ndarray): list of numerical values.
			time (list or np.ndarray): list of uniformly spaced numerical values, identical in length *data*. If *None*, defaults to range(len(*data*)).
			info (list or np.ndarray): list of information associated with each time point.
			tunit (str): the unit of measurement in which the object's time is stored.
			
			If a list of TimeSeries is given for data, it will *merge* them, not concatenate them.
		"""
		self.name = pytime.strftime("TimeSeries-%Y%m%d_%H%M%S")
		self.tunit = tunit
		self.fs = None
		self.Ts = None
		self.time = time
		
		self.data = data
		if type(self.data) == TimeSeries:
			self.time = self.data.time
			self.data = self.data.data
		elif type(self.data)==list and type(self.data[0]) == TimeSeries:
			if not all([len(s)==len(self.data[0]) for s in self.data]):
				raise Exception('Series of varying duration cannot be joined in a new TimeSeries object.')

			self.time = merge_method([s.time for s in self.data], axis=0)
			# Concatenate the TimeSeries objects if:
			# a) they all only contain one series 
			# b) they contain varying numbers of series
			if all([s.n_series==1 for s in self.data]) or ( not all([s.n_series==self.data[0].n_series for s in self.data]) ):
				self.data = np.concatenate([ts.data for ts in self.data])
			# Merge the TimeSeries objects if: they all contain the same number (>1) of series.
			elif all([s.n_series==self.data[0].n_series for s in self.data]):
				self.data = merge_method(np.dstack([s.data for s in self.data]), axis=2)
				
		self.data = np.atleast_2d(self.data).astype(np.float64)

		if self.time != None:
			self.time = np.asarray(self.time).astype(np.float64)
		else:
			self.time = np.array(range(len(self))).astype(np.float64)
		
		self.info = info
		if self.info == None:
			self.info = [None for i in self.time]
		
		if len(np.shape(self.data)) > 2:
			raise Exception("Data contains too many dimensions.")
		if np.shape(self.data)[1] != self.time.size:
			raise Exception("Data series and time vectors contain a different number of samples.")
		if not all(self.time[i] <= self.time[i+1] for i in xrange(len(self.time)-1)):
			raise Exception("Time vector is not sorted.")
	
		self.n_series = np.shape(self.data)[0]
		self._update()
	
	# Public Methods
	def get_series(self, idx):
		return self[str(idx)]
	def append(self, item):
		self.data = np.insert(self.data,np.shape(self.data)[-1],item,axis=len(np.shape(self.data))-1)
		added_time = self.time[-1] + (np.array(range(len(np.shape(item))))+1)*self.Ts
		self.time = np.append(self.time, added_time)
	def append_series(self, item):
		self.data = np.insert(self.data,np.shape(self.data)[0],item,axis=0)
	def normalize(self, minn=0., maxx=1., in_place=False):
		"""Normalize the time series object.
		
		Args:
			minn (float): desired post-normalizaton data minimum
			maxx (float): desired post-normalizaton data maximum
			in_place: apply the normalization to *this* instance of the object
			
		Returns:
			A new time series object, normalized.
		"""
		omin = np.min(self.data, axis=1)
		omax = np.max(self.data, axis=1)
		newdata = np.array([(i-omin[idx])/(omax[idx]-omin[idx]) for idx,i in enumerate(self.data)])
		new = self.copy()
		new.data = newdata
		new.time = self.time
		if in_place:
			self.data = new.data
			self.time = new.time
			self._update()
		return new
	def merge(self, method=np.average):
		series = TimeSeries(data=method(self.data, axis=0), time=self.time, tunit=self.tunit)
		return series
	def take(self, time_range, pad=(0., 0.), reset_time=True, **kwargs):
		"""Extract a range of values from the time series.
		
		Args:
			time_range (list): the start and end times of the range desired.
			pad (list): a list of 2 values specifying the padding to be inserted around specified time range. The first value is subtracted from the start time, and the second value is added to the end time.
			reset_time (bool): set the first element of the resultant time series to time 0.
			
		Notes:
			The *time_range* argument can be either a pair of values (start_time, end_time), or a list of such pairs. In the latter case, this method is applied to each pair of times, and the result is returned in a collection.
			
			If values in *time_range* lie outside the bounds of the time series, or if the padding causes this to be true, the time vector is extrapolated accordingly, and the data for all non-existent points is given as None. 
			
		Returns:
			TimeSeries between supplied time range, if *time_range* is a pair of values
			or
			TimeSeriesCollection of TimeSeries corresponding to each supplied time range , if *time_range* is a list of pairs
		"""
		
		if type(time_range[0]) not in (list, tuple):
			time_range = [time_range]
		stims = TimeSeries(data=[self._take(st, pad=pad, reset_time=reset_time, **kwargs) for st in time_range])
		return stims
	def plot(self, stim_series=None, stacked=True, gap_fraction=0.15, use_idxs=False, normalize=False, show=True, **kwargs):
		"""Plot the time series.
		
		Args:
			use_idxs (bool): ignore time and instead use vector indices as x coordinate.
			normalize (bool): normalize the data before plotting.
			show (bool): show the plot immediately.
			**kwargs: any arguments accepted by *matplotlib.plot*
		"""
		if use_idxs:	t = range(len(self.data))
		else:	t = self.time
			
		if normalize:	d = self.normalize().data
		else:	d = self.data
				
		last_max = 0.
		for idx,series in enumerate(d):
			data = np.ma.masked_array(series,np.isnan(series))
			smin = np.ma.min(data)
			data = data-smin + stacked*last_max
			
			pl.plot(t, data.filled(np.nan), **kwargs)
			
			if stim_series and stacked:
				stim_data = stim_series.data*(np.max(data)-np.min(data)) + np.min(data)
				pl.plot(stim_series.time, stim_data, color='black', ls='dashed')
				
			last_max = np.ma.max(data) + stacked*gap_fraction*(np.ma.max(data) - np.ma.min(data))
			
		if stim_series and not stacked:
			stim_data = stim_series.data*np.max(data)
			pl.plot(stim_series.time, stim_data, color='black', ls='dashed')
	
		pl.xlabel('Time (%s)'%self.tunit)
		if show:	pl.show()
		
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
	def __str__(self):
		return '\n'.join([
		'TimeSeries object.',
		'Number of series: %i'%self.n_series,
		"Length: %i samples."%len(self),
		"Duration: %f seconds."%(self.time[-1]-self.time[0]+self.Ts),
		])
	def __repr__(self):
		return "Time: %s\nData: %s\n"%(self.time.__repr__(),self.data.__repr__())