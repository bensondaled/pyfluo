import numpy as np
import pylab as pl
import copy
import pickle
	
class TimeSeries(object):
	"""A data structure that holds a one-dimensional array of values each associated with a particular time index.
	
	Attributes:
		data (np.ndarray): the data array of the time series.
		
		time (np.ndarray): the time array of the time series.
		
		Ts (float): the sampling period of the time series.
		
		fs (float): the sampling frequency of the time series.
		
		tunit (str): the unit of measurement in which the object's time is stored.
		
	"""
	def __init__(self, data, time=None, tunit='s'):
		"""Initialize a TimeSeries object.
		
		Args:
			data (list or np.ndarray): list of numerical values.
			time (list or np.ndarray): list of uniformly spaced numerical values, identical in length *data*. If *None*, defaults to range(len(*data*)).
			tunit (str): the unit of measurement in which the object's time is stored.
		"""
		self.tunit = tunit
		self.fs = None
		self.Ts = None
		self.data = np.asarray(data,dtype=float)
		
		if time != None:
			self.time = np.asarray(time,dtype=float)
		else:
			self.time = np.asarray(range(len(self.data)))
		if self.data.size != self.time.size:
			raise Exception("Data and time vectors are different lengths!")
		if not all(self.time[i] <= self.time[i+1] for i in xrange(len(self.time)-1)):
			raise Exception("Time vector is not sorted!")
	
		self._update()
	
	# Public Methods
	def copy(self):
		"""Return a deep copy of this object.
		"""
		return copy.deepcopy(self)
	def append(self, item):
		"""Append a sample to the time series.
		
		Args:
			item (int/double/float): new data.
		"""
		self.data.append(item)
		self.time.append(self.time[-1]+self.Ts)
	def time_to_idx(self, t, mode=round):
		t = float(t)
		time_step = t - self.time[0]
		idx = mode(time_step*self.fs)
		return int(idx)
	def resample(self, n, in_place=False):
		"""Resample the time series object.
		
		Args:
			n (int): resampling interval
			in_place: apply the resampling to *this* instance of the object
			
		Returns:
			A new time series object, resampled.
		"""
		new = self.copy()
		new.data = self.data[::n]
		new.time = self.time[::n]
		if in_place:
			self.data = new.data
			self.time = new.time
			self._update()
		return new
	def normalize(self, minn=0., maxx=1., in_place=False):
		"""Normalize the time series object.
		
		Args:
			minn (float): desired post-normalizaton data minimum
			maxx (float): desired post-normalizaton data maximum
			in_place: apply the normalization to *this* instance of the object
			
		Returns:
			A new time series object, normalized.
		"""
		omin = np.min(self.data)
		omax = np.max(self.data)
		newdata = np.array([(i-omin)/(omax-omin) for i in self.data])
		new = self.copy()
		new.data = newdata
		new.time = self.time
		if in_place:
			self.data = new.data
			self.time = new.time
			self._update()
		return new
	def take(self, time_range, pad=(0., 0.), reset_time=True):
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
		from time_series_collection import TimeSeriesCollection
		
		if type(time_range[0]) != list:
			time_range = [time_range]
		stims = [self._take(st, pad=pad, reset_time=reset_time) for st in time_range]
		stims = TimeSeriesCollection(stims)
		if len(stims) == 1:
			stims = stims[0]
		return stims
	def _take(self, time_range, pad=(0.,0.), reset_time=True):
		t1 = time_range[0] - pad[0]
		t2 = time_range[1] + pad[1]
		if t1 > t2:
			t_temp = t1
			t1 = t2
			t2 = t_temp
		idx1 = self.time_to_idx(t1, mode=round)
		idx2 = self.time_to_idx(t2, mode=round)
		
		t = np.take(self.time, range(idx1,idx2+1), mode='clip')
		if idx1<0:	t[:-idx1] = [t[-idx1]-i*self.Ts for i in range(-idx1,0,-1)]
		if idx2>len(self.time)-1:
			t[-(idx2-(len(self.time)-1)):] = [t[-1]+i*self.Ts for i in range(1, idx2-(len(self.time)-1)+1)]
		if reset_time:	t = t - time_range[0]
		
		data = np.take(self.data, range(idx1,idx2+1), mode='clip')
		if idx1<0:	data[:-idx1] = None
		if idx2>len(self.time)-1:	data[-(idx2-(len(self.time)-1)):] = None
		
		return TimeSeries(data, time=t)
	def plot(self,use_idxs=False,normalize=False,show=True,**kwargs):
		"""Plot the time series.
		
		Args:
			use_idxs (bool): ignore time and instead use vector indices as x coordinate.
			normalize (bool): normalize the data before plotting.
			show (bool): show the plot immediately.
			**kwargs: any arguments accepted by *matplotlib.plot*
		"""
		if use_idxs:
			t = range(len(self.data))
		else:
			t = self.time
		if normalize:
			d = self.normalize().data
		else:
			d = self.data
		pl.plot(t,d,**kwargs)
		pl.xlabel('Time (%s)'%self.tunit)
		if show:
			pl.show()
	
	# Saving data
	def save(self, file_name):
		"""Save the time series object using python pickling.
		
		Args:
			file_name: file name.
		"""
		f = open(file_name+'.pfts', 'wb')
		pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
		
	# Special Methods
	def _update(self):
		if self.data.size > 1:
			self.Ts = self.time[1] - self.time[0]
			self.fs = 1/self.Ts

			if not all(self.time[i+1]-self.time[i] == self.Ts for i in xrange(len(self.time)-1)):
				raise Exception("Time vector does not have a consistent sampling period!")
	def __len__(self):
		return self.data.size
	def __contains__(self, item):
		return item in self.data
	def __getitem__(self, idx):
		return self.data[idx]
	def __setitem__(self, idx, val):
		self.data[idx] = val
	def __add__(self, other):
		self.data = self.data + other
		return self
	def __radd__(self, other):
		self.data = self.data + other
		return self
	def __sub__(self, other):
		self.data = self.data - other
		return self
	def __rsub__(self, other):
		self.data = self.data - other
		return self
	def __mul__(self, other):
		self.data = self.data * other
		return self
	def __rmul__(self, other):
		self.data = self.data * other
		return self
	def __div__(self, other):
		self.data = self.data / other
		return self
	def __rdiv__(self, other):
		self.data = self.data / other
		return self
	def __pow__(self, other):
		self.data = self.data ** other
		return self
	def __neg__(self):
		self.data = -self.data
		return self
	def __abs__(self):
		self.data = np.abs(self.data)
		return self
	def __int__(self):
		self.data = self.data.astype(int)
		return self
	def __float__(self):
		self.data = self.data.astype(float)
		return self
	def __str__(self):
		return '\n'.join([
		'TimeSeries object.',
		"Length: %i samples."%len(self),
		"Duration: %f seconds."%(self.time[-1]-self.time[0]+self.Ts),
		])
	def __repr__(self):
		return "Time: %s\nData: %s\n"%(self.time.__repr__(),self.data.__repr__())
	def set_time_unit(unit):
		self.tunit = unit