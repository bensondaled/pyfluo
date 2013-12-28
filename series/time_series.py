import numpy as np
import pylab as pl
	
class TimeSeries(object):
	def __init__(self, data, time=None, tunit='s'):
	
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
	
		if self.data.size > 1:
			self.Ts = self.time[1] - self.time[0]
			self.fs = 1/self.Ts
	
	# Special Methods
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
	
	# Public Methods
	def append(self, item):
		self.data.append(item)
		self.time.append(self.time[-1]+self.Ts)
	def time_to_idx(self, t, mode=round):
		t = float(t)
		time_step = t - self.time[0]
		idx = mode(time_step*self.fs)
		return int(idx)
	def resample(self, n):
		return Trace(self.data[::n], time=self.time[::n])
	def normalize(self, minn=0., maxx=1.):
		omin = np.min(self.data)
		omax = np.max(self.data)
		newdata = np.array([(i-omin)/(omax-omin) for i in self.data])
		return Series1D(data=newdata, time=self.time)
	def take(self, stim_times, *args, **kwargs):
		if type(stim_times[0]) != list:
			stim_times = [stim_times]
		stims = [self._take(st, *args, **kwargs) for st in stim_times]
		return stims
	def _take(self, time_range, pad=(0.,0.), reset_time=True):
		t1 = time_range[0] - pad[0]
		t2 = time_range[1] + pad[1]
		if t1 > t2:
			t_temp = t1
			t1 = t2
			t2 = t_temp
		idx1 = self.time_to_idx(t1, mode=np.floor)
		idx2 = self.time_to_idx(t2, mode=np.ceil)
		
		t = np.take(self.time, range(idx1,idx2+1), mode='clip')
		if idx1<0:	t[:-idx1] = [t[-idx1]-i*self.Ts for i in range(-idx1,0,-1)]
		if idx2>len(self.time)-1:
			t[-(idx2-(len(self.time)-1)):] = [t[-1]+i*self.Ts for i in range(1, idx2-(len(self.time)-1)+1)]
		if reset_time:	t = t - time_range[0]
		
		data = np.take(self.data, range(idx1,idx2+1), mode='clip')
		if idx1<0:	data[:-idx1] = None
		if idx2>len(self.time)-1:	data[-(idx2-(len(self.time)-1)):] = None
		
		return TimeSeries(data, time=t)
	def plot(self,use_idxs=False,norm=False,show=True,**kwargs):
		if use_idxs:
			t = range(len(self.data))
		else:
			t = self.time
		if norm:
			d = self.normalize().data
		else:
			d = self.data
		pl.plot(t,d,**kwargs)
		pl.xlabel('Time (%s)'%self.tunit)
		if show:
			pl.show()