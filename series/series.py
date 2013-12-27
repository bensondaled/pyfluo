import numpy as np
import pylab as pl

def t2idx(array,value):
	return (np.abs(array-value)).argmin()
	
class Series1D(object):
	def __init__(self, data, time=None, tunit='s'):
		self.data = np.asarray(data,dtype=float)
		if time != None:
			self.time = np.asarray(time,dtype=float)
		else:
			self.time = np.asarray(range(len(self.data)))
		if self.data.size != self.time.size:
			raise Exception("Trace and time vectors are different lengths!")
	
		self.tunit = tunit
	
		self._update()

	# Special Methods
	def _update(self):
		if self.data.size > 1:
			self.Ts = self.time[1] - self.time[0]
			self.fs = 1/self.Ts
		else:
			self.Ts = None
			self.fs = None
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
		'Series object.',
		"Length: %i samples."%len(self),
		"Duration: %f seconds."%(self.time[-1]-self.time[0]+self.Ts),
		"Data: " + str(self.data)
		])
	def __repr__(self):
		return self.__str__()
	def set_time_unit(unit):
		self.tunit = unit
	
	# Public Methods	
	def resample(self, n):
		return Trace(self.data[::n], time=self.time[::n])
	def normalize(self, minn=0., maxx=1.):
		omin = np.min(self.data)
		omax = np.max(self.data)
		newdata = np.array([(i-omin)/(omax-omin) for i in self.data])
		return Series1D(data=newdata, time=self.time)
	def take(self, time_range, pad=(0.,0.), reset_time=True):
		t1 = time_range[0] - pad[0]
		t2 = time_range[1] + pad[1]
		idx1 = t2idx(self.time,t1)
		idx2 = t2idx(self.time,t2)
		t = np.take(self.time, range(idx1,idx2), mode='clip')
		if reset_time:
			t = t - time_range[0]
		return FluoSeries(t, np.take(self.data, range(idx1,idx2), mode='clip'))
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