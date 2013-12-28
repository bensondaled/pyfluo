import numpy as np
from time_series import TimeSeries

class FluoSeries(TimeSeries):
	def __init__(self, *args, **kwargs):
		super(FluoSeries, self).__init__(*args, **kwargs)
		self.original_data = self.data
		self.dff = self.to_dff()
		
		self.set_data_dff()
	def set_data_original(self):
		self.data = self.original_data
	def set_data_dff(self):
		self.data = self.dff
	def to_dff(self, tao0=0.2, tao1=0.75, tao2=3.0):
		
		tao0 = tao0 * self.fs
		tao1 = tao1 * self.fs
		tao2 = tao2 * self.fs
		
		f_bar = np.zeros(len(self))
		for idx,i in enumerate(self):
			integ = np.take(self.data, range(int(idx-round(tao1/2)), int(idx+round(tao1/2))), mode='clip')	
			integ = np.sum(integ)
			f_bar[idx] = ( 1/tao1 * integ )
		
		f_not = np.zeros(len(self))
		for idx,i in enumerate(self):
			search = np.take(f_bar, range(int(idx-tao2), idx), mode='clip')
			if np.size(search):
				f_not[idx] = np.min(search)		
				
		r = np.zeros(len(self))
		for idx,i in enumerate(self):
			if i-f_not[idx] == 0:
				r[idx] = 0
			else:
				r[idx] = (i - f_not[idx]) / f_not[idx]
		
		def w(x):
				return np.exp(-np.abs(x)/tao0)

		dff = np.zeros(len(self))
		for t in range(len(self)):
			numerator = 0.
			for q in range(t):
				numerator += (np.take(r, [t-q], mode='clip') * w(q))
			denominator = 0.
			for q in range(t):
				denominator += w(q)
			if numerator == 0 and denominator == 0:
				dff[t] = 0
			else:
				dff[t] = np.divide(numerator, denominator)
	
		return dff

if __name__ == "__main__":	
	pass