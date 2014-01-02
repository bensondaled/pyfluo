import numpy as np
from time_series import TimeSeries, TimeSeriesCollection

def compute_dff(seriess, tao0=0.2, tao1=0.75, tao2=3.0, noise_filter=True):
	# Adapted from Jia et al. 2010 Nature Protocols
	
	if type(seriess) == TimeSeries:
		seriess = [seriess]
	dffs = []
	
	for series in seriess:
	
		tao0 = tao0 * series.fs
		tao1 = tao1 * series.fs
		tao2 = tao2 * series.fs
	
		f_bar = np.zeros(len(series))
		for idx,i in enumerate(series):
			integ = np.take(series.data, range(int(idx-round(tao1/2)), int(idx+round(tao1/2))), mode='clip')	
			integ = np.sum(integ)
			f_bar[idx] = ( 1/tao1 * integ )
	
		f_not = np.zeros(len(series))
		for idx,i in enumerate(series):
			search = np.take(f_bar, range(int(idx-tao2), idx), mode='clip')
			if np.size(search):
				f_not[idx] = np.min(search)		
			
		r = np.zeros(len(series))
		for idx,i in enumerate(series):
			if i-f_not[idx] == 0:
				r[idx] = 0
			else:
				r[idx] = (i - f_not[idx]) / f_not[idx]
	
		def w(x):
			return np.exp(-np.abs(x)/tao0)

		dff = np.zeros(len(series))
		
		if noise_filter:
			for t in range(len(series)):
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
		else:
			dff = r
		
		dffs.append(TimeSeries(data=dff, time=series.time))
	dffs = TimeSeriesCollection(dffs)
		
	if len(dffs) == 1:
		dffs = dffs[0]
	return dffs

if __name__ == "__main__":	
	pass