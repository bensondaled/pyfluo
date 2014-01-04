import numpy as np
from time_series import TimeSeries, TimeSeriesCollection

def compute_dff(series, mode='stim', *args, **kwargs):
	if mode == 'window':
		return dff_window(series, *args, **kwargs)
	elif mode == 'stim':
		return dff_stim(series, *args, **kwargs)

def dff_stim(seriess, stim=None, base_time=0.3):
	if stim==None:
		raise Exception('DFF cannot be calculated using desired mode without stim_series.')
		
	if type(seriess) == TimeSeries:
		seriess = [seriess]
	dffs = []
	for series in seriess:
		traces_aligned = series.take(stim.stim_times, pad=(base_time,base_time))
		baselines = [np.mean(tr.take([-base_time,0.])) for tr in traces_aligned]
		for tr,bl in zip(traces_aligned, baselines):
			for idx,samp in enumerate(tr):
				tr[idx] = (samp - bl)/bl
		dffs.append(traces_aligned)
	
	if len(dffs) == 1:
		dffs = dffs[0]
		
	return dffs
def dff_window(seriess, tao0=0.2, tao1=0.75, tao2=3.0, noise_filter=True):
	# Adapted from Jia et al. 2010 Nature Protocols
	
	if type(seriess) == TimeSeries:
		seriess = [seriess]
	dffs = []
	
	for sidx,series in enumerate(seriess):
			
		tao0 = tao0 * series.fs
		tao1 = tao1 * series.fs
		tao2 = tao2 * series.fs
				
		f_bar = np.zeros(len(series))
		for idx,i in enumerate(series):
			i1 = int(idx-round(tao1/2))
			if i1<0:	i1=0
			i2 = int(idx+round(tao1/2))
			if i2>=len(series.data):	i2=len(series.data)-1
			integ = np.take( series.data, range(i1,i2) )	
			integ = np.sum(integ)
			f_bar[idx] = ( 1/tao1 * integ )
		
		f_not = np.zeros(len(series))
		for idx in range(len(f_not)):
			i1 = int(idx-tao2)
			if i1<0:	i1=0
			search = np.take(f_bar, range(i1, idx))
			if np.size(search):
				f_not[idx] = np.min(search)	
		
		r = np.zeros(len(series))
		for idx,s in enumerate(series):
			if f_not[idx] == 0:
				r[idx] = s
			else:
				r[idx] = (s - f_not[idx]) / f_not[idx]
	
		def w_func(x):
			return np.exp(-np.abs(x)/tao0)
		w = w_func(np.arange(0,len(series)))

		dff = np.zeros(len(series))
		
		if noise_filter:
			for t in range(len(series)):
				numerator = np.sum(r[:t+1]*w[:t+1])
				denominator = np.sum(w[:t+1])
				
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