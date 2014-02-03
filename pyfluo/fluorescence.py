import numpy as np
from time_series import TimeSeries

def compute_dff(series, mode='window', *args, **kwargs):
	"""Compute the "delta-F over F" of a time series signal by calling one of multiple functions for this calculation.
	
	Args:
		series (pyfluo.TimeSeries or list thereof): signal/s
		mode (str): "stim" / "window"
	Returns:
		The return value of the function called.
	"""
	if mode == 'window':
		return dff_window(series, *args, **kwargs)
	elif mode == 'stim':
		return dff_stim(series, *args, **kwargs)

def dff_stim(seriess, stim=None, base_time=0.3):
	"""Calculates delta-F over F using pre-stimulation baseline as F0.
	
	FUNCTION OUT OF DATE, LIKELY CONTAINS BUGS.
	
	Args:
		seriess (pyfluo.TimeSeries): should be passed from *compute_dff*
		stim (pyfluo.StimSeries): if mode is "stim," this argument represents the stimulation associated with *series*
		base_time (float): if mode is "stim," this argument represents the time before stimulation to be averaged as a base line F0.
		
	Returns:
		a TimeSeriesCollection of DFF signals, one per stimulation in *stim* (if *series* is a single TimeSeries), or a list thereof (if *series* is a list of TimeSeries).
	"""
	if stim==None:
		raise Exception('DFF cannot be calculated using desired mode without stim_series.')
		
	dffs = []
	for sidx in range(seriess.n_series):
		series = seriess.get_series(sidx)
		traces_aligned = series.take(stim.stim_times, pad=(base_time,base_time))
		baselines = [np.mean(tr.take([-base_time,0.])) for tr in traces_aligned]
		for tr,bl in zip(traces_aligned, baselines):
			for idx,samp in enumerate(tr):
				tr[idx] = (samp - bl)/bl
		dffs.append(traces_aligned)
	
	if len(dffs) == 1:
		dffs = dffs[0]
		
	return dffs
def dff_window(seriess, tao0=0.2, tao1=0.75, tao2=3.0, noise_filter=False):
	"""Calculates delta-F over F using a sliding window method.
	
	THIS SHOULD EVENTUALLY BE MODIFIED TO MORE EFFICIENTLY MAKE USE OF THE TIME SERIES MATRIX DATA, COMPUTING DFF IN A SINGLE PASS.
	
	Args:
		seriess (pyfluo.TimeSeries): should be passed from *compute_dff*
		tao0 (float): see Jia et al. 2010
		tao1 (float): see Jia et al. 2010
		tao2 (float): see Jia et al. 2010
		noise_filter (bool): include the final noise filtering step of the algorithm
		
	Returns:
		a TimeSeries containing the DFF signal.
		
	Notes:
		Adapted from Jia et al. 2010 Nature Protocols
		
		The main adjustment not specified in the algorithm is how I deal with the beginning and end of the signal. When we're too close to the borders of the signal such that averages/baselines are subject to noise, I allow the function to look in the other direction (forward if at beginning, backward if at end) to make the signal more robust. This is reflected by the variables "forward" and "backward" in the calculation of f_bar and f_not.
	"""
	
	tao0t = tao0
	tao1t = tao1
	tao2t = tao2
	
	dffs = None
		
	for sidx in range(seriess.n_series):
		series = seriess.get_series(sidx)	
		
		tao0 = tao0t * series.fs
		tao1 = tao1t * series.fs
		tao2 = tao2t * series.fs
				
		f_bar = np.zeros(len(series))
		for idx,i in enumerate(series):
			i1 = int(idx-round(tao1/2.))
			forward=0
			if i1<0:
				forward=abs(i1)
				i1=0
			i2 = int(idx+round(tao1/2.)) + forward
			backward=0
			if i2>=len(series):
				backward=i2-len(series)
				i2=len(series.data)-1
			integ = np.take( series.data, range(i1-backward,i2+1))
			integ = np.mean(integ)
			f_bar[idx] = ( integ )
		
		f_not = np.zeros(len(series))
		for idx in range(len(f_not)):
			i1 = int(idx-tao2)
			forward=0
			if i1<0:
				forward=abs(i1)	
				i1=0
			search = np.take(f_bar, range(i1, idx+1+forward))
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
				numerator = np.sum(r[t::-1]*w[:t+1])
				denominator = np.sum(w[:t+1])
				
				if numerator == 0 and denominator == 0:
					dff[t] = 0
				else:
					dff[t] = np.divide(numerator, denominator)				
		else:
			dff = r
		
		if dffs == None:
			dffs = TimeSeries(data=dff, time=series.time)
		else:
			dffs.append_series(dff)
		
	return dffs

if __name__ == "__main__":	
	pass