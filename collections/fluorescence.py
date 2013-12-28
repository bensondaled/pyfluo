import numpy as np
import pylab as pl
from pyfluo.series.fluorescence import FluoSeries

class FluoSeriesCollection():
	def __init__(self, fluo_series, stim_series=None):
		self.series = fluo_series
		self.stim_series = stim_series
	def set_triggers(self, stim_series):
		self.stim_series = stim_series
	def combine(self, combine_triggers=True, combine_traces=False, method=np.average, pad=(0.0,0.0)):
		stim_times = self.stim_series.stim_times
		if combine_triggers:
			avg_stims = []
			for ser in self.series:
				stims = [ser.take(time_range=st, pad=pad) for st in stim_times]
				mstims = np.vstack(stims)
				mstims = np.ma.masked_array(mstims,np.isnan(mstims))
				avg_ser = FluoSeries(data = np.ma.mean(mstims, axis=0).filled(np.nan), time=stims[0].time)
				avg_stims.append(avg_ser)
			series = avg_stims
		else:
			series = self.series
			
		# if combine_traces:
		# 	series = [np.mean(np.vstack(series), axis=0)]
		# 	return FluoSeries(data=series, time=TO BE COMPLETED)
		
		return FluoSeriesCollection(fluo_series=series)
			
	def plot(self, with_stim=True, layered=True):
		gap = 1.0
		last_max = 0.
		pl.figure()
		for idx,s in enumerate(self.series):
			data = s.data
			data = np.ma.masked_array(data,np.isnan(data))
			smin = np.ma.min(data)
			data = data-smin + layered*last_max
			pl.plot(s.time, data.filled(np.nan))
			if with_stim and self.stim_series:
				stim = self.stim_series.data*(np.max(data)-np.min(data)) + last_max
				pl.plot(self.stim_series.time, stim)
			last_max = np.ma.max(data) + layered*gap
		pl.xlabel("Time (%s)"%s.tunit)
		pl.show()

if __name__ == "__main__":	
	pass