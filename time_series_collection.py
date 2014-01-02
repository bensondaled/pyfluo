import numpy as np
import pylab as pl
from time_series import TimeSeries
import pickle

class TimeSeriesCollection():
	def __init__(self, series):
		self.time = series[0].time
		for s in series:
			if len(s) != len(self.time):
				raise Exception('Series provided to TimeSeriesCollection are different in length!')
		
		self.series = series
	
	# Special Methods
	def __len__(self):
		return len(self.series)
	def __getitem__(self, idx):
		return self.series[idx]
	
	def take(self, time_range, pad=(0.0,0.0), merge_method=np.average):
		series = [s.take(time_range, pad=pad) for s in self.series]
		if type(time_range[0]) == list:
			series = [s.merge(method=merge_method) for s in series]
		return TimeSeriesCollection(series)
	def merge(self, method=np.average):
		series = TimeSeries(method(self.series, axis=0), time=self.time)
		return series
	def plot(self, stim_series=None, stacked=True, gap_fraction=0.5):
		allmaxs = []
		last_max = 0.
		for idx,s in enumerate(self.series):
			data = s.data
			data = np.ma.masked_array(data,np.isnan(data))
			smin = np.ma.min(data)
			data = data-smin + stacked*last_max
			pl.plot(s.time, data.filled(np.nan))
			if stim_series and stacked:
				stim_data = stim_series.data*(np.max(data)-np.min(data))
				pl.plot(stim_series.time, stim_data)
			last_max = np.ma.max(data) + stacked*gap_fraction*(np.ma.max(data) - np.ma.min(data))
			allmaxs.append(last_max)
		if stim_series and not stacked:
			stim_data = stim_series.data*max(allmaxs)
			pl.plot(stim_series.time, stim_data)
		pl.xlabel("Time (%s)"%s.tunit)
		pl.show()
		
	# Saving data
	def save(self, file_name):
		f = open(file_name+'.pftsc', 'wb')
		pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":	
	pass