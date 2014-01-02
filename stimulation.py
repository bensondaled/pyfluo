from time_series import TimeSeries
import numpy as np
import pickle
import time as pytime

class StimSeries(TimeSeries):
	def __init__(self, *args, **kwargs):
		self.name = pytime.strftime("StimSeries-%Y%m%d_%H%M%S")
		
		down_sample = kwargs.pop('down_sample', 64) #if not None, give n for resample
				
		super(StimSeries, self).__init__(*args, **kwargs)

		self.original_data = self.data

		if down_sample:
			self.resample(down_sample, in_place=True)
		self.raw_data = np.copy(self.data)
	
		self.stim_idxs = None
		self.stim_times = None
				
		self.convert_to_delta()
		self.process_stim_times()

	def convert_to_delta(self,min_sep_time=0.100,baseline_time=0.1):
		self.start_idxs = []
		self.end_idxs = []
		
		#assumes that signal begins at baseline
		#min_sep_time argument is the minimum TIME between two different triggers in seconds
		baseline_sample = baseline_time * self.fs
		base = np.average(self.data[:baseline_sample])
		base_std = np.average(self.data[:baseline_sample])
		thresh = base+3.*base_std
		min_sep = min_sep_time * self.fs
		up = False
		idxs_down = 0
		delta_sig = np.zeros(self.data.size)
		for idx,d in enumerate(self.data):
			if not up and d>thresh:
				up = True
				delta_sig[idx] = 1.
				self.start_idxs.append(idx)
			elif up and d<thresh:
				if idxs_down > min_sep or idx==len(self.data)-1:
					delta_sig[idx-idxs_down:idx+1] = 0.
					self.end_idxs.append(idx-idxs_down)
					up = False
					idxs_down = 0
				else:
					idxs_down += 1
					delta_sig[idx] = 1.
			elif up:
				delta_sig[idx] = 1.
				idxs_down = 0
		self.data = delta_sig
		#self.data = map(lambda d: d>thresh,self.data)
	def process_stim_times(self, min_duration = 0.1, roundd=True):
		try:
			self.stim_idxs = [[self.start_idxs[i], self.end_idxs[i]] for i in range(len(self.start_idxs))]
		except Exception:
			print "There was an error parsing the stimulation signal. Try viewing it manually to determine problem."
		self.stim_times = [[self.time[idx] for idx in pulse] for pulse in self.stim_idxs]
		
		#correct for min duration
		self.stim_idxs = [idxs for idxs,times in zip(self.stim_idxs,self.stim_times) if times[1]-times[0] >= min_duration]
		self.stim_times = [times for times in self.stim_times if times[1]-times[0] >= min_duration]
		
		self.avg_duration = np.average([t[1]-t[0] for t in self.stim_times])
		self.ustim_idxs = [[s[0], s[0] + round(self.avg_duration*self.fs)] for s in self.stim_idxs]
		self.ustim_times = [[self.time[i] for i in pulse] for pulse in self.ustim_idxs] 