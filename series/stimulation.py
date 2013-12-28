from time_series import TimeSeries
import numpy as np

class StimSeries(TimeSeries):
	def __init__(self, *args, **kwargs):
		uniform = kwargs.pop('uniform', True)
		
		super(StimSeries, self).__init__(*args, **kwargs)
		self.original_data = self.data
	
		self.stim_idxs = None
		self.stim_times = None
		
		self.convert_to_delta()
		self.calc_stim_times()
		
		if uniform:
			self.uniformize()
	def convert_to_delta(self,min_sep_time=0.100,baseline_sample=50):
		#assumes that signal begins at baseline
		#min_sep_time argument is the minimum TIME between two different triggers in seconds
		self.data = self.original_data
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
			elif up and d<thresh:
				if idxs_down > min_sep:
					delta_sig[idx-min_sep:idx+1] = 0.
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
	def calc_stim_times(self, min_duration = 0.1):
		self.stim_idxs = []
		idxs = []
		for idx,d in enumerate(self.data[1:]):
			if d + self.data[idx] == 1:
				idxs.append(idx+1)
				if len(idxs) == 2:
					self.stim_idxs.append(idxs)
					idxs = []
		self.stim_times = [[self.time[idx] for idx in pulse] for pulse in self.stim_idxs]
		
		#correct for min duration
		self.stim_idxs = [idxs for idxs,times in zip(self.stim_idxs,self.stim_times) if times[1]-times[0] > min_duration]
		self.stim_times = [times for times in self.stim_times if times[1]-times[0] > min_duration]
	def uniformize(self, ndigits=2):
		u_stim_times = []
		u_stim_idxs = []
		for stimt,stimi in zip(self.stim_times, self.stim_idxs):
			dur = round(stimt[1]-stimt[0], ndigits)
			u_stim_times.append([stimt[0], stimt[0]+dur])
			u_stim_idxs.append([self.time_to_idx(t) for t in u_stim_times[-1]])
		self.stim_times = u_stim_times
		self.stim_idxs = u_stim_idxs