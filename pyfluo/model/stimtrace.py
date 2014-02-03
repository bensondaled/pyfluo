from bdpyfluo import Trace
from bdpyfluo.gdefs import *

class StimTrace(Trace):
	def __init__(self, *args):
		super(StimTrace, self).__init__(*args)
	def add_stim(self, start_time, firing_rate, duration, magnitude=1.0):
		start_idx = float(t2idx(self.time, start_time))
		end_time = start_time+duration
		end_idx = float(t2idx(self.time, end_time))
		dur_frames = end_idx - start_idx
		total_aps = duration * firing_rate
		ap_per_frame = total_aps / dur_frames
		self.data[start_idx:end_idx] = ap_per_frame * magnitude