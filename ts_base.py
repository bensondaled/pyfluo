import numpy as np
import copy

class TSBase(object):
	def _take(self, time_range, pad=(0.,0.), reset_time=True, safe=True, output_class=None):
		"""Takes time range *inclusively* on both ends.
		
		"""
		take_axis = len(np.shape(self.data))-1
		
		t1 = time_range[0] - pad[0]
		t2 = time_range[1] + pad[1]
		if t1 > t2:
			t_temp = t1
			t1 = t2
			t2 = t_temp
		idx1 = self.time_to_idx(t1, mode=round) #np.floor if inclusion of time point is more important than proximity
		idx2 = self.time_to_idx(t2, mode=round) #np.ceil if inclusion of time point is more important than proximity
		
		#Safe:
		#purpose: to avoid getting different length results despite identical time ranges, because of rounding errors
		if safe:
			duration = t2-t1
			duration_idx = int(self.fs * duration)
			idx2 = idx1 + duration_idx
		#End Safe
				
		t = np.take(self.time, range(idx1,idx2+1), mode='clip')
		if idx1<0:	t[:-idx1] = [t[-idx1]-i*self.Ts for i in range(-idx1,0,-1)]
		if idx2>len(self.time)-1:
			t[-(idx2-(len(self.time)-1)):] = [t[-1]+i*self.Ts for i in range(1, idx2-(len(self.time)-1)+1)]
		if reset_time:	t = t - time_range[0]
		
		data = np.take(self.data, range(idx1,idx2+1), axis=take_axis, mode='clip')
		if idx1<0:	data[:-idx1] = None
		if idx2>len(self.time)-1:	data[-(idx2-(len(self.time)-1)):] = None
		
		add_start=0
		add_end=0
		if idx1<0:	
			add_start=abs(idx1)
			idx1=0
		if idx2>len(self.info)-1:	
			add_end=idx2-(len(self.info)-1)
			idx2=len(self.info)-1
		info =  self.info[idx1:idx2+1]
		info = [None for i in range(add_start)]+ info +[None for i in range(add_end)]
				
		if output_class==None:
			output_class = self.__class__
		return output_class(data=data, time=t, info=info)
	def time_to_idx(self, t, mode=round):
		t = float(t)
		time_step = t - self.time[0]
		idx = mode(time_step*self.fs)
		return int(idx)
	def copy(self):
		"""Return a deep copy of this object.
		"""
		return copy.deepcopy(self)