import numpy as np
import cv2
import pylab as pl
import warnings

class TSBase(np.ndarray):
    def __new__(cls, data, n_dims=0, time=None, info=None, Ts=None, dtype=np.float64, interactive_backend=pl):
        obj = np.asarray(data, dtype=dtype).view(cls)
      
        ### data
        assert obj.ndim in n_dims, 'Movie data has invalid dimensions'

        ### time
        obj.time, obj.Ts = time, Ts
        assert obj.time==None or type(obj.time) in [list, np.ndarray], 'Time vector not in proper format.'
        assert obj.Ts==None or type(obj.Ts) in [float, int], 'Ts (sampling interval) not in proper format.'
        if obj.time == None:
            if obj.Ts == None:
                obj.Ts = 1.
                warnings.warn('No time information supplied; Ts assigned as 1.0')
            obj.time = obj.Ts*np.arange(0,len(obj), dtype=np.float64)
        elif obj.time != None:
            obj.time = np.asarray(obj.time, dtype=np.float64)
            if obj.Ts == None:
                obj.Ts = np.mean(obj.time[1:]-obj.time[:-1])
                warnings.warn('Sampling interval Ts inferred as mean of time vector intervals.')
            elif obj.Ts != None:
                if obj.Ts != np.mean(obj.time[1:]-obj.time[:-1]):
                    warnings.warn('Sampling interval does not match time vector. This may affect future operations.')
        obj.fs = 1./obj.Ts
        
        ### info
        obj.info = info
        if obj.info == None:
            obj.info = [None for _ in xrange(len(obj))]

        ### other attributes
        obj.interactive_backend = interactive_backend

        ### consistency checks
        assert len(obj) == len(obj.time), 'Data and time vectors are different lengths.'
        assert len(obj) == len(obj.info), 'Data and info vectors are different lengths.'

        return obj
    
    def __array_finalize__(self, obj):
        _custom_attrs = ['time','info','Ts','interactive_backend','fs']
        for ca in _custom_attrs:
            setattr(self, ca, getattr(obj, ca, None))

    def __array_wrap__(self, out, context=None):
        return np.ndarray.__array_wrap__(self, out, context)

    def __getslice__(self,start,stop):
        #This is a bug fix, solution found here: http://stackoverflow.com/questions/14553485/numpy-getitem-delayed-evaluation-and-a-1-not-the-same-as-aslice-1-none
        return self.__getitem__(slice(start,stop))
    def __getitem__(self,idx):
        out = super(TSBase,self).__getitem__(idx)
        return out
        #dummy_data = self.view(np.ndarray)
        #dummy_data = dummy_data.__getitem__(idx)
        #new_obj = self.copy()
        #new_obj.data = dummy_data
        #return new_obj

    def t2i(self, t):
        #returns the index most closely associated with time t
        return np.argmin(np.abs(self.time - t))
   # def _take(self, time_range, pad=(0.,0.), reset_time=True, safe=True, output_class=None, take_axis=0):
   #     """Takes time range *inclusively* on both ends.
   #     
   #     """
   #     
   #     t1 = time_range[0] - pad[0]
   #     t2 = time_range[1] + pad[1]
   #     if t1 > t2:
   #         t1,t2 = t2,t1
   #     idx1 = self.time_to_idx(t1, method=round) #np.floor if inclusion of time point is more important than proximity
   #     idx2 = self.time_to_idx(t2, method=round) #np.ceil if inclusion of time point is more important than proximity
   #     
   #     #Safe:
   #     #purpose: to avoid getting different length results despite identical time ranges, because of rounding errors
   #     if safe:
   #         duration = t2-t1
   #         duration_idx = int(round(self.fs * duration))
   #         idx2 = idx1 + duration_idx
   #     #End Safe
   #             
   #     t = np.take(self.time, range(idx1,idx2+1), mode='clip')
   #     if idx1<0:  t[:-idx1] = [t[-idx1]-i*self.Ts for i in range(-idx1,0,-1)]
   #     if idx2>len(self.time)-1:
   #         t[-(idx2-(len(self.time)-1)):] = [t[-1]+i*self.Ts for i in range(1, idx2-(len(self.time)-1)+1)]
   #     if reset_time:  t = t - time_range[0]
   #     
   #     data = np.take(self.data, range(idx1,idx2+1), axis=take_axis, mode='clip')
   #     if idx1<0:  data[...,:-idx1] = None
   #     if idx2>len(self.time)-1:   data[...,-(idx2-(len(self.time)-1)):] = None
   #     
   #     add_start=0
   #     add_end=0
   #     if idx1<0:  
   #         add_start=abs(idx1)
   #         idx1=0
   #     if idx2>len(self.info)-1:   
   #         add_end=idx2-(len(self.info)-1)
   #         idx2=len(self.info)-1
   #     info =  self.info[idx1:idx2+1]
   #     info = [None for i in range(add_start)]+ info +[None for i in range(add_end)]
   #             
   #     if output_class==None:
   #         output_class = self.__class__
   #     return output_class(data=data, time=t, info=info)
