import numpy as np
import cv2
import pylab as pl
from scipy.signal import resample as sp_resample
import warnings

class TSBase(np.ndarray):
    __array_priority__ = 1. #ensures that ufuncs return ROI class instead of np.ndarrays
    _custom_attrs = ['time', 'info', 'Ts', 'fs', '_ndim']
    _custom_attrs_slice = ['time', 'info']
    def __new__(cls, data, _ndim=0, time=None, info=None, Ts=None, dtype=np.float32):
        obj = np.asarray(data, dtype=dtype).view(cls)
      
        ### data
        assert obj.ndim in _ndim, 'Input data has invalid dimensions'
        obj._ndim = _ndim

        ### time
        obj.time, obj.Ts = time, Ts
        assert obj.time is None or type(obj.time) in [list, np.ndarray], 'Time vector not in proper format.'
        assert obj.Ts==None or type(obj.Ts) in [float, int, long, np.float16, np.float32, np.float64], 'Ts (sampling interval) not in proper format.'
        if obj.time is None:
            if obj.Ts == None:
                obj.Ts = 1.
                warnings.warn('No time information supplied; Ts assigned as 1.0')
            obj.time = obj.Ts*np.arange(0,len(obj), dtype=np.float32)
        elif not obj.time is None:
            obj.time = np.asarray(obj.time, dtype=np.float32)
            if obj.Ts == None:
                obj.Ts = np.mean(obj.time[1:]-obj.time[:-1])
                warnings.warn('Sampling interval Ts inferred as mean of time vector intervals.')
            elif obj.Ts != None:
                if round(obj.Ts,7) != round(np.mean(obj.time[1:]-obj.time[:-1]),7):
                    warnings.warn('Sampling interval does not match time vector. This may affect future operations.')
        obj.fs = 1./obj.Ts
        
        ### info
        obj.info = info
        if obj.info == None:
            obj.info = [None for _ in xrange(len(obj))]

        ### other attributes
        #(none)

        ### consistency checks
        assert len(obj) == len(obj.time), 'Data and time vectors are different lengths.'
        assert len(obj) == len(obj.info), 'Data and info vectors are different lengths.'

        return obj
    def __array_finalize__(self, obj):
        if obj is None:
            return

        for ca in TSBase._custom_attrs:
            setattr(self, ca, getattr(obj, ca, None))

    def __array_wrap__(self, out, context=None):
        return np.ndarray.__array_wrap__(self, out, context)

    def __getslice__(self,start,stop):
        #classic bug fix
        return self.__getitem__(slice(start,stop))

    def __getitem__(self,idx):
        out = super(TSBase,self).__getitem__(idx)
        if not isinstance(out,TSBase):
            return out

        if self.ndim < max(self._ndim): #changed to max from min
            pass
        elif self.ndim in self._ndim:
            if type(idx) in (int, float, np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
                return out.view(np.ndarray)
            elif type(idx) == slice:
                idxi = idx
            elif type(idx)==tuple or all([type(i)==slice for i in idx]):
                idxi = idx[0]
            else:
                idxi = idx
            for ca in TSBase._custom_attrs_slice:
                setattr(out, ca, getattr(out, ca, None)[idxi])
        
        return out
    def reset_time(self):
        self.time = self.time - self.time[0]
    def t2i(self, t):
        #returns the index most closely associated with time t
        return np.argmin(np.abs(self.time - t))
    def resample(self, *args, **kwargs):
        """Resample time series object using scipy's resample

        Parameters are those of scipy.signal.resample, with *num* (number of samples in resampled result) as the only mandatory parameter
        """
        if 't' not in kwargs or kwargs['t']==None:
            kwargs['t'] = self.time
        new_data,new_time = sp_resample(self, *args, axis=0, **kwargs)
        return self.__class__(data=new_data, time=new_time)
