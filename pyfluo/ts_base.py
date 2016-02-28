import numpy as np
import pylab as pl
import cv2, warnings, sys
from scipy.signal import resample as sp_resample
from . import fluorescence

_pyver = sys.version_info.major
if _pyver == 3:
    xrange = range


class TSBase(np.ndarray):
    __array_priority__ = 1.
    _custom_attrs = ['Ts']

    def __new__(cls, data, _ndim=0, Ts=None, dtype=np.float32):
        obj = np.asarray(data, dtype=dtype).view(cls)
      
        ### data
        assert obj.ndim in _ndim, 'Input data has invalid dimensions'
        obj._ndim = _ndim

        ### time
        obj.time, obj.Ts = time, Ts
        assert obj.time is None or type(obj.time) in [list, np.ndarray], 'Time vector not in proper format.'
        assert obj.Ts==None or type(obj.Ts) in _numeric_types, 'Ts (sampling interval) not in proper format.'
        if obj.time is None:
            if obj.Ts == None:
                obj.Ts = 1.
                warnings.warn('No time information supplied; Ts assigned as 1.0')
            obj.time = obj.Ts*np.arange(0,len(obj), dtype=np.float64)
        elif not obj.time is None:
            obj.time = np.asarray(obj.time, dtype=np.float64)
            if obj.Ts == None:
                obj.Ts = np.mean(np.diff(obj.time))
                warnings.warn('Sampling interval Ts inferred as mean of time vector intervals.')
            elif obj.Ts != None:
                if round(obj.Ts,7) != round(np.mean(obj.time[1:]-obj.time[:-1]),7):
                    warnings.warn('Sampling interval does not match time vector. This may affect future operations.')
        obj.fs = 1./obj.Ts
        
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

        if False and self.ndim < max(self._ndim): # this is Falsed out until I rediscover why it was introduced
            pass
        elif self.ndim in self._ndim:
            if type(idx) in _numeric_types:
                return out.view(np.ndarray)
            elif type(idx) == slice:
                idxi = idx
            elif type(idx)==tuple or all([type(i)==slice for i in idx]):
                idxi = idx[0]
            else:
                idxi = idx
            for ca in TSBase._custom_attrs_slice:
                setattr(out, ca, getattr(out, ca, None)[idxi].copy())

        out._update() 
        return out
    
