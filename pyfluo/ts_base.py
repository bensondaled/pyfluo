import numpy as np
import cv2
import pylab as pl
from scipy.signal import resample as sp_resample
import warnings

_numeric_types = [int, float, long, np.float16, np.float32, np.float64, np.float128, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]

class TSBase(np.ndarray):
    __array_priority__ = 1. #ensures that ufuncs return ROI class instead of np.ndarrays
    _custom_attrs = ['time', 'info', 'Ts', 'fs', '_ndim']
    _custom_attrs_slice = ['time', 'info'] # should all be numpy arrays
    def __new__(cls, data, _ndim=0, time=None, info=None, Ts=None, dtype=np.float32):
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
        
        ### info
        obj.info = info
        if obj.info == None:
            obj.info = np.array([None for _ in xrange(len(obj))])

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
    def _update(self):
        self.Ts = np.mean(np.diff(self.time))
        self.fs = 1./self.Ts
    def reset_time(self):
        self.time = self.time - self.time[0]
    def t2i(self, t):
        #returns the index most closely associated with time t
        if any([isinstance(t,ty) for ty in [list,tuple,np.ndarray]]):
            t = np.asarray(t)
            end_shape = t.shape
            tt = t.flatten()
            idxs = np.array([np.argmin(np.abs(self.time - ti)) for ti in tt]).reshape(end_shape)
            return idxs
        elif type(t) in _numeric_types:
            return np.argmin(np.abs(self.time - t)) 
    def take(self, times, duration=None, step=1, pad=(0,0), enforce_duration=True, as_copy=True):
        """Extract data from the time series between the times given

        Parameters
        ----------
        times : list-like
            list of time points, ex. [t0, t1]
            or [t0_0, t0_1, t0_2...] if duration is specified
        duration : list-like, number
            value or list of durations associated with start times in *times*
            if single value, taken to be same for all start times
        step : int
            step for slices
        pad : list-like, number
            time to include before or after times supplied
            single number is used on both ends
        enforce_duration : bool
            when single duration given, enforce that every slice is the same size, by using object's fs instead of time vector
        as_copy: bool
            return the taken slices as copies

        Returns
        -------
        Extracted data

        Example
        -------
        Suppose you have a list of event times
        event_times = [0.6,1.2,3.5...]
        and you want to extract the 3 seconds after each time, with a 0.1s pad
        result = trace.take(event_times, durations=3.0, pad=0.1)
        """
        times = np.asarray(times)
        replace_t1_flag = False

        if duration is None:
            if times.ndim == 1:
                times = np.array([times])
            assert times.ndim == 2, 'times must be 2-item time boundaries unless durations are specified'
            assert times.shape[1] == 2, 'times must be 2-item boundaries unless durations are specified'
        elif duration is not None:
            duration = np.asarray(duration)
            if times.ndim == 0:
                times = np.array([times])
                assert duration.ndim == 0, 'More durations than times supplied'
            assert times.ndim == 1, 'times must be 1-item start times when durations are specified'
            assert ((duration.ndim==0) or (len(duration)==len(times))), 'duration must be single value or same number of values as times'
            if duration.ndim == 0:
                if enforce_duration:
                    replace_t1_flag = True
                duration = np.repeat(duration, len(times))
            times = np.array([times,times+duration]).T

        pad = np.asarray(pad)
        if pad.ndim == 0:
            pad = np.array([pad,pad])
        pad[0] = -pad[0]
        times += pad

        idxs = self.t2i(times)
        if replace_t1_flag:
            idxs[:,1] = idxs[:,0] + duration*self.fs

        slices = [slice(i0,i1,step) for i0,i1 in idxs]
        sliced = [self[sl] for sl in slices]
        if as_copy:
            sliced = [i.copy() for i in sliced]

        if len(sliced)==1:
            res = sliced[0]
        else:
            res = sliced

        return res

    def resample(self, *args, **kwargs):
        """Resample time series object using scipy's resample

        Parameters are those of scipy.signal.resample, with *num* (number of samples in resampled result) as the only mandatory parameter
        Difference is that this takes object's time vector into account automatically

        If single param (num) is a float, taken to be multiple for current number of frames.
        """
        args = list(args)
        args.reverse()
        n = args.pop()
        args.reverse()
        if type(n) in [float,np.float16,np.float32,np.float64,np.float128]:
            n = np.round(len(self)*n)

        if 't' not in kwargs or kwargs['t']==None:
            kwargs['t'] = self.time
        new_data,new_time = sp_resample(self, n, *args, axis=0, **kwargs)
        return self.__class__(data=new_data, time=new_time)
