from time_series import TimeSeries
import numpy as np
import pickle
import time as pytime
from pyfluo.util import *

class StimSeries(TimeSeries):
    """A time series specialized for storing stimulation data.
    
    Essentially, this class takes a high-density stimulation signal and simplifies it by downsampling, binarizing, and optionally uniformizing stimulation events.
    
    **Attributes:**
        * **raw_data** (*np.ndarray*): the (possibly down-sampled) data in its raw form, before conversion to a binary signal.
        
        * **stim_idxs** (*list*): a list of value pairs ``(start, end)`` indicating the indices of the time series data at which a stimulus started and ended.
        
        * **stim_times** (*list*): a list of value pairs ``(start, end)`` indicating the time points at which a stimulus started and ended.
        
        * **stim_durations** (*list*): of list of values indicating the duration of each stimulus.
        
        * **example** (*TimeSeries*): an example stimulation created by taking the mean of all stimuli.
        
        * **name** (*str*): a unique name generated for the object when instantiated
        
    """
    def __init__(self, *args, **kwargs):
        """Initialize a StimSeries object.
        
        **Parameters:**
            * **down_sample** (*int*): factor by which to down sample signal before processing. Defaults to 64, meaning that upon resampling, every 64th sample is taken. If ``None``, does not down sample.
            * **uniform** (*bool* / *int*): enforces that stimulation durations be equal by rounding them to the nearest *uniform* digits. Start times of stimulation events are completely perserved, while end times are adjusted slightly to allow for easier behaviour during analysis. Defaults to ``1``. Note that if ``tunit=='s'``, this corresponds to rounding to the nearest 100ms.
        
        (see TimeSeries.__init__ for complete signature)
        
        """
        uniform = kwargs.pop('uniform', True)
        down_sample = kwargs.pop('down_sample', 64) #if not None, give n for resample
        
        super(StimSeries, self).__init__(*args, **kwargs)
        #self.original_data = self.data #if you wanted the original data

        if down_sample:
            self.resample(down_sample, in_place=True)
        #self.raw_data = np.copy(self.data) #if you want to store the downsampled original data
    
        self.stim_idxs = None
        self.stim_times = None

        try:
            self.convert_to_delta()
            self.process_stim_times()

            if uniform:
                self.uniformize(ndigits=uniform)
           
            self.stim_durations =   [i[1]-i[0] for i in self.stim_times]
            self.example = self.take(self.stim_times, pad=(0.1,0.1)).merge()
        except:
            raise Exception('This data could not be converted to a StimSeries. This likely means the signal does not consist of clear stimulation peaks. To view the signal, try storing it in a TimeSeries, and plot that to inspect it.')

    def take(self, *args, **kwargs):
        return super(StimSeries, self).take(*args, output_class=TimeSeries, **kwargs)
    def convert_to_delta(self,min_sep_time=0.100,baseline_time=0.1):
        self.start_idxs = []
        self.end_idxs = []
        
        #assumes that signal begins at baseline
        #min_sep_time argument is the minimum TIME between two different triggers in seconds
        baseline_sample = int(baseline_time * self.fs)
        base = np.average(self[:baseline_sample])
        base_std = np.average(self[:baseline_sample])
        thresh = base+3.*base_std
        min_sep = min_sep_time * self.fs
        up = False
        idxs_down = 0
        delta_sig = np.zeros(len(self))

        for idx,d in enumerate(self):
            if not up and d>thresh:
                up = True
                delta_sig[idx] = 1.
                self.start_idxs.append(idx)
            elif up and d<thresh:
                if idxs_down > min_sep or idx==len(self)-1:
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
    def process_stim_times(self, min_duration = 0.1, roundd=True):
        try:
            self.stim_idxs = [[self.start_idxs[i], self.end_idxs[i]] for i in xrange(len(self.start_idxs))]
        except:
            print "There was an error parsing the stimulation signal. Try viewing it manually to determine problem."
        self.stim_times = [[self.time[idx] for idx in pulse] for pulse in self.stim_idxs]
        
        #correct for min duration
        self.stim_idxs = [idxs for idxs,times in zip(self.stim_idxs,self.stim_times) if times[1]-times[0] >= min_duration]
        self.stim_times = [times for times in self.stim_times if times[1]-times[0] >= min_duration]

    def uniformize(self, ndigits=2):
        #Important: with, for example, ndigits=1, any stimulation duration that's not a multiple of 100ms is rounded to one that is
        durations = []
        u_stim_times = []
        u_stim_idxs = []
        
        durations = [round(s[1]-s[0],ndigits) for s in self.stim_times]
        durations_idx = [self.fs*dur for dur in durations]
        u_stim_idxs = [[i[0], i[0]+idx] for idx,i in zip(durations_idx,self.stim_idxs)]
        u_stim_times = [[self.time[idx] for idx in pulse] for pulse in u_stim_idxs]

        self.stim_times = u_stim_times
        self.stim_idxs = u_stim_idxs
