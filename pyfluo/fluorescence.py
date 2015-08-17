#TODO: choose a df/f method

import numpy as np
import warnings
from util import sliding_window as sw
from util import ProgressBar

def compute_dff(data, percentile=8., window_size=1., step_size=.025, subtract_minimum=True, pad_mode='edge', in_place=False, return_f0=False, prog_bar=True):
    """Compute delta-f-over-f

    Computes the percentile-based delta-f-over-f along the 0th axis of the supplied data.

    Parameters
    ----------
    data : np.ndarray
        n-dimensional data (DFF is taken over axis 0)
    percentile : float 
        percentile of data window to be taken as F0
    window_size : float
        size of window to determine F0, in seconds
    step_size : float
        size of steps used to determine F0, in seconds
    subtract_minimum : bool
        substract minimum value from data before computing
    pad_mode : str 
        mode argument for np.pad, used to specify F0 determination at start of data

    Returns
    -------
    Data of the same shape as input, transformed to DFF
    """
    if not in_place:
        data = data.copy()

    window_size = int(window_size*data.fs)
    step_size = int(step_size*data.fs)
    
    if window_size<1:
        warnings.warn('Requested a window size smaller than sampling interval. Using sampling interval.')
        window_size = 1.
    if step_size<1:
        warnings.warn('Requested a step size smaller than sampling interval. Using sampling interval.')
        step_size = 1.

    if subtract_minimum:
        data -= data.min()
     
    pad_size = window_size - 1
    pad = ((pad_size,0),) + tuple([(0,0) for _ in xrange(data.ndim-1)])
    padded = np.pad(data, pad, mode=pad_mode)

    out_size = ((len(padded) - window_size) // step_size) + 1
    if prog_bar:    pbar = ProgressBar(maxval=out_size).start()
    f0 = []
    for idx,win in enumerate(sw(padded, ws=window_size, ss=step_size)):
        f0.append(np.percentile(win, percentile, axis=0))
        if prog_bar:    pbar.update(idx)
    f0 = np.repeat(f0, step_size, axis=0)[:len(data)]
    if prog_bar:    pbar.finish()
    
    if return_f0:
        return ( (data-f0)/f0, f0 )
    else:
        return (data-f0)/f0

def computeDFF_AG(self,secsWindow=5,quantilMin=8,subtract_minimum=False,squared_F=True):
    """ 
    compute the DFF of the movie
    In order to compute the baseline frames are binned according to the window length parameter
    and then the intermediate values are interpolated. 
    Parameters
    ----------
    secsWindow: length of the windows used to compute the quantile
    quantilMin : value of the quantile

    """
    
    print "computing minimum ..."; sys.stdout.flush()
    minmov=np.min(self.mov)
    if subtract_minimum:
        self.mov=self.mov-np.min(self.mov)+.1
        minmov=np.min(self.mov)

    assert(minmov>0),"All pixels must be nonnegative"                       
    numFrames,linePerFrame,pixPerLine=np.shape(self.mov)
    downsampfact=int(secsWindow/self.frameRate);
    elm_missing=int(np.ceil(numFrames*1.0/downsampfact)*downsampfact-numFrames)
    padbefore=np.floor(elm_missing/2.0)
    padafter=np.ceil(elm_missing/2.0)
    print 'Inizial Size Image:' + np.str(np.shape(self.mov)); sys.stdout.flush()
    self.mov=np.pad(self.mov,((padbefore,padafter),(0,0),(0,0)),mode='reflect')
    numFramesNew,linePerFrame,pixPerLine=np.shape(self.mov)
    #% compute baseline quickly
    print "binning data ..."; sys.stdout.flush()
    movBL=np.reshape(self.mov,(downsampfact,int(numFramesNew/downsampfact),linePerFrame,pixPerLine));
    movBL=np.percentile(movBL,quantilMin,axis=0);
    print "interpolating data ..."; sys.stdout.flush()   
    print movBL.shape        
    movBL=scipy.ndimage.zoom(np.array(movBL,dtype=np.float32),[downsampfact ,1, 1],order=0, mode='constant', cval=0.0, prefilter=False)
    
    #% compute DF/F
    if squared_F:
        self.mov=(self.mov-movBL)/np.sqrt(movBL)
    else:
        self.mov=(self.mov-movBL)/movBL
        
    self.mov=self.mov[padbefore:len(movBL)-padafter,:,:]; 
    print 'Final Size Movie:' +  np.str(self.mov.shape)          
    
