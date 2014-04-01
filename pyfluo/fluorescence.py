from scipy import stats
import numpy as np
from time_series import TimeSeries

def dff_stim(seriess, stim=None, base_time=0.3):
    """Calculates delta-F over F using pre-stimulation baseline as F0.
    
    FUNCTION OUT OF DATE, LIKELY CONTAINS BUGS.
    
    Args:
        seriess (pyfluo.TimeSeries): should be passed from *compute_dff*
        stim (pyfluo.StimSeries): if mode is "stim," this argument represents the stimulation associated with *series*
        base_time (float): if mode is "stim," this argument represents the time before stimulation to be averaged as a base line F0.
        
    Returns:
        a TimeSeriesCollection of DFF signals, one per stimulation in *stim* (if *series* is a single TimeSeries), or a list thereof (if *series* is a list of TimeSeries).
    """
    if stim==None:
        raise Exception('DFF cannot be calculated using desired mode without stim_series.')
        
    dffs = []
    for sidx in range(seriess.n_series):
        series = seriess.get_series(sidx)
        traces_aligned = series.take(stim.stim_times, pad=(base_time,base_time))
        baselines = [np.mean(tr.take([-base_time,0.])) for tr in traces_aligned]
        for tr,bl in zip(traces_aligned, baselines):
            for idx,samp in enumerate(tr):
                tr[idx] = (samp - bl)/bl
        dffs.append(traces_aligned)
    
    if len(dffs) == 1:
        dffs = dffs[0]
        
    return dffs
def compute_dff(series, tau0=0.2, tau1=0.75, tau2=3.0, noise_filter=False):
    """Calculates delta-F over F using a sliding window method.
   
   **Parameters:** 
        * **series** (*pyfluo.TimeSeries*): series of which to compute delta F over F
        * **tau0** (*float*): see Jia et al. 2010
        * **tau1** (*float*): see Jia et al. 2010
        * **tau2** (*float*): see Jia et al. 2010
        * **noise_filter** (*bool*): include the final noise filtering step of the algorithm
        
    **Returns:**
        TimeSeries containing the DFF signal.
        
    **Notes:**
        Adapted from Jia et al. 2010 Nature Protocols
        
        The main adjustment not specified in the algorithm is how I deal with the beginning and end of the signal. When we're too close to the borders of the signal such that averages/baselines are subject to noise, I allow the function to look in the other direction (forward if at beginning, backward if at end) to make the signal more robust. This is reflected by the variables "forward" and "backward" in the calculation of f_bar and f_not.
    """
    
    tau0t = tau0
    tau1t = tau1
    tau2t = tau2
    
    dff = None
        
    tau0 = tau0t * series.fs
    tau1 = tau1t * series.fs
    tau2 = tau2t * series.fs
                
    f_bar = np.zeros(np.shape(series.data))
    for idx in range(len(series)):
        i1 = int(idx-round(tau1/2.))
        forward=0
        if i1<0:
            forward=abs(i1)
            i1=0
        i2 = int(idx+round(tau1/2.)) + forward
        backward=0
        if i2>=len(series):
            backward=i2-len(series)
            i2=len(series)-1
        integ = np.take( series.data, range(i1-backward,i2+1), axis=1)
        integ = np.mean(integ, axis=1)
        f_bar[:,idx] =  integ 
        
    f_not = np.zeros(np.shape(f_bar))
    for idx in range(len(series)):
        i1 = int(idx-tau2)
        forward=0
        if i1<0:
            forward=abs(i1) 
            i1=0
        search = np.take(f_bar, range(i1, idx+1+forward), axis=1)
        if np.size(search):
            f_not[:,idx] = np.min(search, axis=1) 
    
    r = (series.data - f_not) / f_not

    def w_func(x):
        return np.exp(-np.abs(x)/tau0)
    w = w_func(np.arange(0,len(series)))

    dff = np.zeros(np.shape(series.data))
    
    if noise_filter:
        for t in range(len(series)):
            numerator = np.sum(r[:,t::-1]*w[:t+1], axis=1)
            denominator = np.sum(w[:t+1])
            
            dff[:,t] = np.divide(numerator, denominator)              
    else:
        dff = r
    
    return TimeSeries(data=dff, time=series.time)

def subtract_background(data):
    #TODO: implement the nonhomogenous aspect   
    """
    Given a set of pixels each with a time value (rows are time points, columns are pixels), calculate and subtract background using the algorithm described in Chen et al 2006 Biophysical Journal.
    """
    
    y = data 
    y_bar = np.mean(y, axis=0)
    y_tilde = y - y_bar

    R = []
    for pix in range(np.shape(y_tilde)[1]):
        yi = y_tilde[:,pix][:,None]
        yy = np.dot(yi,yi.T)
        R.append(yy)
    R = np.sum(np.dstack(R), axis=2)
    
    eigvals, eigvecs = np.linalg.eig(R)
    f_tilde = np.real(eigvecs[:,np.argmax(eigvals)])[:,None]

    u = y_tilde.T.dot(f_tilde)

    m,yint,r,p,err = stats.linregress(np.squeeze(y_bar) ,np.squeeze(u))
    xint = -yint/m
    m_inverse = 1/m 

    f_bar = m_inverse 
    background = xint

    F = np.zeros(np.shape(y))
    for i in range(np.shape(y)[1]):
        F[:,i] = u[i]*(np.squeeze(f_tilde) + f_bar)

    '''
    #inhomogenous check - still incomplete
    for pix in range(np.shape(F)[1]):
        d = y_bar[pix] - f_bar*u[pix] - background
        var_n = np.var(y_tilde - u[pix]*f_tilde)
        var_d = (f_bar**2 +(1/np.shape(data)[0]))*var_n
        std_d = np.sqrt(var_d)
        print d<4*std_d
    '''

    return F
if __name__ == "__main__":  
    pass
