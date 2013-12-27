"""
Working notes:

With regard to rise times: at any given point, the rise time constant of the fluorescence signal is determined as a function of the change in Ca concentration that just occurred. With a large change, the rise time is very small (thus signal is fast), as shown by Richard's data. The odd case is when the calcium level actually drops, but the fluorescence *still* has some rise to achieve in order to catch up to it. What is the time constant then? For now, I'm making it use the time constant as if the change in Ca were zero, simply extrapolating from Richard's data. This means the longest possible time constant, so slow binding. I think this makes sense. (That point is implemented in rise_times.py)

In ca2f, the 1.0 multiplication factor, which determines sigmoid x-stretch, is arbitrary. The goal is to have it capture the true curve for GCaMP vs Ca, but I'm not sure which parameter about GCaMP controls this. For now, I keep it as 1 since this leaves all Ca levels in the linear range, which I think we assume to be true in experiments.
"""

from bdpyfluo.gdefs import *
from bdpyfluo import Trace
rand = np.random
from rise_times import RiseTimes as RT

class Model():
	def __init__(self, tau_ca_decay=0.040, tau_ind_decay=0.102, kd_ind=0.29, rf_ind=29.2, nh_ind=2.46, baseline_ca=0.180, ca_per_AP=0.050, sigma=0.5, with_noise = False):
		"""
		Initialize a generative model to produce a Ca2+ signal from a spike train.
		
		Physiological parameters:
			tau_ca_decay: all-sources-combined removal time constant of Ca2+ ions from intracellular environment (rise is considered a delta)
				estimating 40 ms for no good reason
			tau_ind_decay: Ca2+ indicator decay time constant
				estimating 71 ms HALF time from richard's data
				t1/2 = tau*ln(2) therefore tau = 102ms
			kd_ind: Ca2+ indicator Kd (dissocation constant), determines the centerpoint of the sigmoid curve that relates dF/F to [Ca]
				using 0.29uM from Richard's data for GCaMP6f 
			rf_ind: Ca2+ indicator dynamic range
				using 29.2 from Richard's data for GCaMP6f
			nh_ind: Ca2+ indicator nH value
				using 2.46 from Richard's data for GCaMP6f
			baseline_ca: baseline level of Ca2+ in cell. 
				Saftenku (2009) suggests 60nM, Richard's data has info with an assumption of 100nM 
				Using 180nM as of now for various reasons.
			ca_per_AP: concentration of Ca2+ influx per action potential in uM.
				Saftenku (2009) reports 200nM and 14uM. Experimenting.
		Computational parameters:
			sigma: standard deviation of gaussian that produces error in fluorescence signal
			fs: sample rate
		"""
		self.tau_ca_decay = tau_ca_decay
		self.kd_ind = kd_ind
		self.rf_ind = rf_ind
		self.nh_ind = nh_ind
		self.ca_per_AP = ca_per_AP
		self.base = baseline_ca
		self.sigma = sigma
		self.tau_ind_decay = tau_ind_decay
		self.with_noise = with_noise
		
		self.RT = RT()
	def stimulate(self, spike_signal):
		"""
		spike_signal: Trace object containing stimulation signal
		"""
		ca = Trace(spike_signal.time)
		dff_ideal = Trace(spike_signal.time)
		dff = Trace(spike_signal.time)
		
		#set initial values, making assumptions about signal prior to this vector
		ca[0] = self.base #assume Ca was at rest level
		dff_ideal[0] = self.ca2f(ca[0])
		dff[0] = dff_ideal[0] #assume enough time has passed such that dff currenty at ideal level
		
		for idx in range(1,len(ca)):
			#calculate Ca2+ level
			dt = ca.time[idx] - ca.time[idx-1]
			ca[idx] = self.base + (ca[idx-1] - self.base) * math.e**(-dt/self.tau_ca_decay) + spike_signal[idx-1]*self.ca_per_AP
			#calculate F_infinity, or the ideal fluorescence level if timecourse was instant
			dff_ideal[idx] = self.ca2f(ca[idx]) + self.with_noise * rand.normal(0,self.sigma**2)
			#calculate real fluorescence level based on temporal dynamics
			if dff_ideal[idx-1] - dff[idx-1] > 0: #on the rise
				dff[idx] = min(dff[idx-1] + dt/self.tau_ind_rise(ca[idx] - ca[idx-1]) * (dff_ideal[idx-1] - dff[idx-1]), dff_ideal[idx-1])
			elif dff_ideal[idx-1] - dff[idx-1] < 0: #on the decay
				dff[idx] = max(dff[idx-1] + dt/self.tau_ind_decay * (self.base - dff[idx-1]), dff_ideal[idx-1])
			else: #not rising or decaying
				dff[idx] = dff[idx-1]

		return dff,dff_ideal,ca
		
	def ca2f(self, ca):
		def sigmoid(x):
			#return dF/F given Ca concentration in uM
			return self.rf_ind * (x**self.nh_ind)/(self.kd_ind**self.nh_ind + x**self.nh_ind)
		return sigmoid(ca)
	def tau_ind_rise(self, ca):
		return 0.1
		return self.RT.get_rt(ca)
		
if __name__ == '__main__':
	import pylab as pl
	gm = Model()
	pl.ion()
	for i in [10**ii for ii in np.arange(-6,2,0.1)]:
		pl.plot(math.log10(i),gm.ca2f(i),'k.')
	pl.show()
	raw_input()