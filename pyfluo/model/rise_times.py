import pylab as pl
import scipy as sp
import numpy as np
import scipy.optimize
import math

#taken from GCaMP6 kinetics document, Figure 1c pink dotted line (GCaMP6f 100nM 37 degrees)

class RiseTimes():
	"""
	call get_rt, supplying the *change* in Ca concentration in micromolar as argument
	"""
	def __init__(self):
		self.fxn = self.make_fxn()

	def get_rt(self,delta_ca):
		if delta_ca > 1.9:
			return 0.34/math.log(2)
		elif delta_ca < 0.:
			return self.fxn(0.)
		else:
			return self.fxn(delta_ca)
		
	def make_fxn(self):
		delta_ca = np.array([0.14, 0.42, 0.86, 1.4, 1.95, 3.9]) #uM
		riset12 = np.array([0.34, 0.15, 0.07, 0.03, 0.02, 0.02]) #(s)
		rise_tau = np.array([i/math.log(2) for i in riset12])
		C = 0.
		
		def fit_exp_linear(x, y, C=C):
			 y = y - C
			 y = np.log(y)
			 K, A_log = np.polyfit(x, y, 1)
			 A = np.exp(A_log)
			 return A, K
		
		A, K = fit_exp_linear(delta_ca[:5], rise_tau[:5])

		def func(x):
			return A * np.exp(K * x) + C

		return func
if __name__ == '__main__':
	rt = RiseTimes()
	pl.ion()
	xs = []
	ys = []
	for i in np.arange(0,7,0.01):
		xs.append(i)
		ys.append(rt.get_rt(i))
	pl.plot(xs,ys)
		
	delta_ca = np.array([0.14, 0.42, 0.86, 1.4, 1.95, 3.9]) #uM
	riset12 = np.array([0.34, 0.15, 0.07, 0.03, 0.02, 0.02]) #(s)
	pl.plot(delta_ca,riset12,'r*')
	
	pl.show()
	raw_input()