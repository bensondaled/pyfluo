from soup import *

tr = np.load('traces.npy')
tr = tr[:,2]
tr = np.tile(tr, 5)
tr = tr[:2**13]

tr = np.sort(tr)
