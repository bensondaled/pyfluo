import numpy as np
import pylab as pl
from matplotlib import path as mpl_path
import pickle

class ROISet(object):
	def __init__(self):
		self.rois = []
	def add(self, roi):
		self.rois.append(roi)
	def remove(self, idxs=None):
		if not idxs:
			idxs=range(len(self))
		for roi in idxs:
			self.rois[roi] = None
		self.rois = [r for r in self.rois if r]
	
	def __getitem__(self, idx):
		return self.rois[idx]
	def __len__(self):
		return len(self.rois)

class ROI(object):
	def __init__(self, shape, pts):
		
		path = mpl_path.Path(pts)
		mask = np.ones(shape,dtype=bool)
		for ridx,row in enumerate(mask):
			for cidx,pt in enumerate(row):
				if path.contains_point([ridx,cidx]):
					mask[cidx,ridx] = False
					
		self.mask = mask
		self.pts = pts
	
	def show(self):
		pl.imshow(np.dstack())
		
		pl.show()
	
	# Saving data
	def save(self, file_name):
		f = open(file_name+'.pfroi', 'wb')
		pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)