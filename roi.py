import numpy as np
import pylab as pl
from matplotlib import path as mpl_path
import matplotlib.cm as mpl_cm
import pickle
import time as pytime

class ROISet(object):
	def __init__(self, rois=None):
		self.name = pytime.strftime("ROISet-%Y%m%d_%H%M%S")
		self.shape = None
		
		self.rois = []
		if type(rois) == ROI:
			self.rois.append(rois)
			self.shape = rois.shape
		elif type(rois) == list:
			self.rois = rois
			self.shape = rois[0].shape
			if not all([roi.shape == self.shape for roi in self.rois]):
				raise Exception('Provided ROIs have inconsistent shapes.')
				
	def add(self, roi):
		if not self.shape:
			self.rois.append(roi)
			self.shape = roi.shape
		elif self.shape and roi.shape == self.shape:
			self.rois.append(roi)
		elif roi.shape != self.shape:
			print "ROI not added: shape is inconsistent with ROISet."
	def remove(self, idxs=None):
		if idxs==None:
			idxs=range(len(self))
		for roi in idxs:
			self.rois[roi] = None
		self.rois = [r for r in self.rois if r]
		
	def show(self, mode='mask', labels=True):
		if len(self):
			if mode == 'mask':
				data = np.sum(np.dstack([r.mask for r in self]),axis=2)
				data = np.ma.masked_where(data == 1, data)
				ax = pl.gca()
				ax.imshow(data)
			elif mode == 'pts':
				for roi in self:
					roi = roi.pts
					pl.scatter(roi[:,0], roi[:,1])
				pl.xlim([0, self.shape[1]])
				pl.ylim([self.shape[0], 0])
			if labels:
				for idx,roi in enumerate(self):
					pl.text(roi.center[0], roi.center[1], str(idx), color='white', weight='bold')
			# pl.gca().xaxis.set_ticks_position('none')
			# pl.gca().yaxis.set_ticks_position('none')
			pl.show()
	
	def __getitem__(self, idx):
		return self.rois[idx]
	def __len__(self):
		return len(self.rois)

class ROI(object):
	def __init__(self, shape, pts):
		self.name = pytime.strftime("ROI-%Y%m%d_%H%M%S")
		self.shape = shape
		
		path = mpl_path.Path(pts)
		mask = np.ones(shape,dtype=bool)
		for ridx,row in enumerate(mask):
			for cidx,pt in enumerate(row):
				if path.contains_point([cidx,ridx]):
					mask[ridx,cidx] = False
					
		self.mask = mask
		self.pts = np.array(pts)
		self.center = self._center()
		
	def _center(self):
		xs = self.pts[:,0]
		ys = self.pts[:,1]
		return (np.mean(xs), np.mean(ys))