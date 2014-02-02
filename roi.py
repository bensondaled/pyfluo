import numpy as np
import pylab as pl
from matplotlib import path as mpl_path
import matplotlib.cm as mpl_cm
import pickle
import time as pytime

class ROISet(object):
	"""An object that holds multiple *ROI* (region of interest) objects as a set.
	
	Attributes:
		shape (tuple): pixel dimensions (y,x) of the ROI objects in this set.
		
		rois (list): list of ROI objects.
		
		name (str): a unique name generated for the object when instantiated
		
	"""
	def __init__(self, rois=None):
		"""Initialize a ROISet object.
		
		Args:
			rois (list): list of ROI objects.
		"""
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
		"""Add a ROI to the ROISet.
		
		Args:
			roi (pyfluo.ROI): ROI to be added.
		"""
		if not self.shape:
			self.rois.append(roi)
			self.shape = roi.shape
		elif self.shape and roi.shape == self.shape:
			self.rois.append(roi)
		elif roi.shape != self.shape:
			print "ROI not added: shape is inconsistent with ROISet."
	def remove(self, idxs=None):
		"""Remove one or multiple ROIs from the ROISet.
		
		Args:
			idxs (int / list): ROI index, or list thereof, to be removed. If None, clears all ROIs.
		"""
		if idxs==None:
			idxs=range(len(self))
		for roi in idxs:
			self.rois[roi] = None
		self.rois = [r for r in self.rois if r]
		
	def show(self, mode='mask', labels=True):
		"""Display all the ROIs of the ROISet.
		
		Args:
			mode ('mask' / 'pts'): specifies how to display the ROIs. If 'mask', displays as filled space. If 'pts', displays as outline of points (those originally selected).
			labels (bool): display labels over ROIs.
		"""
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
	"""An object that represents a region of interest (ROI) within a 2D matrix.
	
	Attributes:
		shape (tuple): pixel dimensions (y,x) of the ROI.
		
		mask (np.ndarray): 2D boolean matrix where True signifies a pixel within the ROI.
		
		pts (np.ndarray): a list storing pairs of values, each corresponding to a point (y,x) selected by the user defining this ROI.
		
		center (2-item tuple): the pixel coordinates (y,x) of the geometrical center of the ROI.
		
		name (str): a unique name generated for the object when instantiated
		
	"""
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