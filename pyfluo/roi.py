import numpy as np
from pyfluo.pf_base import pfBase
import pylab as pl
from matplotlib import path as mpl_path
import matplotlib.cm as mpl_cm
import pickle
import time as pytime

class ROISet(pfBase):
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
        super(ROISet, self).__init__()
        self.shape = None
        
        self.rois = []
        if type(rois) == ROI:
            self.rois.append(rois)
            self.shape = rois.shape
            self.display_shape = rois.display_shape
        elif type(rois) == list:
            self.rois = rois
            self.shape = rois[0].shape
            self.display_shape = rois[0].display_shape
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
            self.display_shape = roi.display_shape
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
        
        colors = mpl_cm.jet(np.linspace(0, 1, len(self)))
        if len(self):
            if mode == 'mask':
                data = np.sum(np.dstack([r.display_mask for r in self]),axis=2)
                data = np.ma.masked_where(data == 1, data)
                ax = pl.gca()
                ax.imshow(data)
            elif mode == 'pts':
                for idx,roi in enumerate(self):
                    roi = roi.display_pts
                    pl.scatter(roi[:,0], roi[:,1], color=colors[idx])
                pl.xlim([0, self.display_shape[1]])
                pl.ylim([self.display_shape[0], 0])
            # pl.gca().xaxis.set_ticks_position('none')
            # pl.gca().yaxis.set_ticks_position('none')
            
            if labels:
                for idx,roi in enumerate(self):
                    pl.text(roi.display_center[0], roi.display_center[1], str(idx), color='white', weight='bold')
            pl.show()
    
    def __getitem__(self, idx):
        return self.rois[idx]
    def __len__(self):
        return len(self.rois)

class ROI(pfBase):
    """An object that represents a region of interest (ROI) within a 2D matrix.
    
    Attributes:
        shape (tuple): pixel dimensions (y,x) of the ROI.
        
        mask (np.ndarray): 2D boolean matrix where True signifies a pixel within the ROI.
        
        pts (np.ndarray): a list storing pairs of values, each corresponding to a point (y,x) selected by the user defining this ROI.
        
        center (2-item tuple): the pixel coordinates (y,x) of the geometrical center of the ROI.
        
        name (str): a unique name generated for the object when instantiated
        
    """
    def __init__(self, shape, pts, display_shape=None):
        super(ROI, self).__init__()
        self.shape = shape
        self.display_shape = display_shape
        if display_shape == None:
            self.display_shape = self.shape
        mask = np.ones(self.shape,dtype=bool)
        display_mask = np.ones(self.display_shape, dtype=bool)
        self.pts = np.array(pts)
        self.center = self._center()
        self.display_center = self.center

        if len(self.shape)==1 and len(pts)==2:
           mask[pts[0]:pts[1]+1] = False
           display_mask = np.tile(mask, self.display_shape[0]).reshape(self.display_shape)
           display_pts = np.array([np.repeat(pts,self.display_shape[0]), np.tile(np.arange(self.display_shape[0]),2)]).T
           self.display_center = [self.display_center, self.display_shape[0]/2.]
        else:
            path = mpl_path.Path(pts)
            for ridx,row in enumerate(mask):
                for cidx,pt in enumerate(row):
                    if path.contains_point([cidx,ridx]):
                        mask[ridx,cidx] = False
            display_mask = mask
            display_pts = np.array(pts)
        
        self.mask = mask
        self.display_mask = display_mask
        self.display_pts = display_pts

    def _center(self):
        return np.mean(self.pts, axis=0)
