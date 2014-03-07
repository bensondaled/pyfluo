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
        
    def show(self, mode='pts', labels=True):
        """Display all the ROIs of the ROISet.
        
        Args:
            mode ('mask' / 'pts'): specifies how to display the ROIs. If 'mask', displays as filled space. If 'pts', displays as outline of points (those originally selected).
            labels (bool): display labels over ROIs.
        """
        
        colors = mpl_cm.jet(np.linspace(0, 1, len(self)))
        if len(self) == 0:
            return

       
        ax = pl.gca()
        xlim = pl.xlim()
        ylim = pl.ylim()
        if len(self.shape)==2:
            if mode == 'mask':
                data = np.sum(np.dstack([r.mask for r in self]),axis=2)
                data = np.ma.masked_where(data == 1, data)
                ax.imshow(data)
            elif mode == 'pts':
                for idx,roi in enumerate(self):
                    roi = roi.pts
                    pl.scatter(roi[:,0], roi[:,1], color=colors[idx])
                #pl.xlim([0, self.shape[1]])
                #pl.ylim([self.shape[0], 0])
            
            if labels:
                for idx,roi in enumerate(self):
                    pl.text(roi.center[0], roi.center[1], str(idx), color='white', weight='bold')
        elif len(self.shape)==1:
           for idx,roi in enumerate(self):
               pts = roi.pts
               pl.scatter(pts, [0. for i in pts], color=colors[idx], marker='|', linewidth=3)
               if labels:
                   pl.text(roi.center, 0., str(idx), color='white', weight='bold')
        ax.set_xlim( [min([xlim[0], 0]), max([xlim[1], self.shape[-1]])] )
        ax.set_ylim( [min([ylim[0], pl.ylim()[0]]), max([ylim[1], pl.ylim()[1]])] )
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
    def __init__(self, shape, pts):
        super(ROI, self).__init__()
        self.shape = shape
        mask = np.ones(self.shape,dtype=bool)

        if len(self.shape)==1:
            pts = np.array(pts)[:,0]
            pts.sort()
            pts = pts[[0,-1]]
            mask[pts[0]:pts[1]+1] = False
        elif len(self.shape)==2:
            path = mpl_path.Path(pts)
            for ridx,row in enumerate(mask):
                for cidx,pt in enumerate(row):
                    if path.contains_point([cidx,ridx]):
                        mask[ridx,cidx] = False
        else:
            raise Exception('ROI class does not support ROIs in >2 dimensions.')
        
        self.pts = np.array(pts)
        self.mask = mask
        self.center = self._center()

    def _center(self):
        return np.mean(self.pts, axis=0)
