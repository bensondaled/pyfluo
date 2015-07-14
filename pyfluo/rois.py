import numpy as np
import cv2
import pylab as pl
import warnings
from PIL import Image, ImageDraw

class ROI(np.ndarray):
    """An object storing ROI information for 1 or more ROIs

    Parameters
    ----------
    mask (np.ndarray): a 2d boolean mask where True indicates interior of ROI
    pts (list / np.ndarray): a list of points defining the border of ROI
    shape (list / tuple): shape of the mask

    There are 2 ways to define an ROI:
    (1) Supply a mask
    (2) Supply both pts and shape

    In either case, the ROI object automatically resolves both the mask and points

    NOTE: slicing and indexing is still not supported due to pts issues. My presumed best solution is to find a way to put _compute_pts into array_finalize, even though it's a bit redundant sometimes
    """
    __array_priority__ = 1. #ensures that ufuncs return ROI class instead of np.ndarrays
    DTYPE = bool
    def __new__(cls, mask=None, pts=None, shape=None):
        if np.any(mask) and np.any(pts):
            warnings.warn('Both mask and points supplied. Mask will be used by default.')
        elif np.any(pts) and shape==None:
            raise Exception('Shape is required to define using points.')

        if not mask is None:
            data = mask
        elif not pts is None:
            pts = np.asarray(pts, dtype=np.int32)
            data = np.zeros(shape, dtype=np.int32)
            cv2.fillConvexPoly(data, pts, (1,1,1), lineType=cv2.CV_AA)
        else:
            raise Exception('Insufficient data supplied.')

        obj = np.asarray(data, dtype=ROI.DTYPE).view(cls)
        assert obj.ndim in [2,3]
        
        obj._compute_pts()
        
        return obj

    def _compute_pts(self):
        data = self
        if self.ndim == 2:
            data = np.array([self])

        self.pts = []
        for r in data:
            pts = np.array(cv2.findContours(r.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0])
            if len(pts) > 1:
                pts = np.concatenate(pts)
            pts = pts.squeeze()
            self.pts.append(pts)
        self.pts = np.array(self.pts).squeeze()

    def add(self, roi):
        """Add an ROI to the ROI object

        Parameters
        ----------
        roi (pyfluo.ROI / np.ndarray): the ROI to be added

        Returns
        -------
        ROI object containing old and new ROIs
        """
        roi = np.asarray(roi, dtype=ROI.DTYPE)
        if self.ndim == 2 and roi.ndim == 2:
            result = np.rollaxis(np.dstack([self,roi]), -1)
        elif self.ndim == 3:
            if roi.ndim == 2:
                to_add = roi[np.newaxis,...]
            elif roi.ndim == 3:
                to_add = roi
            else:
                raise Exception('Supplied ROI to add has improper dimensions.')
            result = np.concatenate([self,to_add])
        else:
            raise Exception('ROI could not be added. Dimension issues.')
        result = result.astype(ROI.DTYPE)
        result._compute_pts()
        return result

    def show(self, mode='pts', labels=True, cmap=pl.cm.jet, **kwargs):
        """Display the ROI(s)
        
        Parameters
        ----------
        mode ('mask' / 'pts'): specifies how to display the ROIs. If 'mask', displays as filled space. If 'pts', displays as outline of points (those originally selected)
        labels (bool): display labels over ROIs
        cmap (matplotlib.LinearSegmentedColormap): color map for display. Options are found in pl.colormaps(), and are accessed as pl.cm.my_favourite_map
        kwargs: any arguments accepted by matplotlib.imshow

        Returns
        -------
        If mode=='mask', the combined mask of all ROIs used for display
        """
       
        cmap = cmap
        colors = cmap(np.linspace(0, 1, len(self)))
        ax = pl.gca()
        xlim,ylim = pl.xlim(),pl.ylim()

        if mode == 'mask':
            mask = self.as3d().copy().view(np.ndarray).astype(np.float32)
            mask *= np.arange(1,len(mask)+1)[...,np.newaxis,np.newaxis]
            mask = np.sum(mask, axis=0)
            mask[mask==0] = None #so background doesn't steal one color from colormap, offsetting correspondence to all other uses of cmap

            pl.imshow(mask, interpolation='nearest', cmap=cmap, **kwargs)
        
        if mode == 'pts' or labels:
            if self.ndim == 2:
                pts = [self.pts]
            else:
                pts = self.pts
            for idx,roi in enumerate(pts):
                if len(roi)==0:
                    continue
                if mode == 'pts':
                    pl.scatter(roi[:,0], roi[:,1], color=colors[idx], marker='|', linewidth=3, **kwargs)
                if labels:
                    center = np.mean(roi, axis=0)
                    if mode=='pts':
                        col = colors[idx]
                    elif mode=='mask':
                        col = 'gray'
                    pl.text(center[0], center[1], str(idx), color=col, weight='bold')
            if mode == 'pts':
                ax.set_xlim( [min([xlim[0], 0]), max([xlim[1], self.shape[-1]])] )
                ax.set_ylim( [min([ylim[0], pl.ylim()[0]]), max([ylim[1], pl.ylim()[1]])] )

        if mode == 'mask':
            return mask

    def as3d(self):
        """Return 3d version of object

        Useful because the object can in principle be 2d or 3d
        """
        if self.ndim == 2:
            return np.rollaxis(np.atleast_3d(self),-1)
        else:
            return self
   
    ### Special methods
    def __array_finalize__(self, obj):
        if obj is None:
            return
        
        _custom_attrs = ['pts']
        for ca in _custom_attrs:
            setattr(self, ca, getattr(obj, ca, None))

        #_compute_pts will ideally go here, but I need to sort out conditions, b/c sometimes __array_finalize__ is called when _compute_pts is impossible
        
    def __array_wrap__(self, out, context=None):
        out._compute_pts()
        return np.ndarray.__array_wrap__(self, out, context)

    def __getslice__(self,start,stop):
        #This is a bug fix, solution found here: http://stackoverflow.com/questions/14553485/numpy-getitem-delayed-evaluation-and-a-1-not-the-same-as-aslice-1-none
        return self.__getitem__(slice(start,stop))