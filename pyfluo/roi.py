# External imports
import numpy as np, pylab as pl, matplotlib.lines as mlines
import cv2, warnings
# Internal imports
from .config import *

def select_roi(img=None, n=0, ax=None, existing=None, mode='polygon', show_mode='mask', cmap=pl.cm.Greys_r, lasso_strictness=1):
    """Select any number of regions of interest (ROI) in the movie.
    
    Parameters
    ----------
    img : np.ndarray
        image over which to select roi
    n : int
        number of ROIs to select
    ax : matplotlib.Axes
        axes on which to show and select. If None, defaults to new, if 'current', defaults to current
    existing : pyfluo.ROI
        pre-existing rois to which to add selections
    mode : 'polygon', 'lasso'
        mode by which to select roi
    show_mode : 'pts', 'mask'
        mode by which to show existing rois
    cmap : matplotlib.LinearSegmentedColormap
        color map with which to display img
    lasso_strictness : float
        number from 0-inf, to do with tolerance for edge finding
        
    Returns
    -------
    ROI object

    Notes
    -----
    Select points by clicking, and hit enter to finalize and ROI. Hit enter again to complete selection process.
    """
    if ax is None and img is None:
        raise Exception('Image or axes must be supplied to select ROI.')

    if ax == None:
        fig = pl.figure()
        ax = fig.add_subplot(111)
    elif ax == 'current':
        ax = pl.gca()
    pl.sca(ax)
    fig = ax.get_figure()

    if img is not None:
        shape = img.shape
    elif ax is not None:
        shape = [abs(np.diff(ax.get_ylim())), abs(np.diff(ax.get_xlim()))]

    q = 0
    while True:
        if n>0 and q>=n:
            break
        if img is not None:
            ax.imshow(img, cmap=cmap)
        if existing is not None:
            existing.show(mode=show_mode)
        if mode == 'polygon':
            pts = fig.ginput(0, timeout=0)
        elif mode == 'lasso':
            pts = lasso(lasso_strictness)

        if pts != []:
            new_roi = ROI(pts=pts, shape=shape)
            if existing is None:
                existing = ROI(pts=pts, shape=shape)
            else:
                existing = existing.add(new_roi)
            q += 1
        elif pts == []:
            break
        ax.cla()
    pl.close()
    return existing

class ROI(np.ndarray):
    """An object storing ROI information for 1 or more ROIs

    Parameters
    ----------
    mask : np.ndarray
        a 2d boolean mask where True indicates interior of ROI
    pts : list, np.ndarray
        a list of points defining the border of ROI
    shape : list, tuple
        shape of the mask

    There are 2 ways to define an ROI:
    (1) Supply a mask
    (2) Supply both pts and shape

    In either case, the ROI object automatically resolves both the mask and points

    """
    __array_priority__ = 1. #ensures that ufuncs return ROI class instead of np.ndarrays
    DTYPE = bool
    _custom_attrs = ['pts']
    _custom_attrs_slice = ['pts']
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
            if CV_VERSION == 2:
                lt = cv2.CV_AA
            elif CV_VERSION == 3:
                lt = cv2.LINE_AA
            cv2.fillConvexPoly(data, pts, (1,1,1), lineType=lt)
        else:
            raise Exception('Insufficient data supplied.')
        obj = np.asarray(data, dtype=ROI.DTYPE).view(cls)
        assert obj.ndim in [2,3]
        
        return obj
    def __init__(self, *args, **kwargs):
        self._compute_pts()

    def _compute_pts(self):
        if CV_VERSION == 2:
            findContoursResultIdx = 0
        elif CV_VERSION == 3:
            findContoursResultIdx = 1
        data = self.copy().view(np.ndarray)
        if self.ndim == 2:
            data = np.array([self])

        selfpts = []
        for r in data:
            pts = np.array(cv2.findContours(r.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[findContoursResultIdx])
            if len(pts) > 1:
                pts = np.concatenate(pts)
            pts = pts.squeeze()
            selfpts.append(pts)
        self.pts = np.asarray(selfpts).squeeze()

    def add(self, roi):
        """Add an ROI to the ROI object

        Parameters
        ----------
        roi : pyfluo.ROI, np.ndarray
            the ROI to be added

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

    def show(self, mode='contours', labels=False, colors=None, cmap=pl.cm.viridis, contours_kwargs=dict(thickness=2), **kwargs):
        """Display the ROI(s)

        NEEDS FIXING
        TODO: adjust how masks are coloured such that the image itself is RGB and doesn't need a colormap to depend on
        
        Parameters
        ----------
        mode : 'mask', 'pts', 'contours'
            specifies how to display the ROIs. If 'mask', displays as filled space. If 'pts', displays as outline of points. If 'contour', draws contours around rois
            mask and contours can be used together, ex. 'mask,contours'
            **contours and pts currently under construction
        labels : bool
            display labels over ROIs
        colors : list-like
            a list of values indicating the relative colors (example: magnitude) of each roi
            needs to be fixed for the possibility of this containing values of 0
            currently buggy for 'pts' mode
        cmap : matplotlib.LinearSegmentedColormap
            color map for display. Options are found in pl.colormaps(), and are accessed as pl.cm.my_favourite_map
        contours_kwargs : dict
            for cv2.drawContours
        kwargs : dict
            any arguments accepted by matplotlib.imshow

        Returns
        -------
        If mode=='mask', the combined mask of all ROIs used for display
        """

        cmap = cmap
        fig = pl.gcf()
        #ax = fig.add_subplot(111)
        ax = pl.gca()
        xlim,ylim = pl.xlim(),pl.ylim()

        mask = self.as3d().copy().view(np.ndarray).astype(np.float32)
        if colors is None:
            colors = np.arange(1,len(mask)+1)
            colors_ = np.linspace(0, 1, len(self))
        else:
            colors_ = colors
        colors_ = cmap(colors_)

        if 'contours' in mode:
            base = np.zeros(mask.shape[1:]+(4,))
            for i,p in enumerate(self.pts):
                cv2.drawContours(base, [p], -1, colors_[i], **contours_kwargs)
            base[...,-1] = (base.sum(axis=-1)!=0).astype(float)
            ims=ax.imshow(base, interpolation='nearest', **kwargs)
            pl.draw()

        if 'mask' in mode:
            mask *= colors[...,np.newaxis,np.newaxis]
            mask = np.max(mask, axis=0)
            mask[mask==0] = None #so background doesn't steal one color from colormap, offsetting correspondence to all other uses of cmap

            alpha = kwargs.pop('alpha', 0.6)

            #cv2.drawContours(mask, self.pts, -1, (255,255,255), **contours_kwargs)
            ims=ax.imshow(mask, interpolation='nearest', cmap=cmap, alpha=alpha, **kwargs)
            pl.draw()

        
        if mode == 'pts' or labels:
            if self.ndim == 2:
                pts = [self.pts]
            else:
                pts = self.pts
            for idx,roi in enumerate(pts):
                if len(roi)==0:
                    continue
                if mode == 'pts':
                    pl.scatter(roi[:,0], roi[:,1], color=colors_[idx], marker='|', linewidth=3, **kwargs)
                if labels:
                    center = np.mean(roi, axis=0)
                    if mode=='pts':
                        col = colors_[idx]
                    elif mode=='mask':
                        col = 'gray'
                    pl.text(center[0], center[1], str(idx), color=col, weight='bold')
            if mode == 'pts':
                ax.set_xlim( [min([xlim[0], 0]), max([xlim[1], self.shape[-1]])] )
                ax.set_ylim( [min([ylim[0], pl.ylim()[0]]), max([ylim[1], pl.ylim()[1]])] )

        pl.gcf().canvas.draw()
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

        for ca in ROI._custom_attrs:
            setattr(self, ca, getattr(obj, ca, None))

    # Commenting this out until I encounter a scenario where it proves necessary
    #def __array_wrap__(self, out, context=None):
    #    out._compute_pts()
    #    return np.ndarray.__array_wrap__(self, out, context)

    def __getslice__(self,start,stop):
        #classic bug fix
        return self.__getitem__(slice(start,stop))

    def __getitem__(self, idx):
        out = super(ROI,self).__getitem__(idx)
        if not isinstance(out, ROI):
            return out
        if self.ndim == 2: #no mods when slicing a single roi
            pass
        elif self.ndim == 3: #multiple rois: need associated pts
            for ca in ROI._custom_attrs_slice:
                setattr(out, ca, getattr(out, ca, None)[idx])
        return out
