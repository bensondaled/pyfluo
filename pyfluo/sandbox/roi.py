# External imports
import numpy as np, pylab as pl, matplotlib.lines as mlines, pandas as pd
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

class ROI(pd.DataFrame):
    """ROI object
    """

    _metadata = ['pts', 'frame_shape']
    _metadata_defaults = [None, None]

    def __init__(self, mask, *args, **kwargs):

        # Set custom fields
        for md,dmd in zip(self._metadata, self._metadata_defaults):
            setattr(self, md, kwargs.pop(md, dmd))

        # Assess data type
        if isinstance(mask, np.ndarray) and mask.ndim==2:
            mask = np.array([mask])
        elif isinstance(mask, ROI):
            return mask
        else:
            pass
            #raise Exception('Mask data type not recognized.')
       
        # Set/adjust frame shape
        if self.frame_shape is None:
            self.frame_shape = np.asarray(mask.shape[1:])
        else:
            self.frame_shape = np.asarray(self.frame_shape)

        # If data ends up in 3d array form, flatten
        if isinstance(mask, np.ndarray) and mask.ndim==3:
            mask = mask.reshape([mask.shape[0], -1])

        # Init object
        super(ROI, self).__init__(mask, *args, **kwargs)

    @staticmethod
    def pts_to_mask(pts, shape):
        pts = np.asarray(pts, dtype=np.int32)
        mask = np.zeros(shape, dtype=np.int32)
        if CV_VERSION == 2:
            lt = cv2.CV_AA
        elif CV_VERSION == 3:
            lt = cv2.LINE_AA
        cv2.fillConvexPoly(mask, pts, (1,1,1), lineType=lt)
        return mask

    @staticmethod
    def mask_to_pts(mask):
        if CV_VERSION == 2:
            findContoursResultIdx = 0
        elif CV_VERSION == 3:
            findContoursResultIdx = 1

        if isinstance(mask, ROI):
            mask = mask.as_3d()

        pts_all = []
        for r in mask:
            pts = np.array(cv2.findContours(r.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[findContoursResultIdx])
            if len(pts) > 1:
                pts = np.concatenate(pts)
            pts = pts.squeeze()
            pts_all.append(pts)
        return np.asarray(pts_all).squeeze()

    @classmethod
    def from_pts(cls, pts, shape):
        mask = ROI.pts_to_mask(pts, shape)
        return cls(mask)

    @property
    def _constructor(self):
        return ROI
    
    @property
    def _constructor_sliced(self):
        return pd.Series

    def as_3d(self):
        return np.asarray(self).reshape([len(self)]+list(self.frame_shape))

    def show(self, mode='mask', labels=True, colors=None, cmap=pl.cm.jet, **kwargs):
        """Display the ROI(s)
        
        Parameters
        ----------
        mode : 'mask', 'pts'
            specifies how to display the ROIs. If 'mask', displays as filled space. If 'pts', displays as outline of points (those originally selected)
        labels : bool
            display labels over ROIs
        colors : list-like
            a list of values indicating the relative colors (example: magnitude) of each roi
            needs to be fixed for the possibility of this containing values of 0
            currently buggy for 'pts' mode
        cmap : matplotlib.LinearSegmentedColormap
            color map for display. Options are found in pl.colormaps(), and are accessed as pl.cm.my_favourite_map
        kwargs : dict
            any arguments accepted by matplotlib.imshow

        Returns
        -------
        If mode=='mask', the combined mask of all ROIs used for display
        """

        cmap = cmap
        fig = pl.gcf()
        ax = pl.gca()
        xlim,ylim = pl.xlim(),pl.ylim()

        mask = self.as_3d().copy().view(np.ndarray).astype(np.float32)
        if colors is None:
            colors = np.arange(1,len(mask)+1)
            colors_ = np.linspace(0, 1, len(self))
        else:
            colors_ = colors
        colors_ = cmap(colors_)

        if mode == 'mask':
            mask *= colors[...,np.newaxis,np.newaxis]
            mask = np.max(mask, axis=0)
            mask[mask==0] = None #so background doesn't steal one color from colormap, offsetting correspondence to all other uses of cmap

            alpha = kwargs.pop('alpha', 0.6)
            ims=ax.imshow(mask, interpolation='nearest', cmap=cmap, alpha=alpha, **kwargs)
            cbar = fig.colorbar(ims)
            cbar.solids.set_edgecolor("face")
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



if __name__ == '__main__':
    roi = ROI(np.random.random([10,200,200]).astype(bool))
