import numpy as np
import cv2
import pylab as pl
import warnings
import matplotlib.lines as mlines
from PIL import Image, ImageDraw

def select_roi(img, n=1, existing=None, mode='lasso', show_mode='pts', cmap=pl.cm.Greys_r, lasso_strictness=1):
    """Select any number of regions of interest (ROI) in the movie.
    
    Parameters
    ----------
    img : np.ndarray
        image over which to select roi
    n : int
        number of ROIs to select
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
    """
    for q in xrange(n):
        pl.clf()
        pl.imshow(img, cmap=cmap)
        if existing is not None:
            existing.show(mode=show_mode)
        if mode == 'polygon':
            pts = pl.ginput(0, timeout=0)
        elif mode == 'lasso':
            pts = lasso(lasso_strictness)

        if pts is not None:
            new_roi = ROI(pts=pts, shape=img.shape)
            if existing is None:
                existing = ROI(pts=pts, shape=img.shape)
            else:
                existing = existing.add(new_roi)
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
            cv2.fillConvexPoly(data, pts, (1,1,1), lineType=cv2.CV_AA)
        else:
            raise Exception('Insufficient data supplied.')
        obj = np.asarray(data, dtype=ROI.DTYPE).view(cls)
        assert obj.ndim in [2,3]
        
        return obj
    def __init__(self, *args, **kwargs):
        self._compute_pts()

    def _compute_pts(self):
        data = self.copy().view(np.ndarray)
        if self.ndim == 2:
            data = np.array([self])

        selfpts = []
        for r in data:
            pts = np.array(cv2.findContours(r.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0])
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

    def show(self, mode='pts', labels=True, cmap=pl.cm.jet, **kwargs):
        """Display the ROI(s)
        
        Parameters
        ----------
        mode : 'mask', 'pts'
            specifies how to display the ROIs. If 'mask', displays as filled space. If 'pts', displays as outline of points (those originally selected)
        labels : bool
            display labels over ROIs
        cmap : matplotlib.LinearSegmentedColormap
            color map for display. Options are found in pl.colormaps(), and are accessed as pl.cm.my_favourite_map
        kwargs : dict
            any arguments accepted by matplotlib.imshow

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


# Lasso
def dist(p1,p2):
    p1,p2 = np.array(p1),np.array(p2)
    d = np.sqrt(np.sum((p1-p2)**2))
    return d

class Lasso(object):
    #TODO: delete points, adaptive edges, max_dist for new pt
    def __init__(self, strictness=1):
        self.on = False
        self.pts = []
        self.marks = []

        self.fig = pl.gcf()
        self.ax = pl.gca()
        pl.autoscale(enable=False)
        self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.cid_click = self.fig.canvas.mpl_connect('button_release_event', self.onclick)
        self.cid_exit = self.fig.canvas.mpl_connect('axes_leave_event', self.onexit)
        self.cid_keyup = self.fig.canvas.mpl_connect('key_release_event', self.on_keyup)

        img = self.ax.get_images()[0].get_array().view(np.ndarray)
        img = (img-img.min())/(img.max()-img.min())
        img *= 255
        img = img.astype(np.uint8)
        edge_img = cv2.Canny(img, img.mean(), img.mean()+strictness*img.std())
        self.edge_pts = np.argwhere(edge_img)
        self.fig.canvas.start_event_loop(timeout=-1)
    def get_best(self, pt):    
        dists = np.array([dist(pt,ep) for ep in self.edge_pts]) #ventually np.vectorize
        best = self.edge_pts[np.argmin(np.abs(dists))]
        return best
    def add_pt(self, pt):
        pt = list(pt)
        if pt not in self.pts:
            line = mlines.Line2D([pt[0]], [pt[1]], marker='+', color='r')
            self.ax.add_line(line)
            self.marks.append(line)
            self.pts.append(pt)
            self.fig.canvas.draw()
    def onclick(self,event):
        if self.on:
            pt = np.array([event.ydata,event.xdata])
            self.add_pt(pt[::-1])
    def onmove(self,event):
        if self.on:
            pt = np.array([event.ydata,event.xdata])
            best = self.get_best(pt)
            self.add_pt(best[::-1])
    def onenter(self,event):
        pass
    def onexit(self,event):
        self.on = False
    def on_keyup(self,event):
        if not self.on:
            self.on = True
        else:
            self.pts = np.array(self.pts)
            self.end()
    def end(self):
        self.fig.canvas.mpl_disconnect(self.cid_exit)
        self.fig.canvas.mpl_disconnect(self.cid_move)
        self.fig.canvas.mpl_disconnect(self.cid_click)
        self.fig.canvas.mpl_disconnect(self.cid_keyup)
        for mark in self.marks:
            mark.remove()
        self.fig.canvas.stop_event_loop()
        self.fig.canvas.draw()

def lasso(strictness):
    print 'Press enter to begin and end.'
    l = Lasso(strictness)
    return l.pts

