# External imports
import warnings
import numpy as np, matplotlib.pyplot as pl, matplotlib.lines as mlines
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Button
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
import collections
# Internal imports
from .config import *
from .util import cell_magic_wand

def select_roi(img=None, n=0, ax=None, existing=None, mode='polygon', show_kw={}, cmap=pl.cm.Greys_r):
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
    show_kw : dict
        kwargs to ROI.show
    cmap : matplotlib.LinearSegmentedColormap
        color map with which to display img
        
    Returns
    -------
    ROI object

    Notes
    -----
    Select points by clicking, and hit enter to finalize and ROI. Hit enter again to complete selection process.
    """
    if ax is None and img is None:
        raise Exception('Image or axes must be supplied to select ROI.')

    figmade = False
    if ax == None:
        figmade = True
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
            existing.show(**show_kw)
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
    if figmade:
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

    def remove(self, idx):
        """Remove ROI from ROI object

        Parameters
        ----------
        idx : int
            idx to remove
        """
        roi = self.as3d()
        result = np.delete(roi, idx, axis=0)
        result = ROI(result)
        return result

    def show(self, ax=None, **patch_kw):
        patch_kw['alpha'] = patch_kw.get('alpha', 0.5)
        patch_kw['edgecolors'] = patch_kw.get('edgecolors', 'none')
        patch_kw['cmap'] = patch_kw.get('cmap', pl.cm.viridis)

        roi = self.as3d()

        if ax is None:
            ax = pl.gca()
            nans = np.zeros(roi.shape[1:], dtype=float)
            nans[:] = np.nan
            ax.imshow(nans)

        coll = PolyCollection(verts=[r.pts for r in roi], array=np.arange(len(roi)), **patch_kw)
        ax.add_collection(coll)

        ax.set_xlim(0, roi.shape[2])
        ax.set_ylim(0, roi.shape[1])

        return ax

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

OBJ,CB,LAB = 0,1,2
class ROIView():
    def __init__(self, img, roi=None):

        # button convention: name: [obj, callback, label]
        self.buts = collections.OrderedDict([   
                    ('select', [None,self.evt_select,'Select']),
                    ('remove', [None,self.evt_remove,'Remove']),
                    ('hideshow', [None,self.evt_hideshow,'Hide']),
                    ('method', [None,self.evt_method,'Wand']),
                        ]) 
        
        # fig & axes
        self.fig = pl.figure()
        self.gs = GridSpec(len(self.buts), 2, left=0, bottom=0, top=1, right=1, width_ratios=[1,10])
        self.ax_fov = self.fig.add_subplot(self.gs[:,1])
        self._im = self.ax_fov.imshow(img, cmap=pl.cm.Greys_r)
        self.ax_fov.set_autoscale_on(False)

        # buttons
        for bi,(name,(obj,cb,lab)) in enumerate(self.buts.items()):
            ax = self.fig.add_subplot(self.gs[bi,0])
            but = Button(ax, lab)
            but.on_clicked(cb)
            self.buts[name][OBJ] = but

        # callbacks
        self.fig.canvas.mpl_connect('button_press_event', self.evt_click)
        self.fig.canvas.mpl_connect('key_press_event', self.evt_key)
        self.fig.canvas.mpl_connect('pick_event', self.evt_pick)

        # runtime
        self._mode = '' # select, remove
        self._method = 'manual' # manual, wand
        self._hiding = False
        self._selection = []
        self._selection_patches = []
        self.roi = roi
        self._roi_patches = []

    def reset_mode(self):
        if self._mode == 'select':
            self.evt_select()
        elif self._mode == 'remove':
            self.evt_remove()

    def evt_select(self, *args):
        but,_,lab = self.buts['select']
        if self._mode == 'select':
            self._mode = ''
            but.label.set_text(lab)
            but.label.set_color('k')
        elif self._mode != 'select':
            self.reset_mode()
            self._mode = 'select'
            but.label.set_text('STOP')
            but.label.set_color('red')
        self.fig.canvas.draw()
    
    def evt_remove(self, *args):
        but,_,lab = self.buts['remove']
        if self._mode == 'remove':
            self._mode = ''
            but.label.set_text(lab)
            but.label.set_color('k')
        elif self._mode != 'remove':
            self.reset_mode()
            self._mode = 'remove'
            but.label.set_text('STOP')
            but.label.set_color('red')
        self.fig.canvas.draw()

    def evt_hideshow(self, *args):
        but,_,lab = self.buts['hideshow']
        if self._hiding:
            but.label.set_text('Hide')
            self._hiding = False
            for p in self._roi_patches:
                p.set_visible(True)
        elif self._hiding == False:
            but.label.set_text('Show')
            self._hiding = True
            for p in self._roi_patches:
                p.set_visible(False)
        self.fig.canvas.draw()
    
    def evt_method(self, *args):
        self._clear_selection()
        but,_,lab = self.buts['method']
        if self._method == 'wand':
            but.label.set_text('Wand')
            self._method = 'manual'
        elif self._method == 'manual':
            but.label.set_text('Manual')
            self._method = 'wand'
        self.fig.canvas.draw()

    def evt_key(self, evt):
        if self._mode != 'select':
            return

        if evt.key == 'enter':
            self.add_roi(pts=self._selection)
            self._clear_selection()
        elif evt.key == 'backspace':
            if len(self._selection) > 0:
                self._selection = self._selection[:-1]
                self._selection_patches[-1].remove()
                self._selection_patches = self._selection_patches[:-1]
                self.fig.canvas.draw()
    
    def _clear_selection(self):
        self._selection = []
        for p in self._selection_patches:
            p.remove()
        self._selection_patches = []
        self.fig.canvas.draw()
    
    def evt_click(self, evt):
        if self._mode != 'select':
            return
        if evt.inaxes != self.ax_fov:
            return

        pt = [evt.xdata, evt.ydata]

        if self._method == 'manual':
            self._selection_patches.append(self.ax_fov.plot(pt[0], pt[1], marker='x', color='orange')[0])
            self._selection.append(pt)

        elif self._method == 'wand':
            mask = cell_magic_wand(self._im.get_array(), pt[::-1], 10, 60)
            self.add_roi(mask=mask)

        self.fig.canvas.draw()

    def evt_pick(self, evt):
        if self._mode != 'remove' or self.roi is None or len(self.roi)==0:
            return
        obj = evt.artist
        idx = self._roi_patches.index(obj)
        self._roi_patches[idx].remove()
        self.roi = self.roi.remove(idx)
        del self._roi_patches[idx]
        self.fig.canvas.draw()

    def set_img(self, img):
        if self._im is None:
            self._im = self.ax_fov.imshow(img, cmap=pl.cm.Greys_r)
        else:
            self._im.set_data(img)

    def add_roi(self, pts=None, mask=None):
        if pts is None and mask is None:
            return

        if pts is None and not np.any(mask):
            return

        if mask is None and len(pts)==0:
            return

        roi = ROI(pts=pts, mask=mask, shape=self._im.get_array().shape)
        if self.roi is None:
            self.roi = roi
        else:
            self.roi = self.roi.add(roi)

        # show
        poly = Polygon(roi.pts, alpha=0.5, picker=5)
        self.ax_fov.add_patch(poly)
        self._roi_patches.append(poly)
        self.fig.canvas.draw()

