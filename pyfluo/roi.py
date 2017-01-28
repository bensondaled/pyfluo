# External imports
import warnings
import numpy as np, matplotlib.pyplot as pl, matplotlib.lines as mlines
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
import collections, os, time
from scipy.spatial.distance import euclidean as dist
# Internal imports
from .config import *
from .util import cell_magic_wand, cell_magic_wand_single_point


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

    def show(self, ax=None, labels=False, label_kw=dict(color='gray'), **patch_kw):
        patch_kw['alpha'] = patch_kw.get('alpha', 0.5)
        patch_kw['edgecolors'] = patch_kw.get('edgecolors', 'none')
        patch_kw['cmap'] = patch_kw.get('cmap', pl.cm.viridis)

        roi = self.as3d()

        made_ax = False
        if ax is None:
            made_ax = True
            ax = pl.gca()

        # show patches
        coll = PolyCollection(verts=[r.pts for r in roi], array=np.arange(len(roi)), **patch_kw)
        ax.add_collection(coll)

        # show labels
        for i,r in enumerate(roi):
            ax.annotate(str(i), np.mean(r.pts, axis=0), **label_kw)

        if made_ax:
            yshape,xshape = roi.shape[1:]
            pl.axis([0,xshape,0,yshape])

        return ax

    def as3d(self):
        """Return 3d version of object

        Useful because the object can in principle be 2d or 3d
        """
        if self.ndim == 2:
            res = np.rollaxis(np.atleast_3d(self),-1)
            res.pts = np.array([res.pts])
            return res
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
    """This class serves as an interface for selecting, inspecting, and modifying ROI objects.

    The interface is entirely matplotlib-based and is non-blocking.

    The ROIView object stores the selected ROI as roiview.roi, and it can be accessed at any time.

    Ideally, use the end() method when done using, so that object cleans up its temporary files and matplotlib objects.
    """
    def __init__(self, img=None, roi=None, iterator=None, roi_show_kw={}):
        """Initialize an ROIView object

        Parameters
        ----------
        img : np.ndarray
            2d array to show as image under ROI
        roi : pyfluo.ROI
            existing ROI to start with
        iterator : iter
            any iterator (can handle next() calls), to be used to supply the underlying image
        """

        # button convention: name: [obj, callback, label]
        self.buts = collections.OrderedDict([   
                    ('select', [None,self.evt_select,'Select (T)']),
                    ('remove', [None,self.evt_remove,'Remove (X)']),
                    ('hideshow', [None,self.evt_hideshow,'Hide (V)']),
                    ('next', [None,self.evt_next,'Next (N)']),
                    ('save', [None,self.cache,'Save (cmd-S)']),
                    ('method', [None,self.evt_method,'Wand (M)']),
                        ]) 
        # sliders convention: name : [obj, min, max, init]
        self.sliders = collections.OrderedDict([   
                    ('min_radius', [None,1,20,8]),
                    ('max_radius', [None,10,100,20]),
                    ('roughness', [None,1,10,2]),
                    ('center_range', [None,1,5,2]),
                        ]) 
        self.wand_params = {name:v for name,(_,_,_,v) in self.sliders.items()}

        if img is None and iterator is not None:
            img = next(iterator)
        
        self._cachename = '_roicache_' + str(time.time()) + '.npy'
        self.roi_show_kw = roi_show_kw
        
        # fig & axes
        self.fig = pl.figure()
        hr = [3]*len(self.buts) + [1]*len(self.sliders)
        self.gs = GridSpec(len(self.buts)+len(self.sliders), 2, left=0, bottom=0, top=1, right=1, width_ratios=[1,10], height_ratios=hr, wspace=0.01, hspace=0.01)
        self.ax_fov = self.fig.add_subplot(self.gs[:,1])
        self._im = self.ax_fov.imshow(img, cmap=pl.cm.Greys_r)
        self.ax_fov.set_autoscale_on(False)

        # buttons
        for bi,(name,(obj,cb,lab)) in enumerate(self.buts.items()):
            ax = self.fig.add_subplot(self.gs[bi,0])
            but = Button(ax, lab)
            but.on_clicked(cb)
            self.buts[name][OBJ] = but
        # sliders
        for si,(name,(obj,minn,maxx,init)) in enumerate(self.sliders.items()):
            ax = self.fig.add_subplot(self.gs[si+len(self.buts),0])
            sli = Slider(ax, name, minn, maxx, init, facecolor='gray', edgecolor='none', alpha=0.5, valfmt='%0.0f')
            sli.label.set_position((0.5,0.5))
            sli.label.set_horizontalalignment('center')
            sli.vline.set_color('k')
            sli.on_changed(self.evt_slide)
            self.sliders[name][OBJ] = sli # hold onto reference to avoid garbage collection

        # callbacks
        self.fig.canvas.mpl_connect('button_press_event', self.evt_click)
        self.fig.canvas.mpl_connect('key_press_event', self.evt_key)
        self.fig.canvas.mpl_connect('pick_event', self.evt_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.evt_motion)
        self.iterator = iterator
        if self.iterator is None:
            self.buts['next'][OBJ].set_active(False)

        # runtime
        self._mode = '' # select, remove
        self._method = 'manual' # manual, wand
        self._hiding = False
        self._selection = []
        self._selection_patches = []
        self.roi = None
        self._roi_patches = []
        self._roi_centers = []
        self.add_roi(mask=roi)
        self.iteri = 0

    def reset_mode(self):
        if self._mode == 'select':
            self.evt_select()
        elif self._mode == 'remove':
            self.evt_remove()
        self.update_patches()

    def evt_slide(self, *args):
        for key,(obj,_,_,_) in self.sliders.items():
            self.wand_params[key] = int(np.round(obj.val))

    def evt_motion(self, evt):
        if self._mode != 'remove':
            return

        if evt.inaxes != self.ax_fov:
            self.update_patches()
            return

        x,y = evt.xdata,evt.ydata
        if self.roi is None or len(self.roi) == 0:
            return
        best = np.argmin([dist((x,y), c) for c in self._roi_centers])

        self.update_patches(draw=False)
        self._roi_patches[best].set_color('red')
        self._roi_patches[best].set_alpha(1.)
        self.fig.canvas.draw()

    def evt_select(self, *args):
        but,_,lab = self.buts['select']
        if self._mode == 'select':
            self._mode = ''
            but.label.set_text(lab)
            but.label.set_color('k')
        elif self._mode != 'select':
            self.reset_mode()
            self._mode = 'select'
            but.label.set_text('STOP (T)')
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
            but.label.set_text('STOP (X)')
            but.label.set_color('red')
        self.fig.canvas.draw()

    def evt_hideshow(self, *args):
        but,_,lab = self.buts['hideshow']
        if self._hiding:
            but.label.set_text('Hide (V)')
            self._hiding = False
            for p in self._roi_patches:
                p.set_visible(True)
        elif self._hiding == False:
            but.label.set_text('Show (V)')
            self._hiding = True
            for p in self._roi_patches:
                p.set_visible(False)
        self.fig.canvas.draw()
    
    def evt_method(self, *args):
        self._clear_selection()
        but,_,lab = self.buts['method']
        if self._method == 'wand':
            but.label.set_text('Wand (M)')
            self._method = 'manual'
        elif self._method == 'manual':
            but.label.set_text('Manual (M)')
            self._method = 'wand'
        self.fig.canvas.draw()

    def evt_next(self, *args):
        if self.iterator is None:
            return
        self._clear_selection()
        try:
            n = next(self.iterator)
            self.set_img(n)
            self.iteri += 1
        except StopIteration:
            self.set_img(np.zeros_like(self._im.get_array()))
            self.buts['next'][OBJ].set_active(False)

        self.cache()

    def cache(self, *args):
        np.save(self._cachename, self.roi)

    def evt_key(self, evt):

        if evt.key == 'z':
            self.remove_roi(-1)

        elif evt.key == 'escape':
            self.reset_mode()

        elif evt.key == 't':
            self.evt_select()
        elif evt.key == 'x':
            self.evt_remove()
        elif evt.key == 'v':
            self.evt_hideshow()
        elif evt.key == 'n':
            self.evt_next()
        elif evt.key == 'super+s':
            self.cache()
        elif evt.key == 'm':
            self.evt_method()

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

        pt = [round(evt.xdata), round(evt.ydata)]

        if self._method == 'manual':
            self._selection_patches.append(self.ax_fov.plot(pt[0], pt[1], marker='x', markersize=1, color='orange')[0])
            self._selection.append(pt)

        elif self._method == 'wand':
            mask = cell_magic_wand(self._im.get_array(), pt[::-1], **self.wand_params)
            if np.any(mask):
                self.add_roi(mask=mask)

        self.fig.canvas.draw()

    def evt_pick(self, evt):
        if self._mode != 'remove':
            return
        obj = evt.artist
        idx = self._roi_patches.index(obj)
        self.remove_roi(idx)

    def remove_roi(self, idx):
        if self.roi is None or len(self.roi)==0:
            return
        self._roi_patches[idx].remove()
        self.roi = self.roi.remove(idx)
        del self._roi_patches[idx]
        del self._roi_centers[idx]
        self.update_patches()

    def set_img(self, img):
        if self._im is None:
            self._im = self.ax_fov.imshow(img, cmap=pl.cm.Greys_r)
        else:
            self._im.set_data(img)
        self.ax_fov.set_ylabel(self.iteri)
        self.fig.canvas.draw()

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
        roi = roi.as3d()
        for r in roi:
            poly = Polygon(r.pts, alpha=0.5, picker=5)
            self.ax_fov.add_patch(poly)
            self._roi_patches.append(poly)
            self._roi_centers.append(np.mean(r.pts, axis=0))
        self.update_patches()

    def update_patches(self, draw=True):
        if self.roi is not None and len(self.roi) > 0:
            cols = pl.cm.viridis(np.linspace(0,1,len(self.roi)))
            for col,p in zip(cols,self._roi_patches):
                p.set_color(col)
                p.set_alpha(0.5)
        if draw:
            self.fig.canvas.draw()

    def end(self):
        if os.path.exists(self._cachename):
            os.remove(self._cachename)
        pl.close(self.fig)

