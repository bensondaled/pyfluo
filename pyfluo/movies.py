import numpy as np
from roi import ROI
from images.tiff import Tiff
from traces import Trace
from motion import correct_motion, apply_motion_correction
import pylab as pl
import cv2
from ts_base import TSBase

class Movie(TSBase):
    """An object storing n sequential 2d images along with a time vector

    Parameters
    ----------
    data : np.ndarray, str, pyfluo.Tiff 
        input data, see below
    time : [optional] np.ndarray, list
        time vector with n elements
    Ts : [optional] float
        sampling period
    info : [optional] list, np.ndarray
        info vector with n elements

    Input data can be supplied in multiple ways:
    (1) as a numpy array of shape (n,y,x)
    (2) a string corresopnding to a file path to a tiff
    (3) a pyfluo.Tiff object

    The data in Movie objects is stored following the standard pyfluo convention in which the 0th axis corresponds to time. For example, movie[0] corresponds to the movie frame at time 0. 

    It should be noted that error checking with regard to the time vector is still under progress. All tested operations are functional, but thorough testing has not yet been performed.
    """
    __array_priority__ = 1. #ensures that ufuncs return ROI class instead of np.ndarrays
    def __new__(cls, data, **kwargs):
        if type(data) == Tiff:
            data = data.data.copy()
        elif type(data) == str:
            data = Tiff(data).data.copy()
        return super(Movie, cls).__new__(cls, data, _ndim=[3], **kwargs)
    def project(self, axis=0, method=np.mean, show=False, rois=None, **kwargs):
        """Flatten/project the movie data across one or many axes
        
        Parameters
        ----------
        axis : int, list
            axis/axes over which to flatten
        method : def
            function to apply across the specified axes
        show : bool
            display the result (if 2d, as image; if 1d, as trace)
        rois : pyfluo.ROI 
            roi to display
            
        Returns
        -------
        The projected image
        """
        pro = np.apply_over_axes(method,self,axes=axis).squeeze()
        
        if show:
            if self.interactive_backend == cv2:
                raise Exception('cv2 backend does not yet support projection.')
            elif self.interactive_backend == pl:
                ax = pl.gca()
                ax.margins(0.)
                if pro.ndim == 2:
                    pl.imshow(pro, cmap=pl.cm.Greys_r, **kwargs)
                    if rois is not None:
                        rois.show(mode='pts',labels=True)
                elif pro.ndim == 1:
                    pl.plot(self.time, pro)
        
        return pro
    def play(self, loop=False, fps=None, scale=1, contrast=1., backend=cv2, **kwargs):
        """Play the movie
        
        Parameter
        ---------
        loop : bool 
            repeat playback upon finishing
        fps : float
            playback rate in frames per second. Defaults to object's fs attribute
        scale : float 
            scaling factor to resize playback images
        contrast : float
            scaling factor for pixel values
        backend : module 
            package to use for playback (ex. pl or cv2). If *None*, defaults to self.interactive_backend

        During playback, 'q' can be used to quit when playback window has focus
            
        """
        if fps==None:
            fps = self.fs
        fpms = fps / 1000.

        if backend == None:
            backend = self.interactive_backend
       
        if backend == pl:
            flag = pl.isinteractive()
            pl.ioff()
            fig = pl.figure()
            ims = [ [pl.imshow(np.atleast_2d(i), cmap=pl.cm.Greys_r, aspect=self.visual_aspect, vmin=np.min(self.data), vmax=np.max(self.data))] for i in self ]
            
            ani = animation.ArtistAnimation(fig, ims, interval=1./fpms, blit=False, repeat=loop, **kwargs)
            pl.show()
            if flag:    pl.ion()
        elif backend == cv2:
            size = tuple(scale*np.array(self.shape)[-1:0:-1])
            minn,maxx = self.min(),self.max()
            def _play_once():
                to_play = contrast * (self+minn)/(maxx-minn)
                to_play[to_play>1.0] = 1.0
                for fr in to_play:
                    fr = cv2.resize(fr,size)
                    cv2.imshow('Movie',fr)
                    k=cv2.waitKey(int(1./fpms))
                    if k == ord('q'):
                        return False
            if loop:
                cont = True
                while cont:
                    cont = _play_once()
            else:
                _play_once()
            cv2.destroyWindow('Movie')
    
    def select_roi(self, n=1, existing=None):
        """Select any number of regions of interest (ROI) in the movie.
        
        Parameters
        ----------
        n : int
            number of ROIs to select
            
        Returns
        -------
        ROI object
        """
        roi = None
        for q in xrange(n):
            pl.clf()
            zp = self.project(show=True, rois=existing)
            pts = pl.ginput(0, timeout=0)

            if pts:
                if roi is None:
                    roi = ROI(pts=pts, shape=zp.shape)
                    if existing is None:
                        existing = roi.copy()
                else:
                    new_roi = ROI(pts=pts, shape=zp.shape)
                    existing = existing.add(new_roi)
                    roi = roi.add(new_roi)
        pl.close()
        return roi
    def extract_by_roi(self, roi, method=np.mean):
        """Extract a time series consisting of one value for each movie frame, attained by performing an operation over the regions of interest (ROI) supplied
        
        Parameters
        ----------
        roi : pyfluo.ROI
            the rois over which to extract data
        method : def 
            the function by which to convert the data within an ROI to a single value. Defaults to np.mean
            
        Returns
        -------
        Trace object, with multiple columns corresponding to multiple ROIs
        """
        return Trace((roi.as3d().reshape((len(roi),-1)).dot(self.reshape((len(self),-1)).T)).T, time=self.time, Ts=self.Ts)

    def correct_motion(self, *params, **kwargs):
        """A convenience method for pyfluo.motion.correct_motion

        If unnamed params are supplied, they are taken to be the output of a prior motion correction and are applied to the movies using pyfluo.motion.apply_motion_correction.

        If named kwargs are supplied, they are passed along to pyfluo.motion.correct_motion.
        """
        if params:
            return apply_motion_correction(self, *params)
        else:
            return correct_motion(self, **kwargs)
