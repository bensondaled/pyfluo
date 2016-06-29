import numpy as np, pylab as pl, pandas as pd
from scipy.ndimage.interpolation import zoom as szoom
from matplotlib import animation
import cv2, sys, tifffile, operator, os, threading

from .roi import ROI, select_roi
from .images import Tiff, AVI, HDF5
from .series import Series
from .config import *

class Movie(np.ndarray):
    """An object for *in-memory* storage of n sequential 2d images along with a sampling rate

    Parameters
    ----------
    data : array-like, str-like
        input data, see below
    Ts : [optional] float
        sampling period

    Input data can be supplied in multiple ways:
    (1) as a numpy array of shape (n,y,x)
    (2) a string [or list thereof] corresponding to a file path to a tiff/avi/hdf5

    The data in Movie objects is stored following the standard pyfluo convention in which the 0th axis corresponds to time. For example, movie[0] corresponds to the movie frame at time 0. 
    """

    __array_priority__ = 1. #ensures that ufuncs return ROI class instead of np.ndarrays
    _custom_attrs = ['Ts', 'filename']
    
    def __new__(cls, data, Ts=1, filename=[], **kwargs):
        assert any([isinstance(data,lt) for lt in PF_list_types+PF_str_types]), 'Movie data not supplied in proper format.'

        # if data is filenames
        if type(data) in PF_str_types:
            filename = data
            suf = os.path.splitext(filename)[-1]

            if suf == '.avi':
                data = AVI(d, **kwargs).data
            elif suf in ['.h5','.hdf5']:
                hs = HDF5(d, **kwargs)
                data = h.data
                Ts = h.Ts #overwrites any supplied Ts with hdf5 file's stored time info
            elif suf in ['.tif','.tiff']:
                tf = tifffile.TiffFile(data)
                data = tf.asarray()

        # convert to a np array (either of strings or data itself)
        data = np.asarray(data)
        obj = data.view(cls)

        obj.Ts = Ts
        obj.filename = filename

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        for ca in Movie._custom_attrs:
            setattr(self, ca, getattr(obj, ca, None))

    def project(self, axis=0, method=np.mean, show=False, roi=None, backend=pl, **kwargs):
        """Flatten/project the movie data across one or many axes
        
        Parameters
        ----------
        axis : int, list
            axis/axes over which to flatten
        method : def
            function to apply across the specified axes
        show : bool
            display the result (if 2d, as image; if 1d, as trace)
        roi : pyfluo.ROI 
            roi to display
        backend : module
            module used for interactive display (only matplotlib currently supported)
            
        Returns
        -------
        The projected image
        """
        if method == None:
            method = np.mean
        pro = np.apply_over_axes(method,self,axes=axis).squeeze()
        
        if show:
            ax = pl.gca()
            ax.margins(0.)
            if pro.ndim == 2:
                pl.imshow(pro, cmap=pl.cm.Greys_r, **kwargs)
                if roi is not None:
                    roi.show()
        
        return pro
    def play(self, loop=False, fps=None, scale=1, contrast=1., show_time=True, backend=cv2, fontsize=1, rolling_mean=1, **kwargs):
        """Play the movie
        
        Parameter
        ---------
        loop : bool 
            repeat playback upon finishing
        fps : float
            playback rate in frames per second. Defaults to 1/Ts
        scale : float 
            scaling factor to resize playback images
        contrast : float
            scaling factor for pixel values (clips so as to produce contrast)
        show_time : bool
            show time on image
        backend : module 
            package to use for playback (ex. pl or cv2)
        fontsize : float
            to display time
        rolling_mean : int
            number of frames to avg in display

        Playback controls (window must have focus):
            'f' : faster
            's' : slower
            'p' : pause / continue
            'r' : reverse
            '=' : zoom in
            '-' : zoom out
            'q' : quit

        Note: many params are not implemented in the matplotlib backend option
            
        """
        if fps==None:
            fps = 1/self.Ts
        fpms = fps / 1000.

        if backend == pl:
            flag = pl.isinteractive()
            pl.ioff()
            fig = pl.figure()
            to_play = self.resize(scale)
            minn,maxx = to_play.min(),to_play.max()
            to_play = contrast * (to_play-minn)/(maxx-minn)
            to_play[to_play>1.0] = 1.0 #clips; this should become a parameter
            im = pl.imshow(np.zeros(self[0].shape), cmap=pl.cm.Greys_r, vmin=np.min(to_play), vmax=np.max(to_play))
            self._idx = 0
            def func(*args):
                self._idx += 1
                if self._idx>=len(to_play):
                    return None
                im.set_array(to_play[self._idx])
                return im,
            
            ani = animation.FuncAnimation(fig, func, interval=1./fpms, blit=False, repeat=loop, **kwargs)
            pl.show()
            if flag:    pl.ion()

        elif backend == cv2:
            title = 'p / q / f / s / r / = / -'
            minn,maxx = self.min(),self.max()
        
            current_idx = 0
            wait = 1./fpms
            paused = False
            oper_idx = 0 #0: add, 1: subtract
            while True:
                if not paused:

                    # get frame
                    size = tuple((scale*np.array(self.shape)[-1:0:-1]).astype(int))
                    font_size = scale * fontsize * min(self[0].shape)/450.
                    t = self.Ts*current_idx
                    fr = self[current_idx:current_idx+rolling_mean].mean(axis=0)
                    fr = contrast * (fr-minn)/(maxx-minn)
                    fr[fr>1.0] = 1.0
                    fr[fr<0.0] = 0.0
                    fr = cv2.resize(fr,size)
                    if show_time:
                        cv2.putText(fr, '{:0.3f}'.format(t), (5,int(30*font_size)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (120,100,80), thickness=1)
                    cv2.imshow(title,fr)
                    
                    # update indices
                    opfunc = [operator.add, operator.sub][oper_idx]
                    current_idx = opfunc(current_idx, rolling_mean)
                    if current_idx >= len(self):
                        if not loop:
                            break
                        elif loop:
                            current_idx = 0

                # user input
                k=cv2.waitKey(int(wait))
                if k == ord('q'):
                    break
                elif k == ord('p'):
                    paused = not paused
                elif k == ord('f'):
                    wait = wait/1.5
                    wait = max([wait, 1])
                elif k == ord('s'):
                    wait = wait*1.5
                elif k == ord('r'):
                    oper_idx = int(not oper_idx)
                elif k == ord('='):
                    scale = scale * 1.5
                elif k == ord('-'):
                    scale = scale / 1.5

            cv2.destroyWindow(title)
    
    def select_roi(self, *args, **kwargs):
        """A convenience method for pyfluo.roi.select_roi(self.project(), *args, **kwargs)

        Parameters
        ----------
        projection_method : def
            'method' parameter for Movie.project, used in display
        *args, **kwargs
            those accepted by pyfluo.roi.select_roi
        """
        zp = self.project(show=False, method=kwargs.pop('projection_method',np.std))
        return select_roi(zp, *args, **kwargs)
    def extract(self, roi):
        """Extract a time series consisting of one value for each movie frame, attained by performing an operation over the regions of interest (ROI) supplied
        
        Parameters
        ----------
        roi : pyfluo.ROI
            the rois over which to extract data
        
        Returns
        -------
        Trace object, with multiple columns corresponding to multiple ROIs
        """
        if roi.ndim == 3:
            flatshape = (len(roi),-1)
            roi_norm = roi / roi.sum(axis=(1,2))[:,None,None]
        elif roi.ndim == 2:
            flatshape = -1
            roi_norm = roi / roi.sum()
        roi_norm = roi_norm.reshape(flatshape)
        self_flat = self.reshape((len(self),-1)).T
        dp = (roi_norm.dot(self_flat)).T
        trace = dp
        
        return Series(trace, index=self.Ts*np.arange(len(self)), Ts=self.Ts)

    def resize(self, factor, order=0):
        """Resize movie using scipy.ndimage.zoom
        
        Parameters
        ----------
        factor : float, tuple, list
            multiplying factor for dimensions (y,x), ex. 0.5 will downsample by 2. If number, it is used for all dimensions
        order : int
            order of interpolation (0=nearest, 1=bilinear, 2=cubic)
        """              
        if type(factor) in [int,float,np.float16,np.float32,np.float64,np.float128,np.int8,np.int16,np.int32,np.int64]:
            factor = [factor,factor]
        elif any([isinstance(factor,t) for t in [list,tuple,np.ndarray]]):
            factor = list(factor)
        else:
            raise Exception('factor parameter not understood')
        res = szoom(self, [1]+factor, order=order)
        res = self.__class__(res, Ts=self.Ts)
        for ca in self._custom_attrs:
            res.__setattr__(ca, self.__getattribute__(ca))
        return res

    def save(self, filename, fmt=None, codec='IYUV', fps=None):

        """Save movie for playback in video player

        Note that this function is intended for saving the movie in an avi/tiff-like format. Saving for further analysis should be performed using pyfluo's io functions.

        Parameters
        ----------
        filename : str
            destination file name, without extension
        fmt : str
            format to save movie, specified as file extension, ex. 'avi', 'mp4', 'tif'
        codec : int
            opencv fourcc code, as chars to be fed to cv2.FOURCC ex. 'IYUV'
        fps : int
            frames per second. Defaults to obj.fs
            applies to only to formats tht store time info, like avi
        """
        if fps == None:
            fps = 1/self.Ts
        filename += '.'+fmt

        if fmt in ['tif','tiff']:
            tifffile.imsave(filename, data=np.asarray(self))
        elif fmt in ['mp4', 'avi']:
            if CV_VERSION == 3:
                codec = cv2.VideoWriter_fourcc(*codec)
            elif CV_VERSION == 2:
                codec = cv2.cv.FOURCC(*codec)
            minn,maxx = np.min(self),np.max(self)
            data = 255 * (self-minn)/(maxx-minn)
            data = data.astype(np.uint8)
            y,x = data[0].shape
            vw = cv2.VideoWriter(filename, codec, fps, (x,y), isColor=True)
            for d in data:
                vw.write(cv2.cvtColor(d, cv2.COLOR_GRAY2BGR))
            vw.release()

