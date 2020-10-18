import numpy as np, pylab as pl, pandas as pd
from PIL import Image, ImageTk
from scipy.ndimage.interpolation import zoom as szoom
from matplotlib import animation
import sys, operator, os, threading, time, tifffile

from .roi import ROI, ROIView
from .images import Tiff, AVI, HDF5
from .series import Series
from .config import *

if PF_pyver == 3:
    import tkinter as tk
elif PF_pyver == 2:
    import Tkinter as tk

try:
    import pyqtgraph as pg
except ImportError:
    pg = None

class Movie(np.ndarray):
    """An object for *in-memory* storage of n sequential 2d images along with a sampling rate

    Parameters
    ----------
    data : array-like, str-like
        input data, see below
    Ts : [optional] float
        sampling period
    filename : str
        path to filename for data

    Input data can be supplied in multiple ways:
    (1) as a numpy array of shape (n,y,x)
    (2) a string [or list thereof] corresponding to a file path to a tiff/avi/hdf5

    The data in Movie objects is stored following the standard pyfluo convention in which the 0th axis corresponds to time. For example, movie[0] corresponds to the movie frame at time 0. 
    """

    __array_priority__ = 1. #ensures that ufuncs return ROI class instead of np.ndarrays
    _custom_attrs = ['Ts', 'filename']
    
    def __new__(cls, data, Ts=1, filename=[], **kwargs):
        assert isinstance(data, PF_list_types+PF_str_types), 'Movie data not supplied in proper format.'
        
        tf = None
        # if data is filenames
        if type(data) in PF_str_types:
            filename = data
            suf = os.path.splitext(filename)[-1]

            if suf == '.avi':
                data = AVI(data, **kwargs).data
            elif suf in ['.h5','.hdf5']:
                hs = HDF5(data, **kwargs)
                data = h.data
                Ts = h.Ts #overwrites any supplied Ts with hdf5 file's stored time info
            elif suf in ['.tif','.tiff']:
                tf = Tiff(data)
                data = tf.data
                Ts = tf.Ts

        # convert to a np array (either of strings or data itself)
        data = np.asarray(data)
        obj = data.view(cls)

        obj.Ts = Ts
        obj.filename = filename
        obj.tif_obj = tf

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
    def play(self, method='pyqtgraph', **kwargs):
        """Play the movie

        Parameters
        ----------
        method : str
            'pyqtgraph', 'tkinter', 'opencv'

        """
        if method == 'opencv':
            return play_mov(self)
        elif method == 'tkinter' or (method == 'pyqtgraph' and pg is None):
            print('Try installing PyQtGraph for another good interactive media player.')
            return MoviePlayer(self, Ts=self.Ts, **kwargs)
        elif method == 'pyqtgraph':
            im = pg.image(self)
            im.fps = int(np.round(1/self.Ts))
            im.play()
            pg.QtGui.QApplication.exec_()
            return im
    
    def select_roi(self, *args, **kwargs):
        """Select ROI using ROIView

        Parameters
        ----------
        projection_method : def
            'method' parameter for Movie.project, used in display
        *args, **kwargs
            those accepted by pyfluo.roi.ROIView
        """
        zp = self.project(show=False, method=kwargs.pop('projection_method',np.mean))
        rv = ROIView(zp)
        return rv
    def extract(self, roi):
        """Extract a time series consisting of one value for each movie frame, attained by performing an operation over the regions of interest (ROI) supplied
        
        Parameters
        ----------
        roi : pyfluo.ROI
            the rois over which to extract data
        
        Returns
        -------
        pyfluo.Series object, with multiple columns corresponding to multiple ROIs


        Note: my gut says np.einsum should be able to further optimize this, yet all benchmarks I've run so far say this is the fastest
        """
        if roi.ndim == 3:
            flatshape = (len(roi),-1)
            roi_norm = roi.astype(float) / roi.sum(axis=(1,2))[:,None,None]
        elif roi.ndim == 2:
            flatshape = -1
            roi_norm = roi.astype(float) / roi.sum()
        roi_norm = roi_norm.reshape(flatshape)
        if self.ndim == 2:
            self_flat = np.array([self])
        else:
            self_flat = self
        self_flat = self_flat.reshape((len(self_flat),-1)).T
        dp = (roi_norm.dot(self_flat)).T
        
        return Series(dp, Ts=self.Ts)

    def resize(self, factor, order=0):
        """Resize movie using scipy.ndimage.zoom
        
        Parameters
        ----------
        factor : float, tuple, list
            multiplying factor for dimensions (y,x), ex. 0.5 will downsample by 2. If number, it is used for all dimensions
        order : int
            order of interpolation (0=nearest, 1=bilinear, 2=cubic)
        """              
        if isinstance(factor, PF_numeric_types):
            factor = [factor,factor]
        elif isinstance(factor, PF_list_types):
            factor = list(factor)
        else:
            raise Exception('factor parameter not understood')
        res = szoom(self, [1]+factor, order=order)
        res = self.__class__(res, Ts=self.Ts)
        for ca in self._custom_attrs:
            res.__setattr__(ca, self.__getattribute__(ca))
        return res

    def resample(self, n=4):
        """Downsample movie by taking mean of every n frames
        """
        if n==1:
            return self

        if len(self)%n == 0:
            working = self
        else:
            working = self[:-(len(self)%n)]

        result = np.nanmean(working.reshape((-1,n,working.shape[1],working.shape[2])), axis=1)
        result.Ts = self.Ts*n
        return result

    def save(self, filename, fmt=None, codec='MJPG', fps=None, minmax=(0,300)):
        """Save movie for playback in video player

        Note that this function is intended for saving the movie in an avi/tiff-like format. Saving for further analysis should be performed using pyfluo's io functions.

        Parameters
        ----------
        filename : str
            destination file name, with or without extension
        fmt : str
            format to save movie, specified as file extension, ex. 'avi', 'mp4', 'tif'
        codec : int
            opencv fourcc code, as chars to be fed to cv2.FOURCC ex. 'IYUV'
        fps : int
            frames per second. Defaults to obj.fs
            applies to only to formats tht store time info, like avi
        minmax : tuple
            min and max pixels values corresponding to black and white. clipping occurs on either end
            if None: min and max value of movie are used
        """
        if fps == None:
            fps = 1/self.Ts
        
        if '.' in filename:
            filename, fmt = os.path.splitext(filename)
            fmt = fmt.strip('.')

        filename += '.'+fmt

        if fmt in ['tif','tiff']:
            tifffile.imsave(filename, data=np.asarray(self))
        elif fmt in ['mp4', 'avi']:
            if CV_VERSION == 3:
                codec = cv2.VideoWriter_fourcc(*codec)
            elif CV_VERSION == 2:
                codec = cv2.cv.FOURCC(*codec)

            if minmax is None:
                minn,maxx = np.min(self),np.max(self)
            else:
                minn,maxx = minmax
            data = np.clip(self, minn, maxx)
            data = data - minn
            data = data / maxx-minn
            data = (255*data).astype(np.uint8)
            y,x = data[0].shape
            vw = cv2.VideoWriter(filename, codec, fps, (x,y), isColor=True)
            for d in data:
                vw.write(cv2.cvtColor(d, cv2.COLOR_GRAY2BGR))
            vw.release()


###################
def play_mov(data, loop=True, fps=None, minmax=(0,300), scale=1, show_time=True, fontsize=1, rolling_mean=1, generator_fxn=None, **kwargs):
    """An opencv-based movie playing function, with keyboard controls

    This works, but has the following disadvantages:
    -requires opencv
    -is blocking

    Parameters
    ----------
    data : see description
        any iterable object where the iteration axis represents time and items of the iterable are 2d matrices, and which has the following methods/properties:
            Ts : sample rate
            shape : (z,y,x)
            min() : minimum value
            max() : maximum value
    loop : bool 
        repeat playback upon finishing
    fps : float
        playback rate in frames per second. Defaults to 1/Ts
    scale : float 
        scaling factor to resize playback images
    show_time : bool
        show time on image
    fontsize : float
        to display time
    rolling_mean : int
        number of frames to avg in display
    generator_fxn : str
        name of function to call to retrieve generator for object. if None, object is considered iterable
        if using this parameter, the data parameter should be an object that implements generator_fxn

    Playback controls (window must have focus):
        'f' : faster
        's' : slower
        'p' : pause / continue
        'r' : reverse
        '=' : zoom in
        '-' : zoom out
        'q' : quit

        
    """
    if fps==None:
        fps = 1/data.Ts
    fpms = fps / 1000.

    title = 'p / q / f / s / r / b / d / = / -'
    cv2.namedWindow(title)
    minn,maxx = minmax
    #minn,maxx = data.min(),data.max() # not needed anymore with new contrast method

    if generator_fxn is not None:
        gfunc = getattr(data,generator_fxn)
        dataiter = gfunc()
    else:
        gfunc = None
        dataiter = data
           
    global current_idx
    global oper_idx
    oper_idx = 0 # add/subtract
    current_idx = 0

    def fnext(d):
        global current_idx
        global oper_idx
        opfunc = [operator.add, operator.sub][oper_idx]
        current_idx = opfunc(current_idx, rolling_mean)
        try:
            return next(d)
        except TypeError:
            if current_idx>=len(data):
                current_idx = len(data)-1
            return data[current_idx]

    wait = 1./fpms
    paused = False
    while True:
        if not paused:

            # get frame
            size = tuple((scale*np.array(data.shape)[-1:0:-1]).astype(int))
            font_size = scale * fontsize * min(data[0].shape)/450.
            t = data.Ts*current_idx

            fr = np.mean([fnext(dataiter) for k in range(rolling_mean)], axis=0)

            fr = np.clip(fr, minn, maxx)
            fr -= minn
            fr /= maxx-minn
            fr = cv2.resize(fr,size)
            if show_time:
                cv2.putText(fr, '{:0.3f}'.format(t), (5,int(30*font_size)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (120,100,80), thickness=1)
            cv2.imshow(title,fr)
            
            # update indices
            if current_idx+rolling_mean >= data.shape[0]:
                if not loop:
                    break
                elif loop:
                    current_idx = 0
                    if gfunc is not None:
                        dataiter = gfunc()

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
        elif k == ord('b'):
            minn -= 20 #brighter
        elif k == ord('d'):
            minn += 20 #dimmer
        elif k == ord('r'):
            oper_idx = int(not oper_idx)
        elif k == ord('='):
            scale = scale * 1.5
        elif k == ord('-'):
            scale = scale / 1.5

    cv2.destroyWindow(title); cv2.waitKey(1)

class MoviePlayer(tk.Frame):
    """A movie player built on Tkinter with the goal of circumventing the need for blocking in the opencv implementation

    This is able to play movies at high framerate, but lacks some of the convenience live controls
    """

    def __init__(self, data, Ts=1, zoom=None):
        # Data handling
        self.data = data

        if self.data.dtype!=np.uint8 or self.data.min()!=0 or self.data.max()!=255:
            minn,maxx = self.data.min(),self.data.max()
            self.data = (255*((self.data-minn)/(maxx-minn))).astype(np.uint8)

        if zoom is not None:
            size = tuple((zoom*np.array(self.data.shape)[-1:0:-1]).astype(int))
            self.data = np.array([cv2.resize(i,size) for i in self.data])

        self.n, self.h, self.w = self.data.shape
        self.imshape = self.data.shape[1:][::-1]
        self.i = 0
        self.Ts = Ts

        if tk._default_root is not None:
            self.root = tk._default_root
            self.master = tk.Toplevel(self.root)
            self.made_root = False
        else:
            self.root = tk.Tk()
            self.master = self.root
            self.made_root = True

        window_size = [self.w, self.h]
        tk.Frame.__init__(self, master=self.master, width=window_size[0], height=window_size[1])
        self.bind("<Destroy>", self.end)

        # Tk setup
        self.canvas = tk.Canvas(self, width=self.w, height=self.h)
        # Populate frame
        self.canvas.place(relx=0,rely=0,x=0,y=0)
        self.pack()

        # Bindings
        self.root.bind('f', self.faster)
        self.root.bind('s', self.slower)
        self.root.bind('r', self.reverse)

        # Run events
        self._inc = 1
        self.recall = None
        self.ended = False
        self.t0 = time.clock()
        self.update_interval = 1 # seconds
        self.after(0, self.step)
        self.mainloop()

    def faster(self, *args):
        self.Ts /= 1.5

    def slower(self, *args):
        self.Ts *= 1.5

    def reverse(self, *args):
        self._inc = -self._inc

    def inc(self):
        n = self._inc
        self.i += n
        over = self.n - self.i
        if over <= 0:
            self.i = abs(over)
        if self.i < 0:
            self.i = self.n + self.i

    def step(self):
        if self.ended:
            return
        self.im = Image.fromarray(self.data[self.i])
        self.inc()
        self.photo = ImageTk.PhotoImage(image=self.im) # 1-3 msec
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW) # 1-5 msec
        if time.clock() - self.t0 > self.update_interval:
            self.update() # 20-40 msec
        self.recall = self.after(int(self.Ts*1000),self.step)

    def end(self, *args):
        self.ended = True
        if self.made_root:
            self.root.quit()
        else:
            self.master.destroy()
            self.root.quit() # no clue why this doesnt kill mpl windows but its okay
