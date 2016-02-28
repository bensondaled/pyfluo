# External imports
import pandas as pd, numpy as np, pylab as pl
import warnings, cv2
# Internal imports
from .config import *
from .images import Tiff, AVI
from .roi import select_roi

class Movie(pd.DataFrame):
    """Movie object
    """

    _metadata = ['Ts', 'frame_shape']
    _metadata_defaults = [None, None]

    def __init__(self, data, *args, **kwargs):

        # Set custom fields
        for md,dmd in zip(self._metadata, self._metadata_defaults):
            setattr(self, md, kwargs.pop(md, dmd))

        # Assess data type
        if isinstance(data, np.ndarray) and data.ndim==2:
            # 2d array passed as data
            if self.frame_shape is None:
                raise Exception('Frame shape must be given when 2d array supplied as movie.')
        elif isinstance(data, Tiff) or isinstance(data, AVI):
            # Tiff/AVI object passed as data, extract a copy of its data attribute
            data = data.data.copy()
        elif any([isinstance(data, dt) for dt in PF_str_types]):
            if data.endswith('.tif') or data.endswith('.tiff'):
                data = Tiff(data).data
            elif data.endswith('.avi'):
                data = AVI(data).data
        elif isinstance(data, Movie):
            return data
        else:
            pass
            #raise Exception('Data type not recognized.')
       
        # Set/adjust Ts
        if self.Ts is not None:
            if 'index' in kwargs:
                warnings.warn('Ts parameter ignored because index was also supplied.')
            else:
                kwargs['index'] = self.Ts*np.arange(0, len(data))
        elif self.Ts is None:
            self.Ts = 1

        # Set/adjust frame shape
        if self.frame_shape is None:
            self.frame_shape = np.asarray(data.shape[1:])
        else:
            self.frame_shape = np.asarray(self.frame_shape)

        # If data ends up in 3d array form, flatten
        if isinstance(data, np.ndarray) and data.ndim==3:
            # 3d array passed as data, flatten frames
            data = data.reshape([data.shape[0], -1])

        # Init object
        super(Movie, self).__init__(data, *args, **kwargs)

    @property
    def _constructor(self):
        return Movie
    
    @property
    def _constructor_sliced(self):
        return pd.Series

    def as_3d(self):
        return np.asarray(self).reshape([len(self)]+list(self.frame_shape))

    def project(self, method=np.mean, show=False, **kwargs):
        res = method( self._get_numeric_data() , **kwargs).reshape(self.frame_shape)
        if show:
            pl.imshow(res)
        return res

    def play(self, contrast=1.0, scale=1.0, loop=True, show_time=True, font_size=1, fps=None):
        minn,maxx = self.values.min(),self.values.max()
        size = tuple((scale*self.frame_shape[::-1]).astype(int))
        if fps==None:
            fps = 1/self.Ts
        fpms = fps / 1000.

        current_idx = 0
        while True:
            t = self.index[current_idx]
            fr = self.iloc[current_idx].reshape(self.frame_shape)
            fr = contrast * (fr-minn)/(maxx-minn)
            fr[fr>1.0] = 1.0 #clips; this should become a parameter
            fr[fr<0.0] = 0.0 #clips; this should become a parameter
            fr = cv2.resize(fr,size)
            if show_time:
                cv2.putText(fr, '%0.3f'%(t), (5,int(30*font_size)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (120,100,80), thickness=1)
            cv2.imshow('Movie', fr)
            k=cv2.waitKey(int(1./fpms))
            if k == ord('q'):
                break
            current_idx += 1
            if current_idx == len(self):
                if not loop:
                    break
                elif loop:
                    current_idx = 0
        cv2.destroyWindow('Movie')
    
    def select_roi(self, *args, **kwargs):
        """A convenience method for pyfluo.roi.select_roi(self.project(), *args, **kwargs)

        Parameters
        ----------
        proj : def
            'method' parameter for Movie.project, used in display
        *args, **kwargs
            those accepted by pyfluo.roi.select_roi
        """
        zp = self.project(show=False, method=kwargs.pop('proj',np.std))
        return select_roi(zp, *args, **kwargs)
    
    def extract_by_roi(self, roi, method=np.mean, as_pd=True):
        """Extract a time series consisting of one value for each movie frame, attained by performing an operation over the regions of interest (ROI) supplied
        
        Parameters
        ----------
        roi : pyfluo.ROI
            the rois over which to extract data
        method : def 
            the function by which to convert the data within an ROI to a single value. Defaults to np.mean
        as_pd : bool
            return as pandas object instead of pyfluo
            
        Returns
        -------
        Trace object, with multiple columns corresponding to multiple ROIs
        """
        if roi.ndim == 3:
            flatshape = (len(roi),-1)
        elif roi.ndim == 2:
            flatshape = -1
        roi_flat = roi.reshape(flatshape)
        self_flat = self.reshape((len(self),-1)).T
        dp = (roi_flat.dot(self_flat)).T
        if not as_pd:
            return Trace(dp/roi_flat.sum(axis=-1), time=self.time.copy(), Ts=self.Ts)
        elif as_pd:
            return pd.DataFrame(dp/roi_flat.sum(axis=-1), index=self.time.copy())
    
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
            fps = self.fs
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

    def motion_correct(self, *args, **kwargs):
        """Convenience method for pyfluo.motion.motion_correct
        """
        return motion_correct(self, *args, **kwargs)


if __name__ == '__main__':
    import tifffile
    data = tifffile.imread('/Users/ben/phd/data/2p/mov.tif')
    m = Movie(data, Ts=1/64)

