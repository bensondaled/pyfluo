from skimage.external import tifffile
import numpy as np

def val_parse(v):
    # parse values from si tags into python objects if possible
    try:
        return eval(v)
    except:
        if v == 'true':
            return True
        elif v == 'false':
            return False
        elif v == 'NaN':
            return np.nan
        elif v == 'inf' or v == 'Inf':
            return np.inf
        else:
            return v
def si_parse(imd):
    # parse image_description field embedded by scanimage
    imd = imd.split('\n')
    imd = [i for i in imd if '=' in i]
    imd = [i.split('=') for i in imd]
    imd = [[ii.strip(' \r') for ii in i] for i in imd]
    imd = {i[0]:val_parse(i[1]) for i in imd}
    return imd

class Tiff(object):
    """An object to store tiff file data

    Parameters
    ----------
    file_path : str 
        path to tiff file, or list thereof

    Attributes
    ----------
    data : np.ndarray
        the data, with shape (n,y,x)

    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.Ts = None

        if isinstance(file_path, str):
            tf = tifffile.TiffFile(self.file_path)
            self.data = tf.asarray()
            self.Ts = self._extract_Ts(tf)
        elif any([isinstance(file_path, t) for t in [np.ndarray,list]]):
            tfs = [tifffile.TiffFile(f) for f in self.file_path if f.endswith('.tif')]
            data = [tf.asarray() for tf in tfs]
            self.data = np.concatenate([d if d.ndim==3 else [d] for d in data], axis=0)
            self.Ts = self._extract_Ts(tfs[0])
    def _extract_Ts(self, t):
        try:
            pg = t.pages[0]
            imd = pg.tags['image_description'].value
            if isinstance(imd, bytes):
                imd = imd.decode('UTF8')
            imd = si_parse(imd)
            fs = imd.get('scanimage.SI.hRoiManager.scanFrameRate', None)
            if fs:
                return 1./fs
            else:
                return None
        except:
            raise
            return None
