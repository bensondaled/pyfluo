#from skimage.external import tifffile
import tifffile #moving away form skimage b/c it's not up to date, check to see if it is by now
import numpy as np, pandas as pd
import os

def extract_i2c(tif):
    pages = tif.pages
    ids = [p.tags['image_description'].value.decode('UTF8') for p in pages]
    i0 = [t.index('I2CData = ')+11 for t in ids]
    i1 = [t[i0i:].index('\n') for t,i0i in zip(ids,i0)]
    data = [t[i0i:i0i+i1i-1].strip(' {}') for i0i,i1i,t in zip(i0,i1,ids)]
    data = [(i,d) for i,d in enumerate(data)]
    data = [(d[0],d[1].split('} {')) for d in data]
    ix,data = zip(*[(d[0],i) for d in data for i in d[1]])
    data = [[float(ii.strip('\' ')) for ii in i.split(',')] if len(i) else [None,None] for i in data]
    data = pd.DataFrame(data, columns=['si_timestamp','data'], index=ix)
    return data

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
    def __init__(self, file_path, load_data=True, **kwargs):
        self.file_path = file_path
        self.Ts = None
        self.tf_obj = None

        if isinstance(file_path, str):
            self.filename = os.path.splitext(os.path.split(file_path)[-1])[0]
            tf = tifffile.TiffFile(self.file_path, **kwargs)
            if load_data:
                self.data = tf.asarray()
            else:
                self.data = None
            self.Ts = self._extract_Ts(tf)
            self.i2c = extract_i2c(tf)
            self.tf_obj = tf
        elif any([isinstance(file_path, t) for t in [np.ndarray,list]]):
            tfs = [tifffile.TiffFile(f, **kwargs) for f in self.file_path if f.endswith('.tif')]
            if self.load_data:
                data = [tf.asarray() for tf in tfs]
            else:
                self.data = None
            self.data = np.concatenate([d if d.ndim==3 else [d] for d in data], axis=0)
            self.Ts = self._extract_Ts(tfs[0])
            self.i2c = np.concatenate([extract_i2c(t) for t in tfs])
    def __len__(self):
        if self.tf_obj:
            return len(self.tf_obj)
        else:
            return None
    def _extract_Ts(self, t):
        try:
            pg = t.pages[0]
            imd = pg.tags['image_description'].value
            if isinstance(imd, bytes):
                imd = imd.decode('UTF8')
            self.tiff_tags = si_parse(imd)
            fs = self.tiff_tags.get('scanimage.SI.hRoiManager.scanFrameRate', None)
            if fs:
                return 1./fs
            else:
                return None
        except:
            self.tiff_tags = None
            return None
