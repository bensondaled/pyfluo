#from skimage.external import tifffile
import tifffile #moving away form skimage b/c it's not up to date, check to see if it is by now
import numpy as np, pandas as pd
import os, glob, h5py, time

from ..config import *
from ..util import Elapsed

def fast_i2c(pages, i2c_type=None):
    if i2c_type is None:
        i2c_type = lambda x: x
    ids = [p.tags['image_description'].value.decode('UTF8') for p in pages]
    i0 = [t.index('I2CData = ')+11 for t in ids]
    i1 = [t[i0i:].index('\n') for t,i0i in zip(ids,i0)]
    data = [t[i0i:i0i+i1i-1].strip(' {}') for i0i,i1i,t in zip(i0,i1,ids)]
    data = [(i,d) for i,d in enumerate(data)]
    data = [(d[0],d[1].split('} {')) for d in data]
    ix,data = zip(*[(d[0],i) for d in data for i in d[1]])
    data = [[i2c_type(ii.strip('\' ')) for ii in i.split(',')] if len(i) else [None,None] for i in data]
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
def si_parse(pg):
    imd = pg.tags['image_description'].value
    if isinstance(imd, bytes):
        imd = imd.decode('UTF8')
    # given one tif page, parse image_description field embedded by scanimage
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
    filename : str
        path to tiff file

    Attributes
    ----------
    data : np.ndarray
        the data, with shape (n,y,x)

    """
    def __init__(self, filename):

        # filenames
        self.file_path = os.path.abspath(filename)
        self.path,self.filename = os.path.split(self.file_path)
        self.name,self.suffix = os.path.splitext(self.filename)

        self.data,self.metadata,self.i2c,self.si_data = self._load_data()
    
    def _load_data(self):
        metadata_columns=['name','n','y','x','Ts']
        metadata = pd.DataFrame([[None]*len(metadata_columns)], columns=metadata_columns)

        with tifffile.TiffFile(self.file_path) as tif:
            data = tif.asarray()

            page = tif.pages[0]
            pagedata = si_parse(page)
            
            metadata['name'] = self.name
            metadata['y'] = page.shape[0]
            metadata['x'] = page.shape[1]
            metadata['Ts'] = 1/pagedata.get('scanimage.SI.hRoiManager.scanFrameRate', None)
            metadata['n'] = len(tif.pages)

            i2c = fast_i2c(tif.pages).dropna()
            i2c.ix[:,'name'] = self.name
            i2c.ix[:,'frame_idx'] = i2c.index
            i2c.reset_index(drop=True, inplace=True)

        return data,metadata,i2c,pagedata

class TiffGroup(object):
    def __init__(self, files, sort=True):

        self._files_orig = files
        files = glob.glob(files)
        files = [f for f in files if f.endswith('.tif')]
        if sort:
            files = sorted(files)

        # filenames
        self.file_paths = [os.path.abspath(f) for f in files]
        self.path = os.path.split(self.file_paths[0])[0]
        self.file_names = [os.path.split(fp)[-1] for fp in self.file_paths]
        self.names = [os.path.splitext(fn)[0] for fn in self.file_names]
        self.nfiles = len(self.names)
        self.common_name = self._determine_common_name()

    def _determine_common_name(self):
        # currently, just chooses longest common string, where underscore-separated chunks are treated as units
        units = [n.split('_') for n in self.names]
        i,diverged = 0,False
        while not diverged:
            i += 1
            diverged = not all([n[:i]==units[0][:i] for n in units[1:]])
        return '_'.join(units[0][:i-1])

    def to_hdf5(self, verbose=True):

        self.hdf_path = os.path.join(self.path, self.common_name+'.h5')
        t0 = time.time()
        
        for idx in range(self.nfiles):
            if verbose:
                print('File ({}/{}): {}'.format(idx+1, self.nfiles, self.names[idx]), flush=True)

            t = Tiff(self.file_paths[idx])

            # store metadata
            with Elapsed('Storing metadata...', verbose=verbose):
                with pd.HDFStore(self.hdf_path) as h:
                    h.put('si_data', pd.Series(t.si_data))
                    info = t.metadata
                    info.index = [idx]*len(info)
                    h.append('info', info)
                    i2c = t.i2c
                    i2c.reset_index(inplace=True, drop=True)
                    i2c.ix[:,'file_idx'] = idx
                    h.append('i2c', i2c)

            # store data
            with Elapsed('Storing data...', verbose=verbose):
                with h5py.File(self.hdf_path) as h:
                    if 'data' not in h:
                        h.create_dataset('data', data=t.data, maxshape=(None,)+t.data.shape[1:], compression='lzf')
                    elif 'data' in h:
                        newshape = (t.data.shape[0]+len(h['data']),) + h['data'].shape[1:]
                        h['data'].resize(newshape)
                        h['data'][-len(t.data):] = t.data

        if verbose:
            print('{} complete ({:0.2f} sec).'.format(self.common_name, time.time()-t0))
