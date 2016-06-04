import re, os, shutil, h5py
from .config import *
from .motion import compute_motion, apply_motion_correction
from .movies import Movie
from .fluorescence import compute_dff
from .images import Tiff
from .roi import select_roi, ROI
from .series import Series
import matplotlib.pyplot as pl
import pandas as pd

"""
This module is inteded to help organize imaging data and its associated data.
A "Group" is simply a directory that stores:
        -movie files (tifs)
        -motion correction data (hdf5)
"""

class Group():
    """
    Specifies a group of movies and their associated rois, series, motion corrections, etc.
    Can technically comprise any combination of objects, but currently intended to store info from 1 FOV over n imaging epochs (whether across days or not)
    """
    def __init__(self, name, data_path='.', recache_metadata=False):
        self.name = name
        self.data_path = data_path
        self.grp_path = os.path.join(self.data_path, self.name)
        self.metadata_path = os.path.join(self.grp_path, '{}_metadata.h5'.format(self.name))
        self.otherdata_path = os.path.join(self.grp_path, '{}_otherdata.h5'.format(self.name))
        self.mc_path = os.path.join(self.grp_path, '{}_mc.h5'.format(self.name))

        if not os.path.exists(self.grp_path):
            self = Group.from_raw(self.name, in_path=self.data_path, out_path=self.data_path)

        self.tif_files = sorted([o for o in os.listdir(self.grp_path) if o.endswith('.tif')])
        self.tif_names = [os.path.splitext(o)[0] for o in self.tif_files]
        self.tif_paths = [os.path.join(self.grp_path, fn) for fn in self.tif_files]

        self._loaded_example = None
        self.extract_metadata(recache_metadata)

    @property
    def example(self):
        if self._loaded_example is None:
            ex = Movie(self.tif_paths[0])
            self._loaded_example = apply_motion_correction(ex, self.mc_path)
        return self._loaded_example

    def extract_metadata(self, recache=False):
        if not os.path.exists(self.metadata_path) or recache:

            if len(self.tif_paths) == 0:
                return

            i2cs = []
            _i2c_ix = 0
            Tss = []

            for tp in self.tif_paths:
                mov = Tiff(tp, load_data=False) 
                i2c = mov.i2c
                if len(i2c):
                    i2c.ix[:,'filename'] = mov.filename
                    i2c.index += _i2c_ix
                    i2cs.append(i2c)
                Tss.append(mov.Ts)
                _i2c_ix += len(mov)

            i2c = pd.concat(i2cs)
            i2c = i2c.dropna()
            i2c.ix[:,'phase'] = (i2c.data-i2c.data.astype(int)).round(1)*10 
            i2c.ix[:,'trial'] = i2c.data.astype(int)
            self.i2c = i2c

            if not all([t==Tss[0] for t in Tss]):
                warnings.warn('Ts\'s do not all align in group. Using mean.')
            self.Ts = float(np.mean(Tss))

            with pd.HDFStore(self.metadata_path) as md:
                md.put('Ts', pd.Series(self.Ts))
                md.put('i2c', self.i2c)
        else:
            with pd.HDFStore(self.metadata_path) as md:
                self.Ts = float(md['Ts'])
                self.i2c = md['i2c']

    @classmethod
    def from_raw(self, name, in_path='.', out_path='.', regex=None, move_files=True):
        """Creates and returns a new group based on the input data

        If regex, uses that to include files
        If not, uses "name in filename" to include files
        """

        # create group
        grp_path = os.path.join(out_path, name)
        if os.path.exists(grp_path):
            raise Exception('Group \'{}\' already exists.'.format(grp_path))
        os.mkdir(grp_path)
        
        # load files in specified directory
        filenames = [os.path.join(in_path, p) for p in os.listdir(in_path)]

        # filter files
        if regex is not None:
            filenames = [fn for fn in filenames if re.search(regex, fn)]
        else:
            filenames = [fn for fn in filenames if name in fn]

        # move/copy files
        if move_files:
            mv_func = shutil.move
        else:
            mv_func = shutil.copy2
        for fn in filenames:
            mv_func(fn, grp_path)
        
        return Group(name, data_path=out_path)
    
    def motion_correct(self):

        out_file = h5py.File(self.mc_path)
       
        did_any = False
        for fp,fn in zip(self.tif_paths,self.tif_names):
            if fn in out_file:
                continue
            did_any = True
            print(fp)
            mov = Movie(fp)
            templ,vals = compute_motion(mov, max_shift=(25,25))
            gr = out_file.create_group(fn)
            gr.create_dataset('shifts', data=vals)
            gr.create_dataset('template', data=templ)

        if did_any:
            
            template_mov = np.asarray([np.asarray(out_file[k]['template']) for k in out_file if 'global' not in k])
            glob_template,glob_vals = compute_motion(template_mov, max_shift=(25,25))
            if 'global_shifts' in out_file:
                del out_file['global_shifts']
            shifts_dataset = out_file.create_dataset('global_shifts', data=glob_vals)
            shifts_dataset.attrs['filenames'] = np.asarray(self.tif_names).astype('|S150')
            if 'global_template' in out_file:
                del out_file['global_template']
            out_file.create_dataset('global_template', data=glob_template)

        out_file.close()

    def get_roi(self, reselect=False):
        with h5py.File(self.otherdata_path) as od:
            if 'roi' in od and not reselect:
                roi = ROI(od['roi'])
            else:
                with h5py.File(self.mc_path) as mc_file:
                    gt = np.asarray(mc_file['global_template'])
                roi = select_roi(img=gt)                
                od.create_dataset('roi', data=np.asarray(roi))
                if 'dff' in od:
                    del od['dff']
        return roi

    def get_dff(self, redo_raw=False, redo_dff=False):
        with h5py.File(self.otherdata_path) as od:
            if not 'roi' in od:
                self.get_roi()

            if 'raw' in od:
                raw_grp = od['raw']
            else:
                raw_grp = od.create_group('raw')

            roi = ROI(np.asarray(od['roi']))
            for filepath,filename in zip(self.tif_paths, self.tif_names):
                if (not redo_raw) and filename in raw_grp:
                    continue
                print('Extracting raw from {}'.format(filepath))
                
                # clear existing if necessary
                if filename in raw_grp and redo_raw:
                    del raw_grp[filename]

                if filename not in raw_grp:
                    mov = apply_motion_correction(Movie(filepath), self.mc_path)
                    tr = mov.extract(roi)
                    raw_grp.create_dataset(filename, data=np.asarray(tr))
            
            if ('dff' not in od) or redo_dff:
                # clear if necessary:
                if 'dff' in od:
                    del od['dff']
                dff = []
                print ('Computing DFF...')
                for fn in self.tif_names:
                    print(fn)
                    r = Series(np.asarray(od['raw'][fn]), Ts=self.Ts)
                    d = np.asarray(compute_dff(r))
                    dff.append(d)
                od.create_dataset('dff', data=np.concatenate(dff))
            else:
                dff = np.asarray(od['dff'])

            raw = [np.asarray(raw_grp[k]) for k in raw_grp]
        return Series(dff, Ts=self.Ts), Series(np.concatenate(raw), Ts=self.Ts)

    def project(self):
        with h5py.File(self.mc_path) as mc_file:
            gt = np.asarray(mc_file['global_template'])
        pl.imshow(gt, cmap=pl.cm.Greys_r)
        roi = self.get_roi()
        roi.show(labels=True)
