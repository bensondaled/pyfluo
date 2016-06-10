import re, os, shutil, h5py, warnings
import matplotlib.pyplot as pl
import pandas as pd
from .config import *
from .motion import compute_motion, apply_motion_correction
from .movies import Movie
from .fluorescence import compute_dff
from .images import Tiff
from .roi import select_roi, ROI
from .series import Series

class Group():
    def __init__(self, name, data_path='.'):

        # user inputs
        self.name = name
        self.data_path = data_path
        self.grp_path = os.path.join(self.data_path, self.name)

        if not os.path.exists(self.grp_path):
            # group files into directory
            self = Group.from_raw(self.name, in_path=self.data_path, out_path=self.data_path)
        
        # default group files
        self.tifdata_path = os.path.join(self.grp_path, '{}_tifdata.h5'.format(self.name))
        self.otherdata_path = os.path.join(self.grp_path, '{}_otherdata.h5'.format(self.name))
        self.mc_path = os.path.join(self.grp_path, '{}_mc.h5'.format(self.name))
        self.mov_path = os.path.join(self.grp_path, '{}_mov.h5'.format(self.name))
        # determine tif data
        self.extract_tifdata()

        self._loaded_example = None

    @property
    def example(self):
        if self._loaded_example is None:
            ex = Movie(self.tif_paths[0])
            self._loaded_example = apply_motion_correction(ex, self.mc_path)
        return self._loaded_example

    def get_mov(self, idx):
        mov = Movie(self.tif_paths[idx])
        mov = apply_motion_correction(mov, self.mc_path)
        return mov

    def merge_movs(self):
        if os.path.exists(self.mov_path):
            ans = input('Path exists. Overwrite? (y/n)')
            if ans == 'y':
                os.remove(self.mov_path)
            else:
                return
        with h5py.File(self.mov_path) as movfile:
            ds = movfile.create_dataset('mov', (self.mov_lengths.sum(),self.y,self.x), compression='gzip')

            idx = 0
            for i,tn in enumerate(self.tif_names):
                mov = self.get_mov(i)
                ds[idx:idx+len(mov)] = np.asarray(mov)
                idx += len(mov)

    def extract_tifdata(self, recache=False):
        """Extracts features of tifs from caches if present, or from files if not yet cached

        Fields of interest:
            -tif file names
            -dimensions of tif files
            -i2c data from tif files
            -sampling interval (Ts) of tifs
        """

        if not os.path.exists(self.tifdata_path) or recache:
            self.tif_files = sorted([o for o in os.listdir(self.grp_path) if o.endswith('.tif')])
            self.tif_names = [os.path.splitext(o)[0] for o in self.tif_files]
            self.tif_paths = [os.path.join(self.grp_path, fn) for fn in self.tif_files]

            if len(self.tif_paths) == 0:
                warnings.warn('No tif files detected in group.')
                return

            i2cs, Tss, shapes = [],[],[]
            _i2c_ix = 0

            for tp in self.tif_paths:
                mov = Tiff(tp, load_data=False) 
                shapes.append(mov.shape)
                i2c = mov.i2c
                if len(i2c):
                    i2c.ix[:,'filename'] = mov.filename
                    i2c.index += _i2c_ix
                    i2cs.append(i2c)
                Tss.append(mov.Ts)
                _i2c_ix += len(mov)

            # i2c
            i2c = pd.concat(i2cs).dropna()
            i2c.ix[:,'phase'] = (i2c.data-i2c.data.astype(int)).round(1)*10 
            i2c.ix[:,'trial'] = i2c.data.astype(int)
            self.i2c = i2c
            self.x,self.y = 512,512
            self.mov_lengths = pd.Series({tp:nf for tp,nf in zip(self.tif_names, nframess)})

            if not all([t==Tss[0] for t in Tss]):
                warnings.warn('Ts\'s do not all align in group. Using mean.')
            self.Ts = float(np.mean(Tss))

            with pd.HDFStore(self.tifdata_path) as md:
                md.put('Ts', pd.Series(self.Ts))
                md.put('i2c', self.i2c)
                md.put('mov_lengths', self.mov_lengths)
        else:
            # these 3 lines temp until I fix structure:
            self.tif_files = sorted([o for o in os.listdir(self.grp_path) if o.endswith('.tif')])
            self.tif_names = [os.path.splitext(o)[0] for o in self.tif_files]
            self.tif_paths = [os.path.join(self.grp_path, fn) for fn in self.tif_files]
            with pd.HDFStore(self.tifdata_path) as md:
                if any([i not in md for i in ['Ts','i2c','mov_lengths']]):
                    return self.extract_tifdata(recache=True)
                self.Ts = float(md['Ts'])
                self.i2c = md['i2c']
                self.mov_lengths = md['mov_lengths']

    @classmethod
    def find_groups(self, path):
        # find all groups in a dir
        return [i for i in os.listdir(path) if os.path.isdir(os.path.join(path,i))]

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

    def get_roi(self, i=None, reselect=False):
        if i is None:
            i = ''
        roiname = 'roi{}'.format(i)
        with h5py.File(self.otherdata_path) as od:
            if roiname in od and not reselect:
                roi = ROI(od[roiname])
            else:
                if i != '': #specific roi was requested but not present
                    return None
                with h5py.File(self.mc_path) as mc_file:
                    gt = np.asarray(mc_file['global_template'])
                roi = select_roi(img=gt)                
                self.set_roi(roi)
        return roi

    def set_roi(self, roi):
        with h5py.File(self.otherdata_path) as od:
            if 'roi' in od:
                i = 0
                while 'roi{}'.format(i) in od:
                    i+=1
                od.move('roi', 'roi{}'.format(i))
                if 'raw' in od:
                    od.move('raw', 'raw{}'.format(i))
                if 'dff' in od:
                    od.move('dff', 'dff{}'.format(i))
            od.create_dataset('roi', data=np.asarray(roi))

    def get_dff(self, i=None, redo_raw=False, redo_dff=False, dff_kwargs={}):
        if i is None:
            i = ''
        dffname = 'dff{}'.format(i)
        rawname = 'raw{}'.format(i)
        roiname = 'roi{}'.format(i)
        with h5py.File(self.otherdata_path) as od:
            if not roiname in od:
                raise Exception('No roi specified for trace extraction.')

            if rawname in od:
                raw_grp = od[rawname]
            else:
                raw_grp = od.create_group(rawname)

            roi = ROI(np.asarray(od[roiname]))
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
            
            raw = [np.asarray(raw_grp[k]) for k in raw_grp]
            raw = Series(np.concatenate(raw), Ts=self.Ts)

            if (dffname not in od) or redo_dff:
                # clear if necessary:
                if dffname in od:
                    del od[dffname]
                print ('Computing DFF...')
                dff = np.asarray(compute_dff(raw, **dff_kwargs))
                od.create_dataset(dffname, data=dff)
            dff = Series(np.asarray(od[dffname]), Ts=self.Ts)

        return dff, raw

    def extract_roi_mov(self, roi, frame_idxs):
        if not os.path.exists(self.mov_path):
            raise Exception('Run merge_movs first, this is too slow if you don\'t have hdf5 version stored.')

        # build bounding box
        args = np.argwhere(roi)
        (ymin,xmin),(ymax,xmax) = args.min(axis=0),args.max(axis=0)
        pad = 3
        
        with h5py.File(self.mov_path) as movfile:
            mov = movfile['mov']
            yslice = slice(max(0,ymin-pad), min(ymax+pad,mov.shape[1]))
            xslice = slice(max(0,xmin-pad), min(xmax+pad,mov.shape[2]))
            chunks = [mov[fi[0]:fi[1],yslice,xslice] for fi in frame_idxs]
        return np.squeeze(chunks)

    def project(self, roi=True, ax=None, show=True):
        if ax is None:
            ax = pl.gca()
        with h5py.File(self.mc_path) as mc_file:
            gt = np.asarray(mc_file['global_template'])
        if show:
            ax.imshow(gt, cmap=pl.cm.Greys_r)
        if roi and show:
            roi = self.get_roi()
            roi.show(labels=True, ax=ax)
        return gt
