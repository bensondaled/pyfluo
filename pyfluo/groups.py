import re, os, shutil, h5py, warnings
import matplotlib.pyplot as pl
import pandas as pd
from .config import *
from .motion import compute_motion, apply_motion_correction
from .movies import Movie
from .fluorescence import compute_dff, detect_transients
from .images import Tiff
from .roi import select_roi, ROI
from .series import Series

class Group():
    #TODO: needs MAJOR overhaul, since I adjusted Tiff and Movie. This will be a far less involved class, just handling manipulations and analysis of already-processed groups
    # Thinking I should rename it to Data, and its role will be to handle all data related to some images, as opposed to Movie which is specifically for in-memory movies
    """
    Stores a set of tif-derived data with associated metadata and computations.
    """
    def __init__(self, name, data_path='.'):

        # user inputs
        self.name = name
        self.data_path = data_path
        self.grp_path = os.path.join(self.data_path, self.name)

        if not os.path.exists(self.grp_path):
            # group files into directory
            self._build_structure(self.name, in_path=self.data_path, out_path=self.data_path)

        # default group files - these paths exist for all groups, whether or not the given file is created yet
        self.tifdata_path = os.path.join(self.grp_path, '{}_tifdata.h5'.format(self.name))
        self.otherdata_path = os.path.join(self.grp_path, '{}_otherdata.h5'.format(self.name))
        self.mc_path = os.path.join(self.grp_path, '{}_mc.h5'.format(self.name))
        self.mov_path = os.path.join(self.grp_path, '{}_mov.h5'.format(self.name))

        # determine tif data
        self._extract_tifdata()
            
        # inferred properties based on tif data
        # tif file details
        self.tif_names = self.shapes.filename.values
        self.tif_files = np.array([tn+'.tif' for tn in self.tif_names])
        self.tif_paths = np.array([os.path.join(self.grp_path, tf) for tf in self.tif_files])
        # tif file shape details
        shp = self.shapes.iloc[0]
        assert (self.shapes.x==shp.x).all()
        assert (self.shapes.y==shp.y).all()
        self.y,self.x = shp[['y','x']]

        # movie format availability details
        self.tifs_available = all([os.path.exists(f) for f in self.tif_paths])
        self.merge_mov_available = os.path.exists(self.mov_path)

        self._loaded_example = None

    @property
    def example(self):
        """
        Retrieves an example movie and stores it in memory for repeated use
        """
        if self._loaded_example is None:
            self._loaded_example = self.get_mov(0)
        return self._loaded_example

    def _get_mov_idxs(self, idx):
        idxs = self.shapes.z.cumsum().values
        idxs = np.append(0, idxs)
        return [idxs[idx],idxs[idx+1]]

    def get_mov(self, idx, crop=False):
        if self.merge_mov_available:
            with h5py.File(self.mov_path) as f:
                filename = f['mov'].attrs['source_names'][idx]
                idxs = self._get_mov_idxs(idx)
                data = np.asarray(f['mov'][idxs[0]:idxs[1]])
                mov = Movie(data, Ts=self.Ts, filename=filename)
        elif self.tifs_available:
            mov = Movie(self.tif_paths[idx])
            mov = apply_motion_correction(mov, self.mc_path, crop=crop)
        else:
            raise Exception('No available source for movie loading.')

        return mov

    def merge_movs(self, remove_tifs=False):
        """
        If tifs are available, copies all data into a single hdf5, removing tifs if desired
        """
        if os.path.exists(self.mov_path):
            ans = input('Path to merged movie exists. Overwrite? (y/n)')
            if ans == 'y':
                os.remove(self.mov_path)
            else:
                return

        assert self.tifs_available, 'Tif files are not present in group directory; merge cannot be made.'

        with h5py.File(self.mov_path) as movfile:
            ds = movfile.create_dataset('mov', (self.shapes.z.sum(),self.y,self.x), compression='gzip', compression_opts=2)

            idx = 0
            indices = []
            for i,tn in enumerate(self.tif_names):
                print(tn)
                mov = self.get_mov(i, crop=False)
                indices.append([idx,idx+len(mov)])
                ds[idx:idx+len(mov)] = np.asarray(mov)
                idx += len(mov)
                if remove_tifs:
                    os.remove(self.tif_paths[i])
                    self.tifs_available = False

            ds.attrs['source_names'] = '\n'.join(self.tif_names)

    def _extract_tifdata(self, recache=False):
        """Extracts features of tifs from caches if present, or from files if not yet cached

        Fields of interest:
            -tif file names
            -dimensions of tif files
            -i2c data from tif files
            -sampling interval (Ts) of tifs
        """

        if (not os.path.exists(self.tifdata_path)) or recache:
            print('Extracting tif data for group...')
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
                expg = mov.tf_obj.pages[0].asarray()
                shapes.append(dict(filename=mov.filename, z=len(mov), y=expg.shape[0], x=expg.shape[1]))
                i2c = mov.i2c
                if len(i2c):
                    i2c.ix[:,'filename'] = mov.filename
                    i2c.index += _i2c_ix
                    i2cs.append(i2c)
                Tss.append(mov.Ts)
                _i2c_ix += len(mov)

            # i2c
            i2c = pd.concat(i2cs).dropna()

            # shapes
            shapes = pd.DataFrame(shapes)

            # Tss
            if not all([t==Tss[0] for t in Tss]):
                warnings.warn('Ts\'s do not all align in group. Using mean.')
            Ts = float(np.mean(Tss))

            with pd.HDFStore(self.tifdata_path) as md:
                md.put('Ts', pd.Series(Ts))
                md.put('i2c', i2c)
                md.put('shapes', shapes)
            
        # in all cases: 
        with pd.HDFStore(self.tifdata_path) as md:
            if any([i not in md for i in ['Ts','i2c','shapes']]):
                return self._extract_tifdata(recache=True)
            self.Ts = float(md['Ts'])
            self.i2c = md['i2c']
            self.shapes = md['shapes']

    def _build_structure(self, name, in_path='.', out_path='.', regex=None, move_files=True):
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
        
    def motion_correct(self, max_shift=25):

        if not self.tifs_available:
            raise Exception('Tifs not available to motion correct. This must be done with tifs, since merge movie stores motion-corrected data.')

        out_file = h5py.File(self.mc_path)

        did_any = False
        for fp,fn in zip(self.tif_paths,self.tif_names):
            if fn in out_file:
                continue
            did_any = True
            print(fp)
            mov = Movie(fp)
            templ,vals = compute_motion(mov, max_shift=max_shift)
            gr = out_file.create_group(fn)
            gr.create_dataset('shifts', data=vals)
            gr.create_dataset('template', data=templ)

        if 'max_shift' not in out_file.attrs:
            out_file.attrs['max_shift'] = max_shift

        if did_any:
            
            template_mov = np.asarray([np.asarray(out_file[k]['template']) for k in out_file if 'global' not in k])
            glob_template,glob_vals = compute_motion(template_mov, max_shift=max_shift)
            if 'global_shifts' in out_file:
                del out_file['global_shifts']
            shifts_dataset = out_file.create_dataset('global_shifts', data=glob_vals)
            shifts_dataset.attrs['filenames'] = np.asarray(self.tif_names).astype('|S150')
            if 'global_template' in out_file:
                del out_file['global_template']
            out_file.create_dataset('global_template', data=glob_template)

        out_file.close()

    def get_motion(self):
        result = []
        with h5py.File(self.mc_path) as f:
            for tn in self.tif_names:
                gr = f[tn]
                ds = gr['shifts']
                result.append(np.asarray(ds))
        return np.concatenate(result)

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

    def get_dff(self, i=None, redo_raw=False, redo_dff=False, redo_transients=False, dff_kwargs={}, transient_kwargs={}):
        if i is None:
            i = ''
        dffname = 'dff{}'.format(i)
        transname = 'transients{}'.format(i)
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

            for fileidx,filename in enumerate(self.tif_names):
                if (not redo_raw) and filename in raw_grp:
                    continue
                print('Extracting raw from {}'.format(filename))
                
                # clear existing if necessary
                if filename in raw_grp and redo_raw:
                    del raw_grp[filename]

                if filename not in raw_grp:
                    mov = self.get_mov(fileidx)
                    tr = mov.extract(roi)
                    raw_grp.create_dataset(filename, data=np.asarray(tr))
            
            raw = [np.asarray(raw_grp[k]) for k in raw_grp]
            raw = Series(np.concatenate(raw), Ts=self.Ts)

            if (dffname not in od) or redo_dff:
                # clear if necessary:
                print ('Computing DFF...')
                dff = np.asarray(compute_dff(raw, **dff_kwargs))
                if dffname in od:
                    del od[dffname]
                od.create_dataset(dffname, data=dff)
            dff = Series(np.asarray(od[dffname]), Ts=self.Ts)

            if (transname not in od) or redo_transients:
                # clear if necessary:
                print ('Computing transients...')
                trans = np.asarray(detect_transients(dff, **transient_kwargs))
                if transname in od:
                    del od[transname]
                od.create_dataset(transname, data=trans)
            trans = Series(np.asarray(od[transname]), Ts=self.Ts)

        return trans, dff, raw

    def extract_roi_mov(self, roi, frame_idxs, mean=False, pad=3):
        # frame_idxs: either 2-tuples, or slices
        if not os.path.exists(self.mov_path):
            raise Exception('Run merge_movs first, this is too slow if you don\'t have hdf5 version stored.')

        for idx,fi in enumerate(frame_idxs):
            if not isinstance(fi,slice):
                frame_idxs[idx] = slice(fi[0], fi[1])

        # build bounding box
        if roi is None:
            roi = np.ones([self.y,self.x])
        args = np.argwhere(roi)
        (ymin,xmin),(ymax,xmax) = args.min(axis=0),args.max(axis=0)
        
        with h5py.File(self.mov_path) as movfile:
            mov = movfile['mov']
            yslice = slice(max(0,ymin-pad), min(ymax+pad,mov.shape[1]))
            xslice = slice(max(0,xmin-pad), min(xmax+pad,mov.shape[2]))
            
            if mean:
                chunk_size = frame_idxs[0].stop-frame_idxs[0].start
                assert [(fi.stop-fi.start)==(chunk_size) for fi in frame_idxs]
                chunks = np.empty([chunk_size,ymax-ymin,xmax-xmin])
            else:
                chunks = []
            for idx,fi in enumerate(frame_idxs):
                ch = mov[fi,yslice,xslice]
                if mean:
                    chunks += ch/len(frame_idxs)
                else:
                    chunks.append(ch)
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
