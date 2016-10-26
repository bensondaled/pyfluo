from __future__ import print_function
import os, h5py, warnings, sys, re
import numpy as np, pandas as pd
import matplotlib.pyplot as pl
from skimage.morphology import erosion, dilation
from skimage.filters import gaussian
from skimage.exposure import equalize_adapthist
from scipy.ndimage import label

from .movies import Movie, play_mov
from .series import Series
from .roi import ROI, select_roi
from .fluorescence import compute_dff, detect_transients
from .segmentation import pca_ica
from .motion import motion_correct, apply_motion_correction
from .util import rolling_correlation
from .config import *

class Data():
    """
    This class is currently experimental. Its goal is to facilitate working with a particular format of data storage that seems to be part of a common pipeline for imaging analysis.

    The intended workflow is as follows:
    (1) collect tifs from a microscope
    (2) send the tifs to a consolidated hdf5 file (using pyfluo.images.TiffGroup.to_hdf5)
    (3) open the resulting file through a Data object, which allows motion correction, segmentation, extractions, etc.
    """
    def __init__(self, data_file):
        """
        """
        
        self.data_file = data_file
            
        with h5py.File(self.data_file, 'r') as h:
            if 'data' in h:
                self.shape = h['data'].shape
                self._data_chunk_size = h['data'].chunks
                self.batch_size = self._data_chunk_size[0]*250 # hard-coded for now, corresponds to 500 frames in the normal case
                self._has_data = True
            else:
                self.shape = None
                self._has_data = False
        with pd.HDFStore(self.data_file) as h:
            self.n_files = len(h.info)
            self.info = h.info
            self.si_data = h.si_data

            if self.shape is None:
                self.shape = (np.sum(self.info.n.values), self.info.y.values[0], self.info.x.values[0])

            if 'i2c' in h:
                self.i2c = h.i2c
                self.i2c = self.i2c.apply(pd.to_numeric, errors='ignore') #backwards compatability. can be deleted soon
                self.i2c.ix[:,'abs_frame_idx'] = self.i2c.apply(self._batch_framei, axis=1)
                try:
                    self.i2c.ix[:,'phase'] = (self.i2c.data-self.i2c.data.astype(int)).round(1)*10   
                    self.i2c.ix[:,'trial'] = self.i2c.data.astype(int)
                except:
                    pass
            self._has_motion_correction = 'motion' in h
            if self._has_motion_correction:
                self.motion = h.motion
                try:
                    self.motion_params = h.get_storer('motion').attrs.params
                except:
                    self.motion_params = None

                xv,yv = self.motion.x.values,self.motion.y.values
                shy,shx = self.shape[1:]
                self.motion_borders = pd.Series(dict(xmin=xv.max(), xmax=min(shx, shx+xv.min()), ymin=yv.max(), ymax=min(shy, shy+yv.min()))) #
        self.Ts = self.info.iloc[0].Ts
        if not (self.info.Ts==self.Ts).all():
            warnings.warn('Sampling periods do not match. This class currently doesn\'t support this. Will replace Ts\'s with mean of Ts\'s.')
            print(self.info)
            self.info.Ts = np.mean(self.info.Ts)

    def si_find(self, query):
        i = np.argwhere([query in k for k in self.si_data.keys()]).squeeze()
        i = np.atleast_1d(i)
        if len(i) == 0:
            return None
        keys = [self.si_data.keys()[idx] for idx in i]
        result = {k:self.si_data[k] for k in keys}
        return result

    def _batch_framei(self, row):
        return self.framei(row.frame_idx, row.file_idx)

    def __getitem__(self, idx):
        with h5py.File(self.data_file, 'r') as h:
            if 'data' in h:
                data = Movie(np.asarray(h['data'][idx]), Ts=self.Ts)
            else:
                return None

        # likely more cases to account for in future
        if isinstance(idx, tuple):
            mc_idx = idx[0]
            warnings.warn('x/y slicing was requested, but Data class will not properly motion correct such segments. Edges will be skewed.')
        else:
            mc_idx = idx

        if self._has_motion_correction:
            with pd.HDFStore(self.data_file) as h:
                motion_data = h.motion.iloc[mc_idx][['x','y']].values
            data = apply_motion_correction(data, motion_data, in_place=True, verbose=False)
        else:
            warnings.warn('Note that no motion correction data is present.')
        
        return data

    def __len__(self):
        # gives length of dataset, i.e. n_frames. See obj.n_files as well
        return self.shape[0]

    def __repr__(self):
        return repr(self.info)

    def framei(self, frame_idx, file_idx=None):
        """
        Converts between absolute frame index in whole dataset and relative file-frame coordinates.
        If only frame_idx given, assumes global and converts to relative.
        If both given, does opposite.
        """
        frame_idx = int(frame_idx)

        if file_idx is None:
            file_idx = self.info.n.cumsum().searchsorted(frame_idx+1)[0]
            frame_idx = frame_idx - np.append(0, self.info.n.cumsum().values)[file_idx] 
            return (frame_idx, file_idx)
        else:
            file_idx = int(file_idx)
            return np.append(0, self.info.n.cumsum().values)[file_idx] + frame_idx

    def motion_correct(self, chunk_size=None, mc_kwargs=dict(max_iters=10, shift_threshold=1.,), compute_kwargs=dict(max_shift=25, resample=3), overwrite=False, verbose=True):
        """
        Corrects motion, first locally in sliding chunks of full dataset, then globally across chunks

        Parameters
        ----------
        chunk_size : int
            number of frames to include in one local-correction chunk. If None, uses sizes of files from which data were derived
        """
        if overwrite:
            self._has_motion_correction = False

        if chunk_size is not None:
            n_chunks = int(np.ceil(len(self)/float(chunk_size)))
            slices = [slice(start, min(start+chunk_size, len(self))) for start in range(0, len(self), chunk_size)]
        elif chunk_size is None:
            n_chunks = self.n_files
            borders = np.append(0,self.info.n.cumsum().values)
            slices = [slice(b0,b1) for b0,b1 in zip(borders[:-1],borders[1:])]

        # check for existing datasets
        with pd.HDFStore(self.data_file) as h:
            if ('motion' in h) and not overwrite:
                return
            elif 'motion' in h and overwrite:
                h.remove('motion')
        with h5py.File(self.data_file) as h:
            if 'templates' in h:
                del h['templates']
        
        # create new datasets
        with h5py.File(self.data_file) as h:
            h.create_dataset('templates', data=np.zeros((n_chunks+1,) + self.shape[1:]), compression='lzf')
        
        # iterate through frame chunks, local corrections
        for idx,sli in enumerate(slices):
            sli_full = np.arange(sli.start, sli.stop)
            size = sli.stop-sli.start

            if verbose:
                print('Chunk {}/{}: frames {}:{}'.format(idx+1,n_chunks,sli.start,sli.stop)); sys.stdout.flush()
            chunk = Movie(self[sli])
            _, templ,vals = motion_correct(chunk, compute_kwargs=compute_kwargs, **mc_kwargs)

            # format result
            mot = pd.DataFrame(columns=['chunk','x','y','metric','x_local','y_local'], index=sli_full, dtype=float)
            mot[['x_local','y_local','metric']] = vals
            mot['chunk'] = idx
            with pd.HDFStore(self.data_file) as h:
                h.append('motion', mot)
            with h5py.File(self.data_file) as h:
                h['templates'][idx] = templ

        # global correction on template movie
        with h5py.File(self.data_file) as h:
            template_mov = Movie(np.asarray(h['templates'][:-1]))
        cka = compute_kwargs.copy()
        cka.update(resample=1)
        _,glob_template,glob_vals = motion_correct(template_mov, compute_kwargs=cka, **mc_kwargs)
        with h5py.File(self.data_file) as h:
            h['templates'][-1] = glob_template
        with pd.HDFStore(self.data_file) as h:
            mot = h.motion
            assert len(glob_vals)==n_chunks
            for ci,gv in enumerate(glob_vals[:,:-1]):
                xy = mot.ix[mot.chunk==ci, ['x_local','y_local']]
                mot.ix[mot.chunk==ci,['x','y']] = xy.values+gv
            # replace table with added values
            h.remove('motion')
            h.put('motion', mot)
            compute_kwargs.update(chunk_size=chunk_size)
            h.get_storer('motion').attrs.params = compute_kwargs

        self._has_motion_correction = True

    def project(self, show=False, equalize=False):
        im = self.mean(axis=0)
        if equalize:
            im = equalize_adapthist((im-im.min())/(im.max()-im.min()))
        if show:
            pl.imshow(im)
        return im

    def _apply_func(self, func, func_name, axis, agg_func=None):
        """Applies arbitrary function to entire dataset in chunks
        Importantly, applies to chunks, then applies to result of that
        This works for functions like max, min, mean, but not all functions

        NOTE: technically mean is slightly wrong here for datasets where chunk size of generator isnt a factor of n frames. the last chunk will be nan-padded but yet its nanmean will be averaged with the rest of the chunks naive to that fact
        """
        ax_str = str(axis)
        if axis is None:
            ax_str = ''
        if agg_func is None:
            agg_func = func
        attr_str = '_{}_{}'.format(func_name, ax_str)
        with h5py.File(self.data_file) as f:
            if '_data_funcs' not in f:
                f.create_group('_data_funcs')
            if attr_str not in f['_data_funcs']:
                res = [func(chunk, axis=axis) for chunk in self.gen(chunk_size=self.batch_size)]
                res = agg_func(res, axis=0)
                f['_data_funcs'].create_dataset(attr_str, data=res)
            else:
                res = np.asarray(f['_data_funcs'][attr_str])
        if isinstance(res, np.ndarray) and res.ndim==0:
            res = float(res)
        return res

    def max(self, axis=None):
        return self._apply_func(np.nanmax, axis=axis, func_name='max')
    def min(self, axis=None):
        return self._apply_func(np.nanmin, axis=axis, func_name='min')
    def mean(self, axis=None):
        return self._apply_func(np.nanmean, axis=axis, func_name='mean')
    def std(self, axis=None):
        # confirmed working, sept 22 2016
        def _std(x, mean=self.mean(axis=axis), axis=None):
            return np.nansum((x-mean)**2, axis=axis), x.shape[axis]
        def _std_combine(x, axis=None):
            n = np.nansum([i[1] for i in x])
            x = [i[0] for i in x]
            return np.sqrt(np.sum(x, axis=0) / (n-1))
        return self._apply_func(_std, axis=axis, func_name='std', agg_func=_std_combine)

    def __max__(self):
        return self.max()
    def __min__(self):
        return self.min()

    @property
    def _latest_roi_idx(self):
        with h5py.File(self.data_file) as f:
            if 'roi' not in f:
                return None
            keys = list(f['roi'].keys())
            if len(keys)==0:
                return None
            matches = [re.match('roi(\d)', s) for s in keys]
            idxs = [int(m.groups()[0]) for m in matches if m]
            return max(idxs)
    
    @property
    def _latest_r_idx(self):
        with h5py.File(self.data_file) as f:
            if 'r' not in f:
                return None
            keys = list(f['r'].keys())
            if len(keys)==0:
                return None
            matches = [re.match('r(\d)', s) for s in keys]
            idxs = [int(m.groups()[0]) for m in matches if m]
            return max(idxs)

    @property
    def _next_roi_idx(self):
        latest_idx = self._latest_roi_idx
        if latest_idx is None:
            nex_idx = 0
        else:
            nex_idx = latest_idx+1
        return nex_idx
    
    @property
    def _next_r_idx(self):
        latest_idx = self._latest_r_idx
        if latest_idx is None:
            nex_idx = 0
        else:
            nex_idx = latest_idx+1
        return nex_idx
    
    @property
    def _latest_segmentation_idx(self):
        with h5py.File(self.data_file) as f:
            if 'segmentation' not in f:
                return None
            keys = list(f['segmentation'].keys())
            if len(keys)==0:
                return None
            matches = [re.match('segmentation(\d)', s) for s in keys]
            idxs = [int(m.groups()[0]) for m in matches if m]
            return max(idxs)

    @property
    def _next_segmentation_idx(self):
        latest_idx = self._latest_segmentation_idx
        if latest_idx is None:
            nex_idx = 0
        else:
            nex_idx = latest_idx+1
        return nex_idx

    def get_segmentation(self, idx=None):
        if idx is None:
            idx = self._latest_segmentation_idx
        if idx is None:
            return None

        with h5py.File(self.data_file) as f:
            seggrp = f['segmentation']
            _segmentation = np.asarray(seggrp['segmentation{}'.format(int(idx))])

        return _segmentation

    def set_roi(self, roi):
        with h5py.File(self.data_file) as f:
            if 'roi' not in f:
                roigrp = f.create_group('roi')
            else:
                roigrp = f['roi']
            roigrp.create_dataset('roi{}'.format(self._next_roi_idx), data=np.asarray(roi), compression='lzf')
    
    def get_roi(self, idx=None):
        if idx is None:
            idx = self._latest_roi_idx
        if idx is None:
            return None

        with h5py.File(self.data_file) as f:
            roigrp = f['roi']
            _roi = ROI(roigrp['roi{}'.format(int(idx))])

        return _roi
    
    def get_r(self, idx=None):
        if self._latest_r_idx is None:
            return None

        if idx is None:
            idx = self._latest_r_idx
        if idx is None:
            return None

        with h5py.File(self.data_file) as f:
            rgrp = f['r']
            _r = np.asarray(rgrp['r{}'.format(int(idx))])

        return _r
    
    def get_maxmov(self, chunk_size=150, resample=3, redo=False, enforce_datatype=np.int16):
        with h5py.File(self.data_file) as f:
            if 'maxmov' in f and not redo:
                    _mm = Movie(np.asarray(f['maxmov']), Ts=f['maxmov'].attrs['Ts'])
            else:
                if not self._has_data:
                    warnings.warn('Data not stored in this file, so cannot make example.')
                    return
                if 'maxmov' in f and redo:
                    del f['maxmov']
               
                gen = self.gen(chunk_size=chunk_size, downsample=resample, enforce_chunk_size=True)
                data = np.array([np.nanmax(g-np.nanmean(g,axis=0), axis=0) for g in gen])
                _mm = Movie(data, Ts=self.Ts*chunk_size)
                ds = f.create_dataset('maxmov', data=_mm, compression='lzf')
                f['maxmov'].attrs['Ts'] = _mm.Ts
                f['maxmov'].attrs['chunk_size'] = chunk_size
                f['maxmov'].attrs['resample'] = resample

        if enforce_datatype is not None:
            _mm = _mm.astype(enforce_datatype)
        return _mm
    
    def get_example(self, slices=None, resample=3, redo=False, enforce_datatype=np.int16):
        # slice is specified only first time, then becomes meaningless once example is extracted; unless redo is used
        with h5py.File(self.data_file) as f:
            if 'example' in f and not redo:
                _example = Movie(np.asarray(f['example']), Ts=f['example'].attrs['Ts'])
            else:
                if 'example' in f and redo:
                    del f['example']
                if not self._has_data:
                    warnings.warn('Data not stored in this file, so cannot make example.')
                    return
                if slices is None:
                    sub_movie_size = 3000
                    n_sub_movies = 3
                    Ts = self.Ts
                    quart = len(self)//(n_sub_movies+1)
                    slices = [slice(quart*i-sub_movie_size//2,quart*i+sub_movie_size//2) for i in range(1,n_sub_movies+1)]
                else:
                    if all([isinstance(s, slice) for s in slices]) and all([s.step==slices[0].step for s in slices]):
                        step = slices[0].step or 1
                    else:
                        step = 1
                    Ts = self.Ts * step
                data = np.concatenate([self[s] for s in slices])
                _example = Movie(data, Ts=Ts)
                _example = _example.resample(resample)
                ds = f.create_dataset('example', data=_example, compression='lzf')
                f['example'].attrs['Ts'] = _example.Ts
                f['example'].attrs['slices'] = str(slices)

        if enforce_datatype is not None:
            _example = _example.astype(enforce_datatype)
        return _example

    def get_tr(self, idx=None, batch=None, verbose=True):
        if idx is None:
            idx = self._latest_roi_idx

        roi = self.get_roi(idx)
        if roi is None:
            return None
        trname = 'tr{}'.format(idx)

        with h5py.File(self.data_file) as f:
            if 'traces' not in f:
                grp = f.create_group('traces')
            elif 'traces' in f:
                grp = f['traces']

            if trname in grp:
                self._tr = Series(np.asarray(grp[trname]), Ts=self.Ts)
            elif trname not in grp:
                if verbose:
                    print ('Extracting traces...'); sys.stdout.flush()
                if batch is None:
                    batch = self.batch_size

                all_tr = []
                for b in range(0,len(self),batch):
                    sl = slice(b,min([len(self), b+batch]))
                    if verbose:
                        print ('Slice: {}-{}, total={}'.format(sl.start,sl.stop,len(self))); sys.stdout.flush()
                    tr = self[sl].extract(roi)
                    all_tr.append(np.asarray(tr))
                self._tr = Series(np.concatenate(all_tr), Ts=self.Ts)
                grp.create_dataset(trname, data=np.asarray(self._tr), compression='lzf')

        return self._tr
    
    def get_dff(self, idx=None, compute_dff_kwargs=dict(window_size=12.), recompute=False, verbose=True):
        if idx is None:
            idx = self._latest_roi_idx

        dffname = 'dff{}'.format(idx)

        with h5py.File(self.data_file) as f:
            grp = f['traces']

            if dffname in grp and not recompute:
                self._dff = Series(np.asarray(grp[dffname]), Ts=self.Ts)

            elif (dffname not in grp) or recompute:
                tr = self.get_tr(idx)
                if verbose:
                    print('Computing DFF...'); sys.stdout.flush()
                if tr is None:
                    return None
                self._dff = compute_dff(tr, verbose=verbose, **compute_dff_kwargs)
                if dffname in grp:
                    del grp[dffname]
                grp.create_dataset(dffname, data=np.asarray(self._dff), compression='lzf')
                grp[dffname].attrs.update(compute_dff_kwargs)

        if self._dff.isnull().values.any():
            warnings.warn('Null values zeroed out in DFF.')
            self._dff[self._dff.isnull()] = 0
        if np.isinf(self._dff).values.any():
            warnings.warn('Inf values zeroed out in DFF.')
            self._dff[np.isinf(self._dff)] = 0
        return self._dff
    
    def get_rollcor(self, idx=None, window=.300, recompute=False, verbose=True):
        # window in seconds
        if idx is None:
            idx = self._latest_roi_idx

        rollcorname = 'rollcor{}'.format(idx)

        with h5py.File(self.data_file) as f:
            grp = f['traces']

            if rollcorname in grp and not recompute:
                _rollcor = Series(np.asarray(grp[rollcorname]), Ts=self.Ts)

            elif (rollcorname not in grp) or recompute:
                dff = self.get_dff(idx)
                if verbose:
                    print('Computing rollcor...'); sys.stdout.flush()
                if dff is None:
                    return None

                corwin_ = int(window/self.Ts)
                _rollcor = rolling_correlation(dff.values, corwin_, verbose=verbose)

                if rollcorname in grp:
                    del grp[rollcorname]

                grp.create_dataset(rollcorname, data=np.asarray(_rollcor), compression='lzf')
                grp[rollcorname].attrs.update(window=window)

        return Series(_rollcor, Ts=self.Ts)
    
    def get_transients(self, idx=None, detect_transients_kwargs={}):
        if idx is None:
            idx = self._latest_roi_idx

        transname = 'transients{}'.format(idx)

        with h5py.File(self.data_file) as f:
            grp = f['traces']

            if transname in grp:
                self._transients = Series(np.asarray(grp[transname]), Ts=self.Ts)

            elif transname not in grp:
                dff = self.get_dff(idx)
                if dff is None:
                    return None
                self._transients = detect_transients(dff, **detect_transients_kwargs)
                grp.create_dataset(transname, data=np.asarray(self._transients), compression='lzf')

        return self._transients

    def gen(self, chunk_size=1, n_frames=None, downsample=None, crop=False, enforce_chunk_size=False, return_idx=False):
        """Data in the form of a generator that motion corrections, crops, applies rolling_mean, etc

        chunk_size : number of frames to include in one chunk *before* downsampling
        n_frames : sum of number of total raw frames included in all yields from this iterator
        enforce_chunk_size : bool, if True, nan-pads the last slice if necessary to make equal chunk size

        yielded items will be of length chunk_size//downsample
        """
        if n_frames is None:
            n_frames = len(self)
        if crop:
            pass #TODO: find max actual motion and cut by that
        if downsample in [None,False]:
            downsample = 1

        nchunks = n_frames//chunk_size
        remainder = n_frames%chunk_size

        for idx in range(nchunks+int(remainder>0)):
            if idx == nchunks:
                _i = slice(idx*chunk_size, None)
                dat = self[_i]
                if enforce_chunk_size:
                    pad_size = chunk_size - len(dat)
                    dat = Movie(np.pad(dat, ((0,pad_size),(0,0),(0,0)), mode='constant', constant_values=(np.nan,)), Ts=self.Ts)
            else:
                _i = slice(idx*chunk_size,idx*chunk_size+chunk_size)
                dat = self[_i]

            if crop:
                pass
                #dat = dat[:,cr:-cr,cr:-cr]

            dat = dat.resample(downsample)

            if return_idx:
                yield (dat, _i)
            else:
                yield dat

    def segment(self, gen_kwargs=dict(chunk_size=3000, n_frames=None, downsample=3), verbose=True, **pca_ica_kwargs):
        def dummy_gen():
            return self.gen(**gen_kwargs)

        if verbose:
            nume = gen_kwargs['n_frames'] or len(self)
            nc = np.ceil(nume/gen_kwargs['chunk_size'])
            print('Segmenting. Generator: {} frames, chunk size {}, downsample {} --> {} chunks.'.format(gen_kwargs['n_frames'], gen_kwargs['chunk_size'], gen_kwargs['downsample'], nc))
        comps = pca_ica(dummy_gen, **pca_ica_kwargs)
        with h5py.File(self.data_file) as h:
            if 'segmentation' not in h:
                grp = h.create_group('segmentation')
            else:
                grp = h['segmentation']
            ds = grp.create_dataset('segmentation{}'.format(self._next_segmentation_idx), data=comps, compression='lzf')
            pca_ica_kwargs.update(gen_kwargs)
            ds.attrs.update(pca_ica_kwargs)

    def play(self, **kwargs):
        play_mov(self, generator_fxn='gen', **kwargs)

    def export(self, out_filename, include_data=False, overwrite=False):
        if os.path.isdir(out_filename):
            out_filename = os.path.join(out_filename, os.path.split(self.data_file)[-1])
        elif not out_filename.endswith('.h5'):
            out_filename += '.h5'

        if os.path.exists(out_filename) and not overwrite:
            print ('File exists and overwrite=False. Returning.')
            return

        with pd.HDFStore(self.data_file) as infile:
            handle = infile.copy(out_filename, overwrite=False)
        handle.close()
        with h5py.File(out_filename) as outfile, h5py.File(self.data_file) as infile:
            for key in infile:
                if key in outfile:
                    continue
                if include_data==False and key=='data':
                    continue
                print('Copying "{}"'.format(key))
                infile.copy(key, outfile, expand_soft=True, expand_refs=True, expand_external=True)
    def import_file(self, filename, backup=True):
        """
        NOTE: overwrites all attributes except data
        """
        if backup:
            self.export('{}_backup.h5'.format(os.path.splitext(self.data_file)[0]))
        with h5py.File(filename) as infile, h5py.File(self.data_file) as datafile:
            for key in infile:
                if key in datafile:
                    del datafile[key]
                infile.copy(key, datafile)

    def refine_roi(self, roi_idx=None, include_rejections=False, verbose=True):
        # Pearson's r over hdf dataset
        if roi_idx is None:
            roi_idx = self._latest_roi_idx

        roi = self.get_roi(roi_idx)
        rs = self.get_r(roi_idx)

        if rs is None:

            sig = self.get_tr(roi_idx).values

            dmean = self.mean(axis=0)
            smean = np.mean(sig, axis=0)

            assert sig.shape[0]==self.shape[0]
            if sig.ndim == 1:
                sig = sig[:,None]

            gen = self.gen(chunk_size=self.batch_size, return_idx=True)

            sums,ns = [[] for i in range(sig.shape[1])],[]
            for gi,idx in gen:
                if verbose:
                    print('Chunk {}'.format(idx)); sys.stdout.flush()
                sigi = sig[idx] - smean
                di = gi - dmean

                assert len(di) == len(sigi)

                for subsig in range(sig.shape[1]):
                    sums[subsig].append(np.nansum(sigi[:,subsig][:,None,None] * di, axis=0))
                ns.append(len(di))

            sums = np.array([np.sum(s, axis=0) for s in sums])
            cov = sums / (np.sum(ns)-1)
            
            rs = [c / (self.std(axis=0)*sd) for c,sd in zip(cov,np.std(sig,axis=0))]

            # store rs
            with h5py.File(self.data_file) as f:
                if 'r' not in f:
                    rgrp = f.create_group('r')
                else:
                    rgrp = f['r']
                rgrp.create_dataset('r{}'.format(roi_idx), data=np.asarray(rs), compression='lzf')

        # binarize correlation images
        rs -= np.median(rs, axis=0)
        rs = np.array([dilation(erosion(gaussian(ri,.2))) for ri in rs])
        orig = np.array([r.flat[ro.flat==True] for r,ro in zip(rs,roi)])
        omeans = np.array([np.mean(o) for o in orig])
        ostd = np.array([np.std(o) for o in orig])
        threshs = omeans - 1.*ostd
        masks = np.array([r>=th for th,r in zip(threshs,rs)])
        masks = np.array([dilation(erosion(gaussian(m,.2))) for m in masks])

        # find best match, using overlap-based metric
        masks_new = []
        rejects = []
        for i,m,r in zip(range(len(roi)),masks,roi):
            l,nl = label(m)
            overlap = [np.sum((l==i) & r) for i in np.arange(1,nl+1)]
            iwin = np.argmax(overlap) + 1
            new = l==iwin
            # verify that area of roi hasn't inflated/shrunk too much, else keep the manual one
            if new.sum()>r.sum()*3.5 or new.sum()<r.sum()*.5:
                rejects.append(i)
                if include_rejections:
                    masks_new.append(r) # original
            else:
                masks_new.append(new) # new

        self.set_roi(np.asarray(masks_new))
        print('Refinement of roi{} complete. {}/{} refinements rejected:\n{}'.format(roi_idx,len(rejects),len(roi),rejects)); sys.stdout.flush()
        #self.get_tr() # extract new traces

    def select_roi(self):
        mm = self.get_maxmov()
        existing = None
        fig = pl.figure('ROI Selection')
        for idx,m in enumerate(mm):
            ax = fig.add_subplot(111)
            ax.set_title('{}/{}'.format(idx+1, len(mm)))
            roi = select_roi(m, ax=ax, existing=existing)
            existing = roi
            np.save('_roitmp.npy', np.asarray(existing))
            fig.clear()
        roi = np.load('_roitmp.npy')
        self.set_roi(roi)
        os.remove('_roitmp.npy')

    def roi_subset(self, keep, roi_idx=None):
        """
        Generate a new roi made of a subset of a current roi
        keep : boolean array of length of roi at roi_idx
        """
        if (not isinstance(keep, np.ndarray)) or (keep.dtype != bool):
            raise Exception('Must supply boolean numpy array as "keep" parameter.')

        roi = self.get_roi(roi_idx)
        if roi is None:
            return
        tr = self.get_tr(roi_idx)
        dff = self.get_dff(roi_idx)

        roi = roi[keep]
        tr = tr.iloc[:,keep]
        dff = dff.iloc[:,keep]

        # set new roi
        self.set_roi(roi)
       
        # set new traces
        idx = self._latest_roi_idx
        trname = 'tr{}'.format(idx)
        dffname = 'dff{}'.format(idx)

        with h5py.File(self.data_file) as f:
            grp = f['traces']
            grp.create_dataset(trname, data=np.asarray(tr), compression='lzf')
            grp.create_dataset(dffname, data=np.asarray(dff), compression='lzf')
            grp[dffname].attrs.update(origin='roi_idx_{}'.format(roi_idx), subset=keep.tolist())
