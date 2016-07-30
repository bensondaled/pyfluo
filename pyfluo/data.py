"""
TODO: 
    -add iterations on motion correction
    -in segmentation, take advantage of IPCA. but preprocessing (mocor, crop, rolling mean) must be done first, so need to implement the use of partial_fit
        - specifically: edit pcaica fxn to accept a generator, and then call next on it with partial fits until it's done
"""

from __future__ import print_function
import os, h5py, warnings, sys, re
import numpy as np, pandas as pd
import matplotlib.pyplot as pl

from .movies import Movie, play_mov
from .series import Series
from .roi import ROI
from .fluorescence import compute_dff, detect_transients
from .segmentation import pca_ica
from .motion import motion_correct, apply_motion_correction

class Data():
    """
    This class is currently experimental. Its goal is to facilitate working with a particular format of data storage that seems to be part of a common pipeline for imaging analysis.

    The intended workflow is as follows:
    (1) collect tifs from a microscope
    (2) send the tifs to a consolidated hdf5 file (using pyfluo.images.TiffGroup.to_hdf5)
    (3) open the resulting file through a Data object, which allows motion correction, segmentation, extractions, etc.
    """
    def __init__(self, data_file):
        
        self.data_file = data_file
            
        with h5py.File(self.data_file, 'r') as h:
            self.shape = h['data'].shape
        with pd.HDFStore(self.data_file) as h:
            self.n_files = len(h.info)
            self.info = h.info
            self.si_data = h.si_data
            self.i2c = h.i2c
            self.i2c = self.i2c.apply(pd.to_numeric, errors='ignore') #backwards compatability. can be deleted soon
            self._has_motion_correction = 'motion' in h
            if self._has_motion_correction:
                self.motion = h.motion
                try:
                    self.motion_params = h.get_storer('motion').attrs.params
                except:
                    self.motion_params = None
        self.Ts = self.info.iloc[0].Ts
        if not (self.info.Ts==self.Ts).all():
            warnings.warn('Sampling periods do not match. This class currently doesn\'t support this. Will replace Ts\'s with mean of Ts\'s.')
            print(self.info)
            self.info.Ts = np.mean(self.info.Ts)

        self.i2c.ix[:,'abs_frame_idx'] = self.i2c.apply(self._batch_framei, axis=1)

    def _batch_framei(self, row):
        return self.framei(row.frame_idx, row.file_idx)

    def __getitem__(self, idx):
        with h5py.File(self.data_file, 'r') as h:
            data = Movie(np.asarray(h['data'][idx]), Ts=self.Ts)

        # likely more cases to account for in future
        if isinstance(idx, tuple):
            mc_idx = idx[0]
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
        with h5py.File(self.data_file, 'r') as h:
            n_frames = len(h['data'])
        return n_frames

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
            h.create_dataset('templates', data=np.zeros((n_chunks+1,) + self.shape[1:]))
        
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

    def show(self):
        im = self.mean(axis=0)
        pl.imshow(im)
        return im

    def _apply_func(self, func, func_name, axis):
        """Applies arbitrary function to entire dataset in chunks
        Importantly, applies to chunks, then applies to result of that
        This works for functions like max, min, mean, but not all functions

        NOTE: technically mean is slightly wrong here for datasets where chunk size of generator isnt a factor of n frames. the last chunk will be nan-padded but yet its nanmean will be averaged with the rest of the chunks naive to that fact
        """
        ax_str = str(axis)
        if axis is None:
            ax_str = ''
        attr_str = '_{}_{}'.format(func_name, ax_str)
        with h5py.File(self.data_file) as f:
            if '_data_funcs' not in f:
                f.create_group('_data_funcs')
            if attr_str not in f['_data_funcs']:
                res = [func(chunk, axis=axis) for chunk in self.gen()]
                res = func(res, axis=0)
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
    def _next_roi_idx(self):
        latest_idx = self._latest_roi_idx
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

    def set_roi(self, roi):
        with h5py.File(self.data_file) as f:
            if 'roi' not in f:
                roigrp = f.create_group('roi')
            else:
                roigrp = f['roi']
            roigrp.create_dataset('roi{}'.format(self._next_roi_idx), data=np.asarray(roi))
   
    def get_roi(self, idx=None):
        if idx is None:
            idx = self._latest_roi_idx
        if idx is None:
            return None

        with h5py.File(self.data_file) as f:
            roigrp = f['roi']
            self._roi = ROI(roigrp['roi{}'.format(int(idx))])

        return self._roi

    def get_tr(self, idx=None, batch=6000, verbose=True):
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
                all_tr = []
                for b in range(0,len(self),batch):
                    sl = slice(b,min([len(self), b+batch]))
                    if verbose:
                        print ('Slice: {}-{}, total={}'.format(sl.start,sl.stop,len(self))); sys.stdout.flush()
                    tr = Movie(self[sl]).extract(roi)
                    all_tr.append(np.asarray(tr))
                self._tr = Series(np.concatenate(all_tr), Ts=self.Ts)
                grp.create_dataset(trname, data=np.asarray(self._tr))

        return self._tr
    
    def get_dff(self, idx=None, compute_dff_kwargs={}, recompute=False, verbose=True):
        if idx is None:
            idx = self._latest_roi_idx

        dffname = 'dff{}'.format(idx)

        with h5py.File(self.data_file) as f:
            grp = f['traces']

            if dffname in grp and not recompute:
                self._dff = Series(np.asarray(grp[dffname]), Ts=self.Ts)

            elif (dffname not in grp) or recompute:
                tr = self.get_tr(idx)
                if tr is None:
                    return None
                self._dff = compute_dff(tr, verbose=verbose, **compute_dff_kwargs)
                if dffname in grp:
                    del grp[dffname]
                grp.create_dataset(dffname, data=np.asarray(self._dff))

        if self._dff.isnull().values.any():
            warnings.warn('Null values zeroed out in DFF.')
            self._dff[self._dff.isnull()] = 0
        if np.isinf(self._dff).values.any():
            warnings.warn('Inf values zeroed out in DFF.')
            self._dff[np.isinf(self._dff)] = 0
        return self._dff
    
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
                grp.create_dataset(transname, data=np.asarray(self._transients))

        return self._transients

    def gen(self, chunk_size=1, n_frames=None, downsample=None, crop=False, enforce_chunk_size=False):
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
                dat = self[idx*chunk_size:]
                if enforce_chunk_size:
                    pad_size = chunk_size - len(dat)
                    dat = np.pad(dat, ((0,pad_size),(0,0),(0,0)), mode='constant', constant_values=(np.nan,))
            else:
                dat = self[idx*chunk_size:idx*chunk_size+chunk_size]

            if crop:
                pass
                #dat = dat[:,cr:-cr,cr:-cr]

            dat = dat.resample(downsample)

            yield dat

    def segment(self, gen_kwargs=dict(chunk_size=2700, n_frames=30000, downsample=3, crop=True), verbose=True, **pca_ica_kwargs):
        def dummy_gen():
            return self.gen(**gen_kwargs)

        if verbose:
            print('Segmenting. Generator: {} frames, chunk size {}, downsample {} --> {} chunks.'.format(gen_kwargs['n_frames'], gen_kwargs['chunk_size'], gen_kwargs['downsample'], np.ceil(gen_kwargs['n_frames']/gen_kwargs['chunk_size'])))
        comps = pca_ica(dummy_gen, **pca_ica_kwargs)
        with h5py.File(self.data_file) as h:
            if 'segmentation' not in h:
                grp = h.create_group('segmentation')
            else:
                grp = h['segmentation']
            ds = grp.create_dataset('segmentation{}'.format(self._next_segmentation_idx), data=comps)
            pca_ica_kwargs.update(gen_kwargs)
            ds.attrs.update(pca_ica_kwargs)

    def play(self, **kwargs):
        play_mov(self, generator_fxn='gen', **kwargs)
