import os, h5py, warnings, sys, re,json
import numpy as np, pandas as pd
import matplotlib.pyplot as pl
from skimage.morphology import erosion, dilation
from skimage.filters import gaussian
from skimage.exposure import equalize_adapthist
from scipy.ndimage import label

from .movies import Movie, play_mov
from .series import Series
from .roi import ROI, ROIView
from .fluorescence import compute_dff, detect_transients
from .segmentation import pca_ica, process_roi
from .motion import motion_correct, apply_motion_correction
from .util import rolling_correlation, ProgressBar
from .oasis import deconvolve
from .config import *

class Data():
    FOV = 1200
    """
    The goal of this class is to facilitate working with a particular format of data storage that seems to be part of a common pipeline for imaging analysis.

    The intended workflow is as follows:
    (1) collect tifs from a microscope
    (2) send the tifs to a consolidated hdf5 file (using pyfluo.images.TiffGroup.to_hdf5)
    (3) open the resulting file through a Data object, which allows motion correction, segmentation, extractions, etc.
    """
    def __init__(self, data_file):
        """Initialize a Data object

        Parameters
        ----------
        data_file : str
            path to hdf-5 file storing dataset
            the format of this file follows the pyfluo convention, handled automatically, for example, by pf.TiffGroup.to_hdf5
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
                self.batch_size = None
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

    @property
    def pixels_per_micron(self):
        y,x = self.shape[1:]
        range_x = self.si_data['scanimage.SI.hRoiManager.scanAngleMultiplierFast']
        range_y = self.si_data['scanimage.SI.hRoiManager.scanAngleMultiplierSlow']
        zoom = self.si_data['scanimage.SI.hRoiManager.scanZoomFactor']
        fov_microns = float(self.FOV) / zoom
        x_microns = float(fov_microns) * range_x
        y_microns = float(fov_microns) * range_y
        ppm_x = float(x) / x_microns
        ppm_y = float(y) / y_microns
        assert ppm_x == ppm_y, 'Pixels per micron seem strange:\ny x = {} {}\nzoom = {}\nrange y x = {} {}'.format(y,x,zoom,range_y,range_x)
        return ppm_x

    def si_find(self, query):
        """Retrieve scanimage data from dataset

        Parameters
        ----------
        query : str
            any substring of the search key from the scanimage parameter dictionary

        Returns
        -------
        result : dict
            sub-dictionary of scanimage data dictionary, where keys contain query
        """
        i = np.argwhere([query.lower() in k.lower() or query.lower() in str(self.si_data[k]).lower() for k in self.si_data.keys()]).squeeze()
        i = np.atleast_1d(i)
        if len(i) == 0:
            return None
        keys = [self.si_data.keys()[idx] for idx in i]
        result = {k:self.si_data[k] for k in keys}
        if 'zoom' in query.lower():
            result.update(self.si_find('scanAngleMultiplier'))
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
        """Convert between absolute frame index in whole dataset and relative file-frame coordinates

        frame_idx : int
            if supplied alone, taken to be a global index and returns the corresponding relative index
        file_idx : int
            if supplied in conjunction with frame_idx, taken to be a relative file-frame coordinate and return the corresponding absolute frame index

        Returns
        -------
        idx : int / tuple
            converted frame index, see parameters for details
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
        """Correct motion, first locally in sliding chunks of full dataset, then globally across chunks

        Parameters
        ----------
        chunk_size : int
            number of frames to include in one local-correction chunk. If None, uses sizes of files from which data were derived

        Motion correction data are stored in the data file, and become available as obj.motion after this is complete.
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
                print('Chunk {}/{}: frames {}:{}'.format(idx+1,n_chunks,sli.start,sli.stop))
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
        with pd.HDFStore(self.data_file) as h:
            self.motion = h.motion

    def project(self, show=False, equalize=False, **kwargs):
        """Mean-project the dataset over the time (0th) axis

        Parameters
        ----------
        show : bool
            display the projected image
        equalize : bool
            normalize and equalize the histogram of the projection
        kwargs : dict
            passed to matplotlib.imshow

        Returns
        -------
        im : np.ndarray
            projected data, of dimensions (data_y_size, data_x_size)
        """
        kwargs['cmap'] = kwargs.get('cmap', pl.cm.Greys_r)

        im = self.mean(axis=0)
        if equalize:
            im = equalize_adapthist((im-im.min())/(im.max()-im.min()))
        if show:
            pl.imshow(im, **kwargs)
        return im

    def _apply_func(self, func, func_name, axis, agg_func=None, verbose=False):
        """Applies arbitrary function/s to entire dataset in chunks
        Importantly, applies to chunks, then applies to result of that
        This works for functions like max, min, mean, but not all functions

        NOTE: technically mean is slightly wrong here for datasets where chunk size of generator isnt a factor of n frames. the last chunk will be nan-padded but yet its nanmean will be averaged with the rest of the chunks naive to that fact

        Parameters
        ----------
        func : def
            function to apply, example np.mean; or list thereof
        func_name : str
            name corresponding to this function, ex. 'mean'; or list thereof
        axis : int
            axis of dataset along which to apply function; or list thereof
        agg_func : def
            function applied to the batches over which *func* was applied. If None, defaults to *func* itself; or list thereof

        Returns
        -------
        Result of running the supplied function/s over the entire dataset
        """

        # convert singles into lists
        if not isinstance(func, PF_list_types):
            func = [func]
        if not isinstance(func_name, PF_list_types):
            func_name = [func_name]
        if not isinstance(axis, PF_list_types):
            axis = [axis]
        if agg_func is None:
            agg_func = [None for _ in func]
        if not isinstance(agg_func, PF_list_types):
            agg_func = [agg_func]
        
        # require dataset
        with h5py.File(self.data_file) as f:
            if '_data_funcs' not in f:
                f.create_group('_data_funcs')

        # check which supplied functions need to be computed
        todo = []
        for funcn,fxn,afxn in zip(func_name,func,agg_func):
            if afxn is None:
                afxn = fxn
            for ax in axis:
                ax_str = str(ax) if ax is not None else ''
                attr_str = '_{}_{}'.format(funcn, ax_str)
                with h5py.File(self.data_file) as f:
                    if attr_str not in f['_data_funcs']:
                        todo.append( (funcn, fxn, afxn, ax) ) # name, func, aggfunc, axis

        if verbose:
            print('Will compute {}'.format(todo))

        # compute new ones
        if len(todo) > 0:
            results = [[] for _ in todo]
            counter = 0
            for chunk in self.gen(chunk_size=self.batch_size):
                counter += 1
                if verbose:
                    print('Chunk number {}'.format(counter))
                for idx,(fn,fxn,afxn,ax) in enumerate(todo):
                    res = fxn(chunk, axis=ax)
                    results[idx].append(res)
            results = [afxn(res, axis=ax) for res,(fn,fxn,afxn,ax) in zip(results,todo)]

            # store results
            with h5py.File(self.data_file) as f:
                for res,(fn,fxn,afxn,ax) in zip(results,todo):
                    ax_str = str(ax) if ax is not None else ''
                    attr_str = '_{}_{}'.format(fn, ax_str)
                    f['_data_funcs'].create_dataset(attr_str, data=res)
        
        # retrieve all desired results
        to_return = []
        for fn,fxn,afxn,ax in zip(func_name, func, agg_func, axis):
            ax_str = str(ax) if ax is not None else ''
            attr_str = '_{}_{}'.format(fn, ax_str)
            with h5py.File(self.data_file) as f:
                ds = np.asarray(f['_data_funcs'][attr_str])
            if isinstance(ds, np.ndarray) and ds.ndim==0:
                ds = float(ds)
            to_return.append(ds)

        # un-nest single requests
        if len(to_return)==1:
            to_return = to_return[0]

        return to_return

    def _all_basic_funcs(self, verbose=True):
        """Compute in batch the max, min, mean, std of the dataset (std is done separately b/c it depends on mean)
        """
        fxns = [np.nanmax, np.nanmin, np.nanmean]
        names = ['max','min','mean']
        axs = [None, 0] 
        _ = self._apply_func(fxns, func_name=names, axis=axs, verbose=verbose)

        _ = self.std(axis=0)

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
    def _latest_ct_idx(self):
        with h5py.File(self.data_file) as f:
            if 'camera_traces' not in f:
                return None
            keys = list(f['camera_traces'].keys())
            if len(keys)==0:
                return None
            matches = [re.match('camera_traces(\d)', s) for s in keys]
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
    def _next_ct_idx(self):
        latest_idx = self._latest_ct_idx
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
        """Assign an ROI to this dataset

        Parameters
        ----------
        roi : pyfluo.ROI
            new roi to assign, will be added incrementally onto the roi dataset in the data file
        """
        with h5py.File(self.data_file) as f:
            if 'roi' not in f:
                roigrp = f.create_group('roi')
            else:
                roigrp = f['roi']
            roigrp.create_dataset('roi{}'.format(self._next_roi_idx), data=np.asarray(roi), compression='lzf')
    
    def get_roi(self, idx=None):
        """Retrieve an ROI from the dataset

        Parameters
        ----------
        idx : int
            idx of roi to retrieve. If None, defaults to highest index available (most recently added)
        """
        if idx is None:
            idx = self._latest_roi_idx
        if idx is None:
            return None

        with h5py.File(self.data_file) as f:
            roigrp = f['roi']
            _roi = ROI(roigrp['roi{}'.format(int(idx))])

        return _roi
    
    def get_r(self, idx=None):
        """Retrieve a correlation matrix (R) from the dataset

        Parameters
        ----------
        idx : int
            idx of R to retrieve. If None, defaults to highest index available (most recently added)
        """
        if self._latest_r_idx is None:
            return None

        if idx is None:
            idx = self._latest_r_idx
        if idx is None:
            return None

        with h5py.File(self.data_file) as f:
            rgrp = f['r']
            if 'r{}'.format(int(idx)) not in rgrp:
                return None
            _r = np.asarray(rgrp['r{}'.format(int(idx))])

        return _r
    
    def get_camera_trace(self, camera_idx, data_file=None, verbose=True):
        """Retrieve stored camera traces from the dataset
        Or compute them using stored ROI and video file at given destination

        """
        field_name = 'camera{}_trace'.format(camera_idx)
        field_name_t = 'camera{}_time'.format(camera_idx)
        with h5py.File(self.data_file) as f:
            if 'cameras' not in f:
                return None
            cgrp = f['cameras']
            
            if field_name in cgrp:
                trace = np.asarray(cgrp[field_name])
                timestamps = np.asarray(cgrp[field_name_t])
                Ts = np.mean(np.diff(timestamps, axis=0)).mean()
                trace = Series(trace, Ts=Ts)

            else:
                if data_file is None:
                    warnings.warn('No traces stored and no data file supplied.')
                    return None

                roi = self.get_camera_roi(camera_idx)
                if roi is None:
                    warnings.warn('Could not generate trace because no ROI is stored.')
                    return None

                if data_file is None or not os.path.exists(data_file):
                    raise Exception('Requested data file {} was not found.'.format(data_file))

                dname = 'mov{}'.format(camera_idx)
                tname = 'ts{}'.format(camera_idx)

                # load in the behavior movie here, calling it mov
                # and ts should be defined as the timestamp values of the movie
                with h5py.File(data_file) as movfile:

                    mov = movfile[dname]
                    ts = np.asarray(movfile[tname])

                    chunk_size = 3000
                    trs = []
                    for i in np.arange(np.ceil(float(len(mov))/chunk_size)):
                        i0 = int(i*chunk_size)
                        i1 = int(i*chunk_size + chunk_size)
                        if i1 > len(mov):
                            i1 = int(len(mov))
                        if verbose:
                            print('Chunk {}: {} - {} / {}'.format(i, i0, i1, len(mov)))
                        submov = Movie(np.asarray(mov[i0:i1]))
                        subtr = submov.extract(roi)
                        trs.append(subtr)

                    trace = np.concatenate(trs)
                    Ts = np.mean(np.diff(ts, axis=0)).mean()
                    timestamps = ts
                trace = Series(trace, Ts=Ts)
                ds = cgrp.create_dataset(field_name, data=np.asarray(trace), compression='lzf')
                f['cameras'][field_name].attrs['Ts'] = Ts
                cgrp.create_dataset(field_name_t, data=np.asarray(timestamps), compression='lzf')

        ret = pd.DataFrame(trace, index=timestamps[:,0]) # note this hardcoded decision
        ret.Ts = Ts
        return ret

    def set_camera_roi(self, roi, camera_idx, overwrite=False):
        field_name = 'camera{}_roi'.format(camera_idx)
        tr_field_name = 'camera{}_trace'.format(camera_idx)
        tr_field_name_t = 'camera{}_time'.format(camera_idx)
        with h5py.File(self.data_file) as f:
            if 'cameras' not in f:
                warnings.warn('Camera not present, so cannot set roi.')
                return None
            grp = f['cameras']

            if field_name in grp and overwrite:
                del grp[field_name]
                ctname = 'camera{}_trace'.format(camera_idx)
                if ctname in grp:
                    del grp[ctname]
            
                if tr_field_name in grp:
                    del grp[tr_field_name]
                if tr_field_name_t in grp:
                    del grp[tr_field_name_t]

            elif field_name in grp and not overwrite:
                warnings.warn('Not setting ROI because roi exists and overwrite=False.')
                return

            grp.create_dataset(field_name, data=roi)
    
    def get_camera_roi(self, camera_idx):
        field_name = 'camera{}_roi'.format(camera_idx)
        with h5py.File(self.data_file) as f:
            if 'cameras' not in f:
                raise Exception('Camera not present, no ROI present.')
            grp = f['cameras']

            if field_name not in grp:
                warnings.warn('ROI not found for requested camera.')
                return None

            roi = np.asarray(grp[field_name])
        return ROI(roi)

    def get_camera_example(self, camera_idx, data_file=None, slices=None, redo=False, return_timestamps=True):
        """
        Get saved camera example for cam0 or cam1
        If not present, make it using data in data_file, expected to have mov0, mov1, ts0, and ts1
        """
        if not isinstance(camera_idx, int):
            raise Exception('camera_idx must be an integer')

        field_name = 'camera{}_example'.format(camera_idx)
        with h5py.File(self.data_file) as f:
            grp = f.require_group('cameras')

            if field_name in grp and not redo:
                _example = Movie(np.asarray(grp[field_name]), Ts=grp[field_name].attrs['Ts'])
                timestamps = grp[field_name].attrs['time']
            else:
                if data_file is None or not os.path.exists(data_file):
                    raise Exception('Requested data file {} was not found.'.format(data_file))

                dname = 'mov{}'.format(camera_idx)
                tname = 'ts{}'.format(camera_idx)

                # load in the behavior movie here, calling it mov
                # and ts should be defined as the timestamp values of the movie
                with h5py.File(data_file) as movfile:

                    mov = movfile[dname]
                    ts = np.asarray(movfile[tname])

                    if field_name in grp and redo:
                        del grp[field_name]
                    if slices is None:
                        sub_movie_size = 1000
                        n_sub_movies = 2
                        quart = len(mov)//(n_sub_movies+1)
                        slices = [slice(quart*i-sub_movie_size//2,quart*i+sub_movie_size//2) for i in range(1,n_sub_movies+1)]
                    else:
                        warnings.warn('Custom slices not yet error-checked; use with caution.')

                    Ts = np.mean(np.diff(ts, axis=0)).mean()
                    data = np.concatenate([mov[s] for s in slices])
                    timestamps = np.concatenate([ts[s] for s in slices])
                _example = Movie(data, Ts=Ts)
                ds = grp.create_dataset(field_name, data=_example, compression='lzf')
                f['cameras'][field_name].attrs['Ts'] = Ts
                f['cameras'][field_name].attrs['time'] = timestamps
                f['cameras'][field_name].attrs['slices'] = str(slices)

        if return_timestamps:
            return _example, timestamps
        else:
            return _example
    
    def get_maxmov(self, chunk_size=2, resample=3, redo=False, enforce_datatype=np.int16):
        """Generate or retrieve a compressed version of the dataset using a rolling maximum method

        Parameters
        ----------
        chunk_size : int
            number of *seconds* per chunk
        resample : int
            factor by which to downsample each chunk (performed by data.gen)
        redo : bool
            if True, deletes existing stored maxmov and generates a new one
        enforce_datatype : numpy dtype
            numpy datatype into which to cast resulting movie

        Returns
        -------
        maxmov : pyfluo.Movie
        """
        chunk_size = int(np.round(chunk_size / self.Ts))

        with h5py.File(self.data_file) as f:
            if 'maxmov' in f and not redo:
                    _mm = Movie(np.asarray(f['maxmov']), Ts=f['maxmov'].attrs['Ts'])
            else:
                if not self._has_data:
                    warnings.warn('Data not stored in this file, so cannot make maxmov.')
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
    
    def get_meanmov(self, slices=None, name=None, attrs={}, verbose=True):
        """Generate or retrieve a meanmov, i.e. a movie comprised of multiple time slices averaged together

        Parameters
        ----------
        slices : list-like
            list of slice objects corresponding to the slices of the movie to be included, must all be same size
        name : str
            a unique identifier for this meanmov
        attrs : dict
            attributes to store in the dataset for this meanmov

        Notes
        -----
        There is not yet explicit support for fancy slicing, like steps other than 1, or sub-frame slices.

        Returns
        -------
        meanmov : pyfluo.Movie
        """

        if slices is None and name is None:
            return None

        if slices is not None:
            assert all([sl.stop-sl.start==slices[0].stop-slices[0].start for sl in slices]), 'Slices must all be the same size for a meanmov.'
            n = slices[0].stop - slices[0].start

        with h5py.File(self.data_file) as f:
            grp = f.require_group('meanmovs')

            if name in grp:
                _mm = Movie(np.asarray(grp[name]), Ts=self.Ts)
                if slices is not None:
                    warnings.warn('Name was found so slices are being ignored.')
            else:
                if not self._has_data:
                    warnings.warn('Data not stored in this file, so cannot make meanmov.')
                    return None

                if slices is None:
                    warnings.warn('Must give slices if name () is not present in meanmovs.'.format(name))
                    return None
              
                data = np.zeros([n, self.shape[1], self.shape[2]])
                for i,sl in enumerate(slices):
                    if verbose:
                        print('Slice {}/{}'.format(i,len(slices)))
                    dat = self[sl]
                    data = data + (1./len(slices)) * dat
                _mm = Movie(data, Ts=self.Ts)

                ds = grp.create_dataset(name, data=_mm, compression='lzf')
                f['meanmovs'][name].attrs.update(attrs)

        return _mm

    def watch(self, name=None, roi=None, pad=10):
        """Convenience method to visualize and retrieve meanmov
        Also accepts an roi index (meaning a single ROI from an ROI set) to zoom on
        """
        if name is None or isinstance(name, int):
            with h5py.File(self.data_file) as h:
                keys = list(h['meanmovs'].keys())
            if name is None:
                for i,k in enumerate(keys):
                    print('{}\t{}'.format(i,k))
                return None
            else:
                name = keys[name]

        mov = self.get_meanmov(name=name)
        if roi is not None:
            roi = self.get_roi()[roi]
            aw = np.argwhere(roi)
            ymin,xmin = np.min(aw,axis=0) - pad
            ymax,xmax = np.max(aw,axis=0) + pad
            ymin,xmin,ymax,xmax = [int(np.round(i)) for i in [ymin,xmin,ymax,xmax]]
            ymin,xmin = [max(0,i) for i in [ymin,xmin]]
            ymax = min(self.shape[1], ymax)
            xmax = min(self.shape[0], xmax)
            mov = mov[:,ymin:ymax,xmin:xmax]

        mov.play()
        return mov
    
    def get_example(self, slices=None, resample=3, redo=False, enforce_datatype=np.int16):
        """Generate or retrieve a subset of the dataset to be used for visual inspection

        Parameters
        ----------
        slices : list-like
            list of slice objects specifying which parts of the dataset to include in result
        resample : int
            factor by which to downsample the resulting movie (using movie.resample)
        redo : bool
            if True, deletes existing stored example and generates a new one
        enforce_datatype : numpy dtype
            numpy datatype into which to cast resulting movie

        Returns
        -------
        example : pyfluo.Movie
        """
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
                    sub_movie_size = 100
                    n_sub_movies = 10
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

    def get_tr(self, idx=None, batch=None, subtract_min=True, verbose=True):
        """Compute or retrieve roi-derived traces from the dataset
        Uses the idx'th roi stored in the dataset to extract traces (one trace per roi, averaged over roi's pixels)

        Parameters
        ----------
        idx : int
            idx corresponding to roi of interest
        batch : int
            size of mini-batches in which to perform extraction. If None, defaults to class instance's batch_size
        subtract_min : bool
            subtract minimum of raw dataset from traces (does not store the traces with the transformation, but rather applies it upon returning the stored values)
            note that this parameter is overwritten to False if scanimage parameters of the dataset indicate that offset has been subtracted during acquisition
        verbose : bool
            display status

        Returns
        -------
        traces : pyfluo.Series
        """
        # subtract_min: will override this option is SubtractOffset was true in si_data

        if np.all(self.si_data['scanimage.SI.hChannels.channelSubtractOffset']):
            subtract_min = False

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
                    print ('Extracting traces...')
                if batch is None:
                    batch = self.batch_size

                all_tr = []
                for b in range(0,len(self),batch):
                    sl = slice(b,min([len(self), b+batch]))
                    if verbose:
                        print ('Slice: {}-{}, total={}'.format(sl.start,sl.stop,len(self)))
                    tr = self[sl].extract(roi)
                    all_tr.append(np.asarray(tr))
                self._tr = Series(np.concatenate(all_tr), Ts=self.Ts)
                grp.create_dataset(trname, data=np.asarray(self._tr), compression='lzf')

        if subtract_min:
            self._tr -= self.min()
        return self._tr
    
    def get_dff(self, idx=None, compute_dff_kwargs=dict(window_size=12.), recompute=False, verbose=True):
        """Compute or retrieve traces-derived deltaF/F from the dataset
        Uses the idx'th traces stored in the dataset to extract DFF

        Parameters
        ----------
        idx : int
            idx corresponding to roi/traces of interest
        compute_dff_kwargs : dict
            kwargs for pf.motion.compute_dff
        recompute : bool
            delete existing dff dataset and recompute
        verbose : bool
            display status

        Returns
        -------
        dff : pyfluo.Series
        """

        if idx is None:
            idx = self._latest_roi_idx

        dffname = 'dff{}'.format(idx)

        with h5py.File(self.data_file) as f:
            if 'traces' not in f:
                raise Exception('DFF has not yet been computed.')

            grp = f['traces']

            if dffname in grp and not recompute:
                self._dff = Series(np.asarray(grp[dffname]), Ts=self.Ts)

            elif (dffname not in grp) or recompute:
                tr = self.get_tr(idx)
                if verbose:
                    print('Computing DFF...')
                if tr is None:
                    return None
                self._dff = compute_dff(tr, verbose=verbose, **compute_dff_kwargs)
                if dffname in grp:
                    del grp[dffname]
                grp.create_dataset(dffname, data=np.asarray(self._dff), compression='lzf')
                grp[dffname].attrs.update(compute_dff_kwargs)

        if np.any(np.isnan(self._dff)):
            warnings.warn('Null values zeroed out in DFF.')
            self._dff[np.isnan(self._dff)] = 0
        if np.any(np.isinf(self._dff)):
            warnings.warn('Inf values zeroed out in DFF.')
            self._dff[np.isinf(self._dff)] = 0
        return self._dff
    
    def get_rollcor(self, idx=None, window=.300, recompute=False, verbose=True):
        """Compute or retrieve rolling correlation matrix for DFF of dataset
        Underlying function used is pyfluo.util.rolling_correlation

        Parameters
        ----------
        idx : int
            idx corresponding to roi/traces of interest
        window : float
            window size in seconds
        recompute : bool
            delete existing rollcor dataset and recompute
        verbose : bool
            display status

        Returns
        -------
        rollcor : pyfluo.Series
        """
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
                    print('Computing rollcor...')
                if dff is None:
                    return None

                corwin_ = int(window/self.Ts)
                _rollcor = rolling_correlation(dff, corwin_, verbose=verbose)

                if rollcorname in grp:
                    del grp[rollcorname]

                grp.create_dataset(rollcorname, data=np.asarray(_rollcor), compression='lzf')
                grp[rollcorname].attrs.update(window=window)

        return Series(_rollcor, Ts=self.Ts)
    
    def get_deconv(self, idx=None, output='s', recompute=False, verbose=True, **deconv_kw):
        """Compute or retrieve deconvolved DFF traces
        Underlying function is OASIS-based deconvolution

        Parameters
        ----------
        idx : int
            idx corresponding to roi/traces of interest
        output : 'c' / 's'
            c : estimated denoised trace
            s : estimated spikes
        recompute : bool
            delete existing rollcor dataset and recompute
        verbose : bool
            display status

        Returns
        -------
        deconv : pyfluo.Series
        """
        # window in seconds
        if idx is None:
            idx = self._latest_roi_idx

        deconvname = 'deconv{}'.format(idx)

        with h5py.File(self.data_file) as f:
            grp = f['traces']

            if deconvname in grp and not recompute:
                _deconv = np.asarray(grp[deconvname])

            elif (deconvname not in grp) or recompute:
                dff = self.get_dff(idx)
                if verbose:
                    print('Computing deconv...')
                if dff is None:
                    return None

                deconv_kw['penalty'] = deconv_kw.pop('penalty', 0)

                b,g,lam = np.zeros(dff.shape[1]), np.zeros(dff.shape[1]), np.zeros(dff.shape[1])
                c,s = np.zeros(dff.shape),np.zeros(dff.shape)
                if verbose:
                    pbar = ProgressBar(maxval=dff.shape[1]).start()
                for i in range(dff.shape[1]):
                    ci,si,bi,gi,lami = deconvolve(dff[:,i], **deconv_kw)
                    b[i] = bi
                    g[i] = gi
                    lam[i] = lami
                    c[:,i] = ci
                    s[:,i] = si
                    if verbose:
                        pbar.update(i)

                if verbose:
                    pbar.finish()
                _deconv = np.array([s,c])

                if deconvname in grp:
                    del grp[deconvname]

                grp.create_dataset(deconvname, data=np.asarray(_deconv), compression='lzf')
                grp[deconvname].attrs.update(deconv_kw)
                grp[deconvname].attrs.update(b=b, g=g, lam=lam)
        
        if output == 's':
            _deconv = _deconv[0]
        elif output == 'c':
            _deconv = _deconv[1]

        return Series(_deconv, Ts=self.Ts)
    
    def gen(self, chunk_size=1, n_frames=None, downsample=None, crop=False, enforce_chunk_size=False, return_idx=False, subtract_mean=False):
        """Data in the form of a generator that motion corrections, crops, applies rolling_mean, etc

        Parameters
        ----------
        chunk_size : int 
            number of frames to include in one chunk *before* downsampling
        n_frames : int 
            sum of number of total raw frames included in all yields from this iterator
        downsample : int
            factor by which to downsample each chunk (uses pf.Movie.resample)
        crop : bool
            (**under construction**) crop generated chunks to motion correction borders
        enforce_chunk_size : bool
            if True, nan-pads the last slice if necessary to make equal chunk size
        return_idx : bool
            return a 2-tuple, (chunk_of_interest, slice_used_to_extract_this_chunk)
        subtract_mean : bool
            subtract pixel-wise mean

        Returns
        -------
        This method is a generator, and as such, acts as an iterator, yielding chunks when next() or iteration is used
        Yielded items will be of length chunk_size//downsample
        """
        if n_frames is None:
            n_frames = len(self)
        if crop:
            mb = self.motion_borders
            xmin,ymin = np.floor(mb[['xmin','ymin']].values).astype(int)
            xmax,ymax = np.ceil(mb[['xmax','ymax']].values).astype(int)
        if downsample in [None,False]:
            downsample = 1

        nchunks = n_frames//chunk_size
        remainder = n_frames%chunk_size

        for idx in range(nchunks+int(remainder>0)):

            if idx == nchunks:
                # last chunk
                _i = slice(idx*chunk_size, None)
                dat = self[_i]

                # special case to handle a single frame, b/c getitem by default squeezes one frame into 2 (as opposed to 3) dimensions
                if dat.ndim == 2:
                    dat = dat[None,...]

                if enforce_chunk_size:
                    pad_size = chunk_size - len(dat)
                    dat = Movie(np.pad(dat, ((0,pad_size),(0,0),(0,0)), mode='constant', constant_values=(np.nan,)), Ts=self.Ts)
            else:
                # all regular chunks
                _i = slice(idx*chunk_size,idx*chunk_size+chunk_size)
                dat = self[_i]

            if crop:
                if dat.ndim == 3:
                    dat = dat[:,ymin:ymax,xmin:xmax]
                elif dat.ndim == 2:
                    dat = dat[ymin:ymax,xmin:xmax]

            dat = dat.resample(downsample)

            if subtract_mean:
                dat = dat - self.mean(axis=0)

            if return_idx:
                yield (dat, _i)
            else:
                yield dat

    def segment(self, gen_kwargs=dict(chunk_size=3000, n_frames=None, downsample=3, subtract_mean=True), verbose=True, **pca_ica_kwargs):
        """Segment the dataset to produce ROI, using pyfluo.pca_ica

        Parameters
        ----------
        gen_kwargs : dict
            kwargs for data.gen, to be used to iterate through data
            subtract_mean = True will be enforced
        verbose : bool
            show status
        pca_ica_kwargs : any keyword arguments accepted by pyfluo.segmentation.pca_ica

        Stores result in the segmentation group of dataset. Retrieve using get_segmentation()
        """
        gen_kwargs['subtract_mean'] = True

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
            segname = 'segmentation{}'.format(self._next_segmentation_idx)
            ds = grp.create_dataset(segname, data=comps, compression='lzf')
            pca_ica_kwargs.update(gen_kwargs=gen_kwargs)
            kw = json.dumps(pca_ica_kwargs)
            grp[segname].attrs.update(kwargs=kw)

    def play(self, **kwargs):
        """Play data as a movie

        Implemented by play_mov, using generator_fxn='gen'

        All kwargs are passed to play_mov
        """
        play_mov(self, generator_fxn='gen', **kwargs)

    def export(self, out_filename, overwrite=False, exclude=['data']):
        """Copy the contents of the dataset to another file

        Parameters
        ----------
        out_filename : str
            path to output file
        overwrite : bool
            overwrite existing file with same name as out_filename
        exclude : list
            names of datasets to explicitly exclude in export
            by default, excludes 'data'
        """
        if os.path.isdir(out_filename):
            out_filename = os.path.join(out_filename, os.path.split(self.data_file)[-1])
        elif not out_filename.endswith('.h5'):
            out_filename += '.h5'

        if os.path.exists(out_filename) and not overwrite:
            print ('File exists and overwrite=False. Returning.')
            return

        # no idea why this is necessary, but it is for py2/3 compatibility of hdf5 files
        with pd.HDFStore(self.data_file) as infile:
            handle = infile.copy(out_filename, overwrite=False)
            handle.close()

        with h5py.File(out_filename) as outfile, h5py.File(self.data_file) as infile:
            for key in infile:
                if key in outfile:
                    continue
                if key in exclude:
                    continue
                print('Copying "{}"'.format(key))
                infile.copy(key, outfile, expand_soft=True, expand_refs=True, expand_external=True)
    def import_file(self, filename, backup=True):
        """Import data from another file into this object
        NOTE: this method overwrites all attributes in the current object except for data

        Parameters
        ----------
        filename : str
            path to file from which to import data
        backup : bool
            back up the current file before running the import operation
        """
        if backup:
            self.export('{}_backup.h5'.format(os.path.splitext(self.data_file)[0]))
        with h5py.File(filename) as infile, h5py.File(self.data_file) as datafile:
            for key in infile:
                if key in datafile:
                    del datafile[key]
                infile.copy(key, datafile)

    def refine_roi(self, roi_idx=None, compare_to_input=True, include_rejections=False, verbose=True):
        """Refine definition of roi using correlation-based method

        The goal of this method is to take as input some manually selected ROIs, and to redefine their borders. This is done by first extracting the trace from each ROI over the whole dataset, then computing the dataset-wide correlation between a given trace and every pixel in the dataset (the result of this operation is stored in the data file as r). Then, the resulting correlation image is searched using basic image processing to find the maximally probable ROI.

        The correlations are computed in a batch-wise manner using specially built iterative methods in this class. Thus, this method will work for datasets of arbitrary size.

        Parameters
        ----------
        roi_idx : int
            index corresponding to roi of interest
        compare_to_input : bool
            compare each new ROI to input and consider rejecting it if not similar enough
        include_rejections : bool
            only applies when compare_to_input is True
            include individual ROIs from the original ROI set even if no suitable correlation-based match is found
        verbose : bool
            show status

        """
        # Pearson's r over hdf dataset
        if roi_idx is None:
            roi_idx = self._latest_roi_idx

        roi = self.get_roi(roi_idx)
        rs = self.get_r(roi_idx)

        if rs is None:
            # compute the correlation coefficient between each trace in sig with every pixel in the movie
            # iterative method for computation of pearson's r

            sig = self.get_tr(roi_idx)

            dmean = self.mean(axis=0)
            smean = np.mean(sig, axis=0)

            assert sig.shape[0]==self.shape[0]
            if sig.ndim == 1:
                sig = sig[:,None]

            gen = self.gen(chunk_size=self.batch_size, return_idx=True)

            sums,ns = np.zeros([sig.shape[1], self.shape[1], self.shape[2]]),[]
            for gi,idx in gen:
                if verbose:
                    print('Chunk {}'.format(idx))
                sigi = sig[idx] - smean
                di = gi - dmean

                assert len(di) == len(sigi)

                sums += np.einsum('ij,ikl->jkl',sigi,di) # rois along first axis, pixels along second and third
                ns.append(len(di))

            cov = sums / (np.sum(ns)-1) # covariance for each roi, rois along 0th axis
            
            rs = [c / (self.std(axis=0)*sd) for c,sd in zip(cov,np.std(sig,axis=0))] # pearson's r for each roi, rois along 0th axis

            # store rs
            with h5py.File(self.data_file) as f:
                rgrp = f.require_group('r')
                rgrp.create_dataset('r{}'.format(roi_idx), data=np.asarray(rs), compression='lzf')

        # all that follows is the image processing to find the best new roi definition
        # theoretically, all these steps should be parameterized in the function definition, though I have achieved such success with this as it is, that I am holding off on that for the moment

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
        for idx,m,r in zip(range(len(roi)),masks,roi):
            l,nl = label(m)
            overlap = [np.sum((l==i) & r) for i in np.arange(1,nl+1)]
            iwin = np.argmax(overlap) + 1
            new = l==iwin
            # verify that area of roi hasn't inflated/shrunk too much, else keep the manual one
            if compare_to_input:
                if new.sum()>r.sum()*3.5 or new.sum()<r.sum()*.5:
                    rejects.append(idx)
                    if include_rejections:
                        masks_new.append(r) # original
                else:
                    masks_new.append(new) # new
            else:
                masks_new.append(new)

        self.set_roi(np.asarray(masks_new))
        if verbose:
            print('Refinement of roi{} complete. {}/{} refinements rejected:\n{}'.format(roi_idx,len(rejects),len(roi),rejects))
        #self.get_tr() # extract new traces

    def select_roi(self, mm_generator_kw={}, **rv_kw):
        """Select ROI for this dataset

        This will use pyfluo.roi.ROIView to facilitate selection of an ROI. It takes advantage of the iterator option in that class, supplying this object's maxmov as the iterator of interest. Thus, it allows selection of ROI across all frames in the compressed dataset.

        All keywords are passed to ROIView

        Returns
        -------
        roiview : pyfluo.ROIView

        Importantly, ROI will only be stored to dataset when user does so manually using set_roi (this is because it cannot be known when the user is finished). Note that ROIView allows intermediate saving of the ROI in progress so as to prevent loss of selections.

        The created ROIview object is not only returned but also stored as obj.roiview.
        """
        y0,x0 = int(np.floor(self.motion_borders.ymin)), int(np.floor(self.motion_borders.xmin))
        y1,x1 = int(np.ceil(self.motion_borders.ymax)), int(np.ceil(self.motion_borders.xmax))

        def mm_mean_subtracted(downsample=5, mean_src='maxmov', equalize=True):
            """
            mean_src : 'maxmov' or 'all'
            """
            with h5py.File(self.data_file) as f:
                if 'maxmov' in f:
                    mean = np.mean(f['maxmov'], axis=0)
                    n = len(f['maxmov'])
                else:
                    raise Exception('No maxmov available.')

            if mean_src == 'maxmov':
                pass
            elif mean_src == 'all':
                mean = self.mean(axis=0)
            
            for i in range(n//downsample):
                with h5py.File(self.data_file) as f:
                    fr = np.max( f['maxmov'][i*downsample : i*downsample+downsample], axis=0 )
                fr = fr-mean
                if equalize:
                    fr = equalize_adapthist((fr-fr.min())/(fr.max()-fr.min()))

                minn = np.nanmin(fr)

                fr[:y0] = minn
                fr[y1:] = minn
                fr[:x0] = minn
                fr[x1:] = minn
                yield fr
        
        inst = mm_mean_subtracted(**mm_generator_kw)
        self.roiview = ROIView(next(inst), iterator=inst, **rv_kw)
        print('Remember to set roi using Data.set_roi(roi_view.roi).\nIf you forgot to store roi_view, it is saved in object as Data.roiview.')

        return self.roiview

    def roi_subset(self, keep, roi_idx=None):
        """Generate a new roi made of a subset of a current roi

        Parameters
        ----------
        keep : np.ndarray
            boolean array of length of roi at roi_idx, where True means keep this ROI
        roi_idx : int
            index of the ROI object of interest in the dataset

        Stores the resulting ROI as the next ROI in the dataset.
        """
        if (not isinstance(keep, np.ndarray)) or (keep.dtype != bool):
            raise Exception('Must supply boolean numpy array as "keep" parameter.')

        roi = self.get_roi(roi_idx)
        if roi is None:
            return
        tr = self.get_tr(roi_idx)
        dff = self.get_dff(roi_idx)

        roi = roi[keep]
        tr = tr[:,keep]
        dff = dff[:,keep]

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

    def process_roi(self, dist_thresh=5, **kwargs):
        """Use pf.process_roi to process the most recently stored roi

        Parameters
        ----------
        dist_thresh : threshold for pf.segmentation.merge_closebys, given here in microns (but if supplied directly in overlap_kw, will be interpreted as pixels)
            if None: will explicitly skip the closeby step, meaning no "close" rois can be merged (only overlaps)
        """
        roi = self.get_roi()
        dff = self.get_dff()

        if roi is None or dff is None:
            warnings.warn('No operation performed because roi/dff is not stored in data.')
            return None
        
        closeby_kw = kwargs.pop('closeby_kw', {})
        if dist_thresh is None:
            closeby_kw.update(distance_thresh = None)
        else:
            dist_thresh_pix = self.pixels_per_micron * dist_thresh
            closeby_kw.update(distance_thresh = dist_thresh_pix)

        roi_new = process_roi(roi, dff, closeby_kw=closeby_kw, **kwargs)
        
        roi_new = self.trim_roi_borders(roi_new)

        self.set_roi(roi_new)

    def trim_roi_borders(self, roi):

        # remove anything outside motion borders
        y0,x0 = int(np.floor(self.motion_borders.ymin)), int(np.floor(self.motion_borders.xmin))
        y1,x1 = int(np.ceil(self.motion_borders.ymax)), int(np.ceil(self.motion_borders.xmax))
        if roi.ndim == 2:
            roi = np.array([roi])
        roi = roi.astype(bool)
        roi[:,:y0,:] = False
        roi[:,y1:,:] = False
        roi[:,:,:x0] = False
        roi[:,:,x1:] = False

        roi = roi[np.any(roi, axis=(1,2))]
        
        return roi
