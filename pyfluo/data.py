import os, h5py, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as pl

from .movies import Movie
from .motion import compute_motion, apply_motion_correction

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
            self.i2c = h.i2c
            self._has_motion_correction = 'motion' in h
        self.Ts = self.info.iloc[0].Ts
        assert (self.info.Ts==self.Ts).all(), 'Sampling periods do not match.'

    def __getitem__(self, idx):
        with h5py.File(self.data_file, 'r') as h:
            data = Movie(np.asarray(h['data'][idx]), Ts=self.Ts)
        if self._has_motion_correction:
            with pd.HDFStore(self.data_file) as h:
                motion_data = h.motion.iloc[idx][['x','y']].values
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

    def motion_correct(self, chunk_size=1000, compute_kwargs=dict(max_shift=25, resample=3), overwrite=False, verbose=True):
        """
        Corrects motion, first locally in sliding chunks of full dataset, then globally across chunks

        Parameters
        ----------
        chunk_size : int
            number of frames to include in one local-correction chunk
        """
        n_chunks = int(np.ceil(len(self)/chunk_size))

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
        
        slices = [slice(start, min(start+chunk_size, len(self))) for start in range(0, len(self), chunk_size)]
        # iterate through frame chunks, local corrections
        for idx,sli in enumerate(slices):
            sli_full = np.arange(sli.start, sli.stop)
            size = sli.stop-sli.start

            if verbose:
                print('Chunk {}/{}: frames {}:{}'.format(idx+1,n_chunks,sli.start,sli.stop), flush=True)
            chunk = Movie(self[sli])
            templ,vals = compute_motion(chunk, **compute_kwargs)

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
        glob_template,glob_vals = compute_motion(template_mov, **compute_kwargs)
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

    def show(self, show_slice=slice(None,None,100)):
        pl.imshow(self[show_slice].mean(axis=0))
