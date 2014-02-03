from pyfluo.movies import MultiChannelMovie
from pyfluo.tiff import CHANNEL_IMG, CHANNEL_STIM
from pyfluo.fluorescence import compute_dff
from pyfluo import save, load
import pylab as pl
import numpy as np
pl.ion()

#globals().update(load('saved'))

dirname='/Volumes/BENSON32GB/WangLab/original_data/08272013/'
names = ["zoomed5x_500ms_500hz0%i.tif"%i for i in range(20,26)]
names = [dirname+i for i in names]
mcm = MultiChannelMovie(names, skip=(10,0))
mov = mcm.get_channel(CHANNEL_IMG)
stim = mcm.get_channel(CHANNEL_STIM).flatten()

mov.select_roi()
traces = mov.extract_by_roi()

dff = compute_dff(traces)
dff_aligned = dff.take(stim.stim_times, pad=(.5,.5)).merge()

dff_aligned.plot(stim=stim.example)
 
# save([mov, stim], 'saved', globals())

