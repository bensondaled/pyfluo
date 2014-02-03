About this library
=====================
The pyfluo library enables easy and efficient manipulation of imaging data using a set of custom-built data structures and functions. 

A quick example
-----------------
Here is a quick-start example to get you moving with pyfluo.

.. code-block:: python
	:linenos:

	from pyfluo import MultiChannelMovie
	from pyfluo.fluorescence import compute_dff
	from pyfluo.tiff import CHANNEL_IMG, CHANNEL_STIM
	from pyfluo.io import save, load
	import os
	import numpy as np
	import pylab as pl
	pl.ion()
	
	# reload previously saved data
	globals().update(load('my_saved_data'))
	
	# specify tif files to be loaded
	dir_name = './lab-data/experiment-june25/'
	names = ["os.path.join(dir_name,file_name) for file_name in os.listdir(dir_name) if '500Hz' in file_name]
	
	# load tif files
	mcm = MultiChannelMovie(names, skip=(10,0))
	mov = mcm.get_channel(CHANNEL_IMG)
	stim = mcm.get_channel(CHANNEL_STIM).flatten()
	
	# play the movie
	mov.play(fps=15)
	
	# select some regions of interest & extract their signals
	mov.select_roi(3)
	signals = mov.extract_by_roi()
	
	# compute delta-f over f of signals
	dff = compute_dff(signals)
	
	# extract and align stimulation events from signals
	dff_stims = dff.take(stim.stim_times, pad=(.5,.5))
	
	# plot the result
	dff_aligned.plot(stim=stim.example)
	
	# save the figure
	pl.savefig('dff_aligned.png')
	# and the data
	save([dff, stim], 'my_new_data', globals())


Import the basics
-------------------
At the start of each project, you will want to import a number of features.

``from pyfluo.movies import MultiChannelMovie``

Load a tiff
----------------


Lists can be unnumbered like:

 * Item Foo
 * Item Bar

Or automatically numbered:

 #. Item 1
 #. Item 2