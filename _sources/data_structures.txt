Core Data Structures
=======================

.. autosummary::
	
	time_series.TimeSeries
    stimulation.StimSeries
	tiff.wanglab.MultiChannelTiff
	movies.Movie
    movies.LineScan

TimeSeries
---------------

.. currentmodule:: time_series

.. autoclass:: TimeSeries
   :members: __init__, get_series, append, append_series, normalize, merge, take, plot

StimSeries
------------

.. currentmodule:: stimulation

.. autoclass:: StimSeries
   :members: __init__

MultiChannelTiff
------------------

.. currentmodule:: tiff.wanglab

.. autoclass:: MultiChannelTiff
   :members: __init__, get_channel

Movie
------------

.. currentmodule:: movies

.. autoclass:: Movie
   :members: __init__, append, take, flatten, select_roi, extract_by_roi, z_project, play


LineScan
------------

.. autoclass:: LineScan
