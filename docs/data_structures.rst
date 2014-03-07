Core Data Structures
=======================

.. autosummary::
	
	time_series.TimeSeries
	tiff.wanglab.MultiChannelTiff
	movies.Movie
    movies.LineScan

TimeSeries
---------------

.. currentmodule:: time_series

.. autoclass:: TimeSeries
   :members: __init__, get_series, append, append_series, normalize, merge, take, plot


MultiChannelTiff
------------

.. currentmodule:: tiff.wanglab

.. autoclass:: MultiChannelTiff
   :members: __init__

Movie
------------

.. autoclass:: Movie
   :members: __init__, append, take, flatten, select_roi, extract_by_roi, z_project, play


LineScan
------------

.. autoclass:: LineScan
