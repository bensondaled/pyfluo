Core Data Structures
=======================

.. autosummary::
	
	traces.Trace
	movies.Movie
    roi.ROI

Traces
---------------

.. currentmodule:: traces

.. autoclass:: Trace
   :members: __init__, plot

Movie
------------

.. currentmodule:: movies

.. autoclass:: Movie
   :members: __init__, select_roi, extract_by_roi, project, play


ROI
------------

.. currentmodule:: roi

.. autoclass:: ROI
    :members: __init__, show
