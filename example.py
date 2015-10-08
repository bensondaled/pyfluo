import numpy as np
from pyfluo import Movie, compute_dff, motion_correct

mov = Movie('/Users/ben/phd/data/mov.tif', Ts=0.064)

mov = motion_correct(mov)

mov = compute_dff(mov, window_size=1.0, step_size=0.100)

roi = mov.select_roi()

tr = mov.extract_by_roi(roi)
tr.plot()
