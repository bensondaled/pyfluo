import numpy as np
from pyfluo import ROI, Tiff, Movie, Trace, compute_dff, pca_ica, comp_to_mask, motion_correct

# load movie
mov = Movie('/Users/ben/phd/data/mov.tif', Ts=0.064)

# correct motion
mov_cor = motion_correct(mov, in_place=False)

#mov = compute_dff(mov, window_size=1.0, step_size=0.100)

roi = mov.select_roi(4, projection_method=np.mean)
#comp = pca_ica(mov, components=12)
#roi = ROI(mask=comp_to_mask(comp))

#tr = mov.extract_by_roi(roi)
#tr.plot()

