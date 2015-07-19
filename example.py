from pyfluo import ROI, Tiff, Movie, Trace, compute_dff, pca_ica, comp_to_mask, correct_motion
import numpy as np

mov = Movie('/Users/ben/PhD/data/mov2.tif', Ts=0.03)
#mov = correct_motion(mov, n_iters=8, crop=True)
#mov = compute_dff(mov, window_size=1.0, step_size=0.100)

#roi = mov.select_roi(3, projection_method=np.mean, lasso_strictness=1.)
#comp = pca_ica(mov, components=12)
#roi = ROI(mask=comp_to_mask(comp))

#tr = mov.extract_by_roi(roi)
#tr.plot()
