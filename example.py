#%%
%load_ext autoreload
%autoreload 2
from pyfluo import ROI, Tiff, Movie, Trace, compute_dff, pca_ica, comp_to_mask, correct_motion
from pyfluo.motion import motion_correct_AG
import numpy as np
from matplotlib import pylab as plt
import cv2
#%%
mov = Movie(r'C:\Users\agiovann\Dropbox\Preanalyzed Data\ExamplesDataAnalysis\PC1\M_FLUO.tif', Ts=0.064)
mov1,templ_,shift_,correlations_ = motion_correct_AG(mov,interpolation=cv2.INTER_LINEAR)

mov2=apply_motion_correction_AG(mov,shift_[:1,:])


#mov = compute_dff(mov, window_size=1.0, step_size=0.100)

#roi = mov.select_roi(3, projection_method=np.mean, lasso_strictness=3.)
#comp = pca_ica(mov, components=12)
#roi = ROI(mask=comp_to_mask(comp))

#tr = mov.extract_by_roi(roi)
#tr.plot()
