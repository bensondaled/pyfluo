from pyfluo import ROI, Tiff, Movie, Trace, compute_dff, pca_ica, comp_to_mask

mov = Movie('/Users/ben/PhD/data/mov.tif', Ts=0.03)
mov = compute_dff(mov, window_size=1.0, step_size=0.100)
mov = mov.correct_motion(n_iters=1)

roi = mov.select_roi(3)
comp = pca_ica(mov)
roi = ROI(mask=comp_to_mask(comp))

tr = mov.extract_by_roi(roi)
tr.plot()
