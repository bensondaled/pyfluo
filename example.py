import pyfluo as pf

# create a movie from a tiff file
mov = pf.Movie('mov.tif', Ts=0.032)

# motion correct the movie
mov = mov.motion_correct(max_shift=10)

# play the movie
mov.play(fps=30, scale=5, contrast=3)

# manually select some ROIs
roi = mov.select_roi()

# display a projection of the movie, with rois on top
mov.project(show=True, roi=roi)

# extract traces
tr = mov.extract_by_roi(roi)

# convert traces to ∆F/F
dff = tr.compute_dff(window_size=1.0, step_size=0.100)

# display traces
dff.plot()

# save everything
pf.save('my_saved_data', movie=mov, traces=dff)
