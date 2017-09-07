import pyfluo as pf

# create a movie from a tiff file
mov = pf.Movie('mov.tif', Ts=0.032)

# motion correct the movie
mov = pf.motion_correct(mov)

# play the movie
mov.play()

# manually select some ROIs using the interactive ROI inspector
roi_view = mov.select_roi()
roi = roi_view.roi # when selection is complete

# display a projection of the movie, with rois on top
mov.project(show=True, roi=roi)

# extract traces
tr = mov.extract(roi)

# convert traces to ∆F/F
dff = pf.compute_dff(tr)

# display traces
dff.plot()
