from .config import _pyver
import warnings
if _pyver == 2:
    warnings.warn('pyfluo is not guaranteed to be compatible with python2x')
else:
    from .movies import Movie
    from .series import Series
    from .roi import ROI, select_roi
    from .io import save, load
    from .fluorescence import compute_dff, detect_transients
    from .images.tiff import Tiff
    from .segmentation import pca_ica, comps_to_roi
    from .motion import motion_correct
    from .groups import Group
