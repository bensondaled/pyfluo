import numpy as np
import pylab as pl
from sklearn.decomposition import IncrementalPCA, FastICA
import multiprocessing as mup
from util import display_time_elapsed

def ipca(mov, components = 50, batch =1000):
    # vectorize the images
    num_frames, h, w = mov.shape
    frame_size = h * w
    frame_samples = np.reshape(mov, (num_frames, frame_size)).T
    
    # run IPCA to approxiate the SVD
    
    ipca_f = IncrementalPCA(n_components=components, batch_size=batch)
    ipca_f.fit(frame_samples)
    
    # construct the reduced version of the movie vectors using only the 
    # principal component projection
    
    proj_frame_vectors = ipca_f.inverse_transform(ipca_f.transform(frame_samples))
        
    # get the temporal principal components (pixel time series) and 
    # associated singular values
    
    eigenseries = ipca_f.components_.T

    # the rows of eigenseries are approximately orthogonal
    # so we can approximately obtain eigenframes by multiplying the 
    # projected frame matrix by this transpose on the right
    
    eigenframes = np.dot(proj_frame_vectors, eigenseries)

    return eigenseries, eigenframes, proj_frame_vectors        

def pca_ica(mov, components=50, batch=1000, mu=0.5, ica_func='logcosh'):
    """Perform iterative PCA/ICA ROI extraction

    Parameters
    ----------
    mov (pyfluo.Movie): input movie
    components (int): number of independent components to return
    batch (int): number of pixels to load into memory simultaneously. More leads to a better fit, but requires more memory.
    mu (float): from 0-1. In spatiotemporal ICA, closer to 1 means more weight on spatial
    ica_func (str): cdf for entropy maximization in ICA

    Returns
    -------
    Array of shape (n,y,x) where n is number of components, and y,x correspond to shape of mov

    """
    p = mup.Process(target=display_time_elapsed)
    p.start()

    eigenseries, eigenframes,_proj = ipca(mov, components, batch)
    # normalize the series

    frame_scale = mu / np.max(eigenframes)
    frame_mean = np.mean(eigenframes, axis = 0)
    n_eigenframes = frame_scale * (eigenframes - frame_mean)

    series_scale = (1-mu) / np.max(eigenframes)
    series_mean = np.mean(eigenseries, axis = 0)
    n_eigenseries = series_scale * (eigenseries - series_mean)

    # build new features from the space/time data
    # and compute ICA on them

    eigenstuff = np.concatenate([n_eigenframes, n_eigenseries])

    ica = FastICA(n_components=components, fun=ica_func)
    joint_ics = ica.fit_transform(eigenstuff)

    # extract the independent frames
    num_frames, h, w = mov.shape
    frame_size = h * w
    ind_frames = joint_ics[:frame_size, :]
    ind_frames = np.reshape(ind_frames.T, (components, h, w))
    
    p.terminate()
    
    return ind_frames  

def comp_to_mask(comp, n_std=3):
    """Convert components (ex. from pyfluo.segmentation.pica) to masks

    Parameters
    ----------
    comp (np.ndarray): 3d array of shape (n,y,x), where n is number of components
    n_std (float): all pixels greater than frame mean plus this many std's will be included in the mask

    Returns
    -------
    Masks in form of 3d array of shape (n,y,x)

    """
    means = np.apply_over_axes(np.mean, comp, [1,2]).squeeze()
    stds = np.apply_over_axes(np.std, comp, [1,2]).squeeze()
    masks = (comp.T > means+n_std*stds).T
    return masks
