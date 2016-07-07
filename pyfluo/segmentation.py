import numpy as np
import pylab as pl
import sys
from sklearn.decomposition import IncrementalPCA, FastICA, NMF
sklNMF = NMF
import multiprocessing as mup
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label
import itertools as it

from .util import ProgressBar, Progress
from .config import cv2

def grid(data, rows=0.5, cols=0.5):
    """Generate index slices to split image/movie into grid

    Parameters
    ----------
    data : np.ndarray
        the data over which to find a grid, can be 2d or 3d
    rows : int, float
        number of rows in grid. if float, as fraction of data height
    cols : int, float
        number of columns in grid. if float, as fraction of data width

    Returns
    -------
    Slices corresponding to grid pieces
    """
    if data.ndim==3:
        shape = data[0].shape
    else:
        shape = data.shape

    if type(rows) in [float,np.float16,np.float32,np.float64,np.float128]:
        rows = int(np.round(rows*shape[0]))
    if type(cols) in [float,np.float16,np.float32,np.float64,np.float128]:
        cols = int(np.round(cols*shape[1]))

    n_tiles = [rows,cols]

    lims = [np.linspace(0,shape[i],nt+1).astype(int) for i,nt in zip([0,1],n_tiles)]
    slices = [[slice(*s) for s in zip(lim[:-1],lim[1:])] for lim in lims]
    slices = list(it.product(*slices))

    return slices

def ipca(mov, components=50, batch=1000):
    # vectorize the images
    num_frames, h, w = np.shape(mov)
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
   
def pca_ica(mov, components=100, batch=10000, mu=0.5, ica_func='logcosh', verbose=True):
    """Perform iterative PCA/ICA ROI extraction

    Parameters
    ----------
    mov : pyfluo.Movie
        input movie
    components : int
        number of independent components to return
    batch : int
        number of pixels to load into memory simultaneously. More leads to a better fit, but requires more memory
    mu : float
        from 0-1. In spatiotemporal ICA, closer to 1 means more weight on spatial
    ica_func : str 
        cdf for entropy maximization in ICA
    verbose : bool
        show time elapsed while running

    Returns
    -------
    Array of shape (n,y,x) where n is number of components, and y,x correspond to shape of mov

    """
    with Progress(msg='PCA', verbose=verbose):
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
    
    with Progress(msg='ICA', verbose=verbose):

        ica = FastICA(n_components=components, fun=ica_func)
        joint_ics = ica.fit_transform(eigenstuff)

    # extract the independent frames
    num_frames, h, w = mov.shape
    frame_size = h * w
    ind_frames = joint_ics[:frame_size, :]
    ind_frames = np.reshape(ind_frames.T, (components, h, w))
        
    return ind_frames  

def NMF(mov,n_components=30, init='nndsvd', beta=1, tol=5e-7, sparseness='components'):
    T,h,w=mov.shape
    Y=np.reshape(mov,(T,h*w))
    Y=Y-np.percentile(Y,1)
    Y=np.clip(Y,0,np.Inf)
    estimator = sklNMF(n_components=n_components, init=init, beta=beta,tol=tol, sparseness=sparseness)
    time_components = estimator.fit_transform(Y)
    space_components = estimator.components.reshape((n_components,h,w))
    return space_components,time_components
        
def comps_to_roi(comps, n_std=4, sigma=(2,2), pixels_thresh=[5,-1], verbose=True):
        """
        Given the spatial components output of the IPCA_stICA function extract possible regions of interest
        The algorithm estimates the significance of a components by thresholding the components after gaussian smoothing
        
        Parameters
        -----------
        comps : np.ndarray 
            spatial components
        n_std : float 
            number of (median-estimated) standard deviations above the median of the spatial component to be considered significant
        sigma : int, list
            parameter for scipy.ndimage.filters.gaussian_filter (i.e. (sigma_y, sigma_x))
        pixels_thresh : list
            [minimum number of pixels in an roi to be considered an roi, maximum]
        verbose : bool
            show status
        """        

        if pixels_thresh[1] == -1:
            pixels_thresh[1] = np.inf

        if comps.ndim == 2:
            comps = np.array([comps])
        n_comps, width, height = comps.shape
        
        all_masks = []
        if verbose:
            pbar = ProgressBar(maxval=len(comps)).start()
        for k,comp in enumerate(comps):
            if sigma:
                comp = gaussian_filter(comp, sigma)
            
            iqr = np.diff(np.percentile(comp, [25 ,75]))
            thresh_from_median = n_std * iqr / 1.35
            sig_pixels = np.abs(comp-np.median(comp)) >= thresh_from_median

            lab, n = label(sig_pixels, np.ones((3,3)))
            all_masks += [np.asarray(lab==l) for l in np.unique(lab) if l>0 and np.sum(lab==l)>=pixels_thresh[0] and np.sum(lab==l)<=pixels_thresh[1]]

            if verbose:
                pbar.update(k)
        if verbose:
            pbar.finish()

        all_masks = np.array(all_masks)
       
        return all_masks

def local_correlations(self,eight_neighbours=False):
     # Output:
     #   rho M x N matrix, cross-correlation with adjacent pixel
     # if eight_neighbours=True it will take the diagonal neighbours too

     rho = np.zeros(np.shape(self.mov)[1:3])
     w_mov = (self.mov - np.mean(self.mov, axis = 0))/np.std(self.mov, axis = 0)
 
     rho_h = np.mean(np.multiply(w_mov[:,:-1,:], w_mov[:,1:,:]), axis = 0)
     rho_w = np.mean(np.multiply(w_mov[:,:,:-1], w_mov[:,:,1:,]), axis = 0)
     
     if True:
         rho_d1 = np.mean(np.multiply(w_mov[:,1:,:-1], w_mov[:,:-1,1:,]), axis = 0)
         rho_d2 = np.mean(np.multiply(w_mov[:,:-1,:-1], w_mov[:,1:,1:,]), axis = 0)


     rho[:-1,:] = rho[:-1,:] + rho_h
     rho[1:,:] = rho[1:,:] + rho_h
     rho[:,:-1] = rho[:,:-1] + rho_w
     rho[:,1:] = rho[:,1:] + rho_w
     
     if eight_neighbours:
         rho[:-1,:-1] = rho[:-1,:-1] + rho_d2
         rho[1:,1:] = rho[1:,1:] + rho_d1
         rho[1:,:-1] = rho[1:,:-1] + rho_d1
         rho[:-1,1:] = rho[:-1,1:] + rho_d2
     
     
     if eight_neighbours:
         neighbors = 8 * np.ones(np.shape(self.mov)[1:3])  
         neighbors[0,:] = neighbors[0,:] - 3
         neighbors[-1,:] = neighbors[-1,:] - 3
         neighbors[:,0] = neighbors[:,0] - 3
         neighbors[:,-1] = neighbors[:,-1] - 3
         neighbors[0,0] = neighbors[0,0] + 1
         neighbors[-1,-1] = neighbors[-1,-1] + 1
         neighbors[-1,0] = neighbors[-1,0] + 1
         neighbors[0,-1] = neighbors[0,-1] + 1
     else:
         neighbors = 4 * np.ones(np.shape(self.mov)[1:3]) 
         neighbors[0,:] = neighbors[0,:] - 1
         neighbors[-1,:] = neighbors[-1,:] - 1
         neighbors[:,0] = neighbors[:,0] - 1
         neighbors[:,-1] = neighbors[:,-1] - 1
     

     rho = np.divide(rho, neighbors)

     return rho
     
