import numpy as np
import pylab as pl
import networkx as nx
from scipy.spatial.distance import pdist
import sys, types, time, warnings
from sklearn.decomposition import IncrementalPCA, FastICA, NMF
sklNMF = NMF
from skimage.measure import perimeter
import multiprocessing as mup
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import erosion, dilation
from skimage.filters import gaussian
from scipy.ndimage import label
from scipy.ndimage import zoom
import itertools as it

from .util import ProgressBar, Progress
from .config import cv2
from .roi import ROI
from .movies import Movie

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

def pca_ica(func, n_components=25, mu=0.5, downsample=(1,.5,.5), do_ica=False, verbose=True, **pca_kw):
    """Segment data using the PCA/ICA method

    Parameters
    ----------
    func : def
        a non-instantiated generator function that yields chunks of data frames for segmentation
    n_components : int
        number of components to keep from PCA
    mu : float 
        from 0-1, determines weight of spatial vs temporal components, >0.5 means more spatial
    downsample : tuple
        resize factor for each dimension (time, y, x). For example, .5 means shrink 2x
    verbose: bool
        show status

    Returns
    -------
    np.ndarray of shape (n,y,x) with n components, and data dimensions (y,x)
    """

    if isinstance(func, Movie):
        mov = func.copy()
        def gen():
            for i in range(1):
                yield mov
        func = gen
    
    ipca = IncrementalPCA(n_components=n_components, **pca_kw)

    # iterative pca
    idx = 1
    t0 = time.time()
    if verbose:
        print('PCA:'); sys.stdout.flush()
    for chunk in func():
        if verbose:
            print('Chunk #{}, t={:0.3f}'.format(idx,time.time()-t0)); sys.stdout.flush()
            idx += 1
        if downsample is not None:
            chunk = zoom(chunk, downsample, order=1)
        frame_shape = chunk.shape[1:]
        chunk = chunk.reshape([len(chunk), -1])
        ipca.partial_fit(chunk)
    if verbose:
        print('PCA took {:0.2f} seconds.'.format(time.time()-t0)); sys.stdout.flush()
    comps = ipca.components_
    
    if not do_ica:
        comps = comps.reshape([len(comps), frame_shape[0], frame_shape[1]])
        if downsample is not None:
            comps = zoom(comps, [1] + [1/i for i in downsample[1:]], order=1)
        return comps

    # reconstruct low-dimensional movie
    reduced_mov = []
    if verbose:
        print('Reconstruct:'); sys.stdout.flush()
        idx = 1
    for chunk in func():
        if verbose:
            print('Chunk #{}'.format(idx)); sys.stdout.flush()
            idx += 1
        if downsample is not None:
            chunk = zoom(chunk, downsample, order=1)
        frame_shape = chunk.shape[1:]
        chunk = chunk.reshape([len(chunk), -1])
        reduced_mov.append(np.dot(comps, chunk.T))
    reduced_mov = np.concatenate(reduced_mov, axis=-1).T # t x n matrix

    #eigen_mov = pf.Movie(res.dot(comps).reshape((-1,)+frame_shape)) # not needed

    comps = comps.T
    ica_space = mu * (comps - comps.mean(axis=0))/comps.max()
    ica_time = (1.-mu) * (reduced_mov - reduced_mov.mean(axis=0))/reduced_mov.max()

    conc = np.concatenate([ica_space,ica_time])

    if np.any(np.isinf(conc)):
        warnings.warn('Infinite values detected after PCA.')
        conc[np.isinf(conc)] = 0
    if np.any(np.isnan(conc)):
        warnings.warn('NaN values detected after PCA.')
        conc[np.isnan(conc)] = 0

    with Progress(msg='ICA', verbose=verbose):
        ica = FastICA(max_iter=500, tol=0.001)
        ica_result = ica.fit_transform(conc)
    
    output_shape = [len(comps.T), frame_shape[0], frame_shape[1]]
    final = ica_result[:np.product(frame_shape)].T.reshape(output_shape)
    if downsample is not None:
        final = zoom(final, [1] + [1/i for i in downsample[1:]], order=1)
    return final

def ipca(mov, components=50, batch=1000):
    """Incremental PCA

    Parameters
    ----------
    mov : np.ndarray
        3d data where time is the 0th axis, shape (n,y,x)
    components : int
        number of components to keep for PCA
    batch : int
        batch_size for updates

    Returns
    -------
    np.ndarray, perserved components after PCA of data
    """

    # vectorize the images
    shape = mov.shape
    num_frames, h, w = shape
    frame_samples = np.reshape(mov, (num_frames, w*h)).T
    
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

    return eigenseries, eigenframes, proj_frame_vectors, shape
   
def pca_ica_old(mov, components=100, batch=10000, mu=0.5, ica_func='logcosh', verbose=True):
    """Perform iterative PCA/ICA ROI extraction

    Parameters
    ----------
    mov : pyfluo.Movie, generator
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
        eigenseries, eigenframes,_proj,shape = ipca(mov, components, batch)

    # normalize the frames
    frame_scale = mu / np.max(eigenframes)
    frame_mean = np.mean(eigenframes, axis=0)
    n_eigenframes = frame_scale * (eigenframes - frame_mean)

    # normalize the series
    series_scale = (1-mu) / np.max(eigenframes)
    series_mean = np.mean(eigenseries, axis=0)
    n_eigenseries = series_scale * (eigenseries - series_mean)

    # build new features from the space/time data
    # and compute ICA on them

    eigenstuff = np.concatenate([n_eigenframes, n_eigenseries])
    
    with Progress(msg='ICA', verbose=verbose):

        ica = FastICA(n_components=components, fun=ica_func)
        joint_ics = ica.fit_transform(eigenstuff)

    # extract the independent frames
    num_frames, h, w = shape
    ind_frames = joint_ics[:w*h, :]
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
       
def comps_to_roi(comps, n_std=3, sigma=(2,2), pixels_thresh=[25,-1], circularity_thresh=[0,1], verbose=True):
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
            
            thresh = n_std * np.std(comp)
            sig_pixels = np.abs(comp-comp.mean()) > thresh

            lab, n = label(sig_pixels, np.ones((3,3)))
            new_masks = [np.asarray(lab==l) for l in np.unique(lab) if l>0 and np.sum(lab==l)>=pixels_thresh[0] and np.sum(lab==l)<=pixels_thresh[1]]
            circ = [circularity(n) for n in new_masks]
            new_masks = [nm for nm,c in zip(new_masks,circ) if c>=circularity_thresh[0] and c<=circularity_thresh[1]]
            all_masks += new_masks

            if verbose:
                pbar.update(k)
        if verbose:
            pbar.finish()

        all_masks = np.array(all_masks)
        centers = [np.sum(np.argwhere(a).mean(axis=0)**2) for a in all_masks]
      
        return ROI(all_masks[np.argsort(centers)])

def circularity(m):
    area = m.sum()
    perim = perimeter(m)
    return 4*np.pi*area/perim**2

def mindist(a,b):
    # compute the minimum distance from a to b
    # meaning: for each point in a, compute distance to all points in b
    # then take minimum of all computed distances
    # a and b are 2d np arrays representing rois
    aw_a = np.argwhere(a)
    aw_b = np.argwhere(b)

    dds = np.array([np.sqrt(np.sum((pt - aw_b)**2, axis=1)) for pt in aw_a])
    return np.min(dds)

def similarity_neighbourhoods(dff, similarity_thresh=.8):
    # given a t x n dff signal, find neighbourhoods of cells with similar (correlated) races
    corr = np.corrcoef(dff.T)
    similar = np.argwhere(corr > similarity_thresh)
    similar = similar[similar[:,0] != similar[:,1]]
    # convert into a graph
    g = nx.Graph()
    g.add_edges_from(similar)
    # get neighbourhoods
    cc = np.array([np.array(list(i)) for i in nx.connected_components(g)])
    return cc

def remove_overlaps(roi, dff, overlap_thresh=.7, debug=False, **similarity_kw):
    # given an roi and its corresponding traces, use similarilty_neighbourhoods and overlap metrics to remove cells that are similar because they are on top of one another
    # thresh: if fraction `overlap_thresh` of roi A lives inside any other roi, remove roi A
    # returns new roi with the merging/removal performed

    roi_ = roi.reshape([len(roi), -1]).astype(int) # flat version for overlap metrics
    cc = similarity_neighbourhoods(dff, **similarity_kw)
    remove = np.array([], dtype=int)

    for cci in cc:
        if debug:
            fig,axs = pl.subplots(1,2)
            roi[cci].show(ax=axs[0])

        rs = roi_[cci]
        sums = rs.sum(axis=1)
        dot = (rs.dot(rs.T)) / sums
        np.fill_diagonal(dot, 0) # diagonal refers to self for each roi
        # each column now represents one roi, and each row is the fraction of itself that lives within the roi of that row
        # so goal is to remove columns that contain anything above threshold
        remove_, = np.where(dot.max(axis=0) > overlap_thresh)
        if len(remove_) < len(cci):
            # some but not all were found to live inside others
            pass
        elif len(remove_) == len(cci):
            # all were found to live inside each other; so take the largest
            keep = np.argmax(sums)
            remove_ = np.arange(len(cci)) != keep
        
        to_keep = np.ones(len(cci)).astype(bool)
        to_keep[remove_] = False
        remove_ = cci[remove_]
        remove = np.append(remove, remove_)
        cci = cci[to_keep]
        
        if debug:
            roi[cci].show(ax=axs[1])

    keeper = np.ones(len(roi)).astype(bool)
    keeper[remove] = False
    roi_new = roi[keeper]
    return roi_new,keeper

def merge_closebys(roi, dff, distance_thresh=20, **similarity_kw):
    # given rois and corresponding dffs, if any pair of rois has similar traces and is close to one another, merge them
    # approach: compute pairwise "mindist", draw edges between those below thresh, and merge any neighbourhoods
    # thresh in pixels
    # if distance_thresh is None, skip this whole step and return same roi

    if distance_thresh is None:
        return roi

    cc = similarity_neighbourhoods(dff, **similarity_kw)
    remove = np.array([], dtype=int)
    to_add = []

    for cci in cc:

        rs = roi[cci]
        mds = np.ones([len(rs),len(rs)])*distance_thresh + 1 # all start above thresh
        for idx,r in enumerate(rs):
            mds[idx,:] = [mindist(r,rs[i2]) for i2 in range(len(rs))]

        close = np.argwhere(mds < distance_thresh)
        close = close[close[:,0] != close[:,1]]
        # convert into a graph
        g = nx.Graph()
        g.add_edges_from(close)
        # get neighbourhoods
        close_nbhds = np.array([np.array(list(i)) for i in nx.connected_components(g)])
        close_nbhds = np.array([cn for cn in close_nbhds if len(cn)>1])
       
        # remove any rois that appear in a neighbourhood that's about to be merged
        if len(close_nbhds) == 0:
            continue
        remove = np.append(remove, cci[np.unique(np.concatenate(close_nbhds))])
        # merge all >1 neighbourhoods
        for cn in close_nbhds:
            new = np.any(roi[cci[cn]], axis=0).astype(bool)
            to_add.append(new)

    keeper = np.ones(len(roi)).astype(bool)
    keeper[remove] = False
    roi_new = roi[keeper]
    if len(to_add) > 0:
        roi_new = roi_new.add(ROI(to_add))
    return roi_new

def process_roi(roi, dff, overlap_kw={}, closeby_kw={}):
    """
    Given roi and dff of corresponding traces, use `remove_overlaps` and `merge_closebys` to "correct" the roi to less redundant set

    Returns new roi
    (because merges occur too, dff cannot be returned and must be re-computed)
    """
    overlap_kw['similarity_thresh'] = overlap_kw.pop('similarity_thresh', .5)
    closeby_kw['similarity_thresh'] = closeby_kw.pop('similarity_thresh', .8)

    r,keep = remove_overlaps(roi, dff, **overlap_kw)
    r = merge_closebys(r, dff[:,keep], **closeby_kw)
    return r

