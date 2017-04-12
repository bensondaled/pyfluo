import numpy as np
import pylab as pl
import networkx as nx
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

def pca_ica(func, n_components=25, mu=0.5, downsample=(.25,.5,.5), do_ica=False, verbose=True):
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
    
    ipca = IncrementalPCA(n_components=n_components)

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
        chunk = chunk.reshape([len(chunk), -1])
        ipca.partial_fit(chunk)
    if verbose:
        print('PCA took {:0.2f} seconds.'.format(time.time()-t0)); sys.stdout.flush()
    comps = ipca.components_

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

    if not do_ica:
        comps = comps.reshape([len(comps), frame_shape[0], frame_shape[1]])
        if downsample is not None:
            comps = zoom(comps, [1] + [1/i for i in downsample[1:]], order=1)
        return comps

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

def merge_roi(roi, overlap_thresh=0.4, merge_mode='largest'):
    # Attempts to merge regions of interest that have overlap
    roi_orig = roi.copy()
    roi = roi.reshape([len(roi), -1]).astype(int)

    norm = np.repeat(roi.sum(axis=1), len(roi)).reshape([len(roi),len(roi)])
    overlap_oflarger = roi.dot(roi.T) / np.max([norm, norm.T], axis=0) # min or max means as pctg of smaller or larger roi
    overlap_ofsmaller = roi.dot(roi.T) / np.min([norm, norm.T], axis=0) # min or max means as pctg of smaller or larger roi
    overlap_oflarger[np.triu_indices_from(overlap_oflarger)] = 0
    overlap_ofsmaller[np.triu_indices_from(overlap_ofsmaller)] = 0

    merge = np.asarray(np.nonzero( (overlap_oflarger>overlap_thresh) | (overlap_ofsmaller>overlap_thresh) )).T # debating between & and | for the operator
    g = nx.Graph()
    g.add_nodes_from(np.arange(len(overlap_ofsmaller)))
    g.add_edges_from(merge)
    merge_grps = list(nx.connected_components(g))

    if merge_mode == 'sum':
        # sum members of a group:
        roi = np.array([(roi_orig[list(mg)].sum(axis=0))>0 for mg in merge_grps])
    elif merge_mode == 'largest':
        # or take largest of a group:
        def largest_of(a):
            return a[np.argmax([i.sum() for i in a])]
        roi = np.array([largest_of(roi_orig[list(mg)]) for mg in merge_grps])

    roi = ROI(roi)
    return roi

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
     
def semiauto(mov, roi, min_size=100, n_std=2., filter_kernel=.75, no_borders=True, return_all=False, verbose=True):
    ## 
    mov = mov-mov.mean(axis=0)

    shape = np.array(mov.shape[1:])

    tr_all = mov.extract(roi)
    if roi.ndim == 2:
        roi = [roi]

    cims = []
    full = []
    result = []

    for idx,r in enumerate(roi):
        tr = tr_all.iloc[:,idx].values
        p = r.pts.mean(axis=0)

        ## Within-roi correlation map
        aw = np.argwhere(r)
        within = mov[:,aw[:,0],aw[:,1]]
        within_ccs = [np.corrcoef(w,tr)[0,1] for w in within.T]
        within_mean = np.mean(within_ccs)
        within_std = np.std(within_ccs)

        ## Fullframe correlation map
        res = np.zeros(shape)
        (maxy,maxx),(miny,minx) = shape-1,[0,0]
        if verbose:
            pbar = ProgressBar(maxval=shape[0]).start()
        for y in np.arange(0,shape[0]):
            if verbose:
                pbar.update(y)
            for x in np.arange(0,shape[1]):
                res[y,x] = np.corrcoef(tr, mov[:,y,x])[0,1]
        if verbose:
            pbar.finish()

        cims.append(res)
        # some filtering
        thresh = within_mean - n_std*within_std
        im = res>thresh
        im = gaussian(im, filter_kernel)
        im = erosion(im)
        im = dilation(im)
        im = im>0

        # labeling
        objs,nob = label(im)
        segs = []
        for i in range(1,nob+1):
            if np.sum(objs==i)<min_size:
                continue
            wh = np.where(objs==i)
            if no_borders and any([np.any((w==0)|(w==d)) for w,d in zip(wh,shape)]):
                continue
            segs.append(objs==i)
        centers = np.array([np.argwhere(s).mean(axis=0) for s in segs])
        dist = np.array([np.sqrt(np.sum(p-c)**2) for c in centers])
        cell = segs[np.argmin(dist)]
        result.append(cell)
        full.append(segs)
    if return_all:
        return ROI(result), [ROI(f) for f in full], np.array(cims)
    else:
        return ROI(result)
    ##
