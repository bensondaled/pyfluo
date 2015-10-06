import numpy as np
import pylab as pl
import cv2
from sklearn.decomposition import IncrementalPCA, FastICA, NMF
sklNMF = NMF
import multiprocessing as mup
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label
from util import display_time_elapsed

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
   
def pca_ica(mov, components=50, batch=1000, mu=0.5, ica_func='logcosh', show_status=True):
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
    show_status : bool
        show time elapsed while running

    Returns
    -------
    Array of shape (n,y,x) where n is number of components, and y,x correspond to shape of mov

    """
    if show_status:
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
    
    if show_status:  p.terminate()
    
    return ind_frames  

def ipca_denoise(mov, components=50, batch=1000):
    """Denoise a movie using IPCA

    Parameters
    ----------
    mov : pyfluo.Movie
        input movie
    components : int
        number of components for IPCA
    batch : int:
        batch argument for IPCA

    Returns
    -------
    clean_vectors : np.ndarray
    """
    _, _, clean_vectors = ipca(mov, components, batch)
    return np.reshape(clean_vectors.T, np.shape(mov))

def NMF(mov,n_components=30, init='nndsvd', beta=1, tol=5e-7, sparseness='components'):
    T,h,w=mov.shape
    Y=np.reshape(mov,(T,h*w))
    Y=Y-np.percentile(Y,1)
    Y=np.clip(Y,0,np.Inf)
    estimator = sklNMF(n_components=n_components, init=init, beta=beta,tol=tol, sparseness=sparseness)
    time_components = estimator.fit_transform(Y)
    space_components = estimator.components.reshape((n_components,h,w))
    return space_components,time_components
        
def comps_to_roi(comps, n_std=4, sigma=(2,2), pixels_thresh=[5,-1]):
        """
        Given the spatial components output of the IPCA_stICA function extract possible regions of interest
        The algorithm estimates the significance of a components by thresholding the components after gaussian smoothing
        
        Parameters
        -----------
        comps : np.ndarray 
            spatial components
        n_std : float 
            number of standard deviations above the mean of the spatial component to be considered signiificant
        sigma : int, list
            parameter for scipy.ndimage.filters.gaussian_filter (i.e. (sigma_y, sigma_x))
        pixels_thresh : list
            [minimum number of pixels in an roi to be considered an roi, maximum]
        """        

        if pixels_thresh[1] == -1:
            pixels_thresh[1] = np.inf

        if comps.ndim == 2:
            comps = np.array([comps])
        n_comps, width, height=comps.shape
        rowcols=int(np.ceil(np.sqrt(n_comps)))
        
        allMasks = []
        maskgrouped = []
        for k,comp in enumerate(comps):
            comp = gaussian_filter(comp, sigma)
            
            maxc = np.percentile(comp,99)
            minc = np.percentile(comp,1)
            q75, q25 = np.percentile(comp, [75 ,25])
            iqr = q75 - q25
            minCompValuePos=np.median(comp)+n_std*iqr/1.35  
            minCompValueNeg=np.median(comp)-n_std*iqr/1.35            

            # got both positive and negative large magnitude pixels
            compabspos=comp*(comp>minCompValuePos)-comp*(comp<minCompValueNeg)

            #height, width = compabs.shape
            labeledpos, n = label(compabspos>0, np.ones((3,3)))
            maskgrouped.append(labeledpos)
            for jj in xrange(1,n+1):
                tmp_mask=np.asarray(labeledpos==jj)
                if np.sum(tmp_mask) >= pixels_thresh[0] and np.sum(tmp_mask) <= pixels_thresh[1]:
                    allMasks.append(tmp_mask)
        return np.array(allMasks)#, np.array(maskgrouped)

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
     
def partition_fov_kmeans(self,tradeoff_weight=.5,fx=.25,fy=.25,n_clusters=4,max_iter=500):
    """ 
    Partition the FOV in clusters that are grouping pixels close in space and in mutual correlation
                    
    Parameters
    ------------------------------
    tradeoff_weight:between 0 and 1 will weight the contributions of distance and correlation in the overall metric
    fx,fy: downsampling factor to apply to the movie 
    n_clusters,max_iter: KMeans algorithm parameters
    
    Outputs
    -------------------------------
    fovs:array 2D encoding the partitions of the FOV
    mcoef: matric of pairwise correlation coefficients
    distanceMatrix: matrix of picel distances
    
    Example
    
    """
    m1=self.copy()
    m1.resize(fx,fy)
    T,h,w=m1.mov.shape
    Y=np.reshape(m1.mov,(T,h*w))
    mcoef=np.corrcoef(Y.T)
    idxA,idxB =  np.meshgrid(range(w),range(h))
    coordmat=np.vstack((idxA.flatten(),idxB.flatten()))
    distanceMatrix=euclidean_distances(coordmat.T)
    distanceMatrix=distanceMatrix/np.max(distanceMatrix)
    estim=KMeans(n_clusters=n_clusters,max_iter=max_iter)
    kk=estim.fit(tradeoff_weight*mcoef-(1-tradeoff_weight)*distanceMatrix)
    labs=kk.labels_
    fovs=np.reshape(labs,(h,w))
    fovs=cv2.resize(np.uint8(fovs),(w,h),1/fx,1/fy,interpolation=cv2.INTER_NEAREST)
    return np.uint8(fovs), mcoef, distanceMatrix
   
