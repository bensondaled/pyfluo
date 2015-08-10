import numpy as np
import pylab as pl
from sklearn.decomposition import IncrementalPCA, FastICA, NMF
import multiprocessing as mup
from util import display_time_elapsed

def IPCA_AG(self, components = 50, batch =1000):

    # Parameters:
    #   components (default 50)
    #     = number of independent components to return
    #   batch (default 1000)
    #     = number of pixels to load into memory simultaneously
    #       in IPCA. More requires more memory but leads to better fit


    # vectorize the images
    num_frames, h, w = np.shape(self.mov);
    frame_size = h * w;
    frame_samples = np.reshape(self.mov, (num_frames, frame_size)).T
    
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


def IPCA_stICA_AG(self, components = 50, batch = 1000, mu = 1, ICAfun = 'logcosh'):
    # Parameters:
    #   components (default 50)
    #     = number of independent components to return
    #   batch (default 1000)
    #     = number of pixels to load into memory simultaneously
    #       in IPCA. More requires more memory but leads to better fit
    #   mu (default 0.05)
    #     = parameter in range [0,1] for spatiotemporal ICA,
    #       higher mu puts more weight on spatial information
    #   ICAFun (default = 'logcosh')
    #     = cdf to use for ICA entropy maximization    
    #
    # Returns:
    #   ind_frames [components, height, width]
    #     = array of independent component "eigenframes"

    eigenseries, eigenframes,_proj = self.IPCA(components, batch)
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

    ica = FastICA(n_components=components, fun=ICAfun)
    joint_ics = ica.fit_transform(eigenstuff)

    # extract the independent frames
    num_frames, h, w = np.shape(self.mov);
    frame_size = h * w;
    ind_frames = joint_ics[:frame_size, :]
    ind_frames = np.reshape(ind_frames.T, (components, h, w))
    
    return ind_frames  


def IPCA_denoise(self, components = 50, batch = 1000):
    _, _, clean_vectors = self.IPCA(components, batch)
    self.mov = np.reshape(clean_vectors.T, np.shape(self.mov))            
    
   
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
         neighbors[0,:] = neighbors[0,:] - 3;
         neighbors[-1,:] = neighbors[-1,:] - 3;
         neighbors[:,0] = neighbors[:,0] - 3;
         neighbors[:,-1] = neighbors[:,-1] - 3;
         neighbors[0,0] = neighbors[0,0] + 1;
         neighbors[-1,-1] = neighbors[-1,-1] + 1;
         neighbors[-1,0] = neighbors[-1,0] + 1;
         neighbors[0,-1] = neighbors[0,-1] + 1;
     else:
         neighbors = 4 * np.ones(np.shape(self.mov)[1:3]) 
         neighbors[0,:] = neighbors[0,:] - 1;
         neighbors[-1,:] = neighbors[-1,:] - 1;
         neighbors[:,0] = neighbors[:,0] - 1;
         neighbors[:,-1] = neighbors[:,-1] - 1;
     

     rho = np.divide(rho, neighbors)

     return rho
     
def partition_FOV_KMeans(self,tradeoff_weight=.5,fx=.25,fy=.25,n_clusters=4,max_iter=500):
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
    idxA,idxB =  np.meshgrid(range(w),range(h));
    coordmat=np.vstack((idxA.flatten(),idxB.flatten()))
    distanceMatrix=euclidean_distances(coordmat.T);
    distanceMatrix=distanceMatrix/np.max(distanceMatrix)
    estim=KMeans(n_clusters=n_clusters,max_iter=max_iter);
    kk=estim.fit(tradeoff_weight*mcoef-(1-tradeoff_weight)*distanceMatrix)
    labs=kk.labels_
    fovs=np.reshape(labs,(h,w))
    fovs=cv2.resize(np.uint8(fovs),(w,h),1/fx,1/fy,interpolation=cv2.INTER_NEAREST)
    return np.uint8(fovs), mcoef, distanceMatrix
   
    


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
    comp : np.ndarray
        3d array of shape (n,y,x), where n is number of components
    n_std : float
        all pixels greater than frame mean plus this many std's will be included in the mask

    Returns
    -------
    Masks in form of 3d array of shape (n,y,x)

    """
    means = np.apply_over_axes(np.mean, comp, [1,2]).squeeze()
    stds = np.apply_over_axes(np.std, comp, [1,2]).squeeze()
    masks = (comp.T > means+n_std*stds).T
    return masks


def NonnegativeMatrixFactorization(mov,n_components=30, init='nndsvd', beta=1,tol=5e-7, sparseness='components'):
        T,h,w=mov.shape
        Y=np.reshape(mov,(T,h*w))
        Y=Y-np.percentile(Y,1)
        Y=np.clip(Y,0,np.Inf)
        estimator=NMF(n_components=n_components, init=init, beta=beta,tol=tol, sparseness=sparseness)
        time_components=estimator.fit_transform(Y)
        components_ = estimator.components_        
        space_components=np.reshape(components_,(n_components,h,w))
        return space_components,time_components
        
def computeDFF(self,secsWindow=5,quantilMin=8,subtract_minimum=False,squared_F=True):
    """ 
    compute the DFF of the movie
    In order to compute the baseline frames are binned according to the window length parameter
    and then the intermediate values are interpolated. 
    Parameters
    ----------
    secsWindow: length of the windows used to compute the quantile
    quantilMin : value of the quantile

    """
    
    print "computing minimum ..."; sys.stdout.flush()
    minmov=np.min(self.mov)
    if subtract_minimum:
        self.mov=self.mov-np.min(self.mov)+.1
        minmov=np.min(self.mov)

    assert(minmov>0),"All pixels must be nonnegative"                       
    numFrames,linePerFrame,pixPerLine=np.shape(self.mov)
    downsampfact=int(secsWindow/self.frameRate);
    elm_missing=int(np.ceil(numFrames*1.0/downsampfact)*downsampfact-numFrames)
    padbefore=np.floor(elm_missing/2.0)
    padafter=np.ceil(elm_missing/2.0)
    print 'Inizial Size Image:' + np.str(np.shape(self.mov)); sys.stdout.flush()
    self.mov=np.pad(self.mov,((padbefore,padafter),(0,0),(0,0)),mode='reflect')
    numFramesNew,linePerFrame,pixPerLine=np.shape(self.mov)
    #% compute baseline quickly
    print "binning data ..."; sys.stdout.flush()
    movBL=np.reshape(self.mov,(downsampfact,int(numFramesNew/downsampfact),linePerFrame,pixPerLine));
    movBL=np.percentile(movBL,quantilMin,axis=0);
    print "interpolating data ..."; sys.stdout.flush()   
    print movBL.shape        
    movBL=scipy.ndimage.zoom(np.array(movBL,dtype=np.float32),[downsampfact ,1, 1],order=0, mode='constant', cval=0.0, prefilter=False)
    
    #% compute DF/F
    if squared_F:
        self.mov=(self.mov-movBL)/np.sqrt(movBL)
    else:
        self.mov=(self.mov-movBL)/movBL
        
    self.mov=self.mov[padbefore:len(movBL)-padafter,:,:]; 
    print 'Final Size Movie:' +  np.str(self.mov.shape)          
    

    
    
   
        
def resize(mov,fx=1,fy=1,fz=1,interpolation=cv2.INTER_AREA):  
    """
    resize movies along axis and interpolate or lowpass when necessary
    
    Parameters
    -------------------
    fx,fy,fz:fraction/multiple of dimension (.5 means the image will be half the size)
    interpolation=cv2.INTER_AREA. Set to none if you do not want interpolation or lowpass
    

    """              
    if fx!=1 or fy!=1:
        print "reshaping along x and y"
        t,h,w=mov.shape
        newshape=(int(w*fy),int(h*fx))
        mov=[];
        print(newshape)
        for frame in mov:                
            mov.append(cv2.resize(frame,newshape,fx=fx,fy=fy,interpolation=interpolation))
        mov=np.asarray(mov)
    if fz!=1:
        print "reshaping along z"            
        t,h,w=mov.shape
        mov=np.reshape(self.mov,(t,h*w))
        mov=cv2.resize(self.mov,(h*w,int(fz*t)),fx=1,fy=fz,interpolation=interpolation)
#            self.mov=cv2.resize(self.mov,(h*w,int(fz*t)),fx=1,fy=fz,interpolation=interpolation)
        mov=np.reshape(self.mov,(int(fz*t),h,w))
        frameRate=self.frameRate/fz


def extractROIsFromPCAICA(spcomps, numSTD=4, gaussiansigmax=2 , gaussiansigmay=2):
        """
        Given the spatial components output of the IPCA_stICA function extract possible regions of interest
        The algorithm estimates the significance of a components by thresholding the components after gaussian smoothing
        Parameters
        -----------
        spcompomps, 3d array containing the spatial components
        numSTD: number of standard deviation above the mean of the spatial component to be considered signiificant
        """        
        
        numcomps, width, height=spcomps.shape
        rowcols=int(np.ceil(np.sqrt(numcomps)));  
        
        #%
        allMasks=[];
        maskgrouped=[];
        for k in xrange(0,numcomps):
            comp=spcomps[k]
#            plt.subplot(rowcols,rowcols,k+1)
            comp=gaussian_filter(comp,[gaussiansigmay,gaussiansigmax])
            
            maxc=np.percentile(comp,99);
            minc=np.percentile(comp,1);
#            comp=np.sign(maxc-np.abs(minc))*comp;
            q75, q25 = np.percentile(comp, [75 ,25])
            iqr = q75 - q25
            minCompValuePos=np.median(comp)+numSTD*iqr/1.35;  
            minCompValueNeg=np.median(comp)-numSTD*iqr/1.35;            

            # got both positive and negative large magnitude pixels
            compabspos=comp*(comp>minCompValuePos)-comp*(comp<minCompValueNeg);


            #height, width = compabs.shape
            labeledpos, n = label(compabspos>0, np.ones((3,3)))
            maskgrouped.append(labeledpos)
            for jj in range(1,n+1):
                tmp_mask=np.asarray(labeledpos==jj)
                allMasks.append(tmp_mask)
#            labeledneg, n = label(compabsneg>0, np.ones((3,3)))
#            maskgrouped.append(labeledneg)
#            for jj in range(n):
#                tmp_mask=np.asarray(labeledneg==jj)
#                allMasks.append(tmp_mask)
#            plt.imshow(labeled)                             
#            plt.axis('off')         
        return allMasks,maskgrouped