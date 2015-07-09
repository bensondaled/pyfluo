import numpy as np
import pylab as pl
from sklearn.decomposition import IncrementalPCA, FastICA

def ipca(mov, components = 50, batch =1000):

    # Parameters:
    #   components (default 50)
    #     = number of independent components to return
    #   batch (default 1000)
    #     = number of pixels to load into memory simultaneously
    #       in IPCA. More requires more memory but leads to better fit


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

def pca_ica(mov, components=50, batch=1000, mu=1, ica_func='logcosh'):
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
    
    return ind_frames  
