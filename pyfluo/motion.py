import numpy as np
import h5py, warnings, sys
from .util import ProgressBar, Progress
from .config import *
try:
    import cv2
except:
    cv2 = None
    from skimage import transform as sktf
    from skimage.feature import match_template

def motion_correct(mov, max_iters=5, shift_threshold=1., reslice=slice(None,None), in_place=True, verbose=True, compute_kwargs={}, apply_kwargs={}):
    """Perform motion correction using template matching.

    max_iters : int
        maximum number of iterations
    shift_threshold : float
        absolute max shift value below which to exit
    reslice : slice
        used to reslice movie, example: slice(1,None,2) gives every other frame starting from 2nd frame
    in_place : bool
        perform on same memory as supplied
    verbose : bool
        show status
    compute_kwargs : dict
        kwargs for compute_motion_correction
    apply_kwargs : dict
        kwargs for apply_motion_correction
   
    Returns
    --------
    corrected movie, template, values

    Note that this function is a convenience function for calling compute_motion_correction followed by apply_motion_correction, multiple times and combining results
    """
    if not in_place:
        mov = mov.copy()
    mov = mov[reslice]
  
    all_vals = []
    for it in range(max_iters):
        if verbose:
            print('Iteration {}'.format(it)); sys.stdout.flush()
        template,vals = compute_motion(mov, **compute_kwargs)
        mov = apply_motion_correction(mov, vals, **apply_kwargs)
        maxshifts = np.abs(vals[:,[0,1]]).max(axis=0)
        all_vals.append(vals)
        if verbose:
            print('Shifts: {}'.format(str(maxshifts))); sys.stdout.flush()
        if np.all(maxshifts < shift_threshold):
            break

    # combine values from iterations
    all_vals = np.array(all_vals)
    return_vals = np.empty([all_vals.shape[1],all_vals.shape[2]])
    return_vals[:,[0,1]] = all_vals[:,:,[0,1]].sum(axis=0)
    return_vals[:,2] = all_vals[-1,:,2]

    return mov,template,return_vals

def retrieve_motion_correction_data(datafile, filename):
    with h5py.File(datafile) as mc: 
        shifts_local = np.asarray(mc[filename]['shifts'])
        shifts_global = np.asarray(mc['global_shifts'])   
        global_names = mc['global_shifts'].attrs['filenames']
        maxshift = mc.attrs['max_shift']
    global_names = [i.decode('UTF8') for i in global_names]
    shift_global = shifts_global[global_names.index(filename)]
    shifts = (shifts_local+shift_global)[:,:2]
    return shifts, maxshift

def apply_motion_correction(mov, shifts, interpolation=None, crop=None, in_place=False, verbose=True):
    """Apply shifts to mov in order to correct motion

    Parameters
    ----------
    mov : pyfluo.Movie
        input movie
    shifts : np.ndarray
        obtained from the function compute_motion, list of [x_shift, y_shift] for each frame. if more than 2 columns, assumes first 2 are the desired ones
    interpolation : def
        interpolation flag for cv2.warpAffine, defaults to cv2.INTER_LINEAR
    crop : bool / int
        whether to crop image to borders of correction. if True, crops to maximum adjustments. if int, crops that number of pixels off all sides
    in_place : bool
        in place
    verbose : bool
        display progress

    This supports the correction of single frames as well, given a single shift
    """
    if interpolation is None and cv2 is not None:
        interpolation = cv2.INTER_LINEAR

    if not in_place:
        mov=mov.copy()

    if mov.ndim==2:
        mov = mov[None,...]

    if type(shifts) in [str] and mov.filename:
        shifts,crop_ = retrieve_motion_correction_data(shifts, mov.filename)
        if crop is None:
            crop = crop_

    if shifts.ndim==1:
        shifts = shifts[None,...]

    if shifts.ndim==2 and shifts.shape[1]==3:
        shifts = shifts[:,:2]

    assert shifts.ndim==2 and shifts.shape[1]==2

    t,h,w=mov.shape
    if verbose:
        print('Applying shifts:')
        pbar = ProgressBar(maxval=len(mov)).start()
    for i,frame in enumerate(mov):
        sh_x_n, sh_y_n = shifts[i]
        if cv2 is not None:
            M = np.float32([[1,0,sh_x_n],[0,1,sh_y_n]])                 
            mov[i] = cv2.warpAffine(frame,M,(w,h),flags=interpolation)
        elif cv2 is None:
            M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n],[0,0,1]])  
            transform = sktf.AffineTransform(matrix=M)
            mov[i] = sktf.warp(frame, transform)
        if verbose:
            pbar.update(i)

    if verbose:
        pbar.finish()

    if crop:
        if crop == True:
            ymax = int(min([0, min(shifts[:,0])]) or None)
            xmax = int(min([0, min(shifts[:,1])]) or None)
            ymin = int(max(shifts[:,0]))
            xmin = int(max(shifts[:,1]))
        elif isinstance(crop, PF_numeric_types):
            crop = int(crop)
            ymax,xmax = -crop,-crop
            ymin,xmin = crop,crop
        mov = mov[:, ymin:ymax, xmin:xmax]

    return mov.squeeze()

def compute_motion(mov, max_shift=(25,25), template=np.median, template_matching_method=None, resample=4, verbose=True):
        """Compute template, shifts, and correlations associated with template-matching-based motion correction

        Parameters
        ----------
        mov : pyfluo.Movie
            input movie
        max_shift : int / list-like
            maximum number of pixels to shift frame on each iteration (by axis if list-like)
        template : np.ndarray / def
            if array, template to be used. if function, that used to compute template (defaults to np.median)
        template_matching_method : opencv constant
            method parameter for cv2.matchTemplate
        resample : int
            avg every n frames before computing template
        verbose : bool
            show progress details, defaults to True
        
        Returns
        -------
        template: np.ndarray
            the template used
        shifts : np.ndarray
            one row per frame, (y, x, metric)
        """
        if template_matching_method is None and cv2 is not None:
            template_matching_method = cv2.TM_CCORR_NORMED
      
        # Parse movie
        mov = mov.astype(np.float32)    
        n_frames,h_i, w_i = mov.shape

        # Parse max_shift param
        if type(max_shift) in [int,float]:
            ms_h = max_shift
            ms_w = max_shift
        elif type(max_shift) in [tuple, list, np.ndarray]:
            ms_h,ms_w = max_shift
        else:
            raise Exception('Max shift should be given as value or 2-item list')
       
        # Parse/generate template
        if callable(template):
            movr = mov.resample(resample)
            with Progress(msg='Computing template', verbose=verbose):
                template=template(movr,axis=0)            
        elif not isinstance(template, np.ndarray):
            raise Exception('Template parameter should be an array or function')
        template_uncropped = template.astype(np.float32)
        template=template_uncropped[ms_h:h_i-ms_h,ms_w:w_i-ms_w]
        h,w = template.shape
        
        vals = np.zeros([n_frames,3])
        if verbose:    
            print('Computing shifts:'); sys.stdout.flush()
            pbar = ProgressBar(maxval=n_frames).start()

        for i,frame in enumerate(mov):

            if verbose: 
                 pbar.update(i)             

            if cv2 is not None:
                res = cv2.matchTemplate(frame, template, template_matching_method)
                avg_metric = np.mean(res)
                top_left = cv2.minMaxLoc(res)[3]
            elif cv2 is None:
                res = match_template(frame, template)
                avg_metric = np.mean(res)
                top_left = np.unravel_index(np.argmax(res), res.shape)

            ## from hereon in, x and y are reversed in naming convention
            sh_y,sh_x = top_left
        
            if (0 < top_left[1] < 2 * ms_h-1) & (0 < top_left[0] < 2 * ms_w-1):
                # if max is internal, check for subpixel shift using gaussian peak registration
                log_xm1_y = np.log(res[sh_x-1,sh_y])             
                log_xp1_y = np.log(res[sh_x+1,sh_y])             
                log_x_ym1 = np.log(res[sh_x,sh_y-1])             
                log_x_yp1 = np.log(res[sh_x,sh_y+1])             
                four_log_xy = 4*np.log(res[sh_x,sh_y])
    
                sh_x_n = -(sh_x - ms_h + (log_xm1_y - log_xp1_y) / (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
                sh_y_n = -(sh_y - ms_w + (log_x_ym1 - log_x_yp1) / (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
            else:
                sh_x_n = -(sh_x - ms_h)
                sh_y_n = -(sh_y - ms_w)

            # NOTE: to correct for reversal in naming convention, vals are placed y, x -- but their meaning is x,y
            vals[i,:] = [sh_y_n, sh_x_n, avg_metric] # X , Y
                
        if verbose: 
            pbar.finish()         

        return template_uncropped, vals
