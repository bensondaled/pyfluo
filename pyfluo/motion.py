import numpy as np
import cv2
from .util import ProgressBar, Progress

def motion_correct(mov, compute_kwargs, apply_kwargs):
    """Perform motion correction using template matching.
   
    Returns
    --------
    corrected movie, template, values

    Note that this function is a convenience function for calling compute_motion_correction followed by apply_motion_correction.
    """
    
    template,vals = compute_motion(mov, **compute_kwargs)
    mov_cor = apply_motion_correction(mov, vals, **apply_kwargs)
    return mov_cor,template,vals


def apply_motion_correction(mov, shifts, interpolation=cv2.INTER_LINEAR, in_place=False):  
    """Apply shifts to mov in order to correct motion

    Parameters
    ----------
    mov : pyfluo.Movie
        input movie
    shifts : np.ndarray
        obtained from the function compute_motion, list of [x_shift, y_shift] for each frame. if more than 2 columns, assumes first 2 are the desired ones
    interpolation : def
        interpolation flag for cv2.warpAffine, defaults to cv2.INTER_LINEAR

    This supports the correction of single frames as well, given a single shift
    """
    if not in_place:
        mov=mov.copy()

    if shifts.dtype.names:
        shifts = shifts[['y_shift','x_shift']].view((float, 2))

    if mov.ndim==2:
        mov = mov[None,...]
    if shifts.ndim==1:
        shifts = shifts[None,...]

    assert shifts.ndim==2 and shifts.shape[1]==2

    t,h,w=mov.shape
    for i,frame in enumerate(mov):
        sh_x_n, sh_y_n = shifts[i]
        M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])                 
        mov[i] = cv2.warpAffine(frame,M,(w,h),flags=interpolation)
    return mov.squeeze()

def compute_motion(mov, max_shift=(5,5), template=np.median, template_matching_method=cv2.TM_CCORR_NORMED, verbose=True):
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
        verbose : bool
            show progress details, defaults to True
        
        Returns
        -------
        template: np.ndarray
            the template used
        shifts : np.ndarray
            one row per frame, see array's dtype for details
        """
      
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
            if verbose:
                print('Computing template:', flush=True)
            with Progress(verbose=verbose):
                template=template(mov,axis=0)            
        elif not isinstance(template, np.ndarray):
            raise Exception('Template parameter should be an array or function')
        template = template.astype(np.float32)
        template=template[ms_h:h_i-ms_h,ms_w:w_i-ms_w]
        h,w = template.shape
        
        vals = np.zeros(n_frames, dtype=[('y_shift',np.float),('x_shift',np.float),('metric',np.float)])
        if verbose:    
            print('Computing shifts:', flush=True)
            pbar = ProgressBar(maxval=n_frames).start()

        for i,frame in enumerate(mov):

            if verbose: 
                 pbar.update(i)             

            res = cv2.matchTemplate(frame, template, template_matching_method)
            avg_metric = np.mean(res)
            top_left = cv2.minMaxLoc(res)[3]
            sh_y,sh_x = top_left
            bottom_right = (top_left[0] + w, top_left[1] + h)
        
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
                    
            vals[i] = (sh_x_n, sh_y_n, avg_metric)
                
        if verbose: 
            pbar.finish()         

        return template, vals
