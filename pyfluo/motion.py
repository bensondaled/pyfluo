import numpy as np
import cv2
from .util import ProgressBar

def apply_motion_correction(mov, shifts, interpolation=cv2.INTER_LINEAR, in_place=False):  
    """Apply shifts to mov in order to correct motion

    Parameters
    ----------
    mov : pyfluo.Movie
        input movie
    shifts : np.ndarray
        obtained from the function compute_motion, list of [x_shift, y_shift] for each frame
    interpolation : def
        interpolation for 

    This supports the correction of single frames as well, given a single shift
    """
    if not in_place:
        mov=mov.copy()

    if len(mov.shape)==2: #single frame
        mov = np.array([mov])
    if len(shifts.shape)==1: #single shift
        shifts = np.array([shifts])

    t,h,w=mov.shape
    for i,frame in enumerate(mov):
        sh_x_n, sh_y_n = shifts[i]
        M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])                 
        mov[i] = cv2.warpAffine(frame,M,(w,h),flags=interpolation)
    return mov.squeeze()
    
def motion_correct(mov, return_vals=False, crop=False, **kwargs):
    """Perform motion correction using template matching.
    
    Parameters
    ----------
    mov : pyfluo.Movie
        input movie
    max_shift : int 
        maximum number of pixels to shift frame on each iteration
    template : np.ndarray / def
        if array, template to be used. if function, that used to compute template (defaults to np.median)
    interpolation : def
        flag for cv2.warpAffine (defaults to cv2.INTER_LINEAR)
    in_place : bool
        modify the supplied array so as not to allocate new memory for a copy
    return_vals : bool
        return the values (templates, (shifts, correlations)) associated with the correction
    prog_bar : bool
        show progress bar, defaults to True
    crop : bool
        crop movie according to max shifts before returning
        TODO! this needs attention
        
    Returns
    -------
    A Movie object in which motion is corrected
    Or, if return_vals: [corrected_movie, values]

    Note that this function is a convenience function for calling compute_motion_correction followed by apply_motion_correction.
    """
    #pull out kwargs for apply_motion_correction, all others assumed to be for compute_motion
    in_place = kwargs.pop('in_place', False)
    interpolation = kwargs.pop('interpolation', cv2.INTER_LINEAR)

    template,vals = compute_motion(mov, in_place=in_place, **kwargs)
    mov_cor = apply_motion_correction(mov, vals[:,:-1], in_place=in_place, interpolation=interpolation)
    if crop:
        shifts = np.sum(vals,axis=0)[:,:2]
        maxx,maxy = (shifts.max(axis=0)).astype(int)
        minx,miny = (shifts.min(axis=0)).astype(int)
        mov_cor = mov_cor[-miny:-maxy, -minx:-maxx]
    ret = mov_cor
    if return_vals:
        ret = [mov_cor, params]
    return ret


def compute_motion(mov, max_shift=(5,5), template=np.median, in_place=False, prog_bar=True):
        """Compute template, shifts, and correlations associated with template-matching-based motion correction

        Parameters
        ----------
        described in motion_correct()
        
        Returns
        -------
        template: np.ndarray
            the template used
        shifts : tuple
            one row per frame, (x_shift, y_shift, correlation)
        """
       
        if not in_place:
            mov = mov.copy()

        mov = mov.astype(np.float32)    
        n_frames_,h_i, w_i = mov.shape

        if type(max_shift) in [int,float]:
            ms_h = max_shift
            ms_w = max_shift
        elif type(max_shift) in [tuple, list, np.ndarray]:
            ms_h,ms_w = max_shift
        else:
            raise Exception('Max shift should be given as value or 2-item list')
        
        if callable(template):
            template=template(mov,axis=0)            
        elif not isinstance(template, np.ndarray):
            raise Exception('template parameter should be an array or function')
        
            
        template=template[ms_h:h_i-ms_h,ms_w:w_i-ms_w].astype(np.float32)    
        h,w = template.shape
        
        shifts=[]   # store the amount of shift in each frame
        if prog_bar:    
            pbar = ProgressBar(maxval=n_frames_).start()
        for i,frame in enumerate(mov):
             if prog_bar:    
                 pbar.update(i)             
             res = cv2.matchTemplate(frame,template,cv2.TM_CCORR_NORMED)
             avg_corr=np.mean(res)
             top_left = cv2.minMaxLoc(res)[3]
             sh_y,sh_x = top_left
             bottom_right = (top_left[0] + w, top_left[1] + h)
        
             if (0 < top_left[1] < 2 * ms_h-1) & (0 < top_left[0] < 2 * ms_w-1):
                 # if max is internal, check for subpixel shift using gaussian
                 # peak registration
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
                     
             shifts.append([sh_x_n,sh_y_n,avg_corr]) 
                 
        if prog_bar:    
            pbar.finish()         
        return np.asarray(template), np.asarray(shifts)
