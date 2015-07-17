import numpy as np
import cv2
from util import ProgressBar

def apply_motion_correction(mov, templates, values):
    h_i,w_i = mov.shape[1:]
    values = np.sum(values,axis=0)
    for fridx,fr in enumerate(mov):
        shift = values[fridx]
        M = np.float32([ [1,0,shift[1]],[0,1,shift[0]] ])
        mov[fridx,:,:] = cv2.warpAffine(fr,M,(w_i,h_i),flags=cv2.INTER_CUBIC)
    return mov
def compute_motion_correction(mov, max_shift=5, sub_pixel=True, template_func=np.median, n_iters=5):
    """Computes motion correction shifts by template matching
    
    Parameters
    ----------
    (described in correct_motion doc)
    
    This can be used on its own to attain only the shifts without correcting the movie
    """
    def _run_iter(mov, base_shape, ms, sub_pixel):
        mov = mov.astype(np.float32)
        h_i,w_i = base_shape
        template=template_func(mov,axis=0)
        template=template[ms:h_i-ms,ms:w_i-ms].astype(np.float32)
        h,w = template.shape

        shifts=[]   # store the amount of shift in each frame
        
        for i,frame in enumerate(mov):
             pbar.update(it_i*len(mov) + i)
             res = cv2.matchTemplate(frame,template,cv2.TM_CCORR_NORMED)
             avg_corr=np.mean(res);
             top_left = cv2.minMaxLoc(res)[3]
             sh_y,sh_x = top_left
             bottom_right = (top_left[0] + w, top_left[1] + h)

             if sub_pixel:
                 if (0 < top_left[1] < 2 * ms-1) & (0 < top_left[0] < 2 * ms-1):
                     # if max is internal, check for subpixel shift using gaussian
                     # peak registration
                     log_xm1_y = np.log(res[sh_x-1,sh_y])
                     log_xp1_y = np.log(res[sh_x+1,sh_y])             
                     log_x_ym1 = np.log(res[sh_x,sh_y-1])             
                     log_x_yp1 = np.log(res[sh_x,sh_y+1])             
                     four_log_xy = 4*np.log(res[sh_x,sh_y])

                     sh_x_n = -(sh_x - ms + (log_xm1_y - log_xp1_y) / (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
                     sh_y_n = -(sh_y - ms + (log_x_ym1 - log_x_yp1) / (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
                 else:
                     sh_x_n = -(sh_x - ms)
                     sh_y_n = -(sh_y - ms)
                         
                 M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])
                 mov[i] = cv2.warpAffine(frame,M,(w_i,h_i),flags=cv2.INTER_CUBIC)
             else:
                 sh_x = -(top_left[1] - ms)
                 sh_y = -(top_left[0] - ms)
                 M = np.float32([[1,0,sh_y],[0,1,sh_x]])
                 mov[i] = cv2.warpAffine(frame,M,(w_i,h_i))
             shifts.append([sh_x_n,sh_y_n,avg_corr]) 
                 
        return (template,np.array(shifts),mov)

    mov_orig = mov.copy()
    h_i,w_i = mov.shape[1:]
    templates = []
    values = []
    n_steps = n_iters*len(mov_orig) #for progress bar
    pbar = ProgressBar(maxval=n_steps).start()
    for it_i in xrange(n_iters):
        pbar.update(it_i*len(mov_orig))
        ti,vi,mov = _run_iter(mov, (h_i,w_i), max_shift, sub_pixel)
        templates.append(ti)
        values.append(vi)
    pbar.finish()
    return np.array(templates), np.array(values)

def correct_motion(mov, return_vals=False, **kwargs):
    """Performs motion correction using template matching.
    
    Parameters
    ----------
    mov : pyfluo.Movie
        input movie
    return_vals : bool
        return the values (templates, (shifts, correlations)) associated with the correction
    max_shift : int 
        maximum number of pixels to shift frame on each iteration
    sub_pixel : bool 
        perform interpolation to correction motion at a sub-pixel resolution
    template_func : def
        function used to compute template for movie (defaults to np.median)
    n_iters : int
        number of iterations to run
        
    Returns
    -------
    A Movie object in which motion is corrected
    Or, if return_vals: [corrected_movie, values]

    Note that this function is a convenience function for calling compute_motion_correction followed by apply_motion_correction.
    """
    params = compute_motion_correction(mov, **kwargs)
    mov_cor = apply_motion_correction(mov, *params)
    to_ret = mov_cor
    if return_vals:
        to_ret = [mov_cor, params]
    return to_ret
