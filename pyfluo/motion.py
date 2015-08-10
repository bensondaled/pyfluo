"""
TODO: transform motion_correct into compute_motion plus apply_shifts
"""
import numpy as np
import cv2
from util import ProgressBar

def apply_motion_correction(mov, templates, values,interpolation=cv2.INTER_LINEAR):
    h_i,w_i = mov.shape[1:]
    values = np.sum(values,axis=0)
    for fridx,fr in enumerate(mov):
        shift = values[fridx]
        M = np.float32([ [1,0,shift[1]],[0,1,shift[0]] ])
        mov[fridx,:,:] = cv2.warpAffine(fr,M,(w_i,h_i),flags=interpolation)
    return mov
  
def apply_motion_correction_AG(mov, shifts,interpolation=cv2.INTER_LINEAR,in_place=False):  
    if not in_place:
        mov=mov.copy()    
    t,h,w=mov.shape
    for i,frame in enumerate(mov):
#         if i%100==99:
#             print "Frame %i"%(i+1); 
         sh_x_n, sh_y_n = shifts[i]
         M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])                 
         mov[i] = cv2.warpAffine(frame,M,(w,h),flags=cv2.INTER_CUBIC)
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



def correct_motion(mov, return_vals=False, crop=True, **kwargs):
    """Performs motion correction using template matching.
    
    Parameters
    ----------
    mov : pyfluo.Movie
        input movie
    return_vals : bool
        return the values (templates, (shifts, correlations)) associated with the correction
    crop : bool
        crop movie according to max shifts before returning
        NOTE this needs attention
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
    if crop:
        shifts = np.sum(params[1],axis=0)[:,:2]
        maxx,maxy = (shifts.max(axis=0)).astype(int)
        minx,miny = (shifts.min(axis=0)).astype(int)
        mov_cor = mov_cor[-miny:-maxy, -minx:-maxx]
    to_ret = mov_cor
    if return_vals:
        to_ret = [mov_cor, params]
    return to_ret


def compute_motion_AG(mov, max_shift_hw=(5,5), show_movie=False,template=np.median,interpolation=cv2.INTER_LINEAR,in_place=False):
        """                
        Performs motion corretion using the opencv matchtemplate function. At every iteration a template is built by taking the median of all frames and then used to align the other frames.
         
        Parameters
        ----------
        max_shift: maximum pixel shifts allowed when correcting
        show_movie : display the movie wile correcting it
        in_place: if True the input vector is overwritten
        
        Returns
        -------
        movCorr: motion corected movie              
        shifts : tuple, contains shifts in x and y and correlation with template
        template: the templates created at each iteration
        """
        
        if not in_place:
            mov=mov.copy()
           
        mov=mov.astype(np.float32)    
        n_frames_,h_i, w_i = mov.shape
        
        ms_h,ms_w=max_shift_hw
        
        if callable(template):
            template=template(mov,axis=0)            
        elif not type(template) == np.ndarray:
            raise Exception('Only matrices or function accepted')
        
            
        template=template[ms_h:h_i-ms_h,ms_w:w_i-ms_w].astype(np.float32)    
        h,w = template.shape      # template width and height
        
        #if show_movie:
        #    cv2.imshow('template',template/255)
        #    cv2.waitKey(2000) 
        #    cv2.destroyAllWindows()
        
        #% run algorithm, press q to stop it 
        shifts=[];   # store the amount of shift in each frame
        pbar = ProgressBar(maxval=n_frames_).start()
        for i,frame in enumerate(mov):
             pbar.update(i)             
             res = cv2.matchTemplate(frame,template,cv2.TM_CCORR_NORMED)
             avg_corr=np.mean(res);
             top_left = cv2.minMaxLoc(res)[3]
             sh_y,sh_x = top_left
             bottom_right = (top_left[0] + w, top_left[1] + h)
        
             if (0 < top_left[1] < 2 * ms_h-1) & (0 < top_left[0] < 2 * ms_w-1):
                 # if max is internal, check for subpixel shift using gaussian
                 # peak registration
                 log_xm1_y = np.log(res[sh_x-1,sh_y]);             
                 log_xp1_y = np.log(res[sh_x+1,sh_y]);             
                 log_x_ym1 = np.log(res[sh_x,sh_y-1]);             
                 log_x_yp1 = np.log(res[sh_x,sh_y+1]);             
                 four_log_xy = 4*np.log(res[sh_x,sh_y]);
    
                 sh_x_n = -(sh_x - ms_h + (log_xm1_y - log_xp1_y) / (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
                 sh_y_n = -(sh_y - ms_w + (log_x_ym1 - log_x_yp1) / (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
             else:
                 sh_x_n = -(sh_x - ms_h)
                 sh_y_n = -(sh_y - ms_w)
                     
             M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])
             mov[i] = cv2.warpAffine(frame,M,(w_i,h_i),flags=interpolation)

             shifts.append([sh_x_n,sh_y_n,avg_corr]) 
                 
             if show_movie:        
                 fr = cv2.resize(mov[i],None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
                 cv2.imshow('frame',fr/255.0)
                 if cv2.waitKey(1) & 0xFF == ord('q'):
                     cv2.destroyAllWindows()
                     break 
        pbar.finish()         
        cv2.destroyAllWindows()
        return (mov,template,shifts)