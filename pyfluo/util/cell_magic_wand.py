###########################################################################
"""
# CellMagicWand
Python implementation of ImageJ Cell Magic Wand tool originally created by Theo Walker and provided at [https://www.maxplanckflorida.org/fitzpatricklab/software/cellMagicWand/](https://www.maxplanckflorida.org/fitzpatricklab/software/cellMagicWand/).

This implementation uses the same dynamic programming and edge following algorithm as the original, but operates on Python Numpy arrays.  This makes it easier to incorporate into automated image analysis pipelines. This implementation is more robust to seed point selection and can also optionally operate on 3D images using information from all z slices. 

## Usage
There are 3 primary functions in `cell_magic_wand.py`, all of which have the same calling conventions. 

* `cell_magic_wand_single_point(image, center, min_radius, max_radius, roughness, zoom_factor)` is analogous to the original ImageJ implementation 

* `cell_magic_wand(image, center, min_radius, max_radius, roughness, zoom_factor, center_range)` runs `cell_magic_wand_single_point` with 9 seed points located in a square around and including the given center. A point is included in the final ROI if it is in the majority of ROIs found from each of the 9 seed points. 

* `cell_magic_wand_3d(image, center, min_radius, max_radius, roughness, zoom_factor, center_range, z_step)` runs `cell_magic_wand` on multiple z slices in a 3D image. A point is included in the final ROI if it is in the majority of ROIs found from the z slices. Note that this function can take considerably longer to run than above 2D functions depending on the z depth and the `z_step` argument.

#### Arguments

* `image` A 2D or 3D image in Numpy array format. A 3D `image` argument to `cell_magic_wand_3D` should have its z dimension as the 0th dimension.  For TIFF images, import `tifffile` (or install using `pip`) and run `image = tifffile.load([image_filename])` to get an image variable with the right format.

* `center` A seed point located as close to the center of the cell as possible. Should be a 2-element tuple of integers `(y,x)` (pixels down, pixels right). <em>If you are having problems, make sure you have the right  coordinate system.</em>

* `min_radius` The minimum distance from the seed point that the edge detection algorithm will search. Increasing `min_radius` helps the tool avoid finding organelles and other boundaries inside the cell.

* `max_radius` The maximum distance from the seed point the edge detection algorithm will search.  Decreasing `max_radius` helps keep the tool from circling multiple cells.

* `roughness` Controls the roughness of the edge of the detected ROI.  Analogous to the roughness parameter of the ImageJ Cell Magic Wand tool but not exactly equivalent. The tool samples points at `2 * pi * max_radius * roughness` angles around the center.  Increasing `roughness` increases runtime but can result in better detection.  The default value is 2 and values [1,10] are reasonable. 

* `zoom_factor` Factor by which the image is upsampled prior to edge detection. In most cases, the default value of 1 is appropriate.  

* `center_range` The distance to `center` of the additional 8 seed points used in the robust `cell_magic_wand` and `cell_magic_wand_3d` functions.  The default 2, so for center `(y,x)`, points `(y-2,x)`, `(y,x+2)`, etc. will be used. Set such that all points used will be inside your cells and near the center.  

* `z_step` The z stride used in `cell_magic_wand_3d`.  For an image with N z-slices, `floor(N/z_stride)` will be used. 

#### Return Values

All `cell_magic_wand` functions return a binary mask that is the same size as the input image with 1s inside the detected cell and 0s elsewhere.

`cell_magic_wand_single_point` also returns an array of points that are on the edge of the detected cell

## Tips
Even with multi-seed sampling, the `center` argument still considerably affects cell magic wand tool accuracy. 

If your image has a lot of contrast variability, preprocessing it with a median filter can improve cell magic wand tool results.  The easiest way is to use the `median_filter` function from `scipy.ndimage.filters`.


"""
# 
# Python implementation of the ImageJ Cell Magic Wand plugin
# (http://www.maxplanckflorida.org/fitzpatricklab/software/cellMagicWand/)
# with modifications to reduce variability due to seed point selection
# and to support edge detection using all z slices of a 3D image
#
# Author: Noah Apthorpe (apthorpe@cs.princeton.edu)
#
# Description: Draws a border within a specified radius
#              around a specified center "seed" point
#              using a polar transform and a dynamic
#              programming edge-following algorithm
#
# Usage: Import and call the cell_magic_wand() function
#        or cell_magic_wand_3d () function with
#        a source image, radius window, and location of center
#        point.  Other parameters set as optional arguments.
#        Returns a binary mask with 1s inside the detected edge and
#        a list of points along the detected edge.
#
###########################################################################

import numpy as np
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_fill_holes


def coord_polar_to_cart(r, theta, center):
    '''Converts polar coordinates around center to Cartesian'''
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return x, y


def coord_cart_to_polar(x, y, center):
    '''Converts Cartesian coordinates to polar'''
    r = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    theta = np.arctan2((y-center[1]), (x-center[0]))
    return r, theta


def image_cart_to_polar(image, center, min_radius, max_radius, phase_width, zoom_factor=1):
    '''Converts an image from cartesian to polar coordinates around center'''

    # Upsample image
    if zoom_factor != 1:
        image = zoom(image, (zoom_factor, zoom_factor), order=4)
        center = (center[0]*zoom_factor + zoom_factor/2, center[1]*zoom_factor + zoom_factor/2)
        min_radius = min_radius * zoom_factor
        max_radius = max_radius * zoom_factor
    
    # pad if necessary
    max_x, max_y = image.shape[0], image.shape[1]
    pad_dist_x = np.max([(center[0] + max_radius) - max_x, -(center[0] - max_radius)])
    pad_dist_y = np.max([(center[1] + max_radius) - max_y, -(center[1] - max_radius)])
    pad_dist = int(np.max([0, pad_dist_x, pad_dist_y]))
    if pad_dist != 0:
        image = np.pad(image, pad_dist, 'constant')

    # coordinate conversion
    theta, r = np.meshgrid(np.linspace(0, 2*np.pi, phase_width),
                           np.arange(min_radius, max_radius))
    x, y = coord_polar_to_cart(r, theta, center)
    x, y = np.round(x), np.round(y)
    x, y = x.astype(int), y.astype(int)
    
    polar = image[x, y]
    polar.reshape((max_radius - min_radius, phase_width))

    return polar


def mask_polar_to_cart(mask, center, min_radius, max_radius, output_shape, zoom_factor=1):
    '''Converts a polar binary mask to Cartesian and places in an image of zeros'''

    # Account for upsampling 
    if zoom_factor != 1:
        center = (center[0]*zoom_factor + zoom_factor/2, center[1]*zoom_factor + zoom_factor/2)
        min_radius = min_radius * zoom_factor
        max_radius = max_radius * zoom_factor
        output_shape = map(lambda a: a * zoom_factor, output_shape)

    # new image
    image = np.zeros(output_shape)

    # coordinate conversion
    theta, r = np.meshgrid(np.linspace(0, 2*np.pi, mask.shape[1]),
                           np.arange(0, max_radius))
    x, y = coord_polar_to_cart(r, theta, center)
    x, y = np.round(x), np.round(y)
    x, y = x.astype(int), y.astype(int)

    x = np.clip(x, 0, image.shape[0]-1)
    y = np.clip(y, 0, image.shape[1]-1)
    ix,iy = np.meshgrid(np.arange(0,mask.shape[1]), np.arange(0,mask.shape[0]))    
    image[x,y] = mask

    # downsample image
    if zoom_factor != 1:
        zf = 1/float(zoom_factor)
        image = zoom(image, (zf, zf), order=4)

    # ensure image remains a filled binary mask
    image = (image > 0.5).astype(int)
    image = binary_fill_holes(image)
    return image
    

def find_edge_2d(polar, min_radius):
    '''Dynamic programming algorithm to find edge given polar image'''
    if len(polar.shape) != 2:
        raise ValueError("argument to find_edge_2d must be 2D")

    # Dynamic programming phase
    values_right_shift = np.pad(polar, ((0, 0), (0, 1)), 'constant', constant_values=0)[:, 1:]
    values_closeright_shift = np.pad(polar, ((1, 0),(0, 1)), 'constant', constant_values=0)[:-1, 1:]
    values_awayright_shift = np.pad(polar, ((0, 1), (0, 1)), 'constant', constant_values=0)[1:, 1:]

    values_move = np.zeros((polar.shape[0], polar.shape[1], 3))
    values_move[:, :, 2] = np.add(polar, values_closeright_shift)  # closeright
    values_move[:, :, 1] = np.add(polar, values_right_shift)  # right
    values_move[:, :, 0] = np.add(polar, values_awayright_shift)  # awayright
    values = np.amax(values_move, axis=2)

    directions = np.argmax(values_move, axis=2)
    directions = np.subtract(directions, 1)
    directions = np.negative(directions)
        
    # Edge following phase
    edge = []
    mask = np.zeros(polar.shape)
    r_max, r = 0, 0
    for i,v in enumerate(values[:,0]):
        if v >= r_max:
            r, r_max = i, v
    edge.append((r+min_radius, 0))
    mask[0:r+1, 0] = 1
    for t in range(1,polar.shape[1]):
        r += directions[r, t-1]
        if r >= directions.shape[0]: r = directions.shape[0]-1
        if r < 0: r = 0
        edge.append((r+min_radius, t))
        mask[0:r+1, t] = 1

    # add to inside of mask accounting for min_radius
    new_mask = np.ones((min_radius+mask.shape[0], mask.shape[1]))
    new_mask[min_radius:, :] = mask
    
    return np.array(edge), new_mask


def edge_polar_to_cart(edge, center):
    '''Converts a list of polar edge points to a list of cartesian edge points'''
    cart_edge = [] 
    for (r,t) in edge:
        x, y = coord_polar_to_cart(r, t, center)
        cart_edge.append((round(x), round(y)))
    return cart_edge


def cell_magic_wand_single_point(image, center, min_radius, max_radius,
                                 roughness=2, zoom_factor=1):
    '''Draws a border within a specified radius around a specified center "seed" point
    using a polar transform and a dynamic programming edge-following algorithm.

    Returns a binary mask with 1s inside the detected edge and
    a list of points along the detected edge.'''
    if roughness < 1:
        roughness = 1
        print("roughness must be >= 1, setting roughness to 1")
    if min_radius < 0:
        min_radius = 0
        print("min_radius must be >=0, setting min_radius to 0")
    if max_radius <= min_radius:
        max_radius = min_radius + 1
        print("max_radius must be larger than min_radius, setting max_radius to " + str(max_radius))
    if zoom_factor <= 0:
        zoom_factor = 1
        print ("negative zoom_factor not allowed, setting zoom_factor to 1")
    phase_width = int(2 * np.pi * max_radius * roughness)
    polar_image = image_cart_to_polar(image, center, min_radius, max_radius,
                                      phase_width=phase_width, zoom_factor=zoom_factor)
    polar_edge, polar_mask = find_edge_2d(polar_image, min_radius)
    cart_edge = edge_polar_to_cart(polar_edge, center)
    cart_mask = mask_polar_to_cart(polar_mask, center, min_radius, max_radius,
                                   image.shape, zoom_factor=zoom_factor)
    return cart_mask, cart_edge


def cell_magic_wand(image, center, min_radius, max_radius,
                    roughness=2, zoom_factor=1, center_range=2):
    '''Runs the cell magic wand tool on multiple points near the supplied center and 
    combines the results for a more robust edge detection then provided by the vanilla wand tool.

    Returns a binary mask with 1s inside detected edge'''
    
    centers = []
    for i in [-center_range, 0, center_range]:
        for j in [-center_range, 0, center_range]:
            centers.append((center[0]+i, center[1]+j))
    masks = np.zeros((image.shape[0], image.shape[1], len(centers)))
    for i, c in enumerate(centers):
        mask, edge = cell_magic_wand_single_point(image, c, min_radius, max_radius,
                                                  roughness=roughness, zoom_factor=zoom_factor)
        masks[:,:,i] = mask
    mean_mask = np.mean(masks, axis=2)
    final_mask = (mean_mask > 0.5).astype(int)
    return final_mask


def cell_magic_wand_3d(image_3d, center, min_radius, max_radius,
                       roughness=2, zoom_factor=1, center_range=2, z_step=1):
    '''Robust cell magic wand tool for 3D images with dimensions (z, x, y) - default for tifffile.load.
    This functions runs the robust wand tool on each z slice in the image and returns the mean mask
    thresholded to 0.5'''
    masks = np.zeros((int(image_3d.shape[0]/z_step), image_3d.shape[1], image_3d.shape[2]))
    for s in range(int(image_3d.shape[0]/z_step)):
        mask = cell_magic_wand(image_3d[s*z_step,:,:], center, min_radius, max_radius,
                               roughness=roughness, zoom_factor=zoom_factor,
                               center_range=center_range)
        masks[s,:,:] = mask
    mean_mask = np.mean(masks, axis=0)
    final_mask = (mean_mask > 0.5).astype(int)
    return final_mask
