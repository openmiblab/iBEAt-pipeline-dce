import vreg
import os
import numpy as np
import dbdicom as db
import napari 
from sklearn.cluster import KMeans
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
import vreg.plot as vplot
import scipy
import pandas as pd
from tqdm import tqdm
from collections.abc import Iterable
import pydmr
import pickle
import matplotlib.pyplot as plt
import matplotlib
import logging



def get_distinct_colors(rois, colormap='jet'):
    
    if len(rois)==1:
        colors = [[255, 0, 0, 0.6]]
    elif len(rois)==2:
        colors = [[255, 0, 0, 0.6], [0, 255, 0, 0.6]]
    elif len(rois)==3:
        colors = [[255, 0, 0, 0.6], [0, 255, 0, 0.6], [0, 0, 255, 0.6]]
    else:
        n = len(rois)
        #cmap = cm.get_cmap(colormap, n)
        cmap = matplotlib.colormaps[colormap]
        colors = [cmap(i)[:3] + (0.9,) for i in np.linspace(0, 1, n)]  # Set alpha to 0.6 for transparency

    return colors


def mosaic_overlay(img, rois, file, colormap='tab20', aspect_ratio=16/9, margin=[15,5,2], show=False):


    # Define RGBA colors (R, G, B, Alpha) — alpha controls transparency
    colors = get_distinct_colors(rois, colormap=colormap)

    # Get all masks as boolean arrays
    masks = [m.astype(bool) for m in rois]

    # Build a single combined mask
    all_masks = masks[0]
    for i in range(1, len(masks)):
        all_masks = np.logical_or(all_masks, masks[i])
    if np.sum(all_masks)==0:
        raise ValueError('Empty masks')
    
    # Find corners of cropped mask
    for x0 in range(all_masks.shape[0]):
        if np.sum(all_masks[x0,:,:]) > 0:
            break
    for x1 in range(all_masks.shape[0]-1, -1, -1):
        if np.sum(all_masks[x1,:,:]) > 0:
            break
    for y0 in range(all_masks.shape[1]):
        if np.sum(all_masks[:,y0,:]) > 0:
            break
    for y1 in range(all_masks.shape[1]-1, -1, -1):
        if np.sum(all_masks[:,y1,:]) > 0:
            break
    for z0 in range(all_masks.shape[2]):
        if np.sum(all_masks[:,:,z0]) > 0:
            break
    for z1 in range(all_masks.shape[2]-1, -1, -1):
        if np.sum(all_masks[:,:,z1]) > 0:
            break

    # Add in the margins       
    x0 = x0-margin[0] if x0-margin[0]>=0 else 0
    y0 = y0-margin[1] if y0-margin[1]>=0 else 0
    z0 = z0-margin[2] if z0-margin[2]>=0 else 0
    x1 = x1+margin[0] if x1+margin[0]<all_masks.shape[0] else all_masks.shape[0]-1
    y1 = y1+margin[1] if y1+margin[1]<all_masks.shape[1] else all_masks.shape[1]-1
    z1 = z1+margin[2] if z1+margin[2]<all_masks.shape[2] else all_masks.shape[2]-1

    # Determine number of rows and columns
    # c*r = n -> c=n/r
    # c*w / r*h = a -> w*n/r = a*r*h -> (w*n) / (a*h) = r**2
    width = x1-x0+1
    height = y1-y0+1
    n_mosaics = z1-z0+1
    nrows = int(np.round(np.sqrt((width*n_mosaics)/(aspect_ratio*height))))
    ncols = int(np.ceil(n_mosaics/nrows))

    # Set up figure 
    fig, ax = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        gridspec_kw = {'wspace':0, 'hspace':0}, 
        figsize=(ncols*width/max([width,height]), nrows*height/max([width,height])),
        dpi=300,
    )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Build figure
    i = 0
    for row in tqdm(ax, desc='Building png'):
        for col in row:

            col.set_xticklabels([])
            col.set_yticklabels([])
            col.set_aspect('equal')
            col.axis("off")

            # Display the background image
            if z0+i < img.shape[2]:
                col.imshow(
                    img[x0:x1+1, y0:y1+1, z0+i].T, 
                    cmap='gray', 
                    interpolation='none', 
                    vmin=0, 
                    vmax=np.mean(img) + 2 * np.std(img),
                )

            # Overlay each mask
            if z0+i <= z1:
                for mask, color in zip(masks, colors):
                    rgba = np.zeros((x1+1-x0, y1+1-y0, 4), dtype=float)
                    for c in range(4):  # RGBA
                        rgba[..., c] = mask[x0:x1+1, y0:y1+1, z0+i] * color[c]
                    col.imshow(rgba.transpose((1,0,2)), interpolation='none')

            i += 1

    # fig.suptitle('Mask overlay', fontsize=14)
    fig.savefig(file, bbox_inches='tight', pad_inches=0)
    plt.savefig(file)
    if show == True:
        plt.show()
    plt.close()

def pad_to_shape(arr, target_shape):
    """Pad array with zeros to match target_shape (no cropping)."""
    pad_width = []
    for i, s in enumerate(arr.shape):
        diff = target_shape[i] - s
        if diff < 0:
            raise ValueError(f"Array is larger than target shape along axis {i}: {s} > {target_shape[i]}")
        pad_width.append((0, diff))
    # If arr has fewer dimensions, pad those too
    while len(pad_width) < len(target_shape):
        pad_width.append((0, target_shape[len(pad_width)]))
    return np.pad(arr, pad_width, mode='constant')

def safe_mask(arr, target_shape):
    if arr is None:
        return np.zeros(target_shape, dtype=bool)
    if arr.shape != target_shape:
        try:
            return pad_to_shape(arr, target_shape)
        except Exception as e:
            logging.warning(f"Could not pad mask to target shape: {e}")
            return np.zeros(target_shape, dtype=bool)
    return arr

def overlay(site, roi, batch_no=None):
    base = [os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients"]
    if batch_no is not None:
        base.append(f"Batch_{batch_no}")
    dir = os.path.join(*base)


    mask_png_dir = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients", 'Displays')
    
    # Logging setup
    logging.basicConfig(
    filename=os.path.join(dir, 'error.log'),
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

    os.makedirs(mask_png_dir, exist_ok=True)
    lkc_database = db.series(dir, desc='DCE_10_LKC')
    rkc_database = db.series(dir, desc='DCE_10_RKC')
    lkm_database = db.series(dir, desc='DCE_10_LKM')
    rkm_database = db.series(dir, desc='DCE_10_RKM')

    map_database = db.series(dir, desc=f'DCE_10_{roi}_rpf_gaps_filled')
    lkc = None
    rkc = None
    lkm = None
    rkm = None
    for map_path in tqdm(map_database, desc=f'Creating {site} CM Mosaics', unit='case'):
        case_id = map_path[1]
        tqdm.write(f'Processing case {case_id}...')

        try:
            lkc = next(db.volume(m) for m in lkc_database if m[1] in map_path)
        except Exception as e:
            logging.error(f'cannot display lkc for case {case_id}: {e}')
        
        try:
            rkc = next(db.volume(m) for m in rkc_database if m[1] in map_path)
        except Exception as e:
            logging.error(f'cannot display rkc for case {case_id}: {e}')

        try:
            lkm = next(db.volume(m) for m in lkm_database if m[1] in map_path)
        except Exception as e:
            logging.error(f'cannot display lkm for case {case_id}: {e}')
        
        try:
            rkm = next(db.volume(m) for m in rkm_database if m[1] in map_path)
        except Exception as e:
            logging.error(f'cannot display rkm for case {case_id}: {e}')

        try:
            _map = db.volume(map_path) 
            _map_arr = _map.values
        except Exception as e:
            logging.error(f'cannot display map for case {case_id}:{e}')

        mask_png = os.path.join(mask_png_dir, f'{case_id}.png')
        
        lkc_arr = None
        rkc_arr = None
        lkm_arr = None 
        rkm_arr = None

        if lkc is not None:
            lkc_arr = lkc.values
        
        if rkc is not None:
            rkc_arr = rkc.values
        
        if lkm is not None:
            lkm_arr = lkm.values
        
        if rkm is not None:
            rkm_arr = rkm.values
        


        masks = [
            safe_mask(lkc_arr, _map_arr.shape),
            safe_mask(rkc_arr, _map_arr.shape),
            safe_mask(lkm_arr, _map_arr.shape),
            safe_mask(rkm_arr, _map_arr.shape),
        ]
        
        target_shape = _map_arr.shape


        # Ensure all masks match shape
        for i, m in enumerate(masks):
            if m.shape != target_shape:
                masks[i] = pad_to_shape(m, target_shape)
        
        if all(np.sum(m) == 0 for m in masks):
            logging.info(f"No valid masks for case {case_id}, skipping overlay.")
            continue

        mosaic_overlay(img=_map_arr, rois=masks, file=mask_png, show=False)



if __name__ == '__main__':
    roi=['lk'] #choose any map
    for r in roi:
        overlay('Sheffield', roi=r, batch_no=1)