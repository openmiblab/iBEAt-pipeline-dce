import os
import numpy as np
import cv2
import dbdicom as db 
import logging
import matplotlib.pyplot as plt
import miblab 
from tqdm import tqdm

#Helper: Add Series Name
def add_series_name(folder, all_series: list):
    new_series_name = "DCE_4_"
    all_series.append(new_series_name)
    return new_series_name


# Helper: Threshold Parameters
def region_grow_thresh(img: np.ndarray, seed: list, threshold: float):
    """Region growing from seed points with intensity similarity threshold"""
    selected = np.zeros(img.shape, dtype=bool)
    height, width = img.shape

    neighbours = [ 
        [1,1], [1,-1], [-1,-1], [-1,1],
        [0, -1], [1, 0], [0, 1], [-1, 0],
    ]

    while seed:
        p = seed.pop()
        selected[p[0], p[1]] = True

        for dx, dy in neighbours:
            x = p[0] + dx
            y = p[1] + dy

            if x < 0 or y < 0 or x >= height or y >= width:
                continue

            if not selected[x, y]:
                if np.abs(img[x, y] - img[p[0], p[1]]) < threshold:
                    selected[x, y] = True
                    seed.append([x, y])

    return selected

# Helper: Segmentation
def segment_for_bari(axial):
    cutRatio = 0.25            # crop window around image center
    filter_kernel = (15, 15)   # gaussian blur
    threshold = 1             # starting similarity tolerance
    max_iteration = 150

    # Crop around center
    cy, cx = axial.shape[0]//2, axial.shape[1]//2
    y0, y1 = int(cy - cy*cutRatio), int(cy + cy*cutRatio)
    x0, x1 = int(cx - cx*cutRatio), int(cx + cx*cutRatio)
    axenh = axial[y0:y1, x0:x1]

    # Smooth image
    axenh_blur = cv2.GaussianBlur(axenh, filter_kernel, 0)

    # Find brightest spot (seed point)
    _, _, _, maxLoc = cv2.minMaxLoc(axenh_blur)
    seed = [[maxLoc[1], maxLoc[0]]]  # (row, col)

    # Progressive region growing
    aif_mask = None
    for i in range(max_iteration):
        aif_mask = region_grow_thresh(axenh_blur, seed.copy(), threshold)

        if aif_mask.sum() > 30 and i > 90:  # stop once region is large enough
            print(f"Segmentation converged at iteration {i}, threshold={threshold}")
            break

        threshold += 1  # relax tolerance gradually


        
    aif_mask_full = np.zeros_like(axial, dtype=np.uint8)
    if aif_mask is not None:
        aif_mask_full[y0:y1, x0:x1] = aif_mask[...,np.newaxis]

    return aif_mask_full

# Helper: Segmentation
def segment_for_bordeaux(axial):
    cutRatio = 0.25
    filter_kernel = (15, 15)
    threshold = 1
    max_iteration = 25

    # Ensure axial is 2D for processing
    if axial.ndim == 3:
        axial_2d = axial[..., 0]
    else:
        axial_2d = axial

    cy, cx = axial_2d.shape[0] // 2, axial_2d.shape[1] // 2
    y0, y1 = int(cy - cy * cutRatio), int(cy + cy * cutRatio)
    x0, x1 = int(cx - cx * cutRatio), int(cx + cx * cutRatio)
    axenh = axial_2d[y0:y1, x0:x1]

    axenh_blur = cv2.GaussianBlur(axenh, filter_kernel, 0)
    _, _, _, maxLoc = cv2.minMaxLoc(axenh_blur)
    seed = [[maxLoc[1], maxLoc[0]]]

    aif_mask = None
    for i in range(max_iteration):
        aif_mask = region_grow_thresh(axenh_blur, seed.copy(), threshold)
        print(f"Iter {i}: thr={threshold}, size={aif_mask.sum()}")
        if aif_mask.sum() > 25 or i < 50:
            print(f"Segmentation converged at iteration {i}, threshold={threshold}")
            break
        threshold += 1

    # Prepare 3D output
    aif_mask_full = np.zeros(axial.shape, dtype=np.uint8)  # keep original shape
    if aif_mask is not None:
        if axial.ndim == 3:  # assign to first channel
            aif_mask_full[y0:y1, x0:x1, 0] = aif_mask
        else:
            aif_mask_full[y0:y1, x0:x1] = aif_mask

    return aif_mask_full



def segment_for_bari(axial):
    cutRatio = 0.25            # crop window around image center
    filter_kernel = (15, 15)   # gaussian blur
    threshold = 1             # starting similarity tolerance
    max_iteration = 150

    # Crop around center
    cy, cx = axial.shape[0]//2, axial.shape[1]//2
    y0, y1 = int(cy - cy*cutRatio), int(cy + cy*cutRatio)
    x0, x1 = int(cx - cx*cutRatio), int(cx + cx*cutRatio)
    axenh = axial[y0:y1, x0:x1]

    # Smooth image
    axenh_blur = cv2.GaussianBlur(axenh, filter_kernel, 0)

    # Find brightest spot (seed point)
    _, _, _, maxLoc = cv2.minMaxLoc(axenh_blur)
    seed = [[maxLoc[1], maxLoc[0]]]  # (row, col)

    # Progressive region growing
    aif_mask = None
    for i in range(max_iteration):
        aif_mask = region_grow_thresh(axenh_blur, seed.copy(), threshold)

        if aif_mask.sum() > 30 and i > 90:  # stop once region is large enough
            print(f"Segmentation converged at iteration {i}, threshold={threshold}")
            break

        threshold += 1  # relax tolerance gradually
    
    # # Visualization
    plt.figure(figsize=(6,6))
    plt.imshow(axenh_blur, cmap="hot")
    if aif_mask is not None:
        plt.contour(aif_mask, colors="cyan", linewidths=1)
    plt.scatter([maxLoc[0]], [maxLoc[1]], c="blue", marker="x")  # seed point
    plt.title("Aorta segmentation")
    plt.show()

        
    aif_mask_full = np.zeros_like(axial, dtype=np.uint8)
    if aif_mask is not None:
        aif_mask_full[y0:y1, x0:x1] = aif_mask[...,np.newaxis]

    return aif_mask_full


def autosegment_aorta(series_vol=None):

    ref_vol = series_vol

    # Create label and save
    label_vol = miblab.totseg(ref_vol, cutoff=0.01, task='total_mr', device='cpu')

    mask_arr = np.where(label_vol.values==23, 1, 0).astype(int)
    return mask_arr



# Main Protocol
def aortaseg(site):
    
    if site == 'Sheffield':
        datapath = os.path.join(os.getcwd(), 'build', 'dce_2_data', site, 'Patients')
        dce_2_seg = db.series(datapath)
    else:
        datapath = os.path.join(os.getcwd(), 'build', 'dce_3_mip', site, 'Patients')
        database = db.series(datapath)
        dce_2_seg = [entry for entry in database if entry[3][0].strip().lower() == 'dce_3_mip']
    
    destpath = os.path.join(os.getcwd(), 'build', 'dce_4_aortaseg', site, 'Patients')
    os.makedirs(destpath, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(destpath, 'error.log'),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )



    pat_series = []
    

    for case in tqdm(dce_2_seg, desc=f'Segmenting {site} Cases', unit='case'):
        try:
            #load series name
            add_series_name(case[1], pat_series)
            mask_path = [destpath, case[1], ('Baseline', 0)]
            
            #check if series exists
            dce_mask = mask_path + [(pat_series[-1] + "aortaseg", 0)]
            if dce_mask in db.series(mask_path):
                 continue
            
            #load volume and segment roi volume
            if site == 'Sheffield':
                seg_vol = db.volume(case, dims='TemporalPositionIdentifier')
                aif_mask = autosegment_aorta(seg_vol)
            elif site =='Bari':
                seg_vol = db.volume(case)
                aif_mask = segment_for_bari(seg_vol.values)
                aif_mask = aif_mask.astype(np.uint16)
            elif site == 'Bordeaux':
                seg_vol = db.volume(case)
                aif_mask = segment_for_bordeaux(seg_vol.values)
                aif_mask = aif_mask.astype(np.uint16)

            # write mask volume to dicom
            db.write_volume((aif_mask, seg_vol.affine), dce_mask, ref=case)

        except Exception as e:
            logging.error(f"Study {case[1]} cannot be assesed: {e}")




# Call Task Site
if __name__ == '__main__':
    aortaseg('Sheffield')

