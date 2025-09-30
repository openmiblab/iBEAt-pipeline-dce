import os
import numpy as np
import cv2
import dbdicom as db 
import logging
import matplotlib.pyplot as plt
import napari
import miblab 


def bari_add_series_name(folder, all_series: list):
    new_series_name = "DCE_4_"
    all_series.append(new_series_name)
    return new_series_name

def leeds_add_series_name(folder, all_series: list):
    new_series_name = "DCE_4_"
    all_series.append(new_series_name)
    return new_series_name

def sheffield_add_series_name(folder, all_series: list):
    new_series_name = "DCE_4_"
    all_series.append(new_series_name)
    return new_series_name

# def segment(axial):

#     cutRatio=0.25             #create a window around the center of the image where the aorta is
#     filter_kernel=(15,15)     #gaussian kernel for smoothing the image to destroy any noisy single high intensity filter
#     threshold = 20             #threshold for the region growing algorithm

#     # Calculate max signal enhancement over a window around the center
#     cy, cx = int(axial.shape[0]/2), int(axial.shape[1]/2)
#     y0, y1 = int(cy-cy*cutRatio), int(cy+cy*cutRatio)
#     x0, x1 = int(cx-cx*cutRatio), int(cx+cx*cutRatio)

#     #axwin = np.zeros(axial.shape)
#     axenh = axial[y0:y1, x0:x1,:]
#     #axwin[y0:y1, x0:x1,:] = axial[y0:y1, x0:x1,:]
#     # axenh = (axenh - np.min(axenh)) / (np.max(axenh) - np.min(axenh))
#     #axenh = np.max(axenh,axis=2) - np.min(axenh,axis=2)

#     # Get 3 seed points with maximal enhhancement values
#     axenh_blur = cv2.GaussianBlur(axenh, filter_kernel,cv2.BORDER_DEFAULT)
#     print("min:", axenh_blur.min(), "max:", axenh_blur.max())
#     plt.imshow(axenh_blur, cmap='hot')
#     # plt.colorbar()
#     # plt.show()
#     _, _, _, p1 = cv2.minMaxLoc(axenh_blur)
#     axenh_blur[p1[1],p1[0]] = 0
#     _, _, _, p2 = cv2.minMaxLoc(axenh_blur)
#     axenh_blur[p2[1],p2[0]] = 0
#     _, _, _, p3 = cv2.minMaxLoc(axenh_blur)
#     axenh_blur[p3[1],p3[0]] = 0

#     _, _, _, maxLoc = cv2.minMaxLoc(axenh_blur)
#     seed = [[maxLoc[1], maxLoc[0]]]
    
#     #aif_mask = axenh_blur >= 0.35

#     max_iteration = 250
#     for i in range(max_iteration):
#         seeds = [[p1[1],p1[0]], [p2[1],p2[0]]]
#         print(seeds)    
#         aif_mask = region_grow_thresh(axenh_blur, seeds.copy(), threshold)
#         print(seeds)
#         if len(aif_mask[aif_mask==1]) > 25:
            
#             break
#         threshold += 1
#         aif_mask = []
    
#     max_iteration = 250
#     aif_mask = None

#     for i in range(max_iteration):
#         seeds = [[p1[1], p1[0]]]   # just one seed (brightest spot)
#         aif_mask = region_grow_thresh(axenh_blur, seeds.copy(), threshold)

#     if aif_mask.sum() > 25:    # number of True pixels
#         print(f"Segmentation found at iteration {i}, threshold={threshold}")
#         break

#     threshold += 2 
            

#     return aif_mask


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

def segment_for_sheffield(axial):
    cutRatio = 0.25            # crop window around image center
    filter_kernel = (15, 15)   # gaussian blur
    threshold = 1             # starting similarity tolerance
    max_iteration = 25

    #choose slice
    axial = axial[:, 70, :]
    axial = np.transpose(axial, (1, 0))



    # Crop around center
    # cy, cx = axial.shape[0]//2, axial.shape[1]//2
    # y0, y1 = int(cy - cy*cutRatio), int(cy + cy*cutRatio)
    # x0, x1 = int(cx - cx*cutRatio), int(cx + cx*cutRatio)
    # axenh = axial[y0:y1, x0:x1]

    # Smooth image
    axenh_blur = cv2.GaussianBlur(axial, filter_kernel, 0)


    _, _, _, maxLoc = cv2.minMaxLoc(axenh_blur)
    seed = [[maxLoc[1], maxLoc[0]]]
    for i in range(max_iteration):
        aif_mask = region_grow_thresh(axenh_blur, seed.copy(), threshold)
        if aif_mask.sum() > 25 and i > 26:
            print(f"Segmentation converged at iteration {i}, threshold={threshold}")
            break
        threshold += 1
    

    # # Visualization
    plt.figure(figsize=(6,6))
    plt.imshow(axenh_blur, cmap="hot")
    if aif_mask is not None:
        plt.contour(aif_mask, colors="cyan", linewidths=1)
    plt.scatter([maxLoc[0]], [maxLoc[1]], c="blue", marker="x")  # seed point
    plt.title("Aorta segmentation")
    plt.show()
    #aif_mask_full = np.zeros_like(axial, dtype=np.uint8)
    #if aif_mask is not None:
        #aif_mask_full[y0:y1, x0:x1] = aif_mask[...,np.newaxis]

    return aif_mask #aif_mask_full

# def segment_for_leeds(axial):
#     cutRatio = 0.25            # crop window around image center
#     filter_kernel = (15, 15)   # gaussian blur
#     threshold = 2             # starting similarity tolerance
#     max_iteration = 25

#     # Crop around center
#     cy, cx = axial.shape[0]//2, axial.shape[1]//2
#     y0, y1 = int(cy - cy*cutRatio), int(cy + cy*cutRatio)
#     x0, x1 = int(cx - cx*cutRatio), int(cx + cx*cutRatio)
#     axenh = axial[y0:y1, x0:x1]

#     # Smooth image
#     axenh_blur = cv2.GaussianBlur(axenh, filter_kernel, 0)

#     # Find brightest spot (seed point)
#     _, _, _, maxLoc = cv2.minMaxLoc(axenh_blur)
#     seed = [[maxLoc[1], maxLoc[0]]]  # (row, col)

#     # Progressive region growing
#     aif_mask = None
#     for i in range(max_iteration):
#         aif_mask = region_grow_thresh(axenh_blur, seed.copy(), threshold)

#         if aif_mask.sum() > 25 and i > 13:  # stop once region is large enough
#             print(f"Segmentation converged at iteration {i}, threshold={threshold}")
#             break

#         threshold += 1  # relax tolerance 

#     # # # Visualization
#     # plt.figure(figsize=(6,6))
#     # plt.imshow(axenh_blur, cmap="hot")
#     # if aif_mask is not None:
#     #     plt.contour(aif_mask, colors="cyan", linewidths=1)
#     # plt.scatter([maxLoc[0]], [maxLoc[1]], c="blue", marker="x")  # seed point
#     # plt.title("Aorta segmentation")
#     # plt.show()
#     aif_mask_full = np.zeros_like(axial, dtype=np.uint8)
#     if aif_mask is not None:
#         aif_mask_full[y0:y1, x0:x1] = aif_mask[...,np.newaxis]

#     return aif_mask_full

# def segment_for_bari(axial):
#     cutRatio = 0.25            # crop window around image center
#     filter_kernel = (15, 15)   # gaussian blur
#     threshold = 1             # starting similarity tolerance
#     max_iteration = 150

#     # Crop around center
#     cy, cx = axial.shape[0]//2, axial.shape[1]//2
#     y0, y1 = int(cy - cy*cutRatio), int(cy + cy*cutRatio)
#     x0, x1 = int(cx - cx*cutRatio), int(cx + cx*cutRatio)
#     axenh = axial[y0:y1, x0:x1]

#     # Smooth image
#     axenh_blur = cv2.GaussianBlur(axenh, filter_kernel, 0)

#     # Find brightest spot (seed point)
#     _, _, _, maxLoc = cv2.minMaxLoc(axenh_blur)
#     seed = [[maxLoc[1], maxLoc[0]]]  # (row, col)

#     # Progressive region growing
#     aif_mask = None
#     for i in range(max_iteration):
#         aif_mask = region_grow_thresh(axenh_blur, seed.copy(), threshold)

#         if aif_mask.sum() > 30 and i > 90:  # stop once region is large enough
#             print(f"Segmentation converged at iteration {i}, threshold={threshold}")
#             break

#         threshold += 1  # relax tolerance gradually
    
    # import matplotlib.pyplot as plt
    # # # Visualization
    # plt.figure(figsize=(6,6))
    # plt.imshow(axenh_blur, cmap="hot")
    # if aif_mask is not None:
    #     plt.contour(aif_mask, colors="cyan", linewidths=1)
    # plt.scatter([maxLoc[0]], [maxLoc[1]], c="blue", marker="x")  # seed point
    # plt.title("Aorta segmentation")
    # plt.show()

        
    # aif_mask_full = np.zeros_like(axial, dtype=np.uint8)
    # if aif_mask is not None:
    #     aif_mask_full[y0:y1, x0:x1] = aif_mask[...,np.newaxis]

    # return aif_mask_full


# def Bari():

#     datapath = os.path.join(os.getcwd(), 'build', 'dce_3_mip', 'Bari', 'Patients')
#     destpath = os.path.join(os.getcwd(), 'build', 'dce_4_aortaseg', 'Bari', 'Patients')
#     os.makedirs(destpath, exist_ok=True)

#     logging.basicConfig(
#         filename=os.path.join(destpath, 'error.log'),
#         filemode='w',
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )

#     database = db.series(datapath)
#     DCE_mip = [entry for entry in database if entry[3][0].strip().lower() == 'dce_3_mip']

#     pat_series = []
    
    


#     for study in DCE_mip:
#         try:
#             bari_add_series_name(study[1], pat_series)
#             mask_path = [destpath, study[1], ('Baseline', 0)]
            
#             dce_mask = mask_path + [(pat_series[-1] + "aortaseg", 0)]
#             if dce_mask in db.series(mask_path):
#                  continue
            
#             mip = db.volume(study)
#             print(mip.shape)
#             aif_mask = segment_for_bari(mip.values)
#             print(aif_mask)
#             aif_mask = aif_mask.astype(np.uint16)

#             db.write_volume((aif_mask, mip.affine), dce_mask, ref=study)

#         except Exception as e:
#             logging.error(f"Study {study[1]} cannot be assesed: {e}")

# def Leeds():

#     datapath = os.path.join(os.getcwd(), 'build', 'dce_3_mip', 'Leeds', 'Patients')
#     destpath = os.path.join(os.getcwd(), 'build', 'dce_4_aortaseg', 'Leeds', 'Patients')
#     os.makedirs(destpath, exist_ok=True)

#     logging.basicConfig(
#         filename=os.path.join(destpath, 'error.log'),
#         filemode='w',
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )

#     database = db.series(datapath)
#     DCE_mip = [entry for entry in database if entry[3][0].strip().lower() == 'dce_3_mip']

#     pat_series = []
    
    


#     for study in DCE_mip:
#         try:
#             leeds_add_series_name(study[1], pat_series)
#             mask_path = [destpath, study[1], ('Baseline', 0)]
#             dce_mask = mask_path + [(pat_series[-1] + "aortaseg", 0)]
#             if dce_mask in db.series(mask_path):
#                 continue
#             mip = db.volume(study)
#             print(mip.shape)
#             aif_mask = segment_for_leeds(mip.values)
#             print(aif_mask)
#             aif_mask = aif_mask.astype(np.uint16)
#             db.write_volume((aif_mask, mip.affine), dce_mask, ref=study)

#         except Exception as e:
#             logging.error(f"Study {study[1]} cannot be assesed: {e}")

def sheffield_patients(segtype=None):

    if segtype == None:
        #manual segmentation
        datapath = os.path.join(os.getcwd(), 'build', 'dce_3_mip', 'Sheffield', 'Patients')
    elif segtype == 'manual':
        datapath = os.path.join(os.getcwd(), 'build', 'dce_3_mip', 'Sheffield', 'Patients')
    elif segtype == 'auto':
        #auto segmentation
        datapath = os.path.join(os.getcwd(), 'build', 'dce_2_data', 'Sheffield', 'Patients')


    destpath = os.path.join(os.getcwd(), 'build', 'dce_4_aortaseg', 'Sheffield', 'Patients')
    os.makedirs(destpath, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(destpath, 'error.log'),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    database = db.series(datapath)
    if segtype == None:
        #manual segmentation
        DCE_mip = [entry for entry in database if entry[3][0].strip().lower() == 'dce_3_mip']
    elif segtype == 'manual':
        DCE_mip = [entry for entry in database if entry[3][0].strip().lower() == 'dce_3_mip']
    elif segtype == 'auto':
        #auto segmentation
        DCE_mip = [entry for entry in database if entry[3][0].strip().lower() == 'dce_1_coronal_kidneys'.lower()]    


    pat_series = []
    
    


    for study in DCE_mip:
        try:
            sheffield_add_series_name(study[1], pat_series)
            mask_path = [destpath, study[1], ('Baseline', 0)]

            if segtype == None:
                #manual segmentation
                dce_mask = mask_path + [(pat_series[-1] + "aortaseg", 0)]
            elif segtype == 'manual':
                dce_mask = mask_path + [(pat_series[-1] + "aortaseg", 0)]
            elif segtype == 'auto':
                #auto segmentation
                dce_mask = mask_path + [(pat_series[-1] + "aortasegauto", 0)]

            if dce_mask in db.series(mask_path):
                continue

            if segtype == None:
                mip = db.volume(study)
            elif segtype == 'manual':
                mip = db.volume(study)
            elif segtype == 'auto':
                #auto segmentation
                mip = db.volume(study, 'TemporalPositionIdentifier')     
            print(mip.shape) 
            if segtype == None:
                #manual segmentation
                aif_mask = segment_for_sheffield(mip.values)
                aif_mask = aif_mask.astype(np.uint16)
            elif segtype == 'manual':
                aif_mask = segment_for_sheffield(mip.values)
                aif_mask = aif_mask.astype(np.uint16)
            elif segtype == 'auto':
                #auto segmentation
                aif_mask = autosegment_aorta(mip)
            
            db.write_volume((aif_mask, mip.affine), dce_mask, ref=study)

        except Exception as e:
            logging.error(f"Study {study[1]} cannot be assesed: {e}")

def autosegment_aorta(series_vol=None):

    # --- 1. Draw lungs on baseline image
    ref_vol = series_vol

    # Create label and save
    label_vol = miblab.totseg(ref_vol, cutoff=0.01, task='total_mr', device='cpu')

    mask_arr = np.where(label_vol.values==23, 1, 0).astype(int)
    return mask_arr



if __name__ == '__main__':
    # Bari()
    # Leeds()
    sheffield_patients(segtype='auto')
