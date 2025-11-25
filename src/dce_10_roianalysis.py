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

#Helper: Add series name
def bari_add_series_name(folder, all_series: list):
    new_series_name = "DCE_10_"
    all_series.append(new_series_name)
    return new_series_name

#Helper: Gaussian filter
def fill_gaps_zonly(input_array, input_geom, sigma_z=1.5):
    """
    Fill gaps in 3D array by Gaussian smoothing only in Z axis.
    Preserves XY detail.

    Parameters:
        input_array (np.ndarray): 3D array with gaps.
        input_geom (np.ndarray): Binary mask (1=valid, 0=gap).
        sigma_z (float): Amount of blur along Z-axis.

    Returns:
        np.ndarray: Filled array.
    """
    input_array = np.nan_to_num(input_array)
    weights = input_geom.astype(float)

    # Set sigma to (0, 0, sigma_z) to only smooth along Z
    sigma = (0, 0, sigma_z)

    smoothed = gaussian_filter(input_array * weights, sigma=sigma)
    norm = gaussian_filter(weights, sigma=sigma)

    with np.errstate(divide='ignore', invalid='ignore'):
        output = smoothed / norm
        output[norm == 0] = 0

    return output

#Helper: Slice like reference before filling gaps
def fill_slice_gaps(series, ref, mask=None):

    ref_volume = ref
    #mask_arr = mask.values
    contrast_limits = [0, 300]
    input_array = np.zeros(ref_volume.shape)
    input_count = np.zeros(ref_volume.shape)
    for slice_vol in series:
        # viewer = napari.Viewer()
        # viewer.add_image(slice_vol.values.T, contrast_limits=contrast_limits)
        # napari.run()
        slice_vol_on_ref = slice_vol.slice_like(ref_volume)
        # viewer = napari.Viewer()
        # viewer.add_image(slice_vol_on_ref.values.T, contrast_limits=contrast_limits)
        # napari.run()
        input_array += slice_vol_on_ref.values      
        input_count[slice_vol_on_ref.values > 0] += 1
    nozero = input_count > 0
    # viewer = napari.Viewer()
    # viewer.add_image(input_array)
    # napari.run()
    input_array[nozero] /= input_count[nozero]
    # viewer = napari.Viewer()
    # viewer.add_image(input_array.T, contrast_limits=contrast_limits)
    # viewer.add_labels(mask_arr.T.astype(int))

    input_geom = np.zeros(ref_volume.shape)
    input_geom[nozero] = 1
    print('Filling slice gaps...')
    output_array = fill_gaps_zonly(input_array, input_geom)

    # viewer = napari.Viewer() 
    # viewer.add_image(output_array.T)
    # viewer.add_labels(mask.values.T.astype(int))
    # napari.run()



    return output_array

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
        colors = [cmap(i)[:3] + (0.6,) for i in np.linspace(0, 1, n)]  # Set alpha to 0.6 for transparency

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


#Helper: K-means for CM segmentation
def kmeans(features, mask=None, roi=None, n_clusters=2, multiple_series=False, normalize=True, return_features=False, site=None, case_id=None, batch_no=None):
    """
    Labels structures in an image
    
    Wrapper for sklearn.cluster.KMeans function. 

    Parameters
    ----------
    input: list of dbdicom series (one for each feature)
    mask: optional mask for clustering
    
    Returns
    -------
    clusters : list of dbdicom series, with labels per cluster.
    """
    pat_series = []
    dest_dir_base = [os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients"]
    if batch_no is not None:
        dest_dir_base.append(f"Batch_{batch_no}")
    dest_dir = os.path.join(*dest_dir_base)
    database = [dest_dir, case_id, ('Baseline', 0)]
    masks_base = [os.getcwd(), 'build', 'kidneyvol_3_edit', site, "Patients"]
    if batch_no is not None:
        masks_base.append(f"Batch_{batch_no}")
    masks_database = os.path.join(*masks_base)
    series_name = bari_add_series_name(case_id, pat_series)

    study_desc = [s for s in db.series(masks_database) if s[1] == case_id]


    # If a mask is provided, map it onto the reference feature and 
    # extract the indices of all pixels under the mask
    if mask is not None:
        mask_array = mask.values
        # mask_array, _ = vreg.mask_array(mask, on=features[0], dim='AcquisitionTime')
        mask_array = np.ravel(mask_array)
        mask_indices = tuple(mask_array.nonzero())

    # # Ensure all the features are in the same geometry as the reference feature
    # features = scipy.overlay(features)

    # Create array with shape (n_samples, n_features) and mask if needed.
    array = []
    for series in features:
        arr = series.values
        shape = arr.shape 
        arr = np.ravel(arr)
        if mask is not None:
            arr = arr[mask_indices]
        #if normalize:
        #    arr = (arr-np.mean(arr))/np.std(arr)
        array.append(arr)
    array = np.vstack(array).T

    # Perform the K-Means clustering.
    print('Clustering. Please be patient - this is hard work..')
    if normalize:
        X = StandardScaler().fit_transform(array)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=3, verbose=1).fit(X)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=3, verbose=1).fit(array)

    # Create an output array for the labels
    if mask is not None:
        print('Creating output array..')
        output_array = np.zeros(shape)
        output_array = np.ravel(output_array)
        output_array[mask_indices] = 1+kmeans.labels_ 
    else:
        output_array = 1+kmeans.labels_
    output_array = output_array.reshape(shape)

    aff = db.volume(study_desc[0]).affine

    # Save the results in DICOM
    print('Saving clusters..')
    if multiple_series:
        # Save each cluster as a separate mask
        clusters = []
        for cluster in range(1,1+n_clusters):
            array_cluster = np.zeros(output_array.shape)
            array_cluster[output_array == cluster] = 1  
            if roi == 'lk':        
                cluster_desc = database + [(series_name + f"kmeans_cluster_{roi}_{str(cluster)}", 0)]
            if roi == 'rk':        
                cluster_desc = database + [(series_name + f"kmeans_cluster_{roi}_{str(cluster)}", 0)]
            db.write_volume((array_cluster, aff), cluster_desc)
            series_cluster = db.volume(cluster_desc)
            #_reset_window(series_cluster, array_cluster)
            clusters.append(series_cluster)
    else:
        print('only one series?')
        # Save the label array in a single series
        # clusters = features[0].new_sibling(SeriesDescription = 'KMeans')
        # clusters.set_array(output_array, headers, pixels_first=True)
        # _reset_window(clusters, output_array)

    # If requested, return features (mean values over clusters + size of cluster)
    if return_features: # move up
        cluster_features = []
        for cluster in range(1,1+n_clusters):
            vals = []
            #locs = (output_array.ravel() == cluster)
            locs = (1+kmeans.labels_ == cluster)
            for feature in range(array.shape[1]):
                val = np.mean(array[:,feature][locs])  
                vals.append(val)
            vals.append(np.sum(locs))
            cluster_features.append(vals) 
        return clusters, cluster_features   

    return clusters

#______________________________________________MAIN PROTOCOL__________________________________#
#Step 1: Get data and create a table
def get_data(site, batch_no=None):
    base = [os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients"]
    if batch_no is not None:
        base.append(f"Batch_{batch_no}")
    base_dir = os.path.join(*base)
    os.makedirs(base_dir, exist_ok=True)
    masks_base = [os.getcwd(), 'build', 'kidneyvol_3_edit', site, "Patients"]
    if batch_no is not None:
        masks_base.append(f"Batch_{batch_no}")
    masks_dir = os.path.join(*masks_base)
    dce_maps_base = [os.getcwd(), 'build', 'dce_9_coreg_dce2dixon', site, "Patients"]
    if batch_no is not None:
        dce_maps_base.append(f"Batch_{batch_no}")
    dce_maps_dir = os.path.join(*dce_maps_base)


    # Load series from databases
    masks_database = db.series(masks_dir)
    dce_database = db.series(dce_maps_dir)
    
    if site =='Bordeaux':
        masks_database = [study for study in masks_database if study[2][0] == 'Baseline']


    #Filter out dce maps to align and make sure it is in the order: rpf, avd, mtt
    rk_maps2fill_database = [
        entry for entry in dce_database 
        if entry[3][0].strip().lower() in
        ('dce_9_rpf_rk_aligned', 'dce_9_avd_rk_aligned', 'dce_9_mtt_rk_aligned') 
    ]

    lk_maps2fill_database = [
        entry for entry in dce_database 
        if entry[3][0].strip().lower() in
        ('dce_9_rpf_lk_aligned', 'dce_9_avd_lk_aligned',  'dce_9_mtt_lk_aligned', ) 
    ]


    rk_mdr_database = [
    entry for entry in dce_database 
    if entry[3][0].strip().lower() in
    ('dce_9_mdr_rk_aligned') 
    ]

    lk_mdr_database = [
    entry for entry in dce_database 
    if entry[3][0].strip().lower() in
        ('dce_9_mdr_lk_aligned') 
    ]


    # Get unique case identifiers
    case_ids = set(entry[1] for entry in rk_maps2fill_database)



    images_and_masks = []
    for case_id in sorted(case_ids):
        # Find corresponding mask study
        mask_path = next((s for s in masks_database if s[1] == case_id), None)
        if mask_path is None:
            print(f"Skipping case {case_id}, study not found in mask database.")
            continue
        
        rk_dce_paths = [s for s in rk_maps2fill_database if s[1] == case_id]
        if rk_dce_paths is None:
            print(f"Skipping case {case_id}, DCE moco series not found.") 
        
        lk_dce_paths = [s for s in lk_maps2fill_database if s[1] == case_id]
        if lk_dce_paths is None:
            print(f"Skipping case {case_id}, DCE moco series not found.") 

        rk_mdr_path = [s for s in rk_mdr_database if s[1] == case_id]
        if rk_dce_paths is None:
            print(f"Skipping case {case_id}, DCE moco series not found.") 
        
        lk_mdr_path = [s for s in lk_mdr_database if s[1] == case_id]
        if lk_dce_paths is None:
            print(f"Skipping case {case_id}, DCE moco series not found.")


        #create data table 
        images_and_masks.append({
            'case': case_id,
            'mask_path': mask_path,            
            'rk_dce_maps': rk_dce_paths,          
            'lk_dce_maps': lk_dce_paths,
            'rk_mdr_path': rk_mdr_path,
            'lk_mdr_path': lk_mdr_path

        })



    # Save the results to file
    output_path = os.path.join(base_dir, f'{site}_images_masks_table.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(images_and_masks, f)
    
    return images_and_masks


#Step 2: Fill DCE Gaps

def fill_dce_maps(site, batch_no=None):
    images_and_masks = get_data(site, batch_no=batch_no)
    pat_series = []
    for entry in tqdm(images_and_masks, desc='Filling DCE Maps + Building C/M masks', unit='case'):
        case_id = entry['case']
        dest_base = [os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients"]
        if batch_no is not None:
            dest_base.append(f"Batch_{batch_no}")
        dest_dir = os.path.join(*dest_base)
        database = [dest_dir, case_id, ('Baseline', 0)]
        series_name = bari_add_series_name(case_id, pat_series)


        rk_rpf_clean = database + [(series_name + "rk_rpf_gaps_filled", 0)]
        lk_rpf_clean = database + [(series_name + "lk_rpf_gaps_filled", 0)]
        rk_avd_clean = database + [(series_name + "rk_avd_gaps_filled", 0)]
        lk_avd_clean = database + [(series_name + "lk_avd_gaps_filled", 0)]
        rk_mtt_clean = database + [(series_name + "rk_mtt_gaps_filled", 0)]
        lk_mtt_clean = database + [(series_name + "lk_mtt_gaps_filled", 0)]

        clean_series = [rk_rpf_clean, lk_rpf_clean,
                        rk_avd_clean, lk_avd_clean,
                        rk_mtt_clean, lk_mtt_clean]
                
        if all(x in db.series(database) for x in clean_series):
            roi = ['lk', 'rk']
            for kidney in roi:
                cortex_medulla(site, case_id, roi=kidney, batch_no=batch_no)
                continue

        # Dictonary
        mask_path = entry['mask_path']   
        rk_dce_map_paths = entry['rk_dce_maps']
        lk_dce_map_paths = entry['lk_dce_maps']

        clean_maps = [  rk_rpf_clean, lk_rpf_clean,
                        rk_avd_clean, lk_avd_clean,
                        rk_mtt_clean, lk_mtt_clean]
                
        if all(x in db.series(database) for x in clean_maps):
            continue

                
        rk_dce_volumes = []
        for map_path in rk_dce_map_paths:
            try:
                rk_dce_volume = db.volumes_2d(map_path)
                rk_dce_volumes.append(rk_dce_volume)
            except Exception as e:
                print(f'cannot load {map_path[3][0]} for {case_id}: {e}')
                continue

        lk_dce_volumes = []
        for map_path in lk_dce_map_paths:
            try:
                lk_dce_volume = db.volumes_2d(map_path)
                lk_dce_volumes.append(lk_dce_volume)
            except Exception as e:
                print(f'cannot load {map_path[3][0]} for {case_id}: {e}')
                
        
        mask = db.volume(mask_path)
        ref_volume = mask
        
        arr = mask.values 
        lk = (arr==1)
        rk = (arr==2)
        

        rk_outputs = []
        for series in rk_dce_volumes:
            output_series = fill_slice_gaps(series, ref_volume, mask=rk)
            rk_outputs.append(output_series)
        
        lk_outputs = []
        for series in lk_dce_volumes:
            output_series = fill_slice_gaps(series, ref_volume, mask=lk)
            lk_outputs.append(output_series)

        
    
        print('Building RPF Volume...')
        try:
            if rk_rpf_clean not in db.series(database):
                db.write_volume((rk_outputs[0], mask.affine), rk_rpf_clean, ref=rk_dce_map_paths[0])
        except Exception as e:
            print(f'cannot build rk_rpf vol for {case_id}: {e}')
    
        try:
            if lk_rpf_clean not in db.series(database):
                db.write_volume((lk_outputs[0], mask.affine), lk_rpf_clean, ref=lk_dce_map_paths[0])
        except Exception as e:
            print(f'cannot build lk_rpf vol for {case_id}: {e}')
                
        print('Building AVD Volume...')
        try:
            if rk_avd_clean not in db.series(database):
                db.write_volume((rk_outputs[1], mask.affine), rk_avd_clean, ref=rk_dce_map_paths[1])
        except Exception as e:
            print(f'cannot build rk_avd vol for {case_id}: {e}')        
        
        try:
            if lk_avd_clean not in db.series(database):
                db.write_volume((lk_outputs[1], mask.affine), lk_avd_clean, ref=lk_dce_map_paths[1])        
        except Exception as e:
            print(f'cannot build lk_avd vol for {case_id}: {e}')  

        print('Building MTT Volume...')
        try:
            if rk_mtt_clean not in db.series(database):
                db.write_volume((rk_outputs[2], mask.affine), rk_mtt_clean, ref=rk_dce_map_paths[2])    
        except Exception as e:
            print(f'cannot build rk_mtt vol for {case_id}: {e}')             
        
        try:
            if lk_mtt_clean not in db.series(database):
                db.write_volume((lk_outputs[2], mask.affine), lk_mtt_clean, ref=lk_dce_map_paths[2])  
        except Exception as e:
            print(f'cannot build lk_mtt vol for {case_id}: {e}') 

        roi = ['lk', 'rk']
        for kidney in roi:
            cortex_medulla(site, case_id, roi=kidney, batch_no=batch_no)



#Step 3: Cortex Medulla Mask Creation
def cortex_medulla(site, case_id, roi=None, batch_no=None): 

    masks_base = [os.getcwd(), 'build', 'kidneyvol_3_edit', site, "Patients"]
    if batch_no is not None:
        masks_base.append(f"Batch_{batch_no}")
    masks_dir = os.path.join(*masks_base)
    mask_database = db.series(masks_dir)
    maps_path_base = [os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients"]
    if batch_no is not None:
        maps_path_base.append(f"Batch_{batch_no}")
    maps_dir = os.path.join(*maps_path_base)
    maps_database = db.series(maps_dir)
    maps = [entry for entry in maps_database if entry[3][0].strip().lower() in
        (f'dce_10_{roi}_rpf_gaps_filled', f'dce_10_{roi}_avd_gaps_filled',  f'dce_10_{roi}_mtt_gaps_filled')] 
    
    pat_series = []
    dest_base = [os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients"]
    if batch_no is not None:
        dest_base.append(f"Batch_{batch_no}")
    dest_dir = os.path.join(*dest_base)
    os.makedirs(dest_dir, exist_ok=True)    
    
    mask_png_dir = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients", 'overlays')
    os.makedirs(mask_png_dir, exist_ok=True)
    mask_png = mask_png_dir + f'{case_id}_{roi}.png'
    database = [dest_dir, case_id, ('Baseline', 0)]
    series_name = bari_add_series_name(case_id, pat_series)


    both_kidneys = next((m for m in mask_database if m[1] == case_id), None)
    if both_kidneys:
        both_kidneys = db.volume(both_kidneys)
        both_kidneys_arr = both_kidneys.values 
        aff = both_kidneys.affine

        rk = (both_kidneys_arr==1)
        lk = (both_kidneys_arr==2)

        if roi == 'rk':
            mask_roi = vreg.volume(rk, aff)
        elif roi =='lk':
            mask_roi = vreg.volume(lk, aff)

        if maps:
            try:
                output = []
                for path in maps:
                    try:
                        volume = db.volume(path)
                        output.append(volume)
                    except Exception as e:
                        print(f'cannot load {path[3][0]} for case {case_id}: {e}')
                        continue
                
                clusters, cluster_features = kmeans(output, mask_roi, roi=roi, n_clusters=3, multiple_series=True, return_features=True, site=site, case_id=case_id, batch_no=batch_no)
                # Background = cluster with smallest AVD
                background = np.argmin([c[1] for c in cluster_features])
                # Cortex = cluster with largest RPF 
                cortex = np.argmax([c[0] for c in cluster_features]) 
                # Medulla = cluster with largest MTT 
                medulla = np.argmax([c[2] for c in cluster_features])
                # Check
                remainder = {0,1,2} - {background, cortex, medulla}
                if len(remainder) > 0:
                    raise ValueError('Problem separating cortex and medulla: identified clusters do not have the expected values.')
            except Exception as e:
                print(f'cannot create mask: {e}')
            
            cortex_lk = database + [(series_name + 'LKC', 0)] 
            medulla_lk = database + [(series_name + 'LKM', 0)]   
            
            lk_cm_desc = database + [(series_name + 'LCM', 0)]

            cortex_rk = database + [(series_name + 'RKC', 0)] 
            medulla_rk = database + [(series_name + 'RKM', 0)]   
             
            rk_cm_desc = database + [(series_name + 'RCM', 0)]

            aff = both_kidneys.affine
            try:
                if roi == 'lk':
                    db.write_volume(clusters[cortex], cortex_lk)
                    db.write_volume(clusters[medulla], medulla_lk)
                
                    cm = np.zeros_like(clusters[background].values)
                    cm[clusters[background].values > 0] = 0
                    cm[clusters[cortex].values > 0] = 3
                    cm[clusters[medulla].values > 0] = 4
                    cm_vol = vreg.volume(cm, aff)
                    mosaic_overlay(output[0], rois=cm_vol, file=mask_png, show=True)
                    db.write_volume(cm_vol, lk_cm_desc)
                
                elif roi == 'rk':
                    db.write_volume(clusters[cortex], cortex_rk)
                    db.write_volume(clusters[medulla], medulla_rk)
                    
                    cm = np.zeros_like(clusters[background].values)
                    cm[clusters[background].values > 0] = 0
                    cm[clusters[cortex].values > 0] = 1
                    cm[clusters[medulla].values > 0] = 2
                    cm_vol = vreg.volume(cm, aff)
                    mosaic_overlay(db.volume(maps[0]), rois=cm_vol, file=mask_png, show=True)
                    # vplot.overlay_2d_cm(db.volume(maps[0]), mask=cm_vol, save_path=mask_png, show=True)
                    db.write_volume(cm_vol, rk_cm_desc)
            except Exception as e:
                tqdm.write(f'Cannot create {roi} volume for {case_id}: {e}')





#Step 4: Extract cortex and medulla input function and write to dmr
def extract_mdr_input_function(site, roi=None, batch_no=None):
    
    # Aligned MDR
    img_database = [os.getcwd(), 'build', 'dce_9_coreg_dce2dixon', site, "Patients"]
    if batch_no is not None:
        img_database.append(f"Batch_{batch_no}")
    img_dir = os.path.join(*img_database)

    # Whole Kidneys
    mask_database = [os.getcwd(), 'build', 'kidneyvol_3_edit', site, "Patients"]
    if batch_no is not None:
        mask_database.append(f"Batch_{batch_no}")
    mask_dir = os.path.join(*mask_database)
    
    # Cortex + Medulla 
    # data_dir = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients")
    # database = db.series(data_dir)

    dest_database = [os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients"]
    if batch_no is not None:
        dest_database.append(f"Batch_{batch_no}")
    dest_dir = os.path.join(*dest_database)

    # Logging setup
    logging.basicConfig(
    filename=os.path.join(dest_dir, 'error.log'),
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

    #______FUNCTIONS______#
    def extract_times(roi=None):
        database = db.series(img_dir)       
        mdr_aligned = [entry for entry in database if entry[3][0].strip().lower() == f'DCE_9_mdr_{roi}_aligned'.lower()]
        time_list = []
        for series in mdr_aligned:
            case_id = series[1]
            if site in ('Sheffield', 'Bordeaux'):
                try:
                    vols = db.volumes_2d(series, dims=['AcquisitionTime'])
                    for vol in vols:
                        time = vol.coords
                        time = [t for sublist in time for t in sublist]
                        time = time - time[0]
                        time_list.append((case_id, vols, time))
                        break     
                except Exception as e:
                    logging.error(f'''Cannot load {roi} mdr volume for case {case_id}: {e}. Skipping''' )        
                    continue
        return time_list 

    # def cortex_on_mdr(cortex_vol=None, roi=None):

    # study = cortex_vol[0]
    # mdr_aligned = [entry for entry in database if entry[3][0].strip().lower() == f'dce_9_mdr_{roi}_aligned'.lower()]
    # for series in mdr_aligned:
    #     vols = db.volumes_2d(series, dims=['AcquisitionTime'])
    #     cortex_on_time_vols = []
    #     for vol in vols:
    #         vol_t = vreg.volume(vol.values[:,:,0,:], vol.affine) 
    #         cortex_on_time_vol = cortex_vol[1].slice_like(vol_t)
    #         cortex_on_time_vols.append(cortex_on_time_vol.values)
    #     cortex_on_vol = []
    #     mask = (np.stack(cortex_on_time_vols, axis=2)).squeeze()
    #     cortex_on_vol.append((study, mask))
    # return cortex_on_vol


    def extract_labels(roi=None, times_inventory=None):

        database = db.series(mask_dir)        
        
        if roi == 'LK':
            #inventory = [entry for entry in database if entry[3][0].strip().lower() == 'dce_10_lkm'.lower()]
            inventory = [entry for entry in database if entry[3][0].strip().lower() == 'kidney_masks'.lower()]
            if site == 'Bordeaux':
                masks_inventory = [entry for entry in inventory if entry[2][0] == 'Baseline']
        elif roi == 'RK':
            #inventory = [entry for entry in database if entry[3][0].strip().lower() == 'dce_10_rkm'.lower()]
            inventory = [entry for entry in database if entry[3][0].strip().lower() == 'kidney_masks'.lower()]
            if site == 'Bordeaux':
                masks_inventory = [entry for entry in inventory if entry[2][0] == 'Baseline']
        
        labels = []
        for case_id, mdr_vols, times in times_inventory:
            mask = None
            mask = next(m for m in masks_inventory if m[1] == case_id)
            if mask is None:
                print(f'No mask found for {case_id}. Skipping')
                continue
            vol = db.volume(mask)
            arr = vol.values
            aff = vol.affine 
            lk = (arr == 1)
            lk_voxels = np.sum(lk)
            rk = (arr == 2)
            rk_voxels = np.sum(rk)
            voxel_size = 1.25*1.25*1.5 # pixel_spacing x slice_thickness
            lk_vol_size = lk_voxels*voxel_size/1000 #convert to ml
            rk_vol_size = rk_voxels*voxel_size/1000
            if roi == 'LK':
                mask_vol = vreg.volume(lk, aff)
            elif roi == 'RK':
                mask_vol = vreg.volume(rk, aff)
            labels.append((case_id, mdr_vols, times, mask_vol, lk_vol_size, rk_vol_size))
        return labels

    def mask_on_mdr(inventory=None):
        new_inv = []
        for case_id, mdr_vols, times, mask_vol, lk_size, rk_size in tqdm(inventory, desc=f'Batch No {batch_no}: Building mdr mask', unit='case'):
            mask_on_time_vols = []
            for vol in mdr_vols:
                vol_t = vreg.volume(vol.values[:,:,0,:], vol.affine) 
                mask_on_time_vol = mask_vol.slice_like(vol_t)
                mask_on_time_vols.append(mask_on_time_vol.values)
            mdr_mask = np.stack(mask_on_time_vols, axis=2)
            new_inv.append((case_id, mdr_vols, mdr_mask, times, lk_size, rk_size))
            print('MDR mask created! Adding to inventory')
        return new_inv

    print('Extracting Times...')
    times_inventory = extract_times(roi=roi)

    print('Extracting Labels to Align...')
    inventory_w_labels = extract_labels(roi=roi, times_inventory=times_inventory)
    
    print('Matching Mask Shape with MDR...')
    inventory = mask_on_mdr(inventory_w_labels)

    # for label in tqdm(labels, desc=f'Processing {roi} Cortex Masks', unit="label"):
    #     masks_on_mdr = mask_on_mdr(case_id, cortex_vol=vol, roi=roi)
    


    if not isinstance(inventory, Iterable) or isinstance(inventory, str):
        inventory = [inventory]

    
    print(f'Getting {roi} values')
    mdr_mask = None
    for case_id, mdr_vols, mdr_mask, times, lk_size, rk_size in tqdm(inventory, unit='case'):
        if mdr_mask is None:
            print(f"""
                    Could not find mdr mask in inventory. See below.
                    - mdr_vol shape: {mdr_vols.shape}
                    - mdr_mask: {mdr_mask}
                """)
        
        dmr_dir = dest_dir + '/DMR'
        os.makedirs(dmr_dir, exist_ok=True)
        
        if site in ('Sheffield', 'Bordeaux'):
            mdr_vol = []
            for vol in mdr_vols:
                vol_arr = vol.values
                mdr_vol.append(vol_arr)
            mdr_vol = (np.stack(mdr_vol, axis=2)).squeeze()

        # make sure mask is 3D (x, y, z)
        if mdr_mask.ndim == 4:
            mask = mdr_mask[..., 0] 

        elif mask.ndim == 3:
            mask = mdr_mask
            
        mask = mask.astype(bool)
        slice_means = []
            # compute slice-wise mean intensities for each timepoint
        for t in range(mdr_vol.shape[-1]):  # loop over time frames
            t_means = []
            for z in range(mdr_vol.shape[2]):  # loop over slices
                masked_values = mdr_vol[:, :, z, t][mask[:, :, z]]
                t_means.append(masked_values.mean() if masked_values.size > 0 else np.nan)
            slice_means.append(t_means)

        # average across slices (shape: n_timepoints,)
        intensities = np.nanmean(slice_means, axis=1)

        # Ensure same length as curve
        if len(times) != len(intensities):
            print(f"Length mismatch for {case_id}: times={len(times)}, curve={len(intensities)}")
            continue
    
        # Convert curve to plain floats
        intensities = [float(val) for val in intensities]
    
        # Build dataframe
        df = pd.DataFrame({
        "Time (s)": times,
        f"{roi}IF": intensities
        })

        if roi == 'LK':
            kid_vol = lk_size
            r = 'kidney_left'
        elif roi == 'RK':
            kid_vol = rk_size
            r = 'kidney_right'
        study = 'Baseline'

        dmr_file = os.path.join(dmr_dir, f"{case_id}_{r}")
        
        dmr_zip = dmr_file + '.dmr.zip'
        if os.path.exists(dmr_zip):
            print(f'{case_id}_{r} IF DMR  already in folder! Skipping')
            continue

        dmr = {'data':{}, 'pars':{}, 'rois':{}}
        dmr['rois'][(f"{case_id}_{r}", study, 'time')] = times
        dmr['rois'][(f"{case_id}_{r}", study, 'signal')] = intensities
        dmr['pars'][(f"{case_id}_{r}", study, 'field_strength')] = 3
        dmr['pars'][(f"{case_id}_{r}", study, 'agent')] = 'gadoterate' 
        dmr['pars'][(f"{case_id}_{r}", study,  'n0')] = 15
        dmr['pars'][(f"{case_id}_{r}", study, 'TR')] = 0.002
        dmr['pars'][(f"{case_id}_{r}", study, 'FA')] = 10
        dmr['pars'][(f"{case_id}_{r}", study, 'vol')] = kid_vol
        dmr['pars'][(f"{case_id}_{r}", study,  'T1')] = 1.4
        
        dmr['data'][('time')] = ['Acquisition time', 'sec', 'float']
        dmr['data'][('signal')] = ['Average signal intensity', 'a.u.', 'float']
        dmr['data'][('field_strength')] = ['B0 magnetic field strength', 'T', 'float']
        dmr['data'][('agent')] = ['Contrast agent generic name', '', 'str']
        dmr['data'][('n0')] = ['Number of precontrast acquisition', '', 'int']
        dmr['data'][('TR')] = ['Repetition Time', 'sec', 'float']
        dmr['data'][('FA')] = ['Flip angle', 'deg', 'float']
        dmr['data'][('vol')] = ['Kidney volume', 'mL', 'float']
        dmr['data'][('T1')] = ['Kidney T1 relaxation time', 'sec', 'float']
        pydmr.write(dmr_file, dmr)
        print(f'{case_id}_{r} input function DMR saved to folder!')
    


if __name__ == '__main__':
    #get_data('Bari')
    batch_no = [2,3,4,5,6,7,8,9]
    for no in batch_no:
        #fill_aligned_mdr('Bordeaux', batch_no=no)
        roi = ['RK', 'LK']
        for m in roi:
        # #     cortex_medulla('Bari', study='1128_003', roi=m)
            extract_mdr_input_function('Bordeaux', roi=m, batch_no=no)