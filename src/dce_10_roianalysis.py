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

    ref_volume = mask
    mask_arr = mask.values
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

#Helper: K-means for CM segmentation
def kmeans(features, mask=None, roi=None, n_clusters=2, multiple_series=False, normalize=True, return_features=False, site=None, study=None):
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
    dest_dir = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients")
    database = [dest_dir, study, ('Baseline', 0)]
    mask_database = os.path.join(os.getcwd(), 'build', 'kidneyvol_3_edit', site, "Patients")
    series_name = bari_add_series_name(study, pat_series)

    study_desc = [s for s in db.series(mask_database) if s[1] == study]


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
def get_data(site):
    base_dir = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients")
    os.makedirs(base_dir, exist_ok=True)
    masks_dir = os.path.join(os.getcwd(), 'build', 'kidneyvol_3_edit', site, "Patients")
    dce_maps_dir = os.path.join(os.getcwd(), 'build', 'dce_9_coreg_dce2dixon', site, "Patients")

    # Load series from databases
    masks_database = db.series(masks_dir)
    dce_database = db.series(dce_maps_dir)


    #Filter out dce maps to align and make sure it is in the order: rpf, avd, mtt
    rk_maps2fill_database = [
        entry for entry in dce_database 
        if entry[3][0].strip().lower() in
        ('dce_9_rpf_rk_aligned', 'dce_9_avd_rk_aligned', 'dce_9_mtt_rk_aligned' ) 
    ]

    lk_maps2fill_database = [
        entry for entry in dce_database 
        if entry[3][0].strip().lower() in
        ('dce_9_rpf_lk_aligned', 'dce_9_avd_lk_aligned',  'dce_9_mtt_lk_aligned', ) 
    ]


    # Get unique case identifiers
    case_id = set(entry[1] for entry in rk_maps2fill_database)

    images_and_masks = []
    for case_id in sorted(case_id):
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



        #create data table 
        images_and_masks.append({
            'case': case_id,
            'mask_path': mask_path,            
            'rk_dce_maps': rk_dce_paths,          
            'lk_dce_maps': lk_dce_paths

        })



    # Save the results to file
    output_path = os.path.join(base_dir, f'{site}_images_masks_table.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(images_and_masks, f)
    
    return images_and_masks


# Step 2: Main Protocol: Filling Gaps
def fillgaps(site):
    
    images_and_masks = get_data(site)
    pat_series = []
    for entry in images_and_masks:
        study = entry['case']
        dest_dir = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients")
        database = [dest_dir, study, ('Baseline', 0)]
        series_name = bari_add_series_name(study, pat_series)


        rk_rpf_clean = database + [(series_name + "rk_rpf_gaps_filled", 0)]
        lk_rpf_clean = database + [(series_name + "lk_rpf_gaps_filled", 0)]
        rk_avd_clean = database + [(series_name + "rk_avd_gaps_filled", 0)]
        lk_avd_clean = database + [(series_name + "lk_avd_gaps_filled", 0)]
        rk_mtt_clean = database + [(series_name + "rk_mtt_gaps_filled", 0)]
        lk_mtt_clean = database + [(series_name + "lk_mtt_gaps_filled", 0)]
        
        clean_series = [rk_rpf_clean, lk_rpf_clean,
                        rk_avd_clean, lk_avd_clean,
                        rk_mtt_clean, lk_mtt_clean ]
                
        if all(x in db.series(database) for x in clean_series):
            roi = ['lk', 'rk']
            for kidney in roi:
                cortex_medulla(site, study, roi=kidney)
            continue

        # Dictonary
        mask_path = entry['mask_path']   
        rk_dce_map_paths = entry['rk_dce_maps']
        lk_dce_map_paths = entry['lk_dce_maps']

        rk_dce_volumes = []
        for map_path in rk_dce_map_paths:
            rk_dce_volume = db.volumes_2d(map_path)
            rk_dce_volumes.append(rk_dce_volume)

        lk_dce_volumes = []
        for map_path in lk_dce_map_paths:
            lk_dce_volume = db.volumes_2d(map_path)
            lk_dce_volumes.append(lk_dce_volume)


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
        if rk_rpf_clean not in db.series(database):
            db.write_volume((rk_outputs[0], rk.affine), rk_rpf_clean, ref=rk_dce_map_paths[0])

        if lk_rpf_clean not in db.series(database):
            db.write_volume((lk_outputs[0], lk.affine), lk_rpf_clean, ref=lk_dce_map_paths[0])
        
        print('Building AVD Volume...')
        if rk_avd_clean not in db.series(database):
            db.write_volume((rk_outputs[1], rk.affine), rk_avd_clean, ref=rk_dce_map_paths[1])
        
        if lk_avd_clean not in db.series(database):
            db.write_volume((lk_outputs[1], lk.affine), lk_avd_clean, ref=lk_dce_map_paths[1])        

        print('Building MTT Volume...')
        if rk_mtt_clean not in db.series(database):
            db.write_volume((rk_outputs[2], rk.affine), rk_mtt_clean, ref=rk_dce_map_paths[2])        
        
        if lk_mtt_clean not in db.series(database):
            db.write_volume((lk_outputs[2], lk.affine), lk_mtt_clean, ref=lk_dce_map_paths[2])  
        

        roi = ['lk', 'rk']
        for kidney in roi:
            cortex_medulla(site, study, roi=kidney)


#Step 3: Cortex Medulla Mask Creation
def cortex_medulla(site, study, roi=None): 

    masks_dir = os.path.join(os.getcwd(), 'build', 'kidneyvol_3_edit', site, "Patients")
    mask_database = db.series(masks_dir)
    maps_path = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients")
    maps_database = db.series(maps_path)
    maps = [entry for entry in maps_database if entry[3][0].strip().lower() in
        (f'dce_10_{roi}_rpf_gaps_filled', f'dce_10_{roi}_avd_gaps_filled',  f'dce_10_{roi}_mtt_gaps_filled')] 
    
    pat_series = []
    dest_dir = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients")
    mask_png = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients", 'overlays', f'{study}_{roi}.png')
    database = [dest_dir, study, ('Baseline', 0)]
    series_name = bari_add_series_name(study, pat_series)


    both_kidneys = next((m for m in mask_database if m[1] == study), None)
    if both_kidneys:
        both_kidneys = db.volume(both_kidneys)
        both_kidneys_arr = both_kidneys.values 
        aff = both_kidneys.affine
        if site == 'Bari':
            rk = (both_kidneys_arr==2)
            lk = (both_kidneys_arr==1)
        else:
            rk = (both_kidneys_arr==1)
            lk = (both_kidneys_arr==2)

        if roi == 'rk':
            mask_roi = vreg.volume(rk, aff)
        else:
            mask_roi = vreg.volume(lk, aff)

        if maps:
            try:
                output = []
                for path in maps:
                        volume = db.volume(path)
                        output.append(volume)
                clusters, cluster_features = kmeans(output, mask_roi, roi=roi, n_clusters=3, multiple_series=True, return_features=True, site=site, study=study)
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

            if roi == 'lk':
                db.write_volume(clusters[cortex], cortex_lk)
                db.write_volume(clusters[medulla], medulla_lk)
            
                cm = np.zeros_like(clusters[background].values)
                cm[clusters[background].values > 0] = 0
                cm[clusters[cortex].values > 0] = 3
                cm[clusters[medulla].values > 0] = 4
                cm_vol = vreg.volume(cm, aff)
                vplot.overlay_2d_cm(output[0], mask=cm_vol, save_path=mask_png, show=False)
                db.write_volume(cm_vol, lk_cm_desc)
            
            elif roi == 'rk':
                db.write_volume(clusters[cortex], cortex_rk)
                db.write_volume(clusters[medulla], medulla_rk)
                
                cm = np.zeros_like(clusters[background].values)
                cm[clusters[background].values > 0] = 0
                cm[clusters[cortex].values > 0] = 1
                cm[clusters[medulla].values > 0] = 2
                cm_vol = vreg.volume(cm, aff)
                vplot.overlay_2d_cm(db.volume(maps[0]), mask=cm_vol, save_path=mask_png, show=True)
                db.write_volume(cm_vol, rk_cm_desc)



#Step 4: Extract cortex and medulla input function and write to dmr
def extract_mdr_input_function(site, roi=None):


    def extract_times(roi=None):
        data_dir = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients")
        database = db.series(data_dir)
        mdr_aligned = [entry for entry in database if entry[3][0].strip().lower() == f'dce_9_mdr_{roi}_aligned'.lower()]
        time_list = []
        for series in mdr_aligned:
            case_id = series[1]
            vols = db.volumes_2d(series, dims=['AcquisitionTime'])
            for vol in vols:
                time = vol.coords
                time = [t for sublist in time for t in sublist]
                time = time - time[0]
                time_list.append((case_id, time))
                break
        return time_list
    
    def cortex_on_mdr(cortex_vol=None, roi=None):

        data_dir = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients")
        database = db.series(data_dir)
        study = cortex_vol[0]
        mdr_aligned = [entry for entry in database if entry[3][0].strip().lower() == f'dce_9_mdr_{roi}_aligned'.lower()]
        for series in mdr_aligned:
            vols = db.volumes_2d(series, dims=['AcquisitionTime'])
            cortex_on_time_vols = []
            for vol in vols:
                vol_t = vreg.volume(vol.values[:,:,0,:], vol.affine) 
                cortex_on_time_vol = cortex_vol[1].slice_like(vol_t)
                cortex_on_time_vols.append(cortex_on_time_vol.values)
            cortex_on_vol = []
            mask = (np.stack(cortex_on_time_vols, axis=2)).squeeze()
            cortex_on_vol.append((study, mask))
        return cortex_on_vol


    
    def extract_labels(roi=None):
        # cortex + medulla 
        # data_dir = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients")
        # database = db.series(data_dir)
        
        # whole kidney
        data_dir = os.path.join(os.getcwd(), 'build', 'kidneyvol_3_edit', site, "Patients")
        database = db.series(data_dir)        
        
        if roi == 'LK':
            #inventory = [entry for entry in database if entry[3][0].strip().lower() == 'dce_10_lkm'.lower()]
            inventory = [entry for entry in database if entry[3][0].strip().lower() == 'kidney_masks'.lower()]
        elif roi == 'RK':
            #inventory = [entry for entry in database if entry[3][0].strip().lower() == 'dce_10_rkm'.lower()]
            inventory = [entry for entry in database if entry[3][0].strip().lower() == 'kidney_masks'.lower()]

        
        labels = []
        for series in inventory:
            case_id = series[1]
            vol = db.volume(series)
            arr = vol.values
            aff = vol.affine 
            lk = (arr == 1)
            lk_voxels = np.sum(lk)
            rk = (arr == 2)
            rk_voxels = np.sum(rk)
            voxel_size = 1.25*1.25*1.5 # pixel_spacing x slice_thickness
            lk_vol = lk_voxels*voxel_size/1000 #convert to ml
            rk_vol = rk_voxels*voxel_size/1000
            if roi == 'LK':
                vol = vreg.volume(lk, aff)
            elif roi == 'RK':
                vol = vreg.volume(rk, aff)
            labels.append((case_id, vol))
        return labels, rk_vol, lk_vol 

    print('Extracting Times...')
    case_id_and_times = extract_times(roi=roi)

    print('Extracting Labels to Align...')
    labels, rk_vol, lk_vol = extract_labels(roi=roi)

    for label in tqdm(labels, desc=f'Processing {roi} Cortex Masks', unit="label"):
        cortex_masks_on_mdr = cortex_on_mdr(cortex_vol=label, roi=roi)
    


    if not isinstance(case_id_and_times, Iterable) or isinstance(case_id_and_times, str):
        case_id_and_times = [case_id_and_times]

    
    print(f'Getting {roi} values')
    for times in tqdm(case_id_and_times, unit='case'):

        case_id = times[0]
        data_dir = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients")
        csv_dir = data_dir + '/CSV' + f'/{case_id}'
        dmr_dir = data_dir + '/DMR' + f'/{case_id}'
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(dmr_dir, exist_ok=True)
        database = db.series(data_dir)
        mdr_aligned_database = [entry for entry in database if entry[3][0].strip().lower() == f'dce_9_mdr_{roi}_aligned'.lower()]
        
        mdr_aligned_series = next(img for img in mdr_aligned_database if img[1] == case_id)
        
        if mdr_aligned_series is None:
            print(f'No aligned {roi} mdr series found')
            continue
        
        mask = next(m for m in cortex_masks_on_mdr if m[0] == case_id)
        
        if mask is not None:
            # get 4D volume: (x, y, z, t)
            volume = db.volumes_2d(mdr_aligned_series, dims=['AcquisitionTime'])
            mdr_vol = []
            for vol in volume:
                vol_arr = vol.values
                mdr_vol.append(vol_arr)
            mdr_vol = (np.stack(mdr_vol, axis=2)).squeeze()


            # make sure mask is 3D (x, y, z)
            if mask[1].ndim == 4:
                mask_3d = mask[1][..., 0] 
                voxels = np.sum(mask_3d==2)
                kidney_vol = voxels * 1.5 * 1.25 * 1.25 /1000

            else:
                mask[1]
                
            mask = mask_3d.astype(bool)
            slice_means = []
             # compute slice-wise mean intensities for each timepoint
            for t in range(mdr_vol.shape[-1]):  # loop over time frames
                t_means = []
                for z in range(mdr_vol.shape[2]):  # loop over slices
                    masked_values = mdr_vol[:, :, z, t][mask[:, :, z]]
                    t_means.append(masked_values.mean() if masked_values.size > 0 else np.nan)
                slice_means.append(t_means)

            # average across slices (shape: n_timepoints,)
            cif_int = np.nanmean(slice_means, axis=1)
        else: 
            print(f'no {roi} mask found in database for {case_id} below')
            continue

    
        # Ensure same length as curve
        if len(times[1]) != len(cif_int):
            print(f"Length mismatch for {case_id}: times={len(times[1])}, curve={len(cif_int)}")
            continue
    
        # Convert curve to plain floats
        cif_int = [float(val) for val in cif_int]
    
        # Build dataframe
        df = pd.DataFrame({
        "Time (s)": times[1],
        "MIF Intensity": cif_int
        })

        if roi == 'LK':
            kid_vol = lk_vol.tolist()
            r = 'kidney_left'
        elif roi == 'RK':
            kid_vol = rk_vol.tolist()
            r = 'kidney_right'
        study = 'Baseline'
    
        # Save CSV
        outpath = os.path.join(csv_dir, f"{case_id}_{r}_if.csv")
        df.to_csv(outpath, index=False)

        print(f"Saved {r} intensity values for {case_id} {outpath}")

        dmr_file = os.path.join(dmr_dir, f"{case_id}_{r}_if")
        dmr = {'data':{}, 'pars':{}, 'rois':{}}
        dmr['rois'][(case_id, study, 'time')] = times[1]
        dmr['rois'][(case_id, study, 'signal')] = cif_int
        dmr['pars'][(case_id, study, 'field_strength')] = 3
        dmr['pars'][(case_id, study, 'agent')] = 'gadoterate' 
        dmr['pars'][(case_id, study, 'n0')] = 15
        dmr['pars'][(case_id, study, 'TR')] = 0.002
        dmr['pars'][(case_id, study, 'FA')] = 10
        dmr['pars'][(case_id, study, f'{r} vol')] = kid_vol
        dmr['pars'][(case_id, study, f'{r} T1')] = 1.4
        dmr['data']['time'] = ['Acquisition time', 'sec', 'float']
        dmr['data']['signal'] = ['Average signal intensity', 'a.u.', 'float']
        dmr['data']['field_strength'] = ['B0 magnetic field strength', 'T', 'float']
        dmr['data']['agent'] = ['Contrast agent generic name', '', 'str']
        dmr['data']['n0'] = ['Number of precontrast acquisition', '', 'int']
        dmr['data']['TR'] = ['Repetition Time', 'sec', 'float']
        dmr['data']['FA'] = ['Flip angle', 'deg', 'float']
        dmr['data'][f'{r} vol'] = ['Kidney volume', 'mL', 'float']
        dmr['data'][f'{r} T1'] = ['Kidney T1 relaxation time', 'sec', 'float']
        pydmr.write(dmr_file, dmr)
    


if __name__ == '__main__':
    get_data('Bari')
    #fillgaps('Bari')
    # roi = ['RK', 'LK']
    # for m in roi:
    #     #     cortex_medulla('Bari', study='1128_003', roi=m)
    #     extract_mdr_input_function('Bari', roi=m)