import numpy as np
import dbdicom as db 
import os
import logging 
import pandas as pd
import pydmr

#testing


def Leeds():
    
    datapath = os.path.join(os.getcwd(), 'build', 'dce_2_data')
    imgpath = os.path.join(datapath, "Leeds", "Patients")
    
    maskpath = os.path.join(os.getcwd(), 'build', 'dce_4_aortaseg')
    segpath = os.path.join(maskpath, "Leeds", "Patients")
    
    dstdatapath = os.path.join(os.getcwd(), 'build', 'dce_5_aorta2csv')
    destpath =  os.path.join(dstdatapath, "Leeds", "Patients")
    os.makedirs(destpath, exist_ok=True)
    
    imgdatabase = db.series(imgpath)
    DCE_aorta = [entry for entry in imgdatabase if entry[3][0].strip().lower() == 'dce_1_axial_aorta']
    

    segdatabase = db.series(segpath)
    
    masks = [entry for entry in segdatabase if entry[3][0].strip().lower() == 'dce_4_aortaseg']
    
    
    def extract_times():
        results = []
        for study in DCE_aorta:
            aorta  = db.volume(study, dims='AcquisitionTime') 
            times_array = aorta._coords.tolist()
            times = [t for sublist in times_array for t in sublist]
            results.append((study[1], times))
    
        return results
    
    def extract_array():
        results = []
        for study in DCE_aorta:
            aorta  = db.volume(study, dims='AcquisitionTime') 
            pixel_array = aorta.values
            results.append((study[1], pixel_array))
        
        return results
    
    def extract_masks():
        results = []
        for study in masks:
            mask = db.volume(study)
            mask_array = mask.values
            results.append((study[1], mask_array))
        return results
    

    study_times = extract_times()
    images = extract_array()
    labels = extract_masks()


    aif_results = {}
    


    for case_id, img in images:   # unpack tuple
        # Find corresponding mask (also a list of tuples)
        mask_tuple = next((m for m in labels if m[0] == case_id), None)
        if mask_tuple is not None:
            mask = np.squeeze(mask_tuple[1])  # remove singleton dims
            img  = np.squeeze(img)

            if img.ndim == 3 and img.shape[-1] > 1:  # (384, 384, T)
                img = np.transpose(img, (2, 0, 1)) 
           
            # Skip empty masks
            if not np.any(mask > 0):
                print(f"Mask for case {case_id} is empty")
                continue

            # Handle 3D time-series (time, y, x)
            if img.ndim == 3:
                aif_curve = []
                for t in range(img.shape[0]):
                    frame = np.squeeze(img[t])
                    masked_frame = frame * mask
                    mean_val = masked_frame[mask > 0].mean()
                    aif_curve.append(mean_val)

            # Handle single-frame 2D image
            elif img.ndim == 2:
                masked_frame = img * mask
                aif_curve = [masked_frame[mask > 0].mean()]

            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")

            aif_results[case_id] = aif_curve
        else:
            print(f"No mask found for case {case_id}")


    for case_id, aif_curve in aif_results.items():
        # Find corresponding times for this case
        time_tuple = next((t for t in study_times if t[0] == case_id), None)
        if time_tuple is None:
            print(f"No acquisition times for case {case_id}")
            continue
    
        times = np.array(time_tuple[1], dtype=float)
        times = times - times[0]  # normalize so first time = 0
    
        # Ensure same length as curve
        if len(times) != len(aif_curve):
            print(f"Length mismatch for {case_id}: times={len(times)}, curve={len(aif_curve)}")
            continue
    
        # floats
        aif_curve = [float(val) for val in aif_curve]
    
        # Build dataframe
        df = pd.DataFrame({
        "Time (s)": times,
        "AIF Intensity": aif_curve
        })
    
    # Save CSV
        outpath = os.path.join(destpath, f"{case_id}_aif.csv")
        df.to_csv(outpath, index=False)
        print(f"Saved AIF curve for {case_id} -> {outpath}")
        df = pd.DataFrame({
        "Time (s)": times,
        "AIF Intensity": aif_curve
        })
        outpath = os.path.join(destpath, f"{case_id}_aif.csv")
        df.to_csv(outpath, index=False)


def Bari():
    
    datapath = os.path.join(os.getcwd(), 'build', 'dce_2_data')
    imgpath = os.path.join(datapath, "Bari", "Patients")
    
    maskpath = os.path.join(os.getcwd(), 'build', 'dce_4_aortaseg')
    segpath = os.path.join(maskpath, "Bari", "Patients")
    
    dstdatapath = os.path.join(os.getcwd(), 'build', 'dce_5_aorta2csv')
    destpath =  os.path.join(dstdatapath, "Bari", "Patients")
    os.makedirs(destpath, exist_ok=True)
    
    imgdatabase = db.series(imgpath)
    DCE_aorta = [entry for entry in imgdatabase if entry[3][0].strip().lower() == 'dce_1_axial_aorta']
    

    segdatabase = db.series(segpath)
    
    masks = [entry for entry in segdatabase if entry[3][0].strip().lower() == 'dce_4_aortaseg']
    
    
    def extract_times():
        results = []
        for study in DCE_aorta:
            aorta  = db.volume(study, dims='AcquisitionTime') 
            times_array = aorta.coords
            times = [t for sublist in times_array for t in sublist]
            results.append((study[1], times))
    
        return results
    
    def extract_array():
        results = []
        for study in DCE_aorta:
            aorta  = db.volume(study, dims='AcquisitionTime') 
            pixel_array = aorta.values
            results.append((study[1], pixel_array))
        
        return results
    
    def extract_masks():
        results = []
        for study in masks:
            mask = db.volume(study)
            mask_array = mask.values
            results.append((study[1], mask_array))
        return results
    

    study_times = extract_times()
    images = extract_array()
    labels = extract_masks()


    aif_results = {}
    


    for case_id, img in images:   # unpack dict
        # Find corresponding mask (also a list of tuples)
        mask_dict = next((m for m in labels if m[0] == case_id), None)
        if mask_dict is not None:
            mask = np.squeeze(mask_dict[1])  # remove singleton dims
            img  = np.squeeze(img)

            if img.ndim == 3 and img.shape[-1] > 1:  # (384, 384, T)
                img = np.transpose(img, (2, 0, 1)) #time first
           
            # Skip empty masks
            if not np.any(mask > 0):
                print(f"Mask for case {case_id} is empty")
                continue

            # Handle 3D time-series (time, y, x)
            if img.ndim == 3:
                aif_curve = []
                for t in range(img.shape[0]):
                    frame = np.squeeze(img[t])
                    masked_frame = frame * mask
                    mean_val = masked_frame[mask > 0].mean()
                    aif_curve.append(mean_val)

            # Handle single-frame 2D image
            elif img.ndim == 2:
                masked_frame = img * mask
                aif_curve = [masked_frame[mask > 0].mean()]

            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")

            aif_results[case_id] = aif_curve
        else:
            print(f"No mask found for case {case_id}")


    for case_id, aif_curve in aif_results.items():
        # Find corresponding times for this case
        time_tuple = next((t for t in study_times if t[0] == case_id), None)
        if time_tuple is None:
            print(f"No acquisition times for case {case_id}")
            continue
    
        times = np.array(time_tuple[1], dtype=float)
        times = times - times[0]  # normalize so first time = 0
    
        # Ensure same length as curve
        if len(times) != len(aif_curve):
            print(f"Length mismatch for {case_id}: times={len(times)}, curve={len(aif_curve)}")
            continue
    
        # Convert curve to plain floats
        aif_curve = [float(val) for val in aif_curve]
    
        # Build dataframe
        df = pd.DataFrame({
        "Time (s)": times,
        "AIF Intensity": aif_curve
        })
    
    # Save CSV
        outpath = os.path.join(destpath, f"{case_id}_aif.csv")
        df.to_csv(outpath, index=False)
        print(f"Saved AIF curve for {case_id} {outpath}")
        df = pd.DataFrame({
        "Time (s)": times,
        "AIF Intensity": aif_curve
        })
        outpath = os.path.join(destpath, f"{case_id}_aif.csv")
        df.to_csv(outpath, index=False)

    #Save to DMR File
        roi = 'aorta' 
        data_dir = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', 'Bari', "Patients") 
        dmr_file = os.path.join(data_dir, 'DMR', case_id,  f"{case_id}_aif")

        dmr = {'data':{}, 'pars':{}, 'rois':{}}
        dmr['rois'][(case_id, roi, 'time')] = times
        dmr['rois'][(case_id, roi, 'signal')] = aif_curve
        dmr['data']['time'] = ['Acquisition time', 'sec', 'float']
        dmr['data']['signal'] = ['Average signal intensity', 'a.u.', 'float']
        pydmr.write(dmr_file, dmr)

def sheffield_patients(segtype=None):
    
    datapath = os.path.join(os.getcwd(), 'build', 'dce_2_data')
    imgpath = os.path.join(datapath, "Sheffield", "Patients")
    
    maskpath = os.path.join(os.getcwd(), 'build', 'dce_4_aortaseg')
    segpath = os.path.join(maskpath, "Sheffield", "Patients")
    
    dstdatapath = os.path.join(os.getcwd(), 'build', 'dce_5_aorta2csv')
    destpath =  os.path.join(dstdatapath, "Sheffield", "Patients")
    os.makedirs(destpath, exist_ok=True)
    
    imgdatabase = db.series(imgpath)
    
    DCE_aorta = [entry for entry in imgdatabase if entry[3][0].strip().lower() == 'dce_1_coronal_kidneys']
    
    
    segdatabase = db.series(segpath)
    
    if segtype == None:
        masks = [entry for entry in segdatabase if entry[3][0].strip().lower() == 'dce_4_aortaseg']
    elif segtype == 'manual':
        masks = [entry for entry in segdatabase if entry[3][0].strip().lower() == 'dce_4_aortaseg']
    elif segtype== 'auto':
        masks = [entry for entry in segdatabase if entry[3][0].strip().lower() == 'dce_4_aortasegauto']
    
    
    def extract_times():
        results = []
        for study in DCE_aorta:
            aorta  = db.volume(study, dims='TriggerTime') #for sheffield trigger time returns acquisition times (msec)
            times_array = aorta.coords
            times = [t for sublist in times_array for t in sublist]
            times = times - times[0]
            times = times/1000 #convert to secs 
            results.append((study[1], times))
    
        return results
    
    def extract_array():
        results = []
        for study in DCE_aorta:
            aorta  = db.volume(study, dims='TemporalPositionIdentifier') 
            pixel_array = aorta.values
            results.append((study[1], pixel_array))
        
        return results
    
    def extract_masks():
        results = []
        for study in masks:
            mask = db.volume(study)
            mask_array = mask.values
            results.append((study[1], mask_array))
        return results
    

    study_times = extract_times()
    images = extract_array()
    labels = extract_masks()


    aif_results = {}
    


    for case_id, img in images:   # unpack dict
        # Find corresponding mask (also a list of tuples)
        mask_dict = next((m for m in labels if m[0] == case_id), None)
        if mask_dict is not None:
            #mask = np.squeeze(mask_dict[1])  # remove singleton dims
            mask = mask_dict[1]
            mask = mask.astype(bool)
            print(mask.shape)

            if segtype=='auto':

                aif_means = []
             # compute slice-wise mean intensities for each timepoint
                for t in range(img.shape[-1]):  # loop over time frames
                    t_means = []
                    for z in range(img.shape[2]):  # loop over slices
                        masked_values = img[:, :, z, t][mask[:, :, z]]
                        t_means.append(masked_values.mean() if masked_values.size > 0 else np.nan)
                    aif_means.append(t_means)

             # average across slices (shape: n_timepoints,)
                aif_int = np.nanmean(aif_means, axis=1)
                aif_results[case_id] = aif_int
                continue

            
            if img.ndim == 3 and img.shape[-1] > 1:  # (384, 384, T)
                img = np.transpose(img, (2, 0, 1)) #time first
            
            if img.ndim == 4:
                img = img[:,0,:,:]
                img = np.transpose(img, (2, 0, 1))
                print(img.shape)


           
            # Skip empty masks
            if not np.any(mask > 0):
                print(f"Mask for case {case_id} is empty")
                continue

            # Handle 3D time-series (time, y, x)
            if img.ndim == 3:
                aif_curve = []
                # mask_expanded = np.zeros((256, 256, 1), dtype=mask.dtype)
                # mask_expanded[:mask.shape[0], :, :] = mask
                # mask = mask_expanded.squeeze()

                # mask = np.expand_dims(mask, axis=-1)
                # scales = np.arange(256) #mask shape (28, 256, 256)

                # mask = mask * scales
                for t in range(img.shape[0]):
                    frame = np.squeeze(img[t])
                    mask = mask.squeeze()
                    frame = np.transpose(frame, (1,0))
                    masked_frame = frame * mask
                    mean_val = masked_frame[mask > 0].mean()
                    aif_curve.append(mean_val)
            
            

            # Handle single-frame 2D image
            elif img.ndim == 2:
                masked_frame = img * mask
                aif_curve = [masked_frame[mask > 0].mean()]

            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")
                

            aif_results[case_id] = aif_curve
        else:
            print(f"No mask found for case {case_id}")


    for case_id, aif_curve in aif_results.items():
        # Find corresponding times for this case
        time_tuple = next((t for t in study_times if t[0] == case_id), None)
        if time_tuple is None:
            print(f"No acquisition times for case {case_id}")
            continue
    
        times = np.array(time_tuple[1], dtype=float)
        times = times - times[0]  # normalize so first time = 0
    
        # Ensure same length as curve
        if len(times) != len(aif_curve):
            print(f"Length mismatch for {case_id}: times={len(times)}, curve={len(aif_curve)}")
            continue
    
        # Convert curve to plain floats
        aif_curve = [float(val) for val in aif_curve]
    
        # Build dataframe
        df = pd.DataFrame({
        "Time (s)": times,
        "AIF Intensity": aif_curve
        })
    
    # Save CSV
        outpath = os.path.join(destpath, f"{case_id}_aif_auto.csv")
        df.to_csv(outpath, index=False)
        print(f"Saved AIF curve for {case_id} {outpath}")
        df = pd.DataFrame({
        "Time (s)": times,
        "AIF Intensity": aif_curve
        })
        outpath = os.path.join(destpath, f"{case_id}_aif_auto.csv")
        df.to_csv(outpath, index=False)

   
    #Save to DMR File
        roi = 'aorta' 
        data_dir = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', 'Bari', "Patients") 
        dmr_file = os.path.join(data_dir, 'DMR', case_id,  f"{case_id}_aif")

        dmr = {'data':{}, 'pars':{}, 'rois':{}}
        dmr['rois'][(case_id, roi, 'time')] = times
        dmr['rois'][(case_id, roi, 'signal')] = aif_curve
        dmr['data']['time'] = ['Acquisition time', 'sec', 'float']
        dmr['data']['signal'] = ['Average signal intensity', 'a.u.', 'float']
        pydmr.write(dmr_file, dmr)



    
if __name__ == '__main__':
    #Bari()
    #Leeds()
    sheffield_patients(segtype='auto')