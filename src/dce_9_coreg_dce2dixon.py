#import elastix
import os
import dbdicom as db
import vreg
import pickle
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time 
from tqdm import tqdm
import vreg.plot as vplot


def bari_add_series_name(folder, all_series: list):
    new_series_name = "DCE_9_"
    all_series.append(new_series_name)
    return new_series_name

def _get_data(site, table_dir=None):
    if table_dir is None:
        table_dir = os.path.join(os.getcwd(), 'build', 'dce_9_coreg_dce2dixon', site, "Patients")
    os.makedirs(table_dir, exist_ok=True)  

    masks_dir = os.path.join(os.getcwd(), 'build', 'kidneyvol_3_edit', site, "Patients")
    dce_maps_dir = os.path.join(os.getcwd(), 'build', 'dce_8_mapping', site, "Patients")
    mdr_dir = os.path.join(os.getcwd(), 'build', 'dce_7_mdreg', site, "Patients")

    # Load series from databases
    masks_database = db.series(masks_dir)
    dce_database = db.series(dce_maps_dir)
    mdr_database = db.series(mdr_dir)
    
    #Filter out motion corrected time series
    mdr_moco = [
        entry for entry in mdr_database 
        if entry[3][0].strip().lower() ==
        'dce_7_mdr_moco'
    ]
    # Filter out dce maps to align
    maps2align_database = [
        entry for entry in dce_database 
        if entry[3][0].strip().lower() in
        ('dce_8_auc_map',  'dce_8_rpf_map', 'dce_8_avd_map', 'dce_8_mtt_map') 
    ]

    # Get unique case identifiers
    cases_str = set(entry[1] for entry in maps2align_database)

    images_and_masks = []
    for case_str in sorted(cases_str):
        # Find corresponding mask study
        mask_study = next((s for s in masks_database if s[1] == case_str), None)
        if mask_study is None:
            print(f"Skipping case {case_str}, study not found in mask database.")
            continue
        
        mdr_study = next((s for s in mdr_moco if s[1] == case_str), None)
        if mdr_study is None:
            print(f"Skipping case {case_str}, DCE mdr series series not found.") 
            continue 


        dce_study = [s for s in maps2align_database if s[1] == case_str]
        if not dce_study:
            print(f"Skipping case {case_str}, DCE maps not found.")  
            continue

        # Load image and mask volumes
        mask_volume = db.volume(mask_study)
    

        ### create AUC slice volumes (x, y, t)
        auc_map_path = dce_study[0]
        auc_slice_volumes = db.volumes_2d(auc_map_path)

        mdr_volumes = db.volumes_2d(mdr_study, dims='AcquistionTime')


        # map_split_to_slice = db.split_series(auc_map_path, 'ImagePositionPatient')
        # sorted_slices = sorted(map_split_to_slice, key=lambda x: x[0])
        # auc_slice_volumes = []
        # for _, slice in sorted_slices:
        #     slice_vol = db.volume(slice)
        #     auc_slice_volumes.append(slice_vol)
        
    

        #Create Data Table 
        images_and_masks.append({
            'case': mask_study[1],
            'dce_slices': auc_slice_volumes,
            'mask_volume': mask_volume,
            'dce_maps': dce_study,
            'mdr_path': mdr_study,
            'mdr_vols': mdr_volumes
        })

    # Save the results to file
    output_path = os.path.join(table_dir, f'{site}_images_masks_table.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(images_and_masks, f)
    
    return images_and_masks



def _coreg(volume, bk, lk, rk):
    # This is the actual coregistration operating on vreg volumes.
    #
    # Registration is performed by 3D translation in 2 steps - first 
    # align caorsely to both kidneys
    # 
    # The function returns a dictionary with two affines: one for 
    # coregistration to the left kidney, and one for the right.

    # Settings
    options = {'coords': 'volume'}
    optimizer = {'method': 'brute'}

    # Translate to both kidneys
    optimizer['grid'] = [[-20, 20, 20],
                         [-20, 20, 20],
                         [-5, 5, 5]]

    print('Coregistering BK...')
    tbk =volume.find_translate_to(bk, optimizer=optimizer, **options) 
    volume = volume.translate(tbk, **options)

    # Per kidney fine tuning
    optimizer['grid'] = 3*[[-2, 2, 10]]

    print('Coregistering LK...')
    tlk = volume.find_translate_to(lk, optimizer=optimizer, **options)
    
    print('Coregistering RK...')
    trk = volume.find_translate_to(rk, optimizer=optimizer, **options)


    # Return affines for each kidney
    return {
        'BK': volume.translate(tbk, **options).affine,
        'LK': volume.translate(tlk, **options).affine,
        'RK': volume.translate(trk, **options).affine
    }




def _align_2d(site, case=None, table_dir=None):

    if table_dir is None:
        # Build default path
        table_path = os.path.join(
            os.getcwd(),
            'build', 'dce_9_coreg_dce2dixon',
            site, "Patients",
            f'{site}_images_masks_table.pkl'
        )
        if os.path.isfile(table_path):
            print('using existing img and masks table')
            # Case 1a: Default file exists → load it
            with open(table_path, "rb") as f:
                images_and_mask_table = pickle.load(f)
        else:
            # Case 1b: No default file → resort to _get_data
            print('creating new img and masks table')
            images_and_mask_table = _get_data(site, table_dir)

    
    pat_series = []
    for entry in images_and_mask_table:
        study = entry['case']
        auc_slice_volumes = entry['dce_slices']
        mask_volume = entry['mask_volume']
        dce_maps = entry['dce_maps']
        mdr_slice_volumes = entry['mdr_vols']
        mdr_path = entry['mdr_path']

        #define paths
        dest_dir = os.path.join(os.getcwd(), 'build', 'dce_9_coreg_dce2dixon', site, "Patients")
        png_save = os.path.join(os.getcwd(), 'build', 'dce_9_coreg_dce2dixon', site, "Patients", 'coreg_png')
        os.makedirs(png_save, exist_ok=True)

        bari_add_series_name(study, pat_series)
        database = [dest_dir, study, ('Baseline', 0)]
        
        #auc paths
        auc_clean_rk = database + [(pat_series[-1] + "auc_rk_aligned", 0)]
        auc_clean_lk = database + [(pat_series[-1] + "auc_lk_aligned", 0)]

        rpf_clean_rk = database + [(pat_series[-1] + "rpf_rk_aligned", 0)]
        rpf_clean_lk = database + [(pat_series[-1] + "rpf_lk_aligned", 0)]

        #avd paths
        avd_clean_rk  = database + [(pat_series[-1] + "avd_rk_aligned", 0)]
        avd_clean_lk  = database + [(pat_series[-1] + "avd_lk_aligned", 0)]

        #mtt paths 
        mtt_clean_rk = database + [(pat_series[-1] + "mtt_rk_aligned", 0)]
        mtt_clean_lk = database + [(pat_series[-1] + "mtt_lk_aligned", 0)]

        #mdr paths
        mdr_clean_rk = database + [(pat_series[-1] + "mdr_rk_aligned", 0)]
        mdr_clean_lk = database + [(pat_series[-1] + "mdr_lk_aligned", 0)]        

        #create LK/RK mask volumes
        bk_vol = mask_volume
        bk = bk_vol.values
        if site == 'Bari':
            rk_arr = (bk == 2)
            lk_arr = (bk == 1)
        else:
            rk_arr = (bk == 1)
            lk_arr = (bk == 2)

        rk = vreg.volume(rk_arr, bk_vol.affine)
        lk = vreg.volume(lk_arr, bk_vol.affine)

        

        # Use bounding boxes speed up the computation
        bk = bk_vol.bounding_box()
        rk = rk.bounding_box()
        lk = lk.bounding_box()

        
        affine = {'LK':[], 'RK':[]}
        # Wrap loop with tqdm to display a progress bar
        for z in tqdm(auc_slice_volumes, desc="Coregistering slices", unit="slice"):
            start_time = time.time()  # Record start time
            
            # Perform the coregistration
            az = _coreg(z, bk, lk, rk)
            
            # Append the results to the affine dictionary
            affine['LK'].append(az['LK'])
            affine['RK'].append(az['RK'])
            
            end_time = time.time()  # Record end time
            time_taken = end_time - start_time  # Calculate time taken for this iteration
            
            #time for each slice
            #print(f"Time taken for slice: {time_taken:.1f} seconds")
        


        ## Assign affine and create dicom ##

        for aff, s in zip(affine['RK'], mdr_slice_volumes):
            s.set_affine(aff)
        db.write_volume(mdr_slice_volumes, mdr_clean_rk, ref=mdr_path)
        

        for aff, s in zip(affine['LK'], mdr_slice_volumes):
            s.set_affine(aff)
        db.write_volume(mdr_slice_volumes, mdr_clean_lk, ref=mdr_path)


        # for aff, s in zip(affine['LK'], auc_slice_volumes):
        #     s.set_affine(aff)
        
        post_auc_lk_vols = []
        for aff, s in zip(affine['LK'], auc_slice_volumes):
            new_slice = vreg.volume(s.values, aff)
            post_auc_lk_vols.append(new_slice)
        
        post_auc_rk_vols = []
        for aff, s in zip(affine['RK'], auc_slice_volumes):
            new_slice = vreg.volume(s.values, aff)
            post_auc_rk_vols.append(new_slice)
            
        
        vplot.overlay_2d_new(pre=auc_slice_volumes, post=post_auc_lk_vols, 
                                         mask=lk, 
                                         save_path = png_save + f'/{study}_auc_coreg_lk.png', 
                                         show=True)
        
        vplot.overlay_2d_new(pre=auc_slice_volumes, post=post_auc_rk_vols, 
                                         mask=rk, 
                                         save_path = png_save + f'/{study}_auc_coreg_rk.png', 
                                         show=True)

        for slice_vol in post_auc_lk_vols:
            db.write_volume(slice_vol, auc_clean_lk, ref=dce_maps[0], append=True)
        
        for slice_vol in post_auc_rk_vols:
            db.write_volume(slice_vol, auc_clean_rk, ref=dce_maps[0], append=True)

        print('Coregistering RPF...')
        rpf_volume = db.volumes_2d(dce_maps[1])
        
        lk_affine =[]
        for vol in post_auc_lk_vols:
            affine = vol.affine
            lk_affine.append(affine) 
        
        rk_affine =[]
        for vol in post_auc_rk_vols:
            affine = vol.affine
            rk_affine.append(affine)
        

        post_rpf_lk_vols = []
        for aff, s in zip(lk_affine, rpf_volume):
            new_slice = vreg.volume(s.values, aff)
            post_rpf_lk_vols.append(new_slice)       
        
        
        post_rpf_rk_vols = []
        for aff, s in zip(rk_affine, rpf_volume):
            new_slice = vreg.volume(s.values, aff)
            post_rpf_rk_vols.append(new_slice)        
        
        for slice_vol in post_rpf_rk_vols:
            db.write_volume(slice_vol, rpf_clean_rk, ref=dce_maps[1], append=True)        
        
        for slice_vol in post_rpf_lk_vols:
            db.write_volume(slice_vol, rpf_clean_lk, ref=dce_maps[1], append=True)        
        
        print('Coregistering AVD...')
        avd_volume = db.volumes_2d(dce_maps[2])        
        
        post_avd_lk_vols = []
        for aff, s in zip(lk_affine, avd_volume):
            new_slice = vreg.volume(s.values, aff)
            post_avd_lk_vols.append(new_slice)    

        post_avd_rk_vols = []
        for aff, s in zip(rk_affine, avd_volume):
            new_slice = vreg.volume(s.values, aff)
            post_avd_rk_vols.append(new_slice)        
        
        for slice_vol in post_avd_rk_vols:
            db.write_volume(slice_vol, avd_clean_rk, ref=dce_maps[2], append=True)
        
        
        for slice_vol in post_avd_lk_vols:
            db.write_volume(slice_vol, avd_clean_lk, ref=dce_maps[2], append=True)        
        

        print('Coregistering MTT...')
        mtt_volume = db.volumes_2d(dce_maps[3])
        post_mtt_rk_vols = []
        for aff, s in zip(rk_affine, mtt_volume):
            new_slice = vreg.volume(s.values, aff)
            post_mtt_rk_vols.append(new_slice)

        post_mtt_lk_vols = []
        for aff, s in zip(lk_affine, mtt_volume):
            new_slice = vreg.volume(s.values, aff)
            post_mtt_lk_vols.append(new_slice)

        for slice_vol in post_mtt_rk_vols:
            db.write_volume(slice_vol, mtt_clean_rk, ref=dce_maps[3], append=True)

        for slice_vol in post_mtt_lk_vols:
            db.write_volume(slice_vol, mtt_clean_lk, ref=dce_maps[3], append=True)



        print('Coregistering complete!')
        


if __name__ == '__main__':
    _align_2d('Bari')

