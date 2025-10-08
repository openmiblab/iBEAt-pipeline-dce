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
import logging


def bari_add_series_name(folder, all_series: list):
    new_series_name = "DCE_9_"
    all_series.append(new_series_name)
    return new_series_name

def _get_data(site, table_dir=None):
    if table_dir is None:
        table_dir = os.path.join(os.getcwd(), 'build', 'dce_9_coreg_dce2dixon', site, "Patients")
    os.makedirs(table_dir, exist_ok=True)  

    base_path = os.path.join(os.getcwd(), 'build')
    masks_dir = os.path.join(base_path, 'kidneyvol_3_edit', site, 'Patients')
    dce_maps_dir = os.path.join(base_path, 'dce_8_mapping', site, "Patients")
    mdr_dir = os.path.join(base_path, 'dce_7_mdreg', site, "Patients")

    # Load masks and series from databases
    mask_database = db.series(masks_dir)
    mdr_database = db.series(mdr_dir, desc='DCE_7_mdr_moco')
    auc_database = db.series(dce_maps_dir, desc='DCE_8_AUC_map')
    rpf_database = db.series(dce_maps_dir, desc='DCE_8_RPF_map')
    avd_database = db.series(dce_maps_dir, desc='DCE_8_AVD_map')
    mtt_database = db.series(dce_maps_dir, desc='DCE_8_MTT_map')

    

    # Get unique case identifiers
    cases_ids = set(entry[1] for entry in mdr_database)

    images_and_masks = []
    for case_id in sorted(cases_ids):
        
        # Find corresponding mask case_id
        mask_path = next((m for m in mask_database if m[1] == case_id), None)
        if mask_path is None:
            print(f"Skipping case {case_id}, case_id not found in mask database.")
            continue
        # Find corresponding mask case_id
        mdr_moco_path = next((img for img in mdr_database if img[1] == case_id), None)
        if mdr_moco_path is None:
            print(f"Skipping case {case_id}, DCE mdr series not found.") 
            continue 

        # Find corresponding DCE maps
        auc_map_path = [_map for _map in auc_database if _map[1] == case_id]
        if not auc_map_path:
            print(f"Skipping case {case_id}, DCE maps not found.")  
            continue
        rpf_map_path = [_map for _map in rpf_database if _map[1] == case_id]
        if not rpf_map_path:
            print(f"Skipping case {case_id}, DCE maps not found.")  
            continue
        avd_map_path = [_map for _map in avd_database if _map[1] == case_id]
        if not avd_map_path:
            print(f"Skipping case {case_id}, DCE maps not found.")  
            continue
        mtt_map_path = [_map for _map in mtt_database if _map[1] == case_id]
        if not mtt_map_path:
            print(f"Skipping case {case_id}, DCE maps not found.")  
            continue


        #Create Data Table 
        images_and_masks.append({
            'case_id': case_id,
            'mask_path': mask_path,
            'mdr_path': mdr_moco_path,
            'auc_map_path': auc_map_path,
            'rpf_map_path': rpf_map_path,
            'avd_map_path': avd_map_path,
            'mtt_map_path': mtt_map_path,
        })

    # Save the results to file
    output_path = os.path.join(table_dir, f'{site}_images_masks_table.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(images_and_masks, f)
    
    return images_and_masks


#Helper: Coreg parameters
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

    tqdm.write('Coregistering BK...')
    tbk =volume.find_translate_to(bk, optimizer=optimizer, **options) 
    volume = volume.translate(tbk, **options)

    # Per kidney fine tuning
    optimizer['grid'] = 3*[[-2, 2, 10]]

    tqdm.write('Coregistering LK...')
    tlk = volume.find_translate_to(lk, optimizer=optimizer, **options)
    
    tqdm.write('Coregistering RK...')
    trk = volume.find_translate_to(rk, optimizer=optimizer, **options)


    # Return affines for each kidney
    return {
        'BK': volume.translate(tbk, **options).affine,
        'LK': volume.translate(tlk, **options).affine,
        'RK': volume.translate(trk, **options).affine
    }


#Main Protocol: 
def _align_2d(site, table_dir=None):
    
    #define paths
    dest_dir = os.path.join(os.getcwd(), 'build', 'dce_9_coreg_dce2dixon', site, "Patients")
    png_save = os.path.join(os.getcwd(), 'build', 'dce_9_coreg_dce2dixon', site, "Patients", 'Coregistered')
    os.makedirs(png_save, exist_ok=True)

    # Logging setup
    logging.basicConfig(
    filename=os.path.join(dest_dir, 'error.log'),
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if table_dir is None:
        # Build default path
        table_path = os.path.join(os.getcwd(),'build', 'dce_9_coreg_dce2dixon', site, "Patients", f'{site}_images_masks_table.pkl')
        if os.path.isfile(table_path):
            tqdm.write('Using existing dictionary...')
            # Case 1a: Default file exists → load it
            with open(table_path, "rb") as f:
                images_and_mask_table = pickle.load(f)
        else:
            # Case 1b: No default file → resort to _get_data
            tqdm.write('Creating new dce img + mask inventory...')
            images_and_mask_table = _get_data(site, table_dir)

    
    pat_series = []
    for entry in tqdm(images_and_mask_table, desc=f'Coregisterating {site} cases...', unit='case'):
        #retrieve case_id
        case_id = entry['case_id']
        tqdm.write(f'Processing case {case_id}...')


        mask_path = entry['mask_path']
        mdr_path = entry['mdr_path']        
        auc_path = entry['auc_map_path']
        rpf_path = entry['rpf_map_path']
        avd_path = entry['avd_map_path']
        mtt_path = entry['mtt_map_path']

        bari_add_series_name(case_id, pat_series)
        database = [dest_dir, case_id, ('Baseline', 0)]
        
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

        # check and continue to next case if all series exist 
        series = [
            auc_clean_rk, auc_clean_lk, rpf_clean_rk, rpf_clean_lk,
            avd_clean_rk, avd_clean_lk, mtt_clean_rk, mtt_clean_lk,
            mdr_clean_rk, mdr_clean_lk
        ]

        if all(x in db.series(database) for x in series):
            print(f'Skipping case {case_id}. All coregistered series in {site} folder')
            continue  
          
        #__________________CHECK AND WRITE ANY MISSING CASES___________________________________#
        ###Easy build for few missing series in file by applying affine from another aligned map.
        ####This prevents from having to run through co-registration again for the few missing series.

        # 1: find the few missing series
        db_series = db.series(database)
        missing_series = [s for s in series if s not in db_series]

        if not missing_series:
            print(f"Skipping case {case_id}. All coregistered series exist in {site} folder.")
            continue

        map_dict = {
            "auc": auc_path,  # e.g. 'DCE_8_AUC_map'
            "rpf": rpf_path,
            "avd": avd_path,
            "mtt": mtt_path,
            "mdr": mdr_path
            }

        if len(missing_series) < 10: 
            print(f"Missing series for case {case_id}:")   
            for s in missing_series:
                print(f"  - {s[3][0]}")
                print('sending to quick write series')
            quick_write_series(missing_series, map_dict, db_series, site, case_id)
            continue
        
        #_________________COREGISTRATION PIPELINE BEGINS HERE_____________________#
        elif len(missing_series) == 10:
            
            #create LK/RK mask volumes
            mask_vol = db.volume(mask_path)
            bk_vol = mask_vol
            bk = bk_vol.values
            lk_arr = (bk == 1)       
            rk_arr = (bk == 2)

            lk = vreg.volume(lk_arr, bk_vol.affine)
            rk = vreg.volume(rk_arr, bk_vol.affine)

            

            # Use bounding boxes speed up the computation
            bk = bk_vol.bounding_box()
            rk = rk.bounding_box()
            lk = lk.bounding_box()

            # Load AUC for coregistration
            auc_slice_vols = db.volumes_2d(auc_path[0])


            #_________________COREGISTRATION_________________________#
            affine = {'LK':[], 'RK':[]}
            # Wrap loop with tqdm to display a progress bar
            for z in tqdm(auc_slice_vols, desc="Coregistering slices", unit="slice"):
                
                #start_time = time.time()  # Record start time
                
                # Perform the coregistration
                az = _coreg(z, bk, lk, rk)
                
                # Append the results to the affine dictionary
                affine['LK'].append(az['LK'])
                affine['RK'].append(az['RK'])
                
                #end_time = time.time()  # Record end time
                #time_taken = end_time - start_time  # Calculate time taken for this iteration
                
                #time for each slice
                #print(f"Time taken for slice: {time_taken:.1f} seconds")
            
            #_______________BUILD ALIGNED MAP VOLUMES__________________#
            # Build AUC volumes
            if auc_clean_lk not in db.series(database):
                try:
                    post_auc_lk_vols = []
                    for aff, s in zip(affine['LK'], auc_slice_vols):
                        new_slice_vol = vreg.volume(s.values, aff) 
                        db.write_volume(new_slice_vol, auc_clean_lk, ref=auc_path[0], append=True)
                        post_auc_lk_vols.append(new_slice_vol)
                except Exception as e:
                    logging.error(f'cannot build AUC LK: {e}')

            
            if auc_clean_rk not in db.series(database):
                try:
                    post_auc_rk_vols = []
                    for aff, s in zip(affine['RK'], auc_slice_vols):
                        new_slice_vol = vreg.volume(s.values, aff)
                        db.write_volume(new_slice_vol, auc_clean_rk, ref=auc_path[0], append=True)
                        post_auc_rk_vols.append(new_slice_vol)
                except Exception as e:
                    logging.error(f'cannot build AUC RK: {e}')

            # Save (and display) pre- and post-reg img and mask overlays
            vplot.overlay_2d_new(
                pre=auc_slice_vols, 
                post=post_auc_lk_vols, 
                mask=lk, 
                save_path = png_save + f'/{case_id}_LK.png', 
                show=False)
            
            vplot.overlay_2d_new(
                pre=auc_slice_vols, 
                post=post_auc_rk_vols, 
                mask=rk, 
                save_path = png_save + f'/{case_id}_RK.png', 
                show=False)
            
            
            # Load LK/RK affines from post_auc lk/rk vols
            lk_affine = []
            for vol in post_auc_lk_vols:
                affine = vol.affine
                lk_affine.append(affine) 
            
            rk_affine = []
            for vol in post_auc_rk_vols:
                affine = vol.affine
                rk_affine.append(affine)


            # Create MDR volume
            tqdm.write('Building aligned MDR series...')
            mdr_slice_vols = db.volumes_2d(mdr_path, 'AcquisitionTime')
            if mdr_clean_rk not in db.series(database):
                try:
                    for aff, s in zip(affine['RK'], mdr_slice_vols):
                        new_slice_vol = s.set_affine(aff)
                        db.write_volume(new_slice_vol, mdr_clean_rk, ref=mdr_path, append=True)
                except Exception as e:
                    logging.error(f'cannot build MDR RK: {e}')

            if mdr_clean_lk not in db.series(database):
                try:
                    for aff, s in zip(affine['LK'], mdr_slice_vols):
                        new_slice_vol = s.set_affine(aff)
                        db.write_volume(new_slice_vol, mdr_clean_lk, ref=mdr_path, append=True)
                except Exception as e:
                    logging.error(f'cannot build MDR LK: {e}')        

            tqdm.write('Building aligned RPF series...')
            rpf_volume = db.volumes_2d(rpf_path[0])
            # Assign affine and build rpf lk/rk vols       
            if rpf_clean_rk not in db.series(database):
                try:
                    for aff, s in zip(rk_affine, rpf_volume):
                        new_slice_vol = vreg.volume(s.values, aff)
                        db.write_volume(new_slice_vol, rpf_clean_rk, ref=rpf_path[0], append=True)  
                except Exception as e:
                    logging.error(f'cannot build RPF RK: {e}')      
            
            if rpf_clean_lk not in db.series(database):
                try:
                    for aff, s in zip(lk_affine, rpf_volume):
                        new_slice_vol = vreg.volume(s.values, aff)
                        db.write_volume(new_slice_vol, rpf_clean_lk, ref=rpf_path[0], append=True)        
                except Exception as e:
                    logging.error(f'cannot build RPF LK: {e}')

            tqdm.write('Building aligned AVD series....')
            avd_volume = db.volumes_2d(avd_path[0])        
            if avd_clean_rk not in db.series(database):
                try:
                    for aff, s in zip(rk_affine, avd_volume):
                        new_slice_vol = vreg.volume(s.values, aff)
                        db.write_volume(new_slice_vol, avd_clean_rk, ref=avd_path[0], append=True)
                except Exception as e:
                    logging.error(f'cannot build AVD RK: {e}')
            
            if avd_clean_lk not in db.series(database):
                try:
                    for aff, s in zip(lk_affine, avd_volume):
                        new_slice_vol = vreg.volume(s.values, aff)
                        db.write_volume(new_slice_vol, avd_clean_lk, ref=avd_path[0], append=True) 
                except Exception as e:
                    logging.error(f'cannot build AVD LK: {e}')       
            

            tqdm.write('Building aligned MTT series....')
            mtt_volume = db.volumes_2d(mtt_path[0])
            if mtt_clean_rk not in db.series(database):
                try:
                    for aff, s in zip(rk_affine, mtt_volume):
                        new_slice_vol = vreg.volume(s.values, aff)
                        db.write_volume(new_slice_vol, mtt_clean_rk, ref=mtt_path[0], append=True)
                except Exception as e:
                    logging.error(f'cannot build MTT RK: {e}')              

            if mtt_clean_lk not in db.series(database):
                try:
                    for aff, s in zip(lk_affine, mtt_volume):
                        new_slice_vol = vreg.volume(s.values, aff)
                        db.write_volume(new_slice_vol, mtt_clean_lk, ref=mtt_path[0], append=True)
                except Exception as e:
                    logging.error(f'cannot build MTT LK: {e}')    



            tqdm.write('Coregistration complete!')
        
def quick_write_series(missing_series, map_dict, db_series, site, case_id):         
    tqdm.write(f'Processing case {case_id}...')
    map_lookup = map_dict 
            
    auc_path = map_dict['auc']
    rpf_path = map_dict['rpf']
    avd_path = map_dict['avd']
    mtt_path = map_dict['mtt']
    mdr_path = map_dict['mdr']

    #dir paths
    dest_dir = os.path.join(os.getcwd(), 'build', 'dce_9_coreg_dce2dixon', site, "Patients")

    pat_series =[]
    bari_add_series_name(case_id, pat_series)
    database = [dest_dir, case_id, ('Baseline', 0)]
    
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



    # 2: determine reference series (first available for rk and lk)
    series_rk = [auc_clean_rk, rpf_clean_rk, avd_clean_rk, mtt_clean_rk, mdr_clean_rk]
    series_lk = [auc_clean_lk, rpf_clean_lk, avd_clean_lk, mtt_clean_lk, mdr_clean_lk]

    ref_rk = next((x for x in series_rk if x in db_series), None)
    ref_lk = next((x for x in series_lk if x in db_series), None)

 
    # 1. Handle missing series
    for s in missing_series:
        side = "rk" if "_rk" in s[3][0] else "lk"

      # 2: load affines if available
        if side == 'rk':
            if ref_rk:
                affine_rk = []
                ref_rk_vols = db.volumes_2d(ref_rk)
                for vol in ref_rk_vols:
                    affine_rk.append(vol.affine)
                print(f"Using {ref_rk[3][0]} affine for right kidney.")
            else:
                print("No RK reference series available.")
        elif side == 'rk':
            if ref_lk:
                affine_lk = []
                ref_lk_vols = db.volumes_2d(ref_lk)
                for vol in ref_lk_vols:
                    affine_lk.append(vol.affine)
                print(f"Using {ref_lk[3][0]} affine for left kidney.")
            else:
                print("No LK reference series available.")
        affine = affine_rk if side == "rk" else affine_lk

        if affine is None:
            print(f"No affine found for {s[3][0]}, skipping.")
            continue

        # 3. Determine which map to use (based on series name)
        if "auc".lower() in s[3][0]:
            map = map_lookup["auc"]
            ref_series = auc_path[0]
            if side == 'lk':
                series_name = auc_clean_lk
            elif side == 'rk':
                series_name = auc_clean_rk
        elif "rpf".lower() in s[3][0]:
            map = map_lookup["rpf"]
            ref_series = rpf_path[0]
            if side == 'lk':
                series_name = rpf_clean_lk
            elif side == 'rk':
                series_name = rpf_clean_rk                
        elif "avd".lower() in s[3][0]:
            map = map_lookup["avd"]
            ref_series = avd_path[0]
            if side == 'lk':
                series_name = avd_clean_lk
            elif side == 'rk':
                series_name = avd_clean_rk                
        elif "mtt".lower() in s[3][0]:
            map = map_lookup["mtt"]
            ref_series = mtt_path[0]
            if side == 'lk':
                series_name = mtt_clean_lk
            elif side == 'rk':
                series_name = mtt_clean_rk
        elif "mdr".lower() in s[3][0]:
            map = map_lookup["mdr"]
            ref_series = mdr_path
            if side == 'lk':
                series_name = mdr_clean_lk
            else:
                series_name = mdr_clean_rk
        else:
            print(f"Cannot determine map type for {s[3][0]}, skipping.")
            continue

        print(f"Assigning {s[3][0]} to map {map[0][3][0]} using {side.upper()} affine.")

        # 4. Load the map (placeholder for your actual map loading)
        if map == 'mdr':
            map_vols = db.volumes_2d(map, 'AcquisitionTime')
        else:
            map_vols = db.volumes_2d(map[0])
        
        if map_vols:
            try:
                for aff, s in zip(affine, map_vols):
                        new_slice_vol = vreg.volume(s.values, aff) 
                        db.write_volume(new_slice_vol, series_name, ref=ref_series, append=True)
            except Exception as e:
                logging.error(f'cannot build {s[3][0]}: {e}')
    tqdm.write(f'All Aligned volumes saved in {case_id} folder')

if __name__ == '__main__':
    #_get_data('Bari')
    _align_2d('Bari')

