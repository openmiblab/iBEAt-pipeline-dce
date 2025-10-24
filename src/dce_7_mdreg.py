import os
import numpy as np
import pandas as pd
import logging
import pickle
import mdreg
import dcmri
import dbdicom as db
from tqdm import tqdm
import vreg
import re



def add_series_name(folder, all_series: list):
    new_series_name = "DCE_7_"
    all_series.append(new_series_name)
    return new_series_name


def load_database(site):
    

    data_path = os.path.join(os.getcwd(), 'build', 'dce_2_data', site, 'Patients')

    database = db.series(data_path)
    DCE_kidneys = [entry for entry in database if entry[3][0].strip().lower() == 'dce_1_kidneys']
    images = []
    for study in DCE_kidneys:
        if site == 'Sheffield':
            volume = db.volume(study, dims='TriggerTime')
        else:
            volume =db.volume(study, dims=['AcquisitionTime'])
        images.append((study[1], study, volume))
    return images

def load_aif(site):

    csv_path = os.path.join(os.getcwd(), 'build', 'dce_5_aorta2csv', site, "Patients")


    csv_files = [f for f in os.listdir(csv_path) if f.endswith("_aif.csv")]
    if not csv_files:
        print("No CSV files found in folder:", csv_path)

    aif_values = []
    for csv_file in csv_files:
        case_id = csv_file.replace("_aif.csv", "")
        df = pd.read_csv(os.path.join(csv_path, csv_file))
        times = np.array(df["Time (s)"]).flatten()
        aif = np.array(df["AIF Intensity"]).flatten()
        aif_values.append((case_id, times, aif))
    return aif_values


def rebuild_mdr_table(site, batch_no=None):
    base_dir = os.path.join(os.getcwd(), 'build', 'dce_7_mdreg', site, "Patients")
    study_list = db.series(os.path.join(os.getcwd(), 'build', 'dce_2_data', site, "Patients"))

    moco_table = []

    # Loop through all case folders
    case_ids = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for case_id in case_ids:
        checkpoint_dir = os.path.join(base_dir, case_id, 'Checkpoint')
        if not os.path.exists(checkpoint_dir):
            print(f"No checkpoint dir for {case_id}, skipping.")
            continue

        # Get all *_coreg_iterX.npy files for this case
        coreg_files = [
            f for f in os.listdir(checkpoint_dir)
            if f.startswith(f"{case_id}_coreg_iter") and f.endswith(".npy")
        ]
        if not coreg_files:
            print(f"No coreg files for {case_id}, skipping.")
            continue

        # Extract iteration numbers from filenames
        iter_numbers = []
        for f in coreg_files:
            match = re.search(r"_iter_(\d+)\.npy$", f)
            if match:
                iter_numbers.append(int(match.group(1)))

        if not iter_numbers:
            print(f"No iteration numbers found for {case_id}, skipping.")
            continue

        # Last iteration = max number
        last_iter = max(iter_numbers)

        # Collect latest files for coreg, defo, model_fit, pars
        paths = {}
        for key in ["coreg", "defo", "model_fit", "pars"]:
            all_files = sorted([
                os.path.join(checkpoint_dir, f) 
                for f in os.listdir(checkpoint_dir) 
                if f.startswith(f"{case_id}_{key}_iter")
            ])
            paths[key] = all_files[-1:] if all_files else []

        # Require coreg to be present
        if not all(paths[k] for k in ["coreg"]):
            print(f"Skipping {case_id}, missing coreg file(s).")
            continue

        # Lookup study from db
        study = next((s for s in study_list if s[1] == case_id), None)
        if study is None:
            print(f"Study not found for {case_id}, skipping.")
            continue

        moco_table.append({
            "case_id": study[1],
            "study": study,
            "paths": paths,
            "iteration": last_iter
        })

    # Save one global checkpoint for all cases
    global_checkpoint = os.path.join(
        base_dir,
        f"{site}_moco_table{f'_{batch_no}' if batch_no is not None else ''}.pkl"
    )
    with open(global_checkpoint, "wb") as f:
        pickle.dump(moco_table, f)

    print(f"Rebuilt moco_table with {len(moco_table)} cases, saved to {global_checkpoint}")


# Step 1: Collect database 
def dce_to_process(site, batch_no=None):
    # Define directories
    save_dir = os.path.join(os.getcwd(), 'build', 'dce_7_mdreg', site, "Patients")
    os.makedirs(save_dir, exist_ok=True)
    pickle_path = os.path.join(save_dir, f"{site}_moco_table{f'_{batch_no}' if batch_no is not None else ''}.pkl")
    data_dir = os.path.join(os.getcwd(), 'build', 'dce_2_data', site, "Patients")


    # #Check existing series via dbtree
    existing_series = db.series(save_dir)
    done_cases_map = {}
    for s in existing_series:
        case_id = s[1]
        series_name_in_db = s[2][0]
        done_cases_map.setdefault(case_id, set()).add(series_name_in_db)

    # Minimal series prefix function
    def series_prefix(case):
        pat_series = []
        prefix = add_series_name(case, pat_series)
        return prefix


    database = db.series(data_dir)
    DCE_kidneys = [entry for entry in database if entry[3][0].strip().lower() == 'dce_1_kidneys'.lower()]
    

    for entry in DCE_kidneys:
        case_id = entry[1]
        study = entry

        # Check if series exists in completed mdr dir
        prefix = series_prefix(case_id)
        required_series = {prefix + "mdr_moco", prefix + "mdr_defo", prefix + "mdr_fit"}
        done_for_case = done_cases_map.get(case_id, set())
        if required_series.issubset(done_for_case):
            print(f"Skipping case {case_id} (all series done)")
            continue
        
        #Define starting iteration 
        iteration = 0

        # Load existing data if file exists
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                images_table = pickle.load(f)
        else:
            images_table = []

        # Remove any old entries for this case
        images_table = [e for e in images_table if e["case_id"] != case_id]

        # Append the new case
        images_table.append({
            'case_id': case_id,
            'study': study,
            'iteration': iteration
        })

        # Write the updated list back
        with open(pickle_path, "wb") as f:
            pickle.dump(images_table, f)

        print(f"Saved {case_id} for 3D MDREG to {site} folder")
      
        
def _mdr_2d(site, batch_no=None, maximum_it=5, start_case=0, end_case=None):


    mdr_table_path = os.path.join(os.getcwd(), 'build', 'dce_7_mdreg', site, 'Patients', f"{site}_moco_table{f'_{batch_no}' if batch_no is not None else ''}.pkl")

    if os.path.exists(mdr_table_path):
        with open(mdr_table_path, "rb") as f:
            mdr_table = pickle.load(f)
    else:
        print('No existing MDR Table to process')

    if end_case is None:
        end_case = len(mdr_table)
    
    batch_table = mdr_table[start_case:end_case+1]

    first_case_id = batch_table[0]['case_id']
    last_case_id = batch_table[-1]['case_id']
    tqdm.write(f"Processing {len(batch_table)} cases from {first_case_id} to {last_case_id}")

        
    for entry in tqdm(batch_table, desc=f'Processing {site} MDREG', unit='case'):
        
        # Working Checkpoint Directory
        case = entry['case_id']
        checkpoint_dir = os.path.join(os.getcwd(), 'build', 'dce_7_mdreg', site, 'Patients', case, 'Checkpoint')
        os.makedirs(checkpoint_dir, exist_ok=True)

        def existing_paths(checkpoint_dir, case, prefix, ext, maxit=5):
                return [
                        os.path.join(checkpoint_dir, f"{case}_{prefix}_iter_{i}{ext}")
                        for i in range(1, maxit+1)
                        if os.path.exists(os.path.join(checkpoint_dir, f"{case}_{prefix}_iter_{i}{ext}"))
                        ]

        # WORKING CASE - BEGIN MDREG
        study = entry['study']
        iteration = entry['iteration'] 
        times_and_aif = load_aif(site)
        aif = next((v for v in times_and_aif if v[0] == case), None)
        time = aif[1].tolist()
        aif_values = aif[2].tolist()
        
        try:
            if iteration == 0:
                if site == 'Bari':
                    img_vol = db.volume(study, dims=['AcquisitionTime'])
                    array = img_vol.values
                elif site == 'Bordeaux':
                    img_vol = db.volumes_2d(study, dims=['AcquisitionTime'])
                    array = []
                    for vol in img_vol:
                        arr = vol.values
                        array.append(arr)
                    array = np.stack(array, axis=2)
                    array = array.squeeze()
        except Exception as e:
            tqdm.write(f'Skipping case {case}. Cannot load volume. {e}')
            continue
        
        if iteration >= maximum_it:
            tqdm.write(f"Case {case} already completed {maximum_it} iterations, skipping.")
            continue

        fit_image = {
                'func': dcmri.pixel_2cfm_linfit,
                'aif': aif_values,
                'time': time,
                'baseline': 15
            }
   
        for iter_num in range(iteration + 1, maximum_it + 1):
            tqdm.write(f'Processing {case}, Iteration={iter_num}')
            # reload from previous iteration
            if iter_num > 1:
                array = np.load(os.path.join(checkpoint_dir, f"{case}_coreg_iter_{iter_num-1}.npy"))
            
            # mdreg
            try:

                coreg, model_fit, defo, pars = mdreg.fit(
                    moving=array,
                    fit_image=fit_image,
                    force_2d=True,
                    maxit=1,
                    verbose=2,
                    )

                #visualise + save mdreg result

                coreg_total, defo_total, model_fit_total, pars_total = coreg, defo, model_fit, pars

                # Save per iteration
                np.save(os.path.join(checkpoint_dir, f"{case}_coreg_iter_{iter_num}.npy"), coreg_total)
                np.save(os.path.join(checkpoint_dir, f"{case}_defo_iter_{iter_num}.npy"), defo_total)
                np.save(os.path.join(checkpoint_dir, f"{case}_model_fit_iter_{iter_num}.npy"), model_fit_total)
                with open(os.path.join(checkpoint_dir, f"{case}_pars_iter_{iter_num}.pkl"), "wb") as f:
                    pickle.dump(pars_total, f)

                tqdm.write(f"Saved iteration {iter_num} for case {case}")
                if iter_num in (1, 5):
                    print('Making animations...')
                    mdreg.plot.series(array, model_fit_total, coreg_total, path=checkpoint_dir, filename=f'MDREG_{case}_{iter_num}', show=False)

            except Exception as e:
                logging.error(f"Iteration {iter_num} failed for case {case}: {e}")
                break
        

            entry['paths'] = {
                    "coreg": existing_paths(checkpoint_dir, case, "coreg", ".npy"),
                    "defo": existing_paths(checkpoint_dir, case, "defo", ".npy"),
                    "model_fit": existing_paths(checkpoint_dir, case, "model_fit", ".npy"),
                    "pars": existing_paths(checkpoint_dir, case, "pars", ".pkl"),
                    }
            entry['iteration'] = iter_num 


            # Save checkpoint after each case
            with open(mdr_table_path, "wb") as f:
                pickle.dump(mdr_table, f)

            tqdm.write(f"Case {case} checkpoint saved. Total completed: {iter_num}")    

def _mdr_3d(site, batch_no=None, maximum_it=5, start_case=0, end_case=None):



    mdr_table_path = os.path.join(os.getcwd(), 'build', 'dce_7_mdreg', site, 'Patients', f"{site}_moco_table{f'_{batch_no}' if batch_no is not None else ''}.pkl")

    if os.path.exists(mdr_table_path):
        with open(mdr_table_path, "rb") as f:
            mdr_table = pickle.load(f)
    else:
        print('No existing MDR Table to process')
    
    if end_case is None:
        end_case = len(mdr_table)

    batch_table = mdr_table[start_case:end_case+1]

    first_case_id = batch_table[0]['case_id']
    last_case_id = batch_table[-1]['case_id']
    tqdm.write(f"Processing {len(batch_table)} cases from {first_case_id} to {last_case_id}")

    for entry in tqdm(batch_table, desc=f'Processing Model Registration for {site} cases', unit='case'):
        
        # Working Checkpoint Directory
        case = entry['case_id']
        checkpoint_dir = os.path.join(os.getcwd(), 'build', 'dce_7_mdreg', site, 'Patients', case, 'Checkpoint')
        os.makedirs(checkpoint_dir, exist_ok=True)

        def existing_paths(checkpoint_dir, case, prefix, ext, maxit=5):
            return [
                    os.path.join(checkpoint_dir, f"{case}_{prefix}_iter_{i}{ext}")
                    for i in range(1, maxit+1)
                    if os.path.exists(os.path.join(checkpoint_dir, f"{case}_{prefix}_iter_{i}{ext}"))
                    ]

        # WORKING CASE - BEGIN MDREG
        study = entry['study'],
        iteration = entry['iteration']
        times_and_aif = load_aif(site)
        aif = load_aif(site)
        aif = next((v for v in times_and_aif if v[0] == case), None)
        aif_values = aif[2].tolist()
        
        try:
            if iteration == 0:
                img_vol = db.volume(study[0], dims=['TriggerTime'])
                array = img_vol.values
        except Exception as e:
            tqdm.write(f'Skipping case {case}. Cannot load volume. {e}')
            continue            
        
        if iteration >= maximum_it:
            tqdm.write(f"Case {case} already completed {maximum_it} iterations, skipping.")
            continue

        fit_deconv = {
                'func': mdreg.fit_deconvolution,
                'aif': aif_values,
                'n0': 15,
                'tol': 0.2,
            }
        fit_skimage = {
                'package': 'skimage',
                'attachment': 10,
                'parallel': False,
                'progress_bar': True,  
            }
        
            # # Coregistration options
        fit_ants = {
            'package': 'ants',
            'type_of_transform': 'SyNOnly',
            'parallel': False,
            'progress_bar': True,  
        }        

        for iter_num in range(iteration + 1, maximum_it + 1):
            tqdm.write(f'Processing {case}, Iteration={iter_num}')
            # reload from previous iteration
            if iter_num > 1:
                array = np.load(os.path.join(checkpoint_dir, f"{case}_coreg_iter_{iter_num-1}.npy"))
            
            # mdreg
            try:

                coreg, defo, model_fit, pars = mdreg.fit(
                    moving=array,
                    fit_coreg=fit_ants,
                    fit_image=fit_deconv,
                    maxit=1,
                    verbose=2,
                    )

                coreg_total, defo_total, model_fit_total, pars_total = coreg, defo, model_fit, pars

                # Save per iteration
                np.save(os.path.join(checkpoint_dir, f"{case}_coreg_iter_{iter_num}.npy"), coreg_total)
                np.save(os.path.join(checkpoint_dir, f"{case}_defo_iter_{iter_num}.npy"), defo_total)
                np.save(os.path.join(checkpoint_dir, f"{case}_model_fit_iter_{iter_num}.npy"), model_fit_total)
                with open(os.path.join(checkpoint_dir, f"{case}_pars_iter_{iter_num}.pkl"), "wb") as f:
                    pickle.dump(pars_total, f)

                tqdm.write(f"Saved iteration {iter_num} for case {case}")
                if iter_num in (1, 5):
                    print('Making animations...')
                    if site == 'Sheffield':
                        mdreg.plot.animation(array, path=checkpoint_dir, filename=f'MDREG_moving_{case}_{iter_num}', show=False)
                        mdreg.plot.animation(coreg, path=checkpoint_dir, filename=f'MDREG_coreg_{case}_{iter_num}', show=False)
                    else:
                        mdreg.plot.series(array, model_fit_total, coreg_total, path=checkpoint_dir, filename=f'MDREG_{case}_{iter_num}', show=False)

            except Exception as e:
                logging.error(f"Iteration {iter_num} failed for case {case}: {e}")
                break
        
            
            entry['paths'] = {
                    "coreg": existing_paths(checkpoint_dir, case, "coreg", ".npy"),
                    "defo": existing_paths(checkpoint_dir, case, "defo", ".npy"),
                    "model_fit": existing_paths(checkpoint_dir, case, "model_fit", ".npy"),
                    "pars": existing_paths(checkpoint_dir, case, "pars", ".pkl"),
                    }
            entry['iteration'] = iter_num 


            # Save checkpoint after each case
            with open(mdr_table_path, "wb") as f:
                pickle.dump(mdr_table, f)

            tqdm.write(f"Case {case} checkpoint saved. Total completed: {iter_num}")  


def write_2_folder(site, batch_no=None):
    mdr_table_path = os.path.join(os.getcwd(), 'build', 'dce_7_mdreg', site, 
        'Patients', f"{site}_moco_table{f'_{batch_no}' if batch_no is not None else ''}.pkl")
    
    if os.path.exists(mdr_table_path):
        with open(mdr_table_path, "rb") as f:
            mdr_table = pickle.load(f)
    else:
        print('No existing MDR Table to process')

    base_dir = os.path.join(os.getcwd(), 'build', 'dce_7_mdreg', site, "Patients")
    os.makedirs(base_dir, exist_ok=True)
    


    for entry in tqdm(mdr_table, desc=f'Writing {site} cases to DICOM', unit='case'):
        #case_id
        case = entry['case_id']
        tqdm.write(f'Processing case {case}...')
        
        # series naming and check/skip if it already exists 
        pat_series = []
        add_series_name(case, pat_series)
        database = [base_dir, case, ('Baseline', 0)]
        mdr_clean = database + [(pat_series[-1] + "mdr_moco", 0)]
        defo_clean = database + [(pat_series[-1] + 'mdr_defo', 0)]
        model_fit_clean = database + [(pat_series[-1] + "mdr_fit", 0)]
        
        series = [mdr_clean, defo_clean, model_fit_clean]       
        if all(x in db.series(database) for x in series):
            print(f'Skipping case {case}. All MDREG DICOMs already in {site} folder')
            continue


        
        #check for completed iterations
        iteration = entry['iteration']
        if iteration < 5:
            print(f'Skipping case {case}. Model registration incomplete or in different batch. No of Iteration(s) = {iteration}.')
            continue
        
        #load study and array paths to dynamic and moco image
        study = entry["study"]  
        paths = entry["paths"]



        #locate last mdreg iteration
        try:
            # Helper to find last iteration file
            def last_file(file_list):
                for f in reversed(file_list):
                    if os.path.exists(f):
                        return f
                return None

            coreg_path = last_file(paths["coreg"])
            defo_path = last_file(paths["defo"])
            model_fit_path = last_file(paths["model_fit"])

            if not coreg_path:
                logging.warning(f"No completed iteration found for case {case}, skipping write.")
                continue

            # Load arrays
            if coreg_path:
                coreg = np.load(coreg_path)
            #print('coreg:', coreg.shape)
            if model_fit_path:
                model_fit = np.load(model_fit_path)
            #print('mf:', model_fit.shape)
            if defo_path:
                defo = np.load(defo_path)
            #print('defo:', defo.shape)

            if model_fit is not None and model_fit.ndim > 5:
                model_fit, defo = defo, model_fit
                print(f'Checkpoint outputs Defo and Model fit are swapped for case {case}. Please Check. Switching...')

            # Extract metadata from image
            if site =='Bari':
                image = db.volume(study, 'AcquisitionTime')
                if image is not None:
                    affine = image.affine
                    coords = image.coords
            elif site=='Bordeaux':
                image_vols = db.volumes_2d(study, 'AcquisitionTime')
                if image_vols is not None:
                    for vol in image_vols:
                        affine = vol.affine
                        coords = vol.coords
                        break
            elif site=='Sheffield':
                image_vol = db.volume(study, 'TriggerTime')
                if image_vol is not None:
                    affine = image_vol.affine
                    coords = image_vol.coords

            #_______________MOCO________________

            tqdm.write('Building MoCo series...')
            #vol = coreg 
            if site in ('Bari', 'Bordeaux'):
                try:
                    if coreg is not None:
                        if mdr_clean not in db.series(base_dir):
                            volume = vreg.volume(coreg, affine, coords, dims=['AcquisitionTime'])
                            db.write_volume(volume, mdr_clean, ref=study, append=True)
                except Exception as e:
                    logging.error(f'Cannot build MoCo series: {e}')     
            elif site == 'Sheffield':
                try:
                    if coreg is not None:
                        if mdr_clean not in db.series(base_dir):
                            volume = vreg.volume(coreg, affine, coords, dims=['TriggerTime'])
                            db.write_volume(volume, mdr_clean, ref=study, append=True)
                except Exception as e:
                    logging.error(f'Cannot build MoCo series: {e}')                    


            # elif site == 'Leeds':
            #     volume = vreg.volume(, affine, coords, dims=['InstanceNumber'])
            #     db.write_volume(volume, mdr_clean, ref=study, append=True)

            #_______________DEFORMATION________________
            tqdm.write('Building Defo series...')
            try:
                if defo is not None:
                    if model_defo not in db.series(base_dir):
                        model_defo = mdreg.defo_norm(defo, 'eumip')
                        db.write_volume((model_defo, affine), defo_clean, ref=study, append=True)
            except Exception as e:
                logging.error(f'Cannot build Defo series: {e}') 

            #_______________MODEL FIT________________
            tqdm.write('Building Model Fit series...')
            if site == 'Bari':
                try:
                    if model_fit is not None:
                        if model_fit_clean not in db.series(base_dir):
                            volume = vreg.volume(model_fit, affine, coords, dims=['AcquisitionTime'])
                            db.write_volume(volume, model_fit_clean, ref=study, append=True)            
                except Exception as e:
                    logging.error(f'Cannot build Model Fit series: {e}')
                    
                
                    

            tqdm.write(f"Case {case} written successfully.")

            # keep final iteration and remove older arrays to save space 
            for f_list in [paths["coreg"], paths["defo"], paths["model_fit"], paths["pars"]]:
                last_f = last_file(f_list)
                for f in f_list:
                    if os.path.exists(f) and f != last_f:
                        os.remove(f)

        except Exception as e:
            logging.error(f"Case {case} cannot be written: {e}")



if __name__ == '__main__':
    #Step: 1 - Prep DCE
    #dce_to_process('Sheffield')

    #Step: 2 - 2D/3D MDREG 
    #_mdr_2d('Bordeaux')
    #_mdr_3d('Sheffield')


    #Optional - Recommended if running in batches, before writing files to DICOM
    rebuild_mdr_table('Sheffield')

    # Step 3 - Write Files to DICOM
    write_2_folder('Sheffield')

