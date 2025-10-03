import os
import dbdicom as db
import dcmri
import numpy as np
import pandas as pd
import logging
import vreg



def add_series_name(folder, all_series: list):
    new_series_name = "DCE_8_"
    all_series.append(new_series_name)
    return new_series_name


def DCE(site):
    #base_dir = os.path.join(os.getcwd(), 'build', 'dce_7_mdreg', site, "Patients")
    base_dir = os.path.join(os.getcwd(), 'build', 'dce_2_data', site, "Patients")
    aif_path = os.path.join(os.getcwd(), 'build', 'dce_5_aorta2csv', site, "Patients")
    dstdatapath = os.path.join(os.getcwd(), 'build', 'dce_8_mapping')
    destpath =  os.path.join(dstdatapath, site, "Patients")
    os.makedirs(destpath, exist_ok=True)

    # Logging setup
    logging.basicConfig(
    filename=os.path.join(destpath, 'error.log'),
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    database = db.series(base_dir)
    #DCE_mdr_moco = [entry for entry in database if entry[3][0].strip().lower() == 'dce_7_mdr_moco']
    DCE_mdr_moco = [entry for entry in database if entry[3][0].strip().lower() == 'dce_1_kidneys']

    csv_files = [f for f in os.listdir(aif_path) if f.endswith("_aif.csv")]
    pat_series = []
    for study in DCE_mdr_moco:
        study_id = study[1]  #case ID
        
        # find matching csv
        aif_file = next((f for f in csv_files if study_id in f), None)

        if aif_file:
            try:

                
                #create folder
                series_name = add_series_name(study[1], pat_series)
                baseline_path = [destpath, study[1], ('Baseline', 0)]

                max_clean = baseline_path + [(series_name + "MAX_map", 0)] 
                auc_clean = baseline_path + [(series_name + 'AUC_map', 0)]
                att_clean = baseline_path + [(series_name + 'ATT_map', 0)]                
                
                fit_clean = baseline_path + [(series_name + 'fit', 0)]
                
                rpf_clean = baseline_path + [(series_name + 'RPF_map', 0)]
                avd_clean = baseline_path + [(series_name + 'AVD_map', 0)]
                mtt_clean = baseline_path + [(series_name + 'MTT_map', 0)]

                fp_clean = baseline_path + [(series_name + 'FP_map', 0)]
                tp_clean = baseline_path + [(series_name + 'TP_map', 0)]
                vp_clean = baseline_path + [(series_name + 'VP_map', 0)]
                
                ft_clean = baseline_path + [(series_name + 'FT_map', 0)]
                tt_clean = baseline_path + [(series_name + 'TT_map', 0)]

                #check + skip if files processed
                series = [max_clean, auc_clean, att_clean, fit_clean, rpf_clean, avd_clean, mtt_clean, 
                          fp_clean, tp_clean, vp_clean, ft_clean, tt_clean]
                
                if all(x in db.series(baseline_path) for x in series):
                    continue

                #load mdr_moco_dicom  #####TEMP FIX######
                if site == 'Leeds':
                    #array = np.load('build/dce_7_mdreg/Leeds/Patients/mdr_prep/4128_056_coreg_iter_5.npy')
                    mdr_moco = db.split_series(study, 'ImagePositionPatient')
                    mdr_moco = sorted(mdr_moco, key=lambda x: x[0])
                    image_vol = []
                    for _, x in mdr_moco:
                        vol = db.volume(x, 'InstanceNumber')
                        #vol = vreg.volume(x.values, x.affine)
                        vol_array = vol.values.squeeze()
                        image_vol.append(vol_array)
                    array = np.stack(image_vol, axis=2)
                    #array = np.transpose(array, (0, 1, 3, 2))
                elif site == 'Bari':
                    vol = db.volume(study, dims=['AcquisitionTime'])
                array = array
                affine = vol.affine
                coords = vol.coords

                # load aif curve from csv
                aif_data = pd.read_csv(os.path.join(aif_path, aif_file))
                aif = aif_data.iloc[:, 1].values  # adjust if column index differs
                time = aif_data.iloc[:, 0].values
                dt = np.mean(np.diff(time)) 

                #calculating maps
                MAX, AUC, ATT, SO = dcmri.pixel_descriptives(array, aif, dt, baseline=15)
                RPF, AVD, MTT = dcmri.pixel_deconvolve(array, aif, dt, baseline=15, regpar=0.1)
                fit, par = dcmri.pixel_2cfm_linfit(array, aif, time, baseline=15)
              
                if max_clean not in db.series(baseline_path):
                    try:
                        print(f'Computing max map for {study_id}')
                        MAX[MAX<0]=0
                        MAX[MAX>10000]=10000
                        db.write_volume((MAX, affine), max_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process MAX map for {study_id}: {e} ')
                       
                if auc_clean not in db.series(baseline_path):
                    try:
                        print(f'Computing auc map for {study_id}')
                        AUC[AUC<0]=0
                        AUC[AUC>10000]=10000
                        db.write_volume((AUC, affine), auc_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process AUC map for {study_id}: {e} ')
                
                if att_clean not in db.series(baseline_path):   
                    
                    try:
                        print(f'Computing att map for {study_id}')
                        ATT[ATT<0]=0
                        ATT[ATT>10000]=10000
                        db.write_volume((ATT, affine), att_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process ATT map for {study_id}: {e} ')                         
                
                if fit_clean not in db.series(baseline_path):

                    try:
                        print(f'Computing fit map for {study_id}')
                        volume = vreg.volume(fit, affine, coords, dims=['AcquisitionTime'])

                        db.write_volume(volume, fit_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process fit map for {study_id}: {e} ') 
                
                if rpf_clean not in db.series(baseline_path):

                    try:
                        print(f'Computing rpf map for {study_id}')
                        RPF[RPF<0]=0
                        RPF[RPF>10000]=10000
                        db.write_volume((RPF, affine), rpf_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process RPF map for {study_id}: {e} ') 

                
                if avd_clean not in db.series(baseline_path):

                    try:
                        print(f'Computing avd map for {study_id}')
                        AVD[AVD<0]=0
                        AVD[AVD>10000]=10000
                        db.write_volume((AVD, affine), avd_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process AVD map for {study_id}: {e} ') 


                if mtt_clean not in db.series(baseline_path):

                    try:
                        print(f'Computing mtt map for {study_id}')
                        MTT[MTT<0]=0
                        MTT[MTT>10000]=10000
                        db.write_volume((MTT, affine), mtt_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process mtt map for {study_id}: {e} ') 

                if fp_clean not in db.series(baseline_path):

                    try:
                        print(f'Computing fp map for {study_id}')
                        par_0 = par[...,0]
                        db.write_volume((par_0, affine), fp_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process fp map for {study_id}: {e}') 

                if tp_clean not in db.series(baseline_path):

                    try:
                        print(f'Computing tp map for {study_id}')
                        par_1 = par[...,1]
                        db.write_volume((par_1, affine), tp_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process tp map for {study_id}: {e}') 

                if vp_clean not in db.series(baseline_path):

                    try:
                        print(f'Computing vp map for {study_id}')
                        par_0_1 = par[...,0]*par[...,1]/60

                        db.write_volume((par_0_1, affine), vp_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process vp map for {study_id}: {e}') 

                if ft_clean not in db.series(baseline_path):

                    try:
                        print(f'Computing ft map for {study_id}')
                        par_2 = par[...,2]
                        db.write_volume((par_2, affine), ft_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process ft map for {study_id}: {e}')  

                if tt_clean not in db.series(baseline_path):

                    try:
                        print(f'Computing tt map for {study_id}')
                        par_3 = par[...,3]
                        db.write_volume((par_3, affine), tt_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process tt map for {study_id}: {e}')                         


            except Exception as e:
                print(f"Error processing {study_id}: {e}")


 

if __name__ == '__main__':
    DCE('Leeds')