import os
import dbdicom as db
import dcmri
import numpy as np
import pandas as pd
import logging
import vreg
from tqdm import tqdm


#Helper: Build series name 
def add_series_name(folder, all_series: list):
    new_series_name = "DCE_8_"
    all_series.append(new_series_name)
    return new_series_name

# Main Protocol
def Mapping(site):
    
    #data directories
    data_dir = os.path.join(os.getcwd(), 'build', 'dce_2_data', site, "Patients")
    base_dir = os.path.join(os.getcwd(), 'build', 'dce_7_mdreg', site, "Patients")
    aif_path = os.path.join(os.getcwd(), 'build', 'dce_5_aorta2csv', site, "Patients")

    #save dir
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
    
    #load mdr database 
    database = db.series(base_dir)
    DCE_mdr_moco = [entry for entry in database if entry[3][0].strip().lower() == 'dce_7_mdr_moco'.lower()]

    csv_files = [f for f in os.listdir(aif_path) if f.endswith("_aif.csv")]
    pat_series = []
    for study in tqdm(DCE_mdr_moco, desc=f'Creating Maps for {site}', unit='case'):
        case_id = study[1]
        tqdm.write(f'Processing case {case_id}')
        
        # find matching csv
        aif_file = next((f for f in csv_files if case_id in f), None)
        
        if aif_file == None:
            tqdm.write(f'No AIF found for case {case_id}. Skipping') 
            continue


        if aif_file:
            try:

                
                # create series name
                series_name = add_series_name(case_id, pat_series)
                baseline_path = [destpath, case_id, ('Baseline', 0)]

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

                #check + skip if all files are processed
                series = [max_clean, auc_clean, att_clean, fit_clean, rpf_clean, avd_clean, mtt_clean, 
                          fp_clean, tp_clean, vp_clean, ft_clean, tt_clean]
                
                if all(x in db.series(baseline_path) for x in series):
                    tqdm.write(f'Skipping case {case_id}. All maps exist in {site} folder')
                    continue

                #load moco_dicom 
                if site == 'Bari' or 'Bordeaux':              
                    # get array, affine coords from moco dce
                    vol = db.volume(study, dims=['AcquisitionTime'])
                    array = vol.values
                    affine = vol.affine
                    coords = vol.coords


                # load aif curve from csv
                aif_data = pd.read_csv(os.path.join(aif_path, aif_file))
                aif = aif_data.iloc[:, 1].values  
                time = aif_data.iloc[:, 0].values
                dt = np.mean(np.diff(time)) 

                # calculating maps
                tqdm.write('Computing MAX, AUC & ATT...')
                MAX, AUC, ATT, SO = dcmri.pixel_descriptives(array, aif, dt, baseline=15)
                tqdm.write('Computing RPF, AVD & MTT...')
                RPF, AVD, MTT = dcmri.pixel_deconvolve(array, aif, dt, baseline=15, regpar=0.1)
                tqdm.write('Computing Fit & remaining maps...')
                fit, par = dcmri.pixel_2cfm_linfit(array, aif, time, baseline=15)
              
                if max_clean not in db.series(baseline_path):
                    try:
                        tqdm.write(f'Building MAX map for {case_id}')
                        MAX[MAX<0]=0
                        MAX[MAX>10000]=10000
                        db.write_volume((MAX, affine), max_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process MAX map for {case_id}: {e} ')
                       
                if auc_clean not in db.series(baseline_path):
                    try:
                        tqdm.write(f'Building AUC map for {case_id}')
                        AUC[AUC<0]=0
                        AUC[AUC>10000]=10000
                        db.write_volume((AUC, affine), auc_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process AUC map for {case_id}: {e} ')
                
                if att_clean not in db.series(baseline_path):   
                    
                    try:
                        tqdm.write(f'Building ATT map for {case_id}')
                        ATT[ATT<0]=0
                        ATT[ATT>10000]=10000
                        db.write_volume((ATT, affine), att_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process ATT map for {case_id}: {e} ')                         
                
                if fit_clean not in db.series(baseline_path):

                    try:
                        tqdm.write(f'Building FIT for {case_id}')
                        volume = vreg.volume(fit, affine, coords, dims=['AcquisitionTime'])

                        db.write_volume(volume, fit_clean, ref=study, append=True)
                    except Exception as e:
                        logging.error(f'cannot process fit map for {case_id}: {e} ') 
                
                if rpf_clean not in db.series(baseline_path):

                    try:
                        tqdm.write(f'Building RPF map for {case_id}')
                        RPF[RPF<0]=0
                        RPF[RPF>10000]=10000
                        db.write_volume((RPF, affine), rpf_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process RPF map for {case_id}: {e} ') 

                
                if avd_clean not in db.series(baseline_path):

                    try:
                        tqdm.write(f'Building AVD map for {case_id}')
                        AVD[AVD<0]=0
                        AVD[AVD>10000]=10000
                        db.write_volume((AVD, affine), avd_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process AVD map for {case_id}: {e} ') 


                if mtt_clean not in db.series(baseline_path):

                    try:
                        tqdm.write(f'Building MTT map for {case_id}')
                        MTT[MTT<0]=0
                        MTT[MTT>10000]=10000
                        db.write_volume((MTT, affine), mtt_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process mtt map for {case_id}: {e} ') 

                if fp_clean not in db.series(baseline_path):

                    try:
                        tqdm.write(f'Building FP map for {case_id}')
                        par_0 = par[...,0]
                        db.write_volume((par_0, affine), fp_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process fp map for {case_id}: {e}') 

                if tp_clean not in db.series(baseline_path):

                    try:
                        tqdm.write(f'Building TP map for {case_id}')
                        par_1 = par[...,1]
                        db.write_volume((par_1, affine), tp_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process tp map for {case_id}: {e}') 

                if vp_clean not in db.series(baseline_path):

                    try:
                        tqdm.write(f'Building VP map for {case_id}')
                        par_0_1 = par[...,0]*par[...,1]/60

                        db.write_volume((par_0_1, affine), vp_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process vp map for {case_id}: {e}') 

                if ft_clean not in db.series(baseline_path):

                    try:
                        tqdm.write(f'Building FT map for {case_id}')
                        par_2 = par[...,2]
                        db.write_volume((par_2, affine), ft_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process ft map for {case_id}: {e}')  

                if tt_clean not in db.series(baseline_path):

                    try:
                        tqdm.write(f'Building TT map for {case_id}')
                        par_3 = par[...,3]
                        db.write_volume((par_3, affine), tt_clean, ref=study)
                    except Exception as e:
                        logging.error(f'cannot process tt map for {case_id}: {e}')       

            except Exception as e:
                logging.error(f"Cannot map {case_id}: {e}")


 
#Call Task
if __name__ == '__main__':
    Mapping('Bordeaux')