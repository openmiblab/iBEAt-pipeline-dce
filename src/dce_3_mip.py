import numpy as np
import os
import logging
import dbdicom as db
from tqdm import tqdm

# List of patients to exclude
EXCLUDE = []

#Helper: Add Series Name
def bari_add_series_name(folder, all_series: list):
    new_series_name = "DCE_3_"
    all_series.append(new_series_name)
    return new_series_name


def max_enh(array_4d, n0=15):
    print(array_4d.shape)
    S0 = np.mean(array_4d[...,:n0], axis=-1)
    Smax = np.max(array_4d, axis=-1)
    array_3d = Smax - S0
    return array_3d 

#Generate Maximum Intensity Projection (MIP)
def Bari():
    datapath = os.path.join(os.getcwd(), 'build', 'dce_2_data')
    sitedatapath = os.path.join(datapath, "Bari", "Patients")
    dstdatapath = os.path.join(os.getcwd(), 'build', 'dce_3_mip')
    destpath =  os.path.join(dstdatapath, "Bari", "Patients")
    os.makedirs(destpath, exist_ok=True)

    # Logging setup
    logging.basicConfig(
        filename=os.path.join(destpath, 'error.log'),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    database = db.series(sitedatapath)
    DCE_axial_aorta = [entry for entry in database if entry[3][0].strip().lower() == 'dce_1_aorta'.lower()]
    pat_series = []
    for study in tqdm(DCE_axial_aorta, desc='Proceesing MIP', unit='case'):
        try:
            #create folder
            series_name = bari_add_series_name(study[1], pat_series)
            baseline_path = [destpath, study[1], ('Baseline', 0)]
            mip_clean = baseline_path + [(series_name + "mip", 0)] 
            
            # Skip if already processed
            if mip_clean in db.series(baseline_path):
                continue

            #build mip volume
            array_4d = db.volume(study, dims=['AcquisitionTime'])
            mip_series = max_enh(array_4d.values)
            db.write_volume((mip_series, array_4d.affine), mip_clean, ref=study)
        except Exception as e:
            logging.error(f"Study {study[1]} cannot be assesed: {e}")

# Call Task Site 
if __name__ == '__main__':
    Bari()




