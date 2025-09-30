import numpy as np
import os
import logging
import dbdicom as db
import napari

# List of patients to exclude
EXCLUDE = [
    4128_054,
    #sheffield Philips TO DO
    7128_002,
    7128_005,
    7128_006,
    7128_007,
    7128_008,
    7128_009,
    7128_010,
    7128_011,
    7128_012,
    7128_014,
    7128_015,
    7128_016,
    7128_017

]

def leeds_add_series_name(folder, all_series: list):
    new_series_name = "DCE_3_"
    all_series.append(new_series_name)
    return new_series_name


def bari_add_series_name(folder, all_series: list):
    new_series_name = "DCE_3_"
    all_series.append(new_series_name)
    return new_series_name

def sheffield_add_series_name(folder, all_series: list):
    new_series_name = "DCE_3_"
    all_series.append(new_series_name)
    return new_series_name

def max_enh(array_4d, n0=10):
    print(array_4d.shape)
    #array_4d = np.transpose(array_4d, (1,0,2,3))
    S0 = np.mean(array_4d[...,:n0], axis=-1)
    Smax = np.max(array_4d, axis=-1)
    array_3d = Smax - S0
    # viewer = napari.Viewer()
    # viewer.add_image(array_3d)
    # napari.run()
    return array_3d # 3D array


def Leeds():
    datapath = os.path.join(os.getcwd(), 'build', 'dce_2_data')
    sitedatapath = os.path.join(datapath, "Leeds", "Patients")
    dstdatapath = os.path.join(os.getcwd(), 'build', 'dce_3_mip')
    destpath =  os.path.join(dstdatapath, "Leeds", "Patients")
    os.makedirs(destpath, exist_ok=True)

    # Logging setup
    logging.basicConfig(
        filename=os.path.join(destpath, 'error.log'),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    database = db.series(sitedatapath)
    DCE_axial_aorta = [entry for entry in database if entry[3][0].strip().lower() == 'dce_1_axial_aorta']
    pat_series = []
    for study in DCE_axial_aorta:
        try:
            #create folder
            leeds_add_series_name(study[1], pat_series)
            baseline_path = [destpath, study[1], ('Baseline', 0)]
            mip_clean = baseline_path + [(pat_series[-1] + "mip", 0)]
            # Skip if already processed
            if mip_clean in db.series(baseline_path):
                continue
            #build mip volume
            array_4d = db.volume(study, dims=['AcquisitionTime'])
            mip_series = max_enh(array_4d.values)
            db.write_volume((mip_series, array_4d.affine), mip_clean, ref=study)
        except Exception as e:
            logging.error(f"Study {study[1]} cannot be assesed: {e}")

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
    DCE_axial_aorta = [entry for entry in database if entry[3][0].strip().lower() == 'dce_1_axial_aorta']
    pat_series = []
    for study in DCE_axial_aorta:
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

def sheffield_patients():
    datapath = os.path.join(os.getcwd(), 'build', 'dce_2_data')
    sitedatapath = os.path.join(datapath, "Sheffield", "Patients")
    dstdatapath = os.path.join(os.getcwd(), 'build', 'dce_3_mip')
    destpath =  os.path.join(dstdatapath, "Sheffield", "Patients")
    os.makedirs(destpath, exist_ok=True)

    # Logging setup
    logging.basicConfig(
        filename=os.path.join(destpath, 'error.log'),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    database = db.series(sitedatapath)


    
    DCE_coronal_kidneys = [entry for entry in database if entry[3][0].strip().lower() == 'dce_1_coronal_kidneys']
    pat_series = []
    for study in DCE_coronal_kidneys:
        
        if study[1] in EXCLUDE:
            continue
        
        try:
            #create folder
            sheffield_add_series_name(study[1], pat_series)
            baseline_path = [destpath, study[1], ('Baseline', 0)]
            mip_clean = baseline_path + [(pat_series[-1] + "mip", 0)] 
            #build mip volume
            array_4d = db.volume(study, dims=['TemporalPositionIdentifier'])
            viewer = napari.Viewer()
            viewer.add_image(array_4d.values)
            napari.run()
            mip_series = max_enh(array_4d.values)
            db.write_volume((mip_series, array_4d.affine), mip_clean, ref=study)
        except Exception as e:
            logging.error(f"Study {study[1]} cannot be assesed: {e}")

if __name__ == '__main__':
    # Bari()
    # Leeds()
    sheffield_patients()



