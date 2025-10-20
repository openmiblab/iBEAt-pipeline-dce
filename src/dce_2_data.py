"""
Create clean database
"""
import os
import zipfile
import shutil
import logging
import tempfile
import numpy as np
import pydicom
from tqdm import tqdm
import dbdicom as db

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

# Paths
downloadpath = os.path.join(os.getcwd(), 'build', 'dce_1_download')
datapath = os.path.join(os.getcwd(), 'build', 'dce_2_data')
os.makedirs(datapath, exist_ok=True)

# Logging setup
logging.basicConfig(
    filename=os.path.join(datapath, 'error.log'),
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Helper: Flatten directory after extraction
def flatten_folder(root_folder):
    for dirpath, _, filenames in os.walk(root_folder, topdown=False):
        for filename in filenames:
            src_path = os.path.join(dirpath, filename)
            dst_path = os.path.join(root_folder, filename)

            if os.path.exists(dst_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(root_folder, f"{base}_{counter}{ext}")
                    counter += 1

            shutil.move(src_path, dst_path)

        if dirpath != root_folder:
            try:
                os.rmdir(dirpath)
            except OSError:
                print(f"Could not remove {dirpath} — not empty or in use.")

# Helper: Standardize patient ID
def bari_ibeat_patient_id(folder):
    if folder[:3] == 'iBE':
        return folder[4:].replace('-', '_')
    else:
        return folder[:4] + '_' + folder[4:]

# Helper: Standardize patient ID
def bordeaux_ibeat_patient_id(folder):
    if folder[:3] == 'iBE':
        return folder[4:].replace('-', '_')
    else:
        return folder[:4] + '_' + folder[4:]

# Helper: Standardize Series Name
def add_series_name(folder, all_series: list):
    new_series_name = "DCE_1_"
    all_series.append(new_series_name)
    return new_series_name
 

def sheffield_ibeat_patient_id(folder):
    id = folder[3:]
    id = id[:4] + '_' + id[4:]
    if id == '2178_157': # Data entry error
        id = '7128_157'
    return id


# Data processing
def bari_patients():

    # Define input and output folders
    sitedownloadpath = os.path.join(downloadpath, "BEAt-DKD-WP4-Bari", "Bari_Patients")
    sitedatapath = os.path.join(datapath, "Bari", "Patients")
    os.makedirs(sitedatapath, exist_ok=True)

    # Loop over all patients
    patients = [f.path for f in os.scandir(sitedownloadpath) if f.is_dir()]
    for pat in tqdm(patients, desc='Building clean database'):

        # Get a standardized ID from the folder name
        pat_id = bari_ibeat_patient_id(os.path.basename(pat))

        # Corrupted data
        if pat_id in EXCLUDE:
            continue

        # Find all zip series, remove those with 'OT' in the name and sort by series number
        all_zip_series = [f for f in os.listdir(pat) if os.path.isfile(os.path.join(pat, f))]
        all_zip_series = [s for s in all_zip_series if 'OT' not in s]
        all_zip_series = sorted(all_zip_series, key=lambda x: int(x[7:-4]))

        # loop over all series
        pat_series = []
        for zip_series in all_zip_series:

            # Get the name of the zip file without extension
            zip_name = zip_series[:-4]

            # Get the harmonized series name 
            try:
                add_series_name(zip_name, pat_series)
            except Exception as e:
                logging.error(f"Patient {pat_id} - error renaming {zip_name}: {e}")
                continue

            # Construct output series
            study = [sitedatapath, pat_id, ('Baseline', 0)]    
            kidney_clean = study + [(pat_series[-1] + 'kidneys', 0)]
            aorta_clean = study + [(pat_series[-1] + 'aorta', 0)]

            # If the series already exists, continue to the next
            if kidney_clean in db.series(study):
                continue

            with tempfile.TemporaryDirectory() as temp_folder:

                # Extract to a temporary folder and flatten it
                os.makedirs(temp_folder, exist_ok=True)
                try:
                    extract_to = os.path.join(temp_folder, zip_name)
                    with zipfile.ZipFile(os.path.join(pat, zip_series), 'r') as zip_ref:
                        zip_ref.extractall(extract_to)
                except Exception as e:
                    logging.error(f"Patient {pat_id} - error extracting {zip_name}: {e}")
                    continue
                flatten_folder(extract_to)

                # read DICOM 
                dce = db.series(extract_to)[0]
                if 'Philips Medical Systems' in str(db.unique('Manufacturer', dce)):
            
                    try:
                        dce_split = db.split_series(dce, 'ImageOrientationPatient') #split DICOM
                    except Exception as e:
                        logging.error(
                            f"Error splitting Bari series {pat_id} "
                            f"{os.path.basename(extract_to)}."
                            f"The series is not included in the database.\n"
                            f"--> Details of the error: {e}")
                        continue
                
                    #define each split 
                    if len(dce_split) >= 2:
                        def compute_normal(orientation):
                            if not isinstance(orientation, (list, tuple)) or len(orientation) != 6:
                                logging.error(f"Bari series {pat_id}: Expected list of 6 values, got: {orientation}")
    
                            row = np.array(orientation[:3])
                            col = np.array(orientation[3:])
                            normal = np.cross(row, col)
                            return normal / np.linalg.norm(normal)

                        # Orientation axis labels
                        axes = ['X (Sagittal)', 'Y (Coronal)', 'Z (Axial)']

                        # Loop through your dce_split list
                        for i, (orientation, _) in enumerate(dce_split):
                            normal = compute_normal(orientation)
                            dominant_axis = np.argmax(np.abs(normal))
                            print(f"DCE Series {i}:")
                            print(f"  Orientation: {orientation}")
                            print(f"  Normal Vector: {normal}")
                            print(f"  Plane: {axes[dominant_axis]}\n")
                            plane = axes[dominant_axis]
                            if plane == 'Y (Coronal)':
                                dce_coronal = dce_split[i][1]
                            elif plane == 'Z (Axial)':
                                dce_axial = dce_split[i][1]
                    else:
                        print("Only one orientation found. Cannot select second series.")
        
                    # Write to separate DICOM folders
                    try:
                        kidney_vols = db.volumes_2d(dce_coronal, 'AcquisitionTime')
                        for vol in kidney_vols:
                            db.write_volume(vol, kidney_clean, ref=dce_coronal, append=True)
                        
                        aorta_vols = db.volumes_2d(dce_axial, 'AcquisitionTime')
                        for vol in aorta_vols:
                            db.write_volume(vol, aorta_clean, ref=dce_axial, append=True)
                    except Exception as e:
                            logging.error(f"Patient {pat_id} - {pat_series[-1]}: {e}")


# Data processing
def bordeaux_patients():

    # Define input and output folders
    sitedownloadpath = os.path.join(downloadpath, "BEAt-DKD-WP4-Bordeaux", "Bordeaux_Patients_Baseline")
    sitedatapath = os.path.join(datapath, "Bordeaux", "Patients")
    os.makedirs(sitedatapath, exist_ok=True)

    # Loop over all patients
    patients = [f.path for f in os.scandir(sitedownloadpath) if f.is_dir()]
    for pat in tqdm(patients, desc='Building clean database'):

        # Get a standardized ID from the folder name
        pat_id = bordeaux_ibeat_patient_id(os.path.basename(pat))

        # Corrupted data
        if pat_id in EXCLUDE:
            continue

        # Find all zip series, remove those with 'OT' in the name and sort by series number
        all_zip_series = [f for f in os.listdir(pat) if os.path.isfile(os.path.join(pat, f))]
        all_zip_series = [s for s in all_zip_series if 'OT' not in s]
        all_zip_series = sorted(all_zip_series, key=lambda x: int(x[7:-4]))

        # loop over all series
        pat_series = []
        for zip_series in all_zip_series:

            # Get the name of the zip file without extension
            zip_name = zip_series[:-4]

            # Get the harmonized series name 
            try:
                add_series_name(zip_name, pat_series)
            except Exception as e:
                logging.error(f"Patient {pat_id} - error renaming {zip_name}: {e}")
                continue

            # Construct output series
            study = [sitedatapath, pat_id, ('Baseline', 0)]    
            kidney_clean = study + [(pat_series[-1] + 'kidneys', 0)]
            aorta_clean = study + [(pat_series[-1] + 'aorta', 0)]

            # If the series already exists, continue to the next
            if kidney_clean in db.series(study):
                continue

            with tempfile.TemporaryDirectory() as temp_folder:

                # Extract to a temporary folder and flatten it
                os.makedirs(temp_folder, exist_ok=True)
                try:
                    extract_to = os.path.join(temp_folder, zip_name)
                    with zipfile.ZipFile(os.path.join(pat, zip_series), 'r') as zip_ref:
                        zip_ref.extractall(extract_to)
                except Exception as e:
                    logging.error(f"Patient {pat_id} - error extracting {zip_name}: {e}")
                    continue
                flatten_folder(extract_to)

                # read DICOM 
                dce = db.series(extract_to)[0]
                if 'SIEMENS' in str(db.unique('Manufacturer', dce)):
            
                    try:
                        dce_split = db.split_series(dce, 'ImageOrientationPatient') #split DICOM
                    except Exception as e:
                        logging.error(
                            f"Error splitting Bari series {pat_id} "
                            f"{os.path.basename(extract_to)}."
                            f"The series is not included in the database.\n"
                            f"--> Details of the error: {e}")
                        continue
                
                    #define each split 
                    if len(dce_split) >= 2:
                        def compute_normal(orientation):
                            if not isinstance(orientation, (list, tuple)) or len(orientation) != 6:
                                logging.error(f"Bari series {pat_id}: Expected list of 6 values, got: {orientation}")
    
                            row = np.array(orientation[:3])
                            col = np.array(orientation[3:])
                            normal = np.cross(row, col)
                            return normal / np.linalg.norm(normal)

                        # Orientation axis labels
                        axes = ['X (Sagittal)', 'Y (Coronal)', 'Z (Axial)']

                        # Loop through your dce_split list
                        for i, (orientation, _) in enumerate(dce_split):
                            normal = compute_normal(orientation)
                            dominant_axis = np.argmax(np.abs(normal))
                            print(f"DCE Series {i}:")
                            print(f"  Orientation: {orientation}")
                            print(f"  Normal Vector: {normal}")
                            print(f"  Plane: {axes[dominant_axis]}\n")
                            plane = axes[dominant_axis]
                            if plane == 'Y (Coronal)':
                                dce_coronal = dce_split[i][1]
                            elif plane == 'Z (Axial)':
                                dce_axial = dce_split[i][1]
                    else:
                        print("Only one orientation found. Cannot select second series.")
        
                    # Write to separate DICOM folders
                    try:
                        kidney_vols = db.volumes_2d(dce_coronal, 'AcquisitionTime')
                        for vol in kidney_vols:
                            db.write_volume(vol, kidney_clean, ref=dce_coronal, append=True)
                        
                        aorta_vols = db.volumes_2d(dce_axial, 'AcquisitionTime')
                        for vol in aorta_vols:
                            db.write_volume(vol, aorta_clean, ref=dce_axial, append=True)
                    except Exception as e:
                            logging.error(f"Patient {pat_id} - {pat_series[-1]}: {e}")

def sheffield_patients():

    # Sheffield site paths
    sitedownloadpath = os.path.join(downloadpath, "BEAt-DKD-WP4-Sheffield")
    sitedatapath = os.path.join(datapath, "Sheffield", "Patients") 
    os.makedirs(sitedatapath, exist_ok=True)

    # Loop over all patients
    patients = [f.path for f in os.scandir(sitedownloadpath) if f.is_dir()]
    for pat in tqdm(patients, desc='Building clean database'):

        # Get standardized ID
        pat_id = sheffield_ibeat_patient_id(os.path.basename(pat))

        # Skip excluded
        if pat_id in EXCLUDE:
            continue

        # Find all zips recursively
        all_zip_series = [
            os.path.join(root, file)
            for root, _, files in os.walk(pat)
            for file in files
            if file.lower().endswith('.zip') and 'OT' not in file
        ]

        # Build series naming
        pat_series = []
        add_series_name(os.path.basename(pat), pat_series)

        # Construct output study paths
        study = [sitedatapath, pat_id, ('Baseline', 0)]
        dce_clean = study + [(pat_series[-1] + 'kidneys', 0)]
        aorta_clean = study + [(pat_series[-1] + 'aorta', 0)]

        # Skip if already processed
        if dce_clean in db.series(study):
            continue

        # Extract all zips for this patient to one temp folder
        with tempfile.TemporaryDirectory() as temp_folder:
            for zip_series in all_zip_series:
                try:
                    with zipfile.ZipFile(zip_series, 'r') as zip_ref:
                        zip_ref.extractall(temp_folder)
                except Exception as e:
                    logging.error(f"Patient {pat_id} - error extracting {zip_series}: {e}")
                    continue
            flatten_folder(temp_folder)

            # Read combined series
            try:
                dce = db.series(temp_folder)[0]
            except Exception as e:
                logging.error(f"Patient {pat_id} - error reading series: {e}")
                continue

            manufacturer = str(db.unique('Manufacturer', dce))

            # --- GE CASE ---
            if 'GE MEDICAL SYSTEMS' in manufacturer:
                try:
                    dce_split = db.split_series(dce, 'ImageOrientationPatient')
                except Exception as e:
                    logging.error(f"Error splitting Sheffield GE series {pat_id}: {e}")
                    continue

                if len(dce_split) >= 1:
                    def compute_normal(orientation):
                        if not isinstance(orientation, (list, tuple)) or len(orientation) != 6:
                            logging.error(
                                f"Sheffield GE {pat_id}: bad orientation {orientation}"
                            )
                        row = np.array(orientation[:3])
                        col = np.array(orientation[3:])
                        normal = np.cross(row, col)
                        return normal / np.linalg.norm(normal)

                    axes = ['X (Sagittal)', 'Y (Coronal)', 'Z (Axial)']

                    # Debug prints for orientation
                    for i, (orientation, _) in enumerate(dce_split):
                        normal = compute_normal(orientation)
                        dominant_axis = np.argmax(np.abs(normal))
                        plane = axes[dominant_axis]
                        print(f"Patient {pat_id} - DCE Series {i}:")
                        print(f"  Orientation: {orientation}")
                        print(f"  Normal Vector: {normal}")
                        print(f"  Plane: {plane}\n")

                        if plane == 'Y (Coronal)':
                            dce_coronal = dce_split[i][1]
                        elif plane == 'Z (Axial)':
                            dce_axial = dce_split[i][1]

                    # Process coronal
                    if plane == 'Y (Coronal)':
                        dce_temporal = db.split_series(dce_coronal, 'TemporalPositionIdentifier')
                        sorted_acquisition = sorted(dce_temporal, key=lambda x: x[0])
                        for _, series_path in sorted_acquisition:
                            try:
                                dce_vol = db.volume(series_path)
                            except Exception as e:
                                logging.error(f"Patient {pat_id} coronal error: {e}")
                            else:
                                db.write_volume(dce_vol, dce_clean, ref=series_path, append=True)

                    # Process axial
                    if plane == 'Z (Axial)':
                        dce_temporal = db.split_series(dce_axial, 'TemporalPositionIdentifier')
                        sorted_acquisition = sorted(dce_temporal, key=lambda x: x[0])
                        for _, series_path in sorted_acquisition:
                            try:
                                dce_vol = db.volume(series_path, dim='Time')
                            except Exception as e:
                                logging.error(f"Patient {pat_id} axial error: {e}")
                            else:
                                db.write_volume(dce_vol, aorta_clean, ref=series_path, append=True)


# Call Task
if __name__ == '__main__':
    #bordeaux_patients()
    sheffield_patients()

