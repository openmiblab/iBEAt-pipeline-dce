import numpy as np
import napari
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget
import SimpleITK as sitk
import os
import dbdicom as db
import re

# ----------------- Functions -----------------

def load_dicom_series(folder):
    dicom_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            try:
                _ = sitk.ReadImage(path)
                dicom_files.append(path)
            except:
                pass
    if not dicom_files:
        raise RuntimeError(f"No DICOM files found in {folder}")
    
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted(dicom_files))
    image = reader.Execute()
    return sitk.GetArrayFromImage(image)

def load_dynamic(folder):

    if '_3_mip' in folder:
        image = db.series(folder, desc='DCE_3_mip')
        image_vol = db.volume(image[0])
    else:
        image = db.series(folder, desc='DCE_1_aorta')
        image_vol = db.volume(image[0], 'AcquisitionTime')
    image_arr = image_vol.values.T
    return image_arr

def add_series_name(folder, all_series: list):
    new_series_name = "DCE_4_"
    all_series.append(new_series_name)
    return new_series_name

def run_viewer(dicom_folder, mask_folder, output_root):
    folder_name = os.path.basename(os.path.normpath(dicom_folder))
    match = re.search(r'\d{4}_\d{3}', folder_name)
    case_id = match.group(0) if match else folder_name

    img = load_dynamic(dicom_folder) 
    mask = load_dicom_series(mask_folder)

    viewer = napari.Viewer()
    viewer.add_image(img, name=f'{case_id}_image')
    mask_layer = viewer.add_labels(mask, name=f'{case_id}_mask')

    def save_mask():
        img_series = db.series(dicom_folder)
        img_vol = db.volume(img_series[0])
        affine = img_vol.affine

        destpath = os.path.join(output_root)
        os.makedirs(destpath, exist_ok=True)

        pat_series = []
        add_series_name(case_id, pat_series)
        mask_path_db = [destpath, case_id, ('Baseline', 0)]
        dce_mask = mask_path_db + [(pat_series[-1] + "aortaseg", 0)]

        edited_mask = mask_layer.data
        db.write_volume((edited_mask.T, affine), dce_mask)

        print(f"Saved case {case_id} new mask to 4a edited aortaseg folder")

    save_button = QPushButton("Save Mask")
    save_button.clicked.connect(save_mask)

    container = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(save_button)
    container.setLayout(layout)
    viewer.window.add_dock_widget(container, area='right')

    napari.run()  # Wait until closed

# ----------------- Paths -----------------

dicom_root       = r"build/dce_2_data/Bordeaux/Patients"
mask_root        = r"build/dce_4_aortaseg/Bordeaux/Patients/Batch_" #current edits: 1, 2, 6, 8, 9
output_mask_root = r"build/dce_4a_edited_aorta/Bordeaux/Patients"

# ----------------- Build Case Mapping -----------------

def get_case_id(folder_name):
    if folder_name == "dbtree.json":
        return None

    match = re.search(r'\d{4}_\d{3}', folder_name)
    return match.group(0) if match else None

# Create dictionaries mapping case_id -> folder path
dicom_folders_dict = {
    cid: os.path.join(dicom_root, f)
    for f in os.listdir(dicom_root)
    if (cid := get_case_id(f)) is not None
}

mask_folders_dict = {
    cid: os.path.join(mask_root, f)
    for f in os.listdir(mask_root)
    if (cid := get_case_id(f)) is not None
}

# ----------------- Edit Masks Functions -----------------

def edit_masks_all():
    for case_id in dicom_folders_dict:
        dicom_folder = dicom_folders_dict[case_id]
        mask_folder = mask_folders_dict.get(case_id)
        if mask_folder:
            run_viewer(dicom_folder, mask_folder, output_mask_root)
        else:
            print(f"No mask found for case {case_id} or check different batch, skipping.")
            continue

def edit_mask(case_id):
    dicom_folder = dicom_folders_dict.get(case_id)
    mask_folder = mask_folders_dict.get(case_id)
    if dicom_folder and mask_folder:
        run_viewer(dicom_folder, mask_folder, output_mask_root)
    else:
        print(f"Case {case_id} not found or missing mask.")

# ----------------- Main -----------------

if __name__ == '__main__':
    # Edit listed case 
    editing_needed = [  
        '2128_001'
    ]
    
    for case in editing_needed:
        edit_mask(case)
    
    # Or edit all cases sequentially
    #edit_masks_all()
