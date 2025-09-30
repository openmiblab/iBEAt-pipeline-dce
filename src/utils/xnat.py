import os
import requests
from requests.auth import HTTPBasicAuth

from tqdm import tqdm
import numpy as np

# import zipfile
# import datetime
# import xnat
# import io
# import pydicom


# def download_series_by_dicom_sequence_name(xnat_url, username, password,
#                                            project_id, subject_id, sequence_name,
#                                            output_dir="xnat_downloads"):
#     """
#     Downloads scan series from XNAT where DICOM SequenceName matches the given value.

#     Args:
#         xnat_url (str): Base XNAT URL.
#         username (str): XNAT login.
#         password (str): XNAT password.
#         project_id (str): Project ID.
#         subject_id (str): Subject ID.
#         sequence_name (str): DICOM SequenceName tag value to match.
#         output_dir (str): Directory to save matched series.
#     """

#     session = requests.Session()
#     session.auth = HTTPBasicAuth(username, password)
#     os.makedirs(output_dir, exist_ok=True)

#     # Get subject experiments
#     exp_url = f"{xnat_url}/data/projects/{project_id}/subjects/{subject_id}/experiments?format=json"
#     r = session.get(exp_url)
#     r.raise_for_status()
#     experiments = r.json()['ResultSet']['Result']

#     for exp in experiments:
#         exp_id = exp['ID']
#         scans_url = f"{xnat_url}/data/experiments/{exp_id}/scans?format=json"
#         r = session.get(scans_url)
#         r.raise_for_status()
#         scans = r.json()['ResultSet']['Result']

#         for scan in scans:
#             scan_id = scan['ID']
#             # Download a sample DICOM file to check SequenceName
#             sample_url = f"{xnat_url}/data/experiments/{exp_id}/scans/{scan_id}/resources/DICOM/files?format=zip"
#             r = session.get(sample_url)
#             if r.status_code != 200:
#                 continue
#             try:
#                 with zipfile.ZipFile(io.BytesIO(r.content)) as zip_file:
#                     # Get the first DICOM file in the archive
#                     dicom_names = [name for name in zip_file.namelist() if name.lower().endswith('.dcm')]
#                     if not dicom_names:
#                         continue
#                     with zip_file.open(dicom_names[0]) as dcm_file:
#                         dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)
#                         seq = getattr(dcm, "SequenceName", None)
#                         if seq == sequence_name:
#                             print(f"Matched scan {scan_id} (SequenceName: {seq})")
#                             out_path = os.path.join(output_dir, f"{subject_id}_{exp_id}_{scan_id}.zip")
#                             with open(out_path, 'wb') as f:
#                                 f.write(r.content)
#                             print(f"Saved scan to {out_path}")
#             except Exception as e:
#                 print(f"Error processing scan {scan_id}: {e}")



def download_scans(
    xnat_url, username, password, output_dir, project_id,
    subject_label=None, experiment_label=None, attr=None, value=None,
):
    """
    Downloads all scan series with a given attribute value.

    Args:
        xnat_url (str): Base URL of the XNAT server.
        username (str): XNAT username.
        password (str): XNAT password.
        output_dir (str): Directory to store downloaded data.
        project_id (str): XNAT project ID.
        subject_label (str): XNAT subject label. If this is None, 
            all subjects are downloaded. Defaults to None.
        experiment_label (str): XNAT experiment label. If this is None, 
            all experiments are downloaded. Defaults to None.
        attr (str ro tuple of str): Attribute(s) to filter by (e.g. 'sequence').
        value: Desired value(s) for the attribute(s).
    """
    if np.isscalar(value):
        value = [value]
    if isinstance(attr, tuple):
        value_list = []
        for v in value:
            if np.isscalar(v):
                value_list.append([v])
            else:
                value_list.append(v)
        value = tuple(value_list)

    session = requests.Session()
    session.auth = HTTPBasicAuth(username, password)

    # Loop over all subjects in the project
    subj_url = f"{xnat_url}/data/projects/{project_id}/subjects?format=json"
    r = session.get(subj_url)
    r.raise_for_status()
    subjects = r.json()['ResultSet']['Result']

    for subj in tqdm(subjects, desc='Scanning subjects..'):
        subject_id = subj['ID']

        # Continue if this subject was not requested
        if subject_label is not None:
            if subj['label'] != subject_label:
                continue

        # Loop over the subject's experiments
        exp_url = f"{xnat_url}/data/projects/{project_id}/subjects/{subject_id}/experiments?format=json"
        r = session.get(exp_url)
        r.raise_for_status()
        experiments = r.json()['ResultSet']['Result']

        for exp in tqdm(experiments, desc='Scanning experiments..'):
            exp_id = exp['ID']

            # Continue if this experiment was not requested
            if experiment_label is not None:
                if exp['label'] != experiment_label:
                    continue

            # Loop over the scans in the experiment
            scans_url = f"{xnat_url}/data/experiments/{exp_id}/scans?format=json"
            r = session.get(scans_url)
            r.raise_for_status()
            scans = r.json()['ResultSet']['Result']

            for scan in scans:
                scan_id = scan['ID']

                # Retrieve scan attributes
                attr_url = f"{xnat_url}/data/experiments/{exp_id}/scans/{scan_id}?format=json"
                r = session.get(attr_url)
                r.raise_for_status()
                scan_attrs = r.json()['items'][0]['data_fields']

                # Continue if the scan does not have the right attributes
                if isinstance(attr, str):
                    if scan_attrs.get(attr) not in value:
                        continue
                if isinstance(attr, tuple):
                    cont = False
                    for i, a in enumerate(attr):
                        if scan_attrs.get(a) not in value[i]:
                            cont=True
                            continue
                    if cont:
                        continue

                # Define download locations
                download_url = f"{xnat_url}/data/experiments/{exp_id}/scans/{scan_id}/resources/DICOM/files?format=zip"
                out_folder = os.path.join(output_dir, project_id, subj['label'], f"{exp['label']}")
                out_path = os.path.join(out_folder, f"series_{scan_id.zfill(2)}.zip")
                
                # Continue if the data have already been downloaded
                if os.path.exists(out_path):
                    continue

                # Download the scan
                os.makedirs(out_folder, exist_ok=True)
                r = session.get(download_url, stream=True)
                with open(out_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)




def _create_user_file():
    # Ask the user for username and password
    username = input("Enter your username: ")
    password = input("Enter your password: ")

    # Create a text file and write the username and password to it
    with open("user_XNAT.txt", "w") as file:
        file.write(f"Username: {username}\n")
        file.write(f"Password: {password}\n")

def _read_user_file():
    # Read the username and password from the text file
    with open("user_XNAT.txt", "r") as file:
        lines = file.readlines()
        username = lines[0].split(":")[1].strip()
        password = lines[1].split(":")[1].strip()

    return username, password

def credentials():
    # Check if the file exists
    if os.path.exists("user_XNAT.txt"):
        # If the file exists, read username and password
        existing_username, existing_password = _read_user_file()
    else:
        # If the file does not exist, create a new file and ask for username and password
        _create_user_file()
        print("User file created successfully.")
        existing_username, existing_password = _read_user_file()
    return existing_username, existing_password
