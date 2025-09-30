import os
import re

import time
from tqdm import tqdm
from osfclient.api import OSF
import traceback


def token():
    # Read the token from the file
    with open("user_OSF.txt", "r") as file:
        lines = file.readlines()
        token = lines[0].strip()
    return token

def count_files(folder_path):
    """Count all files in folder and subfolders."""
    total_files = 0
    for _, _, files in os.walk(folder_path):
        total_files += len(files)
    return total_files

def sanitize_folder_name(name):
    """Sanitize folder names to remove unsafe characters."""
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)



# Upload with osf client


def _upload_folder(osf_token, project_id, local_folder_path, osf_target_folder='', overwrite=False):
    """
    Uploads a folder (including subfolders) to an OSF project or component, resuming by comparing file sizes.

    Parameters:
        osf_token (str): OSF personal access token.
        project_id (str): OSF project ID.
        local_folder_path (str): Local folder to upload.
        osf_target_folder (str): Target folder in OSF storage (optional).
        overwrite (bool): Whether to overwrite existing files. Default is False (skip).
    """
    osf = OSF(token=osf_token)
    project = osf.project(project_id)
    storage = project.storage('osfstorage')

    # Build a dictionary of existing OSF files {path: OSF file object}
    existing_files = {f.path: f for f in storage.files}

    nfiles = count_files(local_folder_path)
    with tqdm(total=nfiles, desc="Copying files") as pbar:
        for root, _, files in os.walk(local_folder_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_folder_path)
                osf_file_path = os.path.join(osf_target_folder, relative_path).replace('\\', '/')

                # Sanitize file path
                parts = osf_file_path.strip('/').split('/')
                sanitized_parts = [sanitize_folder_name(p) for p in parts]
                osf_file_path = '/'.join(sanitized_parts)

                local_file_size = os.path.getsize(local_file_path)
                osf_file_obj = existing_files.get(f'/{osf_file_path}')

                # If the file exists, sizes match and overwrite is False, skip
                # In all other scenarios, delete it first
                if osf_file_obj:
                    if osf_file_obj.size == local_file_size:
                        if not overwrite:
                            pbar.update(1)
                            continue  # Skip upload, already exists with same size
                    try:
                        osf_file_obj.delete()
                    except Exception as e:
                        raise RuntimeError(f"Cannot overwrite {osf_file_path} - failed to delete: {e}")

                # Upload the file
                try:
                    with open(local_file_path, 'rb') as fp:
                        storage.create_file(osf_file_path, fp)
                except Exception as e:
                    print(f"Failed to upload {osf_file_path}: {e}")
                    raise e

                pbar.update(1)


def upload_folder(osf_token, project_id, local_folder_path, osf_target_folder='', overwrite=False):
    """
    Uploads a folder (including subfolders) to an OSF project or component, resuming by comparing file sizes.

    Parameters:
        osf_token (str): OSF personal access token.
        project_id (str): OSF project ID.
        local_folder_path (str): Local folder to upload.
        osf_target_folder (str): Target folder in OSF storage (optional).
        overwrite (bool): Whether to overwrite existing files. Default is False (skip).
    """

    # Allow 10 failed attempts in 10 minutes
    max_attempts = 10
    reset_after_mins = 10

    # Start the clock
    attempts = 1
    start_time = time.time()

    while True:
        if attempts > max_attempts:
            time_elapsed = (time.time()-start_time)/60
            raise RuntimeError(f"❌ Upload stopped after {attempts} interruptions in {time_elapsed} minutes.")
        try:
            _upload_folder(osf_token, project_id, local_folder_path, osf_target_folder, overwrite)
            break
        except Exception as e:
            time_elapsed = (time.time()-start_time)/60
            print(f'Uploaded interrupted after {attempts} attempts in {time_elapsed} minutes. \n' 
                  f'Detailed error message: {traceback.format_exc()}')
            if time_elapsed >= reset_after_mins:
                attempts = 1
                start_time = time.time()
            else:
                attempts +=1

    print("✅ Upload complete!")






# def upload_folder(osf_token, project_id, local_folder_path, osf_target_folder='', overwrite=False):
#     """
#     Uploads a folder (including subfolders) to an OSF project or component, with optional overwrite.

#     Parameters:
#         osf_token (str): OSF personal access token.
#         project_id (str): OSF project ID.
#         local_folder_path (str): Local folder to upload.
#         osf_target_folder (str): Target folder in OSF storage (optional).
#         overwrite (bool): Whether to overwrite existing files. Default is False (skip).
#     """
#     osf = OSF(token=osf_token)
#     project = osf.project(project_id)
#     storage = project.storage('osfstorage')

#     # Build dictionary of existing files for quick lookup
#     existing_files = {f.path: f for f in storage.files}

#     # Count files for the progress bar
#     nfiles = count_files(local_folder_path)
#     with tqdm(total=nfiles, desc="Copying files") as pbar:
#         for root, _, files in os.walk(local_folder_path):
#             for file in files:
#                 local_file_path = os.path.join(root, file)
#                 relative_path = os.path.relpath(local_file_path, local_folder_path)
#                 osf_file_path = os.path.join(osf_target_folder, relative_path).replace('\\', '/')

#                 # Sanitize file path
#                 parts = osf_file_path.strip('/').split('/')
#                 sanitized_parts = [sanitize_folder_name(p) for p in parts]
#                 osf_file_path = '/'.join(sanitized_parts)

#                 if osf_file_path in [f[1:] for f in existing_files]:
#                     if overwrite:
#                         try:
#                             existing_files[osf_file_path].delete()
#                         except Exception as e:
#                             RuntimeError(f"❌ Cannot overwrite {osf_file_path} - failed to delete: {e}")
#                     else:
#                         continue

#                 with open(local_file_path, 'rb') as fp:
#                     storage.create_file(osf_file_path, fp)

#                 pbar.update(1)  # Update progress bar

#     print("✅ Upload complete!")




