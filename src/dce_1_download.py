import os

from utils import xnat

path = os.path.join(os.getcwd(), 'build', 'dce_1_download')  
os.makedirs(path, exist_ok=True)


def bari_patients():
    username, password = xnat.credentials()
    xnat.download_scans(
        xnat_url="https://qib.shef.ac.uk",
        username=username,
        password=password,
        output_dir=path,
        project_id="BEAt-DKD-WP4-Bari",
        subject_label="Bari_Patients",
        attr="series_description",
        value="DCE_kidneys_cor-oblique_fb_wet_pulse",
    )



if __name__=='__main__':
    bari_patients()
