import os

from utils import xnat

path = os.path.join(os.getcwd(), 'build', 'dce_1_download')  
os.makedirs(path, exist_ok=True)


def leeds_patients():
    username, password = xnat.credentials()
    xnat.download_scans(
        xnat_url="https://qib.shef.ac.uk",
        username=username,
        password=password,
        output_dir=path,
        project_id="BEAt-DKD-WP4-Leeds",
        subject_label="Leeds_Patients",
        attr="parameters/sequence",
        value="*tfl2d1_16",
    )

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

def sheffield_patients():
    username, password = xnat.credentials()
    xnat.download_scans(
        xnat_url="https://qib.shef.ac.uk",
        username=username,
        password=password,
        output_dir=path,
        project_id="BEAt-DKD-WP4-Sheffield",
        attr="series_description",
        value=[
            # Philips data
            #"DCE_kidneys_cor-oblique_fb", 
            # GE data
            '3D_DISCO_Dyn_kidneys_cor-oblique_fb',
        ],
    )


if __name__=='__main__':
    #leeds_patients()
    #bari_patients()
    sheffield_patients()