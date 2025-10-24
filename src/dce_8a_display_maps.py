import os
import dbdicom as db
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import vreg.plot as vplot
 

# Main: Load and Display maps 
def map_display(site, map_type):
    
    #data directories
    datapath = os.path.join(os.getcwd(), 'build', 'dce_8_mapping')
    base_dir =  os.path.join(datapath, site, "Patients")
    
    
    #load database 
    database = db.series(base_dir)


    # Logging setup
    logging.basicConfig(
    filename=os.path.join(base_dir, 'error.log'),
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )


    # Directory to save figures
    save_dir = os.path.join(os.getcwd(), 'build', 'dce_8_mapping', site, 'Patients', f'_{map_type}')
    os.makedirs(save_dir, exist_ok=True)
        
    #load maps 
    auc_map_database = [entry for entry in database if entry[3][0].strip().lower() == 'dce_8_auc_map'.lower()]
    rpf_map_database = [entry for entry in database if entry[3][0].strip().lower() == 'dce_8_rpf_map'.lower()]
    avd_map_database = [entry for entry in database if entry[3][0].strip().lower() == 'dce_8_avd_map'.lower()]
    mtt_map_database = [entry for entry in database if entry[3][0].strip().lower() == 'dce_8_mtt_map'.lower()]


    if map_type == 'AUC':
        for map in auc_map_database:
            case_id = map[1]
            auc_vol = db.volume(map)
            vplot.overlay_2d_new(auc_vol, save_path=save_dir + f'/{case_id}.png', show=False)


    elif map_type == 'RPF':
        for map in rpf_map_database:
            case_id = map[1]
            rpf_vol = db.volume(map)
            vplot.overlay_2d_new(rpf_vol, vmin=1, vmax=200, save_path=save_dir + f'/{case_id}.png', show=False)


    elif map_type == 'AVD':
        for map in avd_map_database:
            case_id = map[1]
            avd_vol = db.volume(map)
            vplot.overlay_2d_new(avd_vol, save_path=save_dir + f'/{case_id}.png', show=False)
    

    elif map_type == 'MTT':
        for map in mtt_map_database:
            case_id = map[1]
            mtt_vol = db.volume(map)
            vplot.overlay_2d_new(mtt_vol, vmin=1, vmax=200, save_path=save_dir + f'/{case_id}.png', show=False)


# Call Task 
if __name__ == '__main__':
    for map_type in ['RPF', 'MTT']:
        map_display('Sheffield', map_type=map_type)


