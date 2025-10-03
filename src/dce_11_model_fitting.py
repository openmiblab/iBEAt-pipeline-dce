import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydmr
import os 
import dcmri as dc


# files = [
# 'build/dce_10_roi_analysis/Bari/Patients/DMR/1128_003/1128_003_aif.dmr.zip',
# 'build/dce_10_roi_analysis/Bari/Patients/DMR/1128_003/1128_003_kidney_left_if.dmr.zip',
# 'build/dce_10_roi_analysis/Bari/Patients/DMR/1128_003/1128_003_kidney_right_if.dmr.zip']


datafile = 'build/dce_10_roi_analysis/Bari/Patients/DMR_combined/1128_003/1128_003.dmr.zip'  




# datafile = dc.fetch(bari_test)
dmr = pydmr.read(datafile, 'nest')
rois, pars = dmr['rois'], dmr['pars']

def kidney_model(roi, par, kidney):
    
    # Get B0 and precontrast T1
    B0 = par['field_strength']
    T1 = par[kidney+' T1'] if kidney+' T1' in par else dc.T1(B0, 'kidney')

    # Define tissue model
    model = dc.Kidney(

        # Configuration
        aif = roi['aorta']['signal'],
        t = roi['aorta']['time'],

        # General parameters
        field_strength = B0,
        agent = par['agent'],
        t0 = roi['aorta']['time'][par['n0']],

        # Sequence parameters
        TR = par['TR'],
        FA = par['FA'],

        # Tissue parameters
        vol = par[kidney+' vol'],
        R10 = 1/T1,
        R10a = 1/dc.T1(B0, 'blood'),
    )

    # Customize free parameter ranges
    model.set_free(
        pop = 'Ta',
        Tt = [30, np.inf],
    )

    # Train the kidney model on the data
    xdata = roi['kidney_right']['time']
    ydata = roi['kidney_right']['signal']
    model.train(xdata, ydata)

    return xdata, ydata, model

time, signal, model = kidney_model(
    rois['1128_003'],
    pars['1128_003']['kidney_right'],
    'kidney_right',
)

model.plot(time, signal)

model.print_params(round_to=3)


