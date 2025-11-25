import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydmr
import dcmri as dc
from openpyxl import Workbook
from tqdm import tqdm

# -----------------------------
# DMR Merge Function
# -----------------------------
def combine_dmr(site, case_id, batch_no=None):
    base_dir = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients")

    # AORTA
    if batch_no is not None:
        aorta_dmr = os.path.join(base_dir, f"Batch_{batch_no}", 'DMR', f'{case_id}_aif.dmr.zip')
    else:
        aorta_dmr = os.path.join(base_dir, 'DMR', f'{case_id}_aif.dmr.zip')

    # RK
    if batch_no is not None:
        rk_dmr = os.path.join(base_dir, f"Batch_{batch_no}", 'DMR', f'{case_id}_kidney_right.dmr.zip')
    else:
        rk_dmr = os.path.join(base_dir, 'DMR', f'{case_id}_kidney_right.dmr.zip')

    # LK
    if batch_no is not None:
        lk_dmr = os.path.join(base_dir, f"Batch_{batch_no}", 'DMR', f'{case_id}_kidney_left.dmr.zip')
    else:
        lk_dmr = os.path.join(base_dir, 'DMR', f'{case_id}_kidney_left.dmr.zip')

    # Output DMR
    if batch_no is not None:
        output_dir = os.path.join(base_dir, f"Batch_{batch_no}", 'DMR')
    else:
        output_dir = os.path.join(base_dir, 'DMR')
    os.makedirs(output_dir, exist_ok=True)

    out_dmr = os.path.join(output_dir, f"{case_id}.dmr.zip")

    # Create empty file if it doesn't exist
    open(out_dmr, "wb").close()

    files_to_merge = [aorta_dmr, rk_dmr, lk_dmr]

    pydmr.concat(files_to_merge, out_dmr, cleanup=False)
    print(f'Merged {files_to_merge} → {out_dmr}')


# -----------------------------
# Kidney Model Function
# -----------------------------
def kidney_model(aorta, roi, par, kidney):
    B0 = par['field_strength']
    T1 = par.get(f'{kidney} T1', dc.T1(B0, 'kidney'))

    # Create kidney model
    model = dc.Kidney(
        aif=aorta['signal'],
        t=aorta['time'],
        field_strength=B0,
        agent=par['agent'],
        t0=roi['time'][par['n0']],
        TR=par['TR'],
        FA=par['FA'],
        vol=par['vol'],
        R10=1/T1,
        R10a=1/dc.T1(B0, 'blood'),
    )

    # Fit model
    xdata = roi['time']
    ydata = roi['signal']
    try:
        model.train(xdata, ydata)
    except Exception as e:
        print(f"{e}. Skipping")

    return xdata, ydata, model


# -----------------------------
# Excel Export Function
# -----------------------------
from openpyxl import Workbook
import os

def save_models(all_models, all_model_names, all_case_ids, outfile):
    """
    Save multiple cases and ROIs into one Excel file, one big sheet.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "All_Models"
    # Header row
    ws.append(["Case ID", "ROI", "Parameter", "Description", "Value", "Error", "Units", "Type"])

    free_params = ["Fp", "Tp", "Ft", "Tt"]  # modify as needed

    for case_id, model_list, name_list in zip(all_case_ids, all_models, all_model_names):
        for model, roi_name in zip(model_list, name_list):
            params = model.export_params()
            for pname, pdata in params.items():
                description = pdata[0]
                value = pdata[1]
                units = pdata[2] if len(pdata) > 2 else ""
                error = pdata[3] if len(pdata) > 3 else getattr(model, f"{pname}_std", "")
                param_type = "Free" if pname in free_params else "Derived"
                ws.append([case_id, roi_name, pname, description, value, error, units, param_type])

    wb.save(outfile)
    print(f"Saved all cases into one Excel file → {outfile}")


import os
from tqdm import tqdm
import pydmr

# -----------------------------
# Main Pipeline with Batches
# -----------------------------
if __name__ == "__main__":
    site = 'Bordeaux'

    # Dictionary of batch -> case_ids
    batch_cases = {
        1: ['2128_001', '2128_002', '2128_003', '2128_004'],
        2: ['2128_006', '2128_007', '2128_009', '2128_011'],
        3: ['2128_012', '2128_013', '2128_014', '2128_015', '2128_016'],
        4: ['2128_018', '2128_019', '2128_020', '2128_021', '2128_022'],
        5: ['2128_023', '2128_024', '2128_025', '2128_026', '2128_027'],
        6: ['2128_028', '2128_029', '2128_030', '2128_031', '2128_032'],
        7: ['2128_033', '2128_034', '2128_035', '2128_036', '2128_037'],
        8: ['2128_038', '2128_039', '2128_040', '2128_041', '2128_042'],
        9: ['2128_044', '2128_045', '6128_001', '6128_007', '6128_008']
    }

    rois_to_fit = ['kidney_right', 'kidney_left']

    # Lists to collect all models, names, and case IDs for single Excel
    all_models_per_case = []
    all_model_names_per_case = []
    all_case_ids = []

    base_dir = os.path.join(os.getcwd(), 'build', 'dce_10_roi_analysis', site, "Patients")
    model_path = os.path.join(base_dir, "Model_Fit")
    os.makedirs(model_path, exist_ok=True)
    excel_file = os.path.join(model_path, f"{site}.xlsx")

    # Loop over batches and cases with tqdm
    for batch_no, case_ids in tqdm(batch_cases.items(), desc=f'Evaluating Site {site} Batches', unit='batch'):
        for case_id in tqdm(case_ids, desc=f'Screening Batch {batch_no}', unit='case'):
            print(f"Processing Batch {batch_no}, Case {case_id}")

            # Merge DMR if needed
            # combine_dmr(site, case_id, batch_no=batch_no)

            # Read merged DMR
            dmr_dir = os.path.join(base_dir, f"Batch_{batch_no}", "DMR")
            dmr_file = os.path.join(dmr_dir, f"{case_id}.dmr.zip")
            dmr = pydmr.read(dmr_file, 'nest')
            rois, pars = dmr['rois'], dmr['pars']

            models = []
            model_names = []

            for r in rois_to_fit:
                # Fit kidney model
                time, signal, model = kidney_model(
                    rois[f'{case_id}_aorta']['Baseline'],
                    rois[f'{case_id}_{r}']['Baseline'],
                    pars[f'{case_id}_{r}']['Baseline'],
                    f'{case_id}_{r}',
                )

                # Save plot
                plot_path = os.path.join(model_path, f"{case_id}_{r}.png")
                model.plot(time, signal, fname=plot_path, show=False)

                # Print params
                model.print_params(round_to=3)

                models.append(model)
                model_names.append(f"{case_id}_{r}")

            # Append models and info for Excel
            all_models_per_case.append(models)
            all_model_names_per_case.append(model_names)
            all_case_ids.append(case_id)

    # Save all cases and ROIs into a single Excel file
    save_models(all_models_per_case, all_model_names_per_case, all_case_ids, excel_file)
    print(f"Saved all cases {excel_file}")
