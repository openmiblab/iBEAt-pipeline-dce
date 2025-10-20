import matplotlib.pyplot as plt
import numpy as np
import math
import os
import dbdicom as db
import pandas as pd

#Helper: Plot Max Intensity Projection 
def plot_mip_mosaic(images, destpath):
    n_cases = len(images)
    if n_cases == 0:
        print("No images to plot.")
        return
    
    # Layout for mosaic
    n_cols = math.ceil(math.sqrt(n_cases))
    n_rows = math.ceil(n_cases / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = axes.flatten()

    for i, (case_id, img) in enumerate(images):
        ax = axes[i]

        # Squeeze to drop singleton dims
        img = np.squeeze(img)

        # If 3D (y, x, t) -> MIP across time
        if img.ndim == 3:
            mip = np.max(img, axis=-1)
        # If already 2D
        elif img.ndim == 2:
            mip = img
        else:
            raise ValueError(f"Unsupported shape for case {case_id}: {img.shape}")

        ax.imshow(mip, cmap='gray')
        ax.set_title(str(case_id), fontsize=8)
        ax.axis("off")

    # Hide extra axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    mosaic_path = os.path.join(destpath, "mip_mosaic.png")
    plt.savefig(mosaic_path, dpi=200)
    plt.close()
    print(f"Saved MIP mosaic {mosaic_path}")

# Helper: Plot AIF 
def plot_aif_mosaic(folder_path, destpath):
    # List all CSV files
    csv_files = [f for f in os.listdir(folder_path) if f.endswith("_aif.csv")]
    n_cases = len(csv_files)
    if n_cases == 0:
        print("No CSV files found in folder:", folder_path)
        return

    # Layout for mosaic
    n_cols = math.ceil(math.sqrt(n_cases))
    n_rows = math.ceil(n_cases / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.5*n_rows))
    axes = axes.flatten()

    for i, csv_file in enumerate(csv_files):
        ax = axes[i]

        case_id = csv_file.replace("_aif.csv", "")
        df = pd.read_csv(os.path.join(folder_path, csv_file))
        times = df["Time (s)"].values
        intensities = df["AIF Intensity"].values

        ax.plot(times, intensities, marker='o', linewidth=1)
        ax.set_title(case_id, fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)

    # Hide extra axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_path = os.path.join(destpath, "aif_curve_mosaic.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved AIF mosaic {out_path}")

# Main Protocol 
def checkpoint(site):
    #load directories and databases
    datapath = os.path.join(os.getcwd(), 'build', 'dce_3_mip')
    imgpath = os.path.join(datapath, site, "Patients")
    
    dstdatapath = os.path.join(os.getcwd(), 'build', 'dce_6_checkpoint')
    destpath =  os.path.join(dstdatapath, site, "Patients")
    os.makedirs(destpath, exist_ok=True)

    csvpath = os.path.join(os.getcwd(), 'build', 'dce_5_aorta2csv', site, "Patients")

    imgdatabase = db.series(imgpath)
    DCE_mip = [entry for entry in imgdatabase if entry[3][0].strip().lower() == 'dce_3_mip']

    # Collect all studies into one list
    images = []
    for study in DCE_mip:
        case_id = study[1]   # patient/study ID
        array = db.pixel_data(study) #extract pixel data
        images.append((case_id, array)) 

    # Plot *one mosaic* with all cases
    if site == 'Sheffield':
        plot_aif_mosaic(csvpath, destpath)
    else:
        plot_mip_mosaic(images, destpath)
        plot_aif_mosaic(csvpath, destpath)

# Call Task
if __name__ == '__main__':
    checkpoint('Sheffield')
