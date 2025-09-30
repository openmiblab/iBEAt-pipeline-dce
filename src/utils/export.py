import os
from datetime import date

import pandas as pd

def antaros_to_redcap(input_file, output_path):

    # Read original table
    table = pd.read_csv(input_file)

    # Drop rows without results
    table = table.dropna(subset=['Value'])

    # Replace subject by harmonized ID
    harmonized_id = []
    for s in table['Subject'].values:
        p = s.split('-')
        id = f"{p[0]}-{p[1]}{p[2]}"
        harmonized_id.append(id)
    table['harmonized_id'] = harmonized_id
    table = table.drop('Subject', axis=1)

    # Build parameter column
    parameter = []
    body_part = []
    biomarker = []
    image = []
    category = []
    for idx, row in table.iterrows():
        if row['Parameter Name'] == 'LIVER FAT':
            p = 'liver-r2_star-mean-pdff'
            b = 'liver'
            im = 'r2_star'
            cat = 'mean'
        elif row['Parameter Name'] == 'LIVER R2star':
            p = 'liver-r2_star-mean-R2star'
            b = 'liver'
            im = 'r2_star'
            cat = 'mean'
        elif row['Parameter Name'] == 'VAT':
            p = 'visceral_adipose_tissue-shape-area'
            b = 'visceral_adipose_tissue'
            im = 'mask'
            cat = 'shape'
        elif row['Parameter Name'] == 'PANCREAS FAT':
            p = 'pancreas-r2_star-mean-pdff'
            b = 'pancreas'
            im = 'r2_star'
            cat = 'mean'
        parameter.append(p)
        body_part.append(b)
        biomarker.append(p.split('-')[-1])
        image.append(im)
        category.append(cat)
    table['parameter'] = parameter
    table['body_part'] = body_part
    table['biomarker'] = biomarker
    table['biomarker_category'] = category
    table['image'] = image

    # Rename columns
    table = table.rename(columns={'Units': 'unit', 'Value': 'result', 'Date':'visit_date'})

    # Drop unnecessary columns
    table = table.drop(['Site','Kidney (L or R)','ROI (Cortex or Medulla)', 'Metric','Parameter Name'], axis=1)

    # Add visit number based on date
    
    # -- 1. Sort so earliest comes first
    table = table.sort_values('visit_date')

    # -- 2. Compute duplicate flag
    def visit_nr(group):
        if len(group) == 1:
            group['visit_nr'] = 0  # only row → 0
        else:
            group['visit_nr'] = 0  # default 0
            group.loc[group.index[-1], 'visit_nr'] = 2  # latest row → 1
        return group

    table = table.groupby(['harmonized_id','parameter'], group_keys=False).apply(visit_nr)

    # Reorder columns
    table = table[['harmonized_id', 'visit_nr', 'visit_date', 'parameter', 'result', 'unit', 'body_part', 'image', 'biomarker_category','biomarker']]

    # Rename units
    table['unit'] = table['unit'].replace('cm2', 'cm^2')

    # Write results
    today = date.today().strftime("%Y-%m-%d")
    output_file = os.path.join(output_path, f'Patients_AntarosData_{today}.csv')
    table.to_csv(output_file, index=None)

    # Write in wide format
    table = table.pivot(index=['harmonized_id','visit_nr'], columns='parameter', values='result')
    output_file = os.path.join(output_path, f'Patients_AntarosData_{today}_wide.csv')
    table.to_csv(output_file)

