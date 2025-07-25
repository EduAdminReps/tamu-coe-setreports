"""
ASSESSMENT

OIEE - Office of Institutional Effectiveness & Evaluation
AEFIS - Assessment, Evaluation, Feedback & Intervention System

Ideally, the files in OIEE_Evaluations_Processed/ and AEFIS_Evaluations_Processed/ would be identical.
However, there are several issues that make them differ.
1. The OIEE report format has changed significantly over time, which creates challenges for downstream applications.
2. AEFIS data is generated dynamically, and the web portal restricts query size.
3. Some of the initial OIEE files were missing columns and counts.
4. The AEFIS data structure is cumbersome.
When both files are available for a given department and term, it is best to use the OIEE file.
If an OIEE file is missing, use the corresponding AEFIS file instead.
"""

import pandas as pd
from pathlib import Path

college_id = 'EN'  # Engineering College ID

# List of Engineering Departments and Units
dept_list = [
    'AERO', 'BMEN', 'CHEN',
    'CLEN', 'CSCE', 'CVEN',
    'ECEN', 'ETID', 'ISEN',
    'MEEN', 'MSEN', 'MTDE',
    'NUEN', 'OCEN', 'PETE'
]
# dept_list = ['AERO']

# List of Terms
term_list = [
    '202331', '202111', '202121',
    '202131', '202211', '202221',
    '202231', '202311', '202321',
    '202331', '202411', '202421',
    '202431', '202511'
]

for term in term_list:
    print('Term: ' + term)

    # Read the OIEE data into a DataFrame
    oiee_file = Path('OIEE_Evaluations_Processed/OIEE-Evaluations-EN-' + term + '_processed.csv')
    # Check if the file exists
    if oiee_file.exists():
        # Load the file into a DataFrame
        oiee_df = pd.read_csv(oiee_file)
        # HECM Sections contain letters
        oiee_df['Section'] = oiee_df['Section'].astype(str)
        oiee_flag = True
    else:
        print(f"The file at {oiee_file} does not exist.")
        oiee_flag = False

    dataframes = []
    aefis_flag = True
    for dept in dept_list:
        # Read the AEFIS data into a DataFrame
        aefis_file = Path(f'AEFIS_Evaluations_Processed/{dept}/{dept}-{term}_aefis.csv')
        # Check if the file exists
        if aefis_file.exists():
            # Load the file into a DataFrame
            aefis_dept_df = pd.read_csv(aefis_file)
            # HECM Sections contain letters
            aefis_dept_df['Section'] = aefis_dept_df['Section'].astype(str)
            if not aefis_dept_df.empty:
                dataframes.append(aefis_dept_df)  # Append to the list if successfully loaded
            else:
                print(f"The file at aefis_dept_df is empty.")
        else:
            print(f"The file at {aefis_file.name} does not exist.")
            aefis_flag = False

    if dataframes:
        dataframes = [df for df in dataframes if not df.empty]
        aefis_df = pd.concat(dataframes, ignore_index=True)
        print(f"Combined_df has {len(aefis_df)} rows.")
    else:
        print("No valid dataframes found.")
        aefis_flag = False

    assessment_file = Path(f'ASSESSMENT_Processed/ASSESSMENT-{college_id}-{term}.csv')
    if oiee_flag:
        assessment_df = oiee_df
        assessment_df.to_csv(assessment_file, index=False)
        print(f"Saved file: {assessment_file} using OIEE file.")
    elif aefis_flag:
        assessment_df = aefis_df
        assessment_df.to_csv(assessment_file, index=False)
        print(f"Saved file: {assessment_file} using AEFIS file.")
    else:
        print("Neither AEFIS nor OIEE files exist.")
        # Skip the term if the file does not exist
        continue
