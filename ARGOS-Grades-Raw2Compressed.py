"""
ARGOS - Adhoc Report Generation Output Solution

This script reads CSV files and generates CSV files in a format suitable for Pandas.
The original reports are generated through ARGOS (https://howdy.tamu.edu/) using the routine
called 'ARGOS_Grades_Raw' on the VOAL virtual machine.
Any CSV report in the proper format can act as an input to this script for further processing.

Input Columns:
    Campus, College, Dept, Subject, Number, Term, Section, CRN,
    TIN, UIN, FirstName, LastName, Email, Primary,
    Census, Enrollment, SCH, Method, Type, LectureMin, LabMin
    StudentUIN, Credits, Grade
"""


from pathlib import Path
import os
import glob, sys
import pandas as pd
import numpy as np

from config import COLLEGE_ID, GRADE_POINTS, PATHS

# Use centralized configuration
grade_points = GRADE_POINTS
college_id = COLLEGE_ID
base_path = PATHS['argos_raw']
output_path = PATHS['argos_compressed']

# Ensure output directory exists
Path(output_path).mkdir(parents=True, exist_ok=True)

grade_files = glob.glob(os.path.join(base_path, f'ARGOS-Grades-{college_id}-*.csv'))

# Group by relevant columns
group_columns = [
    'Campus', 'College', 'Dept', 'Subject', 'Number', 'Term',
    'TIN', 'UIN', 'FirstName', 'LastName', 'Email',
    'Type', 'LectureMin', 'LabMin'
]

output_columns = [
    'Campus', 'College', 'Dept', 'Subject', 'Number', 'Term',
    'TIN', 'UIN', 'FirstName', 'LastName', 'Email',
    'Enrollment', 'Count', 'GPA', 'STD', 'InstructorSCH',
    'Type', 'LectureMin', 'LabMin'
]

for file in grade_files:
    # Extract filename
    filename = os.path.basename(file)
    # Extract the term part: removes prefix and suffix
    term = filename.replace(f'ARGOS-Grades-{college_id}-', '').replace('.csv', '')
    # Build output filename
    output_filename = f'ARGOS-Grades-{college_id}-{term}_compressed.csv'
    # Full output path
    output_file = os.path.join(output_path, output_filename)
    print('Processing File: ' + file)

    # Read the CSV file into a DataFrame
    try:
        grades_df = pd.read_csv(file)
    except FileNotFoundError:
        print(f"Error: File '{file}' not found. Skipping.")
        continue
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file}' is empty. Skipping.")
        continue
    except Exception as e:
        print(f"Error reading '{file}': {e}. Skipping.")
        continue
    grades_df = grades_df.drop(columns=['CRN', 'Census', 'Enrollment'])
    print(f'grades_df size: {len(grades_df)}')
    # Filter rows where 'Primary' is 'Y'
    grades_df = grades_df[grades_df['Primary'] == 'Y']
    print(f'grades_df size after Primary pruning: {len(grades_df)}')
    # Filter rows where 'Method' is 'TR' or equivalent
    grades_df = grades_df[(grades_df['Method'].isin(['TR', 'REM', 'DUAL', 'MIXED','WEB']))]
    # grades_df = grades_df[(grades_df['Type'].isin(['LEC', 'LAB', 'LL']))]
    print(f'grades_df size after Method and Type pruning: {len(grades_df)}')
    grades_df = grades_df.drop(columns=['Primary', 'Method'])
    # By default, rows with NaN in any of the grouping columns are omitted from the groupby() output entirely.
    grades_df['LabMin'] = grades_df['LabMin'].fillna(0)
    grades_df['LectureMin'] = grades_df['LectureMin'].fillna(0)

    grades_df[['LectureMin', 'LabMin', 'Credits']] = grades_df[['LectureMin', 'LabMin', 'Credits']].apply(pd.to_numeric,
                                                                                              errors='coerce')
    InstructorCredits_vals = []
    for _, row in grades_df.iterrows():
        if row['Type'] in ['LEC']:
            InstructorCredits = np.minimum(row['Credits'], 3)
        elif row['Type'] in ['LAB']:
            InstructorCredits = np.minimum(row['Credits']/3, 3)
        elif row['Type'] in ['LL']:
            InstructorCredits = np.minimum(np.minimum(row['Credits'], row['LectureMin']), 3)
        else:
            InstructorCredits = np.minimum(row['Credits'], 3)
        InstructorCredits_vals.append(InstructorCredits)
    grades_df['InstructorCredits'] = InstructorCredits_vals
    print(f'InstructorCredits_vals size: {len(InstructorCredits_vals)}')

    # Map grades to grade points, setting invalid grades to NaN
    grades_df['GP'] = grades_df['Grade'].map(grade_points).where(grades_df['Grade'].isin(grade_points.keys()), np.nan)

    # Calculate grade counts
    grades_df[group_columns + ['Section']] = grades_df[group_columns + ['Section']].astype('string')

    # Compute enrollment
    grade_counts_df = grades_df.groupby(group_columns)['Grade'].value_counts().unstack(fill_value=0)
    grade_counts_df['Enrollment'] = grade_counts_df.sum(axis=1)
    grade_counts_df = grade_counts_df.reset_index()

    # Compute student credit hours (SCH)
    grades_df['SCH'] = pd.to_numeric(grades_df['SCH'], errors='coerce')
    section_df = grades_df[group_columns + ['Section', 'SCH']].copy()
    # Only the first occurrence of each unique combination of group_columns + ['Section']
    section_grouped_df = section_df.groupby(group_columns + ['Section'], as_index=False).agg({
        'SCH': 'first'
    })
    grade_sch_df = section_grouped_df.groupby(group_columns)['SCH'].sum().reset_index()

    # Compute 'Count', 'GPA', 'STD', and 'InstructorSCH'
    grades_df[['GP', 'InstructorCredits']] =  grades_df[['GP', 'InstructorCredits']].apply(pd.to_numeric, errors='coerce')
    data_gpa_sch_df = grades_df.groupby(group_columns).agg(
        Count=('GP', 'count'),
        GPA=('GP', 'mean'),        # Mean GPA
        STD=('GP', 'std'),        # Standard Deviation of GPA
        InstructorCreditSum=('InstructorCredits', 'sum')
    )
    # For convenience, set STD to 0 when Count is 1
    data_gpa_sch_df['STD'] = data_gpa_sch_df.apply(lambda row: 0 if row['Count'] <= 1 else row['STD'], axis=1)
    data_gpa_sch_df = data_gpa_sch_df.reset_index()

    # InstructorSCH should not exceed SCH
    data_gpa_sch_df['InstructorSCH'] = np.minimum(data_gpa_sch_df['InstructorCreditSum'], grade_sch_df['SCH'])

    # Flag rows where the minimum was from 'SCH'
    flag_series = data_gpa_sch_df['InstructorSCH'] < data_gpa_sch_df['InstructorCreditSum']
    flag_count = flag_series.sum()
    print(f'Flag for InstructorSCH less than InstructorCredits: {flag_count}')
    # flagged_courses = data_gpa_sch_df[flag_series]
    # print(flagged_courses[['Subject', 'Number']])

    # Combine GPA data with grade counts
    results_df = pd.merge(grade_counts_df, data_gpa_sch_df, on=group_columns, how='outer')
    results_df = pd.merge(results_df, grade_sch_df, on=group_columns, how='outer')
    results_df = results_df.reset_index()

    results_df = results_df[output_columns + ['SCH', 'InstructorCreditSum']]

    # Display the final result
    print(f'Saving File: {output_file}')
    try:
        results_df.to_csv(output_file, index=False)
    except PermissionError:
        print(f"Error: Permission denied writing to '{output_file}'.")
    except Exception as e:
        print(f"Error writing '{output_file}': {e}")
