"""
This script gathers the necessary information to create the visual reports.
It takes as input
    1. ARGOS_Grades_Compressed CSV files
    2. ASSESSMENT_Processed CSV files
The former CSV files contain grade information, and the latter have the student feedback.

The section information between ARGOS and ASSESSMENT (AEFIS/OIEE) often do not match.
The section information is unreliable and must be dropped.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys


########################################################################################################################

def load_csv(file_path):
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return None
    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
        return None


# Define a function to calculate weighted average and combined stddev for multiple questions
def combine_stats(group, questions):
    results = {}
    for q in questions:
        count = group[f'Q{q}_count']
        avg = group[f'Q{q}_avg']
        stddev = group[f'Q{q}_stddev']
        # Total count
        total_count = np.sum(count)
        if total_count == 0:
            # Handle case where total count is zero
            results[f'Q{q}_count'] = 0
            results[f'Q{q}_avg'] = np.nan  # No meaningful average
            results[f'Q{q}_stddev'] = np.nan  # No meaningful standard deviation
        else:
            # Weighted average
            weighted_avg = np.sum(count * avg) / total_count
            # Combined standard deviation
            combined_variance = (
                np.sum(count * (stddev ** 2 + (avg - weighted_avg) ** 2)) / total_count
            )
            combined_stddev = np.sqrt(combined_variance)
            # Store results
            results[f'Q{q}_count'] = total_count
            results[f'Q{q}_avg'] = weighted_avg
            results[f'Q{q}_stddev'] = combined_stddev
    return pd.Series(results)

########################################################################################################################


term_list = [
    '202031', '202111', '202121',
    '202131', '202211', '202221',
    '202231', '202311', '202321',
    '202331', '202411', '202421',
    '202431', '202511'
]
# Merge Keys
merge_keys = ['Dept', 'Term', 'Code', 'Number', 'LastName', 'FirstName', 'UIN']

questions = [3, 4, 5, 7, 8, 9]
assessment_course_columns = ['Dept', 'Year', 'Semester', 'Code', 'Number', 'Term', 'Level']
assessment_instructor = ['LastName', 'FirstName', 'FullName', 'Instructor', 'UIN']
assessment_questions = [f'Q{i}_count' for i in questions] + [f'Q{i}_avg' for i in questions] + [f'Q{i}_stddev' for i in questions]
assessment_participation = ['Enrollment', 'Responses']
assessment_numeric = assessment_participation + assessment_questions
assessment_columns = assessment_course_columns + assessment_instructor + assessment_numeric
grouped_keys = assessment_course_columns + assessment_instructor

argos_columns = ['College', 'Dept', 'Code', 'Number', 'Term',
                 'TIN', 'UIN', 'FirstName', 'LastName', 'Email',
                 'Total', 'Count', 'GPA', 'STD', 'InstructorSCH']
new_column_dict = {'Subject': 'Code', 'Enrollment': 'Total'}

college_id = 'EN'  # Engineering College ID
argos_grades_path = 'ARGOS_Grades_Compressed'
assessment_path = 'ASSESSMENT_Processed'
output_path = 'DATA_Evaluations'

# Ensure output directory exists
Path(output_path).mkdir(parents=True, exist_ok=True)

# assessment_frames = []
for term in term_list:
    print(f'Term: {term}')

    ###########################################################################################################

    # Load ARGOS Grades
    argos_file = Path(argos_grades_path + f'/ARGOS-Grades-{college_id}-{term}_compressed.csv')
    print(f'Processing File: {argos_file}')
    argos_df = load_csv(argos_file)
    if argos_df is None:
        continue
    print(f'ARGOS: {len(argos_df)}')

    # Rename and drop columns as appropriate
    argos_df.rename(columns=new_column_dict, inplace=True)
    argos_df = argos_df[argos_columns]
    # Convert numeric columns to numeric type
    argos_numeric = ['Count', 'GPA', 'STD', 'InstructorSCH']
    argos_df[argos_numeric] = argos_df[argos_numeric].apply(pd.to_numeric, errors='coerce')

    ###########################################################################################################


    # Load ASSESSMENT evaluations
    assessment_file = Path(assessment_path +f'/ASSESSMENT-{college_id}-{term}.csv')
    print(f'Processing File: {assessment_file}')
    assessment_df = load_csv(assessment_file)
    if assessment_df is None:
        continue
    else:
        assessment_df.drop('Email', axis=1, inplace=True, errors='ignore')
    print(f'ASSESSMENT: {len(assessment_df)}')

    # EXCLUDE: WEB sections start with a seven.
    assessment_df = assessment_df[~assessment_df['Section'].astype(str).str.startswith('7')]
    # Convert numeric columns to numeric type
    assessment_df = assessment_df[assessment_columns]
    assessment_df[assessment_numeric] = assessment_df[assessment_numeric].apply(pd.to_numeric, errors='coerce')

    ###########################################################################################################
    # Section numbers do not always match between ASSESSMENT (AEFIS or OIEE), and ARGOS.
    ###########################################################################################################

    # Check if all entries in merged_numeric are present in merged_df.columns
    required_columns = set(assessment_numeric)
    existing_columns = set(assessment_df.columns)
    missing_columns = required_columns - existing_columns
    if missing_columns:
        print(f"Missing columns in assessment_df: {missing_columns}")
        continue
    elif assessment_df.empty:
        print("DataFrame assessment_df is empty.")
        continue
    else:
        grouped_stats_df = assessment_df.groupby(grouped_keys)[assessment_numeric].apply(combine_stats, questions=questions).reset_index(drop=False)
        print(f'grouped_stats_df: {len(grouped_stats_df)}')

        grouped_sum_df = assessment_df[assessment_course_columns + assessment_instructor + assessment_participation]
        # Group the DataFrame and apply function
        grouped_sum_df = grouped_sum_df.groupby(grouped_keys).sum().reset_index()
        # grouped_df[assessment_questions] = grouped_df[assessment_questions].divide(grouped_df['Responses'], axis='index')
        print(f'grouped_sum_df: {len(grouped_sum_df)}')

        grouped_df = pd.merge(grouped_stats_df, grouped_sum_df, on=grouped_keys, how='left')
        print(f'DataFrame after merge: grouped_df of length {str(len(grouped_df))}')

    ###########################################################################################################

    missing_keys_argos = set(merge_keys) - set(argos_df.columns)
    missing_keys_grouped = set(merge_keys) - set(grouped_df.columns)
    if missing_keys_argos:
        print(f"Missing keys in ARGOS DataFrame: {missing_keys_argos}")
        continue
    elif missing_keys_grouped:
        print(f"Missing keys in grouped DataFrame: {missing_keys_grouped}")
        continue
    else:
        evaluation_df = pd.merge(grouped_df, argos_df, on=merge_keys, how='left')
        print(f'DataFrame after merge: evaluation_df of length {str(len(grouped_df))}')

    ###########################################################################################################
    # Sometimes there are discrepancies between the primary instructor in ASSESSMENT and ARGOS.
    # In such cases, the row is dropped.
    # Courses taught as Satisfactory/Unsatisfactory courses without a GPA are dropped.
    ###########################################################################################################
    print(f'DataFrame after ARGOS merge: evaluation_df of length {str(len(evaluation_df))}')
    # Drop rows with NaN values in 'GPA' column
    evaluation_df = evaluation_df.dropna(subset=['GPA'])
    # Drop rows with NaN values in 'UIN' column
    evaluation_df = evaluation_df.dropna(subset=['UIN'])
    print('DataFrame after empty UIN drop: evaluation_df of length ' + str(len(evaluation_df)))
    # evaluation_df.to_csv('DATA_Evaluations/' + dept + '_evaluation.csv', index=False)
    output_file = f'{output_path}/Data-{college_id}-{term}.csv'
    print(f'Saving File: {output_file}')
    try:
        evaluation_df.to_csv(output_file, index=False)
    except PermissionError:
        print(f"Error: Permission denied writing to '{output_file}'.")
    except Exception as e:
        print(f"Error writing '{output_file}': {e}")
