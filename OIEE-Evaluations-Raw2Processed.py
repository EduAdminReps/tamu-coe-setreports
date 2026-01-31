"""
OIEE - Office of Institutional Effectiveness & Evaluation


This script reads CSV files and generates CSV files in a format suitable for Pandas.
The original reports are generated through the Office of Institutional Effectiveness & Evaluation
at https://assessment.tamu.edu/ under "Student Course Evaluation".
The "HelioCampus Dashboard Access" link can be found under "SCE Liaisons".
Click on "Login to HelioCampus SCE Dashboards" and authenticate again.
Within the "Assessment Dashboard", select "Download Overall Report".

After selecting the values for the "Term Name" and "College Name" (All) filters,
the CSV report can then be obtained by clicking on "Choose a format to download",
choosing "Crosstab", and selecting "CSV".

Note: The reports may be encoded using a 16-bit Unicode Transformation Format (UTF-16).
Optionally, they can be saved as UTF-8 encoded files for further processing.

Note: A VPN connection may be required to access the HelioCampus Dashboard from remote locations.
"""

from pathlib import Path
import pandas as pd
import re, html, sys

from config import COLLEGE_ID, DEPT_LIST, QUESTIONS, QUESTIONS_MAX, PATHS, map_course_level


########################################################################################################################

# Define custom aggregation function to pick top element within a group
def first_group_element(series):
    for value in series:
        if pd.notna(value):
            return value
    return None


# Define a function to normalize strings by removing extra spaces and line breaks
def normalize_string(s):
    if not isinstance(s, str):
        return s  # Skip non-strings

    # Convert HTML entities like &nbsp; to actual characters
    s = html.unescape(s)

    # Replace line breaks with spaces
    s = s.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')

    # Collapse multiple spaces into one
    s = re.sub(r'\s+', ' ', s)

    # Trim leading/trailing whitespace
    return s.strip()


########################################################################################################################

# Use centralized question text and scale values from config
questions = QUESTIONS
questions_max = QUESTIONS_MAX

questions_dict = { questions[idx]: 'Q' + str(idx) for idx in range(0, 13) }

# OIEE-specific department mapping (differs from AEFIS naming conventions)
dept_dict = {
    'CS-Aerospace Engineering': 'AERO',
    'CS-Biomedical Engineering': 'BMEN',
    'CS-Chemical Engineering': 'CHEN',
    'CS-Civil & Environmental Engr': 'CVEN',
    'CS-College of Engineering': 'CLEN',
    'CS-Computer Science & Engineering': 'CSCE',
    'CS-Electrical & Computer Eng': 'ECEN',
    'CS-Eng Tech & Ind Distribution': 'ETID',
    'CS-Industrial & Systems Eng': 'ISEN',
    'CS-Materials Science & Engr': 'MSEN',
    'CS-Mechanical Engineering': 'MEEN',
    'CS-Multidisciplinary Engineering': 'MTDE',
    'CS-Nuclear Engineering': 'NUEN',
    'CS-Ocean Engineering': 'OCEN',
    'CS-Petroleum Engineering': 'PETE'
}

columns_dict = {
    'Row': 'Row',
    'Term Name': 'CombinedTerm',
    'Department Name': 'Dept',
    'Course': 'CourseNumber',
    'Subject Code': 'Code',
    'Schedule Code': 'Type',
    'Primary Faculty': 'MCSA_Instructor',
    'Instructor Name (ID)': 'INSTR_Instructor',
    'Is TA': 'ISTA',
    'Enrollment Total': 'Enrollment',
    'Distinct count of Survey Response ID': '_count',
    'Avg. Question Option Value': '_avg',
    'Median Question Option Value': '_median',
    'Std. dev. of Question Option Value': '_stddev'
}

# Define a mapping dictionary and update term
replacement_dict = {'Spring': '11', 'Summer': '21', 'Fall': '31'}
reversed_dict = {value: key for key, value in replacement_dict.items()}

base_columns = [
    'CombinedTerm', 'Dept', 'CourseNumber', 'Code',
    'Enrollment', 'MCSA_Instructor', 'INSTR_Instructor'
]
mcsa_question_columns = [
    'Q1_count', 'Q1_avg', 'Q1_median', 'Q1_stddev', 'Q2_count', 'Q2_avg', 'Q2_median', 'Q2_stddev',
    'Q3_count', 'Q3_avg', 'Q3_median', 'Q3_stddev', 'Q4_count', 'Q4_avg', 'Q4_median', 'Q4_stddev',
    'Q5_count', 'Q5_avg', 'Q5_median', 'Q5_stddev', 'Q6_count', 'Q6_avg', 'Q6_median', 'Q6_stddev',
    'Q7_count', 'Q7_avg', 'Q7_median', 'Q7_stddev',
    'Q11_count', 'Q11_avg', 'Q11_median', 'Q11_stddev', 'Q12_count', 'Q12_avg', 'Q12_median', 'Q12_stddev'
]
instr_question_columns = [
    'Q8_count', 'Q8_avg', 'Q8_median', 'Q8_stddev',
    'Q9_count', 'Q9_avg', 'Q9_median', 'Q9_stddev', 'Q10_count', 'Q10_avg', 'Q10_median', 'Q10_stddev'
]
question_columns = mcsa_question_columns + instr_question_columns
keep_columns = base_columns + question_columns

########################################################################################################################


# Use centralized configuration
college_id = COLLEGE_ID
base_path = PATHS['oiee_evaluations_raw']
output_path = PATHS['oiee_evaluations_processed']

# Ensure output directories exist
Path(output_path).mkdir(parents=True, exist_ok=True)
Path(PATHS['assessment_processed']).mkdir(parents=True, exist_ok=True)

csv_evaluations_files = list(Path(base_path).glob(f'OIEE-Evaluations-{college_id}-*.csv'))

# Initialize an empty DataFrame to store rows with missing email addresses
missing_df = pd.DataFrame(columns=['INSTR_Instructor', 'LastName', 'FirstName', 'FullName', 'Email', 'Dept', 'Term'])

# The NameBind binds the instructor fields to an email address
try:
    NameBind_df = pd.read_csv('NAMES-Course-Bind.csv')
except FileNotFoundError:
    print("Error: 'NAMES-Course-Bind.csv' not found. This file is required.")
    sys.exit(1)
except pd.errors.EmptyDataError:
    print("Error: 'NAMES-Course-Bind.csv' is empty.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading 'NAMES-Course-Bind.csv': {e}")
    sys.exit(1)

# Loop through every EVALUATIONS input file into a dataframe, and append the dataframe to DataFrames list
for file in csv_evaluations_files:
    # Extract filename
    filename = file.name
    # Extract the term part: removes prefix and suffix
    term = filename.replace(f'OIEE-Evaluations-{college_id}-', '').replace('.csv', '')
    # Build output filename
    output_filename = f'OIEE-Evaluations-{college_id}-{term}_processed.csv'
    # Full output path
    output_file = Path(output_path) / output_filename
    print(f"Processing File: {file}")

    # These files have encoding='utf-16'.
    try:
        oiee_df = pd.read_csv(file, encoding='utf-16', sep='\t')
    except FileNotFoundError:
        print(f"Error: File '{file}' not found. Skipping.")
        continue
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file}' is empty. Skipping.")
        continue
    except Exception as e:
        print(f"Error reading '{file}': {e}. Skipping.")
        continue
    print(f'Shape of original oiee_df: {oiee_df.shape}')

    # Rename entries in the first two rows to match column names
    oiee_df.loc[oiee_df.index[0]] = oiee_df.loc[oiee_df.index[0]].apply(
        lambda x: normalize_string(x) if pd.notnull(x) else x
    )
    oiee_df.iloc[0] = oiee_df.iloc[0].replace(questions_dict)
    oiee_df.iloc[1] = oiee_df.iloc[1].replace(columns_dict)

    # Create lists of string elements for row0 and row1
    row0 = oiee_df.iloc[0].fillna('').values.tolist()
    row1 = oiee_df.iloc[1].fillna('').values.tolist()

    # Concatenate corresponding list elements and use as column names
    head_array = [element0 + element1 for element0, element1 in zip(row0, row1)]
    oiee_df.columns = head_array
    # Drop the first two rows, which have become irrelevant
    oiee_df = oiee_df.iloc[2:]
    # print(oiee_df.columns)

    # Older files do not have a 'termname' column. Extract it from file name.
    termname_check = reversed_dict[filename[-6:-4]] + ' ' + filename[-10:-6] + ' - College Station'
    if 'CombinedTerm' in oiee_df.columns:
        if (oiee_df['CombinedTerm'] == termname_check).all():
            pass
        else:
            print('Termname mismatch!')
            print('Term Check: ' + str(termname_check))
            print(oiee_df['CombinedTerm'].unique())
    else:
        oiee_df['CombinedTerm'] = termname_check

    # Reset the index to the default integer index
    oiee_df = oiee_df.reset_index(drop=True)

    # Check if all entries in keep_columns are present in df.columns
    valid_columns = set(keep_columns).intersection(oiee_df.columns)
    if set(valid_columns) != set(keep_columns):
        print('Columns missing!')
        missing_columns = set(keep_columns) - set(valid_columns)
        print('Missing Columns: ' + str(missing_columns))
        oiee_df[list(missing_columns)] = ''
    oiee_df = oiee_df[keep_columns]
    print(f'data_df size: {len(oiee_df)}')

    # Multiple Choice Single Answer (MCSA) - Questions without an instructor in AEFIS
    mcsa_df = oiee_df[oiee_df['INSTR_Instructor'].isna() | (oiee_df['INSTR_Instructor'].str.strip() == '')].copy()
    mcsa_df.drop(columns=['Enrollment', 'MCSA_Instructor', 'INSTR_Instructor'], inplace=True)
    mcsa_df.drop(columns=instr_question_columns, inplace=True)
    print(f'mcsa_df size: {len(mcsa_df)}')
    mcsa_df.drop_duplicates(inplace=True)
    print(f'mcsa_df size: {len(mcsa_df)}')

    # Instructor Multi-Choice (INSTR) - Instructor identified
    instr_df = oiee_df[~(oiee_df['INSTR_Instructor'].isna() | (oiee_df['INSTR_Instructor'].str.strip() == ''))].copy()
    instr_df.drop(columns=mcsa_question_columns, inplace=True)
    instr_df.drop(columns=['MCSA_Instructor'], inplace=True)
    print(f'instr_df size: {len(instr_df)}')

    merged_df = pd.merge(instr_df, mcsa_df, on=['CombinedTerm', 'Dept', 'CourseNumber', 'Code'], how='left')
    print(f'merged_df size: {len(merged_df)}')

    data_df = merged_df.copy()
    # Creating course 'Dept', 'Section', 'Semester', 'Year', 'Location'
    data_df['Dept'] = data_df['Dept'].replace(dept_dict)
    data_df['Section'] = data_df['CourseNumber'].str.split().str[-1]
    data_df['Number'] = data_df['CourseNumber'].str.split().str[1]
    data_df['Semester'] = data_df['CombinedTerm'].str.split().str[0]
    data_df['Year'] = data_df['CombinedTerm'].str.split().str[1]
    data_df['Location'] = data_df['CombinedTerm'].str.split('- ', n=1).str[1]
    # Extract the last number in parentheses at the end of the string
    data_df['UIN'] = data_df['INSTR_Instructor'].str.extract(r'\((\d+)\)\s*$')
    # Remove the last parenthetical group from the FullName
    data_df['INSTR_Instructor'] = data_df['INSTR_Instructor'].str.replace(r'\s*\(\d+\)\s*$', '', regex=True)

    # Drop 'CombinedTerm'
    data_df.drop(columns=['CourseNumber'], inplace=True)
    data_df.drop(columns=['CombinedTerm'], inplace=True)
    # print(data_df.columns)

    # Update base columns
    updated_base_columns = [
        'Dept', 'Location', 'Year', 'Code', 'Number', 'Section',
        'Enrollment', 'INSTR_Instructor'
        ]

    # Apply function map_course_level to create column 'Level'
    data_df['Level'] = data_df['Number'].apply(map_course_level)

    # Group by the columns with the same values and apply the custom aggregation function to all other columns
    data_df = data_df.astype('string')
    print(f'Before grouping size: {len(data_df)}')
    grouped_df = data_df.groupby(updated_base_columns, group_keys=False).apply(
        lambda group: group.apply(first_group_element),
        include_groups=False
    ).reset_index()
    print(f'After grouping size: {len(grouped_df)}')

    # The field Instructor has different format across ASSESSMENT, AEFIS, and ARGOS
    # Remove leading/trailing spaces
    grouped_df['INSTR_Instructor'] = grouped_df['INSTR_Instructor'].str.strip()
    # Check if Instructor contains a comma and split the string accordingly
    if grouped_df['INSTR_Instructor'].str.contains(', ').any():
        grouped_df['FullName'] = grouped_df['INSTR_Instructor'].str.split(', ', n=1).str[::-1].str.join(' ')
    elif grouped_df['INSTR_Instructor'].str.contains(',').any():
        grouped_df['FullName'] = grouped_df['INSTR_Instructor'].str.split(',', n=1).str[::-1].str.join(' ')
    else:
        grouped_df['FullName'] = grouped_df['INSTR_Instructor']

    grouped_df['Term'] = grouped_df['Year'].astype(str) + grouped_df['Semester'].replace(replacement_dict)

    grouped_df.rename(columns={'INSTR_Instructor': 'Instructor'}, inplace=True)
    # Merge grouped_df with NameBind_df on 'FullName' to get 'Instructor', 'LastName', 'FirstName', and 'Email'
    NameBind_df.rename(columns={'Dept': 'HomeDept'}, inplace=True)
    if 'Instructor' in NameBind_df.columns:
        NameBind_df.drop(columns=['Instructor'], inplace=True)
    if 'FullName' in NameBind_df.columns:
        NameBind_df.drop(columns=['FullName'], inplace=True)

    grouped_df['UIN'] = grouped_df['UIN'].astype(str)
    NameBind_df['UIN'] = NameBind_df['UIN'].astype(str)
    NameBind_df = NameBind_df.drop_duplicates(subset='UIN')
    grouped_df = pd.merge(grouped_df, NameBind_df, on=['UIN'], how='left')

    # Create 'Responses' from available data; this is a byproduct of AEFIS & OIEE discrepancies
    grouped_df['Responses'] = grouped_df[['Q3_count', 'Q4_count', 'Q5_count', 'Q7_count', 'Q8_count', 'Q9_count']].apply(pd.to_numeric, errors='coerce').min(axis=1)

    for term in grouped_df['Term'].unique():
        term_df = grouped_df[grouped_df['Term'] == term]
        # Reorder the rows by 'Code', 'Number', and 'Section'
        term_df = term_df.sort_values(by=['Code', 'Number', 'Section'])

        # Display the final result
        print(f'Saving File: {output_file}')
        try:
            term_df.to_csv(output_file, index=False)
        except PermissionError:
            print(f"Error: Permission denied writing to '{output_file}'.")
        except Exception as e:
            print(f"Error writing '{output_file}': {e}")

        # Identify rows with missing email addresses
        missing_email_df = term_df[term_df['Email'].isna()]
        # Extract the values of the specified fields for those rows
        missing_email_df = missing_email_df[['Instructor', 'LastName', 'FirstName', 'Email', 'Dept', 'Term']]
        missing_email_df = missing_email_df.drop_duplicates()
        if not missing_email_df.empty:
            # Display the rows with missing email addresses
            missing_email_df = missing_email_df.rename(columns={'Dept': 'HomeDept'})
            print(missing_email_df)
            # Append the rows with missing email addresses to the missing_df DataFrame
            missing_df = pd.concat([missing_df, missing_email_df], ignore_index=True)

missing_df = missing_df.drop_duplicates()
missing_output = f"{PATHS['assessment_processed']}/missing_emails_oiee.csv"
try:
    missing_df.to_csv(missing_output, index=False)
except PermissionError:
    print(f"Error: Permission denied writing to '{missing_output}'.")
except Exception as e:
    print(f"Error writing '{missing_output}': {e}")
