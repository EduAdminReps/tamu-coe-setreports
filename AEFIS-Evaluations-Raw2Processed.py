"""
AEFIS - Assessment, Evaluation, Feedback & Intervention System

This script reads CSV files and generates CSV files in a format suitable for Pandas.
The original reports are generated through AEFIS (https://tamu.aefis.net/) under
"Student Course Evaluation Results by Instructor Report" with appropriate parameters.
Every report is then exported to an AEFIS CSV file with its inconvenient block format.
Any CSV report in the latter format can act as an input to this script for further processing.

AEFIS Parameters:
    Term: Fall 2023 - College Station
    Institution: TAMU [TAMU]
    College: Engineering [EN]
    Department: CS-Unit Engineering [CS-DEPT]
    Instructor Type: Instructor
    Course Section: No Courses for Selected Faculty
    Question Type: Multi-Choice, Single Answer [MCSA], Multi-Choice, Multi Answer [MCMA] and 4 more...
    Question: No Question
    Show Comments: Yes

The emphasis is on the standard AEFIS questions selected by the 2022 College of Engineering
Student Evaluation of Teaching Task Force.
2022 COE SET Memo - AEFIS.pdf
"""

from pathlib import Path
import pandas as pd
import glob, sys

########################################################################################################################


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
NameBind_merge_columns = ['LastName', 'FirstName', 'Dept', 'Instructor', 'FullName']

# Ensure output directory for missing emails exists
Path('ASSESSMENT_Processed').mkdir(parents=True, exist_ok=True)

# List of standard AEFIS questions
size = 14 # Number of questions + dummy question at index 0
initial_value = None
questions = [initial_value] * size
questions[0] = '' # Leading 0 is a dummy value
questions[1] = 'Begin this course evaluation by reflecting on your own level of engagement and participation in the course. What portion of the class preparation activities (e.g., readings, online modules, videos) and assignments did you complete?'
# Scale (1 - 4) 1 = <50% 4 = >90%
# Type: Multi-Choice, Single Answer
questions[2] = 'Based on what the instructor(s) communicated, and the information provided in the course syllabus, I understood what was expected of me.'
# Scale (1 - 3) 1 = No, I did not understand what was expected of me. 3 = Yes, I understood what was expected of me.
# Type: Multi-Choice, Single Answer
questions[3] = 'This course helped me learn concepts or skills as stated in course objectives/outcomes.'
# Scale (1 - 4) 1 = This course did not help me learn the concepts or skills. 4 = This course definitely helped me learn the concepts or skills.
# Type: Multi-Choice, Single Answer
questions[4] = 'In this course, I engaged in critical thinking and/or problem solving.'
# Scale (1 - 4) 1 = Never 4 = Frequently
# Type: Multi-Choice, Single Answer
questions[5] = 'Please rate the organization of this course.'
# Scale (1 - 4) 1 = Not at all organized 4 = Very well organized
# Type: Multi-Choice, Single Answer
questions[6] = 'In this course, I learned to critically evaluate diverse ideas and perspectives.'
# Scale (1 - 6) 1 = Strongly disagree 6 = Not Applicable
# Type: Multi-Choice, Single Answer
questions[7] = 'Feedback in this course helped me learn. Please note, feedback can be either informal (e.g., in class discussion, chat boards, think-pair-share, office hour discussions, help sessions) or formal (e.g., written or clinical assessments, review of exams, peer reviews, clicker questions).'
# Scale (1 - 6) 1 = No feedback was provided. 6 = Feedback provided was extremely helpful.
# Type: Multi-Choice, Single Answer
questions[8] = 'The instructor fostered an effective learning environment.'
# Scale (1 - 5) 1 = Strongly disagree 5 = Strongly agree
# Type: Instructor Multi-Choice
questions[9] = 'The instructor\'s teaching methods contributed to my learning.'
# Scale (1 - 3) 1 = Did not contribute 3 = Contributed a lot
# Type: Instructor Multi-Choice
questions[10] = 'The instructor encouraged students to take responsibility for their own learning.'
# Scale (1 - 3) 1 = Did not encourage 3 = Frequently encouraged
# Type: Instructor Multi-Choice
questions[11] = 'Is this course required?'
# Scale (1 - 2) 1 = No 2 = Yes
# Type: Multi-Choice, Single Answer
questions[12] = 'Expected Grade in this Course'
# Scale (1 - 8) 1 = A 8 = U
# Type: Multi-Choice, Single Answer
questions[13] = 'Please provide any general comments about this course.'
questions_max = [0, 4, 3, 4, 4, 4, 6, 6, 5, 3, 3, 2, 8, 0] # Leading 0 is a dummy value

# Questions to be included in the processed CSV file
chosen_questions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# List of columns in the processed CSV file
dfs_columns_course = ['Dept', 'Year', 'Semester', 'Location', 'Code', 'Number', 'Section', 'Instructor']
dfs_columns_student_numbers = ['Enrollment', 'Responses']

dfs_columns_numeric = []
for qidx in chosen_questions:
    # Question label qlbl
    qlbl = 'Q' + str(qidx)
    dfs_columns_numeric += [qlbl + '_count', qlbl + '_avg', qlbl + '_stddev', qlbl + '_median']
dfs_columns_numeric = dfs_columns_student_numbers + dfs_columns_numeric
dfs_columns = dfs_columns_course + dfs_columns_numeric
# print(dfs_columns)

# Define a mapping dictionary and update term
replacement_dict = {'Spring': '11', 'Summer': '21', 'Fall': '31'}

# Define a map from course number to course level
def map_course_level(course_level):
    course_level_string = str(course_level)
    course_leading_digit = course_level_string[0]
    if course_leading_digit == '1':
        level = 'one'
    elif course_leading_digit == '2':
        level = 'two'
    elif course_leading_digit == '3':
        level = 'three'
    elif course_leading_digit == '4':
        level = 'four'
    else:
        level = 'grad'
    return level

# List of Engineering Departments and Units
dept_list = ['AERO', 'BMEN', 'CHEN', 'CLEN', 'CSCE', 'CVEN', 'ECEN',
             'ISEN', 'MEEN', 'MSEN', 'MTDE', 'NUEN', 'OCEN', 'PETE']

# dept_list = ['BAEN', 'ETID']

# Initialize an empty DataFrame to store rows with missing email addresses
missing_df = pd.DataFrame(columns=['Instructor', 'LastName', 'FirstName', 'FullName', 'Email', 'Dept', 'Term'])

# Using index dept for 'Dept'
# The files to be process should be in the directory 'AEFIS_Evaluations_Raw/'
# The files in the directory 'AEFIS_Evaluations_Raw/' + dept + '/' are archived
for dept in dept_list:
    print('Department: ' + dept)
    # Ensure department output directory exists
    Path(f'AEFIS_Evaluations_Processed/{dept}').mkdir(parents=True, exist_ok=True)
    # List all departmental CSV files in directory 'AEFIS_Evaluations_Raw/'
    csv_files = glob.glob('AEFIS_Evaluations_Raw/' + dept +'*.csv') # Process new files
    # UNCOMMENT BELOW to process archived files
    # csv_files = glob.glob('AEFIS_Evaluations_Raw/' + dept + '/' + dept + '*.csv')

    # Loop through every CSV input file into a dataframe and process them sequentially
    for file in csv_files:
        print('Processing File: ' + file)
        try:
            data = pd.read_csv(file, header=None)
        except FileNotFoundError:
            print(f"Error: File '{file}' not found. Skipping.")
            continue
        except pd.errors.EmptyDataError:
            print(f"Error: File '{file}' is empty. Skipping.")
            continue
        except Exception as e:
            print(f"Error reading '{file}': {e}. Skipping.")
            continue

        # Concatenate all the dataframes in DataFrames list into a single dataframe
        data = data.reset_index(drop=True)

        rows = []
        for idx, df in data.groupby(data[0].eq('Term').cumsum()):
            df = pd.DataFrame(df.iloc[0:].values, columns=None).copy()

            # Separating the 'Location', 'Semester', and 'Year'
            input_string = df.iloc[0, 1]
            # Split the string into substrings using ' - ' as the delimiter
            parsed_list = input_string.split(' - ') # CHECK CONDITION NEEDED FOR PRODUCTION VERSION
            # Extracting individual component 'Location'
            location = parsed_list[-1]
            # Extracting individual components 'Semester' and 'Year'
            [semester, year] = parsed_list[0].split()

            # Separating the course 'Code', course 'Number', and course 'Section'
            course_string = df.iloc[2, 1]
            # Splitting the string into substrings and extracting components 'Code', 'Number', and 'Section'
            [code, number, section] = course_string.split()

            # Extracting individual components 'Instructor', 'Enrollment', and 'Responses'
            [instructor, enrollment, responses] = [df.iloc[3, 1], df.iloc[4, 1], df.iloc[5, 1]]

            # Creating the beginning of every row
            row = [dept, year, semester, location, code, number, section, instructor, enrollment, responses]
            # Appending numerical values for every question
            for qidx in chosen_questions:
                row.append(responses)
                row.append(df.loc[df[0] == questions[qidx]].values[0][1])
                row.append(df.loc[df[0] == questions[qidx]].values[0][2])
                row.append(df.loc[df[0] == questions[qidx]].values[0][3])
                # row.append(df.loc[df[0] == questions[qidx]].values[0][4])
            rows.append(row)
            dfs = pd.DataFrame(rows, columns=dfs_columns)

        # Creating course 'Term'
        dfs['Term'] = dfs['Year'].astype(str) + dfs['Semester'].replace(replacement_dict)

        # The field Instructor has different format across ASSESSMENT, AEFIS, and ARGOS
        # Creating course 'Instructor' while removing leading/trailing spaces
        dfs['Instructor'] = dfs['Instructor'].str.strip()
        # Creating 'LastName', 'FirstName', and 'FullName'
        # Check if Instructor contains a comma and split the string accordingly
        if dfs['Instructor'].str.contains(',').any():
            dfs[['LastName', 'FirstName']] = dfs['Instructor'].str.split(', ', n=1, expand=True)
            dfs['FullName'] = dfs['FirstName'] + ' ' + dfs['LastName']
        else:
            dfs['FullName'] = dfs['Instructor']

        # Merge dfs with NameBind_df on 'Instructor', 'LastName', 'FirstName', and 'FullName' to get 'Email' and 'UIN'
        dfs = pd.merge(dfs, NameBind_df, on=NameBind_merge_columns, how='left')

        # Apply function map_course_level to create 'Level'
        dfs['Level'] = dfs['Number'].apply(map_course_level)

        for term in dfs['Term'].unique():
            term_df = dfs[dfs['Term'] == term]
            # Reorder the rows by 'Code', 'Number', and 'Section'
            term_df = term_df.sort_values(by=['Code', 'Number', 'Section'])
            output_file = f'AEFIS_Evaluations_Processed/{dept}/{dept}-{term}_aefis.csv'
            print(f'Saving File: {output_file}')
            try:
                term_df.to_csv(output_file, index=False)
            except PermissionError:
                print(f"Error: Permission denied writing to '{output_file}'.")
            except Exception as e:
                print(f"Error writing '{output_file}': {e}")
            # Uncomment to place files in the ASSESSMENT_Processed directory
            # term_df.to_csv('ASSESSMENT_Processed/' + dept + '/' + dept + '-' + str(term) + '_assessment.csv', index=False)

            # Identify rows with missing email addresses
            missing_email_df = dfs[dfs['Email'].isna()]
            # Extract the values of the specified fields for those rows
            # print(missing_email_df.columns)
            missing_email_df = missing_email_df[['Instructor', 'LastName', 'FirstName', 'FullName', 'Email', 'Dept', 'Term']]
            missing_email_df = missing_email_df.drop_duplicates()
            if not missing_email_df.empty:
                # Display the rows with missing email addresses
                print(missing_email_df)
                # Append the rows with missing email addresses to the missing_df DataFrame
                missing_df = pd.concat([missing_df, missing_email_df], ignore_index=True)

missing_df = missing_df.drop_duplicates()
try:
    missing_df.to_csv('ASSESSMENT_Processed/missing_emails_aefis.csv', index=False)
except PermissionError:
    print("Error: Permission denied writing to 'ASSESSMENT_Processed/missing_emails_aefis.csv'.")
except Exception as e:
    print(f"Error writing 'ASSESSMENT_Processed/missing_emails_aefis.csv': {e}")
