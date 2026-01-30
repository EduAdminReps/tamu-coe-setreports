"""
OIEE - Office of Institutional Effectiveness & Evaluation


This script reads CSV files and generates new CSV files in a format suitable for use with Pandas.
The original reports are generated through the Office of Institutional Effectiveness & Evaluation
at https://assessment.tamu.edu/ under "Student Course Evaluation".
To obtain the reports:
  1. The "HelioCampus Dashboard Access" link can be found under "SCE Liaisons."
  2. Click "Login to HelioCampus SCE Dashboards" and authenticate.
  3. Within the "Assessment Dashboard," select "Download Comments."
  4. After choosing the appropriate values for the "Term Name" and "College Name (All)" filters,
  click "Choose a format to download," select "Crosstab," and then choose "CSV."

Note: The reports may be encoded using a 16-bit Unicode Transformation Format (UTF-16).
Optionally, they can be saved as UTF-8 encoded files for further processing.

Note: A VPN connection may be required to access the HelioCampus Dashboard from remote locations.

A preliminary step accomplished in this script is to anonymize the comments before sending them out.
The data collected from the Office of InstitutionalEffectiveness & Evaluation does not contain student names.
However, the comment themselves may reveal the identity of the instructor to the AI model.
When using Google Gemini, settings control how the data is used.
When constructing reports, select the following two options:
  1. "Do not use for training"
  2. "Keep my data private"
Nevertheless, faculty may prefer to have queries anonymized before they are sent to the AI model.
This script accomplishes this by using rapidfuzz to replace instructor first and last names with generic placeholders.
Levenshtein distance is used for approximate matching, in case names are misspelled.
"""

from pathlib import Path
import pandas as pd
from unidecode import unidecode
import unicodedata
import re, html, sys
from rapidfuzz import fuzz


########################################################################################################################

# Optional: fix inherited mojibake
def reverse_mojibake(text):
    if isinstance(text, str):
        try:
            return text.encode('latin1').decode('utf-8')
        except:
            return text
    return text


def normalize_string(s):
    if not isinstance(s, str):
        return s
    s = reverse_mojibake(s)
    s = unicodedata.normalize('NFKC', s)
    s = html.unescape(s)
    s = s.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    s = s.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


# Function to replace first names and last names separately
def anonymize_text(text, FullName):
    """
    Replace occurrences of first and last names in text with generic names.
    Uses fuzzy matching for robustness against misspellings.
    """
    if not isinstance(text, str):
        return ''
    elif not isinstance(FullName, str):
        return text

    # Default anonymized names
    generic_first_name = 'Jordan'
    generic_last_name = 'Taylor'

    # Split FullName safely
    name_parts = FullName.strip().split(maxsplit=1)
    if len(name_parts) == 0:
        return text
    elif len(name_parts) == 1:
        firstname = name_parts[0].lower()
        lastname = ''
    else:
        firstname = name_parts[0].lower()
        lastname = name_parts[1].lower()

    # Start anonymizing
    words = text.split()
    anonymized_words = []
    for word in words:
        word_lower = word.lower()
        first_name_similarity = fuzz.partial_ratio(firstname, word_lower) if firstname else 0
        last_name_similarity = fuzz.partial_ratio(lastname, word_lower) if lastname else 0

        if len(word) <= 1:
            anonymized_words.append(word)
        elif first_name_similarity >= 90:  # Adjust the similarity threshold as needed
            anonymized_words.append(generic_first_name)
        elif last_name_similarity >= 90:  # Adjust the similarity threshold as needed
            anonymized_words.append(generic_last_name)
        else:
            anonymized_words.append(word)

    return ' '.join(anonymized_words)


########################################################################################################################

# List of standard AEFIS questions
size = 14 # Number of questions + dummy question at index 0
initial_value = None
questions = [initial_value] * size
questions[0] = '' # Leading 0 is a dummy value
questions[1] = 'Begin this course evaluation by reflecting on your own level of engagement and participation in the course. What portion of the class preparation activities (e.g., readings, online modules, videos) and assignments did you complete?'
questions[2] = 'Based on what the instructor(s) communicated, and the information provided in the course syllabus, I understood what was expected of me.'
questions[3] = 'This course helped me learn concepts or skills as stated in course objectives/outcomes.'
questions[4] = 'In this course, I engaged in critical thinking and/or problem solving.'
questions[5] = 'Please rate the organization of this course.'
questions[6] = 'In this course, I learned to critically evaluate diverse ideas and perspectives.'
questions[7] = 'Feedback in this course helped me learn. Please note, feedback can be either informal (e.g., in class discussion, chat boards, think-pair-share, office hour discussions, help sessions) or formal (e.g., written or clinical assessments, review of exams, peer reviews, clicker questions).'
questions[8] = 'The instructor fostered an effective learning environment.'
questions[9] = 'The instructor\'s teaching methods contributed to my learning.'
questions[10] = 'The instructor encouraged students to take responsibility for their own learning.'
questions[11] = 'Is this course required?'
questions[12] = 'Expected Grade in this Course'
questions[13] = 'Please provide any general comments about this course.'

questions_dict = { questions[idx]: 'Q' + str(idx) for idx in range(0, 13) }
questions_list = ['Q3', 'Q4', 'Q5', 'Q7', 'Q8', 'Q9']

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
    'Primary Faculty': 'Instructor',
    'Instructor Name (ID)': 'OtherInstructor',
    'Is TA': 'ISTA',
    'Question Code': 'QuestionNumber',
    'Question Description': 'Question',
    'Comment': 'Comment',
    'Text Response': 'Response',
    'No Measure Value': 'Extraneous'
}

# Define a mapping dictionary and update term
replacement_dict = {'Spring': '11', 'Summer': '21', 'Fall': '31'}
reversed_dict = {value: key for key, value in replacement_dict.items()}

# List of Engineering Departments and Units
dept_list = ['AERO', 'BAEN', 'BMEN', 'CHEN', 'CLEN', 'CSCE', 'CVEN', 'ECEN',
             'ETID', 'ISEN', 'MEEN', 'MSEN', 'MTDE', 'NUEN', 'OCEN', 'PETE']

output_columns = [
    'Dept', 'Code', 'Number', 'Section', 'Location', 'Term',
    'UIN', 'Instructor', 'OtherInstructor',
    'QuestionNumber', 'Question', 'Comment', 'Response'
    ]

########################################################################################################################


# List desired CSV files in the directory
college_id = 'EN'  # Engineering College ID
base_path = 'OIEE_Comments_Raw'
output_path = 'OIEE_Comments_Processed'

# Ensure output directory exists
Path(output_path).mkdir(parents=True, exist_ok=True)

csv_comments_files = list(Path(base_path).glob(f'OIEE-Comments-{college_id}-*.csv'))

# Loop through every COMMENTS input file into a dataframe, and append the dataframe to DataFrames list
for file in csv_comments_files:
    # Extract filename
    filename = file.name
    # Extract the term part: removes prefix and suffix
    term = filename.replace(f'OIEE-Comments-{college_id}-', '').replace('.csv', '')
    # Build output filename
    output_filename = f'OIEE-Comments-{college_id}-{term}_processed.csv'
    # Full output path
    output_file = Path(output_path) / output_filename
    print(f'Processing File: {file}')

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

    # Rename columns
    oiee_df.rename(columns=columns_dict, inplace=True)
    oiee_df['Comment'] = oiee_df['Comment'].apply(normalize_string)
    oiee_df['Response'] = oiee_df['Response'].apply(normalize_string)
    # Creating course 'Dept', 'Section', 'Number'
    oiee_df['Dept'] = oiee_df['Dept'].replace(dept_dict)
    oiee_df['Section'] = oiee_df['CourseNumber'].str.split().str[-1]
    oiee_df['Number'] = oiee_df['CourseNumber'].str.split().str[1]
    # Creating course 'Semester', 'Year', 'Location'
    oiee_df['Semester'] = oiee_df['CombinedTerm'].str.split().str[0]
    oiee_df['Year'] = oiee_df['CombinedTerm'].str.split().str[1]
    oiee_df['Location'] = oiee_df['CombinedTerm'].str.split('- ', n=1).str[1]
    # Extracting 'UIN', 'Instructor', and 'OtherInstructor'
    oiee_df['UIN'] = oiee_df['Instructor'].str.extract(r'\((\d+)\)\s*$')
    oiee_df['Instructor'] = oiee_df['Instructor'].str.replace(r'\s*\(\d+\)\s*$', '', regex=True)
    oiee_df['OtherInstructor'] = oiee_df['OtherInstructor'].str.replace(r'\s*\(\d+\)\s*$', '', regex=True)

    # Drop vestigial columns
    oiee_df.drop(columns=['Row', 'CombinedTerm', 'CourseNumber', 'Type', 'Extraneous'], inplace=True)
    # Create Term column
    oiee_df['Term'] = oiee_df['Year'].astype(str) + oiee_df['Semester'].replace(replacement_dict)
    oiee_df.drop(columns=['Year', 'Semester'], inplace=True)
    print(f'Shape of curated oiee_df: {oiee_df.shape}')

    # Prune rows
    oiee_df = oiee_df[oiee_df['ISTA'] == False]
    oiee_df.drop(columns=['ISTA'], inplace=True)
    print(f'Shape of oiee_df without TAs: {oiee_df.shape}')

    # Rename entries in the first two rows to match column names
    oiee_df['Question'] = oiee_df['Question'].apply(
        lambda x: normalize_string(x) if pd.notnull(x) else x
    )
    # print(questions_dict.keys())
    oiee_df = oiee_df[oiee_df['Question'].isin(questions_dict.keys())]
    print(f'Shape of oiee_df with institutional questions only: {oiee_df.shape}')
    oiee_df['QuestionNumber'] = oiee_df['Question'].replace(questions_dict)
    print(f'Shape of oiee_df after substitution: {oiee_df.shape}')
    oiee_df = oiee_df[oiee_df['QuestionNumber'].isin(questions_list)]
    print(f'Shape of oiee_df with SET questions only: {oiee_df.shape}')

    # Reorder columns
    oiee_df = oiee_df[output_columns]
    # print(oiee_df['QuestionNumber'].unique())
    # print(oiee_df.head())
    # print(oiee_df.columns)

    condition = oiee_df['Number'].apply(pd.to_numeric, errors='coerce').isin({291, 299, 381, 385, 399, 484, 485, 491, 681, 684, 685, 691})
    oiee_df = oiee_df[~condition]
    print(f"Shape of oiee_df without SEM's and INS's: {oiee_df.shape}")
    # EXCLUDE: WEB sections start with a seven.
    oiee_df = oiee_df[~oiee_df['Section'].astype(str).str.startswith('7')]
    print(f"Shape of oiee_df without WEB sections: {oiee_df.shape}")


    # Combine 'Comment' and 'Response' columns
    oiee_df['Comment'] = oiee_df['Comment'].where(oiee_df['Comment'].apply(lambda x: isinstance(x, str)), '')
    oiee_df['Response'] = oiee_df['Response'].where(oiee_df['Response'].apply(lambda x: isinstance(x, str)), '')
    oiee_df['OriginalComment'] = oiee_df['Comment'] + oiee_df['Response']
    oiee_df = oiee_df[oiee_df['OriginalComment'] != '']
    oiee_df['OriginalComment'] = oiee_df['OriginalComment'].apply(
        lambda x: normalize_string(x) if pd.notnull(x) else x
    )
    oiee_df.drop(columns=['Comment', 'Response'], inplace=True)
    print(f"Shape of oiee_df with combined string elements: {oiee_df.shape}")


    ####################################################################################################################

    # Apply the anonymization function
    if not oiee_df.empty:
        oiee_df['AnonymizedPrimary'] = oiee_df.apply(lambda row: anonymize_text(row['OriginalComment'], row.get('Instructor', '')), axis=1)
        oiee_df['AnonymizedSecondary'] = oiee_df.apply(lambda row: anonymize_text(row['AnonymizedPrimary'], row.get('OtherInstructor', '')), axis=1)
        oiee_df['Anonymized'] = oiee_df['AnonymizedSecondary'].str.replace('\n', ' ')
    oiee_df.drop(columns=['AnonymizedPrimary', 'AnonymizedSecondary'], inplace=True)
    if 'Anonymized' in oiee_df.columns:
        oiee_df['CharCount'] = oiee_df['Anonymized'].str.len()
    else:
        oiee_df['CharCount'] = 0
    oiee_df = oiee_df[oiee_df['CharCount'] > 0]

    oiee_df['OriginalComment'] = oiee_df['OriginalComment'].apply(lambda x: unidecode(x) if isinstance(x, str) else '')
    oiee_df['Anonymized'] = oiee_df['Anonymized'].apply(lambda x: unidecode(x) if isinstance(x, str) else '')

    print(f"Questions found in oiee_df: {oiee_df['QuestionNumber'].unique()}")

    # Display the final result
    print(f'Saving File: {output_file}')
    try:
        oiee_df.to_csv(output_file, index=False)
    except PermissionError:
        print(f"Error: Permission denied writing to '{output_file}'.")
    except Exception as e:
        print(f"Error writing '{output_file}': {e}")
