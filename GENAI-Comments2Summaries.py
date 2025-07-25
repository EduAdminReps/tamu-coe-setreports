"""
The main function of this script is to pre-process comments to implement hierarchical summarization.
That is, the script summarizes individual comments or paragraphs that are too long.
Some students write very lengthy comments, which can raise issues with the generative AI model.
First, the generative AI model has a limit on the length of the input text.
Importantly, when multi-head attention is applied to aggregated comments, it may give excessive weight to long comments.
It concatenates all the compressed comments corresponding to the same course and instructor.
To prevent these issue, the script summarizes the long comments into shorter summaries, using the generative AI model.
For every long string, it makes a generative AI call to summarize the comments.
If the string is reasonable, it is passed directly to the output.
However, if the string itself is too long, it is truncated to meet the generative AI model limit.
The call is anonymous because the FirstName and LastName were previously replaced with generic names.
The output files are in a format suitable for the application of GENAI-step2-group.
"""

from pathlib import Path
import pandas as pd


########################################################################################################################

def deanonymize_text(text, FullName):

    # Split FullName safely
    name_parts = FullName.strip().split(maxsplit=1)
    if len(name_parts) == 0:
        return text
    elif len(name_parts) == 1:
        FirstName = name_parts[0]
        LastName = ''
    else:
        FirstName = name_parts[0]
        LastName = name_parts[1]

    # Function to restore first name and last name separately
    if not isinstance(text, str):
        return ''
    gen_first_name = 'Jordan'
    gen_last_name = 'Taylor'
    if isinstance(FirstName, str) and isinstance(LastName, str):
        text = text.replace(gen_first_name, FirstName)
        text = text.replace(gen_last_name, LastName)
    return text

########################################################################################################################

import google.generativeai as genai
import google.api_core.exceptions

# Path to your key file
key_path = Path('GENAI_Credentials/genai_api_key.txt')
# Read the key
with key_path.open('r') as f:
    key = f.read().strip()
genai.configure(api_key=key)
model_name = 'gemini-1.5-flash'
model = genai.GenerativeModel(model_name)
lead_prompt = """
The text below contains feedback received through student evaluation.
Provide a concise summary in paragraph form of their comments, and try to avoid explicitly mentioning any names.
"""

def genai_single_summary(anonymized_text):
    """
    Generates a summary of the given text using the Gemini API.

    Args:
        anonymized_text (str): The text to be summarized.
        model: The configured GenerativeModel instance (e.g., genai.GenerativeModel('gemini-pro')).
        lead_prompt (str): The initial prompt to guide the summarization.

    Returns:
        str: The generated summary, 'BLOCKED' if content is blocked, or 'EXCEPTION' on other errors.
    """

    safety_config = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_ONLY_HIGH",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_ONLY_HIGH",
        }
    ]

    # generation_config = types.GenerateContentConfig(safety_settings=safety_settings)

    if len(str(anonymized_text)) > 0:
        try:
            response = model.generate_content(
                lead_prompt + anonymized_text,
                generation_config=genai.types.GenerationConfig(
                    # Only one candidate for now.
                    candidate_count=1,
                    temperature=1.0),
                safety_settings=safety_config
            )
            # Check for successful response
            gemini_summary = response.text

        except genai.types.BlockedPromptException as e:
            # This specific exception is for blocked content.
            print(f"Prompt blocked: {e.response.prompt_feedback}")
            gemini_summary = 'BLOCKED'
        except google.api_core.exceptions.ResourceExhausted as e:
            print(f"Resource exhausted error: {e}")
            gemini_summary = 'EXCEPTION'
        except google.api_core.exceptions.DeadlineExceeded as e:
            print(f"Deadline exceeded: {e}")
            gemini_summary = 'EXCEPTION'
        except google.api_core.exceptions.InternalServerError as e:
            print(f"Internal server error: {e}")
            gemini_summary = 'EXCEPTION'
        except Exception as e:
            # Catch all other unexpected errors
            print(f"An unexpected error occurred: {e}")
            # In case of an unexpected error where response.text might not exist
            # it's safer to set a default 'EXCEPTION' rather than trying response.text
            gemini_summary = 'EXCEPTION'

        print('Gemini Summary: ' + gemini_summary)
        print('Gemini Summary Length: ' + str(len(gemini_summary)))
        return gemini_summary
    else:
        return ''

########################################################################################################################

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

########################################################################################################################


# List desired CSV files in the directory
college_id = 'EN'  # Engineering College ID
base_path = 'OIEE_Comments_Processed'
output_path = 'GENAI'

term = '202511'
csv_files = [
    Path(base_path) / f'OIEE-Comments-{college_id}-{term}_processed.csv',
]
# csv_files = list(Path(base_path).glob(f'OIEE-Comments-{college_id}-{term}_processed.csv'))

for file in csv_files:
    file = Path(file)
    # Extract filename
    filename = file.name
    # Build output filename
    output_filename = f'GENAI-Summaries-{college_id}-{term}.csv'
    # Full output path
    output_file = Path(output_path) / output_filename
    print(f'Processing File: {file}')

    genai_df = pd.read_csv(file)
    print(f'Shape of original genai_df: {genai_df.shape}')

    # Create an empty list to store the processed results
    ai_summary = []
    ai_count = 0
    # Iterate through the DataFrame and make API calls with a delay
    for index, row in genai_df.iterrows():
        anonymized = row['Anonymized']
        if isinstance(anonymized, str):
            if len(anonymized) < 1000:
                ai_summary.append(anonymized)
            elif len(anonymized) < 5000:
                ai_response = genai_single_summary(anonymized)
                ai_summary.append(ai_response)
                ai_count += 1
                print('Excess')
            else:
                anonymized = anonymized[:5000]
                ai_response = genai_single_summary(anonymized)
                ai_summary.append(ai_response)
                ai_count += 1
                print('Truncation')
        else:
            print(f"Anonymized is not a string: {anonymized}")
            ai_summary.append('')
    # Create a new column 'Hierarchical' to store the processed results
    genai_df['Hierarchical'] = ai_summary
    print(genai_df.columns)
    # print(genai_df[['OtherInstructor', 'Anonymized', 'CharCount']].head())
    genai_df.drop(columns=['OtherInstructor', 'Anonymized', 'CharCount'], inplace=True)

    # Add a column to indicate if the prompt was blocked
    genai_df['BlockedPromptException'] = genai_df['Hierarchical'] == 'BLOCKED'
    print(f"Number of Blocked Prompts: {genai_df['BlockedPromptException'].sum()}")


    ####################################################################################################################

    group1_columns = [
        'Location', 'Term', 'Dept', 'Code', 'Number', 'Section', 'UIN', 'Instructor',
        'QuestionNumber', 'Question', 'BlockedPromptException'
    ]

    genai_df['OriginalComment'] = genai_df['OriginalComment'].astype(str)
    genai_df['Hierarchical'] = genai_df['Hierarchical'].astype(str)
    group1_df = genai_df.groupby(group1_columns).agg({
        'OriginalComment': lambda x: ' '.join(x),
        'Hierarchical': lambda x: 'BLOCKED' if (x == 'BLOCKED').any() else ' '.join(x)
    }).reset_index()

    group1_df['CharCount'] = group1_df['Hierarchical'].str.len()

    print('Number of Comments: ' + str(len(group1_df['Hierarchical'])))
    print('Max Char Count: ' + str(max(group1_df['CharCount'])))
    print('Number of Calls: ' + str(len(group1_df[group1_df['CharCount'] > 1200])))

    # Create an empty list to store the processed results
    ai_summary = []
    # Iterate through the DataFrame and make API calls with a delay
    for index, row in group1_df.iterrows():
        grouped = row['Hierarchical']
        if len(grouped) < 500:
            ai_summary.append('')
        elif len(grouped) < 48000:
            ai_response = genai_single_summary(grouped)
            ai_summary.append(ai_response)
            ai_count += 1
        else:
            grouped = grouped[:48000]
            ai_response = genai_single_summary(grouped)
            ai_summary.append(ai_response)
            ai_count += 1
    # Create a new column 'QuestionSummary' to store the processed results
    group1_df['QuestionSummary'] = ai_summary


    ####################################################################################################################

    group2_df = group1_df.copy()
    group2_df.drop(columns=['Section', 'QuestionNumber', 'Question'], inplace=True)

    group2_columns = [
        'Location', 'Term', 'Dept', 'Code', 'Number', 'UIN', 'Instructor',
        'BlockedPromptException'
    ]

    group2_df = group2_df.groupby(group2_columns).agg({
        'OriginalComment': lambda x: ' '.join(x),
        'Hierarchical': lambda x: 'BLOCKED' if (x == 'BLOCKED').any() else ' '.join(x)
    }).reset_index()

    group2_df['CharCount'] = group2_df['Hierarchical'].str.len()

    print('Number of valid comments: ' + str(len(group2_df[~group2_df['BlockedPromptException']]['Hierarchical'])))
    print('Max CharCount: ' + str(max(group2_df['CharCount'])))
    print('Number of calls: ' + str(len(group2_df)))

    # Create an empty list to store the processed results
    ai_summary = []
    # Iterate through the DataFrame and make API calls with a delay
    for index, row in group2_df.iterrows():
        analyzed = row['Hierarchical']
        if analyzed == 'BLOCKED':
            ai_summary.append('BLOCKED')
        elif len(analyzed) < 200:
            ai_summary.append('EXCEPTION - Insufficient feedback provided for meaningful summarization.')
        elif len(analyzed) < 240000:
            ai_response = genai_single_summary(analyzed)
            ai_summary.append(ai_response)
            ai_count += 1
        else:
            analyzed = analyzed[:240000]
            ai_response = genai_single_summary(analyzed)
            ai_summary.append(ai_response)
            ai_count += 1
    # Create a new column 'processed' to store the processed results
    group2_df['Gemini'] = ai_summary

    # Apply function map_course_level to create column 'Level'
    group2_df['Level'] = group2_df['Number'].apply(map_course_level)
    group2_df['Model'] = model_name

    for index, row in group2_df.iterrows():
        group2_df['Deanonymized'] = group2_df.apply(lambda row: deanonymize_text(row['Gemini'], row.get('Instructor')), axis=1)

    # Display the final result
    print(f"Number of AI calls: {ai_count}")
    print(f'Saving File: {output_file}')
    group2_df.to_csv(output_file, index=False)
