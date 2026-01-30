"""
This program is designed to generate PDF reports containing tabular data,
visualizations (bar charts, error bars, and scatter plots), and
contextual explanations for departmental and instructor-specific evaluations.
"""

from pathlib import Path as myPath
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ReportLab related import's
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, PageBreak, Paragraph, Table, TableStyle, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
styles = getSampleStyleSheet()
style = styles["Normal"]
from reportlab.graphics.shapes import *
from reportlab.platypus.flowables import Flowable, CondPageBreak
from reportlab.platypus import Image
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT


ai_style = ParagraphStyle(
    "AIStyle",
    parent=styles["Normal"],
    alignment=TA_LEFT,
    wordWrap="CJK",      # key: allows breaking inside long tokens
    splitLongWords=1,    # key: lets ReportLab split long “words”
    spaceAfter=6,
    keepWithNext = 0,
    keepTogether = 0,
)

###########################################################################################################
# Define functions
###########################################################################################################

def load_csv(file_path):
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None
    return pd.read_csv(file_path)

def fig2image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    fx, fy = fig.get_size_inches()
    return Image(buf, fx * inch, fy * inch)

class VerticalText(Flowable):
    """
    Rotates a text in a table cell.
    """

    def __init__(self, text):
        Flowable.__init__(self)
        self.text = text

    def draw(self):
        canvas = self.canv
        canvas.rotate(90)
        fs = canvas._fontsize
        canvas.translate(1, -fs/1.2)  # canvas._leading?
        canvas.drawString(0, 0, self.text)

    def wrap(self, aW, aH):
        canv = self.canv
        fn, fs = canv._fontname, canv._fontsize
        return canv._leading, 1 + canv.stringWidth(self.text, fn, fs)

def compute_weighted_score(df):
    """
    Compute the weighted average score for each row in the DataFrame using np.inner.
    Scale the vector averages by the question range before multiplying by counts.
    Args:
        df (DataFrame): Input DataFrame containing Q3 to Q9 count and average columns.
    Returns:
        Series: A pandas Series containing the computed weighted scores.
    """
    questions = ['Q3', 'Q4', 'Q5', 'Q7', 'Q8', 'Q9']
    scaling_factors = np.array([1/3, 1/3, 1/3, 1/5, 1/4, 1/2])
    counts = df[[f'{q}_count' for q in questions]].values
    averages = (df[[f'{q}_avg' for q in questions]].values - 1) * scaling_factors
    weighted_sum = np.sum(np.multiply(counts, averages), axis=1)
    total_counts = np.sum(counts, axis=1)
    out = np.divide(
        weighted_sum,
        total_counts,
        out=np.full_like(weighted_sum, np.nan, dtype=float),
        where=total_counts != 0
    )
    return out

###########################################################################################################
# This section contains text that goes into the report.
###########################################################################################################

Q3_text = 'This course helped me learn concepts or skills as stated in course objectives/outcomes.'
Q4_text = 'In this course, I engaged in critical thinking and/or problem solving.'
Q5_text = 'Please rate the organization of this course.'
Q7_text = 'Feedback in this course helped me learn. Please note, feedback can be either informal (e.g., in class discussion, chat boards, think-pair-share, office hour discussions, help sessions) or formal (e.g., written or clinical assessments, review of exams, peer reviews, clicker questions).'
Q8_text = 'The instructor fostered an effective learning environment.'
Q9_text = 'The instructor\'s teaching methods contributed to my learning.'

question_description = [
    'Question 3 (1-4): ' + Q3_text,
    'Question 4 (1-4): ' + Q4_text,
    'Question 5 (1-4): ' + Q5_text,
    'Question 7 (1-6): ' + Q7_text,
    'Question 8 (1-5): ' + Q8_text,
    'Question 9 (1-3): ' + Q9_text
]

question_synopses = [
    'Q3 Concepts',
    'Q4 Thinking',
    'Q5 Organization',
    'Q7 Feedback',
    'Q8 Environment',
    'Q9 Pedagogy',
    'GPA'
]

label_synopses = [
    'Q3\nConcepts\nand Skills',
    'Q4\nCritical\nThinking',
    'Q5\nCourse\nOrganization',
    'Q7\nInstructor\nFeedback',
    'Q8\nLearning\nEnvironment',
    'Q9\nPedagogy\nand Methods',
    'Overall\nCourse\nGPA'
]

eval_text = ('University Guidelines: Complete longitudinal summaries (chronological and in tabular form) \
            of the student evaluations must be presented, with numerical data set in the context of \
            departmental standards and norms. [...] The department must provide these data to the candidates and \
            include these in the department report (candidates do not have access to departmental data) to \
            allow them to address the trends within their impact statement.')

# COE Guidelines: In the comprehensive evaluation of faculty teaching performance, \
# it is crucial to look beyond student feedback and incorporate a multifaceted \
# assessment approach. This may include peer evaluations, which offer insights into \
# teaching methodologies and professional competence from the perspective of colleagues. \
# Additionally, a thorough review of pedagogical materials; such as syllabi, assignments, \
# and examinations; may provide a concrete measure of alignment with educational \
# objectives and standards. Engagement in pedagogical forums should also be promoted, \
# as it reflects a faculty member\'s commitment to ongoing professional development \
# and their active participation in the broader educational community. Such metrics ensure \
# a balanced and in-depth appraisal of teaching effectiveness, recognizing the importance of \
# continuous improvement and innovation in educational practices.'

set_text = 'The questions listed below were identified by the College of Engineering \
            Student Evaluation of Teaching (SET) task force as pertinent for \
            faculty annual reviews, as well as promotion and tenure evaluations.'
# 'While some members desired a single number to represent teaching effectiveness, the group collectively agreed that this was not a desirable option.'

info_text = ('Readers interested in the methodology underlying these reports, including data sources, preprocessing, \
            visualization design, and the AI-assisted summarization pipeline (anonymization, hierarchical \
            aggregation, and exception-handling safeguards), may consult the accompanying arXiv manuscript \
            <i>Teaching at Scale: Leveraging AI to Evaluate and Elevate Engineering Education</i> \
            (<a href="https://arxiv.org/abs/2508.02731">arXiv:2508.02731</a>).')
note_text = 'Reports are based on student evaluation data provided \
            by the TAMU Office of Institutional Effectiveness and Evaluation (OIEE). \
            Grade distributions and class sizes are sourced from ARGOS. \
            Information is collected for primary instructors and traditional (TR) course \
            delivery at the College Station campus. \
            Partial student responses with missing fields may be excluded. \
            Courses without GPA data may be omitted. \
            Cases where the instructor of record differs between OIEE and ARGOS may not be included. \
            Departmental averages are calculated based on all courses at the same level within the unit.'
# Some comments from students raise AI exceptions for harassment and/or hate speech.

###########################################################################################################

level_dict = {'one': '1XX', 'two': '2XX', 'three': '3XX', 'four': '4XX', 'grad': 'Grad'}

current_year = '2026'
# Data terms
term_data_list = [
    # '202031', '202111', '202121',
    # '202131', '202211', '202221',
    '202231', '202311', '202321',
    '202331', '202411', '202421',
    '202431', '202511', '202521',
    '202531'
]
# GENAI terms
term_genai_list = [
    '202331', '202411', '202431', '202511', '202531'
]

# This list determines which instructors will be included in the report.
term_instructor_list = ['202331', '202411', '202431', '202511', '202531']

college_id = 'EN'  # Engineering College ID
# dept_list = ['AERO', 'BMEN', 'CHEN', 'CLEN', 'CSCE', 'CVEN', 'ECEN',
#              'ETID', 'ISEN', 'MEEN', 'MSEN', 'MTDE', 'NUEN', 'OCEN', 'PETE']
# AERO, BMENCHEN, CSCE, CVEN, ISEN, MTDE, NUEN, OCEN, PETE
# dept_list = ['BMEN']

# Questions to process
question_num = [3, 4, 5, 7, 8, 9]

# FLAGS
ai_flag = True

# Initialize an empty list to store DataFrames
dataframes = []
genaiframes = []
# Build one DataFrame
for term_data in term_data_list:
    # Load ASSESSMENT and ARGOS files
    base_path = f'DATA_Evaluations/'
    data_file = myPath(base_path + f'/Data-{college_id}-{term_data}.csv')
    # Load the file into a DataFrame
    data_file_df = load_csv(data_file)
    # df_argos_file = pd.read_csv(argos_file)
    if data_file_df is not None:
        dataframes.append(data_file_df)  # Append to the list if successfully loaded
    else:
        continue

for term_genai in term_genai_list:
    # Load GENAI files
    genai_base_path = f'GENAI'
    genai_file = myPath(genai_base_path + f'/GENAI-Summaries-{college_id}-{term_genai}.csv')
    # Load the file into a DataFrame
    genai_file_df = load_csv(genai_file)
    if genai_file_df is not None:
        genaiframes.append(genai_file_df)  # Append to the list if successfully loaded
    else:
        continue

# Concatenate ASSESSMENT and ARGOS DataFrames into one
if dataframes:
    dataframes = [df for df in dataframes if not df.empty]
    argos_df = pd.concat(dataframes, ignore_index=True)
    argos_df['Term'] = argos_df['Term'].astype(str)
    print(f"Combined ASSESSMENT and ARGOS DataFrame has {len(argos_df)} rows.")
else:
    print("No valid ASSESSMENT and ARGOS data files found.")

# Concatenate GENAI DataFrames into one
if genaiframes:
    genaiframes = [df for df in genaiframes if not df.empty]
    genai_df = pd.concat(genaiframes, ignore_index=True)
    genai_df['Term'] = genai_df['Term'].astype(str)
    print(f"Combined GENAI DataFrame has {len(genai_df)} rows.")
else:
    print("No valid GENAI data files found.")


# Compute the Weighted Score
argos_df['Weighted_Score'] = compute_weighted_score(argos_df)

# Normalize terms for colormap scaling
term_normalizer = mcolors.Normalize(vmin=min(argos_df['Term'].astype(int)),
                                    vmax=max(argos_df['Term'].astype(int)))

quantile_path = f'DATA_DeptStats'
quantiles_file = myPath(quantile_path + f'/{college_id}-quantiles.csv')
quantiles_df = load_csv(quantiles_file)
dept_stats_file = myPath(quantile_path + f'/{college_id}-departmental-statistics.csv')
dept_stats_df = load_csv(dept_stats_file)

# Find the minimum and maximum values
min_year = argos_df['Year'].min()
max_year = argos_df['Year'].max()
delta_year = max_year - min_year
print(f'Number of years covered: {delta_year + 1}')

decimal_columns = ['Q3_avg', 'Q4_avg', 'Q5_avg', 'Q7_avg', 'Q8_avg', 'Q9_avg', 'GPA']
numeric_columns = ['Enrollment', 'Responses'] + decimal_columns
course_columns = ['Level', 'Code', 'Number'] + numeric_columns
semester_columns = ['Dept', 'Term'] + course_columns

# Stylized instructor column names for the PDF report
report_columns = ['Course', 'Term'] + numeric_columns
report_columns_stylized = ['Course', 'Term'] + [VerticalText(i) for i in ['Enrollment', 'Responses'] + question_synopses]


# The Promotion DataFrame
# To get individual reports
# Needs work
faculty_flag = True
if faculty_flag:
    faculty_uin = [000000000]
    faculty_report = SimpleDocTemplate(f'TEMP/{str(faculty_uin[0])}.pdf', pagesize=letter)
    faculty_elements = []

if faculty_flag:
    dept_list = argos_df.loc[argos_df['UIN'].isin(faculty_uin)]['Dept'].unique().tolist()
# else:
#     dept_list = argos_df['Dept'].unique().tolist()

static_elements = []
# REPORT: Add a section title
styles = getSampleStyleSheet()
section_title = Paragraph(f'<font size=14>Student Evaluation of Teaching</font>', styles["Heading1"])
static_elements.append(section_title)
static_elements.append(Paragraph(eval_text))
static_elements.append(Paragraph('<font size=12>SET Task Force Recommendations</font>', styles["Heading2"]))
static_elements.append(Paragraph(set_text))

static_elements.append(Paragraph('<font size=10>Questions</font>', styles["Heading3"]))
list_flowable = ListFlowable(
    [
        ListItem(Paragraph(item, style=getSampleStyleSheet()["Normal"]), bulletColor="black")
        for item in question_description
    ],
    bulletType="bullet",
    leftIndent=10,
    spaceBefore=2,
    spaceAfter=2,
)
static_elements.append(list_flowable)
static_elements.append(Paragraph('<font size=12>Technical Description</font>', styles["Heading2"]))
static_elements.append(Paragraph(info_text))
static_elements.append(Paragraph('<font size=12>Disclaimer</font>', styles["Heading2"]))
static_elements.append(Paragraph(note_text))
static_elements.append(PageBreak())

for dept in dept_list:
    print(f'Proceeding with department: {dept}.')
    # DEPARTMENT REPORT: Create a PDF report canvas for the department
    dept_report = SimpleDocTemplate(f'DATA_Reports/{dept}-{current_year}-Teaching-Report.pdf', pagesize=letter)
    dept_elements = []

    if faculty_flag:
        UIN_list = faculty_uin
    else:
        UIN_list = argos_df.loc[(argos_df['Dept'] == dept) & (argos_df['Term'].isin(term_instructor_list))]['UIN'].unique()
        # UIN_list = UIN_list[:5]
    print(f'The number of instructors in {dept} is {len(UIN_list)}.')
    # Iterate through the instructors
    for UIN in UIN_list:
        instructor_argos_df = argos_df.loc[argos_df['UIN'] == UIN].copy()
        instructor = instructor_argos_df.iloc[0]['Instructor']
        individual_email = str(instructor_argos_df.iloc[0].get('Email', '') or '').strip()
        email_name = individual_email.split('@')[0] if '@' in individual_email else f"UIN_{UIN}"

        try:
            instructor_genai_df = genai_df.loc[genai_df['UIN'] == UIN].copy() # COMMENTS
            instructor_genai_df['Course'] = instructor_genai_df['Code'] + ' ' + instructor_genai_df['Number'].astype(str)
        except NameError:
            print('No GENAI data available for this report.')
            instructor_genai_df = pd.DataFrame()
        except KeyError as e:
            print(f"Missing expected column: {e}")
            instructor_genai_df = pd.DataFrame()

        print(f"Processing {dept} instructor: {instructor}")

        # INDIVIDUAL REPORT: Create a PDF report canvas for given instructor
        out_dir = myPath("DATA_Reports") / dept
        out_dir.mkdir(parents=True, exist_ok=True)

        individual_path = out_dir / f'{individual_email} ({dept}) {current_year}.pdf'
        individual_report = SimpleDocTemplate(str(individual_path), pagesize=letter)
        individual_elements = []

        df_instructor_format = instructor_argos_df.copy()
        df_instructor_format[decimal_columns] = df_instructor_format[decimal_columns].round(2)
        df_instructor_format[['Enrollment', 'Responses', 'InstructorSCH']] = df_instructor_format[['Enrollment', 'Responses', 'InstructorSCH']].astype(int)
        # Add the dataframe to the report after formatting the columns
        df_instructor_format['Course'] = df_instructor_format['Code'] + ' ' + df_instructor_format['Number'].astype(str)

        for level in ['one', 'two', 'three', 'four', 'grad']:
            # course_list = sorted(set(df_instructor_format[df_instructor_format['Level'] == level]['Course'].tolist()))
            course_list = sorted(
                set(
                    df_instructor_format[
                        (df_instructor_format['Level'] == level) &
                        (df_instructor_format['Dept'] == dept)
                        ]['Course'].tolist()
                )
            )
            if course_list:  # Check if the list is not empty (most Pythonic way)
                print(f"Level {level}: {course_list}")
            subtitle = 'Course Level ' + level_dict[level]
            # Creating the AI text for the report
            ai_text = ''
            ai_list = []

            dark_line_positions = []
            # One DataFrame is to build the report, and the other DataFrame is to draw the graph.
            df_instructor_report = pd.DataFrame(columns=report_columns)
            df_instructor_graph = pd.DataFrame(columns=report_columns)
            for course in course_list:
                level = df_instructor_format[df_instructor_format['Course'] == course]['Level'].iloc[0]
                # for term in set(df_instructor_format.loc[df_instructor_format['Course'] == course, 'Term'].tolist()):
                df_instructor_pruned = df_instructor_format.loc[(df_instructor_format['Course'] == course)]
                df_instructor_pruned = df_instructor_pruned.sort_values(by='Term', ascending=True)
                df_compare = dept_stats_df.loc[dept_stats_df['Dept'] == dept]
                if not df_compare.empty:
                    level_avg_row = df_compare.loc[df_compare['Level'] == level].copy()
                    level_avg_row[decimal_columns] = level_avg_row[decimal_columns].round(2)
                    level_avg_row['Course'] = str(dept) + ' ' + level_dict[level]
                    level_avg_row['Term'] = 'all'
                    level_avg_row['Enrollment'] = ''
                    level_avg_row['Responses'] = ''
                    level_stddev_row = level_avg_row[[f'Q{i}_stddev' for i in question_num] + ['STD']].copy()
                    level_avg_row = level_avg_row[report_columns]
                    if df_instructor_graph.empty:
                        df_instructor_graph = df_instructor_pruned[report_columns].copy()
                    else:
                        df_instructor_graph = pd.concat([df_instructor_graph, df_instructor_pruned[report_columns]], ignore_index=True)
                    if df_instructor_report.empty:
                        df_instructor_report = df_instructor_pruned[report_columns].copy()
                    else:
                        df_instructor_report = pd.concat([df_instructor_report, df_instructor_pruned[report_columns]], ignore_index=True)

                try:
                    instructor_genai_df
                    instructor_text_df = instructor_genai_df.loc[instructor_genai_df['Course'] == course] # Comments
                except NameError:
                    print('No GENAI data available for this report.')
                    instructor_text_df = pd.DataFrame()
                if not instructor_text_df.empty:
                    for index, row in instructor_text_df.iterrows():
                        summary_text = str(row['Deanonymized']) if pd.notna(row['Deanonymized']) else ''
                        if summary_text:
                            ai_list.append(f"<b>{course}</b> ({row['Term']}): {summary_text}")
            if ai_list:
                for ai_text in ai_list:
                    print(ai_text)

            # DataFrame: Concatenate avg row to instructor report dataframe
            if df_instructor_report.empty:
                continue
            else:
                df_instructor_report = pd.concat([df_instructor_report, level_avg_row], ignore_index=True)
            dark_line_positions.append(len(df_instructor_report))

            ###########################################################################################################

            # Add the DataFrame to the report
            # table_data = [df_instructor.columns.tolist()] + df_instructor.values.tolist()
            table_data = [report_columns_stylized] + df_instructor_report.values.tolist()
            column_widths = [60] + [60] + [30] * (len(df_instructor_format.columns) - 2)
            table = Table(table_data, colWidths=column_widths, spaceBefore=20, spaceAfter=20)
            style = TableStyle([('ROWPADDING', (0, 0), (-1, -1), 1),
                                ('TEXTCOLOR', (0, 0), (-1, 0), (0, 0, 0)),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                # ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
                                ('FONTSIZE', (0, 0), (-1, 0), 8),
                                ('BACKGROUND', (0, 1), (-1, -1), (1, 1, 1)),
                                ('TEXTCOLOR', (0, 1), (-1, -1), (0, 0, 0)),
                                # ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                                ('FONTSIZE', (0, 1), (-1, -1), 8),
                                ('GRID', (0, 1), (-1, -1), 1, (0.24, 0.22, 0.20))])

            for position in dark_line_positions:
                style.add('LINEBELOW', (0, position), (-1, position), 1, (0, 0, 0))
                # Add gray background to the specified row
                style.add('BACKGROUND', (0, position), (-1, position), (0.8, 0.8, 0.8))

            table.setStyle(style)
            if df_instructor_report.empty:
                pass
            else:
                # INDIVIDUAL REPORT: Add a subsection title with course level
                subsubsection_title = Paragraph(f'<font size=12>{instructor} ({dept}) - {subtitle}</font>', styles["Heading2"])
                individual_elements.append(subsubsection_title)

                ###########################################################################################################

                # Create a line plot
                max_range = np.array([4.0, 4.0, 4.0, 6.0, 5.0, 3.0, 4.0])
                # Plot data
                my_simple_fig, ax1 = plt.subplots(figsize=(6.25, 3.85))

                ax1.set(xticks=[1, 2, 3, 4, 5, 6, 7], xticklabels=label_synopses)
                ax1.set_ylabel('Raw Scores', fontsize=8)
                ax1.set_title('Mean Scores and GPA', fontsize=8)

                # Set the ranges for x-axis and y-axis
                ax1.set(xlim = (0.5, 7.5))  # Set x-axis range from 0 to 6
                ax1.set(ylim = (0, 6.5))  # Set y-axis range from 0 to 12

                # Add grid to ax1
                ax1.grid(True, which='major', linestyle=':', linewidth=0.5, color='lightgray', zorder=2)  # Major gridlines
                ax1.minorticks_on()  # Enable minor ticks
                ax1.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray', zorder=2)  # Minor gridlines

                quantiles_row = quantiles_df.loc[(quantiles_df['Level'] == level) & (quantiles_df['Dept'] == dept)]
                np_q25 = quantiles_row[['Q3_q25', 'Q4_q25', 'Q5_q25', 'Q7_q25', 'Q8_q25', 'Q9_q25', 'GPA_q25']].to_numpy(dtype=np.float64).flatten()
                np_q75 = quantiles_row[['Q3_q75', 'Q4_q75', 'Q5_q75', 'Q7_q75', 'Q8_q75', 'Q9_q75', 'GPA_q75']].to_numpy(dtype=np.float64).flatten()

                df_plot = df_instructor_graph[['Term'] + decimal_columns]
                df_mean = level_avg_row[decimal_columns]
                np_mean = df_mean.to_numpy(dtype=np.float64).flatten()
                # np_stddev1 = level_stddev_row.to_numpy(dtype=np.float64).flatten()
                # np_confidence = (1.96 * np_stddev1)/8.0

                norm = mcolors.Normalize(vmin=0, vmax=1)
                # cmap_name = 'Blues'
                cmap = plt.get_cmap('Blues')
                cmap2 = plt.get_cmap('Reds')

                for row in df_plot.to_numpy(dtype=np.float64):
                    row_shift = (3 * (((row[0] // 100) - min_year) % 10) + ((row[0] % 100) // 10) - 1) / (2 + 3 * delta_year)
                    x_values = np.array([1, 2, 3, 4, 5, 6, 7]) + 0.8 * (row_shift - 0.5)
                    row_colors = cmap(norm(row_shift * np.ones(6)))
                    row_color2 = cmap2(norm(row_shift))
                    row_data = row[-7:-1]
                    ax1.bar([1, 2, 3, 4, 5, 6, 7], max_range, color='none', edgecolor='dimgray', linewidth=1.5, zorder=1)
                    ax1.bar([1, 2, 3, 4, 5, 6, 7], np_q75, color='lightgray', edgecolor='gray', linewidth=1, zorder=1)
                    ax1.bar([1, 2, 3, 4, 5, 6, 7], np_q25, color='white', edgecolor='gray', linewidth=1, zorder=1)
                    ax1.bar([1, 2, 3, 4, 5, 6, 7], np_mean, color='none', edgecolor='black', linewidth=1.5, zorder=3)
                    ax1.bar([1, 2, 3, 4, 5, 6, 7], max_range, color='none', edgecolor='dimgray', linewidth=1.5, zorder=3)
                    ax1.scatter(x_values[:-1], row_data, s=50, c=row_colors, edgecolors='black', linewidths=1.0, zorder=4)
                    ax1.scatter(x_values[-1], row[-1], s=50, color=row_color2, edgecolors='black', linewidths=1.0, zorder=4)
                ax1.tick_params(axis='both', labelsize=7)

                # Legend 1: Bar and Line Meaning
                legend_elements = [
                    Line2D([0], [0], color='dimgray', linewidth=1.5, label='Max'),
                    Line2D([0], [0], color="black", linewidth=1.5, label='Mean'),
                    Line2D([0], [0], color='gray', linewidth=1, label='Quantiles')
                ]

                legend1 = ax1.legend(
                    handles=legend_elements,
                    fontsize=6,
                    loc='upper left',
                    title="Legend",
                    title_fontsize=7,
                    frameon=True
                )
                # Tell matplotlib this legend stays even when another legend is added
                ax1.add_artist(legend1)

                # Legend 2: Years and Color Scale
                years = list(range(int(max_year), int(min_year) - 1, -1))
                year_handles = []
                for y in years:
                    semester = y*100 + 31  # use 31 as the Fall semester
                    t = (3 * (((semester // 100) - min_year) % 10) + ((semester % 100) // 10) - 1) / (2 + 3 * delta_year)
                    year_handles.append(
                        Line2D([0], [0], marker='o', linestyle='None',
                               markersize=6,
                               markerfacecolor=cmap(norm(t)),
                               markeredgecolor='black',
                               markeredgewidth=0.8,
                               label=str(y)
                               )
                    )

                legen2 = ax1.legend(
                    handles=year_handles,
                    fontsize=6,
                    loc='upper right',
                    title='Years',
                    title_fontsize=7, frameon=True
                )

                # Adjust layout to prevent clipping
                plt.tight_layout()

                individual_elements.append(Paragraph('<font size=10>Visualization</font>', styles["Heading3"]))
                visualization_text =\
                    (f'This figure presents average scores for level-{level} courses, with student evaluations shown in blue \
                     and GPA in red. Departmental empirical quantiles (25th and 75th percentiles) are shown in gray. \
                     Each bar represents the progression over time, moving from left to right with a transition from lighter to darker shades.')
                individual_elements.append(Paragraph(visualization_text, styles["Normal"]))
                individual_elements.append(fig2image(my_simple_fig))
                plt.close(my_simple_fig)

                ###########################################################################################################

                # Add the DataFrame to the individual report
                scaling_factors_wp = np.array([1 / 3, 1 / 3, 1 / 3, 1 / 5, 1 / 4, 1 / 2])
                np_mean_wp = np.dot(np_mean[:-1] - 1, scaling_factors_wp) / 6
                np_q25_wp = np.dot(np_q25[:-1] - 1, scaling_factors_wp) / 6
                np_q75_wp = np.dot(np_q75[:-1] - 1, scaling_factors_wp) / 6
                # print([np_q25_wp, np_mean_wp, np_q75_wp])

                ###########################################################################################################

                level_df = argos_df[(argos_df['Level'] == level) & (argos_df['Dept'] == dept)]
                instructor_df = level_df[level_df['UIN'] == UIN]

                # Plot data
                my_other_fig, ax_main = plt.subplots(figsize=(4.58, 3))

                cmap1 = plt.colormaps.get_cmap('Grays')
                # Plot all data points in gray
                ax_main.scatter(
                    level_df['Weighted_Score'],
                    level_df['InstructorSCH'],
                    s=level_df['InstructorSCH'],
                    color='lightgray', edgecolors='gray', alpha=1.0, zorder=3
                )
                ax_main.hlines(
                    y=[300,600,900],  # y positions
                    xmin=-0.025, xmax=1.025,
                    color='lightgray', linewidth=1, zorder=2
                )
                ax_main.set_xlim(-0.025, 1.025)
                ax_main.set_ylim(0, 1000)
                # Plot quantiles
                ax_main.vlines(
                    x=[np_q25_wp, np_q75_wp],  # x positions
                    ymin=0, ymax=1000,
                    color='darkgray', linewidth=1, linestyle='dashed', zorder=4
                )
                # Plot mean
                ax_main.vlines(
                    x=[np_mean_wp],  # x positions
                    ymin=0, ymax=1000,
                    color='black', linewidth=1.5, linestyle='dashed', zorder=4
                )

                # Highlight the specific instructor
                cmap2 = plt.colormaps.get_cmap('Blues')
                if not instructor_df.empty:
                    for _, row in instructor_df.iterrows():
                        term_intensity = term_normalizer(int(row['Term']))
                        highlight_color = cmap2(term_intensity)
                        ax_main.scatter(
                            row['Weighted_Score'],
                            row['InstructorSCH'],
                            s=row['InstructorSCH'],
                            color=highlight_color, edgecolors='k', alpha=1.0, zorder=5
                        )
                        # Update colors for histograms
                        colors = [highlight_color if x == row['Weighted_Score'] else 'gray' for x in
                                  level_df['Weighted_Score']]

                # Set axis labels and grid for the main plot
                ax_main.set_xlabel('Weighted Score', fontsize=8)
                ax_main.set_ylabel('Instructor SCH', fontsize=8)

                # Add grid to ax1
                ax_main.grid(True, which='major', linestyle=':', linewidth=0.5, color='lightgray', zorder=2)  # Major gridlines
                ax_main.minorticks_on()  # Enable minor ticks
                ax_main.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray', zorder=2)  # Minor gridlines

                # Add title
                ax_main.set_title(f'Instructor SCH vs Weighted Scores for level {level}', fontsize=8)
                # Create custom legend handles
                legend_handles = [
                    Patch(facecolor='lightgray', edgecolor='gray', label='All Instructors'),  # Gray for all instructors
                    Patch(facecolor=cmap2(0.6), edgecolor='black', label=instructor)
                ]

                # Add legend
                ax_main.legend(handles=legend_handles, loc='upper left', fontsize=6, title="Legend", title_fontsize=7)
                ax_main.tick_params(axis='both', labelsize=7)

                plt.tight_layout()

                individual_elements.append(fig2image(my_other_fig))
                plt.close(my_other_fig)
                individual_elements.append(Paragraph(
                    'The weighted score refers to a normalized average of student evaluations. \
                    The size and vertical position of each marker corresponds to effective student credit hours (SCH). \
                    Empirical mean and quantiles (25th and 75th percentiles) for student scores are represented by vertical lines.', styles["Normal"]))

                ###########################################################################################################

                # Ensure there's room for heading + short text + at least some rows of the table
                individual_elements.append(CondPageBreak(200))  # tune 160-300

                individual_elements.append(Paragraph('<font size=10>Raw Data</font>', styles["Heading3"]))
                raw_text =\
                    (f'The raw data is displayed in a condensed format. \
                    Sections taught by the same instructor for a given course within the same semester are grouped together. \
                    Departmental averages for all level-{level} courses across semesters are shown in gray.')
                individual_elements.append(Paragraph(raw_text, styles["Normal"]))
                individual_elements.append(table)

                if ai_flag:
                    # First, filter the list to get only the summaries we want to show.
                    valid_ai_summaries = [s for s in ai_list if len(s) > 20]

                    if valid_ai_summaries:
                        # Before starting the AI section, ensure there's enough room for
                        # the title and at least a few lines of text. If not, start a new page.
                        individual_elements.append(CondPageBreak(100))

                        # Add the title for the entire AI summary section
                        subsubsection_title = Paragraph(f"<font size=10>{subtitle} - AI Summary</font>",
                                                        styles["Heading3"])
                        individual_elements.append(subsubsection_title)

                        # Combine all individual summaries into ONE large string.
                        # The <br/><br/> tag creates a nice visual separation (a blank line)
                        # between each summary within the single paragraph block.
                        full_ai_text = "<br/><br/>".join(valid_ai_summaries)

                        # Now, add this single, large Paragraph object. ReportLab will
                        # automatically handle splitting it across pages if it's too long.
                        individual_elements.append(Paragraph(full_ai_text, ai_style))

                # individual_elements.append(CondPageBreak(60))
                individual_elements.append(PageBreak())
                # if ai_flag:
                #     if any(len(s) > 20 for s in ai_list):
                #         individual_elements.append(CondPageBreak(150))
                #         subsubsection_title = Paragraph(f"<font size=10>{subtitle} - AI Summary</font>",
                #                                         styles["Heading3"])
                #         individual_elements.append(subsubsection_title)
                #     for ai_text in ai_list:
                #         if len(ai_text) > 20:
                #             individual_elements.append(Paragraph(ai_text, ai_style))
                #
                # individual_elements.append(CondPageBreak(60))
                # individual_elements.append(PageBreak())

        dept_elements = dept_elements + individual_elements
        # Build individual reports
        individual_report.build(static_elements + individual_elements)

    # Build department report
    if faculty_flag:
        faculty_elements = faculty_elements + dept_elements
    else:
        dept_report.build(static_elements + dept_elements)

if faculty_flag:
    faculty_report.build(static_elements + faculty_elements)

