"""
Centralized configuration for the TAMU COE SET Reports pipeline.

This module contains all shared configuration values used across the data processing
scripts. Import from this module instead of hardcoding values in individual scripts.

Usage:
    from config import COLLEGE_ID, TERM_LIST, DEPT_LIST, DEPT_MAPPING
"""

###################################################################################################
# General Settings
###################################################################################################

COLLEGE_ID = 'EN'  # Engineering College ID
CURRENT_YEAR = '2026'

###################################################################################################
# Term Configuration
###################################################################################################

# Complete list of all available terms (Fall=31, Spring=11, Summer=21)
TERM_LIST_ALL = [
    '202031', '202111', '202121',
    '202131', '202211', '202221',
    '202231', '202311', '202321',
    '202331', '202411', '202421',
    '202431', '202511'
]

# Terms for new processing (Fall=31, Spring=11, Summer=21)
TERM_LIST_NEW = [
    '202331', '202411', '202421',
    '202431', '202511'
]

# Terms for departmental statistics computation (recent terms with reliable data)
TERM_LIST_STATS = [
    '202231', '202311',
    '202331', '202411',
    '202431', '202511'
]

# Terms for loading evaluation data in reports
TERM_LIST_DATA = [
    '202231', '202311', '202321',
    '202331', '202411', '202421',
    '202431', '202511', '202521',
    '202531'
]

# Terms for GENAI comment summaries
TERM_LIST_GENAI = [
    '202331', '202411', '202431', '202511', '202531'
]

# Terms to include instructors in reports
TERM_LIST_INSTRUCTORS = [
    '202331', '202411', '202431', '202511', '202531'
]

###################################################################################################
# Department Configuration
###################################################################################################

# Complete list of all Engineering departments
DEPT_LIST = [
    'AERO', 'BMEN', 'CHEN', 'CLEN', 'CSCE', 'CVEN', 'ECEN',
    'ETID', 'ISEN', 'MEEN', 'MSEN', 'MTDE', 'NUEN', 'OCEN', 'PETE'
]

# Department mapping from OIEE/HelioCampus format to abbreviations
DEPT_MAPPING = {
    'CS-Aerospace Engineering': 'AERO',
    'CS-Biological & Agric Engr': 'BAEN',
    'CS-Biomedical Engineering': 'BMEN',
    'CS-Chemical Engineering': 'CHEN',
    'CS-Civil Engineering': 'CLEN',
    'CS-Computer Sci & Engr': 'CSCE',
    'CS-Civil & Environmental Engr': 'CVEN',
    'CS-Elec & Computer Engr': 'ECEN',
    'CS-Engr Technology &Tic': 'ETID',
    'CS-Industrial & Sys Engr': 'ISEN',
    'CS-Mechanical Engineering': 'MEEN',
    'CS-Materials Sci & Engr': 'MSEN',
    'CS-Multidisciplinary Engr': 'MTDE',
    'CS-Nuclear Engineering': 'NUEN',
    'CS-Ocean Engineering': 'OCEN',
    'CS-Petroleum Engineering': 'PETE',
}

###################################################################################################
# Question Configuration
###################################################################################################

# Questions selected by the COE SET Task Force for evaluations
QUESTION_NUMBERS = [3, 4, 5, 7, 8, 9]

# Full question text for parsing raw AEFIS/OIEE files (index 0 is a dummy placeholder)
QUESTIONS = [
    '',  # Index 0: Dummy placeholder
    'Begin this course evaluation by reflecting on your own level of engagement and participation in the course. What portion of the class preparation activities (e.g., readings, online modules, videos) and assignments did you complete?',
    'Based on what the instructor(s) communicated, and the information provided in the course syllabus, I understood what was expected of me.',
    'This course helped me learn concepts or skills as stated in course objectives/outcomes.',
    'In this course, I engaged in critical thinking and/or problem solving.',
    'Please rate the organization of this course.',
    'In this course, I learned to critically evaluate diverse ideas and perspectives.',
    'Feedback in this course helped me learn. Please note, feedback can be either informal (e.g., in class discussion, chat boards, think-pair-share, office hour discussions, help sessions) or formal (e.g., written or clinical assessments, review of exams, peer reviews, clicker questions).',
    'The instructor fostered an effective learning environment.',
    'The instructor\'s teaching methods contributed to my learning.',
    'The instructor encouraged students to take responsibility for their own learning.',
    'Is this course required?',
    'Expected Grade in this Course',
    'Please provide any general comments about this course.',
]

# Maximum scale values for each question (index 0 is dummy)
QUESTIONS_MAX = [0, 4, 3, 4, 4, 4, 6, 6, 5, 3, 3, 2, 8, 0]

# Question text for report displays
QUESTION_TEXT = {
    3: 'This course helped me learn concepts or skills as stated in course objectives/outcomes.',
    4: 'In this course, I engaged in critical thinking and/or problem solving.',
    5: 'Please rate the organization of this course.',
    7: 'Feedback in this course helped me learn.',
    8: 'The instructor fostered an effective learning environment.',
    9: 'The instructor\'s teaching methods contributed to my learning.',
}

# Short synopses for visualizations
QUESTION_SYNOPSES = ['Q3 Concepts', 'Q4 Thinking', 'Q5 Organization',
                     'Q7 Feedback', 'Q8 Environment', 'Q9 Pedagogy', 'GPA']

# Multi-line labels for charts
QUESTION_LABELS = [
    'Q3\nConcepts\nand Skills',
    'Q4\nCritical\nThinking',
    'Q5\nCourse\nOrganization',
    'Q7\nInstructor\nFeedback',
    'Q8\nLearning\nEnvironment',
    'Q9\nPedagogy\nand Methods',
    'Overall\nCourse\nGPA'
]

###################################################################################################
# Path Configuration
###################################################################################################

PATHS = {
    # Raw input directories
    'argos_raw': 'ARGOS_Grades_Raw',
    'oiee_evaluations_raw': 'OIEE_Evaluations_Raw',
    'oiee_comments_raw': 'OIEE_Comments_Raw',
    'aefis_raw': 'AEFIS_Evaluations_Raw',

    # Processed/intermediate directories
    'argos_compressed': 'ARGOS_Grades_Compressed',
    'oiee_evaluations_processed': 'OIEE_Evaluations_Processed',
    'oiee_comments_processed': 'OIEE_Comments_Processed',
    'aefis_processed': 'AEFIS_Evaluations_Processed',
    'assessment_processed': 'ASSESSMENT_Processed',

    # Output directories
    'data_evaluations': 'DATA_Evaluations',
    'data_dept_stats': 'DATA_DeptStats',
    'data_reports': 'DATA_Reports',
    'genai': 'GENAI',
    'temp': 'TEMP',
}

###################################################################################################
# Course Level Mapping
###################################################################################################

LEVEL_DICT = {
    'one': '1XX',
    'two': '2XX',
    'three': '3XX',
    'four': '4XX',
    'grad': 'Grad'
}

# Course numbers to exclude (seminars, internships, special topics)
EXCLUDED_COURSE_NUMBERS = [291, 299, 381, 385, 481, 484, 485, 489, 491, 685, 691]

###################################################################################################
# Grade Configuration
###################################################################################################

GRADE_POINTS = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}

###################################################################################################
# Utility function for course level mapping
###################################################################################################

def map_course_level(number):
    """
    Map a course number to its level category.

    Args:
        number: Course number (int or str)

    Returns:
        str: Level category ('one', 'two', 'three', 'four', or 'grad')
    """
    try:
        num = int(number)
    except (ValueError, TypeError):
        return 'grad'  # Default for unparseable numbers

    if num < 200:
        return 'one'
    elif num < 300:
        return 'two'
    elif num < 400:
        return 'three'
    elif num < 500:
        return 'four'
    else:
        return 'grad'
