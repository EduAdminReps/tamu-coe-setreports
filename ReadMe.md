# Student Evaluation of Teaching Processing Pipeline

This project is intended for College of Engineering faculty and staff with access to **AEFIS**, **OIEE**, **ARGOS**, and **GENAI API** credentials.  
**Basic Python knowledge and secure handling of sensitive data is assumed.**

---

> **WARNING**  
> Do **not** upload or share any raw or processed data files containing grades, instructor names, student comments, or personal information to public repositories or outside secure systems.  
> This code is intended for secure, internal use only.

---

## Workflow Overview

### **STEP 0: Get Raw Data Files**

**Option 1: HelioCampus (AEFIS) - [https://tamu.aefis.net](https://tamu.aefis.net)**
- Set the following AEFIS parameters:
    - **Term:** Fall 2024 - College Station
    - **Institution:** TAMU [TAMU]
    - **College:** Engineering [EN]
    - **Department:** CS-Unit Engineering [CS-DEPT]
    - **Instructor Type:** Instructor
    - **Course Section:** No Courses for Selected Faculty
    - **Question Type:** Multi-Choice, Single Answer [MCSA], Multi-Choice, Multi Answer [MCMA], and others
    - **Question:** No Question
    - **Show Comments:** No
- **Note:** If the request is too large, split MCSA and INSTR queries.

**Option 2: OIEE - [HelioCampus Dashboard Access](https://assessment.tamu.edu/student-course-evaluations/sce-liaisons/heliocampus-dashboard-access)**
- Access: Student Course Evaluations → HelioCampus Dashboard Access
- Steps:
    1. Connect via VPN if needed
    2. Download Overall Report → Crosstab → CSV
    3. Prune extraneous columns (additional questions)

---

### **Step 1: Process Raw Data Files**

#### **a. Use AEFIS Data**
- **Option 1a:** Run `AEFIS-Evaluations-Raw2Processed.py`  
  - **Input:** Files in `AEFIS_Evaluations_Raw`
  - **Output:** Files in `AEFIS_Evaluations_Processed`
- **Option 1b:** Run `AEFIS-Evaluations-Raw2Processed-split.py` (process files individually)
  - **Input:** Files in `AEFIS_Evaluations_Raw`
  - **Output:** Files in `AEFIS_Evaluations_Processed`
- *Note: The directory structure dictates which files get processed.*

#### **b. Use OIEE Data**
- Run `OIEE-Evaluations-Raw2Processed.py`
  - **Input:** Files in `OIEE_Evaluations_Raw`
  - **Output:** Files in `OIEE_Evaluations_Processed`

---

### **Step 3: Combine and Copy Data**

- Run `ASSESSMENT-CombineSources.py`
    - **Potential Inputs:**  
      - Files in `AEFIS_Evaluations_Processed`
      - Files in `OIEE_Evaluations_Processed`
    - **Output:** Files in `ASSESSMENT_Processed`
- *Note: Both potential inputs have their own issues—choose the best available.*

---

### **Step 4: Get and Process Grades**

1. Launch a VOAL virtual machine (see TAMU bookmarks)
2. Open Howdy on Chrome in the VM
3. Under the 'Employee' tab, launch ARGOS
4. Under COE DataBlocks, select `TR_ARGOS_Grades_Raw` (Teaching Reports)
5. Run `ARGOS-Grades-Raw2Compressed.py`
    - **Input:** Files in `ARGOS_Grades_Raw`
    - **Output:** Files in `ARGOS_Grades_Compressed`

---

### **Step 5: Integrate Student Scores, Grades, and Compute Departmental Statistics**

- Run `DATA-Assessment2Evaluations.py`
- Run `DATA-Evaluations2DeptStats.py`

---

### **Step Generative AI:**
(Optional, for Comment Summarization with GenAI)

1. Run `OIEE-Comments-Raw2Processed.py`
    - **Input:** Files in `OIEE_Comments_Raw`
    - **Output:** Files in `OIEE_Comments_Processed`
2. Run `GENAI-Comments2Summaries.py`
    - **Input:** Credentials in `GENAI_Credentials`
    - **Input:** Files in `OIEE_Comments_Processed`
    - **Output:** Files in `GENAI`

---

### **Last Step: Generate Final Reports**

- Run `DATA-Evaluations2Report.py`
    - **Input:** Files in `DATA_Evaluations`
    - **Input:** Files in `DATA_DeptStats`
    - **Input (if `ai_flag` is set):** Files in `GENAI`
    - **Output:** Reports in `DATA_Reports`

---

## **Notes**

- **All processing scripts must be run in a secure environment.**
- **Do not share any raw or processed data externally.**
- *Contact the project maintainer for questions about workflow, access, or data security.*

