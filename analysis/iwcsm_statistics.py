
###################
### Globals
###################

DATA_DIR = "./supplemental_data/"
PLOT_DIR = "./logs/"

###################
### Imports
###################

## Standard Libraries
from datetime import datetime

## External Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###################
### Helper Functions
###################

## Flatten
flatten = lambda l: [item for sublist in l for item in sublist]

## Union of Sets
set_union = lambda s: sorted(list(set.union(*[i for i in s if isinstance(i, set)])))

def format_float(num):
    """

    """
    num = num.replace("(","").replace(")","")
    mult = 1.0
    if num.endswith("k"):
        num = num[:-1]
        mult = 1000.0
    num = float(num) * mult
    return num
    
def process_size(size):
    """

    """
    if pd.isnull(size):
        return np.nan
    if size == "na":
        return "N/A"
    sizes = size.split(", ")
    sizes = [s.split() for s in sizes]
    sizes = dict((x, format_float(y)) for x, y in sizes)
    return sizes

def process_tasks(tasks):
    """

    """
    if pd.isnull(tasks):
        return np.nan
    if tasks == "na":
        return "N/A"
    tasks = set(tasks.split(", "))
    return tasks

def process_availability(availability):
    """

    """
    if pd.isnull(availability):
        return "Unknown"
    elif availability == "not_available_for_distribution":
        return "Not Available (Prohibited)"
    elif availability == "no_longer_exists":
        return "Not Available (No Longer Exists)"
    elif availability == "available_via_signed_agreement":
        return "Available (Signed Agreement)"
    elif availability == "available_via_author_contact":
        return "Available (Author Discretion)"
    elif availability == "available_via_download":
        return "Available (No Restrictions)"
    elif availability == "reproducible_via_api":
        return "Available (Reproducible via API)"
    elif availability == "pending":
        return "Pending Availability"
    else:
        raise ValueError("Encountered unaccounted availability")

def process_platforms(platforms):
    """

    """
    if pd.isnull(platforms):
        return np.nan
    platforms = set(platforms.split(", "))
    return platforms

def process_annotation_style(style):
    """

    """
    if pd.isnull(style):
        return np.nan
    if style == "na":
        return "N/A"
    style = set(style.split(", "))
    return style

def process_sources(sources):
    """

    """
    sources = str(sources)
    sources = sources.split(", ")
    sources = [int(s) for s in sources]
    return sources

def original_source_check(row):
    """

    """
    if row["paper_id"] in row["source_ids"]:
        return True
    return False

def check_in_set(candidates, query):
    """

    """
    if not isinstance(candidates, set):
        return False
    exists = query in candidates
    return exists

def get_clean_task_name(tasks):
    """

    """
    task_names = []
    if "depression" in tasks:
        task_names.append("Depression")
    if "suicide_(ideation)" in tasks:
        task_names.append("Suicidal Ideation")
    task_names = ", ".join(task_names)
    return task_names

def get_clean_platform_name(platforms):
    """

    """
    platforms = [t.title() for t in list(platforms)]
    platforms = ", ".join(platforms)
    return platforms

def get_clean_availability(availability):
    """

    """
    availability = availability.split("(")[1].split(")")[0]
    return availability

def get_clean_reference(row):
    """

    """
    authors = row["authors"].split(", ")
    year = row["year"]
    reference = "{} ({})"
    if len(authors) == 1:
        reference = reference.format(authors[0], year)
    elif len(authors) == 2:
        authors = " & ".join(authors)
        reference = reference.format(authors, year)
    elif len(authors) == 3:
        authors = ", ".join(authors[:-1]) + ", & " + authors[-1]
        reference = reference.format(authors, year)
    else:
        authors = authors[0] + " et al."
        reference = reference.format(authors, year)
    return reference

###################
### Load/Format Dataset
###################

## Load Dataset
data_df = pd.read_excel(f"{DATA_DIR}data_sources_standardized.xlsx")

## Format Sizes
size_cols = ["n_documents","n_individuals","n_conversations"]
for sc in size_cols:
    data_df[sc] = data_df[sc].map(process_size)

## Format Tasks
data_df["tasks"] = data_df["tasks"].map(process_tasks)
unique_tasks = set_union(data_df["tasks"])
for t in unique_tasks:
    data_df[f"task={t}"] = data_df["tasks"].map(lambda i: check_in_set(i, t))

## Process Availability
data_df["availability"] = data_df["availability"].map(process_availability)

## Process Platforms
data_df["platforms"] = data_df["platforms"].map(process_platforms)
unique_platforms = set_union(data_df["platforms"])
for p in unique_platforms:
    data_df[f"platform={p}"] = data_df["platforms"].map(lambda i: check_in_set(i, p))

## Process Annotation Style
data_df["annotation_style"] = data_df["annotation_style"].map(process_annotation_style)
unique_annot_styles = set_union(data_df["annotation_style"])
for a in unique_annot_styles:
    data_df[f"annotation={a}"] = data_df["annotation_style"].map(lambda i: check_in_set(i, a))

## Process Sources
data_df["source_ids"] = data_df["source_ids"].map(process_sources)
data_df["contains_original_source"] = data_df.apply(original_source_check, axis=1)

## Process Primary Language
data_df["primary_language"] = data_df["primary_language"].str.title()

###################
### Initial Filtering
###################

"""
Notes:
- Begin with 71 papers. Of the 71 papers, 58 contain original datasets. Of the papers
  that used an existing dataset, 6 used the CLPsych 2015 paper. 2 papers used the RSDD
  data set. All others were one-off runs.
- We make no claims that this list is comprehensive. Indeed, the goal of this project
  is to establish a single, evolving repository of literature and data that can be
  used for researchers in this community. It will be housed Github (open for contribution)
  and managed by the authors of this paper.
- Papers span 2012 to 2019. Number of papers generally increasing over time (e.g. 1 in 
  2012 and 16 in 2019, though this could be a facet of the initial search technique)
- Note that "original" distinction might not be perfect. For example, the SMHD data set
  extends and cleans the RSDD dataset (there is a high overlap in users). For some of the
  Twitter datasets (especially those from Glen) which may use barely-modified regular 
  expressions to annotate individuals, there may also be some overlap between the group
  annotated as suffering from a mental health disorder.
- Initial Filtering Criteria
    * Electronic Multimedia Only (e.g. No EHR, Clinical Notes, Last Statements)
    * Mental Health Status Only (e.g. No Billing Codes, No Date Detection, Cyberbullying)
    * Ignore Datasets that lack annotation (e.g. IR, platform data dump with demographics)
    * Ignore Tasks framed more closely as sentiment analysis
    * Unique datasets only (e.g. original)
"""

## Isolate Unique Datasets
data_df =  data_df.loc[data_df["contains_original_source"]]

## Filter Out Tasks that lack annotation
data_df = data_df.loc[~data_df.tasks.isnull()]
data_df = data_df.loc[data_df.tasks != "N/A"]

## Platforms to Ignore
filter_platforms = set(["ehr",
                        "death_row_last_statements",
                        "doctor_patient_conversation",
                        "interview",
                        "phone"])
data_df = data_df.loc[~data_df["platforms"].map(lambda i: all(p in filter_platforms for p in i))]

## Tasks to Ignore
filter_tasks = set(["counseling_outcome",
                    "cyberbullying",
                    "depression_(diagnoses_date)",
                    "psychiatric_(concepts)",
                    "psychiatric_(readmission)",
                    "sentiment"])
data_df = data_df.loc[~data_df["tasks"].map(lambda i: all(p in filter_tasks for p in i))]

###################
### Preliminary Analysis
###################

## Get Filtered Platforms, Tasks
unique_platforms_filtered = [p for p in set_union(data_df["platforms"]) if p not in filter_platforms]
unique_tasks_filtered = [p for p in set_union(data_df["tasks"]) if p not in filter_tasks]
unique_annot_styles_filtered = set_union(data_df["annotation_style"])

## Platform Distribution
platform_distribution = data_df[[f"platform={p}" for p in unique_platforms_filtered]].sum().sort_values()

## Task Distribution
task_distribution = data_df[[f"task={t}" for t in unique_tasks_filtered]].sum().sort_values()

## Annotation Distribution
annot_dist = data_df[[f"annotation={a}" for a in unique_annot_styles_filtered]].sum().sort_values()

## Language Distribution
language_dist = data_df.primary_language.value_counts()

## Availability Distribution
availability_dist = data_df.availability.value_counts()
clinical_annots = ["clinical_diagnoses","medical_history","survey_(clinical)"]
clinical_availability = data_df.loc[data_df.annotation_style.map(lambda i: any(c in i for c in clinical_annots))].availability

"""
Notes:
- Size/Timespan:
    * Filtering leaves us with 44 papers from 2012 to 2019.
    * Average of 5.5 datasets released each year, with the majority
      being released after 2014. Min is 1 (2012). Max is 8 (2018).
- Task
    * 29 types of mental health conditions (majority only occurring 1 or 2 times)
    * Majority of research focuses on Depression (19) and Suicidal Ideation (17)
    * PTSD, Bipolar Personality Disorder, Self Harm, and Eating Disorders all
      have more than 4 unique data sets
- Platform
    * 12 unique electronic media platforms/types (e.g. web search is general)
    * Majority of research uses Twitter (19) and Reddit (12)
- Language
    * 4 unique languages
    * Of the 44 papers, 38 primarily use English text
    * Chinese (4), Korean (1), and Japanese (1) round out the remaining list
    * Chinese datasets use Sina Weibo and Youtube. Japanese uses Twitter. Korean
      uses facebook.
- Annotation Elements
    * 14 unique types of annotation mechanisms identified (not necessarily disjoint)
    * Regular expressions (or keyword matches) were used in 18 of the papers,
      with manual annotation used in 14. Of the 18 papers that used regular expressions,
      9 also used some form of manual annotation on a sample of the data to verify
      the correctness of the labeling schema.
    * Clinically-based surveys (9), community participation (7), and platform activity (3)
      were also regularly used to annotate users
    * Majority of papers annotated on an individual level (29), as opposed to document (15)
- Dataset Availability
    * 8 availability classes
    * 17 of the papers had unknown availability
    * 5 Data sets were known not to be available for distribution
    * 21 of the papers had known availability. Majority of available papers (10)
      require a signed agreement. Reproducible with API (8), available from
      author with permission (2), and available without restriction (1) rounded 
      out the list.
    * Of the 10 papers that used clinical annotation (e.g. mental health test,
      medical history), 2 had known availability (both prohibited) and the rest
      were unknown
    * 1 dataset had pending availability
"""

###################
### Stricter Filtering
###################

"""
- Additional Filtering Criteria
    * Tasks: Depression or Suicidal Ideation
    * Language: English
    * Availability: Known and readily available
"""

## Outline Filtering Criteria
acceptable_tasks = set([
    "depression",
    "suicide_(ideation)"
])
acceptable_language = "English"
acceptable_availability = set([
            "Available (Signed Agreement)",
            "Available (Reproducible via API)",
            "Available (Author Discretion)",
            "Available (No Restrictions)"])

## Apply Additional Filtering Criteria
data_df = data_df.loc[data_df["tasks"].map(lambda i: any(t in acceptable_tasks for t in i))]
data_df = data_df.loc[data_df["availability"].map(lambda i: i in acceptable_availability)]
data_df = data_df.loc[data_df["primary_language"] == acceptable_language]

###################
### Notes
###################

###################
### Figures (Tables)
###################

## Clean DF for Tables
latex_df = data_df[["title",
                    "authors",
                    "year",
                    "platforms",
                    "tasks",
                    "availability"]].copy()
latex_df["tasks"] = latex_df["tasks"].map(get_clean_task_name)
latex_df["platforms"] = latex_df["platforms"].map(get_clean_platform_name)
latex_df["availability"] = latex_df["availability"].map(get_clean_availability)
latex_df["reference"] = latex_df.apply(get_clean_reference, axis = 1)

## Sort Rows, Columns
latex_df.sort_values("year", ascending = True, inplace = True)
latex_df = latex_df[["title",
                     "reference",
                     "platforms",
                     "tasks",
                     "availability"]]
latex_df.reset_index(drop=True, inplace=True)
