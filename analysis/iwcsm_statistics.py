
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
        return "Not Available (Prohibited))"
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
### Data Overview
###################

"""
Notes:
- Begin with 71 papers
- Initial Filtering Criteria
    * Electronic Multimedia Only (e.g. No EHR, Clinical Notes, Last Statements)
    * Mental Health Status Only (e.g. No Billing Codes, No Date Detection, Cyberbullying)
    * Ignore Datasets that lack annotation (e.g. IR, platform data dump with demographics)
    * Ignore Tasks framed more closely as sentiment analysis
    * Unique datasets only (e.g. original)
"""

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

## Isolate Unique Datasets
data_df =  data_df.loc[data_df["contains_original_source"]]

###################
### Stricter Filtering
###################

"""
Master Filtering Criteria
- Tasks: Depression or Suicidal Ideation
- Language: English
- Original Source: True (ignore uses of other datasets)
"""

## Outline Filtering Criteria
acceptable_tasks = set([
    "depression",
    "suicide_(ideation)"
])
