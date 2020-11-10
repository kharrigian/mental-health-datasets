
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
    for t in tasks:
        if t in ["adhd","ocd", "ptsd"]:
            task_names.append(t.upper())
        elif t == 'mental_health_(combined)':
            task_names.append("Mental Health Disorder (General)")
        elif t == "rape_(survivor)":
            task_names.append("Trauma (Rape Survivor)")
        else:
            task_names.append(t.replace("_"," ").title())
    task_names = ", ".join(task_names)
    return task_names

def get_clean_task_abbr(clean_task_names):
    """

    """
    ## Define Abbreviation Map
    abbr_map = {
        "Suicide (Ideation)":"SI",
        "Suicide (Attempt)":"SA",
        "Bipolar Disorder":"BIPD",
        "Borderline Personality Disorder":"BRPD",
        "PTSD": "PTSD",
        "Seasonal Affective Disorder": "SAD",
        "Depression": "DEP",
        "Anxiety": "ANX",
        "Eating": "EAT",
        "Eating (Recovery)":"EATR",
        "OCD": "OCD",
        "Schizophrenia": "SCHZ",
        "ADHD": "ADHD",
        "Psychosis": "PSY",
        "Anxiety (Social)": "ANXS",
        "Self Harm": "SH",
        "Rape (Survivors)": "RS",
        "Panic": "PAN",
        "Trauma": "TRA",
        "Alcoholism": "ALC",
        "Opiate Addiction": "OPAD",
        "Aspergers": "ASP",
        "Autism": "AUT",
        "Opiate Usage": "OPUS",
        "Mental Health Disorder (General)": "MHGEN",
        "Stress": "STR",
        "Stress (Stressor And Subjects)":"STRS"
    }
    ## Split and Replace
    clean_task_names = clean_task_names.split(", ")
    clean_task_names = [abbr_map[t] for t in clean_task_names]
    return ", ".join(clean_task_names)


def get_clean_platform_name(platforms):
    """

    """
    platforms = [t.replace("_"," ").title() for t in list(platforms)]
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

def sum_relevant_sizes(size_dict):
    """

    """
    ## Keys To Consider for Counting Depression/Suicide Data Set Sizes
    acceptable_keys = ["combined",
                       "control",
                       "depression",
                       "depression_(low-mild)",
                       "depression_(high)",
                       "mental_health_(combined)",
                       "increase_activity",
                       "constant_activity",
                       "decrease_activity",
                       "suicide_(ideation)",
                       "suicide_(attempt)"]
    ## Ignore Missing et al.
    if pd.isnull(size_dict):
        return np.nan
    if isinstance(size_dict, str):
        return np.nan
    ## Sum
    total = 0
    for a in acceptable_keys:
        if a in size_dict:
            total += size_dict[a]
    return total

def sum_all_sizes(size_dict):
    """

    """
    ## Ignore Missing et al.
    if pd.isnull(size_dict):
        return np.nan
    if isinstance(size_dict, str):
        return np.nan
    ## Sum
    total = 0
    for x, y in size_dict.items():
        total += y
    return total
    

###################
### Load/Format Dataset
###################

## Load Dataset
data_df = pd.read_excel(f"{DATA_DIR}data_sources_standardized.xlsx")

## Format Sizes
size_cols = ["n_documents","n_individuals","n_conversations"]
for sc in size_cols:
    data_df[sc] = data_df[sc].map(process_size)
    data_df[f"{sc}_relevant_total"] = data_df[sc].map(sum_relevant_sizes)
    data_df[f"{sc}_total"] = data_df[sc].map(sum_all_sizes)

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

## Year Filter 
data_df = data_df.loc[data_df["year"] < 2020].reset_index(drop=True).copy()

###################
### Initial Filtering
###################

"""
Notes:
- Begin with 139 papers. Of the 139 papers, 111 contain original datasets. Of the papers
  that used an existing dataset, 7 used the CLPsych 2015 paper. 3 papers used the RSDD
  data set. 3 used Gktosis et al. ("Language of mental health"), 2 used CLPsych 2016, and the 
  rest of duplicates were one off uses.
- We make no claims that this list is comprehensive. Indeed, the goal of this project
  is to establish a single, evolving repository of literature and data that can be
  used for researchers in this community. It will be housed Github (open for contribution)
  and managed indefinitely by the authors of this paper.
- Papers span 2012 to 2019. Number of papers generally increasing over time (e.g. 1 in 
  2012 and 16 in 2019, though this could be a facet of the initial search technique)
- Note that "original" distinction might not be perfect. For example, the SMHD data set
  extends and cleans the RSDD dataset (there is a high overlap in users). For some of the
  Twitter datasets (especially those from Coppersmith) which may use barely-modified regular 
  expressions to annotate individuals, there may also be some overlap between the group
  annotated as suffering from a mental health disorder.
- Initial Filtering Criteria
    * Electronic Multimedia Only w/ Text Focus (e.g. Includes SMS, Social Media, Forums, Web Search;
      Excludes EHR, Clinical Notes, Last Statements, Mobile Health Apps)
    * Mental Health Status Only (e.g. No Billing Codes, No Diagnoses Date Detection, No Cyberbullying)
    * Ignore Datasets that lack any type of annotation (e.g. IR, platform data dump with demographics)
    * Ignore tasks that don't attempt to adhere to DSM definition of mental health condition (e.g. 
      no sentiment/mood tasks)
    * Unique datasets only (e.g. original)
"""

## Initialize Counts for Filters
filter_counts = dict()

## Isolate Unique Datasets (139 -> 111)
filter_counts["initial_search"] = len(data_df)
data_df =  data_df.loc[data_df["contains_original_source"]]
filter_counts["unique_datasets_only"] = len(data_df)

## Filter Out Tasks that lack annotation (111 -> 108)
data_df = data_df.loc[~data_df.tasks.isnull()]
data_df = data_df.loc[data_df.tasks != "N/A"]

## Platforms to Ignore (108 -> 108: Search-based Filter)
filter_platforms = set(["ehr",
                        "death_row_last_statements",
                        "doctor_patient_conversation",
                        "interview",
                        "phone",
                        'ecological_momentary_assessments',
                        "essays"])
data_df = data_df.loc[~data_df["platforms"].map(lambda i: all(p in filter_platforms for p in i))]
data_df["platforms"] = data_df["platforms"].map(lambda i: set(j for j in i if j not in filter_platforms))

## Tasks to Ignore (108 -> 102)
filter_tasks = set(["counseling_outcome",
                    "cyberbullying",
                    "imminent_death",
                    "depression_(diagnoses_date)",
                    "psychiatric_(concepts)",
                    "psychiatric_(readmission)",
                    "sentiment",
                    "aggression",
                    "breast_cancer",
                    "ehr_categories",
                    "life_satisfaction",
                    "relationships"])
data_df = data_df.loc[~data_df["tasks"].map(lambda i: all(p in filter_tasks for p in i))]
data_df["tasks"] = data_df["tasks"].map(lambda i: set(j for j in i if j not in filter_tasks))
filter_counts["apply_exclusion_criteria"] = len(data_df)

###################
### Preliminary Analysis
###################

"""
Notes:
- Size/Timespan:
    * Filtering described above leaves us with 102 papers from 2012 to 2019.
    * Average of 12.75 datasets released each year, with the majority
      being released after 2012. Min is 1 (2012). Max is 23 (2017).
- Task
    * 36 types of mental health-related modeling tasks (majority only occurring 1 or 2 times)
    * Majority of research focuses on Depression (42) and Suicidal Ideation (26), and Eating (11)
    * PTSD, Self Harm, Anxiety, Schizophrenia, Bipolar Disorder, Stress also
      have more than 4 unique data sets
- Platform
    * 20 unique electronic media platforms/types (e.g. web search is general)
    * Majority of research uses Twitter (47) and Reddit (22)
    * Surprisingly limited Youtube, Instagram, Facebook research. Facebook/Instagram likely
      due to privacy constraints. Dearth of Instagram may also be due to our paper sourcing
      process that focused primarily on availability of text data. Dearth of Youtube is also
      likely due to privacy constraints. There may exist future opportunities here if
      researchers can properly respect privacy constraints (given the prevalence of these
      platforms in society)
- Language
    * 6 unique languages
    * Of the 102 papers, 85 primarily use English text
    * Chinese (10), Japanese (4), Korean (2), Spanish (1), and Portuguese (1) round out the remaining list
    * Chinese datasets use Sina/Tencent Weibo and Youtube. Japanese uses Twitter, 
      Mixi, and Tobyo Toshoshitshu. Korean uses uses Twitter and Facebook.
    * Given that some research has shown cultural differences lead to different
      presentations of mental health disorders, there would be large value in
      new datasets that explore a wider variety of languages or populations. 
    * Several papers commonly mention filtering to English data (inferred via i.e. Langid)
- Annotation Elements
    * 24 unique types of annotation mechanisms identified (not necessarily disjoint)
    * Regular expressions (or keyword matches) were used in 43 of the papers,
      with manual annotation used in 38. Of the 43 papers that used regular expressions,
      23 also used some form of manual annotation on a sample (possibly all) of the data to verify
      the correctness of the labeling schema.
    * Clinically-based surveys (22), community participation/affiliation-based (24), hashtags (5),
      tags (4), interviews (4) and and platform activity (3) were also regularly used to annotate users
    * Majority of papers annotated on an individual level (63), as opposed to document (40) [note
    that one dataset was labeled at both a document and individual level]
- Dataset Availability
    * 8 availability classes
    * 54 of the papers had unknown availability (e.g. not readily distinguishable from paper)
    * 12 Data sets were known not to be available for distribution
    * 35 of the papers had known availability. Majority (18) reproducible with API and some 
      manual effort. (12) available without modification with a signed agreement,
      (2) available from author with permission, and (3) available without restriction.
    * Of the 22 papers that used clinical annotation (e.g. mental health test,
      medical history), 8 had known availability (7 prohibited, 1 no longer exists) and the rest
      were unknown. Anecdotally, we noticed that datasets that weren't reproducible
      via an API (community membership, regex) and had some form of manual annotation
      were often not able to be distributed to the terms of the collection (e.g. IRB/HIPAA)
            - Note: We did not check with authors regarding any non-English datasets.
    * 1 dataset had pending availability (authors still doing primary research with it)
- Dataset Size
    * Document-Level Annotation
        - 3/40 have unknown amounts of documents, 28/40 have unknown amounts of individuals
        - Individuals: Min (33), Max (950000), Mean (144350.18), Median (1227), Std (320485)
        - Documents:  Min (129), Max (117200000), Mean (3407890), Median (6516), Std (19512189)
    * Individual-Level Annotation
        - 40/63 have unknown amounts of documents, 1/63 has unknown amounts of individuals
        - Individuals: Min (14), Max (300038395), Mean (4934830), Median (909), Std (38413897)
        - Documents:  Min (1856), Max (10035292000), Mean (463926900), Median (462447), Std (2137855934)
    * Primarily left-skewed. One concern is that the median number of unique individuals is relatively low. 
    Small sample size means that the data sets are not likely representative of the greater population.
    * As expected, the largest datasets primarily use regular expressions or platform activity 
    (e.g. distant supervision) to create the full sample. That said, they do make sure 
    to use manual annotation on a subset of the data to validate the labeling strategy. Noise is 
    not necessarily considered during the evaluation procedures, however.
"""

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
clinical_annots = ["clinical_diagnoses","survey_(clinical)"]
clinical_availability = data_df.loc[data_df.annotation_style.map(lambda i: any(c in i for c in clinical_annots))][["title","tasks","primary_language","availability"]]

## Size Distribution
document_annot_docs = data_df.loc[(data_df.annotation_level=="document")]["n_documents_total"].sort_values().dropna()
document_annot_inds = data_df.loc[(data_df.annotation_level=="document")]["n_individuals_total"].sort_values().dropna()
individual_annot_docs = data_df.loc[(data_df.annotation_level=="individual")]["n_documents_total"].sort_values().dropna()
individual_annot_inds = data_df.loc[(data_df.annotation_level=="individual")]["n_individuals_total"].sort_values().dropna()

###################
### Stricter Filtering
###################

"""
- Additional Filtering Criteria
    * Availability: Known and readily available
- Note: By nature of how we conducted this review, this strictly disqualifies
        non-English datasets from making it to the final filtered set.
"""

## Outline Filtering Criteria
acceptable_availability = set([
            "Available (Signed Agreement)",
            "Available (Reproducible via API)",
            "Available (Author Discretion)",
            "Available (No Restrictions)"])

## Apply Additional Filtering Criteria
filter_counts["known_availability"] = (data_df["availability"] != "Unknown").sum()
data_df = data_df.loc[data_df["availability"].map(lambda i: i in acceptable_availability)]
filter_counts["available"] = len(data_df)

## Format Filter Counts
filter_counts = pd.Series(filter_counts)

###################
### Figures (Tables)
###################

## Clean DF for Tables
latex_df = data_df[["title",
                    "authors",
                    "year",
                    "platforms",
                    "tasks",
                    "annotation_level",
                    "n_individuals_total",
                    "n_documents_total",
                    "availability"]].copy()
latex_df["tasks"] = latex_df["tasks"].map(lambda x: set(i for i in x if i not in filter_tasks)).map(get_clean_task_name).map(get_clean_task_abbr)
latex_df["platforms"] = latex_df["platforms"].map(get_clean_platform_name)
latex_df["availability"] = latex_df["availability"].map(get_clean_availability)
latex_df["reference"] = latex_df.apply(get_clean_reference, axis = 1)
latex_df["reference"] = latex_df["title"] + " " + latex_df["reference"]
latex_df["annotation_level"] = latex_df["annotation_level"].str.title().map(lambda i: i[:3] + ".")
latex_df["n_individuals_total"] = latex_df["n_individuals_total"].map(lambda i: "{:,d}".format(int(i)) if not pd.isnull(i) else "")
latex_df["n_documents_total"] = latex_df["n_documents_total"].map(lambda i: "{:,d}".format(int(i)) if not pd.isnull(i) else "")

## Sort Rows, Columns
latex_df.sort_values("year", ascending = True, inplace = True)
latex_df = latex_df[["reference",
                     "platforms",
                     "tasks",
                     "annotation_level",
                     "n_individuals_total",
                     "n_documents_total",
                     "availability"]].copy()
latex_df.reset_index(drop=True, inplace=True)
latex_df.rename(columns = {"reference":"Reference",
                           "platforms":"Platform(s)",
                           "tasks":"Task(s)",
                           "annotation_level":"Label Resolution",
                           "n_individuals_total":"# Individuals",
                           "n_documents_total":"# Documents",
                           "availability":"Availability"},
                inplace = True)

###################
### Figures (Plots)
###################

## Search
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(range(filter_counts.shape[0]),
       filter_counts.values,
       color="navy",
       alpha=0.7,
       edgecolor="navy")
for i, c in enumerate(filter_counts.values):
    ax.text(i, c + 1, int(c), fontsize=18, ha="center", va="bottom")
ax.set_xticks(range(filter_counts.shape[0]))
ax.set_xticklabels([i.replace("_","\n").title() for i in filter_counts.index],
                    rotation=45,
                    ha="center")
ax.set_ylabel("# Articles", fontweight="bold", fontsize=28)
ax.set_xlabel("Filtering Stage", fontweight="bold", fontsize=28)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=18)
ax.set_ylim(0,filter_counts.max()+8)
fig.tight_layout()
fig.savefig("./supplemental_data/search.pdf", dpi=300)
plt.close(fig)

