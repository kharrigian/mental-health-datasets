
## Imports
import pandas as pd
from tabulate import tabulate
from datetime import datetime

## Read Data
df = pd.read_excel("data_sources.xlsx")

## Format Cells
newline_replace = lambda x: x.replace("\n","<br/>") if not isinstance(x, float) else x
strip_space = lambda x: x.strip() if not isinstance(x, float) else x
columns_to_format = ["Paper",
                     "Authors",
                     "Platform",
                     "Target Outcomes",
                     "Labeling Methodology",
                     "Size",
                     "Availability",
                     "Additional Comments",
                     "Dataset Link (if any)",
                     "Reference Link"]
for col in columns_to_format:
    df[col] = df[col].map(newline_replace)
    df[col] = df[col].map(strip_space)

## Generate Markdown Table
md_table = tabulate(df, tablefmt="pipe", headers="keys")

## Output
md_output = """
# Mental Health Datasets

The information below is an evolving list of data sets (primarily from electronic/social media) that have been used to model mental-health phenomena. The raw data can be found in `data_sources.xlsx`. If you are an author of any of these papers and feel that anything is misrepresented, please do not hesitate to reach out to me at kharrigian@jhu.edu.

**Last Update**: {}

{}
""".format(datetime.now().isoformat(), md_table)

## Write Out
with open("README.md", "w") as the_file:
    the_file.write(md_output)