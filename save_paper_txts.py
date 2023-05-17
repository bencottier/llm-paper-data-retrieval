import os
import requests
import pandas as pd
from pdfminer.high_level import extract_text

os.makedirs('input_data', exist_ok=True)

# Download dataset from the Parameters, Compute and Data Trends in ML sheet
df = pd.read_csv('https://docs.google.com/spreadsheets/d/1beiholXkLpiYoeyDL64bOHLI-mXZ-_8ZnY1Ed6Sfjh0/export?format=csv#gid=0')

reversed_df = df.iloc[::-1]
df = pd.concat([reversed_df[:10], reversed_df[40:50], reversed_df[80:90]])

# year_start = 2017

# # Recode columns
# df['Publication date'] = pd.to_datetime(df['Publication date'], errors='coerce', dayfirst=True)

# # Filter for papers of only the last 5 years
# df = df[df['Publication date'] > f'{year_start}-01-01']

# Keep only bibliographical data
df = df.filter(['Author(s)', 'Publication date', 'Reference', 'Link'])
df = df[df['Link'].notna()]
# Keep only links which forward to a pdf or an arxiv link
df = df[df['Link'].str.contains('(arxiv|.pdf$)', regex=True)]

keys = ['Number of hardware units', 'Hardware model', 'Training FLOP/s']

# Enable for test running with the first ten papers
# df = df[:10]

print(df['Reference'].values)

# Or a specific paper
# idx = 3
# df = df[idx:idx+1]

for i, row in df.iterrows():
    url = row['Link']

    # replace "abs" with "pdf" in arxiv url links
    url = url.replace('abs', 'pdf')
    print(f"Looking into \"{row['Reference']}\"")

    paper_title = row['Reference'].replace(' ', '_').replace(':', '').replace('"', '').lower()
    filename = 'input_data/' + paper_title + '.txt'
    if os.path.exists(filename):
        print(f"File {filename} already exists, skipping")
        continue

    try:
        response = requests.get(url)
    except Exception as e:
        print(f"There's something wrong with downloading: {e}")
        raise e

    file = open("download.pdf", "wb")
    file.seek(0) # overwrite previous file
    file.write(response.content)
    file.close()

    text = extract_text('download.pdf')

    with open(filename, 'w') as f:
        f.write(text)
