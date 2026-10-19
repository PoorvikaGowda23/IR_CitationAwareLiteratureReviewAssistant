import pandas as pd

df = pd.read_parquet("data/seed_papers.parquet")

print(df.shape)              # (rows, columns)
print(df.columns.tolist())   # should include paper_id, title, abstract, year, tldr, is_survey
print(df.head())

# Surveys that were flagged
surveys = df[df["is_survey"] == True]
print(len(surveys), "surveys flagged")
print(surveys[["paper_id", "title", "year"]].to_string())


import pandas as pd
df = pd.read_parquet("data/seed_papers.parquet")
print(df[["title", "year"]].to_string())

import pandas as pd
df = pd.read_parquet("data/seed_papers.parquet")
print(df.shape, "| surveys:", df["is_survey"].sum())
print("whitespace titles:", df["title"].str.startswith((" ", "\n")).sum())
print("duplicate ids:", df["paper_id"].duplicated().sum())
print(df[df["is_survey"]][["title", "year"]].to_string())