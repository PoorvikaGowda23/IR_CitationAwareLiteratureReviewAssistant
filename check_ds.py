import pandas as pd
papers = pd.read_parquet("data/papers.parquet")
edges = pd.read_parquet("data/edges.parquet")
linked = set(edges["citing_id"]) | set(edges["cited_id"])
print("edges:", len(edges))
print("papers with >=1 edge:", papers["paper_id"].isin(linked).mean().round(3))
print(edges["s2_intent"].value_counts(dropna=False))