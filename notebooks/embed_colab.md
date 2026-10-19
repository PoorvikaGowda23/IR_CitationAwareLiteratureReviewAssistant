# SPECTER2 Embedding on Google Colab (Free T4 GPU)

This guide shows how to run full-corpus embedding in Google Colab on a GPU in ~2 minutes and download the resulting artifacts back to your local `data/` folder.

---

### Step 1: Open Colab & Select GPU Runtime
1. Open [Google Colab](https://colab.research.google.com).
2. Go to **Runtime → Change runtime type** and select **T4 GPU** (under Hardware accelerator).

---

### Step 2: Install Dependencies
Run in Cell 1:
```python
!pip install -q torch transformers adapters faiss-cpu pandas pyarrow tqdm
```

---

### Step 3: Upload `papers.parquet` & Repo Modules
Run in Cell 2:
```python
from google.colab import files
import os

os.makedirs("data", exist_ok=True)
print("Please upload papers.parquet (and config.py, schema.py, embed.py):")
uploaded = files.upload()

for filename in uploaded.keys():
    if filename.endswith(".parquet"):
        os.rename(filename, os.path.join("data", filename))
```

---

### Step 4: Run Embedding Pipeline
Run in Cell 3:
```python
!python embed.py
```
*Expected GPU speed on T4:* ~25–40 papers/second. A 1,500-paper corpus finishes in ~45–60 seconds!

---

### Step 5: Download Artifacts Back to Local `data/`
Run in Cell 4:
```python
from google.colab import files

files.download("data/embeddings.npy")
files.download("data/ids.json")
files.download("data/faiss.index")
```

Place these 3 downloaded files into your local project's `data/` folder.
