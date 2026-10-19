# SciBERT Citation-Intent Fine-tuning — Google Colab Guide

Run each fenced code block as a separate Colab cell **in order**.
The notebook mounts Google Drive so the checkpoint survives runtime resets.

---

## Cell 1 — Mount Google Drive & set checkpoint path

```python
from google.colab import drive
drive.mount("/content/drive")

# Edit this path to wherever you want the checkpoint saved on your Drive
CHECKPOINT_DIR = "/content/drive/MyDrive/scibert_intent"
print("Checkpoint will be saved to:", CHECKPOINT_DIR)
```

---

## Cell 2 — Clone / upload the project and install dependencies

If you pushed the repo to GitHub, clone it:
```python
!git clone https://github.com/<YOUR_USERNAME>/IR_CitationAwareLiteratureReviewAssistant.git
%cd IR_CitationAwareLiteratureReviewAssistant
```

If you are uploading the repo as a zip instead, use:
```python
# from google.colab import files
# files.upload()   # upload the zip
# !unzip <your_zip_name>.zip
# %cd <unzipped_folder_name>
```

Install Python dependencies:
```python
!pip install -q \
    "datasets==2.21.0" \
    transformers \
    torch \
    scikit-learn \
    joblib
```

---

## Cell 3 — Verify GPU is available

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device name:  ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

> **Tip**: In Colab, go to *Runtime → Change runtime type → GPU* (T4 is free).

---

## Cell 4 — Load SciCite dataset and check class balance

```python
from datasets import load_dataset
from intent_classifier import LABEL_NAMES, print_class_balance

ds = load_dataset("allenai/scicite", trust_remote_code=True)

train_texts  = [ex["string"] for ex in ds["train"]]
train_labels = [ex["label"]  for ex in ds["train"]]
val_texts    = [ex["string"] for ex in ds["validation"]]
val_labels   = [ex["label"]  for ex in ds["validation"]]
test_texts   = [ex["string"] for ex in ds["test"]]
test_labels  = [ex["label"]  for ex in ds["test"]]

print_class_balance(train_labels, "train")
print_class_balance(val_labels,   "validation")
```

Expected output:
```
-- Class balance (train, n=8194) --
  background     2266  (27.7%)
  method         4840  (59.1%)
  result         1088  (13.3%)
```

---

## Cell 5 — Fine-tune SciBERT (≈ 15 min on T4)

```python
from intent_classifier import train_scibert

ckpt_path = train_scibert(
    train_texts, train_labels,
    val_texts,   val_labels,
    checkpoint_dir = CHECKPOINT_DIR,   # saves to Google Drive
    epochs     = 3,
    lr         = 2e-5,
    batch_size = 16,
)
print("Best checkpoint at:", ckpt_path)
```

What you will see printed per epoch:
```
Device: cuda
Tokenising (max_len=128) ...
  Train batches: 512 | Val batches: 58
Loading allenai/scibert_scivocab_uncased ...
  Total steps: 1536 | Warmup steps: 154
  Epoch 1/3 | step 50/512 | loss 0.9341
  ...
  --> Epoch 1 val macro-F1: 0.8712  acc: 0.8820
  ** Best checkpoint saved (val F1=0.8712) -> /content/drive/MyDrive/scibert_intent
  Epoch 2/3 | step 50/512 | loss 0.4201
  ...
```

---

## Cell 6 — Evaluate on the test split and save metrics

```python
from intent_classifier import eval_scibert

metrics = eval_scibert(
    test_texts, test_labels,
    checkpoint_dir = CHECKPOINT_DIR,
)
```

This writes `results/scibert_intent.json` with the same schema as
`results/tfidf_intent.json`, so you can diff them directly:

```python
import json, pathlib

tfidf   = json.loads(pathlib.Path("results/tfidf_intent.json").read_text())
scibert = json.loads(pathlib.Path("results/scibert_intent.json").read_text())

print(f"{'Model':<12} {'background F1':>15} {'method F1':>12} {'result F1':>12} {'Macro F1':>10}")
print("-" * 58)
for name, m in [("TF-IDF", tfidf), ("SciBERT", scibert)]:
    row = m["per_class"]
    print(f"{name:<12} {row['background']['f1-score']:>15.4f} "
          f"{row['method']['f1-score']:>12.4f} "
          f"{row['result']['f1-score']:>12.4f} "
          f"{m['macro_f1']:>10.4f}")
```

---

## Cell 7 — Inference on new sentences

```python
from intent_classifier import predict

sentences = [
    "We follow the same preprocessing pipeline as described in [12].",
    "Our approach outperforms all baselines on the TREC-COVID benchmark.",
    "Graph neural networks were first introduced to citation networks in [3].",
]

labels, confs = predict(sentences, checkpoint_dir=CHECKPOINT_DIR)

for s, l, c in zip(sentences, labels, confs):
    print(f"[{l:<12} {c:.2f}]  {s[:70]}")
```

Expected output (approximate):
```
[method       0.94]  We follow the same preprocessing pipeline as described in [12].
[result       0.91]  Our approach outperforms all baselines on the TREC-COVID benchmark.
[background   0.88]  Graph neural networks were first introduced to citation networks in [3].
```

---

## Cell 8 — Copy scibert_intent.json back to Drive

```python
import shutil, pathlib

dest = pathlib.Path(CHECKPOINT_DIR) / "scibert_intent.json"
shutil.copy("results/scibert_intent.json", dest)
print("Metrics also copied to", dest)
```

---

## Tips

| Issue | Fix |
|---|---|
| Runtime disconnects mid-training | The best checkpoint is saved to Drive after every epoch improvement, so you only lose the current epoch |
| Out of memory on T4 | Reduce `batch_size` to 8 in Cell 5 |
| Want to resume eval only | Replace Cell 5 with: `print("Skipping training")` and run Cell 6 |
| Running locally (no GPU) | `python intent_classifier.py --scibert --checkpoint-dir data/scibert_intent` |
