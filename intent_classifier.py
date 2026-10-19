"""
intent_classifier.py
---------------------
Step 8  – TF-IDF + Logistic Regression (train locally, fast)
Step 9  – SciBERT fine-tuning with a plain PyTorch loop (runs on Colab GPU)
Step 10 – Apply fine-tuned SciBERT to edges.parquet -> edges_pred.parquet

SciCite has three labels: background, method, result.

CLI usage:
    python intent_classifier.py                  # TF-IDF: train + eval
    python intent_classifier.py --eval-only      # TF-IDF: eval saved model
    python intent_classifier.py --scibert        # SciBERT: fine-tune + eval
    python intent_classifier.py --scibert --eval-only  # SciBERT: eval checkpoint
    python intent_classifier.py --apply --smoke  # Step 10: smoke mode on 200 edges
    python intent_classifier.py --apply          # Step 10: full run on all edges
"""

import json
import argparse
from pathlib import Path

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from config import DATA_DIR, RESULTS_DIR, SEED

# Paths
MODEL_PATH       = DATA_DIR / "tfidf_intent.joblib"
METRICS_PATH     = RESULTS_DIR / "tfidf_intent.json"
EDGES_PATH       = DATA_DIR / "edges.parquet"
EDGES_PRED_PATH  = DATA_DIR / "edges_pred.parquet"

# SciCite label order  (int -> str)
LABEL_NAMES = ["background", "method", "result"]


# Data loading

def load_scicite():
    """Load SciCite from HuggingFace; return (train_texts, train_labels, test_texts, test_labels)."""
    print("Loading allenai/scicite from HuggingFace datasets ...")
    ds = load_dataset("allenai/scicite", trust_remote_code=True)
    train_texts  = [ex["string"] for ex in ds["train"]]
    train_labels = [ex["label"]  for ex in ds["train"]]
    test_texts   = [ex["string"] for ex in ds["test"]]
    test_labels  = [ex["label"]  for ex in ds["test"]]
    return train_texts, train_labels, test_texts, test_labels


def print_class_balance(labels, split_name="train"):
    """Print absolute count and percentage for each class."""
    total = len(labels)
    counts = {name: 0 for name in LABEL_NAMES}
    for lbl in labels:
        counts[LABEL_NAMES[lbl]] += 1
    print(f"\n-- Class balance ({split_name}, n={total}) --")
    for name, count in counts.items():
        pct = count / total * 100
        print(f"  {name:<12}  {count:>5}  ({pct:.1f}%)")
    print()


# Model training

def train_tfidf(train_texts, train_labels):
    """Fit TfidfVectorizer (1-2 grams, max_features=50000) + LogisticRegression(max_iter=1000)."""
    print("Fitting TF-IDF vectoriser (unigrams + bigrams, max_features=50000) ...")
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )
    X_train = vec.fit_transform(train_texts)
    print(f"  Feature matrix shape: {X_train.shape}")

    print("Training Logistic Regression (max_iter=1000) ...")
    clf = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train, train_labels)
    print("Training complete.\n")
    return vec, clf


# Evaluation

def eval_tfidf(vec, clf, test_texts, test_labels):
    """Evaluate on test split; print report + confusion matrix; save model and metrics."""
    X_test = vec.transform(test_texts)
    preds  = clf.predict(X_test)

    report_str  = classification_report(test_labels, preds, target_names=LABEL_NAMES, digits=4)
    report_dict = classification_report(test_labels, preds, target_names=LABEL_NAMES, output_dict=True)
    macro_f1    = f1_score(test_labels, preds, average="macro")
    cm          = confusion_matrix(test_labels, preds)

    print("-- Classification Report (test split) --")
    print(report_str)
    print(f"Macro F1: {macro_f1:.4f}\n")

    print("-- Confusion Matrix (rows=true, cols=predicted) --")
    header = f"{'':>12}" + "".join(f"{name:>12}" for name in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{LABEL_NAMES[i]:>12}" + "".join(f"{v:>12}" for v in row)
        print(row_str)
    print()

    # Save model
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectoriser": vec, "classifier": clf}, MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")

    # Save metrics
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "macro_f1": round(macro_f1, 6),
        "per_class": {
            label: {
                "precision": round(report_dict[label]["precision"], 6),
                "recall":    round(report_dict[label]["recall"], 6),
                "f1-score":  round(report_dict[label]["f1-score"], 6),
                "support":   int(report_dict[label]["support"]),
            }
            for label in LABEL_NAMES
        },
        "confusion_matrix": cm.tolist(),
        "model_path": str(MODEL_PATH),
        "dataset": "allenai/scicite",
        "vectoriser": "TfidfVectorizer(ngram_range=(1,2), max_features=50000, sublinear_tf=True)",
        "classifier": "LogisticRegression(max_iter=1000, class_weight=balanced)",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved -> {METRICS_PATH}\n")
    return metrics


# Inference helpers for downstream modules

def load_model():
    """Load the saved TF-IDF + LR model. Raises FileNotFoundError if not trained yet."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python intent_classifier.py` first."
        )
    bundle = joblib.load(MODEL_PATH)
    return bundle["vectoriser"], bundle["classifier"]


def predict_intent(texts):
    """Predict citation intent ('background'|'method'|'result') for a list of strings."""
    vec, clf = load_model()
    X = vec.transform(texts)
    return [LABEL_NAMES[i] for i in clf.predict(X)]


# =============================================================================
# Step 9 – SciBERT fine-tuning (plain PyTorch loop, Colab-ready)
# =============================================================================

import random
import math

# Heavy imports are deferred so that importing this module without torch
# (e.g. for the TF-IDF path) does not crash.
def _import_torch():
    import torch
    return torch

def _import_transformers():
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    return AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


# -- SciBERT paths & hyper-parameters -----------------------------------------
SCIBERT_MODEL_ID  = "allenai/scibert_scivocab_uncased"
SCIBERT_MAX_LEN   = 128
SCIBERT_EPOCHS    = 3
SCIBERT_LR        = 2e-5
SCIBERT_BATCH     = 16
SCIBERT_WARMUP    = 0.1   # fraction of total steps used for linear warmup
SCIBERT_CKPT_DIR  = DATA_DIR / "scibert_intent"   # override to GDrive path in Colab
SCIBERT_METRICS   = RESULTS_DIR / "scibert_intent.json"


# -- Dataset -------------------------------------------------------------------

class CitationDataset:
    """Minimal torch Dataset wrapping tokenised citation sentences."""

    def __init__(self, encodings, labels):
        self.encodings = encodings   # BatchEncoding from tokenizer
        self.labels    = labels      # list[int]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        torch = _import_torch()
        item  = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _tokenize(tokenizer, texts, max_len=SCIBERT_MAX_LEN):
    """Return BatchEncoding (input_ids, attention_mask) for a list of strings."""
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors=None,   # lists, not tensors – CitationDataset handles that
    )


# -- Training loop -------------------------------------------------------------

def train_scibert(
    train_texts, train_labels,
    val_texts,   val_labels,
    checkpoint_dir=SCIBERT_CKPT_DIR,
    epochs=SCIBERT_EPOCHS,
    lr=SCIBERT_LR,
    batch_size=SCIBERT_BATCH,
    warmup_frac=SCIBERT_WARMUP,
    max_len=SCIBERT_MAX_LEN,
    seed=SEED,
):
    """
    Fine-tune allenai/scibert_scivocab_uncased with a plain PyTorch loop.

    Stages visible:
      1. Tokenisation (max_len=128)
      2. DataLoader construction
      3. AdamW optimiser
      4. Linear warmup scheduler
      5. Forward / loss / backward / step per batch
      6. Per-epoch validation on val split; best checkpoint kept

    Args:
        train_texts / train_labels : SciCite training split
        val_texts   / val_labels   : SciCite validation split
        checkpoint_dir             : Directory to save best model weights
                                     (set to a Google Drive path in Colab)
        epochs, lr, batch_size     : Training hyper-parameters
        warmup_frac                : Fraction of total steps used for linear warmup
        max_len                    : Tokeniser max sequence length
        seed                       : Random seed for reproducibility

    Returns:
        Path to the best checkpoint directory.
    """
    torch = _import_torch()
    AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup = _import_transformers()

    # Reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Tokenise ──────────────────────────────────────────────────────────
    print(f"Tokenising (max_len={max_len}) ...")
    tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_ID)
    train_enc = _tokenize(tokenizer, train_texts, max_len)
    val_enc   = _tokenize(tokenizer, val_texts,   max_len)

    # ── 2. DataLoaders ───────────────────────────────────────────────────────
    from torch.utils.data import DataLoader
    train_ds = CitationDataset(train_enc, train_labels)
    val_ds   = CitationDataset(val_enc,   val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False)
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── 3. Model ─────────────────────────────────────────────────────────────
    print(f"Loading {SCIBERT_MODEL_ID} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        SCIBERT_MODEL_ID,
        num_labels=len(LABEL_NAMES),
    )
    model.to(device)

    # ── 4. Optimiser + scheduler ─────────────────────────────────────────────
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps  = len(train_loader) * epochs
    warmup_steps = math.ceil(total_steps * warmup_frac)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"  Total steps: {total_steps} | Warmup steps: {warmup_steps}")

    # ── 5 & 6. Training loop ─────────────────────────────────────────────────
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1   = -1.0
    best_ckpt_dir = checkpoint_dir

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)          # forward
            loss    = outputs.loss
            total_loss += loss.item()
            loss.backward()                   # backward
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()                  # step
            scheduler.step()
            optimizer.zero_grad()
            if step % 50 == 0 or step == len(train_loader):
                print(f"  Epoch {epoch}/{epochs} | step {step}/{len(train_loader)} "
                      f"| loss {total_loss / step:.4f}")

        # ---- validate ----
        val_f1, val_acc = _evaluate_loop(model, val_loader, device)
        print(f"  --> Epoch {epoch} val macro-F1: {val_f1:.4f}  acc: {val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1   = val_f1
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"  ** Best checkpoint saved (val F1={best_val_f1:.4f}) -> {checkpoint_dir}")

    print(f"\nTraining done. Best val macro-F1: {best_val_f1:.4f}")
    return checkpoint_dir


def _evaluate_loop(model, loader, device):
    """Run inference on a DataLoader; return (macro_f1, accuracy)."""
    torch   = _import_torch()
    from sklearn.metrics import f1_score as sk_f1

    model.eval()
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch  = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds  = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    macro_f1 = sk_f1(all_labels, all_preds, average="macro")
    accuracy  = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return macro_f1, accuracy


# -- Test evaluation -----------------------------------------------------------

def eval_scibert(
    test_texts, test_labels,
    checkpoint_dir=SCIBERT_CKPT_DIR,
    batch_size=SCIBERT_BATCH * 2,
    max_len=SCIBERT_MAX_LEN,
):
    """
    Load the best checkpoint and evaluate on the test split.
    Prints a full classification report and confusion matrix.
    Saves per-class precision/recall/F1 and macro F1 to results/scibert_intent.json
    in the same format as tfidf_intent.json.

    Returns:
        metrics dict (same structure as eval_tfidf returns).
    """
    torch = _import_torch()
    AutoTokenizer, AutoModelForSequenceClassification, _ = _import_transformers()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)

    print(f"Loading checkpoint from {checkpoint_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model     = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    model.to(device)

    from torch.utils.data import DataLoader
    enc    = _tokenize(tokenizer, test_texts, max_len)
    ds     = CitationDataset(enc, test_labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch  = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds  = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    report_str  = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, digits=4)
    report_dict = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, output_dict=True)
    macro_f1    = f1_score(all_labels, all_preds, average="macro")
    cm          = confusion_matrix(all_labels, all_preds)

    print("\n-- SciBERT Classification Report (test split) --")
    print(report_str)
    print(f"Macro F1: {macro_f1:.4f}\n")

    print("-- Confusion Matrix (rows=true, cols=predicted) --")
    header = f"{'':>12}" + "".join(f"{name:>12}" for name in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{LABEL_NAMES[i]:>12}" + "".join(f"{v:>12}" for v in row))
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "macro_f1": round(macro_f1, 6),
        "per_class": {
            label: {
                "precision": round(report_dict[label]["precision"], 6),
                "recall":    round(report_dict[label]["recall"], 6),
                "f1-score":  round(report_dict[label]["f1-score"], 6),
                "support":   int(report_dict[label]["support"]),
            }
            for label in LABEL_NAMES
        },
        "confusion_matrix": cm.tolist(),
        "checkpoint_dir": str(checkpoint_dir),
        "dataset": "allenai/scicite",
        "model": SCIBERT_MODEL_ID,
        "epochs": SCIBERT_EPOCHS,
        "lr": SCIBERT_LR,
        "batch_size": SCIBERT_BATCH,
        "max_len": SCIBERT_MAX_LEN,
    }
    SCIBERT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved -> {SCIBERT_METRICS}\n")
    return metrics


# -- Inference -----------------------------------------------------------------

def predict(sentences, checkpoint_dir=SCIBERT_CKPT_DIR, batch_size=64):
    """
    Predict citation intent from a saved SciBERT checkpoint.

    Args:
        sentences:      List of raw citation-context strings.
        checkpoint_dir: Path to the saved checkpoint (default: DATA_DIR/scibert_intent).
        batch_size:     Inference batch size.

    Returns:
        labels:      list[str] – predicted label for each sentence
                     ('background', 'method', or 'result').
        confidences: list[float] – softmax probability of the predicted class.
    """
    import torch
    from torch.utils.data import DataLoader
    AutoTokenizer, AutoModelForSequenceClassification, _ = _import_transformers()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model     = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    model.to(device)
    model.eval()

    if not sentences:
        return [], []

    dummy_labels = [0] * len(sentences)
    enc    = _tokenize(tokenizer, sentences, SCIBERT_MAX_LEN)
    ds     = CitationDataset(enc, dummy_labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_labels:  list[str]   = []
    all_confs:   list[float] = []

    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            batch   = {k: v.to(device) for k, v in batch.items()}
            logits  = model(**batch).logits
            probs   = torch.softmax(logits, dim=-1)
            top_p, top_i = probs.max(dim=-1)
            all_labels.extend([LABEL_NAMES[i] for i in top_i.cpu().tolist()])
            all_confs.extend(top_p.cpu().tolist())

    return all_labels, all_confs


def apply_to_corpus(
    edges_path=EDGES_PATH,
    output_path=EDGES_PRED_PATH,
    checkpoint_dir=SCIBERT_CKPT_DIR,
    batch_size=64,
    smoke=False,
    smoke_n=200,
):
    """
    Step 10: Run SciBERT citation-intent inference on context_sentence in edges.parquet.
    Saves edges_pred.parquet with predicted_intent and intent_confidence columns.

    Prints:
      1. Distribution of predicted_intent
      2. Cross-tabulation of predicted_intent against s2_intent (Semantic Scholar label)
      3. Agreement rate where s2_intent is present (mapping 'methodology' -> 'method')

    Args:
        edges_path:     Input parquet file path (default: data/edges.parquet).
        output_path:    Output parquet file path (default: data/edges_pred.parquet).
        checkpoint_dir: Path to fine-tuned SciBERT checkpoint (default: data/scibert_intent).
        batch_size:     Batch size for predict() (default: 64).
        smoke:          If True, run on the first smoke_n edges.
        smoke_n:        Number of edges in smoke mode (default: 200).

    Returns:
        pd.DataFrame with added columns: predicted_intent, intent_confidence.
    """
    import pandas as pd

    edges_path = Path(edges_path)
    output_path = Path(output_path)
    checkpoint_dir = Path(checkpoint_dir)

    print(f"Loading edges from {edges_path} ...")
    df = pd.read_parquet(edges_path)
    total_edges = len(df)
    print(f"Total edges loaded: {total_edges}")

    if smoke:
        n_edges = min(smoke_n, total_edges)
        print(f"\n[SMOKE MODE] Running inference on first {n_edges} edges...")
        df_target = df.head(smoke_n).copy()
    else:
        df_target = df.copy()

    # Extract context sentences
    sentences = df_target["context_sentence"].fillna("").astype(str).tolist()
    print(f"Running predict() on {len(sentences)} citation sentences (batch_size={batch_size}) ...")

    preds, confs = predict(sentences, checkpoint_dir=checkpoint_dir, batch_size=batch_size)

    df_target["predicted_intent"] = preds
    df_target["intent_confidence"] = [round(float(c), 6) for c in confs]

    # Save to edges_pred.parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_target.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df_target)} edges with predictions -> {output_path}")

    # 1. Distribution of predicted_intent
    n = len(df_target)
    counts = df_target["predicted_intent"].value_counts()
    print(f"\n-- Distribution of predicted_intent (n={n}) --")
    for intent, count in counts.items():
        pct = (count / n) * 100
        print(f"  {intent:<12} {count:>6}  ({pct:>5.1f}%)")

    # 2. Cross-tabulation: predicted_intent vs s2_intent
    if "s2_intent" in df_target.columns:
        print("\n-- Cross-tabulation: predicted_intent vs s2_intent --")
        ct = pd.crosstab(
            df_target["predicted_intent"],
            df_target["s2_intent"].fillna("(none)"),
            margins=True,
            margins_name="Total",
        )
        print(ct)

        # 3. Agreement rate where s2_intent is present
        # SciCite uses: background, method, result
        # Semantic Scholar uses: background, methodology, result
        has_s2 = df_target["s2_intent"].notna()
        s2_count = int(has_s2.sum())
        if s2_count > 0:
            s2_norm = df_target.loc[has_s2, "s2_intent"].replace({"methodology": "method"}).str.lower()
            pred_norm = df_target.loc[has_s2, "predicted_intent"].str.lower()
            matches = int((s2_norm == pred_norm).sum())
            agreement = matches / s2_count
            print(f"\nAgreement rate with s2_intent: {agreement:.2%} ({matches}/{s2_count} where s2_intent is non-null)")
            print("  (Note: Semantic Scholar 'methodology' mapped to 'method' for comparison)")
        else:
            print("\nAgreement rate: 0 edges with non-null s2_intent found.")
    else:
        print("\nNote: 's2_intent' column not found in input DataFrame; skipping cross-tabulation.")

    return df_target


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Citation-intent classifier: TF-IDF (Step 8), SciBERT (Step 9), or apply to edges (Step 10)."
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Step 10: Apply SciBERT intent classifier to edges.parquet and save edges_pred.parquet.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="In --apply mode, run on first 200 edges only.",
    )
    parser.add_argument(
        "--smoke-n", type=int, default=200,
        help="Number of edges to process in smoke mode (default: 200).",
    )
    parser.add_argument(
        "--edges-path", type=Path, default=EDGES_PATH,
        help=f"Path to input edges.parquet (default: {EDGES_PATH}).",
    )
    parser.add_argument(
        "--output-path", type=Path, default=EDGES_PRED_PATH,
        help=f"Path to output edges_pred.parquet (default: {EDGES_PRED_PATH}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Inference batch size (default: 64).",
    )
    parser.add_argument(
        "--scibert", action="store_true",
        help="Run SciBERT fine-tuning (Step 9) instead of TF-IDF (Step 8).",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; evaluate a saved model on the test split.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=SCIBERT_CKPT_DIR,
        help=f"SciBERT checkpoint directory (default: {SCIBERT_CKPT_DIR}). "
             "Set to a Google Drive path in Colab, e.g. /content/drive/MyDrive/scibert_intent",
    )
    args = parser.parse_args()

    if args.apply:
        # ── Step 10: Apply to corpus edges ────────────────────────────────────
        apply_to_corpus(
            edges_path=args.edges_path,
            output_path=args.output_path,
            checkpoint_dir=args.checkpoint_dir,
            batch_size=args.batch_size,
            smoke=args.smoke,
            smoke_n=args.smoke_n,
        )

    elif args.scibert:
        # ── SciBERT path ──────────────────────────────────────────────────────
        print("Loading allenai/scicite from HuggingFace datasets ...")
        ds_raw = load_dataset("allenai/scicite", trust_remote_code=True)
        train_texts  = [ex["string"] for ex in ds_raw["train"]]
        train_labels = [ex["label"]  for ex in ds_raw["train"]]
        val_texts    = [ex["string"] for ex in ds_raw["validation"]]
        val_labels   = [ex["label"]  for ex in ds_raw["validation"]]
        test_texts   = [ex["string"] for ex in ds_raw["test"]]
        test_labels  = [ex["label"]  for ex in ds_raw["test"]]

        print_class_balance(train_labels, "train")

        if not args.eval_only:
            train_scibert(
                train_texts, train_labels,
                val_texts,   val_labels,
                checkpoint_dir=args.checkpoint_dir,
            )

        eval_scibert(test_texts, test_labels, checkpoint_dir=args.checkpoint_dir)

    else:
        # ── TF-IDF path ───────────────────────────────────────────────────────
        train_texts, train_labels, test_texts, test_labels = load_scicite()
        print_class_balance(train_labels, split_name="train")

        if args.eval_only:
            print(f"Loading saved model from {MODEL_PATH} ...")
            vec, clf = load_model()
        else:
            vec, clf = train_tfidf(train_texts, train_labels)

        eval_tfidf(vec, clf, test_texts, test_labels)
