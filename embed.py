"""
SPECTER2 document and query embedding generator using task adapters and FAISS indexing.

Functions:
- build_text: concatenates title and abstract with tokenizer sep_token.
- embed_papers: embeds paper corpus with 'proximity' adapter, length-sorting, dynamic padding.
- embed_query: embeds a search query using the 'adhoc_query' adapter.
- build_index: creates a faiss.IndexFlatIP index from L2-normalized vectors.
- load_index: loads saved FAISS index and paper ID list from data/.
"""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

import config
import schema

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# HuggingFace adapter identifiers for SPECTER2
PROXIMITY_ADAPTER: str = "allenai/specter2"
ADHOC_QUERY_ADAPTER: str = "allenai/specter2_adhoc_query"

# Module-level model/tokenizer cache to avoid reloading weights across function calls
_CACHED_TOKENIZER: AutoTokenizer | None = None
_CACHED_MODEL: AutoAdapterModel | None = None
_CACHED_DEVICE: torch.device | None = None


# ── 1. Model & Tokenizer Loader ────────────────────────────────────────────────

def get_device() -> torch.device:
    """Detect and return CUDA GPU device if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        log.info("CUDA not available; using CPU")
    return device


def load_model_and_tokenizer(device: torch.device | None = None) -> tuple[AutoTokenizer, AutoAdapterModel]:
    """Load SPECTER2 base model with 'proximity' and 'adhoc_query' adapters.

    Caches instances at module level so repeated calls reuse weights.

    Args:
        device: Target torch device. If None, auto-detected via get_device().

    Returns:
        Tuple of (tokenizer, adapter_model).
    """
    global _CACHED_TOKENIZER, _CACHED_MODEL, _CACHED_DEVICE

    if device is None:
        device = get_device()

    # Reuse cached instances if device matches
    if _CACHED_MODEL is not None and _CACHED_TOKENIZER is not None and _CACHED_DEVICE == device:
        return _CACHED_TOKENIZER, _CACHED_MODEL

    log.info("Loading base model: %s", config.SPECTER_BASE)
    tokenizer = AutoTokenizer.from_pretrained(config.SPECTER_BASE)
    model = AutoAdapterModel.from_pretrained(config.SPECTER_BASE)

    # Load and register both specialized adapters
    log.info("Loading adapter: %s (load_as='proximity')", PROXIMITY_ADAPTER)
    model.load_adapter(PROXIMITY_ADAPTER, source="hf", load_as="proximity", set_active=True)

    log.info("Loading adapter: %s (load_as='adhoc_query')", ADHOC_QUERY_ADAPTER)
    model.load_adapter(ADHOC_QUERY_ADAPTER, source="hf", load_as="adhoc_query")

    model.to(device)
    model.eval()

    _CACHED_TOKENIZER = tokenizer
    _CACHED_MODEL = model
    _CACHED_DEVICE = device

    return tokenizer, model


# ── 2. Text Builder ────────────────────────────────────────────────────────────

def build_text(row: pd.Series | dict[str, Any], sep_token: str) -> str:
    """Concatenate paper title and abstract separated by tokenizer.sep_token.

    Args:
        row:       Pandas Series or dictionary containing schema.TITLE and schema.ABSTRACT.
        sep_token: Separator string from the tokenizer (e.g. '[SEP]').

    Returns:
        Formatted string: "<title> <sep_token> <abstract>".
    """
    title = str(row.get(schema.TITLE) or "").strip()
    abstract = str(row.get(schema.ABSTRACT) or "").strip()

    if title and abstract:
        return f"{title} {sep_token} {abstract}"
    elif title:
        return title
    return abstract


# ── 3. Embed Papers Corpus ─────────────────────────────────────────────────────

def embed_papers(
    papers_df: pd.DataFrame,
    batch_size: int = 16,
    checkpoint_interval: int = 200,
    device: torch.device | None = None,
) -> np.ndarray:
    """Embed paper title + abstract pairs using SPECTER2 proximity adapter.

    Sorts texts by length for dynamic padding efficiency, saves partial checkpoints
    every checkpoint_interval papers, and restores original DataFrame order.

    Args:
        papers_df:           DataFrame containing schema.TITLE, schema.ABSTRACT, schema.PAPER_ID.
        batch_size:          Inference batch size (default 16).
        checkpoint_interval: Save partial array to data/ every N papers.
        device:              Torch computation device.

    Returns:
        float32 2D numpy array of shape (len(papers_df), hidden_dim), L2-normalized.
    """
    if device is None:
        device = get_device()

    tokenizer, model = load_model_and_tokenizer(device)
    # Activate paper proximity adapter for document-to-document representation
    model.set_active_adapters("proximity")

    num_papers = len(papers_df)
    log.info("Preparing text inputs for %d papers", num_papers)

    # 1. Build text representations
    texts = [build_text(row, tokenizer.sep_token) for _, row in papers_df.iterrows()]

    # 2. Sort by text length to minimize padding overhead within each batch
    lengths = [len(t) for t in texts]
    sorted_indices = sorted(range(num_papers), key=lambda i: lengths[i])
    sorted_texts = [texts[i] for i in sorted_indices]

    # Pre-allocate array for sorted embeddings
    hidden_dim = model.config.hidden_size
    sorted_embeddings = np.zeros((num_papers, hidden_dim), dtype=np.float32)

    log.info("Embedding %d papers in batches of %d...", num_papers, batch_size)
    start_time = time.time()

    # Progress bar with estimated remaining time
    pbar = tqdm(total=num_papers, desc="SPECTER2 Papers", unit="paper")

    checkpoint_file = config.DATA_DIR / "embeddings_partial.npy"

    for i in range(0, num_papers, batch_size):
        batch_texts = sorted_texts[i : i + batch_size]

        # Dynamic padding: pads only to the longest sequence in this specific batch
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # SPECTER2 uses the [CLS] representation (index 0)
            cls_repr = outputs.last_hidden_state[:, 0, :]
            batch_vecs = cls_repr.cpu().numpy().astype(np.float32)

        # L2-normalize each vector in batch so ||v||_2 = 1.0
        norms = np.linalg.norm(batch_vecs, axis=1, keepdims=True)
        # Avoid division by zero on empty edge cases
        batch_vecs = np.divide(batch_vecs, np.maximum(norms, 1e-12))

        sorted_embeddings[i : i + len(batch_texts)] = batch_vecs
        pbar.update(len(batch_texts))

        # Checkpoint partial results every checkpoint_interval papers
        if (i + len(batch_texts)) % checkpoint_interval < batch_size and (i + len(batch_texts)) > 0:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            np.save(checkpoint_file, sorted_embeddings[: i + len(batch_texts)])
            log.info("Checkpoint saved: %d papers → %s", i + len(batch_texts), checkpoint_file)

    pbar.close()
    elapsed = time.time() - start_time
    log.info("Embedding finished in %.2f seconds (%.2f papers/sec)", elapsed, num_papers / max(elapsed, 0.001))

    # Clean up partial checkpoint if run completed cleanly
    if checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
        except OSError:
            pass

    # 3. Invert the sort permutation to restore original papers_df row order
    original_embeddings = np.zeros_like(sorted_embeddings)
    for sorted_pos, orig_idx in enumerate(sorted_indices):
        original_embeddings[orig_idx] = sorted_embeddings[sorted_pos]

    return original_embeddings


# ── 4. Embed Adhoc Search Query ────────────────────────────────────────────────

def embed_query(
    text: str,
    device: torch.device | None = None,
) -> np.ndarray:
    """Embed a search query string using the SPECTER2 adhoc query adapter.

    Args:
        text:   Natural-language search query string.
        device: Torch computation device.

    Returns:
        float32 2D numpy array of shape (1, hidden_dim), L2-normalized.
    """
    if device is None:
        device = get_device()

    tokenizer, model = load_model_and_tokenizer(device)
    # Activate adhoc_query adapter for user search queries
    model.set_active_adapters("adhoc_query")

    inputs = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        cls_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)

    # L2-normalize query vector
    norm = np.linalg.norm(cls_vec, axis=1, keepdims=True)
    norm_vec = np.divide(cls_vec, np.maximum(norm, 1e-12))

    return norm_vec.astype(np.float32)


# ── 5. FAISS Index Builder & Loader ────────────────────────────────────────────

def build_index(matrix: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS IndexFlatIP (Inner Product) index over the embedding matrix.

    Mathematical equivalence note:
    Cosine similarity between vectors u and v is:
        cos(u, v) = (u · v) / (||u||_2 * ||v||_2)
    Because all vectors in `matrix` are L2-normalized (||u||_2 = 1 and ||v||_2 = 1),
    the denominator equals 1.0. Therefore:
        cos(u, v) = u · v = Inner Product
    This allows faiss.IndexFlatIP to compute exact cosine similarity via fast
    SIMD-accelerated dot products without square root operations during search.

    Args:
        matrix: 2D numpy float32 array of shape (num_papers, dim), L2-normalized.

    Returns:
        faiss.IndexFlatIP populated with the vectors.
    """
    num_vectors, dim = matrix.shape
    # Ensure C-contiguous float32 array as required by FAISS
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    log.info("Built faiss.IndexFlatIP with %d vectors (dimension=%d)", num_vectors, dim)
    return index


def load_index(data_dir: Path | str = config.DATA_DIR) -> tuple[faiss.Index, list[str]]:
    """Load the FAISS index and paper ID mapping from disk.

    Args:
        data_dir: Path to directory containing faiss.index and ids.json.

    Returns:
        Tuple of (faiss_index, list_of_paper_ids) in corresponding row order.
    """
    data_dir = Path(data_dir)
    index_path = data_dir / "faiss.index"
    ids_path = data_dir / "ids.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index file not found at: {index_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Paper IDs file not found at: {ids_path}")

    index = faiss.read_index(str(index_path))
    with open(ids_path, "r", encoding="utf-8") as f:
        paper_ids = json.load(f)

    if index.ntotal != len(paper_ids):
        raise ValueError(
            f"FAISS index count ({index.ntotal}) does not match IDs count ({len(paper_ids)})"
        )

    return index, paper_ids


# ── Main block ─────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point: embed corpus, build FAISS index, and save artifacts."""
    parser = argparse.ArgumentParser(description="Embed papers with SPECTER2 and build FAISS index.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test mode: embed only the first 20 papers.",
    )
    args = parser.parse_args()

    overall_start = time.time()

    # Determine input file: prefer expanded papers.parquet; fallback to seed_papers.parquet
    papers_path = config.DATA_DIR / "papers.parquet"
    if not papers_path.exists():
        papers_path = config.DATA_DIR / "seed_papers.parquet"
    if not papers_path.exists():
        raise FileNotFoundError(
            f"No input papers found in {config.DATA_DIR}. Run scrape.py first!"
        )

    log.info("Loading papers from %s", papers_path)
    df = schema.load_papers(papers_path)

    if args.smoke:
        log.info("--smoke flag set: selecting first 20 papers")
        df = df.iloc[:20].reset_index(drop=True)

    # 1. Generate embeddings
    embeddings = embed_papers(df, batch_size=16, checkpoint_interval=200)

    # 2. Build FAISS index
    index = build_index(embeddings)

    # 3. Extract IDs in matching row order
    paper_ids: list[str] = df[schema.PAPER_ID].tolist()

    # 4. Save artifacts to data/
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_file = config.DATA_DIR / "embeddings.npy"
    ids_file = config.DATA_DIR / "ids.json"
    index_file = config.DATA_DIR / "faiss.index"

    np.save(embeddings_file, embeddings)
    with open(ids_file, "w", encoding="utf-8") as f:
        json.dump(paper_ids, f, indent=2)
    faiss.write_index(index, str(index_file))

    total_time = time.time() - overall_start

    print("\n================ EMBEDDING SUMMARY ================")
    print(f"Papers embedded:       {len(df)}")
    print(f"Embedding matrix:      {embeddings.shape} (dtype: {embeddings.dtype})")
    print(f"FAISS index entries:   {index.ntotal}")
    print(f"Saved embeddings to:   {embeddings_file}")
    print(f"Saved IDs mapping to:  {ids_file}")
    print(f"Saved FAISS index to:  {index_file}")
    print(f"Total elapsed time:    {total_time:.2f} seconds")
    print("===================================================\n")


if __name__ == "__main__":
    main()
