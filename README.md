# Citation-Aware Literature Review Assistant

IR lab project on graph neural networks for recommender systems. SPECTER2 + FAISS retrieves candidate papers, a citation graph with intent-typed edges re-ranks them, and an LLM (Groq `openai/gpt-oss-120b`) writes a cited literature review. Evaluated against BM25 and dense baselines, with an ablation study and a faithfulness check.

Full status, results and limitations: [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) and [results/LIMITATIONS_AND_NOTES.md](results/LIMITATIONS_AND_NOTES.md).

## Setup

```
pip install -r requirements.txt
```

Create `.env` in the repo root with `S2_API_KEY` and `GROQ_API_KEY`.

Not in git (too large, rebuild or copy them): `data/scibert_intent/` (fine-tuned SciBERT, see `notebooks/scibert_colab.md`) and `data/extended_intent/` (`python intent_classifier.py --train-extended`).

## Demo

```
python -m streamlit run app.py
```

Use the first preset query ("Session-based recommendation with graph neural networks"); both of its reviews are cached, so no API call is needed.

## Layout

| Path | Contents |
|---|---|
| `config.py`, `schema.py` | Constants and data helpers |
| `scrape.py`, `embed.py` | Corpus building, SPECTER2 embeddings, FAISS index |
| `retrieval_baseline.py`, `retrieval.py` | BM25 / dense baselines, hybrid graph re-ranker |
| `intent_classifier.py`, `fewshot_llm.py` | Intent classifiers (TF-IDF, SciBERT, extended, few-shot LLM) |
| `graph_builder.py` | Citation graph, PageRank, co-citation, communities |
| `evaluate.py`, `tune.py`, `run_ablations.py`, `sanity_checks.py` | Metrics, weight tuning (val), ablation (test) |
| `generate_baseline.py`, `generate.py`, `faithfulness.py` | Review generation and faithfulness evaluation |
| `tests/` | Unit tests (`pytest`) |
| `tools/`, `notebooks/`, `docs/` | Ground-truth helper, Colab guides, project documentation |
| `results/` | Tables, plots, caches; `results/archive/` holds pre-bug-fix copies |

Run order and all commands: section 9 of [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md).
