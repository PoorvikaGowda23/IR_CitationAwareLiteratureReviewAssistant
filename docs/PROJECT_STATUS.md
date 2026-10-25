# Citation-Aware Literature Review Assistant: Project Status

Written for: you and your lab partner, as a record of what is built and what the numbers say.

## 1. Goal

An Information Retrieval lab project in the domain of **graph neural networks for recommender systems**.

1. Retrieve papers for a query with SPECTER2 dense embeddings and a FAISS index.
2. Re-rank them with a **citation graph whose edges are typed by citation intent** (background, method, result, extension, contradiction).
3. Have an LLM (Groq `openai/gpt-oss-120b`) write a cited literature review from the top papers.
4. Evaluate against BM25 and a dense-only baseline, run an ablation study, and measure the faithfulness of the generated claims.

The step plan comes from `Antigravity Prompt Step-by-Step Implementation.pdf` (Steps 0-17).

## 2. Repository layout

| File | Purpose |
|---|---|
| `config.py` | Constants, `.env` loader, intent weights, final re-ranker weights |
| `schema.py` | Column constants and parquet/JSON load/save helpers |
| `scrape.py` | Semantic Scholar scraping, 1-hop corpus expansion, missing-edge fill |
| `embed.py` | SPECTER2 embeddings (base plus document and query adapters) and FAISS `IndexFlatIP` |
| `retrieval_baseline.py` | Dense and BM25 baselines |
| `section_refs.py`, `tools/bootstrap_relevant.py` | Build ground truth for section queries from survey references |
| `evaluate.py` | Ground truth, val/test split, nDCG/MAP/P@k/Recall@k, bootstrap CIs |
| `intent_classifier.py` | TF-IDF, SciBERT and extended 3-class intent classifiers; apply to corpus |
| `fewshot_llm.py` | Few-shot LLM intent baseline, hand-labelling helpers |
| `graph_builder.py` | Weighted directed citation graph, PageRank, co-citation, Louvain |
| `retrieval.py` | Two-stage hybrid re-ranker with contradiction surfacing |
| `tune.py`, `sanity_checks.py` | Weight grid search and diagnostics (val only) |
| `run_ablations.py` | Final test-split ablation |
| `generate_baseline.py`, `generate.py` | Plain RAG review vs. citation-aware (clustered, intent-ordered) review |
| `faithfulness.py` | Claim sampling, blinded judging sheet, LLM judge, Cohen's kappa |
| `tests/` | Unit tests for schema, metrics, graph builder, retrieval |
| `notebooks/` | Colab guides and notebook for embedding and SciBERT training |

## 3. Pipeline and step status

| Step | What | Status |
|---|---|---|
| 0-3 | Scaffold, schema, scraping, corpus of 1500 papers and 2034 edges | Done |
| 4 | SPECTER2 embeddings and FAISS index | Done |
| 5-7 | Baselines, ground truth, evaluation harness, tests | Done |
| 8-10 | TF-IDF and SciBERT intent classifiers, applied to all edges | Done |
| 11 | Citation graph | Done |
| 12 | Two-stage graph re-ranker | Done |
| 13 | Weight tuning, final weights | Done |
| 14a/14b | Baseline and citation-aware review generation | Done |
| 15 | Faithfulness evaluation | Done (40 claims, 20 per system; second labels by Claude, not a human) |
| 16 | Ablation study on test split | Done |
| 17a | Few-shot LLM baseline and hand labels | Done |
| 17b | Extended intent classes (extension, contradiction) | Done |

## 4. Data

- **Corpus:** 1500 papers (1997-2026), 3 survey papers excluded from retrieval. SPECTER2 embeddings are 1500 x 768.
- **Edges:** 2034 citation-context edges. Semantic Scholar intent: background 909, methodology 665, result 38.
- **Ground truth:** 8 queries, 3 val and 5 test.

| Query | Split | Relevant papers |
|---|---|---|
| s1_session | val | 81 |
| s2_kg | val | 23 |
| survey_b | val | 71 |
| s1_gcf | test | 90 |
| s2_contrastive | test | 11 |
| s3_dynamic | test | 16 |
| survey_a | test | 231 |
| survey_c | test | 253 |

  `s1_lightgcn` and `s3_hypergraph` were dropped because they had fewer than 8 relevant papers.
- **Hand labels:** 300 edges, labelled other 252, contradiction 34, extension 14.
- **Citation graph:** 373 nodes, 613 edges, 7 Louvain communities. Edge intents: method 368, background 159, result 38, contradiction 33, extension 15. Only 373 of the 1500 papers have any edge in the graph.
- **Predicted intent over all edges (SciBERT, after the label fix):** method 1015, background 879, result 140. Extended classes: other 1902, contradiction 91, extension 41.

## 5. Results

### 5.1 Intent classification on SciCite test (1859 sentences)

| Model | Macro-F1 | Background F1 | Method F1 | Result F1 |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.7967 | 0.838 | 0.823 | 0.729 |
| SciBERT (3 epochs, lr 2e-5) | **0.8331** | 0.866 | 0.860 | 0.773 |
| Few-shot gpt-oss-120b (300-sentence sample) | 0.7048 | 0.783 | 0.796 | 0.536 |

Fine-tuned SciBERT beats both alternatives. Result is the weakest class for every model, mostly from low precision (0.62-0.64 for the two trained models).

**Extended classes** (hand-labelled data, 5-fold CV, not comparable with SciCite):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| extension | 0.50 | 0.38 | 0.41 | 14 |
| contradiction | 0.70 | 0.70 | 0.65 | 34 |
| other | 0.92 | 0.94 | 0.92 | 252 |

Macro-F1 is 0.662. The extension class is very weak because it has only 14 examples.

### 5.2 Weight tuning (val split, 206-point grid)

| alpha (embedding) | beta (PageRank) | gamma (proximity) | delta (co-citation) | nDCG@10 | MAP |
|---|---|---|---|---|---|
| 0.0 | 0.4 | 0.0 | 0.6 | 0.4414 | 0.1025 |
| 0.1 | 0.0 | 0.3 | 0.6 | 0.4402 | 0.0931 |
| **0.1** | **0.1** | **0.3** | **0.5** | **0.4385** | **0.0937** |

The three top grid rows tie at 0.4414 but set the embedding weight to 0. The chosen **final weights are alpha 0.1, beta 0.1, gamma 0.3, delta 0.5** (the fifth-ranked row), picked to keep the embedding term in the hybrid. Intent weights: background 0.3, all other intents 1.0.

### 5.3 Test-split ablation (5 queries, mean nDCG@10 with 95% bootstrap CI)

| Variant | nDCG@10 | MAP | P@10 | Delta vs dense | p-value |
|---|---|---|---|---|---|
| BM25 | 0.1216 [0.030, 0.214] | 0.0252 | 0.14 | +0.005 | 0.44 |
| Dense baseline | 0.1167 [0.000, 0.278] | 0.0104 | 0.10 | 0 | n/a |
| **Full graph re-ranker** | **0.2092** [0.052, 0.367] | 0.0158 | 0.18 | **+0.0925** | **0.0055** |
| No graph signal | 0.1167 | 0.0104 | 0.10 | 0 | 0.50 |
| Untyped edges | 0.2092 | 0.0159 | 0.18 | +0.0925 | 0.0055 |
| No PageRank | 0.1935 | 0.0149 | 0.16 | +0.0769 | 0.046 |
| No proximity | 0.2734 | 0.0196 | 0.22 | +0.1567 | 0.0 |
| No co-citation | 0.1828 | 0.0130 | 0.18 | +0.0661 | 0.0055 |
| No contradiction surfacing | 0.2092 | 0.0156 | 0.18 | +0.0925 | 0.0055 |

Recall@50 is 0.0777 for every dense-based variant (re-ranking only reorders the stage-1 top 50) and 0.1282 for BM25. Outputs: `results/ablation_table.md`, `.csv`, `ablation.png`.

**Reading the table:**
- The graph re-ranker nearly doubles dense nDCG@10 (0.117 to 0.209).
- Untyped edges score the same as typed edges, so there is no measured benefit from intent typing.
- Contradiction surfacing makes no measurable difference.
- Removing proximity *improves* nDCG@10 to 0.273, even though proximity got weight 0.3 on val. This needs an explanation in the write-up.
- The re-ranker's MAP (0.0158) is below BM25's (0.0252).

### 5.4 Faithfulness (Step 15)

40 claims (20 per system) from 5 queries, judged by `openai/gpt-oss-20b`; reviews written by `openai/gpt-oss-120b`.

| | Baseline | Citation-aware |
|---|---|---|
| LLM judge, strict (supported) | 35% | 25% |
| LLM judge, lenient (supported + partial) | 55% | 45% |
| Second labels (written by Claude), strict / lenient | 45% / 70% | 45% / 70% |

Cohen's kappa between the second labels and the judge is 0.40. Those labels are model-written, not human. 12 of 40 claims cite papers with no abstract in the data. Conclusion: no measurable faithfulness gain for the citation-aware review. Details: `results/LIMITATIONS_AND_NOTES.md`.

## 6. Bug found and fixed: SciCite label order

- The HF label order is `['method', 'background', 'result']`, but the code assumed `['background', 'method', 'result']`.
- **Effect:** model scores did not change (SciBERT macro-F1 0.8331 both times) because training used integer ids. Class names in reports, `predicted_intent` in `edges_pred.parquet` (background 1015 / method 879 before; method 1015 / background 879 after), and therefore the intent weights on graph edges were swapped.
- **Fix (commit `75838b7`)**, then graph rebuilt and tuning re-run (commit `86785ea`). The top tuning row stayed at nDCG@10 0.4414, and MAP moved only slightly (0.1023 to 0.1025).
- Pre-fix copies are kept in `results/archive/` as `*_before_swapfix.*`.

## 7. Caveats and open issues

- **Small evaluation:** 5 test queries, so confidence intervals are wide. Two of the 5 test queries (`survey_a`, `survey_c`) have 231 and 253 relevant papers, which come from survey reference lists. The sections' ground truth is noisy.
- **Small graph:** 373 of 1500 papers have edges, which limits what graph signals can do.
- **Fixed:** edges now store `confidence`; `--no-cache` bypasses the cache in both generators; `fewshot_llm.load_scicite` docstring corrected; the stray root `scibert_intent.json` is deleted; the confusion matrices in `results/*_intent.json` were already in id order (method, background, result) and now carry a `confusion_matrix_labels` key saying so.
- **TL;DR is nearly empty:** `generate.py` now receives `tldr`, but only 17 of 1500 papers (1.1%) have one, so the prompt still falls back to 200 characters of the abstract for almost every paper.
- **Anchor self co-citation:** `compute_cocitation_with_anchors` counts an anchor against itself, so anchors that have citers get a co-citation boost. Left as is, since changing it alters the tuned and ablation results.
- **Git:** many untracked result files (fewshot cache, judge cache, before-swapfix copies, `data/extended_intent/`) and a modified `judging_sheet.csv`.

## 8. Remaining work

1. Optional: have a person relabel the faithfulness sheet (at least the 16 judge/label disagreements).
2. Produce one traced sample review with its claim-to-source mapping.
3. Write the limitations section (noisy ground truth, partial context coverage, small hand-labelled set, small corpus).
4. Explain the ablation oddities (proximity, untyped equal to typed).
5. Fix stale confusion matrices, delete the stray file, commit the untracked results.

## 9. How to run

Run from the repo root with `.env` holding `S2_API_KEY` and `GROQ_API_KEY`. Run `pytest` for the unit tests.

```
pip install -r requirements.txt groq

python scrape.py --seeds-only ; python scrape.py ; python scrape.py --fill-missing-edges
python embed.py                                   # GPU/Colab recommended
python section_refs.py --apply ; python evaluate.py --build
python evaluate.py --split val                    # dense vs BM25

python intent_classifier.py                       # TF-IDF
python intent_classifier.py --scibert             # Colab T4
python intent_classifier.py --apply
python fewshot_llm.py --eval --n 300
python fewshot_llm.py --export --n 300 ; python fewshot_llm.py --load-labels
python intent_classifier.py --train-extended ; python intent_classifier.py --apply-extended

python graph_builder.py
python retrieval.py --query "..." -k 10
python tune.py ; python sanity_checks.py          # val only
python run_ablations.py                           # final test run

python generate_baseline.py --query "..." -k 10
python generate.py --query "..." -k 10
python faithfulness.py --export ; python faithfulness.py --judge --yes
python faithfulness.py --agreement annotated.csv
```

Order matters for the re-ranker: `--apply`, then `--apply-extended`, then `graph_builder.py`, then retrieval and tuning.

## 10. Interpreting the ablation (for the write-up)

**Why the graph re-ranker beats dense (0.117 to 0.209).** Stage 1 returns 50 candidates; the re-ranker then weights co-citation 0.5 and proximity 0.3 against embedding 0.1. Papers that cite or are co-cited with the top-5 anchors are pulled up. This works only for the 373 papers that have graph edges; the other candidates score 0 on three of four signals.

**Why removing proximity helps (0.209 to 0.273).** Proximity was given weight 0.3 because it scored well on the validation split (3 queries). On the test split it adds noise: it rewards any candidate within 3 hops of an anchor, and with a small, dense graph that covers most graph-connected candidates equally. This is a val/test mismatch with very few queries, not evidence that proximity is useless. Say so, and do not retune on test.

**Why typed edges equal untyped edges.** The only intent weight that differs from 1.0 is background (0.3). Co-citation and proximity ignore edge weight; only PageRank uses it, and PageRank has a small weight (0.1). So typing can barely change the ranking. The honest conclusion is that intent typing is not shown to help in this setup.

**Why contradiction surfacing changes nothing.** It appends up to 2 papers after the top-k, and nDCG@10 only looks at the top 10. It is a presentation feature, not a ranking feature, so the metric cannot see it.

**Why MAP is below BM25.** The re-ranker only reorders the stage-1 top 50, so Recall@50 is capped at 0.078 (BM25: 0.128). MAP rewards finding many relevant papers deep in the list, which the cap prevents.

## 11. Limitations

- **Tiny evaluation:** 5 test queries. The bootstrap intervals overlap (full re-ranker [0.052, 0.367] vs dense [0.000, 0.278]); only the paired p-value (0.0055) supports the improvement.
- **Noisy ground truth:** relevance comes from survey reference lists. `survey_a` and `survey_c` have 231 and 253 relevant papers, many only loosely relevant. Papers that are relevant but not in a survey count as wrong.
- **Sparse graph:** 373 of 1500 papers have any edge; 613 edges in total. Graph signals cannot help the other 1127 papers.
- **Small hand-labelled set:** 300 edges, with 14 extension and 34 contradiction examples. The extension class has F1 0.41 and is unreliable.
- **Intent classifier domain shift:** trained on SciCite, applied to citation contexts from a different area (GNN recommenders). The Result class has low precision (0.62-0.64).
- **Generation shows no faithfulness gain:** the 40-claim evaluation found the citation-aware review no more faithful than the baseline, and its second labels are not from a human.
- **Single domain and one LLM:** all results are for graph neural networks in recommender systems, with one generator model (gpt-oss-120b).
