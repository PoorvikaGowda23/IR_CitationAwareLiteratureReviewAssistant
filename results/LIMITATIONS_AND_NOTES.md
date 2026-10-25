# Notes, interpretation and limitations

## 1. How contradiction-flagged papers are scored (step 13 decision)

`retrieve(..., surface_contradictions=True)` appends up to 2 papers after the top-k. They carry `flags = ['contradiction', <id of the paper they contradict>]`.

The evaluation harness (`evaluate.evaluate`, called with k = 50) therefore sees them at ranks 51-52. Current behaviour:

| Metric | Do surfaced papers count? |
|---|---|
| nDCG@10, P@10 | No (outside the top 10) |
| Recall@50 | No (outside the top 50) |
| MAP | Yes, marginally (average precision runs over the full list) |

The two options the PDF asks to choose between:
1. **Count flagged papers like any other result.** Rewards surfacing only if the paper is topically relevant; ignores the purpose (showing counter-evidence).
2. **Exclude them from the ranking metrics and report separately** (e.g. how many flagged papers per query, and how many of them are relevant).

The implemented behaviour is closest to option 2 for nDCG/P/recall. **This is a project decision that has not been confirmed by the team; confirm or change it before submission.** The consequence for the ablation: "No contradiction surfacing" is nearly identical to the full model (nDCG@10 0.2092 vs 0.2092, MAP 0.0156 vs 0.0158), so the ablation cannot show any benefit of surfacing. Its value is qualitative (shown in the UI and the reviews), not measured.

## 2. How to read the retrieval results (test split, 5 queries)

Source: `results/ablation_table.md`. 95% bootstrap CIs are very wide.

- Full graph re-ranker nDCG@10 = 0.209 vs dense baseline 0.117 (+0.093, one-sided paired-bootstrap p ≈ 0.006). BM25 (0.122) is statistically tied with dense.
- **Typed vs untyped edges: no measurable difference** (0.2092 vs 0.2092). Edge intent only changes edge weights, which only enter through weighted PageRank (BETA = 0.1). Proximity and co-citation ignore the weights. So this project cannot claim that intent typing helps retrieval.
- **Signal contributions (drop one signal at a time):** removing co-citation costs the most (-0.026), then PageRank (-0.016). Removing proximity *raises* nDCG@10 to 0.273, so the proximity weight is not clearly useful; the CIs overlap, so this is not a firm conclusion.
- **Recall@50 is 0.078 for every re-ranker.** They all re-order the same 50 stage-1 candidates, so the candidate pool, not the re-ranking, is the ceiling.
- The weights are grid-search "Row 4" (0.1 / 0.1 / 0.3 / 0.5), chosen on validation. It is not the best validation row: the top rows set ALPHA = 0, which would discard the embedding score. Choosing Row 4 keeps the method a hybrid.
- Validation nDCG@10 was about 0.44 but test is 0.21: tuning on only 3 validation queries overfits.

## 3. Faithfulness of generated reviews (step 15)

40 claims (20 per system) from 5 queries, judged by an LLM (`openai/gpt-oss-20b`; the reviews were written by `openai/gpt-oss-120b`).

| | Baseline | Citation-aware |
|---|---|---|
| LLM judge, strict (supported) | 35% | 25% |
| LLM judge, lenient (supported + partial) | 55% | 45% |
| Claude labels, strict / lenient | 45% / 70% | 45% / 70% |

- **Manual labels were written by Claude (blind to the judge), not by a person.** Cohen's kappa between those labels and the judge is **0.40** (moderate). It is a model-vs-model figure, not human validation. A human re-label of at least the 16 disagreements is recommended.
- 12 of 40 claims cite papers with **no abstract** in the data; both the judge and the labels must call these unsupported, which lowers every rate. On the 28 claims that have an abstract: Claude labels 64% / 100% for both systems; judge 50% / 79% (baseline) vs 36% / 64% (citation-aware).
- Conclusion to state: **no measurable faithfulness improvement for the citation-aware reviews.** With 20 claims per system, differences of 10-15 points are within noise.
- Only about a third of retrieved papers receive a real citation-context sentence (the rest fall back to TLDR/abstract), which limits how different the citation-aware prompt is from the baseline.
- Reviews are capped at 2,048 output tokens for both systems (fair comparison), so some end mid-sentence.

## 4. Limitations

- **Tiny, noisy ground truth:** 8 queries in total (3 validation, 5 test). Relevance comes from survey reference lists, so a relevant paper that the survey did not cite counts as irrelevant. Confidence intervals are very wide and most differences are not robust.
- **Small corpus:** 1,500 papers and 2,034 citation edges after filtering; only 373 papers appear in the graph, so many candidates have no graph signal.
- **Citation-context coverage:** every kept edge has a context sentence by construction, but many papers have no incoming context; Semantic Scholar's own intent label exists for only 79% of edges.
- **Small hand-labelled set** for extension/contradiction: 300 sentences, 14 extension and 34 contradiction examples; scores are noisy (macro F1 0.66, extension F1 0.41).
- **Classifier transfer:** SciCite is trained on general scientific text; applying it to this recommender-systems corpus is unvalidated beyond agreement checks with Semantic Scholar labels.
- **Intent typing did not help retrieval** in this experiment (see 2).
- **Faithfulness judged by LLMs** on abstracts only (truncated to 600 characters for the judge), with a small sample and no independent human labels.
- **Cost / rate limits:** all LLM calls use the Groq free tier; reviews and judgments are cached on disk for reproducibility.
