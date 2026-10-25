# Citation-intent classifiers

## A. SciCite (background / method / result), test split

| Classifier | Test set | Macro F1 | F1 background | F1 method | F1 result | Result precision / recall |
|---|---|---|---|---|---|---|
| TF-IDF (1-2 grams) + Logistic Regression | full test, n = 1,859 | 0.797 | 0.838 | 0.823 | 0.729 | 0.624 / 0.876 |
| **SciBERT fine-tuned (3 epochs, lr 2e-5)** | full test, n = 1,859 | **0.833** | 0.866 | 0.860 | 0.773 | 0.638 / 0.981 |
| Few-shot LLM (`openai/gpt-oss-120b`, 6 shots) | seeded sample, n = 300 | 0.705 | 0.783 | 0.795 | 0.536 | 0.473 / 0.619 |

Notes
- SciBERT beats TF-IDF on macro F1 (+0.036), mainly on `background` and `result`. The few-shot LLM is lowest, especially on the rare `result` class.
- The few-shot number is on a 300-sentence sample (42 `result` sentences), so it is noisier and **not strictly comparable** with the two full-test rows.
- Macro F1 is the headline metric because the classes are imbalanced (`result` is the smallest class).

## B. Extension / contradiction / other (separate from SciCite)

Hand-labelled edges from this project's own corpus (300 sentences: 252 other, 34 contradiction, 14 extension). SciBERT step-9 checkpoint with a new 3-class head, 5-fold stratified (grouped) cross-validation, metrics averaged over folds.

| Class | Precision | Recall | F1 | Support (summed over folds) |
|---|---|---|---|---|
| extension | 0.500 | 0.380 | 0.414 | 14 |
| contradiction | 0.699 | 0.698 | 0.648 | 34 |
| other | 0.915 | 0.936 | 0.925 | 252 |
| **macro F1** | | | **0.662** | |

Notes
- **These numbers are not comparable with Section A.** Different labels, different data, much smaller sample.
- Only 14 `extension` examples: its F1 is unreliable. 34 `contradiction` examples is above the 15-example minimum but still small.
- Applied to the corpus (2,034 edges): 91 predicted contradiction and 41 extension. With the confidence > 0.6 rule used when building the graph, 33 contradiction and 15 extension edges end up in the graph.
