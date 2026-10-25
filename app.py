"""Streamlit demo UI: citation-aware literature review assistant.

Run from the repo root with:  python -m streamlit run app.py
"""
from __future__ import annotations

import json
import re

import pandas as pd
import streamlit as st

import config

st.set_page_config(page_title="Citation-Aware Literature Review", layout="wide")

RESULTS = config.RESULTS_DIR
PRESETS = [
    "Session-based recommendation with graph neural networks",  # both reviews cached
    "Graph neural networks for collaborative filtering in recommender systems",
    "Knowledge graph enhanced recommendation",
    "Self-supervised learning for graph-based recommendation",
]
TAG_RE = re.compile(r"\[S2:([^\]]+)\]")
INTENT_CLASSES = ["method", "background", "result"]  # confusion-matrix order


# ── loaders ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_papers() -> pd.DataFrame:
    df = pd.read_parquet(config.DATA_DIR / "papers.parquet")
    df["paper_id"] = df["paper_id"].astype(str)
    return df.set_index("paper_id")


@st.cache_data
def load_json(path: str):
    p = RESULTS / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def load_csv(path: str):
    p = RESULTS / path
    return pd.read_csv(p) if p.exists() else None


def read_text(path: str) -> str | None:
    p = RESULTS / path
    return p.read_text(encoding="utf-8") if p.exists() else None


@st.cache_data(show_spinner=False)
def run_retrieval(query: str, k: int, contradictions: bool):
    from retrieval import retrieve
    from retrieval_baseline import retrieve_baseline, retrieve_bm25

    return (
        retrieve_bm25(query, k=k),
        retrieve_baseline(query, k=k),
        retrieve(query, k=k, surface_contradictions=contradictions),
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def title_of(pid: str) -> str:
    papers = load_papers()
    return str(papers.at[pid, "title"]) if pid in papers.index else pid


def paper_rows(results: list[dict]) -> list[dict]:
    papers = load_papers()
    rows = []
    for r in results:
        pid = str(r["paper_id"])
        row = {
            "#": r["rank"],
            "Title": title_of(pid),
            "Year": int(papers.at[pid, "year"]) if pid in papers.index and pd.notna(papers.at[pid, "year"]) else None,
            "Score": round(float(r["score"]), 3),
            "Flags": ", ".join(map(str, r.get("flags", []))),
        }
        for name, val in (r.get("components") or {}).items():
            row[name] = round(float(val), 3)
        rows.append(row)
    return rows


def resolve_tags(text: str) -> str:
    """Replace [S2:<id>] tags with short bracketed titles."""
    def sub(m: re.Match) -> str:
        return f" *[{title_of(m.group(1))[:50]}]*"
    return TAG_RE.sub(sub, text)


def show_review(rev: dict, with_clusters: bool = False) -> None:
    st.markdown(resolve_tags(rev["text"]))
    n_cited = sum(1 for c in rev["claims"] if c["paper_ids"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sentences", len(rev["claims"]))
    m2.metric("With citation", n_cited)
    m3.metric("Invalid tags", rev["n_invalid_tags"])
    m4.metric("From cache", "yes" if rev["from_cache"] else "no")
    if with_clusters and rev.get("clusters"):
        with st.expander("Paper clusters (Louvain communities)"):
            for cid, pids in rev["clusters"].items():
                st.markdown(f"**Cluster {cid}**")
                for p in pids:
                    st.markdown(f"- {title_of(p)}")
    recs = rev.get("paper_records")
    if recs:
        with st.expander("Evidence trace: what the generator saw for each paper"):
            rows = [
                {
                    "Paper": str(v.get("title", pid))[:70],
                    "Year": v.get("year"),
                    "Intent": v.get("intent"),
                    "Relative to": str(v.get("neighbour") or "")[:50],
                    "Context source": v.get("context_source"),
                    "Context": str(v.get("context") or "")[:220],
                }
                for pid, v in recs.items()
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def confusion_df(d: dict) -> pd.DataFrame:
    labels = d.get("confusion_matrix_labels", INTENT_CLASSES)
    return pd.DataFrame(d["confusion_matrix"], index=[f"true {l}" for l in labels],
                        columns=[f"pred {l}" for l in labels])


def per_class_df(d: dict) -> pd.DataFrame:
    return pd.DataFrame(d["per_class"]).T.round(3)


# ── sidebar ───────────────────────────────────────────────────────────────────

st.title("Citation-Aware Literature Review Assistant")
st.caption("GNNs for recommender systems: SPECTER2 retrieval, intent-typed citation graph re-ranking, cited LLM review")

with st.sidebar:
    st.header("Query")
    preset = st.selectbox("Example queries", ["(custom)"] + PRESETS)
    query = st.text_area("Query", value=PRESETS[0] if preset == "(custom)" else preset, height=100)
    k = st.slider("Top-k papers", 5, 20, 10)
    contradictions = st.checkbox("Surface contradicting papers", value=True)
    use_cache = st.checkbox("Use cached review", value=True)
    gen_model = st.selectbox(
        "Generator model (Groq)",
        ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        help="Use another model if the daily limit of the default is exhausted. "
             "Non-default models cache to results/review_cache_<model> so the reported results stay untouched.",
    )
    go = st.button("Search", type="primary", width="stretch")

if go:
    st.session_state["query"] = query.strip()
    st.session_state.pop("review", None)
    st.session_state.pop("review_base", None)

q = st.session_state.get("query")
(tab_over, tab_ret, tab_rev, tab_int, tab_abl, tab_faith, tab_lim) = st.tabs([
    "Overview", "Live retrieval", "Review: baseline vs citation-aware",
    "Intent classifiers", "Tuning & ablation", "Faithfulness", "Limitations",
])

# ── Overview ──────────────────────────────────────────────────────────────────
with tab_over:
    st.subheader("What this project does")
    st.markdown(
        "A literature-review assistant for **graph neural networks in recommender systems**.\n\n"
        "1. **Retrieve**: SPECTER2 dense embeddings + FAISS give 50 candidate papers.\n"
        "2. **Re-rank**: a citation graph whose edges are typed by *citation intent* "
        "(method / background / result, plus extension and contradiction). Final score = "
        f"{config.ALPHA}·embedding + {config.BETA}·PageRank + {config.GAMMA}·proximity to anchors "
        f"+ {config.DELTA}·co-citation with anchors.\n"
        "3. **Generate**: an LLM (`gpt-oss-120b` on Groq) writes a cited review, grouped by graph community "
        "and ordered by intent.\n"
        "4. **Evaluate**: BM25 and dense baselines, a weight-tuning grid, a test-split ablation, "
        "intent-classifier comparison and a faithfulness check of the generated claims."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers in corpus", "1,500")
    c2.metric("Citation-context edges", "2,034")
    c3.metric("Papers in citation graph", "373")
    c4.metric("Eval queries (val / test)", "3 / 5")
    st.markdown(
        "**Ground truth** comes from survey-paper reference lists (survey_a/b/c and section-level queries). "
        "**Hand labels:** 300 edges (252 other, 34 contradiction, 14 extension) for the extended intent classes."
    )
    st.markdown(
        "**Bug found and fixed:** the SciCite label order is `[method, background, result]`, not "
        "`[background, method, result]`. Model scores were unaffected, but class names and the intent "
        "weights on graph edges were swapped; we fixed it, rebuilt the graph and re-ran tuning."
    )

# ── Live retrieval ────────────────────────────────────────────────────────────
baseline = graph = bm25 = None
if q:
    with st.spinner("Retrieving..."):
        bm25, baseline, graph = run_retrieval(q, k, contradictions)

with tab_ret:
    if not q:
        st.info("Pick a query in the sidebar and press Search.")
    else:
        st.subheader(f"Query: {q}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### BM25 (lexical)")
            st.dataframe(pd.DataFrame(paper_rows(bm25)), hide_index=True, width="stretch")
        with c2:
            st.markdown("#### Dense (SPECTER2 + FAISS)")
            st.dataframe(pd.DataFrame(paper_rows(baseline)), hide_index=True, width="stretch")
        with c3:
            st.markdown("#### Citation-graph re-ranked")
            st.dataframe(pd.DataFrame(paper_rows(graph)), hide_index=True, width="stretch")
        st.caption("Re-ranked columns embedding / pagerank / proximity / cocitation are the min-max "
                   "normalised signals before weighting. Flags mark surfaced contradiction papers.")
        base_ids = {str(r["paper_id"]) for r in baseline}
        new = [title_of(str(r["paper_id"])) for r in graph if str(r["paper_id"]) not in base_ids]
        if new:
            st.info("Papers found only by the graph re-ranker (includes surfaced contradictions):\n\n"
                    + "\n".join(f"- {t}" for t in new))

# ── Reviews ───────────────────────────────────────────────────────────────────
with tab_rev:
    if not q:
        st.info("Pick a query in the sidebar and press Search.")
    else:
        if st.button("Generate both reviews"):
            from generate import generate_review as gen_aware
            from generate_baseline import generate_review as gen_base

            try:
                with st.spinner("Generating reviews (Groq, cached if available)..."):
                    kw = {"k": k, "use_cache": use_cache, "model": gen_model}
                    if gen_model != "openai/gpt-oss-120b":
                        kw["cache_dir"] = RESULTS / f"review_cache_{gen_model.replace('/', '_')}"
                    st.session_state["review_base"] = gen_base(q, baseline, **kw)
                    st.session_state["review"] = gen_aware(q, graph, **kw)
            except Exception as e:  # missing key, network, etc.
                st.error(f"Generation failed: {e}")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Baseline RAG (dense top-k, plain prompt)")
            if st.session_state.get("review_base"):
                show_review(st.session_state["review_base"])
        with right:
            st.markdown("#### Citation-aware (graph re-ranked, clustered, intent-ordered)")
            if st.session_state.get("review"):
                show_review(st.session_state["review"], with_clusters=True)
        st.caption("Only the first preset (session-based) has both reviews cached; any other query calls "
                   "Groq and needs a valid GROQ_API_KEY. Faithfulness of the two systems is compared "
                   "in the Faithfulness tab.")

    traced = read_text("traced_sample_review.md")
    if traced:
        with st.expander("Traced sample review: every claim mapped to its source and intent"):
            st.markdown(traced)

# ── Intent classifiers ────────────────────────────────────────────────────────
with tab_int:
    st.subheader("Citation-intent classification on SciCite test")
    table = read_text("classifier_table.md")
    if table:
        st.markdown(table)
    tf, sb, fs, ex = (load_json(f"{n}_intent.json") for n in ("tfidf", "scibert", "fewshot", "extended"))
    if tf and sb and fs:
        summary = pd.DataFrame({
            "Model": ["TF-IDF + LogReg", "SciBERT (fine-tuned)", "Few-shot gpt-oss-120b (n=300)"],
            "Macro-F1": [tf["macro_f1"], sb["macro_f1"], fs["macro_f1"]],
        }).set_index("Model")
        st.bar_chart(summary)
        cols = st.columns(3)
        for col, name, d in zip(cols, ("TF-IDF", "SciBERT", "Few-shot LLM"), (tf, sb, fs)):
            with col:
                st.markdown(f"**{name}: per class**")
                st.dataframe(per_class_df(d), width="stretch")
                st.markdown("Confusion matrix")
                st.dataframe(confusion_df(d), width="stretch")
    if ex:
        st.subheader("Extended classes: extension / contradiction / other")
        st.caption(ex.get("NOTE", ""))
        st.dataframe(per_class_df(ex), width="stretch")
        st.metric("Macro-F1 (5-fold CV)", f"{ex['macro_f1']:.3f}")

# ── Tuning & ablation ─────────────────────────────────────────────────────────
with tab_abl:
    st.subheader("Weight tuning on the validation split (3 queries)")
    tune = load_csv("tuning_val.csv")
    if tune is not None:
        st.dataframe(tune.head(10).round(4), hide_index=True, width="stretch")
        st.markdown(
            f"**Chosen weights:** alpha={config.ALPHA}, beta={config.BETA}, gamma={config.GAMMA}, "
            f"delta={config.DELTA}. The top grid rows set alpha=0 (no embedding term); we picked a "
            "near-top row that keeps the hybrid. Validation nDCG@10 ≈ 0.44 but test ≈ 0.21, so tuning "
            "on 3 queries overfits."
        )
    st.subheader("Ablation study on the test split (5 queries)")
    abl = load_csv("ablation_table.csv")
    if abl is not None:
        st.dataframe(abl, hide_index=True, width="stretch")
    img = RESULTS / "ablation.png"
    if img.exists():
        st.image(str(img))
    st.markdown(
        "**How to read it**\n"
        "- The graph re-ranker nearly doubles dense nDCG@10 (0.117 → 0.209, paired p≈0.006), but the "
        "confidence intervals overlap (5 queries).\n"
        "- **Typed = untyped edges:** intent only changes edge weights, which only enter PageRank "
        "(weight 0.1). So intent typing is *not shown to help retrieval* here.\n"
        "- **No proximity scores higher (0.273):** proximity was chosen on 3 validation queries; on test it "
        "adds noise in a small, dense graph. A val/test mismatch, not retuned on test.\n"
        "- **Contradiction surfacing changes nothing:** it appends papers after the top-k, which nDCG@10 "
        "cannot see. It is a presentation feature.\n"
        "- **MAP below BM25:** the re-ranker only reorders 50 candidates, so Recall@50 is capped at 0.078 "
        "(BM25: 0.128)."
    )

# ── Faithfulness ──────────────────────────────────────────────────────────────
with tab_faith:
    st.subheader("Faithfulness of generated claims (40 claims, 20 per system)")
    jr = load_json("faithfulness/judge_results.json")
    sheet = load_csv("faithfulness/judging_sheet.csv")
    if jr:
        m = jr["metrics"]
        rows = []
        for key, name in (("baseline", "Baseline RAG"), ("citation_aware", "Citation-aware")):
            rows.append({
                "System": name,
                "LLM judge: strict": f"{m[key]['faithfulness_rate']:.0%}",
                "LLM judge: lenient": f"{m[key]['lenient_rate']:.0%}",
                "Supported / partial / unsupported":
                    f"{m[key]['n_supported']} / {m[key]['n_partial']} / {m[key]['n_unsupported']}",
            })
        if sheet is not None and "label" in sheet.columns:
            sys_of = {c["claim_id"]: c["system"] for c in jr["labelled_claims"]}
            sheet = sheet.assign(system=sheet["claim_id"].map(sys_of))
            for r in rows:
                key = "baseline" if r["System"] == "Baseline RAG" else "citation_aware"
                sub = sheet[sheet["system"] == key]["label"]
                if len(sub):
                    r["Manual labels: strict"] = f"{(sub == 'supported').mean():.0%}"
                    r["Manual labels: lenient"] = f"{sub.isin(['supported', 'partially_supported']).mean():.0%}"
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.warning(
        "The 'manual' labels were written by **Claude (blind to the judge), not by a person**. Cohen's "
        "kappa between those labels and the LLM judge is **0.40** (moderate): a model-vs-model figure, not "
        "human validation. The judge is `gpt-oss-20b`; the reviews were written by `gpt-oss-120b`."
    )
    st.markdown(
        "- **Conclusion: no measurable faithfulness gain for the citation-aware review.** With 20 claims per "
        "system, 10-15 point gaps are within noise.\n"
        "- 12 of 40 claims cite papers with **no abstract** in our data, so they must count as unsupported, "
        "lowering every rate.\n"
        "- Only about a third of retrieved papers have a real citation-context sentence; the rest fall back "
        "to TLDR/abstract, which limits how different the two prompts are."
    )
    if sheet is not None:
        with st.expander("All judged claims"):
            cols = [c for c in ("system", "query", "sentence", "cited_title", "label", "llm_label") if c in sheet.columns]
            st.dataframe(sheet[cols], hide_index=True, width="stretch")

# ── Limitations ───────────────────────────────────────────────────────────────
with tab_lim:
    st.subheader("Limitations and honest caveats")
    st.markdown(
        "- **Tiny, noisy ground truth:** 8 queries (3 val, 5 test) built from survey reference lists; a "
        "relevant paper the survey did not cite counts as wrong. Confidence intervals are wide.\n"
        "- **Small corpus and graph:** 1,500 papers; only 373 appear in the citation graph, so most "
        "candidates have no graph signal.\n"
        "- **Citation-context coverage is partial.**\n"
        "- **Small hand-labelled set:** 14 extension and 34 contradiction examples (extension F1 0.41).\n"
        "- **Classifier transfer:** trained on SciCite, applied to recommender-systems text.\n"
        "- **Intent typing did not help retrieval** in this experiment.\n"
        "- **Faithfulness judged by LLMs** on truncated abstracts, small sample, no human labels.\n"
        "- **Single domain, one generator model**, Groq free tier (responses cached for reproducibility)."
    )
    notes = read_text("LIMITATIONS_AND_NOTES.md")
    if notes:
        with st.expander("Full notes (scoring decisions, interpretation)"):
            st.markdown(notes)
